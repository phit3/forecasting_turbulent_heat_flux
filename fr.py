import logging
import os
import typing

import numpy as np
from typing import List, Tuple

from base import Base
import random
import torch
import torch.nn as nn
import torch.optim as optim
from time import time

from generators import Seq2SeqGenerators


class FR(Base):
    @property
    def encoder_model(self):
        return self.__encoder_model

    @encoder_model.setter
    def encoder_model(self, encoder_model):
        self.__encoder_model = encoder_model

    @property
    def decoder_model(self):
        return self.__decoder_model

    @decoder_model.setter
    def decoder_model(self, decoder_model):
        self.__decoder_model = decoder_model

    @property
    def encoder_optimizer(self):
        return self.__encoder_optimizer

    @encoder_optimizer.setter
    def encoder_optimizer(self, encoder_optimizer):
        self.__encoder_optimizer = encoder_optimizer

    @property
    def decoder_optimizer(self):
        return self.__decoder_optimizer

    @decoder_optimizer.setter
    def decoder_optimizer(self, decoder_optimizer):
        self.__decoder_optimizer = decoder_optimizer

    @property
    def encoder_lr_scheduler(self):
        return self.__encoder_lr_scheduler

    @encoder_lr_scheduler.setter
    def encoder_lr_scheduler(self, encoder_lr_scheduler):
        self.__encoder_lr_scheduler = encoder_lr_scheduler

    @property
    def decoder_lr_scheduler(self):
        return self.__decoder_lr_scheduler

    @decoder_lr_scheduler.setter
    def decoder_lr_scheduler(self, decoder_lr_scheduler):
        self.__decoder_lr_scheduler = decoder_lr_scheduler

    @property
    def current_learning_rate(self):
        return self.decoder_optimizer.param_groups[0]['lr']

    @property
    def number_of_parameters(self):
        return sum(p.numel() for p in self.encoder_model.parameters() if p.requires_grad) + sum(p.numel() for p in self.decoder_model.parameters() if p.requires_grad)

    class Encoder(torch.jit.ScriptModule):
        def __init__(self, input_steps: int, batch_size: int, dimensions: int, latent_dim: int):
            super(FR.Encoder, self).__init__()
            self.input_steps = input_steps
            self.latent_dim = latent_dim
            self.batch_size = batch_size
            self.dimensions = dimensions
            self.encoder_gru = nn.GRU(self.dimensions, self.latent_dim)

        @torch.jit.script_method
        def forward(self, inputs: torch.Tensor, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.encoder_gru(inputs.view(self.input_steps,
                                                self.batch_size,
                                                self.dimensions), states)

    class Decoder(torch.jit.ScriptModule):
        def __init__(self, output_steps: int, batch_size: int, dimensions: int, latent_dim: int):
            super(FR.Decoder, self).__init__()
            self.output_steps = output_steps
            self.latent_dim = latent_dim
            self.batch_size = batch_size
            self.dimensions = dimensions
            self.decoder_gru = nn.GRU(self.dimensions, self.latent_dim)
            self.linear = nn.Linear(self.latent_dim, self.dimensions)

        @torch.jit.script_method
        def forward(self, next_input: torch.Tensor, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            predictions = []
            for j in range(self.output_steps):
                decoder_gru_out, states = self.decoder_gru(next_input.view(1, self.batch_size, -1), states)
                decoder_output = self.linear(decoder_gru_out)

                predictions.append(decoder_output)
                next_input = decoder_output
            return torch.cat(predictions, dim=0).view(self.output_steps, self.batch_size, self.dimensions), states

    def __init__(self, hyper_parameters: typing.Dict[str, typing.Any], force_new: bool = False):
        super(FR, self).__init__(hyper_parameters, force_new)

    def _build_checkpoint(self, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        for f in os.listdir(checkpoint_dir):
            f_path = os.path.join(checkpoint_dir, f)
            if os.path.isfile(os.path.join(f_path, 'encoder')) or os.path.isfile(os.path.join(f_path, 'decoder')):
                if self.force_new:
                    os.remove(f_path)
                else:
                    raise FileExistsError('The checkpoint {} already exists. Please delete the checkpoint file if '
                                          'you want to retrain the network.'.format(checkpoint_dir))
        return checkpoint_dir

    def _build_model(self):
        logging.info('Building model')
        self.encoder_model = self.Encoder(self.input_steps, self.batch_size, self.dimensions, self.latent_dim)
        self.encoder_optimizer = optim.Adam(self.encoder_model.parameters(), lr=self.learning_rate)
        self.encoder_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.encoder_optimizer, mode='min',
                                                                         factor=self.gamma, patience=self.plateau,
                                                                         min_lr=3e-6)
        self.decoder_model = self.Decoder(self.output_steps, self.batch_size, self.dimensions, self.latent_dim)
        self.decoder_optimizer = optim.Adam(self.decoder_model.parameters(), lr=self.learning_rate)
        self.decoder_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, mode='min',
                                                                         factor=self.gamma, patience=self.plateau,
                                                                         min_lr=3e-6)

    def generate_generators(self, datasets_dir: str, dataset: str, split: tuple, max_samples: int = 10000, **kwargs):
        logging.info('Generating generators')
        generators = Seq2SeqGenerators(datasets_dir=datasets_dir, dataset=dataset,
                                       input_steps=self.input_steps,
                                       output_steps=self.output_steps,
                                       batch_size=self.batch_size,
                                       split=split,
                                       max_samples=max_samples)
        return generators

    def _do_batch(self, epoch, batch_number, batch, max_batch_number):
        self.print_progress(epoch, batch_number, max_batch_number)
        (encoder_inputs, decoder_inputs), targets = batch
        encoder_inputs = np.swapaxes(encoder_inputs, 0, 1)
        initial_states = torch.randn(1, self.batch_size, self.latent_dim)
        _, states = self.encoder_model(torch.tensor(encoder_inputs).view(self.input_steps,
                                                                         self.batch_size, -1), initial_states)
        targets = torch.tensor(np.swapaxes(targets, 0, 1)).view(self.output_steps, self.batch_size, -1)
        first_input = torch.tensor(encoder_inputs[-1]).view(1, self.batch_size, -1)
        predictions, states = self.decoder_model(first_input, states)
        loss = 0
        for p, t in zip(predictions, targets):
            loss += self.loss_fct(p, t)
        return predictions, targets, loss

    def _do_epoch(self):
        epoch = self.epoch + 1
        train_steps = int(self.generators.train_samples // self.generators.batch_size)
        valid_steps = int(self.generators.valid_samples // self.generators.batch_size)

        # training phase
        batch_losses = []
        batch_number = 1
        for batch in self.generators.generate_train():
            if batch_number > train_steps:
                break
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            predictions, targets, loss = self._do_batch(epoch, batch_number, batch, train_steps + valid_steps)
            loss.backward()
            self.decoder_optimizer.step()
            self.encoder_optimizer.step()
            batch_losses.append(loss.item() / self.output_steps)
            batch_number += 1

        # validation phase
        batch_val_losses = []
        batch_number = 1
        with torch.no_grad():
            for batch in self.generators.generate_valid():
                if batch_number > valid_steps:
                    break
                predictions, targets, val_loss = self._do_batch(epoch, train_steps + batch_number, batch, train_steps + valid_steps)
                batch_val_losses.append(val_loss.item() / self.output_steps)
                batch_number += 1

        # calculate means
        self.epoch_loss = np.mean(batch_losses)
        self.epoch_val_loss = np.mean(batch_val_losses)
        return self.next_learning_rate()

    def fit_generator(self, generators: Seq2SeqGenerators, checkpoint_dir: str, history_dir: str):
        super(FR, self).fit_generator(generators, checkpoint_dir, history_dir)
        all_val_losses = []
        for epoch in range(self.epochs):
            self.epoch = epoch
            epoch_start = time()
            learning_rate = self._do_epoch()
            # only save weights for improved val_losses
            if len(all_val_losses) == 0 or min(all_val_losses) > self.epoch_val_loss:
                self.save_weights(checkpoint_dir)
            all_val_losses.append(self.epoch_val_loss)
            epoch_values = {'epoch': str(epoch), 'loss': str(self.epoch_loss), 'val_loss': str(self.epoch_val_loss), 'learning_rate': str(learning_rate)}
            self.write_history(history_dir=history_dir, epoch_values=epoch_values)

            print(' - loss: {loss:3.6} - val_loss: {val_loss:3.6} - duration: {duration:05} - lr: {lr:.6}'.format(
                    loss=self.epoch_loss, val_loss=self.epoch_val_loss, duration=time() - epoch_start, lr=learning_rate))
            if self.early_stopping():
                break
        print(self.__class__.__name__, '[best epoch:', self.get_best_epoch(), 'loss', self.loss_of_epoch(self.get_best_epoch()), 'val loss:', self.best_val_loss)

    def save_weights(self, weights_file_path: str):
        encoder_weight_file_path = os.path.join(weights_file_path, 'encoder')
        decoder_weight_file_path = os.path.join(weights_file_path, 'decoder')
        torch.save(self.encoder_model.state_dict(), encoder_weight_file_path)
        torch.save(self.decoder_model.state_dict(), decoder_weight_file_path)

    def load_weights(self, weights_file_path: str):
        if torch.cuda.is_available():
            map_location = torch.device('cuda:0')
        else:
            map_location = torch.device('cpu')
        encoder_weight_file_path = os.path.join(weights_file_path, 'encoder')
        decoder_weight_file_path = os.path.join(weights_file_path, 'decoder')
        self.encoder_model.load_state_dict(torch.load(encoder_weight_file_path, map_location=map_location))
        self.decoder_model.load_state_dict(torch.load(decoder_weight_file_path, map_location=map_location))

    def next_learning_rate(self):
        self.decoder_lr_scheduler.step(self.epoch_loss)
        self.encoder_lr_scheduler.step(self.epoch_loss)
        return self.current_learning_rate

    def predict_generator(self, generators: Seq2SeqGenerators):
        super(FR, self).predict_generator(generators)
        test_steps = int(self.generators.test_samples // self.generators.batch_size)
        with torch.no_grad():
            s = 0
            all_targets = []
            all_predictions = []
            for batch in generators.generate_test():
                if s >= test_steps:
                    break
                (encoder_inputs, _), targets = batch
                encoder_inputs = np.swapaxes(encoder_inputs, 0, 1)
                initial_states = torch.randn(1, self.batch_size, self.latent_dim)
                _, states = self.encoder_model(torch.tensor(encoder_inputs).view(self.input_steps, self.batch_size, -1), initial_states)
                targets = torch.tensor(np.swapaxes(targets, 0, 1)).view(self.output_steps, self.batch_size, -1)
                inputs = torch.tensor(encoder_inputs[-1]).view(1, self.batch_size, -1)
                predictions, states = self.decoder_model(inputs, states)
                all_targets = np.concatenate([all_targets, targets.cpu()], axis=1) if len(all_targets) > 0 else targets.cpu()
                all_predictions = np.concatenate([all_predictions, predictions.cpu()], axis=1) if len(all_predictions) > 0 else predictions.cpu()
                s += 1
        return np.array(all_predictions), np.array(all_targets)

    def predict_sample(self, sample: torch.Tensor):
        with torch.no_grad():
            initial_states = torch.randn(1, self.batch_size, self.latent_dim)
            _, states = self.encoder_model(sample.view(self.input_steps, self.batch_size, self.dimensions).clone().detach(), initial_states)
            dec_input = sample[-1].view(1, self.batch_size, self.dimensions)
            prediction , _ = self.decoder_model(dec_input, states)
        return prediction.view(self.output_steps, self.dimensions)
