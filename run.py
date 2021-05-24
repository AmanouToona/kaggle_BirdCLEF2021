# read config file
import argparse
import yaml

import numpy as np
import pandas as pd

# logger
import logging

import cv2
import soundfile as sf


# path
from pathlib import Path

# pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# SWA  Stochastic Weight Averaging
from torch.optim.swa_utils import AveragedModel, SWALR


# augmentation
import albumentations
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
from albumentations.pytorch import ToTensorV2

# sound
import librosa as lb

from sklearn.model_selection import StratifiedKFold

# seed
import random
import os

import gc

from typing import List, Dict, Tuple

import joblib


from scipy.sparse import coo_matrix
import copy

from tqdm import tqdm

# not kaggle environment
from models import *

import sys

# より安全な eval
from ast import literal_eval

DEBUG = False

# Path ===========================
ROOT = Path.cwd()
# ROOT = Path.cwd().parent  # kaggle
INPUT = ROOT / 'input'
OUTPUT = ROOT / 'output'
DATA = INPUT / ''
TRAIN = DATA / 'train'
TEST = DATA / 'test'
LOG = ROOT / 'log'
CONFIG = ROOT / 'config'

# ================================
train_data_dir = INPUT / 'train_short_audio'
train_soundscape = INPUT / 'train_soundscape_labels.csv'

# data ===========================

# logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TARGET = ['acafly', 'acowoo', 'aldfly', 'ameavo', 'amecro', 'amegfi', 'amekes', 'amepip', 'amered', 'amerob', 'amewig', 
          'amtspa', 'andsol1', 'annhum', 'astfly', 'azaspi1', 'babwar', 'baleag', 'balori', 'banana', 'banswa',
          'banwre1', 'barant1', 'barswa', 'batpig1', 'bawswa1', 'bawwar', 'baywre1', 'bbwduc', 'bcnher', 'belkin1',
          'belvir', 'bewwre', 'bkbmag1', 'bkbplo', 'bkbwar', 'bkcchi', 'bkhgro', 'bkmtou1', 'bknsti', 'blbgra1',
          'blbthr1', 'blcjay1', 'blctan1', 'blhpar1', 'blkpho', 'blsspa1', 'blugrb1', 'blujay', 'bncfly', 'bnhcow',
          'bobfly1',
          'bongul', 'botgra', 'brbmot1', 'brbsol1', 'brcvir1', 'brebla', 'brncre', 'brnjay', 'brnthr', 'brratt1',
          'brwhaw', 'brwpar1', 'btbwar', 'btnwar', 'btywar', 'bucmot2', 'buggna', 'bugtan', 'buhvir', 'bulori',
          'burwar1',
          'bushti', 'butsal1', 'buwtea', 'cacgoo1', 'cacwre', 'calqua', 'caltow', 'cangoo', 'canwar', 'carchi',
          'carwre',
          'casfin', 'caskin', 'caster1', 'casvir', 'categr', 'ccbfin', 'cedwax', 'chbant1', 'chbchi', 'chbwre1',
          'chcant2', 'chispa', 'chswar', 'cinfly2', 'clanut', 'clcrob', 'cliswa', 'cobtan1', 'cocwoo1', 'cogdov', 
          'colcha1', 'coltro1', 'comgol', 'comgra', 'comloo', 'commer', 'compau', 'compot1', 'comrav', 'comyel', 
          'coohaw', 'cotfly1', 'cowscj1', 'cregua1', 'creoro1', 'crfpar', 'cubthr', 'daejun', 'dowwoo', 'ducfly',
          'dusfly', 'easblu', 'easkin', 'easmea', 'easpho', 'eastow', 'eawpew', 'eletro', 'eucdov', 'eursta', 'fepowl', 
          'fiespa', 'flrtan1', 'foxspa', 'gadwal', 'gamqua', 'gartro1', 'gbbgul', 'gbwwre1', 'gcrwar', 'gilwoo',
          'gnttow',
          'gnwtea', 'gocfly1', 'gockin', 'gocspa', 'goftyr1', 'gohque1', 'goowoo1', 'grasal1', 'grbani', 'grbher3', 
          'grcfly', 'greegr', 'grekis', 'grepew', 'grethr1', 'gretin1', 'greyel', 'grhcha1', 'grhowl', 'grnher', 
          'grnjay', 'grtgra', 'grycat', 'gryhaw2', 'gwfgoo', 'haiwoo', 'heptan', 'hergul', 'herthr', 'herwar',
          'higmot1',
          'hofwoo1', 'houfin', 'houspa', 'houwre', 'hutvir', 'incdov', 'indbun', 'kebtou1', 'killde', 'labwoo',
          'larspa',
          'laufal1', 'laugul', 'lazbun', 'leafly', 'leasan', 'lesgol', 'lesgre1', 'lesvio1', 'linspa', 'linwoo1',
          'littin1', 'lobdow', 'lobgna5', 'logshr', 'lotduc', 'lotman1', 'lucwar', 'macwar', 'magwar', 'mallar3',
          'marwre', 'mastro1', 'meapar', 'melbla1', 'monoro1', 'mouchi', 'moudov', 'mouela1', 'mouqua', 'mouwar', 
          'mutswa', 'naswar', 'norcar', 'norfli', 'normoc', 'norpar', 'norsho', 'norwat', 'nrwswa', 'nutwoo', 'oaktit',
          'obnthr1', 'ocbfly1', 'oliwoo1', 'olsfly', 'orbeup1', 'orbspa1', 'orcpar', 'orcwar', 'orfpar', 'osprey', 
          'ovenbi1', 'pabspi1', 'paltan1', 'palwar', 'pasfly', 'pavpig2', 'phivir', 'pibgre', 'pilwoo', 'pinsis', 
          'pirfly1', 'plawre1', 'plaxen1', 'plsvir', 'plupig2', 'prowar', 'purfin', 'purgal2', 'putfru1', 'pygnut',
          'rawwre1', 'rcatan1', 'rebnut', 'rebsap', 'rebwoo', 'redcro', 'reevir1', 'rehbar1', 'relpar', 'reshaw', 
          'rethaw', 'rewbla', 'ribgul', 'rinkin1', 'roahaw', 'robgro', 'rocpig', 'rotbec', 'royter1', 'rthhum',
          'rtlhum',
          'ruboro1', 'rubpep1', 'rubrob', 'rubwre1', 'ruckin', 'rucspa1', 'rucwar', 'rucwar1', 'rudpig', 'rudtur', 
          'rufhum', 'rugdov', 'rumfly1', 'runwre1', 'rutjac1', 'saffin', 'sancra', 'sander', 'savspa', 'saypho', 
          'scamac1', 'scatan', 'scbwre1', 'scptyr1', 'scrtan1', 'semplo', 'shicow', 'sibtan2', 'sinwre1', 'sltred', 
          'smbani', 'snogoo', 'sobtyr1', 'socfly1', 'solsan', 'sonspa', 'soulap1', 'sposan', 'spotow', 'spvear1',
          'squcuc1', 'stbori', 'stejay', 'sthant1', 'sthwoo1', 'strcuc1', 'strfly1', 'strsal1', 'stvhum2',
          'subfly', 'sumtan', 'swaspa', 'swathr', 'tenwar', 'thbeup1', 'thbkin', 'thswar1', 'towsol', 'treswa',
          'trogna1', 'trokin', 'tromoc', 'tropar', 'tropew1', 'tuftit', 'tunswa', 'veery', 'verdin', 'vigswa',
          'warvir', 'wbwwre1', 'webwoo1', 'wegspa1', 'wesant1', 'wesblu', 'weskin', 'wesmea', 'westan', 'wewpew',
          'whbman1', 'whbnut', 'whcpar', 'whcsee1', 'whcspa', 'whevir', 'whfpar1', 'whimbr', 'whiwre1', 'whtdov',
          'whtspa', 'whwbec1', 'whwdov', 'wilfly', 'willet1', 'wilsni1', 'wiltur', 'wlswar', 'wooduc', 'woothr',
          'wrenti', 'y00475', 'yebcha', 'yebela1', 'yebfly', 'yebori1', 'yebsap', 'yebsee1', 'yefgra1', 'yegvir',
          'yehbla', 'yehcar1', 'yelgro', 'yelwar', 'yeofly1', 'yerwar', 'yeteup1', 'yetvir']
N_CLASSES = len(TARGET)
DURATION = 5
SR = 32000  # sampling  rate


# utils ------------------------------------------------------------------------------------------------------------
def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = deterministic


# DataSet ---------------------------------------------------------------------------------------------------------
LABEL_IDS = {label: label_id for label_id, label in enumerate(TARGET)}


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V


def crop_or_pad(y, length):
    if len(y) < length:
        y = np.concatenate([y, length - np.zeros(len(y))])
    elif len(y) > length:
        y = y[:length]
    return y


class MelSpecComputer:
    def __init__(self, sr, n_mels, fmin, fmax, **kwargs):
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        kwargs["n_fft"] = kwargs.get("n_fft", self.sr//10)
        kwargs["hop_length"] = kwargs.get("hop_length", self.sr//(10*4))
        self.kwargs = kwargs

    def __call__(self, y):

        melspec = lb.feature.melspectrogram(
            y, sr=self.sr, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, **self.kwargs,
        )

        melspec = lb.power_to_db(melspec).astype(np.float32)
        return melspec


class BirdCLEFDataset(Dataset):
    def __init__(self, data, sr=SR, n_mels=128, fmin=0, fmax=None, duration=DURATION, step=None, res_type="kaiser_fast",
                 resample=True, label_smoothing=True):

        self.data = data

        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr // 2

        self.duration = duration
        self.audio_length = self.duration * self.sr
        self.step = step or self.audio_length

        self.res_type = res_type
        self.resample = resample
        self.label_smoothing = label_smoothing

        self.mel_spec_computer = MelSpecComputer(sr=self.sr, n_mels=self.n_mels, fmin=self.fmin,
                                                 fmax=self.fmax)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def normalize(image):
        image = image.astype("float32", copy=False) / 255.0
        image = np.stack([image, image, image])
        return image

    def audio_to_image(self, audio):
        melspec = self.mel_spec_computer(audio)
        image = mono_to_color(melspec)
        image = self.normalize(image)
        return image

    def read_file(self, filepath):
        audio, orig_sr = sf.read(filepath, dtype="float32")

        if self.resample and orig_sr != self.sr:
            audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)

        audios = []
        for i in range(self.audio_length, len(audio) + self.step, self.step):
            start = max(0, i - self.audio_length)
            end = start + self.audio_length
            audios.append(audio[start:end])

        if len(audios[-1]) < self.audio_length:
            audios = audios[:-1]

        images = [self.audio_to_image(audio) for audio in audios]
        images = np.stack(images)

        return images

    def __getitem__(self, idx):
        img = self.read_file(self.data.loc[idx, 'filepath'])

        label = np.zeros(N_CLASSES, dtype=np.float32)
        if self.label_smoothing:
            label += 0.0025
        label[LABEL_IDS[self.data.loc[idx, 'primary_label']]] = 0.995

        return img, label


def load_audio_image(df, max_read_samples):
    def load_row(row):
        return row.filename, np.load(str(row.filepath))[:max_read_samples]
    pool = joblib.Parallel(4)
    mapper = joblib.delayed(load_row)
    tasks = [mapper(row) for row in df.itertuples(False)]
    res = pool(tqdm(tasks))
    res = dict(res)
    return res


class BirdClefDatasetnp(Dataset):
    def __init__(self, data, load_idx,  max_read_samples=5, sr=SR, is_train=True, duration=DURATION, distort=False):
        audio_image_store = load_audio_image(data.iloc[load_idx, :], max_read_samples)
        self.audio_image_store = audio_image_store
        self.meta = data.iloc[load_idx, :].copy().reset_index(drop=True)
        self.sr = sr
        self.is_train = is_train
        self.duration = duration
        self.audio_length = self.duration * self.sr
        self.distort = distort

    @staticmethod
    def normalize(image):
        image = image.astype("float32", copy=False) / 255.0
        image = np.stack([image, image, image])
        return image

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        image = self.audio_image_store[row.filename]

        if self.distort:
            prob = [1] * len(image)
            prob[0] = 2
            prob = [i / sum(prob) for i in prob]
            image = image[np.random.choice(a=len(image), size=1, p=prob)]
        else:
            image = image[np.random.choice(len(image))]
        image = self.normalize(image)

        t = np.zeros(N_CLASSES, dtype=np.float32) + 0.0025  # Label smoothing
        t[LABEL_IDS[self.meta.loc[idx, 'primary_label']]] = 0.995

        return image, t


class Normalize:
    def __call__(self, y: np.ndarray):
        max_vol = np.abs(y).max()
        y_vol = y * 1 / max_vol
        return np.asfortranarray(y_vol)


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


# scheduler --------------------------------------------------------------------------------------------------


# validation -----------------------------------------------------------------------------------------------------------
def stratified_k_folds(label_arr: np.array, n_fold: int, size: float = 1):
    split = int(n_fold / size)
    skf = StratifiedKFold(n_splits=split)
    for i, (train_idx, val_idx) in enumerate(skf.split(np.ones_like(label_arr), label_arr)):
        yield train_idx, val_idx
        if i + 1 == n_fold:
            break


# train ----------------------------------------------------------------------------------------------------------------
def train_one_fold(config, train_all):
    torch.backends.cudnn.benchmark = True
    device = torch.device(config['globals']['device'])
    set_seed(config['globals']['seed'])

    swa = False
    if 'swa' in config.keys():
        swa = True

    valid_idx = config['globals']['valid_idx']
    train_idx = config['globals']['train_idx']

    # valid_dataset = BirdCLEFDataset(train_all.loc[valid_idx, :], **config['dataset']['train'])
    # train_dataset = BirdCLEFDataset(train_all.loc[train_idx, :], **config['dataset']['valid'])

    valid_dataset = BirdClefDatasetnp(train_all, valid_idx, **config['dataset']['train'])
    train_dataset = BirdClefDatasetnp(train_all, train_idx, **config['dataset']['valid'])

    logger.debug(f'valid_dataset: {len(valid_dataset)}')
    logger.debug(f'train_dataset: {len(train_dataset)}')

    # DataLoader
    train_loader = DataLoader(train_dataset, **config['loader']['train'])
    valid_loader = DataLoader(valid_dataset, **config['loader']['valid'])

    # model
    model = eval(config['model']['name'])(**config['model']['params'])
    model.to(device)

    # optimizer
    optimizer = getattr(torch.optim, config['optimizer']['name'])(model.parameters(), **config['optimizer']['params'])

    swa_model = None
    swa_scheduler = None
    if swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer=optimizer, swa_lr=0.05)

    # scheduler
    if config['scheduler']['name'] == 'OneCycleLR':
        config['scheduler']['params']['epochs'] = config['globals']['max_epoch']
        config['scheduler']['params']['step_per_epoch'] = len(train_loader)
    elif config['scheduler']['name'] == 'LinearIncrease':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 10 ** (0.4 * epoch - 5))
    else:
        scheduler = getattr(
            torch.optim.lr_scheduler, config['scheduler']['name'])(optimizer, **config['scheduler']['params'])

    # loss
    if hasattr(nn, config['loss']['name']):
        loss_func = getattr(nn, config['loss']['name'])(**config['loss']['params'])
    else:
        loss_func = eval(config['loss']['name'])(**config['loss']['params'])
    loss_func.to(device)

    # Early stopping
    early_stop = EarlyStopping(**config['early_stopping']['params'])

    # Train Loop
    train_losses = []
    valid_losses = []
    iteration = 0
    accumulation = 1
    if 'accumulation' in config:
        accumulation = config['accumulation']

    for epoch in range(config['globals']['max_epoch']):
        logger.info(f'epoch {epoch + 1} / {config["globals"]["max_epoch"]}')

        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for step, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.float().to(device), labels.float().to(device)
            iteration += 1
            # ToDo 経過時間を記録する

            y = model(images)
            loss = loss_func(y, labels)

            loss /= accumulation
            loss.backward()
            running_loss += float(loss.detach()) * accumulation

            images.detach()
            labels.detach()
            del images
            del labels
            del loss  # 計算グラフの削除によるメモリ節約

            if (step + 1) % accumulation == 0 or (step + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

                torch.cuda.empty_cache()
            _ = gc.collect()
        train_losses.append(running_loss / len(train_loader))
        logger.info(f'lr: {optimizer.param_groups[0]["lr"]}')
        logger.info(f'train loss: {train_losses[-1]:.8f}')
        logger.info(f'iteration: {iteration}')

        # evaluation
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.float().to(device), labels.float().to(device)

                y = model(images)

                loss = loss_func(y, labels).detach()
                running_loss += float(loss)

            valid_losses.append(running_loss / len(valid_loader))
            logger.info(f'valid loss: {valid_losses[-1]:.8f}')

        if config['scheduler']['name'] == 'ReduceLROnPlateau':
            scheduler.step(loss)
        else:
            scheduler.step()

        del loss

        if swa and epoch > config['swa']['swa_start']:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # save model
        # torch.save(model.state_dict(), f'models_trained/{config["globals"]["name"]}_epoch{epoch + 1}.pth')
        # ToDo パラメータを逐次保存しもっともよいパラメータを呼び出すように変更する
        # ToDo 保存したloss がオブジェクトになっているので改善する

        # early stopping
        if early_stop.step(valid_losses[-1]):
            break

        _ = gc.collect()

    if swa:
        torch.optim.swa_utils.update_bn(train_loader, swa_model)

    torch.save(model.state_dict(), f'model_trained/{config["globals"]["name"]}.pth')

    epochs = [i + 1 for i in range(len(train_losses))]
    eval_df = pd.DataFrame(index=epochs, columns=['train_eval', 'valid_eval'])
    eval_df['train_eval'] = train_losses
    eval_df['valid_eval'] = valid_losses
    eval_df.to_csv(f'output/{config["globals"]["name"]}_eval.csv')


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False, on=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)
        self.on = on

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        # if torch.isnan(metrics):
        #     return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)


# main ---------------------------------------------------------------------------------------------------------------
def main(kaggle=False):

    if kaggle:
        config = """"""
        config = yaml.safe_load(config)
        config['globals']['name'] = 'kaggle_kernel'
    else:
        parser = argparse.ArgumentParser(description='pytorch runner')
        parser.add_argument('-conf', '--config_file', help='runtime configuration file')
        args = parser.parse_args()
        if args.config_file is None:
            args.config_file = 'trial.yaml'

        print(args.config_file)
        config_file = CONFIG / args.config_file

        with open(str(config_file), 'r') as f:
            config = yaml.safe_load(f)

        config['globals']['name'] = config_file.stem

    # config ファイルから基本情報の読み取り
    folds = config['globals']['folds']
    n_fold = len(folds)

    global DEBUG
    if config['globals']['debug']:
        DEBUG = True

    if DEBUG:
        config['globals']['max_epoch'] = 10
        print('!' * 10, 'debug mode', '!' * 10)

    # config file からの設定
    set_seed(config['globals']['seed'])

    # log ---------------------------------------------
    handler_st = logging.StreamHandler()
    handler_st.setLevel(logging.DEBUG)
    handler_st_format = logging.Formatter('%(asctime)s %(name)s: %(message)s')
    handler_st.setFormatter(handler_st_format)
    logger.addHandler(handler_st)

    # log ファイル
    if not kaggle:
        log_file = LOG / f'log_{config["globals"]["name"]}.log'
        handler_f = logging.FileHandler(log_file, 'a')
        handler_f.setLevel(logging.DEBUG)
        handler_f_format = logging.Formatter('%(asctime)s %(name)s: %(message)s')
        handler_f.setFormatter(handler_f_format)
        logger.addHandler(handler_f)

    # ----------------------------------------------------------------------------------------------------------------
    meta_data = pd.read_csv(INPUT / 'train_metadata.csv')
    meta_data['secondary_labels'] = meta_data['secondary_labels'].apply(literal_eval)  # 文字列を List に変換する

    if DEBUG:
        meta_data = meta_data.sample(frac=0.1)

    # file path の追加
    # meta_data['filepath'] = meta_data.apply(
    #     lambda x: INPUT / 'train_short_audio' / f'{x["primary_label"]}' / f'{x["filename"]}_5.npy', axis=1
    # )

    meta_data['filepath'] = meta_data.apply(
        lambda x: INPUT / 'audio_images_5' / f'{x["primary_label"]}' / f'{x["filename"]}_5.npy', axis=1
    )

    # 全てのラベルが含まれているか確認する
    if set(meta_data['primary_label'].values) == set(TARGET):
        print('all targets')
    else:
        print('missing target')

    # make fold data
    if 'fold_size' in config['globals']:
        size = config['globals']['fold_size']
    else:
        size = 1
    train_valid_idx = list(stratified_k_folds(label_arr=meta_data['primary_label'].values, n_fold=n_fold, size=size))
    torch.cuda.empty_cache()
    _ = gc.collect()

    for fold_id, train_valid_index in enumerate(train_valid_idx):
        logger.info(f'start train fold: {fold_id:02}')
        fold_config = copy.deepcopy(config)
        fold_config['globals']['valid_idx'] = train_valid_index[1]
        fold_config['globals']['train_idx'] = train_valid_index[0]
        fold_config['globals']['name'] += f'_{fold_id:02}'

        train_one_fold(config=fold_config, train_all=meta_data)
        torch.cuda.empty_cache()
        _ = gc.collect()

        if DEBUG:
            break


if __name__ == '__main__':
    main()
