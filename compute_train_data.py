import numpy as np
import librosa as lb
import librosa.display as lbd
import soundfile as sf
from  soundfile import SoundFile
import pandas as pd
from  IPython.display import Audio
from pathlib import Path

from matplotlib import pyplot as plt

from tqdm.notebook import tqdm
import joblib, json

from  sklearn.model_selection  import StratifiedKFold

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

SEED = 107

# ================================
TRAIN_AUDIO_DIR = INPUT / 'train_short_audio'
TRAIN_SOUNDSPACE = INPUT / 'train_soundscape_labels.csv'
TRAIN_AUDIO_IMAGES_SAVE_ROOT = INPUT / 'audio_images_5'
# TRAIN_AUDIO_IMAGES_SAVE_ROOT.mkdir(exist_ok=False, parents=True)

SR = 32000  # sampling  rate
DURATION = 5

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
LABEL_IDS = {label: label_id for label_id, label in enumerate(TARGET)}


def get_audio_info(filepath):
    """Get some properties from  an audio file"""
    with SoundFile(filepath) as f:
        sr = f.samplerate
        frames = f.frames  # frames って何？
        duration = float(frames)/sr
    return {"frames": frames, "sr": sr, "duration": duration}


def make_df(nrows=None):
    df = pd.read_csv(INPUT / "train_metadata.csv", nrows=nrows)

    df["label_id"] = df["primary_label"].map(LABEL_IDS)

    df["filepath"] = df.apply(
        lambda x: INPUT / 'train_short_audio' / f'{x["primary_label"]}' / f'{x["filename"]}', axis=1
    )

    pool = joblib.Parallel(4)
    mapper = joblib.delayed(get_audio_info)
    tasks = [mapper(filepath) for filepath in df.filepath]

    df = pd.concat([df, pd.DataFrame(pool(tqdm(tasks)))], axis=1, sort=False)

    return df


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


def crop_or_pad(y, length, is_train=True, start=None):
    if len(y) < length:
        y = np.concatenate([y, np.zeros(length - len(y))])

        n_repeats = length // len(y)
        epsilon = length % len(y)

        y = np.concatenate([y] * n_repeats + [y[:epsilon]])

    elif len(y) > length:
        if not is_train:
            start = start or 0
        else:
            start = start or np.random.randint(len(y) - length)

        y = y[start:start + length]

    return y


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


class AudioToImage:
    def __init__(self, sr=SR, n_mels=128, fmin=0, fmax=None, duration=DURATION, step=None, res_type="kaiser_fast",
                 resample=True, suffix=''):

        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr // 2

        self.duration = duration
        self.audio_length = self.duration * self.sr
        self.step = step or self.audio_length

        self.res_type = res_type
        self.resample = resample

        self.mel_spec_computer = MelSpecComputer(sr=self.sr, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)

        self.suffix = suffix

    def audio_to_image(self, audio):
        melspec = self.mel_spec_computer(audio)
        image = mono_to_color(melspec)
        return image

    def __call__(self, row, save=True):
        audio, orig_sr = sf.read(row.filepath, dtype="float32")

        if self.resample and orig_sr != self.sr:
            audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)

        audios = [audio[i:i + self.audio_length] for i in
                  range(0, max(1, len(audio) - self.audio_length + 1), self.step)]
        audios[-1] = crop_or_pad(audios[-1], length=self.audio_length)
        images = [self.audio_to_image(audio) for audio in audios]
        images = np.stack(images)

        if save:
            path = TRAIN_AUDIO_IMAGES_SAVE_ROOT / f"{row.primary_label}/{row.filename}_{self.suffix}.npy"
            path.parent.mkdir(exist_ok=True, parents=True)
            np.save(str(path), images)
        else:
            return row.filename, images


def get_audios_as_images(df, suffix=None):
    pool = joblib.Parallel(2)

    if suffix is None:
        suffix = DURATION
    converter = AudioToImage(suffix=suffix)
    mapper = joblib.delayed(converter)
    tasks = [mapper(row) for row in df.itertuples(False)]

    pool(tqdm(tasks))


class AudioToImage2:
    def __init__(self, sr=SR, n_mels=128, fmin=0, fmax=None, duration=DURATION, step=None, res_type="kaiser_fast",
                 resample=True, suffix=''):

        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr // 2

        self.duration = duration
        self.audio_length = self.duration * self.sr
        self.step = step or self.audio_length

        self.res_type = res_type
        self.resample = resample

        self.mel_spec_computer = MelSpecComputer(sr=self.sr, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)

        self.suffix = suffix

    def audio_to_image(self, audio):
        melspec = self.mel_spec_computer(audio)
        image = mono_to_color(melspec)
        return image

    def __call__(self, row, save=True):
        audio, orig_sr = sf.read(row.filepath, dtype="float32")

        if self.resample and orig_sr != self.sr:
            audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)

        audios = [audio[i:i + self.audio_length] for i in
                  range(0, max(1, len(audio) - self.audio_length + 1), self.step)]
        audios[-1] = crop_or_pad(audios[-1], length=self.audio_length)
        images = [self.audio_to_image(audio) for audio in audios]
        images = np.stack(images)

        # if save:
        #     path = TRAIN_AUDIO_IMAGES_SAVE_ROOT / f"{row.primary_label}/{row.filename}_{self.suffix}.npy"
        #     path.parent.mkdir(exist_ok=True, parents=True)
        #     np.save(str(path), images)
        # else:
        #     return row.filename, images


def get_audios_as_images2(df, suffix=None):
    pool = joblib.Parallel(2)

    if suffix is None:
        suffix = DURATION
    converter = AudioToImage(suffix=suffix)
    mapper = joblib.delayed(converter)
    tasks = [mapper(row) for row in df.itertuples(False)]

    pool(tqdm(tasks))


df = make_df(nrows=None)

# df.to_csv("rich_train_metadata.csv", index=True)

get_audios_as_images(df)
