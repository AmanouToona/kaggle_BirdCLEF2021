import numpy as np
import pandas as pd
import soundfile as sf

from pathlib import Path


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

ROOT = Path.cwd()
# ROOT = Path.cwd().parent  # kaggle
INPUT = ROOT / 'input'

df = pd.read_csv(INPUT / 'train_metadata.csv')
df_raw = df.copy()

df['label_id'] = df['primary_label'].map(LABEL_IDS)

df['filepath'] = df.apply(
    lambda x: INPUT / 'train_short_audio' / f'{x["primary_label"]}' / f'{x["filename"]}', axis=1
)

df_raw['audio_len'] = np.nan

for i in df.index:
    audio, origin_sr = sf.read(df.loc[i, 'filepath'], dtype='float32')
    df_raw.loc[i, 'audio_len'] = len(audio)

df_raw.to_csv(INPUT / 'train_metadata_with_audiolen.csv')