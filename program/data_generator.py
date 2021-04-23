import random
import pandas as pd


class Batter:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._2B = random.randint(0, 60)
        self._3B = random.randint(0, 20)
        self._HR = random.randint(0, 60)
        self._H = random.randint(
            self._2B+self._3B+self._HR, 262)
        self._PA = random.randint(self._H, 750)  # 防除 0
        self._BB = random.randint(0, 120)
        self._HBP = random.randint(0, 40)
        self._SF = random.randint(0, 15)
        self._G = random.randint(1, 162)
        self._AB = self._PA  # 不考慮 BB, HBP, SF
        self._R = random.randint(1, 150)
        self._TB = (self._H-self._2B-self._3B-self._HR) * 1 + \
            self._2B * 2 + self._3B * 3 + self._HR * 4
        self._RBI = random.randint(0, 130)
        self._SO = random.randint(0, 200)
        self._SB = random.randint(0, 50)
        self._CS = random.randint(0, 10)
        self._AVG = round(self._H / self._AB, 3)
        self._OBP = round((self._H + self._BB) /
                          (self._AB + self._BB + self._SF), 3)
        self._SLG = round((self._TB / self._AB), 3)
        self._OPS = round((self._OBP + self._SLG), 3)
        self._IBB = random.randint(0, 30)
        self._SAC = random.randint(0, 15)
        self._XBH = self._2B + self._3B + self._HR
        self._GDP = random.randint(0, 30)
        self._GO = random.randint(0, 200)
        self._AO = random.randint(1, 220)  # 防除 0
        self._GOAO = round(self._GO/self._AO, 3)
        self._NP = random.randint(0, 3300)

    def normalize(self, val, minv, maxv):
        return (val - minv) / (maxv - minv)

    def getScore(self):
        _PA = self.normalize(self._PA, 0, 750) * .2
        _BB = self.normalize(self._BB, 0, 120) * .9
        _HBP = self.normalize(self._HBP, 0, 40) * .05
        _SF = self.normalize(self._SF, 0, 15) * .05
        _G = self.normalize(self._G, 1, 162) * .5
        _AB = self.normalize(self._AB, 1, 680) * .1
        _R = self.normalize(self._R, 0, 150) * .5
        _2B = self.normalize(self._2B, 0, 60) * .6
        _3B = self.normalize(self._3B, 0, 20) * .7
        _HR = self.normalize(self._HR, 0, 60) * .8
        _H = self.normalize(self._H, 0, 262) * .9
        _TB = self.normalize(self._TB, 0, 400) * .4
        _RBI = self.normalize(self._RBI, 0, 130) * .8
        _SO = self.normalize(self._SO, 0, 200) * -.9
        _SB = self.normalize(self._SB, 0, 50) * .5
        _CS = self.normalize(self._CS, 0, 10) * -.2
        _AVG = self.normalize(self._AVG, 0, .400) * .9
        _OBP = self.normalize(self._OBP, 0, .450) * .9
        _SLG = self.normalize(self._SLG, 0, .700) * .9
        _OPS = self.normalize(self._OPS, 0, .1150) * 1
        _IBB = self.normalize(self._IBB, 0, 30) * .1
        _SAC = self.normalize(self._SAC, 0, 15) * .05
        _XBH = self.normalize(self._XBH, 0, 130) * .4
        _GDP = self.normalize(self._GDP, 0, 30) * -.2
        _GO = self.normalize(self._GO, 0, 200) * -.2
        _AO = self.normalize(self._AO, 0, 220) * -.2
        _GOAO = self.normalize(self._GOAO, 0, 3) * -.8
        _NP = self.normalize(self._NP, 0, 3300) * .5
        return _PA+_BB+_HBP+_SF+_G+_AB+_R+_2B+_3B+_HR+_H+_TB+_RBI
        +_SO+_SB+_CS+_AVG+_OBP+_SLG+_OPS+_IBB+_SAC+_XBH+_GDP+_GO+_AO+_GOAO+_NP


players = [Batter() for i in range(10000)]
df = pd.DataFrame()

for player in players:
    values = [*player.__dict__.values(), player.getScore()]
    columns = [key[1:] for key in player.__dict__.keys()] + ['score']
    sdf = pd.DataFrame([values], columns=columns)
    df = df.append(sdf, ignore_index=True)

df = df.sort_values(by=['score'], ascending=False).reset_index(drop=True)

df['result'] = 0

# 後 80% 為 no pick
df.loc[(len(df) * .2):(len(df) - 1), 'result'] = 1

# 前 20% 包含 no pick 屬性改為 consider
df.loc[(df.result == 0) & ((df.GOAO > 1.2) | (df.SO > 160) | (
    df.G < 50) | (df.PA < 100)), 'result'] = 2

# 後 80% 包含 pick 屬性改為 consider
df.loc[(df.result == 1) & ((df.H > 160) | (df.AVG > .3) | (
    df.OPS > .75) | (df.BB > 75) | (df.SB > 35)), 'result'] = 2
df = df.sort_values(by=['result'])
df.to_csv('players.csv')
