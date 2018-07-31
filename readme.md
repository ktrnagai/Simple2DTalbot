# 簡易版X線Talbot干渉シミュレーション

Talbot型二次元X線干渉計のシミュレータ。以下の二つのファイルで構成される。
開発言語はPython 3.6その他にnumpy, scipy, matplotlibがあれば実行可能です。

* SimpleMoireSim.py: 二次元Talbot干渉計の簡単なシミュレータ
* FTDemoculation.py: 生成されたモアレ像から被写体の情報を回復するデモデュレータ

## 使い方

単純に

```
SimpleMoireSim.py
```
を実行してモアレを作成し(画像とMoire.npyにデータを保存)
```
DRDemosulation.py
```
を実行して被写体情報の回復(位相回復)を行います(画像のみ)。簡易シミュレータと一般的なフーリエ変換法による位相回復のみです。
