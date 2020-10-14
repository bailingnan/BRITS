# coding: utf-8

import os
import re
import numpy as np
import pandas as pd

import ujson as json

patient_ids = []

for filename in os.listdir("./raw"):
    # the patient data in PhysioNet contains 6-digitsPhysioNet中的患者数据包含6位数字
    match = re.search(
        "\d{6}", filename
    )  # re.match 尝试从字符串的起始位置匹配一个模式，如果不是起始位置匹配成功的话，match()就返回none。
    if match:
        id_ = match.group()  # 使用group(num)匹配对象函数来获取匹配表达式。str:'138781'
        patient_ids.append(id_)

out = pd.read_csv("./raw/Outcomes-a.txt").set_index("RecordID")[
    "In-hospital_death"
]  # 'RecordID`作为索引，是否死亡作为输出

# we select 35 attributes which contains enough non-values
attributes = [
    "DiasABP",
    "HR",
    "Na",
    "Lactate",
    "NIDiasABP",
    "PaO2",
    "WBC",
    "pH",
    "Albumin",
    "ALT",
    "Glucose",
    "SaO2",
    "Temp",
    "AST",
    "Bilirubin",
    "HCO3",
    "BUN",
    "RespRate",
    "Mg",
    "HCT",
    "SysABP",
    "FiO2",
    "K",
    "GCS",
    "Cholesterol",
    "NISysABP",
    "TroponinT",
    "MAP",
    "TroponinI",
    "PaCO2",
    "Platelets",
    "Urine",
    "NIMAP",
    "Creatinine",
    "ALP",
]

# mean and std of 35 attributes
mean = np.array(
    [
        59.540976152469405,
        86.72320413227443,
        139.06972964987443,
        2.8797765291788986,
        58.13833409690321,
        147.4835678885565,
        12.670222585415166,
        7.490957887101613,
        2.922874149659863,
        394.8899400819931,
        141.4867570064675,
        96.66380228136883,
        37.07362841054398,
        505.5576196473552,
        2.906465787821709,
        23.118951553526724,
        27.413004968675743,
        19.64795551193981,
        2.0277491155660416,
        30.692432164676188,
        119.60137167841977,
        0.5404785381886381,
        4.135790642787733,
        11.407767149315339,
        156.51746031746032,
        119.15012244292181,
        1.2004983498349853,
        80.20321011673151,
        7.127188940092161,
        40.39875518672199,
        191.05877024038804,
        116.1171573535279,
        77.08923183026529,
        1.5052390166989214,
        116.77122488658458,
    ]
)

std = np.array(
    [
        13.01436781437145,
        17.789923096504985,
        5.185595006246348,
        2.5287518090506755,
        15.06074282896952,
        85.96290370390257,
        7.649058756791069,
        8.384743923130074,
        0.6515057685658769,
        1201.033856726966,
        67.62249645388543,
        3.294112002091972,
        1.5604879744921516,
        1515.362517984297,
        5.902070316876287,
        4.707600932877377,
        23.403743427107095,
        5.50914416318306,
        0.4220051299992514,
        5.002058959758486,
        23.730556355204214,
        0.18634432509312762,
        0.706337033602292,
        3.967579823394297,
        45.99491531484596,
        21.97610723063014,
        2.716532297586456,
        16.232515568438338,
        9.754483687298688,
        9.062327978713556,
        106.50939503021543,
        170.65318497610315,
        14.856134327604906,
        1.6369529387005546,
        133.96778334724377,
    ]
)

fs = open("./json/jsons", "w")


def to_time_bin(x):
    h, m = map(int, x.split(":"))  # 03:37返回为3，37
    return h  # 返回3


def parse_data(x):  # 传入dataframe,返回列表，每个小时在35个指标上的值
    x = x.set_index("Parameter").to_dict()[
        "Value"
    ]  # {'RecordID': 138781.0, 'Age': 90.0, 'Gender': 0.0, 'Height': -1.0, 'ICUType': 3.0, 'Weight': 52.9}

    values = []

    for attr in attributes:
        if attr in x:
            values.append(x[attr])
        else:
            values.append(np.nan)
    return values


def parse_delta(masks, dir_):  # delta为最后一次观测时间到当前时间st的时间间隔，
    if dir_ == "backward":
        masks = masks[::-1]  # [[1,2],[3,4]]反转为[[3,4],[1,2]]

    deltas = []

    for h in range(48):
        if h == 0:
            deltas.append(np.ones(35))
        else:
            deltas.append(np.ones(35) + (1 - masks[h]) * deltas[-1])  # masks空值为false，0

    return np.array(deltas)


def parse_rec(values, masks, evals, eval_masks, dir_):
    deltas = parse_delta(masks, dir_)

    # only used in GRU-D
    forwards = (
        pd.DataFrame(values).fillna(method="ffill").fillna(0.0).values
    )  # 用t-1的值填补t的值

    rec = {}

    rec["values"] = np.nan_to_num(values).tolist()
    rec["masks"] = masks.astype("int32").tolist()
    # imputation ground-truth
    rec["evals"] = np.nan_to_num(evals).tolist()
    rec["eval_masks"] = eval_masks.astype("int32").tolist()
    rec["forwards"] = forwards.tolist()
    rec["deltas"] = deltas.tolist()

    return rec


def parse_id(id_):
    data = pd.read_csv("./raw/{}.txt".format(id_))
    # accumulate the records within one hour一小时内累积记录
    data["Time"] = data["Time"].apply(lambda x: to_time_bin(x))  # time列转化为小时

    evals = []

    # merge all the metrics within one hour一小时内合并所有指标
    for h in range(48):
        evals.append(parse_data(data[data["Time"] == h]))
    # 大列表套小列表，48个小时，每个小时35个参数，48*35
    evals = (np.array(evals) - mean) / std  # 归一化

    shp = evals.shape  # (48，35)

    evals = evals.reshape(-1)  # (1680,)，摊平

    # randomly eliminate 10% values as the imputation ground-truth随机消除10％的值作为插补基础
    indices = np.where(~np.isnan(evals))[0].tolist()  # 输出不为nan的值的坐标的列表
    indices = np.random.choice(indices, len(indices) // 10)  # 随机抽len/10个数字

    values = evals.copy()  # 复制一份原始数据
    values[indices] = np.nan  # 被选中的坐标的数字在values为np.nan

    masks = ~np.isnan(values)  # 空值为false
    eval_masks = (~np.isnan(values)) ^ (
        ~np.isnan(evals)
    )  # evals和values中的值相异时才为true,即被掩盖的值的坐标才为true

    evals = evals.reshape(shp)  # (48,35),原始数据
    values = values.reshape(shp)  # 10%概率丢弃后的训练数据

    masks = masks.reshape(shp)  # 原始数据中空值为false
    eval_masks = eval_masks.reshape(shp)  # 训练数据中被掩盖的值为true

    label = int(out.loc[int(id_)])

    rec = {"label": label}  # In-hospital_death 表示是否死亡

    # prepare the model for both directions为两个方向准备模型
    # rec为一个字典，key为label,forward,backward，forward和backward分别为字典，有values,masks,evals,eval_masks,
    # forwards(用t-1的值填补t,用0填补),deltas
    rec["forward"] = parse_rec(values, masks, evals, eval_masks, dir_="forward")
    rec["backward"] = parse_rec(
        values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_="backward"
    )

    rec = json.dumps(rec)

    fs.write(rec + "\n")


for id_ in patient_ids:
    print("Processing patient {}".format(id_))
    try:
        parse_id(id_)
    except Exception as e:
        print(e)
        continue

fs.close()
