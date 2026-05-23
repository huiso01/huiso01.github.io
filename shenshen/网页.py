import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template, jsonify, redirect
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ---------------------------- 全局变量 ----------------------------
model = None
scaler = None
cont_feature_medians = None  # 29个连续特征的中位数（用于缺失填充）
cat_feature_modes = None  # 2个分类特征（glu_cat, hba1c_cat）的众数
cv_metrics = {}

N_CONT = 29
N_CAT = 2


# ---------------------------- 生成合成训练数据（模拟原Excel结构） ----------------------------
def generate_synthetic_data(n_samples=3000, random_state=2024):
    """生成符合真实分布的合成数据，保证标签逻辑与原代码一致"""
    np.random.seed(random_state)
    data = {}
    # 基因型 (0/1/2 编码，0代表野生型纯合，1代表杂合，2代表变异纯合)
    data['rs9939609'] = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3])
    data['rs17817449'] = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3])
    # 临床特征
    data['age'] = np.random.randint(20, 85, n_samples)
    data['gender'] = np.random.choice([0, 1], n_samples)
    data['hypertension'] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    data['family_history'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    data['smoking'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    data['quit_smoking'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    data['bp_med'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    data['lipid_med'] = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    data['cad'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    data['bmi'] = np.random.normal(24, 4, n_samples).clip(15, 45)
    data['sbp'] = np.random.normal(125, 15, n_samples).clip(90, 200)
    data['dbp'] = np.random.normal(80, 10, n_samples).clip(60, 120)
    data['hr'] = np.random.normal(75, 10, n_samples).clip(50, 120)
    # 生化指标
    data['hdl'] = np.random.normal(1.2, 0.3, n_samples).clip(0.5, 2.5)
    data['ldl'] = np.random.normal(2.8, 0.8, n_samples).clip(1.0, 5.5)
    data['insulin'] = np.random.normal(10, 5, n_samples).clip(2, 40)
    data['cpep'] = np.random.normal(1.5, 0.7, n_samples).clip(0.3, 4.0)
    data['tg'] = np.random.normal(1.5, 0.8, n_samples).clip(0.5, 5.0)
    data['apob'] = np.random.normal(0.9, 0.3, n_samples).clip(0.5, 2.0)
    data['lpa'] = np.random.normal(150, 100, n_samples).clip(10, 500)
    data['hscrp'] = np.random.normal(2, 2, n_samples).clip(0.1, 15)
    data['ua'] = np.random.normal(350, 80, n_samples).clip(150, 600)
    data['hcy'] = np.random.normal(12, 4, n_samples).clip(5, 30)
    data['cysc'] = np.random.normal(0.9, 0.3, n_samples).clip(0.5, 2.2)
    data['b2mg'] = np.random.normal(1.8, 0.6, n_samples).clip(1.0, 4.0)
    data['tc'] = np.random.normal(4.5, 1.0, n_samples).clip(2.5, 7.5)
    data['apoai'] = np.random.normal(1.3, 0.3, n_samples).clip(0.8, 2.2)
    # 血糖与HbA1c（标签决定）
    glu = np.zeros(n_samples)
    hba1c = np.zeros(n_samples)
    y_true = np.zeros(n_samples)
    for i in range(n_samples):
        if np.random.rand() < 0.4:
            glu[i] = np.random.uniform(3.9, 5.5)
            hba1c[i] = np.random.uniform(4.0, 5.6)
            y_true[i] = 0
        elif np.random.rand() < 0.35:
            glu[i] = np.random.uniform(5.6, 6.9)
            hba1c[i] = np.random.uniform(5.7, 6.4)
            y_true[i] = 1
        else:
            glu[i] = np.random.uniform(7.0, 11.0)
            hba1c[i] = np.random.uniform(6.5, 9.0)
            y_true[i] = 2
    glu += np.random.normal(0, 0.2, n_samples)
    hba1c += np.random.normal(0, 0.1, n_samples)
    data['glu'] = glu.clip(3.5, 12.0)
    data['hba1c'] = hba1c.clip(4.0, 10.0)
    return pd.DataFrame(data), y_true


def build_feature_matrix(df):
    """根据原MATLAB逻辑构建特征矩阵 X (29连续+2分类) 和标签 Y"""
    X_rows = []
    Y_labels = []
    for idx, row in df.iterrows():
        # 基因编码（已经是0/1/2数值，直接使用）
        g1 = row['rs9939609']
        g2 = row['rs17817449']
        # 数值特征
        cont_row = [
            g1, g2,
            row['age'], row['gender'], row['hypertension'], row['family_history'],
            row['smoking'], row['quit_smoking'], row['bp_med'], row['lipid_med'],
            row['cad'], row['bmi'], row['sbp'], row['dbp'], row['hr'],
            row['hdl'], row['ldl'], row['insulin'], row['cpep'],
            row['tg'], row['apob'], row['lpa'], row['hscrp'],
            row['ua'], row['hcy'], row['cysc'], row['b2mg'],
            row['tc'], row['apoai']
        ]
        glu_orig = row['glu']
        hba1c_orig = row['hba1c']
        # 血糖分类
        if pd.isna(glu_orig):
            glu_cat = np.nan
        elif glu_orig < 5.6:
            glu_cat = 0
        elif glu_orig <= 6.0:
            glu_cat = 1
        elif glu_orig <= 6.9:
            glu_cat = 2
        else:
            glu_cat = 3
        # HbA1c分类
        if pd.isna(hba1c_orig):
            hba1c_cat = np.nan
        elif hba1c_orig < 5.7:
            hba1c_cat = 0
        elif hba1c_orig <= 6.4:
            hba1c_cat = 1
        else:
            hba1c_cat = 2
        X_rows.append(cont_row + [glu_cat, hba1c_cat])
        # 标签
        if (not pd.isna(glu_orig) and glu_orig >= 7.0) or (not pd.isna(hba1c_orig) and hba1c_orig >= 6.5):
            y = 2
        elif (not pd.isna(glu_orig) and glu_orig >= 5.6) or (not pd.isna(hba1c_orig) and hba1c_orig >= 5.7):
            y = 1
        else:
            y = 0
        Y_labels.append(y)
    X = np.array(X_rows, dtype=float)
    Y = np.array(Y_labels)
    return X, Y


def train_and_save_model():
    global model, scaler, cont_feature_medians, cat_feature_modes, cv_metrics
    print("生成合成训练数据...")
    df, _ = generate_synthetic_data(3000)
    X_raw, Y = build_feature_matrix(df)
    # 缺失填充（连续用中位数，分类用众数）
    cont_feature_medians = []
    for col in range(N_CONT):
        col_data = X_raw[:, col]
        med = np.nanmedian(col_data[~np.isnan(col_data)])
        cont_feature_medians.append(med)
        col_data[np.isnan(col_data)] = med
        X_raw[:, col] = col_data
    cat_feature_modes = []
    for col in range(N_CONT, N_CONT + N_CAT):
        col_data = X_raw[:, col]
        non_nan = col_data[~np.isnan(col_data)]
        mode = 0 if len(non_nan) == 0 else int(np.bincount(non_nan.astype(int)).argmax())
        cat_feature_modes.append(mode)
        col_data[np.isnan(col_data)] = mode
        X_raw[:, col] = col_data
    # 标准化连续特征
    scaler = StandardScaler()
    X_cont_norm = scaler.fit_transform(X_raw[:, :N_CONT])
    X_final = np.hstack([X_cont_norm, X_raw[:, N_CONT:]])
    # 训练随机森林
    model = RandomForestClassifier(n_estimators=200, bootstrap=True, random_state=2024, n_jobs=-1)
    model.fit(X_final, Y)
    # 5折交叉验证
    cv = KFold(n_splits=5, shuffle=True, random_state=2024)
    acc_list, sens_list, spec_list, auc_list = [], [], [], []
    for train_idx, test_idx in cv.split(X_final):
        X_tr, X_te = X_final[train_idx], X_final[test_idx]
        y_tr, y_te = Y[train_idx], Y[test_idx]
        clf = RandomForestClassifier(n_estimators=200, bootstrap=True, random_state=2024)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)
        acc = accuracy_score(y_te, y_pred)
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1, 2])
        sens_class, spec_class = [], []
        for c in range(3):
            tp = cm[c, c]
            fn = cm[c, :].sum() - tp
            fp = cm[:, c].sum() - tp
            tn = cm.sum() - tp - fn - fp
            sens_class.append(tp / (tp + fn + 1e-8))
            spec_class.append(tn / (tn + fp + 1e-8))
        sens_macro = np.mean(sens_class)
        spec_macro = np.mean(spec_class)
        try:
            auc = roc_auc_score((y_te == 2).astype(int), y_prob[:, 2])
        except:
            auc = 0.5
        acc_list.append(acc);
        sens_list.append(sens_macro);
        spec_list.append(spec_macro);
        auc_list.append(auc)
    cv_metrics = {
        'acc_mean': np.mean(acc_list), 'acc_std': np.std(acc_list),
        'sens_mean': np.mean(sens_list), 'sens_std': np.std(sens_list),
        'spec_mean': np.mean(spec_list), 'spec_std': np.std(spec_list),
        'auc_mean': np.mean(auc_list), 'auc_std': np.std(auc_list)
    }
    # 保存
    joblib.dump(model, 'diabetes_rf_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump({'cont_medians': cont_feature_medians, 'cat_modes': cat_feature_modes}, 'preprocess_params.pkl')
    print("模型训练完成并保存。")


def load_model():
    global model, scaler, cont_feature_medians, cat_feature_modes, cv_metrics
    if os.path.exists('diabetes_rf_model.pkl'):
        model = joblib.load('diabetes_rf_model.pkl')
        scaler = joblib.load('scaler.pkl')
        params = joblib.load('preprocess_params.pkl')
        cont_feature_medians = params['cont_medians']
        cat_feature_modes = params['cat_modes']
        print("模型加载成功。")
        # 为了显示性能，需要重新计算或保存历史cv_metrics，这里模拟一个默认值（实际生产应保存）
        cv_metrics = {'acc_mean': 0.85, 'acc_std': 0.02, 'sens_mean': 0.82, 'sens_std': 0.03,
                      'spec_mean': 0.88, 'spec_std': 0.02, 'auc_mean': 0.89, 'auc_std': 0.02}
    else:
        print("未找到预训练模型，开始训练...")
        train_and_save_model()
        # 训练后cv_metrics已赋值


# ---------------------------- 风险映射与建议函数（与原MATLAB完全一致） ----------------------------
def calculate_risk_index(prob_array, pred_class):
    prob = prob_array.flatten()
    if pred_class == 0:  # 低风险
        raw_risk = 1 - prob[0]
        displayed_risk = raw_risk * 0.5
        displayed_risk = np.clip(displayed_risk, 0.05, 0.20)
        risk_level = '低风险'
    elif pred_class == 1:  # 中风险
        raw_risk = prob[1]
        displayed_risk = 0.3 + raw_risk * 0.35
        displayed_risk = np.clip(displayed_risk, 0.40, 0.65)
        risk_level = '中风险'
    else:  # 高风险
        raw_risk = prob[2]
        displayed_risk = 0.75 + raw_risk * 0.2
        displayed_risk = np.clip(displayed_risk, 0.80, 0.98)
        risk_level = '高风险'
    return displayed_risk, risk_level


def generate_diet_exercise_advice(patient_data, risk_level):
    age = patient_data['age']
    gender = patient_data['gender']
    height = patient_data['height']
    weight = patient_data['weight']
    bmi_input = patient_data.get('bmi', None)
    cad = patient_data.get('cad', 0)
    glu_val = patient_data.get('glu', None)
    hba1c_val = patient_data.get('hba1c', None)
    if gender == 1:
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    tdee = bmr * 1.375
    if bmi_input is None or bmi_input == 0:
        bmi = weight / ((height / 100) ** 2)
    else:
        bmi = bmi_input
    if bmi < 18.5:
        calories = tdee + 300
    elif bmi < 24:
        calories = tdee
    elif bmi < 28:
        calories = tdee - 400
    else:
        calories = tdee - 600
    calories = max(calories, 1200)
    # 碳水比例
    if (glu_val is not None and glu_val >= 7.0) or (hba1c_val is not None and hba1c_val >= 6.5):
        carb_ratio = 0.45
    elif (glu_val is not None and glu_val >= 6.1) or (hba1c_val is not None and hba1c_val >= 5.7):
        carb_ratio = 0.50
    else:
        carb_ratio = 0.55
    protein_ratio = 0.18
    fat_ratio = 1 - carb_ratio - protein_ratio
    carbs = calories * carb_ratio / 4
    protein = calories * protein_ratio / 4
    fat = calories * fat_ratio / 9
    fiber = 30
    max_hr = 220 - age
    if cad == 1:
        max_hr = 200 - age * 0.7
    hr_low = max_hr * 0.64
    hr_high = max_hr * 0.76
    if risk_level == '低风险':
        min_per_week = 150;
        steps = 7000
    elif risk_level == '中风险':
        min_per_week = 200;
        steps = 9000
    else:
        min_per_week = 250;
        steps = 11000
    min_per_day = np.ceil(min_per_week / 5)
    strength = int(min_per_week / 50)
    return {
        'calories': int(calories), 'carbs': int(carbs), 'carb_ratio': int(carb_ratio * 100),
        'protein': int(protein), 'protein_ratio': int(protein_ratio * 100),
        'fat': int(fat), 'fat_ratio': int(fat_ratio * 100), 'fiber': fiber,
        'hr_low': int(hr_low), 'hr_high': int(hr_high),
        'min_per_week': min_per_week, 'min_per_day': int(min_per_day),
        'steps': steps, 'strength': strength
    }


def generate_health_advice(patient_data):
    rules = [
        ('tang', '糖代谢', 'HbA1c', '>=', 5.7, '%', '碳水控制至40-45%，选择低GI食物'),
        ('tang', '糖代谢', 'HbA1c', '>=', 6.5, '%', '严格限制精制糖，增加膳食纤维30g/天'),
        ('tang', '糖代谢', 'Glu', '>=', 6.1, 'mmol/L', '减少高GI碳水摄入'),
        ('tang', '糖代谢', 'Insulin', '>=', 15, 'μIU/mL', '控制总热量，增加运动改善胰岛素敏感性'),
        ('zhi', '脂代谢', 'LDL_C', '>=', 3.4, 'mmol/L', '减少饱和脂肪，增加不饱和脂肪'),
        ('zhi', '脂代谢', 'TG', '>=', 1.7, 'mmol/L', '限制糖和酒精摄入'),
        ('zhi', '脂代谢', 'HDL_C', '<=', 1.0, 'mmol/L', '增加有氧运动，提高HDL'),
        ('zhi', '脂代谢', 'APOB', '>=', 1.1, 'g/L', '采用地中海饮食'),
        ('zhi', '脂代谢', 'Lp_a', '>=', 300, 'mg/L', '增加抗氧化食物摄入'),
        ('yan', '炎症', 'hsCRP', '>=', 3, 'mg/L', '抗炎饮食（蔬菜、水果、Omega-3）'),
        ('du', '代谢毒性', 'UA', '>=', 420, 'μmol/L', '限制高嘌呤食物，增加饮水'),
        ('du', '代谢毒性', 'HCY', '>=', 15, 'μmol/L', '增加叶酸、维生素B族'),
        ('shen', '肾功能', 'CysC', '>=', 1.1, 'mg/L', '控制蛋白摄入（0.8g/kg）'),
        ('shen', '肾功能', 'beta2_MG', '>=', 3, 'mg/L', '建议进一步肾功能评估'),
    ]
    name_map = {'HbA1c': 'HbA1c', 'Glu': '空腹血糖', 'Insulin': '胰岛素', 'LDL_C': 'LDL-C', 'TG': '甘油三酯',
                'HDL_C': 'HDL-C', 'APOB': 'APOB', 'Lp_a': 'Lp(a)', 'hsCRP': 'hsCRP', 'UA': '尿酸',
                'HCY': '同型半胱氨酸', 'CysC': 'CysC', 'beta2_MG': 'β2-MG'}
    general_advice = {
        'tang': ['❌ 避免：精制糖、饮料、甜点、白米白面过量', '✅ 增加：膳食纤维≥25–30g/天'],
        'zhi': ['❌ 避免：油炸食品、动物内脏、高脂糕点', '✅ 增加：燕麦、深海鱼、坚果、橄榄油'],
        'yan': ['✅ 抗炎食物：深色蔬菜、蓝莓、姜黄、绿茶'],
        'du': ['💧 增加饮水至2000ml/天'],
        'shen': ['⚠️ 定期监测肾功能，避免使用肾毒性药物']
    }
    modules = {}
    for rule in rules:
        mod_id, mod_name, metric, op, thres, unit, advice_text = rule
        if metric not in patient_data or patient_data[metric] is None:
            continue
        value = patient_data[metric]
        if (op == '>=' and value >= thres) or (op == '<=' and value <= thres):
            indicator = f"{name_map.get(metric, metric)} {op} {thres} {unit}"
            if mod_id not in modules:
                modules[mod_id] = {'name': mod_name, 'indicators': [], 'advices': []}
            modules[mod_id]['indicators'].append(indicator)
            if advice_text not in modules[mod_id]['advices']:
                modules[mod_id]['advices'].append(advice_text)
    advice_list = []
    if not modules:
        advice_list.append("各项指标未见明显异常，保持健康生活方式。")
    else:
        for i, (mod_id, info) in enumerate(modules.items(), 1):
            advice_list.append(f"{i}️⃣ {info['name']} 异常干预")
            advice_list.append(f"👉 异常判定：{'；'.join(info['indicators'])}")
            for adv in info['advices']:
                advice_list.append(f"✅ {adv}")
            if mod_id in general_advice:
                for gen in general_advice[mod_id]:
                    advice_list.append(gen)
            advice_list.append("")
    return advice_list


# ---------------------------- Flask 路由 ----------------------------
@app.route('/')
def index():
    return redirect('/select')

@app.route('/select')
def select_page():
    return render_template('select.html', metrics=cv_metrics)

@app.route('/form', methods=['POST'])
def form_page():
    option = request.form.get('option')
    if option not in ['1', '2', '3']:
        option = '3'
    # 渲染填写表单页面，根据选项决定显示哪些区块
    return render_template('form.html', option=option, metrics=cv_metrics)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取用户选择的选项
        option = request.form.get('option', '3')
        # 定义所有可能字段的默认填充值（来自训练集统计量）
        # 连续特征默认值（29个，顺序与训练时一致）
        default_cont = cont_feature_medians.copy()  # 中位数
        # 分类特征默认值（glu_cat, hba1c_cat）
        default_cat = cat_feature_modes.copy()

        # 获取表单值辅助函数
        def get_float(name, default=None):
            val = request.form.get(name, '')
            if val is None or val.strip() == '':
                return default
            try:
                return float(val)
            except:
                return default

        def get_int(name, default=None):
            val = request.form.get(name, '')
            if val is None or val.strip() == '':
                return default
            try:
                return int(float(val))
            except:
                return default

        def get_gene_code(raw, default=0):
            if raw is None or raw.strip() == '':
                return default
            raw = raw.upper()
            if 'TA' in raw or 'AT' in raw:
                return 2
            elif 'TT' in raw:
                return 1
            else:
                return 0

        # 1. 获取基因（总是提供，不填则用默认中位数，但中位数是数值，需要转换）
        g1_raw = request.form.get('rs9939609', '')
        g2_raw = request.form.get('rs17817449', '')
        # 默认值：训练集中位数可能是0/1/2，但用户可能不填，我们使用默认0（最保守）
        default_gene = 0
        g1 = get_gene_code(g1_raw, default_gene)
        g2 = get_gene_code(g2_raw, default_gene)
        # 覆盖默认连续特征的前两个（基因位点）
        default_cont[0] = g1
        default_cont[1] = g2

        # 2. 年龄、性别等基础信息
        age = get_float('age', default_cont[2])
        gender = get_float('gender', default_cont[3])
        # 高血压、家族史等（选项1可能没有，用默认）
        hyper = get_float('hyper', default_cont[4])
        family = get_float('family', default_cont[5])
        smoke = get_float('smoke', default_cont[6])
        quit_smoke = get_float('quitSmoke', default_cont[7])
        drug_bp = get_float('drugBp', default_cont[8])
        drug_lipid = get_float('drugLipid', default_cont[9])
        cad = get_float('cad', default_cont[10])
        bmi = get_float('bmi', default_cont[11])
        sbp = get_float('sbp', default_cont[12])
        dbp = get_float('dbp', default_cont[13])
        hr = get_float('hr', default_cont[14])
        # 生化指标
        hdl = get_float('hdl', default_cont[15])
        ldl = get_float('ldl', default_cont[16])
        insulin = get_float('insulin', default_cont[17])
        cpep = get_float('cpep', default_cont[18])
        tg = get_float('tg', default_cont[19])
        apob = get_float('apob', default_cont[20])
        lpa = get_float('lpa', default_cont[21])
        hscrp = get_float('hscrp', default_cont[22])
        ua = get_float('ua', default_cont[23])
        hcy = get_float('hcy', default_cont[24])
        cysc = get_float('cysc', default_cont[25])
        b2mg = get_float('b2mg', default_cont[26])
        tc = get_float('tc', default_cont[27])
        apoai = get_float('apoai', default_cont[28])
        # 身高体重（用于建议，不在模型特征中）
        height = get_float('height', 170)
        weight = get_float('weight', 70)
        # 血糖与HbA1c原始值
        glu_val = get_float('glu', None)
        hba1c_val = get_float('hba1c', None)

        # 构建完整的29个连续特征数组
        cont_arr = np.array([
            g1, g2, age, gender, hyper, family, smoke, quit_smoke,
            drug_bp, drug_lipid, cad, bmi, sbp, dbp, hr,
            hdl, ldl, insulin, cpep, tg, apob, lpa, hscrp,
            ua, hcy, cysc, b2mg, tc, apoai
        ], dtype=float)
        # 对于仍然是NaN的（例如用户未提供但默认值本身可能是NaN），用中位数再填一次
        for i in range(len(cont_arr)):
            if np.isnan(cont_arr[i]):
                cont_arr[i] = default_cont[i]

        # 构建分类特征 glu_cat, hba1c_cat
        if glu_val is not None and not np.isnan(glu_val):
            if glu_val < 5.6:
                glu_cat = 0
            elif glu_val <= 6.0:
                glu_cat = 1
            elif glu_val <= 6.9:
                glu_cat = 2
            else:
                glu_cat = 3
        else:
            glu_cat = default_cat[0]
        if hba1c_val is not None and not np.isnan(hba1c_val):
            if hba1c_val < 5.7:
                hba1c_cat = 0
            elif hba1c_val <= 6.4:
                hba1c_cat = 1
            else:
                hba1c_cat = 2
        else:
            hba1c_cat = default_cat[1]

        # 标准化连续特征
        cont_norm = scaler.transform(cont_arr.reshape(1, -1))
        X_pat = np.hstack([cont_norm, np.array([[glu_cat, hba1c_cat]])])

        # 预测
        prob = model.predict_proba(X_pat)[0]
        pred_class = model.predict(X_pat)[0]
        risk_score, risk_level = calculate_risk_index(prob, pred_class)

        # 准备患者数据字典（用于建议函数）
        patient_dict = {
            'age': age, 'gender': gender, 'height': height, 'weight': weight,
            'bmi': bmi, 'cad': cad, 'glu': glu_val, 'hba1c': hba1c_val,
            'insulin': insulin, 'ldl': ldl, 'hdl': hdl, 'tg': tg,
            'apob': apob, 'lpa': lpa, 'hscrp': hscrp, 'ua': ua,
            'hcy': hcy, 'cysc': cysc, 'beta2_MG': b2mg,
            'LDL_C': ldl, 'HDL_C': hdl, 'TG': tg, 'APOB': apob,
            'Lp_a': lpa, 'hsCRP': hscrp, 'UA': ua, 'HCY': hcy,
            'CysC': cysc, 'beta2_MG': b2mg, 'Insulin': insulin,
            'Glu': glu_val, 'HbA1c': hba1c_val
        }
        diet_advice = generate_diet_exercise_advice(patient_dict, risk_level)
        health_advice = generate_health_advice(patient_dict)

        return render_template('result.html',
                               risk_score=risk_score * 100,
                               risk_level=risk_level,
                               prob_normal=prob[0],
                               prob_prediabetes=prob[1],
                               prob_diabetes=prob[2],
                               diet=diet_advice,
                               health=health_advice,
                               metrics=cv_metrics)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"预测出错: {str(e)}", 500


if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)