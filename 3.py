import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 页面标题
st.title("Iris 花种分类模型")
 
# 加载数据集
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
 
# 训练模型
model = RandomForestClassifier()
model.fit(X_train, y_train)
 
# 用户输入
st.subheader("输入特征值")
sepal_length = st.slider("花萼长度 (cm)", 4.0, 8.0, 5.5)
sepal_width = st.slider("花萼宽度 (cm)", 2.0, 5.0, 3.5)
petal_length = st.slider("花瓣长度 (cm)", 1.0, 7.0, 4.5)
petal_width = st.slider("花瓣宽度 (cm)", 0.1, 3.0, 1.5)
 
# 预测
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)
probability = model.predict_proba(input_data)
 
# 显示结果
st.subheader("预测结果")
st.write(f"预测的花种: {iris.target_names[prediction][0]}")
 
# 显示概率
st.subheader("分类概率")
probability_df = pd.DataFrame({
    '花种': iris.target_names,
    '概率': probability[0]
}).sort_values('概率', ascending=False)
 
st.dataframe(probability_df.style.format({'概率': '{:.2%}'}))
 
# 模型性能
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"模型准确率: {accuracy:.2f}")