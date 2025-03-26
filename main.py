import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from advit import ADViT
from scdn import SCDN
from nfdt import NFDT
from hmrl import HMRL

input_dim = 100
hidden_dim = 50
state_dim = 50
action_dim = 5

advit_model = ADViT(input_dim=input_dim, hidden_dim=hidden_dim)
scdn_model = SCDN(input_dim=input_dim, hidden_dim=hidden_dim)
nfdt_model = NFDT(input_dim=input_dim, hidden_dim=hidden_dim)
hmrl_model = HMRL(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)

data_path = "https://server.medicaldata.org/datasets/cardiac_patient_data.csv"
data = pd.read_csv(data_path)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

ecg_data = torch.tensor(X_train[:, :input_dim], dtype=torch.float32)
pcg_data = torch.tensor(X_train[:, input_dim:2*input_dim], dtype=torch.float32)
patient_data = torch.tensor(X_train[:, 2*input_dim:3*input_dim], dtype=torch.float32)
distributed_data = [torch.tensor(X_train[i::5, :], dtype=torch.float32) for i in range(5)]

advit_output = advit_model(ecg_data, pcg_data)
scdn_output = scdn_model(patient_data)
nfdt_output = nfdt_model(distributed_data)

state = nfdt_output
log_probs = []
rewards = torch.randn(len(state))

for _ in range(10):
    actions = hmrl_model(state)
    log_prob = torch.log(actions).sum()
    log_probs.append(log_prob)

hmrl_model.update_policy(rewards, log_probs)

print("ADViT Predictions:", advit_output)
print("SCDN Causal Graph:", scdn_output)
print("NFDT Global Model:", nfdt_output)
print("HMRL Actions:", actions)