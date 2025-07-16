from model import model_v2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Loading in model 
uploaded_model = model_v2()

#Loading in weights/settings
model_save_path = "DLM_RC.pth"
uploaded_model.load_state_dict(torch.load(model_save_path))

#Converting to eval model
uploaded_model.eval()

def predict(window): #Data is in (3, 128):
	
	window = torch.from_numpy(window).type(torch.float32)
	window = window.reshape(1,3,128)
	
	
	with torch.inference_mode():
		y_logit = uploaded_model(window)
		prediction = torch.softmax(y_logit, dim=1).argmax(dim=1)
	
	return prediction.item()
