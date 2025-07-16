from torch import nn

class model_v2(nn.Module):
	def __init__(self):
		super().__init__()
		
		#Making layers
		self.stack_layers = nn.Sequential(
			nn.Flatten(),
			nn.Linear(in_features=384, out_features=128),
			nn.ReLU(),
			nn.Linear(in_features=128, out_features=64),
			nn.ReLU(),
			nn.Linear(in_features=64, out_features=6)
			)
	
	#Forward function
	def forward(self, x):
		return self.stack_layers(x)

class model_v3(nn.Module):
	def __init__(self):
		super().__init__()
		
		#Defining layers
		self.stack_layers = nn.Sequential(
			nn.Flatten(),
			nn.Linear(in_features=768, out_features=256),
			nn.ReLU(),
			nn.Linear(in_features=256, out_features=128),
			nn.ReLU(),
			nn.Linear(in_features=128, out_features=64),
			nn.ReLU(),
			nn.Linear(in_features=64, out_features=6)
			)
	
	def forward(self, x):
		return self.stack_layers(x)
		
		
		
		
