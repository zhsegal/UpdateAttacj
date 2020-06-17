import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_set = "Cifar"
update_set_size=2

a1 = torch.load(data_set+'saved/delta_set0-1000.pt', map_location=torch.device(device))
a2 = torch.load(data_set+'saved/delta_set1000-2000.pt', map_location=torch.device(device))
a3 = torch.load(data_set+'saved/delta_set2000-3000.pt', map_location=torch.device(device))
a4 = torch.load(data_set+'saved/delta_set3000-4000.pt', map_location=torch.device(device))
a5 = torch.load(data_set+'saved/delta_set4000-5000.pt', map_location=torch.device(device))
a6 = torch.load(data_set+'saved/delta_set5000-6000.pt', map_location=torch.device(device))
a7 = torch.load(data_set+'saved/delta_set6000-7000.pt', map_location=torch.device(device))
a8 = torch.load(data_set+'saved/delta_set7000-8000.pt', map_location=torch.device(device))
a9 = torch.load(data_set+'saved/delta_set8000-9000.pt', map_location=torch.device(device))
a10 = torch.load(data_set+'saved/delta_set9000-10000.pt', map_location=torch.device(device))

list = torch.stack([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]).view(-1, 1000)
mean = torch.mean(list)
var = torch.var(list)
print (var)
var[var == 0] = 1e-10
list=(list - mean) / var
torch.save(list, data_set+'saved/delta_set' + str(update_set_size) + '.pt')
