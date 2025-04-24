from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from tutorial_dataset import MyDataset
# from satellite_dataset import MyDataset
from satellite_tiles_Yuzhou import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch

torch.cuda.empty_cache()

# image_dir = "/home/gridsan/qwang/satellite_images/zoom17/"
# data_dir = "/home/gridsan/qwang/JTL-transit_shared/deep_hybrid_model/data/"
print('Start loading data from Yuzhou train script')

#image_dir = "fml/Streetview images/downloaded_images"
#data_dir = ["fml/Streetview images/segmentation_results_ready.csv"]
#hint_dir = "fml/Streetview images/hint"

image_dir = "fml/TwoCities_Images_Processed"
data_dir = ["fml/Two_cities_final_prompt.csv"]
hint_dir = "blue_shenhaowang/yuzhouchen1/fml/Streetview images/hint"
#hint_dir = "fml/Segmentation_Results"

# Configs
#resume_path = 'GenerativeUrbanDesign/models/control_sd15_ini.ckpt'
resume_path = 'training_logs/lightning_logs/version_65304663/checkpoints/epoch=20-step=164010.ckpt'
#resume_path = './lightning_logs/version_24305430/checkpoints/epoch=4-step=112594.ckpt'

batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

#def custom_collate_fn(batch):
    #collated_batch = {}
    #collated_batch['jpg'] = torch.stack([item['jpg'] for item in batch])
    #collated_batch['hint'] = torch.stack([item['hint'] for item in batch])
    #collated_batch['txt'] = [item['txt'] for item in batch]  # 保留原始字符串列表
    #return collated_batch

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('GenerativeUrbanDesign/models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset(image_dir, data_dir, hint_dir)



dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True) #,collate_fn=custom_collate_fn

logger = ImageLogger(batch_frequency=logger_freq)


trainer = pl.Trainer(
    accelerator='gpu', devices=[0],
    precision=32, 
    callbacks=[logger],
    default_root_dir="training_logs",
    resume_from_checkpoint=resume_path
)




# Train!
trainer.fit(model, dataloader)
#trainer.fit(model, dataloader, ckpt_path=resume_path)