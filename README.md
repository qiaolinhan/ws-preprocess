# ws-preprocess
This is image restoration for UAV based wildfire segmentation, because it will always meet some disturbance, noise or other serious situation. 
The code is based on the fastai packages and the class https://course.fast.ai/videos/?lesson=7.  
The thought may comes from antique fraud, we want to produce a fake antique, and we got other authentic products during its specific age. After learning their features and finish making the fake, we show it to the expert and the expert tells us where we made it wrong. Then, every time we adjust more features make it harder for that expert to tell which is counterfeit.  

If you think its interesting, please star us :) 

## Building the environment 
* Install the Google compute platform on Linux
```
# Create environment variable for correct distribution
export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)"

# Add the Cloud SDK distribution URI as a package source
echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Import the Google Cloud Platform public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Update the package list and install the Cloud SDK
sudo apt-get update && sudo apt-get install google-cloud-sdk
```
  
* Conda install fastai packages
```
sudo /opt/anaconda3/bin/conda install -c fastai fastai
```
**Fastai give us many ideas to avoid running out of GPU RAM, like the `mixed procision training` to make us train the model on 16 bite position.** This is mentioned in class 3 https://course.fast.ai/videos/?lesson=3

## GAN1
This repo is the fist part of the image restoration model.    
This is a **generative adversarial netork (GAN) model**.    
  * **Generator**: Resnet34 pretrained U-net.
  * **Discriminator**: A binary classifier with adaptive binary cross entropy loss, which is foun by fastai function `gan_critic()`.  
  * Using spectral normalization, based on the paper: https://arxiv.org/abs/1802.05957.  
  
