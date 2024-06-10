import torch
import torch.nn as nn
import streamlit as st
import nibabel as nib
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
from scipy import ndimage

class CNN_classification_model(nn.Module):
    def __init__(self):
        super(CNN_classification_model, self).__init__()

        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d((2, 2, 2))
        self.norm3d1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool3d((2, 2, 2))
        self.norm3d2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=0)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool3d((2, 2, 2))
        self.norm3d3 = nn.BatchNorm3d(128)

        self.flat = nn.Flatten()
        # Linear 1
        self.linear1 = nn.Linear(10**3*128, 256)
        # Relu
        self.relu_end = nn.ReLU()
        # BatchNorm1d
        self.norm = nn.BatchNorm1d(256)
        # Dropout
        self.drop = nn.Dropout(p=0.3)
        # Linear 2
        self.linear2 = nn.Linear(256, num_classes) #num_classes
        # nn.Softmax(dim=1)

        #)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.norm3d1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.norm3d2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.norm3d3(x)
        print("maxpool4",x.shape)

        x = self.flat(x)
        #print("flat",x.shape)
        x = self.linear1(x)

        x = self.relu_end(x)
        x = self.norm(x)
        x = self.drop(x)
        out = self.linear2(x)
        #print("out", out)
        return out

def normalize2(volume):
    # Normalize the volume to zero mean and unit variance
    volume = (volume - np.mean(volume)) / np.std(volume)  # <-- Added normalization
    volume = volume.astype("float32")
    return volume

def resize_volume(img,size):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = size
    desired_width = size
    desired_height = size
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 180, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

num_classes = 4
model = CNN_classification_model()
model.load_state_dict(torch.load('saved_model4_96.pth'))
model.eval()

def main():
    #st.title("Preclinical Alzheimer Disease Predictor")
    html_temp = """
        <div style="background:#025246; padding:10px; width:120%;">
        <h2 style="color:white; text-align:center;">Preclinical Alzheimer Disease Predictor Web</h2>
        </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.subheader("By Punnut Chirdchuvutikun from AI Builder camp")
    st.markdown("Upload mri brain scan that preprocess from Freesurfer program that convert from .mgz to .nii file")
    st.markdown("Diagnosis only ALzheimer disease, Normal Forgetfulness of elderly people, Forgetfulness of elderly people that can develop into Alzheimer disease in the future, Healthy person")
    st.markdown("Upload only .nii that have size (256,256,256)")
    uploaded_file = st.file_uploader("Insert a .nii file", type=["nii"])
    click = st.button("Predict")
    if click == True:
        with st.spinner('Waiting for AI to diagnosis...'):
            if uploaded_file is not None:
                print("print", uploaded_file.name)
                try:
                    data = uploaded_file.getvalue()
                    header_size = 352
                    header = data[:header_size]
                    image_data = data[header_size:]
                    data2 = np.frombuffer(image_data, dtype=np.uint8)
                    data2 = data2.reshape((256, 256, 256))
                    data2 = nib.Nifti1Image(data2, affine=np.eye(4))
                except:
                    st.markdown("the .nii file that upload doesn't have exactly (256,256,256) shape")

                fig, ax = plt.subplots(figsize=[10, 5])
                plotting.plot_img(data2, cmap='gray', axes=ax, display_mode='mosaic')  # , cut_coords=(0, 0, 0))
                st.pyplot(fig)

                resize = 96
                nii_img = data2.get_fdata()
                nii_img = normalize2(nii_img)
                nii_img = resize_volume(nii_img, resize)

                nii_img = nii_img.reshape(1, 1, resize, resize, resize)
                nii_img = torch.as_tensor(nii_img)
                gpu = torch.cuda.is_available()
                print("gpu =", gpu)
                if gpu:
                    model.cuda()
                    nii_img.cuda()
                outputs = model(nii_img.to(torch.float32))
                predicted = int(torch.argmax(outputs))
                if predicted == 0:
                    st.subheader("This brain has a: Alzheimer Disease")
                elif predicted == 1:
                    st.subheader("This brain has a: Healthy brain")
                elif predicted == 2:
                    st.subheader("This brain has a: Normal Forgetfulness of elderly people")
                elif predicted == 3:
                    st.subheader(
                        "This brain has a: Forgetfulness of elderly people that can develop into Alzheimer disease in the future")
                else:
                    st.subheader("Can't diagnosis disease of this brain")
            else:
                st.markdown("Please upload the data for the model to diagnosis.")
    st.subheader("A way to convert output .mgz from Freesurfer to .nii file.")
    st.markdown("1. use mri_convert command in Freesurfer (mri_convert Downloads/input.mgz Downloads/output.nii)")
    st.markdown("2. run this code")
    code = '''for root, subdirs, files in os.walk(storePath):
    for f in files:
        if f.endswith('.mgz'):
            start_path = os.path.join(root, f)
            new_path = start_path.replace(storePath, endPath)

            new_path = new_path.replace(".mgz", ".nii")
            print(start_path, ".....start_path")
            print(new_path, ".....end_path")
            if not os.path.exists(new_path.replace(f.replace(".mgz", ".nii"), "")):
                os.makedirs(new_path.replace(f.replace(".mgz", ".nii"), ""))
            img = nib.load(start_path)
            nib.save(img, new_path)'''
    st.code(code, language='python')






if __name__=='__main__':
    main()



