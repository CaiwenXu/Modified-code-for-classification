# Modified-code-for-classification
Modified code for the calssification task using MedicalNet
# Modified code for the calssification task using MedicalNet
Note: 
1.MedicalNet https://github.com/Tencent/MedicalNet
```
    @article{chen2019med3d,
        title={Med3D: Transfer Learning for 3D Medical Image Analysis},
        author={Chen, Sihong and Ma, Kai and Zheng, Yefeng},
        journal={arXiv preprint arXiv:1904.00625},
        year={2019}
    }
```
      
2. Modified code
 1) Load NIFT(batch process): MedicalNet_datasets_COVID19_dataset.py
 2) Train&test for classification (softmaxlayer is included for normalization)
 3) Modifie settings before you run MedicalNet_NEW.py for results
 4) Ground truth, predicted label and score are generated and saved as m_all.mat, pre_all.mat, and po_all.mat.
 5) Visualization part is not included in this code 
 6) Kindly wait for my update. This code is only for your reference.
