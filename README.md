# multichannel-VAE-survival-classification

---

****************Author:**************** Florence Townend

Based on work in Antelmi, L., Ayache, N., Robert, P., & Lorenzi, M. (2019). Sparse Multi-Channel Variational Autoencoder for the Joint Analysis of Heterogeneous Data. *Proceedings of the 36th International Conference on Machine Learning*, 302–311.

This code is used to investigate whether using latent-space-based data fusion to fuse clinical and neuroimaging features will lead to improved survival classification as opposed to uni-modal or concatenation-based models.

The MCVAE creates a joint latent representation of the distinct channels, and this work then takes that latent representation and inputs it into an SVM to classify patients as short- or long-survivors. (originally this was done for motor neurone disease but this work is not application-specific)
