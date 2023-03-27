import itertools
from statistics import mean
from turtle import color

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import rcParams
from matplotlib.patches import Ellipse

import mcvae
from mcvae.diagnostics import *
from mcvae.models import Mcvae, ThreeLayersVAE, TwoLayersVAE


class MCVAE_instance():

    def __init__(self, clin_data_df, bv_data_df, n_latents, n_epochs, lr=1e-2):
        """
        Args:
            clin_data_df (pd.DataFrame): _description_
            bv_data_df (pd.DataFrame): _description_    
            n_latents (int): _description_
            n_epochs (int): _description_
            lr (float, optional): _description_. Defaults to 1e-2.
        """
        self.clin_data_df = clin_data_df 
        self.bv_data_df = bv_data_df
        self.n_latents = n_latents
        self.n_epochs = n_epochs
        self.lr = lr
        self.dropout_test_latdims = 10
        self.survival_column = "long_survival_mri"
        self.mcvae_fit = None

    def mcvae_prepare_data(self):

        """
        Prepares the data for the MCVAE model to be trained on it

        """


        print("...preparing data for MCVAE...")
        # clin_df has all the predictors and the category classifications

        clinic_columns = list(self.clin_data_df.drop(columns=self.survival_column)) # drop the survival column

        bv_columns = list(self.bv_data_df) 

        normalize = lambda _: (_ - _.mean(0)) / _.std(0) # normalize the data

        clinic_value = self.clin_data_df[clinic_columns].values.astype('float') # convert to float
        clinic_value = normalize(clinic_value) # normalize the data

        bv_value = self.bv_data_df[bv_columns].values.astype('float') # convert to float
        bv_value = normalize(bv_value)  # normalize the data

        data_mcvae = [clinic_value, bv_value] 
        data_cols = [clinic_columns, bv_columns]

        data_mcvae_all = [torch.Tensor(_) for _ in data_mcvae] # convert to torch tensor

        self.data_mcvae_all = data_mcvae_all # save the data for later use

    
    def plot_dropouts(self, plot_flag=True):

        if self.mcvae_fit == None:  # if the model has not been trained yet
            # fit a model with 10 lat dims
            self.fit(big_fit=True)  # fit the model with 10 lat dims
            if plot_flag==True: self.plot_dropout_own(self.big_fit_model) 
        
        else:
            # use already_fit_model to plot the dropout
            if plot_flag==True: self.plot_dropout_own(self.mcvae_fit)



    def fit(self, big_fit=False):
        """Training the MCVAE model with the given number of latent dimensions and epochs.
        First train is with the given number of latent dimensions, second train is with 10 latent dimensions.

        Args:
            big_fit (bool, optional): Indicates whether the model has been trained yet with lots of epochs. Defaults to False.
        """
        FORCE_REFIT = True

        if big_fit == False: # if the model has not been trained yet
            print(f"...fitting MCVAE with {self.n_latents} latent dimensions and {self.n_epochs} epochs...")

            init_dict = {
                'n_channels': 2,
                'lat_dim': self.n_latents,
                'n_feats': tuple([self.data_mcvae_all[0].shape[1], self.data_mcvae_all[1].shape[1]]),
            }

            mcvae_fit = Mcvae(**init_dict, sparse=True)
            mcvae_fit.init_loss()   
            mcvae_fit.optimizer = torch.optim.Adam(mcvae_fit.parameters(), lr=self.lr) 
            mcvae_fit.optimize(epochs=self.n_epochs, data=self.data_mcvae_all) # fit the model
            self.mcvae_fit = mcvae_fit # save the model for later use

        else:
            print(f"...fitting MCVAE with {self.dropout_test_latdims} latent dimensions and {self.n_epochs} epochs...")

            init_dict = {
                'n_channels': 2,
                'lat_dim': self.dropout_test_latdims,
                'n_feats': tuple([self.data_mcvae_all[0].shape[1], self.data_mcvae_all[1].shape[1]]),
            }

            mcvae_fit = Mcvae(**init_dict, sparse=True)

            mcvae_fit.init_loss()
            mcvae_fit.optimizer = torch.optim.Adam(mcvae_fit.parameters(), lr=self.lr)
            mcvae_fit.optimize(epochs=self.n_epochs, data=self.data_mcvae_all)
            self.big_fit_model = mcvae_fit
        
    def plot_reconstruction_own(self, path_name=None):
        """
        Plot the reconstruction of the MCVAE model on the training data (scatter plot) 
        and the diagonal line (y=x) for comparison. 
        
        Args:
            path_name ([type], optional): [description]. Defaults to None.

        """

        set_fontsizes() # set the font sizes

        sns.set_style("white") # set the style of the plot
        # figure size in inches
        rcParams['figure.figsize'] = 4,4

        max_lim_x = np.max(np.hstack(self.data_mcvae_all)) # get the max value of the data
        max_lim_y = np.max(np.hstack(self.mcvae_fit.reconstruct(self.data_mcvae_all))) # get the max value of the reconstruction

        
        plt.plot([-10,10],[-10,10],c='blue',linestyle='--',alpha=0.5) # plot the diagonal line
        plt.scatter(np.hstack(self.data_mcvae_all), np.hstack(self.mcvae_fit.reconstruct(self.data_mcvae_all)),color='orangered',marker='o',s=3) # plot the scatter plot
        plt.xlim(-max_lim_x-0.5,max_lim_x+0.5)
        plt.ylim(-max_lim_y-0.5,max_lim_y+0.5)

        plt.xlabel("Real values")
        plt.ylabel("Reconstruction")
        plt.grid(True)
        plt.tight_layout()
        if path_name is not None:
            plt.savefig(f"Figures/{path_name}.pdf", dpi=80)

    
    def plot_loss_own(self, stop_at_convergence=True, fig_path=None, skip=0, path_name=None):
        """
        Plot the loss function of the MCVAE model during training (KL divergence, reconstruction loss, total loss).
        
        Args:
            stop_at_convergence (bool, optional): Whether to stop plot at convergence. Defaults to True.
            fig_path ([type], optional): Path to save figure to . Defaults to None.
            skip (int, optional): How many epochs to skip in figure. Defaults to 0.
            path_name ([type], optional): Alternative path to save figure to. Defaults to None.
            """
        sns.set_style("white")
        # figure size in inches
        rcParams['figure.figsize'] = 7,4

        set_fontsizes()

        legend_labels = ['Total Loss', 'KL Divergence', 'Reconstruction Loss']
        colors_loss = ['mediumturquoise', 'plum', 'coral']

        true_epochs = len(self.mcvae_fit.loss['total']) - 1
        if skip	> 0: 
            print(f'skipping first {skip} epochs where losses might be very high')
        losses = np.array([self.mcvae_fit.loss[key][skip:] for key in self.mcvae_fit.loss.keys()]).T # get the losses
        fig = plt.figure()
        try:
            plt.suptitle('Model ' + str(self.mcvae_fit.model_name) + '\n')
        except:
            pass
        plt.subplot(1, 2, 1)
        plt.grid(False)


        plt.xlabel('Epoch')
        plt.ylabel('Loss (common scale)')
        for i in range(3):
            plt.plot(losses[:,i], color=colors_loss[i],label=legend_labels[i],linewidth=2) 
        if not stop_at_convergence:
            plt.xlim([0, self.mcvae_fit.epochs])
        plt.subplot(1, 2, 2)

        plt.xlabel('Epoch')
        plt.ylabel('Loss (relative scale)')
        max_losses = 1e-8 + np.max(np.abs(losses), axis=0)
        for i in range(3):
            plt.plot((losses / max_losses)[:,i], color=colors_loss[i],label=legend_labels[i],linewidth=2) 
        if not stop_at_convergence:
            plt.xlim([0, self.mcvae_fit.epochs])


        plt.grid(False)
        plt.legend()
        plt.tight_layout()

        if path_name is not None:
            plt.savefig(f"Figures/{path_name}.pdf", dpi=80)

        if fig_path is not None:
            plt.rcParams['figure.figsize'] = (8, 5)
            plt.savefig(f'{fig_path}.png', bbox_inches='tight')
            plt.close()


    def plot_dropout_own(self, model, sort=True, path_name=None):
        """
        Plot the dropout probabilities of the model (sorted by default) 
        
        Args:
            model (MCVAE): the model to plot the dropout probabilities of
            sort (bool, optional): whether to sort the dropout probabilities. Defaults to True.
            path_name (str, optional): path to save the figure to. Defaults to None.
            
        """
        matplotlib.rc_file_defaults()
        sns.set_style("white")

        set_fontsizes()

  
        rcParams['figure.figsize'] = 4,4

        do = model.dropout.detach().numpy().reshape(-1)
        if sort:
            do = np.sort(do)

        plt.figure(figsize=(4,4))
        plt.bar(range(len(do)), do, alpha=1, color='mediumseagreen', ec='darkgreen',) # plot the dropout probabilities
        plt.hlines(0.2,-1,len(do)+1,colors='orangered',linestyles='-',linewidth=2) # plot the diagonal line
        plt.xlim(-0.75,len(do)-0.25)
        plt.xlabel('Latent dimension')
        plt.ylabel('Dropout probability')
        plt.grid(True)
        plt.tight_layout()
        if path_name is not None:
            plt.savefig(f"Figures/{path_name}.pdf", dpi=80)


    def make_mcvae_training_plots(self,):
        """
        Make the training plots for the MCVAE model including the loss, the dropout probabilities and the reconstruction.
        """
        self.plot_loss_own() # plot the loss
        plt.show()

        self.plot_dropout_own(self.mcvae_fit, sort=False) # plot the dropout probabilities
        plt.show()

        self.plot_reconstruction_own() # plot the reconstruction
        plt.show()


    
    def get_avg_zs(self, path_name, constrain_dropout=False):
        """
        Get the latent representations of the data points out of the MCVAE,
        averaged over the different channels.

        Args:
            path_name (_type_): path to save the figure to
            constrain_dropout (bool, optional): whether to constrain number of dims to ones that pass dropout constraints. 
                Only works with sparse MCVAE. Defaults to False.

        """

        if constrain_dropout == True: # if we want to constrain the number of dimensions to those that pass the dropout constraints
            self.indices = list(np.where(self.mcvae_fit.dropout.detach().numpy().flatten() < 0.2)[0]) # get the indices of the dimensions that pass the dropout constraints
        else:
            self.indices = [i for i in range(self.n_latents)] # get the indices of all dimensions
        
        survival_times = self.clin_data_df[self.survival_column].values # get the survival times

        q = self.mcvae_fit.encode(self.data_mcvae_all) # get the latent representations

        latent_vars_ch0 = q[0].loc.detach() # get the latent representations of the first channel
        latent_vars_ch1 = q[1].loc.detach() # get the latent representations of the second channel
        latents = []

        n_dims = latent_vars_ch0.shape[1] # get the number of dimensions

        for i in range(n_dims): # average the latent representations of the two channels
            latent_temp = np.vstack([latent_vars_ch0[:, i], latent_vars_ch1[:, i]])
            latents.append(np.mean(latent_temp, axis=0))

        latents = [latents[i] for i in self.indices] # only keep the dimensions that pass the dropout constraints

        mean_latents = np.vstack([latents]).transpose() 

        col_names = [f'$Z_{i}$' for i in self.indices]
        mean_latent_df = pd.DataFrame(mean_latents, columns=col_names) # make a dataframe with the latent representations
        mean_latent_df['survival_time'] = pd.Series(survival_times).astype('category')
        mean_latent_df.head() # 43 people

        sns.pairplot(mean_latent_df, hue='survival_time', corner=True, palette="coolwarm") # plot the latent representations
        plt.suptitle(f"Survival Time - mean of channels' latent dimensions")
        plt.tight_layout()
        plt.savefig(path_name, dpi=180)

        self.mean_latent_df = mean_latent_df

    

def early_stopping_tol(patience, tolerance, loss_logs, verbose=True):
    """
    Early stopping with tolerance for the loss to be considered as not improving (not decreasing).
    
    Args:
        patience (int): number of epochs to wait before stopping.
        tolerance (float): tolerance for the loss to be considered as not improving.
        loss_logs (list): list of losses.
        verbose (bool, optional): print the epoch chosen. Defaults to True.
        
    Returns:
        int: epoch chosen.
    """
    last_loss = -2000 # arbitrary value
    triggertimes = 0 # number of times the loss has not improved
    done = 0 # whether we have found the epoch chosen

    for i in range(len(loss_logs)):
        current_loss = loss_logs[i] 

        if abs(current_loss - last_loss) < tolerance: 
            # if the loss has not improved
            triggertimes += 1 # increase the number of times the loss has not improved

            if triggertimes >= patience: # if the loss has not improved for the number of epochs we are waiting
                if verbose:
                    print(f'Epoch chosen after early stopping with patience {patience} \
                    and tolerance {tolerance} : {i}.')
                done = 1
                break

        else:
            triggertimes = 0

        last_loss = current_loss
        
    if done == 0:
        if verbose: print("No epoch chosen with this patience.")
        
    return i 


def set_fontsizes():

    matplotlib.rc('font', size=15)          # controls default text sizes
    matplotlib.rc('axes', titlesize=15)     # fontsize of the axes title
    matplotlib.rc('axes', labelsize=15)    # fontsize of the x and y labels
    matplotlib.rc('xtick', labelsize=13)    # fontsize of the tick labels
    matplotlib.rc('ytick', labelsize=13)    # fontsize of the tick labels
    matplotlib.rc('legend', fontsize=13)    # legend fontsize
    matplotlib.rc('figure', titlesize=15)  # fontsize of the figure title





