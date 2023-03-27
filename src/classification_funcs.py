import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, LeaveOneOut, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from statsmodels.stats.contingency_tables import mcnemar


def leave_one_out_run_onemethod(X, y, model_name, model, grid_params):
    """
    Leave one out cross validation for a single model and set of parameters on a single dataset (X, y)
    
    Args:
        X (np.array): data matrix
        y (np.array): labels vector
        model_name (str): name of model (for plotting)
        model (sklearn model): model to run 
        grid_params (dict): grid search parameters for model
        
    Returns:
        f1 (float): f1 score 
        acc (float): accuracy
        auc (float): auc
        y_pred (list): predicted labels
        y_true (list): true labels
        y_probs (list): predicted probabilities
            
    """
    cv = LeaveOneOut() # leave one out cross validation

    y_true, y_pred, y_probs = list(), list(), list() # store results

    i = 0 # counter for grid search
    for train_ix, test_ix in cv.split(X):

        X_train, X_test = X[train_ix, :], X[test_ix, :] # get train and test set 
        y_train, y_test = y[train_ix], y[test_ix] # get train and test set of labels

        # gridsearch
        if i == 0:
            gridmodel = clone(model) # clone model to avoid overwriting
        
            # grid search with 5 fold cross validation and all cores
            grid = GridSearchCV(estimator=gridmodel, param_grid=grid_params, cv=5, n_jobs=-1) 
            
            grid.fit(X_train, y_train) # fit grid search

            best_estimator = grid.best_estimator_ # get best estimator

            params = grid.best_params_  # get best parameters

        
        model_fit = clone(best_estimator) # clone best estimator to avoid overwriting

        model_fit.set_params(**(params)) # set parameters    

        model_fit.fit(X_train, y_train) # fit model with best parameters on train set

        # evaluate model
        yhat = model_fit.predict(X_test) # predict on test set
        yhat_probs = model_fit.predict_proba(X_test) # predict probabilities on test set

        # store
        y_true.append(y_test[0])
        y_pred.append(yhat[0])
        y_probs.append(yhat_probs[0][1])
        
        i += 1

    f1 = f1_score(y_true, y_pred)

    acc = accuracy_score(y_true, y_pred)

    auc = roc_auc_score(y_true, y_probs)

    return f1, acc, auc, y_pred, y_true, y_probs
 




def leave_one_out_all_methods(X, y):
    """
    Leave one out cross validation for all models and parameters on a single dataset (X, y) 
    
    Args:
        X (np.array): data
        y (np.array): labels
        
    Returns:
        F1s_dict (dict): f1 scores
        Accs_dict (dict): accuracies
        aucs_dict (dict): aucs
        preds_dict (dict): predicted labels
        expecteds_dict (dict): true labels
        probs_dict (dict): predicted probabilities
    """



    models, model_params = classification_list_updater()
    F1s_dict = dict.fromkeys(models.keys(),)
    Accs_dict = dict.fromkeys(models.keys(),)
    aucs_dict = dict.fromkeys(models.keys(),)
    preds_dict = dict.fromkeys(models.keys(),)
    expecteds_dict = dict.fromkeys(models.keys(),)
    probs_dict = dict.fromkeys(models.keys(),)

    for model_idx in models.keys():
        # run leave one out cross validation for a single model and set of parameters on a single dataset (X, y)
        f1, acc, auc, preds, expecteds, y_probs = leave_one_out_run_onemethod(X=X, y=y, model_name=model_idx, model=models[model_idx],
                                            grid_params=model_params[model_idx]) 
        

        # store results
        F1s_dict[model_idx] = f1
        Accs_dict[model_idx] = acc
        aucs_dict[model_idx] = auc
        preds_dict[model_idx] = preds
        expecteds_dict[model_idx] = expecteds
        probs_dict[model_idx] = y_probs


    return F1s_dict, Accs_dict, aucs_dict, preds_dict, expecteds_dict, probs_dict




def classification_list_updater():
    """
    Update classification models and parameters
    
    Returns:
        classification_functions (dict): classification models
        classification_grid_params (dict): classification parameters
    """

    # classification models
    classification_functions = {
                        # "Random Forest": RandomForestClassifier(random_state=0), 
                        "Support Vector": SVC(probability=True), 
                        #"KNN": KNeighborsClassifier(),
                        # "Logistic Regression": LogisticRegression()
                        }

    # classification parameters for grid search 
    classification_grid_params = {
                            #"KNN": {"n_neighbors": list(range(1,6))},
                            # "Random Forest": {
                            #                         'n_estimators': [50,100,200,500],
                            #                         'max_features': [.2, .3, .4, .5, .6, .7, .8, .9],
                            #                         'bootstrap': [False, True],
                            #                         'max_depth':[2,4,6,8,10,15],

                            #                     }, 
                            "Support Vector": {'kernel' : ('linear', 'rbf'),
                                                'C' : [1,5,10,100,1000],
                                                'gamma' : [0.001,0.01,0.1,0.5,1,3],
                                                'probability': [True]},
                            # "Logistic Regression": {"C":np.logspace(-4,4,20), 
                            #                         "penalty":["none","l2"],
                            #                         "solver":['saga','liblinear']}                    
                                                }

    return classification_functions, classification_grid_params



def prepare_concatenated_data(clindf, bvdf):
    """
    Prepare concatenated data
    
    Args:
        clindf (pd.DataFrame): clinical data
        bvdf (pd.DataFrame): brain volume data
        
    Returns:
        X_both (np.array): concatenated data
        y_both (np.array): labels
    """
    x_clin, y_clin = prepare_clinical_only_data(clindf) # prepare clinical data 
    x_bv, y_bv = prepare_brain_volume_only_data(clindf, bvdf) # prepare brain volume data

    X_both = np.concatenate((x_clin, x_bv),axis=1) # concatenate clinical and brain volume data
    y_both = y_clin # labels are the same for both


    return X_both, y_both



def prepare_clinical_only_data(clindf):
    """
    Prepare clinical data
    
    Args:
        clindf (pd.DataFrame): clinical data
        
    Returns:
        X_clin (np.array): clinical data
        y_clin (np.array): labels
    """
    # drop long_survival_mri column and convert to numpy array
    X_clin = clindf.drop(columns=["long_survival_mri"]).to_numpy().astype('float')

    # convert long_survival_mri column to numpy array
    y_clin = clindf['long_survival_mri'].to_numpy().astype('float')

    # normalize data function
    normalize = lambda _: (_ - _.mean(0)) / _.std(0)

    # normalize clinical data
    X_clin = normalize(X_clin)

    return X_clin, y_clin



def prepare_brain_volume_only_data(clindf, bvdf):
    """
    Prepare brain volume data
    
    Args:
        
        clindf (pd.DataFrame): clinical data
        bvdf (pd.DataFrame): brain volume data
        
    Returns:
        X_bf (np.array): brain volume data
        y_bf (np.array): labels
    """

    # drop long_survival_mri column and convert to numpy array
    X_bf = bvdf.to_numpy().astype('float')
    # convert long_survival_mri column to numpy array
    y_bf = clindf['long_survival_mri'].to_numpy().astype('float')

    normalize = lambda _: (_ - _.mean(0)) / _.std(0)

    X_bf= normalize(X_bf)

    return X_bf, y_bf


def prepare_mcvae_data_for_classification(mcvae_model, path_name):
    """
    Prepare MCVAE data for classification
    
    Args:
        mcvae_model (MCVAE): MCVAE model
        path_name (str): path to MCVAE data
    
    Returns:
        X_mcvae (np.array): MCVAE data
        y_mcvae (np.array): labels
    """
    # get average z values for each patient and store in dataframe
    mcvae_model.get_avg_zs(path_name)

    # convert to numpy array and convert to float 
    X_mcvae = mcvae_model.mean_latent_df.drop(columns='survival_time').to_numpy().astype('float')
    # convert to numpy array and convert to float
    y_mcvae = mcvae_model.mean_latent_df['survival_time'].to_numpy().astype('float')

    return X_mcvae, y_mcvae



def dataframe_results(f1s, accs, aucs, method_name):
    """
    Create dataframe with results
    
    Args:
        f1s (list): list of f1 scores
        accs (list): list of accuracies
        aucs (list): list of AUCs
        method_name (str): name of method
        
    Returns:
        results (pd.DataFrame): dataframe with results
    """
    # create dataframe with results
    results = pd.DataFrame(columns=['AUC', 'Accuracy', 'F1'], index=['Clinical', 'BV', 'Concat', 'MCVAE'])
    # add results to dataframe
    results['AUC'] = [aucs[i][method_name] for i in range(len(aucs))]
    results['Accuracy'] = [accs[i][method_name] for i in range(len(accs))]
    results['F1'] = [f1s[i][method_name] for i in range(len(f1s))]
    return results





def mcnemar_test(preds, expecteds, f1s, return_flag=False):
    """
    McNemar test
    
    Args:
        preds (list): list of predictions
        expecteds (list): list of expected values
        f1s (list): list of f1 scores
        return_flag (bool): flag to return pvalues
    
    Returns:
        pvalues (dict): dictionary with pvalues
        """
    pvalues = {"Random Forest": [], "Support Vector": []} # dictionary to store pvalues

    for method in f1s[0].keys(): # loop through methods
        print(f"######## {method} ########")

        mcvae_preds = np.array(preds[3][method]) # get predictions for MCVAE
        mcvae_expecteds = np.array(expecteds[3][method]) # get expected values for MCVAE

        mcvae_performance = mcvae_preds == mcvae_expecteds # get performance for MCVAE 

        mcvae_false_positions = np.where(mcvae_performance == False) # get false positions for MCVAE
        mcvae_true_positions = np.where(mcvae_performance == True) # get true positions for MCVAE

        for i, datatype in enumerate(['clinical', 'bv', 'concat']): # loop through data types

            comparing_preds = np.array(preds[i][method]) # get predictions for data type
            comparing_expecteds = np.array(expecteds[i][method]) # get expected values for data type

            comp_performance = comparing_preds == comparing_expecteds # get performance for data type

            comp_false_positions = np.where(comp_performance == False) # get false positions for data type
            comp_true_positions = np.where(comp_performance == True) # get true positions for data type

            # get contingency table
            n00 = len(np.intersect1d(comp_false_positions, mcvae_false_positions))
            n01 = len(np.intersect1d(comp_false_positions, mcvae_true_positions))
            n10 = len(np.intersect1d(comp_true_positions, mcvae_false_positions))
            n11 = len(np.intersect1d(comp_true_positions, mcvae_true_positions))

            contin_table = [[n00, n01], 
                            [n10, n11]]

            # run mcnemar test
            test = mcnemar(contin_table, exact=False, correction=True)

            print(f"{datatype} vs mcvae: {np.round(test.pvalue, 4)}")
            pvalues[method].append(np.round(test.pvalue, 4))

    if return_flag==True:
         return pvalues


def run_all_data_inputs_loo(cdf, bdf, latent_space_instance, path_name, sig_test_flag=False):

    """
    Run all data inputs with leave one out
    
    Args:
        cdf (pd.DataFrame): clinical data
        bdf (pd.DataFrame): brain volume data
        latent_space_instance (MCVAE): MCVAE model
        path_name (str): path to MCVAE data
        sig_test_flag (bool): flag to run significance test
    
    Returns:
        f1s (list): list of f1 scores
        accs (list): list of accuracies
        aucs (list): list of AUCs
        preds (list): list of predictions
        expecteds (list): list of expected values
        probs (list): list of probabilities
    """
    print("Clinical only:")
    X_clin, y_clin = prepare_clinical_only_data(cdf)
    f1s_clin, accs_clin, aucs_clin, preds_clin, expecteds_clin, probs_clin = leave_one_out_all_methods(X_clin, y_clin)
    print(f"F1: {f1s_clin}, Accuracy: {accs_clin}, AUC: {aucs_clin}")

    print("Brain volume only:")
    X_bf, y_bf = prepare_brain_volume_only_data(cdf, bdf)
    f1s_bf, accs_bf, aucs_bf, preds_bf, expecteds_bf,probs_bv = leave_one_out_all_methods(X_bf, y_bf)
    print(f"F1: {f1s_bf}, Accuracy: {accs_bf}, AUC: {aucs_bf}")

    print("Concatenated clinical and brain volume:")
    X_concat, y_concat = prepare_concatenated_data(cdf, bdf)
    f1s_concat, accs_concat, aucs_concat, preds_concat, expecteds_concat,probs_concat = leave_one_out_all_methods(X_concat, y_concat)
    print(f"F1: {f1s_concat}, Accuracy: {accs_concat}, AUC: {aucs_concat}")

    print("MCVAE Latent Space:")
    X_mcvae, y_mcvae = prepare_mcvae_data_for_classification(latent_space_instance, path_name)
    f1s_mcvae, accs_mcvae, aucs_mcvae, preds_mcvae, expecteds_mcvae, probs_mcvae = leave_one_out_all_methods(X_mcvae, y_mcvae)
    print(f"F1: {f1s_mcvae}, Accuracy: {accs_mcvae}, AUC: {aucs_mcvae}")

    f1s_all = [f1s_clin, f1s_bf, f1s_concat, f1s_mcvae]
    accs_all = [accs_clin, accs_bf, accs_concat, accs_mcvae]
    aucs_all = [aucs_clin, aucs_bf, aucs_concat, aucs_mcvae]
    preds_all = [preds_clin, preds_bf, preds_concat, preds_mcvae]
    expecteds_all = [expecteds_clin, expecteds_bf, expecteds_concat, expecteds_mcvae]
    probs_all = [probs_clin, probs_bv, probs_concat, probs_mcvae]

    svm_results = dataframe_results(f1s_all, accs_all, aucs_all, "Support Vector")

    # run significance test
    if sig_test_flag: 
        pvalues = mcnemar_test(preds_all, expecteds_all, f1s_all, return_flag=True)['Support Vector']
        pvalues.append('n/a')
        svm_results.loc[:,'pvalues'] = pvalues

    return svm_results, [f1s_all, accs_all, aucs_all, preds_all, expecteds_all, probs_all]



def loo_final_result(cdf, bdf, latent_space_instance, method_name):
    """
    Run all data inputs with leave one out
    
    Args:
        cdf (pd.DataFrame): clinical data
        bdf (pd.DataFrame): brain volume data
        latent_space_instance (MCVAE): MCVAE model
        method_name (str): name of method to run
    
    Returns:
        f1s (list): list of f1 scores
        accs (list): list of accuracies
        aucs (list): list of AUCs
        preds (list): list of predictions
        expecteds (list): list of expected values
        probs (list): list of probabilities
    """
    # clinical
    X_clin, y_clin = prepare_clinical_only_data(cdf)
    f1s_clin, accs_clin, aucs_clin, preds_clin, expecteds_clin = leave_one_out_all_methods(X_clin, y_clin)
    
    # BV
    X_bf, y_bf = prepare_brain_volume_only_data(cdf, bdf)
    f1s_bf, accs_bf, aucs_bf, preds_bf, expecteds_bf = leave_one_out_all_methods(X_bf, y_bf)
    
    # Concatenated
    X_concat, y_concat = prepare_concatenated_data(cdf, bdf)
    f1s_concat, accs_concat, aucs_concat, preds_concat, expecteds_concat = leave_one_out_all_methods(X_concat, y_concat)

    # MCVAE
    X_mcvae, y_mcvae = prepare_mcvae_data_for_classification(latent_space_instance)
    f1s_mcvae, accs_mcvae, aucs_mcvae, preds_mcvae, expecteds_mcvae = leave_one_out_all_methods(X_mcvae, y_mcvae)
    
    f1s_all = [f1s_clin, f1s_bf, f1s_concat, f1s_mcvae]
    accs_all = [accs_clin, accs_bf, accs_concat, accs_mcvae]
    aucs_all = [aucs_clin, aucs_bf, aucs_concat, aucs_mcvae]
    preds_all = [preds_clin, preds_bf, preds_concat, preds_mcvae]
    expecteds_all = [expecteds_clin, expecteds_bf, expecteds_concat, expecteds_mcvae]

    svm_results = dataframe_results(f1s_all, accs_all, aucs_all, "Support Vector")
    rf_results = dataframe_results(f1s_all, accs_all, aucs_all, "Random Forest")

    pvalues = mcnemar_test(preds_all, expecteds_all, f1s_all, return_flag=True)

    if method_name=="Random Forest":
        return rf_results, pvalues[method_name]
    elif method_name=="Support Vector":
        return svm_results, pvalues[method_name]




def make_boxplot(method_name, metric_name, metrics):
    """
    Make boxplot of metrics
    
    Args:
        method_name (str): name of method
        metric_name (str): name of metric
        metrics (list): list of metrics
    
    Returns:
        None
            """
    
    data = [metrics[i][method_name] for i in range(len(metrics))]

    fig = plt.figure(figsize =(10, 7))

    ax = fig.add_axes([0, 0, 1, 1])

    bp = ax.boxplot(data)

    ax.xaxis.set_ticks([1,2,3,4])
    ax.set_xticklabels(['Clinical', 'BV',
                        'Concatenated', 'MCVAE'])
    ax.get_xaxis().tick_bottom()
    ax.set_ylabel(metric_name)
    ax.set_title(f"{method_name}: {metric_name} after {len(data[0])} tests")

    plt.show()



