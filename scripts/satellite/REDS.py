import numpy as np
import scipy.stats as stat
import scipy.optimize as optimize
from sklearn import linear_model
import time

#Some hyperparameters given by REDS-paper:
R = 100 # number of models kept in the ensemble
m_1 = 0.1 #minibatch factor, for sample size n, the minibatch has size floor(n*m_1) = floor(n/10)
#lambdas = [1e-2, 1e-3, 1e-5]
widths = np.arange(500, 1001, 1)


def ReLU(x):  #For neuron activation
    return x * (x > 0)

def REDS_RFFs(): 
    """

    RFF sampling method accordig to REDS paper

    Returns
    ---
    Omega : array
        array of (2D) frequencies
    """

    J1 = 500  #Number of features for each radius. 
    J2 = 300  #Given in paper
    J3 = 200

    r1 = 0.1 #Also given i paper
    r2 = 0.2
    r3 = 0.3

    I = np.array([[1,0], [0,1]])
    zero_vector = np.array([0,0])

    Omega = np.zeros((J1+J2 +J3, 2))
    Omega[0:J1,:] = (1/r1) * stat.multivariate_normal.rvs(mean= zero_vector, cov = I, size = J1)
    Omega[J1:J1 + J2, :] = (1/r2) * stat.multivariate_normal.rvs(mean= zero_vector, cov = I, size = J2)
    Omega[J1 + J2:J1 + J2 + J3, :] = (1/r3) * stat.multivariate_normal.rvs(mean= zero_vector, cov = I, size = J3)

    print("The first frequencies looks like this:")
    print(Omega[0:5, :])

    return Omega

def modified_RFFs():
    """

    Modified RFF sampling method, to give better performance

    Returns
    ---
    Omega : array
        array of (2D) frequencies
    """

    J1 = 500  #Number of features for each radius. 
    J2 = 300  #Given in paper
    J3 = 200

    r1 = 0.5 #Also given i paper
    r2 = 1
    r3 = 1.5

    I = np.array([[1,0], [0,1]])
    zero_vector = np.array([0,0])

    Omega = np.zeros((J1+J2 +J3, 2))
    Omega[0:J1,:] = (1/r1) * stat.multivariate_normal.rvs(mean= zero_vector, cov = I, size = J1)
    Omega[J1:J1 + J2, :] = (1/r2) * stat.multivariate_normal.rvs(mean= zero_vector, cov = I, size = J2)
    Omega[J1 + J2:J1 + J2 + J3, :] = (1/r3) * stat.multivariate_normal.rvs(mean= zero_vector, cov = I, size = J3)

    print("The first frequencies looks like this:")
    print(Omega[0:5, :])

def ELM_hidden_layer(X, d):
    """
    Wrapper function
    one layer of an Extreme learning machine
    Weights are sampeled uniformly on [-0.1, 0.1]
    ReLU activation

    Parameters
    ---
    X : array 
        design matrix (or output from a previous layer)
    d : int
        width of hidden layer

    Returns
    ---
    A : array
        The output neuron activations for this layer   
    """

    (n, m) = np.shape(X)
    weights = stat.uniform.rvs(loc = -0.1, scale = 0.2, size = m * d)
    weights_matrix = np.reshape(weights, (m, d))

    return ReLU(X @ weights_matrix)

def apply_RFF(Omega, S):
    """
    Wrapper function.
    Transforms the data according to the random fourier features

    Parameters
    ---
    Omega : array
        RFFs
    S : array 
        locations
    
    Returns
    ---
    RFF_design_matrix : array 
        sin(w_j * s_i) or cos(w_j * s_i) for data point i and RFF j
    """
    (n, dim) = np.shape(S)
    (n_RFF, dim_RFF) = np.shape(Omega)
    if(dim != dim_RFF):
        print("RFF and sppatial locations don't match!")
    
    RFF_design_matrix = np.zeros((n, 2 * n_RFF))
    for i in range(n):
        for j in range(n_RFF):
            RFF_design_matrix[i, 2 * j] = np.cos(np.dot(S[i], Omega[j]))
            RFF_design_matrix[i, 2 * j + 1] = np.sin(np.dot(S[i], Omega[j]))

    print(f"The RFF design matrix has dimentions {np.shape(RFF_design_matrix)}")

    return RFF_design_matrix

def REDS(S, Y, S_test, RFF_method, N = 500, R = 100):
    """
    Generates ensamble predictions at locations S and S_test

    Parameters
    ---
    S : 2D array
        training locations design matrix (indexed as [n, x_i])
    Y : 1D array 
        training data
    S_test : 2D array
        prediction locations design matrix (indexed as [n, x_i])
    RFF_method : function
        draws RFFs accordig to some distribution
    N : int
        ensemble size
    R : int
        good models of the ensemble

    Returns
    ---
    Y_hat : 1D array
        predictons at all locatins. n first are at trainin locations
    Q1 : 1D array
        1st quartile of predictions made by the good models of the ensemble
    Q3 : 1D array
        3rd quartile
    n : int
        number of training samples
    """

    #Drawing the fourier features
    print("Draws frequencies using " + RFF_method.__name__)
    Omega = RFF_method()

    #Deriving the training and test sample sizes, number of RFFs
    n = np.shape(S)[0]  #training sample size
    n_tilde = np.shape(S_test)[0]  #sest sample size

    #Minibatch size, given by factor m_1 in paper
    minib_size = int(m_1 *n)

    #Apply RFFs to all the data
    print("applying random frequencies to obtain RFFs")
    RFF_design_matrix = apply_RFF(Omega, np.concatenate((S, S_test), axis = 0))
    
    #Where the ELM outputs will be stored for each model in the ensemble
    Ensemble_pred = np.zeros((n_tilde + n, R))
    Ensemble_train_perf = np.zeros((1, R))

    Worst_RMSE = 0
    
    print("Training ELMs")

    for i in range(N):

        K = stat.randint.rvs(1,4)  #Hidden layers in ELM
        d = widths[stat.randint.rvs(0,np.size(widths))]   #Width of each hidden layer
        lam = stat.uniform.rvs(loc = 0.01, scale = 0.04) #Lasso penalty
        if i % 11 == 3:
            print(f"{100 * i/500} % done training models.")
            print(f"model currently being trained has lambda = {lam}, {K} hidden layers with width {d}")

        Design_matrix = RFF_design_matrix

        for j in range(K):
            Design_matrix = ELM_hidden_layer(Design_matrix, d)
        
        #draw a minibatch from training data
        minibatch_rows = np.random.choice(n, minib_size, replace=False)
        minibatch_train = Design_matrix[minibatch_rows, :]
        minibatch_response = Y[minibatch_rows]

        #transformed data split back into train set
        transf_train = Design_matrix[:n, :]

        #Fits the lasso regression to the minibatch
        Lasso_reg = linear_model.Lasso(alpha=lam, max_iter=1000000, tol=1e-3, warm_start=True, selection='random')
        Lasso_reg.fit(minibatch_train, minibatch_response)

        #training RMSE
        RMSE = np.sqrt(np.mean((Lasso_reg.predict(transf_train) - Y)**2))
        

        #making predictions at all locations
        #We keep only the best R models so far
        if i < R:
            Ensemble_pred[:, i] = Lasso_reg.predict(Design_matrix)
            Worst_RMSE = max(Worst_RMSE, RMSE)
            Ensemble_train_perf[0, i] = RMSE
        else:
            if RMSE < Worst_RMSE:
                replace_idx = np.argmax(Ensemble_train_perf, axis=1)
                Ensemble_pred[:, replace_idx] = Lasso_reg.predict(Design_matrix)[:, np.newaxis]
                Ensemble_train_perf[0, replace_idx] = RMSE
                Worst_RMSE = np.max(Ensemble_train_perf, axis=1)



    #We keep only the R best models of the ensemble according to the RMSE

    cutoff = Worst_RMSE
    good_pred = Ensemble_pred

    print(f"{N} models trained. Selecting the {R} best models. these has training RMSE <= {cutoff}")
    print(f"Shape of good predictions: {np.shape(good_pred)}, eqhal to n + n_tilde: {n+n_tilde}, R: {R}")

    #Final prediction is the median of the good models of the ensemble
    #predictions at training locations are at indexes 0:n (excluding)
    #test location predictions are at indexes n:(n + n_tilde)
    Y_hat = np.median(good_pred, axis= 1)

    #Uncertainty measured as IQR = Q_3 - Q_1 (quartiles) of the good models in the ensemble
    Q1 = np.quantile(good_pred, 0.25, axis = 1) 
    Q3 = np.quantile(good_pred, 0.75, axis = 1)

    return Y_hat, Q1, Q3, n, cutoff

def REDS_UQ(Y_hat, Q1, Q3, n, Y, alpha):
    """
    Uncertainty Quantifier 

    Parameters
    ---
    Y_hat : array
        predictions made by REDS_ensemble
    Q1 : array
        first (lower) quartile of predictions in ensemble
    Q3 : array
        third (higher) quartile
    n : int
        training sample size. the first n indexes of Y_hat are at training locations
    Y : array
        training data
    alpha : float in [0,1]
        confidence level
    
    Returns
    ---
    L : array
        lower bound for the alpha-confidence interval at all locations
    U : array
        upper bound
    """

    #Separates the test data
    Y_hat_train = Y_hat[:n]
    #Uncertainty is sigma * v, this algorithm solves for v
    sigma = Q3[:n] - Q1[:n]

    #Difference between amount of points within the interval and level of significance. We want this to be 0
    def coverage(v):
        L = Y_hat_train - v * sigma
        U = Y_hat_train + v * sigma
        covered = np.logical_and(L < Y,U > Y)

        return np.mean(covered) - alpha
    
    #Solving coverage(v) = 0
    print(f"Solving for the cutoff v that gives a {alpha* 100}% confidence interval")
    v_hat = optimize.brentq(coverage, 0, 500, maxiter = 1000)

    #Computes the interval
    sigma_full = Q3 - Q1
    L = Y_hat - v_hat * sigma_full
    U = Y_hat + v_hat * sigma_full

    return L, U

def test_REDS(run_number, RFF_method):
        
    test_data = np.genfromtxt("data/satellite/test.csv", delimiter = ",")
    train_data = np.genfromtxt("data/satellite/train.csv", delimiter = ",")

    #Selecting spatial coordinates and response
    S = train_data[:, 0:2]
    Y = train_data[:, 2]
    S_test = test_data[:, 0:2]
    Y_test = test_data[:, 2]


    #Run time
    start_time_REDS = time.time()

    #Training and predicting
    Y_hat, Q1, Q3, n, cutoff_RMSE = REDS(S, Y, S_test, RFF_method)

    #End run time
    REDS_run_time = time.time() - start_time_REDS

    #Confidence interval bounds
    L, U = REDS_UQ(Y_hat, Q1, Q3, n, Y, 0.95)

    #Predictions at training locations are the first n indeces,
    #Predictions at test locations are the remaining indeces
    Y_hat_test = Y_hat[n:]
    L_test = L[n:]
    U_test = U[n:]

    #Saves run
    np.savetxt(r"REDS{0}_test_output.csv".format(run_number), np.concatenate((Y_hat_test[:, np.newaxis], L_test[:, np.newaxis], U_test[:, np.newaxis]), axis=1), delimiter=",")
    #Saves run performance statistics
    time_stat = np.array([REDS_run_time])
    np.savetxt(r"REDS{0}_run_time.csv".format(run_number),time_stat, delimiter=",")


for ø in range(0, 10):
    print(f"run: {ø}")
    test_REDS(ø, REDS_RFFs)
