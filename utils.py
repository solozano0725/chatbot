import pickle

#Creation of pickle files to store all Python objects which we'll use to the prediction process
def create_pickle(list, pkl_url): 
    return pickle.dump(list, open(pkl_url,'wb'))


def load_pickle(pkl_url):
    return pickle.load(open(pkl_url,'rb'))