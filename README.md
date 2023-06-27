# The code of the following Paper:
# Federated Knowledge Graph Completion with the Client-Wise Relation Graph

> Abstract: Federated knowledge graphs (FKG) refer to a set of related knowledge graphs stored decentrally in multiple clients. The
FKG completion (FKGC) aims to complete each KG by exploiting local triples and information from other KGs without exposing triples
to each other. Existing embedding learning-based FKGC methods usually averagely aggregate local entity embeddings from all clients
on the server, and then use it to update local embeddings for each client. They often overlook the client-wise relation while aggregating
entity embeddings on the server, affecting the quality of learned embeddings. To tackle this problem, we propose Federated Knowledge
Graph Completion with the Client-Wise Relation Graph (FedG), in which the server leverages the client-wise relation graph to personalize
the aggregation of entity embeddings from other clients for each client. Besides, we propose an embedding ensemble method which
combines the abilities of both personalized and average aggregated entity embeddings to further enhance the performance of FedG. We
conduct extensive experiments on three benchmark datasets to compare our method with the state-of-the-art models and the results show
the effectiveness of our method.


### Installation

Install PyTorch following the instructions on the [PyTorch](https:pytorch.org/).
Our code is written in Python 3.

numpy==1.23.5
dgl-cu117==0.9.1.post1
dglgo==0.0.2
scikit-learn==1.1.3
scipy==1.9.3
torch==1.13.0


### Running the code, 

We take the dataset FB15k237-Fed5.pkl as an example.

#Case 1: When KGE method is R-GCN, the followings are the commands to run the code:
    
    (1) Conducting embedding learning in the setting of FedG algorithm.
        python main.py --data_path ./data/FB15k237-Fed5.pkl --name fb15k237_fed5_rgcn_fedg --setting FedG --mode train --ratio 5 --model TransE --rgcn 1
     
    (2) Conducting embedding learning in the setting of C-FedG algorithm.
        python main.py --data_path ./data/FB15k237-Fed5.pkl --name fb15k237_fed5_rgcn_cfedg --setting CFedG --mode train --ratio 5 --model TransE --rgcn 1
    
    (3) Conducting embedding ensemble with the two kinds of embeddings learned from (1) and (2), respectively.
        python main.py --data_path ./data/FB15k237-Fed5.pkl --name fb15k237_fed5_embedding_ensemble_rgcn
              --setting Model_Fusion --mode train --fusion_method adapt --model TransE --rgcn 1
              --fusion_fedg fb15k237_fed5_rgcn_fedg.best --fusion_cfedg fb15k237_fed5_rgcn_cfedg.best

>Note: Note: The parameters --fusion_fedg and --fusion_cfedg are the learned embeddings from (1) and (2). Their values correspond to parameters --name in (1) and (2), respectively, excepting adding ".best" .

#Case 2: When KGE method is TransE, the followings are the commands to run the code:
    
    (1) Conducting embedding learning in the setting of FedG algorithm.
        python main.py --data_path ./data/FB15k237-Fed5.pkl --name fb15k237_fed5_transe_fedg --setting FedG --mode train --ratio 5 --model TransE
    
    (2) Conducting embedding learning in the setting of C-FedG algorithm.
        python main.py --data_path ./data/FB15k237-Fed5.pkl --name fb15k237_fed5_transe_cfedg --setting CFedG --mode train --ratio 5 --model TransE

    (3) Conducting embedding ensemble with the two kinds of embeddings learned from (1) and (2), respectively.
        python main.py --data_path ./data/FB15k237-Fed5.pkl --name fb15k237_fed5_embedding_ensemble_transe
              --setting Model_Fusion --mode train --fusion_method adapt --model TransE
              --fusion_fedg fb15k237_fed5_transe_fedg.best --fusion_cfedg fb15k237_fed5_transe_cfedg.best

#Case 3: When KGE method is RotatE, the commands to run the code is similar to the Case 2, excepting changing the parameter --model into TransE


### For different datasets and KGE methods, the parameter --ratio is as follows:

| Dataset         | KGE    | FedG  | C-FedG  |
|-----------------|--------|-------|---------|
|                 | TransE | 5     | 5       |
| FB15k-237-Fed5  | RotatE | 5     | 3       |
|                 | R-GCN  | 3     | 3       |
| --------------  | ------ | ----- | ------- |
|                 | TransE | 5     | 5       |
| FB15k-237-Fed5  | RotatE | 5     | 5       |
|                 | R-GCN  | 3     | 3       |
| --------------  | ------ | ----- | ------- |
|                 | TransE | 5     | 5       |
| FB15k-237-Fed10 | RotatE | 5     | 3       |
|                 | R-GCN  | 3     | 3       |

### For the other hyperparameters, we follow the default parameters in the Main.py file. Besides, for fair comparison, we set the same values for common hyperparameters of all competitors(FedE, FedEC).







