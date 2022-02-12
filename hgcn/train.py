from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import sklearn

import optuna

import numpy as np
import optimizers
import torch
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics


def train(args):
    #np.random.seed(args.seed)
    #torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    #if int(args.cuda) >= 0:
    #    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else  int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join("logs/", args.task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])

    logging.info(f'Using: {args.device}')
    #logging.info("Using seed {}.".format(args.seed))

    # Load data
    data = load_data(args, os.path.join("data/", args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape
    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel
        else:
            Model = RECModel
            # No validation for reconstruction task
            args.eval_freq = args.epochs + 1

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    model = Model(args)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())   
    #print(pytorch_total_params)
    #return
    
    
    logging.info(str(model))
    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0 :
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        #model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x]
    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        train_metrics = model.compute_metrics(embeddings, data, 'train')
        train_metrics['loss'].backward()
        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        lr_scheduler.step()
        tm = train_metrics.copy()
        del tm["conf_mat"]
        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   'lr: {}'.format(lr_scheduler.get_lr()[0]),
                                   format_metrics(tm, 'train'),
                                   'time: {:.4f}s'.format(time.time() - t)
                                   ]))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['features'], data['adj_train_norm'])
            val_metrics = model.compute_metrics(embeddings, data, 'val')
            
            vm = val_metrics.copy()
            del vm["conf_mat"]
            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(vm, 'val')]))
            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                best_emb = embeddings.cpu()
                if args.save:
                    np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter == args.patience and epoch > args.min_epochs:
                    logging.info("Early stopping")
                    break

    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['features'], data['adj_train_norm'])
        best_test_metrics = model.compute_metrics(best_emb, data, 'test')
    bvm = best_val_metrics.copy()
    del bvm["conf_mat"]
    btm = best_test_metrics.copy()
    del btm["conf_mat"]
    logging.info(" ".join(["Val set results:", format_metrics(bvm, 'val')]))
    logging.info(" ".join(["Test set results:", format_metrics(btm, 'test')]))
    if args.save:
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)

        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        logging.info(f"Saved model in {save_dir}")

    if not os.path.exists('conf_mats'):
    	os.makedirs('conf_mats')

	
    timestr = time.strftime("%Y%m%d-%H%M%S")

    fname = "conf_mats/"+args.model+"_"+timestr+".npy"

    np.save(fname,best_test_metrics["conf_mat"])

    print(best_test_metrics["conf_mat"])
        
    return best_val_metrics["loss"]
    

if __name__ == '__main__':
    args = parser.parse_args()
    def objective(trial):
        params = {
	    "dim": trial.suggest_int("hidden_dim", 128, 1024),
	    "dropout": trial.suggest_float("dropout", 0.1, 0.7, step=0.1),
	    "lr": trial.suggest_loguniform("learning_rate", 1e-6, 1e-3),
	    "cb-beta": trial.suggest_categorical("cb-beta", [0.999,0.9999,0.99999,0.999999]),
	    "cb-gamma": trial.suggest_categorical("cb-gamma",[2.0,2.5,3.0,3.5,4.0]),
	    "weight-decay": trial.suggest_loguniform("weight-decay", 1e-6, 1e-3),
        }
        args.dim = params["dim"]
        args.dropout = params["dropout"]
        args.lr = params["lr"]
        args.cb_beta = params["cb-beta"]
        args.cb_gamma = params["cb-gamma"]
        args.weight_decay = params["weight-decay"]
        return train(args)
    
    #study = optuna.create_study(direction="minimize")
    #study.optimize(objective, n_trials=20)
    train(args)