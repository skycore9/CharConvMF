import argparse
import sys
from utils import Data
from char_cnn_mf import CharConvCF
from cnn_model.char_cnn import CNN_Module

parser = argparse.ArgumentParser()

# train CNN model
parser.add_argument('-t', '--train_cnn_module', type=bool,
                    help="train cnn module.", default=False)
parser.add_argument("-c", "--cnn_data_path", type=str,
                    help="Path to document data to train CNN.")
parser.add_argument("-p", '--is_polarity', type=bool,
                    help="whether training cnn model is polarity.", default=True)

# train CharConvMF
parser.add_argument("-r", "--raw_rating_data_path", type=str,
                    help="Path to raw rating data. data format - user id::item id::rating")
parser.add_argument("-i", "--raw_item_document_data_path", type=str,
                    help="Path to raw item document data. item document consists of multiple text. data format - item id::text1|text2...")

# set regularization factor
parser.add_argument("-reg_bu", "--regularization_bu", type=float,
                    help="Regularization factor for bu to avoid overfiting.")
parser.add_argument("-reg_wi", "--regularization_bi", type=float,
                    help="Regularization factor for bi (or wi) to avoid overfiting.")
parser.add_argument("-reg_pu", "--regularization_pu", type=float,
                    help="Regularization factor for pu to avoid overfiting.")
parser.add_argument("-reg_qi", "--regularization_qi", type=float,
                    help="Regularization factor for qi to avoid overfiting.")
parser.add_argument("-reg_yj", "--regularization_yj", type=float,
                    help="Regularization factor for yj to avoid overfiting.")

parser.add_argument("-l", "--learning_rate", type=float,
                    help="learning rate for SGD to train model.")

args = parser.parse_args()
is_train_cnn = args.train_cnn_module
if is_train_cnn == False:
    cnn_data_path = args.cnn_data_path
    is_polarity = args.is_polarity
    if cnn_data_path is None:
        sys.exit("Argument missing - cnn_data_path is required")
    if is_polarity is None:
        sys.exit("Argument missing - is_polarity is required")
    cnn_module = CNN_Module(data_path=cnn_data_path, is_polarity=is_polarity)
    cnn_module.train()
else:
    rating_path = args.raw_rating_data_path
    document_path = args.raw_item_document_data_path
    is_polarity = args.is_polarity

    reg_bu = args.regularization_bu
    reg_bi = args.regularization_bi
    reg_pu = args.regularization_pu
    reg_qi = args.regularization_qi
    reg_yj = args.regularization_yj
    learning_rate = args.learning_rate
    if rating_path is None:
        sys.exit("Argument missing - rating_path is required")
    if document_path is None:
        sys.exit("Argument missing - dcument_path is required")
    if reg_bu is None or reg_bi is None \
            or reg_pu is None or reg_qi is None \
            or reg_yj is None:
        sys.exit("Argument missing - regularization_factor is required")

    data_factory = Data(rating_path=rating_path,
                        document_path=document_path,
                        is_polarity=is_polarity)

    # grid search
    print "##################################################################"
    print "\treg_bu=%f, reg_bi=%f, reg_pu=%f, reg_qi=%f, reg_yj=%f"  \
          % (reg_bu, reg_bi, reg_pu, reg_qi, reg_yj)
    print "##################################################################"
    char_mf = CharConvCF(lr_all=learning_rate,
                         reg_bu=reg_bu,
                         reg_bv=reg_bi,
                         reg_pu=reg_pu,
                         reg_qv=reg_qi,
                         reg_yj=reg_yj)
    char_mf.fit(data=data_factory)

