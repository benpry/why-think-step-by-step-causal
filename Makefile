.PHONY: results, results_no_free, eval_results

data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl: code/make_train_data/generate_bayes_nets.py
	python code/make_train_data/generate_bayes_nets.py \
		--n_nets $(N_NETS) \
		--n_nodes $(N_NODES) \
		--n_edges $(N_EDGES)

data/evaluation/causal/true-probs/true-probabilities-net-$(NET_ID).csv: code/evaluate/true_conditional_probs.py data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl
	python code/evaluate/true_conditional_probs.py \
		--net_idx $(NET_ID) \
		--bayes-net-file data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl

data/training-data/causal/selected-pairs/selected-pairs-net-$(NET_ID).csv: data/evaluation/causal/true-probs/casual-true-probabilities-net-$(NET_ID).csv code/make_train_data/select_pairs_to_hold_out.py
	python3 code/make_train_data/select_pairs_to_hold_out.py \
		--net_idx $(NET_ID) \
		--n_pairs $(NUM_PAIRS) \
		--causal

data/scaffolds/causal/scaffolds-net-$(NET_ID).csv: code/scaffold/generate_scaffolds.py data/evaluation/true-probs/true-probabilities-net-$(NET_ID).csv data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl
	python code/scaffold/generate_scaffolds.py \
		--net-idx $(NET_ID) \
		--num-scaffolds $(NUM_PAIRS) \
		--bayes-net-file data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl \
		--causal

data/scaffolds/causal/negative-scaffolds-net-$(NET_ID).csv: code/scaffold/generate_scaffolds.py code/scaffold/generate_negative_scaffolds.py data/evaluation/true-probs/true-probabilities-net-$(NET_ID).csv data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl
	python code/scaffold/generate_negative_scaffolds.py \
		--net-idx $(NET_ID) \
		--num-scaffolds $(NUM_PAIRS) \
		--bayes-net-file data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl \
		--causal

data/training-data/samples/causal_train_samples_$(SAMPLE_FORMAT_STR)_net_$(NET_ID).csv: data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl # data/training-data/selected-pairs/selected-pairs-net-$(NET_ID).csv
	cd code/make_train_data && python generate_training_set.py \
		-n $(N_TRAIN) \
		--sample-format $(SAMPLE_FORMAT) \
		--sample-format-str $(SAMPLE_FORMAT_STR) \
		--net-id $(NET_ID) \
		--exp-p $(EXP_P) \
		--zipf-k $(ZIPF_K) \
		--bayes-net-file data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl \
		--causal \
		--no_eval

$(MODEL_ROOT_FOLDER)/causal-$(MODEL_NAME)/pytorch_model.bin: data/training-data/samples/causal_train_samples_$(SAMPLE_FORMAT_STR)_net_$(NET_ID).csv
	python code/finetune/run_clm.py \
		--model_name_or_path $(BASE_MODEL_PATH) \
		--train_file data/training-data/samples/causal_train_samples_$(SAMPLE_FORMAT_STR)_net_$(NET_ID).csv \
		--per_device_train_batch_size 3 \
		--per_device_eval_batch_size 3 \
		--save_total_limit $(TOTAL_CHECKPOINTS) \
		--save_steps $(CHECKPOINT_INTERVAL) \
		--do_train \
		--num_train_epochs $(N_EPOCHS) \
		--max_steps $(N_TRAIN_STEPS) \
		--output_dir $(MODEL_ROOT_FOLDER)/causal-$(MODEL_NAME)

data/evaluation/causal/base-model-$(BASE_MODEL_NAME)/fixed-gen-probabilities-$(MODEL_NAME).csv: $(MODEL_ROOT_FOLDER)/causal-$(MODEL_NAME)/pytorch_model.bin code/evaluate/fixed_generation_probabilities.py
	python code/evaluate/fixed_generation_probabilities.py \
		--model_folder $(MODEL_ROOT_FOLDER)/causal-$(MODEL_NAME) \
		--base_model_name $(BASE_MODEL_NAME) \
		--net_idx $(NET_ID) \
		--device "cuda:0" \
		--bayes-net-file data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl


data/evaluation/causal/base-model-$(BASE_MODEL_NAME)/free-gen-probabilities-$(MODEL_NAME)-$(NUM_SAMPLES)samples.csv: $(MODEL_ROOT_FOLDER)/causal-$(MODEL_NAME)/pytorch_model.bin code/evaluate/free_generation_probabilities.py
	python code/evaluate/free_generation_probabilities.py \
		--model_folder $(MODEL_ROOT_FOLDER)/causal-$(MODEL_NAME) \
		--base_model_name $(BASE_MODEL_NAME) \
		--net_idx $(NET_ID) \
		--device "cuda:0" \
		--num_samples $(NUM_SAMPLES) \
		--bayes-net-file data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl

data/evaluation/causal/base-model-$(BASE_MODEL_NAME)/scaffolded-gen-probabilities-$(MODEL_NAME)-$(NUM_SAMPLES)samples.csv: $(MODEL_ROOT_FOLDER)/causal-$(MODEL_NAME)/pytorch_model.bin data/scaffolds/scaffolds-net-$(NET_ID).csv code/evaluate/scaffolded_generation_probabilities.py data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl
	python code/evaluate/scaffolded_generation_probabilities.py \
		--model_folder $(MODEL_ROOT_FOLDER)/causal-$(MODEL_NAME) \
		--base_model_name $(BASE_MODEL_NAME) \
		--scaffold_file data/scaffolds/scaffolds-$(MODEL_NAME).json \
		--net_idx $(NET_ID) \
		--device "cuda:0" \
		--num_samples $(NUM_SAMPLES) \
		--bayes-net-file data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl

data/evaluation/causal/base-model-$(BASE_MODEL_NAME)/negative-scaffolded-gen-probabilities-$(MODEL_NAME)-$(NUM_SAMPLES)samples.csv: $(MODEL_ROOT_FOLDER)/causal-$(MODEL_NAME)/pytorch_model.bin data/scaffolds/negative-scaffolds-net-$(NET_ID).csv code/evaluate/scaffolded_generation_probabilities.py data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl
	python code/evaluate/scaffolded_generation_probabilities.py \
		--model_folder $(MODEL_ROOT_FOLDER)/causal-$(MODEL_NAME) \
		--base_model_name $(BASE_MODEL_NAME) \
		--scaffold_file data/scaffolds/scaffolds-$(MODEL_NAME).json \
		--net_idx $(NET_ID) \
		--device "cuda:0" \
		--num_samples $(NUM_SAMPLES) \
		--bayes-net-file data/bayes_nets/nets_n-$(N_NETS)_nodes-$(N_NODES)_edges-$(N_EDGES).pkl \
		--negative

results: data/evaluation/causal/true-probs/true-probabilities-net-$(NET_ID).csv \
 data/evaluation/causal/base-model-$(BASE_MODEL_NAME)/fixed-gen-probabilities-$(MODEL_NAME).csv \
 data/evaluation/causal/base-model-$(BASE_MODEL_NAME)/free-gen-probabilities-$(MODEL_NAME)-$(NUM_SAMPLES)samples.csv \
 data/evaluation/causal/base-model-$(BASE_MODEL_NAME)/scaffolded-gen-probabilities-$(MODEL_NAME)-$(NUM_SAMPLES)samples.csv \
 data/evaluation/base-model-$(BASE_MODEL_NAME)/negative-scaffolded-gen-probabilities-$(MODEL_NAME)-$(NUM_SAMPLES)samples.csv \
 $(MODEL_ROOT_FOLDER)/causal-$(MODEL_NAME)/pytorch_model.bin
