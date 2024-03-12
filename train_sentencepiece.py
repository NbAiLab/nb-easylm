import sentencepiece as spm

spm.SentencePieceTrainer.train(input="/researchdisk/training_dataset_sentences/train.txt", model_prefix="tokenizer",
                                model_type="bpe", split_digits=True, vocab_size=64256, byte_fallback=True,
                                normalization_rule_name="nfkc",
                                user_defined_symbols=["[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"],
                                required_chars="abcdefghijklmnopqrstuvwxyzåäöABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖ",
                                train_extremely_large_corpus=True,
                                input_sentence_size=500000000, shuffle_input_sentence=True,
                                num_threads=96)