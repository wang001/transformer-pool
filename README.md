# transformer-pool
Imporve the Transformer by 1dPool. 用一维池化改进Transformer，附带在LCSTS数据集上的实验效果。

因为语言大多是词组结构，所以自然的想到在Transformer的attention中增加一个池化让相邻的字词具有相同的注意力。
由于MaxPool实现简单，此处使用MaxPool，而且经过简单实验AvgPool效果和MaxPool类似。

## How to modify
I used OpenNMT-py-0.9.1 and changed MultiHeadedAttention.forward() which in file multi_head_attn.py


```
        if self.max_relative_positions > 0 and type == "self":
            scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key
        scores = scores.float()
        ########################### insert our code begin ############################### 
        if mask is not None:  # mask第一次，防止影响max池化结果
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)
        # 测试测试
        head_count_1 = head_count // 3
        head_count_2 = head_count_1 * 2
        # head_count_3 = head_count_1 * 2 + 1
        if key_len >= 2:
            scores[:, :head_count_1, :, :-1] = torch.nn.MaxPool1d(2, stride=1, padding=0)(
                scores[:, :head_count_1, :, :].contiguous().view(batch_size * head_count_1, query_len, -1)).view(
                batch_size, head_count_1, query_len, -1)
        scores[:, head_count_1:head_count_2, :, :] = torch.nn.MaxPool1d(3, stride=1, padding=1)(
            scores[:, head_count_1:head_count_2, :, :].contiguous().view(batch_size * (head_count_2-head_count_1), query_len,
                                                                             -1)).view(batch_size, (head_count_2-head_count_1),
                                                                                       query_len, -1)
        scores = scores.contiguous()
        ########################### insert our code end ################################# 
        if mask is not None:  # mask第二次，对pad的池化导致不为-1e18部分进行清空
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)
        # batch, head_count, query_len, dim_per_head
        context_original = torch.matmul(drop_attn, value)

```

## Result
### char level
Dataset LCSTS(LCSTS: A Large-Scale Chinese Short Text Summarization Dataset)

data from [SDLM-pytorch/Headline_Generation](https://github.com/thunlp/SDLM-pytorch/tree/master/Headline_Generation/OpenNMT-py "LCSTS dataset") or click [here](https://thunlp.oss-cn-qingdao.aliyuncs.com/LCSTS_split_2393662.zip). 

500K examples, 40K steps：

|Models Name | ROUGE-1 | ROUGE-2 | ROUGE-L |
|- | :-: | :-: | :-: |
|OpenNMT-py | 26.9 | 13.9 | 24.3 |
|our | 29.7 | 18.2 | 27.6|

train curve:
![train curve](https://github.com/wang001/transformer-pool/raw/master/pic/lcsts_char/40k_train_pic.png)

valid curve:
![valid curve](https://github.com/wang001/transformer-pool/raw/master/pic/lcsts_char/40k_valid_pic.png)

2.4M examples, 120K steps：

|Model Name | ROUGE-1 | ROUGE-2 | ROUGE-L |
|- | :-: | :-: | :-: |
|OpenNMT-py | 31.5 | 18.6 | 28.9 |
|our | 32.1 | 19.7 | 29.5|

train curve:
![train curve](https://github.com/wang001/transformer-pool/raw/master/pic/lcsts_char/120k_train_pic.png)

valid curve:
![valid curve](https://github.com/wang001/transformer-pool/raw/master/pic/lcsts_char/120k_valid_pic.png)

```
python train.py -data ./datadir/char_shard100000 -save_model ./savedir/transformer1103 -log_file ./logdir/transformer_log_1103 -seed 12345 -layers 1 -heads 16 -word_vec_size 512 -rnn_size 512 -optim adam -encoder_type transformer -decoder_type transformer -position_encoding -learning_rate 0.001 -batch_size 64 -dropout 0.15 -train_steps 120000 -share_embeddings -world_size 1 -gpu_ranks 0 -valid_batch_size 64 -valid_steps 20000 -save_checkpoint_steps 20000 -max_grad_norm 5
```

### word level

2.4M examples, 120K steps：

|Model Name | ROUGE-1 | ROUGE-2 | ROUGE-L |
|- | :-: | :-: | :-: |
|OpenNMT-py | 33.8 | 20.6 | 30.6 |
|our | 35.4 | 22.6 | 32.5|

In this experiment, I tried a bigger model.
```
python train.py -data ./datadir/light_textsum -save_model ./savedir/raw_transformer1105 -log_file ./logdir/raw_transformer_log_1105 -seed 12345 -layers 2 -heads 16 -word_vec_size 512 -rnn_size 512 -optim adam -encoder_type transformer -decoder_type transformer -position_encoding -learning_rate 0.001 -batch_size 64 -dropout 0.15 -train_steps 120000 -share_embeddings -world_size 1 -gpu_ranks 0 -valid_batch_size 64 -valid_steps 20000 -save_checkpoint_steps 20000 -max_grad_norm 5
```
