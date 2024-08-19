import torch
from torch.optim import Adam

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载模型
checkpoint_1 = "/home/ma-user/work/share_base_models/Llama-3-8B/"
model_1 = AutoModelForCausalLM.from_pretrained(checkpoint_1, device_map='auto', torch_dtype=torch.bfloat16,
                                               load_in_8bit=False)
checkpoint_2 = "./model/llama3_8b_base_epoch_3/"
model_2 = AutoModelForCausalLM.from_pretrained(checkpoint_2, device_map='auto', torch_dtype=torch.bfloat16,
                                               load_in_8bit=False)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint_1, padding_side='right', model_max_length=4096,
                                          tokenizer_type='llama')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model_1.resize_token_embeddings(len(tokenizer))
model_2.resize_token_embeddings(len(tokenizer))


# 定义合并模型的函数
def merge_models(model1, model2, alpha):
    merged_model = model1  # 创建合并模型的副本，可以通过复制 model1 的参数来初始化
    alpha = torch.tensor(alpha, dtype=torch.float32)

    # 遍历模型层并合并注意力层的权重和偏置
    for i in range(len(model1.model.layers)):
        # 获取模型第 i 层的注意力层的权重和偏置
        attention_weights_q1 = model1.model.layers[i].self_attn.q_proj.weight.data
        attention_weights_k1 = model1.model.layers[i].self_attn.k_proj.weight.data
        attention_weights_v1 = model1.model.layers[i].self_attn.v_proj.weight.data
        attention_weights_o1 = model1.model.layers[i].self_attn.o_proj.weight.data

        attention_weights_q2 = model2.model.layers[i].self_attn.q_proj.weight.data
        attention_weights_k2 = model2.model.layers[i].self_attn.k_proj.weight.data
        attention_weights_v2 = model2.model.layers[i].self_attn.v_proj.weight.data
        attention_weights_o2 = model2.model.layers[i].self_attn.o_proj.weight.data

        # 按照 alpha 合并权重
        merged_attention_weights_q = alpha * attention_weights_q1 + (1 - alpha) * attention_weights_q2
        merged_attention_weights_k = alpha * attention_weights_k1 + (1 - alpha) * attention_weights_k2
        merged_attention_weights_v = alpha * attention_weights_v1 + (1 - alpha) * attention_weights_v2
        merged_attention_weights_o = alpha * attention_weights_o1 + (1 - alpha) * attention_weights_o2

        # 将合并后的权重赋值给 merged_model
        merged_model.model.layers[i].self_attn.q_proj.weight.data.copy_(merged_attention_weights_q)
        merged_model.model.layers[i].self_attn.k_proj.weight.data.copy_(merged_attention_weights_k)
        merged_model.model.layers[i].self_attn.v_proj.weight.data.copy_(merged_attention_weights_v)
        merged_model.model.layers[i].self_attn.o_proj.weight.data.copy_(merged_attention_weights_o)

    return merged_model


def compute_attention_difference(model1, model2, data_loader):
    model1.eval()
    model2.eval()
    total_mse_difference = 0.0
    total_cosine_difference = 0.0
    total_emd_difference = 0.0
    total_kl_difference = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs, _ = batch
            outputs1 = model1(**inputs, output_attentions=True)
            outputs2 = model2(**inputs, output_attentions=True)

            # 假设我们关注第一层的注意力矩阵
            attention1 = outputs1.attentions[0].mean(0)  # 取批次的平均
            attention2 = outputs2.attentions[0].mean(0)

            # 计算均方误差 (MSE)
            mse_difference = ((attention1 - attention2) ** 2).mean()
            total_mse_difference += mse_difference

            # 计算余弦相似度差异
            attention1_flat = attention1.view(-1)
            attention2_flat = attention2.view(-1)
            cosine_sim = cosine_similarity(attention1_flat, attention2_flat, dim=0)
            cosine_difference = 1 - cosine_sim  # 因为相似度越高，差异越小
            total_cosine_difference += cosine_difference

            # 计算 Earth Mover's Distance (EMD)
            attention1_np = attention1.cpu().numpy().flatten()
            attention2_np = attention2.cpu().numpy().flatten()
            emd_difference = wasserstein_distance(attention1_np, attention2_np)
            total_emd_difference += emd_difference

            # 计算 KL散度 (Kullback-Leibler Divergence)
            attention1_softmax = softmax(attention1, dim=-1)
            attention2_softmax = softmax(attention2, dim=-1)
            kl_difference = kl_div(attention1_softmax.log(), attention2_softmax, reduction='batchmean')
            total_kl_difference += kl_difference

            total_batches += 1

    average_mse_difference = total_mse_difference / total_batches
    average_cosine_difference = total_cosine_difference / total_batches
    average_emd_difference = total_emd_difference / total_batches
    average_kl_difference = total_kl_difference / total_batches

    return {
        'mse_difference': average_mse_difference,
        'cosine_difference': average_cosine_difference,
        'emd_difference': average_emd_difference,
        'kl_difference': average_kl_difference
    }


def find_optimal_alpha(model1, model2, data_loader, initial_alpha=0.5, learning_rate=0.01, steps=100):
    alpha = torch.tensor(initial_alpha, requires_grad=True)
    optimizer = Adam([alpha], lr=learning_rate)

    for step in range(steps):
        optimizer.zero_grad()
        merged_model = merge_models(model1, model2, alpha.item())
        differences = compute_attention_difference(merged_model, model2, data_loader)

        # 这里使用均方误差 (MSE) 作为损失，你可以根据需要修改
        loss = differences['mse_difference']

        loss.backward()
        optimizer.step()

        # 保证 alpha 在 0 到 1 之间
        with torch.no_grad():
            alpha.clamp_(0, 1)

        print(f"Step {step + 1}/{steps}, Alpha: {alpha.item()}, Loss: {loss.item()}")

    return alpha.item()
