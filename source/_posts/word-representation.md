---
title: Word representation in NLP
date: 2017-06-05 21:22:29
tag:
- NLP
auto_excerpt:
  enable: true
  length: 150
---

## One-hot Representation

1.  one 1 and lots of zeroes
2.  维度太大｜V｜，不易计算。
3.  This word representation does not give us directly any notion of similarity.

## SVD based method ［ ］

## Probabilistic Language Model: $P(w_t|w_{1:t-1})$ 

1.  问题：
    Given a sequence of letters, what is the likelihood of the next letter?
2.  统计语言模型:
    *   “统计语言模型把语言(词的序列)看 作一个随机事件,并赋予相应的概率来描述其属于某种语言集合 的可能性。给定一个词汇集合 V ,对于一个由 V 中的词构成的序 列S = ⟨w1, · · · , wT ⟩ ∈ Vn,统计语言模型赋予这个序列一个概率 P (S ),来衡量 S 符合自然语言的语法和语义规则的置信度。”
3.  语言模型的两个基本功能是:
    *   判断一段文本是否符合一种语言的语法和语义规则;
    *   生成符合一种语言语法或语义规则的文本。
4.  公式：
    $$
    \begin{aligned}
    P(S = w_1:T) ) &= P(W_1 = w_1,W_2 = w_2,··· ,W_T = w_T)\\
    &= P (w_1 , w_2 , · · · , w_T )\\
    P(S = w_1:T ) &= P(w_1,··· ,w_T ) \\
    &= P(w_1)P(w_2|w_1)P(w_3|w_1w_2)\cdots P(w_T|w_{1:T-1})\\
    &= \prod_{t=1}^T{P(w_t|w_{1:t-1})}\\
    &(假定P(w_1) = P(w_1|w_0))
    \end{aligned}
    $$
5.  N-gram:
    *   我们假设一个词的概率只依赖于其前面的 n − 1 个词(n 阶马尔可夫性质),即
        $$ P(w_t|w_{1:t-1}) = P(w_{t-n+1}:w_{t-1}) $$
    *   这就是N元(N-gram)语言模型。
    *   那么我们如何知道 $P(w_{t-n+1}:w_{t-1}))$ 的值呢?
        这里有两种方法:
        *   一是假设语言服从多项分布,然后用最大似然估计来计 算多项分布的参数;
        *   二是用一个函数 $f(w_{1:(t−1)})$ 来直接估计下一个词是 $w_t$ 的概率。
6.  一元语言模型［ ］
7.  最大似然估计［ ］
8.  平滑技术［ ］
9.  模型评价［ ］

## Neural Network Language Model (NNLM)

1.  问题： 在统计语言模型中,一个关键的问题是估计 $P(W_t |W_{1:(t−1)} )$,即在时刻(或位置)t,给定历史信息$h_t = w_{1:(t−1)}$ 条件下,词汇表V中的每个词 $v_k(1 ≤ k \le |V|)$ 出现的概率。
    *   这个问题可以转换为一个类别数为 |V| 的多类分类问题,即:
        $$
        \begin{aligned}
        P_\theta(W_t = v_k|h_t = w_{1:(t-1)}) &= P_\theta(v_k|w_{1:(t-1)})\\
        &= f_k(w_{1:(t-1)},\theta)\\
        \end{aligned}
        \\
        $$
    *   f_k是分类函数，估计的词汇表中第k个词出现的后验概率。θ为模型参数。
2.  输入层：
    *   将语言符号序列$w_{1:(t-1)}$ 输入到神经网络模型中,首先需要将这 些符号转换为向量形式。
    *   转换方式可以通过一个词嵌入矩阵来直接映射,也叫作输入词嵌入矩阵或 查询表。词嵌入矩阵M中,第k列向量$m_k ∈ R^{d_1}$ 表示词汇表中第k个词对应的稠密向量。
    *   通过直接映射,我们得到历史信息$w_{1:(t−1)}$ 每个词对应的向量表示$v_{w_1},··· ,v_{w_t−1}$ 。
3.  隐藏层：
    *   隐藏层可以是不同类型的网络,前馈神经网络和循环神经网络,其输
        入为词向量$v_{w_1},··· ,v_{w_t−1}$,输出为一个可以表示**历史信息的向量$h_t$** 。
    *   在神经网络语言模型中,常见的网络类型有以下三种:
        *   1)简单平均.($C_i$ 为每个词的权重)
            $$ h_t = \sum_{i=1}^{t-1}C_iv_{w_i}$$
        *   2)前馈神经网络[Bengio et al., 2003a]
            *   前馈神经网络要求输入的大小是固定的。
            *   拼接为一个维度为$d_1 × (n − 1)$ 的向量$x_t$。
            *   然后将 $x_t$ 输入到由多层前馈神经网络构成的隐藏层,最后一层隐藏层的输出$h_t$。
                $$ h_t = x_t \oplus tanh(Wx_t + c)$$
        *   3)循环神经网络[Mikolov et al., 2010]
            *   和前馈神经网络不同,循环神经网络可以接受变长的输入序列.
            *   依次接受输入$v_{w_1},··· ,v_{w_t−1}$。
            *   $$ h_t = tanh(Uh_{t-1} + Wv_{w_{t-1}} + c)$$
            *   前馈网络语言模型和循环网络语言模型的不同之处在于循环神经网络利用 隐藏状态来记录以前所有时刻的信息,而前馈神经网络只能接受前n − 1个时 刻的信息。
4.  输出层：
    *   输出层为大小为|V|,其接受的输入为历史信息的向量表示$h_t$ ,输 出为词汇表中每个词的后验概率。在神经网络语言模型中,一般使用**softmax**分类器。
    *   $y_t = softmax(\mathbf{O}h_t + b)$
    *   其中,输出向量$y_t ∈ R|V|$ 为一个概率分布,其第k维是词汇表中第k个词出现的后验概率;
    *   $O ∈ R^{|V|×d_2}$ 是最后一层隐藏层到输出层直接的权重。O也叫做输出词嵌入矩阵,矩阵中每一行也可以看作是一个词向量。
    *   在给定历史信息 $h_t$ 条件下,词汇表 V 中第 k 词 $v_k$ 出现的后验概率为:
        $$
        \begin{aligned}
        P_\theta(v_k|h_t) &= y_t^{(k)}\\
        &= softmax(s(v_k,h_t;\theta))\\
        &= \frac{exp(s(v_k,h_t;\theta))}{\Sigma_{j=1}^{|V|}exp(s(v_j,h_t;\theta))}
        \end{aligned}
        $$
    *   其中,**$s(v_k,h_t;θ)=\mathbf{o}^T_k\mathbf{h_t}+b_k$ 为未归一化的得分**,由神经网络计算得到;$θ$表示网络中的所有参数,包括词向量表 M 以及神经网络的权重和偏置。$\mathbf{o}_k$为输出嵌入矩阵O中的第k行向量的转置。分母为**配分函数**。
5.  训练
    *   给定一个训练文本序列$w_1,··· ,w_T$,神经网络语言模型的训练目标为找到一组参数 $θ$,使得**负对数似然函数 $NLL_θ$ 最**小。$R(θ)$ 为正则化项,比如 $R(θ) = \frac{1}{2}|\theta|^2_F$。
        $$
        \begin{aligned}
        NLL_\theta &= -1/T\sum_{t=1}^TlogP_\theta(w_t|w_{1:(t-1)}) + R(\theta)\\
        &= -1/T\sum_{t=1}^TlogP_\theta(w_t|h_t) + R(\theta)\\
        &= -1/T\sum_{t=1}^Tlog\frac{exp(s(w_t,h_t;\theta))}{\sum_{w’\in V}exp(s(w’,h_t;\theta))} + R(\theta)
        \end{aligned}
        $$
    *   参数$θ$ 可以通过梯度下降来迭代计算得到。$\alpha$为学习率。
        $\theta \gets \theta - \alpha\frac{\partial NLL_\theta}{\partial\theta}$
    *   为了使得神经网络语言模型的输出 $P_θ (w|h)$ 为一个概率分布,得分函数会 进行 softmax 归一化。而归一化时要计算配分函数,即对词汇表中所有的词 $w’$ 计算$s(w’, h_t; θ)$ 并求和,计算开销非常大。
    *   在实践中,经常采样一些近似估计的方法来加快训练速度。常用的加快神 经网络语言模型训练速度的方法可以分为两类:
        *   一类是层次化的softmax计算,将标准softmax函数的扁平结构转换为层 次化结构;
        *   另一类是基于采样的方法,通过采样来近似计算更新梯度。
6.  层次化softmax: **Hierarchical Softmax**
    *   考虑两层的来组织词汇表,即将词汇表中词分成 k 组,并每一个词只能属于一个分组,每组大小为 $|V |$ 。假设词 $w$ 所属的组为 $c(w)$,则
        $$
        \begin{aligned}
        P(w|h) &= P(w,c(w)|h)\\
        &= P(w|c(w),h)\times P(c(w)|h)
        \end{aligned}
        $$
        *   其中,$P (c(w)|h)$ 是给定历史信息 $h$ 条件下,类 $c(w)$ 的概率;$P (w|c(w), h)$ 是给定历史信息 $h$ 和类 $c(w)$ 条件下,词 $w$ 的概率。
        *   一个词的概率可以分解为两个概率 $P (w|c(w), h)$ 和 $P (c(w)|h)$ 的乘积,而它们都是利用神经网络来估计。
        *   计算softmax 函数时分别**只需要做 $|V |$ 和 $k$ 次求和**,从而就大大提高了softmax函数的计算速度的计算速度。
    *   为了进一步降低 softmax 的计算复杂度,我们可以**更深层的树结构**来组织词汇表。
        *   假设用二叉树来组织词汇表中的所有词,二叉树的叶子节点代表词汇表中 的词,非叶子节点表示不同层次上的类别。
        *   假设词 v 在二叉树上从根节点到其所在叶子节点的路径长度为 m,其编码 可以表示一个位向量(bit vector):$[b(v, 1), \cdots , b(v, m)]^T$。词 v 的条件概率为
            $$
            \begin{aligned}
            P(v|h) &= P(b(v, 1), \cdots , b(v, m)|h)\\
            &= \prod_{j=1}^mP(b(v,j)|b(v,1),\cdots,b(v,j-1),h)
            \end{aligned}
            $$
        *   这里,$b1(v), · · · , bj−1(v)$ 表示从根节点出发的长度为$j − 1$ 的路径,假设其对应一个节点$n_{j−1}(v)$ ($n_{j−1}(v)$ 是一个非叶子节点),则
            $P(b(v,j)|b(v,1),\cdots,b(v,j-1),h)=P(b(v,j)|n(v,j-1),h)$
        *   因为 $b(v, j) ∈ {0, 1}$,因此 $P(b(v,j)|n(v,j-1),h)$ 可以看作是两类分类问 题,我们可以使用 $logistic$ 回归来进行计算。
            $$
            \begin{aligned}
            P (b(v, j) = 1|n(v, j − 1), h) &= σ(o^\top_{n(v,j−1)}h + b_{n(v,j−1)}),\\
            P (b(v, j) = 0|n(v, j − 1), h) &= 1 - σ(o^\top_{n(v,j−1)}h + b_{n(v,j−1)}),
            \end{aligned}
            $$
        *   若使用平衡二叉树来进行分组,则条件概率估计可以转换为$log_2 |V|$ 个两类 分类问题。这时 softmax 函数可以用 logistic 函数代替, softmax 的计算可以加速 $\frac{|V|}{log_2|V|}$ 倍。
    *   将词汇表中的词按照树结构进行组织,有以下几种转换方式:
        *   利用人工整理的词汇层次结构,比如利用WordNet系统中的 “IS-A”关系(即上下位关系)。例如,“狗”是“动物”的下位词。
        *   使用Huffman编码。Huffman编码对出现概率高的词使用较短的编码,出现概率低的词则使用较长的编码。因此训练速度会更快。
7.  基于采样方法的梯度近似估计
    *   神经网络语言模型的目标函数为:
        $$
        NLL_\theta =
        = -\frac{1}{T}\sum_{t=1}^TlogP_\theta(w_t|h_t) + R(\theta)
        $$
    *   第 $t$ 个样本的目标函数关于 $θ$ 的梯度为
        $$
        \begin{aligned}
        \frac{\partial NLL_\theta(w_t,h_t)}{\partial\theta} &= \frac{\partial logP_\theta(w_t|h_t)}{\partial\theta}\\
        &= -\frac{\partial s(w_t,h_t;\theta)}{\partial\theta} + \sum_{w’}\frac{exp(s(w’,h_t;\theta))}{\sum_{w’}exp(s(w’,h_t;\theta))}\cdot\frac{\partial s(w’,h_t;\theta)}{\partial\theta}\\
        &= -\frac{\partial s(w_t,h_t;\theta)}{\partial\theta} + \sum_{w’}P_\theta(w’|h_t)\cdot\frac{\partial s(w’,h_t;\theta)}{\partial\theta}\\
        &= -\frac{\partial s(w_t,h_t;\theta)}{\partial\theta} + E_{P_\theta(w’|h_t)}\left[\frac{\partial s(w’,h_t;\theta)}{\partial\theta}\right]\\
        \end{aligned}
        $$
    *   最后一步公式中最后一项是计算 $\partial s(w’,h_t;\theta)$ 在分布 $P_θ(w’|h)$ 下的期望。**即对数似然的梯度里面有得分梯度的期望**。
    *   由于语言模型中的词汇表都比较大,训练速度会非常慢。在科学计算领域,对一个已知分布的 $p(x)$,函数 $f (x)$ 的期望可以**通过 $n$ 个样本的均值 $\hat{E}_n [f (x)]$ 来近似**。这种近似计算的方法叫做 Mente-Carlo 方法。从 p(x) 中抽取的一 组变量的过程即为采样过程。比如重要性采样和NCE
8.  重要性采样[ ]
9.  噪声对比估计(NCE): **Noise Contrastive Estimation**[ ]
    *   基本思想是将密度估计问题转换为两类分类问题,从而降低计算复杂度。
    *   噪声对比估计的数学描述如下:
        *   假设有三个分布,一个是需要建模真实数 据分布 $P (x)$;第二是模型分布 $P_θ (x)$,我们期望调整模型参数为 $θ$ 来是的 $P_θ (x)$ 来拟合真实数据分布;第三个是噪声分布 $Q(x)$,用来对比学习。
        *   从 $P (x)$ 和 $Q(x)$ 的混合分布中抽取一个样本 $x$,我们将从 $P (x)$ 中抽取的样本叫做“真实”样本, 从 $Q(x)$ 中抽取的样本叫做噪声样本。我们需要建立一个“辨别者”$D$ 来判断样本 $x$ 是真实样本还是噪声样本。
        *   噪声对比估计是通过调整模型 $P_θ(x)$ 使得“判别者” $D$ 很容易能分别出样本 $x$ 来自哪个分布。
    *   疑问：
        *   为什么可以直接令配分函数为1？
        *   引入噪声的含义是什么？标准是什么？怎么就能近似了？

## Word2vec

1.  $\bf{CBOW}$ (**给上下文预测$w_t$**)
    *   在标准的语言模型中,当前词 $w_t$ 依赖于其前面的词 $w_1:(t−1)$。而在连续词袋模型和 Skip-Gram 模型中,当前词 $w_t$ 依赖于其前后的词。
    *   给定一个词 $w_t$ 的其上下文 $c_t = w_{t–n}, \cdots , w_{t–1}, w_{t+1}, \cdots, w_{t+n}$ 连续词袋模型(Continuous Bag-of-Words,CBOW)是该词 $w_t$ 出现的条件概率为:
        $$
        \begin{aligned}
        P(w_t|c_t) &= softmax(\mathbf{v}’^\top_{w_t}\mathbf{c}_t)\\
        &= \frac{exp(\mathbf{v}’^\top_{w_t}\mathbf{c}_t)}{\sum_{w’\in V}exp(\mathbf{v}’^\top_{w’}\mathbf{c}_t)}
        \end{aligned}
        $$
    *   其中, $\mathbf{c}_t = \sum_{-n\le j \le n, j \ne 0}\mathbf{v}_{w_{t+j}}$ 表示上下文信息。$\mathbf{v}_{w_t}’$ 表示输出词嵌入矩阵 $(hidden→output\space matrix)$ $W’$ 的第 $t$ 列。
    *   在连续词袋模型中,就直接把隐藏层去掉, 大大减少了计算量, 提高了计算速度, 然后用更多的数据来训练模型, 最后的效果也不错。
    *   给定一个训练文本训练 $w_1,··· ,w_T$,连续词袋模型的目标函数为
        $\mathcal{L}_\theta = -\frac{1}{T}\sum_{t=1}^TlogP(w_t|c_t)$
2.  $\bf{Skip-Gram model}$ (**给$w_t$预测上下文**)
    *   Skip-Gram 模型给定一个词 $w_t$,预测词汇表中每个词出现在其上下文中的概率。
        $$
        \begin{aligned}
        P(w_{t+j}|w_t) &= softmax(\mathbf{v}^\top_{w_t}\mathbf{v}_{w_{t+j}}’)\\
        &= \frac{exp(\mathbf{v}^\top_{w_t}\mathbf{v}_{w_{t+j}}’)}{\sum_{w’\in V}exp(\mathbf{v}^\top_{w_t}\mathbf{v}_{w’}’)}
        \end{aligned}
        $$
    *   其中,$\mathbf{v}_w$ 表示词w在输入词嵌入矩阵中的词向量,$\mathbf{v}_w’$ 表示词w在输出词嵌入矩阵中的词向量。
    *   给定一个训练文本训练 $w_1,··· ,w_T$, $\bf{Skip-Gram}$ 模型的目标函数为
        $\mathcal{L}_\theta = -\frac{1}{T}\sum_{t=1}^T\sum_{-n\le j \le n, j \ne 0}logP(w_{t+j}|w_t)$
3.  训练方法
    *   在 Word2Vec 中,连续词袋模型和 Skip-Gram 模型都可以通过两种训练方 法(层次化 softmax 和负采样)来加速训练。
    *   层次化 softmax: 在 Word2Vec 中采用了 Huffman 树来进行词汇表的层次化。
4.  Negative Sampling
    *   思想类似于NCE, 也是找一个噪声分布, 然后将问题转化为模型能够区分好分布和噪声分布的二类分类问题从而避开使用softmax而可以使用logistic,大大减少计算量.
    *   **从本质上讲模型训练的困难就在于softmax的分母的求和**,需要将真个|v|大小的词库进行计算. 无论是层次化还是负采样都是将问题转化为了二类分类问题,避开使用softmax而可以使用logistic.
5.  Word2Vec加速技巧
    *   删除隐藏层,得到上下文c的表示后,直接输入到softmax分类器来预测 输出。也就是说,整个网络的参数只有两个词嵌入表:输入词嵌入表和输 出词嵌入表;
    *   使用层次化softmax或负采样进行加速训练
    *   去除低频词。出现次数小于一个预设值minCount的词直接去除。
    *   对高频词进行降采样。根据公式算出的概率 $Pdiscard(w_t)$ 来跳过词 $w_t$
    *   动态上下文窗口大小。指定一个最大窗口大小值N,对于每个词,从[1,N]
        中随机选取一个值 n 来作为本次的上下文窗口大小,从当前词左右各选取
        n 个词。这样上下文更侧重于邻近词。
    *   噪声分布使用一元语言模型 $U (x)$ 的 $\frac{U (x)^{3/4}}{Z}$ ,Z 为归一化因子。相当于对高频词进行降采样,对低频词进行上采样。

## Reference

大部分内容来源于:[https://nndl.github.io/ch12.pdf](https://nndl.github.io/ch12.pdf)
