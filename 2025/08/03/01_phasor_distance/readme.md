# **机器学习中向量距离与相似性度量综合指南**

- [**机器学习中向量距离与相似性度量综合指南**](#机器学习中向量距离与相似性度量综合指南)
  - [**引言：超越余弦相似性**](#引言超越余弦相似性)
  - [**第一部分：向量比较的基础概念**](#第一部分向量比较的基础概念)
    - [**1. 距离与相似性的二元性**](#1-距离与相似性的二元性)
      - [**定义与关系**](#定义与关系)
      - [**度量的数学性质**](#度量的数学性质)
    - [**2. 高维空间的挑战：维度灾难**](#2-高维空间的挑战维度灾难)
      - [**现象与影响**](#现象与影响)
      - [**对不同度量的影响**](#对不同度量的影响)
    - [**3. 向量的关键属性：模长、方向与归一化**](#3-向量的关键属性模长方向与归一化)
      - [**模长敏感 vs. 方向敏感**](#模长敏感-vs-方向敏感)
      - [**归一化的决定性作用**](#归一化的决定性作用)
  - [**第二部分：距离与相似性度量的系统性考察**](#第二部分距离与相似性度量的系统性考察)
    - [**4. 闵可夫斯基家族（Lp​ 范数）：几何距离的统一框架**](#4-闵可夫斯基家族lp-范数几何距离的统一框架)
      - [**4.1 欧几里得距离（L2​ 范数, p=2）：直观的最短路径**](#41-欧几里得距离l2-范数-p2直观的最短路径)
      - [**4.2 曼哈顿距离（L1​ 范数, p=1）：稳健的网格行者**](#42-曼哈顿距离l1-范数-p1稳健的网格行者)
      - [**4.3 切比雪夫距离（L∞​ 范数, p→∞）：极致的专家**](#43-切比雪夫距离l-范数-p极致的专家)
    - [**5. 方向性与关系性度量**](#5-方向性与关系性度量)
      - [**5.1 余弦相似性与距离：语义的黄金标准**](#51-余弦相似性与距离语义的黄金标准)
      - [**5.2 点积（内积）：混合型度量**](#52-点积内积混合型度量)
    - [**6. 针对集合与二元数据的度量**](#6-针对集合与二元数据的度量)
      - [**6.1 杰卡德指数与距离：重叠度的专家**](#61-杰卡德指数与距离重叠度的专家)
      - [**6.2 汉明距离：字符串比较器**](#62-汉明距离字符串比较器)
    - [**7. 高级统计距离**](#7-高级统计距离)
      - [**7.1 马氏距离：感知相关的度量**](#71-马氏距离感知相关的度量)
      - [**7.2 信息论散度：比较概率分布**](#72-信息论散度比较概率分布)
  - [**第三部分：实际应用与战略选择**](#第三部分实际应用与战略选择)
    - [**8. 选择正确度量的决策框架**](#8-选择正确度量的决策框架)
      - [**对齐的黄金法则**](#对齐的黄金法则)
      - [**度量选择的引导性问卷**](#度量选择的引导性问卷)
      - [**表1：向量距离与相似性度量比较指南**](#表1向量距离与相似性度量比较指南)
    - [**9. 深度应用场景分析**](#9-深度应用场景分析)
      - [**9.1 自然语言处理（NLP）**](#91-自然语言处理nlp)
      - [**9.2 计算机视觉与图像分析**](#92-计算机视觉与图像分析)
      - [**9.3 推荐系统**](#93-推荐系统)
      - [**9.4 聚类与异常检测**](#94-聚类与异常检测)
  - [**结论：度量即信息**](#结论度量即信息)
      - [**引用的资料**](#引用的资料)

## **<font color="DodgerBlue">引言：超越余弦相似性</font>**

在现代人工智能（AI）领域，数据正以前所未有的规模和形式被转化为高维向量嵌入（Vector Embeddings）。从文本、图像到用户偏好，各种信息都被映射到数学空间中，以捕捉其深层的语义关系 <sup><font color="red">1<color></font></sup>。这种向量化表示使得我们能够通过数学运算来发现数据间的内在联系，而其中最基础、最核心的操作便是度量向量之间的“远近”，即距离或相似性。

在众多度量方法中，余弦相似性（Cosine Similarity）已然成为事实上的标准，尤其在自然语言处理（NLP）领域，它通过计算向量间的夹角来衡量语义关联度，表现卓越 <sup><font color="red">1<color></font></sup>。然而，过度依赖单一的默认度量标准往往是次优选择。选择何种距离或相似性度量方法，本身就是一项关键的建模决策，它隐含了对数据特性和问题本质的深刻假设。对于一个给定的问题，“相似”的定义千差万别，而度量方法的选择正是对这一定义的数学诠释。

本报告旨在提供一个全面而深入的分析，系统性地探讨在现代AI应用中至关重要的向量距离与相似性度量。报告将超越简单的定义，深入剖析各种度量的数学性质、几何直观、实际应用中的权衡以及最佳适用场景。本报告的结构将引导读者首先掌握基础理论，然后系统性地审视各类关键度量，最终提供一个实用的应用与选择框架，旨在帮助从业者在面对具体问题时，能够有意识地、有效地做出最优决策。

---

## **<font color="DodgerBlue">第一部分：向量比较的基础概念</font>**

本部分为深入理解不同度量方法之间的权衡奠定必要的理论基础。

### **<font color="DarkViolet">1\. 距离与相似性的二元性</font>**

在量化两个对象之间的关系时，我们通常使用两个密切相关但方向相反的概念：相似性（Similarity）和距离（Distance）。

#### **定义与关系**

相似性度量（Similarity Measure）是指一个函数，其输出值随着两个对象变得更加相似而增大。相反，距离度量（Distance Measure）或度规（Metric）的输出值则随着对象愈发相似而减小，当两个对象完全相同时，距离通常为零 <sup><font color="red">3<color></font></sup>。

这两个概念如同硬币的两面，常常可以相互转化。一个常见的转化模式是，距离可以通过从一个常数（通常是1）中减去相似性得分来得到。

* **余弦距离（Cosine Distance）** 是最典型的例子。它被定义为 1−Cosine Similarity <sup><font color="red">1<color></font></sup>。当两个向量方向完全一致时，余弦相似性为1，对应的余弦距离为0。当它们方向完全相反时，相似性为-1，距离则为2。这个 $$ 的范围清晰地量化了方向上的差异。  
* **杰卡德距离（Jaccard Distance）** 同样遵循此模式，其定义为 1−Jaccard Similarity <sup><font color="red">6<color></font></sup>。这揭示了度量设计中的一种通用范式。

然而，这种转化并非总是简单的 1−x 关系。转化的具体形式会影响度量的最终性质。例如，余弦相似性的范围是 \[−1,1\]，转化后的距离范围是 $$。而杰卡德相似性的范围是 $$，其距离范围也是 $$。这意味着不同度量方法下的“距离”标度并不可直接通用，一个0.5的杰卡德距离与一个0.5的余弦距离所代表的相似程度是完全不同的。在实现如k近邻（k-NN）或聚类等算法时，距离值的尺度和分布会直接影响算法的性能和阈值设定。因此，选择何种转化函数本身就是度量设计的一部分，而不应被视为一个无足轻重的步骤。

#### **度量的数学性质**

一个严格的数学“度量”（Metric）必须满足四个基本条件：

1. **非负性**：任意两点间的距离必须大于等于零。  
2. **同一性**：两点间的距离为零，当且仅当这两点是同一点。  
3. **对称性**：从A到B的距离等于从B到A的距离。  
4. **三角不等式**：从A到C的距离小于或等于从A到B再到C的距离之和。

欧几里得距离和曼哈顿距离等都满足这些公理 <sup><font color="red">10<color></font></sup>。理解这些性质有助于我们区分哪些度量（如欧几里得距离）是严格的度量，而哪些（如KL散度）则不是，后者通常被称为“散度”（Divergence）。

### **<font color="DarkViolet">2\. 高维空间的挑战：维度灾难</font>**

“维度灾难”（Curse of Dimensionality）是处理现代高维向量嵌入（通常具有数百甚至数千个维度）时必须面对的一个核心挑战 <sup><font color="red">1<color></font></sup>。它描述了随着空间维度的增加，一系列反直觉现象的出现。

#### **现象与影响**

在高维空间中，空间的体积增长速度极快，导致有限的数据点变得异常稀疏。最令人困惑的后果是**距离集中现象**：对于一个给定的查询点，其最近邻点和最远邻点的距离之差与平均距离相比，变得微不足道。换言之，所有点到查询点的距离都趋于相等 <sup><font color="red">13<color></font></sup>。这使得基于距离的区分变得极为困难。

#### **对不同度量的影响**

维度灾难并非对所有度量都一视同仁，它对不同度量的影响程度有所不同，这种差异源于它们各自的数学构造。

* **欧几里得距离 (L2​ Norm)**：该度量在高维空间中尤其脆弱。它的计算涉及各维度差值的平方和。在高维情况下，大量维度上的微小、不相关的差异会被平方累加，最终导致任意两点间的距离都很大且彼此接近，使得距离的方差相对于其均值变得很小，从而丧失了区分能力 <sup><font color="red">13<color></font></sup>。  
* **曼哈顿距离 (L1​ Norm)** 和 **余弦相似性**：这两者通常被认为在高维空间中更具可靠性。  
  * 曼哈顿距离计算的是各维度差值的绝对值之和。它对差异的累加是线性的，不像欧几里得距离那样会因为平方操作而放大大量微小差异的影响。因此，当只有少数几个维度存在显著差异时，曼哈ton距离能更有效地捕捉到这种信号 <sup><font color="red">15<color></font></sup>。研究表明，对于高维数据，  
    L1​ 范数通常优于 L2​ 范数，甚至有研究提出使用分数阶范数（Lk​,k\<1）可能效果更佳 <sup><font color="red">13<color></font></sup>。  
  * 余弦相似性则通过完全忽略向量的模长（magnitude），只关注其方向，从而巧妙地规避了维度灾难的部分影响。在高维空间中，向量的模长可能受到各种噪声因素的干扰而变得信息量不足，但其方向仍然可以作为衡量语义相似性的稳定信号 <sup><font color="red">14<color></font></sup>。

这种现象并非随机的经验观察，而是高维空间几何性质与度量计算方式（平方、绝对值、角度计算）相互作用的直接结果。这也催生了对新型度量的研究，例如**维度不敏感欧几里得度量（Dimension Insensitive Euclidean Metric, DIEM）**，旨在提供在不同维度下都具有更好可靠性和泛化能力的解决方案 <sup><font color="red">17<color></font></sup>。

### **<font color="DarkViolet">3\. 向量的关键属性：模长、方向与归一化</font>**

向量的内在信息可以通过两个核心属性来解构：**模长**（Magnitude，或称长度、范数）和**方向**（Direction）18。不同的度量方法对这两个属性的敏感度各不相同，而归一化操作则从根本上改变了问题的性质。

#### **模长敏感 vs. 方向敏感**

* **模长敏感型度量**：**欧几里得距离**和**曼哈顿距离**主要对模长敏感。它们衡量的是向量空间中点的绝对位置差异 <sup><font color="red">20<color></font></sup>。如果将一个向量按比例缩放，它与其他向量的欧几里得距离会随之改变 <sup><font color="red">22<color></font></sup>。  
* **方向敏感型度量**：**余弦相似性**被明确设计为只对方向敏感，而对模长保持不变 <sup><font color="red">4<color></font></sup>。这正是它在文本分析等领域备受青睐的原因——文档的长度（与向量模长相关）不应影响其主题的相似性判断 <sup><font color="red">16<color></font></sup>。  
* **混合型度量**：**点积（Dot Product）** 则是一个混合体，同时对模长和方向都敏感。其公式 A⋅B=∥A∥∥B∥cos(θ) 完美地揭示了这一点：它等于余弦相似性乘以两个向量模长的乘积 <sup><font color="red">1<color></font></sup>。

#### **归一化的决定性作用**

L2​ 归一化（将所有向量缩放至单位长度1）是一个至关重要的操作。它不仅仅是一个预处理步骤，更是一项根本性的建模决策，因为它实际上是在声明：“在这个问题中，只有方向是重要的，模长是无关信息或噪声。”

归一化会带来一个关键的后果：

* **点积与余弦相似性的等价性**：当向量被归一化后，它们的模长 ∥A∥ 和 ∥B∥ 都等于1。此时，点积的公式简化为 A⋅B=cos(θ) <sup><font color="red">18<color></font></sup>。这意味着，对于归一化向量，  
  **点积在数值上等同于余弦相似性** <sup><font color="red">1<color></font></sup>。这是一个极其重要的实践结论，因为点积的计算通常比余弦相似性更快（避免了开方和除法运算）。  
* **欧几里得距离与余弦相似性的关联**：归一化也搭建了欧几里得距离和余弦相似性之间的桥梁。对于单位向量，它们之间的欧几里得距离平方满足关系式 d2=2−2cos(θ)。由于 d 和 cos(θ) 在此关系下是单调相关的，因此使用欧几里得距离和余弦相似性对邻居进行排序，将得到完全相同的结果 <sup><font color="red">14<color></font></sup>。

因此，一个团队在争论应该使用点积还是余弦相似性时，可能在讨论一个伪问题。真正的战略性问题应该是：“我们是否应该对向量进行归一化？”如果答案是肯定的，那么两者之间的选择就变成了计算效率或实现便利性的问题，而非语义上的差异。这个决策在归一化那一步就已经做出了。

---

## **<font color="DodgerBlue">第二部分：距离与相似性度量的系统性考察</font>**

本部分将系统地审视各类度量方法，按照其数学家族进行分类和深入剖析。

### **<font color="DarkViolet">4\. 闵可夫斯基家族（Lp​ 范数）：几何距离的统一框架</font>**

闵可夫斯基距离（Minkowski Distance）是一个广义的度量框架，它通过一个参数 p 将欧几里得距离、曼哈顿距离和切比雪夫距离统一起来 <sup><font color="red">12<color></font></sup>。

其通用公式为：

D(X,Y)=(i=1∑n​∣xi​−yi​∣p)1/p

其中，X 和 Y 是n维空间中的两个向量。参数 p 控制着度量对不同维度差异的敏感度。随着 p 值的增大，度量会越来越侧重于惩罚那个差异最大的维度 <sup><font color="red">12<color></font></sup>。  
这种对差异的惩罚方式选择，实际上反映了解决问题的“误差哲学”。L1​ 范数（曼哈顿距离）对所有维度的单位差异一视同仁，如同“民主投票”。L2​ 范数（欧几里得距离）对大差异施加了平方级的重罚，如同“加权投票”，使得离群值拥有更大的话语权。而 L∞​ 范数（切比雪夫距离）则只关心那个最大的差异，如同“独裁”，由最差表现的维度决定一切。因此，选择 p 值，就是在定义系统对不同类型误差的容忍度。

#### **4.1 欧几里得距离（L2​ 范数, p=2）：直观的最短路径**

* **定义与公式**：欧几里得距离衡量的是两点之间的直线距离，即“两点之间直线最短”的直观概念 <sup><font color="red">2<color></font></sup>。其计算公式为：  
  d(X,Y)=i=1∑n​(xi​−yi​)2​  
  19  
* **几何意义**：在二维空间中，它源于勾股定理，即直角三角形斜边的长度。这个概念被推广到n维空间，它隐含地假设了一个正交且不相关的特征空间 <sup><font color="red">27<color></font></sup>。  
* **应用场景**：  
  * **聚类分析**：作为K-Means等旨在发现球形簇的算法的默认度量 <sup><font color="red">3<color></font></sup>。  
  * **图像处理**：用于比较像素强度或图像特征向量 <sup><font color="red">20<color></font></sup>。但它对图像微小的空间位移或形变非常敏感，可能导致与直觉相悖的结果 <sup><font color="red">31<color></font></sup>。  
  * **推荐系统**：可用于度量用户或物品特征嵌入之间的绝对差异 <sup><font color="red">22<color></font></sup>。  
* **优缺点**：优点是直观、易于理解 <sup><font color="red">15<color></font></sup>。缺点是对特征的尺度敏感，对离群值敏感（因为差异被平方），并且在高维空间中其意义会减弱 <sup><font color="red">13<color></font></sup>。

#### **4.2 曼哈顿距离（L1​ 范数, p=1）：稳健的网格行者**

* **定义与公式**：又称“出租车距离”或“城市街区距离”，它计算的是沿各坐标轴的绝对差值之和 <sup><font color="red">1<color></font></sup>。其计算公式为：  
  d(X,Y)=i=1∑n​∣xi​−yi​∣  
  11  
* **几何意义**：模拟了在网格状城市中，出租车从一点到另一点必须沿街道行驶的路径总长 <sup><font color="red">2<color></font></sup>。  
* **应用场景**：  
  * **高维数据**：在高维和稀疏数据场景（如文本分类）中，通常比欧几里得距离更受欢迎，因为它受维度灾难的影响较小 <sup><font color="red">10<color></font></sup>。  
  * **对离群值的可靠性**：由于差值未经平方，它对单个维度上的极端值不那么敏感，因此更为稳健 <sup><font color="red">10<color></font></sup>。  
  * **基于网格的场景**：是游戏寻路、地理信息系统（GIS）中的城市导航、电路板布局等问题的理想选择 <sup><font color="red">10<color></font></sup>。  
  * **特征工程**：在LASSO回归中，L1范数被用作正则化项，以鼓励模型产生稀疏的权重，从而实现特征选择 <sup><font color="red">35<color></font></sup>。

#### **4.3 切比雪夫距离（L∞​ 范数, p→∞）：极致的专家**

* **定义**：切比雪夫距离是两向量在所有维度上坐标差的绝对值的最大值 <sup><font color="red">12<color></font></sup>。它是闵可夫斯基距离在  
  p 趋于无穷大时的极限情况。  
* **几何意义**：在棋盘上，它等于国王从一个格子移动到另一个格子所需的最少步数 <sup><font color="red">24<color></font></sup>。  
* **应用场景**：适用于那些系统的瓶颈或成本由单个最大偏差决定的场景，例如在物流、仓储或某些棋盘游戏分析中。

### **<font color="DarkViolet">5\. 方向性与关系性度量</font>**

这类度量关注向量间的相对方向或关系，而非其绝对位置。

#### **5.1 余弦相似性与距离：语义的黄金标准**

* **定义与公式**：余弦相似性度量两个非零向量之间夹角的余弦值，因此它是一个关于方向而非模长的度量 <sup><font color="red">1<color></font></sup>。其公式为：  
  Cosine Similarity=∥A∥∥B∥A⋅B​  
  <sup><font color="red">1<color></font></sup>。对应的余弦距离通常定义为  
  1−Cosine Similarity <sup><font color="red">1<color></font></sup>。  
* **性质**：取值范围为-1（完全相反）到1（完全相同），0表示向量正交（无相关性）2。它对向量的长度或尺度不敏感 <sup><font color="red">4<color></font></sup>。  
* **应用场景**：  
  * **NLP与文本分析**：这是比较词嵌入和文档嵌入（如Word2Vec, BERT）的主流度量 <sup><font color="red">1<color></font></sup>。它能自然地处理不同长度的文档，并捕捉语义相似性，因为在语义层面，词语的使用模式（方向）比原始词频（模长）更重要 <sup><font color="red">16<color></font></sup>。  
  * **推荐系统**：用于比较用户或物品的画像，当偏好的模式比偏好的强度更重要时，此度量非常有效 <sup><font color="red">3<color></font></sup>。  
* **局限性**：它完全忽略了模长，但这在某些场景下可能是一个缺点。例如，在评分预测中，一个总是打高分的用户和一个总是打低分的用户，即使品味方向一致，其行为模式也应被视为不同 <sup><font color="red">22<color></font></sup>。此外，研究发现余弦相似性在处理高频词的上下文嵌入时可能存在系统性偏差 <sup><font color="red">40<color></font></sup>。

#### **5.2 点积（内积）：混合型度量**

* **定义与公式**：点积是两个向量对应分量乘积的和 <sup><font color="red">2<color></font></sup>。其公式为：  
  A⋅B=i=1∑n​ai​bi​  
* **性质**：点积同时对模长和方向敏感 <sup><font color="red">1<color></font></sup>。正值表示夹角小于90度，负值表示夹角大于90度，零表示正交 <sup><font color="red">19<color></font></sup>。其取值范围是无界的。  
* **与余弦相似性的关系**：点积是余弦相似性与两向量模长乘积的结合体，即 A⋅B=∥A∥∥B∥cos(θ) <sup><font color="red">18<color></font></sup>。  
* **核心应用**：当向量被L2​归一化后，点积与余弦相似性等价，此时因其计算效率更高（无需开方和除法）而常被优先选用 <sup><font color="red">1<color></font></sup>。对于需要同时考虑方向和模长的场景，无论数据是否归一化，点积都是一个灵活的选择 <sup><font color="red">19<color></font></sup>。

向量模长的来源决定了它应被视为有用信号还是无关噪声，这直接指导了在余弦相似性（视模长为噪声）和点积（视模长为信号）之间的选择。例如，在NLP中，词频会影响word2vec向量的模长 <sup><font color="red">16<color></font></sup>。如果认为高频词（如"the"）的大模长是训练数据的统计副产品，应被忽略，那么余弦相似性是正确的选择。然而，在推荐系统中，用户向量的模长可能代表其活跃度或评分次数。一个高模长向量的用户可能是“专家”或“重度用户”，他与另一个重度用户的相似度，即使品味方向完全相同，也应高于与一个偶尔评分的用户的相似度。在这种情况下，模长是有用信号，使用点积会更恰当地奖励两个高模长向量之间的相似性 <sup><font color="red">18<color></font></sup>。

### **<font color="DarkViolet">6\. 针对集合与二元数据的度量</font>**

这类度量专为处理成员关系（存在或不存在）而非连续数值的数据而设计。

#### **6.1 杰卡德指数与距离：重叠度的专家**

* **定义**：杰卡德指数（Jaccard Index）通过计算两个有限集合的交集大小除以并集大小来度量它们的相似性 <sup><font color="red">7<color></font></sup>。杰卡德距离则为  
  1−Jaccard Index <sup><font color="red">6<color></font></sup>。  
* **性质**：取值范围从0（无交集）到1（完全相同）8。它非常适合处理二元数据或集合数据，因为在这种场景下，一个元素的存在比其不存在或出现的频率更重要 <sup><font color="red">7<color></font></sup>。它有效地忽略了两个集合中都未出现的元素（即0-0匹配）42。  
* **应用场景**：  
  * **推荐系统**：比较用户购买或喜欢的物品集合 <sup><font color="red">7<color></font></sup>。  
  * **计算机视觉**：作为交并比（Intersection over Union, IoU）来评估目标检测中预测边界框与真实边界框的重合度 <sup><font color="red">7<color></font></sup>。  
  * **文档分析**：将文档视为词语的集合进行比较 <sup><font color="red">7<color></font></sup>。

#### **6.2 汉明距离：字符串比较器**

* **定义**：对于两个等长的向量（通常是二进制字符串），汉明距离是指它们在对应位置上符号不同的次数 <sup><font color="red">3<color></font></sup>。  
* **性质**：计算简单。它假设向量长度相等，并且不考虑差异的大小，只关心它们是否不同 <sup><font color="red">37<color></font></sup>。  
* **应用场景**：主要用于通信领域的纠错码理论、遗传学（比较DNA序列）以及比较分类变量。

### **<font color="DarkViolet">7\. 高级统计距离</font>**

这类距离度量通常基于数据的统计分布特性，提供了更深层次的比较方式。

#### **7.1 马氏距离：感知相关的度量**

马氏距离（Mahalanobis Distance）从根本上改变了度量的参照系。传统的几何距离是在一个固定的坐标系中点对点地测量，而马氏距离则是从一个点到整个数据分布进行测量，并使用该分布自身的结构（即形状和方向）作为“标尺”。

* **定义**：它度量的是一个点到一个数据分布中心的距离，同时考虑了变量之间的协方差 <sup><font color="red">46<color></font></sup>。其公式为：  
  D2=(x−μ)TΣ−1(x−μ)

  其中，x 是观测点向量，μ 是分布的均值向量，Σ 是协方差矩阵 <sup><font color="red">47<color></font></sup>。  
* **几何意义**：马氏距离等价于在一个经过“白化”（whitening）变换后的空间中的标准欧几里得距离。在这个变换后的空间里，数据被重新缩放和旋转，使其协方差矩阵变为单位矩阵 <sup><font color="red">51<color></font></sup>。它实际上是以标准差为单位来衡量点与均值的距离，因此等距离点的轮廓是椭圆形而非球形 <sup><font color="red">46<color></font></sup>。  
* **相比欧几里得距离的核心优势**：  
  * **尺度不变性**：自动处理不同尺度下的变量，无需手动归一化 <sup><font color="red">49<color></font></sup>。  
  * **相关性感应**：考虑了变量间的相互依赖关系，这是欧几里得距离完全忽略的 <sup><font color="red">46<color></font></sup>。在一个数据云中，沿着方差大的方向移动一个单位的几何距离，其马氏距离会“更短”；反之，在方差小的方向移动，马氏距离会“更长”。  
* **应用场景**：  
  * **多元离群点/异常检测**：这是其最主要的应用。它能识别出那些具有非寻常*组合*值的点，即使这些点在任何单个维度上看起来都并不异常 <sup><font color="red">46<color></font></sup>。  
  * **分类与聚类**：当数据特征相关时，在k-NN或聚类算法中使用马氏距离作为度量，可以得到更准确的、非球形的簇 <sup><font color="red">30<color></font></sup>。  
* **局限性**：计算需要求协方差矩阵的逆，这在变量高度相关（多重共线性）或数据点数量少于维度数量时，计算成本高且数值不稳定 <sup><font color="red">46<color></font></sup>。

#### **7.2 信息论散度：比较概率分布**

当向量本身代表概率分布时，信息论散度成为合适的度量工具 <sup><font color="red">3<color></font></sup>。

* **KL散度（Kullback-Leibler Divergence）**：  
  * **定义**：KL散度，又称相对熵，衡量当使用一个概率分布 Q 来近似另一个真实分布 P 时所损失的信息量 <sup><font color="red">56<color></font></sup>。  
  * **核心性质：不对称性**：KL(P∥Q)=KL(Q∥P)。这使得它是一种“散度”而非严格的“度量” <sup><font color="red">56<color></font></sup>。选择哪个分布作为参照系至关重要。  
* **JS散度（Jensen-Shannon Divergence）**：  
  * **定义**：JS散度是KL散度的一个对称化、有界版本 <sup><font color="red">57<color></font></sup>。  
  * **工作机制**：它通过引入一个混合分布 M=0.5×(P+Q)，然后计算 P 到 M 和 Q 到 M 的KL散度的平均值，从而解决了KL散度的不对称问题 <sup><font color="red">56<color></font></sup>。  
  * **性质**：JS散度是对称的（JS(P∥Q)=JS(Q∥P)），并且其取值范围有界（\[0,ln2\]），这使得它（的平方根）成为一个真正的度量，在聚类等任务中更稳定、更实用 <sup><font color="red">59<color></font></sup>。

---

## **<font color="DodgerBlue">第三部分：实际应用与战略选择</font>**

本部分将前文的分析综合为可操作的实践指南。

### **<font color="DarkViolet">8\. 选择正确度量的决策框架</font>**

在选择度量方法时，虽然有诸多理论考量，但实践中存在一些可以极大简化决策过程的黄金法则和指导性问题。

#### **对齐的黄金法则**

最重要、最实用的建议是：**在推理（或检索）时使用的相似性度量，应与训练嵌入模型时所用的损失函数保持一致** <sup><font color="red">19<color></font></sup>。例如，如果一个句子转换器模型（Sentence-Transformer）是使用基于余弦相似性的损失函数进行训练的，那么在向量数据库中建立索引时，也应选择余弦相似性，这样才能获得最准确的检索结果。这条法则往往可以超越其他所有考量，成为决策的首要依据。

#### **度量选择的引导性问卷**

若无明确的训练度量可供对齐，或在设计新系统时，可以通过以下一系列问题来引导决策：

1. **你的向量代表什么？**  
   * **空间中的几何点**（如物理坐标、低维特征）：从**欧几里得/曼哈顿距离**入手。  
   * **高维语义内容**（如文本/图像嵌入）：从**余弦相似性/点积**入手。  
   * **二元向量或集合**（如用户购买历史、文档标签）：使用**杰卡德/汉明距离**。  
   * **可能存在相关的多元观测数据**（如传感器读数、金融指标）：强烈考虑**马氏距离**。  
   * **概率分布**：使用**JS散度/KL散度**。  
2. **向量的模长是否携带有效信息？**  
   * **否**（例如，文档长度是应被忽略的噪声）：使用**余弦相似性**。这是大多数语义搜索场景的标准选择。  
   * **是**（例如，代表用户活跃度、信号强度）：使用**欧几里得距离**或**点积**。  
3. **你的数据维度有多高？**  
   * **低维**：**欧几里得距离**通常表现良好且直观。  
   * **高维**：由于维度灾难，**余弦相似性**或**曼哈顿距离（L1​）** 通常比欧几里得距离（L2​）更稳健 <sup><font color="red">13<color></font></sup>。  
4. **你的特征之间是否相关？**  
   * **否/相互独立**：几何距离（欧几里得等）是合适的。  
   * **是**：**马氏距离**在理论上是更优的选择，因为它会消除协方差结构的影响 <sup><font color="red">30<color></font></sup>。  
5. **你是否担心离群值的影响？**  
   * **是**：**曼哈顿距离（L1​）** 比欧几里得距离（L2​）更具可靠性，因为它不会对大的差异进行平方放大 <sup><font color="red">15<color></font></sup>。

#### **表1：向量距离与相似性度量比较指南**

下表为从业者提供了一个快速参考工具，将报告的核心分析浓缩于一处，便于在不同度量之间进行快速比较和初步选择。

| 度量方法 | 公式 | 核心思想 (几何/直观意义) | 对模长敏感? | 对方向敏感? | 处理相关性? | 理想数据类型与维度 | 常见应用 | 优点 | 缺点 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **欧几里得距离 (L2​)** | $ \\sqrt{\\sum (x\_i \- y\_i)^2} $ | 两点间的直线距离（最短路径） | 是 | 否 | 否 | 低维、连续数值 | K-Means聚类、图像处理 | 直观、计算简单 | 对尺度和离群值敏感、高维下表现不佳 |
| **曼哈顿距离 (L1​)** | $ \\sum | x\_i \- y\_i | $ | 沿坐标轴移动的距离和（城市街区） | 是 | 否 | 否 | 高维、稀疏数据 | 文本分类、网格路径规划 |
| **切比雪夫距离 (L∞​)** | $ \\max\_i | x\_i \- y\_i | $ | 所有维度中最大的坐标差 | 是 | 否 | 否 | 特定领域（如物流） | 棋盘游戏、仓储物流 |
| **余弦相似性** | $ \\frac{A \\cdot B}{|A| |B|} $ | 两向量夹角的余弦值（方向一致性） | 否 | 是 | 否 | 高维、文本数据 | NLP、语义搜索、推荐系统 | 尺度不变、高维下可靠 | 忽略了有意义的模长信息 |
| **点积** | $ \\sum a\_i b\_i $ | 向量投影的乘积（方向与模长的结合） | 是 | 是 | 否 | 归一化向量、需要考虑模长的场景 | 向量检索（归一化后）、推荐系统 | 计算高效、结合模长与方向 | 无界、解释性不如余弦 |
| **杰卡德距离** | $ 1 \- \\frac{ | A \\cap B | }{ | A \\cup B | } $ | 集合的不重叠程度 | 否 | 否 | 否 |
| **汉明距离** | 不同位置的数量 | 两个等长字符串的差异字符数 | 否 | 否 | 否 | 分类变量、等长字符串 | 纠错码、基因序列比较 | 计算简单、概念清晰 | 要求向量等长、忽略差异大小 |
| **马氏距离** | $ \\sqrt{(x-\\mu)^T \\Sigma^{-1} (x-\\mu)} $ | 考虑数据分布形状的统计距离 | 否（尺度不变） | 否 | 是 | 多元、相关特征数据 | 多元异常检测、非球形聚类 | 尺度不变、考虑相关性 | 计算复杂、对协方差矩阵求逆可能不稳定 |

### **<font color="DarkViolet">9\. 深度应用场景分析</font>**

#### **9.1 自然语言处理（NLP）**

* **余弦相似性的主导地位**：在NLP领域，余弦相似性是衡量词嵌入或句子嵌入之间语义相似度的黄金标准 <sup><font color="red">4<color></font></sup>。其核心优势在于对模长的不敏感性。在文本表示中，向量的模长往往与一些非语义特征（如词频或文档长度）相关联 <sup><font color="red">16<color></font></sup>。例如，一篇长文档的向量模长可能远大于一篇短文档，但如果它们讨论的是同一主题，我们希望它们的相似度很高。余弦相似性通过只关注向量方向，完美地满足了这一需求。  
* **细微之处的问题**：然而，即便是余弦相似性也并非完美无瑕。有研究指出，它在处理像BERT这样的上下文嵌入时，会系统性地低估高频词的相似度。这被归因于高频词和低频词在表征空间中形成了不同的几何结构，导致了这种偏差 <sup><font color="red">40<color></font></sup>。

#### **9.2 计算机视觉与图像分析**

* **像素空间 vs. 特征空间**：在图像分析中，必须区分是在哪个空间中进行度量。直接在**像素空间**比较，**欧几里得距离**虽然常用，但存在严重缺陷：它对图像微小的平移、旋转或形变极其敏感，可能导致两个看起来非常相似的图像（如一个物体移动了一个像素）产生巨大的距离值 <sup><font color="red">31<color></font></sup>。为了解决这个问题，研究人员提出了如IMED（Image Euclidean Distance）等更稳健的度量方法 <sup><font color="red">31<color></font></sup>。  
* **深度学习时代**：当今的主流方法是先用深度模型（如CNN、ViT）将图像提取为高维**特征向量**，然后再进行比较。此时，度量的选择又回到了**欧几里得距离 vs. 余弦相似性**的经典权衡上，取决于特征向量的模长是否包含有用的信息。  
* **目标检测**：在目标检测任务中，**杰卡德指数**以\*\*交并比（IoU）\*\*的形式，成为评估预测边界框与真实边界框重合度的标准度量 <sup><font color="red">7<color></font></sup>。

#### **9.3 推荐系统**

推荐系统是一个综合运用多种度量方法的领域 <sup><font color="red">33<color></font></sup>。

* **物品相似度计算**：在**基于内容的过滤**中，度量方法的选择取决于物品特征的表示方式。如果物品由一组标签或属性描述，**杰卡德相似性**非常适合比较这些集合的重叠度 <sup><font color="red">43<color></font></sup>。如果物品被表示为高维特征向量（例如，从文本描述或图像中提取），则  
  **余弦相似性**是更常见的选择。  
* **用户/物品嵌入比较**：在**协同过滤**中，模型（如矩阵分解）会学习到用户和物品的低维嵌入向量。这些嵌入向量之间的距离可以用**欧几里得距离**或**余弦相似性**来度量，以找出相似的用户或推荐相关的物品 <sup><font color="red">33<color></font></sup>。  
* **超越相似度：评估推荐列表质量**：一个更深层次的应用是使用距离度量来评估最终生成的推荐列表的*质量*。例如，可以通过计算推荐列表中物品两两之间的平均**余弦距离**来衡量推荐的**多样性（Diversity）**——距离越大，说明推荐的物品种类越丰富。同样，可以计算推荐物品与用户历史行为物品之间的距离，来衡量推荐的**新颖性（Novelty）或惊喜度（Serendipity）** <sup><font color="red">39<color></font></sup>。

#### **9.4 聚类与异常检测**

在这些无监督学习任务中，度量的选择从根本上定义了“簇”的形状和“异常”的含义。

* **度量定义簇的形状**：聚类算法的目标是根据距离将数据点分组。因此，度量的选择直接决定了算法能发现什么样的簇。**欧几里得距离**假定并倾向于发现球形或凸形的簇 <sup><font color="red">30<color></font></sup>。  
  **马氏距离**则能够发现遵循数据内在相关性结构的椭球形、有方向性的簇 <sup><font color="red">30<color></font></sup>。  
* **度量定义异常的本质**：一个数据点是否是“离群点”，完全取决于我们如何度量它与“正常”数据群体的距离。一个点在**欧几里得距离**下可能是离群点（因为它离所有点都很远），但在**马氏距离**下可能不是（如果它恰好位于数据分布的主轴方向上，即使很远）。反之，**马氏距离**最擅长发现那些在任何单一维度上都不突出，但其特征组合却极不寻常的多元离群点 <sup><font color="red">46<color></font></sup>。

---

## **<font color="DodgerBlue">结论：度量即信息</font>**

本报告系统性地探讨了向量距离与相似性度量的广阔领域，其核心论点可以概括为：度量方法的选择并非一个可以随意调整的超参数，而是一项深刻的、依赖于具体情境的建模决策。它向算法传达了关于“相似性”本质的信息。

我们总结出指导这一决策的几个关键轴心：数据的内在类型（几何点、语义内容、集合或分布），数据所处的维度，向量模长所承载的意义（是信号还是噪声），以及特征之间是否存在相关性。

在所有这些考量之上，我们重申了那条最具有实践价值的“黄金法则”：**尽可能使推理阶段的度量与模型训练阶段的损失函数保持一致** <sup><font color="red">19<color></font></sup>。这种对齐能够最大化地发挥嵌入向量的潜力，确保检索或比较的结果与模型学习到的语义空间相吻合。

最后，对向量度量的研究远未结束。随着数据维度不断攀升，传统度量的局限性日益凸显，这推动了学术界和工业界对更可靠、更具泛化能力的度量方法的探索。诸如维度不敏感欧几里得度量（DIEM）等新兴研究 17，预示着一个未来——在这个未来中，我们的度量工具将能更好地应对现代高维数据的复杂挑战。最终，本报告旨在赋予从业者一种能力，即以分析的严谨性和战略的意图性来审视和选择度量方法，从而构建出更智能、更精准的AI系统。

#### **<font color="LightSeaGreen">引用的资料</font>**

1. Distance Metrics in Vector Search | Weaviate, 访问时间为 八月 5, 2025， [https://weaviate.io/blog/distance-metrics-in-vector-search](https://weaviate.io/blog/distance-metrics-in-vector-search)  
2. What is Vector Similarity? Understanding its Role in AI Applications. \- Qdrant, 访问时间为 八月 5, 2025， [https://qdrant.tech/blog/what-is-vector-similarity/](https://qdrant.tech/blog/what-is-vector-similarity/)  
3. Similarity measure \- Wikipedia, 访问时间为 八月 5, 2025， [https://en.wikipedia.org/wiki/Similarity\_measure](https://en.wikipedia.org/wiki/Similarity_measure)  
4. What is the role of cosine similarity in embeddings? \- Milvus, 访问时间为 八月 5, 2025， [https://milvus.io/ai-quick-reference/what-is-the-role-of-cosine-similarity-in-embeddings](https://milvus.io/ai-quick-reference/what-is-the-role-of-cosine-similarity-in-embeddings)  
5. arxiv.org, 访问时间为 八月 5, 2025， [https://arxiv.org/html/2408.07706v1\#:\~:text=Similarity%20and%20distance%20measures%20are,score%20if%20they%20are%20identical.](https://arxiv.org/html/2408.07706v1#:~:text=Similarity%20and%20distance%20measures%20are,score%20if%20they%20are%20identical.)  
6. docs.oracle.com, 访问时间为 八月 5, 2025， [https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/jaccard-similarity.html\#:\~:text=The%20Jaccard%20distance%20can%20be,that%20of%20a%20similarity%20calculation.](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/jaccard-similarity.html#:~:text=The%20Jaccard%20distance%20can%20be,that%20of%20a%20similarity%20calculation.)  
7. ❓ What is a Jaccard Distance : definition, examples of use., 访问时间为 八月 5, 2025， [https://ai-terms-glossary.com/item/jaccard-distance/](https://ai-terms-glossary.com/item/jaccard-distance/)  
8. Jaccard Similarity Made Simple: A Beginner's Guide to Data Comparison, 访问时间为 八月 5, 2025， [https://mayurdhvajsinhjadeja.medium.com/jaccard-similarity-34e2c15fb524](https://mayurdhvajsinhjadeja.medium.com/jaccard-similarity-34e2c15fb524)  
9. Jaccard index \- Wikipedia, 访问时间为 八月 5, 2025， [https://en.wikipedia.org/wiki/Jaccard\_index](https://en.wikipedia.org/wiki/Jaccard_index)  
10. What is Manhattan Distance? A Deep Dive | DataCamp, 访问时间为 八月 5, 2025， [https://www.datacamp.com/tutorial/manhattan-distance](https://www.datacamp.com/tutorial/manhattan-distance)  
11. Manhattan Distance: A Deep Dive \- Number Analytics, 访问时间为 八月 5, 2025， [https://www.numberanalytics.com/blog/manhattan-distance-deep-dive](https://www.numberanalytics.com/blog/manhattan-distance-deep-dive)  
12. Minkowski Distance: A Comprehensive Guide \- DataCamp, 访问时间为 八月 5, 2025， [https://www.datacamp.com/tutorial/minkowski-distance](https://www.datacamp.com/tutorial/minkowski-distance)  
13. Why is Euclidean distance not a good metric in high dimensions? \- Cross Validated, 访问时间为 八月 5, 2025， [https://stats.stackexchange.com/questions/99171/why-is-euclidean-distance-not-a-good-metric-in-high-dimensions](https://stats.stackexchange.com/questions/99171/why-is-euclidean-distance-not-a-good-metric-in-high-dimensions)  
14. Why Cosine Similarity for Transformer Text Embeddings? : r/learnmachinelearning \- Reddit, 访问时间为 八月 5, 2025， [https://www.reddit.com/r/learnmachinelearning/comments/12cp2cg/why\_cosine\_similarity\_for\_transformer\_text/](https://www.reddit.com/r/learnmachinelearning/comments/12cp2cg/why_cosine_similarity_for_transformer_text/)  
15. Decoding Distance: Understanding Euclidean and Manhattan Metrics \- Kaggle, 访问时间为 八月 5, 2025， [https://www.kaggle.com/discussions/general/576329](https://www.kaggle.com/discussions/general/576329)  
16. Why do we use cosine similarity on Word2Vec (instead of Euclidean distance)? \- Quora, 访问时间为 八月 5, 2025， [https://www.quora.com/Why-do-we-use-cosine-similarity-on-Word2Vec-instead-of-Euclidean-distance](https://www.quora.com/Why-do-we-use-cosine-similarity-on-Word2Vec-instead-of-Euclidean-distance)  
17. Surpassing Cosine Similarity for Multidimensional Comparisons: Dimension Insensitive Euclidean Metric \- arXiv, 访问时间为 八月 5, 2025， [https://arxiv.org/html/2407.08623v4](https://arxiv.org/html/2407.08623v4)  
18. Understanding Vector Similarity for Machine Learning | by Frederik vom Lehn \- Medium, 访问时间为 八月 5, 2025， [https://medium.com/advanced-deep-learning/understanding-vector-similarity-b9c10f7506de](https://medium.com/advanced-deep-learning/understanding-vector-similarity-b9c10f7506de)  
19. Similarity Metrics for Vector Search \- Zilliz blog, 访问时间为 八月 5, 2025， [https://zilliz.com/blog/similarity-metrics-for-vector-search](https://zilliz.com/blog/similarity-metrics-for-vector-search)  
20. Unveiling the Power: Cosine Similarity vs Euclidean Distance \- MyScale, 访问时间为 八月 5, 2025， [https://myscale.com/blog/power-cosine-similarity-vs-euclidean-distance-explained/](https://myscale.com/blog/power-cosine-similarity-vs-euclidean-distance-explained/)  
21. Euclidean Distance vs Cosine Similarity | Baeldung on Computer ..., 访问时间为 八月 5, 2025， [https://www.baeldung.com/cs/euclidean-distance-vs-cosine-similarity](https://www.baeldung.com/cs/euclidean-distance-vs-cosine-similarity)  
22. Vector Similarity Explained | Pinecone, 访问时间为 八月 5, 2025， [https://www.pinecone.io/learn/vector-similarity/](https://www.pinecone.io/learn/vector-similarity/)  
23. Euclidean vs. Cosine Distance \- Chris Emmery, 访问时间为 八月 5, 2025， [https://cmry.github.io/notes/euclidean-v-cosine](https://cmry.github.io/notes/euclidean-v-cosine)  
24. Minkowski distance \- Wikipedia, 访问时间为 八月 5, 2025， [https://en.wikipedia.org/wiki/Minkowski\_distance](https://en.wikipedia.org/wiki/Minkowski_distance)  
25. Minkowski Metric for ML \- Number Analytics, 访问时间为 八月 5, 2025， [https://www.numberanalytics.com/blog/minkowski-metric-for-machine-learning](https://www.numberanalytics.com/blog/minkowski-metric-for-machine-learning)  
26. Minkowski Distance \- LogicPlum, 访问时间为 八月 5, 2025， [https://logicplum.com/blog/knowledge-base/minkowski-distance/](https://logicplum.com/blog/knowledge-base/minkowski-distance/)  
27. Understanding Euclidean Distance: From Theory to Practice ..., 访问时间为 八月 5, 2025， [https://www.datacamp.com/tutorial/euclidean-distance](https://www.datacamp.com/tutorial/euclidean-distance)  
28. Euclidean Distance \- GeeksforGeeks, 访问时间为 八月 5, 2025， [https://www.geeksforgeeks.org/maths/euclidean-distance/](https://www.geeksforgeeks.org/maths/euclidean-distance/)  
29. Distance Metrics: A Guide for Data Science Applications | by Sanjay Kumar PhD, 访问时间为 八月 5, 2025， [https://skphd.medium.com/distance-metrics-a-guide-for-data-science-applications-8a6598a7ce90](https://skphd.medium.com/distance-metrics-a-guide-for-data-science-applications-8a6598a7ce90)  
30. Ultimate Distance Metrics for Clustering \- Number Analytics, 访问时间为 八月 5, 2025， [https://www.numberanalytics.com/blog/distance-metrics-clustering-guide](https://www.numberanalytics.com/blog/distance-metrics-clustering-guide)  
31. On the Euclidean Distance of Images \- ResearchGate, 访问时间为 八月 5, 2025， [https://www.researchgate.net/file.PostFileLoader.html?id=55c2ca635e9d97bebd8b4595\&assetKey=AS%3A273825961316352%401442296604149](https://www.researchgate.net/file.PostFileLoader.html?id=55c2ca635e9d97bebd8b4595&assetKey=AS:273825961316352@1442296604149)  
32. How to Calculate Distance Using Computer Vision Models? \- Ultralytics, 访问时间为 八月 5, 2025， [https://www.ultralytics.com/blog/how-to-calculate-distance-using-computer-vision-models](https://www.ultralytics.com/blog/how-to-calculate-distance-using-computer-vision-models)  
33. A Comprehensive Survey of Evaluation Techniques for Recommendation Systems \- arXiv, 访问时间为 八月 5, 2025， [https://arxiv.org/html/2312.16015v2](https://arxiv.org/html/2312.16015v2)  
34. Manhattan distance \[Explained\] \- OpenGenus IQ, 访问时间为 八月 5, 2025， [https://iq.opengenus.org/manhattan-distance/](https://iq.opengenus.org/manhattan-distance/)  
35. Manhattan Distance \- LogicPlum, 访问时间为 八月 5, 2025， [https://logicplum.com/blog/knowledge-base/manhattan-distance/](https://logicplum.com/blog/knowledge-base/manhattan-distance/)  
36. When would one use Manhattan distance as opposed to Euclidean distance?, 访问时间为 八月 5, 2025， [https://datascience.stackexchange.com/questions/20075/when-would-one-use-manhattan-distance-as-opposed-to-euclidean-distance](https://datascience.stackexchange.com/questions/20075/when-would-one-use-manhattan-distance-as-opposed-to-euclidean-distance)  
37. 9 Distance Measures in Data Science \- Maarten Grootendorst, 访问时间为 八月 5, 2025， [https://www.maartengrootendorst.com/blog/distances/](https://www.maartengrootendorst.com/blog/distances/)  
38. Understanding Cosine Similarity and Word Embeddings | by Spencer Porter \- Medium, 访问时间为 八月 5, 2025， [https://spencerporter2.medium.com/understanding-cosine-similarity-and-word-embeddings-dbf19362a3c](https://spencerporter2.medium.com/understanding-cosine-similarity-and-word-embeddings-dbf19362a3c)  
39. 10 metrics to evaluate recommender and ranking systems \- Evidently AI, 访问时间为 八月 5, 2025， [https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems](https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems)  
40. Problems with Cosine as a Measure of Embedding Similarity for High Frequency Words, 访问时间为 八月 5, 2025， [https://aclanthology.org/2022.acl-short.45/](https://aclanthology.org/2022.acl-short.45/)  
41. Why does word2Vec use cosine similarity? \- Stack Overflow, 访问时间为 八月 5, 2025， [https://stackoverflow.com/questions/38423387/why-does-word2vec-use-cosine-similarity](https://stackoverflow.com/questions/38423387/why-does-word2vec-use-cosine-similarity)  
42. Jaccard Similarity – LearnDataSci, 访问时间为 八月 5, 2025， [https://www.learndatasci.com/glossary/jaccard-similarity/](https://www.learndatasci.com/glossary/jaccard-similarity/)  
43. Recommender Systems: Machine Learning Metrics and Business ..., 访问时间为 八月 5, 2025， [https://neptune.ai/blog/recommender-systems-metrics](https://neptune.ai/blog/recommender-systems-metrics)  
44. Jaccard Similarity \- Oracle AI Vector Search User's Guide, 访问时间为 八月 5, 2025， [https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/jaccard-similarity.html](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/jaccard-similarity.html)  
45. Measuring distance between vectors \- probability \- Stack Overflow, 访问时间为 八月 5, 2025， [https://stackoverflow.com/questions/19048089/measuring-distance-between-vectors](https://stackoverflow.com/questions/19048089/measuring-distance-between-vectors)  
46. Mahalanobis Distance: Simple Definition, Examples \- Statistics How ..., 访问时间为 八月 5, 2025， [https://www.statisticshowto.com/mahalanobis-distance/](https://www.statisticshowto.com/mahalanobis-distance/)  
47. “Unlocking the Power of Mahalanobis Distance: Exploring Multivariate Data Analysis with Python” | by Vishal Sharma | Medium, 访问时间为 八月 5, 2025， [https://medium.com/@the\_daft\_introvert/mahalanobis-distance-5c11a757b099](https://medium.com/@the_daft_introvert/mahalanobis-distance-5c11a757b099)  
48. The Ultimate Guide to Mahalanobis Distance \- Number Analytics, 访问时间为 八月 5, 2025， [https://www.numberanalytics.com/blog/ultimate-mahalanobis-distance-guide](https://www.numberanalytics.com/blog/ultimate-mahalanobis-distance-guide)  
49. Mastering Mahalanobis Distance in ML \- Number Analytics, 访问时间为 八月 5, 2025， [https://www.numberanalytics.com/blog/mahalanobis-distance-machine-learning](https://www.numberanalytics.com/blog/mahalanobis-distance-machine-learning)  
50. Exploring Common Distance Measures for Machine Learning and Data Science: A Comparative Analysis | by Sahel Eskandar | Medium, 访问时间为 八月 5, 2025， [https://medium.com/@eskandar.sahel/exploring-common-distance-measures-for-machine-learning-and-data-science-a-comparative-analysis-ea0216c93ba3](https://medium.com/@eskandar.sahel/exploring-common-distance-measures-for-machine-learning-and-data-science-a-comparative-analysis-ea0216c93ba3)  
51. Mahalanobis distance \- Wikipedia, 访问时间为 八月 5, 2025， [https://en.wikipedia.org/wiki/Mahalanobis\_distance](https://en.wikipedia.org/wiki/Mahalanobis_distance)  
52. unit of measure \- Mahalanobis distance connection with Euclidean ..., 访问时间为 八月 5, 2025， [https://math.stackexchange.com/questions/3947977/mahalanobis-distance-connection-with-euclidean-distance](https://math.stackexchange.com/questions/3947977/mahalanobis-distance-connection-with-euclidean-distance)  
53. Euclidean distance and the Mahalanobis distance (and the error ellipse) \- YouTube, 访问时间为 八月 5, 2025， [https://www.youtube.com/watch?v=xXhLvheEF7o](https://www.youtube.com/watch?v=xXhLvheEF7o)  
54. Comparison of the Mahalanobis distance with the Euclidean distance on Reuters-21578 \- ResearchGate, 访问时间为 八月 5, 2025， [https://www.researchgate.net/figure/Comparison-of-the-Mahalanobis-distance-with-the-Euclidean-distance-on-Reuters-21578\_fig5\_277195972](https://www.researchgate.net/figure/Comparison-of-the-Mahalanobis-distance-with-the-Euclidean-distance-on-Reuters-21578_fig5_277195972)  
55. Multivariate Distances: Mahalanobis vs. Euclidean \- Water Programming \- WordPress.com, 访问时间为 八月 5, 2025， [https://waterprogramming.wordpress.com/2018/07/23/multivariate-distances-mahalanobis-vs-euclidean/](https://waterprogramming.wordpress.com/2018/07/23/multivariate-distances-mahalanobis-vs-euclidean/)  
56. Understanding KL Divergence, Entropy, and Related Concepts \- Towards Data Science, 访问时间为 八月 5, 2025， [https://towardsdatascience.com/understanding-kl-divergence-entropy-and-related-concepts-75e766a2fd9e/](https://towardsdatascience.com/understanding-kl-divergence-entropy-and-related-concepts-75e766a2fd9e/)  
57. How to Calculate the KL Divergence for Machine Learning \- MachineLearningMastery.com, 访问时间为 八月 5, 2025， [https://machinelearningmastery.com/divergence-between-probability-distributions/](https://machinelearningmastery.com/divergence-between-probability-distributions/)  
58. arize.com, 访问时间为 八月 5, 2025， [https://arize.com/blog-course/jensen-shannon-divergence/\#:\~:text=Jensen%2DShannon%20is%20an%20asymmetric,distributions%20are%20from%20each%20other.](https://arize.com/blog-course/jensen-shannon-divergence/#:~:text=Jensen%2DShannon%20is%20an%20asymmetric,distributions%20are%20from%20each%20other.)  
59. The Ultimate Guide to Jensen-Shannon Divergence \- Number Analytics, 访问时间为 八月 5, 2025， [https://www.numberanalytics.com/blog/ultimate-guide-jensen-shannon-divergence](https://www.numberanalytics.com/blog/ultimate-guide-jensen-shannon-divergence)  
60. Understanding KL Divergence, Entropy, and Related Concepts ..., 访问时间为 八月 5, 2025， [https://towardsdatascience.com/understanding-kl-divergence-entropy-and-related-concepts-75e766a2fd9e](https://towardsdatascience.com/understanding-kl-divergence-entropy-and-related-concepts-75e766a2fd9e)  
61. \[1904.04017\] On a generalization of the Jensen-Shannon divergence and the JS-symmetrization of distances relying on abstract means \- arXiv, 访问时间为 八月 5, 2025， [https://arxiv.org/abs/1904.04017](https://arxiv.org/abs/1904.04017)  
62. Jensen Shannon Divergence vs Kullback-Leibler Divergence? \- Cross Validated, 访问时间为 八月 5, 2025， [https://stats.stackexchange.com/questions/117225/jensen-shannon-divergence-vs-kullback-leibler-divergence](https://stats.stackexchange.com/questions/117225/jensen-shannon-divergence-vs-kullback-leibler-divergence)  
63. What's the best index distance metric for my Pinecone vector database, filled with a series of similarly formatted Markdown files? \- Stack Overflow, 访问时间为 八月 5, 2025， [https://stackoverflow.com/questions/76894530/whats-the-best-index-distance-metric-for-my-pinecone-vector-database-filled-wi](https://stackoverflow.com/questions/76894530/whats-the-best-index-distance-metric-for-my-pinecone-vector-database-filled-wi)  
64. recommender system \- Distance between users \- Data Science Stack Exchange, 访问时间为 八月 5, 2025， [https://datascience.stackexchange.com/questions/53956/distance-between-users](https://datascience.stackexchange.com/questions/53956/distance-between-users)