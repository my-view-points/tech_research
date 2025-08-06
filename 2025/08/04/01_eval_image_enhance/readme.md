# **评估图像质量增强对隧道场景下目标追踪效果影响的实验方案**

- [**评估图像质量增强对隧道场景下目标追踪效果影响的实验方案**](#评估图像质量增强对隧道场景下目标追踪效果影响的实验方案)
  - [**第一部分：核心理念与基本假设**](#第一部分核心理念与基本假设)
    - [**1.1 问题陈述与实验目标**](#11-问题陈述与实验目标)
    - [**1.2 核心假设**](#12-核心假设)
  - [**第二部分：实验组件选择**](#第二部分实验组件选择)
    - [**2.1 待评估的图像增强库（自变量）**](#21-待评估的图像增强库自变量)
    - [**2.2 固定的目标追踪系统（常量）**](#22-固定的目标追踪系统常量)
    - [**2.3 评估工具与指标（因变量）**](#23-评估工具与指标因变量)
  - [**第三部分：实验协议**](#第三部分实验协议)
    - [**3.1 数据准备**](#31-数据准备)
    - [**3.2 实验流程**](#32-实验流程)
    - [**3.3 性能评估**](#33-性能评估)
  - [**第四部分：结果分析与决策**](#第四部分结果分析与决策)
    - [**4.1 定量分析**](#41-定量分析)
    - [**4.2 成本效益分析**](#42-成本效益分析)
    - [**4.3 定性分析**](#43-定性分析)
    - [**4.4 最终建议**](#44-最终建议)
      - [**引用的资料**](#引用的资料)

## **<font color="DodgerBlue">第一部分：核心理念与基本假设</font>**

### **<font color="DarkViolet">1.1 问题陈述与实验目标</font>**

在隧道、夜间或恶劣天气等视觉条件不佳的环境中，图像质量（如低光照、对比度不足、噪声、模糊）是限制目标检测与追踪系统性能的关键瓶颈 <sup><font color="red">1<color></font></sup>。理论上，通过在追踪流程前加入一个图像增强预处理模块，可以提升图像的视觉质量，从而让下游的目标检测器和追踪器工作在更有利的数据上 <sup><font color="red">3<color></font></sup>。

然而，这种“先增强，后追踪”的策略并非总是有效的。研究表明，为人类视觉感知优化的增强算法可能并不会给计算机视觉任务带来同等的性能提升，甚至可能因为引入非预期的伪影（artifacts）、改变了模型训练时所依赖的特征分布，而对检测和追踪性能产生负面影响 <sup><font color="red">3<color></font></sup>。

因此，本方案的核心目标是：**通过一个严格的对照实验，定量地、可复现地评估一系列开源图像质量增强库，对一个固定的、先进的多目标追踪（MOT）系统在隧道场景下的性能影响，并最终为选择最优的增强策略提供数据驱动的决策依据。**

### **<font color="DarkViolet">1.2 核心假设</font>**

本评估方案基于以下两个核心假设：

1. **主要增益假设**：有效的图像增强将主要通过提升**检测阶段**的性能来改善追踪结果。通过增强图像的对比度、亮度和清晰度，检测器能够更准确地定位目标并减少漏检（False Negatives, FN），这将直接体现在\*\*检测准确度（DetA）\*\*指标的提升上 <sup><font color="red">1<color></font></sup>。  
2. **潜在风险假设**：不恰当或过度的增强可能会引入噪声、模糊边缘或扭曲颜色，从而导致检测器产生更多的虚假检测（False Positives, FP），或干扰追踪器基于外观的重识别（Re-ID）模块，最终可能损害\*\*关联准确度（AssA）\*\*或整体性能 <sup><font color="red">8<color></font></sup>。

## **<font color="DodgerBlue">第二部分：实验组件选择</font>**

为了确保实验的有效性和代表性，我们需要精心选择构成实验流程的各个技术组件。

### **<font color="DarkViolet">2.1 待评估的图像增强库（自变量）</font>**

我们选择一组功能多样、技术路线不同的开源图像增强库进行测试。这个组合旨在覆盖从经典图像处理到现代深度学习的多种方法。

**表1：推荐评估的开源图像增强库**

| 库/技术类别 | 核心原理 | 主要实现 | 针对隧道场景的潜在优势 | 需要关注的潜在问题 |  |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **经典基线：OpenCV** | 提供一系列经典的图像处理算法，如直方图均衡化（HE）、自适应直方图均衡化（CLAHE）、伽马校正、锐化和去噪滤波器 <sup><font color="red">9<color></font></sup>。 | Python, C++ 9 | 计算成本极低，易于实现和部署。可作为评估复杂方法附加值的性能基准。 | 效果通常有限，可能无法处理复杂的、非均匀的光照问题。 |  |
| **深度学习：低光照增强** | 基于深度学习（如GAN或Transformer）的模型，专门用于从低光照图像中恢复细节、色彩和对比度 <sup><font color="red">1<color></font></sup>。 | 多个GitHub开源项目，如Illumination-Adaptive-Transformer 12 | 专为解决隧道环境的核心痛点（光线不足）而设计，有望显著提升图像质量。 | 计算开销较大，可能引入伪影，需要仔细评估其对检测器的真实影响。 |  |
| **深度学习：超分辨率与去噪** | 使用深度网络（如RDN, RRDN/ESRGAN）来提升图像分辨率并去除噪声，这些模型通常也具备改善整体图像质量的能力 <sup><font color="red">13<color></font></sup>。 | image-super-resolution (ISR) 13 | 能够处理因传感器或压缩导致的噪声和模糊，使车辆轮廓更清晰。 | 主要为超分设计，在纯增强任务上可能不是最优选择，且计算密集。 |  |
| **深度学习：恶劣天气适应** | 旨在通过可微图像处理（DIP）或生成模型（如Diffusion Models）来移除雾、雨、雪等天气影响，或直接生成增强数据 <sup><font color="red">14<color></font></sup>。 | IA-YOLO 14, | Instruct-Pix2Pix 16 | 提供了应对隧道内可能存在的烟雾或水汽的先进解决方案。 | 实现和训练可能更复杂，实时性是主要挑战。 |

### **<font color="DarkViolet">2.2 固定的目标追踪系统（常量）</font>**

为了隔离图像增强库带来的影响，整个评估过程中必须使用完全相同的一套目标检测和追踪系统。

* **目标检测器**：推荐使用**YOLO系列**（如YOLOv8或YOLOv10）。YOLO是业界公认的兼具高速度和高精度的标准检测器，拥有庞大的社区和丰富的预训练模型 <sup><font color="red">16<color></font></sup>。  
* **目标追踪器**：强烈推荐使用**ByteTrack** <sup><font color="red">18<color></font></sup>。ByteTrack的关联策略（BYTE）通过利用低置信度的检测框来处理遮挡，这使得它的性能直接受益于检测质量的提升 <sup><font color="red">22<color></font></sup>。如果图像增强能让被遮挡的车辆产生一个哪怕是低分的检测框，ByteTrack就有机会将其正确关联，从而减少轨迹断裂。这种机制使得ByteTrack成为检验增强效果的理想“试纸”。

### **<font color="DarkViolet">2.3 评估工具与指标（因变量）</font>**

* **评估工具**：推荐使用 **TrackEval** <sup><font color="red">25<color></font></sup>。它是MOTChallenge等权威基准的官方评估工具，能确保评估结果的公正性和可比性。它支持所有必要的现代追踪指标，并且运行速度快 <sup><font color="red">25<color></font></sup>。  
* **核心评估指标**：  
  * **HOTA (Higher Order Tracking Accuracy)**：作为首要的综合性指标。HOTA通过几何平均数平衡了检测和关联的性能（HOTA=DetA×AssA​），使其成为衡量整体追踪质量最均衡的选择 <sup><font color="red">33<color></font></sup>。  
  * **DetA (Detection Accuracy)**：用于直接量化图像增强对**检测器**性能的影响。这是验证我们核心假设1的关键。  
  * **AssA (Association Accuracy)**：用于衡量图像增强对**关联算法**的间接影响，包括正面（更清晰的特征）和负面（伪影干扰）的效应。  
* **辅助诊断指标**：  
  * **IDF1**：一个以关联为中心的指标，对身份保持的连贯性非常敏感 <sup><font color="red">38<color></font></sup>。  
  * **IDs (ID Switches)**：身份切换的绝对次数，一个非常直观且关键的错误类型 <sup><font color="red">39<color></font></sup>。  
  * **FP / FN**：假阳性和假阴性的绝对数量，用于深入分析DetA变化的原因 <sup><font color="red">25<color></font></sup>。  
  * **FPS (Frames Per Second)**：衡量整个流程（增强+追踪）的处理速度，评估其实时性。

## **<font color="DodgerBlue">第三部分：实验协议</font>**

本部分提供一个清晰、可执行的A/B测试流程。

### **<font color="DarkViolet">3.1 数据准备</font>**

1. **视频数据**：准备好作为输入的隧道连续画面视频。  
2. **真值（Ground Truth）标注**：对视频中的所有车辆进行精确标注，为每个目标在每一帧都分配一个唯一的、持续的ID。  
3. **格式化**：将标注数据整理成TrackEval兼容的**MOTChallenge格式**。每个视频序列需要一个gt.txt文件和一个seqinfo.ini文件，并按标准目录结构存放 <sup><font color="red">26<color></font></sup>。

### **<font color="DarkViolet">3.2 实验流程</font>**

**实验一：基线性能（对照组）**

1. **输入**：原始的、未经任何处理的隧道视频序列。  
2. **处理**：直接将原始视频输入到固定的MOT系统（YOLOv8 \+ ByteTrack）。  
3. **输出**：生成追踪结果文件 results\_baseline.txt。

实验二：增强后性能（实验组）  
此实验需要为每一个待评估的增强库重复进行。以“增强库A”为例：

1. **输入**：原始的隧道视频序列。  
2. **预处理**：使用“增强库A”对原始视频的每一帧进行处理，生成一个新的、增强后的视频序列。  
3. **处理**：将**增强后**的视频输入到**完全相同**的MOT系统（YOLOv8 \+ ByteTrack）。  
4. **输出**：生成追踪结果文件 results\_enhancement\_A.txt。  
5. 对“增强库B”、“增强库C”等重复以上步骤，得到各自的结果文件。

### **<font color="DarkViolet">3.3 性能评估</font>**

1. **执行评估**：使用TrackEval工具，将每个实验（基线和所有增强实验）生成的追踪结果文件与真值（Ground Truth）进行比较。  
   Bash  
   \# 示例命令  
   python scripts/run\_mot\_challenge.py \--BENCHMARK TunnelData \--METRICS HOTA CLEAR Identity \--TRACKERS\_TO\_EVAL baseline enhancement\_A enhancement\_B

2. **数据收集**：TrackEval将为每个实验条件生成一套完整的性能指标。

## **<font color="DodgerBlue">第四部分：结果分析与决策</font>**

### **<font color="DarkViolet">4.1 定量分析</font>**

创建一个汇总表格，清晰地对比所有实验条件下的关键指标。

**表2：评估结果汇总与分析示例**

| 实验条件 | HOTA (↑) | Δ HOTA | DetA (↑) | Δ DetA | AssA (↑) | Δ AssA | IDs (↓) | FP (↓) | FN (↓) | FPS (↑) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **基线 (无增强)** | 60.2 | \- | 65.0 | \- | 55.8 | \- | 150 | 2000 | 5000 | 30 |
| **增强库A (CLAHE)** | 62.5 | \+2.3 | 68.1 | \+3.1 | 57.5 | \+1.7 | 140 | 2100 | 4500 | 28 |
| **增强库B (低光照DL)** | 65.8 | \+5.6 | 72.0 | \+7.0 | 60.1 | \+4.3 | 125 | 1900 | 3800 | 15 |
| **增强库C (过度锐化)** | 59.1 | \-1.1 | 64.5 | \-0.5 | 54.0 | \-1.8 | 165 | 2500 | 5100 | 27 |

**分析要点**：

1. **整体效果**：首先根据**HOTA**得分对所有增强库进行排名。得分最高的库是综合性能最佳的。  
2. **效果归因**：  
   * 观察**Δ DetA**（DetA的变化量）。一个大的正向Δ DetA强有力地证明了该增强库有效改善了检测器的性能，这是最理想的情况（如示例中的增强库B）。  
   * 观察**Δ AssA**。如果Δ AssA也为正，说明增强后的图像特征对关联也有帮助。如果为负，则可能引入了干扰（如示例中的增强库C，过度锐化可能产生了更多噪点，导致FP增加和关联错误）。  
3. **错误类型分析**：通过对比FP和FN的绝对数量变化，可以更深入地理解DetA变化的原因。例如，一个好的增强应该显著降低FN（漏检），而不过度增加FP（误检）。

### **<font color="DarkViolet">4.2 成本效益分析</font>**

性能提升并非没有代价。必须将每个增强库引入的额外计算成本（即FPS的下降）纳入考量。一个将HOTA提升1%但使处理速度减半的库，可能在实时应用中并不可取。

### **<font color="DarkViolet">4.3 定性分析</font>**

除了数字，还应进行可视化检查。随机抽取一些关键帧，对比原始图像、增强后图像以及追踪结果。

* **成功案例**：寻找那些增强后成功追踪，但在原始图像中失败的案例。这能直观地展示增强的价值。  
* **失败案例**：寻找那些增强后反而追踪失败或产生错误（如FP、IDs）的案例。这有助于理解增强算法的局限性和副作用 <sup><font color="red">41<color></font></sup>。

### **<font color="DarkViolet">4.4 最终建议</font>**

最终的决策应基于对以下三个维度的综合权衡：

1. **追踪性能提升**：以HOTA为主要衡量标准，DetA为关键诊断依据。  
2. **计算开销**：以FPS为衡量标准，评估其是否满足实际应用场景的实时性要求。  
3. **鲁棒性**：通过定性分析，评估增强算法是否会在特定情况下引入灾难性的失败模式。

通过执行这一套完整的评估方案，您将能够超越“感觉上更好”的主观判断，获得坚实的量化证据，从而科学地选择最适合您隧道车辆追踪任务的图像增强库。

#### **<font color="LightSeaGreen">引用的资料</font>**

1. Two-stage object detection in low-light environments using deep learning image enhancement \- PMC, 访问时间为 八月 5, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC12190514/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12190514/)  
2. Review of AI Image Enhancement Techniques for In-Vehicle Vision Systems Under Adverse Weather Conditions \- Bohrium, 访问时间为 八月 5, 2025， [https://www.bohrium.com/paper-details/review-of-ai-image-enhancement-techniques-for-in-vehicle-vision-systems-under-adverse-weather-conditions/1100021575643562017-57802](https://www.bohrium.com/paper-details/review-of-ai-image-enhancement-techniques-for-in-vehicle-vision-systems-under-adverse-weather-conditions/1100021575643562017-57802)  
3. Dynamic Low-Light Image Enhancement for Object Detection Via End-To-End Training, 访问时间为 八月 5, 2025， [https://wuyirui.github.io/papers/ICPR2020-01.pdf](https://wuyirui.github.io/papers/ICPR2020-01.pdf)  
4. Image Enhancement Guided Object Detection in Visually Degraded Scenes \- PubMed, 访问时间为 八月 5, 2025， [https://pubmed.ncbi.nlm.nih.gov/37220059/](https://pubmed.ncbi.nlm.nih.gov/37220059/)  
5. A Study of Image Pre-processing for Faster Object Recognition \- arXiv, 访问时间为 八月 5, 2025， [https://arxiv.org/pdf/2011.06928](https://arxiv.org/pdf/2011.06928)  
6. A Study of Image Pre-processing for Faster Object Recognition \- ResearchGate, 访问时间为 八月 5, 2025， [https://www.researchgate.net/publication/345915258\_A\_Study\_of\_Image\_Pre-processing\_for\_Faster\_Object\_Recognition](https://www.researchgate.net/publication/345915258_A_Study_of_Image_Pre-processing_for_Faster_Object_Recognition)  
7. Are Poor Object Detection Results On Enhanced Images Due to Missing Human Labels?, 访问时间为 八月 5, 2025， [https://openaccess.thecvf.com/content/WACV2025W/MaCVi/papers/Lucas\_Underwater\_Image\_Enhancement\_and\_Object\_Detection\_Are\_Poor\_Object\_Detection\_WACVW\_2025\_paper.pdf](https://openaccess.thecvf.com/content/WACV2025W/MaCVi/papers/Lucas_Underwater_Image_Enhancement_and_Object_Detection_Are_Poor_Object_Detection_WACVW_2025_paper.pdf)  
8. Understanding the Influence of Image Enhancement on Underwater Object Detection: A Quantitative and Qualitative Study \- MDPI, 访问时间为 八月 5, 2025， [https://www.mdpi.com/2072-4292/17/2/185](https://www.mdpi.com/2072-4292/17/2/185)  
9. Free and Open-Source Computer Vision Tools | by ODSC \- Open Data Science | Jul, 2025, 访问时间为 八月 5, 2025， [https://odsc.medium.com/free-and-open-source-computer-vision-tools-89414fa92dc9](https://odsc.medium.com/free-and-open-source-computer-vision-tools-89414fa92dc9)  
10. OpenCV \- Open Computer Vision Library, 访问时间为 八月 5, 2025， [https://opencv.org/](https://opencv.org/)  
11. Introduction to Object Detection Using Image Processing \- GeeksforGeeks, 访问时间为 八月 5, 2025， [https://www.geeksforgeeks.org/deep-learning/introduction-to-object-detection-using-image-processing/](https://www.geeksforgeeks.org/deep-learning/introduction-to-object-detection-using-image-processing/)  
12. low-light-image-enhancement · GitHub Topics, 访问时间为 八月 5, 2025， [https://github.com/topics/low-light-image-enhancement](https://github.com/topics/low-light-image-enhancement)  
13. idealo/image-super-resolution: Super-scale your images and run experiments with Residual Dense and Adversarial Networks. \- GitHub, 访问时间为 八月 5, 2025， [https://github.com/idealo/image-super-resolution](https://github.com/idealo/image-super-resolution)  
14. Adaptive image enhancement technology based on bad weather \- SPIE Digital Library, 访问时间为 八月 5, 2025， [https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13269/1326904/Adaptive-image-enhancement-technology-based-on-bad-weather/10.1117/12.3045552.full](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13269/1326904/Adaptive-image-enhancement-technology-based-on-bad-weather/10.1117/12.3045552.full)  
15. Robust Object Detection in Challenging Weather Conditions \- CVF Open Access, 访问时间为 八月 5, 2025， [https://openaccess.thecvf.com/content/WACV2024/papers/Gupta\_Robust\_Object\_Detection\_in\_Challenging\_Weather\_Conditions\_WACV\_2024\_paper.pdf](https://openaccess.thecvf.com/content/WACV2024/papers/Gupta_Robust_Object_Detection_in_Challenging_Weather_Conditions_WACV_2024_paper.pdf)  
16. Object detection in adverse weather conditions for autonomous vehicles using Instruct Pix2Pix. \- arXiv, 访问时间为 八月 5, 2025， [https://arxiv.org/html/2505.08228v2](https://arxiv.org/html/2505.08228v2)  
17. Top 5 Open-Source Computer Vision Models \- Unitlab Blogs, 访问时间为 八月 5, 2025， [https://blog.unitlab.ai/top-5-open-source-computer-vision-models/](https://blog.unitlab.ai/top-5-open-source-computer-vision-models/)  
18. ByteTrack: Multi-Object Tracking by Associating Every Detection Box, 访问时间为 八月 5, 2025， [https://www.ecva.net/papers/eccv\_2022/papers\_ECCV/papers/136820001.pdf](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf)  
19. \[2110.06864\] ByteTrack: Multi-Object Tracking by Associating Every Detection Box \- ar5iv, 访问时间为 八月 5, 2025， [https://ar5iv.labs.arxiv.org/html/2110.06864](https://ar5iv.labs.arxiv.org/html/2110.06864)  
20. \[ECCV 2022\] ByteTrack: Multi-Object Tracking by Associating Every Detection Box \- GitHub, 访问时间为 八月 5, 2025， [https://github.com/FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack)  
21. \[2303.15334\] ByteTrackV2: 2D and 3D Multi-Object Tracking by Associating Every Detection Box \- arXiv, 访问时间为 八月 5, 2025， [https://arxiv.org/abs/2303.15334](https://arxiv.org/abs/2303.15334)  
22. Comparison of BYTE and DeepSORT using light detec- tion models on the MOT17 validation set. \- ResearchGate, 访问时间为 八月 5, 2025， [https://www.researchgate.net/figure/Comparison-of-BYTE-and-DeepSORT-using-light-detec-tion-models-on-the-MOT17-validation\_tbl3\_355237366](https://www.researchgate.net/figure/Comparison-of-BYTE-and-DeepSORT-using-light-detec-tion-models-on-the-MOT17-validation_tbl3_355237366)  
23. (PDF) ByteTrack: Multi-Object Tracking by Associating Every Detection Box \- ResearchGate, 访问时间为 八月 5, 2025， [https://www.researchgate.net/publication/355237366\_ByteTrack\_Multi-Object\_Tracking\_by\_Associating\_Every\_Detection\_Box](https://www.researchgate.net/publication/355237366_ByteTrack_Multi-Object_Tracking_by_Associating_Every_Detection_Box)  
24. ByteTrack: Multi-Object Tracking by Associating Every Detection Box | Luffca, 访问时间为 八月 5, 2025， [https://www.luffca.com/2023/06/multiple-object-tracking-bytetrack/](https://www.luffca.com/2023/06/multiple-object-tracking-bytetrack/)  
25. JonathonLuiten/TrackEval: HOTA (and other) evaluation ... \- GitHub, 访问时间为 八月 5, 2025， [https://github.com/JonathonLuiten/TrackEval](https://github.com/JonathonLuiten/TrackEval)  
26. sn-trackeval/docs/MOTChallenge-Official/Readme.md at main \- GitHub, 访问时间为 八月 5, 2025， [https://github.com/SoccerNet/sn-trackeval/blob/main/docs/MOTChallenge-Official/Readme.md](https://github.com/SoccerNet/sn-trackeval/blob/main/docs/MOTChallenge-Official/Readme.md)  
27. TrackingLaboratory \- GitHub, 访问时间为 八月 5, 2025， [https://github.com/TrackingLaboratory](https://github.com/TrackingLaboratory)  
28. nekorobov/HOTA-metrics: HOTA (and other) evaluation metrics for Multi-Object Tracking (MOT). \- GitHub, 访问时间为 八月 5, 2025， [https://github.com/nekorobov/HOTA-metrics](https://github.com/nekorobov/HOTA-metrics)  
29. How to generate evaluation metrics for tracking, such as MOTA, IDF1, HOTA, etc. ? · Issue \#8142 \- GitHub, 访问时间为 八月 5, 2025， [https://github.com/ultralytics/ultralytics/issues/8142](https://github.com/ultralytics/ultralytics/issues/8142)  
30. 30-A/trackeval\_lite: Evaluation metrics for Multi-Object Tracking (MOT). \- GitHub, 访问时间为 八月 5, 2025， [https://github.com/30-A/trackeval\_lite](https://github.com/30-A/trackeval_lite)  
31. trackeval · GitHub Topics, 访问时间为 八月 5, 2025， [https://github.com/topics/trackeval](https://github.com/topics/trackeval)  
32. dvl-tum/TrackEvalForGHOST: Adaption of TrackEval code for half validation split. \- GitHub, 访问时间为 八月 5, 2025， [https://github.com/dvl-tum/TrackEvalForGHOST](https://github.com/dvl-tum/TrackEvalForGHOST)  
33. How to evaluate tracking with the HOTA metrics \- Autonomous ..., 访问时间为 八月 5, 2025， [https://autonomousvision.github.io/hota-metrics/](https://autonomousvision.github.io/hota-metrics/)  
34. HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking \- Andreas Geiger, 访问时间为 八月 5, 2025， [https://www.cvlibs.net/publications/Luiten2020IJCV.pdf](https://www.cvlibs.net/publications/Luiten2020IJCV.pdf)  
35. Introduction to Tracker KPI. Multi-Object Tracking (MOT) is the task… | by Sadbodh Sharma | Digital Engineering @ Centific | Medium, 访问时间为 八月 5, 2025， [https://medium.com/digital-engineering-centific/introduction-to-tracker-kpi-6aed380dd688](https://medium.com/digital-engineering-centific/introduction-to-tracker-kpi-6aed380dd688)  
36. A simple tracking example highlighting one of the main differences... \- ResearchGate, 访问时间为 八月 5, 2025， [https://www.researchgate.net/figure/A-simple-tracking-example-highlighting-one-of-the-main-differences-between-evaluation\_fig1\_345343240](https://www.researchgate.net/figure/A-simple-tracking-example-highlighting-one-of-the-main-differences-between-evaluation_fig1_345343240)  
37. Understanding Object Tracking Metrics \- Miguel Mendez, 访问时间为 八月 5, 2025， [https://miguel-mendez-ai.com/2024/08/25/mot-tracking-metrics](https://miguel-mendez-ai.com/2024/08/25/mot-tracking-metrics)  
38. CSMOT: Make One-Shot Multi-Object Tracking in Crowded Scenes Great Again \- PMC, 访问时间为 八月 5, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC10098982/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10098982/)  
39. Introduction to Multiple Object Tracking and Recent Developments \- Datature, 访问时间为 八月 5, 2025， [https://www.datature.io/blog/introduction-to-multiple-object-tracking-and-recent-developments](https://www.datature.io/blog/introduction-to-multiple-object-tracking-and-recent-developments)  
40. microsoft.github.io, 访问时间为 八月 5, 2025， [https://microsoft.github.io/computervision-recipes/scenarios/tracking/FAQ.html\#:\~:text=ID%2Dswitch%20measures%20when%20the,tracked%20in%20frames%204%2D5.](https://microsoft.github.io/computervision-recipes/scenarios/tracking/FAQ.html#:~:text=ID%2Dswitch%20measures%20when%20the,tracked%20in%20frames%204%2D5.)  
41. A Comprehensive Study of Object Tracking in Low-Light Environments \- PMC, 访问时间为 八月 5, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC11244102/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11244102/)