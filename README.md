# <div align="center">YOLOv3 & Random Forest aplicadas no monitoramento de equipamentos em subestações de energia elétrica</div>

<p>
Com o constante crescimento do sistema elétrico, se torna cada vez mais importante a eficiência no método de execução das manutenções nesses ambientes. A inspeção termográfica é uma dessas alternativas, pois é eficaz para a previsão de falhas nos equipamentos das subestações de energia. Porém, a segurança dos operadores e a necessidade de um grau de experiência elevado, faz com que a atividade se torne muito dispendiosa. A ideia central do presente trabalho é propor uma automatização parcial do processo das inspeções termográficas de subestações de energia elétrica utilizando a YOLOv3 e o ensemble floresta aleatória para detecção de equipamentos presentes em subestação de energia elétrica. Utilizando um banco de dados de imagens de chaves seccionadoras de uma subestação de energia elétrica, a tecnologia pode identificar e segmentar regiões de aquecimento de maneira satisfatória os equipamentos e com isso pode auxiliar as concessionárias de energia a tomarem decisões referentes à manutenção preditiva. A rede neural alcançou níveis de precisão acima dos 70%, o que mostra um desempenho satisfatório.
</p>

## <div>Introdução</div>
<p>
As subestações de energia elétrica (SE) constituem uma peça fundamental do sistema elétrico de potência, sendo responsáveis pela operação segura e confiável da rede elétrica (MAMEDE FILHO, 2021). A correta operação e manutenção dos equipamentos que compõem as subestações é tão importante quanto a expansão do sistema elétrico, o que permite manter sua eficiência alinhada com as necessidades crescentes dos consumidores.
</p>
Algumas falhas que ocorrem em equipamentos de subestações estão geralmente associadas a uma elevação anormal da sua temperatura de trabalho. Essas anomalias térmicas podem ser detectadas por meio de inspeção termográfica, cujo processo realiza o apontamento com precisão de pontos sobreaquecidos dos equipamentos defeituosos, possibilitando quantificar esse aumento irregular de temperatura (WANDERLEY NETO, 2007). Segunda a norma, o intervalo recomendado entre as inspeções termográficas para sistemas elétricos, é de 6 (seis) meses, não devendo ultrapassar 18 meses, caso haja a impossibilidade de cumprir a recomendação.
</p>
Em via disso, torna-se estratégico para as concessionárias de energia elétrica disporem de ferramentas que possibilitem a detecção inteligente e automática de equipamentos com falhas em subestações, promovendo a substituição das inspeções tradicionais de patrulha com alto coeficiente de risco e baixa eficiência.
</p>

## <div>YOLO: detecção de objetos em tempo real</div>
<p>
Proposta por Redmon et al. (2016), a rede YOLO utiliza uma única rede neural convolucional na detecção e classificação de objetos. A popularidade deste método se deu pela sua alta velocidade de processamento e precisão nos resultados obtidos. A YOLO usa uma única CNN para prever as caixas delimitadoras e a probabilidade de classe para objetos detectados em uma determinada imagem de entrada. Uma única arquitetura é responsável por detectar e localizar os objetos em uma imagem, permitindo que a YOLO tenha um bom desempenho em situações de tempo real em comparação com os métodos mais antigos.
</p>
Em sua terceira versão (Redmon e Farhadi, 2018), a YOLO funciona por meio da divisão da imagem de entrada em uma grade maior de células, contendo um número fixo de “caixas de âncora” (anchor boxes) para cada célula. Cada caixa de âncora corresponde a formas pré-definidas de caixas delimitadoras que foram previamente calculadas de acordo com os objetos do conjunto de treinamento. Por exemplo, uma anchor box para o objeto “carro” terá um formato de caixa em paisagem, devido a frequência com que caixas horizontais contendo “carros” aparecem durante o treinamento da rede.
</p>
Na YOLOv3, a grade padrão é uma matriz com 13 linhas e 13 colunas, correspondendo a 169 células, onde cada célula possui cinco âncoras, totalizando 845 previsões de possíveis de caixas delimitadoras. Cada caixa delimitadora é definida a partir de duas coordenadas relativas à matriz da imagem, correspondendo a posição central do objeto (“x” e “y”) e as duas dimensões de largura (“w”) e altura (“h”). A rede neural será capaz de prever objetos em áreas específicas da imagem a partir das âncoras distribuídas em cada célula.
</p>

</a><div align="center">
<a href="https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/">
<img src="https://github.com/Rhayron/YOLOv3_detection_segmentation/blob/main/assets/dog.png" width="27%"/>
</div></a>

Considerando o exemplo da Figura, a imagem de entrada é dividia em uma grade de células 13 x 13. Em seguida, a célula (na imagem de entrada) contendo o centro da caixa verdade, chamada de ground truth (caixa que realmente contém o objeto), é escolhida para ser a responsável pela previsão. Na Figura, a célula marcada em vermelho contém o centro da caixa ground truth, marcada em amarela, e será a responsável pela detecção do cachorro. Esta célula pode prever três caixas delimitadoras, ou seja, a YOLOv3 tem três âncoras, que resultam em previsão de três caixas delimitadoras por célula. A bounding box responsável por detectar o cão será aquela cuja âncora tem a maior IoU com a caixa verdade.
</p>
O valor das coordenadas das caixas prevista tx, ty..th são normalizados, valores entre 0 e 1. A YOLO prevê deslocamentos das coordenadas do centro da caixa delimitação em relação ao canto superior esquerdo da célula de grade que está prevendo o objeto, normalizado pelas dimensões da célula. Para o caso da Figura abaixo, se a previsão para o centro é (0,4, 0,7), então isso significa que o centro está em (6,4, 6,7) na grade 13 x 13, considerando que as coordenadas superior esquerda da célula vermelha são (6,6). O valor de B representa o número de caixas delimitadoras que cada célula pode prever. Em outras palavras, cada uma dessas caixas delimitadoras B pode se especializar na detecção de um certo tipo de objeto.
</p>

</a><div align="center">
<a href="https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/">
<img src="https://github.com/Rhayron/YOLOv3_detection_segmentation/blob/main/assets/HowYoloWorks.png" width="50%"/>
</div></a>

As probabilidades de classe representam as chances do objeto detectado ser pertencente a uma determinada classe (cão, gato, carro, chaves seccionadoras, etc). A função de perda na rede leva em consideração a pontuação de objetividade, a classificação de objetos e a regressão das coordenadas que está relacionada às dimensões da caixa delimitadora.
</p>
Dessa forma a rede neural YOLO filtra as previsões através da definição de uma pontuação de objetividade mínima (limite inferior). Além disso, durante o treinamento, uma técnica para filtrar as previsões do detector de objetos chamada Non-maximum Suppression (NMS)[ Non-Maximum Suppression (ou supressão não máxima) é uma técnica usada em várias tarefas de visão computacional. É uma classe de algoritmos utilizadas para selecionar uma entidade (por exemplo, caixas delimitadoras) de muitas entidades sobrepostas, segundo alguns critérios de seleção. Os critérios comumente utilizados são alguma forma de medida de sobreposição (por exemplo, Intersecção sobre União - IoU).] é aplicado para remover previsões redundantes. As previsões que correspondem ao mesmo objeto são comparadas e apenas aquelas com a maior confiança são mantidas, conforme mostrado na Figura abaixo.
</p>

</a><div width="2%" align="center">
<a href="https://openaccess.thecvf.com/content_cvpr_2017/html/Hosang_Learning_Non-Maximum_Suppression_CVPR_2017_paper.html">
<img src="https://github.com/Rhayron/YOLOv3_detection_segmentation/blob/main/assets/NMS.png" width="30%"/>
</div></a>

Para cada uma das versões da YOLO, os autores também lançaram uma variante chamada “YOLO tiny”. As versões tiny são menores em comparação com as versões originais no número de camadas. Apesar de ser uma rede mais simples, são mais rápidas do que as versões originais, no entanto são menos precisas. Assim, as variantes tiny da YOLO têm sido frequentemente utilizadas em aplicações com o objetivo de se obter maior velocidade de processamento em troca de menor precisão nas detecções (LAROCA et al., 2019).


### <div>YOLOv3</div>

<p>
Redmon e Farhadi (2018) lançaram a YOLOv3 como uma rede composta por 106 camadas, 53 para o backbone (“darknet-53”) e as outras 53 camadas responsáveis pela de detecção de objetos, mantendo a característica de ser uma rede neural totalmente convolucional.
</p>
Em comparação com YOLOv2, os autores aplicaram algumas alterações como classificação multi-rótulo, ou seja, um objeto passou a poder ser anexado a mais de uma classe (por exemplo, "árvore" e "pinheiro"), três escalas diferentes de previsão e aumento na quantidade de caixas de âncoras, o que, consequentemente, aumentou o número de caixas previstas para uma dada imagem de entrada.
</p>
Essa abordagem melhora a precisão da rede na detecção de pequenos objetos. Com essa nova configuração a imagem de entrada pode ser dividida em uma grade 13 × 13 para detectar objetos grandes, uma grade de células 26 × 26 para a detecção de objetos médios e uma grade 52 × 52 para os pequenos objetos.
</p>
Em vez de 5 âncoras por célula, a quantidade foi aumentada para 9, onde há 3 âncoras para cada escala. Enquanto o YOLOv2 pode prever 845 caixas delimitadoras, esta nova configuração permite que a YOLO preveja 10.647 caixas para cada imagem. O aumento no número de caixas previstas aliada ao aumento no número de camadas da rede, tornou a YOLOv3 mais lenta, sendo necessário maiores cronogramas de treinamento. No entanto, ela ainda é mais rápida que os principais concorrentes (Faster R-CNN, R-CNN, entre outros), como releva o Gráfico a seguir.

</a><div width="2%" align="center">
<a href="https://arxiv.org/abs/1804.02767">
<img src="https://github.com/Rhayron/YOLOv3_detection_segmentation/blob/main/assets/YOLOv3_peformance.png" width="60%"/>
</div></a>

A YOLO v3 funciona no mesmo nível de outros detectores de última geração, como a RetinaNet, embora seja consideravelmente mais rápido, no benchmark COCO mAP 50. No entanto, a YOLO perde em benchmarks COCO mAP-50 com um valor mais alto de IoU, usado para filtrar o número de caixas detectadas.
</p>

## <div>Random Forest</div>
Florestas aleatórias (RF – Random Forest) de Breiman (2001) é um algoritmo de aprendizado de máquina supervisionado, onde são empregados para aprender uma função que combina um conjunto de variáveis, com o objetivo de prever uma outra variável.
</p>
Dependendo do tipo das variáveis dependentes, os algoritmos de aprendizagem supervisionados podem ser classificados em algoritmos de regressão e classificação. Nos algoritmos de regressão, a variável dependente é quantitativa, enquanto nos algoritmos de classificação, a variável dependente é qualitativa (Hastie et al. pp. 9-11, 2015).
</p>

### <div>Árvores de classificação e regressão</div>
Árvores de classificação e regressão (CARTs - Classification and Regression Trees) são métodos para particionar o espaço de variáveis de entrada com base em um conjunto de regras em uma árvore de decisão, onde cada nó se divide de acordo com uma regra de decisão (como exemplificado na Figura abaixo). Desta forma, o espaço variável é particionado em subconjuntos e o modelo é ajustado a cada subconjunto.
</p>

</a><div width="2%" align="center">
<img src="https://github.com/Rhayron/YOLOv3_detection_segmentation/blob/main/assets/CART.png" width="60%"/>
</div></a>

</p>
YOLOv3 🚀 is a family of object detection architectures and models pretrained on the COCO dataset, and represents <a href="https://ultralytics.com">Ultralytics</a>
 open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.
</p>

