# <div align="center">YOLOv3 & Random Forest aplicadas no monitoramento de equipamentos em subestações de energia elétrica</div>

<a href="https://ultralytics.com">Ultralytics</a>

Com o constante crescimento do sistema elétrico, se torna cada vez mais importante a eficiência no método de execução das manutenções nesses ambientes. A inspeção termográfica é uma dessas alternativas, pois é eficaz para a previsão de falhas nos equipamentos das subestações de energia. Porém, a segurança dos operadores e a necessidade de um grau de experiência elevado, faz com que a atividade se torne muito dispendiosa. A ideia central do presente trabalho é propor uma automatização parcial do processo das inspeções termográficas de subestações de energia elétrica utilizando a YOLOv3 e o ensemble floresta aleatória para detecção de equipamentos presentes em subestação de energia elétrica. Utilizando um banco de dados de imagens de chaves seccionadoras de uma subestação de energia elétrica, a tecnologia pode identificar e segmentar regiões de aquecimento de maneira satisfatória os equipamentos e com isso pode auxiliar as concessionárias de energia a tomarem decisões referentes à manutenção preditiva. A rede neural alcançou níveis de precisão acima dos 70%, o que mostra um desempenho satisfatório.

## <div>Introdução</div>

As subestações de energia elétrica (SE) constituem uma peça fundamental do sistema elétrico de potência, sendo responsáveis pela operação segura e confiável da rede elétrica (MAMEDE FILHO, 2021). A correta operação e manutenção dos equipamentos que compõem as subestações é tão importante quanto a expansão do sistema elétrico, o que permite manter sua eficiência alinhada com as necessidades crescentes dos consumidores.

Algumas falhas que ocorrem em equipamentos de subestações estão geralmente associadas a uma elevação anormal da sua temperatura de trabalho. Essas anomalias térmicas podem ser detectadas por meio de inspeção termográfica, cujo processo realiza o apontamento com precisão de pontos sobreaquecidos dos equipamentos defeituosos, possibilitando quantificar esse aumento irregular de temperatura (WANDERLEY NETO, 2007). Segunda a norma, o intervalo recomendado entre as inspeções termográficas para sistemas elétricos, é de 6 (seis) meses, não devendo ultrapassar 18 meses, caso haja a impossibilidade de cumprir a recomendação.

Em via disso, torna-se estratégico para as concessionárias de energia elétrica disporem de ferramentas que possibilitem a detecção inteligente e automática de equipamentos com falhas em subestações, promovendo a substituição das inspeções tradicionais de patrulha com alto coeficiente de risco e baixa eficiência.

## <div>YOLO: detecção de objetos em tempo real</div>

Proposta por Redmon et al. (2016), a rede YOLO utiliza uma única rede neural convolucional na detecção e classificação de objetos. A popularidade deste método se deu pela sua alta velocidade de processamento e precisão nos resultados obtidos. A YOLO usa uma única CNN para prever as caixas delimitadoras e a probabilidade de classe para objetos detectados em uma determinada imagem de entrada. Uma única arquitetura é responsável por detectar e localizar os objetos em uma imagem, permitindo que a YOLO tenha um bom desempenho em situações de tempo real em comparação com os métodos mais antigos.

Em sua terceira versão (Redmon e Farhadi, 2018), a YOLO funciona por meio da divisão da imagem de entrada em uma grade maior de células, contendo um número fixo de “caixas de âncora” (anchor boxes) para cada célula. Cada caixa de âncora corresponde a formas pré-definidas de caixas delimitadoras que foram previamente calculadas de acordo com os objetos do conjunto de treinamento. Por exemplo, uma anchor box para o objeto “carro” terá um formato de caixa em paisagem, devido a frequência com que caixas horizontais contendo “carros” aparecem durante o treinamento da rede.

Na YOLOv3, a grade padrão é uma matriz com 13 linhas e 13 colunas, correspondendo a 169 células, onde cada célula possui cinco âncoras, totalizando 845 previsões de possíveis de caixas delimitadoras. Cada caixa delimitadora é definida a partir de duas coordenadas relativas à matriz da imagem, correspondendo a posição central do objeto (“x” e “y”) e as duas dimensões de largura (“w”) e altura (“h”). A rede neural será capaz de prever objetos em áreas específicas da imagem a partir das âncoras distribuídas em cada célula.

</a><div align="center">
<a href="https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/">
<img src="https://github.com/Rhayron/YOLOv3_detection_segmentation/blob/main/assets/dog.png" width="27%"/>
</div></a>

Considerando o exemplo da Figura, a imagem de entrada é dividia em uma grade de células 13 x 13. Em seguida, a célula (na imagem de entrada) contendo o centro da caixa verdade, chamada de ground truth (caixa que realmente contém o objeto), é escolhida para ser a responsável pela previsão. Na Figura, a célula marcada em vermelho contém o centro da caixa ground truth, marcada em amarela, e será a responsável pela detecção do cachorro. Esta célula pode prever três caixas delimitadoras, ou seja, a YOLOv3 tem três âncoras, que resultam em previsão de três caixas delimitadoras por célula. A bounding box responsável por detectar o cão será aquela cuja âncora tem a maior IoU com a caixa verdade.

O valor das coordenadas das caixas prevista tx, ty..th são normalizados, valores entre 0 e 1. A YOLO prevê deslocamentos das coordenadas do centro da caixa delimitação em relação ao canto superior esquerdo da célula de grade que está prevendo o objeto, normalizado pelas dimensões da célula. Para o caso da Figura abaixo, se a previsão para o centro é (0,4, 0,7), então isso significa que o centro está em (6,4, 6,7) na grade 13 x 13, considerando que as coordenadas superior esquerda da célula vermelha são (6,6). O valor de B representa o número de caixas delimitadoras que cada célula pode prever. Em outras palavras, cada uma dessas caixas delimitadoras B pode se especializar na detecção de um certo tipo de objeto.


</a><div align="center">
<a href="https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/">
<img src="https://github.com/Rhayron/YOLOv3_detection_segmentation/blob/main/assets/HowYoloWorks.png" width="50%"/>
</div></a>

As probabilidades de classe representam as chances do objeto detectado ser pertencente a uma determinada classe (cão, gato, carro, chaves seccionadoras, etc). A função de perda na rede leva em consideração a pontuação de objetividade, a classificação de objetos e a regressão das coordenadas que está relacionada às dimensões da caixa delimitadora.

Dessa forma a rede neural YOLO filtra as previsões através da definição de uma pontuação de objetividade mínima (limite inferior). Além disso, durante o treinamento, uma técnica para filtrar as previsões do detector de objetos chamada Non-maximum Suppression (NMS)[ Non-Maximum Suppression (ou supressão não máxima) é uma técnica usada em várias tarefas de visão computacional. É uma classe de algoritmos utilizadas para selecionar uma entidade (por exemplo, caixas delimitadoras) de muitas entidades sobrepostas, segundo alguns critérios de seleção. Os critérios comumente utilizados são alguma forma de medida de sobreposição (por exemplo, Intersecção sobre União - IoU).] é aplicado para remover previsões redundantes. As previsões que correspondem ao mesmo objeto são comparadas e apenas aquelas com a maior confiança são mantidas, conforme mostrado na Figura abaixo.

</a><div width="2%" align="center">
<a href="https://openaccess.thecvf.com/content_cvpr_2017/html/Hosang_Learning_Non-Maximum_Suppression_CVPR_2017_paper.html">
<img src="https://github.com/Rhayron/YOLOv3_detection_segmentation/blob/main/assets/NMS.png" width="30%"/>
</div></a>

Para cada uma das versões da YOLO, os autores também lançaram uma variante chamada “YOLO tiny”. As versões tiny são menores em comparação com as versões originais no número de camadas. Apesar de ser uma rede mais simples, são mais rápidas do que as versões originais, no entanto são menos precisas. Assim, as variantes tiny da YOLO têm sido frequentemente utilizadas em aplicações com o objetivo de se obter maior velocidade de processamento em troca de menor precisão nas detecções (LAROCA et al., 2019).

### <div>YOLOv3</div>

Redmon e Farhadi (2018) lançaram a YOLOv3 como uma rede composta por 106 camadas, 53 para o backbone (“darknet-53”) e as outras 53 camadas responsáveis pela de detecção de objetos, mantendo a característica de ser uma rede neural totalmente convolucional.

Em comparação com YOLOv2, os autores aplicaram algumas alterações como classificação multi-rótulo, ou seja, um objeto passou a poder ser anexado a mais de uma classe (por exemplo, "árvore" e "pinheiro"), três escalas diferentes de previsão e aumento na quantidade de caixas de âncoras, o que, consequentemente, aumentou o número de caixas previstas para uma dada imagem de entrada.

Essa abordagem melhora a precisão da rede na detecção de pequenos objetos. Com essa nova configuração a imagem de entrada pode ser dividida em uma grade 13 × 13 para detectar objetos grandes, uma grade de células 26 × 26 para a detecção de objetos médios e uma grade 52 × 52 para os pequenos objetos.

Em vez de 5 âncoras por célula, a quantidade foi aumentada para 9, onde há 3 âncoras para cada escala. Enquanto o YOLOv2 pode prever 845 caixas delimitadoras, esta nova configuração permite que a YOLO preveja 10.647 caixas para cada imagem. O aumento no número de caixas previstas aliada ao aumento no número de camadas da rede, tornou a YOLOv3 mais lenta, sendo necessário maiores cronogramas de treinamento. No entanto, ela ainda é mais rápida que os principais concorrentes (Faster R-CNN, R-CNN, entre outros), como releva o Gráfico a seguir.

</a><div width="2%" align="center">
<a href="https://arxiv.org/abs/1804.02767">
<img src="https://github.com/Rhayron/YOLOv3_detection_segmentation/blob/main/assets/YOLOv3_peformance.png" width="60%"/>
</div></a>

A YOLO v3 funciona no mesmo nível de outros detectores de última geração, como a RetinaNet, embora seja consideravelmente mais rápido, no benchmark COCO mAP 50. No entanto, a YOLO perde em benchmarks COCO mAP-50 com um valor mais alto de IoU, usado para filtrar o número de caixas detectadas.

## <div>Random Forest</div>
Florestas aleatórias (RF – Random Forest) de Breiman (2001) é um algoritmo de aprendizado de máquina supervisionado, onde são empregados para aprender uma função que combina um conjunto de variáveis, com o objetivo de prever uma outra variável.
</p>
Dependendo do tipo das variáveis dependentes, os algoritmos de aprendizagem supervisionados podem ser classificados em algoritmos de regressão e classificação. Nos algoritmos de regressão, a variável dependente é quantitativa, enquanto nos algoritmos de classificação, a variável dependente é qualitativa (Hastie et al. pp. 9-11, 2015).

### <div>Árvores de classificação e regressão</div>
Árvores de classificação e regressão (CARTs - Classification and Regression Trees) são métodos para particionar o espaço de variáveis de entrada com base em um conjunto de regras em uma árvore de decisão, onde cada nó se divide de acordo com uma regra de decisão (como exemplificado na Figura abaixo). Desta forma, o espaço variável é particionado em subconjuntos e o modelo é ajustado a cada subconjunto.

</a><div width="2%" align="center">
<img src="https://github.com/Rhayron/YOLOv3_detection_segmentation/blob/main/assets/CART.png" width="60%"/>
</div></a>

### <div>Bagging</div>
Bagging (abreviação de agregação bootstrap) é um método de aprendizagem proposto por Breiman, 2001. No método de bagging, os diferentes conjuntos utilizados no treinamento dos modelos são produzidos por amostragens aleatórias com reposição. Com isso qualquer padrão tem a mesma probabilidade de aparecer novamente em um novo conjunto de treinamento. Uma amostragem é feita a partir dos dados originais para, em seguida, treinar o modelo (por exemplo, um CART) usando as amostras geradas. O procedimento de amostragem e treinamento são repetidos várias vezes. A previsão do método de Bagging é a média das previsões, o que permite reduzir a variância da função preditora.

### <div>Segmentação de imagens usando RF</div>
Florestas aleatórias nada mais são que a aplicação do método de Bagging em modelos CARTs com algum grau adicional de indeterminação. O Bagging de CARTs é necessário para aliviar a instabilidade do modelo (vide Ziegler et al., 2004). Além disso, a aleatoriedade é utilizada para reduzir a correlação entre as árvores e, consequentemente, reduzir a variância das previsões, ou seja, a média das árvores. O processo é realizado através da seleção aleatória das variáveis preditoras que serão candidatas para a divisão. Já a previsão na regressão é realizada pela média das previsões de cada árvore, enquanto na classificação é realizada pela obtenção da maioria dos votos da classe a partir dos votos individuais da classe da árvore (Hastie et al. 2015).

Conforme observado em Biau e Scornet (2016) os dois principais parâmetros dos algoritmos de RF são: o número de árvores treinadas e o número de variáveis preditoras selecionadas aleatoriamente. Outros parâmetros relevantes são o tamanho das amostras de dados usados em cada árvore e o número máximo de nós em cada folha, cujo valor é utilizado para impedir que a árvore se expanda indefinidamente.

A segmentação é o processo de agrupar uma imagem em várias sub-regiões coerentes de acordo com os recursos extraídos, por exemplo, atributos de cor ou textura, e classificar cada sub-região em uma das classes predeterminadas. Esses recursos descrevem cada pixel da imagem e suas regiões vizinhas com base em informações espaciais e relacionadas à escala em várias resoluções. A segmentação também pode ser vista como uma forma de compressão (reshape) de imagem que é um passo crucial na etapa de aprendizagem do modelo.

Em termos gerais, as técnicas de segmentação são divididas em duas categorias sendo elas, supervisionadas e não supervisionadas. O paradigma de segmentação supervisionada incorpora conhecimento prévio no processamento de imagem por meio de amostras de treinamento, assim como as redes neurais artificiais. Floresta aleatória (RF) está entre as técnicas de segmentação supervisionada. A Figura 22 mostra a estrutura de uma rede de segmentação baseada em RF. A rede é composta por quatro componentes:

</a><div width="2%" align="center">
<img src="https://github.com/Rhayron/YOLOv3_detection_segmentation/blob/main/assets/ImageSegmentation.png" width="60%"/>
</div></a>

Extração de recursos: a extração de recursos geralmente é realizada usando um banco com um conjunto de filtros pré-especificados. Tal banco de filtros pode gerar diversas representações a partir da imagem de entrada. Além disso, como os filtros não são aprendidos a partir dos dados, o banco de filtros precisa ser projetado com base na tarefa a ser realizada;

Seleção de recursos: Em contraste com o aprendizado profundo, onde os recursos são aprendidos e guiados por dados de treinamento (aprendizagem não supervisionadas) os recursos são bastante genéricos e, portanto, podem não ser boas representações para a tarefa de segmentação. Além disso, pode haver recursos redundantes que aumentam o overfitting do modelo. Os algoritmos de seleção de recursos são mecanismos para destilar bons recursos de recursos redundantes ou ruidosos. Os algoritmos de seleção de recursos podem ser supervisionados ou não;

Mapas de recursos aleatórios: Trata-se de uma função que captura a relação não linear entre as representações de dados de entrada e rótulos em algoritmos estatísticos de aprendizado de máquina. No caso de uma RF, o modelo cria uma floresta inteira de árvores de decisão aleatórias não correlacionadas para chegar à melhor resposta possível (Bootstrap) ou são usados para superar um problema de compensação de variância de viés. Em geral, o erro de aprendizagem pode ser explicado em termos de viés e variância. Por exemplo, se o viés for alto, os resultados do teste serão imprecisos; e se a variância for alta, o modelo é adequado apenas para determinado conjunto de dados (ou seja, overfitting ou instabilidade).

Após o treinamento, as previsões no conjunto de dados de teste, podem ser feitas de duas maneiras:
* Calculando a média das previsões de todas as árvores individuais;
* Obtendo a maioria dos votos para o caso de um problema de classificação.

O viés no erro de aprendizado é reduzido pela média dos resultados das respectivas árvores e, embora as previsões de uma única árvore sejam altamente sensíveis ao seu conjunto de treinamento, a média das árvores individuais não é sensível, desde que as árvores não sejam correlacionadas. Se as árvores são independentes umas das outras, então o teorema do limite central garantiria a redução da variância. A floresta aleatória usa um algoritmo que seleciona um subconjunto aleatório de recursos no processo de divisão de cada candidato para reduzir a correlação de árvores em uma amostra de ensacamento (HO, 2002).

Outra vantagem da RF é que é fácil de usar e requer ajuste de apenas três hiperparâmetros, ou seja, o número de árvores, o número de feições usadas em uma árvore e a taxa de amostragem para ensacamento. Além disso, os resultados de RF possuem alta precisão com estabilidade, porém, o processo interno do mesmo é uma espécie de caixa preta como em muitos modelos de deep learning.

## <div>Resultados e discursões</div>

Para realização do treinamento da rede neural responsável pela detecção das chaves seccionadoras foi utilizada uma base de dados contendo 2607 imagens ópticas. Os equipamentos usados durante o desenvolvimento incluem computadores para execução de softwares além da câmera térmica portátil para captura das fotos. A YOLOv3 foi treinado usando um computador portátil com acesso a uma máquina virtual da plataforma Google Colab, que disponibiliza em seu serviço em nuvem uma GPU.

O banco de dados utilizado contém imagens ópticas registradas em períodos diurnos, em dias diferentes e com variações no nível de iluminação, o que possibilita uma melhor capacidade de generalização do modelo de detecção durante a etapa de treinamento. O banco de dados contém imagens ópticas de chaves seccionadoras, sendo este dividido em três subconjuntos: conjuntos de treino, validação e teste. Os dados de treinamento são usados para ajustar os parâmetros (por exemplo, os pesos de conexão entre os neurônios) do modelo. Já os dados de validação são um conjunto de exemplos usados para ajustar os hiperparâmetros (ou seja, a arquitetura) da RNA. O desempenho da rede é então avaliado por meio da função de erro utilizando o conjunto de validação que é independente do conjunto de treino. Uma vez que este procedimento pode levar a algum sobreajuste no conjunto de validação, o desempenho da rede deve ser verificado medindo seu desempenho em um terceiro conjunto independente dos dados de validação e treino, denominado conjunto de teste.

Como é importante conhecer os dados com os quais se está trabalhando, foi realizado um levantamento da ocorrência de cada uma das classes, ou seja, os tipos de chaves seccionadoras ao longo do banco de dados. No Gráfico abaixo pode ser visto as distribuições das instâncias passadas para rede durante a etapa de treinamento. Como era de se esperar, observa-se que as chaves seccionadoras abertas ocorrem de maneira mais esparsa ao longo das imagens, enquanto as chaves fechadas são mais recorrentes no banco de dados, isso por que é mais difícil a ocorrência desse tipo de chave na subestação que foram capturadas as imagens para o presente projeto.

</a><div width="2%" align="center">
<img src="https://github.com/Rhayron/YOLOv3_detection_segmentation/blob/main/assets/Instancias.png" width="40%"/>
</div></a>

### Treinamento da YOLOv3

Optou-se pelo treinamento do modelo de detecção das chaves seccionadoras em duas versões da YOLO, a quinta e a terceira. As versões mais antigas da YOLO, como a YOLOv3, podem fornecer desempenho de detecção semelhante e localização mais precisa dos objetos. No entanto, a velocidade de treinamento da YOLOv5 é uma grande vantagem em comparação às outras versões.

Um notebook (como é chamado o algoritmo e todas suas anotações no Google Colab) foi implementado para os primeiros testes, com todos os passos necessários para treinar e validar o desempenho do modelo. O procedimento de treinamento consistiu em 300 épocas (onde, uma época consiste num ciclo de treinamento completo para determinada amostra), que levaram em torno de 24 horas para o conjunto de dados. Das 2607 imagens, 2086 foram utilizadas para treinamento e 521 no conjunto de teste. Os conjuntos de dados são separados de maneira aleatória, isso para garantir que o modelo não fique viciado e tendencioso. Para o segundo experimento, foi utilizado a terceira versão da YOLO que em seu treinamento levou cerca de 48 horas para conclusão com as mesmas 300 épocas.

### Desempenho para detecção de chaves seccionadoras

O Gráfico abaixo mostra os resultados da mAP obtidos a partir do treinamento dos modelos da YOLOv5 e YOLOv3. Com base nos resultados é possível observar que as duas versões tiveram desempenho bem parecidos, a única diferença fica por conta da terceira versão que conseguiu resultados melhores com menos épocas.

</a><div width="2%" align="center">
<img src="https://github.com/Rhayron/YOLOv3_detection_segmentation/blob/main/assets/map.png" width="80%"/>
</div></a>

Em termos de precisão do modelo, as versões também apresentam resultados bastante similares, ficando ambas acima dos 70% de precisão em alguns momentos, como revela o Gráfico abaixo. Com base no gráfico de precisão, o treinamento poderia ter sido interrompido antes das 150 interações, obtendo a mesma performance além de poupar recursos computacionais como o tempo de uso de GPU no Google Colab.

</a><div width="2%" align="center">
<img src="https://github.com/Rhayron/YOLOv3_detection_segmentation/blob/main/assets/precision.png" width="80%"/>
</div></a>

A Figura abaixo, mostra a matrize de confusão, onde a diagonal principal apresentou o melhor resultado que os demais pontos, o que mostra que a rede funciona de maneira adequada para as classes estipuladas. O background, que é a classe que designa o fundo das imagens, foi o que teve maior problema, pois em  67% na YOLOv3 foi considerada como Chave Seccionadora Lâmina (Fechada).

</a><div width="2%" align="center">
<img src="https://github.com/Rhayron/YOLOv3_detection_segmentation/blob/main/assets/ConfusionMatrix.png" width="60%"/>
</div></a>

A YOLO mostra um funcionamento satisfatório, apresentando detecções com mais de 80% de precisão. No entanto, em alguns casos o modelo treinado confunde outros equipamentos na subestação como chaves seccionadoras, além de apresentar certa dificuldade em detectar as chaves quando há interferências provocadas pelos raios solares.

Problemas como a forte presença do sol interferindo nas detecções podem ser resolvidos corrigindo manualmente as caixas delimitadoras erradas e inserindo-as em um novo treinamento do modelo (realimentação positiva). Além disso fornecer para a rede mais exemplos de imagens onde há raios solares ajudaria a mitigar o problema.

Nos testes realizados, o parâmetro de IoU foi reduzido para 0,25. Isso significa que uma caixa de detecção é considerada válida para IoU ≥ 25%. Ao considerar um limiar de IoU menor, é possível visualizar um número mais significativo de detecções inválidas, ou seja, aparecem mais exemplos de falsos positivos na análise de cada imagem.

### Segmentação da imagem

Para realizar a segmentação das imagens térmicas, primeiro é preciso processar as imagens óticas e térmicas obtidas pela câmera, uma vez que as lentes ótica e térmica estão em perpectivas diferentes nos dois tipos de imagens. Dessa forma uma caixa delimitadora detectada pela YOLOv3 na imagem ótica apresentaria coordenadas diferentes na imagem térmica.

Utilizando as técnicas de homografia é possível transformar a imagem térmica, de forma a permanecerem ambas imagens com mesma perspectiva. Assim, é possível identificar as coordenadas homográficas entre as duas imagens e obter ambas imagens no plano de perspectiva.Com as mesmas coordenadas homográficas obtidas do processo de calibração, é possível alinhas os termogramas das chaves seccionadoras com as imagens ópticas.Após colocar o termograma no mesmo plano de perspectiva é possível detectar objetos na imagem IR a partir das bounding boxes resultantes do processo de detecção por meio da YOLOv3 na imagem óptica.

Posteriormente, é utilizado o Random Forest para a segmentação dos pontos quentes dentro das bounding boxes que foram identificadas anteriormente com os equipamentos presentes na subestação. A partir das coordenadas das caixas delimitadoras na imagem IR, é feito um recorte na imagem das chaves seccionadoras detectadas e, posteriormente, aplicado o algoritmo de Random Forest, para a segmentação da região de aquecimento no recorte da imagem IR. 

O algoritmo de RF não consegue boa precisão de segmentação para os casos onde há pouca variação relativa de temperatura na imagem. No geral, a rede YOLO apresentou bons resultados de detecção das chaves seccionadoras, assim como bons resultados de segmentação dos termogramas. Com esses resultados, é possível automatizar o processo de inspeção desses equipamentos de subestações e tornar, assim, um processo mais confiável, seguro e robusto como um todo.

## <div>Conclusão</div>

Neste trabalho, foram apresentados alguns conceitos envolvendo a aplicação de conceitos de visão computacional em inspeções de rotina visando detectar falhas em equipamentos elétricos. Tais ferramentas foram unidas para formar um procedimento genérico e inteligente de segmentação de pontos quentes em chaves seccionadoras presentes numa SE, através da utilização de uma RNA para de detecção automáticas das chaves nas imagens óticas, aliada a segmentação de regiões de sobreaquecimentos nas imagens térmicas.

Deve-se ressaltar que os algoritmos envolvidos em tal procedimento, especialmente para segmentação de imagens e identificação de alvos, não se dedicam à análise e diagnóstico dos equipamentos em si, mas apenas um apontamento nas imagens infravermelhas das regiões de aquecimento que podem ser ou não provenientes de uma falha.

Após um exaustivo trabalho de anotação das imagens para treinamento da RNA, foi possível alcançar bons resultados com o treinamento da rede neural YOLOv3 para a identificação dos quatro tipos de chaves seccionadoras. Através da matriz de confusão, pode-se perceber que o modelo alcançou índices maiores que 85% para todos os tipos de chaves seccionadoras usadas no treinamento. Já no processo de segmentação de imagem, foram obtidos resultados preliminares satisfatórios com a utilização do algoritmo de floresta aleatória, comprovando sua eficácia e capacidade de utilização para imagens térmicas.

Para trabalhos futuros, há a possibilidades de aumentar o número de imagens do banco de dados, o que pode ajudar a melhorar os resultados obtidos pela YOLO, principalmente para os casos onde há forte interferência de raios solares. Além disso, pode-se desenvolver um sistema automático de captação de imagens utilizando câmeras móveis dentro da subestação elétrica, fornecendo um banco contínuo e atualizado de imagens da situação da SE. Dessa forma, um operador do sistema poderia analisar em tempo real o comportamento dos equipamentos, visto que a YOLO consegue atingir altas taxas de FPS nas detecções.

Do ponto de vista da inspeção elétrica, as medições termográficas têm utilidades indiscutíveis. Em vez de esperar as falhas do equipamento, deve-se optar por uma manutenção preditiva. A implementação do presente trabalho em um ambiente real, pode auxiliar técnicos menos experientes, visto que a rede neural faz a parte da detecção dos objetos e o ensemble Random Forest segmenta dos pontos quentes das imagens IR, restando ao técnico a avaliação e diagnóstico da temperatura resultante da segmentação de cada equipamento.

<details open>
<summary>Tutorials</summary>

* [Train Test](https://github.com/Rhayron/YOLOv3_detection_segmentation/blob/main/TrainTest.ipynb)&nbsp; 🚀 RECOMMENDED
* [Train Custom RF model]()&nbsp; 🌟 NEW

</details>

## <div>Environments</div>

<div align="center">
    <a href="https://colab.research.google.com/drive/1LEuVVoTscsaqlbqM21sDZWgRpuCajSkb?authuser=2">
        <img src="https://github.com/Rhayron/YOLOv3_detection_segmentation/blob/main/assets/logo-colab-small.png" width="15%"/>
    </a>
</div>

