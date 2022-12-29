from detect_utils import detect_boxes

#classes: ['Chave Seccionadora Lamina (Aberta)',
#          'Chave Seccionadora Lamina (Fechada)',
#          'Chave Seccionadora Tandem (Aberta)',
#          'Chave Seccionadora Tandem (Fechada)',
#          'Disjuntor',
#          'Fusivel',
#          'Isolador disco de vidro',
#          'Isolador pino de porcelana conjunto',
#          'Isolador pino de porcelana individual',
#          'Mufla',
#          'Para-raio',
#          'Religador',
#          'Transformador',
#          'Transformador de Corrente (TC)',
#          'Transformador de Potencial (TP)']

# lamina fechada 1, tandem fechada 3, para-raio 10
boxes = detect_boxes(source = "./sample_images/", classes = [1, 3, 10])

for det in boxes:
    print(f'Image: {det["img_name"]}')
    print("Boxes [x, y, w, h, confidence, class]: ")
    for box in det["boxes"]:
        print(box)