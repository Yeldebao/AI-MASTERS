# Notas del orador — Kvasir v2: CNNs vs. Transformers

## 1. CNNs vs. Transformers en endoscopía gastrointestinal

(15 s) Comparo cuatro arquitecturas y un ensamble para clasificar endoscopías de Kvasir v2. Todas con el mismo protocolo, así que las diferencias vienen de la arquitectura, no del entrenamiento.

## 2. ¿Qué arquitectura clasifica mejor la endoscopía con pocos datos?

(25 s) Kvasir v2 tiene 8,000 imágenes en 8 clases balanceadas: puntos anatómicos como píloro, ciego y línea Z; hallazgos como esofagitis, pólipos y colitis; y procedimientos con tinción azul. La pregunta: con solo 5,600 imágenes de entrenamiento, ¿qué familia funciona mejor y, sobre todo, por qué?

## 3. Cuatro arquitecturas, elegidas por una razón

(40 s) ResNet-50 porque es el codo de la curva: la 101 y la 152 cuestan el doble o triple por uno o dos puntos, y con pocos datos solo sobreajustan más. EfficientNet-B0 porque es la única diseñada para 224 píxeles; las otras escalan la resolución y romperían el protocolo. ViT-S porque tiene el mismo cómputo que ResNet-50: comparo convolución contra atención con igual presupuesto. Y el híbrido lo construí yo: ResNet-50 hasta el stage 3, 196 tokens y cuatro capas Transformer, como el ViT híbrido o TransUNet.

## 4. Un protocolo común para los cuatro modelos

(30 s) Todos comparten el protocolo. Aumento datos con flips y rotaciones, pero no toco el tono: el color es diagnóstico. 224 píxeles porque es la resolución nativa de los pesos; batch 32 para tener 175 actualizaciones por época; AdamW con 1e-4 para ajustar sin borrar lo aprendido; y warm-up más coseno, porque al inicio la cabeza aleatoria genera gradientes que dañarían los pesos preentrenados.

## 5. ViT-S/16 es el mejor modelo individual

(30 s) Los cuatro superan 0.90 de macro-F1, el criterio que fijé antes de entrenar. ViT-S es el mejor, con 0.936, y McNemar confirma que supera significativamente a los otros tres. ResNet, EfficientNet y el híbrido no se distinguen entre sí. El ensamble tiene el número más alto, pero frente a ViT-S la diferencia no es significativa y cuesta el doble.

## 6. Menos cómputo no significa menos latencia

(25 s) A la izquierda, desempeño contra cómputo: EfficientNet-B0 logra el 97 por ciento del mejor F1 con una doceava parte de las operaciones. Pero a la derecha, con una sola imagen, es el más lento: sus capas depthwise están limitadas por memoria. La eficiencia depende de dónde se despliegue el modelo.

## 7. Convergen en 2–3 épocas; el híbrido tiene sobreajuste

(20 s) Por el preentrenamiento, todos llegan al 95 por ciento de su mejor F1 en dos o tres épocas. ViT-S siguió mejorando hasta la 19. En el híbrido la pérdida de validación sube después de la época 7: sus capas Transformer nuevas no se estabilizan con tan pocos datos.

## 8. El preentrenamiento pesa más que la arquitectura

(30 s) Cambié un hiperparámetro a la vez sobre EfficientNet-B0. Lo que más pesa, por mucho, es el preentrenamiento: sin él se pierden entre 0.19 y 0.26, de seis a nueve veces la diferencia entre arquitecturas. Un learning rate de 1e-5 subajusta y 64 píxeles pierde detalle. La sorpresa: 1e-3 mejoró a B0, así que el learning rate común era conservador para él.

## 9. Falta de atención

(30 s) Grad-CAM en las CNNs y attention rollout en los Transformers. ResNet-50 da los mapas más nítidos, sobre la lesión; ViT-S es difuso y a veces mira el recuadro verde; el híbrido es el más focalizado. El error más común es esofagitis contra línea Z, y los mapas sí miran la unión esófago-gástrica: no se equivocan de lugar, la diferencia es sutil.

## 10. Por qué ganó o perdió cada modelo

(35 s) ViT-S ganó, pero fue preentrenado con diez veces más imágenes: lo honesto es decir que gana un Transformer bien preentrenado, no que la atención le gane a la convolución. ResNet-50 converge rápido y explica mejor, ideal para clínicos. EfficientNet casi iguala con una doceava parte del cómputo. Y el híbrido perdió porque cambié el stage 4 preentrenado por capas aleatorias; aun así, sus errores son los más distintos a los de ViT, por eso entra al ensamble.

## 11. Lo que importa no es solo la arquitectura

(25 s) En resumen: ViT-S es el mejor, con la advertencia del preentrenamiento; el preentrenamiento pesa más que la arquitectura; la eficiencia depende del despliegue; y todos aprendieron un atajo que hay que quitar. Como trabajo futuro, separar arquitectura de preentrenamiento y validar en otro hospital. Todo está en GitHub. Gracias.
