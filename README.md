# PetFinder Adoption Prediction 

Este repositorio contiene el desarrollo y la implementación de múltiples modelos de Machine Learning para la competencia [PetFinder.my Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction) de Kaggle. 

El proyecto fue desarrollado como trabajo práctico para la asignatura **Laboratorio de Implementación II** de la Maestría en Ciencia de Datos de la Universidad Austral.

---

## El Problema

Cada día, millones de animales sufren en las calles o son sacrificados en refugios en todo el mundo. Encontrarles un hogar no solo salva vidas, sino que crea familias más felices. 

**PetFinder.my**, la principal plataforma de bienestar animal de Malasia, cuenta con una base de datos de más de 150.000 animales. Se ha observado que las tasas de adopción están fuertemente correlacionadas con los metadatos asociados a sus perfiles online (texto descriptivo, características tabulares y fotografías). 

El objetivo de este proyecto es desarrollar diferentes modelos capaces de predecir la "adoptabilidad" de las mascotas, específicamente, **qué tan rápido serán adoptadas**. Estas herramientas buscan guiar a refugios y rescatistas para mejorar el atractivo de los perfiles de los animales y reducir el sufrimiento animal.

---

## Variable Objetivo: `AdoptionSpeed`

El target de este proyecto es un problema de clasificación ordinal. Se busca predecir la categoría de velocidad de adopción, donde un valor más bajo indica una adopción más rápida. 

Las categorías se definen de la siguiente manera:
*   **0:** La mascota fue adoptada el mismo día en que fue listada.
*   **1:** La mascota fue adoptada entre 1 y 7 días (1ra semana) después de ser listada.
*   **2:** La mascota fue adoptada entre 8 y 30 días (1er mes) después de ser listada.
*   **3:** La mascota fue adoptada entre 31 y 90 días (2do y 3er mes) después de ser listada.
*   **4:** No hubo adopción después de 100 días de estar listada.

> **Nota sobre los datos:** Para este desarrollo no se utilizó el dataset de `test` original de la competencia. Se trabajó exclusivamente con el dataset de `train`, el cual fue dividido internamente (Split 80/20 estratificado) para las fases de entrenamiento y validación.

### Métrica de Evaluación
Los modelos se evalúan utilizando **Quadratic Weighted Kappa (QWK)**, una métrica que mide el nivel de acuerdo entre dos calificaciones (las reales vs. las predichas). Esta métrica es ideal para problemas ordinales, ya que penaliza los errores de forma proporcional al cuadrado de la distancia entre las clases. Varía típicamente de 0 (acuerdo aleatorio) a 1 (acuerdo perfecto).

---

## Modelos Desarrollados

Para abordar la naturaleza multimodal del dataset, se construyeron tres flujos de trabajo (*pipelines*) independientes y un ensamble final:

### 1. Modelo Tabular (LightGBM) 
*   **Enfoque:** Algoritmo de *Gradient Boosting Trees* para el procesamiento de la información estructurada.
*   **Feature Engineering:** Se decodificaron variables categóricas, se crearon nuevas características agrupadas (ej. métricas compuestas de salud, *bins* de edad y tarifas) y se analizaron las frecuencias de razas y colores.
*   **Evaluación:** Validación mediante *Stratified K-Fold* de 5 particiones.

### 2. Modelo de Imágenes (ResNet50) 
*   **Arquitectura:** *Transfer Learning* partiendo de **ResNet50** pre-entrenado en ImageNet, adaptando la capa de salida para 5 clases y aplicando *fine-tuning* completo. Se utilizó únicamente la foto principal (perfil) de cada mascota.
*   **Procesamiento:** Se implementaron técnicas de *Data Augmentation* avanzadas (*AutoAugment - ImageNetPolicy* y *Cutout*). La función de pérdida (*CrossEntropyLoss*) fue ponderada para manejar el desbalance de las clases.
*   **Optimización:** Búsqueda bayesiana de hiperparámetros utilizando **OPTUNA** (algoritmo TPE con poda *MedianPruner*).

### 3. Modelo de Texto (DistilBERT) 
*   **Arquitectura:** Modelo de lenguaje NLP pre-entrenado (*Hugging Face*) para clasificar las descripciones de las mascotas.
*   **Procesamiento:** Tokenización mediante `DistilBertTokenizerFast` con *padding* dinámico y truncamiento a 512 tokens. Las mascotas sin descripción fueron excluidas de este pipeline específico.

### 4. Ensamble Ponderado 
*   **Metodología:** Integración de los tres modelos anteriores. Se aplicó una normalización Min-Max a las predicciones crudas (*logits* / *scores*) de cada pipeline para llevarlas a la misma escala [0, 1].
*   **Optimización:** Búsqueda en grilla 2D (Grid Search) para encontrar los pesos óptimos de los modelos y maximizar la métrica QWK combinada. Las mascotas sin datos visuales o de texto fueron cubiertas por el modelo tabular.

---

##  Estructura del Repositorio y Reproducibilidad

En este repositorio encontrarás el código fuente de los modelos y el ensamble final. Debido al volumen de los datos y pesos de los modelos, los datasets crudos y los archivos muy pesados se gestionan externamente.

*   `imagenes_g8.ipynb`: Pipeline completo de procesamiento y entrenamiento del modelo visual (ResNet50). Preparado para correr con Google Colab (ver carpeta).
*   `texto_g8.ipynb`: Pipeline completo del procesamiento de lenguaje natural (DistilBERT). Preparado para correr con Google Colab (ver carpeta).
*   `ensemble_g8.ipynb`: Script final que consolida las predicciones (archivos `.joblib`) de los tres modelos y calcula el QWK maximizado.
*   `EDA&Tabulares_g8.zip`: Carpeta comprimida que contiene el Análisis Exploratorio de Datos (EDA) y el entrenamiento del modelo LightGBM para correr de forma local..

### Cómo correr el proyecto

Para ejecutar los notebooks de Deep Learning (`imagenes_g8.ipynb` y `texto_g8.ipynb`) que requieren aceleración por GPU, se preparó un entorno en la nube. 

Toda la estructura de carpetas, datasets originales, y archivos generados (`.joblib`, `.pth`) se encuentran alojados en Google Drive. Podés crear acceso directo a tu propio Drive y correr los notebooks directamente en **Google Colab**:

**[Acceder al Google Drive del Proyecto (Datos y Modelos)](https://drive.google.com/drive/u/0/folders/1zbFLDWMyFpbnAMQDKJBdY2IbGoBCiMHQ)**

---

## Autores - Grupo 8
*   Analía Ale
*   Diego Farfán
*   Jaime López Garrido
*   Gabriel Martina
*   Lucía Pereyra Huertas
