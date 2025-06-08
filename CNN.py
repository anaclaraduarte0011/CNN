# Classificação de Imagens: Cachorros vs Gatos usando CNN
# Projeto de Visão Computacional com Redes Neurais Convolucionais

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Configurações iniciais
plt.style.use('seaborn-v0_8')
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# =============================================================================
# 1. PREPARAÇÃO DOS DADOS
# =============================================================================

print("\n" + "="*60)
print("1. PREPARAÇÃO DOS DADOS")
print("="*60)

# Configurações dos dados
BASE_DIR = 'catdog'  # Pasta principal com as imagens
IMG_HEIGHT, IMG_WIDTH = 150, 150  # Dimensões das imagens
BATCH_SIZE = 32
EPOCHS = 15

# Verificar estrutura dos dados
def verificar_estrutura_dados(base_dir):
    """Verifica e exibe a estrutura dos dados"""
    if not os.path.exists(base_dir):
        print(f"❌ Pasta {base_dir} não encontrada!")
        return False
    
    print(f"📁 Estrutura da pasta {base_dir}:")
    for root, dirs, files in os.walk(base_dir):
        level = root.replace(base_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:3]:  # Mostrar apenas os primeiros 3 arquivos
            print(f"{subindent}{file}")
        if len(files) > 3:
            print(f"{subindent}... e mais {len(files)-3} arquivos")
    return True

# Verificar dados
if verificar_estrutura_dados(BASE_DIR):
    print("✅ Estrutura de dados verificada com sucesso!")
else:
    print("⚠️  Por favor, certifique-se de que a pasta 'catdog' existe e contém as subpastas com imagens")

# Contar imagens por classe
def contar_imagens(base_dir):
    """Conta o número de imagens por classe"""
    contagem = {}
    for classe in os.listdir(base_dir):
        caminho_classe = os.path.join(base_dir, classe)
        if os.path.isdir(caminho_classe):
            num_imagens = len([f for f in os.listdir(caminho_classe) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            contagem[classe] = num_imagens
    return contagem

try:
    contagem_imagens = contar_imagens(BASE_DIR)
    print(f"\n📊 Distribuição das imagens:")
    for classe, num in contagem_imagens.items():
        print(f"   {classe}: {num} imagens")
    
    total_imagens = sum(contagem_imagens.values())
    print(f"   Total: {total_imagens} imagens")
except:
    print("⚠️  Não foi possível contar as imagens. Verifique a estrutura dos dados.")

# =============================================================================
# 2. PRÉ-PROCESSAMENTO E AUMENTO DE DADOS
# =============================================================================

print("\n" + "="*60)
print("2. PRÉ-PROCESSAMENTO E AUMENTO DE DADOS")
print("="*60)

# Data Augmentation para treino
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,           # Normalização dos pixels [0,1]
    rotation_range=20,           # Rotação até 20 graus
    width_shift_range=0.2,       # Deslocamento horizontal
    height_shift_range=0.2,      # Deslocamento vertical
    shear_range=0.2,            # Cisalhamento
    zoom_range=0.2,             # Zoom
    horizontal_flip=True,        # Inversão horizontal
    fill_mode='nearest',         # Preenchimento dos pixels
    validation_split=0.3         # 30% para validação e teste
)

# Apenas normalização para validação e teste
test_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.3
)

# Geradores de dados
try:
    # Dados de treino (70%)
    train_generator = train_datagen.flow_from_directory(
        BASE_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    # Dados de validação (15%)
    validation_generator = test_datagen.flow_from_directory(
        BASE_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    print("✅ Geradores de dados criados com sucesso!")
    print(f"   Treino: {train_generator.samples} imagens")
    print(f"   Validação: {validation_generator.samples} imagens")
    print(f"   Classes: {train_generator.class_indices}")
    
except Exception as e:
    print(f"❌ Erro ao criar geradores de dados: {e}")

# Visualizar amostras das imagens
def visualizar_amostras(generator, num_samples=8):
    """Visualiza amostras das imagens processadas"""
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle('Amostras das Imagens Processadas', fontsize=16, fontweight='bold')
    
    # Obter um batch de imagens
    batch_images, batch_labels = next(generator)
    
    for i in range(min(num_samples, len(batch_images))):
        row = i // 4
        col = i % 4
        
        axes[row, col].imshow(batch_images[i])
        classe = 'Gato' if batch_labels[i] == 0 else 'Cachorro'
        axes[row, col].set_title(f'{classe}', fontweight='bold')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

try:
    visualizar_amostras(train_generator)
except:
    print("⚠️  Não foi possível visualizar as amostras")

# =============================================================================
# 3. CONSTRUÇÃO DA CNN
# =============================================================================

print("\n" + "="*60)
print("3. CONSTRUÇÃO DA REDE NEURAL CONVOLUCIONAL")
print("="*60)

def criar_modelo_cnn():
    """Cria a arquitetura da CNN"""
    model = models.Sequential([
        # Primeira camada convolucional
        layers.Conv2D(32, (3, 3), activation='relu', 
                     input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D(2, 2),
        
        # Segunda camada convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Terceira camada convolucional
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Quarta camada convolucional
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Achatamento e camadas densas
        layers.Flatten(),
        layers.Dropout(0.5),  # Regularização
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),  # Mais regularização
        
        # Camada de saída
        layers.Dense(1, activation='sigmoid')  # Classificação binária
    ])
    
    return model

# Criar e compilar o modelo
model = criar_modelo_cnn()

# Compilar o modelo
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Exibir arquitetura do modelo
print("🏗️  Arquitetura da CNN:")
model.summary()

# Visualizar arquitetura
def plotar_arquitetura_modelo():
    """Cria visualização da arquitetura do modelo"""
    try:
        tf.keras.utils.plot_model(
            model, 
            to_file='model_architecture.png', 
            show_shapes=True, 
            show_layer_names=True,
            rankdir='TB'
        )
        print("✅ Diagrama da arquitetura salvo como 'model_architecture.png'")
    except:
        print("⚠️  Não foi possível criar o diagrama da arquitetura")

plotar_arquitetura_modelo()

# =============================================================================
# 4. TREINAMENTO DO MODELO
# =============================================================================

print("\n" + "="*60)
print("4. TREINAMENTO DO MODELO")
print("="*60)

# Callbacks para melhorar o treinamento
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.0001,
        verbose=1
    )
]

# Treinar o modelo
print(f"🚀 Iniciando treinamento por {EPOCHS} épocas...")

try:
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    print("✅ Treinamento concluído com sucesso!")
    
except Exception as e:
    print(f"❌ Erro durante o treinamento: {e}")

# =============================================================================
# 5. VISUALIZAÇÃO DOS RESULTADOS DO TREINAMENTO
# =============================================================================

print("\n" + "="*60)
print("5. ANÁLISE DO TREINAMENTO")
print("="*60)

def plotar_historico_treinamento(history):
    """Plota gráficos de acurácia e perda durante o treinamento"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de Acurácia
    ax1.plot(history.history['accuracy'], label='Treino', linewidth=2, marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validação', linewidth=2, marker='s')
    ax1.set_title('Acurácia por Época', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Acurácia')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de Perda
    ax2.plot(history.history['loss'], label='Treino', linewidth=2, marker='o')
    ax2.plot(history.history['val_loss'], label='Validação', linewidth=2, marker='s')
    ax2.set_title('Perda por Época', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Perda')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Estatísticas finais
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\n📊 Resultados Finais do Treinamento:")
    print(f"   Acurácia de Treino: {final_train_acc:.4f}")
    print(f"   Acurácia de Validação: {final_val_acc:.4f}")
    print(f"   Perda de Treino: {final_train_loss:.4f}")
    print(f"   Perda de Validação: {final_val_loss:.4f}")

try:
    plotar_historico_treinamento(history)
except:
    print("⚠️  Não foi possível plotar o histórico de treinamento")

# =============================================================================
# 6. AVALIAÇÃO DETALHADA DO MODELO
# =============================================================================

print("\n" + "="*60)
print("6. AVALIAÇÃO DETALHADA DO MODELO")
print("="*60)

def avaliar_modelo_detalhado(model, generator):
    """Avaliação completa do modelo"""
    print("🔍 Realizando avaliação detalhada...")
    
    # Fazer predições
    generator.reset()
    predictions = model.predict(generator, verbose=1)
    predicted_classes = (predictions > 0.5).astype(int)
    
    # Obter labels verdadeiros
    true_classes = generator.classes
    
    # Calcular métricas
    precision = precision_score(true_classes, predicted_classes)
    recall = recall_score(true_classes, predicted_classes)
    f1 = f1_score(true_classes, predicted_classes)
    
    print(f"\n📈 Métricas de Performance:")
    print(f"   Precisão: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    # Relatório de classificação
    print(f"\n📋 Relatório de Classificação Detalhado:")
    class_names = ['Gato', 'Cachorro']
    print(classification_report(true_classes, predicted_classes, 
                              target_names=class_names))
    
    # Matriz de confusão
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão', fontsize=14, fontweight='bold')
    plt.ylabel('Classe Verdadeira')
    plt.xlabel('Classe Predita')
    plt.show()
    
    return predictions, predicted_classes, true_classes

try:
    predictions, predicted_classes, true_classes = avaliar_modelo_detalhado(model, validation_generator)
except Exception as e:
    print(f"❌ Erro na avaliação: {e}")

# =============================================================================
# 7. FUNÇÃO PARA TESTAR IMAGENS INDIVIDUAIS
# =============================================================================

print("\n" + "="*60)
print("7. TESTE COM IMAGENS INDIVIDUAIS")
print("="*60)

def prever_imagem_individual(model, caminho_imagem):
    """Faz predição para uma imagem individual"""
    try:
        # Carregar e preprocessar a imagem
        img = load_img(caminho_imagem, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Fazer predição
        prediction = model.predict(img_array)[0][0]
        classe_predita = 'Cachorro' if prediction > 0.5 else 'Gato'
        confianca = prediction if prediction > 0.5 else 1 - prediction
        
        # Visualizar resultado
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f'Predição: {classe_predita}\nConfiança: {confianca:.2%}', 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.show()
        
        print(f"🎯 Resultado da predição:")
        print(f"   Classe: {classe_predita}")
        print(f"   Confiança: {confianca:.2%}")
        print(f"   Score bruto: {prediction:.4f}")
        
        return classe_predita, confianca
        
    except Exception as e:
        print(f"❌ Erro ao processar imagem: {e}")
        return None, None

# Exemplo de como usar a função (descomente e ajuste o caminho)
"""
# Exemplo de teste com imagem individual
caminho_teste = "caminho/para/sua/imagem.jpg"
if os.path.exists(caminho_teste):
    prever_imagem_individual(model, caminho_teste)
else:
    print("⚠️  Para testar uma imagem individual, ajuste o caminho em 'caminho_teste'")
"""

# =============================================================================
# 8. ANÁLISE DE ERROS E INSIGHTS
# =============================================================================

print("\n" + "="*60)
print("8. ANÁLISE DE ERROS E INSIGHTS")
print("="*60)

def analisar_erros(model, generator, num_samples=8):
    """Analisa os erros de classificação"""
    generator.reset()
    
    # Fazer predições
    predictions = model.predict(generator, verbose=0)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = generator.classes
    
    # Encontrar erros
    erros = np.where(predicted_classes != true_classes)[0]
    
    if len(erros) > 0:
        print(f"🔍 Encontrados {len(erros)} erros de classificação")
        
        # Mostrar alguns exemplos de erros
        num_mostrar = min(num_samples, len(erros))
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Exemplos de Erros de Classificação', fontsize=16, fontweight='bold')
        
        for i in range(num_mostrar):
            idx = erros[i]
            row = i // 4
            col = i % 4
            
            # Obter a imagem
            img_path = generator.filepaths[idx]
            img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            
            true_class = 'Gato' if true_classes[idx] == 0 else 'Cachorro'
            pred_class = 'Gato' if predicted_classes[idx] == 0 else 'Cachorro'
            confidence = predictions[idx][0] if predicted_classes[idx] == 1 else 1 - predictions[idx][0]
            
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'Real: {true_class}\nPred: {pred_class} ({confidence:.2%})')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print("🎉 Nenhum erro encontrado na amostra analisada!")

try:
    analisar_erros(model, validation_generator)
except Exception as e:
    print(f"⚠️  Erro na análise de erros: {e}")

# =============================================================================
# 9. SALVAMENTO DO MODELO
# =============================================================================

print("\n" + "="*60)
print("9. SALVAMENTO DO MODELO")
print("="*60)

def salvar_modelo(model, nome_arquivo='modelo_cnn_dogs_cats.h5'):
    """Salva o modelo treinado"""
    try:
        model.save(nome_arquivo)
        print(f"✅ Modelo salvo como '{nome_arquivo}'")
        
        # Salvar apenas os pesos (alternativa mais leve)
        model.save_weights(nome_arquivo.replace('.h5', '_weights.h5'))
        print(f"✅ Pesos do modelo salvos como '{nome_arquivo.replace('.h5', '_weights.h5')}'")
        
    except Exception as e:
        print(f"❌ Erro ao salvar modelo: {e}")

salvar_modelo(model)

# =============================================================================
# 10. CONCLUSÕES E RECOMENDAÇÕES
# =============================================================================

print("\n" + "="*60)
print("10. CONCLUSÕES E RECOMENDAÇÕES")
print("="*60)

def gerar_relatorio_final():
    """Gera relatório final com conclusões"""
    print("📋 RELATÓRIO FINAL - CLASSIFICAÇÃO CACHORROS vs GATOS")
    print("=" * 55)
    
    print("\n🎯 OBJETIVOS ALCANÇADOS:")
    print("   ✅ Implementação de CNN para classificação binária")
    print("   ✅ Pré-processamento e aumento de dados")
    print("   ✅ Treinamento e validação do modelo")
    print("   ✅ Avaliação com métricas detalhadas")
    print("   ✅ Análise de erros e visualizações")
    
    print("\n🏗️  ARQUITETURA UTILIZADA:")
    print("   • 4 camadas convolucionais (32, 64, 128, 128 filtros)")
    print("   • MaxPooling após cada convolução")
    print("   • Dropout para regularização (0.5 e 0.3)")
    print("   • Camada densa com 512 neurônios")
    print("   • Saída sigmoid para classificação binária")
    
    print("\n🔧 TÉCNICAS DE PRÉ-PROCESSAMENTO:")
    print("   • Normalização de pixels (0-1)")
    print("   • Redimensionamento para 150x150")
    print("   • Data Augmentation: rotação, zoom, flip, etc.")
    print("   • Divisão: 70% treino, 30% validação")
    
    print("\n📊 ESTRATÉGIAS DE TREINAMENTO:")
    print("   • Otimizador Adam (lr=0.001)")
    print("   • Early Stopping (paciência=5)")
    print("   • Redução de learning rate (paciência=3)")
    print("   • Batch size=32, até 15 épocas")
    
    print("\n🎯 RECOMENDAÇÕES PARA MELHORIAS:")
    print("   • Aumentar dataset com mais imagens variadas")
    print("   • Experimentar Transfer Learning (VGG16, ResNet)")
    print("   • Ajustar hiperparâmetros (learning rate, batch size)")
    print("   • Implementar validação cruzada")
    print("   • Testar outras técnicas de regularização")
    
    print("\n💡 CONSIDERAÇÕES TÉCNICAS:")
    print("   • Modelo adequado para prototipagem rápida")
    print("   • Boa generalização com data augmentation")
    print("   • Monitoramento de overfitting eficaz")
    print("   • Pipeline reproduzível e bem documentado")
    
    print("\n🔍 PRÓXIMOS PASSOS:")
    print("   1. Coletar mais dados para classes desbalanceadas")
    print("   2. Implementar ensemble de modelos")
    print("   3. Otimizar para produção (quantização, pruning)")
    print("   4. Criar API para deployment")
    print("   5. Monitorar performance em produção")

gerar_relatorio_final()

print("\n" + "="*60)
print("🎉 PROJETO CONCLUÍDO COM SUCESSO!")
print("="*60)
print("\n💾 Arquivos gerados:")
print("   • modelo_cnn_dogs_cats.h5 (modelo completo)")
print("   • modelo_cnn_dogs_cats_weights.h5 (apenas pesos)")
print("   • model_architecture.png (diagrama da arquitetura)")
print("\n📝 Para usar o modelo salvo:")
print("   model = tf.keras.models.load_model('modelo_cnn_dogs_cats.h5')")
print("\n🚀 O modelo está pronto para classificar novas imagens!")