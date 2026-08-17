"""
PhytoGATE: Phytosanitary Gated Attention & Texture Ensemble Architecture
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers

def build_data_augmentation_layer():
    """ Spatial and color data augmentation layer. """
    return models.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.20),
        layers.RandomContrast(0.20),
        layers.RandomTranslation(0.12, 0.12)
    ], name="data_augmentation")

def build_phytogate_model(pca_dim, num_classes, img_size=(224, 224, 3)):
    """
    Constructs the complete PhytoGATE Architecture:
      - Stream A: PCA-compressed handcrafted features (128-dim projection)
      - Stream B: Dual Spatial Deep Ensemble (EfficientNetB0 + DenseNet121 GAP features, 512-dim)
      - Gating: Independent Sigmoid Cross-Attention Gates (g_A, g_B)
      - Classifier: Dense(384) -> Dense(192) -> Softmax
    """
    img_input = layers.Input(shape=img_size, dtype='uint8', name="img_input")
    x_float = layers.Lambda(lambda x: tf.cast(x, tf.float32), name="cast_float32")(img_input)
    aug = build_data_augmentation_layer()(x_float)
    
    prep_eff = tf.keras.applications.efficientnet.preprocess_input(aug)
    prep_dense = tf.keras.applications.densenet.preprocess_input(aug)
    
    base_effnet = tf.keras.applications.EfficientNetB0(
        input_shape=img_size, include_top=False, weights='imagenet'
    )
    base_effnet.trainable = False
    sp1 = base_effnet(prep_eff)
    gap1 = layers.GlobalAveragePooling2D(name="gap_eff")(sp1)
    b1_dense = layers.Dense(256, activation='swish', name="b1_dense")(gap1)
    b1_bn = layers.BatchNormalization(name="b1_bn")(b1_dense)
    
    base_densenet = tf.keras.applications.DenseNet121(
        input_shape=img_size, include_top=False, weights='imagenet'
    )
    base_densenet.trainable = False
    sp2 = base_densenet(prep_dense)
    gap2 = layers.GlobalAveragePooling2D(name="gap_dense")(sp2)
    b2_dense = layers.Dense(256, activation='swish', name="b2_dense")(gap2)
    b2_bn = layers.BatchNormalization(name="b2_bn")(b2_dense)
    
    stream_b_concat = layers.Concatenate(name="stream_b_ensemble")([b1_bn, b2_bn])
    
    stream_a_input = layers.Input(shape=(pca_dim,), dtype='float32', name="handcrafted_input")
    stream_a_dense = layers.Dense(128, activation='relu', name="stream_a_dense")(stream_a_input)
    stream_a_bn = layers.BatchNormalization(name="stream_a_bn")(stream_a_dense)
    
    gate_a = layers.Dense(128, activation='sigmoid', name="attention_gate_stream_a")(stream_a_bn)
    gated_stream_a = layers.Multiply(name="gated_handcrafted")([stream_a_bn, gate_a])
    
    gate_b = layers.Dense(512, activation='sigmoid', name="attention_gate_stream_b")(stream_b_concat)
    gated_stream_b = layers.Multiply(name="gated_spatial_ensemble")([stream_b_concat, gate_b])
    
    fused_features = layers.Concatenate(name="dual_gated_fusion")([gated_stream_b, gated_stream_a])
    
    x = layers.Dense(384, activation='swish', kernel_regularizer=regularizers.l2(1e-4), name="classifier_dense_1")(fused_features)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    
    x = layers.Dense(192, activation='swish', kernel_regularizer=regularizers.l2(1e-4), name="classifier_dense_2")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name="classification_head")(x)
    
    hybrid_model = models.Model(inputs=[img_input, stream_a_input], outputs=outputs, name="PhytoGATE_Gated_Hybrid")
    return hybrid_model, base_effnet, base_densenet
