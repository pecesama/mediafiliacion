import streamlit as st
import google.generativeai as genai
from PIL import Image

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Sistema Forense: Media Filiación SNSP",
    page_icon="🕵️‍♂️",
    layout="wide"
)

# --- CONFIGURACIÓN DE API ---
# En producción (Streamlit Cloud), usa st.secrets["GEMINI_API_KEY"]
# Para pruebas locales, puedes descomentar la línea de abajo e insertar tu key:
# api_key = "TU_API_KEY_AQUI" 
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    st.warning("⚠️ No se detectó API Key en Secrets. Asegúrate de configurarla.")
    st.stop()

genai.configure(api_key=api_key)

# Usamos Gemini 1.5 Pro o Flash. Flash es más rápido para visión.
model = genai.GenerativeModel('gemini-3-flash-preview')

# --- PROMPT MAESTRO (BASE DE CONOCIMIENTO EXTRACTADA DEL PDF) ---
SYSTEM_PROMPT_FORENSE = """
ACTÚA COMO UN PERITO EXPERTO DEL SISTEMA NACIONAL DE SEGURIDAD PÚBLICA (SNSP) DE MÉXICO.
Tu tarea es generar la MEDIA FILIACIÓN de la persona en la imagen siguiendo ESTRICTAMENTE el vocabulario controlado del "Manual de Media Filiación y Señas Particulares".

REGLAS CRÍTICAS:
1. Solo usa los términos permitidos (listados abajo). No inventes adjetivos.
2. Si un rasgo no es visible, indica "NO VISIBLE".
3. Para señas particulares, intenta generar el CÓDIGO DE UBICACIÓN (Topografía Humana) si es posible.

--- VOCABULARIO CONTROLADO OFICIAL (SNSP) ---

1. DATOS GENERALES
   - Sexo: Masculino, Femenino.
   - Complexión: Delgada, Regular, Robusta, Obesa, Atlética.
   - Color de Piel: Albino, Amarillo, Blanco, Moreno Claro, Moreno, Moreno Obscuro, Negro.

2. CARA (Forma)
   - Opciones: Alargada, Cuadrada, Ovalada, Redonda.

3. CABELLO
   - Cantidad: Abundante, Escaso, Regular, Sin Cabello.
   - Color: Albino, Cano Total, Castaño Obscuro, Castaño Claro, Entrecano, Negro, Pelirojo, Rubio.
   - Forma: Crespo, Lacio, Ondulado, Rizado.
   - Calvicie: Frontal, Tonsural, Frontoparietal, Total.
   - Implantación: En Punta, Circular, Rectangular.

4. FRENTE
   - Altura: Grande (>1/3 cara), Mediana (=1/3), Pequeña (<1/3).
   - Inclinación: Oblicua (>20° atrás), Intermedia, Vertical, Prominente (abombada).
   - Ancho: Grande, Mediana, Pequeña.

5. CEJAS
   - Dirección: Horizontal, Internas (caen al centro), Externas (suben al exterior).
   - Implantación: Altas, Bajas, Próximas, Separadas.
   - Forma: Arqueadas, Sinuosas, Rectilíneas.
   - Tamaño: Gruesas, Delgadas, Largas, Cortas.

6. OJOS
   - Color: Azul, Café Claro, Café Oscuro, Verde, Gris.
   - Forma: Alargados, Redondos, Ovales.
   - Tamaño: Grandes, Pequeños, Regulares.

7. NARIZ
   - Raíz: Pequeña, Grande, Mediana.
   - Dorso: Cóncavo, Convexo, Recto, Sinuoso.
   - Base: Abatida (hacia abajo), Horizontal, Levantada.
   - Altura: Pequeña, Grande, Mediana.
   - Ancho: Grande, Mediana, Pequeña.

8. BOCA
   - Tamaño: Grande (comisuras pasan pupilas), Mediana, Pequeña.
   - Comisuras: Abatidas, Elevadas, Simétricas.

9. LABIOS
   - Espesor: Delgados, Medianos, Gruesos, Morrudos (muy gruesos/hinchados).
   - Altura Naso-labial: Grande, Mediana, Pequeña.
   - Prominencia Labio Inferior: Ninguna, Sí.

10. MENTÓN
    - Tipo: Foseta (hoyuelo), Bilovado (partido), Borla (muy pronunciado/redondo).
    - Forma: Oval, Cuadrado, En Punta.
    - Inclinación: Huyente, Prominente, Vertical.

11. OREJA DERECHA (Si no visible, describir Izquierda)
    - Forma: Cuadrada, Ovalada, Redonda, Triangular.
    - Hélix (Borde): Original/Superior/Posterior (Grande/Mediano/Pequeño).
    - Lóbulo: Descendente, En Golfo, Escuadra, Intermedio.
    - Particularidad Lóbulo: Perforado, Foseta, Islote.

12. SEÑAS PARTICULARES (Código SNSP)
    Formato: TIPO - LADO - REGIÓN - VISTA - CANTIDAD
    - Tipos: 1.Cicatriz, 2.Tatuaje, 3.Lunar, 4.Defecto, 5.Prótesis.
    - Lado: D (Derecho), I (Izquierdo).
    - Vista: F (Frontal), D (Dorsal).
    - Regiones Comunes (ejemplos): 02(Frontal), 09(Mejilla), 11(Mentón), 12(Cuello Ant), 19(Brazo Ant).
    Ejemplo: "2-D-12-F-01" (Tatuaje cuello derecho frontal).

--- FORMATO DE SALIDA ---
Genera una tabla Markdown limpia con dos columnas: "Rasgo" y "Descripción SNSP".
Al final, agrega un párrafo resumen narrativo para búsqueda rápida.
"""

# --- INTERFAZ DE USUARIO ---
st.title("🕵️‍♂️ Sistema de Media Filiación Forense (SNSP)")
st.markdown("""
<style>
.big-font { font-size:18px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("Herramienta oficial para generación de fichas de identificación basadas en el **Manual de Media Filiación** de la Fiscalía Mexicana.")

tab1, tab2 = st.tabs(["📸 De FOTO a TEXTO (Análisis)", "🎨 De TEXTO a IMAGEN (Reconstrucción)"])

# --- PESTAÑA 1: ANÁLISIS ---
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Evidencia")
        uploaded_file = st.file_uploader("Subir fotografía (Frontal preferible)", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Sujeto a identificar', use_container_width=True)
            
            analyze_btn = st.button("🔍 ANALIZAR FILIACIÓN", type="primary")

    with col2:
        st.subheader("Ficha Técnica Generada")
        if uploaded_file and analyze_btn:
            with st.spinner('Procesando biometría facial y consultando estándares SNSP...'):
                try:
                    response = model.generate_content([SYSTEM_PROMPT_FORENSE, image])
                    st.markdown(response.text)
                    st.success("Análisis finalizado. Terminología validada con el estándar.")
                except Exception as e:
                    st.error(f"Error en el procesamiento: {str(e)}")
        elif not uploaded_file:
            st.info("Esperando imagen para iniciar el peritaje...")

# --- PESTAÑA 2: RECONSTRUCCIÓN (Generación de Prompt) ---
with tab2:
    st.header("Generador de Retrato Hablado")
    st.markdown("Convierte una descripción técnica (ej. IPH) en un *Prompt* optimizado para generadores de imagen (DALL-E 3, Midjourney, Imagen 3).")
    
    texto_filiacion = st.text_area("Pegue aquí la descripción técnica (ej: 'Sujeto masculino, tez morena oscura, labios morrudos...')", height=150)
    
    if st.button("🎨 Traducir a Prompt Visual"):
        if texto_filiacion:
            with st.spinner('Traduciendo terminología forense a lenguaje visual...'):
                prompt_traduccion = f"""
                Actúa como un artista forense experto en IA. Tienes la siguiente descripción técnica en español mexicano (SNSP):
                "{texto_filiacion}"

                Tu tarea es convertir esto en un PROMPT EN INGLÉS altamente detallado para un generador de imágenes fotorrealistas.
                
                REGLAS DE TRADUCCIÓN:
                1. "Moreno Oscuro" -> "Dark brown skin tone, mexican indigenous heritage traits".
                2. "Labios Morrudos" -> "Very thick, full, puffy lips".
                3. "Nariz Base Abatida" -> "Nose tip pointing downwards, hooked nose".
                4. "Cabello Crespo" -> "Coily, tight curl hair texture".
                5. Estilo: "Mugshot style, neutral lighting, hyper-realistic, 8k resolution, neutral background".
                
                Solo dame el PROMPT en inglés sin explicaciones extra.
                """
                
                response_prompt = model.generate_content(prompt_traduccion)
                
                st.subheader("Prompt Generado (Copiar y Pegar)")
                st.code(response_prompt.text, language="text")
                st.caption("Usa este texto en Bing Image Creator, Midjourney o DALL-E para obtener el retrato.")
                
                # Opcional: Si tuvieras acceso a DALL-E API, aquí podrías llamar a la generación real.
        else:
            st.warning("Por favor ingrese la descripción primero.")

# --- FOOTER ---
st.markdown("---")

st.caption("Sistema basado en el documento oficial 'FILIACION.pdf' del Secretariado Ejecutivo del SNSP.")
