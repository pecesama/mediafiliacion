# -*- coding: utf-8 -*-
"""
Sistema de Media Filiación Forense v2.0

Cambios frente a v1.0:
- Agrega el flujo de 2 imágenes: frontal obligatoria + perfil opcional/recomendado.
- Migra a Google Gen AI SDK (de `google-genai`).
- Integra generación directa de retrato hablado con Gemini 3.1 Flash Image (Nano Banana 2).
"""

from __future__ import annotations

import os
from datetime import datetime
from io import BytesIO
from typing import Any, List, Tuple

import streamlit as st
from PIL import Image
from fpdf import FPDF

try:
    from google import genai
    from google.genai import types
except Exception as import_error:  # pragma: no cover
    st.error(
        "No se pudo importar `google-genai`. Instala dependencias con: "
        "`pip install -r requirements.txt`."
    )
    st.exception(import_error)
    st.stop()

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Media Filiación SNSP v2.0",
    page_icon="🕵️",
    layout="wide",
)

APP_VERSION = "2.0"
DEFAULT_ANALYSIS_MODEL = "gemini-3.5-flash"
DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image"

# ─────────────────────────────────────────────────────────────
# SECRETOS / CLIENTE GEMINI
# ─────────────────────────────────────────────────────────────

def secret_or_env(name: str, default: Any = "") -> Any:
    """Lee primero Streamlit secrets y luego variables de entorno."""
    value = None
    try:
        value = st.secrets.get(name)  # type: ignore[attr-defined]
    except Exception:
        value = None
    if value is None or value == "":
        value = os.getenv(name)
    return default if value is None or value == "" else value


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "si", "sí"}


ANALYSIS_MODEL_NAME = str(secret_or_env("GEMINI_TEXT_MODEL", DEFAULT_ANALYSIS_MODEL))
IMAGE_MODEL_NAME = str(secret_or_env("GEMINI_IMAGE_MODEL", DEFAULT_IMAGE_MODEL))
BACKEND = str(secret_or_env("GOOGLE_GENAI_BACKEND", "developer_api")).strip().lower()
API_KEY = secret_or_env("GEMINI_API_KEY") or secret_or_env("GOOGLE_API_KEY")
PROJECT = secret_or_env("GOOGLE_CLOUD_PROJECT")
LOCATION = str(secret_or_env("GOOGLE_CLOUD_LOCATION", "global"))
USE_VERTEX_FLAG = parse_bool(secret_or_env("USE_VERTEX_AI", False)) or parse_bool(
    secret_or_env("GOOGLE_GENAI_USE_VERTEXAI", False)
)
USE_ENTERPRISE_FLAG = parse_bool(secret_or_env("GOOGLE_GENAI_USE_ENTERPRISE", False))


def build_genai_client() -> genai.Client:
    """
    Construye un cliente de Google Gen AI.
    """
    if API_KEY:
        os.environ.setdefault("GEMINI_API_KEY", str(API_KEY))
        os.environ.setdefault("GOOGLE_API_KEY", str(API_KEY))

    if BACKEND in {"agent_platform", "enterprise", "google_cloud"} or USE_ENTERPRISE_FLAG:
        os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] = "True"
        if PROJECT:
            os.environ["GOOGLE_CLOUD_PROJECT"] = str(PROJECT)
        os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION or "global"
        return genai.Client(http_options=types.HttpOptions(api_version="v1"))

    if BACKEND in {"vertex", "vertex_ai"} or USE_VERTEX_FLAG:
        if not PROJECT:
            raise RuntimeError("Falta GOOGLE_CLOUD_PROJECT para usar Vertex AI / Agent Platform.")
        return genai.Client(
            vertexai=True,
            project=str(PROJECT),
            location=LOCATION or "global",
            http_options=types.HttpOptions(api_version="v1"),
        )

    if not API_KEY:
        raise RuntimeError(
            "Falta GEMINI_API_KEY. Configúrala en `.streamlit/secrets.toml` "
            "o usa GOOGLE_GENAI_BACKEND='agent_platform' con credenciales de Google Cloud."
        )
    return genai.Client(api_key=str(API_KEY))


try:
    client = build_genai_client()
except Exception as exc:
    st.error(f"No se pudo inicializar Gemini: {exc}")
    st.info(
        "Copia `.streamlit/secrets.toml.example` a `.streamlit/secrets.toml` y configura "
        "tu API key o tu proyecto de Gemini Enterprise Agent Platform."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def load_uploaded_image(uploaded_file) -> Image.Image | None:
    if uploaded_file is None:
        return None
    return Image.open(uploaded_file).convert("RGB")


def collect_text(response) -> str:
    """Extrae texto de una respuesta generate_content de forma tolerante."""
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()

    chunks: List[str] = []
    try:
        for candidate in response.candidates or []:
            for part in candidate.content.parts or []:
                if getattr(part, "text", None):
                    chunks.append(part.text)
    except Exception:
        pass
    return "\n".join(chunks).strip()


def collect_text_and_images(response) -> Tuple[str, List[Image.Image]]:
    """Extrae texto e imágenes inline de una respuesta de Gemini Image."""
    texts: List[str] = []
    images: List[Image.Image] = []

    try:
        for candidate in response.candidates or []:
            for part in candidate.content.parts or []:
                if getattr(part, "text", None):
                    texts.append(part.text)
                inline_data = getattr(part, "inline_data", None)
                if inline_data is not None and getattr(inline_data, "data", None):
                    images.append(Image.open(BytesIO(inline_data.data)).convert("RGB"))
    except Exception as parse_error:
        texts.append(f"No se pudieron leer todas las partes de la respuesta: {parse_error}")

    return "\n".join(texts).strip(), images


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_generation_config():
    """Config documentada: solicitar modalidades texto + imagen; fallback a strings."""
    try:
        return types.GenerateContentConfig(
            response_modalities=[types.Modality.TEXT, types.Modality.IMAGE]
        )
    except Exception:
        return types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])

def generar_pdf_filiacion(num_ficha: str, metadata: str, texto_markdown: str) -> bytes:
    """Construye un documento PDF institucional con los resultados."""
    class PDFReport(FPDF):
        def header(self):
            # Encabezado Oficial
            self.set_font("helvetica", "B", 14)
            self.cell(0, 10, "SISTEMA NACIONAL DE SEGURIDAD PÚBLICA", align="C", new_x="LMARGIN", new_y="NEXT")
            self.set_font("helvetica", "B", 12)
            self.cell(0, 8, "REGISTRO NACIONAL DE IDENTIFICACIÓN - MEDIA FILIACIÓN", align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(5)

        def footer(self):
            # Pie de página con numeración
            self.set_y(-15)
            self.set_font("helvetica", "I", 8)
            self.cell(0, 10, f"Página {self.page_no()}/{{nb}}", align="C")

    pdf = PDFReport()
    pdf.add_page()
    
    # --- SECCIÓN DE METADATOS ---
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 8, "DATOS DEL EXPEDIENTE:", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("helvetica", size=10)
    # Escribimos cada línea de los metadatos
    for linea in metadata.split('\n'):
        if linea.strip():
            pdf.cell(0, 6, linea.strip(), new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(5)
    
    # --- SECCIÓN DE RESULTADOS (Limpieza de Markdown) ---
    pdf.set_font("helvetica", size=10)
    # El modelo de IA nos da Markdown (con **, ##, y tablas con |). 
    # Hacemos una limpieza básica para que se lea limpio en el PDF.
    for linea in texto_markdown.split('\n'):
        linea_limpia = linea.replace('**', '').replace('##', '').replace('|', '  ')
        if linea_limpia.strip() == '---':
            pdf.ln(5)
            continue
            
        pdf.multi_cell(0, 6, linea_limpia, new_x="LMARGIN", new_y="NEXT")
        
    # --- AVISO LEGAL AL FINAL ---
    pdf.ln(10)
    pdf.set_font("helvetica", "I", 8)
    aviso = "Advertencia: Resultado asistido por IA. Requiere validación humana y uso conforme a normativa aplicable. Herramienta de apoyo técnico, no debe usarse para identificación automática concluyente."
    pdf.multi_cell(0, 5, aviso, new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())

# ─────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT_FORENSE = """
ACTÚA COMO AUXILIAR TÉCNICO PARA MEDIA FILIACIÓN CON VOCABULARIO SNSP DE MÉXICO.
Tu tarea es describir rasgos visibles. No identifiques a la persona, no infieras nombre,
nacionalidad, etnia, origen, ocupación, antecedentes, religión, estado de salud, ni identidad.

IMÁGENES DISPONIBLES: {imagenes_disponibles}
LADO DE PERFIL PROPORCIONADO: {lado_perfil}

═══════════════════════════════════════════════
REGLAS CRÍTICAS
═══════════════════════════════════════════════
1. Usa solo vocabulario controlado. No inventes categorías.
2. Si un rasgo no es visible, escribe: NO VISIBLE.
3. Si un rasgo requiere perfil y no hay perfil útil, escribe: REQUIERE FOTO DE PERFIL.
4. Si un rasgo no puede determinarse con certeza, escribe: NO DETERMINABLE DESDE LA VISTA DISPONIBLE.
5. Agrega confianza por rasgo: ALTA, MEDIA, BAJA o —.
6. Color de piel debe ser solo una categoría técnica del catálogo; NO lo traduzcas a etnia, herencia o nacionalidad.
7. Complexión no debe inferirse con alta confianza si solo se ve rostro; usa BAJA o NO VISIBLE.
8. Estatura, peso y sangre/RH no se estiman desde foto.
9. Si hay perfil izquierdo, no lo trates como oreja derecha; aclara que la oreja derecha no fue visible si aplica.

═══════════════════════════════════════════════
VISTAS NECESARIAS
═══════════════════════════════════════════════
FRONTAL:
- Forma de cara, cabello, frente altura/ancho, cejas, ojos, nariz raíz/altura/ancho,
  boca, labios, mentón tipo/forma, accesorios y señas visibles.

PERFIL:
- Frente inclinación.
- Nariz dorso.
- Nariz base.
- Mentón inclinación.
- Oreja visible, idealmente derecha.

═══════════════════════════════════════════════
VOCABULARIO CONTROLADO
═══════════════════════════════════════════════
Sexo: Masculino | Femenino | NO DETERMINABLE
Complexión: Delgada | Regular | Robusta | Obesa | Atlética | NO VISIBLE
Color de Piel: Albino | Amarillo | Blanco | Moreno Claro | Moreno | Moreno Obscuro | Negro
Cara — Forma: Alargada | Cuadrada | Ovalada | Redonda
Cabello — Cantidad: Abundante | Escaso | Regular | Sin Cabello
Cabello — Color: Albino | Cano Total | Castaño Obscuro | Castaño Claro | Entrecano | Negro | Pelirojo | Rubio
Cabello — Forma: Crespo | Lacio | Ondulado | Rizado
Cabello — Calvicie: Frontal | Tonsural | Frontoparietal | Total | Ninguna
Cabello — Implantación: En Punta | Circular | Rectangular | NO VISIBLE
Frente — Altura: Grande | Mediana | Pequeña
Frente — Inclinación: Oblicua | Intermedia | Vertical | Prominente
Frente — Ancho: Grande | Mediana | Pequeña
Cejas — Dirección: Horizontal | Internas | Externas
Cejas — Implantación: Altas | Bajas | Próximas | Separadas
Cejas — Forma: Arqueadas | Arqueadas Sinuosas | Rectilíneas | Rectilíneas Sinuosas
Cejas — Tamaño: Gruesas | Delgadas | Largas | Cortas
Ojos — Color: Azul | Café Claro | Café Oscuro | Verde | Gris | NO DETERMINABLE
Ojos — Forma: Alargados | Redondos | Ovales
Ojos — Tamaño: Grandes | Pequeños | Regulares
Nariz — Raíz: Pequeña | Grande | Mediana
Nariz — Dorso: Cóncavo | Convexo | Recto | Sinuoso
Nariz — Base: Abatida | Horizontal | Levantada
Nariz — Altura: Pequeña | Grande | Mediana
Nariz — Ancho: Grande | Mediana | Pequeña
Boca — Tamaño: Grande | Mediana | Pequeña
Boca — Comisuras: Abatidas | Elevadas | Simétricas
Labios — Espesor: Delgados | Medianos | Gruesos | Morrudos
Labios — Altura Naso-labial: Grande | Mediana | Pequeña
Labios — Prominencia: Labio Inferior | Ninguna
Oreja D — Forma: Cuadrada | Ovalada | Redonda | Triangular | NO VISIBLE
Oreja D — Hélix: Grande | Mediano | Pequeño | NO VISIBLE
Oreja D — Adherencia: Unida | Separada | Muy separada | NO VISIBLE
Oreja D — Lóbulo Forma: Descendente | En Golfo | Escuadra | Intermedio | NO VISIBLE
Oreja D — Lóbulo Particularidad: Perforado | Foseta | Islote | Ninguna | NO VISIBLE
Oreja D — Lóbulo Dimensión: Grande | Mediano | Pequeño | NO VISIBLE
Mentón — Tipo: Foseta | Bilovado | Borla | Normal
Mentón — Forma: Oval | Cuadrado | En Punta
Mentón — Inclinación: Huyente | Prominente | Vertical
Usa lentes: Sí | No

Señas particulares:
Formato recomendado: TIPO-LADO-REGIÓN-VISTA-CANTIDAD.
Tipos: 1.Cicatriz | 2.Tatuaje | 3.Lunar | 4.Defecto | 5.Prótesis.

═══════════════════════════════════════════════
FORMATO DE SALIDA
═══════════════════════════════════════════════
## FICHA DE MEDIA FILIACIÓN

| Rasgo | Descripción | Confianza | Vista usada | Observación |
|---|---|---|---|---|
| Sexo | ... | ... | FRONTAL | ... |
| Complexión | ... | ... | FRONTAL | ... |
| Color de Piel | ... | ... | FRONTAL | ... |
| Cara — Forma | ... | ... | FRONTAL | ... |
| Cabello — Cantidad | ... | ... | FRONTAL/PERFIL | ... |
| Cabello — Color | ... | ... | FRONTAL/PERFIL | ... |
| Cabello — Forma | ... | ... | FRONTAL/PERFIL | ... |
| Cabello — Calvicie | ... | ... | FRONTAL/PERFIL | ... |
| Cabello — Implantación | ... | ... | FRONTAL | ... |
| Frente — Altura | ... | ... | FRONTAL | ... |
| Frente — Inclinación | ... | ... | PERFIL | ... |
| Frente — Ancho | ... | ... | FRONTAL | ... |
| Cejas — Dirección | ... | ... | FRONTAL | ... |
| Cejas — Implantación | ... | ... | FRONTAL | ... |
| Cejas — Forma | ... | ... | FRONTAL | ... |
| Cejas — Tamaño | ... | ... | FRONTAL | ... |
| Ojos — Color | ... | ... | FRONTAL | ... |
| Ojos — Forma | ... | ... | FRONTAL | ... |
| Ojos — Tamaño | ... | ... | FRONTAL | ... |
| Nariz — Raíz | ... | ... | FRONTAL/PERFIL | ... |
| Nariz — Dorso | ... | ... | PERFIL | ... |
| Nariz — Base | ... | ... | PERFIL | ... |
| Nariz — Altura | ... | ... | FRONTAL/PERFIL | ... |
| Nariz — Ancho | ... | ... | FRONTAL | ... |
| Boca — Tamaño | ... | ... | FRONTAL | ... |
| Boca — Comisuras | ... | ... | FRONTAL | ... |
| Labios — Espesor | ... | ... | FRONTAL | ... |
| Labios — Altura Naso-labial | ... | ... | FRONTAL/PERFIL | ... |
| Labios — Prominencia | ... | ... | FRONTAL/PERFIL | ... |
| Oreja D — Forma | ... | ... | PERFIL/FRONTAL | ... |
| Oreja D — Hélix | ... | ... | PERFIL/FRONTAL | ... |
| Oreja D — Adherencia | ... | ... | PERFIL/FRONTAL | ... |
| Oreja D — Lóbulo Forma | ... | ... | PERFIL/FRONTAL | ... |
| Oreja D — Lóbulo Particularidad | ... | ... | PERFIL/FRONTAL | ... |
| Oreja D — Lóbulo Dimensión | ... | ... | PERFIL/FRONTAL | ... |
| Mentón — Tipo | ... | ... | FRONTAL | ... |
| Mentón — Forma | ... | ... | FRONTAL | ... |
| Mentón — Inclinación | ... | ... | PERFIL | ... |
| Accesorios — Lentes | ... | ... | FRONTAL | ... |

## SEÑAS PARTICULARES
| Código | Tipo | Descripción | Ubicación | Vista |
|---|---|---|---|---|
[filas o "Ninguna observable"]

## RESUMEN NARRATIVO
[Párrafo breve, neutral y útil para búsqueda; omite rasgos no determinados.]

## ADVERTENCIAS DEL ANÁLISIS
[Limitaciones por calidad, falta de perfil, oclusiones, iluminación, cabello cubriendo oreja, etc.]
"""

def build_portrait_prompt(texto_filiacion: str, estilo: str, angulo: str, aspect_ratio: str, acabado: str) -> str:
    return f"""
Genera UNA imagen de retrato hablado neutral a partir de la siguiente ficha técnica de media filiación.

La imagen debe representar una persona adulta genérica basada únicamente en los rasgos físicos descritos.
No intentes identificar a una persona real. No agregues rasgos no descritos. No infieras etnia,
nacionalidad, religión, profesión, condición social, antecedentes ni culpabilidad.

No incluyas texto, etiquetas, marcas de agua visibles, logos institucionales, armas, uniformes, esposas,
escena de delito ni ambiente policial. Fondo liso y neutro.

ESTILO: {estilo}
ÁNGULO / COMPOSICIÓN: {angulo}
RELACIÓN DE ASPECTO SOLICITADA: {aspect_ratio}
ACABADO: {acabado}

FICHA TÉCNICA:
{texto_filiacion}

Instrucciones visuales:
- Expresión neutra e iluminación uniforme.
- Mantén proporciones faciales coherentes con la ficha.
- Si la composición es "frontal + perfil", crea un solo lienzo con ambas vistas lado a lado, sin texto.
- Si un rasgo dice NO VISIBLE, REQUIERE FOTO DE PERFIL o NO DETERMINABLE, no lo inventes ni lo enfatices.
""".strip()

# ─────────────────────────────────────────────────────────────
# UI PRINCIPAL
# ─────────────────────────────────────────────────────────────

# 1. CSS para reducir el espacio en blanco superior
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        [data-testid="stMetricValue"] { font-size: 14px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🕵️ Sistema de Media Filiación Forense · v2.0")

st.markdown("---")

tab1, tab2 = st.tabs(["📸 FOTO → FICHA", "📝 FICHA → IMAGEN"])

# ══════════════════════════════════════════
# TAB 1 — FOTO A FICHA
# ══════════════════════════════════════════
with tab1:
    col_ev, col_res = st.columns([1, 2])

    with col_ev:
        st.subheader("Evidencia fotográfica")

        st.markdown("##### 📷 Vista frontal *(obligatoria)*")
        uploaded_frontal = st.file_uploader(
            "Sube la fotografía frontal",
            type=["jpg", "jpeg", "png"],
            key="frontal",
        )

        st.markdown("##### 📷 Vista de perfil *(opcional, recomendada)*")
        lado_perfil = st.radio(
            "Lado de perfil",
            ["Perfil derecho", "Perfil izquierdo", "Perfil no especificado"],
            horizontal=True,
        )
        st.caption(
            "La segunda imagen permite determinar frente inclinación, nariz dorso/base, "
            "mentón inclinación y oreja visible."
        )
        uploaded_perfil = st.file_uploader(
            "Sube una fotografía lateral de perfil",
            type=["jpg", "jpeg", "png"],
            key="perfil",
        )

        img_frontal = load_uploaded_image(uploaded_frontal)
        img_perfil = load_uploaded_image(uploaded_perfil)

        if img_frontal:
            st.image(img_frontal, caption="Vista frontal", use_container_width=True)
        if img_perfil:
            st.image(img_perfil, caption=lado_perfil, use_container_width=True)

        if img_frontal and img_perfil:
            st.success("✅ Captura suficiente: frontal + perfil")
        elif img_frontal:
            st.warning("⚠️ Solo frontal: los rasgos de perfil se marcarán como no determinables")

        with st.expander("📋 Metadatos"):
            num_ficha = st.text_input("Número de ficha", placeholder="MF-2026-001")
            revisor = st.text_input("Persona revisora / perito", placeholder="Nombre completo")
            expediente = st.text_input("No. de expediente", placeholder="Número")

        with st.expander("📖 ¿Qué aporta cada vista?"):
            st.success("**Frontal:** cara, cabello, frente altura/ancho, cejas, ojos, nariz raíz/altura/ancho, boca, labios, mentón tipo/forma, accesorios y señas visibles.", icon="🟢")
            st.info("**Perfil:** frente inclinación, nariz dorso, nariz base y mentón inclinación. También ayuda con oreja y contorno lateral.", icon="🔵")

        analyze_btn = st.button(
            "🔍 GENERAR MEDIA FILIACIÓN",
            type="primary",
            disabled=not bool(img_frontal),
        )

    with col_res:
        st.subheader("Ficha técnica SNSP")

        with st.expander("ℹ️ Leyenda de confianza"):
            st.markdown(
                """
| Nivel | Criterio |
|---|---|
| **ALTA** | Rasgo claramente visible, sin ambigüedad |
| **MEDIA** | Visible con cierta ambigüedad |
| **BAJA** | Estimado con poca certeza visual |
| **—** | No visible, no determinable o requiere otra vista |
"""
            )

        if analyze_btn and img_frontal:
            with st.spinner("Procesando imágenes y aplicando vocabulario controlado…"):
                try:
                    imagenes_disponibles = "FRONTAL" + (f" + {lado_perfil.upper()}" if img_perfil else "")
                    prompt = SYSTEM_PROMPT_FORENSE.format(
                        imagenes_disponibles=imagenes_disponibles,
                        lado_perfil=lado_perfil,
                    )

                    contents: List[Any] = [prompt, "IMAGEN FRONTAL:", img_frontal]
                    if img_perfil:
                        contents += [f"IMAGEN DE PERFIL ({lado_perfil}):", img_perfil]

                    response = client.models.generate_content(
                        model=ANALYSIS_MODEL_NAME,
                        contents=contents,
                    )
                    resultado = collect_text(response)
                    if not resultado:
                        raise RuntimeError("El modelo no devolvió texto legible.")

                    st.markdown(resultado)
                    st.success("✅ Análisis completado. Requiere revisión humana antes de uso formal.")

                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    imagenes_label = imagenes_disponibles
                    metadata = (
                        f"Fecha: {ts}\n"
                        f"Ficha: {num_ficha or '—'}\n"
                        f"Revisor/Perito: {revisor or '—'}\n"
                        f"Expediente: {expediente or '—'}\n"
                        f"Modelo IA: {ANALYSIS_MODEL_NAME}\n"
                        f"Imágenes: {imagenes_label}\n"
                    )
                    st.markdown("---")
                    st.markdown(metadata.replace("\n", "  \n"))

                    # 1. Mostrar metadatos en pantalla
                    st.markdown("---")
                    st.markdown(metadata.replace("\n", "  \n"))

                    # 2. Construir documento PDF en memoria
                    pdf_bytes = generar_pdf_filiacion(num_ficha, metadata, resultado)
                    
                    # 3. Botón de descarga de PDF
                    filename = f"Media_Filiacion_{num_ficha or 'SNSP'}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                    st.download_button(
                        label="📥 Descargar Ficha Formal (.pdf)",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        type="primary" # Lo resalta en color
                    )

                    st.session_state["ultima_ficha"] = resultado
                    st.info("La ficha quedó disponible en la pestaña FICHA → IMAGEN.")

                except Exception as e:
                    st.error(f"Error en el procesamiento: {e}")
                    st.info("Verifica credenciales, disponibilidad del modelo y tamaño/calidad de las imágenes.")
        elif not img_frontal:
            st.info("👆 Sube una fotografía frontal para iniciar.")

# ══════════════════════════════════════════
# TAB 2 — FICHA A IMAGEN DIRECTA
# ══════════════════════════════════════════
with tab2:
    st.header("🎨 Generación del retrato hablado")

    valor_default = st.session_state.get("ultima_ficha", "")
    usar_anterior = False
    if valor_default:
        usar_anterior = st.checkbox("Usar la ficha generada en la pestaña anterior", value=True)

    texto_filiacion = st.text_area(
        "Ficha técnica SNSP o descripción de media filiación:",
        value=valor_default if usar_anterior else "",
        height=240,
        placeholder=(
            "Ejemplo: persona de sexo masculino, complexión regular, color de piel moreno, "
            "cara ovalada, cabello negro lacio regular, cejas rectilíneas gruesas, "
            "ojos café oscuro ovales, nariz mediana, labios medianos, mentón normal…"
        ),
    )

    col_o1, col_o2, col_o3 = st.columns(3)
    
    with col_o1:
        estilo = st.selectbox(
            "Estilo de imagen",
            [
                "Retrato de identificación neutral, fotográfico",
                "Boceto forense técnico, lápiz, blanco y negro",
                "Retrato hiperrealista sobrio, fondo neutro",
                "Ilustración técnica limpia, no caricatura",
            ],
            index=1 # Preseleccionamos "Boceto forense" como valor por defecto
        )
        
    with col_o2:
        angulo = st.selectbox(
            "Ángulo / composición",
            [
                "Frontal (Vista directa a la cámara)",
                "Perfil (Vista completamente de lado)",
                "Tres cuartos (Rostro ligeramente girado)",
                "Vista doble (Frente y perfil en la misma imagen)",
            ],
        )
        
    with col_o3:
        # Lógica de HCI: Si elige Vista doble, sugerimos Horizontal (índice 3). 
        # Si elige un solo rostro, sugerimos Vertical (índice 0).
        indice_proporcion = 3 if "Vista doble" in angulo else 0
        
        aspect_ratio = st.selectbox(
            "Formato de imagen (Proporción)",
            [
                "3:4 (Vertical / Tipo retrato estándar)",
                "1:1 (Cuadrada)",
                "4:5 (Vertical más corto)",
                "4:3 (Horizontal tradicional)",
                "16:9 (Panorámica / Pantalla ancha)",
            ],
            index=indice_proporcion
        )

    acabado = st.selectbox(
        "Acabado",
        [
            "alta nitidez, iluminación uniforme, fondo gris claro",
            "documental, neutro, sin dramatización",
            "boceto forense, líneas limpias, fondo neutro",
        ],
        index=2 # Preseleccionamos el boceto forense por seguridad legal
    )

    prompt_final = ""
    if texto_filiacion.strip():
        prompt_final = build_portrait_prompt(
            texto_filiacion=texto_filiacion.strip(),
            estilo=estilo,
            angulo=angulo,
            aspect_ratio=aspect_ratio,
            acabado=acabado,
        )

    with st.expander("🔎 Ver prompt final"):
        st.code(prompt_final or "Ingresa una ficha para construir el prompt.", language="text")

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        st.download_button(
            "📋 Descargar prompt (.txt)",
            prompt_final or "",
            f"prompt_retrato_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            "text/plain",
            disabled=not bool(prompt_final),
        )
    with col_btn2:
        st.info("💡 Copia o descarga el prompt y pégalo en herramientas generadoras de imágenes.")
        
        # --- OPCIÓN 1: GENERACIÓN DIRECTA API ---
        # Código comentado temporalmente por límites de cuota (Requiere facturación activa)
        # generar_imagen = st.button(
        #     "🖼️ Generar imagen con Gemini",
        #     type="primary",
        #     disabled=not bool(prompt_final),
        # )

    # --- LÓGICA DE GENERACIÓN DE IMAGEN COMENTADA ---
    # if generar_imagen and prompt_final:
    #     with st.spinner(f"Generando imagen con {IMAGE_MODEL_NAME}…"):
    #         try:
    #             response_img = client.models.generate_content(
    #                 model=IMAGE_MODEL_NAME,
    #                 contents=prompt_final,
    #                 config=image_generation_config(),
    #             )
    #             texto_modelo, imagenes = collect_text_and_images(response_img)
    #
    #             if texto_modelo:
    #                 st.markdown("#### Respuesta del modelo")
    #                 st.markdown(texto_modelo)
    #
    #             if not imagenes:
    #                 st.warning(
    #                     "El modelo no devolvió imagen. Puede deberse a credenciales, cuota, disponibilidad del modelo "
    #                     "o filtros de seguridad."
    #                 )
    #             else:
    #                 st.markdown("#### Imagen generada")
    #                 ts_img = datetime.now().strftime("%Y%m%d_%H%M%S")
    #                 for idx, imagen in enumerate(imagenes, start=1):
    #                     st.image(imagen, caption=f"Retrato generado {idx}", use_container_width=True)
    #                     st.download_button(
    #                         f"📥 Descargar imagen {idx} (.png)",
    #                         image_to_png_bytes(imagen),
    #                         f"retrato_hablado_{ts_img}_{idx}.png",
    #                         "image/png",
    #                         key=f"download_img_{idx}_{ts_img}",
    #                     )
    #
    #         except Exception as e:
    #             st.error(f"Error al generar imagen: {e}")
    #             st.info(
    #                 "Verifica que tu modelo de imagen esté disponible en tu cuenta/proyecto, "
    #                 "que Agent Platform esté habilitado si usas Google Cloud y que tengas cuota."
    #             )

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
left_f, right_f = st.columns([3, 1])
with left_f:
    st.caption(
        "Sistema basado en el Registro Nacional de Identificación — Media Filiación del SNSP. "
        "⚠️ **Resultado asistido por IA**; requiere revisión humana y uso conforme a normativa aplicable."
    )
with right_f:
    st.caption(
        "Desarrollado por el IHCLab de la Facultad de Telemática en la Universidad de Colima",
        unsafe_allow_html=True
    )