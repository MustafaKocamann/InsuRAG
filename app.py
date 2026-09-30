"""
Insurellm Enterprise RAG Intelligence Platform
Professional Corporate Gradio Interface
"""

import os
import glob
from pathlib import Path
from typing import Generator, List, Tuple, Any, Dict

import gradio as gr
from dotenv import load_dotenv

from implementation.answer import (
    stream_answer_question,
    fetch_context,
    DEFAULT_MODEL,
    DEFAULT_RETRIEVAL_K,
    SYSTEM_PROMPT,
)
from implementation.ingest import run_ingestion, KNOWLEDGE_BASE, DB_NAME

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Data & Stats Helpers
# ---------------------------------------------------------------------------
def get_kb_stats() -> Dict[str, Any]:
    """Scan the knowledge base and return category statistics."""
    kb_path = Path(KNOWLEDGE_BASE)
    categories = ["company", "products", "contracts", "employees"]
    stats: Dict[str, Any] = {}
    total_files = 0

    for cat in categories:
        cat_dir = kb_path / cat
        if cat_dir.exists():
            files = list(cat_dir.glob("**/*.md"))
            stats[cat] = len(files)
            total_files += len(files)
        else:
            stats[cat] = 0

    stats["total"] = total_files
    return stats


# ---------------------------------------------------------------------------
# Custom Enterprise Theme and CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ==========================================================================
   INSURELLM ENTERPRISE DESIGN SYSTEM
   ========================================================================== */

:root {
    --ins-primary: #0F172A;
    --ins-accent: #2563EB;
    --ins-accent-hover: #1D4ED8;
    --ins-accent-soft: rgba(37, 99, 235, 0.1);
    --ins-success: #10B981;
    --ins-card-bg: #FFFFFF;
    --ins-border: #E2E8F0;
    --ins-text: #0F172A;
    --ins-text-muted: #64748B;
    --radius-lg: 14px;
    --radius-md: 10px;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -1px rgba(0,0,0,0.04);
    --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -2px rgba(0,0,0,0.03);
}

.gradio-container {
    max-width: 1440px !important;
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* Header Container */
.header-wrapper {
    background: linear-gradient(135deg, #0A192F 0%, #1E3A8A 50%, #0F172A 100%);
    border-radius: var(--radius-lg);
    padding: 24px 32px;
    margin-bottom: 20px;
    color: #FFFFFF;
    box-shadow: 0 12px 30px -10px rgba(15, 23, 42, 0.35);
    display: flex;
    justify-content: space-between;
    align-items: center;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.header-brand {
    display: flex;
    align-items: center;
    gap: 16px;
}

.header-logo-icon {
    font-size: 38px;
    background: rgba(255, 255, 255, 0.12);
    width: 60px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: inset 0 2px 4px rgba(255, 255, 255, 0.2);
}

.header-title-box h1 {
    margin: 0;
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #FFFFFF !important;
}

.header-title-box p {
    margin: 4px 0 0 0;
    font-size: 13.5px;
    color: #94A3B8;
    font-weight: 400;
}

.header-badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.status-badge {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(8px);
    padding: 7px 13px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    color: #E2E8F0;
    display: flex;
    align-items: center;
    gap: 6px;
}

.status-dot-active {
    width: 8px;
    height: 8px;
    background-color: #10B981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10B981;
    display: inline-block;
}

/* Question Chips / Fast Action Buttons */
.chips-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 12px;
}

.chip-btn {
    border-radius: 20px !important;
    padding: 6px 14px !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    border: 1px solid var(--ins-border) !important;
    background: #F8FAFC !important;
    color: #334155 !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

.chip-btn:hover {
    background: #EFF6FF !important;
    border-color: #93C5FD !important;
    color: #1D4ED8 !important;
    transform: translateY(-1px);
}

/* Source Citation Card Styling */
.source-panel-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: var(--radius-lg);
    padding: 16px;
    height: 520px;
    overflow-y: auto;
}

.source-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: var(--radius-md);
    padding: 12px 14px;
    margin-bottom: 12px;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease;
}

.source-card:hover {
    border-color: #3B82F6;
    box-shadow: var(--shadow-md);
}

.source-badge {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 4px;
    display: inline-block;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}

.badge-products { background: #DBEAFE; color: #1E40AF; }
.badge-contracts { background: #FEF3C7; color: #92400E; }
.badge-employees { background: #E0E7FF; color: #3730A3; }
.badge-company { background: #DCFCE7; color: #166534; }
.badge-general { background: #F1F5F9; color: #475569; }

.source-filename {
    font-size: 13px;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 4px;
}

.source-content {
    font-size: 12px;
    color: #475569;
    line-height: 1.5;
    background: #F8FAFC;
    padding: 8px 10px;
    border-radius: 6px;
    border-left: 3px solid #3B82F6;
    max-height: 140px;
    overflow-y: auto;
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
}

/* Stat Cards in Tab 2 */
.stat-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: var(--radius-lg);
    padding: 20px;
    box-shadow: var(--shadow-sm);
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.stat-number {
    font-size: 32px;
    font-weight: 800;
    color: #1E3A8A;
    margin: 8px 0 4px 0;
}

.stat-label {
    font-size: 13px;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Architecture Flow Steps */
.arch-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: var(--radius-md);
    padding: 16px;
    margin-bottom: 12px;
    border-left: 4px solid #2563EB;
}
"""


def render_sources_html(docs: List[Any]) -> str:
    """Format retrieved Document chunks into a clean, corporate HTML citation panel."""
    if not docs:
        return """
        <div style="text-align: center; color: #94A3B8; padding: 40px 10px;">
            <div style="font-size: 32px; margin-bottom: 8px;">📂</div>
            <div style="font-weight: 600; font-size: 14px;">Henüz Kaynak Çağrılmadı</div>
            <div style="font-size: 12px; margin-top: 4px;">Sohbet başlattığınızda RAG tarafından başvurulan referans dokümanlar burada listelenecektir.</div>
        </div>
        """

    html = '<div style="font-size: 13px; font-weight: 700; color: #1E293B; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;">'
    html += f'<span>📑 İlgili Referans Dokümanlar ({len(docs)})</span>'
    html += '<span style="font-size: 11px; font-weight: 600; color: #10B981; background: #ECFDF5; padding: 2px 8px; border-radius: 10px;">Doğrulandı</span></div>'

    for idx, doc in enumerate(docs, 1):
        metadata = getattr(doc, "metadata", {}) or {}
        raw_source = metadata.get("source", "")
        filename = metadata.get("file_name") or os.path.basename(raw_source) or f"Doküman #{idx}"
        doc_type = metadata.get("doc_type") or "Genel"

        # Determine badge class
        cat_lower = str(doc_type).lower()
        if "product" in cat_lower:
            badge_class = "badge-products"
            icon = "🛡️"
        elif "contract" in cat_lower:
            badge_class = "badge-contracts"
            icon = "📜"
        elif "employee" in cat_lower:
            badge_class = "badge-employees"
            icon = "👥"
        elif "company" in cat_lower:
            badge_class = "badge-company"
            icon = "🏢"
        else:
            badge_class = "badge-general"
            icon = "📄"

        content = getattr(doc, "page_content", "")
        # Clean extra whitespace
        content_preview = content.strip().replace("<", "&lt;").replace(">", "&gt;")

        html += f"""
        <div class="source-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="source-badge {badge_class}">{icon} {doc_type}</span>
                <span style="font-size: 11px; color: #94A3B8; font-weight: 500;">#{idx}</span>
            </div>
            <div class="source-filename" title="{raw_source}">{filename}</div>
            <div class="source-content">{content_preview}</div>
        </div>
        """

    return html


# ---------------------------------------------------------------------------
# Core Chat Controller (Streaming)
# ---------------------------------------------------------------------------
def chat_stream_handler(
    user_message: str,
    chat_history: List[Dict[str, str]],
    top_k: int,
    temperature: float,
    model_name: str,
    custom_system_prompt: str,
) -> Generator[Tuple[List[Dict[str, str]], str, str], None, None]:
    """
    Handle chat interaction with token-by-token streaming and dynamic citations.
    Yields: (updated_history, empty_input_box, sources_html)
    """
    if not user_message or not user_message.strip():
        yield chat_history, "", render_sources_html([])
        return

    # 1. Append user message to history
    updated_history = list(chat_history)
    updated_history.append({"role": "user", "content": user_message})

    # 2. Append empty assistant message for streaming
    updated_history.append({"role": "assistant", "content": "..."})
    yield updated_history, "", """
    <div style="text-align: center; color: #64748B; padding: 25px 10px;">
        <div style="font-size: 24px; animation: spin 1s linear infinite;">⏳</div>
        <div style="font-weight: 600; font-size: 13px; margin-top: 6px;">Vektör Tabanında İlgili Bilgiler Aranıyor...</div>
    </div>
    """

    # 3. Stream from answer.py
    try:
        prompt_template = custom_system_prompt.strip() if custom_system_prompt else SYSTEM_PROMPT
        last_docs = []

        for partial_text, docs in stream_answer_question(
            question=user_message,
            history=updated_history[:-2],  # Exclude current query and placeholder
            k=int(top_k),
            temperature=float(temperature),
            model_name=model_name,
            system_prompt_template=prompt_template,
        ):
            last_docs = docs
            updated_history[-1]["content"] = partial_text
            # Yield partial answer and updated sources
            yield updated_history, "", render_sources_html(docs)

    except Exception as e:
        error_msg = f"⚠️ **Sorgu işlenirken bir hata oluştu:** `{str(e)}`\n\n*Lütfen `.env` dosyanızdaki GROQ_API_KEY anahtarını ve internet bağlantınızı kontrol ediniz.*"
        updated_history[-1]["content"] = error_msg
        yield updated_history, "", render_sources_html([])


def handle_clear_chat() -> Tuple[List[Dict[str, str]], str, str]:
    """Clear chat history and reset sources."""
    return [], "", render_sources_html([])


# ---------------------------------------------------------------------------
# Knowledge Base Search & Ingestion Handlers
# ---------------------------------------------------------------------------
def semantic_search_handler(query: str, k: int) -> str:
    """Perform raw vector search without LLM generation and display formatted results."""
    if not query or not query.strip():
        return "<p style='color: #64748B;'>Arama yapmak için lütfen bir anahtar kelime veya soru giriniz.</p>"

    docs = fetch_context(query.strip(), k=int(k))
    if not docs:
        return "<p style='color: #EF4444;'>Eşleşen herhangi bir doküman bulunamadı.</p>"

    return render_sources_html(docs)


def trigger_reingestion_handler() -> str:
    """Re-run embedding pipeline and return status."""
    try:
        stats = run_ingestion()
        return (
            f"✅ **Yeniden Dizinleme Başarılı!**\n\n"
            f"- **Yüklenen Doküman Sayısı:** {stats.get('documents')}\n"
            f"- **Oluşturulan Parça (Chunk):** {stats.get('chunks')}\n"
            f"- **Vektör Veritabanı Kaydı:** {stats.get('vectors'):,} adet\n"
            f"- **Vektör Boyutu (Dimension):** {stats.get('dimensions')}d\n\n"
            f"*Veritabanı en güncel durumdadır.*"
        )
    except Exception as e:
        return f"❌ **Dizinleme Hatası:** {str(e)}"


# ---------------------------------------------------------------------------
# Gradio UI Construction
# ---------------------------------------------------------------------------
def create_app() -> gr.Blocks:
    kb_stats = get_kb_stats()

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Plus Jakarta Sans"), gr.themes.GoogleFont("Inter"), "sans-serif"],
    )

    with gr.Blocks(title="Insurellm | Enterprise InsurTech RAG Platform") as demo:
        # ==========================================
        # HEADER APP BAR
        # ==========================================
        header_html = f"""
        <div class="header-wrapper">
            <div class="header-brand">
                <div class="header-logo-icon">🛡️</div>
                <div class="header-title-box">
                    <h1>INSURELLM ENTERPRISE</h1>
                    <p>Yapay Zeka Destekli Kurumsal Sigortacılık Bilgi ve Karar Destek Sistemi</p>
                </div>
            </div>
            <div class="header-badges">
                <div class="status-badge"><span class="status-dot-active"></span> Sistem: Aktif</div>
                <div class="status-badge">⚡ Model: Qwen 3.8-27B (Groq)</div>
                <div class="status-badge">🗄️ ChromaDB: {kb_stats.get('total', 76)} Döküman</div>
                <div class="status-badge">🧠 Embeddings: all-MiniLM-L6-v2</div>
            </div>
        </div>
        """
        gr.HTML(header_html)

        # ==========================================
        # MAIN NAVIGATION TABS
        # ==========================================
        with gr.Tabs():
            # ------------------------------------------------------------------
            # TAB 1: RAG CHAT ASSISTANT
            # ------------------------------------------------------------------
            with gr.Tab("💬 Kurumsal AI Danışmanı", id="tab_chat"):
                with gr.Row():
                    # Left Column: Chat Area (70%)
                    with gr.Column(scale=7):
                        chatbot = gr.Chatbot(
                            height=520,
                            layout="bubble",
                            placeholder="### 🛡️ Insurellm Enterprise AI Asistanına Hoş Geldiniz\n\nŞirket poliçeleri, sözleşmeler, ürünler (Homellm, Bizllm vb.) veya çalışanlar hakkında her şeyi sorabilirsiniz. Aşağıdaki hızlı soru butonlarını kullanarak da hemen başlayabilirsiniz.",
                        )

                        # Quick Question Chips
                        gr.HTML('<div style="font-size: 12px; font-weight: 700; color: #64748B; margin: 6px 0 4px 0;">⚡ HIZLI KURUMSAL SORULAR:</div>')
                        with gr.Row(elem_classes=["chips-container"]):
                            chip1 = gr.Button("🛡️ Homellm Konut Poliçesi Kapsamı", elem_classes=["chip-btn"], size="sm")
                            chip2 = gr.Button("👤 CEO Avery Lancaster Kimdir?", elem_classes=["chip-btn"], size="sm")
                            chip3 = gr.Button("📜 Apex Reinsurance Anlaşma Detayları", elem_classes=["chip-btn"], size="sm")
                            chip4 = gr.Button("👥 Uzaktan Çalışma & İK Politikası", elem_classes=["chip-btn"], size="sm")
                            chip5 = gr.Button("🚗 Carllm Kasko Teminat Limitleri", elem_classes=["chip-btn"], size="sm")

                        # Chat Input Row
                        with gr.Row():
                            user_input = gr.Textbox(
                                show_label=False,
                                placeholder="Insurellm ürünleri, poliçeler, çalışanlar veya sözleşmeler hakkında sorunuzu buraya yazın... (Enter)",
                                container=False,
                                scale=8,
                                lines=2,
                                max_lines=4,
                            )
                            send_btn = gr.Button("Gönder", variant="primary", scale=1)

                        # Secondary Action Buttons
                        with gr.Row():
                            clear_btn = gr.Button("🗑️ Sohbeti Temizle", size="sm", variant="secondary")
                            stop_btn = gr.Button("🛑 Yanıtı Durdur", size="sm", variant="stop")

                    # Right Column: Source Citations & Audit Panel (30%)
                    with gr.Column(scale=3):
                        gr.HTML('<div style="font-size: 14px; font-weight: 700; color: #0F172A; margin-bottom: 8px;">🔍 RAG DOĞRULAMA & REFERANSLAR</div>')
                        sources_display = gr.HTML(
                            render_sources_html([]),
                            elem_classes=["source-panel-box"],
                        )

                # Advanced Settings Accordion
                with gr.Accordion("⚙️ Model ve Getirme (Retrieval) Parametreleri", open=False):
                    with gr.Row():
                        param_top_k = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=DEFAULT_RETRIEVAL_K,
                            step=1,
                            label="Alakalı Doküman Sayısı (Top-K Retrieval)",
                            info="Modelin bağlamına eklenecek en yakın parça sayısı.",
                        )
                        param_temp = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.05,
                            label="Model Sıcaklığı (Temperature)",
                            info="0.0 değeri kurumsal ve en doğru bilgi yanıtları için önerilir.",
                        )
                        param_model = gr.Dropdown(
                            choices=[
                                "qwen/qwen3.8-27b",
                                "llama-3.3-70b-versatile",
                                "llama-3.1-8b-instant",
                            ],
                            value=DEFAULT_MODEL,
                            label="Groq LLM Modeli",
                        )

                    param_system_prompt = gr.Textbox(
                        label="Özel Sistem Promptu Şablonu",
                        value=SYSTEM_PROMPT,
                        lines=5,
                        info="{context} alanı dinamik olarak getirilen dokümanlarla değiştirilecektir.",
                    )

            # ------------------------------------------------------------------
            # TAB 2: KNOWLEDGE BASE EXPLORER & INGESTION
            # ------------------------------------------------------------------
            with gr.Tab("📊 Bilgi Tabanı ve Vektör Gezgini", id="tab_kb"):
                # Top Stat Cards
                stats_html = f"""
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px;">
                    <div class="stat-card">
                        <div style="font-size: 24px;">🏢</div>
                        <div class="stat-number">{kb_stats.get('company', 4)}</div>
                        <div class="stat-label">Kurumsal & Kültür</div>
                    </div>
                    <div class="stat-card">
                        <div style="font-size: 24px;">🛡️</div>
                        <div class="stat-number">{kb_stats.get('products', 8)}</div>
                        <div class="stat-label">Sigorta Ürünleri</div>
                    </div>
                    <div class="stat-card">
                        <div style="font-size: 24px;">📜</div>
                        <div class="stat-number">{kb_stats.get('contracts', 32)}</div>
                        <div class="stat-label">Kurumsal Sözleşmeler</div>
                    </div>
                    <div class="stat-card">
                        <div style="font-size: 24px;">👥</div>
                        <div class="stat-number">{kb_stats.get('employees', 32)}</div>
                        <div class="stat-label">Çalışan & İK Kayıtları</div>
                    </div>
                </div>
                """
                gr.HTML(stats_html)

                with gr.Row():
                    with gr.Column(scale=6):
                        gr.Markdown("### 🔍 Canlı Semantik Doküman Arama (Vector Search Inspector)")
                        gr.Markdown("LLM yanıtı üretmeden doğrudan ChromaDB vektör uzayında anlamsal benzerlik araması yapın.")

                        with gr.Row():
                            kb_search_input = gr.Textbox(
                                show_label=False,
                                placeholder="Örn: reasürans teminat oranı, CEO maaşı, konut su hasarı muafiyeti...",
                                scale=8,
                            )
                            kb_search_btn = gr.Button("Semantik Ara", variant="primary", scale=2)

                        kb_k_slider = gr.Slider(minimum=1, maximum=10, value=4, step=1, label="Getirilecek Doküman Sayısı")
                        kb_search_results = gr.HTML(
                            "<p style='color: #64748B;'>Arama sonuçları burada kartlar halinde listelenecektir.</p>",
                            elem_classes=["source-panel-box"],
                        )

                    with gr.Column(scale=4):
                        gr.Markdown("### 🔄 Dizinleme & Senkronizasyon (Re-Ingestion)")
                        gr.Markdown(
                            "Yerel `knowledge-base/` dizinindeki Markdown dosyaları `RecursiveCharacterTextSplitter` "
                            "ile 500 karakterlik parçalara ayrılarak ChromaDB veritabanında güncellenir."
                        )

                        reindex_btn = gr.Button("⚡ Bilgi Tabanını Yeniden İndeksle", variant="secondary", size="lg")
                        reindex_status = gr.Markdown(
                            "*Dizinlemeyi başlatmak için yukarıdaki butona tıklayın.*",
                        )

            # ------------------------------------------------------------------
            # TAB 3: SYSTEM ARCHITECTURE & ENTERPRISE COMPLIANCE
            # ------------------------------------------------------------------
            with gr.Tab("🏛️ Sistem Mimarisi & Standartlar", id="tab_arch"):
                arch_html = """
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px;">
                    <div>
                        <h3 style="color: #0F172A; font-weight: 700; margin-bottom: 12px;">📐 RAG Pipeline Mimarisi</h3>
                        <div class="arch-card">
                            <strong style="color: #2563EB;">1. Bilgi Entegrasyonu (Document Ingestion)</strong>
                            <p style="margin: 4px 0 0 0; font-size: 13px; color: #475569;">
                                <code>knowledge-base/</code> altındaki poliçeler, ürün rehberleri, sözleşmeler ve İK dokümanları UTF-8 olarak yüklenir.
                            </p>
                        </div>
                        <div class="arch-card">
                            <strong style="color: #2563EB;">2. Parçalama (Chunking Engine)</strong>
                            <p style="margin: 4px 0 0 0; font-size: 13px; color: #475569;">
                                <code>RecursiveCharacterTextSplitter</code> (chunk_size=500, chunk_overlap=200) ile anlamsal bütünlük korunarak parçalanır.
                            </p>
                        </div>
                        <div class="arch-card">
                            <strong style="color: #2563EB;">3. Vektör Temsili (Dense Embeddings)</strong>
                            <p style="margin: 4px 0 0 0; font-size: 13px; color: #475569;">
                                Hugging Face <code>all-MiniLM-L6-v2</code> açık kaynak embedding modeli ile 384 boyutlu vektörler üretilir.
                            </p>
                        </div>
                        <div class="arch-card">
                            <strong style="color: #2563EB;">4. Vektör Belleği (Chroma Vector Database)</strong>
                            <p style="margin: 4px 0 0 0; font-size: 13px; color: #475569;">
                                Yüksek performanslı ve yerel ChromaDB ile kosinüs benzerliği üzerinden milisaniyelik semantik arama yapılır.
                            </p>
                        </div>
                        <div class="arch-card">
                            <strong style="color: #2563EB;">5. Kurumsal LLM Çıkarımı (Groq LPU Acceleration)</strong>
                            <p style="margin: 4px 0 0 0; font-size: 13px; color: #475569;">
                                Groq LPU altyapısında koşan <code>qwen/qwen3.8-27b</code> modeli, yalnızca getirilen bağlama (grounded context) sadık kalarak yanıt üretir.
                            </p>
                        </div>
                    </div>

                    <div>
                        <h3 style="color: #0F172A; font-weight: 700; margin-bottom: 12px;">🛡️ Kurumsal Güvenilirlik & Halüsinasyon Önleme</h3>
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
                            <h4 style="margin: 0 0 8px 0; color: #1E3A8A; font-size: 15px;">🔒 Sıfır Halüsinasyon Politikası</h4>
                            <p style="font-size: 13px; color: #475569; line-height: 1.6; margin: 0;">
                                Sistem, bilgi tabanında karşılığı bulunmayan veya doğrulanmamış sorulara spekülatif cevap vermez. Model promptunda zorunlu tutulan kurumsal kural uyarınca, yeterli veri olmadığında net şekilde bilgisinin olmadığını beyan eder.
                            </p>
                        </div>

                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
                            <h4 style="margin: 0 0 8px 0; color: #1E3A8A; font-size: 15px;">📑 Kaynak Şeffaflığı & Denetlenebilirlik</h4>
                            <p style="font-size: 13px; color: #475569; line-height: 1.6; margin: 0;">
                                Üretilen her yanıtın sağ tarafındaki <strong>Referanslar</strong> paneli, ilgili bilginin hangi dökümandan ve hangi metin parçasından alındığını doğrudan gösterir.
                            </p>
                        </div>

                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px;">
                            <h4 style="margin: 0 0 8px 0; color: #1E3A8A; font-size: 15px;">💼 Kurumsal Portföy</h4>
                            <p style="font-size: 13px; color: #475569; line-height: 1.6; margin: 0;">
                                <strong>Desteklenen Ürün Ailesi:</strong> Homellm (Konut), Bizllm (Ticari), Carllm (Kasko & Trafik), Healthllm (Sağlık), Claimllm (Hasar Yönetimi), Cyberllm (Siber Risk), Travelllm (Seyahat).
                            </p>
                        </div>
                    </div>
                </div>
                """
                gr.HTML(arch_html)

        # ==========================================
        # EVENT LISTENERS & WIRING
        # ==========================================
        # Chat stream event
        chat_stream_args = [
            user_input,
            chatbot,
            param_top_k,
            param_temp,
            param_model,
            param_system_prompt,
        ]

        chat_outputs = [chatbot, user_input, sources_display]

        send_event = send_btn.click(
            fn=chat_stream_handler,
            inputs=chat_stream_args,
            outputs=chat_outputs,
        )

        submit_event = user_input.submit(
            fn=chat_stream_handler,
            inputs=chat_stream_args,
            outputs=chat_outputs,
        )

        # Stop button
        stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[send_event, submit_event])

        # Clear button
        clear_btn.click(
            fn=handle_clear_chat,
            inputs=None,
            outputs=[chatbot, user_input, sources_display],
        )

        # Quick question chip listeners
        def set_chip_and_send(chip_text: str):
            clean_question = chip_text.split(" ", 1)[-1]  # Remove emoji
            return clean_question

        chip1.click(lambda: "Homellm konut sigortası neleri kapsar ve poliçe detayları nelerdir?", None, user_input)
        chip2.click(lambda: "Insurellm CEO'su Avery Lancaster kimdir, eğitim ve kariyer geçmişi nedir?", None, user_input)
        chip3.click(lambda: "Apex Reinsurance ile yapılan reasürans anlaşmasının şartları nelerdir?", None, user_input)
        chip4.click(lambda: "Insurellm şirketinin çalışma modeli, uzaktan çalışma ve İK politikası nasıldır?", None, user_input)
        chip5.click(lambda: "Carllm kasko sigortası teminat limitleri ve şartları nelerdir?", None, user_input)

        # Tab 2: Semantic Search & Re-Ingest listeners
        kb_search_btn.click(
            fn=semantic_search_handler,
            inputs=[kb_search_input, kb_k_slider],
            outputs=[kb_search_results],
        )
        kb_search_input.submit(
            fn=semantic_search_handler,
            inputs=[kb_search_input, kb_k_slider],
            outputs=[kb_search_results],
        )

        reindex_btn.click(
            fn=trigger_reingestion_handler,
            inputs=None,
            outputs=[reindex_status],
        )

    return demo


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------
app = create_app()

if __name__ == "__main__":
    enterprise_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Plus Jakarta Sans"), gr.themes.GoogleFont("Inter"), "sans-serif"],
    )
    app.queue().launch(
        theme=enterprise_theme,
        css=CUSTOM_CSS,
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )

