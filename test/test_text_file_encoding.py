"""User-supplied text files must be read as UTF-8, not with the locale encoding.

On Windows the default encoding of ``open()`` is the ANSI code page (usually cp1252), so
a UTF-8 file with non-ASCII text was silently turned into mojibake. These tests force
that default on every OS, so they also fail on Linux if a call site loses ``encoding``.
"""

import builtins
import csv
import io
import json

import pytest

import gradio as gr
from gradio.components.label import LabelData
from gradio.external import _text_encoded_file_as_prompt
from gradio.image_utils import extract_svg_content

TEXT = "日本語 café ✓"


@pytest.fixture(autouse=True)
def windows_default_encoding(monkeypatch):
    """Make text-mode ``open()`` without ``encoding`` behave like Windows (cp1252)."""
    real_open = io.open

    def open_with_cp1252_default(
        file, mode="r", buffering=-1, encoding=None, *args, **kwargs
    ):
        # pathlib passes "locale" (or "utf-8" in UTF-8 mode) when no encoding is given
        if encoding in (None, "locale") and "b" not in mode:
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_with_cp1252_default)
    monkeypatch.setattr(io, "open", open_with_cp1252_default)  # used by pathlib
    monkeypatch.setattr(
        io, "text_encoding", lambda encoding, stacklevel=2: encoding or "locale"
    )


def test_fixture_reproduces_windows_default(tmp_path):
    path = tmp_path / "note.txt"
    path.write_bytes(TEXT.encode("utf-8"))
    assert path.read_text() != TEXT
    assert path.read_text(encoding="utf-8") == TEXT


def test_extract_svg_content_reads_utf8(tmp_path):
    svg = tmp_path / "icon.svg"
    markup = f'<svg xmlns="http://www.w3.org/2000/svg"><text>{TEXT}</text></svg>'
    svg.write_bytes(markup.encode("utf-8"))
    assert extract_svg_content(svg) == markup


def test_label_postprocess_reads_utf8_json_file(tmp_path):
    path = tmp_path / "label.json"
    data = {"label": TEXT, "confidences": [{"label": TEXT, "confidence": 1.0}]}
    path.write_bytes(json.dumps(data, ensure_ascii=False).encode("utf-8"))
    output = gr.Label().postprocess(str(path))
    assert isinstance(output, LabelData)
    assert output.label == TEXT
    assert output.confidences is not None
    assert output.confidences[0].label == TEXT


def test_examples_from_directory_reads_utf8_log_csv(tmp_path):
    with open(tmp_path / "log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "timestamp"])
        writer.writerow([TEXT, "2026-01-01"])
    with gr.Blocks():
        textbox = gr.Textbox()
        examples = gr.Examples(examples=str(tmp_path), inputs=[textbox])
    assert examples.examples == [[TEXT]]


def test_chat_text_file_attachment_reads_utf8(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(TEXT.encode("utf-8"))
    assert _text_encoded_file_as_prompt(str(path)) == f"\n## notes.txt\n{TEXT}"


def test_load_openapi_reads_utf8_spec_file(tmp_path):
    spec = {
        "openapi": "3.0.0",
        "info": {"title": TEXT},
        "paths": {"/items": {"get": {"responses": {"200": {"description": "ok"}}}}},
    }
    path = tmp_path / "openapi.json"
    path.write_bytes(json.dumps(spec, ensure_ascii=False).encode("utf-8"))
    demo = gr.load_openapi(str(path), base_url="http://localhost:1")
    assert demo.title == TEXT
