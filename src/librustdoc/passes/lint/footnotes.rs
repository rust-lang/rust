use std::ops::Range;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::DiagDecorator;
use rustc_hir::HirId;
use rustc_lint_defs::Applicability;
use rustc_resolve::rustdoc::pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};
use rustc_resolve::rustdoc::source_span_for_markdown_range;

use crate::clean::Item;
use crate::core::DocContext;

// based on
// https://github.com/pulldown-cmark/pulldown-cmark/blob/fc8fe713f58d7f4495038b48fe76c1f101fb3af1/pulldown-cmark/src/linklabel.rs#L65

fn scan_ch(ch: u8, dox: &[u8], i: &mut usize) -> Option<()> {
    if dox.get(*i) == Some(&ch) {
        *i += 1;
        Some(())
    } else {
        None
    }
}

fn scan_footnote_ref(dox: &[u8], in_table: bool) -> Option<usize> {
    let mut i = 0;
    scan_ch(b'[', dox, &mut i)?;
    scan_ch(b'^', dox, &mut i)?;
    if dox.get(i) == Some(&b']') {
        return None;
    }
    while let Some(&ch) = dox.get(i) {
        if ch == b']'
            || ch == b'['
            || ch == b'\r'
            || ch == b'\n'
            || (in_table && ch == b'|')
            // these two cause false negatives in obscure corner cases,
            // but there's another warning from the unescaped_backticks
            // and invalid_html_tags lints when they do
            || ch == b'`'
            || ch == b'<'
        {
            break;
        } else if in_table
            && ch == b'\\'
            && dox.get(i + 1) == Some(&b'\\')
            && dox.get(i + 2) == Some(&b'|')
        {
            i += 3;
        } else if ch == b'\\' && dox.get(i + 1).copied().map_or(false, is_ascii_punctuation) {
            i += 2;
        } else {
            i += 1;
        }
    }
    scan_ch(b']', dox, &mut i)?;
    Some(i)
}

fn is_ascii_punctuation(c: u8) -> bool {
    c < 128 && (PUNCT_MASKS_ASCII[(c / 16) as usize] & (1 << (c & 15))) != 0
}

const PUNCT_MASKS_ASCII: [u16; 8] = [
    0x0000, // U+0000...U+000F
    0x0000, // U+0010...U+001F
    0xfffe, // U+0020...U+002F
    0xfc00, // U+0030...U+003F
    0x0001, // U+0040...U+004F
    0xf800, // U+0050...U+005F
    0x0001, // U+0060...U+006F
    0x7800, // U+0070...U+007F
];

pub(crate) fn visit_item(cx: &DocContext<'_>, item: &Item, hir_id: HirId, dox: &str) {
    let tcx = cx.tcx;

    let mut missing_footnote_references = FxHashSet::default();
    let mut footnote_references = FxHashSet::default();
    let mut footnote_definitions = FxHashMap::default();
    let mut in_table = false;

    let options = Options::ENABLE_FOOTNOTES | Options::ENABLE_TABLES;
    let mut parser = Parser::new_ext(dox, options).into_offset_iter().peekable();
    while let Some((event, span)) = parser.next() {
        match event {
            Event::Text(text)
                if text.starts_with("[")
                    && (span.start == 0 || dox.as_bytes()[span.start - 1] != b'\\')
                    && let Some(len) =
                        scan_footnote_ref(&dox.as_bytes()[span.start..], in_table) =>
            {
                missing_footnote_references
                    .insert(Range { start: span.start, end: span.start + len });
            }
            Event::FootnoteReference(label) => {
                footnote_references.insert(label);
            }
            Event::Start(Tag::FootnoteDefinition(label)) => {
                footnote_definitions.insert(label, span.start + 1);
            }
            Event::Start(Tag::Table(_)) => in_table = true,
            Event::End(TagEnd::Table) => in_table = false,
            _ => {}
        }
    }

    #[allow(rustc::potential_query_instability)]
    for (footnote, span) in footnote_definitions {
        if !footnote_references.contains(&footnote) {
            let (span, _) = source_span_for_markdown_range(
                tcx,
                dox,
                &(span..span + 1),
                &item.attrs.doc_strings,
            )
            .unwrap_or_else(|| (item.attr_span(tcx), false));

            tcx.emit_node_span_lint(
                crate::lint::UNUSED_FOOTNOTE_DEFINITION,
                hir_id,
                span,
                DiagDecorator(|lint| {
                    lint.primary_message("unused footnote definition");
                }),
            );
        }
    }

    #[allow(rustc::potential_query_instability)]
    for span in missing_footnote_references {
        let ref_span = source_span_for_markdown_range(tcx, dox, &span, &item.attrs.doc_strings)
            .map(|(span, _)| span)
            .unwrap_or_else(|| item.attr_span(tcx));

        tcx.emit_node_span_lint(
            crate::lint::BROKEN_FOOTNOTE,
            hir_id,
            ref_span,
            DiagDecorator(|lint| {
                lint.primary_message("no footnote definition matching this footnote");
                lint.span_suggestion(
                    ref_span.shrink_to_lo(),
                    "if it should not be a footnote, escape it",
                    format!("\\{}", &dox[span]),
                    Applicability::MaybeIncorrect,
                );
            }),
        );
    }
}
