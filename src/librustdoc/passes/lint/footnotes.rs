use std::ops::Range;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::DiagDecorator;
use rustc_hir::HirId;
use rustc_lint_defs::Applicability;
use rustc_resolve::rustdoc::pulldown_cmark::{Event, Options, Parser, Tag};
use rustc_resolve::rustdoc::source_span_for_markdown_range;

use crate::clean::Item;
use crate::core::DocContext;

pub(crate) fn visit_item(cx: &DocContext<'_>, item: &Item, hir_id: HirId, dox: &str) {
    let tcx = cx.tcx;

    let mut missing_footnote_references = FxHashSet::default();
    let mut footnote_references = FxHashSet::default();
    let mut footnote_definitions = FxHashMap::default();

    let options = Options::ENABLE_FOOTNOTES;
    let mut parser = Parser::new_ext(dox, options).into_offset_iter().peekable();
    while let Some((event, span)) = parser.next() {
        match event {
            Event::Text(text)
                if &*text == "["
                    && (span.start == 0 || dox.as_bytes().get(span.start - 1) != Some(&b'\\'))
                    && let Some(len) = scan_footnote_ref(&dox[span.start..]) =>
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
                    "\\",
                    Applicability::MaybeIncorrect,
                );
            }),
        );
    }
}

fn scan_footnote_ref(dox: &str) -> Option<usize> {
    let dox = dox.as_bytes();
    let mut i = 0;
    if dox.get(i) != Some(&b'[') {
        return None;
    }
    i += 1;
    if dox.get(i) != Some(&b'^') {
        return None;
    }
    i += 1;
    while let Some(&c) = dox.get(i) {
        if c == b']' {
            i += 1;
            return Some(i);
        }
        if c == b'\r' || c == b'\n' || c == b'[' {
            // Can't nest things like this.
            break;
        }
        if c == b'\\' {
            i += 1;
        }
        if dox.get(i) == Some(&b'\r') || dox.get(i) == Some(&b'\n') {
            // Can't have line breaks in footnote refs
            break;
        }
        i += 1;
    }
    None
}
