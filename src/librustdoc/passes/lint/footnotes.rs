//! Detects specific markdown syntax that's different between pulldown-cmark
//! 0.9 and 0.11.
//!
//! This is a mitigation for old parser bugs that affected some
//! real crates' docs. The old parser claimed to comply with CommonMark,
//! but it did not. These warnings will eventually be removed,
//! though some of them may become Clippy lints.
//!
//! <https://github.com/rust-lang/rust/pull/121659#issuecomment-1992752820>
//!
//! <https://rustc-dev-guide.rust-lang.org/bug-fix-procedure.html#add-the-lint-to-the-list-of-removed-lists>

use std::ops::Range;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
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
                    && let Some((Event::Text(text), _)) = parser.peek()
                    && text.trim_start().starts_with('^')
                    && parser.next().is_some()
                    && let Some((Event::Text(text), end_span)) = parser.peek()
                    && &**text == "]" =>
            {
                missing_footnote_references.insert(Range { start: span.start, end: end_span.end });
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

            tcx.node_span_lint(crate::lint::UNUSED_FOOTNOTE_DEFINITION, hir_id, span, |lint| {
                lint.primary_message("unused footnote definition");
            });
        }
    }

    #[allow(rustc::potential_query_instability)]
    for span in missing_footnote_references {
        let (ref_span, precise) =
            source_span_for_markdown_range(tcx, dox, &span, &item.attrs.doc_strings)
                .unwrap_or_else(|| (item.attr_span(tcx), false));

        if precise {
            tcx.node_span_lint(crate::lint::BROKEN_FOOTNOTE, hir_id, ref_span, |lint| {
                lint.primary_message("no footnote definition matching this footnote");
                lint.span_suggestion(
                    ref_span.shrink_to_lo(),
                    "if it should not be a footnote, escape it",
                    "\\",
                    Applicability::MaybeIncorrect,
                );
            });
        }
    }
}
