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

use std::collections::{BTreeMap, BTreeSet};

use rustc_hir::HirId;
use rustc_lint_defs::Applicability;
use rustc_resolve::rustdoc::source_span_for_markdown_range;
use {pulldown_cmark as cmarkn, pulldown_cmark_old as cmarko};

use crate::clean::Item;
use crate::core::DocContext;

pub(crate) fn visit_item(cx: &DocContext<'_>, item: &Item, hir_id: HirId, dox: &str) {
    let tcx = cx.tcx;

    // P1: unintended strikethrough was fixed by requiring single-tildes to flank
    // the same way underscores do, so nothing is done here

    // P2: block quotes without following space parsed wrong
    //
    // This is the set of starting points for block quotes with no space after
    // the `>`. It is populated by the new parser, and if the old parser fails to
    // clear it out, it'll produce a warning.
    let mut spaceless_block_quotes = BTreeSet::new();

    // P3: missing footnote references
    //
    // This is populated by listening for FootnoteReference from
    // the new parser and old parser.
    let mut missing_footnote_references = BTreeMap::new();
    let mut found_footnote_references = BTreeSet::new();

    // populate problem cases from new parser
    {
        pub fn main_body_opts_new() -> cmarkn::Options {
            cmarkn::Options::ENABLE_TABLES
                | cmarkn::Options::ENABLE_FOOTNOTES
                | cmarkn::Options::ENABLE_STRIKETHROUGH
                | cmarkn::Options::ENABLE_TASKLISTS
                | cmarkn::Options::ENABLE_SMART_PUNCTUATION
        }
        let parser_new = cmarkn::Parser::new_ext(dox, main_body_opts_new()).into_offset_iter();
        for (event, span) in parser_new {
            if let cmarkn::Event::Start(cmarkn::Tag::BlockQuote(_)) = event {
                if !dox[span.clone()].starts_with("> ") {
                    spaceless_block_quotes.insert(span.start);
                }
            }
            if let cmarkn::Event::FootnoteReference(_) = event {
                found_footnote_references.insert(span.start + 1);
            }
        }
    }

    // remove cases where they don't actually differ
    {
        pub fn main_body_opts_old() -> cmarko::Options {
            cmarko::Options::ENABLE_TABLES
                | cmarko::Options::ENABLE_FOOTNOTES
                | cmarko::Options::ENABLE_STRIKETHROUGH
                | cmarko::Options::ENABLE_TASKLISTS
                | cmarko::Options::ENABLE_SMART_PUNCTUATION
        }
        let parser_old = cmarko::Parser::new_ext(dox, main_body_opts_old()).into_offset_iter();
        for (event, span) in parser_old {
            if let cmarko::Event::Start(cmarko::Tag::BlockQuote) = event
                && !dox[span.clone()].starts_with("> ")
            {
                spaceless_block_quotes.remove(&span.start);
            }
            if let cmarko::Event::FootnoteReference(_) = event
                && !found_footnote_references.contains(&(span.start + 1))
            {
                missing_footnote_references.insert(span.start + 1, span);
            }
        }
    }

    for start in spaceless_block_quotes {
        let (span, precise) =
            source_span_for_markdown_range(tcx, dox, &(start..start + 1), &item.attrs.doc_strings)
                .map(|span| (span, true))
                .unwrap_or_else(|| (item.attr_span(tcx), false));

        tcx.node_span_lint(crate::lint::UNPORTABLE_MARKDOWN, hir_id, span, |lint| {
            lint.primary_message("unportable markdown");
            lint.help("confusing block quote with no space after the `>` marker".to_string());
            if precise {
                lint.span_suggestion(
                    span.shrink_to_hi(),
                    "if the quote is intended, add a space",
                    " ",
                    Applicability::MaybeIncorrect,
                );
                lint.span_suggestion(
                    span.shrink_to_lo(),
                    "if it should not be a quote, escape it",
                    "\\",
                    Applicability::MaybeIncorrect,
                );
            }
        });
    }
    for (_caret, span) in missing_footnote_references {
        let (ref_span, precise) =
            source_span_for_markdown_range(tcx, dox, &span, &item.attrs.doc_strings)
                .map(|span| (span, true))
                .unwrap_or_else(|| (item.attr_span(tcx), false));

        tcx.node_span_lint(crate::lint::UNPORTABLE_MARKDOWN, hir_id, ref_span, |lint| {
            lint.primary_message("unportable markdown");
            if precise {
                lint.span_suggestion(
                    ref_span.shrink_to_lo(),
                    "if it should not be a footnote, escape it",
                    "\\",
                    Applicability::MaybeIncorrect,
                );
            }
            if dox.as_bytes().get(span.end) == Some(&b'[') {
                lint.help("confusing footnote reference and link");
                if precise {
                    lint.span_suggestion(
                        ref_span.shrink_to_hi(),
                        "if the footnote is intended, add a space",
                        " ",
                        Applicability::MaybeIncorrect,
                    );
                } else {
                    lint.help("there should be a space between the link and the footnote");
                }
            }
        });
    }
}
