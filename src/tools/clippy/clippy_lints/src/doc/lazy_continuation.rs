use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use itertools::Itertools;
use rustc_errors::Applicability;
use rustc_lint::LateContext;
use rustc_span::{BytePos, Span};
use std::cmp::Ordering;
use std::ops::Range;

use super::{DOC_LAZY_CONTINUATION, DOC_OVERINDENTED_LIST_ITEMS};

fn map_container_to_text(c: &super::Container) -> &'static str {
    match c {
        super::Container::Blockquote => "> ",
        // numbered list can have up to nine digits, plus the dot, plus four spaces on either side
        super::Container::List(indent) => &"                  "[0..*indent],
    }
}

// TODO: Adjust the parameters as necessary
pub(super) fn check(
    cx: &LateContext<'_>,
    doc: &str,
    range: Range<usize>,
    mut span: Span,
    containers: &[super::Container],
) {
    if doc[range.clone()].contains('\t') {
        // We don't do tab stops correctly.
        return;
    }

    // Blockquote
    let ccount = doc[range.clone()].chars().filter(|c| *c == '>').count();
    let blockquote_level = containers
        .iter()
        .filter(|c| matches!(c, super::Container::Blockquote))
        .count();
    if ccount < blockquote_level {
        span_lint_and_then(
            cx,
            DOC_LAZY_CONTINUATION,
            span,
            "doc quote line without `>` marker",
            |diag| {
                let mut doc_start_range = &doc[range];
                let mut suggested = String::new();
                for c in containers {
                    let text = map_container_to_text(c);
                    if doc_start_range.starts_with(text) {
                        doc_start_range = &doc_start_range[text.len()..];
                        span = span.with_lo(
                            span.lo() + BytePos(u32::try_from(text.len()).expect("text is not 2**32 or bigger")),
                        );
                    } else if matches!(c, super::Container::Blockquote)
                        && let Some(i) = doc_start_range.find('>')
                    {
                        doc_start_range = &doc_start_range[i + 1..];
                        span = span
                            .with_lo(span.lo() + BytePos(u32::try_from(i).expect("text is not 2**32 or bigger") + 1));
                    } else {
                        suggested.push_str(text);
                    }
                }
                diag.span_suggestion_verbose(
                    span,
                    "add markers to start of line",
                    suggested,
                    Applicability::MachineApplicable,
                );
                diag.help("if this not intended to be a quote at all, escape it with `\\>`");
            },
        );
        return;
    }

    if ccount != 0 && blockquote_level != 0 {
        // If this doc is a blockquote, we don't go further.
        return;
    }

    // List
    let leading_spaces = doc[range].chars().filter(|c| *c == ' ').count();
    let list_indentation = containers
        .iter()
        .map(|c| {
            if let super::Container::List(indent) = c {
                *indent
            } else {
                0
            }
        })
        .sum();
    match leading_spaces.cmp(&list_indentation) {
        Ordering::Less => span_lint_and_then(
            cx,
            DOC_LAZY_CONTINUATION,
            span,
            "doc list item without indentation",
            |diag| {
                // simpler suggestion style for indentation
                let indent = list_indentation - leading_spaces;
                diag.span_suggestion_verbose(
                    span.shrink_to_hi(),
                    "indent this line",
                    std::iter::repeat_n(" ", indent).join(""),
                    Applicability::MaybeIncorrect,
                );
                diag.help("if this is supposed to be its own paragraph, add a blank line");
            },
        ),
        Ordering::Greater => {
            let sugg = std::iter::repeat_n(" ", list_indentation).join("");
            span_lint_and_sugg(
                cx,
                DOC_OVERINDENTED_LIST_ITEMS,
                span,
                "doc list item overindented",
                format!("try using `{sugg}` ({list_indentation} spaces)"),
                sugg,
                Applicability::MaybeIncorrect,
            );
        },
        Ordering::Equal => {},
    }
}
