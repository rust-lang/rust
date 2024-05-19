use super::{EMPTY_LINE_AFTER_DOC_COMMENTS, EMPTY_LINE_AFTER_OUTER_ATTR};
use clippy_utils::diagnostics::span_lint;
use clippy_utils::source::{is_present_in_source, snippet_opt, without_block_comments};
use rustc_ast::{AttrKind, AttrStyle};
use rustc_lint::EarlyContext;
use rustc_span::Span;

/// Check for empty lines after outer attributes.
///
/// Attributes and documentation comments are both considered outer attributes
/// by the AST. However, the average user likely considers them to be different.
/// Checking for empty lines after each of these attributes is split into two different
/// lints but can share the same logic.
pub(super) fn check(cx: &EarlyContext<'_>, item: &rustc_ast::Item) {
    let mut iter = item.attrs.iter().peekable();
    while let Some(attr) = iter.next() {
        if (matches!(attr.kind, AttrKind::Normal(..)) || matches!(attr.kind, AttrKind::DocComment(..)))
            && attr.style == AttrStyle::Outer
            && is_present_in_source(cx, attr.span)
        {
            let begin_of_attr_to_item = Span::new(attr.span.lo(), item.span.lo(), item.span.ctxt(), item.span.parent());
            let end_of_attr_to_next_attr_or_item = Span::new(
                attr.span.hi(),
                iter.peek().map_or(item.span.lo(), |next_attr| next_attr.span.lo()),
                item.span.ctxt(),
                item.span.parent(),
            );

            if let Some(snippet) = snippet_opt(cx, end_of_attr_to_next_attr_or_item) {
                let lines = snippet.split('\n').collect::<Vec<_>>();
                let lines = without_block_comments(lines);

                if lines.iter().filter(|l| l.trim().is_empty()).count() > 2 {
                    let (lint_msg, lint_type) = match attr.kind {
                        AttrKind::DocComment(..) => (
                            "found an empty line after a doc comment. \
                            Perhaps you need to use `//!` to make a comment on a module, remove the empty line, or make a regular comment with `//`?",
                            EMPTY_LINE_AFTER_DOC_COMMENTS,
                        ),
                        AttrKind::Normal(..) => (
                            "found an empty line after an outer attribute. \
                            Perhaps you forgot to add a `!` to make it an inner attribute?",
                            EMPTY_LINE_AFTER_OUTER_ATTR,
                        ),
                    };

                    span_lint(cx, lint_type, begin_of_attr_to_item, lint_msg);
                }
            }
        }
    }
}
