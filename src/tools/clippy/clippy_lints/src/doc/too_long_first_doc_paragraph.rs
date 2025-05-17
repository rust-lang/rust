use rustc_attr_data_structures::AttributeKind;
use rustc_errors::Applicability;
use rustc_hir::{Attribute, Item, ItemKind};
use rustc_lint::LateContext;

use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_from_proc_macro;
use clippy_utils::source::snippet_opt;

use super::TOO_LONG_FIRST_DOC_PARAGRAPH;

pub(super) fn check(
    cx: &LateContext<'_>,
    item: &Item<'_>,
    attrs: &[Attribute],
    mut first_paragraph_len: usize,
    check_private_items: bool,
) {
    if !check_private_items && !cx.effective_visibilities.is_exported(item.owner_id.def_id) {
        return;
    }
    if first_paragraph_len <= 200
        || !matches!(
            item.kind,
            // This is the list of items which can be documented AND are displayed on the module
            // page. So associated items or impl blocks are not part of this list.
            ItemKind::Static(..)
                | ItemKind::Const(..)
                | ItemKind::Fn { .. }
                | ItemKind::Macro(..)
                | ItemKind::Mod(..)
                | ItemKind::TyAlias(..)
                | ItemKind::Enum(..)
                | ItemKind::Struct(..)
                | ItemKind::Union(..)
                | ItemKind::Trait(..)
                | ItemKind::TraitAlias(..)
        )
    {
        return;
    }

    let mut spans = Vec::new();
    let mut should_suggest_empty_doc = false;

    for attr in attrs {
        if let Attribute::Parsed(AttributeKind::DocComment { span, comment, .. }) = attr {
            spans.push(span);
            let doc = comment.as_str();
            let doc = doc.trim();
            if spans.len() == 1 {
                // We make this suggestion only if the first doc line ends with a punctuation
                // because it might just need to add an empty line with `///`.
                should_suggest_empty_doc = doc.ends_with('.') || doc.ends_with('!') || doc.ends_with('?');
            } else if spans.len() == 2 {
                // We make this suggestion only if the second doc line is not empty.
                should_suggest_empty_doc &= !doc.is_empty();
            }

            let len = doc.chars().count();
            if len >= first_paragraph_len {
                break;
            }
            first_paragraph_len -= len;
        }
    }

    let &[first_span, .., last_span] = spans.as_slice() else {
        return;
    };
    if is_from_proc_macro(cx, item) {
        return;
    }

    span_lint_and_then(
        cx,
        TOO_LONG_FIRST_DOC_PARAGRAPH,
        first_span.with_hi(last_span.lo()),
        "first doc comment paragraph is too long",
        |diag| {
            if should_suggest_empty_doc
                && let Some(second_span) = spans.get(1)
                && let new_span = first_span.with_hi(second_span.lo()).with_lo(first_span.hi())
                && let Some(snippet) = snippet_opt(cx, new_span)
            {
                let Some(first) = snippet_opt(cx, *first_span) else {
                    return;
                };
                let Some(comment_form) = first.get(..3) else {
                    return;
                };

                diag.span_suggestion(
                    new_span,
                    "add an empty line",
                    format!("{snippet}{comment_form}{snippet}"),
                    Applicability::MachineApplicable,
                );
            }
        },
    );
}
