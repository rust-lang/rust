use super::{Attribute, DEPRECATED_ATTRIBUTES_WITHOUT_NOTE, DEPRECATED_ATTRIBUTES_WITHOUT_SINCE};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use rustc_ast::{MetaItemInner, MetaItemKind};
use rustc_errors::Applicability;
use rustc_lint::EarlyContext;
use rustc_span::sym;

pub(super) fn check_fields<'cx>(cx: &EarlyContext<'cx>, items: Option<&[MetaItemInner]>, attr: &'cx Attribute) {
    let mut has_note = attr.value_span().is_some();
    let mut has_since = false;

    if let Some(items) = items {
        for item_inner in items {
            if let Some(item) = MetaItemInner::meta_item(item_inner)
                && let MetaItemKind::NameValue(_) = &item.kind
            {
                if item.path == sym::note {
                    has_note = true;
                }
                if item.path == sym::since {
                    has_since = true;
                }
            }
        }
    }

    if !has_note {
        let mut applicability = Applicability::HasPlaceholders;

        let suggestion = build_suggestion(cx, items, &mut applicability, "note = /* note */)]");

        span_lint_and_sugg(
            cx,
            DEPRECATED_ATTRIBUTES_WITHOUT_NOTE,
            attr.span,
            "`deprecated` attribute without note could be confusing",
            "add a note",
            suggestion,
            applicability,
        );
    }

    if !has_since {
        let mut applicability = Applicability::HasPlaceholders;

        let suggestion = if let Some(eq_note) = attr.value_span() {
            let snippet = snippet_with_applicability(cx, eq_note, "/* note */", &mut applicability);
            format!("#[deprecated(note = {snippet}, since = /* version */)]")
        } else {
            build_suggestion(cx, items, &mut applicability, "since = /* version */)]")
        };

        span_lint_and_sugg(
            cx,
            DEPRECATED_ATTRIBUTES_WITHOUT_SINCE,
            attr.span,
            "`deprecated` attribute without specifying what version of the crate deprecated the item",
            "add a version",
            suggestion,
            applicability,
        );
    }
}

fn build_suggestion(
    cx: &EarlyContext<'_>,
    items: Option<&[MetaItemInner]>,
    applicability: &mut Applicability,
    placeholder: &str,
) -> String {
    items.map_or_else(
        || format!("#[deprecated({placeholder}"),
        |items| {
            let mut attr_with_fields = String::from("#[deprecated(");
            for item in items {
                let snippet = snippet_with_applicability(cx, item.span(), "_", applicability);
                attr_with_fields.push_str(&snippet);
                attr_with_fields.push_str(", ");
            }
            attr_with_fields.push_str(placeholder);
            attr_with_fields
        },
    )
}
