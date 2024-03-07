use super::{Attribute, MAYBE_MISUSED_CFG};
use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_ast::{MetaItemKind, NestedMetaItem};
use rustc_errors::Applicability;
use rustc_lint::EarlyContext;
use rustc_span::sym;

pub(super) fn check(cx: &EarlyContext<'_>, attr: &Attribute) {
    if attr.has_name(sym::cfg)
        && let Some(items) = attr.meta_item_list()
    {
        check_nested_misused_cfg(cx, &items);
    }
}

fn check_nested_misused_cfg(cx: &EarlyContext<'_>, items: &[NestedMetaItem]) {
    for item in items {
        if let NestedMetaItem::MetaItem(meta) = item {
            if let Some(ident) = meta.ident()
                && ident.name.as_str() == "features"
                && let Some(val) = meta.value_str()
            {
                span_lint_and_sugg(
                    cx,
                    MAYBE_MISUSED_CFG,
                    meta.span,
                    "'feature' may be misspelled as 'features'",
                    "did you mean",
                    format!("feature = \"{val}\""),
                    Applicability::MaybeIncorrect,
                );
            }
            if let MetaItemKind::List(list) = &meta.kind {
                check_nested_misused_cfg(cx, list);
            // If this is not a list, then we check for `cfg(test)`.
            } else if let Some(ident) = meta.ident()
                && matches!(ident.name.as_str(), "tests" | "Test")
            {
                span_lint_and_sugg(
                    cx,
                    MAYBE_MISUSED_CFG,
                    meta.span,
                    &format!("'test' may be misspelled as '{}'", ident.name.as_str()),
                    "did you mean",
                    "test".to_string(),
                    Applicability::MaybeIncorrect,
                );
            }
        }
    }
}
