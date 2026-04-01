use super::{Attribute, NON_MINIMAL_CFG};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::SpanRangeExt;
use rustc_ast::{MetaItemInner, MetaItemKind};
use rustc_errors::Applicability;
use rustc_lint::EarlyContext;
use rustc_span::sym;

pub(super) fn check(cx: &EarlyContext<'_>, attr: &Attribute) {
    if attr.has_name(sym::cfg)
        && let Some(items) = attr.meta_item_list()
    {
        check_nested_cfg(cx, &items);
    }
}

fn check_nested_cfg(cx: &EarlyContext<'_>, items: &[MetaItemInner]) {
    for item in items {
        if let MetaItemInner::MetaItem(meta) = item {
            if !meta.has_name(sym::any) && !meta.has_name(sym::all) {
                continue;
            }
            if let MetaItemKind::List(list) = &meta.kind {
                check_nested_cfg(cx, list);
                if list.len() == 1 {
                    span_lint_and_then(
                        cx,
                        NON_MINIMAL_CFG,
                        meta.span,
                        "unneeded sub `cfg` when there is only one condition",
                        |diag| {
                            if let Some(snippet) = list[0].span().get_source_text(cx) {
                                diag.span_suggestion(
                                    meta.span,
                                    "try",
                                    snippet.to_owned(),
                                    Applicability::MaybeIncorrect,
                                );
                            }
                        },
                    );
                } else if list.is_empty() && meta.has_name(sym::all) {
                    span_lint_and_then(
                        cx,
                        NON_MINIMAL_CFG,
                        meta.span,
                        "unneeded sub `cfg` when there is no condition",
                        |_| {},
                    );
                }
            }
        }
    }
}
