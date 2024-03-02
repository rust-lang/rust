use super::{Attribute, ALLOW_ATTRIBUTES_WITHOUT_REASON};
use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::is_from_proc_macro;
use rustc_ast::{MetaItemKind, NestedMetaItem};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_span::sym;
use rustc_span::symbol::Symbol;

pub(super) fn check<'cx>(cx: &LateContext<'cx>, name: Symbol, items: &[NestedMetaItem], attr: &'cx Attribute) {
    // Check for the feature
    if !cx.tcx.features().lint_reasons {
        return;
    }

    // Check if the reason is present
    if let Some(item) = items.last().and_then(NestedMetaItem::meta_item)
        && let MetaItemKind::NameValue(_) = &item.kind
        && item.path == sym::reason
    {
        return;
    }

    // Check if the attribute is in an external macro and therefore out of the developer's control
    if in_external_macro(cx.sess(), attr.span) || is_from_proc_macro(cx, &attr) {
        return;
    }

    span_lint_and_help(
        cx,
        ALLOW_ATTRIBUTES_WITHOUT_REASON,
        attr.span,
        &format!("`{}` attribute without specifying a reason", name.as_str()),
        None,
        "try adding a reason at the end with `, reason = \"..\"`",
    );
}
