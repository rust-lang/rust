use super::{ALLOW_ATTRIBUTES_WITHOUT_REASON, Attribute};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_from_proc_macro;
use rustc_ast::{MetaItemInner, MetaItemKind};
use rustc_lint::{EarlyContext, LintContext};
use rustc_span::sym;
use rustc_span::symbol::Symbol;

pub(super) fn check<'cx>(cx: &EarlyContext<'cx>, name: Symbol, items: &[MetaItemInner], attr: &'cx Attribute) {
    // Check if the reason is present
    if let Some(item) = items.last().and_then(MetaItemInner::meta_item)
        && let MetaItemKind::NameValue(_) = &item.kind
        && item.path == sym::reason
    {
        return;
    }

    // Check if the attribute is in an external macro and therefore out of the developer's control
    if attr.span.in_external_macro(cx.sess().source_map()) || is_from_proc_macro(cx, attr) {
        return;
    }

    #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
    span_lint_and_then(
        cx,
        ALLOW_ATTRIBUTES_WITHOUT_REASON,
        attr.span,
        format!("`{name}` attribute without specifying a reason"),
        |diag| {
            diag.help("try adding a reason at the end with `, reason = \"..\"`");
        },
    );
}
