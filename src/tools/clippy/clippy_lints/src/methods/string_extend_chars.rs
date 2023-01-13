use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::method_chain_args;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_lang_item;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty;

use super::STRING_EXTEND_CHARS;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, recv: &hir::Expr<'_>, arg: &hir::Expr<'_>) {
    let obj_ty = cx.typeck_results().expr_ty(recv).peel_refs();
    if !is_type_lang_item(cx, obj_ty, hir::LangItem::String) {
        return;
    }
    if let Some(arglists) = method_chain_args(arg, &["chars"]) {
        let target = &arglists[0].0;
        let self_ty = cx.typeck_results().expr_ty(target).peel_refs();
        let ref_str = if *self_ty.kind() == ty::Str {
            if matches!(target.kind, hir::ExprKind::Index(..)) {
                "&"
            } else {
                ""
            }
        } else if is_type_lang_item(cx, self_ty, hir::LangItem::String) {
            "&"
        } else {
            return;
        };

        let mut applicability = Applicability::MachineApplicable;
        span_lint_and_sugg(
            cx,
            STRING_EXTEND_CHARS,
            expr.span,
            "calling `.extend(_.chars())`",
            "try this",
            format!(
                "{}.push_str({ref_str}{})",
                snippet_with_applicability(cx, recv.span, "..", &mut applicability),
                snippet_with_applicability(cx, target.span, "..", &mut applicability)
            ),
            applicability,
        );
    }
}
