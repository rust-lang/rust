use super::EXPLICIT_INTO_ITER_LOOP;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{match_trait_method, paths};
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::TyS;

pub(super) fn check(cx: &LateContext<'_>, self_arg: &'hir Expr<'hir>, call_expr: &Expr<'_>) {
    let self_ty = cx.typeck_results().expr_ty(self_arg);
    let self_ty_adjusted = cx.typeck_results().expr_ty_adjusted(self_arg);
    if !(TyS::same_type(self_ty, self_ty_adjusted) && match_trait_method(cx, call_expr, &paths::INTO_ITERATOR)) {
        return;
    }

    let mut applicability = Applicability::MachineApplicable;
    let object = snippet_with_applicability(cx, self_arg.span, "_", &mut applicability);
    span_lint_and_sugg(
        cx,
        EXPLICIT_INTO_ITER_LOOP,
        call_expr.span,
        "it is more concise to loop over containers instead of using explicit \
            iteration methods",
        "to write this more concisely, try",
        object.to_string(),
        applicability,
    );
}
