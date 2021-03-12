use super::EXPLICIT_INTO_ITER_LOOP;
use crate::utils::{snippet_with_applicability, span_lint_and_sugg};
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::TyS;

pub(super) fn check(cx: &LateContext<'_>, args: &'hir [Expr<'hir>], arg: &Expr<'_>) {
    let receiver_ty = cx.typeck_results().expr_ty(&args[0]);
    let receiver_ty_adjusted = cx.typeck_results().expr_ty_adjusted(&args[0]);
    if !TyS::same_type(receiver_ty, receiver_ty_adjusted) {
        return;
    }

    let mut applicability = Applicability::MachineApplicable;
    let object = snippet_with_applicability(cx, args[0].span, "_", &mut applicability);
    span_lint_and_sugg(
        cx,
        EXPLICIT_INTO_ITER_LOOP,
        arg.span,
        "it is more concise to loop over containers instead of using explicit \
            iteration methods",
        "to write this more concisely, try",
        object.to_string(),
        applicability,
    );
}
