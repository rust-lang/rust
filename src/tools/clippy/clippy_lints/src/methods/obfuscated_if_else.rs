use super::OBFUSCATED_IF_ELSE;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    then_recv: &'tcx hir::Expr<'_>,
    then_arg: &'tcx hir::Expr<'_>,
    unwrap_arg: &'tcx hir::Expr<'_>,
) {
    // something.then_some(blah).unwrap_or(blah)
    // ^^^^^^^^^-then_recv ^^^^-then_arg   ^^^^- unwrap_arg
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^- expr

    let recv_ty = cx.typeck_results().expr_ty(then_recv);

    if recv_ty.is_bool() {
        let mut applicability = Applicability::MachineApplicable;
        let sugg = format!(
            "if {} {{ {} }} else {{ {} }}",
            snippet_with_applicability(cx, then_recv.span, "..", &mut applicability),
            snippet_with_applicability(cx, then_arg.span, "..", &mut applicability),
            snippet_with_applicability(cx, unwrap_arg.span, "..", &mut applicability)
        );

        span_lint_and_sugg(
            cx,
            OBFUSCATED_IF_ELSE,
            expr.span,
            "use of `.then_some(..).unwrap_or(..)` can be written \
            more clearly with `if .. else ..`",
            "try",
            sugg,
            applicability,
        );
    }
}
