use super::OBFUSCATED_IF_ELSE;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::eager_or_lazy::switch_to_eager_eval;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sugg::Sugg;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::ExprKind;
use rustc_lint::LateContext;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    then_recv: &'tcx hir::Expr<'_>,
    then_arg: &'tcx hir::Expr<'_>,
    unwrap_arg: &'tcx hir::Expr<'_>,
    then_method_name: &str,
) {
    let recv_ty = cx.typeck_results().expr_ty(then_recv);

    if recv_ty.is_bool() {
        let mut applicability = if switch_to_eager_eval(cx, then_arg) && switch_to_eager_eval(cx, unwrap_arg) {
            Applicability::MachineApplicable
        } else {
            Applicability::MaybeIncorrect
        };

        let if_then = match then_method_name {
            "then" if let ExprKind::Closure(closure) = then_arg.kind => {
                let body = cx.tcx.hir_body(closure.body);
                snippet_with_applicability(cx, body.value.span, "..", &mut applicability)
            },
            "then_some" => snippet_with_applicability(cx, then_arg.span, "..", &mut applicability),
            _ => String::new().into(),
        };

        let sugg = format!(
            "if {} {{ {} }} else {{ {} }}",
            Sugg::hir_with_applicability(cx, then_recv, "..", &mut applicability),
            if_then,
            snippet_with_applicability(cx, unwrap_arg.span, "..", &mut applicability)
        );

        span_lint_and_sugg(
            cx,
            OBFUSCATED_IF_ELSE,
            expr.span,
            "this method chain can be written more clearly with `if .. else ..`",
            "try",
            sugg,
            applicability,
        );
    }
}
