use super::OBFUSCATED_IF_ELSE;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::eager_or_lazy::switch_to_eager_eval;
use clippy_utils::get_parent_expr;
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
    unwrap_arg: Option<&'tcx hir::Expr<'_>>,
    then_method_name: &str,
    unwrap_method_name: &str,
) {
    let recv_ty = cx.typeck_results().expr_ty(then_recv);

    if recv_ty.is_bool() {
        let then_eager = switch_to_eager_eval(cx, then_arg);
        let unwrap_eager = unwrap_arg.is_none_or(|arg| switch_to_eager_eval(cx, arg));

        let mut applicability = if then_eager && unwrap_eager {
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
            _ => return,
        };

        // FIXME: Add `unwrap_or_else` and `unwrap_or_default` symbol
        let els = match unwrap_method_name {
            "unwrap_or" => snippet_with_applicability(cx, unwrap_arg.unwrap().span, "..", &mut applicability),
            "unwrap_or_else" if let ExprKind::Closure(closure) = unwrap_arg.unwrap().kind => {
                let body = cx.tcx.hir_body(closure.body);
                snippet_with_applicability(cx, body.value.span, "..", &mut applicability)
            },
            "unwrap_or_else" if let ExprKind::Path(_) = unwrap_arg.unwrap().kind => {
                snippet_with_applicability(cx, unwrap_arg.unwrap().span, "_", &mut applicability) + "()"
            },
            "unwrap_or_default" => "Default::default()".into(),
            _ => return,
        };

        let sugg = format!(
            "if {} {{ {} }} else {{ {} }}",
            Sugg::hir_with_applicability(cx, then_recv, "..", &mut applicability),
            if_then,
            els
        );

        // To be parsed as an expression, the `if { … } else { … }` as the left operand of a binary operator
        // requires parentheses.
        let sugg = if let Some(parent_expr) = get_parent_expr(cx, expr)
            && let ExprKind::Binary(_, left, _) = parent_expr.kind
            && left.hir_id == expr.hir_id
        {
            format!("({sugg})")
        } else {
            sugg
        };

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
