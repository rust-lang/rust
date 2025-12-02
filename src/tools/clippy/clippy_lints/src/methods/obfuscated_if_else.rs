use super::OBFUSCATED_IF_ELSE;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::eager_or_lazy::switch_to_eager_eval;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::sugg::Sugg;
use clippy_utils::{get_parent_expr, sym};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::Symbol;

#[expect(clippy::needless_pass_by_value)]
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    then_recv: &'tcx Expr<'_>,
    then_arg: &'tcx Expr<'_>,
    then_method_name: Symbol,
    unwrap: Unwrap<'tcx>,
) {
    let recv_ty = cx.typeck_results().expr_ty(then_recv);

    if recv_ty.is_bool() {
        let then_eager = switch_to_eager_eval(cx, then_arg);
        let unwrap_eager = unwrap.arg().is_none_or(|arg| switch_to_eager_eval(cx, arg));

        let mut applicability = if then_eager && unwrap_eager {
            Applicability::MachineApplicable
        } else {
            Applicability::MaybeIncorrect
        };

        let if_then = match then_method_name {
            sym::then if let ExprKind::Closure(closure) = then_arg.kind => {
                let body = cx.tcx.hir_body(closure.body);
                snippet_with_applicability(cx, body.value.span, "..", &mut applicability)
            },
            sym::then_some => snippet_with_applicability(cx, then_arg.span, "..", &mut applicability),
            _ => return,
        };

        let els = match unwrap {
            Unwrap::Or(arg) => snippet_with_applicability(cx, arg.span, "..", &mut applicability),
            Unwrap::OrElse(arg) => match arg.kind {
                ExprKind::Closure(closure) => {
                    let body = cx.tcx.hir_body(closure.body);
                    snippet_with_applicability(cx, body.value.span, "..", &mut applicability)
                },
                ExprKind::Path(_) => snippet_with_applicability(cx, arg.span, "_", &mut applicability) + "()",
                _ => return,
            },
            Unwrap::OrDefault => "Default::default()".into(),
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

pub(super) enum Unwrap<'tcx> {
    Or(&'tcx Expr<'tcx>),
    OrElse(&'tcx Expr<'tcx>),
    OrDefault,
}

impl<'tcx> Unwrap<'tcx> {
    fn arg(&self) -> Option<&'tcx Expr<'tcx>> {
        match self {
            Self::Or(a) | Self::OrElse(a) => Some(a),
            Self::OrDefault => None,
        }
    }
}
