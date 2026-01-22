use clippy_utils::consts::ConstEvalCtxt;
use clippy_utils::consts::Constant::Int;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::sugg::Sugg;
use clippy_utils::{eq_expr_value, sym};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, PathSegment};
use rustc_lint::LateContext;
use rustc_span::source_map::Spanned;

use super::IMPRECISE_FLOPS;

pub(super) fn detect(cx: &LateContext<'_>, receiver: &Expr<'_>, app: &mut Applicability) -> Option<String> {
    if let ExprKind::Binary(
        Spanned {
            node: BinOpKind::Add, ..
        },
        add_lhs,
        add_rhs,
    ) = receiver.kind
    {
        // check if expression of the form x * x + y * y
        if let ExprKind::Binary(
            Spanned {
                node: BinOpKind::Mul, ..
            },
            lmul_lhs,
            lmul_rhs,
        ) = add_lhs.kind
            && let ExprKind::Binary(
                Spanned {
                    node: BinOpKind::Mul, ..
                },
                rmul_lhs,
                rmul_rhs,
            ) = add_rhs.kind
            && eq_expr_value(cx, lmul_lhs, lmul_rhs)
            && eq_expr_value(cx, rmul_lhs, rmul_rhs)
        {
            return Some(format!(
                "{}.hypot({})",
                Sugg::hir_with_applicability(cx, lmul_lhs, "_", app).maybe_paren(),
                Sugg::hir_with_applicability(cx, rmul_lhs, "_", app)
            ));
        }

        // check if expression of the form x.powi(2) + y.powi(2)
        if let ExprKind::MethodCall(PathSegment { ident: lmethod, .. }, largs_0, [largs_1, ..], _) = add_lhs.kind
            && let ExprKind::MethodCall(PathSegment { ident: rmethod, .. }, rargs_0, [rargs_1, ..], _) = add_rhs.kind
            && lmethod.name == sym::powi
            && rmethod.name == sym::powi
            && let ecx = ConstEvalCtxt::new(cx)
            && let Some(lvalue) = ecx.eval(largs_1)
            && let Some(rvalue) = ecx.eval(rargs_1)
            && Int(2) == lvalue
            && Int(2) == rvalue
        {
            return Some(format!(
                "{}.hypot({})",
                Sugg::hir_with_applicability(cx, largs_0, "_", app).maybe_paren(),
                Sugg::hir_with_applicability(cx, rargs_0, "_", app)
            ));
        }
    }

    None
}

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, receiver: &Expr<'_>) {
    let mut app = Applicability::MachineApplicable;
    if let Some(message) = detect(cx, receiver, &mut app) {
        span_lint_and_sugg(
            cx,
            IMPRECISE_FLOPS,
            expr.span,
            "hypotenuse can be computed more accurately",
            "consider using",
            message,
            app,
        );
    }
}
