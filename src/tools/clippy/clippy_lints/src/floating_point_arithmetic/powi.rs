use clippy_utils::consts::ConstEvalCtxt;
use clippy_utils::consts::Constant::Int;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sugg::Sugg;
use clippy_utils::{get_parent_expr, sym};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, PathSegment};
use rustc_lint::LateContext;
use rustc_span::source_map::Spanned;

use super::SUBOPTIMAL_FLOPS;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, receiver: &Expr<'_>, args: &[Expr<'_>]) {
    if let Some(value) = ConstEvalCtxt::new(cx).eval(&args[0])
        && value == Int(2)
        && let Some(parent) = get_parent_expr(cx, expr)
    {
        if let Some(grandparent) = get_parent_expr(cx, parent)
            && let ExprKind::MethodCall(PathSegment { ident: method, .. }, receiver, ..) = grandparent.kind
            && method.name == sym::sqrt
            // we don't care about the applicability as this is an early-return condition
            && super::hypot::detect(cx, receiver, &mut Applicability::Unspecified).is_some()
        {
            return;
        }

        if let ExprKind::Binary(
            Spanned {
                node: op @ (BinOpKind::Add | BinOpKind::Sub),
                ..
            },
            lhs,
            rhs,
        ) = parent.kind
        {
            span_lint_and_then(
                cx,
                SUBOPTIMAL_FLOPS,
                parent.span,
                "multiply and add expressions can be calculated more efficiently and accurately",
                |diag| {
                    let other_addend = if lhs.hir_id == expr.hir_id { rhs } else { lhs };

                    // Negate expr if original code has subtraction and expr is on the right side
                    let maybe_neg_sugg = |expr, hir_id, app: &mut _| {
                        let sugg = Sugg::hir_with_applicability(cx, expr, "_", app);
                        if matches!(op, BinOpKind::Sub) && hir_id == rhs.hir_id {
                            -sugg
                        } else {
                            sugg
                        }
                    };

                    let mut app = Applicability::MachineApplicable;
                    diag.span_suggestion(
                        parent.span,
                        "consider using",
                        format!(
                            "{}.mul_add({}, {})",
                            Sugg::hir_with_applicability(cx, receiver, "_", &mut app).maybe_paren(),
                            maybe_neg_sugg(receiver, expr.hir_id, &mut app),
                            maybe_neg_sugg(other_addend, other_addend.hir_id, &mut app),
                        ),
                        app,
                    );
                },
            );
        }
    }
}
