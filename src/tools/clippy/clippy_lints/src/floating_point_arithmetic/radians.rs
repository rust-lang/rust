use clippy_utils::consts::ConstEvalCtxt;
use clippy_utils::consts::Constant::{F32, F64};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sugg::Sugg;
use rustc_ast::ast;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::source_map::Spanned;
use std::f32::consts as f32_consts;
use std::f64::consts as f64_consts;

use super::SUBOPTIMAL_FLOPS;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if let ExprKind::Binary(
        Spanned {
            node: BinOpKind::Div, ..
        },
        div_lhs,
        div_rhs,
    ) = expr.kind
        && let ExprKind::Binary(
            Spanned {
                node: BinOpKind::Mul, ..
            },
            mul_lhs,
            mul_rhs,
        ) = div_lhs.kind
        && let ecx = ConstEvalCtxt::new(cx)
        && let Some(rvalue) = ecx.eval(div_rhs)
        && let Some(lvalue) = ecx.eval(mul_rhs)
    {
        // TODO: also check for constant values near PI/180 or 180/PI
        if (F32(f32_consts::PI) == rvalue || F64(f64_consts::PI) == rvalue)
            && (F32(180_f32) == lvalue || F64(180_f64) == lvalue)
        {
            span_lint_and_then(
                cx,
                SUBOPTIMAL_FLOPS,
                expr.span,
                "conversion to degrees can be done more accurately",
                |diag| {
                    let mut app = Applicability::MachineApplicable;
                    let recv = Sugg::hir_with_applicability(cx, mul_lhs, "num", &mut app);
                    let proposal = if let ExprKind::Lit(literal) = mul_lhs.kind
                        && let ast::LitKind::Float(ref value, float_type) = literal.node
                        && float_type == ast::LitFloatType::Unsuffixed
                    {
                        if value.as_str().ends_with('.') {
                            format!("{recv}0_f64.to_degrees()")
                        } else {
                            format!("{recv}_f64.to_degrees()")
                        }
                    } else {
                        format!("{}.to_degrees()", recv.maybe_paren())
                    };
                    diag.span_suggestion(expr.span, "consider using", proposal, app);
                },
            );
        } else if (F32(180_f32) == rvalue || F64(180_f64) == rvalue)
            && (F32(f32_consts::PI) == lvalue || F64(f64_consts::PI) == lvalue)
        {
            span_lint_and_then(
                cx,
                SUBOPTIMAL_FLOPS,
                expr.span,
                "conversion to radians can be done more accurately",
                |diag| {
                    let mut app = Applicability::MachineApplicable;
                    let recv = Sugg::hir_with_applicability(cx, mul_lhs, "num", &mut app);
                    let proposal = if let ExprKind::Lit(literal) = mul_lhs.kind
                        && let ast::LitKind::Float(ref value, float_type) = literal.node
                        && float_type == ast::LitFloatType::Unsuffixed
                    {
                        if value.as_str().ends_with('.') {
                            format!("{recv}0_f64.to_radians()")
                        } else {
                            format!("{recv}_f64.to_radians()")
                        }
                    } else {
                        format!("{}.to_radians()", recv.maybe_paren())
                    };
                    diag.span_suggestion(expr.span, "consider using", proposal, app);
                },
            );
        }
    }
}
