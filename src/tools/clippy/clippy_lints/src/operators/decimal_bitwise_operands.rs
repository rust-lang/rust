use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::numeric_literal;
use clippy_utils::numeric_literal::NumericLiteral;
use clippy_utils::source::SpanRangeExt;
use rustc_ast::LitKind;
use rustc_data_structures::packed::Pu128;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::Span;

use super::DECIMAL_BITWISE_OPERANDS;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, op: BinOpKind, left: &'tcx Expr<'_>, right: &'tcx Expr<'_>) {
    if !matches!(op, BinOpKind::BitAnd | BinOpKind::BitOr | BinOpKind::BitXor) {
        return;
    }

    for expr in [left, right] {
        check_expr(cx, expr);
    }
}

fn check_expr(cx: &LateContext<'_>, expr: &Expr<'_>) {
    match &expr.kind {
        ExprKind::Block(block, _) => {
            if let Some(block_expr) = block.expr {
                check_expr(cx, block_expr);
            }
        },
        ExprKind::Cast(cast_expr, _) => {
            check_expr(cx, cast_expr);
        },
        ExprKind::Unary(_, unary_expr) => {
            check_expr(cx, unary_expr);
        },
        ExprKind::AddrOf(_, _, addr_of_expr) => {
            check_expr(cx, addr_of_expr);
        },
        ExprKind::Lit(lit) => {
            if let LitKind::Int(Pu128(val), _) = lit.node
                && !is_single_digit(val)
                && !is_power_of_twoish(val)
                && let Some(src) = lit.span.get_source_text(cx)
                && let Some(num_lit) = NumericLiteral::from_lit_kind(&src, &lit.node)
                && num_lit.is_decimal()
            {
                emit_lint(cx, lit.span, num_lit.suffix, val);
            }
        },
        _ => (),
    }
}

fn is_power_of_twoish(val: u128) -> bool {
    val.is_power_of_two() || val.wrapping_add(1).is_power_of_two()
}

fn is_single_digit(val: u128) -> bool {
    val <= 9
}

fn emit_lint(cx: &LateContext<'_>, span: Span, suffix: Option<&str>, val: u128) {
    span_lint_and_help(
        cx,
        DECIMAL_BITWISE_OPERANDS,
        span,
        "using decimal literal for bitwise operation",
        None,
        format!(
            "use binary ({}), hex ({}), or octal ({}) notation for better readability",
            numeric_literal::format(&format!("{val:#b}"), suffix, false),
            numeric_literal::format(&format!("{val:#x}"), suffix, false),
            numeric_literal::format(&format!("{val:#o}"), suffix, false),
        ),
    );
}
