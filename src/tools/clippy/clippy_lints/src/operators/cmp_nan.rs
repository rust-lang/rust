use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::span_lint;
use clippy_utils::in_constant;
use rustc_hir::{BinOpKind, Expr};
use rustc_lint::LateContext;

use super::CMP_NAN;

pub(super) fn check(cx: &LateContext<'_>, e: &Expr<'_>, op: BinOpKind, lhs: &Expr<'_>, rhs: &Expr<'_>) {
    if op.is_comparison() && !in_constant(cx, e.hir_id) && (is_nan(cx, lhs) || is_nan(cx, rhs)) {
        span_lint(
            cx,
            CMP_NAN,
            e.span,
            "doomed comparison with `NAN`, use `{f32,f64}::is_nan()` instead",
        );
    }
}

fn is_nan(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    if let Some((value, _)) = constant(cx, cx.typeck_results(), e) {
        match value {
            Constant::F32(num) => num.is_nan(),
            Constant::F64(num) => num.is_nan(),
            _ => false,
        }
    } else {
        false
    }
}
