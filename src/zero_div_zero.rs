use rustc::lint::*;
use rustc_front::hir::*;

use utils::span_help_and_lint;
use consts::{Constant, constant_simple, FloatWidth};

/// ZeroDivZeroPass is a pass that checks for a binary expression that consists
/// of 0.0/0.0, which is always NaN. It is more clear to replace instances of
/// 0.0/0.0 with std::f32::NaN or std::f64::NaN, depending on the precision.
pub struct ZeroDivZeroPass;

/// **What it does:** This lint checks for `0.0 / 0.0`
///
/// **Why is this bad?** It's less readable than `std::f32::NAN` or `std::f64::NAN`
///
/// **Known problems:** None
///
/// **Example** `0.0f32 / 0.0`
declare_lint!(pub ZERO_DIVIDED_BY_ZERO, Warn,
              "usage of `0.0 / 0.0` to obtain NaN instead of std::f32::NaN or std::f64::NaN");

impl LintPass for ZeroDivZeroPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(ZERO_DIVIDED_BY_ZERO)
    }
}

impl LateLintPass for ZeroDivZeroPass {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        // check for instances of 0.0/0.0
        if_let_chain! {
            [
                let ExprBinary(ref op, ref left, ref right) = expr.node,
                let BinOp_::BiDiv = op.node,
                // TODO - constant_simple does not fold many operations involving floats.
                // That's probably fine for this lint - it's pretty unlikely that someone would
                // do something like 0.0/(2.0 - 2.0), but it would be nice to warn on that case too.
                let Some(Constant::ConstantFloat(ref lhs_value, lhs_width)) = constant_simple(left),
                let Some(Constant::ConstantFloat(ref rhs_value, rhs_width)) = constant_simple(right),
                let Some(0.0) = lhs_value.parse().ok(),
                let Some(0.0) = rhs_value.parse().ok()
            ],
            {
                // since we're about to suggest a use of std::f32::NaN or std::f64::NaN,
                // match the precision of the literals that are given.
                let float_type = match (lhs_width, rhs_width) {
                    (FloatWidth::Fw64, _)
                    | (_, FloatWidth::Fw64) => "f64",
                    _ => "f32"
                };
                span_help_and_lint(cx, ZERO_DIVIDED_BY_ZERO, expr.span,
                    "constant division of 0.0 with 0.0 will always result in NaN",
                    &format!("Consider using `std::{}::NAN` if you would like a constant representing NaN", float_type));
            }
        }
    }
}
