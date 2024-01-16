use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::span_lint;
use clippy_utils::{clip, method_chain_args, sext};
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty, UintTy};

use super::CAST_SIGN_LOSS;

/// A list of methods that can never return a negative value.
/// Includes methods that panic rather than returning a negative value.
///
/// Methods that can overflow and return a negative value must not be included in this list,
/// because casting their return values can still result in sign loss.
const METHODS_RET_POSITIVE: &[&str] = &[
    "checked_abs",
    "saturating_abs",
    "isqrt",
    "checked_isqrt",
    "rem_euclid",
    "checked_rem_euclid",
    "wrapping_rem_euclid",
];

/// A list of methods that act like `pow()`, and can never return:
/// - a negative value from a non-negative base
/// - a negative value from a negative base and even exponent
/// - a non-negative value from a negative base and odd exponent
///
/// Methods that can overflow and return a negative value must not be included in this list,
/// because casting their return values can still result in sign loss.
const METHODS_POW: &[&str] = &["pow", "saturating_pow", "checked_pow"];

/// A list of methods that act like `unwrap()`, and don't change the sign of the inner value.
const METHODS_UNWRAP: &[&str] = &["unwrap", "unwrap_unchecked", "expect", "into_ok"];

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, cast_op: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>) {
    if should_lint(cx, cast_op, cast_from, cast_to) {
        span_lint(
            cx,
            CAST_SIGN_LOSS,
            expr.span,
            &format!("casting `{cast_from}` to `{cast_to}` may lose the sign of the value"),
        );
    }
}

fn should_lint(cx: &LateContext<'_>, cast_op: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>) -> bool {
    match (cast_from.is_integral(), cast_to.is_integral()) {
        (true, true) => {
            if !cast_from.is_signed() || cast_to.is_signed() {
                return false;
            }

            // Don't lint if `cast_op` is known to be positive, ignoring overflow.
            if let Sign::ZeroOrPositive = expr_sign(cx, cast_op, cast_from) {
                return false;
            }

            let (mut uncertain_count, mut negative_count) = (0, 0);
            // Peel off possible binary expressions, for example:
            // x * x / y => [x, x, y]
            // a % b => [a]
            let exprs = exprs_with_selected_binop_peeled(cast_op);
            for expr in exprs {
                let ty = cx.typeck_results().expr_ty(expr);
                match expr_sign(cx, expr, ty) {
                    Sign::Negative => negative_count += 1,
                    Sign::Uncertain => uncertain_count += 1,
                    Sign::ZeroOrPositive => (),
                };
            }

            // Lint if there are any uncertain results (because they could be negative or positive),
            // or an odd number of negative results.
            uncertain_count > 0 || negative_count % 2 == 1
        },

        (false, true) => !cast_to.is_signed(),

        (_, _) => false,
    }
}

fn get_const_int_eval(cx: &LateContext<'_>, expr: &Expr<'_>, ty: Ty<'_>) -> Option<i128> {
    if let Constant::Int(n) = constant(cx, cx.typeck_results(), expr)?
        && let ty::Int(ity) = *ty.kind()
    {
        return Some(sext(cx.tcx, n, ity));
    }
    None
}

enum Sign {
    ZeroOrPositive,
    Negative,
    Uncertain,
}

fn expr_sign(cx: &LateContext<'_>, expr: &Expr<'_>, ty: Ty<'_>) -> Sign {
    // Try evaluate this expr first to see if it's positive
    if let Some(val) = get_const_int_eval(cx, expr, ty) {
        return if val >= 0 { Sign::ZeroOrPositive } else { Sign::Negative };
    }
    // Calling on methods that always return non-negative values.
    if let ExprKind::MethodCall(path, caller, args, ..) = expr.kind {
        let mut method_name = path.ident.name.as_str();

        if method_name == "unwrap"
            && let Some(arglist) = method_chain_args(expr, &["unwrap"])
            && let ExprKind::MethodCall(inner_path, ..) = &arglist[0].0.kind
        {
            method_name = inner_path.ident.name.as_str();
        }
        if method_name == "expect"
            && let Some(arglist) = method_chain_args(expr, &["expect"])
            && let ExprKind::MethodCall(inner_path, ..) = &arglist[0].0.kind
        {
            method_name = inner_path.ident.name.as_str();
        }

        if method_name == "pow"
            && let [arg] = args
        {
            return pow_call_result_sign(cx, caller, arg);
        } else if METHODS_RET_POSITIVE.iter().any(|&name| method_name == name) {
            return Sign::ZeroOrPositive;
        }
    }

    Sign::Uncertain
}

/// Return the sign of the `pow` call's result, ignoring overflow.
///
/// If the base is positive, the result is always positive.
/// If the base is negative, and the exponent is a even number, the result is always positive,
/// otherwise if the exponent is an odd number, the result is always negative.
///
/// If either value can't be evaluated, [`Sign::Uncertain`] will be returned.
fn pow_call_result_sign(cx: &LateContext<'_>, base: &Expr<'_>, exponent: &Expr<'_>) -> Sign {
    let base_ty = cx.typeck_results().expr_ty(base);
    let Some(base_val) = get_const_int_eval(cx, base, base_ty) else {
        return Sign::Uncertain;
    };
    // Non-negative bases raised to non-negative exponents are always non-negative, ignoring overflow.
    // (Rust's integer pow() function takes an unsigned exponent.)
    if base_val >= 0 {
        return Sign::ZeroOrPositive;
    }

    let Some(Constant::Int(n)) = constant(cx, cx.typeck_results(), exponent) else {
        return Sign::Uncertain;
    };

    // A negative value raised to an even exponent is non-negative, and an odd exponent
    // is negative, ignoring overflow.
    if clip(cx.tcx, n, UintTy::U32) % 2 == 0 {
        return Sign::ZeroOrPositive;
    } else {
        return Sign::Negative;
    }
}

/// Peels binary operators such as [`BinOpKind::Mul`], [`BinOpKind::Div`] or [`BinOpKind::Rem`],
/// which the result could always be positive under certain conditions, ignoring overflow.
///
/// Expressions using other operators are preserved, so we can try to evaluate them later.
fn exprs_with_selected_binop_peeled<'a>(expr: &'a Expr<'_>) -> Vec<&'a Expr<'a>> {
    #[inline]
    fn collect_operands<'a>(expr: &'a Expr<'a>, operands: &mut Vec<&'a Expr<'a>>) {
        match expr.kind {
            ExprKind::Binary(op, lhs, rhs) => {
                if matches!(op.node, BinOpKind::Mul | BinOpKind::Div) {
                    // For binary operators which both contribute to the sign of the result,
                    // collect all their operands, recursively. This ignores overflow.
                    collect_operands(lhs, operands);
                    collect_operands(rhs, operands);
                } else if matches!(op.node, BinOpKind::Rem) {
                    // For binary operators where the left hand side determines the sign of the result,
                    // only collect that side, recursively. Overflow panics, so this always holds.
                    //
                    // > Given remainder = dividend % divisor, the remainder will have the same sign as the dividend
                    // https://doc.rust-lang.org/reference/expressions/operator-expr.html#arithmetic-and-logical-binary-operators
                    collect_operands(lhs, operands);
                } else {
                    // The sign of the result of other binary operators depends on the values of the operands,
                    // so try to evaluate the expression.
                    operands.push(expr);
                }
            },
            // For other expressions, including unary operators and constants, try to evaluate the expression.
            _ => operands.push(expr),
        }
    }

    let mut res = vec![];
    collect_operands(expr, &mut res);
    res
}
