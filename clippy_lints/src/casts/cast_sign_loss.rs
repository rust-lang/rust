use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::span_lint;
use clippy_utils::{method_chain_args, sext};
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};

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

/// A list of methods that act like `pow()`. See `pow_call_result_sign()` for details.
///
/// Methods that can overflow and return a negative value must not be included in this list,
/// because casting their return values can still result in sign loss.
const METHODS_POW: &[&str] = &["pow", "saturating_pow", "checked_pow"];

/// A list of methods that act like `unwrap()`, and don't change the sign of the inner value.
const METHODS_UNWRAP: &[&str] = &["unwrap", "unwrap_unchecked", "expect", "into_ok"];

pub(super) fn check<'cx>(
    cx: &LateContext<'cx>,
    expr: &Expr<'_>,
    cast_op: &Expr<'_>,
    cast_from: Ty<'cx>,
    cast_to: Ty<'_>,
) {
    if should_lint(cx, cast_op, cast_from, cast_to) {
        span_lint(
            cx,
            CAST_SIGN_LOSS,
            expr.span,
            &format!("casting `{cast_from}` to `{cast_to}` may lose the sign of the value"),
        );
    }
}

fn should_lint<'cx>(cx: &LateContext<'cx>, cast_op: &Expr<'_>, cast_from: Ty<'cx>, cast_to: Ty<'_>) -> bool {
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
                match expr_sign(cx, expr, None) {
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

fn get_const_signed_int_eval<'cx>(
    cx: &LateContext<'cx>,
    expr: &Expr<'_>,
    ty: impl Into<Option<Ty<'cx>>>,
) -> Option<i128> {
    let ty = ty.into().unwrap_or_else(|| cx.typeck_results().expr_ty(expr));

    if let Constant::Int(n) = constant(cx, cx.typeck_results(), expr)?
        && let ty::Int(ity) = *ty.kind()
    {
        return Some(sext(cx.tcx, n, ity));
    }
    None
}

fn get_const_unsigned_int_eval<'cx>(
    cx: &LateContext<'cx>,
    expr: &Expr<'_>,
    ty: impl Into<Option<Ty<'cx>>>,
) -> Option<u128> {
    let ty = ty.into().unwrap_or_else(|| cx.typeck_results().expr_ty(expr));

    if let Constant::Int(n) = constant(cx, cx.typeck_results(), expr)?
        && let ty::Uint(_ity) = *ty.kind()
    {
        return Some(n);
    }
    None
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Sign {
    ZeroOrPositive,
    Negative,
    Uncertain,
}

fn expr_sign<'cx>(cx: &LateContext<'cx>, expr: &Expr<'_>, ty: impl Into<Option<Ty<'cx>>>) -> Sign {
    // Try evaluate this expr first to see if it's positive
    if let Some(val) = get_const_signed_int_eval(cx, expr, ty) {
        return if val >= 0 { Sign::ZeroOrPositive } else { Sign::Negative };
    }
    if let Some(_val) = get_const_unsigned_int_eval(cx, expr, None) {
        return Sign::ZeroOrPositive;
    }

    // Calling on methods that always return non-negative values.
    if let ExprKind::MethodCall(path, caller, args, ..) = expr.kind {
        let mut method_name = path.ident.name.as_str();

        // Peel unwrap(), expect(), etc.
        while let Some(&found_name) = METHODS_UNWRAP.iter().find(|&name| &method_name == name)
            && let Some(arglist) = method_chain_args(expr, &[found_name])
            && let ExprKind::MethodCall(inner_path, ..) = &arglist[0].0.kind
        {
            // The original type has changed, but we can't use `ty` here anyway, because it has been
            // moved.
            method_name = inner_path.ident.name.as_str();
        }

        if METHODS_POW.iter().any(|&name| method_name == name)
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
/// If the exponent is a even number, the result is always positive,
/// Otherwise, if the base is negative, and the exponent is an odd number, the result is always
/// negative.
///
/// Otherwise, returns [`Sign::Uncertain`].
fn pow_call_result_sign(cx: &LateContext<'_>, base: &Expr<'_>, exponent: &Expr<'_>) -> Sign {
    let base_sign = expr_sign(cx, base, None);

    // Rust's integer pow() functions take an unsigned exponent.
    let exponent_val = get_const_unsigned_int_eval(cx, exponent, None);
    let exponent_is_even = exponent_val.map(|val| val % 2 == 0);

    match (base_sign, exponent_is_even) {
        // Non-negative bases always return non-negative results, ignoring overflow.
        (Sign::ZeroOrPositive, _) |
        // Any base raised to an even exponent is non-negative.
        // These both hold even if we don't know the value of the base.
        (_, Some(true))
            => Sign::ZeroOrPositive,

        // A negative base raised to an odd exponent is non-negative.
        (Sign::Negative, Some(false)) => Sign::Negative,

        // Negative/unknown base to an unknown exponent, or unknown base to an odd exponent.
        // Could be negative or positive depending on the actual values.
        (Sign::Negative | Sign::Uncertain, None) |
        (Sign::Uncertain, Some(false)) => Sign::Uncertain,
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
