use std::convert::Infallible;
use std::ops::ControlFlow;

use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint;
use clippy_utils::visitors::{Descend, for_each_expr_without_closures};
use clippy_utils::{method_chain_args, sext, sym};
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::Symbol;

use super::CAST_SIGN_LOSS;

/// A list of methods that can never return a negative value.
/// Includes methods that panic rather than returning a negative value.
///
/// Methods that can overflow and return a negative value must not be included in this list,
/// because casting their return values can still result in sign loss.
const METHODS_RET_POSITIVE: &[Symbol] = &[
    sym::checked_abs,
    sym::saturating_abs,
    sym::isqrt,
    sym::checked_isqrt,
    sym::rem_euclid,
    sym::checked_rem_euclid,
    sym::wrapping_rem_euclid,
];

/// A list of methods that act like `pow()`. See `pow_call_result_sign()` for details.
///
/// Methods that can overflow and return a negative value must not be included in this list,
/// because casting their return values can still result in sign loss.
const METHODS_POW: &[Symbol] = &[sym::pow, sym::saturating_pow, sym::checked_pow];

/// A list of methods that act like `unwrap()`, and don't change the sign of the inner value.
const METHODS_UNWRAP: &[Symbol] = &[sym::unwrap, sym::unwrap_unchecked, sym::expect, sym::into_ok];

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
            format!("casting `{cast_from}` to `{cast_to}` may lose the sign of the value"),
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

            if let Sign::ZeroOrPositive = expr_muldiv_sign(cx, cast_op) {
                return false;
            }

            if let Sign::ZeroOrPositive = expr_add_sign(cx, cast_op) {
                return false;
            }

            true
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

    if let Constant::Int(n) = ConstEvalCtxt::new(cx).eval(expr)?
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

    if let Constant::Int(n) = ConstEvalCtxt::new(cx).eval(expr)?
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

fn expr_sign<'cx, 'tcx>(cx: &LateContext<'cx>, mut expr: &'tcx Expr<'tcx>, ty: impl Into<Option<Ty<'cx>>>) -> Sign {
    // Try evaluate this expr first to see if it's positive
    if let Some(val) = get_const_signed_int_eval(cx, expr, ty) {
        return if val >= 0 { Sign::ZeroOrPositive } else { Sign::Negative };
    }
    if let Some(_val) = get_const_unsigned_int_eval(cx, expr, None) {
        return Sign::ZeroOrPositive;
    }

    // Calling on methods that always return non-negative values.
    if let ExprKind::MethodCall(path, caller, args, ..) = expr.kind {
        let mut method_name = path.ident.name;

        // Peel unwrap(), expect(), etc.
        while let Some(&found_name) = METHODS_UNWRAP.iter().find(|&name| &method_name == name)
            && let Some(arglist) = method_chain_args(expr, &[found_name])
            && let ExprKind::MethodCall(inner_path, recv, ..) = &arglist[0].0.kind
        {
            // The original type has changed, but we can't use `ty` here anyway, because it has been
            // moved.
            method_name = inner_path.ident.name;
            expr = recv;
        }

        if METHODS_POW.contains(&method_name)
            && let [arg] = args
        {
            return pow_call_result_sign(cx, caller, arg);
        } else if METHODS_RET_POSITIVE.contains(&method_name) {
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

/// Peels binary operators such as [`BinOpKind::Mul`] or [`BinOpKind::Rem`],
/// where the result could always be positive. See [`exprs_with_muldiv_binop_peeled()`] for details.
///
/// Returns the sign of the list of peeled expressions.
fn expr_muldiv_sign(cx: &LateContext<'_>, expr: &Expr<'_>) -> Sign {
    let mut negative_count = 0;

    // Peel off possible binary expressions, for example:
    // x * x / y => [x, x, y]
    // a % b => [a]
    let exprs = exprs_with_muldiv_binop_peeled(expr);
    for expr in exprs {
        match expr_sign(cx, expr, None) {
            Sign::Negative => negative_count += 1,
            // A mul/div is:
            // - uncertain if there are any uncertain values (because they could be negative or positive),
            Sign::Uncertain => return Sign::Uncertain,
            Sign::ZeroOrPositive => (),
        }
    }

    // A mul/div is:
    // - negative if there are an odd number of negative values,
    // - positive or zero otherwise.
    if negative_count % 2 == 1 {
        Sign::Negative
    } else {
        Sign::ZeroOrPositive
    }
}

/// Peels binary operators such as [`BinOpKind::Add`], where the result could always be positive.
/// See [`exprs_with_add_binop_peeled()`] for details.
///
/// Returns the sign of the list of peeled expressions.
fn expr_add_sign(cx: &LateContext<'_>, expr: &Expr<'_>) -> Sign {
    let mut negative_count = 0;
    let mut positive_count = 0;

    // Peel off possible binary expressions, for example:
    // a + b + c => [a, b, c]
    let exprs = exprs_with_add_binop_peeled(expr);
    for expr in exprs {
        match expr_sign(cx, expr, None) {
            Sign::Negative => negative_count += 1,
            // A sum is:
            // - uncertain if there are any uncertain values (because they could be negative or positive),
            Sign::Uncertain => return Sign::Uncertain,
            Sign::ZeroOrPositive => positive_count += 1,
        }
    }

    // A sum is:
    // - positive or zero if there are only positive (or zero) values,
    // - negative if there are only negative (or zero) values, or
    // - uncertain if there are both.
    // We could split Zero out into its own variant, but we don't yet.
    if negative_count == 0 {
        Sign::ZeroOrPositive
    } else if positive_count == 0 {
        Sign::Negative
    } else {
        Sign::Uncertain
    }
}

/// Peels binary operators such as [`BinOpKind::Mul`], [`BinOpKind::Div`] or [`BinOpKind::Rem`],
/// where the result depends on:
///
/// - the number of negative values in the entire expression, or
/// - the number of negative values on the left hand side of the expression.
///
/// Ignores overflow.
///
///
/// Expressions using other operators are preserved, so we can try to evaluate them later.
fn exprs_with_muldiv_binop_peeled<'e>(expr: &'e Expr<'_>) -> Vec<&'e Expr<'e>> {
    let mut res = vec![];

    for_each_expr_without_closures(expr, |sub_expr| -> ControlFlow<Infallible, Descend> {
        // We don't check for mul/div/rem methods here, but we could.
        if let ExprKind::Binary(op, lhs, _rhs) = sub_expr.kind {
            if matches!(op.node, BinOpKind::Mul | BinOpKind::Div) {
                // For binary operators where both sides contribute to the sign of the result,
                // collect all their operands, recursively. This ignores overflow.
                ControlFlow::Continue(Descend::Yes)
            } else if matches!(op.node, BinOpKind::Rem | BinOpKind::Shr) {
                // For binary operators where the left hand side determines the sign of the result,
                // only collect that side, recursively. Overflow panics, so this always holds.
                //
                // Large left shifts turn negatives into zeroes, so we can't use it here.
                //
                // > Given remainder = dividend % divisor, the remainder will have the same sign as the dividend
                // > ...
                // > Arithmetic right shift on signed integer types
                // https://doc.rust-lang.org/reference/expressions/operator-expr.html#arithmetic-and-logical-binary-operators

                // We want to descend into the lhs, but skip the rhs.
                // That's tricky to do using for_each_expr(), so we just keep the lhs intact.
                res.push(lhs);
                ControlFlow::Continue(Descend::No)
            } else {
                // The sign of the result of other binary operators depends on the values of the operands,
                // so try to evaluate the expression.
                res.push(sub_expr);
                ControlFlow::Continue(Descend::No)
            }
        } else {
            // For other expressions, including unary operators and constants, try to evaluate the expression.
            res.push(sub_expr);
            ControlFlow::Continue(Descend::No)
        }
    });

    res
}

/// Peels binary operators such as [`BinOpKind::Add`], where the result depends on:
///
/// - all the expressions being positive, or
/// - all the expressions being negative.
///
/// Ignores overflow.
///
/// Expressions using other operators are preserved, so we can try to evaluate them later.
fn exprs_with_add_binop_peeled<'e>(expr: &'e Expr<'_>) -> Vec<&'e Expr<'e>> {
    let mut res = vec![];

    for_each_expr_without_closures(expr, |sub_expr| -> ControlFlow<Infallible, Descend> {
        // We don't check for add methods here, but we could.
        if let ExprKind::Binary(op, _lhs, _rhs) = sub_expr.kind {
            if matches!(op.node, BinOpKind::Add) {
                // For binary operators where both sides contribute to the sign of the result,
                // collect all their operands, recursively. This ignores overflow.
                ControlFlow::Continue(Descend::Yes)
            } else {
                // The sign of the result of other binary operators depends on the values of the operands,
                // so try to evaluate the expression.
                res.push(sub_expr);
                ControlFlow::Continue(Descend::No)
            }
        } else {
            // For other expressions, including unary operators and constants, try to evaluate the expression.
            res.push(sub_expr);
            ControlFlow::Continue(Descend::No)
        }
    });

    res
}
