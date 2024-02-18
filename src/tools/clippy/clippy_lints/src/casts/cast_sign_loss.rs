use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::span_lint;
use clippy_utils::{clip, method_chain_args, sext};
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty, UintTy};

use super::CAST_SIGN_LOSS;

const METHODS_RET_POSITIVE: &[&str] = &["abs", "checked_abs", "rem_euclid", "checked_rem_euclid"];

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

            // Don't lint if `cast_op` is known to be positive.
            if let Sign::ZeroOrPositive = expr_sign(cx, cast_op, cast_from) {
                return false;
            }

            let (mut uncertain_count, mut negative_count) = (0, 0);
            // Peel off possible binary expressions, e.g. x * x * y => [x, x, y]
            let Some(exprs) = exprs_with_selected_binop_peeled(cast_op) else {
                // Assume cast sign lose if we cannot determine the sign of `cast_op`
                return true;
            };
            for expr in exprs {
                let ty = cx.typeck_results().expr_ty(expr);
                match expr_sign(cx, expr, ty) {
                    Sign::Negative => negative_count += 1,
                    Sign::Uncertain => uncertain_count += 1,
                    Sign::ZeroOrPositive => (),
                };
            }

            // Lint if there are odd number of uncertain or negative results
            uncertain_count % 2 == 1 || negative_count % 2 == 1
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

/// Return the sign of the `pow` call's result.
///
/// If the caller is a positive number, the result is always positive,
/// If the `power_of` is a even number, the result is always positive as well,
/// Otherwise a [`Sign::Uncertain`] will be returned.
fn pow_call_result_sign(cx: &LateContext<'_>, caller: &Expr<'_>, power_of: &Expr<'_>) -> Sign {
    let caller_ty = cx.typeck_results().expr_ty(caller);
    if let Some(caller_val) = get_const_int_eval(cx, caller, caller_ty)
        && caller_val >= 0
    {
        return Sign::ZeroOrPositive;
    }

    if let Some(Constant::Int(n)) = constant(cx, cx.typeck_results(), power_of)
        && clip(cx.tcx, n, UintTy::U32) % 2 == 0
    {
        return Sign::ZeroOrPositive;
    }

    Sign::Uncertain
}

/// Peels binary operators such as [`BinOpKind::Mul`], [`BinOpKind::Div`] or [`BinOpKind::Rem`],
/// which the result could always be positive under certain condition.
///
/// Other operators such as `+`/`-` causing the result's sign hard to determine, which we will
/// return `None`
fn exprs_with_selected_binop_peeled<'a>(expr: &'a Expr<'_>) -> Option<Vec<&'a Expr<'a>>> {
    #[inline]
    fn collect_operands<'a>(expr: &'a Expr<'a>, operands: &mut Vec<&'a Expr<'a>>) -> Option<()> {
        match expr.kind {
            ExprKind::Binary(op, lhs, rhs) => {
                if matches!(op.node, BinOpKind::Mul | BinOpKind::Div | BinOpKind::Rem) {
                    collect_operands(lhs, operands);
                    operands.push(rhs);
                } else {
                    // Things are complicated when there are other binary ops exist,
                    // abort checking by returning `None` for now.
                    return None;
                }
            },
            _ => operands.push(expr),
        }
        Some(())
    }

    let mut res = vec![];
    collect_operands(expr, &mut res)?;
    Some(res)
}
