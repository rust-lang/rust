use clippy_utils::consts::{constant_with_source, Constant};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::get_item_name;
use clippy_utils::sugg::Sugg;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, UnOp};
use rustc_lint::LateContext;
use rustc_middle::ty;

use super::{FLOAT_CMP, FLOAT_CMP_CONST};

pub(crate) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    op: BinOpKind,
    left: &'tcx Expr<'_>,
    right: &'tcx Expr<'_>,
) {
    if (op == BinOpKind::Eq || op == BinOpKind::Ne) && (is_float(cx, left) || is_float(cx, right)) {
        let left_is_local = match constant_with_source(cx, cx.typeck_results(), left) {
            Some((c, s)) if !is_allowed(&c) => s.is_local(),
            Some(_) => return,
            None => true,
        };
        let right_is_local = match constant_with_source(cx, cx.typeck_results(), right) {
            Some((c, s)) if !is_allowed(&c) => s.is_local(),
            Some(_) => return,
            None => true,
        };

        // Allow comparing the results of signum()
        if is_signum(cx, left) && is_signum(cx, right) {
            return;
        }

        if let Some(name) = get_item_name(cx, expr) {
            let name = name.as_str();
            if name == "eq" || name == "ne" || name == "is_nan" || name.starts_with("eq_") || name.ends_with("_eq") {
                return;
            }
        }
        let is_comparing_arrays = is_array(cx, left) || is_array(cx, right);
        let (lint, msg) = get_lint_and_message(left_is_local && right_is_local, is_comparing_arrays);
        span_lint_and_then(cx, lint, expr.span, msg, |diag| {
            let lhs = Sugg::hir(cx, left, "..");
            let rhs = Sugg::hir(cx, right, "..");

            if !is_comparing_arrays {
                diag.span_suggestion(
                    expr.span,
                    "consider comparing them within some margin of error",
                    format!(
                        "({}).abs() {} error_margin",
                        lhs - rhs,
                        if op == BinOpKind::Eq { '<' } else { '>' }
                    ),
                    Applicability::HasPlaceholders, // snippet
                );
            }
            diag.note("`f32::EPSILON` and `f64::EPSILON` are available for the `error_margin`");
        });
    }
}

fn get_lint_and_message(is_local: bool, is_comparing_arrays: bool) -> (&'static rustc_lint::Lint, &'static str) {
    if is_local {
        (
            FLOAT_CMP,
            if is_comparing_arrays {
                "strict comparison of `f32` or `f64` arrays"
            } else {
                "strict comparison of `f32` or `f64`"
            },
        )
    } else {
        (
            FLOAT_CMP_CONST,
            if is_comparing_arrays {
                "strict comparison of `f32` or `f64` constant arrays"
            } else {
                "strict comparison of `f32` or `f64` constant"
            },
        )
    }
}

fn is_allowed(val: &Constant<'_>) -> bool {
    match val {
        &Constant::F32(f) => f == 0.0 || f.is_infinite(),
        &Constant::F64(f) => f == 0.0 || f.is_infinite(),
        Constant::Vec(vec) => vec.iter().all(|f| match f {
            Constant::F32(f) => *f == 0.0 || (*f).is_infinite(),
            Constant::F64(f) => *f == 0.0 || (*f).is_infinite(),
            _ => false,
        }),
        _ => false,
    }
}

// Return true if `expr` is the result of `signum()` invoked on a float value.
fn is_signum(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    // The negation of a signum is still a signum
    if let ExprKind::Unary(UnOp::Neg, child_expr) = expr.kind {
        return is_signum(cx, child_expr);
    }

    if_chain! {
        if let ExprKind::MethodCall(method_name, self_arg, ..) = expr.kind;
        if sym!(signum) == method_name.ident.name;
        // Check that the receiver of the signum() is a float (expressions[0] is the receiver of
        // the method call)
        then {
            return is_float(cx, self_arg);
        }
    }
    false
}

fn is_float(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let value = &cx.typeck_results().expr_ty(expr).peel_refs().kind();

    if let ty::Array(arr_ty, _) = value {
        return matches!(arr_ty.kind(), ty::Float(_));
    };

    matches!(value, ty::Float(_))
}

fn is_array(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    matches!(&cx.typeck_results().expr_ty(expr).peel_refs().kind(), ty::Array(_, _))
}
