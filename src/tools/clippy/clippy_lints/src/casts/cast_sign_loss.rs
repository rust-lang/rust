use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::span_lint;
use clippy_utils::{method_chain_args, sext};
use if_chain::if_chain;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};

use super::CAST_SIGN_LOSS;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, cast_op: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>) {
    if should_lint(cx, cast_op, cast_from, cast_to) {
        span_lint(
            cx,
            CAST_SIGN_LOSS,
            expr.span,
            &format!(
                "casting `{}` to `{}` may lose the sign of the value",
                cast_from, cast_to
            ),
        );
    }
}

fn should_lint(cx: &LateContext<'_>, cast_op: &Expr<'_>, cast_from: Ty<'_>, cast_to: Ty<'_>) -> bool {
    match (cast_from.is_integral(), cast_to.is_integral()) {
        (true, true) => {
            if !cast_from.is_signed() || cast_to.is_signed() {
                return false;
            }

            // Don't lint for positive constants.
            let const_val = constant(cx, cx.typeck_results(), cast_op);
            if_chain! {
                if let Some((Constant::Int(n), _)) = const_val;
                if let ty::Int(ity) = *cast_from.kind();
                if sext(cx.tcx, n, ity) >= 0;
                then {
                    return false;
                }
            }

            // Don't lint for the result of methods that always return non-negative values.
            if let ExprKind::MethodCall(path, ..) = cast_op.kind {
                let mut method_name = path.ident.name.as_str();
                let allowed_methods = ["abs", "checked_abs", "rem_euclid", "checked_rem_euclid"];

                if_chain! {
                    if method_name == "unwrap";
                    if let Some(arglist) = method_chain_args(cast_op, &["unwrap"]);
                    if let ExprKind::MethodCall(inner_path, ..) = &arglist[0].0.kind;
                    then {
                        method_name = inner_path.ident.name.as_str();
                    }
                }

                if allowed_methods.iter().any(|&name| method_name == name) {
                    return false;
                }
            }

            true
        },

        (false, true) => !cast_to.is_signed(),

        (_, _) => false,
    }
}
