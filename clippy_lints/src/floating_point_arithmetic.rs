use crate::consts::{
    constant, Constant,
    Constant::{F32, F64},
};
use crate::utils::*;
use if_chain::if_chain;
use rustc::declare_lint_pass;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc_errors::Applicability;
use rustc_session::declare_tool_lint;
use std::f32::consts as f32_consts;
use std::f64::consts as f64_consts;

declare_clippy_lint! {
    /// **What it does:** Looks for floating-point expressions that
    /// can be expressed using built-in methods to improve accuracy,
    /// performance and/or succinctness.
    ///
    /// **Why is this bad?** Negatively affects accuracy, performance
    /// and/or readability.
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    ///
    /// ```rust
    /// use std::f32::consts::E;
    ///
    /// let a = 3f32;
    /// let _ = (2f32).powf(a);
    /// let _ = E.powf(a);
    /// let _ = a.powf(1.0 / 2.0);
    /// let _ = a.powf(1.0 / 3.0);
    /// let _ = a.log(2.0);
    /// let _ = a.log(10.0);
    /// let _ = a.log(E);
    /// let _ = (1.0 + a).ln();
    /// let _ = a.exp() - 1.0;
    /// let _ = a.powf(2.0);
    /// ```
    ///
    /// is better expressed as
    ///
    /// ```rust
    /// use std::f32::consts::E;
    ///
    /// let a = 3f32;
    /// let _ = a.exp2();
    /// let _ = a.exp();
    /// let _ = a.sqrt();
    /// let _ = a.cbrt();
    /// let _ = a.log2();
    /// let _ = a.log10();
    /// let _ = a.ln();
    /// let _ = a.ln_1p();
    /// let _ = a.exp_m1();
    /// let _ = a.powi(2);
    /// ```
    pub FLOATING_POINT_IMPROVEMENTS,
    nursery,
    "looks for improvements to floating-point expressions"
}

declare_lint_pass!(FloatingPointArithmetic => [FLOATING_POINT_IMPROVEMENTS]);

// Returns the specialized log method for a given base if base is constant
// and is one of 2, 10 and e
fn get_specialized_log_method(cx: &LateContext<'_, '_>, base: &Expr) -> Option<&'static str> {
    if let Some((value, _)) = constant(cx, cx.tables, base) {
        if F32(2.0) == value || F64(2.0) == value {
            return Some("log2");
        } else if F32(10.0) == value || F64(10.0) == value {
            return Some("log10");
        } else if F32(f32_consts::E) == value || F64(f64_consts::E) == value {
            return Some("ln");
        }
    }

    None
}

fn check_log_base(cx: &LateContext<'_, '_>, expr: &Expr, args: &HirVec<Expr>) {
    if let Some(method) = get_specialized_log_method(cx, &args[1]) {
        span_lint_and_sugg(
            cx,
            FLOATING_POINT_IMPROVEMENTS,
            expr.span,
            "logarithm for bases 2, 10 and e can be computed more accurately",
            "consider using",
            format!("{}.{}()", sugg::Sugg::hir(cx, &args[0], ".."), method),
            Applicability::MachineApplicable,
        );
    }
}

// TODO: Lint expressions of the form `(x + y).ln()` where y > 1 and
// suggest usage of `(x + (y - 1)).ln_1p()` instead
fn check_ln1p(cx: &LateContext<'_, '_>, expr: &Expr, args: &HirVec<Expr>) {
    if_chain! {
        if let ExprKind::Binary(op, ref lhs, ref rhs) = &args[0].kind;
        if op.node == BinOpKind::Add;
        then {
            let recv = match (constant(cx, cx.tables, lhs), constant(cx, cx.tables, rhs)) {
                (Some((value, _)), _) if F32(1.0) == value || F64(1.0) == value => rhs,
                (_, Some((value, _))) if F32(1.0) == value || F64(1.0) == value => lhs,
                _ => return,
            };

            span_lint_and_sugg(
                cx,
                FLOATING_POINT_IMPROVEMENTS,
                expr.span,
                "ln(1 + x) can be computed more accurately",
                "consider using",
                format!("{}.ln_1p()", sugg::Sugg::hir(cx, recv, "..").maybe_par()),
                Applicability::MachineApplicable,
            );
        }
    }
}

// Returns an integer if the float constant is a whole number and it
// can be converted to an integer without loss
// TODO: Add a better check to determine whether the float can be
// casted without loss
#[allow(clippy::cast_possible_truncation)]
fn get_integer_from_float_constant(value: &Constant) -> Option<i64> {
    match value {
        F32(num) if (num.trunc() - num).abs() <= std::f32::EPSILON => {
            if *num > -16_777_217.0 && *num < 16_777_217.0 {
                Some(num.round() as i64)
            } else {
                None
            }
        },
        F64(num) if (num.trunc() - num).abs() <= std::f64::EPSILON => {
            if *num > -9_007_199_254_740_993.0 && *num < 9_007_199_254_740_993.0 {
                Some(num.round() as i64)
            } else {
                None
            }
        },
        _ => None,
    }
}

fn check_powf(cx: &LateContext<'_, '_>, expr: &Expr, args: &HirVec<Expr>) {
    // Check receiver
    if let Some((value, _)) = constant(cx, cx.tables, &args[0]) {
        let method;

        if F32(f32_consts::E) == value || F64(f64_consts::E) == value {
            method = "exp";
        } else if F32(2.0) == value || F64(2.0) == value {
            method = "exp2";
        } else {
            return;
        }

        span_lint_and_sugg(
            cx,
            FLOATING_POINT_IMPROVEMENTS,
            expr.span,
            "exponent for bases 2 and e can be computed more accurately",
            "consider using",
            format!("{}.{}()", sugg::Sugg::hir(cx, &args[1], "..").maybe_par(), method),
            Applicability::MachineApplicable,
        );
    }

    // Check argument
    if let Some((value, _)) = constant(cx, cx.tables, &args[1]) {
        let help;
        let method;

        if F32(1.0 / 2.0) == value || F64(1.0 / 2.0) == value {
            help = "square-root of a number can be computed more efficiently and accurately";
            method = "sqrt";
        } else if F32(1.0 / 3.0) == value || F64(1.0 / 3.0) == value {
            help = "cube-root of a number can be computed more accurately";
            method = "cbrt";
        } else if let Some(exponent) = get_integer_from_float_constant(&value) {
            span_lint_and_sugg(
                cx,
                FLOATING_POINT_IMPROVEMENTS,
                expr.span,
                "exponentiation with integer powers can be computed more efficiently",
                "consider using",
                format!("{}.powi({})", sugg::Sugg::hir(cx, &args[0], ".."), exponent),
                Applicability::MachineApplicable,
            );

            return;
        } else {
            return;
        }

        span_lint_and_sugg(
            cx,
            FLOATING_POINT_IMPROVEMENTS,
            expr.span,
            help,
            "consider using",
            format!("{}.{}()", sugg::Sugg::hir(cx, &args[0], ".."), method),
            Applicability::MachineApplicable,
        );
    }
}

// TODO: Lint expressions of the form `x.exp() - y` where y > 1
// and suggest usage of `x.exp_m1() - (y - 1)` instead
fn check_expm1(cx: &LateContext<'_, '_>, expr: &Expr) {
    if_chain! {
        if let ExprKind::Binary(op, ref lhs, ref rhs) = expr.kind;
        if op.node == BinOpKind::Sub;
        if cx.tables.expr_ty(lhs).is_floating_point();
        if let Some((value, _)) = constant(cx, cx.tables, rhs);
        if F32(1.0) == value || F64(1.0) == value;
        if let ExprKind::MethodCall(ref path, _, ref method_args) = lhs.kind;
        if path.ident.name.as_str() == "exp";
        then {
            span_lint_and_sugg(
                cx,
                FLOATING_POINT_IMPROVEMENTS,
                expr.span,
                "(e.pow(x) - 1) can be computed more accurately",
                "consider using",
                format!(
                    "{}.exp_m1()",
                    sugg::Sugg::hir(cx, &method_args[0], "..")
                ),
                Applicability::MachineApplicable,
            );
        }
    }
}

// Checks whether two expressions evaluate to the same value
fn are_exprs_equivalent(cx: &LateContext<'_, '_>, left: &Expr, right: &Expr) -> bool {
    // Checks whether the values are constant and equal
    if_chain! {
        if let Some((left_value, _)) = constant(cx, cx.tables, left);
        if let Some((right_value, _)) = constant(cx, cx.tables, right);
        if left_value == right_value;
        then {
            return true;
        }
    }

    // Checks whether the expressions resolve to the same variable
    if_chain! {
        if let ExprKind::Path(ref left_qpath) = left.kind;
        if let QPath::Resolved(_, ref left_path) = *left_qpath;
        if left_path.segments.len() == 1;
        if let def::Res::Local(left_local_id) = qpath_res(cx, left_qpath, left.hir_id);
        if let ExprKind::Path(ref right_qpath) = right.kind;
        if let QPath::Resolved(_, ref right_path) = *right_qpath;
        if right_path.segments.len() == 1;
        if let def::Res::Local(right_local_id) = qpath_res(cx, right_qpath, right.hir_id);
        if left_local_id == right_local_id;
        then {
            return true;
        }
    }

    false
}

fn check_log_division(cx: &LateContext<'_, '_>, expr: &Expr) {
    let log_methods = ["log", "log2", "log10", "ln"];

    if_chain! {
        if let ExprKind::Binary(op, ref lhs, ref rhs) = expr.kind;
        if op.node == BinOpKind::Div;
        if cx.tables.expr_ty(lhs).is_floating_point();
        if let ExprKind::MethodCall(left_path, _, left_args) = &lhs.kind;
        if let ExprKind::MethodCall(right_path, _, right_args) = &rhs.kind;
        let left_method = left_path.ident.name.as_str();
        if left_method == right_path.ident.name.as_str();
        if log_methods.iter().any(|&method| left_method == method);
        then {
            let left_recv = &left_args[0];
            let right_recv = &right_args[0];

            // Return early when bases are not equal
            if left_method == "log" && !are_exprs_equivalent(cx, &left_args[1], &right_args[1]) {
                return;
            }

            // Reduce the expression further for bases 2, 10 and e
            let suggestion = if let Some(method) = get_specialized_log_method(cx, right_recv) {
                format!("{}.{}()", sugg::Sugg::hir(cx, left_recv, ".."), method)
            } else {
                format!(
                    "{}.log({})",
                    sugg::Sugg::hir(cx, left_recv, ".."),
                    sugg::Sugg::hir(cx, right_recv, "..")
                )
            };

            span_lint_and_sugg(
                cx,
                FLOATING_POINT_IMPROVEMENTS,
                expr.span,
                "x.log(b) / y.log(b) can be reduced to x.log(y)",
                "consider using",
                suggestion,
                Applicability::MachineApplicable,
            );
        }
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for FloatingPointArithmetic {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let ExprKind::MethodCall(ref path, _, args) = &expr.kind {
            let recv_ty = cx.tables.expr_ty(&args[0]);

            if recv_ty.is_floating_point() {
                match &*path.ident.name.as_str() {
                    "ln" => check_ln1p(cx, expr, args),
                    "log" => check_log_base(cx, expr, args),
                    "powf" => check_powf(cx, expr, args),
                    _ => {},
                }
            }
        } else {
            check_expm1(cx, expr);
            check_log_division(cx, expr);
        }
    }
}
