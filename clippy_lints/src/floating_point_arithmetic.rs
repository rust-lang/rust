use crate::consts::{
    constant,
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
    /// **What it does:** Looks for numerically unstable floating point
    /// computations and suggests better alternatives.
    ///
    /// **Why is this bad?** Numerically unstable floating point computations
    /// cause rounding errors to magnify and distorts the results strongly.
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    ///
    /// ```rust
    /// use std::f32::consts::E;
    ///
    /// let a = 1f32.log(2.0);
    /// let b = 1f32.log(10.0);
    /// let c = 1f32.log(E);
    /// ```
    ///
    /// is better expressed as
    ///
    /// ```rust
    /// let a = 1f32.log2();
    /// let b = 1f32.log10();
    /// let c = 1f32.ln();
    /// ```
    pub INACCURATE_FLOATING_POINT_COMPUTATION,
    nursery,
    "checks for numerically unstable floating point computations"
}

declare_clippy_lint! {
    /// **What it does:** Looks for inefficient floating point computations
    /// and suggests faster alternatives.
    ///
    /// **Why is this bad?** Lower performance.
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    ///
    /// ```rust
    /// use std::f32::consts::E;
    ///
    /// let a = (2f32).powf(3.0);
    /// let c = E.powf(3.0);
    /// ```
    ///
    /// is better expressed as
    ///
    /// ```rust
    /// let a = (3f32).exp2();
    /// let b = (3f32).exp();
    /// ```
    pub SLOW_FLOATING_POINT_COMPUTATION,
    nursery,
    "checks for inefficient floating point computations"
}

declare_lint_pass!(FloatingPointArithmetic => [
    INACCURATE_FLOATING_POINT_COMPUTATION,
    SLOW_FLOATING_POINT_COMPUTATION
]);

fn check_log_base(cx: &LateContext<'_, '_>, expr: &Expr, args: &HirVec<Expr>) {
    let recv = &args[0];
    let arg = sugg::Sugg::hir(cx, recv, "..").maybe_par();

    if let Some((value, _)) = constant(cx, cx.tables, &args[1]) {
        let method;

        if F32(2.0) == value || F64(2.0) == value {
            method = "log2";
        } else if F32(10.0) == value || F64(10.0) == value {
            method = "log10";
        } else if F32(f32_consts::E) == value || F64(f64_consts::E) == value {
            method = "ln";
        } else {
            return;
        }

        span_lint_and_sugg(
            cx,
            INACCURATE_FLOATING_POINT_COMPUTATION,
            expr.span,
            "logarithm for bases 2, 10 and e can be computed more accurately",
            "consider using",
            format!("{}.{}()", arg, method),
            Applicability::MachineApplicable,
        );
    }
}

// TODO: Lint expressions of the form `(x + 1).ln()` and `(x + y).ln()`
// where y > 1 and suggest usage of `(x + (y - 1)).ln_1p()` instead
fn check_ln1p(cx: &LateContext<'_, '_>, expr: &Expr, args: &HirVec<Expr>) {
    if_chain! {
        if let ExprKind::Binary(op, ref lhs, ref rhs) = &args[0].kind;
        if op.node == BinOpKind::Add;
        if let Some((value, _)) = constant(cx, cx.tables, lhs);
        if F32(1.0) == value || F64(1.0) == value;
        then {
            let arg = sugg::Sugg::hir(cx, rhs, "..").maybe_par();

            span_lint_and_sugg(
                cx,
                INACCURATE_FLOATING_POINT_COMPUTATION,
                expr.span,
                "ln(1 + x) can be computed more accurately",
                "consider using",
                format!("{}.ln_1p()", arg),
                Applicability::MachineApplicable,
            );
        }
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
            SLOW_FLOATING_POINT_COMPUTATION,
            expr.span,
            "exponent for bases 2 and e can be computed more efficiently",
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
            help = "square-root of a number can be computer more efficiently";
            method = "sqrt";
        } else if F32(1.0 / 3.0) == value || F64(1.0 / 3.0) == value {
            help = "cube-root of a number can be computer more efficiently";
            method = "cbrt";
        } else {
            return;
        }

        span_lint_and_sugg(
            cx,
            SLOW_FLOATING_POINT_COMPUTATION,
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
                INACCURATE_FLOATING_POINT_COMPUTATION,
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
        }
    }
}
