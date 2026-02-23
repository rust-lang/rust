use clippy_utils::res::{MaybeDef, MaybeTypeckRes};
use clippy_utils::{is_in_const_context, is_no_std_crate, sym};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

mod custom_abs;
mod expm1;
mod hypot;
mod lib;
mod ln1p;
mod log_base;
mod log_division;
mod mul_add;
mod powf;
mod powi;
mod radians;

declare_clippy_lint! {
    /// ### What it does
    /// Looks for floating-point expressions that
    /// can be expressed using built-in methods to improve accuracy
    /// at the cost of performance.
    ///
    /// ### Why is this bad?
    /// Negatively impacts accuracy.
    ///
    /// ### Example
    /// ```no_run
    /// let a = 3f32;
    /// let _ = a.powf(1.0 / 3.0);
    /// let _ = (1.0 + a).ln();
    /// let _ = a.exp() - 1.0;
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let a = 3f32;
    /// let _ = a.cbrt();
    /// let _ = a.ln_1p();
    /// let _ = a.exp_m1();
    /// ```
    #[clippy::version = "1.43.0"]
    pub IMPRECISE_FLOPS,
    nursery,
    "usage of imprecise floating point operations"
}

declare_clippy_lint! {
    /// ### What it does
    /// Looks for floating-point expressions that
    /// can be expressed using built-in methods to improve both
    /// accuracy and performance.
    ///
    /// ### Why is this bad?
    /// Negatively impacts accuracy and performance.
    ///
    /// ### Example
    /// ```no_run
    /// use std::f32::consts::E;
    ///
    /// let a = 3f32;
    /// let _ = (2f32).powf(a);
    /// let _ = E.powf(a);
    /// let _ = a.powf(1.0 / 2.0);
    /// let _ = a.log(2.0);
    /// let _ = a.log(10.0);
    /// let _ = a.log(E);
    /// let _ = a.powf(2.0);
    /// let _ = a * 2.0 + 4.0;
    /// let _ = if a < 0.0 {
    ///     -a
    /// } else {
    ///     a
    /// };
    /// let _ = if a < 0.0 {
    ///     a
    /// } else {
    ///     -a
    /// };
    /// ```
    ///
    /// is better expressed as
    ///
    /// ```no_run
    /// use std::f32::consts::E;
    ///
    /// let a = 3f32;
    /// let _ = a.exp2();
    /// let _ = a.exp();
    /// let _ = a.sqrt();
    /// let _ = a.log2();
    /// let _ = a.log10();
    /// let _ = a.ln();
    /// let _ = a.powi(2);
    /// let _ = a.mul_add(2.0, 4.0);
    /// let _ = a.abs();
    /// let _ = -a.abs();
    /// ```
    #[clippy::version = "1.43.0"]
    pub SUBOPTIMAL_FLOPS,
    nursery,
    "usage of sub-optimal floating point operations"
}

declare_lint_pass!(FloatingPointArithmetic => [
    IMPRECISE_FLOPS,
    SUBOPTIMAL_FLOPS
]);

impl<'tcx> LateLintPass<'tcx> for FloatingPointArithmetic {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        // All of these operations are currently not const and are in std.
        if is_in_const_context(cx) {
            return;
        }

        if let ExprKind::MethodCall(path, receiver, args, _) = expr.kind {
            let recv_ty = cx.typeck_results().expr_ty(receiver);

            if recv_ty.is_floating_point() && !is_no_std_crate(cx) && cx.ty_based_def(expr).opt_parent(cx).is_impl(cx) {
                match path.ident.name {
                    sym::ln => ln1p::check(cx, expr, receiver),
                    sym::log => log_base::check(cx, expr, receiver, args),
                    sym::powf => powf::check(cx, expr, receiver, args),
                    sym::powi => powi::check(cx, expr, receiver, args),
                    sym::sqrt => hypot::check(cx, expr, receiver),
                    _ => {},
                }
            }
        } else {
            if !is_no_std_crate(cx) {
                expm1::check(cx, expr);
                mul_add::check(cx, expr);
                custom_abs::check(cx, expr);
                log_division::check(cx, expr);
            }
            radians::check(cx, expr);
        }
    }
}
