use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_from_proc_macro;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `as` conversions.
    ///
    /// Note that this lint is specialized in linting *every single* use of `as`
    /// regardless of whether good alternatives exist or not.
    /// If you want more precise lints for `as`, please consider using these separate lints:
    /// `unnecessary_cast`, `cast_lossless/cast_possible_truncation/cast_possible_wrap/cast_precision_loss/cast_sign_loss`,
    /// `fn_to_numeric_cast(_with_truncation)`, `char_lit_as_u8`, `ref_to_mut` and `ptr_as_ptr`.
    /// There is a good explanation the reason why this lint should work in this way and how it is useful
    /// [in this issue](https://github.com/rust-lang/rust-clippy/issues/5122).
    ///
    /// ### Why restrict this?
    /// `as` conversions will perform many kinds of
    /// conversions, including silently lossy conversions and dangerous coercions.
    /// There are cases when it makes sense to use `as`, so the lint is
    /// Allow by default.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let a: u32;
    /// ...
    /// f(a as u16);
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// f(a.try_into()?);
    ///
    /// // or
    ///
    /// f(a.try_into().expect("Unexpected u16 overflow in f"));
    /// ```
    #[clippy::version = "1.41.0"]
    pub AS_CONVERSIONS,
    restriction,
    "using a potentially dangerous silent `as` conversion"
}

declare_lint_pass!(AsConversions => [AS_CONVERSIONS]);

impl<'tcx> LateLintPass<'tcx> for AsConversions {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
        if let ExprKind::Cast(_, _) = expr.kind
            && !expr.span.in_external_macro(cx.sess().source_map())
            && !is_from_proc_macro(cx, expr)
        {
            #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
            span_lint_and_then(
                cx,
                AS_CONVERSIONS,
                expr.span,
                "using a potentially dangerous silent `as` conversion",
                |diag| {
                    diag.help("consider using a safe wrapper for this conversion");
                },
            );
        }
    }
}
