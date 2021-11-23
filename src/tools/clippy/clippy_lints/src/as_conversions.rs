use clippy_utils::diagnostics::span_lint_and_help;
use rustc_ast::ast::{Expr, ExprKind};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `as` conversions.
    ///
    /// Note that this lint is specialized in linting *every single* use of `as`
    /// regardless of whether good alternatives exist or not.
    /// If you want more precise lints for `as`, please consider using these separate lints:
    /// `unnecessary_cast`, `cast_lossless/possible_truncation/possible_wrap/precision_loss/sign_loss`,
    /// `fn_to_numeric_cast(_with_truncation)`, `char_lit_as_u8`, `ref_to_mut` and `ptr_as_ptr`.
    /// There is a good explanation the reason why this lint should work in this way and how it is useful
    /// [in this issue](https://github.com/rust-lang/rust-clippy/issues/5122).
    ///
    /// ### Why is this bad?
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
    /// Usually better represents the semantics you expect:
    /// ```rust,ignore
    /// f(a.try_into()?);
    /// ```
    /// or
    /// ```rust,ignore
    /// f(a.try_into().expect("Unexpected u16 overflow in f"));
    /// ```
    ///
    #[clippy::version = "1.41.0"]
    pub AS_CONVERSIONS,
    restriction,
    "using a potentially dangerous silent `as` conversion"
}

declare_lint_pass!(AsConversions => [AS_CONVERSIONS]);

impl EarlyLintPass for AsConversions {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if in_external_macro(cx.sess, expr.span) {
            return;
        }

        if let ExprKind::Cast(_, _) = expr.kind {
            span_lint_and_help(
                cx,
                AS_CONVERSIONS,
                expr.span,
                "using a potentially dangerous silent `as` conversion",
                None,
                "consider using a safe wrapper for this conversion",
            );
        }
    }
}
