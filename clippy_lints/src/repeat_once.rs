use clippy_utils::consts::{constant_context, Constant};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::in_macro;
use clippy_utils::source::snippet;
use clippy_utils::ty::is_type_diagnostic_item;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `.repeat(1)` and suggest the following method for each types.
    /// - `.to_string()` for `str`
    /// - `.clone()` for `String`
    /// - `.to_vec()` for `slice`
    ///
    /// The lint will evaluate constant expressions and values as arguments of `.repeat(..)` and emit a message if
    /// they are equivalent to `1`. (Related discussion in [rust-clippy#7306](https://github.com/rust-lang/rust-clippy/issues/7306))
    ///
    /// ### Why is this bad?
    /// For example, `String.repeat(1)` is equivalent to `.clone()`. If cloning
    /// the string is the intention behind this, `clone()` should be used.
    ///
    /// ### Example
    /// ```rust
    /// fn main() {
    ///     let x = String::from("hello world").repeat(1);
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// fn main() {
    ///     let x = String::from("hello world").clone();
    /// }
    /// ```
    pub REPEAT_ONCE,
    complexity,
    "using `.repeat(1)` instead of `String.clone()`, `str.to_string()` or `slice.to_vec()` "
}

declare_lint_pass!(RepeatOnce => [REPEAT_ONCE]);

impl<'tcx> LateLintPass<'tcx> for RepeatOnce {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::MethodCall(path, _, [receiver, count], _) = &expr.kind;
            if path.ident.name == sym!(repeat);
            if constant_context(cx, cx.typeck_results()).expr(count) == Some(Constant::Int(1));
            if !in_macro(receiver.span);
            then {
                let ty = cx.typeck_results().expr_ty(receiver).peel_refs();
                if ty.is_str() {
                    span_lint_and_sugg(
                        cx,
                        REPEAT_ONCE,
                        expr.span,
                        "calling `repeat(1)` on str",
                        "consider using `.to_string()` instead",
                        format!("{}.to_string()", snippet(cx, receiver.span, r#""...""#)),
                        Applicability::MachineApplicable,
                    );
                } else if ty.builtin_index().is_some() {
                    span_lint_and_sugg(
                        cx,
                        REPEAT_ONCE,
                        expr.span,
                        "calling `repeat(1)` on slice",
                        "consider using `.to_vec()` instead",
                        format!("{}.to_vec()", snippet(cx, receiver.span, r#""...""#)),
                        Applicability::MachineApplicable,
                    );
                } else if is_type_diagnostic_item(cx, ty, sym::String) {
                    span_lint_and_sugg(
                        cx,
                        REPEAT_ONCE,
                        expr.span,
                        "calling `repeat(1)` on a string literal",
                        "consider using `.clone()` instead",
                        format!("{}.clone()", snippet(cx, receiver.span, r#""...""#)),
                        Applicability::MachineApplicable,
                    );
                }
            }
        }
    }
}
