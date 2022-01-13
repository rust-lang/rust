use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::ty::is_type_lang_item;
use clippy_utils::{match_function_call, paths};
use rustc_hir::{lang_items, Expr};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Prevents the safe `std::mem::drop` function from being called on `std::mem::ManuallyDrop`.
    ///
    /// ### Why is this bad?
    /// The safe `drop` function does not drop the inner value of a `ManuallyDrop`.
    ///
    /// ### Known problems
    /// Does not catch cases if the user binds `std::mem::drop`
    /// to a different name and calls it that way.
    ///
    /// ### Example
    /// ```rust
    /// struct S;
    /// drop(std::mem::ManuallyDrop::new(S));
    /// ```
    /// Use instead:
    /// ```rust
    /// struct S;
    /// unsafe {
    ///     std::mem::ManuallyDrop::drop(&mut std::mem::ManuallyDrop::new(S));
    /// }
    /// ```
    #[clippy::version = "1.49.0"]
    pub UNDROPPED_MANUALLY_DROPS,
    correctness,
    "use of safe `std::mem::drop` function to drop a std::mem::ManuallyDrop, which will not drop the inner value"
}

declare_lint_pass!(UndroppedManuallyDrops => [UNDROPPED_MANUALLY_DROPS]);

impl<'tcx> LateLintPass<'tcx> for UndroppedManuallyDrops {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let Some([arg_0, ..]) = match_function_call(cx, expr, &paths::DROP) {
            let ty = cx.typeck_results().expr_ty(arg_0);
            if is_type_lang_item(cx, ty, lang_items::LangItem::ManuallyDrop) {
                span_lint_and_help(
                    cx,
                    UNDROPPED_MANUALLY_DROPS,
                    expr.span,
                    "the inner value of this ManuallyDrop will not be dropped",
                    None,
                    "to drop a `ManuallyDrop<T>`, use std::mem::ManuallyDrop::drop",
                );
            }
        }
    }
}
