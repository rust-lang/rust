use rustc_lint::{LateLintPass, LateContext};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_hir::*;
use crate::utils::{match_function_call, is_type_lang_item, paths, span_lint_and_help};

declare_clippy_lint! {
    /// **What it does:** Prevents the safe `std::mem::drop` function from being called on `std::mem::ManuallyDrop`.
    ///
    /// **Why is this bad?** The safe `drop` function does not drop the inner value of a `ManuallyDrop`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// struct S;
    /// drop(std::mem::ManuallyDrop::new(S));
    /// ```
    /// Use instead:
    /// ```rust
    /// struct S;
    /// unsafe {
    ///     std::mem::ManuallyDrop::drop(std::mem::ManuallyDrop::new(S));
    /// }
    /// ```
    pub UNDROPPED_MANUALLY_DROPS,
    correctness,
    "use of safe `std::mem::drop` function to drop a std::mem::ManuallyDrop, which will not drop the inner value"
}

declare_lint_pass!(UndroppedManuallyDrops => [UNDROPPED_MANUALLY_DROPS]);

impl LateLintPass<'tcx> for UndroppedManuallyDrops {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let Some(ref args) = match_function_call(cx, expr, &paths::DROP) {
            let ty = cx.typeck_results().expr_ty(&args[0]);
            if is_type_lang_item(cx, ty, lang_items::LangItem::ManuallyDrop) {
                span_lint_and_help(
                    cx,
                    UNDROPPED_MANUALLY_DROPS,
                    expr.span,
                    "the inner value of this ManuallyDrop will not be dropped",
                    None,
                    "to drop a `ManuallyDrop<T>`, use std::mem::ManuallyDrop::drop"
                );
            }
        }
    }
}