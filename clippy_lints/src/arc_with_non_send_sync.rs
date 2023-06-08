use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::last_path_segment;
use clippy_utils::ty::{implements_trait, is_type_diagnostic_item};
use if_chain::if_chain;

use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_lint::LateLintPass;
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::sym;

declare_clippy_lint! {
    /// ### What it does.
    /// This lint warns when you use `Arc` with a type that does not implement `Send` or `Sync`.
    ///
    /// ### Why is this bad?
    /// Wrapping a type in Arc doesn't add thread safety to the underlying data, so data races
    /// could occur when touching the underlying data.
    ///
    /// ### Example
    /// ```rust
    /// # use std::cell::RefCell;
    /// # use std::sync::Arc;
    ///
    /// fn main() {
    ///     // This is safe, as `i32` implements `Send` and `Sync`.
    ///     let a = Arc::new(42);
    ///
    ///     // This is not safe, as `RefCell` does not implement `Sync`.
    ///     let b = Arc::new(RefCell::new(42));
    /// }
    /// ```
    #[clippy::version = "1.72.0"]
    pub ARC_WITH_NON_SEND_SYNC,
    correctness,
    "using `Arc` with a type that does not implement `Send` or `Sync`"
}
declare_lint_pass!(ArcWithNonSendSync => [ARC_WITH_NON_SEND_SYNC]);

impl LateLintPass<'_> for ArcWithNonSendSync {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        let ty = cx.typeck_results().expr_ty(expr);
        if_chain! {
            if is_type_diagnostic_item(cx, ty, sym::Arc);
            if let ExprKind::Call(func, [arg]) = expr.kind;
            if let ExprKind::Path(func_path) = func.kind;
            if last_path_segment(&func_path).ident.name == sym::new;
            if let arg_ty = cx.typeck_results().expr_ty(arg);
            if !matches!(arg_ty.kind(), ty::Param(_));
            if !cx.tcx
                .lang_items()
                .sync_trait()
                .map_or(false, |id| implements_trait(cx, arg_ty, id, &[])) ||
                !cx.tcx
                .get_diagnostic_item(sym::Send)
                .map_or(false, |id| implements_trait(cx, arg_ty, id, &[]));

            then {
                span_lint_and_help(
                    cx,
                    ARC_WITH_NON_SEND_SYNC,
                    expr.span,
                    "usage of `Arc<T>` where `T` is not `Send` or `Sync`",
                    None,
                    "consider using `Rc<T>` instead or wrapping `T` in a std::sync type like \
                    `Mutex<T>`",
                );
            }
        }
    }
}
