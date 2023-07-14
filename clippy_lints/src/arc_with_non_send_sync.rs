use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::last_path_segment;
use clippy_utils::ty::{implements_trait, is_type_diagnostic_item};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_middle::ty::print::with_forced_trimmed_paths;
use rustc_middle::ty::GenericArgKind;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::sym;

declare_clippy_lint! {
    /// ### What it does.
    /// This lint warns when you use `Arc` with a type that does not implement `Send` or `Sync`.
    ///
    /// ### Why is this bad?
    /// `Arc<T>` is only `Send`/`Sync` when `T` is [both `Send` and `Sync`](https://doc.rust-lang.org/std/sync/struct.Arc.html#impl-Send-for-Arc%3CT%3E),
    /// either `T` should be made `Send + Sync` or an `Rc` should be used instead of an `Arc`
    ///
    /// ### Example
    /// ```rust
    /// # use std::cell::RefCell;
    /// # use std::sync::Arc;
    ///
    /// fn main() {
    ///     // This is fine, as `i32` implements `Send` and `Sync`.
    ///     let a = Arc::new(42);
    ///
    ///     // `RefCell` is `!Sync`, so either the `Arc` should be replaced with an `Rc`
    ///     // or the `RefCell` replaced with something like a `RwLock`
    ///     let b = Arc::new(RefCell::new(42));
    /// }
    /// ```
    #[clippy::version = "1.72.0"]
    pub ARC_WITH_NON_SEND_SYNC,
    suspicious,
    "using `Arc` with a type that does not implement `Send` or `Sync`"
}
declare_lint_pass!(ArcWithNonSendSync => [ARC_WITH_NON_SEND_SYNC]);

impl LateLintPass<'_> for ArcWithNonSendSync {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        let ty = cx.typeck_results().expr_ty(expr);
        if is_type_diagnostic_item(cx, ty, sym::Arc)
            && let ExprKind::Call(func, [arg]) = expr.kind
            && let ExprKind::Path(func_path) = func.kind
            && last_path_segment(&func_path).ident.name == sym::new
            && let arg_ty = cx.typeck_results().expr_ty(arg)
            // make sure that the type is not and does not contain any type parameters
            && arg_ty.walk().all(|arg| {
                !matches!(arg.unpack(), GenericArgKind::Type(ty) if matches!(ty.kind(), ty::Param(_)))
            })
            && let Some(send) = cx.tcx.get_diagnostic_item(sym::Send)
            && let Some(sync) = cx.tcx.lang_items().sync_trait()
            && let [is_send, is_sync] = [send, sync].map(|id| implements_trait(cx, arg_ty, id, &[]))
            && !(is_send && is_sync)
        {
            span_lint_and_then(
                cx,
                ARC_WITH_NON_SEND_SYNC,
                expr.span,
                "usage of an `Arc` that is not `Send` or `Sync`",
                |diag| with_forced_trimmed_paths!({
                    if !is_send {
                        diag.note(format!("the trait `Send` is not implemented for `{arg_ty}`"));
                    }
                    if !is_sync {
                        diag.note(format!("the trait `Sync` is not implemented for `{arg_ty}`"));
                    }

                    diag.note(format!("required for `{ty}` to implement `Send` and `Sync`"));

                    diag.help("consider using an `Rc` instead or wrapping the inner type with a `Mutex`");
                }
            ));
        }
    }
}
