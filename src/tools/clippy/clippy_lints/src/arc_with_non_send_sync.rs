use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_from_proc_macro;
use clippy_utils::ty::{implements_trait, is_type_diagnostic_item};
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_middle::ty::GenericArgKind;
use rustc_middle::ty::print::with_forced_trimmed_paths;
use rustc_session::declare_lint_pass;
use rustc_span::symbol::sym;

declare_clippy_lint! {
    /// ### What it does.
    /// This lint warns when you use `Arc` with a type that does not implement `Send` or `Sync`.
    ///
    /// ### Why is this bad?
    /// `Arc<T>` is a thread-safe `Rc<T>` and guarantees that updates to the reference counter
    /// use atomic operations. To send an `Arc<T>` across thread boundaries and
    /// share ownership between multiple threads, `T` must be [both `Send` and `Sync`](https://doc.rust-lang.org/std/sync/struct.Arc.html#thread-safety),
    /// so either `T` should be made `Send + Sync` or an `Rc` should be used instead of an `Arc`.
    ///
    /// ### Example
    /// ```no_run
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
    "using `Arc` with a type that does not implement `Send` and `Sync`"
}
declare_lint_pass!(ArcWithNonSendSync => [ARC_WITH_NON_SEND_SYNC]);

impl<'tcx> LateLintPass<'tcx> for ArcWithNonSendSync {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let ExprKind::Call(func, [arg]) = expr.kind
            && let ExprKind::Path(QPath::TypeRelative(func_ty, func_name)) = func.kind
            && func_name.ident.name == sym::new
            && !expr.span.from_expansion()
            && is_type_diagnostic_item(cx, cx.typeck_results().node_type(func_ty.hir_id), sym::Arc)
            && let arg_ty = cx.typeck_results().expr_ty(arg)
            // make sure that the type is not and does not contain any type parameters
            && arg_ty.walk().all(|arg| {
                !matches!(arg.kind(), GenericArgKind::Type(ty) if matches!(ty.kind(), ty::Param(_)))
            })
            && let Some(send) = cx.tcx.get_diagnostic_item(sym::Send)
            && let Some(sync) = cx.tcx.lang_items().sync_trait()
            && let [is_send, is_sync] = [send, sync].map(|id| implements_trait(cx, arg_ty, id, &[]))
            && let reason = match (is_send, is_sync) {
                (false, false) => "neither `Send` nor `Sync`",
                (false, true) => "not `Send`",
                (true, false) => "not `Sync`",
                _ => return,
            }
            && !is_from_proc_macro(cx, expr)
        {
            span_lint_and_then(
                cx,
                ARC_WITH_NON_SEND_SYNC,
                expr.span,
                "usage of an `Arc` that is not `Send` and `Sync`",
                |diag| {
                    with_forced_trimmed_paths!({
                        diag.note(format!(
                            "`Arc<{arg_ty}>` is not `Send` and `Sync` as `{arg_ty}` is {reason}"
                        ));
                        diag.help("if the `Arc` will not used be across threads replace it with an `Rc`");
                        diag.help(format!(
                            "otherwise make `{arg_ty}` `Send` and `Sync` or consider a wrapper type such as `Mutex`"
                        ));
                    });
                },
            );
        }
    }
}
