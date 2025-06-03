use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::visitors::for_each_expr_without_closures;
use clippy_utils::{eq_expr_value, higher, sym};
use core::ops::ControlFlow;
use rustc_errors::Diag;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::edition::Edition::Edition2024;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `Mutex::lock` calls in `if let` expression
    /// with lock calls in any of the else blocks.
    ///
    /// ### Disabled starting in Edition 2024
    /// This lint is effectively disabled starting in
    /// Edition 2024 as `if let ... else` scoping was reworked
    /// such that this is no longer an issue. See
    /// [Proposal: stabilize if_let_rescope for Edition 2024](https://github.com/rust-lang/rust/issues/131154)
    ///
    /// ### Why is this bad?
    /// The Mutex lock remains held for the whole
    /// `if let ... else` block and deadlocks.
    ///
    /// ### Example
    /// ```rust,ignore
    /// if let Ok(thing) = mutex.lock() {
    ///     do_thing();
    /// } else {
    ///     mutex.lock();
    /// }
    /// ```
    /// Should be written
    /// ```rust,ignore
    /// let locked = mutex.lock();
    /// if let Ok(thing) = locked {
    ///     do_thing(thing);
    /// } else {
    ///     use_locked(locked);
    /// }
    /// ```
    #[clippy::version = "1.45.0"]
    pub IF_LET_MUTEX,
    correctness,
    "locking a `Mutex` in an `if let` block can cause deadlocks"
}

declare_lint_pass!(IfLetMutex => [IF_LET_MUTEX]);

impl<'tcx> LateLintPass<'tcx> for IfLetMutex {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if cx.tcx.sess.edition() >= Edition2024 {
            return;
        }

        if let Some(higher::IfLet {
            let_expr,
            if_then,
            if_else: Some(if_else),
            ..
        }) = higher::IfLet::hir(cx, expr)
            && let Some(op_mutex) = for_each_expr_without_closures(let_expr, |e| mutex_lock_call(cx, e, None))
            && let Some(arm_mutex) =
                for_each_expr_without_closures((if_then, if_else), |e| mutex_lock_call(cx, e, Some(op_mutex)))
        {
            let diag = |diag: &mut Diag<'_, ()>| {
                diag.span_label(
                    op_mutex.span,
                    "this Mutex will remain locked for the entire `if let`-block...",
                );
                diag.span_label(
                    arm_mutex.span,
                    "... and is tried to lock again here, which will always deadlock.",
                );
                diag.help("move the lock call outside of the `if let ...` expression");
            };
            span_lint_and_then(
                cx,
                IF_LET_MUTEX,
                expr.span,
                "calling `Mutex::lock` inside the scope of another `Mutex::lock` causes a deadlock",
                diag,
            );
        }
    }
}

fn mutex_lock_call<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    op_mutex: Option<&'tcx Expr<'_>>,
) -> ControlFlow<&'tcx Expr<'tcx>> {
    if let ExprKind::MethodCall(path, self_arg, [], _) = &expr.kind
        && path.ident.name == sym::lock
        && let ty = cx.typeck_results().expr_ty(self_arg).peel_refs()
        && is_type_diagnostic_item(cx, ty, sym::Mutex)
        && op_mutex.is_none_or(|op| eq_expr_value(cx, self_arg, op))
    {
        ControlFlow::Break(self_arg)
    } else {
        ControlFlow::Continue(())
    }
}
