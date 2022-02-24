use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Mutability};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `&mut Mutex::lock` calls
    ///
    /// ### Why is this bad?
    /// `Mutex::lock` is less efficient than
    /// calling `Mutex::get_mut`. In addition you also have a statically
    /// guarantee that the mutex isn't locked, instead of just a runtime
    /// guarantee.
    ///
    /// ### Example
    /// ```rust
    /// use std::sync::{Arc, Mutex};
    ///
    /// let mut value_rc = Arc::new(Mutex::new(42_u8));
    /// let value_mutex = Arc::get_mut(&mut value_rc).unwrap();
    ///
    /// let mut value = value_mutex.lock().unwrap();
    /// *value += 1;
    /// ```
    /// Use instead:
    /// ```rust
    /// use std::sync::{Arc, Mutex};
    ///
    /// let mut value_rc = Arc::new(Mutex::new(42_u8));
    /// let value_mutex = Arc::get_mut(&mut value_rc).unwrap();
    ///
    /// let value = value_mutex.get_mut().unwrap();
    /// *value += 1;
    /// ```
    #[clippy::version = "1.49.0"]
    pub MUT_MUTEX_LOCK,
    style,
    "`&mut Mutex::lock` does unnecessary locking"
}

declare_lint_pass!(MutMutexLock => [MUT_MUTEX_LOCK]);

impl<'tcx> LateLintPass<'tcx> for MutMutexLock {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, ex: &'tcx Expr<'tcx>) {
        if_chain! {
            if let ExprKind::MethodCall(path, [self_arg, ..], _) = &ex.kind;
            if path.ident.name == sym!(lock);
            let ty = cx.typeck_results().expr_ty(self_arg);
            if let ty::Ref(_, inner_ty, Mutability::Mut) = ty.kind();
            if is_type_diagnostic_item(cx, *inner_ty, sym::Mutex);
            then {
                span_lint_and_sugg(
                    cx,
                    MUT_MUTEX_LOCK,
                    path.ident.span,
                    "calling `&mut Mutex::lock` unnecessarily locks an exclusive (mutable) reference",
                    "change this to",
                    "get_mut".to_owned(),
                    Applicability::MaybeIncorrect,
                );
            }
        }
    }
}
