use crate::utils::{is_type_diagnostic_item, span_lint_and_help};
use if_chain::if_chain;
use rustc_hir::{Expr, ExprKind, Mutability};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for `&mut Mutex::lock` calls
    ///
    /// **Why is this bad?** `Mutex::lock` is less efficient than
    /// calling `Mutex::get_mut`
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// use std::sync::{Arc, Mutex};
    ///
    /// let mut value_rc = Arc::new(Mutex::new(42_u8));
    /// let value_mutex = Arc::get_mut(&mut value_rc).unwrap();
    ///
    /// let value = value_mutex.lock().unwrap();
    /// do_stuff(value);
    /// ```
    /// Use instead:
    /// ```rust
    /// use std::sync::{Arc, Mutex};
    ///
    /// let mut value_rc = Arc::new(Mutex::new(42_u8));
    /// let value_mutex = Arc::get_mut(&mut value_rc).unwrap();
    ///
    /// let value = value_mutex.get_mut().unwrap();
    /// do_stuff(value);
    /// ```
    pub MUT_MUTEX_LOCK,
    correctness,
    "`&mut Mutex::lock` does unnecessary locking"
}

declare_lint_pass!(MutMutexLock => [MUT_MUTEX_LOCK]);

impl<'tcx> LateLintPass<'tcx> for MutMutexLock {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, ex: &'tcx Expr<'tcx>) {
        if_chain! {
            if is_mut_mutex_lock_call(cx, ex).is_some();
            then {
                span_lint_and_help(
                    cx,
                    MUT_MUTEX_LOCK,
                    ex.span,
                    "calling `&mut Mutex::lock` unnecessarily locks an exclusive (mutable) reference",
                    None,
                    "use `&mut Mutex::get_mut` instead",
                );
            }
        }
    }
}

fn is_mut_mutex_lock_call<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<&'tcx Expr<'tcx>> {
    if_chain! {
        if let ExprKind::MethodCall(path, _span, args, _) = &expr.kind;
        if path.ident.name == sym!(lock);
        let ty = cx.typeck_results().expr_ty(&args[0]);
        if let ty::Ref(_, inner_ty, Mutability::Mut) = ty.kind();
        if is_type_diagnostic_item(cx, inner_ty, sym!(mutex_type));
        then {
            Some(&args[0])
        } else {
            None
        }
    }
}
