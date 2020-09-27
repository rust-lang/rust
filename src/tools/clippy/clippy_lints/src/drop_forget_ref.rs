use crate::utils::{is_copy, match_def_path, paths, qpath_res, span_lint_and_note};
use if_chain::if_chain;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for calls to `std::mem::drop` with a reference
    /// instead of an owned value.
    ///
    /// **Why is this bad?** Calling `drop` on a reference will only drop the
    /// reference itself, which is a no-op. It will not call the `drop` method (from
    /// the `Drop` trait implementation) on the underlying referenced value, which
    /// is likely what was intended.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// let mut lock_guard = mutex.lock();
    /// std::mem::drop(&lock_guard) // Should have been drop(lock_guard), mutex
    /// // still locked
    /// operation_that_requires_mutex_to_be_unlocked();
    /// ```
    pub DROP_REF,
    correctness,
    "calls to `std::mem::drop` with a reference instead of an owned value"
}

declare_clippy_lint! {
    /// **What it does:** Checks for calls to `std::mem::forget` with a reference
    /// instead of an owned value.
    ///
    /// **Why is this bad?** Calling `forget` on a reference will only forget the
    /// reference itself, which is a no-op. It will not forget the underlying
    /// referenced
    /// value, which is likely what was intended.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let x = Box::new(1);
    /// std::mem::forget(&x) // Should have been forget(x), x will still be dropped
    /// ```
    pub FORGET_REF,
    correctness,
    "calls to `std::mem::forget` with a reference instead of an owned value"
}

declare_clippy_lint! {
    /// **What it does:** Checks for calls to `std::mem::drop` with a value
    /// that derives the Copy trait
    ///
    /// **Why is this bad?** Calling `std::mem::drop` [does nothing for types that
    /// implement Copy](https://doc.rust-lang.org/std/mem/fn.drop.html), since the
    /// value will be copied and moved into the function on invocation.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let x: i32 = 42; // i32 implements Copy
    /// std::mem::drop(x) // A copy of x is passed to the function, leaving the
    ///                   // original unaffected
    /// ```
    pub DROP_COPY,
    correctness,
    "calls to `std::mem::drop` with a value that implements Copy"
}

declare_clippy_lint! {
    /// **What it does:** Checks for calls to `std::mem::forget` with a value that
    /// derives the Copy trait
    ///
    /// **Why is this bad?** Calling `std::mem::forget` [does nothing for types that
    /// implement Copy](https://doc.rust-lang.org/std/mem/fn.drop.html) since the
    /// value will be copied and moved into the function on invocation.
    ///
    /// An alternative, but also valid, explanation is that Copy types do not
    /// implement
    /// the Drop trait, which means they have no destructors. Without a destructor,
    /// there
    /// is nothing for `std::mem::forget` to ignore.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let x: i32 = 42; // i32 implements Copy
    /// std::mem::forget(x) // A copy of x is passed to the function, leaving the
    ///                     // original unaffected
    /// ```
    pub FORGET_COPY,
    correctness,
    "calls to `std::mem::forget` with a value that implements Copy"
}

const DROP_REF_SUMMARY: &str = "calls to `std::mem::drop` with a reference instead of an owned value. \
                                Dropping a reference does nothing.";
const FORGET_REF_SUMMARY: &str = "calls to `std::mem::forget` with a reference instead of an owned value. \
                                  Forgetting a reference does nothing.";
const DROP_COPY_SUMMARY: &str = "calls to `std::mem::drop` with a value that implements `Copy`. \
                                 Dropping a copy leaves the original intact.";
const FORGET_COPY_SUMMARY: &str = "calls to `std::mem::forget` with a value that implements `Copy`. \
                                   Forgetting a copy leaves the original intact.";

declare_lint_pass!(DropForgetRef => [DROP_REF, FORGET_REF, DROP_COPY, FORGET_COPY]);

impl<'tcx> LateLintPass<'tcx> for DropForgetRef {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::Call(ref path, ref args) = expr.kind;
            if let ExprKind::Path(ref qpath) = path.kind;
            if args.len() == 1;
            if let Some(def_id) = qpath_res(cx, qpath, path.hir_id).opt_def_id();
            then {
                let lint;
                let msg;
                let arg = &args[0];
                let arg_ty = cx.typeck_results().expr_ty(arg);

                if let ty::Ref(..) = arg_ty.kind() {
                    if match_def_path(cx, def_id, &paths::DROP) {
                        lint = DROP_REF;
                        msg = DROP_REF_SUMMARY.to_string();
                    } else if match_def_path(cx, def_id, &paths::MEM_FORGET) {
                        lint = FORGET_REF;
                        msg = FORGET_REF_SUMMARY.to_string();
                    } else {
                        return;
                    }
                    span_lint_and_note(cx,
                                       lint,
                                       expr.span,
                                       &msg,
                                       Some(arg.span),
                                       &format!("argument has type `{}`", arg_ty));
                } else if is_copy(cx, arg_ty) {
                    if match_def_path(cx, def_id, &paths::DROP) {
                        lint = DROP_COPY;
                        msg = DROP_COPY_SUMMARY.to_string();
                    } else if match_def_path(cx, def_id, &paths::MEM_FORGET) {
                        lint = FORGET_COPY;
                        msg = FORGET_COPY_SUMMARY.to_string();
                    } else {
                        return;
                    }
                    span_lint_and_note(cx,
                                       lint,
                                       expr.span,
                                       &msg,
                                       Some(arg.span),
                                       &format!("argument has type {}", arg_ty));
                }
            }
        }
    }
}
