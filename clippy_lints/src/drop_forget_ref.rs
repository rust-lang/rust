use rustc::lint::*;
use rustc::ty;
use rustc::hir::*;
use utils::{is_copy, match_def_path, opt_def_id, paths, span_note_and_lint};

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
/// ```rust
/// let mut lock_guard = mutex.lock();
/// std::mem::drop(&lock_guard) // Should have been drop(lock_guard), mutex
/// // still locked
/// operation_that_requires_mutex_to_be_unlocked();
/// ```
declare_clippy_lint! {
    pub DROP_REF,
    correctness,
    "calls to `std::mem::drop` with a reference instead of an owned value"
}

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
declare_clippy_lint! {
    pub FORGET_REF,
    correctness,
    "calls to `std::mem::forget` with a reference instead of an owned value"
}

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
/// let x:i32 = 42;   // i32 implements Copy
/// std::mem::drop(x) // A copy of x is passed to the function, leaving the
/// // original unaffected
/// ```
declare_clippy_lint! {
    pub DROP_COPY,
    correctness,
    "calls to `std::mem::drop` with a value that implements Copy"
}

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
/// let x:i32 = 42;     // i32 implements Copy
/// std::mem::forget(x) // A copy of x is passed to the function, leaving the
/// // original unaffected
/// ```
declare_clippy_lint! {
    pub FORGET_COPY,
    correctness,
    "calls to `std::mem::forget` with a value that implements Copy"
}

const DROP_REF_SUMMARY: &str = "calls to `std::mem::drop` with a reference instead of an owned value. \
                                Dropping a reference does nothing.";
const FORGET_REF_SUMMARY: &str = "calls to `std::mem::forget` with a reference instead of an owned value. \
                                  Forgetting a reference does nothing.";
const DROP_COPY_SUMMARY: &str = "calls to `std::mem::drop` with a value that implements Copy. \
                                 Dropping a copy leaves the original intact.";
const FORGET_COPY_SUMMARY: &str = "calls to `std::mem::forget` with a value that implements Copy. \
                                   Forgetting a copy leaves the original intact.";

#[allow(missing_copy_implementations)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(DROP_REF, FORGET_REF, DROP_COPY, FORGET_COPY)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if_chain! {
            if let ExprCall(ref path, ref args) = expr.node;
            if let ExprPath(ref qpath) = path.node;
            if args.len() == 1;
            if let Some(def_id) = opt_def_id(cx.tables.qpath_def(qpath, path.hir_id));
            then {
                let lint;
                let msg;
                let arg = &args[0];
                let arg_ty = cx.tables.expr_ty(arg);

                if let ty::TyRef(..) = arg_ty.sty {
                    if match_def_path(cx.tcx, def_id, &paths::DROP) {
                        lint = DROP_REF;
                        msg = DROP_REF_SUMMARY.to_string();
                    } else if match_def_path(cx.tcx, def_id, &paths::MEM_FORGET) {
                        lint = FORGET_REF;
                        msg = FORGET_REF_SUMMARY.to_string();
                    } else {
                        return;
                    }
                    span_note_and_lint(cx,
                                       lint,
                                       expr.span,
                                       &msg,
                                       arg.span,
                                       &format!("argument has type {}", arg_ty));
                } else if is_copy(cx, arg_ty) {
                    if match_def_path(cx.tcx, def_id, &paths::DROP) {
                        lint = DROP_COPY;
                        msg = DROP_COPY_SUMMARY.to_string();
                    } else if match_def_path(cx.tcx, def_id, &paths::MEM_FORGET) {
                        lint = FORGET_COPY;
                        msg = FORGET_COPY_SUMMARY.to_string();
                    } else {
                        return;
                    }
                    span_note_and_lint(cx,
                                       lint,
                                       expr.span,
                                       &msg,
                                       arg.span,
                                       &format!("argument has type {}", arg_ty));
                }
            }
        }
    }
}
