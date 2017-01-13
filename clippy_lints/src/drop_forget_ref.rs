use rustc::lint::*;
use rustc::ty;
use rustc::hir::*;
use utils::{match_def_path, paths, span_note_and_lint};

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
/// std::mem::drop(&lock_guard) // Should have been drop(lock_guard), mutex still locked
/// operation_that_requires_mutex_to_be_unlocked();
/// ```
declare_lint! {
    pub DROP_REF,
    Warn,
    "calls to `std::mem::drop` with a reference instead of an owned value"
}

/// **What it does:** Checks for calls to `std::mem::forget` with a reference
/// instead of an owned value.
///
/// **Why is this bad?** Calling `forget` on a reference will only forget the
/// reference itself, which is a no-op. It will not forget the underlying referenced
/// value, which is likely what was intended.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// let x = Box::new(1);
/// std::mem::forget(&x) // Should have been forget(x), x will still be dropped
/// ```
declare_lint! {
    pub FORGET_REF,
    Warn,
    "calls to `std::mem::forget` with a reference instead of an owned value"
}

#[allow(missing_copy_implementations)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(DROP_REF, FORGET_REF)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if_let_chain!{[
            let ExprCall(ref path, ref args) = expr.node,
            let ExprPath(ref qpath) = path.node,
            args.len() == 1,
        ], {
            let def_id = cx.tables.qpath_def(qpath, path.id).def_id();
            let lint;
            let msg;
            if match_def_path(cx.tcx, def_id, &paths::DROP) {
                lint = DROP_REF;
                msg = "call to `std::mem::drop` with a reference argument. \
                       Dropping a reference does nothing";
            } else if match_def_path(cx.tcx, def_id, &paths::MEM_FORGET) {
                lint = FORGET_REF;
                msg = "call to `std::mem::forget` with a reference argument. \
                       Forgetting a reference does nothing";
            } else {
                return;
            }
            let arg = &args[0];
            let arg_ty = cx.tables.expr_ty(arg);
            if let ty::TyRef(..) = arg_ty.sty {
                span_note_and_lint(cx,
                                   lint,
                                   expr.span,
                                   msg,
                                   arg.span,
                                   &format!("argument has type {}", arg_ty.sty));
            }
        }}
    }
}
