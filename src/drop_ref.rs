use rustc::lint::*;
use rustc::ty;
use rustc::hir::*;
use syntax::codemap::Span;
use utils::{match_def_path, paths, span_note_and_lint};

/// **What it does:** This lint checks for calls to `std::mem::drop` with a reference instead of an owned value.
///
/// **Why is this bad?** Calling `drop` on a reference will only drop the reference itself, which is a no-op. It will not call the `drop` method (from the `Drop` trait implementation) on the underlying referenced value, which is likely what was intended.
///
/// **Known problems:** None
///
/// **Example:**
/// ```rust
/// let mut lock_guard = mutex.lock();
/// std::mem::drop(&lock_guard) // Should have been drop(lock_guard), mutex still locked
/// operation_that_requires_mutex_to_be_unlocked();
/// ```
declare_lint! {
    pub DROP_REF, Warn,
    "call to `std::mem::drop` with a reference instead of an owned value, \
    which will not call the `Drop::drop` method on the underlying value"
}

#[allow(missing_copy_implementations)]
pub struct DropRefPass;

impl LintPass for DropRefPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(DROP_REF)
    }
}

impl LateLintPass for DropRefPass {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if let ExprCall(ref path, ref args) = expr.node {
            if let ExprPath(None, _) = path.node {
                let def_id = cx.tcx.def_map.borrow()[&path.id].def_id();
                if match_def_path(cx, def_id, &paths::DROP) {
                    if args.len() != 1 {
                        return;
                    }
                    check_drop_arg(cx, expr.span, &*args[0]);
                }
            }
        }
    }
}

fn check_drop_arg(cx: &LateContext, call_span: Span, arg: &Expr) {
    let arg_ty = cx.tcx.expr_ty(arg);
    if let ty::TyRef(..) = arg_ty.sty {
        span_note_and_lint(cx,
                           DROP_REF,
                           call_span,
                           "call to `std::mem::drop` with a reference argument. \
                           Dropping a reference does nothing",
                           arg.span,
                           &format!("argument has type {}", arg_ty.sty));
    }
}
