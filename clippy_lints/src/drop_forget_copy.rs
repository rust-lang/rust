use rustc::lint::*;
use rustc::hir::*;
use utils::{match_def_path, paths, span_note_and_lint, is_copy};

const DROP_COPY_SUMMARY:&'static str = "calls to `std::mem::drop` with a value that implements Copy";
const FORGET_COPY_SUMMARY:&'static str = "calls to `std::mem::forget` with a value that implements Copy";

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
/// std::mem::drop(x) // A copy of x is passed to the function, leaving the original unaffected
/// ```
declare_lint! {
    pub DROP_COPY,
    Warn,
    DROP_COPY_SUMMARY
}

/// **What it does:** Checks for calls to `std::mem::forget` with a value that
/// derives the Copy trait
///
/// **Why is this bad?** Calling `std::mem::forget` [does nothing for types that
/// implement Copy](https://doc.rust-lang.org/std/mem/fn.drop.html) since the
/// value will be copied and moved into the function on invocation.
///
/// An alternative, but also valid, explanation is that Copy types do not implement
/// the Drop trait, which means they have no destructors. Without a destructor, there
/// is nothing for `std::mem::forget` to ignore.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// let x:i32 = 42;     // i32 implements Copy
/// std::mem::forget(x) // A copy of x is passed to the function, leaving the original unaffected
/// ```
declare_lint! {
    pub FORGET_COPY,
    Warn,
    FORGET_COPY_SUMMARY
}

#[allow(missing_copy_implementations)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(DROP_COPY, FORGET_COPY)
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
                lint = DROP_COPY;
                msg = DROP_COPY_SUMMARY.to_string() + ". Dropping a copy leaves the original intact.";
            } else if match_def_path(cx.tcx, def_id, &paths::MEM_FORGET) {
                lint = FORGET_COPY;
                msg = FORGET_COPY_SUMMARY.to_string() + ". Forgetting a copy leaves the original intact.";
            } else {
                return;
            }

            let arg = &args[0];
            let arg_ty = cx.tables.expr_ty(arg);
            if is_copy(cx, arg_ty, cx.tcx.hir.get_parent(arg.id)) {
                span_note_and_lint(cx,
                                   lint,
                                   expr.span,
                                   &msg,
                                   arg.span,
                                   &format!("argument has type {}", arg_ty.sty));
            }
        }}
    }
}
