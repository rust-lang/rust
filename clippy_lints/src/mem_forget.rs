use rustc::lint::*;
use rustc::hir::{Expr, ExprCall, ExprPath};
use utils::{match_def_path, paths, span_lint};

/// **What it does:** Checks for usage of `std::mem::forget(t)` where `t` is
/// `Drop`.
///
/// **Why is this bad?** `std::mem::forget(t)` prevents `t` from running its
/// destructor, possibly causing leaks.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// mem::forget(Rc::new(55)))
/// ```
declare_lint! {
    pub MEM_FORGET,
    Allow,
    "`mem::forget` usage on `Drop` types, likely to cause memory leaks"
}

pub struct MemForget;

impl LintPass for MemForget {
    fn get_lints(&self) -> LintArray {
        lint_array![MEM_FORGET]
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for MemForget {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        if let ExprCall(ref path_expr, ref args) = e.node {
            if let ExprPath(ref qpath) = path_expr.node {
                let def_id = cx.tables.qpath_def(qpath, path_expr.hir_id).def_id();
                if match_def_path(cx.tcx, def_id, &paths::MEM_FORGET) {
                    let forgot_ty = cx.tables.expr_ty(&args[0]);

                    if match forgot_ty.ty_adt_def() {
                        Some(def) => def.has_dtor(cx.tcx),
                        _ => false,
                    }
                    {
                        span_lint(cx, MEM_FORGET, e.span, "usage of mem::forget on Drop type");
                    }
                }
            }
        }
    }
}
