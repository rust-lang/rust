use rustc::lint::*;
use rustc::hir::{Expr, ExprCall, ExprPath};
use utils::{match_def_path, paths, span_lint};

/// **What it does:** This lint checks for usage of `std::mem::forget(_)`.
///
/// **Why is this bad?** `std::mem::forget(t)` prevents `t` from running its destructor, possibly causing leaks
///
/// **Known problems:** None.
///
/// **Example:** `std::mem::forget(_))`
declare_lint! {
    pub MEM_FORGET,
    Allow,
    "`std::mem::forget` usage is likely to cause memory leaks"
}

pub struct MemForget;

impl LintPass for MemForget {
    fn get_lints(&self) -> LintArray {
        lint_array![MEM_FORGET]
    }
}

impl LateLintPass for MemForget {
    fn check_expr(&mut self, cx: &LateContext, e: &Expr) {
        if let ExprCall(ref path_expr, _) = e.node {
            if let ExprPath(None, _) = path_expr.node {
                let def_id = cx.tcx.def_map.borrow()[&path_expr.id].def_id();

                if match_def_path(cx, def_id, &paths::MEM_FORGET) {
                    span_lint(cx, MEM_FORGET, e.span, "usage of std::mem::forget");
                }
            }
        }
    }
}
