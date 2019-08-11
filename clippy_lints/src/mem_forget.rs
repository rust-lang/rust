use crate::utils::{match_def_path, paths, span_lint};
use rustc::hir::{Expr, ExprKind};
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
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
    /// # use std::mem;
    /// # use std::rc::Rc;
    /// mem::forget(Rc::new(55))
    /// ```
    pub MEM_FORGET,
    restriction,
    "`mem::forget` usage on `Drop` types, likely to cause memory leaks"
}

declare_lint_pass!(MemForget => [MEM_FORGET]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for MemForget {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        if let ExprKind::Call(ref path_expr, ref args) = e.node {
            if let ExprKind::Path(ref qpath) = path_expr.node {
                if let Some(def_id) = cx.tables.qpath_res(qpath, path_expr.hir_id).opt_def_id() {
                    if match_def_path(cx, def_id, &paths::MEM_FORGET) {
                        let forgot_ty = cx.tables.expr_ty(&args[0]);

                        if forgot_ty.ty_adt_def().map_or(false, |def| def.has_dtor(cx.tcx)) {
                            span_lint(cx, MEM_FORGET, e.span, "usage of mem::forget on Drop type");
                        }
                    }
                }
            }
        }
    }
}
