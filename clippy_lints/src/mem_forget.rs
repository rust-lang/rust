use clippy_utils::diagnostics::span_lint;
use clippy_utils::{match_def_path, paths};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `std::mem::forget(t)` where `t` is
    /// `Drop`.
    ///
    /// ### Why is this bad?
    /// `std::mem::forget(t)` prevents `t` from running its
    /// destructor, possibly causing leaks.
    ///
    /// ### Example
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

impl<'tcx> LateLintPass<'tcx> for MemForget {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if let ExprKind::Call(path_expr, [ref first_arg, ..]) = e.kind {
            if let ExprKind::Path(ref qpath) = path_expr.kind {
                if let Some(def_id) = cx.qpath_res(qpath, path_expr.hir_id).opt_def_id() {
                    if match_def_path(cx, def_id, &paths::MEM_FORGET) {
                        let forgot_ty = cx.typeck_results().expr_ty(first_arg);

                        if forgot_ty.ty_adt_def().map_or(false, |def| def.has_dtor(cx.tcx)) {
                            span_lint(cx, MEM_FORGET, e.span, "usage of `mem::forget` on `Drop` type");
                        }
                    }
                }
            }
        }
    }
}
