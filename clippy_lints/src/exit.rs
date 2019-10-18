use crate::utils::{match_def_path, paths, qpath_res, span_lint};
use if_chain::if_chain;
use rustc::hir::{Expr, ExprKind};
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** `exit()`  terminates the program and doesn't provide a
    /// stack trace.
    ///
    /// **Why is this bad?** Ideally a program is terminated by finishing
    /// the main function.
    ///
    /// **Known problems:** This can be valid code in main() to return
    /// errors
    ///
    /// **Example:**
    /// ```ignore
    /// std::process::exit(0)
    /// ```
    pub EXIT,
    restriction,
    "`std::process::exit` is called, terminating the program"
}

declare_lint_pass!(Exit => [EXIT]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Exit {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        if_chain! {
            if let ExprKind::Call(ref path_expr, ref _args) = e.kind;
            if let ExprKind::Path(ref path) = path_expr.kind;
            if let Some(def_id) = qpath_res(cx, path, path_expr.hir_id).opt_def_id();
            if match_def_path(cx, def_id, &paths::EXIT);
            then {
                span_lint(cx, EXIT, e.span, "usage of `process::exit`");
            }

        }
    }
}
