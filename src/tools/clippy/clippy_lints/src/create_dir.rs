use crate::utils::{match_def_path, paths, snippet, span_lint_and_sugg};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks usage of `std::fs::create_dir` and suggest using `std::fs::create_dir_all` instead.
    ///
    /// **Why is this bad?** Sometimes `std::fs::create_dir` is mistakenly chosen over `std::fs::create_dir_all`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// std::fs::create_dir("foo");
    /// ```
    /// Use instead:
    /// ```rust
    /// std::fs::create_dir_all("foo");
    /// ```
    pub CREATE_DIR,
    restriction,
    "calling `std::fs::create_dir` instead of `std::fs::create_dir_all`"
}

declare_lint_pass!(CreateDir => [CREATE_DIR]);

impl LateLintPass<'_> for CreateDir {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if_chain! {
            if let ExprKind::Call(ref func, ref args) = expr.kind;
            if let ExprKind::Path(ref path) = func.kind;
            if let Some(def_id) = cx.qpath_res(path, func.hir_id).opt_def_id();
            if match_def_path(cx, def_id, &paths::STD_FS_CREATE_DIR);
            then {
                span_lint_and_sugg(
                    cx,
                    CREATE_DIR,
                    expr.span,
                    "calling `std::fs::create_dir` where there may be a better way",
                    "consider calling `std::fs::create_dir_all` instead",
                    format!("create_dir_all({})", snippet(cx, args[0].span, "..")),
                    Applicability::MaybeIncorrect,
                )
            }
        }
    }
}
