use clippy_utils::diagnostics::span_lint;
use clippy_utils::{is_entrypoint_fn, match_def_path, paths};
use if_chain::if_chain;
use rustc_hir::{Expr, ExprKind, Item, ItemKind, Node};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// `exit()`  terminates the program and doesn't provide a
    /// stack trace.
    ///
    /// ### Why is this bad?
    /// Ideally a program is terminated by finishing
    /// the main function.
    ///
    /// ### Example
    /// ```ignore
    /// std::process::exit(0)
    /// ```
    #[clippy::version = "1.41.0"]
    pub EXIT,
    restriction,
    "`std::process::exit` is called, terminating the program"
}

declare_lint_pass!(Exit => [EXIT]);

impl<'tcx> LateLintPass<'tcx> for Exit {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::Call(path_expr, _args) = e.kind;
            if let ExprKind::Path(ref path) = path_expr.kind;
            if let Some(def_id) = cx.qpath_res(path, path_expr.hir_id).opt_def_id();
            if match_def_path(cx, def_id, &paths::EXIT);
            let parent = cx.tcx.hir().get_parent_item(e.hir_id);
            if let Some(Node::Item(Item{kind: ItemKind::Fn(..), ..})) = cx.tcx.hir().find(parent);
            // If the next item up is a function we check if it is an entry point
            // and only then emit a linter warning
            let def_id = cx.tcx.hir().local_def_id(parent);
            if !is_entrypoint_fn(cx, def_id.to_def_id());
            then {
                span_lint(cx, EXIT, e.span, "usage of `process::exit`");
            }
        }
    }
}
