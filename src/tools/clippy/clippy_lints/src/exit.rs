use clippy_utils::diagnostics::span_lint;
use clippy_utils::{is_entrypoint_fn, match_def_path, paths};
use if_chain::if_chain;
use rustc_hir::{Expr, ExprKind, Item, ItemKind, Node};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Detects calls to the `exit()` function which terminates the program.
    ///
    /// ### Why is this bad?
    /// Exit terminates the program at the location it is called. For unrecoverable
    /// errors `panics` should be used to provide a stacktrace and potentualy other
    /// information. A normal termination or one with an error code should happen in
    /// the main function.
    ///
    /// ### Example
    /// ```
    /// std::process::exit(0)
    /// ```
    ///
    /// Use instead:
    ///
    /// ```ignore
    /// // To provide a stacktrace and additional information
    /// panic!("message");
    ///
    /// // or a main method with a return
    /// fn main() -> Result<(), i32> {
    ///     Ok(())
    /// }
    /// ```
    #[clippy::version = "1.41.0"]
    pub EXIT,
    restriction,
    "detects `std::process::exit` calls"
}

declare_lint_pass!(Exit => [EXIT]);

impl<'tcx> LateLintPass<'tcx> for Exit {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::Call(path_expr, _args) = e.kind;
            if let ExprKind::Path(ref path) = path_expr.kind;
            if let Some(def_id) = cx.qpath_res(path, path_expr.hir_id).opt_def_id();
            if match_def_path(cx, def_id, &paths::EXIT);
            let parent = cx.tcx.hir().get_parent_item(e.hir_id).def_id;
            if let Some(Node::Item(Item{kind: ItemKind::Fn(..), ..})) = cx.tcx.hir().find_by_def_id(parent);
            // If the next item up is a function we check if it is an entry point
            // and only then emit a linter warning
            if !is_entrypoint_fn(cx, parent.to_def_id());
            then {
                span_lint(cx, EXIT, e.span, "usage of `process::exit`");
            }
        }
    }
}
