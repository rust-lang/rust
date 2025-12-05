use clippy_utils::diagnostics::span_lint;
use rustc_hir::{Expr, ExprKind, Item, ItemKind, OwnerNode};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Detects calls to the `exit()` function that are not in the `main` function. Calls to `exit()`
    /// immediately terminate the program.
    ///
    /// ### Why restrict this?
    /// `exit()` immediately terminates the program with no information other than an exit code.
    /// This provides no means to troubleshoot a problem, and may be an unexpected side effect.
    ///
    /// Codebases may use this lint to require that all exits are performed either by panicking
    /// (which produces a message, a code location, and optionally a backtrace)
    /// or by calling `exit()` from `main()` (which is a single place to look).
    ///
    /// ### Good example
    /// ```no_run
    /// fn main() {
    ///     std::process::exit(0);
    /// }
    /// ```
    ///
    /// ### Bad example
    /// ```no_run
    /// fn main() {
    ///     other_function();
    /// }
    ///
    /// fn other_function() {
    ///     std::process::exit(0);
    /// }
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
    "detects `std::process::exit` calls outside of `main`"
}

declare_lint_pass!(Exit => [EXIT]);

impl<'tcx> LateLintPass<'tcx> for Exit {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if let ExprKind::Call(path_expr, [_]) = e.kind
            && let ExprKind::Path(ref path) = path_expr.kind
            && let Some(def_id) = cx.qpath_res(path, path_expr.hir_id).opt_def_id()
            && cx.tcx.is_diagnostic_item(sym::process_exit, def_id)
            && let parent = cx.tcx.hir_get_parent_item(e.hir_id)
            && let OwnerNode::Item(Item{kind: ItemKind::Fn{ ident, .. }, ..}) = cx.tcx.hir_owner_node(parent)
            // If the next item up is a function we check if it isn't named "main"
            // and only then emit a linter warning

            // if you instead check for the parent of the `exit()` call being the entrypoint function, as this worked before,
            // in compilation contexts like --all-targets (which include --tests), you get false positives
            // because in a test context, main is not the entrypoint function
            && ident.name != sym::main
        {
            span_lint(cx, EXIT, e.span, "usage of `process::exit`");
        }
    }
}
