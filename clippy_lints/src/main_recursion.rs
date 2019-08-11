use rustc::hir::{Crate, Expr, ExprKind, QPath};
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_tool_lint, impl_lint_pass};
use syntax::symbol::sym;

use crate::utils::{is_entrypoint_fn, snippet, span_help_and_lint};
use if_chain::if_chain;

declare_clippy_lint! {
    /// **What it does:** Checks for recursion using the entrypoint.
    ///
    /// **Why is this bad?** Apart from special setups (which we could detect following attributes like #![no_std]),
    /// recursing into main() seems like an unintuitive antipattern we should be able to detect.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```no_run
    /// fn main() {
    ///     main();
    /// }
    /// ```
    pub MAIN_RECURSION,
    style,
    "recursion using the entrypoint"
}

#[derive(Default)]
pub struct MainRecursion {
    has_no_std_attr: bool,
}

impl_lint_pass!(MainRecursion => [MAIN_RECURSION]);

impl LateLintPass<'_, '_> for MainRecursion {
    fn check_crate(&mut self, _: &LateContext<'_, '_>, krate: &Crate) {
        self.has_no_std_attr = krate.attrs.iter().any(|attr| attr.path == sym::no_std);
    }

    fn check_expr_post(&mut self, cx: &LateContext<'_, '_>, expr: &Expr) {
        if self.has_no_std_attr {
            return;
        }

        if_chain! {
            if let ExprKind::Call(func, _) = &expr.node;
            if let ExprKind::Path(path) = &func.node;
            if let QPath::Resolved(_, path) = &path;
            if let Some(def_id) = path.res.opt_def_id();
            if is_entrypoint_fn(cx, def_id);
            then {
                span_help_and_lint(
                    cx,
                    MAIN_RECURSION,
                    func.span,
                    &format!("recursing into entrypoint `{}`", snippet(cx, func.span, "main")),
                    "consider using another function for this recursion"
                )
            }
        }
    }
}
