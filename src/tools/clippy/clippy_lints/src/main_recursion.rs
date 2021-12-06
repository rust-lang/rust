use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::source::snippet;
use clippy_utils::{is_entrypoint_fn, is_no_std_crate};
use if_chain::if_chain;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for recursion using the entrypoint.
    ///
    /// ### Why is this bad?
    /// Apart from special setups (which we could detect following attributes like #![no_std]),
    /// recursing into main() seems like an unintuitive antipattern we should be able to detect.
    ///
    /// ### Example
    /// ```no_run
    /// fn main() {
    ///     main();
    /// }
    /// ```
    #[clippy::version = "1.38.0"]
    pub MAIN_RECURSION,
    style,
    "recursion using the entrypoint"
}

#[derive(Default)]
pub struct MainRecursion {
    has_no_std_attr: bool,
}

impl_lint_pass!(MainRecursion => [MAIN_RECURSION]);

impl LateLintPass<'_> for MainRecursion {
    fn check_crate(&mut self, cx: &LateContext<'_>) {
        self.has_no_std_attr = is_no_std_crate(cx);
    }

    fn check_expr_post(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if self.has_no_std_attr {
            return;
        }

        if_chain! {
            if let ExprKind::Call(func, _) = &expr.kind;
            if let ExprKind::Path(QPath::Resolved(_, path)) = &func.kind;
            if let Some(def_id) = path.res.opt_def_id();
            if is_entrypoint_fn(cx, def_id);
            then {
                span_lint_and_help(
                    cx,
                    MAIN_RECURSION,
                    func.span,
                    &format!("recursing into entrypoint `{}`", snippet(cx, func.span, "main")),
                    None,
                    "consider using another function for this recursion"
                )
            }
        }
    }
}
