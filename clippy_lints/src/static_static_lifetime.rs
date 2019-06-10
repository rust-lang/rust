use crate::redundant_static_lifetime::RedundantStaticLifetime;
use crate::utils::in_macro_or_desugar;
use rustc::lint::{EarlyContext, EarlyLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use syntax::ast::*;

declare_clippy_lint! {
    /// **What it does:** Checks for statics with an explicit `'static` lifetime.
    ///
    /// **Why is this bad?** Adding `'static` to every reference can create very
    /// complicated types.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// static FOO: &'static [(&'static str, &'static str, fn(&Bar) -> bool)] =
    /// &[...]
    /// ```
    /// This code can be rewritten as
    /// ```ignore
    ///  static FOO: &[(&str, &str, fn(&Bar) -> bool)] = &[...]
    /// ```
    pub STATIC_STATIC_LIFETIME,
    style,
    "Using explicit `'static` lifetime for statics when elision rules would allow omitting them."
}

declare_lint_pass!(StaticStatic => [STATIC_STATIC_LIFETIME]);

impl StaticStatic {
    // Recursively visit types
    fn visit_type(&mut self, ty: &Ty, cx: &EarlyContext<'_>) {
        let mut rsl =
            RedundantStaticLifetime::new(STATIC_STATIC_LIFETIME, "Statics have by default a `'static` lifetime");
        rsl.visit_type(ty, cx)
    }
}

impl EarlyLintPass for StaticStatic {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        if !in_macro_or_desugar(item.span) {
            // Match only statics...
            if let ItemKind::Static(ref var_type, _, _) = item.node {
                self.visit_type(var_type, cx);
            }
        }
    }
}
