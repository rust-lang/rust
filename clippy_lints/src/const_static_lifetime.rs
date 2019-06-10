use crate::redundant_static_lifetime::RedundantStaticLifetime;
use crate::utils::in_macro_or_desugar;
use rustc::lint::{EarlyContext, EarlyLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use syntax::ast::*;

declare_clippy_lint! {
    /// **What it does:** Checks for constants with an explicit `'static` lifetime.
    ///
    /// **Why is this bad?** Adding `'static` to every reference can create very
    /// complicated types.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// const FOO: &'static [(&'static str, &'static str, fn(&Bar) -> bool)] =
    /// &[...]
    /// ```
    /// This code can be rewritten as
    /// ```ignore
    ///  const FOO: &[(&str, &str, fn(&Bar) -> bool)] = &[...]
    /// ```
    pub CONST_STATIC_LIFETIME,
    style,
    "Using explicit `'static` lifetime for constants when elision rules would allow omitting them."
}

declare_lint_pass!(StaticConst => [CONST_STATIC_LIFETIME]);

impl StaticConst {
    // Recursively visit types
    fn visit_type(&mut self, ty: &Ty, cx: &EarlyContext<'_>) {
        let mut rsl =
            RedundantStaticLifetime::new(CONST_STATIC_LIFETIME, "Constants have by default a `'static` lifetime");
        rsl.visit_type(ty, cx)
    }
}

impl EarlyLintPass for StaticConst {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        if !in_macro_or_desugar(item.span) {
            // Match only constants...
            if let ItemKind::Const(ref var_type, _) = item.node {
                self.visit_type(var_type, cx);
            }
        }
    }

    // Don't check associated consts because `'static` cannot be elided on those (issue #2438)
}
