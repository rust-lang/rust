use rustc::lint::*;
use syntax::ast::*;
use utils::span_lint;

/// **What it does:** Checks for modules that have the same name as their parent module
///
/// **Why is this bad?** A typical beginner mistake is to have `mod foo;` and again `mod foo { .. }` in `foo.rs`.
///                      The expectation is that items inside the inner `mod foo { .. }` are then available
///                      through `foo::x`, but they are only available through `foo::foo::x`.
///                      If this is done on purpose, it would be better to choose a more representative module name.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// // lib.rs
/// mod foo;
/// // foo.rs
/// mod foo {
///     ...
/// }
/// ```
declare_lint! {
    pub MODULE_INCEPTION,
    Warn,
    "modules that have the same name as their parent module"
}

pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array![MODULE_INCEPTION]
    }
}

impl EarlyLintPass for Pass {
    fn check_item(&mut self, cx: &EarlyContext, item: &Item) {
        if let ItemKind::Mod(ref module) = item.node {
            for sub_item in &module.items {
                if let ItemKind::Mod(_) = sub_item.node {
                    if item.ident == sub_item.ident {
                        span_lint(cx, MODULE_INCEPTION, sub_item.span,
                                  "module has the same name as its containing module");
                    }
                }
            }
        }
    }
}
