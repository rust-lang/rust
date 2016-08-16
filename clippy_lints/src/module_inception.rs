use rustc::lint::*;
use syntax::ast::*;
use utils::span_lint;

/// **What it does:** Checks for modules that have the same name as their parent module
///
/// **Why is this bad?** A typical beginner mistake is to have `mod foo;` and again `mod foo { .. }` in `foo.rs`
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
