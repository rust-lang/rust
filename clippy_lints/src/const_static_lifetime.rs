use syntax::ast::{Item, ItemKind, TyKind, Ty};
use rustc::lint::{LintPass, EarlyLintPass, LintArray, EarlyContext};
use utils::{span_help_and_lint, in_macro};

/// **What it does:** Checks for constants with an explicit `'static` lifetime.
///
/// **Why is this bad?** Adding `'static` to every reference can create very complicated types.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
///  const FOO: &'static [(&'static str, &'static str, fn(&Bar) -> bool)] = &[..]
/// ```
/// This code can be rewritten as
/// ```rust
///  const FOO: &[(&str, &str, fn(&Bar) -> bool)] = &[...]
/// ```

declare_lint! {
    pub CONST_STATIC_LIFETIME, 
    Warn,
    "Using explicit `'static` lifetime for constants when elision rules would allow omitting them."
}

pub struct StaticConst;

impl LintPass for StaticConst {
    fn get_lints(&self) -> LintArray {
        lint_array!(CONST_STATIC_LIFETIME)
    }
}

impl StaticConst {
    // Recursively visit types
    fn visit_type(&mut self, ty: &Ty, cx: &EarlyContext) {
        match ty.node {
            // Be carefull of nested structures (arrays and tuples)
            TyKind::Array(ref ty, _) => {
                println!("array");
                self.visit_type(&*ty, cx);
            },
            TyKind::Tup(ref tup) => {
                for tup_ty in tup {
                    self.visit_type(&*tup_ty, cx);
                }
            },
            // This is what we are looking for !
            TyKind::Rptr(ref optional_lifetime, ref borrow_type) => {
                // Match the 'static lifetime
                if let Some(lifetime) = *optional_lifetime {
                    if let TyKind::Path(_, _) = borrow_type.ty.node {
                        // Verify that the path is a str
                        if lifetime.ident.name == "'static" {
                            span_help_and_lint(cx,
                                               CONST_STATIC_LIFETIME,
                                               lifetime.span,
                                               "Constants have by default a `'static` lifetime",
                                               "consider removing `'static`");
                        }
                    }
                }
                self.visit_type(&*borrow_type.ty, cx);
            },
            TyKind::Slice(ref ty) => {
                self.visit_type(&ty, cx);
            },
            _ => {},
        }
    }
}

impl EarlyLintPass for StaticConst {
    fn check_item(&mut self, cx: &EarlyContext, item: &Item) {
        if !in_macro(item.span) {
            // Match only constants...
            if let ItemKind::Const(ref var_type, _) = item.node {
                self.visit_type(var_type, cx);
            }
        }
    }
}
