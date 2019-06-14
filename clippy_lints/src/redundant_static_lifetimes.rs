use crate::utils::{in_macro_or_desugar, snippet, span_lint_and_then};
use rustc::lint::{EarlyContext, EarlyLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use syntax::ast::*;

declare_clippy_lint! {
    /// **What it does:** Checks for constants and statics with an explicit `'static` lifetime.
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
    /// static FOO: &'static [(&'static str, &'static str, fn(&Bar) -> bool)] =
    /// &[...]
    /// ```
    /// This code can be rewritten as
    /// ```ignore
    ///  const FOO: &[(&str, &str, fn(&Bar) -> bool)] = &[...]
    ///  static FOO: &[(&str, &str, fn(&Bar) -> bool)] = &[...]
    /// ```
    pub REDUNDANT_STATIC_LIFETIMES,
    style,
    "Using explicit `'static` lifetime for constants or statics when elision rules would allow omitting them."
}

declare_lint_pass!(RedundantStaticLifetimes => [REDUNDANT_STATIC_LIFETIMES]);

impl RedundantStaticLifetimes {
    // Recursively visit types
    fn visit_type(&mut self, ty: &Ty, cx: &EarlyContext<'_>, reason: &str) {
        match ty.node {
            // Be careful of nested structures (arrays and tuples)
            TyKind::Array(ref ty, _) => {
                self.visit_type(&*ty, cx, reason);
            },
            TyKind::Tup(ref tup) => {
                for tup_ty in tup {
                    self.visit_type(&*tup_ty, cx, reason);
                }
            },
            // This is what we are looking for !
            TyKind::Rptr(ref optional_lifetime, ref borrow_type) => {
                // Match the 'static lifetime
                if let Some(lifetime) = *optional_lifetime {
                    match borrow_type.ty.node {
                        TyKind::Path(..) | TyKind::Slice(..) | TyKind::Array(..) | TyKind::Tup(..) => {
                            if lifetime.ident.name == syntax::symbol::kw::StaticLifetime {
                                let snip = snippet(cx, borrow_type.ty.span, "<type>");
                                let sugg = format!("&{}", snip);
                                span_lint_and_then(cx, REDUNDANT_STATIC_LIFETIMES, lifetime.ident.span, reason, |db| {
                                    db.span_suggestion(
                                        ty.span,
                                        "consider removing `'static`",
                                        sugg,
                                        Applicability::MachineApplicable, //snippet
                                    );
                                });
                            }
                        },
                        _ => {},
                    }
                }
                self.visit_type(&*borrow_type.ty, cx, reason);
            },
            TyKind::Slice(ref ty) => {
                self.visit_type(ty, cx, reason);
            },
            _ => {},
        }
    }
}

impl EarlyLintPass for RedundantStaticLifetimes {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        if !in_macro_or_desugar(item.span) {
            if let ItemKind::Const(ref var_type, _) = item.node {
                self.visit_type(var_type, cx, "Constants have by default a `'static` lifetime");
                // Don't check associated consts because `'static` cannot be elided on those (issue #2438)
            }

            if let ItemKind::Static(ref var_type, _, _) = item.node {
                self.visit_type(var_type, cx, "Statics have by default a `'static` lifetime");
            }
        }
    }
}
