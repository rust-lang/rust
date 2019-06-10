use crate::utils::{snippet, span_lint_and_then};
use rustc::lint::{EarlyContext, Lint};
use rustc_errors::Applicability;
use syntax::ast::*;

pub struct RedundantStaticLifetime {
    lint: &'static Lint,
    reason: &'static str,
}

impl RedundantStaticLifetime {
    pub fn new(lint: &'static Lint, reason: &'static str) -> Self {
        Self { lint, reason }
    }
    // Recursively visit types
    pub fn visit_type(&mut self, ty: &Ty, cx: &EarlyContext<'_>) {
        match ty.node {
            // Be careful of nested structures (arrays and tuples)
            TyKind::Array(ref ty, _) => {
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
                    match borrow_type.ty.node {
                        TyKind::Path(..) | TyKind::Slice(..) | TyKind::Array(..) | TyKind::Tup(..) => {
                            if lifetime.ident.name == syntax::symbol::kw::StaticLifetime {
                                let snip = snippet(cx, borrow_type.ty.span, "<type>");
                                let sugg = format!("&{}", snip);
                                span_lint_and_then(cx, self.lint, lifetime.ident.span, self.reason, |db| {
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
                self.visit_type(&*borrow_type.ty, cx);
            },
            TyKind::Slice(ref ty) => {
                self.visit_type(ty, cx);
            },
            _ => {},
        }
    }
}
