use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet;
use rustc_ast::ast::{Item, ItemKind, Ty, TyKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for constants and statics with an explicit `'static` lifetime.
    ///
    /// ### Why is this bad?
    /// Adding `'static` to every reference can create very
    /// complicated types.
    ///
    /// ### Example
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
    #[clippy::version = "1.37.0"]
    pub REDUNDANT_STATIC_LIFETIMES,
    style,
    "Using explicit `'static` lifetime for constants or statics when elision rules would allow omitting them."
}

pub struct RedundantStaticLifetimes {
    msrv: Msrv,
}

impl RedundantStaticLifetimes {
    #[must_use]
    pub fn new(msrv: Msrv) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(RedundantStaticLifetimes => [REDUNDANT_STATIC_LIFETIMES]);

impl RedundantStaticLifetimes {
    // Recursively visit types
    fn visit_type(ty: &Ty, cx: &EarlyContext<'_>, reason: &str) {
        match ty.kind {
            // Be careful of nested structures (arrays and tuples)
            TyKind::Array(ref ty, _) | TyKind::Slice(ref ty) => {
                Self::visit_type(ty, cx, reason);
            },
            TyKind::Tup(ref tup) => {
                for tup_ty in tup {
                    Self::visit_type(tup_ty, cx, reason);
                }
            },
            // This is what we are looking for !
            TyKind::Ref(ref optional_lifetime, ref borrow_type) => {
                // Match the 'static lifetime
                if let Some(lifetime) = *optional_lifetime {
                    match borrow_type.ty.kind {
                        TyKind::Path(..) | TyKind::Slice(..) | TyKind::Array(..) | TyKind::Tup(..) => {
                            if lifetime.ident.name == rustc_span::symbol::kw::StaticLifetime {
                                let snip = snippet(cx, borrow_type.ty.span, "<type>");
                                let sugg = format!("&{}{snip}", borrow_type.mutbl.prefix_str());
                                span_lint_and_then(
                                    cx,
                                    REDUNDANT_STATIC_LIFETIMES,
                                    lifetime.ident.span,
                                    reason,
                                    |diag| {
                                        diag.span_suggestion(
                                            ty.span,
                                            "consider removing `'static`",
                                            sugg,
                                            Applicability::MachineApplicable, //snippet
                                        );
                                    },
                                );
                            }
                        },
                        _ => {},
                    }
                }
                Self::visit_type(&borrow_type.ty, cx, reason);
            },
            _ => {},
        }
    }
}

impl EarlyLintPass for RedundantStaticLifetimes {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        if !self.msrv.meets(msrvs::STATIC_IN_CONST) {
            return;
        }

        if !item.span.from_expansion() {
            if let ItemKind::Const(_, ref var_type, _) = item.kind {
                Self::visit_type(var_type, cx, "constants have by default a `'static` lifetime");
                // Don't check associated consts because `'static` cannot be elided on those (issue
                // #2438)
            }

            if let ItemKind::Static(ref var_type, _, _) = item.kind {
                Self::visit_type(var_type, cx, "statics have by default a `'static` lifetime");
            }
        }
    }

    extract_msrv_attr!(EarlyContext);
}
