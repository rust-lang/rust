//! checks for `#[inline]` on trait methods without bodies

use crate::utils::span_lint_and_then;
use crate::utils::sugg::DiagnosticBuilderExt;
use rustc_ast::ast::Attribute;
use rustc_errors::Applicability;
use rustc_hir::{TraitFn, TraitItem, TraitItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{sym, Symbol};

declare_clippy_lint! {
    /// **What it does:** Checks for `#[inline]` on trait methods without bodies
    ///
    /// **Why is this bad?** Only implementations of trait methods may be inlined.
    /// The inline attribute is ignored for trait methods without bodies.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// trait Animal {
    ///     #[inline]
    ///     fn name(&self) -> &'static str;
    /// }
    /// ```
    pub INLINE_FN_WITHOUT_BODY,
    correctness,
    "use of `#[inline]` on trait methods without bodies"
}

declare_lint_pass!(InlineFnWithoutBody => [INLINE_FN_WITHOUT_BODY]);

impl<'tcx> LateLintPass<'tcx> for InlineFnWithoutBody {
    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx TraitItem<'_>) {
        if let TraitItemKind::Fn(_, TraitFn::Required(_)) = item.kind {
            check_attrs(cx, item.ident.name, &item.attrs);
        }
    }
}

fn check_attrs(cx: &LateContext<'_>, name: Symbol, attrs: &[Attribute]) {
    for attr in attrs {
        if !attr.has_name(sym::inline) {
            continue;
        }

        span_lint_and_then(
            cx,
            INLINE_FN_WITHOUT_BODY,
            attr.span,
            &format!("use of `#[inline]` on trait method `{}` which has no body", name),
            |diag| {
                diag.suggest_remove_item(cx, attr.span, "remove", Applicability::MachineApplicable);
            },
        );
    }
}
