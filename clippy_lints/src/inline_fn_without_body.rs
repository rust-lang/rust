//! checks for `#[inline]` on trait methods without bodies

use crate::utils::span_lint_and_then;
use crate::utils::sugg::DiagnosticBuilderExt;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_tool_lint, lint_array};
use rustc_errors::Applicability;
use syntax::ast::{Attribute, Name};

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

#[derive(Copy, Clone)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(INLINE_FN_WITHOUT_BODY)
    }

    fn name(&self) -> &'static str {
        "InlineFnWithoutBody"
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_trait_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx TraitItem) {
        if let TraitItemKind::Method(_, TraitMethod::Required(_)) = item.node {
            check_attrs(cx, item.ident.name, &item.attrs);
        }
    }
}

fn check_attrs(cx: &LateContext<'_, '_>, name: Name, attrs: &[Attribute]) {
    for attr in attrs {
        if !attr.check_name("inline") {
            continue;
        }

        span_lint_and_then(
            cx,
            INLINE_FN_WITHOUT_BODY,
            attr.span,
            &format!("use of `#[inline]` on trait method `{}` which has no body", name),
            |db| {
                db.suggest_remove_item(cx, attr.span, "remove", Applicability::MachineApplicable);
            },
        );
    }
}
