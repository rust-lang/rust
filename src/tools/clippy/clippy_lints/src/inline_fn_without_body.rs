use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sugg::DiagExt;
use rustc_errors::Applicability;
use rustc_hir::attrs::AttributeKind;
use rustc_hir::{TraitFn, TraitItem, TraitItemKind, find_attr};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `#[inline]` on trait methods without bodies
    ///
    /// ### Why is this bad?
    /// Only implementations of trait methods may be inlined.
    /// The inline attribute is ignored for trait methods without bodies.
    ///
    /// ### Example
    /// ```no_run
    /// trait Animal {
    ///     #[inline]
    ///     fn name(&self) -> &'static str;
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub INLINE_FN_WITHOUT_BODY,
    correctness,
    "use of `#[inline]` on trait methods without bodies"
}

declare_lint_pass!(InlineFnWithoutBody => [INLINE_FN_WITHOUT_BODY]);

impl<'tcx> LateLintPass<'tcx> for InlineFnWithoutBody {
    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx TraitItem<'_>) {
        if let TraitItemKind::Fn(_, TraitFn::Required(_)) = item.kind
            && let Some(attr_span) = find_attr!(cx
                    .tcx
                    .hir_attrs(item.hir_id()),
                    AttributeKind::Inline(_, span) => *span
            )
        {
            span_lint_and_then(
                cx,
                INLINE_FN_WITHOUT_BODY,
                attr_span,
                format!("use of `#[inline]` on trait method `{}` which has no body", item.ident),
                |diag| {
                    diag.suggest_remove_item(cx, attr_span, "remove", Applicability::MachineApplicable);
                },
            );
        }
    }
}
