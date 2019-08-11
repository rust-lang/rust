use crate::utils::{is_automatically_derived, span_lint_hir};
use if_chain::if_chain;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for manual re-implementations of `PartialEq::ne`.
    ///
    /// **Why is this bad?** `PartialEq::ne` is required to always return the
    /// negated result of `PartialEq::eq`, which is exactly what the default
    /// implementation does. Therefore, there should never be any need to
    /// re-implement it.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// struct Foo;
    ///
    /// impl PartialEq for Foo {
    ///    fn eq(&self, other: &Foo) -> bool { true }
    ///    fn ne(&self, other: &Foo) -> bool { !(self == other) }
    /// }
    /// ```
    pub PARTIALEQ_NE_IMPL,
    complexity,
    "re-implementing `PartialEq::ne`"
}

declare_lint_pass!(PartialEqNeImpl => [PARTIALEQ_NE_IMPL]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for PartialEqNeImpl {
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item) {
        if_chain! {
            if let ItemKind::Impl(_, _, _, _, Some(ref trait_ref), _, ref impl_items) = item.node;
            if !is_automatically_derived(&*item.attrs);
            if let Some(eq_trait) = cx.tcx.lang_items().eq_trait();
            if trait_ref.path.res.def_id() == eq_trait;
            then {
                for impl_item in impl_items {
                    if impl_item.ident.name == sym!(ne) {
                        span_lint_hir(
                            cx,
                            PARTIALEQ_NE_IMPL,
                            impl_item.id.hir_id,
                            impl_item.span,
                            "re-implementing `PartialEq::ne` is unnecessary",
                        );
                    }
                }
            }
        };
    }
}
