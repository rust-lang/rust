use clippy_utils::diagnostics::span_lint_hir;
use rustc_hir::{Impl, Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for manual re-implementations of `PartialEq::ne`.
    ///
    /// ### Why is this bad?
    /// `PartialEq::ne` is required to always return the
    /// negated result of `PartialEq::eq`, which is exactly what the default
    /// implementation does. Therefore, there should never be any need to
    /// re-implement it.
    ///
    /// ### Example
    /// ```no_run
    /// struct Foo;
    ///
    /// impl PartialEq for Foo {
    ///     fn eq(&self, other: &Foo) -> bool { true }
    ///     fn ne(&self, other: &Foo) -> bool { !(self == other) }
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub PARTIALEQ_NE_IMPL,
    complexity,
    "re-implementing `PartialEq::ne`"
}

declare_lint_pass!(PartialEqNeImpl => [PARTIALEQ_NE_IMPL]);

impl<'tcx> LateLintPass<'tcx> for PartialEqNeImpl {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if let ItemKind::Impl(Impl {
            of_trait: Some(of_trait),
            items: impl_items,
            ..
        }) = item.kind
            && !cx.tcx.is_automatically_derived(item.owner_id.to_def_id())
            && let Some(eq_trait) = cx.tcx.lang_items().eq_trait()
            && of_trait.trait_ref.path.res.def_id() == eq_trait
        {
            for impl_item in impl_items {
                if cx.tcx.item_name(impl_item.owner_id) == sym::ne {
                    span_lint_hir(
                        cx,
                        PARTIALEQ_NE_IMPL,
                        impl_item.hir_id(),
                        cx.tcx.def_span(impl_item.owner_id),
                        "re-implementing `PartialEq::ne` is unnecessary",
                    );
                }
            }
        }
    }
}
