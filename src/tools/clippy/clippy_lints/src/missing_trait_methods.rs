use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_lint_allowed;
use clippy_utils::macros::span_is_local;
use rustc_hir::def_id::DefIdSet;
use rustc_hir::{Impl, Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks if a provided method is used implicitly by a trait
    /// implementation.
    ///
    /// ### Why restrict this?
    /// To ensure that a certain implementation implements every method; for example,
    /// a wrapper type where every method should delegate to the corresponding method of
    /// the inner type's implementation.
    ///
    /// This lint should typically be enabled on a specific trait `impl` item
    /// rather than globally.
    ///
    /// ### Example
    /// ```no_run
    /// trait Trait {
    ///     fn required();
    ///
    ///     fn provided() {}
    /// }
    ///
    /// # struct Type;
    /// #[warn(clippy::missing_trait_methods)]
    /// impl Trait for Type {
    ///     fn required() { /* ... */ }
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// trait Trait {
    ///     fn required();
    ///
    ///     fn provided() {}
    /// }
    ///
    /// # struct Type;
    /// #[warn(clippy::missing_trait_methods)]
    /// impl Trait for Type {
    ///     fn required() { /* ... */ }
    ///
    ///     fn provided() { /* ... */ }
    /// }
    /// ```
    #[clippy::version = "1.66.0"]
    pub MISSING_TRAIT_METHODS,
    restriction,
    "trait implementation uses default provided method"
}
declare_lint_pass!(MissingTraitMethods => [MISSING_TRAIT_METHODS]);

impl<'tcx> LateLintPass<'tcx> for MissingTraitMethods {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if !is_lint_allowed(cx, MISSING_TRAIT_METHODS, item.hir_id())
            && span_is_local(item.span)
            && let ItemKind::Impl(Impl {
                of_trait: Some(of_trait),
                ..
            }) = item.kind
            && let Some(trait_id) = of_trait.trait_ref.trait_def_id()
        {
            let trait_item_ids: DefIdSet = cx
                .tcx
                .associated_items(item.owner_id)
                .in_definition_order()
                .filter_map(|assoc_item| assoc_item.expect_trait_impl().ok())
                .collect();

            for assoc in cx
                .tcx
                .provided_trait_methods(trait_id)
                .filter(|assoc| !trait_item_ids.contains(&assoc.def_id))
            {
                span_lint_and_then(
                    cx,
                    MISSING_TRAIT_METHODS,
                    cx.tcx.def_span(item.owner_id),
                    format!("missing trait method provided by default: `{}`", assoc.name()),
                    |diag| {
                        diag.span_help(cx.tcx.def_span(assoc.def_id), "implement the method");
                    },
                );
            }
        }
    }
}
