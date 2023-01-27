use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::is_lint_allowed;
use clippy_utils::macros::span_is_local;
use rustc_hir::def_id::DefIdMap;
use rustc_hir::{Impl, Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::AssocItem;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks if a provided method is used implicitly by a trait
    /// implementation. A usage example would be a wrapper where every method
    /// should perform some operation before delegating to the inner type's
    /// implemenation.
    ///
    /// This lint should typically be enabled on a specific trait `impl` item
    /// rather than globally.
    ///
    /// ### Why is this bad?
    /// Indicates that a method is missing.
    ///
    /// ### Example
    /// ```rust
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
    /// ```rust
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
                items,
                of_trait: Some(trait_ref),
                ..
            }) = item.kind
            && let Some(trait_id) = trait_ref.trait_def_id()
        {
            let mut provided: DefIdMap<&AssocItem> = cx
                .tcx
                .provided_trait_methods(trait_id)
                .map(|assoc| (assoc.def_id, assoc))
                .collect();

            for impl_item in *items {
                if let Some(def_id) = impl_item.trait_item_def_id {
                    provided.remove(&def_id);
                }
            }

            cx.tcx.with_stable_hashing_context(|hcx| {
                for assoc in provided.values_sorted(&hcx, true) {
                    let source_map = cx.tcx.sess.source_map();
                    let definition_span = source_map.guess_head_span(cx.tcx.def_span(assoc.def_id));

                    span_lint_and_help(
                        cx,
                        MISSING_TRAIT_METHODS,
                        source_map.guess_head_span(item.span),
                        &format!("missing trait method provided by default: `{}`", assoc.name),
                        Some(definition_span),
                        "implement the method",
                    );
                }
            });
        }
    }
}
