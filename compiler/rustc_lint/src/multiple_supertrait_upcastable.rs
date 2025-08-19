use rustc_hir as hir;
use rustc_session::{declare_lint, declare_lint_pass};

use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `multiple_supertrait_upcastable` lint detects when a dyn-compatible trait has multiple
    /// supertraits.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![feature(multiple_supertrait_upcastable)]
    /// trait A {}
    /// trait B {}
    ///
    /// #[warn(multiple_supertrait_upcastable)]
    /// trait C: A + B {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// To support upcasting with multiple supertraits, we need to store multiple vtables and this
    /// can result in extra space overhead, even if no code actually uses upcasting.
    /// This lint allows users to identify when such scenarios occur and to decide whether the
    /// additional overhead is justified.
    pub MULTIPLE_SUPERTRAIT_UPCASTABLE,
    Allow,
    "detect when a dyn-compatible trait has multiple supertraits",
    @feature_gate = multiple_supertrait_upcastable;
}

declare_lint_pass!(MultipleSupertraitUpcastable => [MULTIPLE_SUPERTRAIT_UPCASTABLE]);

impl<'tcx> LateLintPass<'tcx> for MultipleSupertraitUpcastable {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        let def_id = item.owner_id.to_def_id();
        // NOTE(nbdd0121): use `dyn_compatibility_violations` instead of `is_dyn_compatible` because
        // the latter will report `where_clause_object_safety` lint.
        if let hir::ItemKind::Trait(_, _, _, ident, ..) = item.kind
            && cx.tcx.is_dyn_compatible(def_id)
        {
            let direct_super_traits_iter = cx
                .tcx
                .explicit_super_predicates_of(def_id)
                .iter_identity_copied()
                .filter_map(|(pred, _)| pred.as_trait_clause())
                .filter(|pred| !cx.tcx.is_lang_item(pred.def_id(), hir::LangItem::MetaSized))
                .filter(|pred| !cx.tcx.is_default_trait(pred.def_id()));
            if direct_super_traits_iter.count() > 1 {
                cx.emit_span_lint(
                    MULTIPLE_SUPERTRAIT_UPCASTABLE,
                    cx.tcx.def_span(def_id),
                    crate::lints::MultipleSupertraitUpcastable { ident },
                );
            }
        }
    }
}
