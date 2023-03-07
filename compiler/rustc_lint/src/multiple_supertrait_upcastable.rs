use crate::{LateContext, LateLintPass, LintContext};

use rustc_hir as hir;
use rustc_span::sym;

declare_lint! {
    /// The `multiple_supertrait_upcastable` lint detects when an object-safe trait has multiple
    /// supertraits.
    ///
    /// ### Example
    ///
    /// ```rust
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
    "detect when an object-safe trait has multiple supertraits",
    @feature_gate = sym::multiple_supertrait_upcastable;
}

declare_lint_pass!(MultipleSupertraitUpcastable => [MULTIPLE_SUPERTRAIT_UPCASTABLE]);

impl<'tcx> LateLintPass<'tcx> for MultipleSupertraitUpcastable {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        let def_id = item.owner_id.to_def_id();
        // NOTE(nbdd0121): use `object_safety_violations` instead of `check_is_object_safe` because
        // the latter will report `where_clause_object_safety` lint.
        if let hir::ItemKind::Trait(_, _, _, _, _) = item.kind
            && cx.tcx.object_safety_violations(def_id).is_empty()
        {
            let direct_super_traits_iter = cx.tcx
                    .super_predicates_of(def_id)
                    .predicates
                    .into_iter()
                    .filter_map(|(pred, _)| pred.to_opt_poly_trait_pred());
            if direct_super_traits_iter.count() > 1 {
                cx.emit_spanned_lint(
                    MULTIPLE_SUPERTRAIT_UPCASTABLE,
                    cx.tcx.def_span(def_id),
                    crate::lints::MultipleSupertraitUpcastable {
                        ident: item.ident
                    },
                );
            }
        }
    }
}
