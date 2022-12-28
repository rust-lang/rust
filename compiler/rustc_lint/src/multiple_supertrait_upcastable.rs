use crate::{LateContext, LateLintPass, LintContext};

use rustc_errors::DelayDm;
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
        if let hir::ItemKind::Trait(_, _, _, _, _) = item.kind
            && cx.tcx.is_object_safe(def_id)
        {
            let direct_super_traits_iter = cx.tcx
                    .super_predicates_of(def_id)
                    .predicates
                    .into_iter()
                    .filter_map(|(pred, _)| pred.to_opt_poly_trait_pred());
            if direct_super_traits_iter.count() > 1 {
                cx.struct_span_lint(
                    MULTIPLE_SUPERTRAIT_UPCASTABLE,
                    cx.tcx.def_span(def_id),
                    DelayDm(|| {
                        format!(
                            "`{}` is object-safe and has multiple supertraits",
                            item.ident,
                        )
                    }),
                    |diag| diag,
                );
            }
        }
    }
}
