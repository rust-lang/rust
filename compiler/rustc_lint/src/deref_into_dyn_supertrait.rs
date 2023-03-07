use crate::{
    lints::{SupertraitAsDerefTarget, SupertraitAsDerefTargetLabel},
    LateContext, LateLintPass, LintContext,
};

use rustc_hir as hir;
use rustc_middle::{traits::util::supertraits, ty};
use rustc_span::sym;

declare_lint! {
    /// The `deref_into_dyn_supertrait` lint is output whenever there is a use of the
    /// `Deref` implementation with a `dyn SuperTrait` type as `Output`.
    ///
    /// These implementations will become shadowed when the `trait_upcasting` feature is stabilized.
    /// The `deref` functions will no longer be called implicitly, so there might be behavior change.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(deref_into_dyn_supertrait)]
    /// #![allow(dead_code)]
    ///
    /// use core::ops::Deref;
    ///
    /// trait A {}
    /// trait B: A {}
    /// impl<'a> Deref for dyn 'a + B {
    ///     type Target = dyn A;
    ///     fn deref(&self) -> &Self::Target {
    ///         todo!()
    ///     }
    /// }
    ///
    /// fn take_a(_: &dyn A) { }
    ///
    /// fn take_b(b: &dyn B) {
    ///     take_a(b);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The dyn upcasting coercion feature adds new coercion rules, taking priority
    /// over certain other coercion rules, which will cause some behavior change.
    pub DEREF_INTO_DYN_SUPERTRAIT,
    Warn,
    "`Deref` implementation usage with a supertrait trait object for output might be shadowed in the future",
    @future_incompatible = FutureIncompatibleInfo {
        reference: "issue #89460 <https://github.com/rust-lang/rust/issues/89460>",
    };
}

declare_lint_pass!(DerefIntoDynSupertrait => [DEREF_INTO_DYN_SUPERTRAIT]);

impl<'tcx> LateLintPass<'tcx> for DerefIntoDynSupertrait {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        // `Deref` is being implemented for `t`
        if let hir::ItemKind::Impl(impl_) = item.kind
            && let Some(trait_) = &impl_.of_trait
            && let t = cx.tcx.type_of(item.owner_id).subst_identity()
            && let opt_did @ Some(did) = trait_.trait_def_id()
            && opt_did == cx.tcx.lang_items().deref_trait()
            // `t` is `dyn t_principal`
            && let ty::Dynamic(data, _, ty::Dyn) = t.kind()
            && let Some(t_principal) = data.principal()
            // `<T as Deref>::Target` is `dyn target_principal`
            && let Some(target) = cx.get_associated_type(t, did, "Target")
            && let ty::Dynamic(data, _, ty::Dyn) = target.kind()
            && let Some(target_principal) = data.principal()
            // `target_principal` is a supertrait of `t_principal`
            && supertraits(cx.tcx, t_principal.with_self_ty(cx.tcx, cx.tcx.types.trait_object_dummy_self))
                .any(|sup| sup.map_bound(|x| ty::ExistentialTraitRef::erase_self_ty(cx.tcx, x)) == target_principal)
        {
            let label = impl_.items.iter().find_map(|i| (i.ident.name == sym::Target).then_some(i.span)).map(|label| SupertraitAsDerefTargetLabel {
                label,
            });
            cx.emit_spanned_lint(DEREF_INTO_DYN_SUPERTRAIT, cx.tcx.def_span(item.owner_id.def_id), SupertraitAsDerefTarget {
                t,
                target_principal,
                label,
            });
        }
    }
}
