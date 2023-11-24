use crate::{
    lints::{SupertraitAsDerefTarget, SupertraitAsDerefTargetLabel},
    LateContext, LateLintPass, LintContext,
};

use rustc_hir as hir;
use rustc_middle::ty;
use rustc_span::sym;
use rustc_trait_selection::traits::supertraits;

declare_lint! {
    /// The `deref_into_dyn_supertrait` lint is output whenever there is a use of the
    /// `Deref` implementation with a `dyn SuperTrait` type as `Output`.
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
    /// The implicit dyn upcasting coercion take priority over those `Deref` impls.
    pub DEREF_INTO_DYN_SUPERTRAIT,
    Warn,
    "`Deref` implementation usage with a supertrait trait object for output are shadow by implicit coercion",
}

declare_lint_pass!(DerefIntoDynSupertrait => [DEREF_INTO_DYN_SUPERTRAIT]);

impl<'tcx> LateLintPass<'tcx> for DerefIntoDynSupertrait {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        let tcx = cx.tcx;
        // `Deref` is being implemented for `t`
        if let hir::ItemKind::Impl(impl_) = item.kind
            // the trait is a `Deref` implementation
            && let Some(trait_) = &impl_.of_trait
            && let Some(did) = trait_.trait_def_id()
            && Some(did) == tcx.lang_items().deref_trait()
            // the self type is `dyn t_principal`
            && let self_ty = tcx.type_of(item.owner_id).instantiate_identity()
            && let ty::Dynamic(data, _, ty::Dyn) = self_ty.kind()
            && let Some(self_principal) = data.principal()
            // `<T as Deref>::Target` is `dyn target_principal`
            && let Some(target) = cx.get_associated_type(self_ty, did, "Target")
            && let ty::Dynamic(data, _, ty::Dyn) = target.kind()
            && let Some(target_principal) = data.principal()
            // `target_principal` is a supertrait of `t_principal`
            && let Some(supertrait_principal) = supertraits(tcx, self_principal.with_self_ty(tcx, self_ty))
                .find(|supertrait| supertrait.def_id() == target_principal.def_id())
        {
            // erase regions in self type for better diagnostic presentation
            let (self_ty, target_principal, supertrait_principal) =
                tcx.erase_regions((self_ty, target_principal, supertrait_principal));
            let label2 = impl_
                .items
                .iter()
                .find_map(|i| (i.ident.name == sym::Target).then_some(i.span))
                .map(|label| SupertraitAsDerefTargetLabel { label });
            let span = tcx.def_span(item.owner_id.def_id);
            cx.emit_spanned_lint(
                DEREF_INTO_DYN_SUPERTRAIT,
                span,
                SupertraitAsDerefTarget {
                    self_ty,
                    supertrait_principal: supertrait_principal.map_bound(|trait_ref| {
                        ty::ExistentialTraitRef::erase_self_ty(tcx, trait_ref)
                    }),
                    target_principal,
                    label: span,
                    label2,
                },
            );
        }
    }
}
