use rustc_hir::{self as hir, LangItem};
use rustc_middle::ty;
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::sym;
use rustc_trait_selection::traits::supertraits;

use crate::lints::{SupertraitAsDerefTarget, SupertraitAsDerefTargetLabel};
use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `deref_into_dyn_supertrait` lint is emitted whenever there is a `Deref` implementation
    /// for `dyn SubTrait` with a `dyn SuperTrait` type as the `Output` type.
    ///
    /// These implementations are "shadowed" by trait upcasting (stabilized since
    /// 1.86.0). The `deref` functions is no longer called implicitly, which might
    /// change behavior compared to previous rustc versions.
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
    /// The trait upcasting coercion added a new coercion rule, taking priority over certain other
    /// coercion rules, which causes some behavior change compared to older `rustc` versions.
    ///
    /// `deref` can be still called explicitly, it just isn't called as part of a deref coercion
    /// (since trait upcasting coercion takes priority).
    pub DEREF_INTO_DYN_SUPERTRAIT,
    Allow,
    "`Deref` implementation with a supertrait trait object for output is shadowed by trait upcasting",
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
            && tcx.is_lang_item(did, LangItem::Deref)
            // the self type is `dyn t_principal`
            && let self_ty = tcx.type_of(item.owner_id).instantiate_identity()
            && let ty::Dynamic(data, _, ty::Dyn) = self_ty.kind()
            && let Some(self_principal) = data.principal()
            // `<T as Deref>::Target` is `dyn target_principal`
            && let Some(target) = cx.get_associated_type(self_ty, did, sym::Target)
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
            cx.emit_span_lint(
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
