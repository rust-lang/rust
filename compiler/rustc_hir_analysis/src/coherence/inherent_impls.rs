//! The code in this module gathers up all of the inherent impls in
//! the current crate and organizes them in a map. It winds up
//! touching the whole crate and thus must be recomputed completely
//! for any change, but it is very cheap to compute. In practice, most
//! code in the compiler never *directly* requests this map. Instead,
//! it requests the inherent impls specific to some type (via
//! `tcx.inherent_impls(def_id)`). That value, however,
//! is computed by selecting an idea from this table.

use rustc_hir as hir;
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::find_attr;
use rustc_middle::bug;
use rustc_middle::ty::fast_reject::{SimplifiedType, TreatParams, simplify_type};
use rustc_middle::ty::{self, CrateInherentImpls, Ty, TyCtxt};
use rustc_span::{ErrorGuaranteed, sym};

use crate::errors;

/// On-demand query: yields a map containing all types mapped to their inherent impls.
pub(crate) fn crate_inherent_impls(
    tcx: TyCtxt<'_>,
    (): (),
) -> (&'_ CrateInherentImpls, Result<(), ErrorGuaranteed>) {
    let mut collect = InherentCollect { tcx, impls_map: Default::default() };

    let mut res = Ok(());
    for id in tcx.hir_free_items() {
        res = res.and(collect.check_item(id));
    }

    (tcx.arena.alloc(collect.impls_map), res)
}

pub(crate) fn crate_inherent_impls_validity_check(
    tcx: TyCtxt<'_>,
    (): (),
) -> Result<(), ErrorGuaranteed> {
    tcx.crate_inherent_impls(()).1
}

pub(crate) fn crate_incoherent_impls(tcx: TyCtxt<'_>, simp: SimplifiedType) -> &[DefId] {
    let (crate_map, _) = tcx.crate_inherent_impls(());
    tcx.arena.alloc_from_iter(
        crate_map.incoherent_impls.get(&simp).unwrap_or(&Vec::new()).iter().map(|d| d.to_def_id()),
    )
}

/// On-demand query: yields a vector of the inherent impls for a specific type.
pub(crate) fn inherent_impls(tcx: TyCtxt<'_>, ty_def_id: LocalDefId) -> &[DefId] {
    let (crate_map, _) = tcx.crate_inherent_impls(());
    match crate_map.inherent_impls.get(&ty_def_id) {
        Some(v) => &v[..],
        None => &[],
    }
}

struct InherentCollect<'tcx> {
    tcx: TyCtxt<'tcx>,
    impls_map: CrateInherentImpls,
}

impl<'tcx> InherentCollect<'tcx> {
    fn check_def_id(
        &mut self,
        impl_def_id: LocalDefId,
        self_ty: Ty<'tcx>,
        ty_def_id: DefId,
    ) -> Result<(), ErrorGuaranteed> {
        if let Some(ty_def_id) = ty_def_id.as_local() {
            // Add the implementation to the mapping from implementation to base
            // type def ID, if there is a base type for this implementation and
            // the implementation does not have any associated traits.
            let vec = self.impls_map.inherent_impls.entry(ty_def_id).or_default();
            vec.push(impl_def_id.to_def_id());
            return Ok(());
        }

        if self.tcx.features().rustc_attrs() {
            let items = self.tcx.associated_item_def_ids(impl_def_id);

            if !self.tcx.has_attr(ty_def_id, sym::rustc_has_incoherent_inherent_impls) {
                let impl_span = self.tcx.def_span(impl_def_id);
                return Err(self.tcx.dcx().emit_err(errors::InherentTyOutside { span: impl_span }));
            }

            for &impl_item in items {
                if !find_attr!(
                    self.tcx.get_all_attrs(impl_item),
                    AttributeKind::AllowIncoherentImpl(_)
                ) {
                    let impl_span = self.tcx.def_span(impl_def_id);
                    return Err(self.tcx.dcx().emit_err(errors::InherentTyOutsideRelevant {
                        span: impl_span,
                        help_span: self.tcx.def_span(impl_item),
                    }));
                }
            }

            if let Some(simp) = simplify_type(self.tcx, self_ty, TreatParams::InstantiateWithInfer)
            {
                self.impls_map.incoherent_impls.entry(simp).or_default().push(impl_def_id);
            } else {
                bug!("unexpected self type: {:?}", self_ty);
            }
            Ok(())
        } else {
            let impl_span = self.tcx.def_span(impl_def_id);
            Err(self.tcx.dcx().emit_err(errors::InherentTyOutsideNew { span: impl_span }))
        }
    }

    fn check_primitive_impl(
        &mut self,
        impl_def_id: LocalDefId,
        ty: Ty<'tcx>,
    ) -> Result<(), ErrorGuaranteed> {
        let items = self.tcx.associated_item_def_ids(impl_def_id);
        if !self.tcx.hir_rustc_coherence_is_core() {
            if self.tcx.features().rustc_attrs() {
                for &impl_item in items {
                    if !find_attr!(
                        self.tcx.get_all_attrs(impl_item),
                        AttributeKind::AllowIncoherentImpl(_)
                    ) {
                        let span = self.tcx.def_span(impl_def_id);
                        return Err(self.tcx.dcx().emit_err(errors::InherentTyOutsidePrimitive {
                            span,
                            help_span: self.tcx.def_span(impl_item),
                        }));
                    }
                }
            } else {
                let span = self.tcx.def_span(impl_def_id);
                let mut note = None;
                if let ty::Ref(_, subty, _) = ty.kind() {
                    note = Some(errors::InherentPrimitiveTyNote { subty: *subty });
                }
                return Err(self.tcx.dcx().emit_err(errors::InherentPrimitiveTy { span, note }));
            }
        }

        if let Some(simp) = simplify_type(self.tcx, ty, TreatParams::InstantiateWithInfer) {
            self.impls_map.incoherent_impls.entry(simp).or_default().push(impl_def_id);
        } else {
            bug!("unexpected primitive type: {:?}", ty);
        }
        Ok(())
    }

    fn check_item(&mut self, id: hir::ItemId) -> Result<(), ErrorGuaranteed> {
        if !matches!(self.tcx.def_kind(id.owner_id), DefKind::Impl { of_trait: false }) {
            return Ok(());
        }

        let id = id.owner_id.def_id;
        let item_span = self.tcx.def_span(id);
        let self_ty = self.tcx.type_of(id).instantiate_identity();
        let mut self_ty = self.tcx.peel_off_free_alias_tys(self_ty);
        // We allow impls on pattern types exactly when we allow impls on the base type.
        // FIXME(pattern_types): Figure out the exact coherence rules we want here.
        while let ty::Pat(base, _) = *self_ty.kind() {
            self_ty = base;
        }
        match *self_ty.kind() {
            ty::Adt(def, _) => self.check_def_id(id, self_ty, def.did()),
            ty::Foreign(did) => self.check_def_id(id, self_ty, did),
            ty::Dynamic(data, ..) if data.principal_def_id().is_some() => {
                self.check_def_id(id, self_ty, data.principal_def_id().unwrap())
            }
            ty::Dynamic(..) => {
                Err(self.tcx.dcx().emit_err(errors::InherentDyn { span: item_span }))
            }
            ty::Pat(_, _) => unreachable!(),
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Str
            | ty::Array(..)
            | ty::Slice(_)
            | ty::RawPtr(_, _)
            | ty::Ref(..)
            | ty::Never
            | ty::FnPtr(..)
            | ty::Tuple(..)
            | ty::UnsafeBinder(_) => self.check_primitive_impl(id, self_ty),
            ty::Alias(ty::Projection | ty::Inherent | ty::Opaque, _) | ty::Param(_) => {
                Err(self.tcx.dcx().emit_err(errors::InherentNominal { span: item_span }))
            }
            ty::FnDef(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..)
            | ty::CoroutineWitness(..) => {
                Err(self.tcx.dcx().delayed_bug("cannot define inherent `impl` for closure types"))
            }
            ty::Alias(ty::Free, _) | ty::Bound(..) | ty::Placeholder(_) | ty::Infer(_) => {
                bug!("unexpected impl self type of impl: {:?} {:?}", id, self_ty);
            }
            // We could bail out here, but that will silence other useful errors.
            ty::Error(_) => Ok(()),
        }
    }
}
