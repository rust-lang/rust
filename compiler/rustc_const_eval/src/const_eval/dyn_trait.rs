use rustc_middle::mir::interpret::{CtfeProvenance, InterpResult, Scalar, interp_ok};
use rustc_middle::ty::{Region, Ty};
use rustc_middle::{span_bug, ty};
use rustc_span::def_id::DefId;
use rustc_span::sym;

use crate::const_eval::CompileTimeMachine;
use crate::interpret::{Immediate, InterpCx, MPlaceTy, MemoryKind, Writeable};
impl<'tcx> InterpCx<'tcx, CompileTimeMachine<'tcx>> {
    pub(crate) fn write_dyn_trait_type_info(
        &mut self,
        dyn_place: impl Writeable<'tcx, CtfeProvenance>,
        data: &'tcx ty::List<ty::Binder<'tcx, ty::ExistentialPredicate<'tcx>>>,
        region: Region<'tcx>,
    ) -> InterpResult<'tcx> {
        let tcx = self.tcx.tcx;

        // Find the principal trait ref (for super trait collection), collect auto traits,
        // and collect all projection predicates (used when computing TypeId for each supertrait).
        let mut principal: Option<ty::Binder<'tcx, ty::ExistentialTraitRef<'tcx>>> = None;
        let mut auto_traits_def_ids: Vec<ty::Binder<'tcx, DefId>> = Vec::new();
        let mut projections: Vec<ty::Binder<'tcx, ty::ExistentialProjection<'tcx>>> = Vec::new();

        for b in data.iter() {
            match b.skip_binder() {
                ty::ExistentialPredicate::Trait(tr) => principal = Some(b.rebind(tr)),
                ty::ExistentialPredicate::AutoTrait(did) => auto_traits_def_ids.push(b.rebind(did)),
                ty::ExistentialPredicate::Projection(p) => projections.push(b.rebind(p)),
            }
        }

        // This is to make principal dyn type include Trait and projection predicates, excluding auto traits.
        let principal_ty: Option<Ty<'tcx>> = principal.map(|_tr| {
            let preds = tcx
                .mk_poly_existential_predicates_from_iter(data.iter().filter(|b| {
                    !matches!(b.skip_binder(), ty::ExistentialPredicate::AutoTrait(_))
                }));
            Ty::new_dynamic(tcx, preds, region)
        });

        // DynTrait { predicates: &'static [Trait] }
        for (field_idx, field) in
            dyn_place.layout().ty.ty_adt_def().unwrap().non_enum_variant().fields.iter_enumerated()
        {
            let field_place = self.project_field(&dyn_place, field_idx)?;
            match field.name {
                sym::predicates => {
                    self.write_dyn_trait_predicates_slice(
                        &field_place,
                        principal_ty,
                        &auto_traits_def_ids,
                        region,
                    )?;
                }
                other => {
                    span_bug!(self.tcx.def_span(field.did), "unimplemented DynTrait field {other}")
                }
            }
        }

        interp_ok(())
    }

    fn mk_dyn_principal_auto_trait_ty(
        &self,
        auto_trait_def_id: ty::Binder<'tcx, DefId>,
        region: Region<'tcx>,
    ) -> Ty<'tcx> {
        let tcx = self.tcx.tcx;

        // Preserve the binder vars from the original auto-trait predicate.
        let pred_inner = ty::ExistentialPredicate::AutoTrait(auto_trait_def_id.skip_binder());
        let pred = ty::Binder::bind_with_vars(pred_inner, auto_trait_def_id.bound_vars());

        let preds = tcx.mk_poly_existential_predicates_from_iter([pred].into_iter());
        Ty::new_dynamic(tcx, preds, region)
    }

    fn write_dyn_trait_predicates_slice(
        &mut self,
        slice_place: &impl Writeable<'tcx, CtfeProvenance>,
        principal_ty: Option<Ty<'tcx>>,
        auto_trait_def_ids: &[ty::Binder<'tcx, DefId>],
        region: Region<'tcx>,
    ) -> InterpResult<'tcx> {
        let tcx = self.tcx.tcx;

        // total entries in DynTrait predicates
        let total_len = principal_ty.map(|_| 1).unwrap_or(0) + auto_trait_def_ids.len();

        // element type = DynTraitPredicate
        let slice_ty = slice_place.layout().ty.builtin_deref(false).unwrap(); // [DynTraitPredicate]
        let elem_ty = slice_ty.sequence_element_type(tcx); // DynTraitPredicate

        let arr_layout = self.layout_of(Ty::new_array(tcx, elem_ty, total_len as u64))?;
        let arr_place = self.allocate(arr_layout, MemoryKind::Stack)?;
        let mut elems = self.project_array_fields(&arr_place)?;

        // principal entry (if any) - NOT an auto trait
        if let Some(principal_ty) = principal_ty {
            let Some((_i, elem_place)) = elems.next(self)? else {
                span_bug!(self.tcx.span, "DynTrait.predicates length computed wrong (principal)");
            };
            self.write_dyn_trait_predicate(elem_place, principal_ty, false)?;
        }

        // auto trait entries - these ARE auto traits
        for auto in auto_trait_def_ids {
            let Some((_i, elem_place)) = elems.next(self)? else {
                span_bug!(self.tcx.span, "DynTrait.predicates length computed wrong (auto)");
            };
            let auto_ty = self.mk_dyn_principal_auto_trait_ty(*auto, region);
            self.write_dyn_trait_predicate(elem_place, auto_ty, true)?;
        }

        let arr_place = arr_place.map_provenance(CtfeProvenance::as_immutable);
        let imm = Immediate::new_slice(arr_place.ptr(), total_len as u64, self);
        self.write_immediate(imm, slice_place)
    }

    fn write_dyn_trait_predicate(
        &mut self,
        predicate_place: MPlaceTy<'tcx>,
        trait_ty: Ty<'tcx>,
        is_auto: bool,
    ) -> InterpResult<'tcx> {
        // DynTraitPredicate { trait_ty: Trait }
        for (field_idx, field) in predicate_place
            .layout
            .ty
            .ty_adt_def()
            .unwrap()
            .non_enum_variant()
            .fields
            .iter_enumerated()
        {
            let field_place = self.project_field(&predicate_place, field_idx)?;
            match field.name {
                sym::trait_ty => {
                    // Now write the Trait struct
                    self.write_trait(field_place, trait_ty, is_auto)?;
                }
                other => {
                    span_bug!(
                        self.tcx.def_span(field.did),
                        "unimplemented DynTraitPredicate field {other}"
                    )
                }
            }
        }
        interp_ok(())
    }
    fn write_trait(
        &mut self,
        trait_place: MPlaceTy<'tcx>,
        trait_ty: Ty<'tcx>,
        is_auto: bool,
    ) -> InterpResult<'tcx> {
        // Trait { ty: TypeId, is_auto: bool }
        for (field_idx, field) in
            trait_place.layout.ty.ty_adt_def().unwrap().non_enum_variant().fields.iter_enumerated()
        {
            let field_place = self.project_field(&trait_place, field_idx)?;
            match field.name {
                sym::ty => {
                    self.write_type_id(trait_ty, &field_place)?;
                }
                sym::is_auto => {
                    self.write_scalar(Scalar::from_bool(is_auto), &field_place)?;
                }
                other => {
                    span_bug!(self.tcx.def_span(field.did), "unimplemented Trait field {other}")
                }
            }
        }
        interp_ok(())
    }
}
