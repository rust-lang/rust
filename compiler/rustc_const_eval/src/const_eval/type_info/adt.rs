use rustc_abi::{FieldIdx, VariantIdx};
use rustc_middle::ty::{
    AdtDef, AdtKind, Const, ConstKind, GenericArgKind, GenericArgs, Region, Ty,
};
use rustc_span::{bug, span_bug, sym};

use crate::const_eval::CompileTimeMachine;
use crate::interpret::{
    CtfeProvenance, InterpCx, InterpResult, MPlaceTy, Projectable, Writeable, interp_ok,
};

impl<'tcx> InterpCx<'tcx, CompileTimeMachine<'tcx>> {
    // FIXME(type_info): No semver considerations for now
    pub(crate) fn write_adt_type_info(
        &mut self,
        place: &impl Writeable<'tcx, CtfeProvenance>,
        adt_def: AdtDef<'tcx>,
    ) -> InterpResult<'tcx, VariantIdx> {
        let variant_idx = match adt_def.adt_kind() {
            AdtKind::Struct => {
                let (variant, _) = self.project_downcast_named(place, sym::Struct)?;
                variant
            }
            AdtKind::Union => {
                let (variant, _) = self.project_downcast_named(place, sym::Union)?;
                variant
            }
            AdtKind::Enum => {
                let (variant, _) = self.project_downcast_named(place, sym::Enum)?;
                variant
            }
        };
        interp_ok(variant_idx)
    }

    pub(super) fn write_generics(
        &mut self,
        place: &impl Writeable<'tcx, CtfeProvenance>,
        generics: &'tcx GenericArgs<'tcx>,
    ) -> InterpResult<'tcx> {
        self.allocate_fill_and_write_slice_ptr(place, generics.len() as u64, |this, i, place| {
            match generics[i as usize].kind() {
                GenericArgKind::Lifetime(region) => this.write_generic_lifetime(region, place),
                GenericArgKind::Type(ty) => this.write_generic_type(ty, place),
                GenericArgKind::Const(c) => this.write_generic_const(c, place),
            }
        })
    }

    fn write_generic_lifetime(
        &mut self,
        _region: Region<'tcx>,
        place: MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let (variant_idx, _) = self.project_downcast_named(&place, sym::Lifetime)?;
        self.write_discriminant(variant_idx, &place)?;
        interp_ok(())
    }

    fn write_generic_type(&mut self, ty: Ty<'tcx>, place: MPlaceTy<'tcx>) -> InterpResult<'tcx> {
        let (variant_idx, variant_place) = self.project_downcast_named(&place, sym::Type)?;
        let generic_type_place = self.project_field(&variant_place, FieldIdx::ZERO)?;

        for (field_idx, field_def) in generic_type_place
            .layout()
            .ty
            .ty_adt_def()
            .unwrap()
            .non_enum_variant()
            .fields
            .iter_enumerated()
        {
            let field_place = self.project_field(&generic_type_place, field_idx)?;
            match field_def.name {
                sym::ty => self.write_type_id(ty, &field_place)?,
                other => span_bug!(self.tcx.def_span(field_def.did), "unimplemented field {other}"),
            }
        }

        self.write_discriminant(variant_idx, &place)?;
        interp_ok(())
    }

    fn write_generic_const(&mut self, c: Const<'tcx>, place: MPlaceTy<'tcx>) -> InterpResult<'tcx> {
        let ConstKind::Value(c) = c.kind() else { bug!("expected a computed const, got {c:?}") };

        let (variant_idx, variant_place) = self.project_downcast_named(&place, sym::Const)?;
        let const_place = self.project_field(&variant_place, FieldIdx::ZERO)?;

        for (field_idx, field_def) in const_place
            .layout()
            .ty
            .ty_adt_def()
            .unwrap()
            .non_enum_variant()
            .fields
            .iter_enumerated()
        {
            let field_place = self.project_field(&const_place, field_idx)?;
            match field_def.name {
                sym::ty => self.write_type_id(c.ty, &field_place)?,
                other => span_bug!(self.tcx.def_span(field_def.did), "unimplemented field {other}"),
            }
        }

        self.write_discriminant(variant_idx, &place)?;
        interp_ok(())
    }
}
