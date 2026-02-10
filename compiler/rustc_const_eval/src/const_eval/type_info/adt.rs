use rustc_abi::{FieldIdx, VariantIdx};
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{
    AdtDef, AdtKind, Const, ConstKind, GenericArgKind, GenericArgs, Region, Ty, VariantDef,
};
use rustc_middle::{bug, span_bug};
use rustc_span::sym;

use crate::const_eval::CompileTimeMachine;
use crate::interpret::{
    CtfeProvenance, InterpCx, InterpResult, MPlaceTy, Projectable, Scalar, Writeable, interp_ok,
};

impl<'tcx> InterpCx<'tcx, CompileTimeMachine<'tcx>> {
    // FIXME(type_info): No semver considerations for now
    pub(crate) fn write_adt_type_info(
        &mut self,
        place: &impl Writeable<'tcx, CtfeProvenance>,
        adt: (Ty<'tcx>, AdtDef<'tcx>),
        generics: &'tcx GenericArgs<'tcx>,
    ) -> InterpResult<'tcx, VariantIdx> {
        let (adt_ty, adt_def) = adt;
        let variant_idx = match adt_def.adt_kind() {
            AdtKind::Struct => {
                let (variant, variant_place) = self.downcast(place, sym::Struct)?;
                let place = self.project_field(&variant_place, FieldIdx::ZERO)?;
                self.write_struct_type_info(
                    place,
                    (adt_ty, adt_def.variant(VariantIdx::ZERO)),
                    generics,
                )?;
                variant
            }
            AdtKind::Union => {
                let (variant, variant_place) = self.downcast(place, sym::Union)?;
                let place = self.project_field(&variant_place, FieldIdx::ZERO)?;
                self.write_union_type_info(
                    place,
                    (adt_ty, adt_def.variant(VariantIdx::ZERO)),
                    generics,
                )?;
                variant
            }
            AdtKind::Enum => {
                let (variant, variant_place) = self.downcast(place, sym::Enum)?;
                let place = self.project_field(&variant_place, FieldIdx::ZERO)?;
                self.write_enum_type_info(place, adt, generics)?;
                variant
            }
        };
        interp_ok(variant_idx)
    }

    pub(crate) fn write_struct_type_info(
        &mut self,
        place: impl Writeable<'tcx, CtfeProvenance>,
        struct_: (Ty<'tcx>, &'tcx VariantDef),
        generics: &'tcx GenericArgs<'tcx>,
    ) -> InterpResult<'tcx> {
        let (struct_ty, struct_def) = struct_;
        let struct_layout = self.layout_of(struct_ty)?;

        for (field_idx, field) in
            place.layout().ty.ty_adt_def().unwrap().non_enum_variant().fields.iter_enumerated()
        {
            let field_place = self.project_field(&place, field_idx)?;

            match field.name {
                sym::generics => self.write_generics(field_place, generics)?,
                sym::fields => {
                    self.write_variant_fields(field_place, struct_def, struct_layout, generics)?
                }
                sym::non_exhaustive => {
                    let is_non_exhaustive = struct_def.is_field_list_non_exhaustive();
                    self.write_scalar(Scalar::from_bool(is_non_exhaustive), &field_place)?
                }
                other => span_bug!(self.tcx.def_span(field.did), "unimplemented field {other}"),
            }
        }

        interp_ok(())
    }

    pub(crate) fn write_union_type_info(
        &mut self,
        place: impl Writeable<'tcx, CtfeProvenance>,
        union_: (Ty<'tcx>, &'tcx VariantDef),
        generics: &'tcx GenericArgs<'tcx>,
    ) -> InterpResult<'tcx> {
        let (union_ty, union_def) = union_;
        let union_layout = self.layout_of(union_ty)?;

        for (field_idx, field) in
            place.layout().ty.ty_adt_def().unwrap().non_enum_variant().fields.iter_enumerated()
        {
            let field_place = self.project_field(&place, field_idx)?;

            match field.name {
                sym::generics => self.write_generics(field_place, generics)?,
                sym::fields => {
                    self.write_variant_fields(field_place, union_def, union_layout, generics)?
                }
                sym::non_exhaustive => {
                    let is_non_exhaustive = union_def.is_field_list_non_exhaustive();
                    self.write_scalar(Scalar::from_bool(is_non_exhaustive), &field_place)?
                }
                other => span_bug!(self.tcx.def_span(field.did), "unimplemented field {other}"),
            }
        }

        interp_ok(())
    }

    pub(crate) fn write_enum_type_info(
        &mut self,
        place: impl Writeable<'tcx, CtfeProvenance>,
        enum_: (Ty<'tcx>, AdtDef<'tcx>),
        generics: &'tcx GenericArgs<'tcx>,
    ) -> InterpResult<'tcx> {
        let (enum_ty, enum_def) = enum_;
        let enum_layout = self.layout_of(enum_ty)?;

        for (field_idx, field) in
            place.layout().ty.ty_adt_def().unwrap().non_enum_variant().fields.iter_enumerated()
        {
            let field_place = self.project_field(&place, field_idx)?;

            match field.name {
                sym::generics => self.write_generics(field_place, generics)?,
                sym::variants => {
                    self.allocate_fill_and_write_slice_ptr(
                        field_place,
                        enum_def.variants().len() as u64,
                        |this, i, place| {
                            let variant_idx = VariantIdx::from_usize(i as usize);
                            let variant_def = &enum_def.variants()[variant_idx];
                            let variant_layout = enum_layout.for_variant(this, variant_idx);
                            this.write_enum_variant(place, (variant_layout, &variant_def), generics)
                        },
                    )?;
                }
                sym::non_exhaustive => {
                    let is_non_exhaustive = enum_def.is_variant_list_non_exhaustive();
                    self.write_scalar(Scalar::from_bool(is_non_exhaustive), &field_place)?
                }
                other => span_bug!(self.tcx.def_span(field.did), "unimplemented field {other}"),
            }
        }

        interp_ok(())
    }

    fn write_enum_variant(
        &mut self,
        place: impl Writeable<'tcx, CtfeProvenance>,
        variant: (TyAndLayout<'tcx>, &'tcx VariantDef),
        generics: &'tcx GenericArgs<'tcx>,
    ) -> InterpResult<'tcx> {
        let (variant_layout, variant_def) = variant;

        for (field_idx, field_def) in
            place.layout().ty.ty_adt_def().unwrap().non_enum_variant().fields.iter_enumerated()
        {
            let field_place = self.project_field(&place, field_idx)?;
            match field_def.name {
                sym::name => {
                    let name_place = self.allocate_str_dedup(variant_def.name.as_str())?;
                    let ptr = self.mplace_to_ref(&name_place)?;
                    self.write_immediate(*ptr, &field_place)?
                }
                sym::fields => {
                    self.write_variant_fields(field_place, &variant_def, variant_layout, generics)?
                }
                sym::non_exhaustive => {
                    let is_non_exhaustive = variant_def.is_field_list_non_exhaustive();
                    self.write_scalar(Scalar::from_bool(is_non_exhaustive), &field_place)?
                }
                other => span_bug!(self.tcx.def_span(field_def.did), "unimplemented field {other}"),
            }
        }
        interp_ok(())
    }

    // Write fields for struct, enum variants
    fn write_variant_fields(
        &mut self,
        place: impl Writeable<'tcx, CtfeProvenance>,
        variant_def: &'tcx VariantDef,
        variant_layout: TyAndLayout<'tcx>,
        generics: &'tcx GenericArgs<'tcx>,
    ) -> InterpResult<'tcx> {
        self.allocate_fill_and_write_slice_ptr(
            place,
            variant_def.fields.len() as u64,
            |this, i, place| {
                let field_def = &variant_def.fields[FieldIdx::from_usize(i as usize)];
                let field_ty = field_def.ty(*this.tcx, generics);
                this.write_field(field_ty, place, variant_layout, Some(field_def.name), i)
            },
        )
    }

    fn write_generics(
        &mut self,
        place: impl Writeable<'tcx, CtfeProvenance>,
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
        let (variant_idx, _) = self.downcast(&place, sym::Lifetime)?;
        self.write_discriminant(variant_idx, &place)?;
        interp_ok(())
    }

    fn write_generic_type(&mut self, ty: Ty<'tcx>, place: MPlaceTy<'tcx>) -> InterpResult<'tcx> {
        let (variant_idx, variant_place) = self.downcast(&place, sym::Type)?;
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

        let (variant_idx, variant_place) = self.downcast(&place, sym::Const)?;
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
