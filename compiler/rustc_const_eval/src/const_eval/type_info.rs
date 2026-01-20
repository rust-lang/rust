use std::borrow::Cow;

use rustc_abi::{FieldIdx, VariantIdx};
use rustc_ast::Mutability;
use rustc_hir::LangItem;
use rustc_middle::mir::interpret::{CtfeProvenance, Scalar};
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{
    self, AdtDef, AdtKind, Const, ConstKind, GenericArgKind, GenericArgs, Region, ScalarInt, Ty,
    VariantDef,
};
use rustc_middle::{bug, span_bug};
use rustc_span::{Symbol, sym};

use crate::const_eval::CompileTimeMachine;
use crate::interpret::{
    Immediate, InterpCx, InterpResult, MPlaceTy, MemoryKind, Projectable, Writeable, interp_ok,
};

impl<'tcx> InterpCx<'tcx, CompileTimeMachine<'tcx>> {
    fn downcast(
        &self,
        place: &(impl Writeable<'tcx, CtfeProvenance> + 'tcx),
        name: Symbol,
    ) -> InterpResult<'tcx, (VariantIdx, impl Writeable<'tcx, CtfeProvenance> + 'tcx)> {
        let variants = place.layout().ty.ty_adt_def().unwrap().variants();
        let variant_id = variants
            .iter_enumerated()
            .find(|(_idx, var)| var.name == name)
            .unwrap_or_else(|| panic!("got {name} but expected one of {variants:#?}"))
            .0;

        interp_ok((variant_id, self.project_downcast(place, variant_id)?))
    }

    // A general method to write an array to a static slice place.
    fn project_write_array(
        &mut self,
        slice_place: impl Writeable<'tcx, CtfeProvenance>,
        len: u64,
        writer: impl Fn(&mut Self, /* index */ u64, MPlaceTy<'tcx>) -> InterpResult<'tcx>,
    ) -> InterpResult<'tcx> {
        // Array element type
        let field_ty = slice_place
            .layout()
            .ty
            .builtin_deref(false)
            .unwrap()
            .sequence_element_type(self.tcx.tcx);

        // Allocate an array
        let array_layout = self.layout_of(Ty::new_array(self.tcx.tcx, field_ty, len))?;
        let array_place = self.allocate(array_layout, MemoryKind::Stack)?;

        // Fill the array fields
        let mut field_places = self.project_array_fields(&array_place)?;
        while let Some((i, place)) = field_places.next(self)? {
            writer(self, i, place)?;
        }

        // Write the slice pointing to the array
        let array_place = array_place.map_provenance(CtfeProvenance::as_immutable);
        let ptr = Immediate::new_slice(array_place.ptr(), len, self);
        self.write_immediate(ptr, &slice_place)
    }

    /// Writes a `core::mem::type_info::TypeInfo` for a given type, `ty` to the given place.
    pub(crate) fn write_type_info(
        &mut self,
        ty: Ty<'tcx>,
        dest: &(impl Writeable<'tcx, CtfeProvenance> + 'tcx),
    ) -> InterpResult<'tcx> {
        let ty_struct = self.tcx.require_lang_item(LangItem::Type, self.tcx.span);
        let ty_struct = self.tcx.type_of(ty_struct).no_bound_vars().unwrap();
        assert_eq!(ty_struct, dest.layout().ty);
        let ty_struct = ty_struct.ty_adt_def().unwrap().non_enum_variant();
        // Fill all fields of the `TypeInfo` struct.
        for (idx, field) in ty_struct.fields.iter_enumerated() {
            let field_dest = self.project_field(dest, idx)?;
            let ptr_bit_width = || self.tcx.data_layout.pointer_size().bits();
            match field.name {
                sym::kind => {
                    let variant_index = match ty.kind() {
                        ty::Tuple(fields) => {
                            let (variant, variant_place) =
                                self.downcast(&field_dest, sym::Tuple)?;
                            // project to the single tuple variant field of `type_info::Tuple` struct type
                            let tuple_place = self.project_field(&variant_place, FieldIdx::ZERO)?;
                            assert_eq!(
                                1,
                                tuple_place
                                    .layout()
                                    .ty
                                    .ty_adt_def()
                                    .unwrap()
                                    .non_enum_variant()
                                    .fields
                                    .len()
                            );
                            self.write_tuple_fields(tuple_place, fields, ty)?;
                            variant
                        }
                        ty::Array(ty, len) => {
                            let (variant, variant_place) =
                                self.downcast(&field_dest, sym::Array)?;
                            let array_place = self.project_field(&variant_place, FieldIdx::ZERO)?;

                            self.write_array_type_info(array_place, *ty, *len)?;

                            variant
                        }
                        ty::Adt(adt_def, generics) => {
                            // TODO(type_info): Handle union
                            if !adt_def.is_struct() && !adt_def.is_enum() {
                                self.downcast(&field_dest, sym::Other)?.0
                            } else {
                                self.write_adt_type_info(&field_dest, (ty, *adt_def), generics)?
                            }
                        }
                        ty::Bool => {
                            let (variant, _variant_place) =
                                self.downcast(&field_dest, sym::Bool)?;
                            variant
                        }
                        ty::Char => {
                            let (variant, _variant_place) =
                                self.downcast(&field_dest, sym::Char)?;
                            variant
                        }
                        ty::Int(int_ty) => {
                            let (variant, variant_place) = self.downcast(&field_dest, sym::Int)?;
                            let place = self.project_field(&variant_place, FieldIdx::ZERO)?;
                            self.write_int_type_info(
                                place,
                                int_ty.bit_width().unwrap_or_else(/* isize */ ptr_bit_width),
                                true,
                            )?;
                            variant
                        }
                        ty::Uint(uint_ty) => {
                            let (variant, variant_place) = self.downcast(&field_dest, sym::Int)?;
                            let place = self.project_field(&variant_place, FieldIdx::ZERO)?;
                            self.write_int_type_info(
                                place,
                                uint_ty.bit_width().unwrap_or_else(/* usize */ ptr_bit_width),
                                false,
                            )?;
                            variant
                        }
                        ty::Float(float_ty) => {
                            let (variant, variant_place) =
                                self.downcast(&field_dest, sym::Float)?;
                            let place = self.project_field(&variant_place, FieldIdx::ZERO)?;
                            self.write_float_type_info(place, float_ty.bit_width())?;
                            variant
                        }
                        ty::Str => {
                            let (variant, _variant_place) = self.downcast(&field_dest, sym::Str)?;
                            variant
                        }
                        ty::Ref(_, ty, mutability) => {
                            let (variant, variant_place) =
                                self.downcast(&field_dest, sym::Reference)?;
                            let reference_place =
                                self.project_field(&variant_place, FieldIdx::ZERO)?;
                            self.write_reference_type_info(reference_place, *ty, *mutability)?;

                            variant
                        }
                        ty::Foreign(_)
                        | ty::Pat(_, _)
                        | ty::Slice(_)
                        | ty::RawPtr(..)
                        | ty::FnDef(..)
                        | ty::FnPtr(..)
                        | ty::UnsafeBinder(..)
                        | ty::Dynamic(..)
                        | ty::Closure(..)
                        | ty::CoroutineClosure(..)
                        | ty::Coroutine(..)
                        | ty::CoroutineWitness(..)
                        | ty::Never
                        | ty::Alias(..)
                        | ty::Param(_)
                        | ty::Bound(..)
                        | ty::Placeholder(_)
                        | ty::Infer(..)
                        | ty::Error(_) => self.downcast(&field_dest, sym::Other)?.0,
                    };
                    self.write_discriminant(variant_index, &field_dest)?
                }
                sym::size => {
                    let layout = self.layout_of(ty)?;
                    let variant_index = if layout.is_sized() {
                        let (variant, variant_place) = self.downcast(&field_dest, sym::Some)?;
                        let size_field_place =
                            self.project_field(&variant_place, FieldIdx::ZERO)?;
                        self.write_scalar(
                            ScalarInt::try_from_target_usize(layout.size.bytes(), self.tcx.tcx)
                                .unwrap(),
                            &size_field_place,
                        )?;
                        variant
                    } else {
                        self.downcast(&field_dest, sym::None)?.0
                    };
                    self.write_discriminant(variant_index, &field_dest)?;
                }
                other => span_bug!(self.tcx.span, "unknown `Type` field {other}"),
            }
        }

        interp_ok(())
    }

    // TODO(type_info): Remove this method, use `project_write_array` as it's more general.
    pub(crate) fn write_tuple_fields(
        &mut self,
        tuple_place: impl Writeable<'tcx, CtfeProvenance>,
        fields: &[Ty<'tcx>],
        tuple_ty: Ty<'tcx>,
    ) -> InterpResult<'tcx> {
        // project into the `type_info::Tuple::fields` field
        let fields_slice_place = self.project_field(&tuple_place, FieldIdx::ZERO)?;
        // get the `type_info::Field` type from `fields: &[Field]`
        let field_type = fields_slice_place
            .layout()
            .ty
            .builtin_deref(false)
            .unwrap()
            .sequence_element_type(self.tcx.tcx);
        // Create an array with as many elements as the number of fields in the inspected tuple
        let fields_layout =
            self.layout_of(Ty::new_array(self.tcx.tcx, field_type, fields.len() as u64))?;
        let fields_place = self.allocate(fields_layout, MemoryKind::Stack)?;
        let mut fields_places = self.project_array_fields(&fields_place)?;

        let tuple_layout = self.layout_of(tuple_ty)?;

        while let Some((i, place)) = fields_places.next(self)? {
            let field_ty = fields[i as usize];
            self.write_field(field_ty, place, tuple_layout, None, i)?;
        }

        let fields_place = fields_place.map_provenance(CtfeProvenance::as_immutable);

        let ptr = Immediate::new_slice(fields_place.ptr(), fields.len() as u64, self);

        self.write_immediate(ptr, &fields_slice_place)
    }

    // Write fields for struct, enum variants
    fn write_variant_fields(
        &mut self,
        place: impl Writeable<'tcx, CtfeProvenance>,
        variant_def: &'tcx VariantDef,
        variant_layout: TyAndLayout<'tcx>,
        generics: &'tcx GenericArgs<'tcx>,
    ) -> InterpResult<'tcx> {
        self.project_write_array(place, variant_def.fields.len() as u64, |this, i, place| {
            let field_def = &variant_def.fields[FieldIdx::from_usize(i as usize)];
            let field_ty = field_def.ty(*this.tcx, generics);
            this.write_field(field_ty, place, variant_layout, Some(field_def.name), i)
        })
    }

    fn write_field(
        &mut self,
        field_ty: Ty<'tcx>,
        place: MPlaceTy<'tcx>,
        layout: TyAndLayout<'tcx>,
        name: Option<Symbol>,
        idx: u64,
    ) -> InterpResult<'tcx> {
        for (field_idx, field_ty_field) in
            place.layout.ty.ty_adt_def().unwrap().non_enum_variant().fields.iter_enumerated()
        {
            let field_place = self.project_field(&place, field_idx)?;
            match field_ty_field.name {
                sym::name => {
                    let name = match name.as_ref() {
                        Some(name) => Cow::Borrowed(name.as_str()),
                        None => Cow::Owned(idx.to_string()), // For tuples
                    };
                    let name_place = self.allocate_str_dedup(&name)?;
                    let ptr = self.mplace_to_ref(&name_place)?;
                    self.write_immediate(*ptr, &field_place)?
                }
                sym::ty => self.write_type_id(field_ty, &field_place)?,
                sym::offset => {
                    let offset = layout.fields.offset(idx as usize);
                    self.write_scalar(
                        ScalarInt::try_from_target_usize(offset.bytes(), self.tcx.tcx).unwrap(),
                        &field_place,
                    )?;
                }
                other => {
                    span_bug!(self.tcx.def_span(field_ty_field.did), "unimplemented field {other}")
                }
            }
        }
        interp_ok(())
    }

    pub(crate) fn write_array_type_info(
        &mut self,
        place: impl Writeable<'tcx, CtfeProvenance>,
        ty: Ty<'tcx>,
        len: Const<'tcx>,
    ) -> InterpResult<'tcx> {
        // Iterate over all fields of `type_info::Array`.
        for (field_idx, field) in
            place.layout().ty.ty_adt_def().unwrap().non_enum_variant().fields.iter_enumerated()
        {
            let field_place = self.project_field(&place, field_idx)?;

            match field.name {
                // Write the `TypeId` of the array's elements to the `element_ty` field.
                sym::element_ty => self.write_type_id(ty, &field_place)?,
                // Write the length of the array to the `len` field.
                sym::len => self.write_scalar(len.to_leaf(), &field_place)?,
                other => span_bug!(self.tcx.def_span(field.did), "unimplemented field {other}"),
            }
        }

        interp_ok(())
    }

    // FIXME(type_info): No semver considerations for now
    pub(crate) fn write_adt_type_info(
        &mut self,
        place: &(impl Writeable<'tcx, CtfeProvenance> + 'tcx),
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
            AdtKind::Enum => {
                let (variant, variant_place) = self.downcast(place, sym::Enum)?;
                let place = self.project_field(&variant_place, FieldIdx::ZERO)?;
                self.write_enum_type_info(place, adt, generics)?;
                variant
            }
            AdtKind::Union => todo!(),
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
                // TODO(type_info): Write more info
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
                    self.project_write_array(
                        field_place,
                        enum_def.variants().len() as u64,
                        |this, i, place| {
                            let variant_idx = VariantIdx::from_usize(i as usize);
                            let variant_def = &enum_def.variants()[variant_idx];
                            let variant_layout = enum_layout.for_variant(this, variant_idx);
                            // TODO(type_info): Is it correct to use enum_ty here? If yes, leave some explanation.
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

    fn write_generics(
        &mut self,
        place: impl Writeable<'tcx, CtfeProvenance>,
        generics: &'tcx GenericArgs<'tcx>,
    ) -> InterpResult<'tcx> {
        self.project_write_array(place, generics.len() as u64, |this, i, place| {
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

    fn write_int_type_info(
        &mut self,
        place: impl Writeable<'tcx, CtfeProvenance>,
        bit_width: u64,
        signed: bool,
    ) -> InterpResult<'tcx> {
        for (field_idx, field) in
            place.layout().ty.ty_adt_def().unwrap().non_enum_variant().fields.iter_enumerated()
        {
            let field_place = self.project_field(&place, field_idx)?;
            match field.name {
                sym::bits => self.write_scalar(
                    Scalar::from_u32(bit_width.try_into().expect("bit_width overflowed")),
                    &field_place,
                )?,
                sym::signed => self.write_scalar(Scalar::from_bool(signed), &field_place)?,
                other => span_bug!(self.tcx.def_span(field.did), "unimplemented field {other}"),
            }
        }
        interp_ok(())
    }

    fn write_float_type_info(
        &mut self,
        place: impl Writeable<'tcx, CtfeProvenance>,
        bit_width: u64,
    ) -> InterpResult<'tcx> {
        for (field_idx, field) in
            place.layout().ty.ty_adt_def().unwrap().non_enum_variant().fields.iter_enumerated()
        {
            let field_place = self.project_field(&place, field_idx)?;
            match field.name {
                sym::bits => self.write_scalar(
                    Scalar::from_u32(bit_width.try_into().expect("bit_width overflowed")),
                    &field_place,
                )?,
                other => span_bug!(self.tcx.def_span(field.did), "unimplemented field {other}"),
            }
        }
        interp_ok(())
    }

    pub(crate) fn write_reference_type_info(
        &mut self,
        place: impl Writeable<'tcx, CtfeProvenance>,
        ty: Ty<'tcx>,
        mutability: Mutability,
    ) -> InterpResult<'tcx> {
        // Iterate over all fields of `type_info::Reference`.
        for (field_idx, field) in
            place.layout().ty.ty_adt_def().unwrap().non_enum_variant().fields.iter_enumerated()
        {
            let field_place = self.project_field(&place, field_idx)?;

            match field.name {
                // Write the `TypeId` of the reference's inner type to the `ty` field.
                sym::pointee => self.write_type_id(ty, &field_place)?,
                // Write the boolean representing the reference's mutability to the `mutable` field.
                sym::mutable => {
                    self.write_scalar(Scalar::from_bool(mutability.is_mut()), &field_place)?
                }
                other => span_bug!(self.tcx.def_span(field.did), "unimplemented field {other}"),
            }
        }
        interp_ok(())
    }
}
