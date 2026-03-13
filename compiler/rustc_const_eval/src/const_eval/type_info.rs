mod adt;

use std::borrow::Cow;

use rustc_abi::{ExternAbi, FieldIdx, VariantIdx};
use rustc_ast::Mutability;
use rustc_hir::LangItem;
use rustc_middle::span_bug;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{self, Const, FnHeader, FnSigTys, ScalarInt, Ty, TyCtxt};
use rustc_span::{Symbol, sym};

use crate::const_eval::CompileTimeMachine;
use crate::interpret::{
    CtfeProvenance, Immediate, InterpCx, InterpResult, MPlaceTy, MemoryKind, Projectable, Scalar,
    Writeable, interp_ok,
};

impl<'tcx> InterpCx<'tcx, CompileTimeMachine<'tcx>> {
    /// Equivalent to `project_downcast`, but identifies the variant by name instead of index.
    fn downcast<'a>(
        &self,
        place: &(impl Writeable<'tcx, CtfeProvenance> + 'a),
        name: Symbol,
    ) -> InterpResult<'tcx, (VariantIdx, impl Writeable<'tcx, CtfeProvenance> + 'a)> {
        let variants = place.layout().ty.ty_adt_def().unwrap().variants();
        let variant_idx = variants
            .iter_enumerated()
            .find(|(_idx, var)| var.name == name)
            .unwrap_or_else(|| panic!("got {name} but expected one of {variants:#?}"))
            .0;

        interp_ok((variant_idx, self.project_downcast(place, variant_idx)?))
    }

    // A general method to write an array to a static slice place.
    fn allocate_fill_and_write_slice_ptr(
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
        dest: &impl Writeable<'tcx, CtfeProvenance>,
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
                            self.write_tuple_type_info(tuple_place, fields, ty)?;
                            variant
                        }
                        ty::Array(ty, len) => {
                            let (variant, variant_place) =
                                self.downcast(&field_dest, sym::Array)?;
                            let array_place = self.project_field(&variant_place, FieldIdx::ZERO)?;

                            self.write_array_type_info(array_place, *ty, *len)?;

                            variant
                        }
                        ty::Slice(ty) => {
                            let (variant, variant_place) =
                                self.downcast(&field_dest, sym::Slice)?;
                            let slice_place = self.project_field(&variant_place, FieldIdx::ZERO)?;

                            self.write_slice_type_info(slice_place, *ty)?;

                            variant
                        }
                        ty::Adt(adt_def, generics) => {
                            self.write_adt_type_info(&field_dest, (ty, *adt_def), generics)?
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
                        ty::RawPtr(ty, mutability) => {
                            let (variant, variant_place) =
                                self.downcast(&field_dest, sym::Pointer)?;
                            let pointer_place =
                                self.project_field(&variant_place, FieldIdx::ZERO)?;

                            self.write_pointer_type_info(pointer_place, *ty, *mutability)?;

                            variant
                        }
                        ty::Dynamic(predicates, region) => {
                            let (variant, variant_place) =
                                self.downcast(&field_dest, sym::DynTrait)?;
                            let dyn_place = self.project_field(&variant_place, FieldIdx::ZERO)?;
                            self.write_dyn_trait_type_info(dyn_place, *predicates, *region)?;
                            variant
                        }
                        ty::FnPtr(sig, fn_header) => {
                            let (variant, variant_place) =
                                self.downcast(&field_dest, sym::FnPtr)?;
                            let fn_ptr_place =
                                self.project_field(&variant_place, FieldIdx::ZERO)?;

                            // FIXME: handle lifetime bounds
                            let sig = sig.skip_binder();

                            self.write_fn_ptr_type_info(fn_ptr_place, &sig, fn_header)?;
                            variant
                        }
                        ty::Foreign(_)
                        | ty::Pat(_, _)
                        | ty::FnDef(..)
                        | ty::UnsafeBinder(..)
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
                sym::ty => {
                    let field_ty = self.tcx.erase_and_anonymize_regions(field_ty);
                    self.write_type_id(field_ty, &field_place)?
                }
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

    pub(crate) fn write_tuple_type_info(
        &mut self,
        tuple_place: impl Writeable<'tcx, CtfeProvenance>,
        fields: &[Ty<'tcx>],
        tuple_ty: Ty<'tcx>,
    ) -> InterpResult<'tcx> {
        let tuple_layout = self.layout_of(tuple_ty)?;
        let fields_slice_place = self.project_field(&tuple_place, FieldIdx::ZERO)?;
        self.allocate_fill_and_write_slice_ptr(
            fields_slice_place,
            fields.len() as u64,
            |this, i, place| {
                let field_ty = fields[i as usize];
                this.write_field(field_ty, place, tuple_layout, None, i)
            },
        )
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

    pub(crate) fn write_slice_type_info(
        &mut self,
        place: impl Writeable<'tcx, CtfeProvenance>,
        ty: Ty<'tcx>,
    ) -> InterpResult<'tcx> {
        // Iterate over all fields of `type_info::Slice`.
        for (field_idx, field) in
            place.layout().ty.ty_adt_def().unwrap().non_enum_variant().fields.iter_enumerated()
        {
            let field_place = self.project_field(&place, field_idx)?;

            match field.name {
                // Write the `TypeId` of the slice's elements to the `element_ty` field.
                sym::element_ty => self.write_type_id(ty, &field_place)?,
                other => span_bug!(self.tcx.def_span(field.did), "unimplemented field {other}"),
            }
        }

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

    pub(crate) fn write_fn_ptr_type_info(
        &mut self,
        place: impl Writeable<'tcx, CtfeProvenance>,
        sig: &FnSigTys<TyCtxt<'tcx>>,
        fn_header: &FnHeader<TyCtxt<'tcx>>,
    ) -> InterpResult<'tcx> {
        let FnHeader { safety, c_variadic, abi } = fn_header;

        for (field_idx, field) in
            place.layout().ty.ty_adt_def().unwrap().non_enum_variant().fields.iter_enumerated()
        {
            let field_place = self.project_field(&place, field_idx)?;

            match field.name {
                sym::unsafety => {
                    self.write_scalar(Scalar::from_bool(safety.is_unsafe()), &field_place)?;
                }
                sym::abi => match abi {
                    ExternAbi::C { .. } => {
                        let (rust_variant, _rust_place) =
                            self.downcast(&field_place, sym::ExternC)?;
                        self.write_discriminant(rust_variant, &field_place)?;
                    }
                    ExternAbi::Rust => {
                        let (rust_variant, _rust_place) =
                            self.downcast(&field_place, sym::ExternRust)?;
                        self.write_discriminant(rust_variant, &field_place)?;
                    }
                    other_abi => {
                        let (variant, variant_place) = self.downcast(&field_place, sym::Named)?;
                        let str_place = self.allocate_str_dedup(other_abi.as_str())?;
                        let str_ref = self.mplace_to_ref(&str_place)?;
                        let payload = self.project_field(&variant_place, FieldIdx::ZERO)?;
                        self.write_immediate(*str_ref, &payload)?;
                        self.write_discriminant(variant, &field_place)?;
                    }
                },
                sym::inputs => {
                    let inputs = sig.inputs();
                    self.allocate_fill_and_write_slice_ptr(
                        field_place,
                        inputs.len() as _,
                        |this, i, place| this.write_type_id(inputs[i as usize], &place),
                    )?;
                }
                sym::output => {
                    let output = sig.output();
                    self.write_type_id(output, &field_place)?;
                }
                sym::variadic => {
                    self.write_scalar(Scalar::from_bool(*c_variadic), &field_place)?;
                }
                other => span_bug!(self.tcx.def_span(field.did), "unimplemented field {other}"),
            }
        }

        interp_ok(())
    }

    pub(crate) fn write_pointer_type_info(
        &mut self,
        place: impl Writeable<'tcx, CtfeProvenance>,
        ty: Ty<'tcx>,
        mutability: Mutability,
    ) -> InterpResult<'tcx> {
        // Iterate over all fields of `type_info::Pointer`.
        for (field_idx, field) in
            place.layout().ty.ty_adt_def().unwrap().non_enum_variant().fields.iter_enumerated()
        {
            let field_place = self.project_field(&place, field_idx)?;

            match field.name {
                // Write the `TypeId` of the pointer's inner type to the `ty` field.
                sym::pointee => self.write_type_id(ty, &field_place)?,
                // Write the boolean representing the pointer's mutability to the `mutable` field.
                sym::mutable => {
                    self.write_scalar(Scalar::from_bool(mutability.is_mut()), &field_place)?
                }
                other => span_bug!(self.tcx.def_span(field.did), "unimplemented field {other}"),
            }
        }

        interp_ok(())
    }
}
