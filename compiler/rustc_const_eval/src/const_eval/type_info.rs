use rustc_abi::FieldIdx;
use rustc_hir::LangItem;
use rustc_middle::mir::interpret::CtfeProvenance;
use rustc_middle::span_bug;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{self, Const, ScalarInt, Ty};
use rustc_span::{Symbol, sym};

use crate::const_eval::CompileTimeMachine;
use crate::interpret::{
    Immediate, InterpCx, InterpResult, MPlaceTy, MemoryKind, Writeable, interp_ok,
};

impl<'tcx> InterpCx<'tcx, CompileTimeMachine<'tcx>> {
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
            let downcast = |name: Symbol| {
                let variants = field_dest.layout().ty.ty_adt_def().unwrap().variants();
                let variant_id = variants
                    .iter_enumerated()
                    .find(|(_idx, var)| var.name == name)
                    .unwrap_or_else(|| panic!("got {name} but expected one of {variants:#?}"))
                    .0;

                interp_ok((variant_id, self.project_downcast(&field_dest, variant_id)?))
            };
            match field.name {
                sym::kind => {
                    let variant_index = match ty.kind() {
                        ty::Tuple(fields) => {
                            let (variant, variant_place) = downcast(sym::Tuple)?;
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
                            let (variant, variant_place) = downcast(sym::Array)?;
                            let array_place = self.project_field(&variant_place, FieldIdx::ZERO)?;

                            self.write_array_type_info(array_place, *ty, *len)?;

                            variant
                        }
                        // For now just merge all primitives into one `Leaf` variant with no data
                        ty::Uint(_) | ty::Int(_) | ty::Float(_) | ty::Char | ty::Bool => {
                            downcast(sym::Leaf)?.0
                        }
                        ty::Adt(_, _)
                        | ty::Foreign(_)
                        | ty::Str
                        | ty::Pat(_, _)
                        | ty::Slice(_)
                        | ty::RawPtr(..)
                        | ty::Ref(..)
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
                        | ty::Error(_) => downcast(sym::Other)?.0,
                    };
                    self.write_discriminant(variant_index, &field_dest)?
                }
                sym::size => {
                    let layout = self.layout_of(ty)?;
                    let variant_index = if layout.is_sized() {
                        let (variant, variant_place) = downcast(sym::Some)?;
                        let size_field_place =
                            self.project_field(&variant_place, FieldIdx::ZERO)?;
                        self.write_scalar(
                            ScalarInt::try_from_target_usize(layout.size.bytes(), self.tcx.tcx)
                                .unwrap(),
                            &size_field_place,
                        )?;
                        variant
                    } else {
                        downcast(sym::None)?.0
                    };
                    self.write_discriminant(variant_index, &field_dest)?;
                }
                other => span_bug!(self.tcx.span, "unknown `Type` field {other}"),
            }
        }

        interp_ok(())
    }

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
            self.write_field(field_ty, place, tuple_layout, i)?;
        }

        let fields_place = fields_place.map_provenance(CtfeProvenance::as_immutable);

        let ptr = Immediate::new_slice(fields_place.ptr(), fields.len() as u64, self);

        self.write_immediate(ptr, &fields_slice_place)
    }

    fn write_field(
        &mut self,
        field_ty: Ty<'tcx>,
        place: MPlaceTy<'tcx>,
        layout: TyAndLayout<'tcx>,
        idx: u64,
    ) -> InterpResult<'tcx> {
        for (field_idx, field_ty_field) in
            place.layout.ty.ty_adt_def().unwrap().non_enum_variant().fields.iter_enumerated()
        {
            let field_place = self.project_field(&place, field_idx)?;
            match field_ty_field.name {
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
}
