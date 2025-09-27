use rustc_abi::FieldIdx;
use rustc_hir::LangItem;
use rustc_middle::mir::interpret::CtfeProvenance;
use rustc_middle::span_bug;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{self, ScalarInt, Ty};
use rustc_span::{Symbol, sym};

use crate::const_eval::CompileTimeMachine;
use crate::interpret::{InterpCx, InterpResult, MPlaceTy, MemoryKind, Writeable, interp_ok};

impl<'tcx> InterpCx<'tcx, CompileTimeMachine<'tcx>> {
    pub(crate) fn write_type_info(
        &mut self,
        ty: Ty<'tcx>,
        dest: &impl Writeable<'tcx, CtfeProvenance>,
    ) -> InterpResult<'tcx> {
        let ty_struct = self.tcx.require_lang_item(LangItem::Type, self.tcx.span);
        let ty_struct = self.tcx.type_of(ty_struct).instantiate_identity();
        assert_eq!(ty_struct, dest.layout().ty);
        let ty_struct = ty_struct.ty_adt_def().unwrap().non_enum_variant();
        for (idx, field) in ty_struct.fields.iter_enumerated() {
            let field_dest = self.project_field(dest, idx)?;
            let downcast = |name: Symbol| {
                let variant_id = field_dest
                    .layout()
                    .ty
                    .ty_adt_def()
                    .unwrap()
                    .variants()
                    .iter_enumerated()
                    .find(|(_idx, var)| var.name == name)
                    .unwrap()
                    .0;

                interp_ok((variant_id, self.project_downcast(&field_dest, variant_id)?))
            };
            match field.name {
                sym::kind => {
                    let variant_index = match ty.kind() {
                        ty::Tuple(fields) => {
                            let (variant, variant_place) = downcast(sym::Tuple)?;
                            // `Tuple` struct
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
                        ty::Uint(_) | ty::Int(_) | ty::Float(_) | ty::Char | ty::Bool => {
                            downcast(sym::Leaf)?.0
                        }
                        ty::Adt(_, _)
                        | ty::Foreign(_)
                        | ty::Str
                        | ty::Array(_, _)
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
        // `fields` field
        let fields_slice_place = self.project_field(&tuple_place, FieldIdx::ZERO)?;
        let field_type = fields_slice_place
            .layout()
            .ty
            .builtin_deref(false)
            .unwrap()
            .sequence_element_type(self.tcx.tcx);
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

        let mut ptr = self.mplace_to_ref(&fields_place)?;
        ptr.layout = self.layout_of(Ty::new_imm_ref(
            self.tcx.tcx,
            self.tcx.lifetimes.re_static,
            fields_layout.ty,
        ))?;

        let slice_type = Ty::new_imm_ref(
            self.tcx.tcx,
            self.tcx.lifetimes.re_static,
            Ty::new_slice(self.tcx.tcx, field_type),
        );
        let slice_type = self.layout_of(slice_type)?;
        self.unsize_into(&ptr.into(), slice_type, &fields_slice_place)?;
        interp_ok(())
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
}
