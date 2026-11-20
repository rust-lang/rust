use rustc_abi::{FIRST_VARIANT, FieldIdx, VariantIdx};
use rustc_ast::Mutability;
use rustc_attr_ir::HasAttrs;
use rustc_middle::mir;
use rustc_middle::ty::layout::{HasTyCtxt, HasTypingEnv};
use rustc_middle::ty::{self, FieldInfo, ScalarInt, Ty};
use rustc_span::{Symbol, span_bug, sym};

use crate::const_eval::CompileTimeMachine;
use crate::interpret::{
    CtfeProvenance, Immediate, InterpCx, InterpResult, OpTy, PlaceTy, Scalar, interp_ok, throw_ub,
    type_implements_dyn_trait,
};

impl<'tcx> InterpCx<'tcx, CompileTimeMachine<'tcx>> {
    /// intrinsics related to type_info
    pub(crate) fn call_type_info_intrinsic(
        &mut self,
        intrinsic_name: Symbol,
        _instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: &PlaceTy<'tcx, CtfeProvenance>,
        target: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx, bool> {
        match intrinsic_name {
            sym::type_id_vtable => {
                let tp_ty = self.read_type_id(&args[0])?;
                let result_ty = self.read_type_id(&args[1])?;

                let (implements_trait, preds) = type_implements_dyn_trait(self, tp_ty, result_ty)?;

                if implements_trait {
                    let vtable_ptr = self.get_vtable_ptr(tp_ty, preds)?;
                    // Writing a non-null pointer into an `Option<NonNull>` will automatically make it `Some`.
                    self.write_pointer(vtable_ptr, dest)?;
                } else {
                    // Write `None`
                    self.write_discriminant(FIRST_VARIANT, dest)?;
                }
            }

            sym::type_id_type_of => {
                let ty = self.read_type_id(&args[0])?;
                self.write_type_kind(ty, dest)?;
            }

            sym::type_id_is_signed => {
                let ty = self.read_type_id(&args[0])?;
                self.write_scalar(Scalar::from_bool(ty.is_signed()), dest)?;
            }

            sym::type_id_points_mutably => {
                let ty: Ty<'tcx> = self.read_type_id(&args[0])?;
                let is_mutable = matches!(
                    ty.kind(),
                    ty::RawPtr(_, Mutability::Mut) | &ty::Ref(_, _, Mutability::Mut)
                );
                self.write_scalar(Scalar::from_bool(is_mutable), dest)?;
            }

            sym::type_id_size_of => {
                let ty = self.read_type_id(&args[0])?;
                let layout = self.layout_of(ty)?;

                self.write_option(layout.is_sized().then(|| ()), dest, |ecx, field, _| {
                    ecx.write_scalar(
                        ScalarInt::try_from_target_usize(layout.size.bytes(), ecx.tcx.tcx).unwrap(),
                        &field,
                    )
                })?
            }

            sym::type_id_element_ty => {
                let ty = self.read_type_id(&args[0])?;

                self.write_option(
                    match ty.kind() {
                        ty::Array(ty, _) | ty::Slice(ty) => Some(ty),
                        _ => None,
                    },
                    dest,
                    |ecx, place, ty| ecx.write_type_id(*ty, &place),
                )?
            }

            sym::type_id_array_len => {
                let ty = self.read_type_id(&args[0])?;
                let len = if let ty::Array(_, len) = ty.kind() {
                    len.to_leaf().to_target_usize(self.tcx.tcx())
                } else {
                    0
                };
                self.write_scalar(Scalar::from_target_usize(len, self), dest)?;
            }

            sym::type_id_fields => {
                let ty = self.read_type_id(&args[0])?;
                let variant_idx = self.read_target_usize(&args[1])? as usize;

                let variants_num =
                    ty.ty_adt_def().map(|adt_def| adt_def.variants().len()).unwrap_or(1);
                if variant_idx >= variants_num {
                    throw_ub!(BoundsCheckFailed {
                        len: variants_num as u64,
                        index: variant_idx as u64
                    });
                }

                let fields_num = match ty.kind() {
                    ty::Adt(adt_def, _) => {
                        let variant_def = &adt_def.variants()[VariantIdx::from_usize(variant_idx)];
                        variant_def.fields.len()
                    }
                    ty::Tuple(fields) => fields.len(),
                    _ => 0, // Other types have no fields
                };

                self.write_scalar(Scalar::from_target_usize(fields_num as u64, self), dest)?;
            }

            sym::type_id_field_representing_type => {
                let ty = self.read_type_id(&args[0])?;
                let variant_idx = self.read_target_usize(&args[1])? as usize;
                let field_idx = self.read_target_usize(&args[2])? as usize;

                let variants_num =
                    ty.ty_adt_def().map(|adt_def| adt_def.variants().len()).unwrap_or(1);
                if variant_idx >= variants_num {
                    throw_ub!(BoundsCheckFailed {
                        len: variants_num as u64,
                        index: variant_idx as u64
                    });
                }

                let fields_num = match ty.kind() {
                    ty::Adt(adt_def, _) => {
                        let variant_def = &adt_def.variants()[VariantIdx::from_usize(variant_idx)];
                        variant_def.fields.len()
                    }
                    ty::Tuple(fields) => fields.len(),
                    _ => 0, // Other types have no fields
                };
                if field_idx >= fields_num {
                    throw_ub!(BoundsCheckFailed {
                        len: fields_num as u64,
                        index: field_idx as u64
                    });
                }

                let frt = Ty::new_field_representing_type(
                    *self.tcx,
                    ty,
                    VariantIdx::from_usize(variant_idx),
                    FieldIdx::from_usize(field_idx),
                );
                self.write_type_id(frt, dest)?;
            }

            sym::type_id_function_ptr => {
                let ty = self.read_type_id(&args[0])?;

                self.write_option(
                    match ty.kind() {
                        ty::FnPtr(sig, fn_header) => Some((sig, fn_header)),
                        _ => None,
                    },
                    dest,
                    |ecx, place, (sig, fn_header)| {
                        let sig = sig.skip_binder(); // FIXME: handle lifetime bounds
                        ecx.write_fn_ptr_type_info(place, &sig, fn_header)
                    },
                )?
            }

            sym::type_id_dyn_trait => {
                let ty: Ty<'tcx> = self.read_type_id(&args[0])?;
                self.write_option(
                    match ty.kind() {
                        ty::Dynamic(p, r) => Some((p, r)),
                        _ => None,
                    },
                    dest,
                    |ecx, place, (p, r)| ecx.write_dyn_trait_type_info(place, *p, *r),
                )?
            }

            sym::type_id_points_to => {
                let ty = self.read_type_id(&args[0])?;
                self.write_option(
                    match ty.kind() {
                        ty::RawPtr(pointee_ty, _) | ty::Ref(_, pointee_ty, _) => Some(pointee_ty),
                        _ => None,
                    },
                    dest,
                    |ecx, place, pointee_ty| ecx.write_type_id(*pointee_ty, &place),
                )?
            }

            sym::type_id_variants => {
                let ty = self.read_type_id(&args[0])?;
                let variants_num = ty.ty_adt_def().map(|def| def.variants().len()).unwrap_or(1);
                self.write_scalar(Scalar::from_target_usize(variants_num as u64, self), dest)?;
            }

            sym::type_id_variant_name => {
                let base = self.read_type_id(&args[0])?;

                let field_name = if let ty::Adt(def, _) = base.kind() {
                    let variant_idx = self.read_target_usize(&args[1])? as usize;
                    if variant_idx >= def.variants().len() {
                        throw_ub!(BoundsCheckFailed {
                            len: def.variants().len() as u64,
                            index: variant_idx as u64
                        });
                    }
                    let variant_idx = VariantIdx::from_usize(variant_idx);
                    def.variant(variant_idx).name
                } else {
                    span_bug!(self.cur_span(), "expected enum type, got {base}")
                };
                let ptr = self.allocate_bytes_dedup(field_name.as_str().as_bytes())?;
                self.write_immediate(
                    Immediate::ScalarPair(
                        Scalar::from_pointer(ptr, self),
                        Scalar::from_target_usize(field_name.as_str().len() as u64, self),
                    ),
                    dest,
                )?;
            }

            sym::type_id_variant_reflected_attrs => {
                let base = self.read_type_id(&args[0])?;

                if let ty::Adt(def, _) = base.kind() {
                    let variant_idx = self.read_target_usize(&args[1])? as usize;
                    if variant_idx >= def.variants().len() {
                        throw_ub!(BoundsCheckFailed {
                            len: def.variants().len() as u64,
                            index: variant_idx as u64
                        });
                    }
                    let variant_idx = VariantIdx::from_usize(variant_idx);

                    self.write_reflect_attributes(
                        dest,
                        def.variant(variant_idx).def_id.get_attrs(&self.tcx),
                    )?;
                } else {
                    span_bug!(self.cur_span(), "expected enum type, got {base}")
                };
            }

            sym::type_id_variant_non_exhaustive => {
                let base = self.read_type_id(&args[0])?;

                let non_exhaustive = if let ty::Adt(def, _) = base.kind() {
                    let variant_idx = self.read_target_usize(&args[1])? as usize;
                    if variant_idx >= def.variants().len() {
                        throw_ub!(BoundsCheckFailed {
                            len: def.variants().len() as u64,
                            index: variant_idx as u64
                        });
                    }
                    let variant_idx = VariantIdx::from_usize(variant_idx);
                    def.variant(variant_idx).is_field_list_non_exhaustive()
                } else {
                    span_bug!(self.cur_span(), "expected enum type, got {base}")
                };
                self.write_scalar(Scalar::from_bool(non_exhaustive), dest)?;
            }

            sym::field_representing_type_name => {
                let frt_ty = self.read_type_id(&args[0])?;

                let field_name = if let ty::Adt(def, args) = frt_ty.kind()
                    && let Some(FieldInfo { name, .. }) =
                        def.field_representing_type_info(self.tcx.tcx, *args)
                {
                    name
                } else {
                    span_bug!(self.cur_span(), "expected field representing type, got {frt_ty}")
                };
                let ptr = self.allocate_bytes_dedup(field_name.as_str().as_bytes())?;
                self.write_immediate(
                    Immediate::ScalarPair(
                        Scalar::from_pointer(ptr, self),
                        Scalar::from_target_usize(field_name.as_str().len() as u64, self),
                    ),
                    dest,
                )?;
            }

            sym::field_representing_type_offset => {
                let frt_ty = self.read_type_id(&args[0])?;

                let (ty, variant, field) = if let ty::Adt(def, args) = frt_ty.kind()
                    && let Some(FieldInfo { base, variant_idx, field_idx, .. }) =
                        def.field_representing_type_info(self.tcx.tcx, *args)
                {
                    (base, variant_idx, field_idx)
                } else {
                    span_bug!(self.cur_span(), "expected field representing type, got {frt_ty}")
                };
                let layout = self.layout_of(ty)?;
                let cx = ty::layout::LayoutCx::new(self.tcx.tcx, self.typing_env());

                let layout = layout.for_variant(&cx, variant);
                let offset = layout.fields.offset(field.index()).bytes();

                self.write_scalar(Scalar::from_target_usize(offset, self), dest)?;
            }

            sym::field_representing_type_actual_type_id => {
                let frt_ty = self.read_type_id(&args[0])?;

                let field_ty = if let ty::Adt(def, args) = frt_ty.kind()
                    && let Some(FieldInfo { ty, .. }) =
                        def.field_representing_type_info(self.tcx.tcx, *args)
                {
                    self.tcx.erase_and_anonymize_regions(ty)
                } else {
                    span_bug!(self.cur_span(), "expected field representing type, got {frt_ty}")
                };
                self.write_type_id(field_ty, dest)?;
            }

            sym::type_id_reflected_attrs => {
                let ty = self.read_type_id(&args[0])?;
                let attrs = match ty.kind() {
                    ty::Adt(def, args)
                        if let Some(FieldInfo { base, variant_idx, field_idx, .. }) =
                            def.field_representing_type_info(self.tcx(), *args) =>
                    {
                        match base.kind() {
                            ty::Adt(base_def, _) => {
                                let field = &base_def.variants()[variant_idx].fields[field_idx];
                                field.did.get_attrs(&self.tcx)
                            }
                            ty::Tuple(_) => &[],
                            _ => {
                                // only adts and tuples have fields
                                span_bug!(self.cur_span(), "expected adt or tuple type, got {ty}");
                            }
                        }
                    }
                    ty::Adt(def, _) => def.did().get_attrs(&self.tcx),
                    _ => &[],
                };

                self.write_reflect_attributes(dest, attrs)?;
            }

            sym::type_id_docs => {
                let ty = self.read_type_id(&args[0])?;
                let attrs = match ty.kind() {
                    ty::Adt(def, args)
                        if let Some(FieldInfo { base, variant_idx, field_idx, .. }) =
                            def.field_representing_type_info(self.tcx(), *args) =>
                    {
                        match base.kind() {
                            ty::Adt(base_def, _) => {
                                let field = &base_def.variants()[variant_idx].fields[field_idx];
                                Some(field.did)
                            }
                            ty::Tuple(_) => None,
                            _ => {
                                // only adts and tuples have fields
                                span_bug!(self.cur_span(), "expected adt or tuple type, got {ty}");
                            }
                        }
                    }
                    ty::Adt(def, _) => Some(def.did()),
                    _ => None,
                };

                self.write_doc_attributes(dest, attrs)?;
            }

            sym::type_id_variant_docs => {
                let base = self.read_type_id(&args[0])?;

                if let ty::Adt(def, _) = base.kind() {
                    let variant_idx = self.read_target_usize(&args[1])? as usize;
                    if variant_idx >= def.variants().len() {
                        throw_ub!(BoundsCheckFailed {
                            len: def.variants().len() as u64,
                            index: variant_idx as u64
                        });
                    }
                    let var = def.variant(VariantIdx::from_usize(variant_idx)).def_id;
                    self.write_doc_attributes(dest, Some(var))?;
                } else {
                    span_bug!(self.cur_span(), "expected enum type, got {base}")
                };
            }

            sym::type_id_generics => {
                let ty = self.read_type_id(&args[0])?;
                self.write_type_id_generics(dest, ty)?;
            }

            sym::type_id_non_exhaustive => {
                let ty = self.read_type_id(&args[0])?;

                // FIXME(reflection): need a way to obtain non-exhaustiveness of a variant's fields.
                let non_exhaustive = if let ty::Adt(def, _) = ty.kind() {
                    if def.is_enum() {
                        def.is_variant_list_non_exhaustive()
                    } else {
                        def.non_enum_variant().is_field_list_non_exhaustive()
                    }
                } else {
                    false
                };

                self.write_scalar(Scalar::from_bool(non_exhaustive), dest)?;
            }
            _ => return interp_ok(false),
        }

        self.return_to_block(target)?;
        interp_ok(true)
    }
}
