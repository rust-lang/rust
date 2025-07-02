use std::assert_matches::assert_matches;

use rustc_abi::{FieldIdx, Integer};
use rustc_apfloat::ieee::{Double, Half, Quad, Single};
use rustc_apfloat::{Float, FloatConvert};
use rustc_middle::mir::CastKind;
use rustc_middle::mir::interpret::{InterpResult, PointerArithmetic, Scalar};
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::layout::{IntegerExt, TyAndLayout};
use rustc_middle::ty::{self, FloatTy, Ty};
use rustc_middle::{bug, span_bug};
use tracing::trace;

use super::util::ensure_monomorphic_enough;
use super::{
    FnVal, ImmTy, Immediate, InterpCx, Machine, OpTy, PlaceTy, err_inval, interp_ok, throw_ub,
    throw_ub_custom,
};
use crate::fluent_generated as fluent;

impl<'tcx, M: Machine<'tcx>> InterpCx<'tcx, M> {
    pub fn cast(
        &mut self,
        src: &OpTy<'tcx, M::Provenance>,
        cast_kind: CastKind,
        cast_ty: Ty<'tcx>,
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        // `cast_ty` will often be the same as `dest.ty`, but not always, since subtyping is still
        // possible.
        let cast_layout =
            if cast_ty == dest.layout.ty { dest.layout } else { self.layout_of(cast_ty)? };
        // FIXME: In which cases should we trigger UB when the source is uninit?
        match cast_kind {
            CastKind::PointerCoercion(PointerCoercion::Unsize, _) => {
                self.unsize_into(src, cast_layout, dest)?;
            }

            CastKind::PointerExposeProvenance => {
                let src = self.read_immediate(src)?;
                let res = self.pointer_expose_provenance_cast(&src, cast_layout)?;
                self.write_immediate(*res, dest)?;
            }

            CastKind::PointerWithExposedProvenance => {
                let src = self.read_immediate(src)?;
                let res = self.pointer_with_exposed_provenance_cast(&src, cast_layout)?;
                self.write_immediate(*res, dest)?;
            }

            CastKind::IntToInt | CastKind::IntToFloat => {
                let src = self.read_immediate(src)?;
                let res = self.int_to_int_or_float(&src, cast_layout)?;
                self.write_immediate(*res, dest)?;
            }

            CastKind::FloatToFloat | CastKind::FloatToInt => {
                let src = self.read_immediate(src)?;
                let res = self.float_to_float_or_int(&src, cast_layout)?;
                self.write_immediate(*res, dest)?;
            }

            CastKind::FnPtrToPtr | CastKind::PtrToPtr => {
                let src = self.read_immediate(src)?;
                let res = self.ptr_to_ptr(&src, cast_layout)?;
                self.write_immediate(*res, dest)?;
            }

            CastKind::PointerCoercion(
                PointerCoercion::MutToConstPointer | PointerCoercion::ArrayToPointer,
                _,
            ) => {
                bug!("{cast_kind:?} casts are for borrowck only, not runtime MIR");
            }

            CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer, _) => {
                // All reifications must be monomorphic, bail out otherwise.
                ensure_monomorphic_enough(*self.tcx, src.layout.ty)?;

                // The src operand does not matter, just its type
                match *src.layout.ty.kind() {
                    ty::FnDef(def_id, args) => {
                        let instance = ty::Instance::resolve_for_fn_ptr(
                            *self.tcx,
                            self.typing_env,
                            def_id,
                            args,
                        )
                        .ok_or_else(|| err_inval!(TooGeneric))?;

                        let fn_ptr = self.fn_ptr(FnVal::Instance(instance));
                        self.write_pointer(fn_ptr, dest)?;
                    }
                    _ => span_bug!(self.cur_span(), "reify fn pointer on {}", src.layout.ty),
                }
            }

            CastKind::PointerCoercion(PointerCoercion::UnsafeFnPointer, _) => {
                let src = self.read_immediate(src)?;
                match cast_ty.kind() {
                    ty::FnPtr(..) => {
                        // No change to value
                        self.write_immediate(*src, dest)?;
                    }
                    _ => span_bug!(self.cur_span(), "fn to unsafe fn cast on {}", cast_ty),
                }
            }

            CastKind::PointerCoercion(PointerCoercion::ClosureFnPointer(_), _) => {
                // All reifications must be monomorphic, bail out otherwise.
                ensure_monomorphic_enough(*self.tcx, src.layout.ty)?;

                // The src operand does not matter, just its type
                match *src.layout.ty.kind() {
                    ty::Closure(def_id, args) => {
                        let instance = ty::Instance::resolve_closure(
                            *self.tcx,
                            def_id,
                            args,
                            ty::ClosureKind::FnOnce,
                        );
                        let fn_ptr = self.fn_ptr(FnVal::Instance(instance));
                        self.write_pointer(fn_ptr, dest)?;
                    }
                    _ => span_bug!(self.cur_span(), "closure fn pointer on {}", src.layout.ty),
                }
            }

            CastKind::Transmute => {
                assert!(src.layout.is_sized());
                assert!(dest.layout.is_sized());
                assert_eq!(cast_ty, dest.layout.ty); // we otherwise ignore `cast_ty` enirely...
                if src.layout.size != dest.layout.size {
                    throw_ub_custom!(
                        fluent::const_eval_invalid_transmute,
                        src_bytes = src.layout.size.bytes(),
                        dest_bytes = dest.layout.size.bytes(),
                        src = src.layout.ty,
                        dest = dest.layout.ty,
                    );
                }

                self.copy_op_allow_transmute(src, dest)?;
            }
        }
        interp_ok(())
    }

    /// Handles 'IntToInt' and 'IntToFloat' casts.
    pub fn int_to_int_or_float(
        &self,
        src: &ImmTy<'tcx, M::Provenance>,
        cast_to: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        assert!(src.layout.ty.is_integral() || src.layout.ty.is_char() || src.layout.ty.is_bool());
        assert!(cast_to.ty.is_floating_point() || cast_to.ty.is_integral() || cast_to.ty.is_char());

        interp_ok(ImmTy::from_scalar(
            self.cast_from_int_like(src.to_scalar(), src.layout, cast_to.ty)?,
            cast_to,
        ))
    }

    /// Handles 'FloatToFloat' and 'FloatToInt' casts.
    pub fn float_to_float_or_int(
        &self,
        src: &ImmTy<'tcx, M::Provenance>,
        cast_to: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        let ty::Float(fty) = src.layout.ty.kind() else {
            bug!("FloatToFloat/FloatToInt cast: source type {} is not a float type", src.layout.ty)
        };
        let val = match fty {
            FloatTy::F16 => self.cast_from_float(src.to_scalar().to_f16()?, cast_to.ty),
            FloatTy::F32 => self.cast_from_float(src.to_scalar().to_f32()?, cast_to.ty),
            FloatTy::F64 => self.cast_from_float(src.to_scalar().to_f64()?, cast_to.ty),
            FloatTy::F128 => self.cast_from_float(src.to_scalar().to_f128()?, cast_to.ty),
        };
        interp_ok(ImmTy::from_scalar(val, cast_to))
    }

    /// Handles 'FnPtrToPtr' and 'PtrToPtr' casts.
    pub fn ptr_to_ptr(
        &self,
        src: &ImmTy<'tcx, M::Provenance>,
        cast_to: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        assert!(src.layout.ty.is_any_ptr());
        assert!(cast_to.ty.is_raw_ptr());
        // Handle casting any ptr to raw ptr (might be a wide ptr).
        if cast_to.size == src.layout.size {
            // Thin or wide pointer that just has the ptr kind of target type changed.
            return interp_ok(ImmTy::from_immediate(**src, cast_to));
        } else {
            // Casting the metadata away from a wide ptr.
            assert_eq!(src.layout.size, 2 * self.pointer_size());
            assert_eq!(cast_to.size, self.pointer_size());
            assert!(src.layout.ty.is_raw_ptr());
            return match **src {
                Immediate::ScalarPair(data, _) => interp_ok(ImmTy::from_scalar(data, cast_to)),
                Immediate::Scalar(..) => span_bug!(
                    self.cur_span(),
                    "{:?} input to a fat-to-thin cast ({} -> {})",
                    *src,
                    src.layout.ty,
                    cast_to.ty
                ),
                Immediate::Uninit => throw_ub!(InvalidUninitBytes(None)),
            };
        }
    }

    pub fn pointer_expose_provenance_cast(
        &mut self,
        src: &ImmTy<'tcx, M::Provenance>,
        cast_to: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        assert_matches!(src.layout.ty.kind(), ty::RawPtr(_, _) | ty::FnPtr(..));
        assert!(cast_to.ty.is_integral());

        let scalar = src.to_scalar();
        let ptr = scalar.to_pointer(self)?;
        match ptr.into_pointer_or_addr() {
            Ok(ptr) => M::expose_provenance(self, ptr.provenance)?,
            Err(_) => {} // Do nothing, exposing an invalid pointer (`None` provenance) is a NOP.
        };
        interp_ok(ImmTy::from_scalar(
            self.cast_from_int_like(scalar, src.layout, cast_to.ty)?,
            cast_to,
        ))
    }

    pub fn pointer_with_exposed_provenance_cast(
        &self,
        src: &ImmTy<'tcx, M::Provenance>,
        cast_to: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        assert!(src.layout.ty.is_integral());
        assert_matches!(cast_to.ty.kind(), ty::RawPtr(_, _));

        // First cast to usize.
        let scalar = src.to_scalar();
        let addr = self.cast_from_int_like(scalar, src.layout, self.tcx.types.usize)?;
        let addr = addr.to_target_usize(self)?;

        // Then turn address into pointer.
        let ptr = M::ptr_from_addr_cast(self, addr)?;
        interp_ok(ImmTy::from_scalar(Scalar::from_maybe_pointer(ptr, self), cast_to))
    }

    /// Low-level cast helper function. This works directly on scalars and can take 'int-like' input
    /// type (basically everything with a scalar layout) to int/float/char types.
    fn cast_from_int_like(
        &self,
        scalar: Scalar<M::Provenance>, // input value (there is no ScalarTy so we separate data+layout)
        src_layout: TyAndLayout<'tcx>,
        cast_ty: Ty<'tcx>,
    ) -> InterpResult<'tcx, Scalar<M::Provenance>> {
        // Let's make sure v is sign-extended *if* it has a signed type.
        let signed = src_layout.backend_repr.is_signed(); // Also asserts that abi is `Scalar`.

        let v = match src_layout.ty.kind() {
            ty::Uint(_) | ty::RawPtr(..) | ty::FnPtr(..) => scalar.to_uint(src_layout.size)?,
            ty::Int(_) => scalar.to_int(src_layout.size)? as u128, // we will cast back to `i128` below if the sign matters
            ty::Bool => scalar.to_bool()?.into(),
            ty::Char => scalar.to_char()?.into(),
            _ => span_bug!(self.cur_span(), "invalid int-like cast from {}", src_layout.ty),
        };

        interp_ok(match *cast_ty.kind() {
            // int -> int
            ty::Int(_) | ty::Uint(_) => {
                let size = match *cast_ty.kind() {
                    ty::Int(t) => Integer::from_int_ty(self, t).size(),
                    ty::Uint(t) => Integer::from_uint_ty(self, t).size(),
                    _ => bug!(),
                };
                let v = size.truncate(v);
                Scalar::from_uint(v, size)
            }

            // signed int -> float
            ty::Float(fty) if signed => {
                let v = v as i128;
                match fty {
                    FloatTy::F16 => Scalar::from_f16(Half::from_i128(v).value),
                    FloatTy::F32 => Scalar::from_f32(Single::from_i128(v).value),
                    FloatTy::F64 => Scalar::from_f64(Double::from_i128(v).value),
                    FloatTy::F128 => Scalar::from_f128(Quad::from_i128(v).value),
                }
            }
            // unsigned int -> float
            ty::Float(fty) => match fty {
                FloatTy::F16 => Scalar::from_f16(Half::from_u128(v).value),
                FloatTy::F32 => Scalar::from_f32(Single::from_u128(v).value),
                FloatTy::F64 => Scalar::from_f64(Double::from_u128(v).value),
                FloatTy::F128 => Scalar::from_f128(Quad::from_u128(v).value),
            },

            // u8 -> char
            ty::Char => Scalar::from_u32(u8::try_from(v).unwrap().into()),

            // Casts to bool are not permitted by rustc, no need to handle them here.
            _ => span_bug!(self.cur_span(), "invalid int to {} cast", cast_ty),
        })
    }

    /// Low-level cast helper function. Converts an apfloat `f` into int or float types.
    fn cast_from_float<F>(&self, f: F, dest_ty: Ty<'tcx>) -> Scalar<M::Provenance>
    where
        F: Float
            + Into<Scalar<M::Provenance>>
            + FloatConvert<Half>
            + FloatConvert<Single>
            + FloatConvert<Double>
            + FloatConvert<Quad>,
    {
        match *dest_ty.kind() {
            // float -> uint
            ty::Uint(t) => {
                let size = Integer::from_uint_ty(self, t).size();
                // `to_u128` is a saturating cast, which is what we need
                // (https://doc.rust-lang.org/nightly/nightly-rustc/rustc_apfloat/trait.Float.html#method.to_i128_r).
                let v = f.to_u128(size.bits_usize()).value;
                // This should already fit the bit width
                Scalar::from_uint(v, size)
            }
            // float -> int
            ty::Int(t) => {
                let size = Integer::from_int_ty(self, t).size();
                // `to_i128` is a saturating cast, which is what we need
                // (https://doc.rust-lang.org/nightly/nightly-rustc/rustc_apfloat/trait.Float.html#method.to_i128_r).
                let v = f.to_i128(size.bits_usize()).value;
                Scalar::from_int(v, size)
            }
            // float -> float
            ty::Float(fty) => match fty {
                FloatTy::F16 => {
                    Scalar::from_f16(self.adjust_nan(f.convert(&mut false).value, &[f]))
                }
                FloatTy::F32 => {
                    Scalar::from_f32(self.adjust_nan(f.convert(&mut false).value, &[f]))
                }
                FloatTy::F64 => {
                    Scalar::from_f64(self.adjust_nan(f.convert(&mut false).value, &[f]))
                }
                FloatTy::F128 => {
                    Scalar::from_f128(self.adjust_nan(f.convert(&mut false).value, &[f]))
                }
            },
            // That's it.
            _ => span_bug!(self.cur_span(), "invalid float to {} cast", dest_ty),
        }
    }

    /// `src` is a *pointer to* a `source_ty`, and in `dest` we should store a pointer to th same
    /// data at type `cast_ty`.
    fn unsize_into_ptr(
        &mut self,
        src: &OpTy<'tcx, M::Provenance>,
        dest: &PlaceTy<'tcx, M::Provenance>,
        // The pointee types
        source_ty: Ty<'tcx>,
        cast_ty: Ty<'tcx>,
    ) -> InterpResult<'tcx> {
        // A<Struct> -> A<Trait> conversion
        let (src_pointee_ty, dest_pointee_ty) =
            self.tcx.struct_lockstep_tails_for_codegen(source_ty, cast_ty, self.typing_env);

        match (src_pointee_ty.kind(), dest_pointee_ty.kind()) {
            (&ty::Array(_, length), &ty::Slice(_)) => {
                let ptr = self.read_pointer(src)?;
                let val = Immediate::new_slice(
                    ptr,
                    length
                        .try_to_target_usize(*self.tcx)
                        .expect("expected monomorphic const in const eval"),
                    self,
                );
                self.write_immediate(val, dest)
            }
            (ty::Dynamic(data_a, _, ty::Dyn), ty::Dynamic(data_b, _, ty::Dyn)) => {
                let val = self.read_immediate(src)?;
                // MIR building generates odd NOP casts, prevent them from causing unexpected trouble.
                // See <https://github.com/rust-lang/rust/issues/128880>.
                // FIXME: ideally we wouldn't have to do this.
                if data_a == data_b {
                    return self.write_immediate(*val, dest);
                }
                // Take apart the old pointer, and find the dynamic type.
                let (old_data, old_vptr) = val.to_scalar_pair();
                let old_data = old_data.to_pointer(self)?;
                let old_vptr = old_vptr.to_pointer(self)?;
                let ty = self.get_ptr_vtable_ty(old_vptr, Some(data_a))?;

                // Sanity-check that `supertrait_vtable_slot` in this type's vtable indeed produces
                // our destination trait.
                let vptr_entry_idx =
                    self.tcx.supertrait_vtable_slot((src_pointee_ty, dest_pointee_ty));
                let vtable_entries = self.vtable_entries(data_a.principal(), ty);
                if let Some(entry_idx) = vptr_entry_idx {
                    let Some(&ty::VtblEntry::TraitVPtr(upcast_trait_ref)) =
                        vtable_entries.get(entry_idx)
                    else {
                        span_bug!(
                            self.cur_span(),
                            "invalid vtable entry index in {} -> {} upcast",
                            src_pointee_ty,
                            dest_pointee_ty
                        );
                    };
                    let erased_trait_ref =
                        ty::ExistentialTraitRef::erase_self_ty(*self.tcx, upcast_trait_ref);
                    assert_eq!(
                        data_b.principal().map(|b| {
                            self.tcx.normalize_erasing_late_bound_regions(self.typing_env, b)
                        }),
                        Some(erased_trait_ref),
                    );
                } else {
                    // In this case codegen would keep using the old vtable. We don't want to do
                    // that as it has the wrong trait. The reason codegen can do this is that
                    // one vtable is a prefix of the other, so we double-check that.
                    let vtable_entries_b = self.vtable_entries(data_b.principal(), ty);
                    assert!(&vtable_entries[..vtable_entries_b.len()] == vtable_entries_b);
                };

                // Get the destination trait vtable and return that.
                let new_vptr = self.get_vtable_ptr(ty, data_b)?;
                self.write_immediate(Immediate::new_dyn_trait(old_data, new_vptr, self), dest)
            }
            (_, &ty::Dynamic(data, _, ty::Dyn)) => {
                // Initial cast from sized to dyn trait
                let vtable = self.get_vtable_ptr(src_pointee_ty, data)?;
                let ptr = self.read_pointer(src)?;
                let val = Immediate::new_dyn_trait(ptr, vtable, &*self.tcx);
                self.write_immediate(val, dest)
            }
            _ => {
                // Do not ICE if we are not monomorphic enough.
                ensure_monomorphic_enough(*self.tcx, src.layout.ty)?;
                ensure_monomorphic_enough(*self.tcx, cast_ty)?;

                span_bug!(
                    self.cur_span(),
                    "invalid pointer unsizing {} -> {}",
                    src.layout.ty,
                    cast_ty
                )
            }
        }
    }

    pub fn unsize_into(
        &mut self,
        src: &OpTy<'tcx, M::Provenance>,
        cast_ty: TyAndLayout<'tcx>,
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        trace!("Unsizing {:?} of type {} into {}", *src, src.layout.ty, cast_ty.ty);
        match (src.layout.ty.kind(), cast_ty.ty.kind()) {
            (&ty::Ref(_, s, _), &ty::Ref(_, c, _) | &ty::RawPtr(c, _))
            | (&ty::RawPtr(s, _), &ty::RawPtr(c, _)) => self.unsize_into_ptr(src, dest, s, c),
            (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) => {
                assert_eq!(def_a, def_b); // implies same number of fields

                // Unsizing of generic struct with pointer fields, like `Arc<T>` -> `Arc<Trait>`.
                // There can be extra fields as long as they don't change their type or are 1-ZST.
                // There might also be no field that actually needs unsizing.
                let mut found_cast_field = false;
                for i in 0..src.layout.fields.count() {
                    let cast_ty_field = cast_ty.field(self, i);
                    let i = FieldIdx::from_usize(i);
                    let src_field = self.project_field(src, i)?;
                    let dst_field = self.project_field(dest, i)?;
                    if src_field.layout.is_1zst() && cast_ty_field.is_1zst() {
                        // Skip 1-ZST fields.
                    } else if src_field.layout.ty == cast_ty_field.ty {
                        self.copy_op(&src_field, &dst_field)?;
                    } else {
                        if found_cast_field {
                            span_bug!(self.cur_span(), "unsize_into: more than one field to cast");
                        }
                        found_cast_field = true;
                        self.unsize_into(&src_field, cast_ty_field, &dst_field)?;
                    }
                }
                interp_ok(())
            }
            _ => {
                // Do not ICE if we are not monomorphic enough.
                ensure_monomorphic_enough(*self.tcx, src.layout.ty)?;
                ensure_monomorphic_enough(*self.tcx, cast_ty.ty)?;

                span_bug!(
                    self.cur_span(),
                    "unsize_into: invalid conversion: {:?} -> {:?}",
                    src.layout,
                    dest.layout
                )
            }
        }
    }
}
