use std::assert_matches::assert_matches;

use rustc_apfloat::ieee::{Double, Single};
use rustc_apfloat::{Float, FloatConvert};
use rustc_middle::mir::interpret::{InterpResult, PointerArithmetic, Scalar};
use rustc_middle::mir::CastKind;
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::layout::{IntegerExt, LayoutOf, TyAndLayout};
use rustc_middle::ty::{self, FloatTy, Ty, TypeAndMut};
use rustc_target::abi::Integer;
use rustc_type_ir::TyKind::*;

use super::{
    util::ensure_monomorphic_enough, FnVal, ImmTy, Immediate, InterpCx, Machine, OpTy, PlaceTy,
};

use crate::fluent_generated as fluent;

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
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
            CastKind::PointerCoercion(PointerCoercion::Unsize) => {
                self.unsize_into(src, cast_layout, dest)?;
            }

            CastKind::PointerExposeAddress => {
                let src = self.read_immediate(src)?;
                let res = self.pointer_expose_address_cast(&src, cast_layout)?;
                self.write_immediate(*res, dest)?;
            }

            CastKind::PointerFromExposedAddress => {
                let src = self.read_immediate(src)?;
                let res = self.pointer_from_exposed_address_cast(&src, cast_layout)?;
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
            ) => {
                // These are NOPs, but can be wide pointers.
                let v = self.read_immediate(src)?;
                self.write_immediate(*v, dest)?;
            }

            CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer) => {
                // All reifications must be monomorphic, bail out otherwise.
                ensure_monomorphic_enough(*self.tcx, src.layout.ty)?;

                // The src operand does not matter, just its type
                match *src.layout.ty.kind() {
                    ty::FnDef(def_id, args) => {
                        let instance = ty::Instance::resolve_for_fn_ptr(
                            *self.tcx,
                            self.param_env,
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

            CastKind::PointerCoercion(PointerCoercion::UnsafeFnPointer) => {
                let src = self.read_immediate(src)?;
                match cast_ty.kind() {
                    ty::FnPtr(_) => {
                        // No change to value
                        self.write_immediate(*src, dest)?;
                    }
                    _ => span_bug!(self.cur_span(), "fn to unsafe fn cast on {}", cast_ty),
                }
            }

            CastKind::PointerCoercion(PointerCoercion::ClosureFnPointer(_)) => {
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

            CastKind::DynStar => {
                if let ty::Dynamic(data, _, ty::DynStar) = cast_ty.kind() {
                    // Initial cast from sized to dyn trait
                    let vtable = self.get_vtable_ptr(src.layout.ty, data.principal())?;
                    let vtable = Scalar::from_maybe_pointer(vtable, self);
                    let data = self.read_immediate(src)?.to_scalar();
                    let _assert_pointer_like = data.to_pointer(self)?;
                    let val = Immediate::ScalarPair(data, vtable);
                    self.write_immediate(val, dest)?;
                } else {
                    bug!()
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
        Ok(())
    }

    /// Handles 'IntToInt' and 'IntToFloat' casts.
    pub fn int_to_int_or_float(
        &self,
        src: &ImmTy<'tcx, M::Provenance>,
        cast_to: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        assert!(src.layout.ty.is_integral() || src.layout.ty.is_char() || src.layout.ty.is_bool());
        assert!(cast_to.ty.is_floating_point() || cast_to.ty.is_integral() || cast_to.ty.is_char());

        Ok(ImmTy::from_scalar(
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
        use rustc_type_ir::TyKind::*;

        let Float(fty) = src.layout.ty.kind() else {
            bug!("FloatToFloat/FloatToInt cast: source type {} is not a float type", src.layout.ty)
        };
        let val = match fty {
            FloatTy::F16 => unimplemented!("f16_f128"),
            FloatTy::F32 => self.cast_from_float(src.to_scalar().to_f32()?, cast_to.ty),
            FloatTy::F64 => self.cast_from_float(src.to_scalar().to_f64()?, cast_to.ty),
            FloatTy::F128 => unimplemented!("f16_f128"),
        };
        Ok(ImmTy::from_scalar(val, cast_to))
    }

    /// Handles 'FnPtrToPtr' and 'PtrToPtr' casts.
    pub fn ptr_to_ptr(
        &self,
        src: &ImmTy<'tcx, M::Provenance>,
        cast_to: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        assert!(src.layout.ty.is_any_ptr());
        assert!(cast_to.ty.is_unsafe_ptr());
        // Handle casting any ptr to raw ptr (might be a fat ptr).
        if cast_to.size == src.layout.size {
            // Thin or fat pointer that just hast the ptr kind of target type changed.
            return Ok(ImmTy::from_immediate(**src, cast_to));
        } else {
            // Casting the metadata away from a fat ptr.
            assert_eq!(src.layout.size, 2 * self.pointer_size());
            assert_eq!(cast_to.size, self.pointer_size());
            assert!(src.layout.ty.is_unsafe_ptr());
            return match **src {
                Immediate::ScalarPair(data, _) => Ok(ImmTy::from_scalar(data, cast_to)),
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

    pub fn pointer_expose_address_cast(
        &mut self,
        src: &ImmTy<'tcx, M::Provenance>,
        cast_to: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        assert_matches!(src.layout.ty.kind(), ty::RawPtr(_) | ty::FnPtr(_));
        assert!(cast_to.ty.is_integral());

        let scalar = src.to_scalar();
        let ptr = scalar.to_pointer(self)?;
        match ptr.into_pointer_or_addr() {
            Ok(ptr) => M::expose_ptr(self, ptr)?,
            Err(_) => {} // Do nothing, exposing an invalid pointer (`None` provenance) is a NOP.
        };
        Ok(ImmTy::from_scalar(self.cast_from_int_like(scalar, src.layout, cast_to.ty)?, cast_to))
    }

    pub fn pointer_from_exposed_address_cast(
        &self,
        src: &ImmTy<'tcx, M::Provenance>,
        cast_to: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        assert!(src.layout.ty.is_integral());
        assert_matches!(cast_to.ty.kind(), ty::RawPtr(_));

        // First cast to usize.
        let scalar = src.to_scalar();
        let addr = self.cast_from_int_like(scalar, src.layout, self.tcx.types.usize)?;
        let addr = addr.to_target_usize(self)?;

        // Then turn address into pointer.
        let ptr = M::ptr_from_addr_cast(self, addr)?;
        Ok(ImmTy::from_scalar(Scalar::from_maybe_pointer(ptr, self), cast_to))
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
        let signed = src_layout.abi.is_signed(); // Also asserts that abi is `Scalar`.

        let v = scalar.to_bits(src_layout.size)?;
        let v = if signed { self.sign_extend(v, src_layout) } else { v };
        trace!("cast_from_scalar: {}, {} -> {}", v, src_layout.ty, cast_ty);

        Ok(match *cast_ty.kind() {
            // int -> int
            Int(_) | Uint(_) => {
                let size = match *cast_ty.kind() {
                    Int(t) => Integer::from_int_ty(self, t).size(),
                    Uint(t) => Integer::from_uint_ty(self, t).size(),
                    _ => bug!(),
                };
                let v = size.truncate(v);
                Scalar::from_uint(v, size)
            }

            // signed int -> float
            Float(fty) if signed => {
                let v = v as i128;
                match fty {
                    FloatTy::F16 => unimplemented!("f16_f128"),
                    FloatTy::F32 => Scalar::from_f32(Single::from_i128(v).value),
                    FloatTy::F64 => Scalar::from_f64(Double::from_i128(v).value),
                    FloatTy::F128 => unimplemented!("f16_f128"),
                }
            }
            // unsigned int -> float
            Float(fty) => match fty {
                FloatTy::F16 => unimplemented!("f16_f128"),
                FloatTy::F32 => Scalar::from_f32(Single::from_u128(v).value),
                FloatTy::F64 => Scalar::from_f64(Double::from_u128(v).value),
                FloatTy::F128 => unimplemented!("f16_f128"),
            },

            // u8 -> char
            Char => Scalar::from_u32(u8::try_from(v).unwrap().into()),

            // Casts to bool are not permitted by rustc, no need to handle them here.
            _ => span_bug!(self.cur_span(), "invalid int to {} cast", cast_ty),
        })
    }

    /// Low-level cast helper function. Converts an apfloat `f` into int or float types.
    fn cast_from_float<F>(&self, f: F, dest_ty: Ty<'tcx>) -> Scalar<M::Provenance>
    where
        F: Float + Into<Scalar<M::Provenance>> + FloatConvert<Single> + FloatConvert<Double>,
    {
        use rustc_type_ir::TyKind::*;

        fn adjust_nan<
            'mir,
            'tcx: 'mir,
            M: Machine<'mir, 'tcx>,
            F1: rustc_apfloat::Float + FloatConvert<F2>,
            F2: rustc_apfloat::Float,
        >(
            ecx: &InterpCx<'mir, 'tcx, M>,
            f1: F1,
            f2: F2,
        ) -> F2 {
            if f2.is_nan() { M::generate_nan(ecx, &[f1]) } else { f2 }
        }

        match *dest_ty.kind() {
            // float -> uint
            Uint(t) => {
                let size = Integer::from_uint_ty(self, t).size();
                // `to_u128` is a saturating cast, which is what we need
                // (https://doc.rust-lang.org/nightly/nightly-rustc/rustc_apfloat/trait.Float.html#method.to_i128_r).
                let v = f.to_u128(size.bits_usize()).value;
                // This should already fit the bit width
                Scalar::from_uint(v, size)
            }
            // float -> int
            Int(t) => {
                let size = Integer::from_int_ty(self, t).size();
                // `to_i128` is a saturating cast, which is what we need
                // (https://doc.rust-lang.org/nightly/nightly-rustc/rustc_apfloat/trait.Float.html#method.to_i128_r).
                let v = f.to_i128(size.bits_usize()).value;
                Scalar::from_int(v, size)
            }
            // float -> float
            Float(fty) => match fty {
                FloatTy::F16 => unimplemented!("f16_f128"),
                FloatTy::F32 => Scalar::from_f32(adjust_nan(self, f, f.convert(&mut false).value)),
                FloatTy::F64 => Scalar::from_f64(adjust_nan(self, f, f.convert(&mut false).value)),
                FloatTy::F128 => unimplemented!("f16_f128"),
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
            self.tcx.struct_lockstep_tails_erasing_lifetimes(source_ty, cast_ty, self.param_env);

        match (&src_pointee_ty.kind(), &dest_pointee_ty.kind()) {
            (&ty::Array(_, length), &ty::Slice(_)) => {
                let ptr = self.read_pointer(src)?;
                // u64 cast is from usize to u64, which is always good
                let val = Immediate::new_slice(
                    ptr,
                    length.eval_target_usize(*self.tcx, self.param_env),
                    self,
                );
                self.write_immediate(val, dest)
            }
            (ty::Dynamic(data_a, _, ty::Dyn), ty::Dynamic(data_b, _, ty::Dyn)) => {
                let val = self.read_immediate(src)?;
                if data_a.principal() == data_b.principal() {
                    // A NOP cast that doesn't actually change anything, should be allowed even with mismatching vtables.
                    return self.write_immediate(*val, dest);
                }
                let (old_data, old_vptr) = val.to_scalar_pair();
                let old_data = old_data.to_pointer(self)?;
                let old_vptr = old_vptr.to_pointer(self)?;
                let (ty, old_trait) = self.get_ptr_vtable(old_vptr)?;
                if old_trait != data_a.principal() {
                    throw_ub_custom!(fluent::const_eval_upcast_mismatch);
                }
                let new_vptr = self.get_vtable_ptr(ty, data_b.principal())?;
                self.write_immediate(Immediate::new_dyn_trait(old_data, new_vptr, self), dest)
            }
            (_, &ty::Dynamic(data, _, ty::Dyn)) => {
                // Initial cast from sized to dyn trait
                let vtable = self.get_vtable_ptr(src_pointee_ty, data.principal())?;
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
        match (&src.layout.ty.kind(), &cast_ty.ty.kind()) {
            (&ty::Ref(_, s, _), &ty::Ref(_, c, _) | &ty::RawPtr(TypeAndMut { ty: c, .. }))
            | (&ty::RawPtr(TypeAndMut { ty: s, .. }), &ty::RawPtr(TypeAndMut { ty: c, .. })) => {
                self.unsize_into_ptr(src, dest, *s, *c)
            }
            (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) => {
                assert_eq!(def_a, def_b); // implies same number of fields

                // Unsizing of generic struct with pointer fields, like `Arc<T>` -> `Arc<Trait>`.
                // There can be extra fields as long as they don't change their type or are 1-ZST.
                // There might also be no field that actually needs unsizing.
                let mut found_cast_field = false;
                for i in 0..src.layout.fields.count() {
                    let cast_ty_field = cast_ty.field(self, i);
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
                Ok(())
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
