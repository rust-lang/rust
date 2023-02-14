use std::assert_matches::assert_matches;

use rustc_apfloat::ieee::{Double, Single};
use rustc_apfloat::{Float, FloatConvert};
use rustc_middle::mir::interpret::{InterpResult, PointerArithmetic, Scalar};
use rustc_middle::mir::CastKind;
use rustc_middle::ty::adjustment::PointerCast;
use rustc_middle::ty::layout::{IntegerExt, LayoutOf, TyAndLayout};
use rustc_middle::ty::{self, FloatTy, Ty, TypeAndMut};
use rustc_target::abi::Integer;
use rustc_type_ir::sty::TyKind::*;

use super::{
    util::ensure_monomorphic_enough, FnVal, ImmTy, Immediate, InterpCx, Machine, OpTy, PlaceTy,
};

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    pub fn cast(
        &mut self,
        src: &OpTy<'tcx, M::Provenance>,
        cast_kind: CastKind,
        cast_ty: Ty<'tcx>,
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        use rustc_middle::mir::CastKind::*;
        // FIXME: In which cases should we trigger UB when the source is uninit?
        match cast_kind {
            Pointer(PointerCast::Unsize) => {
                let cast_ty = self.layout_of(cast_ty)?;
                self.unsize_into(src, cast_ty, dest)?;
            }

            PointerExposeAddress => {
                let src = self.read_immediate(src)?;
                let res = self.pointer_expose_address_cast(&src, cast_ty)?;
                self.write_immediate(res, dest)?;
            }

            PointerFromExposedAddress => {
                let src = self.read_immediate(src)?;
                let res = self.pointer_from_exposed_address_cast(&src, cast_ty)?;
                self.write_immediate(res, dest)?;
            }

            IntToInt | IntToFloat => {
                let src = self.read_immediate(src)?;
                let res = self.int_to_int_or_float(&src, cast_ty)?;
                self.write_immediate(res, dest)?;
            }

            FloatToFloat | FloatToInt => {
                let src = self.read_immediate(src)?;
                let res = self.float_to_float_or_int(&src, cast_ty)?;
                self.write_immediate(res, dest)?;
            }

            FnPtrToPtr | PtrToPtr => {
                let src = self.read_immediate(&src)?;
                let res = self.ptr_to_ptr(&src, cast_ty)?;
                self.write_immediate(res, dest)?;
            }

            Pointer(PointerCast::MutToConstPointer | PointerCast::ArrayToPointer) => {
                // These are NOPs, but can be wide pointers.
                let v = self.read_immediate(src)?;
                self.write_immediate(*v, dest)?;
            }

            Pointer(PointerCast::ReifyFnPointer) => {
                // The src operand does not matter, just its type
                match *src.layout.ty.kind() {
                    ty::FnDef(def_id, substs) => {
                        // All reifications must be monomorphic, bail out otherwise.
                        ensure_monomorphic_enough(*self.tcx, src.layout.ty)?;

                        let instance = ty::Instance::resolve_for_fn_ptr(
                            *self.tcx,
                            self.param_env,
                            def_id,
                            substs,
                        )
                        .ok_or_else(|| err_inval!(TooGeneric))?;

                        let fn_ptr = self.create_fn_alloc_ptr(FnVal::Instance(instance));
                        self.write_pointer(fn_ptr, dest)?;
                    }
                    _ => span_bug!(self.cur_span(), "reify fn pointer on {:?}", src.layout.ty),
                }
            }

            Pointer(PointerCast::UnsafeFnPointer) => {
                let src = self.read_immediate(src)?;
                match cast_ty.kind() {
                    ty::FnPtr(_) => {
                        // No change to value
                        self.write_immediate(*src, dest)?;
                    }
                    _ => span_bug!(self.cur_span(), "fn to unsafe fn cast on {:?}", cast_ty),
                }
            }

            Pointer(PointerCast::ClosureFnPointer(_)) => {
                // The src operand does not matter, just its type
                match *src.layout.ty.kind() {
                    ty::Closure(def_id, substs) => {
                        // All reifications must be monomorphic, bail out otherwise.
                        ensure_monomorphic_enough(*self.tcx, src.layout.ty)?;

                        let instance = ty::Instance::resolve_closure(
                            *self.tcx,
                            def_id,
                            substs,
                            ty::ClosureKind::FnOnce,
                        )
                        .ok_or_else(|| err_inval!(TooGeneric))?;
                        let fn_ptr = self.create_fn_alloc_ptr(FnVal::Instance(instance));
                        self.write_pointer(fn_ptr, dest)?;
                    }
                    _ => span_bug!(self.cur_span(), "closure fn pointer on {:?}", src.layout.ty),
                }
            }

            DynStar => {
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
        }
        Ok(())
    }

    /// Handles 'IntToInt' and 'IntToFloat' casts.
    pub fn int_to_int_or_float(
        &self,
        src: &ImmTy<'tcx, M::Provenance>,
        cast_ty: Ty<'tcx>,
    ) -> InterpResult<'tcx, Immediate<M::Provenance>> {
        assert!(src.layout.ty.is_integral() || src.layout.ty.is_char() || src.layout.ty.is_bool());
        assert!(cast_ty.is_floating_point() || cast_ty.is_integral() || cast_ty.is_char());

        Ok(self.cast_from_int_like(src.to_scalar(), src.layout, cast_ty)?.into())
    }

    /// Handles 'FloatToFloat' and 'FloatToInt' casts.
    pub fn float_to_float_or_int(
        &self,
        src: &ImmTy<'tcx, M::Provenance>,
        cast_ty: Ty<'tcx>,
    ) -> InterpResult<'tcx, Immediate<M::Provenance>> {
        use rustc_type_ir::sty::TyKind::*;

        match src.layout.ty.kind() {
            // Floating point
            Float(FloatTy::F32) => {
                return Ok(self.cast_from_float(src.to_scalar().to_f32()?, cast_ty).into());
            }
            Float(FloatTy::F64) => {
                return Ok(self.cast_from_float(src.to_scalar().to_f64()?, cast_ty).into());
            }
            _ => {
                bug!("Can't cast 'Float' type into {:?}", cast_ty);
            }
        }
    }

    /// Handles 'FnPtrToPtr' and 'PtrToPtr' casts.
    pub fn ptr_to_ptr(
        &self,
        src: &ImmTy<'tcx, M::Provenance>,
        cast_ty: Ty<'tcx>,
    ) -> InterpResult<'tcx, Immediate<M::Provenance>> {
        assert!(src.layout.ty.is_any_ptr());
        assert!(cast_ty.is_unsafe_ptr());
        // Handle casting any ptr to raw ptr (might be a fat ptr).
        let dest_layout = self.layout_of(cast_ty)?;
        if dest_layout.size == src.layout.size {
            // Thin or fat pointer that just hast the ptr kind of target type changed.
            return Ok(**src);
        } else {
            // Casting the metadata away from a fat ptr.
            assert_eq!(src.layout.size, 2 * self.pointer_size());
            assert_eq!(dest_layout.size, self.pointer_size());
            assert!(src.layout.ty.is_unsafe_ptr());
            return match **src {
                Immediate::ScalarPair(data, _) => Ok(data.into()),
                Immediate::Scalar(..) => span_bug!(
                    self.cur_span(),
                    "{:?} input to a fat-to-thin cast ({:?} -> {:?})",
                    *src,
                    src.layout.ty,
                    cast_ty
                ),
                Immediate::Uninit => throw_ub!(InvalidUninitBytes(None)),
            };
        }
    }

    pub fn pointer_expose_address_cast(
        &mut self,
        src: &ImmTy<'tcx, M::Provenance>,
        cast_ty: Ty<'tcx>,
    ) -> InterpResult<'tcx, Immediate<M::Provenance>> {
        assert_matches!(src.layout.ty.kind(), ty::RawPtr(_) | ty::FnPtr(_));
        assert!(cast_ty.is_integral());

        let scalar = src.to_scalar();
        let ptr = scalar.to_pointer(self)?;
        match ptr.into_pointer_or_addr() {
            Ok(ptr) => M::expose_ptr(self, ptr)?,
            Err(_) => {} // Do nothing, exposing an invalid pointer (`None` provenance) is a NOP.
        };
        Ok(self.cast_from_int_like(scalar, src.layout, cast_ty)?.into())
    }

    pub fn pointer_from_exposed_address_cast(
        &self,
        src: &ImmTy<'tcx, M::Provenance>,
        cast_ty: Ty<'tcx>,
    ) -> InterpResult<'tcx, Immediate<M::Provenance>> {
        assert!(src.layout.ty.is_integral());
        assert_matches!(cast_ty.kind(), ty::RawPtr(_));

        // First cast to usize.
        let scalar = src.to_scalar();
        let addr = self.cast_from_int_like(scalar, src.layout, self.tcx.types.usize)?;
        let addr = addr.to_machine_usize(self)?;

        // Then turn address into pointer.
        let ptr = M::ptr_from_addr_cast(&self, addr)?;
        Ok(Scalar::from_maybe_pointer(ptr, self).into())
    }

    /// Low-level cast helper function. This works directly on scalars and can take 'int-like' input
    /// type (basically everything with a scalar layout) to int/float/char types.
    pub fn cast_from_int_like(
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
            Int(_) | Uint(_) => {
                let size = match *cast_ty.kind() {
                    Int(t) => Integer::from_int_ty(self, t).size(),
                    Uint(t) => Integer::from_uint_ty(self, t).size(),
                    _ => bug!(),
                };
                let v = size.truncate(v);
                Scalar::from_uint(v, size)
            }

            Float(FloatTy::F32) if signed => Scalar::from_f32(Single::from_i128(v as i128).value),
            Float(FloatTy::F64) if signed => Scalar::from_f64(Double::from_i128(v as i128).value),
            Float(FloatTy::F32) => Scalar::from_f32(Single::from_u128(v).value),
            Float(FloatTy::F64) => Scalar::from_f64(Double::from_u128(v).value),

            Char => {
                // `u8` to `char` cast
                Scalar::from_u32(u8::try_from(v).unwrap().into())
            }

            // Casts to bool are not permitted by rustc, no need to handle them here.
            _ => span_bug!(self.cur_span(), "invalid int to {:?} cast", cast_ty),
        })
    }

    /// Low-level cast helper function. Converts an apfloat `f` into int or float types.
    fn cast_from_float<F>(&self, f: F, dest_ty: Ty<'tcx>) -> Scalar<M::Provenance>
    where
        F: Float + Into<Scalar<M::Provenance>> + FloatConvert<Single> + FloatConvert<Double>,
    {
        use rustc_type_ir::sty::TyKind::*;
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
            // float -> f32
            Float(FloatTy::F32) => Scalar::from_f32(f.convert(&mut false).value),
            // float -> f64
            Float(FloatTy::F64) => Scalar::from_f64(f.convert(&mut false).value),
            // That's it.
            _ => span_bug!(self.cur_span(), "invalid float to {:?} cast", dest_ty),
        }
    }

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
                let ptr = self.read_scalar(src)?;
                // u64 cast is from usize to u64, which is always good
                let val = Immediate::new_slice(
                    ptr,
                    length.eval_target_usize(*self.tcx, self.param_env),
                    self,
                );
                self.write_immediate(val, dest)
            }
            (ty::Dynamic(data_a, ..), ty::Dynamic(data_b, ..)) => {
                let val = self.read_immediate(src)?;
                if data_a.principal() == data_b.principal() {
                    // A NOP cast that doesn't actually change anything, should be allowed even with mismatching vtables.
                    return self.write_immediate(*val, dest);
                }
                let (old_data, old_vptr) = val.to_scalar_pair();
                let old_vptr = old_vptr.to_pointer(self)?;
                let (ty, old_trait) = self.get_ptr_vtable(old_vptr)?;
                if old_trait != data_a.principal() {
                    throw_ub_format!("upcast on a pointer whose vtable does not match its type");
                }
                let new_vptr = self.get_vtable_ptr(ty, data_b.principal())?;
                self.write_immediate(Immediate::new_dyn_trait(old_data, new_vptr, self), dest)
            }
            (_, &ty::Dynamic(data, _, ty::Dyn)) => {
                // Initial cast from sized to dyn trait
                let vtable = self.get_vtable_ptr(src_pointee_ty, data.principal())?;
                let ptr = self.read_scalar(src)?;
                let val = Immediate::new_dyn_trait(ptr, vtable, &*self.tcx);
                self.write_immediate(val, dest)
            }

            _ => {
                span_bug!(self.cur_span(), "invalid unsizing {:?} -> {:?}", src.layout.ty, cast_ty)
            }
        }
    }

    fn unsize_into(
        &mut self,
        src: &OpTy<'tcx, M::Provenance>,
        cast_ty: TyAndLayout<'tcx>,
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        trace!("Unsizing {:?} of type {} into {:?}", *src, src.layout.ty, cast_ty.ty);
        match (&src.layout.ty.kind(), &cast_ty.ty.kind()) {
            (&ty::Ref(_, s, _), &ty::Ref(_, c, _) | &ty::RawPtr(TypeAndMut { ty: c, .. }))
            | (&ty::RawPtr(TypeAndMut { ty: s, .. }), &ty::RawPtr(TypeAndMut { ty: c, .. })) => {
                self.unsize_into_ptr(src, dest, *s, *c)
            }
            (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) => {
                assert_eq!(def_a, def_b);

                // unsizing of generic struct with pointer fields
                // Example: `Arc<T>` -> `Arc<Trait>`
                // here we need to increase the size of every &T thin ptr field to a fat ptr
                for i in 0..src.layout.fields.count() {
                    let cast_ty_field = cast_ty.field(self, i);
                    if cast_ty_field.is_zst() {
                        continue;
                    }
                    let src_field = self.operand_field(src, i)?;
                    let dst_field = self.place_field(dest, i)?;
                    if src_field.layout.ty == cast_ty_field.ty {
                        self.copy_op(&src_field, &dst_field, /*allow_transmute*/ false)?;
                    } else {
                        self.unsize_into(&src_field, cast_ty_field, &dst_field)?;
                    }
                }
                Ok(())
            }
            _ => span_bug!(
                self.cur_span(),
                "unsize_into: invalid conversion: {:?} -> {:?}",
                src.layout,
                dest.layout
            ),
        }
    }
}
