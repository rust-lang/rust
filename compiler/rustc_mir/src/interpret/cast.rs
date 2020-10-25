use std::convert::TryFrom;

use rustc_apfloat::ieee::{Double, Single};
use rustc_apfloat::{Float, FloatConvert};
use rustc_ast::FloatTy;
use rustc_attr as attr;
use rustc_middle::mir::interpret::{InterpResult, PointerArithmetic, Scalar};
use rustc_middle::mir::CastKind;
use rustc_middle::ty::adjustment::PointerCast;
use rustc_middle::ty::layout::{IntegerExt, TyAndLayout};
use rustc_middle::ty::{self, Ty, TypeAndMut};
use rustc_span::symbol::sym;
use rustc_target::abi::{Integer, LayoutOf, Variants};

use super::{
    truncate, util::ensure_monomorphic_enough, FnVal, ImmTy, Immediate, InterpCx, Machine, OpTy,
    PlaceTy,
};

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    pub fn cast(
        &mut self,
        src: OpTy<'tcx, M::PointerTag>,
        cast_kind: CastKind,
        cast_ty: Ty<'tcx>,
        dest: PlaceTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx> {
        use rustc_middle::mir::CastKind::*;
        // FIXME: In which cases should we trigger UB when the source is uninit?
        match cast_kind {
            Pointer(PointerCast::Unsize) => {
                let cast_ty = self.layout_of(cast_ty)?;
                self.unsize_into(src, cast_ty, dest)?;
            }

            Misc => {
                let src = self.read_immediate(src)?;
                let res = self.misc_cast(src, cast_ty)?;
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

                        if self.tcx.has_attr(def_id, sym::rustc_args_required_const) {
                            span_bug!(
                                self.cur_span(),
                                "reifying a fn ptr that requires const arguments"
                            );
                        }

                        let instance = ty::Instance::resolve_for_fn_ptr(
                            *self.tcx,
                            self.param_env,
                            def_id,
                            substs,
                        )
                        .ok_or_else(|| err_inval!(TooGeneric))?;

                        let fn_ptr = self.memory.create_fn_alloc(FnVal::Instance(instance));
                        self.write_scalar(fn_ptr, dest)?;
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
                        );
                        let fn_ptr = self.memory.create_fn_alloc(FnVal::Instance(instance));
                        self.write_scalar(fn_ptr, dest)?;
                    }
                    _ => span_bug!(self.cur_span(), "closure fn pointer on {:?}", src.layout.ty),
                }
            }
        }
        Ok(())
    }

    fn misc_cast(
        &self,
        src: ImmTy<'tcx, M::PointerTag>,
        cast_ty: Ty<'tcx>,
    ) -> InterpResult<'tcx, Immediate<M::PointerTag>> {
        use rustc_middle::ty::TyKind::*;
        trace!("Casting {:?}: {:?} to {:?}", *src, src.layout.ty, cast_ty);

        match src.layout.ty.kind() {
            // Floating point
            Float(FloatTy::F32) => {
                return Ok(self.cast_from_float(src.to_scalar()?.to_f32()?, cast_ty).into());
            }
            Float(FloatTy::F64) => {
                return Ok(self.cast_from_float(src.to_scalar()?.to_f64()?, cast_ty).into());
            }
            // The rest is integer/pointer-"like", including fn ptr casts and casts from enums that
            // are represented as integers.
            _ => assert!(
                src.layout.ty.is_bool()
                    || src.layout.ty.is_char()
                    || src.layout.ty.is_enum()
                    || src.layout.ty.is_integral()
                    || src.layout.ty.is_any_ptr(),
                "Unexpected cast from type {:?}",
                src.layout.ty
            ),
        }

        // # First handle non-scalar source values.

        // Handle cast from a ZST enum (0 or 1 variants).
        match src.layout.variants {
            Variants::Single { index } => {
                if src.layout.abi.is_uninhabited() {
                    // This is dead code, because an uninhabited enum is UB to
                    // instantiate.
                    throw_ub!(Unreachable);
                }
                if let Some(discr) = src.layout.ty.discriminant_for_variant(*self.tcx, index) {
                    assert!(src.layout.is_zst());
                    let discr_layout = self.layout_of(discr.ty)?;
                    return Ok(self.cast_from_scalar(discr.val, discr_layout, cast_ty).into());
                }
            }
            Variants::Multiple { .. } => {}
        }

        // Handle casting any ptr to raw ptr (might be a fat ptr).
        if src.layout.ty.is_any_ptr() && cast_ty.is_unsafe_ptr() {
            let dest_layout = self.layout_of(cast_ty)?;
            if dest_layout.size == src.layout.size {
                // Thin or fat pointer that just hast the ptr kind of target type changed.
                return Ok(*src);
            } else {
                // Casting the metadata away from a fat ptr.
                assert_eq!(src.layout.size, 2 * self.memory.pointer_size());
                assert_eq!(dest_layout.size, self.memory.pointer_size());
                assert!(src.layout.ty.is_unsafe_ptr());
                return match *src {
                    Immediate::ScalarPair(data, _) => Ok(data.into()),
                    Immediate::Scalar(..) => span_bug!(
                        self.cur_span(),
                        "{:?} input to a fat-to-thin cast ({:?} -> {:?})",
                        *src,
                        src.layout.ty,
                        cast_ty
                    ),
                };
            }
        }

        // # The remaining source values are scalar.

        // For all remaining casts, we either
        // (a) cast a raw ptr to usize, or
        // (b) cast from an integer-like (including bool, char, enums).
        // In both cases we want the bits.
        let bits = self.force_bits(src.to_scalar()?, src.layout.size)?;
        Ok(self.cast_from_scalar(bits, src.layout, cast_ty).into())
    }

    pub(super) fn cast_from_scalar(
        &self,
        v: u128, // raw bits (there is no ScalarTy so we separate data+layout)
        src_layout: TyAndLayout<'tcx>,
        cast_ty: Ty<'tcx>,
    ) -> Scalar<M::PointerTag> {
        // Let's make sure v is sign-extended *if* it has a signed type.
        let signed = src_layout.abi.is_signed(); // Also asserts that abi is `Scalar`.
        let v = if signed { self.sign_extend(v, src_layout) } else { v };
        trace!("cast_from_scalar: {}, {} -> {}", v, src_layout.ty, cast_ty);
        use rustc_middle::ty::TyKind::*;
        match *cast_ty.kind() {
            Int(_) | Uint(_) | RawPtr(_) => {
                let size = match *cast_ty.kind() {
                    Int(t) => Integer::from_attr(self, attr::IntType::SignedInt(t)).size(),
                    Uint(t) => Integer::from_attr(self, attr::IntType::UnsignedInt(t)).size(),
                    RawPtr(_) => self.pointer_size(),
                    _ => bug!(),
                };
                let v = truncate(v, size);
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
        }
    }

    fn cast_from_float<F>(&self, f: F, dest_ty: Ty<'tcx>) -> Scalar<M::PointerTag>
    where
        F: Float + Into<Scalar<M::PointerTag>> + FloatConvert<Single> + FloatConvert<Double>,
    {
        use rustc_middle::ty::TyKind::*;
        match *dest_ty.kind() {
            // float -> uint
            Uint(t) => {
                let size = Integer::from_attr(self, attr::IntType::UnsignedInt(t)).size();
                // `to_u128` is a saturating cast, which is what we need
                // (https://doc.rust-lang.org/nightly/nightly-rustc/rustc_apfloat/trait.Float.html#method.to_i128_r).
                let v = f.to_u128(size.bits_usize()).value;
                // This should already fit the bit width
                Scalar::from_uint(v, size)
            }
            // float -> int
            Int(t) => {
                let size = Integer::from_attr(self, attr::IntType::SignedInt(t)).size();
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
        src: OpTy<'tcx, M::PointerTag>,
        dest: PlaceTy<'tcx, M::PointerTag>,
        // The pointee types
        source_ty: Ty<'tcx>,
        cast_ty: Ty<'tcx>,
    ) -> InterpResult<'tcx> {
        // A<Struct> -> A<Trait> conversion
        let (src_pointee_ty, dest_pointee_ty) =
            self.tcx.struct_lockstep_tails_erasing_lifetimes(source_ty, cast_ty, self.param_env);

        match (&src_pointee_ty.kind(), &dest_pointee_ty.kind()) {
            (&ty::Array(_, length), &ty::Slice(_)) => {
                let ptr = self.read_immediate(src)?.to_scalar()?;
                // u64 cast is from usize to u64, which is always good
                let val =
                    Immediate::new_slice(ptr, length.eval_usize(*self.tcx, self.param_env), self);
                self.write_immediate(val, dest)
            }
            (&ty::Dynamic(..), &ty::Dynamic(..)) => {
                // For now, upcasts are limited to changes in marker
                // traits, and hence never actually require an actual
                // change to the vtable.
                let val = self.read_immediate(src)?;
                self.write_immediate(*val, dest)
            }
            (_, &ty::Dynamic(ref data, _)) => {
                // Initial cast from sized to dyn trait
                let vtable = self.get_vtable(src_pointee_ty, data.principal())?;
                let ptr = self.read_immediate(src)?.to_scalar()?;
                let val = Immediate::new_dyn_trait(ptr, vtable);
                self.write_immediate(val, dest)
            }

            _ => {
                span_bug!(self.cur_span(), "invalid unsizing {:?} -> {:?}", src.layout.ty, cast_ty)
            }
        }
    }

    fn unsize_into(
        &mut self,
        src: OpTy<'tcx, M::PointerTag>,
        cast_ty: TyAndLayout<'tcx>,
        dest: PlaceTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx> {
        trace!("Unsizing {:?} of type {} into {:?}", *src, src.layout.ty, cast_ty.ty);
        match (&src.layout.ty.kind(), &cast_ty.ty.kind()) {
            (&ty::Ref(_, s, _), &ty::Ref(_, c, _) | &ty::RawPtr(TypeAndMut { ty: c, .. }))
            | (&ty::RawPtr(TypeAndMut { ty: s, .. }), &ty::RawPtr(TypeAndMut { ty: c, .. })) => {
                self.unsize_into_ptr(src, dest, s, c)
            }
            (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) => {
                assert_eq!(def_a, def_b);
                if def_a.is_box() || def_b.is_box() {
                    if !def_a.is_box() || !def_b.is_box() {
                        span_bug!(
                            self.cur_span(),
                            "invalid unsizing between {:?} -> {:?}",
                            src.layout.ty,
                            cast_ty.ty
                        );
                    }
                    return self.unsize_into_ptr(
                        src,
                        dest,
                        src.layout.ty.boxed_ty(),
                        cast_ty.ty.boxed_ty(),
                    );
                }

                // unsizing of generic struct with pointer fields
                // Example: `Arc<T>` -> `Arc<Trait>`
                // here we need to increase the size of every &T thin ptr field to a fat ptr
                for i in 0..src.layout.fields.count() {
                    let cast_ty_field = cast_ty.field(self, i)?;
                    if cast_ty_field.is_zst() {
                        continue;
                    }
                    let src_field = self.operand_field(src, i)?;
                    let dst_field = self.place_field(dest, i)?;
                    if src_field.layout.ty == cast_ty_field.ty {
                        self.copy_op(src_field, dst_field)?;
                    } else {
                        self.unsize_into(src_field, cast_ty_field, dst_field)?;
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
