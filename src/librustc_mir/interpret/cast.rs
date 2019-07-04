use rustc::ty::{self, Ty, TypeAndMut};
use rustc::ty::layout::{self, TyLayout, Size};
use rustc::ty::adjustment::{PointerCast};
use syntax::ast::FloatTy;
use syntax::symbol::sym;

use rustc_apfloat::ieee::{Single, Double};
use rustc_apfloat::{Float, FloatConvert};
use rustc::mir::interpret::{
    Scalar, InterpResult, Pointer, PointerArithmetic, InterpError,
};
use rustc::mir::CastKind;

use super::{InterpCx, Machine, PlaceTy, OpTy, Immediate};

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    fn type_is_fat_ptr(&self, ty: Ty<'tcx>) -> bool {
        match ty.sty {
            ty::RawPtr(ty::TypeAndMut { ty, .. }) |
            ty::Ref(_, ty, _) => !self.type_is_sized(ty),
            ty::Adt(def, _) if def.is_box() => !self.type_is_sized(ty.boxed_ty()),
            _ => false,
        }
    }

    pub fn cast(
        &mut self,
        src: OpTy<'tcx, M::PointerTag>,
        kind: CastKind,
        dest: PlaceTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx> {
        use rustc::mir::CastKind::*;
        match kind {
            Pointer(PointerCast::Unsize) => {
                self.unsize_into(src, dest)?;
            }

            Misc | Pointer(PointerCast::MutToConstPointer) => {
                let src = self.read_immediate(src)?;

                if self.type_is_fat_ptr(src.layout.ty) {
                    match (*src, self.type_is_fat_ptr(dest.layout.ty)) {
                        // pointers to extern types
                        (Immediate::Scalar(_),_) |
                        // slices and trait objects to other slices/trait objects
                        (Immediate::ScalarPair(..), true) => {
                            // No change to immediate
                            self.write_immediate(*src, dest)?;
                        }
                        // slices and trait objects to thin pointers (dropping the metadata)
                        (Immediate::ScalarPair(data, _), false) => {
                            self.write_scalar(data, dest)?;
                        }
                    }
                } else {
                    match src.layout.variants {
                        layout::Variants::Single { index } => {
                            if let Some(discr) =
                                src.layout.ty.discriminant_for_variant(*self.tcx, index)
                            {
                                // Cast from a univariant enum
                                assert!(src.layout.is_zst());
                                return self.write_scalar(
                                    Scalar::from_uint(discr.val, dest.layout.size),
                                    dest);
                            }
                        }
                        layout::Variants::Multiple { .. } => {},
                    }

                    let dest_val = self.cast_scalar(src.to_scalar()?, src.layout, dest.layout)?;
                    self.write_scalar(dest_val, dest)?;
                }
            }

            Pointer(PointerCast::ReifyFnPointer) => {
                // The src operand does not matter, just its type
                match src.layout.ty.sty {
                    ty::FnDef(def_id, substs) => {
                        if self.tcx.has_attr(def_id, sym::rustc_args_required_const) {
                            bug!("reifying a fn ptr that requires const arguments");
                        }
                        let instance: InterpResult<'tcx, _> = ty::Instance::resolve(
                            *self.tcx,
                            self.param_env,
                            def_id,
                            substs,
                        ).ok_or_else(|| InterpError::TooGeneric.into());
                        let fn_ptr = self.memory.create_fn_alloc(instance?);
                        self.write_scalar(Scalar::Ptr(fn_ptr.into()), dest)?;
                    }
                    _ => bug!("reify fn pointer on {:?}", src.layout.ty),
                }
            }

            Pointer(PointerCast::UnsafeFnPointer) => {
                let src = self.read_immediate(src)?;
                match dest.layout.ty.sty {
                    ty::FnPtr(_) => {
                        // No change to value
                        self.write_immediate(*src, dest)?;
                    }
                    _ => bug!("fn to unsafe fn cast on {:?}", dest.layout.ty),
                }
            }

            Pointer(PointerCast::ClosureFnPointer(_)) => {
                // The src operand does not matter, just its type
                match src.layout.ty.sty {
                    ty::Closure(def_id, substs) => {
                        let substs = self.subst_and_normalize_erasing_regions(substs)?;
                        let instance = ty::Instance::resolve_closure(
                            *self.tcx,
                            def_id,
                            substs,
                            ty::ClosureKind::FnOnce,
                        );
                        let fn_ptr = self.memory.create_fn_alloc(instance);
                        let val = Immediate::Scalar(Scalar::Ptr(fn_ptr.into()).into());
                        self.write_immediate(val, dest)?;
                    }
                    _ => bug!("closure fn pointer on {:?}", src.layout.ty),
                }
            }
        }
        Ok(())
    }

    fn cast_scalar(
        &self,
        val: Scalar<M::PointerTag>,
        src_layout: TyLayout<'tcx>,
        dest_layout: TyLayout<'tcx>,
    ) -> InterpResult<'tcx, Scalar<M::PointerTag>> {
        use rustc::ty::TyKind::*;
        trace!("Casting {:?}: {:?} to {:?}", val, src_layout.ty, dest_layout.ty);

        match src_layout.ty.sty {
            // Floating point
            Float(FloatTy::F32) => self.cast_from_float(val.to_f32()?, dest_layout.ty),
            Float(FloatTy::F64) => self.cast_from_float(val.to_f64()?, dest_layout.ty),
            // Integer(-like), including fn ptr casts and casts from enums that
            // are represented as integers (this excludes univariant enums, which
            // are handled in `cast` directly).
            _ => {
                assert!(
                    src_layout.ty.is_bool()       || src_layout.ty.is_char()     ||
                    src_layout.ty.is_enum()       || src_layout.ty.is_integral() ||
                    src_layout.ty.is_unsafe_ptr() || src_layout.ty.is_fn_ptr()   ||
                    src_layout.ty.is_region_ptr(),
                    "Unexpected cast from type {:?}", src_layout.ty
                );
                match val.to_bits_or_ptr(src_layout.size, self) {
                    Err(ptr) => self.cast_from_ptr(ptr, src_layout, dest_layout),
                    Ok(data) => self.cast_from_int(data, src_layout, dest_layout),
                }
            }
        }
    }

    fn cast_from_int(
        &self,
        v: u128, // raw bits
        src_layout: TyLayout<'tcx>,
        dest_layout: TyLayout<'tcx>,
    ) -> InterpResult<'tcx, Scalar<M::PointerTag>> {
        // Let's make sure v is sign-extended *if* it has a signed type.
        let signed = src_layout.abi.is_signed();
        let v = if signed {
            self.sign_extend(v, src_layout)
        } else {
            v
        };
        trace!("cast_from_int: {}, {}, {}", v, src_layout.ty, dest_layout.ty);
        use rustc::ty::TyKind::*;
        match dest_layout.ty.sty {
            Int(_) | Uint(_) | RawPtr(_) => {
                let v = self.truncate(v, dest_layout);
                Ok(Scalar::from_uint(v, dest_layout.size))
            }

            Float(FloatTy::F32) if signed => Ok(Scalar::from_f32(
                Single::from_i128(v as i128).value
            )),
            Float(FloatTy::F64) if signed => Ok(Scalar::from_f64(
                Double::from_i128(v as i128).value
            )),
            Float(FloatTy::F32) => Ok(Scalar::from_f32(
                Single::from_u128(v).value
            )),
            Float(FloatTy::F64) => Ok(Scalar::from_f64(
                Double::from_u128(v).value
            )),

            Char => {
                // `u8` to `char` cast
                debug_assert_eq!(v as u8 as u128, v);
                Ok(Scalar::from_uint(v, Size::from_bytes(4)))
            },

            // Casts to bool are not permitted by rustc, no need to handle them here.
            _ => err!(Unimplemented(format!("int to {:?} cast", dest_layout.ty))),
        }
    }

    fn cast_from_float<F>(
        &self,
        f: F,
        dest_ty: Ty<'tcx>
    ) -> InterpResult<'tcx, Scalar<M::PointerTag>>
    where F: Float + Into<Scalar<M::PointerTag>> + FloatConvert<Single> + FloatConvert<Double>
    {
        use rustc::ty::TyKind::*;
        match dest_ty.sty {
            // float -> uint
            Uint(t) => {
                let width = t.bit_width().unwrap_or_else(|| self.pointer_size().bits() as usize);
                let v = f.to_u128(width).value;
                // This should already fit the bit width
                Ok(Scalar::from_uint(v, Size::from_bits(width as u64)))
            },
            // float -> int
            Int(t) => {
                let width = t.bit_width().unwrap_or_else(|| self.pointer_size().bits() as usize);
                let v = f.to_i128(width).value;
                Ok(Scalar::from_int(v, Size::from_bits(width as u64)))
            },
            // float -> f32
            Float(FloatTy::F32) =>
                Ok(Scalar::from_f32(f.convert(&mut false).value)),
            // float -> f64
            Float(FloatTy::F64) =>
                Ok(Scalar::from_f64(f.convert(&mut false).value)),
            // That's it.
            _ => bug!("invalid float to {:?} cast", dest_ty),
        }
    }

    fn cast_from_ptr(
        &self,
        ptr: Pointer<M::PointerTag>,
        src_layout: TyLayout<'tcx>,
        dest_layout: TyLayout<'tcx>,
    ) -> InterpResult<'tcx, Scalar<M::PointerTag>> {
        use rustc::ty::TyKind::*;

        match dest_layout.ty.sty {
            // Casting to a reference or fn pointer is not permitted by rustc,
            // no need to support it here.
            RawPtr(_) => Ok(ptr.into()),
            Int(_) | Uint(_) => {
                let size = self.memory.pointer_size();

                match self.force_bits(Scalar::Ptr(ptr), size) {
                    Ok(bits) => self.cast_from_int(bits, src_layout, dest_layout),
                    Err(_) if dest_layout.size == size => Ok(ptr.into()),
                    Err(e) => Err(e),
                }
            }
            _ => bug!("invalid MIR: ptr to {:?} cast", dest_layout.ty)
        }
    }

    fn unsize_into_ptr(
        &mut self,
        src: OpTy<'tcx, M::PointerTag>,
        dest: PlaceTy<'tcx, M::PointerTag>,
        // The pointee types
        sty: Ty<'tcx>,
        dty: Ty<'tcx>,
    ) -> InterpResult<'tcx> {
        // A<Struct> -> A<Trait> conversion
        let (src_pointee_ty, dest_pointee_ty) = self.tcx.struct_lockstep_tails(sty, dty);

        match (&src_pointee_ty.sty, &dest_pointee_ty.sty) {
            (&ty::Array(_, length), &ty::Slice(_)) => {
                let ptr = self.read_immediate(src)?.to_scalar_ptr()?;
                // u64 cast is from usize to u64, which is always good
                let val = Immediate::new_slice(
                    ptr,
                    length.unwrap_usize(self.tcx.tcx),
                    self,
                );
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
                let ptr = self.read_immediate(src)?.to_scalar_ptr()?;
                let val = Immediate::new_dyn_trait(ptr, vtable);
                self.write_immediate(val, dest)
            }

            _ => bug!("invalid unsizing {:?} -> {:?}", src.layout.ty, dest.layout.ty),
        }
    }

    fn unsize_into(
        &mut self,
        src: OpTy<'tcx, M::PointerTag>,
        dest: PlaceTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx> {
        trace!("Unsizing {:?} into {:?}", src, dest);
        match (&src.layout.ty.sty, &dest.layout.ty.sty) {
            (&ty::Ref(_, s, _), &ty::Ref(_, d, _)) |
            (&ty::Ref(_, s, _), &ty::RawPtr(TypeAndMut { ty: d, .. })) |
            (&ty::RawPtr(TypeAndMut { ty: s, .. }),
             &ty::RawPtr(TypeAndMut { ty: d, .. })) => {
                self.unsize_into_ptr(src, dest, s, d)
            }
            (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) => {
                assert_eq!(def_a, def_b);
                if def_a.is_box() || def_b.is_box() {
                    if !def_a.is_box() || !def_b.is_box() {
                        bug!("invalid unsizing between {:?} -> {:?}", src.layout, dest.layout);
                    }
                    return self.unsize_into_ptr(
                        src,
                        dest,
                        src.layout.ty.boxed_ty(),
                        dest.layout.ty.boxed_ty(),
                    );
                }

                // unsizing of generic struct with pointer fields
                // Example: `Arc<T>` -> `Arc<Trait>`
                // here we need to increase the size of every &T thin ptr field to a fat ptr
                for i in 0..src.layout.fields.count() {
                    let dst_field = self.place_field(dest, i as u64)?;
                    if dst_field.layout.is_zst() {
                        continue;
                    }
                    let src_field = self.operand_field(src, i as u64)?;
                    if src_field.layout.ty == dst_field.layout.ty {
                        self.copy_op(src_field, dst_field)?;
                    } else {
                        self.unsize_into(src_field, dst_field)?;
                    }
                }
                Ok(())
            }
            _ => {
                bug!(
                    "unsize_into: invalid conversion: {:?} -> {:?}",
                    src.layout,
                    dest.layout
                )
            }
        }
    }
}
