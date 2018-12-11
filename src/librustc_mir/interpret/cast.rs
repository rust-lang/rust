// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty::{self, Ty, TypeAndMut};
use rustc::ty::layout::{self, TyLayout, Size};
use syntax::ast::{FloatTy, IntTy, UintTy};

use rustc_apfloat::ieee::{Single, Double};
use rustc::mir::interpret::{
    Scalar, EvalResult, Pointer, PointerArithmetic, EvalErrorKind, truncate
};
use rustc::mir::CastKind;
use rustc_apfloat::Float;

use super::{EvalContext, Machine, PlaceTy, OpTy, Immediate};

impl<'a, 'mir, 'tcx, M: Machine<'a, 'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
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
    ) -> EvalResult<'tcx> {
        use rustc::mir::CastKind::*;
        match kind {
            Unsize => {
                self.unsize_into(src, dest)?;
            }

            Misc => {
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
                            if let Some(def) = src.layout.ty.ty_adt_def() {
                                // Cast from a univariant enum
                                assert!(src.layout.is_zst());
                                let discr_val = def
                                    .discriminant_for_variant(*self.tcx, index)
                                    .val;
                                return self.write_scalar(
                                    Scalar::from_uint(discr_val, dest.layout.size),
                                    dest);
                            }
                        }
                        layout::Variants::Tagged { .. } |
                        layout::Variants::NicheFilling { .. } => {},
                    }

                    let dest_val = self.cast_scalar(src.to_scalar()?, src.layout, dest.layout)?;
                    self.write_scalar(dest_val, dest)?;
                }
            }

            ReifyFnPointer => {
                // The src operand does not matter, just its type
                match src.layout.ty.sty {
                    ty::FnDef(def_id, substs) => {
                        if self.tcx.has_attr(def_id, "rustc_args_required_const") {
                            bug!("reifying a fn ptr that requires \
                                    const arguments");
                        }
                        let instance: EvalResult<'tcx, _> = ty::Instance::resolve(
                            *self.tcx,
                            self.param_env,
                            def_id,
                            substs,
                        ).ok_or_else(|| EvalErrorKind::TooGeneric.into());
                        let fn_ptr = self.memory.create_fn_alloc(instance?).with_default_tag();
                        self.write_scalar(Scalar::Ptr(fn_ptr.into()), dest)?;
                    }
                    ref other => bug!("reify fn pointer on {:?}", other),
                }
            }

            UnsafeFnPointer => {
                let src = self.read_immediate(src)?;
                match dest.layout.ty.sty {
                    ty::FnPtr(_) => {
                        // No change to value
                        self.write_immediate(*src, dest)?;
                    }
                    ref other => bug!("fn to unsafe fn cast on {:?}", other),
                }
            }

            ClosureFnPointer => {
                // The src operand does not matter, just its type
                match src.layout.ty.sty {
                    ty::Closure(def_id, substs) => {
                        let substs = self.tcx.subst_and_normalize_erasing_regions(
                            self.substs(),
                            ty::ParamEnv::reveal_all(),
                            &substs,
                        );
                        let instance = ty::Instance::resolve_closure(
                            *self.tcx,
                            def_id,
                            substs,
                            ty::ClosureKind::FnOnce,
                        );
                        let fn_ptr = self.memory.create_fn_alloc(instance).with_default_tag();
                        let val = Immediate::Scalar(Scalar::Ptr(fn_ptr.into()).into());
                        self.write_immediate(val, dest)?;
                    }
                    ref other => bug!("closure fn pointer on {:?}", other),
                }
            }
        }
        Ok(())
    }

    pub(super) fn cast_scalar(
        &self,
        val: Scalar<M::PointerTag>,
        src_layout: TyLayout<'tcx>,
        dest_layout: TyLayout<'tcx>,
    ) -> EvalResult<'tcx, Scalar<M::PointerTag>> {
        use rustc::ty::TyKind::*;
        trace!("Casting {:?}: {:?} to {:?}", val, src_layout.ty, dest_layout.ty);

        match val {
            Scalar::Ptr(ptr) => self.cast_from_ptr(ptr, dest_layout.ty),
            Scalar::Bits { bits, size } => {
                debug_assert_eq!(size as u64, src_layout.size.bytes());
                debug_assert_eq!(truncate(bits, Size::from_bytes(size.into())), bits,
                    "Unexpected value of size {} before casting", size);

                let res = match src_layout.ty.sty {
                    Float(fty) => self.cast_from_float(bits, fty, dest_layout.ty)?,
                    _ => self.cast_from_int(bits, src_layout, dest_layout)?,
                };

                // Sanity check
                match res {
                    Scalar::Ptr(_) => bug!("Fabricated a ptr value from an int...?"),
                    Scalar::Bits { bits, size } => {
                        debug_assert_eq!(size as u64, dest_layout.size.bytes());
                        debug_assert_eq!(truncate(bits, Size::from_bytes(size.into())), bits,
                            "Unexpected value of size {} after casting", size);
                    }
                }
                // Done
                Ok(res)
            }
        }
    }

    fn cast_from_int(
        &self,
        v: u128,
        src_layout: TyLayout<'tcx>,
        dest_layout: TyLayout<'tcx>,
    ) -> EvalResult<'tcx, Scalar<M::PointerTag>> {
        let signed = src_layout.abi.is_signed();
        let v = if signed {
            self.sign_extend(v, src_layout)
        } else {
            v
        };
        trace!("cast_from_int: {}, {}, {}", v, src_layout.ty, dest_layout.ty);
        use rustc::ty::TyKind::*;
        match dest_layout.ty.sty {
            Int(_) | Uint(_) => {
                let v = self.truncate(v, dest_layout);
                Ok(Scalar::from_uint(v, dest_layout.size))
            }

            Float(FloatTy::F32) if signed => Ok(Scalar::from_uint(
                Single::from_i128(v as i128).value.to_bits(),
                Size::from_bits(32)
            )),
            Float(FloatTy::F64) if signed => Ok(Scalar::from_uint(
                Double::from_i128(v as i128).value.to_bits(),
                Size::from_bits(64)
            )),
            Float(FloatTy::F32) => Ok(Scalar::from_uint(
                Single::from_u128(v).value.to_bits(),
                Size::from_bits(32)
            )),
            Float(FloatTy::F64) => Ok(Scalar::from_uint(
                Double::from_u128(v).value.to_bits(),
                Size::from_bits(64)
            )),

            Char => {
                // `u8` to `char` cast
                debug_assert_eq!(v as u8 as u128, v);
                Ok(Scalar::from_uint(v, Size::from_bytes(4)))
            },

            // No alignment check needed for raw pointers.
            // But we have to truncate to target ptr size.
            RawPtr(_) => {
                Ok(Scalar::from_uint(
                    self.truncate_to_ptr(v).0,
                    self.pointer_size(),
                ))
            },

            // Casts to bool are not permitted by rustc, no need to handle them here.
            _ => err!(Unimplemented(format!("int to {:?} cast", dest_layout.ty))),
        }
    }

    fn cast_from_float(
        &self,
        bits: u128,
        fty: FloatTy,
        dest_ty: Ty<'tcx>
    ) -> EvalResult<'tcx, Scalar<M::PointerTag>> {
        use rustc::ty::TyKind::*;
        use rustc_apfloat::FloatConvert;
        match dest_ty.sty {
            // float -> uint
            Uint(t) => {
                let width = t.bit_width().unwrap_or_else(|| self.pointer_size().bits() as usize);
                let v = match fty {
                    FloatTy::F32 => Single::from_bits(bits).to_u128(width).value,
                    FloatTy::F64 => Double::from_bits(bits).to_u128(width).value,
                };
                // This should already fit the bit width
                Ok(Scalar::from_uint(v, Size::from_bits(width as u64)))
            },
            // float -> int
            Int(t) => {
                let width = t.bit_width().unwrap_or_else(|| self.pointer_size().bits() as usize);
                let v = match fty {
                    FloatTy::F32 => Single::from_bits(bits).to_i128(width).value,
                    FloatTy::F64 => Double::from_bits(bits).to_i128(width).value,
                };
                Ok(Scalar::from_int(v, Size::from_bits(width as u64)))
            },
            // f64 -> f32
            Float(FloatTy::F32) if fty == FloatTy::F64 => {
                Ok(Scalar::from_uint(
                    Single::to_bits(Double::from_bits(bits).convert(&mut false).value),
                    Size::from_bits(32),
                ))
            },
            // f32 -> f64
            Float(FloatTy::F64) if fty == FloatTy::F32 => {
                Ok(Scalar::from_uint(
                    Double::to_bits(Single::from_bits(bits).convert(&mut false).value),
                    Size::from_bits(64),
                ))
            },
            // identity cast
            Float(FloatTy:: F64) => Ok(Scalar::from_uint(bits, Size::from_bits(64))),
            Float(FloatTy:: F32) => Ok(Scalar::from_uint(bits, Size::from_bits(32))),
            _ => err!(Unimplemented(format!("float to {:?} cast", dest_ty))),
        }
    }

    fn cast_from_ptr(
        &self,
        ptr: Pointer<M::PointerTag>,
        ty: Ty<'tcx>
    ) -> EvalResult<'tcx, Scalar<M::PointerTag>> {
        use rustc::ty::TyKind::*;
        match ty.sty {
            // Casting to a reference or fn pointer is not permitted by rustc,
            // no need to support it here.
            RawPtr(_) |
            Int(IntTy::Isize) |
            Uint(UintTy::Usize) => Ok(ptr.into()),
            Int(_) | Uint(_) => err!(ReadPointerAsBytes),
            _ => err!(Unimplemented(format!("ptr to {:?} cast", ty))),
        }
    }

    fn unsize_into_ptr(
        &mut self,
        src: OpTy<'tcx, M::PointerTag>,
        dest: PlaceTy<'tcx, M::PointerTag>,
        // The pointee types
        sty: Ty<'tcx>,
        dty: Ty<'tcx>,
    ) -> EvalResult<'tcx> {
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
    ) -> EvalResult<'tcx> {
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
                    let src_field = match src.try_as_mplace() {
                        Ok(mplace) => {
                            let src_field = self.mplace_field(mplace, i as u64)?;
                            src_field.into()
                        }
                        Err(..) => {
                            let src_field_layout = src.layout.field(self, i)?;
                            // this must be a field covering the entire thing
                            assert_eq!(src.layout.fields.offset(i).bytes(), 0);
                            assert_eq!(src_field_layout.size, src.layout.size);
                            // just sawp out the layout
                            OpTy { op: src.op, layout: src_field_layout }
                        }
                    };
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
