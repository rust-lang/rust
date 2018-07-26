use rustc::ty::{self, Ty};
use rustc::ty::layout::{self, LayoutOf};
use syntax::ast::{FloatTy, IntTy, UintTy};

use rustc_apfloat::ieee::{Single, Double};
use super::{EvalContext, Machine};
use rustc::mir::interpret::{Scalar, EvalResult, Pointer, PointerArithmetic, Value, EvalErrorKind};
use rustc::mir::CastKind;
use rustc_apfloat::Float;
use interpret::eval_context::ValTy;
use interpret::Place;

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    crate fn cast(
        &mut self,
        src: ValTy<'tcx>,
        kind: CastKind,
        dest_ty: Ty<'tcx>,
        dest: Place,
    ) -> EvalResult<'tcx> {
        use rustc::mir::CastKind::*;
        match kind {
            Unsize => {
                let src_layout = self.layout_of(src.ty)?;
                let dst_layout = self.layout_of(dest_ty)?;
                self.unsize_into(src.value, src_layout, dest, dst_layout)?;
            }

            Misc => {
                if self.type_is_fat_ptr(src.ty) {
                    match (src.value, self.type_is_fat_ptr(dest_ty)) {
                        (Value::ByRef { .. }, _) |
                        // pointers to extern types
                        (Value::Scalar(_),_) |
                        // slices and trait objects to other slices/trait objects
                        (Value::ScalarPair(..), true) => {
                            let valty = ValTy {
                                value: src.value,
                                ty: dest_ty,
                            };
                            self.write_value(valty, dest)?;
                        }
                        // slices and trait objects to thin pointers (dropping the metadata)
                        (Value::ScalarPair(data, _), false) => {
                            let valty = ValTy {
                                value: Value::Scalar(data),
                                ty: dest_ty,
                            };
                            self.write_value(valty, dest)?;
                        }
                    }
                } else {
                    let src_layout = self.layout_of(src.ty)?;
                    match src_layout.variants {
                        layout::Variants::Single { index } => {
                            if let Some(def) = src.ty.ty_adt_def() {
                                let discr_val = def
                                    .discriminant_for_variant(*self.tcx, index)
                                    .val;
                                let defined = self
                                    .layout_of(dest_ty)
                                    .unwrap()
                                    .size
                                    .bits() as u8;
                                return self.write_scalar(
                                    dest,
                                    Scalar::Bits {
                                        bits: discr_val,
                                        defined,
                                    },
                                    dest_ty);
                            }
                        }
                        layout::Variants::Tagged { .. } |
                        layout::Variants::NicheFilling { .. } => {},
                    }

                    let src_val = self.value_to_scalar(src)?;
                    let dest_val = self.cast_scalar(src_val, src.ty, dest_ty)?;
                    let valty = ValTy {
                        value: Value::Scalar(dest_val),
                        ty: dest_ty,
                    };
                    self.write_value(valty, dest)?;
                }
            }

            ReifyFnPointer => {
                match src.ty.sty {
                    ty::TyFnDef(def_id, substs) => {
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
                        let fn_ptr = self.memory.create_fn_alloc(instance?);
                        let valty = ValTy {
                            value: Value::Scalar(fn_ptr.into()),
                            ty: dest_ty,
                        };
                        self.write_value(valty, dest)?;
                    }
                    ref other => bug!("reify fn pointer on {:?}", other),
                }
            }

            UnsafeFnPointer => {
                match dest_ty.sty {
                    ty::TyFnPtr(_) => {
                        let mut src = src;
                        src.ty = dest_ty;
                        self.write_value(src, dest)?;
                    }
                    ref other => bug!("fn to unsafe fn cast on {:?}", other),
                }
            }

            ClosureFnPointer => {
                match src.ty.sty {
                    ty::TyClosure(def_id, substs) => {
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
                        let fn_ptr = self.memory.create_fn_alloc(instance);
                        let valty = ValTy {
                            value: Value::Scalar(fn_ptr.into()),
                            ty: dest_ty,
                        };
                        self.write_value(valty, dest)?;
                    }
                    ref other => bug!("closure fn pointer on {:?}", other),
                }
            }
        }
        Ok(())
    }

    pub(super) fn cast_scalar(
        &self,
        val: Scalar,
        src_ty: Ty<'tcx>,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, Scalar> {
        use rustc::ty::TypeVariants::*;
        trace!("Casting {:?}: {:?} to {:?}", val, src_ty, dest_ty);

        match val {
            Scalar::Bits { defined: 0, .. } => Ok(val),
            Scalar::Ptr(ptr) => self.cast_from_ptr(ptr, dest_ty),
            Scalar::Bits { bits, .. } => {
                // TODO(oli-obk): check defined bits here
                match src_ty.sty {
                    TyFloat(fty) => self.cast_from_float(bits, fty, dest_ty),
                    _ => self.cast_from_int(bits, src_ty, dest_ty),
                }
            }
        }
    }

    fn cast_from_int(
        &self,
        v: u128,
        src_ty: Ty<'tcx>,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, Scalar> {
        let signed = self.layout_of(src_ty)?.abi.is_signed();
        let v = if signed {
            self.sign_extend(v, src_ty)?
        } else {
            v
        };
        trace!("cast_from_int: {}, {}, {}", v, src_ty, dest_ty);
        use rustc::ty::TypeVariants::*;
        match dest_ty.sty {
            TyInt(_) | TyUint(_) => {
                let v = self.truncate(v, dest_ty)?;
                Ok(Scalar::Bits {
                    bits: v,
                    defined: self.layout_of(dest_ty).unwrap().size.bits() as u8,
                })
            }

            TyFloat(FloatTy::F32) if signed => Ok(Scalar::Bits {
                bits: Single::from_i128(v as i128).value.to_bits(),
                defined: 32,
            }),
            TyFloat(FloatTy::F64) if signed => Ok(Scalar::Bits {
                bits: Double::from_i128(v as i128).value.to_bits(),
                defined: 64,
            }),
            TyFloat(FloatTy::F32) => Ok(Scalar::Bits {
                bits: Single::from_u128(v).value.to_bits(),
                defined: 32,
            }),
            TyFloat(FloatTy::F64) => Ok(Scalar::Bits {
                bits: Double::from_u128(v).value.to_bits(),
                defined: 64,
            }),

            TyChar if v as u8 as u128 == v => Ok(Scalar::Bits { bits: v, defined: 32 }),
            TyChar => err!(InvalidChar(v)),

            // No alignment check needed for raw pointers.  But we have to truncate to target ptr size.
            TyRawPtr(_) => {
                Ok(Scalar::Bits {
                    bits: self.memory.truncate_to_ptr(v).0 as u128,
                    defined: self.memory.pointer_size().bits() as u8,
                })
            },

            // Casts to bool are not permitted by rustc, no need to handle them here.
            _ => err!(Unimplemented(format!("int to {:?} cast", dest_ty))),
        }
    }

    fn cast_from_float(&self, bits: u128, fty: FloatTy, dest_ty: Ty<'tcx>) -> EvalResult<'tcx, Scalar> {
        use rustc::ty::TypeVariants::*;
        use rustc_apfloat::FloatConvert;
        match dest_ty.sty {
            // float -> uint
            TyUint(t) => {
                let width = t.bit_width().unwrap_or(self.memory.pointer_size().bits() as usize);
                match fty {
                    FloatTy::F32 => Ok(Scalar::Bits {
                        bits: Single::from_bits(bits).to_u128(width).value,
                        defined: width as u8,
                    }),
                    FloatTy::F64 => Ok(Scalar::Bits {
                        bits: Double::from_bits(bits).to_u128(width).value,
                        defined: width as u8,
                    }),
                }
            },
            // float -> int
            TyInt(t) => {
                let width = t.bit_width().unwrap_or(self.memory.pointer_size().bits() as usize);
                match fty {
                    FloatTy::F32 => Ok(Scalar::Bits {
                        bits: Single::from_bits(bits).to_i128(width).value as u128,
                        defined: width as u8,
                    }),
                    FloatTy::F64 => Ok(Scalar::Bits {
                        bits: Double::from_bits(bits).to_i128(width).value as u128,
                        defined: width as u8,
                    }),
                }
            },
            // f64 -> f32
            TyFloat(FloatTy::F32) if fty == FloatTy::F64 => {
                Ok(Scalar::Bits {
                    bits: Single::to_bits(Double::from_bits(bits).convert(&mut false).value),
                    defined: 32,
                })
            },
            // f32 -> f64
            TyFloat(FloatTy::F64) if fty == FloatTy::F32 => {
                Ok(Scalar::Bits {
                    bits: Double::to_bits(Single::from_bits(bits).convert(&mut false).value),
                    defined: 64,
                })
            },
            // identity cast
            TyFloat(FloatTy:: F64) => Ok(Scalar::Bits {
                bits,
                defined: 64,
            }),
            TyFloat(FloatTy:: F32) => Ok(Scalar::Bits {
                bits,
                defined: 32,
            }),
            _ => err!(Unimplemented(format!("float to {:?} cast", dest_ty))),
        }
    }

    fn cast_from_ptr(&self, ptr: Pointer, ty: Ty<'tcx>) -> EvalResult<'tcx, Scalar> {
        use rustc::ty::TypeVariants::*;
        match ty.sty {
            // Casting to a reference or fn pointer is not permitted by rustc, no need to support it here.
            TyRawPtr(_) |
            TyInt(IntTy::Isize) |
            TyUint(UintTy::Usize) => Ok(ptr.into()),
            TyInt(_) | TyUint(_) => err!(ReadPointerAsBytes),
            _ => err!(Unimplemented(format!("ptr to {:?} cast", ty))),
        }
    }
}
