use rustc::ty::Ty;
use rustc::ty::layout::LayoutOf;
use syntax::ast::{FloatTy, IntTy, UintTy};

use rustc_apfloat::ieee::{Single, Double};
use super::{EvalContext, Machine};
use rustc::mir::interpret::{Scalar, EvalResult, Pointer, PointerArithmetic};
use rustc_apfloat::Float;

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
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
