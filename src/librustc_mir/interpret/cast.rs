use rustc::ty::Ty;
use rustc::ty::layout::LayoutOf;
use syntax::ast::{FloatTy, IntTy, UintTy};

use rustc_const_math::ConstFloat;
use super::{EvalContext, Machine};
use rustc::mir::interpret::{PrimVal, EvalResult, MemoryPointer, PointerArithmetic};
use rustc_apfloat::ieee::{Single, Double};
use rustc_apfloat::Float;

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    pub(super) fn cast_primval(
        &self,
        val: PrimVal,
        src_ty: Ty<'tcx>,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, PrimVal> {
        use rustc::ty::TypeVariants::*;
        trace!("Casting {:?}: {:?} to {:?}", val, src_ty, dest_ty);

        match val {
            PrimVal::Undef => Ok(PrimVal::Undef),
            PrimVal::Ptr(ptr) => self.cast_from_ptr(ptr, dest_ty),
            PrimVal::Bytes(b) => {
                match src_ty.sty {
                    TyFloat(fty) => self.cast_from_float(b, fty, dest_ty),
                    _ => self.cast_from_int(b, src_ty, dest_ty),
                }
            }
        }
    }

    fn cast_from_int(
        &self,
        v: u128,
        src_ty: Ty<'tcx>,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, PrimVal> {
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
                Ok(PrimVal::Bytes(v))
            }

            TyFloat(fty) if signed => Ok(PrimVal::Bytes(ConstFloat::from_i128(v as i128, fty).bits)),
            TyFloat(fty) => Ok(PrimVal::Bytes(ConstFloat::from_u128(v, fty).bits)),

            TyChar if v as u8 as u128 == v => Ok(PrimVal::Bytes(v)),
            TyChar => err!(InvalidChar(v)),

            // No alignment check needed for raw pointers.  But we have to truncate to target ptr size.
            TyRawPtr(_) => {
                Ok(PrimVal::Bytes(self.memory.truncate_to_ptr(v).0 as u128))
            },

            // Casts to bool are not permitted by rustc, no need to handle them here.
            _ => err!(Unimplemented(format!("int to {:?} cast", dest_ty))),
        }
    }

    fn cast_from_float(&self, bits: u128, fty: FloatTy, dest_ty: Ty<'tcx>) -> EvalResult<'tcx, PrimVal> {
        use rustc::ty::TypeVariants::*;
        use rustc_apfloat::FloatConvert;
        match dest_ty.sty {
            // float -> uint
            TyUint(t) => {
                let width = t.bit_width().unwrap_or(self.memory.pointer_size() as usize * 8);
                match fty {
                    FloatTy::F32 => Ok(PrimVal::Bytes(Single::from_bits(bits).to_u128(width).value)),
                    FloatTy::F64 => Ok(PrimVal::Bytes(Double::from_bits(bits).to_u128(width).value)),
                }
            },
            // float -> int
            TyInt(t) => {
                let width = t.bit_width().unwrap_or(self.memory.pointer_size() as usize * 8);
                match fty {
                    FloatTy::F32 => Ok(PrimVal::from_i128(Single::from_bits(bits).to_i128(width).value)),
                    FloatTy::F64 => Ok(PrimVal::from_i128(Double::from_bits(bits).to_i128(width).value)),
                }
            },
            // f64 -> f32
            TyFloat(FloatTy::F32) if fty == FloatTy::F64 => {
                Ok(PrimVal::Bytes(Single::to_bits(Double::from_bits(bits).convert(&mut false).value)))
            },
            // f32 -> f64
            TyFloat(FloatTy::F64) if fty == FloatTy::F32 => {
                Ok(PrimVal::Bytes(Double::to_bits(Single::from_bits(bits).convert(&mut false).value)))
            },
            // identity cast
            TyFloat(_) => Ok(PrimVal::Bytes(bits)),
            _ => err!(Unimplemented(format!("float to {:?} cast", dest_ty))),
        }
    }

    fn cast_from_ptr(&self, ptr: MemoryPointer, ty: Ty<'tcx>) -> EvalResult<'tcx, PrimVal> {
        use rustc::ty::TypeVariants::*;
        match ty.sty {
            // Casting to a reference or fn pointer is not permitted by rustc, no need to support it here.
            TyRawPtr(_) |
            TyInt(IntTy::Isize) |
            TyUint(UintTy::Usize) => Ok(PrimVal::Ptr(ptr)),
            TyInt(_) | TyUint(_) => err!(ReadPointerAsBytes),
            _ => err!(Unimplemented(format!("ptr to {:?} cast", ty))),
        }
    }
}
