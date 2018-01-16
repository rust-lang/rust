use rustc::ty::Ty;
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
        trace!("Casting {:?}: {:?} to {:?}", val, src_ty, dest_ty);
        let src_kind = self.ty_to_primval_kind(src_ty)?;

        match val {
            PrimVal::Undef => Ok(PrimVal::Undef),
            PrimVal::Ptr(ptr) => self.cast_from_ptr(ptr, dest_ty),
            val @ PrimVal::Bytes(_) => {
                use rustc::mir::interpret::PrimValKind::*;
                match src_kind {
                    F32 => self.cast_from_float(val.to_f32()?, dest_ty),
                    F64 => self.cast_from_float(val.to_f64()?, dest_ty),

                    I8 | I16 | I32 | I64 | I128 => {
                        self.cast_from_signed_int(val.to_i128()?, dest_ty)
                    }

                    Bool | Char | U8 | U16 | U32 | U64 | U128 | FnPtr | Ptr => {
                        self.cast_from_int(val.to_u128()?, dest_ty, false)
                    }
                }
            }
        }
    }

    fn cast_from_signed_int(&self, val: i128, ty: Ty<'tcx>) -> EvalResult<'tcx, PrimVal> {
        self.cast_from_int(val as u128, ty, val < 0)
    }

    fn int_to_int(&self, v: i128, ty: IntTy) -> u128 {
        match ty {
            IntTy::I8 => v as i8 as u128,
            IntTy::I16 => v as i16 as u128,
            IntTy::I32 => v as i32 as u128,
            IntTy::I64 => v as i64 as u128,
            IntTy::I128 => v as u128,
            IntTy::Isize => {
                let ty = self.tcx.sess.target.isize_ty;
                self.int_to_int(v, ty)
            }
        }
    }
    fn int_to_uint(&self, v: u128, ty: UintTy) -> u128 {
        match ty {
            UintTy::U8 => v as u8 as u128,
            UintTy::U16 => v as u16 as u128,
            UintTy::U32 => v as u32 as u128,
            UintTy::U64 => v as u64 as u128,
            UintTy::U128 => v,
            UintTy::Usize => {
                let ty = self.tcx.sess.target.usize_ty;
                self.int_to_uint(v, ty)
            }
        }
    }

    fn cast_from_int(
        &self,
        v: u128,
        ty: Ty<'tcx>,
        negative: bool,
    ) -> EvalResult<'tcx, PrimVal> {
        trace!("cast_from_int: {}, {}, {}", v, ty, negative);
        use rustc::ty::TypeVariants::*;
        match ty.sty {
            // Casts to bool are not permitted by rustc, no need to handle them here.
            TyInt(ty) => Ok(PrimVal::Bytes(self.int_to_int(v as i128, ty))),
            TyUint(ty) => Ok(PrimVal::Bytes(self.int_to_uint(v, ty))),

            TyFloat(fty) if negative => Ok(PrimVal::Bytes(ConstFloat::from_i128(v as i128, fty).bits)),
            TyFloat(fty) => Ok(PrimVal::Bytes(ConstFloat::from_u128(v, fty).bits)),

            TyChar if v as u8 as u128 == v => Ok(PrimVal::Bytes(v)),
            TyChar => err!(InvalidChar(v)),

            // No alignment check needed for raw pointers.  But we have to truncate to target ptr size.
            TyRawPtr(_) => Ok(PrimVal::Bytes(self.memory.truncate_to_ptr(v).0 as u128)),

            _ => err!(Unimplemented(format!("int to {:?} cast", ty))),
        }
    }

    fn cast_from_float(&self, val: ConstFloat, ty: Ty<'tcx>) -> EvalResult<'tcx, PrimVal> {
        use rustc::ty::TypeVariants::*;
        match ty.sty {
            TyUint(t) => {
                let width = t.bit_width().unwrap_or(self.memory.pointer_size() as usize * 8);
                match val.ty {
                    FloatTy::F32 => Ok(PrimVal::Bytes(Single::from_bits(val.bits).to_u128(width).value)),
                    FloatTy::F64 => Ok(PrimVal::Bytes(Double::from_bits(val.bits).to_u128(width).value)),
                }
            },

            TyInt(t) => {
                let width = t.bit_width().unwrap_or(self.memory.pointer_size() as usize * 8);
                match val.ty {
                    FloatTy::F32 => Ok(PrimVal::from_i128(Single::from_bits(val.bits).to_i128(width).value)),
                    FloatTy::F64 => Ok(PrimVal::from_i128(Double::from_bits(val.bits).to_i128(width).value)),
                }
            },

            TyFloat(fty) => Ok(PrimVal::from_float(val.convert(fty))),
            _ => err!(Unimplemented(format!("float to {:?} cast", ty))),
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
