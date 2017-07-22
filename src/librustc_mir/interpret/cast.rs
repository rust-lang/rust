use rustc::ty::{self, Ty};
use syntax::ast::{FloatTy, IntTy, UintTy};

use error::{EvalResult, EvalError};
use eval_context::EvalContext;
use value::PrimVal;
use memory::{MemoryPointer, HasDataLayout};

impl<'a, 'tcx> EvalContext<'a, 'tcx> {
    pub(super) fn cast_primval(
        &self,
        val: PrimVal,
        src_ty: Ty<'tcx>,
        dest_ty: Ty<'tcx>
    ) -> EvalResult<'tcx, PrimVal> {
        let src_kind = self.ty_to_primval_kind(src_ty)?;

        match val {
            PrimVal::Undef => Ok(PrimVal::Undef),
            PrimVal::Ptr(ptr) => self.cast_from_ptr(ptr, dest_ty),
            val @ PrimVal::Bytes(_) => {
                use value::PrimValKind::*;
                match src_kind {
                    F32 => self.cast_from_float(val.to_f32()? as f64, dest_ty),
                    F64 => self.cast_from_float(val.to_f64()?, dest_ty),

                    I8 | I16 | I32 | I64 | I128 => {
                        self.cast_from_signed_int(val.to_i128()?, dest_ty)
                    },

                    Bool | Char | U8 | U16 | U32 | U64 | U128 | FnPtr | Ptr => {
                        self.cast_from_int(val.to_u128()?, dest_ty, false)
                    },
                }
            }
        }
    }

    fn cast_from_signed_int(&self, val: i128, ty: ty::Ty<'tcx>) -> EvalResult<'tcx, PrimVal> {
        self.cast_from_int(val as u128, ty, val < 0)
    }

    fn cast_from_int(&self, v: u128, ty: ty::Ty<'tcx>, negative: bool) -> EvalResult<'tcx, PrimVal> {
        use rustc::ty::TypeVariants::*;
        match ty.sty {
            // Casts to bool are not permitted by rustc, no need to handle them here.

            TyInt(IntTy::I8)  => Ok(PrimVal::Bytes(v as i128 as i8  as u128)),
            TyInt(IntTy::I16) => Ok(PrimVal::Bytes(v as i128 as i16 as u128)),
            TyInt(IntTy::I32) => Ok(PrimVal::Bytes(v as i128 as i32 as u128)),
            TyInt(IntTy::I64) => Ok(PrimVal::Bytes(v as i128 as i64 as u128)),
            TyInt(IntTy::I128) => Ok(PrimVal::Bytes(v as u128)),

            TyUint(UintTy::U8)  => Ok(PrimVal::Bytes(v as u8  as u128)),
            TyUint(UintTy::U16) => Ok(PrimVal::Bytes(v as u16 as u128)),
            TyUint(UintTy::U32) => Ok(PrimVal::Bytes(v as u32 as u128)),
            TyUint(UintTy::U64) => Ok(PrimVal::Bytes(v as u64 as u128)),
            TyUint(UintTy::U128) => Ok(PrimVal::Bytes(v)),

            TyInt(IntTy::Is) => {
                let int_ty = self.tcx.sess.target.int_type;
                let ty = self.tcx.mk_mach_int(int_ty);
                self.cast_from_int(v, ty, negative)
            }

            TyUint(UintTy::Us) => {
                let uint_ty = self.tcx.sess.target.uint_type;
                let ty = self.tcx.mk_mach_uint(uint_ty);
                self.cast_from_int(v, ty, negative)
            }

            TyFloat(FloatTy::F64) if negative => Ok(PrimVal::from_f64(v as i128 as f64)),
            TyFloat(FloatTy::F64)             => Ok(PrimVal::from_f64(v as f64)),
            TyFloat(FloatTy::F32) if negative => Ok(PrimVal::from_f32(v as i128 as f32)),
            TyFloat(FloatTy::F32)             => Ok(PrimVal::from_f32(v as f32)),

            TyChar if v as u8 as u128 == v => Ok(PrimVal::Bytes(v)),
            TyChar => Err(EvalError::InvalidChar(v)),

            // No alignment check needed for raw pointers.  But we have to truncate to target ptr size.
            TyRawPtr(_) => Ok(PrimVal::Bytes(self.memory.truncate_to_ptr(v).0 as u128)),

            _ => Err(EvalError::Unimplemented(format!("int to {:?} cast", ty))),
        }
    }

    fn cast_from_float(&self, val: f64, ty: Ty<'tcx>) -> EvalResult<'tcx, PrimVal> {
        use rustc::ty::TypeVariants::*;
        match ty.sty {
            // Casting negative floats to unsigned integers yields zero.
            TyUint(_) if val < 0.0 => self.cast_from_int(0, ty, false),
            TyInt(_)  if val < 0.0 => self.cast_from_int(val as i128 as u128, ty, true),

            TyInt(_) | ty::TyUint(_) => self.cast_from_int(val as u128, ty, false),

            TyFloat(FloatTy::F64) => Ok(PrimVal::from_f64(val)),
            TyFloat(FloatTy::F32) => Ok(PrimVal::from_f32(val as f32)),
            _ => Err(EvalError::Unimplemented(format!("float to {:?} cast", ty))),
        }
    }

    fn cast_from_ptr(&self, ptr: MemoryPointer, ty: Ty<'tcx>) -> EvalResult<'tcx, PrimVal> {
        use rustc::ty::TypeVariants::*;
        match ty.sty {
            // Casting to a reference or fn pointer is not permitted by rustc, no need to support it here.
            TyRawPtr(_) | TyInt(IntTy::Is) | TyUint(UintTy::Us) =>
                Ok(PrimVal::Ptr(ptr)),
            TyInt(_) | TyUint(_) => Err(EvalError::ReadPointerAsBytes),
            _ => Err(EvalError::Unimplemented(format!("ptr to {:?} cast", ty))),
        }
    }
}
