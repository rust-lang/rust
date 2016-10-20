
use super::{
    EvalContext,
};
use error::{EvalResult, EvalError};
use rustc::ty;
use primval::PrimVal;
use memory::Pointer;

use rustc::ty::Ty;
use syntax::ast::{FloatTy, IntTy, UintTy};

impl<'a, 'tcx> EvalContext<'a, 'tcx> {
    pub(super) fn cast_primval(&self, val: PrimVal, ty: Ty<'tcx>) -> EvalResult<'tcx, PrimVal> {
        use primval::PrimValKind::*;
        match val.kind {
            F32 => self.cast_float(val.to_f32() as f64, ty),
            F64 => self.cast_float(val.to_f64(), ty),

            I8 | I16 | I32 | I64 => self.cast_signed_int(val.bits as i64, ty),

            Bool | Char | U8 | U16 | U32 | U64 => self.cast_int(val.bits, ty, false),

            FnPtr(alloc) | Ptr(alloc) => {
                let ptr = Pointer::new(alloc, val.bits as usize);
                self.cast_ptr(ptr, ty)
            }
        }
    }

    fn cast_signed_int(&self, val: i64, ty: ty::Ty<'tcx>) -> EvalResult<'tcx, PrimVal> {
        self.cast_int(val as u64, ty, val < 0)
    }

    fn cast_int(&self, v: u64, ty: ty::Ty<'tcx>, negative: bool) -> EvalResult<'tcx, PrimVal> {
        use primval::PrimValKind::*;
        use rustc::ty::TypeVariants::*;
        match ty.sty {
            TyBool if v == 0 => Ok(PrimVal::from_bool(false)),
            TyBool if v == 1 => Ok(PrimVal::from_bool(true)),
            TyBool => Err(EvalError::InvalidBool),

            TyInt(IntTy::I8)  => Ok(PrimVal::new(v as i64 as i8  as u64, I8)),
            TyInt(IntTy::I16) => Ok(PrimVal::new(v as i64 as i16 as u64, I16)),
            TyInt(IntTy::I32) => Ok(PrimVal::new(v as i64 as i32 as u64, I32)),
            TyInt(IntTy::I64) => Ok(PrimVal::new(v as i64 as i64 as u64, I64)),

            TyUint(UintTy::U8)  => Ok(PrimVal::new(v as u8  as u64, U8)),
            TyUint(UintTy::U16) => Ok(PrimVal::new(v as u16 as u64, U16)),
            TyUint(UintTy::U32) => Ok(PrimVal::new(v as u32 as u64, U32)),
            TyUint(UintTy::U64) => Ok(PrimVal::new(v, U64)),

            TyInt(IntTy::Is) => {
                let int_ty = self.tcx.sess.target.int_type;
                let ty = self.tcx.mk_mach_int(int_ty);
                self.cast_int(v, ty, negative)
            }

            TyUint(UintTy::Us) => {
                let uint_ty = self.tcx.sess.target.uint_type;
                let ty = self.tcx.mk_mach_uint(uint_ty);
                self.cast_int(v, ty, negative)
            }

            TyFloat(FloatTy::F64) if negative => Ok(PrimVal::from_f64(v as i64 as f64)),
            TyFloat(FloatTy::F64)             => Ok(PrimVal::from_f64(v as f64)),
            TyFloat(FloatTy::F32) if negative => Ok(PrimVal::from_f32(v as i64 as f32)),
            TyFloat(FloatTy::F32)             => Ok(PrimVal::from_f32(v as f32)),

            TyChar if v as u8 as u64 == v => Ok(PrimVal::new(v, Char)),
            TyChar => Err(EvalError::InvalidChar(v)),

            TyRawPtr(_) => Ok(PrimVal::from_ptr(Pointer::from_int(v as usize))),

            _ => Err(EvalError::Unimplemented(format!("int to {:?} cast", ty))),
        }
    }

    fn cast_float(&self, val: f64, ty: Ty<'tcx>) -> EvalResult<'tcx, PrimVal> {
        use rustc::ty::TypeVariants::*;
        match ty.sty {
            // Casting negative floats to unsigned integers yields zero.
            TyUint(_) if val < 0.0 => self.cast_int(0, ty, false),
            TyInt(_)  if val < 0.0 => self.cast_int(val as i64 as u64, ty, true),

            TyInt(_) | ty::TyUint(_) => self.cast_int(val as u64, ty, false),

            TyFloat(FloatTy::F64) => Ok(PrimVal::from_f64(val)),
            TyFloat(FloatTy::F32) => Ok(PrimVal::from_f32(val as f32)),
            _ => Err(EvalError::Unimplemented(format!("float to {:?} cast", ty))),
        }
    }

    fn cast_ptr(&self, ptr: Pointer, ty: Ty<'tcx>) -> EvalResult<'tcx, PrimVal> {
        use primval::PrimValKind::*;
        use rustc::ty::TypeVariants::*;
        match ty.sty {
            TyRef(..) | TyRawPtr(_) => Ok(PrimVal::from_ptr(ptr)),
            TyFnPtr(_) => Ok(PrimVal::from_fn_ptr(ptr)),

            TyInt(IntTy::I8)  => Ok(PrimVal::new(ptr.to_int()? as u64, I8)),
            TyInt(IntTy::I16) => Ok(PrimVal::new(ptr.to_int()? as u64, I16)),
            TyInt(IntTy::I32) => Ok(PrimVal::new(ptr.to_int()? as u64, I32)),
            TyInt(IntTy::I64) => Ok(PrimVal::new(ptr.to_int()? as u64, I64)),

            TyUint(UintTy::U8)  => Ok(PrimVal::new(ptr.to_int()? as u64, U8)),
            TyUint(UintTy::U16) => Ok(PrimVal::new(ptr.to_int()? as u64, U16)),
            TyUint(UintTy::U32) => Ok(PrimVal::new(ptr.to_int()? as u64, U32)),
            TyUint(UintTy::U64) => Ok(PrimVal::new(ptr.to_int()? as u64, U64)),

            _ => Err(EvalError::Unimplemented(format!("ptr to {:?} cast", ty))),
        }
    }
}
