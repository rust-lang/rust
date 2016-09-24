
use super::{
    EvalContext,
};
use error::{EvalResult, EvalError};
use rustc::ty;
use primval::PrimVal;
use memory::Pointer;

use rustc::ty::Ty;
use syntax::ast;

impl<'a, 'tcx> EvalContext<'a, 'tcx> {
    pub(super) fn cast_primval(&self, val: PrimVal, ty: Ty<'tcx>) -> EvalResult<'tcx, PrimVal> {
        use primval::PrimVal::*;
        match val {
            Bool(b) => self.cast_const_int(b as u64, ty, false),
            F32(f) => self.cast_const_float(f as f64, ty),
            F64(f) => self.cast_const_float(f, ty),
            I8(i) => self.cast_signed_int(i as i64, ty),
            I16(i) => self.cast_signed_int(i as i64, ty),
            I32(i) => self.cast_signed_int(i as i64, ty),
            I64(i) => self.cast_signed_int(i, ty),
            U8(u) => self.cast_const_int(u as u64, ty, false),
            U16(u) => self.cast_const_int(u as u64, ty, false),
            U32(u) => self.cast_const_int(u as u64, ty, false),
            Char(c) => self.cast_const_int(c as u64, ty, false),
            U64(u) => self.cast_const_int(u, ty, false),
            FnPtr(ptr) |
            Ptr(ptr) => self.cast_ptr(ptr, ty),
        }
    }

    fn cast_ptr(&self, ptr: Pointer, ty: Ty<'tcx>) -> EvalResult<'tcx, PrimVal> {
        use primval::PrimVal::*;
        match ty.sty {
            ty::TyRef(..) |
            ty::TyRawPtr(_) => Ok(Ptr(ptr)),
            ty::TyFnPtr(_) => Ok(FnPtr(ptr)),
            _ => Err(EvalError::Unimplemented(format!("ptr to {:?} cast", ty))),
        }
    }

    fn cast_signed_int(&self, val: i64, ty: ty::Ty<'tcx>) -> EvalResult<'tcx, PrimVal> {
        self.cast_const_int(val as u64, ty, val < 0)
    }

    fn cast_const_int(&self, v: u64, ty: ty::Ty<'tcx>, negative: bool) -> EvalResult<'tcx, PrimVal> {
        use primval::PrimVal::*;
        match ty.sty {
            ty::TyBool if v == 0 => Ok(Bool(false)),
            ty::TyBool if v == 1 => Ok(Bool(true)),
            ty::TyBool => Err(EvalError::InvalidBool),
            ty::TyInt(ast::IntTy::I8) => Ok(I8(v as i64 as i8)),
            ty::TyInt(ast::IntTy::I16) => Ok(I16(v as i64 as i16)),
            ty::TyInt(ast::IntTy::I32) => Ok(I32(v as i64 as i32)),
            ty::TyInt(ast::IntTy::I64) => Ok(I64(v as i64)),
            ty::TyInt(ast::IntTy::Is) => {
                let int_ty = self.tcx.sess.target.int_type;
                let ty = self.tcx.mk_mach_int(int_ty);
                self.cast_const_int(v, ty, negative)
            },
            ty::TyUint(ast::UintTy::U8) => Ok(U8(v as u8)),
            ty::TyUint(ast::UintTy::U16) => Ok(U16(v as u16)),
            ty::TyUint(ast::UintTy::U32) => Ok(U32(v as u32)),
            ty::TyUint(ast::UintTy::U64) => Ok(U64(v)),
            ty::TyUint(ast::UintTy::Us) => {
                let uint_ty = self.tcx.sess.target.uint_type;
                let ty = self.tcx.mk_mach_uint(uint_ty);
                self.cast_const_int(v, ty, negative)
            },
            ty::TyFloat(ast::FloatTy::F64) if negative => Ok(F64(v as i64 as f64)),
            ty::TyFloat(ast::FloatTy::F64) => Ok(F64(v as f64)),
            ty::TyFloat(ast::FloatTy::F32) if negative => Ok(F32(v as i64 as f32)),
            ty::TyFloat(ast::FloatTy::F32) => Ok(F32(v as f32)),
            ty::TyRawPtr(_) => Ok(Ptr(Pointer::from_int(v as usize))),
            ty::TyChar if v as u8 as u64 == v => Ok(Char(v as u8 as char)),
            ty::TyChar => Err(EvalError::InvalidChar(v)),
            _ => Err(EvalError::Unimplemented(format!("int to {:?} cast", ty))),
        }
    }

    fn cast_const_float(&self, val: f64, ty: Ty<'tcx>) -> EvalResult<'tcx, PrimVal> {
        use primval::PrimVal::*;
        match ty.sty {
            // casting negative floats to unsigned integers yields zero
            ty::TyUint(_) if val < 0.0 => self.cast_const_int(0, ty, false),
            ty::TyInt(_) if val < 0.0 => self.cast_const_int(val as i64 as u64, ty, true),
            ty::TyInt(_) | ty::TyUint(_) => self.cast_const_int(val as u64, ty, false),
            ty::TyFloat(ast::FloatTy::F64) => Ok(F64(val)),
            ty::TyFloat(ast::FloatTy::F32) => Ok(F32(val as f32)),
            _ => Err(EvalError::Unimplemented(format!("float to {:?} cast", ty))),
        }
    }
}
