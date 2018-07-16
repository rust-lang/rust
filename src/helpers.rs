use mir;
use rustc::ty::Ty;
use rustc::ty::layout::{LayoutOf, Size};

use super::{Scalar, ScalarExt, EvalResult, EvalContext, ValTy};
use rustc_mir::interpret::sign_extend;

pub trait EvalContextExt<'tcx> {
    fn wrapping_pointer_offset(
        &self,
        ptr: Scalar,
        pointee_ty: Ty<'tcx>,
        offset: i64,
    ) -> EvalResult<'tcx, Scalar>;

    fn pointer_offset(
        &self,
        ptr: Scalar,
        pointee_ty: Ty<'tcx>,
        offset: i64,
    ) -> EvalResult<'tcx, Scalar>;

    fn value_to_isize(
        &self,
        value: ValTy<'tcx>,
    ) -> EvalResult<'tcx, i64>;

    fn value_to_usize(
        &self,
        value: ValTy<'tcx>,
    ) -> EvalResult<'tcx, u64>;

    fn value_to_i32(
        &self,
        value: ValTy<'tcx>,
    ) -> EvalResult<'tcx, i32>;

    fn value_to_u8(
        &self,
        value: ValTy<'tcx>,
    ) -> EvalResult<'tcx, u8>;
}

impl<'a, 'mir, 'tcx> EvalContextExt<'tcx> for EvalContext<'a, 'mir, 'tcx, super::Evaluator<'tcx>> {
    fn wrapping_pointer_offset(
        &self,
        ptr: Scalar,
        pointee_ty: Ty<'tcx>,
        offset: i64,
    ) -> EvalResult<'tcx, Scalar> {
        // FIXME: assuming here that type size is < i64::max_value()
        let pointee_size = self.layout_of(pointee_ty)?.size.bytes() as i64;
        let offset = offset.overflowing_mul(pointee_size).0;
        ptr.ptr_wrapping_signed_offset(offset, self)
    }

    fn pointer_offset(
        &self,
        ptr: Scalar,
        pointee_ty: Ty<'tcx>,
        offset: i64,
    ) -> EvalResult<'tcx, Scalar> {
        // This function raises an error if the offset moves the pointer outside of its allocation.  We consider
        // ZSTs their own huge allocation that doesn't overlap with anything (and nothing moves in there because the size is 0).
        // We also consider the NULL pointer its own separate allocation, and all the remaining integers pointers their own
        // allocation.

        if ptr.is_null()? {
            // NULL pointers must only be offset by 0
            return if offset == 0 {
                Ok(ptr)
            } else {
                err!(InvalidNullPointerUsage)
            };
        }
        // FIXME: assuming here that type size is < i64::max_value()
        let pointee_size = self.layout_of(pointee_ty)?.size.bytes() as i64;
         if let Some(offset) = offset.checked_mul(pointee_size) {
            let ptr = ptr.ptr_signed_offset(offset, self)?;
            // Do not do bounds-checking for integers; they can never alias a normal pointer anyway.
            if let Scalar::Ptr(ptr) = ptr {
                self.memory.check_bounds(ptr, false)?;
            } else if ptr.is_null()? {
                // We moved *to* a NULL pointer.  That seems wrong, LLVM considers the NULL pointer its own small allocation.  Reject this, for now.
                return err!(InvalidNullPointerUsage);
            }
            Ok(ptr)
        } else {
            err!(Overflow(mir::BinOp::Mul))
        }
    }

    fn value_to_isize(
        &self,
        value: ValTy<'tcx>,
    ) -> EvalResult<'tcx, i64> {
        assert_eq!(value.ty, self.tcx.types.isize);
        let raw = self.value_to_scalar(value)?.to_bits(self.memory.pointer_size())?;
        let raw = sign_extend(self.tcx.tcx, raw, self.tcx.types.isize)?;
        Ok(raw as i64)
    }

    fn value_to_usize(
        &self,
        value: ValTy<'tcx>,
    ) -> EvalResult<'tcx, u64> {
        assert_eq!(value.ty, self.tcx.types.usize);
        self.value_to_scalar(value)?.to_bits(self.memory.pointer_size()).map(|v| v as u64)
    }

    fn value_to_i32(
        &self,
        value: ValTy<'tcx>,
    ) -> EvalResult<'tcx, i32> {
        assert_eq!(value.ty, self.tcx.types.i32);
        let raw = self.value_to_scalar(value)?.to_bits(Size::from_bits(32))?;
        let raw = sign_extend(self.tcx.tcx, raw, self.tcx.types.i32)?;
        Ok(raw as i32)
    }

    fn value_to_u8(
        &self,
        value: ValTy<'tcx>,
    ) -> EvalResult<'tcx, u8> {
        assert_eq!(value.ty, self.tcx.types.u8);
        self.value_to_scalar(value)?.to_bits(Size::from_bits(8)).map(|v| v as u8)
    }
}
