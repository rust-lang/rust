use super::{Pointer, EvalResult, PrimVal, EvalContext};
use rustc::ty::Ty;
use rustc::ty::layout::LayoutOf;

pub trait EvalContextExt<'tcx> {
    fn wrapping_pointer_offset(
        &self,
        ptr: Pointer,
        pointee_ty: Ty<'tcx>,
        offset: i64,
    ) -> EvalResult<'tcx, Pointer>;

    fn pointer_offset(
        &self,
        ptr: Pointer,
        pointee_ty: Ty<'tcx>,
        offset: i64,
    ) -> EvalResult<'tcx, Pointer>;
}

impl<'a, 'mir, 'tcx> EvalContextExt<'tcx> for EvalContext<'a, 'mir, 'tcx, super::Evaluator<'tcx>> {
    fn wrapping_pointer_offset(
        &self,
        ptr: Pointer,
        pointee_ty: Ty<'tcx>,
        offset: i64,
    ) -> EvalResult<'tcx, Pointer> {
        // FIXME: assuming here that type size is < i64::max_value()
        let pointee_size = self.layout_of(pointee_ty)?.size.bytes() as i64;
        let offset = offset.overflowing_mul(pointee_size).0;
        ptr.wrapping_signed_offset(offset, self)
    }

    fn pointer_offset(
        &self,
        ptr: Pointer,
        pointee_ty: Ty<'tcx>,
        offset: i64,
    ) -> EvalResult<'tcx, Pointer> {
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
        return if let Some(offset) = offset.checked_mul(pointee_size) {
            let ptr = ptr.signed_offset(offset, self)?;
            // Do not do bounds-checking for integers; they can never alias a normal pointer anyway.
            if let PrimVal::Ptr(ptr) = ptr.into_inner_primval() {
                self.memory.check_bounds(ptr, false)?;
            } else if ptr.is_null()? {
                // We moved *to* a NULL pointer.  That seems wrong, LLVM considers the NULL pointer its own small allocation.  Reject this, for now.
                return err!(InvalidNullPointerUsage);
            }
            Ok(ptr)
        } else {
            err!(OverflowingMath)
        };
    }
}
