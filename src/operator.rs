use std::convert::TryFrom;

use rustc::mir;
use rustc::ty::{
    layout::{LayoutOf, Size},
    Ty,
};

use crate::*;

pub trait EvalContextExt<'tcx> {
    fn binary_ptr_op(
        &self,
        bin_op: mir::BinOp,
        left: ImmTy<'tcx, Tag>,
        right: ImmTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, (Scalar<Tag>, bool, Ty<'tcx>)>;

    fn ptr_eq(&self, left: Scalar<Tag>, right: Scalar<Tag>) -> InterpResult<'tcx, bool>;

    fn pointer_offset_inbounds(
        &self,
        ptr: Scalar<Tag>,
        pointee_ty: Ty<'tcx>,
        offset: i64,
    ) -> InterpResult<'tcx, Scalar<Tag>>;
}

impl<'mir, 'tcx> EvalContextExt<'tcx> for super::MiriEvalContext<'mir, 'tcx> {
    fn binary_ptr_op(
        &self,
        bin_op: mir::BinOp,
        left: ImmTy<'tcx, Tag>,
        right: ImmTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, (Scalar<Tag>, bool, Ty<'tcx>)> {
        use rustc::mir::BinOp::*;

        trace!("ptr_op: {:?} {:?} {:?}", *left, bin_op, *right);

        Ok(match bin_op {
            Eq | Ne => {
                // This supports fat pointers.
                #[rustfmt::skip]
                let eq = match (*left, *right) {
                    (Immediate::Scalar(left), Immediate::Scalar(right)) => {
                        self.ptr_eq(left.not_undef()?, right.not_undef()?)?
                    }
                    (Immediate::ScalarPair(left1, left2), Immediate::ScalarPair(right1, right2)) => {
                        self.ptr_eq(left1.not_undef()?, right1.not_undef()?)?
                            && self.ptr_eq(left2.not_undef()?, right2.not_undef()?)?
                    }
                    _ => bug!("Type system should not allow comparing Scalar with ScalarPair"),
                };
                (Scalar::from_bool(if bin_op == Eq { eq } else { !eq }), false, self.tcx.types.bool)
            }

            Lt | Le | Gt | Ge => {
                // Just compare the integers.
                // TODO: Do we really want to *always* do that, even when comparing two live in-bounds pointers?
                let left = self.force_bits(left.to_scalar()?, left.layout.size)?;
                let right = self.force_bits(right.to_scalar()?, right.layout.size)?;
                let res = match bin_op {
                    Lt => left < right,
                    Le => left <= right,
                    Gt => left > right,
                    Ge => left >= right,
                    _ => bug!("We already established it has to be one of these operators."),
                };
                (Scalar::from_bool(res), false, self.tcx.types.bool)
            }

            Offset => {
                let pointee_ty =
                    left.layout.ty.builtin_deref(true).expect("Offset called on non-ptr type").ty;
                let ptr = self.pointer_offset_inbounds(
                    left.to_scalar()?,
                    pointee_ty,
                    right.to_scalar()?.to_machine_isize(self)?,
                )?;
                (ptr, false, left.layout.ty)
            }

            _ => bug!("Invalid operator on pointers: {:?}", bin_op),
        })
    }

    fn ptr_eq(&self, left: Scalar<Tag>, right: Scalar<Tag>) -> InterpResult<'tcx, bool> {
        let size = self.pointer_size();
        // Just compare the integers.
        // TODO: Do we really want to *always* do that, even when comparing two live in-bounds pointers?
        let left = self.force_bits(left, size)?;
        let right = self.force_bits(right, size)?;
        Ok(left == right)
    }

    /// Raises an error if the offset moves the pointer outside of its allocation.
    /// For integers, we consider each of them their own tiny allocation of size 0,
    /// so offset-by-0 is okay for them -- except for NULL, which we rule out entirely.
    fn pointer_offset_inbounds(
        &self,
        ptr: Scalar<Tag>,
        pointee_ty: Ty<'tcx>,
        offset: i64,
    ) -> InterpResult<'tcx, Scalar<Tag>> {
        let pointee_size = i64::try_from(self.layout_of(pointee_ty)?.size.bytes()).unwrap();
        let offset = offset
            .checked_mul(pointee_size)
            .ok_or_else(|| err_panic!(Overflow(mir::BinOp::Mul)))?;
        // We do this first, to rule out overflows.
        let offset_ptr = ptr.ptr_signed_offset(offset, self)?;
        // What we need to check is that starting at `min(ptr, offset_ptr)`,
        // we could do an access of size `abs(offset)`. Alignment does not matter.
        let (min_ptr, abs_offset) = if offset >= 0 {
            (ptr, u64::try_from(offset).unwrap())
        } else {
            // Negative offset.
            // If the negation overflows, the result will be negative so the try_from will fail.
            (offset_ptr, u64::try_from(-offset).unwrap())
        };
        self.memory.check_ptr_access_align(
            min_ptr,
            Size::from_bytes(abs_offset),
            None,
            CheckInAllocMsg::InboundsTest,
        )?;
        // That's it!
        Ok(offset_ptr)
    }
}
