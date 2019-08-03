use rustc::ty::{Ty, layout::LayoutOf};
use rustc::mir;

use crate::*;

pub trait EvalContextExt<'tcx> {
    fn pointer_inbounds(
        &self,
        ptr: Pointer<Tag>
    ) -> InterpResult<'tcx>;

    fn binary_ptr_op(
        &self,
        bin_op: mir::BinOp,
        left: ImmTy<'tcx, Tag>,
        right: ImmTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, (Scalar<Tag>, bool)>;

    fn ptr_eq(
        &self,
        left: Scalar<Tag>,
        right: Scalar<Tag>,
    ) -> InterpResult<'tcx, bool>;

    fn pointer_offset_inbounds(
        &self,
        ptr: Scalar<Tag>,
        pointee_ty: Ty<'tcx>,
        offset: i64,
    ) -> InterpResult<'tcx, Scalar<Tag>>;
}

impl<'mir, 'tcx> EvalContextExt<'tcx> for super::MiriEvalContext<'mir, 'tcx> {
    /// Test if the pointer is in-bounds of a live allocation.
    #[inline]
    fn pointer_inbounds(&self, ptr: Pointer<Tag>) -> InterpResult<'tcx> {
        let (size, _align) = self.memory().get_size_and_align(ptr.alloc_id, AllocCheck::Live)?;
        ptr.check_in_alloc(size, CheckInAllocMsg::InboundsTest)
    }

    fn binary_ptr_op(
        &self,
        bin_op: mir::BinOp,
        left: ImmTy<'tcx, Tag>,
        right: ImmTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, (Scalar<Tag>, bool)> {
        use rustc::mir::BinOp::*;

        trace!("ptr_op: {:?} {:?} {:?}", *left, bin_op, *right);

        Ok(match bin_op {
            Eq | Ne => {
                // This supports fat pointers.
                let eq = match (*left, *right) {
                    (Immediate::Scalar(left), Immediate::Scalar(right)) =>
                        self.ptr_eq(left.not_undef()?, right.not_undef()?)?,
                    (Immediate::ScalarPair(left1, left2), Immediate::ScalarPair(right1, right2)) =>
                        self.ptr_eq(left1.not_undef()?, right1.not_undef()?)? &&
                        self.ptr_eq(left2.not_undef()?, right2.not_undef()?)?,
                    _ => bug!("Type system should not allow comparing Scalar with ScalarPair"),
                };
                (Scalar::from_bool(if bin_op == Eq { eq } else { !eq }), false)
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
                (Scalar::from_bool(res), false)
            }

            Offset => {
                let pointee_ty = left.layout.ty
                    .builtin_deref(true)
                    .expect("Offset called on non-ptr type")
                    .ty;
                let ptr = self.pointer_offset_inbounds(
                    left.to_scalar()?,
                    pointee_ty,
                    right.to_scalar()?.to_isize(self)?,
                )?;
                (ptr, false)
            }

            _ => bug!("Invalid operator on pointers: {:?}", bin_op)
        })
    }

    fn ptr_eq(
        &self,
        left: Scalar<Tag>,
        right: Scalar<Tag>,
    ) -> InterpResult<'tcx, bool> {
        let size = self.pointer_size();
        // Just compare the integers.
        // TODO: Do we really want to *always* do that, even when comparing two live in-bounds pointers?
        let left = self.force_bits(left, size)?;
        let right = self.force_bits(right, size)?;
        Ok(left == right)
    }

    /// Raises an error if the offset moves the pointer outside of its allocation.
    /// We consider ZSTs their own huge allocation that doesn't overlap with anything (and nothing
    /// moves in there because the size is 0). We also consider the NULL pointer its own separate
    /// allocation, and all the remaining integers pointers their own allocation.
    fn pointer_offset_inbounds(
        &self,
        ptr: Scalar<Tag>,
        pointee_ty: Ty<'tcx>,
        offset: i64,
    ) -> InterpResult<'tcx, Scalar<Tag>> {
        // FIXME: assuming here that type size is less than `i64::max_value()`.
        let pointee_size = self.layout_of(pointee_ty)?.size.bytes() as i64;
        let offset = offset
            .checked_mul(pointee_size)
            .ok_or_else(|| err_panic!(Overflow(mir::BinOp::Mul)))?;
        // Now let's see what kind of pointer this is.
        let ptr = if offset == 0 {
            match ptr {
                Scalar::Ptr(ptr) => ptr,
                Scalar::Raw { .. } => {
                    // Offset 0 on an integer. We accept that, pretending there is
                    // a little zero-sized allocation here.
                    return Ok(ptr);
                }
            }
        } else {
            // Offset > 0. We *require* a pointer.
            self.force_ptr(ptr)?
        };
        // Both old and new pointer must be in-bounds of a *live* allocation.
        // (Of the same allocation, but that part is trivial with our representation.)
        self.pointer_inbounds(ptr)?;
        let ptr = ptr.signed_offset(offset, self)?;
        self.pointer_inbounds(ptr)?;
        Ok(Scalar::Ptr(ptr))
    }
}
