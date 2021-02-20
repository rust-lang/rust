use log::trace;

use rustc_middle::{mir, ty::Ty};

use crate::*;

pub trait EvalContextExt<'tcx> {
    fn binary_ptr_op(
        &self,
        bin_op: mir::BinOp,
        left: &ImmTy<'tcx, Tag>,
        right: &ImmTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, (Scalar<Tag>, bool, Ty<'tcx>)>;

    fn ptr_eq(&self, left: Scalar<Tag>, right: Scalar<Tag>) -> InterpResult<'tcx, bool>;
}

impl<'mir, 'tcx> EvalContextExt<'tcx> for super::MiriEvalContext<'mir, 'tcx> {
    fn binary_ptr_op(
        &self,
        bin_op: mir::BinOp,
        left: &ImmTy<'tcx, Tag>,
        right: &ImmTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, (Scalar<Tag>, bool, Ty<'tcx>)> {
        use rustc_middle::mir::BinOp::*;

        trace!("ptr_op: {:?} {:?} {:?}", *left, bin_op, *right);

        Ok(match bin_op {
            Eq | Ne => {
                // This supports fat pointers.
                #[rustfmt::skip]
                let eq = match (**left, **right) {
                    (Immediate::Scalar(left), Immediate::Scalar(right)) => {
                        self.ptr_eq(left.check_init()?, right.check_init()?)?
                    }
                    (Immediate::ScalarPair(left1, left2), Immediate::ScalarPair(right1, right2)) => {
                        self.ptr_eq(left1.check_init()?, right1.check_init()?)?
                            && self.ptr_eq(left2.check_init()?, right2.check_init()?)?
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
                let ptr = self.ptr_offset_inbounds(
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
}
