use log::trace;

use rustc_middle::{mir, ty::Ty};
use rustc_target::abi::Size;

use crate::*;

pub trait EvalContextExt<'tcx> {
    fn binary_ptr_op(
        &self,
        bin_op: mir::BinOp,
        left: &ImmTy<'tcx, Provenance>,
        right: &ImmTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, (Scalar<Provenance>, bool, Ty<'tcx>)>;
}

impl<'mir, 'tcx> EvalContextExt<'tcx> for super::MiriInterpCx<'mir, 'tcx> {
    fn binary_ptr_op(
        &self,
        bin_op: mir::BinOp,
        left: &ImmTy<'tcx, Provenance>,
        right: &ImmTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, (Scalar<Provenance>, bool, Ty<'tcx>)> {
        use rustc_middle::mir::BinOp::*;

        trace!("ptr_op: {:?} {:?} {:?}", *left, bin_op, *right);

        Ok(match bin_op {
            Eq | Ne | Lt | Le | Gt | Ge => {
                assert_eq!(left.layout.abi, right.layout.abi); // types an differ, e.g. fn ptrs with different `for`
                let size = self.pointer_size();
                // Just compare the bits. ScalarPairs are compared lexicographically.
                // We thus always compare pairs and simply fill scalars up with 0.
                let left = match **left {
                    Immediate::Scalar(l) => (l.to_bits(size)?, 0),
                    Immediate::ScalarPair(l1, l2) => (l1.to_bits(size)?, l2.to_bits(size)?),
                    Immediate::Uninit => panic!("we should never see uninit data here"),
                };
                let right = match **right {
                    Immediate::Scalar(r) => (r.to_bits(size)?, 0),
                    Immediate::ScalarPair(r1, r2) => (r1.to_bits(size)?, r2.to_bits(size)?),
                    Immediate::Uninit => panic!("we should never see uninit data here"),
                };
                let res = match bin_op {
                    Eq => left == right,
                    Ne => left != right,
                    Lt => left < right,
                    Le => left <= right,
                    Gt => left > right,
                    Ge => left >= right,
                    _ => bug!(),
                };
                (Scalar::from_bool(res), false, self.tcx.types.bool)
            }

            Offset => {
                assert!(left.layout.ty.is_unsafe_ptr());
                let ptr = left.to_scalar().to_pointer(self)?;
                let offset = right.to_scalar().to_target_isize(self)?;

                let pointee_ty =
                    left.layout.ty.builtin_deref(true).expect("Offset called on non-ptr type").ty;
                let ptr = self.ptr_offset_inbounds(ptr, pointee_ty, offset)?;
                (Scalar::from_maybe_pointer(ptr, self), false, left.layout.ty)
            }

            // Some more operations are possible with atomics.
            // The return value always has the provenance of the *left* operand.
            Add | Sub | BitOr | BitAnd | BitXor => {
                assert!(left.layout.ty.is_unsafe_ptr());
                assert!(right.layout.ty.is_unsafe_ptr());
                let ptr = left.to_scalar().to_pointer(self)?;
                // We do the actual operation with usize-typed scalars.
                let left = ImmTy::from_uint(ptr.addr().bytes(), self.machine.layouts.usize);
                let right = ImmTy::from_uint(
                    right.to_scalar().to_target_usize(self)?,
                    self.machine.layouts.usize,
                );
                let (result, overflowing, _ty) =
                    self.overflowing_binary_op(bin_op, &left, &right)?;
                // Construct a new pointer with the provenance of `ptr` (the LHS).
                let result_ptr =
                    Pointer::new(ptr.provenance, Size::from_bytes(result.to_target_usize(self)?));
                (Scalar::from_maybe_pointer(result_ptr, self), overflowing, left.layout.ty)
            }

            _ => span_bug!(self.cur_span(), "Invalid operator on pointers: {:?}", bin_op),
        })
    }
}
