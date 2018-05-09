use rustc::ty;
use rustc::mir;

use super::*;

use helpers::EvalContextExt as HelperEvalContextExt;

pub trait EvalContextExt<'tcx> {
    fn ptr_op(
        &self,
        bin_op: mir::BinOp,
        left: PrimVal,
        left_ty: ty::Ty<'tcx>,
        right: PrimVal,
        right_ty: ty::Ty<'tcx>,
    ) -> EvalResult<'tcx, Option<(PrimVal, bool)>>;

    fn ptr_int_arithmetic(
        &self,
        bin_op: mir::BinOp,
        left: MemoryPointer,
        right: i128,
        signed: bool,
    ) -> EvalResult<'tcx, (PrimVal, bool)>;
}

impl<'a, 'mir, 'tcx> EvalContextExt<'tcx> for EvalContext<'a, 'mir, 'tcx, super::Evaluator<'tcx>> {
    fn ptr_op(
        &self,
        bin_op: mir::BinOp,
        left: PrimVal,
        left_ty: ty::Ty<'tcx>,
        right: PrimVal,
        right_ty: ty::Ty<'tcx>,
    ) -> EvalResult<'tcx, Option<(PrimVal, bool)>> {
        use rustc::mir::interpret::PrimValKind::*;
        use rustc::mir::BinOp::*;
        let usize = PrimValKind::from_uint_size(self.memory.pointer_size());
        let isize = PrimValKind::from_int_size(self.memory.pointer_size());
        let left_kind = self.ty_to_primval_kind(left_ty)?;
        let right_kind = self.ty_to_primval_kind(right_ty)?;
        match bin_op {
            Offset if left_kind == Ptr && right_kind == usize => {
                let pointee_ty = left_ty
                    .builtin_deref(true)
                    .expect("Offset called on non-ptr type")
                    .ty;
                let ptr = self.pointer_offset(
                    left.into(),
                    pointee_ty,
                    right.to_bytes()? as i64,
                )?;
                Ok(Some((ptr.into_inner_primval(), false)))
            }
            // These work on anything
            Eq if left_kind == right_kind => {
                let result = match (left, right) {
                    (PrimVal::Bytes(left), PrimVal::Bytes(right)) => left == right,
                    (PrimVal::Ptr(left), PrimVal::Ptr(right)) => left == right,
                    (PrimVal::Undef, _) |
                    (_, PrimVal::Undef) => return err!(ReadUndefBytes),
                    _ => false,
                };
                Ok(Some((PrimVal::from_bool(result), false)))
            }
            Ne if left_kind == right_kind => {
                let result = match (left, right) {
                    (PrimVal::Bytes(left), PrimVal::Bytes(right)) => left != right,
                    (PrimVal::Ptr(left), PrimVal::Ptr(right)) => left != right,
                    (PrimVal::Undef, _) |
                    (_, PrimVal::Undef) => return err!(ReadUndefBytes),
                    _ => true,
                };
                Ok(Some((PrimVal::from_bool(result), false)))
            }
            // These need both pointers to be in the same allocation
            Lt | Le | Gt | Ge | Sub
                if left_kind == right_kind &&
                       (left_kind == Ptr || left_kind == usize || left_kind == isize) &&
                       left.is_ptr() && right.is_ptr() => {
                let left = left.to_ptr()?;
                let right = right.to_ptr()?;
                if left.alloc_id == right.alloc_id {
                    let res = match bin_op {
                        Lt => left.offset < right.offset,
                        Le => left.offset <= right.offset,
                        Gt => left.offset > right.offset,
                        Ge => left.offset >= right.offset,
                        Sub => {
                            return self.binary_op(
                                Sub,
                                PrimVal::Bytes(left.offset as u128),
                                self.tcx.types.usize,
                                PrimVal::Bytes(right.offset as u128),
                                self.tcx.types.usize,
                            ).map(Some)
                        }
                        _ => bug!("We already established it has to be one of these operators."),
                    };
                    Ok(Some((PrimVal::from_bool(res), false)))
                } else {
                    // Both are pointers, but from different allocations.
                    err!(InvalidPointerMath)
                }
            }
            // These work if one operand is a pointer, the other an integer
            Add | BitAnd | Sub
                if left_kind == right_kind && (left_kind == usize || left_kind == isize) &&
                       left.is_ptr() && right.is_bytes() => {
                // Cast to i128 is fine as we checked the kind to be ptr-sized
                self.ptr_int_arithmetic(
                    bin_op,
                    left.to_ptr()?,
                    right.to_bytes()? as i128,
                    left_kind == isize,
                ).map(Some)
            }
            Add | BitAnd
                if left_kind == right_kind && (left_kind == usize || left_kind == isize) &&
                       left.is_bytes() && right.is_ptr() => {
                // This is a commutative operation, just swap the operands
                self.ptr_int_arithmetic(
                    bin_op,
                    right.to_ptr()?,
                    left.to_bytes()? as i128,
                    left_kind == isize,
                ).map(Some)
            }
            _ => Ok(None),
        }
    }

    fn ptr_int_arithmetic(
        &self,
        bin_op: mir::BinOp,
        left: MemoryPointer,
        right: i128,
        signed: bool,
    ) -> EvalResult<'tcx, (PrimVal, bool)> {
        use rustc::mir::BinOp::*;

        fn map_to_primval((res, over): (MemoryPointer, bool)) -> (PrimVal, bool) {
            (PrimVal::Ptr(res), over)
        }

        Ok(match bin_op {
            Sub =>
                // The only way this can overflow is by underflowing, so signdeness of the right operands does not matter
                map_to_primval(left.overflowing_signed_offset(-right, self)),
            Add if signed =>
                map_to_primval(left.overflowing_signed_offset(right, self)),
            Add if !signed =>
                map_to_primval(left.overflowing_offset(right as u64, self)),

            BitAnd if !signed => {
                let base_mask : u64 = !(self.memory.get(left.alloc_id)?.align.abi() - 1);
                let right = right as u64;
                if right & base_mask == base_mask {
                    // Case 1: The base address bits are all preserved, i.e., right is all-1 there
                    (PrimVal::Ptr(MemoryPointer::new(left.alloc_id, left.offset & right)), false)
                } else if right & base_mask == 0 {
                    // Case 2: The base address bits are all taken away, i.e., right is all-0 there
                    (PrimVal::from_u128((left.offset & right) as u128), false)
                } else {
                    return err!(ReadPointerAsBytes);
                }
            }

            _ => {
                let msg = format!("unimplemented binary op on pointer {:?}: {:?}, {:?} ({})", bin_op, left, right, if signed { "signed" } else { "unsigned" });
                return err!(Unimplemented(msg));
            }
        })
    }
}
