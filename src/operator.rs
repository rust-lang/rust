use rustc::ty;
use rustc::ty::layout::Primitive;
use rustc::mir;

use super::*;

use helpers::EvalContextExt as HelperEvalContextExt;

pub trait EvalContextExt<'tcx> {
    fn ptr_op(
        &self,
        bin_op: mir::BinOp,
        left: Scalar,
        left_ty: ty::Ty<'tcx>,
        right: Scalar,
        right_ty: ty::Ty<'tcx>,
    ) -> EvalResult<'tcx, Option<(Scalar, bool)>>;

    fn ptr_int_arithmetic(
        &self,
        bin_op: mir::BinOp,
        left: Pointer,
        right: i128,
        signed: bool,
    ) -> EvalResult<'tcx, (Scalar, bool)>;
}

impl<'a, 'mir, 'tcx> EvalContextExt<'tcx> for EvalContext<'a, 'mir, 'tcx, super::Evaluator<'tcx>> {
    fn ptr_op(
        &self,
        bin_op: mir::BinOp,
        left: Scalar,
        left_ty: ty::Ty<'tcx>,
        right: Scalar,
        right_ty: ty::Ty<'tcx>,
    ) -> EvalResult<'tcx, Option<(Scalar, bool)>> {
        use rustc::mir::BinOp::*;
        use rustc::ty::layout::Integer::*;
        let usize = Primitive::Int(match self.memory.pointer_size().bytes() {
            1 => I8,
            2 => I16,
            4 => I32,
            8 => I64,
            16 => I128,
            _ => unreachable!(),
        }, false);
        let isize = Primitive::Int(match self.memory.pointer_size().bytes() {
            1 => I8,
            2 => I16,
            4 => I32,
            8 => I64,
            16 => I128,
            _ => unreachable!(),
        }, true);
        let left_kind = match self.layout_of(left_ty)?.abi {
            ty::layout::Abi::Scalar(ref scalar) => scalar.value,
            _ => Err(EvalErrorKind::TypeNotPrimitive(left_ty))?,
        };
        let right_kind = match self.layout_of(right_ty)?.abi {
            ty::layout::Abi::Scalar(ref scalar) => scalar.value,
            _ => Err(EvalErrorKind::TypeNotPrimitive(right_ty))?,
        };
        match bin_op {
            Offset if left_kind == Primitive::Pointer && right_kind == usize => {
                let pointee_ty = left_ty
                    .builtin_deref(true)
                    .expect("Offset called on non-ptr type")
                    .ty;
                let ptr = self.pointer_offset(
                    left.into(),
                    pointee_ty,
                    right.to_bits(self.memory.pointer_size())? as i64,
                )?;
                Ok(Some((ptr, false)))
            }
            // These work on anything
            Eq if left_kind == right_kind => {
                let result = match (left, right) {
                    (Scalar::Bits { .. }, Scalar::Bits { .. }) => {
                        left.to_bits(self.memory.pointer_size())? == right.to_bits(self.memory.pointer_size())?
                    },
                    (Scalar::Ptr(left), Scalar::Ptr(right)) => left == right,
                    _ => false,
                };
                Ok(Some((Scalar::from_bool(result), false)))
            }
            Ne if left_kind == right_kind => {
                let result = match (left, right) {
                    (Scalar::Bits { .. }, Scalar::Bits { .. }) => {
                        left.to_bits(self.memory.pointer_size())? != right.to_bits(self.memory.pointer_size())?
                    },
                    (Scalar::Ptr(left), Scalar::Ptr(right)) => left != right,
                    _ => true,
                };
                Ok(Some((Scalar::from_bool(result), false)))
            }
            // These need both pointers to be in the same allocation
            Lt | Le | Gt | Ge | Sub
                if left_kind == right_kind &&
                       (left_kind == Primitive::Pointer || left_kind == usize || left_kind == isize) &&
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
                                Scalar::Bits { bits: left.offset.bytes() as u128, defined: self.memory.pointer_size().bits() as u8 },
                                self.tcx.types.usize,
                                Scalar::Bits { bits: right.offset.bytes() as u128, defined: self.memory.pointer_size().bits() as u8 },
                                self.tcx.types.usize,
                            ).map(Some)
                        }
                        _ => bug!("We already established it has to be one of these operators."),
                    };
                    Ok(Some((Scalar::from_bool(res), false)))
                } else {
                    // Both are pointers, but from different allocations.
                    err!(InvalidPointerMath)
                }
            }
            // These work if one operand is a pointer, the other an integer
            Add | BitAnd | Sub
                if left_kind == right_kind && (left_kind == usize || left_kind == isize) &&
                       left.is_ptr() && right.is_bits() => {
                // Cast to i128 is fine as we checked the kind to be ptr-sized
                self.ptr_int_arithmetic(
                    bin_op,
                    left.to_ptr()?,
                    right.to_bits(self.memory.pointer_size())? as i128,
                    left_kind == isize,
                ).map(Some)
            }
            Add | BitAnd
                if left_kind == right_kind && (left_kind == usize || left_kind == isize) &&
                       left.is_bits() && right.is_ptr() => {
                // This is a commutative operation, just swap the operands
                self.ptr_int_arithmetic(
                    bin_op,
                    right.to_ptr()?,
                    left.to_bits(self.memory.pointer_size())? as i128,
                    left_kind == isize,
                ).map(Some)
            }
            _ => Ok(None),
        }
    }

    fn ptr_int_arithmetic(
        &self,
        bin_op: mir::BinOp,
        left: Pointer,
        right: i128,
        signed: bool,
    ) -> EvalResult<'tcx, (Scalar, bool)> {
        use rustc::mir::BinOp::*;

        fn map_to_primval((res, over): (Pointer, bool)) -> (Scalar, bool) {
            (Scalar::Ptr(res), over)
        }

        Ok(match bin_op {
            Sub =>
                // The only way this can overflow is by underflowing, so signdeness of the right operands does not matter
                map_to_primval(left.overflowing_signed_offset(-right, self)),
            Add if signed =>
                map_to_primval(left.overflowing_signed_offset(right, self)),
            Add if !signed =>
                map_to_primval(left.overflowing_offset(Size::from_bytes(right as u64), self)),

            BitAnd if !signed => {
                let base_mask : u64 = !(self.memory.get(left.alloc_id)?.align.abi() - 1);
                let right = right as u64;
                if right & base_mask == base_mask {
                    // Case 1: The base address bits are all preserved, i.e., right is all-1 there
                    (Scalar::Ptr(Pointer::new(left.alloc_id, Size::from_bytes(left.offset.bytes() & right))), false)
                } else if right & base_mask == 0 {
                    // Case 2: The base address bits are all taken away, i.e., right is all-0 there
                    (Scalar::Bits { bits: (left.offset.bytes() & right) as u128, defined: 128 }, false)
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
