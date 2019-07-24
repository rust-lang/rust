use rustc::ty::{Ty, layout::{Size, LayoutOf}};
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

    fn ptr_int_arithmetic(
        &self,
        bin_op: mir::BinOp,
        left: Pointer<Tag>,
        right: u128,
        signed: bool,
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

        // Operations that support fat pointers
        match bin_op {
            Eq | Ne => {
                let eq = match (*left, *right) {
                    (Immediate::Scalar(left), Immediate::Scalar(right)) =>
                        self.ptr_eq(left.not_undef()?, right.not_undef()?)?,
                    (Immediate::ScalarPair(left1, left2), Immediate::ScalarPair(right1, right2)) =>
                        self.ptr_eq(left1.not_undef()?, right1.not_undef()?)? &&
                        self.ptr_eq(left2.not_undef()?, right2.not_undef()?)?,
                    _ => bug!("Type system should not allow comparing Scalar with ScalarPair"),
                };
                return Ok((Scalar::from_bool(if bin_op == Eq { eq } else { !eq }), false));
            }
            _ => {},
        }

        // Now we expect no more fat pointers.
        let left_layout = left.layout;
        let left = left.to_scalar()?;
        let right_layout = right.layout;
        let right = right.to_scalar()?;

        Ok(match bin_op {
            Offset => {
                let pointee_ty = left_layout.ty
                    .builtin_deref(true)
                    .expect("Offset called on non-ptr type")
                    .ty;
                let ptr = self.pointer_offset_inbounds(
                    left,
                    pointee_ty,
                    right.to_isize(self)?,
                )?;
                (ptr, false)
            }
            // These need both to be pointer, and fail if they are not in the same location
            Lt | Le | Gt | Ge | Sub if left.is_ptr() && right.is_ptr() => {
                let left = left.assert_ptr();
                let right = right.assert_ptr();
                if left.alloc_id == right.alloc_id {
                    let res = match bin_op {
                        Lt => left.offset < right.offset,
                        Le => left.offset <= right.offset,
                        Gt => left.offset > right.offset,
                        Ge => left.offset >= right.offset,
                        Sub => {
                            // subtract the offsets
                            let left_offset = Scalar::from_uint(left.offset.bytes(), self.memory().pointer_size());
                            let right_offset = Scalar::from_uint(right.offset.bytes(), self.memory().pointer_size());
                            let layout = self.layout_of(self.tcx.types.usize)?;
                            return self.binary_op(
                                Sub,
                                ImmTy::from_scalar(left_offset, layout),
                                ImmTy::from_scalar(right_offset, layout),
                            )
                        }
                        _ => bug!("We already established it has to be one of these operators."),
                    };
                    (Scalar::from_bool(res), false)
                } else {
                    // Both are pointers, but from different allocations.
                    throw_unsup!(InvalidPointerMath)
                }
            }
            Lt | Le | Gt | Ge if left.is_bits() && right.is_bits() => {
                let left = left.assert_bits(self.memory().pointer_size());
                let right = right.assert_bits(self.memory().pointer_size());
                let res = match bin_op {
                    Lt => left < right,
                    Le => left <= right,
                    Gt => left > right,
                    Ge => left >= right,
                    _ => bug!("We already established it has to be one of these operators."),
                };
                Ok((Scalar::from_bool(res), false))
            }
            Gt | Ge if left.is_ptr() && right.is_bits() => {
                // "ptr >[=] integer" can be tested if the integer is small enough.
                let left = left.assert_ptr();
                let right = right.assert_bits(self.memory().pointer_size());
                let (_alloc_size, alloc_align) = self.memory()
                    .get_size_and_align(left.alloc_id, AllocCheck::MaybeDead)
                    .expect("alloc info with MaybeDead cannot fail");
                let min_ptr_val = u128::from(alloc_align.bytes()) + u128::from(left.offset.bytes());
                let result = match bin_op {
                    Gt => min_ptr_val > right,
                    Ge => min_ptr_val >= right,
                    _ => bug!(),
                };
                if result {
                    // Definitely true!
                    (Scalar::from_bool(true), false)
                } else {
                    // Sorry, can't tell.
                    throw_unsup!(InvalidPointerMath)
                }
            }
            // These work if the left operand is a pointer, and the right an integer
            Add | BitAnd | Sub | Rem if left.is_ptr() && right.is_bits() => {
                // Cast to i128 is fine as we checked the kind to be ptr-sized
                self.ptr_int_arithmetic(
                    bin_op,
                    left.assert_ptr(),
                    right.assert_bits(self.memory().pointer_size()),
                    right_layout.abi.is_signed(),
                )?
            }
            // Commutative operators also work if the integer is on the left
            Add | BitAnd if left.is_bits() && right.is_ptr() => {
                // This is a commutative operation, just swap the operands
                self.ptr_int_arithmetic(
                    bin_op,
                    right.assert_ptr(),
                    left.assert_bits(self.memory().pointer_size()),
                    left_layout.abi.is_signed(),
                )?
            }
            // Nothing else works
            _ => throw_unsup!(InvalidPointerMath),
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

    fn ptr_int_arithmetic(
        &self,
        bin_op: mir::BinOp,
        left: Pointer<Tag>,
        right: u128,
        signed: bool,
    ) -> InterpResult<'tcx, (Scalar<Tag>, bool)> {
        use rustc::mir::BinOp::*;

        fn map_to_primval((res, over): (Pointer<Tag>, bool)) -> (Scalar<Tag>, bool) {
            (Scalar::Ptr(res), over)
        }

        Ok(match bin_op {
            Sub =>
                // The only way this can overflow is by underflowing, so signdeness of the right
                // operands does not matter.
                map_to_primval(left.overflowing_signed_offset(-(right as i128), self)),
            Add if signed =>
                map_to_primval(left.overflowing_signed_offset(right as i128, self)),
            Add if !signed =>
                map_to_primval(left.overflowing_offset(Size::from_bytes(right as u64), self)),

            BitAnd if !signed => {
                let ptr_base_align = self.memory().get_size_and_align(left.alloc_id, AllocCheck::MaybeDead)
                    .expect("alloc info with MaybeDead cannot fail")
                    .1.bytes();
                let base_mask = {
                    // FIXME: use `interpret::truncate`, once that takes a `Size` instead of a `Layout`.
                    let shift = 128 - self.memory().pointer_size().bits();
                    let value = !(ptr_base_align as u128 - 1);
                    // Truncate (shift left to drop out leftover values, shift right to fill with zeroes).
                    (value << shift) >> shift
                };
                let ptr_size = self.memory().pointer_size();
                trace!("ptr BitAnd, align {}, operand {:#010x}, base_mask {:#010x}",
                    ptr_base_align, right, base_mask);
                if right & base_mask == base_mask {
                    // Case 1: the base address bits are all preserved, i.e., right is all-1 there.
                    let offset = (left.offset.bytes() as u128 & right) as u64;
                    (
                        Scalar::Ptr(Pointer::new_with_tag(
                            left.alloc_id,
                            Size::from_bytes(offset),
                            left.tag,
                        )),
                        false,
                    )
                } else if right & base_mask == 0 {
                    // Case 2: the base address bits are all taken away, i.e., right is all-0 there.
                    let v = Scalar::from_uint((left.offset.bytes() as u128) & right, ptr_size);
                    (v, false)
                } else {
                    throw_unsup!(ReadPointerAsBytes);
                }
            }

            Rem if !signed => {
                // Doing modulo a divisor of the alignment is allowed.
                // (Intuition: modulo a divisor leaks less information.)
                let ptr_base_align = self.memory().get_size_and_align(left.alloc_id, AllocCheck::MaybeDead)
                    .expect("alloc info with MaybeDead cannot fail")
                    .1.bytes();
                let right = right as u64;
                let ptr_size = self.memory().pointer_size();
                if right == 1 {
                    // Modulo 1 is always 0.
                    (Scalar::from_uint(0u32, ptr_size), false)
                } else if ptr_base_align % right == 0 {
                    // The base address would be cancelled out by the modulo operation, so we can
                    // just take the modulo of the offset.
                    (
                        Scalar::from_uint((left.offset.bytes() % right) as u128, ptr_size),
                        false,
                    )
                } else {
                    throw_unsup!(ReadPointerAsBytes);
                }
            }

            _ => {
                let msg = format!(
                    "unimplemented binary op on pointer {:?}: {:?}, {:?} ({})",
                    bin_op,
                    left,
                    right,
                    if signed { "signed" } else { "unsigned" }
                );
                throw_unsup!(Unimplemented(msg));
            }
        })
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
