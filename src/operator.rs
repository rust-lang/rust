use rustc::ty::Ty;
use rustc::mir;

use crate::*;

pub trait EvalContextExt<'tcx> {
    fn ptr_op(
        &self,
        bin_op: mir::BinOp,
        left: ImmTy<'tcx, Tag>,
        right: ImmTy<'tcx, Tag>,
    ) -> EvalResult<'tcx, (Scalar<Tag>, bool)>;

    fn ptr_int_arithmetic(
        &self,
        bin_op: mir::BinOp,
        left: Pointer<Tag>,
        right: u128,
        signed: bool,
    ) -> EvalResult<'tcx, (Scalar<Tag>, bool)>;

    fn ptr_eq(
        &self,
        left: Scalar<Tag>,
        right: Scalar<Tag>,
    ) -> EvalResult<'tcx, bool>;

    fn pointer_offset_inbounds(
        &self,
        ptr: Scalar<Tag>,
        pointee_ty: Ty<'tcx>,
        offset: i64,
    ) -> EvalResult<'tcx, Scalar<Tag>>;
}

impl<'a, 'mir, 'tcx> EvalContextExt<'tcx> for super::MiriEvalContext<'a, 'mir, 'tcx> {
    fn ptr_op(
        &self,
        bin_op: mir::BinOp,
        left: ImmTy<'tcx, Tag>,
        right: ImmTy<'tcx, Tag>,
    ) -> EvalResult<'tcx, (Scalar<Tag>, bool)> {
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
        debug_assert!(left.is_ptr() || right.is_ptr() || bin_op == Offset);

        match bin_op {
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
                Ok((ptr, false))
            }
            // These need both to be pointer, and fail if they are not in the same location
            Lt | Le | Gt | Ge | Sub if left.is_ptr() && right.is_ptr() => {
                let left = left.to_ptr().expect("we checked is_ptr");
                let right = right.to_ptr().expect("we checked is_ptr");
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
                    Ok((Scalar::from_bool(res), false))
                } else {
                    // Both are pointers, but from different allocations.
                    err!(InvalidPointerMath)
                }
            }
            // These work if the left operand is a pointer, and the right an integer
            Add | BitAnd | Sub | Rem if left.is_ptr() && right.is_bits() => {
                // Cast to i128 is fine as we checked the kind to be ptr-sized
                self.ptr_int_arithmetic(
                    bin_op,
                    left.to_ptr().expect("we checked is_ptr"),
                    right.to_bits(self.memory().pointer_size()).expect("we checked is_bits"),
                    right_layout.abi.is_signed(),
                )
            }
            // Commutative operators also work if the integer is on the left
            Add | BitAnd if left.is_bits() && right.is_ptr() => {
                // This is a commutative operation, just swap the operands
                self.ptr_int_arithmetic(
                    bin_op,
                    right.to_ptr().expect("we checked is_ptr"),
                    left.to_bits(self.memory().pointer_size()).expect("we checked is_bits"),
                    left_layout.abi.is_signed(),
                )
            }
            // Nothing else works
            _ => err!(InvalidPointerMath),
        }
    }

    fn ptr_eq(
        &self,
        left: Scalar<Tag>,
        right: Scalar<Tag>,
    ) -> EvalResult<'tcx, bool> {
        let size = self.pointer_size();
        Ok(match (left, right) {
            (Scalar::Bits { .. }, Scalar::Bits { .. }) =>
                left.to_bits(size)? == right.to_bits(size)?,
            (Scalar::Ptr(left), Scalar::Ptr(right)) => {
                // Comparison illegal if one of them is out-of-bounds, *unless* they
                // are in the same allocation.
                if left.alloc_id == right.alloc_id {
                    left.offset == right.offset
                } else {
                    // This accepts one-past-the end. Thus, there is still technically
                    // some non-determinism that we do not fully rule out when two
                    // allocations sit right next to each other. The C/C++ standards are
                    // somewhat fuzzy about this case, so pragmatically speaking I think
                    // for now this check is "good enough".
                    // FIXME: Once we support intptrcast, we could try to fix these holes.
                    // Dead allocations in miri cannot overlap with live allocations, but
                    // on read hardware this can easily happen. Thus for comparisons we require
                    // both pointers to be live.
                    self.memory().check_bounds_ptr(left, InboundsCheck::Live)?;
                    self.memory().check_bounds_ptr(right, InboundsCheck::Live)?;
                    // Two in-bounds pointers, we can compare across allocations.
                    left == right
                }
            }
            // Comparing ptr and integer.
            (Scalar::Ptr(ptr), Scalar::Bits { bits, size }) |
            (Scalar::Bits { bits, size }, Scalar::Ptr(ptr)) => {
                assert_eq!(size as u64, self.pointer_size().bytes());
                let bits = bits as u64;

                // Case I: Comparing real pointers with "small" integers.
                // Really we should only do this for NULL, but pragmatically speaking on non-bare-metal systems,
                // an allocation will never be at the very bottom of the address space.
                // Such comparisons can arise when comparing empty slices, which sometimes are "fake"
                // integer pointers (okay because the slice is empty) and sometimes point into a
                // real allocation.
                // The most common source of such integer pointers is `NonNull::dangling()`, which
                // equals the type's alignment. i128 might have an alignment of 16 bytes, but few types have
                // alignment 32 or higher, hence the limit of 32.
                // FIXME: Once we support intptrcast, we could try to fix these holes.
                if bits < 32 {
                    // Test if the ptr is in-bounds. Then it cannot be NULL.
                    // Even dangling pointers cannot be NULL.
                    if self.memory().check_bounds_ptr(ptr, InboundsCheck::MaybeDead).is_ok() {
                        return Ok(false);
                    }
                }

                let (alloc_size, alloc_align) = self.memory()
                    .get_size_and_align(ptr.alloc_id, InboundsCheck::MaybeDead)
                    .expect("determining size+align of dead ptr cannot fail");

                // Case II: Alignment gives it away
                if ptr.offset.bytes() % alloc_align.bytes() == 0 {
                    // The offset maintains the allocation alignment, so we know `base+offset`
                    // is aligned by `alloc_align`.
                    // FIXME: We could be even more general, e.g., offset 2 into a 4-aligned
                    // allocation cannot equal 3.
                    if bits % alloc_align.bytes() != 0 {
                        // The integer is *not* aligned. So they cannot be equal.
                        return Ok(false);
                    }
                }
                // Case III: The integer is too big, and the allocation goes on a bit
                // without wrapping around the address space.
                {
                    // Compute the highest address at which this allocation could live.
                    // Substract one more, because it must be possible to add the size
                    // to the base address without overflowing; that is, the very last address
                    // of the address space is never dereferencable (but it can be in-bounds, i.e.,
                    // one-past-the-end).
                    let max_base_addr =
                        ((1u128 << self.pointer_size().bits())
                         - u128::from(alloc_size.bytes())
                         - 1
                        ) as u64;
                    if let Some(max_addr) = max_base_addr.checked_add(ptr.offset.bytes()) {
                        if bits > max_addr {
                            // The integer is too big, this cannot possibly be equal.
                            return Ok(false)
                        }
                    }
                }

                // None of the supported cases.
                return err!(InvalidPointerMath);
            }
        })
    }

    fn ptr_int_arithmetic(
        &self,
        bin_op: mir::BinOp,
        left: Pointer<Tag>,
        right: u128,
        signed: bool,
    ) -> EvalResult<'tcx, (Scalar<Tag>, bool)> {
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
                let ptr_base_align = self.memory().get(left.alloc_id)?.align.bytes();
                let base_mask = {
                    // FIXME: use `interpret::truncate`, once that takes a `Size` instead of a `Layout`.
                    let shift = 128 - self.memory().pointer_size().bits();
                    let value = !(ptr_base_align as u128 - 1);
                    // Truncate (shift left to drop out leftover values, shift right to fill with zeroes).
                    (value << shift) >> shift
                };
                let ptr_size = self.memory().pointer_size().bytes() as u8;
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
                    (Scalar::Bits { bits: (left.offset.bytes() as u128) & right, size: ptr_size }, false)
                } else {
                    return err!(ReadPointerAsBytes);
                }
            }

            Rem if !signed => {
                // Doing modulo a divisor of the alignment is allowed.
                // (Intuition: modulo a divisor leaks less information.)
                let ptr_base_align = self.memory().get(left.alloc_id)?.align.bytes();
                let right = right as u64;
                let ptr_size = self.memory().pointer_size().bytes() as u8;
                if right == 1 {
                    // Modulo 1 is always 0.
                    (Scalar::Bits { bits: 0, size: ptr_size }, false)
                } else if ptr_base_align % right == 0 {
                    // The base address would be cancelled out by the modulo operation, so we can
                    // just take the modulo of the offset.
                    (
                        Scalar::Bits {
                            bits: (left.offset.bytes() % right) as u128,
                            size: ptr_size
                        },
                        false,
                    )
                } else {
                    return err!(ReadPointerAsBytes);
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
                return err!(Unimplemented(msg));
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
    ) -> EvalResult<'tcx, Scalar<Tag>> {
        // FIXME: assuming here that type size is less than `i64::max_value()`.
        let pointee_size = self.layout_of(pointee_ty)?.size.bytes() as i64;
        let offset = offset
            .checked_mul(pointee_size)
            .ok_or_else(|| InterpError::Overflow(mir::BinOp::Mul))?;
        // Now let's see what kind of pointer this is.
        if let Scalar::Ptr(ptr) = ptr {
            // Both old and new pointer must be in-bounds of a *live* allocation.
            // (Of the same allocation, but that part is trivial with our representation.)
            self.memory().check_bounds_ptr(ptr, InboundsCheck::Live)?;
            let ptr = ptr.signed_offset(offset, self)?;
            self.memory().check_bounds_ptr(ptr, InboundsCheck::Live)?;
            Ok(Scalar::Ptr(ptr))
        } else {
            // An integer pointer. They can only be offset by 0, and we pretend there
            // is a little zero-sized allocation here.
            if offset == 0 {
                Ok(ptr)
            } else {
                err!(InvalidPointerMath)
            }
        }
    }
}
