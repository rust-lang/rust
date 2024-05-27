use either::Either;

use rustc_apfloat::{Float, FloatConvert};
use rustc_middle::mir;
use rustc_middle::mir::interpret::{InterpResult, Scalar};
use rustc_middle::ty::layout::{LayoutOf, TyAndLayout};
use rustc_middle::ty::{self, FloatTy, ScalarInt};
use rustc_middle::{bug, span_bug};
use rustc_span::symbol::sym;
use tracing::trace;

use super::{err_ub, throw_ub, ImmTy, InterpCx, Machine};

impl<'tcx, M: Machine<'tcx>> InterpCx<'tcx, M> {
    fn three_way_compare<T: Ord>(&self, lhs: T, rhs: T) -> ImmTy<'tcx, M::Provenance> {
        let res = Ord::cmp(&lhs, &rhs);
        return ImmTy::from_ordering(res, *self.tcx);
    }

    fn binary_char_op(&self, bin_op: mir::BinOp, l: char, r: char) -> ImmTy<'tcx, M::Provenance> {
        use rustc_middle::mir::BinOp::*;

        if bin_op == Cmp {
            return self.three_way_compare(l, r);
        }

        let res = match bin_op {
            Eq => l == r,
            Ne => l != r,
            Lt => l < r,
            Le => l <= r,
            Gt => l > r,
            Ge => l >= r,
            _ => span_bug!(self.cur_span(), "Invalid operation on char: {:?}", bin_op),
        };
        ImmTy::from_bool(res, *self.tcx)
    }

    fn binary_bool_op(&self, bin_op: mir::BinOp, l: bool, r: bool) -> ImmTy<'tcx, M::Provenance> {
        use rustc_middle::mir::BinOp::*;

        let res = match bin_op {
            Eq => l == r,
            Ne => l != r,
            Lt => l < r,
            Le => l <= r,
            Gt => l > r,
            Ge => l >= r,
            BitAnd => l & r,
            BitOr => l | r,
            BitXor => l ^ r,
            _ => span_bug!(self.cur_span(), "Invalid operation on bool: {:?}", bin_op),
        };
        ImmTy::from_bool(res, *self.tcx)
    }

    fn binary_float_op<F: Float + FloatConvert<F> + Into<Scalar<M::Provenance>>>(
        &self,
        bin_op: mir::BinOp,
        layout: TyAndLayout<'tcx>,
        l: F,
        r: F,
    ) -> ImmTy<'tcx, M::Provenance> {
        use rustc_middle::mir::BinOp::*;

        // Performs appropriate non-deterministic adjustments of NaN results.
        let adjust_nan =
            |f: F| -> F { if f.is_nan() { M::generate_nan(self, &[l, r]) } else { f } };

        match bin_op {
            Eq => ImmTy::from_bool(l == r, *self.tcx),
            Ne => ImmTy::from_bool(l != r, *self.tcx),
            Lt => ImmTy::from_bool(l < r, *self.tcx),
            Le => ImmTy::from_bool(l <= r, *self.tcx),
            Gt => ImmTy::from_bool(l > r, *self.tcx),
            Ge => ImmTy::from_bool(l >= r, *self.tcx),
            Add => ImmTy::from_scalar(adjust_nan((l + r).value).into(), layout),
            Sub => ImmTy::from_scalar(adjust_nan((l - r).value).into(), layout),
            Mul => ImmTy::from_scalar(adjust_nan((l * r).value).into(), layout),
            Div => ImmTy::from_scalar(adjust_nan((l / r).value).into(), layout),
            Rem => ImmTy::from_scalar(adjust_nan((l % r).value).into(), layout),
            _ => span_bug!(self.cur_span(), "invalid float op: `{:?}`", bin_op),
        }
    }

    fn binary_int_op(
        &self,
        bin_op: mir::BinOp,
        left: &ImmTy<'tcx, M::Provenance>,
        right: &ImmTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        use rustc_middle::mir::BinOp::*;

        // This checks the size, so that we can just assert it below.
        let l = left.to_scalar_int()?;
        let r = right.to_scalar_int()?;
        // Prepare to convert the values to signed or unsigned form.
        let l_signed = || l.assert_int(left.layout.size);
        let l_unsigned = || l.assert_uint(left.layout.size);
        let r_signed = || r.assert_int(right.layout.size);
        let r_unsigned = || r.assert_uint(right.layout.size);

        let throw_ub_on_overflow = match bin_op {
            AddUnchecked => Some(sym::unchecked_add),
            SubUnchecked => Some(sym::unchecked_sub),
            MulUnchecked => Some(sym::unchecked_mul),
            ShlUnchecked => Some(sym::unchecked_shl),
            ShrUnchecked => Some(sym::unchecked_shr),
            _ => None,
        };
        let with_overflow = bin_op.is_overflowing();

        // Shift ops can have an RHS with a different numeric type.
        if matches!(bin_op, Shl | ShlUnchecked | Shr | ShrUnchecked) {
            let size = left.layout.size.bits();
            // Compute the equivalent shift modulo `size` that is in the range `0..size`. (This is
            // the one MIR operator that does *not* directly map to a single LLVM operation.)
            let (shift_amount, overflow) = if right.layout.abi.is_signed() {
                let shift_amount = r_signed();
                let overflow = shift_amount < 0 || shift_amount >= i128::from(size);
                // Deliberately wrapping `as` casts: shift_amount *can* be negative, but the result
                // of the `as` will be equal modulo `size` (since it is a power of two).
                let masked_amount = (shift_amount as u128) % u128::from(size);
                assert_eq!(overflow, shift_amount != i128::try_from(masked_amount).unwrap());
                (masked_amount, overflow)
            } else {
                let shift_amount = r_unsigned();
                let overflow = shift_amount >= u128::from(size);
                let masked_amount = shift_amount % u128::from(size);
                assert_eq!(overflow, shift_amount != masked_amount);
                (masked_amount, overflow)
            };
            let shift_amount = u32::try_from(shift_amount).unwrap(); // we masked so this will always fit
            // Compute the shifted result.
            let result = if left.layout.abi.is_signed() {
                let l = l_signed();
                let result = match bin_op {
                    Shl | ShlUnchecked => l.checked_shl(shift_amount).unwrap(),
                    Shr | ShrUnchecked => l.checked_shr(shift_amount).unwrap(),
                    _ => bug!(),
                };
                ScalarInt::truncate_from_int(result, left.layout.size).0
            } else {
                let l = l_unsigned();
                let result = match bin_op {
                    Shl | ShlUnchecked => l.checked_shl(shift_amount).unwrap(),
                    Shr | ShrUnchecked => l.checked_shr(shift_amount).unwrap(),
                    _ => bug!(),
                };
                ScalarInt::truncate_from_uint(result, left.layout.size).0
            };

            if overflow && let Some(intrinsic) = throw_ub_on_overflow {
                throw_ub!(ShiftOverflow {
                    intrinsic,
                    shift_amount: if right.layout.abi.is_signed() {
                        Either::Right(r_signed())
                    } else {
                        Either::Left(r_unsigned())
                    }
                });
            }

            return Ok(ImmTy::from_scalar_int(result, left.layout));
        }

        // For the remaining ops, the types must be the same on both sides
        if left.layout.ty != right.layout.ty {
            span_bug!(
                self.cur_span(),
                "invalid asymmetric binary op {bin_op:?}: {l:?} ({l_ty}), {r:?} ({r_ty})",
                l_ty = left.layout.ty,
                r_ty = right.layout.ty,
            )
        }

        let size = left.layout.size;

        // Operations that need special treatment for signed integers
        if left.layout.abi.is_signed() {
            let op: Option<fn(&i128, &i128) -> bool> = match bin_op {
                Lt => Some(i128::lt),
                Le => Some(i128::le),
                Gt => Some(i128::gt),
                Ge => Some(i128::ge),
                _ => None,
            };
            if let Some(op) = op {
                return Ok(ImmTy::from_bool(op(&l_signed(), &r_signed()), *self.tcx));
            }
            if bin_op == Cmp {
                return Ok(self.three_way_compare(l_signed(), r_signed()));
            }
            let op: Option<fn(i128, i128) -> (i128, bool)> = match bin_op {
                Div if r.is_null() => throw_ub!(DivisionByZero),
                Rem if r.is_null() => throw_ub!(RemainderByZero),
                Div => Some(i128::overflowing_div),
                Rem => Some(i128::overflowing_rem),
                Add | AddUnchecked | AddWithOverflow => Some(i128::overflowing_add),
                Sub | SubUnchecked | SubWithOverflow => Some(i128::overflowing_sub),
                Mul | MulUnchecked | MulWithOverflow => Some(i128::overflowing_mul),
                _ => None,
            };
            if let Some(op) = op {
                let l = l_signed();
                let r = r_signed();

                // We need a special check for overflowing Rem and Div since they are *UB*
                // on overflow, which can happen with "int_min $OP -1".
                if matches!(bin_op, Rem | Div) {
                    if l == size.signed_int_min() && r == -1 {
                        if bin_op == Rem {
                            throw_ub!(RemainderOverflow)
                        } else {
                            throw_ub!(DivisionOverflow)
                        }
                    }
                }

                let (result, oflo) = op(l, r);
                // This may be out-of-bounds for the result type, so we have to truncate.
                // If that truncation loses any information, we have an overflow.
                let (result, lossy) = ScalarInt::truncate_from_int(result, left.layout.size);
                let overflow = oflo || lossy;
                if overflow && let Some(intrinsic) = throw_ub_on_overflow {
                    throw_ub!(ArithOverflow { intrinsic });
                }
                let res = ImmTy::from_scalar_int(result, left.layout);
                return Ok(if with_overflow {
                    let overflow = ImmTy::from_bool(overflow, *self.tcx);
                    ImmTy::from_pair(res, overflow, *self.tcx)
                } else {
                    res
                });
            }
        }
        // From here on it's okay to treat everything as unsigned.
        let l = l_unsigned();
        let r = r_unsigned();

        if bin_op == Cmp {
            return Ok(self.three_way_compare(l, r));
        }

        Ok(match bin_op {
            Eq => ImmTy::from_bool(l == r, *self.tcx),
            Ne => ImmTy::from_bool(l != r, *self.tcx),

            Lt => ImmTy::from_bool(l < r, *self.tcx),
            Le => ImmTy::from_bool(l <= r, *self.tcx),
            Gt => ImmTy::from_bool(l > r, *self.tcx),
            Ge => ImmTy::from_bool(l >= r, *self.tcx),

            BitOr => ImmTy::from_uint(l | r, left.layout),
            BitAnd => ImmTy::from_uint(l & r, left.layout),
            BitXor => ImmTy::from_uint(l ^ r, left.layout),

            _ => {
                assert!(!left.layout.abi.is_signed());
                let op: fn(u128, u128) -> (u128, bool) = match bin_op {
                    Add | AddUnchecked | AddWithOverflow => u128::overflowing_add,
                    Sub | SubUnchecked | SubWithOverflow => u128::overflowing_sub,
                    Mul | MulUnchecked | MulWithOverflow => u128::overflowing_mul,
                    Div if r == 0 => throw_ub!(DivisionByZero),
                    Rem if r == 0 => throw_ub!(RemainderByZero),
                    Div => u128::overflowing_div,
                    Rem => u128::overflowing_rem,
                    _ => span_bug!(
                        self.cur_span(),
                        "invalid binary op {:?}: {:?}, {:?} (both {})",
                        bin_op,
                        left,
                        right,
                        right.layout.ty,
                    ),
                };
                let (result, oflo) = op(l, r);
                // Truncate to target type.
                // If that truncation loses any information, we have an overflow.
                let (result, lossy) = ScalarInt::truncate_from_uint(result, left.layout.size);
                let overflow = oflo || lossy;
                if overflow && let Some(intrinsic) = throw_ub_on_overflow {
                    throw_ub!(ArithOverflow { intrinsic });
                }
                let res = ImmTy::from_scalar_int(result, left.layout);
                if with_overflow {
                    let overflow = ImmTy::from_bool(overflow, *self.tcx);
                    ImmTy::from_pair(res, overflow, *self.tcx)
                } else {
                    res
                }
            }
        })
    }

    fn binary_ptr_op(
        &self,
        bin_op: mir::BinOp,
        left: &ImmTy<'tcx, M::Provenance>,
        right: &ImmTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        use rustc_middle::mir::BinOp::*;

        match bin_op {
            // Pointer ops that are always supported.
            Offset => {
                let ptr = left.to_scalar().to_pointer(self)?;
                let offset_count = right.to_scalar().to_target_isize(self)?;
                let pointee_ty = left.layout.ty.builtin_deref(true).unwrap();

                // We cannot overflow i64 as a type's size must be <= isize::MAX.
                let pointee_size = i64::try_from(self.layout_of(pointee_ty)?.size.bytes()).unwrap();
                // The computed offset, in bytes, must not overflow an isize.
                // `checked_mul` enforces a too small bound, but no actual allocation can be big enough for
                // the difference to be noticeable.
                let offset_bytes =
                    offset_count.checked_mul(pointee_size).ok_or(err_ub!(PointerArithOverflow))?;

                let offset_ptr = self.ptr_offset_inbounds(ptr, offset_bytes)?;
                Ok(ImmTy::from_scalar(Scalar::from_maybe_pointer(offset_ptr, self), left.layout))
            }

            // Fall back to machine hook so Miri can support more pointer ops.
            _ => M::binary_ptr_op(self, bin_op, left, right),
        }
    }

    /// Returns the result of the specified operation.
    ///
    /// Whether this produces a scalar or a pair depends on the specific `bin_op`.
    pub fn binary_op(
        &self,
        bin_op: mir::BinOp,
        left: &ImmTy<'tcx, M::Provenance>,
        right: &ImmTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        trace!(
            "Running binary op {:?}: {:?} ({}), {:?} ({})",
            bin_op,
            *left,
            left.layout.ty,
            *right,
            right.layout.ty
        );

        match left.layout.ty.kind() {
            ty::Char => {
                assert_eq!(left.layout.ty, right.layout.ty);
                let left = left.to_scalar();
                let right = right.to_scalar();
                Ok(self.binary_char_op(bin_op, left.to_char()?, right.to_char()?))
            }
            ty::Bool => {
                assert_eq!(left.layout.ty, right.layout.ty);
                let left = left.to_scalar();
                let right = right.to_scalar();
                Ok(self.binary_bool_op(bin_op, left.to_bool()?, right.to_bool()?))
            }
            ty::Float(fty) => {
                assert_eq!(left.layout.ty, right.layout.ty);
                let layout = left.layout;
                let left = left.to_scalar();
                let right = right.to_scalar();
                Ok(match fty {
                    FloatTy::F16 => unimplemented!("f16_f128"),
                    FloatTy::F32 => {
                        self.binary_float_op(bin_op, layout, left.to_f32()?, right.to_f32()?)
                    }
                    FloatTy::F64 => {
                        self.binary_float_op(bin_op, layout, left.to_f64()?, right.to_f64()?)
                    }
                    FloatTy::F128 => unimplemented!("f16_f128"),
                })
            }
            _ if left.layout.ty.is_integral() => {
                // the RHS type can be different, e.g. for shifts -- but it has to be integral, too
                assert!(
                    right.layout.ty.is_integral(),
                    "Unexpected types for BinOp: {} {:?} {}",
                    left.layout.ty,
                    bin_op,
                    right.layout.ty
                );

                self.binary_int_op(bin_op, left, right)
            }
            _ if left.layout.ty.is_any_ptr() => {
                // The RHS type must be a `pointer` *or an integer type* (for `Offset`).
                // (Even when both sides are pointers, their type might differ, see issue #91636)
                assert!(
                    right.layout.ty.is_any_ptr() || right.layout.ty.is_integral(),
                    "Unexpected types for BinOp: {} {:?} {}",
                    left.layout.ty,
                    bin_op,
                    right.layout.ty
                );

                self.binary_ptr_op(bin_op, left, right)
            }
            _ => span_bug!(
                self.cur_span(),
                "Invalid MIR: bad LHS type for binop: {}",
                left.layout.ty
            ),
        }
    }

    /// Returns the result of the specified operation, whether it overflowed, and
    /// the result type.
    pub fn unary_op(
        &self,
        un_op: mir::UnOp,
        val: &ImmTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        use rustc_middle::mir::UnOp::*;

        let layout = val.layout;
        let val = val.to_scalar();
        trace!("Running unary op {:?}: {:?} ({})", un_op, val, layout.ty);

        match layout.ty.kind() {
            ty::Bool => {
                let val = val.to_bool()?;
                let res = match un_op {
                    Not => !val,
                    _ => span_bug!(self.cur_span(), "Invalid bool op {:?}", un_op),
                };
                Ok(ImmTy::from_bool(res, *self.tcx))
            }
            ty::Float(fty) => {
                // No NaN adjustment here, `-` is a bitwise operation!
                let res = match (un_op, fty) {
                    (Neg, FloatTy::F32) => Scalar::from_f32(-val.to_f32()?),
                    (Neg, FloatTy::F64) => Scalar::from_f64(-val.to_f64()?),
                    _ => span_bug!(self.cur_span(), "Invalid float op {:?}", un_op),
                };
                Ok(ImmTy::from_scalar(res, layout))
            }
            _ => {
                assert!(layout.ty.is_integral());
                let val = val.to_bits(layout.size)?;
                let res = match un_op {
                    Not => self.truncate(!val, layout), // bitwise negation, then truncate
                    Neg => {
                        // arithmetic negation
                        assert!(layout.abi.is_signed());
                        let val = self.sign_extend(val, layout) as i128;
                        let res = val.wrapping_neg();
                        let res = res as u128;
                        // Truncate to target type.
                        self.truncate(res, layout)
                    }
                };
                Ok(ImmTy::from_uint(res, layout))
            }
        }
    }
}
