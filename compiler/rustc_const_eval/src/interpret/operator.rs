use rustc_apfloat::{Float, FloatConvert};
use rustc_middle::mir;
use rustc_middle::mir::interpret::{InterpResult, Scalar};
use rustc_middle::ty::layout::{LayoutOf, TyAndLayout};
use rustc_middle::ty::{self, FloatTy, Ty};
use rustc_span::symbol::sym;
use rustc_target::abi::Abi;

use super::{ImmTy, Immediate, InterpCx, Machine, PlaceTy};

use crate::fluent_generated as fluent;

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    /// Applies the binary operation `op` to the two operands and writes a tuple of the result
    /// and a boolean signifying the potential overflow to the destination.
    pub fn binop_with_overflow(
        &mut self,
        op: mir::BinOp,
        left: &ImmTy<'tcx, M::Provenance>,
        right: &ImmTy<'tcx, M::Provenance>,
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        let (val, overflowed) = self.overflowing_binary_op(op, left, right)?;
        debug_assert_eq!(
            Ty::new_tup(self.tcx.tcx, &[val.layout.ty, self.tcx.types.bool]),
            dest.layout.ty,
            "type mismatch for result of {op:?}",
        );
        // Write the result to `dest`.
        if let Abi::ScalarPair(..) = dest.layout.abi {
            // We can use the optimized path and avoid `place_field` (which might do
            // `force_allocation`).
            let pair = Immediate::ScalarPair(val.to_scalar(), Scalar::from_bool(overflowed));
            self.write_immediate(pair, dest)?;
        } else {
            assert!(self.tcx.sess.opts.unstable_opts.randomize_layout);
            // With randomized layout, `(int, bool)` might cease to be a `ScalarPair`, so we have to
            // do a component-wise write here. This code path is slower than the above because
            // `place_field` will have to `force_allocate` locals here.
            let val_field = self.project_field(dest, 0)?;
            self.write_scalar(val.to_scalar(), &val_field)?;
            let overflowed_field = self.project_field(dest, 1)?;
            self.write_scalar(Scalar::from_bool(overflowed), &overflowed_field)?;
        }
        Ok(())
    }

    /// Applies the binary operation `op` to the arguments and writes the result to the
    /// destination.
    pub fn binop_ignore_overflow(
        &mut self,
        op: mir::BinOp,
        left: &ImmTy<'tcx, M::Provenance>,
        right: &ImmTy<'tcx, M::Provenance>,
        dest: &PlaceTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx> {
        let val = self.wrapping_binary_op(op, left, right)?;
        assert_eq!(val.layout.ty, dest.layout.ty, "type mismatch for result of {op:?}");
        self.write_immediate(*val, dest)
    }
}

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    fn binary_char_op(
        &self,
        bin_op: mir::BinOp,
        l: char,
        r: char,
    ) -> (ImmTy<'tcx, M::Provenance>, bool) {
        use rustc_middle::mir::BinOp::*;

        let res = match bin_op {
            Eq => l == r,
            Ne => l != r,
            Lt => l < r,
            Le => l <= r,
            Gt => l > r,
            Ge => l >= r,
            _ => span_bug!(self.cur_span(), "Invalid operation on char: {:?}", bin_op),
        };
        (ImmTy::from_bool(res, *self.tcx), false)
    }

    fn binary_bool_op(
        &self,
        bin_op: mir::BinOp,
        l: bool,
        r: bool,
    ) -> (ImmTy<'tcx, M::Provenance>, bool) {
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
        (ImmTy::from_bool(res, *self.tcx), false)
    }

    fn binary_float_op<F: Float + FloatConvert<F> + Into<Scalar<M::Provenance>>>(
        &self,
        bin_op: mir::BinOp,
        layout: TyAndLayout<'tcx>,
        l: F,
        r: F,
    ) -> (ImmTy<'tcx, M::Provenance>, bool) {
        use rustc_middle::mir::BinOp::*;

        // Performs appropriate non-deterministic adjustments of NaN results.
        let adjust_nan =
            |f: F| -> F { if f.is_nan() { M::generate_nan(self, &[l, r]) } else { f } };

        let val = match bin_op {
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
        };
        (val, false)
    }

    fn binary_int_op(
        &self,
        bin_op: mir::BinOp,
        // passing in raw bits
        l: u128,
        left_layout: TyAndLayout<'tcx>,
        r: u128,
        right_layout: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, (ImmTy<'tcx, M::Provenance>, bool)> {
        use rustc_middle::mir::BinOp::*;

        let throw_ub_on_overflow = match bin_op {
            AddUnchecked => Some(sym::unchecked_add),
            SubUnchecked => Some(sym::unchecked_sub),
            MulUnchecked => Some(sym::unchecked_mul),
            ShlUnchecked => Some(sym::unchecked_shl),
            ShrUnchecked => Some(sym::unchecked_shr),
            _ => None,
        };

        // Shift ops can have an RHS with a different numeric type.
        if matches!(bin_op, Shl | ShlUnchecked | Shr | ShrUnchecked) {
            let size = left_layout.size.bits();
            // The shift offset is implicitly masked to the type size. (This is the one MIR operator
            // that does *not* directly map to a single LLVM operation.) Compute how much we
            // actually shift and whether there was an overflow due to shifting too much.
            let (shift_amount, overflow) = if right_layout.abi.is_signed() {
                let shift_amount = self.sign_extend(r, right_layout) as i128;
                let overflow = shift_amount < 0 || shift_amount >= i128::from(size);
                let masked_amount = (shift_amount as u128) % u128::from(size);
                debug_assert_eq!(overflow, shift_amount != (masked_amount as i128));
                (masked_amount, overflow)
            } else {
                let shift_amount = r;
                let masked_amount = shift_amount % u128::from(size);
                (masked_amount, shift_amount != masked_amount)
            };
            let shift_amount = u32::try_from(shift_amount).unwrap(); // we masked so this will always fit
            // Compute the shifted result.
            let result = if left_layout.abi.is_signed() {
                let l = self.sign_extend(l, left_layout) as i128;
                let result = match bin_op {
                    Shl | ShlUnchecked => l.checked_shl(shift_amount).unwrap(),
                    Shr | ShrUnchecked => l.checked_shr(shift_amount).unwrap(),
                    _ => bug!(),
                };
                result as u128
            } else {
                match bin_op {
                    Shl | ShlUnchecked => l.checked_shl(shift_amount).unwrap(),
                    Shr | ShrUnchecked => l.checked_shr(shift_amount).unwrap(),
                    _ => bug!(),
                }
            };
            let truncated = self.truncate(result, left_layout);

            if overflow && let Some(intrinsic_name) = throw_ub_on_overflow {
                throw_ub_custom!(
                    fluent::const_eval_overflow_shift,
                    val = if right_layout.abi.is_signed() {
                        (self.sign_extend(r, right_layout) as i128).to_string()
                    } else {
                        r.to_string()
                    },
                    name = intrinsic_name
                );
            }

            return Ok((ImmTy::from_uint(truncated, left_layout), overflow));
        }

        // For the remaining ops, the types must be the same on both sides
        if left_layout.ty != right_layout.ty {
            span_bug!(
                self.cur_span(),
                "invalid asymmetric binary op {bin_op:?}: {l:?} ({l_ty}), {r:?} ({r_ty})",
                l_ty = left_layout.ty,
                r_ty = right_layout.ty,
            )
        }

        let size = left_layout.size;

        // Operations that need special treatment for signed integers
        if left_layout.abi.is_signed() {
            let op: Option<fn(&i128, &i128) -> bool> = match bin_op {
                Lt => Some(i128::lt),
                Le => Some(i128::le),
                Gt => Some(i128::gt),
                Ge => Some(i128::ge),
                _ => None,
            };
            if let Some(op) = op {
                let l = self.sign_extend(l, left_layout) as i128;
                let r = self.sign_extend(r, right_layout) as i128;
                return Ok((ImmTy::from_bool(op(&l, &r), *self.tcx), false));
            }
            let op: Option<fn(i128, i128) -> (i128, bool)> = match bin_op {
                Div if r == 0 => throw_ub!(DivisionByZero),
                Rem if r == 0 => throw_ub!(RemainderByZero),
                Div => Some(i128::overflowing_div),
                Rem => Some(i128::overflowing_rem),
                Add | AddUnchecked => Some(i128::overflowing_add),
                Sub | SubUnchecked => Some(i128::overflowing_sub),
                Mul | MulUnchecked => Some(i128::overflowing_mul),
                _ => None,
            };
            if let Some(op) = op {
                let l = self.sign_extend(l, left_layout) as i128;
                let r = self.sign_extend(r, right_layout) as i128;

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
                // This may be out-of-bounds for the result type, so we have to truncate ourselves.
                // If that truncation loses any information, we have an overflow.
                let result = result as u128;
                let truncated = self.truncate(result, left_layout);
                let overflow = oflo || self.sign_extend(truncated, left_layout) != result;
                if overflow && let Some(intrinsic_name) = throw_ub_on_overflow {
                    throw_ub_custom!(fluent::const_eval_overflow, name = intrinsic_name);
                }
                return Ok((ImmTy::from_uint(truncated, left_layout), overflow));
            }
        }

        let val = match bin_op {
            Eq => ImmTy::from_bool(l == r, *self.tcx),
            Ne => ImmTy::from_bool(l != r, *self.tcx),

            Lt => ImmTy::from_bool(l < r, *self.tcx),
            Le => ImmTy::from_bool(l <= r, *self.tcx),
            Gt => ImmTy::from_bool(l > r, *self.tcx),
            Ge => ImmTy::from_bool(l >= r, *self.tcx),

            BitOr => ImmTy::from_uint(l | r, left_layout),
            BitAnd => ImmTy::from_uint(l & r, left_layout),
            BitXor => ImmTy::from_uint(l ^ r, left_layout),

            Add | AddUnchecked | Sub | SubUnchecked | Mul | MulUnchecked | Rem | Div => {
                assert!(!left_layout.abi.is_signed());
                let op: fn(u128, u128) -> (u128, bool) = match bin_op {
                    Add | AddUnchecked => u128::overflowing_add,
                    Sub | SubUnchecked => u128::overflowing_sub,
                    Mul | MulUnchecked => u128::overflowing_mul,
                    Div if r == 0 => throw_ub!(DivisionByZero),
                    Rem if r == 0 => throw_ub!(RemainderByZero),
                    Div => u128::overflowing_div,
                    Rem => u128::overflowing_rem,
                    _ => bug!(),
                };
                let (result, oflo) = op(l, r);
                // Truncate to target type.
                // If that truncation loses any information, we have an overflow.
                let truncated = self.truncate(result, left_layout);
                let overflow = oflo || truncated != result;
                if overflow && let Some(intrinsic_name) = throw_ub_on_overflow {
                    throw_ub_custom!(fluent::const_eval_overflow, name = intrinsic_name);
                }
                return Ok((ImmTy::from_uint(truncated, left_layout), overflow));
            }

            _ => span_bug!(
                self.cur_span(),
                "invalid binary op {:?}: {:?}, {:?} (both {})",
                bin_op,
                l,
                r,
                right_layout.ty,
            ),
        };

        Ok((val, false))
    }

    fn binary_ptr_op(
        &self,
        bin_op: mir::BinOp,
        left: &ImmTy<'tcx, M::Provenance>,
        right: &ImmTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, (ImmTy<'tcx, M::Provenance>, bool)> {
        use rustc_middle::mir::BinOp::*;

        match bin_op {
            // Pointer ops that are always supported.
            Offset => {
                let ptr = left.to_scalar().to_pointer(self)?;
                let offset_count = right.to_scalar().to_target_isize(self)?;
                let pointee_ty = left.layout.ty.builtin_deref(true).unwrap().ty;

                // We cannot overflow i64 as a type's size must be <= isize::MAX.
                let pointee_size = i64::try_from(self.layout_of(pointee_ty)?.size.bytes()).unwrap();
                // The computed offset, in bytes, must not overflow an isize.
                // `checked_mul` enforces a too small bound, but no actual allocation can be big enough for
                // the difference to be noticeable.
                let offset_bytes =
                    offset_count.checked_mul(pointee_size).ok_or(err_ub!(PointerArithOverflow))?;

                let offset_ptr = self.ptr_offset_inbounds(ptr, offset_bytes)?;
                Ok((
                    ImmTy::from_scalar(Scalar::from_maybe_pointer(offset_ptr, self), left.layout),
                    false,
                ))
            }

            // Fall back to machine hook so Miri can support more pointer ops.
            _ => M::binary_ptr_op(self, bin_op, left, right),
        }
    }

    /// Returns the result of the specified operation, and whether it overflowed.
    pub fn overflowing_binary_op(
        &self,
        bin_op: mir::BinOp,
        left: &ImmTy<'tcx, M::Provenance>,
        right: &ImmTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, (ImmTy<'tcx, M::Provenance>, bool)> {
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

                let l = left.to_scalar().to_bits(left.layout.size)?;
                let r = right.to_scalar().to_bits(right.layout.size)?;
                self.binary_int_op(bin_op, l, left.layout, r, right.layout)
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

    #[inline]
    pub fn wrapping_binary_op(
        &self,
        bin_op: mir::BinOp,
        left: &ImmTy<'tcx, M::Provenance>,
        right: &ImmTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        let (val, _overflow) = self.overflowing_binary_op(bin_op, left, right)?;
        Ok(val)
    }

    /// Returns the result of the specified operation, whether it overflowed, and
    /// the result type.
    pub fn overflowing_unary_op(
        &self,
        un_op: mir::UnOp,
        val: &ImmTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, (ImmTy<'tcx, M::Provenance>, bool)> {
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
                Ok((ImmTy::from_bool(res, *self.tcx), false))
            }
            ty::Float(fty) => {
                // No NaN adjustment here, `-` is a bitwise operation!
                let res = match (un_op, fty) {
                    (Neg, FloatTy::F32) => Scalar::from_f32(-val.to_f32()?),
                    (Neg, FloatTy::F64) => Scalar::from_f64(-val.to_f64()?),
                    _ => span_bug!(self.cur_span(), "Invalid float op {:?}", un_op),
                };
                Ok((ImmTy::from_scalar(res, layout), false))
            }
            _ => {
                assert!(layout.ty.is_integral());
                let val = val.to_bits(layout.size)?;
                let (res, overflow) = match un_op {
                    Not => (self.truncate(!val, layout), false), // bitwise negation, then truncate
                    Neg => {
                        // arithmetic negation
                        assert!(layout.abi.is_signed());
                        let val = self.sign_extend(val, layout) as i128;
                        let (res, overflow) = val.overflowing_neg();
                        let res = res as u128;
                        // Truncate to target type.
                        // If that truncation loses any information, we have an overflow.
                        let truncated = self.truncate(res, layout);
                        (truncated, overflow || self.sign_extend(truncated, layout) != res)
                    }
                };
                Ok((ImmTy::from_uint(res, layout), overflow))
            }
        }
    }

    #[inline]
    pub fn wrapping_unary_op(
        &self,
        un_op: mir::UnOp,
        val: &ImmTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        let (val, _overflow) = self.overflowing_unary_op(un_op, val)?;
        Ok(val)
    }
}
