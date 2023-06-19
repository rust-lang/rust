use rustc_apfloat::Float;
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
        let (val, overflowed, ty) = self.overflowing_binary_op(op, &left, &right)?;
        debug_assert_eq!(
            self.tcx.mk_tup(&[ty, self.tcx.types.bool]),
            dest.layout.ty,
            "type mismatch for result of {:?}",
            op,
        );
        // Write the result to `dest`.
        if let Abi::ScalarPair(..) = dest.layout.abi {
            // We can use the optimized path and avoid `place_field` (which might do
            // `force_allocation`).
            let pair = Immediate::ScalarPair(val, Scalar::from_bool(overflowed));
            self.write_immediate(pair, dest)?;
        } else {
            assert!(self.tcx.sess.opts.unstable_opts.randomize_layout);
            // With randomized layout, `(int, bool)` might cease to be a `ScalarPair`, so we have to
            // do a component-wise write here. This code path is slower than the above because
            // `place_field` will have to `force_allocate` locals here.
            let val_field = self.place_field(&dest, 0)?;
            self.write_scalar(val, &val_field)?;
            let overflowed_field = self.place_field(&dest, 1)?;
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
        let (val, _overflowed, ty) = self.overflowing_binary_op(op, left, right)?;
        assert_eq!(ty, dest.layout.ty, "type mismatch for result of {:?}", op);
        self.write_scalar(val, dest)
    }
}

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    fn binary_char_op(
        &self,
        bin_op: mir::BinOp,
        l: char,
        r: char,
    ) -> (Scalar<M::Provenance>, bool, Ty<'tcx>) {
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
        (Scalar::from_bool(res), false, self.tcx.types.bool)
    }

    fn binary_bool_op(
        &self,
        bin_op: mir::BinOp,
        l: bool,
        r: bool,
    ) -> (Scalar<M::Provenance>, bool, Ty<'tcx>) {
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
        (Scalar::from_bool(res), false, self.tcx.types.bool)
    }

    fn binary_float_op<F: Float + Into<Scalar<M::Provenance>>>(
        &self,
        bin_op: mir::BinOp,
        ty: Ty<'tcx>,
        l: F,
        r: F,
    ) -> (Scalar<M::Provenance>, bool, Ty<'tcx>) {
        use rustc_middle::mir::BinOp::*;

        let (val, ty) = match bin_op {
            Eq => (Scalar::from_bool(l == r), self.tcx.types.bool),
            Ne => (Scalar::from_bool(l != r), self.tcx.types.bool),
            Lt => (Scalar::from_bool(l < r), self.tcx.types.bool),
            Le => (Scalar::from_bool(l <= r), self.tcx.types.bool),
            Gt => (Scalar::from_bool(l > r), self.tcx.types.bool),
            Ge => (Scalar::from_bool(l >= r), self.tcx.types.bool),
            Add => ((l + r).value.into(), ty),
            Sub => ((l - r).value.into(), ty),
            Mul => ((l * r).value.into(), ty),
            Div => ((l / r).value.into(), ty),
            Rem => ((l % r).value.into(), ty),
            _ => span_bug!(self.cur_span(), "invalid float op: `{:?}`", bin_op),
        };
        (val, false, ty)
    }

    fn binary_int_op(
        &self,
        bin_op: mir::BinOp,
        // passing in raw bits
        l: u128,
        left_layout: TyAndLayout<'tcx>,
        r: u128,
        right_layout: TyAndLayout<'tcx>,
    ) -> InterpResult<'tcx, (Scalar<M::Provenance>, bool, Ty<'tcx>)> {
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
            let size = u128::from(left_layout.size.bits());
            // Even if `r` is signed, we treat it as if it was unsigned (i.e., we use its
            // zero-extended form). This matches the codegen backend:
            // <https://github.com/rust-lang/rust/blob/c274e4969f058b1c644243181ece9f829efa7594/compiler/rustc_codegen_ssa/src/base.rs#L315-L317>.
            // The overflow check is also ignorant to the sign:
            // <https://github.com/rust-lang/rust/blob/c274e4969f058b1c644243181ece9f829efa7594/compiler/rustc_codegen_ssa/src/mir/rvalue.rs#L728>.
            // This would behave rather strangely if we had integer types of size 256: a shift by
            // -1i8 would actually shift by 255, but that would *not* be considered overflowing. A
            // shift by -1i16 though would be considered overflowing. If we had integers of size
            // 512, then a shift by -1i8 would even produce a different result than one by -1i16:
            // the first shifts by 255, the latter by u16::MAX % 512 = 511. Lucky enough, our
            // integers are maximally 128bits wide, so negative shifts *always* overflow and we have
            // consistent results for the same value represented at different bit widths.
            assert!(size <= 128);
            let original_r = r;
            let overflow = r >= size;
            // The shift offset is implicitly masked to the type size, to make sure this operation
            // is always defined. This is the one MIR operator that does *not* directly map to a
            // single LLVM operation. See
            // <https://github.com/rust-lang/rust/blob/c274e4969f058b1c644243181ece9f829efa7594/compiler/rustc_codegen_ssa/src/common.rs#L131-L158>
            // for the corresponding truncation in our codegen backends.
            let r = r % size;
            let r = u32::try_from(r).unwrap(); // we masked so this will always fit
            let result = if left_layout.abi.is_signed() {
                let l = self.sign_extend(l, left_layout) as i128;
                let result = match bin_op {
                    Shl | ShlUnchecked => l.checked_shl(r).unwrap(),
                    Shr | ShrUnchecked => l.checked_shr(r).unwrap(),
                    _ => bug!(),
                };
                result as u128
            } else {
                match bin_op {
                    Shl | ShlUnchecked => l.checked_shl(r).unwrap(),
                    Shr | ShrUnchecked => l.checked_shr(r).unwrap(),
                    _ => bug!(),
                }
            };
            let truncated = self.truncate(result, left_layout);

            if overflow && let Some(intrinsic_name) = throw_ub_on_overflow {
                throw_ub_custom!(
                    fluent::const_eval_overflow_shift,
                    val = original_r,
                    name = intrinsic_name
                );
            }

            return Ok((Scalar::from_uint(truncated, left_layout.size), overflow, left_layout.ty));
        }

        // For the remaining ops, the types must be the same on both sides
        if left_layout.ty != right_layout.ty {
            span_bug!(
                self.cur_span(),
                "invalid asymmetric binary op {:?}: {:?} ({:?}), {:?} ({:?})",
                bin_op,
                l,
                left_layout.ty,
                r,
                right_layout.ty,
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
                return Ok((Scalar::from_bool(op(&l, &r)), false, self.tcx.types.bool));
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
                return Ok((Scalar::from_uint(truncated, size), overflow, left_layout.ty));
            }
        }

        let (val, ty) = match bin_op {
            Eq => (Scalar::from_bool(l == r), self.tcx.types.bool),
            Ne => (Scalar::from_bool(l != r), self.tcx.types.bool),

            Lt => (Scalar::from_bool(l < r), self.tcx.types.bool),
            Le => (Scalar::from_bool(l <= r), self.tcx.types.bool),
            Gt => (Scalar::from_bool(l > r), self.tcx.types.bool),
            Ge => (Scalar::from_bool(l >= r), self.tcx.types.bool),

            BitOr => (Scalar::from_uint(l | r, size), left_layout.ty),
            BitAnd => (Scalar::from_uint(l & r, size), left_layout.ty),
            BitXor => (Scalar::from_uint(l ^ r, size), left_layout.ty),

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
                return Ok((Scalar::from_uint(truncated, size), overflow, left_layout.ty));
            }

            _ => span_bug!(
                self.cur_span(),
                "invalid binary op {:?}: {:?}, {:?} (both {:?})",
                bin_op,
                l,
                r,
                right_layout.ty,
            ),
        };

        Ok((val, false, ty))
    }

    fn binary_ptr_op(
        &self,
        bin_op: mir::BinOp,
        left: &ImmTy<'tcx, M::Provenance>,
        right: &ImmTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, (Scalar<M::Provenance>, bool, Ty<'tcx>)> {
        use rustc_middle::mir::BinOp::*;

        match bin_op {
            // Pointer ops that are always supported.
            Offset => {
                let ptr = left.to_scalar().to_pointer(self)?;
                let offset_count = right.to_scalar().to_target_isize(self)?;
                let pointee_ty = left.layout.ty.builtin_deref(true).unwrap().ty;

                let offset_ptr = self.ptr_offset_inbounds(ptr, pointee_ty, offset_count)?;
                Ok((Scalar::from_maybe_pointer(offset_ptr, self), false, left.layout.ty))
            }

            // Fall back to machine hook so Miri can support more pointer ops.
            _ => M::binary_ptr_op(self, bin_op, left, right),
        }
    }

    /// Returns the result of the specified operation, whether it overflowed, and
    /// the result type.
    pub fn overflowing_binary_op(
        &self,
        bin_op: mir::BinOp,
        left: &ImmTy<'tcx, M::Provenance>,
        right: &ImmTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, (Scalar<M::Provenance>, bool, Ty<'tcx>)> {
        trace!(
            "Running binary op {:?}: {:?} ({:?}), {:?} ({:?})",
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
                let ty = left.layout.ty;
                let left = left.to_scalar();
                let right = right.to_scalar();
                Ok(match fty {
                    FloatTy::F32 => {
                        self.binary_float_op(bin_op, ty, left.to_f32()?, right.to_f32()?)
                    }
                    FloatTy::F64 => {
                        self.binary_float_op(bin_op, ty, left.to_f64()?, right.to_f64()?)
                    }
                })
            }
            _ if left.layout.ty.is_integral() => {
                // the RHS type can be different, e.g. for shifts -- but it has to be integral, too
                assert!(
                    right.layout.ty.is_integral(),
                    "Unexpected types for BinOp: {:?} {:?} {:?}",
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
                    "Unexpected types for BinOp: {:?} {:?} {:?}",
                    left.layout.ty,
                    bin_op,
                    right.layout.ty
                );

                self.binary_ptr_op(bin_op, left, right)
            }
            _ => span_bug!(
                self.cur_span(),
                "Invalid MIR: bad LHS type for binop: {:?}",
                left.layout.ty
            ),
        }
    }

    /// Typed version of `overflowing_binary_op`, returning an `ImmTy`. Also ignores overflows.
    #[inline]
    pub fn binary_op(
        &self,
        bin_op: mir::BinOp,
        left: &ImmTy<'tcx, M::Provenance>,
        right: &ImmTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        let (val, _overflow, ty) = self.overflowing_binary_op(bin_op, left, right)?;
        Ok(ImmTy::from_scalar(val, self.layout_of(ty)?))
    }

    /// Returns the result of the specified operation, whether it overflowed, and
    /// the result type.
    pub fn overflowing_unary_op(
        &self,
        un_op: mir::UnOp,
        val: &ImmTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, (Scalar<M::Provenance>, bool, Ty<'tcx>)> {
        use rustc_middle::mir::UnOp::*;

        let layout = val.layout;
        let val = val.to_scalar();
        trace!("Running unary op {:?}: {:?} ({:?})", un_op, val, layout.ty);

        match layout.ty.kind() {
            ty::Bool => {
                let val = val.to_bool()?;
                let res = match un_op {
                    Not => !val,
                    _ => span_bug!(self.cur_span(), "Invalid bool op {:?}", un_op),
                };
                Ok((Scalar::from_bool(res), false, self.tcx.types.bool))
            }
            ty::Float(fty) => {
                let res = match (un_op, fty) {
                    (Neg, FloatTy::F32) => Scalar::from_f32(-val.to_f32()?),
                    (Neg, FloatTy::F64) => Scalar::from_f64(-val.to_f64()?),
                    _ => span_bug!(self.cur_span(), "Invalid float op {:?}", un_op),
                };
                Ok((res, false, layout.ty))
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
                Ok((Scalar::from_uint(res, layout.size), overflow, layout.ty))
            }
        }
    }

    pub fn unary_op(
        &self,
        un_op: mir::UnOp,
        val: &ImmTy<'tcx, M::Provenance>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        let (val, _overflow, ty) = self.overflowing_unary_op(un_op, val)?;
        Ok(ImmTy::from_scalar(val, self.layout_of(ty)?))
    }
}
