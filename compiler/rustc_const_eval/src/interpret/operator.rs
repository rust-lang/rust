use either::Either;
use rustc_abi::Size;
use rustc_apfloat::{Float, FloatConvert};
use rustc_middle::mir::NullOp;
use rustc_middle::mir::interpret::{InterpResult, PointerArithmetic, Scalar};
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{self, FloatTy, ScalarInt, Ty};
use rustc_middle::{bug, mir, span_bug};
use rustc_span::sym;
use tracing::trace;

use super::{ImmTy, InterpCx, Machine, MemPlaceMeta, interp_ok, throw_ub};

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
        let adjust_nan = |f: F| -> F { self.adjust_nan(f, &[l, r]) };

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
        let l_signed = || l.to_int(left.layout.size);
        let l_unsigned = || l.to_uint(left.layout.size);
        let r_signed = || r.to_int(right.layout.size);
        let r_unsigned = || r.to_uint(right.layout.size);

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
            let l_bits = left.layout.size.bits();
            // Compute the equivalent shift modulo `size` that is in the range `0..size`. (This is
            // the one MIR operator that does *not* directly map to a single LLVM operation.)
            let (shift_amount, overflow) = if right.layout.backend_repr.is_signed() {
                let shift_amount = r_signed();
                let rem = shift_amount.rem_euclid(l_bits.into());
                // `rem` is guaranteed positive, so the `unwrap` cannot fail
                (u128::try_from(rem).unwrap(), rem != shift_amount)
            } else {
                let shift_amount = r_unsigned();
                let rem = shift_amount.rem_euclid(l_bits.into());
                (rem, rem != shift_amount)
            };
            let shift_amount = u32::try_from(shift_amount).unwrap(); // we brought this in the range `0..size` so this will always fit
            // Compute the shifted result.
            let result = if left.layout.backend_repr.is_signed() {
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
                    shift_amount: if right.layout.backend_repr.is_signed() {
                        Either::Right(r_signed())
                    } else {
                        Either::Left(r_unsigned())
                    }
                });
            }

            return interp_ok(ImmTy::from_scalar_int(result, left.layout));
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
        if left.layout.backend_repr.is_signed() {
            let op: Option<fn(&i128, &i128) -> bool> = match bin_op {
                Lt => Some(i128::lt),
                Le => Some(i128::le),
                Gt => Some(i128::gt),
                Ge => Some(i128::ge),
                _ => None,
            };
            if let Some(op) = op {
                return interp_ok(ImmTy::from_bool(op(&l_signed(), &r_signed()), *self.tcx));
            }
            if bin_op == Cmp {
                return interp_ok(self.three_way_compare(l_signed(), r_signed()));
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
                return interp_ok(if with_overflow {
                    let overflow = ImmTy::from_bool(overflow, *self.tcx);
                    ImmTy::from_pair(res, overflow, self)
                } else {
                    res
                });
            }
        }
        // From here on it's okay to treat everything as unsigned.
        let l = l_unsigned();
        let r = r_unsigned();

        if bin_op == Cmp {
            return interp_ok(self.three_way_compare(l, r));
        }

        interp_ok(match bin_op {
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
                assert!(!left.layout.backend_repr.is_signed());
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
                    ImmTy::from_pair(res, overflow, self)
                } else {
                    res
                }
            }
        })
    }

    /// Computes the total size of this access, `count * elem_size`,
    /// checking for overflow beyond isize::MAX.
    pub fn compute_size_in_bytes(&self, elem_size: Size, count: u64) -> Option<Size> {
        // `checked_mul` applies `u64` limits independent of the target pointer size... but the
        // subsequent check for `max_size_of_val` means we also handle 32bit targets correctly.
        // (We cannot use `Size::checked_mul` as that enforces `obj_size_bound` as the limit, which
        // would be wrong here.)
        elem_size
            .bytes()
            .checked_mul(count)
            .map(Size::from_bytes)
            .filter(|&total| total <= self.max_size_of_val())
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
                let pointee_ty = left.layout.ty.builtin_deref(true).unwrap();
                let pointee_layout = self.layout_of(pointee_ty)?;
                assert!(pointee_layout.is_sized());

                // The size always fits in `i64` as it can be at most `isize::MAX`.
                let pointee_size = i64::try_from(pointee_layout.size.bytes()).unwrap();
                // This uses the same type as `right`, which can be `isize` or `usize`.
                // `pointee_size` is guaranteed to fit into both types.
                let pointee_size = ImmTy::from_int(pointee_size, right.layout);
                // Multiply element size and element count.
                let (val, overflowed) = self
                    .binary_op(mir::BinOp::MulWithOverflow, right, &pointee_size)?
                    .to_scalar_pair();
                // This must not overflow.
                if overflowed.to_bool()? {
                    throw_ub!(PointerArithOverflow)
                }

                let offset_bytes = val.to_target_isize(self)?;
                if !right.layout.backend_repr.is_signed() && offset_bytes < 0 {
                    // We were supposed to do an unsigned offset but the result is negative -- this
                    // can only mean that the cast wrapped around.
                    throw_ub!(PointerArithOverflow)
                }
                let offset_ptr = self.ptr_offset_inbounds(ptr, offset_bytes)?;
                interp_ok(ImmTy::from_scalar(
                    Scalar::from_maybe_pointer(offset_ptr, self),
                    left.layout,
                ))
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
            bin_op, *left, left.layout.ty, *right, right.layout.ty
        );

        match left.layout.ty.kind() {
            ty::Char => {
                assert_eq!(left.layout.ty, right.layout.ty);
                let left = left.to_scalar();
                let right = right.to_scalar();
                interp_ok(self.binary_char_op(bin_op, left.to_char()?, right.to_char()?))
            }
            ty::Bool => {
                assert_eq!(left.layout.ty, right.layout.ty);
                let left = left.to_scalar();
                let right = right.to_scalar();
                interp_ok(self.binary_bool_op(bin_op, left.to_bool()?, right.to_bool()?))
            }
            ty::Float(fty) => {
                assert_eq!(left.layout.ty, right.layout.ty);
                let layout = left.layout;
                let left = left.to_scalar();
                let right = right.to_scalar();
                interp_ok(match fty {
                    FloatTy::F16 => {
                        self.binary_float_op(bin_op, layout, left.to_f16()?, right.to_f16()?)
                    }
                    FloatTy::F32 => {
                        self.binary_float_op(bin_op, layout, left.to_f32()?, right.to_f32()?)
                    }
                    FloatTy::F64 => {
                        self.binary_float_op(bin_op, layout, left.to_f64()?, right.to_f64()?)
                    }
                    FloatTy::F128 => {
                        self.binary_float_op(bin_op, layout, left.to_f128()?, right.to_f128()?)
                    }
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
        trace!("Running unary op {:?}: {:?} ({})", un_op, val, layout.ty);

        match layout.ty.kind() {
            ty::Bool => {
                let val = val.to_scalar();
                let val = val.to_bool()?;
                let res = match un_op {
                    Not => !val,
                    _ => span_bug!(self.cur_span(), "Invalid bool op {:?}", un_op),
                };
                interp_ok(ImmTy::from_bool(res, *self.tcx))
            }
            ty::Float(fty) => {
                let val = val.to_scalar();
                if un_op != Neg {
                    span_bug!(self.cur_span(), "Invalid float op {:?}", un_op);
                }

                // No NaN adjustment here, `-` is a bitwise operation!
                let res = match fty {
                    FloatTy::F16 => Scalar::from_f16(-val.to_f16()?),
                    FloatTy::F32 => Scalar::from_f32(-val.to_f32()?),
                    FloatTy::F64 => Scalar::from_f64(-val.to_f64()?),
                    FloatTy::F128 => Scalar::from_f128(-val.to_f128()?),
                };
                interp_ok(ImmTy::from_scalar(res, layout))
            }
            ty::Int(..) => {
                let val = val.to_scalar().to_int(layout.size)?;
                let res = match un_op {
                    Not => !val,
                    Neg => val.wrapping_neg(),
                    _ => span_bug!(self.cur_span(), "Invalid integer op {:?}", un_op),
                };
                let res = ScalarInt::truncate_from_int(res, layout.size).0;
                interp_ok(ImmTy::from_scalar(res.into(), layout))
            }
            ty::Uint(..) => {
                let val = val.to_scalar().to_uint(layout.size)?;
                let res = match un_op {
                    Not => !val,
                    _ => span_bug!(self.cur_span(), "Invalid unsigned integer op {:?}", un_op),
                };
                let res = ScalarInt::truncate_from_uint(res, layout.size).0;
                interp_ok(ImmTy::from_scalar(res.into(), layout))
            }
            ty::RawPtr(..) | ty::Ref(..) => {
                assert_eq!(un_op, PtrMetadata);
                let (_, meta) = val.to_scalar_and_meta();
                interp_ok(match meta {
                    MemPlaceMeta::Meta(scalar) => {
                        let ty = un_op.ty(*self.tcx, val.layout.ty);
                        let layout = self.layout_of(ty)?;
                        ImmTy::from_scalar(scalar, layout)
                    }
                    MemPlaceMeta::None => {
                        let unit_layout = self.layout_of(self.tcx.types.unit)?;
                        ImmTy::uninit(unit_layout)
                    }
                })
            }
            _ => {
                bug!("Unexpected unary op argument {val:?}")
            }
        }
    }

    pub fn nullary_op(
        &self,
        null_op: NullOp<'tcx>,
        arg_ty: Ty<'tcx>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::Provenance>> {
        use rustc_middle::mir::NullOp::*;

        let layout = self.layout_of(arg_ty)?;
        let usize_layout = || self.layout_of(self.tcx.types.usize).unwrap();

        interp_ok(match null_op {
            SizeOf => {
                if !layout.is_sized() {
                    span_bug!(self.cur_span(), "unsized type for `NullaryOp::SizeOf`");
                }
                let val = layout.size.bytes();
                ImmTy::from_uint(val, usize_layout())
            }
            AlignOf => {
                if !layout.is_sized() {
                    span_bug!(self.cur_span(), "unsized type for `NullaryOp::AlignOf`");
                }
                let val = layout.align.abi.bytes();
                ImmTy::from_uint(val, usize_layout())
            }
            OffsetOf(fields) => {
                let val =
                    self.tcx.offset_of_subfield(self.typing_env, layout, fields.iter()).bytes();
                ImmTy::from_uint(val, usize_layout())
            }
            UbChecks => ImmTy::from_bool(M::ub_checks(self)?, *self.tcx),
            ContractChecks => ImmTy::from_bool(M::contract_checks(self)?, *self.tcx),
        })
    }
}
