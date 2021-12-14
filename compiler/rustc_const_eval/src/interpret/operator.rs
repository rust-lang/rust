use std::convert::TryFrom;

use rustc_apfloat::Float;
use rustc_middle::mir;
use rustc_middle::mir::interpret::{InterpResult, Scalar};
use rustc_middle::ty::layout::{LayoutOf, TyAndLayout};
use rustc_middle::ty::{self, FloatTy, Ty};

use super::{ImmTy, Immediate, InterpCx, Machine, PlaceTy};

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    /// Applies the binary operation `op` to the two operands and writes a tuple of the result
    /// and a boolean signifying the potential overflow to the destination.
    pub fn binop_with_overflow(
        &mut self,
        op: mir::BinOp,
        left: &ImmTy<'tcx, M::PointerTag>,
        right: &ImmTy<'tcx, M::PointerTag>,
        dest: &PlaceTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx> {
        let (val, overflowed, ty) = self.overflowing_binary_op(op, &left, &right)?;
        debug_assert_eq!(
            self.tcx.intern_tup(&[ty, self.tcx.types.bool]),
            dest.layout.ty,
            "type mismatch for result of {:?}",
            op,
        );
        let val = Immediate::ScalarPair(val.into(), Scalar::from_bool(overflowed).into());
        self.write_immediate(val, dest)
    }

    /// Applies the binary operation `op` to the arguments and writes the result to the
    /// destination.
    pub fn binop_ignore_overflow(
        &mut self,
        op: mir::BinOp,
        left: &ImmTy<'tcx, M::PointerTag>,
        right: &ImmTy<'tcx, M::PointerTag>,
        dest: &PlaceTy<'tcx, M::PointerTag>,
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
    ) -> (Scalar<M::PointerTag>, bool, Ty<'tcx>) {
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
    ) -> (Scalar<M::PointerTag>, bool, Ty<'tcx>) {
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

    fn binary_float_op<F: Float + Into<Scalar<M::PointerTag>>>(
        &self,
        bin_op: mir::BinOp,
        ty: Ty<'tcx>,
        l: F,
        r: F,
    ) -> (Scalar<M::PointerTag>, bool, Ty<'tcx>) {
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
    ) -> InterpResult<'tcx, (Scalar<M::PointerTag>, bool, Ty<'tcx>)> {
        use rustc_middle::mir::BinOp::*;

        // Shift ops can have an RHS with a different numeric type.
        if bin_op == Shl || bin_op == Shr {
            let signed = left_layout.abi.is_signed();
            let size = u128::from(left_layout.size.bits());
            let overflow = r >= size;
            // The shift offset is implicitly masked to the type size, to make sure this operation
            // is always defined. This is the one MIR operator that does *not* directly map to a
            // single LLVM operation. See
            // <https://github.com/rust-lang/rust/blob/a3b9405ae7bb6ab4e8103b414e75c44598a10fd2/compiler/rustc_codegen_ssa/src/common.rs#L131-L158>
            // for the corresponding truncation in our codegen backends.
            let r = r % size;
            let r = u32::try_from(r).unwrap(); // we masked so this will always fit
            let result = if signed {
                let l = self.sign_extend(l, left_layout) as i128;
                let result = match bin_op {
                    Shl => l.checked_shl(r).unwrap(),
                    Shr => l.checked_shr(r).unwrap(),
                    _ => bug!("it has already been checked that this is a shift op"),
                };
                result as u128
            } else {
                match bin_op {
                    Shl => l.checked_shl(r).unwrap(),
                    Shr => l.checked_shr(r).unwrap(),
                    _ => bug!("it has already been checked that this is a shift op"),
                }
            };
            let truncated = self.truncate(result, left_layout);
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
                Add => Some(i128::overflowing_add),
                Sub => Some(i128::overflowing_sub),
                Mul => Some(i128::overflowing_mul),
                _ => None,
            };
            if let Some(op) = op {
                let r = self.sign_extend(r, right_layout) as i128;
                // We need a special check for overflowing remainder:
                // "int_min % -1" overflows and returns 0, but after casting things to a larger int
                // type it does *not* overflow nor give an unrepresentable result!
                if bin_op == Rem {
                    if r == -1 && l == (1 << (size.bits() - 1)) {
                        return Ok((Scalar::from_int(0, size), true, left_layout.ty));
                    }
                }
                let l = self.sign_extend(l, left_layout) as i128;

                let (result, oflo) = op(l, r);
                // This may be out-of-bounds for the result type, so we have to truncate ourselves.
                // If that truncation loses any information, we have an overflow.
                let result = result as u128;
                let truncated = self.truncate(result, left_layout);
                return Ok((
                    Scalar::from_uint(truncated, size),
                    oflo || self.sign_extend(truncated, left_layout) != result,
                    left_layout.ty,
                ));
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

            Add | Sub | Mul | Rem | Div => {
                assert!(!left_layout.abi.is_signed());
                let op: fn(u128, u128) -> (u128, bool) = match bin_op {
                    Add => u128::overflowing_add,
                    Sub => u128::overflowing_sub,
                    Mul => u128::overflowing_mul,
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
                return Ok((
                    Scalar::from_uint(truncated, size),
                    oflo || truncated != result,
                    left_layout.ty,
                ));
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

    /// Returns the result of the specified operation, whether it overflowed, and
    /// the result type.
    pub fn overflowing_binary_op(
        &self,
        bin_op: mir::BinOp,
        left: &ImmTy<'tcx, M::PointerTag>,
        right: &ImmTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx, (Scalar<M::PointerTag>, bool, Ty<'tcx>)> {
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
                let left = left.to_scalar()?;
                let right = right.to_scalar()?;
                Ok(self.binary_char_op(bin_op, left.to_char()?, right.to_char()?))
            }
            ty::Bool => {
                assert_eq!(left.layout.ty, right.layout.ty);
                let left = left.to_scalar()?;
                let right = right.to_scalar()?;
                Ok(self.binary_bool_op(bin_op, left.to_bool()?, right.to_bool()?))
            }
            ty::Float(fty) => {
                assert_eq!(left.layout.ty, right.layout.ty);
                let ty = left.layout.ty;
                let left = left.to_scalar()?;
                let right = right.to_scalar()?;
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

                let l = left.to_scalar()?.to_bits(left.layout.size)?;
                let r = right.to_scalar()?.to_bits(right.layout.size)?;
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

                M::binary_ptr_op(self, bin_op, left, right)
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
        left: &ImmTy<'tcx, M::PointerTag>,
        right: &ImmTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::PointerTag>> {
        let (val, _overflow, ty) = self.overflowing_binary_op(bin_op, left, right)?;
        Ok(ImmTy::from_scalar(val, self.layout_of(ty)?))
    }

    /// Returns the result of the specified operation, whether it overflowed, and
    /// the result type.
    pub fn overflowing_unary_op(
        &self,
        un_op: mir::UnOp,
        val: &ImmTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx, (Scalar<M::PointerTag>, bool, Ty<'tcx>)> {
        use rustc_middle::mir::UnOp::*;

        let layout = val.layout;
        let val = val.to_scalar()?;
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
        val: &ImmTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::PointerTag>> {
        let (val, _overflow, ty) = self.overflowing_unary_op(un_op, val)?;
        Ok(ImmTy::from_scalar(val, self.layout_of(ty)?))
    }
}
