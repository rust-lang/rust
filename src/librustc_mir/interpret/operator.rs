use rustc::mir;
use rustc::ty::{self, layout::TyLayout};
use syntax::ast::FloatTy;
use rustc_apfloat::Float;
use rustc::mir::interpret::{InterpResult, Scalar};

use super::{InterpCx, PlaceTy, Immediate, Machine, ImmTy};


impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    /// Applies the binary operation `op` to the two operands and writes a tuple of the result
    /// and a boolean signifying the potential overflow to the destination.
    pub fn binop_with_overflow(
        &mut self,
        op: mir::BinOp,
        left: ImmTy<'tcx, M::PointerTag>,
        right: ImmTy<'tcx, M::PointerTag>,
        dest: PlaceTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx> {
        let (val, overflowed) = self.binary_op(op, left, right)?;
        let val = Immediate::ScalarPair(val.into(), Scalar::from_bool(overflowed).into());
        self.write_immediate(val, dest)
    }

    /// Applies the binary operation `op` to the arguments and writes the result to the
    /// destination.
    pub fn binop_ignore_overflow(
        &mut self,
        op: mir::BinOp,
        left: ImmTy<'tcx, M::PointerTag>,
        right: ImmTy<'tcx, M::PointerTag>,
        dest: PlaceTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx> {
        let (val, _overflowed) = self.binary_op(op, left, right)?;
        self.write_scalar(val, dest)
    }
}

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    fn binary_char_op(
        &self,
        bin_op: mir::BinOp,
        l: char,
        r: char,
    ) -> (Scalar<M::PointerTag>, bool) {
        use rustc::mir::BinOp::*;

        let res = match bin_op {
            Eq => l == r,
            Ne => l != r,
            Lt => l < r,
            Le => l <= r,
            Gt => l > r,
            Ge => l >= r,
            _ => bug!("Invalid operation on char: {:?}", bin_op),
        };
        return (Scalar::from_bool(res), false);
    }

    fn binary_bool_op(
        &self,
        bin_op: mir::BinOp,
        l: bool,
        r: bool,
    ) -> (Scalar<M::PointerTag>, bool) {
        use rustc::mir::BinOp::*;

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
            _ => bug!("Invalid operation on bool: {:?}", bin_op),
        };
        return (Scalar::from_bool(res), false);
    }

    fn binary_float_op<F: Float + Into<Scalar<M::PointerTag>>>(
        &self,
        bin_op: mir::BinOp,
        l: F,
        r: F,
    ) -> (Scalar<M::PointerTag>, bool) {
        use rustc::mir::BinOp::*;

        let val = match bin_op {
            Eq => Scalar::from_bool(l == r),
            Ne => Scalar::from_bool(l != r),
            Lt => Scalar::from_bool(l < r),
            Le => Scalar::from_bool(l <= r),
            Gt => Scalar::from_bool(l > r),
            Ge => Scalar::from_bool(l >= r),
            Add => (l + r).value.into(),
            Sub => (l - r).value.into(),
            Mul => (l * r).value.into(),
            Div => (l / r).value.into(),
            Rem => (l % r).value.into(),
            _ => bug!("invalid float op: `{:?}`", bin_op),
        };
        return (val, false);
    }

    fn binary_int_op(
        &self,
        bin_op: mir::BinOp,
        // passing in raw bits
        l: u128,
        left_layout: TyLayout<'tcx>,
        r: u128,
        right_layout: TyLayout<'tcx>,
    ) -> InterpResult<'tcx, (Scalar<M::PointerTag>, bool)> {
        use rustc::mir::BinOp::*;

        // Shift ops can have an RHS with a different numeric type.
        if bin_op == Shl || bin_op == Shr {
            let signed = left_layout.abi.is_signed();
            let mut oflo = (r as u32 as u128) != r;
            let mut r = r as u32;
            let size = left_layout.size;
            oflo |= r >= size.bits() as u32;
            if oflo {
                r %= size.bits() as u32;
            }
            let result = if signed {
                let l = self.sign_extend(l, left_layout) as i128;
                let result = match bin_op {
                    Shl => l << r,
                    Shr => l >> r,
                    _ => bug!("it has already been checked that this is a shift op"),
                };
                result as u128
            } else {
                match bin_op {
                    Shl => l << r,
                    Shr => l >> r,
                    _ => bug!("it has already been checked that this is a shift op"),
                }
            };
            let truncated = self.truncate(result, left_layout);
            return Ok((Scalar::from_uint(truncated, size), oflo));
        }

        // For the remaining ops, the types must be the same on both sides
        if left_layout.ty != right_layout.ty {
            let msg = format!(
                "unimplemented asymmetric binary op {:?}: {:?} ({:?}), {:?} ({:?})",
                bin_op,
                l,
                left_layout.ty,
                r,
                right_layout.ty
            );
            return err!(Unimplemented(msg));
        }

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
                return Ok((Scalar::from_bool(op(&l, &r)), false));
            }
            let op: Option<fn(i128, i128) -> (i128, bool)> = match bin_op {
                Div if r == 0 => return err!(DivisionByZero),
                Rem if r == 0 => return err!(RemainderByZero),
                Div => Some(i128::overflowing_div),
                Rem => Some(i128::overflowing_rem),
                Add => Some(i128::overflowing_add),
                Sub => Some(i128::overflowing_sub),
                Mul => Some(i128::overflowing_mul),
                _ => None,
            };
            if let Some(op) = op {
                let l128 = self.sign_extend(l, left_layout) as i128;
                let r = self.sign_extend(r, right_layout) as i128;
                let size = left_layout.size;
                match bin_op {
                    Rem | Div => {
                        // int_min / -1
                        if r == -1 && l == (1 << (size.bits() - 1)) {
                            return Ok((Scalar::from_uint(l, size), true));
                        }
                    },
                    _ => {},
                }
                trace!("{}, {}, {}", l, l128, r);
                let (result, mut oflo) = op(l128, r);
                trace!("{}, {}", result, oflo);
                if !oflo && size.bits() != 128 {
                    let max = 1 << (size.bits() - 1);
                    oflo = result >= max || result < -max;
                }
                // this may be out-of-bounds for the result type, so we have to truncate ourselves
                let result = result as u128;
                let truncated = self.truncate(result, left_layout);
                return Ok((Scalar::from_uint(truncated, size), oflo));
            }
        }

        let size = left_layout.size;

        // only ints left
        let val = match bin_op {
            Eq => Scalar::from_bool(l == r),
            Ne => Scalar::from_bool(l != r),

            Lt => Scalar::from_bool(l < r),
            Le => Scalar::from_bool(l <= r),
            Gt => Scalar::from_bool(l > r),
            Ge => Scalar::from_bool(l >= r),

            BitOr => Scalar::from_uint(l | r, size),
            BitAnd => Scalar::from_uint(l & r, size),
            BitXor => Scalar::from_uint(l ^ r, size),

            Add | Sub | Mul | Rem | Div => {
                debug_assert!(!left_layout.abi.is_signed());
                let op: fn(u128, u128) -> (u128, bool) = match bin_op {
                    Add => u128::overflowing_add,
                    Sub => u128::overflowing_sub,
                    Mul => u128::overflowing_mul,
                    Div if r == 0 => return err!(DivisionByZero),
                    Rem if r == 0 => return err!(RemainderByZero),
                    Div => u128::overflowing_div,
                    Rem => u128::overflowing_rem,
                    _ => bug!(),
                };
                let (result, oflo) = op(l, r);
                let truncated = self.truncate(result, left_layout);
                return Ok((Scalar::from_uint(truncated, size), oflo || truncated != result));
            }

            _ => {
                let msg = format!(
                    "unimplemented binary op {:?}: {:?}, {:?} (both {:?})",
                    bin_op,
                    l,
                    r,
                    right_layout.ty,
                );
                return err!(Unimplemented(msg));
            }
        };

        Ok((val, false))
    }

    /// Returns the result of the specified operation and whether it overflowed.
    #[inline]
    pub fn binary_op(
        &self,
        bin_op: mir::BinOp,
        left: ImmTy<'tcx, M::PointerTag>,
        right: ImmTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx, (Scalar<M::PointerTag>, bool)> {
        trace!("Running binary op {:?}: {:?} ({:?}), {:?} ({:?})",
            bin_op, *left, left.layout.ty, *right, right.layout.ty);

        match left.layout.ty.sty {
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
                let left = left.to_scalar()?;
                let right = right.to_scalar()?;
                Ok(match fty {
                    FloatTy::F32 => self.binary_float_op(bin_op, left.to_f32()?, right.to_f32()?),
                    FloatTy::F64 => self.binary_float_op(bin_op, left.to_f64()?, right.to_f64()?),
                })
            }
            _ => {
                // Must be integer(-like) types.  Don't forget about == on fn pointers.
                assert!(
                    left.layout.ty.is_integral()   ||
                    left.layout.ty.is_unsafe_ptr() || left.layout.ty.is_fn_ptr(),
                    "Unexpected LHS type {:?} for BinOp {:?}", left.layout.ty, bin_op);
                assert!(
                    right.layout.ty.is_integral()   ||
                    right.layout.ty.is_unsafe_ptr() || right.layout.ty.is_fn_ptr(),
                    "Unexpected RHS type {:?} for BinOp {:?}", right.layout.ty, bin_op);

                // Handle operations that support pointer values
                if left.to_scalar_ptr()?.is_ptr() ||
                    right.to_scalar_ptr()?.is_ptr() ||
                    bin_op == mir::BinOp::Offset
                {
                    return M::ptr_op(self, bin_op, left, right);
                }

                // Everything else only works with "proper" bits
                let l = left.to_bits().expect("we checked is_ptr");
                let r = right.to_bits().expect("we checked is_ptr");
                self.binary_int_op(bin_op, l, left.layout, r, right.layout)
            }
        }
    }

    pub fn unary_op(
        &self,
        un_op: mir::UnOp,
        val: ImmTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx, Scalar<M::PointerTag>> {
        use rustc::mir::UnOp::*;

        let layout = val.layout;
        let val = val.to_scalar()?;
        trace!("Running unary op {:?}: {:?} ({:?})", un_op, val, layout.ty);

        match layout.ty.sty {
            ty::Bool => {
                let val = val.to_bool()?;
                let res = match un_op {
                    Not => !val,
                    _ => bug!("Invalid bool op {:?}", un_op)
                };
                Ok(Scalar::from_bool(res))
            }
            ty::Float(fty) => {
                let res = match (un_op, fty) {
                    (Neg, FloatTy::F32) => Scalar::from_f32(-val.to_f32()?),
                    (Neg, FloatTy::F64) => Scalar::from_f64(-val.to_f64()?),
                    _ => bug!("Invalid float op {:?}", un_op)
                };
                Ok(res)
            }
            _ => {
                assert!(layout.ty.is_integral());
                let val = self.force_bits(val, layout.size)?;
                let res = match un_op {
                    Not => !val,
                    Neg => {
                        assert!(layout.abi.is_signed());
                        (-(val as i128)) as u128
                    }
                };
                // res needs tuncating
                Ok(Scalar::from_uint(self.truncate(res, layout), layout.size))
            }
        }
    }
}
