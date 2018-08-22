// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::mir;
use rustc::ty::{self, layout::{self, TyLayout}};
use syntax::ast::FloatTy;
use rustc_apfloat::ieee::{Double, Single};
use rustc_apfloat::Float;
use rustc::mir::interpret::{EvalResult, Scalar};

use super::{EvalContext, PlaceTy, Value, Machine, ValTy};


impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    /// Applies the binary operation `op` to the two operands and writes a tuple of the result
    /// and a boolean signifying the potential overflow to the destination.
    pub fn binop_with_overflow(
        &mut self,
        op: mir::BinOp,
        left: ValTy<'tcx>,
        right: ValTy<'tcx>,
        dest: PlaceTy<'tcx>,
    ) -> EvalResult<'tcx> {
        let (val, overflowed) = self.binary_op(op, left, right)?;
        let val = Value::ScalarPair(val.into(), Scalar::from_bool(overflowed).into());
        self.write_value(val, dest)
    }

    /// Applies the binary operation `op` to the arguments and writes the result to the
    /// destination.
    pub fn binop_ignore_overflow(
        &mut self,
        op: mir::BinOp,
        left: ValTy<'tcx>,
        right: ValTy<'tcx>,
        dest: PlaceTy<'tcx>,
    ) -> EvalResult<'tcx> {
        let (val, _overflowed) = self.binary_op(op, left, right)?;
        self.write_scalar(val, dest)
    }
}

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    /// Returns the result of the specified operation and whether it overflowed.
    pub fn binary_op(
        &self,
        bin_op: mir::BinOp,
        ValTy { value: left, layout: left_layout }: ValTy<'tcx>,
        ValTy { value: right, layout: right_layout }: ValTy<'tcx>,
    ) -> EvalResult<'tcx, (Scalar, bool)> {
        use rustc::mir::BinOp::*;

        let left = left.to_scalar()?;
        let right = right.to_scalar()?;

        let left_kind = match left_layout.abi {
            layout::Abi::Scalar(ref scalar) => scalar.value,
            _ => return err!(TypeNotPrimitive(left_layout.ty)),
        };
        let right_kind = match right_layout.abi {
            layout::Abi::Scalar(ref scalar) => scalar.value,
            _ => return err!(TypeNotPrimitive(right_layout.ty)),
        };
        trace!("Running binary op {:?}: {:?} ({:?}), {:?} ({:?})",
        bin_op, left, left_kind, right, right_kind);

        // I: Handle operations that support pointers
        if !left_kind.is_float() && !right_kind.is_float() {
            if let Some(handled) =
                M::try_ptr_op(self, bin_op, left, left_layout, right, right_layout)?
            {
                return Ok(handled);
            }
        }

        // II: From now on, everything must be bytes, no pointers
        let l = left.to_bits(left_layout.size)?;
        let r = right.to_bits(right_layout.size)?;

        // These ops can have an RHS with a different numeric type.
        if right_kind.is_int() && (bin_op == Shl || bin_op == Shr) {
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
            return Ok((Scalar::Bits {
                bits: truncated,
                size: size.bytes() as u8,
            }, oflo));
        }

        if left_kind != right_kind {
            let msg = format!(
                "unimplemented binary op {:?}: {:?} ({:?}), {:?} ({:?})",
                bin_op,
                left,
                left_kind,
                right,
                right_kind
            );
            return err!(Unimplemented(msg));
        }

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
                            return Ok((Scalar::Bits { bits: l, size: size.bytes() as u8 }, true));
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
                let result = result as u128;
                let truncated = self.truncate(result, left_layout);
                return Ok((Scalar::Bits {
                    bits: truncated,
                    size: size.bytes() as u8,
                }, oflo));
            }
        }

        if let ty::Float(fty) = left_layout.ty.sty {
            macro_rules! float_math {
                ($ty:path, $size:expr) => {{
                    let l = <$ty>::from_bits(l);
                    let r = <$ty>::from_bits(r);
                    let bitify = |res: ::rustc_apfloat::StatusAnd<$ty>| Scalar::Bits {
                        bits: res.value.to_bits(),
                        size: $size,
                    };
                    let val = match bin_op {
                        Eq => Scalar::from_bool(l == r),
                        Ne => Scalar::from_bool(l != r),
                        Lt => Scalar::from_bool(l < r),
                        Le => Scalar::from_bool(l <= r),
                        Gt => Scalar::from_bool(l > r),
                        Ge => Scalar::from_bool(l >= r),
                        Add => bitify(l + r),
                        Sub => bitify(l - r),
                        Mul => bitify(l * r),
                        Div => bitify(l / r),
                        Rem => bitify(l % r),
                        _ => bug!("invalid float op: `{:?}`", bin_op),
                    };
                    return Ok((val, false));
                }};
            }
            match fty {
                FloatTy::F32 => float_math!(Single, 4),
                FloatTy::F64 => float_math!(Double, 8),
            }
        }

        let size = left_layout.size.bytes() as u8;

        // only ints left
        let val = match bin_op {
            Eq => Scalar::from_bool(l == r),
            Ne => Scalar::from_bool(l != r),

            Lt => Scalar::from_bool(l < r),
            Le => Scalar::from_bool(l <= r),
            Gt => Scalar::from_bool(l > r),
            Ge => Scalar::from_bool(l >= r),

            BitOr => Scalar::Bits { bits: l | r, size },
            BitAnd => Scalar::Bits { bits: l & r, size },
            BitXor => Scalar::Bits { bits: l ^ r, size },

            Add | Sub | Mul | Rem | Div => {
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
                return Ok((Scalar::Bits {
                    bits: truncated,
                    size,
                }, oflo || truncated != result));
            }

            _ => {
                let msg = format!(
                    "unimplemented binary op {:?}: {:?} ({:?}), {:?} ({:?})",
                    bin_op,
                    left,
                    left_layout.ty,
                    right,
                    right_layout.ty,
                );
                return err!(Unimplemented(msg));
            }
        };

        Ok((val, false))
    }

    pub fn unary_op(
        &self,
        un_op: mir::UnOp,
        val: Scalar,
        layout: TyLayout<'tcx>,
    ) -> EvalResult<'tcx, Scalar> {
        use rustc::mir::UnOp::*;
        use rustc_apfloat::ieee::{Single, Double};
        use rustc_apfloat::Float;

        let size = layout.size;
        let bytes = val.to_bits(size)?;

        let result_bytes = match (un_op, &layout.ty.sty) {

            (Not, ty::Bool) => !val.to_bool()? as u128,

            (Not, _) => !bytes,

            (Neg, ty::Float(FloatTy::F32)) => Single::to_bits(-Single::from_bits(bytes)),
            (Neg, ty::Float(FloatTy::F64)) => Double::to_bits(-Double::from_bits(bytes)),

            (Neg, _) if bytes == (1 << (size.bits() - 1)) => return err!(OverflowNeg),
            (Neg, _) => (-(bytes as i128)) as u128,
        };

        Ok(Scalar::Bits {
            bits: self.truncate(result_bytes, layout),
            size: size.bytes() as u8,
        })
    }
}
