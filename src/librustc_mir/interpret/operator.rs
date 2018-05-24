use rustc::mir;
use rustc::ty::{self, Ty, layout};
use syntax::ast::FloatTy;
use rustc::ty::layout::LayoutOf;
use rustc_apfloat::ieee::{Double, Single};
use rustc_apfloat::Float;

use super::{EvalContext, Place, Machine, ValTy};

use rustc::mir::interpret::{EvalResult, Scalar, Value};

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    fn binop_with_overflow(
        &mut self,
        op: mir::BinOp,
        left: ValTy<'tcx>,
        right: ValTy<'tcx>,
    ) -> EvalResult<'tcx, (Scalar, bool)> {
        let left_val = self.value_to_scalar(left)?;
        let right_val = self.value_to_scalar(right)?;
        self.binary_op(op, left_val, left.ty, right_val, right.ty)
    }

    /// Applies the binary operation `op` to the two operands and writes a tuple of the result
    /// and a boolean signifying the potential overflow to the destination.
    pub fn intrinsic_with_overflow(
        &mut self,
        op: mir::BinOp,
        left: ValTy<'tcx>,
        right: ValTy<'tcx>,
        dest: Place,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx> {
        let (val, overflowed) = self.binop_with_overflow(op, left, right)?;
        let val = Value::ScalarPair(val, Scalar::from_bool(overflowed));
        let valty = ValTy {
            value: val,
            ty: dest_ty,
        };
        self.write_value(valty, dest)
    }

    /// Applies the binary operation `op` to the arguments and writes the result to the
    /// destination. Returns `true` if the operation overflowed.
    pub fn intrinsic_overflowing(
        &mut self,
        op: mir::BinOp,
        left: ValTy<'tcx>,
        right: ValTy<'tcx>,
        dest: Place,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, bool> {
        let (val, overflowed) = self.binop_with_overflow(op, left, right)?;
        self.write_scalar(dest, val, dest_ty)?;
        Ok(overflowed)
    }
}

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    /// Returns the result of the specified operation and whether it overflowed.
    pub fn binary_op(
        &self,
        bin_op: mir::BinOp,
        left: Scalar,
        left_ty: Ty<'tcx>,
        right: Scalar,
        right_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, (Scalar, bool)> {
        use rustc::mir::BinOp::*;

        let left_layout = self.layout_of(left_ty)?;
        let right_layout = self.layout_of(right_ty)?;

        let left_kind = match left_layout.abi {
            layout::Abi::Scalar(ref scalar) => scalar.value,
            _ => return err!(TypeNotPrimitive(left_ty)),
        };
        let right_kind = match right_layout.abi {
            layout::Abi::Scalar(ref scalar) => scalar.value,
            _ => return err!(TypeNotPrimitive(right_ty)),
        };
        trace!("Running binary op {:?}: {:?} ({:?}), {:?} ({:?})", bin_op, left, left_kind, right, right_kind);

        // I: Handle operations that support pointers
        if !left_kind.is_float() && !right_kind.is_float() {
            if let Some(handled) = M::try_ptr_op(self, bin_op, left, left_ty, right, right_ty)? {
                return Ok(handled);
            }
        }

        // II: From now on, everything must be bytes, no pointers
        let l = left.to_bits(left_layout.size)?;
        let r = right.to_bits(right_layout.size)?;

        // These ops can have an RHS with a different numeric type.
        if right_kind.is_int() && (bin_op == Shl || bin_op == Shr) {
            let signed = left_layout.abi.is_signed();
            let mut r = r as u32;
            let size = left_layout.size.bits() as u32;
            let oflo = r >= size;
            if oflo {
                r %= size;
            }
            let result = if signed {
                let l = self.sign_extend(l, left_ty)? as i128;
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
            let truncated = self.truncate(result, left_ty)?;
            return Ok((Scalar::Bits {
                bits: truncated,
                defined: size as u8,
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
                let l = self.sign_extend(l, left_ty)? as i128;
                let r = self.sign_extend(r, right_ty)? as i128;
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
                let l128 = self.sign_extend(l, left_ty)? as i128;
                let r = self.sign_extend(r, right_ty)? as i128;
                let size = left_layout.size.bits();
                match bin_op {
                    Rem | Div => {
                        // int_min / -1
                        if r == -1 && l == (1 << (size - 1)) {
                            return Ok((Scalar::Bits { bits: l, defined: size as u8 }, true));
                        }
                    },
                    _ => {},
                }
                trace!("{}, {}, {}", l, l128, r);
                let (result, mut oflo) = op(l128, r);
                trace!("{}, {}", result, oflo);
                if !oflo && size != 128 {
                    let max = 1 << (size - 1);
                    oflo = result >= max || result < -max;
                }
                let result = result as u128;
                let truncated = self.truncate(result, left_ty)?;
                return Ok((Scalar::Bits {
                    bits: truncated,
                    defined: size as u8,
                }, oflo));
            }
        }

        if let ty::TyFloat(fty) = left_ty.sty {
            macro_rules! float_math {
                ($ty:path, $bitsize:expr) => {{
                    let l = <$ty>::from_bits(l);
                    let r = <$ty>::from_bits(r);
                    let bitify = |res: ::rustc_apfloat::StatusAnd<$ty>| Scalar::Bits {
                        bits: res.value.to_bits(),
                        defined: $bitsize,
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
                FloatTy::F32 => float_math!(Single, 32),
                FloatTy::F64 => float_math!(Double, 64),
            }
        }

        let bit_width = self.layout_of(left_ty).unwrap().size.bits() as u8;

        // only ints left
        let val = match bin_op {
            Eq => Scalar::from_bool(l == r),
            Ne => Scalar::from_bool(l != r),

            Lt => Scalar::from_bool(l < r),
            Le => Scalar::from_bool(l <= r),
            Gt => Scalar::from_bool(l > r),
            Ge => Scalar::from_bool(l >= r),

            BitOr => Scalar::Bits { bits: l | r, defined: bit_width },
            BitAnd => Scalar::Bits { bits: l & r, defined: bit_width },
            BitXor => Scalar::Bits { bits: l ^ r, defined: bit_width },

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
                let truncated = self.truncate(result, left_ty)?;
                return Ok((Scalar::Bits {
                    bits: truncated,
                    defined: bit_width,
                }, oflo || truncated != result));
            }

            _ => {
                let msg = format!(
                    "unimplemented binary op {:?}: {:?} ({:?}), {:?} ({:?})",
                    bin_op,
                    left,
                    left_ty,
                    right,
                    right_ty,
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
        ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, Scalar> {
        use rustc::mir::UnOp::*;
        use rustc_apfloat::ieee::{Single, Double};
        use rustc_apfloat::Float;

        let size = self.layout_of(ty)?.size;
        let bytes = val.to_bits(size)?;
        let size = size.bits();

        let result_bytes = match (un_op, &ty.sty) {

            (Not, ty::TyBool) => !val.to_bool()? as u128,

            (Not, _) => !bytes,

            (Neg, ty::TyFloat(FloatTy::F32)) => Single::to_bits(-Single::from_bits(bytes)),
            (Neg, ty::TyFloat(FloatTy::F64)) => Double::to_bits(-Double::from_bits(bytes)),

            (Neg, _) if bytes == (1 << (size - 1)) => return err!(OverflowNeg),
            (Neg, _) => (-(bytes as i128)) as u128,
        };

        Ok(Scalar::Bits {
            bits: self.truncate(result_bytes, ty)?,
            defined: size as u8,
        })
    }
}
