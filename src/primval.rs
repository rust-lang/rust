#![allow(unknown_lints)]
#![allow(float_cmp)]

use rustc::mir::repr as mir;

use error::{EvalError, EvalResult};
use memory::Pointer;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrimVal {
    Bool(bool),
    I8(i8), I16(i16), I32(i32), I64(i64),
    U8(u8), U16(u16), U32(u32), U64(u64),

    AbstractPtr(Pointer),
    FnPtr(Pointer),
    IntegerPtr(u64),
    Char(char),

    F32(f32), F64(f64),
}

/// returns the result of the operation and whether the operation overflowed
pub fn binary_op<'tcx>(bin_op: mir::BinOp, left: PrimVal, right: PrimVal) -> EvalResult<'tcx, (PrimVal, bool)> {
    use rustc::mir::repr::BinOp::*;
    use self::PrimVal::*;

    macro_rules! overflow {
        ($v:ident, $v2:ident, $l:ident, $op:ident, $r:ident) => ({
            let (val, of) = $l.$op($r);
            if of {
                return Ok(($v(val), true));
            } else {
                $v(val)
            }
        })
    }

    macro_rules! int_binops {
        ($v:ident, $l:ident, $r:ident) => ({
            match bin_op {
                Add    => overflow!($v, $v, $l, overflowing_add, $r),
                Sub    => overflow!($v, $v, $l, overflowing_sub, $r),
                Mul    => overflow!($v, $v, $l, overflowing_mul, $r),
                Div    => overflow!($v, $v, $l, overflowing_div, $r),
                Rem    => overflow!($v, $v, $l, overflowing_rem, $r),
                BitXor => $v($l ^ $r),
                BitAnd => $v($l & $r),
                BitOr  => $v($l | $r),

                // these have already been handled
                Shl => unreachable!(),
                Shr => unreachable!(),

                Eq => Bool($l == $r),
                Ne => Bool($l != $r),
                Lt => Bool($l < $r),
                Le => Bool($l <= $r),
                Gt => Bool($l > $r),
                Ge => Bool($l >= $r),
            }
        })
    }

    macro_rules! float_binops {
        ($v:ident, $l:ident, $r:ident) => ({
            match bin_op {
                Add    => $v($l + $r),
                Sub    => $v($l - $r),
                Mul    => $v($l * $r),
                Div    => $v($l / $r),
                Rem    => $v($l % $r),

                // invalid float ops
                BitXor => unreachable!(),
                BitAnd => unreachable!(),
                BitOr  => unreachable!(),
                Shl => unreachable!(),
                Shr => unreachable!(),

                Eq => Bool($l == $r),
                Ne => Bool($l != $r),
                Lt => Bool($l < $r),
                Le => Bool($l <= $r),
                Gt => Bool($l > $r),
                Ge => Bool($l >= $r),
            }
        })
    }

    fn unrelated_ptr_ops<'tcx>(bin_op: mir::BinOp) -> EvalResult<'tcx, PrimVal> {
        use rustc::mir::repr::BinOp::*;
        match bin_op {
            Eq => Ok(Bool(false)),
            Ne => Ok(Bool(true)),
            Lt | Le | Gt | Ge => Err(EvalError::InvalidPointerMath),
            _ => unimplemented!(),
        }
    }

    match bin_op {
        // can have rhs with a different numeric type
        Shl | Shr => {
            // these numbers are the maximum number a bitshift rhs could possibly have
            // e.g. u16 can be bitshifted by 0..16, so masking with 0b1111 (16 - 1) will ensure we are in that range
            let type_bits: u32 = match left {
                I8(_) | U8(_) => 8,
                I16(_) | U16(_) => 16,
                I32(_) | U32(_) => 32,
                I64(_) | U64(_) => 64,
                _ => unreachable!(),
            };
            assert!(type_bits.is_power_of_two());
            // turn into `u32` because `overflowing_sh{l,r}` only take `u32`
            let r = match right {
                I8(i) => i as u32,
                I16(i) => i as u32,
                I32(i) => i as u32,
                I64(i) => i as u32,
                U8(i) => i as u32,
                U16(i) => i as u32,
                U32(i) => i as u32,
                U64(i) => i as u32,
                _ => panic!("bad MIR: bitshift rhs is not integral"),
            };
            // apply mask
            let r = r & (type_bits - 1);
            macro_rules! shift {
                ($v:ident, $l:ident, $r:ident) => ({
                    match bin_op {
                        Shl => overflow!($v, U32, $l, overflowing_shl, $r),
                        Shr => overflow!($v, U32, $l, overflowing_shr, $r),
                        _ => unreachable!(),
                    }
                })
            }
            let val = match left {
                I8(l) => shift!(I8, l, r),
                I16(l) => shift!(I16, l, r),
                I32(l) => shift!(I32, l, r),
                I64(l) => shift!(I64, l, r),
                U8(l) => shift!(U8, l, r),
                U16(l) => shift!(U16, l, r),
                U32(l) => shift!(U32, l, r),
                U64(l) => shift!(U64, l, r),
                _ => unreachable!(),
            };
            return Ok((val, false));
        },
        _ => {},
    }

    let val = match (left, right) {
        (I8(l),  I8(r))  => int_binops!(I8, l, r),
        (I16(l), I16(r)) => int_binops!(I16, l, r),
        (I32(l), I32(r)) => int_binops!(I32, l, r),
        (I64(l), I64(r)) => int_binops!(I64, l, r),
        (U8(l),  U8(r))  => int_binops!(U8, l, r),
        (U16(l), U16(r)) => int_binops!(U16, l, r),
        (U32(l), U32(r)) => int_binops!(U32, l, r),
        (U64(l), U64(r)) => int_binops!(U64, l, r),
        (F32(l), F32(r)) => float_binops!(F32, l, r),
        (F64(l), F64(r)) => float_binops!(F64, l, r),
        (Char(l), Char(r)) => match bin_op {
            Eq => Bool(l == r),
            Ne => Bool(l != r),
            Lt => Bool(l < r),
            Le => Bool(l <= r),
            Gt => Bool(l > r),
            Ge => Bool(l >= r),
            _ => panic!("invalid char op: {:?}", bin_op),
        },

        (Bool(l), Bool(r)) => {
            Bool(match bin_op {
                Eq => l == r,
                Ne => l != r,
                Lt => l < r,
                Le => l <= r,
                Gt => l > r,
                Ge => l >= r,
                BitOr => l | r,
                BitXor => l ^ r,
                BitAnd => l & r,
                Add | Sub | Mul | Div | Rem | Shl | Shr => return Err(EvalError::InvalidBoolOp(bin_op)),
            })
        }

        (IntegerPtr(l), IntegerPtr(r)) => int_binops!(IntegerPtr, l, r),

        (AbstractPtr(_), IntegerPtr(_)) |
        (IntegerPtr(_), AbstractPtr(_)) |
        (FnPtr(_), AbstractPtr(_)) |
        (AbstractPtr(_), FnPtr(_)) |
        (FnPtr(_), IntegerPtr(_)) |
        (IntegerPtr(_), FnPtr(_)) =>
            unrelated_ptr_ops(bin_op)?,

        (FnPtr(l_ptr), FnPtr(r_ptr)) => match bin_op {
            Eq => Bool(l_ptr == r_ptr),
            Ne => Bool(l_ptr != r_ptr),
            _ => return Err(EvalError::Unimplemented(format!("unimplemented fn ptr comparison: {:?}", bin_op))),
        },

        (AbstractPtr(l_ptr), AbstractPtr(r_ptr)) => {
            if l_ptr.alloc_id != r_ptr.alloc_id {
                return Ok((unrelated_ptr_ops(bin_op)?, false));
            }

            let l = l_ptr.offset;
            let r = r_ptr.offset;

            match bin_op {
                Eq => Bool(l == r),
                Ne => Bool(l != r),
                Lt => Bool(l < r),
                Le => Bool(l <= r),
                Gt => Bool(l > r),
                Ge => Bool(l >= r),
                _ => return Err(EvalError::Unimplemented(format!("unimplemented ptr op: {:?}", bin_op))),
            }
        }

        (l, r) => return Err(EvalError::Unimplemented(format!("unimplemented binary op: {:?}, {:?}, {:?}", l, r, bin_op))),
    };

    Ok((val, false))
}

pub fn unary_op<'tcx>(un_op: mir::UnOp, val: PrimVal) -> EvalResult<'tcx, PrimVal> {
    use rustc::mir::repr::UnOp::*;
    use self::PrimVal::*;
    match (un_op, val) {
        (Not, Bool(b)) => Ok(Bool(!b)),
        (Not, I8(n))  => Ok(I8(!n)),
        (Neg, I8(n))  => Ok(I8(-n)),
        (Not, I16(n)) => Ok(I16(!n)),
        (Neg, I16(n)) => Ok(I16(-n)),
        (Not, I32(n)) => Ok(I32(!n)),
        (Neg, I32(n)) => Ok(I32(-n)),
        (Not, I64(n)) => Ok(I64(!n)),
        (Neg, I64(n)) => Ok(I64(-n)),
        (Not, U8(n))  => Ok(U8(!n)),
        (Not, U16(n)) => Ok(U16(!n)),
        (Not, U32(n)) => Ok(U32(!n)),
        (Not, U64(n)) => Ok(U64(!n)),

        (Neg, F64(n)) => Ok(F64(-n)),
        (Neg, F32(n)) => Ok(F32(-n)),
        _ => Err(EvalError::Unimplemented(format!("unimplemented unary op: {:?}, {:?}", un_op, val))),
    }
}
