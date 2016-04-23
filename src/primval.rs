use rustc::mir::repr as mir;

use error::{EvalError, EvalResult};
use memory::Pointer;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrimVal {
    Bool(bool),
    I8(i8), I16(i16), I32(i32), I64(i64),
    U8(u8), U16(u16), U32(u32), U64(u64),

    AbstractPtr(Pointer),
    IntegerPtr(u64),
}

pub fn binary_op(bin_op: mir::BinOp, left: PrimVal, right: PrimVal) -> EvalResult<PrimVal> {
    use rustc::mir::repr::BinOp::*;
    use self::PrimVal::*;

    macro_rules! int_binops {
        ($v:ident, $l:ident, $r:ident) => ({
            match bin_op {
                Add    => $v($l + $r),
                Sub    => $v($l - $r),
                Mul    => $v($l * $r),
                Div    => $v($l / $r),
                Rem    => $v($l % $r),
                BitXor => $v($l ^ $r),
                BitAnd => $v($l & $r),
                BitOr  => $v($l | $r),

                // TODO(tsion): Can have differently-typed RHS.
                Shl => $v($l << $r),
                Shr => $v($l >> $r),

                Eq => Bool($l == $r),
                Ne => Bool($l != $r),
                Lt => Bool($l < $r),
                Le => Bool($l <= $r),
                Gt => Bool($l > $r),
                Ge => Bool($l >= $r),
            }
        })
    }

    fn unrelated_ptr_ops(bin_op: mir::BinOp) -> EvalResult<PrimVal> {
        use rustc::mir::repr::BinOp::*;
        match bin_op {
            Eq => Ok(Bool(false)),
            Ne => Ok(Bool(true)),
            Lt | Le | Gt | Ge => Err(EvalError::InvalidPointerMath),
            _ => unimplemented!(),
        }
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

        (Bool(l), Bool(r)) => {
            Bool(match bin_op {
                Eq => l == r,
                Ne => l != r,
                Lt => l < r,
                Le => l <= r,
                Gt => l > r,
                Ge => l >= r,
                _ => panic!("invalid binary operation on booleans: {:?}", bin_op),
            })
        }

        (IntegerPtr(l), IntegerPtr(r)) => int_binops!(IntegerPtr, l, r),

        (AbstractPtr(_), IntegerPtr(_)) | (IntegerPtr(_), AbstractPtr(_)) =>
            return unrelated_ptr_ops(bin_op),

        (AbstractPtr(l_ptr), AbstractPtr(r_ptr)) => {
            if l_ptr.alloc_id != r_ptr.alloc_id {
                return unrelated_ptr_ops(bin_op);
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
                _ => unimplemented!(),
            }
        }

        _ => unimplemented!(),
    };

    Ok(val)
}

pub fn unary_op(un_op: mir::UnOp, val: PrimVal) -> PrimVal {
    use rustc::mir::repr::UnOp::*;
    use self::PrimVal::*;
    match (un_op, val) {
        (Not, Bool(b)) => Bool(!b),
        (Not, I8(n))  => I8(!n),
        (Neg, I8(n))  => I8(-n),
        (Not, I16(n)) => I16(!n),
        (Neg, I16(n)) => I16(-n),
        (Not, I32(n)) => I32(!n),
        (Neg, I32(n)) => I32(-n),
        (Not, I64(n)) => I64(!n),
        (Neg, I64(n)) => I64(-n),
        (Not, U8(n))  => U8(!n),
        (Not, U16(n)) => U16(!n),
        (Not, U32(n)) => U32(!n),
        (Not, U64(n)) => U64(!n),
        _ => unimplemented!(),
    }
}
