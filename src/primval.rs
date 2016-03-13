use rustc::mir::repr as mir;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrimVal {
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
}

pub fn binary_op(bin_op: mir::BinOp, left: PrimVal, right: PrimVal) -> PrimVal {
    macro_rules! int_binops {
        ($v:ident, $l:ident, $r:ident) => ({
            use rustc::mir::repr::BinOp::*;
            use self::PrimVal::*;
            match bin_op {
                Add => $v($l + $r),
                Sub => $v($l - $r),
                Mul => $v($l * $r),
                Div => $v($l / $r),
                Rem => $v($l % $r),
                BitXor => $v($l ^ $r),
                BitAnd => $v($l & $r),
                BitOr => $v($l | $r),

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

    use self::PrimVal::*;
    match (left, right) {
        (I8(l), I8(r)) => int_binops!(I8, l, r),
        (I16(l), I16(r)) => int_binops!(I16, l, r),
        (I32(l), I32(r)) => int_binops!(I32, l, r),
        (I64(l), I64(r)) => int_binops!(I64, l, r),
        _ => unimplemented!(),
    }
}

pub fn unary_op(un_op: mir::UnOp, val: PrimVal) -> PrimVal {
    use rustc::mir::repr::UnOp::*;
    use self::PrimVal::*;
    match (un_op, val) {
        (Not, Bool(b)) => Bool(!b),
        (Not, I8(n)) => I8(!n),
        (Neg, I8(n)) => I8(-n),
        (Not, I16(n)) => I16(!n),
        (Neg, I16(n)) => I16(-n),
        (Not, I32(n)) => I32(!n),
        (Neg, I32(n)) => I32(-n),
        (Not, I64(n)) => I64(!n),
        (Neg, I64(n)) => I64(-n),
        _ => unimplemented!(),
    }
}
