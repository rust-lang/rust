use rustc::mir;
use rustc::ty::{self, Ty};

use error::{EvalError, EvalResult};
use eval_context::EvalContext;
use lvalue::Lvalue;
use memory::Pointer;
use value::{
    PrimVal,
    PrimValKind,
    Value,
    bytes_to_f32,
    bytes_to_f64,
    f32_to_bytes,
    f64_to_bytes,
    bytes_to_bool,
};

impl<'a, 'tcx> EvalContext<'a, 'tcx> {
    fn binop_with_overflow(
        &mut self,
        op: mir::BinOp,
        left: &mir::Operand<'tcx>,
        right: &mir::Operand<'tcx>,
    ) -> EvalResult<'tcx, (PrimVal, bool)> {
        let left_ty    = self.operand_ty(left);
        let right_ty   = self.operand_ty(right);
        let left_val   = self.eval_operand_to_primval(left)?;
        let right_val  = self.eval_operand_to_primval(right)?;
        self.binary_op(op, left_val, left_ty, right_val, right_ty)
    }

    /// Applies the binary operation `op` to the two operands and writes a tuple of the result
    /// and a boolean signifying the potential overflow to the destination.
    pub(super) fn intrinsic_with_overflow(
        &mut self,
        op: mir::BinOp,
        left: &mir::Operand<'tcx>,
        right: &mir::Operand<'tcx>,
        dest: Lvalue<'tcx>,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx> {
        let (val, overflowed) = self.binop_with_overflow(op, left, right)?;
        let val = Value::ByValPair(val, PrimVal::from_bool(overflowed));
        self.write_value(val, dest, dest_ty)
    }

    /// Applies the binary operation `op` to the arguments and writes the result to the
    /// destination. Returns `true` if the operation overflowed.
    pub(super) fn intrinsic_overflowing(
        &mut self,
        op: mir::BinOp,
        left: &mir::Operand<'tcx>,
        right: &mir::Operand<'tcx>,
        dest: Lvalue<'tcx>,
        dest_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, bool> {
        let (val, overflowed) = self.binop_with_overflow(op, left, right)?;
        self.write_primval(dest, val, dest_ty)?;
        Ok(overflowed)
    }
}

macro_rules! overflow {
    ($op:ident, $l:expr, $r:expr) => ({
        let (val, overflowed) = $l.$op($r);
        let primval = PrimVal::Bytes(val as u128);
        Ok((primval, overflowed))
    })
}

macro_rules! int_arithmetic {
    ($kind:expr, $int_op:ident, $l:expr, $r:expr) => ({
        let l = $l;
        let r = $r;
        match $kind {
            I8  => overflow!($int_op, l as i8,  r as i8),
            I16 => overflow!($int_op, l as i16, r as i16),
            I32 => overflow!($int_op, l as i32, r as i32),
            I64 => overflow!($int_op, l as i64, r as i64),
            I128 => overflow!($int_op, l as i128, r as i128),
            U8  => overflow!($int_op, l as u8,  r as u8),
            U16 => overflow!($int_op, l as u16, r as u16),
            U32 => overflow!($int_op, l as u32, r as u32),
            U64 => overflow!($int_op, l as u64, r as u64),
            U128 => overflow!($int_op, l as u128, r as u128),
            _ => bug!("int_arithmetic should only be called on int primvals"),
        }
    })
}

macro_rules! int_shift {
    ($kind:expr, $int_op:ident, $l:expr, $r:expr) => ({
        let l = $l;
        let r = $r;
        match $kind {
            I8  => overflow!($int_op, l as i8,  r),
            I16 => overflow!($int_op, l as i16, r),
            I32 => overflow!($int_op, l as i32, r),
            I64 => overflow!($int_op, l as i64, r),
            I128 => overflow!($int_op, l as i128, r),
            U8  => overflow!($int_op, l as u8,  r),
            U16 => overflow!($int_op, l as u16, r),
            U32 => overflow!($int_op, l as u32, r),
            U64 => overflow!($int_op, l as u64, r),
            U128 => overflow!($int_op, l as u128, r),
            _ => bug!("int_shift should only be called on int primvals"),
        }
    })
}

macro_rules! float_arithmetic {
    ($from_bytes:ident, $to_bytes:ident, $float_op:tt, $l:expr, $r:expr) => ({
        let l = $from_bytes($l);
        let r = $from_bytes($r);
        let bytes = $to_bytes(l $float_op r);
        PrimVal::Bytes(bytes)
    })
}

macro_rules! f32_arithmetic {
    ($float_op:tt, $l:expr, $r:expr) => (
        float_arithmetic!(bytes_to_f32, f32_to_bytes, $float_op, $l, $r)
    )
}

macro_rules! f64_arithmetic {
    ($float_op:tt, $l:expr, $r:expr) => (
        float_arithmetic!(bytes_to_f64, f64_to_bytes, $float_op, $l, $r)
    )
}

impl<'a, 'tcx> EvalContext<'a, 'tcx> {
    /// Returns the result of the specified operation and whether it overflowed.
    pub fn binary_op(
        &self,
        bin_op: mir::BinOp,
        left: PrimVal,
        left_ty: Ty<'tcx>,
        right: PrimVal,
        right_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, (PrimVal, bool)> {
        use rustc::mir::BinOp::*;
        use value::PrimValKind::*;

        // FIXME(solson): Temporary hack. It will go away when we get rid of Pointer's ability to store
        // plain bytes, and leave that to PrimVal::Bytes.
        fn normalize(val: PrimVal) -> PrimVal {
            if let PrimVal::Ptr(ptr) = val {
                if let Ok(bytes) = ptr.to_int() {
                    return PrimVal::Bytes(bytes as u128);
                }
            }
            val
        }
        let (left, right) = (normalize(left), normalize(right));

        let left_kind  = self.ty_to_primval_kind(left_ty)?;
        let right_kind = self.ty_to_primval_kind(right_ty)?;

        // Offset is handled early, before we dispatch to unrelated_ptr_ops.  We have to also catch the case where both arguments *are* convertible to integers.
        if bin_op == Offset {
            if left_kind == Ptr && right_kind == PrimValKind::from_uint_size(self.memory.pointer_size()) {
                let pointee_ty = left_ty.builtin_deref(true, ty::LvaluePreference::NoPreference).expect("Offset called on non-ptr type").ty;
                let ptr = self.pointer_offset(left.to_ptr()?, pointee_ty, right.to_bytes()? as i64)?;
                return Ok((PrimVal::Ptr(ptr), false));
            } else {
                bug!("Offset used with wrong type");
            }
        }

        let (l, r) = match (left, right) {
            (PrimVal::Bytes(left_bytes), PrimVal::Bytes(right_bytes)) => (left_bytes, right_bytes),

            // One argument is a pointer value -- this is handled separately
            (PrimVal::Ptr(left_ptr), PrimVal::Ptr(right_ptr)) => {
                return self.ptr_ops(bin_op, left_ptr, left_kind, right_ptr, right_kind);
            }
            (PrimVal::Ptr(ptr), PrimVal::Bytes(bytes)) => {
                return self.ptr_ops(bin_op, ptr, left_kind, Pointer::from_int(bytes as u64), right_kind);
            }
            (PrimVal::Bytes(bytes), PrimVal::Ptr(ptr)) => {
                return self.ptr_ops(bin_op, Pointer::from_int(bytes as u64), left_kind, ptr, right_kind);
            }

            (PrimVal::Undef, _) | (_, PrimVal::Undef) => return Err(EvalError::ReadUndefBytes),
        };

        // These ops can have an RHS with a different numeric type.
        if right_kind.is_int() && (bin_op == Shl || bin_op == Shr) {
            return match bin_op {
                Shl => int_shift!(left_kind, overflowing_shl, l, r as u32),
                Shr => int_shift!(left_kind, overflowing_shr, l, r as u32),
                _ => bug!("it has already been checked that this is a shift op"),
            };
        }

        if left_kind != right_kind {
            let msg = format!("unimplemented binary op {:?}: {:?} ({:?}), {:?} ({:?})", bin_op, left, left_kind, right, right_kind);
            return Err(EvalError::Unimplemented(msg));
        }

        let val = match (bin_op, left_kind) {
            (Eq, F32) => PrimVal::from_bool(bytes_to_f32(l) == bytes_to_f32(r)),
            (Ne, F32) => PrimVal::from_bool(bytes_to_f32(l) != bytes_to_f32(r)),
            (Lt, F32) => PrimVal::from_bool(bytes_to_f32(l) <  bytes_to_f32(r)),
            (Le, F32) => PrimVal::from_bool(bytes_to_f32(l) <= bytes_to_f32(r)),
            (Gt, F32) => PrimVal::from_bool(bytes_to_f32(l) >  bytes_to_f32(r)),
            (Ge, F32) => PrimVal::from_bool(bytes_to_f32(l) >= bytes_to_f32(r)),

            (Eq, F64) => PrimVal::from_bool(bytes_to_f64(l) == bytes_to_f64(r)),
            (Ne, F64) => PrimVal::from_bool(bytes_to_f64(l) != bytes_to_f64(r)),
            (Lt, F64) => PrimVal::from_bool(bytes_to_f64(l) <  bytes_to_f64(r)),
            (Le, F64) => PrimVal::from_bool(bytes_to_f64(l) <= bytes_to_f64(r)),
            (Gt, F64) => PrimVal::from_bool(bytes_to_f64(l) >  bytes_to_f64(r)),
            (Ge, F64) => PrimVal::from_bool(bytes_to_f64(l) >= bytes_to_f64(r)),

            (Add, F32) => f32_arithmetic!(+, l, r),
            (Sub, F32) => f32_arithmetic!(-, l, r),
            (Mul, F32) => f32_arithmetic!(*, l, r),
            (Div, F32) => f32_arithmetic!(/, l, r),
            (Rem, F32) => f32_arithmetic!(%, l, r),

            (Add, F64) => f64_arithmetic!(+, l, r),
            (Sub, F64) => f64_arithmetic!(-, l, r),
            (Mul, F64) => f64_arithmetic!(*, l, r),
            (Div, F64) => f64_arithmetic!(/, l, r),
            (Rem, F64) => f64_arithmetic!(%, l, r),

            (Eq, _) => PrimVal::from_bool(l == r),
            (Ne, _) => PrimVal::from_bool(l != r),
            (Lt, k) if k.is_signed_int() => PrimVal::from_bool((l as i128) < (r as i128)),
            (Lt, _) => PrimVal::from_bool(l <  r),
            (Le, k) if k.is_signed_int() => PrimVal::from_bool((l as i128) <= (r as i128)),
            (Le, _) => PrimVal::from_bool(l <= r),
            (Gt, k) if k.is_signed_int() => PrimVal::from_bool((l as i128) > (r as i128)),
            (Gt, _) => PrimVal::from_bool(l >  r),
            (Ge, k) if k.is_signed_int() => PrimVal::from_bool((l as i128) >= (r as i128)),
            (Ge, _) => PrimVal::from_bool(l >= r),

            (BitOr,  _) => PrimVal::Bytes(l | r),
            (BitAnd, _) => PrimVal::Bytes(l & r),
            (BitXor, _) => PrimVal::Bytes(l ^ r),

            (Add, k) if k.is_int() => return int_arithmetic!(k, overflowing_add, l, r),
            (Sub, k) if k.is_int() => return int_arithmetic!(k, overflowing_sub, l, r),
            (Mul, k) if k.is_int() => return int_arithmetic!(k, overflowing_mul, l, r),
            (Div, k) if k.is_int() => return int_arithmetic!(k, overflowing_div, l, r),
            (Rem, k) if k.is_int() => return int_arithmetic!(k, overflowing_rem, l, r),

            _ => {
                let msg = format!("unimplemented binary op {:?}: {:?} ({:?}), {:?} ({:?})", bin_op, left, left_kind, right, right_kind);
                return Err(EvalError::Unimplemented(msg));
            }
        };

        Ok((val, false))
    }

    fn ptr_ops(
        &self,
        bin_op: mir::BinOp,
        left: Pointer,
        left_kind: PrimValKind,
        right: Pointer,
        right_kind: PrimValKind,
    ) -> EvalResult<'tcx, (PrimVal, bool)> {
        use rustc::mir::BinOp::*;
        use value::PrimValKind::*;

        if left_kind != right_kind || !(left_kind.is_ptr() || left_kind == PrimValKind::from_uint_size(self.memory.pointer_size())) {
            let msg = format!("unimplemented binary op {:?}: {:?} ({:?}), {:?} ({:?})", bin_op, left, left_kind, right, right_kind);
            return Err(EvalError::Unimplemented(msg));
        }

        let val = match bin_op {
            Eq => PrimVal::from_bool(left == right),
            Ne => PrimVal::from_bool(left != right),
            Lt | Le | Gt | Ge => {
                if left.alloc_id == right.alloc_id {
                    PrimVal::from_bool(match bin_op {
                        Lt => left.offset < right.offset,
                        Le => left.offset <= right.offset,
                        Gt => left.offset > right.offset,
                        Ge => left.offset >= right.offset,
                        _ => bug!("We already established it has to be a comparison operator."),
                    })
                } else {
                    return Err(EvalError::InvalidPointerMath);
                }
            }
            Sub => {
                if left.alloc_id == right.alloc_id {
                    return int_arithmetic!(left_kind, overflowing_sub, left.offset, right.offset);
                } else {
                    return Err(EvalError::InvalidPointerMath);
                }
            }
            _ => {
                return Err(EvalError::ReadPointerAsBytes);
            }
        };
        Ok((val, false))
    }
}

pub fn unary_op<'tcx>(
    un_op: mir::UnOp,
    val: PrimVal,
    val_kind: PrimValKind,
) -> EvalResult<'tcx, PrimVal> {
    use rustc::mir::UnOp::*;
    use value::PrimValKind::*;

    let bytes = val.to_bytes()?;

    let result_bytes = match (un_op, val_kind) {
        (Not, Bool) => !bytes_to_bool(bytes) as u128,

        (Not, U8)  => !(bytes as u8) as u128,
        (Not, U16) => !(bytes as u16) as u128,
        (Not, U32) => !(bytes as u32) as u128,
        (Not, U64) => !(bytes as u64) as u128,
        (Not, U128) => !bytes,

        (Not, I8)  => !(bytes as i8) as u128,
        (Not, I16) => !(bytes as i16) as u128,
        (Not, I32) => !(bytes as i32) as u128,
        (Not, I64) => !(bytes as i64) as u128,
        (Not, I128) => !(bytes as i128) as u128,

        (Neg, I8)  => -(bytes as i8) as u128,
        (Neg, I16) => -(bytes as i16) as u128,
        (Neg, I32) => -(bytes as i32) as u128,
        (Neg, I64) => -(bytes as i64) as u128,
        (Neg, I128) => -(bytes as i128) as u128,

        (Neg, F32) => f32_to_bytes(-bytes_to_f32(bytes)),
        (Neg, F64) => f64_to_bytes(-bytes_to_f64(bytes)),

        _ => {
            let msg = format!("unimplemented unary op: {:?}, {:?}", un_op, val);
            return Err(EvalError::Unimplemented(msg));
        }
    };

    Ok(PrimVal::Bytes(result_bytes))
}
