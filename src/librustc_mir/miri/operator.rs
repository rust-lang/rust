use rustc::mir;
use rustc::ty::{self, Ty};

use error::{EvalError, EvalResult};
use eval_context::EvalContext;
use memory::MemoryPointer;
use lvalue::Lvalue;
use value::{
    PrimVal,
    PrimValKind,
    Value,
    bytes_to_f32,
    bytes_to_f64,
    f32_to_bytes,
    f64_to_bytes,
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
        use value::PrimValKind::*;
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
        let r_wrapped = r as u32;
        match $kind {
            I8  => overflow!($int_op, l as i8,  r_wrapped),
            I16 => overflow!($int_op, l as i16, r_wrapped),
            I32 => overflow!($int_op, l as i32, r_wrapped),
            I64 => overflow!($int_op, l as i64, r_wrapped),
            I128 => overflow!($int_op, l as i128, r_wrapped),
            U8  => overflow!($int_op, l as u8,  r_wrapped),
            U16 => overflow!($int_op, l as u16, r_wrapped),
            U32 => overflow!($int_op, l as u32, r_wrapped),
            U64 => overflow!($int_op, l as u64, r_wrapped),
            U128 => overflow!($int_op, l as u128, r_wrapped),
            _ => bug!("int_shift should only be called on int primvals"),
        }.map(|(val, over)| (val, over || r != r_wrapped as u128))
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

        let left_kind  = self.ty_to_primval_kind(left_ty)?;
        let right_kind = self.ty_to_primval_kind(right_ty)?;
        //trace!("Running binary op {:?}: {:?} ({:?}), {:?} ({:?})", bin_op, left, left_kind, right, right_kind);

        // I: Handle operations that support pointers
        let usize = PrimValKind::from_uint_size(self.memory.pointer_size());
        let isize = PrimValKind::from_int_size(self.memory.pointer_size());
        if !left_kind.is_float() && !right_kind.is_float() {
            if (!left.is_bytes() && !right.is_bytes()) && self.const_env() {
                return Err(EvalError::NeedsRfc("Pointer arithmetic or comparison".to_string()));
            }
            match bin_op {
                Offset if left_kind == Ptr && right_kind == usize => {
                    let pointee_ty = left_ty.builtin_deref(true, ty::LvaluePreference::NoPreference).expect("Offset called on non-ptr type").ty;
                    let ptr = self.pointer_offset(left.into(), pointee_ty, right.to_bytes()? as i64)?;
                    return Ok((ptr.into_inner_primval(), false));
                },
                // These work on anything
                Eq if left_kind == right_kind => {
                    let result = match (left, right) {
                        (PrimVal::Bytes(left), PrimVal::Bytes(right)) => left == right,
                        (PrimVal::Ptr(left), PrimVal::Ptr(right)) => left == right,
                        (PrimVal::Undef, _) | (_, PrimVal::Undef) => return Err(EvalError::ReadUndefBytes),
                        _ => false,
                    };
                    return Ok((PrimVal::from_bool(result), false));
                }
                Ne if left_kind == right_kind => {
                    let result = match (left, right) {
                        (PrimVal::Bytes(left), PrimVal::Bytes(right)) => left != right,
                        (PrimVal::Ptr(left), PrimVal::Ptr(right)) => left != right,
                        (PrimVal::Undef, _) | (_, PrimVal::Undef) => return Err(EvalError::ReadUndefBytes),
                        _ => true,
                    };
                    return Ok((PrimVal::from_bool(result), false));
                }
                // These need both pointers to be in the same allocation
                Lt | Le | Gt | Ge | Sub
                if left_kind == right_kind
                && (left_kind == Ptr || left_kind == usize || left_kind == isize)
                && left.is_ptr() && right.is_ptr() => {
                    let left = left.to_ptr()?;
                    let right = right.to_ptr()?;
                    if left.alloc_id == right.alloc_id {
                        let res = match bin_op {
                            Lt => left.offset < right.offset,
                            Le => left.offset <= right.offset,
                            Gt => left.offset > right.offset,
                            Ge => left.offset >= right.offset,
                            Sub => {
                                return int_arithmetic!(left_kind, overflowing_sub, left.offset, right.offset);
                            }
                            _ => bug!("We already established it has to be one of these operators."),
                        };
                        return Ok((PrimVal::from_bool(res), false));
                    } else {
                        // Both are pointers, but from different allocations.
                        return Err(EvalError::InvalidPointerMath);
                    }
                }
                // These work if one operand is a pointer, the other an integer
                Add | BitAnd | Sub
                if left_kind == right_kind && (left_kind == usize || left_kind == isize)
                && left.is_ptr() && right.is_bytes() => {
                    // Cast to i128 is fine as we checked the kind to be ptr-sized
                    return self.ptr_int_arithmetic(bin_op, left.to_ptr()?, right.to_bytes()? as i128, left_kind == isize);
                }
                Add | BitAnd
                if left_kind == right_kind && (left_kind == usize || left_kind == isize)
                && left.is_bytes() && right.is_ptr() => {
                    // This is a commutative operation, just swap the operands
                    return self.ptr_int_arithmetic(bin_op, right.to_ptr()?, left.to_bytes()? as i128, left_kind == isize);
                }
                _ => {}
            }
        }

        // II: From now on, everything must be bytes, no pointers
        let l = left.to_bytes()?;
        let r = right.to_bytes()?;

        // These ops can have an RHS with a different numeric type.
        if right_kind.is_int() && (bin_op == Shl || bin_op == Shr) {
            return match bin_op {
                Shl => int_shift!(left_kind, overflowing_shl, l, r),
                Shr => int_shift!(left_kind, overflowing_shr, l, r),
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

    fn ptr_int_arithmetic(
        &self,
        bin_op: mir::BinOp,
        left: MemoryPointer,
        right: i128,
        signed: bool,
    ) -> EvalResult<'tcx, (PrimVal, bool)> {
        use rustc::mir::BinOp::*;

        fn map_to_primval((res, over) : (MemoryPointer, bool)) -> (PrimVal, bool) {
            (PrimVal::Ptr(res), over)
        }

        Ok(match bin_op {
            Sub =>
                // The only way this can overflow is by underflowing, so signdeness of the right operands does not matter
                map_to_primval(left.overflowing_signed_offset(-right, self.memory.layout)),
            Add if signed =>
                map_to_primval(left.overflowing_signed_offset(right, self.memory.layout)),
            Add if !signed =>
                map_to_primval(left.overflowing_offset(right as u64, self.memory.layout)),

            BitAnd if !signed => {
                let base_mask : u64 = !(self.memory.get(left.alloc_id)?.align - 1);
                let right = right as u64;
                if right & base_mask == base_mask {
                    // Case 1: The base address bits are all preserved, i.e., right is all-1 there
                    (PrimVal::Ptr(MemoryPointer::new(left.alloc_id, left.offset & right)), false)
                } else if right & base_mask == 0 {
                    // Case 2: The base address bits are all taken away, i.e., right is all-0 there
                    (PrimVal::from_u128((left.offset & right) as u128), false)
                } else {
                    return Err(EvalError::ReadPointerAsBytes);
                }
            }

            _ => {
                let msg = format!("unimplemented binary op on pointer {:?}: {:?}, {:?} ({})", bin_op, left, right, if signed { "signed" } else { "unsigned" });
                return Err(EvalError::Unimplemented(msg));
            }
        })
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
        (Not, Bool) => !val.to_bool()? as u128,

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
