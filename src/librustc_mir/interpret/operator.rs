use rustc::mir;
use rustc::ty::Ty;
use rustc_const_math::ConstFloat;
use syntax::ast::FloatTy;
use std::cmp::Ordering;

use super::{EvalContext, Place, Machine, ValTy};

use rustc::mir::interpret::{EvalResult, PrimVal, PrimValKind, Value, bytes_to_f32, bytes_to_f64};

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    fn binop_with_overflow(
        &mut self,
        op: mir::BinOp,
        left: ValTy<'tcx>,
        right: ValTy<'tcx>,
    ) -> EvalResult<'tcx, (PrimVal, bool)> {
        let left_val = self.value_to_primval(left)?;
        let right_val = self.value_to_primval(right)?;
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
        let val = Value::ByValPair(val, PrimVal::from_bool(overflowed));
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
        self.write_primval(dest, val, dest_ty)?;
        Ok(overflowed)
    }
}

macro_rules! overflow {
    (overflowing_div, $l:expr, $r:expr) => ({
        let (val, overflowed) = if $r == 0 {
            ($l, true)
        } else {
            $l.overflowing_div($r)
        };
        let primval = PrimVal::Bytes(val as u128);
        Ok((primval, overflowed))
    });
    (overflowing_rem, $l:expr, $r:expr) => ({
        let (val, overflowed) = if $r == 0 {
            ($l, true)
        } else {
            $l.overflowing_rem($r)
        };
        let primval = PrimVal::Bytes(val as u128);
        Ok((primval, overflowed))
    });
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
        use rustc::mir::interpret::PrimValKind::*;
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

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
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
        use rustc::mir::interpret::PrimValKind::*;

        let left_kind = self.ty_to_primval_kind(left_ty)?;
        let right_kind = self.ty_to_primval_kind(right_ty)?;
        //trace!("Running binary op {:?}: {:?} ({:?}), {:?} ({:?})", bin_op, left, left_kind, right, right_kind);

        // I: Handle operations that support pointers
        if !left_kind.is_float() && !right_kind.is_float() {
            if let Some(handled) = M::try_ptr_op(self, bin_op, left, left_ty, right, right_ty)? {
                return Ok(handled);
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

        let float_op = |op, l, r, ty| {
            let l = ConstFloat {
                bits: l,
                ty,
            };
            let r = ConstFloat {
                bits: r,
                ty,
            };
            match op {
                Eq => PrimVal::from_bool(l.try_cmp(r).unwrap() == Ordering::Equal),
                Ne => PrimVal::from_bool(l.try_cmp(r).unwrap() != Ordering::Equal),
                Lt => PrimVal::from_bool(l.try_cmp(r).unwrap() == Ordering::Less),
                Le => PrimVal::from_bool(l.try_cmp(r).unwrap() != Ordering::Greater),
                Gt => PrimVal::from_bool(l.try_cmp(r).unwrap() == Ordering::Greater),
                Ge => PrimVal::from_bool(l.try_cmp(r).unwrap() != Ordering::Less),
                Add => PrimVal::Bytes((l + r).unwrap().bits),
                Sub => PrimVal::Bytes((l - r).unwrap().bits),
                Mul => PrimVal::Bytes((l * r).unwrap().bits),
                Div => PrimVal::Bytes((l / r).unwrap().bits),
                Rem => PrimVal::Bytes((l % r).unwrap().bits),
                _ => bug!("invalid float op: `{:?}`", op),
            }
        };

        let val = match (bin_op, left_kind) {
            (_, F32) => float_op(bin_op, l, r, FloatTy::F32),
            (_, F64) => float_op(bin_op, l, r, FloatTy::F64),


            (Eq, _) => PrimVal::from_bool(l == r),
            (Ne, _) => PrimVal::from_bool(l != r),

            (Lt, k) if k.is_signed_int() => PrimVal::from_bool((l as i128) < (r as i128)),
            (Lt, _) => PrimVal::from_bool(l < r),
            (Le, k) if k.is_signed_int() => PrimVal::from_bool((l as i128) <= (r as i128)),
            (Le, _) => PrimVal::from_bool(l <= r),
            (Gt, k) if k.is_signed_int() => PrimVal::from_bool((l as i128) > (r as i128)),
            (Gt, _) => PrimVal::from_bool(l > r),
            (Ge, k) if k.is_signed_int() => PrimVal::from_bool((l as i128) >= (r as i128)),
            (Ge, _) => PrimVal::from_bool(l >= r),

            (BitOr, _) => PrimVal::Bytes(l | r),
            (BitAnd, _) => PrimVal::Bytes(l & r),
            (BitXor, _) => PrimVal::Bytes(l ^ r),

            (Add, k) if k.is_int() => return int_arithmetic!(k, overflowing_add, l, r),
            (Sub, k) if k.is_int() => return int_arithmetic!(k, overflowing_sub, l, r),
            (Mul, k) if k.is_int() => return int_arithmetic!(k, overflowing_mul, l, r),
            (Div, k) if k.is_int() => return int_arithmetic!(k, overflowing_div, l, r),
            (Rem, k) if k.is_int() => return int_arithmetic!(k, overflowing_rem, l, r),

            _ => {
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
    use rustc::mir::interpret::PrimValKind::*;

    let bytes = val.to_bytes()?;

    let result_bytes = match (un_op, val_kind) {
        (Not, Bool) => !val.to_bool()? as u128,

        (Not, U8) => !(bytes as u8) as u128,
        (Not, U16) => !(bytes as u16) as u128,
        (Not, U32) => !(bytes as u32) as u128,
        (Not, U64) => !(bytes as u64) as u128,
        (Not, U128) => !bytes,

        (Not, I8) => !(bytes as i8) as u128,
        (Not, I16) => !(bytes as i16) as u128,
        (Not, I32) => !(bytes as i32) as u128,
        (Not, I64) => !(bytes as i64) as u128,
        (Not, I128) => !(bytes as i128) as u128,

        (Neg, I8) if bytes == i8::min_value() as u128 => return err!(OverflowingMath),
        (Neg, I8) => -(bytes as i8) as u128,
        (Neg, I16) if bytes == i16::min_value() as u128 => return err!(OverflowingMath),
        (Neg, I16) => -(bytes as i16) as u128,
        (Neg, I32) if bytes == i32::min_value() as u128 => return err!(OverflowingMath),
        (Neg, I32) => -(bytes as i32) as u128,
        (Neg, I64) if bytes == i64::min_value() as u128 => return err!(OverflowingMath),
        (Neg, I64) => -(bytes as i64) as u128,
        (Neg, I128) if bytes == i128::min_value() as u128 => return err!(OverflowingMath),
        (Neg, I128) => -(bytes as i128) as u128,

        (Neg, F32) => (-bytes_to_f32(bytes)).bits,
        (Neg, F64) => (-bytes_to_f64(bytes)).bits,

        _ => {
            let msg = format!("unimplemented unary op: {:?}, {:?}", un_op, val);
            return err!(Unimplemented(msg));
        }
    };

    Ok(PrimVal::Bytes(result_bytes))
}
