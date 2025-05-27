//! Various operations on integer and floating-point numbers

use crate::codegen_f16_f128;
use crate::prelude::*;

fn bin_op_to_intcc(bin_op: BinOp, signed: bool) -> IntCC {
    use BinOp::*;
    use IntCC::*;
    match bin_op {
        Eq => Equal,
        Lt => {
            if signed {
                SignedLessThan
            } else {
                UnsignedLessThan
            }
        }
        Le => {
            if signed {
                SignedLessThanOrEqual
            } else {
                UnsignedLessThanOrEqual
            }
        }
        Ne => NotEqual,
        Ge => {
            if signed {
                SignedGreaterThanOrEqual
            } else {
                UnsignedGreaterThanOrEqual
            }
        }
        Gt => {
            if signed {
                SignedGreaterThan
            } else {
                UnsignedGreaterThan
            }
        }
        _ => unreachable!(),
    }
}

fn codegen_three_way_compare<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    signed: bool,
    lhs: Value,
    rhs: Value,
) -> CValue<'tcx> {
    // This emits `(lhs > rhs) - (lhs < rhs)`, which is cranelift's preferred form per
    // <https://github.com/bytecodealliance/wasmtime/blob/8052bb9e3b792503b225f2a5b2ba3bc023bff462/cranelift/codegen/src/prelude_opt.isle#L41-L47>
    let gt_cc = crate::num::bin_op_to_intcc(BinOp::Gt, signed);
    let lt_cc = crate::num::bin_op_to_intcc(BinOp::Lt, signed);
    let gt = fx.bcx.ins().icmp(gt_cc, lhs, rhs);
    let lt = fx.bcx.ins().icmp(lt_cc, lhs, rhs);
    let val = fx.bcx.ins().isub(gt, lt);
    CValue::by_val(val, fx.layout_of(fx.tcx.ty_ordering_enum(Some(fx.mir.span))))
}

fn codegen_compare_bin_op<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    bin_op: BinOp,
    signed: bool,
    lhs: Value,
    rhs: Value,
) -> CValue<'tcx> {
    let intcc = crate::num::bin_op_to_intcc(bin_op, signed);
    let val = fx.bcx.ins().icmp(intcc, lhs, rhs);
    CValue::by_val(val, fx.layout_of(fx.tcx.types.bool))
}

pub(crate) fn codegen_binop<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    bin_op: BinOp,
    in_lhs: CValue<'tcx>,
    in_rhs: CValue<'tcx>,
) -> CValue<'tcx> {
    match bin_op {
        BinOp::Eq | BinOp::Lt | BinOp::Le | BinOp::Ne | BinOp::Ge | BinOp::Gt => {
            match in_lhs.layout().ty.kind() {
                ty::Bool | ty::Uint(_) | ty::Int(_) | ty::Char => {
                    let signed = type_sign(in_lhs.layout().ty);
                    let lhs = in_lhs.load_scalar(fx);
                    let rhs = in_rhs.load_scalar(fx);

                    return codegen_compare_bin_op(fx, bin_op, signed, lhs, rhs);
                }
                _ => {}
            }
        }
        BinOp::Cmp => match in_lhs.layout().ty.kind() {
            ty::Bool | ty::Uint(_) | ty::Int(_) | ty::Char => {
                let signed = type_sign(in_lhs.layout().ty);
                let lhs = in_lhs.load_scalar(fx);
                let rhs = in_rhs.load_scalar(fx);

                return codegen_three_way_compare(fx, signed, lhs, rhs);
            }
            _ => {}
        },
        _ => {}
    }

    match in_lhs.layout().ty.kind() {
        ty::Bool => crate::num::codegen_bool_binop(fx, bin_op, in_lhs, in_rhs),
        ty::Uint(_) | ty::Int(_) => crate::num::codegen_int_binop(fx, bin_op, in_lhs, in_rhs),
        ty::Float(_) => crate::num::codegen_float_binop(fx, bin_op, in_lhs, in_rhs),
        ty::RawPtr(..) | ty::FnPtr(..) => crate::num::codegen_ptr_binop(fx, bin_op, in_lhs, in_rhs),
        _ => unreachable!("{:?}({:?}, {:?})", bin_op, in_lhs.layout().ty, in_rhs.layout().ty),
    }
}

fn codegen_bool_binop<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    bin_op: BinOp,
    in_lhs: CValue<'tcx>,
    in_rhs: CValue<'tcx>,
) -> CValue<'tcx> {
    let lhs = in_lhs.load_scalar(fx);
    let rhs = in_rhs.load_scalar(fx);

    let b = fx.bcx.ins();
    let res = match bin_op {
        BinOp::BitXor => b.bxor(lhs, rhs),
        BinOp::BitAnd => b.band(lhs, rhs),
        BinOp::BitOr => b.bor(lhs, rhs),
        // Compare binops handles by `codegen_binop`.
        _ => unreachable!("{:?}({:?}, {:?})", bin_op, in_lhs, in_rhs),
    };

    CValue::by_val(res, fx.layout_of(fx.tcx.types.bool))
}

pub(crate) fn codegen_int_binop<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    bin_op: BinOp,
    in_lhs: CValue<'tcx>,
    in_rhs: CValue<'tcx>,
) -> CValue<'tcx> {
    if !matches!(bin_op, BinOp::Shl | BinOp::ShlUnchecked | BinOp::Shr | BinOp::ShrUnchecked) {
        assert_eq!(
            in_lhs.layout().ty,
            in_rhs.layout().ty,
            "int binop requires lhs and rhs of same type"
        );
    }

    if let Some(res) = crate::codegen_i128::maybe_codegen(fx, bin_op, in_lhs, in_rhs) {
        return res;
    }

    let signed = type_sign(in_lhs.layout().ty);

    let lhs = in_lhs.load_scalar(fx);
    let rhs = in_rhs.load_scalar(fx);

    let b = fx.bcx.ins();
    // FIXME trap on overflow for the Unchecked versions
    let val = match bin_op {
        BinOp::Add | BinOp::AddUnchecked => b.iadd(lhs, rhs),
        BinOp::Sub | BinOp::SubUnchecked => b.isub(lhs, rhs),
        BinOp::Mul | BinOp::MulUnchecked => b.imul(lhs, rhs),
        BinOp::Div => {
            if signed {
                b.sdiv(lhs, rhs)
            } else {
                b.udiv(lhs, rhs)
            }
        }
        BinOp::Rem => {
            if signed {
                b.srem(lhs, rhs)
            } else {
                b.urem(lhs, rhs)
            }
        }
        BinOp::BitXor => b.bxor(lhs, rhs),
        BinOp::BitAnd => b.band(lhs, rhs),
        BinOp::BitOr => b.bor(lhs, rhs),
        BinOp::Shl | BinOp::ShlUnchecked => b.ishl(lhs, rhs),
        BinOp::Shr | BinOp::ShrUnchecked => {
            if signed {
                b.sshr(lhs, rhs)
            } else {
                b.ushr(lhs, rhs)
            }
        }
        BinOp::Offset => unreachable!("Offset is not an integer operation"),
        BinOp::AddWithOverflow | BinOp::SubWithOverflow | BinOp::MulWithOverflow => {
            unreachable!("Overflow binops handled by `codegen_checked_int_binop`")
        }
        // Compare binops handles by `codegen_binop`.
        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Cmp => {
            unreachable!("{:?}({:?}, {:?})", bin_op, in_lhs.layout().ty, in_rhs.layout().ty);
        }
    };

    CValue::by_val(val, in_lhs.layout())
}

pub(crate) fn codegen_checked_int_binop<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    bin_op: BinOp,
    in_lhs: CValue<'tcx>,
    in_rhs: CValue<'tcx>,
) -> CValue<'tcx> {
    let lhs = in_lhs.load_scalar(fx);
    let rhs = in_rhs.load_scalar(fx);

    let signed = type_sign(in_lhs.layout().ty);

    let (res, has_overflow) = match bin_op {
        BinOp::Add => {
            /*let (val, c_out) = fx.bcx.ins().iadd_cout(lhs, rhs);
            (val, c_out)*/
            // FIXME(CraneStation/cranelift#849) legalize iadd_cout for i8 and i16
            let val = fx.bcx.ins().iadd(lhs, rhs);
            let has_overflow = if !signed {
                fx.bcx.ins().icmp(IntCC::UnsignedLessThan, val, lhs)
            } else {
                let rhs_is_negative = fx.bcx.ins().icmp_imm(IntCC::SignedLessThan, rhs, 0);
                let slt = fx.bcx.ins().icmp(IntCC::SignedLessThan, val, lhs);
                fx.bcx.ins().bxor(rhs_is_negative, slt)
            };
            (val, has_overflow)
        }
        BinOp::Sub => {
            /*let (val, b_out) = fx.bcx.ins().isub_bout(lhs, rhs);
            (val, b_out)*/
            // FIXME(CraneStation/cranelift#849) legalize isub_bout for i8 and i16
            let val = fx.bcx.ins().isub(lhs, rhs);
            let has_overflow = if !signed {
                fx.bcx.ins().icmp(IntCC::UnsignedGreaterThan, val, lhs)
            } else {
                let rhs_is_negative = fx.bcx.ins().icmp_imm(IntCC::SignedLessThan, rhs, 0);
                let sgt = fx.bcx.ins().icmp(IntCC::SignedGreaterThan, val, lhs);
                fx.bcx.ins().bxor(rhs_is_negative, sgt)
            };
            (val, has_overflow)
        }
        BinOp::Mul => {
            if let Some(res) = crate::codegen_i128::maybe_codegen_mul_checked(fx, in_lhs, in_rhs) {
                return res;
            }

            let ty = fx.bcx.func.dfg.value_type(lhs);
            match ty {
                types::I8 | types::I16 | types::I32 if !signed => {
                    let lhs = fx.bcx.ins().uextend(ty.double_width().unwrap(), lhs);
                    let rhs = fx.bcx.ins().uextend(ty.double_width().unwrap(), rhs);
                    let val = fx.bcx.ins().imul(lhs, rhs);
                    let has_overflow = fx.bcx.ins().icmp_imm(
                        IntCC::UnsignedGreaterThan,
                        val,
                        (1 << ty.bits()) - 1,
                    );
                    let val = fx.bcx.ins().ireduce(ty, val);
                    (val, has_overflow)
                }
                types::I8 | types::I16 | types::I32 if signed => {
                    let lhs = fx.bcx.ins().sextend(ty.double_width().unwrap(), lhs);
                    let rhs = fx.bcx.ins().sextend(ty.double_width().unwrap(), rhs);
                    let val = fx.bcx.ins().imul(lhs, rhs);
                    let has_underflow =
                        fx.bcx.ins().icmp_imm(IntCC::SignedLessThan, val, -(1 << (ty.bits() - 1)));
                    let has_overflow = fx.bcx.ins().icmp_imm(
                        IntCC::SignedGreaterThan,
                        val,
                        (1 << (ty.bits() - 1)) - 1,
                    );
                    let val = fx.bcx.ins().ireduce(ty, val);
                    (val, fx.bcx.ins().bor(has_underflow, has_overflow))
                }
                types::I64 => {
                    let val = fx.bcx.ins().imul(lhs, rhs);
                    let has_overflow = if !signed {
                        let val_hi = fx.bcx.ins().umulhi(lhs, rhs);
                        fx.bcx.ins().icmp_imm(IntCC::NotEqual, val_hi, 0)
                    } else {
                        // Based on LLVM's instruction sequence for compiling
                        // a.checked_mul(b).is_some() to riscv64gc:
                        // mulh    a2, a0, a1
                        // mul     a0, a0, a1
                        // srai    a0, a0, 63
                        // xor     a0, a0, a2
                        // snez    a0, a0
                        let val_hi = fx.bcx.ins().smulhi(lhs, rhs);
                        let val_sign = fx.bcx.ins().sshr_imm(val, i64::from(ty.bits() - 1));
                        let xor = fx.bcx.ins().bxor(val_hi, val_sign);
                        fx.bcx.ins().icmp_imm(IntCC::NotEqual, xor, 0)
                    };
                    (val, has_overflow)
                }
                types::I128 => {
                    unreachable!("i128 should have been handled by codegen_i128::maybe_codegen")
                }
                _ => unreachable!("invalid non-integer type {}", ty),
            }
        }
        _ => bug!("binop {:?} on checked int/uint lhs: {:?} rhs: {:?}", bin_op, in_lhs, in_rhs),
    };

    let out_layout = fx.layout_of(Ty::new_tup(fx.tcx, &[in_lhs.layout().ty, fx.tcx.types.bool]));
    CValue::by_val_pair(res, has_overflow, out_layout)
}

pub(crate) fn codegen_saturating_int_binop<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    bin_op: BinOp,
    lhs: CValue<'tcx>,
    rhs: CValue<'tcx>,
) -> CValue<'tcx> {
    assert_eq!(lhs.layout().ty, rhs.layout().ty);

    let signed = type_sign(lhs.layout().ty);
    let clif_ty = fx.clif_type(lhs.layout().ty).unwrap();
    let (min, max) = type_min_max_value(&mut fx.bcx, clif_ty, signed);

    let checked_res = crate::num::codegen_checked_int_binop(fx, bin_op, lhs, rhs);
    let (val, has_overflow) = checked_res.load_scalar_pair(fx);

    let val = match (bin_op, signed) {
        (BinOp::Add, false) => fx.bcx.ins().select(has_overflow, max, val),
        (BinOp::Sub, false) => fx.bcx.ins().select(has_overflow, min, val),
        (BinOp::Add, true) => {
            let rhs = rhs.load_scalar(fx);
            let rhs_ge_zero = fx.bcx.ins().icmp_imm(IntCC::SignedGreaterThanOrEqual, rhs, 0);
            let sat_val = fx.bcx.ins().select(rhs_ge_zero, max, min);
            fx.bcx.ins().select(has_overflow, sat_val, val)
        }
        (BinOp::Sub, true) => {
            let rhs = rhs.load_scalar(fx);
            let rhs_ge_zero = fx.bcx.ins().icmp_imm(IntCC::SignedGreaterThanOrEqual, rhs, 0);
            let sat_val = fx.bcx.ins().select(rhs_ge_zero, min, max);
            fx.bcx.ins().select(has_overflow, sat_val, val)
        }
        _ => unreachable!(),
    };

    CValue::by_val(val, lhs.layout())
}

pub(crate) fn codegen_float_binop<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    bin_op: BinOp,
    in_lhs: CValue<'tcx>,
    in_rhs: CValue<'tcx>,
) -> CValue<'tcx> {
    assert_eq!(in_lhs.layout().ty, in_rhs.layout().ty);

    let lhs = in_lhs.load_scalar(fx);
    let rhs = in_rhs.load_scalar(fx);

    // FIXME(bytecodealliance/wasmtime#8312): Remove once backend lowerings have
    // been added to Cranelift.
    let (lhs, rhs) = if *in_lhs.layout().ty.kind() == ty::Float(FloatTy::F16) {
        (codegen_f16_f128::f16_to_f32(fx, lhs), codegen_f16_f128::f16_to_f32(fx, rhs))
    } else {
        (lhs, rhs)
    };
    let b = fx.bcx.ins();
    let res = match bin_op {
        // FIXME(bytecodealliance/wasmtime#8312): Remove once backend lowerings
        // have been added to Cranelift.
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div
            if *in_lhs.layout().ty.kind() == ty::Float(FloatTy::F128) =>
        {
            codegen_f16_f128::codegen_f128_binop(fx, bin_op, lhs, rhs)
        }
        BinOp::Add => b.fadd(lhs, rhs),
        BinOp::Sub => b.fsub(lhs, rhs),
        BinOp::Mul => b.fmul(lhs, rhs),
        BinOp::Div => b.fdiv(lhs, rhs),
        BinOp::Rem => {
            let (name, ty, lhs, rhs) = match in_lhs.layout().ty.kind() {
                ty::Float(FloatTy::F16) => (
                    "fmodf",
                    types::F32,
                    // FIXME(bytecodealliance/wasmtime#8312): Already converted
                    // by the FIXME above.
                    // fx.bcx.ins().fpromote(types::F32, lhs),
                    // fx.bcx.ins().fpromote(types::F32, rhs),
                    lhs,
                    rhs,
                ),
                ty::Float(FloatTy::F32) => ("fmodf", types::F32, lhs, rhs),
                ty::Float(FloatTy::F64) => ("fmod", types::F64, lhs, rhs),
                ty::Float(FloatTy::F128) => ("fmodf128", types::F128, lhs, rhs),
                _ => bug!(),
            };

            let ret_val = fx.lib_call(
                name,
                vec![AbiParam::new(ty), AbiParam::new(ty)],
                vec![AbiParam::new(ty)],
                &[lhs, rhs],
            )[0];

            let ret_val = if *in_lhs.layout().ty.kind() == ty::Float(FloatTy::F16) {
                // FIXME(bytecodealliance/wasmtime#8312): Use native Cranelift
                // operation once Cranelift backend lowerings have been
                // implemented.
                codegen_f16_f128::f32_to_f16(fx, ret_val)
            } else {
                ret_val
            };
            return CValue::by_val(ret_val, in_lhs.layout());
        }
        BinOp::Eq | BinOp::Lt | BinOp::Le | BinOp::Ne | BinOp::Ge | BinOp::Gt => {
            let fltcc = match bin_op {
                BinOp::Eq => FloatCC::Equal,
                BinOp::Lt => FloatCC::LessThan,
                BinOp::Le => FloatCC::LessThanOrEqual,
                BinOp::Ne => FloatCC::NotEqual,
                BinOp::Ge => FloatCC::GreaterThanOrEqual,
                BinOp::Gt => FloatCC::GreaterThan,
                _ => unreachable!(),
            };
            // FIXME(bytecodealliance/wasmtime#8312): Replace with Cranelift
            // `fcmp` once `f16`/`f128` backend lowerings have been added to
            // Cranelift.
            let val = codegen_f16_f128::fcmp(fx, fltcc, lhs, rhs);
            return CValue::by_val(val, fx.layout_of(fx.tcx.types.bool));
        }
        _ => unreachable!("{:?}({:?}, {:?})", bin_op, in_lhs, in_rhs),
    };

    // FIXME(bytecodealliance/wasmtime#8312): Remove once backend lowerings have
    // been added to Cranelift.
    let res = if *in_lhs.layout().ty.kind() == ty::Float(FloatTy::F16) {
        codegen_f16_f128::f32_to_f16(fx, res)
    } else {
        res
    };
    CValue::by_val(res, in_lhs.layout())
}

fn codegen_ptr_binop<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    bin_op: BinOp,
    in_lhs: CValue<'tcx>,
    in_rhs: CValue<'tcx>,
) -> CValue<'tcx> {
    let is_thin_ptr = in_lhs
        .layout()
        .ty
        .builtin_deref(true)
        .map(|ty| !fx.tcx.type_has_metadata(ty, ty::TypingEnv::fully_monomorphized()))
        .unwrap_or(true);

    if is_thin_ptr {
        match bin_op {
            BinOp::Eq | BinOp::Lt | BinOp::Le | BinOp::Ne | BinOp::Ge | BinOp::Gt => {
                let lhs = in_lhs.load_scalar(fx);
                let rhs = in_rhs.load_scalar(fx);

                codegen_compare_bin_op(fx, bin_op, false, lhs, rhs)
            }
            BinOp::Offset => {
                let pointee_ty = in_lhs.layout().ty.builtin_deref(true).unwrap();
                let (base, offset) = (in_lhs, in_rhs.load_scalar(fx));
                let pointee_size = fx.layout_of(pointee_ty).size.bytes();
                let ptr_diff = fx.bcx.ins().imul_imm(offset, pointee_size as i64);
                let base_val = base.load_scalar(fx);
                let res = fx.bcx.ins().iadd(base_val, ptr_diff);
                CValue::by_val(res, base.layout())
            }
            _ => unreachable!("{:?}({:?}, {:?})", bin_op, in_lhs, in_rhs),
        }
    } else {
        let (lhs_ptr, lhs_extra) = in_lhs.load_scalar_pair(fx);
        let (rhs_ptr, rhs_extra) = in_rhs.load_scalar_pair(fx);

        let res = match bin_op {
            BinOp::Eq => {
                let ptr_eq = fx.bcx.ins().icmp(IntCC::Equal, lhs_ptr, rhs_ptr);
                let extra_eq = fx.bcx.ins().icmp(IntCC::Equal, lhs_extra, rhs_extra);
                fx.bcx.ins().band(ptr_eq, extra_eq)
            }
            BinOp::Ne => {
                let ptr_ne = fx.bcx.ins().icmp(IntCC::NotEqual, lhs_ptr, rhs_ptr);
                let extra_ne = fx.bcx.ins().icmp(IntCC::NotEqual, lhs_extra, rhs_extra);
                fx.bcx.ins().bor(ptr_ne, extra_ne)
            }
            BinOp::Lt | BinOp::Le | BinOp::Ge | BinOp::Gt => {
                let ptr_eq = fx.bcx.ins().icmp(IntCC::Equal, lhs_ptr, rhs_ptr);

                let ptr_cmp = fx.bcx.ins().icmp(bin_op_to_intcc(bin_op, false), lhs_ptr, rhs_ptr);
                let extra_cmp =
                    fx.bcx.ins().icmp(bin_op_to_intcc(bin_op, false), lhs_extra, rhs_extra);

                fx.bcx.ins().select(ptr_eq, extra_cmp, ptr_cmp)
            }
            _ => panic!("bin_op {:?} on ptr", bin_op),
        };

        CValue::by_val(res, fx.layout_of(fx.tcx.types.bool))
    }
}

// In Rust floating point min and max don't propagate NaN. In Cranelift they do however.
// For this reason it is necessary to use `a.is_nan() ? b : (a >= b ? b : a)` for `minnumf*`
// and `a.is_nan() ? b : (a <= b ? b : a)` for `maxnumf*`. NaN checks are done by comparing
// a float against itself. Only in case of NaN is it not equal to itself.
pub(crate) fn codegen_float_min(fx: &mut FunctionCx<'_, '_, '_>, a: Value, b: Value) -> Value {
    // FIXME(bytecodealliance/wasmtime#8312): Replace with Cranelift `fcmp` once
    // `f16`/`f128` backend lowerings have been added to Cranelift.
    let a_is_nan = codegen_f16_f128::fcmp(fx, FloatCC::NotEqual, a, a);
    let a_ge_b = codegen_f16_f128::fcmp(fx, FloatCC::GreaterThanOrEqual, a, b);
    let temp = fx.bcx.ins().select(a_ge_b, b, a);
    fx.bcx.ins().select(a_is_nan, b, temp)
}

pub(crate) fn codegen_float_max(fx: &mut FunctionCx<'_, '_, '_>, a: Value, b: Value) -> Value {
    // FIXME(bytecodealliance/wasmtime#8312): Replace with Cranelift `fcmp` once
    // `f16`/`f128` backend lowerings have been added to Cranelift.
    let a_is_nan = codegen_f16_f128::fcmp(fx, FloatCC::NotEqual, a, a);
    let a_le_b = codegen_f16_f128::fcmp(fx, FloatCC::LessThanOrEqual, a, b);
    let temp = fx.bcx.ins().select(a_le_b, b, a);
    fx.bcx.ins().select(a_is_nan, b, temp)
}
