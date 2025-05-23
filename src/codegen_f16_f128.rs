use crate::prelude::*;

pub(crate) fn f16_to_f32(fx: &mut FunctionCx<'_, '_, '_>, value: Value) -> Value {
    let (value, arg_ty) =
        if fx.tcx.sess.target.vendor == "apple" && fx.tcx.sess.target.arch == "x86_64" {
            (
                fx.bcx.ins().bitcast(types::I16, MemFlags::new(), value),
                lib_call_arg_param(fx.tcx, types::I16, false),
            )
        } else {
            (value, AbiParam::new(types::F16))
        };
    fx.lib_call("__extendhfsf2", vec![arg_ty], vec![AbiParam::new(types::F32)], &[value])[0]
}

fn f16_to_f64(fx: &mut FunctionCx<'_, '_, '_>, value: Value) -> Value {
    let ret = f16_to_f32(fx, value);
    fx.bcx.ins().fpromote(types::F64, ret)
}

pub(crate) fn f32_to_f16(fx: &mut FunctionCx<'_, '_, '_>, value: Value) -> Value {
    let ret_ty = if fx.tcx.sess.target.vendor == "apple" && fx.tcx.sess.target.arch == "x86_64" {
        types::I16
    } else {
        types::F16
    };
    let ret = fx.lib_call(
        "__truncsfhf2",
        vec![AbiParam::new(types::F32)],
        vec![AbiParam::new(ret_ty)],
        &[value],
    )[0];
    if ret_ty == types::I16 { fx.bcx.ins().bitcast(types::F16, MemFlags::new(), ret) } else { ret }
}

fn f64_to_f16(fx: &mut FunctionCx<'_, '_, '_>, value: Value) -> Value {
    let ret_ty = if fx.tcx.sess.target.vendor == "apple" && fx.tcx.sess.target.arch == "x86_64" {
        types::I16
    } else {
        types::F16
    };
    let ret = fx.lib_call(
        "__truncdfhf2",
        vec![AbiParam::new(types::F64)],
        vec![AbiParam::new(ret_ty)],
        &[value],
    )[0];
    if ret_ty == types::I16 { fx.bcx.ins().bitcast(types::F16, MemFlags::new(), ret) } else { ret }
}

pub(crate) fn fcmp(fx: &mut FunctionCx<'_, '_, '_>, cc: FloatCC, lhs: Value, rhs: Value) -> Value {
    let ty = fx.bcx.func.dfg.value_type(lhs);
    match ty {
        types::F32 | types::F64 => fx.bcx.ins().fcmp(cc, lhs, rhs),
        types::F16 => {
            let lhs = f16_to_f32(fx, lhs);
            let rhs = f16_to_f32(fx, rhs);
            fx.bcx.ins().fcmp(cc, lhs, rhs)
        }
        types::F128 => {
            let (name, int_cc) = match cc {
                FloatCC::Equal => ("__eqtf2", IntCC::Equal),
                FloatCC::NotEqual => ("__netf2", IntCC::NotEqual),
                FloatCC::LessThan => ("__lttf2", IntCC::SignedLessThan),
                FloatCC::LessThanOrEqual => ("__letf2", IntCC::SignedLessThanOrEqual),
                FloatCC::GreaterThan => ("__gttf2", IntCC::SignedGreaterThan),
                FloatCC::GreaterThanOrEqual => ("__getf2", IntCC::SignedGreaterThanOrEqual),
                _ => unreachable!("not currently used in rustc_codegen_cranelift: {cc:?}"),
            };
            let res = fx.lib_call(
                name,
                vec![AbiParam::new(types::F128), AbiParam::new(types::F128)],
                // FIXME(rust-lang/compiler-builtins#919): This should be `I64` on non-AArch64
                // architectures, but switching it before compiler-builtins is fixed causes test
                // failures.
                vec![AbiParam::new(types::I32)],
                &[lhs, rhs],
            )[0];
            let zero = fx.bcx.ins().iconst(types::I32, 0);
            let res = fx.bcx.ins().icmp(int_cc, res, zero);
            res
        }
        _ => unreachable!("{ty:?}"),
    }
}

pub(crate) fn codegen_f128_binop(
    fx: &mut FunctionCx<'_, '_, '_>,
    bin_op: BinOp,
    lhs: Value,
    rhs: Value,
) -> Value {
    let name = match bin_op {
        BinOp::Add => "__addtf3",
        BinOp::Sub => "__subtf3",
        BinOp::Mul => "__multf3",
        BinOp::Div => "__divtf3",
        _ => unreachable!("handled in `codegen_float_binop`"),
    };
    fx.lib_call(
        name,
        vec![AbiParam::new(types::F128), AbiParam::new(types::F128)],
        vec![AbiParam::new(types::F128)],
        &[lhs, rhs],
    )[0]
}

pub(crate) fn neg_f16(fx: &mut FunctionCx<'_, '_, '_>, value: Value) -> Value {
    let bits = fx.bcx.ins().bitcast(types::I16, MemFlags::new(), value);
    let bits = fx.bcx.ins().bxor_imm(bits, 0x8000);
    fx.bcx.ins().bitcast(types::F16, MemFlags::new(), bits)
}

pub(crate) fn neg_f128(fx: &mut FunctionCx<'_, '_, '_>, value: Value) -> Value {
    let bits = fx.bcx.ins().bitcast(types::I128, MemFlags::new(), value);
    let (low, high) = fx.bcx.ins().isplit(bits);
    let high = fx.bcx.ins().bxor_imm(high, 0x8000_0000_0000_0000_u64 as i64);
    let bits = fx.bcx.ins().iconcat(low, high);
    fx.bcx.ins().bitcast(types::F128, MemFlags::new(), bits)
}

pub(crate) fn abs_f16(fx: &mut FunctionCx<'_, '_, '_>, value: Value) -> Value {
    let bits = fx.bcx.ins().bitcast(types::I16, MemFlags::new(), value);
    let bits = fx.bcx.ins().band_imm(bits, 0x7fff);
    fx.bcx.ins().bitcast(types::F16, MemFlags::new(), bits)
}

pub(crate) fn abs_f128(fx: &mut FunctionCx<'_, '_, '_>, value: Value) -> Value {
    let bits = fx.bcx.ins().bitcast(types::I128, MemFlags::new(), value);
    let (low, high) = fx.bcx.ins().isplit(bits);
    let high = fx.bcx.ins().band_imm(high, 0x7fff_ffff_ffff_ffff_u64 as i64);
    let bits = fx.bcx.ins().iconcat(low, high);
    fx.bcx.ins().bitcast(types::F128, MemFlags::new(), bits)
}

pub(crate) fn copysign_f16(fx: &mut FunctionCx<'_, '_, '_>, lhs: Value, rhs: Value) -> Value {
    let lhs = fx.bcx.ins().bitcast(types::I16, MemFlags::new(), lhs);
    let rhs = fx.bcx.ins().bitcast(types::I16, MemFlags::new(), rhs);
    let res = fx.bcx.ins().band_imm(lhs, 0x7fff);
    let sign = fx.bcx.ins().band_imm(rhs, 0x8000);
    let res = fx.bcx.ins().bor(res, sign);
    fx.bcx.ins().bitcast(types::F16, MemFlags::new(), res)
}

pub(crate) fn copysign_f128(fx: &mut FunctionCx<'_, '_, '_>, lhs: Value, rhs: Value) -> Value {
    let lhs = fx.bcx.ins().bitcast(types::I128, MemFlags::new(), lhs);
    let rhs = fx.bcx.ins().bitcast(types::I128, MemFlags::new(), rhs);
    let (low, lhs_high) = fx.bcx.ins().isplit(lhs);
    let (_, rhs_high) = fx.bcx.ins().isplit(rhs);
    let high = fx.bcx.ins().band_imm(lhs_high, 0x7fff_ffff_ffff_ffff_u64 as i64);
    let sign = fx.bcx.ins().band_imm(rhs_high, 0x8000_0000_0000_0000_u64 as i64);
    let high = fx.bcx.ins().bor(high, sign);
    let res = fx.bcx.ins().iconcat(low, high);
    fx.bcx.ins().bitcast(types::F128, MemFlags::new(), res)
}

pub(crate) fn codegen_cast(
    fx: &mut FunctionCx<'_, '_, '_>,
    from: Value,
    from_signed: bool,
    to_ty: Type,
    to_signed: bool,
) -> Value {
    let from_ty = fx.bcx.func.dfg.value_type(from);
    if from_ty.is_float() && to_ty.is_float() {
        let name = match (from_ty, to_ty) {
            (types::F16, types::F32) => return f16_to_f32(fx, from),
            (types::F16, types::F64) => return f16_to_f64(fx, from),
            (types::F16, types::F128) => "__extendhftf2",
            (types::F32, types::F128) => "__extendsftf2",
            (types::F64, types::F128) => "__extenddftf2",
            (types::F128, types::F64) => "__trunctfdf2",
            (types::F128, types::F32) => "__trunctfsf2",
            (types::F128, types::F16) => "__trunctfhf2",
            (types::F64, types::F16) => return f64_to_f16(fx, from),
            (types::F32, types::F16) => return f32_to_f16(fx, from),
            _ => unreachable!("{from_ty:?} -> {to_ty:?}"),
        };
        fx.lib_call(name, vec![AbiParam::new(from_ty)], vec![AbiParam::new(to_ty)], &[from])[0]
    } else if from_ty.is_int() && to_ty == types::F16 {
        let res = clif_int_or_float_cast(fx, from, from_signed, types::F32, false);
        f32_to_f16(fx, res)
    } else if from_ty == types::F16 && to_ty.is_int() {
        let from = f16_to_f32(fx, from);
        clif_int_or_float_cast(fx, from, false, to_ty, to_signed)
    } else if from_ty.is_int() && to_ty == types::F128 {
        let (from, from_ty) = if from_ty.bits() < 32 {
            (clif_int_or_float_cast(fx, from, from_signed, types::I32, from_signed), types::I32)
        } else {
            (from, from_ty)
        };
        let name = format!(
            "__float{sign}{size}itf",
            sign = if from_signed { "" } else { "un" },
            size = match from_ty {
                types::I32 => 's',
                types::I64 => 'd',
                types::I128 => 't',
                _ => unreachable!("{from_ty:?}"),
            },
        );
        fx.lib_call(
            &name,
            vec![lib_call_arg_param(fx.tcx, from_ty, from_signed)],
            vec![AbiParam::new(to_ty)],
            &[from],
        )[0]
    } else if from_ty == types::F128 && to_ty.is_int() {
        let ret_ty = if to_ty.bits() < 32 { types::I32 } else { to_ty };
        let name = format!(
            "__fix{sign}tf{size}i",
            sign = if from_signed { "" } else { "un" },
            size = match ret_ty {
                types::I32 => 's',
                types::I64 => 'd',
                types::I128 => 't',
                _ => unreachable!("{from_ty:?}"),
            },
        );
        let ret =
            fx.lib_call(&name, vec![AbiParam::new(from_ty)], vec![AbiParam::new(to_ty)], &[from])
                [0];
        let val = if ret_ty == to_ty {
            ret
        } else {
            let (min, max) = match (to_ty, to_signed) {
                (types::I8, false) => (0, i64::from(u8::MAX)),
                (types::I16, false) => (0, i64::from(u16::MAX)),
                (types::I8, true) => (i64::from(i8::MIN as u32), i64::from(i8::MAX as u32)),
                (types::I16, true) => (i64::from(i16::MIN as u32), i64::from(i16::MAX as u32)),
                _ => unreachable!("{to_ty:?}"),
            };
            let min_val = fx.bcx.ins().iconst(types::I32, min);
            let max_val = fx.bcx.ins().iconst(types::I32, max);

            let val = if to_signed {
                let has_underflow = fx.bcx.ins().icmp_imm(IntCC::SignedLessThan, ret, min);
                let has_overflow = fx.bcx.ins().icmp_imm(IntCC::SignedGreaterThan, ret, max);
                let bottom_capped = fx.bcx.ins().select(has_underflow, min_val, ret);
                fx.bcx.ins().select(has_overflow, max_val, bottom_capped)
            } else {
                let has_overflow = fx.bcx.ins().icmp_imm(IntCC::UnsignedGreaterThan, ret, max);
                fx.bcx.ins().select(has_overflow, max_val, ret)
            };
            fx.bcx.ins().ireduce(to_ty, val)
        };

        if let Some(false) = fx.tcx.sess.opts.unstable_opts.saturating_float_casts {
            return val;
        }

        let is_not_nan = fcmp(fx, FloatCC::Equal, from, from);
        let zero = type_zero_value(&mut fx.bcx, to_ty);
        fx.bcx.ins().select(is_not_nan, val, zero)
    } else {
        unreachable!("{from_ty:?} -> {to_ty:?}");
    }
}

pub(crate) fn fma_f16(fx: &mut FunctionCx<'_, '_, '_>, x: Value, y: Value, z: Value) -> Value {
    let x = f16_to_f64(fx, x);
    let y = f16_to_f64(fx, y);
    let z = f16_to_f64(fx, z);
    let res = fx.bcx.ins().fma(x, y, z);
    f64_to_f16(fx, res)
}

pub(crate) fn fmin_f128(fx: &mut FunctionCx<'_, '_, '_>, a: Value, b: Value) -> Value {
    fx.lib_call(
        "fminimumf128",
        vec![AbiParam::new(types::F128), AbiParam::new(types::F128)],
        vec![AbiParam::new(types::F128)],
        &[a, b],
    )[0]
}

pub(crate) fn fmax_f128(fx: &mut FunctionCx<'_, '_, '_>, a: Value, b: Value) -> Value {
    fx.lib_call(
        "fmaximumf128",
        vec![AbiParam::new(types::F128), AbiParam::new(types::F128)],
        vec![AbiParam::new(types::F128)],
        &[a, b],
    )[0]
}
