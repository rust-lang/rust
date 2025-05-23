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
