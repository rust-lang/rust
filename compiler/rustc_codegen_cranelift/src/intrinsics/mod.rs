//! Codegen of intrinsics. This includes functions marked with the `#[rustc_intrinsic]` attribute
//! and LLVM intrinsics that have symbol names starting with `llvm.`.

macro_rules! intrinsic_args {
    ($fx:expr, $args:expr => ($($arg:tt),*); $intrinsic:expr) => {
        #[allow(unused_parens)]
        let ($($arg),*) = if let [$($arg),*] = $args {
            ($(codegen_operand($fx, &($arg).node)),*)
        } else {
            $crate::intrinsics::bug_on_incorrect_arg_count($intrinsic);
        };
    }
}

mod llvm;
mod llvm_aarch64;
mod llvm_x86;
mod simd;

use cranelift_codegen::ir::AtomicRmwOp;
use rustc_middle::ty;
use rustc_middle::ty::GenericArgsRef;
use rustc_middle::ty::layout::ValidityRequirement;
use rustc_middle::ty::print::{with_no_trimmed_paths, with_no_visible_paths};
use rustc_span::source_map::Spanned;
use rustc_span::{Symbol, sym};

pub(crate) use self::llvm::codegen_llvm_intrinsic_call;
use crate::cast::clif_intcast;
use crate::codegen_f16_f128;
use crate::prelude::*;

fn bug_on_incorrect_arg_count(intrinsic: impl std::fmt::Display) -> ! {
    bug!("wrong number of args for intrinsic {}", intrinsic);
}

fn report_atomic_type_validation_error<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    intrinsic: Symbol,
    span: Span,
    ty: Ty<'tcx>,
) {
    fx.tcx.dcx().span_err(
        span,
        format!(
            "`{}` intrinsic: expected basic integer or raw pointer type, found `{:?}`",
            intrinsic, ty
        ),
    );
    // Prevent verifier error
    fx.bcx.ins().trap(TrapCode::user(1 /* unreachable */).unwrap());
}

pub(crate) fn clif_vector_type<'tcx>(tcx: TyCtxt<'tcx>, layout: TyAndLayout<'tcx>) -> Type {
    let (element, count) = match layout.backend_repr {
        BackendRepr::SimdVector { element, count } => (element, count),
        _ => unreachable!(),
    };

    scalar_to_clif_type(tcx, element).by(u32::try_from(count).unwrap()).unwrap()
}

fn simd_for_each_lane<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    val: CValue<'tcx>,
    ret: CPlace<'tcx>,
    f: &dyn Fn(&mut FunctionCx<'_, '_, 'tcx>, Ty<'tcx>, Ty<'tcx>, Value) -> Value,
) {
    let layout = val.layout();

    let (lane_count, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
    let lane_layout = fx.layout_of(lane_ty);
    let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);
    let ret_lane_layout = fx.layout_of(ret_lane_ty);
    assert_eq!(lane_count, ret_lane_count);

    for lane_idx in 0..lane_count {
        let lane = val.value_lane(fx, lane_idx).load_scalar(fx);

        let res_lane = f(fx, lane_layout.ty, ret_lane_layout.ty, lane);
        let res_lane = CValue::by_val(res_lane, ret_lane_layout);

        ret.place_lane(fx, lane_idx).write_cvalue(fx, res_lane);
    }
}

fn simd_pair_for_each_lane_typed<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    x: CValue<'tcx>,
    y: CValue<'tcx>,
    ret: CPlace<'tcx>,
    f: &dyn Fn(&mut FunctionCx<'_, '_, 'tcx>, CValue<'tcx>, CValue<'tcx>) -> CValue<'tcx>,
) {
    assert_eq!(x.layout(), y.layout());
    let layout = x.layout();

    let (lane_count, _lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
    let (ret_lane_count, _ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);
    assert_eq!(lane_count, ret_lane_count);

    for lane_idx in 0..lane_count {
        let x_lane = x.value_lane(fx, lane_idx);
        let y_lane = y.value_lane(fx, lane_idx);

        let res_lane = f(fx, x_lane, y_lane);

        ret.place_lane(fx, lane_idx).write_cvalue(fx, res_lane);
    }
}

fn simd_pair_for_each_lane<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    x: CValue<'tcx>,
    y: CValue<'tcx>,
    ret: CPlace<'tcx>,
    f: &dyn Fn(&mut FunctionCx<'_, '_, 'tcx>, Ty<'tcx>, Ty<'tcx>, Value, Value) -> Value,
) {
    assert_eq!(x.layout(), y.layout());
    let layout = x.layout();

    let (lane_count, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
    let lane_layout = fx.layout_of(lane_ty);
    let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);
    let ret_lane_layout = fx.layout_of(ret_lane_ty);
    assert_eq!(lane_count, ret_lane_count);

    for lane_idx in 0..lane_count {
        let x_lane = x.value_lane(fx, lane_idx).load_scalar(fx);
        let y_lane = y.value_lane(fx, lane_idx).load_scalar(fx);

        let res_lane = f(fx, lane_layout.ty, ret_lane_layout.ty, x_lane, y_lane);
        let res_lane = CValue::by_val(res_lane, ret_lane_layout);

        ret.place_lane(fx, lane_idx).write_cvalue(fx, res_lane);
    }
}

fn simd_horizontal_pair_for_each_lane<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    x: CValue<'tcx>,
    y: CValue<'tcx>,
    ret: CPlace<'tcx>,
    f: &dyn Fn(&mut FunctionCx<'_, '_, 'tcx>, Ty<'tcx>, Ty<'tcx>, Value, Value) -> Value,
) {
    assert_eq!(x.layout(), y.layout());
    let layout = x.layout();

    let (lane_count, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
    let lane_layout = fx.layout_of(lane_ty);
    let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);
    let ret_lane_layout = fx.layout_of(ret_lane_ty);
    assert_eq!(lane_count, ret_lane_count);

    for lane_idx in 0..lane_count {
        let src = if lane_idx < (lane_count / 2) { x } else { y };
        let src_idx = lane_idx % (lane_count / 2);

        let lhs_lane = src.value_lane(fx, src_idx * 2).load_scalar(fx);
        let rhs_lane = src.value_lane(fx, src_idx * 2 + 1).load_scalar(fx);

        let res_lane = f(fx, lane_layout.ty, ret_lane_layout.ty, lhs_lane, rhs_lane);
        let res_lane = CValue::by_val(res_lane, ret_lane_layout);

        ret.place_lane(fx, lane_idx).write_cvalue(fx, res_lane);
    }
}

fn simd_trio_for_each_lane<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    x: CValue<'tcx>,
    y: CValue<'tcx>,
    z: CValue<'tcx>,
    ret: CPlace<'tcx>,
    f: &dyn Fn(&mut FunctionCx<'_, '_, 'tcx>, Ty<'tcx>, Ty<'tcx>, Value, Value, Value) -> Value,
) {
    assert_eq!(x.layout(), y.layout());
    let layout = x.layout();

    let (lane_count, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
    let lane_layout = fx.layout_of(lane_ty);
    let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);
    let ret_lane_layout = fx.layout_of(ret_lane_ty);
    assert_eq!(lane_count, ret_lane_count);

    for lane_idx in 0..lane_count {
        let x_lane = x.value_lane(fx, lane_idx).load_scalar(fx);
        let y_lane = y.value_lane(fx, lane_idx).load_scalar(fx);
        let z_lane = z.value_lane(fx, lane_idx).load_scalar(fx);

        let res_lane = f(fx, lane_layout.ty, ret_lane_layout.ty, x_lane, y_lane, z_lane);
        let res_lane = CValue::by_val(res_lane, ret_lane_layout);

        ret.place_lane(fx, lane_idx).write_cvalue(fx, res_lane);
    }
}

fn simd_reduce<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    val: CValue<'tcx>,
    acc: Option<Value>,
    ret: CPlace<'tcx>,
    f: &dyn Fn(&mut FunctionCx<'_, '_, 'tcx>, Ty<'tcx>, Value, Value) -> Value,
) {
    let (lane_count, lane_ty) = val.layout().ty.simd_size_and_type(fx.tcx);
    let lane_layout = fx.layout_of(lane_ty);
    assert_eq!(lane_layout, ret.layout());

    let (mut res_val, start_lane) =
        if let Some(acc) = acc { (acc, 0) } else { (val.value_lane(fx, 0).load_scalar(fx), 1) };
    for lane_idx in start_lane..lane_count {
        let lane = val.value_lane(fx, lane_idx).load_scalar(fx);
        res_val = f(fx, lane_layout.ty, res_val, lane);
    }
    let res = CValue::by_val(res_val, lane_layout);
    ret.write_cvalue(fx, res);
}

// FIXME move all uses to `simd_reduce`
fn simd_reduce_bool<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    val: CValue<'tcx>,
    ret: CPlace<'tcx>,
    f: &dyn Fn(&mut FunctionCx<'_, '_, 'tcx>, Value, Value) -> Value,
) {
    let (lane_count, _lane_ty) = val.layout().ty.simd_size_and_type(fx.tcx);
    assert!(ret.layout().ty.is_bool());

    let res_val = val.value_lane(fx, 0).load_scalar(fx);
    let mut res_val = fx.bcx.ins().band_imm(res_val, 1); // mask to boolean
    for lane_idx in 1..lane_count {
        let lane = val.value_lane(fx, lane_idx).load_scalar(fx);
        let lane = fx.bcx.ins().band_imm(lane, 1); // mask to boolean
        res_val = f(fx, res_val, lane);
    }
    let res_val = if fx.bcx.func.dfg.value_type(res_val) != types::I8 {
        fx.bcx.ins().ireduce(types::I8, res_val)
    } else {
        res_val
    };
    let res = CValue::by_val(res_val, ret.layout());
    ret.write_cvalue(fx, res);
}

fn bool_to_zero_or_max_uint<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    ty: Ty<'tcx>,
    val: Value,
) -> Value {
    let ty = fx.clif_type(ty).unwrap();

    let int_ty = match ty {
        types::F16 => types::I16,
        types::F32 => types::I32,
        types::F64 => types::I64,
        types::F128 => types::I128,
        ty => ty,
    };

    let mut res = fx.bcx.ins().bmask(int_ty, val);

    if ty.is_float() {
        res = codegen_bitcast(fx, ty, res);
    }

    res
}

pub(crate) fn codegen_intrinsic_call<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    instance: Instance<'tcx>,
    args: &[Spanned<mir::Operand<'tcx>>],
    destination: CPlace<'tcx>,
    target: Option<BasicBlock>,
    source_info: mir::SourceInfo,
) -> Result<(), Instance<'tcx>> {
    let intrinsic = fx.tcx.item_name(instance.def_id());
    let instance_args = instance.args;

    if intrinsic.as_str().starts_with("simd_") {
        self::simd::codegen_simd_intrinsic_call(
            fx,
            intrinsic,
            instance_args,
            args,
            destination,
            target.expect("target for simd intrinsic"),
            source_info.span,
        );
    } else if codegen_float_intrinsic_call(fx, intrinsic, args, destination) {
        let ret_block = fx.get_block(target.expect("target for float intrinsic"));
        fx.bcx.ins().jump(ret_block, &[]);
    } else {
        codegen_regular_intrinsic_call(
            fx,
            instance,
            intrinsic,
            instance_args,
            args,
            destination,
            target,
            source_info,
        )?;
    }
    Ok(())
}

fn codegen_float_intrinsic_call<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    intrinsic: Symbol,
    args: &[Spanned<mir::Operand<'tcx>>],
    ret: CPlace<'tcx>,
) -> bool {
    let (name, arg_count, ty, clif_ty) = match intrinsic {
        sym::expf16 => ("expf16", 1, fx.tcx.types.f16, types::F16),
        sym::expf32 => ("expf", 1, fx.tcx.types.f32, types::F32),
        sym::expf64 => ("exp", 1, fx.tcx.types.f64, types::F64),
        sym::expf128 => ("expf128", 1, fx.tcx.types.f128, types::F128),
        sym::exp2f16 => ("exp2f16", 1, fx.tcx.types.f16, types::F16),
        sym::exp2f32 => ("exp2f", 1, fx.tcx.types.f32, types::F32),
        sym::exp2f64 => ("exp2", 1, fx.tcx.types.f64, types::F64),
        sym::exp2f128 => ("exp2f128", 1, fx.tcx.types.f128, types::F128),
        sym::sqrtf16 => ("sqrtf16", 1, fx.tcx.types.f16, types::F16),
        sym::sqrtf32 => ("sqrtf", 1, fx.tcx.types.f32, types::F32),
        sym::sqrtf64 => ("sqrt", 1, fx.tcx.types.f64, types::F64),
        sym::sqrtf128 => ("sqrtf128", 1, fx.tcx.types.f128, types::F128),
        sym::powif16 => ("__powisf2", 2, fx.tcx.types.f16, types::F16), // compiler-builtins
        sym::powif32 => ("__powisf2", 2, fx.tcx.types.f32, types::F32), // compiler-builtins
        sym::powif64 => ("__powidf2", 2, fx.tcx.types.f64, types::F64), // compiler-builtins
        sym::powif128 => ("__powitf2", 2, fx.tcx.types.f128, types::F128), // compiler-builtins
        sym::powf16 => ("powf16", 2, fx.tcx.types.f16, types::F16),
        sym::powf32 => ("powf", 2, fx.tcx.types.f32, types::F32),
        sym::powf64 => ("pow", 2, fx.tcx.types.f64, types::F64),
        sym::powf128 => ("powf128", 2, fx.tcx.types.f128, types::F128),
        sym::logf16 => ("logf16", 1, fx.tcx.types.f16, types::F16),
        sym::logf32 => ("logf", 1, fx.tcx.types.f32, types::F32),
        sym::logf64 => ("log", 1, fx.tcx.types.f64, types::F64),
        sym::logf128 => ("logf128", 1, fx.tcx.types.f128, types::F128),
        sym::log2f16 => ("log2f16", 1, fx.tcx.types.f16, types::F16),
        sym::log2f32 => ("log2f", 1, fx.tcx.types.f32, types::F32),
        sym::log2f64 => ("log2", 1, fx.tcx.types.f64, types::F64),
        sym::log2f128 => ("log2f128", 1, fx.tcx.types.f128, types::F128),
        sym::log10f16 => ("log10f16", 1, fx.tcx.types.f16, types::F16),
        sym::log10f32 => ("log10f", 1, fx.tcx.types.f32, types::F32),
        sym::log10f64 => ("log10", 1, fx.tcx.types.f64, types::F64),
        sym::log10f128 => ("log10f128", 1, fx.tcx.types.f128, types::F128),
        sym::fabsf16 => ("fabsf16", 1, fx.tcx.types.f16, types::F16),
        sym::fabsf32 => ("fabsf", 1, fx.tcx.types.f32, types::F32),
        sym::fabsf64 => ("fabs", 1, fx.tcx.types.f64, types::F64),
        sym::fabsf128 => ("fabsf128", 1, fx.tcx.types.f128, types::F128),
        sym::fmaf16 => ("fmaf16", 3, fx.tcx.types.f16, types::F16),
        sym::fmaf32 => ("fmaf", 3, fx.tcx.types.f32, types::F32),
        sym::fmaf64 => ("fma", 3, fx.tcx.types.f64, types::F64),
        sym::fmaf128 => ("fmaf128", 3, fx.tcx.types.f128, types::F128),
        // FIXME: calling `fma` from libc without FMA target feature uses expensive sofware emulation
        sym::fmuladdf16 => ("fmaf16", 3, fx.tcx.types.f16, types::F16), // TODO: use cranelift intrinsic analogous to llvm.fmuladd.f16
        sym::fmuladdf32 => ("fmaf", 3, fx.tcx.types.f32, types::F32), // TODO: use cranelift intrinsic analogous to llvm.fmuladd.f32
        sym::fmuladdf64 => ("fma", 3, fx.tcx.types.f64, types::F64), // TODO: use cranelift intrinsic analogous to llvm.fmuladd.f64
        sym::fmuladdf128 => ("fmaf128", 3, fx.tcx.types.f128, types::F128), // TODO: use cranelift intrinsic analogous to llvm.fmuladd.f128
        sym::copysignf16 => ("copysignf16", 2, fx.tcx.types.f16, types::F16),
        sym::copysignf32 => ("copysignf", 2, fx.tcx.types.f32, types::F32),
        sym::copysignf64 => ("copysign", 2, fx.tcx.types.f64, types::F64),
        sym::copysignf128 => ("copysignf128", 2, fx.tcx.types.f128, types::F128),
        sym::floorf16 => ("floorf16", 1, fx.tcx.types.f16, types::F16),
        sym::floorf32 => ("floorf", 1, fx.tcx.types.f32, types::F32),
        sym::floorf64 => ("floor", 1, fx.tcx.types.f64, types::F64),
        sym::floorf128 => ("floorf128", 1, fx.tcx.types.f128, types::F128),
        sym::ceilf16 => ("ceilf16", 1, fx.tcx.types.f16, types::F16),
        sym::ceilf32 => ("ceilf", 1, fx.tcx.types.f32, types::F32),
        sym::ceilf64 => ("ceil", 1, fx.tcx.types.f64, types::F64),
        sym::ceilf128 => ("ceilf128", 1, fx.tcx.types.f128, types::F128),
        sym::truncf16 => ("truncf16", 1, fx.tcx.types.f16, types::F16),
        sym::truncf32 => ("truncf", 1, fx.tcx.types.f32, types::F32),
        sym::truncf64 => ("trunc", 1, fx.tcx.types.f64, types::F64),
        sym::truncf128 => ("truncf128", 1, fx.tcx.types.f128, types::F128),
        sym::round_ties_even_f16 => ("rintf16", 1, fx.tcx.types.f16, types::F16),
        sym::round_ties_even_f32 => ("rintf", 1, fx.tcx.types.f32, types::F32),
        sym::round_ties_even_f64 => ("rint", 1, fx.tcx.types.f64, types::F64),
        sym::round_ties_even_f128 => ("rintf128", 1, fx.tcx.types.f128, types::F128),
        sym::roundf16 => ("roundf16", 1, fx.tcx.types.f16, types::F16),
        sym::roundf32 => ("roundf", 1, fx.tcx.types.f32, types::F32),
        sym::roundf64 => ("round", 1, fx.tcx.types.f64, types::F64),
        sym::roundf128 => ("roundf128", 1, fx.tcx.types.f128, types::F128),
        sym::sinf16 => ("sinf16", 1, fx.tcx.types.f16, types::F16),
        sym::sinf32 => ("sinf", 1, fx.tcx.types.f32, types::F32),
        sym::sinf64 => ("sin", 1, fx.tcx.types.f64, types::F64),
        sym::sinf128 => ("sinf128", 1, fx.tcx.types.f128, types::F128),
        sym::cosf16 => ("cosf16", 1, fx.tcx.types.f16, types::F16),
        sym::cosf32 => ("cosf", 1, fx.tcx.types.f32, types::F32),
        sym::cosf64 => ("cos", 1, fx.tcx.types.f64, types::F64),
        sym::cosf128 => ("cosf128", 1, fx.tcx.types.f128, types::F128),
        _ => return false,
    };

    if args.len() != arg_count {
        bug!("wrong number of args for intrinsic {:?}", intrinsic);
    }

    let (a, b, c);
    let args = match args {
        [x] => {
            a = [codegen_operand(fx, &x.node).load_scalar(fx)];
            &a as &[_]
        }
        [x, y] => {
            b = [
                codegen_operand(fx, &x.node).load_scalar(fx),
                codegen_operand(fx, &y.node).load_scalar(fx),
            ];
            &b
        }
        [x, y, z] => {
            c = [
                codegen_operand(fx, &x.node).load_scalar(fx),
                codegen_operand(fx, &y.node).load_scalar(fx),
                codegen_operand(fx, &z.node).load_scalar(fx),
            ];
            &c
        }
        _ => unreachable!(),
    };

    let layout = fx.layout_of(ty);
    // FIXME(bytecodealliance/wasmtime#8312): Use native Cranelift operations
    // for `f16` and `f128` once the lowerings have been implemented in Cranelift.
    let res = match intrinsic {
        sym::fmaf16 | sym::fmuladdf16 => {
            CValue::by_val(codegen_f16_f128::fma_f16(fx, args[0], args[1], args[2]), layout)
        }
        sym::fmaf32 | sym::fmaf64 | sym::fmuladdf32 | sym::fmuladdf64 => {
            CValue::by_val(fx.bcx.ins().fma(args[0], args[1], args[2]), layout)
        }
        sym::copysignf16 => {
            CValue::by_val(codegen_f16_f128::copysign_f16(fx, args[0], args[1]), layout)
        }
        sym::copysignf128 => {
            CValue::by_val(codegen_f16_f128::copysign_f128(fx, args[0], args[1]), layout)
        }
        sym::copysignf32 | sym::copysignf64 => {
            CValue::by_val(fx.bcx.ins().fcopysign(args[0], args[1]), layout)
        }
        sym::fabsf16 => CValue::by_val(codegen_f16_f128::abs_f16(fx, args[0]), layout),
        sym::fabsf128 => CValue::by_val(codegen_f16_f128::abs_f128(fx, args[0]), layout),
        sym::fabsf32
        | sym::fabsf64
        | sym::floorf32
        | sym::floorf64
        | sym::ceilf32
        | sym::ceilf64
        | sym::truncf32
        | sym::truncf64
        | sym::round_ties_even_f32
        | sym::round_ties_even_f64
        | sym::sqrtf32
        | sym::sqrtf64 => {
            let val = match intrinsic {
                sym::fabsf32 | sym::fabsf64 => fx.bcx.ins().fabs(args[0]),
                sym::floorf32 | sym::floorf64 => fx.bcx.ins().floor(args[0]),
                sym::ceilf32 | sym::ceilf64 => fx.bcx.ins().ceil(args[0]),
                sym::truncf32 | sym::truncf64 => fx.bcx.ins().trunc(args[0]),
                sym::round_ties_even_f32 | sym::round_ties_even_f64 => {
                    fx.bcx.ins().nearest(args[0])
                }
                sym::sqrtf32 | sym::sqrtf64 => fx.bcx.ins().sqrt(args[0]),
                _ => unreachable!(),
            };

            CValue::by_val(val, layout)
        }

        // These intrinsics aren't supported natively by Cranelift.
        // Lower them to a libcall.
        sym::powif16 | sym::powif32 | sym::powif64 | sym::powif128 => {
            let temp;
            let (clif_ty, args) = if intrinsic == sym::powif16 {
                temp = [codegen_f16_f128::f16_to_f32(fx, args[0]), args[1]];
                (types::F32, temp.as_slice())
            } else {
                (clif_ty, args)
            };
            let input_tys: Vec<_> =
                vec![AbiParam::new(clif_ty), lib_call_arg_param(fx.tcx, types::I32, true)];
            let ret_val = fx.lib_call(name, input_tys, vec![AbiParam::new(clif_ty)], &args)[0];
            let ret_val = if intrinsic == sym::powif16 {
                codegen_f16_f128::f32_to_f16(fx, ret_val)
            } else {
                ret_val
            };
            CValue::by_val(ret_val, fx.layout_of(ty))
        }
        sym::powf16 => {
            // FIXME(f16_f128): Rust `compiler-builtins` doesn't export `powf16` yet.
            let x = codegen_f16_f128::f16_to_f32(fx, args[0]);
            let y = codegen_f16_f128::f16_to_f32(fx, args[1]);
            let ret_val = fx.lib_call(
                "powf",
                vec![AbiParam::new(types::F32), AbiParam::new(types::F32)],
                vec![AbiParam::new(types::F32)],
                &[x, y],
            )[0];
            CValue::by_val(codegen_f16_f128::f32_to_f16(fx, ret_val), fx.layout_of(ty))
        }
        _ => {
            let input_tys: Vec<_> = args.iter().map(|_| AbiParam::new(clif_ty)).collect();
            let ret_val = fx.lib_call(name, input_tys, vec![AbiParam::new(clif_ty)], &args)[0];
            CValue::by_val(ret_val, fx.layout_of(ty))
        }
    };

    ret.write_cvalue(fx, res);

    true
}

fn codegen_regular_intrinsic_call<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    instance: Instance<'tcx>,
    intrinsic: Symbol,
    generic_args: GenericArgsRef<'tcx>,
    args: &[Spanned<mir::Operand<'tcx>>],
    ret: CPlace<'tcx>,
    destination: Option<BasicBlock>,
    source_info: mir::SourceInfo,
) -> Result<(), Instance<'tcx>> {
    assert_eq!(generic_args, instance.args);
    let usize_layout = fx.layout_of(fx.tcx.types.usize);

    match intrinsic {
        sym::abort => {
            fx.bcx.set_cold_block(fx.bcx.current_block().unwrap());
            fx.bcx.ins().trap(TrapCode::user(2).unwrap());
            return Ok(());
        }
        sym::breakpoint => {
            intrinsic_args!(fx, args => (); intrinsic);

            fx.bcx.ins().debugtrap();
        }
        sym::copy => {
            intrinsic_args!(fx, args => (src, dst, count); intrinsic);
            let src = src.load_scalar(fx);
            let dst = dst.load_scalar(fx);
            let count = count.load_scalar(fx);

            let elem_ty = generic_args.type_at(0);
            let elem_size: u64 = fx.layout_of(elem_ty).size.bytes();
            assert_eq!(args.len(), 3);
            let byte_amount =
                if elem_size != 1 { fx.bcx.ins().imul_imm(count, elem_size as i64) } else { count };

            // FIXME emit_small_memmove
            fx.bcx.call_memmove(fx.target_config, dst, src, byte_amount);
        }
        sym::volatile_copy_memory | sym::volatile_copy_nonoverlapping_memory => {
            // NOTE: the volatile variants have src and dst swapped
            intrinsic_args!(fx, args => (dst, src, count); intrinsic);
            let dst = dst.load_scalar(fx);
            let src = src.load_scalar(fx);
            let count = count.load_scalar(fx);

            let elem_ty = generic_args.type_at(0);
            let elem_size: u64 = fx.layout_of(elem_ty).size.bytes();
            assert_eq!(args.len(), 3);
            let byte_amount =
                if elem_size != 1 { fx.bcx.ins().imul_imm(count, elem_size as i64) } else { count };

            // FIXME make the copy actually volatile when using emit_small_mem{cpy,move}
            if intrinsic == sym::volatile_copy_nonoverlapping_memory {
                // FIXME emit_small_memcpy
                fx.bcx.call_memcpy(fx.target_config, dst, src, byte_amount);
            } else {
                // FIXME emit_small_memmove
                fx.bcx.call_memmove(fx.target_config, dst, src, byte_amount);
            }
        }
        sym::size_of_val => {
            intrinsic_args!(fx, args => (ptr); intrinsic);

            let layout = fx.layout_of(generic_args.type_at(0));
            // Note: Can't use is_unsized here as truly unsized types need to take the fixed size
            // branch
            let meta = if let BackendRepr::ScalarPair(_, _) = ptr.layout().backend_repr {
                Some(ptr.load_scalar_pair(fx).1)
            } else {
                None
            };
            let (size, _align) = crate::unsize::size_and_align_of(fx, layout, meta);
            ret.write_cvalue(fx, CValue::by_val(size, usize_layout));
        }
        sym::min_align_of_val => {
            intrinsic_args!(fx, args => (ptr); intrinsic);

            let layout = fx.layout_of(generic_args.type_at(0));
            // Note: Can't use is_unsized here as truly unsized types need to take the fixed size
            // branch
            let meta = if let BackendRepr::ScalarPair(_, _) = ptr.layout().backend_repr {
                Some(ptr.load_scalar_pair(fx).1)
            } else {
                None
            };
            let (_size, align) = crate::unsize::size_and_align_of(fx, layout, meta);
            ret.write_cvalue(fx, CValue::by_val(align, usize_layout));
        }

        sym::vtable_size => {
            intrinsic_args!(fx, args => (vtable); intrinsic);
            let vtable = vtable.load_scalar(fx);

            let size = crate::vtable::size_of_obj(fx, vtable);
            ret.write_cvalue(fx, CValue::by_val(size, usize_layout));
        }

        sym::vtable_align => {
            intrinsic_args!(fx, args => (vtable); intrinsic);
            let vtable = vtable.load_scalar(fx);

            let align = crate::vtable::min_align_of_obj(fx, vtable);
            ret.write_cvalue(fx, CValue::by_val(align, usize_layout));
        }

        sym::exact_div => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            // FIXME trap on inexact
            let res = crate::num::codegen_int_binop(fx, BinOp::Div, x, y);
            ret.write_cvalue(fx, res);
        }
        sym::saturating_add | sym::saturating_sub => {
            intrinsic_args!(fx, args => (lhs, rhs); intrinsic);

            assert_eq!(lhs.layout().ty, rhs.layout().ty);
            let bin_op = match intrinsic {
                sym::saturating_add => BinOp::Add,
                sym::saturating_sub => BinOp::Sub,
                _ => unreachable!(),
            };

            let res = crate::num::codegen_saturating_int_binop(fx, bin_op, lhs, rhs);
            ret.write_cvalue(fx, res);
        }
        sym::rotate_left => {
            intrinsic_args!(fx, args => (x, y); intrinsic);
            let y = y.load_scalar(fx);

            let layout = x.layout();
            let x = x.load_scalar(fx);
            let res = fx.bcx.ins().rotl(x, y);
            ret.write_cvalue(fx, CValue::by_val(res, layout));
        }
        sym::rotate_right => {
            intrinsic_args!(fx, args => (x, y); intrinsic);
            let y = y.load_scalar(fx);

            let layout = x.layout();
            let x = x.load_scalar(fx);
            let res = fx.bcx.ins().rotr(x, y);
            ret.write_cvalue(fx, CValue::by_val(res, layout));
        }

        // The only difference between offset and arith_offset is regarding UB. Because Cranelift
        // doesn't have UB both are codegen'ed the same way
        sym::arith_offset => {
            intrinsic_args!(fx, args => (base, offset); intrinsic);
            let offset = offset.load_scalar(fx);

            let pointee_ty = base.layout().ty.builtin_deref(true).unwrap();
            let pointee_size = fx.layout_of(pointee_ty).size.bytes();
            let ptr_diff = if pointee_size != 1 {
                fx.bcx.ins().imul_imm(offset, pointee_size as i64)
            } else {
                offset
            };
            let base_val = base.load_scalar(fx);
            let res = fx.bcx.ins().iadd(base_val, ptr_diff);
            ret.write_cvalue(fx, CValue::by_val(res, base.layout()));
        }

        sym::ptr_mask => {
            intrinsic_args!(fx, args => (ptr, mask); intrinsic);
            let ptr_layout = ptr.layout();
            let ptr = ptr.load_scalar(fx);
            let mask = mask.load_scalar(fx);
            let res = fx.bcx.ins().band(ptr, mask);
            ret.write_cvalue(fx, CValue::by_val(res, ptr_layout));
        }

        sym::write_bytes | sym::volatile_set_memory => {
            intrinsic_args!(fx, args => (dst, val, count); intrinsic);
            let val = val.load_scalar(fx);
            let count = count.load_scalar(fx);

            let pointee_ty = dst.layout().ty.builtin_deref(true).unwrap();
            let pointee_size = fx.layout_of(pointee_ty).size.bytes();
            let count = if pointee_size != 1 {
                fx.bcx.ins().imul_imm(count, pointee_size as i64)
            } else {
                count
            };
            let dst_ptr = dst.load_scalar(fx);
            // FIXME make the memset actually volatile when switching to emit_small_memset
            // FIXME use emit_small_memset
            fx.bcx.call_memset(fx.target_config, dst_ptr, val, count);
        }
        sym::ctlz | sym::ctlz_nonzero => {
            intrinsic_args!(fx, args => (arg); intrinsic);
            let val = arg.load_scalar(fx);

            // FIXME trap on `ctlz_nonzero` with zero arg.
            let res = fx.bcx.ins().clz(val);
            let res = clif_intcast(fx, res, types::I32, false);
            let res = CValue::by_val(res, ret.layout());
            ret.write_cvalue(fx, res);
        }
        sym::cttz | sym::cttz_nonzero => {
            intrinsic_args!(fx, args => (arg); intrinsic);
            let val = arg.load_scalar(fx);

            // FIXME trap on `cttz_nonzero` with zero arg.
            let res = fx.bcx.ins().ctz(val);
            let res = clif_intcast(fx, res, types::I32, false);
            let res = CValue::by_val(res, ret.layout());
            ret.write_cvalue(fx, res);
        }
        sym::ctpop => {
            intrinsic_args!(fx, args => (arg); intrinsic);
            let val = arg.load_scalar(fx);

            let res = fx.bcx.ins().popcnt(val);
            let res = clif_intcast(fx, res, types::I32, false);
            let res = CValue::by_val(res, ret.layout());
            ret.write_cvalue(fx, res);
        }
        sym::bitreverse => {
            intrinsic_args!(fx, args => (arg); intrinsic);
            let val = arg.load_scalar(fx);

            let res = fx.bcx.ins().bitrev(val);
            let res = CValue::by_val(res, arg.layout());
            ret.write_cvalue(fx, res);
        }
        sym::bswap => {
            intrinsic_args!(fx, args => (arg); intrinsic);
            let val = arg.load_scalar(fx);

            let res = if fx.bcx.func.dfg.value_type(val) == types::I8 {
                val
            } else {
                fx.bcx.ins().bswap(val)
            };
            let res = CValue::by_val(res, arg.layout());
            ret.write_cvalue(fx, res);
        }
        sym::assert_inhabited | sym::assert_zero_valid | sym::assert_mem_uninitialized_valid => {
            intrinsic_args!(fx, args => (); intrinsic);

            let ty = generic_args.type_at(0);

            let requirement = ValidityRequirement::from_intrinsic(intrinsic);

            if let Some(requirement) = requirement {
                let do_panic = !fx
                    .tcx
                    .check_validity_requirement((
                        requirement,
                        ty::TypingEnv::fully_monomorphized().as_query_input(ty),
                    ))
                    .expect("expect to have layout during codegen");

                if do_panic {
                    let layout = fx.layout_of(ty);
                    let msg_str = with_no_visible_paths!({
                        with_no_trimmed_paths!({
                            if layout.is_uninhabited() {
                                // Use this error even for the other intrinsics as it is more precise.
                                format!("attempted to instantiate uninhabited type `{}`", ty)
                            } else if intrinsic == sym::assert_zero_valid {
                                format!(
                                    "attempted to zero-initialize type `{}`, which is invalid",
                                    ty
                                )
                            } else {
                                format!(
                                    "attempted to leave type `{}` uninitialized, which is invalid",
                                    ty
                                )
                            }
                        })
                    });
                    crate::base::codegen_panic_nounwind(fx, &msg_str, Some(source_info.span));
                    return Ok(());
                }
            }
        }

        sym::volatile_load | sym::unaligned_volatile_load => {
            intrinsic_args!(fx, args => (ptr); intrinsic);

            // Cranelift treats loads as volatile by default
            // FIXME correctly handle unaligned_volatile_load
            let inner_layout = fx.layout_of(ptr.layout().ty.builtin_deref(true).unwrap());
            let val = CValue::by_ref(Pointer::new(ptr.load_scalar(fx)), inner_layout);
            ret.write_cvalue(fx, val);
        }
        sym::volatile_store | sym::unaligned_volatile_store | sym::nontemporal_store => {
            intrinsic_args!(fx, args => (ptr, val); intrinsic);
            let ptr = ptr.load_scalar(fx);

            // Cranelift treats stores as volatile by default
            // FIXME correctly handle unaligned_volatile_store
            // FIXME actually do nontemporal stores if requested (but do not just emit MOVNT on x86;
            // see the LLVM backend for details)
            let dest = CPlace::for_ptr(Pointer::new(ptr), val.layout());
            dest.write_cvalue(fx, val);
        }

        sym::pref_align_of
        | sym::needs_drop
        | sym::type_id
        | sym::type_name
        | sym::variant_count => {
            intrinsic_args!(fx, args => (); intrinsic);

            let const_val = fx
                .tcx
                .const_eval_instance(
                    ty::TypingEnv::fully_monomorphized(),
                    instance,
                    source_info.span,
                )
                .unwrap();
            let val = crate::constant::codegen_const_value(fx, const_val, ret.layout().ty);
            ret.write_cvalue(fx, val);
        }

        sym::ptr_offset_from | sym::ptr_offset_from_unsigned => {
            intrinsic_args!(fx, args => (ptr, base); intrinsic);
            let ptr = ptr.load_scalar(fx);
            let base = base.load_scalar(fx);
            let ty = generic_args.type_at(0);

            let pointee_size: u64 = fx.layout_of(ty).size.bytes();
            let diff_bytes = fx.bcx.ins().isub(ptr, base);
            // FIXME this can be an exact division.
            let val = if intrinsic == sym::ptr_offset_from_unsigned {
                let usize_layout = fx.layout_of(fx.tcx.types.usize);
                // Because diff_bytes ULE isize::MAX, this would be fine as signed,
                // but unsigned is slightly easier to codegen, so might as well.
                CValue::by_val(fx.bcx.ins().udiv_imm(diff_bytes, pointee_size as i64), usize_layout)
            } else {
                let isize_layout = fx.layout_of(fx.tcx.types.isize);
                CValue::by_val(fx.bcx.ins().sdiv_imm(diff_bytes, pointee_size as i64), isize_layout)
            };
            ret.write_cvalue(fx, val);
        }

        sym::caller_location => {
            intrinsic_args!(fx, args => (); intrinsic);

            let caller_location = fx.get_caller_location(source_info);
            ret.write_cvalue(fx, caller_location);
        }

        _ if intrinsic.as_str().starts_with("atomic_fence") => {
            intrinsic_args!(fx, args => (); intrinsic);

            fx.bcx.ins().fence();
        }
        _ if intrinsic.as_str().starts_with("atomic_singlethreadfence") => {
            intrinsic_args!(fx, args => (); intrinsic);

            // FIXME use a compiler fence once Cranelift supports it
            fx.bcx.ins().fence();
        }
        _ if intrinsic.as_str().starts_with("atomic_load") => {
            intrinsic_args!(fx, args => (ptr); intrinsic);
            let ptr = ptr.load_scalar(fx);

            let ty = generic_args.type_at(0);
            match ty.kind() {
                ty::Uint(UintTy::U128) | ty::Int(IntTy::I128) => {
                    // FIXME implement 128bit atomics
                    if fx.tcx.is_compiler_builtins(LOCAL_CRATE) {
                        // special case for compiler-builtins to avoid having to patch it
                        crate::base::codegen_panic_nounwind(
                            fx,
                            "128bit atomics not yet supported",
                            None,
                        );
                        return Ok(());
                    } else {
                        fx.tcx
                            .dcx()
                            .span_fatal(source_info.span, "128bit atomics not yet supported");
                    }
                }
                ty::Uint(_) | ty::Int(_) | ty::RawPtr(..) => {}
                _ => {
                    report_atomic_type_validation_error(fx, intrinsic, source_info.span, ty);
                    return Ok(());
                }
            }
            let clif_ty = fx.clif_type(ty).unwrap();

            let val = fx.bcx.ins().atomic_load(clif_ty, MemFlags::trusted(), ptr);

            let val = CValue::by_val(val, fx.layout_of(ty));
            ret.write_cvalue(fx, val);
        }
        _ if intrinsic.as_str().starts_with("atomic_store") => {
            intrinsic_args!(fx, args => (ptr, val); intrinsic);
            let ptr = ptr.load_scalar(fx);

            let ty = generic_args.type_at(0);
            match ty.kind() {
                ty::Uint(UintTy::U128) | ty::Int(IntTy::I128) => {
                    // FIXME implement 128bit atomics
                    if fx.tcx.is_compiler_builtins(LOCAL_CRATE) {
                        // special case for compiler-builtins to avoid having to patch it
                        crate::base::codegen_panic_nounwind(
                            fx,
                            "128bit atomics not yet supported",
                            None,
                        );
                        return Ok(());
                    } else {
                        fx.tcx
                            .dcx()
                            .span_fatal(source_info.span, "128bit atomics not yet supported");
                    }
                }
                ty::Uint(_) | ty::Int(_) | ty::RawPtr(..) => {}
                _ => {
                    report_atomic_type_validation_error(fx, intrinsic, source_info.span, ty);
                    return Ok(());
                }
            }

            let val = val.load_scalar(fx);

            fx.bcx.ins().atomic_store(MemFlags::trusted(), val, ptr);
        }
        _ if intrinsic.as_str().starts_with("atomic_xchg") => {
            intrinsic_args!(fx, args => (ptr, new); intrinsic);
            let ptr = ptr.load_scalar(fx);

            let layout = new.layout();
            match layout.ty.kind() {
                ty::Uint(_) | ty::Int(_) | ty::RawPtr(..) => {}
                _ => {
                    report_atomic_type_validation_error(fx, intrinsic, source_info.span, layout.ty);
                    return Ok(());
                }
            }
            let ty = fx.clif_type(layout.ty).unwrap();

            let new = new.load_scalar(fx);

            let old = fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Xchg, ptr, new);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        }
        _ if intrinsic.as_str().starts_with("atomic_cxchg") => {
            // both atomic_cxchg_* and atomic_cxchgweak_*
            intrinsic_args!(fx, args => (ptr, test_old, new); intrinsic);
            let ptr = ptr.load_scalar(fx);

            let layout = new.layout();
            match layout.ty.kind() {
                ty::Uint(_) | ty::Int(_) | ty::RawPtr(..) => {}
                _ => {
                    report_atomic_type_validation_error(fx, intrinsic, source_info.span, layout.ty);
                    return Ok(());
                }
            }

            let test_old = test_old.load_scalar(fx);
            let new = new.load_scalar(fx);

            let old = fx.bcx.ins().atomic_cas(MemFlags::trusted(), ptr, test_old, new);
            let is_eq = fx.bcx.ins().icmp(IntCC::Equal, old, test_old);

            let ret_val = CValue::by_val_pair(old, is_eq, ret.layout());
            ret.write_cvalue(fx, ret_val)
        }

        _ if intrinsic.as_str().starts_with("atomic_xadd") => {
            intrinsic_args!(fx, args => (ptr, amount); intrinsic);
            let ptr = ptr.load_scalar(fx);

            let layout = amount.layout();
            match layout.ty.kind() {
                ty::Uint(_) | ty::Int(_) | ty::RawPtr(..) => {}
                _ => {
                    report_atomic_type_validation_error(fx, intrinsic, source_info.span, layout.ty);
                    return Ok(());
                }
            }
            let ty = fx.clif_type(layout.ty).unwrap();

            let amount = amount.load_scalar(fx);

            let old =
                fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Add, ptr, amount);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        }
        _ if intrinsic.as_str().starts_with("atomic_xsub") => {
            intrinsic_args!(fx, args => (ptr, amount); intrinsic);
            let ptr = ptr.load_scalar(fx);

            let layout = amount.layout();
            match layout.ty.kind() {
                ty::Uint(_) | ty::Int(_) | ty::RawPtr(..) => {}
                _ => {
                    report_atomic_type_validation_error(fx, intrinsic, source_info.span, layout.ty);
                    return Ok(());
                }
            }
            let ty = fx.clif_type(layout.ty).unwrap();

            let amount = amount.load_scalar(fx);

            let old =
                fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Sub, ptr, amount);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        }
        _ if intrinsic.as_str().starts_with("atomic_and") => {
            intrinsic_args!(fx, args => (ptr, src); intrinsic);
            let ptr = ptr.load_scalar(fx);

            let layout = src.layout();
            match layout.ty.kind() {
                ty::Uint(_) | ty::Int(_) | ty::RawPtr(..) => {}
                _ => {
                    report_atomic_type_validation_error(fx, intrinsic, source_info.span, layout.ty);
                    return Ok(());
                }
            }
            let ty = fx.clif_type(layout.ty).unwrap();

            let src = src.load_scalar(fx);

            let old = fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::And, ptr, src);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        }
        _ if intrinsic.as_str().starts_with("atomic_or") => {
            intrinsic_args!(fx, args => (ptr, src); intrinsic);
            let ptr = ptr.load_scalar(fx);

            let layout = src.layout();
            match layout.ty.kind() {
                ty::Uint(_) | ty::Int(_) | ty::RawPtr(..) => {}
                _ => {
                    report_atomic_type_validation_error(fx, intrinsic, source_info.span, layout.ty);
                    return Ok(());
                }
            }
            let ty = fx.clif_type(layout.ty).unwrap();

            let src = src.load_scalar(fx);

            let old = fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Or, ptr, src);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        }
        _ if intrinsic.as_str().starts_with("atomic_xor") => {
            intrinsic_args!(fx, args => (ptr, src); intrinsic);
            let ptr = ptr.load_scalar(fx);

            let layout = src.layout();
            match layout.ty.kind() {
                ty::Uint(_) | ty::Int(_) | ty::RawPtr(..) => {}
                _ => {
                    report_atomic_type_validation_error(fx, intrinsic, source_info.span, layout.ty);
                    return Ok(());
                }
            }
            let ty = fx.clif_type(layout.ty).unwrap();

            let src = src.load_scalar(fx);

            let old = fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Xor, ptr, src);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        }
        _ if intrinsic.as_str().starts_with("atomic_nand") => {
            intrinsic_args!(fx, args => (ptr, src); intrinsic);
            let ptr = ptr.load_scalar(fx);

            let layout = src.layout();
            match layout.ty.kind() {
                ty::Uint(_) | ty::Int(_) | ty::RawPtr(..) => {}
                _ => {
                    report_atomic_type_validation_error(fx, intrinsic, source_info.span, layout.ty);
                    return Ok(());
                }
            }
            let ty = fx.clif_type(layout.ty).unwrap();

            let src = src.load_scalar(fx);

            let old = fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Nand, ptr, src);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        }
        _ if intrinsic.as_str().starts_with("atomic_max") => {
            intrinsic_args!(fx, args => (ptr, src); intrinsic);
            let ptr = ptr.load_scalar(fx);

            let layout = src.layout();
            match layout.ty.kind() {
                ty::Int(_) => {}
                _ => {
                    report_atomic_type_validation_error(fx, intrinsic, source_info.span, layout.ty);
                    return Ok(());
                }
            }
            let ty = fx.clif_type(layout.ty).unwrap();

            let src = src.load_scalar(fx);

            let old = fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Smax, ptr, src);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        }
        _ if intrinsic.as_str().starts_with("atomic_umax") => {
            intrinsic_args!(fx, args => (ptr, src); intrinsic);
            let ptr = ptr.load_scalar(fx);

            let layout = src.layout();
            match layout.ty.kind() {
                ty::Uint(_) => {}
                _ => {
                    report_atomic_type_validation_error(fx, intrinsic, source_info.span, layout.ty);
                    return Ok(());
                }
            }
            let ty = fx.clif_type(layout.ty).unwrap();

            let src = src.load_scalar(fx);

            let old = fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Umax, ptr, src);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        }
        _ if intrinsic.as_str().starts_with("atomic_min") => {
            intrinsic_args!(fx, args => (ptr, src); intrinsic);
            let ptr = ptr.load_scalar(fx);

            let layout = src.layout();
            match layout.ty.kind() {
                ty::Int(_) => {}
                _ => {
                    report_atomic_type_validation_error(fx, intrinsic, source_info.span, layout.ty);
                    return Ok(());
                }
            }
            let ty = fx.clif_type(layout.ty).unwrap();

            let src = src.load_scalar(fx);

            let old = fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Smin, ptr, src);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        }
        _ if intrinsic.as_str().starts_with("atomic_umin") => {
            intrinsic_args!(fx, args => (ptr, src); intrinsic);
            let ptr = ptr.load_scalar(fx);

            let layout = src.layout();
            match layout.ty.kind() {
                ty::Uint(_) => {}
                _ => {
                    report_atomic_type_validation_error(fx, intrinsic, source_info.span, layout.ty);
                    return Ok(());
                }
            }
            let ty = fx.clif_type(layout.ty).unwrap();

            let src = src.load_scalar(fx);

            let old = fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Umin, ptr, src);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        }

        sym::minimumf16 => {
            intrinsic_args!(fx, args => (a, b); intrinsic);
            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            // FIXME(bytecodealliance/wasmtime#8312): Use `fmin` directly once
            // Cranelift backend lowerings are implemented.
            let a = codegen_f16_f128::f16_to_f32(fx, a);
            let b = codegen_f16_f128::f16_to_f32(fx, b);
            let val = fx.bcx.ins().fmin(a, b);
            let val = codegen_f16_f128::f32_to_f16(fx, val);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f16));
            ret.write_cvalue(fx, val);
        }
        sym::minimumf32 => {
            intrinsic_args!(fx, args => (a, b); intrinsic);
            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            let val = fx.bcx.ins().fmin(a, b);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f32));
            ret.write_cvalue(fx, val);
        }
        sym::minimumf64 => {
            intrinsic_args!(fx, args => (a, b); intrinsic);
            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            let val = fx.bcx.ins().fmin(a, b);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f64));
            ret.write_cvalue(fx, val);
        }
        sym::minimumf128 => {
            intrinsic_args!(fx, args => (a, b); intrinsic);
            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            // FIXME(bytecodealliance/wasmtime#8312): Use `fmin` once  Cranelift
            // backend lowerings are implemented.
            let val = codegen_f16_f128::fmin_f128(fx, a, b);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f128));
            ret.write_cvalue(fx, val);
        }
        sym::maximumf16 => {
            intrinsic_args!(fx, args => (a, b); intrinsic);
            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            // FIXME(bytecodealliance/wasmtime#8312): Use `fmax` directly once
            // Cranelift backend lowerings are implemented.
            let a = codegen_f16_f128::f16_to_f32(fx, a);
            let b = codegen_f16_f128::f16_to_f32(fx, b);
            let val = fx.bcx.ins().fmax(a, b);
            let val = codegen_f16_f128::f32_to_f16(fx, val);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f16));
            ret.write_cvalue(fx, val);
        }
        sym::maximumf32 => {
            intrinsic_args!(fx, args => (a, b); intrinsic);
            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            let val = fx.bcx.ins().fmax(a, b);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f32));
            ret.write_cvalue(fx, val);
        }
        sym::maximumf64 => {
            intrinsic_args!(fx, args => (a, b); intrinsic);
            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            let val = fx.bcx.ins().fmax(a, b);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f64));
            ret.write_cvalue(fx, val);
        }
        sym::maximumf128 => {
            intrinsic_args!(fx, args => (a, b); intrinsic);
            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            // FIXME(bytecodealliance/wasmtime#8312): Use `fmax` once  Cranelift
            // backend lowerings are implemented.
            let val = codegen_f16_f128::fmax_f128(fx, a, b);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f128));
            ret.write_cvalue(fx, val);
        }

        sym::minnumf16 => {
            intrinsic_args!(fx, args => (a, b); intrinsic);
            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            let val = crate::num::codegen_float_min(fx, a, b);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f16));
            ret.write_cvalue(fx, val);
        }
        sym::minnumf32 => {
            intrinsic_args!(fx, args => (a, b); intrinsic);
            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            let val = crate::num::codegen_float_min(fx, a, b);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f32));
            ret.write_cvalue(fx, val);
        }
        sym::minnumf64 => {
            intrinsic_args!(fx, args => (a, b); intrinsic);
            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            let val = crate::num::codegen_float_min(fx, a, b);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f64));
            ret.write_cvalue(fx, val);
        }
        sym::minnumf128 => {
            intrinsic_args!(fx, args => (a, b); intrinsic);
            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            let val = crate::num::codegen_float_min(fx, a, b);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f128));
            ret.write_cvalue(fx, val);
        }
        sym::maxnumf16 => {
            intrinsic_args!(fx, args => (a, b); intrinsic);
            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            let val = crate::num::codegen_float_max(fx, a, b);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f16));
            ret.write_cvalue(fx, val);
        }
        sym::maxnumf32 => {
            intrinsic_args!(fx, args => (a, b); intrinsic);
            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            let val = crate::num::codegen_float_max(fx, a, b);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f32));
            ret.write_cvalue(fx, val);
        }
        sym::maxnumf64 => {
            intrinsic_args!(fx, args => (a, b); intrinsic);
            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            let val = crate::num::codegen_float_max(fx, a, b);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f64));
            ret.write_cvalue(fx, val);
        }
        sym::maxnumf128 => {
            intrinsic_args!(fx, args => (a, b); intrinsic);
            let a = a.load_scalar(fx);
            let b = b.load_scalar(fx);

            let val = crate::num::codegen_float_max(fx, a, b);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f128));
            ret.write_cvalue(fx, val);
        }

        sym::catch_unwind => {
            intrinsic_args!(fx, args => (f, data, catch_fn); intrinsic);
            let f = f.load_scalar(fx);
            let data = data.load_scalar(fx);
            let _catch_fn = catch_fn.load_scalar(fx);

            // FIXME once unwinding is supported, change this to actually catch panics
            let f_sig = fx.bcx.func.import_signature(Signature {
                call_conv: fx.target_config.default_call_conv,
                params: vec![AbiParam::new(pointer_ty(fx.tcx))],
                returns: vec![],
            });

            fx.bcx.ins().call_indirect(f_sig, f, &[data]);

            let layout = fx.layout_of(fx.tcx.types.i32);
            let ret_val = CValue::by_val(fx.bcx.ins().iconst(types::I32, 0), layout);
            ret.write_cvalue(fx, ret_val);
        }

        sym::fadd_fast
        | sym::fsub_fast
        | sym::fmul_fast
        | sym::fdiv_fast
        | sym::frem_fast
        | sym::fadd_algebraic
        | sym::fsub_algebraic
        | sym::fmul_algebraic
        | sym::fdiv_algebraic
        | sym::frem_algebraic => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            let res = crate::num::codegen_float_binop(
                fx,
                match intrinsic {
                    sym::fadd_fast | sym::fadd_algebraic => BinOp::Add,
                    sym::fsub_fast | sym::fsub_algebraic => BinOp::Sub,
                    sym::fmul_fast | sym::fmul_algebraic => BinOp::Mul,
                    sym::fdiv_fast | sym::fdiv_algebraic => BinOp::Div,
                    sym::frem_fast | sym::frem_algebraic => BinOp::Rem,
                    _ => unreachable!(),
                },
                x,
                y,
            );
            ret.write_cvalue(fx, res);
        }
        sym::float_to_int_unchecked => {
            intrinsic_args!(fx, args => (f); intrinsic);
            let f = f.load_scalar(fx);

            let res = crate::cast::clif_int_or_float_cast(
                fx,
                f,
                false,
                fx.clif_type(ret.layout().ty).unwrap(),
                type_sign(ret.layout().ty),
            );
            ret.write_cvalue(fx, CValue::by_val(res, ret.layout()));
        }

        sym::raw_eq => {
            intrinsic_args!(fx, args => (lhs_ref, rhs_ref); intrinsic);
            let lhs_ref = lhs_ref.load_scalar(fx);
            let rhs_ref = rhs_ref.load_scalar(fx);

            let size = fx.layout_of(generic_args.type_at(0)).layout.size();
            // FIXME add and use emit_small_memcmp
            let is_eq_value = if size == Size::ZERO {
                // No bytes means they're trivially equal
                fx.bcx.ins().iconst(types::I8, 1)
            } else if let Some(clty) = size.bits().try_into().ok().and_then(Type::int) {
                // Can't use `trusted` for these loads; they could be unaligned.
                let mut flags = MemFlags::new();
                flags.set_notrap();
                let lhs_val = fx.bcx.ins().load(clty, flags, lhs_ref, 0);
                let rhs_val = fx.bcx.ins().load(clty, flags, rhs_ref, 0);
                fx.bcx.ins().icmp(IntCC::Equal, lhs_val, rhs_val)
            } else {
                // Just call `memcmp` (like slices do in core) when the
                // size is too large or it's not a power-of-two.
                let signed_bytes = i64::try_from(size.bytes()).unwrap();
                let bytes_val = fx.bcx.ins().iconst(fx.pointer_type, signed_bytes);
                let params = vec![AbiParam::new(fx.pointer_type); 3];
                let returns = vec![AbiParam::new(types::I32)];
                let args = &[lhs_ref, rhs_ref, bytes_val];
                let cmp = fx.lib_call("memcmp", params, returns, args)[0];
                fx.bcx.ins().icmp_imm(IntCC::Equal, cmp, 0)
            };
            ret.write_cvalue(fx, CValue::by_val(is_eq_value, ret.layout()));
        }

        sym::compare_bytes => {
            intrinsic_args!(fx, args => (lhs_ptr, rhs_ptr, bytes_val); intrinsic);
            let lhs_ptr = lhs_ptr.load_scalar(fx);
            let rhs_ptr = rhs_ptr.load_scalar(fx);
            let bytes_val = bytes_val.load_scalar(fx);

            let params = vec![AbiParam::new(fx.pointer_type); 3];
            let returns = vec![AbiParam::new(types::I32)];
            let args = &[lhs_ptr, rhs_ptr, bytes_val];
            // Here we assume that the `memcmp` provided by the target is a NOP for size 0.
            let cmp = fx.lib_call("memcmp", params, returns, args)[0];
            ret.write_cvalue(fx, CValue::by_val(cmp, ret.layout()));
        }

        sym::black_box => {
            intrinsic_args!(fx, args => (a); intrinsic);

            // FIXME implement black_box semantics
            ret.write_cvalue(fx, a);
        }

        // FIXME implement variadics in cranelift
        sym::va_copy | sym::va_arg | sym::va_end => {
            fx.tcx.dcx().span_fatal(
                source_info.span,
                "Defining variadic functions is not yet supported by Cranelift",
            );
        }

        sym::cold_path => {
            fx.bcx.set_cold_block(fx.bcx.current_block().unwrap());
        }

        // Unimplemented intrinsics must have a fallback body. The fallback body is obtained
        // by converting the `InstanceKind::Intrinsic` to an `InstanceKind::Item`.
        _ => {
            let intrinsic = fx.tcx.intrinsic(instance.def_id()).unwrap();
            if intrinsic.must_be_overridden {
                span_bug!(
                    source_info.span,
                    "intrinsic {} must be overridden by codegen_cranelift, but isn't",
                    intrinsic.name,
                );
            }
            return Err(Instance::new_raw(instance.def_id(), instance.args));
        }
    }

    let ret_block = fx.get_block(destination.unwrap());
    fx.bcx.ins().jump(ret_block, &[]);
    Ok(())
}
