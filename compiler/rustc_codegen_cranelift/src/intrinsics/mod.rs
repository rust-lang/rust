//! Codegen of intrinsics. This includes `extern "rust-intrinsic"`, `extern "platform-intrinsic"`
//! and LLVM intrinsics that have symbol names starting with `llvm.`.

macro_rules! intrinsic_args {
    ($fx:expr, $args:expr => ($($arg:tt),*); $intrinsic:expr) => {
        #[allow(unused_parens)]
        let ($($arg),*) = if let [$($arg),*] = $args {
            ($(codegen_operand($fx, $arg)),*)
        } else {
            $crate::intrinsics::bug_on_incorrect_arg_count($intrinsic);
        };
    }
}

mod cpuid;
mod llvm;
mod llvm_aarch64;
mod llvm_x86;
mod simd;

pub(crate) use cpuid::codegen_cpuid_call;
pub(crate) use llvm::codegen_llvm_intrinsic_call;

use rustc_middle::ty;
use rustc_middle::ty::layout::{HasParamEnv, InitKind};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::subst::SubstsRef;
use rustc_span::symbol::{kw, sym, Symbol};

use crate::prelude::*;
use cranelift_codegen::ir::AtomicRmwOp;

fn bug_on_incorrect_arg_count(intrinsic: impl std::fmt::Display) -> ! {
    bug!("wrong number of args for intrinsic {}", intrinsic);
}

fn report_atomic_type_validation_error<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    intrinsic: Symbol,
    span: Span,
    ty: Ty<'tcx>,
) {
    fx.tcx.sess.span_err(
        span,
        &format!(
            "`{}` intrinsic: expected basic integer or raw pointer type, found `{:?}`",
            intrinsic, ty
        ),
    );
    // Prevent verifier error
    fx.bcx.ins().trap(TrapCode::UnreachableCodeReached);
}

pub(crate) fn clif_vector_type<'tcx>(tcx: TyCtxt<'tcx>, layout: TyAndLayout<'tcx>) -> Option<Type> {
    let (element, count) = match layout.abi {
        Abi::Vector { element, count } => (element, count),
        _ => unreachable!(),
    };

    match scalar_to_clif_type(tcx, element).by(u32::try_from(count).unwrap()) {
        // Cranelift currently only implements icmp for 128bit vectors.
        Some(vector_ty) if vector_ty.bits() == 128 => Some(vector_ty),
        _ => None,
    }
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
        types::F32 => types::I32,
        types::F64 => types::I64,
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
    args: &[mir::Operand<'tcx>],
    destination: CPlace<'tcx>,
    target: Option<BasicBlock>,
    source_info: mir::SourceInfo,
) {
    let intrinsic = fx.tcx.item_name(instance.def_id());
    let substs = instance.substs;

    if intrinsic.as_str().starts_with("simd_") {
        self::simd::codegen_simd_intrinsic_call(
            fx,
            intrinsic,
            substs,
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
            substs,
            args,
            destination,
            target,
            source_info,
        );
    }
}

fn codegen_float_intrinsic_call<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    intrinsic: Symbol,
    args: &[mir::Operand<'tcx>],
    ret: CPlace<'tcx>,
) -> bool {
    let (name, arg_count, ty) = match intrinsic {
        sym::expf32 => ("expf", 1, fx.tcx.types.f32),
        sym::expf64 => ("exp", 1, fx.tcx.types.f64),
        sym::exp2f32 => ("exp2f", 1, fx.tcx.types.f32),
        sym::exp2f64 => ("exp2", 1, fx.tcx.types.f64),
        sym::sqrtf32 => ("sqrtf", 1, fx.tcx.types.f32),
        sym::sqrtf64 => ("sqrt", 1, fx.tcx.types.f64),
        sym::powif32 => ("__powisf2", 2, fx.tcx.types.f32), // compiler-builtins
        sym::powif64 => ("__powidf2", 2, fx.tcx.types.f64), // compiler-builtins
        sym::powf32 => ("powf", 2, fx.tcx.types.f32),
        sym::powf64 => ("pow", 2, fx.tcx.types.f64),
        sym::logf32 => ("logf", 1, fx.tcx.types.f32),
        sym::logf64 => ("log", 1, fx.tcx.types.f64),
        sym::log2f32 => ("log2f", 1, fx.tcx.types.f32),
        sym::log2f64 => ("log2", 1, fx.tcx.types.f64),
        sym::log10f32 => ("log10f", 1, fx.tcx.types.f32),
        sym::log10f64 => ("log10", 1, fx.tcx.types.f64),
        sym::fabsf32 => ("fabsf", 1, fx.tcx.types.f32),
        sym::fabsf64 => ("fabs", 1, fx.tcx.types.f64),
        sym::fmaf32 => ("fmaf", 3, fx.tcx.types.f32),
        sym::fmaf64 => ("fma", 3, fx.tcx.types.f64),
        sym::copysignf32 => ("copysignf", 2, fx.tcx.types.f32),
        sym::copysignf64 => ("copysign", 2, fx.tcx.types.f64),
        sym::floorf32 => ("floorf", 1, fx.tcx.types.f32),
        sym::floorf64 => ("floor", 1, fx.tcx.types.f64),
        sym::ceilf32 => ("ceilf", 1, fx.tcx.types.f32),
        sym::ceilf64 => ("ceil", 1, fx.tcx.types.f64),
        sym::truncf32 => ("truncf", 1, fx.tcx.types.f32),
        sym::truncf64 => ("trunc", 1, fx.tcx.types.f64),
        sym::roundf32 => ("roundf", 1, fx.tcx.types.f32),
        sym::roundf64 => ("round", 1, fx.tcx.types.f64),
        sym::sinf32 => ("sinf", 1, fx.tcx.types.f32),
        sym::sinf64 => ("sin", 1, fx.tcx.types.f64),
        sym::cosf32 => ("cosf", 1, fx.tcx.types.f32),
        sym::cosf64 => ("cos", 1, fx.tcx.types.f64),
        _ => return false,
    };

    if args.len() != arg_count {
        bug!("wrong number of args for intrinsic {:?}", intrinsic);
    }

    let (a, b, c);
    let args = match args {
        [x] => {
            a = [codegen_operand(fx, x)];
            &a as &[_]
        }
        [x, y] => {
            b = [codegen_operand(fx, x), codegen_operand(fx, y)];
            &b
        }
        [x, y, z] => {
            c = [codegen_operand(fx, x), codegen_operand(fx, y), codegen_operand(fx, z)];
            &c
        }
        _ => unreachable!(),
    };

    let layout = fx.layout_of(ty);
    let res = match intrinsic {
        sym::fmaf32 | sym::fmaf64 => {
            let a = args[0].load_scalar(fx);
            let b = args[1].load_scalar(fx);
            let c = args[2].load_scalar(fx);
            CValue::by_val(fx.bcx.ins().fma(a, b, c), layout)
        }
        sym::copysignf32 | sym::copysignf64 => {
            let a = args[0].load_scalar(fx);
            let b = args[1].load_scalar(fx);
            CValue::by_val(fx.bcx.ins().fcopysign(a, b), layout)
        }
        sym::fabsf32
        | sym::fabsf64
        | sym::floorf32
        | sym::floorf64
        | sym::ceilf32
        | sym::ceilf64
        | sym::truncf32
        | sym::truncf64 => {
            let a = args[0].load_scalar(fx);

            let val = match intrinsic {
                sym::fabsf32 | sym::fabsf64 => fx.bcx.ins().fabs(a),
                sym::floorf32 | sym::floorf64 => fx.bcx.ins().floor(a),
                sym::ceilf32 | sym::ceilf64 => fx.bcx.ins().ceil(a),
                sym::truncf32 | sym::truncf64 => fx.bcx.ins().trunc(a),
                _ => unreachable!(),
            };

            CValue::by_val(val, layout)
        }
        // These intrinsics aren't supported natively by Cranelift.
        // Lower them to a libcall.
        _ => fx.easy_call(name, &args, ty),
    };

    ret.write_cvalue(fx, res);

    true
}

fn codegen_regular_intrinsic_call<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    instance: Instance<'tcx>,
    intrinsic: Symbol,
    substs: SubstsRef<'tcx>,
    args: &[mir::Operand<'tcx>],
    ret: CPlace<'tcx>,
    destination: Option<BasicBlock>,
    source_info: mir::SourceInfo,
) {
    let usize_layout = fx.layout_of(fx.tcx.types.usize);

    match intrinsic {
        sym::abort => {
            fx.bcx.ins().trap(TrapCode::User(0));
            return;
        }
        sym::likely | sym::unlikely => {
            intrinsic_args!(fx, args => (a); intrinsic);

            ret.write_cvalue(fx, a);
        }
        sym::breakpoint => {
            intrinsic_args!(fx, args => (); intrinsic);

            fx.bcx.ins().debugtrap();
        }
        sym::copy | sym::copy_nonoverlapping => {
            intrinsic_args!(fx, args => (src, dst, count); intrinsic);
            let src = src.load_scalar(fx);
            let dst = dst.load_scalar(fx);
            let count = count.load_scalar(fx);

            let elem_ty = substs.type_at(0);
            let elem_size: u64 = fx.layout_of(elem_ty).size.bytes();
            assert_eq!(args.len(), 3);
            let byte_amount =
                if elem_size != 1 { fx.bcx.ins().imul_imm(count, elem_size as i64) } else { count };

            if intrinsic == sym::copy_nonoverlapping {
                // FIXME emit_small_memcpy
                fx.bcx.call_memcpy(fx.target_config, dst, src, byte_amount);
            } else {
                // FIXME emit_small_memmove
                fx.bcx.call_memmove(fx.target_config, dst, src, byte_amount);
            }
        }
        sym::volatile_copy_memory | sym::volatile_copy_nonoverlapping_memory => {
            // NOTE: the volatile variants have src and dst swapped
            intrinsic_args!(fx, args => (dst, src, count); intrinsic);
            let dst = dst.load_scalar(fx);
            let src = src.load_scalar(fx);
            let count = count.load_scalar(fx);

            let elem_ty = substs.type_at(0);
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

            let layout = fx.layout_of(substs.type_at(0));
            // Note: Can't use is_unsized here as truly unsized types need to take the fixed size
            // branch
            let size = if let Abi::ScalarPair(_, _) = ptr.layout().abi {
                let (_ptr, info) = ptr.load_scalar_pair(fx);
                let (size, _align) = crate::unsize::size_and_align_of_dst(fx, layout, info);
                size
            } else {
                fx.bcx.ins().iconst(fx.pointer_type, layout.size.bytes() as i64)
            };
            ret.write_cvalue(fx, CValue::by_val(size, usize_layout));
        }
        sym::min_align_of_val => {
            intrinsic_args!(fx, args => (ptr); intrinsic);

            let layout = fx.layout_of(substs.type_at(0));
            // Note: Can't use is_unsized here as truly unsized types need to take the fixed size
            // branch
            let align = if let Abi::ScalarPair(_, _) = ptr.layout().abi {
                let (_ptr, info) = ptr.load_scalar_pair(fx);
                let (_size, align) = crate::unsize::size_and_align_of_dst(fx, layout, info);
                align
            } else {
                fx.bcx.ins().iconst(fx.pointer_type, layout.align.abi.bytes() as i64)
            };
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

        sym::unchecked_add
        | sym::unchecked_sub
        | sym::unchecked_mul
        | sym::unchecked_div
        | sym::exact_div
        | sym::unchecked_rem
        | sym::unchecked_shl
        | sym::unchecked_shr => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            // FIXME trap on overflow
            let bin_op = match intrinsic {
                sym::unchecked_add => BinOp::Add,
                sym::unchecked_sub => BinOp::Sub,
                sym::unchecked_mul => BinOp::Mul,
                sym::unchecked_div | sym::exact_div => BinOp::Div,
                sym::unchecked_rem => BinOp::Rem,
                sym::unchecked_shl => BinOp::Shl,
                sym::unchecked_shr => BinOp::Shr,
                _ => unreachable!(),
            };
            let res = crate::num::codegen_int_binop(fx, bin_op, x, y);
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
        sym::offset | sym::arith_offset => {
            intrinsic_args!(fx, args => (base, offset); intrinsic);
            let offset = offset.load_scalar(fx);

            let pointee_ty = base.layout().ty.builtin_deref(true).unwrap().ty;
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
            let ptr = ptr.load_scalar(fx);
            let mask = mask.load_scalar(fx);
            fx.bcx.ins().band(ptr, mask);
        }

        sym::transmute => {
            intrinsic_args!(fx, args => (from); intrinsic);

            if ret.layout().abi.is_uninhabited() {
                crate::base::codegen_panic(fx, "Transmuting to uninhabited type.", source_info);
                return;
            }

            ret.write_cvalue_transmute(fx, from);
        }
        sym::write_bytes | sym::volatile_set_memory => {
            intrinsic_args!(fx, args => (dst, val, count); intrinsic);
            let val = val.load_scalar(fx);
            let count = count.load_scalar(fx);

            let pointee_ty = dst.layout().ty.builtin_deref(true).unwrap().ty;
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
            let res = CValue::by_val(res, arg.layout());
            ret.write_cvalue(fx, res);
        }
        sym::cttz | sym::cttz_nonzero => {
            intrinsic_args!(fx, args => (arg); intrinsic);
            let val = arg.load_scalar(fx);

            // FIXME trap on `cttz_nonzero` with zero arg.
            let res = fx.bcx.ins().ctz(val);
            let res = CValue::by_val(res, arg.layout());
            ret.write_cvalue(fx, res);
        }
        sym::ctpop => {
            intrinsic_args!(fx, args => (arg); intrinsic);
            let val = arg.load_scalar(fx);

            let res = fx.bcx.ins().popcnt(val);
            let res = CValue::by_val(res, arg.layout());
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

            let ty = substs.type_at(0);
            let layout = fx.layout_of(ty);
            if layout.abi.is_uninhabited() {
                with_no_trimmed_paths!({
                    crate::base::codegen_panic_nounwind(
                        fx,
                        &format!("attempted to instantiate uninhabited type `{}`", layout.ty),
                        source_info,
                    )
                });
                return;
            }

            if intrinsic == sym::assert_zero_valid
                && !fx
                    .tcx
                    .check_validity_of_init((InitKind::Zero, fx.param_env().and(ty)))
                    .expect("expected to have layout during codegen")
            {
                with_no_trimmed_paths!({
                    crate::base::codegen_panic_nounwind(
                        fx,
                        &format!(
                            "attempted to zero-initialize type `{}`, which is invalid",
                            layout.ty
                        ),
                        source_info,
                    );
                });
                return;
            }

            if intrinsic == sym::assert_mem_uninitialized_valid
                && !fx
                    .tcx
                    .check_validity_of_init((
                        InitKind::UninitMitigated0x01Fill,
                        fx.param_env().and(ty),
                    ))
                    .expect("expected to have layout during codegen")
            {
                with_no_trimmed_paths!({
                    crate::base::codegen_panic_nounwind(
                        fx,
                        &format!(
                            "attempted to leave type `{}` uninitialized, which is invalid",
                            layout.ty
                        ),
                        source_info,
                    )
                });
                return;
            }
        }

        sym::volatile_load | sym::unaligned_volatile_load => {
            intrinsic_args!(fx, args => (ptr); intrinsic);

            // Cranelift treats loads as volatile by default
            // FIXME correctly handle unaligned_volatile_load
            let inner_layout = fx.layout_of(ptr.layout().ty.builtin_deref(true).unwrap().ty);
            let val = CValue::by_ref(Pointer::new(ptr.load_scalar(fx)), inner_layout);
            ret.write_cvalue(fx, val);
        }
        sym::volatile_store | sym::unaligned_volatile_store => {
            intrinsic_args!(fx, args => (ptr, val); intrinsic);
            let ptr = ptr.load_scalar(fx);

            // Cranelift treats stores as volatile by default
            // FIXME correctly handle unaligned_volatile_store
            let dest = CPlace::for_ptr(Pointer::new(ptr), val.layout());
            dest.write_cvalue(fx, val);
        }

        sym::pref_align_of
        | sym::needs_drop
        | sym::type_id
        | sym::type_name
        | sym::variant_count => {
            intrinsic_args!(fx, args => (); intrinsic);

            let const_val =
                fx.tcx.const_eval_instance(ParamEnv::reveal_all(), instance, None).unwrap();
            let val = crate::constant::codegen_const_value(fx, const_val, ret.layout().ty);
            ret.write_cvalue(fx, val);
        }

        sym::ptr_offset_from | sym::ptr_offset_from_unsigned => {
            intrinsic_args!(fx, args => (ptr, base); intrinsic);
            let ptr = ptr.load_scalar(fx);
            let base = base.load_scalar(fx);
            let ty = substs.type_at(0);

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

        sym::ptr_guaranteed_cmp => {
            intrinsic_args!(fx, args => (a, b); intrinsic);

            let val = crate::num::codegen_ptr_binop(fx, BinOp::Eq, a, b).load_scalar(fx);
            ret.write_cvalue(fx, CValue::by_val(val, fx.layout_of(fx.tcx.types.u8)));
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

            let ty = substs.type_at(0);
            match ty.kind() {
                ty::Uint(UintTy::U128) | ty::Int(IntTy::I128) => {
                    // FIXME implement 128bit atomics
                    if fx.tcx.is_compiler_builtins(LOCAL_CRATE) {
                        // special case for compiler-builtins to avoid having to patch it
                        crate::trap::trap_unimplemented(fx, "128bit atomics not yet supported");
                        return;
                    } else {
                        fx.tcx
                            .sess
                            .span_fatal(source_info.span, "128bit atomics not yet supported");
                    }
                }
                ty::Uint(_) | ty::Int(_) | ty::RawPtr(..) => {}
                _ => {
                    report_atomic_type_validation_error(fx, intrinsic, source_info.span, ty);
                    return;
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

            let ty = substs.type_at(0);
            match ty.kind() {
                ty::Uint(UintTy::U128) | ty::Int(IntTy::I128) => {
                    // FIXME implement 128bit atomics
                    if fx.tcx.is_compiler_builtins(LOCAL_CRATE) {
                        // special case for compiler-builtins to avoid having to patch it
                        crate::trap::trap_unimplemented(fx, "128bit atomics not yet supported");
                        return;
                    } else {
                        fx.tcx
                            .sess
                            .span_fatal(source_info.span, "128bit atomics not yet supported");
                    }
                }
                ty::Uint(_) | ty::Int(_) | ty::RawPtr(..) => {}
                _ => {
                    report_atomic_type_validation_error(fx, intrinsic, source_info.span, ty);
                    return;
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
                    return;
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
                    return;
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
                    return;
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
                    return;
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
                    return;
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
                    return;
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
                    return;
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
                    return;
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
                ty::Uint(_) | ty::Int(_) | ty::RawPtr(..) => {}
                _ => {
                    report_atomic_type_validation_error(fx, intrinsic, source_info.span, layout.ty);
                    return;
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
                ty::Uint(_) | ty::Int(_) | ty::RawPtr(..) => {}
                _ => {
                    report_atomic_type_validation_error(fx, intrinsic, source_info.span, layout.ty);
                    return;
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
                ty::Uint(_) | ty::Int(_) | ty::RawPtr(..) => {}
                _ => {
                    report_atomic_type_validation_error(fx, intrinsic, source_info.span, layout.ty);
                    return;
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
                ty::Uint(_) | ty::Int(_) | ty::RawPtr(..) => {}
                _ => {
                    report_atomic_type_validation_error(fx, intrinsic, source_info.span, layout.ty);
                    return;
                }
            }
            let ty = fx.clif_type(layout.ty).unwrap();

            let src = src.load_scalar(fx);

            let old = fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Umin, ptr, src);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
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

        kw::Try => {
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

            let layout = ret.layout();
            let ret_val = CValue::const_val(fx, layout, ty::ScalarInt::null(layout.size));
            ret.write_cvalue(fx, ret_val);
        }

        sym::fadd_fast | sym::fsub_fast | sym::fmul_fast | sym::fdiv_fast | sym::frem_fast => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            let res = crate::num::codegen_float_binop(
                fx,
                match intrinsic {
                    sym::fadd_fast => BinOp::Add,
                    sym::fsub_fast => BinOp::Sub,
                    sym::fmul_fast => BinOp::Mul,
                    sym::fdiv_fast => BinOp::Div,
                    sym::frem_fast => BinOp::Rem,
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

            let size = fx.layout_of(substs.type_at(0)).layout.size();
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

        sym::const_allocate => {
            intrinsic_args!(fx, args => (_size, _align); intrinsic);

            // returns a null pointer at runtime.
            let null = fx.bcx.ins().iconst(fx.pointer_type, 0);
            ret.write_cvalue(fx, CValue::by_val(null, ret.layout()));
        }

        sym::const_deallocate => {
            intrinsic_args!(fx, args => (_ptr, _size, _align); intrinsic);
            // nop at runtime.
        }

        sym::black_box => {
            intrinsic_args!(fx, args => (a); intrinsic);

            // FIXME implement black_box semantics
            ret.write_cvalue(fx, a);
        }

        // FIXME implement variadics in cranelift
        sym::va_copy | sym::va_arg | sym::va_end => {
            fx.tcx.sess.span_fatal(
                source_info.span,
                "Defining variadic functions is not yet supported by Cranelift",
            );
        }

        _ => {
            fx.tcx
                .sess
                .span_fatal(source_info.span, &format!("unsupported intrinsic {}", intrinsic));
        }
    }

    let ret_block = fx.get_block(destination.unwrap());
    fx.bcx.ins().jump(ret_block, &[]);
}
