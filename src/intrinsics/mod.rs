//! Codegen of intrinsics. This includes `extern "rust-intrinsic"`, `extern "platform-intrinsic"`
//! and LLVM intrinsics that have symbol names starting with `llvm.`.

mod cpuid;
mod llvm;
mod simd;

pub(crate) use cpuid::codegen_cpuid_call;
pub(crate) use llvm::codegen_llvm_intrinsic_call;

use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_span::symbol::{kw, sym};

use crate::prelude::*;
use cranelift_codegen::ir::AtomicRmwOp;

macro intrinsic_pat {
    (_) => {
        _
    },
    ($name:ident) => {
        sym::$name
    },
    (kw.$name:ident) => {
        kw::$name
    },
    ($name:literal) => {
        $name
    },
}

macro intrinsic_arg {
    (o $fx:expr, $arg:ident) => {
        $arg
    },
    (c $fx:expr, $arg:ident) => {
        codegen_operand($fx, $arg)
    },
    (v $fx:expr, $arg:ident) => {
        codegen_operand($fx, $arg).load_scalar($fx)
    }
}

macro intrinsic_substs {
    ($substs:expr, $index:expr,) => {},
    ($substs:expr, $index:expr, $first:ident $(,$rest:ident)*) => {
        let $first = $substs.type_at($index);
        intrinsic_substs!($substs, $index+1, $($rest),*);
    }
}

macro intrinsic_match {
    ($fx:expr, $intrinsic:expr, $substs:expr, $args:expr,
    _ => $unknown:block;
    $(
        $($($name:tt).*)|+ $(if $cond:expr)?, $(<$($subst:ident),*>)? ($($a:ident $arg:ident),*) $content:block;
    )*) => {
        let _ = $substs; // Silence warning when substs is unused.
        match $intrinsic {
            $(
                $(intrinsic_pat!($($name).*))|* $(if $cond)? => {
                    #[allow(unused_parens, non_snake_case)]
                    {
                        $(
                            intrinsic_substs!($substs, 0, $($subst),*);
                        )?
                        if let [$($arg),*] = $args {
                            let ($($arg,)*) = (
                                $(intrinsic_arg!($a $fx, $arg),)*
                            );
                            #[warn(unused_parens, non_snake_case)]
                            {
                                $content
                            }
                        } else {
                            bug!("wrong number of args for intrinsic {:?}", $intrinsic);
                        }
                    }
                }
            )*
            _ => $unknown,
        }
    }
}

macro call_intrinsic_match {
    ($fx:expr, $intrinsic:expr, $substs:expr, $ret:expr, $destination:expr, $args:expr, $(
        $name:ident($($arg:ident),*) -> $ty:ident => $func:ident,
    )*) => {
        match $intrinsic {
            $(
                sym::$name => {
                    assert!($substs.is_noop());
                    if let [$(ref $arg),*] = *$args {
                        let ($($arg,)*) = (
                            $(codegen_operand($fx, $arg),)*
                        );
                        let res = $fx.easy_call(stringify!($func), &[$($arg),*], $fx.tcx.types.$ty);
                        $ret.write_cvalue($fx, res);

                        if let Some((_, dest)) = $destination {
                            let ret_block = $fx.get_block(dest);
                            $fx.bcx.ins().jump(ret_block, &[]);
                            return;
                        } else {
                            unreachable!();
                        }
                    } else {
                        bug!("wrong number of args for intrinsic {:?}", $intrinsic);
                    }
                }
            )*
            _ => {}
        }
    }
}

macro validate_atomic_type($fx:ident, $intrinsic:ident, $span:ident, $ty:expr) {
    match $ty.kind() {
        ty::Uint(_) | ty::Int(_) | ty::RawPtr(..) => {}
        _ => {
            $fx.tcx.sess.span_err(
                $span,
                &format!(
                    "`{}` intrinsic: expected basic integer or raw pointer type, found `{:?}`",
                    $intrinsic, $ty
                ),
            );
            // Prevent verifier error
            crate::trap::trap_unreachable($fx, "compilation should not have succeeded");
            return;
        }
    }
}

macro validate_simd_type($fx:ident, $intrinsic:ident, $span:ident, $ty:expr) {
    if !$ty.is_simd() {
        $fx.tcx.sess.span_err($span, &format!("invalid monomorphization of `{}` intrinsic: expected SIMD input type, found non-SIMD `{}`", $intrinsic, $ty));
        // Prevent verifier error
        crate::trap::trap_unreachable($fx, "compilation should not have succeeded");
        return;
    }
}

pub(crate) fn clif_vector_type<'tcx>(tcx: TyCtxt<'tcx>, layout: TyAndLayout<'tcx>) -> Option<Type> {
    let (element, count) = match layout.abi {
        Abi::Vector { element, count } => (element, count),
        _ => unreachable!(),
    };

    match scalar_to_clif_type(tcx, element).by(u16::try_from(count).unwrap()) {
        // Cranelift currently only implements icmp for 128bit vectors.
        Some(vector_ty) if vector_ty.bits() == 128 => Some(vector_ty),
        _ => None,
    }
}

fn simd_for_each_lane<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    val: CValue<'tcx>,
    ret: CPlace<'tcx>,
    f: impl Fn(
        &mut FunctionCx<'_, '_, 'tcx>,
        TyAndLayout<'tcx>,
        TyAndLayout<'tcx>,
        Value,
    ) -> CValue<'tcx>,
) {
    let layout = val.layout();

    let (lane_count, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
    let lane_layout = fx.layout_of(lane_ty);
    let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);
    let ret_lane_layout = fx.layout_of(ret_lane_ty);
    assert_eq!(lane_count, ret_lane_count);

    for lane_idx in 0..lane_count {
        let lane = val.value_lane(fx, lane_idx).load_scalar(fx);

        let res_lane = f(fx, lane_layout, ret_lane_layout, lane);

        ret.place_lane(fx, lane_idx).write_cvalue(fx, res_lane);
    }
}

fn simd_pair_for_each_lane<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    x: CValue<'tcx>,
    y: CValue<'tcx>,
    ret: CPlace<'tcx>,
    f: impl Fn(
        &mut FunctionCx<'_, '_, 'tcx>,
        TyAndLayout<'tcx>,
        TyAndLayout<'tcx>,
        Value,
        Value,
    ) -> CValue<'tcx>,
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

        let res_lane = f(fx, lane_layout, ret_lane_layout, x_lane, y_lane);

        ret.place_lane(fx, lane_idx).write_cvalue(fx, res_lane);
    }
}

fn simd_reduce<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    val: CValue<'tcx>,
    acc: Option<Value>,
    ret: CPlace<'tcx>,
    f: impl Fn(&mut FunctionCx<'_, '_, 'tcx>, TyAndLayout<'tcx>, Value, Value) -> Value,
) {
    let (lane_count, lane_ty) = val.layout().ty.simd_size_and_type(fx.tcx);
    let lane_layout = fx.layout_of(lane_ty);
    assert_eq!(lane_layout, ret.layout());

    let (mut res_val, start_lane) =
        if let Some(acc) = acc { (acc, 0) } else { (val.value_lane(fx, 0).load_scalar(fx), 1) };
    for lane_idx in start_lane..lane_count {
        let lane = val.value_lane(fx, lane_idx).load_scalar(fx);
        res_val = f(fx, lane_layout, res_val, lane);
    }
    let res = CValue::by_val(res_val, lane_layout);
    ret.write_cvalue(fx, res);
}

// FIXME move all uses to `simd_reduce`
fn simd_reduce_bool<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    val: CValue<'tcx>,
    ret: CPlace<'tcx>,
    f: impl Fn(&mut FunctionCx<'_, '_, 'tcx>, Value, Value) -> Value,
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
    layout: TyAndLayout<'tcx>,
    val: Value,
) -> CValue<'tcx> {
    let ty = fx.clif_type(layout.ty).unwrap();

    let int_ty = match ty {
        types::F32 => types::I32,
        types::F64 => types::I64,
        ty => ty,
    };

    let val = fx.bcx.ins().bint(int_ty, val);
    let mut res = fx.bcx.ins().ineg(val);

    if ty.is_float() {
        res = fx.bcx.ins().bitcast(ty, res);
    }

    CValue::by_val(res, layout)
}

macro simd_cmp {
    ($fx:expr, $cc:ident|$cc_f:ident($x:ident, $y:ident) -> $ret:ident) => {
        let vector_ty = clif_vector_type($fx.tcx, $x.layout());

        if let Some(vector_ty) = vector_ty {
            let x = $x.load_scalar($fx);
            let y = $y.load_scalar($fx);
            let val = if vector_ty.lane_type().is_float() {
                $fx.bcx.ins().fcmp(FloatCC::$cc_f, x, y)
            } else {
                $fx.bcx.ins().icmp(IntCC::$cc, x, y)
            };

            // HACK This depends on the fact that icmp for vectors represents bools as 0 and !0, not 0 and 1.
            let val = $fx.bcx.ins().raw_bitcast(vector_ty, val);

            $ret.write_cvalue($fx, CValue::by_val(val, $ret.layout()));
        } else {
            simd_pair_for_each_lane(
                $fx,
                $x,
                $y,
                $ret,
                |fx, lane_layout, res_lane_layout, x_lane, y_lane| {
                    let res_lane = match lane_layout.ty.kind() {
                        ty::Uint(_) | ty::Int(_) => fx.bcx.ins().icmp(IntCC::$cc, x_lane, y_lane),
                        ty::Float(_) => fx.bcx.ins().fcmp(FloatCC::$cc_f, x_lane, y_lane),
                        _ => unreachable!("{:?}", lane_layout.ty),
                    };
                    bool_to_zero_or_max_uint(fx, res_lane_layout, res_lane)
                },
            );
        }
    },
    ($fx:expr, $cc_u:ident|$cc_s:ident|$cc_f:ident($x:ident, $y:ident) -> $ret:ident) => {
        // FIXME use vector icmp when possible
        simd_pair_for_each_lane(
            $fx,
            $x,
            $y,
            $ret,
            |fx, lane_layout, res_lane_layout, x_lane, y_lane| {
                let res_lane = match lane_layout.ty.kind() {
                    ty::Uint(_) => fx.bcx.ins().icmp(IntCC::$cc_u, x_lane, y_lane),
                    ty::Int(_) => fx.bcx.ins().icmp(IntCC::$cc_s, x_lane, y_lane),
                    ty::Float(_) => fx.bcx.ins().fcmp(FloatCC::$cc_f, x_lane, y_lane),
                    _ => unreachable!("{:?}", lane_layout.ty),
                };
                bool_to_zero_or_max_uint(fx, res_lane_layout, res_lane)
            },
        );
    },
}

macro simd_int_binop {
    ($fx:expr, $op:ident($x:ident, $y:ident) -> $ret:ident) => {
        simd_int_binop!($fx, $op|$op($x, $y) -> $ret);
    },
    ($fx:expr, $op_u:ident|$op_s:ident($x:ident, $y:ident) -> $ret:ident) => {
        simd_pair_for_each_lane(
            $fx,
            $x,
            $y,
            $ret,
            |fx, lane_layout, ret_lane_layout, x_lane, y_lane| {
                let res_lane = match lane_layout.ty.kind() {
                    ty::Uint(_) => fx.bcx.ins().$op_u(x_lane, y_lane),
                    ty::Int(_) => fx.bcx.ins().$op_s(x_lane, y_lane),
                    _ => unreachable!("{:?}", lane_layout.ty),
                };
                CValue::by_val(res_lane, ret_lane_layout)
            },
        );
    },
}

macro simd_int_flt_binop {
    ($fx:expr, $op:ident|$op_f:ident($x:ident, $y:ident) -> $ret:ident) => {
        simd_int_flt_binop!($fx, $op|$op|$op_f($x, $y) -> $ret);
    },
    ($fx:expr, $op_u:ident|$op_s:ident|$op_f:ident($x:ident, $y:ident) -> $ret:ident) => {
        simd_pair_for_each_lane(
            $fx,
            $x,
            $y,
            $ret,
            |fx, lane_layout, ret_lane_layout, x_lane, y_lane| {
                let res_lane = match lane_layout.ty.kind() {
                    ty::Uint(_) => fx.bcx.ins().$op_u(x_lane, y_lane),
                    ty::Int(_) => fx.bcx.ins().$op_s(x_lane, y_lane),
                    ty::Float(_) => fx.bcx.ins().$op_f(x_lane, y_lane),
                    _ => unreachable!("{:?}", lane_layout.ty),
                };
                CValue::by_val(res_lane, ret_lane_layout)
            },
        );
    },
}

macro simd_flt_binop($fx:expr, $op:ident($x:ident, $y:ident) -> $ret:ident) {
    simd_pair_for_each_lane(
        $fx,
        $x,
        $y,
        $ret,
        |fx, lane_layout, ret_lane_layout, x_lane, y_lane| {
            let res_lane = match lane_layout.ty.kind() {
                ty::Float(_) => fx.bcx.ins().$op(x_lane, y_lane),
                _ => unreachable!("{:?}", lane_layout.ty),
            };
            CValue::by_val(res_lane, ret_lane_layout)
        },
    );
}

pub(crate) fn codegen_intrinsic_call<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    instance: Instance<'tcx>,
    args: &[mir::Operand<'tcx>],
    destination: Option<(CPlace<'tcx>, BasicBlock)>,
    span: Span,
) {
    let def_id = instance.def_id();
    let substs = instance.substs;

    let intrinsic = fx.tcx.item_name(def_id);

    let ret = match destination {
        Some((place, _)) => place,
        None => {
            // Insert non returning intrinsics here
            match intrinsic {
                sym::abort => {
                    trap_abort(fx, "Called intrinsic::abort.");
                }
                sym::transmute => {
                    crate::base::codegen_panic(fx, "Transmuting to uninhabited type.", span);
                }
                _ => unimplemented!("unsupported instrinsic {}", intrinsic),
            }
            return;
        }
    };

    if intrinsic.as_str().starts_with("simd_") {
        self::simd::codegen_simd_intrinsic_call(fx, instance, args, ret, span);
        let ret_block = fx.get_block(destination.expect("SIMD intrinsics don't diverge").1);
        fx.bcx.ins().jump(ret_block, &[]);
        return;
    }

    let usize_layout = fx.layout_of(fx.tcx.types.usize);

    call_intrinsic_match! {
        fx, intrinsic, substs, ret, destination, args,
        expf32(flt) -> f32 => expf,
        expf64(flt) -> f64 => exp,
        exp2f32(flt) -> f32 => exp2f,
        exp2f64(flt) -> f64 => exp2,
        sqrtf32(flt) -> f32 => sqrtf,
        sqrtf64(flt) -> f64 => sqrt,
        powif32(a, x) -> f32 => __powisf2, // compiler-builtins
        powif64(a, x) -> f64 => __powidf2, // compiler-builtins
        powf32(a, x) -> f32 => powf,
        powf64(a, x) -> f64 => pow,
        logf32(flt) -> f32 => logf,
        logf64(flt) -> f64 => log,
        log2f32(flt) -> f32 => log2f,
        log2f64(flt) -> f64 => log2,
        log10f32(flt) -> f32 => log10f,
        log10f64(flt) -> f64 => log10,
        fabsf32(flt) -> f32 => fabsf,
        fabsf64(flt) -> f64 => fabs,
        fmaf32(x, y, z) -> f32 => fmaf,
        fmaf64(x, y, z) -> f64 => fma,
        copysignf32(x, y) -> f32 => copysignf,
        copysignf64(x, y) -> f64 => copysign,

        // rounding variants
        // FIXME use clif insts
        floorf32(flt) -> f32 => floorf,
        floorf64(flt) -> f64 => floor,
        ceilf32(flt) -> f32 => ceilf,
        ceilf64(flt) -> f64 => ceil,
        truncf32(flt) -> f32 => truncf,
        truncf64(flt) -> f64 => trunc,
        roundf32(flt) -> f32 => roundf,
        roundf64(flt) -> f64 => round,

        // trigonometry
        sinf32(flt) -> f32 => sinf,
        sinf64(flt) -> f64 => sin,
        cosf32(flt) -> f32 => cosf,
        cosf64(flt) -> f64 => cos,
    }

    intrinsic_match! {
        fx, intrinsic, substs, args,
        _ => {
            fx.tcx.sess.span_fatal(span, &format!("unsupported intrinsic {}", intrinsic));
        };

        assume, (c _a) {};
        likely | unlikely, (c a) {
            ret.write_cvalue(fx, a);
        };
        breakpoint, () {
            fx.bcx.ins().debugtrap();
        };
        copy | copy_nonoverlapping, <elem_ty> (v src, v dst, v count) {
            let elem_size: u64 = fx.layout_of(elem_ty).size.bytes();
            assert_eq!(args.len(), 3);
            let byte_amount = if elem_size != 1 {
                fx.bcx.ins().imul_imm(count, elem_size as i64)
            } else {
                count
            };

            if intrinsic == sym::copy_nonoverlapping {
                // FIXME emit_small_memcpy
                fx.bcx.call_memcpy(fx.module.target_config(), dst, src, byte_amount);
            } else {
                // FIXME emit_small_memmove
                fx.bcx.call_memmove(fx.module.target_config(), dst, src, byte_amount);
            }
        };
        // NOTE: the volatile variants have src and dst swapped
        volatile_copy_memory | volatile_copy_nonoverlapping_memory, <elem_ty> (v dst, v src, v count) {
            let elem_size: u64 = fx.layout_of(elem_ty).size.bytes();
            assert_eq!(args.len(), 3);
            let byte_amount = if elem_size != 1 {
                fx.bcx.ins().imul_imm(count, elem_size as i64)
            } else {
                count
            };

            // FIXME make the copy actually volatile when using emit_small_mem{cpy,move}
            if intrinsic == sym::volatile_copy_nonoverlapping_memory {
                // FIXME emit_small_memcpy
                fx.bcx.call_memcpy(fx.module.target_config(), dst, src, byte_amount);
            } else {
                // FIXME emit_small_memmove
                fx.bcx.call_memmove(fx.module.target_config(), dst, src, byte_amount);
            }
        };
        size_of_val, <T> (c ptr) {
            let layout = fx.layout_of(T);
            let size = if layout.is_unsized() {
                let (_ptr, info) = ptr.load_scalar_pair(fx);
                let (size, _align) = crate::unsize::size_and_align_of_dst(fx, layout, info);
                size
            } else {
                fx
                    .bcx
                    .ins()
                    .iconst(fx.pointer_type, layout.size.bytes() as i64)
            };
            ret.write_cvalue(fx, CValue::by_val(size, usize_layout));
        };
        min_align_of_val, <T> (c ptr) {
            let layout = fx.layout_of(T);
            let align = if layout.is_unsized() {
                let (_ptr, info) = ptr.load_scalar_pair(fx);
                let (_size, align) = crate::unsize::size_and_align_of_dst(fx, layout, info);
                align
            } else {
                fx
                    .bcx
                    .ins()
                    .iconst(fx.pointer_type, layout.align.abi.bytes() as i64)
            };
            ret.write_cvalue(fx, CValue::by_val(align, usize_layout));
        };

        unchecked_add | unchecked_sub | unchecked_div | exact_div | unchecked_rem
        | unchecked_shl | unchecked_shr, (c x, c y) {
            // FIXME trap on overflow
            let bin_op = match intrinsic {
                sym::unchecked_add => BinOp::Add,
                sym::unchecked_sub => BinOp::Sub,
                sym::unchecked_div | sym::exact_div => BinOp::Div,
                sym::unchecked_rem => BinOp::Rem,
                sym::unchecked_shl => BinOp::Shl,
                sym::unchecked_shr => BinOp::Shr,
                _ => unreachable!(),
            };
            let res = crate::num::codegen_int_binop(fx, bin_op, x, y);
            ret.write_cvalue(fx, res);
        };
        add_with_overflow | sub_with_overflow | mul_with_overflow, (c x, c y) {
            assert_eq!(x.layout().ty, y.layout().ty);
            let bin_op = match intrinsic {
                sym::add_with_overflow => BinOp::Add,
                sym::sub_with_overflow => BinOp::Sub,
                sym::mul_with_overflow => BinOp::Mul,
                _ => unreachable!(),
            };

            let res = crate::num::codegen_checked_int_binop(
                fx,
                bin_op,
                x,
                y,
            );
            ret.write_cvalue(fx, res);
        };
        saturating_add | saturating_sub, <T> (c lhs, c rhs) {
            assert_eq!(lhs.layout().ty, rhs.layout().ty);
            let bin_op = match intrinsic {
                sym::saturating_add => BinOp::Add,
                sym::saturating_sub => BinOp::Sub,
                _ => unreachable!(),
            };

            let signed = type_sign(T);

            let checked_res = crate::num::codegen_checked_int_binop(
                fx,
                bin_op,
                lhs,
                rhs,
            );

            let (val, has_overflow) = checked_res.load_scalar_pair(fx);
            let clif_ty = fx.clif_type(T).unwrap();

            let (min, max) = type_min_max_value(&mut fx.bcx, clif_ty, signed);

            let val = match (intrinsic, signed) {
                (sym::saturating_add, false) => fx.bcx.ins().select(has_overflow, max, val),
                (sym::saturating_sub, false) => fx.bcx.ins().select(has_overflow, min, val),
                (sym::saturating_add, true) => {
                    let rhs = rhs.load_scalar(fx);
                    let rhs_ge_zero = fx.bcx.ins().icmp_imm(IntCC::SignedGreaterThanOrEqual, rhs, 0);
                    let sat_val = fx.bcx.ins().select(rhs_ge_zero, max, min);
                    fx.bcx.ins().select(has_overflow, sat_val, val)
                }
                (sym::saturating_sub, true) => {
                    let rhs = rhs.load_scalar(fx);
                    let rhs_ge_zero = fx.bcx.ins().icmp_imm(IntCC::SignedGreaterThanOrEqual, rhs, 0);
                    let sat_val = fx.bcx.ins().select(rhs_ge_zero, min, max);
                    fx.bcx.ins().select(has_overflow, sat_val, val)
                }
                _ => unreachable!(),
            };

            let res = CValue::by_val(val, fx.layout_of(T));

            ret.write_cvalue(fx, res);
        };
        rotate_left, <T>(v x, v y) {
            let layout = fx.layout_of(T);
            let res = fx.bcx.ins().rotl(x, y);
            ret.write_cvalue(fx, CValue::by_val(res, layout));
        };
        rotate_right, <T>(v x, v y) {
            let layout = fx.layout_of(T);
            let res = fx.bcx.ins().rotr(x, y);
            ret.write_cvalue(fx, CValue::by_val(res, layout));
        };

        // The only difference between offset and arith_offset is regarding UB. Because Cranelift
        // doesn't have UB both are codegen'ed the same way
        offset | arith_offset, (c base, v offset) {
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
        };

        transmute, (c from) {
            ret.write_cvalue_transmute(fx, from);
        };
        write_bytes | volatile_set_memory, (c dst, v val, v count) {
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
            fx.bcx.call_memset(fx.module.target_config(), dst_ptr, val, count);
        };
        ctlz | ctlz_nonzero, <T> (v arg) {
            // FIXME trap on `ctlz_nonzero` with zero arg.
            let res = fx.bcx.ins().clz(arg);
            let res = CValue::by_val(res, fx.layout_of(T));
            ret.write_cvalue(fx, res);
        };
        cttz | cttz_nonzero, <T> (v arg) {
            // FIXME trap on `cttz_nonzero` with zero arg.
            let res = fx.bcx.ins().ctz(arg);
            let res = CValue::by_val(res, fx.layout_of(T));
            ret.write_cvalue(fx, res);
        };
        ctpop, <T> (v arg) {
            let res = fx.bcx.ins().popcnt(arg);
            let res = CValue::by_val(res, fx.layout_of(T));
            ret.write_cvalue(fx, res);
        };
        bitreverse, <T> (v arg) {
            let res = fx.bcx.ins().bitrev(arg);
            let res = CValue::by_val(res, fx.layout_of(T));
            ret.write_cvalue(fx, res);
        };
        bswap, <T> (v arg) {
            // FIXME(CraneStation/cranelift#794) add bswap instruction to cranelift
            fn swap(bcx: &mut FunctionBuilder<'_>, v: Value) -> Value {
                match bcx.func.dfg.value_type(v) {
                    types::I8 => v,

                    // https://code.woboq.org/gcc/include/bits/byteswap.h.html
                    types::I16 => {
                        let tmp1 = bcx.ins().ishl_imm(v, 8);
                        let n1 = bcx.ins().band_imm(tmp1, 0xFF00);

                        let tmp2 = bcx.ins().ushr_imm(v, 8);
                        let n2 = bcx.ins().band_imm(tmp2, 0x00FF);

                        bcx.ins().bor(n1, n2)
                    }
                    types::I32 => {
                        let tmp1 = bcx.ins().ishl_imm(v, 24);
                        let n1 = bcx.ins().band_imm(tmp1, 0xFF00_0000);

                        let tmp2 = bcx.ins().ishl_imm(v, 8);
                        let n2 = bcx.ins().band_imm(tmp2, 0x00FF_0000);

                        let tmp3 = bcx.ins().ushr_imm(v, 8);
                        let n3 = bcx.ins().band_imm(tmp3, 0x0000_FF00);

                        let tmp4 = bcx.ins().ushr_imm(v, 24);
                        let n4 = bcx.ins().band_imm(tmp4, 0x0000_00FF);

                        let or_tmp1 = bcx.ins().bor(n1, n2);
                        let or_tmp2 = bcx.ins().bor(n3, n4);
                        bcx.ins().bor(or_tmp1, or_tmp2)
                    }
                    types::I64 => {
                        let tmp1 = bcx.ins().ishl_imm(v, 56);
                        let n1 = bcx.ins().band_imm(tmp1, 0xFF00_0000_0000_0000u64 as i64);

                        let tmp2 = bcx.ins().ishl_imm(v, 40);
                        let n2 = bcx.ins().band_imm(tmp2, 0x00FF_0000_0000_0000u64 as i64);

                        let tmp3 = bcx.ins().ishl_imm(v, 24);
                        let n3 = bcx.ins().band_imm(tmp3, 0x0000_FF00_0000_0000u64 as i64);

                        let tmp4 = bcx.ins().ishl_imm(v, 8);
                        let n4 = bcx.ins().band_imm(tmp4, 0x0000_00FF_0000_0000u64 as i64);

                        let tmp5 = bcx.ins().ushr_imm(v, 8);
                        let n5 = bcx.ins().band_imm(tmp5, 0x0000_0000_FF00_0000u64 as i64);

                        let tmp6 = bcx.ins().ushr_imm(v, 24);
                        let n6 = bcx.ins().band_imm(tmp6, 0x0000_0000_00FF_0000u64 as i64);

                        let tmp7 = bcx.ins().ushr_imm(v, 40);
                        let n7 = bcx.ins().band_imm(tmp7, 0x0000_0000_0000_FF00u64 as i64);

                        let tmp8 = bcx.ins().ushr_imm(v, 56);
                        let n8 = bcx.ins().band_imm(tmp8, 0x0000_0000_0000_00FFu64 as i64);

                        let or_tmp1 = bcx.ins().bor(n1, n2);
                        let or_tmp2 = bcx.ins().bor(n3, n4);
                        let or_tmp3 = bcx.ins().bor(n5, n6);
                        let or_tmp4 = bcx.ins().bor(n7, n8);

                        let or_tmp5 = bcx.ins().bor(or_tmp1, or_tmp2);
                        let or_tmp6 = bcx.ins().bor(or_tmp3, or_tmp4);
                        bcx.ins().bor(or_tmp5, or_tmp6)
                    }
                    types::I128 => {
                        let (lo, hi) = bcx.ins().isplit(v);
                        let lo = swap(bcx, lo);
                        let hi = swap(bcx, hi);
                        bcx.ins().iconcat(hi, lo)
                    }
                    ty => unreachable!("bswap {}", ty),
                }
            }
            let res = CValue::by_val(swap(&mut fx.bcx, arg), fx.layout_of(T));
            ret.write_cvalue(fx, res);
        };
        assert_inhabited | assert_zero_valid | assert_uninit_valid, <T> () {
            let layout = fx.layout_of(T);
            if layout.abi.is_uninhabited() {
                with_no_trimmed_paths(|| crate::base::codegen_panic(
                    fx,
                    &format!("attempted to instantiate uninhabited type `{}`", T),
                    span,
                ));
                return;
            }

            if intrinsic == sym::assert_zero_valid && !layout.might_permit_raw_init(fx, /*zero:*/ true) {
                with_no_trimmed_paths(|| crate::base::codegen_panic(
                    fx,
                    &format!("attempted to zero-initialize type `{}`, which is invalid", T),
                    span,
                ));
                return;
            }

            if intrinsic == sym::assert_uninit_valid && !layout.might_permit_raw_init(fx, /*zero:*/ false) {
                with_no_trimmed_paths(|| crate::base::codegen_panic(
                    fx,
                    &format!("attempted to leave type `{}` uninitialized, which is invalid", T),
                    span,
                ));
                return;
            }
        };

        volatile_load | unaligned_volatile_load, (c ptr) {
            // Cranelift treats loads as volatile by default
            // FIXME correctly handle unaligned_volatile_load
            let inner_layout =
                fx.layout_of(ptr.layout().ty.builtin_deref(true).unwrap().ty);
            let val = CValue::by_ref(Pointer::new(ptr.load_scalar(fx)), inner_layout);
            ret.write_cvalue(fx, val);
        };
        volatile_store | unaligned_volatile_store, (v ptr, c val) {
            // Cranelift treats stores as volatile by default
            // FIXME correctly handle unaligned_volatile_store
            let dest = CPlace::for_ptr(Pointer::new(ptr), val.layout());
            dest.write_cvalue(fx, val);
        };

        pref_align_of | min_align_of | needs_drop | type_id | type_name | variant_count, () {
            let const_val =
                fx.tcx.const_eval_instance(ParamEnv::reveal_all(), instance, None).unwrap();
            let val = crate::constant::codegen_const_value(
                fx,
                const_val,
                ret.layout().ty,
            );
            ret.write_cvalue(fx, val);
        };

        ptr_offset_from, <T> (v ptr, v base) {
            let isize_layout = fx.layout_of(fx.tcx.types.isize);

            let pointee_size: u64 = fx.layout_of(T).size.bytes();
            let diff = fx.bcx.ins().isub(ptr, base);
            // FIXME this can be an exact division.
            let val = CValue::by_val(fx.bcx.ins().sdiv_imm(diff, pointee_size as i64), isize_layout);
            ret.write_cvalue(fx, val);
        };

        ptr_guaranteed_eq, (c a, c b) {
            let val = crate::num::codegen_ptr_binop(fx, BinOp::Eq, a, b);
            ret.write_cvalue(fx, val);
        };

        ptr_guaranteed_ne, (c a, c b) {
            let val = crate::num::codegen_ptr_binop(fx, BinOp::Ne, a, b);
            ret.write_cvalue(fx, val);
        };

        caller_location, () {
            let caller_location = fx.get_caller_location(span);
            ret.write_cvalue(fx, caller_location);
        };

        _ if intrinsic.as_str().starts_with("atomic_fence"), () {
            fx.bcx.ins().fence();
        };
        _ if intrinsic.as_str().starts_with("atomic_singlethreadfence"), () {
            // FIXME use a compiler fence once Cranelift supports it
            fx.bcx.ins().fence();
        };
        _ if intrinsic.as_str().starts_with("atomic_load"), <T> (v ptr) {
            validate_atomic_type!(fx, intrinsic, span, T);
            let ty = fx.clif_type(T).unwrap();

            let val = fx.bcx.ins().atomic_load(ty, MemFlags::trusted(), ptr);

            let val = CValue::by_val(val, fx.layout_of(T));
            ret.write_cvalue(fx, val);
        };
        _ if intrinsic.as_str().starts_with("atomic_store"), (v ptr, c val) {
            validate_atomic_type!(fx, intrinsic, span, val.layout().ty);

            let val = val.load_scalar(fx);

            fx.bcx.ins().atomic_store(MemFlags::trusted(), val, ptr);
        };
        _ if intrinsic.as_str().starts_with("atomic_xchg"), (v ptr, c new) {
            let layout = new.layout();
            validate_atomic_type!(fx, intrinsic, span, layout.ty);
            let ty = fx.clif_type(layout.ty).unwrap();

            let new = new.load_scalar(fx);

            let old = fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Xchg, ptr, new);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        };
        _ if intrinsic.as_str().starts_with("atomic_cxchg"), (v ptr, c test_old, c new) { // both atomic_cxchg_* and atomic_cxchgweak_*
            let layout = new.layout();
            validate_atomic_type!(fx, intrinsic, span, layout.ty);

            let test_old = test_old.load_scalar(fx);
            let new = new.load_scalar(fx);

            let old = fx.bcx.ins().atomic_cas(MemFlags::trusted(), ptr, test_old, new);
            let is_eq = fx.bcx.ins().icmp(IntCC::Equal, old, test_old);

            let ret_val = CValue::by_val_pair(old, fx.bcx.ins().bint(types::I8, is_eq), ret.layout());
            ret.write_cvalue(fx, ret_val)
        };

        _ if intrinsic.as_str().starts_with("atomic_xadd"), (v ptr, c amount) {
            let layout = amount.layout();
            validate_atomic_type!(fx, intrinsic, span, layout.ty);
            let ty = fx.clif_type(layout.ty).unwrap();

            let amount = amount.load_scalar(fx);

            let old = fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Add, ptr, amount);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        };
        _ if intrinsic.as_str().starts_with("atomic_xsub"), (v ptr, c amount) {
            let layout = amount.layout();
            validate_atomic_type!(fx, intrinsic, span, layout.ty);
            let ty = fx.clif_type(layout.ty).unwrap();

            let amount = amount.load_scalar(fx);

            let old = fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Sub, ptr, amount);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        };
        _ if intrinsic.as_str().starts_with("atomic_and"), (v ptr, c src) {
            let layout = src.layout();
            validate_atomic_type!(fx, intrinsic, span, layout.ty);
            let ty = fx.clif_type(layout.ty).unwrap();

            let src = src.load_scalar(fx);

            let old = fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::And, ptr, src);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        };
        _ if intrinsic.as_str().starts_with("atomic_or"), (v ptr, c src) {
            let layout = src.layout();
            validate_atomic_type!(fx, intrinsic, span, layout.ty);
            let ty = fx.clif_type(layout.ty).unwrap();

            let src = src.load_scalar(fx);

            let old = fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Or, ptr, src);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        };
        _ if intrinsic.as_str().starts_with("atomic_xor"), (v ptr, c src) {
            let layout = src.layout();
            validate_atomic_type!(fx, intrinsic, span, layout.ty);
            let ty = fx.clif_type(layout.ty).unwrap();

            let src = src.load_scalar(fx);

            let old = fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Xor, ptr, src);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        };
        _ if intrinsic.as_str().starts_with("atomic_nand"), (v ptr, c src) {
            let layout = src.layout();
            validate_atomic_type!(fx, intrinsic, span, layout.ty);
            let ty = fx.clif_type(layout.ty).unwrap();

            let src = src.load_scalar(fx);

            let old = fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Nand, ptr, src);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        };
        _ if intrinsic.as_str().starts_with("atomic_max"), (v ptr, c src) {
            let layout = src.layout();
            validate_atomic_type!(fx, intrinsic, span, layout.ty);
            let ty = fx.clif_type(layout.ty).unwrap();

            let src = src.load_scalar(fx);

            let old = fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Smax, ptr, src);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        };
        _ if intrinsic.as_str().starts_with("atomic_umax"), (v ptr, c src) {
            let layout = src.layout();
            validate_atomic_type!(fx, intrinsic, span, layout.ty);
            let ty = fx.clif_type(layout.ty).unwrap();

            let src = src.load_scalar(fx);

            let old = fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Umax, ptr, src);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        };
        _ if intrinsic.as_str().starts_with("atomic_min"), (v ptr, c src) {
            let layout = src.layout();
            validate_atomic_type!(fx, intrinsic, span, layout.ty);
            let ty = fx.clif_type(layout.ty).unwrap();

            let src = src.load_scalar(fx);

            let old = fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Smin, ptr, src);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        };
        _ if intrinsic.as_str().starts_with("atomic_umin"), (v ptr, c src) {
            let layout = src.layout();
            validate_atomic_type!(fx, intrinsic, span, layout.ty);
            let ty = fx.clif_type(layout.ty).unwrap();

            let src = src.load_scalar(fx);

            let old = fx.bcx.ins().atomic_rmw(ty, MemFlags::trusted(), AtomicRmwOp::Umin, ptr, src);

            let old = CValue::by_val(old, layout);
            ret.write_cvalue(fx, old);
        };

        // In Rust floating point min and max don't propagate NaN. In Cranelift they do however.
        // For this reason it is necessary to use `a.is_nan() ? b : (a >= b ? b : a)` for `minnumf*`
        // and `a.is_nan() ? b : (a <= b ? b : a)` for `maxnumf*`. NaN checks are done by comparing
        // a float against itself. Only in case of NaN is it not equal to itself.
        minnumf32, (v a, v b) {
            let a_is_nan = fx.bcx.ins().fcmp(FloatCC::NotEqual, a, a);
            let a_ge_b = fx.bcx.ins().fcmp(FloatCC::GreaterThanOrEqual, a, b);
            let temp = fx.bcx.ins().select(a_ge_b, b, a);
            let val = fx.bcx.ins().select(a_is_nan, b, temp);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f32));
            ret.write_cvalue(fx, val);
        };
        minnumf64, (v a, v b) {
            let a_is_nan = fx.bcx.ins().fcmp(FloatCC::NotEqual, a, a);
            let a_ge_b = fx.bcx.ins().fcmp(FloatCC::GreaterThanOrEqual, a, b);
            let temp = fx.bcx.ins().select(a_ge_b, b, a);
            let val = fx.bcx.ins().select(a_is_nan, b, temp);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f64));
            ret.write_cvalue(fx, val);
        };
        maxnumf32, (v a, v b) {
            let a_is_nan = fx.bcx.ins().fcmp(FloatCC::NotEqual, a, a);
            let a_le_b = fx.bcx.ins().fcmp(FloatCC::LessThanOrEqual, a, b);
            let temp = fx.bcx.ins().select(a_le_b, b, a);
            let val = fx.bcx.ins().select(a_is_nan, b, temp);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f32));
            ret.write_cvalue(fx, val);
        };
        maxnumf64, (v a, v b) {
            let a_is_nan = fx.bcx.ins().fcmp(FloatCC::NotEqual, a, a);
            let a_le_b = fx.bcx.ins().fcmp(FloatCC::LessThanOrEqual, a, b);
            let temp = fx.bcx.ins().select(a_le_b, b, a);
            let val = fx.bcx.ins().select(a_is_nan, b, temp);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f64));
            ret.write_cvalue(fx, val);
        };

        kw.Try, (v f, v data, v _catch_fn) {
            // FIXME once unwinding is supported, change this to actually catch panics
            let f_sig = fx.bcx.func.import_signature(Signature {
                call_conv: CallConv::triple_default(fx.triple()),
                params: vec![AbiParam::new(fx.bcx.func.dfg.value_type(data))],
                returns: vec![],
            });

            fx.bcx.ins().call_indirect(f_sig, f, &[data]);

            let layout = ret.layout();
            let ret_val = CValue::const_val(fx, layout, ty::ScalarInt::null(layout.size));
            ret.write_cvalue(fx, ret_val);
        };

        fadd_fast | fsub_fast | fmul_fast | fdiv_fast | frem_fast, (c x, c y) {
            let res = crate::num::codegen_float_binop(fx, match intrinsic {
                sym::fadd_fast => BinOp::Add,
                sym::fsub_fast => BinOp::Sub,
                sym::fmul_fast => BinOp::Mul,
                sym::fdiv_fast => BinOp::Div,
                sym::frem_fast => BinOp::Rem,
                _ => unreachable!(),
            }, x, y);
            ret.write_cvalue(fx, res);
        };
        float_to_int_unchecked, (v f) {
            let res = crate::cast::clif_int_or_float_cast(
                fx,
                f,
                false,
                fx.clif_type(ret.layout().ty).unwrap(),
                type_sign(ret.layout().ty),
            );
            ret.write_cvalue(fx, CValue::by_val(res, ret.layout()));
        };

        raw_eq, <T>(v lhs_ref, v rhs_ref) {
            fn type_by_size(size: Size) -> Option<Type> {
                Type::int(size.bits().try_into().ok()?)
            }

            let size = fx.layout_of(T).layout.size;
            // FIXME add and use emit_small_memcmp
            let is_eq_value =
                if size == Size::ZERO {
                    // No bytes means they're trivially equal
                    fx.bcx.ins().iconst(types::I8, 1)
                } else if let Some(clty) = type_by_size(size) {
                    // Can't use `trusted` for these loads; they could be unaligned.
                    let mut flags = MemFlags::new();
                    flags.set_notrap();
                    let lhs_val = fx.bcx.ins().load(clty, flags, lhs_ref, 0);
                    let rhs_val = fx.bcx.ins().load(clty, flags, rhs_ref, 0);
                    let eq = fx.bcx.ins().icmp(IntCC::Equal, lhs_val, rhs_val);
                    fx.bcx.ins().bint(types::I8, eq)
                } else {
                    // Just call `memcmp` (like slices do in core) when the
                    // size is too large or it's not a power-of-two.
                    let signed_bytes = i64::try_from(size.bytes()).unwrap();
                    let bytes_val = fx.bcx.ins().iconst(fx.pointer_type, signed_bytes);
                    let params = vec![AbiParam::new(fx.pointer_type); 3];
                    let returns = vec![AbiParam::new(types::I32)];
                    let args = &[lhs_ref, rhs_ref, bytes_val];
                    let cmp = fx.lib_call("memcmp", params, returns, args)[0];
                    let eq = fx.bcx.ins().icmp_imm(IntCC::Equal, cmp, 0);
                    fx.bcx.ins().bint(types::I8, eq)
                };
            ret.write_cvalue(fx, CValue::by_val(is_eq_value, ret.layout()));
        };

        black_box, (c a) {
            // FIXME implement black_box semantics
            ret.write_cvalue(fx, a);
        };
    }

    if let Some((_, dest)) = destination {
        let ret_block = fx.get_block(dest);
        fx.bcx.ins().jump(ret_block, &[]);
    } else {
        trap_unreachable(fx, "[corruption] Diverging intrinsic returned.");
    }
}
