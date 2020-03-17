pub mod llvm;
mod simd;

use crate::prelude::*;

macro intrinsic_pat {
    (_) => {
        _
    },
    ($name:ident) => {
        stringify!($name)
    },
    ($name:literal) => {
        stringify!($name)
    },
    ($x:ident . $($xs:tt).*) => {
        concat!(stringify!($x), ".", intrinsic_pat!($($xs).*))
    }
}

macro intrinsic_arg {
    (o $fx:expr, $arg:ident) => {
        $arg
    },
    (c $fx:expr, $arg:ident) => {
        trans_operand($fx, $arg)
    },
    (v $fx:expr, $arg:ident) => {
        trans_operand($fx, $arg).load_scalar($fx)
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
                stringify!($name) => {
                    assert!($substs.is_noop());
                    if let [$(ref $arg),*] = *$args {
                        let ($($arg,)*) = (
                            $(trans_operand($fx, $arg),)*
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

macro atomic_binop_return_old($fx:expr, $op:ident<$T:ident>($ptr:ident, $src:ident) -> $ret:ident)  {
    crate::atomic_shim::lock_global_lock($fx);

    let clif_ty = $fx.clif_type($T).unwrap();
    let old = $fx.bcx.ins().load(clif_ty, MemFlags::new(), $ptr, 0);
    let new = $fx.bcx.ins().$op(old, $src);
    $fx.bcx.ins().store(MemFlags::new(), new, $ptr, 0);
    $ret.write_cvalue($fx, CValue::by_val(old, $fx.layout_of($T)));

    crate::atomic_shim::unlock_global_lock($fx);
}

macro atomic_minmax($fx:expr, $cc:expr, <$T:ident> ($ptr:ident, $src:ident) -> $ret:ident) {
    crate::atomic_shim::lock_global_lock($fx);

    // Read old
    let clif_ty = $fx.clif_type($T).unwrap();
    let old = $fx.bcx.ins().load(clif_ty, MemFlags::new(), $ptr, 0);

    // Compare
    let is_eq = codegen_icmp($fx, IntCC::SignedGreaterThan, old, $src);
    let new = $fx.bcx.ins().select(is_eq, old, $src);

    // Write new
    $fx.bcx.ins().store(MemFlags::new(), new, $ptr, 0);

    let ret_val = CValue::by_val(old, $ret.layout());
    $ret.write_cvalue($fx, ret_val);

    crate::atomic_shim::unlock_global_lock($fx);
}

fn lane_type_and_count<'tcx>(
    tcx: TyCtxt<'tcx>,
    layout: TyLayout<'tcx>,
) -> (TyLayout<'tcx>, u32) {
    assert!(layout.ty.is_simd());
    let lane_count = match layout.fields {
        layout::FieldPlacement::Array { stride: _, count } => u32::try_from(count).unwrap(),
        _ => unreachable!("lane_type_and_count({:?})", layout),
    };
    let lane_layout = layout.field(&ty::layout::LayoutCx {
        tcx,
        param_env: ParamEnv::reveal_all(),
    }, 0).unwrap();
    (lane_layout, lane_count)
}

fn simd_for_each_lane<'tcx, B: Backend>(
    fx: &mut FunctionCx<'_, 'tcx, B>,
    val: CValue<'tcx>,
    ret: CPlace<'tcx>,
    f: impl Fn(
        &mut FunctionCx<'_, 'tcx, B>,
        TyLayout<'tcx>,
        TyLayout<'tcx>,
        Value,
    ) -> CValue<'tcx>,
) {
    let layout = val.layout();

    let (lane_layout, lane_count) = lane_type_and_count(fx.tcx, layout);
    let (ret_lane_layout, ret_lane_count) = lane_type_and_count(fx.tcx, ret.layout());
    assert_eq!(lane_count, ret_lane_count);

    for lane_idx in 0..lane_count {
        let lane_idx = mir::Field::new(lane_idx.try_into().unwrap());
        let lane = val.value_field(fx, lane_idx).load_scalar(fx);

        let res_lane = f(fx, lane_layout, ret_lane_layout, lane);

        ret.place_field(fx, lane_idx).write_cvalue(fx, res_lane);
    }
}

fn simd_pair_for_each_lane<'tcx, B: Backend>(
    fx: &mut FunctionCx<'_, 'tcx, B>,
    x: CValue<'tcx>,
    y: CValue<'tcx>,
    ret: CPlace<'tcx>,
    f: impl Fn(
        &mut FunctionCx<'_, 'tcx, B>,
        TyLayout<'tcx>,
        TyLayout<'tcx>,
        Value,
        Value,
    ) -> CValue<'tcx>,
) {
    assert_eq!(x.layout(), y.layout());
    let layout = x.layout();

    let (lane_layout, lane_count) = lane_type_and_count(fx.tcx, layout);
    let (ret_lane_layout, ret_lane_count) = lane_type_and_count(fx.tcx, ret.layout());
    assert_eq!(lane_count, ret_lane_count);

    for lane in 0..lane_count {
        let lane = mir::Field::new(lane.try_into().unwrap());
        let x_lane = x.value_field(fx, lane).load_scalar(fx);
        let y_lane = y.value_field(fx, lane).load_scalar(fx);

        let res_lane = f(fx, lane_layout, ret_lane_layout, x_lane, y_lane);

        ret.place_field(fx, lane).write_cvalue(fx, res_lane);
    }
}

fn bool_to_zero_or_max_uint<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    layout: TyLayout<'tcx>,
    val: Value,
) -> CValue<'tcx> {
    let ty = fx.clif_type(layout.ty).unwrap();

    let int_ty = match ty {
        types::F32 => types::I32,
        types::F64 => types::I64,
        ty => ty,
    };

    let zero = fx.bcx.ins().iconst(int_ty, 0);
    let max = fx
        .bcx
        .ins()
        .iconst(int_ty, (u64::max_value() >> (64 - int_ty.bits())) as i64);
    let mut res = fx.bcx.ins().select(val, max, zero);

    if ty.is_float() {
        res = fx.bcx.ins().bitcast(ty, res);
    }

    CValue::by_val(res, layout)
}

macro simd_cmp {
    ($fx:expr, $cc:ident($x:ident, $y:ident) -> $ret:ident) => {
        simd_pair_for_each_lane(
            $fx,
            $x,
            $y,
            $ret,
            |fx, lane_layout, res_lane_layout, x_lane, y_lane| {
                let res_lane = match lane_layout.ty.kind {
                    ty::Uint(_) | ty::Int(_) => codegen_icmp(fx, IntCC::$cc, x_lane, y_lane),
                    _ => unreachable!("{:?}", lane_layout.ty),
                };
                bool_to_zero_or_max_uint(fx, res_lane_layout, res_lane)
            },
        );
    },
    ($fx:expr, $cc_u:ident|$cc_s:ident($x:ident, $y:ident) -> $ret:ident) => {
        simd_pair_for_each_lane(
            $fx,
            $x,
            $y,
            $ret,
            |fx, lane_layout, res_lane_layout, x_lane, y_lane| {
                let res_lane = match lane_layout.ty.kind {
                    ty::Uint(_) => codegen_icmp(fx, IntCC::$cc_u, x_lane, y_lane),
                    ty::Int(_) => codegen_icmp(fx, IntCC::$cc_s, x_lane, y_lane),
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
                let res_lane = match lane_layout.ty.kind {
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
                let res_lane = match lane_layout.ty.kind {
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
            let res_lane = match lane_layout.ty.kind {
                ty::Float(_) => fx.bcx.ins().$op(x_lane, y_lane),
                _ => unreachable!("{:?}", lane_layout.ty),
            };
            CValue::by_val(res_lane, ret_lane_layout)
        },
    );
}

pub fn codegen_intrinsic_call<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    instance: Instance<'tcx>,
    args: &[mir::Operand<'tcx>],
    destination: Option<(CPlace<'tcx>, BasicBlock)>,
    span: Span,
) {
    let def_id = instance.def_id();
    let substs = instance.substs;

    let intrinsic = fx.tcx.item_name(def_id).as_str();
    let intrinsic = &intrinsic[..];

    let ret = match destination {
        Some((place, _)) => place,
        None => {
            // Insert non returning intrinsics here
            match intrinsic {
                "abort" => {
                    trap_panic(fx, "Called intrinsic::abort.");
                }
                "unreachable" => {
                    trap_unreachable(fx, "[corruption] Called intrinsic::unreachable.");
                }
                "transmute" => {
                    trap_unreachable(
                        fx,
                        "[corruption] Transmuting to uninhabited type.",
                    );
                }
                _ => unimplemented!("unsupported instrinsic {}", intrinsic),
            }
            return;
        }
    };

    if intrinsic.starts_with("simd_") {
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
        tanf32(flt) -> f32 => tanf,
        tanf64(flt) -> f64 => tan,
    }

    intrinsic_match! {
        fx, intrinsic, substs, args,
        _ => {
            unimpl!("unsupported intrinsic {}", intrinsic)
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
            let elem_size = fx
                .bcx
                .ins()
                .iconst(fx.pointer_type, elem_size as i64);
            assert_eq!(args.len(), 3);
            let byte_amount = fx.bcx.ins().imul(count, elem_size);

            if intrinsic.ends_with("_nonoverlapping") {
                // FIXME emit_small_memcpy
                fx.bcx.call_memcpy(fx.module.target_config(), dst, src, byte_amount);
            } else {
                // FIXME emit_small_memmove
                fx.bcx.call_memmove(fx.module.target_config(), dst, src, byte_amount);
            }
        };
        discriminant_value, (c ptr) {
            let pointee_layout = fx.layout_of(ptr.layout().ty.builtin_deref(true).unwrap().ty);
            let val = CValue::by_ref(Pointer::new(ptr.load_scalar(fx)), pointee_layout);
            let discr = crate::discriminant::codegen_get_discriminant(fx, val, ret.layout());
            ret.write_cvalue(fx, discr);
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

        _ if intrinsic.starts_with("unchecked_") || intrinsic == "exact_div", (c x, c y) {
            // FIXME trap on overflow
            let bin_op = match intrinsic {
                "unchecked_sub" => BinOp::Sub,
                "unchecked_div" | "exact_div" => BinOp::Div,
                "unchecked_rem" => BinOp::Rem,
                "unchecked_shl" => BinOp::Shl,
                "unchecked_shr" => BinOp::Shr,
                _ => unreachable!("intrinsic {}", intrinsic),
            };
            let res = crate::num::trans_int_binop(fx, bin_op, x, y);
            ret.write_cvalue(fx, res);
        };
        _ if intrinsic.ends_with("_with_overflow"), (c x, c y) {
            assert_eq!(x.layout().ty, y.layout().ty);
            let bin_op = match intrinsic {
                "add_with_overflow" => BinOp::Add,
                "sub_with_overflow" => BinOp::Sub,
                "mul_with_overflow" => BinOp::Mul,
                _ => unreachable!("intrinsic {}", intrinsic),
            };

            let res = crate::num::trans_checked_int_binop(
                fx,
                bin_op,
                x,
                y,
            );
            ret.write_cvalue(fx, res);
        };
        _ if intrinsic.starts_with("wrapping_"), (c x, c y) {
            assert_eq!(x.layout().ty, y.layout().ty);
            let bin_op = match intrinsic {
                "wrapping_add" => BinOp::Add,
                "wrapping_sub" => BinOp::Sub,
                "wrapping_mul" => BinOp::Mul,
                _ => unreachable!("intrinsic {}", intrinsic),
            };
            let res = crate::num::trans_int_binop(
                fx,
                bin_op,
                x,
                y,
            );
            ret.write_cvalue(fx, res);
        };
        _ if intrinsic.starts_with("saturating_"), <T> (c lhs, c rhs) {
            assert_eq!(lhs.layout().ty, rhs.layout().ty);
            let bin_op = match intrinsic {
                "saturating_add" => BinOp::Add,
                "saturating_sub" => BinOp::Sub,
                _ => unreachable!("intrinsic {}", intrinsic),
            };

            let signed = type_sign(T);

            let checked_res = crate::num::trans_checked_int_binop(
                fx,
                bin_op,
                lhs,
                rhs,
            );

            let (val, has_overflow) = checked_res.load_scalar_pair(fx);
            let clif_ty = fx.clif_type(T).unwrap();

            // `select.i8` is not implemented by Cranelift.
            let has_overflow = fx.bcx.ins().uextend(types::I32, has_overflow);

            let (min, max) = type_min_max_value(clif_ty, signed);
            let min = fx.bcx.ins().iconst(clif_ty, min);
            let max = fx.bcx.ins().iconst(clif_ty, max);

            let val = match (intrinsic, signed) {
                ("saturating_add", false) => fx.bcx.ins().select(has_overflow, max, val),
                ("saturating_sub", false) => fx.bcx.ins().select(has_overflow, min, val),
                ("saturating_add", true) => {
                    let rhs = rhs.load_scalar(fx);
                    let rhs_ge_zero = fx.bcx.ins().icmp_imm(IntCC::SignedGreaterThanOrEqual, rhs, 0);
                    let sat_val = fx.bcx.ins().select(rhs_ge_zero, max, min);
                    fx.bcx.ins().select(has_overflow, sat_val, val)
                }
                ("saturating_sub", true) => {
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
            let ptr_diff = fx.bcx.ins().imul_imm(offset, pointee_size as i64);
            let base_val = base.load_scalar(fx);
            let res = fx.bcx.ins().iadd(base_val, ptr_diff);
            ret.write_cvalue(fx, CValue::by_val(res, base.layout()));
        };

        transmute, <src_ty, dst_ty> (c from) {
            assert_eq!(from.layout().ty, src_ty);
            let (addr, meta) = from.force_stack(fx);
            assert!(meta.is_none());
            let dst_layout = fx.layout_of(dst_ty);
            ret.write_cvalue(fx, CValue::by_ref(addr, dst_layout))
        };
        init, () {
            let layout = ret.layout();
            if layout.abi == Abi::Uninhabited {
                crate::trap::trap_panic(fx, "[panic] Called intrinsic::init for uninhabited type.");
                return;
            }

            match *ret.inner() {
                CPlaceInner::NoPlace => {}
                CPlaceInner::Var(var) => {
                    let clif_ty = fx.clif_type(layout.ty).unwrap();
                    let val = match clif_ty {
                        types::I8 | types::I16 | types::I32 | types::I64 => fx.bcx.ins().iconst(clif_ty, 0),
                        types::I128 => {
                            let zero = fx.bcx.ins().iconst(types::I64, 0);
                            fx.bcx.ins().iconcat(zero, zero)
                        }
                        types::F32 => {
                            let zero = fx.bcx.ins().iconst(types::I32, 0);
                            fx.bcx.ins().bitcast(types::F32, zero)
                        }
                        types::F64 => {
                            let zero = fx.bcx.ins().iconst(types::I64, 0);
                            fx.bcx.ins().bitcast(types::F64, zero)
                        }
                        _ => panic!("clif_type returned {}", clif_ty),
                    };
                    fx.bcx.set_val_label(val, cranelift_codegen::ir::ValueLabel::from_u32(var.as_u32()));
                    fx.bcx.def_var(mir_var(var), val);
                }
                _ => {
                    let addr = ret.to_ptr(fx).get_addr(fx);
                    let layout = ret.layout();
                    fx.bcx.emit_small_memset(fx.module.target_config(), addr, 0, layout.size.bytes(), 1);
                }
            }
        };
        uninit, () {
            let layout = ret.layout();
            if layout.abi == Abi::Uninhabited {
                crate::trap::trap_panic(fx, "[panic] Called intrinsic::uninit for uninhabited type.");
                return;
            }
            match *ret.inner() {
                CPlaceInner::NoPlace => {},
                CPlaceInner::Var(var) => {
                    let clif_ty = fx.clif_type(layout.ty).unwrap();
                    let val = match clif_ty {
                        types::I8 | types::I16 | types::I32 | types::I64 => fx.bcx.ins().iconst(clif_ty, 42),
                        types::I128 => {
                            let zero = fx.bcx.ins().iconst(types::I64, 0);
                            let fourty_two = fx.bcx.ins().iconst(types::I64, 42);
                            fx.bcx.ins().iconcat(fourty_two, zero)
                        }
                        types::F32 => {
                            let zero = fx.bcx.ins().iconst(types::I32, 0xdeadbeef);
                            fx.bcx.ins().bitcast(types::F32, zero)
                        }
                        types::F64 => {
                            let zero = fx.bcx.ins().iconst(types::I64, 0xcafebabedeadbeefu64 as i64);
                            fx.bcx.ins().bitcast(types::F64, zero)
                        }
                        _ => panic!("clif_type returned {}", clif_ty),
                    };
                    fx.bcx.set_val_label(val, cranelift_codegen::ir::ValueLabel::from_u32(var.as_u32()));
                    fx.bcx.def_var(mir_var(var), val);
                }
                CPlaceInner::Addr(_, _) => {
                    // Don't write to `ret`, as the destination memory is already uninitialized.
                }
            }
        };
        write_bytes, (c dst, v val, v count) {
            let pointee_ty = dst.layout().ty.builtin_deref(true).unwrap().ty;
            let pointee_size = fx.layout_of(pointee_ty).size.bytes();
            let count = fx.bcx.ins().imul_imm(count, pointee_size as i64);
            let dst_ptr = dst.load_scalar(fx);
            fx.bcx.call_memset(fx.module.target_config(), dst_ptr, val, count);
        };
        ctlz | ctlz_nonzero, <T> (v arg) {
            // FIXME trap on `ctlz_nonzero` with zero arg.
            let res = if T == fx.tcx.types.u128 || T == fx.tcx.types.i128 {
                // FIXME verify this algorithm is correct
                let (lsb, msb) = fx.bcx.ins().isplit(arg);
                let lsb_lz = fx.bcx.ins().clz(lsb);
                let msb_lz = fx.bcx.ins().clz(msb);
                let msb_is_zero = fx.bcx.ins().icmp_imm(IntCC::Equal, msb, 0);
                let lsb_lz_plus_64 = fx.bcx.ins().iadd_imm(lsb_lz, 64);
                let res = fx.bcx.ins().select(msb_is_zero, lsb_lz_plus_64, msb_lz);
                fx.bcx.ins().uextend(types::I128, res)
            } else {
                fx.bcx.ins().clz(arg)
            };
            let res = CValue::by_val(res, fx.layout_of(T));
            ret.write_cvalue(fx, res);
        };
        cttz | cttz_nonzero, <T> (v arg) {
            // FIXME trap on `cttz_nonzero` with zero arg.
            let res = if T == fx.tcx.types.u128 || T == fx.tcx.types.i128 {
                // FIXME verify this algorithm is correct
                let (lsb, msb) = fx.bcx.ins().isplit(arg);
                let lsb_tz = fx.bcx.ins().ctz(lsb);
                let msb_tz = fx.bcx.ins().ctz(msb);
                let lsb_is_zero = fx.bcx.ins().icmp_imm(IntCC::Equal, lsb, 0);
                let msb_tz_plus_64 = fx.bcx.ins().iadd_imm(msb_tz, 64);
                let res = fx.bcx.ins().select(lsb_is_zero, msb_tz_plus_64, lsb_tz);
                fx.bcx.ins().uextend(types::I128, res)
            } else {
                fx.bcx.ins().ctz(arg)
            };
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
            fn swap(bcx: &mut FunctionBuilder, v: Value) -> Value {
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
            };
            let res = CValue::by_val(swap(&mut fx.bcx, arg), fx.layout_of(T));
            ret.write_cvalue(fx, res);
        };
        assert_inhabited | assert_zero_valid | assert_any_valid, <T> () {
            let layout = fx.layout_of(T);
            if layout.abi.is_uninhabited() {
                crate::trap::trap_panic(fx, &format!("attempted to instantiate uninhabited type `{}`", T));
                return;
            }

            if intrinsic == "assert_zero_valid" && !layout.might_permit_raw_init(fx, /*zero:*/ true).unwrap() {
                crate::trap::trap_panic(fx, &format!("attempted to zero-initialize type `{}`, which is invalid", T));
                return;
            }

            if intrinsic == "assert_any_valid" && !layout.might_permit_raw_init(fx, /*zero:*/ false).unwrap() {
                crate::trap::trap_panic(fx, &format!("attempted to leave type `{}` uninitialized, which is invalid", T));
                return;
            }
        };

        volatile_load, (c ptr) {
            // Cranelift treats loads as volatile by default
            let inner_layout =
                fx.layout_of(ptr.layout().ty.builtin_deref(true).unwrap().ty);
            let val = CValue::by_ref(Pointer::new(ptr.load_scalar(fx)), inner_layout);
            ret.write_cvalue(fx, val);
        };
        volatile_store, (v ptr, c val) {
            // Cranelift treats stores as volatile by default
            let dest = CPlace::for_ptr(Pointer::new(ptr), val.layout());
            dest.write_cvalue(fx, val);
        };

        size_of | pref_align_of | min_align_of | needs_drop | type_id | type_name, () {
            let const_val =
                fx.tcx.const_eval_instance(ParamEnv::reveal_all(), instance, None).unwrap();
            let val = crate::constant::trans_const_value(
                fx,
                ty::Const::from_value(fx.tcx, const_val, ret.layout().ty),
            );
            ret.write_cvalue(fx, val);
        };

        ptr_offset_from, <T> (v ptr, v base) {
            let isize_layout = fx.layout_of(fx.tcx.types.isize);

            let pointee_size: u64 = fx.layout_of(T).size.bytes();
            let diff = fx.bcx.ins().isub(ptr, base);
            // FIXME this can be an exact division.
            let val = CValue::by_val(fx.bcx.ins().udiv_imm(diff, pointee_size as i64), isize_layout);
            ret.write_cvalue(fx, val);
        };

        caller_location, () {
            let caller_location = fx.get_caller_location(span);
            ret.write_cvalue(fx, caller_location);
        };

        _ if intrinsic.starts_with("atomic_fence"), () {
            crate::atomic_shim::lock_global_lock(fx);
            crate::atomic_shim::unlock_global_lock(fx);
        };
        _ if intrinsic.starts_with("atomic_singlethreadfence"), () {
            crate::atomic_shim::lock_global_lock(fx);
            crate::atomic_shim::unlock_global_lock(fx);
        };
        _ if intrinsic.starts_with("atomic_load"), (c ptr) {
            crate::atomic_shim::lock_global_lock(fx);

            let inner_layout =
                fx.layout_of(ptr.layout().ty.builtin_deref(true).unwrap().ty);
            let val = CValue::by_ref(Pointer::new(ptr.load_scalar(fx)), inner_layout);
            ret.write_cvalue(fx, val);

            crate::atomic_shim::unlock_global_lock(fx);
        };
        _ if intrinsic.starts_with("atomic_store"), (v ptr, c val) {
            crate::atomic_shim::lock_global_lock(fx);

            let dest = CPlace::for_ptr(Pointer::new(ptr), val.layout());
            dest.write_cvalue(fx, val);

            crate::atomic_shim::unlock_global_lock(fx);
        };
        _ if intrinsic.starts_with("atomic_xchg"), <T> (v ptr, c src) {
            crate::atomic_shim::lock_global_lock(fx);

            // Read old
            let clif_ty = fx.clif_type(T).unwrap();
            let old = fx.bcx.ins().load(clif_ty, MemFlags::new(), ptr, 0);
            ret.write_cvalue(fx, CValue::by_val(old, fx.layout_of(T)));

            // Write new
            let dest = CPlace::for_ptr(Pointer::new(ptr), src.layout());
            dest.write_cvalue(fx, src);

            crate::atomic_shim::unlock_global_lock(fx);
        };
        _ if intrinsic.starts_with("atomic_cxchg"), <T> (v ptr, v test_old, v new) { // both atomic_cxchg_* and atomic_cxchgweak_*
            crate::atomic_shim::lock_global_lock(fx);

            // Read old
            let clif_ty = fx.clif_type(T).unwrap();
            let old = fx.bcx.ins().load(clif_ty, MemFlags::new(), ptr, 0);

            // Compare
            let is_eq = codegen_icmp(fx, IntCC::Equal, old, test_old);
            let new = fx.bcx.ins().select(is_eq, new, old); // Keep old if not equal to test_old

            // Write new
            fx.bcx.ins().store(MemFlags::new(), new, ptr, 0);

            let ret_val = CValue::by_val_pair(old, fx.bcx.ins().bint(types::I8, is_eq), ret.layout());
            ret.write_cvalue(fx, ret_val);

            crate::atomic_shim::unlock_global_lock(fx);
        };

        _ if intrinsic.starts_with("atomic_xadd"), <T> (v ptr, v amount) {
            atomic_binop_return_old! (fx, iadd<T>(ptr, amount) -> ret);
        };
        _ if intrinsic.starts_with("atomic_xsub"), <T> (v ptr, v amount) {
            atomic_binop_return_old! (fx, isub<T>(ptr, amount) -> ret);
        };
        _ if intrinsic.starts_with("atomic_and"), <T> (v ptr, v src) {
            atomic_binop_return_old! (fx, band<T>(ptr, src) -> ret);
        };
        _ if intrinsic.starts_with("atomic_nand"), <T> (v ptr, v src) {
            crate::atomic_shim::lock_global_lock(fx);

            let clif_ty = fx.clif_type(T).unwrap();
            let old = fx.bcx.ins().load(clif_ty, MemFlags::new(), ptr, 0);
            let and = fx.bcx.ins().band(old, src);
            let new = fx.bcx.ins().bnot(and);
            fx.bcx.ins().store(MemFlags::new(), new, ptr, 0);
            ret.write_cvalue(fx, CValue::by_val(old, fx.layout_of(T)));

            crate::atomic_shim::unlock_global_lock(fx);
        };
        _ if intrinsic.starts_with("atomic_or"), <T> (v ptr, v src) {
            atomic_binop_return_old! (fx, bor<T>(ptr, src) -> ret);
        };
        _ if intrinsic.starts_with("atomic_xor"), <T> (v ptr, v src) {
            atomic_binop_return_old! (fx, bxor<T>(ptr, src) -> ret);
        };

        _ if intrinsic.starts_with("atomic_max"), <T> (v ptr, v src) {
            atomic_minmax!(fx, IntCC::SignedGreaterThan, <T> (ptr, src) -> ret);
        };
        _ if intrinsic.starts_with("atomic_umax"), <T> (v ptr, v src) {
            atomic_minmax!(fx, IntCC::UnsignedGreaterThan, <T> (ptr, src) -> ret);
        };
        _ if intrinsic.starts_with("atomic_min"), <T> (v ptr, v src) {
            atomic_minmax!(fx, IntCC::SignedLessThan, <T> (ptr, src) -> ret);
        };
        _ if intrinsic.starts_with("atomic_umin"), <T> (v ptr, v src) {
            atomic_minmax!(fx, IntCC::UnsignedLessThan, <T> (ptr, src) -> ret);
        };

        minnumf32, (v a, v b) {
            let val = fx.bcx.ins().fmin(a, b);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f32));
            ret.write_cvalue(fx, val);
        };
        minnumf64, (v a, v b) {
            let val = fx.bcx.ins().fmin(a, b);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f64));
            ret.write_cvalue(fx, val);
        };
        maxnumf32, (v a, v b) {
            let val = fx.bcx.ins().fmax(a, b);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f32));
            ret.write_cvalue(fx, val);
        };
        maxnumf64, (v a, v b) {
            let val = fx.bcx.ins().fmax(a, b);
            let val = CValue::by_val(val, fx.layout_of(fx.tcx.types.f64));
            ret.write_cvalue(fx, val);
        };

        try, (v f, v data, v _local_ptr) {
            // FIXME once unwinding is supported, change this to actually catch panics
            let f_sig = fx.bcx.func.import_signature(Signature {
                call_conv: CallConv::triple_default(fx.triple()),
                params: vec![AbiParam::new(fx.bcx.func.dfg.value_type(data))],
                returns: vec![],
            });

            fx.bcx.ins().call_indirect(f_sig, f, &[data]);

            let ret_val = CValue::const_val(fx, ret.layout(), 0);
            ret.write_cvalue(fx, ret_val);
        };
    }

    if let Some((_, dest)) = destination {
        let ret_block = fx.get_block(dest);
        fx.bcx.ins().jump(ret_block, &[]);
    } else {
        trap_unreachable(fx, "[corruption] Diverging intrinsic returned.");
    }
}
