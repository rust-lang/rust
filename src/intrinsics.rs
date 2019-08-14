use crate::prelude::*;

use rustc::ty::subst::SubstsRef;

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

pub macro intrinsic_match {
    ($fx:expr, $intrinsic:expr, $substs:expr, $args:expr,
    _ => $unknown:block;
    $(
        $($($name:tt).*)|+ $(if $cond:expr)?, $(<$($subst:ident),*>)? ($($a:ident $arg:ident),*) $content:block;
    )*) => {
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

macro_rules! call_intrinsic_match {
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
                            let ret_ebb = $fx.get_ebb(dest);
                            $fx.bcx.ins().jump(ret_ebb, &[]);
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

macro_rules! atomic_binop_return_old {
    ($fx:expr, $op:ident<$T:ident>($ptr:ident, $src:ident) -> $ret:ident) => {
        let clif_ty = $fx.clif_type($T).unwrap();
        let old = $fx.bcx.ins().load(clif_ty, MemFlags::new(), $ptr, 0);
        let new = $fx.bcx.ins().$op(old, $src);
        $fx.bcx.ins().store(MemFlags::new(), new, $ptr, 0);
        $ret.write_cvalue($fx, CValue::by_val(old, $fx.layout_of($T)));
    };
}

macro_rules! atomic_minmax {
    ($fx:expr, $cc:expr, <$T:ident> ($ptr:ident, $src:ident) -> $ret:ident) => {
        // Read old
        let clif_ty = $fx.clif_type($T).unwrap();
        let old = $fx.bcx.ins().load(clif_ty, MemFlags::new(), $ptr, 0);

        // Compare
        let is_eq = $fx.bcx.ins().icmp(IntCC::SignedGreaterThan, old, $src);
        let new = crate::common::codegen_select(&mut $fx.bcx, is_eq, old, $src);

        // Write new
        $fx.bcx.ins().store(MemFlags::new(), new, $ptr, 0);

        let ret_val = CValue::by_val(old, $ret.layout());
        $ret.write_cvalue($fx, ret_val);
    };
}

pub fn lane_type_and_count<'tcx>(
    fx: &FunctionCx<'_, 'tcx, impl Backend>,
    layout: TyLayout<'tcx>,
    intrinsic: &str,
) -> (TyLayout<'tcx>, u32) {
    assert!(layout.ty.is_simd());
    let lane_count = match layout.fields {
        layout::FieldPlacement::Array { stride: _, count } => u32::try_from(count).unwrap(),
        _ => panic!("Non vector type {:?} passed to or returned from simd_* intrinsic {}", layout.ty, intrinsic),
    };
    let lane_layout = layout.field(fx, 0);
    (lane_layout, lane_count)
}

pub fn simd_for_each_lane<'tcx, B: Backend>(
    fx: &mut FunctionCx<'_, 'tcx, B>,
    intrinsic: &str,
    x: CValue<'tcx>,
    y: CValue<'tcx>,
    ret: CPlace<'tcx>,
    f: impl Fn(&mut FunctionCx<'_, 'tcx, B>, TyLayout<'tcx>, TyLayout<'tcx>, Value, Value) -> CValue<'tcx>,
) {
    assert_eq!(x.layout(), y.layout());
    let layout = x.layout();

    let (lane_layout, lane_count) = lane_type_and_count(fx, layout, intrinsic);
    let (ret_lane_layout, ret_lane_count) = lane_type_and_count(fx, ret.layout(), intrinsic);
    assert_eq!(lane_count, ret_lane_count);

    for lane in 0..lane_count {
        let lane = mir::Field::new(lane.try_into().unwrap());
        let x_lane = x.value_field(fx, lane).load_scalar(fx);
        let y_lane = y.value_field(fx, lane).load_scalar(fx);

        let res_lane = f(fx, lane_layout, ret_lane_layout, x_lane, y_lane);

        ret.place_field(fx, lane).write_cvalue(fx, res_lane);
    }
}

pub fn bool_to_zero_or_max_uint<'tcx>(
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
    let max = fx.bcx.ins().iconst(int_ty, (u64::max_value() >> (64 - int_ty.bits())) as i64);
    let mut res = crate::common::codegen_select(&mut fx.bcx, val, max, zero);

    if ty.is_float() {
        res = fx.bcx.ins().bitcast(ty, res);
    }

    CValue::by_val(res, layout)
}

macro_rules! simd_cmp {
    ($fx:expr, $intrinsic:expr, $cc:ident($x:ident, $y:ident) -> $ret:ident) => {
        simd_for_each_lane($fx, $intrinsic, $x, $y, $ret, |fx, lane_layout, res_lane_layout, x_lane, y_lane| {
            let res_lane = match lane_layout.ty.sty {
                ty::Uint(_) | ty::Int(_) => fx.bcx.ins().icmp(IntCC::$cc, x_lane, y_lane),
                _ => unreachable!("{:?}", lane_layout.ty),
            };
            bool_to_zero_or_max_uint(fx, res_lane_layout, res_lane)
        });
    };
    ($fx:expr, $intrinsic:expr, $cc_u:ident|$cc_s:ident($x:ident, $y:ident) -> $ret:ident) => {
        simd_for_each_lane($fx, $intrinsic, $x, $y, $ret, |fx, lane_layout, res_lane_layout, x_lane, y_lane| {
            let res_lane = match lane_layout.ty.sty {
                ty::Uint(_) => fx.bcx.ins().icmp(IntCC::$cc_u, x_lane, y_lane),
                ty::Int(_) => fx.bcx.ins().icmp(IntCC::$cc_s, x_lane, y_lane),
                _ => unreachable!("{:?}", lane_layout.ty),
            };
            bool_to_zero_or_max_uint(fx, res_lane_layout, res_lane)
        });
    };

}

macro_rules! simd_int_binop {
    ($fx:expr, $intrinsic:expr, $op:ident($x:ident, $y:ident) -> $ret:ident) => {
        simd_for_each_lane($fx, $intrinsic, $x, $y, $ret, |fx, lane_layout, ret_lane_layout, x_lane, y_lane| {
            let res_lane = match lane_layout.ty.sty {
                ty::Uint(_) | ty::Int(_) => fx.bcx.ins().$op(x_lane, y_lane),
                _ => unreachable!("{:?}", lane_layout.ty),
            };
            CValue::by_val(res_lane, ret_lane_layout)
        });
    };
    ($fx:expr, $intrinsic:expr, $op_u:ident|$op_s:ident($x:ident, $y:ident) -> $ret:ident) => {
        simd_for_each_lane($fx, $intrinsic, $x, $y, $ret, |fx, lane_layout, ret_lane_layout, x_lane, y_lane| {
            let res_lane = match lane_layout.ty.sty {
                ty::Uint(_) => fx.bcx.ins().$op_u(x_lane, y_lane),
                ty::Int(_) => fx.bcx.ins().$op_s(x_lane, y_lane),
                _ => unreachable!("{:?}", lane_layout.ty),
            };
            CValue::by_val(res_lane, ret_lane_layout)
        });
    };
}

macro_rules! simd_int_flt_binop {
    ($fx:expr, $intrinsic:expr, $op:ident|$op_f:ident($x:ident, $y:ident) -> $ret:ident) => {
        simd_for_each_lane($fx, $intrinsic, $x, $y, $ret, |fx, lane_layout, ret_lane_layout, x_lane, y_lane| {
            let res_lane = match lane_layout.ty.sty {
                ty::Uint(_) | ty::Int(_) => fx.bcx.ins().$op(x_lane, y_lane),
                ty::Float(_) => fx.bcx.ins().$op_f(x_lane, y_lane),
                _ => unreachable!("{:?}", lane_layout.ty),
            };
            CValue::by_val(res_lane, ret_lane_layout)
        });
    };
    ($fx:expr, $intrinsic:expr, $op_u:ident|$op_s:ident|$op_f:ident($x:ident, $y:ident) -> $ret:ident) => {
        simd_for_each_lane($fx, $intrinsic, $x, $y, $ret, |fx, lane_layout, ret_lane_layout, x_lane, y_lane| {
            let res_lane = match lane_layout.ty.sty {
                ty::Uint(_) => fx.bcx.ins().$op_u(x_lane, y_lane),
                ty::Int(_) => fx.bcx.ins().$op_s(x_lane, y_lane),
                ty::Float(_) => fx.bcx.ins().$op_f(x_lane, y_lane),
                _ => unreachable!("{:?}", lane_layout.ty),
            };
            CValue::by_val(res_lane, ret_lane_layout)
        });
    };
}

macro_rules! simd_flt_binop {
    ($fx:expr, $intrinsic:expr, $op:ident($x:ident, $y:ident) -> $ret:ident) => {
        simd_for_each_lane($fx, $intrinsic, $x, $y, $ret, |fx, lane_layout, ret_lane_layout, x_lane, y_lane| {
            let res_lane = match lane_layout.ty.sty {
                ty::Float(_) => fx.bcx.ins().$op(x_lane, y_lane),
                _ => unreachable!("{:?}", lane_layout.ty),
            };
            CValue::by_val(res_lane, ret_lane_layout)
        });
    }
}

pub fn codegen_intrinsic_call<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    def_id: DefId,
    substs: SubstsRef<'tcx>,
    args: &[mir::Operand<'tcx>],
    destination: Option<(CPlace<'tcx>, BasicBlock)>,
) {
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
                    trap_unreachable(fx, "[corruption] Called intrinsic::transmute with uninhabited argument.");
                }
                _ => unimplemented!("unsupported instrinsic {}", intrinsic),
            }
            return;
        }
    };

    let u64_layout = fx.layout_of(fx.tcx.types.u64);
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
        logf32(flt) -> f32 => logf,
        logf64(flt) -> f64 => log,
        log2f32(flt) -> f32 => log2f,
        log2f64(flt) -> f64 => log2,
        fabsf32(flt) -> f32 => fabsf,
        fabsf64(flt) -> f64 => fabs,
        fmaf32(x, y, z) -> f32 => fmaf,
        fmaf64(x, y, z) -> f64 => fma,

        // rounding variants
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
                fx.bcx.call_memcpy(fx.module.target_config(), dst, src, byte_amount);
            } else {
                fx.bcx.call_memmove(fx.module.target_config(), dst, src, byte_amount);
            }
        };
        discriminant_value, (c val) {
            let pointee_layout = fx.layout_of(val.layout().ty.builtin_deref(true).unwrap().ty);
            let place = CPlace::for_addr(val.load_scalar(fx), pointee_layout);
            let discr = crate::discriminant::codegen_get_discriminant(fx, place, ret.layout());
            ret.write_cvalue(fx, discr);
        };
        size_of, <T> () {
            let size_of = fx.layout_of(T).size.bytes();
            let size_of = CValue::const_val(fx, usize_layout.ty, size_of.into());
            ret.write_cvalue(fx, size_of);
        };
        size_of_val, <T> (c ptr) {
            let layout = fx.layout_of(T);
            let size = if layout.is_unsized() {
                let (_ptr, info) = ptr.load_scalar_pair(fx);
                let (size, _align) = crate::unsize::size_and_align_of_dst(fx, layout.ty, info);
                size
            } else {
                fx
                    .bcx
                    .ins()
                    .iconst(fx.pointer_type, layout.size.bytes() as i64)
            };
            ret.write_cvalue(fx, CValue::by_val(size, usize_layout));
        };
        min_align_of, <T> () {
            let min_align = fx.layout_of(T).align.abi.bytes();
            let min_align = CValue::const_val(fx, usize_layout.ty, min_align.into());
            ret.write_cvalue(fx, min_align);
        };
        min_align_of_val, <T> (c ptr) {
            let layout = fx.layout_of(T);
            let align = if layout.is_unsized() {
                let (_ptr, info) = ptr.load_scalar_pair(fx);
                let (_size, align) = crate::unsize::size_and_align_of_dst(fx, layout.ty, info);
                align
            } else {
                fx
                    .bcx
                    .ins()
                    .iconst(fx.pointer_type, layout.align.abi.bytes() as i64)
            };
            ret.write_cvalue(fx, CValue::by_val(align, usize_layout));
        };
        pref_align_of, <T> () {
            let pref_align = fx.layout_of(T).align.pref.bytes();
            let pref_align = CValue::const_val(fx, usize_layout.ty, pref_align.into());
            ret.write_cvalue(fx, pref_align);
        };


        type_id, <T> () {
            let type_id = fx.tcx.type_id_hash(T);
            let type_id = CValue::const_val(fx, u64_layout.ty, type_id.into());
            ret.write_cvalue(fx, type_id);
        };
        type_name, <T> () {
            let type_name = fx.tcx.type_name(T);
            let type_name = crate::constant::trans_const_value(fx, type_name);
            ret.write_cvalue(fx, type_name);
        };

        _ if intrinsic.starts_with("unchecked_") || intrinsic == "exact_div", (c x, c y) {
            // FIXME trap on overflow
            let bin_op = match intrinsic {
                "unchecked_sub" => BinOp::Sub,
                "unchecked_div" | "exact_div" => BinOp::Div,
                "unchecked_rem" => BinOp::Rem,
                "unchecked_shl" => BinOp::Shl,
                "unchecked_shr" => BinOp::Shr,
                _ => unimplemented!("intrinsic {}", intrinsic),
            };
            let res = crate::num::trans_int_binop(fx, bin_op, x, y, ret.layout().ty);
            ret.write_cvalue(fx, res);
        };
        _ if intrinsic.ends_with("_with_overflow"), (c x, c y) {
            assert_eq!(x.layout().ty, y.layout().ty);
            let bin_op = match intrinsic {
                "add_with_overflow" => BinOp::Add,
                "sub_with_overflow" => BinOp::Sub,
                "mul_with_overflow" => BinOp::Mul,
                _ => unimplemented!("intrinsic {}", intrinsic),
            };

            let res = crate::num::trans_checked_int_binop(
                fx,
                bin_op,
                x,
                y,
                ret.layout().ty,
            );
            ret.write_cvalue(fx, res);
        };
        _ if intrinsic.starts_with("overflowing_"), (c x, c y) {
            assert_eq!(x.layout().ty, y.layout().ty);
            let bin_op = match intrinsic {
                "overflowing_add" => BinOp::Add,
                "overflowing_sub" => BinOp::Sub,
                "overflowing_mul" => BinOp::Mul,
                _ => unimplemented!("intrinsic {}", intrinsic),
            };
            let res = crate::num::trans_int_binop(
                fx,
                bin_op,
                x,
                y,
                ret.layout().ty,
            );
            ret.write_cvalue(fx, res);
        };
        _ if intrinsic.starts_with("saturating_"), <T> (c x, c y) {
            assert_eq!(x.layout().ty, y.layout().ty);
            let bin_op = match intrinsic {
                "saturating_add" => BinOp::Add,
                "saturating_sub" => BinOp::Sub,
                _ => unimplemented!("intrinsic {}", intrinsic),
            };

            let signed = type_sign(T);

            let checked_res = crate::num::trans_checked_int_binop(
                fx,
                bin_op,
                x,
                y,
                fx.tcx.mk_tup([T, fx.tcx.types.bool].into_iter()),
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
                ("saturating_add", true) => unimplemented!(),
                ("saturating_sub", true) => unimplemented!(),
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
            let addr = from.force_stack(fx);
            let dst_layout = fx.layout_of(dst_ty);
            ret.write_cvalue(fx, CValue::by_ref(addr, dst_layout))
        };
        init, () {
            if ret.layout().abi == Abi::Uninhabited {
                crate::trap::trap_panic(fx, "[panic] Called intrinsic::init for uninhabited type.");
                return;
            }

            match ret {
                CPlace::NoPlace(_layout) => {}
                CPlace::Var(var, layout) => {
                    let clif_ty = fx.clif_type(layout.ty).unwrap();
                    let val = match clif_ty {
                        types::I8 | types::I16 | types::I32 | types::I64 => fx.bcx.ins().iconst(clif_ty, 0),
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
                    fx.bcx.def_var(mir_var(var), val);
                }
                _ => {
                    let addr = ret.to_addr(fx);
                    let layout = ret.layout();
                    fx.bcx.emit_small_memset(fx.module.target_config(), addr, 0, layout.size.bytes(), 1);
                }
            }
        };
        uninit, () {
            if ret.layout().abi == Abi::Uninhabited {
                crate::trap::trap_panic(fx, "[panic] Called intrinsic::uninit for uninhabited type.");
                return;
            }
            match ret {
                CPlace::NoPlace(_layout) => {},
                CPlace::Var(var, layout) => {
                    let clif_ty = fx.clif_type(layout.ty).unwrap();
                    let val = match clif_ty {
                        types::I8 | types::I16 | types::I32 | types::I64 => fx.bcx.ins().iconst(clif_ty, 42),
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
                    fx.bcx.def_var(mir_var(var), val);
                }
                CPlace::Addr(_, _, _) | CPlace::Stack(_, _) => {
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
                fx.bcx.ins().select(msb_is_zero, lsb_lz_plus_64, msb_lz)
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
                fx.bcx.ins().select(lsb_is_zero, msb_tz_plus_64, lsb_tz)
            } else {
                fx.bcx.ins().ctz(arg)
            };
            let res = CValue::by_val(res, fx.layout_of(T));
            ret.write_cvalue(fx, res);
        };
        ctpop, <T> (v arg) {
            let res = CValue::by_val(fx.bcx.ins().popcnt(arg), fx.layout_of(T));
            ret.write_cvalue(fx, res);
        };
        bitreverse, <T> (v arg) {
            let res = CValue::by_val(fx.bcx.ins().bitrev(arg), fx.layout_of(T));
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
                    ty => unimplemented!("bswap {}", ty),
                }
            };
            let res = CValue::by_val(swap(&mut fx.bcx, arg), fx.layout_of(T));
            ret.write_cvalue(fx, res);
        };
        needs_drop, <T> () {
            let needs_drop = if T.needs_drop(fx.tcx, ParamEnv::reveal_all()) {
                1
            } else {
                0
            };
            let needs_drop = CValue::const_val(fx, fx.tcx.types.bool, needs_drop);
            ret.write_cvalue(fx, needs_drop);
        };
        panic_if_uninhabited, <T> () {
            if fx.layout_of(T).abi.is_uninhabited() {
                crate::trap::trap_panic(fx, "[panic] Called intrinsic::panic_if_uninhabited for uninhabited type.");
                return;
            }
        };

        volatile_load, (c ptr) {
            // Cranelift treats loads as volatile by default
            let inner_layout =
                fx.layout_of(ptr.layout().ty.builtin_deref(true).unwrap().ty);
            let val = CValue::by_ref(ptr.load_scalar(fx), inner_layout);
            ret.write_cvalue(fx, val);
        };
        volatile_store, (v ptr, c val) {
            // Cranelift treats stores as volatile by default
            let dest = CPlace::for_addr(ptr, val.layout());
            dest.write_cvalue(fx, val);
        };

        _ if intrinsic.starts_with("atomic_fence"), () {};
        _ if intrinsic.starts_with("atomic_singlethreadfence"), () {};
        _ if intrinsic.starts_with("atomic_load"), (c ptr) {
            let inner_layout =
                fx.layout_of(ptr.layout().ty.builtin_deref(true).unwrap().ty);
            let val = CValue::by_ref(ptr.load_scalar(fx), inner_layout);
            ret.write_cvalue(fx, val);
        };
        _ if intrinsic.starts_with("atomic_store"), (v ptr, c val) {
            let dest = CPlace::for_addr(ptr, val.layout());
            dest.write_cvalue(fx, val);
        };
        _ if intrinsic.starts_with("atomic_xchg"), <T> (v ptr, c src) {
            // Read old
            let clif_ty = fx.clif_type(T).unwrap();
            let old = fx.bcx.ins().load(clif_ty, MemFlags::new(), ptr, 0);
            ret.write_cvalue(fx, CValue::by_val(old, fx.layout_of(T)));

            // Write new
            let dest = CPlace::for_addr(ptr, src.layout());
            dest.write_cvalue(fx, src);
        };
        _ if intrinsic.starts_with("atomic_cxchg"), <T> (v ptr, v test_old, v new) { // both atomic_cxchg_* and atomic_cxchgweak_*
            // Read old
            let clif_ty = fx.clif_type(T).unwrap();
            let old = fx.bcx.ins().load(clif_ty, MemFlags::new(), ptr, 0);

            // Compare
            let is_eq = fx.bcx.ins().icmp(IntCC::Equal, old, test_old);
            let new = crate::common::codegen_select(&mut fx.bcx, is_eq, new, old); // Keep old if not equal to test_old

            // Write new
            fx.bcx.ins().store(MemFlags::new(), new, ptr, 0);

            let ret_val = CValue::by_val_pair(old, fx.bcx.ins().bint(types::I8, is_eq), ret.layout());
            ret.write_cvalue(fx, ret_val);
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
            let clif_ty = fx.clif_type(T).unwrap();
            let old = fx.bcx.ins().load(clif_ty, MemFlags::new(), ptr, 0);
            let and = fx.bcx.ins().band(old, src);
            let new = fx.bcx.ins().bnot(and);
            fx.bcx.ins().store(MemFlags::new(), new, ptr, 0);
            ret.write_cvalue(fx, CValue::by_val(old, fx.layout_of(T)));
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

        simd_cast, (c a) {
            let (lane_layout, lane_count) = lane_type_and_count(fx, a.layout(), intrinsic);
            let (ret_lane_layout, ret_lane_count) = lane_type_and_count(fx, ret.layout(), intrinsic);
            assert_eq!(lane_count, ret_lane_count);

            let ret_lane_ty = fx.clif_type(ret_lane_layout.ty).unwrap();

            let from_signed = type_sign(lane_layout.ty);
            let to_signed = type_sign(ret_lane_layout.ty);

            for lane in 0..lane_count {
                let lane = mir::Field::new(lane.try_into().unwrap());

                let a_lane = a.value_field(fx, lane).load_scalar(fx);
                let res = clif_int_or_float_cast(fx, a_lane, from_signed, ret_lane_ty, to_signed);
                ret.place_field(fx, lane).write_cvalue(fx, CValue::by_val(res, ret_lane_layout));
            }
        };

        simd_eq, (c x, c y) {
            simd_cmp!(fx, intrinsic, Equal(x, y) -> ret);
        };
        simd_ne, (c x, c y) {
            simd_cmp!(fx, intrinsic, NotEqual(x, y) -> ret);
        };
        simd_lt, (c x, c y) {
            simd_cmp!(fx, intrinsic, UnsignedLessThan|SignedLessThan(x, y) -> ret);
        };
        simd_le, (c x, c y) {
            simd_cmp!(fx, intrinsic, UnsignedLessThanOrEqual|SignedLessThanOrEqual(x, y) -> ret);
        };
        simd_gt, (c x, c y) {
            simd_cmp!(fx, intrinsic, UnsignedGreaterThan|SignedGreaterThan(x, y) -> ret);
        };
        simd_ge, (c x, c y) {
            simd_cmp!(fx, intrinsic, UnsignedGreaterThanOrEqual|SignedGreaterThanOrEqual(x, y) -> ret);
        };

        // simd_shuffle32<T, U>(x: T, y: T, idx: [u32; 32]) -> U
        _ if intrinsic.starts_with("simd_shuffle"), (c x, c y, o idx) {
            let n: u32 = intrinsic["simd_shuffle".len()..].parse().unwrap();

            assert_eq!(x.layout(), y.layout());
            let layout = x.layout();

            let (lane_type, lane_count) = lane_type_and_count(fx, layout, intrinsic);
            let (ret_lane_type, ret_lane_count) = lane_type_and_count(fx, ret.layout(), intrinsic);

            assert_eq!(lane_type, ret_lane_type);
            assert_eq!(n, ret_lane_count);

            let total_len = lane_count * 2;

            let indexes = {
                use rustc::mir::interpret::*;
                let idx_const = crate::constant::mir_operand_get_const_val(fx, idx).expect("simd_shuffle* idx not const");

                let idx_bytes = match idx_const.val {
                    ConstValue::ByRef { alloc, offset } => {
                        let ptr = Pointer::new(AllocId(0 /* dummy */), offset);
                        let size = Size::from_bytes(4 * u64::from(ret_lane_count) /* size_of([u32; ret_lane_count]) */);
                        alloc.get_bytes(fx, ptr, size).unwrap()
                    }
                    _ => unreachable!("{:?}", idx_const),
                };

                (0..ret_lane_count).map(|i| {
                    let i = usize::try_from(i).unwrap();
                    let idx = rustc::mir::interpret::read_target_uint(
                        fx.tcx.data_layout.endian,
                        &idx_bytes[4*i.. 4*i + 4],
                    ).expect("read_target_uint");
                    u32::try_from(idx).expect("try_from u32")
                }).collect::<Vec<u32>>()
            };

            for &idx in &indexes {
                assert!(idx < total_len, "idx {} out of range 0..{}", idx, total_len);
            }

            for (out_idx, in_idx) in indexes.into_iter().enumerate() {
                let in_lane = if in_idx < lane_count {
                    x.value_field(fx, mir::Field::new(in_idx.try_into().unwrap()))
                } else {
                    y.value_field(fx, mir::Field::new((in_idx - lane_count).try_into().unwrap()))
                };
                let out_lane = ret.place_field(fx, mir::Field::new(out_idx));
                out_lane.write_cvalue(fx, in_lane);
            }
        };

        simd_extract, (c v, o idx) {
            let idx_const = crate::constant::mir_operand_get_const_val(fx, idx).expect("simd_extract* idx not const");
            let idx = idx_const.val.try_to_bits(Size::from_bytes(4 /* u32*/)).expect(&format!("kind not scalar: {:?}", idx_const));
            let (_lane_type, lane_count) = lane_type_and_count(fx, v.layout(), intrinsic);
            if idx >= lane_count.into() {
                fx.tcx.sess.span_fatal(fx.mir.span, &format!("[simd_extract] idx {} >= lane_count {}", idx, lane_count));
            }

            let ret_lane = v.value_field(fx, mir::Field::new(idx.try_into().unwrap()));
            ret.write_cvalue(fx, ret_lane);
        };

        simd_add, (c x, c y) {
            simd_int_flt_binop!(fx, intrinsic, iadd|fadd(x, y) -> ret);
        };
        simd_sub, (c x, c y) {
            simd_int_flt_binop!(fx, intrinsic, isub|fsub(x, y) -> ret);
        };
        simd_mul, (c x, c y) {
            simd_int_flt_binop!(fx, intrinsic, imul|fmul(x, y) -> ret);
        };
        simd_div, (c x, c y) {
            simd_int_flt_binop!(fx, intrinsic, udiv|sdiv|fdiv(x, y) -> ret);
        };
        simd_shl, (c x, c y) {
            simd_int_binop!(fx, intrinsic, ishl(x, y) -> ret);
        };
        simd_shr, (c x, c y) {
            simd_int_binop!(fx, intrinsic, ushr|sshr(x, y) -> ret);
        };
        simd_and, (c x, c y) {
            simd_int_binop!(fx, intrinsic, band(x, y) -> ret);
        };
        simd_or, (c x, c y) {
            simd_int_binop!(fx, intrinsic, bor(x, y) -> ret);
        };
        simd_xor, (c x, c y) {
            simd_int_binop!(fx, intrinsic, bxor(x, y) -> ret);
        };

        simd_fmin, (c x, c y) {
            simd_flt_binop!(fx, intrinsic, fmin(x, y) -> ret);
        };
        simd_fmax, (c x, c y) {
            simd_flt_binop!(fx, intrinsic, fmax(x, y) -> ret);
        };

        try, (v f, v data, v _local_ptr) {
            // FIXME once unwinding is supported, change this to actually catch panics
            let f_sig = fx.bcx.func.import_signature(Signature {
                call_conv: cranelift::codegen::isa::CallConv::SystemV,
                params: vec![AbiParam::new(fx.bcx.func.dfg.value_type(data))],
                returns: vec![],
            });

            fx.bcx.ins().call_indirect(f_sig, f, &[data]);

            let ret_val = CValue::const_val(fx, ret.layout().ty, 0);
            ret.write_cvalue(fx, ret_val);
        };
    }

    if let Some((_, dest)) = destination {
        let ret_ebb = fx.get_ebb(dest);
        fx.bcx.ins().jump(ret_ebb, &[]);
    } else {
        trap_unreachable(fx, "[corruption] Diverging intrinsic returned.");
    }
}
