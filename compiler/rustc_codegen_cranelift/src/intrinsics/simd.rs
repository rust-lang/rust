//! Codegen `extern "platform-intrinsic"` intrinsics.

use super::*;
use crate::prelude::*;

pub(super) fn codegen_simd_intrinsic_call<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    instance: Instance<'tcx>,
    args: &[mir::Operand<'tcx>],
    ret: CPlace<'tcx>,
    span: Span,
) {
    let def_id = instance.def_id();
    let substs = instance.substs;

    let intrinsic = fx.tcx.item_name(def_id);

    intrinsic_match! {
        fx, intrinsic, substs, args,
        _ => {
            fx.tcx.sess.span_fatal(span, &format!("Unknown SIMD intrinsic {}", intrinsic));
        };

        simd_cast, (c a) {
            validate_simd_type!(fx, intrinsic, span, a.layout().ty);
            simd_for_each_lane(fx, a, ret, |fx, lane_layout, ret_lane_layout, lane| {
                let ret_lane_ty = fx.clif_type(ret_lane_layout.ty).unwrap();

                let from_signed = type_sign(lane_layout.ty);
                let to_signed = type_sign(ret_lane_layout.ty);

                let ret_lane = clif_int_or_float_cast(fx, lane, from_signed, ret_lane_ty, to_signed);
                CValue::by_val(ret_lane, ret_lane_layout)
            });
        };

        simd_eq, (c x, c y) {
            validate_simd_type!(fx, intrinsic, span, x.layout().ty);
            simd_cmp!(fx, Equal|Equal(x, y) -> ret);
        };
        simd_ne, (c x, c y) {
            validate_simd_type!(fx, intrinsic, span, x.layout().ty);
            simd_cmp!(fx, NotEqual|NotEqual(x, y) -> ret);
        };
        simd_lt, (c x, c y) {
            validate_simd_type!(fx, intrinsic, span, x.layout().ty);
            simd_cmp!(fx, UnsignedLessThan|SignedLessThan|LessThan(x, y) -> ret);
        };
        simd_le, (c x, c y) {
            validate_simd_type!(fx, intrinsic, span, x.layout().ty);
            simd_cmp!(fx, UnsignedLessThanOrEqual|SignedLessThanOrEqual|LessThanOrEqual(x, y) -> ret);
        };
        simd_gt, (c x, c y) {
            validate_simd_type!(fx, intrinsic, span, x.layout().ty);
            simd_cmp!(fx, UnsignedGreaterThan|SignedGreaterThan|GreaterThan(x, y) -> ret);
        };
        simd_ge, (c x, c y) {
            validate_simd_type!(fx, intrinsic, span, x.layout().ty);
            simd_cmp!(
                fx,
                UnsignedGreaterThanOrEqual|SignedGreaterThanOrEqual|GreaterThanOrEqual
                (x, y) -> ret
            );
        };

        // simd_shuffle32<T, U>(x: T, y: T, idx: [u32; 32]) -> U
        _ if intrinsic.as_str().starts_with("simd_shuffle"), (c x, c y, o idx) {
            validate_simd_type!(fx, intrinsic, span, x.layout().ty);

            // If this intrinsic is the older "simd_shuffleN" form, simply parse the integer.
            // If there is no suffix, use the index array length.
            let n: u16 = if intrinsic == sym::simd_shuffle {
                // Make sure this is actually an array, since typeck only checks the length-suffixed
                // version of this intrinsic.
                let idx_ty = fx.monomorphize(idx.ty(fx.mir, fx.tcx));
                match idx_ty.kind() {
                    ty::Array(ty, len) if matches!(ty.kind(), ty::Uint(ty::UintTy::U32)) => {
                        len.try_eval_usize(fx.tcx, ty::ParamEnv::reveal_all()).unwrap_or_else(|| {
                            span_bug!(span, "could not evaluate shuffle index array length")
                        }).try_into().unwrap()
                    }
                    _ => {
                        fx.tcx.sess.span_err(
                            span,
                            &format!(
                                "simd_shuffle index must be an array of `u32`, got `{}`",
                                idx_ty,
                            ),
                        );
                        // Prevent verifier error
                        crate::trap::trap_unreachable(fx, "compilation should not have succeeded");
                        return;
                    }
                }
            } else {
                intrinsic.as_str()["simd_shuffle".len()..].parse().unwrap()
            };

            assert_eq!(x.layout(), y.layout());
            let layout = x.layout();

            let (lane_count, lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
            let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);

            assert_eq!(lane_ty, ret_lane_ty);
            assert_eq!(u64::from(n), ret_lane_count);

            let total_len = lane_count * 2;

            let indexes = {
                use rustc_middle::mir::interpret::*;
                let idx_const = crate::constant::mir_operand_get_const_val(fx, idx).expect("simd_shuffle* idx not const");

                let idx_bytes = match idx_const {
                    ConstValue::ByRef { alloc, offset } => {
                        let size = Size::from_bytes(4 * ret_lane_count /* size_of([u32; ret_lane_count]) */);
                        alloc.get_bytes(fx, alloc_range(offset, size)).unwrap()
                    }
                    _ => unreachable!("{:?}", idx_const),
                };

                (0..ret_lane_count).map(|i| {
                    let i = usize::try_from(i).unwrap();
                    let idx = rustc_middle::mir::interpret::read_target_uint(
                        fx.tcx.data_layout.endian,
                        &idx_bytes[4*i.. 4*i + 4],
                    ).expect("read_target_uint");
                    u16::try_from(idx).expect("try_from u32")
                }).collect::<Vec<u16>>()
            };

            for &idx in &indexes {
                assert!(u64::from(idx) < total_len, "idx {} out of range 0..{}", idx, total_len);
            }

            for (out_idx, in_idx) in indexes.into_iter().enumerate() {
                let in_lane = if u64::from(in_idx) < lane_count {
                    x.value_lane(fx, in_idx.into())
                } else {
                    y.value_lane(fx, u64::from(in_idx) - lane_count)
                };
                let out_lane = ret.place_lane(fx, u64::try_from(out_idx).unwrap());
                out_lane.write_cvalue(fx, in_lane);
            }
        };

        simd_insert, (c base, o idx, c val) {
            // FIXME validate
            let idx_const = if let Some(idx_const) = crate::constant::mir_operand_get_const_val(fx, idx) {
                idx_const
            } else {
                fx.tcx.sess.span_fatal(
                    span,
                    "Index argument for `simd_insert` is not a constant",
                );
            };

            let idx = idx_const.try_to_bits(Size::from_bytes(4 /* u32*/)).unwrap_or_else(|| panic!("kind not scalar: {:?}", idx_const));
            let (lane_count, _lane_ty) = base.layout().ty.simd_size_and_type(fx.tcx);
            if idx >= lane_count.into() {
                fx.tcx.sess.span_fatal(fx.mir.span, &format!("[simd_insert] idx {} >= lane_count {}", idx, lane_count));
            }

            ret.write_cvalue(fx, base);
            let ret_lane = ret.place_field(fx, mir::Field::new(idx.try_into().unwrap()));
            ret_lane.write_cvalue(fx, val);
        };

        simd_extract, (c v, o idx) {
            validate_simd_type!(fx, intrinsic, span, v.layout().ty);
            let idx_const = if let Some(idx_const) = crate::constant::mir_operand_get_const_val(fx, idx) {
                idx_const
            } else {
                fx.tcx.sess.span_warn(
                    span,
                    "Index argument for `simd_extract` is not a constant",
                );
                let res = crate::trap::trap_unimplemented_ret_value(
                    fx,
                    ret.layout(),
                    "Index argument for `simd_extract` is not a constant",
                );
                ret.write_cvalue(fx, res);
                return;
            };

            let idx = idx_const.try_to_bits(Size::from_bytes(4 /* u32*/)).unwrap_or_else(|| panic!("kind not scalar: {:?}", idx_const));
            let (lane_count, _lane_ty) = v.layout().ty.simd_size_and_type(fx.tcx);
            if idx >= lane_count.into() {
                fx.tcx.sess.span_fatal(fx.mir.span, &format!("[simd_extract] idx {} >= lane_count {}", idx, lane_count));
            }

            let ret_lane = v.value_lane(fx, idx.try_into().unwrap());
            ret.write_cvalue(fx, ret_lane);
        };

        simd_neg, (c a) {
            validate_simd_type!(fx, intrinsic, span, a.layout().ty);
            simd_for_each_lane(fx, a, ret, |fx, lane_layout, ret_lane_layout, lane| {
                let ret_lane = match lane_layout.ty.kind() {
                    ty::Int(_) => fx.bcx.ins().ineg(lane),
                    ty::Float(_) => fx.bcx.ins().fneg(lane),
                    _ => unreachable!(),
                };
                CValue::by_val(ret_lane, ret_lane_layout)
            });
        };

        simd_fabs, (c a) {
            validate_simd_type!(fx, intrinsic, span, a.layout().ty);
            simd_for_each_lane(fx, a, ret, |fx, _lane_layout, ret_lane_layout, lane| {
                let ret_lane = fx.bcx.ins().fabs(lane);
                CValue::by_val(ret_lane, ret_lane_layout)
            });
        };

        simd_fsqrt, (c a) {
            validate_simd_type!(fx, intrinsic, span, a.layout().ty);
            simd_for_each_lane(fx, a, ret, |fx, _lane_layout, ret_lane_layout, lane| {
                let ret_lane = fx.bcx.ins().sqrt(lane);
                CValue::by_val(ret_lane, ret_lane_layout)
            });
        };

        simd_add, (c x, c y) {
            validate_simd_type!(fx, intrinsic, span, x.layout().ty);
            simd_int_flt_binop!(fx, iadd|fadd(x, y) -> ret);
        };
        simd_sub, (c x, c y) {
            validate_simd_type!(fx, intrinsic, span, x.layout().ty);
            simd_int_flt_binop!(fx, isub|fsub(x, y) -> ret);
        };
        simd_mul, (c x, c y) {
            validate_simd_type!(fx, intrinsic, span, x.layout().ty);
            simd_int_flt_binop!(fx, imul|fmul(x, y) -> ret);
        };
        simd_div, (c x, c y) {
            validate_simd_type!(fx, intrinsic, span, x.layout().ty);
            simd_int_flt_binop!(fx, udiv|sdiv|fdiv(x, y) -> ret);
        };
        simd_rem, (c x, c y) {
            validate_simd_type!(fx, intrinsic, span, x.layout().ty);
            simd_pair_for_each_lane(fx, x, y, ret, |fx, lane_layout, ret_lane_layout, x_lane, y_lane| {
                let res_lane = match lane_layout.ty.kind() {
                    ty::Uint(_) => fx.bcx.ins().urem(x_lane, y_lane),
                    ty::Int(_) => fx.bcx.ins().srem(x_lane, y_lane),
                    ty::Float(FloatTy::F32) => fx.lib_call(
                        "fmodf",
                        vec![AbiParam::new(types::F32), AbiParam::new(types::F32)],
                        vec![AbiParam::new(types::F32)],
                        &[x_lane, y_lane],
                    )[0],
                    ty::Float(FloatTy::F64) => fx.lib_call(
                        "fmod",
                        vec![AbiParam::new(types::F64), AbiParam::new(types::F64)],
                        vec![AbiParam::new(types::F64)],
                        &[x_lane, y_lane],
                    )[0],
                    _ => unreachable!("{:?}", lane_layout.ty),
                };
                CValue::by_val(res_lane, ret_lane_layout)
            });
        };
        simd_shl, (c x, c y) {
            validate_simd_type!(fx, intrinsic, span, x.layout().ty);
            simd_int_binop!(fx, ishl(x, y) -> ret);
        };
        simd_shr, (c x, c y) {
            validate_simd_type!(fx, intrinsic, span, x.layout().ty);
            simd_int_binop!(fx, ushr|sshr(x, y) -> ret);
        };
        simd_and, (c x, c y) {
            validate_simd_type!(fx, intrinsic, span, x.layout().ty);
            simd_int_binop!(fx, band(x, y) -> ret);
        };
        simd_or, (c x, c y) {
            validate_simd_type!(fx, intrinsic, span, x.layout().ty);
            simd_int_binop!(fx, bor(x, y) -> ret);
        };
        simd_xor, (c x, c y) {
            validate_simd_type!(fx, intrinsic, span, x.layout().ty);
            simd_int_binop!(fx, bxor(x, y) -> ret);
        };

        simd_fma, (c a, c b, c c) {
            validate_simd_type!(fx, intrinsic, span, a.layout().ty);
            assert_eq!(a.layout(), b.layout());
            assert_eq!(a.layout(), c.layout());
            let layout = a.layout();

            let (lane_count, _lane_ty) = layout.ty.simd_size_and_type(fx.tcx);
            let (ret_lane_count, ret_lane_ty) = ret.layout().ty.simd_size_and_type(fx.tcx);
            assert_eq!(lane_count, ret_lane_count);
            let ret_lane_layout = fx.layout_of(ret_lane_ty);

            for lane in 0..lane_count {
                let a_lane = a.value_lane(fx, lane).load_scalar(fx);
                let b_lane = b.value_lane(fx, lane).load_scalar(fx);
                let c_lane = c.value_lane(fx, lane).load_scalar(fx);

                let mul_lane = fx.bcx.ins().fmul(a_lane, b_lane);
                let res_lane = CValue::by_val(fx.bcx.ins().fadd(mul_lane, c_lane), ret_lane_layout);

                ret.place_lane(fx, lane).write_cvalue(fx, res_lane);
            }
        };

        simd_fmin, (c x, c y) {
            validate_simd_type!(fx, intrinsic, span, x.layout().ty);
            simd_flt_binop!(fx, fmin(x, y) -> ret);
        };
        simd_fmax, (c x, c y) {
            validate_simd_type!(fx, intrinsic, span, x.layout().ty);
            simd_flt_binop!(fx, fmax(x, y) -> ret);
        };

        simd_round, (c a) {
            validate_simd_type!(fx, intrinsic, span, a.layout().ty);
            simd_for_each_lane(fx, a, ret, |fx, lane_layout, ret_lane_layout, lane| {
                let res_lane = match lane_layout.ty.kind() {
                    ty::Float(FloatTy::F32) => fx.lib_call(
                        "roundf",
                        vec![AbiParam::new(types::F32)],
                        vec![AbiParam::new(types::F32)],
                        &[lane],
                    )[0],
                    ty::Float(FloatTy::F64) => fx.lib_call(
                        "round",
                        vec![AbiParam::new(types::F64)],
                        vec![AbiParam::new(types::F64)],
                        &[lane],
                    )[0],
                    _ => unreachable!("{:?}", lane_layout.ty),
                };
                CValue::by_val(res_lane, ret_lane_layout)
            });
        };
        simd_ceil, (c a) {
            validate_simd_type!(fx, intrinsic, span, a.layout().ty);
            simd_for_each_lane(fx, a, ret, |fx, _lane_layout, ret_lane_layout, lane| {
                let ret_lane = fx.bcx.ins().ceil(lane);
                CValue::by_val(ret_lane, ret_lane_layout)
            });
        };
        simd_floor, (c a) {
            validate_simd_type!(fx, intrinsic, span, a.layout().ty);
            simd_for_each_lane(fx, a, ret, |fx, _lane_layout, ret_lane_layout, lane| {
                let ret_lane = fx.bcx.ins().floor(lane);
                CValue::by_val(ret_lane, ret_lane_layout)
            });
        };
        simd_trunc, (c a) {
            validate_simd_type!(fx, intrinsic, span, a.layout().ty);
            simd_for_each_lane(fx, a, ret, |fx, _lane_layout, ret_lane_layout, lane| {
                let ret_lane = fx.bcx.ins().trunc(lane);
                CValue::by_val(ret_lane, ret_lane_layout)
            });
        };

        simd_reduce_add_ordered | simd_reduce_add_unordered, (c v, v acc) {
            validate_simd_type!(fx, intrinsic, span, v.layout().ty);
            simd_reduce(fx, v, Some(acc), ret, |fx, lane_layout, a, b| {
                if lane_layout.ty.is_floating_point() {
                    fx.bcx.ins().fadd(a, b)
                } else {
                    fx.bcx.ins().iadd(a, b)
                }
            });
        };

        simd_reduce_mul_ordered | simd_reduce_mul_unordered, (c v, v acc) {
            validate_simd_type!(fx, intrinsic, span, v.layout().ty);
            simd_reduce(fx, v, Some(acc), ret, |fx, lane_layout, a, b| {
                if lane_layout.ty.is_floating_point() {
                    fx.bcx.ins().fmul(a, b)
                } else {
                    fx.bcx.ins().imul(a, b)
                }
            });
        };

        simd_reduce_all, (c v) {
            validate_simd_type!(fx, intrinsic, span, v.layout().ty);
            simd_reduce_bool(fx, v, ret, |fx, a, b| fx.bcx.ins().band(a, b));
        };

        simd_reduce_any, (c v) {
            validate_simd_type!(fx, intrinsic, span, v.layout().ty);
            simd_reduce_bool(fx, v, ret, |fx, a, b| fx.bcx.ins().bor(a, b));
        };

        simd_reduce_and, (c v) {
            validate_simd_type!(fx, intrinsic, span, v.layout().ty);
            simd_reduce(fx, v, None, ret, |fx, _layout, a, b| fx.bcx.ins().band(a, b));
        };

        simd_reduce_or, (c v) {
            validate_simd_type!(fx, intrinsic, span, v.layout().ty);
            simd_reduce(fx, v, None, ret, |fx, _layout, a, b| fx.bcx.ins().bor(a, b));
        };

        simd_reduce_xor, (c v) {
            validate_simd_type!(fx, intrinsic, span, v.layout().ty);
            simd_reduce(fx, v, None, ret, |fx, _layout, a, b| fx.bcx.ins().bxor(a, b));
        };

        simd_reduce_min, (c v) {
            validate_simd_type!(fx, intrinsic, span, v.layout().ty);
            simd_reduce(fx, v, None, ret, |fx, layout, a, b| {
                let lt = match layout.ty.kind() {
                    ty::Int(_) => fx.bcx.ins().icmp(IntCC::SignedLessThan, a, b),
                    ty::Uint(_) => fx.bcx.ins().icmp(IntCC::UnsignedLessThan, a, b),
                    ty::Float(_) => fx.bcx.ins().fcmp(FloatCC::LessThan, a, b),
                    _ => unreachable!(),
                };
                fx.bcx.ins().select(lt, a, b)
            });
        };

        simd_reduce_max, (c v) {
            validate_simd_type!(fx, intrinsic, span, v.layout().ty);
            simd_reduce(fx, v, None, ret, |fx, layout, a, b| {
                let gt = match layout.ty.kind() {
                    ty::Int(_) => fx.bcx.ins().icmp(IntCC::SignedGreaterThan, a, b),
                    ty::Uint(_) => fx.bcx.ins().icmp(IntCC::UnsignedGreaterThan, a, b),
                    ty::Float(_) => fx.bcx.ins().fcmp(FloatCC::GreaterThan, a, b),
                    _ => unreachable!(),
                };
                fx.bcx.ins().select(gt, a, b)
            });
        };

        simd_select, (c m, c a, c b) {
            validate_simd_type!(fx, intrinsic, span, m.layout().ty);
            validate_simd_type!(fx, intrinsic, span, a.layout().ty);
            assert_eq!(a.layout(), b.layout());

            let (lane_count, lane_ty) = a.layout().ty.simd_size_and_type(fx.tcx);
            let lane_layout = fx.layout_of(lane_ty);

            for lane in 0..lane_count {
                let m_lane = m.value_lane(fx, lane).load_scalar(fx);
                let a_lane = a.value_lane(fx, lane).load_scalar(fx);
                let b_lane = b.value_lane(fx, lane).load_scalar(fx);

                let m_lane = fx.bcx.ins().icmp_imm(IntCC::Equal, m_lane, 0);
                let res_lane = CValue::by_val(fx.bcx.ins().select(m_lane, b_lane, a_lane), lane_layout);

                ret.place_lane(fx, lane).write_cvalue(fx, res_lane);
            }
        };

        // simd_saturating_*
        // simd_bitmask
        // simd_scatter
        // simd_gather
    }
}
