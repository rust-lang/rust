//! Codegen `extern "platform-intrinsic"` intrinsics.

use super::*;
use crate::prelude::*;

pub(super) fn codegen_simd_intrinsic_call<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    instance: Instance<'tcx>,
    args: &[mir::Operand<'tcx>],
    ret: CPlace<'tcx>,
    span: Span,
) {
    let def_id = instance.def_id();
    let substs = instance.substs;

    let intrinsic = fx.tcx.item_name(def_id).as_str();
    let intrinsic = &intrinsic[..];

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
        _ if intrinsic.starts_with("simd_shuffle"), (c x, c y, o idx) {
            validate_simd_type!(fx, intrinsic, span, x.layout().ty);

            let n: u16 = intrinsic["simd_shuffle".len()..].parse().unwrap();

            assert_eq!(x.layout(), y.layout());
            let layout = x.layout();

            let (lane_type, lane_count) = lane_type_and_count(fx.tcx, layout);
            let (ret_lane_type, ret_lane_count) = lane_type_and_count(fx.tcx, ret.layout());

            assert_eq!(lane_type, ret_lane_type);
            assert_eq!(n, ret_lane_count);

            let total_len = lane_count * 2;

            let indexes = {
                use rustc_middle::mir::interpret::*;
                let idx_const = crate::constant::mir_operand_get_const_val(fx, idx).expect("simd_shuffle* idx not const");

                let idx_bytes = match idx_const.val {
                    ty::ConstKind::Value(ConstValue::ByRef { alloc, offset }) => {
                        let ptr = Pointer::new(AllocId(0 /* dummy */), offset);
                        let size = Size::from_bytes(4 * u64::from(ret_lane_count) /* size_of([u32; ret_lane_count]) */);
                        alloc.get_bytes(fx, ptr, size).unwrap()
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
                assert!(idx < total_len, "idx {} out of range 0..{}", idx, total_len);
            }

            for (out_idx, in_idx) in indexes.into_iter().enumerate() {
                let in_lane = if in_idx < lane_count {
                    x.value_field(fx, mir::Field::new(in_idx.into()))
                } else {
                    y.value_field(fx, mir::Field::new((in_idx - lane_count).into()))
                };
                let out_lane = ret.place_field(fx, mir::Field::new(out_idx));
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

            let idx = idx_const.val.try_to_bits(Size::from_bytes(4 /* u32*/)).unwrap_or_else(|| panic!("kind not scalar: {:?}", idx_const));
            let (_lane_type, lane_count) = lane_type_and_count(fx.tcx, base.layout());
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

            let idx = idx_const.val.try_to_bits(Size::from_bytes(4 /* u32*/)).unwrap_or_else(|| panic!("kind not scalar: {:?}", idx_const));
            let (_lane_type, lane_count) = lane_type_and_count(fx.tcx, v.layout());
            if idx >= lane_count.into() {
                fx.tcx.sess.span_fatal(fx.mir.span, &format!("[simd_extract] idx {} >= lane_count {}", idx, lane_count));
            }

            let ret_lane = v.value_field(fx, mir::Field::new(idx.try_into().unwrap()));
            ret.write_cvalue(fx, ret_lane);
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

            let (_lane_layout, lane_count) = lane_type_and_count(fx.tcx, layout);
            let (ret_lane_layout, ret_lane_count) = lane_type_and_count(fx.tcx, ret.layout());
            assert_eq!(lane_count, ret_lane_count);

            for lane in 0..lane_count {
                let lane = mir::Field::new(lane.into());
                let a_lane = a.value_field(fx, lane).load_scalar(fx);
                let b_lane = b.value_field(fx, lane).load_scalar(fx);
                let c_lane = c.value_field(fx, lane).load_scalar(fx);

                let mul_lane = fx.bcx.ins().fmul(a_lane, b_lane);
                let res_lane = CValue::by_val(fx.bcx.ins().fadd(mul_lane, c_lane), ret_lane_layout);

                ret.place_field(fx, lane).write_cvalue(fx, res_lane);
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

        simd_reduce_add_ordered | simd_reduce_add_unordered, (c v) {
            validate_simd_type!(fx, intrinsic, span, v.layout().ty);
            simd_reduce(fx, v, ret, |fx, lane_layout, a, b| {
                if lane_layout.ty.is_floating_point() {
                    fx.bcx.ins().fadd(a, b)
                } else {
                    fx.bcx.ins().iadd(a, b)
                }
            });
        };

        simd_reduce_mul_ordered | simd_reduce_mul_unordered, (c v) {
            validate_simd_type!(fx, intrinsic, span, v.layout().ty);
            simd_reduce(fx, v, ret, |fx, lane_layout, a, b| {
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

        // simd_fabs
        // simd_saturating_add
        // simd_bitmask
        // simd_select
        // simd_rem
    }
}
