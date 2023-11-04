//! Emulate AArch64 LLVM intrinsics

use rustc_middle::ty::GenericArgsRef;

use crate::intrinsics::*;
use crate::prelude::*;

pub(crate) fn codegen_aarch64_llvm_intrinsic_call<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    intrinsic: &str,
    _args: GenericArgsRef<'tcx>,
    args: &[mir::Operand<'tcx>],
    ret: CPlace<'tcx>,
    target: Option<BasicBlock>,
) {
    // llvm.aarch64.neon.sqshl.v*i*

    match intrinsic {
        "llvm.aarch64.isb" => {
            fx.bcx.ins().fence();
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.abs.v") => {
            intrinsic_args!(fx, args => (a); intrinsic);

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| {
                fx.bcx.ins().iabs(lane)
            });
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.cls.v") => {
            intrinsic_args!(fx, args => (a); intrinsic);

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| {
                fx.bcx.ins().cls(lane)
            });
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.rbit.v") => {
            intrinsic_args!(fx, args => (a); intrinsic);

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| {
                fx.bcx.ins().bitrev(lane)
            });
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.sqadd.v")
            || intrinsic.starts_with("llvm.aarch64.neon.uqadd.v") =>
        {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_pair_for_each_lane_typed(fx, x, y, ret, &|fx, x_lane, y_lane| {
                crate::num::codegen_saturating_int_binop(fx, BinOp::Add, x_lane, y_lane)
            });
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.sqsub.v")
            || intrinsic.starts_with("llvm.aarch64.neon.uqsub.v") =>
        {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_pair_for_each_lane_typed(fx, x, y, ret, &|fx, x_lane, y_lane| {
                crate::num::codegen_saturating_int_binop(fx, BinOp::Sub, x_lane, y_lane)
            });
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.smax.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| {
                    let gt = fx.bcx.ins().icmp(IntCC::SignedGreaterThan, x_lane, y_lane);
                    fx.bcx.ins().select(gt, x_lane, y_lane)
                },
            );
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.umax.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| {
                    let gt = fx.bcx.ins().icmp(IntCC::UnsignedGreaterThan, x_lane, y_lane);
                    fx.bcx.ins().select(gt, x_lane, y_lane)
                },
            );
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.smaxv.i") => {
            intrinsic_args!(fx, args => (v); intrinsic);

            simd_reduce(fx, v, None, ret, &|fx, _ty, a, b| {
                let gt = fx.bcx.ins().icmp(IntCC::SignedGreaterThan, a, b);
                fx.bcx.ins().select(gt, a, b)
            });
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.umaxv.i") => {
            intrinsic_args!(fx, args => (v); intrinsic);

            simd_reduce(fx, v, None, ret, &|fx, _ty, a, b| {
                let gt = fx.bcx.ins().icmp(IntCC::UnsignedGreaterThan, a, b);
                fx.bcx.ins().select(gt, a, b)
            });
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.smin.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| {
                    let gt = fx.bcx.ins().icmp(IntCC::SignedLessThan, x_lane, y_lane);
                    fx.bcx.ins().select(gt, x_lane, y_lane)
                },
            );
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.umin.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| {
                    let gt = fx.bcx.ins().icmp(IntCC::UnsignedLessThan, x_lane, y_lane);
                    fx.bcx.ins().select(gt, x_lane, y_lane)
                },
            );
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.sminv.i") => {
            intrinsic_args!(fx, args => (v); intrinsic);

            simd_reduce(fx, v, None, ret, &|fx, _ty, a, b| {
                let gt = fx.bcx.ins().icmp(IntCC::SignedLessThan, a, b);
                fx.bcx.ins().select(gt, a, b)
            });
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.uminv.i") => {
            intrinsic_args!(fx, args => (v); intrinsic);

            simd_reduce(fx, v, None, ret, &|fx, _ty, a, b| {
                let gt = fx.bcx.ins().icmp(IntCC::UnsignedLessThan, a, b);
                fx.bcx.ins().select(gt, a, b)
            });
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.umaxp.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_horizontal_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| fx.bcx.ins().umax(x_lane, y_lane),
            );
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.smaxp.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_horizontal_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| fx.bcx.ins().smax(x_lane, y_lane),
            );
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.uminp.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_horizontal_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| fx.bcx.ins().umin(x_lane, y_lane),
            );
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.sminp.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_horizontal_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| fx.bcx.ins().smin(x_lane, y_lane),
            );
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.fminp.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_horizontal_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| fx.bcx.ins().fmin(x_lane, y_lane),
            );
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.fmaxp.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_horizontal_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| fx.bcx.ins().fmax(x_lane, y_lane),
            );
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.addp.v") => {
            intrinsic_args!(fx, args => (x, y); intrinsic);

            simd_horizontal_pair_for_each_lane(
                fx,
                x,
                y,
                ret,
                &|fx, _lane_ty, _res_lane_ty, x_lane, y_lane| fx.bcx.ins().iadd(x_lane, y_lane),
            );
        }

        // FIXME generalize vector types
        "llvm.aarch64.neon.tbl1.v16i8" => {
            intrinsic_args!(fx, args => (t, idx); intrinsic);

            let zero = fx.bcx.ins().iconst(types::I8, 0);
            for i in 0..16 {
                let idx_lane = idx.value_lane(fx, i).load_scalar(fx);
                let is_zero =
                    fx.bcx.ins().icmp_imm(IntCC::UnsignedGreaterThanOrEqual, idx_lane, 16);
                let t_idx = fx.bcx.ins().uextend(fx.pointer_type, idx_lane);
                let t_lane = t.value_lane_dyn(fx, t_idx).load_scalar(fx);
                let res = fx.bcx.ins().select(is_zero, zero, t_lane);
                ret.place_lane(fx, i).to_ptr().store(fx, res, MemFlags::trusted());
            }
        }

        /*
        _ if intrinsic.starts_with("llvm.aarch64.neon.sshl.v")
            || intrinsic.starts_with("llvm.aarch64.neon.sqshl.v")
            // FIXME split this one out once saturating is implemented
            || intrinsic.starts_with("llvm.aarch64.neon.sqshlu.v") =>
        {
            intrinsic_args!(fx, args => (a, b); intrinsic);

            simd_pair_for_each_lane(fx, a, b, ret, &|fx, _lane_ty, _res_lane_ty, a, b| {
                // FIXME saturate?
                fx.bcx.ins().ishl(a, b)
            });
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.sqshrn.v") => {
            let (a, imm32) = match args {
                [a, imm32] => (a, imm32),
                _ => bug!("wrong number of args for intrinsic {intrinsic}"),
            };
            let a = codegen_operand(fx, a);
            let imm32 = crate::constant::mir_operand_get_const_val(fx, imm32)
                .expect("llvm.aarch64.neon.sqshrn.v* imm32 not const");

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| match imm32
                .try_to_bits(Size::from_bytes(4))
                .unwrap_or_else(|| panic!("imm32 not scalar: {:?}", imm32))
            {
                imm32 if imm32 < 32 => fx.bcx.ins().sshr_imm(lane, i64::from(imm32 as u8)),
                _ => fx.bcx.ins().iconst(types::I32, 0),
            });
        }

        _ if intrinsic.starts_with("llvm.aarch64.neon.sqshrun.v") => {
            let (a, imm32) = match args {
                [a, imm32] => (a, imm32),
                _ => bug!("wrong number of args for intrinsic {intrinsic}"),
            };
            let a = codegen_operand(fx, a);
            let imm32 = crate::constant::mir_operand_get_const_val(fx, imm32)
                .expect("llvm.aarch64.neon.sqshrn.v* imm32 not const");

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| match imm32
                .try_to_bits(Size::from_bytes(4))
                .unwrap_or_else(|| panic!("imm32 not scalar: {:?}", imm32))
            {
                imm32 if imm32 < 32 => fx.bcx.ins().ushr_imm(lane, i64::from(imm32 as u8)),
                _ => fx.bcx.ins().iconst(types::I32, 0),
            });
        }
        */
        _ => {
            fx.tcx.sess.warn(format!(
                "unsupported AArch64 llvm intrinsic {}; replacing with trap",
                intrinsic
            ));
            crate::trap::trap_unimplemented(fx, intrinsic);
            return;
        }
    }

    let dest = target.expect("all llvm intrinsics used by stdlib should return");
    let ret_block = fx.get_block(dest);
    fx.bcx.ins().jump(ret_block, &[]);
}
