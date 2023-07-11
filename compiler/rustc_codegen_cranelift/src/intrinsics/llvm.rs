//! Emulate LLVM intrinsics

use crate::intrinsics::*;
use crate::prelude::*;

use rustc_middle::ty::GenericArgsRef;

pub(crate) fn codegen_llvm_intrinsic_call<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    intrinsic: &str,
    generic_args: GenericArgsRef<'tcx>,
    args: &[mir::Operand<'tcx>],
    ret: CPlace<'tcx>,
    target: Option<BasicBlock>,
) {
    if intrinsic.starts_with("llvm.aarch64") {
        return llvm_aarch64::codegen_aarch64_llvm_intrinsic_call(
            fx,
            intrinsic,
            generic_args,
            args,
            ret,
            target,
        );
    }
    if intrinsic.starts_with("llvm.x86") {
        return llvm_x86::codegen_x86_llvm_intrinsic_call(
            fx,
            intrinsic,
            generic_args,
            args,
            ret,
            target,
        );
    }

    match intrinsic {
        _ if intrinsic.starts_with("llvm.ctlz.v") => {
            intrinsic_args!(fx, args => (a); intrinsic);

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| {
                fx.bcx.ins().clz(lane)
            });
        }

        _ if intrinsic.starts_with("llvm.ctpop.v") => {
            intrinsic_args!(fx, args => (a); intrinsic);

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| {
                fx.bcx.ins().popcnt(lane)
            });
        }

        _ => {
            fx.tcx
                .sess
                .warn(format!("unsupported llvm intrinsic {}; replacing with trap", intrinsic));
            crate::trap::trap_unimplemented(fx, intrinsic);
            return;
        }
    }

    let dest = target.expect("all llvm intrinsics used by stdlib should return");
    let ret_block = fx.get_block(dest);
    fx.bcx.ins().jump(ret_block, &[]);
}
