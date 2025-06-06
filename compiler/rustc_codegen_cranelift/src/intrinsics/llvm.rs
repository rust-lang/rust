//! Emulate LLVM intrinsics

use crate::intrinsics::*;
use crate::prelude::*;

pub(crate) fn codegen_llvm_intrinsic_call<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    intrinsic: &str,
    args: &[Spanned<mir::Operand<'tcx>>],
    ret: CPlace<'tcx>,
    target: Option<BasicBlock>,
    span: Span,
) {
    if intrinsic.starts_with("llvm.aarch64") {
        return llvm_aarch64::codegen_aarch64_llvm_intrinsic_call(fx, intrinsic, args, ret, target);
    }
    if intrinsic.starts_with("llvm.x86") {
        return llvm_x86::codegen_x86_llvm_intrinsic_call(fx, intrinsic, args, ret, target, span);
    }

    match intrinsic {
        "llvm.prefetch" => {
            // Nothing to do. This is merely a perf hint.
        }

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

        _ if intrinsic.starts_with("llvm.fma.v") => {
            intrinsic_args!(fx, args => (x,y,z); intrinsic);

            simd_trio_for_each_lane(
                fx,
                x,
                y,
                z,
                ret,
                &|fx, _lane_ty, _res_lane_ty, lane_x, lane_y, lane_z| {
                    fx.bcx.ins().fma(lane_x, lane_y, lane_z)
                },
            );
        }

        "llvm.fptosi.sat.v4i32.v4f32" => {
            intrinsic_args!(fx, args => (a); intrinsic);

            simd_for_each_lane(fx, a, ret, &|fx, _lane_ty, _res_lane_ty, lane| {
                fx.bcx.ins().fcvt_to_sint_sat(types::I32, lane)
            });
        }

        _ => {
            fx.tcx
                .dcx()
                .warn(format!("unsupported llvm intrinsic {}; replacing with trap", intrinsic));
            let msg = format!(
                "{intrinsic} is not yet supported.\n\
                 See https://github.com/rust-lang/rustc_codegen_cranelift/issues/171\n\
                 Please open an issue at https://github.com/rust-lang/rustc_codegen_cranelift/issues"
            );
            crate::base::codegen_panic_nounwind(fx, &msg, span);
            return;
        }
    }

    let dest = target.expect("all llvm intrinsics used by stdlib should return");
    let ret_block = fx.get_block(dest);
    fx.bcx.ins().jump(ret_block, &[]);
}
