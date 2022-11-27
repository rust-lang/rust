//! Emulate LLVM intrinsics

use crate::intrinsics::*;
use crate::prelude::*;

use rustc_middle::ty::subst::SubstsRef;

pub(crate) fn codegen_llvm_intrinsic_call<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    intrinsic: &str,
    substs: SubstsRef<'tcx>,
    args: &[mir::Operand<'tcx>],
    ret: CPlace<'tcx>,
    target: Option<BasicBlock>,
) {
    if intrinsic.starts_with("llvm.x86") {
        return llvm_x86::codegen_x86_llvm_intrinsic_call(fx, intrinsic, substs, args, ret, target);
    }

    match intrinsic {
        _ => {
            fx.tcx
                .sess
                .warn(&format!("unsupported llvm intrinsic {}; replacing with trap", intrinsic));
            crate::trap::trap_unimplemented(fx, intrinsic);
            return;
        }
    }

    let dest = target.expect("all llvm intrinsics used by stdlib should return");
    let ret_block = fx.get_block(dest);
    fx.bcx.ins().jump(ret_block, &[]);
}

