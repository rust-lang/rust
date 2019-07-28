use crate::prelude::*;

use rustc::ty::subst::SubstsRef;

pub fn codegen_llvm_intrinsic_call<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    intrinsic: &str,
    substs: SubstsRef<'tcx>,
    args: Vec<CValue<'tcx>>,
    destination: Option<(CPlace<'tcx>, BasicBlock)>,
) {
    fx.tcx.sess.warn(&format!("unsupported llvm intrinsic {}; replacing with trap", intrinsic));
    crate::trap::trap_unimplemented(fx, intrinsic);

    if let Some((_, dest)) = destination {
        let ret_ebb = fx.get_ebb(dest);
        fx.bcx.ins().jump(ret_ebb, &[]);
    } else {
        trap_unreachable(fx, "[corruption] Diverging intrinsic returned.");
    }
}

// llvm.x86.sse2.pmovmskb.128
// llvm.x86.avx2.vperm2i128
// llvm.x86.ssse3.pshuf.b.128
// llvm.x86.avx2.pshuf.b
// llvm.x86.avx2.pmovmskb
// llvm.x86.avx2.psrli.w
// llvm.x86.sse2.psrli.w
