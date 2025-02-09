use rustc_ast::expand::allocator::NO_ALLOC_SHIM_IS_UNSTABLE;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::{DebugInfo, OomStrategy};

use crate::common::AsCCharPtr;
use crate::llvm::{self, False};
use crate::{ModuleLlvm, debuginfo};

pub(crate) unsafe fn codegen(tcx: TyCtxt<'_>, module_llvm: &mut ModuleLlvm, module_name: &str) {
    let llcx = &*module_llvm.llcx;
    let llmod = module_llvm.llmod();
    let i8 = unsafe { llvm::LLVMInt8TypeInContext(llcx) };

    unsafe {
        // __rust_alloc_error_handler_should_panic
        let name = OomStrategy::SYMBOL;
        let ll_g = llvm::LLVMRustGetOrInsertGlobal(llmod, name.as_c_char_ptr(), name.len(), i8);
        llvm::set_visibility(ll_g, llvm::Visibility::from_generic(tcx.sess.default_visibility()));
        let val = tcx.sess.opts.unstable_opts.oom.should_panic();
        let llval = llvm::LLVMConstInt(i8, val as u64, False);
        llvm::LLVMSetInitializer(ll_g, llval);

        let name = NO_ALLOC_SHIM_IS_UNSTABLE;
        let ll_g = llvm::LLVMRustGetOrInsertGlobal(llmod, name.as_c_char_ptr(), name.len(), i8);
        llvm::set_visibility(ll_g, llvm::Visibility::from_generic(tcx.sess.default_visibility()));
        let llval = llvm::LLVMConstInt(i8, 0, False);
        llvm::LLVMSetInitializer(ll_g, llval);
    }

    if tcx.sess.opts.debuginfo != DebugInfo::None {
        let dbg_cx = debuginfo::CodegenUnitDebugContext::new(llmod);
        debuginfo::metadata::build_compile_unit_di_node(tcx, module_name, &dbg_cx);
        dbg_cx.finalize(tcx.sess);
    }
}
