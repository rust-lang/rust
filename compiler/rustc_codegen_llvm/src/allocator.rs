use rustc_ast::expand::allocator::NO_ALLOC_SHIM_IS_UNSTABLE;
use rustc_codegen_ssa::traits::BaseTypeCodegenMethods as _;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::{DebugInfo, OomStrategy};
use rustc_symbol_mangling::mangle_internal_symbol;

use crate::llvm::{self, False};
use crate::{SimpleCx, debuginfo};

pub(crate) unsafe fn codegen(tcx: TyCtxt<'_>, cx: SimpleCx<'_>, module_name: &str) {
    let i8 = cx.type_i8();

    unsafe {
        // __rust_alloc_error_handler_should_panic
        let name = mangle_internal_symbol(tcx, OomStrategy::SYMBOL);
        let ll_g = cx.declare_global(&name, i8);
        llvm::set_visibility(ll_g, llvm::Visibility::from_generic(tcx.sess.default_visibility()));
        let val = tcx.sess.opts.unstable_opts.oom.should_panic();
        let llval = llvm::LLVMConstInt(i8, val as u64, False);
        llvm::set_initializer(ll_g, llval);

        let name = mangle_internal_symbol(tcx, NO_ALLOC_SHIM_IS_UNSTABLE);
        let ll_g = cx.declare_global(&name, i8);
        llvm::set_visibility(ll_g, llvm::Visibility::from_generic(tcx.sess.default_visibility()));
        let llval = llvm::LLVMConstInt(i8, 0, False);
        llvm::set_initializer(ll_g, llval);
    }

    if tcx.sess.opts.debuginfo != DebugInfo::None {
        let dbg_cx = debuginfo::CodegenUnitDebugContext::new(cx.llmod);
        debuginfo::metadata::build_compile_unit_di_node(tcx, module_name, &dbg_cx);
        dbg_cx.finalize(tcx.sess);
    }
}
