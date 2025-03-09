use rustc_middle::ty::TyCtxt;
use rustc_session::config::DebugInfo;

use crate::common::AsCCharPtr;
use crate::llvm::{self, Context, False, Module};
use crate::{ModuleLlvm, attributes, debuginfo};

const FEATURE_GATE_SYMBOL: &str = "__rust_personality_stub_is_unstable";

pub(crate) unsafe fn codegen(tcx: TyCtxt<'_>, module_llvm: &mut ModuleLlvm, module_name: &str) {
    let llcx = &*module_llvm.llcx;
    let llmod = module_llvm.llmod();

    // rust alloc error handler
    create_stub_personality(tcx, llcx, llmod);

    if tcx.sess.opts.debuginfo != DebugInfo::None {
        let dbg_cx = debuginfo::CodegenUnitDebugContext::new(llmod);
        debuginfo::metadata::build_compile_unit_di_node(tcx, module_name, &dbg_cx);
        dbg_cx.finalize(tcx.sess);
    }
}

fn create_stub_personality(tcx: TyCtxt<'_>, llcx: &Context, llmod: &Module) {
    unsafe {
        let llfn = declare_stub_personality(tcx, llcx, llmod);

        let i8_ty = llvm::LLVMInt8TypeInContext(llcx);

        let feature_gate = llvm::LLVMRustGetOrInsertGlobal(
            llmod,
            FEATURE_GATE_SYMBOL.as_c_char_ptr(),
            FEATURE_GATE_SYMBOL.len(),
            i8_ty,
        );

        let (trap_fn, trap_ty) = trap_intrinsic(llcx, llmod);

        let llbb = llvm::LLVMAppendBasicBlockInContext(llcx, llfn, c"entry".as_ptr());
        let llbuilder = llvm::LLVMCreateBuilderInContext(llcx);
        llvm::LLVMPositionBuilderAtEnd(llbuilder, llbb);

        let load = llvm::LLVMBuildLoad2(llbuilder, i8_ty, feature_gate, c"".as_ptr());
        llvm::LLVMSetVolatile(load, llvm::True);

        llvm::LLVMBuildCallWithOperandBundles(
            llbuilder,
            trap_ty,
            trap_fn,
            [].as_ptr(),
            0,
            [].as_ptr(),
            0,
            c"".as_ptr(),
        );

        llvm::LLVMBuildUnreachable(llbuilder);

        llvm::LLVMDisposeBuilder(llbuilder);
    }
}

fn declare_stub_personality<'ll>(
    tcx: TyCtxt<'_>,
    llcx: &'ll Context,
    llmod: &'ll Module,
) -> &'ll llvm::Value {
    let name = "rust_eh_personality";
    unsafe {
        let no_return = llvm::AttributeKind::NoReturn.create_attr(llcx);

        let ty = llvm::LLVMFunctionType(llvm::LLVMVoidTypeInContext(llcx), [].as_ptr(), 0, False);
        let llfn = llvm::LLVMRustGetOrInsertFunction(llmod, name.as_c_char_ptr(), name.len(), ty);
        attributes::apply_to_llfn(llfn, llvm::AttributePlace::Function, &[no_return]);
        llvm::set_visibility(llfn, llvm::Visibility::from_generic(tcx.sess.default_visibility()));

        if tcx.sess.must_emit_unwind_tables() {
            let uwtable =
                attributes::uwtable_attr(llcx, tcx.sess.opts.unstable_opts.use_sync_unwind);
            attributes::apply_to_llfn(llfn, llvm::AttributePlace::Function, &[uwtable]);
        }

        llfn
    }
}

fn trap_intrinsic<'ll>(
    llcx: &'ll Context,
    llmod: &'ll Module,
) -> (&'ll llvm::Value, &'ll llvm::Type) {
    let name = "llvm.trap";

    unsafe {
        let no_return = llvm::AttributeKind::NoReturn.create_attr(llcx);
        let llty = llvm::LLVMFunctionType(llvm::LLVMVoidTypeInContext(llcx), [].as_ptr(), 0, False);
        let llfn = llvm::LLVMRustGetOrInsertFunction(llmod, name.as_c_char_ptr(), name.len(), llty);
        llvm::SetFunctionCallConv(llfn, llvm::CCallConv);
        attributes::apply_to_llfn(llfn, llvm::AttributePlace::Function, &[no_return]);

        (llfn, llty)
    }
}
