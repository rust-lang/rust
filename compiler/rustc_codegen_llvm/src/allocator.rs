use libc::c_uint;
use rustc_ast::expand::allocator::{
    ALLOCATOR_METHODS, AllocatorKind, AllocatorTy, NO_ALLOC_SHIM_IS_UNSTABLE,
    alloc_error_handler_name, default_fn_name, global_fn_name,
};
use rustc_middle::bug;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::{DebugInfo, OomStrategy};

use crate::llvm::{self, Context, False, Module, True, Type};
use crate::{ModuleLlvm, attributes, debuginfo};

pub(crate) unsafe fn codegen(
    tcx: TyCtxt<'_>,
    module_llvm: &mut ModuleLlvm,
    module_name: &str,
    kind: AllocatorKind,
    alloc_error_handler_kind: AllocatorKind,
) {
    let llcx = &*module_llvm.llcx;
    let llmod = module_llvm.llmod();
    let usize = unsafe {
        match tcx.sess.target.pointer_width {
            16 => llvm::LLVMInt16TypeInContext(llcx),
            32 => llvm::LLVMInt32TypeInContext(llcx),
            64 => llvm::LLVMInt64TypeInContext(llcx),
            tws => bug!("Unsupported target word size for int: {}", tws),
        }
    };
    let i8 = unsafe { llvm::LLVMInt8TypeInContext(llcx) };
    let i8p = unsafe { llvm::LLVMPointerTypeInContext(llcx, 0) };

    if kind == AllocatorKind::Default {
        for method in ALLOCATOR_METHODS {
            let mut args = Vec::with_capacity(method.inputs.len());
            for input in method.inputs.iter() {
                match input.ty {
                    AllocatorTy::Layout => {
                        args.push(usize); // size
                        args.push(usize); // align
                    }
                    AllocatorTy::Ptr => args.push(i8p),
                    AllocatorTy::Usize => args.push(usize),

                    AllocatorTy::ResultPtr | AllocatorTy::Unit => panic!("invalid allocator arg"),
                }
            }
            let output = match method.output {
                AllocatorTy::ResultPtr => Some(i8p),
                AllocatorTy::Unit => None,

                AllocatorTy::Layout | AllocatorTy::Usize | AllocatorTy::Ptr => {
                    panic!("invalid allocator output")
                }
            };

            let from_name = global_fn_name(method.name);
            let to_name = default_fn_name(method.name);

            create_wrapper_function(tcx, llcx, llmod, &from_name, &to_name, &args, output, false);
        }
    }

    // rust alloc error handler
    create_wrapper_function(
        tcx,
        llcx,
        llmod,
        "__rust_alloc_error_handler",
        alloc_error_handler_name(alloc_error_handler_kind),
        &[usize, usize], // size, align
        None,
        true,
    );

    unsafe {
        // __rust_alloc_error_handler_should_panic
        let name = OomStrategy::SYMBOL;
        let ll_g = llvm::LLVMRustGetOrInsertGlobal(llmod, name.as_ptr().cast(), name.len(), i8);
        llvm::LLVMRustSetVisibility(
            ll_g,
            llvm::Visibility::from_generic(tcx.sess.default_visibility()),
        );
        let val = tcx.sess.opts.unstable_opts.oom.should_panic();
        let llval = llvm::LLVMConstInt(i8, val as u64, False);
        llvm::LLVMSetInitializer(ll_g, llval);

        let name = NO_ALLOC_SHIM_IS_UNSTABLE;
        let ll_g = llvm::LLVMRustGetOrInsertGlobal(llmod, name.as_ptr().cast(), name.len(), i8);
        llvm::LLVMRustSetVisibility(
            ll_g,
            llvm::Visibility::from_generic(tcx.sess.default_visibility()),
        );
        let llval = llvm::LLVMConstInt(i8, 0, False);
        llvm::LLVMSetInitializer(ll_g, llval);
    }

    if tcx.sess.opts.debuginfo != DebugInfo::None {
        let dbg_cx = debuginfo::CodegenUnitDebugContext::new(llmod);
        debuginfo::metadata::build_compile_unit_di_node(tcx, module_name, &dbg_cx);
        dbg_cx.finalize(tcx.sess);
    }
}

fn create_wrapper_function(
    tcx: TyCtxt<'_>,
    llcx: &Context,
    llmod: &Module,
    from_name: &str,
    to_name: &str,
    args: &[&Type],
    output: Option<&Type>,
    no_return: bool,
) {
    unsafe {
        let ty = llvm::LLVMFunctionType(
            output.unwrap_or_else(|| llvm::LLVMVoidTypeInContext(llcx)),
            args.as_ptr(),
            args.len() as c_uint,
            False,
        );
        let llfn = llvm::LLVMRustGetOrInsertFunction(
            llmod,
            from_name.as_ptr().cast(),
            from_name.len(),
            ty,
        );
        let no_return = if no_return {
            // -> ! DIFlagNoReturn
            let no_return = llvm::AttributeKind::NoReturn.create_attr(llcx);
            attributes::apply_to_llfn(llfn, llvm::AttributePlace::Function, &[no_return]);
            Some(no_return)
        } else {
            None
        };

        llvm::LLVMRustSetVisibility(
            llfn,
            llvm::Visibility::from_generic(tcx.sess.default_visibility()),
        );

        if tcx.sess.must_emit_unwind_tables() {
            let uwtable =
                attributes::uwtable_attr(llcx, tcx.sess.opts.unstable_opts.use_sync_unwind);
            attributes::apply_to_llfn(llfn, llvm::AttributePlace::Function, &[uwtable]);
        }

        let callee =
            llvm::LLVMRustGetOrInsertFunction(llmod, to_name.as_ptr().cast(), to_name.len(), ty);
        if let Some(no_return) = no_return {
            // -> ! DIFlagNoReturn
            attributes::apply_to_llfn(callee, llvm::AttributePlace::Function, &[no_return]);
        }
        llvm::LLVMRustSetVisibility(callee, llvm::Visibility::Hidden);

        let llbb = llvm::LLVMAppendBasicBlockInContext(llcx, llfn, c"entry".as_ptr());

        let llbuilder = llvm::LLVMCreateBuilderInContext(llcx);
        llvm::LLVMPositionBuilderAtEnd(llbuilder, llbb);
        let args = args
            .iter()
            .enumerate()
            .map(|(i, _)| llvm::LLVMGetParam(llfn, i as c_uint))
            .collect::<Vec<_>>();
        let ret = llvm::LLVMRustBuildCall(
            llbuilder,
            ty,
            callee,
            args.as_ptr(),
            args.len() as c_uint,
            [].as_ptr(),
            0 as c_uint,
        );
        llvm::LLVMSetTailCall(ret, True);
        if output.is_some() {
            llvm::LLVMBuildRet(llbuilder, ret);
        } else {
            llvm::LLVMBuildRetVoid(llbuilder);
        }
        llvm::LLVMDisposeBuilder(llbuilder);
    }
}
