use crate::attributes;
use libc::c_uint;
use rustc_ast::expand::allocator::{AllocatorKind, AllocatorTy, ALLOCATOR_METHODS};
use rustc_middle::bug;
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::sym;

use crate::llvm::{self, False, True};
use crate::ModuleLlvm;

pub(crate) unsafe fn codegen(
    tcx: TyCtxt<'_>,
    mods: &mut ModuleLlvm,
    kind: AllocatorKind,
    has_alloc_error_handler: bool,
) {
    let llcx = &*mods.llcx;
    let llmod = mods.llmod();
    let usize = match tcx.sess.target.pointer_width {
        16 => llvm::LLVMInt16TypeInContext(llcx),
        32 => llvm::LLVMInt32TypeInContext(llcx),
        64 => llvm::LLVMInt64TypeInContext(llcx),
        tws => bug!("Unsupported target word size for int: {}", tws),
    };
    let i8 = llvm::LLVMInt8TypeInContext(llcx);
    let i8p = llvm::LLVMPointerType(i8, 0);
    let void = llvm::LLVMVoidTypeInContext(llcx);

    for method in ALLOCATOR_METHODS {
        let mut args = Vec::with_capacity(method.inputs.len());
        for ty in method.inputs.iter() {
            match *ty {
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
        let ty = llvm::LLVMFunctionType(
            output.unwrap_or(void),
            args.as_ptr(),
            args.len() as c_uint,
            False,
        );
        let name = format!("__rust_{}", method.name);
        let llfn = llvm::LLVMRustGetOrInsertFunction(llmod, name.as_ptr().cast(), name.len(), ty);

        if tcx.sess.target.default_hidden_visibility {
            llvm::LLVMRustSetVisibility(llfn, llvm::Visibility::Hidden);
        }
        if tcx.sess.must_emit_unwind_tables() {
            attributes::emit_uwtable(llfn, true);
        }

        let callee = kind.fn_name(method.name);
        let callee =
            llvm::LLVMRustGetOrInsertFunction(llmod, callee.as_ptr().cast(), callee.len(), ty);
        llvm::LLVMRustSetVisibility(callee, llvm::Visibility::Hidden);

        let llbb = llvm::LLVMAppendBasicBlockInContext(llcx, llfn, "entry\0".as_ptr().cast());

        let llbuilder = llvm::LLVMCreateBuilderInContext(llcx);
        llvm::LLVMPositionBuilderAtEnd(llbuilder, llbb);
        let args = args
            .iter()
            .enumerate()
            .map(|(i, _)| llvm::LLVMGetParam(llfn, i as c_uint))
            .collect::<Vec<_>>();
        let ret =
            llvm::LLVMRustBuildCall(llbuilder, callee, args.as_ptr(), args.len() as c_uint, None);
        llvm::LLVMSetTailCall(ret, True);
        if output.is_some() {
            llvm::LLVMBuildRet(llbuilder, ret);
        } else {
            llvm::LLVMBuildRetVoid(llbuilder);
        }
        llvm::LLVMDisposeBuilder(llbuilder);
    }

    // rust alloc error handler
    let args = [usize, usize]; // size, align

    let ty = llvm::LLVMFunctionType(void, args.as_ptr(), args.len() as c_uint, False);
    let name = "__rust_alloc_error_handler".to_string();
    let llfn = llvm::LLVMRustGetOrInsertFunction(llmod, name.as_ptr().cast(), name.len(), ty);
    // -> ! DIFlagNoReturn
    llvm::Attribute::NoReturn.apply_llfn(llvm::AttributePlace::Function, llfn);

    if tcx.sess.target.default_hidden_visibility {
        llvm::LLVMRustSetVisibility(llfn, llvm::Visibility::Hidden);
    }
    if tcx.sess.must_emit_unwind_tables() {
        attributes::emit_uwtable(llfn, true);
    }

    let kind = if has_alloc_error_handler { AllocatorKind::Global } else { AllocatorKind::Default };
    let callee = kind.fn_name(sym::oom);
    let callee = llvm::LLVMRustGetOrInsertFunction(llmod, callee.as_ptr().cast(), callee.len(), ty);
    // -> ! DIFlagNoReturn
    llvm::Attribute::NoReturn.apply_llfn(llvm::AttributePlace::Function, callee);
    llvm::LLVMRustSetVisibility(callee, llvm::Visibility::Hidden);

    let llbb = llvm::LLVMAppendBasicBlockInContext(llcx, llfn, "entry\0".as_ptr().cast());

    let llbuilder = llvm::LLVMCreateBuilderInContext(llcx);
    llvm::LLVMPositionBuilderAtEnd(llbuilder, llbb);
    let args = args
        .iter()
        .enumerate()
        .map(|(i, _)| llvm::LLVMGetParam(llfn, i as c_uint))
        .collect::<Vec<_>>();
    let ret = llvm::LLVMRustBuildCall(llbuilder, callee, args.as_ptr(), args.len() as c_uint, None);
    llvm::LLVMSetTailCall(ret, True);
    llvm::LLVMBuildRetVoid(llbuilder);
    llvm::LLVMDisposeBuilder(llbuilder);
}
