use std::ffi::{CString, c_uint};

use llvm::Linkage::*;
use rustc_codegen_ssa::back::write::CodegenContext;

use crate::llvm::{self, Linkage};
use crate::{LlvmCodegenBackend, SimpleCx};

fn add_unnamed_global_in_addrspace<'ll>(
    cx: &SimpleCx<'ll>,
    name: &str,
    initializer: &'ll llvm::Value,
    l: Linkage,
    addrspace: u32,
) -> &'ll llvm::Value {
    let llglobal = add_global_in_addrspace(cx, name, initializer, l, addrspace);
    unsafe { llvm::LLVMSetUnnamedAddress(llglobal, llvm::UnnamedAddr::Global) };
    llglobal
}

pub(crate) fn add_global_in_addrspace<'ll>(
    cx: &SimpleCx<'ll>,
    name: &str,
    initializer: &'ll llvm::Value,
    l: Linkage,
    addrspace: u32,
) -> &'ll llvm::Value {
    let c_name = CString::new(name).unwrap();
    let llglobal: &'ll llvm::Value = llvm::add_global_in_addrspace(
        cx.llmod,
        cx.val_ty(initializer),
        &c_name,
        addrspace as c_uint,
    );
    llvm::set_global_constant(llglobal, true);
    llvm::set_linkage(llglobal, l);
    llvm::set_initializer(llglobal, initializer);
    llglobal
}

#[allow(unused)]
pub(crate) fn gen_asdf<'ll>(cgcx: &CodegenContext<LlvmCodegenBackend>, _old_cx: &SimpleCx<'ll>) {
    let llcx = unsafe { llvm::LLVMRustContextCreate(false) };
    let module_name = CString::new("offload.wrapper.module").unwrap();
    let llmod = unsafe { llvm::LLVMModuleCreateWithNameInContext(module_name.as_ptr(), llcx) };
    let cx = SimpleCx::new(llmod, llcx, cgcx.pointer_size);
    let initializer = cx.get_const_i32(0);
    add_unnamed_global_in_addrspace(&cx, "__omp_rtl_debug_kind", initializer, WeakODRLinkage, 1);
    add_unnamed_global_in_addrspace(
        &cx,
        "__omp_rtl_assume_teams_oversubscription",
        initializer,
        WeakODRLinkage,
        1,
    );
    add_unnamed_global_in_addrspace(
        &cx,
        "__omp_rtl_assume_threads_oversubscription",
        initializer,
        WeakODRLinkage,
        1,
    );
    add_unnamed_global_in_addrspace(
        &cx,
        "__omp_rtl_assume_no_thread_state",
        initializer,
        WeakODRLinkage,
        1,
    );
    add_unnamed_global_in_addrspace(
        &cx,
        "__oclc_ABI_version",
        cx.get_const_i32(500),
        WeakODRLinkage,
        4,
    );
    unsafe {
        llvm::LLVMPrintModuleToFile(
            llmod,
            CString::new("rustmagic-openmp-amdgcn-amd-amdhsa-gfx90a.ll").unwrap().as_ptr(),
            std::ptr::null_mut(),
        );

        // Clean up
        llvm::LLVMDisposeModule(llmod);
        llvm::LLVMContextDispose(llcx);
    }
    // TODO: addressspace 1 or 4
}
// source_filename = "mem.cpp"
// GPU: target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
// CPU: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// target triple = "amdgcn-amd-amdhsa"
//
// @__omp_rtl_debug_kind = weak_odr hidden local_unnamed_addr addrspace(1) constant i32 0
// @__omp_rtl_assume_teams_oversubscription = weak_odr hidden local_unnamed_addr addrspace(1) constant i32 0
// @__omp_rtl_assume_threads_oversubscription = weak_odr hidden local_unnamed_addr addrspace(1) constant i32 0
// @__omp_rtl_assume_no_thread_state = weak_odr hidden local_unnamed_addr addrspace(1) constant i32 0
// @__omp_rtl_assume_no_nested_parallelism = weak_odr hidden local_unnamed_addr addrspace(1) constant i32 0
// @__oclc_ABI_version = weak_odr hidden local_unnamed_addr addrspace(4) constant i32 500
//
// !llvm.module.flags = !{!0, !1, !2, !3, !4}
// !opencl.ocl.version = !{!5}
// !llvm.ident = !{!6, !7}
//
// !0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
// !1 = !{i32 1, !"wchar_size", i32 4}
// !2 = !{i32 7, !"openmp", i32 51}
// !3 = !{i32 7, !"openmp-device", i32 51}
// !4 = !{i32 8, !"PIC Level", i32 2}
// !5 = !{i32 2, i32 0}
// !6 = !{!"clang version 20.1.5-rust-1.89.0-nightly (https://github.com/rust-lang/llvm-project.git c1118fdbb3024157df7f4cfe765f2b0b4339e8a2)"}
// !7 = !{!"AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.0 25133 c7fe45cf4b819c5991fe208aaa96edf142730f1d)"}
