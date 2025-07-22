use std::ffi::CString;

use llvm::Linkage::*;
use rustc_abi::{AddressSpace, Align};
use rustc_codegen_ssa::back::write::CodegenContext;
use rustc_codegen_ssa::traits::BaseTypeCodegenMethods;

use crate::builder::gpu_offload::*;
use crate::llvm::{self, Linkage, Type, Visibility};
use crate::{LlvmCodegenBackend, ModuleLlvm, SimpleCx};

pub(crate) fn create_struct_ty<'ll>(
    cx: &'ll SimpleCx<'_>,
    name: &str,
    tys: &[&'ll llvm::Type],
) -> &'ll llvm::Type {
    let entry_struct_name = CString::new(name).unwrap();
    unsafe {
        let entry_struct = llvm::LLVMStructCreateNamed(cx.llcx, entry_struct_name.as_ptr());
        llvm::LLVMStructSetBody(entry_struct, tys.as_ptr(), tys.len() as u32, 0);
        entry_struct
    }
}

pub(crate) fn add_global_decl<'ll>(
    cx: &SimpleCx<'ll>,
    ty: &'ll Type,
    name: &str,
    l: Linkage,
    hidden: bool,
) -> &'ll llvm::Value {
    let c_name = CString::new(name).unwrap();
    let llglobal: &'ll llvm::Value = llvm::add_global(cx.llmod, ty, &c_name);
    llvm::set_global_constant(llglobal, true);
    llvm::set_linkage(llglobal, l);
    if hidden {
        llvm::set_visibility(llglobal, Visibility::Hidden);
    }
    llglobal
}

// We don't copy types from other functions because we generate a new module and context.
// Bringing in types from other contexts would likely cause issues.
pub(crate) fn gen_image_wrapper_module(cgcx: &CodegenContext<LlvmCodegenBackend>) {
    let dl_cstr = CString::new("e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9").unwrap();
    let target_cstr = CString::new("amdgcn-amd-amdhsa").unwrap();
    let name = "offload.wrapper.module";
    let m: crate::ModuleLlvm =
        ModuleLlvm::new_simple(name, dl_cstr.into_raw(), target_cstr.into_raw(), &cgcx).unwrap();
    let cx = SimpleCx::new(m.llmod(), m.llcx, cgcx.pointer_size);
    let tptr = cx.type_ptr();
    let tptr1 = cx.type_ptr_ext(AddressSpace(1));
    let ti64 = cx.type_i64();
    let ti32 = cx.type_i32();
    let ti16 = cx.type_i16();

    let entry_fields = [ti64, ti16, ti16, ti32, tptr, tptr, ti64, ti64, tptr];
    create_struct_ty(&cx, "__tgt_offload_entry", &entry_fields);
    create_struct_ty(&cx, "__tgt_device_image", &[tptr, tptr, tptr, tptr]);
    create_struct_ty(&cx, "__tgt_bin_desc", &[ti32, tptr, tptr, tptr]);

    let offload_entry_ty = add_tgt_offload_entry(&cx);
    let offload_entry_arr = cx.type_array(offload_entry_ty, 0);

    let name = "__start_omp_offloading_entries";
    add_global_decl(&cx, offload_entry_arr, name, ExternalLinkage, true);

    let name = "__stop_omp_offloading_entries";
    add_global_decl(&cx, offload_entry_arr, name, ExternalLinkage, true);

    let name = "__dummy.omp_offloading_entries";
    let llglobal = add_global_decl(&cx, offload_entry_arr, name, InternalLinkage, false);

    let c_section_name = CString::new("omp_offloading_entries").unwrap();
    llvm::set_section(llglobal, &c_section_name);
    let zeroinit = cx.const_null(offload_entry_arr);
    llvm::set_initializer(llglobal, zeroinit);

    CString::new("llvm.compiler.used").unwrap();
    let arr_val = cx.const_array(tptr1, &[llglobal]);
    let c_section_name = CString::new("llvm.metadata").unwrap();
    let llglobal = add_global(&cx, "llvm.compiler.used", arr_val, AppendingLinkage);
    llvm::set_section(llglobal, &c_section_name);
    llvm::set_global_constant(llglobal, false);

    //@llvm.compiler.used = appending global [1 x ptr] [ptr @__dummy.omp_offloading_entries], section "llvm.metadata"

    let mapper_fn_ty = cx.type_func(&[tptr], cx.type_void());
    declare_offload_fn(&cx, &"__tgt_register_lib", mapper_fn_ty);
    declare_offload_fn(&cx, &"__tgt_unregister_lib", mapper_fn_ty);
    declare_offload_fn(&cx, &"atexit", cx.type_func(&[tptr], ti32));

    let unknown_txt = "11111111111111";
    let c_entry_name = CString::new(unknown_txt).unwrap();
    let c_val = c_entry_name.as_bytes_with_nul();
    let initializer = crate::common::bytes_in_context(cx.llcx, c_val);
    let llglobal =
        add_unnamed_global(&cx, &".omp_offloading.device_image", initializer, InternalLinkage);
    let c_section_name = CString::new(".llvm.offloading").unwrap();
    llvm::set_section(llglobal, &c_section_name);
    llvm::set_alignment(llglobal, Align::EIGHT);

    unsafe {
        llvm::LLVMPrintModuleToFile(
            cx.llmod,
            CString::new("rustmagic.openmp.image.wrapper.ll").unwrap().as_ptr(),
            std::ptr::null_mut(),
        );
    }
}
