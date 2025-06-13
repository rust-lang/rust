use std::ffi::{CStr, CString};

use crate::builder::{SBuilder, UNNAMED};
use crate::common::AsCCharPtr;
use crate::llvm::AttributePlace::Function;
use crate::llvm::{self, Linkage, Visibility, build_string};
use crate::{LlvmCodegenBackend, ModuleLlvm, SimpleCx, attributes};
use rustc_codegen_ssa::back::write::{CodegenContext, FatLtoInput};

use llvm::Linkage::*;
use rustc_abi::Align;
use rustc_codegen_ssa::traits::BaseTypeCodegenMethods;

fn create_struct_ty<'ll>(
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

// We don't copy types from other functions because we generate a new module and context.
// Bringing in types from other contexts would likely cause issues.
pub(crate) fn gen_image_wrapper_module<'ll>(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    old_cx: &SimpleCx<'ll>,
) {
    unsafe {
        let llcx = llvm::LLVMRustContextCreate(false);
        let module_name = CString::new("offload.wrapper.module").unwrap();
        let llmod = llvm::LLVMModuleCreateWithNameInContext(module_name.as_ptr(), llcx);
        let cx = SimpleCx::new(llmod, llcx, cgcx.pointer_size);
        let tptr = cx.type_ptr();
        let ti64 = cx.type_i64();
        let ti32 = cx.type_i32();
        let ti16 = cx.type_i16();
        let ti8 = cx.type_i8();
        let dl_cstr = llvm::LLVMGetDataLayoutStr(old_cx.llmod);
        llvm::LLVMSetDataLayout(llmod, dl_cstr);
        let target_cstr = llvm::LLVMGetTarget(old_cx.llmod);
        let target = CStr::from_ptr(target_cstr).to_string_lossy().into_owned();
        llvm::LLVMSetTarget(llmod, target_cstr);

        let mut entry_fields = [ti64, ti16, ti16, ti32, tptr, tptr, ti64, ti64, tptr];
        let tgt_entry = create_struct_ty(&cx, "__tgt_offload_entry", &entry_fields);
        let tgt_image = create_struct_ty(&cx, "__tgt_device_image", &[tptr, tptr, tptr, tptr]);
        let tgt_desc = create_struct_ty(&cx, "__tgt_bin_desc", &[ti32, tptr, tptr, tptr]);

        let offload_entry_ty = add_tgt_offload_entry(&cx);
        let offload_entry_arr = cx.type_array(offload_entry_ty, 0);

        let c_name = CString::new("__start_omp_offloading_entries").unwrap();
        let llglobal = llvm::add_global(cx.llmod, offload_entry_arr, &c_name);
        llvm::set_global_constant(llglobal, true);
        llvm::set_linkage(llglobal, ExternalLinkage);
        llvm::set_visibility(llglobal, Visibility::Hidden);
        let c_name = CString::new("__stop_omp_offloading_entries").unwrap();
        let llglobal = llvm::add_global(cx.llmod, offload_entry_arr, &c_name);
        llvm::set_global_constant(llglobal, true);
        llvm::set_linkage(llglobal, ExternalLinkage);
        llvm::set_visibility(llglobal, Visibility::Hidden);

        let c_name = CString::new("__dummy.omp_offloading_entries").unwrap();
        let llglobal = llvm::add_global(cx.llmod, offload_entry_arr, &c_name);
        llvm::set_global_constant(llglobal, true);
        llvm::set_linkage(llglobal, InternalLinkage);
        let c_section_name = CString::new("omp_offloading_entries").unwrap();
        llvm::set_section(llglobal, &c_section_name);
        let zeroinit = cx.const_null(offload_entry_arr);
        llvm::set_initializer(llglobal, zeroinit);

        let c_name = CString::new("llvm.compiler.used").unwrap();
        let arr_val = cx.const_array(tptr, &[llglobal]);
        let c_section_name = CString::new("llvm.metadata").unwrap();
        let llglobal = add_global(&cx, "llvm.compiler.used", arr_val, AppendingLinkage);
        llvm::set_section(llglobal, &c_section_name);
        llvm::set_global_constant(llglobal, false);

        //@llvm.compiler.used = appending global [1 x ptr] [ptr @__dummy.omp_offloading_entries], section "llvm.metadata"

        let mapper_fn_ty = cx.type_func(&[tptr], cx.type_void());
        let foo = crate::declare::declare_simple_fn(
            &cx,
            &"__tgt_unregister_lib",
            llvm::CallConv::CCallConv,
            llvm::UnnamedAddr::No,
            llvm::Visibility::Default,
            mapper_fn_ty,
        );
        let bar = crate::declare::declare_simple_fn(
            &cx,
            &"__tgt_register_lib",
            llvm::CallConv::CCallConv,
            llvm::UnnamedAddr::No,
            llvm::Visibility::Default,
            mapper_fn_ty,
        );

        // @__start_omp_offloading_entries = external hidden constant [0 x %struct.__tgt_offload_entry]
        // @__stop_omp_offloading_entries = external hidden constant [0 x %struct.__tgt_offload_entry]
        // @__dummy.omp_offloading_entries = internal constant [0 x %struct.__tgt_offload_entry] zeroinitializer, section "omp_offloading_entries"

        llvm::LLVMPrintModuleToFile(
            llmod,
            CString::new("rustmagic.openmp.image.wrapper.ll").unwrap().as_ptr(),
            std::ptr::null_mut(),
        );

        // Clean up
        llvm::LLVMDisposeModule(llmod);
        llvm::LLVMContextDispose(llcx);
    }
}

pub(crate) fn handle_gpu_code<'ll>(
    cgcx: &CodegenContext<LlvmCodegenBackend>,
    cx: &'ll SimpleCx<'_>,
) {
    if cx.get_function("gen_tgt_offload").is_some() {
        let (offload_entry_ty, at_one, begin, update, end, fn_ty) = gen_globals(&cx);

        dbg!("created struct");
        let mut o_types = vec![];
        let mut kernels = vec![];
        for num in 0..9 {
            let kernel = cx.get_function(&format!("kernel_{num}"));
            if let Some(kernel) = kernel {
                o_types.push(gen_define_handling(&cx, kernel, offload_entry_ty, num));
                kernels.push(kernel);
            }
        }
        dbg!("gen_call_handling");
        gen_call_handling(&cx, &kernels, at_one, begin, update, end, fn_ty, &o_types);
        gen_image_wrapper_module(&cgcx, &cx);
    } else {
        dbg!("no marker found");
    }
}

fn add_tgt_offload_entry<'ll>(cx: &'ll SimpleCx<'_>) -> &'ll llvm::Type {
    let offload_entry_ty = cx.type_named_struct("struct.__tgt_offload_entry");
    let tptr = cx.type_ptr();
    let ti64 = cx.type_i64();
    let ti32 = cx.type_i32();
    let ti16 = cx.type_i16();
    let ti8 = cx.type_i8();
    let tarr = cx.type_array(ti32, 3);
    let entry_elements = vec![ti64, ti16, ti16, ti32, tptr, tptr, ti64, ti64, tptr];
    cx.set_struct_body(offload_entry_ty, &entry_elements, false);
    offload_entry_ty
}

fn gen_globals<'ll>(
    cx: &'ll SimpleCx<'_>,
) -> (
    &'ll llvm::Type,
    &'ll llvm::Value,
    &'ll llvm::Value,
    &'ll llvm::Value,
    &'ll llvm::Value,
    &'ll llvm::Type,
) {
    let offload_entry_ty = add_tgt_offload_entry(&cx);
    let kernel_arguments_ty = cx.type_named_struct("struct.__tgt_kernel_arguments");
    let tptr = cx.type_ptr();
    let ti64 = cx.type_i64();
    let ti32 = cx.type_i32();
    let ti16 = cx.type_i16();
    let ti8 = cx.type_i8();
    let tarr = cx.type_array(ti32, 3);

    // @0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
    let unknown_txt = ";unknown;unknown;0;0;;";
    let c_entry_name = CString::new(unknown_txt).unwrap();
    let c_val = c_entry_name.as_bytes_with_nul();
    let initializer = crate::common::bytes_in_context(cx.llcx, c_val);
    let at_zero = add_unnamed_global(&cx, &"", initializer, PrivateLinkage);
    llvm::set_alignment(at_zero, Align::ONE);

    // @1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8
    let struct_ident_ty = cx.type_named_struct("struct.ident_t");
    let struct_elems: Vec<&llvm::Value> = vec![
        cx.get_const_i32(0),
        cx.get_const_i32(2),
        cx.get_const_i32(0),
        cx.get_const_i32(22),
        at_zero,
    ];
    let struct_elems_ty: Vec<_> = struct_elems.iter().map(|&x| cx.val_ty(x)).collect();
    let initializer = crate::common::named_struct(struct_ident_ty, &struct_elems);
    cx.set_struct_body(struct_ident_ty, &struct_elems_ty, false);
    let at_one = add_unnamed_global(&cx, &"", initializer, PrivateLinkage);
    llvm::set_alignment(at_one, Align::EIGHT);

    // coppied from LLVM
    // typedef struct {
    //   uint64_t Reserved;
    //   uint16_t Version;
    //   uint16_t Kind;
    //   uint32_t Flags;
    //   void *Address;
    //   char *SymbolName;
    //   uint64_t Size;
    //   uint64_t Data;
    //   void *AuxAddr;
    // } __tgt_offload_entry;
    let kernel_elements =
        vec![ti32, ti32, tptr, tptr, tptr, tptr, tptr, tptr, ti64, ti64, tarr, tarr, ti32];

    cx.set_struct_body(kernel_arguments_ty, &kernel_elements, false);
    let global = cx.declare_global("my_struct_global2", kernel_arguments_ty);
    //@my_struct_global = external global %struct.__tgt_offload_entry
    //@my_struct_global2 = external global %struct.__tgt_kernel_arguments
    dbg!(&kernel_arguments_ty);
    //LLVMTypeRef elements[9] = {i64Ty, i16Ty, i16Ty, i32Ty, ptrTy, ptrTy, i64Ty, i64Ty, ptrTy};
    //LLVMStructSetBody(structTy, elements, 9, 0);

    // New, to test memtransfer
    // ; Function Attrs: nounwind
    // declare void @__tgt_target_data_begin_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr) #3
    //
    // ; Function Attrs: nounwind
    // declare void @__tgt_target_data_update_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr) #3
    //
    // ; Function Attrs: nounwind
    // declare void @__tgt_target_data_end_mapper(ptr, i64, i32, ptr, ptr, ptr, ptr, ptr, ptr) #3

    let mapper_begin = "__tgt_target_data_begin_mapper";
    let mapper_update = String::from("__tgt_target_data_update_mapper");
    let mapper_end = String::from("__tgt_target_data_end_mapper");
    let args = vec![tptr, ti64, ti32, tptr, tptr, tptr, tptr, tptr, tptr];
    let mapper_fn_ty = cx.type_func(&args, cx.type_void());
    let foo = crate::declare::declare_simple_fn(
        &cx,
        &mapper_begin,
        llvm::CallConv::CCallConv,
        llvm::UnnamedAddr::No,
        llvm::Visibility::Default,
        mapper_fn_ty,
    );
    let bar = crate::declare::declare_simple_fn(
        &cx,
        &mapper_update,
        llvm::CallConv::CCallConv,
        llvm::UnnamedAddr::No,
        llvm::Visibility::Default,
        mapper_fn_ty,
    );
    let baz = crate::declare::declare_simple_fn(
        &cx,
        &mapper_end,
        llvm::CallConv::CCallConv,
        llvm::UnnamedAddr::No,
        llvm::Visibility::Default,
        mapper_fn_ty,
    );
    let nounwind = llvm::AttributeKind::NoUnwind.create_attr(cx.llcx);
    attributes::apply_to_llfn(foo, Function, &[nounwind]);
    attributes::apply_to_llfn(bar, Function, &[nounwind]);
    attributes::apply_to_llfn(baz, Function, &[nounwind]);

    (offload_entry_ty, at_one, foo, bar, baz, mapper_fn_ty)
}

fn add_priv_unnamed_arr<'ll>(cx: &SimpleCx<'ll>, name: &str, vals: &[u64]) -> &'ll llvm::Value {
    let ti64 = cx.type_i64();
    let size_ty = cx.type_array(ti64, vals.len() as u64);
    let mut size_val = Vec::with_capacity(vals.len());
    for &val in vals {
        size_val.push(cx.get_const_i64(val));
    }
    let initializer = cx.const_array(ti64, &size_val);
    add_unnamed_global(cx, name, initializer, PrivateLinkage)
}

fn add_unnamed_global<'ll>(
    cx: &SimpleCx<'ll>,
    name: &str,
    initializer: &'ll llvm::Value,
    l: Linkage,
) -> &'ll llvm::Value {
    let llglobal = add_global(cx, name, initializer, l);
    unsafe { llvm::LLVMSetUnnamedAddress(llglobal, llvm::UnnamedAddr::Global) };
    llglobal
}

fn add_global<'ll>(
    cx: &SimpleCx<'ll>,
    name: &str,
    initializer: &'ll llvm::Value,
    l: Linkage,
) -> &'ll llvm::Value {
    let c_name = CString::new(name).unwrap();
    let llglobal: &'ll llvm::Value = llvm::add_global(cx.llmod, cx.val_ty(initializer), &c_name);
    llvm::set_global_constant(llglobal, true);
    llvm::set_linkage(llglobal, l);
    llvm::set_initializer(llglobal, initializer);
    llglobal
}

fn gen_define_handling<'ll>(
    cx: &'ll SimpleCx<'_>,
    kernel: &'ll llvm::Value,
    offload_entry_ty: &'ll llvm::Type,
    num: i64,
) -> &'ll llvm::Value {
    let types = cx.func_params_types(cx.get_type_of_global(kernel));
    // It seems like non-pointer values are automatically mapped. So here, we focus on pointer (or
    // reference) types.
    let num_ptr_types = types
        .iter()
        .map(|&x| matches!(cx.type_kind(x), rustc_codegen_ssa::common::TypeKind::Pointer))
        .count();

    // We do not know their size anymore at this level, so hardcode a placeholder.
    // A follow-up pr will track these from the frontend, where we still have Rust types.
    // Then, we will be able to figure out that e.g. `&[f32;1024]` will result in 32*1024 bytes.
    // I decided that 1024 bytes is a great placeholder value for now.
    let o_sizes =
        add_priv_unnamed_arr(&cx, &format!(".offload_sizes.{num}"), &vec![1024; num_ptr_types]);
    // Here we figure out whether something needs to be copied to the gpu (=1), from the gpu (=2),
    // or both to and from the gpu (=3). Other values shouldn't affect us for now.
    // A non-mutable reference or pointer will be 1, an array that's not read, but fully overwritten
    // will be 2. For now, everything is 3, untill we have our frontend set up.
    let o_types =
        add_priv_unnamed_arr(&cx, &format!(".offload_maptypes.{num}"), &vec![3; num_ptr_types]);
    // Next: For each function, generate these three entries. A weak constant,
    // the llvm.rodata entry name, and  the omp_offloading_entries value

    // @.__omp_offloading_86fafab6_c40006a1__Z3fooPSt7complexIdES1_S0_m_l7.region_id = weak constant i8 0
    // @.offloading.entry_name = internal unnamed_addr constant [66 x i8] c"__omp_offloading_86fafab6_c40006a1__Z3fooPSt7complexIdES1_S0_m_l7\00", section ".llvm.rodata.offloading", align 1
    let name = format!(".kernel_{num}.region_id");
    let initializer = cx.get_const_i8(0);
    let region_id = add_unnamed_global(&cx, &name, initializer, WeakAnyLinkage);

    let c_entry_name = CString::new(format!("kernel_{num}")).unwrap();
    let c_val = c_entry_name.as_bytes_with_nul();
    let foo = format!(".offloading.entry_name.{num}");

    let initializer = crate::common::bytes_in_context(cx.llcx, c_val);
    let llglobal = add_unnamed_global(&cx, &foo, initializer, InternalLinkage);
    llvm::set_alignment(llglobal, Align::ONE);
    let c_section_name = CString::new(".llvm.rodata.offloading").unwrap();
    llvm::set_section(llglobal, &c_section_name);

    // Not actively used yet, for calling real kernels
    let name = format!(".offloading.entry.kernel_{num}");
    let ci64_0 = cx.get_const_i64(0);
    let ci16_1 = cx.get_const_i16(1);
    let elems: Vec<&llvm::Value> = vec![
        ci64_0,
        ci16_1,
        ci16_1,
        cx.get_const_i32(0),
        region_id,
        llglobal,
        ci64_0,
        ci64_0,
        cx.const_null(cx.type_ptr()),
    ];

    let initializer = crate::common::named_struct(offload_entry_ty, &elems);
    let c_name = CString::new(name).unwrap();
    let llglobal = llvm::add_global(cx.llmod, offload_entry_ty, &c_name);
    llvm::set_global_constant(llglobal, true);
    llvm::set_linkage(llglobal, WeakAnyLinkage);
    llvm::set_initializer(llglobal, initializer);
    llvm::set_alignment(llglobal, Align::ONE);
    let c_section_name = CString::new(".omp_offloading_entries").unwrap();
    llvm::set_section(llglobal, &c_section_name);
    // rustc
    // @.offloading.entry.kernel_3 = weak constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 1, i32 0, ptr @.kernel_3.region_id, ptr @.offloading.entry_name.3, i64 0, i64 0, ptr null }, section ".omp_offloading_entries", align 1
    // clang
    // @.offloading.entry.__omp_offloading_86fafab6_c40006a1__Z3fooPSt7complexIdES1_S0_m_l7 = weak constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 1, i32 0, ptr @.__omp_offloading_86fafab6_c40006a1__Z3fooPSt7complexIdES1_S0_m_l7.region_id, ptr @.offloading.entry_name, i64 0, i64 0, ptr null }, section "omp_offloading_entries", align 1

    //
    // 1. @.offload_sizes.{num} = private unnamed_addr constant [4 x i64] [i64 8, i64 0, i64 16, i64 0]
    // 2. @.offload_maptypes
    // 3. @.__omp_offloading_<hash>_fnc_name_<hash> = weak constant i8 0
    // 4. @.offloading.entry_name = internal unnamed_addr constant [66 x i8] c"__omp_offloading_86fafab6_c40006a1__Z3fooPSt7complexIdES1_S0_m_l7\00", section ".llvm.rodata.offloading", align 1
    // 5. @.offloading.entry.__omp_offloading_86fafab6_c40006a1__Z3fooPSt7complexIdES1_S0_m_l7 = weak constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 1, i32 0, ptr @.__omp_offloading_86fafab6_c40006a1__Z3fooPSt7complexIdES1_S0_m_l7.region_id, ptr @.offloading.entry_name, i64 0, i64 0, ptr null }, section "omp_offloading_entries", align 1
    o_types
}

// For each kernel *call*, we now use some of our previous declared globals to move data to and from
// the gpu. We don't have a proper frontend yet, so we assume that every call to a kernel function
// from main is intended to run on the GPU. For now, we only handle the data transfer part of it.
// If two consecutive kernels use the same memory, we still move it to the host and back to the gpu.
// Since in our frontend users (by default) don't have to specify data transfer, this is something
// we should optimize in the future! We also assume that everything should be copied back and forth,
// but sometimes we can directly zero-allocate on the device and only move back, or if something is
// immutable, we might only copy it to the device, but not back.
//
// Current steps:
// 0. Alloca some variables for the following steps
// 1. set insert point before kernel call.
// 2. generate all the GEPS and stores, to be used in 3)
// 3. generate __tgt_target_data_begin calls to move data to the GPU
//
// unchanged: keep kernel call. Later move the kernel to the GPU
//
// 4. set insert point after kernel call.
// 5. generate all the GEPS and stores, to be used in 6)
// 6. generate __tgt_target_data_end calls to move data from the GPU
fn gen_call_handling<'ll>(
    cx: &'ll SimpleCx<'_>,
    kernels: &[&'ll llvm::Value],
    s_ident_t: &'ll llvm::Value,
    begin: &'ll llvm::Value,
    update: &'ll llvm::Value,
    end: &'ll llvm::Value,
    fn_ty: &'ll llvm::Type,
    o_types: &[&'ll llvm::Value],
) {
    let main_fn = cx.get_function("main");
    if let Some(main_fn) = main_fn {
        let kernel_name = "kernel_1";
        let call = unsafe {
            llvm::LLVMRustGetFunctionCall(main_fn, kernel_name.as_c_char_ptr(), kernel_name.len())
        };
        let kernel_call = if call.is_some() {
            dbg!("found kernel call");
            call.unwrap()
        } else {
            return;
        };
        let kernel_call_bb = unsafe { llvm::LLVMGetInstructionParent(kernel_call) };
        let called = unsafe { llvm::LLVMGetCalledValue(kernel_call).unwrap() };
        let mut builder = SBuilder::build(cx, kernel_call_bb);

        let types = cx.func_params_types(cx.get_type_of_global(called));
        dbg!(&types);
        let num_args = types.len() as u64;
        let mut names: Vec<&llvm::Value> = Vec::with_capacity(num_args as usize);

        // Step 0)
        unsafe { llvm::LLVMRustPositionBuilderPastAllocas(builder.llbuilder, main_fn) };
        let ty = cx.type_array(cx.type_ptr(), num_args);
        // Baseptr are just the input pointer to the kernel, stored in a local alloca
        let a1 = builder.my_alloca2(ty, Align::EIGHT, ".offload_baseptrs");
        // Ptrs are the result of a gep into the baseptr, at least for our trivial types.
        let a2 = builder.my_alloca2(ty, Align::EIGHT, ".offload_ptrs");
        // These represent the sizes in bytes, e.g. the entry for `&[f64; 16]` will be 8*16.
        let ty2 = cx.type_array(cx.type_i64(), num_args);
        let a4 = builder.my_alloca2(ty2, Align::EIGHT, ".offload_sizes");
        // Now we allocate once per function param, a copy to be passed to one of our maps.
        let mut vals = vec![];
        let mut geps = vec![];
        let i32_0 = cx.get_const_i32(0);
        for (index, in_ty) in types.iter().enumerate() {
            // get function arg, store it into the alloca, and read it.
            let p = llvm::get_param(called, index as u32);
            let name = llvm::get_value_name(p);
            let name = str::from_utf8(name).unwrap();
            let arg_name = CString::new(format!("{name}.addr")).unwrap();
            let alloca =
                unsafe { llvm::LLVMBuildAlloca(builder.llbuilder, in_ty, arg_name.as_ptr()) };
            builder.store(p, alloca, Align::EIGHT);
            let val = builder.load(in_ty, alloca, Align::EIGHT);
            let gep = builder.inbounds_gep(cx.type_f32(), val, &[i32_0]);
            vals.push(val);
            geps.push(gep);
        }

        // Step 1)
        unsafe { llvm::LLVMRustPositionBefore(builder.llbuilder, kernel_call) };
        for i in 0..num_args {
            let idx = cx.get_const_i32(i);
            let gep1 = builder.inbounds_gep(ty, a1, &[i32_0, idx]);
            builder.store(vals[i as usize], gep1, Align::EIGHT);
            let gep2 = builder.inbounds_gep(ty, a2, &[i32_0, idx]);
            builder.store(geps[i as usize], gep2, Align::EIGHT);
            let gep3 = builder.inbounds_gep(ty2, a4, &[i32_0, idx]);
            builder.store(cx.get_const_i64(1024), gep3, Align::EIGHT);
        }

        // Step 2)
        let gep1 = builder.inbounds_gep(ty, a1, &[i32_0, i32_0]);
        let gep2 = builder.inbounds_gep(ty, a2, &[i32_0, i32_0]);
        let gep3 = builder.inbounds_gep(ty2, a4, &[i32_0, i32_0]);

        let nullptr = cx.const_null(cx.type_ptr());
        let o_type = o_types[0];
        let args = vec![
            s_ident_t,
            cx.get_const_i64(u64::MAX),
            cx.get_const_i32(num_args),
            gep1,
            gep2,
            gep3,
            o_type,
            nullptr,
            nullptr,
        ];
        builder.call(fn_ty, begin, &args, None);

        // Step 4)
        unsafe { llvm::LLVMRustPositionAfter(builder.llbuilder, kernel_call) };

        let gep1 = builder.inbounds_gep(ty, a1, &[i32_0, i32_0]);
        let gep2 = builder.inbounds_gep(ty, a2, &[i32_0, i32_0]);
        let gep3 = builder.inbounds_gep(ty2, a4, &[i32_0, i32_0]);

        let nullptr = cx.const_null(cx.type_ptr());
        let o_type = o_types[0];
        let args = vec![
            s_ident_t,
            cx.get_const_i64(u64::MAX),
            cx.get_const_i32(num_args),
            gep1,
            gep2,
            gep3,
            o_type,
            nullptr,
            nullptr,
        ];
        builder.call(fn_ty, end, &args, None);

        // call void @__tgt_target_data_begin_mapper(ptr @1, i64 -1, i32 3, ptr %27, ptr %28, ptr %29, ptr @.offload_maptypes, ptr null, ptr null)
        // call void @__tgt_target_data_update_mapper(ptr @1, i64 -1, i32 2, ptr %46, ptr %47, ptr %48, ptr @.offload_maptypes.1, ptr null, ptr null)
        // call void @__tgt_target_data_end_mapper(ptr @1, i64 -1, i32 3, ptr %49, ptr %50, ptr %51, ptr @.offload_maptypes, ptr null, ptr null)
        // What is @1? Random but fixed:
        // @0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
        // @1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8
    }
}
