use std::ffi::CString;

use bitflags::Flags;
use llvm::Linkage::*;
use rustc_abi::Align;
use rustc_codegen_ssa::mir::operand::{OperandRef, OperandValue};
use rustc_codegen_ssa::traits::{BaseTypeCodegenMethods, BuilderMethods};
use rustc_middle::bug;
use rustc_middle::ty::offload_meta::{MappingFlags, OffloadMetadata, OffloadSize};

use crate::builder::Builder;
use crate::builder::gpu_helper::*;
use crate::common::CodegenCx;
use crate::intrinsic::TransferType;
use crate::llvm::AttributePlace::Function;
use crate::llvm::{self, Linkage, Type, Value};
use crate::{SimpleCx, attributes};

//int32 __kmpc_omp_taskwait(ident_t *loc_ref, kmp_int32 gtid);
//int32 __kmpc_global_thread_num(ident_t *);

// LLVM kernel-independent globals required for offloading
pub(crate) struct OffloadGlobals<'ll> {
    pub launcher_fn: &'ll llvm::Value,
    pub launcher_ty: &'ll llvm::Type,

    pub kernel_args_ty: &'ll llvm::Type,

    pub offload_entry_ty: &'ll llvm::Type,

    pub begin_mapper: &'ll llvm::Value,
    pub end_mapper: &'ll llvm::Value,
    pub mapper_fn_ty: &'ll llvm::Type,

    pub nowait_begin_mapper: &'ll llvm::Value,
    pub nowait_mapper_fn_ty: &'ll llvm::Type,

    pub ident_t_global: &'ll llvm::Value,

    pub taskwait: &'ll llvm::Value,
    pub taskwait_ty: &'ll llvm::Type,
    pub threadnum: &'ll llvm::Value,
    pub threadnum_ty: &'ll llvm::Type,

    pub async_info_create: &'ll Value,
    pub async_info_create_ty: &'ll Type,

    pub async_info_synchronize: &'ll Value,
    pub async_info_synchronize_ty: &'ll Type,

    pub async_info_destroy: &'ll Value,
    pub async_info_destroy_ty: &'ll Type,

    pub async_kernel_launcher: &'ll Value,
    pub async_kernel_launcher_ty: &'ll Type,
}

impl<'ll> OffloadGlobals<'ll> {
    pub(crate) fn declare(cx: &CodegenCx<'ll, '_>) -> Self {
        let (launcher_fn, launcher_ty) = generate_launcher(cx);
        let kernel_args_ty = KernelArgsTy::new_decl(cx);
        let offload_entry_ty = TgtOffloadEntry::new_decl(cx);
        let (begin_mapper, _, end_mapper, mapper_fn_ty) = gen_tgt_data_mappers(cx);
        let (nowait_begin_mapper, nowait_mapper_fn_ty) = gen_tgt_data_nowait_mappers(cx);
        let ident_t_global = generate_at_one(cx);
        let (taskwait, taskwait_ty, threadnum, threadnum_ty) = generate_sync(cx);
        // ptr __tgt_async_info_create(i64)
        let async_info_create_ty = cx.type_func(&[cx.type_i64()], cx.type_ptr());

        // i32 __tgt_async_info_synchronize(ptr)
        let async_info_synchronize_ty = cx.type_func(&[cx.type_ptr()], cx.type_i32());

        // void __tgt_async_info_destroy(ptr)
        let async_info_destroy_ty = cx.type_func(&[cx.type_ptr()], cx.type_void());

        // i32 __tgt_target_kernel_async(
        //     ptr ident,
        //     i64 device,
        //     i32 num_teams,
        //     i32 thread_limit,
        //     ptr host_ptr,
        //     ptr kernel_args,
        //     ptr async_info)
        let async_kernel_launcher_ty = cx.type_func(
            &[
                cx.type_ptr(),
                cx.type_i64(),
                cx.type_i32(),
                cx.type_i32(),
                cx.type_ptr(),
                cx.type_ptr(),
                cx.type_ptr(),
            ],
            cx.type_i32(),
        );
        // We want LLVM's openmp-opt pass to pick up and optimize this module, since it covers both
        // openmp and offload optimizations.
        llvm::add_module_flag_u32(cx.llmod(), llvm::ModuleFlagMergeBehavior::Max, "openmp", 51);

        let async_info_create =
            declare_offload_fn(cx, "__tgt_async_info_create", async_info_create_ty);

        let async_info_synchronize =
            declare_offload_fn(cx, "__tgt_async_info_synchronize", async_info_synchronize_ty);

        let async_info_destroy =
            declare_offload_fn(cx, "__tgt_async_info_destroy", async_info_destroy_ty);

        let async_kernel_launcher =
            declare_offload_fn(cx, "__tgt_target_kernel_async", async_kernel_launcher_ty);
        OffloadGlobals {
            launcher_fn,
            launcher_ty,
            kernel_args_ty,
            offload_entry_ty,
            begin_mapper,
            nowait_begin_mapper,
            end_mapper,
            mapper_fn_ty,
            nowait_mapper_fn_ty,
            ident_t_global,
            taskwait,
            taskwait_ty,
            threadnum,
            threadnum_ty,
            async_info_create,
            async_info_create_ty,

            async_info_synchronize,
            async_info_synchronize_ty,

            async_info_destroy,
            async_info_destroy_ty,

            async_kernel_launcher,
            async_kernel_launcher_ty,
        }
    }
}

// We need to register offload before using it. We also should unregister it once we are done, for
// good measures. Previously we have done so before and after each individual offload intrinsic
// call, but that comes at a performance cost. The repeated (un)register calls might also confuse
// the LLVM ompOpt pass, which tries to move operations to a better location. The easiest solution,
// which we copy from clang, is to just have those two calls once, in the global ctor/dtor section
// of the final binary.
pub(crate) fn register_offload<'ll>(cx: &CodegenCx<'ll, '_>) {
    // First we check quickly whether we already have done our setup, in which case we return early.
    // Shouldn't be needed for correctness.
    let register_lib_name = "__tgt_register_lib";
    if cx.get_function(register_lib_name).is_some() {
        return;
    }

    let reg_lib_decl = cx.type_func(&[cx.type_ptr()], cx.type_void());
    let register_lib = declare_offload_fn(&cx, register_lib_name, reg_lib_decl);
    let unregister_lib = declare_offload_fn(&cx, "__tgt_unregister_lib", reg_lib_decl);

    let ptr_null = cx.const_null(cx.type_ptr());
    let const_struct = cx.const_struct(&[cx.get_const_i32(0), ptr_null, ptr_null, ptr_null], false);
    let omp_descriptor =
        add_global(cx, ".omp_offloading.descriptor", const_struct, InternalLinkage);
    // @.omp_offloading.descriptor = internal constant %__tgt_bin_desc { i32 1, ptr @.omp_offloading.device_images, ptr @__start_llvm_offload_entries, ptr @__stop_llvm_offload_entries }
    // @.omp_offloading.descriptor = internal constant %__tgt_bin_desc { i32 0, ptr null, ptr null, ptr null }

    let atexit = cx.type_func(&[cx.type_ptr()], cx.type_i32());
    let atexit_fn = declare_offload_fn(cx, "atexit", atexit);

    // FIXME(offload): Drop this, once we fully automated our offload compilation pipeline, since
    // LLVM will initialize them for us if it sees gpu kernels being registered.
    let init_ty = cx.type_func(&[], cx.type_void());
    let init_rtls = declare_offload_fn(cx, "__tgt_init_all_rtls", init_ty);

    let desc_ty = cx.type_func(&[], cx.type_void());
    let reg_name = ".omp_offloading.descriptor_reg";
    let unreg_name = ".omp_offloading.descriptor_unreg";
    let desc_reg_fn = declare_offload_fn(cx, reg_name, desc_ty);
    let desc_unreg_fn = declare_offload_fn(cx, unreg_name, desc_ty);
    llvm::set_linkage(desc_reg_fn, InternalLinkage);
    llvm::set_linkage(desc_unreg_fn, InternalLinkage);
    llvm::set_section(desc_reg_fn, c".text.startup");
    llvm::set_section(desc_unreg_fn, c".text.startup");

    // define internal void @.omp_offloading.descriptor_reg() section ".text.startup" {
    // entry:
    //   call void @__tgt_register_lib(ptr @.omp_offloading.descriptor)
    //   call void @__tgt_init_all_rtls()
    //   %0 = call i32 @atexit(ptr @.omp_offloading.descriptor_unreg)
    //   ret void
    // }
    let bb = Builder::append_block(cx, desc_reg_fn, "entry");
    let mut a = Builder::build(cx, bb);
    a.call(reg_lib_decl, None, None, register_lib, &[omp_descriptor], None, None);
    a.call(init_ty, None, None, init_rtls, &[], None, None);
    a.call(atexit, None, None, atexit_fn, &[desc_unreg_fn], None, None);
    a.ret_void();

    // define internal void @.omp_offloading.descriptor_unreg() section ".text.startup" {
    // entry:
    //   call void @__tgt_unregister_lib(ptr @.omp_offloading.descriptor)
    //   ret void
    // }
    let bb = Builder::append_block(cx, desc_unreg_fn, "entry");
    let mut a = Builder::build(cx, bb);
    a.call(reg_lib_decl, None, None, unregister_lib, &[omp_descriptor], None, None);
    a.ret_void();

    // @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 101, ptr @.omp_offloading.descriptor_reg, ptr null }]
    let args = vec![cx.get_const_i32(101), desc_reg_fn, ptr_null];
    let const_struct = cx.const_struct(&args, false);
    let arr = cx.const_array(cx.val_ty(const_struct), &[const_struct]);
    add_global(cx, "llvm.global_ctors", arr, AppendingLinkage);
}

pub(crate) struct OffloadKernelDims<'ll> {
    num_workgroups: &'ll Value,
    threads_per_block: &'ll Value,
    workgroup_dims: &'ll Value,
    thread_dims: &'ll Value,
}

impl<'ll> OffloadKernelDims<'ll> {
    pub(crate) fn from_operands<'tcx>(
        builder: &mut Builder<'_, 'll, 'tcx>,
        workgroup_op: &OperandRef<'tcx, &'ll llvm::Value>,
        thread_op: &OperandRef<'tcx, &'ll llvm::Value>,
    ) -> Self {
        let cx = builder.cx;
        let arr_ty = cx.type_array(cx.type_i32(), 3);
        let four = Align::from_bytes(4).unwrap();

        let OperandValue::Ref(place) = workgroup_op.val else {
            bug!("expected array operand by reference");
        };
        let workgroup_val = builder.load(arr_ty, place.llval, four);

        let OperandValue::Ref(place) = thread_op.val else {
            bug!("expected array operand by reference");
        };
        let thread_val = builder.load(arr_ty, place.llval, four);

        fn mul_dim3<'ll, 'tcx>(
            builder: &mut Builder<'_, 'll, 'tcx>,
            arr: &'ll Value,
        ) -> &'ll Value {
            let x = builder.extract_value(arr, 0);
            let y = builder.extract_value(arr, 1);
            let z = builder.extract_value(arr, 2);

            let xy = builder.mul(x, y);
            builder.mul(xy, z)
        }

        let num_workgroups = mul_dim3(builder, workgroup_val);
        let threads_per_block = mul_dim3(builder, thread_val);

        OffloadKernelDims {
            workgroup_dims: workgroup_val,
            thread_dims: thread_val,
            num_workgroups,
            threads_per_block,
        }
    }
}

// ; Function Attrs: nounwind
// declare i32 @__tgt_target_kernel(ptr, i64, i32, i32, ptr, ptr) #2
fn generate_launcher<'ll>(cx: &CodegenCx<'ll, '_>) -> (&'ll llvm::Value, &'ll llvm::Type) {
    let tptr = cx.type_ptr();
    let ti64 = cx.type_i64();
    let ti32 = cx.type_i32();
    let args = vec![tptr, ti64, ti32, ti32, tptr, tptr];
    let tgt_fn_ty = cx.type_func(&args, ti32);
    let name = "__tgt_target_kernel";
    let tgt_decl = declare_offload_fn(&cx, name, tgt_fn_ty);
    let nounwind = llvm::AttributeKind::NoUnwind.create_attr(cx.llcx);
    attributes::apply_to_llfn(tgt_decl, Function, &[nounwind]);
    (tgt_decl, tgt_fn_ty)
}

fn generate_sync<'ll>(
    cx: &CodegenCx<'ll, '_>,
) -> (&'ll llvm::Value, &'ll llvm::Type, &'ll llvm::Value, &'ll llvm::Type) {
    let tptr = cx.type_ptr();
    let ti32 = cx.type_i32();
    let args1 = vec![tptr, ti32];
    let args2 = vec![tptr];
    let fn_ty1 = cx.type_func(&args1, ti32);
    let fn_ty2 = cx.type_func(&args2, ti32);
    let name1 = "__kmpc_omp_taskwait";
    let name2 = "__kmpc_global_thread_num";
    let decl1 = declare_offload_fn(&cx, name1, fn_ty1);
    let decl2 = declare_offload_fn(&cx, name2, fn_ty2);

    let nounwind = llvm::AttributeKind::NoUnwind.create_attr(cx.llcx);
    attributes::apply_to_llfn(decl1, Function, &[nounwind]);
    attributes::apply_to_llfn(decl2, Function, &[nounwind]);
    //int32 __kmpc_omp_taskwait(ident_t *loc_ref, kmp_int32 gtid);
    //int32 __kmpc_global_thread_num(ident_t *);
    (decl1, fn_ty1, decl2, fn_ty2)
}

// What is our @1 here? A magic global, used in our data_{begin/update/end}_mapper:
// @0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
// @1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8
// FIXME(offload): @0 should include the file name (e.g. lib.rs) in which the function to be
// offloaded was defined.
pub(crate) fn generate_at_one<'ll>(cx: &CodegenCx<'ll, '_>) -> &'ll llvm::Value {
    let unknown_txt = ";unknown;unknown;0;0;;";
    let c_entry_name = CString::new(unknown_txt).unwrap();
    let c_val = c_entry_name.as_bytes_with_nul();
    let initializer = crate::common::bytes_in_context(cx.llcx, c_val);
    let at_zero = add_unnamed_global(&cx, &"", initializer, PrivateLinkage);
    llvm::set_alignment(at_zero, Align::ONE);

    // @1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8
    let struct_ident_ty = cx.type_named_struct("struct.ident_t");
    let struct_elems = vec![
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
    at_one
}

pub(crate) struct TgtOffloadEntry {
    //   uint64_t Reserved;
    //   uint16_t Version;
    //   uint16_t Kind;
    //   uint32_t Flags; Flags associated with the entry (see Target Region Entry Flags)
    //   void *Address; Address of global symbol within device image (function or global)
    //   char *SymbolName;
    //   uint64_t Size; Size of the entry info (0 if it is a function)
    //   uint64_t Data;
    //   void *AuxAddr;
}

impl TgtOffloadEntry {
    pub(crate) fn new_decl<'ll>(cx: &CodegenCx<'ll, '_>) -> &'ll llvm::Type {
        let offload_entry_ty = cx.type_named_struct("struct.__tgt_offload_entry");
        let tptr = cx.type_ptr();
        let ti64 = cx.type_i64();
        let ti32 = cx.type_i32();
        let ti16 = cx.type_i16();
        // For each kernel to run on the gpu, we will later generate one entry of this type.
        // copied from LLVM
        let entry_elements = vec![ti64, ti16, ti16, ti32, tptr, tptr, ti64, ti64, tptr];
        cx.set_struct_body(offload_entry_ty, &entry_elements, false);
        offload_entry_ty
    }

    fn new<'ll>(
        cx: &CodegenCx<'ll, '_>,
        region_id: &'ll Value,
        llglobal: &'ll Value,
    ) -> [&'ll Value; 9] {
        let reserved = cx.get_const_i64(0);
        let version = cx.get_const_i16(1);
        let kind = cx.get_const_i16(1);
        let flags = cx.get_const_i32(0);
        let size = cx.get_const_i64(0);
        let data = cx.get_const_i64(0);
        let aux_addr = cx.const_null(cx.type_ptr());
        [reserved, version, kind, flags, region_id, llglobal, size, data, aux_addr]
    }
}

// Taken from the LLVM APITypes.h declaration:
struct KernelArgsTy {
    //  uint32_t Version = 0; // Version of this struct for ABI compatibility.
    //  uint32_t NumArgs = 0; // Number of arguments in each input pointer.
    //  void **ArgBasePtrs =
    //      nullptr;                 // Base pointer of each argument (e.g. a struct).
    //  void **ArgPtrs = nullptr;    // Pointer to the argument data.
    //  int64_t *ArgSizes = nullptr; // Size of the argument data in bytes.
    //  int64_t *ArgTypes = nullptr; // Type of the data (e.g. to / from).
    //  void **ArgNames = nullptr;   // Name of the data for debugging, possibly null.
    //  void **ArgMappers = nullptr; // User-defined mappers, possibly null.
    //  uint64_t Tripcount =
    // 0; // Tripcount for the teams / distribute loop, 0 otherwise.
    // struct {
    //    uint64_t NoWait : 1; // Was this kernel spawned with a `nowait` clause.
    //    uint64_t IsCUDA : 1; // Was this kernel spawned via CUDA.
    //    uint64_t Unused : 62;
    //  } Flags = {0, 0, 0}; // totals to 64 Bit, 8 Byte
    //  // The number of teams (for x,y,z dimension).
    //  uint32_t NumTeams[3] = {0, 0, 0};
    //  // The number of threads (for x,y,z dimension).
    //  uint32_t ThreadLimit[3] = {0, 0, 0};
    //  uint32_t DynCGroupMem = 0; // Amount of dynamic cgroup memory requested.
}

impl KernelArgsTy {
    const OFFLOAD_VERSION: u64 = 3;
    const FLAGS: u64 = 0;
    const TRIPCOUNT: u64 = 0;
    fn new_decl<'ll>(cx: &CodegenCx<'ll, '_>) -> &'ll Type {
        let kernel_arguments_ty = cx.type_named_struct("struct.__tgt_kernel_arguments");
        let tptr = cx.type_ptr();
        let ti64 = cx.type_i64();
        let ti32 = cx.type_i32();
        let tarr = cx.type_array(ti32, 3);

        let kernel_elements =
            vec![ti32, ti32, tptr, tptr, tptr, tptr, tptr, tptr, ti64, ti64, tarr, tarr, ti32];

        cx.set_struct_body(kernel_arguments_ty, &kernel_elements, false);
        kernel_arguments_ty
    }

    fn new<'ll, 'tcx>(
        cx: &CodegenCx<'ll, 'tcx>,
        num_args: u64,
        memtransfer_types: &'ll Value,
        geps: [&'ll Value; 3],
        workgroup_dims: &'ll Value,
        thread_dims: &'ll Value,
        dyn_cache: &'ll Value,
    ) -> [(Align, &'ll str, &'ll Value); 13] {
        let four = Align::from_bytes(4).expect("4 Byte alignment should work");
        let eight = Align::EIGHT;

        [
            (four, "Version", cx.get_const_i32(KernelArgsTy::OFFLOAD_VERSION)),
            (four, "NumArgs", cx.get_const_i32(num_args)),
            (eight, "ArgBasePtrs", geps[0]),
            (eight, "ArgPtrs", geps[1]),
            (eight, "ArgSizes", geps[2]),
            (eight, "ArgTypes", memtransfer_types),
            // The next two are debug infos. FIXME(offload): set them
            (eight, "ArgNames", cx.const_null(cx.type_ptr())), // dbg
            (eight, "ArgMappers", cx.const_null(cx.type_ptr())), // dbg
            (eight, "Tripcount", cx.get_const_i64(KernelArgsTy::TRIPCOUNT)),
            (eight, "Flags", cx.get_const_i64(KernelArgsTy::FLAGS)),
            (four, "NumTeams", workgroup_dims),
            (four, "ThreadLimit", thread_dims),
            (four, "DynCGroupMem", dyn_cache),
        ]
    }
}

// Contains LLVM values needed to manage offloading for a single kernel.
#[derive(Copy, Clone)]
pub(crate) struct OffloadKernelGlobals<'ll> {
    pub offload_sizes: &'ll llvm::Value,
    pub memtransfer_begin: &'ll llvm::Value,
    pub memtransfer_kernel: Option<&'ll llvm::Value>,
    pub memtransfer_end: &'ll llvm::Value,
    pub region_id: &'ll llvm::Value,
}

fn gen_tgt_data_nowait_mappers<'ll>(
    cx: &CodegenCx<'ll, '_>,
) -> (&'ll llvm::Value, &'ll llvm::Type) {
    let tptr = cx.type_ptr();
    let ti64 = cx.type_i64();
    let ti32 = cx.type_i32();

    let args = vec![tptr, ti64, ti32, tptr, tptr, tptr, tptr, tptr, tptr, ti32, tptr, ti32, tptr];
    let mapper_fn_ty = cx.type_func(&args, cx.type_void());
    let mapper_begin = "__tgt_target_data_begin_nowait_mapper";
    let begin_mapper_decl = declare_offload_fn(&cx, mapper_begin, mapper_fn_ty);

    let nounwind = llvm::AttributeKind::NoUnwind.create_attr(cx.llcx);
    attributes::apply_to_llfn(begin_mapper_decl, Function, &[nounwind]);

    (begin_mapper_decl, mapper_fn_ty)
}

fn gen_tgt_data_mappers<'ll>(
    cx: &CodegenCx<'ll, '_>,
) -> (&'ll llvm::Value, &'ll llvm::Value, &'ll llvm::Value, &'ll llvm::Type) {
    let tptr = cx.type_ptr();
    let ti64 = cx.type_i64();
    let ti32 = cx.type_i32();

    let args = vec![tptr, ti64, ti32, tptr, tptr, tptr, tptr, tptr, tptr];
    let mapper_fn_ty = cx.type_func(&args, cx.type_void());
    let mapper_begin = "__tgt_target_data_begin_mapper";
    let mapper_update = "__tgt_target_data_update_mapper";
    let mapper_end = "__tgt_target_data_end_mapper";
    let begin_mapper_decl = declare_offload_fn(&cx, mapper_begin, mapper_fn_ty);
    let update_mapper_decl = declare_offload_fn(&cx, mapper_update, mapper_fn_ty);
    let end_mapper_decl = declare_offload_fn(&cx, mapper_end, mapper_fn_ty);

    let nounwind = llvm::AttributeKind::NoUnwind.create_attr(cx.llcx);
    attributes::apply_to_llfn(begin_mapper_decl, Function, &[nounwind]);
    attributes::apply_to_llfn(update_mapper_decl, Function, &[nounwind]);
    attributes::apply_to_llfn(end_mapper_decl, Function, &[nounwind]);

    (begin_mapper_decl, update_mapper_decl, end_mapper_decl, mapper_fn_ty)
}

fn add_priv_unnamed_arr<'ll>(cx: &SimpleCx<'ll>, name: &str, vals: &[u64]) -> &'ll llvm::Value {
    let ti64 = cx.type_i64();
    let mut size_val = Vec::with_capacity(vals.len());
    for &val in vals {
        size_val.push(cx.get_const_i64(val));
    }
    let initializer = cx.const_array(ti64, &size_val);
    add_unnamed_global(cx, name, initializer, PrivateLinkage)
}

pub(crate) fn add_unnamed_global<'ll>(
    cx: &SimpleCx<'ll>,
    name: &str,
    initializer: &'ll llvm::Value,
    l: Linkage,
) -> &'ll llvm::Value {
    let llglobal = add_global(cx, name, initializer, l);
    llvm::LLVMSetUnnamedAddress(llglobal, llvm::UnnamedAddr::Global);
    llglobal
}

pub(crate) fn add_global<'ll>(
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

// This function returns a memtransfer value which encodes how arguments to this kernel shall be
// mapped to/from the gpu. It also returns a region_id with the name of this kernel, to be
// concatenated into the list of region_ids.
pub(crate) fn gen_define_handling<'ll>(
    cx: &CodegenCx<'ll, '_>,
    metadata: &[OffloadMetadata],
    symbol: String,
    offload_globals: &OffloadGlobals<'ll>,
    transfer: TransferType,
) -> OffloadKernelGlobals<'ll> {
    if let Some(entry) = cx.offload_kernel_cache.borrow().get(&symbol) {
        return *entry;
    }

    let offload_entry_ty = offload_globals.offload_entry_ty;
    let gen_kernel = matches!(transfer, TransferType::Kernel);

    let (sizes, transfer): (Vec<_>, Vec<_>) =
        metadata.iter().map(|m| (m.payload_size, m.mode)).unzip();
    // Our begin mapper should only see simplified information about which args have to be
    // transferred to the device, the end mapper only about which args should be transferred back.
    // Any information beyond that makes it harder for LLVM's opt pass to evaluate whether it can
    // safely move (=optimize) the LLVM-IR location of this data transfer. Only the mapping types
    // mentioned below are handled, so make sure that we don't generate any other ones.
    let handled_mappings = MappingFlags::TO
        | MappingFlags::FROM
        | MappingFlags::TARGET_PARAM
        | MappingFlags::LITERAL
        | MappingFlags::IMPLICIT;
    for arg in &transfer {
        debug_assert!(!arg.contains_unknown_bits());
        debug_assert!(handled_mappings.contains(*arg));
    }

    let valid_begin_mappings = MappingFlags::TO | MappingFlags::LITERAL | MappingFlags::IMPLICIT;
    let transfer_to: Vec<u64> =
        transfer.iter().map(|m| m.intersection(valid_begin_mappings).bits()).collect();
    //dbg!(&transfer);
    let transfer_from: Vec<u64> =
        transfer.iter().map(|m| m.intersection(MappingFlags::FROM).bits()).collect();
    let valid_kernel_mappings = MappingFlags::LITERAL | MappingFlags::IMPLICIT;
    // FIXME(offload): add `OMP_MAP_TARGET_PARAM = 0x20` only if necessary
    let transfer_kernel: Vec<u64> = transfer
        .iter()
        .map(|m| (m.intersection(valid_kernel_mappings) | MappingFlags::TARGET_PARAM).bits())
        .collect();

    let actual_sizes = sizes
        .iter()
        .map(|s| match s {
            OffloadSize::Static(sz) => *sz,
            // NOTE(Sa4dUs): set `.offload_sizes` entry to 0 for sizes that we determine at runtime, just like clang
            _ => 0,
        })
        .collect::<Vec<_>>();
    let offload_sizes =
        add_priv_unnamed_arr(&cx, &format!(".offload_sizes.{symbol}"), &actual_sizes);
    let memtransfer_begin =
        add_priv_unnamed_arr(&cx, &format!(".offload_maptypes.{symbol}.begin"), &transfer_to);

    let memtransfer_kernel = if gen_kernel {
        Some(add_priv_unnamed_arr(
            &cx,
            &format!(".offload_maptypes.{symbol}.kernel"),
            &transfer_kernel,
        ))
    } else {
        None
    };
    let memtransfer_end =
        add_priv_unnamed_arr(&cx, &format!(".offload_maptypes.{symbol}.end"), &transfer_from);

    // Next: For each function, generate these three entries. A weak constant,
    // the llvm.rodata entry name, and  the llvm_offload_entries value

    let name = format!(".{symbol}.region_id");
    let initializer = cx.get_const_i8(0);
    let region_id = add_global(&cx, &name, initializer, WeakAnyLinkage);

    if gen_kernel {
        let c_entry_name = CString::new(symbol.clone()).unwrap();
        let c_val = c_entry_name.as_bytes_with_nul();
        let offload_entry_name = format!(".offloading.entry_name.{symbol}");

        let initializer = crate::common::bytes_in_context(cx.llcx, c_val);
        let llglobal = add_unnamed_global(&cx, &offload_entry_name, initializer, InternalLinkage);
        llvm::set_alignment(llglobal, Align::ONE);
        llvm::set_section(llglobal, c".llvm.rodata.offloading");

        let name = format!(".offloading.entry.{symbol}");

        // See the __tgt_offload_entry documentation above.
        let elems = TgtOffloadEntry::new(&cx, region_id, llglobal);

        let initializer = crate::common::named_struct(offload_entry_ty, &elems);
        let c_name = CString::new(name).unwrap();
        let offload_entry = llvm::add_global(cx.llmod, offload_entry_ty, &c_name);
        llvm::set_global_constant(offload_entry, true);
        llvm::set_linkage(offload_entry, WeakAnyLinkage);
        llvm::set_initializer(offload_entry, initializer);
        llvm::set_alignment(offload_entry, Align::EIGHT);
        let c_section_name = CString::new("llvm_offload_entries").unwrap();
        llvm::set_section(offload_entry, &c_section_name);

        cx.add_compiler_used_global(offload_entry);
    }

    let result = OffloadKernelGlobals {
        offload_sizes,
        memtransfer_begin,
        memtransfer_kernel,
        memtransfer_end,
        region_id,
    };

    cx.offload_kernel_cache.borrow_mut().insert(symbol, result);

    result
}

fn declare_offload_fn<'ll>(
    cx: &CodegenCx<'ll, '_>,
    name: &str,
    ty: &'ll llvm::Type,
) -> &'ll llvm::Value {
    crate::declare::declare_simple_fn(
        cx,
        name,
        llvm::CallConv::CCallConv,
        llvm::UnnamedAddr::No,
        llvm::Visibility::Default,
        ty,
    )
}

// For each kernel *call*, we now use some of our previous declared globals to move data to and from
// the gpu. For now, we only handle the data transfer part of it.
// If two consecutive kernels use the same memory, we still move it to the host and back to the gpu.
// Since in our frontend users (by default) don't have to specify data transfer, this is something
// we should optimize in the future! In some cases we can directly zero-allocate on the device and
// only move data back, or if something is immutable, we might only copy it to the device.
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
pub(crate) fn gen_call_handling<'ll, 'tcx>(
    builder: &mut Builder<'_, 'll, 'tcx>,
    offload_data: &OffloadKernelGlobals<'ll>,
    args: &[&'ll Value],
    types: &[&Type],
    metadata: &[OffloadMetadata],
    offload_globals: &OffloadGlobals<'ll>,
    offload_dims: &OffloadKernelDims<'ll>,
    dyn_cache: &'ll Value,
) {
    let cx = builder.cx;
    let OffloadKernelGlobals {
        offload_sizes,
        memtransfer_begin,
        memtransfer_kernel,
        memtransfer_end,
        region_id,
    } = offload_data;
    let memtransfer_kernel = memtransfer_kernel.unwrap();
    let OffloadKernelDims { num_workgroups, threads_per_block, workgroup_dims, thread_dims } =
        offload_dims;

    let has_dynamic = metadata.iter().any(|m| !matches!(m.payload_size, OffloadSize::Static(_)));

    let tgt_decl = offload_globals.launcher_fn;
    let tgt_target_kernel_ty = offload_globals.launcher_ty;

    let tgt_kernel_decl = offload_globals.kernel_args_ty;
    let begin_mapper_decl = offload_globals.begin_mapper;
    let end_mapper_decl = offload_globals.end_mapper;
    let fn_ty = offload_globals.mapper_fn_ty;

    let (ty, ty2, a1, a2, a4) =
        preper_datatransfers(builder, args, types, offload_sizes, metadata, has_dynamic);
    let num_args = types.len() as u64;
    assert_eq!(num_args as usize, args.len());

    let bb = builder.llbb();
    unsafe {
        llvm::LLVMRustPositionBuilderPastAllocas(&builder.llbuilder, builder.llfn());
    }
    //%kernel_args = alloca %struct.__tgt_kernel_arguments, align 8
    let a5 = builder.direct_alloca(tgt_kernel_decl, Align::EIGHT, "kernel_args");
    unsafe {
        llvm::LLVMPositionBuilderAtEnd(&builder.llbuilder, bb);
    }

    // Step 2)
    let s_ident_t = offload_globals.ident_t_global;
    let geps = get_geps(builder, ty, ty2, a1, a2, a4, has_dynamic);
    generate_mapper_call(
        builder,
        geps,
        offload_globals,
        memtransfer_begin,
        begin_mapper_decl,
        fn_ty,
        num_args,
        s_ident_t,
        TransferType::Kernel,
    );
    let values = KernelArgsTy::new(
        &cx,
        num_args,
        memtransfer_kernel,
        geps,
        workgroup_dims,
        thread_dims,
        dyn_cache,
    );

    // Step 3)
    // Here we fill the KernelArgsTy, see the documentation above
    let i32_0 = cx.get_const_i32(0);
    for (i, value) in values.iter().enumerate() {
        let ptr = builder.inbounds_gep(tgt_kernel_decl, a5, &[i32_0, cx.get_const_i32(i as u64)]);
        let name = std::ffi::CString::new(value.1).unwrap();
        llvm::set_value_name(ptr, &name.as_bytes());

        builder.store(value.2, ptr, value.0);
    }

    let args = vec![
        s_ident_t,
        // FIXME(offload) give users a way to select which GPU to use.
        cx.get_const_i64(u64::MAX), // MAX == -1.
        num_workgroups,
        threads_per_block,
        region_id,
        a5,
    ];
    builder.call(tgt_target_kernel_ty, None, None, tgt_decl, &args, None, None);
    // %41 = call i32 @__tgt_target_kernel(ptr @1, i64 -1, i32 2097152, i32 256, ptr @.kernel_1.region_id, ptr %kernel_args)

    // Step 4)
    let geps = get_geps(builder, ty, ty2, a1, a2, a4, has_dynamic);
    generate_mapper_call(
        builder,
        geps,
        offload_globals,
        memtransfer_end,
        end_mapper_decl,
        fn_ty,
        num_args,
        s_ident_t,
        TransferType::Kernel,
    );
}
