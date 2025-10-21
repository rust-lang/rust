use std::ffi::CString;

use llvm::Linkage::*;
use rustc_abi::Align;
use rustc_codegen_ssa::back::write::CodegenContext;
use rustc_codegen_ssa::traits::BaseTypeCodegenMethods;
use rustc_middle::ty::{self, PseudoCanonicalInput, Ty, TyCtxt, TypingEnv};

use crate::builder::SBuilder;
use crate::llvm::AttributePlace::Function;
use crate::llvm::{self, BasicBlock, Linkage, Type, Value};
use crate::{LlvmCodegenBackend, SimpleCx, attributes};

pub(crate) fn handle_gpu_code<'ll>(
    _cgcx: &CodegenContext<LlvmCodegenBackend>,
    _cx: &'ll SimpleCx<'_>,
) {
    /*
    // The offload memory transfer type for each kernel
    let mut memtransfer_types = vec![];
    let mut region_ids = vec![];
    let offload_entry_ty = TgtOffloadEntry::new_decl(&cx);
    for num in 0..9 {
        let kernel = cx.get_function(&format!("kernel_{num}"));
        if let Some(kernel) = kernel {
            let (o, k) = gen_define_handling(&cx, kernel, offload_entry_ty, num);
            memtransfer_types.push(o);
            region_ids.push(k);
        }
    }

    gen_call_handling(&cx, &memtransfer_types, &region_ids);
    */
}

// ; Function Attrs: nounwind
// declare i32 @__tgt_target_kernel(ptr, i64, i32, i32, ptr, ptr) #2
fn generate_launcher<'ll>(cx: &'ll SimpleCx<'_>) -> (&'ll llvm::Value, &'ll llvm::Type) {
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

// What is our @1 here? A magic global, used in our data_{begin/update/end}_mapper:
// @0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
// @1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8
// FIXME(offload): @0 should include the file name (e.g. lib.rs) in which the function to be
// offloaded was defined.
fn generate_at_one<'ll>(cx: &'ll SimpleCx<'_>) -> &'ll llvm::Value {
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
    pub(crate) fn new_decl<'ll>(cx: &'ll SimpleCx<'_>) -> &'ll llvm::Type {
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
        cx: &'ll SimpleCx<'_>,
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
    fn new_decl<'ll>(cx: &'ll SimpleCx<'_>) -> &'ll Type {
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

    fn new<'ll>(
        cx: &'ll SimpleCx<'_>,
        num_args: u64,
        memtransfer_types: &[&'ll Value],
        geps: [&'ll Value; 3],
    ) -> [(Align, &'ll Value); 13] {
        let four = Align::from_bytes(4).expect("4 Byte alignment should work");
        let eight = Align::EIGHT;

        let ti32 = cx.type_i32();
        let ci32_0 = cx.get_const_i32(0);
        [
            (four, cx.get_const_i32(KernelArgsTy::OFFLOAD_VERSION)),
            (four, cx.get_const_i32(num_args)),
            (eight, geps[0]),
            (eight, geps[1]),
            (eight, geps[2]),
            (eight, memtransfer_types[0]),
            // The next two are debug infos. FIXME(offload): set them
            (eight, cx.const_null(cx.type_ptr())), // dbg
            (eight, cx.const_null(cx.type_ptr())), // dbg
            (eight, cx.get_const_i64(KernelArgsTy::TRIPCOUNT)),
            (eight, cx.get_const_i64(KernelArgsTy::FLAGS)),
            (four, cx.const_array(ti32, &[cx.get_const_i32(2097152), ci32_0, ci32_0])),
            (four, cx.const_array(ti32, &[cx.get_const_i32(256), ci32_0, ci32_0])),
            (four, cx.get_const_i32(0)),
        ]
    }
}

fn gen_tgt_data_mappers<'ll>(
    cx: &'ll SimpleCx<'_>,
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
pub(crate) fn gen_define_handling<'ll, 'tcx>(
    cx: &SimpleCx<'ll>,
    tcx: TyCtxt<'tcx>,
    kernel: &'ll llvm::Value,
    offload_entry_ty: &'ll llvm::Type,
    // TODO(Sa4dUs): Define a typetree once i have a better idea of what do we exactly need
    tt: Vec<Ty<'tcx>>,
    symbol: &str,
) -> (&'ll llvm::Value, &'ll llvm::Value) {
    let types = cx.func_params_types(cx.get_type_of_global(kernel));
    // It seems like non-pointer values are automatically mapped. So here, we focus on pointer (or
    // reference) types.
    let num_ptr_types = types
        .iter()
        .filter(|&x| matches!(cx.type_kind(x), rustc_codegen_ssa::common::TypeKind::Pointer))
        .count();

    // TODO(Sa4dUs): Add typetrees here
    let ptr_sizes = types
        .iter()
        .zip(tt)
        .filter_map(|(&x, ty)| match cx.type_kind(x) {
            rustc_codegen_ssa::common::TypeKind::Pointer => Some(get_payload_size(tcx, ty)),
            _ => None,
        })
        .collect::<Vec<u64>>();

    // We do not know their size anymore at this level, so hardcode a placeholder.
    // A follow-up pr will track these from the frontend, where we still have Rust types.
    // Then, we will be able to figure out that e.g. `&[f32;256]` will result in 4*256 bytes.
    // I decided that 1024 bytes is a great placeholder value for now.
    add_priv_unnamed_arr(&cx, &format!(".offload_sizes.{symbol}"), &ptr_sizes);
    // Here we figure out whether something needs to be copied to the gpu (=1), from the gpu (=2),
    // or both to and from the gpu (=3). Other values shouldn't affect us for now.
    // A non-mutable reference or pointer will be 1, an array that's not read, but fully overwritten
    // will be 2. For now, everything is 3, until we have our frontend set up.
    // 1+2+32: 1 (MapTo), 2 (MapFrom), 32 (Add one extra input ptr per function, to be used later).
    let memtransfer_types = add_priv_unnamed_arr(
        &cx,
        &format!(".offload_maptypes.{symbol}"),
        &vec![1 + 2 + 32; num_ptr_types],
    );

    // Next: For each function, generate these three entries. A weak constant,
    // the llvm.rodata entry name, and  the llvm_offload_entries value

    let name = format!(".{symbol}.region_id");
    let initializer = cx.get_const_i8(0);
    let region_id = add_unnamed_global(&cx, &name, initializer, WeakAnyLinkage);

    let c_entry_name = CString::new(symbol).unwrap();
    let c_val = c_entry_name.as_bytes_with_nul();
    let offload_entry_name = format!(".offloading.entry_name.{symbol}");

    let initializer = crate::common::bytes_in_context(cx.llcx, c_val);
    let llglobal = add_unnamed_global(&cx, &offload_entry_name, initializer, InternalLinkage);
    llvm::set_alignment(llglobal, Align::ONE);
    llvm::set_section(llglobal, c".llvm.rodata.offloading");

    // Not actively used yet, for calling real kernels
    let name = format!(".offloading.entry.{symbol}");

    // See the __tgt_offload_entry documentation above.
    let elems = TgtOffloadEntry::new(&cx, region_id, llglobal);

    let initializer = crate::common::named_struct(offload_entry_ty, &elems);
    let c_name = CString::new(name).unwrap();
    let llglobal = llvm::add_global(cx.llmod, offload_entry_ty, &c_name);
    llvm::set_global_constant(llglobal, true);
    llvm::set_linkage(llglobal, WeakAnyLinkage);
    llvm::set_initializer(llglobal, initializer);
    llvm::set_alignment(llglobal, Align::EIGHT);
    let c_section_name = CString::new("llvm_offload_entries").unwrap();
    llvm::set_section(llglobal, &c_section_name);
    (memtransfer_types, region_id)
}

// TODO(Sa4dUs): move this to a proper place
fn get_payload_size<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> u64 {
    match ty.kind() {
        /*
        rustc_middle::infer::canonical::ir::TyKind::Bool => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Char => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Int(int_ty) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Uint(uint_ty) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Float(float_ty) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Adt(_, _) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Foreign(_) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Str => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Array(_, _) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Pat(_, _) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Slice(_) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::RawPtr(_, mutability) => todo!(),
        */
        ty::Ref(_, inner, _) => get_payload_size(tcx, *inner),
        /*
        rustc_middle::infer::canonical::ir::TyKind::FnDef(_, _) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::FnPtr(binder, fn_header) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::UnsafeBinder(unsafe_binder_inner) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Dynamic(_, _) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Closure(_, _) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::CoroutineClosure(_, _) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Coroutine(_, _) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::CoroutineWitness(_, _) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Never => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Tuple(_) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Alias(alias_ty_kind, alias_ty) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Param(_) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Bound(bound_var_index_kind, _) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Placeholder(_) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Infer(infer_ty) => todo!(),
        rustc_middle::infer::canonical::ir::TyKind::Error(_) => todo!(),
        */
        _ => {
            tcx
                // TODO(Sa4dUs): Maybe `.as_query_input()`?
                .layout_of(PseudoCanonicalInput {
                    typing_env: TypingEnv::fully_monomorphized(),
                    value: ty,
                })
                .unwrap()
                .size
                .bytes()
        }
    }
}

fn declare_offload_fn<'ll>(
    cx: &'ll SimpleCx<'_>,
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
pub(crate) fn gen_call_handling<'ll>(
    cx: &SimpleCx<'ll>,
    bb: &BasicBlock,
    kernels: &[&'ll llvm::Value],
    memtransfer_types: &[&'ll llvm::Value],
    region_ids: &[&'ll llvm::Value],
    llfn: &'ll Value,
) {
    let (tgt_decl, tgt_target_kernel_ty) = generate_launcher(&cx);
    // %struct.__tgt_bin_desc = type { i32, ptr, ptr, ptr }
    let tptr = cx.type_ptr();
    let ti32 = cx.type_i32();
    let tgt_bin_desc_ty = vec![ti32, tptr, tptr, tptr];
    let tgt_bin_desc = cx.type_named_struct("struct.__tgt_bin_desc");
    cx.set_struct_body(tgt_bin_desc, &tgt_bin_desc_ty, false);

    let tgt_kernel_decl = KernelArgsTy::new_decl(&cx);
    let (begin_mapper_decl, _, end_mapper_decl, fn_ty) = gen_tgt_data_mappers(&cx);

    let mut builder = SBuilder::build(cx, bb);

    let types = cx.func_params_types(cx.get_type_of_global(kernels[0]));
    let num_args = types.len() as u64;

    // Step 0)
    // %struct.__tgt_bin_desc = type { i32, ptr, ptr, ptr }
    // %6 = alloca %struct.__tgt_bin_desc, align 8
    let tgt_bin_desc_alloca = builder.direct_alloca(tgt_bin_desc, Align::EIGHT, "EmptyDesc");

    let ty = cx.type_array(cx.type_ptr(), num_args);
    // Baseptr are just the input pointer to the kernel, stored in a local alloca
    let a1 = builder.direct_alloca(ty, Align::EIGHT, ".offload_baseptrs");
    // Ptrs are the result of a gep into the baseptr, at least for our trivial types.
    let a2 = builder.direct_alloca(ty, Align::EIGHT, ".offload_ptrs");
    // These represent the sizes in bytes, e.g. the entry for `&[f64; 16]` will be 8*16.
    let ty2 = cx.type_array(cx.type_i64(), num_args);
    let a4 = builder.direct_alloca(ty2, Align::EIGHT, ".offload_sizes");

    //%kernel_args = alloca %struct.__tgt_kernel_arguments, align 8
    let a5 = builder.direct_alloca(tgt_kernel_decl, Align::EIGHT, "kernel_args");

    // Step 1)
    builder.memset(tgt_bin_desc_alloca, cx.get_const_i8(0), cx.get_const_i64(32), Align::EIGHT);

    // Now we allocate once per function param, a copy to be passed to one of our maps.
    let mut vals = vec![];
    let mut geps = vec![];
    let i32_0 = cx.get_const_i32(0);
    for index in 0..num_args {
        let v = unsafe { llvm::LLVMGetParam(llfn, index as u32) };
        let gep = builder.inbounds_gep(cx.type_f32(), v, &[i32_0]);
        vals.push(v);
        geps.push(gep);
    }

    let mapper_fn_ty = cx.type_func(&[cx.type_ptr()], cx.type_void());
    let register_lib_decl = declare_offload_fn(&cx, "__tgt_register_lib", mapper_fn_ty);
    let unregister_lib_decl = declare_offload_fn(&cx, "__tgt_unregister_lib", mapper_fn_ty);
    let init_ty = cx.type_func(&[], cx.type_void());
    let init_rtls_decl = declare_offload_fn(cx, "__tgt_init_all_rtls", init_ty);

    // FIXME(offload): Later we want to add them to the wrapper code, rather than our main function.
    // call void @__tgt_register_lib(ptr noundef %6)
    builder.call(mapper_fn_ty, register_lib_decl, &[tgt_bin_desc_alloca], None);
    // call void @__tgt_init_all_rtls()
    builder.call(init_ty, init_rtls_decl, &[], None);

    for i in 0..num_args {
        let idx = cx.get_const_i32(i);
        let gep1 = builder.inbounds_gep(ty, a1, &[i32_0, idx]);
        builder.store(vals[i as usize], gep1, Align::EIGHT);
        let gep2 = builder.inbounds_gep(ty, a2, &[i32_0, idx]);
        builder.store(geps[i as usize], gep2, Align::EIGHT);
        let gep3 = builder.inbounds_gep(ty2, a4, &[i32_0, idx]);
        // As mentioned above, we don't use Rust type information yet. So for now we will just
        // assume that we have 1024 bytes, 256 f32 values.
        // FIXME(offload): write an offload frontend and handle arbitrary types.
        builder.store(cx.get_const_i64(1024), gep3, Align::EIGHT);
    }

    // For now we have a very simplistic indexing scheme into our
    // offload_{baseptrs,ptrs,sizes}. We will probably improve this along with our gpu frontend pr.
    fn get_geps<'a, 'll>(
        builder: &mut SBuilder<'a, 'll>,
        cx: &'ll SimpleCx<'ll>,
        ty: &'ll Type,
        ty2: &'ll Type,
        a1: &'ll Value,
        a2: &'ll Value,
        a4: &'ll Value,
    ) -> [&'ll Value; 3] {
        let i32_0 = cx.get_const_i32(0);

        let gep1 = builder.inbounds_gep(ty, a1, &[i32_0, i32_0]);
        let gep2 = builder.inbounds_gep(ty, a2, &[i32_0, i32_0]);
        let gep3 = builder.inbounds_gep(ty2, a4, &[i32_0, i32_0]);
        [gep1, gep2, gep3]
    }

    fn generate_mapper_call<'a, 'll>(
        builder: &mut SBuilder<'a, 'll>,
        cx: &'ll SimpleCx<'ll>,
        geps: [&'ll Value; 3],
        o_type: &'ll Value,
        fn_to_call: &'ll Value,
        fn_ty: &'ll Type,
        num_args: u64,
        s_ident_t: &'ll Value,
    ) {
        let nullptr = cx.const_null(cx.type_ptr());
        let i64_max = cx.get_const_i64(u64::MAX);
        let num_args = cx.get_const_i32(num_args);
        let args =
            vec![s_ident_t, i64_max, num_args, geps[0], geps[1], geps[2], o_type, nullptr, nullptr];
        builder.call(fn_ty, fn_to_call, &args, None);
    }

    // Step 2)
    let s_ident_t = generate_at_one(&cx);
    let o = memtransfer_types[0];
    let geps = get_geps(&mut builder, &cx, ty, ty2, a1, a2, a4);
    generate_mapper_call(&mut builder, &cx, geps, o, begin_mapper_decl, fn_ty, num_args, s_ident_t);
    let values = KernelArgsTy::new(&cx, num_args, memtransfer_types, geps);

    // Step 3)
    // Here we fill the KernelArgsTy, see the documentation above
    for (i, value) in values.iter().enumerate() {
        let ptr = builder.inbounds_gep(tgt_kernel_decl, a5, &[i32_0, cx.get_const_i32(i as u64)]);
        builder.store(value.1, ptr, value.0);
    }

    let args = vec![
        s_ident_t,
        // FIXME(offload) give users a way to select which GPU to use.
        cx.get_const_i64(u64::MAX), // MAX == -1.
        // FIXME(offload): Don't hardcode the numbers of threads in the future.
        cx.get_const_i32(2097152),
        cx.get_const_i32(256),
        region_ids[0],
        a5,
    ];
    builder.call(tgt_target_kernel_ty, tgt_decl, &args, None);
    // %41 = call i32 @__tgt_target_kernel(ptr @1, i64 -1, i32 2097152, i32 256, ptr @.kernel_1.region_id, ptr %kernel_args)

    // Step 4)
    let geps = get_geps(&mut builder, &cx, ty, ty2, a1, a2, a4);
    generate_mapper_call(&mut builder, &cx, geps, o, end_mapper_decl, fn_ty, num_args, s_ident_t);

    builder.call(mapper_fn_ty, unregister_lib_decl, &[tgt_bin_desc_alloca], None);

    drop(builder);
}
