use crate::SimpleCx;
use crate::builder::Builder;
use crate::builder::gpu_offload::{OffloadGlobals, get_or_create_async_info};
use crate::intrinsic::TransferType;
use crate::llvm;
use crate::llvm::{Type, Value};
use rustc_abi::Align;
use rustc_codegen_ssa::MemFlags;
use rustc_codegen_ssa::common::TypeKind;
use rustc_codegen_ssa::traits::{BaseTypeCodegenMethods, BuilderMethods};
use rustc_middle::bug;
use rustc_middle::ty::offload_meta::{OffloadMetadata, OffloadSize};

pub(crate) fn scalar_width<'ll>(cx: &'ll SimpleCx<'_>, ty: &'ll Type) -> u64 {
    match cx.type_kind(ty) {
        TypeKind::Half
        | TypeKind::Float
        | TypeKind::Double
        | TypeKind::X86_FP80
        | TypeKind::FP128
        | TypeKind::PPC_FP128 => cx.float_width(ty) as u64,
        TypeKind::Integer => cx.int_width(ty),
        other => bug!("scalar_width was called on a non scalar type {other:?}"),
    }
}

fn get_runtime_size<'ll, 'tcx>(
    builder: &mut Builder<'_, 'll, 'tcx>,
    args: &[&'ll Value],
    index: usize,
    meta: &OffloadMetadata,
) -> &'ll Value {
    match meta.payload_size {
        OffloadSize::Slice { element_size } => {
            let length_idx = index + 1;
            let length = args[length_idx];
            let length_i64 = builder.intcast(length, builder.cx.type_i64(), false);
            builder.mul(length_i64, builder.cx.get_const_i64(element_size))
        }
        _ => bug!("unexpected offload size {:?}", meta.payload_size),
    }
}

// For now we have a very simplistic indexing scheme into our
// offload_{baseptrs,ptrs,sizes}. We will probably improve this along with our gpu frontend pr.
pub(crate) fn get_geps<'ll, 'tcx>(
    builder: &mut Builder<'_, 'll, 'tcx>,
    ty: &'ll Type,
    ty2: &'ll Type,
    a1: &'ll Value,
    a2: &'ll Value,
    a4: &'ll Value,
    is_dynamic: bool,
) -> [&'ll Value; 3] {
    let cx = builder.cx;
    let i32_0 = cx.get_const_i32(0);

    let gep1 = builder.inbounds_gep(ty, a1, &[i32_0, i32_0]);
    let gep2 = builder.inbounds_gep(ty, a2, &[i32_0, i32_0]);
    let gep3 = if is_dynamic { builder.inbounds_gep(ty2, a4, &[i32_0, i32_0]) } else { a4 };
    [gep1, gep2, gep3]
}

fn synchronize_async_info<'ll, 'tcx>(
    builder: &mut Builder<'_, 'll, 'tcx>,
    offload_globals: &OffloadGlobals<'ll>,
) {
    let async_info = get_or_create_async_info(builder, offload_globals);

    builder.call(
        offload_globals.async_info_synchronize_ty,
        None,
        None,
        offload_globals.async_info_synchronize,
        &[async_info],
        None,
        None,
    );
}

pub(crate) fn generate_mapper_call<'ll, 'tcx>(
    builder: &mut Builder<'_, 'll, 'tcx>,
    geps: [&'ll Value; 3],
    offload_globals: &OffloadGlobals<'ll>,
    o_type: &'ll Value,
    fn_to_call: &'ll Value,
    fn_ty: &'ll Type,
    num_args: u64,
    s_ident_t: &'ll Value,
    transfer: TransferType,
) {
    let cx = builder.cx;
    let nullptr = cx.const_null(cx.type_ptr());
    let i64_max = cx.get_const_i64(u64::MAX);
    let num_args = cx.get_const_i32(num_args);
    let mut args =
        vec![s_ident_t, i64_max, num_args, geps[0], geps[1], geps[2], o_type, nullptr, nullptr];
    if matches!(transfer, TransferType::NowaitBegin) {
        let i32_0 = cx.get_const_i32(0);
        args.append(&mut vec![i32_0, nullptr, i32_0, nullptr]);
    }
    if matches!(transfer, TransferType::End) {
        synchronize_async_info(builder, offload_globals);
        let a = offload_globals.taskwait;
        let b = offload_globals.taskwait_ty;
        let c = offload_globals.threadnum;
        let d = offload_globals.threadnum_ty;
        dbg!(&c);
        dbg!(&d);
        dbg!("first");
        let tid = builder.call(d, None, None, c, &vec![s_ident_t], None, None);
        let args2 = vec![s_ident_t, tid];
        dbg!(&a);
        dbg!(&b);
        dbg!("second");
        builder.call(b, None, None, a, &args2, None, None);
    }
    builder.call(fn_ty, None, None, fn_to_call, &args, None, None);
}

pub(crate) fn preper_datatransfers<'ll, 'tcx>(
    builder: &mut Builder<'_, 'll, 'tcx>,
    args: &[&'ll Value],
    types: &[&Type],
    offload_sizes: &'ll Value,
    metadata: &[OffloadMetadata],
    has_dynamic: bool,
) -> (&'ll Type, &'ll Type, &'ll Value, &'ll Value, &'ll Value) {
    let cx = builder.cx;
    let num_args = types.len() as u64;
    let bb = builder.llbb();

    // Step 0)
    unsafe {
        llvm::LLVMRustPositionBuilderPastAllocas(&builder.llbuilder, builder.llfn());
    }

    let ty = cx.type_array(cx.type_ptr(), num_args);
    // Baseptr are just the input pointer to the kernel, stored in a local alloca
    let a1 = builder.direct_alloca(ty, Align::EIGHT, ".offload_baseptrs");
    // Ptrs are the result of a gep into the baseptr, at least for our trivial types.
    let a2 = builder.direct_alloca(ty, Align::EIGHT, ".offload_ptrs");
    // These represent the sizes in bytes, e.g. the entry for `&[f64; 16]` will be 8*16.
    let ty2 = cx.type_array(cx.type_i64(), num_args);

    let a4 = if has_dynamic {
        let alloc = builder.direct_alloca(ty2, Align::EIGHT, ".offload_sizes");

        builder.memcpy(
            alloc,
            Align::EIGHT,
            offload_sizes,
            Align::EIGHT,
            cx.get_const_i64(8 * args.len() as u64),
            MemFlags::empty(),
            None,
        );

        alloc
    } else {
        offload_sizes
    };

    // Step 1)
    unsafe {
        llvm::LLVMPositionBuilderAtEnd(&builder.llbuilder, bb);
    }

    // Now we allocate once per function param, a copy to be passed to one of our maps.
    let mut vals = vec![];
    let mut geps = vec![];
    let i32_0 = cx.get_const_i32(0);
    for &v in args {
        let ty = cx.val_ty(v);
        let ty_kind = cx.type_kind(ty);
        let (base_val, gep_base) = match ty_kind {
            TypeKind::Pointer => (v, v),
            TypeKind::Half | TypeKind::Float | TypeKind::Double | TypeKind::Integer => {
                // FIXME(Sa4dUs): check for `f128` support, latest NVIDIA cards support it
                let num_bits = scalar_width(cx, ty);

                let bb = builder.llbb();
                unsafe {
                    llvm::LLVMRustPositionBuilderPastAllocas(builder.llbuilder, builder.llfn());
                }
                let addr = builder.direct_alloca(cx.type_i64(), Align::EIGHT, "addr");
                unsafe {
                    llvm::LLVMPositionBuilderAtEnd(builder.llbuilder, bb);
                }

                let cast = builder.bitcast(v, cx.type_ix(num_bits));
                let value = builder.zext(cast, cx.type_i64());
                builder.store(value, addr, Align::EIGHT);
                (value, addr)
            }
            other => bug!("offload does not support {other:?}"),
        };

        let gep = builder.inbounds_gep(cx.type_f32(), gep_base, &[i32_0]);

        vals.push(base_val);
        geps.push(gep);
    }

    for i in 0..num_args {
        let idx = cx.get_const_i32(i);
        let gep1 = builder.inbounds_gep(ty, a1, &[i32_0, idx]);
        builder.store(vals[i as usize], gep1, Align::EIGHT);
        let gep2 = builder.inbounds_gep(ty, a2, &[i32_0, idx]);
        builder.store(geps[i as usize], gep2, Align::EIGHT);

        if !matches!(metadata[i as usize].payload_size, OffloadSize::Static(_)) {
            let gep3 = builder.inbounds_gep(ty2, a4, &[i32_0, idx]);
            let size_val = get_runtime_size(builder, args, i as usize, &metadata[i as usize]);
            builder.store(size_val, gep3, Align::EIGHT);
        }
    }
    (ty, ty2, a1, a2, a4)
}
