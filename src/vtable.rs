//! Codegen vtables and vtable accesses.
//!
//! See `rustc_codegen_ssa/src/meth.rs` for reference.
// FIXME dedup this logic between miri, cg_llvm and cg_clif

use crate::prelude::*;
use ty::VtblEntry;

fn vtable_memflags() -> MemFlags {
    let mut flags = MemFlags::trusted(); // A vtable access is always aligned and will never trap.
    flags.set_readonly(); // A vtable is always read-only.
    flags
}

pub(crate) fn drop_fn_of_obj(fx: &mut FunctionCx<'_, '_, '_>, vtable: Value) -> Value {
    let usize_size = fx.layout_of(fx.tcx.types.usize).size.bytes() as usize;
    fx.bcx.ins().load(
        pointer_ty(fx.tcx),
        vtable_memflags(),
        vtable,
        (ty::COMMON_VTABLE_ENTRIES_DROPINPLACE * usize_size) as i32,
    )
}

pub(crate) fn size_of_obj(fx: &mut FunctionCx<'_, '_, '_>, vtable: Value) -> Value {
    let usize_size = fx.layout_of(fx.tcx.types.usize).size.bytes() as usize;
    fx.bcx.ins().load(
        pointer_ty(fx.tcx),
        vtable_memflags(),
        vtable,
        (ty::COMMON_VTABLE_ENTRIES_SIZE * usize_size) as i32,
    )
}

pub(crate) fn min_align_of_obj(fx: &mut FunctionCx<'_, '_, '_>, vtable: Value) -> Value {
    let usize_size = fx.layout_of(fx.tcx.types.usize).size.bytes() as usize;
    fx.bcx.ins().load(
        pointer_ty(fx.tcx),
        vtable_memflags(),
        vtable,
        (ty::COMMON_VTABLE_ENTRIES_SIZE * usize_size) as i32,
    )
}

pub(crate) fn get_ptr_and_method_ref<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    arg: CValue<'tcx>,
    idx: usize,
) -> (Value, Value) {
    let (ptr, vtable) = if let Abi::ScalarPair(_, _) = arg.layout().abi {
        arg.load_scalar_pair(fx)
    } else {
        let (ptr, vtable) = arg.try_to_ptr().unwrap();
        (ptr.get_addr(fx), vtable.unwrap())
    };

    let usize_size = fx.layout_of(fx.tcx.types.usize).size.bytes();
    let func_ref = fx.bcx.ins().load(
        pointer_ty(fx.tcx),
        vtable_memflags(),
        vtable,
        (idx * usize_size as usize) as i32,
    );
    (ptr, func_ref)
}

pub(crate) fn get_vtable<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    layout: TyAndLayout<'tcx>,
    trait_ref: Option<ty::PolyExistentialTraitRef<'tcx>>,
) -> Value {
    let data_id = if let Some(data_id) = fx.vtables.get(&(layout.ty, trait_ref)) {
        *data_id
    } else {
        let data_id = build_vtable(fx, layout, trait_ref);
        fx.vtables.insert((layout.ty, trait_ref), data_id);
        data_id
    };

    let local_data_id = fx.module.declare_data_in_func(data_id, &mut fx.bcx.func);
    fx.bcx.ins().global_value(fx.pointer_type, local_data_id)
}

fn build_vtable<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    layout: TyAndLayout<'tcx>,
    trait_ref: Option<ty::PolyExistentialTraitRef<'tcx>>,
) -> DataId {
    let tcx = fx.tcx;
    let usize_size = fx.layout_of(fx.tcx.types.usize).size.bytes() as usize;

    let drop_in_place_fn = import_function(
        tcx,
        fx.module,
        Instance::resolve_drop_in_place(tcx, layout.ty).polymorphize(fx.tcx),
    );

    let vtable_entries = if let Some(trait_ref) = trait_ref {
        tcx.vtable_entries(trait_ref.with_self_ty(tcx, layout.ty))
    } else {
        ty::COMMON_VTABLE_ENTRIES
    };

    let mut data_ctx = DataContext::new();
    let mut data = ::std::iter::repeat(0u8)
        .take(vtable_entries.len() * usize_size)
        .collect::<Vec<u8>>()
        .into_boxed_slice();

    for (idx, entry) in vtable_entries.iter().enumerate() {
        match entry {
            VtblEntry::MetadataSize => {
                write_usize(fx.tcx, &mut data, idx, layout.size.bytes());
            }
            VtblEntry::MetadataAlign => {
                write_usize(fx.tcx, &mut data, idx, layout.align.abi.bytes());
            }
            VtblEntry::MetadataDropInPlace | VtblEntry::Vacant | VtblEntry::Method(_, _) => {}
        }
    }
    data_ctx.define(data);

    for (idx, entry) in vtable_entries.iter().enumerate() {
        match entry {
            VtblEntry::MetadataDropInPlace => {
                let func_ref = fx.module.declare_func_in_data(drop_in_place_fn, &mut data_ctx);
                data_ctx.write_function_addr((idx * usize_size) as u32, func_ref);
            }
            VtblEntry::Method(def_id, substs) => {
                let func_id = import_function(
                    tcx,
                    fx.module,
                    Instance::resolve_for_vtable(tcx, ParamEnv::reveal_all(), *def_id, substs)
                        .unwrap()
                        .polymorphize(fx.tcx),
                );
                let func_ref = fx.module.declare_func_in_data(func_id, &mut data_ctx);
                data_ctx.write_function_addr((idx * usize_size) as u32, func_ref);
            }
            VtblEntry::MetadataSize | VtblEntry::MetadataAlign | VtblEntry::Vacant => {}
        }
    }

    data_ctx.set_align(fx.tcx.data_layout.pointer_align.pref.bytes());

    let data_id = fx.module.declare_anonymous_data(false, false).unwrap();

    fx.module.define_data(data_id, &data_ctx).unwrap();

    data_id
}

fn write_usize(tcx: TyCtxt<'_>, buf: &mut [u8], idx: usize, num: u64) {
    let pointer_size =
        tcx.layout_of(ParamEnv::reveal_all().and(tcx.types.usize)).unwrap().size.bytes() as usize;
    let target = &mut buf[idx * pointer_size..(idx + 1) * pointer_size];

    match tcx.data_layout.endian {
        rustc_target::abi::Endian::Little => match pointer_size {
            4 => target.copy_from_slice(&(num as u32).to_le_bytes()),
            8 => target.copy_from_slice(&(num as u64).to_le_bytes()),
            _ => todo!("pointer size {} is not yet supported", pointer_size),
        },
        rustc_target::abi::Endian::Big => match pointer_size {
            4 => target.copy_from_slice(&(num as u32).to_be_bytes()),
            8 => target.copy_from_slice(&(num as u64).to_be_bytes()),
            _ => todo!("pointer size {} is not yet supported", pointer_size),
        },
    }
}
