//! Codegen vtables and vtable accesses.
//!
//! See librustc_codegen_llvm/meth.rs for reference
// FIXME dedup this logic between miri, cg_llvm and cg_clif

use crate::prelude::*;

const DROP_FN_INDEX: usize = 0;
const SIZE_INDEX: usize = 1;
const ALIGN_INDEX: usize = 2;

fn vtable_memflags() -> MemFlags {
    let mut flags = MemFlags::trusted(); // A vtable access is always aligned and will never trap.
    flags.set_readonly(); // A vtable is always read-only.
    flags
}

pub(crate) fn drop_fn_of_obj(fx: &mut FunctionCx<'_, '_, impl Module>, vtable: Value) -> Value {
    let usize_size = fx.layout_of(fx.tcx.types.usize).size.bytes() as usize;
    fx.bcx.ins().load(
        pointer_ty(fx.tcx),
        vtable_memflags(),
        vtable,
        (DROP_FN_INDEX * usize_size) as i32,
    )
}

pub(crate) fn size_of_obj(fx: &mut FunctionCx<'_, '_, impl Module>, vtable: Value) -> Value {
    let usize_size = fx.layout_of(fx.tcx.types.usize).size.bytes() as usize;
    fx.bcx.ins().load(
        pointer_ty(fx.tcx),
        vtable_memflags(),
        vtable,
        (SIZE_INDEX * usize_size) as i32,
    )
}

pub(crate) fn min_align_of_obj(fx: &mut FunctionCx<'_, '_, impl Module>, vtable: Value) -> Value {
    let usize_size = fx.layout_of(fx.tcx.types.usize).size.bytes() as usize;
    fx.bcx.ins().load(
        pointer_ty(fx.tcx),
        vtable_memflags(),
        vtable,
        (ALIGN_INDEX * usize_size) as i32,
    )
}

pub(crate) fn get_ptr_and_method_ref<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
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
        ((idx + 3) * usize_size as usize) as i32,
    );
    (ptr, func_ref)
}

pub(crate) fn get_vtable<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    layout: TyAndLayout<'tcx>,
    trait_ref: Option<ty::PolyExistentialTraitRef<'tcx>>,
) -> Value {
    let data_id = if let Some(data_id) = fx.cx.vtables.get(&(layout.ty, trait_ref)) {
        *data_id
    } else {
        let data_id = build_vtable(fx, layout, trait_ref);
        fx.cx.vtables.insert((layout.ty, trait_ref), data_id);
        data_id
    };

    let local_data_id = fx.cx.module.declare_data_in_func(data_id, &mut fx.bcx.func);
    fx.bcx.ins().global_value(fx.pointer_type, local_data_id)
}

fn build_vtable<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    layout: TyAndLayout<'tcx>,
    trait_ref: Option<ty::PolyExistentialTraitRef<'tcx>>,
) -> DataId {
    let tcx = fx.tcx;
    let usize_size = fx.layout_of(fx.tcx.types.usize).size.bytes() as usize;

    let drop_in_place_fn = import_function(
        tcx,
        &mut fx.cx.module,
        Instance::resolve_drop_in_place(tcx, layout.ty).polymorphize(fx.tcx),
    );

    let mut components: Vec<_> = vec![Some(drop_in_place_fn), None, None];

    let methods_root;
    let methods = if let Some(trait_ref) = trait_ref {
        methods_root = tcx.vtable_methods(trait_ref.with_self_ty(tcx, layout.ty));
        methods_root.iter()
    } else {
        (&[]).iter()
    };
    let methods = methods.cloned().map(|opt_mth| {
        opt_mth.map(|(def_id, substs)| {
            import_function(
                tcx,
                &mut fx.cx.module,
                Instance::resolve_for_vtable(tcx, ParamEnv::reveal_all(), def_id, substs)
                    .unwrap()
                    .polymorphize(fx.tcx),
            )
        })
    });
    components.extend(methods);

    let mut data_ctx = DataContext::new();
    let mut data = ::std::iter::repeat(0u8)
        .take(components.len() * usize_size)
        .collect::<Vec<u8>>()
        .into_boxed_slice();

    write_usize(fx.tcx, &mut data, SIZE_INDEX, layout.size.bytes());
    write_usize(fx.tcx, &mut data, ALIGN_INDEX, layout.align.abi.bytes());
    data_ctx.define(data);

    for (i, component) in components.into_iter().enumerate() {
        if let Some(func_id) = component {
            let func_ref = fx.cx.module.declare_func_in_data(func_id, &mut data_ctx);
            data_ctx.write_function_addr((i * usize_size) as u32, func_ref);
        }
    }

    data_ctx.set_align(fx.tcx.data_layout.pointer_align.pref.bytes());

    let data_id = fx
        .cx
        .module
        .declare_data(
            &format!(
                "__vtable.{}.for.{:?}.{}",
                trait_ref
                    .as_ref()
                    .map(|trait_ref| format!("{:?}", trait_ref.skip_binder()).into())
                    .unwrap_or(std::borrow::Cow::Borrowed("???")),
                layout.ty,
                fx.cx.vtables.len(),
            ),
            Linkage::Local,
            false,
            false,
        )
        .unwrap();

    // FIXME don't duplicate definitions in lazy jit mode
    let _ = fx.cx.module.define_data(data_id, &data_ctx);

    data_id
}

fn write_usize(tcx: TyCtxt<'_>, buf: &mut [u8], idx: usize, num: u64) {
    let pointer_size = tcx
        .layout_of(ParamEnv::reveal_all().and(tcx.types.usize))
        .unwrap()
        .size
        .bytes() as usize;
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
