//! See librustc_codegen_llvm/meth.rs for reference

use crate::prelude::*;

const DROP_FN_INDEX: usize = 0;
const SIZE_INDEX: usize = 1;
const ALIGN_INDEX: usize = 2;

pub fn drop_fn_of_obj<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    vtable: Value,
) -> Value {
    let usize_size = fx.layout_of(fx.tcx.types.usize).size.bytes() as usize;
    fx.bcx.ins().load(
        pointer_ty(fx.tcx),
        MemFlags::new(),
        vtable,
        (DROP_FN_INDEX * usize_size) as i32,
    )
}

pub fn size_of_obj<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    vtable: Value,
) -> Value {
    let usize_size = fx.layout_of(fx.tcx.types.usize).size.bytes() as usize;
    fx.bcx.ins().load(
        pointer_ty(fx.tcx),
        MemFlags::new(),
        vtable,
        (SIZE_INDEX * usize_size) as i32,
    )
}

pub fn min_align_of_obj<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    vtable: Value,
) -> Value {
    let usize_size = fx.layout_of(fx.tcx.types.usize).size.bytes() as usize;
    fx.bcx.ins().load(
        pointer_ty(fx.tcx),
        MemFlags::new(),
        vtable,
        (ALIGN_INDEX * usize_size) as i32,
    )
}

pub fn get_ptr_and_method_ref<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    arg: CValue<'tcx>,
    idx: usize,
) -> (Value, Value) {
    let (ptr, vtable) = arg.load_scalar_pair(fx);
    let usize_size = fx.layout_of(fx.tcx.types.usize).size.bytes();
    let func_ref = fx.bcx.ins().load(
        pointer_ty(fx.tcx),
        MemFlags::new(),
        vtable,
        ((idx + 3) * usize_size as usize) as i32,
    );
    (ptr, func_ref)
}

pub fn get_vtable<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    ty: Ty<'tcx>,
    trait_ref: Option<ty::PolyExistentialTraitRef<'tcx>>,
) -> Value {
    let data_id = if let Some(data_id) = fx.caches.vtables.get(&(ty, trait_ref)) {
        *data_id
    } else {
        let data_id = build_vtable(fx, ty, trait_ref);
        fx.caches.vtables.insert((ty, trait_ref), data_id);
        data_id
    };

    let local_data_id = fx.module.declare_data_in_func(data_id, &mut fx.bcx.func);
    fx.bcx.ins().global_value(fx.pointer_type, local_data_id)
}

fn build_vtable<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    ty: Ty<'tcx>,
    trait_ref: Option<ty::PolyExistentialTraitRef<'tcx>>,
) -> DataId {
    let tcx = fx.tcx;
    let usize_size = fx.layout_of(fx.tcx.types.usize).size.bytes() as usize;

    let drop_in_place_fn = import_function(
        tcx,
        fx.module,
        Instance::resolve_drop_in_place(tcx, ty),
    );

    let mut components: Vec<_> = vec![Some(drop_in_place_fn), None, None];

    let methods_root;
    let methods = if let Some(trait_ref) = trait_ref {
        methods_root = tcx.vtable_methods(trait_ref.with_self_ty(tcx, ty));
        methods_root.iter()
    } else {
        (&[]).iter()
    };
    let methods = methods.cloned().map(|opt_mth| {
        opt_mth.map_or(None, |(def_id, substs)| {
            Some(import_function(
                tcx,
                fx.module,
                Instance::resolve_for_vtable(tcx, ParamEnv::reveal_all(), def_id, substs).unwrap(),
            ))
        })
    });
    components.extend(methods);

    let mut data_ctx = DataContext::new();
    let mut data = ::std::iter::repeat(0u8)
        .take(components.len() * usize_size)
        .collect::<Vec<u8>>()
        .into_boxed_slice();

    let layout = tcx.layout_of(ParamEnv::reveal_all().and(ty)).unwrap();
    write_usize(fx.tcx, &mut data, SIZE_INDEX, layout.size.bytes());
    write_usize(fx.tcx, &mut data, ALIGN_INDEX, layout.align.abi.bytes());
    data_ctx.define(data);

    for (i, component) in components.into_iter().enumerate() {
        if let Some(func_id) = component {
            let func_ref = fx.module.declare_func_in_data(func_id, &mut data_ctx);
            data_ctx.write_function_addr((i * usize_size) as u32, func_ref);
        }
    }

    let data_id = fx
        .module
        .declare_data(
            &format!("vtable.{:?}.for.{:?}", trait_ref, ty),
            Linkage::Local,
            false,
            Some(fx.tcx.data_layout.pointer_align.pref.bytes().try_into().unwrap())
        )
        .unwrap();

    match fx.module.define_data(data_id, &data_ctx) {
        Ok(()) | Err(cranelift_module::ModuleError::DuplicateDefinition(_)) => {}
        err => err.unwrap(),
    }

    data_id
}

fn write_usize(tcx: TyCtxt, buf: &mut [u8], idx: usize, num: u64) {
    use byteorder::{BigEndian, LittleEndian, WriteBytesExt};

    let usize_size = tcx
        .layout_of(ParamEnv::reveal_all().and(tcx.types.usize))
        .unwrap()
        .size
        .bytes() as usize;
    let mut target = &mut buf[idx * usize_size..(idx + 1) * usize_size];

    match tcx.data_layout.endian {
        layout::Endian::Little => target.write_uint::<LittleEndian>(num, usize_size),
        layout::Endian::Big => target.write_uint::<BigEndian>(num, usize_size),
    }
    .unwrap()
}
