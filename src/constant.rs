use crate::prelude::*;
use rustc::ty::Const;
use rustc::mir::interpret::{ConstValue, GlobalId, AllocId, read_target_uint};
use rustc_mir::interpret::{CompileTimeEvaluator, Memory, MemoryKind};
use cranelift_module::*;

pub fn trans_promoted<'a, 'tcx: 'a>(fx: &mut FunctionCx<'a, 'tcx>, promoted: Promoted) -> CPlace<'tcx> {
    let const_ = fx
        .tcx
        .const_eval(ParamEnv::reveal_all().and(GlobalId {
            instance: fx.instance,
            promoted: Some(promoted),
        }))
        .unwrap();

    let const_ = force_eval_const(fx, const_);
    trans_const_place(fx, const_)
}

pub fn trans_constant<'a, 'tcx: 'a>(fx: &mut FunctionCx<'a, 'tcx>, constant: &Constant<'tcx>) -> CValue<'tcx> {
    let const_ = fx.monomorphize(&constant.literal);
    let const_ = force_eval_const(fx, const_);
    trans_const_value(fx, const_)
}

fn force_eval_const<'a, 'tcx: 'a>(fx: &FunctionCx<'a, 'tcx>, const_: &'tcx Const<'tcx>) -> &'tcx Const<'tcx> {
    match const_.val {
        ConstValue::Unevaluated(def_id, ref substs) => {
            let param_env = ParamEnv::reveal_all();
            let instance = Instance::resolve(fx.tcx, param_env, def_id, substs).unwrap();
            let cid = GlobalId {
                instance,
                promoted: None,
            };
            fx.tcx.const_eval(param_env.and(cid)).unwrap()
        },
        _ => const_,
    }
}

fn trans_const_value<'a, 'tcx: 'a>(fx: &mut FunctionCx<'a, 'tcx>, const_: &'tcx Const<'tcx>) -> CValue<'tcx> {
    let ty = fx.monomorphize(&const_.ty);
    let layout = fx.layout_of(ty);
    match ty.sty {
        TypeVariants::TyBool => {
            let bits = const_.val.to_bits(layout.size).unwrap();
            CValue::const_val(fx, ty, bits as u64 as i64)
        }
        TypeVariants::TyUint(_) => {
            let bits = const_.val.to_bits(layout.size).unwrap();
            CValue::const_val(fx, ty, bits as u64 as i64)
        }
        TypeVariants::TyInt(_) => {
            let bits = const_.val.to_bits(layout.size).unwrap();
            CValue::const_val(fx, ty, bits as i128 as i64)
        }
        TypeVariants::TyFnDef(def_id, substs) => {
            let func_ref = fx.get_function_ref(Instance::new(def_id, substs));
            CValue::Func(func_ref, layout)
        }
        _ => {
            trans_const_place(fx, const_).to_cvalue(fx)
        }
    }
}

fn trans_const_place<'a, 'tcx: 'a>(fx: &mut FunctionCx<'a, 'tcx>, const_: &'tcx Const<'tcx>) -> CPlace<'tcx> {
    let ty = fx.monomorphize(&const_.ty);
    let layout = fx.layout_of(ty);
    if true {
        // TODO: cranelift-module api seems to be used wrong,
        // thus causing panics for some consts, so this disables it
        return CPlace::Addr(fx.bcx.ins().iconst(types::I64, 0), layout);
    }
    let mut memory = Memory::<CompileTimeEvaluator>::new(fx.tcx.at(DUMMY_SP), ());
    let alloc = fx.tcx.const_value_to_allocation(const_);
    //println!("const value: {:?} allocation: {:?}", value, alloc);
    let alloc_id = memory.allocate_value(alloc.clone(), MemoryKind::Stack).unwrap();
    let data_id = get_global_for_alloc_id(fx, &memory, alloc_id);
    let local_data_id = fx.module.declare_data_in_func(data_id, &mut fx.bcx.func);
    // TODO: does global_value return a ptr of a val?
    let global_ptr = fx.bcx.ins().global_value(types::I64, local_data_id);
    CPlace::Addr(global_ptr, layout)
}

// If ret.1 is true, then the global didn't exist before
fn define_global_for_alloc_id(fx: &mut FunctionCx, alloc_id: AllocId, todo: &mut HashMap<AllocId, DataId>) -> (DataId, bool) {
    use std::collections::hash_map::Entry;
    match fx.constants.entry(alloc_id) {
        Entry::Occupied(mut occ) => {
            (*occ.get_mut(), false)
        }
        Entry::Vacant(vac) => {
            let data_id = fx.module.declare_data(&alloc_id.0.to_string(), Linkage::Local, false).unwrap();
            todo.insert(alloc_id, data_id);
            vac.insert(data_id);
            (data_id, true)
        }
    }
}

fn get_global_for_alloc_id(fx: &mut FunctionCx, memory: &Memory<CompileTimeEvaluator>, alloc_id: AllocId) -> DataId {
    if let Some(data_id) = fx.constants.get(&alloc_id) {
        return *data_id;
    }

    let mut todo = HashMap::new();
    let mut done = HashSet::new();
    define_global_for_alloc_id(fx, alloc_id, &mut todo);

    while let Some((alloc_id, data_id)) = { let next = todo.drain().next(); next } {
        println!("cur: {:?}:{:?} todo: {:?} done: {:?}", alloc_id, data_id, todo, done);

        let alloc = memory.get(alloc_id).unwrap();
        let mut data_ctx = DataContext::new();

        data_ctx.define(alloc.bytes.to_vec().into_boxed_slice(), Writability::Readonly);

        for &(offset, reloc) in alloc.relocations.iter() {
            let data_id = define_global_for_alloc_id(fx, reloc, &mut todo).0;

            let reloc_offset = {
                let endianness = memory.endianness();
                let offset = offset.bytes() as usize;
                let ptr_size = fx.tcx.data_layout.pointer_size;
                let bytes = &alloc.bytes[offset..offset + ptr_size.bytes() as usize];
                read_target_uint(endianness, bytes).unwrap()
            };

            // TODO: is this a correct usage of the api
            let global_value = fx.module.declare_data_in_data(data_id, &mut data_ctx);
            data_ctx.write_data_addr(reloc_offset as u32, global_value, 0);
        }

        fx.module.define_data(data_id, &data_ctx).unwrap();
        done.insert(data_id);
    }
    for data_id in done.drain() {
        fx.module.finalize_data(data_id);
    }
    *fx.constants.get(&alloc_id).unwrap()
}
