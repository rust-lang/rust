use prelude::*;
use rustc::mir::interpret::{GlobalId, AllocId, read_target_uint};
use rustc_mir::interpret::{CompileTimeEvaluator, Memory, MemoryKind};
use cranelift_module::*;

pub fn trans_constant<'a, 'tcx: 'a>(fx: &mut FunctionCx<'a, 'tcx>, const_: &Constant<'tcx>) -> CValue<'tcx> {
    let value = match const_.literal {
        Literal::Value { value } => value,
        Literal::Promoted { index } => fx
            .tcx
            .const_eval(ParamEnv::reveal_all().and(GlobalId {
                instance: fx.instance,
                promoted: Some(index),
            }))
            .unwrap(),
    };
    fx.tcx.sess.warn(&format!("const: {:?}", value));

    let ty = fx.monomorphize(&const_.ty);
    let layout = fx.layout_of(ty);
    match ty.sty {
        TypeVariants::TyBool => {
            let bits = value.to_scalar().unwrap().to_bits(layout.size).unwrap();
            CValue::const_val(fx, ty, bits as u64 as i64)
        }
        TypeVariants::TyUint(_) => {
            let bits = value.to_scalar().unwrap().to_bits(layout.size).unwrap();
            CValue::const_val(fx, ty, bits as u64 as i64)
        }
        TypeVariants::TyInt(_) => {
            let bits = value.to_scalar().unwrap().to_bits(layout.size).unwrap();
            CValue::const_val(fx, ty, bits as i128 as i64)
        }
        TypeVariants::TyFnDef(def_id, substs) => {
            let func_ref = fx.get_function_ref(Instance::new(def_id, substs));
            CValue::Func(func_ref, layout)
        }
        _ => {
            if true {
                // TODO: cranelift-module api seems to be used wrong,
                // thus causing panics for some consts, so this disables it
                return CValue::ByRef(fx.bcx.ins().iconst(types::I64, 0), layout);
            }
            let mut memory = Memory::<CompileTimeEvaluator>::new(fx.tcx.at(DUMMY_SP), ());
            let alloc = fx.tcx.const_value_to_allocation(value);
            //println!("const value: {:?} allocation: {:?}", value, alloc);
            let alloc_id = memory.allocate_value(alloc.clone(), MemoryKind::Stack).unwrap();
            let data_id = get_global_for_alloc_id(fx, &memory, alloc_id);
            let local_data_id = fx.module.declare_data_in_func(data_id, &mut fx.bcx.func);
            // TODO: does global_value return a ptr of a val?
            let global_ptr = fx.bcx.ins().global_value(types::I64, local_data_id);
            CValue::ByRef(global_ptr, layout)
        }
    }
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
