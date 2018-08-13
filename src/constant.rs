use cranelift_module::*;
use crate::prelude::*;
use rustc::mir::interpret::{read_target_uint, AllocId, ConstValue, GlobalId};
use rustc::ty::Const;
use rustc_mir::interpret::{CompileTimeEvaluator, Memory};

#[derive(Default)]
pub struct ConstantCx {
    todo_allocs: HashSet<AllocId>,
    done: HashSet<DataId>,
}

impl ConstantCx {
    pub fn finalize<'a, 'tcx: 'a, B: Backend>(mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>, module: &mut Module<B>) {
        println!("todo allocs: {:?}", self.todo_allocs);
        define_all_allocs(tcx, module, &mut self);
        println!("done {:?}", self.done);
        for data_id in self.done.drain() {
            module.finalize_data(data_id);
        }
    }
}

pub fn codegen_static<'a, 'tcx: 'a, B: Backend>(cx: &mut CodegenCx<'a, 'tcx, B>, def_id: DefId) {
    unimpl!("static mono item {:?}", def_id);
}

pub fn codegen_static_ref<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx>,
    static_: &Static<'tcx>,
) -> CPlace<'tcx> {
    unimpl!("static place {:?} ty {:?}", static_.def_id, static_.ty);
}

pub fn trans_promoted<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx>,
    promoted: Promoted,
) -> CPlace<'tcx> {
    let const_ = fx
        .tcx
        .const_eval(ParamEnv::reveal_all().and(GlobalId {
            instance: fx.instance,
            promoted: Some(promoted),
        })).unwrap();

    let const_ = force_eval_const(fx, const_);
    trans_const_place(fx, const_)
}

pub fn trans_constant<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx>,
    constant: &Constant<'tcx>,
) -> CValue<'tcx> {
    let const_ = fx.monomorphize(&constant.literal);
    let const_ = force_eval_const(fx, const_);
    trans_const_value(fx, const_)
}

fn force_eval_const<'a, 'tcx: 'a>(
    fx: &FunctionCx<'a, 'tcx>,
    const_: &'tcx Const<'tcx>,
) -> &'tcx Const<'tcx> {
    match const_.val {
        ConstValue::Unevaluated(def_id, ref substs) => {
            let param_env = ParamEnv::reveal_all();
            let instance = Instance::resolve(fx.tcx, param_env, def_id, substs).unwrap();
            let cid = GlobalId {
                instance,
                promoted: None,
            };
            fx.tcx.const_eval(param_env.and(cid)).unwrap()
        }
        _ => const_,
    }
}

fn trans_const_value<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx>,
    const_: &'tcx Const<'tcx>,
) -> CValue<'tcx> {
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
        _ => trans_const_place(fx, const_).to_cvalue(fx),
    }
}

fn trans_const_place<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx>,
    const_: &'tcx Const<'tcx>,
) -> CPlace<'tcx> {
    let ty = fx.monomorphize(&const_.ty);
    let layout = fx.layout_of(ty);

    let alloc = fx.tcx.const_value_to_allocation(const_);
    //println!("const value: {:?} allocation: {:?}", value, alloc);
    let alloc_id = fx.tcx.alloc_map.lock().allocate(alloc);
    let data_id = get_global_for_alloc_id(
        fx.module,
        fx.constants,
        alloc_id,
    );
    let local_data_id = fx.module.declare_data_in_func(data_id, &mut fx.bcx.func);
    // TODO: does global_value return a ptr of a val?
    let global_ptr = fx.bcx.ins().global_value(types::I64, local_data_id);
    CPlace::Addr(global_ptr, layout)
}

// If ret.1 is true, then the global didn't exist before
fn define_global_for_alloc_id<'a, 'tcx: 'a, B: Backend>(
    module: &mut Module<B>,
    cx: &mut ConstantCx,
    alloc_id: AllocId,
) -> DataId {
    module
        .declare_data(&alloc_id.0.to_string(), Linkage::Local, false)
        .unwrap()
}

fn get_global_for_alloc_id<'a, 'tcx: 'a, B: Backend + 'a>(
    module: &mut Module<B>,
    cx: &mut ConstantCx,
    alloc_id: AllocId,
) -> DataId {
    cx.todo_allocs.insert(alloc_id);
    let data_id = define_global_for_alloc_id(module, cx, alloc_id);
    data_id
}

fn define_all_allocs<'a, 'tcx: 'a, B: Backend + 'a> (
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    module: &mut Module<B>,
    cx: &mut ConstantCx,
) {
    let memory = Memory::<CompileTimeEvaluator>::new(tcx.at(DUMMY_SP), ());

    loop {
        let alloc_id = {
            if let Some(alloc_id) = cx.todo_allocs.iter().next().map(|alloc_id| *alloc_id) {
                cx.todo_allocs.remove(&alloc_id);
                alloc_id
            } else {
                break;
            }
        };

        let data_id = define_global_for_alloc_id(module, cx, alloc_id);
        println!("alloc_id {} data_id {}", alloc_id, data_id);
        if cx.done.contains(&data_id) {
            continue;
        }

        let alloc = memory.get(alloc_id).unwrap();
        //let alloc = tcx.alloc_map.lock().get(alloc_id).unwrap();
        let mut data_ctx = DataContext::new();

        data_ctx.define(
            alloc.bytes.to_vec().into_boxed_slice(),
            Writability::Readonly,
        );

        for &(offset, reloc) in alloc.relocations.iter() {
            cx.todo_allocs.insert(reloc);
            let data_id = define_global_for_alloc_id(module, cx, reloc);

            let reloc_offset = {
                let endianness = memory.endianness();
                let offset = offset.bytes() as usize;
                let ptr_size = tcx.data_layout.pointer_size;
                let bytes = &alloc.bytes[offset..offset + ptr_size.bytes() as usize];
                read_target_uint(endianness, bytes).unwrap()
            };

            let global_value = module.declare_data_in_data(data_id, &mut data_ctx);
            data_ctx.write_data_addr(reloc_offset as u32, global_value, 0);
        }

        module.define_data(data_id, &data_ctx).unwrap();
        cx.done.insert(data_id);
    }

    assert!(cx.todo_allocs.is_empty(), "{:?}", cx.todo_allocs);
}
