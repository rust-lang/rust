use std::borrow::Cow;

use rustc::mir::interpret::{
    read_target_uint, AllocId, GlobalAlloc, Allocation, ConstValue, InterpResult, GlobalId, Scalar,
};
use rustc::ty::{Const, layout::Align};
use rustc_mir::interpret::{
    InterpCx, ImmTy, Machine, Memory, MemoryKind, OpTy, PlaceTy, Pointer,
    StackPopCleanup,
};

use cranelift_module::*;

use crate::prelude::*;

#[derive(Default)]
pub struct ConstantCx {
    todo: HashSet<TodoItem>,
    done: HashSet<DataId>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
enum TodoItem {
    Alloc(AllocId),
    Static(DefId),
}

impl ConstantCx {
    pub fn finalize(
        mut self,
        tcx: TyCtxt<'_>,
        module: &mut Module<impl Backend>,
    ) {
        //println!("todo {:?}", self.todo);
        define_all_allocs(tcx, module, &mut self);
        //println!("done {:?}", self.done);
        self.done.clear();
    }
}

pub fn codegen_static(ccx: &mut ConstantCx, def_id: DefId) {
    ccx.todo.insert(TodoItem::Static(def_id));
}

pub fn codegen_static_ref<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    def_id: DefId,
    ty: Ty<'tcx>,
) -> CPlace<'tcx> {
    let linkage = crate::linkage::get_static_ref_linkage(fx.tcx, def_id);
    let data_id = data_id_for_static(fx.tcx, fx.module, def_id, linkage);
    cplace_for_dataid(fx, ty, data_id)
}

pub fn trans_promoted<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    promoted: Promoted,
    dest_ty: Ty<'tcx>,
) -> CPlace<'tcx> {
    match fx
        .tcx
        .const_eval(ParamEnv::reveal_all().and(GlobalId {
            instance: fx.instance,
            promoted: Some(promoted),
        }))
    {
        Ok(const_) => {
            let cplace = trans_const_place(fx, const_);
            debug_assert_eq!(cplace.layout(), fx.layout_of(dest_ty));
            cplace
        }
        Err(_) => {
            crate::trap::trap_unreachable_ret_place(
                fx,
                fx.layout_of(dest_ty),
                "[panic] Tried to get value of promoted value with errored during const eval.",
            )
        }
    }
}

pub fn trans_constant<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    constant: &Constant<'tcx>,
) -> CValue<'tcx> {
    let const_ = force_eval_const(fx, &constant.literal);
    trans_const_value(fx, const_)
}

pub fn force_eval_const<'a, 'tcx: 'a>(
    fx: &FunctionCx<'a, 'tcx, impl Backend>,
    const_: &'tcx Const,
) -> &'tcx Const<'tcx> {
    match const_.val {
        ConstValue::Unevaluated(def_id, ref substs) => {
            let param_env = ParamEnv::reveal_all();
            let substs = fx.monomorphize(substs);
            let instance = Instance::resolve(fx.tcx, param_env, def_id, substs).unwrap();
            let cid = GlobalId {
                instance,
                promoted: None,
            };
            fx.tcx.const_eval(param_env.and(cid)).unwrap()
        }
        _ => fx.monomorphize(&const_),
    }
}

pub fn trans_const_value<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    const_: &'tcx Const<'tcx>,
) -> CValue<'tcx> {
    let ty = fx.monomorphize(&const_.ty);
    let layout = fx.layout_of(ty);
    match ty.sty {
        ty::Bool | ty::Uint(_) => {
            let bits = const_.val.try_to_bits(layout.size).unwrap();
            CValue::const_val(fx, ty, bits)
        }
        ty::Int(_) => {
            let bits = const_.val.try_to_bits(layout.size).unwrap();
            CValue::const_val(fx, ty, rustc::mir::interpret::sign_extend(bits, layout.size))
        }
        ty::FnDef(_def_id, _substs) => CValue::by_ref(
            fx.bcx
                .ins()
                .iconst(fx.pointer_type, fx.pointer_type.bytes() as i64),
            layout,
        ),
        _ => trans_const_place(fx, const_).to_cvalue(fx),
    }
}

fn trans_const_place<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    const_: &'tcx Const<'tcx>,
) -> CPlace<'tcx> {
    // Adapted from https://github.com/rust-lang/rust/pull/53671/files#diff-e0b58bb6712edaa8595ad7237542c958L551
    let result = || -> InterpResult<'tcx, &'tcx Allocation> {
        let mut ecx = InterpCx::new(
            fx.tcx.at(DUMMY_SP),
            ty::ParamEnv::reveal_all(),
            TransPlaceInterpreter,
            (),
        );
        ecx.push_stack_frame(
            fx.instance,
            DUMMY_SP,
            fx.mir,
            None,
            StackPopCleanup::None { cleanup: false },
        )
        .unwrap();
        let op = ecx.eval_operand(
            &Operand::Constant(Box::new(Constant {
                span: DUMMY_SP,
                ty: const_.ty,
                user_ty: None,
                literal: const_,
            })),
            None,
        )?;
        let ptr = ecx.allocate(op.layout, MemoryKind::Stack);
        ecx.copy_op(op, ptr.into())?;
        let alloc = ecx.memory().get(ptr.to_ref().to_scalar()?.to_ptr()?.alloc_id)?;
        Ok(fx.tcx.intern_const_alloc(alloc.clone()))
    };
    let alloc = result().expect("unable to convert ConstValue to Allocation");

    //println!("const value: {:?} allocation: {:?}", value, alloc);
    let alloc_id = fx.tcx.alloc_map.lock().create_memory_alloc(alloc);
    fx.constants.todo.insert(TodoItem::Alloc(alloc_id));
    let data_id = data_id_for_alloc_id(fx.module, alloc_id, alloc.align);
    cplace_for_dataid(fx, const_.ty, data_id)
}

fn data_id_for_alloc_id<B: Backend>(module: &mut Module<B>, alloc_id: AllocId, align: Align) -> DataId {
    module
        .declare_data(&format!("__alloc_{}", alloc_id.0), Linkage::Local, false, Some(align.bytes() as u8))
        .unwrap()
}

fn data_id_for_static(
    tcx: TyCtxt<'_>,
    module: &mut Module<impl Backend>,
    def_id: DefId,
    linkage: Linkage,
) -> DataId {
    let symbol_name = tcx.symbol_name(Instance::mono(tcx, def_id)).as_str();
    let ty = tcx.type_of(def_id);
    let is_mutable = if tcx.is_mutable_static(def_id) {
        true
    } else {
        !ty.is_freeze(tcx, ParamEnv::reveal_all(), DUMMY_SP)
    };
    let align = tcx.layout_of(ParamEnv::reveal_all().and(ty)).unwrap().align.pref.bytes();

    let data_id = module
        .declare_data(&*symbol_name, linkage, is_mutable, Some(align.try_into().unwrap()))
        .unwrap();

    if linkage == Linkage::Preemptible {
        if let ty::RawPtr(_) = tcx.type_of(def_id).sty {
        } else {
            tcx.sess.span_fatal(tcx.def_span(def_id), "must have type `*const T` or `*mut T`")
        }

        let mut data_ctx = DataContext::new();
        let zero_bytes = std::iter::repeat(0)
            .take(pointer_ty(tcx).bytes() as usize)
            .collect::<Vec<u8>>()
            .into_boxed_slice();
        data_ctx.define(zero_bytes);
        match module.define_data(data_id, &data_ctx) {
            // Everytime a weak static is referenced, there will be a zero pointer definition,
            // so duplicate definitions are expected and allowed.
            Err(ModuleError::DuplicateDefinition(_)) => {}
            res => res.unwrap(),
        }
    }

    data_id
}

fn cplace_for_dataid<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    ty: Ty<'tcx>,
    data_id: DataId,
) -> CPlace<'tcx> {
    let local_data_id = fx.module.declare_data_in_func(data_id, &mut fx.bcx.func);
    let global_ptr = fx.bcx.ins().global_value(fx.pointer_type, local_data_id);
    let layout = fx.layout_of(fx.monomorphize(&ty));
    assert!(!layout.is_unsized(), "unsized statics aren't supported");
    CPlace::for_addr(global_ptr, layout)
}

fn define_all_allocs(
    tcx: TyCtxt<'_>,
    module: &mut Module<impl Backend>,
    cx: &mut ConstantCx,
) {
    let memory = Memory::<TransPlaceInterpreter>::new(tcx.at(DUMMY_SP), ());

    while let Some(todo_item) = pop_set(&mut cx.todo) {
        let (data_id, alloc) = match todo_item {
            TodoItem::Alloc(alloc_id) => {
                //println!("alloc_id {}", alloc_id);
                let alloc = memory.get(alloc_id).unwrap();
                let data_id = data_id_for_alloc_id(module, alloc_id, alloc.align);
                (data_id, alloc)
            }
            TodoItem::Static(def_id) => {
                //println!("static {:?}", def_id);

                if tcx.is_foreign_item(def_id) {
                    continue;
                }

                let instance = ty::Instance::mono(tcx, def_id);
                let cid = GlobalId {
                    instance,
                    promoted: None,
                };
                let const_ = tcx.const_eval(ParamEnv::reveal_all().and(cid)).unwrap();

                let alloc = match const_.val {
                    ConstValue::ByRef { alloc, offset } if offset.bytes() == 0 => alloc,
                    _ => bug!("static const eval returned {:#?}", const_),
                };

                // FIXME set correct linkage
                let data_id = data_id_for_static(tcx, module, def_id, Linkage::Export);
                (data_id, alloc)
            }
        };

        //("data_id {}", data_id);
        if cx.done.contains(&data_id) {
            continue;
        }

        let mut data_ctx = DataContext::new();

        data_ctx.define(alloc.bytes.to_vec().into_boxed_slice());

        for &(offset, (_tag, reloc)) in alloc.relocations.iter() {
            let addend = {
                let endianness = tcx.data_layout.endian;
                let offset = offset.bytes() as usize;
                let ptr_size = tcx.data_layout.pointer_size;
                let bytes = &alloc.bytes[offset..offset + ptr_size.bytes() as usize];
                read_target_uint(endianness, bytes).unwrap()
            };

            let data_id = match tcx.alloc_map.lock().get(reloc).unwrap() {
                GlobalAlloc::Function(instance) => {
                    assert_eq!(addend, 0);
                    let func_id = crate::abi::import_function(tcx, module, instance);
                    let local_func_id = module.declare_func_in_data(func_id, &mut data_ctx);
                    data_ctx.write_function_addr(offset.bytes() as u32, local_func_id);
                    continue;
                }
                GlobalAlloc::Memory(_) => {
                    cx.todo.insert(TodoItem::Alloc(reloc));
                    data_id_for_alloc_id(module, reloc, alloc.align)
                }
                GlobalAlloc::Static(def_id) => {
                    cx.todo.insert(TodoItem::Static(def_id));
                    let linkage = crate::linkage::get_static_ref_linkage(tcx, def_id);
                    data_id_for_static(tcx, module, def_id, linkage)
                }
            };

            let global_value = module.declare_data_in_data(data_id, &mut data_ctx);
            data_ctx.write_data_addr(offset.bytes() as u32, global_value, addend as i64);
        }

        module.define_data(data_id, &data_ctx).unwrap();
        cx.done.insert(data_id);
    }

    assert!(cx.todo.is_empty(), "{:?}", cx.todo);
}

fn pop_set<T: Copy + Eq + ::std::hash::Hash>(set: &mut HashSet<T>) -> Option<T> {
    if let Some(elem) = set.iter().next().map(|elem| *elem) {
        set.remove(&elem);
        Some(elem)
    } else {
        None
    }
}

struct TransPlaceInterpreter;

impl<'mir, 'tcx> Machine<'mir, 'tcx> for TransPlaceInterpreter {
    type MemoryKinds = !;
    type ExtraFnVal = !;
    type PointerTag = ();
    type AllocExtra = ();
    type MemoryExtra = ();
    type FrameExtra = ();
    type MemoryMap = FxHashMap<AllocId, (MemoryKind<!>, Allocation<()>)>;

    const CHECK_ALIGN: bool = true;
    const STATIC_KIND: Option<!> = None;

    fn enforce_validity(_: &InterpCx<'mir, 'tcx, Self>) -> bool {
        false
    }

    fn before_terminator(_: &mut InterpCx<'mir, 'tcx, Self>) -> InterpResult<'tcx> {
        panic!();
    }

    fn find_fn(
        _: &mut InterpCx<'mir, 'tcx, Self>,
        _: Instance<'tcx>,
        _: &[OpTy<'tcx>],
        _: Option<PlaceTy<'tcx>>,
        _: Option<BasicBlock>,
    ) -> InterpResult<'tcx, Option<&'mir Body<'tcx>>> {
        panic!();
    }

    fn call_intrinsic(
        _: &mut InterpCx<'mir, 'tcx, Self>,
        _: Instance<'tcx>,
        _: &[OpTy<'tcx>],
        _: PlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        panic!();
    }

    fn find_foreign_static(
        _: TyCtxt<'tcx>,
        _: DefId,
    ) -> InterpResult<'tcx, Cow<'tcx, Allocation>> {
        panic!();
    }

    fn binary_ptr_op(
        _: &InterpCx<'mir, 'tcx, Self>,
        _: mir::BinOp,
        _: ImmTy<'tcx>,
        _: ImmTy<'tcx>,
    ) -> InterpResult<'tcx, (Scalar, bool)> {
        panic!();
    }

    fn ptr_to_int(
        _: &Memory<'mir, 'tcx, Self>,
        _: Pointer<()>,
    ) -> InterpResult<'tcx, u64> {
        panic!();
    }

    fn box_alloc(_: &mut InterpCx<'mir, 'tcx, Self>, _: PlaceTy<'tcx>) -> InterpResult<'tcx> {
        panic!();
    }

    fn tag_allocation<'b>(
        _: &(),
        _: AllocId,
        alloc: Cow<'b, Allocation>,
        _: Option<MemoryKind<!>>,
    ) -> (Cow<'b, Allocation<(), ()>>, ()) {
        (alloc, ())
    }

    fn tag_static_base_pointer(_: &(), _: AllocId) -> Self::PointerTag {
        ()
    }

    fn call_extra_fn(
        _: &mut InterpCx<'mir, 'tcx, Self>,
        _: !,
        _: &[OpTy<'tcx, ()>],
        _: Option<PlaceTy<'tcx, ()>>,
        _: Option<BasicBlock>,
    ) -> InterpResult<'tcx> {
        unreachable!();
    }

    fn stack_push(_: &mut InterpCx<'mir, 'tcx, Self>) -> InterpResult<'tcx> {
        Ok(())
    }

    fn stack_pop(_: &mut InterpCx<'mir, 'tcx, Self>, _: ()) -> InterpResult<'tcx> {
        Ok(())
    }
}

pub fn mir_operand_get_const_val<'tcx>(
    fx: &FunctionCx<'_, 'tcx, impl Backend>,
    operand: &Operand<'tcx>,
) -> Result<&'tcx Const<'tcx>, String> {
    let place = match operand {
        Operand::Copy(place) => place,
        Operand::Constant(const_) => return Ok(force_eval_const(fx, const_.literal)),
        _ => return Err(format!("{:?}", operand)),
    };

    assert!(place.projection.is_none());
    let static_ = match &place.base {
        PlaceBase::Static(static_) => {
            static_
        }
        PlaceBase::Local(_) => return Err("local".to_string()),
    };

    Ok(match &static_.kind {
        StaticKind::Static(_) => unimplemented!(),
        StaticKind::Promoted(promoted) => {
            fx.tcx.const_eval(ParamEnv::reveal_all().and(GlobalId {
                instance: fx.instance,
                promoted: Some(*promoted),
            })).unwrap()
        }
    })
}
