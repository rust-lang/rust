use std::borrow::Cow;

use rustc_span::DUMMY_SP;

use rustc::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc::mir::interpret::{
    read_target_uint, AllocId, Allocation, ConstValue, GlobalAlloc, InterpResult, Scalar,
};
use rustc::ty::{layout::Align, Const, ConstKind};
use rustc_mir::interpret::{
    ImmTy, InterpCx, Machine, Memory, MemoryKind, OpTy, PlaceTy, Pointer,
    StackPopCleanup, StackPopInfo,
};

use cranelift_codegen::ir::GlobalValue;
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
    pub fn finalize(mut self, tcx: TyCtxt<'_>, module: &mut Module<impl Backend>) {
        //println!("todo {:?}", self.todo);
        define_all_allocs(tcx, module, &mut self);
        //println!("done {:?}", self.done);
        self.done.clear();
    }
}

pub fn codegen_static(constants_cx: &mut ConstantCx, def_id: DefId) {
    constants_cx.todo.insert(TodoItem::Static(def_id));
}

fn codegen_static_ref<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    def_id: DefId,
    layout: TyLayout<'tcx>,
) -> CPlace<'tcx> {
    let linkage = crate::linkage::get_static_ref_linkage(fx.tcx, def_id);
    let data_id = data_id_for_static(fx.tcx, fx.module, def_id, linkage);
    let local_data_id = fx.module.declare_data_in_func(data_id, &mut fx.bcx.func);
    #[cfg(debug_assertions)]
    fx.add_entity_comment(local_data_id, format!("{:?}", def_id));
    cplace_for_dataid(fx, layout, local_data_id)
}

pub fn trans_constant<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    constant: &Constant<'tcx>,
) -> CValue<'tcx> {
    let const_ = match constant.literal.val {
        ConstKind::Unevaluated(def_id, ref substs, promoted) if fx.tcx.is_static(def_id) => {
            assert!(substs.is_empty());
            assert!(promoted.is_none());

            return codegen_static_ref(
                fx,
                def_id,
                fx.layout_of(fx.monomorphize(&constant.literal.ty)),
            ).to_cvalue(fx);
        }
        _ => fx.monomorphize(&constant.literal).eval(fx.tcx, ParamEnv::reveal_all()),
    };

    trans_const_value(fx, const_)
}

pub fn trans_const_value<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    const_: &'tcx Const<'tcx>,
) -> CValue<'tcx> {
    let ty = fx.monomorphize(&const_.ty);
    let layout = fx.layout_of(ty);

    if layout.is_zst() {
        return CValue::by_ref(
            crate::Pointer::const_addr(fx, i64::try_from(layout.align.pref.bytes()).unwrap()),
            layout,
        );
    }
    let const_val = match const_.val {
        ConstKind::Value(const_val) => const_val,
        _ => unreachable!("Const {:?} should have been evaluated", const_),
    };

    match const_val {
        ConstValue::Scalar(x) => {
            if fx.clif_type(layout.ty).is_none() {
                return trans_const_place(fx, const_).to_cvalue(fx);
            }

            match x {
                Scalar::Raw { data, size } => {
                    assert_eq!(u64::from(size), layout.size.bytes());
                    return CValue::const_val(fx, layout, data);
                }
                Scalar::Ptr(ptr) => {
                    let alloc_kind = fx.tcx.alloc_map.lock().get(ptr.alloc_id);
                    let base_addr = match alloc_kind {
                        Some(GlobalAlloc::Memory(alloc)) => {
                            fx.constants_cx.todo.insert(TodoItem::Alloc(ptr.alloc_id));
                            let data_id = data_id_for_alloc_id(fx.module, ptr.alloc_id, alloc.align);
                            let local_data_id = fx.module.declare_data_in_func(data_id, &mut fx.bcx.func);
                            #[cfg(debug_assertions)]
                            fx.add_entity_comment(local_data_id, format!("{:?}", ptr.alloc_id));
                            fx.bcx.ins().global_value(fx.pointer_type, local_data_id)
                        }
                        Some(GlobalAlloc::Function(instance)) => {
                            let func_id = crate::abi::import_function(fx.tcx, fx.module, instance);
                            let local_func_id = fx.module.declare_func_in_func(func_id, &mut fx.bcx.func);
                            fx.bcx.ins().func_addr(fx.pointer_type, local_func_id)
                        }
                        Some(GlobalAlloc::Static(def_id)) => {
                            assert!(fx.tcx.is_static(def_id));
                            let linkage = crate::linkage::get_static_ref_linkage(fx.tcx, def_id);
                            let data_id = data_id_for_static(fx.tcx, fx.module, def_id, linkage);
                            let local_data_id = fx.module.declare_data_in_func(data_id, &mut fx.bcx.func);
                            #[cfg(debug_assertions)]
                            fx.add_entity_comment(local_data_id, format!("{:?}", def_id));
                            fx.bcx.ins().global_value(fx.pointer_type, local_data_id)
                        }
                        None => bug!("missing allocation {:?}", ptr.alloc_id),
                    };
                    let val = fx.bcx.ins().iadd_imm(base_addr, i64::try_from(ptr.offset.bytes()).unwrap());
                    return CValue::by_val(val, layout);
                }
            }
        }
        ConstValue::ByRef { alloc, offset } => {
            let alloc_id = fx.tcx.alloc_map.lock().create_memory_alloc(alloc);
            fx.constants_cx.todo.insert(TodoItem::Alloc(alloc_id));
            let data_id = data_id_for_alloc_id(fx.module, alloc_id, alloc.align);
            let local_data_id = fx.module.declare_data_in_func(data_id, &mut fx.bcx.func);
            let global_ptr = fx.bcx.ins().global_value(fx.pointer_type, local_data_id);
            assert!(!layout.is_unsized(), "unsized ConstValue::ByRef not supported");
            CValue::by_ref(
                crate::pointer::Pointer::new(global_ptr)
                    .offset_i64(fx, i64::try_from(offset.bytes()).unwrap()),
                layout,
            )
        }
        ConstValue::Slice { data: _, start: _, end: _ } => {
            trans_const_place(fx, const_).to_cvalue(fx)
        }
    }
}

fn trans_const_place<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
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
                user_ty: None,
                literal: const_,
            })),
            None,
        )?;
        let ptr = ecx.allocate(op.layout, MemoryKind::Stack);
        ecx.copy_op(op, ptr.into())?;
        let alloc = ecx
            .memory
            .get_raw(ptr.to_ref().to_scalar()?.assert_ptr().alloc_id)?;
        Ok(fx.tcx.intern_const_alloc(alloc.clone()))
    };
    let alloc = result().expect("unable to convert ConstKind to Allocation");

    //println!("const value: {:?} allocation: {:?}", value, alloc);
    let alloc_id = fx.tcx.alloc_map.lock().create_memory_alloc(alloc);
    fx.constants_cx.todo.insert(TodoItem::Alloc(alloc_id));
    let data_id = data_id_for_alloc_id(fx.module, alloc_id, alloc.align);
    let local_data_id = fx.module.declare_data_in_func(data_id, &mut fx.bcx.func);
    #[cfg(debug_assertions)]
    fx.add_entity_comment(local_data_id, format!("{:?}", alloc_id));
    cplace_for_dataid(fx, fx.layout_of(const_.ty), local_data_id)
}

fn data_id_for_alloc_id<B: Backend>(
    module: &mut Module<B>,
    alloc_id: AllocId,
    align: Align,
) -> DataId {
    module
        .declare_data(
            &format!("__alloc_{}", alloc_id.0),
            Linkage::Local,
            false,
            false,
            Some(align.bytes() as u8),
        )
        .unwrap()
}

fn data_id_for_static(
    tcx: TyCtxt<'_>,
    module: &mut Module<impl Backend>,
    def_id: DefId,
    linkage: Linkage,
) -> DataId {
    let instance = Instance::mono(tcx, def_id);
    let symbol_name = tcx.symbol_name(instance).name.as_str();
    let ty = instance.monomorphic_ty(tcx);
    let is_mutable = if tcx.is_mutable_static(def_id) {
        true
    } else {
        !ty.is_freeze(tcx, ParamEnv::reveal_all(), DUMMY_SP)
    };
    let align = tcx
        .layout_of(ParamEnv::reveal_all().and(ty))
        .unwrap()
        .align
        .pref
        .bytes();

    let attrs = tcx.codegen_fn_attrs(def_id);

    let data_id = module
        .declare_data(
            &*symbol_name,
            linkage,
            is_mutable,
            attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL),
            Some(align.try_into().unwrap()),
        )
        .unwrap();

    if linkage == Linkage::Preemptible {
        if let ty::RawPtr(_) = ty.kind {
        } else {
            tcx.sess.span_fatal(
                tcx.def_span(def_id),
                "must have type `*const T` or `*mut T` due to `#[linkage]` attribute",
            )
        }

        let mut data_ctx = DataContext::new();
        data_ctx.define_zeroinit(pointer_ty(tcx).bytes() as usize);
        match module.define_data(data_id, &data_ctx) {
            // Everytime a weak static is referenced, there will be a zero pointer definition,
            // so duplicate definitions are expected and allowed.
            Err(ModuleError::DuplicateDefinition(_)) => {}
            res => res.unwrap(),
        }
    }

    data_id
}

fn cplace_for_dataid<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    layout: TyLayout<'tcx>,
    local_data_id: GlobalValue,
) -> CPlace<'tcx> {
    let global_ptr = fx.bcx.ins().global_value(fx.pointer_type, local_data_id);
    assert!(!layout.is_unsized(), "unsized statics aren't supported");
    CPlace::for_ptr(crate::pointer::Pointer::new(global_ptr), layout)
}

fn define_all_allocs(tcx: TyCtxt<'_>, module: &mut Module<impl Backend>, cx: &mut ConstantCx) {
    let memory = Memory::<TransPlaceInterpreter>::new(tcx.at(DUMMY_SP), ());

    while let Some(todo_item) = pop_set(&mut cx.todo) {
        let (data_id, alloc) = match todo_item {
            TodoItem::Alloc(alloc_id) => {
                //println!("alloc_id {}", alloc_id);
                let alloc = memory.get_raw(alloc_id).unwrap();
                let data_id = data_id_for_alloc_id(module, alloc_id, alloc.align);
                (data_id, alloc)
            }
            TodoItem::Static(def_id) => {
                //println!("static {:?}", def_id);

                if tcx.is_foreign_item(def_id) {
                    continue;
                }

                let const_ = tcx.const_eval_poly(def_id).unwrap();

                let alloc = match const_ {
                    ConstValue::ByRef { alloc, offset } if offset.bytes() == 0 => alloc,
                    _ => bug!("static const eval returned {:#?}", const_),
                };

                let data_id = data_id_for_static(
                    tcx,
                    module,
                    def_id,
                    if tcx.is_reachable_non_generic(def_id) {
                        Linkage::Export
                    } else {
                        Linkage::Export // FIXME Set hidden visibility
                    },
                );
                (data_id, alloc)
            }
        };

        //("data_id {}", data_id);
        if cx.done.contains(&data_id) {
            continue;
        }

        let mut data_ctx = DataContext::new();

        let bytes = alloc.inspect_with_undef_and_ptr_outside_interpreter(0..alloc.len()).to_vec();
        data_ctx.define(bytes.into_boxed_slice());

        for &(offset, (_tag, reloc)) in alloc.relocations().iter() {
            let addend = {
                let endianness = tcx.data_layout.endian;
                let offset = offset.bytes() as usize;
                let ptr_size = tcx.data_layout.pointer_size;
                let bytes = &alloc.inspect_with_undef_and_ptr_outside_interpreter(offset..offset + ptr_size.bytes() as usize);
                read_target_uint(endianness, bytes).unwrap()
            };

            // Don't inline `reloc_target_alloc` into the match. That would cause `tcx.alloc_map`
            // to be locked for the duration of the match. `data_id_for_static` however may try
            // to lock `tcx.alloc_map` itself while calculating the layout of the target static.
            // This would cause a panic in single threaded rustc and a deadlock for parallel rustc.
            let reloc_target_alloc = tcx.alloc_map.lock().get(reloc).unwrap();
            let data_id = match reloc_target_alloc {
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
                    if tcx.codegen_fn_attrs(def_id).flags.contains(CodegenFnAttrFlags::THREAD_LOCAL) {
                        tcx.sess.fatal(&format!("Allocation {:?} contains reference to TLS value {:?}", alloc, def_id));
                    }

                    // Don't push a `TodoItem::Static` here, as it will cause statics used by
                    // multiple crates to be duplicated between them. It isn't necessary anyway,
                    // as it will get pushed by `codegen_static` when necessary.
                    data_id_for_static(
                        tcx,
                        module,
                        def_id,
                        crate::linkage::get_static_ref_linkage(tcx, def_id),
                    )
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

    fn find_mir_or_eval_fn(
        _: &mut InterpCx<'mir, 'tcx, Self>,
        _: Span,
        _: Instance<'tcx>,
        _: &[OpTy<'tcx>],
        _: Option<(PlaceTy<'tcx>, BasicBlock)>,
        _: Option<BasicBlock>,
    ) -> InterpResult<'tcx, Option<&'mir Body<'tcx>>> {
        panic!();
    }

    fn call_intrinsic(
        _: &mut InterpCx<'mir, 'tcx, Self>,
        _: Span,
        _: Instance<'tcx>,
        _: &[OpTy<'tcx>],
        _: Option<(PlaceTy<'tcx>, BasicBlock)>,
        _: Option<BasicBlock>,
    ) -> InterpResult<'tcx> {
        panic!();
    }

    fn binary_ptr_op(
        _: &InterpCx<'mir, 'tcx, Self>,
        _: mir::BinOp,
        _: ImmTy<'tcx>,
        _: ImmTy<'tcx>,
    ) -> InterpResult<'tcx, (Scalar, bool, Ty<'tcx>)> {
        panic!();
    }

    fn ptr_to_int(_: &Memory<'mir, 'tcx, Self>, _: Pointer<()>) -> InterpResult<'tcx, u64> {
        panic!();
    }

    fn box_alloc(_: &mut InterpCx<'mir, 'tcx, Self>, _: PlaceTy<'tcx>) -> InterpResult<'tcx> {
        panic!();
    }

    fn init_allocation_extra<'b>(
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
        _: Option<(PlaceTy<'tcx, ()>, BasicBlock)>,
        _: Option<BasicBlock>,
    ) -> InterpResult<'tcx> {
        unreachable!();
    }

    fn stack_push(_: &mut InterpCx<'mir, 'tcx, Self>) -> InterpResult<'tcx> {
        Ok(())
    }

    fn stack_pop(_: &mut InterpCx<'mir, 'tcx, Self>, _: (), _: bool) -> InterpResult<'tcx, StackPopInfo> {
        Ok(StackPopInfo::Normal)
    }

    fn assert_panic(
        _: &mut InterpCx<'mir, 'tcx, Self>,
        _: &mir::AssertKind<Operand<'tcx>>,
        _: Option<BasicBlock>,
    ) -> InterpResult<'tcx> {
        unreachable!()
    }
}

pub fn mir_operand_get_const_val<'tcx>(
    fx: &FunctionCx<'_, 'tcx, impl Backend>,
    operand: &Operand<'tcx>,
) -> Option<&'tcx Const<'tcx>> {
    match operand {
        Operand::Copy(_) | Operand::Move(_) => None,
        Operand::Constant(const_) => {
            Some(fx.monomorphize(&const_.literal).eval(fx.tcx, ParamEnv::reveal_all()))
        }
    }
}
