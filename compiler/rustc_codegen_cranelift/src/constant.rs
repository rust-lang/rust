//! Handling of `static`s, `const`s and promoted allocations

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::interpret::{
    read_target_uint, AllocId, ConstAllocation, ConstValue, ErrorHandled, GlobalAlloc, Scalar,
};

use cranelift_module::*;

use crate::prelude::*;

pub(crate) struct ConstantCx {
    todo: Vec<TodoItem>,
    done: FxHashSet<DataId>,
    anon_allocs: FxHashMap<AllocId, DataId>,
}

#[derive(Copy, Clone, Debug)]
enum TodoItem {
    Alloc(AllocId),
    Static(DefId),
}

impl ConstantCx {
    pub(crate) fn new() -> Self {
        ConstantCx { todo: vec![], done: FxHashSet::default(), anon_allocs: FxHashMap::default() }
    }

    pub(crate) fn finalize(mut self, tcx: TyCtxt<'_>, module: &mut dyn Module) {
        define_all_allocs(tcx, module, &mut self);
        self.done.clear();
    }
}

pub(crate) fn check_constants(fx: &mut FunctionCx<'_, '_, '_>) -> bool {
    let mut all_constants_ok = true;
    for constant in &fx.mir.required_consts {
        if eval_mir_constant(fx, constant).is_none() {
            all_constants_ok = false;
        }
    }
    all_constants_ok
}

pub(crate) fn codegen_static(tcx: TyCtxt<'_>, module: &mut dyn Module, def_id: DefId) {
    let mut constants_cx = ConstantCx::new();
    constants_cx.todo.push(TodoItem::Static(def_id));
    constants_cx.finalize(tcx, module);
}

pub(crate) fn codegen_tls_ref<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    def_id: DefId,
    layout: TyAndLayout<'tcx>,
) -> CValue<'tcx> {
    let tls_ptr = if !def_id.is_local() && fx.tcx.needs_thread_local_shim(def_id) {
        let instance = ty::Instance {
            def: ty::InstanceDef::ThreadLocalShim(def_id),
            substs: ty::InternalSubsts::empty(),
        };
        let func_ref = fx.get_function_ref(instance);
        let call = fx.bcx.ins().call(func_ref, &[]);
        fx.bcx.func.dfg.first_result(call)
    } else {
        let data_id = data_id_for_static(fx.tcx, fx.module, def_id, false);
        let local_data_id = fx.module.declare_data_in_func(data_id, &mut fx.bcx.func);
        if fx.clif_comments.enabled() {
            fx.add_comment(local_data_id, format!("tls {:?}", def_id));
        }
        fx.bcx.ins().tls_value(fx.pointer_type, local_data_id)
    };
    CValue::by_val(tls_ptr, layout)
}

pub(crate) fn eval_mir_constant<'tcx>(
    fx: &FunctionCx<'_, '_, 'tcx>,
    constant: &Constant<'tcx>,
) -> Option<(ConstValue<'tcx>, Ty<'tcx>)> {
    let constant_kind = fx.monomorphize(constant.literal);
    let uv = match constant_kind {
        ConstantKind::Ty(const_) => match const_.kind() {
            ty::ConstKind::Unevaluated(uv) => uv.expand(),
            ty::ConstKind::Value(val) => {
                return Some((fx.tcx.valtree_to_const_val((const_.ty(), val)), const_.ty()));
            }
            err => span_bug!(
                constant.span,
                "encountered bad ConstKind after monomorphizing: {:?}",
                err
            ),
        },
        ConstantKind::Unevaluated(mir::UnevaluatedConst { def, .. }, _)
            if fx.tcx.is_static(def.did) =>
        {
            span_bug!(constant.span, "MIR constant refers to static");
        }
        ConstantKind::Unevaluated(uv, _) => uv,
        ConstantKind::Val(val, _) => return Some((val, constant_kind.ty())),
    };

    let val = fx
        .tcx
        .const_eval_resolve(ty::ParamEnv::reveal_all(), uv, None)
        .map_err(|err| match err {
            ErrorHandled::Reported(_) => {
                fx.tcx.sess.span_err(constant.span, "erroneous constant encountered");
            }
            ErrorHandled::TooGeneric => {
                span_bug!(constant.span, "codegen encountered polymorphic constant: {:?}", err);
            }
        })
        .ok();
    val.map(|val| (val, constant_kind.ty()))
}

pub(crate) fn codegen_constant_operand<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    constant: &Constant<'tcx>,
) -> CValue<'tcx> {
    let (const_val, ty) = eval_mir_constant(fx, constant).unwrap_or_else(|| {
        span_bug!(constant.span, "erroneous constant not captured by required_consts")
    });

    codegen_const_value(fx, const_val, ty)
}

pub(crate) fn codegen_const_value<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    const_val: ConstValue<'tcx>,
    ty: Ty<'tcx>,
) -> CValue<'tcx> {
    let layout = fx.layout_of(ty);
    assert!(layout.is_sized(), "unsized const value");

    if layout.is_zst() {
        return CValue::by_ref(crate::Pointer::dangling(layout.align.pref), layout);
    }

    match const_val {
        ConstValue::ZeroSized => unreachable!(), // we already handles ZST above
        ConstValue::Scalar(x) => match x {
            Scalar::Int(int) => {
                if fx.clif_type(layout.ty).is_some() {
                    return CValue::const_val(fx, layout, int);
                } else {
                    let raw_val = int.to_bits(int.size()).unwrap();
                    let val = match int.size().bytes() {
                        1 => fx.bcx.ins().iconst(types::I8, raw_val as i64),
                        2 => fx.bcx.ins().iconst(types::I16, raw_val as i64),
                        4 => fx.bcx.ins().iconst(types::I32, raw_val as i64),
                        8 => fx.bcx.ins().iconst(types::I64, raw_val as i64),
                        16 => {
                            let lsb = fx.bcx.ins().iconst(types::I64, raw_val as u64 as i64);
                            let msb =
                                fx.bcx.ins().iconst(types::I64, (raw_val >> 64) as u64 as i64);
                            fx.bcx.ins().iconcat(lsb, msb)
                        }
                        _ => unreachable!(),
                    };

                    let place = CPlace::new_stack_slot(fx, layout);
                    place.to_ptr().store(fx, val, MemFlags::trusted());
                    place.to_cvalue(fx)
                }
            }
            Scalar::Ptr(ptr, _size) => {
                let (alloc_id, offset) = ptr.into_parts(); // we know the `offset` is relative
                let base_addr = match fx.tcx.global_alloc(alloc_id) {
                    GlobalAlloc::Memory(alloc) => {
                        let data_id = data_id_for_alloc_id(
                            &mut fx.constants_cx,
                            fx.module,
                            alloc_id,
                            alloc.inner().mutability,
                        );
                        let local_data_id =
                            fx.module.declare_data_in_func(data_id, &mut fx.bcx.func);
                        if fx.clif_comments.enabled() {
                            fx.add_comment(local_data_id, format!("{:?}", alloc_id));
                        }
                        fx.bcx.ins().global_value(fx.pointer_type, local_data_id)
                    }
                    GlobalAlloc::Function(instance) => {
                        let func_id = crate::abi::import_function(fx.tcx, fx.module, instance);
                        let local_func_id =
                            fx.module.declare_func_in_func(func_id, &mut fx.bcx.func);
                        fx.bcx.ins().func_addr(fx.pointer_type, local_func_id)
                    }
                    GlobalAlloc::VTable(ty, trait_ref) => {
                        let alloc_id = fx.tcx.vtable_allocation((ty, trait_ref));
                        let alloc = fx.tcx.global_alloc(alloc_id).unwrap_memory();
                        // FIXME: factor this common code with the `Memory` arm into a function?
                        let data_id = data_id_for_alloc_id(
                            &mut fx.constants_cx,
                            fx.module,
                            alloc_id,
                            alloc.inner().mutability,
                        );
                        let local_data_id =
                            fx.module.declare_data_in_func(data_id, &mut fx.bcx.func);
                        fx.bcx.ins().global_value(fx.pointer_type, local_data_id)
                    }
                    GlobalAlloc::Static(def_id) => {
                        assert!(fx.tcx.is_static(def_id));
                        let data_id = data_id_for_static(fx.tcx, fx.module, def_id, false);
                        let local_data_id =
                            fx.module.declare_data_in_func(data_id, &mut fx.bcx.func);
                        if fx.clif_comments.enabled() {
                            fx.add_comment(local_data_id, format!("{:?}", def_id));
                        }
                        fx.bcx.ins().global_value(fx.pointer_type, local_data_id)
                    }
                };
                let val = if offset.bytes() != 0 {
                    fx.bcx.ins().iadd_imm(base_addr, i64::try_from(offset.bytes()).unwrap())
                } else {
                    base_addr
                };
                CValue::by_val(val, layout)
            }
        },
        ConstValue::ByRef { alloc, offset } => CValue::by_ref(
            pointer_for_allocation(fx, alloc)
                .offset_i64(fx, i64::try_from(offset.bytes()).unwrap()),
            layout,
        ),
        ConstValue::Slice { data, start, end } => {
            let ptr = pointer_for_allocation(fx, data)
                .offset_i64(fx, i64::try_from(start).unwrap())
                .get_addr(fx);
            let len = fx
                .bcx
                .ins()
                .iconst(fx.pointer_type, i64::try_from(end.checked_sub(start).unwrap()).unwrap());
            CValue::by_val_pair(ptr, len, layout)
        }
    }
}

fn pointer_for_allocation<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    alloc: ConstAllocation<'tcx>,
) -> crate::pointer::Pointer {
    let alloc_id = fx.tcx.create_memory_alloc(alloc);
    let data_id = data_id_for_alloc_id(
        &mut fx.constants_cx,
        &mut *fx.module,
        alloc_id,
        alloc.inner().mutability,
    );

    let local_data_id = fx.module.declare_data_in_func(data_id, &mut fx.bcx.func);
    if fx.clif_comments.enabled() {
        fx.add_comment(local_data_id, format!("{:?}", alloc_id));
    }
    let global_ptr = fx.bcx.ins().global_value(fx.pointer_type, local_data_id);
    crate::pointer::Pointer::new(global_ptr)
}

pub(crate) fn data_id_for_alloc_id(
    cx: &mut ConstantCx,
    module: &mut dyn Module,
    alloc_id: AllocId,
    mutability: rustc_hir::Mutability,
) -> DataId {
    cx.todo.push(TodoItem::Alloc(alloc_id));
    *cx.anon_allocs
        .entry(alloc_id)
        .or_insert_with(|| module.declare_anonymous_data(mutability.is_mut(), false).unwrap())
}

fn data_id_for_static(
    tcx: TyCtxt<'_>,
    module: &mut dyn Module,
    def_id: DefId,
    definition: bool,
) -> DataId {
    let attrs = tcx.codegen_fn_attrs(def_id);

    let instance = Instance::mono(tcx, def_id).polymorphize(tcx);
    let symbol_name = tcx.symbol_name(instance).name;
    let ty = instance.ty(tcx, ParamEnv::reveal_all());
    let is_mutable = if tcx.is_mutable_static(def_id) {
        true
    } else {
        !ty.is_freeze(tcx, ParamEnv::reveal_all())
    };
    let align = tcx.layout_of(ParamEnv::reveal_all().and(ty)).unwrap().align.pref.bytes();

    if let Some(import_linkage) = attrs.import_linkage {
        assert!(!definition);

        let linkage = if import_linkage == rustc_middle::mir::mono::Linkage::ExternalWeak
            || import_linkage == rustc_middle::mir::mono::Linkage::WeakAny
        {
            Linkage::Preemptible
        } else {
            Linkage::Import
        };

        let data_id = match module.declare_data(
            symbol_name,
            linkage,
            is_mutable,
            attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL),
        ) {
            Ok(data_id) => data_id,
            Err(ModuleError::IncompatibleDeclaration(_)) => tcx.sess.fatal(&format!(
                "attempt to declare `{symbol_name}` as static, but it was already declared as function"
            )),
            Err(err) => Err::<_, _>(err).unwrap(),
        };

        // Comment copied from https://github.com/rust-lang/rust/blob/45060c2a66dfd667f88bd8b94261b28a58d85bd5/src/librustc_codegen_llvm/consts.rs#L141
        // Declare an internal global `extern_with_linkage_foo` which
        // is initialized with the address of `foo`. If `foo` is
        // discarded during linking (for example, if `foo` has weak
        // linkage and there are no definitions), then
        // `extern_with_linkage_foo` will instead be initialized to
        // zero.

        let ref_name = format!("_rust_extern_with_linkage_{}", symbol_name);
        let ref_data_id = module.declare_data(&ref_name, Linkage::Local, false, false).unwrap();
        let mut data_ctx = DataContext::new();
        data_ctx.set_align(align);
        let data = module.declare_data_in_data(data_id, &mut data_ctx);
        data_ctx.define(std::iter::repeat(0).take(pointer_ty(tcx).bytes() as usize).collect());
        data_ctx.write_data_addr(0, data, 0);
        match module.define_data(ref_data_id, &data_ctx) {
            // Every time the static is referenced there will be another definition of this global,
            // so duplicate definitions are expected and allowed.
            Err(ModuleError::DuplicateDefinition(_)) => {}
            res => res.unwrap(),
        }

        return ref_data_id;
    }

    let linkage = if definition {
        crate::linkage::get_static_linkage(tcx, def_id)
    } else if attrs.linkage == Some(rustc_middle::mir::mono::Linkage::ExternalWeak)
        || attrs.linkage == Some(rustc_middle::mir::mono::Linkage::WeakAny)
    {
        Linkage::Preemptible
    } else {
        Linkage::Import
    };

    let data_id = match module.declare_data(
        symbol_name,
        linkage,
        is_mutable,
        attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL),
    ) {
        Ok(data_id) => data_id,
        Err(ModuleError::IncompatibleDeclaration(_)) => tcx.sess.fatal(&format!(
            "attempt to declare `{symbol_name}` as static, but it was already declared as function"
        )),
        Err(err) => Err::<_, _>(err).unwrap(),
    };

    data_id
}

fn define_all_allocs(tcx: TyCtxt<'_>, module: &mut dyn Module, cx: &mut ConstantCx) {
    while let Some(todo_item) = cx.todo.pop() {
        let (data_id, alloc, section_name) = match todo_item {
            TodoItem::Alloc(alloc_id) => {
                let alloc = match tcx.global_alloc(alloc_id) {
                    GlobalAlloc::Memory(alloc) => alloc,
                    GlobalAlloc::Function(_) | GlobalAlloc::Static(_) | GlobalAlloc::VTable(..) => {
                        unreachable!()
                    }
                };
                let data_id = *cx.anon_allocs.entry(alloc_id).or_insert_with(|| {
                    module.declare_anonymous_data(alloc.inner().mutability.is_mut(), false).unwrap()
                });
                (data_id, alloc, None)
            }
            TodoItem::Static(def_id) => {
                let section_name = tcx.codegen_fn_attrs(def_id).link_section;

                let alloc = tcx.eval_static_initializer(def_id).unwrap();

                let data_id = data_id_for_static(tcx, module, def_id, true);
                (data_id, alloc, section_name)
            }
        };

        if cx.done.contains(&data_id) {
            continue;
        }

        let mut data_ctx = DataContext::new();
        let alloc = alloc.inner();
        data_ctx.set_align(alloc.align.bytes());

        if let Some(section_name) = section_name {
            let (segment_name, section_name) = if tcx.sess.target.is_like_osx {
                let section_name = section_name.as_str();
                if let Some(names) = section_name.split_once(',') {
                    names
                } else {
                    tcx.sess.fatal(&format!(
                        "#[link_section = \"{}\"] is not valid for macos target: must be segment and section separated by comma",
                        section_name
                    ));
                }
            } else {
                ("", section_name.as_str())
            };
            data_ctx.set_segment_section(segment_name, section_name);
        }

        let bytes = alloc.inspect_with_uninit_and_ptr_outside_interpreter(0..alloc.len()).to_vec();
        data_ctx.define(bytes.into_boxed_slice());

        for &(offset, alloc_id) in alloc.provenance().ptrs().iter() {
            let addend = {
                let endianness = tcx.data_layout.endian;
                let offset = offset.bytes() as usize;
                let ptr_size = tcx.data_layout.pointer_size;
                let bytes = &alloc.inspect_with_uninit_and_ptr_outside_interpreter(
                    offset..offset + ptr_size.bytes() as usize,
                );
                read_target_uint(endianness, bytes).unwrap()
            };

            let reloc_target_alloc = tcx.global_alloc(alloc_id);
            let data_id = match reloc_target_alloc {
                GlobalAlloc::Function(instance) => {
                    assert_eq!(addend, 0);
                    let func_id =
                        crate::abi::import_function(tcx, module, instance.polymorphize(tcx));
                    let local_func_id = module.declare_func_in_data(func_id, &mut data_ctx);
                    data_ctx.write_function_addr(offset.bytes() as u32, local_func_id);
                    continue;
                }
                GlobalAlloc::Memory(target_alloc) => {
                    data_id_for_alloc_id(cx, module, alloc_id, target_alloc.inner().mutability)
                }
                GlobalAlloc::VTable(ty, trait_ref) => {
                    let alloc_id = tcx.vtable_allocation((ty, trait_ref));
                    data_id_for_alloc_id(cx, module, alloc_id, Mutability::Not)
                }
                GlobalAlloc::Static(def_id) => {
                    if tcx.codegen_fn_attrs(def_id).flags.contains(CodegenFnAttrFlags::THREAD_LOCAL)
                    {
                        tcx.sess.fatal(&format!(
                            "Allocation {:?} contains reference to TLS value {:?}",
                            alloc_id, def_id
                        ));
                    }

                    // Don't push a `TodoItem::Static` here, as it will cause statics used by
                    // multiple crates to be duplicated between them. It isn't necessary anyway,
                    // as it will get pushed by `codegen_static` when necessary.
                    data_id_for_static(tcx, module, def_id, false)
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

/// Used only for intrinsic implementations that need a compile-time constant
pub(crate) fn mir_operand_get_const_val<'tcx>(
    fx: &FunctionCx<'_, '_, 'tcx>,
    operand: &Operand<'tcx>,
) -> Option<ConstValue<'tcx>> {
    match operand {
        Operand::Constant(const_) => Some(eval_mir_constant(fx, const_).unwrap().0),
        // FIXME(rust-lang/rust#85105): Casts like `IMM8 as u32` result in the const being stored
        // inside a temporary before being passed to the intrinsic requiring the const argument.
        // This code tries to find a single constant defining definition of the referenced local.
        Operand::Copy(place) | Operand::Move(place) => {
            if !place.projection.is_empty() {
                return None;
            }
            let mut computed_const_val = None;
            for bb_data in fx.mir.basic_blocks.iter() {
                for stmt in &bb_data.statements {
                    match &stmt.kind {
                        StatementKind::Assign(local_and_rvalue) if &local_and_rvalue.0 == place => {
                            match &local_and_rvalue.1 {
                                Rvalue::Cast(
                                    CastKind::IntToInt
                                    | CastKind::FloatToFloat
                                    | CastKind::FloatToInt
                                    | CastKind::IntToFloat
                                    | CastKind::FnPtrToPtr
                                    | CastKind::PtrToPtr,
                                    operand,
                                    ty,
                                ) => {
                                    if computed_const_val.is_some() {
                                        return None; // local assigned twice
                                    }
                                    if !matches!(ty.kind(), ty::Uint(_) | ty::Int(_)) {
                                        return None;
                                    }
                                    let const_val = mir_operand_get_const_val(fx, operand)?;
                                    if fx.layout_of(*ty).size
                                        != const_val.try_to_scalar_int()?.size()
                                    {
                                        return None;
                                    }
                                    computed_const_val = Some(const_val);
                                }
                                Rvalue::Use(operand) => {
                                    computed_const_val = mir_operand_get_const_val(fx, operand)
                                }
                                _ => return None,
                            }
                        }
                        StatementKind::SetDiscriminant { place: stmt_place, variant_index: _ }
                            if &**stmt_place == place =>
                        {
                            return None;
                        }
                        StatementKind::Intrinsic(ref intrinsic) => match **intrinsic {
                            NonDivergingIntrinsic::CopyNonOverlapping(..) => return None,
                            NonDivergingIntrinsic::Assume(..) => {}
                        },
                        // conservative handling
                        StatementKind::Assign(_)
                        | StatementKind::FakeRead(_)
                        | StatementKind::SetDiscriminant { .. }
                        | StatementKind::Deinit(_)
                        | StatementKind::StorageLive(_)
                        | StatementKind::StorageDead(_)
                        | StatementKind::Retag(_, _)
                        | StatementKind::AscribeUserType(_, _)
                        | StatementKind::PlaceMention(..)
                        | StatementKind::Coverage(_)
                        | StatementKind::ConstEvalCounter
                        | StatementKind::Nop => {}
                    }
                }
                match &bb_data.terminator().kind {
                    TerminatorKind::Goto { .. }
                    | TerminatorKind::SwitchInt { .. }
                    | TerminatorKind::Resume
                    | TerminatorKind::Abort
                    | TerminatorKind::Return
                    | TerminatorKind::Unreachable
                    | TerminatorKind::Drop { .. }
                    | TerminatorKind::Assert { .. } => {}
                    TerminatorKind::Yield { .. }
                    | TerminatorKind::GeneratorDrop
                    | TerminatorKind::FalseEdge { .. }
                    | TerminatorKind::FalseUnwind { .. } => unreachable!(),
                    TerminatorKind::InlineAsm { .. } => return None,
                    TerminatorKind::Call { destination, target: Some(_), .. }
                        if destination == place =>
                    {
                        return None;
                    }
                    TerminatorKind::Call { .. } => {}
                }
            }
            computed_const_val
        }
    }
}
