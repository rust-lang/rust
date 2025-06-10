//! Handling of `static`s, `const`s and promoted allocations

use std::cmp::Ordering;

use cranelift_module::*;
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir::interpret::{AllocId, GlobalAlloc, Scalar, read_target_uint};
use rustc_middle::ty::{ExistentialTraitRef, ScalarInt};

use crate::prelude::*;

pub(crate) struct ConstantCx {
    todo: Vec<TodoItem>,
    anon_allocs: FxHashMap<AllocId, DataId>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
enum TodoItem {
    Alloc(AllocId),
    Static(DefId),
}

impl ConstantCx {
    pub(crate) fn new() -> Self {
        ConstantCx { todo: vec![], anon_allocs: FxHashMap::default() }
    }

    pub(crate) fn finalize(mut self, tcx: TyCtxt<'_>, module: &mut dyn Module) {
        define_all_allocs(tcx, module, &mut self);
    }
}

pub(crate) fn codegen_static(tcx: TyCtxt<'_>, module: &mut dyn Module, def_id: DefId) -> DataId {
    let mut constants_cx = ConstantCx::new();
    constants_cx.todo.push(TodoItem::Static(def_id));
    constants_cx.finalize(tcx, module);

    data_id_for_static(
        tcx, module, def_id, false,
        // For a declaration the stated mutability doesn't matter.
        false,
    )
}

pub(crate) fn codegen_tls_ref<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    def_id: DefId,
    layout: TyAndLayout<'tcx>,
) -> CValue<'tcx> {
    let tls_ptr = if !def_id.is_local() && fx.tcx.needs_thread_local_shim(def_id) {
        let instance = ty::Instance {
            def: ty::InstanceKind::ThreadLocalShim(def_id),
            args: ty::GenericArgs::empty(),
        };
        let func_ref = fx.get_function_ref(instance);
        let call = fx.bcx.ins().call(func_ref, &[]);
        fx.bcx.func.dfg.first_result(call)
    } else {
        let data_id = data_id_for_static(
            fx.tcx, fx.module, def_id, false,
            // For a declaration the stated mutability doesn't matter.
            false,
        );
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
    constant: &ConstOperand<'tcx>,
) -> (ConstValue<'tcx>, Ty<'tcx>) {
    let cv = fx.monomorphize(constant.const_);
    // This cannot fail because we checked all required_consts in advance.
    let val = cv
        .eval(fx.tcx, ty::TypingEnv::fully_monomorphized(), constant.span)
        .expect("erroneous constant missed by mono item collection");
    (val, cv.ty())
}

pub(crate) fn codegen_constant_operand<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    constant: &ConstOperand<'tcx>,
) -> CValue<'tcx> {
    let (const_val, ty) = eval_mir_constant(fx, constant);
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
        return CValue::zst(layout);
    }

    match const_val {
        ConstValue::ZeroSized => unreachable!(), // we already handled ZST above
        ConstValue::Scalar(x) => match x {
            Scalar::Int(int) => {
                if fx.clif_type(layout.ty).is_some() {
                    return CValue::const_val(fx, layout, int);
                } else {
                    let raw_val = int.size().truncate(int.to_bits(int.size()));
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

                    // FIXME avoid this extra copy to the stack and directly write to the final
                    // destination
                    let place = CPlace::new_stack_slot(fx, layout);
                    place.to_ptr().store(fx, val, MemFlags::trusted());
                    place.to_cvalue(fx)
                }
            }
            Scalar::Ptr(ptr, _size) => {
                let (prov, offset) = ptr.into_parts(); // we know the `offset` is relative
                let alloc_id = prov.alloc_id();
                let base_addr = match fx.tcx.global_alloc(alloc_id) {
                    GlobalAlloc::Memory(alloc) => {
                        if alloc.inner().len() == 0 {
                            assert_eq!(offset, Size::ZERO);
                            fx.bcx.ins().iconst(fx.pointer_type, alloc.inner().align.bytes() as i64)
                        } else {
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
                    }
                    GlobalAlloc::Function { instance, .. } => {
                        let func_id = crate::abi::import_function(fx.tcx, fx.module, instance);
                        let local_func_id =
                            fx.module.declare_func_in_func(func_id, &mut fx.bcx.func);
                        fx.bcx.ins().func_addr(fx.pointer_type, local_func_id)
                    }
                    GlobalAlloc::VTable(ty, dyn_ty) => {
                        let data_id = data_id_for_vtable(
                            fx.tcx,
                            &mut fx.constants_cx,
                            fx.module,
                            ty,
                            dyn_ty.principal().map(|principal| {
                                fx.tcx.instantiate_bound_regions_with_erased(principal)
                            }),
                        );
                        let local_data_id =
                            fx.module.declare_data_in_func(data_id, &mut fx.bcx.func);
                        fx.bcx.ins().global_value(fx.pointer_type, local_data_id)
                    }
                    GlobalAlloc::Static(def_id) => {
                        assert!(fx.tcx.is_static(def_id));
                        let data_id = data_id_for_static(
                            fx.tcx, fx.module, def_id, false,
                            // For a declaration the stated mutability doesn't matter.
                            false,
                        );
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
        ConstValue::Indirect { alloc_id, offset } => CValue::by_ref(
            pointer_for_allocation(fx, alloc_id)
                .offset_i64(fx, i64::try_from(offset.bytes()).unwrap()),
            layout,
        ),
        ConstValue::Slice { data, meta } => {
            let alloc_id = fx.tcx.reserve_and_set_memory_alloc(data);
            let ptr = pointer_for_allocation(fx, alloc_id).get_addr(fx);
            let len = fx.bcx.ins().iconst(fx.pointer_type, meta as i64);
            CValue::by_val_pair(ptr, len, layout)
        }
    }
}

fn pointer_for_allocation<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    alloc_id: AllocId,
) -> crate::pointer::Pointer {
    let alloc = fx.tcx.global_alloc(alloc_id).unwrap_memory();
    let data_id =
        data_id_for_alloc_id(&mut fx.constants_cx, fx.module, alloc_id, alloc.inner().mutability);

    let local_data_id = fx.module.declare_data_in_func(data_id, &mut fx.bcx.func);
    if fx.clif_comments.enabled() {
        fx.add_comment(local_data_id, format!("{:?}", alloc_id));
    }
    let global_ptr = fx.bcx.ins().global_value(fx.pointer_type, local_data_id);
    crate::pointer::Pointer::new(global_ptr)
}

fn data_id_for_alloc_id(
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

pub(crate) fn data_id_for_vtable<'tcx>(
    tcx: TyCtxt<'tcx>,
    cx: &mut ConstantCx,
    module: &mut dyn Module,
    ty: Ty<'tcx>,
    trait_ref: Option<ExistentialTraitRef<'tcx>>,
) -> DataId {
    let alloc_id = tcx.vtable_allocation((ty, trait_ref));
    data_id_for_alloc_id(cx, module, alloc_id, Mutability::Not)
}

fn data_id_for_static(
    tcx: TyCtxt<'_>,
    module: &mut dyn Module,
    def_id: DefId,
    definition: bool,
    definition_writable: bool,
) -> DataId {
    let attrs = tcx.codegen_fn_attrs(def_id);

    let instance = Instance::mono(tcx, def_id);
    let symbol_name = tcx.symbol_name(instance).name;

    if let Some(import_linkage) = attrs.import_linkage {
        assert!(!definition);
        assert!(!tcx.is_mutable_static(def_id));

        let ty = instance.ty(tcx, ty::TypingEnv::fully_monomorphized());
        let align = tcx
            .layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(ty))
            .unwrap()
            .align
            .abi
            .bytes();

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
            false,
            attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL),
        ) {
            Ok(data_id) => data_id,
            Err(ModuleError::IncompatibleDeclaration(_)) => tcx.dcx().fatal(format!(
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
        let mut data = DataDescription::new();
        data.set_align(align);
        let data_gv = module.declare_data_in_data(data_id, &mut data);
        data.define(std::iter::repeat(0).take(pointer_ty(tcx).bytes() as usize).collect());
        data.write_data_addr(0, data_gv, 0);
        match module.define_data(ref_data_id, &data) {
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
        definition_writable,
        attrs.flags.contains(CodegenFnAttrFlags::THREAD_LOCAL),
    ) {
        Ok(data_id) => data_id,
        Err(ModuleError::IncompatibleDeclaration(_)) => tcx.dcx().fatal(format!(
            "attempt to declare `{symbol_name}` as static, but it was already declared as function"
        )),
        Err(err) => Err::<_, _>(err).unwrap(),
    };

    data_id
}

fn define_all_allocs(tcx: TyCtxt<'_>, module: &mut dyn Module, cx: &mut ConstantCx) {
    let mut done = FxHashSet::default();
    while let Some(todo_item) = cx.todo.pop() {
        if !done.insert(todo_item) {
            continue;
        }

        let (data_id, alloc, section_name) = match todo_item {
            TodoItem::Alloc(alloc_id) => {
                let alloc = match tcx.global_alloc(alloc_id) {
                    GlobalAlloc::Memory(alloc) => alloc,
                    GlobalAlloc::Function { .. }
                    | GlobalAlloc::Static(_)
                    | GlobalAlloc::VTable(..) => {
                        unreachable!()
                    }
                };
                // FIXME: should we have a cache so we don't do this multiple times for the same `ConstAllocation`?
                let data_id = *cx.anon_allocs.entry(alloc_id).or_insert_with(|| {
                    module.declare_anonymous_data(alloc.inner().mutability.is_mut(), false).unwrap()
                });
                (data_id, alloc, None)
            }
            TodoItem::Static(def_id) => {
                let section_name = tcx.codegen_fn_attrs(def_id).link_section;

                let alloc = tcx.eval_static_initializer(def_id).unwrap();

                let data_id = data_id_for_static(
                    tcx,
                    module,
                    def_id,
                    true,
                    alloc.inner().mutability == Mutability::Mut,
                );
                (data_id, alloc, section_name)
            }
        };

        let mut data = DataDescription::new();
        let alloc = alloc.inner();
        data.set_align(alloc.align.bytes());

        if let Some(section_name) = section_name {
            let (segment_name, section_name) = if tcx.sess.target.is_like_darwin {
                // See https://github.com/llvm/llvm-project/blob/main/llvm/lib/MC/MCSectionMachO.cpp
                let mut parts = section_name.as_str().split(',');
                let Some(segment_name) = parts.next() else {
                    tcx.dcx().fatal(format!(
                        "#[link_section = \"{}\"] is not valid for macos target: must be segment and section separated by comma",
                        section_name
                    ));
                };
                let Some(section_name) = parts.next() else {
                    tcx.dcx().fatal(format!(
                        "#[link_section = \"{}\"] is not valid for macos target: must be segment and section separated by comma",
                        section_name
                    ));
                };
                if section_name.len() > 16 {
                    tcx.dcx().fatal(format!(
                        "#[link_section = \"{}\"] is not valid for macos target: section name bigger than 16 bytes",
                        section_name
                    ));
                }
                let section_type = parts.next().unwrap_or("regular");
                if section_type != "regular" && section_type != "cstring_literals" {
                    tcx.dcx().fatal(format!(
                        "#[link_section = \"{}\"] is not supported: unsupported section type {}",
                        section_name, section_type,
                    ));
                }
                let _attrs = parts.next();
                if parts.next().is_some() {
                    tcx.dcx().fatal(format!(
                        "#[link_section = \"{}\"] is not valid for macos target: too many components",
                        section_name
                    ));
                }
                // FIXME(bytecodealliance/wasmtime#8901) set S_CSTRING_LITERALS section type when
                // cstring_literals is specified
                (segment_name, section_name)
            } else {
                ("", section_name.as_str())
            };
            data.set_segment_section(segment_name, section_name);
        }

        let bytes = alloc.inspect_with_uninit_and_ptr_outside_interpreter(0..alloc.len()).to_vec();
        data.define(bytes.into_boxed_slice());

        for &(offset, prov) in alloc.provenance().ptrs().iter() {
            let alloc_id = prov.alloc_id();
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
                GlobalAlloc::Function { instance, .. } => {
                    assert_eq!(addend, 0);
                    let func_id = crate::abi::import_function(tcx, module, instance);
                    let local_func_id = module.declare_func_in_data(func_id, &mut data);
                    data.write_function_addr(offset.bytes() as u32, local_func_id);
                    continue;
                }
                GlobalAlloc::Memory(target_alloc) => {
                    data_id_for_alloc_id(cx, module, alloc_id, target_alloc.inner().mutability)
                }
                GlobalAlloc::VTable(ty, dyn_ty) => data_id_for_vtable(
                    tcx,
                    cx,
                    module,
                    ty,
                    dyn_ty
                        .principal()
                        .map(|principal| tcx.instantiate_bound_regions_with_erased(principal)),
                ),
                GlobalAlloc::Static(def_id) => {
                    if tcx.codegen_fn_attrs(def_id).flags.contains(CodegenFnAttrFlags::THREAD_LOCAL)
                    {
                        tcx.dcx().fatal(format!(
                            "Allocation {:?} contains reference to TLS value {:?}",
                            alloc_id, def_id
                        ));
                    }

                    // Don't push a `TodoItem::Static` here, as it will cause statics used by
                    // multiple crates to be duplicated between them. It isn't necessary anyway,
                    // as it will get pushed by `codegen_static` when necessary.
                    data_id_for_static(
                        tcx, module, def_id, false,
                        // For a declaration the stated mutability doesn't matter.
                        false,
                    )
                }
            };

            let global_value = module.declare_data_in_data(data_id, &mut data);
            data.write_data_addr(offset.bytes() as u32, global_value, addend as i64);
        }

        module.define_data(data_id, &data).unwrap();
    }

    assert!(cx.todo.is_empty(), "{:?}", cx.todo);
}

/// Used only for intrinsic implementations that need a compile-time constant
///
/// All uses of this function are a bug inside stdarch. [`eval_mir_constant`]
/// should be used everywhere, but for some vendor intrinsics stdarch forgets
/// to wrap the immediate argument in `const {}`, necesitating this hack to get
/// the correct value at compile time instead.
pub(crate) fn mir_operand_get_const_val<'tcx>(
    fx: &FunctionCx<'_, '_, 'tcx>,
    operand: &Operand<'tcx>,
) -> Option<ScalarInt> {
    match operand {
        Operand::Constant(const_) => eval_mir_constant(fx, const_).0.try_to_scalar_int(),
        // FIXME(rust-lang/rust#85105): Casts like `IMM8 as u32` result in the const being stored
        // inside a temporary before being passed to the intrinsic requiring the const argument.
        // This code tries to find a single constant defining definition of the referenced local.
        Operand::Copy(place) | Operand::Move(place) => {
            if !place.projection.is_empty() {
                return None;
            }
            let mut computed_scalar_int = None;
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
                                    if computed_scalar_int.is_some() {
                                        return None; // local assigned twice
                                    }
                                    if !matches!(ty.kind(), ty::Uint(_) | ty::Int(_)) {
                                        return None;
                                    }
                                    let scalar_int = mir_operand_get_const_val(fx, operand)?;
                                    let scalar_int =
                                        match fx.layout_of(*ty).size.cmp(&scalar_int.size()) {
                                            Ordering::Equal => scalar_int,
                                            Ordering::Less => match ty.kind() {
                                                ty::Uint(_) => ScalarInt::try_from_uint(
                                                    scalar_int.to_uint(scalar_int.size()),
                                                    fx.layout_of(*ty).size,
                                                )
                                                .unwrap(),
                                                ty::Int(_) => ScalarInt::try_from_int(
                                                    scalar_int.to_int(scalar_int.size()),
                                                    fx.layout_of(*ty).size,
                                                )
                                                .unwrap(),
                                                _ => unreachable!(),
                                            },
                                            Ordering::Greater => return None,
                                        };
                                    computed_scalar_int = Some(scalar_int);
                                }
                                Rvalue::Use(operand) => {
                                    computed_scalar_int = mir_operand_get_const_val(fx, operand)
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
                        | StatementKind::BackwardIncompatibleDropHint { .. }
                        | StatementKind::Nop => {}
                    }
                }
                match &bb_data.terminator().kind {
                    TerminatorKind::Goto { .. }
                    | TerminatorKind::SwitchInt { .. }
                    | TerminatorKind::UnwindResume
                    | TerminatorKind::UnwindTerminate(_)
                    | TerminatorKind::Return
                    | TerminatorKind::Unreachable
                    | TerminatorKind::Drop { .. }
                    | TerminatorKind::Assert { .. } => {}
                    TerminatorKind::Yield { .. }
                    | TerminatorKind::CoroutineDrop
                    | TerminatorKind::FalseEdge { .. }
                    | TerminatorKind::FalseUnwind { .. } => unreachable!(),
                    TerminatorKind::InlineAsm { .. } => return None,
                    TerminatorKind::Call { destination, target: Some(_), .. }
                        if destination == place =>
                    {
                        return None;
                    }
                    TerminatorKind::TailCall { .. } => return None,
                    TerminatorKind::Call { .. } => {}
                }
            }
            computed_scalar_int
        }
    }
}
