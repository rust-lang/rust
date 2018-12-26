use std::borrow::Cow;
use std::iter;

use rustc::hir;
use rustc_target::spec::abi::Abi;

use crate::prelude::*;

#[derive(Copy, Clone, Debug)]
enum PassMode {
    NoPass,
    ByVal(Type),
    ByRef,
}

impl PassMode {
    fn get_param_ty(self, fx: &FunctionCx<impl Backend>) -> Type {
        match self {
            PassMode::NoPass => unimplemented!("pass mode nopass"),
            PassMode::ByVal(clif_type) => clif_type,
            PassMode::ByRef => fx.pointer_type,
        }
    }
}

fn get_pass_mode<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    abi: Abi,
    ty: Ty<'tcx>,
    is_return: bool,
) -> PassMode {
    assert!(!tcx
        .layout_of(ParamEnv::reveal_all().and(ty))
        .unwrap()
        .is_unsized());
    if let ty::Never = ty.sty {
        if is_return {
            PassMode::NoPass
        } else {
            PassMode::ByRef
        }
    } else if ty.sty == tcx.mk_unit().sty {
        if is_return {
            PassMode::NoPass
        } else {
            PassMode::ByRef
        }
    } else if let Some(ret_ty) = crate::common::clif_type_from_ty(tcx, ty) {
        PassMode::ByVal(ret_ty)
    } else {
        if abi == Abi::C {
            unimpl!(
                "Non scalars are not yet supported for \"C\" abi ({:?}) is_return: {:?}",
                ty,
                is_return
            );
        }
        PassMode::ByRef
    }
}

fn adjust_arg_for_abi<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    sig: FnSig<'tcx>,
    arg: CValue<'tcx>,
) -> Value {
    match get_pass_mode(fx.tcx, sig.abi, arg.layout().ty, false) {
        PassMode::NoPass => unimplemented!("pass mode nopass"),
        PassMode::ByVal(_) => arg.load_value(fx),
        PassMode::ByRef => arg.force_stack(fx),
    }
}

fn clif_sig_from_fn_ty<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    fn_ty: Ty<'tcx>,
) -> Signature {
    let sig = ty_fn_sig(tcx, fn_ty);
    if sig.variadic {
        unimpl!("Variadic function are not yet supported");
    }
    let (call_conv, inputs, output): (CallConv, Vec<Ty>, Ty) = match sig.abi {
        Abi::Rust => (CallConv::SystemV, sig.inputs().to_vec(), sig.output()),
        Abi::C => (CallConv::SystemV, sig.inputs().to_vec(), sig.output()),
        Abi::RustCall => {
            assert_eq!(sig.inputs().len(), 2);
            let extra_args = match sig.inputs().last().unwrap().sty {
                ty::Tuple(ref tupled_arguments) => tupled_arguments,
                _ => bug!("argument to function with \"rust-call\" ABI is not a tuple"),
            };
            let mut inputs: Vec<Ty> = vec![sig.inputs()[0]];
            inputs.extend(extra_args.into_iter());
            (CallConv::SystemV, inputs, sig.output())
        }
        Abi::System => bug!("system abi should be selected elsewhere"),
        Abi::RustIntrinsic => (CallConv::SystemV, sig.inputs().to_vec(), sig.output()),
        _ => unimplemented!("unsupported abi {:?}", sig.abi),
    };

    let inputs = inputs
        .into_iter()
        .filter_map(|ty| match get_pass_mode(tcx, sig.abi, ty, false) {
            PassMode::ByVal(clif_ty) => Some(clif_ty),
            PassMode::NoPass => unimplemented!("pass mode nopass"),
            PassMode::ByRef => Some(pointer_ty(tcx)),
        });

    let (params, returns) = match get_pass_mode(tcx, sig.abi, output, true) {
        PassMode::NoPass => (inputs.map(AbiParam::new).collect(), vec![]),
        PassMode::ByVal(ret_ty) => (
            inputs.map(AbiParam::new).collect(),
            vec![AbiParam::new(ret_ty)],
        ),
        PassMode::ByRef => {
            (
                Some(pointer_ty(tcx)) // First param is place to put return val
                    .into_iter()
                    .chain(inputs)
                    .map(AbiParam::new)
                    .collect(),
                vec![],
            )
        }
    };

    Signature {
        params,
        returns,
        call_conv,
    }
}

pub fn ty_fn_sig<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: Ty<'tcx>) -> ty::FnSig<'tcx> {
    let sig = match ty.sty {
        ty::FnDef(..) |
        // Shims currently have type TyFnPtr. Not sure this should remain.
        ty::FnPtr(_) => ty.fn_sig(tcx),
        ty::Closure(def_id, substs) => {
            let sig = substs.closure_sig(def_id, tcx);

            let env_ty = tcx.closure_env_ty(def_id, substs).unwrap();
            sig.map_bound(|sig| tcx.mk_fn_sig(
                iter::once(*env_ty.skip_binder()).chain(sig.inputs().iter().cloned()),
                sig.output(),
                sig.variadic,
                sig.unsafety,
                sig.abi
            ))
        }
        ty::Generator(def_id, substs, _) => {
            let sig = substs.poly_sig(def_id, tcx);

            let env_region = ty::ReLateBound(ty::INNERMOST, ty::BrEnv);
            let env_ty = tcx.mk_mut_ref(tcx.mk_region(env_region), ty);

            sig.map_bound(|sig| {
                let state_did = tcx.lang_items().gen_state().unwrap();
                let state_adt_ref = tcx.adt_def(state_did);
                let state_substs = tcx.intern_substs(&[
                    sig.yield_ty.into(),
                    sig.return_ty.into(),
                ]);
                let ret_ty = tcx.mk_adt(state_adt_ref, state_substs);

                tcx.mk_fn_sig(iter::once(env_ty),
                    ret_ty,
                    false,
                    hir::Unsafety::Normal,
                    Abi::Rust
                )
            })
        }
        _ => bug!("unexpected type {:?} to ty_fn_sig", ty)
    };
    tcx.normalize_erasing_late_bound_regions(ParamEnv::reveal_all(), &sig)
}

pub fn get_function_name_and_sig<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    inst: Instance<'tcx>,
) -> (String, Signature) {
    assert!(!inst.substs.needs_infer() && !inst.substs.has_param_types());
    let fn_ty = inst.ty(tcx);
    let sig = clif_sig_from_fn_ty(tcx, fn_ty);
    (tcx.symbol_name(inst).as_str().to_string(), sig)
}

impl<'a, 'tcx: 'a, B: Backend + 'a> FunctionCx<'a, 'tcx, B> {
    /// Instance must be monomorphized
    pub fn get_function_id(&mut self, inst: Instance<'tcx>) -> FuncId {
        let (name, sig) = get_function_name_and_sig(self.tcx, inst);
        self.module
            .declare_function(&name, Linkage::Import, &sig)
            .unwrap()
    }

    /// Instance must be monomorphized
    pub fn get_function_ref(&mut self, inst: Instance<'tcx>) -> FuncRef {
        let func_id = self.get_function_id(inst);
        self.module
            .declare_func_in_func(func_id, &mut self.bcx.func)
    }

    fn lib_call(
        &mut self,
        name: &str,
        input_tys: Vec<types::Type>,
        output_ty: Option<types::Type>,
        args: &[Value],
    ) -> Option<Value> {
        let sig = Signature {
            params: input_tys.iter().cloned().map(AbiParam::new).collect(),
            returns: output_ty
                .map(|output_ty| vec![AbiParam::new(output_ty)])
                .unwrap_or(Vec::new()),
            call_conv: CallConv::SystemV,
        };
        let func_id = self
            .module
            .declare_function(&name, Linkage::Import, &sig)
            .unwrap();
        let func_ref = self
            .module
            .declare_func_in_func(func_id, &mut self.bcx.func);
        let call_inst = self.bcx.ins().call(func_ref, args);
        if output_ty.is_none() {
            return None;
        }
        let results = self.bcx.inst_results(call_inst);
        assert_eq!(results.len(), 1);
        Some(results[0])
    }

    pub fn easy_call(
        &mut self,
        name: &str,
        args: &[CValue<'tcx>],
        return_ty: Ty<'tcx>,
    ) -> CValue<'tcx> {
        let (input_tys, args): (Vec<_>, Vec<_>) = args
            .into_iter()
            .map(|arg| {
                (
                    self.clif_type(arg.layout().ty).unwrap(),
                    arg.load_value(self),
                )
            })
            .unzip();
        let return_layout = self.layout_of(return_ty);
        let return_ty = if let ty::Tuple(tup) = return_ty.sty {
            if !tup.is_empty() {
                bug!("easy_call( (...) -> <non empty tuple> ) is not allowed");
            }
            None
        } else {
            Some(self.clif_type(return_ty).unwrap())
        };
        if let Some(val) = self.lib_call(name, input_tys, return_ty, &args) {
            CValue::ByVal(val, return_layout)
        } else {
            CValue::ByRef(self.bcx.ins().iconst(self.pointer_type, 0), return_layout)
        }
    }

    fn self_sig(&self) -> FnSig<'tcx> {
        ty_fn_sig(self.tcx, self.instance.ty(self.tcx))
    }

    fn return_type(&self) -> Ty<'tcx> {
        self.self_sig().output()
    }
}

fn add_local_comment<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    msg: &str,
    local: mir::Local,
    local_field: Option<usize>,
    param: Option<Value>,
    pass_mode: Option<PassMode>,
    ssa: crate::analyze::Flags,
    ty: Ty<'tcx>,
) {
    let local_field = if let Some(local_field) = local_field {
        Cow::Owned(format!(".{}", local_field))
    } else {
        Cow::Borrowed("")
    };
    let param = if let Some(param) = param {
        Cow::Owned(format!("= {:?}", param))
    } else {
        Cow::Borrowed("-")
    };
    let pass_mode = if let Some(pass_mode) = pass_mode {
        Cow::Owned(format!("{:?}", pass_mode))
    } else {
        Cow::Borrowed("-")
    };
    fx.add_global_comment(format!(
        "{msg:5} {local:>3}{local_field:<5} {param:10} {pass_mode:20} {ssa:10} {ty:?}",
        msg=msg, local=format!("{:?}", local), local_field=local_field, param=param, pass_mode=pass_mode, ssa=format!("{:?}", ssa), ty=ty,
    ));
}

fn add_local_header_comment(fx: &mut FunctionCx<impl Backend>) {
    fx.add_global_comment(format!("msg   loc.idx    param    pass mode            ssa flags  ty"));
}

fn local_place<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    local: Local,
    layout: TyLayout<'tcx>,
    is_ssa: bool,
) -> CPlace<'tcx> {
    let place = if is_ssa {
        fx.bcx.declare_var(mir_var(local), fx.clif_type(layout.ty).unwrap());
        CPlace::Var(local, layout)
    } else {
        let stack_slot = fx.bcx.create_stack_slot(StackSlotData {
            kind: StackSlotKind::ExplicitSlot,
            size: layout.size.bytes() as u32,
            offset: None,
        });

        CPlace::from_stack_slot(fx, stack_slot, layout.ty)
    };

    debug_assert!(fx.local_map.insert(local, place).is_none());
    fx.local_map[&local]
}

fn param_to_cvalue<'a, 'tcx: 'a>(fx: &FunctionCx<'a, 'tcx, impl Backend>, ebb_param: Value, layout: TyLayout<'tcx>) -> CValue<'tcx> {
    match get_pass_mode(fx.tcx, fx.self_sig().abi, layout.ty, false) {
        PassMode::NoPass => unimplemented!("pass mode nopass"),
        PassMode::ByVal(_) => CValue::ByVal(ebb_param, layout),
        PassMode::ByRef => CValue::ByRef(ebb_param, layout),
    }
}

pub fn codegen_fn_prelude<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    start_ebb: Ebb,
) {
    let ssa_analyzed = crate::analyze::analyze(fx);
    fx.add_global_comment(format!("ssa {:?}", ssa_analyzed));

    let ret_layout = fx.layout_of(fx.return_type());
    let output_pass_mode = get_pass_mode(fx.tcx, fx.self_sig().abi, fx.return_type(), true);
    let ret_param = match output_pass_mode {
        PassMode::NoPass => None,
        PassMode::ByVal(_) => None,
        PassMode::ByRef => Some(fx.bcx.append_ebb_param(start_ebb, fx.pointer_type)),
    };

    enum ArgKind {
        Normal(Value),
        Spread(Vec<Value>),
    }

    let func_params = fx
        .mir
        .args_iter()
        .map(|local| {
            let arg_ty = fx.monomorphize(&fx.mir.local_decls[local].ty);

            // Adapted from https://github.com/rust-lang/rust/blob/145155dc96757002c7b2e9de8489416e2fdbbd57/src/librustc_codegen_llvm/mir/mod.rs#L442-L482
            if Some(local) == fx.mir.spread_arg {
                // This argument (e.g. the last argument in the "rust-call" ABI)
                // is a tuple that was spread at the ABI level and now we have
                // to reconstruct it into a tuple local variable, from multiple
                // individual function arguments.

                let tupled_arg_tys = match arg_ty.sty {
                    ty::Tuple(ref tys) => tys,
                    _ => bug!("spread argument isn't a tuple?! but {:?}", arg_ty),
                };

                let mut ebb_params = Vec::new();
                for (i, arg_ty) in tupled_arg_tys.iter().enumerate() {
                    let pass_mode = get_pass_mode(fx.tcx, fx.self_sig().abi, arg_ty, false);;
                    let clif_type = pass_mode.get_param_ty(fx);
                    let ebb_param = fx.bcx.append_ebb_param(start_ebb, clif_type);
                    add_local_comment(fx, "arg", local, Some(i), Some(ebb_param), Some(pass_mode), ssa_analyzed[&local], arg_ty);
                    ebb_params.push(ebb_param);
                }

                (local, ArgKind::Spread(ebb_params), arg_ty)
            } else {
                let clif_type =
                    get_pass_mode(fx.tcx, fx.self_sig().abi, arg_ty, false).get_param_ty(fx);
                let ebb_param = fx.bcx.append_ebb_param(start_ebb, clif_type);
                let pass_mode = get_pass_mode(fx.tcx, fx.self_sig().abi, arg_ty, false);
                add_local_comment(fx, "arg", local, None, Some(ebb_param), Some(pass_mode), ssa_analyzed[&local], arg_ty);
                (
                    local,
                    ArgKind::Normal(ebb_param),
                    arg_ty,
                )
            }
        })
        .collect::<Vec<(Local, ArgKind, Ty)>>();

    fx.bcx.switch_to_block(start_ebb);

    match output_pass_mode {
        PassMode::NoPass => {
            let null = fx.bcx.ins().iconst(fx.pointer_type, 0);
            fx.local_map.insert(
                RETURN_PLACE,
                CPlace::Addr(null, None, fx.layout_of(fx.return_type())),
            );
        }
        PassMode::ByVal(ret_ty) => {
            fx.bcx.declare_var(mir_var(RETURN_PLACE), ret_ty);
            fx.local_map
                .insert(RETURN_PLACE, CPlace::Var(RETURN_PLACE, ret_layout));
        }
        PassMode::ByRef => {
            fx.local_map.insert(
                RETURN_PLACE,
                CPlace::Addr(ret_param.unwrap(), None, ret_layout),
            );
        }
    }

    add_local_header_comment(fx);
    add_local_comment(fx, "ret", RETURN_PLACE, None, ret_param, Some(output_pass_mode), ssa_analyzed[&RETURN_PLACE], ret_layout.ty);

    for (local, arg_kind, ty) in func_params {
        let layout = fx.layout_of(ty);

        let is_ssa = !ssa_analyzed
            .get(&local)
            .unwrap()
            .contains(crate::analyze::Flags::NOT_SSA);

        let place = local_place(fx, local, layout, is_ssa);

        match arg_kind {
            ArgKind::Normal(ebb_param) => {
                let cvalue = param_to_cvalue(fx, ebb_param, layout);
                place.write_cvalue(fx, cvalue);
            }
            ArgKind::Spread(ebb_params) => {
                for (i, ebb_param) in ebb_params.into_iter().enumerate() {
                    let sub_place = place.place_field(fx, mir::Field::new(i));
                    let cvalue = param_to_cvalue(fx, ebb_param, sub_place.layout());
                    sub_place.write_cvalue(fx, cvalue);
                }
            }
        }
    }

    for local in fx.mir.vars_and_temps_iter() {
        let ty = fx.mir.local_decls[local].ty;
        let layout = fx.layout_of(ty);

        add_local_comment(fx, "local", local, None, None, None, ssa_analyzed[&local], ty);

        let is_ssa = !ssa_analyzed
            .get(&local)
            .unwrap()
            .contains(crate::analyze::Flags::NOT_SSA);

        local_place(fx, local, layout, is_ssa);
    }

    fx.bcx
        .ins()
        .jump(*fx.ebb_map.get(&START_BLOCK).unwrap(), &[]);
}

pub fn codegen_terminator_call<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    func: &Operand<'tcx>,
    args: &[Operand<'tcx>],
    destination: &Option<(Place<'tcx>, BasicBlock)>,
) {
    let fn_ty = fx.monomorphize(&func.ty(fx.mir, fx.tcx));
    let sig = ty_fn_sig(fx.tcx, fn_ty);

    // Unpack arguments tuple for closures
    let args = if sig.abi == Abi::RustCall {
        assert_eq!(args.len(), 2, "rust-call abi requires two arguments");
        let self_arg = trans_operand(fx, &args[0]);
        let pack_arg = trans_operand(fx, &args[1]);
        let mut args = Vec::new();
        args.push(self_arg);
        match pack_arg.layout().ty.sty {
            ty::Tuple(ref tupled_arguments) => {
                for (i, _) in tupled_arguments.iter().enumerate() {
                    args.push(pack_arg.value_field(fx, mir::Field::new(i)));
                }
            }
            _ => bug!("argument to function with \"rust-call\" ABI is not a tuple"),
        }
        args
    } else {
        args.into_iter()
            .map(|arg| trans_operand(fx, arg))
            .collect::<Vec<_>>()
    };

    let destination = destination
        .as_ref()
        .map(|&(ref place, bb)| (trans_place(fx, place), bb));

    if let ty::FnDef(def_id, substs) = fn_ty.sty {
        let instance = ty::Instance::resolve(
            fx.tcx,
            ty::ParamEnv::reveal_all(),
            def_id,
            substs,
        ).unwrap();

        match instance.def {
            InstanceDef::Intrinsic(_) => {
                crate::intrinsics::codegen_intrinsic_call(fx, def_id, substs, args, destination);
                return;
            }
            InstanceDef::DropGlue(_, None) => {
                // empty drop glue - a nop.
                let (_, dest) = destination.expect("Non terminating drop_in_place_real???");
                let ret_ebb = fx.get_ebb(dest);
                fx.bcx.ins().jump(ret_ebb, &[]);
                return;
            }
            _ => {}
        }
    }

    codegen_call_inner(
        fx,
        Some(func),
        fn_ty,
        args,
        destination.map(|(place, _)| place),
    );

    if let Some((_, dest)) = destination {
        let ret_ebb = fx.get_ebb(dest);
        fx.bcx.ins().jump(ret_ebb, &[]);
    } else {
        trap_unreachable(&mut fx.bcx);
    }
}

pub fn codegen_call_inner<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    func: Option<&Operand<'tcx>>,
    fn_ty: Ty<'tcx>,
    args: Vec<CValue<'tcx>>,
    ret_place: Option<CPlace<'tcx>>,
) {
    let sig = ty_fn_sig(fx.tcx, fn_ty);

    let ret_layout = fx.layout_of(sig.output());

    let output_pass_mode = get_pass_mode(fx.tcx, sig.abi, sig.output(), true);
    let return_ptr = match output_pass_mode {
        PassMode::NoPass => None,
        PassMode::ByRef => match ret_place {
            Some(ret_place) => Some(ret_place.expect_addr()),
            None => Some(fx.bcx.ins().iconst(fx.pointer_type, 0)),
        },
        PassMode::ByVal(_) => None,
    };

    let instance = match fn_ty.sty {
        ty::FnDef(def_id, substs) => {
            Some(Instance::resolve(fx.tcx, ParamEnv::reveal_all(), def_id, substs).unwrap())
        }
        _ => None,
    };

    let func_ref: Option<Value>; // Indirect call target

    let first_arg = {
        if let Some(Instance {
            def: InstanceDef::Virtual(_, idx),
            ..
        }) = instance
        {
            let (ptr, method) = crate::vtable::get_ptr_and_method_ref(fx, args[0], idx);
            func_ref = Some(method);
            Some(ptr)
        } else {
            func_ref = if instance.is_none() {
                let func = trans_operand(fx, func.expect("indirect call without func Operand"));
                Some(func.load_value(fx))
            } else {
                None
            };

            args.get(0).map(|arg| adjust_arg_for_abi(fx, sig, *arg))
        }
        .into_iter()
    };

    let call_args: Vec<Value> = return_ptr
        .into_iter()
        .chain(first_arg)
        .chain(
            args.into_iter()
                .skip(1)
                .map(|arg| adjust_arg_for_abi(fx, sig, arg)),
        )
        .collect::<Vec<_>>();

    let sig = fx.bcx.import_signature(clif_sig_from_fn_ty(fx.tcx, fn_ty));
    let call_inst = if let Some(func_ref) = func_ref {
        fx.bcx.ins().call_indirect(sig, func_ref, &call_args)
    } else {
        let func_ref = fx.get_function_ref(instance.expect("non-indirect call on non-FnDef type"));
        fx.bcx.ins().call(func_ref, &call_args)
    };

    match output_pass_mode {
        PassMode::NoPass => {}
        PassMode::ByVal(_) => {
            if let Some(ret_place) = ret_place {
                let results = fx.bcx.inst_results(call_inst);
                ret_place.write_cvalue(fx, CValue::ByVal(results[0], ret_layout));
            }
        }
        PassMode::ByRef => {}
    }
}

pub fn codegen_return(fx: &mut FunctionCx<impl Backend>) {
    match get_pass_mode(fx.tcx, fx.self_sig().abi, fx.return_type(), true) {
        PassMode::NoPass | PassMode::ByRef => {
            fx.bcx.ins().return_(&[]);
        }
        PassMode::ByVal(_) => {
            let place = fx.get_local_place(RETURN_PLACE);
            let ret_val = place.to_cvalue(fx).load_value(fx);
            fx.bcx.ins().return_(&[ret_val]);
        }
    }
}
