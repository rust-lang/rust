use std::iter;

use rustc::hir;
use rustc_target::spec::abi::Abi;

use crate::prelude::*;

#[derive(Debug)]
enum PassMode {
    NoPass,
    ByVal(Type),
    ByRef,
}

impl PassMode {
    fn get_param_ty(self, fx: &FunctionCx<impl Backend>) -> Type {
        match self {
            PassMode::NoPass => unimplemented!("pass mode nopass"),
            PassMode::ByVal(cton_type) => cton_type,
            PassMode::ByRef => fx.module.pointer_type(),
        }
    }
}

fn get_pass_mode<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    abi: Abi,
    ty: Ty<'tcx>,
    is_return: bool,
) -> PassMode {
    assert!(
        !tcx.layout_of(ParamEnv::reveal_all().and(ty))
            .unwrap()
            .is_unsized()
    );
    if ty.sty == tcx.mk_unit().sty {
        if is_return {
            //if false {
            PassMode::NoPass
        } else {
            PassMode::ByRef
        }
    } else if let Some(ret_ty) = crate::common::cton_type_from_ty(tcx, ty) {
        PassMode::ByVal(ret_ty)
    } else {
        if abi == Abi::C {
            unimplemented!("Non scalars are not yet supported for \"C\" abi");
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

pub fn cton_sig_from_fn_ty<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    fn_ty: Ty<'tcx>,
) -> Signature {
    let sig = ty_fn_sig(tcx, fn_ty);
    assert!(!sig.variadic, "Variadic function are not yet supported");
    let (call_conv, inputs, output): (CallConv, Vec<Ty>, Ty) = match sig.abi {
        Abi::Rust => (CallConv::Fast, sig.inputs().to_vec(), sig.output()),
        Abi::C => (CallConv::SystemV, sig.inputs().to_vec(), sig.output()),
        Abi::RustCall => {
            assert_eq!(sig.inputs().len(), 2);
            let extra_args = match sig.inputs().last().unwrap().sty {
                ty::Tuple(ref tupled_arguments) => tupled_arguments,
                _ => bug!("argument to function with \"rust-call\" ABI is not a tuple"),
            };
            let mut inputs: Vec<Ty> = vec![sig.inputs()[0]];
            inputs.extend(extra_args.into_iter());
            (CallConv::Fast, inputs, sig.output())
        }
        Abi::System => bug!("system abi should be selected elsewhere"),
        Abi::RustIntrinsic => (CallConv::SystemV, sig.inputs().to_vec(), sig.output()),
        _ => unimplemented!("unsupported abi {:?}", sig.abi),
    };

    let inputs = inputs
        .into_iter()
        .filter_map(|ty| match get_pass_mode(tcx, sig.abi, ty, false) {
            PassMode::ByVal(cton_ty) => Some(cton_ty),
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

fn ty_fn_sig<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: Ty<'tcx>) -> ty::FnSig<'tcx> {
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
    let sig = cton_sig_from_fn_ty(tcx, fn_ty);
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
                    self.cton_type(arg.layout().ty).unwrap(),
                    arg.load_value(self),
                )
            }).unzip();
        let return_layout = self.layout_of(return_ty);
        let return_ty = if let ty::Tuple(tup) = return_ty.sty {
            if !tup.is_empty() {
                bug!("easy_call( (...) -> <non empty tuple> ) is not allowed");
            }
            None
        } else {
            Some(self.cton_type(return_ty).unwrap())
        };
        if let Some(val) = self.lib_call(name, input_tys, return_ty, &args) {
            CValue::ByVal(val, return_layout)
        } else {
            CValue::ByRef(
                self.bcx.ins().iconst(self.module.pointer_type(), 0),
                return_layout,
            )
        }
    }

    fn self_sig(&self) -> FnSig<'tcx> {
        ty_fn_sig(self.tcx, self.instance.ty(self.tcx))
    }

    fn return_type(&self) -> Ty<'tcx> {
        self.self_sig().output()
    }
}

pub fn codegen_fn_prelude<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    start_ebb: Ebb,
) {
    let ssa_analyzed = crate::analyze::analyze(fx);

    let ret_layout = fx.layout_of(fx.return_type());
    let output_pass_mode = get_pass_mode(fx.tcx, fx.self_sig().abi, fx.return_type(), true);
    let ret_param = match output_pass_mode {
        PassMode::NoPass => None,
        PassMode::ByVal(_) => None,
        PassMode::ByRef => Some(fx.bcx.append_ebb_param(start_ebb, fx.module.pointer_type())),
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
                for arg_ty in tupled_arg_tys.iter() {
                    let cton_type =
                        get_pass_mode(fx.tcx, fx.self_sig().abi, arg_ty, false).get_param_ty(fx);
                    ebb_params.push(fx.bcx.append_ebb_param(start_ebb, cton_type));
                }

                (local, ArgKind::Spread(ebb_params), arg_ty)
            } else {
                let cton_type =
                    get_pass_mode(fx.tcx, fx.self_sig().abi, arg_ty, false).get_param_ty(fx);
                (
                    local,
                    ArgKind::Normal(fx.bcx.append_ebb_param(start_ebb, cton_type)),
                    arg_ty,
                )
            }
        }).collect::<Vec<(Local, ArgKind, Ty)>>();

    fx.bcx.switch_to_block(start_ebb);

    fx.top_nop = Some(fx.bcx.ins().nop());
    fx.add_global_comment(format!("ssa {:?}", ssa_analyzed));

    match output_pass_mode {
        PassMode::NoPass => {
            let null = fx.bcx.ins().iconst(fx.module.pointer_type(), 0);
            //unimplemented!("pass mode nopass");
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

    for (local, arg_kind, ty) in func_params {
        let layout = fx.layout_of(ty);

        if let ArgKind::Normal(ebb_param) = arg_kind {
            if !ssa_analyzed
                .get(&local)
                .unwrap()
                .contains(crate::analyze::Flags::NOT_SSA)
            {
                fx.bcx
                    .declare_var(mir_var(local), fx.cton_type(ty).unwrap());
                match get_pass_mode(fx.tcx, fx.self_sig().abi, ty, false) {
                    PassMode::NoPass => unimplemented!("pass mode nopass"),
                    PassMode::ByVal(_) => fx.bcx.def_var(mir_var(local), ebb_param),
                    PassMode::ByRef => {
                        let val = CValue::ByRef(ebb_param, fx.layout_of(ty)).load_value(fx);
                        fx.bcx.def_var(mir_var(local), val);
                    }
                }
                fx.local_map.insert(local, CPlace::Var(local, layout));
                continue;
            }
        }

        let stack_slot = fx.bcx.create_stack_slot(StackSlotData {
            kind: StackSlotKind::ExplicitSlot,
            size: layout.size.bytes() as u32,
            offset: None,
        });

        let place = CPlace::from_stack_slot(fx, stack_slot, ty);

        match arg_kind {
            ArgKind::Normal(ebb_param) => match get_pass_mode(fx.tcx, fx.self_sig().abi, ty, false)
            {
                PassMode::NoPass => unimplemented!("pass mode nopass"),
                PassMode::ByVal(_) => {
                    place.write_cvalue(fx, CValue::ByVal(ebb_param, place.layout()))
                }
                PassMode::ByRef => place.write_cvalue(fx, CValue::ByRef(ebb_param, place.layout())),
            },
            ArgKind::Spread(ebb_params) => {
                for (i, ebb_param) in ebb_params.into_iter().enumerate() {
                    let sub_place = place.place_field(fx, mir::Field::new(i));
                    match get_pass_mode(fx.tcx, fx.self_sig().abi, sub_place.layout().ty, false) {
                        PassMode::NoPass => unimplemented!("pass mode nopass"),
                        PassMode::ByVal(_) => {
                            sub_place.write_cvalue(fx, CValue::ByVal(ebb_param, sub_place.layout()))
                        }
                        PassMode::ByRef => {
                            sub_place.write_cvalue(fx, CValue::ByRef(ebb_param, sub_place.layout()))
                        }
                    }
                }
            }
        }
        fx.local_map.insert(local, place);
    }

    for local in fx.mir.vars_and_temps_iter() {
        let ty = fx.mir.local_decls[local].ty;
        let layout = fx.layout_of(ty);

        let place = if ssa_analyzed
            .get(&local)
            .unwrap()
            .contains(crate::analyze::Flags::NOT_SSA)
        {
            let stack_slot = fx.bcx.create_stack_slot(StackSlotData {
                kind: StackSlotKind::ExplicitSlot,
                size: layout.size.bytes() as u32,
                offset: None,
            });
            CPlace::from_stack_slot(fx, stack_slot, ty)
        } else {
            fx.bcx
                .declare_var(mir_var(local), fx.cton_type(ty).unwrap());
            CPlace::Var(local, layout)
        };

        fx.local_map.insert(local, place);
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
    let fn_ty = fx.monomorphize(&func.ty(&fx.mir.local_decls, fx.tcx));
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

    if !codegen_intrinsic_call(fx, fn_ty, &args, destination) {
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
            fx.bcx.ins().trap(TrapCode::User(!0));
        }
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
            None => Some(fx.bcx.ins().iconst(fx.module.pointer_type(), 0)),
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
        }.into_iter()
    };

    let call_args: Vec<Value> = return_ptr
        .into_iter()
        .chain(first_arg)
        .chain(
            args.into_iter()
                .skip(1)
                .map(|arg| adjust_arg_for_abi(fx, sig, arg)),
        ).collect::<Vec<_>>();

    let sig = fx.bcx.import_signature(cton_sig_from_fn_ty(fx.tcx, fn_ty));
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

fn codegen_intrinsic_call<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    fn_ty: Ty<'tcx>,
    args: &[CValue<'tcx>],
    destination: Option<(CPlace<'tcx>, BasicBlock)>,
) -> bool {
    if let ty::FnDef(def_id, substs) = fn_ty.sty {
        let sig = ty_fn_sig(fx.tcx, fn_ty);

        if sig.abi == Abi::RustIntrinsic {
            let intrinsic = fx.tcx.item_name(def_id).as_str();
            let intrinsic = &intrinsic[..];

            let ret = match destination {
                Some((place, _)) => place,
                None => {
                    // Insert non returning intrinsics here
                    match intrinsic {
                        "abort" => {
                            fx.bcx.ins().trap(TrapCode::User(!0 - 1));
                        }
                        "unreachable" => {
                            fx.bcx.ins().trap(TrapCode::User(!0 - 1));
                        }
                        _ => unimplemented!("unsupported instrinsic {}", intrinsic),
                    }
                    return true;
                }
            };

            let nil_ty = fx.tcx.mk_unit();
            let u64_layout = fx.layout_of(fx.tcx.types.u64);
            let usize_layout = fx.layout_of(fx.tcx.types.usize);

            match intrinsic {
                "assume" => {
                    assert_eq!(args.len(), 1);
                }
                "arith_offset" => {
                    assert_eq!(args.len(), 2);
                    let base = args[0].load_value(fx);
                    let offset = args[1].load_value(fx);
                    let res = fx.bcx.ins().iadd(base, offset);
                    let res = CValue::ByVal(res, ret.layout());
                    ret.write_cvalue(fx, res);
                }
                "likely" | "unlikely" => {
                    assert_eq!(args.len(), 1);
                    ret.write_cvalue(fx, args[0]);
                }
                "copy" | "copy_nonoverlapping" => {
                    let elem_ty = substs.type_at(0);
                    let elem_size: u64 = fx.layout_of(elem_ty).size.bytes();
                    let elem_size = fx
                        .bcx
                        .ins()
                        .iconst(fx.module.pointer_type(), elem_size as i64);
                    assert_eq!(args.len(), 3);
                    let src = args[0];
                    let dst = args[1];
                    let count = args[2].load_value(fx);
                    let byte_amount = fx.bcx.ins().imul(count, elem_size);
                    fx.easy_call(
                        "memmove",
                        &[dst, src, CValue::ByVal(byte_amount, usize_layout)],
                        nil_ty,
                    );
                }
                "discriminant_value" => {
                    assert_eq!(args.len(), 1);
                    let discr = crate::base::trans_get_discriminant(fx, args[0], ret.layout());
                    ret.write_cvalue(fx, discr);
                }
                "size_of" => {
                    assert_eq!(args.len(), 0);
                    let size_of = fx.layout_of(substs.type_at(0)).size.bytes();
                    let size_of = CValue::const_val(fx, usize_layout.ty, size_of as i64);
                    ret.write_cvalue(fx, size_of);
                }
                "size_of_val" => {
                    assert_eq!(args.len(), 1);
                    let layout = fx.layout_of(substs.type_at(0));
                    let size = match &layout.ty.sty {
                        _ if !layout.is_unsized() => {
                            fx.bcx.ins().iconst(fx.module.pointer_type(), layout.size.bytes() as i64)
                        }
                        ty::Slice(elem) => {
                            let len = args[0].load_value_pair(fx).1;
                            let elem_size = fx.layout_of(elem).size.bytes();
                            fx.bcx.ins().imul_imm(len, elem_size as i64)
                        }
                        ty::Dynamic(..) => crate::vtable::size_of_obj(fx, args[0]),
                        ty => unimplemented!("size_of_val for {:?}", ty),
                    };
                    ret.write_cvalue(fx, CValue::ByVal(size, usize_layout));
                }
                "type_id" => {
                    assert_eq!(args.len(), 0);
                    let type_id = fx.tcx.type_id_hash(substs.type_at(0));
                    let type_id = CValue::const_val(fx, u64_layout.ty, type_id as i64);
                    ret.write_cvalue(fx, type_id);
                }
                "min_align_of" => {
                    assert_eq!(args.len(), 0);
                    let min_align = fx.layout_of(substs.type_at(0)).align.abi();
                    let min_align = CValue::const_val(fx, usize_layout.ty, min_align as i64);
                    ret.write_cvalue(fx, min_align);
                }
                _ if intrinsic.starts_with("unchecked_") => {
                    assert_eq!(args.len(), 2);
                    let bin_op = match intrinsic {
                        "unchecked_div" => BinOp::Div,
                        "unchecked_rem" => BinOp::Rem,
                        "unchecked_shl" => BinOp::Shl,
                        "unchecked_shr" => BinOp::Shr,
                        _ => unimplemented!("intrinsic {}", intrinsic),
                    };
                    let res = match ret.layout().ty.sty {
                        ty::Uint(_) => crate::base::trans_int_binop(
                            fx,
                            bin_op,
                            args[0],
                            args[1],
                            ret.layout().ty,
                            false,
                        ),
                        ty::Int(_) => crate::base::trans_int_binop(
                            fx,
                            bin_op,
                            args[0],
                            args[1],
                            ret.layout().ty,
                            true,
                        ),
                        _ => panic!(),
                    };
                    ret.write_cvalue(fx, res);
                }
                _ if intrinsic.ends_with("_with_overflow") => {
                    assert_eq!(args.len(), 2);
                    assert_eq!(args[0].layout().ty, args[1].layout().ty);
                    let bin_op = match intrinsic {
                        "add_with_overflow" => BinOp::Add,
                        "sub_with_overflow" => BinOp::Sub,
                        "mul_with_overflow" => BinOp::Mul,
                        _ => unimplemented!("intrinsic {}", intrinsic),
                    };
                    let res = match args[0].layout().ty.sty {
                        ty::Uint(_) => crate::base::trans_checked_int_binop(
                            fx,
                            bin_op,
                            args[0],
                            args[1],
                            ret.layout().ty,
                            false,
                        ),
                        ty::Int(_) => crate::base::trans_checked_int_binop(
                            fx,
                            bin_op,
                            args[0],
                            args[1],
                            ret.layout().ty,
                            true,
                        ),
                        _ => panic!(),
                    };
                    ret.write_cvalue(fx, res);
                }
                _ if intrinsic.starts_with("overflowing_") => {
                    assert_eq!(args.len(), 2);
                    assert_eq!(args[0].layout().ty, args[1].layout().ty);
                    let bin_op = match intrinsic {
                        "overflowing_add" => BinOp::Add,
                        "overflowing_sub" => BinOp::Sub,
                        "overflowing_mul" => BinOp::Mul,
                        _ => unimplemented!("intrinsic {}", intrinsic),
                    };
                    let res = match args[0].layout().ty.sty {
                        ty::Uint(_) => crate::base::trans_int_binop(
                            fx,
                            bin_op,
                            args[0],
                            args[1],
                            ret.layout().ty,
                            false,
                        ),
                        ty::Int(_) => crate::base::trans_int_binop(
                            fx,
                            bin_op,
                            args[0],
                            args[1],
                            ret.layout().ty,
                            true,
                        ),
                        _ => panic!(),
                    };
                    ret.write_cvalue(fx, res);
                }
                "offset" => {
                    assert_eq!(args.len(), 2);
                    let base = args[0].load_value(fx);
                    let offset = args[1].load_value(fx);
                    let res = fx.bcx.ins().iadd(base, offset);
                    ret.write_cvalue(fx, CValue::ByVal(res, args[0].layout()));
                }
                "transmute" => {
                    assert_eq!(args.len(), 1);
                    let src_ty = substs.type_at(0);
                    let dst_ty = substs.type_at(1);
                    assert_eq!(args[0].layout().ty, src_ty);
                    let addr = args[0].force_stack(fx);
                    let dst_layout = fx.layout_of(dst_ty);
                    ret.write_cvalue(fx, CValue::ByRef(addr, dst_layout))
                }
                "uninit" => {
                    assert_eq!(args.len(), 0);
                    let ty = substs.type_at(0);
                    let layout = fx.layout_of(ty);
                    let stack_slot = fx.bcx.create_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size: layout.size.bytes() as u32,
                        offset: None,
                    });

                    let uninit_place = CPlace::from_stack_slot(fx, stack_slot, ty);
                    let uninit_val = uninit_place.to_cvalue(fx);
                    ret.write_cvalue(fx, uninit_val);
                }
                "ctlz" | "ctlz_nonzero" => {
                    assert_eq!(args.len(), 1);
                    let arg = args[0].load_value(fx);
                    let res = CValue::ByVal(fx.bcx.ins().clz(arg), args[0].layout());
                    ret.write_cvalue(fx, res);
                }
                "cttz" | "cttz_nonzero" => {
                    assert_eq!(args.len(), 1);
                    let arg = args[0].load_value(fx);
                    let res = CValue::ByVal(fx.bcx.ins().clz(arg), args[0].layout());
                    ret.write_cvalue(fx, res);
                }
                "ctpop" => {
                    assert_eq!(args.len(), 1);
                    let arg = args[0].load_value(fx);
                    let res = CValue::ByVal(fx.bcx.ins().popcnt(arg), args[0].layout());
                    ret.write_cvalue(fx, res);
                }
                "needs_drop" => {
                    assert_eq!(args.len(), 0);
                    let ty = substs.type_at(0);
                    let needs_drop = if ty.needs_drop(fx.tcx, ParamEnv::reveal_all()) {
                        1
                    } else {
                        0
                    };
                    let needs_drop = CValue::const_val(fx, fx.tcx.types.bool, needs_drop);
                    ret.write_cvalue(fx, needs_drop);
                }
                _ => unimpl!("unsupported intrinsic {}", intrinsic),
            }

            if let Some((_, dest)) = destination {
                let ret_ebb = fx.get_ebb(dest);
                fx.bcx.ins().jump(ret_ebb, &[]);
            } else {
                fx.bcx.ins().trap(TrapCode::User(!0));
            }
            return true;
        }
    }

    false
}
