use prelude::*;

pub fn cton_sig_from_fn_sig<'a, 'tcx: 'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, sig: PolyFnSig<'tcx>, substs: &Substs<'tcx>) -> Signature {
    let sig = tcx.subst_and_normalize_erasing_regions(substs, ParamEnv::reveal_all(), &sig);
    cton_sig_from_mono_fn_sig(tcx, sig)
}

pub fn cton_sig_from_instance<'a, 'tcx: 'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, inst: Instance<'tcx>) -> Signature {
    let fn_ty = inst.ty(tcx);
    let sig = fn_ty.fn_sig(tcx);
    cton_sig_from_mono_fn_sig(tcx, sig)
}

pub fn cton_sig_from_mono_fn_sig<'a, 'tcx: 'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, sig: PolyFnSig<'tcx>) -> Signature {
    // TODO: monomorphize signature

    let sig = tcx.normalize_erasing_late_bound_regions(ParamEnv::reveal_all(), &sig);
    let inputs = sig.inputs();
    let _output = sig.output();
    assert!(!sig.variadic, "Variadic function are not yet supported");
    let call_conv = match sig.abi {
        _ => CallConv::SystemV,
    };
    Signature {
        params: Some(types::I64).into_iter() // First param is place to put return val
            .chain(inputs.into_iter().map(|ty| cton_type_from_ty(tcx, ty).unwrap_or(types::I64)))
            .map(AbiParam::new).collect(),
        returns: vec![],
        call_conv,
        argument_bytes: None,
    }
}

impl<'a, 'tcx: 'a> FunctionCx<'a, 'tcx> {
    pub fn get_function_ref(&mut self, inst: Instance<'tcx>) -> FuncRef {
        let tcx = self.tcx;
        let module = &mut self.module;
        let func_id = *self.def_id_fn_id_map.entry(inst).or_insert_with(|| {
            let sig = cton_sig_from_instance(tcx, inst);
            module.declare_function(&tcx.absolute_item_path_str(inst.def_id()), Linkage::Local, &sig).unwrap()
        });
        module.declare_func_in_func(func_id, &mut self.bcx.func)
    }
}

pub fn codegen_fn_prelude<'a, 'tcx: 'a>(fx: &mut FunctionCx<'a, 'tcx>, start_ebb: Ebb) {
    let ret_param = fx.bcx.append_ebb_param(start_ebb, types::I64);
    let _ = fx.bcx.create_stack_slot(StackSlotData {
        kind: StackSlotKind::ExplicitSlot,
        size: 0,
        offset: None,
    }); // Dummy stack slot for debugging

    let func_params = fx.mir.args_iter().map(|local| {
        let layout = fx.layout_of(fx.mir.local_decls[local].ty);
        let stack_slot = fx.bcx.create_stack_slot(StackSlotData {
            kind: StackSlotKind::ExplicitSlot,
            size: layout.size.bytes() as u32,
            offset: None,
        });
        let ty = fx.mir.local_decls[local].ty;
        let cton_type = fx.cton_type(ty).unwrap_or(types::I64);
        (local, fx.bcx.append_ebb_param(start_ebb, cton_type), ty, stack_slot)
    }).collect::<Vec<(Local, Value, Ty, StackSlot)>>();

    let ret_layout = fx.layout_of(fx.instance.ty(fx.tcx).fn_sig(fx.tcx).skip_binder().output());
    fx.local_map.insert(RETURN_PLACE, CPlace::Addr(ret_param, ret_layout));

    for (local, ebb_param, ty, stack_slot) in func_params {
        let place = CPlace::from_stack_slot(fx, stack_slot, ty);
        if fx.cton_type(ty).is_some() {
            place.write_cvalue(fx, CValue::ByVal(ebb_param, place.layout()));
        } else {
            place.write_cvalue(fx, CValue::ByRef(ebb_param, place.layout()));
        }
        fx.local_map.insert(local, place);
    }

    for local in fx.mir.vars_and_temps_iter() {
        let ty = fx.mir.local_decls[local].ty;
        let layout = fx.layout_of(ty);
        let stack_slot = fx.bcx.create_stack_slot(StackSlotData {
            kind: StackSlotKind::ExplicitSlot,
            size: layout.size.bytes() as u32,
            offset: None,
        });
        let place = CPlace::from_stack_slot(fx, stack_slot, ty);
        fx.local_map.insert(local, place);
    }
}

pub fn codegen_call<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx>,
    func: &Operand<'tcx>,
    args: &[Operand<'tcx>],
    destination: &Option<(Place<'tcx>, BasicBlock)>,
) -> Inst {
    let func = ::base::trans_operand(fx, func);
    let func_ty = func.layout().ty;
    let return_place = if let Some((place, _)) = destination {
        ::base::trans_place(fx, place).expect_addr()
    } else {
        fx.bcx.ins().iconst(types::I64, 0)
    };
    let args = Some(return_place)
        .into_iter()
        .chain(
            args
                .into_iter()
                .map(|arg| {
                    let arg = ::base::trans_operand(fx, arg);
                    if let Some(_) = fx.cton_type(arg.layout().ty) {
                        arg.load_value(fx)
                    } else {
                        arg.force_stack(fx)
                    }
                })
        ).collect::<Vec<_>>();
    let inst = match func {
        CValue::Func(func, _) => {
            fx.bcx.ins().call(func, &args)
        }
        func => {
            let func = func.load_value(fx);
            let sig = match func_ty.sty {
                TypeVariants::TyFnDef(def_id, _substs) => fx.tcx.fn_sig(def_id),
                TypeVariants::TyFnPtr(fn_sig) => fn_sig,
                _ => bug!("Calling non function type {:?}", func_ty),
            };
            let sig = fx.bcx.import_signature(cton_sig_from_fn_sig(fx.tcx, sig, fx.param_substs));
            fx.bcx.ins().call_indirect(sig, func, &args)
        }
    };
    if let Some((_, dest)) = *destination {
        let ret_ebb = fx.get_ebb(dest);
        fx.bcx.ins().jump(ret_ebb, &[]);
    } else {
        fx.bcx.ins().trap(TrapCode::User(!0));
    }
    inst
}
