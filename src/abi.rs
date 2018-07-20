use std::iter;

use rustc::hir;
use rustc_target::spec::abi::Abi;

use prelude::*;

pub fn cton_sig_from_fn_ty<'a, 'tcx: 'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, fn_ty: Ty<'tcx>) -> Signature {
    let sig = ty_fn_sig(tcx, fn_ty);
    let sig = tcx.normalize_erasing_late_bound_regions(ParamEnv::reveal_all(), &sig);
    assert!(!sig.variadic, "Variadic function are not yet supported");
    let (call_conv, inputs, _output): (CallConv, Vec<Ty>, Ty) = match sig.abi {
        Abi::Rust => (CallConv::SystemV, sig.inputs().to_vec(), sig.output()),
        Abi::RustCall => {
            unimplemented!();
        }
        Abi::System => bug!("system abi should be selected elsewhere"),
        _ => unimplemented!("unsupported abi {:?}", sig.abi),
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

fn ty_fn_sig<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    ty: Ty<'tcx>
) -> ty::PolyFnSig<'tcx> {
    match ty.sty {
        ty::TyFnDef(..) |
        // Shims currently have type TyFnPtr. Not sure this should remain.
        ty::TyFnPtr(_) => ty.fn_sig(tcx),
        ty::TyClosure(def_id, substs) => {
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
        ty::TyGenerator(def_id, substs, _) => {
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
    }
}

impl<'a, 'tcx: 'a> FunctionCx<'a, 'tcx> {
    /// Instance must be monomorphized
    pub fn get_function_ref(&mut self, inst: Instance<'tcx>) -> FuncRef {
        assert!(!inst.substs.needs_infer() && !inst.substs.has_param_types());
        let tcx = self.tcx;
        let module = &mut self.module;
        let func_id = *self.def_id_fn_id_map.entry(inst).or_insert_with(|| {
            let fn_ty = inst.ty(tcx);
            let sig = cton_sig_from_fn_ty(tcx, fn_ty);
            module.declare_function(&tcx.absolute_item_path_str(inst.def_id()), Linkage::Local, &sig).unwrap()
        });
        module.declare_func_in_func(func_id, &mut self.bcx.func)
    }

    fn self_sig(&self) -> FnSig<'tcx> {
        let sig = ty_fn_sig(self.tcx, self.instance.ty(self.tcx));
        self.tcx.normalize_erasing_late_bound_regions(ParamEnv::reveal_all(), &sig)
    }

    fn return_type(&self) -> Ty<'tcx> {
        self.self_sig().output()
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

    let ret_layout = fx.layout_of(fx.return_type());
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
) {
    let func = ::base::trans_operand(fx, func);
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
    match func {
        CValue::Func(func, _) => {
            fx.bcx.ins().call(func, &args);
        }
        func => {
            let func_ty = func.layout().ty;
            let func = func.load_value(fx);
            let sig = fx.bcx.import_signature(cton_sig_from_fn_ty(fx.tcx, func_ty));
            fx.bcx.ins().call_indirect(sig, func, &args);
        }
    }
    if let Some((_, dest)) = *destination {
        let ret_ebb = fx.get_ebb(dest);
        fx.bcx.ins().jump(ret_ebb, &[]);
    } else {
        fx.bcx.ins().trap(TrapCode::User(!0));
    }
}
