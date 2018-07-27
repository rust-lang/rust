use std::iter;

use rustc::hir;
use rustc_target::spec::abi::Abi;

use prelude::*;

pub fn cton_sig_from_fn_ty<'a, 'tcx: 'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, fn_ty: Ty<'tcx>) -> Signature {
    let sig = ty_fn_sig(tcx, fn_ty);
    assert!(!sig.variadic, "Variadic function are not yet supported");
    let (call_conv, inputs, _output): (CallConv, Vec<Ty>, Ty) = match sig.abi {
        Abi::Rust => (CallConv::SystemV, sig.inputs().to_vec(), sig.output()),
        Abi::RustCall => {
            println!("rust-call sig: {:?} inputs: {:?} output: {:?}", sig, sig.inputs(), sig.output());
            assert_eq!(sig.inputs().len(), 2);
            let extra_args = match sig.inputs().last().unwrap().sty {
                ty::TyTuple(ref tupled_arguments) => tupled_arguments,
                _ => bug!("argument to function with \"rust-call\" ABI is not a tuple"),
            };
            let mut inputs: Vec<Ty> = vec![sig.inputs()[0]];
            inputs.extend(extra_args.into_iter());
            (
                CallConv::SystemV,
                inputs,
                sig.output(),
            )
        }
        Abi::System => bug!("system abi should be selected elsewhere"),
        // TODO: properly implement intrinsics
        Abi::RustIntrinsic => (CallConv::SystemV, sig.inputs().to_vec(), sig.output()),
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
) -> ty::FnSig<'tcx> {
    let sig = match ty.sty {
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
    };
    tcx.normalize_erasing_late_bound_regions(ParamEnv::reveal_all(), &sig)
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
            let def_path_based_names = ::rustc_mir::monomorphize::item::DefPathBasedNames::new(tcx, false, false);
            let mut name = String::new();
            def_path_based_names.push_instance_as_string(inst, &mut name);
            module.declare_function(&name, Linkage::Local, &sig).unwrap()
        });
        module.declare_func_in_func(func_id, &mut self.bcx.func)
    }

    fn self_sig(&self) -> FnSig<'tcx> {
        ty_fn_sig(self.tcx, self.instance.ty(self.tcx))
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
    let fn_ty = func.layout().ty;
    let sig = ty_fn_sig(fx.tcx, fn_ty);

    let return_place = if let Some((place, _)) = destination {
        Some(::base::trans_place(fx, place))
    } else {
        None
    };

    // Unpack arguments tuple for closures
    let args = if sig.abi == Abi::RustCall {
        assert_eq!(args.len(), 2, "rust-call abi requires two arguments");
        let self_arg = ::base::trans_operand(fx, &args[0]);
        let pack_arg = ::base::trans_operand(fx, &args[1]);
        let mut args = Vec::new();
        args.push(self_arg);
        match pack_arg.layout().ty.sty {
            ty::TyTuple(ref tupled_arguments) => {
                for (i, _) in tupled_arguments.iter().enumerate() {
                    args.push(pack_arg.value_field(fx, mir::Field::new(i)));
                }
            },
            _ => bug!("argument to function with \"rust-call\" ABI is not a tuple"),
        }
        args
    } else {
        args
            .into_iter()
            .map(|arg| {
                ::base::trans_operand(fx, arg)
            })
            .collect::<Vec<_>>()
    };

    if let TypeVariants::TyFnDef(def_id, substs) = fn_ty.sty {
        if sig.abi == Abi::RustIntrinsic {
            let intrinsic = fx.tcx.item_name(def_id).as_str();
            let intrinsic = &intrinsic[..];

            let usize_layout = fx.layout_of(fx.tcx.types.usize);
            let ret = return_place.unwrap();
            match intrinsic {
                "copy" => {
                    /*let elem_ty = substs.type_at(0);
                    assert_eq!(args.len(), 3);
                    let src = args[0];
                    let dst = args[1];
                    let count = args[2];*/
                    unimplemented!("copy");
                }
                "size_of" => {
                    assert_eq!(args.len(), 0);
                    let size_of = fx.layout_of(substs.type_at(0)).size.bytes();
                    let size_of = CValue::const_val(fx, usize_layout.ty, size_of as i64);
                    ret.write_cvalue(fx, size_of);
                }
                _ if intrinsic.starts_with("unchecked_") => {
                    assert_eq!(args.len(), 2);
                    let lhs = args[0].load_value(fx);
                    let rhs = args[1].load_value(fx);
                    let bin_op = match intrinsic {
                        "unchecked_div" => BinOp::Div,
                        "unchecked_rem" => BinOp::Rem,
                        "unchecked_shl" => BinOp::Shl,
                        "unchecked_shr" => BinOp::Shr,
                        _ => unimplemented!("intrinsic {}", intrinsic),
                    };
                    let res = match ret.layout().ty.sty {
                        TypeVariants::TyUint(_) => {
                            ::base::trans_int_binop(fx, bin_op, lhs, rhs, args[0].layout().ty, false, false)
                        }
                        TypeVariants::TyInt(_) => {
                            ::base::trans_int_binop(fx, bin_op, lhs, rhs, args[0].layout().ty, true, false)
                        }
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
                _ => fx.tcx.sess.fatal(&format!("unsupported intrinsic {}", intrinsic)),
            }
            if let Some((_, dest)) = *destination {
                let ret_ebb = fx.get_ebb(dest);
                fx.bcx.ins().jump(ret_ebb, &[]);
            } else {
                fx.bcx.ins().trap(TrapCode::User(!0));
            }
            return;
        }
    }

    let return_ptr = match return_place {
        Some(place) => place.expect_addr(),
        None => fx.bcx.ins().iconst(types::I64, 0),
    };

    let call_args = Some(return_ptr).into_iter().chain(args.into_iter().map(|arg| {
        if fx.cton_type(arg.layout().ty).is_some() {
            arg.load_value(fx)
        } else {
            arg.force_stack(fx)
        }
    })).collect::<Vec<_>>();

    match func {
        CValue::Func(func, _) => {
            fx.bcx.ins().call(func, &call_args);
        }
        func => {
            let func_ty = func.layout().ty;
            let func = func.load_value(fx);
            let sig = fx.bcx.import_signature(cton_sig_from_fn_ty(fx.tcx, func_ty));
            fx.bcx.ins().call_indirect(sig, func, &call_args);
        }
    }
    if let Some((_, dest)) = *destination {
        let ret_ebb = fx.get_ebb(dest);
        fx.bcx.ins().jump(ret_ebb, &[]);
    } else {
        fx.bcx.ins().trap(TrapCode::User(!0));
    }
}
