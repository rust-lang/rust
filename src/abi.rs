use std::iter;

use rustc::hir;
use rustc_target::spec::abi::Abi;

use crate::prelude::*;

pub fn cton_sig_from_fn_ty<'a, 'tcx: 'a>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    fn_ty: Ty<'tcx>,
) -> Signature {
    let sig = ty_fn_sig(tcx, fn_ty);
    assert!(!sig.variadic, "Variadic function are not yet supported");
    let (call_conv, inputs, _output): (CallConv, Vec<Ty>, Ty) = match sig.abi {
        Abi::Rust => (CallConv::Fast, sig.inputs().to_vec(), sig.output()),
        Abi::C => (CallConv::SystemV, sig.inputs().to_vec(), sig.output()),
        Abi::RustCall => {
            println!(
                "rust-call sig: {:?} inputs: {:?} output: {:?}",
                sig,
                sig.inputs(),
                sig.output()
            );
            assert_eq!(sig.inputs().len(), 2);
            let extra_args = match sig.inputs().last().unwrap().sty {
                ty::TyTuple(ref tupled_arguments) => tupled_arguments,
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
    Signature {
        params: Some(types::I64).into_iter() // First param is place to put return val
            .chain(inputs.into_iter().map(|ty| {
                let cton_ty = cton_type_from_ty(tcx, ty);
                if let Some(cton_ty) = cton_ty {
                    cton_ty
                } else {
                    if sig.abi == Abi::C {
                        unimplemented!("Non scalars are not yet supported for \"C\" abi");
                    }
                    types::I64
                }
            }))
            .map(AbiParam::new).collect(),
        returns: vec![],
        call_conv,
        argument_bytes: None,
    }
}

fn ty_fn_sig<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: Ty<'tcx>) -> ty::FnSig<'tcx> {
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
        let fn_ty = inst.ty(self.tcx);
        let sig = cton_sig_from_fn_ty(self.tcx, fn_ty);
        let def_path_based_names =
            ::rustc_mir::monomorphize::item::DefPathBasedNames::new(self.tcx, false, false);
        let mut name = String::new();
        def_path_based_names.push_instance_as_string(inst, &mut name);
        let func_id = self
            .module
            .declare_function(&name, Linkage::Import, &sig)
            .unwrap();
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
            returns: vec![AbiParam::new(output_ty.unwrap_or(types::VOID))],
            call_conv: CallConv::SystemV,
            argument_bytes: None,
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
        let return_ty = if let TypeVariants::TyTuple(tup) = return_ty.sty {
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
            CValue::ByRef(self.bcx.ins().iconst(types::I64, 0), return_layout)
        }
    }

    fn self_sig(&self) -> FnSig<'tcx> {
        ty_fn_sig(self.tcx, self.instance.ty(self.tcx))
    }

    fn return_type(&self) -> Ty<'tcx> {
        self.self_sig().output()
    }
}

pub fn codegen_fn_prelude<'a, 'tcx: 'a>(fx: &mut FunctionCx<'a, 'tcx>, start_ebb: Ebb) {
    match fx.self_sig().abi {
        Abi::Rust | Abi::RustCall => {}
        _ => unimplemented!("declared function with non \"rust\" or \"rust-call\" abi"),
    }

    let ret_param = fx.bcx.append_ebb_param(start_ebb, types::I64);
    let _ = fx.bcx.create_stack_slot(StackSlotData {
        kind: StackSlotKind::ExplicitSlot,
        size: 0,
        offset: None,
    }); // Dummy stack slot for debugging

    enum ArgKind {
        Normal(Value),
        Spread(Vec<Value>),
    }

    let func_params = fx.mir.args_iter().map(|local| {
        let arg_ty = fx.mir.local_decls[local].ty;

        // Adapted from https://github.com/rust-lang/rust/blob/145155dc96757002c7b2e9de8489416e2fdbbd57/src/librustc_codegen_llvm/mir/mod.rs#L442-L482
        if Some(local) == fx.mir.spread_arg {
            // This argument (e.g. the last argument in the "rust-call" ABI)
            // is a tuple that was spread at the ABI level and now we have
            // to reconstruct it into a tuple local variable, from multiple
            // individual function arguments.

            let tupled_arg_tys = match arg_ty.sty {
                ty::TyTuple(ref tys) => tys,
                _ => bug!("spread argument isn't a tuple?!")
            };

            let mut ebb_params = Vec::new();
            for arg_ty in tupled_arg_tys.iter() {
                let cton_type = fx.cton_type(arg_ty).unwrap_or(types::I64);
                ebb_params.push(fx.bcx.append_ebb_param(start_ebb, cton_type));
            }

            (local, ArgKind::Spread(ebb_params), arg_ty)
        } else {
            let cton_type = fx.cton_type(arg_ty).unwrap_or(types::I64);
            (local, ArgKind::Normal(fx.bcx.append_ebb_param(start_ebb, cton_type)), arg_ty)
        }
    }).collect::<Vec<(Local, ArgKind, Ty)>>();

    let ret_layout = fx.layout_of(fx.return_type());
    fx.local_map
        .insert(RETURN_PLACE, CPlace::Addr(ret_param, ret_layout));

    for (local, arg_kind, ty) in func_params {
        let layout = fx.layout_of(ty);
        let stack_slot = fx.bcx.create_stack_slot(StackSlotData {
            kind: StackSlotKind::ExplicitSlot,
            size: layout.size.bytes() as u32,
            offset: None,
        });

        let place = CPlace::from_stack_slot(fx, stack_slot, ty);

        match arg_kind {
            ArgKind::Normal(ebb_param) => {
                if fx.cton_type(ty).is_some() {
                    place.write_cvalue(fx, CValue::ByVal(ebb_param, place.layout()));
                } else {
                    place.write_cvalue(fx, CValue::ByRef(ebb_param, place.layout()));
                }
            }
            ArgKind::Spread(ebb_params) => {
                for (i, ebb_param) in ebb_params.into_iter().enumerate() {
                    let sub_place = place.place_field(fx, mir::Field::new(i));
                    if fx.cton_type(sub_place.layout().ty).is_some() {
                        sub_place.write_cvalue(fx, CValue::ByVal(ebb_param, sub_place.layout()));
                    } else {
                        sub_place.write_cvalue(fx, CValue::ByRef(ebb_param, sub_place.layout()));
                    }
                }
            }
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
    let func = trans_operand(fx, func);
    let fn_ty = func.layout().ty;
    let sig = ty_fn_sig(fx.tcx, fn_ty);

    let return_place = if let Some((place, _)) = destination {
        Some(trans_place(fx, place))
    } else {
        None
    };

    // Unpack arguments tuple for closures
    let args = if sig.abi == Abi::RustCall {
        assert_eq!(args.len(), 2, "rust-call abi requires two arguments");
        let self_arg = trans_operand(fx, &args[0]);
        let pack_arg = trans_operand(fx, &args[1]);
        let mut args = Vec::new();
        args.push(self_arg);
        match pack_arg.layout().ty.sty {
            ty::TyTuple(ref tupled_arguments) => {
                for (i, _) in tupled_arguments.iter().enumerate() {
                    args.push(pack_arg.value_field(fx, mir::Field::new(i)));
                }
            }
            _ => bug!("argument to function with \"rust-call\" ABI is not a tuple"),
        }
        println!(
            "{:?} {:?}",
            pack_arg.layout().ty,
            args.iter().map(|a| a.layout().ty).collect::<Vec<_>>()
        );
        args
    } else {
        args.into_iter()
            .map(|arg| trans_operand(fx, arg))
            .collect::<Vec<_>>()
    };

    if let TypeVariants::TyFnDef(def_id, substs) = fn_ty.sty {
        if sig.abi == Abi::RustIntrinsic {
            let intrinsic = fx.tcx.item_name(def_id).as_str();
            let intrinsic = &intrinsic[..];

            let nil_ty = fx.tcx.mk_nil();
            let usize_layout = fx.layout_of(fx.tcx.types.usize);
            let ret = return_place.unwrap();
            match intrinsic {
                "abort" => {
                    fx.bcx.ins().trap(TrapCode::User(!0 - 1));
                }
                "assume" => {
                    assert_eq!(args.len(), 1);
                }
                "likely" | "unlikely" => {
                    assert_eq!(args.len(), 1);
                    ret.write_cvalue(fx, args[0]);
                }
                "copy" | "copy_nonoverlapping" => {
                    let elem_ty = substs.type_at(0);
                    let elem_size: u64 = fx.layout_of(elem_ty).size.bytes();
                    let elem_size = fx.bcx.ins().iconst(types::I64, elem_size as i64);
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
                    unimplemented!("copy");
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
                "type_id" => {
                    assert_eq!(args.len(), 0);
                    let type_id = fx.tcx.type_id_hash(substs.type_at(0));
                    let type_id = CValue::const_val(fx, usize_layout.ty, type_id as i64);
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
                        TypeVariants::TyUint(_) => crate::base::trans_int_binop(
                            fx,
                            bin_op,
                            args[0],
                            args[1],
                            ret.layout().ty,
                            false,
                            false,
                        ),
                        TypeVariants::TyInt(_) => crate::base::trans_int_binop(
                            fx,
                            bin_op,
                            args[0],
                            args[1],
                            ret.layout().ty,
                            true,
                            false,
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
                _ => fx
                    .tcx
                    .sess
                    .fatal(&format!("unsupported intrinsic {}", intrinsic)),
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

    let call_args = Some(return_ptr)
        .into_iter()
        .chain(args.into_iter().map(|arg| {
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
            let sig = fx
                .bcx
                .import_signature(cton_sig_from_fn_ty(fx.tcx, func_ty));
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
