#[cfg(debug_assertions)]
mod comments;
mod pass_mode;
mod returning;

use rustc_target::spec::abi::Abi;

use self::pass_mode::*;
use crate::prelude::*;

pub use self::returning::codegen_return;

fn clif_sig_from_fn_sig<'tcx>(
    tcx: TyCtxt<'tcx>,
    sig: FnSig<'tcx>,
    is_vtable_fn: bool,
) -> Signature {
    let abi = match sig.abi {
        Abi::System => {
            if tcx.sess.target.target.options.is_like_windows {
                unimplemented!()
            } else {
                Abi::C
            }
        }
        abi => abi,
    };
    let (call_conv, inputs, output): (CallConv, Vec<Ty>, Ty) = match abi {
        Abi::Rust => (CallConv::SystemV, sig.inputs().to_vec(), sig.output()),
        Abi::C => (CallConv::SystemV, sig.inputs().to_vec(), sig.output()),
        Abi::RustCall => {
            assert_eq!(sig.inputs().len(), 2);
            let extra_args = match sig.inputs().last().unwrap().sty {
                ty::Tuple(ref tupled_arguments) => tupled_arguments,
                _ => bug!("argument to function with \"rust-call\" ABI is not a tuple"),
            };
            let mut inputs: Vec<Ty> = vec![sig.inputs()[0]];
            inputs.extend(extra_args.types());
            (CallConv::SystemV, inputs, sig.output())
        }
        Abi::System => unreachable!(),
        Abi::RustIntrinsic => (CallConv::SystemV, sig.inputs().to_vec(), sig.output()),
        _ => unimplemented!("unsupported abi {:?}", sig.abi),
    };

    let inputs = inputs
        .into_iter()
        .enumerate()
        .map(|(i, ty)| {
            let mut layout = tcx.layout_of(ParamEnv::reveal_all().and(ty)).unwrap();
            if i == 0 && is_vtable_fn {
                // Virtual calls turn their self param into a thin pointer.
                // See https://github.com/rust-lang/rust/blob/37b6a5e5e82497caf5353d9d856e4eb5d14cbe06/src/librustc/ty/layout.rs#L2519-L2572 for more info
                layout = tcx
                    .layout_of(ParamEnv::reveal_all().and(tcx.mk_mut_ptr(tcx.mk_unit())))
                    .unwrap();
            }
            get_pass_mode(tcx, layout).get_param_ty(tcx).into_iter()
        })
        .flatten();

    let (params, returns) = match get_pass_mode(
        tcx,
        tcx.layout_of(ParamEnv::reveal_all().and(output)).unwrap(),
    ) {
        PassMode::NoPass => (inputs.map(AbiParam::new).collect(), vec![]),
        PassMode::ByVal(ret_ty) => (
            inputs.map(AbiParam::new).collect(),
            vec![AbiParam::new(ret_ty)],
        ),
        PassMode::ByValPair(ret_ty_a, ret_ty_b) => (
            inputs.map(AbiParam::new).collect(),
            vec![AbiParam::new(ret_ty_a), AbiParam::new(ret_ty_b)],
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

pub fn get_function_name_and_sig<'tcx>(
    tcx: TyCtxt<'tcx>,
    inst: Instance<'tcx>,
    support_vararg: bool,
) -> (String, Signature) {
    assert!(!inst.substs.needs_infer() && !inst.substs.has_param_types());
    let fn_sig =
        tcx.normalize_erasing_late_bound_regions(ParamEnv::reveal_all(), &inst.fn_sig(tcx));
    if fn_sig.c_variadic && !support_vararg {
        unimpl!("Variadic function definitions are not yet supported");
    }
    let sig = clif_sig_from_fn_sig(tcx, fn_sig, false);
    (tcx.symbol_name(inst).name.as_str().to_string(), sig)
}

/// Instance must be monomorphized
pub fn import_function<'tcx>(
    tcx: TyCtxt<'tcx>,
    module: &mut Module<impl Backend>,
    inst: Instance<'tcx>,
) -> FuncId {
    let (name, sig) = get_function_name_and_sig(tcx, inst, true);
    module
        .declare_function(&name, Linkage::Import, &sig)
        .unwrap()
}

impl<'tcx, B: Backend + 'static> FunctionCx<'_, 'tcx, B> {
    /// Instance must be monomorphized
    pub fn get_function_ref(&mut self, inst: Instance<'tcx>) -> FuncRef {
        let func_id = import_function(self.tcx, self.module, inst);
        let func_ref = self
            .module
            .declare_func_in_func(func_id, &mut self.bcx.func);

        #[cfg(debug_assertions)]
        self.add_entity_comment(func_ref, format!("{:?}", inst));

        func_ref
    }

    fn lib_call(
        &mut self,
        name: &str,
        input_tys: Vec<types::Type>,
        output_tys: Vec<types::Type>,
        args: &[Value],
    ) -> &[Value] {
        let sig = Signature {
            params: input_tys.iter().cloned().map(AbiParam::new).collect(),
            returns: output_tys.iter().cloned().map(AbiParam::new).collect(),
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
        #[cfg(debug_assertions)]
        {
            self.add_comment(call_inst, format!("easy_call {}", name));
        }
        let results = self.bcx.inst_results(call_inst);
        assert!(results.len() <= 2, "{}", results.len());
        results
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
                    arg.load_scalar(self),
                )
            })
            .unzip();
        let return_layout = self.layout_of(return_ty);
        let return_tys = if let ty::Tuple(tup) = return_ty.sty {
            tup.types().map(|ty| self.clif_type(ty).unwrap()).collect()
        } else {
            vec![self.clif_type(return_ty).unwrap()]
        };
        let ret_vals = self.lib_call(name, input_tys, return_tys, &args);
        match *ret_vals {
            [] => CValue::by_ref(
                self.bcx
                    .ins()
                    .iconst(self.pointer_type, self.pointer_type.bytes() as i64),
                return_layout,
            ),
            [val] => CValue::by_val(val, return_layout),
            [val, extra] => CValue::by_val_pair(val, extra, return_layout),
            _ => unreachable!(),
        }
    }

    fn self_sig(&self) -> FnSig<'tcx> {
        self.tcx.normalize_erasing_late_bound_regions(
            ParamEnv::reveal_all(),
            &self.instance.fn_sig(self.tcx),
        )
    }

    fn return_layout(&self) -> TyLayout<'tcx> {
        self.layout_of(self.self_sig().output())
    }
}

fn local_place<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    local: Local,
    layout: TyLayout<'tcx>,
    is_ssa: bool,
) -> CPlace<'tcx> {
    let place = if is_ssa {
        CPlace::new_var(fx, local, layout)
    } else {
        CPlace::new_stack_slot(fx, layout.ty)
    };

    #[cfg(debug_assertions)]
    self::comments::add_local_place_comments(fx, place, local);

    let prev_place = fx.local_map.insert(local, place);
    debug_assert!(prev_place.is_none());
    fx.local_map[&local]
}

pub fn codegen_fn_prelude(fx: &mut FunctionCx<'_, '_, impl Backend>, start_ebb: Ebb) {
    let ssa_analyzed = crate::analyze::analyze(fx);

    #[cfg(debug_assertions)]
    self::comments::add_args_header_comment(fx);

    self::returning::codegen_return_param(fx, &ssa_analyzed, start_ebb);

    // None means pass_mode == NoPass
    enum ArgKind<'tcx> {
        Normal(Option<CValue<'tcx>>),
        Spread(Vec<Option<CValue<'tcx>>>),
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

                let mut params = Vec::new();
                for (i, arg_ty) in tupled_arg_tys.types().enumerate() {
                    let param = cvalue_for_param(fx, start_ebb, local, Some(i), arg_ty);
                    params.push(param);
                }

                (local, ArgKind::Spread(params), arg_ty)
            } else {
                let param = cvalue_for_param(fx, start_ebb, local, None, arg_ty);
                (local, ArgKind::Normal(param), arg_ty)
            }
        })
        .collect::<Vec<(Local, ArgKind, Ty)>>();

    fx.bcx.switch_to_block(start_ebb);

    #[cfg(debug_assertions)]
    self::comments::add_locals_header_comment(fx);

    for (local, arg_kind, ty) in func_params {
        let layout = fx.layout_of(ty);

        let is_ssa = !ssa_analyzed
            .get(&local)
            .unwrap()
            .contains(crate::analyze::Flags::NOT_SSA);

        match arg_kind {
            ArgKind::Normal(Some(val)) => {
                if let Some(addr) = val.try_to_addr() {
                    let local_decl = &fx.mir.local_decls[local];
                    //                             v this ! is important
                    let internally_mutable = !val.layout().ty.is_freeze(
                        fx.tcx,
                        ParamEnv::reveal_all(),
                        local_decl.source_info.span,
                    );
                    if local_decl.mutability == mir::Mutability::Not && internally_mutable {
                        // We wont mutate this argument, so it is fine to borrow the backing storage
                        // of this argument, to prevent a copy.

                        let place = CPlace::for_addr(addr, val.layout());

                        #[cfg(debug_assertions)]
                        self::comments::add_local_place_comments(fx, place, local);

                        let prev_place = fx.local_map.insert(local, place);
                        debug_assert!(prev_place.is_none());
                        continue;
                    }
                }
            }
            _ => {}
        }

        let place = local_place(fx, local, layout, is_ssa);

        match arg_kind {
            ArgKind::Normal(param) => {
                if let Some(param) = param {
                    place.write_cvalue(fx, param);
                }
            }
            ArgKind::Spread(params) => {
                for (i, param) in params.into_iter().enumerate() {
                    if let Some(param) = param {
                        place
                            .place_field(fx, mir::Field::new(i))
                            .write_cvalue(fx, param);
                    }
                }
            }
        }
    }

    for local in fx.mir.vars_and_temps_iter() {
        let ty = fx.mir.local_decls[local].ty;
        let layout = fx.layout_of(ty);

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

pub fn codegen_terminator_call<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    func: &Operand<'tcx>,
    args: &[Operand<'tcx>],
    destination: &Option<(Place<'tcx>, BasicBlock)>,
) {
    let fn_ty = fx.monomorphize(&func.ty(fx.mir, fx.tcx));
    let sig = fx
        .tcx
        .normalize_erasing_late_bound_regions(ParamEnv::reveal_all(), &fn_ty.fn_sig(fx.tcx));

    let destination = destination
        .as_ref()
        .map(|&(ref place, bb)| (trans_place(fx, place), bb));

    if let ty::FnDef(def_id, substs) = fn_ty.sty {
        let instance =
            ty::Instance::resolve(fx.tcx, ty::ParamEnv::reveal_all(), def_id, substs).unwrap();

        if fx.tcx.symbol_name(instance).name.as_str().starts_with("llvm.") {
            crate::llvm_intrinsics::codegen_llvm_intrinsic_call(
                fx,
                &fx.tcx.symbol_name(instance).name.as_str(),
                substs,
                args,
                destination,
            );
            return;
        }

        match instance.def {
            InstanceDef::Intrinsic(_) => {
                crate::intrinsics::codegen_intrinsic_call(fx, instance, args, destination);
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
        trap_unreachable(fx, "[corruption] Diverging function returned");
    }
}

fn codegen_call_inner<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    func: Option<&Operand<'tcx>>,
    fn_ty: Ty<'tcx>,
    args: Vec<CValue<'tcx>>,
    ret_place: Option<CPlace<'tcx>>,
) {
    let fn_sig = fx
        .tcx
        .normalize_erasing_late_bound_regions(ParamEnv::reveal_all(), &fn_ty.fn_sig(fx.tcx));

    let instance = match fn_ty.sty {
        ty::FnDef(def_id, substs) => {
            Some(Instance::resolve(fx.tcx, ParamEnv::reveal_all(), def_id, substs).unwrap())
        }
        _ => None,
    };

    //   | indirect call target
    //   |         | the first argument to be passed
    //   v         v          v virtual calls are special cased below
    let (func_ref, first_arg, is_virtual_call) = match instance {
        // Trait object call
        Some(Instance {
            def: InstanceDef::Virtual(_, idx),
            ..
        }) => {
            #[cfg(debug_assertions)]
            {
                let nop_inst = fx.bcx.ins().nop();
                fx.add_comment(
                    nop_inst,
                    format!(
                        "virtual call; self arg pass mode: {:?}",
                        get_pass_mode(fx.tcx, args[0].layout())
                    ),
                );
            }
            let (ptr, method) = crate::vtable::get_ptr_and_method_ref(fx, args[0], idx);
            (Some(method), Single(ptr), true)
        }

        // Normal call
        Some(_) => (
            None,
            args.get(0)
                .map(|arg| adjust_arg_for_abi(fx, *arg))
                .unwrap_or(Empty),
            false,
        ),

        // Indirect call
        None => {
            #[cfg(debug_assertions)]
            {
                let nop_inst = fx.bcx.ins().nop();
                fx.add_comment(nop_inst, "indirect call");
            }
            let func = trans_operand(fx, func.expect("indirect call without func Operand"))
                .load_scalar(fx);
            (
                Some(func),
                args.get(0)
                    .map(|arg| adjust_arg_for_abi(fx, *arg))
                    .unwrap_or(Empty),
                false,
            )
        }
    };

    let (call_inst, call_args) =
        self::returning::codegen_with_call_return_arg(fx, fn_sig, ret_place, |fx, return_ptr| {
            let call_args: Vec<Value> = return_ptr
                .into_iter()
                .chain(first_arg.into_iter())
                .chain(
                    args.into_iter()
                        .skip(1)
                        .map(|arg| adjust_arg_for_abi(fx, arg).into_iter())
                        .flatten(),
                )
                .collect::<Vec<_>>();

            let call_inst = if let Some(func_ref) = func_ref {
                let sig =
                    fx.bcx
                        .import_signature(clif_sig_from_fn_sig(fx.tcx, fn_sig, is_virtual_call));
                fx.bcx.ins().call_indirect(sig, func_ref, &call_args)
            } else {
                let func_ref =
                    fx.get_function_ref(instance.expect("non-indirect call on non-FnDef type"));
                fx.bcx.ins().call(func_ref, &call_args)
            };

            (call_inst, call_args)
        });

    // FIXME find a cleaner way to support varargs
    if fn_sig.c_variadic {
        if fn_sig.abi != Abi::C {
            unimpl!("Variadic call for non-C abi {:?}", fn_sig.abi);
        }
        let sig_ref = fx.bcx.func.dfg.call_signature(call_inst).unwrap();
        let abi_params = call_args
            .into_iter()
            .map(|arg| {
                let ty = fx.bcx.func.dfg.value_type(arg);
                if !ty.is_int() {
                    // FIXME set %al to upperbound on float args once floats are supported
                    unimpl!("Non int ty {:?} for variadic call", ty);
                }
                AbiParam::new(ty)
            })
            .collect::<Vec<AbiParam>>();
        fx.bcx.func.dfg.signatures[sig_ref].params = abi_params;
    }
}

pub fn codegen_drop<'tcx>(fx: &mut FunctionCx<'_, 'tcx, impl Backend>, drop_place: CPlace<'tcx>) {
    let ty = drop_place.layout().ty;
    let drop_fn = Instance::resolve_drop_in_place(fx.tcx, ty);

    if let ty::InstanceDef::DropGlue(_, None) = drop_fn.def {
        // we don't actually need to drop anything
    } else {
        let drop_fn_ty = drop_fn.ty(fx.tcx);
        match ty.sty {
            ty::Dynamic(..) => {
                let (ptr, vtable) = drop_place.to_addr_maybe_unsized(fx);
                let drop_fn = crate::vtable::drop_fn_of_obj(fx, vtable.unwrap());

                let fn_sig = fx.tcx.normalize_erasing_late_bound_regions(
                    ParamEnv::reveal_all(),
                    &drop_fn_ty.fn_sig(fx.tcx),
                );

                assert_eq!(fn_sig.output(), fx.tcx.mk_unit());

                let sig = fx
                    .bcx
                    .import_signature(clif_sig_from_fn_sig(fx.tcx, fn_sig, true));
                fx.bcx.ins().call_indirect(sig, drop_fn, &[ptr]);
            }
            _ => {
                let arg_place = CPlace::new_stack_slot(
                    fx,
                    fx.tcx.mk_ref(
                        &ty::RegionKind::ReErased,
                        TypeAndMut {
                            ty,
                            mutbl: crate::rustc::hir::Mutability::MutMutable,
                        },
                    ),
                );
                drop_place.write_place_ref(fx, arg_place);
                let arg_value = arg_place.to_cvalue(fx);
                codegen_call_inner(fx, None, drop_fn_ty, vec![arg_value], None);
            }
        }
    }
}
