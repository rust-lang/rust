//! Handling of everything related to the calling convention. Also fills `fx.local_map`.

mod comments;
mod pass_mode;
mod returning;

use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::ty::layout::FnAbiOf;
use rustc_target::abi::call::{Conv, FnAbi};
use rustc_target::spec::abi::Abi;

use cranelift_codegen::ir::{AbiParam, SigRef};

use self::pass_mode::*;
use crate::prelude::*;

pub(crate) use self::returning::codegen_return;

fn clif_sig_from_fn_abi<'tcx>(
    tcx: TyCtxt<'tcx>,
    triple: &target_lexicon::Triple,
    fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
) -> Signature {
    let call_conv = match fn_abi.conv {
        Conv::Rust | Conv::C => CallConv::triple_default(triple),
        Conv::X86_64SysV => CallConv::SystemV,
        Conv::X86_64Win64 => CallConv::WindowsFastcall,
        Conv::ArmAapcs
        | Conv::CCmseNonSecureCall
        | Conv::Msp430Intr
        | Conv::PtxKernel
        | Conv::X86Fastcall
        | Conv::X86Intr
        | Conv::X86Stdcall
        | Conv::X86ThisCall
        | Conv::X86VectorCall
        | Conv::AmdGpuKernel
        | Conv::AvrInterrupt
        | Conv::AvrNonBlockingInterrupt => todo!("{:?}", fn_abi.conv),
    };
    let inputs = fn_abi.args.iter().map(|arg_abi| arg_abi.get_abi_param(tcx).into_iter()).flatten();

    let (return_ptr, returns) = fn_abi.ret.get_abi_return(tcx);
    // Sometimes the first param is an pointer to the place where the return value needs to be stored.
    let params: Vec<_> = return_ptr.into_iter().chain(inputs).collect();

    Signature { params, returns, call_conv }
}

pub(crate) fn get_function_sig<'tcx>(
    tcx: TyCtxt<'tcx>,
    triple: &target_lexicon::Triple,
    inst: Instance<'tcx>,
) -> Signature {
    assert!(!inst.substs.needs_infer());
    clif_sig_from_fn_abi(
        tcx,
        triple,
        &RevealAllLayoutCx(tcx).fn_abi_of_instance(inst, ty::List::empty()),
    )
}

/// Instance must be monomorphized
pub(crate) fn import_function<'tcx>(
    tcx: TyCtxt<'tcx>,
    module: &mut dyn Module,
    inst: Instance<'tcx>,
) -> FuncId {
    let name = tcx.symbol_name(inst).name;
    let sig = get_function_sig(tcx, module.isa().triple(), inst);
    module.declare_function(name, Linkage::Import, &sig).unwrap()
}

impl<'tcx> FunctionCx<'_, '_, 'tcx> {
    /// Instance must be monomorphized
    pub(crate) fn get_function_ref(&mut self, inst: Instance<'tcx>) -> FuncRef {
        let func_id = import_function(self.tcx, self.module, inst);
        let func_ref = self.module.declare_func_in_func(func_id, &mut self.bcx.func);

        if self.clif_comments.enabled() {
            self.add_comment(func_ref, format!("{:?}", inst));
        }

        func_ref
    }

    pub(crate) fn lib_call(
        &mut self,
        name: &str,
        params: Vec<AbiParam>,
        returns: Vec<AbiParam>,
        args: &[Value],
    ) -> &[Value] {
        let sig = Signature { params, returns, call_conv: CallConv::triple_default(self.triple()) };
        let func_id = self.module.declare_function(name, Linkage::Import, &sig).unwrap();
        let func_ref = self.module.declare_func_in_func(func_id, &mut self.bcx.func);
        let call_inst = self.bcx.ins().call(func_ref, args);
        if self.clif_comments.enabled() {
            self.add_comment(call_inst, format!("easy_call {}", name));
        }
        let results = self.bcx.inst_results(call_inst);
        assert!(results.len() <= 2, "{}", results.len());
        results
    }

    pub(crate) fn easy_call(
        &mut self,
        name: &str,
        args: &[CValue<'tcx>],
        return_ty: Ty<'tcx>,
    ) -> CValue<'tcx> {
        let (input_tys, args): (Vec<_>, Vec<_>) = args
            .iter()
            .map(|arg| {
                (AbiParam::new(self.clif_type(arg.layout().ty).unwrap()), arg.load_scalar(self))
            })
            .unzip();
        let return_layout = self.layout_of(return_ty);
        let return_tys = if let ty::Tuple(tup) = return_ty.kind() {
            tup.types().map(|ty| AbiParam::new(self.clif_type(ty).unwrap())).collect()
        } else {
            vec![AbiParam::new(self.clif_type(return_ty).unwrap())]
        };
        let ret_vals = self.lib_call(name, input_tys, return_tys, &args);
        match *ret_vals {
            [] => CValue::by_ref(
                Pointer::const_addr(self, i64::from(self.pointer_type.bytes())),
                return_layout,
            ),
            [val] => CValue::by_val(val, return_layout),
            [val, extra] => CValue::by_val_pair(val, extra, return_layout),
            _ => unreachable!(),
        }
    }
}

/// Make a [`CPlace`] capable of holding value of the specified type.
fn make_local_place<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    local: Local,
    layout: TyAndLayout<'tcx>,
    is_ssa: bool,
) -> CPlace<'tcx> {
    let place = if is_ssa {
        if let rustc_target::abi::Abi::ScalarPair(_, _) = layout.abi {
            CPlace::new_var_pair(fx, local, layout)
        } else {
            CPlace::new_var(fx, local, layout)
        }
    } else {
        CPlace::new_stack_slot(fx, layout)
    };

    self::comments::add_local_place_comments(fx, place, local);

    place
}

pub(crate) fn codegen_fn_prelude<'tcx>(fx: &mut FunctionCx<'_, '_, 'tcx>, start_block: Block) {
    fx.bcx.append_block_params_for_function_params(start_block);

    fx.bcx.switch_to_block(start_block);
    fx.bcx.ins().nop();

    let ssa_analyzed = crate::analyze::analyze(fx);

    self::comments::add_args_header_comment(fx);

    let mut block_params_iter = fx.bcx.func.dfg.block_params(start_block).to_vec().into_iter();
    let ret_place =
        self::returning::codegen_return_param(fx, &ssa_analyzed, &mut block_params_iter);
    assert_eq!(fx.local_map.push(ret_place), RETURN_PLACE);

    // None means pass_mode == NoPass
    enum ArgKind<'tcx> {
        Normal(Option<CValue<'tcx>>),
        Spread(Vec<Option<CValue<'tcx>>>),
    }

    let fn_abi = fx.fn_abi.take().unwrap();
    let mut arg_abis_iter = fn_abi.args.iter();

    let func_params = fx
        .mir
        .args_iter()
        .map(|local| {
            let arg_ty = fx.monomorphize(fx.mir.local_decls[local].ty);

            // Adapted from https://github.com/rust-lang/rust/blob/145155dc96757002c7b2e9de8489416e2fdbbd57/src/librustc_codegen_llvm/mir/mod.rs#L442-L482
            if Some(local) == fx.mir.spread_arg {
                // This argument (e.g. the last argument in the "rust-call" ABI)
                // is a tuple that was spread at the ABI level and now we have
                // to reconstruct it into a tuple local variable, from multiple
                // individual function arguments.

                let tupled_arg_tys = match arg_ty.kind() {
                    ty::Tuple(ref tys) => tys,
                    _ => bug!("spread argument isn't a tuple?! but {:?}", arg_ty),
                };

                let mut params = Vec::new();
                for (i, _arg_ty) in tupled_arg_tys.types().enumerate() {
                    let arg_abi = arg_abis_iter.next().unwrap();
                    let param =
                        cvalue_for_param(fx, Some(local), Some(i), arg_abi, &mut block_params_iter);
                    params.push(param);
                }

                (local, ArgKind::Spread(params), arg_ty)
            } else {
                let arg_abi = arg_abis_iter.next().unwrap();
                let param =
                    cvalue_for_param(fx, Some(local), None, arg_abi, &mut block_params_iter);
                (local, ArgKind::Normal(param), arg_ty)
            }
        })
        .collect::<Vec<(Local, ArgKind<'tcx>, Ty<'tcx>)>>();

    assert!(fx.caller_location.is_none());
    if fx.instance.def.requires_caller_location(fx.tcx) {
        // Store caller location for `#[track_caller]`.
        let arg_abi = arg_abis_iter.next().unwrap();
        fx.caller_location =
            Some(cvalue_for_param(fx, None, None, arg_abi, &mut block_params_iter).unwrap());
    }

    assert!(arg_abis_iter.next().is_none(), "ArgAbi left behind");
    fx.fn_abi = Some(fn_abi);
    assert!(block_params_iter.next().is_none(), "arg_value left behind");

    self::comments::add_locals_header_comment(fx);

    for (local, arg_kind, ty) in func_params {
        let layout = fx.layout_of(ty);

        let is_ssa = ssa_analyzed[local] == crate::analyze::SsaKind::Ssa;

        // While this is normally an optimization to prevent an unnecessary copy when an argument is
        // not mutated by the current function, this is necessary to support unsized arguments.
        if let ArgKind::Normal(Some(val)) = arg_kind {
            if let Some((addr, meta)) = val.try_to_ptr() {
                // Ownership of the value at the backing storage for an argument is passed to the
                // callee per the ABI, so it is fine to borrow the backing storage of this argument
                // to prevent a copy.

                let place = if let Some(meta) = meta {
                    CPlace::for_ptr_with_extra(addr, meta, val.layout())
                } else {
                    CPlace::for_ptr(addr, val.layout())
                };

                self::comments::add_local_place_comments(fx, place, local);

                assert_eq!(fx.local_map.push(place), local);
                continue;
            }
        }

        let place = make_local_place(fx, local, layout, is_ssa);
        assert_eq!(fx.local_map.push(place), local);

        match arg_kind {
            ArgKind::Normal(param) => {
                if let Some(param) = param {
                    place.write_cvalue(fx, param);
                }
            }
            ArgKind::Spread(params) => {
                for (i, param) in params.into_iter().enumerate() {
                    if let Some(param) = param {
                        place.place_field(fx, mir::Field::new(i)).write_cvalue(fx, param);
                    }
                }
            }
        }
    }

    for local in fx.mir.vars_and_temps_iter() {
        let ty = fx.monomorphize(fx.mir.local_decls[local].ty);
        let layout = fx.layout_of(ty);

        let is_ssa = ssa_analyzed[local] == crate::analyze::SsaKind::Ssa;

        let place = make_local_place(fx, local, layout, is_ssa);
        assert_eq!(fx.local_map.push(place), local);
    }

    fx.bcx.ins().jump(*fx.block_map.get(START_BLOCK).unwrap(), &[]);
}

struct CallArgument<'tcx> {
    value: CValue<'tcx>,
    is_owned: bool,
}

// FIXME avoid intermediate `CValue` before calling `adjust_arg_for_abi`
fn codegen_call_argument_operand<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    operand: &Operand<'tcx>,
) -> CallArgument<'tcx> {
    CallArgument {
        value: codegen_operand(fx, operand),
        is_owned: matches!(operand, Operand::Move(_)),
    }
}

pub(crate) fn codegen_terminator_call<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    span: Span,
    func: &Operand<'tcx>,
    args: &[Operand<'tcx>],
    mir_dest: Option<(Place<'tcx>, BasicBlock)>,
) {
    let fn_ty = fx.monomorphize(func.ty(fx.mir, fx.tcx));
    let fn_sig =
        fx.tcx.normalize_erasing_late_bound_regions(ParamEnv::reveal_all(), fn_ty.fn_sig(fx.tcx));

    let destination = mir_dest.map(|(place, bb)| (codegen_place(fx, place), bb));

    // Handle special calls like instrinsics and empty drop glue.
    let instance = if let ty::FnDef(def_id, substs) = *fn_ty.kind() {
        let instance = ty::Instance::resolve(fx.tcx, ty::ParamEnv::reveal_all(), def_id, substs)
            .unwrap()
            .unwrap()
            .polymorphize(fx.tcx);

        if fx.tcx.symbol_name(instance).name.starts_with("llvm.") {
            crate::intrinsics::codegen_llvm_intrinsic_call(
                fx,
                &fx.tcx.symbol_name(instance).name,
                substs,
                args,
                destination,
            );
            return;
        }

        match instance.def {
            InstanceDef::Intrinsic(_) => {
                crate::intrinsics::codegen_intrinsic_call(fx, instance, args, destination, span);
                return;
            }
            InstanceDef::DropGlue(_, None) => {
                // empty drop glue - a nop.
                let (_, dest) = destination.expect("Non terminating drop_in_place_real???");
                let ret_block = fx.get_block(dest);
                fx.bcx.ins().jump(ret_block, &[]);
                return;
            }
            _ => Some(instance),
        }
    } else {
        None
    };

    let extra_args = &args[fn_sig.inputs().len()..];
    let extra_args = fx
        .tcx
        .mk_type_list(extra_args.iter().map(|op_arg| fx.monomorphize(op_arg.ty(fx.mir, fx.tcx))));
    let fn_abi = if let Some(instance) = instance {
        RevealAllLayoutCx(fx.tcx).fn_abi_of_instance(instance, extra_args)
    } else {
        RevealAllLayoutCx(fx.tcx).fn_abi_of_fn_ptr(fn_ty.fn_sig(fx.tcx), extra_args)
    };

    let is_cold = instance
        .map(|inst| fx.tcx.codegen_fn_attrs(inst.def_id()).flags.contains(CodegenFnAttrFlags::COLD))
        .unwrap_or(false);
    if is_cold {
        // FIXME Mark current_block block as cold once Cranelift supports it
    }

    // Unpack arguments tuple for closures
    let mut args = if fn_sig.abi == Abi::RustCall {
        assert_eq!(args.len(), 2, "rust-call abi requires two arguments");
        let self_arg = codegen_call_argument_operand(fx, &args[0]);
        let pack_arg = codegen_call_argument_operand(fx, &args[1]);

        let tupled_arguments = match pack_arg.value.layout().ty.kind() {
            ty::Tuple(ref tupled_arguments) => tupled_arguments,
            _ => bug!("argument to function with \"rust-call\" ABI is not a tuple"),
        };

        let mut args = Vec::with_capacity(1 + tupled_arguments.len());
        args.push(self_arg);
        for i in 0..tupled_arguments.len() {
            args.push(CallArgument {
                value: pack_arg.value.value_field(fx, mir::Field::new(i)),
                is_owned: pack_arg.is_owned,
            });
        }
        args
    } else {
        args.iter().map(|arg| codegen_call_argument_operand(fx, arg)).collect::<Vec<_>>()
    };

    // Pass the caller location for `#[track_caller]`.
    if instance.map(|inst| inst.def.requires_caller_location(fx.tcx)).unwrap_or(false) {
        let caller_location = fx.get_caller_location(span);
        args.push(CallArgument { value: caller_location, is_owned: false });
    }

    let args = args;
    assert_eq!(fn_abi.args.len(), args.len());

    enum CallTarget {
        Direct(FuncRef),
        Indirect(SigRef, Value),
    }

    let (func_ref, first_arg_override) = match instance {
        // Trait object call
        Some(Instance { def: InstanceDef::Virtual(_, idx), .. }) => {
            if fx.clif_comments.enabled() {
                let nop_inst = fx.bcx.ins().nop();
                fx.add_comment(
                    nop_inst,
                    format!("virtual call; self arg pass mode: {:?}", &fn_abi.args[0]),
                );
            }

            let (ptr, method) = crate::vtable::get_ptr_and_method_ref(fx, args[0].value, idx);
            let sig = clif_sig_from_fn_abi(fx.tcx, fx.triple(), &fn_abi);
            let sig = fx.bcx.import_signature(sig);

            (CallTarget::Indirect(sig, method), Some(ptr))
        }

        // Normal call
        Some(instance) => {
            let func_ref = fx.get_function_ref(instance);
            (CallTarget::Direct(func_ref), None)
        }

        // Indirect call
        None => {
            if fx.clif_comments.enabled() {
                let nop_inst = fx.bcx.ins().nop();
                fx.add_comment(nop_inst, "indirect call");
            }

            let func = codegen_operand(fx, func).load_scalar(fx);
            let sig = clif_sig_from_fn_abi(fx.tcx, fx.triple(), &fn_abi);
            let sig = fx.bcx.import_signature(sig);

            (CallTarget::Indirect(sig, func), None)
        }
    };

    let ret_place = destination.map(|(place, _)| place);
    self::returning::codegen_with_call_return_arg(fx, &fn_abi.ret, ret_place, |fx, return_ptr| {
        let call_args = return_ptr
            .into_iter()
            .chain(first_arg_override.into_iter())
            .chain(
                args.into_iter()
                    .enumerate()
                    .skip(if first_arg_override.is_some() { 1 } else { 0 })
                    .map(|(i, arg)| {
                        adjust_arg_for_abi(fx, arg.value, &fn_abi.args[i], arg.is_owned).into_iter()
                    })
                    .flatten(),
            )
            .collect::<Vec<Value>>();

        let call_inst = match func_ref {
            CallTarget::Direct(func_ref) => fx.bcx.ins().call(func_ref, &call_args),
            CallTarget::Indirect(sig, func_ptr) => {
                fx.bcx.ins().call_indirect(sig, func_ptr, &call_args)
            }
        };

        // FIXME find a cleaner way to support varargs
        if fn_sig.c_variadic {
            if !matches!(fn_sig.abi, Abi::C { .. }) {
                fx.tcx
                    .sess
                    .span_fatal(span, &format!("Variadic call for non-C abi {:?}", fn_sig.abi));
            }
            let sig_ref = fx.bcx.func.dfg.call_signature(call_inst).unwrap();
            let abi_params = call_args
                .into_iter()
                .map(|arg| {
                    let ty = fx.bcx.func.dfg.value_type(arg);
                    if !ty.is_int() {
                        // FIXME set %al to upperbound on float args once floats are supported
                        fx.tcx
                            .sess
                            .span_fatal(span, &format!("Non int ty {:?} for variadic call", ty));
                    }
                    AbiParam::new(ty)
                })
                .collect::<Vec<AbiParam>>();
            fx.bcx.func.dfg.signatures[sig_ref].params = abi_params;
        }

        call_inst
    });

    if let Some((_, dest)) = destination {
        let ret_block = fx.get_block(dest);
        fx.bcx.ins().jump(ret_block, &[]);
    } else {
        trap_unreachable(fx, "[corruption] Diverging function returned");
    }
}

pub(crate) fn codegen_drop<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    span: Span,
    drop_place: CPlace<'tcx>,
) {
    let ty = drop_place.layout().ty;
    let drop_instance = Instance::resolve_drop_in_place(fx.tcx, ty).polymorphize(fx.tcx);

    if let ty::InstanceDef::DropGlue(_, None) = drop_instance.def {
        // we don't actually need to drop anything
    } else {
        match ty.kind() {
            ty::Dynamic(..) => {
                let (ptr, vtable) = drop_place.to_ptr_maybe_unsized();
                let ptr = ptr.get_addr(fx);
                let drop_fn = crate::vtable::drop_fn_of_obj(fx, vtable.unwrap());

                // FIXME(eddyb) perhaps move some of this logic into
                // `Instance::resolve_drop_in_place`?
                let virtual_drop = Instance {
                    def: ty::InstanceDef::Virtual(drop_instance.def_id(), 0),
                    substs: drop_instance.substs,
                };
                let fn_abi =
                    RevealAllLayoutCx(fx.tcx).fn_abi_of_instance(virtual_drop, ty::List::empty());

                let sig = clif_sig_from_fn_abi(fx.tcx, fx.triple(), &fn_abi);
                let sig = fx.bcx.import_signature(sig);
                fx.bcx.ins().call_indirect(sig, drop_fn, &[ptr]);
            }
            _ => {
                assert!(!matches!(drop_instance.def, InstanceDef::Virtual(_, _)));

                let fn_abi =
                    RevealAllLayoutCx(fx.tcx).fn_abi_of_instance(drop_instance, ty::List::empty());

                let arg_value = drop_place.place_ref(
                    fx,
                    fx.layout_of(fx.tcx.mk_ref(
                        &ty::RegionKind::ReErased,
                        TypeAndMut { ty, mutbl: crate::rustc_hir::Mutability::Mut },
                    )),
                );
                let arg_value = adjust_arg_for_abi(fx, arg_value, &fn_abi.args[0], true);

                let mut call_args: Vec<Value> = arg_value.into_iter().collect::<Vec<_>>();

                if drop_instance.def.requires_caller_location(fx.tcx) {
                    // Pass the caller location for `#[track_caller]`.
                    let caller_location = fx.get_caller_location(span);
                    call_args.extend(
                        adjust_arg_for_abi(fx, caller_location, &fn_abi.args[1], false).into_iter(),
                    );
                }

                let func_ref = fx.get_function_ref(drop_instance);
                fx.bcx.ins().call(func_ref, &call_args);
            }
        }
    }
}
