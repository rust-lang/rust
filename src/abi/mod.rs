//! Handling of everything related to the calling convention. Also fills `fx.local_map`.

mod comments;
mod pass_mode;
mod returning;

use std::borrow::Cow;

use cranelift_module::ModuleError;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::ty::layout::FnAbiOf;
use rustc_session::Session;
use rustc_target::abi::call::{Conv, FnAbi};
use rustc_target::spec::abi::Abi;

use cranelift_codegen::ir::{AbiParam, SigRef};

use self::pass_mode::*;
use crate::prelude::*;

pub(crate) use self::returning::codegen_return;

fn clif_sig_from_fn_abi<'tcx>(
    tcx: TyCtxt<'tcx>,
    default_call_conv: CallConv,
    fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
) -> Signature {
    let call_conv = conv_to_call_conv(tcx.sess, fn_abi.conv, default_call_conv);

    let inputs = fn_abi.args.iter().flat_map(|arg_abi| arg_abi.get_abi_param(tcx).into_iter());

    let (return_ptr, returns) = fn_abi.ret.get_abi_return(tcx);
    // Sometimes the first param is an pointer to the place where the return value needs to be stored.
    let params: Vec<_> = return_ptr.into_iter().chain(inputs).collect();

    Signature { params, returns, call_conv }
}

pub(crate) fn conv_to_call_conv(sess: &Session, c: Conv, default_call_conv: CallConv) -> CallConv {
    match c {
        Conv::Rust | Conv::C => default_call_conv,
        Conv::RustCold => CallConv::Cold,
        Conv::X86_64SysV => CallConv::SystemV,
        Conv::X86_64Win64 => CallConv::WindowsFastcall,

        // Should already get a back compat warning
        Conv::X86Fastcall | Conv::X86Stdcall | Conv::X86ThisCall | Conv::X86VectorCall => {
            default_call_conv
        }

        Conv::X86Intr => sess.fatal("x86-interrupt call conv not yet implemented"),

        Conv::ArmAapcs => sess.fatal("aapcs call conv not yet implemented"),
        Conv::CCmseNonSecureCall => {
            sess.fatal("C-cmse-nonsecure-call call conv is not yet implemented");
        }

        Conv::Msp430Intr
        | Conv::PtxKernel
        | Conv::AmdGpuKernel
        | Conv::AvrInterrupt
        | Conv::AvrNonBlockingInterrupt => {
            unreachable!("tried to use {c:?} call conv which only exists on an unsupported target");
        }
    }
}

pub(crate) fn get_function_sig<'tcx>(
    tcx: TyCtxt<'tcx>,
    default_call_conv: CallConv,
    inst: Instance<'tcx>,
) -> Signature {
    assert!(!inst.args.has_infer());
    clif_sig_from_fn_abi(
        tcx,
        default_call_conv,
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
    let sig = get_function_sig(tcx, module.target_config().default_call_conv, inst);
    match module.declare_function(name, Linkage::Import, &sig) {
        Ok(func_id) => func_id,
        Err(ModuleError::IncompatibleDeclaration(_)) => tcx.sess.fatal(format!(
            "attempt to declare `{name}` as function, but it was already declared as static"
        )),
        Err(ModuleError::IncompatibleSignature(_, prev_sig, new_sig)) => tcx.sess.fatal(format!(
            "attempt to declare `{name}` with signature {new_sig:?}, \
             but it was already declared with signature {prev_sig:?}"
        )),
        Err(err) => Err::<_, _>(err).unwrap(),
    }
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
    ) -> Cow<'_, [Value]> {
        if self.tcx.sess.target.is_like_windows {
            let (mut params, mut args): (Vec<_>, Vec<_>) =
                params
                    .into_iter()
                    .zip(args)
                    .map(|(param, &arg)| {
                        if param.value_type == types::I128 {
                            let arg_ptr = Pointer::stack_slot(self.bcx.create_sized_stack_slot(
                                StackSlotData { kind: StackSlotKind::ExplicitSlot, size: 16 },
                            ));
                            arg_ptr.store(self, arg, MemFlags::trusted());
                            (AbiParam::new(self.pointer_type), arg_ptr.get_addr(self))
                        } else {
                            (param, arg)
                        }
                    })
                    .unzip();

            let indirect_ret_val = returns.len() == 1 && returns[0].value_type == types::I128;

            if indirect_ret_val {
                params.insert(0, AbiParam::new(self.pointer_type));
                let ret_ptr =
                    Pointer::stack_slot(self.bcx.create_sized_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size: 16,
                    }));
                args.insert(0, ret_ptr.get_addr(self));
                self.lib_call_unadjusted(name, params, vec![], &args);
                return Cow::Owned(vec![ret_ptr.load(self, types::I128, MemFlags::trusted())]);
            } else {
                return self.lib_call_unadjusted(name, params, returns, &args);
            }
        }

        self.lib_call_unadjusted(name, params, returns, args)
    }

    pub(crate) fn lib_call_unadjusted(
        &mut self,
        name: &str,
        params: Vec<AbiParam>,
        returns: Vec<AbiParam>,
        args: &[Value],
    ) -> Cow<'_, [Value]> {
        let sig = Signature { params, returns, call_conv: self.target_config.default_call_conv };
        let func_id = self.module.declare_function(name, Linkage::Import, &sig).unwrap();
        let func_ref = self.module.declare_func_in_func(func_id, &mut self.bcx.func);
        if self.clif_comments.enabled() {
            self.add_comment(func_ref, format!("{:?}", name));
        }
        let call_inst = self.bcx.ins().call(func_ref, args);
        if self.clif_comments.enabled() {
            self.add_comment(call_inst, format!("lib_call {}", name));
        }
        let results = self.bcx.inst_results(call_inst);
        assert!(results.len() <= 2, "{}", results.len());
        Cow::Borrowed(results)
    }
}

/// Make a [`CPlace`] capable of holding value of the specified type.
fn make_local_place<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    local: Local,
    layout: TyAndLayout<'tcx>,
    is_ssa: bool,
) -> CPlace<'tcx> {
    if layout.is_unsized() {
        fx.tcx.sess.span_fatal(
            fx.mir.local_decls[local].source_info.span,
            "unsized locals are not yet supported",
        );
    }
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

    // FIXME implement variadics in cranelift
    if fn_abi.c_variadic {
        fx.tcx.sess.span_fatal(
            fx.mir.span,
            "Defining variadic functions is not yet supported by Cranelift",
        );
    }

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
                for (i, _arg_ty) in tupled_arg_tys.iter().enumerate() {
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

        let layout = fx.layout_of(ty);
        let is_ssa = ssa_analyzed[local].is_ssa(fx, ty);
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
                        place.place_field(fx, FieldIdx::new(i)).write_cvalue(fx, param);
                    }
                }
            }
        }
    }

    for local in fx.mir.vars_and_temps_iter() {
        let ty = fx.monomorphize(fx.mir.local_decls[local].ty);
        let layout = fx.layout_of(ty);

        let is_ssa = ssa_analyzed[local].is_ssa(fx, ty);

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
    source_info: mir::SourceInfo,
    func: &Operand<'tcx>,
    args: &[Operand<'tcx>],
    destination: Place<'tcx>,
    target: Option<BasicBlock>,
) {
    let func = codegen_operand(fx, func);
    let fn_sig = func.layout().ty.fn_sig(fx.tcx);

    let ret_place = codegen_place(fx, destination);

    // Handle special calls like intrinsics and empty drop glue.
    let instance = if let ty::FnDef(def_id, fn_args) = *func.layout().ty.kind() {
        let instance =
            ty::Instance::expect_resolve(fx.tcx, ty::ParamEnv::reveal_all(), def_id, fn_args)
                .polymorphize(fx.tcx);

        if fx.tcx.symbol_name(instance).name.starts_with("llvm.") {
            crate::intrinsics::codegen_llvm_intrinsic_call(
                fx,
                &fx.tcx.symbol_name(instance).name,
                fn_args,
                args,
                ret_place,
                target,
            );
            return;
        }

        match instance.def {
            InstanceDef::Intrinsic(_) => {
                crate::intrinsics::codegen_intrinsic_call(
                    fx,
                    instance,
                    args,
                    ret_place,
                    target,
                    source_info,
                );
                return;
            }
            InstanceDef::DropGlue(_, None) => {
                // empty drop glue - a nop.
                let dest = target.expect("Non terminating drop_in_place_real???");
                let ret_block = fx.get_block(dest);
                fx.bcx.ins().jump(ret_block, &[]);
                return;
            }
            _ => Some(instance),
        }
    } else {
        None
    };

    let extra_args = &args[fn_sig.inputs().skip_binder().len()..];
    let extra_args = fx.tcx.mk_type_list_from_iter(
        extra_args.iter().map(|op_arg| fx.monomorphize(op_arg.ty(fx.mir, fx.tcx))),
    );
    let fn_abi = if let Some(instance) = instance {
        RevealAllLayoutCx(fx.tcx).fn_abi_of_instance(instance, extra_args)
    } else {
        RevealAllLayoutCx(fx.tcx).fn_abi_of_fn_ptr(fn_sig, extra_args)
    };

    let is_cold = if fn_sig.abi() == Abi::RustCold {
        true
    } else {
        instance.is_some_and(|inst| {
            fx.tcx.codegen_fn_attrs(inst.def_id()).flags.contains(CodegenFnAttrFlags::COLD)
        })
    };
    if is_cold {
        fx.bcx.set_cold_block(fx.bcx.current_block().unwrap());
        if let Some(destination_block) = target {
            fx.bcx.set_cold_block(fx.get_block(destination_block));
        }
    }

    // Unpack arguments tuple for closures
    let mut args = if fn_sig.abi() == Abi::RustCall {
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
                value: pack_arg.value.value_field(fx, FieldIdx::new(i)),
                is_owned: pack_arg.is_owned,
            });
        }
        args
    } else {
        args.iter().map(|arg| codegen_call_argument_operand(fx, arg)).collect::<Vec<_>>()
    };

    // Pass the caller location for `#[track_caller]`.
    if instance.is_some_and(|inst| inst.def.requires_caller_location(fx.tcx)) {
        let caller_location = fx.get_caller_location(source_info);
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
            let sig = clif_sig_from_fn_abi(fx.tcx, fx.target_config.default_call_conv, &fn_abi);
            let sig = fx.bcx.import_signature(sig);

            (CallTarget::Indirect(sig, method), Some(ptr.get_addr(fx)))
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

            let func = func.load_scalar(fx);
            let sig = clif_sig_from_fn_abi(fx.tcx, fx.target_config.default_call_conv, &fn_abi);
            let sig = fx.bcx.import_signature(sig);

            (CallTarget::Indirect(sig, func), None)
        }
    };

    self::returning::codegen_with_call_return_arg(fx, &fn_abi.ret, ret_place, |fx, return_ptr| {
        let call_args = return_ptr
            .into_iter()
            .chain(first_arg_override.into_iter())
            .chain(
                args.into_iter()
                    .enumerate()
                    .skip(if first_arg_override.is_some() { 1 } else { 0 })
                    .flat_map(|(i, arg)| {
                        adjust_arg_for_abi(fx, arg.value, &fn_abi.args[i], arg.is_owned).into_iter()
                    }),
            )
            .collect::<Vec<Value>>();

        let call_inst = match func_ref {
            CallTarget::Direct(func_ref) => fx.bcx.ins().call(func_ref, &call_args),
            CallTarget::Indirect(sig, func_ptr) => {
                fx.bcx.ins().call_indirect(sig, func_ptr, &call_args)
            }
        };

        // FIXME find a cleaner way to support varargs
        if fn_sig.c_variadic() {
            if !matches!(fn_sig.abi(), Abi::C { .. }) {
                fx.tcx.sess.span_fatal(
                    source_info.span,
                    format!("Variadic call for non-C abi {:?}", fn_sig.abi()),
                );
            }
            let sig_ref = fx.bcx.func.dfg.call_signature(call_inst).unwrap();
            let abi_params = call_args
                .into_iter()
                .map(|arg| {
                    let ty = fx.bcx.func.dfg.value_type(arg);
                    if !ty.is_int() {
                        // FIXME set %al to upperbound on float args once floats are supported
                        fx.tcx.sess.span_fatal(
                            source_info.span,
                            format!("Non int ty {:?} for variadic call", ty),
                        );
                    }
                    AbiParam::new(ty)
                })
                .collect::<Vec<AbiParam>>();
            fx.bcx.func.dfg.signatures[sig_ref].params = abi_params;
        }

        call_inst
    });

    if let Some(dest) = target {
        let ret_block = fx.get_block(dest);
        fx.bcx.ins().jump(ret_block, &[]);
    } else {
        fx.bcx.ins().trap(TrapCode::UnreachableCodeReached);
    }
}

pub(crate) fn codegen_drop<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    source_info: mir::SourceInfo,
    drop_place: CPlace<'tcx>,
) {
    let ty = drop_place.layout().ty;
    let drop_instance = Instance::resolve_drop_in_place(fx.tcx, ty).polymorphize(fx.tcx);

    if let ty::InstanceDef::DropGlue(_, None) = drop_instance.def {
        // we don't actually need to drop anything
    } else {
        match ty.kind() {
            ty::Dynamic(_, _, ty::Dyn) => {
                // IN THIS ARM, WE HAVE:
                // ty = *mut (dyn Trait)
                // which is: exists<T> ( *mut T,    Vtable<T: Trait> )
                //                       args[0]    args[1]
                //
                // args = ( Data, Vtable )
                //                  |
                //                  v
                //                /-------\
                //                | ...   |
                //                \-------/
                //
                let (ptr, vtable) = drop_place.to_ptr_unsized();
                let ptr = ptr.get_addr(fx);
                let drop_fn = crate::vtable::drop_fn_of_obj(fx, vtable);

                // FIXME(eddyb) perhaps move some of this logic into
                // `Instance::resolve_drop_in_place`?
                let virtual_drop = Instance {
                    def: ty::InstanceDef::Virtual(drop_instance.def_id(), 0),
                    args: drop_instance.args,
                };
                let fn_abi =
                    RevealAllLayoutCx(fx.tcx).fn_abi_of_instance(virtual_drop, ty::List::empty());

                let sig = clif_sig_from_fn_abi(fx.tcx, fx.target_config.default_call_conv, &fn_abi);
                let sig = fx.bcx.import_signature(sig);
                fx.bcx.ins().call_indirect(sig, drop_fn, &[ptr]);
            }
            ty::Dynamic(_, _, ty::DynStar) => {
                // IN THIS ARM, WE HAVE:
                // ty = *mut (dyn* Trait)
                // which is: *mut exists<T: sizeof(T) == sizeof(usize)> (T, Vtable<T: Trait>)
                //
                // args = [ * ]
                //          |
                //          v
                //      ( Data, Vtable )
                //                |
                //                v
                //              /-------\
                //              | ...   |
                //              \-------/
                //
                //
                // WE CAN CONVERT THIS INTO THE ABOVE LOGIC BY DOING
                //
                // data = &(*args[0]).0    // gives a pointer to Data above (really the same pointer)
                // vtable = (*args[0]).1   // loads the vtable out
                // (data, vtable)          // an equivalent Rust `*mut dyn Trait`
                //
                // SO THEN WE CAN USE THE ABOVE CODE.
                let (data, vtable) = drop_place.to_cvalue(fx).dyn_star_force_data_on_stack(fx);
                let drop_fn = crate::vtable::drop_fn_of_obj(fx, vtable);

                let virtual_drop = Instance {
                    def: ty::InstanceDef::Virtual(drop_instance.def_id(), 0),
                    args: drop_instance.args,
                };
                let fn_abi =
                    RevealAllLayoutCx(fx.tcx).fn_abi_of_instance(virtual_drop, ty::List::empty());

                let sig = clif_sig_from_fn_abi(fx.tcx, fx.target_config.default_call_conv, &fn_abi);
                let sig = fx.bcx.import_signature(sig);
                fx.bcx.ins().call_indirect(sig, drop_fn, &[data]);
            }
            _ => {
                assert!(!matches!(drop_instance.def, InstanceDef::Virtual(_, _)));

                let fn_abi =
                    RevealAllLayoutCx(fx.tcx).fn_abi_of_instance(drop_instance, ty::List::empty());

                let arg_value = drop_place.place_ref(
                    fx,
                    fx.layout_of(Ty::new_ref(
                        fx.tcx,
                        fx.tcx.lifetimes.re_erased,
                        TypeAndMut { ty, mutbl: crate::rustc_hir::Mutability::Mut },
                    )),
                );
                let arg_value = adjust_arg_for_abi(fx, arg_value, &fn_abi.args[0], true);

                let mut call_args: Vec<Value> = arg_value.into_iter().collect::<Vec<_>>();

                if drop_instance.def.requires_caller_location(fx.tcx) {
                    // Pass the caller location for `#[track_caller]`.
                    let caller_location = fx.get_caller_location(source_info);
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
