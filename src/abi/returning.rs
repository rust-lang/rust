//! Return value handling

use crate::abi::pass_mode::*;
use crate::prelude::*;

use rustc_middle::ty::layout::FnAbiExt;
use rustc_target::abi::call::{ArgAbi, FnAbi, PassMode};

/// Can the given type be returned into an ssa var or does it need to be returned on the stack.
pub(crate) fn can_return_to_ssa_var<'tcx>(
    fx: &FunctionCx<'_, 'tcx, impl Module>,
    func: &mir::Operand<'tcx>,
    args: &[mir::Operand<'tcx>],
) -> bool {
    let fn_ty = fx.monomorphize(func.ty(fx.mir, fx.tcx));
    let fn_sig = fx
        .tcx
        .normalize_erasing_late_bound_regions(ParamEnv::reveal_all(), fn_ty.fn_sig(fx.tcx));

    // Handle special calls like instrinsics and empty drop glue.
    let instance = if let ty::FnDef(def_id, substs) = *fn_ty.kind() {
        let instance = ty::Instance::resolve(fx.tcx, ty::ParamEnv::reveal_all(), def_id, substs)
            .unwrap()
            .unwrap()
            .polymorphize(fx.tcx);

        match instance.def {
            InstanceDef::Intrinsic(_) | InstanceDef::DropGlue(_, _) => {
                return true;
            }
            _ => Some(instance),
        }
    } else {
        None
    };

    let extra_args = &args[fn_sig.inputs().len()..];
    let extra_args = extra_args
        .iter()
        .map(|op_arg| fx.monomorphize(op_arg.ty(fx.mir, fx.tcx)))
        .collect::<Vec<_>>();
    let fn_abi = if let Some(instance) = instance {
        FnAbi::of_instance(&RevealAllLayoutCx(fx.tcx), instance, &extra_args)
    } else {
        FnAbi::of_fn_ptr(
            &RevealAllLayoutCx(fx.tcx),
            fn_ty.fn_sig(fx.tcx),
            &extra_args,
        )
    };
    match fn_abi.ret.mode {
        PassMode::Ignore | PassMode::Direct(_) | PassMode::Pair(_, _) => true,
        // FIXME Make it possible to return Cast and Indirect to an ssa var.
        PassMode::Cast(_) | PassMode::Indirect { .. } => false,
    }
}

/// Return a place where the return value of the current function can be written to. If necessary
/// this adds an extra parameter pointing to where the return value needs to be stored.
pub(super) fn codegen_return_param<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    ssa_analyzed: &rustc_index::vec::IndexVec<Local, crate::analyze::SsaKind>,
    start_block: Block,
) -> CPlace<'tcx> {
    let (ret_place, ret_param) = match fx.fn_abi.as_ref().unwrap().ret.mode {
        PassMode::Ignore => (
            CPlace::no_place(fx.fn_abi.as_ref().unwrap().ret.layout),
            Empty,
        ),
        PassMode::Direct(_) | PassMode::Pair(_, _) => {
            let is_ssa = ssa_analyzed[RETURN_PLACE] == crate::analyze::SsaKind::Ssa;
            (
                super::make_local_place(
                    fx,
                    RETURN_PLACE,
                    fx.fn_abi.as_ref().unwrap().ret.layout,
                    is_ssa,
                ),
                Empty,
            )
        }
        PassMode::Cast(_)
        | PassMode::Indirect {
            attrs: _,
            extra_attrs: None,
            on_stack: _,
        } => {
            let ret_param = fx.bcx.append_block_param(start_block, fx.pointer_type);
            (
                CPlace::for_ptr(
                    Pointer::new(ret_param),
                    fx.fn_abi.as_ref().unwrap().ret.layout,
                ),
                Single(ret_param),
            )
        }
        PassMode::Indirect {
            attrs: _,
            extra_attrs: Some(_),
            on_stack: _,
        } => unreachable!("unsized return value"),
    };

    #[cfg(not(debug_assertions))]
    let _ = ret_param;

    #[cfg(debug_assertions)]
    crate::abi::comments::add_arg_comment(
        fx,
        "ret",
        Some(RETURN_PLACE),
        None,
        ret_param,
        fx.fn_abi.as_ref().unwrap().ret.mode,
        fx.fn_abi.as_ref().unwrap().ret.layout,
    );

    ret_place
}

/// Invokes the closure with if necessary a value representing the return pointer. When the closure
/// returns the call return value(s) if any are written to the correct place.
pub(super) fn codegen_with_call_return_arg<'tcx, M: Module, T>(
    fx: &mut FunctionCx<'_, 'tcx, M>,
    ret_arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
    ret_place: Option<CPlace<'tcx>>,
    f: impl FnOnce(&mut FunctionCx<'_, 'tcx, M>, Option<Value>) -> (Inst, T),
) -> (Inst, T) {
    let return_ptr = match ret_arg_abi.mode {
        PassMode::Ignore => None,
        PassMode::Cast(_)
        | PassMode::Indirect {
            attrs: _,
            extra_attrs: None,
            on_stack: _,
        } => match ret_place {
            Some(ret_place) => Some(ret_place.to_ptr().get_addr(fx)),
            None => Some(fx.bcx.ins().iconst(fx.pointer_type, 43)), // FIXME allocate temp stack slot
        },
        PassMode::Indirect {
            attrs: _,
            extra_attrs: Some(_),
            on_stack: _,
        } => unreachable!("unsized return value"),
        PassMode::Direct(_) | PassMode::Pair(_, _) => None,
    };

    let (call_inst, meta) = f(fx, return_ptr);

    match ret_arg_abi.mode {
        PassMode::Ignore => {}
        PassMode::Direct(_) => {
            if let Some(ret_place) = ret_place {
                let ret_val = fx.bcx.inst_results(call_inst)[0];
                ret_place.write_cvalue(fx, CValue::by_val(ret_val, ret_arg_abi.layout));
            }
        }
        PassMode::Pair(_, _) => {
            if let Some(ret_place) = ret_place {
                let ret_val_a = fx.bcx.inst_results(call_inst)[0];
                let ret_val_b = fx.bcx.inst_results(call_inst)[1];
                ret_place.write_cvalue(
                    fx,
                    CValue::by_val_pair(ret_val_a, ret_val_b, ret_arg_abi.layout),
                );
            }
        }
        PassMode::Cast(_)
        | PassMode::Indirect {
            attrs: _,
            extra_attrs: None,
            on_stack: _,
        } => {}
        PassMode::Indirect {
            attrs: _,
            extra_attrs: Some(_),
            on_stack: _,
        } => unreachable!("unsized return value"),
    }

    (call_inst, meta)
}

/// Codegen a return instruction with the right return value(s) if any.
pub(crate) fn codegen_return(fx: &mut FunctionCx<'_, '_, impl Module>) {
    match fx.fn_abi.as_ref().unwrap().ret.mode {
        PassMode::Ignore
        | PassMode::Cast(_)
        | PassMode::Indirect {
            attrs: _,
            extra_attrs: None,
            on_stack: _,
        } => {
            fx.bcx.ins().return_(&[]);
        }
        PassMode::Indirect {
            attrs: _,
            extra_attrs: Some(_),
            on_stack: _,
        } => unreachable!("unsized return value"),
        PassMode::Direct(_) => {
            let place = fx.get_local_place(RETURN_PLACE);
            let ret_val = place.to_cvalue(fx).load_scalar(fx);
            fx.bcx.ins().return_(&[ret_val]);
        }
        PassMode::Pair(_, _) => {
            let place = fx.get_local_place(RETURN_PLACE);
            let (ret_val_a, ret_val_b) = place.to_cvalue(fx).load_scalar_pair(fx);
            fx.bcx.ins().return_(&[ret_val_a, ret_val_b]);
        }
    }
}
