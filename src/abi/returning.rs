//! Return value handling

use crate::abi::pass_mode::*;
use crate::prelude::*;

use rustc_target::abi::call::PassMode as RustcPassMode;

fn return_layout<'a, 'tcx>(fx: &mut FunctionCx<'a, 'tcx, impl Module>) -> TyAndLayout<'tcx> {
    fx.layout_of(fx.monomorphize(&fx.mir.local_decls[RETURN_PLACE].ty))
}

/// Can the given type be returned into an ssa var or does it need to be returned on the stack.
pub(crate) fn can_return_to_ssa_var<'tcx>(
    tcx: TyCtxt<'tcx>,
    dest_layout: TyAndLayout<'tcx>,
) -> bool {
    match get_arg_abi(tcx, dest_layout).mode {
        RustcPassMode::Ignore | RustcPassMode::Direct(_) | RustcPassMode::Pair(_, _) => true,
        // FIXME Make it possible to return Cast and Indirect to an ssa var.
        RustcPassMode::Cast(_) | RustcPassMode::Indirect { .. } => false,
    }
}

/// Return a place where the return value of the current function can be written to. If necessary
/// this adds an extra parameter pointing to where the return value needs to be stored.
pub(super) fn codegen_return_param<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    ssa_analyzed: &rustc_index::vec::IndexVec<Local, crate::analyze::SsaKind>,
    start_block: Block,
) -> CPlace<'tcx> {
    let ret_layout = return_layout(fx);
    let ret_arg_abi = get_arg_abi(fx.tcx, ret_layout);
    let (ret_place, ret_param) = match ret_arg_abi.mode {
        RustcPassMode::Ignore => (CPlace::no_place(ret_layout), Empty),
        RustcPassMode::Direct(_) | RustcPassMode::Pair(_, _) => {
            let is_ssa = ssa_analyzed[RETURN_PLACE] == crate::analyze::SsaKind::Ssa;
            (
                super::make_local_place(fx, RETURN_PLACE, ret_layout, is_ssa),
                Empty,
            )
        }
        RustcPassMode::Cast(_)
        | RustcPassMode::Indirect {
            attrs: _,
            extra_attrs: None,
            on_stack: _,
        } => {
            let ret_param = fx.bcx.append_block_param(start_block, fx.pointer_type);
            (
                CPlace::for_ptr(Pointer::new(ret_param), ret_layout),
                Single(ret_param),
            )
        }
        RustcPassMode::Indirect {
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
        &ret_arg_abi,
    );

    ret_place
}

/// Invokes the closure with if necessary a value representing the return pointer. When the closure
/// returns the call return value(s) if any are written to the correct place.
pub(super) fn codegen_with_call_return_arg<'tcx, M: Module, T>(
    fx: &mut FunctionCx<'_, 'tcx, M>,
    fn_sig: FnSig<'tcx>,
    ret_place: Option<CPlace<'tcx>>,
    f: impl FnOnce(&mut FunctionCx<'_, 'tcx, M>, Option<Value>) -> (Inst, T),
) -> (Inst, T) {
    let ret_layout = fx.layout_of(fn_sig.output());

    let output_arg_abi = get_arg_abi(fx.tcx, ret_layout);
    let return_ptr = match output_arg_abi.mode {
        RustcPassMode::Ignore => None,
        RustcPassMode::Cast(_)
        | RustcPassMode::Indirect {
            attrs: _,
            extra_attrs: None,
            on_stack: _,
        } => match ret_place {
            Some(ret_place) => Some(ret_place.to_ptr().get_addr(fx)),
            None => Some(fx.bcx.ins().iconst(fx.pointer_type, 43)), // FIXME allocate temp stack slot
        },
        RustcPassMode::Indirect {
            attrs: _,
            extra_attrs: Some(_),
            on_stack: _,
        } => unreachable!("unsized return value"),
        RustcPassMode::Direct(_) | RustcPassMode::Pair(_, _) => None,
    };

    let (call_inst, meta) = f(fx, return_ptr);

    match output_arg_abi.mode {
        RustcPassMode::Ignore => {}
        RustcPassMode::Direct(_) => {
            if let Some(ret_place) = ret_place {
                let ret_val = fx.bcx.inst_results(call_inst)[0];
                ret_place.write_cvalue(fx, CValue::by_val(ret_val, ret_layout));
            }
        }
        RustcPassMode::Pair(_, _) => {
            if let Some(ret_place) = ret_place {
                let ret_val_a = fx.bcx.inst_results(call_inst)[0];
                let ret_val_b = fx.bcx.inst_results(call_inst)[1];
                ret_place.write_cvalue(fx, CValue::by_val_pair(ret_val_a, ret_val_b, ret_layout));
            }
        }
        RustcPassMode::Cast(_)
        | RustcPassMode::Indirect {
            attrs: _,
            extra_attrs: None,
            on_stack: _,
        } => {}
        RustcPassMode::Indirect {
            attrs: _,
            extra_attrs: Some(_),
            on_stack: _,
        } => unreachable!("unsized return value"),
    }

    (call_inst, meta)
}

/// Codegen a return instruction with the right return value(s) if any.
pub(crate) fn codegen_return(fx: &mut FunctionCx<'_, '_, impl Module>) {
    match get_arg_abi(fx.tcx, return_layout(fx)).mode {
        RustcPassMode::Ignore
        | RustcPassMode::Cast(_)
        | RustcPassMode::Indirect {
            attrs: _,
            extra_attrs: None,
            on_stack: _,
        } => {
            fx.bcx.ins().return_(&[]);
        }
        RustcPassMode::Indirect {
            attrs: _,
            extra_attrs: Some(_),
            on_stack: _,
        } => unreachable!("unsized return value"),
        RustcPassMode::Direct(_) => {
            let place = fx.get_local_place(RETURN_PLACE);
            let ret_val = place.to_cvalue(fx).load_scalar(fx);
            fx.bcx.ins().return_(&[ret_val]);
        }
        RustcPassMode::Pair(_, _) => {
            let place = fx.get_local_place(RETURN_PLACE);
            let (ret_val_a, ret_val_b) = place.to_cvalue(fx).load_scalar_pair(fx);
            fx.bcx.ins().return_(&[ret_val_a, ret_val_b]);
        }
    }
}
