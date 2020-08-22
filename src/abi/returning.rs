use crate::abi::pass_mode::*;
use crate::prelude::*;

fn return_layout<'a, 'tcx>(fx: &mut FunctionCx<'a, 'tcx, impl Backend>) -> TyAndLayout<'tcx> {
    fx.layout_of(fx.monomorphize(&fx.mir.local_decls[RETURN_PLACE].ty))
}

pub(crate) fn can_return_to_ssa_var<'tcx>(tcx: TyCtxt<'tcx>, dest_layout: TyAndLayout<'tcx>) -> bool {
    match get_pass_mode(tcx, dest_layout) {
        PassMode::NoPass | PassMode::ByVal(_) => true,
        // FIXME Make it possible to return ByValPair and ByRef to an ssa var.
        PassMode::ByValPair(_, _) | PassMode::ByRef { size: _ } => false
    }
}

pub(super) fn codegen_return_param(
    fx: &mut FunctionCx<'_, '_, impl Backend>,
    ssa_analyzed: &rustc_index::vec::IndexVec<Local, crate::analyze::SsaKind>,
    start_block: Block,
) {
    let ret_layout = return_layout(fx);
    let ret_pass_mode = get_pass_mode(fxcodegen_cx.tcx, ret_layout);
    let ret_param = match ret_pass_mode {
        PassMode::NoPass => {
            fx.local_map
                .insert(RETURN_PLACE, CPlace::no_place(ret_layout));
            Empty
        }
        PassMode::ByVal(_) | PassMode::ByValPair(_, _) => {
            let is_ssa = ssa_analyzed[RETURN_PLACE] == crate::analyze::SsaKind::Ssa;

            super::local_place(fx, RETURN_PLACE, ret_layout, is_ssa);

            Empty
        }
        PassMode::ByRef { size: Some(_) } => {
            let ret_param = fx.bcx.append_block_param(start_block, fx.pointer_type);
            fx.local_map
                .insert(RETURN_PLACE, CPlace::for_ptr(Pointer::new(ret_param), ret_layout));

            Single(ret_param)
        }
        PassMode::ByRef { size: None } => todo!(),
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
        ret_pass_mode,
        ret_layout.ty,
    );
}

pub(super) fn codegen_with_call_return_arg<'tcx, B: Backend, T>(
    fx: &mut FunctionCx<'_, 'tcx, B>,
    fn_sig: FnSig<'tcx>,
    ret_place: Option<CPlace<'tcx>>,
    f: impl FnOnce(&mut FunctionCx<'_, 'tcx, B>, Option<Value>) -> (Inst, T),
) -> (Inst, T) {
    let ret_layout = fx.layout_of(fn_sig.output());

    let output_pass_mode = get_pass_mode(fxcodegen_cx.tcx, ret_layout);
    let return_ptr = match output_pass_mode {
        PassMode::NoPass => None,
        PassMode::ByRef { size: Some(_)} => match ret_place {
            Some(ret_place) => Some(ret_place.to_ptr().get_addr(fx)),
            None => Some(fx.bcx.ins().iconst(fx.pointer_type, 43)), // FIXME allocate temp stack slot
        },
        PassMode::ByRef { size: None } => todo!(),
        PassMode::ByVal(_) | PassMode::ByValPair(_, _) => None,
    };

    let (call_inst, meta) = f(fx, return_ptr);

    match output_pass_mode {
        PassMode::NoPass => {}
        PassMode::ByVal(_) => {
            if let Some(ret_place) = ret_place {
                let ret_val = fx.bcx.inst_results(call_inst)[0];
                ret_place.write_cvalue(fx, CValue::by_val(ret_val, ret_layout));
            }
        }
        PassMode::ByValPair(_, _) => {
            if let Some(ret_place) = ret_place {
                let ret_val_a = fx.bcx.inst_results(call_inst)[0];
                let ret_val_b = fx.bcx.inst_results(call_inst)[1];
                ret_place.write_cvalue(fx, CValue::by_val_pair(ret_val_a, ret_val_b, ret_layout));
            }
        }
        PassMode::ByRef { size: Some(_) } => {}
        PassMode::ByRef { size: None } => todo!(),
    }

    (call_inst, meta)
}

pub(crate) fn codegen_return(fx: &mut FunctionCx<'_, '_, impl Backend>) {
    match get_pass_mode(fxcodegen_cx.tcx, return_layout(fx)) {
        PassMode::NoPass | PassMode::ByRef { size: Some(_) } => {
            fx.bcx.ins().return_(&[]);
        }
        PassMode::ByRef { size: None } => todo!(),
        PassMode::ByVal(_) => {
            let place = fx.get_local_place(RETURN_PLACE);
            let ret_val = place.to_cvalue(fx).load_scalar(fx);
            fx.bcx.ins().return_(&[ret_val]);
        }
        PassMode::ByValPair(_, _) => {
            let place = fx.get_local_place(RETURN_PLACE);
            let (ret_val_a, ret_val_b) = place.to_cvalue(fx).load_scalar_pair(fx);
            fx.bcx.ins().return_(&[ret_val_a, ret_val_b]);
        }
    }
}
