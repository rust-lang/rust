use crate::prelude::*;
use crate::abi::pass_mode::*;

pub fn codegen_return_param(
    fx: &mut FunctionCx<impl Backend>,
    ssa_analyzed: &HashMap<Local, crate::analyze::Flags>,
    start_ebb: Ebb,
) {
    let ret_layout = fx.return_layout();
    let output_pass_mode = get_pass_mode(fx.tcx, fx.return_layout());

    let ret_param = match output_pass_mode {
        PassMode::NoPass => {
            fx.local_map
                .insert(RETURN_PLACE, CPlace::no_place(ret_layout));
            Empty
        }
        PassMode::ByVal(_) | PassMode::ByValPair(_, _) => {
            let is_ssa = !ssa_analyzed
                .get(&RETURN_PLACE)
                .unwrap()
                .contains(crate::analyze::Flags::NOT_SSA);

            super::local_place(fx, RETURN_PLACE, ret_layout, is_ssa);

            Empty
        }
        PassMode::ByRef => {
            let ret_param = fx.bcx.append_ebb_param(start_ebb, fx.pointer_type);
            fx.local_map.insert(
                RETURN_PLACE,
                CPlace::for_addr(ret_param, ret_layout),
            );

            Single(ret_param)
        }
    };

    #[cfg(debug_assertions)]
    crate::abi::comments::add_arg_comment(
        fx,
        "ret",
        RETURN_PLACE,
        None,
        ret_param,
        output_pass_mode,
        ret_layout.ty,
    );
}

pub fn codegen_with_call_return_arg<'tcx, B: Backend, T>(
    fx: &mut FunctionCx<'_, 'tcx, B>,
    fn_sig: FnSig<'tcx>,
    ret_place: Option<CPlace<'tcx>>,
    f: impl FnOnce(&mut FunctionCx<'_, 'tcx, B>, Option<Value>) -> (Inst, T),
) -> (Inst, T) {
    let ret_layout = fx.layout_of(fn_sig.output());

    let output_pass_mode = get_pass_mode(fx.tcx, ret_layout);
    let return_ptr = match output_pass_mode {
        PassMode::NoPass => None,
        PassMode::ByRef => match ret_place {
            Some(ret_place) => Some(ret_place.to_addr(fx)),
            None => Some(fx.bcx.ins().iconst(fx.pointer_type, 43)),
        },
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
        PassMode::ByRef => {}
    }

    (call_inst, meta)
}

pub fn codegen_return(fx: &mut FunctionCx<impl Backend>) {
    match get_pass_mode(fx.tcx, fx.return_layout()) {
        PassMode::NoPass | PassMode::ByRef => {
            fx.bcx.ins().return_(&[]);
        }
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
