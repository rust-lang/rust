//! Return value handling

use crate::prelude::*;

use rustc_target::abi::call::{ArgAbi, PassMode};
use smallvec::{smallvec, SmallVec};

/// Return a place where the return value of the current function can be written to. If necessary
/// this adds an extra parameter pointing to where the return value needs to be stored.
pub(super) fn codegen_return_param<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    ssa_analyzed: &rustc_index::vec::IndexVec<Local, crate::analyze::SsaKind>,
    block_params_iter: &mut impl Iterator<Item = Value>,
) -> CPlace<'tcx> {
    let (ret_place, ret_param): (_, SmallVec<[_; 2]>) = match fx.fn_abi.as_ref().unwrap().ret.mode {
        PassMode::Ignore | PassMode::Direct(_) | PassMode::Pair(_, _) | PassMode::Cast(..) => {
            let is_ssa = ssa_analyzed[RETURN_PLACE] == crate::analyze::SsaKind::Ssa;
            (
                super::make_local_place(
                    fx,
                    RETURN_PLACE,
                    fx.fn_abi.as_ref().unwrap().ret.layout,
                    is_ssa,
                ),
                smallvec![],
            )
        }
        PassMode::Indirect { attrs: _, extra_attrs: None, on_stack: _ } => {
            let ret_param = block_params_iter.next().unwrap();
            assert_eq!(fx.bcx.func.dfg.value_type(ret_param), fx.pointer_type);
            (
                CPlace::for_ptr(Pointer::new(ret_param), fx.fn_abi.as_ref().unwrap().ret.layout),
                smallvec![ret_param],
            )
        }
        PassMode::Indirect { attrs: _, extra_attrs: Some(_), on_stack: _ } => {
            unreachable!("unsized return value")
        }
    };

    crate::abi::comments::add_arg_comment(
        fx,
        "ret",
        Some(RETURN_PLACE),
        None,
        &ret_param,
        &fx.fn_abi.as_ref().unwrap().ret.mode,
        fx.fn_abi.as_ref().unwrap().ret.layout,
    );

    ret_place
}

/// Invokes the closure with if necessary a value representing the return pointer. When the closure
/// returns the call return value(s) if any are written to the correct place.
pub(super) fn codegen_with_call_return_arg<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    ret_arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
    ret_place: CPlace<'tcx>,
    f: impl FnOnce(&mut FunctionCx<'_, '_, 'tcx>, Option<Value>) -> Inst,
) {
    let (ret_temp_place, return_ptr) = match ret_arg_abi.mode {
        PassMode::Ignore => (None, None),
        PassMode::Indirect { attrs: _, extra_attrs: None, on_stack: _ } => {
            if matches!(ret_place.inner(), CPlaceInner::Addr(_, None)) {
                // This is an optimization to prevent unnecessary copies of the return value when
                // the return place is already a memory place as opposed to a register.
                // This match arm can be safely removed.
                (None, Some(ret_place.to_ptr().get_addr(fx)))
            } else {
                let place = CPlace::new_stack_slot(fx, ret_arg_abi.layout);
                (Some(place), Some(place.to_ptr().get_addr(fx)))
            }
        }
        PassMode::Indirect { attrs: _, extra_attrs: Some(_), on_stack: _ } => {
            unreachable!("unsized return value")
        }
        PassMode::Direct(_) | PassMode::Pair(_, _) | PassMode::Cast(..) => (None, None),
    };

    let call_inst = f(fx, return_ptr);

    match ret_arg_abi.mode {
        PassMode::Ignore => {}
        PassMode::Direct(_) => {
            let ret_val = fx.bcx.inst_results(call_inst)[0];
            ret_place.write_cvalue(fx, CValue::by_val(ret_val, ret_arg_abi.layout));
        }
        PassMode::Pair(_, _) => {
            let ret_val_a = fx.bcx.inst_results(call_inst)[0];
            let ret_val_b = fx.bcx.inst_results(call_inst)[1];
            ret_place
                .write_cvalue(fx, CValue::by_val_pair(ret_val_a, ret_val_b, ret_arg_abi.layout));
        }
        PassMode::Cast(ref cast, _) => {
            let results =
                fx.bcx.inst_results(call_inst).iter().copied().collect::<SmallVec<[Value; 2]>>();
            let result =
                super::pass_mode::from_casted_value(fx, &results, ret_place.layout(), cast);
            ret_place.write_cvalue(fx, result);
        }
        PassMode::Indirect { attrs: _, extra_attrs: None, on_stack: _ } => {
            if let Some(ret_temp_place) = ret_temp_place {
                // If ret_temp_place is None, it is not necessary to copy the return value.
                let ret_temp_value = ret_temp_place.to_cvalue(fx);
                ret_place.write_cvalue(fx, ret_temp_value);
            }
        }
        PassMode::Indirect { attrs: _, extra_attrs: Some(_), on_stack: _ } => {
            unreachable!("unsized return value")
        }
    }
}

/// Codegen a return instruction with the right return value(s) if any.
pub(crate) fn codegen_return(fx: &mut FunctionCx<'_, '_, '_>) {
    match fx.fn_abi.as_ref().unwrap().ret.mode {
        PassMode::Ignore | PassMode::Indirect { attrs: _, extra_attrs: None, on_stack: _ } => {
            fx.bcx.ins().return_(&[]);
        }
        PassMode::Indirect { attrs: _, extra_attrs: Some(_), on_stack: _ } => {
            unreachable!("unsized return value")
        }
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
        PassMode::Cast(ref cast, _) => {
            let place = fx.get_local_place(RETURN_PLACE);
            let ret_val = place.to_cvalue(fx);
            let ret_vals = super::pass_mode::to_casted_value(fx, ret_val, cast);
            fx.bcx.ins().return_(&ret_vals);
        }
    }
}
