//! Replaces 128-bit operators with lang item calls where necessary

use cranelift_codegen::ir::ArgumentPurpose;

use crate::prelude::*;

pub(crate) fn maybe_codegen<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    bin_op: BinOp,
    lhs: CValue<'tcx>,
    rhs: CValue<'tcx>,
) -> Option<CValue<'tcx>> {
    if lhs.layout().ty != fx.tcx.types.u128
        && lhs.layout().ty != fx.tcx.types.i128
        && rhs.layout().ty != fx.tcx.types.u128
        && rhs.layout().ty != fx.tcx.types.i128
    {
        return None;
    }

    let is_signed = type_sign(lhs.layout().ty);

    match bin_op {
        BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => None,
        BinOp::Add | BinOp::AddUnchecked | BinOp::Sub | BinOp::SubUnchecked => None,
        BinOp::Mul | BinOp::MulUnchecked => None,
        BinOp::Offset => unreachable!("offset should only be used on pointers, not 128bit ints"),
        BinOp::Div | BinOp::Rem => {
            let name = match (bin_op, is_signed) {
                (BinOp::Div, false) => "__udivti3",
                (BinOp::Div, true) => "__divti3",
                (BinOp::Rem, false) => "__umodti3",
                (BinOp::Rem, true) => "__modti3",
                _ => unreachable!(),
            };
            let args = [lhs.load_scalar(fx), rhs.load_scalar(fx)];
            let ret_val = fx.lib_call(
                name,
                vec![AbiParam::new(types::I128), AbiParam::new(types::I128)],
                vec![AbiParam::new(types::I128)],
                &args,
            )[0];
            Some(CValue::by_val(ret_val, lhs.layout()))
        }
        BinOp::Lt | BinOp::Le | BinOp::Eq | BinOp::Ge | BinOp::Gt | BinOp::Ne | BinOp::Cmp => None,
        BinOp::Shl | BinOp::ShlUnchecked | BinOp::Shr | BinOp::ShrUnchecked => None,
        BinOp::AddWithOverflow | BinOp::SubWithOverflow | BinOp::MulWithOverflow => unreachable!(),
    }
}

pub(crate) fn maybe_codegen_mul_checked<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    lhs: CValue<'tcx>,
    rhs: CValue<'tcx>,
) -> Option<CValue<'tcx>> {
    if lhs.layout().ty != fx.tcx.types.u128
        && lhs.layout().ty != fx.tcx.types.i128
        && rhs.layout().ty != fx.tcx.types.u128
        && rhs.layout().ty != fx.tcx.types.i128
    {
        return None;
    }

    let is_signed = type_sign(lhs.layout().ty);
    let oflow_out_place = CPlace::new_stack_slot(fx, fx.layout_of(fx.tcx.types.i32));
    let param_types = vec![
        AbiParam::new(types::I128),
        AbiParam::new(types::I128),
        AbiParam::special(fx.pointer_type, ArgumentPurpose::Normal),
    ];
    let args = [lhs.load_scalar(fx), rhs.load_scalar(fx), oflow_out_place.to_ptr().get_addr(fx)];
    let ret = fx.lib_call(
        if is_signed { "__rust_i128_mulo" } else { "__rust_u128_mulo" },
        param_types,
        vec![AbiParam::new(types::I128)],
        &args,
    );
    let mul = ret[0];
    let oflow = oflow_out_place.to_cvalue(fx).load_scalar(fx);
    let oflow = clif_intcast(fx, oflow, types::I8, false);
    let layout = fx.layout_of(Ty::new_tup(fx.tcx, &[lhs.layout().ty, fx.tcx.types.bool]));
    Some(CValue::by_val_pair(mul, oflow, layout))
}
