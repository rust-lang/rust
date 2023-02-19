//! Replaces 128-bit operators with lang item calls where necessary

use cranelift_codegen::ir::ArgumentPurpose;

use crate::prelude::*;

pub(crate) fn maybe_codegen<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    bin_op: BinOp,
    checked: bool,
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
        BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => {
            assert!(!checked);
            None
        }
        BinOp::Add | BinOp::Sub if !checked => None,
        BinOp::Mul if !checked || is_signed => {
            if !checked {
                let val_ty = if is_signed { fx.tcx.types.i128 } else { fx.tcx.types.u128 };
                if fx.tcx.sess.target.is_like_windows {
                    let ret_place = CPlace::new_stack_slot(fx, lhs.layout());
                    let (lhs_ptr, lhs_extra) = lhs.force_stack(fx);
                    let (rhs_ptr, rhs_extra) = rhs.force_stack(fx);
                    assert!(lhs_extra.is_none());
                    assert!(rhs_extra.is_none());
                    let args = [
                        ret_place.to_ptr().get_addr(fx),
                        lhs_ptr.get_addr(fx),
                        rhs_ptr.get_addr(fx),
                    ];
                    fx.lib_call(
                        "__multi3",
                        vec![
                            AbiParam::special(fx.pointer_type, ArgumentPurpose::StructReturn),
                            AbiParam::new(fx.pointer_type),
                            AbiParam::new(fx.pointer_type),
                        ],
                        vec![],
                        &args,
                    );
                    Some(ret_place.to_cvalue(fx))
                } else {
                    Some(fx.easy_call("__multi3", &[lhs, rhs], val_ty))
                }
            } else {
                let out_ty = fx.tcx.intern_tup(&[lhs.layout().ty, fx.tcx.types.bool]);
                let oflow = CPlace::new_stack_slot(fx, fx.layout_of(fx.tcx.types.i32));
                let lhs = lhs.load_scalar(fx);
                let rhs = rhs.load_scalar(fx);
                let oflow_ptr = oflow.to_ptr().get_addr(fx);
                let res = fx.lib_call(
                    "__muloti4",
                    vec![
                        AbiParam::new(types::I128),
                        AbiParam::new(types::I128),
                        AbiParam::new(fx.pointer_type),
                    ],
                    vec![AbiParam::new(types::I128)],
                    &[lhs, rhs, oflow_ptr],
                )[0];
                let oflow = oflow.to_cvalue(fx).load_scalar(fx);
                let oflow = fx.bcx.ins().ireduce(types::I8, oflow);
                Some(CValue::by_val_pair(res, oflow, fx.layout_of(out_ty)))
            }
        }
        BinOp::Add | BinOp::Sub | BinOp::Mul => {
            assert!(checked);
            let out_ty = fx.tcx.intern_tup(&[lhs.layout().ty, fx.tcx.types.bool]);
            let out_place = CPlace::new_stack_slot(fx, fx.layout_of(out_ty));
            let (param_types, args) = if fx.tcx.sess.target.is_like_windows {
                let (lhs_ptr, lhs_extra) = lhs.force_stack(fx);
                let (rhs_ptr, rhs_extra) = rhs.force_stack(fx);
                assert!(lhs_extra.is_none());
                assert!(rhs_extra.is_none());
                (
                    vec![
                        AbiParam::special(fx.pointer_type, ArgumentPurpose::StructReturn),
                        AbiParam::new(fx.pointer_type),
                        AbiParam::new(fx.pointer_type),
                    ],
                    [out_place.to_ptr().get_addr(fx), lhs_ptr.get_addr(fx), rhs_ptr.get_addr(fx)],
                )
            } else {
                (
                    vec![
                        AbiParam::special(fx.pointer_type, ArgumentPurpose::StructReturn),
                        AbiParam::new(types::I128),
                        AbiParam::new(types::I128),
                    ],
                    [out_place.to_ptr().get_addr(fx), lhs.load_scalar(fx), rhs.load_scalar(fx)],
                )
            };
            let name = match (bin_op, is_signed) {
                (BinOp::Add, false) => "__rust_u128_addo",
                (BinOp::Add, true) => "__rust_i128_addo",
                (BinOp::Sub, false) => "__rust_u128_subo",
                (BinOp::Sub, true) => "__rust_i128_subo",
                (BinOp::Mul, false) => "__rust_u128_mulo",
                _ => unreachable!(),
            };
            fx.lib_call(name, param_types, vec![], &args);
            Some(out_place.to_cvalue(fx))
        }
        BinOp::Offset => unreachable!("offset should only be used on pointers, not 128bit ints"),
        BinOp::Div | BinOp::Rem => {
            assert!(!checked);
            let name = match (bin_op, is_signed) {
                (BinOp::Div, false) => "__udivti3",
                (BinOp::Div, true) => "__divti3",
                (BinOp::Rem, false) => "__umodti3",
                (BinOp::Rem, true) => "__modti3",
                _ => unreachable!(),
            };
            if fx.tcx.sess.target.is_like_windows {
                let (lhs_ptr, lhs_extra) = lhs.force_stack(fx);
                let (rhs_ptr, rhs_extra) = rhs.force_stack(fx);
                assert!(lhs_extra.is_none());
                assert!(rhs_extra.is_none());
                let args = [lhs_ptr.get_addr(fx), rhs_ptr.get_addr(fx)];
                let ret = fx.lib_call(
                    name,
                    vec![AbiParam::new(fx.pointer_type), AbiParam::new(fx.pointer_type)],
                    vec![AbiParam::new(types::I64X2)],
                    &args,
                )[0];
                // FIXME use bitcast instead of store to get from i64x2 to i128
                let ret_place = CPlace::new_stack_slot(fx, lhs.layout());
                ret_place.to_ptr().store(fx, ret, MemFlags::trusted());
                Some(ret_place.to_cvalue(fx))
            } else {
                Some(fx.easy_call(name, &[lhs, rhs], lhs.layout().ty))
            }
        }
        BinOp::Lt | BinOp::Le | BinOp::Eq | BinOp::Ge | BinOp::Gt | BinOp::Ne => {
            assert!(!checked);
            None
        }
        BinOp::Shl | BinOp::Shr => None,
    }
}
