//! Replaces 128-bit operators with lang item calls

use crate::prelude::*;

pub fn maybe_codegen<'a, 'tcx>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    bin_op: BinOp,
    checked: bool,
    is_signed: bool,
    lhs: CValue<'tcx>,
    rhs: CValue<'tcx>,
    out_ty: Ty<'tcx>,
) -> Option<CValue<'tcx>> {
    if lhs.layout().ty != fx.tcx.types.u128 && lhs.layout().ty != fx.tcx.types.i128 {
        return None;
    }

    let lhs_val = lhs.load_scalar(fx);
    let rhs_val = rhs.load_scalar(fx);

    match bin_op {
        BinOp::Add | BinOp::Sub | BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => return None,
        BinOp::Offset => unreachable!("offset should only be used on pointers, not 128bit ints"),
        BinOp::Mul => {
            let res = if checked {
                if is_signed {
                    let oflow_place = CPlace::new_stack_slot(fx, fx.tcx.types.i32);
                    let oflow_addr = oflow_place.to_addr(fx);
                    let oflow_addr = CValue::by_val(oflow_addr, fx.layout_of(fx.tcx.mk_mut_ptr(fx.tcx.types.i32)));
                    let val = fx.easy_call("__muloti4", &[lhs, rhs, oflow_addr], fx.tcx.types.i128);
                    let val = val.load_scalar(fx);
                    let oflow = oflow_place.to_cvalue(fx).load_scalar(fx);
                    let oflow = fx.bcx.ins().icmp_imm(IntCC::NotEqual, oflow, 0);
                    let oflow = fx.bcx.ins().bint(types::I8, oflow);
                    CValue::by_val_pair(val, oflow, fx.layout_of(out_ty))
                } else {
                    // FIXME implement it
                let out_layout = fx.layout_of(out_ty);
                    return Some(crate::trap::trap_unreachable_ret_value(fx, out_layout, format!("unimplemented 128bit checked binop unsigned mul")));
                }
            } else {
                let val_ty = if is_signed { fx.tcx.types.i128 } else { fx.tcx.types.u128 };
                fx.easy_call("__multi3", &[lhs, rhs], val_ty)
            };
            return Some(res);
        }
        BinOp::Div => {
            let res = if checked {
                // FIXME implement it
                let out_layout = fx.layout_of(out_ty);
                return Some(crate::trap::trap_unreachable_ret_value(fx, out_layout, format!("unimplemented 128bit checked binop div")));
            } else {
                if is_signed {
                    fx.easy_call("__divti3", &[lhs, rhs], fx.tcx.types.i128)
                } else {
                    fx.easy_call("__udivti3", &[lhs, rhs], fx.tcx.types.u128)
                }
            };
            return Some(res);
        }
        BinOp::Rem => {
            let res = if checked {
                // FIXME implement it
                let out_layout = fx.layout_of(out_ty);
                return Some(crate::trap::trap_unreachable_ret_value(fx, out_layout, format!("unimplemented 128bit checked binop rem")));
            } else {
                if is_signed {
                    fx.easy_call("__modti3", &[lhs, rhs], fx.tcx.types.i128)
                } else {
                    fx.easy_call("__umodti3", &[lhs, rhs], fx.tcx.types.u128)
                }
            };
            return Some(res);
        }
        BinOp::Lt | BinOp::Le | BinOp::Eq | BinOp::Ge | BinOp::Gt | BinOp::Ne => {
            assert!(!checked);
            let (lhs_lsb, lhs_msb) = fx.bcx.ins().isplit(lhs_val);
            let (rhs_lsb, rhs_msb) = fx.bcx.ins().isplit(rhs_val);
            let res = match (bin_op, is_signed) {
                (BinOp::Eq, _) => {
                    let lsb_eq = fx.bcx.ins().icmp(IntCC::Equal, lhs_lsb, rhs_lsb);
                    let msb_eq = fx.bcx.ins().icmp(IntCC::Equal, lhs_msb, rhs_msb);
                    fx.bcx.ins().band(lsb_eq, msb_eq)
                }
                (BinOp::Ne, _) => {
                    let lsb_ne = fx.bcx.ins().icmp(IntCC::NotEqual, lhs_lsb, rhs_lsb);
                    let msb_ne = fx.bcx.ins().icmp(IntCC::NotEqual, lhs_msb, rhs_msb);
                    fx.bcx.ins().bor(lsb_ne, msb_ne)
                }
                _ => {
                    // FIXME implement it
                    let out_layout = fx.layout_of(out_ty);
                    return Some(crate::trap::trap_unreachable_ret_value(fx, out_layout, format!("unimplemented 128bit binop {:?}", bin_op)));
                },
            };

            let res = fx.bcx.ins().bint(types::I8, res);
            let res = CValue::by_val(res, fx.layout_of(fx.tcx.types.bool));
            return Some(res);
        }
        BinOp::Shl | BinOp::Shr => {
            // FIXME implement it
            let out_layout = fx.layout_of(out_ty);
            return Some(crate::trap::trap_unreachable_ret_value(fx, out_layout, format!("unimplemented 128bit binop {:?}", bin_op)));
        }
    }
}
