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

    let lhs_val = lhs.load_scalar(fx);
    let rhs_val = rhs.load_scalar(fx);

    let is_signed = type_sign(lhs.layout().ty);

    match bin_op {
        BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => {
            assert!(!checked);
            None
        }
        BinOp::Add | BinOp::Sub if !checked => None,
        BinOp::Mul if !checked => {
            let val_ty = if is_signed { fx.tcx.types.i128 } else { fx.tcx.types.u128 };
            Some(fx.easy_call("__multi3", &[lhs, rhs], val_ty))
        }
        BinOp::Add | BinOp::Sub | BinOp::Mul => {
            assert!(checked);
            let out_ty = fx.tcx.mk_tup([lhs.layout().ty, fx.tcx.types.bool].iter());
            let out_place = CPlace::new_stack_slot(fx, fx.layout_of(out_ty));
            let param_types = vec![
                AbiParam::special(pointer_ty(fx.tcx), ArgumentPurpose::StructReturn),
                AbiParam::new(types::I128),
                AbiParam::new(types::I128),
            ];
            let args = [out_place.to_ptr().get_addr(fx), lhs.load_scalar(fx), rhs.load_scalar(fx)];
            let name = match (bin_op, is_signed) {
                (BinOp::Add, false) => "__rust_u128_addo",
                (BinOp::Add, true) => "__rust_i128_addo",
                (BinOp::Sub, false) => "__rust_u128_subo",
                (BinOp::Sub, true) => "__rust_i128_subo",
                (BinOp::Mul, false) => "__rust_u128_mulo",
                (BinOp::Mul, true) => "__rust_i128_mulo",
                _ => unreachable!(),
            };
            fx.lib_call(name, param_types, vec![], &args);
            Some(out_place.to_cvalue(fx))
        }
        BinOp::Offset => unreachable!("offset should only be used on pointers, not 128bit ints"),
        BinOp::Div => {
            assert!(!checked);
            if is_signed {
                Some(fx.easy_call("__divti3", &[lhs, rhs], fx.tcx.types.i128))
            } else {
                Some(fx.easy_call("__udivti3", &[lhs, rhs], fx.tcx.types.u128))
            }
        }
        BinOp::Rem => {
            assert!(!checked);
            if is_signed {
                Some(fx.easy_call("__modti3", &[lhs, rhs], fx.tcx.types.i128))
            } else {
                Some(fx.easy_call("__umodti3", &[lhs, rhs], fx.tcx.types.u128))
            }
        }
        BinOp::Lt | BinOp::Le | BinOp::Eq | BinOp::Ge | BinOp::Gt | BinOp::Ne => {
            assert!(!checked);
            None
        }
        BinOp::Shl | BinOp::Shr => {
            let is_overflow = if checked {
                // rhs >= 128

                // FIXME support non 128bit rhs
                /*let (rhs_lsb, rhs_msb) = fx.bcx.ins().isplit(rhs_val);
                let rhs_msb_gt_0 = fx.bcx.ins().icmp_imm(IntCC::NotEqual, rhs_msb, 0);
                let rhs_lsb_ge_128 = fx.bcx.ins().icmp_imm(IntCC::SignedGreaterThan, rhs_lsb, 127);
                let is_overflow = fx.bcx.ins().bor(rhs_msb_gt_0, rhs_lsb_ge_128);*/
                let is_overflow = fx.bcx.ins().bconst(types::B1, false);

                Some(fx.bcx.ins().bint(types::I8, is_overflow))
            } else {
                None
            };

            let truncated_rhs = clif_intcast(fx, rhs_val, types::I32, false);
            let val = match bin_op {
                BinOp::Shl => fx.bcx.ins().ishl(lhs_val, truncated_rhs),
                BinOp::Shr => {
                    if is_signed {
                        fx.bcx.ins().sshr(lhs_val, truncated_rhs)
                    } else {
                        fx.bcx.ins().ushr(lhs_val, truncated_rhs)
                    }
                }
                _ => unreachable!(),
            };
            if let Some(is_overflow) = is_overflow {
                let out_ty = fx.tcx.mk_tup([lhs.layout().ty, fx.tcx.types.bool].iter());
                Some(CValue::by_val_pair(val, is_overflow, fx.layout_of(out_ty)))
            } else {
                Some(CValue::by_val(val, lhs.layout()))
            }
        }
    }
}
