//! Replaces 128-bit operators with lang item calls where necessary

use crate::prelude::*;

pub(crate) fn maybe_codegen<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    bin_op: BinOp,
    checked: bool,
    lhs: CValue<'tcx>,
    rhs: CValue<'tcx>,
) -> Option<CValue<'tcx>> {
    if lhs.layout().ty != fx.tcx.types.u128 && lhs.layout().ty != fx.tcx.types.i128 {
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
        BinOp::Add => {
            let out_ty = fx.tcx.mk_tup([lhs.layout().ty, fx.tcx.types.bool].iter());
            return Some(if is_signed {
                fx.easy_call("__rust_i128_addo", &[lhs, rhs], out_ty)
            } else {
                fx.easy_call("__rust_u128_addo", &[lhs, rhs], out_ty)
            });
        }
        BinOp::Sub => {
            let out_ty = fx.tcx.mk_tup([lhs.layout().ty, fx.tcx.types.bool].iter());
            return Some(if is_signed {
                fx.easy_call("__rust_i128_subo", &[lhs, rhs], out_ty)
            } else {
                fx.easy_call("__rust_u128_subo", &[lhs, rhs], out_ty)
            });
        }
        BinOp::Offset => unreachable!("offset should only be used on pointers, not 128bit ints"),
        BinOp::Mul => {
            let res = if checked {
                let out_ty = fx.tcx.mk_tup([lhs.layout().ty, fx.tcx.types.bool].iter());
                if is_signed {
                    fx.easy_call("__rust_i128_mulo", &[lhs, rhs], out_ty)
                } else {
                    fx.easy_call("__rust_u128_mulo", &[lhs, rhs], out_ty)
                }
            } else {
                let val_ty = if is_signed {
                    fx.tcx.types.i128
                } else {
                    fx.tcx.types.u128
                };
                fx.easy_call("__multi3", &[lhs, rhs], val_ty)
            };
            Some(res)
        }
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

            // Optimize `val >> 64`, because compiler_builtins uses it to deconstruct an 128bit
            // integer into its lsb and msb.
            // https://github.com/rust-lang-nursery/compiler-builtins/blob/79a6a1603d5672cbb9187ff41ff4d9b5048ac1cb/src/int/mod.rs#L217
            if resolve_value_imm(fx.bcx.func, rhs_val) == Some(64) {
                let (lhs_lsb, lhs_msb) = fx.bcx.ins().isplit(lhs_val);
                let all_zeros = fx.bcx.ins().iconst(types::I64, 0);
                let val = match (bin_op, is_signed) {
                    (BinOp::Shr, false) => {
                        let val = fx.bcx.ins().iconcat(lhs_msb, all_zeros);
                        Some(CValue::by_val(val, fx.layout_of(fx.tcx.types.u128)))
                    }
                    (BinOp::Shr, true) => {
                        let sign = fx.bcx.ins().icmp_imm(IntCC::SignedLessThan, lhs_msb, 0);
                        let all_ones = fx.bcx.ins().iconst(types::I64, u64::MAX as i64);
                        let all_sign_bits = fx.bcx.ins().select(sign, all_zeros, all_ones);

                        let val = fx.bcx.ins().iconcat(lhs_msb, all_sign_bits);
                        Some(CValue::by_val(val, fx.layout_of(fx.tcx.types.i128)))
                    }
                    (BinOp::Shl, _) => {
                        let val_ty = if is_signed {
                            fx.tcx.types.i128
                        } else {
                            fx.tcx.types.u128
                        };
                        let val = fx.bcx.ins().iconcat(all_zeros, lhs_lsb);
                        Some(CValue::by_val(val, fx.layout_of(val_ty)))
                    }
                    _ => None,
                };
                if let Some(val) = val {
                    if let Some(is_overflow) = is_overflow {
                        let out_ty = fx.tcx.mk_tup([lhs.layout().ty, fx.tcx.types.bool].iter());
                        let val = val.load_scalar(fx);
                        return Some(CValue::by_val_pair(val, is_overflow, fx.layout_of(out_ty)));
                    } else {
                        return Some(val);
                    }
                }
            }

            let truncated_rhs = clif_intcast(fx, rhs_val, types::I32, false);
            let truncated_rhs = CValue::by_val(truncated_rhs, fx.layout_of(fx.tcx.types.u32));
            let val = match (bin_op, is_signed) {
                (BinOp::Shl, false) => {
                    fx.easy_call("__ashlti3", &[lhs, truncated_rhs], fx.tcx.types.u128)
                }
                (BinOp::Shl, true) => {
                    fx.easy_call("__ashlti3", &[lhs, truncated_rhs], fx.tcx.types.i128)
                }
                (BinOp::Shr, false) => {
                    fx.easy_call("__lshrti3", &[lhs, truncated_rhs], fx.tcx.types.u128)
                }
                (BinOp::Shr, true) => {
                    fx.easy_call("__ashrti3", &[lhs, truncated_rhs], fx.tcx.types.i128)
                }
                (_, _) => unreachable!(),
            };
            if let Some(is_overflow) = is_overflow {
                let out_ty = fx.tcx.mk_tup([lhs.layout().ty, fx.tcx.types.bool].iter());
                let val = val.load_scalar(fx);
                Some(CValue::by_val_pair(val, is_overflow, fx.layout_of(out_ty)))
            } else {
                Some(val)
            }
        }
    }
}
