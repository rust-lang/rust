use crate::prelude::*;

pub fn bin_op_to_intcc(bin_op: BinOp, signed: bool) -> Option<IntCC> {
    use BinOp::*;
    use IntCC::*;
    Some(match bin_op {
        Eq => Equal,
        Lt => if signed { SignedLessThan } else { UnsignedLessThan},
        Le => if signed { SignedLessThanOrEqual } else { UnsignedLessThanOrEqual},
        Ne => NotEqual,
        Ge => if signed { SignedGreaterThanOrEqual } else { UnsignedGreaterThanOrEqual },
        Gt => if signed { SignedGreaterThan } else { UnsignedGreaterThan },
        _ => return None,
    })
}

fn codegen_compare_bin_op<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    bin_op: BinOp,
    signed: bool,
    lhs: Value,
    rhs: Value,
) -> CValue<'tcx> {
    let intcc = crate::num::bin_op_to_intcc(bin_op, signed).unwrap();
    let val = codegen_icmp(fx, intcc, lhs, rhs);
    let val = fx.bcx.ins().bint(types::I8, val);
    CValue::by_val(val, fx.layout_of(fx.tcx.types.bool))
}

pub fn codegen_binop<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    bin_op: BinOp,
    in_lhs: CValue<'tcx>,
    in_rhs: CValue<'tcx>,
) -> CValue<'tcx> {
    match bin_op {
        BinOp::Eq | BinOp::Lt | BinOp::Le | BinOp::Ne | BinOp::Ge | BinOp::Gt => {
            match in_lhs.layout().ty.sty {
                ty::Bool | ty::Uint(_) | ty::Int(_) | ty::Char => {
                    let signed = type_sign(in_lhs.layout().ty);
                    let lhs = in_lhs.load_scalar(fx);
                    let rhs = in_rhs.load_scalar(fx);

                    let (lhs, rhs) = if
                        (bin_op == BinOp::Eq || bin_op == BinOp::Ne)
                        && (in_lhs.layout().ty.sty == fx.tcx.types.i8.sty || in_lhs.layout().ty.sty == fx.tcx.types.i16.sty)
                    {
                        // FIXME(CraneStation/cranelift#896) icmp_imm.i8/i16 with eq/ne for signed ints is implemented wrong.
                        (
                            fx.bcx.ins().sextend(types::I32, lhs),
                            fx.bcx.ins().sextend(types::I32, rhs),
                        )
                    } else {
                        (lhs, rhs)
                    };

                    return codegen_compare_bin_op(fx, bin_op, signed, lhs, rhs);
                }
                _ => {}
            }
        }
        _ => {}
    }

    match in_lhs.layout().ty.sty {
        ty::Bool => crate::num::trans_bool_binop(fx, bin_op, in_lhs, in_rhs),
        ty::Uint(_) | ty::Int(_)=> {
            crate::num::trans_int_binop(fx, bin_op, in_lhs, in_rhs)
        }
        ty::Float(_) => crate::num::trans_float_binop(fx, bin_op, in_lhs, in_rhs),
        ty::RawPtr(..) | ty::FnPtr(..) => {
            crate::num::trans_ptr_binop(fx, bin_op, in_lhs, in_rhs)
        }
        _ => unimplemented!("{:?}({:?}, {:?})", bin_op, in_lhs.layout().ty, in_rhs.layout().ty),
    }
}

pub fn trans_bool_binop<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    bin_op: BinOp,
    in_lhs: CValue<'tcx>,
    in_rhs: CValue<'tcx>,
) -> CValue<'tcx> {
    let lhs = in_lhs.load_scalar(fx);
    let rhs = in_rhs.load_scalar(fx);

    let b = fx.bcx.ins();
    let res = match bin_op {
        BinOp::BitXor => b.bxor(lhs, rhs),
        BinOp::BitAnd => b.band(lhs, rhs),
        BinOp::BitOr => b.bor(lhs, rhs),
        // Compare binops handles by `codegen_binop`.
        _ => unreachable!("{:?}({:?}, {:?})", bin_op, in_lhs, in_rhs),
    };

    CValue::by_val(res, fx.layout_of(fx.tcx.types.bool))
}

pub fn trans_int_binop<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    bin_op: BinOp,
    in_lhs: CValue<'tcx>,
    in_rhs: CValue<'tcx>,
) -> CValue<'tcx> {
    if bin_op != BinOp::Shl && bin_op != BinOp::Shr {
        assert_eq!(
            in_lhs.layout().ty,
            in_rhs.layout().ty,
            "int binop requires lhs and rhs of same type"
        );
    }

    if let Some(res) = crate::codegen_i128::maybe_codegen(fx, bin_op, false, in_lhs, in_rhs) {
        return res;
    }

    let signed = type_sign(in_lhs.layout().ty);

    let lhs = in_lhs.load_scalar(fx);
    let rhs = in_rhs.load_scalar(fx);

    let b = fx.bcx.ins();
    let val = match bin_op {
        BinOp::Add => b.iadd(lhs, rhs),
        BinOp::Sub => b.isub(lhs, rhs),
        BinOp::Mul => b.imul(lhs, rhs),
        BinOp::Div => if signed { b.sdiv(lhs, rhs) } else { b.udiv(lhs, rhs) },
        BinOp::Rem => if signed { b.srem(lhs, rhs) } else { b.urem(lhs, rhs) },
        BinOp::BitXor => b.bxor(lhs, rhs),
        BinOp::BitAnd => b.band(lhs, rhs),
        BinOp::BitOr => b.bor(lhs, rhs),
        BinOp::Shl => {
            let lhs_ty = fx.bcx.func.dfg.value_type(lhs);
            let rhs = clif_intcast(fx, rhs, lhs_ty, false);
            fx.bcx.ins().ishl(lhs, rhs)
        }
        BinOp::Shr => {
            let lhs_ty = fx.bcx.func.dfg.value_type(lhs);
            let rhs = clif_intcast(fx, rhs, lhs_ty, false);
            if signed {
                fx.bcx.ins().sshr(lhs, rhs)
            } else {
                fx.bcx.ins().ushr(lhs, rhs)
            }
        }
        // Compare binops handles by `codegen_binop`.
        _ => unreachable!("{:?}({:?}, {:?})", bin_op, in_lhs.layout().ty, in_rhs.layout().ty),
    };

    CValue::by_val(val, in_lhs.layout())
}

pub fn trans_checked_int_binop<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    bin_op: BinOp,
    in_lhs: CValue<'tcx>,
    in_rhs: CValue<'tcx>,
) -> CValue<'tcx> {
    if bin_op != BinOp::Shl && bin_op != BinOp::Shr {
        assert_eq!(
            in_lhs.layout().ty,
            in_rhs.layout().ty,
            "checked int binop requires lhs and rhs of same type"
        );
    }

    let lhs = in_lhs.load_scalar(fx);
    let rhs = in_rhs.load_scalar(fx);

    if let Some(res) = crate::codegen_i128::maybe_codegen(fx, bin_op, true, in_lhs, in_rhs) {
        return res;
    }

    let signed = type_sign(in_lhs.layout().ty);

    let (res, has_overflow) = match bin_op {
        BinOp::Add => {
            /*let (val, c_out) = fx.bcx.ins().iadd_cout(lhs, rhs);
            (val, c_out)*/
            // FIXME(CraneStation/cranelift#849) legalize iadd_cout for i8 and i16
            let val = fx.bcx.ins().iadd(lhs, rhs);
            let has_overflow = if !signed {
                fx.bcx.ins().icmp(IntCC::UnsignedLessThan, val, lhs)
            } else {
                let rhs_is_negative = fx.bcx.ins().icmp_imm(IntCC::SignedLessThan, rhs, 0);
                let slt = fx.bcx.ins().icmp(IntCC::SignedLessThan, val, lhs);
                fx.bcx.ins().bxor(rhs_is_negative, slt)
            };
            (val, has_overflow)
        }
        BinOp::Sub => {
            /*let (val, b_out) = fx.bcx.ins().isub_bout(lhs, rhs);
            (val, b_out)*/
            // FIXME(CraneStation/cranelift#849) legalize isub_bout for i8 and i16
            let val = fx.bcx.ins().isub(lhs, rhs);
            let has_overflow = if !signed {
                fx.bcx.ins().icmp(IntCC::UnsignedGreaterThan, val, lhs)
            } else {
                let rhs_is_negative = fx.bcx.ins().icmp_imm(IntCC::SignedLessThan, rhs, 0);
                let sgt = fx.bcx.ins().icmp(IntCC::SignedGreaterThan, val, lhs);
                fx.bcx.ins().bxor(rhs_is_negative, sgt)
            };
            (val, has_overflow)
        }
        BinOp::Mul => {
            let val = fx.bcx.ins().imul(lhs, rhs);
            /*let val_hi = if !signed {
                fx.bcx.ins().umulhi(lhs, rhs)
            } else {
                fx.bcx.ins().smulhi(lhs, rhs)
            };
            let has_overflow = fx.bcx.ins().icmp_imm(IntCC::NotEqual, val_hi, 0);*/
            // TODO: check for overflow
            let has_overflow = fx.bcx.ins().bconst(types::B1, false);
            (val, has_overflow)
        }
        BinOp::Shl => {
            let val = fx.bcx.ins().ishl(lhs, rhs);
            // TODO: check for overflow
            let has_overflow = fx.bcx.ins().bconst(types::B1, false);
            (val, has_overflow)
        }
        BinOp::Shr => {
            let val = if !signed {
                fx.bcx.ins().ushr(lhs, rhs)
            } else {
                fx.bcx.ins().sshr(lhs, rhs)
            };
            // TODO: check for overflow
            let has_overflow = fx.bcx.ins().bconst(types::B1, false);
            (val, has_overflow)
        }
        _ => bug!(
            "binop {:?} on checked int/uint lhs: {:?} rhs: {:?}",
            bin_op,
            in_lhs,
            in_rhs
        ),
    };

    let has_overflow = fx.bcx.ins().bint(types::I8, has_overflow);
    let out_place = CPlace::new_stack_slot(fx, fx.tcx.mk_tup([in_lhs.layout().ty, fx.tcx.types.bool].iter()));
    let out_layout = out_place.layout();
    out_place.write_cvalue(fx, CValue::by_val_pair(res, has_overflow, out_layout));

    out_place.to_cvalue(fx)
}

pub fn trans_float_binop<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    bin_op: BinOp,
    in_lhs: CValue<'tcx>,
    in_rhs: CValue<'tcx>,
) -> CValue<'tcx> {
    assert_eq!(in_lhs.layout().ty, in_rhs.layout().ty);

    let lhs = in_lhs.load_scalar(fx);
    let rhs = in_rhs.load_scalar(fx);

    let b = fx.bcx.ins();
    let res = match bin_op {
        BinOp::Add => b.fadd(lhs, rhs),
        BinOp::Sub => b.fsub(lhs, rhs),
        BinOp::Mul => b.fmul(lhs, rhs),
        BinOp::Div => b.fdiv(lhs, rhs),
        BinOp::Rem => {
            let name = match in_lhs.layout().ty.sty {
                ty::Float(FloatTy::F32) => "fmodf",
                ty::Float(FloatTy::F64) => "fmod",
                _ => bug!(),
            };
            return fx.easy_call(name, &[in_lhs, in_rhs], in_lhs.layout().ty);
        }
        BinOp::Eq | BinOp::Lt | BinOp::Le | BinOp::Ne | BinOp::Ge | BinOp::Gt => {
            let fltcc = match bin_op {
                BinOp::Eq => FloatCC::Equal,
                BinOp::Lt => FloatCC::LessThan,
                BinOp::Le => FloatCC::LessThanOrEqual,
                BinOp::Ne => FloatCC::NotEqual,
                BinOp::Ge => FloatCC::GreaterThanOrEqual,
                BinOp::Gt => FloatCC::GreaterThan,
                _ => unreachable!(),
            };
            let val = fx.bcx.ins().fcmp(fltcc, lhs, rhs);
            let val = fx.bcx.ins().bint(types::I8, val);
            return CValue::by_val(val, fx.layout_of(fx.tcx.types.bool));
        }
        _ => unreachable!("{:?}({:?}, {:?})", bin_op, in_lhs, in_rhs),
    };

    CValue::by_val(res, in_lhs.layout())
}

pub fn trans_ptr_binop<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    bin_op: BinOp,
    in_lhs: CValue<'tcx>,
    in_rhs: CValue<'tcx>,
) -> CValue<'tcx> {
    let not_fat = match in_lhs.layout().ty.sty {
        ty::RawPtr(TypeAndMut { ty, mutbl: _ }) => {
            ty.is_sized(fx.tcx.at(DUMMY_SP), ParamEnv::reveal_all())
        }
        ty::FnPtr(..) => true,
        _ => bug!("trans_ptr_binop on non ptr"),
    };
    if not_fat {
        match bin_op {
            BinOp::Eq | BinOp::Lt | BinOp::Le | BinOp::Ne | BinOp::Ge | BinOp::Gt => {
                let lhs = in_lhs.load_scalar(fx);
                let rhs = in_rhs.load_scalar(fx);

                return codegen_compare_bin_op(fx, bin_op, false, lhs, rhs);;
            }
            BinOp::Offset => {
                let (base, offset) = (in_lhs, in_rhs.load_scalar(fx));
                let pointee_ty = base.layout().ty.builtin_deref(true).unwrap().ty;
                let pointee_size = fx.layout_of(pointee_ty).size.bytes();
                let ptr_diff = fx.bcx.ins().imul_imm(offset, pointee_size as i64);
                let base_val = base.load_scalar(fx);
                let res = fx.bcx.ins().iadd(base_val, ptr_diff);
                return CValue::by_val(res, base.layout());
            }
            _ => unreachable!("{:?}({:?}, {:?})", bin_op, in_lhs, in_rhs),
        };
    } else {
        let (lhs_ptr, lhs_extra) = in_lhs.load_scalar_pair(fx);
        let (rhs_ptr, rhs_extra) = in_rhs.load_scalar_pair(fx);

        let res = match bin_op {
            BinOp::Eq => {
                let ptr_eq = fx.bcx.ins().icmp(IntCC::Equal, lhs_ptr, rhs_ptr);
                let extra_eq = fx.bcx.ins().icmp(IntCC::Equal, lhs_extra, rhs_extra);
                fx.bcx.ins().band(ptr_eq, extra_eq)
            }
            BinOp::Ne => {
                let ptr_ne = fx.bcx.ins().icmp(IntCC::NotEqual, lhs_ptr, rhs_ptr);
                let extra_ne = fx.bcx.ins().icmp(IntCC::NotEqual, lhs_extra, rhs_extra);
                fx.bcx.ins().bor(ptr_ne, extra_ne)
            }
            BinOp::Lt | BinOp::Le | BinOp::Ge | BinOp::Gt => {
                let ptr_eq = fx.bcx.ins().icmp(IntCC::Equal, lhs_ptr, rhs_ptr);

                let ptr_cmp = fx.bcx.ins().icmp(bin_op_to_intcc(bin_op, false).unwrap(), lhs_ptr, rhs_ptr);
                let extra_cmp = fx.bcx.ins().icmp(bin_op_to_intcc(bin_op, false).unwrap(), lhs_extra, rhs_extra);

                fx.bcx.ins().select(ptr_eq, extra_cmp, ptr_cmp)
            }
            _ => panic!("bin_op {:?} on ptr", bin_op),
        };

        CValue::by_val(fx.bcx.ins().bint(types::I8, res), fx.layout_of(fx.tcx.types.bool))
    }
}
