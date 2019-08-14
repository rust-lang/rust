use crate::prelude::*;

macro_rules! binop_match {
    (@single $fx:expr, $bug_fmt:expr, $var:expr, $signed:expr, $lhs:expr, $rhs:expr, $ret_ty:expr, bug) => {
        bug!("binop {} on {} lhs: {:?} rhs: {:?}", stringify!($var), $bug_fmt, $lhs, $rhs)
    };
    (@single $fx:expr, $bug_fmt:expr, $var:expr, $signed:expr, $lhs:expr, $rhs:expr, $ret_ty:expr, icmp($cc:ident)) => {{
        assert_eq!($fx.tcx.types.bool, $ret_ty);
        let ret_layout = $fx.layout_of($ret_ty);

        let b = $fx.bcx.ins().icmp(IntCC::$cc, $lhs, $rhs);
        CValue::by_val($fx.bcx.ins().bint(types::I8, b), ret_layout)
    }};
    (@single $fx:expr, $bug_fmt:expr, $var:expr, $signed:expr, $lhs:expr, $rhs:expr, $ret_ty:expr, fcmp($cc:ident)) => {{
        assert_eq!($fx.tcx.types.bool, $ret_ty);
        let ret_layout = $fx.layout_of($ret_ty);
        let b = $fx.bcx.ins().fcmp(FloatCC::$cc, $lhs, $rhs);
        CValue::by_val($fx.bcx.ins().bint(types::I8, b), ret_layout)
    }};
    (@single $fx:expr, $bug_fmt:expr, $var:expr, $signed:expr, $lhs:expr, $rhs:expr, $ret_ty:expr, custom(|| $body:expr)) => {{
        $body
    }};
    (@single $fx:expr, $bug_fmt:expr, $var:expr, $signed:expr, $lhs:expr, $rhs:expr, $ret_ty:expr, $name:ident) => {{
        let ret_layout = $fx.layout_of($ret_ty);
        CValue::by_val($fx.bcx.ins().$name($lhs, $rhs), ret_layout)
    }};
    (
        $fx:expr, $bin_op:expr, $signed:expr, $lhs:expr, $rhs:expr, $ret_ty:expr, $bug_fmt:expr;
        $(
            $var:ident ($sign:pat) $name:tt $( ( $($next:tt)* ) )? ;
        )*
    ) => {{
        let lhs = $lhs.load_scalar($fx);
        let rhs = $rhs.load_scalar($fx);
        match ($bin_op, $signed) {
            $(
                (BinOp::$var, $sign) => binop_match!(@single $fx, $bug_fmt, $var, $signed, lhs, rhs, $ret_ty, $name $( ( $($next)* ) )?),
            )*
        }
    }}
}

pub fn trans_bool_binop<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    bin_op: BinOp,
    lhs: CValue<'tcx>,
    rhs: CValue<'tcx>,
) -> CValue<'tcx> {
    let res = binop_match! {
        fx, bin_op, false, lhs, rhs, fx.tcx.types.bool, "bool";
        Add (_) bug;
        Sub (_) bug;
        Mul (_) bug;
        Div (_) bug;
        Rem (_) bug;
        BitXor (_) bxor;
        BitAnd (_) band;
        BitOr (_) bor;
        Shl (_) bug;
        Shr (_) bug;

        Eq (_) icmp(Equal);
        Lt (_) icmp(UnsignedLessThan);
        Le (_) icmp(UnsignedLessThanOrEqual);
        Ne (_) icmp(NotEqual);
        Ge (_) icmp(UnsignedGreaterThanOrEqual);
        Gt (_) icmp(UnsignedGreaterThan);

        Offset (_) bug;
    };

    res
}

pub fn trans_int_binop<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    bin_op: BinOp,
    lhs: CValue<'tcx>,
    rhs: CValue<'tcx>,
    out_ty: Ty<'tcx>,
) -> CValue<'tcx> {
    if bin_op != BinOp::Shl && bin_op != BinOp::Shr {
        assert_eq!(
            lhs.layout().ty,
            rhs.layout().ty,
            "int binop requires lhs and rhs of same type"
        );
    }

    match out_ty.sty {
        ty::Bool | ty::Uint(_) | ty::Int(_) => {}
        _ => unreachable!("Out ty {:?} is not an integer or bool", out_ty),
    }

    if let Some(res) = crate::codegen_i128::maybe_codegen(fx, bin_op, false, lhs, rhs, out_ty) {
        return res;
    }

    let signed = type_sign(lhs.layout().ty);

    let (lhs, rhs) = if
        (bin_op == BinOp::Eq || bin_op == BinOp::Ne)
        && (lhs.layout().ty.sty == fx.tcx.types.i8.sty || lhs.layout().ty.sty == fx.tcx.types.i16.sty)
    {
        // FIXME(CraneStation/cranelift#896) icmp_imm.i8/i16 with eq/ne for signed ints is implemented wrong.
        let lhs = lhs.load_scalar(fx);
        let rhs = rhs.load_scalar(fx);
        (
            CValue::by_val(fx.bcx.ins().sextend(types::I32, lhs), fx.layout_of(fx.tcx.types.i32)),
            CValue::by_val(fx.bcx.ins().sextend(types::I32, rhs), fx.layout_of(fx.tcx.types.i32)),
        )
    } else {
        (lhs, rhs)
    };

    binop_match! {
        fx, bin_op, signed, lhs, rhs, out_ty, "int/uint";
        Add (_) iadd;
        Sub (_) isub;
        Mul (_) imul;
        Div (false) udiv;
        Div (true) sdiv;
        Rem (false) urem;
        Rem (true) srem;
        BitXor (_) bxor;
        BitAnd (_) band;
        BitOr (_) bor;
        Shl (_) ishl;
        Shr (false) ushr;
        Shr (true) sshr;

        Eq (_) icmp(Equal);
        Lt (false) icmp(UnsignedLessThan);
        Lt (true) icmp(SignedLessThan);
        Le (false) icmp(UnsignedLessThanOrEqual);
        Le (true) icmp(SignedLessThanOrEqual);
        Ne (_) icmp(NotEqual);
        Ge (false) icmp(UnsignedGreaterThanOrEqual);
        Ge (true) icmp(SignedGreaterThanOrEqual);
        Gt (false) icmp(UnsignedGreaterThan);
        Gt (true) icmp(SignedGreaterThan);

        Offset (_) bug;
    }
}

pub fn trans_checked_int_binop<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    bin_op: BinOp,
    in_lhs: CValue<'tcx>,
    in_rhs: CValue<'tcx>,
    out_ty: Ty<'tcx>,
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

    if let Some(res) = crate::codegen_i128::maybe_codegen(fx, bin_op, true, in_lhs, in_rhs, out_ty) {
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
    let out_place = CPlace::new_stack_slot(fx, out_ty);
    let out_layout = out_place.layout();
    out_place.write_cvalue(fx, CValue::by_val_pair(res, has_overflow, out_layout));

    out_place.to_cvalue(fx)
}

pub fn trans_float_binop<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    bin_op: BinOp,
    lhs: CValue<'tcx>,
    rhs: CValue<'tcx>,
    ty: Ty<'tcx>,
) -> CValue<'tcx> {
    let res = binop_match! {
        fx, bin_op, false, lhs, rhs, ty, "float";
        Add (_) fadd;
        Sub (_) fsub;
        Mul (_) fmul;
        Div (_) fdiv;
        Rem (_) custom(|| {
            assert_eq!(lhs.layout().ty, ty);
            assert_eq!(rhs.layout().ty, ty);
            match ty.sty {
                ty::Float(FloatTy::F32) => fx.easy_call("fmodf", &[lhs, rhs], ty),
                ty::Float(FloatTy::F64) => fx.easy_call("fmod", &[lhs, rhs], ty),
                _ => bug!(),
            }
        });
        BitXor (_) bxor;
        BitAnd (_) band;
        BitOr (_) bor;
        Shl (_) bug;
        Shr (_) bug;

        Eq (_) fcmp(Equal);
        Lt (_) fcmp(LessThan);
        Le (_) fcmp(LessThanOrEqual);
        Ne (_) fcmp(NotEqual);
        Ge (_) fcmp(GreaterThanOrEqual);
        Gt (_) fcmp(GreaterThan);

        Offset (_) bug;
    };

    res
}

pub fn trans_char_binop<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    bin_op: BinOp,
    lhs: CValue<'tcx>,
    rhs: CValue<'tcx>,
    ty: Ty<'tcx>,
) -> CValue<'tcx> {
    let res = binop_match! {
        fx, bin_op, false, lhs, rhs, ty, "char";
        Add (_) bug;
        Sub (_) bug;
        Mul (_) bug;
        Div (_) bug;
        Rem (_) bug;
        BitXor (_) bug;
        BitAnd (_) bug;
        BitOr (_) bug;
        Shl (_) bug;
        Shr (_) bug;

        Eq (_) icmp(Equal);
        Lt (_) icmp(UnsignedLessThan);
        Le (_) icmp(UnsignedLessThanOrEqual);
        Ne (_) icmp(NotEqual);
        Ge (_) icmp(UnsignedGreaterThanOrEqual);
        Gt (_) icmp(UnsignedGreaterThan);

        Offset (_) bug;
    };

    res
}

pub fn trans_ptr_binop<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    bin_op: BinOp,
    lhs: CValue<'tcx>,
    rhs: CValue<'tcx>,
    ret_ty: Ty<'tcx>,
) -> CValue<'tcx> {
    let not_fat = match lhs.layout().ty.sty {
        ty::RawPtr(TypeAndMut { ty, mutbl: _ }) => {
            ty.is_sized(fx.tcx.at(DUMMY_SP), ParamEnv::reveal_all())
        }
        ty::FnPtr(..) => true,
        _ => bug!("trans_ptr_binop on non ptr"),
    };
    if not_fat {
        if let BinOp::Offset = bin_op {
            let (base, offset) = (lhs, rhs.load_scalar(fx));
            let pointee_ty = base.layout().ty.builtin_deref(true).unwrap().ty;
            let pointee_size = fx.layout_of(pointee_ty).size.bytes();
            let ptr_diff = fx.bcx.ins().imul_imm(offset, pointee_size as i64);
            let base_val = base.load_scalar(fx);
            let res = fx.bcx.ins().iadd(base_val, ptr_diff);
            return CValue::by_val(res, base.layout());
        }

        binop_match! {
            fx, bin_op, false, lhs, rhs, ret_ty, "ptr";
            Add (_) bug;
            Sub (_) bug;
            Mul (_) bug;
            Div (_) bug;
            Rem (_) bug;
            BitXor (_) bug;
            BitAnd (_) bug;
            BitOr (_) bug;
            Shl (_) bug;
            Shr (_) bug;

            Eq (_) icmp(Equal);
            Lt (_) icmp(UnsignedLessThan);
            Le (_) icmp(UnsignedLessThanOrEqual);
            Ne (_) icmp(NotEqual);
            Ge (_) icmp(UnsignedGreaterThanOrEqual);
            Gt (_) icmp(UnsignedGreaterThan);

            Offset (_) bug; // Handled above
        }
    } else {
        let (lhs_ptr, lhs_extra) = lhs.load_scalar_pair(fx);
        let (rhs_ptr, rhs_extra) = rhs.load_scalar_pair(fx);

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

                let ptr_cmp = fx.bcx.ins().icmp(match bin_op {
                    BinOp::Lt => IntCC::UnsignedLessThan,
                    BinOp::Le => IntCC::UnsignedLessThanOrEqual,
                    BinOp::Ge => IntCC::UnsignedGreaterThanOrEqual,
                    BinOp::Gt => IntCC::UnsignedGreaterThan,
                    _ => unreachable!(),
                }, lhs_ptr, rhs_ptr);

                let extra_cmp = fx.bcx.ins().icmp(match bin_op {
                    BinOp::Lt => IntCC::UnsignedLessThan,
                    BinOp::Le => IntCC::UnsignedLessThanOrEqual,
                    BinOp::Ge => IntCC::UnsignedGreaterThanOrEqual,
                    BinOp::Gt => IntCC::UnsignedGreaterThan,
                    _ => unreachable!(),
                }, lhs_extra, rhs_extra);

                fx.bcx.ins().select(ptr_eq, extra_cmp, ptr_cmp)
            }
            _ => panic!("bin_op {:?} on ptr", bin_op),
        };

        assert_eq!(fx.tcx.types.bool, ret_ty);
        let ret_layout = fx.layout_of(ret_ty);
        CValue::by_val(fx.bcx.ins().bint(types::I8, res), ret_layout)
    }
}
