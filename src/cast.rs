use crate::prelude::*;

pub fn clif_intcast<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    val: Value,
    to: Type,
    signed: bool,
) -> Value {
    let from = fx.bcx.func.dfg.value_type(val);
    match (from, to) {
        // equal
        (_, _) if from == to => val,

        // extend
        (_, types::I128) => {
            let wider = if from == types::I64 {
                val
            } else if signed {
                fx.bcx.ins().sextend(types::I64, val)
            } else {
                fx.bcx.ins().uextend(types::I64, val)
            };
            let zero = fx.bcx.ins().iconst(types::I64, 0);
            fx.bcx.ins().iconcat(wider, zero)
        }
        (_, _) if to.wider_or_equal(from) => {
            if signed {
                fx.bcx.ins().sextend(to, val)
            } else {
                fx.bcx.ins().uextend(to, val)
            }
        }

        // reduce
        (types::I128, _) => {
            let (lsb, _msb) = fx.bcx.ins().isplit(val);
            if to == types::I64 {
                lsb
            } else {
                fx.bcx.ins().ireduce(to, lsb)
            }
        }
        (_, _) => {
            fx.bcx.ins().ireduce(to, val)
        }
    }
}

pub fn clif_int_or_float_cast(
    fx: &mut FunctionCx<'_, '_, impl Backend>,
    from: Value,
    to_ty: Type,
    signed: bool,
) -> Value {
    let from_ty = fx.bcx.func.dfg.value_type(from);

    if from_ty.is_int() && to_ty.is_int() {
        // int-like -> int-like
        clif_intcast(
            fx,
            from,
            to_ty,
            signed,
        )
    } else if from_ty.is_int() && to_ty.is_float() {
        // int-like -> float
        if signed {
            fx.bcx.ins().fcvt_from_sint(to_ty, from)
        } else {
            fx.bcx.ins().fcvt_from_uint(to_ty, from)
        }
    } else if from_ty.is_float() && to_ty.is_int() {
        // float -> int-like
        if signed {
            fx.bcx.ins().fcvt_to_sint_sat(to_ty, from)
        } else {
            fx.bcx.ins().fcvt_to_uint_sat(to_ty, from)
        }
    } else if from_ty.is_float() && to_ty.is_float() {
        // float -> float
        match (from_ty, to_ty) {
            (types::F32, types::F64) => {
                fx.bcx.ins().fpromote(types::F64, from)
            }
            (types::F64, types::F32) => {
                fx.bcx.ins().fdemote(types::F32, from)
            }
            _ => from,
        }
    } else {
        unreachable!("cast value from {:?} to {:?}", from_ty, to_ty);
    }
}
