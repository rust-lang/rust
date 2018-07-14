use prelude::*;

pub fn trans_constant<'a, 'tcx: 'a>(fx: &mut FunctionCx<'a, 'tcx>, const_: &Constant<'tcx>) -> CValue<'tcx> {
    use rustc::mir::interpret::{Scalar, ConstValue, GlobalId};

    let value = match const_.literal {
        Literal::Value { value } => value,
        Literal::Promoted { index } => fx
            .tcx
            .const_eval(ParamEnv::reveal_all().and(GlobalId {
                instance: fx.instance,
                promoted: Some(index),
            }))
            .unwrap(),
    };

    let layout = fx.layout_of(const_.ty);
    match const_.ty.sty {
        TypeVariants::TyBool => {
            let bits = value.to_scalar().unwrap().to_bits(layout.size).unwrap();
            CValue::const_val(fx, const_.ty, bits as u64 as i64)
        }
        TypeVariants::TyUint(_) => {
            let bits = value.to_scalar().unwrap().to_bits(layout.size).unwrap();
            CValue::const_val(fx, const_.ty, bits as u64 as i64)
        }
        TypeVariants::TyInt(_) => {
            let bits = value.to_scalar().unwrap().to_bits(layout.size).unwrap();
            CValue::const_val(fx, const_.ty, bits as i128 as i64)
        }
        TypeVariants::TyFnDef(def_id, substs) => {
            let func_ref = fx.get_function_ref(Instance::new(def_id, substs));
            CValue::Func(func_ref, fx.layout_of(const_.ty))
        }
        _ => unimplemented!("value {:?} ty {:?}", value, const_.ty),
    }
}