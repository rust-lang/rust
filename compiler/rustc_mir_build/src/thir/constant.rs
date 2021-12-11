use rustc_apfloat::Float;
use rustc_ast as ast;
use rustc_middle::mir::interpret::{
    Allocation, ConstValue, LitToConstError, LitToConstInput, Scalar,
};
use rustc_middle::ty::{self, ParamEnv, TyCtxt};
use rustc_span::symbol::Symbol;
use rustc_target::abi::Size;

crate fn lit_to_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    lit_input: LitToConstInput<'tcx>,
) -> Result<&'tcx ty::Const<'tcx>, LitToConstError> {
    let LitToConstInput { lit, ty, neg } = lit_input;

    let trunc = |n| {
        let param_ty = ParamEnv::reveal_all().and(ty);
        let width = tcx.layout_of(param_ty).map_err(|_| LitToConstError::Reported)?.size;
        trace!("trunc {} with size {} and shift {}", n, width.bits(), 128 - width.bits());
        let result = width.truncate(n);
        trace!("trunc result: {}", result);
        Ok(ConstValue::Scalar(Scalar::from_uint(result, width)))
    };

    let lit = match (lit, &ty.kind()) {
        (ast::LitKind::Str(s, _), ty::Ref(_, inner_ty, _)) if inner_ty.is_str() => {
            let s = s.as_str();
            let allocation = Allocation::from_bytes_byte_aligned_immutable(s.as_bytes());
            let allocation = tcx.intern_const_alloc(allocation);
            ConstValue::Slice { data: allocation, start: 0, end: s.len() }
        }
        (ast::LitKind::ByteStr(data), ty::Ref(_, inner_ty, _))
            if matches!(inner_ty.kind(), ty::Slice(_)) =>
        {
            let allocation = Allocation::from_bytes_byte_aligned_immutable(data as &[u8]);
            let allocation = tcx.intern_const_alloc(allocation);
            ConstValue::Slice { data: allocation, start: 0, end: data.len() }
        }
        (ast::LitKind::ByteStr(data), ty::Ref(_, inner_ty, _)) if inner_ty.is_array() => {
            let id = tcx.allocate_bytes(data);
            ConstValue::Scalar(Scalar::from_pointer(id.into(), &tcx))
        }
        (ast::LitKind::Byte(n), ty::Uint(ty::UintTy::U8)) => {
            ConstValue::Scalar(Scalar::from_uint(*n, Size::from_bytes(1)))
        }
        (ast::LitKind::Int(n, _), ty::Uint(_)) | (ast::LitKind::Int(n, _), ty::Int(_)) => {
            trunc(if neg { (*n as i128).overflowing_neg().0 as u128 } else { *n })?
        }
        (ast::LitKind::Float(n, _), ty::Float(fty)) => {
            parse_float(*n, *fty, neg).ok_or(LitToConstError::Reported)?
        }
        (ast::LitKind::Bool(b), ty::Bool) => ConstValue::Scalar(Scalar::from_bool(*b)),
        (ast::LitKind::Char(c), ty::Char) => ConstValue::Scalar(Scalar::from_char(*c)),
        (ast::LitKind::Err(_), _) => return Err(LitToConstError::Reported),
        _ => return Err(LitToConstError::TypeError),
    };
    Ok(ty::Const::from_value(tcx, lit, ty))
}

fn parse_float<'tcx>(num: Symbol, fty: ty::FloatTy, neg: bool) -> Option<ConstValue<'tcx>> {
    let num = num.as_str();
    use rustc_apfloat::ieee::{Double, Single};
    let scalar = match fty {
        ty::FloatTy::F32 => {
            let rust_f = match num.parse::<f32>() {
                Ok(f) => f,
                Err(_) => return None,
            };
            let mut f = num.parse::<Single>().unwrap_or_else(|e| {
                panic!("apfloat::ieee::Single failed to parse `{}`: {:?}", num, e)
            });
            assert!(
                u128::from(rust_f.to_bits()) == f.to_bits(),
                "apfloat::ieee::Single gave different result for `{}`: \
                 {}({:#x}) vs Rust's {}({:#x})",
                rust_f,
                f,
                f.to_bits(),
                Single::from_bits(rust_f.to_bits().into()),
                rust_f.to_bits()
            );
            if neg {
                f = -f;
            }
            Scalar::from_f32(f)
        }
        ty::FloatTy::F64 => {
            let rust_f = match num.parse::<f64>() {
                Ok(f) => f,
                Err(_) => return None,
            };
            let mut f = num.parse::<Double>().unwrap_or_else(|e| {
                panic!("apfloat::ieee::Double failed to parse `{}`: {:?}", num, e)
            });
            assert!(
                u128::from(rust_f.to_bits()) == f.to_bits(),
                "apfloat::ieee::Double gave different result for `{}`: \
                 {}({:#x}) vs Rust's {}({:#x})",
                rust_f,
                f,
                f.to_bits(),
                Double::from_bits(rust_f.to_bits().into()),
                rust_f.to_bits()
            );
            if neg {
                f = -f;
            }
            Scalar::from_f64(f)
        }
    };

    Some(ConstValue::Scalar(scalar))
}
