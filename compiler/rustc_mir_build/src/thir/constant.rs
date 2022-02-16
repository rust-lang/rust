use rustc_apfloat::ieee::{Double, Single};
use rustc_apfloat::Float;
use rustc_ast as ast;
use rustc_middle::mir::interpret::{LitToConstError, LitToConstInput, Scalar};
use rustc_middle::ty::{self, ParamEnv, ScalarInt, TyCtxt};
use rustc_span::symbol::Symbol;

pub(crate) fn lit_to_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    lit_input: LitToConstInput<'tcx>,
) -> Result<ty::Const<'tcx>, LitToConstError> {
    let LitToConstInput { lit, ty, neg } = lit_input;

    let trunc = |n| {
        let param_ty = ParamEnv::reveal_all().and(ty);
        let width = tcx.layout_of(param_ty).map_err(|_| LitToConstError::Reported)?.size;
        trace!("trunc {} with size {} and shift {}", n, width.bits(), 128 - width.bits());
        let result = width.truncate(n);
        trace!("trunc result: {}", result);

        Ok(ScalarInt::try_from_uint(result, width)
            .unwrap_or_else(|| bug!("expected to create ScalarInt from uint {:?}", result)))
    };

    let valtree = match (lit, &ty.kind()) {
        (ast::LitKind::Str(s, _), ty::Ref(_, inner_ty, _)) if inner_ty.is_str() => {
            let str_bytes = s.as_str().as_bytes();
            ty::ValTree::from_raw_bytes(tcx, str_bytes)
        }
        (ast::LitKind::ByteStr(data), ty::Ref(_, inner_ty, _))
            if matches!(inner_ty.kind(), ty::Slice(_)) =>
        {
            let bytes = data as &[u8];
            ty::ValTree::from_raw_bytes(tcx, bytes)
        }
        (ast::LitKind::ByteStr(data), ty::Ref(_, inner_ty, _)) if inner_ty.is_array() => {
            let bytes = data as &[u8];
            ty::ValTree::from_raw_bytes(tcx, bytes)
        }
        (ast::LitKind::Byte(n), ty::Uint(ty::UintTy::U8)) => {
            ty::ValTree::from_scalar_int((*n).into())
        }
        (ast::LitKind::Int(n, _), ty::Uint(_)) | (ast::LitKind::Int(n, _), ty::Int(_)) => {
            let scalar_int =
                trunc(if neg { (*n as i128).overflowing_neg().0 as u128 } else { *n })?;
            ty::ValTree::from_scalar_int(scalar_int)
        }
        (ast::LitKind::Float(n, _), ty::Float(fty)) => {
            parse_float_into_valtree(*n, *fty, neg).ok_or(LitToConstError::Reported)?
        }
        (ast::LitKind::Bool(b), ty::Bool) => ty::ValTree::from_scalar_int((*b).into()),
        (ast::LitKind::Char(c), ty::Char) => ty::ValTree::from_scalar_int((*c).into()),
        (ast::LitKind::Err(_), _) => return Err(LitToConstError::Reported),
        _ => return Err(LitToConstError::TypeError),
    };

    Ok(ty::Const::from_value(tcx, valtree, ty))
}

pub(crate) fn parse_float_into_scalar(
    num: Symbol,
    float_ty: ty::FloatTy,
    neg: bool,
) -> Option<Scalar> {
    let num = num.as_str();
    match float_ty {
        ty::FloatTy::F32 => {
            let Ok(rust_f) = num.parse::<f32>() else { return None };
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

            Some(Scalar::from_f32(f))
        }
        ty::FloatTy::F64 => {
            let Ok(rust_f) = num.parse::<f64>() else { return None };
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

            Some(Scalar::from_f64(f))
        }
    }
}

fn parse_float_into_valtree<'tcx>(
    num: Symbol,
    float_ty: ty::FloatTy,
    neg: bool,
) -> Option<ty::ValTree<'tcx>> {
    parse_float_into_scalar(num, float_ty, neg).map(|s| ty::ValTree::Leaf(s.try_to_int().unwrap()))
}
