use rustc_ast as ast;
use rustc_middle::mir::interpret::{LitToConstError, LitToConstInput};
use rustc_middle::ty::{self, ParamEnv, ScalarInt, TyCtxt};

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
        (ast::LitKind::Bool(b), ty::Bool) => ty::ValTree::from_scalar_int((*b).into()),
        (ast::LitKind::Char(c), ty::Char) => ty::ValTree::from_scalar_int((*c).into()),
        (ast::LitKind::Err, _) => return Err(LitToConstError::Reported),
        _ => return Err(LitToConstError::TypeError),
    };

    Ok(ty::Const::from_value(tcx, valtree, ty))
}
