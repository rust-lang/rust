use rustc_ast as ast;
use rustc_middle::mir::interpret::{LitToConstError, LitToConstInput};
use rustc_middle::ty::{self, ParamEnv, ScalarInt, TyCtxt};
use rustc_span::DUMMY_SP;

use crate::build::parse_float_into_scalar;

pub(crate) fn lit_to_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    lit_input: LitToConstInput<'tcx>,
) -> Option<ty::Const<'tcx>> {
    let LitToConstInput { lit, ty, neg } = lit_input;

    let valtree = match (lit, &ty.kind()) {
        (ast::LitKind::Str(s, _), ty::Ref(_, inner_ty, _)) if inner_ty.is_str() => {
            let str_bytes = s.as_str().as_bytes();
            ty::ValTree::from_raw_bytes(tcx, str_bytes)
        }
        (ast::LitKind::ByteStr(data, _), ty::Ref(_, inner_ty, _))
            if matches!(inner_ty.kind(), ty::Slice(_)) =>
        {
            let bytes = data as &[u8];
            ty::ValTree::from_raw_bytes(tcx, bytes)
        }
        (ast::LitKind::ByteStr(data, _), ty::Ref(_, inner_ty, _)) if inner_ty.is_array() => {
            let bytes = data as &[u8];
            ty::ValTree::from_raw_bytes(tcx, bytes)
        }
        (ast::LitKind::Byte(n), ty::Uint(ty::UintTy::U8)) => {
            ty::ValTree::from_scalar_int((*n).into())
        }
        (ast::LitKind::CStr(data, _), ty::Ref(_, inner_ty, _)) if matches!(inner_ty.kind(), ty::Adt(def, _) if Some(def.did()) == tcx.lang_items().c_str()) =>
        {
            let bytes = data as &[u8];
            ty::ValTree::from_raw_bytes(tcx, bytes)
        }
        (ast::LitKind::Int(n, _), ty::Uint(_)) | (ast::LitKind::Int(n, _), ty::Int(_)) => {
            let n = if neg { (*n as i128).overflowing_neg().0 as u128 } else { *n };
            let param_ty = ParamEnv::reveal_all().and(ty);
            let width = tcx
                .layout_of(param_ty)
                .unwrap_or_else(|_| bug!("should always be able to compute the layout of a scalar"))
                .size;
            trace!("trunc {} with size {} and shift {}", n, width.bits(), 128 - width.bits());
            let result = width.truncate(n);
            trace!("trunc result: {}", result);

            let scalar_int = ScalarInt::try_from_uint(result, width)
                .unwrap_or_else(|| bug!("expected to create ScalarInt from uint {:?}", result));

            ty::ValTree::from_scalar_int(scalar_int)
        }
        (ast::LitKind::Bool(b), ty::Bool) => ty::ValTree::from_scalar_int((*b).into()),
        (ast::LitKind::Char(c), ty::Char) => ty::ValTree::from_scalar_int((*c).into()),
        // If there's a type mismatch, it may be a legitimate type mismatch, or
        // it might be an unnormalized type. Return `None`, which falls back
        // to an unevaluated const, and the error will be caught (or we will see
        // that it's ok) elsewhere.
        _ => return None,
    };

    Some(ty::Const::new_value(tcx, valtree, ty))
}

/// The old "lit_to_const", which assumes that the type passed in `lit_input`
/// is normalized, and will error out if that is not true. This should only
/// be used in pattern building code (and ideally we'd get rid of this altogether).
pub(crate) fn lit_to_const_for_patterns<'tcx>(
    tcx: TyCtxt<'tcx>,
    lit_input: LitToConstInput<'tcx>,
) -> Result<ty::Const<'tcx>, LitToConstError> {
    let LitToConstInput { lit, ty, neg } = lit_input;

    let trunc = |n| {
        let param_ty = ParamEnv::reveal_all().and(ty);
        let width = tcx
            .layout_of(param_ty)
            .map_err(|_| {
                LitToConstError::Reported(tcx.sess.delay_span_bug(
                    DUMMY_SP,
                    format!("couldn't compute width of literal: {:?}", lit_input.lit),
                ))
            })?
            .size;
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
        (ast::LitKind::ByteStr(data, _), ty::Ref(_, inner_ty, _))
            if matches!(inner_ty.kind(), ty::Slice(_)) =>
        {
            let bytes = data as &[u8];
            ty::ValTree::from_raw_bytes(tcx, bytes)
        }
        (ast::LitKind::ByteStr(data, _), ty::Ref(_, inner_ty, _)) if inner_ty.is_array() => {
            let bytes = data as &[u8];
            ty::ValTree::from_raw_bytes(tcx, bytes)
        }
        (ast::LitKind::Byte(n), ty::Uint(ty::UintTy::U8)) => {
            ty::ValTree::from_scalar_int((*n).into())
        }
        (ast::LitKind::CStr(data, _), ty::Ref(_, inner_ty, _)) if matches!(inner_ty.kind(), ty::Adt(def, _) if Some(def.did()) == tcx.lang_items().c_str()) =>
        {
            let bytes = data as &[u8];
            ty::ValTree::from_raw_bytes(tcx, bytes)
        }
        (ast::LitKind::Int(n, _), ty::Uint(_)) | (ast::LitKind::Int(n, _), ty::Int(_)) => {
            let scalar_int =
                trunc(if neg { (*n as i128).overflowing_neg().0 as u128 } else { *n })?;
            ty::ValTree::from_scalar_int(scalar_int)
        }
        (ast::LitKind::Bool(b), ty::Bool) => ty::ValTree::from_scalar_int((*b).into()),
        (ast::LitKind::Float(n, _), ty::Float(fty)) => {
            let bits = parse_float_into_scalar(*n, *fty, neg)
                .ok_or_else(|| {
                    LitToConstError::Reported(tcx.sess.delay_span_bug(
                        DUMMY_SP,
                        format!("couldn't parse float literal: {:?}", lit_input.lit),
                    ))
                })?
                .assert_int();
            ty::ValTree::from_scalar_int(bits)
        }
        (ast::LitKind::Char(c), ty::Char) => ty::ValTree::from_scalar_int((*c).into()),
        (ast::LitKind::Err, _) => {
            return Err(LitToConstError::Reported(
                tcx.sess.delay_span_bug(DUMMY_SP, "encountered LitKind::Err during mir build"),
            ));
        }
        _ => return Err(LitToConstError::TypeError),
    };

    Ok(ty::Const::new_value(tcx, valtree, ty))
}
