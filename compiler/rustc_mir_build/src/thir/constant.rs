use rustc_ast as ast;
use rustc_middle::mir::interpret::{LitToConstError, LitToConstInput};
use rustc_middle::ty::{self, ParamEnv, ScalarInt, Ty, TyCtxt};
use rustc_span::DUMMY_SP;

fn trunc<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    lit: &ast::LitKind,
    n: u128,
) -> Result<ScalarInt, LitToConstError> {
    let param_ty = ParamEnv::reveal_all().and(ty);
    let width =
        tcx.layout_of(param_ty)
            .map_err(|_| {
                LitToConstError::Reported(tcx.sess.delay_span_bug(
                    DUMMY_SP,
                    format!("couldn't compute width of literal: {:?}", lit),
                ))
            })?
            .size;
    trace!("trunc {} with size {} and shift {}", n, width.bits(), 128 - width.bits());
    let result = width.truncate(n);
    trace!("trunc result: {}", result);

    Ok(ScalarInt::try_from_uint(result, width)
        .unwrap_or_else(|| bug!("expected to create ScalarInt from uint {:?}", result)))
}

fn get_valtree<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    neg: bool,
    lit: &ast::LitKind,
) -> Result<ty::ValTree<'tcx>, LitToConstError> {
    Ok(match (lit, &ty.kind()) {
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
            let scalar_int = trunc(
                tcx,
                ty,
                lit,
                if neg { (*n as i128).overflowing_neg().0 as u128 } else { *n },
            )?;
            ty::ValTree::from_scalar_int(scalar_int)
        }
        (ast::LitKind::Bool(b), ty::Bool) => ty::ValTree::from_scalar_int((*b).into()),
        (ast::LitKind::Char(c), ty::Char) => ty::ValTree::from_scalar_int((*c).into()),
        (ast::LitKind::Err, _) => {
            return Err(LitToConstError::Reported(
                tcx.sess.delay_span_bug(DUMMY_SP, "encountered LitKind::Err during mir build"),
            ));
        }
        (_, ty::Projection(ty::ProjectionTy { substs, item_def_id })) => {
            let binder_ty = tcx.bound_type_of(*item_def_id);
            let ty = binder_ty.subst(tcx, substs);
            get_valtree(tcx, ty, neg, lit)?
        }
        _ => return Err(LitToConstError::TypeError),
    })
}

pub(crate) fn lit_to_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    lit_input: LitToConstInput<'tcx>,
) -> Result<ty::Const<'tcx>, LitToConstError> {
    let LitToConstInput { lit, ty, neg } = lit_input;

    let valtree = get_valtree(tcx, ty, neg, lit)?;
    Ok(ty::Const::from_value(tcx, valtree, ty))
}
