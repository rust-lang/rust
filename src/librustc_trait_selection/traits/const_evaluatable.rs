use rustc_middle::ty::{self, TypeFoldable};
use rustc_infer::infer::InferCtxt;
use rustc_middle::ty::subst::SubstsRef;
use rustc_span::Span;
use rustc_span::def_id::DefId;
use rustc_middle::mir::interpret::ErrorHandled;
use rustc_hir::def::DefKind;

pub fn is_const_evaluatable<'cx, 'tcx>(
    infcx: &InferCtxt<'cx, 'tcx>,
    def: ty::WithOptConstParam<DefId>,
    substs: SubstsRef<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    span: Span,
) -> Result<(), ErrorHandled>
{
    let def_kind = infcx.tcx.def_kind(def.did);
    match def_kind {
        DefKind::AnonConst => {
            let mir_body = if let Some(def) = def.as_const_arg() {
                infcx.tcx.optimized_mir_of_const_arg(def)
            } else {
                infcx.tcx.optimized_mir(def.did)
            };
            if mir_body.is_polymorphic {
                return Err(ErrorHandled::TooGeneric);
            }
        }
        _ => {
            if substs.has_param_types_or_consts() {
                return Err(ErrorHandled::TooGeneric);
            }
        }
    }

    match infcx.const_eval_resolve(
        param_env,
        def,
        substs,
        None,
        Some(span),
    ) {
        Ok(_) => Ok(()),
        Err(err) => {
            if matches!(err, ErrorHandled::TooGeneric) {
                infcx.tcx.sess.delay_span_bug(
                    span,
                    &format!("ConstEvaluatable too generic: {:?}, {:?}, {:?}", def, substs, param_env),
                );
            }
            Err(err)
        }
    }
}