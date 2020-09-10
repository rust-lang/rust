use rustc_hir::def::DefKind;
use rustc_infer::infer::InferCtxt;
use rustc_middle::mir::interpret::ErrorHandled;
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{self, TypeFoldable};
use rustc_session::lint;
use rustc_span::def_id::DefId;
use rustc_span::Span;

pub fn is_const_evaluatable<'cx, 'tcx>(
    infcx: &InferCtxt<'cx, 'tcx>,
    def: ty::WithOptConstParam<DefId>,
    substs: SubstsRef<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    span: Span,
) -> Result<(), ErrorHandled> {
    debug!("is_const_evaluatable({:?}, {:?})", def, substs);
    if infcx.tcx.features().const_evaluatable_checked {
        // FIXME(const_evaluatable_checked): Actually look into generic constants to
        // implement const equality.
        for pred in param_env.caller_bounds() {
            match pred.skip_binders() {
                ty::PredicateAtom::ConstEvaluatable(b_def, b_substs) => {
                    debug!("is_const_evaluatable: caller_bound={:?}, {:?}", b_def, b_substs);
                    if b_def == def && b_substs == substs {
                        debug!("is_const_evaluatable: caller_bound ~~> ok");
                        return Ok(());
                    }
                }
                _ => {} // don't care
            }
        }
    }

    let future_compat_lint = || {
        if let Some(local_def_id) = def.did.as_local() {
            infcx.tcx.struct_span_lint_hir(
                lint::builtin::CONST_EVALUATABLE_UNCHECKED,
                infcx.tcx.hir().local_def_id_to_hir_id(local_def_id),
                span,
                |err| {
                    err.build("cannot use constants which depend on generic parameters in types")
                        .emit();
                },
            );
        }
    };

    // FIXME: We should only try to evaluate a given constant here if it is fully concrete
    // as we don't want to allow things like `[u8; std::mem::size_of::<*mut T>()]`.
    //
    // We previously did not check this, so we only emit a future compat warning if
    // const evaluation succeeds and the given constant is still polymorphic for now
    // and hopefully soon change this to an error.
    //
    // See #74595 for more details about this.
    let concrete = infcx.const_eval_resolve(param_env, def, substs, None, Some(span));

    if concrete.is_ok() && substs.has_param_types_or_consts() {
        match infcx.tcx.def_kind(def.did) {
            DefKind::AnonConst => {
                let mir_body = if let Some(def) = def.as_const_arg() {
                    infcx.tcx.optimized_mir_of_const_arg(def)
                } else {
                    infcx.tcx.optimized_mir(def.did)
                };

                if mir_body.is_polymorphic {
                    future_compat_lint();
                }
            }
            _ => future_compat_lint(),
        }
    }

    debug!(?concrete, "is_const_evaluatable");
    concrete.map(drop)
}
