//! Checking that constant values used in types can be successfully evaluated.
//!
//! For concrete constants, this is fairly simple as we can just try and evaluate it.
//!
//! When dealing with polymorphic constants, for example `size_of::<T>() - 1`,
//! this is not as easy.
//!
//! In this case we try to build an abstract representation of this constant using
//! `thir_abstract_const` which can then be checked for structural equality with other
//! generic constants mentioned in the `caller_bounds` of the current environment.

use rustc_hir::def::DefKind;
use rustc_infer::infer::InferCtxt;
use rustc_middle::bug;
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::abstract_const::NotConstEvaluatable;
use rustc_middle::ty::{self, TyCtxt, TypeVisitable, TypeVisitableExt, TypeVisitor};
use rustc_span::{DUMMY_SP, Span};
use tracing::{debug, instrument};

use super::EvaluateConstErr;
use crate::traits::ObligationCtxt;

/// Check if a given constant can be evaluated.
#[instrument(skip(infcx), level = "debug")]
pub fn is_const_evaluatable<'tcx>(
    infcx: &InferCtxt<'tcx>,
    unexpanded_ct: ty::Const<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    span: Span,
) -> Result<(), NotConstEvaluatable> {
    let tcx = infcx.tcx;
    match tcx.expand_abstract_consts(unexpanded_ct).kind() {
        ty::ConstKind::Unevaluated(_) | ty::ConstKind::Expr(_) => (),
        ty::ConstKind::Param(_)
        | ty::ConstKind::Bound(_, _)
        | ty::ConstKind::Placeholder(_)
        | ty::ConstKind::Value(_)
        | ty::ConstKind::Error(_) => return Ok(()),
        ty::ConstKind::Infer(_) => return Err(NotConstEvaluatable::MentionsInfer),
    };

    if tcx.features().generic_const_exprs() {
        let ct = tcx.expand_abstract_consts(unexpanded_ct);

        let is_anon_ct = if let ty::ConstKind::Unevaluated(uv) = ct.kind() {
            tcx.def_kind(uv.def) == DefKind::AnonConst
        } else {
            false
        };

        if !is_anon_ct {
            if satisfied_from_param_env(tcx, infcx, ct, param_env) {
                return Ok(());
            }
            if ct.has_non_region_infer() {
                return Err(NotConstEvaluatable::MentionsInfer);
            } else if ct.has_non_region_param() {
                return Err(NotConstEvaluatable::MentionsParam);
            }
        }

        match unexpanded_ct.kind() {
            ty::ConstKind::Expr(_) => {
                // FIXME(generic_const_exprs): we have a fully concrete `ConstKind::Expr`, but
                // haven't implemented evaluating `ConstKind::Expr` yet, so we are unable to tell
                // if it is evaluatable or not. As this is unreachable for now, we can simple ICE
                // here.
                tcx.dcx().span_bug(span, "evaluating `ConstKind::Expr` is not currently supported");
            }
            ty::ConstKind::Unevaluated(_) => {
                match crate::traits::try_evaluate_const(infcx, unexpanded_ct, param_env) {
                    Err(EvaluateConstErr::HasGenericsOrInfers) => {
                        Err(NotConstEvaluatable::Error(infcx.dcx().span_delayed_bug(
                            span,
                            "Missing value for constant, but no error reported?",
                        )))
                    }
                    Err(
                        EvaluateConstErr::EvaluationFailure(e)
                        | EvaluateConstErr::InvalidConstParamTy(e),
                    ) => Err(NotConstEvaluatable::Error(e)),
                    Ok(_) => Ok(()),
                }
            }
            _ => bug!("unexpected constkind in `is_const_evalautable: {unexpanded_ct:?}`"),
        }
    } else if tcx.features().min_generic_const_args() {
        // This is a sanity check to make sure that non-generics consts are checked to
        // be evaluatable in case they aren't cchecked elsewhere. This will NOT error
        // if the const uses generics, as desired.
        crate::traits::evaluate_const(infcx, unexpanded_ct, param_env);
        Ok(())
    } else {
        let uv = match unexpanded_ct.kind() {
            ty::ConstKind::Unevaluated(uv) => uv,
            ty::ConstKind::Expr(_) => {
                bug!("`ConstKind::Expr` without `feature(generic_const_exprs)` enabled")
            }
            _ => bug!("unexpected constkind in `is_const_evalautable: {unexpanded_ct:?}`"),
        };

        match crate::traits::try_evaluate_const(infcx, unexpanded_ct, param_env) {
            // If we're evaluating a generic foreign constant, under a nightly compiler while
            // the current crate does not enable `feature(generic_const_exprs)`, abort
            // compilation with a useful error.
            Err(_)
                if tcx.sess.is_nightly_build()
                    && satisfied_from_param_env(
                        tcx,
                        infcx,
                        tcx.expand_abstract_consts(unexpanded_ct),
                        param_env,
                    ) =>
            {
                tcx.dcx()
                    .struct_span_fatal(
                        // Slightly better span than just using `span` alone
                        if span == DUMMY_SP { tcx.def_span(uv.def) } else { span },
                        "failed to evaluate generic const expression",
                    )
                    .with_note("the crate this constant originates from uses `#![feature(generic_const_exprs)]`")
                    .with_span_suggestion_verbose(
                        DUMMY_SP,
                        "consider enabling this feature",
                        "#![feature(generic_const_exprs)]\n",
                        rustc_errors::Applicability::MaybeIncorrect,
                    )
                    .emit()
            }

            Err(EvaluateConstErr::HasGenericsOrInfers) => {
                let err = if uv.has_non_region_infer() {
                    NotConstEvaluatable::MentionsInfer
                } else if uv.has_non_region_param() {
                    NotConstEvaluatable::MentionsParam
                } else {
                    let guar = infcx.dcx().span_delayed_bug(
                        span,
                        "Missing value for constant, but no error reported?",
                    );
                    NotConstEvaluatable::Error(guar)
                };

                Err(err)
            }
            Err(
                EvaluateConstErr::EvaluationFailure(e) | EvaluateConstErr::InvalidConstParamTy(e),
            ) => Err(NotConstEvaluatable::Error(e)),
            Ok(_) => Ok(()),
        }
    }
}

#[instrument(skip(infcx, tcx), level = "debug")]
fn satisfied_from_param_env<'tcx>(
    tcx: TyCtxt<'tcx>,
    infcx: &InferCtxt<'tcx>,
    ct: ty::Const<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
) -> bool {
    // Try to unify with each subtree in the AbstractConst to allow for
    // `N + 1` being const evaluatable even if theres only a `ConstEvaluatable`
    // predicate for `(N + 1) * 2`
    struct Visitor<'a, 'tcx> {
        ct: ty::Const<'tcx>,
        param_env: ty::ParamEnv<'tcx>,

        infcx: &'a InferCtxt<'tcx>,
        single_match: Option<Result<ty::Const<'tcx>, ()>>,
    }

    impl<'a, 'tcx> TypeVisitor<TyCtxt<'tcx>> for Visitor<'a, 'tcx> {
        fn visit_const(&mut self, c: ty::Const<'tcx>) {
            debug!("is_const_evaluatable: candidate={:?}", c);
            if self.infcx.probe(|_| {
                let ocx = ObligationCtxt::new(self.infcx);
                ocx.eq(&ObligationCause::dummy(), self.param_env, c, self.ct).is_ok()
                    && ocx.select_all_or_error().is_empty()
            }) {
                self.single_match = match self.single_match {
                    None => Some(Ok(c)),
                    Some(Ok(o)) if o == c => Some(Ok(c)),
                    Some(_) => Some(Err(())),
                };
            }

            if let ty::ConstKind::Expr(e) = c.kind() {
                e.visit_with(self);
            } else {
                // FIXME(generic_const_exprs): This doesn't recurse into `<T as Trait<U>>::ASSOC`'s args.
                // This is currently unobservable as `<T as Trait<{ U + 1 }>>::ASSOC` creates an anon const
                // with its own `ConstEvaluatable` bound in the param env which we will visit separately.
                //
                // If we start allowing directly writing `ConstKind::Expr` without an intermediate anon const
                // this will be incorrect. It might be worth investigating making `predicates_of` elaborate
                // all of the `ConstEvaluatable` bounds rather than having a visitor here.
            }
        }
    }

    let mut single_match: Option<Result<ty::Const<'tcx>, ()>> = None;

    for pred in param_env.caller_bounds() {
        match pred.kind().skip_binder() {
            ty::ClauseKind::ConstEvaluatable(ce) => {
                let b_ct = tcx.expand_abstract_consts(ce);
                let mut v = Visitor { ct, infcx, param_env, single_match };
                let _ = b_ct.visit_with(&mut v);

                single_match = v.single_match;
            }
            _ => {} // don't care
        }
    }

    if let Some(Ok(c)) = single_match {
        let ocx = ObligationCtxt::new(infcx);
        assert!(ocx.eq(&ObligationCause::dummy(), param_env, c, ct).is_ok());
        assert!(ocx.select_all_or_error().is_empty());
        return true;
    }

    debug!("is_const_evaluatable: no");
    false
}
