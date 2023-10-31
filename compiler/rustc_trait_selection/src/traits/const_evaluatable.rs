//! Checking that constant values used in types can be successfully evaluated.
//!
//! For concrete constants, this is fairly simple as we can just try and evaluate it.
//!
//! When dealing with polymorphic constants, for example `std::mem::size_of::<T>() - 1`,
//! this is not as easy.
//!
//! In this case we try to build an abstract representation of this constant using
//! `thir_abstract_const` which can then be checked for structural equality with other
//! generic constants mentioned in the `caller_bounds` of the current environment.
use rustc_hir::def::DefKind;
use rustc_infer::infer::InferCtxt;
use rustc_middle::mir::interpret::ErrorHandled;

use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::abstract_const::NotConstEvaluatable;
use rustc_middle::ty::{self, TyCtxt, TypeVisitable, TypeVisitableExt, TypeVisitor};

use rustc_span::Span;
use std::ops::ControlFlow;

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

    if tcx.features().generic_const_exprs {
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
                // FIXME(generic_const_exprs): we have a `ConstKind::Expr` which is fully concrete, but
                // currently it is not possible to evaluate `ConstKind::Expr` so we are unable to tell if it
                // is evaluatable or not. For now we just ICE until this is implemented.
                Err(NotConstEvaluatable::Error(tcx.sess.delay_span_bug(
                    span,
                    "evaluating `ConstKind::Expr` is not currently supported",
                )))
            }
            ty::ConstKind::Unevaluated(uv) => {
                let concrete = infcx.const_eval_resolve(param_env, uv, Some(span));
                match concrete {
                    Err(ErrorHandled::TooGeneric) => {
                        Err(NotConstEvaluatable::Error(infcx.tcx.sess.delay_span_bug(
                            span,
                            "Missing value for constant, but no error reported?",
                        )))
                    }
                    Err(ErrorHandled::Reported(e)) => Err(NotConstEvaluatable::Error(e.into())),
                    Ok(_) => Ok(()),
                }
            }
            _ => bug!("unexpected constkind in `is_const_evalautable: {unexpanded_ct:?}`"),
        }
    } else {
        let uv = match unexpanded_ct.kind() {
            ty::ConstKind::Unevaluated(uv) => uv,
            ty::ConstKind::Expr(_) => {
                bug!("`ConstKind::Expr` without `feature(generic_const_exprs)` enabled")
            }
            _ => bug!("unexpected constkind in `is_const_evalautable: {unexpanded_ct:?}`"),
        };

        // FIXME: We should only try to evaluate a given constant here if it is fully concrete
        // as we don't want to allow things like `[u8; std::mem::size_of::<*mut T>()]`.
        //
        // We previously did not check this, so we only emit a future compat warning if
        // const evaluation succeeds and the given constant is still polymorphic for now
        // and hopefully soon change this to an error.
        //
        // See #74595 for more details about this.
        let concrete = infcx.const_eval_resolve(param_env, uv, Some(span));
        match concrete {
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
                tcx.sess
                    .struct_span_fatal(
                        // Slightly better span than just using `span` alone
                        if span == rustc_span::DUMMY_SP { tcx.def_span(uv.def) } else { span },
                        "failed to evaluate generic const expression",
                    )
                    .note("the crate this constant originates from uses `#![feature(generic_const_exprs)]`")
                    .span_suggestion_verbose(
                        rustc_span::DUMMY_SP,
                        "consider enabling this feature",
                        "#![feature(generic_const_exprs)]\n",
                        rustc_errors::Applicability::MaybeIncorrect,
                    )
                    .emit()
            }

            Err(ErrorHandled::TooGeneric) => {
                let err = if uv.has_non_region_infer() {
                    NotConstEvaluatable::MentionsInfer
                } else if uv.has_non_region_param() {
                    NotConstEvaluatable::MentionsParam
                } else {
                    let guar = infcx
                        .tcx
                        .sess
                        .delay_span_bug(span, "Missing value for constant, but no error reported?");
                    NotConstEvaluatable::Error(guar)
                };

                Err(err)
            }
            Err(ErrorHandled::Reported(e)) => Err(NotConstEvaluatable::Error(e.into())),
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
        type BreakTy = ();
        fn visit_const(&mut self, c: ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
            debug!("is_const_evaluatable: candidate={:?}", c);
            if self.infcx.probe(|_| {
                let ocx = ObligationCtxt::new(self.infcx);
                ocx.eq(&ObligationCause::dummy(), self.param_env, c.ty(), self.ct.ty()).is_ok()
                    && ocx.eq(&ObligationCause::dummy(), self.param_env, c, self.ct).is_ok()
                    && ocx.select_all_or_error().is_empty()
            }) {
                self.single_match = match self.single_match {
                    None => Some(Ok(c)),
                    Some(Ok(o)) if o == c => Some(Ok(c)),
                    Some(_) => Some(Err(())),
                };
            }

            if let ty::ConstKind::Expr(e) = c.kind() {
                e.visit_with(self)
            } else {
                // FIXME(generic_const_exprs): This doesn't recurse into `<T as Trait<U>>::ASSOC`'s args.
                // This is currently unobservable as `<T as Trait<{ U + 1 }>>::ASSOC` creates an anon const
                // with its own `ConstEvaluatable` bound in the param env which we will visit separately.
                //
                // If we start allowing directly writing `ConstKind::Expr` without an intermediate anon const
                // this will be incorrect. It might be worth investigating making `predicates_of` elaborate
                // all of the `ConstEvaluatable` bounds rather than having a visitor here.
                ControlFlow::Continue(())
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
        assert!(ocx.eq(&ObligationCause::dummy(), param_env, c.ty(), ct.ty()).is_ok());
        assert!(ocx.eq(&ObligationCause::dummy(), param_env, c, ct).is_ok());
        assert!(ocx.select_all_or_error().is_empty());
        return true;
    }

    debug!("is_const_evaluatable: no");
    false
}
