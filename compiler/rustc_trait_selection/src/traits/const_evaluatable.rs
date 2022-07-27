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
use rustc_infer::infer::InferCtxt;
use rustc_middle::mir::interpret::ErrorHandled;

use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::abstract_const::NotConstEvaluatable;
use rustc_middle::ty::{self, TyCtxt, TypeVisitable, TypeVisitor};

use rustc_span::Span;
use std::ops::ControlFlow;

/// Check if a given constant can be evaluated.
#[instrument(skip(infcx), level = "debug")]
pub fn is_const_evaluatable<'tcx>(
    infcx: &InferCtxt<'tcx>,
    ct: ty::Const<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    span: Span,
) -> Result<(), NotConstEvaluatable> {
    let tcx = infcx.tcx;
    let uv = match ct.kind() {
        ty::ConstKind::Unevaluated(uv) => uv,
        // should be recursivee fixes.
        ty::ConstKind::Expr(..) => todo!(),
        ty::ConstKind::Param(_)
        | ty::ConstKind::Bound(_, _)
        | ty::ConstKind::Placeholder(_)
        | ty::ConstKind::Value(_)
        | ty::ConstKind::Error(_) => return Ok(()),
        ty::ConstKind::Infer(_) => return Err(NotConstEvaluatable::MentionsInfer),
    };

    if tcx.features().generic_const_exprs {
        let substs = tcx.erase_regions(uv.substs);
        if let Some(ct) =
            tcx.expand_bound_abstract_const(tcx.bound_abstract_const(uv.def), substs)?
        {
            if satisfied_from_param_env(tcx, infcx, ct, param_env)? {
                return Ok(());
            }
            if ct.has_non_region_infer() {
                return Err(NotConstEvaluatable::MentionsInfer);
            } else if ct.has_non_region_param() {
                return Err(NotConstEvaluatable::MentionsParam);
            }
        }
        let concrete = infcx.const_eval_resolve(param_env, uv, Some(span));
        match concrete {
            Err(ErrorHandled::TooGeneric) => Err(NotConstEvaluatable::Error(
                infcx
                    .tcx
                    .sess
                    .delay_span_bug(span, "Missing value for constant, but no error reported?"),
            )),
            Err(ErrorHandled::Reported(e)) => Err(NotConstEvaluatable::Error(e)),
            Ok(_) => Ok(()),
        }
    } else {
        // FIXME: We should only try to evaluate a given constant here if it is fully concrete
        // as we don't want to allow things like `[u8; std::mem::size_of::<*mut T>()]`.
        //
        // We previously did not check this, so we only emit a future compat warning if
        // const evaluation succeeds and the given constant is still polymorphic for now
        // and hopefully soon change this to an error.
        //
        // See #74595 for more details about this.
        let concrete = infcx.const_eval_resolve(param_env, uv, Some(span));

        let substs = tcx.erase_regions(uv.substs);
        match concrete {
          // If we're evaluating a foreign constant, under a nightly compiler without generic
          // const exprs, AND it would've passed if that expression had been evaluated with
          // generic const exprs, then suggest using generic const exprs.
          Err(_) if tcx.sess.is_nightly_build()
            && let Ok(Some(ct)) =
            tcx.expand_bound_abstract_const(tcx.bound_abstract_const(uv.def), substs)
            && let ty::ConstKind::Expr(_expr) = ct.kind()
            && satisfied_from_param_env(tcx, infcx, ct, param_env) == Ok(true) => {
              tcx.sess
                  .struct_span_fatal(
                      // Slightly better span than just using `span` alone
                      if span == rustc_span::DUMMY_SP { tcx.def_span(uv.def.did) } else { span },
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
                    let guar = infcx.tcx.sess.delay_span_bug(span, format!("Missing value for constant, but no error reported?"));
                    NotConstEvaluatable::Error(guar)
                };

                Err(err)
            },
            Err(ErrorHandled::Reported(e)) => Err(NotConstEvaluatable::Error(e)),
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
) -> Result<bool, NotConstEvaluatable> {
    for pred in param_env.caller_bounds() {
        match pred.kind().skip_binder() {
            ty::PredicateKind::ConstEvaluatable(uv) => {
                let ty::ConstKind::Unevaluated(uv) = uv.kind() else {
                    continue
                };
                let substs = tcx.erase_regions(uv.substs);
                let Some(b_ct) =
                tcx.expand_bound_abstract_const(tcx.bound_abstract_const(uv.def), substs)? else {
                    return Ok(false);
                };

                // Try to unify with each subtree in the AbstractConst to allow for
                // `N + 1` being const evaluatable even if theres only a `ConstEvaluatable`
                // predicate for `(N + 1) * 2`
                struct Visitor<'a, 'tcx> {
                    ct: ty::Const<'tcx>,
                    param_env: ty::ParamEnv<'tcx>,

                    infcx: &'a InferCtxt<'tcx>,
                }
                impl<'a, 'tcx> TypeVisitor<'tcx> for Visitor<'a, 'tcx> {
                    type BreakTy = ();
                    fn visit_const(&mut self, c: ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
                        if c.ty() == self.ct.ty()
                            && let Ok(_nested_obligations) = self
                                .infcx
                                .at(&ObligationCause::dummy(), self.param_env)
                                .eq(c, self.ct)
                        {
                            //let obligations = nested_obligations.into_obligations();
                            ControlFlow::BREAK
                        } else if let ty::ConstKind::Expr(e) = c.kind() {
                            e.visit_with(self)
                        } else {
                            ControlFlow::CONTINUE
                        }
                    }
                }

                let mut v = Visitor { ct, infcx, param_env };
                let result = b_ct.visit_with(&mut v);

                if let ControlFlow::Break(()) = result {
                    debug!("is_const_evaluatable: abstract_const ~~> ok");
                    return Ok(true);
                }
            }
            _ => {} // don't care
        }
    }

    Ok(false)
}
