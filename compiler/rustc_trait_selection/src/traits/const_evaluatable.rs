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
use rustc_middle::ty::abstract_const::{
    walk_abstract_const, AbstractConst, ConstUnifyCtxt, FailureKind, Node, NotConstEvaluatable,
};
use rustc_middle::ty::{self, TyCtxt, TypeVisitable};
use rustc_session::lint;
use rustc_span::Span;

use std::cmp;
use std::ops::ControlFlow;

/// Check if a given constant can be evaluated.
#[instrument(skip(infcx), level = "debug")]
pub fn is_const_evaluatable<'cx, 'tcx>(
    infcx: &InferCtxt<'cx, 'tcx>,
    uv: ty::Unevaluated<'tcx, ()>,
    param_env: ty::ParamEnv<'tcx>,
    span: Span,
) -> Result<(), NotConstEvaluatable> {
    let tcx = infcx.tcx;

    if tcx.features().generic_const_exprs {
        if let Some(ct) = AbstractConst::new(tcx, uv)? {
            if satisfied_from_param_env(tcx, ct, param_env)? {
                return Ok(());
            }

            let mut failure_kind = FailureKind::Concrete;
            walk_abstract_const::<!, _>(tcx, ct, |node| match node.root(tcx) {
                Node::Leaf(leaf) => {
                    if leaf.has_infer_types_or_consts() {
                        failure_kind = FailureKind::MentionsInfer;
                    } else if leaf.has_param_types_or_consts() {
                        failure_kind = cmp::min(failure_kind, FailureKind::MentionsParam);
                    }

                    ControlFlow::CONTINUE
                }
                Node::Cast(_, _, ty) => {
                    if ty.has_infer_types_or_consts() {
                        failure_kind = FailureKind::MentionsInfer;
                    } else if ty.has_param_types_or_consts() {
                        failure_kind = cmp::min(failure_kind, FailureKind::MentionsParam);
                    }

                    ControlFlow::CONTINUE
                }
                Node::Binop(_, _, _) | Node::UnaryOp(_, _) | Node::FunctionCall(_, _) => {
                    ControlFlow::CONTINUE
                }
            });

            match failure_kind {
                FailureKind::MentionsInfer => {
                    return Err(NotConstEvaluatable::MentionsInfer);
                }
                FailureKind::MentionsParam => {
                    return Err(NotConstEvaluatable::MentionsParam);
                }
                // returned below
                FailureKind::Concrete => {}
            }
        }
        let concrete = infcx.const_eval_resolve(param_env, uv.expand(), Some(span));
        match concrete {
            Err(ErrorHandled::TooGeneric) => Err(if !uv.has_infer_types_or_consts() {
                infcx
                    .tcx
                    .sess
                    .delay_span_bug(span, &format!("unexpected `TooGeneric` for {:?}", uv));
                NotConstEvaluatable::MentionsParam
            } else {
                NotConstEvaluatable::MentionsInfer
            }),
            Err(ErrorHandled::Linted) => {
                let reported = infcx
                    .tcx
                    .sess
                    .delay_span_bug(span, "constant in type had error reported as lint");
                Err(NotConstEvaluatable::Error(reported))
            }
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
        let concrete = infcx.const_eval_resolve(param_env, uv.expand(), Some(span));

        match concrete {
          // If we're evaluating a foreign constant, under a nightly compiler without generic
          // const exprs, AND it would've passed if that expression had been evaluated with
          // generic const exprs, then suggest using generic const exprs.
          Err(_) if tcx.sess.is_nightly_build()
            && let Ok(Some(ct)) = AbstractConst::new(tcx, uv)
            && satisfied_from_param_env(tcx, ct, param_env) == Ok(true) => {
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

            Err(ErrorHandled::TooGeneric) => Err(if uv.has_infer_types_or_consts() {
                NotConstEvaluatable::MentionsInfer
                } else {
                NotConstEvaluatable::MentionsParam
            }),
            Err(ErrorHandled::Linted) => {
                let reported =
                    infcx.tcx.sess.delay_span_bug(span, "constant in type had error reported as lint");
                Err(NotConstEvaluatable::Error(reported))
            }
            Err(ErrorHandled::Reported(e)) => Err(NotConstEvaluatable::Error(e)),
            Ok(_) => {
              if uv.substs.has_param_types_or_consts() {
                  assert!(matches!(infcx.tcx.def_kind(uv.def.did), DefKind::AnonConst));
                  let mir_body = infcx.tcx.mir_for_ctfe_opt_const_arg(uv.def);

                  if mir_body.is_polymorphic {
                      let Some(local_def_id) = uv.def.did.as_local() else { return Ok(()) };
                      tcx.struct_span_lint_hir(
                          lint::builtin::CONST_EVALUATABLE_UNCHECKED,
                          tcx.hir().local_def_id_to_hir_id(local_def_id),
                          span,
                          |err| {
                              err.build("cannot use constants which depend on generic parameters in types").emit();
                        })
                  }
              }

              Ok(())
            },
        }
    }
}

#[instrument(skip(tcx), level = "debug")]
fn satisfied_from_param_env<'tcx>(
    tcx: TyCtxt<'tcx>,
    ct: AbstractConst<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
) -> Result<bool, NotConstEvaluatable> {
    for pred in param_env.caller_bounds() {
        match pred.kind().skip_binder() {
            ty::PredicateKind::ConstEvaluatable(uv) => {
                if let Some(b_ct) = AbstractConst::new(tcx, uv)? {
                    let const_unify_ctxt = ConstUnifyCtxt { tcx, param_env };

                    // Try to unify with each subtree in the AbstractConst to allow for
                    // `N + 1` being const evaluatable even if theres only a `ConstEvaluatable`
                    // predicate for `(N + 1) * 2`
                    let result = walk_abstract_const(tcx, b_ct, |b_ct| {
                        match const_unify_ctxt.try_unify(ct, b_ct) {
                            true => ControlFlow::BREAK,
                            false => ControlFlow::CONTINUE,
                        }
                    });

                    if let ControlFlow::Break(()) = result {
                        debug!("is_const_evaluatable: abstract_const ~~> ok");
                        return Ok(true);
                    }
                }
            }
            _ => {} // don't care
        }
    }

    Ok(false)
}
