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
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def::DefKind;
use rustc_infer::infer::InferCtxt;
use rustc_middle::mir::interpret::ErrorHandled;
use rustc_middle::ty::abstract_const::{
    walk_abstract_const, AbstractConst, FailureKind, Node, NotConstEvaluatable,
};
use rustc_middle::ty::{self, TyCtxt, TypeVisitable};
use rustc_session::lint;
use rustc_span::Span;

use std::iter;
use std::ops::ControlFlow;

pub struct ConstUnifyCtxt<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub param_env: ty::ParamEnv<'tcx>,
}

impl<'tcx> ConstUnifyCtxt<'tcx> {
    // Substitutes generics repeatedly to allow AbstractConsts to unify where a
    // ConstKind::Unevaluated could be turned into an AbstractConst that would unify e.g.
    // Param(N) should unify with Param(T), substs: [Unevaluated("T2", [Unevaluated("T3", [Param(N)])])]
    #[inline]
    #[instrument(skip(self), level = "debug")]
    fn try_replace_substs_in_root(
        &self,
        mut abstr_const: AbstractConst<'tcx>,
    ) -> Option<AbstractConst<'tcx>> {
        while let Node::Leaf(ct) = abstr_const.root(self.tcx) {
            match AbstractConst::from_const(self.tcx, ct) {
                Ok(Some(act)) => abstr_const = act,
                Ok(None) => break,
                Err(_) => return None,
            }
        }

        Some(abstr_const)
    }

    /// Tries to unify two abstract constants using structural equality.
    #[instrument(skip(self), level = "debug")]
    pub fn try_unify(&self, a: AbstractConst<'tcx>, b: AbstractConst<'tcx>) -> bool {
        let a = if let Some(a) = self.try_replace_substs_in_root(a) {
            a
        } else {
            return true;
        };

        let b = if let Some(b) = self.try_replace_substs_in_root(b) {
            b
        } else {
            return true;
        };

        let a_root = a.root(self.tcx);
        let b_root = b.root(self.tcx);
        debug!(?a_root, ?b_root);

        match (a_root, b_root) {
            (Node::Leaf(a_ct), Node::Leaf(b_ct)) => {
                let a_ct = a_ct.eval(self.tcx, self.param_env);
                debug!("a_ct evaluated: {:?}", a_ct);
                let b_ct = b_ct.eval(self.tcx, self.param_env);
                debug!("b_ct evaluated: {:?}", b_ct);

                if a_ct.ty() != b_ct.ty() {
                    return false;
                }

                match (a_ct.kind(), b_ct.kind()) {
                    // We can just unify errors with everything to reduce the amount of
                    // emitted errors here.
                    (ty::ConstKind::Error(_), _) | (_, ty::ConstKind::Error(_)) => true,
                    (ty::ConstKind::Param(a_param), ty::ConstKind::Param(b_param)) => {
                        a_param == b_param
                    }
                    (ty::ConstKind::Value(a_val), ty::ConstKind::Value(b_val)) => a_val == b_val,
                    // If we have `fn a<const N: usize>() -> [u8; N + 1]` and `fn b<const M: usize>() -> [u8; 1 + M]`
                    // we do not want to use `assert_eq!(a(), b())` to infer that `N` and `M` have to be `1`. This
                    // means that we only allow inference variables if they are equal.
                    (ty::ConstKind::Infer(a_val), ty::ConstKind::Infer(b_val)) => a_val == b_val,
                    // We expand generic anonymous constants at the start of this function, so this
                    // branch should only be taking when dealing with associated constants, at
                    // which point directly comparing them seems like the desired behavior.
                    //
                    // FIXME(generic_const_exprs): This isn't actually the case.
                    // We also take this branch for concrete anonymous constants and
                    // expand generic anonymous constants with concrete substs.
                    (ty::ConstKind::Unevaluated(a_uv), ty::ConstKind::Unevaluated(b_uv)) => {
                        a_uv == b_uv
                    }
                    // FIXME(generic_const_exprs): We may want to either actually try
                    // to evaluate `a_ct` and `b_ct` if they are are fully concrete or something like
                    // this, for now we just return false here.
                    _ => false,
                }
            }
            (Node::Binop(a_op, al, ar), Node::Binop(b_op, bl, br)) if a_op == b_op => {
                self.try_unify(a.subtree(al), b.subtree(bl))
                    && self.try_unify(a.subtree(ar), b.subtree(br))
            }
            (Node::UnaryOp(a_op, av), Node::UnaryOp(b_op, bv)) if a_op == b_op => {
                self.try_unify(a.subtree(av), b.subtree(bv))
            }
            (Node::FunctionCall(a_f, a_args), Node::FunctionCall(b_f, b_args))
                if a_args.len() == b_args.len() =>
            {
                self.try_unify(a.subtree(a_f), b.subtree(b_f))
                    && iter::zip(a_args, b_args)
                        .all(|(&an, &bn)| self.try_unify(a.subtree(an), b.subtree(bn)))
            }
            (Node::Cast(a_kind, a_operand, a_ty), Node::Cast(b_kind, b_operand, b_ty))
                if (a_ty == b_ty) && (a_kind == b_kind) =>
            {
                self.try_unify(a.subtree(a_operand), b.subtree(b_operand))
            }
            // use this over `_ => false` to make adding variants to `Node` less error prone
            (Node::Cast(..), _)
            | (Node::FunctionCall(..), _)
            | (Node::UnaryOp(..), _)
            | (Node::Binop(..), _)
            | (Node::Leaf(..), _) => false,
        }
    }
}

#[instrument(skip(tcx), level = "debug")]
pub fn try_unify_abstract_consts<'tcx>(
    tcx: TyCtxt<'tcx>,
    (a, b): (ty::UnevaluatedConst<'tcx>, ty::UnevaluatedConst<'tcx>),
    param_env: ty::ParamEnv<'tcx>,
) -> bool {
    (|| {
        if let Some(a) = AbstractConst::new(tcx, a)? {
            if let Some(b) = AbstractConst::new(tcx, b)? {
                let const_unify_ctxt = ConstUnifyCtxt { tcx, param_env };
                return Ok(const_unify_ctxt.try_unify(a, b));
            }
        }

        Ok(false)
    })()
    .unwrap_or_else(|_: ErrorGuaranteed| true)
    // FIXME(generic_const_exprs): We should instead have this
    // method return the resulting `ty::Const` and return `ConstKind::Error`
    // on `ErrorGuaranteed`.
}

/// Check if a given constant can be evaluated.
#[instrument(skip(infcx), level = "debug")]
pub fn is_const_evaluatable<'cx, 'tcx>(
    infcx: &InferCtxt<'cx, 'tcx>,
    uv: ty::UnevaluatedConst<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    span: Span,
) -> Result<(), NotConstEvaluatable> {
    let tcx = infcx.tcx;

    if tcx.features().generic_const_exprs {
        if let Some(ct) = AbstractConst::new(tcx, uv)? {
            if satisfied_from_param_env(tcx, ct, param_env)? {
                return Ok(());
            }
            match ct.unify_failure_kind(tcx) {
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
        let concrete = infcx.const_eval_resolve(param_env, uv, Some(span));
        match concrete {
            Err(ErrorHandled::TooGeneric) => {
                Err(NotConstEvaluatable::Error(infcx.tcx.sess.delay_span_bug(
                    span,
                    format!("Missing value for constant, but no error reported?"),
                )))
            }
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
        let concrete = infcx.const_eval_resolve(param_env, uv, Some(span));

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

            Err(ErrorHandled::TooGeneric) => {
                let err = if uv.has_infer_types_or_consts() {
                    NotConstEvaluatable::MentionsInfer
                } else if uv.has_param_types_or_consts() {
                    NotConstEvaluatable::MentionsParam
                } else {
                    let guar = infcx.tcx.sess.delay_span_bug(span, format!("Missing value for constant, but no error reported?"));
                    NotConstEvaluatable::Error(guar)
                };

                Err(err)
            },
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
                            "cannot use constants which depend on generic parameters in types",
                            |err| err
                        )
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
