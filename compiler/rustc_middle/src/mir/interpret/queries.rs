use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_session::lint;
use rustc_span::{DUMMY_SP, Span};
use tracing::{debug, instrument};

use super::{
    ErrorHandled, EvalToAllocationRawResult, EvalToConstValueResult, GlobalId, ReportedErrorInfo,
};
use crate::mir::interpret::ValTreeCreationError;
use crate::ty::{self, ConstToValTreeResult, GenericArgs, TyCtxt, TypeVisitableExt};
use crate::{error, mir};

impl<'tcx> TyCtxt<'tcx> {
    /// Evaluates a constant without providing any generic parameters. This is useful to evaluate consts
    /// that can't take any generic arguments like const items or enum discriminants. If a
    /// generic parameter is used within the constant `ErrorHandled::TooGeneric` will be returned.
    #[instrument(skip(self), level = "debug")]
    pub fn const_eval_poly(self, def_id: DefId) -> EvalToConstValueResult<'tcx> {
        // In some situations def_id will have generic parameters within scope, but they aren't allowed
        // to be used. So we can't use `Instance::mono`, instead we feed unresolved generic parameters
        // into `const_eval` which will return `ErrorHandled::TooGeneric` if any of them are
        // encountered.
        let args = GenericArgs::identity_for_item(self, def_id);
        let instance = ty::Instance::new_raw(def_id, args);
        let cid = GlobalId { instance, promoted: None };
        let typing_env = ty::TypingEnv::post_analysis(self, def_id);
        self.const_eval_global_id(typing_env, cid, DUMMY_SP)
    }

    /// Evaluates a constant without providing any generic parameters. This is useful to evaluate consts
    /// that can't take any generic arguments like const items or enum discriminants. If a
    /// generic parameter is used within the constant `ErrorHandled::TooGeneric` will be returned.
    #[instrument(skip(self), level = "debug")]
    pub fn const_eval_poly_to_alloc(self, def_id: DefId) -> EvalToAllocationRawResult<'tcx> {
        // In some situations def_id will have generic parameters within scope, but they aren't allowed
        // to be used. So we can't use `Instance::mono`, instead we feed unresolved generic parameters
        // into `const_eval` which will return `ErrorHandled::TooGeneric` if any of them are
        // encountered.
        let args = GenericArgs::identity_for_item(self, def_id);
        let instance = ty::Instance::new_raw(def_id, args);
        let cid = GlobalId { instance, promoted: None };
        let typing_env = ty::TypingEnv::post_analysis(self, def_id);
        let inputs = self.erase_regions(typing_env.as_query_input(cid));
        self.eval_to_allocation_raw(inputs)
    }

    /// Resolves and evaluates a constant.
    ///
    /// The constant can be located on a trait like `<A as B>::C`, in which case the given
    /// generic parameters and environment are used to resolve the constant. Alternatively if the
    /// constant has generic parameters in scope the generic parameters are used to evaluate the value of
    /// the constant. For example in `fn foo<T>() { let _ = [0; bar::<T>()]; }` the repeat count
    /// constant `bar::<T>()` requires a instantiation for `T`, if the instantiation for `T` is still
    /// too generic for the constant to be evaluated then `Err(ErrorHandled::TooGeneric)` is
    /// returned.
    #[instrument(level = "debug", skip(self))]
    pub fn const_eval_resolve(
        self,
        typing_env: ty::TypingEnv<'tcx>,
        ct: mir::UnevaluatedConst<'tcx>,
        span: Span,
    ) -> EvalToConstValueResult<'tcx> {
        // Cannot resolve `Unevaluated` constants that contain inference
        // variables. We reject those here since `resolve`
        // would fail otherwise.
        //
        // When trying to evaluate constants containing inference variables,
        // use `Infcx::const_eval_resolve` instead.
        if ct.args.has_non_region_infer() {
            bug!("did not expect inference variables here");
        }

        // FIXME: maybe have a separate version for resolving mir::UnevaluatedConst?
        match ty::Instance::try_resolve(self, typing_env, ct.def, ct.args) {
            Ok(Some(instance)) => {
                let cid = GlobalId { instance, promoted: ct.promoted };
                self.const_eval_global_id(typing_env, cid, span)
            }
            // For errors during resolution, we deliberately do not point at the usage site of the constant,
            // since for these errors the place the constant is used shouldn't matter.
            Ok(None) => Err(ErrorHandled::TooGeneric(DUMMY_SP)),
            Err(err) => {
                Err(ErrorHandled::Reported(ReportedErrorInfo::non_const_eval_error(err), DUMMY_SP))
            }
        }
    }

    #[instrument(level = "debug", skip(self))]
    pub fn const_eval_resolve_for_typeck(
        self,
        typing_env: ty::TypingEnv<'tcx>,
        ct: ty::UnevaluatedConst<'tcx>,
        span: Span,
    ) -> ConstToValTreeResult<'tcx> {
        // Cannot resolve `Unevaluated` constants that contain inference
        // variables. We reject those here since `resolve`
        // would fail otherwise.
        //
        // When trying to evaluate constants containing inference variables,
        // use `Infcx::const_eval_resolve` instead.
        if ct.args.has_non_region_infer() {
            bug!("did not expect inference variables here");
        }

        let cid = match ty::Instance::try_resolve(self, typing_env, ct.def, ct.args) {
            Ok(Some(instance)) => GlobalId { instance, promoted: None },
            // For errors during resolution, we deliberately do not point at the usage site of the constant,
            // since for these errors the place the constant is used shouldn't matter.
            Ok(None) => return Err(ErrorHandled::TooGeneric(DUMMY_SP).into()),
            Err(err) => {
                return Err(ErrorHandled::Reported(
                    ReportedErrorInfo::non_const_eval_error(err),
                    DUMMY_SP,
                )
                .into());
            }
        };

        self.const_eval_global_id_for_typeck(typing_env, cid, span).inspect(|_| {
            // We are emitting the lint here instead of in `is_const_evaluatable`
            // as we normalize obligations before checking them, and normalization
            // uses this function to evaluate this constant.
            //
            // @lcnr believes that successfully evaluating even though there are
            // used generic parameters is a bug of evaluation, so checking for it
            // here does feel somewhat sensible.
            if !self.features().generic_const_exprs()
                && ct.args.has_non_region_param()
                // We only FCW for anon consts as repeat expr counts with anon consts are the only place
                // that we have a back compat hack for. We don't need to check this is a const argument
                // as only anon consts as const args should get evaluated "for the type system".
                //
                // If we don't *only* FCW anon consts we can wind up incorrectly FCW'ing uses of assoc
                // consts in pattern positions. #140447
                && self.def_kind(cid.instance.def_id()) == DefKind::AnonConst
            {
                let mir_body = self.mir_for_ctfe(cid.instance.def_id());
                if mir_body.is_polymorphic {
                    let Some(local_def_id) = ct.def.as_local() else { return };
                    self.node_span_lint(
                        lint::builtin::CONST_EVALUATABLE_UNCHECKED,
                        self.local_def_id_to_hir_id(local_def_id),
                        self.def_span(ct.def),
                        |lint| {
                            lint.primary_message(
                                "cannot use constants which depend on generic parameters in types",
                            );
                        },
                    )
                }
            }
        })
    }

    pub fn const_eval_instance(
        self,
        typing_env: ty::TypingEnv<'tcx>,
        instance: ty::Instance<'tcx>,
        span: Span,
    ) -> EvalToConstValueResult<'tcx> {
        self.const_eval_global_id(typing_env, GlobalId { instance, promoted: None }, span)
    }

    /// Evaluate a constant to a `ConstValue`.
    #[instrument(skip(self), level = "debug")]
    pub fn const_eval_global_id(
        self,
        typing_env: ty::TypingEnv<'tcx>,
        cid: GlobalId<'tcx>,
        span: Span,
    ) -> EvalToConstValueResult<'tcx> {
        // Const-eval shouldn't depend on lifetimes at all, so we can erase them, which should
        // improve caching of queries.
        let inputs =
            self.erase_regions(typing_env.with_post_analysis_normalized(self).as_query_input(cid));
        if !span.is_dummy() {
            // The query doesn't know where it is being invoked, so we need to fix the span.
            self.at(span).eval_to_const_value_raw(inputs).map_err(|e| e.with_span(span))
        } else {
            self.eval_to_const_value_raw(inputs)
        }
    }

    /// Evaluate a constant to a type-level constant.
    #[instrument(skip(self), level = "debug")]
    pub fn const_eval_global_id_for_typeck(
        self,
        typing_env: ty::TypingEnv<'tcx>,
        cid: GlobalId<'tcx>,
        span: Span,
    ) -> ConstToValTreeResult<'tcx> {
        // Const-eval shouldn't depend on lifetimes at all, so we can erase them, which should
        // improve caching of queries.
        let inputs =
            self.erase_regions(typing_env.with_post_analysis_normalized(self).as_query_input(cid));
        debug!(?inputs);
        let res = if !span.is_dummy() {
            // The query doesn't know where it is being invoked, so we need to fix the span.
            self.at(span).eval_to_valtree(inputs).map_err(|e| e.with_span(span))
        } else {
            self.eval_to_valtree(inputs)
        };
        match res {
            Ok(valtree) => Ok(Ok(valtree)),
            Err(err) => {
                match err {
                    // Let the caller decide how to handle this.
                    ValTreeCreationError::NonSupportedType(ty) => Ok(Err(ty)),
                    // Report the others.
                    ValTreeCreationError::NodesOverflow => {
                        let handled = self.dcx().emit_err(error::MaxNumNodesInValtree {
                            span,
                            global_const_id: cid.display(self),
                        });
                        Err(ReportedErrorInfo::allowed_in_infallible(handled).into())
                    }
                    ValTreeCreationError::InvalidConst => {
                        let handled = self.dcx().emit_err(error::InvalidConstInValtree {
                            span,
                            global_const_id: cid.display(self),
                        });
                        Err(ReportedErrorInfo::allowed_in_infallible(handled).into())
                    }
                    ValTreeCreationError::ErrorHandled(handled) => Err(handled),
                }
            }
        }
    }
}
