use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_lint_defs::builtin::CONST_EVALUATABLE_UNCHECKED;
use rustc_span::{DUMMY_SP, Span, bug};
use tracing::{debug, instrument};

use super::{
    ErrorHandled, EvalToAllocationRawResult, EvalToConstValueResult, GlobalId, ReportedErrorInfo,
};
use crate::mir::interpret::ValTreeCreationError;
use crate::ty::{self, ConstToValTreeResult, GenericArgs, TyCtxt, TypeVisitableExt};
use crate::{diagnostics, mir};

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
        let inputs = self.erase_and_anonymize_regions(typing_env.as_query_input(cid));
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
                if let Some(direct_eval) =
                    self.const_eval_if_direct_const(typing_env, ct, instance, span)
                {
                    direct_eval
                } else {
                    let cid = GlobalId { instance, promoted: ct.promoted };
                    self.const_eval_global_id(typing_env, cid, span)
                }
            }
            // For errors during resolution, we deliberately do not point at the usage site of the constant,
            // since for these errors the place the constant is used shouldn't matter.
            Ok(None) => Err(ErrorHandled::TooGeneric(DUMMY_SP)),
            Err(err) => {
                Err(ErrorHandled::Reported(ReportedErrorInfo::non_const_eval_error(err), DUMMY_SP))
            }
        }
    }

    /// Checks if `resolved` is a directly represented const (i.e. `gca!` const) and if so,
    /// evaluates it.
    fn const_eval_if_direct_const(
        self,
        typing_env: ty::TypingEnv<'tcx>,
        unresolved: mir::UnevaluatedConst<'tcx>,
        resolved: ty::Instance<'tcx>,
        span: Span,
    ) -> Option<EvalToConstValueResult<'tcx>> {
        let ty::InstanceKind::Item(def_id) = resolved.def else {
            return None;
        };
        let def_kind = self.def_kind(def_id);
        let (DefKind::Const | DefKind::AssocConst) = def_kind else {
            return None;
        };
        if self.const_of_item(def_id).is_none() {
            return None;
        };
        let alias_const = if def_kind == DefKind::Const {
            let kind = ty::AliasConstKind::Free { def_id };
            ty::AliasConst::new(self, kind, resolved.args)
        } else {
            if let DefKind::Impl { of_trait: false } = self.def_kind(self.parent(def_id)) {
                let kind = ty::AliasConstKind::InherentImpl { def_id };
                ty::AliasConst::new(self, kind, resolved.args)
            } else {
                // HACK(khyperia): the trait solver does not support a resolved impl
                // item def_id inside Projection, so pass in the unresolved impl, and
                // let the trait solver re-resolve it. This must always resolve to the
                // same impl.
                let kind = ty::AliasConstKind::Projection { def_id: unresolved.def };
                ty::AliasConst::new(self, kind, unresolved.args)
            }
        };
        Some(self.const_eval_direct_const(typing_env, alias_const, span))
    }

    /// `alias_const` *must* be a const item whose `const_of_item` returns `Some`
    fn const_eval_direct_const(
        self,
        typing_env: ty::TypingEnv<'tcx>,
        alias_const: ty::AliasConst<'tcx>,
        span: Span,
    ) -> EvalToConstValueResult<'tcx> {
        let value = ty::Const::new_alias(self, ty::IsRigid::No, alias_const);
        let unnormalized = ty::Unnormalized::new(value);
        let normalized = match self.try_normalize_erasing_regions(typing_env, unnormalized) {
            Ok(value) => value,
            Err(erroring_term) => {
                return match erroring_term.error_reported() {
                    Ok(()) => Err(ErrorHandled::TooGeneric(span)),
                    Err(guar) => Err(ErrorHandled::Reported(
                        ReportedErrorInfo::non_const_eval_error(guar),
                        span,
                    )),
                };
            }
        };
        if let Err(guar) = normalized.error_reported() {
            Err(ErrorHandled::Reported(ReportedErrorInfo::non_const_eval_error(guar), span))
        } else if let ty::ConstKind::Value(value) = normalized.kind() {
            Ok(self.valtree_to_const_val(value))
        } else {
            Err(ErrorHandled::TooGeneric(span))
        }
    }

    #[instrument(level = "debug", skip(self))]
    pub fn const_eval_resolve_for_typeck(
        self,
        typing_env: ty::TypingEnv<'tcx>,
        ct: ty::AliasConst<'tcx>,
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

        let def_id = match ct.kind {
            ty::AliasConstKind::InherentSelf { .. } => {
                bug!("got AliasConstKind::InherentSelf in const_eval_resolve_for_typeck")
            }
            ty::AliasConstKind::Projection { def_id }
            | ty::AliasConstKind::InherentImpl { def_id }
            | ty::AliasConstKind::Free { def_id }
            | ty::AliasConstKind::Anon { def_id } => def_id,
        };

        // We need to use post analysis mode for instance resolving to succeed.
        // Otherwise we often bail due to impl candidate not being final.
        // Revealing opaques here should be fine as we proved the trait ref
        // with rigid opaques in typeck.
        let ty::PseudoCanonicalInput { typing_env, value: args } =
            typing_env.with_post_analysis_normalized_invalidating_rigid_aliases(self, ct.args);

        let cid = match ty::Instance::try_resolve(self, typing_env, def_id, args) {
            Ok(Some(instance)) => GlobalId { instance, promoted: None },
            // For errors during resolution, we deliberately do not point at the usage site of the constant,
            // since for these errors the place the constant is used shouldn't matter.
            Ok(None) => return Err(ErrorHandled::TooGeneric(DUMMY_SP)),
            Err(err) => {
                return Err(ErrorHandled::Reported(
                    ReportedErrorInfo::non_const_eval_error(err),
                    DUMMY_SP,
                ));
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
            let def_id = cid.instance.def_id();
            if !self.features().generic_const_exprs()
                && args.has_non_region_param()
                // We only FCW for anon consts as repeat expr counts with anon consts are the only place
                // that we have a back compat hack for. We don't need to check this is a const argument
                // as only anon consts as const args should get evaluated "for the type system".
                //
                // If we don't *only* FCW anon consts we can wind up incorrectly FCW'ing uses of assoc
                // consts in pattern positions. #140447
                && self.def_kind(def_id) == DefKind::AnonConst
                && !self.is_trivial_const(def_id)
                && let Some(local_def_id) = def_id.as_local()
            {
                let mir_body = self.mir_for_ctfe(def_id);
                if mir_body.is_polymorphic {
                    self.emit_node_span_lint(
                        CONST_EVALUATABLE_UNCHECKED,
                        self.local_def_id_to_hir_id(local_def_id),
                        self.def_span(def_id),
                        rustc_errors::DiagDecorator(|lint| {
                            lint.primary_message(
                                "cannot use constants which depend on generic parameters in types",
                            );
                        }),
                    )
                }
            }
        })
    }

    /// NOTE: The only callsite of this function is rustc_public.
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
        let inputs = self.erase_and_anonymize_regions(
            typing_env.with_post_analysis_normalized_invalidating_rigid_aliases(self, cid),
        );
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
        let inputs = self.erase_and_anonymize_regions(
            typing_env.with_post_analysis_normalized_invalidating_rigid_aliases(self, cid),
        );
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
                        let handled = self.dcx().emit_err(diagnostics::MaxNumNodesInValtree {
                            span,
                            global_const_id: cid.display(self),
                        });
                        Err(ReportedErrorInfo::allowed_in_infallible(handled).into())
                    }
                    ValTreeCreationError::InvalidConst => {
                        let handled = self.dcx().emit_err(diagnostics::InvalidConstInValtree {
                            span,
                            global_const_id: cid.display(self),
                        });
                        Err(ReportedErrorInfo::allowed_in_infallible(handled).into())
                    }
                    ValTreeCreationError::CyclicConst => {
                        let handled = self.dcx().emit_err(diagnostics::CyclicConstInValtree {
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
