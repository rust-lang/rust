use super::callee::DeferredCallResolution;

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sync::Lrc;
use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::HirIdMap;
use rustc_infer::infer;
use rustc_infer::infer::{InferCtxt, InferOk, TyCtxtInferExt};
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::visit::TypeVisitable;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::def_id::LocalDefIdMap;
use rustc_span::{self, Span};
use rustc_trait_selection::infer::InferCtxtExt as _;
use rustc_trait_selection::traits::{
    self, ObligationCause, ObligationCtxt, TraitEngine, TraitEngineExt as _,
};

use std::cell::RefCell;
use std::ops::Deref;

/// Closures defined within the function. For example:
/// ```ignore (illustrative)
/// fn foo() {
///     bar(move|| { ... })
/// }
/// ```
/// Here, the function `foo()` and the closure passed to
/// `bar()` will each have their own `FnCtxt`, but they will
/// share the inherited fields.
pub struct Inherited<'a, 'tcx> {
    pub(super) infcx: InferCtxt<'a, 'tcx>,

    pub(super) typeck_results: &'a RefCell<ty::TypeckResults<'tcx>>,

    pub(super) locals: RefCell<HirIdMap<super::LocalTy<'tcx>>>,

    pub(super) fulfillment_cx: RefCell<Box<dyn TraitEngine<'tcx>>>,

    // When we process a call like `c()` where `c` is a closure type,
    // we may not have decided yet whether `c` is a `Fn`, `FnMut`, or
    // `FnOnce` closure. In that case, we defer full resolution of the
    // call until upvar inference can kick in and make the
    // decision. We keep these deferred resolutions grouped by the
    // def-id of the closure, so that once we decide, we can easily go
    // back and process them.
    pub(super) deferred_call_resolutions: RefCell<LocalDefIdMap<Vec<DeferredCallResolution<'tcx>>>>,

    pub(super) deferred_cast_checks: RefCell<Vec<super::cast::CastCheck<'tcx>>>,

    pub(super) deferred_transmute_checks: RefCell<Vec<(Ty<'tcx>, Ty<'tcx>, Span)>>,

    pub(super) deferred_asm_checks: RefCell<Vec<(&'tcx hir::InlineAsm<'tcx>, hir::HirId)>>,

    pub(super) deferred_generator_interiors:
        RefCell<Vec<(hir::BodyId, Ty<'tcx>, hir::GeneratorKind)>>,

    pub(super) body_id: Option<hir::BodyId>,

    /// Whenever we introduce an adjustment from `!` into a type variable,
    /// we record that type variable here. This is later used to inform
    /// fallback. See the `fallback` module for details.
    pub(super) diverging_type_vars: RefCell<FxHashSet<Ty<'tcx>>>,
}

impl<'a, 'tcx> Deref for Inherited<'a, 'tcx> {
    type Target = InferCtxt<'a, 'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.infcx
    }
}

/// A temporary returned by `Inherited::build(...)`. This is necessary
/// for multiple `InferCtxt` to share the same `in_progress_typeck_results`
/// without using `Rc` or something similar.
pub struct InheritedBuilder<'tcx> {
    infcx: infer::InferCtxtBuilder<'tcx>,
    def_id: LocalDefId,
}

impl<'tcx> Inherited<'_, 'tcx> {
    pub fn build(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> InheritedBuilder<'tcx> {
        let hir_owner = tcx.hir().local_def_id_to_hir_id(def_id).owner;

        InheritedBuilder {
            infcx: tcx
                .infer_ctxt()
                .ignoring_regions()
                .with_fresh_in_progress_typeck_results(hir_owner)
                .with_normalize_fn_sig_for_diagnostic(Lrc::new(move |infcx, fn_sig| {
                    if fn_sig.has_escaping_bound_vars() {
                        return fn_sig;
                    }
                    infcx.probe(|_| {
                        let ocx = ObligationCtxt::new_in_snapshot(infcx);
                        let normalized_fn_sig = ocx.normalize(
                            ObligationCause::dummy(),
                            // FIXME(compiler-errors): This is probably not the right param-env...
                            infcx.tcx.param_env(def_id),
                            fn_sig,
                        );
                        if ocx.select_all_or_error().is_empty() {
                            let normalized_fn_sig =
                                infcx.resolve_vars_if_possible(normalized_fn_sig);
                            if !normalized_fn_sig.needs_infer() {
                                return normalized_fn_sig;
                            }
                        }
                        fn_sig
                    })
                })),
            def_id,
        }
    }
}

impl<'tcx> InheritedBuilder<'tcx> {
    pub fn enter<F, R>(&mut self, f: F) -> R
    where
        F: for<'a> FnOnce(Inherited<'a, 'tcx>) -> R,
    {
        let def_id = self.def_id;
        self.infcx.enter(|infcx| f(Inherited::new(infcx, def_id)))
    }
}

impl<'a, 'tcx> Inherited<'a, 'tcx> {
    fn new(infcx: InferCtxt<'a, 'tcx>, def_id: LocalDefId) -> Self {
        let tcx = infcx.tcx;
        let body_id = tcx.hir().maybe_body_owned_by(def_id);
        let typeck_results =
            infcx.in_progress_typeck_results.expect("building `FnCtxt` without typeck results");

        Inherited {
            typeck_results,
            infcx,
            fulfillment_cx: RefCell::new(<dyn TraitEngine<'_>>::new(tcx)),
            locals: RefCell::new(Default::default()),
            deferred_call_resolutions: RefCell::new(Default::default()),
            deferred_cast_checks: RefCell::new(Vec::new()),
            deferred_transmute_checks: RefCell::new(Vec::new()),
            deferred_asm_checks: RefCell::new(Vec::new()),
            deferred_generator_interiors: RefCell::new(Vec::new()),
            diverging_type_vars: RefCell::new(Default::default()),
            body_id,
        }
    }

    #[instrument(level = "debug", skip(self))]
    pub(super) fn register_predicate(&self, obligation: traits::PredicateObligation<'tcx>) {
        if obligation.has_escaping_bound_vars() {
            span_bug!(obligation.cause.span, "escaping bound vars in predicate {:?}", obligation);
        }
        self.fulfillment_cx.borrow_mut().register_predicate_obligation(self, obligation);
    }

    pub(super) fn register_predicates<I>(&self, obligations: I)
    where
        I: IntoIterator<Item = traits::PredicateObligation<'tcx>>,
    {
        for obligation in obligations {
            self.register_predicate(obligation);
        }
    }

    pub(super) fn register_infer_ok_obligations<T>(&self, infer_ok: InferOk<'tcx, T>) -> T {
        self.register_predicates(infer_ok.obligations);
        infer_ok.value
    }

    pub(super) fn normalize_associated_types_in<T>(
        &self,
        span: Span,
        body_id: hir::HirId,
        param_env: ty::ParamEnv<'tcx>,
        value: T,
    ) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        self.normalize_associated_types_in_with_cause(
            ObligationCause::misc(span, body_id),
            param_env,
            value,
        )
    }

    pub(super) fn normalize_associated_types_in_with_cause<T>(
        &self,
        cause: ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        value: T,
    ) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        let ok = self.partially_normalize_associated_types_in(cause, param_env, value);
        debug!(?ok);
        self.register_infer_ok_obligations(ok)
    }
}
