use super::callee::DeferredCallResolution;
use super::MaybeInProgressTables;

use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::def_id::{DefIdMap, LocalDefId};
use rustc_hir::HirIdMap;
use rustc_infer::infer;
use rustc_infer::infer::{InferCtxt, InferOk, TyCtxtInferExt};
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::{self, Span};
use rustc_trait_selection::infer::InferCtxtExt as _;
use rustc_trait_selection::traits::{self, ObligationCause, TraitEngine, TraitEngineExt};

use std::cell::RefCell;
use std::ops::Deref;

/// Closures defined within the function. For example:
///
///     fn foo() {
///         bar(move|| { ... })
///     }
///
/// Here, the function `foo()` and the closure passed to
/// `bar()` will each have their own `FnCtxt`, but they will
/// share the inherited fields.
pub struct Inherited<'a, 'tcx> {
    pub(super) infcx: InferCtxt<'a, 'tcx>,

    pub(super) typeck_results: super::MaybeInProgressTables<'a, 'tcx>,

    pub(super) locals: RefCell<HirIdMap<super::LocalTy<'tcx>>>,

    pub(super) fulfillment_cx: RefCell<Box<dyn TraitEngine<'tcx>>>,

    // Some additional `Sized` obligations badly affect type inference.
    // These obligations are added in a later stage of typeck.
    pub(super) deferred_sized_obligations:
        RefCell<Vec<(Ty<'tcx>, Span, traits::ObligationCauseCode<'tcx>)>>,

    // When we process a call like `c()` where `c` is a closure type,
    // we may not have decided yet whether `c` is a `Fn`, `FnMut`, or
    // `FnOnce` closure. In that case, we defer full resolution of the
    // call until upvar inference can kick in and make the
    // decision. We keep these deferred resolutions grouped by the
    // def-id of the closure, so that once we decide, we can easily go
    // back and process them.
    pub(super) deferred_call_resolutions: RefCell<DefIdMap<Vec<DeferredCallResolution<'tcx>>>>,

    pub(super) deferred_cast_checks: RefCell<Vec<super::cast::CastCheck<'tcx>>>,

    pub(super) deferred_generator_interiors:
        RefCell<Vec<(hir::BodyId, Ty<'tcx>, hir::GeneratorKind)>>,

    /// Reports whether this is in a const context.
    pub(super) constness: hir::Constness,

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

/// Helper type of a temporary returned by `Inherited::build(...)`.
/// Necessary because we can't write the following bound:
/// `F: for<'b, 'tcx> where 'tcx FnOnce(Inherited<'b, 'tcx>)`.
pub struct InheritedBuilder<'tcx> {
    infcx: infer::InferCtxtBuilder<'tcx>,
    def_id: LocalDefId,
}

impl Inherited<'_, 'tcx> {
    pub fn build(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> InheritedBuilder<'tcx> {
        let hir_owner = tcx.hir().local_def_id_to_hir_id(def_id).owner;

        InheritedBuilder {
            infcx: tcx.infer_ctxt().with_fresh_in_progress_typeck_results(hir_owner),
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

impl Inherited<'a, 'tcx> {
    pub(super) fn new(infcx: InferCtxt<'a, 'tcx>, def_id: LocalDefId) -> Self {
        let tcx = infcx.tcx;
        let item_id = tcx.hir().local_def_id_to_hir_id(def_id);
        Self::with_constness(infcx, def_id, tcx.hir().get(item_id).constness_for_typeck())
    }

    pub(super) fn with_constness(
        infcx: InferCtxt<'a, 'tcx>,
        def_id: LocalDefId,
        constness: hir::Constness,
    ) -> Self {
        let tcx = infcx.tcx;
        let item_id = tcx.hir().local_def_id_to_hir_id(def_id);
        let body_id = tcx.hir().maybe_body_owned_by(item_id);

        Inherited {
            typeck_results: MaybeInProgressTables {
                maybe_typeck_results: infcx.in_progress_typeck_results,
            },
            infcx,
            fulfillment_cx: RefCell::new(<dyn TraitEngine<'_>>::new(tcx)),
            locals: RefCell::new(Default::default()),
            deferred_sized_obligations: RefCell::new(Vec::new()),
            deferred_call_resolutions: RefCell::new(Default::default()),
            deferred_cast_checks: RefCell::new(Vec::new()),
            deferred_generator_interiors: RefCell::new(Vec::new()),
            diverging_type_vars: RefCell::new(Default::default()),
            constness,
            body_id,
        }
    }

    pub(super) fn register_predicate(&self, obligation: traits::PredicateObligation<'tcx>) {
        debug!("register_predicate({:?})", obligation);
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
