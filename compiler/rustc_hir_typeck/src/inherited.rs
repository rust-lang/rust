use super::callee::DeferredCallResolution;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::HirIdMap;
use rustc_infer::infer;
use rustc_infer::infer::{DefiningAnchor, InferCtxt, InferOk, TyCtxtInferExt};
use rustc_middle::ty::visit::TypeVisitableExt;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::def_id::LocalDefIdMap;
use rustc_span::{self, Span};
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use rustc_trait_selection::traits::{self, PredicateObligation, TraitEngine, TraitEngineExt as _};

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
pub struct Inherited<'tcx> {
    pub(super) infcx: InferCtxt<'tcx>,

    pub(super) typeck_results: RefCell<ty::TypeckResults<'tcx>>,

    pub(super) locals: RefCell<HirIdMap<super::LocalTy<'tcx>>>,

    pub(super) fulfillment_cx: RefCell<Box<dyn TraitEngine<'tcx>>>,

    /// Some additional `Sized` obligations badly affect type inference.
    /// These obligations are added in a later stage of typeck.
    /// Removing these may also cause additional complications, see #101066.
    pub(super) deferred_sized_obligations:
        RefCell<Vec<(Ty<'tcx>, Span, traits::ObligationCauseCode<'tcx>)>>,

    /// When we process a call like `c()` where `c` is a closure type,
    /// we may not have decided yet whether `c` is a `Fn`, `FnMut`, or
    /// `FnOnce` closure. In that case, we defer full resolution of the
    /// call until upvar inference can kick in and make the
    /// decision. We keep these deferred resolutions grouped by the
    /// def-id of the closure, so that once we decide, we can easily go
    /// back and process them.
    pub(super) deferred_call_resolutions: RefCell<LocalDefIdMap<Vec<DeferredCallResolution<'tcx>>>>,

    pub(super) deferred_cast_checks: RefCell<Vec<super::cast::CastCheck<'tcx>>>,

    pub(super) deferred_transmute_checks: RefCell<Vec<(Ty<'tcx>, Ty<'tcx>, hir::HirId)>>,

    pub(super) deferred_asm_checks: RefCell<Vec<(&'tcx hir::InlineAsm<'tcx>, hir::HirId)>>,

    pub(super) deferred_generator_interiors:
        RefCell<Vec<(LocalDefId, hir::BodyId, Ty<'tcx>, hir::GeneratorKind)>>,

    /// Whenever we introduce an adjustment from `!` into a type variable,
    /// we record that type variable here. This is later used to inform
    /// fallback. See the `fallback` module for details.
    pub(super) diverging_type_vars: RefCell<FxHashSet<Ty<'tcx>>>,

    pub(super) infer_var_info: RefCell<FxHashMap<ty::TyVid, ty::InferVarInfo>>,
}

impl<'tcx> Deref for Inherited<'tcx> {
    type Target = InferCtxt<'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.infcx
    }
}

/// A temporary returned by `Inherited::build(...)`. This is necessary
/// for multiple `InferCtxt` to share the same `typeck_results`
/// without using `Rc` or something similar.
pub struct InheritedBuilder<'tcx> {
    infcx: infer::InferCtxtBuilder<'tcx>,
    typeck_results: RefCell<ty::TypeckResults<'tcx>>,
}

impl<'tcx> Inherited<'tcx> {
    pub fn build(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> InheritedBuilder<'tcx> {
        let hir_owner = tcx.hir().local_def_id_to_hir_id(def_id).owner;

        InheritedBuilder {
            infcx: tcx
                .infer_ctxt()
                .ignoring_regions()
                .with_opaque_type_inference(DefiningAnchor::Bind(hir_owner.def_id)),
            typeck_results: RefCell::new(ty::TypeckResults::new(hir_owner)),
        }
    }
}

impl<'tcx> InheritedBuilder<'tcx> {
    pub fn enter<F, R>(mut self, f: F) -> R
    where
        F: FnOnce(&Inherited<'tcx>) -> R,
    {
        f(&Inherited::new(self.infcx.build(), self.typeck_results))
    }
}

impl<'tcx> Inherited<'tcx> {
    fn new(infcx: InferCtxt<'tcx>, typeck_results: RefCell<ty::TypeckResults<'tcx>>) -> Self {
        let tcx = infcx.tcx;

        Inherited {
            typeck_results,
            infcx,
            fulfillment_cx: RefCell::new(<dyn TraitEngine<'_>>::new(tcx)),
            locals: RefCell::new(Default::default()),
            deferred_sized_obligations: RefCell::new(Vec::new()),
            deferred_call_resolutions: RefCell::new(Default::default()),
            deferred_cast_checks: RefCell::new(Vec::new()),
            deferred_transmute_checks: RefCell::new(Vec::new()),
            deferred_asm_checks: RefCell::new(Vec::new()),
            deferred_generator_interiors: RefCell::new(Vec::new()),
            diverging_type_vars: RefCell::new(Default::default()),
            infer_var_info: RefCell::new(Default::default()),
        }
    }

    #[instrument(level = "debug", skip(self))]
    pub(super) fn register_predicate(&self, obligation: traits::PredicateObligation<'tcx>) {
        if obligation.has_escaping_bound_vars() {
            span_bug!(obligation.cause.span, "escaping bound vars in predicate {:?}", obligation);
        }

        self.update_infer_var_info(&obligation);

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

    pub fn update_infer_var_info(&self, obligation: &PredicateObligation<'tcx>) {
        let infer_var_info = &mut self.infer_var_info.borrow_mut();

        // (*) binder skipped
        if let ty::PredicateKind::Clause(ty::Clause::Trait(tpred)) = obligation.predicate.kind().skip_binder()
            && let Some(ty) = self.shallow_resolve(tpred.self_ty()).ty_vid().map(|t| self.root_var(t))
            && self.tcx.lang_items().sized_trait().map_or(false, |st| st != tpred.trait_ref.def_id)
        {
            let new_self_ty = self.tcx.types.unit;

            // Then construct a new obligation with Self = () added
            // to the ParamEnv, and see if it holds.
            let o = obligation.with(self.tcx,
                obligation
                    .predicate
                    .kind()
                    .rebind(
                        // (*) binder moved here
                        ty::PredicateKind::Clause(ty::Clause::Trait(tpred.with_self_ty(self.tcx, new_self_ty)))
                    ),
            );
            // Don't report overflow errors. Otherwise equivalent to may_hold.
            if let Ok(result) = self.probe(|_| self.evaluate_obligation(&o)) && result.may_apply() {
                infer_var_info.entry(ty).or_default().self_in_trait = true;
            }
        }

        if let ty::PredicateKind::Clause(ty::Clause::Projection(predicate)) =
            obligation.predicate.kind().skip_binder()
        {
            // If the projection predicate (Foo::Bar == X) has X as a non-TyVid,
            // we need to make it into one.
            if let Some(vid) = predicate.term.ty().and_then(|ty| ty.ty_vid()) {
                debug!("infer_var_info: {:?}.output = true", vid);
                infer_var_info.entry(vid).or_default().output = true;
            }
        }
    }
}
