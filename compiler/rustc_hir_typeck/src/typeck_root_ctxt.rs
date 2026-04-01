use std::cell::{Cell, RefCell};
use std::ops::Deref;

use rustc_data_structures::unord::UnordSet;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::{self as hir, HirId, HirIdMap};
use rustc_infer::infer::{InferCtxt, InferOk, OpaqueTypeStorageEntries, TyCtxtInferExt};
use rustc_middle::span_bug;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitableExt, TypingMode};
use rustc_span::Span;
use rustc_span::def_id::LocalDefIdMap;
use rustc_trait_selection::traits::{self, FulfillmentError, TraitEngine, TraitEngineExt as _};
use tracing::instrument;

use super::callee::DeferredCallResolution;

/// Data shared between a "typeck root" and its nested bodies,
/// e.g. closures defined within the function. For example:
/// ```ignore (illustrative)
/// fn foo() {
///     bar(move || { ... })
/// }
/// ```
/// Here, the function `foo()` and the closure passed to
/// `bar()` will each have their own `FnCtxt`, but they will
/// share the inference context, will process obligations together,
/// can access each other's local types (scoping permitted), etc.
pub(crate) struct TypeckRootCtxt<'tcx> {
    pub(super) infcx: InferCtxt<'tcx>,

    pub(super) typeck_results: RefCell<ty::TypeckResults<'tcx>>,

    pub(super) locals: RefCell<HirIdMap<Ty<'tcx>>>,

    pub(super) fulfillment_cx: RefCell<Box<dyn TraitEngine<'tcx, FulfillmentError<'tcx>>>>,

    // Used to detect opaque types uses added after we've already checked them.
    //
    // See [FnCtxt::detect_opaque_types_added_during_writeback] for more details.
    pub(super) checked_opaque_types_storage_entries: Cell<Option<OpaqueTypeStorageEntries>>,

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

    pub(super) deferred_transmute_checks: RefCell<Vec<(Ty<'tcx>, Ty<'tcx>, HirId)>>,

    pub(super) deferred_asm_checks: RefCell<Vec<(&'tcx hir::InlineAsm<'tcx>, HirId)>>,

    pub(super) deferred_repeat_expr_checks:
        RefCell<Vec<(&'tcx hir::Expr<'tcx>, Ty<'tcx>, ty::Const<'tcx>)>>,

    /// Whenever we introduce an adjustment from `!` into a type variable,
    /// we record that type variable here. This is later used to inform
    /// fallback. See the `fallback` module for details.
    pub(super) diverging_type_vars: RefCell<UnordSet<Ty<'tcx>>>,
}

impl<'tcx> Deref for TypeckRootCtxt<'tcx> {
    type Target = InferCtxt<'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.infcx
    }
}

impl<'tcx> TypeckRootCtxt<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> Self {
        let hir_owner = tcx.local_def_id_to_hir_id(def_id).owner;

        let infcx = tcx
            .infer_ctxt()
            .ignoring_regions()
            .in_hir_typeck()
            .build(TypingMode::typeck_for_body(tcx, def_id));
        let typeck_results = RefCell::new(ty::TypeckResults::new(hir_owner));
        let fulfillment_cx = RefCell::new(<dyn TraitEngine<'_, _>>::new(&infcx));

        TypeckRootCtxt {
            infcx,
            typeck_results,
            locals: RefCell::new(Default::default()),
            fulfillment_cx,
            checked_opaque_types_storage_entries: Cell::new(None),
            deferred_sized_obligations: RefCell::new(Vec::new()),
            deferred_call_resolutions: RefCell::new(Default::default()),
            deferred_cast_checks: RefCell::new(Vec::new()),
            deferred_transmute_checks: RefCell::new(Vec::new()),
            deferred_asm_checks: RefCell::new(Vec::new()),
            deferred_repeat_expr_checks: RefCell::new(Vec::new()),
            diverging_type_vars: RefCell::new(Default::default()),
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
}
