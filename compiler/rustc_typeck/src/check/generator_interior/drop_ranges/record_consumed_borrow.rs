use crate::{
    check::FnCtxt,
    expr_use_visitor::{self, ExprUseVisitor},
};
use hir::{HirId, HirIdMap, HirIdSet, Body, def_id::DefId};
use rustc_hir as hir;
use rustc_middle::hir::{
    map::Map,
    place::{Place, PlaceBase},
};
use rustc_middle::ty;

/// Works with ExprUseVisitor to find interesting values for the drop range analysis.
///
/// Interesting values are those that are either dropped or borrowed. For dropped values, we also
/// record the parent expression, which is the point where the drop actually takes place.
pub struct ExprUseDelegate<'tcx> {
    pub(super) hir: Map<'tcx>,
    /// Records the variables/expressions that are dropped by a given expression.
    ///
    /// The key is the hir-id of the expression, and the value is a set or hir-ids for variables
    /// or values that are consumed by that expression.
    ///
    /// Note that this set excludes "partial drops" -- for example, a statement like `drop(x.y)` is
    /// not considered a drop of `x`, although it would be a drop of `x.y`.
    pub(super) consumed_places: HirIdMap<HirIdSet>,
    /// A set of hir-ids of values or variables that are borrowed at some point within the body.
    pub(super) borrowed_places: HirIdSet,
}

impl<'tcx> ExprUseDelegate<'tcx> {
    pub fn new(hir: Map<'tcx>) -> Self {
        Self { hir, consumed_places: <_>::default(), borrowed_places: <_>::default() }
    }

    pub fn consume_body(
        &mut self,
        fcx: &'_ FnCtxt<'_, 'tcx>,
        def_id: DefId,
        body: &'tcx Body<'tcx>,
    ) {
        // Run ExprUseVisitor to find where values are consumed.
        ExprUseVisitor::new(
            self,
            &fcx.infcx,
            def_id.expect_local(),
            fcx.param_env,
            &fcx.typeck_results.borrow(),
        )
        .consume_body(body);
    }

    fn mark_consumed(&mut self, consumer: HirId, target: HirId) {
        if !self.consumed_places.contains_key(&consumer) {
            self.consumed_places.insert(consumer, <_>::default());
        }
        self.consumed_places.get_mut(&consumer).map(|places| places.insert(target));
    }
}

impl<'tcx> expr_use_visitor::Delegate<'tcx> for ExprUseDelegate<'tcx> {
    fn consume(
        &mut self,
        place_with_id: &expr_use_visitor::PlaceWithHirId<'tcx>,
        diag_expr_id: HirId,
    ) {
        let parent = match self.hir.find_parent_node(place_with_id.hir_id) {
            Some(parent) => parent,
            None => place_with_id.hir_id,
        };
        debug!(
            "consume {:?}; diag_expr_id={:?}, using parent {:?}",
            place_with_id, diag_expr_id, parent
        );
        self.mark_consumed(parent, place_with_id.hir_id);
        place_hir_id(&place_with_id.place).map(|place| self.mark_consumed(parent, place));
    }

    fn borrow(
        &mut self,
        place_with_id: &expr_use_visitor::PlaceWithHirId<'tcx>,
        _diag_expr_id: HirId,
        _bk: rustc_middle::ty::BorrowKind,
    ) {
        place_hir_id(&place_with_id.place).map(|place| self.borrowed_places.insert(place));
    }

    fn mutate(
        &mut self,
        _assignee_place: &expr_use_visitor::PlaceWithHirId<'tcx>,
        _diag_expr_id: HirId,
    ) {
    }

    fn fake_read(
        &mut self,
        _place: expr_use_visitor::Place<'tcx>,
        _cause: rustc_middle::mir::FakeReadCause,
        _diag_expr_id: HirId,
    ) {
    }
}

/// Gives the hir_id associated with a place if one exists. This is the hir_id that we want to
/// track for a value in the drop range analysis.
fn place_hir_id(place: &Place<'_>) -> Option<HirId> {
    match place.base {
        PlaceBase::Rvalue | PlaceBase::StaticItem => None,
        PlaceBase::Local(hir_id)
        | PlaceBase::Upvar(ty::UpvarId { var_path: ty::UpvarPath { hir_id }, .. }) => Some(hir_id),
    }
}
