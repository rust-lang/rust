use super::TrackedValue;
use crate::{
    check::FnCtxt,
    expr_use_visitor::{self, ExprUseVisitor},
};
use hir::{def_id::DefId, Body, HirId, HirIdMap};
use rustc_data_structures::stable_set::FxHashSet;
use rustc_hir as hir;
use rustc_middle::hir::map::Map;

pub(super) fn find_consumed_and_borrowed<'a, 'tcx>(
    fcx: &'a FnCtxt<'a, 'tcx>,
    def_id: DefId,
    body: &'tcx Body<'tcx>,
) -> ConsumedAndBorrowedPlaces {
    let mut expr_use_visitor = ExprUseDelegate::new(fcx.tcx.hir());
    expr_use_visitor.consume_body(fcx, def_id, body);
    expr_use_visitor.places
}

pub(super) struct ConsumedAndBorrowedPlaces {
    /// Records the variables/expressions that are dropped by a given expression.
    ///
    /// The key is the hir-id of the expression, and the value is a set or hir-ids for variables
    /// or values that are consumed by that expression.
    ///
    /// Note that this set excludes "partial drops" -- for example, a statement like `drop(x.y)` is
    /// not considered a drop of `x`, although it would be a drop of `x.y`.
    pub(super) consumed: HirIdMap<FxHashSet<TrackedValue>>,
    /// A set of hir-ids of values or variables that are borrowed at some point within the body.
    pub(super) borrowed: FxHashSet<TrackedValue>,
}

/// Works with ExprUseVisitor to find interesting values for the drop range analysis.
///
/// Interesting values are those that are either dropped or borrowed. For dropped values, we also
/// record the parent expression, which is the point where the drop actually takes place.
struct ExprUseDelegate<'tcx> {
    hir: Map<'tcx>,
    places: ConsumedAndBorrowedPlaces,
}

impl<'tcx> ExprUseDelegate<'tcx> {
    fn new(hir: Map<'tcx>) -> Self {
        Self {
            hir,
            places: ConsumedAndBorrowedPlaces {
                consumed: <_>::default(),
                borrowed: <_>::default(),
            },
        }
    }

    fn consume_body(&mut self, fcx: &'_ FnCtxt<'_, 'tcx>, def_id: DefId, body: &'tcx Body<'tcx>) {
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

    fn mark_consumed(&mut self, consumer: HirId, target: TrackedValue) {
        if !self.places.consumed.contains_key(&consumer) {
            self.places.consumed.insert(consumer, <_>::default());
        }
        self.places.consumed.get_mut(&consumer).map(|places| places.insert(target));
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
        place_with_id
            .try_into()
            .map_or((), |tracked_value| self.mark_consumed(parent, tracked_value));
    }

    fn borrow(
        &mut self,
        place_with_id: &expr_use_visitor::PlaceWithHirId<'tcx>,
        _diag_expr_id: HirId,
        _bk: rustc_middle::ty::BorrowKind,
    ) {
        self.places
            .borrowed
            .insert(TrackedValue::from_place_with_projections_allowed(place_with_id));
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
