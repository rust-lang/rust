use super::TrackedValue;
use crate::{
    expr_use_visitor::{self, ExprUseVisitor},
    FnCtxt,
};
use hir::{def_id::DefId, Body, HirId, HirIdMap};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_middle::ty::{ParamEnv, TyCtxt};
use rustc_middle::{
    hir::place::{PlaceBase, Projection, ProjectionKind},
    ty::TypeVisitableExt,
};

pub(super) fn find_consumed_and_borrowed<'a, 'tcx>(
    fcx: &'a FnCtxt<'a, 'tcx>,
    def_id: DefId,
    body: &'tcx Body<'tcx>,
) -> ConsumedAndBorrowedPlaces {
    let mut expr_use_visitor = ExprUseDelegate::new(fcx.tcx, fcx.param_env);
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

    /// A set of hir-ids of values or variables that are borrowed at some point within the body.
    pub(super) borrowed_temporaries: FxHashSet<HirId>,
}

/// Works with ExprUseVisitor to find interesting values for the drop range analysis.
///
/// Interesting values are those that are either dropped or borrowed. For dropped values, we also
/// record the parent expression, which is the point where the drop actually takes place.
struct ExprUseDelegate<'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    places: ConsumedAndBorrowedPlaces,
}

impl<'tcx> ExprUseDelegate<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, param_env: ParamEnv<'tcx>) -> Self {
        Self {
            tcx,
            param_env,
            places: ConsumedAndBorrowedPlaces {
                consumed: <_>::default(),
                borrowed: <_>::default(),
                borrowed_temporaries: <_>::default(),
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
        self.places.consumed.entry(consumer).or_insert_with(|| <_>::default());

        debug!(?consumer, ?target, "mark_consumed");
        self.places.consumed.get_mut(&consumer).map(|places| places.insert(target));
    }

    fn borrow_place(&mut self, place_with_id: &expr_use_visitor::PlaceWithHirId<'tcx>) {
        self.places
            .borrowed
            .insert(TrackedValue::from_place_with_projections_allowed(place_with_id));

        // Ordinarily a value is consumed by it's parent, but in the special case of a
        // borrowed RValue, we create a reference that lives as long as the temporary scope
        // for that expression (typically, the innermost statement, but sometimes the enclosing
        // block). We record this fact here so that later in generator_interior
        // we can use the correct scope.
        //
        // We special case borrows through a dereference (`&*x`, `&mut *x` where `x` is
        // some rvalue expression), since these are essentially a copy of a pointer.
        // In other words, this borrow does not refer to the
        // temporary (`*x`), but to the referent (whatever `x` is a borrow of).
        //
        // We were considering that we might encounter problems down the line if somehow,
        // some part of the compiler were to look at this result and try to use it to
        // drive a borrowck-like analysis (this does not currently happen, as of this writing).
        // But even this should be fine, because the lifetime of the dereferenced reference
        // found in the rvalue is only significant as an intermediate 'link' to the value we
        // are producing, and we separately track whether that value is live over a yield.
        // Example:
        //
        // ```notrust
        // fn identity<T>(x: &mut T) -> &mut T { x }
        // let a: A = ...;
        // let y: &'y mut A = &mut *identity(&'a mut a);
        //                    ^^^^^^^^^^^^^^^^^^^^^^^^^ the borrow we are talking about
        // ```
        //
        // The expression `*identity(...)` is a deref of an rvalue,
        // where the `identity(...)` (the rvalue) produces a return type
        // of `&'rv mut A`, where `'a: 'rv`. We then assign this result to
        // `'y`, resulting in (transitively) `'a: 'y` (i.e., while `y` is in use,
        // `a` will be considered borrowed). Other parts of the code will ensure
        // that if `y` is live over a yield, `&'y mut A` appears in the generator
        // state. If `'y` is live, then any sound region analysis must conclude
        // that `'a` is also live. So if this causes a bug, blame some other
        // part of the code!
        let is_deref = place_with_id
            .place
            .projections
            .iter()
            .any(|Projection { kind, .. }| *kind == ProjectionKind::Deref);

        if let (false, PlaceBase::Rvalue) = (is_deref, place_with_id.place.base) {
            self.places.borrowed_temporaries.insert(place_with_id.hir_id);
        }
    }
}

impl<'tcx> expr_use_visitor::Delegate<'tcx> for ExprUseDelegate<'tcx> {
    fn consume(
        &mut self,
        place_with_id: &expr_use_visitor::PlaceWithHirId<'tcx>,
        diag_expr_id: HirId,
    ) {
        let hir = self.tcx.hir();
        let parent = match hir.opt_parent_id(place_with_id.hir_id) {
            Some(parent) => parent,
            None => place_with_id.hir_id,
        };
        debug!(
            "consume {:?}; diag_expr_id={}, using parent {}",
            place_with_id,
            hir.node_to_string(diag_expr_id),
            hir.node_to_string(parent)
        );
        place_with_id
            .try_into()
            .map_or((), |tracked_value| self.mark_consumed(parent, tracked_value));
    }

    fn borrow(
        &mut self,
        place_with_id: &expr_use_visitor::PlaceWithHirId<'tcx>,
        diag_expr_id: HirId,
        bk: rustc_middle::ty::BorrowKind,
    ) {
        debug!(
            "borrow: place_with_id = {place_with_id:#?}, diag_expr_id={diag_expr_id:#?}, \
            borrow_kind={bk:#?}"
        );

        self.borrow_place(place_with_id);
    }

    fn copy(
        &mut self,
        place_with_id: &expr_use_visitor::PlaceWithHirId<'tcx>,
        _diag_expr_id: HirId,
    ) {
        debug!("copy: place_with_id = {place_with_id:?}");

        self.places
            .borrowed
            .insert(TrackedValue::from_place_with_projections_allowed(place_with_id));

        // For copied we treat this mostly like a borrow except that we don't add the place
        // to borrowed_temporaries because the copy is consumed.
    }

    fn mutate(
        &mut self,
        assignee_place: &expr_use_visitor::PlaceWithHirId<'tcx>,
        diag_expr_id: HirId,
    ) {
        debug!("mutate {assignee_place:?}; diag_expr_id={diag_expr_id:?}");

        if assignee_place.place.base == PlaceBase::Rvalue
            && assignee_place.place.projections.is_empty()
        {
            // Assigning to an Rvalue is illegal unless done through a dereference. We would have
            // already gotten a type error, so we will just return here.
            return;
        }

        // If the type being assigned needs dropped, then the mutation counts as a borrow
        // since it is essentially doing `Drop::drop(&mut x); x = new_value;`.
        let ty = self.tcx.erase_regions(assignee_place.place.base_ty);
        if ty.needs_infer() {
            self.tcx.sess.delay_span_bug(
                self.tcx.hir().span(assignee_place.hir_id),
                &format!("inference variables in {ty}"),
            );
        } else if ty.needs_drop(self.tcx, self.param_env) {
            self.places
                .borrowed
                .insert(TrackedValue::from_place_with_projections_allowed(assignee_place));
        }
    }

    fn bind(
        &mut self,
        binding_place: &expr_use_visitor::PlaceWithHirId<'tcx>,
        diag_expr_id: HirId,
    ) {
        debug!("bind {binding_place:?}; diag_expr_id={diag_expr_id:?}");
    }

    fn fake_read(
        &mut self,
        place_with_id: &expr_use_visitor::PlaceWithHirId<'tcx>,
        cause: rustc_middle::mir::FakeReadCause,
        diag_expr_id: HirId,
    ) {
        debug!(
            "fake_read place_with_id={place_with_id:?}; cause={cause:?}; diag_expr_id={diag_expr_id:?}"
        );

        // fake reads happen in places like the scrutinee of a match expression.
        // we treat those as a borrow, much like a copy: the idea is that we are
        // transiently creating a `&T` ref that we can read from to observe the current
        // value (this `&T` is immediately dropped afterwards).
        self.borrow_place(place_with_id);
    }
}
