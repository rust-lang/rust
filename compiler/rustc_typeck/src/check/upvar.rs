//! ### Inferring borrow kinds for upvars
//!
//! Whenever there is a closure expression, we need to determine how each
//! upvar is used. We do this by initially assigning each upvar an
//! immutable "borrow kind" (see `ty::BorrowKind` for details) and then
//! "escalating" the kind as needed. The borrow kind proceeds according to
//! the following lattice:
//!
//!     ty::ImmBorrow -> ty::UniqueImmBorrow -> ty::MutBorrow
//!
//! So, for example, if we see an assignment `x = 5` to an upvar `x`, we
//! will promote its borrow kind to mutable borrow. If we see an `&mut x`
//! we'll do the same. Naturally, this applies not just to the upvar, but
//! to everything owned by `x`, so the result is the same for something
//! like `x.f = 5` and so on (presuming `x` is not a borrowed pointer to a
//! struct). These adjustments are performed in
//! `adjust_upvar_borrow_kind()` (you can trace backwards through the code
//! from there).
//!
//! The fact that we are inferring borrow kinds as we go results in a
//! semi-hacky interaction with mem-categorization. In particular,
//! mem-categorization will query the current borrow kind as it
//! categorizes, and we'll return the *current* value, but this may get
//! adjusted later. Therefore, in this module, we generally ignore the
//! borrow kind (and derived mutabilities) that are returned from
//! mem-categorization, since they may be inaccurate. (Another option
//! would be to use a unification scheme, where instead of returning a
//! concrete borrow kind like `ty::ImmBorrow`, we return a
//! `ty::InferBorrow(upvar_id)` or something like that, but this would
//! then mean that all later passes would have to check for these figments
//! and report an error, and it just seems like more mess in the end.)

use super::FnCtxt;

use crate::expr_use_visitor as euv;
use rustc_data_structures::fx::FxIndexMap;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_infer::infer::UpvarRegion;
use rustc_middle::hir::place::{PlaceBase, PlaceWithHirId};
use rustc_middle::ty::{self, Ty, TyCtxt, UpvarSubsts};
use rustc_span::{Span, Symbol};
use std::collections::hash_map::Entry;

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub fn closure_analyze(&self, body: &'tcx hir::Body<'tcx>) {
        InferBorrowKindVisitor { fcx: self }.visit_body(body);

        // it's our job to process these.
        assert!(self.deferred_call_resolutions.borrow().is_empty());
    }
}

struct InferBorrowKindVisitor<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for InferBorrowKindVisitor<'a, 'tcx> {
    type Map = intravisit::ErasedMap<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::Closure(cc, _, body_id, _, _) = expr.kind {
            let body = self.fcx.tcx.hir().body(body_id);
            self.visit_body(body);
            self.fcx.analyze_closure(expr.hir_id, expr.span, body, cc);
        }

        intravisit::walk_expr(self, expr);
    }
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    /// Analysis starting point.
    fn analyze_closure(
        &self,
        closure_hir_id: hir::HirId,
        span: Span,
        body: &hir::Body<'_>,
        capture_clause: hir::CaptureBy,
    ) {
        debug!("analyze_closure(id={:?}, body.id={:?})", closure_hir_id, body.id());

        // Extract the type of the closure.
        let ty = self.node_ty(closure_hir_id);
        let (closure_def_id, substs) = match *ty.kind() {
            ty::Closure(def_id, substs) => (def_id, UpvarSubsts::Closure(substs)),
            ty::Generator(def_id, substs, _) => (def_id, UpvarSubsts::Generator(substs)),
            ty::Error(_) => {
                // #51714: skip analysis when we have already encountered type errors
                return;
            }
            _ => {
                span_bug!(
                    span,
                    "type of closure expr {:?} is not a closure {:?}",
                    closure_hir_id,
                    ty
                );
            }
        };

        let infer_kind = if let UpvarSubsts::Closure(closure_substs) = substs {
            self.closure_kind(closure_substs).is_none().then_some(closure_substs)
        } else {
            None
        };

        if let Some(upvars) = self.tcx.upvars_mentioned(closure_def_id) {
            let mut closure_captures: FxIndexMap<hir::HirId, ty::UpvarId> =
                FxIndexMap::with_capacity_and_hasher(upvars.len(), Default::default());
            for (&var_hir_id, _) in upvars.iter() {
                let upvar_id = ty::UpvarId {
                    var_path: ty::UpvarPath { hir_id: var_hir_id },
                    closure_expr_id: closure_def_id.expect_local(),
                };
                debug!("seed upvar_id {:?}", upvar_id);
                // Adding the upvar Id to the list of Upvars, which will be added
                // to the map for the closure at the end of the for loop.
                closure_captures.insert(var_hir_id, upvar_id);

                let capture_kind = match capture_clause {
                    hir::CaptureBy::Value => ty::UpvarCapture::ByValue(None),
                    hir::CaptureBy::Ref => {
                        let origin = UpvarRegion(upvar_id, span);
                        let upvar_region = self.next_region_var(origin);
                        let upvar_borrow =
                            ty::UpvarBorrow { kind: ty::ImmBorrow, region: upvar_region };
                        ty::UpvarCapture::ByRef(upvar_borrow)
                    }
                };

                self.typeck_results.borrow_mut().upvar_capture_map.insert(upvar_id, capture_kind);
            }
            // Add the vector of upvars to the map keyed with the closure id.
            // This gives us an easier access to them without having to call
            // tcx.upvars again..
            if !closure_captures.is_empty() {
                self.typeck_results
                    .borrow_mut()
                    .closure_captures
                    .insert(closure_def_id, closure_captures);
            }
        }

        let body_owner_def_id = self.tcx.hir().body_owner_def_id(body.id());
        assert_eq!(body_owner_def_id.to_def_id(), closure_def_id);
        let mut delegate = InferBorrowKind {
            fcx: self,
            closure_def_id,
            current_closure_kind: ty::ClosureKind::LATTICE_BOTTOM,
            current_origin: None,
            adjust_upvar_captures: ty::UpvarCaptureMap::default(),
        };
        euv::ExprUseVisitor::new(
            &mut delegate,
            &self.infcx,
            body_owner_def_id,
            self.param_env,
            &self.typeck_results.borrow(),
        )
        .consume_body(body);

        if let Some(closure_substs) = infer_kind {
            // Unify the (as yet unbound) type variable in the closure
            // substs with the kind we inferred.
            let inferred_kind = delegate.current_closure_kind;
            let closure_kind_ty = closure_substs.as_closure().kind_ty();
            self.demand_eqtype(span, inferred_kind.to_ty(self.tcx), closure_kind_ty);

            // If we have an origin, store it.
            if let Some(origin) = delegate.current_origin {
                self.typeck_results
                    .borrow_mut()
                    .closure_kind_origins_mut()
                    .insert(closure_hir_id, origin);
            }
        }

        self.typeck_results.borrow_mut().upvar_capture_map.extend(delegate.adjust_upvar_captures);

        // Now that we've analyzed the closure, we know how each
        // variable is borrowed, and we know what traits the closure
        // implements (Fn vs FnMut etc). We now have some updates to do
        // with that information.
        //
        // Note that no closure type C may have an upvar of type C
        // (though it may reference itself via a trait object). This
        // results from the desugaring of closures to a struct like
        // `Foo<..., UV0...UVn>`. If one of those upvars referenced
        // C, then the type would have infinite size (and the
        // inference algorithm will reject it).

        // Equate the type variables for the upvars with the actual types.
        let final_upvar_tys = self.final_upvar_tys(closure_hir_id);
        debug!(
            "analyze_closure: id={:?} substs={:?} final_upvar_tys={:?}",
            closure_hir_id, substs, final_upvar_tys
        );

        // Build a tuple (U0..Un) of the final upvar types U0..Un
        // and unify the upvar tupe type in the closure with it:
        let final_tupled_upvars_type = self.tcx.mk_tup(final_upvar_tys.iter());
        self.demand_suptype(span, substs.tupled_upvars_ty(), final_tupled_upvars_type);

        // If we are also inferred the closure kind here,
        // process any deferred resolutions.
        let deferred_call_resolutions = self.remove_deferred_call_resolutions(closure_def_id);
        for deferred_call_resolution in deferred_call_resolutions {
            deferred_call_resolution.resolve(self);
        }
    }

    // Returns a list of `Ty`s for each upvar.
    fn final_upvar_tys(&self, closure_id: hir::HirId) -> Vec<Ty<'tcx>> {
        // Presently an unboxed closure type cannot "escape" out of a
        // function, so we will only encounter ones that originated in the
        // local crate or were inlined into it along with some function.
        // This may change if abstract return types of some sort are
        // implemented.
        let tcx = self.tcx;
        let closure_def_id = tcx.hir().local_def_id(closure_id);

        tcx.upvars_mentioned(closure_def_id)
            .iter()
            .flat_map(|upvars| {
                upvars.iter().map(|(&var_hir_id, _)| {
                    let upvar_ty = self.node_ty(var_hir_id);
                    let upvar_id = ty::UpvarId {
                        var_path: ty::UpvarPath { hir_id: var_hir_id },
                        closure_expr_id: closure_def_id,
                    };
                    let capture = self.typeck_results.borrow().upvar_capture(upvar_id);

                    debug!("var_id={:?} upvar_ty={:?} capture={:?}", var_hir_id, upvar_ty, capture);

                    match capture {
                        ty::UpvarCapture::ByValue(_) => upvar_ty,
                        ty::UpvarCapture::ByRef(borrow) => tcx.mk_ref(
                            borrow.region,
                            ty::TypeAndMut { ty: upvar_ty, mutbl: borrow.kind.to_mutbl_lossy() },
                        ),
                    }
                })
            })
            .collect()
    }
}

struct InferBorrowKind<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,

    // The def-id of the closure whose kind and upvar accesses are being inferred.
    closure_def_id: DefId,

    // The kind that we have inferred that the current closure
    // requires. Note that we *always* infer a minimal kind, even if
    // we don't always *use* that in the final result (i.e., sometimes
    // we've taken the closure kind from the expectations instead, and
    // for generators we don't even implement the closure traits
    // really).
    current_closure_kind: ty::ClosureKind,

    // If we modified `current_closure_kind`, this field contains a `Some()` with the
    // variable access that caused us to do so.
    current_origin: Option<(Span, Symbol)>,

    // For each upvar that we access, we track the minimal kind of
    // access we need (ref, ref mut, move, etc).
    adjust_upvar_captures: ty::UpvarCaptureMap<'tcx>,
}

impl<'a, 'tcx> InferBorrowKind<'a, 'tcx> {
    fn adjust_upvar_borrow_kind_for_consume(
        &mut self,
        place_with_id: &PlaceWithHirId<'tcx>,
        mode: euv::ConsumeMode,
    ) {
        debug!(
            "adjust_upvar_borrow_kind_for_consume(place_with_id={:?}, mode={:?})",
            place_with_id, mode
        );

        // we only care about moves
        match mode {
            euv::Copy => {
                return;
            }
            euv::Move => {}
        }

        let tcx = self.fcx.tcx;
        let upvar_id = if let PlaceBase::Upvar(upvar_id) = place_with_id.place.base {
            upvar_id
        } else {
            return;
        };

        debug!("adjust_upvar_borrow_kind_for_consume: upvar={:?}", upvar_id);

        let usage_span = tcx.hir().span(place_with_id.hir_id);

        // To move out of an upvar, this must be a FnOnce closure
        self.adjust_closure_kind(
            upvar_id.closure_expr_id,
            ty::ClosureKind::FnOnce,
            usage_span,
            var_name(tcx, upvar_id.var_path.hir_id),
        );

        // In a case like `let pat = upvar`, don't use the span
        // of the pattern, as this just looks confusing.
        let by_value_span = match tcx.hir().get(place_with_id.hir_id) {
            hir::Node::Pat(_) => None,
            _ => Some(usage_span),
        };

        let new_capture = ty::UpvarCapture::ByValue(by_value_span);
        match self.adjust_upvar_captures.entry(upvar_id) {
            Entry::Occupied(mut e) => {
                match e.get() {
                    // We always overwrite `ByRef`, since we require
                    // that the upvar be available by value.
                    //
                    // If we had a previous by-value usage without a specific
                    // span, use ours instead. Otherwise, keep the first span
                    // we encountered, since there isn't an obviously better one.
                    ty::UpvarCapture::ByRef(_) | ty::UpvarCapture::ByValue(None) => {
                        e.insert(new_capture);
                    }
                    _ => {}
                }
            }
            Entry::Vacant(e) => {
                e.insert(new_capture);
            }
        }
    }

    /// Indicates that `place_with_id` is being directly mutated (e.g., assigned
    /// to). If the place is based on a by-ref upvar, this implies that
    /// the upvar must be borrowed using an `&mut` borrow.
    fn adjust_upvar_borrow_kind_for_mut(&mut self, place_with_id: &PlaceWithHirId<'tcx>) {
        debug!("adjust_upvar_borrow_kind_for_mut(place_with_id={:?})", place_with_id);

        if let PlaceBase::Upvar(upvar_id) = place_with_id.place.base {
            let mut borrow_kind = ty::MutBorrow;
            for pointer_ty in place_with_id.place.deref_tys() {
                match pointer_ty.kind() {
                    // Raw pointers don't inherit mutability.
                    ty::RawPtr(_) => return,
                    // assignment to deref of an `&mut`
                    // borrowed pointer implies that the
                    // pointer itself must be unique, but not
                    // necessarily *mutable*
                    ty::Ref(.., hir::Mutability::Mut) => borrow_kind = ty::UniqueImmBorrow,
                    _ => (),
                }
            }
            self.adjust_upvar_deref(
                upvar_id,
                self.fcx.tcx.hir().span(place_with_id.hir_id),
                borrow_kind,
            );
        }
    }

    fn adjust_upvar_borrow_kind_for_unique(&mut self, place_with_id: &PlaceWithHirId<'tcx>) {
        debug!("adjust_upvar_borrow_kind_for_unique(place_with_id={:?})", place_with_id);

        if let PlaceBase::Upvar(upvar_id) = place_with_id.place.base {
            if place_with_id.place.deref_tys().any(ty::TyS::is_unsafe_ptr) {
                // Raw pointers don't inherit mutability.
                return;
            }
            // for a borrowed pointer to be unique, its base must be unique
            self.adjust_upvar_deref(
                upvar_id,
                self.fcx.tcx.hir().span(place_with_id.hir_id),
                ty::UniqueImmBorrow,
            );
        }
    }

    fn adjust_upvar_deref(
        &mut self,
        upvar_id: ty::UpvarId,
        place_span: Span,
        borrow_kind: ty::BorrowKind,
    ) {
        assert!(match borrow_kind {
            ty::MutBorrow => true,
            ty::UniqueImmBorrow => true,

            // imm borrows never require adjusting any kinds, so we don't wind up here
            ty::ImmBorrow => false,
        });

        let tcx = self.fcx.tcx;

        // if this is an implicit deref of an
        // upvar, then we need to modify the
        // borrow_kind of the upvar to make sure it
        // is inferred to mutable if necessary
        self.adjust_upvar_borrow_kind(upvar_id, borrow_kind);

        // also need to be in an FnMut closure since this is not an ImmBorrow
        self.adjust_closure_kind(
            upvar_id.closure_expr_id,
            ty::ClosureKind::FnMut,
            place_span,
            var_name(tcx, upvar_id.var_path.hir_id),
        );
    }

    /// We infer the borrow_kind with which to borrow upvars in a stack closure.
    /// The borrow_kind basically follows a lattice of `imm < unique-imm < mut`,
    /// moving from left to right as needed (but never right to left).
    /// Here the argument `mutbl` is the borrow_kind that is required by
    /// some particular use.
    fn adjust_upvar_borrow_kind(&mut self, upvar_id: ty::UpvarId, kind: ty::BorrowKind) {
        let upvar_capture = self
            .adjust_upvar_captures
            .get(&upvar_id)
            .copied()
            .unwrap_or_else(|| self.fcx.typeck_results.borrow().upvar_capture(upvar_id));
        debug!(
            "adjust_upvar_borrow_kind(upvar_id={:?}, upvar_capture={:?}, kind={:?})",
            upvar_id, upvar_capture, kind
        );

        match upvar_capture {
            ty::UpvarCapture::ByValue(_) => {
                // Upvar is already by-value, the strongest criteria.
            }
            ty::UpvarCapture::ByRef(mut upvar_borrow) => {
                match (upvar_borrow.kind, kind) {
                    // Take RHS:
                    (ty::ImmBorrow, ty::UniqueImmBorrow | ty::MutBorrow)
                    | (ty::UniqueImmBorrow, ty::MutBorrow) => {
                        upvar_borrow.kind = kind;
                        self.adjust_upvar_captures
                            .insert(upvar_id, ty::UpvarCapture::ByRef(upvar_borrow));
                    }
                    // Take LHS:
                    (ty::ImmBorrow, ty::ImmBorrow)
                    | (ty::UniqueImmBorrow, ty::ImmBorrow | ty::UniqueImmBorrow)
                    | (ty::MutBorrow, _) => {}
                }
            }
        }
    }

    fn adjust_closure_kind(
        &mut self,
        closure_id: LocalDefId,
        new_kind: ty::ClosureKind,
        upvar_span: Span,
        var_name: Symbol,
    ) {
        debug!(
            "adjust_closure_kind(closure_id={:?}, new_kind={:?}, upvar_span={:?}, var_name={})",
            closure_id, new_kind, upvar_span, var_name
        );

        // Is this the closure whose kind is currently being inferred?
        if closure_id.to_def_id() != self.closure_def_id {
            debug!("adjust_closure_kind: not current closure");
            return;
        }

        // closures start out as `Fn`.
        let existing_kind = self.current_closure_kind;

        debug!(
            "adjust_closure_kind: closure_id={:?}, existing_kind={:?}, new_kind={:?}",
            closure_id, existing_kind, new_kind
        );

        match (existing_kind, new_kind) {
            (ty::ClosureKind::Fn, ty::ClosureKind::Fn)
            | (ty::ClosureKind::FnMut, ty::ClosureKind::Fn | ty::ClosureKind::FnMut)
            | (ty::ClosureKind::FnOnce, _) => {
                // no change needed
            }

            (ty::ClosureKind::Fn, ty::ClosureKind::FnMut | ty::ClosureKind::FnOnce)
            | (ty::ClosureKind::FnMut, ty::ClosureKind::FnOnce) => {
                // new kind is stronger than the old kind
                self.current_closure_kind = new_kind;
                self.current_origin = Some((upvar_span, var_name));
            }
        }
    }
}

impl<'a, 'tcx> euv::Delegate<'tcx> for InferBorrowKind<'a, 'tcx> {
    fn consume(&mut self, place_with_id: &PlaceWithHirId<'tcx>, mode: euv::ConsumeMode) {
        debug!("consume(place_with_id={:?},mode={:?})", place_with_id, mode);
        self.adjust_upvar_borrow_kind_for_consume(place_with_id, mode);
    }

    fn borrow(&mut self, place_with_id: &PlaceWithHirId<'tcx>, bk: ty::BorrowKind) {
        debug!("borrow(place_with_id={:?}, bk={:?})", place_with_id, bk);

        match bk {
            ty::ImmBorrow => {}
            ty::UniqueImmBorrow => {
                self.adjust_upvar_borrow_kind_for_unique(place_with_id);
            }
            ty::MutBorrow => {
                self.adjust_upvar_borrow_kind_for_mut(place_with_id);
            }
        }
    }

    fn mutate(&mut self, assignee_place: &PlaceWithHirId<'tcx>) {
        debug!("mutate(assignee_place={:?})", assignee_place);

        self.adjust_upvar_borrow_kind_for_mut(assignee_place);
    }
}

fn var_name(tcx: TyCtxt<'_>, var_hir_id: hir::HirId) -> Symbol {
    tcx.hir().name(var_hir_id)
}
