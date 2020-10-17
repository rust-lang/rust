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
use rustc_middle::hir::place::{Place, PlaceBase, PlaceWithHirId};
use rustc_middle::ty::{self, Ty, TyCtxt, UpvarSubsts};
use rustc_span::sym;
use rustc_span::{Span, Symbol};

macro_rules! log_capture_analysis {
    ($fcx:expr, $closure_def_id:expr, $fmt:literal) => {
        if $fcx.should_log_capture_analysis($closure_def_id) {
            print!("For closure={:?}: ", $closure_def_id);
            println!($fmt);
        }
    };

    ($fcx:expr, $closure_def_id:expr, $fmt:literal, $($args:expr),*) => {
        if $fcx.should_log_capture_analysis($closure_def_id) {
            print!("For closure={:?}: ", $closure_def_id);
            println!($fmt, $($args),*);
        }
    };
}

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

        let local_def_id = closure_def_id.expect_local();

        let mut capture_information = FxIndexMap::<Place<'tcx>, ty::CaptureInfo<'tcx>>::default();
        if self.tcx.features().capture_disjoint_fields {
            log_capture_analysis!(self, closure_def_id, "Using new-style capture analysis");
        } else {
            log_capture_analysis!(self, closure_def_id, "Using old-style capture analysis");
            if let Some(upvars) = self.tcx.upvars_mentioned(closure_def_id) {
                for (&var_hir_id, _) in upvars.iter() {
                    let place = self.place_for_root_variable(local_def_id, var_hir_id);

                    debug!("seed place {:?}", place);

                    let upvar_id = ty::UpvarId::new(var_hir_id, local_def_id);
                    let capture_kind = self.init_capture_kind(capture_clause, upvar_id, span);
                    let info = ty::CaptureInfo { expr_id: None, capture_kind };

                    capture_information.insert(place, info);
                }
            }
        }

        let body_owner_def_id = self.tcx.hir().body_owner_def_id(body.id());
        assert_eq!(body_owner_def_id.to_def_id(), closure_def_id);
        let mut delegate = InferBorrowKind {
            fcx: self,
            closure_def_id,
            closure_span: span,
            capture_clause,
            current_closure_kind: ty::ClosureKind::LATTICE_BOTTOM,
            current_origin: None,
            capture_information,
        };
        euv::ExprUseVisitor::new(
            &mut delegate,
            &self.infcx,
            body_owner_def_id,
            self.param_env,
            &self.typeck_results.borrow(),
        )
        .consume_body(body);

        log_capture_analysis!(
            self,
            closure_def_id,
            "capture information: {:#?}",
            delegate.capture_information
        );

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

        self.set_closure_captures(closure_def_id, &delegate);

        self.typeck_results
            .borrow_mut()
            .closure_capture_information
            .insert(closure_def_id, delegate.capture_information);

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

        self.typeck_results
            .borrow()
            .closure_captures
            .get(&closure_def_id.to_def_id())
            .iter()
            .flat_map(|upvars| {
                upvars.iter().map(|(&var_hir_id, _)| {
                    let upvar_ty = self.node_ty(var_hir_id);
                    let upvar_id = ty::UpvarId::new(var_hir_id, closure_def_id);
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

    fn set_closure_captures(
        &self,
        closure_def_id: DefId,
        inferred_info: &InferBorrowKind<'_, 'tcx>,
    ) {
        let mut closure_captures: FxIndexMap<hir::HirId, ty::UpvarId> = Default::default();

        for (place, capture_info) in inferred_info.capture_information.iter() {
            let upvar_id = match place.base {
                PlaceBase::Upvar(upvar_id) => upvar_id,
                base => bug!("Expected upvar, found={:?}", base),
            };

            assert_eq!(upvar_id.closure_expr_id, closure_def_id.expect_local());

            let var_hir_id = upvar_id.var_path.hir_id;
            closure_captures.insert(var_hir_id, upvar_id);

            let new_capture_kind = if let Some(capture_kind) =
                self.typeck_results.borrow_mut().upvar_capture_map.get(&upvar_id)
            {
                // upvar_capture_map only stores the UpvarCapture (CaptureKind),
                // so we create a fake capture info with no expression.
                let fake_capture_info =
                    ty::CaptureInfo { expr_id: None, capture_kind: capture_kind.clone() };
                self.determine_capture_info(fake_capture_info, capture_info.clone()).capture_kind
            } else {
                capture_info.capture_kind
            };
            self.typeck_results.borrow_mut().upvar_capture_map.insert(upvar_id, new_capture_kind);
        }

        if !closure_captures.is_empty() {
            self.typeck_results
                .borrow_mut()
                .closure_captures
                .insert(closure_def_id, closure_captures);
        }
    }

    fn init_capture_kind(
        &self,
        capture_clause: hir::CaptureBy,
        upvar_id: ty::UpvarId,
        closure_span: Span,
    ) -> ty::UpvarCapture<'tcx> {
        match capture_clause {
            hir::CaptureBy::Value => ty::UpvarCapture::ByValue(None),
            hir::CaptureBy::Ref => {
                let origin = UpvarRegion(upvar_id, closure_span);
                let upvar_region = self.next_region_var(origin);
                let upvar_borrow = ty::UpvarBorrow { kind: ty::ImmBorrow, region: upvar_region };
                ty::UpvarCapture::ByRef(upvar_borrow)
            }
        }
    }

    fn place_for_root_variable(
        &self,
        closure_def_id: LocalDefId,
        var_hir_id: hir::HirId,
    ) -> Place<'tcx> {
        let upvar_id = ty::UpvarId::new(var_hir_id, closure_def_id);

        Place {
            base_ty: self.node_ty(var_hir_id),
            base: PlaceBase::Upvar(upvar_id),
            projections: Default::default(),
        }
    }

    fn should_log_capture_analysis(&self, closure_def_id: DefId) -> bool {
        self.tcx.has_attr(closure_def_id, sym::rustc_capture_analysis)
    }

    /// Helper function to determine if we need to escalate CaptureKind from
    /// CaptureInfo A to B and returns the escalated CaptureInfo.
    /// (Note: CaptureInfo contains CaptureKind and an expression that led to capture it in that way)
    ///
    /// If both `CaptureKind`s are considered equivalent, then the CaptureInformation is selected based
    /// on the expression they point to,
    /// - Some(expr) is preferred over None.
    /// - Non-Pattern expressions are preferred over pattern expressions, since pattern expressions
    /// can be confusing
    ///
    /// If the CaptureKind and Expression are considered to be equivalent, then `CaptureInfo` A is
    /// preferred.
    fn determine_capture_info(
        &self,
        capture_info_a: ty::CaptureInfo<'tcx>,
        capture_info_b: ty::CaptureInfo<'tcx>,
    ) -> ty::CaptureInfo<'tcx> {
        // If the capture kind is equivalent then, we don't need to escalate and can compare the
        // expressions.
        let eq_capture_kind = match (capture_info_a.capture_kind, capture_info_b.capture_kind) {
            (ty::UpvarCapture::ByValue(_), ty::UpvarCapture::ByValue(_)) => {
                // Both are either Some or both are either None
                // !(span_a.is_some() ^ span_b.is_some())
                true
            }
            (ty::UpvarCapture::ByRef(ref_a), ty::UpvarCapture::ByRef(ref_b)) => {
                ref_a.kind == ref_b.kind
            }
            _ => false,
        };

        if eq_capture_kind {
            match (capture_info_a.expr_id, capture_info_b.expr_id) {
                (Some(_), None) | (None, None) => return capture_info_a,
                (None, Some(_)) => return capture_info_b,
                (Some(_), Some(_)) => {}
            };

            // Safe to unwrap here, since both are Some(_)
            let expr_kind_a = self.tcx.hir().get(capture_info_a.expr_id.unwrap());
            let expr_kind_b = self.tcx.hir().get(capture_info_b.expr_id.unwrap());

            // If A is a pattern and B is not a pattern, then choose B else choose A.
            if matches!(expr_kind_a, hir::Node::Pat(_)) && !matches!(expr_kind_b, hir::Node::Pat(_))
            {
                return capture_info_b;
            } else {
                return capture_info_a;
            }
        }

        match (capture_info_a.capture_kind, capture_info_b.capture_kind) {
            (ty::UpvarCapture::ByValue(_), _) => capture_info_a,
            (_, ty::UpvarCapture::ByValue(_)) => capture_info_b,
            (ty::UpvarCapture::ByRef(ref_a), ty::UpvarCapture::ByRef(ref_b)) => {
                match (ref_a.kind, ref_b.kind) {
                    // Take RHS:
                    (ty::ImmBorrow, ty::UniqueImmBorrow | ty::MutBorrow)
                    | (ty::UniqueImmBorrow, ty::MutBorrow) => capture_info_b,
                    // Take LHS:
                    (ty::UniqueImmBorrow, ty::ImmBorrow) | (ty::MutBorrow, _) => capture_info_a,
                    (ty::ImmBorrow, ty::ImmBorrow) | (ty::UniqueImmBorrow, ty::UniqueImmBorrow) => {
                        bug!("Expected unequal capture kinds");
                    }
                }
            }
        }
    }
}

struct InferBorrowKind<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,

    // The def-id of the closure whose kind and upvar accesses are being inferred.
    closure_def_id: DefId,

    closure_span: Span,

    capture_clause: hir::CaptureBy,

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
    capture_information: FxIndexMap<Place<'tcx>, ty::CaptureInfo<'tcx>>,
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

        let capture_info = ty::CaptureInfo {
            expr_id: Some(place_with_id.hir_id),
            capture_kind: ty::UpvarCapture::ByValue(Some(usage_span)),
        };

        let curr_info = self.capture_information[&place_with_id.place];
        let updated_info = self.fcx.determine_capture_info(curr_info, capture_info);

        self.capture_information[&place_with_id.place] = updated_info;
    }

    /// Indicates that `place_with_id` is being directly mutated (e.g., assigned
    /// to). If the place is based on a by-ref upvar, this implies that
    /// the upvar must be borrowed using an `&mut` borrow.
    fn adjust_upvar_borrow_kind_for_mut(&mut self, place_with_id: &PlaceWithHirId<'tcx>) {
        debug!("adjust_upvar_borrow_kind_for_mut(place_with_id={:?})", place_with_id);

        if let PlaceBase::Upvar(_) = place_with_id.place.base {
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
            self.adjust_upvar_deref(place_with_id, borrow_kind);
        }
    }

    fn adjust_upvar_borrow_kind_for_unique(&mut self, place_with_id: &PlaceWithHirId<'tcx>) {
        debug!("adjust_upvar_borrow_kind_for_unique(place_with_id={:?})", place_with_id);

        if let PlaceBase::Upvar(_) = place_with_id.place.base {
            if place_with_id.place.deref_tys().any(ty::TyS::is_unsafe_ptr) {
                // Raw pointers don't inherit mutability.
                return;
            }
            // for a borrowed pointer to be unique, its base must be unique
            self.adjust_upvar_deref(place_with_id, ty::UniqueImmBorrow);
        }
    }

    fn adjust_upvar_deref(
        &mut self,
        place_with_id: &PlaceWithHirId<'tcx>,
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
        self.adjust_upvar_borrow_kind(place_with_id, borrow_kind);

        if let PlaceBase::Upvar(upvar_id) = place_with_id.place.base {
            self.adjust_closure_kind(
                upvar_id.closure_expr_id,
                ty::ClosureKind::FnMut,
                tcx.hir().span(place_with_id.hir_id),
                var_name(tcx, upvar_id.var_path.hir_id),
            );
        }
    }

    /// We infer the borrow_kind with which to borrow upvars in a stack closure.
    /// The borrow_kind basically follows a lattice of `imm < unique-imm < mut`,
    /// moving from left to right as needed (but never right to left).
    /// Here the argument `mutbl` is the borrow_kind that is required by
    /// some particular use.
    fn adjust_upvar_borrow_kind(
        &mut self,
        place_with_id: &PlaceWithHirId<'tcx>,
        kind: ty::BorrowKind,
    ) {
        let curr_capture_info = self.capture_information[&place_with_id.place];

        debug!(
            "adjust_upvar_borrow_kind(place={:?}, capture_info={:?}, kind={:?})",
            place_with_id, curr_capture_info, kind
        );

        if let ty::UpvarCapture::ByValue(_) = curr_capture_info.capture_kind {
            // It's already capture by value, we don't need to do anything here
            return;
        } else if let ty::UpvarCapture::ByRef(curr_upvar_borrow) = curr_capture_info.capture_kind {
            // Use the same region as the current capture information
            // Doesn't matter since only one of the UpvarBorrow will be used.
            let new_upvar_borrow = ty::UpvarBorrow { kind, region: curr_upvar_borrow.region };

            let capture_info = ty::CaptureInfo {
                expr_id: Some(place_with_id.hir_id),
                capture_kind: ty::UpvarCapture::ByRef(new_upvar_borrow),
            };
            let updated_info = self.fcx.determine_capture_info(curr_capture_info, capture_info);
            self.capture_information[&place_with_id.place] = updated_info;
        };
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

    fn init_capture_info_for_place(&mut self, place_with_id: &PlaceWithHirId<'tcx>) {
        if let PlaceBase::Upvar(upvar_id) = place_with_id.place.base {
            assert_eq!(self.closure_def_id.expect_local(), upvar_id.closure_expr_id);

            debug!("Capturing new place {:?}", place_with_id);

            let capture_kind =
                self.fcx.init_capture_kind(self.capture_clause, upvar_id, self.closure_span);
            let expr_id = Some(place_with_id.hir_id);
            let capture_info = ty::CaptureInfo { expr_id, capture_kind };

            self.capture_information.insert(place_with_id.place.clone(), capture_info);
        } else {
            debug!("Not upvar: {:?}", place_with_id);
        }
    }
}

impl<'a, 'tcx> euv::Delegate<'tcx> for InferBorrowKind<'a, 'tcx> {
    fn consume(&mut self, place_with_id: &PlaceWithHirId<'tcx>, mode: euv::ConsumeMode) {
        debug!("consume(place_with_id={:?},mode={:?})", place_with_id, mode);

        if !self.capture_information.contains_key(&place_with_id.place) {
            self.init_capture_info_for_place(place_with_id);
        }

        self.adjust_upvar_borrow_kind_for_consume(place_with_id, mode);
    }

    fn borrow(&mut self, place_with_id: &PlaceWithHirId<'tcx>, bk: ty::BorrowKind) {
        debug!("borrow(place_with_id={:?}, bk={:?})", place_with_id, bk);

        if !self.capture_information.contains_key(&place_with_id.place) {
            self.init_capture_info_for_place(place_with_id);
        }

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

        if !self.capture_information.contains_key(&assignee_place.place) {
            self.init_capture_info_for_place(assignee_place);
        }

        self.adjust_upvar_borrow_kind_for_mut(assignee_place);
    }
}

fn var_name(tcx: TyCtxt<'_>, var_hir_id: hir::HirId) -> Symbol {
    tcx.hir().name(var_hir_id)
}
