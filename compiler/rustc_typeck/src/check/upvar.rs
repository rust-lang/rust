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
use rustc_middle::hir::place::{Place, PlaceBase, PlaceWithHirId, ProjectionKind};
use rustc_middle::ty::{self, Ty, TyCtxt, UpvarSubsts};
use rustc_span::sym;
use rustc_span::{Span, Symbol};

/// Describe the relationship between the paths of two places
/// eg:
/// - `foo` is ancestor of `foo.bar.baz`
/// - `foo.bar.baz` is an descendant of `foo.bar`
/// - `foo.bar` and `foo.baz` are divergent
enum PlaceAncestryRelation {
    Ancestor,
    Descendant,
    Divergent,
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

        let mut capture_information: FxIndexMap<Place<'tcx>, ty::CaptureInfo<'tcx>> =
            Default::default();
        if !self.tcx.features().capture_disjoint_fields {
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

        debug!(
            "For closure={:?}, capture_information={:#?}",
            closure_def_id, delegate.capture_information
        );
        self.log_capture_analysis_first_pass(closure_def_id, &delegate.capture_information, span);

        if let Some(closure_substs) = infer_kind {
            // Unify the (as yet unbound) type variable in the closure
            // substs with the kind we inferred.
            let inferred_kind = delegate.current_closure_kind;
            let closure_kind_ty = closure_substs.as_closure().kind_ty();
            self.demand_eqtype(span, inferred_kind.to_ty(self.tcx), closure_kind_ty);

            // If we have an origin, store it.
            if let Some(origin) = delegate.current_origin.clone() {
                let origin = if self.tcx.features().capture_disjoint_fields {
                    origin
                } else {
                    // FIXME(project-rfc-2229#26): Once rust-lang#80092 is merged, we should restrict the
                    // precision of origin as well. Otherwise, this will cause issues when project-rfc-2229#26
                    // is fixed as we might see Index projections in the origin, which we can't print because
                    // we don't store enough information.
                    (origin.0, Place { projections: vec![], ..origin.1 })
                };

                self.typeck_results
                    .borrow_mut()
                    .closure_kind_origins_mut()
                    .insert(closure_hir_id, origin);
            }
        }

        self.compute_min_captures(closure_def_id, delegate);
        self.log_closure_min_capture_info(closure_def_id, span);

        self.min_captures_to_closure_captures_bridge(closure_def_id);

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
        let final_upvar_tys = self.final_upvar_tys(closure_def_id);
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
    fn final_upvar_tys(&self, closure_id: DefId) -> Vec<Ty<'tcx>> {
        // Presently an unboxed closure type cannot "escape" out of a
        // function, so we will only encounter ones that originated in the
        // local crate or were inlined into it along with some function.
        // This may change if abstract return types of some sort are
        // implemented.
        let tcx = self.tcx;

        self.typeck_results
            .borrow()
            .closure_min_captures_flattened(closure_id)
            .map(|captured_place| {
                let upvar_ty = captured_place.place.ty();
                let capture = captured_place.info.capture_kind;

                debug!(
                    "place={:?} upvar_ty={:?} capture={:?}",
                    captured_place.place, upvar_ty, capture
                );

                match capture {
                    ty::UpvarCapture::ByValue(_) => upvar_ty,
                    ty::UpvarCapture::ByRef(borrow) => tcx.mk_ref(
                        borrow.region,
                        ty::TypeAndMut { ty: upvar_ty, mutbl: borrow.kind.to_mutbl_lossy() },
                    ),
                }
            })
            .collect()
    }

    /// Bridge for closure analysis
    /// ----------------------------
    ///
    /// For closure with DefId `c`, the bridge converts structures required for supporting RFC 2229,
    /// to structures currently used in the compiler for handling closure captures.
    ///
    /// For example the following structure will be converted:
    ///
    /// closure_min_captures
    /// foo -> [ {foo.x, ImmBorrow}, {foo.y, MutBorrow} ]
    /// bar -> [ {bar.z, ByValue}, {bar.q, MutBorrow} ]
    ///
    /// to
    ///
    /// 1. closure_captures
    /// foo -> UpvarId(foo, c), bar -> UpvarId(bar, c)
    ///
    /// 2. upvar_capture_map
    /// UpvarId(foo,c) -> MutBorrow, UpvarId(bar, c) -> ByValue
    fn min_captures_to_closure_captures_bridge(&self, closure_def_id: DefId) {
        let mut closure_captures: FxIndexMap<hir::HirId, ty::UpvarId> = Default::default();
        let mut upvar_capture_map = ty::UpvarCaptureMap::default();

        if let Some(min_captures) =
            self.typeck_results.borrow().closure_min_captures.get(&closure_def_id)
        {
            for (var_hir_id, min_list) in min_captures.iter() {
                for captured_place in min_list {
                    let place = &captured_place.place;
                    let capture_info = captured_place.info;

                    let upvar_id = match place.base {
                        PlaceBase::Upvar(upvar_id) => upvar_id,
                        base => bug!("Expected upvar, found={:?}", base),
                    };

                    assert_eq!(upvar_id.var_path.hir_id, *var_hir_id);
                    assert_eq!(upvar_id.closure_expr_id, closure_def_id.expect_local());

                    closure_captures.insert(*var_hir_id, upvar_id);

                    let new_capture_kind =
                        if let Some(capture_kind) = upvar_capture_map.get(&upvar_id) {
                            // upvar_capture_map only stores the UpvarCapture (CaptureKind),
                            // so we create a fake capture info with no expression.
                            let fake_capture_info =
                                ty::CaptureInfo { expr_id: None, capture_kind: *capture_kind };
                            determine_capture_info(fake_capture_info, capture_info).capture_kind
                        } else {
                            capture_info.capture_kind
                        };
                    upvar_capture_map.insert(upvar_id, new_capture_kind);
                }
            }
        }
        debug!("For closure_def_id={:?}, closure_captures={:#?}", closure_def_id, closure_captures);
        debug!(
            "For closure_def_id={:?}, upvar_capture_map={:#?}",
            closure_def_id, upvar_capture_map
        );

        if !closure_captures.is_empty() {
            self.typeck_results
                .borrow_mut()
                .closure_captures
                .insert(closure_def_id, closure_captures);

            self.typeck_results.borrow_mut().upvar_capture_map.extend(upvar_capture_map);
        }
    }

    /// Analyzes the information collected by `InferBorrowKind` to compute the min number of
    /// Places (and corresponding capture kind) that we need to keep track of to support all
    /// the required captured paths.
    ///
    /// Eg:
    /// ```rust,no_run
    /// struct Point { x: i32, y: i32 }
    ///
    /// let s: String;  // hir_id_s
    /// let mut p: Point; // his_id_p
    /// let c = || {
    ///        println!("{}", s);  // L1
    ///        p.x += 10;  // L2
    ///        println!("{}" , p.y) // L3
    ///        println!("{}", p) // L4
    ///        drop(s);   // L5
    /// };
    /// ```
    /// and let hir_id_L1..5 be the expressions pointing to use of a captured variable on
    /// the lines L1..5 respectively.
    ///
    /// InferBorrowKind results in a structure like this:
    ///
    /// ```
    /// {
    ///       Place(base: hir_id_s, projections: [], ....) -> (hir_id_L5, ByValue),
    ///       Place(base: hir_id_p, projections: [Field(0, 0)], ...) -> (hir_id_L2, ByRef(MutBorrow))
    ///       Place(base: hir_id_p, projections: [Field(1, 0)], ...) -> (hir_id_L3, ByRef(ImmutBorrow))
    ///       Place(base: hir_id_p, projections: [], ...) -> (hir_id_L4, ByRef(ImmutBorrow))
    /// ```
    ///
    /// After the min capture analysis, we get:
    /// ```
    /// {
    ///       hir_id_s -> [
    ///            Place(base: hir_id_s, projections: [], ....) -> (hir_id_L4, ByValue)
    ///       ],
    ///       hir_id_p -> [
    ///            Place(base: hir_id_p, projections: [], ...) -> (hir_id_L2, ByRef(MutBorrow)),
    ///       ],
    /// ```
    fn compute_min_captures(
        &self,
        closure_def_id: DefId,
        inferred_info: InferBorrowKind<'_, 'tcx>,
    ) {
        let mut root_var_min_capture_list: ty::RootVariableMinCaptureList<'_> = Default::default();

        for (place, capture_info) in inferred_info.capture_information.into_iter() {
            let var_hir_id = match place.base {
                PlaceBase::Upvar(upvar_id) => upvar_id.var_path.hir_id,
                base => bug!("Expected upvar, found={:?}", base),
            };

            // Arrays are captured in entirety, drop Index projections and projections
            // after Index projections.
            let first_index_projection =
                place.projections.split(|proj| ProjectionKind::Index == proj.kind).next();
            let place = Place {
                base_ty: place.base_ty,
                base: place.base,
                projections: first_index_projection.map_or(Vec::new(), |p| p.to_vec()),
            };

            let min_cap_list = match root_var_min_capture_list.get_mut(&var_hir_id) {
                None => {
                    let min_cap_list = vec![ty::CapturedPlace { place, info: capture_info }];
                    root_var_min_capture_list.insert(var_hir_id, min_cap_list);
                    continue;
                }
                Some(min_cap_list) => min_cap_list,
            };

            // Go through each entry in the current list of min_captures
            // - if ancestor is found, update it's capture kind to account for current place's
            // capture information.
            //
            // - if descendant is found, remove it from the list, and update the current place's
            // capture information to account for the descendants's capture kind.
            //
            // We can never be in a case where the list contains both an ancestor and a descendant
            // Also there can only be ancestor but in case of descendants there might be
            // multiple.

            let mut descendant_found = false;
            let mut updated_capture_info = capture_info;
            min_cap_list.retain(|possible_descendant| {
                match determine_place_ancestry_relation(&place, &possible_descendant.place) {
                    // current place is ancestor of possible_descendant
                    PlaceAncestryRelation::Ancestor => {
                        descendant_found = true;
                        updated_capture_info =
                            determine_capture_info(updated_capture_info, possible_descendant.info);
                        false
                    }

                    _ => true,
                }
            });

            let mut ancestor_found = false;
            if !descendant_found {
                for possible_ancestor in min_cap_list.iter_mut() {
                    match determine_place_ancestry_relation(&place, &possible_ancestor.place) {
                        // current place is descendant of possible_ancestor
                        PlaceAncestryRelation::Descendant => {
                            ancestor_found = true;
                            possible_ancestor.info =
                                determine_capture_info(possible_ancestor.info, capture_info);

                            // Only one ancestor of the current place will be in the list.
                            break;
                        }
                        _ => {}
                    }
                }
            }

            // Only need to insert when we don't have an ancestor in the existing min capture list
            if !ancestor_found {
                let captured_place =
                    ty::CapturedPlace { place: place.clone(), info: updated_capture_info };
                min_cap_list.push(captured_place);
            }
        }

        debug!("For closure={:?}, min_captures={:#?}", closure_def_id, root_var_min_capture_list);

        if !root_var_min_capture_list.is_empty() {
            self.typeck_results
                .borrow_mut()
                .closure_min_captures
                .insert(closure_def_id, root_var_min_capture_list);
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

    fn log_capture_analysis_first_pass(
        &self,
        closure_def_id: rustc_hir::def_id::DefId,
        capture_information: &FxIndexMap<Place<'tcx>, ty::CaptureInfo<'tcx>>,
        closure_span: Span,
    ) {
        if self.should_log_capture_analysis(closure_def_id) {
            let mut diag =
                self.tcx.sess.struct_span_err(closure_span, "First Pass analysis includes:");
            for (place, capture_info) in capture_information {
                let capture_str = construct_capture_info_string(self.tcx, place, capture_info);
                let output_str = format!("Capturing {}", capture_str);

                let span = capture_info.expr_id.map_or(closure_span, |e| self.tcx.hir().span(e));
                diag.span_note(span, &output_str);
            }
            diag.emit();
        }
    }

    fn log_closure_min_capture_info(&self, closure_def_id: DefId, closure_span: Span) {
        if self.should_log_capture_analysis(closure_def_id) {
            if let Some(min_captures) =
                self.typeck_results.borrow().closure_min_captures.get(&closure_def_id)
            {
                let mut diag =
                    self.tcx.sess.struct_span_err(closure_span, "Min Capture analysis includes:");

                for (_, min_captures_for_var) in min_captures {
                    for capture in min_captures_for_var {
                        let place = &capture.place;
                        let capture_info = &capture.info;

                        let capture_str =
                            construct_capture_info_string(self.tcx, place, capture_info);
                        let output_str = format!("Min Capture {}", capture_str);

                        let span =
                            capture_info.expr_id.map_or(closure_span, |e| self.tcx.hir().span(e));
                        diag.span_note(span, &output_str);
                    }
                }
                diag.emit();
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
    current_origin: Option<(Span, Place<'tcx>)>,

    /// For each Place that is captured by the closure, we track the minimal kind of
    /// access we need (ref, ref mut, move, etc) and the expression that resulted in such access.
    ///
    /// Consider closure where s.str1 is captured via an ImmutableBorrow and
    /// s.str2 via a MutableBorrow
    ///
    /// ```rust,no_run
    /// struct SomeStruct { str1: String, str2: String }
    ///
    /// // Assume that the HirId for the variable definition is `V1`
    /// let mut s = SomeStruct { str1: format!("s1"), str2: format!("s2") }
    ///
    /// let fix_s = |new_s2| {
    ///     // Assume that the HirId for the expression `s.str1` is `E1`
    ///     println!("Updating SomeStruct with str1=", s.str1);
    ///     // Assume that the HirId for the expression `*s.str2` is `E2`
    ///     s.str2 = new_s2;
    /// };
    /// ```
    ///
    /// For closure `fix_s`, (at a high level) the map contains
    ///
    /// Place { V1, [ProjectionKind::Field(Index=0, Variant=0)] } : CaptureKind { E1, ImmutableBorrow }
    /// Place { V1, [ProjectionKind::Field(Index=1, Variant=0)] } : CaptureKind { E2, MutableBorrow }
    capture_information: FxIndexMap<Place<'tcx>, ty::CaptureInfo<'tcx>>,
}

impl<'a, 'tcx> InferBorrowKind<'a, 'tcx> {
    fn adjust_upvar_borrow_kind_for_consume(
        &mut self,
        place_with_id: &PlaceWithHirId<'tcx>,
        diag_expr_id: hir::HirId,
        mode: euv::ConsumeMode,
    ) {
        debug!(
            "adjust_upvar_borrow_kind_for_consume(place_with_id={:?}, diag_expr_id={:?}, mode={:?})",
            place_with_id, diag_expr_id, mode
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

        let usage_span = tcx.hir().span(diag_expr_id);

        // To move out of an upvar, this must be a FnOnce closure
        self.adjust_closure_kind(
            upvar_id.closure_expr_id,
            ty::ClosureKind::FnOnce,
            usage_span,
            place_with_id.place.clone(),
        );

        let capture_info = ty::CaptureInfo {
            expr_id: Some(diag_expr_id),
            capture_kind: ty::UpvarCapture::ByValue(Some(usage_span)),
        };

        let curr_info = self.capture_information[&place_with_id.place];
        let updated_info = determine_capture_info(curr_info, capture_info);

        self.capture_information[&place_with_id.place] = updated_info;
    }

    /// Indicates that `place_with_id` is being directly mutated (e.g., assigned
    /// to). If the place is based on a by-ref upvar, this implies that
    /// the upvar must be borrowed using an `&mut` borrow.
    fn adjust_upvar_borrow_kind_for_mut(
        &mut self,
        place_with_id: &PlaceWithHirId<'tcx>,
        diag_expr_id: hir::HirId,
    ) {
        debug!(
            "adjust_upvar_borrow_kind_for_mut(place_with_id={:?}, diag_expr_id={:?})",
            place_with_id, diag_expr_id
        );

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
            self.adjust_upvar_deref(place_with_id, diag_expr_id, borrow_kind);
        }
    }

    fn adjust_upvar_borrow_kind_for_unique(
        &mut self,
        place_with_id: &PlaceWithHirId<'tcx>,
        diag_expr_id: hir::HirId,
    ) {
        debug!(
            "adjust_upvar_borrow_kind_for_unique(place_with_id={:?}, diag_expr_id={:?})",
            place_with_id, diag_expr_id
        );

        if let PlaceBase::Upvar(_) = place_with_id.place.base {
            if place_with_id.place.deref_tys().any(ty::TyS::is_unsafe_ptr) {
                // Raw pointers don't inherit mutability.
                return;
            }
            // for a borrowed pointer to be unique, its base must be unique
            self.adjust_upvar_deref(place_with_id, diag_expr_id, ty::UniqueImmBorrow);
        }
    }

    fn adjust_upvar_deref(
        &mut self,
        place_with_id: &PlaceWithHirId<'tcx>,
        diag_expr_id: hir::HirId,
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
        self.adjust_upvar_borrow_kind(place_with_id, diag_expr_id, borrow_kind);

        if let PlaceBase::Upvar(upvar_id) = place_with_id.place.base {
            self.adjust_closure_kind(
                upvar_id.closure_expr_id,
                ty::ClosureKind::FnMut,
                tcx.hir().span(diag_expr_id),
                place_with_id.place.clone(),
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
        diag_expr_id: hir::HirId,
        kind: ty::BorrowKind,
    ) {
        let curr_capture_info = self.capture_information[&place_with_id.place];

        debug!(
            "adjust_upvar_borrow_kind(place={:?}, diag_expr_id={:?}, capture_info={:?}, kind={:?})",
            place_with_id, diag_expr_id, curr_capture_info, kind
        );

        if let ty::UpvarCapture::ByValue(_) = curr_capture_info.capture_kind {
            // It's already captured by value, we don't need to do anything here
            return;
        } else if let ty::UpvarCapture::ByRef(curr_upvar_borrow) = curr_capture_info.capture_kind {
            // Use the same region as the current capture information
            // Doesn't matter since only one of the UpvarBorrow will be used.
            let new_upvar_borrow = ty::UpvarBorrow { kind, region: curr_upvar_borrow.region };

            let capture_info = ty::CaptureInfo {
                expr_id: Some(diag_expr_id),
                capture_kind: ty::UpvarCapture::ByRef(new_upvar_borrow),
            };
            let updated_info = determine_capture_info(curr_capture_info, capture_info);
            self.capture_information[&place_with_id.place] = updated_info;
        };
    }

    fn adjust_closure_kind(
        &mut self,
        closure_id: LocalDefId,
        new_kind: ty::ClosureKind,
        upvar_span: Span,
        place: Place<'tcx>,
    ) {
        debug!(
            "adjust_closure_kind(closure_id={:?}, new_kind={:?}, upvar_span={:?}, place={:?})",
            closure_id, new_kind, upvar_span, place
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
                self.current_origin = Some((upvar_span, place));
            }
        }
    }

    fn init_capture_info_for_place(
        &mut self,
        place_with_id: &PlaceWithHirId<'tcx>,
        diag_expr_id: hir::HirId,
    ) {
        if let PlaceBase::Upvar(upvar_id) = place_with_id.place.base {
            assert_eq!(self.closure_def_id.expect_local(), upvar_id.closure_expr_id);

            let capture_kind =
                self.fcx.init_capture_kind(self.capture_clause, upvar_id, self.closure_span);

            let expr_id = Some(diag_expr_id);
            let capture_info = ty::CaptureInfo { expr_id, capture_kind };

            debug!("Capturing new place {:?}, capture_info={:?}", place_with_id, capture_info);

            self.capture_information.insert(place_with_id.place.clone(), capture_info);
        } else {
            debug!("Not upvar: {:?}", place_with_id);
        }
    }
}

impl<'a, 'tcx> euv::Delegate<'tcx> for InferBorrowKind<'a, 'tcx> {
    fn consume(
        &mut self,
        place_with_id: &PlaceWithHirId<'tcx>,
        diag_expr_id: hir::HirId,
        mode: euv::ConsumeMode,
    ) {
        debug!(
            "consume(place_with_id={:?}, diag_expr_id={:?}, mode={:?})",
            place_with_id, diag_expr_id, mode
        );
        if !self.capture_information.contains_key(&place_with_id.place) {
            self.init_capture_info_for_place(place_with_id, diag_expr_id);
        }

        self.adjust_upvar_borrow_kind_for_consume(place_with_id, diag_expr_id, mode);
    }

    fn borrow(
        &mut self,
        place_with_id: &PlaceWithHirId<'tcx>,
        diag_expr_id: hir::HirId,
        bk: ty::BorrowKind,
    ) {
        debug!(
            "borrow(place_with_id={:?}, diag_expr_id={:?}, bk={:?})",
            place_with_id, diag_expr_id, bk
        );

        if !self.capture_information.contains_key(&place_with_id.place) {
            self.init_capture_info_for_place(place_with_id, diag_expr_id);
        }

        match bk {
            ty::ImmBorrow => {}
            ty::UniqueImmBorrow => {
                self.adjust_upvar_borrow_kind_for_unique(&place_with_id, diag_expr_id);
            }
            ty::MutBorrow => {
                self.adjust_upvar_borrow_kind_for_mut(&place_with_id, diag_expr_id);
            }
        }
    }

    fn mutate(&mut self, assignee_place: &PlaceWithHirId<'tcx>, diag_expr_id: hir::HirId) {
        debug!("mutate(assignee_place={:?}, diag_expr_id={:?})", assignee_place, diag_expr_id);

        if !self.capture_information.contains_key(&assignee_place.place) {
            self.init_capture_info_for_place(assignee_place, diag_expr_id);
        }

        self.adjust_upvar_borrow_kind_for_mut(assignee_place, diag_expr_id);
    }
}

fn construct_capture_info_string(
    tcx: TyCtxt<'_>,
    place: &Place<'tcx>,
    capture_info: &ty::CaptureInfo<'tcx>,
) -> String {
    let variable_name = match place.base {
        PlaceBase::Upvar(upvar_id) => var_name(tcx, upvar_id.var_path.hir_id).to_string(),
        _ => bug!("Capture_information should only contain upvars"),
    };

    let mut projections_str = String::new();
    for (i, item) in place.projections.iter().enumerate() {
        let proj = match item.kind {
            ProjectionKind::Field(a, b) => format!("({:?}, {:?})", a, b),
            ProjectionKind::Deref => String::from("Deref"),
            ProjectionKind::Index => String::from("Index"),
            ProjectionKind::Subslice => String::from("Subslice"),
        };
        if i != 0 {
            projections_str.push_str(",");
        }
        projections_str.push_str(proj.as_str());
    }

    let capture_kind_str = match capture_info.capture_kind {
        ty::UpvarCapture::ByValue(_) => "ByValue".into(),
        ty::UpvarCapture::ByRef(borrow) => format!("{:?}", borrow.kind),
    };
    format!("{}[{}] -> {}", variable_name, projections_str, capture_kind_str)
}

fn var_name(tcx: TyCtxt<'_>, var_hir_id: hir::HirId) -> Symbol {
    tcx.hir().name(var_hir_id)
}

/// Helper function to determine if we need to escalate CaptureKind from
/// CaptureInfo A to B and returns the escalated CaptureInfo.
/// (Note: CaptureInfo contains CaptureKind and an expression that led to capture it in that way)
///
/// If both `CaptureKind`s are considered equivalent, then the CaptureInfo is selected based
/// on the `CaptureInfo` containing an associated expression id.
///
/// If both the CaptureKind and Expression are considered to be equivalent,
/// then `CaptureInfo` A is preferred. This can be useful in cases where we want to priortize
/// expressions reported back to the user as part of diagnostics based on which appears earlier
/// in the closure. This can be acheived simply by calling
/// `determine_capture_info(existing_info, current_info)`. This works out because the
/// expressions that occur earlier in the closure body than the current expression are processed before.
/// Consider the following example
/// ```rust,no_run
/// struct Point { x: i32, y: i32 }
/// let mut p: Point { x: 10, y: 10 };
///
/// let c = || {
///     p.x     += 10;
/// // ^ E1 ^
///     // ...
///     // More code
///     // ...
///     p.x += 10; // E2
/// // ^ E2 ^
/// };
/// ```
/// `CaptureKind` associated with both `E1` and `E2` will be ByRef(MutBorrow),
/// and both have an expression associated, however for diagnostics we prefer reporting
/// `E1` since it appears earlier in the closure body. When `E2` is being processed we
/// would've already handled `E1`, and have an existing capture_information for it.
/// Calling `determine_capture_info(existing_info_e1, current_info_e2)` will return
/// `existing_info_e1` in this case, allowing us to point to `E1` in case of diagnostics.
fn determine_capture_info(
    capture_info_a: ty::CaptureInfo<'tcx>,
    capture_info_b: ty::CaptureInfo<'tcx>,
) -> ty::CaptureInfo<'tcx> {
    // If the capture kind is equivalent then, we don't need to escalate and can compare the
    // expressions.
    let eq_capture_kind = match (capture_info_a.capture_kind, capture_info_b.capture_kind) {
        (ty::UpvarCapture::ByValue(_), ty::UpvarCapture::ByValue(_)) => {
            // We don't need to worry about the spans being ignored here.
            //
            // The expr_id in capture_info corresponds to the span that is stored within
            // ByValue(span) and therefore it gets handled with priortizing based on
            // expressions below.
            true
        }
        (ty::UpvarCapture::ByRef(ref_a), ty::UpvarCapture::ByRef(ref_b)) => {
            ref_a.kind == ref_b.kind
        }
        (ty::UpvarCapture::ByValue(_), _) | (ty::UpvarCapture::ByRef(_), _) => false,
    };

    if eq_capture_kind {
        match (capture_info_a.expr_id, capture_info_b.expr_id) {
            (Some(_), _) | (None, None) => capture_info_a,
            (None, Some(_)) => capture_info_b,
        }
    } else {
        // We select the CaptureKind which ranks higher based the following priority order:
        // ByValue > MutBorrow > UniqueImmBorrow > ImmBorrow
        match (capture_info_a.capture_kind, capture_info_b.capture_kind) {
            (ty::UpvarCapture::ByValue(_), _) => capture_info_a,
            (_, ty::UpvarCapture::ByValue(_)) => capture_info_b,
            (ty::UpvarCapture::ByRef(ref_a), ty::UpvarCapture::ByRef(ref_b)) => {
                match (ref_a.kind, ref_b.kind) {
                    // Take LHS:
                    (ty::UniqueImmBorrow | ty::MutBorrow, ty::ImmBorrow)
                    | (ty::MutBorrow, ty::UniqueImmBorrow) => capture_info_a,

                    // Take RHS:
                    (ty::ImmBorrow, ty::UniqueImmBorrow | ty::MutBorrow)
                    | (ty::UniqueImmBorrow, ty::MutBorrow) => capture_info_b,

                    (ty::ImmBorrow, ty::ImmBorrow)
                    | (ty::UniqueImmBorrow, ty::UniqueImmBorrow)
                    | (ty::MutBorrow, ty::MutBorrow) => {
                        bug!("Expected unequal capture kinds");
                    }
                }
            }
        }
    }
}

/// Determines the Ancestry relationship of Place A relative to Place B
///
/// `PlaceAncestryRelation::Ancestor` implies Place A is ancestor of Place B
/// `PlaceAncestryRelation::Descendant` implies Place A is descendant of Place B
/// `PlaceAncestryRelation::Divergent` implies neither of them is the ancestor of the other.
fn determine_place_ancestry_relation(
    place_a: &Place<'tcx>,
    place_b: &Place<'tcx>,
) -> PlaceAncestryRelation {
    // If Place A and Place B, don't start off from the same root variable, they are divergent.
    if place_a.base != place_b.base {
        return PlaceAncestryRelation::Divergent;
    }

    // Assume of length of projections_a = n
    let projections_a = &place_a.projections;

    // Assume of length of projections_b = m
    let projections_b = &place_b.projections;

    let mut same_initial_projections = true;

    for (proj_a, proj_b) in projections_a.iter().zip(projections_b.iter()) {
        if proj_a != proj_b {
            same_initial_projections = false;
            break;
        }
    }

    if same_initial_projections {
        // First min(n, m) projections are the same
        // Select Ancestor/Descendant
        if projections_b.len() >= projections_a.len() {
            PlaceAncestryRelation::Ancestor
        } else {
            PlaceAncestryRelation::Descendant
        }
    } else {
        PlaceAncestryRelation::Divergent
    }
}
