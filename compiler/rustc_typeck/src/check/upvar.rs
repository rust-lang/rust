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
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_infer::infer::UpvarRegion;
use rustc_middle::hir::place::{Place, PlaceBase, PlaceWithHirId, Projection, ProjectionKind};
use rustc_middle::mir::FakeReadCause;
use rustc_middle::ty::{self, TraitRef, Ty, TyCtxt, TypeckResults, UpvarSubsts};
use rustc_session::lint;
use rustc_span::sym;
use rustc_span::{MultiSpan, Span, Symbol};
use rustc_trait_selection::traits::{Obligation, ObligationCause};

use rustc_data_structures::stable_set::FxHashSet;
use rustc_index::vec::Idx;
use rustc_target::abi::VariantIdx;

use std::iter;

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

/// Intermediate format to store a captured `Place` and associated `ty::CaptureInfo`
/// during capture analysis. Information in this map feeds into the minimum capture
/// analysis pass.
type InferredCaptureInformation<'tcx> = FxIndexMap<Place<'tcx>, ty::CaptureInfo<'tcx>>;

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
            self.fcx.analyze_closure(expr.hir_id, expr.span, body_id, body, cc);
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
        body_id: hir::BodyId,
        body: &'tcx hir::Body<'tcx>,
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

        let body_owner_def_id = self.tcx.hir().body_owner_def_id(body.id());
        assert_eq!(body_owner_def_id.to_def_id(), closure_def_id);
        let mut delegate = InferBorrowKind {
            fcx: self,
            closure_def_id,
            closure_span: span,
            capture_clause,
            current_closure_kind: ty::ClosureKind::LATTICE_BOTTOM,
            current_origin: None,
            capture_information: Default::default(),
            fake_reads: Default::default(),
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

        self.compute_min_captures(closure_def_id, delegate.capture_information);

        let closure_hir_id = self.tcx.hir().local_def_id_to_hir_id(local_def_id);

        if should_do_disjoint_capture_migration_analysis(self.tcx, closure_hir_id) {
            self.perform_2229_migration_anaysis(closure_def_id, body_id, capture_clause, span);
        }

        // We now fake capture information for all variables that are mentioned within the closure
        // We do this after handling migrations so that min_captures computes before
        if !self.tcx.features().capture_disjoint_fields {
            let mut capture_information: InferredCaptureInformation<'tcx> = Default::default();

            if let Some(upvars) = self.tcx.upvars_mentioned(closure_def_id) {
                for var_hir_id in upvars.keys() {
                    let place = self.place_for_root_variable(local_def_id, *var_hir_id);

                    debug!("seed place {:?}", place);

                    let upvar_id = ty::UpvarId::new(*var_hir_id, local_def_id);
                    let capture_kind =
                        self.init_capture_kind_for_place(&place, capture_clause, upvar_id, span);
                    let fake_info = ty::CaptureInfo {
                        capture_kind_expr_id: None,
                        path_expr_id: None,
                        capture_kind,
                    };

                    capture_information.insert(place, fake_info);
                }
            }

            // This will update the min captures based on this new fake information.
            self.compute_min_captures(closure_def_id, capture_information);
        }

        if let Some(closure_substs) = infer_kind {
            // Unify the (as yet unbound) type variable in the closure
            // substs with the kind we inferred.
            let inferred_kind = delegate.current_closure_kind;
            let closure_kind_ty = closure_substs.as_closure().kind_ty();
            self.demand_eqtype(span, inferred_kind.to_ty(self.tcx), closure_kind_ty);

            // If we have an origin, store it.
            if let Some(origin) = delegate.current_origin.clone() {
                let origin = if self.tcx.features().capture_disjoint_fields {
                    (origin.0, restrict_capture_precision(origin.1))
                } else {
                    (origin.0, Place { projections: vec![], ..origin.1 })
                };

                self.typeck_results
                    .borrow_mut()
                    .closure_kind_origins_mut()
                    .insert(closure_hir_id, origin);
            }
        }

        self.log_closure_min_capture_info(closure_def_id, span);

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

        let fake_reads = delegate
            .fake_reads
            .into_iter()
            .map(|(place, cause, hir_id)| (place, cause, hir_id))
            .collect();
        self.typeck_results.borrow_mut().closure_fake_reads.insert(closure_def_id, fake_reads);

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
        self.typeck_results
            .borrow()
            .closure_min_captures_flattened(closure_id)
            .map(|captured_place| {
                let upvar_ty = captured_place.place.ty();
                let capture = captured_place.info.capture_kind;

                debug!(
                    "final_upvar_tys: place={:?} upvar_ty={:?} capture={:?}, mutability={:?}",
                    captured_place.place, upvar_ty, capture, captured_place.mutability,
                );

                match capture {
                    ty::UpvarCapture::ByValue(_) => upvar_ty,
                    ty::UpvarCapture::ByRef(borrow) => self.tcx.mk_ref(
                        borrow.region,
                        ty::TypeAndMut { ty: upvar_ty, mutbl: borrow.kind.to_mutbl_lossy() },
                    ),
                }
            })
            .collect()
    }

    /// Analyzes the information collected by `InferBorrowKind` to compute the min number of
    /// Places (and corresponding capture kind) that we need to keep track of to support all
    /// the required captured paths.
    ///
    ///
    /// Note: If this function is called multiple times for the same closure, it will update
    ///       the existing min_capture map that is stored in TypeckResults.
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
    /// ```text
    /// {
    ///       Place(base: hir_id_s, projections: [], ....) -> {
    ///                                                            capture_kind_expr: hir_id_L5,
    ///                                                            path_expr_id: hir_id_L5,
    ///                                                            capture_kind: ByValue
    ///                                                       },
    ///       Place(base: hir_id_p, projections: [Field(0, 0)], ...) -> {
    ///                                                                     capture_kind_expr: hir_id_L2,
    ///                                                                     path_expr_id: hir_id_L2,
    ///                                                                     capture_kind: ByValue
    ///                                                                 },
    ///       Place(base: hir_id_p, projections: [Field(1, 0)], ...) -> {
    ///                                                                     capture_kind_expr: hir_id_L3,
    ///                                                                     path_expr_id: hir_id_L3,
    ///                                                                     capture_kind: ByValue
    ///                                                                 },
    ///       Place(base: hir_id_p, projections: [], ...) -> {
    ///                                                          capture_kind_expr: hir_id_L4,
    ///                                                          path_expr_id: hir_id_L4,
    ///                                                          capture_kind: ByValue
    ///                                                      },
    /// ```
    ///
    /// After the min capture analysis, we get:
    /// ```text
    /// {
    ///       hir_id_s -> [
    ///            Place(base: hir_id_s, projections: [], ....) -> {
    ///                                                                capture_kind_expr: hir_id_L5,
    ///                                                                path_expr_id: hir_id_L5,
    ///                                                                capture_kind: ByValue
    ///                                                            },
    ///       ],
    ///       hir_id_p -> [
    ///            Place(base: hir_id_p, projections: [], ...) -> {
    ///                                                               capture_kind_expr: hir_id_L2,
    ///                                                               path_expr_id: hir_id_L4,
    ///                                                               capture_kind: ByValue
    ///                                                           },
    ///       ],
    /// ```
    fn compute_min_captures(
        &self,
        closure_def_id: DefId,
        capture_information: InferredCaptureInformation<'tcx>,
    ) {
        if capture_information.is_empty() {
            return;
        }

        let mut typeck_results = self.typeck_results.borrow_mut();

        let mut root_var_min_capture_list =
            typeck_results.closure_min_captures.remove(&closure_def_id).unwrap_or_default();

        for (place, capture_info) in capture_information.into_iter() {
            let var_hir_id = match place.base {
                PlaceBase::Upvar(upvar_id) => upvar_id.var_path.hir_id,
                base => bug!("Expected upvar, found={:?}", base),
            };

            let place = restrict_capture_precision(place);

            let min_cap_list = match root_var_min_capture_list.get_mut(&var_hir_id) {
                None => {
                    let mutability = self.determine_capture_mutability(&typeck_results, &place);
                    let min_cap_list =
                        vec![ty::CapturedPlace { place, info: capture_info, mutability }];
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
                        let backup_path_expr_id = updated_capture_info.path_expr_id;

                        updated_capture_info =
                            determine_capture_info(updated_capture_info, possible_descendant.info);

                        // we need to keep the ancestor's `path_expr_id`
                        updated_capture_info.path_expr_id = backup_path_expr_id;
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
                            let backup_path_expr_id = possible_ancestor.info.path_expr_id;
                            possible_ancestor.info =
                                determine_capture_info(possible_ancestor.info, capture_info);

                            // we need to keep the ancestor's `path_expr_id`
                            possible_ancestor.info.path_expr_id = backup_path_expr_id;

                            // Only one ancestor of the current place will be in the list.
                            break;
                        }
                        _ => {}
                    }
                }
            }

            // Only need to insert when we don't have an ancestor in the existing min capture list
            if !ancestor_found {
                let mutability = self.determine_capture_mutability(&typeck_results, &place);
                let captured_place =
                    ty::CapturedPlace { place, info: updated_capture_info, mutability };
                min_cap_list.push(captured_place);
            }
        }

        debug!("For closure={:?}, min_captures={:#?}", closure_def_id, root_var_min_capture_list);
        typeck_results.closure_min_captures.insert(closure_def_id, root_var_min_capture_list);
    }

    /// Perform the migration analysis for RFC 2229, and emit lint
    /// `disjoint_capture_drop_reorder` if needed.
    fn perform_2229_migration_anaysis(
        &self,
        closure_def_id: DefId,
        body_id: hir::BodyId,
        capture_clause: hir::CaptureBy,
        span: Span,
    ) {
        let (need_migrations, reasons) = self.compute_2229_migrations(
            closure_def_id,
            span,
            capture_clause,
            self.typeck_results.borrow().closure_min_captures.get(&closure_def_id),
        );

        if !need_migrations.is_empty() {
            let (migration_string, migrated_variables_concat) =
                migration_suggestion_for_2229(self.tcx, &need_migrations);

            let local_def_id = closure_def_id.expect_local();
            let closure_hir_id = self.tcx.hir().local_def_id_to_hir_id(local_def_id);
            self.tcx.struct_span_lint_hir(
                lint::builtin::DISJOINT_CAPTURE_MIGRATION,
                closure_hir_id,
                span,
                |lint| {
                    let mut diagnostics_builder = lint.build(
                        format!(
                            "{} affected for closure because of `capture_disjoint_fields`",
                            reasons
                        )
                        .as_str(),
                    );
                    let closure_body_span = self.tcx.hir().span(body_id.hir_id);
                    let (sugg, app) =
                        match self.tcx.sess.source_map().span_to_snippet(closure_body_span) {
                            Ok(s) => {
                                let trimmed = s.trim_start();

                                // If the closure contains a block then replace the opening brace
                                // with "{ let _ = (..); "
                                let sugg = if let Some('{') = trimmed.chars().next() {
                                    format!("{{ {}; {}", migration_string, &trimmed[1..])
                                } else {
                                    format!("{{ {}; {} }}", migration_string, s)
                                };
                                (sugg, Applicability::MachineApplicable)
                            }
                            Err(_) => (migration_string.clone(), Applicability::HasPlaceholders),
                        };

                    let diagnostic_msg = format!(
                        "add a dummy let to cause {} to be fully captured",
                        migrated_variables_concat
                    );

                    diagnostics_builder.span_suggestion(
                        closure_body_span,
                        &diagnostic_msg,
                        sugg,
                        app,
                    );
                    diagnostics_builder.emit();
                },
            );
        }
    }

    /// Combines all the reasons for 2229 migrations
    fn compute_2229_migrations_reasons(
        &self,
        auto_trait_reasons: FxHashSet<&str>,
        drop_reason: bool,
    ) -> String {
        let mut reasons = String::new();

        if auto_trait_reasons.len() > 0 {
            reasons = format!(
                "{} trait implementation",
                auto_trait_reasons.clone().into_iter().collect::<Vec<&str>>().join(", ")
            );
        }

        if auto_trait_reasons.len() > 0 && drop_reason {
            reasons = format!("{}, and ", reasons);
        }

        if drop_reason {
            reasons = format!("{}drop order", reasons);
        }

        reasons
    }

    /// Returns true if `ty` may implement `trait_def_id`
    fn ty_impls_trait(
        &self,
        ty: Ty<'tcx>,
        cause: &ObligationCause<'tcx>,
        trait_def_id: DefId,
    ) -> bool {
        use crate::rustc_middle::ty::ToPredicate;
        use crate::rustc_middle::ty::WithConstness;
        use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
        let tcx = self.infcx.tcx;

        let trait_ref = TraitRef { def_id: trait_def_id, substs: tcx.mk_substs_trait(ty, &[]) };

        let obligation = Obligation::new(
            cause.clone(),
            self.param_env,
            trait_ref.without_const().to_predicate(tcx),
        );

        self.infcx.predicate_may_hold(&obligation)
    }

    /// Returns true if migration is needed for trait for the provided var_hir_id
    fn need_2229_migrations_for_trait(
        &self,
        min_captures: Option<&ty::RootVariableMinCaptureList<'tcx>>,
        var_hir_id: hir::HirId,
        check_trait: Option<DefId>,
    ) -> bool {
        let root_var_min_capture_list = if let Some(root_var_min_capture_list) =
            min_captures.and_then(|m| m.get(&var_hir_id))
        {
            root_var_min_capture_list
        } else {
            return false;
        };

        let ty = self.infcx.resolve_vars_if_possible(self.node_ty(var_hir_id));

        let cause = ObligationCause::misc(self.tcx.hir().span(var_hir_id), self.body_id);

        let obligation_should_hold = check_trait
            .map(|check_trait| self.ty_impls_trait(ty, &cause, check_trait))
            .unwrap_or(false);

        // Check whether catpured fields also implement the trait

        for capture in root_var_min_capture_list.iter() {
            let ty = capture.place.ty();

            let obligation_holds_for_capture = check_trait
                .map(|check_trait| self.ty_impls_trait(ty, &cause, check_trait))
                .unwrap_or(false);

            if !obligation_holds_for_capture && obligation_should_hold {
                return true;
            }
        }
        false
    }

    /// Figures out the list of root variables (and their types) that aren't completely
    /// captured by the closure when `capture_disjoint_fields` is enabled and auto-traits
    /// differ between the root variable and the captured paths.
    ///
    /// The output list would include a root variable if:
    /// - It would have been captured into the closure when `capture_disjoint_fields` wasn't
    ///   enabled, **and**
    /// - It wasn't completely captured by the closure, **and**
    /// - One of the paths captured does not implement all the auto-traits its root variable
    ///   implements.
    fn compute_2229_migrations_for_trait(
        &self,
        min_captures: Option<&ty::RootVariableMinCaptureList<'tcx>>,
        var_hir_id: hir::HirId,
    ) -> Option<FxHashSet<&str>> {
        let tcx = self.infcx.tcx;

        // Check whether catpured fields also implement the trait
        let mut auto_trait_reasons = FxHashSet::default();

        if self.need_2229_migrations_for_trait(
            min_captures,
            var_hir_id,
            tcx.lang_items().clone_trait(),
        ) {
            auto_trait_reasons.insert("`Clone`");
        }

        if self.need_2229_migrations_for_trait(
            min_captures,
            var_hir_id,
            tcx.lang_items().sync_trait(),
        ) {
            auto_trait_reasons.insert("`Sync`");
        }

        if self.need_2229_migrations_for_trait(
            min_captures,
            var_hir_id,
            tcx.lang_items().send_trait(),
        ) {
            auto_trait_reasons.insert("`Send`");
        }

        if self.need_2229_migrations_for_trait(
            min_captures,
            var_hir_id,
            tcx.lang_items().unpin_trait(),
        ) {
            auto_trait_reasons.insert("`Unpin`");
        }

        if self.need_2229_migrations_for_trait(
            min_captures,
            var_hir_id,
            tcx.lang_items().unwind_safe_trait(),
        ) {
            auto_trait_reasons.insert("`UnwindSafe`");
        }

        if self.need_2229_migrations_for_trait(
            min_captures,
            var_hir_id,
            tcx.lang_items().ref_unwind_safe_trait(),
        ) {
            auto_trait_reasons.insert("`RefUnwindSafe`");
        }

        if auto_trait_reasons.len() > 0 {
            return Some(auto_trait_reasons);
        }

        return None;
    }

    /// Figures out the list of root variables (and their types) that aren't completely
    /// captured by the closure when `capture_disjoint_fields` is enabled and drop order of
    /// some path starting at that root variable **might** be affected.
    ///
    /// The output list would include a root variable if:
    /// - It would have been moved into the closure when `capture_disjoint_fields` wasn't
    ///   enabled, **and**
    /// - It wasn't completely captured by the closure, **and**
    /// - One of the paths starting at this root variable, that is not captured needs Drop.
    ///
    /// This function only returns true for significant drops. A type is considerent to have a
    /// significant drop if it's Drop implementation is not annotated by `rustc_insignificant_dtor`.
    fn compute_2229_migrations_for_drop(
        &self,
        closure_def_id: DefId,
        closure_span: Span,
        min_captures: Option<&ty::RootVariableMinCaptureList<'tcx>>,
        closure_clause: hir::CaptureBy,
        var_hir_id: hir::HirId,
    ) -> bool {
        let ty = self.infcx.resolve_vars_if_possible(self.node_ty(var_hir_id));

        if !ty.has_significant_drop(self.tcx, self.tcx.param_env(closure_def_id.expect_local())) {
            return false;
        }

        let root_var_min_capture_list = if let Some(root_var_min_capture_list) =
            min_captures.and_then(|m| m.get(&var_hir_id))
        {
            root_var_min_capture_list
        } else {
            // The upvar is mentioned within the closure but no path starting from it is
            // used.

            match closure_clause {
                // Only migrate if closure is a move closure
                hir::CaptureBy::Value => return true,
                hir::CaptureBy::Ref => {}
            }

            return false;
        };

        let projections_list = root_var_min_capture_list
            .iter()
            .filter_map(|captured_place| match captured_place.info.capture_kind {
                // Only care about captures that are moved into the closure
                ty::UpvarCapture::ByValue(..) => Some(captured_place.place.projections.as_slice()),
                ty::UpvarCapture::ByRef(..) => None,
            })
            .collect::<Vec<_>>();

        let is_moved = !projections_list.is_empty();

        let is_not_completely_captured =
            root_var_min_capture_list.iter().any(|capture| capture.place.projections.len() > 0);

        if is_moved
            && is_not_completely_captured
            && self.has_significant_drop_outside_of_captures(
                closure_def_id,
                closure_span,
                ty,
                projections_list,
            )
        {
            return true;
        }

        return false;
    }

    /// Figures out the list of root variables (and their types) that aren't completely
    /// captured by the closure when `capture_disjoint_fields` is enabled and either drop
    /// order of some path starting at that root variable **might** be affected or auto-traits
    /// differ between the root variable and the captured paths.
    ///
    /// The output list would include a root variable if:
    /// - It would have been moved into the closure when `capture_disjoint_fields` wasn't
    ///   enabled, **and**
    /// - It wasn't completely captured by the closure, **and**
    /// - One of the paths starting at this root variable, that is not captured needs Drop **or**
    /// - One of the paths captured does not implement all the auto-traits its root variable
    ///   implements.
    ///
    /// Returns a tuple containing a vector of HirIds as well as a String containing the reason
    /// why root variables whose HirId is contained in the vector should be fully captured.
    fn compute_2229_migrations(
        &self,
        closure_def_id: DefId,
        closure_span: Span,
        closure_clause: hir::CaptureBy,
        min_captures: Option<&ty::RootVariableMinCaptureList<'tcx>>,
    ) -> (Vec<hir::HirId>, String) {
        let upvars = if let Some(upvars) = self.tcx.upvars_mentioned(closure_def_id) {
            upvars
        } else {
            return (Vec::new(), format!(""));
        };

        let mut need_migrations = Vec::new();
        let mut auto_trait_reasons = FxHashSet::default();
        let mut drop_reorder_reason = false;

        // Perform auto-trait analysis
        for (&var_hir_id, _) in upvars.iter() {
            let mut need_migration = false;
            if let Some(trait_migration_cause) =
                self.compute_2229_migrations_for_trait(min_captures, var_hir_id)
            {
                need_migration = true;
                auto_trait_reasons.extend(trait_migration_cause);
            }

            if self.compute_2229_migrations_for_drop(
                closure_def_id,
                closure_span,
                min_captures,
                closure_clause,
                var_hir_id,
            ) {
                need_migration = true;
                drop_reorder_reason = true;
            }

            if need_migration {
                need_migrations.push(var_hir_id);
            }
        }

        (
            need_migrations,
            self.compute_2229_migrations_reasons(auto_trait_reasons, drop_reorder_reason),
        )
    }

    /// This is a helper function to `compute_2229_migrations_precise_pass`. Provided the type
    /// of a root variable and a list of captured paths starting at this root variable (expressed
    /// using list of `Projection` slices), it returns true if there is a path that is not
    /// captured starting at this root variable that implements Drop.
    ///
    /// The way this function works is at a given call it looks at type `base_path_ty` of some base
    /// path say P and then list of projection slices which represent the different captures moved
    /// into the closure starting off of P.
    ///
    /// This will make more sense with an example:
    ///
    /// ```rust
    /// #![feature(capture_disjoint_fields)]
    ///
    /// struct FancyInteger(i32); // This implements Drop
    ///
    /// struct Point { x: FancyInteger, y: FancyInteger }
    /// struct Color;
    ///
    /// struct Wrapper { p: Point, c: Color }
    ///
    /// fn f(w: Wrapper) {
    ///   let c = || {
    ///       // Closure captures w.p.x and w.c by move.
    ///   };
    ///
    ///   c();
    /// }
    /// ```
    ///
    /// If `capture_disjoint_fields` wasn't enabled the closure would've moved `w` instead of the
    /// precise paths. If we look closely `w.p.y` isn't captured which implements Drop and
    /// therefore Drop ordering would change and we want this function to return true.
    ///
    /// Call stack to figure out if we need to migrate for `w` would look as follows:
    ///
    /// Our initial base path is just `w`, and the paths captured from it are `w[p, x]` and
    /// `w[c]`.
    /// Notation:
    /// - Ty(place): Type of place
    /// - `(a, b)`: Represents the function parameters `base_path_ty` and `captured_by_move_projs`
    /// respectively.
    /// ```
    ///                  (Ty(w), [ &[p, x], &[c] ])
    ///                                 |
    ///                    ----------------------------
    ///                    |                          |
    ///                    v                          v
    ///        (Ty(w.p), [ &[x] ])          (Ty(w.c), [ &[] ]) // I(1)
    ///                    |                          |
    ///                    v                          v
    ///        (Ty(w.p), [ &[x] ])                 false
    ///                    |
    ///                    |
    ///          -------------------------------
    ///          |                             |
    ///          v                             v
    ///     (Ty((w.p).x), [ &[] ])     (Ty((w.p).y), []) // IMP 2
    ///          |                             |
    ///          v                             v
    ///        false              NeedsSignificantDrop(Ty(w.p.y))
    ///                                        |
    ///                                        v
    ///                                      true
    /// ```
    ///
    /// IMP 1 `(Ty(w.c), [ &[] ])`: Notice the single empty slice inside `captured_projs`.
    ///                             This implies that the `w.c` is completely captured by the closure.
    ///                             Since drop for this path will be called when the closure is
    ///                             dropped we don't need to migrate for it.
    ///
    /// IMP 2 `(Ty((w.p).y), [])`: Notice that `captured_projs` is empty. This implies that this
    ///                             path wasn't captured by the closure. Also note that even
    ///                             though we didn't capture this path, the function visits it,
    ///                             which is kind of the point of this function. We then return
    ///                             if the type of `w.p.y` implements Drop, which in this case is
    ///                             true.
    ///
    /// Consider another example:
    ///
    /// ```rust
    /// struct X;
    /// impl Drop for X {}
    ///
    /// struct Y(X);
    /// impl Drop for Y {}
    ///
    /// fn foo() {
    ///     let y = Y(X);
    ///     let c = || move(y.0);
    /// }
    /// ```
    ///
    /// Note that `y.0` is captured by the closure. When this function is called for `y`, it will
    /// return true, because even though all paths starting at `y` are captured, `y` itself
    /// implements Drop which will be affected since `y` isn't completely captured.
    fn has_significant_drop_outside_of_captures(
        &self,
        closure_def_id: DefId,
        closure_span: Span,
        base_path_ty: Ty<'tcx>,
        captured_by_move_projs: Vec<&[Projection<'tcx>]>,
    ) -> bool {
        let needs_drop = |ty: Ty<'tcx>| {
            ty.has_significant_drop(self.tcx, self.tcx.param_env(closure_def_id.expect_local()))
        };

        let is_drop_defined_for_ty = |ty: Ty<'tcx>| {
            let drop_trait = self.tcx.require_lang_item(hir::LangItem::Drop, Some(closure_span));
            let ty_params = self.tcx.mk_substs_trait(base_path_ty, &[]);
            self.tcx.type_implements_trait((
                drop_trait,
                ty,
                ty_params,
                self.tcx.param_env(closure_def_id.expect_local()),
            ))
        };

        let is_drop_defined_for_ty = is_drop_defined_for_ty(base_path_ty);

        // If there is a case where no projection is applied on top of current place
        // then there must be exactly one capture corresponding to such a case. Note that this
        // represents the case of the path being completely captured by the variable.
        //
        // eg. If `a.b` is captured and we are processing `a.b`, then we can't have the closure also
        //     capture `a.b.c`, because that voilates min capture.
        let is_completely_captured = captured_by_move_projs.iter().any(|projs| projs.is_empty());

        assert!(!is_completely_captured || (captured_by_move_projs.len() == 1));

        if is_completely_captured {
            // The place is captured entirely, so doesn't matter if needs dtor, it will be drop
            // when the closure is dropped.
            return false;
        }

        if captured_by_move_projs.is_empty() {
            return needs_drop(base_path_ty);
        }

        if is_drop_defined_for_ty {
            // If drop is implemented for this type then we need it to be fully captured,
            // and we know it is not completely captured because of the previous checks.

            // Note that this is a bug in the user code that will be reported by the
            // borrow checker, since we can't move out of drop types.

            // The bug exists in the user's code pre-migration, and we don't migrate here.
            return false;
        }

        match base_path_ty.kind() {
            // Observations:
            // - `captured_by_move_projs` is not empty. Therefore we can call
            //   `captured_by_move_projs.first().unwrap()` safely.
            // - All entries in `captured_by_move_projs` have atleast one projection.
            //   Therefore we can call `captured_by_move_projs.first().unwrap().first().unwrap()` safely.

            // We don't capture derefs in case of move captures, which would have be applied to
            // access any further paths.
            ty::Adt(def, _) if def.is_box() => unreachable!(),
            ty::Ref(..) => unreachable!(),
            ty::RawPtr(..) => unreachable!(),

            ty::Adt(def, substs) => {
                // Multi-varaint enums are captured in entirety,
                // which would've been handled in the case of single empty slice in `captured_by_move_projs`.
                assert_eq!(def.variants.len(), 1);

                // Only Field projections can be applied to a non-box Adt.
                assert!(
                    captured_by_move_projs.iter().all(|projs| matches!(
                        projs.first().unwrap().kind,
                        ProjectionKind::Field(..)
                    ))
                );
                def.variants.get(VariantIdx::new(0)).unwrap().fields.iter().enumerate().any(
                    |(i, field)| {
                        let paths_using_field = captured_by_move_projs
                            .iter()
                            .filter_map(|projs| {
                                if let ProjectionKind::Field(field_idx, _) =
                                    projs.first().unwrap().kind
                                {
                                    if (field_idx as usize) == i { Some(&projs[1..]) } else { None }
                                } else {
                                    unreachable!();
                                }
                            })
                            .collect();

                        let after_field_ty = field.ty(self.tcx, substs);
                        self.has_significant_drop_outside_of_captures(
                            closure_def_id,
                            closure_span,
                            after_field_ty,
                            paths_using_field,
                        )
                    },
                )
            }

            ty::Tuple(..) => {
                // Only Field projections can be applied to a tuple.
                assert!(
                    captured_by_move_projs.iter().all(|projs| matches!(
                        projs.first().unwrap().kind,
                        ProjectionKind::Field(..)
                    ))
                );

                base_path_ty.tuple_fields().enumerate().any(|(i, element_ty)| {
                    let paths_using_field = captured_by_move_projs
                        .iter()
                        .filter_map(|projs| {
                            if let ProjectionKind::Field(field_idx, _) = projs.first().unwrap().kind
                            {
                                if (field_idx as usize) == i { Some(&projs[1..]) } else { None }
                            } else {
                                unreachable!();
                            }
                        })
                        .collect();

                    self.has_significant_drop_outside_of_captures(
                        closure_def_id,
                        closure_span,
                        element_ty,
                        paths_using_field,
                    )
                })
            }

            // Anything else would be completely captured and therefore handled already.
            _ => unreachable!(),
        }
    }

    fn init_capture_kind_for_place(
        &self,
        place: &Place<'tcx>,
        capture_clause: hir::CaptureBy,
        upvar_id: ty::UpvarId,
        closure_span: Span,
    ) -> ty::UpvarCapture<'tcx> {
        match capture_clause {
            // In case of a move closure if the data is accessed through a reference we
            // want to capture by ref to allow precise capture using reborrows.
            //
            // If the data will be moved out of this place, then the place will be truncated
            // at the first Deref in `adjust_upvar_borrow_kind_for_consume` and then moved into
            // the closure.
            hir::CaptureBy::Value if !place.deref_tys().any(ty::TyS::is_ref) => {
                ty::UpvarCapture::ByValue(None)
            }
            hir::CaptureBy::Value | hir::CaptureBy::Ref => {
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

                let span =
                    capture_info.path_expr_id.map_or(closure_span, |e| self.tcx.hir().span(e));
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

                        if capture.info.path_expr_id != capture.info.capture_kind_expr_id {
                            let path_span = capture_info
                                .path_expr_id
                                .map_or(closure_span, |e| self.tcx.hir().span(e));
                            let capture_kind_span = capture_info
                                .capture_kind_expr_id
                                .map_or(closure_span, |e| self.tcx.hir().span(e));

                            let mut multi_span: MultiSpan =
                                MultiSpan::from_spans(vec![path_span, capture_kind_span]);

                            let capture_kind_label =
                                construct_capture_kind_reason_string(self.tcx, place, capture_info);
                            let path_label = construct_path_string(self.tcx, place);

                            multi_span.push_span_label(path_span, path_label);
                            multi_span.push_span_label(capture_kind_span, capture_kind_label);

                            diag.span_note(multi_span, &output_str);
                        } else {
                            let span = capture_info
                                .path_expr_id
                                .map_or(closure_span, |e| self.tcx.hir().span(e));

                            diag.span_note(span, &output_str);
                        };
                    }
                }
                diag.emit();
            }
        }
    }

    /// A captured place is mutable if
    /// 1. Projections don't include a Deref of an immut-borrow, **and**
    /// 2. PlaceBase is mut or projections include a Deref of a mut-borrow.
    fn determine_capture_mutability(
        &self,
        typeck_results: &'a TypeckResults<'tcx>,
        place: &Place<'tcx>,
    ) -> hir::Mutability {
        let var_hir_id = match place.base {
            PlaceBase::Upvar(upvar_id) => upvar_id.var_path.hir_id,
            _ => unreachable!(),
        };

        let bm = *typeck_results.pat_binding_modes().get(var_hir_id).expect("missing binding mode");

        let mut is_mutbl = match bm {
            ty::BindByValue(mutability) => mutability,
            ty::BindByReference(_) => hir::Mutability::Not,
        };

        for pointer_ty in place.deref_tys() {
            match pointer_ty.kind() {
                // We don't capture derefs of raw ptrs
                ty::RawPtr(_) => unreachable!(),

                // Derefencing a mut-ref allows us to mut the Place if we don't deref
                // an immut-ref after on top of this.
                ty::Ref(.., hir::Mutability::Mut) => is_mutbl = hir::Mutability::Mut,

                // The place isn't mutable once we dereference a immutable reference.
                ty::Ref(.., hir::Mutability::Not) => return hir::Mutability::Not,

                // Dereferencing a box doesn't change mutability
                ty::Adt(def, ..) if def.is_box() => {}

                unexpected_ty => bug!("deref of unexpected pointer type {:?}", unexpected_ty),
            }
        }

        is_mutbl
    }
}

/// Truncate the capture so that the place being borrowed is in accordance with RFC 1240,
/// which states that it's unsafe to take a reference into a struct marked `repr(packed)`.
fn restrict_repr_packed_field_ref_capture<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    place: &Place<'tcx>,
) -> Place<'tcx> {
    let pos = place.projections.iter().enumerate().position(|(i, p)| {
        let ty = place.ty_before_projection(i);

        // Return true for fields of packed structs, unless those fields have alignment 1.
        match p.kind {
            ProjectionKind::Field(..) => match ty.kind() {
                ty::Adt(def, _) if def.repr.packed() => {
                    match tcx.layout_raw(param_env.and(p.ty)) {
                        Ok(layout) if layout.align.abi.bytes() == 1 => {
                            // if the alignment is 1, the type can't be further
                            // disaligned.
                            debug!(
                                "restrict_repr_packed_field_ref_capture: ({:?}) - align = 1",
                                place
                            );
                            false
                        }
                        _ => {
                            debug!("restrict_repr_packed_field_ref_capture: ({:?}) - true", place);
                            true
                        }
                    }
                }

                _ => false,
            },
            _ => false,
        }
    });

    let mut place = place.clone();

    if let Some(pos) = pos {
        place.projections.truncate(pos);
    }

    place
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
    /// ```
    /// Place { V1, [ProjectionKind::Field(Index=0, Variant=0)] } : CaptureKind { E1, ImmutableBorrow }
    /// Place { V1, [ProjectionKind::Field(Index=1, Variant=0)] } : CaptureKind { E2, MutableBorrow }
    /// ```
    capture_information: InferredCaptureInformation<'tcx>,
    fake_reads: Vec<(Place<'tcx>, FakeReadCause, hir::HirId)>,
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

        match (self.capture_clause, mode) {
            // In non-move closures, we only care about moves
            (hir::CaptureBy::Ref, euv::Copy) => return,

            // We want to capture Copy types that read through a ref via a reborrow
            (hir::CaptureBy::Value, euv::Copy)
                if place_with_id.place.deref_tys().any(ty::TyS::is_ref) =>
            {
                return;
            }

            (hir::CaptureBy::Ref, euv::Move) | (hir::CaptureBy::Value, euv::Move | euv::Copy) => {}
        };

        let place = truncate_capture_for_move(place_with_id.place.clone());
        let place_with_id = PlaceWithHirId { place: place.clone(), hir_id: place_with_id.hir_id };

        if !self.capture_information.contains_key(&place) {
            self.init_capture_info_for_place(&place_with_id, diag_expr_id);
        }

        let tcx = self.fcx.tcx;
        let upvar_id = if let PlaceBase::Upvar(upvar_id) = place_with_id.place.base {
            upvar_id
        } else {
            return;
        };

        debug!("adjust_upvar_borrow_kind_for_consume: upvar={:?}", upvar_id);

        let usage_span = tcx.hir().span(diag_expr_id);

        if matches!(mode, euv::Move) {
            // To move out of an upvar, this must be a FnOnce closure
            self.adjust_closure_kind(
                upvar_id.closure_expr_id,
                ty::ClosureKind::FnOnce,
                usage_span,
                place.clone(),
            );
        }

        let capture_info = ty::CaptureInfo {
            capture_kind_expr_id: Some(diag_expr_id),
            path_expr_id: Some(diag_expr_id),
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
                capture_kind_expr_id: Some(diag_expr_id),
                path_expr_id: Some(diag_expr_id),
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

            let capture_kind = self.fcx.init_capture_kind_for_place(
                &place_with_id.place,
                self.capture_clause,
                upvar_id,
                self.closure_span,
            );

            let expr_id = Some(diag_expr_id);
            let capture_info = ty::CaptureInfo {
                capture_kind_expr_id: expr_id,
                path_expr_id: expr_id,
                capture_kind,
            };

            debug!("Capturing new place {:?}, capture_info={:?}", place_with_id, capture_info);

            self.capture_information.insert(place_with_id.place.clone(), capture_info);
        } else {
            debug!("Not upvar: {:?}", place_with_id);
        }
    }
}

impl<'a, 'tcx> euv::Delegate<'tcx> for InferBorrowKind<'a, 'tcx> {
    fn fake_read(&mut self, place: Place<'tcx>, cause: FakeReadCause, diag_expr_id: hir::HirId) {
        if let PlaceBase::Upvar(_) = place.base {
            self.fake_reads.push((place, cause, diag_expr_id));
        }
    }

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

        let place = restrict_repr_packed_field_ref_capture(
            self.fcx.tcx,
            self.fcx.param_env,
            &place_with_id.place,
        );
        let place_with_id = PlaceWithHirId { place, ..*place_with_id };

        if !self.capture_information.contains_key(&place_with_id.place) {
            self.init_capture_info_for_place(&place_with_id, diag_expr_id);
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

        self.borrow(assignee_place, diag_expr_id, ty::BorrowKind::MutBorrow);
    }
}

/// Truncate projections so that following rules are obeyed by the captured `place`:
/// - No projections are applied to raw pointers, since these require unsafe blocks. We capture
///   them completely.
/// - No Index projections are captured, since arrays are captured completely.
fn restrict_capture_precision<'tcx>(mut place: Place<'tcx>) -> Place<'tcx> {
    if place.projections.is_empty() {
        // Nothing to do here
        return place;
    }

    if place.base_ty.is_unsafe_ptr() {
        place.projections.truncate(0);
        return place;
    }

    let mut truncated_length = usize::MAX;

    for (i, proj) in place.projections.iter().enumerate() {
        if proj.ty.is_unsafe_ptr() {
            // Don't apply any projections on top of an unsafe ptr
            truncated_length = truncated_length.min(i + 1);
            break;
        }
        match proj.kind {
            ProjectionKind::Index => {
                // Arrays are completely captured, so we drop Index projections
                truncated_length = truncated_length.min(i);
                break;
            }
            ProjectionKind::Deref => {}
            ProjectionKind::Field(..) => {} // ignore
            ProjectionKind::Subslice => {}  // We never capture this
        }
    }

    let length = place.projections.len().min(truncated_length);

    place.projections.truncate(length);

    place
}

/// Truncates a place so that the resultant capture doesn't move data out of a reference
fn truncate_capture_for_move(mut place: Place<'tcx>) -> Place<'tcx> {
    if let Some(i) = place.projections.iter().position(|proj| proj.kind == ProjectionKind::Deref) {
        // We only drop Derefs in case of move closures
        // There might be an index projection or raw ptr ahead, so we don't stop here.
        place.projections.truncate(i);
    }

    place
}

fn construct_place_string(tcx: TyCtxt<'_>, place: &Place<'tcx>) -> String {
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
            projections_str.push(',');
        }
        projections_str.push_str(proj.as_str());
    }

    format!("{}[{}]", variable_name, projections_str)
}

fn construct_capture_kind_reason_string(
    tcx: TyCtxt<'_>,
    place: &Place<'tcx>,
    capture_info: &ty::CaptureInfo<'tcx>,
) -> String {
    let place_str = construct_place_string(tcx, &place);

    let capture_kind_str = match capture_info.capture_kind {
        ty::UpvarCapture::ByValue(_) => "ByValue".into(),
        ty::UpvarCapture::ByRef(borrow) => format!("{:?}", borrow.kind),
    };

    format!("{} captured as {} here", place_str, capture_kind_str)
}

fn construct_path_string(tcx: TyCtxt<'_>, place: &Place<'tcx>) -> String {
    let place_str = construct_place_string(tcx, &place);

    format!("{} used here", place_str)
}

fn construct_capture_info_string(
    tcx: TyCtxt<'_>,
    place: &Place<'tcx>,
    capture_info: &ty::CaptureInfo<'tcx>,
) -> String {
    let place_str = construct_place_string(tcx, &place);

    let capture_kind_str = match capture_info.capture_kind {
        ty::UpvarCapture::ByValue(_) => "ByValue".into(),
        ty::UpvarCapture::ByRef(borrow) => format!("{:?}", borrow.kind),
    };
    format!("{} -> {}", place_str, capture_kind_str)
}

fn var_name(tcx: TyCtxt<'_>, var_hir_id: hir::HirId) -> Symbol {
    tcx.hir().name(var_hir_id)
}

fn should_do_disjoint_capture_migration_analysis(tcx: TyCtxt<'_>, closure_id: hir::HirId) -> bool {
    let (level, _) = tcx.lint_level_at_node(lint::builtin::DISJOINT_CAPTURE_MIGRATION, closure_id);

    !matches!(level, lint::Level::Allow)
}

/// Return a two string tuple (s1, s2)
/// - s1: Line of code that is needed for the migration: eg: `let _ = (&x, ...)`.
/// - s2: Comma separated names of the variables being migrated.
fn migration_suggestion_for_2229(
    tcx: TyCtxt<'_>,
    need_migrations: &Vec<hir::HirId>,
) -> (String, String) {
    let need_migrations_variables =
        need_migrations.iter().map(|v| var_name(tcx, *v)).collect::<Vec<_>>();

    let migration_ref_concat =
        need_migrations_variables.iter().map(|v| format!("&{}", v)).collect::<Vec<_>>().join(", ");

    let migration_string = if 1 == need_migrations.len() {
        format!("let _ = {}", migration_ref_concat)
    } else {
        format!("let _ = ({})", migration_ref_concat)
    };

    let migrated_variables_concat =
        need_migrations_variables.iter().map(|v| format!("`{}`", v)).collect::<Vec<_>>().join(", ");

    (migration_string, migrated_variables_concat)
}

/// Helper function to determine if we need to escalate CaptureKind from
/// CaptureInfo A to B and returns the escalated CaptureInfo.
/// (Note: CaptureInfo contains CaptureKind and an expression that led to capture it in that way)
///
/// If both `CaptureKind`s are considered equivalent, then the CaptureInfo is selected based
/// on the `CaptureInfo` containing an associated `capture_kind_expr_id`.
///
/// It is the caller's duty to figure out which path_expr_id to use.
///
/// If both the CaptureKind and Expression are considered to be equivalent,
/// then `CaptureInfo` A is preferred. This can be useful in cases where we want to priortize
/// expressions reported back to the user as part of diagnostics based on which appears earlier
/// in the closure. This can be achieved simply by calling
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
        match (capture_info_a.capture_kind_expr_id, capture_info_b.capture_kind_expr_id) {
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

    let same_initial_projections =
        iter::zip(projections_a, projections_b).all(|(proj_a, proj_b)| proj_a == proj_b);

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
