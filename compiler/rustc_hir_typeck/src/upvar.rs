//! ### Inferring borrow kinds for upvars
//!
//! Whenever there is a closure expression, we need to determine how each
//! upvar is used. We do this by initially assigning each upvar an
//! immutable "borrow kind" (see `ty::BorrowKind` for details) and then
//! "escalating" the kind as needed. The borrow kind proceeds according to
//! the following lattice:
//! ```ignore (not-rust)
//! ty::ImmBorrow -> ty::UniqueImmBorrow -> ty::MutBorrow
//! ```
//! So, for example, if we see an assignment `x = 5` to an upvar `x`, we
//! will promote its borrow kind to mutable borrow. If we see an `&mut x`
//! we'll do the same. Naturally, this applies not just to the upvar, but
//! to everything owned by `x`, so the result is the same for something
//! like `x.f = 5` and so on (presuming `x` is not a borrowed pointer to a
//! struct). These adjustments are performed in
//! `adjust_for_non_move_closure` (you can trace backwards through the code
//! from there).
//!
//! The fact that we are inferring borrow kinds as we go results in a
//! semi-hacky interaction with the way `ExprUseVisitor` is computing
//! `Place`s. In particular, it will query the current borrow kind as it
//! goes, and we'll return the *current* value, but this may get
//! adjusted later. Therefore, in this module, we generally ignore the
//! borrow kind (and derived mutabilities) that `ExprUseVisitor` returns
//! within `Place`s, since they may be inaccurate. (Another option
//! would be to use a unification scheme, where instead of returning a
//! concrete borrow kind like `ty::ImmBorrow`, we return a
//! `ty::InferBorrow(upvar_id)` or something like that, but this would
//! then mean that all later passes would have to check for these figments
//! and report an error, and it just seems like more mess in the end.)

use std::iter;

use rustc_abi::FIRST_VARIANT;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::unord::{ExtendUnord, UnordSet};
use rustc_errors::{Applicability, MultiSpan};
use rustc_hir as hir;
use rustc_hir::HirId;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::{self, Visitor};
use rustc_middle::hir::place::{Place, PlaceBase, PlaceWithHirId, Projection, ProjectionKind};
use rustc_middle::mir::FakeReadCause;
use rustc_middle::traits::ObligationCauseCode;
use rustc_middle::ty::{
    self, BorrowKind, ClosureSizeProfileData, Ty, TyCtxt, TypeVisitableExt as _, TypeckResults,
    UpvarArgs, UpvarCapture,
};
use rustc_middle::{bug, span_bug};
use rustc_session::lint;
use rustc_span::{BytePos, Pos, Span, Symbol, sym};
use rustc_trait_selection::infer::InferCtxtExt;
use tracing::{debug, instrument};

use super::FnCtxt;
use crate::expr_use_visitor as euv;

/// Describe the relationship between the paths of two places
/// eg:
/// - `foo` is ancestor of `foo.bar.baz`
/// - `foo.bar.baz` is an descendant of `foo.bar`
/// - `foo.bar` and `foo.baz` are divergent
enum PlaceAncestryRelation {
    Ancestor,
    Descendant,
    SamePlace,
    Divergent,
}

/// Intermediate format to store a captured `Place` and associated `ty::CaptureInfo`
/// during capture analysis. Information in this map feeds into the minimum capture
/// analysis pass.
type InferredCaptureInformation<'tcx> = Vec<(Place<'tcx>, ty::CaptureInfo)>;

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub(crate) fn closure_analyze(&self, body: &'tcx hir::Body<'tcx>) {
        InferBorrowKindVisitor { fcx: self }.visit_body(body);

        // it's our job to process these.
        assert!(self.deferred_call_resolutions.borrow().is_empty());
    }
}

/// Intermediate format to store the hir_id pointing to the use that resulted in the
/// corresponding place being captured and a String which contains the captured value's
/// name (i.e: a.b.c)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum UpvarMigrationInfo {
    /// We previously captured all of `x`, but now we capture some sub-path.
    CapturingPrecise { source_expr: Option<HirId>, var_name: String },
    CapturingNothing {
        // where the variable appears in the closure (but is not captured)
        use_span: Span,
    },
}

/// Reasons that we might issue a migration warning.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct MigrationWarningReason {
    /// When we used to capture `x` in its entirety, we implemented the auto-trait(s)
    /// in this vec, but now we don't.
    auto_traits: Vec<&'static str>,

    /// When we used to capture `x` in its entirety, we would execute some destructors
    /// at a different time.
    drop_order: bool,
}

impl MigrationWarningReason {
    fn migration_message(&self) -> String {
        let base = "changes to closure capture in Rust 2021 will affect";
        if !self.auto_traits.is_empty() && self.drop_order {
            format!("{base} drop order and which traits the closure implements")
        } else if self.drop_order {
            format!("{base} drop order")
        } else {
            format!("{base} which traits the closure implements")
        }
    }
}

/// Intermediate format to store information needed to generate a note in the migration lint.
struct MigrationLintNote {
    captures_info: UpvarMigrationInfo,

    /// reasons why migration is needed for this capture
    reason: MigrationWarningReason,
}

/// Intermediate format to store the hir id of the root variable and a HashSet containing
/// information on why the root variable should be fully captured
struct NeededMigration {
    var_hir_id: HirId,
    diagnostics_info: Vec<MigrationLintNote>,
}

struct InferBorrowKindVisitor<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for InferBorrowKindVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        match expr.kind {
            hir::ExprKind::Closure(&hir::Closure { capture_clause, body: body_id, .. }) => {
                let body = self.fcx.tcx.hir_body(body_id);
                self.visit_body(body);
                self.fcx.analyze_closure(expr.hir_id, expr.span, body_id, body, capture_clause);
            }
            _ => {}
        }

        intravisit::walk_expr(self, expr);
    }

    fn visit_inline_const(&mut self, c: &'tcx hir::ConstBlock) {
        let body = self.fcx.tcx.hir_body(c.body);
        self.visit_body(body);
    }
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    /// Analysis starting point.
    #[instrument(skip(self, body), level = "debug")]
    fn analyze_closure(
        &self,
        closure_hir_id: HirId,
        span: Span,
        body_id: hir::BodyId,
        body: &'tcx hir::Body<'tcx>,
        mut capture_clause: hir::CaptureBy,
    ) {
        // Extract the type of the closure.
        let ty = self.node_ty(closure_hir_id);
        let (closure_def_id, args, infer_kind) = match *ty.kind() {
            ty::Closure(def_id, args) => {
                (def_id, UpvarArgs::Closure(args), self.closure_kind(ty).is_none())
            }
            ty::CoroutineClosure(def_id, args) => {
                (def_id, UpvarArgs::CoroutineClosure(args), self.closure_kind(ty).is_none())
            }
            ty::Coroutine(def_id, args) => (def_id, UpvarArgs::Coroutine(args), false),
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
        let args = self.resolve_vars_if_possible(args);
        let closure_def_id = closure_def_id.expect_local();

        assert_eq!(self.tcx.hir_body_owner_def_id(body.id()), closure_def_id);
        let mut delegate = InferBorrowKind {
            closure_def_id,
            capture_information: Default::default(),
            fake_reads: Default::default(),
        };

        let _ = euv::ExprUseVisitor::new(
            &FnCtxt::new(self, self.tcx.param_env(closure_def_id), closure_def_id),
            &mut delegate,
        )
        .consume_body(body);

        // There are several curious situations with coroutine-closures where
        // analysis is too aggressive with borrows when the coroutine-closure is
        // marked `move`. Specifically:
        //
        // 1. If the coroutine-closure was inferred to be `FnOnce` during signature
        // inference, then it's still possible that we try to borrow upvars from
        // the coroutine-closure because they are not used by the coroutine body
        // in a way that forces a move. See the test:
        // `async-await/async-closures/force-move-due-to-inferred-kind.rs`.
        //
        // 2. If the coroutine-closure is forced to be `FnOnce` due to the way it
        // uses its upvars (e.g. it consumes a non-copy value), but not *all* upvars
        // would force the closure to `FnOnce`.
        // See the test: `async-await/async-closures/force-move-due-to-actually-fnonce.rs`.
        //
        // This would lead to an impossible to satisfy situation, since `AsyncFnOnce`
        // coroutine bodies can't borrow from their parent closure. To fix this,
        // we force the inner coroutine to also be `move`. This only matters for
        // coroutine-closures that are `move` since otherwise they themselves will
        // be borrowing from the outer environment, so there's no self-borrows occurring.
        if let UpvarArgs::Coroutine(..) = args
            && let hir::CoroutineKind::Desugared(_, hir::CoroutineSource::Closure) =
                self.tcx.coroutine_kind(closure_def_id).expect("coroutine should have kind")
            && let parent_hir_id =
                self.tcx.local_def_id_to_hir_id(self.tcx.local_parent(closure_def_id))
            && let parent_ty = self.node_ty(parent_hir_id)
            && let hir::CaptureBy::Value { move_kw } =
                self.tcx.hir_node(parent_hir_id).expect_closure().capture_clause
        {
            // (1.) Closure signature inference forced this closure to `FnOnce`.
            if let Some(ty::ClosureKind::FnOnce) = self.closure_kind(parent_ty) {
                capture_clause = hir::CaptureBy::Value { move_kw };
            }
            // (2.) The way that the closure uses its upvars means it's `FnOnce`.
            else if self.coroutine_body_consumes_upvars(closure_def_id, body) {
                capture_clause = hir::CaptureBy::Value { move_kw };
            }
        }

        // As noted in `lower_coroutine_body_with_moved_arguments`, we default the capture mode
        // to `ByRef` for the `async {}` block internal to async fns/closure. This means
        // that we would *not* be moving all of the parameters into the async block in all cases.
        // For example, when one of the arguments is `Copy`, we turn a consuming use into a copy of
        // a reference, so for `async fn x(t: i32) {}`, we'd only take a reference to `t`.
        //
        // We force all of these arguments to be captured by move before we do expr use analysis.
        //
        // FIXME(async_closures): This could be cleaned up. It's a bit janky that we're just
        // moving all of the `LocalSource::AsyncFn` locals here.
        if let Some(hir::CoroutineKind::Desugared(
            _,
            hir::CoroutineSource::Fn | hir::CoroutineSource::Closure,
        )) = self.tcx.coroutine_kind(closure_def_id)
        {
            let hir::ExprKind::Block(block, _) = body.value.kind else {
                bug!();
            };
            for stmt in block.stmts {
                let hir::StmtKind::Let(hir::LetStmt {
                    init: Some(init),
                    source: hir::LocalSource::AsyncFn,
                    pat,
                    ..
                }) = stmt.kind
                else {
                    bug!();
                };
                let hir::PatKind::Binding(hir::BindingMode(hir::ByRef::No, _), _, _, _) = pat.kind
                else {
                    // Complex pattern, skip the non-upvar local.
                    continue;
                };
                let hir::ExprKind::Path(hir::QPath::Resolved(_, path)) = init.kind else {
                    bug!();
                };
                let hir::def::Res::Local(local_id) = path.res else {
                    bug!();
                };
                let place = self.place_for_root_variable(closure_def_id, local_id);
                delegate.capture_information.push((
                    place,
                    ty::CaptureInfo {
                        capture_kind_expr_id: Some(init.hir_id),
                        path_expr_id: Some(init.hir_id),
                        capture_kind: UpvarCapture::ByValue,
                    },
                ));
            }
        }

        debug!(
            "For closure={:?}, capture_information={:#?}",
            closure_def_id, delegate.capture_information
        );

        self.log_capture_analysis_first_pass(closure_def_id, &delegate.capture_information, span);

        let (capture_information, closure_kind, origin) = self
            .process_collected_capture_information(capture_clause, &delegate.capture_information);

        self.compute_min_captures(closure_def_id, capture_information, span);

        let closure_hir_id = self.tcx.local_def_id_to_hir_id(closure_def_id);

        if should_do_rust_2021_incompatible_closure_captures_analysis(self.tcx, closure_hir_id) {
            self.perform_2229_migration_analysis(closure_def_id, body_id, capture_clause, span);
        }

        let after_feature_tys = self.final_upvar_tys(closure_def_id);

        // We now fake capture information for all variables that are mentioned within the closure
        // We do this after handling migrations so that min_captures computes before
        if !enable_precise_capture(span) {
            let mut capture_information: InferredCaptureInformation<'tcx> = Default::default();

            if let Some(upvars) = self.tcx.upvars_mentioned(closure_def_id) {
                for var_hir_id in upvars.keys() {
                    let place = self.place_for_root_variable(closure_def_id, *var_hir_id);

                    debug!("seed place {:?}", place);

                    let capture_kind = self.init_capture_kind_for_place(&place, capture_clause);
                    let fake_info = ty::CaptureInfo {
                        capture_kind_expr_id: None,
                        path_expr_id: None,
                        capture_kind,
                    };

                    capture_information.push((place, fake_info));
                }
            }

            // This will update the min captures based on this new fake information.
            self.compute_min_captures(closure_def_id, capture_information, span);
        }

        let before_feature_tys = self.final_upvar_tys(closure_def_id);

        if infer_kind {
            // Unify the (as yet unbound) type variable in the closure
            // args with the kind we inferred.
            let closure_kind_ty = match args {
                UpvarArgs::Closure(args) => args.as_closure().kind_ty(),
                UpvarArgs::CoroutineClosure(args) => args.as_coroutine_closure().kind_ty(),
                UpvarArgs::Coroutine(_) => unreachable!("coroutines don't have an inferred kind"),
            };
            self.demand_eqtype(
                span,
                Ty::from_closure_kind(self.tcx, closure_kind),
                closure_kind_ty,
            );

            // If we have an origin, store it.
            if let Some(mut origin) = origin {
                if !enable_precise_capture(span) {
                    // Without precise captures, we just capture the base and ignore
                    // the projections.
                    origin.1.projections.clear()
                }

                self.typeck_results
                    .borrow_mut()
                    .closure_kind_origins_mut()
                    .insert(closure_hir_id, origin);
            }
        }

        // For coroutine-closures, we additionally must compute the
        // `coroutine_captures_by_ref_ty` type, which is used to generate the by-ref
        // version of the coroutine-closure's output coroutine.
        if let UpvarArgs::CoroutineClosure(args) = args
            && !args.references_error()
        {
            let closure_env_region: ty::Region<'_> = ty::Region::new_bound(
                self.tcx,
                ty::INNERMOST,
                ty::BoundRegion { var: ty::BoundVar::ZERO, kind: ty::BoundRegionKind::ClosureEnv },
            );

            let num_args = args
                .as_coroutine_closure()
                .coroutine_closure_sig()
                .skip_binder()
                .tupled_inputs_ty
                .tuple_fields()
                .len();
            let typeck_results = self.typeck_results.borrow();

            let tupled_upvars_ty_for_borrow = Ty::new_tup_from_iter(
                self.tcx,
                ty::analyze_coroutine_closure_captures(
                    typeck_results.closure_min_captures_flattened(closure_def_id),
                    typeck_results
                        .closure_min_captures_flattened(
                            self.tcx.coroutine_for_closure(closure_def_id).expect_local(),
                        )
                        // Skip the captures that are just moving the closure's args
                        // into the coroutine. These are always by move, and we append
                        // those later in the `CoroutineClosureSignature` helper functions.
                        .skip(num_args),
                    |(_, parent_capture), (_, child_capture)| {
                        // This is subtle. See documentation on function.
                        let needs_ref = should_reborrow_from_env_of_parent_coroutine_closure(
                            parent_capture,
                            child_capture,
                        );

                        let upvar_ty = child_capture.place.ty();
                        let capture = child_capture.info.capture_kind;
                        // Not all upvars are captured by ref, so use
                        // `apply_capture_kind_on_capture_ty` to ensure that we
                        // compute the right captured type.
                        return apply_capture_kind_on_capture_ty(
                            self.tcx,
                            upvar_ty,
                            capture,
                            if needs_ref {
                                closure_env_region
                            } else {
                                self.tcx.lifetimes.re_erased
                            },
                        );
                    },
                ),
            );
            let coroutine_captures_by_ref_ty = Ty::new_fn_ptr(
                self.tcx,
                ty::Binder::bind_with_vars(
                    self.tcx.mk_fn_sig(
                        [],
                        tupled_upvars_ty_for_borrow,
                        false,
                        hir::Safety::Safe,
                        rustc_abi::ExternAbi::Rust,
                    ),
                    self.tcx.mk_bound_variable_kinds(&[ty::BoundVariableKind::Region(
                        ty::BoundRegionKind::ClosureEnv,
                    )]),
                ),
            );
            self.demand_eqtype(
                span,
                args.as_coroutine_closure().coroutine_captures_by_ref_ty(),
                coroutine_captures_by_ref_ty,
            );

            // Additionally, we can now constrain the coroutine's kind type.
            //
            // We only do this if `infer_kind`, because if we have constrained
            // the kind from closure signature inference, the kind inferred
            // for the inner coroutine may actually be more restrictive.
            if infer_kind {
                let ty::Coroutine(_, coroutine_args) =
                    *self.typeck_results.borrow().expr_ty(body.value).kind()
                else {
                    bug!();
                };
                self.demand_eqtype(
                    span,
                    coroutine_args.as_coroutine().kind_ty(),
                    Ty::from_coroutine_closure_kind(self.tcx, closure_kind),
                );
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
        debug!(?closure_hir_id, ?args, ?final_upvar_tys);

        if self.tcx.features().unsized_locals() || self.tcx.features().unsized_fn_params() {
            for capture in
                self.typeck_results.borrow().closure_min_captures_flattened(closure_def_id)
            {
                if let UpvarCapture::ByValue = capture.info.capture_kind {
                    self.require_type_is_sized(
                        capture.place.ty(),
                        capture.get_path_span(self.tcx),
                        ObligationCauseCode::SizedClosureCapture(closure_def_id),
                    );
                }
            }
        }

        // Build a tuple (U0..Un) of the final upvar types U0..Un
        // and unify the upvar tuple type in the closure with it:
        let final_tupled_upvars_type = Ty::new_tup(self.tcx, &final_upvar_tys);
        self.demand_suptype(span, args.tupled_upvars_ty(), final_tupled_upvars_type);

        let fake_reads = delegate.fake_reads;

        self.typeck_results.borrow_mut().closure_fake_reads.insert(closure_def_id, fake_reads);

        if self.tcx.sess.opts.unstable_opts.profile_closures {
            self.typeck_results.borrow_mut().closure_size_eval.insert(
                closure_def_id,
                ClosureSizeProfileData {
                    before_feature_tys: Ty::new_tup(self.tcx, &before_feature_tys),
                    after_feature_tys: Ty::new_tup(self.tcx, &after_feature_tys),
                },
            );
        }

        // If we are also inferred the closure kind here,
        // process any deferred resolutions.
        let deferred_call_resolutions = self.remove_deferred_call_resolutions(closure_def_id);
        for deferred_call_resolution in deferred_call_resolutions {
            deferred_call_resolution.resolve(self);
        }
    }

    /// Determines whether the body of the coroutine uses its upvars in a way that
    /// consumes (i.e. moves) the value, which would force the coroutine to `FnOnce`.
    /// In a more detailed comment above, we care whether this happens, since if
    /// this happens, we want to force the coroutine to move all of the upvars it
    /// would've borrowed from the parent coroutine-closure.
    ///
    /// This only really makes sense to be called on the child coroutine of a
    /// coroutine-closure.
    fn coroutine_body_consumes_upvars(
        &self,
        coroutine_def_id: LocalDefId,
        body: &'tcx hir::Body<'tcx>,
    ) -> bool {
        // This block contains argument capturing details. Since arguments
        // aren't upvars, we do not care about them for determining if the
        // coroutine body actually consumes its upvars.
        let hir::ExprKind::Block(&hir::Block { expr: Some(body), .. }, None) = body.value.kind
        else {
            bug!();
        };
        // Specifically, we only care about the *real* body of the coroutine.
        // We skip out into the drop-temps within the block of the body in order
        // to skip over the args of the desugaring.
        let hir::ExprKind::DropTemps(body) = body.kind else {
            bug!();
        };

        let mut delegate = InferBorrowKind {
            closure_def_id: coroutine_def_id,
            capture_information: Default::default(),
            fake_reads: Default::default(),
        };

        let _ = euv::ExprUseVisitor::new(
            &FnCtxt::new(self, self.tcx.param_env(coroutine_def_id), coroutine_def_id),
            &mut delegate,
        )
        .consume_expr(body);

        let (_, kind, _) = self.process_collected_capture_information(
            hir::CaptureBy::Ref,
            &delegate.capture_information,
        );

        matches!(kind, ty::ClosureKind::FnOnce)
    }

    // Returns a list of `Ty`s for each upvar.
    fn final_upvar_tys(&self, closure_id: LocalDefId) -> Vec<Ty<'tcx>> {
        self.typeck_results
            .borrow()
            .closure_min_captures_flattened(closure_id)
            .map(|captured_place| {
                let upvar_ty = captured_place.place.ty();
                let capture = captured_place.info.capture_kind;

                debug!(?captured_place.place, ?upvar_ty, ?capture, ?captured_place.mutability);

                apply_capture_kind_on_capture_ty(
                    self.tcx,
                    upvar_ty,
                    capture,
                    self.tcx.lifetimes.re_erased,
                )
            })
            .collect()
    }

    /// Adjusts the closure capture information to ensure that the operations aren't unsafe,
    /// and that the path can be captured with required capture kind (depending on use in closure,
    /// move closure etc.)
    ///
    /// Returns the set of adjusted information along with the inferred closure kind and span
    /// associated with the closure kind inference.
    ///
    /// Note that we *always* infer a minimal kind, even if
    /// we don't always *use* that in the final result (i.e., sometimes
    /// we've taken the closure kind from the expectations instead, and
    /// for coroutines we don't even implement the closure traits
    /// really).
    ///
    /// If we inferred that the closure needs to be FnMut/FnOnce, last element of the returned tuple
    /// contains a `Some()` with the `Place` that caused us to do so.
    fn process_collected_capture_information(
        &self,
        capture_clause: hir::CaptureBy,
        capture_information: &InferredCaptureInformation<'tcx>,
    ) -> (InferredCaptureInformation<'tcx>, ty::ClosureKind, Option<(Span, Place<'tcx>)>) {
        let mut closure_kind = ty::ClosureKind::LATTICE_BOTTOM;
        let mut origin: Option<(Span, Place<'tcx>)> = None;

        let processed = capture_information
            .iter()
            .cloned()
            .map(|(place, mut capture_info)| {
                // Apply rules for safety before inferring closure kind
                let (place, capture_kind) =
                    restrict_capture_precision(place, capture_info.capture_kind);

                let (place, capture_kind) = truncate_capture_for_optimization(place, capture_kind);

                let usage_span = if let Some(usage_expr) = capture_info.path_expr_id {
                    self.tcx.hir_span(usage_expr)
                } else {
                    unreachable!()
                };

                let updated = match capture_kind {
                    ty::UpvarCapture::ByValue => match closure_kind {
                        ty::ClosureKind::Fn | ty::ClosureKind::FnMut => {
                            (ty::ClosureKind::FnOnce, Some((usage_span, place.clone())))
                        }
                        // If closure is already FnOnce, don't update
                        ty::ClosureKind::FnOnce => (closure_kind, origin.take()),
                    },

                    ty::UpvarCapture::ByRef(
                        ty::BorrowKind::Mutable | ty::BorrowKind::UniqueImmutable,
                    ) => {
                        match closure_kind {
                            ty::ClosureKind::Fn => {
                                (ty::ClosureKind::FnMut, Some((usage_span, place.clone())))
                            }
                            // Don't update the origin
                            ty::ClosureKind::FnMut | ty::ClosureKind::FnOnce => {
                                (closure_kind, origin.take())
                            }
                        }
                    }

                    _ => (closure_kind, origin.take()),
                };

                closure_kind = updated.0;
                origin = updated.1;

                let (place, capture_kind) = match capture_clause {
                    hir::CaptureBy::Value { .. } => adjust_for_move_closure(place, capture_kind),
                    hir::CaptureBy::Use { .. } => adjust_for_use_closure(place, capture_kind),
                    hir::CaptureBy::Ref => adjust_for_non_move_closure(place, capture_kind),
                };

                // This restriction needs to be applied after we have handled adjustments for `move`
                // closures. We want to make sure any adjustment that might make us move the place into
                // the closure gets handled.
                let (place, capture_kind) =
                    restrict_precision_for_drop_types(self, place, capture_kind);

                capture_info.capture_kind = capture_kind;
                (place, capture_info)
            })
            .collect();

        (processed, closure_kind, origin)
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
    /// ```
    /// #[derive(Debug)]
    /// struct Point { x: i32, y: i32 }
    ///
    /// let s = String::from("s");  // hir_id_s
    /// let mut p = Point { x: 2, y: -2 }; // his_id_p
    /// let c = || {
    ///        println!("{s:?}");  // L1
    ///        p.x += 10;  // L2
    ///        println!("{}" , p.y); // L3
    ///        println!("{p:?}"); // L4
    ///        drop(s);   // L5
    /// };
    /// ```
    /// and let hir_id_L1..5 be the expressions pointing to use of a captured variable on
    /// the lines L1..5 respectively.
    ///
    /// InferBorrowKind results in a structure like this:
    ///
    /// ```ignore (illustrative)
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
    /// }
    /// ```
    ///
    /// After the min capture analysis, we get:
    /// ```ignore (illustrative)
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
    /// }
    /// ```
    fn compute_min_captures(
        &self,
        closure_def_id: LocalDefId,
        capture_information: InferredCaptureInformation<'tcx>,
        closure_span: Span,
    ) {
        if capture_information.is_empty() {
            return;
        }

        let mut typeck_results = self.typeck_results.borrow_mut();

        let mut root_var_min_capture_list =
            typeck_results.closure_min_captures.remove(&closure_def_id).unwrap_or_default();

        for (mut place, capture_info) in capture_information.into_iter() {
            let var_hir_id = match place.base {
                PlaceBase::Upvar(upvar_id) => upvar_id.var_path.hir_id,
                base => bug!("Expected upvar, found={:?}", base),
            };
            let var_ident = self.tcx.hir_ident(var_hir_id);

            let Some(min_cap_list) = root_var_min_capture_list.get_mut(&var_hir_id) else {
                let mutability = self.determine_capture_mutability(&typeck_results, &place);
                let min_cap_list =
                    vec![ty::CapturedPlace { var_ident, place, info: capture_info, mutability }];
                root_var_min_capture_list.insert(var_hir_id, min_cap_list);
                continue;
            };

            // Go through each entry in the current list of min_captures
            // - if ancestor is found, update its capture kind to account for current place's
            // capture information.
            //
            // - if descendant is found, remove it from the list, and update the current place's
            // capture information to account for the descendant's capture kind.
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

                        let mut possible_descendant = possible_descendant.clone();
                        let backup_path_expr_id = updated_capture_info.path_expr_id;

                        // Truncate the descendant (already in min_captures) to be same as the ancestor to handle any
                        // possible change in capture mode.
                        truncate_place_to_len_and_update_capture_kind(
                            &mut possible_descendant.place,
                            &mut possible_descendant.info.capture_kind,
                            place.projections.len(),
                        );

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
                        PlaceAncestryRelation::SamePlace => {
                            ancestor_found = true;
                            possible_ancestor.info = determine_capture_info(
                                possible_ancestor.info,
                                updated_capture_info,
                            );

                            // Only one related place will be in the list.
                            break;
                        }
                        // current place is descendant of possible_ancestor
                        PlaceAncestryRelation::Descendant => {
                            ancestor_found = true;
                            let backup_path_expr_id = possible_ancestor.info.path_expr_id;

                            // Truncate the descendant (current place) to be same as the ancestor to handle any
                            // possible change in capture mode.
                            truncate_place_to_len_and_update_capture_kind(
                                &mut place,
                                &mut updated_capture_info.capture_kind,
                                possible_ancestor.place.projections.len(),
                            );

                            possible_ancestor.info = determine_capture_info(
                                possible_ancestor.info,
                                updated_capture_info,
                            );

                            // we need to keep the ancestor's `path_expr_id`
                            possible_ancestor.info.path_expr_id = backup_path_expr_id;

                            // Only one related place will be in the list.
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
                    ty::CapturedPlace { var_ident, place, info: updated_capture_info, mutability };
                min_cap_list.push(captured_place);
            }
        }

        debug!(
            "For closure={:?}, min_captures before sorting={:?}",
            closure_def_id, root_var_min_capture_list
        );

        // Now that we have the minimized list of captures, sort the captures by field id.
        // This causes the closure to capture the upvars in the same order as the fields are
        // declared which is also the drop order. Thus, in situations where we capture all the
        // fields of some type, the observable drop order will remain the same as it previously
        // was even though we're dropping each capture individually.
        // See https://github.com/rust-lang/project-rfc-2229/issues/42 and
        // `tests/ui/closures/2229_closure_analysis/preserve_field_drop_order.rs`.
        for (_, captures) in &mut root_var_min_capture_list {
            captures.sort_by(|capture1, capture2| {
                fn is_field<'a>(p: &&Projection<'a>) -> bool {
                    match p.kind {
                        ProjectionKind::Field(_, _) => true,
                        ProjectionKind::Deref
                        | ProjectionKind::OpaqueCast
                        | ProjectionKind::UnwrapUnsafeBinder => false,
                        p @ (ProjectionKind::Subslice | ProjectionKind::Index) => {
                            bug!("ProjectionKind {:?} was unexpected", p)
                        }
                    }
                }

                // Need to sort only by Field projections, so filter away others.
                // A previous implementation considered other projection types too
                // but that caused ICE #118144
                let capture1_field_projections = capture1.place.projections.iter().filter(is_field);
                let capture2_field_projections = capture2.place.projections.iter().filter(is_field);

                for (p1, p2) in capture1_field_projections.zip(capture2_field_projections) {
                    // We do not need to look at the `Projection.ty` fields here because at each
                    // step of the iteration, the projections will either be the same and therefore
                    // the types must be as well or the current projection will be different and
                    // we will return the result of comparing the field indexes.
                    match (p1.kind, p2.kind) {
                        (ProjectionKind::Field(i1, _), ProjectionKind::Field(i2, _)) => {
                            // Compare only if paths are different.
                            // Otherwise continue to the next iteration
                            if i1 != i2 {
                                return i1.cmp(&i2);
                            }
                        }
                        // Given the filter above, this arm should never be hit
                        (l, r) => bug!("ProjectionKinds {:?} or {:?} were unexpected", l, r),
                    }
                }

                self.dcx().span_delayed_bug(
                    closure_span,
                    format!(
                        "two identical projections: ({:?}, {:?})",
                        capture1.place.projections, capture2.place.projections
                    ),
                );
                std::cmp::Ordering::Equal
            });
        }

        debug!(
            "For closure={:?}, min_captures after sorting={:#?}",
            closure_def_id, root_var_min_capture_list
        );
        typeck_results.closure_min_captures.insert(closure_def_id, root_var_min_capture_list);
    }

    /// Perform the migration analysis for RFC 2229, and emit lint
    /// `disjoint_capture_drop_reorder` if needed.
    fn perform_2229_migration_analysis(
        &self,
        closure_def_id: LocalDefId,
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

            let closure_hir_id = self.tcx.local_def_id_to_hir_id(closure_def_id);
            let closure_head_span = self.tcx.def_span(closure_def_id);
            self.tcx.node_span_lint(
                lint::builtin::RUST_2021_INCOMPATIBLE_CLOSURE_CAPTURES,
                closure_hir_id,
                closure_head_span,
                |lint| {
                    lint.primary_message(reasons.migration_message());

                    for NeededMigration { var_hir_id, diagnostics_info } in &need_migrations {
                        // Labels all the usage of the captured variable and why they are responsible
                        // for migration being needed
                        for lint_note in diagnostics_info.iter() {
                            match &lint_note.captures_info {
                                UpvarMigrationInfo::CapturingPrecise { source_expr: Some(capture_expr_id), var_name: captured_name } => {
                                    let cause_span = self.tcx.hir_span(*capture_expr_id);
                                    lint.span_label(cause_span, format!("in Rust 2018, this closure captures all of `{}`, but in Rust 2021, it will only capture `{}`",
                                        self.tcx.hir_name(*var_hir_id),
                                        captured_name,
                                    ));
                                }
                                UpvarMigrationInfo::CapturingNothing { use_span } => {
                                    lint.span_label(*use_span, format!("in Rust 2018, this causes the closure to capture `{}`, but in Rust 2021, it has no effect",
                                        self.tcx.hir_name(*var_hir_id),
                                    ));
                                }

                                _ => { }
                            }

                            // Add a label pointing to where a captured variable affected by drop order
                            // is dropped
                            if lint_note.reason.drop_order {
                                let drop_location_span = drop_location_span(self.tcx, closure_hir_id);

                                match &lint_note.captures_info {
                                    UpvarMigrationInfo::CapturingPrecise { var_name: captured_name, .. } => {
                                        lint.span_label(drop_location_span, format!("in Rust 2018, `{}` is dropped here, but in Rust 2021, only `{}` will be dropped here as part of the closure",
                                            self.tcx.hir_name(*var_hir_id),
                                            captured_name,
                                        ));
                                    }
                                    UpvarMigrationInfo::CapturingNothing { use_span: _ } => {
                                        lint.span_label(drop_location_span, format!("in Rust 2018, `{v}` is dropped here along with the closure, but in Rust 2021 `{v}` is not part of the closure",
                                            v = self.tcx.hir_name(*var_hir_id),
                                        ));
                                    }
                                }
                            }

                            // Add a label explaining why a closure no longer implements a trait
                            for &missing_trait in &lint_note.reason.auto_traits {
                                // not capturing something anymore cannot cause a trait to fail to be implemented:
                                match &lint_note.captures_info {
                                    UpvarMigrationInfo::CapturingPrecise { var_name: captured_name, .. } => {
                                        let var_name = self.tcx.hir_name(*var_hir_id);
                                        lint.span_label(closure_head_span, format!("\
                                        in Rust 2018, this closure implements {missing_trait} \
                                        as `{var_name}` implements {missing_trait}, but in Rust 2021, \
                                        this closure will no longer implement {missing_trait} \
                                        because `{var_name}` is not fully captured \
                                        and `{captured_name}` does not implement {missing_trait}"));
                                    }

                                    // Cannot happen: if we don't capture a variable, we impl strictly more traits
                                    UpvarMigrationInfo::CapturingNothing { use_span } => span_bug!(*use_span, "missing trait from not capturing something"),
                                }
                            }
                        }
                    }
                    lint.note("for more information, see <https://doc.rust-lang.org/nightly/edition-guide/rust-2021/disjoint-capture-in-closures.html>");

                    let diagnostic_msg = format!(
                        "add a dummy let to cause {migrated_variables_concat} to be fully captured"
                    );

                    let closure_span = self.tcx.hir_span_with_body(closure_hir_id);
                    let mut closure_body_span = {
                        // If the body was entirely expanded from a macro
                        // invocation, i.e. the body is not contained inside the
                        // closure span, then we walk up the expansion until we
                        // find the span before the expansion.
                        let s = self.tcx.hir_span_with_body(body_id.hir_id);
                        s.find_ancestor_inside(closure_span).unwrap_or(s)
                    };

                    if let Ok(mut s) = self.tcx.sess.source_map().span_to_snippet(closure_body_span) {
                        if s.starts_with('$') {
                            // Looks like a macro fragment. Try to find the real block.
                            if let hir::Node::Expr(&hir::Expr {
                                kind: hir::ExprKind::Block(block, ..), ..
                            }) = self.tcx.hir_node(body_id.hir_id) {
                                // If the body is a block (with `{..}`), we use the span of that block.
                                // E.g. with a `|| $body` expanded from a `m!({ .. })`, we use `{ .. }`, and not `$body`.
                                // Since we know it's a block, we know we can insert the `let _ = ..` without
                                // breaking the macro syntax.
                                if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(block.span) {
                                    closure_body_span = block.span;
                                    s = snippet;
                                }
                            }
                        }

                        let mut lines = s.lines();
                        let line1 = lines.next().unwrap_or_default();

                        if line1.trim_end() == "{" {
                            // This is a multi-line closure with just a `{` on the first line,
                            // so we put the `let` on its own line.
                            // We take the indentation from the next non-empty line.
                            let line2 = lines.find(|line| !line.is_empty()).unwrap_or_default();
                            let indent = line2.split_once(|c: char| !c.is_whitespace()).unwrap_or_default().0;
                            lint.span_suggestion(
                                closure_body_span.with_lo(closure_body_span.lo() + BytePos::from_usize(line1.len())).shrink_to_lo(),
                                diagnostic_msg,
                                format!("\n{indent}{migration_string};"),
                                Applicability::MachineApplicable,
                            );
                        } else if line1.starts_with('{') {
                            // This is a closure with its body wrapped in
                            // braces, but with more than just the opening
                            // brace on the first line. We put the `let`
                            // directly after the `{`.
                            lint.span_suggestion(
                                closure_body_span.with_lo(closure_body_span.lo() + BytePos(1)).shrink_to_lo(),
                                diagnostic_msg,
                                format!(" {migration_string};"),
                                Applicability::MachineApplicable,
                            );
                        } else {
                            // This is a closure without braces around the body.
                            // We add braces to add the `let` before the body.
                            lint.multipart_suggestion(
                                diagnostic_msg,
                                vec![
                                    (closure_body_span.shrink_to_lo(), format!("{{ {migration_string}; ")),
                                    (closure_body_span.shrink_to_hi(), " }".to_string()),
                                ],
                                Applicability::MachineApplicable
                            );
                        }
                    } else {
                        lint.span_suggestion(
                            closure_span,
                            diagnostic_msg,
                            migration_string,
                            Applicability::HasPlaceholders
                        );
                    }
                },
            );
        }
    }

    /// Combines all the reasons for 2229 migrations
    fn compute_2229_migrations_reasons(
        &self,
        auto_trait_reasons: UnordSet<&'static str>,
        drop_order: bool,
    ) -> MigrationWarningReason {
        MigrationWarningReason {
            auto_traits: auto_trait_reasons.into_sorted_stable_ord(),
            drop_order,
        }
    }

    /// Figures out the list of root variables (and their types) that aren't completely
    /// captured by the closure when `capture_disjoint_fields` is enabled and auto-traits
    /// differ between the root variable and the captured paths.
    ///
    /// Returns a tuple containing a HashMap of CapturesInfo that maps to a HashSet of trait names
    /// if migration is needed for traits for the provided var_hir_id, otherwise returns None
    fn compute_2229_migrations_for_trait(
        &self,
        min_captures: Option<&ty::RootVariableMinCaptureList<'tcx>>,
        var_hir_id: HirId,
        closure_clause: hir::CaptureBy,
    ) -> Option<FxIndexMap<UpvarMigrationInfo, UnordSet<&'static str>>> {
        let auto_traits_def_id = [
            self.tcx.lang_items().clone_trait(),
            self.tcx.lang_items().sync_trait(),
            self.tcx.get_diagnostic_item(sym::Send),
            self.tcx.lang_items().unpin_trait(),
            self.tcx.get_diagnostic_item(sym::unwind_safe_trait),
            self.tcx.get_diagnostic_item(sym::ref_unwind_safe_trait),
        ];
        const AUTO_TRAITS: [&str; 6] =
            ["`Clone`", "`Sync`", "`Send`", "`Unpin`", "`UnwindSafe`", "`RefUnwindSafe`"];

        let root_var_min_capture_list = min_captures.and_then(|m| m.get(&var_hir_id))?;

        let ty = self.resolve_vars_if_possible(self.node_ty(var_hir_id));

        let ty = match closure_clause {
            hir::CaptureBy::Value { .. } => ty, // For move closure the capture kind should be by value
            hir::CaptureBy::Ref | hir::CaptureBy::Use { .. } => {
                // For non move closure the capture kind is the max capture kind of all captures
                // according to the ordering ImmBorrow < UniqueImmBorrow < MutBorrow < ByValue
                let mut max_capture_info = root_var_min_capture_list.first().unwrap().info;
                for capture in root_var_min_capture_list.iter() {
                    max_capture_info = determine_capture_info(max_capture_info, capture.info);
                }

                apply_capture_kind_on_capture_ty(
                    self.tcx,
                    ty,
                    max_capture_info.capture_kind,
                    self.tcx.lifetimes.re_erased,
                )
            }
        };

        let mut obligations_should_hold = Vec::new();
        // Checks if a root variable implements any of the auto traits
        for check_trait in auto_traits_def_id.iter() {
            obligations_should_hold.push(check_trait.is_some_and(|check_trait| {
                self.infcx
                    .type_implements_trait(check_trait, [ty], self.param_env)
                    .must_apply_modulo_regions()
            }));
        }

        let mut problematic_captures = FxIndexMap::default();
        // Check whether captured fields also implement the trait
        for capture in root_var_min_capture_list.iter() {
            let ty = apply_capture_kind_on_capture_ty(
                self.tcx,
                capture.place.ty(),
                capture.info.capture_kind,
                self.tcx.lifetimes.re_erased,
            );

            // Checks if a capture implements any of the auto traits
            let mut obligations_holds_for_capture = Vec::new();
            for check_trait in auto_traits_def_id.iter() {
                obligations_holds_for_capture.push(check_trait.is_some_and(|check_trait| {
                    self.infcx
                        .type_implements_trait(check_trait, [ty], self.param_env)
                        .must_apply_modulo_regions()
                }));
            }

            let mut capture_problems = UnordSet::default();

            // Checks if for any of the auto traits, one or more trait is implemented
            // by the root variable but not by the capture
            for (idx, _) in obligations_should_hold.iter().enumerate() {
                if !obligations_holds_for_capture[idx] && obligations_should_hold[idx] {
                    capture_problems.insert(AUTO_TRAITS[idx]);
                }
            }

            if !capture_problems.is_empty() {
                problematic_captures.insert(
                    UpvarMigrationInfo::CapturingPrecise {
                        source_expr: capture.info.path_expr_id,
                        var_name: capture.to_string(self.tcx),
                    },
                    capture_problems,
                );
            }
        }
        if !problematic_captures.is_empty() {
            return Some(problematic_captures);
        }
        None
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
    /// This function only returns a HashSet of CapturesInfo for significant drops. If there
    /// are no significant drops than None is returned
    #[instrument(level = "debug", skip(self))]
    fn compute_2229_migrations_for_drop(
        &self,
        closure_def_id: LocalDefId,
        closure_span: Span,
        min_captures: Option<&ty::RootVariableMinCaptureList<'tcx>>,
        closure_clause: hir::CaptureBy,
        var_hir_id: HirId,
    ) -> Option<FxIndexSet<UpvarMigrationInfo>> {
        let ty = self.resolve_vars_if_possible(self.node_ty(var_hir_id));

        // FIXME(#132279): Using `non_body_analysis` here feels wrong.
        if !ty.has_significant_drop(
            self.tcx,
            ty::TypingEnv::non_body_analysis(self.tcx, closure_def_id),
        ) {
            debug!("does not have significant drop");
            return None;
        }

        let Some(root_var_min_capture_list) = min_captures.and_then(|m| m.get(&var_hir_id)) else {
            // The upvar is mentioned within the closure but no path starting from it is
            // used. This occurs when you have (e.g.)
            //
            // ```
            // let x = move || {
            //     let _ = y;
            // });
            // ```
            debug!("no path starting from it is used");

            match closure_clause {
                // Only migrate if closure is a move closure
                hir::CaptureBy::Value { .. } => {
                    let mut diagnostics_info = FxIndexSet::default();
                    let upvars =
                        self.tcx.upvars_mentioned(closure_def_id).expect("must be an upvar");
                    let upvar = upvars[&var_hir_id];
                    diagnostics_info
                        .insert(UpvarMigrationInfo::CapturingNothing { use_span: upvar.span });
                    return Some(diagnostics_info);
                }
                hir::CaptureBy::Ref | hir::CaptureBy::Use { .. } => {}
            }

            return None;
        };
        debug!(?root_var_min_capture_list);

        let mut projections_list = Vec::new();
        let mut diagnostics_info = FxIndexSet::default();

        for captured_place in root_var_min_capture_list.iter() {
            match captured_place.info.capture_kind {
                // Only care about captures that are moved into the closure
                ty::UpvarCapture::ByValue | ty::UpvarCapture::ByUse => {
                    projections_list.push(captured_place.place.projections.as_slice());
                    diagnostics_info.insert(UpvarMigrationInfo::CapturingPrecise {
                        source_expr: captured_place.info.path_expr_id,
                        var_name: captured_place.to_string(self.tcx),
                    });
                }
                ty::UpvarCapture::ByRef(..) => {}
            }
        }

        debug!(?projections_list);
        debug!(?diagnostics_info);

        let is_moved = !projections_list.is_empty();
        debug!(?is_moved);

        let is_not_completely_captured =
            root_var_min_capture_list.iter().any(|capture| !capture.place.projections.is_empty());
        debug!(?is_not_completely_captured);

        if is_moved
            && is_not_completely_captured
            && self.has_significant_drop_outside_of_captures(
                closure_def_id,
                closure_span,
                ty,
                projections_list,
            )
        {
            return Some(diagnostics_info);
        }

        None
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
    /// Returns a tuple containing a vector of MigrationDiagnosticInfo, as well as a String
    /// containing the reason why root variables whose HirId is contained in the vector should
    /// be captured
    #[instrument(level = "debug", skip(self))]
    fn compute_2229_migrations(
        &self,
        closure_def_id: LocalDefId,
        closure_span: Span,
        closure_clause: hir::CaptureBy,
        min_captures: Option<&ty::RootVariableMinCaptureList<'tcx>>,
    ) -> (Vec<NeededMigration>, MigrationWarningReason) {
        let Some(upvars) = self.tcx.upvars_mentioned(closure_def_id) else {
            return (Vec::new(), MigrationWarningReason::default());
        };

        let mut need_migrations = Vec::new();
        let mut auto_trait_migration_reasons = UnordSet::default();
        let mut drop_migration_needed = false;

        // Perform auto-trait analysis
        for (&var_hir_id, _) in upvars.iter() {
            let mut diagnostics_info = Vec::new();

            let auto_trait_diagnostic = self
                .compute_2229_migrations_for_trait(min_captures, var_hir_id, closure_clause)
                .unwrap_or_default();

            let drop_reorder_diagnostic = if let Some(diagnostics_info) = self
                .compute_2229_migrations_for_drop(
                    closure_def_id,
                    closure_span,
                    min_captures,
                    closure_clause,
                    var_hir_id,
                ) {
                drop_migration_needed = true;
                diagnostics_info
            } else {
                FxIndexSet::default()
            };

            // Combine all the captures responsible for needing migrations into one IndexSet
            let mut capture_diagnostic = drop_reorder_diagnostic.clone();
            for key in auto_trait_diagnostic.keys() {
                capture_diagnostic.insert(key.clone());
            }

            let mut capture_diagnostic = capture_diagnostic.into_iter().collect::<Vec<_>>();
            capture_diagnostic.sort_by_cached_key(|info| match info {
                UpvarMigrationInfo::CapturingPrecise { source_expr: _, var_name } => {
                    (0, Some(var_name.clone()))
                }
                UpvarMigrationInfo::CapturingNothing { use_span: _ } => (1, None),
            });
            for captures_info in capture_diagnostic {
                // Get the auto trait reasons of why migration is needed because of that capture, if there are any
                let capture_trait_reasons =
                    if let Some(reasons) = auto_trait_diagnostic.get(&captures_info) {
                        reasons.clone()
                    } else {
                        UnordSet::default()
                    };

                // Check if migration is needed because of drop reorder as a result of that capture
                let capture_drop_reorder_reason = drop_reorder_diagnostic.contains(&captures_info);

                // Combine all the reasons of why the root variable should be captured as a result of
                // auto trait implementation issues
                auto_trait_migration_reasons.extend_unord(capture_trait_reasons.items().copied());

                diagnostics_info.push(MigrationLintNote {
                    captures_info,
                    reason: self.compute_2229_migrations_reasons(
                        capture_trait_reasons,
                        capture_drop_reorder_reason,
                    ),
                });
            }

            if !diagnostics_info.is_empty() {
                need_migrations.push(NeededMigration { var_hir_id, diagnostics_info });
            }
        }
        (
            need_migrations,
            self.compute_2229_migrations_reasons(
                auto_trait_migration_reasons,
                drop_migration_needed,
            ),
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
    /// ```rust,edition2021
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
    /// ```ignore (illustrative)
    ///                  (Ty(w), [ &[p, x], &[c] ])
    /// //                              |
    /// //                 ----------------------------
    /// //                 |                          |
    /// //                 v                          v
    ///        (Ty(w.p), [ &[x] ])          (Ty(w.c), [ &[] ]) // I(1)
    /// //                 |                          |
    /// //                 v                          v
    ///        (Ty(w.p), [ &[x] ])                 false
    /// //                 |
    /// //                 |
    /// //       -------------------------------
    /// //       |                             |
    /// //       v                             v
    ///     (Ty((w.p).x), [ &[] ])     (Ty((w.p).y), []) // IMP 2
    /// //       |                             |
    /// //       v                             v
    ///        false              NeedsSignificantDrop(Ty(w.p.y))
    /// //                                     |
    /// //                                     v
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
    /// ```ignore (pseudo-rust)
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
        closure_def_id: LocalDefId,
        closure_span: Span,
        base_path_ty: Ty<'tcx>,
        captured_by_move_projs: Vec<&[Projection<'tcx>]>,
    ) -> bool {
        // FIXME(#132279): Using `non_body_analysis` here feels wrong.
        let needs_drop = |ty: Ty<'tcx>| {
            ty.has_significant_drop(
                self.tcx,
                ty::TypingEnv::non_body_analysis(self.tcx, closure_def_id),
            )
        };

        let is_drop_defined_for_ty = |ty: Ty<'tcx>| {
            let drop_trait = self.tcx.require_lang_item(hir::LangItem::Drop, closure_span);
            self.infcx
                .type_implements_trait(drop_trait, [ty], self.tcx.param_env(closure_def_id))
                .must_apply_modulo_regions()
        };

        let is_drop_defined_for_ty = is_drop_defined_for_ty(base_path_ty);

        // If there is a case where no projection is applied on top of current place
        // then there must be exactly one capture corresponding to such a case. Note that this
        // represents the case of the path being completely captured by the variable.
        //
        // eg. If `a.b` is captured and we are processing `a.b`, then we can't have the closure also
        //     capture `a.b.c`, because that violates min capture.
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
            // - All entries in `captured_by_move_projs` have at least one projection.
            //   Therefore we can call `captured_by_move_projs.first().unwrap().first().unwrap()` safely.

            // We don't capture derefs in case of move captures, which would have be applied to
            // access any further paths.
            ty::Adt(def, _) if def.is_box() => unreachable!(),
            ty::Ref(..) => unreachable!(),
            ty::RawPtr(..) => unreachable!(),

            ty::Adt(def, args) => {
                // Multi-variant enums are captured in entirety,
                // which would've been handled in the case of single empty slice in `captured_by_move_projs`.
                assert_eq!(def.variants().len(), 1);

                // Only Field projections can be applied to a non-box Adt.
                assert!(
                    captured_by_move_projs.iter().all(|projs| matches!(
                        projs.first().unwrap().kind,
                        ProjectionKind::Field(..)
                    ))
                );
                def.variants().get(FIRST_VARIANT).unwrap().fields.iter_enumerated().any(
                    |(i, field)| {
                        let paths_using_field = captured_by_move_projs
                            .iter()
                            .filter_map(|projs| {
                                if let ProjectionKind::Field(field_idx, _) =
                                    projs.first().unwrap().kind
                                {
                                    if field_idx == i { Some(&projs[1..]) } else { None }
                                } else {
                                    unreachable!();
                                }
                            })
                            .collect();

                        let after_field_ty = field.ty(self.tcx, args);
                        self.has_significant_drop_outside_of_captures(
                            closure_def_id,
                            closure_span,
                            after_field_ty,
                            paths_using_field,
                        )
                    },
                )
            }

            ty::Tuple(fields) => {
                // Only Field projections can be applied to a tuple.
                assert!(
                    captured_by_move_projs.iter().all(|projs| matches!(
                        projs.first().unwrap().kind,
                        ProjectionKind::Field(..)
                    ))
                );

                fields.iter().enumerate().any(|(i, element_ty)| {
                    let paths_using_field = captured_by_move_projs
                        .iter()
                        .filter_map(|projs| {
                            if let ProjectionKind::Field(field_idx, _) = projs.first().unwrap().kind
                            {
                                if field_idx.index() == i { Some(&projs[1..]) } else { None }
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
    ) -> ty::UpvarCapture {
        match capture_clause {
            // In case of a move closure if the data is accessed through a reference we
            // want to capture by ref to allow precise capture using reborrows.
            //
            // If the data will be moved out of this place, then the place will be truncated
            // at the first Deref in `adjust_for_move_closure` and then moved into the closure.
            //
            // For example:
            //
            // struct Buffer<'a> {
            //     x: &'a String,
            //     y: Vec<u8>,
            // }
            //
            // fn get<'a>(b: Buffer<'a>) -> impl Sized + 'a {
            //     let c = move || b.x;
            //     drop(b);
            //     c
            // }
            //
            // Even though the closure is declared as move, when we are capturing borrowed data (in
            // this case, *b.x) we prefer to capture by reference.
            // Otherwise you'd get an error in 2021 immediately because you'd be trying to take
            // ownership of the (borrowed) String or else you'd take ownership of b, as in 2018 and
            // before, which is also an error.
            hir::CaptureBy::Value { .. } if !place.deref_tys().any(Ty::is_ref) => {
                ty::UpvarCapture::ByValue
            }
            hir::CaptureBy::Use { .. } if !place.deref_tys().any(Ty::is_ref) => {
                ty::UpvarCapture::ByUse
            }
            hir::CaptureBy::Value { .. } | hir::CaptureBy::Use { .. } | hir::CaptureBy::Ref => {
                ty::UpvarCapture::ByRef(BorrowKind::Immutable)
            }
        }
    }

    fn place_for_root_variable(
        &self,
        closure_def_id: LocalDefId,
        var_hir_id: HirId,
    ) -> Place<'tcx> {
        let upvar_id = ty::UpvarId::new(var_hir_id, closure_def_id);

        Place {
            base_ty: self.node_ty(var_hir_id),
            base: PlaceBase::Upvar(upvar_id),
            projections: Default::default(),
        }
    }

    fn should_log_capture_analysis(&self, closure_def_id: LocalDefId) -> bool {
        self.tcx.has_attr(closure_def_id, sym::rustc_capture_analysis)
    }

    fn log_capture_analysis_first_pass(
        &self,
        closure_def_id: LocalDefId,
        capture_information: &InferredCaptureInformation<'tcx>,
        closure_span: Span,
    ) {
        if self.should_log_capture_analysis(closure_def_id) {
            let mut diag =
                self.dcx().struct_span_err(closure_span, "First Pass analysis includes:");
            for (place, capture_info) in capture_information {
                let capture_str = construct_capture_info_string(self.tcx, place, capture_info);
                let output_str = format!("Capturing {capture_str}");

                let span = capture_info.path_expr_id.map_or(closure_span, |e| self.tcx.hir_span(e));
                diag.span_note(span, output_str);
            }
            diag.emit();
        }
    }

    fn log_closure_min_capture_info(&self, closure_def_id: LocalDefId, closure_span: Span) {
        if self.should_log_capture_analysis(closure_def_id) {
            if let Some(min_captures) =
                self.typeck_results.borrow().closure_min_captures.get(&closure_def_id)
            {
                let mut diag =
                    self.dcx().struct_span_err(closure_span, "Min Capture analysis includes:");

                for (_, min_captures_for_var) in min_captures {
                    for capture in min_captures_for_var {
                        let place = &capture.place;
                        let capture_info = &capture.info;

                        let capture_str =
                            construct_capture_info_string(self.tcx, place, capture_info);
                        let output_str = format!("Min Capture {capture_str}");

                        if capture.info.path_expr_id != capture.info.capture_kind_expr_id {
                            let path_span = capture_info
                                .path_expr_id
                                .map_or(closure_span, |e| self.tcx.hir_span(e));
                            let capture_kind_span = capture_info
                                .capture_kind_expr_id
                                .map_or(closure_span, |e| self.tcx.hir_span(e));

                            let mut multi_span: MultiSpan =
                                MultiSpan::from_spans(vec![path_span, capture_kind_span]);

                            let capture_kind_label =
                                construct_capture_kind_reason_string(self.tcx, place, capture_info);
                            let path_label = construct_path_string(self.tcx, place);

                            multi_span.push_span_label(path_span, path_label);
                            multi_span.push_span_label(capture_kind_span, capture_kind_label);

                            diag.span_note(multi_span, output_str);
                        } else {
                            let span = capture_info
                                .path_expr_id
                                .map_or(closure_span, |e| self.tcx.hir_span(e));

                            diag.span_note(span, output_str);
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

        let mut is_mutbl = bm.1;

        for pointer_ty in place.deref_tys() {
            match self.structurally_resolve_type(self.tcx.hir_span(var_hir_id), pointer_ty).kind() {
                // We don't capture derefs of raw ptrs
                ty::RawPtr(_, _) => unreachable!(),

                // Dereferencing a mut-ref allows us to mut the Place if we don't deref
                // an immut-ref after on top of this.
                ty::Ref(.., hir::Mutability::Mut) => is_mutbl = hir::Mutability::Mut,

                // The place isn't mutable once we dereference an immutable reference.
                ty::Ref(.., hir::Mutability::Not) => return hir::Mutability::Not,

                // Dereferencing a box doesn't change mutability
                ty::Adt(def, ..) if def.is_box() => {}

                unexpected_ty => span_bug!(
                    self.tcx.hir_span(var_hir_id),
                    "deref of unexpected pointer type {:?}",
                    unexpected_ty
                ),
            }
        }

        is_mutbl
    }
}

/// Determines whether a child capture that is derived from a parent capture
/// should be borrowed with the lifetime of the parent coroutine-closure's env.
///
/// There are two cases when this needs to happen:
///
/// (1.) Are we borrowing data owned by the parent closure? We can determine if
/// that is the case by checking if the parent capture is by move, EXCEPT if we
/// apply a deref projection of an immutable reference, reborrows of immutable
/// references which aren't restricted to the LUB of the lifetimes of the deref
/// chain. This is why `&'short mut &'long T` can be reborrowed as `&'long T`.
///
/// ```rust
/// let x = &1i32; // Let's call this lifetime `'1`.
/// let c = async move || {
///     println!("{:?}", *x);
///     // Even though the inner coroutine borrows by ref, we're only capturing `*x`,
///     // not `x`, so the inner closure is allowed to reborrow the data for `'1`.
/// };
/// ```
///
/// (2.) If a coroutine is mutably borrowing from a parent capture, then that
/// mutable borrow cannot live for longer than either the parent *or* the borrow
/// that we have on the original upvar. Therefore we always need to borrow the
/// child capture with the lifetime of the parent coroutine-closure's env.
///
/// ```rust
/// let mut x = 1i32;
/// let c = async || {
///     x = 1;
///     // The parent borrows `x` for some `&'1 mut i32`.
///     // However, when we call `c()`, we implicitly autoref for the signature of
///     // `AsyncFnMut::async_call_mut`. Let's call that lifetime `'call`. Since
///     // the maximum that `&'call mut &'1 mut i32` can be reborrowed is `&'call mut i32`,
///     // the inner coroutine should capture w/ the lifetime of the coroutine-closure.
/// };
/// ```
///
/// If either of these cases apply, then we should capture the borrow with the
/// lifetime of the parent coroutine-closure's env. Luckily, if this function is
/// not correct, then the program is not unsound, since we still borrowck and validate
/// the choices made from this function -- the only side-effect is that the user
/// may receive unnecessary borrowck errors.
fn should_reborrow_from_env_of_parent_coroutine_closure<'tcx>(
    parent_capture: &ty::CapturedPlace<'tcx>,
    child_capture: &ty::CapturedPlace<'tcx>,
) -> bool {
    // (1.)
    (!parent_capture.is_by_ref()
        // This is just inlined `place.deref_tys()` but truncated to just
        // the child projections. Namely, look for a `&T` deref, since we
        // can always extend `&'short mut &'long T` to `&'long T`.
        && !child_capture
            .place
            .projections
            .iter()
            .enumerate()
            .skip(parent_capture.place.projections.len())
            .any(|(idx, proj)| {
                matches!(proj.kind, ProjectionKind::Deref)
                    && matches!(
                        child_capture.place.ty_before_projection(idx).kind(),
                        ty::Ref(.., ty::Mutability::Not)
                    )
            }))
        // (2.)
        || matches!(child_capture.info.capture_kind, UpvarCapture::ByRef(ty::BorrowKind::Mutable))
}

/// Truncate the capture so that the place being borrowed is in accordance with RFC 1240,
/// which states that it's unsafe to take a reference into a struct marked `repr(packed)`.
fn restrict_repr_packed_field_ref_capture<'tcx>(
    mut place: Place<'tcx>,
    mut curr_borrow_kind: ty::UpvarCapture,
) -> (Place<'tcx>, ty::UpvarCapture) {
    let pos = place.projections.iter().enumerate().position(|(i, p)| {
        let ty = place.ty_before_projection(i);

        // Return true for fields of packed structs.
        match p.kind {
            ProjectionKind::Field(..) => match ty.kind() {
                ty::Adt(def, _) if def.repr().packed() => {
                    // We stop here regardless of field alignment. Field alignment can change as
                    // types change, including the types of private fields in other crates, and that
                    // shouldn't affect how we compute our captures.
                    true
                }

                _ => false,
            },
            _ => false,
        }
    });

    if let Some(pos) = pos {
        truncate_place_to_len_and_update_capture_kind(&mut place, &mut curr_borrow_kind, pos);
    }

    (place, curr_borrow_kind)
}

/// Returns a Ty that applies the specified capture kind on the provided capture Ty
fn apply_capture_kind_on_capture_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    capture_kind: UpvarCapture,
    region: ty::Region<'tcx>,
) -> Ty<'tcx> {
    match capture_kind {
        ty::UpvarCapture::ByValue | ty::UpvarCapture::ByUse => ty,
        ty::UpvarCapture::ByRef(kind) => Ty::new_ref(tcx, region, ty, kind.to_mutbl_lossy()),
    }
}

/// Returns the Span of where the value with the provided HirId would be dropped
fn drop_location_span(tcx: TyCtxt<'_>, hir_id: HirId) -> Span {
    let owner_id = tcx.hir_get_enclosing_scope(hir_id).unwrap();

    let owner_node = tcx.hir_node(owner_id);
    let owner_span = match owner_node {
        hir::Node::Item(item) => match item.kind {
            hir::ItemKind::Fn { body: owner_id, .. } => tcx.hir_span(owner_id.hir_id),
            _ => {
                bug!("Drop location span error: need to handle more ItemKind '{:?}'", item.kind);
            }
        },
        hir::Node::Block(block) => tcx.hir_span(block.hir_id),
        hir::Node::TraitItem(item) => tcx.hir_span(item.hir_id()),
        hir::Node::ImplItem(item) => tcx.hir_span(item.hir_id()),
        _ => {
            bug!("Drop location span error: need to handle more Node '{:?}'", owner_node);
        }
    };
    tcx.sess.source_map().end_point(owner_span)
}

struct InferBorrowKind<'tcx> {
    // The def-id of the closure whose kind and upvar accesses are being inferred.
    closure_def_id: LocalDefId,

    /// For each Place that is captured by the closure, we track the minimal kind of
    /// access we need (ref, ref mut, move, etc) and the expression that resulted in such access.
    ///
    /// Consider closure where s.str1 is captured via an ImmutableBorrow and
    /// s.str2 via a MutableBorrow
    ///
    /// ```rust,no_run
    /// struct SomeStruct { str1: String, str2: String };
    ///
    /// // Assume that the HirId for the variable definition is `V1`
    /// let mut s = SomeStruct { str1: format!("s1"), str2: format!("s2") };
    ///
    /// let fix_s = |new_s2| {
    ///     // Assume that the HirId for the expression `s.str1` is `E1`
    ///     println!("Updating SomeStruct with str1={0}", s.str1);
    ///     // Assume that the HirId for the expression `*s.str2` is `E2`
    ///     s.str2 = new_s2;
    /// };
    /// ```
    ///
    /// For closure `fix_s`, (at a high level) the map contains
    ///
    /// ```ignore (illustrative)
    /// Place { V1, [ProjectionKind::Field(Index=0, Variant=0)] } : CaptureKind { E1, ImmutableBorrow }
    /// Place { V1, [ProjectionKind::Field(Index=1, Variant=0)] } : CaptureKind { E2, MutableBorrow }
    /// ```
    capture_information: InferredCaptureInformation<'tcx>,
    fake_reads: Vec<(Place<'tcx>, FakeReadCause, HirId)>,
}

impl<'tcx> euv::Delegate<'tcx> for InferBorrowKind<'tcx> {
    fn fake_read(
        &mut self,
        place_with_id: &PlaceWithHirId<'tcx>,
        cause: FakeReadCause,
        diag_expr_id: HirId,
    ) {
        let PlaceBase::Upvar(_) = place_with_id.place.base else { return };

        // We need to restrict Fake Read precision to avoid fake reading unsafe code,
        // such as deref of a raw pointer.
        let dummy_capture_kind = ty::UpvarCapture::ByRef(ty::BorrowKind::Immutable);

        let (place, _) =
            restrict_capture_precision(place_with_id.place.clone(), dummy_capture_kind);

        let (place, _) = restrict_repr_packed_field_ref_capture(place, dummy_capture_kind);
        self.fake_reads.push((place, cause, diag_expr_id));
    }

    #[instrument(skip(self), level = "debug")]
    fn consume(&mut self, place_with_id: &PlaceWithHirId<'tcx>, diag_expr_id: HirId) {
        let PlaceBase::Upvar(upvar_id) = place_with_id.place.base else { return };
        assert_eq!(self.closure_def_id, upvar_id.closure_expr_id);

        self.capture_information.push((
            place_with_id.place.clone(),
            ty::CaptureInfo {
                capture_kind_expr_id: Some(diag_expr_id),
                path_expr_id: Some(diag_expr_id),
                capture_kind: ty::UpvarCapture::ByValue,
            },
        ));
    }

    #[instrument(skip(self), level = "debug")]
    fn use_cloned(&mut self, place_with_id: &PlaceWithHirId<'tcx>, diag_expr_id: HirId) {
        let PlaceBase::Upvar(upvar_id) = place_with_id.place.base else { return };
        assert_eq!(self.closure_def_id, upvar_id.closure_expr_id);

        self.capture_information.push((
            place_with_id.place.clone(),
            ty::CaptureInfo {
                capture_kind_expr_id: Some(diag_expr_id),
                path_expr_id: Some(diag_expr_id),
                capture_kind: ty::UpvarCapture::ByUse,
            },
        ));
    }

    #[instrument(skip(self), level = "debug")]
    fn borrow(
        &mut self,
        place_with_id: &PlaceWithHirId<'tcx>,
        diag_expr_id: HirId,
        bk: ty::BorrowKind,
    ) {
        let PlaceBase::Upvar(upvar_id) = place_with_id.place.base else { return };
        assert_eq!(self.closure_def_id, upvar_id.closure_expr_id);

        // The region here will get discarded/ignored
        let capture_kind = ty::UpvarCapture::ByRef(bk);

        // We only want repr packed restriction to be applied to reading references into a packed
        // struct, and not when the data is being moved. Therefore we call this method here instead
        // of in `restrict_capture_precision`.
        let (place, mut capture_kind) =
            restrict_repr_packed_field_ref_capture(place_with_id.place.clone(), capture_kind);

        // Raw pointers don't inherit mutability
        if place_with_id.place.deref_tys().any(Ty::is_raw_ptr) {
            capture_kind = ty::UpvarCapture::ByRef(ty::BorrowKind::Immutable);
        }

        self.capture_information.push((
            place,
            ty::CaptureInfo {
                capture_kind_expr_id: Some(diag_expr_id),
                path_expr_id: Some(diag_expr_id),
                capture_kind,
            },
        ));
    }

    #[instrument(skip(self), level = "debug")]
    fn mutate(&mut self, assignee_place: &PlaceWithHirId<'tcx>, diag_expr_id: HirId) {
        self.borrow(assignee_place, diag_expr_id, ty::BorrowKind::Mutable);
    }
}

/// Rust doesn't permit moving fields out of a type that implements drop
fn restrict_precision_for_drop_types<'a, 'tcx>(
    fcx: &'a FnCtxt<'a, 'tcx>,
    mut place: Place<'tcx>,
    mut curr_mode: ty::UpvarCapture,
) -> (Place<'tcx>, ty::UpvarCapture) {
    let is_copy_type = fcx.infcx.type_is_copy_modulo_regions(fcx.param_env, place.ty());

    if let (false, UpvarCapture::ByValue) = (is_copy_type, curr_mode) {
        for i in 0..place.projections.len() {
            match place.ty_before_projection(i).kind() {
                ty::Adt(def, _) if def.destructor(fcx.tcx).is_some() => {
                    truncate_place_to_len_and_update_capture_kind(&mut place, &mut curr_mode, i);
                    break;
                }
                _ => {}
            }
        }
    }

    (place, curr_mode)
}

/// Truncate `place` so that an `unsafe` block isn't required to capture it.
/// - No projections are applied to raw pointers, since these require unsafe blocks. We capture
///   them completely.
/// - No projections are applied on top of Union ADTs, since these require unsafe blocks.
fn restrict_precision_for_unsafe(
    mut place: Place<'_>,
    mut curr_mode: ty::UpvarCapture,
) -> (Place<'_>, ty::UpvarCapture) {
    if place.base_ty.is_raw_ptr() {
        truncate_place_to_len_and_update_capture_kind(&mut place, &mut curr_mode, 0);
    }

    if place.base_ty.is_union() {
        truncate_place_to_len_and_update_capture_kind(&mut place, &mut curr_mode, 0);
    }

    for (i, proj) in place.projections.iter().enumerate() {
        if proj.ty.is_raw_ptr() {
            // Don't apply any projections on top of a raw ptr.
            truncate_place_to_len_and_update_capture_kind(&mut place, &mut curr_mode, i + 1);
            break;
        }

        if proj.ty.is_union() {
            // Don't capture precise fields of a union.
            truncate_place_to_len_and_update_capture_kind(&mut place, &mut curr_mode, i + 1);
            break;
        }
    }

    (place, curr_mode)
}

/// Truncate projections so that following rules are obeyed by the captured `place`:
/// - No Index projections are captured, since arrays are captured completely.
/// - No unsafe block is required to capture `place`
/// Returns the truncated place and updated capture mode.
fn restrict_capture_precision(
    place: Place<'_>,
    curr_mode: ty::UpvarCapture,
) -> (Place<'_>, ty::UpvarCapture) {
    let (mut place, mut curr_mode) = restrict_precision_for_unsafe(place, curr_mode);

    if place.projections.is_empty() {
        // Nothing to do here
        return (place, curr_mode);
    }

    for (i, proj) in place.projections.iter().enumerate() {
        match proj.kind {
            ProjectionKind::Index | ProjectionKind::Subslice => {
                // Arrays are completely captured, so we drop Index and Subslice projections
                truncate_place_to_len_and_update_capture_kind(&mut place, &mut curr_mode, i);
                return (place, curr_mode);
            }
            ProjectionKind::Deref => {}
            ProjectionKind::OpaqueCast => {}
            ProjectionKind::Field(..) => {}
            ProjectionKind::UnwrapUnsafeBinder => {}
        }
    }

    (place, curr_mode)
}

/// Truncate deref of any reference.
fn adjust_for_move_closure(
    mut place: Place<'_>,
    mut kind: ty::UpvarCapture,
) -> (Place<'_>, ty::UpvarCapture) {
    let first_deref = place.projections.iter().position(|proj| proj.kind == ProjectionKind::Deref);

    if let Some(idx) = first_deref {
        truncate_place_to_len_and_update_capture_kind(&mut place, &mut kind, idx);
    }

    (place, ty::UpvarCapture::ByValue)
}

/// Truncate deref of any reference.
fn adjust_for_use_closure(
    mut place: Place<'_>,
    mut kind: ty::UpvarCapture,
) -> (Place<'_>, ty::UpvarCapture) {
    let first_deref = place.projections.iter().position(|proj| proj.kind == ProjectionKind::Deref);

    if let Some(idx) = first_deref {
        truncate_place_to_len_and_update_capture_kind(&mut place, &mut kind, idx);
    }

    (place, ty::UpvarCapture::ByUse)
}

/// Adjust closure capture just that if taking ownership of data, only move data
/// from enclosing stack frame.
fn adjust_for_non_move_closure(
    mut place: Place<'_>,
    mut kind: ty::UpvarCapture,
) -> (Place<'_>, ty::UpvarCapture) {
    let contains_deref =
        place.projections.iter().position(|proj| proj.kind == ProjectionKind::Deref);

    match kind {
        ty::UpvarCapture::ByValue | ty::UpvarCapture::ByUse => {
            if let Some(idx) = contains_deref {
                truncate_place_to_len_and_update_capture_kind(&mut place, &mut kind, idx);
            }
        }

        ty::UpvarCapture::ByRef(..) => {}
    }

    (place, kind)
}

fn construct_place_string<'tcx>(tcx: TyCtxt<'_>, place: &Place<'tcx>) -> String {
    let variable_name = match place.base {
        PlaceBase::Upvar(upvar_id) => var_name(tcx, upvar_id.var_path.hir_id).to_string(),
        _ => bug!("Capture_information should only contain upvars"),
    };

    let mut projections_str = String::new();
    for (i, item) in place.projections.iter().enumerate() {
        let proj = match item.kind {
            ProjectionKind::Field(a, b) => format!("({a:?}, {b:?})"),
            ProjectionKind::Deref => String::from("Deref"),
            ProjectionKind::Index => String::from("Index"),
            ProjectionKind::Subslice => String::from("Subslice"),
            ProjectionKind::OpaqueCast => String::from("OpaqueCast"),
            ProjectionKind::UnwrapUnsafeBinder => String::from("UnwrapUnsafeBinder"),
        };
        if i != 0 {
            projections_str.push(',');
        }
        projections_str.push_str(proj.as_str());
    }

    format!("{variable_name}[{projections_str}]")
}

fn construct_capture_kind_reason_string<'tcx>(
    tcx: TyCtxt<'_>,
    place: &Place<'tcx>,
    capture_info: &ty::CaptureInfo,
) -> String {
    let place_str = construct_place_string(tcx, place);

    let capture_kind_str = match capture_info.capture_kind {
        ty::UpvarCapture::ByValue => "ByValue".into(),
        ty::UpvarCapture::ByUse => "ByUse".into(),
        ty::UpvarCapture::ByRef(kind) => format!("{kind:?}"),
    };

    format!("{place_str} captured as {capture_kind_str} here")
}

fn construct_path_string<'tcx>(tcx: TyCtxt<'_>, place: &Place<'tcx>) -> String {
    let place_str = construct_place_string(tcx, place);

    format!("{place_str} used here")
}

fn construct_capture_info_string<'tcx>(
    tcx: TyCtxt<'_>,
    place: &Place<'tcx>,
    capture_info: &ty::CaptureInfo,
) -> String {
    let place_str = construct_place_string(tcx, place);

    let capture_kind_str = match capture_info.capture_kind {
        ty::UpvarCapture::ByValue => "ByValue".into(),
        ty::UpvarCapture::ByUse => "ByUse".into(),
        ty::UpvarCapture::ByRef(kind) => format!("{kind:?}"),
    };
    format!("{place_str} -> {capture_kind_str}")
}

fn var_name(tcx: TyCtxt<'_>, var_hir_id: HirId) -> Symbol {
    tcx.hir_name(var_hir_id)
}

#[instrument(level = "debug", skip(tcx))]
fn should_do_rust_2021_incompatible_closure_captures_analysis(
    tcx: TyCtxt<'_>,
    closure_id: HirId,
) -> bool {
    if tcx.sess.at_least_rust_2021() {
        return false;
    }

    let level = tcx
        .lint_level_at_node(lint::builtin::RUST_2021_INCOMPATIBLE_CLOSURE_CAPTURES, closure_id)
        .level;

    !matches!(level, lint::Level::Allow)
}

/// Return a two string tuple (s1, s2)
/// - s1: Line of code that is needed for the migration: eg: `let _ = (&x, ...)`.
/// - s2: Comma separated names of the variables being migrated.
fn migration_suggestion_for_2229(
    tcx: TyCtxt<'_>,
    need_migrations: &[NeededMigration],
) -> (String, String) {
    let need_migrations_variables = need_migrations
        .iter()
        .map(|NeededMigration { var_hir_id: v, .. }| var_name(tcx, *v))
        .collect::<Vec<_>>();

    let migration_ref_concat =
        need_migrations_variables.iter().map(|v| format!("&{v}")).collect::<Vec<_>>().join(", ");

    let migration_string = if 1 == need_migrations.len() {
        format!("let _ = {migration_ref_concat}")
    } else {
        format!("let _ = ({migration_ref_concat})")
    };

    let migrated_variables_concat =
        need_migrations_variables.iter().map(|v| format!("`{v}`")).collect::<Vec<_>>().join(", ");

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
/// then `CaptureInfo` A is preferred. This can be useful in cases where we want to prioritize
/// expressions reported back to the user as part of diagnostics based on which appears earlier
/// in the closure. This can be achieved simply by calling
/// `determine_capture_info(existing_info, current_info)`. This works out because the
/// expressions that occur earlier in the closure body than the current expression are processed before.
/// Consider the following example
/// ```rust,no_run
/// struct Point { x: i32, y: i32 }
/// let mut p = Point { x: 10, y: 10 };
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
    capture_info_a: ty::CaptureInfo,
    capture_info_b: ty::CaptureInfo,
) -> ty::CaptureInfo {
    // If the capture kind is equivalent then, we don't need to escalate and can compare the
    // expressions.
    let eq_capture_kind = match (capture_info_a.capture_kind, capture_info_b.capture_kind) {
        (ty::UpvarCapture::ByValue, ty::UpvarCapture::ByValue) => true,
        (ty::UpvarCapture::ByUse, ty::UpvarCapture::ByUse) => true,
        (ty::UpvarCapture::ByRef(ref_a), ty::UpvarCapture::ByRef(ref_b)) => ref_a == ref_b,
        (ty::UpvarCapture::ByValue, _)
        | (ty::UpvarCapture::ByUse, _)
        | (ty::UpvarCapture::ByRef(_), _) => false,
    };

    if eq_capture_kind {
        match (capture_info_a.capture_kind_expr_id, capture_info_b.capture_kind_expr_id) {
            (Some(_), _) | (None, None) => capture_info_a,
            (None, Some(_)) => capture_info_b,
        }
    } else {
        // We select the CaptureKind which ranks higher based the following priority order:
        // (ByUse | ByValue) > MutBorrow > UniqueImmBorrow > ImmBorrow
        match (capture_info_a.capture_kind, capture_info_b.capture_kind) {
            (ty::UpvarCapture::ByUse, ty::UpvarCapture::ByValue)
            | (ty::UpvarCapture::ByValue, ty::UpvarCapture::ByUse) => {
                bug!("Same capture can't be ByUse and ByValue at the same time")
            }
            (ty::UpvarCapture::ByValue, ty::UpvarCapture::ByValue)
            | (ty::UpvarCapture::ByUse, ty::UpvarCapture::ByUse)
            | (ty::UpvarCapture::ByValue | ty::UpvarCapture::ByUse, ty::UpvarCapture::ByRef(_)) => {
                capture_info_a
            }
            (ty::UpvarCapture::ByRef(_), ty::UpvarCapture::ByValue | ty::UpvarCapture::ByUse) => {
                capture_info_b
            }
            (ty::UpvarCapture::ByRef(ref_a), ty::UpvarCapture::ByRef(ref_b)) => {
                match (ref_a, ref_b) {
                    // Take LHS:
                    (BorrowKind::UniqueImmutable | BorrowKind::Mutable, BorrowKind::Immutable)
                    | (BorrowKind::Mutable, BorrowKind::UniqueImmutable) => capture_info_a,

                    // Take RHS:
                    (BorrowKind::Immutable, BorrowKind::UniqueImmutable | BorrowKind::Mutable)
                    | (BorrowKind::UniqueImmutable, BorrowKind::Mutable) => capture_info_b,

                    (BorrowKind::Immutable, BorrowKind::Immutable)
                    | (BorrowKind::UniqueImmutable, BorrowKind::UniqueImmutable)
                    | (BorrowKind::Mutable, BorrowKind::Mutable) => {
                        bug!("Expected unequal capture kinds");
                    }
                }
            }
        }
    }
}

/// Truncates `place` to have up to `len` projections.
/// `curr_mode` is the current required capture kind for the place.
/// Returns the truncated `place` and the updated required capture kind.
///
/// Note: Capture kind changes from `MutBorrow` to `UniqueImmBorrow` if the truncated part of the `place`
/// contained `Deref` of `&mut`.
fn truncate_place_to_len_and_update_capture_kind<'tcx>(
    place: &mut Place<'tcx>,
    curr_mode: &mut ty::UpvarCapture,
    len: usize,
) {
    let is_mut_ref = |ty: Ty<'_>| matches!(ty.kind(), ty::Ref(.., hir::Mutability::Mut));

    // If the truncated part of the place contains `Deref` of a `&mut` then convert MutBorrow ->
    // UniqueImmBorrow
    // Note that if the place contained Deref of a raw pointer it would've not been MutBorrow, so
    // we don't need to worry about that case here.
    match curr_mode {
        ty::UpvarCapture::ByRef(ty::BorrowKind::Mutable) => {
            for i in len..place.projections.len() {
                if place.projections[i].kind == ProjectionKind::Deref
                    && is_mut_ref(place.ty_before_projection(i))
                {
                    *curr_mode = ty::UpvarCapture::ByRef(ty::BorrowKind::UniqueImmutable);
                    break;
                }
            }
        }

        ty::UpvarCapture::ByRef(..) => {}
        ty::UpvarCapture::ByValue | ty::UpvarCapture::ByUse => {}
    }

    place.projections.truncate(len);
}

/// Determines the Ancestry relationship of Place A relative to Place B
///
/// `PlaceAncestryRelation::Ancestor` implies Place A is ancestor of Place B
/// `PlaceAncestryRelation::Descendant` implies Place A is descendant of Place B
/// `PlaceAncestryRelation::Divergent` implies neither of them is the ancestor of the other.
fn determine_place_ancestry_relation<'tcx>(
    place_a: &Place<'tcx>,
    place_b: &Place<'tcx>,
) -> PlaceAncestryRelation {
    // If Place A and Place B don't start off from the same root variable, they are divergent.
    if place_a.base != place_b.base {
        return PlaceAncestryRelation::Divergent;
    }

    // Assume of length of projections_a = n
    let projections_a = &place_a.projections;

    // Assume of length of projections_b = m
    let projections_b = &place_b.projections;

    let same_initial_projections =
        iter::zip(projections_a, projections_b).all(|(proj_a, proj_b)| proj_a.kind == proj_b.kind);

    if same_initial_projections {
        use std::cmp::Ordering;

        // First min(n, m) projections are the same
        // Select Ancestor/Descendant
        match projections_b.len().cmp(&projections_a.len()) {
            Ordering::Greater => PlaceAncestryRelation::Ancestor,
            Ordering::Equal => PlaceAncestryRelation::SamePlace,
            Ordering::Less => PlaceAncestryRelation::Descendant,
        }
    } else {
        PlaceAncestryRelation::Divergent
    }
}

/// Reduces the precision of the captured place when the precision doesn't yield any benefit from
/// borrow checking perspective, allowing us to save us on the size of the capture.
///
///
/// Fields that are read through a shared reference will always be read via a shared ref or a copy,
/// and therefore capturing precise paths yields no benefit. This optimization truncates the
/// rightmost deref of the capture if the deref is applied to a shared ref.
///
/// Reason we only drop the last deref is because of the following edge case:
///
/// ```
/// # struct A { field_of_a: Box<i32> }
/// # struct B {}
/// # struct C<'a>(&'a i32);
/// struct MyStruct<'a> {
///    a: &'static A,
///    b: B,
///    c: C<'a>,
/// }
///
/// fn foo<'a, 'b>(m: &'a MyStruct<'b>) -> impl FnMut() + 'static {
///     || drop(&*m.a.field_of_a)
///     // Here we really do want to capture `*m.a` because that outlives `'static`
///
///     // If we capture `m`, then the closure no longer outlives `'static`
///     // it is constrained to `'a`
/// }
/// ```
fn truncate_capture_for_optimization(
    mut place: Place<'_>,
    mut curr_mode: ty::UpvarCapture,
) -> (Place<'_>, ty::UpvarCapture) {
    let is_shared_ref = |ty: Ty<'_>| matches!(ty.kind(), ty::Ref(.., hir::Mutability::Not));

    // Find the rightmost deref (if any). All the projections that come after this
    // are fields or other "in-place pointer adjustments"; these refer therefore to
    // data owned by whatever pointer is being dereferenced here.
    let idx = place.projections.iter().rposition(|proj| ProjectionKind::Deref == proj.kind);

    match idx {
        // If that pointer is a shared reference, then we don't need those fields.
        Some(idx) if is_shared_ref(place.ty_before_projection(idx)) => {
            truncate_place_to_len_and_update_capture_kind(&mut place, &mut curr_mode, idx + 1)
        }
        None | Some(_) => {}
    }

    (place, curr_mode)
}

/// Precise capture is enabled if user is using Rust Edition 2021 or higher.
/// `span` is the span of the closure.
fn enable_precise_capture(span: Span) -> bool {
    // We use span here to ensure that if the closure was generated by a macro with a different
    // edition.
    span.at_least_rust_2021()
}
