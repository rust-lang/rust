// ignore-tidy-filelength

#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]

use std::iter;
use std::ops::ControlFlow;

use either::Either;
use hir::{ClosureKind, Path};
use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::codes::*;
use rustc_errors::{Applicability, Diag, MultiSpan, struct_span_code_err};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit::{Visitor, walk_block, walk_expr};
use rustc_hir::{CoroutineDesugaring, CoroutineKind, CoroutineSource, LangItem, PatField};
use rustc_middle::bug;
use rustc_middle::hir::nested_filter::OnlyBodies;
use rustc_middle::mir::{
    self, AggregateKind, BindingForm, BorrowKind, ClearCrossCrate, ConstraintCategory,
    FakeBorrowKind, FakeReadCause, LocalDecl, LocalInfo, LocalKind, Location, MutBorrowKind,
    Operand, Place, PlaceRef, PlaceTy, ProjectionElem, Rvalue, Statement, StatementKind,
    Terminator, TerminatorKind, VarBindingForm, VarDebugInfoContents,
};
use rustc_middle::ty::print::PrintTraitRefExt as _;
use rustc_middle::ty::{
    self, PredicateKind, Ty, TyCtxt, TypeSuperVisitable, TypeVisitor, Upcast,
    suggest_constraining_type_params,
};
use rustc_mir_dataflow::move_paths::{InitKind, MoveOutIndex, MovePathIndex};
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_span::hygiene::DesugaringKind;
use rustc_span::{BytePos, Ident, Span, Symbol, kw, sym};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::error_reporting::traits::FindExprBySpan;
use rustc_trait_selection::error_reporting::traits::call_kind::CallKind;
use rustc_trait_selection::infer::InferCtxtExt;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt as _;
use rustc_trait_selection::traits::{
    Obligation, ObligationCause, ObligationCtxt, supertrait_def_ids,
};
use tracing::{debug, instrument};

use super::explain_borrow::{BorrowExplanation, LaterUseKind};
use super::{DescribePlaceOpt, RegionName, RegionNameSource, UseSpans};
use crate::borrow_set::{BorrowData, TwoPhaseActivation};
use crate::diagnostics::conflict_errors::StorageDeadOrDrop::LocalStorageDead;
use crate::diagnostics::{CapturedMessageOpt, call_kind, find_all_local_uses};
use crate::prefixes::IsPrefixOf;
use crate::{InitializationRequiringAction, MirBorrowckCtxt, WriteKind, borrowck_errors};

#[derive(Debug)]
struct MoveSite {
    /// Index of the "move out" that we found. The `MoveData` can
    /// then tell us where the move occurred.
    moi: MoveOutIndex,

    /// `true` if we traversed a back edge while walking from the point
    /// of error to the move site.
    traversed_back_edge: bool,
}

/// Which case a StorageDeadOrDrop is for.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum StorageDeadOrDrop<'tcx> {
    LocalStorageDead,
    BoxedStorageDead,
    Destructor(Ty<'tcx>),
}

impl<'infcx, 'tcx> MirBorrowckCtxt<'_, 'infcx, 'tcx> {
    pub(crate) fn report_use_of_moved_or_uninitialized(
        &mut self,
        location: Location,
        desired_action: InitializationRequiringAction,
        (moved_place, used_place, span): (PlaceRef<'tcx>, PlaceRef<'tcx>, Span),
        mpi: MovePathIndex,
    ) {
        debug!(
            "report_use_of_moved_or_uninitialized: location={:?} desired_action={:?} \
             moved_place={:?} used_place={:?} span={:?} mpi={:?}",
            location, desired_action, moved_place, used_place, span, mpi
        );

        let use_spans =
            self.move_spans(moved_place, location).or_else(|| self.borrow_spans(span, location));
        let span = use_spans.args_or_use();

        let (move_site_vec, maybe_reinitialized_locations) = self.get_moved_indexes(location, mpi);
        debug!(
            "report_use_of_moved_or_uninitialized: move_site_vec={:?} use_spans={:?}",
            move_site_vec, use_spans
        );
        let move_out_indices: Vec<_> =
            move_site_vec.iter().map(|move_site| move_site.moi).collect();

        if move_out_indices.is_empty() {
            let root_local = used_place.local;

            if !self.uninitialized_error_reported.insert(root_local) {
                debug!(
                    "report_use_of_moved_or_uninitialized place: error about {:?} suppressed",
                    root_local
                );
                return;
            }

            let err = self.report_use_of_uninitialized(
                mpi,
                used_place,
                moved_place,
                desired_action,
                span,
                use_spans,
            );
            self.buffer_error(err);
        } else {
            if let Some((reported_place, _)) = self.has_move_error(&move_out_indices) {
                if used_place.is_prefix_of(*reported_place) {
                    debug!(
                        "report_use_of_moved_or_uninitialized place: error suppressed mois={:?}",
                        move_out_indices
                    );
                    return;
                }
            }

            let is_partial_move = move_site_vec.iter().any(|move_site| {
                let move_out = self.move_data.moves[(*move_site).moi];
                let moved_place = &self.move_data.move_paths[move_out.path].place;
                // `*(_1)` where `_1` is a `Box` is actually a move out.
                let is_box_move = moved_place.as_ref().projection == [ProjectionElem::Deref]
                    && self.body.local_decls[moved_place.local].ty.is_box();

                !is_box_move
                    && used_place != moved_place.as_ref()
                    && used_place.is_prefix_of(moved_place.as_ref())
            });

            let partial_str = if is_partial_move { "partial " } else { "" };
            let partially_str = if is_partial_move { "partially " } else { "" };

            let mut err = self.cannot_act_on_moved_value(
                span,
                desired_action.as_noun(),
                partially_str,
                self.describe_place_with_options(
                    moved_place,
                    DescribePlaceOpt { including_downcast: true, including_tuple_field: true },
                ),
            );

            let reinit_spans = maybe_reinitialized_locations
                .iter()
                .take(3)
                .map(|loc| {
                    self.move_spans(self.move_data.move_paths[mpi].place.as_ref(), *loc)
                        .args_or_use()
                })
                .collect::<Vec<Span>>();

            let reinits = maybe_reinitialized_locations.len();
            if reinits == 1 {
                err.span_label(reinit_spans[0], "this reinitialization might get skipped");
            } else if reinits > 1 {
                err.span_note(
                    MultiSpan::from_spans(reinit_spans),
                    if reinits <= 3 {
                        format!("these {reinits} reinitializations might get skipped")
                    } else {
                        format!(
                            "these 3 reinitializations and {} other{} might get skipped",
                            reinits - 3,
                            if reinits == 4 { "" } else { "s" }
                        )
                    },
                );
            }

            let closure = self.add_moved_or_invoked_closure_note(location, used_place, &mut err);

            let mut is_loop_move = false;
            let mut seen_spans = FxIndexSet::default();

            for move_site in &move_site_vec {
                let move_out = self.move_data.moves[(*move_site).moi];
                let moved_place = &self.move_data.move_paths[move_out.path].place;

                let move_spans = self.move_spans(moved_place.as_ref(), move_out.source);
                let move_span = move_spans.args_or_use();

                let is_move_msg = move_spans.for_closure();

                let is_loop_message = location == move_out.source || move_site.traversed_back_edge;

                if location == move_out.source {
                    is_loop_move = true;
                }

                let mut has_suggest_reborrow = false;
                if !seen_spans.contains(&move_span) {
                    self.suggest_ref_or_clone(
                        mpi,
                        &mut err,
                        move_spans,
                        moved_place.as_ref(),
                        &mut has_suggest_reborrow,
                        closure,
                    );

                    let msg_opt = CapturedMessageOpt {
                        is_partial_move,
                        is_loop_message,
                        is_move_msg,
                        is_loop_move,
                        has_suggest_reborrow,
                        maybe_reinitialized_locations_is_empty: maybe_reinitialized_locations
                            .is_empty(),
                    };
                    self.explain_captures(
                        &mut err,
                        span,
                        move_span,
                        move_spans,
                        *moved_place,
                        msg_opt,
                    );
                }
                seen_spans.insert(move_span);
            }

            use_spans.var_path_only_subdiag(&mut err, desired_action);

            if !is_loop_move {
                err.span_label(
                    span,
                    format!(
                        "value {} here after {partial_str}move",
                        desired_action.as_verb_in_past_tense(),
                    ),
                );
            }

            let ty = used_place.ty(self.body, self.infcx.tcx).ty;
            let needs_note = match ty.kind() {
                ty::Closure(id, _) => {
                    self.infcx.tcx.closure_kind_origin(id.expect_local()).is_none()
                }
                _ => true,
            };

            let mpi = self.move_data.moves[move_out_indices[0]].path;
            let place = &self.move_data.move_paths[mpi].place;
            let ty = place.ty(self.body, self.infcx.tcx).ty;

            if self.infcx.param_env.caller_bounds().iter().any(|c| {
                c.as_trait_clause().is_some_and(|pred| {
                    pred.skip_binder().self_ty() == ty && self.infcx.tcx.is_fn_trait(pred.def_id())
                })
            }) {
                // Suppress the next suggestion since we don't want to put more bounds onto
                // something that already has `Fn`-like bounds (or is a closure), so we can't
                // restrict anyways.
            } else {
                let copy_did = self.infcx.tcx.require_lang_item(LangItem::Copy, span);
                self.suggest_adding_bounds(&mut err, ty, copy_did, span);
            }

            let opt_name = self.describe_place_with_options(
                place.as_ref(),
                DescribePlaceOpt { including_downcast: true, including_tuple_field: true },
            );
            let note_msg = match opt_name {
                Some(name) => format!("`{name}`"),
                None => "value".to_owned(),
            };
            if needs_note {
                if let Some(local) = place.as_local() {
                    let span = self.body.local_decls[local].source_info.span;
                    err.subdiagnostic(crate::session_diagnostics::TypeNoCopy::Label {
                        is_partial_move,
                        ty,
                        place: &note_msg,
                        span,
                    });
                } else {
                    err.subdiagnostic(crate::session_diagnostics::TypeNoCopy::Note {
                        is_partial_move,
                        ty,
                        place: &note_msg,
                    });
                };
            }

            if let UseSpans::FnSelfUse {
                kind: CallKind::DerefCoercion { deref_target_span, deref_target_ty, .. },
                ..
            } = use_spans
            {
                err.note(format!(
                    "{} occurs due to deref coercion to `{deref_target_ty}`",
                    desired_action.as_noun(),
                ));

                // Check first whether the source is accessible (issue #87060)
                if let Some(deref_target_span) = deref_target_span
                    && self.infcx.tcx.sess.source_map().is_span_accessible(deref_target_span)
                {
                    err.span_note(deref_target_span, "deref defined here");
                }
            }

            self.buffer_move_error(move_out_indices, (used_place, err));
        }
    }

    fn suggest_ref_or_clone(
        &self,
        mpi: MovePathIndex,
        err: &mut Diag<'infcx>,
        move_spans: UseSpans<'tcx>,
        moved_place: PlaceRef<'tcx>,
        has_suggest_reborrow: &mut bool,
        moved_or_invoked_closure: bool,
    ) {
        let move_span = match move_spans {
            UseSpans::ClosureUse { capture_kind_span, .. } => capture_kind_span,
            _ => move_spans.args_or_use(),
        };
        struct ExpressionFinder<'hir> {
            expr_span: Span,
            expr: Option<&'hir hir::Expr<'hir>>,
            pat: Option<&'hir hir::Pat<'hir>>,
            parent_pat: Option<&'hir hir::Pat<'hir>>,
            tcx: TyCtxt<'hir>,
        }
        impl<'hir> Visitor<'hir> for ExpressionFinder<'hir> {
            type NestedFilter = OnlyBodies;

            fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
                self.tcx
            }

            fn visit_expr(&mut self, e: &'hir hir::Expr<'hir>) {
                if e.span == self.expr_span {
                    self.expr = Some(e);
                }
                hir::intravisit::walk_expr(self, e);
            }
            fn visit_pat(&mut self, p: &'hir hir::Pat<'hir>) {
                if p.span == self.expr_span {
                    self.pat = Some(p);
                }
                if let hir::PatKind::Binding(hir::BindingMode::NONE, _, i, sub) = p.kind {
                    if i.span == self.expr_span || p.span == self.expr_span {
                        self.pat = Some(p);
                    }
                    // Check if we are in a situation of `ident @ ident` where we want to suggest
                    // `ref ident @ ref ident` or `ref ident @ Struct { ref ident }`.
                    if let Some(subpat) = sub
                        && self.pat.is_none()
                    {
                        self.visit_pat(subpat);
                        if self.pat.is_some() {
                            self.parent_pat = Some(p);
                        }
                        return;
                    }
                }
                hir::intravisit::walk_pat(self, p);
            }
        }
        let tcx = self.infcx.tcx;
        if let Some(body) = tcx.hir_maybe_body_owned_by(self.mir_def_id()) {
            let expr = body.value;
            let place = &self.move_data.move_paths[mpi].place;
            let span = place.as_local().map(|local| self.body.local_decls[local].source_info.span);
            let mut finder = ExpressionFinder {
                expr_span: move_span,
                expr: None,
                pat: None,
                parent_pat: None,
                tcx,
            };
            finder.visit_expr(expr);
            if let Some(span) = span
                && let Some(expr) = finder.expr
            {
                for (_, expr) in tcx.hir_parent_iter(expr.hir_id) {
                    if let hir::Node::Expr(expr) = expr {
                        if expr.span.contains(span) {
                            // If the let binding occurs within the same loop, then that
                            // loop isn't relevant, like in the following, the outermost `loop`
                            // doesn't play into `x` being moved.
                            // ```
                            // loop {
                            //     let x = String::new();
                            //     loop {
                            //         foo(x);
                            //     }
                            // }
                            // ```
                            break;
                        }
                        if let hir::ExprKind::Loop(.., loop_span) = expr.kind {
                            err.span_label(loop_span, "inside of this loop");
                        }
                    }
                }
                let typeck = self.infcx.tcx.typeck(self.mir_def_id());
                let parent = self.infcx.tcx.parent_hir_node(expr.hir_id);
                let (def_id, call_id, args, offset) = if let hir::Node::Expr(parent_expr) = parent
                    && let hir::ExprKind::MethodCall(_, _, args, _) = parent_expr.kind
                {
                    let def_id = typeck.type_dependent_def_id(parent_expr.hir_id);
                    (def_id, Some(parent_expr.hir_id), args, 1)
                } else if let hir::Node::Expr(parent_expr) = parent
                    && let hir::ExprKind::Call(call, args) = parent_expr.kind
                    && let ty::FnDef(def_id, _) = typeck.node_type(call.hir_id).kind()
                {
                    (Some(*def_id), Some(call.hir_id), args, 0)
                } else {
                    (None, None, &[][..], 0)
                };
                let ty = place.ty(self.body, self.infcx.tcx).ty;

                let mut can_suggest_clone = true;
                if let Some(def_id) = def_id
                    && let Some(pos) = args.iter().position(|arg| arg.hir_id == expr.hir_id)
                {
                    // The move occurred as one of the arguments to a function call. Is that
                    // argument generic? `def_id` can't be a closure here, so using `fn_sig` is fine
                    let arg_param = if self.infcx.tcx.def_kind(def_id).is_fn_like()
                        && let sig =
                            self.infcx.tcx.fn_sig(def_id).instantiate_identity().skip_binder()
                        && let Some(arg_ty) = sig.inputs().get(pos + offset)
                        && let ty::Param(arg_param) = arg_ty.kind()
                    {
                        Some(arg_param)
                    } else {
                        None
                    };

                    // If the moved value is a mut reference, it is used in a
                    // generic function and it's type is a generic param, it can be
                    // reborrowed to avoid moving.
                    // for example:
                    // struct Y(u32);
                    // x's type is '& mut Y' and it is used in `fn generic<T>(x: T) {}`.
                    if let ty::Ref(_, _, hir::Mutability::Mut) = ty.kind()
                        && arg_param.is_some()
                    {
                        *has_suggest_reborrow = true;
                        self.suggest_reborrow(err, expr.span, moved_place);
                        return;
                    }

                    // If the moved place is used generically by the callee and a reference to it
                    // would still satisfy any bounds on its type, suggest borrowing.
                    if let Some(&param) = arg_param
                        && let Some(generic_args) = call_id.and_then(|id| typeck.node_args_opt(id))
                        && let Some(ref_mutability) = self.suggest_borrow_generic_arg(
                            err,
                            def_id,
                            generic_args,
                            param,
                            moved_place,
                            pos + offset,
                            ty,
                            expr.span,
                        )
                    {
                        can_suggest_clone = ref_mutability.is_mut();
                    } else if let Some(local_def_id) = def_id.as_local()
                        && let node = self.infcx.tcx.hir_node_by_def_id(local_def_id)
                        && let Some(fn_decl) = node.fn_decl()
                        && let Some(ident) = node.ident()
                        && let Some(arg) = fn_decl.inputs.get(pos + offset)
                    {
                        // If we can't suggest borrowing in the call, but the function definition
                        // is local, instead offer changing the function to borrow that argument.
                        let mut span: MultiSpan = arg.span.into();
                        span.push_span_label(
                            arg.span,
                            "this parameter takes ownership of the value".to_string(),
                        );
                        let descr = match node.fn_kind() {
                            Some(hir::intravisit::FnKind::ItemFn(..)) | None => "function",
                            Some(hir::intravisit::FnKind::Method(..)) => "method",
                            Some(hir::intravisit::FnKind::Closure) => "closure",
                        };
                        span.push_span_label(ident.span, format!("in this {descr}"));
                        err.span_note(
                            span,
                            format!(
                                "consider changing this parameter type in {descr} `{ident}` to \
                                 borrow instead if owning the value isn't necessary",
                            ),
                        );
                    }
                }
                if let hir::Node::Expr(parent_expr) = parent
                    && let hir::ExprKind::Call(call_expr, _) = parent_expr.kind
                    && let hir::ExprKind::Path(hir::QPath::LangItem(LangItem::IntoIterIntoIter, _)) =
                        call_expr.kind
                {
                    // Do not suggest `.clone()` in a `for` loop, we already suggest borrowing.
                } else if let UseSpans::FnSelfUse { kind: CallKind::Normal { .. }, .. } = move_spans
                {
                    // We already suggest cloning for these cases in `explain_captures`.
                } else if moved_or_invoked_closure {
                    // Do not suggest `closure.clone()()`.
                } else if let UseSpans::ClosureUse {
                    closure_kind:
                        ClosureKind::Coroutine(CoroutineKind::Desugared(_, CoroutineSource::Block)),
                    ..
                } = move_spans
                    && can_suggest_clone
                {
                    self.suggest_cloning(err, ty, expr, Some(move_spans));
                } else if self.suggest_hoisting_call_outside_loop(err, expr) && can_suggest_clone {
                    // The place where the type moves would be misleading to suggest clone.
                    // #121466
                    self.suggest_cloning(err, ty, expr, Some(move_spans));
                }
            }

            self.suggest_ref_for_dbg_args(expr, place, move_span, err);

            // it's useless to suggest inserting `ref` when the span don't comes from local code
            if let Some(pat) = finder.pat
                && !move_span.is_dummy()
                && !self.infcx.tcx.sess.source_map().is_imported(move_span)
            {
                let mut sugg = vec![(pat.span.shrink_to_lo(), "ref ".to_string())];
                if let Some(pat) = finder.parent_pat {
                    sugg.insert(0, (pat.span.shrink_to_lo(), "ref ".to_string()));
                }
                err.multipart_suggestion_verbose(
                    "borrow this binding in the pattern to avoid moving the value",
                    sugg,
                    Applicability::MachineApplicable,
                );
            }
        }
    }

    // for dbg!(x) which may take ownership, suggest dbg!(&x) instead
    // but here we actually do not check whether the macro name is `dbg!`
    // so that we may extend the scope a bit larger to cover more cases
    fn suggest_ref_for_dbg_args(
        &self,
        body: &hir::Expr<'_>,
        place: &Place<'tcx>,
        move_span: Span,
        err: &mut Diag<'infcx>,
    ) {
        let var_info = self.body.var_debug_info.iter().find(|info| match info.value {
            VarDebugInfoContents::Place(ref p) => p == place,
            _ => false,
        });
        let arg_name = if let Some(var_info) = var_info {
            var_info.name
        } else {
            return;
        };
        struct MatchArgFinder {
            expr_span: Span,
            match_arg_span: Option<Span>,
            arg_name: Symbol,
        }
        impl Visitor<'_> for MatchArgFinder {
            fn visit_expr(&mut self, e: &hir::Expr<'_>) {
                // dbg! is expanded into a match pattern, we need to find the right argument span
                if let hir::ExprKind::Match(expr, ..) = &e.kind
                    && let hir::ExprKind::Path(hir::QPath::Resolved(
                        _,
                        path @ Path { segments: [seg], .. },
                    )) = &expr.kind
                    && seg.ident.name == self.arg_name
                    && self.expr_span.source_callsite().contains(expr.span)
                {
                    self.match_arg_span = Some(path.span);
                }
                hir::intravisit::walk_expr(self, e);
            }
        }

        let mut finder = MatchArgFinder { expr_span: move_span, match_arg_span: None, arg_name };
        finder.visit_expr(body);
        if let Some(macro_arg_span) = finder.match_arg_span {
            err.span_suggestion_verbose(
                macro_arg_span.shrink_to_lo(),
                "consider borrowing instead of transferring ownership",
                "&",
                Applicability::MachineApplicable,
            );
        }
    }

    pub(crate) fn suggest_reborrow(
        &self,
        err: &mut Diag<'infcx>,
        span: Span,
        moved_place: PlaceRef<'tcx>,
    ) {
        err.span_suggestion_verbose(
            span.shrink_to_lo(),
            format!(
                "consider creating a fresh reborrow of {} here",
                self.describe_place(moved_place)
                    .map(|n| format!("`{n}`"))
                    .unwrap_or_else(|| "the mutable reference".to_string()),
            ),
            "&mut *",
            Applicability::MachineApplicable,
        );
    }

    /// If a place is used after being moved as an argument to a function, the function is generic
    /// in that argument, and a reference to the argument's type would still satisfy the function's
    /// bounds, suggest borrowing. This covers, e.g., borrowing an `impl Fn()` argument being passed
    /// in an `impl FnOnce()` position.
    /// Returns `Some(mutability)` when suggesting to borrow with mutability `mutability`, or `None`
    /// if no suggestion is made.
    fn suggest_borrow_generic_arg(
        &self,
        err: &mut Diag<'_>,
        callee_did: DefId,
        generic_args: ty::GenericArgsRef<'tcx>,
        param: ty::ParamTy,
        moved_place: PlaceRef<'tcx>,
        moved_arg_pos: usize,
        moved_arg_ty: Ty<'tcx>,
        place_span: Span,
    ) -> Option<ty::Mutability> {
        let tcx = self.infcx.tcx;
        let sig = tcx.fn_sig(callee_did).instantiate_identity().skip_binder();
        let clauses = tcx.predicates_of(callee_did);

        // First, is there at least one method on one of `param`'s trait bounds?
        // This keeps us from suggesting borrowing the argument to `mem::drop`, e.g.
        if !clauses.instantiate_identity(tcx).predicates.iter().any(|clause| {
            clause.as_trait_clause().is_some_and(|tc| {
                tc.self_ty().skip_binder().is_param(param.index)
                    && tc.polarity() == ty::PredicatePolarity::Positive
                    && supertrait_def_ids(tcx, tc.def_id())
                        .flat_map(|trait_did| tcx.associated_items(trait_did).in_definition_order())
                        .any(|item| item.is_method())
            })
        }) {
            return None;
        }

        // Try borrowing a shared reference first, then mutably.
        if let Some(mutbl) = [ty::Mutability::Not, ty::Mutability::Mut].into_iter().find(|&mutbl| {
            let re = self.infcx.tcx.lifetimes.re_erased;
            let ref_ty = Ty::new_ref(self.infcx.tcx, re, moved_arg_ty, mutbl);

            // Ensure that substituting `ref_ty` in the callee's signature doesn't break
            // other inputs or the return type.
            let new_args = tcx.mk_args_from_iter(generic_args.iter().enumerate().map(
                |(i, arg)| {
                    if i == param.index as usize { ref_ty.into() } else { arg }
                },
            ));
            let can_subst = |ty: Ty<'tcx>| {
                // Normalize before comparing to see through type aliases and projections.
                let old_ty = ty::EarlyBinder::bind(ty).instantiate(tcx, generic_args);
                let new_ty = ty::EarlyBinder::bind(ty).instantiate(tcx, new_args);
                if let Ok(old_ty) = tcx.try_normalize_erasing_regions(
                    self.infcx.typing_env(self.infcx.param_env),
                    old_ty,
                ) && let Ok(new_ty) = tcx.try_normalize_erasing_regions(
                    self.infcx.typing_env(self.infcx.param_env),
                    new_ty,
                ) {
                    old_ty == new_ty
                } else {
                    false
                }
            };
            if !can_subst(sig.output())
                || sig
                    .inputs()
                    .iter()
                    .enumerate()
                    .any(|(i, &input_ty)| i != moved_arg_pos && !can_subst(input_ty))
            {
                return false;
            }

            // Test the callee's predicates, substituting in `ref_ty` for the moved argument type.
            clauses.instantiate(tcx, new_args).predicates.iter().all(|&(mut clause)| {
                // Normalize before testing to see through type aliases and projections.
                if let Ok(normalized) = tcx.try_normalize_erasing_regions(
                    self.infcx.typing_env(self.infcx.param_env),
                    clause,
                ) {
                    clause = normalized;
                }
                self.infcx.predicate_must_hold_modulo_regions(&Obligation::new(
                    tcx,
                    ObligationCause::dummy(),
                    self.infcx.param_env,
                    clause,
                ))
            })
        }) {
            let place_desc = if let Some(desc) = self.describe_place(moved_place) {
                format!("`{desc}`")
            } else {
                "here".to_owned()
            };
            err.span_suggestion_verbose(
                place_span.shrink_to_lo(),
                format!("consider {}borrowing {place_desc}", mutbl.mutably_str()),
                mutbl.ref_prefix_str(),
                Applicability::MaybeIncorrect,
            );
            Some(mutbl)
        } else {
            None
        }
    }

    fn report_use_of_uninitialized(
        &self,
        mpi: MovePathIndex,
        used_place: PlaceRef<'tcx>,
        moved_place: PlaceRef<'tcx>,
        desired_action: InitializationRequiringAction,
        span: Span,
        use_spans: UseSpans<'tcx>,
    ) -> Diag<'infcx> {
        // We need all statements in the body where the binding was assigned to later find all
        // the branching code paths where the binding *wasn't* assigned to.
        let inits = &self.move_data.init_path_map[mpi];
        let move_path = &self.move_data.move_paths[mpi];
        let decl_span = self.body.local_decls[move_path.place.local].source_info.span;
        let mut spans_set = FxIndexSet::default();
        for init_idx in inits {
            let init = &self.move_data.inits[*init_idx];
            let span = init.span(self.body);
            if !span.is_dummy() {
                spans_set.insert(span);
            }
        }
        let spans: Vec<_> = spans_set.into_iter().collect();

        let (name, desc) = match self.describe_place_with_options(
            moved_place,
            DescribePlaceOpt { including_downcast: true, including_tuple_field: true },
        ) {
            Some(name) => (format!("`{name}`"), format!("`{name}` ")),
            None => ("the variable".to_string(), String::new()),
        };
        let path = match self.describe_place_with_options(
            used_place,
            DescribePlaceOpt { including_downcast: true, including_tuple_field: true },
        ) {
            Some(name) => format!("`{name}`"),
            None => "value".to_string(),
        };

        // We use the statements were the binding was initialized, and inspect the HIR to look
        // for the branching codepaths that aren't covered, to point at them.
        let tcx = self.infcx.tcx;
        let body = tcx.hir_body_owned_by(self.mir_def_id());
        let mut visitor = ConditionVisitor { tcx, spans, name, errors: vec![] };
        visitor.visit_body(&body);
        let spans = visitor.spans;

        let mut show_assign_sugg = false;
        let isnt_initialized = if let InitializationRequiringAction::PartialAssignment
        | InitializationRequiringAction::Assignment = desired_action
        {
            // The same error is emitted for bindings that are *sometimes* initialized and the ones
            // that are *partially* initialized by assigning to a field of an uninitialized
            // binding. We differentiate between them for more accurate wording here.
            "isn't fully initialized"
        } else if !spans.iter().any(|i| {
            // We filter these to avoid misleading wording in cases like the following,
            // where `x` has an `init`, but it is in the same place we're looking at:
            // ```
            // let x;
            // x += 1;
            // ```
            !i.contains(span)
            // We filter these to avoid incorrect main message on `match-cfg-fake-edges.rs`
            && !visitor
                .errors
                .iter()
                .map(|(sp, _)| *sp)
                .any(|sp| span < sp && !sp.contains(span))
        }) {
            show_assign_sugg = true;
            "isn't initialized"
        } else {
            "is possibly-uninitialized"
        };

        let used = desired_action.as_general_verb_in_past_tense();
        let mut err = struct_span_code_err!(
            self.dcx(),
            span,
            E0381,
            "{used} binding {desc}{isnt_initialized}"
        );
        use_spans.var_path_only_subdiag(&mut err, desired_action);

        if let InitializationRequiringAction::PartialAssignment
        | InitializationRequiringAction::Assignment = desired_action
        {
            err.help(
                "partial initialization isn't supported, fully initialize the binding with a \
                 default value and mutate it, or use `std::mem::MaybeUninit`",
            );
        }
        err.span_label(span, format!("{path} {used} here but it {isnt_initialized}"));

        let mut shown = false;
        for (sp, label) in visitor.errors {
            if sp < span && !sp.overlaps(span) {
                // When we have a case like `match-cfg-fake-edges.rs`, we don't want to mention
                // match arms coming after the primary span because they aren't relevant:
                // ```
                // let x;
                // match y {
                //     _ if { x = 2; true } => {}
                //     _ if {
                //         x; //~ ERROR
                //         false
                //     } => {}
                //     _ => {} // We don't want to point to this.
                // };
                // ```
                err.span_label(sp, label);
                shown = true;
            }
        }
        if !shown {
            for sp in &spans {
                if *sp < span && !sp.overlaps(span) {
                    err.span_label(*sp, "binding initialized here in some conditions");
                }
            }
        }

        err.span_label(decl_span, "binding declared here but left uninitialized");
        if show_assign_sugg {
            struct LetVisitor {
                decl_span: Span,
                sugg_span: Option<Span>,
            }

            impl<'v> Visitor<'v> for LetVisitor {
                fn visit_stmt(&mut self, ex: &'v hir::Stmt<'v>) {
                    if self.sugg_span.is_some() {
                        return;
                    }

                    // FIXME: We make sure that this is a normal top-level binding,
                    // but we could suggest `todo!()` for all uninitialized bindings in the pattern
                    if let hir::StmtKind::Let(hir::LetStmt { span, ty, init: None, pat, .. }) =
                        &ex.kind
                        && let hir::PatKind::Binding(..) = pat.kind
                        && span.contains(self.decl_span)
                    {
                        self.sugg_span = ty.map_or(Some(self.decl_span), |ty| Some(ty.span));
                    }
                    hir::intravisit::walk_stmt(self, ex);
                }
            }

            let mut visitor = LetVisitor { decl_span, sugg_span: None };
            visitor.visit_body(&body);
            if let Some(span) = visitor.sugg_span {
                self.suggest_assign_value(&mut err, moved_place, span);
            }
        }
        err
    }

    fn suggest_assign_value(
        &self,
        err: &mut Diag<'_>,
        moved_place: PlaceRef<'tcx>,
        sugg_span: Span,
    ) {
        let ty = moved_place.ty(self.body, self.infcx.tcx).ty;
        debug!("ty: {:?}, kind: {:?}", ty, ty.kind());

        let Some(assign_value) = self.infcx.err_ctxt().ty_kind_suggestion(self.infcx.param_env, ty)
        else {
            return;
        };

        err.span_suggestion_verbose(
            sugg_span.shrink_to_hi(),
            "consider assigning a value",
            format!(" = {assign_value}"),
            Applicability::MaybeIncorrect,
        );
    }

    /// In a move error that occurs on a call within a loop, we try to identify cases where cloning
    /// the value would lead to a logic error. We infer these cases by seeing if the moved value is
    /// part of the logic to break the loop, either through an explicit `break` or if the expression
    /// is part of a `while let`.
    fn suggest_hoisting_call_outside_loop(&self, err: &mut Diag<'_>, expr: &hir::Expr<'_>) -> bool {
        let tcx = self.infcx.tcx;
        let mut can_suggest_clone = true;

        // If the moved value is a locally declared binding, we'll look upwards on the expression
        // tree until the scope where it is defined, and no further, as suggesting to move the
        // expression beyond that point would be illogical.
        let local_hir_id = if let hir::ExprKind::Path(hir::QPath::Resolved(
            _,
            hir::Path { res: hir::def::Res::Local(local_hir_id), .. },
        )) = expr.kind
        {
            Some(local_hir_id)
        } else {
            // This case would be if the moved value comes from an argument binding, we'll just
            // look within the entire item, that's fine.
            None
        };

        /// This will allow us to look for a specific `HirId`, in our case `local_hir_id` where the
        /// binding was declared, within any other expression. We'll use it to search for the
        /// binding declaration within every scope we inspect.
        struct Finder {
            hir_id: hir::HirId,
        }
        impl<'hir> Visitor<'hir> for Finder {
            type Result = ControlFlow<()>;
            fn visit_pat(&mut self, pat: &'hir hir::Pat<'hir>) -> Self::Result {
                if pat.hir_id == self.hir_id {
                    return ControlFlow::Break(());
                }
                hir::intravisit::walk_pat(self, pat)
            }
            fn visit_expr(&mut self, ex: &'hir hir::Expr<'hir>) -> Self::Result {
                if ex.hir_id == self.hir_id {
                    return ControlFlow::Break(());
                }
                hir::intravisit::walk_expr(self, ex)
            }
        }
        // The immediate HIR parent of the moved expression. We'll look for it to be a call.
        let mut parent = None;
        // The top-most loop where the moved expression could be moved to a new binding.
        let mut outer_most_loop: Option<&hir::Expr<'_>> = None;
        for (_, node) in tcx.hir_parent_iter(expr.hir_id) {
            let e = match node {
                hir::Node::Expr(e) => e,
                hir::Node::LetStmt(hir::LetStmt { els: Some(els), .. }) => {
                    let mut finder = BreakFinder { found_breaks: vec![], found_continues: vec![] };
                    finder.visit_block(els);
                    if !finder.found_breaks.is_empty() {
                        // Don't suggest clone as it could be will likely end in an infinite
                        // loop.
                        // let Some(_) = foo(non_copy.clone()) else { break; }
                        // ---                       ^^^^^^^^         -----
                        can_suggest_clone = false;
                    }
                    continue;
                }
                _ => continue,
            };
            if let Some(&hir_id) = local_hir_id {
                if (Finder { hir_id }).visit_expr(e).is_break() {
                    // The current scope includes the declaration of the binding we're accessing, we
                    // can't look up any further for loops.
                    break;
                }
            }
            if parent.is_none() {
                parent = Some(e);
            }
            match e.kind {
                hir::ExprKind::Let(_) => {
                    match tcx.parent_hir_node(e.hir_id) {
                        hir::Node::Expr(hir::Expr {
                            kind: hir::ExprKind::If(cond, ..), ..
                        }) => {
                            if (Finder { hir_id: expr.hir_id }).visit_expr(cond).is_break() {
                                // The expression where the move error happened is in a `while let`
                                // condition Don't suggest clone as it will likely end in an
                                // infinite loop.
                                // while let Some(_) = foo(non_copy.clone()) { }
                                // ---------                       ^^^^^^^^
                                can_suggest_clone = false;
                            }
                        }
                        _ => {}
                    }
                }
                hir::ExprKind::Loop(..) => {
                    outer_most_loop = Some(e);
                }
                _ => {}
            }
        }
        let loop_count: usize = tcx
            .hir_parent_iter(expr.hir_id)
            .map(|(_, node)| match node {
                hir::Node::Expr(hir::Expr { kind: hir::ExprKind::Loop(..), .. }) => 1,
                _ => 0,
            })
            .sum();

        let sm = tcx.sess.source_map();
        if let Some(in_loop) = outer_most_loop {
            let mut finder = BreakFinder { found_breaks: vec![], found_continues: vec![] };
            finder.visit_expr(in_loop);
            // All of the spans for `break` and `continue` expressions.
            let spans = finder
                .found_breaks
                .iter()
                .chain(finder.found_continues.iter())
                .map(|(_, span)| *span)
                .filter(|span| {
                    !matches!(
                        span.desugaring_kind(),
                        Some(DesugaringKind::ForLoop | DesugaringKind::WhileLoop)
                    )
                })
                .collect::<Vec<Span>>();
            // All of the spans for the loops above the expression with the move error.
            let loop_spans: Vec<_> = tcx
                .hir_parent_iter(expr.hir_id)
                .filter_map(|(_, node)| match node {
                    hir::Node::Expr(hir::Expr { span, kind: hir::ExprKind::Loop(..), .. }) => {
                        Some(*span)
                    }
                    _ => None,
                })
                .collect();
            // It is possible that a user written `break` or `continue` is in the wrong place. We
            // point them out at the user for them to make a determination. (#92531)
            if !spans.is_empty() && loop_count > 1 {
                // Getting fancy: if the spans of the loops *do not* overlap, we only use the line
                // number when referring to them. If there *are* overlaps (multiple loops on the
                // same line) then we use the more verbose span output (`file.rs:col:ll`).
                let mut lines: Vec<_> =
                    loop_spans.iter().map(|sp| sm.lookup_char_pos(sp.lo()).line).collect();
                lines.sort();
                lines.dedup();
                let fmt_span = |span: Span| {
                    if lines.len() == loop_spans.len() {
                        format!("line {}", sm.lookup_char_pos(span.lo()).line)
                    } else {
                        sm.span_to_diagnostic_string(span)
                    }
                };
                let mut spans: MultiSpan = spans.into();
                // Point at all the `continue`s and explicit `break`s in the relevant loops.
                for (desc, elements) in [
                    ("`break` exits", &finder.found_breaks),
                    ("`continue` advances", &finder.found_continues),
                ] {
                    for (destination, sp) in elements {
                        if let Ok(hir_id) = destination.target_id
                            && let hir::Node::Expr(expr) = tcx.hir_node(hir_id)
                            && !matches!(
                                sp.desugaring_kind(),
                                Some(DesugaringKind::ForLoop | DesugaringKind::WhileLoop)
                            )
                        {
                            spans.push_span_label(
                                *sp,
                                format!("this {desc} the loop at {}", fmt_span(expr.span)),
                            );
                        }
                    }
                }
                // Point at all the loops that are between this move and the parent item.
                for span in loop_spans {
                    spans.push_span_label(sm.guess_head_span(span), "");
                }

                // note: verify that your loop breaking logic is correct
                //   --> $DIR/nested-loop-moved-value-wrong-continue.rs:41:17
                //    |
                // 28 |     for foo in foos {
                //    |     ---------------
                // ...
                // 33 |         for bar in &bars {
                //    |         ----------------
                // ...
                // 41 |                 continue;
                //    |                 ^^^^^^^^ this `continue` advances the loop at line 33
                err.span_note(spans, "verify that your loop breaking logic is correct");
            }
            if let Some(parent) = parent
                && let hir::ExprKind::MethodCall(..) | hir::ExprKind::Call(..) = parent.kind
            {
                // FIXME: We could check that the call's *parent* takes `&mut val` to make the
                // suggestion more targeted to the `mk_iter(val).next()` case. Maybe do that only to
                // check for whether to suggest `let value` or `let mut value`.

                let span = in_loop.span;
                if !finder.found_breaks.is_empty()
                    && let Ok(value) = sm.span_to_snippet(parent.span)
                {
                    // We know with high certainty that this move would affect the early return of a
                    // loop, so we suggest moving the expression with the move out of the loop.
                    let indent = if let Some(indent) = sm.indentation_before(span) {
                        format!("\n{indent}")
                    } else {
                        " ".to_string()
                    };
                    err.multipart_suggestion(
                        "consider moving the expression out of the loop so it is only moved once",
                        vec![
                            (span.shrink_to_lo(), format!("let mut value = {value};{indent}")),
                            (parent.span, "value".to_string()),
                        ],
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        }
        can_suggest_clone
    }

    /// We have `S { foo: val, ..base }`, and we suggest instead writing
    /// `S { foo: val, bar: base.bar.clone(), .. }` when valid.
    fn suggest_cloning_on_functional_record_update(
        &self,
        err: &mut Diag<'_>,
        ty: Ty<'tcx>,
        expr: &hir::Expr<'_>,
    ) {
        let typeck_results = self.infcx.tcx.typeck(self.mir_def_id());
        let hir::ExprKind::Struct(struct_qpath, fields, hir::StructTailExpr::Base(base)) =
            expr.kind
        else {
            return;
        };
        let hir::QPath::Resolved(_, path) = struct_qpath else { return };
        let hir::def::Res::Def(_, def_id) = path.res else { return };
        let Some(expr_ty) = typeck_results.node_type_opt(expr.hir_id) else { return };
        let ty::Adt(def, args) = expr_ty.kind() else { return };
        let hir::ExprKind::Path(hir::QPath::Resolved(None, path)) = base.kind else { return };
        let (hir::def::Res::Local(_)
        | hir::def::Res::Def(
            DefKind::Const | DefKind::ConstParam | DefKind::Static { .. } | DefKind::AssocConst,
            _,
        )) = path.res
        else {
            return;
        };
        let Ok(base_str) = self.infcx.tcx.sess.source_map().span_to_snippet(base.span) else {
            return;
        };

        // 1. look for the fields of type `ty`.
        // 2. check if they are clone and add them to suggestion
        // 3. check if there are any values left to `..` and remove it if not
        // 4. emit suggestion to clone the field directly as `bar: base.bar.clone()`

        let mut final_field_count = fields.len();
        let Some(variant) = def.variants().iter().find(|variant| variant.def_id == def_id) else {
            // When we have an enum, look for the variant that corresponds to the variant the user
            // wrote.
            return;
        };
        let mut sugg = vec![];
        for field in &variant.fields {
            // In practice unless there are more than one field with the same type, we'll be
            // suggesting a single field at a type, because we don't aggregate multiple borrow
            // checker errors involving the functional record update syntax into a single one.
            let field_ty = field.ty(self.infcx.tcx, args);
            let ident = field.ident(self.infcx.tcx);
            if field_ty == ty && fields.iter().all(|field| field.ident.name != ident.name) {
                // Suggest adding field and cloning it.
                sugg.push(format!("{ident}: {base_str}.{ident}.clone()"));
                final_field_count += 1;
            }
        }
        let (span, sugg) = match fields {
            [.., last] => (
                if final_field_count == variant.fields.len() {
                    // We'll remove the `..base` as there aren't any fields left.
                    last.span.shrink_to_hi().with_hi(base.span.hi())
                } else {
                    last.span.shrink_to_hi()
                },
                format!(", {}", sugg.join(", ")),
            ),
            // Account for no fields in suggestion span.
            [] => (
                expr.span.with_lo(struct_qpath.span().hi()),
                if final_field_count == variant.fields.len() {
                    // We'll remove the `..base` as there aren't any fields left.
                    format!(" {{ {} }}", sugg.join(", "))
                } else {
                    format!(" {{ {}, ..{base_str} }}", sugg.join(", "))
                },
            ),
        };
        let prefix = if !self.implements_clone(ty) {
            let msg = format!("`{ty}` doesn't implement `Copy` or `Clone`");
            if let ty::Adt(def, _) = ty.kind() {
                err.span_note(self.infcx.tcx.def_span(def.did()), msg);
            } else {
                err.note(msg);
            }
            format!("if `{ty}` implemented `Clone`, you could ")
        } else {
            String::new()
        };
        let msg = format!(
            "{prefix}clone the value from the field instead of using the functional record update \
             syntax",
        );
        err.span_suggestion_verbose(span, msg, sugg, Applicability::MachineApplicable);
    }

    pub(crate) fn suggest_cloning(
        &self,
        err: &mut Diag<'_>,
        ty: Ty<'tcx>,
        expr: &'tcx hir::Expr<'tcx>,
        use_spans: Option<UseSpans<'tcx>>,
    ) {
        if let hir::ExprKind::Struct(_, _, hir::StructTailExpr::Base(_)) = expr.kind {
            // We have `S { foo: val, ..base }`. In `check_aggregate_rvalue` we have a single
            // `Location` that covers both the `S { ... }` literal, all of its fields and the
            // `base`. If the move happens because of `S { foo: val, bar: base.bar }` the `expr`
            //  will already be correct. Instead, we see if we can suggest writing.
            self.suggest_cloning_on_functional_record_update(err, ty, expr);
            return;
        }

        if self.implements_clone(ty) {
            self.suggest_cloning_inner(err, ty, expr);
        } else if let ty::Adt(def, args) = ty.kind()
            && def.did().as_local().is_some()
            && def.variants().iter().all(|variant| {
                variant
                    .fields
                    .iter()
                    .all(|field| self.implements_clone(field.ty(self.infcx.tcx, args)))
            })
        {
            let ty_span = self.infcx.tcx.def_span(def.did());
            let mut span: MultiSpan = ty_span.into();
            span.push_span_label(ty_span, "consider implementing `Clone` for this type");
            span.push_span_label(expr.span, "you could clone this value");
            err.span_note(
                span,
                format!("if `{ty}` implemented `Clone`, you could clone the value"),
            );
        } else if let ty::Param(param) = ty.kind()
            && let Some(_clone_trait_def) = self.infcx.tcx.lang_items().clone_trait()
            && let generics = self.infcx.tcx.generics_of(self.mir_def_id())
            && let generic_param = generics.type_param(*param, self.infcx.tcx)
            && let param_span = self.infcx.tcx.def_span(generic_param.def_id)
            && if let Some(UseSpans::FnSelfUse { kind, .. }) = use_spans
                && let CallKind::FnCall { fn_trait_id, self_ty } = kind
                && let ty::Param(_) = self_ty.kind()
                && ty == self_ty
                && self.infcx.tcx.fn_trait_kind_from_def_id(fn_trait_id).is_some()
            {
                // Do not suggest `F: FnOnce() + Clone`.
                false
            } else {
                true
            }
        {
            let mut span: MultiSpan = param_span.into();
            span.push_span_label(
                param_span,
                "consider constraining this type parameter with `Clone`",
            );
            span.push_span_label(expr.span, "you could clone this value");
            err.span_help(
                span,
                format!("if `{ty}` implemented `Clone`, you could clone the value"),
            );
        }
    }

    pub(crate) fn implements_clone(&self, ty: Ty<'tcx>) -> bool {
        let Some(clone_trait_def) = self.infcx.tcx.lang_items().clone_trait() else { return false };
        self.infcx
            .type_implements_trait(clone_trait_def, [ty], self.infcx.param_env)
            .must_apply_modulo_regions()
    }

    /// Given an expression, check if it is a method call `foo.clone()`, where `foo` and
    /// `foo.clone()` both have the same type, returning the span for `.clone()` if so.
    pub(crate) fn clone_on_reference(&self, expr: &hir::Expr<'_>) -> Option<Span> {
        let typeck_results = self.infcx.tcx.typeck(self.mir_def_id());
        if let hir::ExprKind::MethodCall(segment, rcvr, args, span) = expr.kind
            && let Some(expr_ty) = typeck_results.node_type_opt(expr.hir_id)
            && let Some(rcvr_ty) = typeck_results.node_type_opt(rcvr.hir_id)
            && rcvr_ty == expr_ty
            && segment.ident.name == sym::clone
            && args.is_empty()
        {
            Some(span)
        } else {
            None
        }
    }

    fn in_move_closure(&self, expr: &hir::Expr<'_>) -> bool {
        for (_, node) in self.infcx.tcx.hir_parent_iter(expr.hir_id) {
            if let hir::Node::Expr(hir::Expr { kind: hir::ExprKind::Closure(closure), .. }) = node
                && let hir::CaptureBy::Value { .. } = closure.capture_clause
            {
                // `move || x.clone()` will not work. FIXME: suggest `let y = x.clone(); move || y`
                return true;
            }
        }
        false
    }

    fn suggest_cloning_inner(
        &self,
        err: &mut Diag<'_>,
        ty: Ty<'tcx>,
        expr: &hir::Expr<'_>,
    ) -> bool {
        let tcx = self.infcx.tcx;
        if let Some(_) = self.clone_on_reference(expr) {
            // Avoid redundant clone suggestion already suggested in `explain_captures`.
            // See `tests/ui/moves/needs-clone-through-deref.rs`
            return false;
        }
        // We don't want to suggest `.clone()` in a move closure, since the value has already been
        // captured.
        if self.in_move_closure(expr) {
            return false;
        }
        // We also don't want to suggest cloning a closure itself, since the value has already been
        // captured.
        if let hir::ExprKind::Closure(_) = expr.kind {
            return false;
        }
        // Try to find predicates on *generic params* that would allow copying `ty`
        let mut suggestion =
            if let Some(symbol) = tcx.hir_maybe_get_struct_pattern_shorthand_field(expr) {
                format!(": {symbol}.clone()")
            } else {
                ".clone()".to_owned()
            };
        let mut sugg = Vec::with_capacity(2);
        let mut inner_expr = expr;
        let mut is_raw_ptr = false;
        let typeck_result = self.infcx.tcx.typeck(self.mir_def_id());
        // Remove uses of `&` and `*` when suggesting `.clone()`.
        while let hir::ExprKind::AddrOf(.., inner) | hir::ExprKind::Unary(hir::UnOp::Deref, inner) =
            &inner_expr.kind
        {
            if let hir::ExprKind::AddrOf(_, hir::Mutability::Mut, _) = inner_expr.kind {
                // We assume that `&mut` refs are desired for their side-effects, so cloning the
                // value wouldn't do what the user wanted.
                return false;
            }
            inner_expr = inner;
            if let Some(inner_type) = typeck_result.node_type_opt(inner.hir_id) {
                if matches!(inner_type.kind(), ty::RawPtr(..)) {
                    is_raw_ptr = true;
                    break;
                }
            }
        }
        // Cloning the raw pointer doesn't make sense in some cases and would cause a type mismatch
        // error. (see #126863)
        if inner_expr.span.lo() != expr.span.lo() && !is_raw_ptr {
            // Remove "(*" or "(&"
            sugg.push((expr.span.with_hi(inner_expr.span.lo()), String::new()));
        }
        // Check whether `expr` is surrounded by parentheses or not.
        let span = if inner_expr.span.hi() != expr.span.hi() {
            // Account for `(*x)` to suggest `x.clone()`.
            if is_raw_ptr {
                expr.span.shrink_to_hi()
            } else {
                // Remove the close parenthesis ")"
                expr.span.with_lo(inner_expr.span.hi())
            }
        } else {
            if is_raw_ptr {
                sugg.push((expr.span.shrink_to_lo(), "(".to_string()));
                suggestion = ").clone()".to_string();
            }
            expr.span.shrink_to_hi()
        };
        sugg.push((span, suggestion));
        let msg = if let ty::Adt(def, _) = ty.kind()
            && [tcx.get_diagnostic_item(sym::Arc), tcx.get_diagnostic_item(sym::Rc)]
                .contains(&Some(def.did()))
        {
            "clone the value to increment its reference count"
        } else {
            "consider cloning the value if the performance cost is acceptable"
        };
        err.multipart_suggestion_verbose(msg, sugg, Applicability::MachineApplicable);
        true
    }

    fn suggest_adding_bounds(&self, err: &mut Diag<'_>, ty: Ty<'tcx>, def_id: DefId, span: Span) {
        let tcx = self.infcx.tcx;
        let generics = tcx.generics_of(self.mir_def_id());

        let Some(hir_generics) = tcx
            .typeck_root_def_id(self.mir_def_id().to_def_id())
            .as_local()
            .and_then(|def_id| tcx.hir_get_generics(def_id))
        else {
            return;
        };
        // Try to find predicates on *generic params* that would allow copying `ty`
        let ocx = ObligationCtxt::new_with_diagnostics(self.infcx);
        let cause = ObligationCause::misc(span, self.mir_def_id());

        ocx.register_bound(cause, self.infcx.param_env, ty, def_id);
        let errors = ocx.select_all_or_error();

        // Only emit suggestion if all required predicates are on generic
        let predicates: Result<Vec<_>, _> = errors
            .into_iter()
            .map(|err| match err.obligation.predicate.kind().skip_binder() {
                PredicateKind::Clause(ty::ClauseKind::Trait(predicate)) => {
                    match *predicate.self_ty().kind() {
                        ty::Param(param_ty) => Ok((
                            generics.type_param(param_ty, tcx),
                            predicate.trait_ref.print_trait_sugared().to_string(),
                            Some(predicate.trait_ref.def_id),
                        )),
                        _ => Err(()),
                    }
                }
                _ => Err(()),
            })
            .collect();

        if let Ok(predicates) = predicates {
            suggest_constraining_type_params(
                tcx,
                hir_generics,
                err,
                predicates.iter().map(|(param, constraint, def_id)| {
                    (param.name.as_str(), &**constraint, *def_id)
                }),
                None,
            );
        }
    }

    pub(crate) fn report_move_out_while_borrowed(
        &mut self,
        location: Location,
        (place, span): (Place<'tcx>, Span),
        borrow: &BorrowData<'tcx>,
    ) {
        debug!(
            "report_move_out_while_borrowed: location={:?} place={:?} span={:?} borrow={:?}",
            location, place, span, borrow
        );
        let value_msg = self.describe_any_place(place.as_ref());
        let borrow_msg = self.describe_any_place(borrow.borrowed_place.as_ref());

        let borrow_spans = self.retrieve_borrow_spans(borrow);
        let borrow_span = borrow_spans.args_or_use();

        let move_spans = self.move_spans(place.as_ref(), location);
        let span = move_spans.args_or_use();

        let mut err = self.cannot_move_when_borrowed(
            span,
            borrow_span,
            &self.describe_any_place(place.as_ref()),
            &borrow_msg,
            &value_msg,
        );
        self.note_due_to_edition_2024_opaque_capture_rules(borrow, &mut err);

        borrow_spans.var_path_only_subdiag(&mut err, crate::InitializationRequiringAction::Borrow);

        move_spans.var_subdiag(&mut err, None, |kind, var_span| {
            use crate::session_diagnostics::CaptureVarCause::*;
            match kind {
                hir::ClosureKind::Coroutine(_) => MoveUseInCoroutine { var_span },
                hir::ClosureKind::Closure | hir::ClosureKind::CoroutineClosure(_) => {
                    MoveUseInClosure { var_span }
                }
            }
        });

        self.explain_why_borrow_contains_point(location, borrow, None)
            .add_explanation_to_diagnostic(&self, &mut err, "", Some(borrow_span), None);
        self.suggest_copy_for_type_in_cloned_ref(&mut err, place);
        let typeck_results = self.infcx.tcx.typeck(self.mir_def_id());
        if let Some(expr) = self.find_expr(borrow_span) {
            // This is a borrow span, so we want to suggest cloning the referent.
            if let hir::ExprKind::AddrOf(_, _, borrowed_expr) = expr.kind
                && let Some(ty) = typeck_results.expr_ty_opt(borrowed_expr)
            {
                self.suggest_cloning(&mut err, ty, borrowed_expr, Some(move_spans));
            } else if typeck_results.expr_adjustments(expr).first().is_some_and(|adj| {
                matches!(
                    adj.kind,
                    ty::adjustment::Adjust::Borrow(ty::adjustment::AutoBorrow::Ref(
                        ty::adjustment::AutoBorrowMutability::Not
                            | ty::adjustment::AutoBorrowMutability::Mut {
                                allow_two_phase_borrow: ty::adjustment::AllowTwoPhase::No
                            }
                    ))
                )
            }) && let Some(ty) = typeck_results.expr_ty_opt(expr)
            {
                self.suggest_cloning(&mut err, ty, expr, Some(move_spans));
            }
        }
        self.buffer_error(err);
    }

    pub(crate) fn report_use_while_mutably_borrowed(
        &self,
        location: Location,
        (place, _span): (Place<'tcx>, Span),
        borrow: &BorrowData<'tcx>,
    ) -> Diag<'infcx> {
        let borrow_spans = self.retrieve_borrow_spans(borrow);
        let borrow_span = borrow_spans.args_or_use();

        // Conflicting borrows are reported separately, so only check for move
        // captures.
        let use_spans = self.move_spans(place.as_ref(), location);
        let span = use_spans.var_or_use();

        // If the attempted use is in a closure then we do not care about the path span of the
        // place we are currently trying to use we call `var_span_label` on `borrow_spans` to
        // annotate if the existing borrow was in a closure.
        let mut err = self.cannot_use_when_mutably_borrowed(
            span,
            &self.describe_any_place(place.as_ref()),
            borrow_span,
            &self.describe_any_place(borrow.borrowed_place.as_ref()),
        );
        self.note_due_to_edition_2024_opaque_capture_rules(borrow, &mut err);

        borrow_spans.var_subdiag(&mut err, Some(borrow.kind), |kind, var_span| {
            use crate::session_diagnostics::CaptureVarCause::*;
            let place = &borrow.borrowed_place;
            let desc_place = self.describe_any_place(place.as_ref());
            match kind {
                hir::ClosureKind::Coroutine(_) => {
                    BorrowUsePlaceCoroutine { place: desc_place, var_span, is_single_var: true }
                }
                hir::ClosureKind::Closure | hir::ClosureKind::CoroutineClosure(_) => {
                    BorrowUsePlaceClosure { place: desc_place, var_span, is_single_var: true }
                }
            }
        });

        self.explain_why_borrow_contains_point(location, borrow, None)
            .add_explanation_to_diagnostic(&self, &mut err, "", None, None);
        err
    }

    pub(crate) fn report_conflicting_borrow(
        &self,
        location: Location,
        (place, span): (Place<'tcx>, Span),
        gen_borrow_kind: BorrowKind,
        issued_borrow: &BorrowData<'tcx>,
    ) -> Diag<'infcx> {
        let issued_spans = self.retrieve_borrow_spans(issued_borrow);
        let issued_span = issued_spans.args_or_use();

        let borrow_spans = self.borrow_spans(span, location);
        let span = borrow_spans.args_or_use();

        let container_name = if issued_spans.for_coroutine() || borrow_spans.for_coroutine() {
            "coroutine"
        } else {
            "closure"
        };

        let (desc_place, msg_place, msg_borrow, union_type_name) =
            self.describe_place_for_conflicting_borrow(place, issued_borrow.borrowed_place);

        let explanation = self.explain_why_borrow_contains_point(location, issued_borrow, None);
        let second_borrow_desc = if explanation.is_explained() { "second " } else { "" };

        // FIXME: supply non-"" `opt_via` when appropriate
        let first_borrow_desc;
        let mut err = match (gen_borrow_kind, issued_borrow.kind) {
            (
                BorrowKind::Shared | BorrowKind::Fake(FakeBorrowKind::Deep),
                BorrowKind::Mut { kind: MutBorrowKind::Default | MutBorrowKind::TwoPhaseBorrow },
            ) => {
                first_borrow_desc = "mutable ";
                let mut err = self.cannot_reborrow_already_borrowed(
                    span,
                    &desc_place,
                    &msg_place,
                    "immutable",
                    issued_span,
                    "it",
                    "mutable",
                    &msg_borrow,
                    None,
                );
                self.suggest_slice_method_if_applicable(
                    &mut err,
                    place,
                    issued_borrow.borrowed_place,
                    span,
                    issued_span,
                );
                err
            }
            (
                BorrowKind::Mut { kind: MutBorrowKind::Default | MutBorrowKind::TwoPhaseBorrow },
                BorrowKind::Shared | BorrowKind::Fake(FakeBorrowKind::Deep),
            ) => {
                first_borrow_desc = "immutable ";
                let mut err = self.cannot_reborrow_already_borrowed(
                    span,
                    &desc_place,
                    &msg_place,
                    "mutable",
                    issued_span,
                    "it",
                    "immutable",
                    &msg_borrow,
                    None,
                );
                self.suggest_slice_method_if_applicable(
                    &mut err,
                    place,
                    issued_borrow.borrowed_place,
                    span,
                    issued_span,
                );
                self.suggest_binding_for_closure_capture_self(&mut err, &issued_spans);
                self.suggest_using_closure_argument_instead_of_capture(
                    &mut err,
                    issued_borrow.borrowed_place,
                    &issued_spans,
                );
                err
            }

            (
                BorrowKind::Mut { kind: MutBorrowKind::Default | MutBorrowKind::TwoPhaseBorrow },
                BorrowKind::Mut { kind: MutBorrowKind::Default | MutBorrowKind::TwoPhaseBorrow },
            ) => {
                first_borrow_desc = "first ";
                let mut err = self.cannot_mutably_borrow_multiply(
                    span,
                    &desc_place,
                    &msg_place,
                    issued_span,
                    &msg_borrow,
                    None,
                );
                self.suggest_slice_method_if_applicable(
                    &mut err,
                    place,
                    issued_borrow.borrowed_place,
                    span,
                    issued_span,
                );
                self.suggest_using_closure_argument_instead_of_capture(
                    &mut err,
                    issued_borrow.borrowed_place,
                    &issued_spans,
                );
                self.explain_iterator_advancement_in_for_loop_if_applicable(
                    &mut err,
                    span,
                    &issued_spans,
                );
                err
            }

            (
                BorrowKind::Mut { kind: MutBorrowKind::ClosureCapture },
                BorrowKind::Mut { kind: MutBorrowKind::ClosureCapture },
            ) => {
                first_borrow_desc = "first ";
                self.cannot_uniquely_borrow_by_two_closures(span, &desc_place, issued_span, None)
            }

            (BorrowKind::Mut { .. }, BorrowKind::Fake(FakeBorrowKind::Shallow)) => {
                if let Some(immutable_section_description) =
                    self.classify_immutable_section(issued_borrow.assigned_place)
                {
                    let mut err = self.cannot_mutate_in_immutable_section(
                        span,
                        issued_span,
                        &desc_place,
                        immutable_section_description,
                        "mutably borrow",
                    );
                    borrow_spans.var_subdiag(
                        &mut err,
                        Some(BorrowKind::Mut { kind: MutBorrowKind::ClosureCapture }),
                        |kind, var_span| {
                            use crate::session_diagnostics::CaptureVarCause::*;
                            match kind {
                                hir::ClosureKind::Coroutine(_) => BorrowUsePlaceCoroutine {
                                    place: desc_place,
                                    var_span,
                                    is_single_var: true,
                                },
                                hir::ClosureKind::Closure
                                | hir::ClosureKind::CoroutineClosure(_) => BorrowUsePlaceClosure {
                                    place: desc_place,
                                    var_span,
                                    is_single_var: true,
                                },
                            }
                        },
                    );
                    return err;
                } else {
                    first_borrow_desc = "immutable ";
                    self.cannot_reborrow_already_borrowed(
                        span,
                        &desc_place,
                        &msg_place,
                        "mutable",
                        issued_span,
                        "it",
                        "immutable",
                        &msg_borrow,
                        None,
                    )
                }
            }

            (BorrowKind::Mut { kind: MutBorrowKind::ClosureCapture }, _) => {
                first_borrow_desc = "first ";
                self.cannot_uniquely_borrow_by_one_closure(
                    span,
                    container_name,
                    &desc_place,
                    "",
                    issued_span,
                    "it",
                    "",
                    None,
                )
            }

            (
                BorrowKind::Shared | BorrowKind::Fake(FakeBorrowKind::Deep),
                BorrowKind::Mut { kind: MutBorrowKind::ClosureCapture },
            ) => {
                first_borrow_desc = "first ";
                self.cannot_reborrow_already_uniquely_borrowed(
                    span,
                    container_name,
                    &desc_place,
                    "",
                    "immutable",
                    issued_span,
                    "",
                    None,
                    second_borrow_desc,
                )
            }

            (BorrowKind::Mut { .. }, BorrowKind::Mut { kind: MutBorrowKind::ClosureCapture }) => {
                first_borrow_desc = "first ";
                self.cannot_reborrow_already_uniquely_borrowed(
                    span,
                    container_name,
                    &desc_place,
                    "",
                    "mutable",
                    issued_span,
                    "",
                    None,
                    second_borrow_desc,
                )
            }

            (
                BorrowKind::Shared | BorrowKind::Fake(FakeBorrowKind::Deep),
                BorrowKind::Shared | BorrowKind::Fake(_),
            )
            | (
                BorrowKind::Fake(FakeBorrowKind::Shallow),
                BorrowKind::Mut { .. } | BorrowKind::Shared | BorrowKind::Fake(_),
            ) => {
                unreachable!()
            }
        };
        self.note_due_to_edition_2024_opaque_capture_rules(issued_borrow, &mut err);

        if issued_spans == borrow_spans {
            borrow_spans.var_subdiag(&mut err, Some(gen_borrow_kind), |kind, var_span| {
                use crate::session_diagnostics::CaptureVarCause::*;
                match kind {
                    hir::ClosureKind::Coroutine(_) => BorrowUsePlaceCoroutine {
                        place: desc_place,
                        var_span,
                        is_single_var: false,
                    },
                    hir::ClosureKind::Closure | hir::ClosureKind::CoroutineClosure(_) => {
                        BorrowUsePlaceClosure { place: desc_place, var_span, is_single_var: false }
                    }
                }
            });
        } else {
            issued_spans.var_subdiag(&mut err, Some(issued_borrow.kind), |kind, var_span| {
                use crate::session_diagnostics::CaptureVarCause::*;
                let borrow_place = &issued_borrow.borrowed_place;
                let borrow_place_desc = self.describe_any_place(borrow_place.as_ref());
                match kind {
                    hir::ClosureKind::Coroutine(_) => {
                        FirstBorrowUsePlaceCoroutine { place: borrow_place_desc, var_span }
                    }
                    hir::ClosureKind::Closure | hir::ClosureKind::CoroutineClosure(_) => {
                        FirstBorrowUsePlaceClosure { place: borrow_place_desc, var_span }
                    }
                }
            });

            borrow_spans.var_subdiag(&mut err, Some(gen_borrow_kind), |kind, var_span| {
                use crate::session_diagnostics::CaptureVarCause::*;
                match kind {
                    hir::ClosureKind::Coroutine(_) => {
                        SecondBorrowUsePlaceCoroutine { place: desc_place, var_span }
                    }
                    hir::ClosureKind::Closure | hir::ClosureKind::CoroutineClosure(_) => {
                        SecondBorrowUsePlaceClosure { place: desc_place, var_span }
                    }
                }
            });
        }

        if union_type_name != "" {
            err.note(format!(
                "{msg_place} is a field of the union `{union_type_name}`, so it overlaps the field {msg_borrow}",
            ));
        }

        explanation.add_explanation_to_diagnostic(
            &self,
            &mut err,
            first_borrow_desc,
            None,
            Some((issued_span, span)),
        );

        self.suggest_using_local_if_applicable(&mut err, location, issued_borrow, explanation);
        self.suggest_copy_for_type_in_cloned_ref(&mut err, place);

        err
    }

    fn suggest_copy_for_type_in_cloned_ref(&self, err: &mut Diag<'infcx>, place: Place<'tcx>) {
        let tcx = self.infcx.tcx;
        let Some(body_id) = tcx.hir_node(self.mir_hir_id()).body_id() else { return };

        struct FindUselessClone<'tcx> {
            tcx: TyCtxt<'tcx>,
            typeck_results: &'tcx ty::TypeckResults<'tcx>,
            clones: Vec<&'tcx hir::Expr<'tcx>>,
        }
        impl<'tcx> FindUselessClone<'tcx> {
            fn new(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> Self {
                Self { tcx, typeck_results: tcx.typeck(def_id), clones: vec![] }
            }
        }
        impl<'tcx> Visitor<'tcx> for FindUselessClone<'tcx> {
            fn visit_expr(&mut self, ex: &'tcx hir::Expr<'tcx>) {
                if let hir::ExprKind::MethodCall(..) = ex.kind
                    && let Some(method_def_id) =
                        self.typeck_results.type_dependent_def_id(ex.hir_id)
                    && self.tcx.is_lang_item(self.tcx.parent(method_def_id), LangItem::Clone)
                {
                    self.clones.push(ex);
                }
                hir::intravisit::walk_expr(self, ex);
            }
        }

        let mut expr_finder = FindUselessClone::new(tcx, self.mir_def_id());

        let body = tcx.hir_body(body_id).value;
        expr_finder.visit_expr(body);

        struct Holds<'tcx> {
            ty: Ty<'tcx>,
        }

        impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for Holds<'tcx> {
            type Result = std::ops::ControlFlow<()>;

            fn visit_ty(&mut self, t: Ty<'tcx>) -> Self::Result {
                if t == self.ty {
                    return ControlFlow::Break(());
                }
                t.super_visit_with(self)
            }
        }

        let mut types_to_constrain = FxIndexSet::default();

        let local_ty = self.body.local_decls[place.local].ty;
        let typeck_results = tcx.typeck(self.mir_def_id());
        let clone = tcx.require_lang_item(LangItem::Clone, body.span);
        for expr in expr_finder.clones {
            if let hir::ExprKind::MethodCall(_, rcvr, _, span) = expr.kind
                && let Some(rcvr_ty) = typeck_results.node_type_opt(rcvr.hir_id)
                && let Some(ty) = typeck_results.node_type_opt(expr.hir_id)
                && rcvr_ty == ty
                && let ty::Ref(_, inner, _) = rcvr_ty.kind()
                && let inner = inner.peel_refs()
                && (Holds { ty: inner }).visit_ty(local_ty).is_break()
                && let None =
                    self.infcx.type_implements_trait_shallow(clone, inner, self.infcx.param_env)
            {
                err.span_label(
                    span,
                    format!(
                        "this call doesn't do anything, the result is still `{rcvr_ty}` \
                             because `{inner}` doesn't implement `Clone`",
                    ),
                );
                types_to_constrain.insert(inner);
            }
        }
        for ty in types_to_constrain {
            self.suggest_adding_bounds_or_derive(err, ty, clone, body.span);
        }
    }

    pub(crate) fn suggest_adding_bounds_or_derive(
        &self,
        err: &mut Diag<'_>,
        ty: Ty<'tcx>,
        trait_def_id: DefId,
        span: Span,
    ) {
        self.suggest_adding_bounds(err, ty, trait_def_id, span);
        if let ty::Adt(..) = ty.kind() {
            // The type doesn't implement the trait.
            let trait_ref =
                ty::Binder::dummy(ty::TraitRef::new(self.infcx.tcx, trait_def_id, [ty]));
            let obligation = Obligation::new(
                self.infcx.tcx,
                ObligationCause::dummy(),
                self.infcx.param_env,
                trait_ref,
            );
            self.infcx.err_ctxt().suggest_derive(
                &obligation,
                err,
                trait_ref.upcast(self.infcx.tcx),
            );
        }
    }

    #[instrument(level = "debug", skip(self, err))]
    fn suggest_using_local_if_applicable(
        &self,
        err: &mut Diag<'_>,
        location: Location,
        issued_borrow: &BorrowData<'tcx>,
        explanation: BorrowExplanation<'tcx>,
    ) {
        let used_in_call = matches!(
            explanation,
            BorrowExplanation::UsedLater(
                _,
                LaterUseKind::Call | LaterUseKind::Other,
                _call_span,
                _
            )
        );
        if !used_in_call {
            debug!("not later used in call");
            return;
        }
        if matches!(
            self.body.local_decls[issued_borrow.borrowed_place.local].local_info(),
            LocalInfo::IfThenRescopeTemp { .. }
        ) {
            // A better suggestion will be issued by the `if_let_rescope` lint
            return;
        }

        let use_span = if let BorrowExplanation::UsedLater(_, LaterUseKind::Other, use_span, _) =
            explanation
        {
            Some(use_span)
        } else {
            None
        };

        let outer_call_loc =
            if let TwoPhaseActivation::ActivatedAt(loc) = issued_borrow.activation_location {
                loc
            } else {
                issued_borrow.reserve_location
            };
        let outer_call_stmt = self.body.stmt_at(outer_call_loc);

        let inner_param_location = location;
        let Some(inner_param_stmt) = self.body.stmt_at(inner_param_location).left() else {
            debug!("`inner_param_location` {:?} is not for a statement", inner_param_location);
            return;
        };
        let Some(&inner_param) = inner_param_stmt.kind.as_assign().map(|(p, _)| p) else {
            debug!(
                "`inner_param_location` {:?} is not for an assignment: {:?}",
                inner_param_location, inner_param_stmt
            );
            return;
        };
        let inner_param_uses = find_all_local_uses::find(self.body, inner_param.local);
        let Some((inner_call_loc, inner_call_term)) =
            inner_param_uses.into_iter().find_map(|loc| {
                let Either::Right(term) = self.body.stmt_at(loc) else {
                    debug!("{:?} is a statement, so it can't be a call", loc);
                    return None;
                };
                let TerminatorKind::Call { args, .. } = &term.kind else {
                    debug!("not a call: {:?}", term);
                    return None;
                };
                debug!("checking call args for uses of inner_param: {:?}", args);
                args.iter()
                    .map(|a| &a.node)
                    .any(|a| a == &Operand::Move(inner_param))
                    .then_some((loc, term))
            })
        else {
            debug!("no uses of inner_param found as a by-move call arg");
            return;
        };
        debug!("===> outer_call_loc = {:?}, inner_call_loc = {:?}", outer_call_loc, inner_call_loc);

        let inner_call_span = inner_call_term.source_info.span;
        let outer_call_span = match use_span {
            Some(span) => span,
            None => outer_call_stmt.either(|s| s.source_info, |t| t.source_info).span,
        };
        if outer_call_span == inner_call_span || !outer_call_span.contains(inner_call_span) {
            // FIXME: This stops the suggestion in some cases where it should be emitted.
            //        Fix the spans for those cases so it's emitted correctly.
            debug!(
                "outer span {:?} does not strictly contain inner span {:?}",
                outer_call_span, inner_call_span
            );
            return;
        }
        err.span_help(
            inner_call_span,
            format!(
                "try adding a local storing this{}...",
                if use_span.is_some() { "" } else { " argument" }
            ),
        );
        err.span_help(
            outer_call_span,
            format!(
                "...and then using that local {}",
                if use_span.is_some() { "here" } else { "as the argument to this call" }
            ),
        );
    }

    pub(crate) fn find_expr(&self, span: Span) -> Option<&'tcx hir::Expr<'tcx>> {
        let tcx = self.infcx.tcx;
        let body_id = tcx.hir_node(self.mir_hir_id()).body_id()?;
        let mut expr_finder = FindExprBySpan::new(span, tcx);
        expr_finder.visit_expr(tcx.hir_body(body_id).value);
        expr_finder.result
    }

    fn suggest_slice_method_if_applicable(
        &self,
        err: &mut Diag<'_>,
        place: Place<'tcx>,
        borrowed_place: Place<'tcx>,
        span: Span,
        issued_span: Span,
    ) {
        let tcx = self.infcx.tcx;

        let has_split_at_mut = |ty: Ty<'tcx>| {
            let ty = ty.peel_refs();
            match ty.kind() {
                ty::Array(..) | ty::Slice(..) => true,
                ty::Adt(def, _) if tcx.get_diagnostic_item(sym::Vec) == Some(def.did()) => true,
                _ if ty == tcx.types.str_ => true,
                _ => false,
            }
        };
        if let ([ProjectionElem::Index(index1)], [ProjectionElem::Index(index2)])
        | (
            [ProjectionElem::Deref, ProjectionElem::Index(index1)],
            [ProjectionElem::Deref, ProjectionElem::Index(index2)],
        ) = (&place.projection[..], &borrowed_place.projection[..])
        {
            let decl1 = &self.body.local_decls[*index1];
            let decl2 = &self.body.local_decls[*index2];

            let mut note_default_suggestion = || {
                err.help(
                    "consider using `.split_at_mut(position)` or similar method to obtain two \
                     mutable non-overlapping sub-slices",
                )
                .help(
                    "consider using `.swap(index_1, index_2)` to swap elements at the specified \
                     indices",
                );
            };

            let Some(index1) = self.find_expr(decl1.source_info.span) else {
                note_default_suggestion();
                return;
            };

            let Some(index2) = self.find_expr(decl2.source_info.span) else {
                note_default_suggestion();
                return;
            };

            let sm = tcx.sess.source_map();

            let Ok(index1_str) = sm.span_to_snippet(index1.span) else {
                note_default_suggestion();
                return;
            };

            let Ok(index2_str) = sm.span_to_snippet(index2.span) else {
                note_default_suggestion();
                return;
            };

            let Some(object) = tcx.hir_parent_id_iter(index1.hir_id).find_map(|id| {
                if let hir::Node::Expr(expr) = tcx.hir_node(id)
                    && let hir::ExprKind::Index(obj, ..) = expr.kind
                {
                    Some(obj)
                } else {
                    None
                }
            }) else {
                note_default_suggestion();
                return;
            };

            let Ok(obj_str) = sm.span_to_snippet(object.span) else {
                note_default_suggestion();
                return;
            };

            let Some(swap_call) = tcx.hir_parent_id_iter(object.hir_id).find_map(|id| {
                if let hir::Node::Expr(call) = tcx.hir_node(id)
                    && let hir::ExprKind::Call(callee, ..) = call.kind
                    && let hir::ExprKind::Path(qpath) = callee.kind
                    && let hir::QPath::Resolved(None, res) = qpath
                    && let hir::def::Res::Def(_, did) = res.res
                    && tcx.is_diagnostic_item(sym::mem_swap, did)
                {
                    Some(call)
                } else {
                    None
                }
            }) else {
                let hir::Node::Expr(parent) = tcx.parent_hir_node(index1.hir_id) else { return };
                let hir::ExprKind::Index(_, idx1, _) = parent.kind else { return };
                let hir::Node::Expr(parent) = tcx.parent_hir_node(index2.hir_id) else { return };
                let hir::ExprKind::Index(_, idx2, _) = parent.kind else { return };
                if !idx1.equivalent_for_indexing(idx2) {
                    err.help("use `.split_at_mut(position)` to obtain two mutable non-overlapping sub-slices");
                }
                return;
            };

            err.span_suggestion(
                swap_call.span,
                "use `.swap()` to swap elements at the specified indices instead",
                format!("{obj_str}.swap({index1_str}, {index2_str})"),
                Applicability::MachineApplicable,
            );
            return;
        }
        let place_ty = PlaceRef::ty(&place.as_ref(), self.body, tcx).ty;
        let borrowed_place_ty = PlaceRef::ty(&borrowed_place.as_ref(), self.body, tcx).ty;
        if !has_split_at_mut(place_ty) && !has_split_at_mut(borrowed_place_ty) {
            // Only mention `split_at_mut` on `Vec`, array and slices.
            return;
        }
        let Some(index1) = self.find_expr(span) else { return };
        let hir::Node::Expr(parent) = tcx.parent_hir_node(index1.hir_id) else { return };
        let hir::ExprKind::Index(_, idx1, _) = parent.kind else { return };
        let Some(index2) = self.find_expr(issued_span) else { return };
        let hir::Node::Expr(parent) = tcx.parent_hir_node(index2.hir_id) else { return };
        let hir::ExprKind::Index(_, idx2, _) = parent.kind else { return };
        if idx1.equivalent_for_indexing(idx2) {
            // `let a = &mut foo[0]` and `let b = &mut foo[0]`? Don't mention `split_at_mut`
            return;
        }
        err.help("use `.split_at_mut(position)` to obtain two mutable non-overlapping sub-slices");
    }

    /// Suggest using `while let` for call `next` on an iterator in a for loop.
    ///
    /// For example:
    /// ```ignore (illustrative)
    ///
    /// for x in iter {
    ///     ...
    ///     iter.next()
    /// }
    /// ```
    pub(crate) fn explain_iterator_advancement_in_for_loop_if_applicable(
        &self,
        err: &mut Diag<'_>,
        span: Span,
        issued_spans: &UseSpans<'tcx>,
    ) {
        let issue_span = issued_spans.args_or_use();
        let tcx = self.infcx.tcx;

        let Some(body_id) = tcx.hir_node(self.mir_hir_id()).body_id() else { return };
        let typeck_results = tcx.typeck(self.mir_def_id());

        struct ExprFinder<'hir> {
            issue_span: Span,
            expr_span: Span,
            body_expr: Option<&'hir hir::Expr<'hir>>,
            loop_bind: Option<&'hir Ident>,
            loop_span: Option<Span>,
            head_span: Option<Span>,
            pat_span: Option<Span>,
            head: Option<&'hir hir::Expr<'hir>>,
        }
        impl<'hir> Visitor<'hir> for ExprFinder<'hir> {
            fn visit_expr(&mut self, ex: &'hir hir::Expr<'hir>) {
                // Try to find
                // let result = match IntoIterator::into_iter(<head>) {
                //     mut iter => {
                //         [opt_ident]: loop {
                //             match Iterator::next(&mut iter) {
                //                 None => break,
                //                 Some(<pat>) => <body>,
                //             };
                //         }
                //     }
                // };
                // corresponding to the desugaring of a for loop `for <pat> in <head> { <body> }`.
                if let hir::ExprKind::Call(path, [arg]) = ex.kind
                    && let hir::ExprKind::Path(hir::QPath::LangItem(LangItem::IntoIterIntoIter, _)) =
                        path.kind
                    && arg.span.contains(self.issue_span)
                {
                    // Find `IntoIterator::into_iter(<head>)`
                    self.head = Some(arg);
                }
                if let hir::ExprKind::Loop(
                    hir::Block { stmts: [stmt, ..], .. },
                    _,
                    hir::LoopSource::ForLoop,
                    _,
                ) = ex.kind
                    && let hir::StmtKind::Expr(hir::Expr {
                        kind: hir::ExprKind::Match(call, [_, bind, ..], _),
                        span: head_span,
                        ..
                    }) = stmt.kind
                    && let hir::ExprKind::Call(path, _args) = call.kind
                    && let hir::ExprKind::Path(hir::QPath::LangItem(LangItem::IteratorNext, _)) =
                        path.kind
                    && let hir::PatKind::Struct(path, [field, ..], _) = bind.pat.kind
                    && let hir::QPath::LangItem(LangItem::OptionSome, pat_span) = path
                    && call.span.contains(self.issue_span)
                {
                    // Find `<pat>` and the span for the whole `for` loop.
                    if let PatField {
                        pat: hir::Pat { kind: hir::PatKind::Binding(_, _, ident, ..), .. },
                        ..
                    } = field
                    {
                        self.loop_bind = Some(ident);
                    }
                    self.head_span = Some(*head_span);
                    self.pat_span = Some(pat_span);
                    self.loop_span = Some(stmt.span);
                }

                if let hir::ExprKind::MethodCall(body_call, recv, ..) = ex.kind
                    && body_call.ident.name == sym::next
                    && recv.span.source_equal(self.expr_span)
                {
                    self.body_expr = Some(ex);
                }

                hir::intravisit::walk_expr(self, ex);
            }
        }
        let mut finder = ExprFinder {
            expr_span: span,
            issue_span,
            loop_bind: None,
            body_expr: None,
            head_span: None,
            loop_span: None,
            pat_span: None,
            head: None,
        };
        finder.visit_expr(tcx.hir_body(body_id).value);

        if let Some(body_expr) = finder.body_expr
            && let Some(loop_span) = finder.loop_span
            && let Some(def_id) = typeck_results.type_dependent_def_id(body_expr.hir_id)
            && let Some(trait_did) = tcx.trait_of_item(def_id)
            && tcx.is_diagnostic_item(sym::Iterator, trait_did)
        {
            if let Some(loop_bind) = finder.loop_bind {
                err.note(format!(
                    "a for loop advances the iterator for you, the result is stored in `{}`",
                    loop_bind.name,
                ));
            } else {
                err.note(
                    "a for loop advances the iterator for you, the result is stored in its pattern",
                );
            }
            let msg = "if you want to call `next` on a iterator within the loop, consider using \
                       `while let`";
            if let Some(head) = finder.head
                && let Some(pat_span) = finder.pat_span
                && loop_span.contains(body_expr.span)
                && loop_span.contains(head.span)
            {
                let sm = self.infcx.tcx.sess.source_map();

                let mut sugg = vec![];
                if let hir::ExprKind::Path(hir::QPath::Resolved(None, _)) = head.kind {
                    // A bare path doesn't need a `let` assignment, it's already a simple
                    // binding access.
                    // As a new binding wasn't added, we don't need to modify the advancing call.
                    sugg.push((loop_span.with_hi(pat_span.lo()), "while let Some(".to_string()));
                    sugg.push((
                        pat_span.shrink_to_hi().with_hi(head.span.lo()),
                        ") = ".to_string(),
                    ));
                    sugg.push((head.span.shrink_to_hi(), ".next()".to_string()));
                } else {
                    // Needs a new a `let` binding.
                    let indent = if let Some(indent) = sm.indentation_before(loop_span) {
                        format!("\n{indent}")
                    } else {
                        " ".to_string()
                    };
                    let Ok(head_str) = sm.span_to_snippet(head.span) else {
                        err.help(msg);
                        return;
                    };
                    sugg.push((
                        loop_span.with_hi(pat_span.lo()),
                        format!("let iter = {head_str};{indent}while let Some("),
                    ));
                    sugg.push((
                        pat_span.shrink_to_hi().with_hi(head.span.hi()),
                        ") = iter.next()".to_string(),
                    ));
                    // As a new binding was added, we should change how the iterator is advanced to
                    // use the newly introduced binding.
                    if let hir::ExprKind::MethodCall(_, recv, ..) = body_expr.kind
                        && let hir::ExprKind::Path(hir::QPath::Resolved(None, ..)) = recv.kind
                    {
                        // As we introduced a `let iter = <head>;`, we need to change where the
                        // already borrowed value was accessed from `<recv>.next()` to
                        // `iter.next()`.
                        sugg.push((recv.span, "iter".to_string()));
                    }
                }
                err.multipart_suggestion(msg, sugg, Applicability::MaybeIncorrect);
            } else {
                err.help(msg);
            }
        }
    }

    /// Suggest using closure argument instead of capture.
    ///
    /// For example:
    /// ```ignore (illustrative)
    /// struct S;
    ///
    /// impl S {
    ///     fn call(&mut self, f: impl Fn(&mut Self)) { /* ... */ }
    ///     fn x(&self) {}
    /// }
    ///
    ///     let mut v = S;
    ///     v.call(|this: &mut S| v.x());
    /// //  ^\                    ^-- help: try using the closure argument: `this`
    /// //    *-- error: cannot borrow `v` as mutable because it is also borrowed as immutable
    /// ```
    fn suggest_using_closure_argument_instead_of_capture(
        &self,
        err: &mut Diag<'_>,
        borrowed_place: Place<'tcx>,
        issued_spans: &UseSpans<'tcx>,
    ) {
        let &UseSpans::ClosureUse { capture_kind_span, .. } = issued_spans else { return };
        let tcx = self.infcx.tcx;

        // Get the type of the local that we are trying to borrow
        let local = borrowed_place.local;
        let local_ty = self.body.local_decls[local].ty;

        // Get the body the error happens in
        let Some(body_id) = tcx.hir_node(self.mir_hir_id()).body_id() else { return };

        let body_expr = tcx.hir_body(body_id).value;

        struct ClosureFinder<'hir> {
            tcx: TyCtxt<'hir>,
            borrow_span: Span,
            res: Option<(&'hir hir::Expr<'hir>, &'hir hir::Closure<'hir>)>,
            /// The path expression with the `borrow_span` span
            error_path: Option<(&'hir hir::Expr<'hir>, &'hir hir::QPath<'hir>)>,
        }
        impl<'hir> Visitor<'hir> for ClosureFinder<'hir> {
            type NestedFilter = OnlyBodies;

            fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
                self.tcx
            }

            fn visit_expr(&mut self, ex: &'hir hir::Expr<'hir>) {
                if let hir::ExprKind::Path(qpath) = &ex.kind
                    && ex.span == self.borrow_span
                {
                    self.error_path = Some((ex, qpath));
                }

                if let hir::ExprKind::Closure(closure) = ex.kind
                    && ex.span.contains(self.borrow_span)
                    // To support cases like `|| { v.call(|this| v.get()) }`
                    // FIXME: actually support such cases (need to figure out how to move from the
                    // capture place to original local).
                    && self.res.as_ref().is_none_or(|(prev_res, _)| prev_res.span.contains(ex.span))
                {
                    self.res = Some((ex, closure));
                }

                hir::intravisit::walk_expr(self, ex);
            }
        }

        // Find the closure that most tightly wraps `capture_kind_span`
        let mut finder =
            ClosureFinder { tcx, borrow_span: capture_kind_span, res: None, error_path: None };
        finder.visit_expr(body_expr);
        let Some((closure_expr, closure)) = finder.res else { return };

        let typeck_results = tcx.typeck(self.mir_def_id());

        // Check that the parent of the closure is a method call,
        // with receiver matching with local's type (modulo refs)
        if let hir::Node::Expr(parent) = tcx.parent_hir_node(closure_expr.hir_id) {
            if let hir::ExprKind::MethodCall(_, recv, ..) = parent.kind {
                let recv_ty = typeck_results.expr_ty(recv);

                if recv_ty.peel_refs() != local_ty {
                    return;
                }
            }
        }

        // Get closure's arguments
        let ty::Closure(_, args) = typeck_results.expr_ty(closure_expr).kind() else {
            /* hir::Closure can be a coroutine too */
            return;
        };
        let sig = args.as_closure().sig();
        let tupled_params = tcx.instantiate_bound_regions_with_erased(
            sig.inputs().iter().next().unwrap().map_bound(|&b| b),
        );
        let ty::Tuple(params) = tupled_params.kind() else { return };

        // Find the first argument with a matching type and get its identifier.
        let Some(this_name) = params.iter().zip(tcx.hir_body_param_idents(closure.body)).find_map(
            |(param_ty, ident)| {
                // FIXME: also support deref for stuff like `Rc` arguments
                if param_ty.peel_refs() == local_ty { ident } else { None }
            },
        ) else {
            return;
        };

        let spans;
        if let Some((_path_expr, qpath)) = finder.error_path
            && let hir::QPath::Resolved(_, path) = qpath
            && let hir::def::Res::Local(local_id) = path.res
        {
            // Find all references to the problematic variable in this closure body

            struct VariableUseFinder {
                local_id: hir::HirId,
                spans: Vec<Span>,
            }
            impl<'hir> Visitor<'hir> for VariableUseFinder {
                fn visit_expr(&mut self, ex: &'hir hir::Expr<'hir>) {
                    if let hir::ExprKind::Path(qpath) = &ex.kind
                        && let hir::QPath::Resolved(_, path) = qpath
                        && let hir::def::Res::Local(local_id) = path.res
                        && local_id == self.local_id
                    {
                        self.spans.push(ex.span);
                    }

                    hir::intravisit::walk_expr(self, ex);
                }
            }

            let mut finder = VariableUseFinder { local_id, spans: Vec::new() };
            finder.visit_expr(tcx.hir_body(closure.body).value);

            spans = finder.spans;
        } else {
            spans = vec![capture_kind_span];
        }

        err.multipart_suggestion(
            "try using the closure argument",
            iter::zip(spans, iter::repeat(this_name.to_string())).collect(),
            Applicability::MaybeIncorrect,
        );
    }

    fn suggest_binding_for_closure_capture_self(
        &self,
        err: &mut Diag<'_>,
        issued_spans: &UseSpans<'tcx>,
    ) {
        let UseSpans::ClosureUse { capture_kind_span, .. } = issued_spans else { return };

        struct ExpressionFinder<'tcx> {
            capture_span: Span,
            closure_change_spans: Vec<Span>,
            closure_arg_span: Option<Span>,
            in_closure: bool,
            suggest_arg: String,
            tcx: TyCtxt<'tcx>,
            closure_local_id: Option<hir::HirId>,
            closure_call_changes: Vec<(Span, String)>,
        }
        impl<'hir> Visitor<'hir> for ExpressionFinder<'hir> {
            fn visit_expr(&mut self, e: &'hir hir::Expr<'hir>) {
                if e.span.contains(self.capture_span)
                    && let hir::ExprKind::Closure(&hir::Closure {
                        kind: hir::ClosureKind::Closure,
                        body,
                        fn_arg_span,
                        fn_decl: hir::FnDecl { inputs, .. },
                        ..
                    }) = e.kind
                    && let hir::Node::Expr(body) = self.tcx.hir_node(body.hir_id)
                {
                    self.suggest_arg = "this: &Self".to_string();
                    if inputs.len() > 0 {
                        self.suggest_arg.push_str(", ");
                    }
                    self.in_closure = true;
                    self.closure_arg_span = fn_arg_span;
                    self.visit_expr(body);
                    self.in_closure = false;
                }
                if let hir::Expr { kind: hir::ExprKind::Path(path), .. } = e
                    && let hir::QPath::Resolved(_, hir::Path { segments: [seg], .. }) = path
                    && seg.ident.name == kw::SelfLower
                    && self.in_closure
                {
                    self.closure_change_spans.push(e.span);
                }
                hir::intravisit::walk_expr(self, e);
            }

            fn visit_local(&mut self, local: &'hir hir::LetStmt<'hir>) {
                if let hir::Pat { kind: hir::PatKind::Binding(_, hir_id, _ident, _), .. } =
                    local.pat
                    && let Some(init) = local.init
                    && let &hir::Expr {
                        kind:
                            hir::ExprKind::Closure(&hir::Closure {
                                kind: hir::ClosureKind::Closure,
                                ..
                            }),
                        ..
                    } = init
                    && init.span.contains(self.capture_span)
                {
                    self.closure_local_id = Some(*hir_id);
                }

                hir::intravisit::walk_local(self, local);
            }

            fn visit_stmt(&mut self, s: &'hir hir::Stmt<'hir>) {
                if let hir::StmtKind::Semi(e) = s.kind
                    && let hir::ExprKind::Call(
                        hir::Expr { kind: hir::ExprKind::Path(path), .. },
                        args,
                    ) = e.kind
                    && let hir::QPath::Resolved(_, hir::Path { segments: [seg], .. }) = path
                    && let Res::Local(hir_id) = seg.res
                    && Some(hir_id) == self.closure_local_id
                {
                    let (span, arg_str) = if args.len() > 0 {
                        (args[0].span.shrink_to_lo(), "self, ".to_string())
                    } else {
                        let span = e.span.trim_start(seg.ident.span).unwrap_or(e.span);
                        (span, "(self)".to_string())
                    };
                    self.closure_call_changes.push((span, arg_str));
                }
                hir::intravisit::walk_stmt(self, s);
            }
        }

        if let hir::Node::ImplItem(hir::ImplItem {
            kind: hir::ImplItemKind::Fn(_fn_sig, body_id),
            ..
        }) = self.infcx.tcx.hir_node(self.mir_hir_id())
            && let hir::Node::Expr(expr) = self.infcx.tcx.hir_node(body_id.hir_id)
        {
            let mut finder = ExpressionFinder {
                capture_span: *capture_kind_span,
                closure_change_spans: vec![],
                closure_arg_span: None,
                in_closure: false,
                suggest_arg: String::new(),
                closure_local_id: None,
                closure_call_changes: vec![],
                tcx: self.infcx.tcx,
            };
            finder.visit_expr(expr);

            if finder.closure_change_spans.is_empty() || finder.closure_call_changes.is_empty() {
                return;
            }

            let sm = self.infcx.tcx.sess.source_map();
            let sugg = finder
                .closure_arg_span
                .map(|span| (sm.next_point(span.shrink_to_lo()).shrink_to_hi(), finder.suggest_arg))
                .into_iter()
                .chain(
                    finder.closure_change_spans.into_iter().map(|span| (span, "this".to_string())),
                )
                .chain(finder.closure_call_changes)
                .collect();

            err.multipart_suggestion_verbose(
                "try explicitly passing `&Self` into the closure as an argument",
                sugg,
                Applicability::MachineApplicable,
            );
        }
    }

    /// Returns the description of the root place for a conflicting borrow and the full
    /// descriptions of the places that caused the conflict.
    ///
    /// In the simplest case, where there are no unions involved, if a mutable borrow of `x` is
    /// attempted while a shared borrow is live, then this function will return:
    /// ```
    /// ("x", "", "")
    /// # ;
    /// ```
    /// In the simple union case, if a mutable borrow of a union field `x.z` is attempted while
    /// a shared borrow of another field `x.y`, then this function will return:
    /// ```
    /// ("x", "x.z", "x.y")
    /// # ;
    /// ```
    /// In the more complex union case, where the union is a field of a struct, then if a mutable
    /// borrow of a union field in a struct `x.u.z` is attempted while a shared borrow of
    /// another field `x.u.y`, then this function will return:
    /// ```
    /// ("x.u", "x.u.z", "x.u.y")
    /// # ;
    /// ```
    /// This is used when creating error messages like below:
    ///
    /// ```text
    /// cannot borrow `a.u` (via `a.u.z.c`) as immutable because it is also borrowed as
    /// mutable (via `a.u.s.b`) [E0502]
    /// ```
    fn describe_place_for_conflicting_borrow(
        &self,
        first_borrowed_place: Place<'tcx>,
        second_borrowed_place: Place<'tcx>,
    ) -> (String, String, String, String) {
        // Define a small closure that we can use to check if the type of a place
        // is a union.
        let union_ty = |place_base| {
            // Need to use fn call syntax `PlaceRef::ty` to determine the type of `place_base`;
            // using a type annotation in the closure argument instead leads to a lifetime error.
            let ty = PlaceRef::ty(&place_base, self.body, self.infcx.tcx).ty;
            ty.ty_adt_def().filter(|adt| adt.is_union()).map(|_| ty)
        };

        // Start with an empty tuple, so we can use the functions on `Option` to reduce some
        // code duplication (particularly around returning an empty description in the failure
        // case).
        Some(())
            .filter(|_| {
                // If we have a conflicting borrow of the same place, then we don't want to add
                // an extraneous "via x.y" to our diagnostics, so filter out this case.
                first_borrowed_place != second_borrowed_place
            })
            .and_then(|_| {
                // We're going to want to traverse the first borrowed place to see if we can find
                // field access to a union. If we find that, then we will keep the place of the
                // union being accessed and the field that was being accessed so we can check the
                // second borrowed place for the same union and an access to a different field.
                for (place_base, elem) in first_borrowed_place.iter_projections().rev() {
                    match elem {
                        ProjectionElem::Field(field, _) if union_ty(place_base).is_some() => {
                            return Some((place_base, field));
                        }
                        _ => {}
                    }
                }
                None
            })
            .and_then(|(target_base, target_field)| {
                // With the place of a union and a field access into it, we traverse the second
                // borrowed place and look for an access to a different field of the same union.
                for (place_base, elem) in second_borrowed_place.iter_projections().rev() {
                    if let ProjectionElem::Field(field, _) = elem {
                        if let Some(union_ty) = union_ty(place_base) {
                            if field != target_field && place_base == target_base {
                                return Some((
                                    self.describe_any_place(place_base),
                                    self.describe_any_place(first_borrowed_place.as_ref()),
                                    self.describe_any_place(second_borrowed_place.as_ref()),
                                    union_ty.to_string(),
                                ));
                            }
                        }
                    }
                }
                None
            })
            .unwrap_or_else(|| {
                // If we didn't find a field access into a union, or both places match, then
                // only return the description of the first place.
                (
                    self.describe_any_place(first_borrowed_place.as_ref()),
                    "".to_string(),
                    "".to_string(),
                    "".to_string(),
                )
            })
    }

    /// This means that some data referenced by `borrow` needs to live
    /// past the point where the StorageDeadOrDrop of `place` occurs.
    /// This is usually interpreted as meaning that `place` has too
    /// short a lifetime. (But sometimes it is more useful to report
    /// it as a more direct conflict between the execution of a
    /// `Drop::drop` with an aliasing borrow.)
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn report_borrowed_value_does_not_live_long_enough(
        &mut self,
        location: Location,
        borrow: &BorrowData<'tcx>,
        place_span: (Place<'tcx>, Span),
        kind: Option<WriteKind>,
    ) {
        let drop_span = place_span.1;
        let borrowed_local = borrow.borrowed_place.local;

        let borrow_spans = self.retrieve_borrow_spans(borrow);
        let borrow_span = borrow_spans.var_or_use_path_span();

        let proper_span = self.body.local_decls[borrowed_local].source_info.span;

        if self.access_place_error_reported.contains(&(Place::from(borrowed_local), borrow_span)) {
            debug!(
                "suppressing access_place error when borrow doesn't live long enough for {:?}",
                borrow_span
            );
            return;
        }

        self.access_place_error_reported.insert((Place::from(borrowed_local), borrow_span));

        if self.body.local_decls[borrowed_local].is_ref_to_thread_local() {
            let err =
                self.report_thread_local_value_does_not_live_long_enough(drop_span, borrow_span);
            self.buffer_error(err);
            return;
        }

        if let StorageDeadOrDrop::Destructor(dropped_ty) =
            self.classify_drop_access_kind(borrow.borrowed_place.as_ref())
        {
            // If a borrow of path `B` conflicts with drop of `D` (and
            // we're not in the uninteresting case where `B` is a
            // prefix of `D`), then report this as a more interesting
            // destructor conflict.
            if !borrow.borrowed_place.as_ref().is_prefix_of(place_span.0.as_ref()) {
                self.report_borrow_conflicts_with_destructor(
                    location, borrow, place_span, kind, dropped_ty,
                );
                return;
            }
        }

        let place_desc = self.describe_place(borrow.borrowed_place.as_ref());

        let kind_place = kind.filter(|_| place_desc.is_some()).map(|k| (k, place_span.0));
        let explanation = self.explain_why_borrow_contains_point(location, borrow, kind_place);

        debug!(?place_desc, ?explanation);

        let mut err = match (place_desc, explanation) {
            // If the outlives constraint comes from inside the closure,
            // for example:
            //
            // let x = 0;
            // let y = &x;
            // Box::new(|| y) as Box<Fn() -> &'static i32>
            //
            // then just use the normal error. The closure isn't escaping
            // and `move` will not help here.
            (
                Some(name),
                BorrowExplanation::UsedLater(_, LaterUseKind::ClosureCapture, var_or_use_span, _),
            ) if borrow_spans.for_coroutine() || borrow_spans.for_closure() => self
                .report_escaping_closure_capture(
                    borrow_spans,
                    borrow_span,
                    &RegionName {
                        name: self.synthesize_region_name(),
                        source: RegionNameSource::Static,
                    },
                    ConstraintCategory::CallArgument(None),
                    var_or_use_span,
                    &format!("`{name}`"),
                    "block",
                ),
            (
                Some(name),
                BorrowExplanation::MustBeValidFor {
                    category:
                        category @ (ConstraintCategory::Return(_)
                        | ConstraintCategory::CallArgument(_)
                        | ConstraintCategory::OpaqueType),
                    from_closure: false,
                    ref region_name,
                    span,
                    ..
                },
            ) if borrow_spans.for_coroutine() || borrow_spans.for_closure() => self
                .report_escaping_closure_capture(
                    borrow_spans,
                    borrow_span,
                    region_name,
                    category,
                    span,
                    &format!("`{name}`"),
                    "function",
                ),
            (
                name,
                BorrowExplanation::MustBeValidFor {
                    category: ConstraintCategory::Assignment,
                    from_closure: false,
                    region_name:
                        RegionName {
                            source: RegionNameSource::AnonRegionFromUpvar(upvar_span, upvar_name),
                            ..
                        },
                    span,
                    ..
                },
            ) => self.report_escaping_data(borrow_span, &name, upvar_span, upvar_name, span),
            (Some(name), explanation) => self.report_local_value_does_not_live_long_enough(
                location,
                &name,
                borrow,
                drop_span,
                borrow_spans,
                explanation,
            ),
            (None, explanation) => self.report_temporary_value_does_not_live_long_enough(
                location,
                borrow,
                drop_span,
                borrow_spans,
                proper_span,
                explanation,
            ),
        };
        self.note_due_to_edition_2024_opaque_capture_rules(borrow, &mut err);

        self.buffer_error(err);
    }

    fn report_local_value_does_not_live_long_enough(
        &self,
        location: Location,
        name: &str,
        borrow: &BorrowData<'tcx>,
        drop_span: Span,
        borrow_spans: UseSpans<'tcx>,
        explanation: BorrowExplanation<'tcx>,
    ) -> Diag<'infcx> {
        debug!(
            "report_local_value_does_not_live_long_enough(\
             {:?}, {:?}, {:?}, {:?}, {:?}\
             )",
            location, name, borrow, drop_span, borrow_spans
        );

        let borrow_span = borrow_spans.var_or_use_path_span();
        if let BorrowExplanation::MustBeValidFor {
            category,
            span,
            ref opt_place_desc,
            from_closure: false,
            ..
        } = explanation
        {
            if let Err(diag) = self.try_report_cannot_return_reference_to_local(
                borrow,
                borrow_span,
                span,
                category,
                opt_place_desc.as_ref(),
            ) {
                return diag;
            }
        }

        let name = format!("`{name}`");

        let mut err = self.path_does_not_live_long_enough(borrow_span, &name);

        if let Some(annotation) = self.annotate_argument_and_return_for_borrow(borrow) {
            let region_name = annotation.emit(self, &mut err);

            err.span_label(
                borrow_span,
                format!("{name} would have to be valid for `{region_name}`..."),
            );

            err.span_label(
                drop_span,
                format!(
                    "...but {name} will be dropped here, when the {} returns",
                    self.infcx
                        .tcx
                        .opt_item_name(self.mir_def_id().to_def_id())
                        .map(|name| format!("function `{name}`"))
                        .unwrap_or_else(|| {
                            match &self.infcx.tcx.def_kind(self.mir_def_id()) {
                                DefKind::Closure
                                    if self
                                        .infcx
                                        .tcx
                                        .is_coroutine(self.mir_def_id().to_def_id()) =>
                                {
                                    "enclosing coroutine"
                                }
                                DefKind::Closure => "enclosing closure",
                                kind => bug!("expected closure or coroutine, found {:?}", kind),
                            }
                            .to_string()
                        })
                ),
            );

            err.note(
                "functions cannot return a borrow to data owned within the function's scope, \
                    functions can only return borrows to data passed as arguments",
            );
            err.note(
                "to learn more, visit <https://doc.rust-lang.org/book/ch04-02-\
                    references-and-borrowing.html#dangling-references>",
            );

            if let BorrowExplanation::MustBeValidFor { .. } = explanation {
            } else {
                explanation.add_explanation_to_diagnostic(&self, &mut err, "", None, None);
            }
        } else {
            err.span_label(borrow_span, "borrowed value does not live long enough");
            err.span_label(drop_span, format!("{name} dropped here while still borrowed"));

            borrow_spans.args_subdiag(&mut err, |args_span| {
                crate::session_diagnostics::CaptureArgLabel::Capture {
                    is_within: borrow_spans.for_coroutine(),
                    args_span,
                }
            });

            explanation.add_explanation_to_diagnostic(&self, &mut err, "", Some(borrow_span), None);
        }

        err
    }

    fn report_borrow_conflicts_with_destructor(
        &mut self,
        location: Location,
        borrow: &BorrowData<'tcx>,
        (place, drop_span): (Place<'tcx>, Span),
        kind: Option<WriteKind>,
        dropped_ty: Ty<'tcx>,
    ) {
        debug!(
            "report_borrow_conflicts_with_destructor(\
             {:?}, {:?}, ({:?}, {:?}), {:?}\
             )",
            location, borrow, place, drop_span, kind,
        );

        let borrow_spans = self.retrieve_borrow_spans(borrow);
        let borrow_span = borrow_spans.var_or_use();

        let mut err = self.cannot_borrow_across_destructor(borrow_span);

        let what_was_dropped = match self.describe_place(place.as_ref()) {
            Some(name) => format!("`{name}`"),
            None => String::from("temporary value"),
        };

        let label = match self.describe_place(borrow.borrowed_place.as_ref()) {
            Some(borrowed) => format!(
                "here, drop of {what_was_dropped} needs exclusive access to `{borrowed}`, \
                 because the type `{dropped_ty}` implements the `Drop` trait"
            ),
            None => format!(
                "here is drop of {what_was_dropped}; whose type `{dropped_ty}` implements the `Drop` trait"
            ),
        };
        err.span_label(drop_span, label);

        // Only give this note and suggestion if they could be relevant.
        let explanation =
            self.explain_why_borrow_contains_point(location, borrow, kind.map(|k| (k, place)));
        match explanation {
            BorrowExplanation::UsedLater { .. }
            | BorrowExplanation::UsedLaterWhenDropped { .. } => {
                err.note("consider using a `let` binding to create a longer lived value");
            }
            _ => {}
        }

        explanation.add_explanation_to_diagnostic(&self, &mut err, "", None, None);

        self.buffer_error(err);
    }

    fn report_thread_local_value_does_not_live_long_enough(
        &self,
        drop_span: Span,
        borrow_span: Span,
    ) -> Diag<'infcx> {
        debug!(
            "report_thread_local_value_does_not_live_long_enough(\
             {:?}, {:?}\
             )",
            drop_span, borrow_span
        );

        // `TerminatorKind::Return`'s span (the `drop_span` here) `lo` can be subtly wrong and point
        // at a single character after the end of the function. This is somehow relied upon in
        // existing diagnostics, and changing this in `rustc_mir_build` makes diagnostics worse in
        // general. We fix these here.
        let sm = self.infcx.tcx.sess.source_map();
        let end_of_function = if drop_span.is_empty()
            && let Ok(adjusted_span) = sm.span_extend_prev_while(drop_span, |c| c == '}')
        {
            adjusted_span
        } else {
            drop_span
        };
        self.thread_local_value_does_not_live_long_enough(borrow_span)
            .with_span_label(
                borrow_span,
                "thread-local variables cannot be borrowed beyond the end of the function",
            )
            .with_span_label(end_of_function, "end of enclosing function is here")
    }

    #[instrument(level = "debug", skip(self))]
    fn report_temporary_value_does_not_live_long_enough(
        &self,
        location: Location,
        borrow: &BorrowData<'tcx>,
        drop_span: Span,
        borrow_spans: UseSpans<'tcx>,
        proper_span: Span,
        explanation: BorrowExplanation<'tcx>,
    ) -> Diag<'infcx> {
        if let BorrowExplanation::MustBeValidFor { category, span, from_closure: false, .. } =
            explanation
        {
            if let Err(diag) = self.try_report_cannot_return_reference_to_local(
                borrow,
                proper_span,
                span,
                category,
                None,
            ) {
                return diag;
            }
        }

        let mut err = self.temporary_value_borrowed_for_too_long(proper_span);
        err.span_label(proper_span, "creates a temporary value which is freed while still in use");
        err.span_label(drop_span, "temporary value is freed at the end of this statement");

        match explanation {
            BorrowExplanation::UsedLater(..)
            | BorrowExplanation::UsedLaterInLoop(..)
            | BorrowExplanation::UsedLaterWhenDropped { .. } => {
                // Only give this note and suggestion if it could be relevant.
                let sm = self.infcx.tcx.sess.source_map();
                let mut suggested = false;
                let msg = "consider using a `let` binding to create a longer lived value";

                /// We check that there's a single level of block nesting to ensure always correct
                /// suggestions. If we don't, then we only provide a free-form message to avoid
                /// misleading users in cases like `tests/ui/nll/borrowed-temporary-error.rs`.
                /// We could expand the analysis to suggest hoising all of the relevant parts of
                /// the users' code to make the code compile, but that could be too much.
                /// We found the `prop_expr` by the way to check whether the expression is a
                /// `FormatArguments`, which is a special case since it's generated by the
                /// compiler.
                struct NestedStatementVisitor<'tcx> {
                    span: Span,
                    current: usize,
                    found: usize,
                    prop_expr: Option<&'tcx hir::Expr<'tcx>>,
                    call: Option<&'tcx hir::Expr<'tcx>>,
                }

                impl<'tcx> Visitor<'tcx> for NestedStatementVisitor<'tcx> {
                    fn visit_block(&mut self, block: &'tcx hir::Block<'tcx>) {
                        self.current += 1;
                        walk_block(self, block);
                        self.current -= 1;
                    }
                    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
                        if let hir::ExprKind::MethodCall(_, rcvr, _, _) = expr.kind {
                            if self.span == rcvr.span.source_callsite() {
                                self.call = Some(expr);
                            }
                        }
                        if self.span == expr.span.source_callsite() {
                            self.found = self.current;
                            if self.prop_expr.is_none() {
                                self.prop_expr = Some(expr);
                            }
                        }
                        walk_expr(self, expr);
                    }
                }
                let source_info = self.body.source_info(location);
                let proper_span = proper_span.source_callsite();
                if let Some(scope) = self.body.source_scopes.get(source_info.scope)
                    && let ClearCrossCrate::Set(scope_data) = &scope.local_data
                    && let Some(id) = self.infcx.tcx.hir_node(scope_data.lint_root).body_id()
                    && let hir::ExprKind::Block(block, _) = self.infcx.tcx.hir_body(id).value.kind
                {
                    for stmt in block.stmts {
                        let mut visitor = NestedStatementVisitor {
                            span: proper_span,
                            current: 0,
                            found: 0,
                            prop_expr: None,
                            call: None,
                        };
                        visitor.visit_stmt(stmt);

                        let typeck_results = self.infcx.tcx.typeck(self.mir_def_id());
                        let expr_ty: Option<Ty<'_>> =
                            visitor.prop_expr.map(|expr| typeck_results.expr_ty(expr).peel_refs());

                        if visitor.found == 0
                            && stmt.span.contains(proper_span)
                            && let Some(p) = sm.span_to_margin(stmt.span)
                            && let Ok(s) = sm.span_to_snippet(proper_span)
                        {
                            if let Some(call) = visitor.call
                                && let hir::ExprKind::MethodCall(path, _, [], _) = call.kind
                                && path.ident.name == sym::iter
                                && let Some(ty) = expr_ty
                            {
                                err.span_suggestion_verbose(
                                    path.ident.span,
                                    format!(
                                        "consider consuming the `{ty}` when turning it into an \
                                         `Iterator`",
                                    ),
                                    "into_iter",
                                    Applicability::MaybeIncorrect,
                                );
                            }

                            let mutability = if matches!(borrow.kind(), BorrowKind::Mut { .. }) {
                                "mut "
                            } else {
                                ""
                            };

                            let addition =
                                format!("let {}binding = {};\n{}", mutability, s, " ".repeat(p));
                            err.multipart_suggestion_verbose(
                                msg,
                                vec![
                                    (stmt.span.shrink_to_lo(), addition),
                                    (proper_span, "binding".to_string()),
                                ],
                                Applicability::MaybeIncorrect,
                            );

                            suggested = true;
                            break;
                        }
                    }
                }
                if !suggested {
                    err.note(msg);
                }
            }
            _ => {}
        }
        explanation.add_explanation_to_diagnostic(&self, &mut err, "", None, None);

        borrow_spans.args_subdiag(&mut err, |args_span| {
            crate::session_diagnostics::CaptureArgLabel::Capture {
                is_within: borrow_spans.for_coroutine(),
                args_span,
            }
        });

        err
    }

    fn try_report_cannot_return_reference_to_local(
        &self,
        borrow: &BorrowData<'tcx>,
        borrow_span: Span,
        return_span: Span,
        category: ConstraintCategory<'tcx>,
        opt_place_desc: Option<&String>,
    ) -> Result<(), Diag<'infcx>> {
        let return_kind = match category {
            ConstraintCategory::Return(_) => "return",
            ConstraintCategory::Yield => "yield",
            _ => return Ok(()),
        };

        // FIXME use a better heuristic than Spans
        let reference_desc = if return_span == self.body.source_info(borrow.reserve_location).span {
            "reference to"
        } else {
            "value referencing"
        };

        let (place_desc, note) = if let Some(place_desc) = opt_place_desc {
            let local_kind = if let Some(local) = borrow.borrowed_place.as_local() {
                match self.body.local_kind(local) {
                    LocalKind::Temp if self.body.local_decls[local].is_user_variable() => {
                        "local variable "
                    }
                    LocalKind::Arg
                        if !self.upvars.is_empty() && local == ty::CAPTURE_STRUCT_LOCAL =>
                    {
                        "variable captured by `move` "
                    }
                    LocalKind::Arg => "function parameter ",
                    LocalKind::ReturnPointer | LocalKind::Temp => {
                        bug!("temporary or return pointer with a name")
                    }
                }
            } else {
                "local data "
            };
            (format!("{local_kind}`{place_desc}`"), format!("`{place_desc}` is borrowed here"))
        } else {
            let local = borrow.borrowed_place.local;
            match self.body.local_kind(local) {
                LocalKind::Arg => (
                    "function parameter".to_string(),
                    "function parameter borrowed here".to_string(),
                ),
                LocalKind::Temp
                    if self.body.local_decls[local].is_user_variable()
                        && !self.body.local_decls[local]
                            .source_info
                            .span
                            .in_external_macro(self.infcx.tcx.sess.source_map()) =>
                {
                    ("local binding".to_string(), "local binding introduced here".to_string())
                }
                LocalKind::ReturnPointer | LocalKind::Temp => {
                    ("temporary value".to_string(), "temporary value created here".to_string())
                }
            }
        };

        let mut err = self.cannot_return_reference_to_local(
            return_span,
            return_kind,
            reference_desc,
            &place_desc,
        );

        if return_span != borrow_span {
            err.span_label(borrow_span, note);

            let tcx = self.infcx.tcx;

            let return_ty = self.regioncx.universal_regions().unnormalized_output_ty;

            // to avoid panics
            if let Some(iter_trait) = tcx.get_diagnostic_item(sym::Iterator)
                && self
                    .infcx
                    .type_implements_trait(iter_trait, [return_ty], self.infcx.param_env)
                    .must_apply_modulo_regions()
            {
                err.span_suggestion_hidden(
                    return_span.shrink_to_hi(),
                    "use `.collect()` to allocate the iterator",
                    ".collect::<Vec<_>>()",
                    Applicability::MaybeIncorrect,
                );
            }
        }

        Err(err)
    }

    #[instrument(level = "debug", skip(self))]
    fn report_escaping_closure_capture(
        &self,
        use_span: UseSpans<'tcx>,
        var_span: Span,
        fr_name: &RegionName,
        category: ConstraintCategory<'tcx>,
        constraint_span: Span,
        captured_var: &str,
        scope: &str,
    ) -> Diag<'infcx> {
        let tcx = self.infcx.tcx;
        let args_span = use_span.args_or_use();

        let (sugg_span, suggestion) = match tcx.sess.source_map().span_to_snippet(args_span) {
            Ok(string) => {
                let coro_prefix = if let Some(sub) = string.strip_prefix("async") {
                    let trimmed_sub = sub.trim_end();
                    if trimmed_sub.ends_with("gen") {
                        // `async` is 5 chars long.
                        Some((trimmed_sub.len() + 5) as _)
                    } else {
                        // `async` is 5 chars long.
                        Some(5)
                    }
                } else if string.starts_with("gen") {
                    // `gen` is 3 chars long
                    Some(3)
                } else if string.starts_with("static") {
                    // `static` is 6 chars long
                    // This is used for `!Unpin` coroutines
                    Some(6)
                } else {
                    None
                };
                if let Some(n) = coro_prefix {
                    let pos = args_span.lo() + BytePos(n);
                    (args_span.with_lo(pos).with_hi(pos), " move")
                } else {
                    (args_span.shrink_to_lo(), "move ")
                }
            }
            Err(_) => (args_span, "move |<args>| <body>"),
        };
        let kind = match use_span.coroutine_kind() {
            Some(coroutine_kind) => match coroutine_kind {
                CoroutineKind::Desugared(CoroutineDesugaring::Gen, kind) => match kind {
                    CoroutineSource::Block => "gen block",
                    CoroutineSource::Closure => "gen closure",
                    CoroutineSource::Fn => {
                        bug!("gen block/closure expected, but gen function found.")
                    }
                },
                CoroutineKind::Desugared(CoroutineDesugaring::AsyncGen, kind) => match kind {
                    CoroutineSource::Block => "async gen block",
                    CoroutineSource::Closure => "async gen closure",
                    CoroutineSource::Fn => {
                        bug!("gen block/closure expected, but gen function found.")
                    }
                },
                CoroutineKind::Desugared(CoroutineDesugaring::Async, async_kind) => {
                    match async_kind {
                        CoroutineSource::Block => "async block",
                        CoroutineSource::Closure => "async closure",
                        CoroutineSource::Fn => {
                            bug!("async block/closure expected, but async function found.")
                        }
                    }
                }
                CoroutineKind::Coroutine(_) => "coroutine",
            },
            None => "closure",
        };

        let mut err = self.cannot_capture_in_long_lived_closure(
            args_span,
            kind,
            captured_var,
            var_span,
            scope,
        );
        err.span_suggestion_verbose(
            sugg_span,
            format!(
                "to force the {kind} to take ownership of {captured_var} (and any \
                 other referenced variables), use the `move` keyword"
            ),
            suggestion,
            Applicability::MachineApplicable,
        );

        match category {
            ConstraintCategory::Return(_) | ConstraintCategory::OpaqueType => {
                let msg = format!("{kind} is returned here");
                err.span_note(constraint_span, msg);
            }
            ConstraintCategory::CallArgument(_) => {
                fr_name.highlight_region_name(&mut err);
                if matches!(
                    use_span.coroutine_kind(),
                    Some(CoroutineKind::Desugared(CoroutineDesugaring::Async, _))
                ) {
                    err.note(
                        "async blocks are not executed immediately and must either take a \
                         reference or ownership of outside variables they use",
                    );
                } else {
                    let msg = format!("{scope} requires argument type to outlive `{fr_name}`");
                    err.span_note(constraint_span, msg);
                }
            }
            _ => bug!(
                "report_escaping_closure_capture called with unexpected constraint \
                 category: `{:?}`",
                category
            ),
        }

        err
    }

    fn report_escaping_data(
        &self,
        borrow_span: Span,
        name: &Option<String>,
        upvar_span: Span,
        upvar_name: Symbol,
        escape_span: Span,
    ) -> Diag<'infcx> {
        let tcx = self.infcx.tcx;

        let escapes_from = tcx.def_descr(self.mir_def_id().to_def_id());

        let mut err =
            borrowck_errors::borrowed_data_escapes_closure(tcx, escape_span, escapes_from);

        err.span_label(
            upvar_span,
            format!("`{upvar_name}` declared here, outside of the {escapes_from} body"),
        );

        err.span_label(borrow_span, format!("borrow is only valid in the {escapes_from} body"));

        if let Some(name) = name {
            err.span_label(
                escape_span,
                format!("reference to `{name}` escapes the {escapes_from} body here"),
            );
        } else {
            err.span_label(escape_span, format!("reference escapes the {escapes_from} body here"));
        }

        err
    }

    fn get_moved_indexes(
        &self,
        location: Location,
        mpi: MovePathIndex,
    ) -> (Vec<MoveSite>, Vec<Location>) {
        fn predecessor_locations<'tcx>(
            body: &mir::Body<'tcx>,
            location: Location,
        ) -> impl Iterator<Item = Location> {
            if location.statement_index == 0 {
                let predecessors = body.basic_blocks.predecessors()[location.block].to_vec();
                Either::Left(predecessors.into_iter().map(move |bb| body.terminator_loc(bb)))
            } else {
                Either::Right(std::iter::once(Location {
                    statement_index: location.statement_index - 1,
                    ..location
                }))
            }
        }

        let mut mpis = vec![mpi];
        let move_paths = &self.move_data.move_paths;
        mpis.extend(move_paths[mpi].parents(move_paths).map(|(mpi, _)| mpi));

        let mut stack = Vec::new();
        let mut back_edge_stack = Vec::new();

        predecessor_locations(self.body, location).for_each(|predecessor| {
            if location.dominates(predecessor, self.dominators()) {
                back_edge_stack.push(predecessor)
            } else {
                stack.push(predecessor);
            }
        });

        let mut reached_start = false;

        /* Check if the mpi is initialized as an argument */
        let mut is_argument = false;
        for arg in self.body.args_iter() {
            if let Some(path) = self.move_data.rev_lookup.find_local(arg) {
                if mpis.contains(&path) {
                    is_argument = true;
                }
            }
        }

        let mut visited = FxIndexSet::default();
        let mut move_locations = FxIndexSet::default();
        let mut reinits = vec![];
        let mut result = vec![];

        let mut dfs_iter = |result: &mut Vec<MoveSite>, location: Location, is_back_edge: bool| {
            debug!(
                "report_use_of_moved_or_uninitialized: (current_location={:?}, back_edge={})",
                location, is_back_edge
            );

            if !visited.insert(location) {
                return true;
            }

            // check for moves
            let stmt_kind =
                self.body[location.block].statements.get(location.statement_index).map(|s| &s.kind);
            if let Some(StatementKind::StorageDead(..)) = stmt_kind {
                // This analysis only tries to find moves explicitly written by the user, so we
                // ignore the move-outs created by `StorageDead` and at the beginning of a
                // function.
            } else {
                // If we are found a use of a.b.c which was in error, then we want to look for
                // moves not only of a.b.c but also a.b and a.
                //
                // Note that the moves data already includes "parent" paths, so we don't have to
                // worry about the other case: that is, if there is a move of a.b.c, it is already
                // marked as a move of a.b and a as well, so we will generate the correct errors
                // there.
                for moi in &self.move_data.loc_map[location] {
                    debug!("report_use_of_moved_or_uninitialized: moi={:?}", moi);
                    let path = self.move_data.moves[*moi].path;
                    if mpis.contains(&path) {
                        debug!(
                            "report_use_of_moved_or_uninitialized: found {:?}",
                            move_paths[path].place
                        );
                        result.push(MoveSite { moi: *moi, traversed_back_edge: is_back_edge });
                        move_locations.insert(location);

                        // Strictly speaking, we could continue our DFS here. There may be
                        // other moves that can reach the point of error. But it is kind of
                        // confusing to highlight them.
                        //
                        // Example:
                        //
                        // ```
                        // let a = vec![];
                        // let b = a;
                        // let c = a;
                        // drop(a); // <-- current point of error
                        // ```
                        //
                        // Because we stop the DFS here, we only highlight `let c = a`,
                        // and not `let b = a`. We will of course also report an error at
                        // `let c = a` which highlights `let b = a` as the move.
                        return true;
                    }
                }
            }

            // check for inits
            let mut any_match = false;
            for ii in &self.move_data.init_loc_map[location] {
                let init = self.move_data.inits[*ii];
                match init.kind {
                    InitKind::Deep | InitKind::NonPanicPathOnly => {
                        if mpis.contains(&init.path) {
                            any_match = true;
                        }
                    }
                    InitKind::Shallow => {
                        if mpi == init.path {
                            any_match = true;
                        }
                    }
                }
            }
            if any_match {
                reinits.push(location);
                return true;
            }
            false
        };

        while let Some(location) = stack.pop() {
            if dfs_iter(&mut result, location, false) {
                continue;
            }

            let mut has_predecessor = false;
            predecessor_locations(self.body, location).for_each(|predecessor| {
                if location.dominates(predecessor, self.dominators()) {
                    back_edge_stack.push(predecessor)
                } else {
                    stack.push(predecessor);
                }
                has_predecessor = true;
            });

            if !has_predecessor {
                reached_start = true;
            }
        }
        if (is_argument || !reached_start) && result.is_empty() {
            // Process back edges (moves in future loop iterations) only if
            // the move path is definitely initialized upon loop entry,
            // to avoid spurious "in previous iteration" errors.
            // During DFS, if there's a path from the error back to the start
            // of the function with no intervening init or move, then the
            // move path may be uninitialized at loop entry.
            while let Some(location) = back_edge_stack.pop() {
                if dfs_iter(&mut result, location, true) {
                    continue;
                }

                predecessor_locations(self.body, location)
                    .for_each(|predecessor| back_edge_stack.push(predecessor));
            }
        }

        // Check if we can reach these reinits from a move location.
        let reinits_reachable = reinits
            .into_iter()
            .filter(|reinit| {
                let mut visited = FxIndexSet::default();
                let mut stack = vec![*reinit];
                while let Some(location) = stack.pop() {
                    if !visited.insert(location) {
                        continue;
                    }
                    if move_locations.contains(&location) {
                        return true;
                    }
                    stack.extend(predecessor_locations(self.body, location));
                }
                false
            })
            .collect::<Vec<Location>>();
        (result, reinits_reachable)
    }

    pub(crate) fn report_illegal_mutation_of_borrowed(
        &mut self,
        location: Location,
        (place, span): (Place<'tcx>, Span),
        loan: &BorrowData<'tcx>,
    ) {
        let loan_spans = self.retrieve_borrow_spans(loan);
        let loan_span = loan_spans.args_or_use();

        let descr_place = self.describe_any_place(place.as_ref());
        if let BorrowKind::Fake(_) = loan.kind {
            if let Some(section) = self.classify_immutable_section(loan.assigned_place) {
                let mut err = self.cannot_mutate_in_immutable_section(
                    span,
                    loan_span,
                    &descr_place,
                    section,
                    "assign",
                );

                loan_spans.var_subdiag(&mut err, Some(loan.kind), |kind, var_span| {
                    use crate::session_diagnostics::CaptureVarCause::*;
                    match kind {
                        hir::ClosureKind::Coroutine(_) => BorrowUseInCoroutine { var_span },
                        hir::ClosureKind::Closure | hir::ClosureKind::CoroutineClosure(_) => {
                            BorrowUseInClosure { var_span }
                        }
                    }
                });

                self.buffer_error(err);

                return;
            }
        }

        let mut err = self.cannot_assign_to_borrowed(span, loan_span, &descr_place);
        self.note_due_to_edition_2024_opaque_capture_rules(loan, &mut err);

        loan_spans.var_subdiag(&mut err, Some(loan.kind), |kind, var_span| {
            use crate::session_diagnostics::CaptureVarCause::*;
            match kind {
                hir::ClosureKind::Coroutine(_) => BorrowUseInCoroutine { var_span },
                hir::ClosureKind::Closure | hir::ClosureKind::CoroutineClosure(_) => {
                    BorrowUseInClosure { var_span }
                }
            }
        });

        self.explain_why_borrow_contains_point(location, loan, None)
            .add_explanation_to_diagnostic(&self, &mut err, "", None, None);

        self.explain_deref_coercion(loan, &mut err);

        self.buffer_error(err);
    }

    fn explain_deref_coercion(&mut self, loan: &BorrowData<'tcx>, err: &mut Diag<'_>) {
        let tcx = self.infcx.tcx;
        if let Some(Terminator { kind: TerminatorKind::Call { call_source, fn_span, .. }, .. }) =
            &self.body[loan.reserve_location.block].terminator
            && let Some((method_did, method_args)) = mir::find_self_call(
                tcx,
                self.body,
                loan.assigned_place.local,
                loan.reserve_location.block,
            )
            && let CallKind::DerefCoercion { deref_target_span, deref_target_ty, .. } = call_kind(
                self.infcx.tcx,
                self.infcx.typing_env(self.infcx.param_env),
                method_did,
                method_args,
                *fn_span,
                call_source.from_hir_call(),
                self.infcx.tcx.fn_arg_idents(method_did)[0],
            )
        {
            err.note(format!("borrow occurs due to deref coercion to `{deref_target_ty}`"));
            if let Some(deref_target_span) = deref_target_span {
                err.span_note(deref_target_span, "deref defined here");
            }
        }
    }

    /// Reports an illegal reassignment; for example, an assignment to
    /// (part of) a non-`mut` local that occurs potentially after that
    /// local has already been initialized. `place` is the path being
    /// assigned; `err_place` is a place providing a reason why
    /// `place` is not mutable (e.g., the non-`mut` local `x` in an
    /// assignment to `x.f`).
    pub(crate) fn report_illegal_reassignment(
        &mut self,
        (place, span): (Place<'tcx>, Span),
        assigned_span: Span,
        err_place: Place<'tcx>,
    ) {
        let (from_arg, local_decl) = match err_place.as_local() {
            Some(local) => {
                (self.body.local_kind(local) == LocalKind::Arg, Some(&self.body.local_decls[local]))
            }
            None => (false, None),
        };

        // If root local is initialized immediately (everything apart from let
        // PATTERN;) then make the error refer to that local, rather than the
        // place being assigned later.
        let (place_description, assigned_span) = match local_decl {
            Some(LocalDecl {
                local_info:
                    ClearCrossCrate::Set(
                        box LocalInfo::User(BindingForm::Var(VarBindingForm {
                            opt_match_place: None,
                            ..
                        }))
                        | box LocalInfo::StaticRef { .. }
                        | box LocalInfo::Boring,
                    ),
                ..
            })
            | None => (self.describe_any_place(place.as_ref()), assigned_span),
            Some(decl) => (self.describe_any_place(err_place.as_ref()), decl.source_info.span),
        };
        let mut err = self.cannot_reassign_immutable(span, &place_description, from_arg);
        let msg = if from_arg {
            "cannot assign to immutable argument"
        } else {
            "cannot assign twice to immutable variable"
        };
        if span != assigned_span && !from_arg {
            err.span_label(assigned_span, format!("first assignment to {place_description}"));
        }
        if let Some(decl) = local_decl
            && decl.can_be_made_mutable()
        {
            err.span_suggestion_verbose(
                decl.source_info.span.shrink_to_lo(),
                "consider making this binding mutable",
                "mut ".to_string(),
                Applicability::MachineApplicable,
            );
            if !from_arg
                && matches!(
                    decl.local_info(),
                    LocalInfo::User(BindingForm::Var(VarBindingForm {
                        opt_match_place: Some((Some(_), _)),
                        ..
                    }))
                )
            {
                err.span_suggestion_verbose(
                    decl.source_info.span.shrink_to_lo(),
                    "to modify the original value, take a borrow instead",
                    "ref mut ".to_string(),
                    Applicability::MaybeIncorrect,
                );
            }
        }
        err.span_label(span, msg);
        self.buffer_error(err);
    }

    fn classify_drop_access_kind(&self, place: PlaceRef<'tcx>) -> StorageDeadOrDrop<'tcx> {
        let tcx = self.infcx.tcx;
        let (kind, _place_ty) = place.projection.iter().fold(
            (LocalStorageDead, PlaceTy::from_ty(self.body.local_decls[place.local].ty)),
            |(kind, place_ty), &elem| {
                (
                    match elem {
                        ProjectionElem::Deref => match kind {
                            StorageDeadOrDrop::LocalStorageDead
                            | StorageDeadOrDrop::BoxedStorageDead => {
                                assert!(
                                    place_ty.ty.is_box(),
                                    "Drop of value behind a reference or raw pointer"
                                );
                                StorageDeadOrDrop::BoxedStorageDead
                            }
                            StorageDeadOrDrop::Destructor(_) => kind,
                        },
                        ProjectionElem::OpaqueCast { .. }
                        | ProjectionElem::Field(..)
                        | ProjectionElem::Downcast(..) => {
                            match place_ty.ty.kind() {
                                ty::Adt(def, _) if def.has_dtor(tcx) => {
                                    // Report the outermost adt with a destructor
                                    match kind {
                                        StorageDeadOrDrop::Destructor(_) => kind,
                                        StorageDeadOrDrop::LocalStorageDead
                                        | StorageDeadOrDrop::BoxedStorageDead => {
                                            StorageDeadOrDrop::Destructor(place_ty.ty)
                                        }
                                    }
                                }
                                _ => kind,
                            }
                        }
                        ProjectionElem::ConstantIndex { .. }
                        | ProjectionElem::Subslice { .. }
                        | ProjectionElem::Subtype(_)
                        | ProjectionElem::Index(_)
                        | ProjectionElem::UnwrapUnsafeBinder(_) => kind,
                    },
                    place_ty.projection_ty(tcx, elem),
                )
            },
        );
        kind
    }

    /// Describe the reason for the fake borrow that was assigned to `place`.
    fn classify_immutable_section(&self, place: Place<'tcx>) -> Option<&'static str> {
        use rustc_middle::mir::visit::Visitor;
        struct FakeReadCauseFinder<'tcx> {
            place: Place<'tcx>,
            cause: Option<FakeReadCause>,
        }
        impl<'tcx> Visitor<'tcx> for FakeReadCauseFinder<'tcx> {
            fn visit_statement(&mut self, statement: &Statement<'tcx>, _: Location) {
                match statement {
                    Statement { kind: StatementKind::FakeRead(box (cause, place)), .. }
                        if *place == self.place =>
                    {
                        self.cause = Some(*cause);
                    }
                    _ => (),
                }
            }
        }
        let mut visitor = FakeReadCauseFinder { place, cause: None };
        visitor.visit_body(self.body);
        match visitor.cause {
            Some(FakeReadCause::ForMatchGuard) => Some("match guard"),
            Some(FakeReadCause::ForIndex) => Some("indexing expression"),
            _ => None,
        }
    }

    /// Annotate argument and return type of function and closure with (synthesized) lifetime for
    /// borrow of local value that does not live long enough.
    fn annotate_argument_and_return_for_borrow(
        &self,
        borrow: &BorrowData<'tcx>,
    ) -> Option<AnnotatedBorrowFnSignature<'tcx>> {
        // Define a fallback for when we can't match a closure.
        let fallback = || {
            let is_closure = self.infcx.tcx.is_closure_like(self.mir_def_id().to_def_id());
            if is_closure {
                None
            } else {
                let ty = self.infcx.tcx.type_of(self.mir_def_id()).instantiate_identity();
                match ty.kind() {
                    ty::FnDef(_, _) | ty::FnPtr(..) => self.annotate_fn_sig(
                        self.mir_def_id(),
                        self.infcx.tcx.fn_sig(self.mir_def_id()).instantiate_identity(),
                    ),
                    _ => None,
                }
            }
        };

        // In order to determine whether we need to annotate, we need to check whether the reserve
        // place was an assignment into a temporary.
        //
        // If it was, we check whether or not that temporary is eventually assigned into the return
        // place. If it was, we can add annotations about the function's return type and arguments
        // and it'll make sense.
        let location = borrow.reserve_location;
        debug!("annotate_argument_and_return_for_borrow: location={:?}", location);
        if let Some(Statement { kind: StatementKind::Assign(box (reservation, _)), .. }) =
            &self.body[location.block].statements.get(location.statement_index)
        {
            debug!("annotate_argument_and_return_for_borrow: reservation={:?}", reservation);
            // Check that the initial assignment of the reserve location is into a temporary.
            let mut target = match reservation.as_local() {
                Some(local) if self.body.local_kind(local) == LocalKind::Temp => local,
                _ => return None,
            };

            // Next, look through the rest of the block, checking if we are assigning the
            // `target` (that is, the place that contains our borrow) to anything.
            let mut annotated_closure = None;
            for stmt in &self.body[location.block].statements[location.statement_index + 1..] {
                debug!(
                    "annotate_argument_and_return_for_borrow: target={:?} stmt={:?}",
                    target, stmt
                );
                if let StatementKind::Assign(box (place, rvalue)) = &stmt.kind {
                    if let Some(assigned_to) = place.as_local() {
                        debug!(
                            "annotate_argument_and_return_for_borrow: assigned_to={:?} \
                             rvalue={:?}",
                            assigned_to, rvalue
                        );
                        // Check if our `target` was captured by a closure.
                        if let Rvalue::Aggregate(
                            box AggregateKind::Closure(def_id, args),
                            operands,
                        ) = rvalue
                        {
                            let def_id = def_id.expect_local();
                            for operand in operands {
                                let (Operand::Copy(assigned_from) | Operand::Move(assigned_from)) =
                                    operand
                                else {
                                    continue;
                                };
                                debug!(
                                    "annotate_argument_and_return_for_borrow: assigned_from={:?}",
                                    assigned_from
                                );

                                // Find the local from the operand.
                                let Some(assigned_from_local) =
                                    assigned_from.local_or_deref_local()
                                else {
                                    continue;
                                };

                                if assigned_from_local != target {
                                    continue;
                                }

                                // If a closure captured our `target` and then assigned
                                // into a place then we should annotate the closure in
                                // case it ends up being assigned into the return place.
                                annotated_closure =
                                    self.annotate_fn_sig(def_id, args.as_closure().sig());
                                debug!(
                                    "annotate_argument_and_return_for_borrow: \
                                     annotated_closure={:?} assigned_from_local={:?} \
                                     assigned_to={:?}",
                                    annotated_closure, assigned_from_local, assigned_to
                                );

                                if assigned_to == mir::RETURN_PLACE {
                                    // If it was assigned directly into the return place, then
                                    // return now.
                                    return annotated_closure;
                                } else {
                                    // Otherwise, update the target.
                                    target = assigned_to;
                                }
                            }

                            // If none of our closure's operands matched, then skip to the next
                            // statement.
                            continue;
                        }

                        // Otherwise, look at other types of assignment.
                        let assigned_from = match rvalue {
                            Rvalue::Ref(_, _, assigned_from) => assigned_from,
                            Rvalue::Use(operand) => match operand {
                                Operand::Copy(assigned_from) | Operand::Move(assigned_from) => {
                                    assigned_from
                                }
                                _ => continue,
                            },
                            _ => continue,
                        };
                        debug!(
                            "annotate_argument_and_return_for_borrow: \
                             assigned_from={:?}",
                            assigned_from,
                        );

                        // Find the local from the rvalue.
                        let Some(assigned_from_local) = assigned_from.local_or_deref_local() else {
                            continue;
                        };
                        debug!(
                            "annotate_argument_and_return_for_borrow: \
                             assigned_from_local={:?}",
                            assigned_from_local,
                        );

                        // Check if our local matches the target - if so, we've assigned our
                        // borrow to a new place.
                        if assigned_from_local != target {
                            continue;
                        }

                        // If we assigned our `target` into a new place, then we should
                        // check if it was the return place.
                        debug!(
                            "annotate_argument_and_return_for_borrow: \
                             assigned_from_local={:?} assigned_to={:?}",
                            assigned_from_local, assigned_to
                        );
                        if assigned_to == mir::RETURN_PLACE {
                            // If it was then return the annotated closure if there was one,
                            // else, annotate this function.
                            return annotated_closure.or_else(fallback);
                        }

                        // If we didn't assign into the return place, then we just update
                        // the target.
                        target = assigned_to;
                    }
                }
            }

            // Check the terminator if we didn't find anything in the statements.
            let terminator = &self.body[location.block].terminator();
            debug!(
                "annotate_argument_and_return_for_borrow: target={:?} terminator={:?}",
                target, terminator
            );
            if let TerminatorKind::Call { destination, target: Some(_), args, .. } =
                &terminator.kind
            {
                if let Some(assigned_to) = destination.as_local() {
                    debug!(
                        "annotate_argument_and_return_for_borrow: assigned_to={:?} args={:?}",
                        assigned_to, args
                    );
                    for operand in args {
                        let (Operand::Copy(assigned_from) | Operand::Move(assigned_from)) =
                            &operand.node
                        else {
                            continue;
                        };
                        debug!(
                            "annotate_argument_and_return_for_borrow: assigned_from={:?}",
                            assigned_from,
                        );

                        if let Some(assigned_from_local) = assigned_from.local_or_deref_local() {
                            debug!(
                                "annotate_argument_and_return_for_borrow: assigned_from_local={:?}",
                                assigned_from_local,
                            );

                            if assigned_to == mir::RETURN_PLACE && assigned_from_local == target {
                                return annotated_closure.or_else(fallback);
                            }
                        }
                    }
                }
            }
        }

        // If we haven't found an assignment into the return place, then we need not add
        // any annotations.
        debug!("annotate_argument_and_return_for_borrow: none found");
        None
    }

    /// Annotate the first argument and return type of a function signature if they are
    /// references.
    fn annotate_fn_sig(
        &self,
        did: LocalDefId,
        sig: ty::PolyFnSig<'tcx>,
    ) -> Option<AnnotatedBorrowFnSignature<'tcx>> {
        debug!("annotate_fn_sig: did={:?} sig={:?}", did, sig);
        let is_closure = self.infcx.tcx.is_closure_like(did.to_def_id());
        let fn_hir_id = self.infcx.tcx.local_def_id_to_hir_id(did);
        let fn_decl = self.infcx.tcx.hir_fn_decl_by_hir_id(fn_hir_id)?;

        // We need to work out which arguments to highlight. We do this by looking
        // at the return type, where there are three cases:
        //
        // 1. If there are named arguments, then we should highlight the return type and
        //    highlight any of the arguments that are also references with that lifetime.
        //    If there are no arguments that have the same lifetime as the return type,
        //    then don't highlight anything.
        // 2. The return type is a reference with an anonymous lifetime. If this is
        //    the case, then we can take advantage of (and teach) the lifetime elision
        //    rules.
        //
        //    We know that an error is being reported. So the arguments and return type
        //    must satisfy the elision rules. Therefore, if there is a single argument
        //    then that means the return type and first (and only) argument have the same
        //    lifetime and the borrow isn't meeting that, we can highlight the argument
        //    and return type.
        //
        //    If there are multiple arguments then the first argument must be self (else
        //    it would not satisfy the elision rules), so we can highlight self and the
        //    return type.
        // 3. The return type is not a reference. In this case, we don't highlight
        //    anything.
        let return_ty = sig.output();
        match return_ty.skip_binder().kind() {
            ty::Ref(return_region, _, _) if return_region.has_name() && !is_closure => {
                // This is case 1 from above, return type is a named reference so we need to
                // search for relevant arguments.
                let mut arguments = Vec::new();
                for (index, argument) in sig.inputs().skip_binder().iter().enumerate() {
                    if let ty::Ref(argument_region, _, _) = argument.kind()
                        && argument_region == return_region
                    {
                        // Need to use the `rustc_middle::ty` types to compare against the
                        // `return_region`. Then use the `rustc_hir` type to get only
                        // the lifetime span.
                        match &fn_decl.inputs[index].kind {
                            hir::TyKind::Ref(lifetime, _) => {
                                // With access to the lifetime, we can get
                                // the span of it.
                                arguments.push((*argument, lifetime.ident.span));
                            }
                            // Resolve `self` whose self type is `&T`.
                            hir::TyKind::Path(hir::QPath::Resolved(None, path)) => {
                                if let Res::SelfTyAlias { alias_to, .. } = path.res
                                    && let Some(alias_to) = alias_to.as_local()
                                    && let hir::Impl { self_ty, .. } = self
                                        .infcx
                                        .tcx
                                        .hir_node_by_def_id(alias_to)
                                        .expect_item()
                                        .expect_impl()
                                    && let hir::TyKind::Ref(lifetime, _) = self_ty.kind
                                {
                                    arguments.push((*argument, lifetime.ident.span));
                                }
                            }
                            _ => {
                                // Don't ICE though. It might be a type alias.
                            }
                        }
                    }
                }

                // We need to have arguments. This shouldn't happen, but it's worth checking.
                if arguments.is_empty() {
                    return None;
                }

                // We use a mix of the HIR and the Ty types to get information
                // as the HIR doesn't have full types for closure arguments.
                let return_ty = sig.output().skip_binder();
                let mut return_span = fn_decl.output.span();
                if let hir::FnRetTy::Return(ty) = &fn_decl.output {
                    if let hir::TyKind::Ref(lifetime, _) = ty.kind {
                        return_span = lifetime.ident.span;
                    }
                }

                Some(AnnotatedBorrowFnSignature::NamedFunction {
                    arguments,
                    return_ty,
                    return_span,
                })
            }
            ty::Ref(_, _, _) if is_closure => {
                // This is case 2 from above but only for closures, return type is anonymous
                // reference so we select
                // the first argument.
                let argument_span = fn_decl.inputs.first()?.span;
                let argument_ty = sig.inputs().skip_binder().first()?;

                // Closure arguments are wrapped in a tuple, so we need to get the first
                // from that.
                if let ty::Tuple(elems) = argument_ty.kind() {
                    let &argument_ty = elems.first()?;
                    if let ty::Ref(_, _, _) = argument_ty.kind() {
                        return Some(AnnotatedBorrowFnSignature::Closure {
                            argument_ty,
                            argument_span,
                        });
                    }
                }

                None
            }
            ty::Ref(_, _, _) => {
                // This is also case 2 from above but for functions, return type is still an
                // anonymous reference so we select the first argument.
                let argument_span = fn_decl.inputs.first()?.span;
                let argument_ty = *sig.inputs().skip_binder().first()?;

                let return_span = fn_decl.output.span();
                let return_ty = sig.output().skip_binder();

                // We expect the first argument to be a reference.
                match argument_ty.kind() {
                    ty::Ref(_, _, _) => {}
                    _ => return None,
                }

                Some(AnnotatedBorrowFnSignature::AnonymousFunction {
                    argument_ty,
                    argument_span,
                    return_ty,
                    return_span,
                })
            }
            _ => {
                // This is case 3 from above, return type is not a reference so don't highlight
                // anything.
                None
            }
        }
    }
}

#[derive(Debug)]
enum AnnotatedBorrowFnSignature<'tcx> {
    NamedFunction {
        arguments: Vec<(Ty<'tcx>, Span)>,
        return_ty: Ty<'tcx>,
        return_span: Span,
    },
    AnonymousFunction {
        argument_ty: Ty<'tcx>,
        argument_span: Span,
        return_ty: Ty<'tcx>,
        return_span: Span,
    },
    Closure {
        argument_ty: Ty<'tcx>,
        argument_span: Span,
    },
}

impl<'tcx> AnnotatedBorrowFnSignature<'tcx> {
    /// Annotate the provided diagnostic with information about borrow from the fn signature that
    /// helps explain.
    pub(crate) fn emit(&self, cx: &MirBorrowckCtxt<'_, '_, 'tcx>, diag: &mut Diag<'_>) -> String {
        match self {
            &AnnotatedBorrowFnSignature::Closure { argument_ty, argument_span } => {
                diag.span_label(
                    argument_span,
                    format!("has type `{}`", cx.get_name_for_ty(argument_ty, 0)),
                );

                cx.get_region_name_for_ty(argument_ty, 0)
            }
            &AnnotatedBorrowFnSignature::AnonymousFunction {
                argument_ty,
                argument_span,
                return_ty,
                return_span,
            } => {
                let argument_ty_name = cx.get_name_for_ty(argument_ty, 0);
                diag.span_label(argument_span, format!("has type `{argument_ty_name}`"));

                let return_ty_name = cx.get_name_for_ty(return_ty, 0);
                let types_equal = return_ty_name == argument_ty_name;
                diag.span_label(
                    return_span,
                    format!(
                        "{}has type `{}`",
                        if types_equal { "also " } else { "" },
                        return_ty_name,
                    ),
                );

                diag.note(
                    "argument and return type have the same lifetime due to lifetime elision rules",
                );
                diag.note(
                    "to learn more, visit <https://doc.rust-lang.org/book/ch10-03-\
                     lifetime-syntax.html#lifetime-elision>",
                );

                cx.get_region_name_for_ty(return_ty, 0)
            }
            AnnotatedBorrowFnSignature::NamedFunction { arguments, return_ty, return_span } => {
                // Region of return type and arguments checked to be the same earlier.
                let region_name = cx.get_region_name_for_ty(*return_ty, 0);
                for (_, argument_span) in arguments {
                    diag.span_label(*argument_span, format!("has lifetime `{region_name}`"));
                }

                diag.span_label(*return_span, format!("also has lifetime `{region_name}`",));

                diag.help(format!(
                    "use data from the highlighted arguments which match the `{region_name}` lifetime of \
                     the return type",
                ));

                region_name
            }
        }
    }
}

/// Detect whether one of the provided spans is a statement nested within the top-most visited expr
struct ReferencedStatementsVisitor<'a>(&'a [Span]);

impl<'v> Visitor<'v> for ReferencedStatementsVisitor<'_> {
    type Result = ControlFlow<()>;
    fn visit_stmt(&mut self, s: &'v hir::Stmt<'v>) -> Self::Result {
        match s.kind {
            hir::StmtKind::Semi(expr) if self.0.contains(&expr.span) => ControlFlow::Break(()),
            _ => ControlFlow::Continue(()),
        }
    }
}

/// Look for `break` expressions within any arbitrary expressions. We'll do this to infer
/// whether this is a case where the moved value would affect the exit of a loop, making it
/// unsuitable for a `.clone()` suggestion.
struct BreakFinder {
    found_breaks: Vec<(hir::Destination, Span)>,
    found_continues: Vec<(hir::Destination, Span)>,
}
impl<'hir> Visitor<'hir> for BreakFinder {
    fn visit_expr(&mut self, ex: &'hir hir::Expr<'hir>) {
        match ex.kind {
            hir::ExprKind::Break(destination, _) => {
                self.found_breaks.push((destination, ex.span));
            }
            hir::ExprKind::Continue(destination) => {
                self.found_continues.push((destination, ex.span));
            }
            _ => {}
        }
        hir::intravisit::walk_expr(self, ex);
    }
}

/// Given a set of spans representing statements initializing the relevant binding, visit all the
/// function expressions looking for branching code paths that *do not* initialize the binding.
struct ConditionVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    spans: Vec<Span>,
    name: String,
    errors: Vec<(Span, String)>,
}

impl<'v, 'tcx> Visitor<'v> for ConditionVisitor<'tcx> {
    fn visit_expr(&mut self, ex: &'v hir::Expr<'v>) {
        match ex.kind {
            hir::ExprKind::If(cond, body, None) => {
                // `if` expressions with no `else` that initialize the binding might be missing an
                // `else` arm.
                if ReferencedStatementsVisitor(&self.spans).visit_expr(body).is_break() {
                    self.errors.push((
                        cond.span,
                        format!(
                            "if this `if` condition is `false`, {} is not initialized",
                            self.name,
                        ),
                    ));
                    self.errors.push((
                        ex.span.shrink_to_hi(),
                        format!("an `else` arm might be missing here, initializing {}", self.name),
                    ));
                }
            }
            hir::ExprKind::If(cond, body, Some(other)) => {
                // `if` expressions where the binding is only initialized in one of the two arms
                // might be missing a binding initialization.
                let a = ReferencedStatementsVisitor(&self.spans).visit_expr(body).is_break();
                let b = ReferencedStatementsVisitor(&self.spans).visit_expr(other).is_break();
                match (a, b) {
                    (true, true) | (false, false) => {}
                    (true, false) => {
                        if other.span.is_desugaring(DesugaringKind::WhileLoop) {
                            self.errors.push((
                                cond.span,
                                format!(
                                    "if this condition isn't met and the `while` loop runs 0 \
                                     times, {} is not initialized",
                                    self.name
                                ),
                            ));
                        } else {
                            self.errors.push((
                                body.span.shrink_to_hi().until(other.span),
                                format!(
                                    "if the `if` condition is `false` and this `else` arm is \
                                     executed, {} is not initialized",
                                    self.name
                                ),
                            ));
                        }
                    }
                    (false, true) => {
                        self.errors.push((
                            cond.span,
                            format!(
                                "if this condition is `true`, {} is not initialized",
                                self.name
                            ),
                        ));
                    }
                }
            }
            hir::ExprKind::Match(e, arms, loop_desugar) => {
                // If the binding is initialized in one of the match arms, then the other match
                // arms might be missing an initialization.
                let results: Vec<bool> = arms
                    .iter()
                    .map(|arm| ReferencedStatementsVisitor(&self.spans).visit_arm(arm).is_break())
                    .collect();
                if results.iter().any(|x| *x) && !results.iter().all(|x| *x) {
                    for (arm, seen) in arms.iter().zip(results) {
                        if !seen {
                            if loop_desugar == hir::MatchSource::ForLoopDesugar {
                                self.errors.push((
                                    e.span,
                                    format!(
                                        "if the `for` loop runs 0 times, {} is not initialized",
                                        self.name
                                    ),
                                ));
                            } else if let Some(guard) = &arm.guard {
                                if matches!(
                                    self.tcx.hir_node(arm.body.hir_id),
                                    hir::Node::Expr(hir::Expr { kind: hir::ExprKind::Ret(_), .. })
                                ) {
                                    continue;
                                }
                                self.errors.push((
                                    arm.pat.span.to(guard.span),
                                    format!(
                                        "if this pattern and condition are matched, {} is not \
                                         initialized",
                                        self.name
                                    ),
                                ));
                            } else {
                                if matches!(
                                    self.tcx.hir_node(arm.body.hir_id),
                                    hir::Node::Expr(hir::Expr { kind: hir::ExprKind::Ret(_), .. })
                                ) {
                                    continue;
                                }
                                self.errors.push((
                                    arm.pat.span,
                                    format!(
                                        "if this pattern is matched, {} is not initialized",
                                        self.name
                                    ),
                                ));
                            }
                        }
                    }
                }
            }
            // FIXME: should we also account for binops, particularly `&&` and `||`? `try` should
            // also be accounted for. For now it is fine, as if we don't find *any* relevant
            // branching code paths, we point at the places where the binding *is* initialized for
            // *some* context.
            _ => {}
        }
        walk_expr(self, ex);
    }
}
