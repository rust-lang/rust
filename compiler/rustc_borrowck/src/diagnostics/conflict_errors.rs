use either::Either;
use rustc_const_eval::util::CallKind;
use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{
    struct_span_err, Applicability, Diagnostic, DiagnosticBuilder, ErrorGuaranteed, MultiSpan,
};
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_hir::intravisit::{walk_block, walk_expr, Visitor};
use rustc_hir::{AsyncGeneratorKind, GeneratorKind, LangItem};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::ObligationCause;
use rustc_middle::mir::tcx::PlaceTy;
use rustc_middle::mir::{
    self, AggregateKind, BindingForm, BorrowKind, ClearCrossCrate, ConstraintCategory,
    FakeReadCause, LocalDecl, LocalInfo, LocalKind, Location, Operand, Place, PlaceRef,
    ProjectionElem, Rvalue, Statement, StatementKind, Terminator, TerminatorKind, VarBindingForm,
};
use rustc_middle::ty::{self, suggest_constraining_type_params, PredicateKind, Ty};
use rustc_mir_dataflow::move_paths::{InitKind, MoveOutIndex, MovePathIndex};
use rustc_span::def_id::LocalDefId;
use rustc_span::hygiene::DesugaringKind;
use rustc_span::symbol::{kw, sym};
use rustc_span::{BytePos, Span, Symbol};
use rustc_trait_selection::infer::InferCtxtExt;

use crate::borrow_set::TwoPhaseActivation;
use crate::borrowck_errors;

use crate::diagnostics::conflict_errors::StorageDeadOrDrop::LocalStorageDead;
use crate::diagnostics::find_all_local_uses;
use crate::diagnostics::mutability_errors::mut_borrow_of_mutable_ref;
use crate::{
    borrow_set::BorrowData, diagnostics::Instance, prefixes::IsPrefixOf,
    InitializationRequiringAction, MirBorrowckCtxt, PrefixSet, WriteKind,
};

use super::{
    explain_borrow::{BorrowExplanation, LaterUseKind},
    DescribePlaceOpt, RegionName, RegionNameSource, UseSpans,
};

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

impl<'cx, 'tcx> MirBorrowckCtxt<'cx, 'tcx> {
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
            let root_place = PlaceRef { projection: &[], ..used_place };

            if !self.uninitialized_error_reported.insert(root_place) {
                debug!(
                    "report_use_of_moved_or_uninitialized place: error about {:?} suppressed",
                    root_place
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
                if self.prefixes(*reported_place, PrefixSet::All).any(|p| p == used_place) {
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
                    &if reinits <= 3 {
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
            let mut in_pattern = false;
            let mut seen_spans = FxHashSet::default();

            for move_site in &move_site_vec {
                let move_out = self.move_data.moves[(*move_site).moi];
                let moved_place = &self.move_data.move_paths[move_out.path].place;

                let move_spans = self.move_spans(moved_place.as_ref(), move_out.source);
                let move_span = move_spans.args_or_use();

                let move_msg = if move_spans.for_closure() { " into closure" } else { "" };

                let loop_message = if location == move_out.source || move_site.traversed_back_edge {
                    ", in previous iteration of loop"
                } else {
                    ""
                };

                if location == move_out.source {
                    is_loop_move = true;
                }

                if !seen_spans.contains(&move_span) {
                    if !closure {
                        self.suggest_ref_or_clone(
                            mpi,
                            move_span,
                            &mut err,
                            &mut in_pattern,
                            move_spans,
                        );
                    }

                    self.explain_captures(
                        &mut err,
                        span,
                        move_span,
                        move_spans,
                        *moved_place,
                        partially_str,
                        loop_message,
                        move_msg,
                        is_loop_move,
                        maybe_reinitialized_locations.is_empty(),
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
                    let tables = self.infcx.tcx.typeck(id.expect_local());
                    let hir_id = self.infcx.tcx.hir().local_def_id_to_hir_id(id.expect_local());

                    tables.closure_kind_origins().get(hir_id).is_none()
                }
                _ => true,
            };

            let mpi = self.move_data.moves[move_out_indices[0]].path;
            let place = &self.move_data.move_paths[mpi].place;
            let ty = place.ty(self.body, self.infcx.tcx).ty;

            // If we're in pattern, we do nothing in favor of the previous suggestion (#80913).
            // Same for if we're in a loop, see #101119.
            if is_loop_move & !in_pattern && !matches!(use_spans, UseSpans::ClosureUse { .. }) {
                if let ty::Ref(_, _, hir::Mutability::Mut) = ty.kind() {
                    // We have a `&mut` ref, we need to reborrow on each iteration (#62112).
                    err.span_suggestion_verbose(
                        span.shrink_to_lo(),
                        &format!(
                            "consider creating a fresh reborrow of {} here",
                            self.describe_place(moved_place)
                                .map(|n| format!("`{n}`"))
                                .unwrap_or_else(|| "the mutable reference".to_string()),
                        ),
                        "&mut *",
                        Applicability::MachineApplicable,
                    );
                }
            }

            let opt_name = self.describe_place_with_options(
                place.as_ref(),
                DescribePlaceOpt { including_downcast: true, including_tuple_field: true },
            );
            let note_msg = match opt_name {
                Some(name) => format!("`{name}`"),
                None => "value".to_owned(),
            };
            if self.suggest_borrow_fn_like(&mut err, ty, &move_site_vec, &note_msg) {
                // Suppress the next suggestion since we don't want to put more bounds onto
                // something that already has `Fn`-like bounds (or is a closure), so we can't
                // restrict anyways.
            } else {
                self.suggest_adding_copy_bounds(&mut err, ty, span);
            }

            if needs_note {
                let span = if let Some(local) = place.as_local() {
                    Some(self.body.local_decls[local].source_info.span)
                } else {
                    None
                };
                self.note_type_does_not_implement_copy(&mut err, &note_msg, ty, span, partial_str);
            }

            if let UseSpans::FnSelfUse {
                kind: CallKind::DerefCoercion { deref_target, deref_target_ty, .. },
                ..
            } = use_spans
            {
                err.note(&format!(
                    "{} occurs due to deref coercion to `{deref_target_ty}`",
                    desired_action.as_noun(),
                ));

                // Check first whether the source is accessible (issue #87060)
                if self.infcx.tcx.sess.source_map().is_span_accessible(deref_target) {
                    err.span_note(deref_target, "deref defined here");
                }
            }

            self.buffer_move_error(move_out_indices, (used_place, err));
        }
    }

    fn suggest_ref_or_clone(
        &mut self,
        mpi: MovePathIndex,
        move_span: Span,
        err: &mut DiagnosticBuilder<'_, ErrorGuaranteed>,
        in_pattern: &mut bool,
        move_spans: UseSpans<'_>,
    ) {
        struct ExpressionFinder<'hir> {
            expr_span: Span,
            expr: Option<&'hir hir::Expr<'hir>>,
            pat: Option<&'hir hir::Pat<'hir>>,
            parent_pat: Option<&'hir hir::Pat<'hir>>,
        }
        impl<'hir> Visitor<'hir> for ExpressionFinder<'hir> {
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
                if let hir::PatKind::Binding(hir::BindingAnnotation::NONE, _, i, sub) = p.kind {
                    if i.span == self.expr_span || p.span == self.expr_span {
                        self.pat = Some(p);
                    }
                    // Check if we are in a situation of `ident @ ident` where we want to suggest
                    // `ref ident @ ref ident` or `ref ident @ Struct { ref ident }`.
                    if let Some(subpat) = sub && self.pat.is_none() {
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
        let hir = self.infcx.tcx.hir();
        if let Some(hir::Node::Item(hir::Item {
            kind: hir::ItemKind::Fn(_, _, body_id),
            ..
        })) = hir.find(self.mir_hir_id())
            && let Some(hir::Node::Expr(expr)) = hir.find(body_id.hir_id)
        {
            let place = &self.move_data.move_paths[mpi].place;
            let span = place.as_local()
                .map(|local| self.body.local_decls[local].source_info.span);
            let mut finder = ExpressionFinder {
                expr_span: move_span,
                expr: None,
                pat: None,
                parent_pat: None,
            };
            finder.visit_expr(expr);
            if let Some(span) = span && let Some(expr) = finder.expr {
                for (_, expr) in hir.parent_iter(expr.hir_id) {
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
                let hir_id = hir.parent_id(expr.hir_id);
                if let Some(parent) = hir.find(hir_id) {
                    let (def_id, args, offset) = if let hir::Node::Expr(parent_expr) = parent
                        && let hir::ExprKind::MethodCall(_, _, args, _) = parent_expr.kind
                        && let Some(def_id) = typeck.type_dependent_def_id(parent_expr.hir_id)
                    {
                        (def_id.as_local(), args, 1)
                    } else if let hir::Node::Expr(parent_expr) = parent
                        && let hir::ExprKind::Call(call, args) = parent_expr.kind
                        && let ty::FnDef(def_id, _) = typeck.node_type(call.hir_id).kind()
                    {
                        (def_id.as_local(), args, 0)
                    } else {
                        (None, &[][..], 0)
                    };
                    if let Some(def_id) = def_id
                        && let Some(node) = hir.find(hir.local_def_id_to_hir_id(def_id))
                        && let Some(fn_sig) = node.fn_sig()
                        && let Some(ident) = node.ident()
                        && let Some(pos) = args.iter().position(|arg| arg.hir_id == expr.hir_id)
                        && let Some(arg) = fn_sig.decl.inputs.get(pos + offset)
                    {
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
                        span.push_span_label(
                            ident.span,
                            format!("in this {descr}"),
                        );
                        err.span_note(
                            span,
                            format!(
                                "consider changing this parameter type in {descr} `{ident}` to \
                                 borrow instead if owning the value isn't necessary",
                            ),
                        );
                    }
                    let place = &self.move_data.move_paths[mpi].place;
                    let ty = place.ty(self.body, self.infcx.tcx).ty;
                    if let hir::Node::Expr(parent_expr) = parent
                        && let hir::ExprKind::Call(call_expr, _) = parent_expr.kind
                        && let hir::ExprKind::Path(
                            hir::QPath::LangItem(LangItem::IntoIterIntoIter, _, _)
                        ) = call_expr.kind
                    {
                        // Do not suggest `.clone()` in a `for` loop, we already suggest borrowing.
                    } else if let UseSpans::FnSelfUse {
                        kind: CallKind::Normal { .. },
                        ..
                    } = move_spans {
                        // We already suggest cloning for these cases in `explain_captures`.
                    } else {
                        self.suggest_cloning(err, ty, move_span);
                    }
                }
            }
            if let Some(pat) = finder.pat {
                *in_pattern = true;
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

    fn report_use_of_uninitialized(
        &self,
        mpi: MovePathIndex,
        used_place: PlaceRef<'tcx>,
        moved_place: PlaceRef<'tcx>,
        desired_action: InitializationRequiringAction,
        span: Span,
        use_spans: UseSpans<'tcx>,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        // We need all statements in the body where the binding was assigned to to later find all
        // the branching code paths where the binding *wasn't* assigned to.
        let inits = &self.move_data.init_path_map[mpi];
        let move_path = &self.move_data.move_paths[mpi];
        let decl_span = self.body.local_decls[move_path.place.local].source_info.span;
        let mut spans = vec![];
        for init_idx in inits {
            let init = &self.move_data.inits[*init_idx];
            let span = init.span(&self.body);
            if !span.is_dummy() {
                spans.push(span);
            }
        }

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
        let map = self.infcx.tcx.hir();
        let body_id = map.body_owned_by(self.mir_def_id());
        let body = map.body(body_id);

        let mut visitor = ConditionVisitor { spans: &spans, name: &name, errors: vec![] };
        visitor.visit_body(&body);

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
        let mut err =
            struct_span_err!(self, span, E0381, "{used} binding {desc}{isnt_initialized}");
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
                err.span_label(sp, &label);
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
                    if let hir::StmtKind::Local(hir::Local {
                            span, ty, init: None, ..
                        }) = &ex.kind && span.contains(self.decl_span) {
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
        err: &mut Diagnostic,
        moved_place: PlaceRef<'tcx>,
        sugg_span: Span,
    ) {
        let ty = moved_place.ty(self.body, self.infcx.tcx).ty;
        debug!("ty: {:?}, kind: {:?}", ty, ty.kind());

        let tcx = self.infcx.tcx;
        let implements_default = |ty, param_env| {
            let Some(default_trait) = tcx.get_diagnostic_item(sym::Default) else {
                return false;
            };
            // Regions are already solved, so we must use a fresh InferCtxt,
            // but the type has region variables, so erase those.
            tcx.infer_ctxt()
                .build()
                .type_implements_trait(default_trait, [tcx.erase_regions(ty)], param_env)
                .must_apply_modulo_regions()
        };

        let assign_value = match ty.kind() {
            ty::Bool => "false",
            ty::Float(_) => "0.0",
            ty::Int(_) | ty::Uint(_) => "0",
            ty::Never | ty::Error(_) => "",
            ty::Adt(def, _) if Some(def.did()) == tcx.get_diagnostic_item(sym::Vec) => "vec![]",
            ty::Adt(_, _) if implements_default(ty, self.param_env) => "Default::default()",
            _ => "todo!()",
        };

        if !assign_value.is_empty() {
            err.span_suggestion_verbose(
                sugg_span.shrink_to_hi(),
                "consider assigning a value",
                format!(" = {}", assign_value),
                Applicability::MaybeIncorrect,
            );
        }
    }

    fn suggest_borrow_fn_like(
        &self,
        err: &mut Diagnostic,
        ty: Ty<'tcx>,
        move_sites: &[MoveSite],
        value_name: &str,
    ) -> bool {
        let tcx = self.infcx.tcx;

        // Find out if the predicates show that the type is a Fn or FnMut
        let find_fn_kind_from_did = |(pred, _): (ty::Predicate<'tcx>, _)| {
            if let ty::PredicateKind::Clause(ty::Clause::Trait(pred)) = pred.kind().skip_binder()
                && pred.self_ty() == ty
            {
                if Some(pred.def_id()) == tcx.lang_items().fn_trait() {
                    return Some(hir::Mutability::Not);
                } else if Some(pred.def_id()) == tcx.lang_items().fn_mut_trait() {
                    return Some(hir::Mutability::Mut);
                }
            }
            None
        };

        // If the type is opaque/param/closure, and it is Fn or FnMut, let's suggest (mutably)
        // borrowing the type, since `&mut F: FnMut` iff `F: FnMut` and similarly for `Fn`.
        // These types seem reasonably opaque enough that they could be substituted with their
        // borrowed variants in a function body when we see a move error.
        let borrow_level = match *ty.kind() {
            ty::Param(_) => tcx
                .explicit_predicates_of(self.mir_def_id().to_def_id())
                .predicates
                .iter()
                .copied()
                .find_map(find_fn_kind_from_did),
            ty::Alias(ty::Opaque, ty::AliasTy { def_id, substs, .. }) => tcx
                .bound_explicit_item_bounds(def_id)
                .subst_iter_copied(tcx, substs)
                .find_map(find_fn_kind_from_did),
            ty::Closure(_, substs) => match substs.as_closure().kind() {
                ty::ClosureKind::Fn => Some(hir::Mutability::Not),
                ty::ClosureKind::FnMut => Some(hir::Mutability::Mut),
                _ => None,
            },
            _ => None,
        };

        let Some(borrow_level) = borrow_level else { return false; };
        let sugg = move_sites
            .iter()
            .map(|move_site| {
                let move_out = self.move_data.moves[(*move_site).moi];
                let moved_place = &self.move_data.move_paths[move_out.path].place;
                let move_spans = self.move_spans(moved_place.as_ref(), move_out.source);
                let move_span = move_spans.args_or_use();
                let suggestion = borrow_level.ref_prefix_str().to_owned();
                (move_span.shrink_to_lo(), suggestion)
            })
            .collect();
        err.multipart_suggestion_verbose(
            format!("consider {}borrowing {value_name}", borrow_level.mutably_str()),
            sugg,
            Applicability::MaybeIncorrect,
        );
        true
    }

    fn suggest_cloning(&self, err: &mut Diagnostic, ty: Ty<'tcx>, span: Span) {
        let tcx = self.infcx.tcx;
        // Try to find predicates on *generic params* that would allow copying `ty`
        let infcx = tcx.infer_ctxt().build();

        if let Some(clone_trait_def) = tcx.lang_items().clone_trait()
            && infcx
                .type_implements_trait(
                    clone_trait_def,
                    [tcx.erase_regions(ty)],
                    self.param_env,
                )
                .must_apply_modulo_regions()
        {
            err.span_suggestion_verbose(
                span.shrink_to_hi(),
                "consider cloning the value if the performance cost is acceptable",
                ".clone()",
                Applicability::MachineApplicable,
            );
        }
    }

    fn suggest_adding_copy_bounds(&self, err: &mut Diagnostic, ty: Ty<'tcx>, span: Span) {
        let tcx = self.infcx.tcx;
        let generics = tcx.generics_of(self.mir_def_id());

        let Some(hir_generics) = tcx
            .typeck_root_def_id(self.mir_def_id().to_def_id())
            .as_local()
            .and_then(|def_id| tcx.hir().get_generics(def_id))
        else { return; };
        // Try to find predicates on *generic params* that would allow copying `ty`
        let infcx = tcx.infer_ctxt().build();
        let copy_did = infcx.tcx.require_lang_item(LangItem::Copy, Some(span));
        let cause = ObligationCause::new(
            span,
            self.mir_def_id(),
            rustc_infer::traits::ObligationCauseCode::MiscObligation,
        );
        let errors = rustc_trait_selection::traits::fully_solve_bound(
            &infcx,
            cause,
            self.param_env,
            // Erase any region vids from the type, which may not be resolved
            infcx.tcx.erase_regions(ty),
            copy_did,
        );

        // Only emit suggestion if all required predicates are on generic
        let predicates: Result<Vec<_>, _> = errors
            .into_iter()
            .map(|err| match err.obligation.predicate.kind().skip_binder() {
                PredicateKind::Clause(ty::Clause::Trait(predicate)) => {
                    match predicate.self_ty().kind() {
                        ty::Param(param_ty) => Ok((
                            generics.type_param(param_ty, tcx),
                            predicate.trait_ref.print_only_trait_path().to_string(),
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
                predicates
                    .iter()
                    .map(|(param, constraint)| (param.name.as_str(), &**constraint, None)),
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

        borrow_spans.var_path_only_subdiag(&mut err, crate::InitializationRequiringAction::Borrow);

        move_spans.var_span_label(
            &mut err,
            format!("move occurs due to use{}", move_spans.describe()),
            "moved",
        );

        self.explain_why_borrow_contains_point(location, borrow, None)
            .add_explanation_to_diagnostic(
                self.infcx.tcx,
                &self.body,
                &self.local_names,
                &mut err,
                "",
                Some(borrow_span),
                None,
            );
        self.buffer_error(err);
    }

    pub(crate) fn report_use_while_mutably_borrowed(
        &mut self,
        location: Location,
        (place, _span): (Place<'tcx>, Span),
        borrow: &BorrowData<'tcx>,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        let borrow_spans = self.retrieve_borrow_spans(borrow);
        let borrow_span = borrow_spans.args_or_use();

        // Conflicting borrows are reported separately, so only check for move
        // captures.
        let use_spans = self.move_spans(place.as_ref(), location);
        let span = use_spans.var_or_use();

        // If the attempted use is in a closure then we do not care about the path span of the place we are currently trying to use
        // we call `var_span_label` on `borrow_spans` to annotate if the existing borrow was in a closure
        let mut err = self.cannot_use_when_mutably_borrowed(
            span,
            &self.describe_any_place(place.as_ref()),
            borrow_span,
            &self.describe_any_place(borrow.borrowed_place.as_ref()),
        );
        borrow_spans.var_subdiag(&mut err, Some(borrow.kind), |kind, var_span| {
            use crate::session_diagnostics::CaptureVarCause::*;
            let place = &borrow.borrowed_place;
            let desc_place = self.describe_any_place(place.as_ref());
            match kind {
                Some(_) => BorrowUsePlaceGenerator { place: desc_place, var_span },
                None => BorrowUsePlaceClosure { place: desc_place, var_span },
            }
        });

        self.explain_why_borrow_contains_point(location, borrow, None)
            .add_explanation_to_diagnostic(
                self.infcx.tcx,
                &self.body,
                &self.local_names,
                &mut err,
                "",
                None,
                None,
            );
        err
    }

    pub(crate) fn report_conflicting_borrow(
        &mut self,
        location: Location,
        (place, span): (Place<'tcx>, Span),
        gen_borrow_kind: BorrowKind,
        issued_borrow: &BorrowData<'tcx>,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        let issued_spans = self.retrieve_borrow_spans(issued_borrow);
        let issued_span = issued_spans.args_or_use();

        let borrow_spans = self.borrow_spans(span, location);
        let span = borrow_spans.args_or_use();

        let container_name = if issued_spans.for_generator() || borrow_spans.for_generator() {
            "generator"
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
            (BorrowKind::Shared, BorrowKind::Mut { .. }) => {
                first_borrow_desc = "mutable ";
                self.cannot_reborrow_already_borrowed(
                    span,
                    &desc_place,
                    &msg_place,
                    "immutable",
                    issued_span,
                    "it",
                    "mutable",
                    &msg_borrow,
                    None,
                )
            }
            (BorrowKind::Mut { .. }, BorrowKind::Shared) => {
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
                self.suggest_binding_for_closure_capture_self(
                    &mut err,
                    issued_borrow.borrowed_place,
                    &issued_spans,
                );
                err
            }

            (BorrowKind::Mut { .. }, BorrowKind::Mut { .. }) => {
                first_borrow_desc = "first ";
                let mut err = self.cannot_mutably_borrow_multiply(
                    span,
                    &desc_place,
                    &msg_place,
                    issued_span,
                    &msg_borrow,
                    None,
                );
                self.suggest_split_at_mut_if_applicable(
                    &mut err,
                    place,
                    issued_borrow.borrowed_place,
                );
                err
            }

            (BorrowKind::Unique, BorrowKind::Unique) => {
                first_borrow_desc = "first ";
                self.cannot_uniquely_borrow_by_two_closures(span, &desc_place, issued_span, None)
            }

            (BorrowKind::Mut { .. } | BorrowKind::Unique, BorrowKind::Shallow) => {
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
                    borrow_spans.var_span_label(
                        &mut err,
                        format!(
                            "borrow occurs due to use of {}{}",
                            desc_place,
                            borrow_spans.describe(),
                        ),
                        "immutable",
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

            (BorrowKind::Unique, _) => {
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

            (BorrowKind::Shared, BorrowKind::Unique) => {
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

            (BorrowKind::Mut { .. }, BorrowKind::Unique) => {
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

            (BorrowKind::Shared, BorrowKind::Shared | BorrowKind::Shallow)
            | (
                BorrowKind::Shallow,
                BorrowKind::Mut { .. }
                | BorrowKind::Unique
                | BorrowKind::Shared
                | BorrowKind::Shallow,
            ) => unreachable!(),
        };

        if issued_spans == borrow_spans {
            borrow_spans.var_span_label(
                &mut err,
                format!("borrows occur due to use of {}{}", desc_place, borrow_spans.describe(),),
                gen_borrow_kind.describe_mutability(),
            );
        } else {
            let borrow_place = &issued_borrow.borrowed_place;
            let borrow_place_desc = self.describe_any_place(borrow_place.as_ref());
            issued_spans.var_span_label(
                &mut err,
                format!(
                    "first borrow occurs due to use of {}{}",
                    borrow_place_desc,
                    issued_spans.describe(),
                ),
                issued_borrow.kind.describe_mutability(),
            );

            borrow_spans.var_span_label(
                &mut err,
                format!(
                    "second borrow occurs due to use of {}{}",
                    desc_place,
                    borrow_spans.describe(),
                ),
                gen_borrow_kind.describe_mutability(),
            );
        }

        if union_type_name != "" {
            err.note(&format!(
                "{} is a field of the union `{}`, so it overlaps the field {}",
                msg_place, union_type_name, msg_borrow,
            ));
        }

        explanation.add_explanation_to_diagnostic(
            self.infcx.tcx,
            &self.body,
            &self.local_names,
            &mut err,
            first_borrow_desc,
            None,
            Some((issued_span, span)),
        );

        self.suggest_using_local_if_applicable(&mut err, location, issued_borrow, explanation);

        err
    }

    #[instrument(level = "debug", skip(self, err))]
    fn suggest_using_local_if_applicable(
        &self,
        err: &mut Diagnostic,
        location: Location,
        issued_borrow: &BorrowData<'tcx>,
        explanation: BorrowExplanation<'tcx>,
    ) {
        let used_in_call = matches!(
            explanation,
            BorrowExplanation::UsedLater(LaterUseKind::Call | LaterUseKind::Other, _call_span, _)
        );
        if !used_in_call {
            debug!("not later used in call");
            return;
        }

        let use_span =
            if let BorrowExplanation::UsedLater(LaterUseKind::Other, use_span, _) = explanation {
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
        let Some((inner_call_loc, inner_call_term)) = inner_param_uses.into_iter().find_map(|loc| {
            let Either::Right(term) = self.body.stmt_at(loc) else {
                debug!("{:?} is a statement, so it can't be a call", loc);
                return None;
            };
            let TerminatorKind::Call { args, .. } = &term.kind else {
                debug!("not a call: {:?}", term);
                return None;
            };
            debug!("checking call args for uses of inner_param: {:?}", args);
            args.contains(&Operand::Move(inner_param)).then_some((loc, term))
        }) else {
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
            &format!(
                "try adding a local storing this{}...",
                if use_span.is_some() { "" } else { " argument" }
            ),
        );
        err.span_help(
            outer_call_span,
            &format!(
                "...and then using that local {}",
                if use_span.is_some() { "here" } else { "as the argument to this call" }
            ),
        );
    }

    fn suggest_split_at_mut_if_applicable(
        &self,
        err: &mut Diagnostic,
        place: Place<'tcx>,
        borrowed_place: Place<'tcx>,
    ) {
        if let ([ProjectionElem::Index(_)], [ProjectionElem::Index(_)]) =
            (&place.projection[..], &borrowed_place.projection[..])
        {
            err.help(
                "consider using `.split_at_mut(position)` or similar method to obtain \
                     two mutable non-overlapping sub-slices",
            );
        }
    }

    fn suggest_binding_for_closure_capture_self(
        &self,
        err: &mut Diagnostic,
        borrowed_place: Place<'tcx>,
        issued_spans: &UseSpans<'tcx>,
    ) {
        let UseSpans::ClosureUse { capture_kind_span, .. } = issued_spans else { return };
        let hir = self.infcx.tcx.hir();

        // check whether the borrowed place is capturing `self` by mut reference
        let local = borrowed_place.local;
        let Some(_) = self
            .body
            .local_decls
            .get(local)
            .map(|l| mut_borrow_of_mutable_ref(l, self.local_names[local])) else { return };

        struct ExpressionFinder<'hir> {
            capture_span: Span,
            closure_change_spans: Vec<Span>,
            closure_arg_span: Option<Span>,
            in_closure: bool,
            suggest_arg: String,
            hir: rustc_middle::hir::map::Map<'hir>,
            closure_local_id: Option<hir::HirId>,
            closure_call_changes: Vec<(Span, String)>,
        }
        impl<'hir> Visitor<'hir> for ExpressionFinder<'hir> {
            fn visit_expr(&mut self, e: &'hir hir::Expr<'hir>) {
                if e.span.contains(self.capture_span) {
                    if let hir::ExprKind::Closure(&hir::Closure {
                            movability: None,
                            body,
                            fn_arg_span,
                            fn_decl: hir::FnDecl{ inputs, .. },
                            ..
                        }) = e.kind &&
                        let Some(hir::Node::Expr(body )) = self.hir.find(body.hir_id) {
                            self.suggest_arg = "this: &Self".to_string();
                            if inputs.len() > 0 {
                                self.suggest_arg.push_str(", ");
                            }
                            self.in_closure = true;
                            self.closure_arg_span = fn_arg_span;
                            self.visit_expr(body);
                            self.in_closure = false;
                    }
                }
                if let hir::Expr { kind: hir::ExprKind::Path(path), .. } = e {
                    if let hir::QPath::Resolved(_, hir::Path { segments: [seg], ..}) = path &&
                        seg.ident.name == kw::SelfLower && self.in_closure {
                            self.closure_change_spans.push(e.span);
                    }
                }
                hir::intravisit::walk_expr(self, e);
            }

            fn visit_local(&mut self, local: &'hir hir::Local<'hir>) {
                if let hir::Pat { kind: hir::PatKind::Binding(_, hir_id, _ident, _), .. } = local.pat &&
                    let Some(init) = local.init
                {
                    if let hir::Expr { kind: hir::ExprKind::Closure(&hir::Closure {
                            movability: None,
                            ..
                        }), .. } = init &&
                        init.span.contains(self.capture_span) {
                            self.closure_local_id = Some(*hir_id);
                    }
                }
                hir::intravisit::walk_local(self, local);
            }

            fn visit_stmt(&mut self, s: &'hir hir::Stmt<'hir>) {
                if let hir::StmtKind::Semi(e) = s.kind &&
                    let hir::ExprKind::Call(hir::Expr { kind: hir::ExprKind::Path(path), ..}, args) = e.kind &&
                    let hir::QPath::Resolved(_, hir::Path { segments: [seg], ..}) = path &&
                    let Res::Local(hir_id) = seg.res &&
                        Some(hir_id) == self.closure_local_id {
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

        if let Some(hir::Node::ImplItem(
                    hir::ImplItem { kind: hir::ImplItemKind::Fn(_fn_sig, body_id), .. }
                )) = hir.find(self.mir_hir_id()) &&
            let Some(hir::Node::Expr(expr)) = hir.find(body_id.hir_id) {
            let mut finder = ExpressionFinder {
                capture_span: *capture_kind_span,
                closure_change_spans: vec![],
                closure_arg_span: None,
                in_closure: false,
                suggest_arg: String::new(),
                closure_local_id: None,
                closure_call_changes: vec![],
                hir,
            };
            finder.visit_expr(expr);

            if finder.closure_change_spans.is_empty() || finder.closure_call_changes.is_empty() {
                return;
            }

            let mut sugg = vec![];
            let sm = self.infcx.tcx.sess.source_map();

            if let Some(span) = finder.closure_arg_span {
                sugg.push((sm.next_point(span.shrink_to_lo()).shrink_to_hi(), finder.suggest_arg));
            }
            for span in finder.closure_change_spans {
                sugg.push((span, "this".to_string()));
            }

            for (span, suggest) in finder.closure_call_changes {
                sugg.push((span, suggest));
            }

            err.multipart_suggestion_verbose(
                "try explicitly pass `&Self` into the Closure as an argument",
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
    pub(crate) fn describe_place_for_conflicting_borrow(
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

    /// Reports StorageDeadOrDrop of `place` conflicts with `borrow`.
    ///
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
        let root_place =
            self.prefixes(borrow.borrowed_place.as_ref(), PrefixSet::All).last().unwrap();

        let borrow_spans = self.retrieve_borrow_spans(borrow);
        let borrow_span = borrow_spans.var_or_use_path_span();

        assert!(root_place.projection.is_empty());
        let proper_span = self.body.local_decls[root_place.local].source_info.span;

        let root_place_projection = self.infcx.tcx.intern_place_elems(root_place.projection);

        if self.access_place_error_reported.contains(&(
            Place { local: root_place.local, projection: root_place_projection },
            borrow_span,
        )) {
            debug!(
                "suppressing access_place error when borrow doesn't live long enough for {:?}",
                borrow_span
            );
            return;
        }

        self.access_place_error_reported.insert((
            Place { local: root_place.local, projection: root_place_projection },
            borrow_span,
        ));

        let borrowed_local = borrow.borrowed_place.local;
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
        let explanation = self.explain_why_borrow_contains_point(location, &borrow, kind_place);

        debug!(?place_desc, ?explanation);

        let err = match (place_desc, explanation) {
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
                BorrowExplanation::UsedLater(LaterUseKind::ClosureCapture, var_or_use_span, _),
            ) => self.report_escaping_closure_capture(
                borrow_spans,
                borrow_span,
                &RegionName {
                    name: self.synthesize_region_name(),
                    source: RegionNameSource::Static,
                },
                ConstraintCategory::CallArgument(None),
                var_or_use_span,
                &format!("`{}`", name),
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
            ) if borrow_spans.for_generator() | borrow_spans.for_closure() => self
                .report_escaping_closure_capture(
                    borrow_spans,
                    borrow_span,
                    region_name,
                    category,
                    span,
                    &format!("`{}`", name),
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
                &borrow,
                drop_span,
                borrow_spans,
                explanation,
            ),
            (None, explanation) => self.report_temporary_value_does_not_live_long_enough(
                location,
                &borrow,
                drop_span,
                borrow_spans,
                proper_span,
                explanation,
            ),
        };

        self.buffer_error(err);
    }

    fn report_local_value_does_not_live_long_enough(
        &mut self,
        location: Location,
        name: &str,
        borrow: &BorrowData<'tcx>,
        drop_span: Span,
        borrow_spans: UseSpans<'tcx>,
        explanation: BorrowExplanation<'tcx>,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
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
            if let Some(diag) = self.try_report_cannot_return_reference_to_local(
                borrow,
                borrow_span,
                span,
                category,
                opt_place_desc.as_ref(),
            ) {
                return diag;
            }
        }

        let mut err = self.path_does_not_live_long_enough(borrow_span, &format!("`{}`", name));

        if let Some(annotation) = self.annotate_argument_and_return_for_borrow(borrow) {
            let region_name = annotation.emit(self, &mut err);

            err.span_label(
                borrow_span,
                format!("`{}` would have to be valid for `{}`...", name, region_name),
            );

            let fn_hir_id = self.mir_hir_id();
            err.span_label(
                drop_span,
                format!(
                    "...but `{}` will be dropped here, when the {} returns",
                    name,
                    self.infcx
                        .tcx
                        .hir()
                        .opt_name(fn_hir_id)
                        .map(|name| format!("function `{}`", name))
                        .unwrap_or_else(|| {
                            match &self
                                .infcx
                                .tcx
                                .typeck(self.mir_def_id())
                                .node_type(fn_hir_id)
                                .kind()
                            {
                                ty::Closure(..) => "enclosing closure",
                                ty::Generator(..) => "enclosing generator",
                                kind => bug!("expected closure or generator, found {:?}", kind),
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
                explanation.add_explanation_to_diagnostic(
                    self.infcx.tcx,
                    &self.body,
                    &self.local_names,
                    &mut err,
                    "",
                    None,
                    None,
                );
            }
        } else {
            err.span_label(borrow_span, "borrowed value does not live long enough");
            err.span_label(drop_span, format!("`{}` dropped here while still borrowed", name));

            let within = if borrow_spans.for_generator() { " by generator" } else { "" };

            borrow_spans.args_span_label(&mut err, format!("value captured here{}", within));

            explanation.add_explanation_to_diagnostic(
                self.infcx.tcx,
                &self.body,
                &self.local_names,
                &mut err,
                "",
                Some(borrow_span),
                None,
            );
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
            Some(name) => format!("`{}`", name),
            None => String::from("temporary value"),
        };

        let label = match self.describe_place(borrow.borrowed_place.as_ref()) {
            Some(borrowed) => format!(
                "here, drop of {D} needs exclusive access to `{B}`, \
                 because the type `{T}` implements the `Drop` trait",
                D = what_was_dropped,
                T = dropped_ty,
                B = borrowed
            ),
            None => format!(
                "here is drop of {D}; whose type `{T}` implements the `Drop` trait",
                D = what_was_dropped,
                T = dropped_ty
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

        explanation.add_explanation_to_diagnostic(
            self.infcx.tcx,
            &self.body,
            &self.local_names,
            &mut err,
            "",
            None,
            None,
        );

        self.buffer_error(err);
    }

    fn report_thread_local_value_does_not_live_long_enough(
        &mut self,
        drop_span: Span,
        borrow_span: Span,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        debug!(
            "report_thread_local_value_does_not_live_long_enough(\
             {:?}, {:?}\
             )",
            drop_span, borrow_span
        );

        let mut err = self.thread_local_value_does_not_live_long_enough(borrow_span);

        err.span_label(
            borrow_span,
            "thread-local variables cannot be borrowed beyond the end of the function",
        );
        err.span_label(drop_span, "end of enclosing function is here");

        err
    }

    #[instrument(level = "debug", skip(self))]
    fn report_temporary_value_does_not_live_long_enough(
        &mut self,
        location: Location,
        borrow: &BorrowData<'tcx>,
        drop_span: Span,
        borrow_spans: UseSpans<'tcx>,
        proper_span: Span,
        explanation: BorrowExplanation<'tcx>,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        if let BorrowExplanation::MustBeValidFor { category, span, from_closure: false, .. } =
            explanation
        {
            if let Some(diag) = self.try_report_cannot_return_reference_to_local(
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
                struct NestedStatementVisitor {
                    span: Span,
                    current: usize,
                    found: usize,
                }

                impl<'tcx> Visitor<'tcx> for NestedStatementVisitor {
                    fn visit_block(&mut self, block: &hir::Block<'tcx>) {
                        self.current += 1;
                        walk_block(self, block);
                        self.current -= 1;
                    }
                    fn visit_expr(&mut self, expr: &hir::Expr<'tcx>) {
                        if self.span == expr.span {
                            self.found = self.current;
                        }
                        walk_expr(self, expr);
                    }
                }
                let source_info = self.body.source_info(location);
                if let Some(scope) = self.body.source_scopes.get(source_info.scope)
                    && let ClearCrossCrate::Set(scope_data) = &scope.local_data
                    && let Some(node) = self.infcx.tcx.hir().find(scope_data.lint_root)
                    && let Some(id) = node.body_id()
                    && let hir::ExprKind::Block(block, _) = self.infcx.tcx.hir().body(id).value.kind
                {
                    for stmt in block.stmts {
                        let mut visitor = NestedStatementVisitor {
                            span: proper_span,
                            current: 0,
                            found: 0,
                        };
                        visitor.visit_stmt(stmt);
                        if visitor.found == 0
                            && stmt.span.contains(proper_span)
                            && let Some(p) = sm.span_to_margin(stmt.span)
                            && let Ok(s) = sm.span_to_snippet(proper_span)
                        {
                            let addition = format!("let binding = {};\n{}", s, " ".repeat(p));
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
        explanation.add_explanation_to_diagnostic(
            self.infcx.tcx,
            &self.body,
            &self.local_names,
            &mut err,
            "",
            None,
            None,
        );

        let within = if borrow_spans.for_generator() { " by generator" } else { "" };

        borrow_spans.args_span_label(&mut err, format!("value captured here{}", within));

        err
    }

    fn try_report_cannot_return_reference_to_local(
        &self,
        borrow: &BorrowData<'tcx>,
        borrow_span: Span,
        return_span: Span,
        category: ConstraintCategory<'tcx>,
        opt_place_desc: Option<&String>,
    ) -> Option<DiagnosticBuilder<'cx, ErrorGuaranteed>> {
        let return_kind = match category {
            ConstraintCategory::Return(_) => "return",
            ConstraintCategory::Yield => "yield",
            _ => return None,
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
                    LocalKind::ReturnPointer | LocalKind::Temp => {
                        bug!("temporary or return pointer with a name")
                    }
                    LocalKind::Var => "local variable ",
                    LocalKind::Arg
                        if !self.upvars.is_empty() && local == ty::CAPTURE_STRUCT_LOCAL =>
                    {
                        "variable captured by `move` "
                    }
                    LocalKind::Arg => "function parameter ",
                }
            } else {
                "local data "
            };
            (
                format!("{}`{}`", local_kind, place_desc),
                format!("`{}` is borrowed here", place_desc),
            )
        } else {
            let root_place =
                self.prefixes(borrow.borrowed_place.as_ref(), PrefixSet::All).last().unwrap();
            let local = root_place.local;
            match self.body.local_kind(local) {
                LocalKind::ReturnPointer | LocalKind::Temp => {
                    ("temporary value".to_string(), "temporary value created here".to_string())
                }
                LocalKind::Arg => (
                    "function parameter".to_string(),
                    "function parameter borrowed here".to_string(),
                ),
                LocalKind::Var => {
                    ("local binding".to_string(), "local binding introduced here".to_string())
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
            let return_ty = tcx.erase_regions(return_ty);

            // to avoid panics
            if let Some(iter_trait) = tcx.get_diagnostic_item(sym::Iterator)
                && self
                    .infcx
                    .type_implements_trait(iter_trait, [return_ty], self.param_env)
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

        Some(err)
    }

    #[instrument(level = "debug", skip(self))]
    fn report_escaping_closure_capture(
        &mut self,
        use_span: UseSpans<'tcx>,
        var_span: Span,
        fr_name: &RegionName,
        category: ConstraintCategory<'tcx>,
        constraint_span: Span,
        captured_var: &str,
        scope: &str,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        let tcx = self.infcx.tcx;
        let args_span = use_span.args_or_use();

        let (sugg_span, suggestion) = match tcx.sess.source_map().span_to_snippet(args_span) {
            Ok(string) => {
                if string.starts_with("async ") {
                    let pos = args_span.lo() + BytePos(6);
                    (args_span.with_lo(pos).with_hi(pos), "move ")
                } else if string.starts_with("async|") {
                    let pos = args_span.lo() + BytePos(5);
                    (args_span.with_lo(pos).with_hi(pos), " move")
                } else {
                    (args_span.shrink_to_lo(), "move ")
                }
            }
            Err(_) => (args_span, "move |<args>| <body>"),
        };
        let kind = match use_span.generator_kind() {
            Some(generator_kind) => match generator_kind {
                GeneratorKind::Async(async_kind) => match async_kind {
                    AsyncGeneratorKind::Block => "async block",
                    AsyncGeneratorKind::Closure => "async closure",
                    _ => bug!("async block/closure expected, but async function found."),
                },
                GeneratorKind::Gen => "generator",
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
            &format!(
                "to force the {} to take ownership of {} (and any \
                 other referenced variables), use the `move` keyword",
                kind, captured_var
            ),
            suggestion,
            Applicability::MachineApplicable,
        );

        match category {
            ConstraintCategory::Return(_) | ConstraintCategory::OpaqueType => {
                let msg = format!("{} is returned here", kind);
                err.span_note(constraint_span, &msg);
            }
            ConstraintCategory::CallArgument(_) => {
                fr_name.highlight_region_name(&mut err);
                if matches!(use_span.generator_kind(), Some(GeneratorKind::Async(_))) {
                    err.note(
                        "async blocks are not executed immediately and must either take a \
                         reference or ownership of outside variables they use",
                    );
                } else {
                    let msg = format!("{scope} requires argument type to outlive `{fr_name}`");
                    err.span_note(constraint_span, &msg);
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
        &mut self,
        borrow_span: Span,
        name: &Option<String>,
        upvar_span: Span,
        upvar_name: Symbol,
        escape_span: Span,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        let tcx = self.infcx.tcx;

        let (_, escapes_from) = tcx.article_and_description(self.mir_def_id().to_def_id());

        let mut err =
            borrowck_errors::borrowed_data_escapes_closure(tcx, escape_span, escapes_from);

        err.span_label(
            upvar_span,
            format!("`{}` declared here, outside of the {} body", upvar_name, escapes_from),
        );

        err.span_label(borrow_span, format!("borrow is only valid in the {} body", escapes_from));

        if let Some(name) = name {
            err.span_label(
                escape_span,
                format!("reference to `{}` escapes the {} body here", name, escapes_from),
            );
        } else {
            err.span_label(
                escape_span,
                format!("reference escapes the {} body here", escapes_from),
            );
        }

        err
    }

    fn get_moved_indexes(
        &mut self,
        location: Location,
        mpi: MovePathIndex,
    ) -> (Vec<MoveSite>, Vec<Location>) {
        fn predecessor_locations<'tcx, 'a>(
            body: &'a mir::Body<'tcx>,
            location: Location,
        ) -> impl Iterator<Item = Location> + Captures<'tcx> + 'a {
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
            let path = self.move_data.rev_lookup.find_local(arg);
            if mpis.contains(&path) {
                is_argument = true;
            }
        }

        let mut visited = FxHashSet::default();
        let mut move_locations = FxHashSet::default();
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
                // this analysis only tries to find moves explicitly
                // written by the user, so we ignore the move-outs
                // created by `StorageDead` and at the beginning
                // of a function.
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
            return false;
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
            /* Process back edges (moves in future loop iterations) only if
               the move path is definitely initialized upon loop entry,
               to avoid spurious "in previous iteration" errors.
               During DFS, if there's a path from the error back to the start
               of the function with no intervening init or move, then the
               move path may be uninitialized at loop entry.
            */
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
                let mut visited = FxHashSet::default();
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
        if loan.kind == BorrowKind::Shallow {
            if let Some(section) = self.classify_immutable_section(loan.assigned_place) {
                let mut err = self.cannot_mutate_in_immutable_section(
                    span,
                    loan_span,
                    &descr_place,
                    section,
                    "assign",
                );
                loan_spans.var_span_label(
                    &mut err,
                    format!("borrow occurs due to use{}", loan_spans.describe()),
                    loan.kind.describe_mutability(),
                );

                self.buffer_error(err);

                return;
            }
        }

        let mut err = self.cannot_assign_to_borrowed(span, loan_span, &descr_place);

        loan_spans.var_span_label(
            &mut err,
            format!("borrow occurs due to use{}", loan_spans.describe()),
            loan.kind.describe_mutability(),
        );

        self.explain_why_borrow_contains_point(location, loan, None).add_explanation_to_diagnostic(
            self.infcx.tcx,
            &self.body,
            &self.local_names,
            &mut err,
            "",
            None,
            None,
        );

        self.explain_deref_coercion(loan, &mut err);

        self.buffer_error(err);
    }

    fn explain_deref_coercion(&mut self, loan: &BorrowData<'tcx>, err: &mut Diagnostic) {
        let tcx = self.infcx.tcx;
        if let (
            Some(Terminator { kind: TerminatorKind::Call { from_hir_call: false, .. }, .. }),
            Some((method_did, method_substs)),
        ) = (
            &self.body[loan.reserve_location.block].terminator,
            rustc_const_eval::util::find_self_call(
                tcx,
                self.body,
                loan.assigned_place.local,
                loan.reserve_location.block,
            ),
        ) {
            if tcx.is_diagnostic_item(sym::deref_method, method_did) {
                let deref_target =
                    tcx.get_diagnostic_item(sym::deref_target).and_then(|deref_target| {
                        Instance::resolve(tcx, self.param_env, deref_target, method_substs)
                            .transpose()
                    });
                if let Some(Ok(instance)) = deref_target {
                    let deref_target_ty = instance.ty(tcx, self.param_env);
                    err.note(&format!(
                        "borrow occurs due to deref coercion to `{}`",
                        deref_target_ty
                    ));
                    err.span_note(tcx.def_span(instance.def_id()), "deref defined here");
                }
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
        _location: Location,
        (place, span): (Place<'tcx>, Span),
        assigned_span: Span,
        err_place: Place<'tcx>,
    ) {
        let (from_arg, local_decl, local_name) = match err_place.as_local() {
            Some(local) => (
                self.body.local_kind(local) == LocalKind::Arg,
                Some(&self.body.local_decls[local]),
                self.local_names[local],
            ),
            None => (false, None, None),
        };

        // If root local is initialized immediately (everything apart from let
        // PATTERN;) then make the error refer to that local, rather than the
        // place being assigned later.
        let (place_description, assigned_span) = match local_decl {
            Some(LocalDecl {
                local_info:
                    Some(box LocalInfo::User(
                        ClearCrossCrate::Clear
                        | ClearCrossCrate::Set(BindingForm::Var(VarBindingForm {
                            opt_match_place: None,
                            ..
                        })),
                    ))
                    | Some(box LocalInfo::StaticRef { .. })
                    | None,
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
            err.span_label(assigned_span, format!("first assignment to {}", place_description));
        }
        if let Some(decl) = local_decl
            && let Some(name) = local_name
            && decl.can_be_made_mutable()
        {
            err.span_suggestion(
                decl.source_info.span,
                "consider making this binding mutable",
                format!("mut {}", name),
                Applicability::MachineApplicable,
            );
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
                        | ProjectionElem::Index(_) => kind,
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
        visitor.visit_body(&self.body);
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
            let is_closure = self.infcx.tcx.is_closure(self.mir_def_id().to_def_id());
            if is_closure {
                None
            } else {
                let ty = self.infcx.tcx.type_of(self.mir_def_id()).subst_identity();
                match ty.kind() {
                    ty::FnDef(_, _) | ty::FnPtr(_) => self.annotate_fn_sig(
                        self.mir_def_id(),
                        self.infcx.tcx.fn_sig(self.mir_def_id()).subst_identity(),
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
                            box AggregateKind::Closure(def_id, substs),
                            operands,
                        ) = rvalue
                        {
                            let def_id = def_id.expect_local();
                            for operand in operands {
                                let (Operand::Copy(assigned_from) | Operand::Move(assigned_from)) = operand else {
                                    continue;
                                };
                                debug!(
                                    "annotate_argument_and_return_for_borrow: assigned_from={:?}",
                                    assigned_from
                                );

                                // Find the local from the operand.
                                let Some(assigned_from_local) = assigned_from.local_or_deref_local() else {
                                    continue;
                                };

                                if assigned_from_local != target {
                                    continue;
                                }

                                // If a closure captured our `target` and then assigned
                                // into a place then we should annotate the closure in
                                // case it ends up being assigned into the return place.
                                annotated_closure =
                                    self.annotate_fn_sig(def_id, substs.as_closure().sig());
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
                        let Some(assigned_from_local) = assigned_from.local_or_deref_local() else { continue };
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
                        let (Operand::Copy(assigned_from) | Operand::Move(assigned_from)) = operand else {
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
        let is_closure = self.infcx.tcx.is_closure(did.to_def_id());
        let fn_hir_id = self.infcx.tcx.hir().local_def_id_to_hir_id(did);
        let fn_decl = self.infcx.tcx.hir().fn_decl_by_hir_id(fn_hir_id)?;

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
                    if let ty::Ref(argument_region, _, _) = argument.kind() {
                        if argument_region == return_region {
                            // Need to use the `rustc_middle::ty` types to compare against the
                            // `return_region`. Then use the `rustc_hir` type to get only
                            // the lifetime span.
                            if let hir::TyKind::Ref(lifetime, _) = &fn_decl.inputs[index].kind {
                                // With access to the lifetime, we can get
                                // the span of it.
                                arguments.push((*argument, lifetime.ident.span));
                            } else {
                                bug!("ty type is a ref but hir type is not");
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
    pub(crate) fn emit(&self, cx: &mut MirBorrowckCtxt<'_, 'tcx>, diag: &mut Diagnostic) -> String {
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
                diag.span_label(argument_span, format!("has type `{}`", argument_ty_name));

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
                    diag.span_label(*argument_span, format!("has lifetime `{}`", region_name));
                }

                diag.span_label(*return_span, format!("also has lifetime `{}`", region_name,));

                diag.help(&format!(
                    "use data from the highlighted arguments which match the `{}` lifetime of \
                     the return type",
                    region_name,
                ));

                region_name
            }
        }
    }
}

/// Detect whether one of the provided spans is a statement nested within the top-most visited expr
struct ReferencedStatementsVisitor<'a>(&'a [Span], bool);

impl<'a, 'v> Visitor<'v> for ReferencedStatementsVisitor<'a> {
    fn visit_stmt(&mut self, s: &'v hir::Stmt<'v>) {
        match s.kind {
            hir::StmtKind::Semi(expr) if self.0.contains(&expr.span) => {
                self.1 = true;
            }
            _ => {}
        }
    }
}

/// Given a set of spans representing statements initializing the relevant binding, visit all the
/// function expressions looking for branching code paths that *do not* initialize the binding.
struct ConditionVisitor<'b> {
    spans: &'b [Span],
    name: &'b str,
    errors: Vec<(Span, String)>,
}

impl<'b, 'v> Visitor<'v> for ConditionVisitor<'b> {
    fn visit_expr(&mut self, ex: &'v hir::Expr<'v>) {
        match ex.kind {
            hir::ExprKind::If(cond, body, None) => {
                // `if` expressions with no `else` that initialize the binding might be missing an
                // `else` arm.
                let mut v = ReferencedStatementsVisitor(self.spans, false);
                v.visit_expr(body);
                if v.1 {
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
                let mut a = ReferencedStatementsVisitor(self.spans, false);
                a.visit_expr(body);
                let mut b = ReferencedStatementsVisitor(self.spans, false);
                b.visit_expr(other);
                match (a.1, b.1) {
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
                    .map(|arm| {
                        let mut v = ReferencedStatementsVisitor(self.spans, false);
                        v.visit_arm(arm);
                        v.1
                    })
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
                                self.errors.push((
                                    arm.pat.span.to(guard.body().span),
                                    format!(
                                        "if this pattern and condition are matched, {} is not \
                                         initialized",
                                        self.name
                                    ),
                                ));
                            } else {
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
