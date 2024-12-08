#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]

use rustc_errors::{Applicability, Diag};
use rustc_hir::intravisit::Visitor;
use rustc_hir::{CaptureBy, ExprKind, HirId, Node};
use rustc_middle::bug;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty};
use rustc_mir_dataflow::move_paths::{LookupResult, MovePathIndex};
use rustc_span::{BytePos, ExpnKind, MacroKind, Span};
use rustc_trait_selection::error_reporting::traits::FindExprBySpan;
use rustc_trait_selection::infer::InferCtxtExt;
use tracing::debug;

use crate::MirBorrowckCtxt;
use crate::diagnostics::{CapturedMessageOpt, DescribePlaceOpt, UseSpans};
use crate::prefixes::PrefixSet;

#[derive(Debug)]
pub(crate) enum IllegalMoveOriginKind<'tcx> {
    /// Illegal move due to attempt to move from behind a reference.
    BorrowedContent {
        /// The place the reference refers to: if erroneous code was trying to
        /// move from `(*x).f` this will be `*x`.
        target_place: Place<'tcx>,
    },

    /// Illegal move due to attempt to move from field of an ADT that
    /// implements `Drop`. Rust maintains invariant that all `Drop`
    /// ADT's remain fully-initialized so that user-defined destructor
    /// can safely read from all of the ADT's fields.
    InteriorOfTypeWithDestructor { container_ty: Ty<'tcx> },

    /// Illegal move due to attempt to move out of a slice or array.
    InteriorOfSliceOrArray { ty: Ty<'tcx>, is_index: bool },
}

#[derive(Debug)]
pub(crate) struct MoveError<'tcx> {
    place: Place<'tcx>,
    location: Location,
    kind: IllegalMoveOriginKind<'tcx>,
}

impl<'tcx> MoveError<'tcx> {
    pub(crate) fn new(
        place: Place<'tcx>,
        location: Location,
        kind: IllegalMoveOriginKind<'tcx>,
    ) -> Self {
        MoveError { place, location, kind }
    }
}

// Often when desugaring a pattern match we may have many individual moves in
// MIR that are all part of one operation from the user's point-of-view. For
// example:
//
// let (x, y) = foo()
//
// would move x from the 0 field of some temporary, and y from the 1 field. We
// group such errors together for cleaner error reporting.
//
// Errors are kept separate if they are from places with different parent move
// paths. For example, this generates two errors:
//
// let (&x, &y) = (&String::new(), &String::new());
#[derive(Debug)]
enum GroupedMoveError<'tcx> {
    // Place expression can't be moved from,
    // e.g., match x[0] { s => (), } where x: &[String]
    MovesFromPlace {
        original_path: Place<'tcx>,
        span: Span,
        move_from: Place<'tcx>,
        kind: IllegalMoveOriginKind<'tcx>,
        binds_to: Vec<Local>,
    },
    // Part of a value expression can't be moved from,
    // e.g., match &String::new() { &x => (), }
    MovesFromValue {
        original_path: Place<'tcx>,
        span: Span,
        move_from: MovePathIndex,
        kind: IllegalMoveOriginKind<'tcx>,
        binds_to: Vec<Local>,
    },
    // Everything that isn't from pattern matching.
    OtherIllegalMove {
        original_path: Place<'tcx>,
        use_spans: UseSpans<'tcx>,
        kind: IllegalMoveOriginKind<'tcx>,
    },
}

impl<'infcx, 'tcx> MirBorrowckCtxt<'_, 'infcx, 'tcx> {
    pub(crate) fn report_move_errors(&mut self) {
        let grouped_errors = self.group_move_errors();
        for error in grouped_errors {
            self.report(error);
        }
    }

    fn group_move_errors(&mut self) -> Vec<GroupedMoveError<'tcx>> {
        let mut grouped_errors = Vec::new();
        let errors = std::mem::take(&mut self.move_errors);
        for error in errors {
            self.append_to_grouped_errors(&mut grouped_errors, error);
        }
        grouped_errors
    }

    fn append_to_grouped_errors(
        &self,
        grouped_errors: &mut Vec<GroupedMoveError<'tcx>>,
        error: MoveError<'tcx>,
    ) {
        let MoveError { place: original_path, location, kind } = error;

        // Note: that the only time we assign a place isn't a temporary
        // to a user variable is when initializing it.
        // If that ever stops being the case, then the ever initialized
        // flow could be used.
        if let Some(StatementKind::Assign(box (place, Rvalue::Use(Operand::Move(move_from))))) =
            self.body.basic_blocks[location.block]
                .statements
                .get(location.statement_index)
                .map(|stmt| &stmt.kind)
        {
            if let Some(local) = place.as_local() {
                let local_decl = &self.body.local_decls[local];
                // opt_match_place is the
                // match_span is the span of the expression being matched on
                // match *x.y { ... }        match_place is Some(*x.y)
                //       ^^^^                match_span is the span of *x.y
                //
                // opt_match_place is None for let [mut] x = ... statements,
                // whether or not the right-hand side is a place expression
                if let LocalInfo::User(BindingForm::Var(VarBindingForm {
                    opt_match_place: Some((opt_match_place, match_span)),
                    binding_mode: _,
                    opt_ty_info: _,
                    pat_span: _,
                })) = *local_decl.local_info()
                {
                    let stmt_source_info = self.body.source_info(location);
                    self.append_binding_error(
                        grouped_errors,
                        kind,
                        original_path,
                        *move_from,
                        local,
                        opt_match_place,
                        match_span,
                        stmt_source_info.span,
                    );
                    return;
                }
            }
        }

        let move_spans = self.move_spans(original_path.as_ref(), location);
        grouped_errors.push(GroupedMoveError::OtherIllegalMove {
            use_spans: move_spans,
            original_path,
            kind,
        });
    }

    fn append_binding_error(
        &self,
        grouped_errors: &mut Vec<GroupedMoveError<'tcx>>,
        kind: IllegalMoveOriginKind<'tcx>,
        original_path: Place<'tcx>,
        move_from: Place<'tcx>,
        bind_to: Local,
        match_place: Option<Place<'tcx>>,
        match_span: Span,
        statement_span: Span,
    ) {
        debug!(?match_place, ?match_span, "append_binding_error");

        let from_simple_let = match_place.is_none();
        let match_place = match_place.unwrap_or(move_from);

        match self.move_data.rev_lookup.find(match_place.as_ref()) {
            // Error with the match place
            LookupResult::Parent(_) => {
                for ge in &mut *grouped_errors {
                    if let GroupedMoveError::MovesFromPlace { span, binds_to, .. } = ge
                        && match_span == *span
                    {
                        debug!("appending local({bind_to:?}) to list");
                        if !binds_to.is_empty() {
                            binds_to.push(bind_to);
                        }
                        return;
                    }
                }
                debug!("found a new move error location");

                // Don't need to point to x in let x = ... .
                let (binds_to, span) = if from_simple_let {
                    (vec![], statement_span)
                } else {
                    (vec![bind_to], match_span)
                };
                grouped_errors.push(GroupedMoveError::MovesFromPlace {
                    span,
                    move_from,
                    original_path,
                    kind,
                    binds_to,
                });
            }
            // Error with the pattern
            LookupResult::Exact(_) => {
                let LookupResult::Parent(Some(mpi)) =
                    self.move_data.rev_lookup.find(move_from.as_ref())
                else {
                    // move_from should be a projection from match_place.
                    unreachable!("Probably not unreachable...");
                };
                for ge in &mut *grouped_errors {
                    if let GroupedMoveError::MovesFromValue {
                        span,
                        move_from: other_mpi,
                        binds_to,
                        ..
                    } = ge
                    {
                        if match_span == *span && mpi == *other_mpi {
                            debug!("appending local({bind_to:?}) to list");
                            binds_to.push(bind_to);
                            return;
                        }
                    }
                }
                debug!("found a new move error location");
                grouped_errors.push(GroupedMoveError::MovesFromValue {
                    span: match_span,
                    move_from: mpi,
                    original_path,
                    kind,
                    binds_to: vec![bind_to],
                });
            }
        };
    }

    fn report(&mut self, error: GroupedMoveError<'tcx>) {
        let (mut err, err_span) = {
            let (span, use_spans, original_path, kind) = match error {
                GroupedMoveError::MovesFromPlace { span, original_path, ref kind, .. }
                | GroupedMoveError::MovesFromValue { span, original_path, ref kind, .. } => {
                    (span, None, original_path, kind)
                }
                GroupedMoveError::OtherIllegalMove { use_spans, original_path, ref kind } => {
                    (use_spans.args_or_use(), Some(use_spans), original_path, kind)
                }
            };
            debug!(
                "report: original_path={:?} span={:?}, kind={:?} \
                   original_path.is_upvar_field_projection={:?}",
                original_path,
                span,
                kind,
                self.is_upvar_field_projection(original_path.as_ref())
            );
            if self.has_ambiguous_copy(original_path.ty(self.body, self.infcx.tcx).ty) {
                // If the type may implement Copy, skip the error.
                // It's an error with the Copy implementation (e.g. duplicate Copy) rather than borrow check
                self.dcx().span_delayed_bug(
                    span,
                    "Type may implement copy, but there is no other error.",
                );
                return;
            }
            (
                match kind {
                    &IllegalMoveOriginKind::BorrowedContent { target_place } => self
                        .report_cannot_move_from_borrowed_content(
                            original_path,
                            target_place,
                            span,
                            use_spans,
                        ),
                    &IllegalMoveOriginKind::InteriorOfTypeWithDestructor { container_ty: ty } => {
                        self.cannot_move_out_of_interior_of_drop(span, ty)
                    }
                    &IllegalMoveOriginKind::InteriorOfSliceOrArray { ty, is_index } => {
                        self.cannot_move_out_of_interior_noncopy(span, ty, Some(is_index))
                    }
                },
                span,
            )
        };

        self.add_move_hints(error, &mut err, err_span);
        self.buffer_error(err);
    }

    fn has_ambiguous_copy(&mut self, ty: Ty<'tcx>) -> bool {
        let Some(copy_trait_def) = self.infcx.tcx.lang_items().copy_trait() else { return false };
        // This is only going to be ambiguous if there are incoherent impls, because otherwise
        // ambiguity should never happen in MIR.
        self.infcx.type_implements_trait(copy_trait_def, [ty], self.infcx.param_env).may_apply()
    }

    fn report_cannot_move_from_static(&mut self, place: Place<'tcx>, span: Span) -> Diag<'infcx> {
        let description = if place.projection.len() == 1 {
            format!("static item {}", self.describe_any_place(place.as_ref()))
        } else {
            let base_static = PlaceRef { local: place.local, projection: &[ProjectionElem::Deref] };

            format!(
                "{} as {} is a static item",
                self.describe_any_place(place.as_ref()),
                self.describe_any_place(base_static),
            )
        };

        self.cannot_move_out_of(span, &description)
    }

    fn suggest_clone_of_captured_var_in_move_closure(
        &self,
        err: &mut Diag<'_>,
        upvar_hir_id: HirId,
        upvar_name: &str,
        use_spans: Option<UseSpans<'tcx>>,
    ) {
        let tcx = self.infcx.tcx;
        let typeck_results = tcx.typeck(self.mir_def_id());
        let Some(use_spans) = use_spans else { return };
        // We only care about the case where a closure captured a binding.
        let UseSpans::ClosureUse { args_span, .. } = use_spans else { return };
        let Some(body_id) = tcx.hir_node(self.mir_hir_id()).body_id() else { return };
        // Fetch the type of the expression corresponding to the closure-captured binding.
        let Some(captured_ty) = typeck_results.node_type_opt(upvar_hir_id) else { return };
        if !self.implements_clone(captured_ty) {
            // We only suggest cloning the captured binding if the type can actually be cloned.
            return;
        };
        // Find the closure that captured the binding.
        let mut expr_finder = FindExprBySpan::new(args_span, tcx);
        expr_finder.include_closures = true;
        expr_finder.visit_expr(tcx.hir().body(body_id).value);
        let Some(closure_expr) = expr_finder.result else { return };
        let ExprKind::Closure(closure) = closure_expr.kind else { return };
        // We'll only suggest cloning the binding if it's a `move` closure.
        let CaptureBy::Value { .. } = closure.capture_clause else { return };
        // Find the expression within the closure where the binding is consumed.
        let mut suggested = false;
        let use_span = use_spans.var_or_use();
        let mut expr_finder = FindExprBySpan::new(use_span, tcx);
        expr_finder.include_closures = true;
        expr_finder.visit_expr(tcx.hir().body(body_id).value);
        let Some(use_expr) = expr_finder.result else { return };
        let parent = tcx.parent_hir_node(use_expr.hir_id);
        if let Node::Expr(expr) = parent
            && let ExprKind::Assign(lhs, ..) = expr.kind
            && lhs.hir_id == use_expr.hir_id
        {
            // Cloning the value being assigned makes no sense:
            //
            // error[E0507]: cannot move out of `var`, a captured variable in an `FnMut` closure
            //   --> $DIR/option-content-move2.rs:11:9
            //    |
            // LL |     let mut var = None;
            //    |         ------- captured outer variable
            // LL |     func(|| {
            //    |          -- captured by this `FnMut` closure
            // LL |         // Shouldn't suggest `move ||.as_ref()` here
            // LL |         move || {
            //    |         ^^^^^^^ `var` is moved here
            // LL |
            // LL |             var = Some(NotCopyable);
            //    |             ---
            //    |             |
            //    |             variable moved due to use in closure
            //    |             move occurs because `var` has type `Option<NotCopyable>`, which does not implement the `Copy` trait
            //    |
            return;
        }

        // Search for an appropriate place for the structured `.clone()` suggestion to be applied.
        // If we encounter a statement before the borrow error, we insert a statement there.
        for (_, node) in tcx.hir().parent_iter(closure_expr.hir_id) {
            if let Node::Stmt(stmt) = node {
                let padding = tcx
                    .sess
                    .source_map()
                    .indentation_before(stmt.span)
                    .unwrap_or_else(|| "    ".to_string());
                err.multipart_suggestion_verbose(
                    "clone the value before moving it into the closure",
                    vec![
                        (
                            stmt.span.shrink_to_lo(),
                            format!("let value = {upvar_name}.clone();\n{padding}"),
                        ),
                        (use_span, "value".to_string()),
                    ],
                    Applicability::MachineApplicable,
                );
                suggested = true;
                break;
            } else if let Node::Expr(expr) = node
                && let ExprKind::Closure(_) = expr.kind
            {
                // We want to suggest cloning only on the first closure, not
                // subsequent ones (like `ui/suggestions/option-content-move2.rs`).
                break;
            }
        }
        if !suggested {
            // If we couldn't find a statement for us to insert a new `.clone()` statement before,
            // we have a bare expression, so we suggest the creation of a new block inline to go
            // from `move || val` to `{ let value = val.clone(); move || value }`.
            let padding = tcx
                .sess
                .source_map()
                .indentation_before(closure_expr.span)
                .unwrap_or_else(|| "    ".to_string());
            err.multipart_suggestion_verbose(
                "clone the value before moving it into the closure",
                vec![
                    (
                        closure_expr.span.shrink_to_lo(),
                        format!("{{\n{padding}let value = {upvar_name}.clone();\n{padding}"),
                    ),
                    (use_spans.var_or_use(), "value".to_string()),
                    (closure_expr.span.shrink_to_hi(), format!("\n{padding}}}")),
                ],
                Applicability::MachineApplicable,
            );
        }
    }

    fn report_cannot_move_from_borrowed_content(
        &mut self,
        move_place: Place<'tcx>,
        deref_target_place: Place<'tcx>,
        span: Span,
        use_spans: Option<UseSpans<'tcx>>,
    ) -> Diag<'infcx> {
        let tcx = self.infcx.tcx;
        // Inspect the type of the content behind the
        // borrow to provide feedback about why this
        // was a move rather than a copy.
        let ty = deref_target_place.ty(self.body, tcx).ty;
        let upvar_field = self
            .prefixes(move_place.as_ref(), PrefixSet::All)
            .find_map(|p| self.is_upvar_field_projection(p));

        let deref_base = match deref_target_place.projection.as_ref() {
            [proj_base @ .., ProjectionElem::Deref] => {
                PlaceRef { local: deref_target_place.local, projection: proj_base }
            }
            _ => bug!("deref_target_place is not a deref projection"),
        };

        if let PlaceRef { local, projection: [] } = deref_base {
            let decl = &self.body.local_decls[local];
            if decl.is_ref_for_guard() {
                return self
                    .cannot_move_out_of(
                        span,
                        &format!("`{}` in pattern guard", self.local_names[local].unwrap()),
                    )
                    .with_note(
                        "variables bound in patterns cannot be moved from \
                         until after the end of the pattern guard",
                    );
            } else if decl.is_ref_to_static() {
                return self.report_cannot_move_from_static(move_place, span);
            }
        }

        debug!("report: ty={:?}", ty);
        let mut err = match ty.kind() {
            ty::Array(..) | ty::Slice(..) => {
                self.cannot_move_out_of_interior_noncopy(span, ty, None)
            }
            ty::Closure(def_id, closure_args)
                if def_id.as_local() == Some(self.mir_def_id()) && upvar_field.is_some() =>
            {
                let closure_kind_ty = closure_args.as_closure().kind_ty();
                let closure_kind = match closure_kind_ty.to_opt_closure_kind() {
                    Some(kind @ (ty::ClosureKind::Fn | ty::ClosureKind::FnMut)) => kind,
                    Some(ty::ClosureKind::FnOnce) => {
                        bug!("closure kind does not match first argument type")
                    }
                    None => bug!("closure kind not inferred by borrowck"),
                };
                let capture_description =
                    format!("captured variable in an `{closure_kind}` closure");

                let upvar = &self.upvars[upvar_field.unwrap().index()];
                let upvar_hir_id = upvar.get_root_variable();
                let upvar_name = upvar.to_string(tcx);
                let upvar_span = tcx.hir().span(upvar_hir_id);

                let place_name = self.describe_any_place(move_place.as_ref());

                let place_description =
                    if self.is_upvar_field_projection(move_place.as_ref()).is_some() {
                        format!("{place_name}, a {capture_description}")
                    } else {
                        format!("{place_name}, as `{upvar_name}` is a {capture_description}")
                    };

                debug!(
                    "report: closure_kind_ty={:?} closure_kind={:?} place_description={:?}",
                    closure_kind_ty, closure_kind, place_description,
                );

                let closure_span = tcx.def_span(def_id);
                let mut err = self
                    .cannot_move_out_of(span, &place_description)
                    .with_span_label(upvar_span, "captured outer variable")
                    .with_span_label(
                        closure_span,
                        format!("captured by this `{closure_kind}` closure"),
                    );
                self.suggest_clone_of_captured_var_in_move_closure(
                    &mut err,
                    upvar_hir_id,
                    &upvar_name,
                    use_spans,
                );
                err
            }
            _ => {
                let source = self.borrowed_content_source(deref_base);
                let move_place_ref = move_place.as_ref();
                match (
                    self.describe_place_with_options(
                        move_place_ref,
                        DescribePlaceOpt {
                            including_downcast: false,
                            including_tuple_field: false,
                        },
                    ),
                    self.describe_name(move_place_ref),
                    source.describe_for_named_place(),
                ) {
                    (Some(place_desc), Some(name), Some(source_desc)) => self.cannot_move_out_of(
                        span,
                        &format!("`{place_desc}` as enum variant `{name}` which is behind a {source_desc}"),
                    ),
                    (Some(place_desc), Some(name), None) => self.cannot_move_out_of(
                        span,
                        &format!("`{place_desc}` as enum variant `{name}`"),
                    ),
                    (Some(place_desc), _, Some(source_desc)) => self.cannot_move_out_of(
                        span,
                        &format!("`{place_desc}` which is behind a {source_desc}"),
                    ),
                    (_, _, _) => self.cannot_move_out_of(
                        span,
                        &source.describe_for_unnamed_place(tcx),
                    ),
                }
            }
        };
        let msg_opt = CapturedMessageOpt {
            is_partial_move: false,
            is_loop_message: false,
            is_move_msg: false,
            is_loop_move: false,
            has_suggest_reborrow: false,
            maybe_reinitialized_locations_is_empty: true,
        };
        if let Some(use_spans) = use_spans {
            self.explain_captures(&mut err, span, span, use_spans, move_place, msg_opt);
        }
        err
    }

    fn add_move_hints(&self, error: GroupedMoveError<'tcx>, err: &mut Diag<'_>, span: Span) {
        match error {
            GroupedMoveError::MovesFromPlace { mut binds_to, move_from, .. } => {
                self.add_borrow_suggestions(err, span);
                if binds_to.is_empty() {
                    let place_ty = move_from.ty(self.body, self.infcx.tcx).ty;
                    let place_desc = match self.describe_place(move_from.as_ref()) {
                        Some(desc) => format!("`{desc}`"),
                        None => "value".to_string(),
                    };

                    if let Some(expr) = self.find_expr(span) {
                        self.suggest_cloning(err, place_ty, expr, None);
                    }

                    err.subdiagnostic(crate::session_diagnostics::TypeNoCopy::Label {
                        is_partial_move: false,
                        ty: place_ty,
                        place: &place_desc,
                        span,
                    });
                } else {
                    binds_to.sort();
                    binds_to.dedup();

                    self.add_move_error_details(err, &binds_to);
                }
            }
            GroupedMoveError::MovesFromValue { mut binds_to, .. } => {
                binds_to.sort();
                binds_to.dedup();
                self.add_move_error_suggestions(err, &binds_to);
                self.add_move_error_details(err, &binds_to);
            }
            // No binding. Nothing to suggest.
            GroupedMoveError::OtherIllegalMove { ref original_path, use_spans, .. } => {
                let use_span = use_spans.var_or_use();
                let place_ty = original_path.ty(self.body, self.infcx.tcx).ty;
                let place_desc = match self.describe_place(original_path.as_ref()) {
                    Some(desc) => format!("`{desc}`"),
                    None => "value".to_string(),
                };

                if let Some(expr) = self.find_expr(use_span) {
                    self.suggest_cloning(err, place_ty, expr, Some(use_spans));
                }

                err.subdiagnostic(crate::session_diagnostics::TypeNoCopy::Label {
                    is_partial_move: false,
                    ty: place_ty,
                    place: &place_desc,
                    span: use_span,
                });

                use_spans.args_subdiag(err, |args_span| {
                    crate::session_diagnostics::CaptureArgLabel::MoveOutPlace {
                        place: place_desc,
                        args_span,
                    }
                });

                self.add_note_for_packed_struct_derive(err, original_path.local);
            }
        }
    }

    fn add_borrow_suggestions(&self, err: &mut Diag<'_>, span: Span) {
        match self.infcx.tcx.sess.source_map().span_to_snippet(span) {
            Ok(snippet) if snippet.starts_with('*') => {
                let sp = span.with_lo(span.lo() + BytePos(1));
                let inner = self.find_expr(sp);
                let mut is_raw_ptr = false;
                if let Some(inner) = inner {
                    let typck_result = self.infcx.tcx.typeck(self.mir_def_id());
                    if let Some(inner_type) = typck_result.node_type_opt(inner.hir_id) {
                        if matches!(inner_type.kind(), ty::RawPtr(..)) {
                            is_raw_ptr = true;
                        }
                    }
                }
                // If the `inner` is a raw pointer, do not suggest removing the "*", see #126863
                // FIXME: need to check whether the assigned object can be a raw pointer, see `tests/ui/borrowck/issue-20801.rs`.
                if !is_raw_ptr {
                    err.span_suggestion_verbose(
                        span.with_hi(span.lo() + BytePos(1)),
                        "consider removing the dereference here",
                        String::new(),
                        Applicability::MaybeIncorrect,
                    );
                }
            }
            _ => {
                err.span_suggestion_verbose(
                    span.shrink_to_lo(),
                    "consider borrowing here",
                    '&',
                    Applicability::MaybeIncorrect,
                );
            }
        }
    }

    fn add_move_error_suggestions(&self, err: &mut Diag<'_>, binds_to: &[Local]) {
        let mut suggestions: Vec<(Span, String, String)> = Vec::new();
        for local in binds_to {
            let bind_to = &self.body.local_decls[*local];
            if let LocalInfo::User(BindingForm::Var(VarBindingForm { pat_span, .. })) =
                *bind_to.local_info()
            {
                let Ok(pat_snippet) = self.infcx.tcx.sess.source_map().span_to_snippet(pat_span)
                else {
                    continue;
                };
                let Some(stripped) = pat_snippet.strip_prefix('&') else {
                    suggestions.push((
                        bind_to.source_info.span.shrink_to_lo(),
                        "consider borrowing the pattern binding".to_string(),
                        "ref ".to_string(),
                    ));
                    continue;
                };
                let inner_pat_snippet = stripped.trim_start();
                let (pat_span, suggestion, to_remove) = if inner_pat_snippet.starts_with("mut")
                    && inner_pat_snippet["mut".len()..].starts_with(rustc_lexer::is_whitespace)
                {
                    let inner_pat_snippet = inner_pat_snippet["mut".len()..].trim_start();
                    let pat_span = pat_span.with_hi(
                        pat_span.lo()
                            + BytePos((pat_snippet.len() - inner_pat_snippet.len()) as u32),
                    );
                    (pat_span, String::new(), "mutable borrow")
                } else {
                    let pat_span = pat_span.with_hi(
                        pat_span.lo()
                            + BytePos(
                                (pat_snippet.len() - inner_pat_snippet.trim_start().len()) as u32,
                            ),
                    );
                    (pat_span, String::new(), "borrow")
                };
                suggestions.push((
                    pat_span,
                    format!("consider removing the {to_remove}"),
                    suggestion,
                ));
            }
        }
        suggestions.sort_unstable_by_key(|&(span, _, _)| span);
        suggestions.dedup_by_key(|&mut (span, _, _)| span);
        for (span, msg, suggestion) in suggestions {
            err.span_suggestion_verbose(span, msg, suggestion, Applicability::MachineApplicable);
        }
    }

    fn add_move_error_details(&self, err: &mut Diag<'_>, binds_to: &[Local]) {
        for (j, local) in binds_to.iter().enumerate() {
            let bind_to = &self.body.local_decls[*local];
            let binding_span = bind_to.source_info.span;

            if j == 0 {
                err.span_label(binding_span, "data moved here");
            } else {
                err.span_label(binding_span, "...and here");
            }

            if binds_to.len() == 1 {
                let place_desc = &format!("`{}`", self.local_names[*local].unwrap());

                if let Some(expr) = self.find_expr(binding_span) {
                    self.suggest_cloning(err, bind_to.ty, expr, None);
                }

                err.subdiagnostic(crate::session_diagnostics::TypeNoCopy::Label {
                    is_partial_move: false,
                    ty: bind_to.ty,
                    place: place_desc,
                    span: binding_span,
                });
            }
        }

        if binds_to.len() > 1 {
            err.note(
                "move occurs because these variables have types that don't implement the `Copy` \
                 trait",
            );
        }
    }

    /// Adds an explanatory note if the move error occurs in a derive macro
    /// expansion of a packed struct.
    /// Such errors happen because derive macro expansions shy away from taking
    /// references to the struct's fields since doing so would be undefined behaviour
    fn add_note_for_packed_struct_derive(&self, err: &mut Diag<'_>, local: Local) {
        let local_place: PlaceRef<'tcx> = local.into();
        let local_ty = local_place.ty(self.body.local_decls(), self.infcx.tcx).ty.peel_refs();

        if let Some(adt) = local_ty.ty_adt_def()
            && adt.repr().packed()
            && let ExpnKind::Macro(MacroKind::Derive, name) =
                self.body.span.ctxt().outer_expn_data().kind
        {
            err.note(format!("`#[derive({name})]` triggers a move because taking references to the fields of a packed struct is undefined behaviour"));
        }
    }
}
