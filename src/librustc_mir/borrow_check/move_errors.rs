use core::unicode::property::Pattern_White_Space;

use rustc::mir::*;
use rustc::ty;
use rustc_errors::{DiagnosticBuilder,Applicability};
use syntax_pos::Span;

use crate::borrow_check::MirBorrowckCtxt;
use crate::borrow_check::prefixes::PrefixSet;
use crate::borrow_check::error_reporting::UseSpans;
use crate::dataflow::move_paths::{
    IllegalMoveOrigin, IllegalMoveOriginKind,
    LookupResult, MoveError, MovePathIndex,
};

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
        use_spans: UseSpans,
        kind: IllegalMoveOriginKind<'tcx>,
    },
}

impl<'a, 'tcx> MirBorrowckCtxt<'a, 'tcx> {
    pub(crate) fn report_move_errors(&mut self, move_errors: Vec<(Place<'tcx>, MoveError<'tcx>)>) {
        let grouped_errors = self.group_move_errors(move_errors);
        for error in grouped_errors {
            self.report(error);
        }
    }

    fn group_move_errors(
        &self,
        errors: Vec<(Place<'tcx>, MoveError<'tcx>)>
    ) -> Vec<GroupedMoveError<'tcx>> {
        let mut grouped_errors = Vec::new();
        for (original_path, error) in errors {
            self.append_to_grouped_errors(&mut grouped_errors, original_path, error);
        }
        grouped_errors
    }

    fn append_to_grouped_errors(
        &self,
        grouped_errors: &mut Vec<GroupedMoveError<'tcx>>,
        original_path: Place<'tcx>,
        error: MoveError<'tcx>,
    ) {
        match error {
            MoveError::UnionMove { .. } => {
                unimplemented!("don't know how to report union move errors yet.")
            }
            MoveError::IllegalMove {
                cannot_move_out_of: IllegalMoveOrigin { location, kind },
            } => {
                // Note: that the only time we assign a place isn't a temporary
                // to a user variable is when initializing it.
                // If that ever stops being the case, then the ever initialized
                // flow could be used.
                if let Some(StatementKind::Assign(
                    Place {
                        base: PlaceBase::Local(local),
                        projection: None,
                    },
                    box Rvalue::Use(Operand::Move(move_from)),
                )) = self.body.basic_blocks()[location.block]
                    .statements
                    .get(location.statement_index)
                    .map(|stmt| &stmt.kind)
                {
                    let local_decl = &self.body.local_decls[*local];
                    // opt_match_place is the
                    // match_span is the span of the expression being matched on
                    // match *x.y { ... }        match_place is Some(*x.y)
                    //       ^^^^                match_span is the span of *x.y
                    //
                    // opt_match_place is None for let [mut] x = ... statements,
                    // whether or not the right-hand side is a place expression
                    if let Some(ClearCrossCrate::Set(BindingForm::Var(VarBindingForm {
                        opt_match_place: Some((ref opt_match_place, match_span)),
                        binding_mode: _,
                        opt_ty_info: _,
                        pat_span: _,
                    }))) = local_decl.is_user_variable
                    {
                        let stmt_source_info = self.body.source_info(location);
                        self.append_binding_error(
                            grouped_errors,
                            kind,
                            original_path,
                            move_from,
                            *local,
                            opt_match_place,
                            match_span,
                            stmt_source_info.span,
                        );
                        return;
                    }
                }

                let move_spans = self.move_spans(original_path.as_ref(), location);
                grouped_errors.push(GroupedMoveError::OtherIllegalMove {
                    use_spans: move_spans,
                    original_path,
                    kind,
                });
            }
        }
    }

    fn append_binding_error(
        &self,
        grouped_errors: &mut Vec<GroupedMoveError<'tcx>>,
        kind: IllegalMoveOriginKind<'tcx>,
        original_path: Place<'tcx>,
        move_from: &Place<'tcx>,
        bind_to: Local,
        match_place: &Option<Place<'tcx>>,
        match_span: Span,
        statement_span: Span,
    ) {
        debug!(
            "append_binding_error(match_place={:?}, match_span={:?})",
            match_place, match_span
        );

        let from_simple_let = match_place.is_none();
        let match_place = match_place.as_ref().unwrap_or(move_from);

        match self.move_data.rev_lookup.find(match_place.as_ref()) {
            // Error with the match place
            LookupResult::Parent(_) => {
                for ge in &mut *grouped_errors {
                    if let GroupedMoveError::MovesFromPlace { span, binds_to, .. } = ge {
                        if match_span == *span {
                            debug!("appending local({:?}) to list", bind_to);
                            if !binds_to.is_empty() {
                                binds_to.push(bind_to);
                            }
                            return;
                        }
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
                    move_from: match_place.clone(),
                    original_path,
                    kind,
                    binds_to,
                });
            }
            // Error with the pattern
            LookupResult::Exact(_) => {
                let mpi = match self.move_data.rev_lookup.find(move_from.as_ref()) {
                    LookupResult::Parent(Some(mpi)) => mpi,
                    // move_from should be a projection from match_place.
                    _ => unreachable!("Probably not unreachable..."),
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
                            debug!("appending local({:?}) to list", bind_to);
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
            let (span, original_path, kind): (Span, &Place<'tcx>, &IllegalMoveOriginKind<'_>) =
                match error {
                    GroupedMoveError::MovesFromPlace { span, ref original_path, ref kind, .. } |
                    GroupedMoveError::MovesFromValue { span, ref original_path, ref kind, .. } => {
                        (span, original_path, kind)
                    }
                    GroupedMoveError::OtherIllegalMove {
                        use_spans,
                        ref original_path,
                        ref kind
                    } => {
                        (use_spans.args_or_use(), original_path, kind)
                    },
                };
            debug!("report: original_path={:?} span={:?}, kind={:?} \
                   original_path.is_upvar_field_projection={:?}", original_path, span, kind,
                   self.is_upvar_field_projection(original_path.as_ref()));
            (
                match kind {
                    IllegalMoveOriginKind::Static => {
                        self.report_cannot_move_from_static(original_path, span)
                    }
                    IllegalMoveOriginKind::BorrowedContent { target_place } => {
                        self.report_cannot_move_from_borrowed_content(
                            original_path,
                            target_place,
                            span,
                        )
                    }
                    IllegalMoveOriginKind::InteriorOfTypeWithDestructor { container_ty: ty } => {
                        self.cannot_move_out_of_interior_of_drop(span, ty)
                    }
                    IllegalMoveOriginKind::InteriorOfSliceOrArray { ty, is_index } =>
                        self.cannot_move_out_of_interior_noncopy(
                            span, ty, Some(*is_index),
                        ),
                },
                span,
            )
        };

        self.add_move_hints(error, &mut err, err_span);
        err.buffer(&mut self.errors_buffer);
    }

    fn report_cannot_move_from_static(
        &mut self,
        place: &Place<'tcx>,
        span: Span
    ) -> DiagnosticBuilder<'a> {
        let description = if place.projection.is_none() {
            format!("static item `{}`", self.describe_place(place.as_ref()).unwrap())
        } else {
            let mut base_static = &place.projection;
            while let Some(box Projection { base: Some(ref proj), .. }) = base_static {
                base_static = &proj.base;
            }
            let base_static = PlaceRef {
                base: &place.base,
                projection: base_static,
            };

            format!(
                "`{:?}` as `{:?}` is a static item",
                self.describe_place(place.as_ref()).unwrap(),
                self.describe_place(base_static).unwrap(),
            )
        };

        self.cannot_move_out_of(span, &description)
    }

    fn report_cannot_move_from_borrowed_content(
        &mut self,
        move_place: &Place<'tcx>,
        deref_target_place: &Place<'tcx>,
        span: Span,
    ) -> DiagnosticBuilder<'a> {
        // Inspect the type of the content behind the
        // borrow to provide feedback about why this
        // was a move rather than a copy.
        let ty = deref_target_place.ty(self.body, self.infcx.tcx).ty;
        let upvar_field = self.prefixes(move_place.as_ref(), PrefixSet::All)
            .find_map(|p| self.is_upvar_field_projection(p));

        let deref_base = match deref_target_place.projection {
            Some(box Projection { ref base, elem: ProjectionElem::Deref }) => PlaceRef {
                base: &deref_target_place.base,
                projection: base,
            },
            _ => bug!("deref_target_place is not a deref projection"),
        };

        if let PlaceRef {
            base: PlaceBase::Local(local),
            projection: None,
        } = deref_base {
            let decl = &self.body.local_decls[*local];
            if decl.is_ref_for_guard() {
                let mut err = self.cannot_move_out_of(
                    span,
                    &format!("`{}` in pattern guard", decl.name.unwrap()),
                );
                err.note(
                    "variables bound in patterns cannot be moved from \
                     until after the end of the pattern guard");
                return err;
            }
        }

        debug!("report: ty={:?}", ty);
        let mut err = match ty.sty {
            ty::Array(..) | ty::Slice(..) =>
                self.cannot_move_out_of_interior_noncopy(span, ty, None),
            ty::Closure(def_id, closure_substs)
                if def_id == self.mir_def_id && upvar_field.is_some()
            => {
                let closure_kind_ty = closure_substs.closure_kind_ty(def_id, self.infcx.tcx);
                let closure_kind = closure_kind_ty.to_opt_closure_kind();
                let capture_description = match closure_kind {
                    Some(ty::ClosureKind::Fn) => {
                        "captured variable in an `Fn` closure"
                    }
                    Some(ty::ClosureKind::FnMut) => {
                        "captured variable in an `FnMut` closure"
                    }
                    Some(ty::ClosureKind::FnOnce) => {
                        bug!("closure kind does not match first argument type")
                    }
                    None => bug!("closure kind not inferred by borrowck"),
                };

                let upvar = &self.upvars[upvar_field.unwrap().index()];
                let upvar_hir_id = upvar.var_hir_id;
                let upvar_name = upvar.name;
                let upvar_span = self.infcx.tcx.hir().span(upvar_hir_id);

                let place_name = self.describe_place(move_place.as_ref()).unwrap();

                let place_description = if self
                    .is_upvar_field_projection(move_place.as_ref())
                    .is_some()
                {
                    format!("`{}`, a {}", place_name, capture_description)
                } else {
                    format!(
                        "`{}`, as `{}` is a {}",
                        place_name,
                        upvar_name,
                        capture_description,
                    )
                };

                debug!(
                    "report: closure_kind_ty={:?} closure_kind={:?} place_description={:?}",
                    closure_kind_ty, closure_kind, place_description,
                );

                let mut diag = self.cannot_move_out_of(span, &place_description);

                diag.span_label(upvar_span, "captured outer variable");

                diag
            }
            _ => {
                let source = self.borrowed_content_source(deref_base);
                match (
                    self.describe_place(move_place.as_ref()),
                    source.describe_for_named_place(),
                ) {
                    (Some(place_desc), Some(source_desc)) => {
                        self.cannot_move_out_of(
                            span,
                            &format!("`{}` which is behind a {}", place_desc, source_desc),
                        )
                    }
                    (_, _) => {
                        self.cannot_move_out_of(
                            span,
                            &source.describe_for_unnamed_place(),
                        )
                    }
                }
            },
        };
        let move_ty = format!(
            "{:?}",
            move_place.ty(self.body, self.infcx.tcx).ty,
        );
        let snippet = self.infcx.tcx.sess.source_map().span_to_snippet(span).unwrap();
        let is_option = move_ty.starts_with("std::option::Option");
        let is_result = move_ty.starts_with("std::result::Result");
        if  is_option || is_result {
            err.span_suggestion(
                span,
                &format!("consider borrowing the `{}`'s content", if is_option {
                    "Option"
                } else {
                    "Result"
                }),
                format!("{}.as_ref()", snippet),
                Applicability::MaybeIncorrect,
            );
        }
        err
    }

    fn add_move_hints(
        &self,
        error: GroupedMoveError<'tcx>,
        err: &mut DiagnosticBuilder<'a>,
        span: Span,
    ) {
        let snippet = self.infcx.tcx.sess.source_map().span_to_snippet(span).unwrap();
        match error {
            GroupedMoveError::MovesFromPlace {
                mut binds_to,
                move_from,
                ..
            } => {
                err.span_suggestion(
                    span,
                    "consider borrowing here",
                    format!("&{}", snippet),
                    Applicability::Unspecified,
                );

                if binds_to.is_empty() {
                    let place_ty = move_from.ty(self.body, self.infcx.tcx).ty;
                    let place_desc = match self.describe_place(move_from.as_ref()) {
                        Some(desc) => format!("`{}`", desc),
                        None => format!("value"),
                    };

                    self.note_type_does_not_implement_copy(
                        err,
                        &place_desc,
                        place_ty,
                        Some(span)
                    );
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
                let span = use_spans.var_or_use();
                let place_ty = original_path.ty(self.body, self.infcx.tcx).ty;
                let place_desc = match self.describe_place(original_path.as_ref()) {
                    Some(desc) => format!("`{}`", desc),
                    None => format!("value"),
                };
                self.note_type_does_not_implement_copy(
                    err,
                    &place_desc,
                    place_ty,
                    Some(span),
                );

                use_spans.args_span_label(err, format!("move out of {} occurs here", place_desc));
                use_spans.var_span_label(
                    err,
                    format!("move occurs due to use{}", use_spans.describe()),
                );
            },
        }
    }

    fn add_move_error_suggestions(
        &self,
        err: &mut DiagnosticBuilder<'a>,
        binds_to: &[Local],
    ) {
        let mut suggestions: Vec<(Span, &str, String)> = Vec::new();
        for local in binds_to {
            let bind_to = &self.body.local_decls[*local];
            if let Some(
                ClearCrossCrate::Set(BindingForm::Var(VarBindingForm {
                    pat_span,
                    ..
                }))
            ) = bind_to.is_user_variable {
                let pat_snippet = self.infcx.tcx.sess.source_map()
                    .span_to_snippet(pat_span)
                    .unwrap();
                if pat_snippet.starts_with('&') {
                    let pat_snippet = pat_snippet[1..].trim_start();
                    let suggestion;
                    let to_remove;
                    if pat_snippet.starts_with("mut")
                        && pat_snippet["mut".len()..].starts_with(Pattern_White_Space)
                    {
                        suggestion = pat_snippet["mut".len()..].trim_start();
                        to_remove = "&mut";
                    } else {
                        suggestion = pat_snippet;
                        to_remove = "&";
                    }
                    suggestions.push((
                        pat_span,
                        to_remove,
                        suggestion.to_owned(),
                    ));
                }
            }
        }
        suggestions.sort_unstable_by_key(|&(span, _, _)| span);
        suggestions.dedup_by_key(|&mut (span, _, _)| span);
        for (span, to_remove, suggestion) in suggestions {
            err.span_suggestion(
                span,
                &format!("consider removing the `{}`", to_remove),
                suggestion,
                Applicability::MachineApplicable,
            );
        }
    }

    fn add_move_error_details(
        &self,
        err: &mut DiagnosticBuilder<'a>,
        binds_to: &[Local],
    ) {
        let mut noncopy_var_spans = Vec::new();
        for (j, local) in binds_to.into_iter().enumerate() {
            let bind_to = &self.body.local_decls[*local];
            let binding_span = bind_to.source_info.span;

            if j == 0 {
                err.span_label(binding_span, "data moved here");
            } else {
                err.span_label(binding_span, "...and here");
            }

            if binds_to.len() == 1 {
                self.note_type_does_not_implement_copy(
                    err,
                    &format!("`{}`", bind_to.name.unwrap()),
                    bind_to.ty,
                    Some(binding_span)
                );
            } else {
                noncopy_var_spans.push(binding_span);
            }
        }

        if binds_to.len() > 1 {
            err.span_note(
                noncopy_var_spans,
                "move occurs because these variables have types that \
                    don't implement the `Copy` trait",
            );
        }
    }
}
