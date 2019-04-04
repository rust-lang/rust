use core::unicode::property::Pattern_White_Space;
use std::fmt::{self, Display};

use rustc::mir::*;
use rustc::ty;
use rustc_errors::{DiagnosticBuilder,Applicability};
use syntax_pos::Span;

use crate::borrow_check::MirBorrowckCtxt;
use crate::borrow_check::prefixes::PrefixSet;
use crate::dataflow::move_paths::{
    IllegalMoveOrigin, IllegalMoveOriginKind, InitLocation,
    LookupResult, MoveError, MovePathIndex,
};
use crate::util::borrowck_errors::{BorrowckErrors, Origin};

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
        span: Span,
        kind: IllegalMoveOriginKind<'tcx>,
    },
}

enum BorrowedContentSource {
    Arc,
    Rc,
    DerefRawPointer,
    Other,
}

impl Display for BorrowedContentSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            BorrowedContentSource::Arc => write!(f, "an `Arc`"),
            BorrowedContentSource::Rc => write!(f, "an `Rc`"),
            BorrowedContentSource::DerefRawPointer => write!(f, "dereference of raw pointer"),
            BorrowedContentSource::Other => write!(f, "borrowed content"),
        }
    }
}

impl<'a, 'gcx, 'tcx> MirBorrowckCtxt<'a, 'gcx, 'tcx> {
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
                let stmt_source_info = self.mir.source_info(location);
                // Note: that the only time we assign a place isn't a temporary
                // to a user variable is when initializing it.
                // If that ever stops being the case, then the ever initialized
                // flow could be used.
                if let Some(StatementKind::Assign(
                    Place::Base(PlaceBase::Local(local)),
                    box Rvalue::Use(Operand::Move(move_from)),
                )) = self.mir.basic_blocks()[location.block]
                    .statements
                    .get(location.statement_index)
                    .map(|stmt| &stmt.kind)
                {
                    let local_decl = &self.mir.local_decls[*local];
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
                grouped_errors.push(GroupedMoveError::OtherIllegalMove {
                    span: stmt_source_info.span,
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

        match self.move_data.rev_lookup.find(match_place) {
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
                let mpi = match self.move_data.rev_lookup.find(move_from) {
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
                    GroupedMoveError::MovesFromPlace {
                        span,
                        ref original_path,
                        ref kind,
                        ..
                    } |
                    GroupedMoveError::MovesFromValue { span, ref original_path, ref kind, .. } |
                    GroupedMoveError::OtherIllegalMove { span, ref original_path, ref kind } => {
                        (span, original_path, kind)
                    },
                };
            let origin = Origin::Mir;
            debug!("report: original_path={:?} span={:?}, kind={:?} \
                   original_path.is_upvar_field_projection={:?}", original_path, span, kind,
                   original_path.is_upvar_field_projection(self.mir, &self.infcx.tcx));
            (
                match kind {
                    IllegalMoveOriginKind::Static => {
                        self.infcx.tcx.cannot_move_out_of(span, "static item", origin)
                    }
                    IllegalMoveOriginKind::BorrowedContent { target_place: place } => {
                        // Inspect the type of the content behind the
                        // borrow to provide feedback about why this
                        // was a move rather than a copy.
                        let ty = place.ty(self.mir, self.infcx.tcx).ty;
                        let is_upvar_field_projection =
                            self.prefixes(&original_path, PrefixSet::All)
                            .any(|p| p.is_upvar_field_projection(self.mir, &self.infcx.tcx)
                                 .is_some());
                        debug!("report: ty={:?}", ty);
                        match ty.sty {
                            ty::Array(..) | ty::Slice(..) =>
                                self.infcx.tcx.cannot_move_out_of_interior_noncopy(
                                    span, ty, None, origin
                                ),
                            ty::Closure(def_id, closure_substs)
                                if !self.mir.upvar_decls.is_empty() && is_upvar_field_projection
                            => {
                                let closure_kind_ty =
                                    closure_substs.closure_kind_ty(def_id, self.infcx.tcx);
                                let closure_kind = closure_kind_ty.to_opt_closure_kind();
                                let place_description = match closure_kind {
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
                                debug!("report: closure_kind_ty={:?} closure_kind={:?} \
                                       place_description={:?}", closure_kind_ty, closure_kind,
                                       place_description);

                                let mut diag = self.infcx.tcx.cannot_move_out_of(
                                    span, place_description, origin);

                                for prefix in self.prefixes(&original_path, PrefixSet::All) {
                                    if let Some(field) = prefix.is_upvar_field_projection(
                                            self.mir, &self.infcx.tcx) {
                                        let upvar_decl = &self.mir.upvar_decls[field.index()];
                                        let upvar_hir_id =
                                            upvar_decl.var_hir_id.assert_crate_local();
                                        let upvar_span = self.infcx.tcx.hir().span_by_hir_id(
                                            upvar_hir_id);
                                        diag.span_label(upvar_span, "captured outer variable");
                                        break;
                                    }
                                }

                                diag
                            }
                            _ => {
                                let source = self.borrowed_content_source(place);
                                self.infcx.tcx.cannot_move_out_of(
                                    span, &source.to_string(), origin
                                )
                            },
                        }
                    }
                    IllegalMoveOriginKind::InteriorOfTypeWithDestructor { container_ty: ty } => {
                        self.infcx.tcx
                            .cannot_move_out_of_interior_of_drop(span, ty, origin)
                    }
                    IllegalMoveOriginKind::InteriorOfSliceOrArray { ty, is_index } =>
                        self.infcx.tcx.cannot_move_out_of_interior_noncopy(
                            span, ty, Some(*is_index), origin
                        ),
                },
                span,
            )
        };

        self.add_move_hints(error, &mut err, err_span);
        err.buffer(&mut self.errors_buffer);
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
                let try_remove_deref = match move_from {
                    Place::Projection(box PlaceProjection {
                        elem: ProjectionElem::Deref,
                        ..
                    }) => true,
                    _ => false,
                };
                if try_remove_deref && snippet.starts_with('*') {
                    // The snippet doesn't start with `*` in (e.g.) index
                    // expressions `a[b]`, which roughly desugar to
                    // `*Index::index(&a, b)` or
                    // `*IndexMut::index_mut(&mut a, b)`.
                    err.span_suggestion(
                        span,
                        "consider removing the `*`",
                        snippet[1..].to_owned(),
                        Applicability::Unspecified,
                    );
                } else {
                    err.span_suggestion(
                        span,
                        "consider borrowing here",
                        format!("&{}", snippet),
                        Applicability::Unspecified,
                    );
                }

                binds_to.sort();
                binds_to.dedup();
                self.add_move_error_details(err, &binds_to);
            }
            GroupedMoveError::MovesFromValue { mut binds_to, .. } => {
                binds_to.sort();
                binds_to.dedup();
                self.add_move_error_suggestions(err, &binds_to);
                self.add_move_error_details(err, &binds_to);
            }
            // No binding. Nothing to suggest.
            GroupedMoveError::OtherIllegalMove { .. } => (),
        }
    }

    fn add_move_error_suggestions(
        &self,
        err: &mut DiagnosticBuilder<'a>,
        binds_to: &[Local],
    ) {
        let mut suggestions: Vec<(Span, &str, String)> = Vec::new();
        for local in binds_to {
            let bind_to = &self.mir.local_decls[*local];
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
            let bind_to = &self.mir.local_decls[*local];
            let binding_span = bind_to.source_info.span;

            if j == 0 {
                err.span_label(binding_span, "data moved here");
            } else {
                err.span_label(binding_span, "...and here");
            }

            if binds_to.len() == 1 {
                err.span_note(
                    binding_span,
                    &format!(
                        "move occurs because `{}` has type `{}`, \
                            which does not implement the `Copy` trait",
                        bind_to.name.unwrap(),
                        bind_to.ty
                    ),
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

    fn borrowed_content_source(&self, place: &Place<'tcx>) -> BorrowedContentSource {
        // Look up the provided place and work out the move path index for it,
        // we'll use this to work back through where this value came from and check whether it
        // was originally part of an `Rc` or `Arc`.
        let initial_mpi = match self.move_data.rev_lookup.find(place) {
            LookupResult::Exact(mpi) | LookupResult::Parent(Some(mpi)) => mpi,
            _ => return BorrowedContentSource::Other,
        };

        let mut queue = vec![initial_mpi];
        let mut visited = Vec::new();
        debug!("borrowed_content_source: queue={:?}", queue);
        while let Some(mpi) = queue.pop() {
            debug!(
                "borrowed_content_source: mpi={:?} queue={:?} visited={:?}",
                mpi, queue, visited
            );

            // Don't visit the same path twice.
            if visited.contains(&mpi) {
                continue;
            }
            visited.push(mpi);

            for i in &self.move_data.init_path_map[mpi] {
                let init = &self.move_data.inits[*i];
                debug!("borrowed_content_source: init={:?}", init);
                // We're only interested in statements that initialized a value, not the
                // initializations from arguments.
                let loc = match init.location {
                    InitLocation::Statement(stmt) => stmt,
                    _ => continue,
                };

                let bbd = &self.mir[loc.block];
                let is_terminator = bbd.statements.len() == loc.statement_index;
                debug!("borrowed_content_source: loc={:?} is_terminator={:?}", loc, is_terminator);
                if !is_terminator {
                    let stmt = &bbd.statements[loc.statement_index];
                    debug!("borrowed_content_source: stmt={:?}", stmt);
                    // We're only interested in assignments (in particular, where the
                    // assignment came from - was it an `Rc` or `Arc`?).
                    if let StatementKind::Assign(_, box Rvalue::Ref(_, _, source)) = &stmt.kind {
                        let ty = source.ty(self.mir, self.infcx.tcx).ty;
                        let ty = match ty.sty {
                            ty::TyKind::Ref(_, ty, _) => ty,
                            _ => ty,
                        };
                        debug!("borrowed_content_source: ty={:?}", ty);

                        if ty.is_arc() {
                            return BorrowedContentSource::Arc;
                        } else if ty.is_rc() {
                            return BorrowedContentSource::Rc;
                        } else {
                            queue.push(init.path);
                        }
                    }
                } else if let Some(Terminator {
                    kind: TerminatorKind::Call { args, .. },
                    ..
                }) = &bbd.terminator {
                    for arg in args {
                        let source = match arg {
                            Operand::Copy(place) | Operand::Move(place) => place,
                            _ => continue,
                        };

                        let ty = source.ty(self.mir, self.infcx.tcx).ty;
                        let ty = match ty.sty {
                            ty::TyKind::Ref(_, ty, _) => ty,
                            _ => ty,
                        };
                        debug!("borrowed_content_source: ty={:?}", ty);

                        if ty.is_arc() {
                            return BorrowedContentSource::Arc;
                        } else if ty.is_rc() {
                            return BorrowedContentSource::Rc;
                        } else {
                            queue.push(init.path);
                        }
                    }
                }
            }
        }

        // If we didn't find an `Arc` or an `Rc`, then check specifically for
        // a dereference of a place that has the type of a raw pointer.
        // We can't use `place.ty(..).to_ty(..)` here as that strips away the raw pointer.
        if let Place::Projection(box Projection {
            base,
            elem: ProjectionElem::Deref,
        }) = place {
            if base.ty(self.mir, self.infcx.tcx).ty.is_unsafe_ptr() {
                return BorrowedContentSource::DerefRawPointer;
            }
        }

        BorrowedContentSource::Other
    }
}
