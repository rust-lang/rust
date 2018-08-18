// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::unicode::property::Pattern_White_Space;
use rustc::mir::*;
use rustc::ty;
use rustc_errors::DiagnosticBuilder;
use rustc_data_structures::indexed_vec::Idx;
use syntax_pos::Span;

use borrow_check::MirBorrowckCtxt;
use borrow_check::prefixes::PrefixSet;
use dataflow::move_paths::{IllegalMoveOrigin, IllegalMoveOriginKind};
use dataflow::move_paths::{LookupResult, MoveError, MovePathIndex};
use util::borrowck_errors::{BorrowckErrors, Origin};

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
    // e.g. match x[0] { s => (), } where x: &[String]
    MovesFromPlace {
        original_path: Place<'tcx>,
        span: Span,
        move_from: Place<'tcx>,
        kind: IllegalMoveOriginKind<'tcx>,
        binds_to: Vec<Local>,
    },
    // Part of a value expression can't be moved from,
    // e.g. match &String::new() { &x => (), }
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
                    Place::Local(local),
                    Rvalue::Use(Operand::Move(move_from)),
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
            let (span, original_path, kind): (Span, &Place<'tcx>, &IllegalMoveOriginKind) =
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
                   original_path.is_upvar_field_projection(self.mir, &self.tcx));
            (
                match kind {
                    IllegalMoveOriginKind::Static => {
                        self.tcx.cannot_move_out_of(span, "static item", origin)
                    }
                    IllegalMoveOriginKind::BorrowedContent { target_place: place } => {
                        // Inspect the type of the content behind the
                        // borrow to provide feedback about why this
                        // was a move rather than a copy.
                        let ty = place.ty(self.mir, self.tcx).to_ty(self.tcx);
                        let is_upvar_field_projection =
                            self.prefixes(&original_path, PrefixSet::All)
                            .any(|p| p.is_upvar_field_projection(self.mir, &self.tcx)
                                 .is_some());
                        match ty.sty {
                            ty::TyArray(..) | ty::TySlice(..) => self
                                .tcx
                                .cannot_move_out_of_interior_noncopy(span, ty, None, origin),
                            ty::TyClosure(def_id, closure_substs)
                                if !self.mir.upvar_decls.is_empty() && is_upvar_field_projection
                            => {
                                let closure_kind_ty =
                                    closure_substs.closure_kind_ty(def_id, self.tcx);
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

                                let mut diag = self.tcx.cannot_move_out_of(
                                    span, place_description, origin);

                                for prefix in self.prefixes(&original_path, PrefixSet::All) {
                                    if let Some(field) = prefix.is_upvar_field_projection(
                                            self.mir, &self.tcx) {
                                        let upvar_decl = &self.mir.upvar_decls[field.index()];
                                        let upvar_hir_id =
                                            upvar_decl.var_hir_id.assert_crate_local();
                                        let upvar_node_id =
                                            self.tcx.hir.hir_to_node_id(upvar_hir_id);
                                        let upvar_span = self.tcx.hir.span(upvar_node_id);
                                        diag.span_label(upvar_span, "captured outer variable");
                                        break;
                                    }
                                }

                                diag
                            }
                            _ => self
                                .tcx
                                .cannot_move_out_of(span, "borrowed content", origin),
                        }
                    }
                    IllegalMoveOriginKind::InteriorOfTypeWithDestructor { container_ty: ty } => {
                        self.tcx
                            .cannot_move_out_of_interior_of_drop(span, ty, origin)
                    }
                    IllegalMoveOriginKind::InteriorOfSliceOrArray { ty, is_index } => self
                        .tcx
                        .cannot_move_out_of_interior_noncopy(span, ty, Some(*is_index), origin),
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
        let snippet = self.tcx.sess.source_map().span_to_snippet(span).unwrap();
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
                    );
                } else {
                    err.span_suggestion(
                        span,
                        "consider borrowing here",
                        format!("&{}", snippet),
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
                let pat_snippet = self
                    .tcx.sess.source_map()
                    .span_to_snippet(pat_span)
                    .unwrap();
                if pat_snippet.starts_with('&') {
                    let pat_snippet = pat_snippet[1..].trim_left();
                    let suggestion;
                    let to_remove;
                    if pat_snippet.starts_with("mut")
                        && pat_snippet["mut".len()..].starts_with(Pattern_White_Space)
                    {
                        suggestion = pat_snippet["mut".len()..].trim_left();
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
                suggestion
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
                err.span_label(binding_span, format!("data moved here"));
            } else {
                err.span_label(binding_span, format!("...and here"));
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
}
