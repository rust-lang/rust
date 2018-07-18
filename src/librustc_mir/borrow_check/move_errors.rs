// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir;
use rustc::mir::*;
use rustc::ty;
use rustc_data_structures::indexed_vec::Idx;
use rustc_errors::DiagnosticBuilder;
use syntax_pos::Span;

use borrow_check::MirBorrowckCtxt;
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
    // Match place can't be moved from
    // e.g. match x[0] { s => (), } where x: &[String]
    MovesFromMatchPlace {
        span: Span,
        move_from: Place<'tcx>,
        kind: IllegalMoveOriginKind<'tcx>,
        binds_to: Vec<Local>,
    },
    // Part of a pattern can't be moved from,
    // e.g. match &String::new() { &x => (), }
    MovesFromPattern {
        span: Span,
        move_from: MovePathIndex,
        kind: IllegalMoveOriginKind<'tcx>,
        binds_to: Vec<Local>,
    },
    // Everything that isn't from pattern matching.
    OtherIllegalMove {
        span: Span,
        kind: IllegalMoveOriginKind<'tcx>,
    },
}

impl<'a, 'gcx, 'tcx> MirBorrowckCtxt<'a, 'gcx, 'tcx> {
    pub(crate) fn report_move_errors(&self, move_errors: Vec<MoveError<'tcx>>) {
        let grouped_errors = self.group_move_errors(move_errors);
        for error in grouped_errors {
            self.report(error);
        }
    }

    fn group_move_errors(&self, errors: Vec<MoveError<'tcx>>) -> Vec<GroupedMoveError<'tcx>> {
        let mut grouped_errors = Vec::new();
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
        match error {
            MoveError::UnionMove { .. } => {
                unimplemented!("don't know how to report union move errors yet.")
            }
            MoveError::IllegalMove {
                cannot_move_out_of: IllegalMoveOrigin { location, kind },
            } => {
                let stmt_source_info = self.mir.source_info(location);
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
                    }))) = local_decl.is_user_variable
                    {
                        // HACK use scopes to determine if this assignment is
                        // the initialization of a variable.
                        // FIXME(matthewjasper) This would probably be more
                        // reliable if it used the ever initialized dataflow
                        // but move errors are currently reported before the
                        // rest of borrowck has run.
                        if self
                            .mir
                            .is_sub_scope(local_decl.source_info.scope, stmt_source_info.scope)
                        {
                            self.append_binding_error(
                                grouped_errors,
                                kind,
                                move_from,
                                *local,
                                opt_match_place,
                                match_span,
                            );
                            return;
                        }
                    }
                }
                grouped_errors.push(GroupedMoveError::OtherIllegalMove {
                    span: stmt_source_info.span,
                    kind,
                });
            }
        }
    }

    fn append_binding_error(
        &self,
        grouped_errors: &mut Vec<GroupedMoveError<'tcx>>,
        kind: IllegalMoveOriginKind<'tcx>,
        move_from: &Place<'tcx>,
        bind_to: Local,
        match_place: &Option<Place<'tcx>>,
        match_span: Span,
    ) {
        debug!(
            "append_to_grouped_errors(match_place={:?}, match_span={:?})",
            match_place, match_span
        );

        let from_simple_let = match_place.is_none();
        let match_place = match_place.as_ref().unwrap_or(move_from);

        match self.move_data.rev_lookup.find(match_place) {
            // Error with the match place
            LookupResult::Parent(_) => {
                for ge in &mut *grouped_errors {
                    if let GroupedMoveError::MovesFromMatchPlace { span, binds_to, .. } = ge {
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
                let binds_to = if from_simple_let {
                    vec![]
                } else {
                    vec![bind_to]
                };
                grouped_errors.push(GroupedMoveError::MovesFromMatchPlace {
                    span: match_span,
                    move_from: match_place.clone(),
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
                    if let GroupedMoveError::MovesFromPattern {
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
                grouped_errors.push(GroupedMoveError::MovesFromPattern {
                    span: match_span,
                    move_from: mpi,
                    kind,
                    binds_to: vec![bind_to],
                });
            }
        };
    }

    fn report(&self, error: GroupedMoveError<'tcx>) {
        let (mut err, err_span) = {
            let (span, kind): (Span, &IllegalMoveOriginKind) = match error {
                GroupedMoveError::MovesFromMatchPlace { span, ref kind, .. }
                | GroupedMoveError::MovesFromPattern { span, ref kind, .. }
                | GroupedMoveError::OtherIllegalMove { span, ref kind } => (span, kind),
            };
            let origin = Origin::Mir;
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
                        match ty.sty {
                            ty::TyArray(..) | ty::TySlice(..) => self
                                .tcx
                                .cannot_move_out_of_interior_noncopy(span, ty, None, origin),
                            ty::TyClosure(def_id, closure_substs)
                                if !self.mir.upvar_decls.is_empty()
                                    && {
                                        match place {
                                            Place::Projection(ref proj) => {
                                                proj.base == Place::Local(Local::new(1))
                                            }
                                            Place::Local(_) | Place::Static(_) => unreachable!(),
                                        }
                                    } =>
                            {
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
                                self.tcx.cannot_move_out_of(span, place_description, origin)
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
        err.emit();
    }

    fn add_move_hints(
        &self,
        error: GroupedMoveError<'tcx>,
        err: &mut DiagnosticBuilder<'a>,
        span: Span,
    ) {
        match error {
            GroupedMoveError::MovesFromMatchPlace {
                mut binds_to,
                move_from,
                ..
            } => {
                // Ok to suggest a borrow, since the target can't be moved from
                // anyway.
                if let Ok(snippet) = self.tcx.sess.codemap().span_to_snippet(span) {
                    match move_from {
                        Place::Projection(ref proj)
                            if self.suitable_to_remove_deref(proj, &snippet) =>
                        {
                            err.span_suggestion(
                                span,
                                "consider removing this dereference operator",
                                format!("{}", &snippet[1..]),
                            );
                        }
                        _ => {
                            err.span_suggestion(
                                span,
                                "consider using a reference instead",
                                format!("&{}", snippet),
                            );
                        }
                    }

                    binds_to.sort();
                    binds_to.dedup();
                    for local in binds_to {
                        let bind_to = &self.mir.local_decls[local];
                        let binding_span = bind_to.source_info.span;
                        err.span_label(
                            binding_span,
                            format!(
                                "move occurs because {} has type `{}`, \
                                 which does not implement the `Copy` trait",
                                bind_to.name.unwrap(),
                                bind_to.ty
                            ),
                        );
                    }
                }
            }
            GroupedMoveError::MovesFromPattern { mut binds_to, .. } => {
                // Suggest ref, since there might be a move in
                // another match arm
                binds_to.sort();
                binds_to.dedup();
                for local in binds_to {
                    let bind_to = &self.mir.local_decls[local];
                    let binding_span = bind_to.source_info.span;

                    // Suggest ref mut when the user has already written mut.
                    let ref_kind = match bind_to.mutability {
                        Mutability::Not => "ref",
                        Mutability::Mut => "ref mut",
                    };
                    match bind_to.name {
                        Some(name) => {
                            err.span_suggestion(
                                binding_span,
                                "to prevent move, use ref or ref mut",
                                format!("{} {:?}", ref_kind, name),
                            );
                        }
                        None => {
                            err.span_label(
                                span,
                                format!("Local {:?} is not suitable for ref", bind_to),
                            );
                        }
                    }
                }
            }
            // Nothing to suggest.
            GroupedMoveError::OtherIllegalMove { .. } => (),
        }
    }

    fn suitable_to_remove_deref(&self, proj: &PlaceProjection<'tcx>, snippet: &str) -> bool {
        let is_shared_ref = |ty: ty::Ty| match ty.sty {
            ty::TypeVariants::TyRef(.., hir::Mutability::MutImmutable) => true,
            _ => false,
        };

        proj.elem == ProjectionElem::Deref && snippet.starts_with('*') && match proj.base {
            Place::Local(local) => {
                let local_decl = &self.mir.local_decls[local];
                // If this is a temporary, then this could be from an
                // overloaded * operator.
                local_decl.is_user_variable.is_some() && is_shared_ref(local_decl.ty)
            }
            Place::Static(ref st) => is_shared_ref(st.ty),
            Place::Projection(ref proj) => match proj.elem {
                ProjectionElem::Field(_, ty) => is_shared_ref(ty),
                _ => false,
            },
        }
    }
}
