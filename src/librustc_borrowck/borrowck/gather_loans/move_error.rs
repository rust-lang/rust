// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrowck::BorrowckCtxt;
use rustc::middle::mem_categorization as mc;
use rustc::middle::mem_categorization::Categorization;
use rustc::middle::mem_categorization::InteriorOffsetKind as Kind;
use rustc::ty;
use syntax::ast;
use syntax_pos;
use errors::DiagnosticBuilder;

pub struct MoveErrorCollector<'tcx> {
    errors: Vec<MoveError<'tcx>>
}

impl<'tcx> MoveErrorCollector<'tcx> {
    pub fn new() -> MoveErrorCollector<'tcx> {
        MoveErrorCollector {
            errors: Vec::new()
        }
    }

    pub fn add_error(&mut self, error: MoveError<'tcx>) {
        self.errors.push(error);
    }

    pub fn report_potential_errors<'a>(&self, bccx: &BorrowckCtxt<'a, 'tcx>) {
        report_move_errors(bccx, &self.errors)
    }
}

pub struct MoveError<'tcx> {
    move_from: mc::cmt<'tcx>,
    move_to: Option<MoveSpanAndPath>
}

impl<'tcx> MoveError<'tcx> {
    pub fn with_move_info(move_from: mc::cmt<'tcx>,
                          move_to: Option<MoveSpanAndPath>)
                          -> MoveError<'tcx> {
        MoveError {
            move_from: move_from,
            move_to: move_to,
        }
    }
}

#[derive(Clone)]
pub struct MoveSpanAndPath {
    pub span: syntax_pos::Span,
    pub name: ast::Name,
}

pub struct GroupedMoveErrors<'tcx> {
    move_from: mc::cmt<'tcx>,
    move_to_places: Vec<MoveSpanAndPath>
}

fn report_move_errors<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                                errors: &Vec<MoveError<'tcx>>) {
    let grouped_errors = group_errors_with_same_origin(errors);
    for error in &grouped_errors {
        let mut err = report_cannot_move_out_of(bccx, error.move_from.clone());
        let mut is_first_note = true;
        for move_to in &error.move_to_places {
            err = note_move_destination(err, move_to.span,
                                  move_to.name, is_first_note);
            is_first_note = false;
        }
        err.emit();
    }
}

fn group_errors_with_same_origin<'tcx>(errors: &Vec<MoveError<'tcx>>)
                                       -> Vec<GroupedMoveErrors<'tcx>> {
    let mut grouped_errors = Vec::new();
    for error in errors {
        append_to_grouped_errors(&mut grouped_errors, error)
    }
    return grouped_errors;

    fn append_to_grouped_errors<'tcx>(grouped_errors: &mut Vec<GroupedMoveErrors<'tcx>>,
                                      error: &MoveError<'tcx>) {
        let move_from_id = error.move_from.id;
        debug!("append_to_grouped_errors(move_from_id={})", move_from_id);
        let move_to = if error.move_to.is_some() {
            vec![error.move_to.clone().unwrap()]
        } else {
            Vec::new()
        };
        for ge in &mut *grouped_errors {
            if move_from_id == ge.move_from.id && error.move_to.is_some() {
                debug!("appending move_to to list");
                ge.move_to_places.extend(move_to);
                return
            }
        }
        debug!("found a new move from location");
        grouped_errors.push(GroupedMoveErrors {
            move_from: error.move_from.clone(),
            move_to_places: move_to
        })
    }
}

// (keep in sync with gather_moves::check_and_get_illegal_move_origin )
fn report_cannot_move_out_of<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                                       move_from: mc::cmt<'tcx>)
                                       -> DiagnosticBuilder<'a> {
    match move_from.cat {
        Categorization::Deref(.., mc::BorrowedPtr(..)) |
        Categorization::Deref(.., mc::Implicit(..)) |
        Categorization::Deref(.., mc::UnsafePtr(..)) |
        Categorization::StaticItem => {
            let mut err = struct_span_err!(bccx, move_from.span, E0507,
                             "cannot move out of {}",
                             move_from.descriptive_string(bccx.tcx));
            err.span_label(
                move_from.span,
                &format!("cannot move out of {}", move_from.descriptive_string(bccx.tcx))
                );
            err
        }

        Categorization::Interior(ref b, mc::InteriorElement(ik, _)) => {
            match (&b.ty.sty, ik) {
                (&ty::TySlice(..), _) |
                (_, Kind::Index) => {
                    let mut err = struct_span_err!(bccx, move_from.span, E0508,
                                                   "cannot move out of type `{}`, \
                                                    a non-copy array",
                                                   b.ty);
                    err.span_label(move_from.span, &format!("cannot move out of here"));
                    err
                }
                (_, Kind::Pattern) => {
                    span_bug!(move_from.span, "this path should not cause illegal move");
                }
            }
        }

        Categorization::Downcast(ref b, _) |
        Categorization::Interior(ref b, mc::InteriorField(_)) => {
            match b.ty.sty {
                ty::TyAdt(def, _) if def.has_dtor() => {
                    let mut err = struct_span_err!(bccx, move_from.span, E0509,
                                                   "cannot move out of type `{}`, \
                                                   which implements the `Drop` trait",
                                                   b.ty);
                    err.span_label(move_from.span, &format!("cannot move out of here"));
                    err
                },
                _ => {
                    span_bug!(move_from.span, "this path should not cause illegal move");
                }
            }
        }
        _ => {
            span_bug!(move_from.span, "this path should not cause illegal move");
        }
    }
}

fn note_move_destination(mut err: DiagnosticBuilder,
                         move_to_span: syntax_pos::Span,
                         pat_name: ast::Name,
                         is_first_note: bool) -> DiagnosticBuilder {
    if is_first_note {
        err.span_label(
            move_to_span,
            &format!("hint: to prevent move, use `ref {0}` or `ref mut {0}`",
                     pat_name));
        err
    } else {
        err.span_label(move_to_span,
                      &format!("...and here (use `ref {0}` or `ref mut {0}`)",
                               pat_name));
        err
    }
}
