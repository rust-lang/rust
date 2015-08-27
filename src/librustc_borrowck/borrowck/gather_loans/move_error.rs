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
use rustc::middle::mem_categorization::InteriorOffsetKind as Kind;
use rustc::middle::ty;
use std::cell::RefCell;
use syntax::ast;
use syntax::codemap;
use syntax::print::pprust;

pub struct MoveErrorCollector<'tcx> {
    errors: RefCell<Vec<MoveError<'tcx>>>
}

impl<'tcx> MoveErrorCollector<'tcx> {
    pub fn new() -> MoveErrorCollector<'tcx> {
        MoveErrorCollector {
            errors: RefCell::new(Vec::new())
        }
    }

    pub fn add_error(&self, error: MoveError<'tcx>) {
        self.errors.borrow_mut().push(error);
    }

    pub fn report_potential_errors<'a>(&self, bccx: &BorrowckCtxt<'a, 'tcx>) {
        report_move_errors(bccx, &*self.errors.borrow())
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
    pub span: codemap::Span,
    pub ident: ast::Ident
}

pub struct GroupedMoveErrors<'tcx> {
    move_from: mc::cmt<'tcx>,
    move_to_places: Vec<MoveSpanAndPath>
}

fn report_move_errors<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                                errors: &Vec<MoveError<'tcx>>) {
    let grouped_errors = group_errors_with_same_origin(errors);
    for error in &grouped_errors {
        report_cannot_move_out_of(bccx, error.move_from.clone());
        let mut is_first_note = true;
        for move_to in &error.move_to_places {
            note_move_destination(bccx, move_to.span,
                                  &move_to.ident, is_first_note);
            is_first_note = false;
        }
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
            vec!(error.move_to.clone().unwrap())
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
                                       move_from: mc::cmt<'tcx>) {
    match move_from.cat {
        mc::cat_deref(_, _, mc::BorrowedPtr(..)) |
        mc::cat_deref(_, _, mc::Implicit(..)) |
        mc::cat_deref(_, _, mc::UnsafePtr(..)) |
        mc::cat_static_item => {
            bccx.span_err(move_from.span,
                          &format!("cannot move out of {}",
                                  move_from.descriptive_string(bccx.tcx)));
        }

        mc::cat_interior(ref b, mc::InteriorElement(Kind::Index, _)) => {
            let expr = bccx.tcx.map.expect_expr(move_from.id);
            if let ast::ExprIndex(..) = expr.node {
                bccx.span_err(move_from.span,
                              &format!("cannot move out of type `{}`, \
                                        a non-copy fixed-size array",
                                       b.ty));
            }
        }

        mc::cat_downcast(ref b, _) |
        mc::cat_interior(ref b, mc::InteriorField(_)) => {
            match b.ty.sty {
                ty::TyStruct(def, _) |
                ty::TyEnum(def, _) if def.has_dtor() => {
                    bccx.span_err(
                        move_from.span,
                        &format!("cannot move out of type `{}`, \
                                 which defines the `Drop` trait",
                                b.ty));
                },
                _ => {
                    bccx.span_bug(move_from.span, "this path should not cause illegal move")
                }
            }
        }
        _ => {
            bccx.span_bug(move_from.span, "this path should not cause illegal move")
        }
    }
}

fn note_move_destination(bccx: &BorrowckCtxt,
                         move_to_span: codemap::Span,
                         pat_ident: &ast::Ident,
                         is_first_note: bool) {
    let pat_name = pprust::ident_to_string(pat_ident);
    if is_first_note {
        bccx.span_note(
            move_to_span,
            "attempting to move value to here");
        bccx.fileline_help(
            move_to_span,
            &format!("to prevent the move, \
                     use `ref {0}` or `ref mut {0}` to capture value by \
                     reference",
                    pat_name));
    } else {
        bccx.span_note(move_to_span,
                       &format!("and here (use `ref {0}` or `ref mut {0}`)",
                               pat_name));
    }
}
