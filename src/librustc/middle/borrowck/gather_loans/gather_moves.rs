// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Computes moves.
 */

use mc = middle::mem_categorization;
use middle::borrowck::*;
use middle::borrowck::move_data::*;
use middle::moves;
use middle::ty;
use syntax::ast;
use syntax::codemap::Span;
use util::ppaux::{Repr, UserString};

pub fn gather_decl(bccx: &BorrowckCtxt,
                   move_data: &MoveData,
                   decl_id: ast::NodeId,
                   _decl_span: Span,
                   var_id: ast::NodeId) {
    let loan_path = @LpVar(var_id);
    move_data.add_move(bccx.tcx, loan_path, decl_id, Declared);
}

pub fn gather_move_from_expr(bccx: &BorrowckCtxt,
                             move_data: &MoveData,
                             move_expr: &ast::Expr,
                             cmt: mc::cmt) {
    gather_move(bccx, move_data, move_expr.id, MoveExpr, cmt);
}

pub fn gather_move_from_pat(bccx: &BorrowckCtxt,
                            move_data: &MoveData,
                            move_pat: &ast::Pat,
                            cmt: mc::cmt) {
    gather_move(bccx, move_data, move_pat.id, MovePat, cmt);
}

pub fn gather_captures(bccx: &BorrowckCtxt,
                       move_data: &MoveData,
                       closure_expr: &ast::Expr) {
    let capture_map = bccx.capture_map.borrow();
    let captured_vars = capture_map.get().get(&closure_expr.id);
    for captured_var in captured_vars.borrow().iter() {
        match captured_var.mode {
            moves::CapMove => {
                let cmt = bccx.cat_captured_var(closure_expr.id,
                                                closure_expr.span,
                                                captured_var);
                gather_move(bccx, move_data, closure_expr.id, Captured, cmt);
            }
            moves::CapCopy | moves::CapRef => {}
        }
    }
}

fn gather_move(bccx: &BorrowckCtxt,
               move_data: &MoveData,
               move_id: ast::NodeId,
               move_kind: MoveKind,
               cmt: mc::cmt) {
    debug!("gather_move(move_id={}, cmt={})",
           move_id, cmt.repr(bccx.tcx));

    if !check_is_legal_to_move_from(bccx, cmt, cmt) {
        return;
    }

    match opt_loan_path(cmt) {
        Some(loan_path) => {
            move_data.add_move(bccx.tcx, loan_path, move_id, move_kind);
        }
        None => {
            // move from rvalue or unsafe pointer, hence ok
        }
    }
}

pub fn gather_assignment(bccx: &BorrowckCtxt,
                         move_data: &MoveData,
                         assignment_id: ast::NodeId,
                         assignment_span: Span,
                         assignee_loan_path: @LoanPath,
                         assignee_id: ast::NodeId) {
    move_data.add_assignment(bccx.tcx,
                             assignee_loan_path,
                             assignment_id,
                             assignment_span,
                             assignee_id);
}

fn check_is_legal_to_move_from(bccx: &BorrowckCtxt,
                               cmt0: mc::cmt,
                               cmt: mc::cmt) -> bool {
    match cmt.cat {
        mc::cat_deref(_, _, mc::BorrowedPtr(..)) |
        mc::cat_deref(_, _, mc::GcPtr) |
        mc::cat_deref(_, _, mc::UnsafePtr(..)) |
        mc::cat_upvar(..) | mc::cat_static_item |
        mc::cat_copied_upvar(mc::CopiedUpvar { onceness: ast::Many, .. }) => {
            bccx.span_err(
                cmt0.span,
                format!("cannot move out of {}",
                        bccx.cmt_to_str(cmt)));
            false
        }

        // Can move out of captured upvars only if the destination closure
        // type is 'once'. 1-shot stack closures emit the copied_upvar form
        // (see mem_categorization.rs).
        mc::cat_copied_upvar(mc::CopiedUpvar { onceness: ast::Once, .. }) => {
            true
        }

        mc::cat_rvalue(..) |
        mc::cat_local(..) |
        mc::cat_arg(..) => {
            true
        }

        mc::cat_downcast(b) |
        mc::cat_interior(b, _) => {
            match ty::get(b.ty).sty {
                ty::ty_struct(did, _) | ty::ty_enum(did, _) => {
                    if ty::has_dtor(bccx.tcx, did) {
                        bccx.span_err(
                            cmt0.span,
                            format!("cannot move out of type `{}`, \
                                  which defines the `Drop` trait",
                                 b.ty.user_string(bccx.tcx)));
                        false
                    } else {
                        check_is_legal_to_move_from(bccx, cmt0, b)
                    }
                }
                _ => {
                    check_is_legal_to_move_from(bccx, cmt0, b)
                }
            }
        }

        mc::cat_deref(b, _, mc::OwnedPtr) |
        mc::cat_discr(b, _) => {
            check_is_legal_to_move_from(bccx, cmt0, b)
        }
    }
}
