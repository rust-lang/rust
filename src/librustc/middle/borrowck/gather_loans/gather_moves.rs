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
use middle::borrowck::gather_loans::move_error::{MoveError, MoveErrorCollector};
use middle::borrowck::gather_loans::move_error::MoveSpanAndPath;
use middle::borrowck::move_data::*;
use euv = middle::expr_use_visitor;
use middle::ty;
use syntax::ast;
use syntax::codemap::Span;
use util::ppaux::Repr;

use std::rc::Rc;

struct GatherMoveInfo {
    id: ast::NodeId,
    kind: MoveKind,
    cmt: mc::cmt,
    span_path_opt: Option<MoveSpanAndPath>
}

pub fn gather_decl(bccx: &BorrowckCtxt,
                   move_data: &MoveData,
                   decl_id: ast::NodeId,
                   _decl_span: Span,
                   var_id: ast::NodeId) {
    let loan_path = Rc::new(LpVar(var_id));
    move_data.add_move(bccx.tcx, loan_path, decl_id, Declared);
}

pub fn gather_move_from_expr(bccx: &BorrowckCtxt,
                             move_data: &MoveData,
                             move_error_collector: &MoveErrorCollector,
                             move_expr_id: ast::NodeId,
                             cmt: mc::cmt,
                             move_reason: euv::MoveReason) {
    let kind = match move_reason {
        euv::DirectRefMove | euv::PatBindingMove => MoveExpr,
        euv::CaptureMove => Captured
    };
    let move_info = GatherMoveInfo {
        id: move_expr_id,
        kind: kind,
        cmt: cmt,
        span_path_opt: None,
    };
    gather_move(bccx, move_data, move_error_collector, move_info);
}

pub fn gather_move_from_pat(bccx: &BorrowckCtxt,
                            move_data: &MoveData,
                            move_error_collector: &MoveErrorCollector,
                            move_pat: &ast::Pat,
                            cmt: mc::cmt) {
    let pat_span_path_opt = match move_pat.node {
        ast::PatIdent(_, ref path1, _) => {
            Some(MoveSpanAndPath{span: move_pat.span,
                                 ident: path1.node})
        },
        _ => None,
    };
    let move_info = GatherMoveInfo {
        id: move_pat.id,
        kind: MovePat,
        cmt: cmt,
        span_path_opt: pat_span_path_opt,
    };
    gather_move(bccx, move_data, move_error_collector, move_info);
}

fn gather_move(bccx: &BorrowckCtxt,
               move_data: &MoveData,
               move_error_collector: &MoveErrorCollector,
               move_info: GatherMoveInfo) {
    debug!("gather_move(move_id={}, cmt={})",
           move_info.id, move_info.cmt.repr(bccx.tcx));

    let potentially_illegal_move =
                check_and_get_illegal_move_origin(bccx, &move_info.cmt);
    match potentially_illegal_move {
        Some(illegal_move_origin) => {
            debug!("illegal_move_origin={}", illegal_move_origin.repr(bccx.tcx));
            let error = MoveError::with_move_info(illegal_move_origin,
                                                  move_info.span_path_opt);
            move_error_collector.add_error(error);
            return
        }
        None => ()
    }

    match opt_loan_path(&move_info.cmt) {
        Some(loan_path) => {
            move_data.add_move(bccx.tcx, loan_path,
                               move_info.id, move_info.kind);
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
                         assignee_loan_path: Rc<LoanPath>,
                         assignee_id: ast::NodeId,
                         mode: euv::MutateMode) {
    move_data.add_assignment(bccx.tcx,
                             assignee_loan_path,
                             assignment_id,
                             assignment_span,
                             assignee_id,
                             mode);
}

fn check_and_get_illegal_move_origin(bccx: &BorrowckCtxt,
                                     cmt: &mc::cmt) -> Option<mc::cmt> {
    match cmt.cat {
        mc::cat_deref(_, _, mc::BorrowedPtr(..)) |
        mc::cat_deref(_, _, mc::GcPtr) |
        mc::cat_deref(_, _, mc::UnsafePtr(..)) |
        mc::cat_upvar(..) | mc::cat_static_item |
        mc::cat_copied_upvar(mc::CopiedUpvar { onceness: ast::Many, .. }) => {
            Some(cmt.clone())
        }

        // Can move out of captured upvars only if the destination closure
        // type is 'once'. 1-shot stack closures emit the copied_upvar form
        // (see mem_categorization.rs).
        mc::cat_copied_upvar(mc::CopiedUpvar { onceness: ast::Once, .. }) => {
            None
        }

        mc::cat_rvalue(..) |
        mc::cat_local(..) |
        mc::cat_arg(..) => {
            None
        }

        mc::cat_downcast(ref b) |
        mc::cat_interior(ref b, _) => {
            match ty::get(b.ty).sty {
                ty::ty_struct(did, _) | ty::ty_enum(did, _) => {
                    if ty::has_dtor(bccx.tcx, did) {
                        Some(cmt.clone())
                    } else {
                        check_and_get_illegal_move_origin(bccx, b)
                    }
                }
                _ => {
                    check_and_get_illegal_move_origin(bccx, b)
                }
            }
        }

        mc::cat_deref(ref b, _, mc::OwnedPtr) |
        mc::cat_discr(ref b, _) => {
            check_and_get_illegal_move_origin(bccx, b)
        }
    }
}
