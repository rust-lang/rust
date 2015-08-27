// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Computes moves.

use borrowck::*;
use borrowck::gather_loans::move_error::MoveSpanAndPath;
use borrowck::gather_loans::move_error::{MoveError, MoveErrorCollector};
use borrowck::move_data::*;
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::mem_categorization as mc;
use rustc::middle::mem_categorization::InteriorOffsetKind as Kind;
use rustc::middle::ty;

use std::rc::Rc;
use syntax::ast;
use syntax::codemap::Span;

struct GatherMoveInfo<'tcx> {
    id: ast::NodeId,
    kind: MoveKind,
    cmt: mc::cmt<'tcx>,
    span_path_opt: Option<MoveSpanAndPath>
}

pub fn gather_decl<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                             move_data: &MoveData<'tcx>,
                             decl_id: ast::NodeId,
                             _decl_span: Span,
                             var_id: ast::NodeId) {
    let ty = bccx.tcx.node_id_to_type(var_id);
    let loan_path = Rc::new(LoanPath::new(LpVar(var_id), ty));
    move_data.add_move(bccx.tcx, loan_path, decl_id, Declared);
}

pub fn gather_move_from_expr<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                                       move_data: &MoveData<'tcx>,
                                       move_error_collector: &MoveErrorCollector<'tcx>,
                                       move_expr_id: ast::NodeId,
                                       cmt: mc::cmt<'tcx>,
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

pub fn gather_match_variant<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                                      move_data: &MoveData<'tcx>,
                                      _move_error_collector: &MoveErrorCollector<'tcx>,
                                      move_pat: &ast::Pat,
                                      cmt: mc::cmt<'tcx>,
                                      mode: euv::MatchMode) {
    let tcx = bccx.tcx;
    debug!("gather_match_variant(move_pat={}, cmt={:?}, mode={:?})",
           move_pat.id, cmt, mode);

    let opt_lp = opt_loan_path(&cmt);
    match opt_lp {
        Some(lp) => {
            match lp.kind {
                LpDowncast(ref base_lp, _) =>
                    move_data.add_variant_match(
                        tcx, lp.clone(), move_pat.id, base_lp.clone(), mode),
                _ => panic!("should only call gather_match_variant \
                             for cat_downcast cmt"),
            }
        }
        None => {
            // We get None when input to match is non-path (e.g.
            // temporary result like a function call). Since no
            // loan-path is being matched, no need to record a
            // downcast.
            return;
        }
    }
}

pub fn gather_move_from_pat<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                                      move_data: &MoveData<'tcx>,
                                      move_error_collector: &MoveErrorCollector<'tcx>,
                                      move_pat: &ast::Pat,
                                      cmt: mc::cmt<'tcx>) {
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

fn gather_move<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                         move_data: &MoveData<'tcx>,
                         move_error_collector: &MoveErrorCollector<'tcx>,
                         move_info: GatherMoveInfo<'tcx>) {
    debug!("gather_move(move_id={}, cmt={:?})",
           move_info.id, move_info.cmt);

    let potentially_illegal_move =
                check_and_get_illegal_move_origin(bccx, &move_info.cmt);
    match potentially_illegal_move {
        Some(illegal_move_origin) => {
            debug!("illegal_move_origin={:?}", illegal_move_origin);
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
            // move from rvalue or raw pointer, hence ok
        }
    }
}

pub fn gather_assignment<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                                   move_data: &MoveData<'tcx>,
                                   assignment_id: ast::NodeId,
                                   assignment_span: Span,
                                   assignee_loan_path: Rc<LoanPath<'tcx>>,
                                   assignee_id: ast::NodeId,
                                   mode: euv::MutateMode) {
    move_data.add_assignment(bccx.tcx,
                             assignee_loan_path,
                             assignment_id,
                             assignment_span,
                             assignee_id,
                             mode);
}

// (keep in sync with move_error::report_cannot_move_out_of )
fn check_and_get_illegal_move_origin<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                                               cmt: &mc::cmt<'tcx>)
                                               -> Option<mc::cmt<'tcx>> {
    match cmt.cat {
        mc::cat_deref(_, _, mc::BorrowedPtr(..)) |
        mc::cat_deref(_, _, mc::Implicit(..)) |
        mc::cat_deref(_, _, mc::UnsafePtr(..)) |
        mc::cat_static_item => {
            Some(cmt.clone())
        }

        mc::cat_rvalue(..) |
        mc::cat_local(..) |
        mc::cat_upvar(..) => {
            None
        }

        mc::cat_downcast(ref b, _) |
        mc::cat_interior(ref b, mc::InteriorField(_)) |
        mc::cat_interior(ref b, mc::InteriorElement(Kind::Pattern, _)) => {
            match b.ty.sty {
                ty::TyStruct(def, _) | ty::TyEnum(def, _) => {
                    if def.has_dtor() {
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

        mc::cat_interior(_, mc::InteriorElement(Kind::Index, _)) => {
            // Forbid move of arr[i] for arr: [T; 3]; see RFC 533.
            Some(cmt.clone())
        }

        mc::cat_deref(ref b, _, mc::Unique) => {
            check_and_get_illegal_move_origin(bccx, b)
        }
    }
}
