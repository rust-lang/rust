//! Computes moves.

use crate::borrowck::*;
use crate::borrowck::move_data::*;
use rustc::middle::mem_categorization as mc;
use rustc::middle::mem_categorization::Categorization;
use rustc::middle::mem_categorization::InteriorOffsetKind as Kind;
use rustc::ty::{self, Ty};

use std::rc::Rc;
use syntax_pos::Span;
use log::debug;

struct GatherMoveInfo<'c, 'tcx> {
    id: hir::ItemLocalId,
    cmt: &'c mc::cmt_<'tcx>,
}

pub fn gather_decl<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                             move_data: &MoveData<'tcx>,
                             var_id: hir::HirId,
                             var_ty: Ty<'tcx>) {
    let loan_path = Rc::new(LoanPath::new(LpVar(var_id), var_ty));
    move_data.add_move(bccx.tcx, loan_path, var_id.local_id);
}

pub fn gather_move_from_expr<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                                       move_data: &MoveData<'tcx>,
                                       move_expr_id: hir::ItemLocalId,
                                       cmt: &mc::cmt_<'tcx>) {
    let move_info = GatherMoveInfo {
        id: move_expr_id,
        cmt,
    };
    gather_move(bccx, move_data, move_info);
}

pub fn gather_move_from_pat<'a, 'c, 'tcx>(
    bccx: &BorrowckCtxt<'a, 'tcx>,
    move_data: &MoveData<'tcx>,
    move_pat: &hir::Pat,
    cmt: &'c mc::cmt_<'tcx>,
) {
    let move_info = GatherMoveInfo {
        id: move_pat.hir_id.local_id,
        cmt,
    };

    debug!("gather_move_from_pat: move_pat={:?}", move_pat);

    gather_move(bccx, move_data, move_info);
}

fn gather_move<'a, 'c, 'tcx>(
    bccx: &BorrowckCtxt<'a, 'tcx>,
    move_data: &MoveData<'tcx>,
    move_info: GatherMoveInfo<'c, 'tcx>,
) {
    debug!("gather_move(move_id={:?}, cmt={:?})",
           move_info.id, move_info.cmt);

    let potentially_illegal_move = check_and_get_illegal_move_origin(bccx, move_info.cmt);
    if let Some(_) = potentially_illegal_move {
        bccx.signal_error();
        return;
    }

    match opt_loan_path(&move_info.cmt) {
        Some(loan_path) => {
            move_data.add_move(bccx.tcx, loan_path,
                               move_info.id);
        }
        None => {
            // move from rvalue or raw pointer, hence ok
        }
    }
}

pub fn gather_assignment<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                                   move_data: &MoveData<'tcx>,
                                   assignment_id: hir::ItemLocalId,
                                   assignment_span: Span,
                                   assignee_loan_path: Rc<LoanPath<'tcx>>) {
    move_data.add_assignment(bccx.tcx,
                             assignee_loan_path,
                             assignment_id,
                             assignment_span);
}

// (keep in sync with move_error::report_cannot_move_out_of )
fn check_and_get_illegal_move_origin<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                                               cmt: &mc::cmt_<'tcx>)
                                               -> Option<mc::cmt_<'tcx>> {
    match cmt.cat {
        Categorization::Deref(_, mc::BorrowedPtr(..)) |
        Categorization::Deref(_, mc::UnsafePtr(..)) |
        Categorization::ThreadLocal(..) |
        Categorization::StaticItem => {
            Some(cmt.clone())
        }

        Categorization::Rvalue(..) |
        Categorization::Local(..) |
        Categorization::Upvar(..) => {
            None
        }

        Categorization::Downcast(ref b, _) |
        Categorization::Interior(ref b, mc::InteriorField(_)) |
        Categorization::Interior(ref b, mc::InteriorElement(Kind::Pattern)) => {
            match b.ty.kind {
                ty::Adt(def, _) => {
                    if def.has_dtor(bccx.tcx) {
                        Some(cmt.clone())
                    } else {
                        check_and_get_illegal_move_origin(bccx, b)
                    }
                }
                ty::Slice(..) => Some(cmt.clone()),
                _ => {
                    check_and_get_illegal_move_origin(bccx, b)
                }
            }
        }

        Categorization::Interior(_, mc::InteriorElement(Kind::Index)) => {
            // Forbid move of arr[i] for arr: [T; 3]; see RFC 533.
            Some(cmt.clone())
        }

        Categorization::Deref(ref b, mc::Unique) => {
            check_and_get_illegal_move_origin(bccx, b)
        }
    }
}
