//! Computes moves.

use crate::borrowck::*;
use crate::borrowck::gather_loans::move_error::MovePlace;
use crate::borrowck::gather_loans::move_error::{MoveError, MoveErrorCollector};
use crate::borrowck::move_data::*;
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::mem_categorization as mc;
use rustc::middle::mem_categorization::Categorization;
use rustc::middle::mem_categorization::InteriorOffsetKind as Kind;
use rustc::ty::{self, Ty};

use std::rc::Rc;
use syntax_pos::Span;
use rustc::hir::*;
use rustc::hir::Node;
use log::debug;

struct GatherMoveInfo<'c, 'tcx> {
    id: hir::ItemLocalId,
    kind: MoveKind,
    cmt: &'c mc::cmt_<'tcx>,
    span_path_opt: Option<MovePlace<'tcx>>,
}

/// Represents the kind of pattern
#[derive(Debug, Clone, Copy)]
pub enum PatternSource<'tcx> {
    MatchExpr(&'tcx Expr),
    LetDecl(&'tcx Local),
    Other,
}

/// Analyzes the context where the pattern appears to determine the
/// kind of hint we want to give. In particular, if the pattern is in a `match`
/// or nested within other patterns, we want to suggest a `ref` binding:
///
///     let (a, b) = v[0]; // like the `a` and `b` patterns here
///     match v[0] { a => ... } // or the `a` pattern here
///
/// But if the pattern is the outermost pattern in a `let`, we would rather
/// suggest that the author add a `&` to the initializer:
///
///     let x = v[0]; // suggest `&v[0]` here
///
/// In this latter case, this function will return `PatternSource::LetDecl`
/// with a reference to the let
fn get_pattern_source<'tcx>(tcx: TyCtxt<'tcx>, pat: &Pat) -> PatternSource<'tcx> {

    let parent = tcx.hir().get_parent_node(pat.hir_id);

    match tcx.hir().get(parent) {
        Node::Expr(ref e) => {
            // the enclosing expression must be a `match` or something else
            assert!(match e.node {
                        ExprKind::Match(..) => true,
                        _ => return PatternSource::Other,
                    });
            PatternSource::MatchExpr(e)
        }
        Node::Local(local) => PatternSource::LetDecl(local),
        _ => return PatternSource::Other,

    }
}

pub fn gather_decl<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                             move_data: &MoveData<'tcx>,
                             var_id: hir::HirId,
                             var_ty: Ty<'tcx>) {
    let loan_path = Rc::new(LoanPath::new(LpVar(var_id), var_ty));
    move_data.add_move(bccx.tcx, loan_path, var_id.local_id, Declared);
}

pub fn gather_move_from_expr<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                                       move_data: &MoveData<'tcx>,
                                       move_error_collector: &mut MoveErrorCollector<'tcx>,
                                       move_expr_id: hir::ItemLocalId,
                                       cmt: &mc::cmt_<'tcx>,
                                       move_reason: euv::MoveReason) {
    let kind = match move_reason {
        euv::DirectRefMove | euv::PatBindingMove => MoveExpr,
        euv::CaptureMove => Captured
    };
    let move_info = GatherMoveInfo {
        id: move_expr_id,
        kind,
        cmt,
        span_path_opt: None,
    };
    gather_move(bccx, move_data, move_error_collector, move_info);
}

pub fn gather_move_from_pat<'a, 'c, 'tcx>(
    bccx: &BorrowckCtxt<'a, 'tcx>,
    move_data: &MoveData<'tcx>,
    move_error_collector: &mut MoveErrorCollector<'tcx>,
    move_pat: &hir::Pat,
    cmt: &'c mc::cmt_<'tcx>,
) {
    let source = get_pattern_source(bccx.tcx,move_pat);
    let pat_span_path_opt = match move_pat.node {
        PatKind::Binding(_, _, ident, _) => {
            Some(MovePlace {
                     span: move_pat.span,
                     name: ident.name,
                     pat_source: source,
                 })
        }
        _ => None,
    };
    let move_info = GatherMoveInfo {
        id: move_pat.hir_id.local_id,
        kind: MovePat,
        cmt,
        span_path_opt: pat_span_path_opt,
    };

    debug!("gather_move_from_pat: move_pat={:?} source={:?}",
           move_pat,
           source);

    gather_move(bccx, move_data, move_error_collector, move_info);
}

fn gather_move<'a, 'c, 'tcx>(
    bccx: &BorrowckCtxt<'a, 'tcx>,
    move_data: &MoveData<'tcx>,
    move_error_collector: &mut MoveErrorCollector<'tcx>,
    move_info: GatherMoveInfo<'c, 'tcx>,
) {
    debug!("gather_move(move_id={:?}, cmt={:?})",
           move_info.id, move_info.cmt);

    let potentially_illegal_move = check_and_get_illegal_move_origin(bccx, move_info.cmt);
    if let Some(illegal_move_origin) = potentially_illegal_move {
        debug!("illegal_move_origin={:?}", illegal_move_origin);
        let error = MoveError::with_move_info(Rc::new(illegal_move_origin),
                                              move_info.span_path_opt);
        move_error_collector.add_error(error);
        return;
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
            match b.ty.sty {
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
