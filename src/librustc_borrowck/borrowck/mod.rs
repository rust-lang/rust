// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! See The Book chapter on the borrow checker for more details.

#![allow(non_camel_case_types)]

pub use self::LoanPathKind::*;
pub use self::LoanPathElem::*;
pub use self::bckerr_code::*;
pub use self::AliasableViolationKind::*;
pub use self::MovedValueUseKind::*;

use self::InteriorKind::*;

use rustc::middle::cfg;
use rustc::middle::dataflow::DataFlowContext;
use rustc::middle::dataflow::BitwiseOperator;
use rustc::middle::dataflow::DataFlowOperator;
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::mem_categorization as mc;
use rustc::middle::region;
use rustc::middle::ty::{self, Ty};
use rustc::util::ppaux::{note_and_explain_region, Repr, UserString};
use std::rc::Rc;
use std::string::String;
use syntax::ast;
use syntax::ast_map;
use syntax::ast_map::blocks::{FnLikeNode, FnParts};
use syntax::ast_util;
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::visit;
use syntax::visit::{Visitor, FnKind};
use syntax::ast::{FnDecl, Block, NodeId};

pub mod check_loans;

pub mod gather_loans;

pub mod move_data;

#[derive(Clone, Copy)]
pub struct LoanDataFlowOperator;

pub type LoanDataFlow<'a, 'tcx> = DataFlowContext<'a, 'tcx, LoanDataFlowOperator>;

impl<'a, 'tcx, 'v> Visitor<'v> for BorrowckCtxt<'a, 'tcx> {
    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v FnDecl,
                b: &'v Block, s: Span, id: ast::NodeId) {
        borrowck_fn(self, fk, fd, b, s, id);
    }

    fn visit_item(&mut self, item: &ast::Item) {
        borrowck_item(self, item);
    }
}

pub fn check_crate(tcx: &ty::ctxt) {
    let mut bccx = BorrowckCtxt {
        tcx: tcx,
        stats: BorrowStats {
            loaned_paths_same: 0,
            loaned_paths_imm: 0,
            stable_paths: 0,
            guaranteed_paths: 0
        }
    };

    visit::walk_crate(&mut bccx, tcx.map.krate());

    if tcx.sess.borrowck_stats() {
        println!("--- borrowck stats ---");
        println!("paths requiring guarantees: {}",
                 bccx.stats.guaranteed_paths);
        println!("paths requiring loans     : {}",
                 make_stat(&bccx, bccx.stats.loaned_paths_same));
        println!("paths requiring imm loans : {}",
                 make_stat(&bccx, bccx.stats.loaned_paths_imm));
        println!("stable paths              : {}",
                 make_stat(&bccx, bccx.stats.stable_paths));
    }

    fn make_stat(bccx: &BorrowckCtxt, stat: uint) -> String {
        let total = bccx.stats.guaranteed_paths as f64;
        let perc = if total == 0.0 { 0.0 } else { stat as f64 * 100.0 / total };
        format!("{} ({:.0}%)", stat, perc)
    }
}

fn borrowck_item(this: &mut BorrowckCtxt, item: &ast::Item) {
    // Gather loans for items. Note that we don't need
    // to check loans for single expressions. The check
    // loan step is intended for things that have a data
    // flow dependent conditions.
    match item.node {
        ast::ItemStatic(_, _, ref ex) |
        ast::ItemConst(_, ref ex) => {
            gather_loans::gather_loans_in_static_initializer(this, &**ex);
        }
        _ => {
            visit::walk_item(this, item);
        }
    }
}

/// Collection of conclusions determined via borrow checker analyses.
pub struct AnalysisData<'a, 'tcx: 'a> {
    pub all_loans: Vec<Loan<'tcx>>,
    pub loans: DataFlowContext<'a, 'tcx, LoanDataFlowOperator>,
    pub move_data: move_data::FlowedMoveData<'a, 'tcx>,
}

fn borrowck_fn(this: &mut BorrowckCtxt,
               fk: FnKind,
               decl: &ast::FnDecl,
               body: &ast::Block,
               sp: Span,
               id: ast::NodeId) {
    debug!("borrowck_fn(id={})", id);
    let cfg = cfg::CFG::new(this.tcx, body);
    let AnalysisData { all_loans,
                       loans: loan_dfcx,
                       move_data:flowed_moves } =
        build_borrowck_dataflow_data(this, fk, decl, &cfg, body, sp, id);

    move_data::fragments::instrument_move_fragments(&flowed_moves.move_data,
                                                    this.tcx, sp, id);

    check_loans::check_loans(this,
                             &loan_dfcx,
                             flowed_moves,
                             &all_loans[..],
                             id,
                             decl,
                             body);

    visit::walk_fn(this, fk, decl, body, sp);
}

fn build_borrowck_dataflow_data<'a, 'tcx>(this: &mut BorrowckCtxt<'a, 'tcx>,
                                          fk: FnKind,
                                          decl: &ast::FnDecl,
                                          cfg: &cfg::CFG,
                                          body: &ast::Block,
                                          sp: Span,
                                          id: ast::NodeId) -> AnalysisData<'a, 'tcx> {
    // Check the body of fn items.
    let id_range = ast_util::compute_id_range_for_fn_body(fk, decl, body, sp, id);
    let (all_loans, move_data) =
        gather_loans::gather_loans_in_fn(this, id, decl, body);

    let mut loan_dfcx =
        DataFlowContext::new(this.tcx,
                             "borrowck",
                             Some(decl),
                             cfg,
                             LoanDataFlowOperator,
                             id_range,
                             all_loans.len());
    for (loan_idx, loan) in all_loans.iter().enumerate() {
        loan_dfcx.add_gen(loan.gen_scope.node_id(), loan_idx);
        loan_dfcx.add_kill(loan.kill_scope.node_id(), loan_idx);
    }
    loan_dfcx.add_kills_from_flow_exits(cfg);
    loan_dfcx.propagate(cfg, body);

    let flowed_moves = move_data::FlowedMoveData::new(move_data,
                                                      this.tcx,
                                                      cfg,
                                                      id_range,
                                                      decl,
                                                      body);

    AnalysisData { all_loans: all_loans,
                   loans: loan_dfcx,
                   move_data:flowed_moves }
}

/// This and a `ty::ctxt` is all you need to run the dataflow analyses
/// used in the borrow checker.
pub struct FnPartsWithCFG<'a> {
    pub fn_parts: FnParts<'a>,
    pub cfg:  &'a cfg::CFG,
}

impl<'a> FnPartsWithCFG<'a> {
    pub fn from_fn_like(f: &'a FnLikeNode,
                        g: &'a cfg::CFG) -> FnPartsWithCFG<'a> {
        FnPartsWithCFG { fn_parts: f.to_fn_parts(), cfg: g }
    }
}

/// Accessor for introspective clients inspecting `AnalysisData` and
/// the `BorrowckCtxt` itself , e.g. the flowgraph visualizer.
pub fn build_borrowck_dataflow_data_for_fn<'a, 'tcx>(
    tcx: &'a ty::ctxt<'tcx>,
    input: FnPartsWithCFG<'a>) -> (BorrowckCtxt<'a, 'tcx>, AnalysisData<'a, 'tcx>) {

    let mut bccx = BorrowckCtxt {
        tcx: tcx,
        stats: BorrowStats {
            loaned_paths_same: 0,
            loaned_paths_imm: 0,
            stable_paths: 0,
            guaranteed_paths: 0
        }
    };

    let p = input.fn_parts;

    let dataflow_data = build_borrowck_dataflow_data(&mut bccx,
                                                     p.kind,
                                                     &*p.decl,
                                                     input.cfg,
                                                     &*p.body,
                                                     p.span,
                                                     p.id);

    (bccx, dataflow_data)
}

// ----------------------------------------------------------------------
// Type definitions

pub struct BorrowckCtxt<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,

    // Statistics:
    stats: BorrowStats
}

struct BorrowStats {
    loaned_paths_same: uint,
    loaned_paths_imm: uint,
    stable_paths: uint,
    guaranteed_paths: uint
}

pub type BckResult<'tcx, T> = Result<T, BckError<'tcx>>;

///////////////////////////////////////////////////////////////////////////
// Loans and loan paths

/// Record of a loan that was issued.
pub struct Loan<'tcx> {
    index: uint,
    loan_path: Rc<LoanPath<'tcx>>,
    kind: ty::BorrowKind,
    restricted_paths: Vec<Rc<LoanPath<'tcx>>>,

    /// gen_scope indicates where loan is introduced. Typically the
    /// loan is introduced at the point of the borrow, but in some
    /// cases, notably method arguments, the loan may be introduced
    /// only later, once it comes into scope.  See also
    /// `GatherLoanCtxt::compute_gen_scope`.
    gen_scope: region::CodeExtent,

    /// kill_scope indicates when the loan goes out of scope.  This is
    /// either when the lifetime expires or when the local variable
    /// which roots the loan-path goes out of scope, whichever happens
    /// faster. See also `GatherLoanCtxt::compute_kill_scope`.
    kill_scope: region::CodeExtent,
    span: Span,
    cause: euv::LoanCause,
}

impl<'tcx> Loan<'tcx> {
    pub fn loan_path(&self) -> Rc<LoanPath<'tcx>> {
        self.loan_path.clone()
    }
}

#[derive(Eq, Hash, Debug)]
pub struct LoanPath<'tcx> {
    kind: LoanPathKind<'tcx>,
    ty: ty::Ty<'tcx>,
}

impl<'tcx> PartialEq for LoanPath<'tcx> {
    fn eq(&self, that: &LoanPath<'tcx>) -> bool {
        let r = self.kind == that.kind;
        debug_assert!(self.ty == that.ty || !r,
                      "Somehow loan paths are equal though their tys are not.");
        r
    }
}

#[derive(PartialEq, Eq, Hash, Debug)]
pub enum LoanPathKind<'tcx> {
    LpVar(ast::NodeId),                         // `x` in doc.rs
    LpUpvar(ty::UpvarId),                       // `x` captured by-value into closure
    LpDowncast(Rc<LoanPath<'tcx>>, ast::DefId), // `x` downcast to particular enum variant
    LpExtend(Rc<LoanPath<'tcx>>, mc::MutabilityCategory, LoanPathElem)
}

impl<'tcx> LoanPath<'tcx> {
    fn new(kind: LoanPathKind<'tcx>, ty: ty::Ty<'tcx>) -> LoanPath<'tcx> {
        LoanPath { kind: kind, ty: ty }
    }

    fn to_type(&self) -> ty::Ty<'tcx> { self.ty }
}

// FIXME (pnkfelix): See discussion here
// https://github.com/pnkfelix/rust/commit/
//     b2b39e8700e37ad32b486b9a8409b50a8a53aa51#commitcomment-7892003
static DOWNCAST_PRINTED_OPERATOR : &'static str = " as ";

// A local, "cleaned" version of `mc::InteriorKind` that drops
// information that is not relevant to loan-path analysis. (In
// particular, the distinction between how precisely a array-element
// is tracked is irrelevant here.)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum InteriorKind {
    InteriorField(mc::FieldName),
    InteriorElement(mc::ElementKind),
}

trait ToInteriorKind { fn cleaned(self) -> InteriorKind; }
impl ToInteriorKind for mc::InteriorKind {
    fn cleaned(self) -> InteriorKind {
        match self {
            mc::InteriorField(name) => InteriorField(name),
            mc::InteriorElement(_, elem_kind) => InteriorElement(elem_kind),
        }
    }
}

#[derive(Copy, PartialEq, Eq, Hash, Debug)]
pub enum LoanPathElem {
    LpDeref(mc::PointerKind),    // `*LV` in doc.rs
    LpInterior(InteriorKind),    // `LV.f` in doc.rs
}

pub fn closure_to_block(closure_id: ast::NodeId,
                        tcx: &ty::ctxt) -> ast::NodeId {
    match tcx.map.get(closure_id) {
        ast_map::NodeExpr(expr) => match expr.node {
            ast::ExprClosure(_, _, ref block) => {
                block.id
            }
            _ => {
                panic!("encountered non-closure id: {}", closure_id)
            }
        },
        _ => panic!("encountered non-expr id: {}", closure_id)
    }
}

impl<'tcx> LoanPath<'tcx> {
    pub fn kill_scope(&self, tcx: &ty::ctxt<'tcx>) -> region::CodeExtent {
        match self.kind {
            LpVar(local_id) => tcx.region_maps.var_scope(local_id),
            LpUpvar(upvar_id) => {
                let block_id = closure_to_block(upvar_id.closure_expr_id, tcx);
                region::CodeExtent::from_node_id(block_id)
            }
            LpDowncast(ref base, _) |
            LpExtend(ref base, _, _) => base.kill_scope(tcx),
        }
    }

    fn has_fork(&self, other: &LoanPath<'tcx>) -> bool {
        match (&self.kind, &other.kind) {
            (&LpExtend(ref base, _, LpInterior(id)), &LpExtend(ref base2, _, LpInterior(id2))) =>
                if id == id2 {
                    base.has_fork(&**base2)
                } else {
                    true
                },
            (&LpExtend(ref base, _, LpDeref(_)), _) => base.has_fork(other),
            (_, &LpExtend(ref base, _, LpDeref(_))) => self.has_fork(&**base),
            _ => false,
        }
    }

    fn depth(&self) -> uint {
        match self.kind {
            LpExtend(ref base, _, LpDeref(_)) => base.depth(),
            LpExtend(ref base, _, LpInterior(_)) => base.depth() + 1,
            _ => 0,
        }
    }

    fn common(&self, other: &LoanPath<'tcx>) -> Option<LoanPath<'tcx>> {
        match (&self.kind, &other.kind) {
            (&LpExtend(ref base, a, LpInterior(id)),
             &LpExtend(ref base2, _, LpInterior(id2))) => {
                if id == id2 {
                    base.common(&**base2).map(|x| {
                        let xd = x.depth();
                        if base.depth() == xd && base2.depth() == xd {
                            assert_eq!(base.ty, base2.ty);
                            assert_eq!(self.ty, other.ty);
                            LoanPath {
                                kind: LpExtend(Rc::new(x), a, LpInterior(id)),
                                ty: self.ty,
                            }
                        } else {
                            x
                        }
                    })
                } else {
                    base.common(&**base2)
                }
            }
            (&LpExtend(ref base, _, LpDeref(_)), _) => base.common(other),
            (_, &LpExtend(ref other, _, LpDeref(_))) => self.common(&**other),
            (&LpVar(id), &LpVar(id2)) => {
                if id == id2 {
                    assert_eq!(self.ty, other.ty);
                    Some(LoanPath { kind: LpVar(id), ty: self.ty })
                } else {
                    None
                }
            }
            (&LpUpvar(id), &LpUpvar(id2)) => {
                if id == id2 {
                    assert_eq!(self.ty, other.ty);
                    Some(LoanPath { kind: LpUpvar(id), ty: self.ty })
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

pub fn opt_loan_path<'tcx>(cmt: &mc::cmt<'tcx>) -> Option<Rc<LoanPath<'tcx>>> {
    //! Computes the `LoanPath` (if any) for a `cmt`.
    //! Note that this logic is somewhat duplicated in
    //! the method `compute()` found in `gather_loans::restrictions`,
    //! which allows it to share common loan path pieces as it
    //! traverses the CMT.

    let new_lp = |v: LoanPathKind<'tcx>| Rc::new(LoanPath::new(v, cmt.ty));

    match cmt.cat {
        mc::cat_rvalue(..) |
        mc::cat_static_item => {
            None
        }

        mc::cat_local(id) => {
            Some(new_lp(LpVar(id)))
        }

        mc::cat_upvar(mc::Upvar { id, .. }) => {
            Some(new_lp(LpUpvar(id)))
        }

        mc::cat_deref(ref cmt_base, _, pk) => {
            opt_loan_path(cmt_base).map(|lp| {
                new_lp(LpExtend(lp, cmt.mutbl, LpDeref(pk)))
            })
        }

        mc::cat_interior(ref cmt_base, ik) => {
            opt_loan_path(cmt_base).map(|lp| {
                new_lp(LpExtend(lp, cmt.mutbl, LpInterior(ik.cleaned())))
            })
        }

        mc::cat_downcast(ref cmt_base, variant_def_id) =>
            opt_loan_path(cmt_base)
            .map(|lp| {
                new_lp(LpDowncast(lp, variant_def_id))
            }),

    }
}

///////////////////////////////////////////////////////////////////////////
// Errors

// Errors that can occur
#[derive(PartialEq)]
pub enum bckerr_code {
    err_mutbl,
    err_out_of_scope(ty::Region, ty::Region), // superscope, subscope
    err_borrowed_pointer_too_short(ty::Region, ty::Region), // loan, ptr
}

// Combination of an error code and the categorization of the expression
// that caused it
#[derive(PartialEq)]
pub struct BckError<'tcx> {
    span: Span,
    cause: euv::LoanCause,
    cmt: mc::cmt<'tcx>,
    code: bckerr_code
}

#[derive(Copy)]
pub enum AliasableViolationKind {
    MutabilityViolation,
    BorrowViolation(euv::LoanCause)
}

#[derive(Copy, Debug)]
pub enum MovedValueUseKind {
    MovedInUse,
    MovedInCapture,
}

///////////////////////////////////////////////////////////////////////////
// Misc

impl<'a, 'tcx> BorrowckCtxt<'a, 'tcx> {
    pub fn is_subregion_of(&self, r_sub: ty::Region, r_sup: ty::Region)
                           -> bool {
        self.tcx.region_maps.is_subregion_of(r_sub, r_sup)
    }

    pub fn report(&self, err: BckError<'tcx>) {
        self.span_err(
            err.span,
            &self.bckerr_to_string(&err)[]);
        self.note_and_explain_bckerr(err);
    }

    pub fn report_use_of_moved_value<'b>(&self,
                                         use_span: Span,
                                         use_kind: MovedValueUseKind,
                                         lp: &LoanPath<'tcx>,
                                         the_move: &move_data::Move,
                                         moved_lp: &LoanPath<'tcx>,
                                         param_env: &ty::ParameterEnvironment<'b,'tcx>) {
        let verb = match use_kind {
            MovedInUse => "use",
            MovedInCapture => "capture",
        };

        let (ol, moved_lp_msg) = match the_move.kind {
            move_data::Declared => {
                self.tcx.sess.span_err(
                    use_span,
                    &format!("{} of possibly uninitialized variable: `{}`",
                            verb,
                            self.loan_path_to_string(lp))[]);
                (self.loan_path_to_string(moved_lp),
                 String::new())
            }
            _ => {
                // If moved_lp is something like `x.a`, and lp is something like `x.b`, we would
                // normally generate a rather confusing message:
                //
                //     error: use of moved value: `x.b`
                //     note: `x.a` moved here...
                //
                // What we want to do instead is get the 'common ancestor' of the two moves and
                // use that for most of the message instead, giving is something like this:
                //
                //     error: use of moved value: `x`
                //     note: `x` moved here (through moving `x.a`)...

                let common = moved_lp.common(lp);
                let has_common = common.is_some();
                let has_fork = moved_lp.has_fork(lp);
                let (nl, ol, moved_lp_msg) =
                    if has_fork && has_common {
                        let nl = self.loan_path_to_string(&common.unwrap());
                        let ol = nl.clone();
                        let moved_lp_msg = format!(" (through moving `{}`)",
                                                   self.loan_path_to_string(moved_lp));
                        (nl, ol, moved_lp_msg)
                    } else {
                        (self.loan_path_to_string(lp),
                         self.loan_path_to_string(moved_lp),
                         String::new())
                    };

                let partial = moved_lp.depth() > lp.depth();
                let msg = if !has_fork && partial { "partially " }
                          else if has_fork && !has_common { "collaterally "}
                          else { "" };
                self.tcx.sess.span_err(
                    use_span,
                    &format!("{} of {}moved value: `{}`",
                            verb,
                            msg,
                            nl)[]);
                (ol, moved_lp_msg)
            }
        };

        match the_move.kind {
            move_data::Declared => {}

            move_data::MoveExpr => {
                let (expr_ty, expr_span) = match self.tcx
                                                     .map
                                                     .find(the_move.id) {
                    Some(ast_map::NodeExpr(expr)) => {
                        (ty::expr_ty_adjusted(self.tcx, &*expr), expr.span)
                    }
                    r => {
                        self.tcx.sess.bug(&format!("MoveExpr({}) maps to \
                                                   {:?}, not Expr",
                                                  the_move.id,
                                                  r)[])
                    }
                };
                let (suggestion, _) =
                    move_suggestion(param_env, expr_span, expr_ty, ("moved by default", ""));
                self.tcx.sess.span_note(
                    expr_span,
                    &format!("`{}` moved here{} because it has type `{}`, which is {}",
                            ol,
                            moved_lp_msg,
                            expr_ty.user_string(self.tcx),
                            suggestion)[]);
            }

            move_data::MovePat => {
                let pat_ty = ty::node_id_to_type(self.tcx, the_move.id);
                let span = self.tcx.map.span(the_move.id);
                self.tcx.sess.span_note(span,
                    &format!("`{}` moved here{} because it has type `{}`, \
                             which is moved by default",
                            ol,
                            moved_lp_msg,
                            pat_ty.user_string(self.tcx))[]);
                self.tcx.sess.span_help(span,
                    "use `ref` to override");
            }

            move_data::Captured => {
                let (expr_ty, expr_span) = match self.tcx
                                                     .map
                                                     .find(the_move.id) {
                    Some(ast_map::NodeExpr(expr)) => {
                        (ty::expr_ty_adjusted(self.tcx, &*expr), expr.span)
                    }
                    r => {
                        self.tcx.sess.bug(&format!("Captured({}) maps to \
                                                   {:?}, not Expr",
                                                  the_move.id,
                                                  r)[])
                    }
                };
                let (suggestion, help) =
                    move_suggestion(param_env,
                                    expr_span,
                                    expr_ty,
                                    ("moved by default",
                                     "make a copy and capture that instead to override"));
                self.tcx.sess.span_note(
                    expr_span,
                    &format!("`{}` moved into closure environment here{} because it \
                            has type `{}`, which is {}",
                            ol,
                            moved_lp_msg,
                            expr_ty.user_string(self.tcx),
                            suggestion)[]);
                self.tcx.sess.span_help(expr_span, help);
            }
        }

        fn move_suggestion<'a,'tcx>(param_env: &ty::ParameterEnvironment<'a,'tcx>,
                                    span: Span,
                                    ty: Ty<'tcx>,
                                    default_msgs: (&'static str, &'static str))
                                    -> (&'static str, &'static str) {
            match ty.sty {
                _ => {
                    if ty::type_moves_by_default(param_env, span, ty) {
                        ("non-copyable",
                         "perhaps you meant to use `clone()`?")
                    } else {
                        default_msgs
                    }
                }
            }
        }
    }

    pub fn report_partial_reinitialization_of_uninitialized_structure(
            &self,
            span: Span,
            lp: &LoanPath<'tcx>) {
        self.tcx
            .sess
            .span_err(span,
                      (format!("partial reinitialization of uninitialized \
                               structure `{}`",
                               self.loan_path_to_string(lp))).as_slice());
    }

    pub fn report_reassigned_immutable_variable(&self,
                                                span: Span,
                                                lp: &LoanPath<'tcx>,
                                                assign:
                                                &move_data::Assignment) {
        self.tcx.sess.span_err(
            span,
            &format!("re-assignment of immutable variable `{}`",
                    self.loan_path_to_string(lp))[]);
        self.tcx.sess.span_note(assign.span, "prior assignment occurs here");
    }

    pub fn span_err(&self, s: Span, m: &str) {
        self.tcx.sess.span_err(s, m);
    }

    pub fn span_bug(&self, s: Span, m: &str) {
        self.tcx.sess.span_bug(s, m);
    }

    pub fn span_note(&self, s: Span, m: &str) {
        self.tcx.sess.span_note(s, m);
    }

    pub fn span_end_note(&self, s: Span, m: &str) {
        self.tcx.sess.span_end_note(s, m);
    }

    pub fn span_help(&self, s: Span, m: &str) {
        self.tcx.sess.span_help(s, m);
    }

    pub fn bckerr_to_string(&self, err: &BckError<'tcx>) -> String {
        match err.code {
            err_mutbl => {
                let descr = match err.cmt.note {
                    mc::NoteClosureEnv(_) | mc::NoteUpvarRef(_) => {
                        self.cmt_to_string(&*err.cmt)
                    }
                    _ => match opt_loan_path(&err.cmt) {
                        None => {
                            format!("{} {}",
                                    err.cmt.mutbl.to_user_str(),
                                    self.cmt_to_string(&*err.cmt))
                        }
                        Some(lp) => {
                            format!("{} {} `{}`",
                                    err.cmt.mutbl.to_user_str(),
                                    self.cmt_to_string(&*err.cmt),
                                    self.loan_path_to_string(&*lp))
                        }
                    }
                };

                match err.cause {
                    euv::ClosureCapture(_) => {
                        format!("closure cannot assign to {}", descr)
                    }
                    euv::OverloadedOperator |
                    euv::AddrOf |
                    euv::RefBinding |
                    euv::AutoRef |
                    euv::ForLoop |
                    euv::MatchDiscriminant => {
                        format!("cannot borrow {} as mutable", descr)
                    }
                    euv::ClosureInvocation => {
                        self.tcx.sess.span_bug(err.span,
                            "err_mutbl with a closure invocation");
                    }
                }
            }
            err_out_of_scope(..) => {
                let msg = match opt_loan_path(&err.cmt) {
                    None => "borrowed value".to_string(),
                    Some(lp) => {
                        format!("`{}`", self.loan_path_to_string(&*lp))
                    }
                };
                format!("{} does not live long enough", msg)
            }
            err_borrowed_pointer_too_short(..) => {
                let descr = match opt_loan_path(&err.cmt) {
                    Some(lp) => {
                        format!("`{}`", self.loan_path_to_string(&*lp))
                    }
                    None => self.cmt_to_string(&*err.cmt),
                };

                format!("lifetime of {} is too short to guarantee \
                                its contents can be safely reborrowed",
                               descr)
            }
        }
    }

    pub fn report_aliasability_violation(&self,
                                         span: Span,
                                         kind: AliasableViolationKind,
                                         cause: mc::AliasableReason) {
        let mut is_closure = false;
        let prefix = match kind {
            MutabilityViolation => {
                "cannot assign to data"
            }
            BorrowViolation(euv::ClosureCapture(_)) |
            BorrowViolation(euv::OverloadedOperator) |
            BorrowViolation(euv::AddrOf) |
            BorrowViolation(euv::AutoRef) |
            BorrowViolation(euv::RefBinding) |
            BorrowViolation(euv::MatchDiscriminant) => {
                "cannot borrow data mutably"
            }

            BorrowViolation(euv::ClosureInvocation) => {
                is_closure = true;
                "closure invocation"
            }

            BorrowViolation(euv::ForLoop) => {
                "`for` loop"
            }
        };

        match cause {
            mc::AliasableOther => {
                self.tcx.sess.span_err(
                    span,
                    &format!("{} in an aliasable location",
                             prefix)[]);
            }
            mc::AliasableClosure(id) => {
                self.tcx.sess.span_err(span,
                                       &format!("{} in a captured outer \
                                                variable in an `Fn` closure", prefix));
                if let BorrowViolation(euv::ClosureCapture(_)) = kind {
                    // The aliasability violation with closure captures can
                    // happen for nested closures, so we know the enclosing
                    // closure incorrectly accepts an `Fn` while it needs to
                    // be `FnMut`.
                    span_help!(self.tcx.sess, self.tcx.map.span(id),
                           "consider changing this to accept closures that implement `FnMut`");
                } else {
                    span_help!(self.tcx.sess, self.tcx.map.span(id),
                           "consider changing this closure to take self by mutable reference");
                }
            }
            mc::AliasableStatic(..) |
            mc::AliasableStaticMut(..) => {
                self.tcx.sess.span_err(
                    span,
                    &format!("{} in a static location", prefix)[]);
            }
            mc::AliasableBorrowed => {
                self.tcx.sess.span_err(
                    span,
                    &format!("{} in a `&` reference", prefix)[]);
            }
        }

        if is_closure {
            self.tcx.sess.span_help(
                span,
                "closures behind references must be called via `&mut`");
        }
    }

    pub fn note_and_explain_bckerr(&self, err: BckError<'tcx>) {
        let code = err.code;
        match code {
            err_mutbl(..) => {
                match err.cmt.note {
                    mc::NoteClosureEnv(upvar_id) | mc::NoteUpvarRef(upvar_id) => {
                        // If this is an `Fn` closure, it simply can't mutate upvars.
                        // If it's an `FnMut` closure, the original variable was declared immutable.
                        // We need to determine which is the case here.
                        let kind = match err.cmt.upvar().unwrap().cat {
                            mc::cat_upvar(mc::Upvar { kind, .. }) => kind,
                            _ => unreachable!()
                        };
                        if kind == ty::FnClosureKind {
                            self.tcx.sess.span_help(
                                self.tcx.map.span(upvar_id.closure_expr_id),
                                "consider changing this closure to take \
                                 self by mutable reference");
                        }
                    }
                    _ => {}
                }
            }

            err_out_of_scope(super_scope, sub_scope) => {
                note_and_explain_region(
                    self.tcx,
                    "reference must be valid for ",
                    sub_scope,
                    "...");
                let suggestion = if is_statement_scope(self.tcx, super_scope) {
                    Some("consider using a `let` binding to increase its lifetime")
                } else {
                    None
                };
                let span = note_and_explain_region(
                    self.tcx,
                    "...but borrowed value is only valid for ",
                    super_scope,
                    "");
                match (span, suggestion) {
                    (_, None) => {},
                    (Some(span), Some(msg)) => self.tcx.sess.span_help(span, msg),
                    (None, Some(msg)) => self.tcx.sess.help(msg),
                }
            }

            err_borrowed_pointer_too_short(loan_scope, ptr_scope) => {
                let descr = match opt_loan_path(&err.cmt) {
                    Some(lp) => {
                        format!("`{}`", self.loan_path_to_string(&*lp))
                    }
                    None => self.cmt_to_string(&*err.cmt),
                };
                note_and_explain_region(
                    self.tcx,
                    &format!("{} would have to be valid for ",
                            descr)[],
                    loan_scope,
                    "...");
                note_and_explain_region(
                    self.tcx,
                    &format!("...but {} is only valid for ", descr)[],
                    ptr_scope,
                    "");
            }
        }
    }

    pub fn append_loan_path_to_string(&self,
                                      loan_path: &LoanPath<'tcx>,
                                      out: &mut String) {
        match loan_path.kind {
            LpUpvar(ty::UpvarId{ var_id: id, closure_expr_id: _ }) |
            LpVar(id) => {
                out.push_str(&ty::local_var_name_str(self.tcx, id));
            }

            LpDowncast(ref lp_base, variant_def_id) => {
                out.push('(');
                self.append_loan_path_to_string(&**lp_base, out);
                out.push_str(DOWNCAST_PRINTED_OPERATOR);
                out.push_str(&ty::item_path_str(self.tcx, variant_def_id)[]);
                out.push(')');
            }


            LpExtend(ref lp_base, _, LpInterior(InteriorField(fname))) => {
                self.append_autoderefd_loan_path_to_string(&**lp_base, out);
                match fname {
                    mc::NamedField(fname) => {
                        out.push('.');
                        out.push_str(&token::get_name(fname));
                    }
                    mc::PositionalField(idx) => {
                        out.push('.');
                        out.push_str(&idx.to_string()[]);
                    }
                }
            }

            LpExtend(ref lp_base, _, LpInterior(InteriorElement(..))) => {
                self.append_autoderefd_loan_path_to_string(&**lp_base, out);
                out.push_str("[..]");
            }

            LpExtend(ref lp_base, _, LpDeref(_)) => {
                out.push('*');
                self.append_loan_path_to_string(&**lp_base, out);
            }
        }
    }

    pub fn append_autoderefd_loan_path_to_string(&self,
                                                 loan_path: &LoanPath<'tcx>,
                                                 out: &mut String) {
        match loan_path.kind {
            LpExtend(ref lp_base, _, LpDeref(_)) => {
                // For a path like `(*x).f` or `(*x)[3]`, autoderef
                // rules would normally allow users to omit the `*x`.
                // So just serialize such paths to `x.f` or x[3]` respectively.
                self.append_autoderefd_loan_path_to_string(&**lp_base, out)
            }

            LpDowncast(ref lp_base, variant_def_id) => {
                out.push('(');
                self.append_autoderefd_loan_path_to_string(&**lp_base, out);
                out.push(':');
                out.push_str(&ty::item_path_str(self.tcx, variant_def_id)[]);
                out.push(')');
            }

            LpVar(..) | LpUpvar(..) | LpExtend(_, _, LpInterior(..)) => {
                self.append_loan_path_to_string(loan_path, out)
            }
        }
    }

    pub fn loan_path_to_string(&self, loan_path: &LoanPath<'tcx>) -> String {
        let mut result = String::new();
        self.append_loan_path_to_string(loan_path, &mut result);
        result
    }

    pub fn cmt_to_string(&self, cmt: &mc::cmt_<'tcx>) -> String {
        cmt.descriptive_string(self.tcx)
    }
}

fn is_statement_scope(tcx: &ty::ctxt, region: ty::Region) -> bool {
     match region {
         ty::ReScope(scope) => {
             match tcx.map.find(scope.node_id()) {
                 Some(ast_map::NodeStmt(_)) => true,
                 _ => false
             }
         }
         _ => false
     }
}

impl BitwiseOperator for LoanDataFlowOperator {
    #[inline]
    fn join(&self, succ: uint, pred: uint) -> uint {
        succ | pred // loans from both preds are in scope
    }
}

impl DataFlowOperator for LoanDataFlowOperator {
    #[inline]
    fn initial_value(&self) -> bool {
        false // no loans in scope by default
    }
}

impl<'tcx> Repr<'tcx> for InteriorKind {
    fn repr(&self, _tcx: &ty::ctxt<'tcx>) -> String {
        match *self {
            InteriorField(mc::NamedField(fld)) =>
                format!("{}", token::get_name(fld)),
            InteriorField(mc::PositionalField(i)) => format!("#{}", i),
            InteriorElement(..) => "[]".to_string(),
        }
    }
}

impl<'tcx> Repr<'tcx> for Loan<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        format!("Loan_{}({}, {:?}, {:?}-{:?}, {})",
                 self.index,
                 self.loan_path.repr(tcx),
                 self.kind,
                 self.gen_scope,
                 self.kill_scope,
                 self.restricted_paths.repr(tcx))
    }
}

impl<'tcx> Repr<'tcx> for LoanPath<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        match self.kind {
            LpVar(id) => {
                format!("$({})", tcx.map.node_to_string(id))
            }

            LpUpvar(ty::UpvarId{ var_id, closure_expr_id }) => {
                let s = tcx.map.node_to_string(var_id);
                format!("$({} captured by id={})", s, closure_expr_id)
            }

            LpDowncast(ref lp, variant_def_id) => {
                let variant_str = if variant_def_id.krate == ast::LOCAL_CRATE {
                    ty::item_path_str(tcx, variant_def_id)
                } else {
                    variant_def_id.repr(tcx)
                };
                format!("({}{}{})", lp.repr(tcx), DOWNCAST_PRINTED_OPERATOR, variant_str)
            }

            LpExtend(ref lp, _, LpDeref(_)) => {
                format!("{}.*", lp.repr(tcx))
            }

            LpExtend(ref lp, _, LpInterior(ref interior)) => {
                format!("{}.{}", lp.repr(tcx), interior.repr(tcx))
            }
        }
    }
}

impl<'tcx> UserString<'tcx> for LoanPath<'tcx> {
    fn user_string(&self, tcx: &ty::ctxt<'tcx>) -> String {
        match self.kind {
            LpVar(id) => {
                format!("$({})", tcx.map.node_to_user_string(id))
            }

            LpUpvar(ty::UpvarId{ var_id, closure_expr_id: _ }) => {
                let s = tcx.map.node_to_user_string(var_id);
                format!("$({} captured by closure)", s)
            }

            LpDowncast(ref lp, variant_def_id) => {
                let variant_str = if variant_def_id.krate == ast::LOCAL_CRATE {
                    ty::item_path_str(tcx, variant_def_id)
                } else {
                    variant_def_id.repr(tcx)
                };
                format!("({}{}{})", lp.user_string(tcx), DOWNCAST_PRINTED_OPERATOR, variant_str)
            }

            LpExtend(ref lp, _, LpDeref(_)) => {
                format!("{}.*", lp.user_string(tcx))
            }

            LpExtend(ref lp, _, LpInterior(ref interior)) => {
                format!("{}.{}", lp.user_string(tcx), interior.repr(tcx))
            }
        }
    }
}
