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

use rustc::dep_graph::DepNode;
use rustc::front::map as hir_map;
use rustc::front::map::blocks::FnParts;
use rustc::middle::cfg;
use rustc::middle::dataflow::DataFlowContext;
use rustc::middle::dataflow::BitwiseOperator;
use rustc::middle::dataflow::DataFlowOperator;
use rustc::middle::dataflow::KillFrom;
use rustc::middle::def_id::DefId;
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::free_region::FreeRegionMap;
use rustc::middle::mem_categorization as mc;
use rustc::middle::mem_categorization::Categorization;
use rustc::middle::region;
use rustc::middle::ty::{self, Ty};

use std::fmt;
use std::mem;
use std::rc::Rc;
use syntax::ast::{self, NodeId};
use syntax::codemap::Span;

use rustc_front::hir;
use rustc_front::hir::{FnDecl, Block};
use rustc_front::intravisit;
use rustc_front::intravisit::{Visitor, FnKind};
use rustc_front::util as hir_util;

pub mod check_loans;

pub mod gather_loans;

pub mod move_data;

#[derive(Clone, Copy)]
pub struct LoanDataFlowOperator;

pub type LoanDataFlow<'a, 'tcx> = DataFlowContext<'a, 'tcx, LoanDataFlowOperator>;

impl<'a, 'tcx, 'v> Visitor<'v> for BorrowckCtxt<'a, 'tcx> {
    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v FnDecl,
                b: &'v Block, s: Span, id: ast::NodeId) {
        match fk {
            FnKind::ItemFn(..) |
            FnKind::Method(..) => {
                let new_free_region_map = self.tcx.free_region_map(id);
                let old_free_region_map =
                    mem::replace(&mut self.free_region_map, new_free_region_map);
                borrowck_fn(self, fk, fd, b, s, id);
                self.free_region_map = old_free_region_map;
            }

            FnKind::Closure => {
                borrowck_fn(self, fk, fd, b, s, id);
            }
        }
    }

    fn visit_item(&mut self, item: &hir::Item) {
        borrowck_item(self, item);
    }

    fn visit_trait_item(&mut self, ti: &hir::TraitItem) {
        if let hir::ConstTraitItem(_, Some(ref expr)) = ti.node {
            gather_loans::gather_loans_in_static_initializer(self, &*expr);
        }
        intravisit::walk_trait_item(self, ti);
    }

    fn visit_impl_item(&mut self, ii: &hir::ImplItem) {
        if let hir::ImplItemKind::Const(_, ref expr) = ii.node {
            gather_loans::gather_loans_in_static_initializer(self, &*expr);
        }
        intravisit::walk_impl_item(self, ii);
    }
}

pub fn check_crate(tcx: &ty::ctxt) {
    let mut bccx = BorrowckCtxt {
        tcx: tcx,
        free_region_map: FreeRegionMap::new(),
        stats: BorrowStats {
            loaned_paths_same: 0,
            loaned_paths_imm: 0,
            stable_paths: 0,
            guaranteed_paths: 0
        }
    };

    tcx.visit_all_items_in_krate(DepNode::BorrowCheck, &mut bccx);

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

    fn make_stat(bccx: &BorrowckCtxt, stat: usize) -> String {
        let total = bccx.stats.guaranteed_paths as f64;
        let perc = if total == 0.0 { 0.0 } else { stat as f64 * 100.0 / total };
        format!("{} ({:.0}%)", stat, perc)
    }
}

fn borrowck_item(this: &mut BorrowckCtxt, item: &hir::Item) {
    // Gather loans for items. Note that we don't need
    // to check loans for single expressions. The check
    // loan step is intended for things that have a data
    // flow dependent conditions.
    match item.node {
        hir::ItemStatic(_, _, ref ex) |
        hir::ItemConst(_, ref ex) => {
            gather_loans::gather_loans_in_static_initializer(this, &**ex);
        }
        _ => { }
    }

    intravisit::walk_item(this, item);
}

/// Collection of conclusions determined via borrow checker analyses.
pub struct AnalysisData<'a, 'tcx: 'a> {
    pub all_loans: Vec<Loan<'tcx>>,
    pub loans: DataFlowContext<'a, 'tcx, LoanDataFlowOperator>,
    pub move_data: move_data::FlowedMoveData<'a, 'tcx>,
}

fn borrowck_fn(this: &mut BorrowckCtxt,
               fk: FnKind,
               decl: &hir::FnDecl,
               body: &hir::Block,
               sp: Span,
               id: ast::NodeId) {
    debug!("borrowck_fn(id={})", id);
    let cfg = cfg::CFG::new(this.tcx, body);
    let AnalysisData { all_loans,
                       loans: loan_dfcx,
                       move_data: flowed_moves } =
        build_borrowck_dataflow_data(this, fk, decl, &cfg, body, sp, id);

    move_data::fragments::instrument_move_fragments(&flowed_moves.move_data,
                                                    this.tcx,
                                                    sp,
                                                    id);
    move_data::fragments::build_unfragmented_map(this,
                                                 &flowed_moves.move_data,
                                                 id);

    check_loans::check_loans(this,
                             &loan_dfcx,
                             &flowed_moves,
                             &all_loans[..],
                             id,
                             decl,
                             body);

    intravisit::walk_fn(this, fk, decl, body, sp);
}

fn build_borrowck_dataflow_data<'a, 'tcx>(this: &mut BorrowckCtxt<'a, 'tcx>,
                                          fk: FnKind,
                                          decl: &hir::FnDecl,
                                          cfg: &cfg::CFG,
                                          body: &hir::Block,
                                          sp: Span,
                                          id: ast::NodeId)
                                          -> AnalysisData<'a, 'tcx>
{
    // Check the body of fn items.
    let tcx = this.tcx;
    let id_range = hir_util::compute_id_range_for_fn_body(fk, decl, body, sp, id);
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
        loan_dfcx.add_gen(loan.gen_scope.node_id(&tcx.region_maps), loan_idx);
        loan_dfcx.add_kill(KillFrom::ScopeEnd,
                           loan.kill_scope.node_id(&tcx.region_maps), loan_idx);
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

/// Accessor for introspective clients inspecting `AnalysisData` and
/// the `BorrowckCtxt` itself , e.g. the flowgraph visualizer.
pub fn build_borrowck_dataflow_data_for_fn<'a, 'tcx>(
    tcx: &'a ty::ctxt<'tcx>,
    fn_parts: FnParts<'a>,
    cfg: &cfg::CFG)
    -> (BorrowckCtxt<'a, 'tcx>, AnalysisData<'a, 'tcx>)
{

    let mut bccx = BorrowckCtxt {
        tcx: tcx,
        free_region_map: FreeRegionMap::new(),
        stats: BorrowStats {
            loaned_paths_same: 0,
            loaned_paths_imm: 0,
            stable_paths: 0,
            guaranteed_paths: 0
        }
    };

    let dataflow_data = build_borrowck_dataflow_data(&mut bccx,
                                                     fn_parts.kind,
                                                     &*fn_parts.decl,
                                                     cfg,
                                                     &*fn_parts.body,
                                                     fn_parts.span,
                                                     fn_parts.id);

    (bccx, dataflow_data)
}

// ----------------------------------------------------------------------
// Type definitions

pub struct BorrowckCtxt<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,

    // Hacky. As we visit various fns, we have to load up the
    // free-region map for each one. This map is computed by during
    // typeck for each fn item and stored -- closures just use the map
    // from the fn item that encloses them. Since we walk the fns in
    // order, we basically just overwrite this field as we enter a fn
    // item and restore it afterwards in a stack-like fashion. Then
    // the borrow checking code can assume that `free_region_map` is
    // always the correct map for the current fn. Feels like it'd be
    // better to just recompute this, rather than store it, but it's a
    // bit of a pain to factor that code out at the moment.
    free_region_map: FreeRegionMap,

    // Statistics:
    stats: BorrowStats
}

struct BorrowStats {
    loaned_paths_same: usize,
    loaned_paths_imm: usize,
    stable_paths: usize,
    guaranteed_paths: usize
}

pub type BckResult<'tcx, T> = Result<T, BckError<'tcx>>;

///////////////////////////////////////////////////////////////////////////
// Loans and loan paths

/// Record of a loan that was issued.
pub struct Loan<'tcx> {
    index: usize,
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

#[derive(Eq, Hash)]
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
    LpVar(ast::NodeId),                         // `x` in README.md
    LpUpvar(ty::UpvarId),                       // `x` captured by-value into closure
    LpDowncast(Rc<LoanPath<'tcx>>, DefId), // `x` downcast to particular enum variant
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
const DOWNCAST_PRINTED_OPERATOR: &'static str = " as ";

// A local, "cleaned" version of `mc::InteriorKind` that drops
// information that is not relevant to loan-path analysis. (In
// particular, the distinction between how precisely an array-element
// is tracked is irrelevant here.)
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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

// This can be:
// - a pointer dereference (`*LV` in README.md)
// - a field reference, with an optional definition of the containing
//   enum variant (`LV.f` in README.md)
// `DefId` is present when the field is part of struct that is in
// a variant of an enum. For instance in:
// `enum E { X { foo: u32 }, Y { foo: u32 }}`
// each `foo` is qualified by the definitition id of the variant (`X` or `Y`).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum LoanPathElem {
    LpDeref(mc::PointerKind),
    LpInterior(Option<DefId>, InteriorKind),
}

pub fn closure_to_block(closure_id: ast::NodeId,
                        tcx: &ty::ctxt) -> ast::NodeId {
    match tcx.map.get(closure_id) {
        hir_map::NodeExpr(expr) => match expr.node {
            hir::ExprClosure(_, _, ref block) => {
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
                tcx.region_maps.node_extent(block_id)
            }
            LpDowncast(ref base, _) |
            LpExtend(ref base, _, _) => base.kill_scope(tcx),
        }
    }

    fn has_fork(&self, other: &LoanPath<'tcx>) -> bool {
        match (&self.kind, &other.kind) {
            (&LpExtend(ref base, _, LpInterior(opt_variant_id, id)),
             &LpExtend(ref base2, _, LpInterior(opt_variant_id2, id2))) =>
                if id == id2 && opt_variant_id == opt_variant_id2 {
                    base.has_fork(&**base2)
                } else {
                    true
                },
            (&LpExtend(ref base, _, LpDeref(_)), _) => base.has_fork(other),
            (_, &LpExtend(ref base, _, LpDeref(_))) => self.has_fork(&**base),
            _ => false,
        }
    }

    fn depth(&self) -> usize {
        match self.kind {
            LpExtend(ref base, _, LpDeref(_)) => base.depth(),
            LpExtend(ref base, _, LpInterior(_, _)) => base.depth() + 1,
            _ => 0,
        }
    }

    fn common(&self, other: &LoanPath<'tcx>) -> Option<LoanPath<'tcx>> {
        match (&self.kind, &other.kind) {
            (&LpExtend(ref base, a, LpInterior(opt_variant_id, id)),
             &LpExtend(ref base2, _, LpInterior(opt_variant_id2, id2))) => {
                if id == id2 && opt_variant_id == opt_variant_id2 {
                    base.common(&**base2).map(|x| {
                        let xd = x.depth();
                        if base.depth() == xd && base2.depth() == xd {
                            assert_eq!(base.ty, base2.ty);
                            assert_eq!(self.ty, other.ty);
                            LoanPath {
                                kind: LpExtend(Rc::new(x), a, LpInterior(opt_variant_id, id)),
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
        Categorization::Rvalue(..) |
        Categorization::StaticItem => {
            None
        }

        Categorization::Local(id) => {
            Some(new_lp(LpVar(id)))
        }

        Categorization::Upvar(mc::Upvar { id, .. }) => {
            Some(new_lp(LpUpvar(id)))
        }

        Categorization::Deref(ref cmt_base, _, pk) => {
            opt_loan_path(cmt_base).map(|lp| {
                new_lp(LpExtend(lp, cmt.mutbl, LpDeref(pk)))
            })
        }

        Categorization::Interior(ref cmt_base, ik) => {
            opt_loan_path(cmt_base).map(|lp| {
                let opt_variant_id = match cmt_base.cat {
                    Categorization::Downcast(_, did) =>  Some(did),
                    _ => None
                };
                new_lp(LpExtend(lp, cmt.mutbl, LpInterior(opt_variant_id, ik.cleaned())))
            })
        }

        Categorization::Downcast(ref cmt_base, variant_def_id) =>
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
    cause: AliasableViolationKind,
    cmt: mc::cmt<'tcx>,
    code: bckerr_code
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AliasableViolationKind {
    MutabilityViolation,
    BorrowViolation(euv::LoanCause)
}

#[derive(Copy, Clone, Debug)]
pub enum MovedValueUseKind {
    MovedInUse,
    MovedInCapture,
}

///////////////////////////////////////////////////////////////////////////
// Misc

impl<'a, 'tcx> BorrowckCtxt<'a, 'tcx> {
    pub fn is_subregion_of(&self, r_sub: ty::Region, r_sup: ty::Region)
                           -> bool
    {
        self.free_region_map.is_subregion_of(self.tcx, r_sub, r_sup)
    }

    pub fn report(&self, err: BckError<'tcx>) {
        // Catch and handle some particular cases.
        match (&err.code, &err.cause) {
            (&err_out_of_scope(ty::ReScope(_), ty::ReStatic),
             &BorrowViolation(euv::ClosureCapture(span))) |
            (&err_out_of_scope(ty::ReScope(_), ty::ReFree(..)),
             &BorrowViolation(euv::ClosureCapture(span))) => {
                return self.report_out_of_scope_escaping_closure_capture(&err, span);
            }
            _ => { }
        }

        // General fallback.
        self.span_err(
            err.span,
            &self.bckerr_to_string(&err));
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
                span_err!(
                    self.tcx.sess, use_span, E0381,
                    "{} of possibly uninitialized variable: `{}`",
                    verb,
                    self.loan_path_to_string(lp));

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
                span_err!(
                    self.tcx.sess, use_span, E0382,
                    "{} of {}moved value: `{}`",
                    verb, msg, nl);
                (ol, moved_lp_msg)
            }
        };

        match the_move.kind {
            move_data::Declared => {}

            move_data::MoveExpr => {
                let (expr_ty, expr_span) = match self.tcx
                                                     .map
                                                     .find(the_move.id) {
                    Some(hir_map::NodeExpr(expr)) => {
                        (self.tcx.expr_ty_adjusted(&*expr), expr.span)
                    }
                    r => {
                        self.tcx.sess.bug(&format!("MoveExpr({}) maps to \
                                                   {:?}, not Expr",
                                                  the_move.id,
                                                  r))
                    }
                };
                let (suggestion, _) =
                    move_suggestion(param_env, expr_span, expr_ty, ("moved by default", ""));
                // If the two spans are the same, it's because the expression will be evaluated
                // multiple times. Avoid printing the same span and adjust the wording so it makes
                // more sense that it's from multiple evalutations.
                if expr_span == use_span {
                    self.tcx.sess.note(
                        &format!("`{}` was previously moved here{} because it has type `{}`, \
                                  which is {}",
                                 ol,
                                 moved_lp_msg,
                                 expr_ty,
                                 suggestion));
                } else {
                    self.tcx.sess.span_note(
                        expr_span,
                        &format!("`{}` moved here{} because it has type `{}`, which is {}",
                                 ol,
                                 moved_lp_msg,
                                 expr_ty,
                                 suggestion));
                }
            }

            move_data::MovePat => {
                let pat_ty = self.tcx.node_id_to_type(the_move.id);
                let span = self.tcx.map.span(the_move.id);
                self.tcx.sess.span_note(span,
                    &format!("`{}` moved here{} because it has type `{}`, \
                             which is moved by default",
                            ol,
                            moved_lp_msg,
                            pat_ty));
                match self.tcx.sess.codemap().span_to_snippet(span) {
                    Ok(string) => {
                        self.tcx.sess.span_suggestion(
                            span,
                            &format!("if you would like to borrow the value instead, \
                                      use a `ref` binding as shown:"),
                            format!("ref {}", string));
                    },
                    Err(_) => {
                        self.tcx.sess.fileline_help(span,
                            "use `ref` to override");
                    },
                }
            }

            move_data::Captured => {
                let (expr_ty, expr_span) = match self.tcx
                                                     .map
                                                     .find(the_move.id) {
                    Some(hir_map::NodeExpr(expr)) => {
                        (self.tcx.expr_ty_adjusted(&*expr), expr.span)
                    }
                    r => {
                        self.tcx.sess.bug(&format!("Captured({}) maps to \
                                                   {:?}, not Expr",
                                                  the_move.id,
                                                  r))
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
                            moved_lp.ty,
                            suggestion));
                self.tcx.sess.fileline_help(expr_span, help);
            }
        }

        fn move_suggestion<'a,'tcx>(param_env: &ty::ParameterEnvironment<'a,'tcx>,
                                    span: Span,
                                    ty: Ty<'tcx>,
                                    default_msgs: (&'static str, &'static str))
                                    -> (&'static str, &'static str) {
            match ty.sty {
                _ => {
                    if ty.moves_by_default(param_env, span) {
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
        span_err!(
            self.tcx.sess, span, E0383,
            "partial reinitialization of uninitialized structure `{}`",
            self.loan_path_to_string(lp));
    }

    pub fn report_reassigned_immutable_variable(&self,
                                                span: Span,
                                                lp: &LoanPath<'tcx>,
                                                assign:
                                                &move_data::Assignment) {
        span_err!(
            self.tcx.sess, span, E0384,
            "re-assignment of immutable variable `{}`",
            self.loan_path_to_string(lp));
        self.tcx.sess.span_note(assign.span, "prior assignment occurs here");
    }

    pub fn span_err(&self, s: Span, m: &str) {
        self.tcx.sess.span_err(s, m);
    }

    pub fn span_err_with_code(&self, s: Span, msg: &str, code: &str) {
        self.tcx.sess.span_err_with_code(s, msg, code);
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

    pub fn fileline_help(&self, s: Span, m: &str) {
        self.tcx.sess.fileline_help(s, m);
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
                    MutabilityViolation => {
                        format!("cannot assign to {}", descr)
                    }
                    BorrowViolation(euv::ClosureCapture(_)) => {
                        format!("closure cannot assign to {}", descr)
                    }
                    BorrowViolation(euv::OverloadedOperator) |
                    BorrowViolation(euv::AddrOf) |
                    BorrowViolation(euv::RefBinding) |
                    BorrowViolation(euv::AutoRef) |
                    BorrowViolation(euv::AutoUnsafe) |
                    BorrowViolation(euv::ForLoop) |
                    BorrowViolation(euv::MatchDiscriminant) => {
                        format!("cannot borrow {} as mutable", descr)
                    }
                    BorrowViolation(euv::ClosureInvocation) => {
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
                let descr = self.cmt_to_path_or_string(&err.cmt);
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
            BorrowViolation(euv::AutoUnsafe) |
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
                span_err!(
                    self.tcx.sess, span, E0385,
                    "{} in an aliasable location", prefix);
            }
            mc::AliasableReason::UnaliasableImmutable => {
                span_err!(
                    self.tcx.sess, span, E0386,
                    "{} in an immutable container", prefix);
            }
            mc::AliasableClosure(id) => {
                span_err!(
                    self.tcx.sess, span, E0387,
                    "{} in a captured outer variable in an `Fn` closure", prefix);
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
            mc::AliasableStatic |
            mc::AliasableStaticMut => {
                span_err!(
                    self.tcx.sess, span, E0388,
                    "{} in a static location", prefix);
            }
            mc::AliasableBorrowed => {
                span_err!(
                    self.tcx.sess, span, E0389,
                    "{} in a `&` reference", prefix);
            }
        }

        if is_closure {
            self.tcx.sess.fileline_help(
                span,
                "closures behind references must be called via `&mut`");
        }
    }

    fn report_out_of_scope_escaping_closure_capture(&self,
                                                    err: &BckError<'tcx>,
                                                    capture_span: Span)
    {
        let cmt_path_or_string = self.cmt_to_path_or_string(&err.cmt);

        span_err!(
            self.tcx.sess, err.span, E0373,
            "closure may outlive the current function, \
             but it borrows {}, \
             which is owned by the current function",
            cmt_path_or_string);

        self.tcx.sess.span_note(
            capture_span,
            &format!("{} is borrowed here",
                     cmt_path_or_string));

        let suggestion =
            match self.tcx.sess.codemap().span_to_snippet(err.span) {
                Ok(string) => format!("move {}", string),
                Err(_) => format!("move |<args>| <body>")
            };

        self.tcx.sess.span_suggestion(
            err.span,
            &format!("to force the closure to take ownership of {} \
                      (and any other referenced variables), \
                      use the `move` keyword, as shown:",
                     cmt_path_or_string),
            suggestion);
    }

    pub fn note_and_explain_bckerr(&self, err: BckError<'tcx>) {
        let code = err.code;
        match code {
            err_mutbl => {
                match err.cmt.note {
                    mc::NoteClosureEnv(upvar_id) | mc::NoteUpvarRef(upvar_id) => {
                        // If this is an `Fn` closure, it simply can't mutate upvars.
                        // If it's an `FnMut` closure, the original variable was declared immutable.
                        // We need to determine which is the case here.
                        let kind = match err.cmt.upvar().unwrap().cat {
                            Categorization::Upvar(mc::Upvar { kind, .. }) => kind,
                            _ => unreachable!()
                        };
                        if kind == ty::FnClosureKind {
                            self.tcx.sess.span_help(
                                self.tcx.map.span(upvar_id.closure_expr_id),
                                "consider changing this closure to take \
                                 self by mutable reference");
                        }
                    }
                    _ => {
                        if let Categorization::Local(local_id) = err.cmt.cat {
                            let span = self.tcx.map.span(local_id);
                            if let Ok(snippet) = self.tcx.sess.codemap().span_to_snippet(span) {
                                self.tcx.sess.span_suggestion(
                                    span,
                                    &format!("to make the {} mutable, use `mut` as shown:",
                                             self.cmt_to_string(&err.cmt)),
                                    format!("mut {}", snippet));
                            }
                        }
                    }
                }
            }

            err_out_of_scope(super_scope, sub_scope) => {
                self.tcx.note_and_explain_region(
                    "reference must be valid for ",
                    sub_scope,
                    "...");
                self.tcx.note_and_explain_region(
                    "...but borrowed value is only valid for ",
                    super_scope,
                    "");
                if let Some(span) = statement_scope_span(self.tcx, super_scope) {
                    self.tcx.sess.span_help(span,
                        "consider using a `let` binding to increase its lifetime");
                }
            }

            err_borrowed_pointer_too_short(loan_scope, ptr_scope) => {
                let descr = match opt_loan_path(&err.cmt) {
                    Some(lp) => {
                        format!("`{}`", self.loan_path_to_string(&*lp))
                    }
                    None => self.cmt_to_string(&*err.cmt),
                };
                self.tcx.note_and_explain_region(
                    &format!("{} would have to be valid for ",
                            descr),
                    loan_scope,
                    "...");
                self.tcx.note_and_explain_region(
                    &format!("...but {} is only valid for ", descr),
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
                out.push_str(&self.tcx.local_var_name_str(id));
            }

            LpDowncast(ref lp_base, variant_def_id) => {
                out.push('(');
                self.append_loan_path_to_string(&**lp_base, out);
                out.push_str(DOWNCAST_PRINTED_OPERATOR);
                out.push_str(&self.tcx.item_path_str(variant_def_id));
                out.push(')');
            }


            LpExtend(ref lp_base, _, LpInterior(_, InteriorField(fname))) => {
                self.append_autoderefd_loan_path_to_string(&**lp_base, out);
                match fname {
                    mc::NamedField(fname) => {
                        out.push('.');
                        out.push_str(&fname.as_str());
                    }
                    mc::PositionalField(idx) => {
                        out.push('.');
                        out.push_str(&idx.to_string());
                    }
                }
            }

            LpExtend(ref lp_base, _, LpInterior(_, InteriorElement(..))) => {
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
                out.push_str(&self.tcx.item_path_str(variant_def_id));
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

    pub fn cmt_to_path_or_string(&self, cmt: &mc::cmt<'tcx>) -> String {
        match opt_loan_path(cmt) {
            Some(lp) => format!("`{}`", self.loan_path_to_string(&lp)),
            None => self.cmt_to_string(cmt),
        }
    }
}

fn statement_scope_span(tcx: &ty::ctxt, region: ty::Region) -> Option<Span> {
    match region {
        ty::ReScope(scope) => {
            match tcx.map.find(scope.node_id(&tcx.region_maps)) {
                Some(hir_map::NodeStmt(stmt)) => Some(stmt.span),
                _ => None
            }
        }
        _ => None
    }
}

impl BitwiseOperator for LoanDataFlowOperator {
    #[inline]
    fn join(&self, succ: usize, pred: usize) -> usize {
        succ | pred // loans from both preds are in scope
    }
}

impl DataFlowOperator for LoanDataFlowOperator {
    #[inline]
    fn initial_value(&self) -> bool {
        false // no loans in scope by default
    }
}

impl<'tcx> fmt::Debug for InteriorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            InteriorField(mc::NamedField(fld)) => write!(f, "{}", fld),
            InteriorField(mc::PositionalField(i)) => write!(f, "#{}", i),
            InteriorElement(..) => write!(f, "[]"),
        }
    }
}

impl<'tcx> fmt::Debug for Loan<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Loan_{}({:?}, {:?}, {:?}-{:?}, {:?})",
               self.index,
               self.loan_path,
               self.kind,
               self.gen_scope,
               self.kill_scope,
               self.restricted_paths)
    }
}

impl<'tcx> fmt::Debug for LoanPath<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.kind {
            LpVar(id) => {
                write!(f, "$({})", ty::tls::with(|tcx| tcx.map.node_to_string(id)))
            }

            LpUpvar(ty::UpvarId{ var_id, closure_expr_id }) => {
                let s = ty::tls::with(|tcx| tcx.map.node_to_string(var_id));
                write!(f, "$({} captured by id={})", s, closure_expr_id)
            }

            LpDowncast(ref lp, variant_def_id) => {
                let variant_str = if variant_def_id.is_local() {
                    ty::tls::with(|tcx| tcx.item_path_str(variant_def_id))
                } else {
                    format!("{:?}", variant_def_id)
                };
                write!(f, "({:?}{}{})", lp, DOWNCAST_PRINTED_OPERATOR, variant_str)
            }

            LpExtend(ref lp, _, LpDeref(_)) => {
                write!(f, "{:?}.*", lp)
            }

            LpExtend(ref lp, _, LpInterior(_, ref interior)) => {
                write!(f, "{:?}.{:?}", lp, interior)
            }
        }
    }
}

impl<'tcx> fmt::Display for LoanPath<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.kind {
            LpVar(id) => {
                write!(f, "$({})", ty::tls::with(|tcx| tcx.map.node_to_user_string(id)))
            }

            LpUpvar(ty::UpvarId{ var_id, closure_expr_id: _ }) => {
                let s = ty::tls::with(|tcx| tcx.map.node_to_user_string(var_id));
                write!(f, "$({} captured by closure)", s)
            }

            LpDowncast(ref lp, variant_def_id) => {
                let variant_str = if variant_def_id.is_local() {
                    ty::tls::with(|tcx| tcx.item_path_str(variant_def_id))
                } else {
                    format!("{:?}", variant_def_id)
                };
                write!(f, "({}{}{})", lp, DOWNCAST_PRINTED_OPERATOR, variant_str)
            }

            LpExtend(ref lp, _, LpDeref(_)) => {
                write!(f, "{}.*", lp)
            }

            LpExtend(ref lp, _, LpInterior(_, ref interior)) => {
                write!(f, "{}.{:?}", lp, interior)
            }
        }
    }
}
