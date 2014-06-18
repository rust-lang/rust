// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! See doc.rs for a thorough explanation of the borrow checker */

#![allow(non_camel_case_types)]

use middle::cfg;
use middle::dataflow::DataFlowContext;
use middle::dataflow::BitwiseOperator;
use middle::dataflow::DataFlowOperator;
use middle::def;
use euv = middle::expr_use_visitor;
use mc = middle::mem_categorization;
use middle::ty;
use util::ppaux::{note_and_explain_region, Repr, UserString};

use std::cell::{Cell};
use std::rc::Rc;
use std::gc::{Gc, GC};
use std::string::String;
use syntax::ast;
use syntax::ast_map;
use syntax::ast_util;
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::visit;
use syntax::visit::{Visitor, FnKind};
use syntax::ast::{FnDecl, Block, NodeId};

macro_rules! if_ok(
    ($inp: expr) => (
        match $inp {
            Ok(v) => { v }
            Err(e) => { return Err(e); }
        }
    )
)

pub mod doc;

pub mod check_loans;

pub mod gather_loans;

pub mod move_data;

#[deriving(Clone)]
pub struct LoanDataFlowOperator;

pub type LoanDataFlow<'a> = DataFlowContext<'a, LoanDataFlowOperator>;

impl<'a> Visitor<()> for BorrowckCtxt<'a> {
    fn visit_fn(&mut self, fk: &FnKind, fd: &FnDecl,
                b: &Block, s: Span, n: NodeId, _: ()) {
        borrowck_fn(self, fk, fd, b, s, n);
    }

    fn visit_item(&mut self, item: &ast::Item, _: ()) {
        borrowck_item(self, item);
    }
}

pub fn check_crate(tcx: &ty::ctxt,
                   krate: &ast::Crate) {
    let mut bccx = BorrowckCtxt {
        tcx: tcx,
        stats: box(GC) BorrowStats {
            loaned_paths_same: Cell::new(0),
            loaned_paths_imm: Cell::new(0),
            stable_paths: Cell::new(0),
            guaranteed_paths: Cell::new(0),
        }
    };

    visit::walk_crate(&mut bccx, krate, ());

    if tcx.sess.borrowck_stats() {
        println!("--- borrowck stats ---");
        println!("paths requiring guarantees: {}",
                 bccx.stats.guaranteed_paths.get());
        println!("paths requiring loans     : {}",
                 make_stat(&bccx, bccx.stats.loaned_paths_same.get()));
        println!("paths requiring imm loans : {}",
                 make_stat(&bccx, bccx.stats.loaned_paths_imm.get()));
        println!("stable paths              : {}",
                 make_stat(&bccx, bccx.stats.stable_paths.get()));
    }

    fn make_stat(bccx: &BorrowckCtxt, stat: uint) -> String {
        let stat_f = stat as f64;
        let total = bccx.stats.guaranteed_paths.get() as f64;
        format!("{} ({:.0f}%)", stat  , stat_f * 100.0 / total)
    }
}

fn borrowck_item(this: &mut BorrowckCtxt, item: &ast::Item) {
    // Gather loans for items. Note that we don't need
    // to check loans for single expressions. The check
    // loan step is intended for things that have a data
    // flow dependent conditions.
    match item.node {
        ast::ItemStatic(_, _, ex) => {
            gather_loans::gather_loans_in_static_initializer(this, &*ex);
        }
        _ => {
            visit::walk_item(this, item, ());
        }
    }
}

fn borrowck_fn(this: &mut BorrowckCtxt,
               fk: &FnKind,
               decl: &ast::FnDecl,
               body: &ast::Block,
               sp: Span,
               id: ast::NodeId) {
    debug!("borrowck_fn(id={})", id);

    // Check the body of fn items.
    let id_range = ast_util::compute_id_range_for_fn_body(fk, decl, body, sp, id);
    let (all_loans, move_data) =
        gather_loans::gather_loans_in_fn(this, decl, body);
    let cfg = cfg::CFG::new(this.tcx, body);

    let mut loan_dfcx =
        DataFlowContext::new(this.tcx,
                             "borrowck",
                             Some(decl),
                             &cfg,
                             LoanDataFlowOperator,
                             id_range,
                             all_loans.len());
    for (loan_idx, loan) in all_loans.iter().enumerate() {
        loan_dfcx.add_gen(loan.gen_scope, loan_idx);
        loan_dfcx.add_kill(loan.kill_scope, loan_idx);
    }
    loan_dfcx.add_kills_from_flow_exits(&cfg);
    loan_dfcx.propagate(&cfg, body);

    let flowed_moves = move_data::FlowedMoveData::new(move_data,
                                                      this.tcx,
                                                      &cfg,
                                                      id_range,
                                                      decl,
                                                      body);

    check_loans::check_loans(this, &loan_dfcx, flowed_moves,
                             all_loans.as_slice(), decl, body);

    visit::walk_fn(this, fk, decl, body, sp, ());
}

// ----------------------------------------------------------------------
// Type definitions

pub struct BorrowckCtxt<'a> {
    tcx: &'a ty::ctxt,

    // Statistics:
    stats: Gc<BorrowStats>,
}

pub struct BorrowStats {
    loaned_paths_same: Cell<uint>,
    loaned_paths_imm: Cell<uint>,
    stable_paths: Cell<uint>,
    guaranteed_paths: Cell<uint>,
}

pub type BckResult<T> = Result<T, BckError>;

#[deriving(PartialEq)]
pub enum PartialTotal {
    Partial,   // Loan affects some portion
    Total      // Loan affects entire path
}

///////////////////////////////////////////////////////////////////////////
// Loans and loan paths

/// Record of a loan that was issued.
pub struct Loan {
    index: uint,
    loan_path: Rc<LoanPath>,
    kind: ty::BorrowKind,
    restricted_paths: Vec<Rc<LoanPath>>,
    gen_scope: ast::NodeId,
    kill_scope: ast::NodeId,
    span: Span,
    cause: euv::LoanCause,
}

#[deriving(PartialEq, Eq, Hash)]
pub enum LoanPath {
    LpVar(ast::NodeId),               // `x` in doc.rs
    LpUpvar(ty::UpvarId),             // `x` captured by-value into closure
    LpExtend(Rc<LoanPath>, mc::MutabilityCategory, LoanPathElem)
}

#[deriving(PartialEq, Eq, Hash)]
pub enum LoanPathElem {
    LpDeref(mc::PointerKind),    // `*LV` in doc.rs
    LpInterior(mc::InteriorKind) // `LV.f` in doc.rs
}

pub fn closure_to_block(closure_id: ast::NodeId,
                    tcx: &ty::ctxt) -> ast::NodeId {
    match tcx.map.get(closure_id) {
        ast_map::NodeExpr(expr) => match expr.node {
            ast::ExprProc(_decl, block) |
            ast::ExprFnBlock(_decl, block) => { block.id }
            _ => fail!("encountered non-closure id: {}", closure_id)
        },
        _ => fail!("encountered non-expr id: {}", closure_id)
    }
}

impl LoanPath {
    pub fn kill_scope(&self, tcx: &ty::ctxt) -> ast::NodeId {
        match *self {
            LpVar(local_id) => tcx.region_maps.var_scope(local_id),
            LpUpvar(upvar_id) =>
                closure_to_block(upvar_id.closure_expr_id, tcx),
            LpExtend(ref base, _, _) => base.kill_scope(tcx),
        }
    }
}

pub fn opt_loan_path(cmt: &mc::cmt) -> Option<Rc<LoanPath>> {
    //! Computes the `LoanPath` (if any) for a `cmt`.
    //! Note that this logic is somewhat duplicated in
    //! the method `compute()` found in `gather_loans::restrictions`,
    //! which allows it to share common loan path pieces as it
    //! traverses the CMT.

    match cmt.cat {
        mc::cat_rvalue(..) |
        mc::cat_static_item |
        mc::cat_copied_upvar(mc::CopiedUpvar { onceness: ast::Many, .. }) => {
            None
        }

        mc::cat_local(id) |
        mc::cat_arg(id) => {
            Some(Rc::new(LpVar(id)))
        }

        mc::cat_upvar(ty::UpvarId {var_id: id, closure_expr_id: proc_id}, _) |
        mc::cat_copied_upvar(mc::CopiedUpvar { upvar_id: id,
                                               onceness: _,
                                               capturing_proc: proc_id }) => {
            let upvar_id = ty::UpvarId{ var_id: id, closure_expr_id: proc_id };
            Some(Rc::new(LpUpvar(upvar_id)))
        }

        mc::cat_deref(ref cmt_base, _, pk) => {
            opt_loan_path(cmt_base).map(|lp| {
                Rc::new(LpExtend(lp, cmt.mutbl, LpDeref(pk)))
            })
        }

        mc::cat_interior(ref cmt_base, ik) => {
            opt_loan_path(cmt_base).map(|lp| {
                Rc::new(LpExtend(lp, cmt.mutbl, LpInterior(ik)))
            })
        }

        mc::cat_downcast(ref cmt_base) |
        mc::cat_discr(ref cmt_base, _) => {
            opt_loan_path(cmt_base)
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Errors

// Errors that can occur
#[deriving(PartialEq)]
pub enum bckerr_code {
    err_mutbl,
    err_out_of_scope(ty::Region, ty::Region), // superscope, subscope
    err_borrowed_pointer_too_short(ty::Region, ty::Region), // loan, ptr
}

// Combination of an error code and the categorization of the expression
// that caused it
#[deriving(PartialEq)]
pub struct BckError {
    span: Span,
    cause: euv::LoanCause,
    cmt: mc::cmt,
    code: bckerr_code
}

pub enum AliasableViolationKind {
    MutabilityViolation,
    BorrowViolation(euv::LoanCause)
}

pub enum MovedValueUseKind {
    MovedInUse,
    MovedInCapture,
}

///////////////////////////////////////////////////////////////////////////
// Misc

impl<'a> BorrowckCtxt<'a> {
    pub fn is_subregion_of(&self, r_sub: ty::Region, r_sup: ty::Region)
                           -> bool {
        self.tcx.region_maps.is_subregion_of(r_sub, r_sup)
    }

    pub fn is_subscope_of(&self, r_sub: ast::NodeId, r_sup: ast::NodeId)
                          -> bool {
        self.tcx.region_maps.is_subscope_of(r_sub, r_sup)
    }

    pub fn mc(&self) -> mc::MemCategorizationContext<'a,ty::ctxt> {
        mc::MemCategorizationContext::new(self.tcx)
    }

    pub fn cat_expr(&self, expr: &ast::Expr) -> mc::cmt {
        match self.mc().cat_expr(expr) {
            Ok(c) => c,
            Err(()) => {
                self.tcx.sess.span_bug(expr.span, "error in mem categorization");
            }
        }
    }

    pub fn cat_expr_unadjusted(&self, expr: &ast::Expr) -> mc::cmt {
        match self.mc().cat_expr_unadjusted(expr) {
            Ok(c) => c,
            Err(()) => {
                self.tcx.sess.span_bug(expr.span, "error in mem categorization");
            }
        }
    }

    pub fn cat_expr_autoderefd(&self,
                               expr: &ast::Expr,
                               adj: &ty::AutoAdjustment)
                               -> mc::cmt {
        let r = match *adj {
            ty::AutoAddEnv(..) | ty::AutoObject(..) => {
                // no autoderefs
                self.mc().cat_expr_unadjusted(expr)
            }

            ty::AutoDerefRef(
                ty::AutoDerefRef {
                    autoderefs: autoderefs, ..}) => {
                self.mc().cat_expr_autoderefd(expr, autoderefs)
            }
        };

        match r {
            Ok(c) => c,
            Err(()) => {
                self.tcx.sess.span_bug(expr.span,
                                       "error in mem categorization");
            }
        }
    }

    pub fn cat_def(&self,
                   id: ast::NodeId,
                   span: Span,
                   ty: ty::t,
                   def: def::Def)
                   -> mc::cmt {
        match self.mc().cat_def(id, span, ty, def) {
            Ok(c) => c,
            Err(()) => {
                self.tcx.sess.span_bug(span, "error in mem categorization");
            }
        }
    }

    pub fn cat_captured_var(&self,
                            closure_id: ast::NodeId,
                            closure_span: Span,
                            upvar_def: def::Def)
                            -> mc::cmt {
        // Create the cmt for the variable being borrowed, from the
        // caller's perspective
        let var_id = upvar_def.def_id().node;
        let var_ty = ty::node_id_to_type(self.tcx, var_id);
        self.cat_def(closure_id, closure_span, var_ty, upvar_def)
    }

    pub fn cat_discr(&self, cmt: mc::cmt, match_id: ast::NodeId) -> mc::cmt {
        Rc::new(mc::cmt_ {
            cat: mc::cat_discr(cmt.clone(), match_id),
            mutbl: cmt.mutbl.inherit(),
            ..*cmt
        })
    }

    pub fn cat_pattern(&self,
                       cmt: mc::cmt,
                       pat: &ast::Pat,
                       op: |mc::cmt, &ast::Pat|) {
        let r = self.mc().cat_pattern(cmt, pat, |_,x,y| op(x,y));
        assert!(r.is_ok());
    }

    pub fn report(&self, err: BckError) {
        self.span_err(
            err.span,
            self.bckerr_to_str(&err).as_slice());
        self.note_and_explain_bckerr(err);
    }

    pub fn report_use_of_moved_value(&self,
                                     use_span: Span,
                                     use_kind: MovedValueUseKind,
                                     lp: &LoanPath,
                                     move: &move_data::Move,
                                     moved_lp: &LoanPath) {
        let verb = match use_kind {
            MovedInUse => "use",
            MovedInCapture => "capture",
        };

        match move.kind {
            move_data::Declared => {
                self.tcx.sess.span_err(
                    use_span,
                    format!("{} of possibly uninitialized variable: `{}`",
                            verb,
                            self.loan_path_to_str(lp)).as_slice());
            }
            _ => {
                let partially = if lp == moved_lp {""} else {"partially "};
                self.tcx.sess.span_err(
                    use_span,
                    format!("{} of {}moved value: `{}`",
                            verb,
                            partially,
                            self.loan_path_to_str(lp)).as_slice());
            }
        }

        match move.kind {
            move_data::Declared => {}

            move_data::MoveExpr => {
                let (expr_ty, expr_span) = match self.tcx.map.find(move.id) {
                    Some(ast_map::NodeExpr(expr)) => {
                        (ty::expr_ty_adjusted(self.tcx, &*expr), expr.span)
                    }
                    r => {
                        self.tcx.sess.bug(format!("MoveExpr({:?}) maps to \
                                                   {:?}, not Expr",
                                                  move.id,
                                                  r).as_slice())
                    }
                };
                let suggestion = move_suggestion(self.tcx, expr_ty,
                        "moved by default (use `copy` to override)");
                self.tcx.sess.span_note(
                    expr_span,
                    format!("`{}` moved here because it has type `{}`, which is {}",
                            self.loan_path_to_str(moved_lp),
                            expr_ty.user_string(self.tcx),
                            suggestion).as_slice());
            }

            move_data::MovePat => {
                let pat_ty = ty::node_id_to_type(self.tcx, move.id);
                self.tcx.sess.span_note(self.tcx.map.span(move.id),
                    format!("`{}` moved here because it has type `{}`, \
                             which is moved by default (use `ref` to \
                             override)",
                            self.loan_path_to_str(moved_lp),
                            pat_ty.user_string(self.tcx)).as_slice());
            }

            move_data::Captured => {
                let (expr_ty, expr_span) = match self.tcx.map.find(move.id) {
                    Some(ast_map::NodeExpr(expr)) => {
                        (ty::expr_ty_adjusted(self.tcx, &*expr), expr.span)
                    }
                    r => {
                        self.tcx.sess.bug(format!("Captured({:?}) maps to \
                                                   {:?}, not Expr",
                                                  move.id,
                                                  r).as_slice())
                    }
                };
                let suggestion = move_suggestion(self.tcx, expr_ty,
                        "moved by default (make a copy and \
                         capture that instead to override)");
                self.tcx.sess.span_note(
                    expr_span,
                    format!("`{}` moved into closure environment here because it \
                            has type `{}`, which is {}",
                            self.loan_path_to_str(moved_lp),
                            expr_ty.user_string(self.tcx),
                            suggestion).as_slice());
            }
        }

        fn move_suggestion(tcx: &ty::ctxt, ty: ty::t, default_msg: &'static str)
                          -> &'static str {
            match ty::get(ty).sty {
                ty::ty_closure(box ty::ClosureTy {
                        store: ty::RegionTraitStore(..),
                        ..
                    }) =>
                    "a non-copyable stack closure (capture it in a new closure, \
                     e.g. `|x| f(x)`, to override)",
                _ if ty::type_moves_by_default(tcx, ty) =>
                    "non-copyable (perhaps you meant to use clone()?)",
                _ => default_msg,
            }
        }
    }

    pub fn report_reassigned_immutable_variable(&self,
                                                span: Span,
                                                lp: &LoanPath,
                                                assign:
                                                &move_data::Assignment) {
        self.tcx.sess.span_err(
            span,
            format!("re-assignment of immutable variable `{}`",
                    self.loan_path_to_str(lp)).as_slice());
        self.tcx.sess.span_note(assign.span, "prior assignment occurs here");
    }

    pub fn span_err(&self, s: Span, m: &str) {
        self.tcx.sess.span_err(s, m);
    }

    pub fn span_note(&self, s: Span, m: &str) {
        self.tcx.sess.span_note(s, m);
    }

    pub fn span_end_note(&self, s: Span, m: &str) {
        self.tcx.sess.span_end_note(s, m);
    }

    pub fn bckerr_to_str(&self, err: &BckError) -> String {
        match err.code {
            err_mutbl => {
                let descr = match opt_loan_path(&err.cmt) {
                    None => {
                        format!("{} {}",
                                err.cmt.mutbl.to_user_str(),
                                self.cmt_to_str(&*err.cmt))
                    }
                    Some(lp) => {
                        format!("{} {} `{}`",
                                err.cmt.mutbl.to_user_str(),
                                self.cmt_to_str(&*err.cmt),
                                self.loan_path_to_str(&*lp))
                    }
                };

                match err.cause {
                    euv::ClosureCapture(_) => {
                        format!("closure cannot assign to {}", descr)
                    }
                    euv::OverloadedOperator |
                    euv::AddrOf |
                    euv::RefBinding |
                    euv::AutoRef => {
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
                        format!("`{}`", self.loan_path_to_str(&*lp))
                    }
                };
                format!("{} does not live long enough", msg)
            }
            err_borrowed_pointer_too_short(..) => {
                let descr = match opt_loan_path(&err.cmt) {
                    Some(lp) => {
                        format!("`{}`", self.loan_path_to_str(&*lp))
                    }
                    None => self.cmt_to_str(&*err.cmt),
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
        let prefix = match kind {
            MutabilityViolation => {
                "cannot assign to data"
            }
            BorrowViolation(euv::ClosureCapture(_)) => {
                // I don't think we can get aliasability violations
                // with closure captures, so no need to come up with a
                // good error message. The reason this cannot happen
                // is because we only capture local variables in
                // closures, and those are never aliasable.
                self.tcx.sess.span_bug(
                    span,
                    "aliasability violation with closure");
            }
            BorrowViolation(euv::OverloadedOperator) |
            BorrowViolation(euv::AddrOf) |
            BorrowViolation(euv::AutoRef) |
            BorrowViolation(euv::RefBinding) => {
                "cannot borrow data mutably"
            }

            BorrowViolation(euv::ClosureInvocation) => {
                "closure invocation"
            }
        };

        match cause {
            mc::AliasableOther => {
                self.tcx.sess.span_err(
                    span,
                    format!("{} in an aliasable location",
                             prefix).as_slice());
            }
            mc::AliasableStatic(..) |
            mc::AliasableStaticMut(..) => {
                self.tcx.sess.span_err(
                    span,
                    format!("{} in a static location", prefix).as_slice());
            }
            mc::AliasableManaged => {
                self.tcx.sess.span_err(
                    span,
                    format!("{} in a `@` pointer", prefix).as_slice());
            }
            mc::AliasableBorrowed => {
                self.tcx.sess.span_err(
                    span,
                    format!("{} in a `&` reference", prefix).as_slice());
            }
        }
    }

    pub fn note_and_explain_bckerr(&self, err: BckError) {
        let code = err.code;
        match code {
            err_mutbl(..) => { }

            err_out_of_scope(super_scope, sub_scope) => {
                note_and_explain_region(
                    self.tcx,
                    "reference must be valid for ",
                    sub_scope,
                    "...");
                let suggestion = if is_statement_scope(self.tcx, super_scope) {
                    "; consider using a `let` binding to increase its lifetime"
                } else {
                    ""
                };
                note_and_explain_region(
                    self.tcx,
                    "...but borrowed value is only valid for ",
                    super_scope,
                    suggestion);
            }

            err_borrowed_pointer_too_short(loan_scope, ptr_scope) => {
                let descr = match opt_loan_path(&err.cmt) {
                    Some(lp) => {
                        format!("`{}`", self.loan_path_to_str(&*lp))
                    }
                    None => self.cmt_to_str(&*err.cmt),
                };
                note_and_explain_region(
                    self.tcx,
                    format!("{} would have to be valid for ",
                            descr).as_slice(),
                    loan_scope,
                    "...");
                note_and_explain_region(
                    self.tcx,
                    format!("...but {} is only valid for ", descr).as_slice(),
                    ptr_scope,
                    "");
            }
        }
    }

    pub fn append_loan_path_to_str(&self,
                                   loan_path: &LoanPath,
                                   out: &mut String) {
        match *loan_path {
            LpUpvar(ty::UpvarId{ var_id: id, closure_expr_id: _ }) |
            LpVar(id) => {
                out.push_str(ty::local_var_name_str(self.tcx, id).get());
            }

            LpExtend(ref lp_base, _, LpInterior(mc::InteriorField(fname))) => {
                self.append_autoderefd_loan_path_to_str(&**lp_base, out);
                match fname {
                    mc::NamedField(fname) => {
                        out.push_char('.');
                        out.push_str(token::get_name(fname).get());
                    }
                    mc::PositionalField(idx) => {
                        out.push_char('#'); // invent a notation here
                        out.push_str(idx.to_str().as_slice());
                    }
                }
            }

            LpExtend(ref lp_base, _, LpInterior(mc::InteriorElement(_))) => {
                self.append_autoderefd_loan_path_to_str(&**lp_base, out);
                out.push_str("[..]");
            }

            LpExtend(ref lp_base, _, LpDeref(_)) => {
                out.push_char('*');
                self.append_loan_path_to_str(&**lp_base, out);
            }
        }
    }

    pub fn append_autoderefd_loan_path_to_str(&self,
                                              loan_path: &LoanPath,
                                              out: &mut String) {
        match *loan_path {
            LpExtend(ref lp_base, _, LpDeref(_)) => {
                // For a path like `(*x).f` or `(*x)[3]`, autoderef
                // rules would normally allow users to omit the `*x`.
                // So just serialize such paths to `x.f` or x[3]` respectively.
                self.append_autoderefd_loan_path_to_str(&**lp_base, out)
            }

            LpVar(..) | LpUpvar(..) | LpExtend(_, _, LpInterior(..)) => {
                self.append_loan_path_to_str(loan_path, out)
            }
        }
    }

    pub fn loan_path_to_str(&self, loan_path: &LoanPath) -> String {
        let mut result = String::new();
        self.append_loan_path_to_str(loan_path, &mut result);
        result
    }

    pub fn cmt_to_str(&self, cmt: &mc::cmt_) -> String {
        self.mc().cmt_to_str(cmt)
    }
}

fn is_statement_scope(tcx: &ty::ctxt, region: ty::Region) -> bool {
     match region {
         ty::ReScope(node_id) => {
             match tcx.map.find(node_id) {
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

impl Repr for Loan {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        (format!("Loan_{:?}({}, {:?}, {:?}-{:?}, {})",
                 self.index,
                 self.loan_path.repr(tcx),
                 self.kind,
                 self.gen_scope,
                 self.kill_scope,
                 self.restricted_paths.repr(tcx))).to_string()
    }
}

impl Repr for LoanPath {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        match self {
            &LpVar(id) => {
                (format!("$({})", tcx.map.node_to_str(id))).to_string()
            }

            &LpUpvar(ty::UpvarId{ var_id, closure_expr_id }) => {
                let s = tcx.map.node_to_str(var_id);
                let s = format!("$({} captured by id={})", s, closure_expr_id);
                s.to_string()
            }

            &LpExtend(ref lp, _, LpDeref(_)) => {
                (format!("{}.*", lp.repr(tcx))).to_string()
            }

            &LpExtend(ref lp, _, LpInterior(ref interior)) => {
                (format!("{}.{}",
                         lp.repr(tcx),
                         interior.repr(tcx))).to_string()
            }
        }
    }
}
