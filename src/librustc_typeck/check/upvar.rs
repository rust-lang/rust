// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! ### Inferring borrow kinds for upvars
//!
//! Whenever there is a closure expression, we need to determine how each
//! upvar is used. We do this by initially assigning each upvar an
//! immutable "borrow kind" (see `ty::BorrowKind` for details) and then
//! "escalating" the kind as needed. The borrow kind proceeds according to
//! the following lattice:
//!
//!     ty::ImmBorrow -> ty::UniqueImmBorrow -> ty::MutBorrow
//!
//! So, for example, if we see an assignment `x = 5` to an upvar `x`, we
//! will promote its borrow kind to mutable borrow. If we see an `&mut x`
//! we'll do the same. Naturally, this applies not just to the upvar, but
//! to everything owned by `x`, so the result is the same for something
//! like `x.f = 5` and so on (presuming `x` is not a borrowed pointer to a
//! struct). These adjustments are performed in
//! `adjust_upvar_borrow_kind()` (you can trace backwards through the code
//! from there).
//!
//! The fact that we are inferring borrow kinds as we go results in a
//! semi-hacky interaction with mem-categorization. In particular,
//! mem-categorization will query the current borrow kind as it
//! categorizes, and we'll return the *current* value, but this may get
//! adjusted later. Therefore, in this module, we generally ignore the
//! borrow kind (and derived mutabilities) that are returned from
//! mem-categorization, since they may be inaccurate. (Another option
//! would be to use a unification scheme, where instead of returning a
//! concrete borrow kind like `ty::ImmBorrow`, we return a
//! `ty::InferBorrow(upvar_id)` or something like that, but this would
//! then mean that all later passes would have to check for these figments
//! and report an error, and it just seems like more mess in the end.)

use super::FnCtxt;

use middle::expr_use_visitor as euv;
use middle::mem_categorization as mc;
use middle::ty::{self};
use middle::infer::{InferCtxt, UpvarRegion};
use syntax::ast;
use syntax::codemap::Span;
use syntax::visit::{self, Visitor};
use util::ppaux::Repr;

///////////////////////////////////////////////////////////////////////////
// PUBLIC ENTRY POINTS

pub fn closure_analyze_fn(fcx: &FnCtxt,
                          _id: ast::NodeId,
                          decl: &ast::FnDecl,
                          body: &ast::Block) {
    let mut seed = SeedBorrowKind::new(fcx);
    seed.visit_block(body);

    let mut adjust = AdjustBorrowKind::new(fcx);
    adjust.analyze_fn(decl, body);
}

///////////////////////////////////////////////////////////////////////////
// SEED BORROW KIND

struct SeedBorrowKind<'a,'tcx:'a> {
    fcx: &'a FnCtxt<'a,'tcx>,
}

impl<'a, 'tcx, 'v> Visitor<'v> for SeedBorrowKind<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &ast::Expr) {
        match expr.node {
            ast::ExprClosure(cc, _, _, ref body) => {
                self.check_closure(expr, cc, &**body);
            }

            _ => { }
        }

        visit::walk_expr(self, expr);
    }

    fn visit_fn(&mut self,
                fn_kind: visit::FnKind<'v>,
                decl: &'v ast::FnDecl,
                block: &'v ast::Block,
                span: Span,
                _id: ast::NodeId)
    {
        match fn_kind {
            visit::FkItemFn(..) | visit::FkMethod(..) => {
                // ignore nested fn items
            }
            visit::FkFnBlock => {
                visit::walk_fn(self, fn_kind, decl, block, span);
            }
        }
    }
}

impl<'a,'tcx> SeedBorrowKind<'a,'tcx> {
    fn new(fcx: &'a FnCtxt<'a,'tcx>) -> SeedBorrowKind<'a,'tcx> {
        SeedBorrowKind { fcx: fcx }
    }

    fn tcx(&self) -> &'a ty::ctxt<'tcx> {
        self.fcx.tcx()
    }

    fn infcx(&self) -> &'a InferCtxt<'a,'tcx> {
        self.fcx.infcx()
    }

    fn check_closure(&mut self,
                     expr: &ast::Expr,
                     capture_clause: ast::CaptureClause,
                     _body: &ast::Block)
    {
        let is_old_skool_closure = match self.fcx.expr_ty(expr).sty {
            _ => false,
        };

        match capture_clause {
            ast::CaptureByValue if !is_old_skool_closure => {
            }
            _ => {
                ty::with_freevars(self.tcx(), expr.id, |freevars| {
                    for freevar in freevars.iter() {
                        let var_node_id = freevar.def.local_node_id();
                        let upvar_id = ty::UpvarId { var_id: var_node_id,
                                                     closure_expr_id: expr.id };
                        debug!("seed upvar_id {}", upvar_id);
                        let origin = UpvarRegion(upvar_id, expr.span);
                        let freevar_region = self.infcx().next_region_var(origin);
                        let upvar_borrow = ty::UpvarBorrow { kind: ty::ImmBorrow,
                                                             region: freevar_region };
                        self.fcx.inh.upvar_borrow_map.borrow_mut().insert(upvar_id,
                                                                          upvar_borrow);
                    }
                });
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// ADJUST BORROW KIND

struct AdjustBorrowKind<'a,'tcx:'a> {
    fcx: &'a FnCtxt<'a,'tcx>
}

impl<'a,'tcx> AdjustBorrowKind<'a,'tcx>{
    fn new(fcx: &'a FnCtxt<'a,'tcx>) -> AdjustBorrowKind<'a,'tcx> {
        AdjustBorrowKind { fcx: fcx }
    }

    fn tcx(&self) -> &'a ty::ctxt<'tcx> {
        self.fcx.tcx()
    }

    fn analyze_fn(&mut self, decl: &ast::FnDecl, body: &ast::Block) {
        /*!
         * Analysis starting point.
         */

        self.visit_block(body);

        debug!("analyzing fn body with id {}", body.id);

        let mut euv = euv::ExprUseVisitor::new(self, self.fcx);
        euv.walk_fn(decl, body);
    }

    /// Indicates that `cmt` is being directly mutated (e.g., assigned
    /// to). If cmt contains any by-ref upvars, this implies that
    /// those upvars must be borrowed using an `&mut` borow.
    fn adjust_upvar_borrow_kind_for_mut(&mut self, cmt: mc::cmt<'tcx>) {
        debug!("adjust_upvar_borrow_kind_for_mut(cmt={})",
               cmt.repr(self.tcx()));

        match cmt.cat.clone() {
            mc::cat_deref(base, _, mc::Unique) |
            mc::cat_interior(base, _) |
            mc::cat_downcast(base, _) => {
                // Interior or owned data is mutable if base is
                // mutable, so iterate to the base.
                self.adjust_upvar_borrow_kind_for_mut(base);
            }

            mc::cat_deref(base, _, mc::BorrowedPtr(..)) |
            mc::cat_deref(base, _, mc::Implicit(..)) => {
                if let mc::NoteUpvarRef(upvar_id) = cmt.note {
                    // if this is an implicit deref of an
                    // upvar, then we need to modify the
                    // borrow_kind of the upvar to make sure it
                    // is inferred to mutable if necessary
                    let mut upvar_borrow_map = self.fcx.inh.upvar_borrow_map.borrow_mut();
                    let ub = &mut upvar_borrow_map[upvar_id];
                    self.adjust_upvar_borrow_kind(upvar_id, ub, ty::MutBorrow);
                } else {
                    // assignment to deref of an `&mut`
                    // borrowed pointer implies that the
                    // pointer itself must be unique, but not
                    // necessarily *mutable*
                    self.adjust_upvar_borrow_kind_for_unique(base);
                }
            }

            mc::cat_deref(_, _, mc::UnsafePtr(..)) |
            mc::cat_static_item |
            mc::cat_rvalue(_) |
            mc::cat_local(_) |
            mc::cat_upvar(..) => {
                return;
            }
        }
    }

    fn adjust_upvar_borrow_kind_for_unique(&self, cmt: mc::cmt<'tcx>) {
        debug!("adjust_upvar_borrow_kind_for_unique(cmt={})",
               cmt.repr(self.tcx()));

        match cmt.cat.clone() {
            mc::cat_deref(base, _, mc::Unique) |
            mc::cat_interior(base, _) |
            mc::cat_downcast(base, _) => {
                // Interior or owned data is unique if base is
                // unique.
                self.adjust_upvar_borrow_kind_for_unique(base);
            }

            mc::cat_deref(base, _, mc::BorrowedPtr(..)) |
            mc::cat_deref(base, _, mc::Implicit(..)) => {
                if let mc::NoteUpvarRef(upvar_id) = cmt.note {
                    // if this is an implicit deref of an
                    // upvar, then we need to modify the
                    // borrow_kind of the upvar to make sure it
                    // is inferred to unique if necessary
                    let mut ub = self.fcx.inh.upvar_borrow_map.borrow_mut();
                    let ub = &mut ub[upvar_id];
                    self.adjust_upvar_borrow_kind(upvar_id, ub, ty::UniqueImmBorrow);
                } else {
                    // for a borrowed pointer to be unique, its
                    // base must be unique
                    self.adjust_upvar_borrow_kind_for_unique(base);
                }
            }

            mc::cat_deref(_, _, mc::UnsafePtr(..)) |
            mc::cat_static_item |
            mc::cat_rvalue(_) |
            mc::cat_local(_) |
            mc::cat_upvar(..) => {
            }
        }
    }

    /// We infer the borrow_kind with which to borrow upvars in a stack closure. The borrow_kind
    /// basically follows a lattice of `imm < unique-imm < mut`, moving from left to right as needed
    /// (but never right to left). Here the argument `mutbl` is the borrow_kind that is required by
    /// some particular use.
    fn adjust_upvar_borrow_kind(&self,
                                upvar_id: ty::UpvarId,
                                upvar_borrow: &mut ty::UpvarBorrow,
                                kind: ty::BorrowKind) {
        debug!("adjust_upvar_borrow_kind: id={} kind=({} -> {})",
               upvar_id, upvar_borrow.kind, kind);

        match (upvar_borrow.kind, kind) {
            // Take RHS:
            (ty::ImmBorrow, ty::UniqueImmBorrow) |
            (ty::ImmBorrow, ty::MutBorrow) |
            (ty::UniqueImmBorrow, ty::MutBorrow) => {
                upvar_borrow.kind = kind;
            }
            // Take LHS:
            (ty::ImmBorrow, ty::ImmBorrow) |
            (ty::UniqueImmBorrow, ty::ImmBorrow) |
            (ty::UniqueImmBorrow, ty::UniqueImmBorrow) |
            (ty::MutBorrow, _) => {
            }
        }
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for AdjustBorrowKind<'a, 'tcx> {
    fn visit_fn(&mut self,
                fn_kind: visit::FnKind<'v>,
                decl: &'v ast::FnDecl,
                body: &'v ast::Block,
                span: Span,
                _id: ast::NodeId)
    {
        match fn_kind {
            visit::FkItemFn(..) | visit::FkMethod(..) => {
                // ignore nested fn items
            }
            visit::FkFnBlock => {
                self.analyze_fn(decl, body);
                visit::walk_fn(self, fn_kind, decl, body, span);
            }
        }
    }
}

impl<'a,'tcx> euv::Delegate<'tcx> for AdjustBorrowKind<'a,'tcx> {
    fn consume(&mut self,
               _consume_id: ast::NodeId,
               _consume_span: Span,
               _cmt: mc::cmt<'tcx>,
               _mode: euv::ConsumeMode)
    {}

    fn matched_pat(&mut self,
                   _matched_pat: &ast::Pat,
                   _cmt: mc::cmt<'tcx>,
                   _mode: euv::MatchMode)
    {}

    fn consume_pat(&mut self,
                   _consume_pat: &ast::Pat,
                   _cmt: mc::cmt<'tcx>,
                   _mode: euv::ConsumeMode)
    {}

    fn borrow(&mut self,
              borrow_id: ast::NodeId,
              _borrow_span: Span,
              cmt: mc::cmt<'tcx>,
              _loan_region: ty::Region,
              bk: ty::BorrowKind,
              _loan_cause: euv::LoanCause)
    {
        debug!("borrow(borrow_id={}, cmt={}, bk={})",
               borrow_id, cmt.repr(self.tcx()), bk);

        match bk {
            ty::ImmBorrow => { }
            ty::UniqueImmBorrow => {
                self.adjust_upvar_borrow_kind_for_unique(cmt);
            }
            ty::MutBorrow => {
                self.adjust_upvar_borrow_kind_for_mut(cmt);
            }
        }
    }

    fn decl_without_init(&mut self,
                         _id: ast::NodeId,
                         _span: Span)
    {}

    fn mutate(&mut self,
              _assignment_id: ast::NodeId,
              _assignment_span: Span,
              assignee_cmt: mc::cmt<'tcx>,
              _mode: euv::MutateMode)
    {
        debug!("mutate(assignee_cmt={})",
               assignee_cmt.repr(self.tcx()));

        self.adjust_upvar_borrow_kind_for_mut(assignee_cmt);
    }
}


