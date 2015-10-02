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

use check::demand;
use middle::expr_use_visitor as euv;
use middle::mem_categorization as mc;
use middle::ty::{self, Ty};
use middle::infer::{InferCtxt, UpvarRegion};
use std::collections::HashSet;
use syntax::ast;
use syntax::codemap::Span;
use rustc_front::hir;
use rustc_front::visit::{self, Visitor};

///////////////////////////////////////////////////////////////////////////
// PUBLIC ENTRY POINTS

pub fn closure_analyze_fn(fcx: &FnCtxt,
                          _id: ast::NodeId,
                          _decl: &hir::FnDecl,
                          body: &hir::Block)
{
    let mut seed = SeedBorrowKind::new(fcx);
    seed.visit_block(body);
    let closures_with_inferred_kinds = seed.closures_with_inferred_kinds;

    let mut adjust = AdjustBorrowKind::new(fcx, &closures_with_inferred_kinds);
    adjust.visit_block(body);

    // it's our job to process these.
    assert!(fcx.inh.deferred_call_resolutions.borrow().is_empty());
}

pub fn closure_analyze_const(fcx: &FnCtxt,
                             body: &hir::Expr)
{
    let mut seed = SeedBorrowKind::new(fcx);
    seed.visit_expr(body);
    let closures_with_inferred_kinds = seed.closures_with_inferred_kinds;

    let mut adjust = AdjustBorrowKind::new(fcx, &closures_with_inferred_kinds);
    adjust.visit_expr(body);

    // it's our job to process these.
    assert!(fcx.inh.deferred_call_resolutions.borrow().is_empty());
}

///////////////////////////////////////////////////////////////////////////
// SEED BORROW KIND

struct SeedBorrowKind<'a,'tcx:'a> {
    fcx: &'a FnCtxt<'a,'tcx>,
    closures_with_inferred_kinds: HashSet<ast::NodeId>,
}

impl<'a, 'tcx, 'v> Visitor<'v> for SeedBorrowKind<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &hir::Expr) {
        match expr.node {
            hir::ExprClosure(cc, _, ref body) => {
                self.check_closure(expr, cc, &**body);
            }

            _ => { }
        }

        visit::walk_expr(self, expr);
    }

    // Skip all items; they aren't in the same context.
    fn visit_item(&mut self, _: &'v hir::Item) { }
}

impl<'a,'tcx> SeedBorrowKind<'a,'tcx> {
    fn new(fcx: &'a FnCtxt<'a,'tcx>) -> SeedBorrowKind<'a,'tcx> {
        SeedBorrowKind { fcx: fcx, closures_with_inferred_kinds: HashSet::new() }
    }

    fn tcx(&self) -> &'a ty::ctxt<'tcx> {
        self.fcx.tcx()
    }

    fn infcx(&self) -> &'a InferCtxt<'a,'tcx> {
        self.fcx.infcx()
    }

    fn check_closure(&mut self,
                     expr: &hir::Expr,
                     capture_clause: hir::CaptureClause,
                     _body: &hir::Block)
    {
        let closure_def_id = self.tcx().map.local_def_id(expr.id);
        if !self.fcx.inh.tables.borrow().closure_kinds.contains_key(&closure_def_id) {
            self.closures_with_inferred_kinds.insert(expr.id);
            self.fcx.inh.tables.borrow_mut().closure_kinds
                                            .insert(closure_def_id, ty::FnClosureKind);
            debug!("check_closure: adding closure_id={:?} to closures_with_inferred_kinds",
                   closure_def_id);
        }

        self.tcx().with_freevars(expr.id, |freevars| {
            for freevar in freevars {
                let var_node_id = freevar.def.var_id();
                let upvar_id = ty::UpvarId { var_id: var_node_id,
                                             closure_expr_id: expr.id };
                debug!("seed upvar_id {:?}", upvar_id);

                let capture_kind = match capture_clause {
                    hir::CaptureByValue => {
                        ty::UpvarCapture::ByValue
                    }
                    hir::CaptureByRef => {
                        let origin = UpvarRegion(upvar_id, expr.span);
                        let freevar_region = self.infcx().next_region_var(origin);
                        let upvar_borrow = ty::UpvarBorrow { kind: ty::ImmBorrow,
                                                             region: freevar_region };
                        ty::UpvarCapture::ByRef(upvar_borrow)
                    }
                };

                self.fcx.inh.tables.borrow_mut().upvar_capture_map.insert(upvar_id, capture_kind);
            }
        });
    }
}

///////////////////////////////////////////////////////////////////////////
// ADJUST BORROW KIND

struct AdjustBorrowKind<'a,'tcx:'a> {
    fcx: &'a FnCtxt<'a,'tcx>,
    closures_with_inferred_kinds: &'a HashSet<ast::NodeId>,
}

impl<'a,'tcx> AdjustBorrowKind<'a,'tcx> {
    fn new(fcx: &'a FnCtxt<'a,'tcx>,
           closures_with_inferred_kinds: &'a HashSet<ast::NodeId>)
           -> AdjustBorrowKind<'a,'tcx> {
        AdjustBorrowKind { fcx: fcx, closures_with_inferred_kinds: closures_with_inferred_kinds }
    }

    fn analyze_closure(&mut self,
                       id: ast::NodeId,
                       span: Span,
                       decl: &hir::FnDecl,
                       body: &hir::Block) {
        /*!
         * Analysis starting point.
         */

        debug!("analyze_closure(id={:?}, body.id={:?})", id, body.id);

        {
            let mut euv = euv::ExprUseVisitor::new(self, self.fcx.infcx());
            euv.walk_fn(decl, body);
        }

        // Now that we've analyzed the closure, we know how each
        // variable is borrowed, and we know what traits the closure
        // implements (Fn vs FnMut etc). We now have some updates to do
        // with that information.
        //
        // Note that no closure type C may have an upvar of type C
        // (though it may reference itself via a trait object). This
        // results from the desugaring of closures to a struct like
        // `Foo<..., UV0...UVn>`. If one of those upvars referenced
        // C, then the type would have infinite size (and the
        // inference algorithm will reject it).

        // Extract the type variables UV0...UVn.
        let closure_substs = match self.fcx.node_ty(id).sty {
            ty::TyClosure(_, ref substs) => substs,
            ref t => {
                self.fcx.tcx().sess.span_bug(
                    span,
                    &format!("type of closure expr {:?} is not a closure {:?}",
                             id, t));
            }
        };

        // Equate the type variables with the actual types.
        let final_upvar_tys = self.final_upvar_tys(id);
        debug!("analyze_closure: id={:?} closure_substs={:?} final_upvar_tys={:?}",
               id, closure_substs, final_upvar_tys);
        for (&upvar_ty, final_upvar_ty) in closure_substs.upvar_tys.iter().zip(final_upvar_tys) {
            demand::eqtype(self.fcx, span, final_upvar_ty, upvar_ty);
        }

        // Now we must process and remove any deferred resolutions,
        // since we have a concrete closure kind.
        let closure_def_id = self.fcx.tcx().map.local_def_id(id);
        if self.closures_with_inferred_kinds.contains(&id) {
            let mut deferred_call_resolutions =
                self.fcx.remove_deferred_call_resolutions(closure_def_id);
            for deferred_call_resolution in &mut deferred_call_resolutions {
                deferred_call_resolution.resolve(self.fcx);
            }
        }
    }

    // Returns a list of `ClosureUpvar`s for each upvar.
    fn final_upvar_tys(&mut self, closure_id: ast::NodeId) -> Vec<Ty<'tcx>> {
        // Presently an unboxed closure type cannot "escape" out of a
        // function, so we will only encounter ones that originated in the
        // local crate or were inlined into it along with some function.
        // This may change if abstract return types of some sort are
        // implemented.
        let tcx = self.fcx.tcx();
        tcx.with_freevars(closure_id, |freevars| {
            freevars.iter()
                    .map(|freevar| {
                        let freevar_node_id = freevar.def.var_id();
                        let freevar_ty = self.fcx.node_ty(freevar_node_id);
                        let upvar_id = ty::UpvarId {
                            var_id: freevar_node_id,
                            closure_expr_id: closure_id
                        };
                        let capture = self.fcx.infcx().upvar_capture(upvar_id).unwrap();

                        debug!("freevar_node_id={:?} freevar_ty={:?} capture={:?}",
                               freevar_node_id, freevar_ty, capture);

                        match capture {
                            ty::UpvarCapture::ByValue => freevar_ty,
                            ty::UpvarCapture::ByRef(borrow) =>
                                tcx.mk_ref(tcx.mk_region(borrow.region),
                                           ty::TypeAndMut {
                                               ty: freevar_ty,
                                               mutbl: borrow.kind.to_mutbl_lossy(),
                                           }),
                        }
                    })
                    .collect()
            })
    }

    fn adjust_upvar_borrow_kind_for_consume(&self,
                                            cmt: mc::cmt<'tcx>,
                                            mode: euv::ConsumeMode)
    {
        debug!("adjust_upvar_borrow_kind_for_consume(cmt={:?}, mode={:?})",
               cmt, mode);

        // we only care about moves
        match mode {
            euv::Copy => { return; }
            euv::Move(_) => { }
        }

        // watch out for a move of the deref of a borrowed pointer;
        // for that to be legal, the upvar would have to be borrowed
        // by value instead
        let guarantor = cmt.guarantor();
        debug!("adjust_upvar_borrow_kind_for_consume: guarantor={:?}",
               guarantor);
        match guarantor.cat {
            mc::cat_deref(_, _, mc::BorrowedPtr(..)) |
            mc::cat_deref(_, _, mc::Implicit(..)) => {
                match cmt.note {
                    mc::NoteUpvarRef(upvar_id) => {
                        debug!("adjust_upvar_borrow_kind_for_consume: \
                                setting upvar_id={:?} to by value",
                               upvar_id);

                        // to move out of an upvar, this must be a FnOnce closure
                        self.adjust_closure_kind(upvar_id.closure_expr_id, ty::FnOnceClosureKind);

                        let upvar_capture_map =
                            &mut self.fcx.inh.tables.borrow_mut().upvar_capture_map;
                        upvar_capture_map.insert(upvar_id, ty::UpvarCapture::ByValue);
                    }
                    mc::NoteClosureEnv(upvar_id) => {
                        // we get just a closureenv ref if this is a
                        // `move` closure, or if the upvar has already
                        // been inferred to by-value. In any case, we
                        // must still adjust the kind of the closure
                        // to be a FnOnce closure to permit moves out
                        // of the environment.
                        self.adjust_closure_kind(upvar_id.closure_expr_id, ty::FnOnceClosureKind);
                    }
                    mc::NoteNone => {
                    }
                }
            }
            _ => { }
        }
    }

    /// Indicates that `cmt` is being directly mutated (e.g., assigned
    /// to). If cmt contains any by-ref upvars, this implies that
    /// those upvars must be borrowed using an `&mut` borrow.
    fn adjust_upvar_borrow_kind_for_mut(&mut self, cmt: mc::cmt<'tcx>) {
        debug!("adjust_upvar_borrow_kind_for_mut(cmt={:?})",
               cmt);

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
                if !self.try_adjust_upvar_deref(&cmt.note, ty::MutBorrow) {
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
        debug!("adjust_upvar_borrow_kind_for_unique(cmt={:?})",
               cmt);

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
                if !self.try_adjust_upvar_deref(&cmt.note, ty::UniqueImmBorrow) {
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

    fn try_adjust_upvar_deref(&self,
                              note: &mc::Note,
                              borrow_kind: ty::BorrowKind)
                              -> bool
    {
        assert!(match borrow_kind {
            ty::MutBorrow => true,
            ty::UniqueImmBorrow => true,

            // imm borrows never require adjusting any kinds, so we don't wind up here
            ty::ImmBorrow => false,
        });

        match *note {
            mc::NoteUpvarRef(upvar_id) => {
                // if this is an implicit deref of an
                // upvar, then we need to modify the
                // borrow_kind of the upvar to make sure it
                // is inferred to mutable if necessary
                {
                    let upvar_capture_map = &mut self.fcx.inh.tables.borrow_mut().upvar_capture_map;
                    let ub = upvar_capture_map.get_mut(&upvar_id).unwrap();
                    self.adjust_upvar_borrow_kind(upvar_id, ub, borrow_kind);
                }

                // also need to be in an FnMut closure since this is not an ImmBorrow
                self.adjust_closure_kind(upvar_id.closure_expr_id, ty::FnMutClosureKind);

                true
            }
            mc::NoteClosureEnv(upvar_id) => {
                // this kind of deref occurs in a `move` closure, or
                // for a by-value upvar; in either case, to mutate an
                // upvar, we need to be an FnMut closure
                self.adjust_closure_kind(upvar_id.closure_expr_id, ty::FnMutClosureKind);

                true
            }
            mc::NoteNone => {
                false
            }
        }
    }

    /// We infer the borrow_kind with which to borrow upvars in a stack closure. The borrow_kind
    /// basically follows a lattice of `imm < unique-imm < mut`, moving from left to right as needed
    /// (but never right to left). Here the argument `mutbl` is the borrow_kind that is required by
    /// some particular use.
    fn adjust_upvar_borrow_kind(&self,
                                upvar_id: ty::UpvarId,
                                upvar_capture: &mut ty::UpvarCapture,
                                kind: ty::BorrowKind) {
        debug!("adjust_upvar_borrow_kind(upvar_id={:?}, upvar_capture={:?}, kind={:?})",
               upvar_id, upvar_capture, kind);

        match *upvar_capture {
            ty::UpvarCapture::ByValue => {
                // Upvar is already by-value, the strongest criteria.
            }
            ty::UpvarCapture::ByRef(ref mut upvar_borrow) => {
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
    }

    fn adjust_closure_kind(&self,
                           closure_id: ast::NodeId,
                           new_kind: ty::ClosureKind) {
        debug!("adjust_closure_kind(closure_id={}, new_kind={:?})",
               closure_id, new_kind);

        if !self.closures_with_inferred_kinds.contains(&closure_id) {
            return;
        }

        let closure_def_id = self.fcx.tcx().map.local_def_id(closure_id);
        let closure_kinds = &mut self.fcx.inh.tables.borrow_mut().closure_kinds;
        let existing_kind = *closure_kinds.get(&closure_def_id).unwrap();

        debug!("adjust_closure_kind: closure_id={}, existing_kind={:?}, new_kind={:?}",
               closure_id, existing_kind, new_kind);

        match (existing_kind, new_kind) {
            (ty::FnClosureKind, ty::FnClosureKind) |
            (ty::FnMutClosureKind, ty::FnClosureKind) |
            (ty::FnMutClosureKind, ty::FnMutClosureKind) |
            (ty::FnOnceClosureKind, _) => {
                // no change needed
            }

            (ty::FnClosureKind, ty::FnMutClosureKind) |
            (ty::FnClosureKind, ty::FnOnceClosureKind) |
            (ty::FnMutClosureKind, ty::FnOnceClosureKind) => {
                // new kind is stronger than the old kind
                closure_kinds.insert(closure_def_id, new_kind);
            }
        }
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for AdjustBorrowKind<'a, 'tcx> {
    fn visit_fn(&mut self,
                fn_kind: visit::FnKind<'v>,
                decl: &'v hir::FnDecl,
                body: &'v hir::Block,
                span: Span,
                id: ast::NodeId)
    {
        visit::walk_fn(self, fn_kind, decl, body, span);
        self.analyze_closure(id, span, decl, body);
    }

    // Skip all items; they aren't in the same context.
    fn visit_item(&mut self, _: &'v hir::Item) { }
}

impl<'a,'tcx> euv::Delegate<'tcx> for AdjustBorrowKind<'a,'tcx> {
    fn consume(&mut self,
               _consume_id: ast::NodeId,
               _consume_span: Span,
               cmt: mc::cmt<'tcx>,
               mode: euv::ConsumeMode)
    {
        debug!("consume(cmt={:?},mode={:?})", cmt, mode);
        self.adjust_upvar_borrow_kind_for_consume(cmt, mode);
    }

    fn matched_pat(&mut self,
                   _matched_pat: &hir::Pat,
                   _cmt: mc::cmt<'tcx>,
                   _mode: euv::MatchMode)
    {}

    fn consume_pat(&mut self,
                   _consume_pat: &hir::Pat,
                   cmt: mc::cmt<'tcx>,
                   mode: euv::ConsumeMode)
    {
        debug!("consume_pat(cmt={:?},mode={:?})", cmt, mode);
        self.adjust_upvar_borrow_kind_for_consume(cmt, mode);
    }

    fn borrow(&mut self,
              borrow_id: ast::NodeId,
              _borrow_span: Span,
              cmt: mc::cmt<'tcx>,
              _loan_region: ty::Region,
              bk: ty::BorrowKind,
              _loan_cause: euv::LoanCause)
    {
        debug!("borrow(borrow_id={}, cmt={:?}, bk={:?})",
               borrow_id, cmt, bk);

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
        debug!("mutate(assignee_cmt={:?})",
               assignee_cmt);

        self.adjust_upvar_borrow_kind_for_mut(assignee_cmt);
    }
}
