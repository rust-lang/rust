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
use middle::mem_categorization::Categorization;
use rustc::ty::{self, Ty};
use rustc::infer::UpvarRegion;
use syntax::ast;
use syntax_pos::Span;
use rustc::hir;
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};
use rustc::util::nodemap::NodeMap;

///////////////////////////////////////////////////////////////////////////
// PUBLIC ENTRY POINTS

impl<'a, 'gcx, 'tcx> FnCtxt<'a, 'gcx, 'tcx> {
    pub fn closure_analyze(&self, body: &'gcx hir::Body) {
        let mut seed = SeedBorrowKind::new(self);
        seed.visit_body(body);

        let mut adjust = AdjustBorrowKind::new(self, seed.temp_closure_kinds);
        adjust.visit_body(body);

        // it's our job to process these.
        assert!(self.deferred_call_resolutions.borrow().is_empty());
    }
}

///////////////////////////////////////////////////////////////////////////
// SEED BORROW KIND

struct SeedBorrowKind<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    fcx: &'a FnCtxt<'a, 'gcx, 'tcx>,
    temp_closure_kinds: NodeMap<ty::ClosureKind>,
}

impl<'a, 'gcx, 'tcx> Visitor<'gcx> for SeedBorrowKind<'a, 'gcx, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'gcx> {
        NestedVisitorMap::None
    }

    fn visit_expr(&mut self, expr: &'gcx hir::Expr) {
        match expr.node {
            hir::ExprClosure(cc, _, body_id, _) => {
                let body = self.fcx.tcx.map.body(body_id);
                self.visit_body(body);
                self.check_closure(expr, cc);
            }

            _ => { }
        }

        intravisit::walk_expr(self, expr);
    }
}

impl<'a, 'gcx, 'tcx> SeedBorrowKind<'a, 'gcx, 'tcx> {
    fn new(fcx: &'a FnCtxt<'a, 'gcx, 'tcx>) -> SeedBorrowKind<'a, 'gcx, 'tcx> {
        SeedBorrowKind { fcx: fcx, temp_closure_kinds: NodeMap() }
    }

    fn check_closure(&mut self,
                     expr: &hir::Expr,
                     capture_clause: hir::CaptureClause)
    {
        if !self.fcx.tables.borrow().closure_kinds.contains_key(&expr.id) {
            self.temp_closure_kinds.insert(expr.id, ty::ClosureKind::Fn);
            debug!("check_closure: adding closure {:?} as Fn", expr.id);
        }

        self.fcx.tcx.with_freevars(expr.id, |freevars| {
            for freevar in freevars {
                let def_id = freevar.def.def_id();
                let var_node_id = self.fcx.tcx.map.as_local_node_id(def_id).unwrap();
                let upvar_id = ty::UpvarId { var_id: var_node_id,
                                             closure_expr_id: expr.id };
                debug!("seed upvar_id {:?}", upvar_id);

                let capture_kind = match capture_clause {
                    hir::CaptureByValue => {
                        ty::UpvarCapture::ByValue
                    }
                    hir::CaptureByRef => {
                        let origin = UpvarRegion(upvar_id, expr.span);
                        let freevar_region = self.fcx.next_region_var(origin);
                        let upvar_borrow = ty::UpvarBorrow { kind: ty::ImmBorrow,
                                                             region: freevar_region };
                        ty::UpvarCapture::ByRef(upvar_borrow)
                    }
                };

                self.fcx.tables.borrow_mut().upvar_capture_map.insert(upvar_id, capture_kind);
            }
        });
    }
}

///////////////////////////////////////////////////////////////////////////
// ADJUST BORROW KIND

struct AdjustBorrowKind<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    fcx: &'a FnCtxt<'a, 'gcx, 'tcx>,
    temp_closure_kinds: NodeMap<ty::ClosureKind>,
}

impl<'a, 'gcx, 'tcx> AdjustBorrowKind<'a, 'gcx, 'tcx> {
    fn new(fcx: &'a FnCtxt<'a, 'gcx, 'tcx>,
           temp_closure_kinds: NodeMap<ty::ClosureKind>)
           -> AdjustBorrowKind<'a, 'gcx, 'tcx> {
        AdjustBorrowKind { fcx: fcx, temp_closure_kinds: temp_closure_kinds }
    }

    fn analyze_closure(&mut self,
                       id: ast::NodeId,
                       span: Span,
                       body: &hir::Body) {
        /*!
         * Analysis starting point.
         */

        debug!("analyze_closure(id={:?}, body.id={:?})", id, body.id());

        {
            let mut euv =
                euv::ExprUseVisitor::with_options(self,
                                                  self.fcx,
                                                  mc::MemCategorizationOptions {
                                                      during_closure_kind_inference: true
                                                  });
            euv.consume_body(body);
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
        let (def_id, closure_substs) = match self.fcx.node_ty(id).sty {
            ty::TyClosure(def_id, substs) => (def_id, substs),
            ref t => {
                span_bug!(
                    span,
                    "type of closure expr {:?} is not a closure {:?}",
                    id, t);
            }
        };

        // Equate the type variables with the actual types.
        let final_upvar_tys = self.final_upvar_tys(id);
        debug!("analyze_closure: id={:?} closure_substs={:?} final_upvar_tys={:?}",
               id, closure_substs, final_upvar_tys);
        for (upvar_ty, final_upvar_ty) in
            closure_substs.upvar_tys(def_id, self.fcx.tcx).zip(final_upvar_tys)
        {
            self.fcx.demand_eqtype(span, final_upvar_ty, upvar_ty);
        }

        // If we are also inferred the closure kind here, update the
        // main table and process any deferred resolutions.
        let closure_def_id = self.fcx.tcx.map.local_def_id(id);
        if let Some(&kind) = self.temp_closure_kinds.get(&id) {
            self.fcx.tables.borrow_mut().closure_kinds.insert(id, kind);
            debug!("closure_kind({:?}) = {:?}", closure_def_id, kind);

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
        let tcx = self.fcx.tcx;
        tcx.with_freevars(closure_id, |freevars| {
            freevars.iter().map(|freevar| {
                let def_id = freevar.def.def_id();
                let var_id = tcx.map.as_local_node_id(def_id).unwrap();
                let freevar_ty = self.fcx.node_ty(var_id);
                let upvar_id = ty::UpvarId {
                    var_id: var_id,
                    closure_expr_id: closure_id
                };
                let capture = self.fcx.upvar_capture(upvar_id).unwrap();

                debug!("var_id={:?} freevar_ty={:?} capture={:?}",
                       var_id, freevar_ty, capture);

                match capture {
                    ty::UpvarCapture::ByValue => freevar_ty,
                    ty::UpvarCapture::ByRef(borrow) =>
                        tcx.mk_ref(borrow.region,
                                    ty::TypeAndMut {
                                        ty: freevar_ty,
                                        mutbl: borrow.kind.to_mutbl_lossy(),
                                    }),
                }
            }).collect()
        })
    }

    fn adjust_upvar_borrow_kind_for_consume(&mut self,
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
            Categorization::Deref(.., mc::BorrowedPtr(..)) |
            Categorization::Deref(.., mc::Implicit(..)) => {
                match cmt.note {
                    mc::NoteUpvarRef(upvar_id) => {
                        debug!("adjust_upvar_borrow_kind_for_consume: \
                                setting upvar_id={:?} to by value",
                               upvar_id);

                        // to move out of an upvar, this must be a FnOnce closure
                        self.adjust_closure_kind(upvar_id.closure_expr_id,
                                                 ty::ClosureKind::FnOnce);

                        let upvar_capture_map =
                            &mut self.fcx.tables.borrow_mut().upvar_capture_map;
                        upvar_capture_map.insert(upvar_id, ty::UpvarCapture::ByValue);
                    }
                    mc::NoteClosureEnv(upvar_id) => {
                        // we get just a closureenv ref if this is a
                        // `move` closure, or if the upvar has already
                        // been inferred to by-value. In any case, we
                        // must still adjust the kind of the closure
                        // to be a FnOnce closure to permit moves out
                        // of the environment.
                        self.adjust_closure_kind(upvar_id.closure_expr_id,
                                                 ty::ClosureKind::FnOnce);
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
            Categorization::Deref(base, _, mc::Unique) |
            Categorization::Interior(base, _) |
            Categorization::Downcast(base, _) => {
                // Interior or owned data is mutable if base is
                // mutable, so iterate to the base.
                self.adjust_upvar_borrow_kind_for_mut(base);
            }

            Categorization::Deref(base, _, mc::BorrowedPtr(..)) |
            Categorization::Deref(base, _, mc::Implicit(..)) => {
                if !self.try_adjust_upvar_deref(&cmt.note, ty::MutBorrow) {
                    // assignment to deref of an `&mut`
                    // borrowed pointer implies that the
                    // pointer itself must be unique, but not
                    // necessarily *mutable*
                    self.adjust_upvar_borrow_kind_for_unique(base);
                }
            }

            Categorization::Deref(.., mc::UnsafePtr(..)) |
            Categorization::StaticItem |
            Categorization::Rvalue(_) |
            Categorization::Local(_) |
            Categorization::Upvar(..) => {
                return;
            }
        }
    }

    fn adjust_upvar_borrow_kind_for_unique(&mut self, cmt: mc::cmt<'tcx>) {
        debug!("adjust_upvar_borrow_kind_for_unique(cmt={:?})",
               cmt);

        match cmt.cat.clone() {
            Categorization::Deref(base, _, mc::Unique) |
            Categorization::Interior(base, _) |
            Categorization::Downcast(base, _) => {
                // Interior or owned data is unique if base is
                // unique.
                self.adjust_upvar_borrow_kind_for_unique(base);
            }

            Categorization::Deref(base, _, mc::BorrowedPtr(..)) |
            Categorization::Deref(base, _, mc::Implicit(..)) => {
                if !self.try_adjust_upvar_deref(&cmt.note, ty::UniqueImmBorrow) {
                    // for a borrowed pointer to be unique, its
                    // base must be unique
                    self.adjust_upvar_borrow_kind_for_unique(base);
                }
            }

            Categorization::Deref(.., mc::UnsafePtr(..)) |
            Categorization::StaticItem |
            Categorization::Rvalue(_) |
            Categorization::Local(_) |
            Categorization::Upvar(..) => {
            }
        }
    }

    fn try_adjust_upvar_deref(&mut self,
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
                    let upvar_capture_map = &mut self.fcx.tables.borrow_mut().upvar_capture_map;
                    let ub = upvar_capture_map.get_mut(&upvar_id).unwrap();
                    self.adjust_upvar_borrow_kind(upvar_id, ub, borrow_kind);
                }

                // also need to be in an FnMut closure since this is not an ImmBorrow
                self.adjust_closure_kind(upvar_id.closure_expr_id, ty::ClosureKind::FnMut);

                true
            }
            mc::NoteClosureEnv(upvar_id) => {
                // this kind of deref occurs in a `move` closure, or
                // for a by-value upvar; in either case, to mutate an
                // upvar, we need to be an FnMut closure
                self.adjust_closure_kind(upvar_id.closure_expr_id, ty::ClosureKind::FnMut);

                true
            }
            mc::NoteNone => {
                false
            }
        }
    }

    /// We infer the borrow_kind with which to borrow upvars in a stack closure.
    /// The borrow_kind basically follows a lattice of `imm < unique-imm < mut`,
    /// moving from left to right as needed (but never right to left).
    /// Here the argument `mutbl` is the borrow_kind that is required by
    /// some particular use.
    fn adjust_upvar_borrow_kind(&mut self,
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

    fn adjust_closure_kind(&mut self,
                           closure_id: ast::NodeId,
                           new_kind: ty::ClosureKind) {
        debug!("adjust_closure_kind(closure_id={}, new_kind={:?})",
               closure_id, new_kind);

        if let Some(&existing_kind) = self.temp_closure_kinds.get(&closure_id) {
            debug!("adjust_closure_kind: closure_id={}, existing_kind={:?}, new_kind={:?}",
                   closure_id, existing_kind, new_kind);

            match (existing_kind, new_kind) {
                (ty::ClosureKind::Fn, ty::ClosureKind::Fn) |
                (ty::ClosureKind::FnMut, ty::ClosureKind::Fn) |
                (ty::ClosureKind::FnMut, ty::ClosureKind::FnMut) |
                (ty::ClosureKind::FnOnce, _) => {
                    // no change needed
                }

                (ty::ClosureKind::Fn, ty::ClosureKind::FnMut) |
                (ty::ClosureKind::Fn, ty::ClosureKind::FnOnce) |
                (ty::ClosureKind::FnMut, ty::ClosureKind::FnOnce) => {
                    // new kind is stronger than the old kind
                    self.temp_closure_kinds.insert(closure_id, new_kind);
                }
            }
        }
    }
}

impl<'a, 'gcx, 'tcx> Visitor<'gcx> for AdjustBorrowKind<'a, 'gcx, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'gcx> {
        NestedVisitorMap::None
    }

    fn visit_fn(&mut self,
                fn_kind: intravisit::FnKind<'gcx>,
                decl: &'gcx hir::FnDecl,
                body: hir::BodyId,
                span: Span,
                id: ast::NodeId)
    {
        intravisit::walk_fn(self, fn_kind, decl, body, span, id);

        let body = self.fcx.tcx.map.body(body);
        self.visit_body(body);
        self.analyze_closure(id, span, body);
    }
}

impl<'a, 'gcx, 'tcx> euv::Delegate<'tcx> for AdjustBorrowKind<'a, 'gcx, 'tcx> {
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
              _loan_region: &'tcx ty::Region,
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
