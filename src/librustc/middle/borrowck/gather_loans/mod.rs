// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ----------------------------------------------------------------------
// Gathering loans
//
// The borrow check proceeds in two phases. In phase one, we gather the full
// set of loans that are required at any point.  These are sorted according to
// their associated scopes.  In phase two, checking loans, we will then make
// sure that all of these loans are honored.


use middle::borrowck::*;
use middle::borrowck::move_data::MoveData;
use mc = middle::mem_categorization;
use middle::pat_util;
use middle::ty::{ty_region};
use middle::ty;
use util::common::indenter;
use util::ppaux::{Repr};

use std::cell::RefCell;
use syntax::ast;
use syntax::ast_util::IdRange;
use syntax::codemap::Span;
use syntax::print::pprust;
use syntax::visit;
use syntax::visit::{Visitor, FnKind};
use syntax::ast::{Expr, FnDecl, Block, NodeId, Stmt, Pat, Local};

mod lifetime;
mod restrictions;
mod gather_moves;

/// Context used while gathering loans:
///
/// - `bccx`: the borrow check context
/// - `item_ub`: the id of the block for the enclosing fn/method item
/// - `root_ub`: the id of the outermost block for which we can root
///   an `@T`.  This is the id of the innermost enclosing
///   loop or function body.
///
/// The role of `root_ub` is to prevent us from having to accumulate
/// vectors of rooted items at runtime.  Consider this case:
///
///     fn foo(...) -> int {
///         let mut ptr: &int;
///         while some_cond {
///             let x: @int = ...;
///             ptr = &*x;
///         }
///         *ptr
///     }
///
/// If we are not careful here, we would infer the scope of the borrow `&*x`
/// to be the body of the function `foo()` as a whole.  We would then
/// have root each `@int` that is produced, which is an unbounded number.
/// No good.  Instead what will happen is that `root_ub` will be set to the
/// body of the while loop and we will refuse to root the pointer `&*x`
/// because it would have to be rooted for a region greater than `root_ub`.
struct GatherLoanCtxt<'a> {
    bccx: &'a BorrowckCtxt,
    id_range: IdRange,
    move_data: move_data::MoveData,
    all_loans: @RefCell<~[Loan]>,
    item_ub: ast::NodeId,
    repeating_ids: ~[ast::NodeId]
}

impl<'a> visit::Visitor<()> for GatherLoanCtxt<'a> {
    fn visit_expr(&mut self, ex: &Expr, _: ()) {
        gather_loans_in_expr(self, ex);
    }
    fn visit_block(&mut self, b: &Block, _: ()) {
        gather_loans_in_block(self, b);
    }
    fn visit_fn(&mut self, fk: &FnKind, fd: &FnDecl, b: &Block,
                s: Span, n: NodeId, _: ()) {
        gather_loans_in_fn(self, fk, fd, b, s, n);
    }
    fn visit_stmt(&mut self, s: &Stmt, _: ()) {
        visit::walk_stmt(self, s, ());
    }
    fn visit_pat(&mut self, p: &Pat, _: ()) {
        add_pat_to_id_range(self, p);
    }
    fn visit_local(&mut self, l: &Local, _: ()) {
        gather_loans_in_local(self, l);
    }

    // #7740: Do not visit items here, not even fn items nor methods
    // of impl items; the outer loop in borrowck/mod will visit them
    // for us in turn.  Thus override visit_item's walk with a no-op.
    fn visit_item(&mut self, _: &ast::Item, _: ()) { }
}

pub fn gather_loans(bccx: &BorrowckCtxt, decl: &ast::FnDecl, body: &ast::Block)
                    -> (IdRange, @RefCell<~[Loan]>, move_data::MoveData) {
    let mut glcx = GatherLoanCtxt {
        bccx: bccx,
        id_range: IdRange::max(),
        all_loans: @RefCell::new(~[]),
        item_ub: body.id,
        repeating_ids: ~[body.id],
        move_data: MoveData::new()
    };
    glcx.gather_fn_arg_patterns(decl, body);

    glcx.visit_block(body, ());
    return (glcx.id_range, glcx.all_loans, glcx.move_data);
}

fn add_pat_to_id_range(this: &mut GatherLoanCtxt,
                       p: &ast::Pat) {
    // NB: This visitor function just adds the pat ids into the id
    // range. We gather loans that occur in patterns using the
    // `gather_pat()` method below. Eventually these two should be
    // brought together.
    this.id_range.add(p.id);
    visit::walk_pat(this, p, ());
}

fn gather_loans_in_fn(this: &mut GatherLoanCtxt, fk: &FnKind,
                      decl: &ast::FnDecl, body: &ast::Block,
                      sp: Span, id: ast::NodeId) {
    match fk {
        &visit::FkItemFn(..) | &visit::FkMethod(..) => {
            fail!("cannot occur, due to visit_item override");
        }

        // Visit closures as part of the containing item.
        &visit::FkFnBlock(..) => {
            this.push_repeating_id(body.id);
            visit::walk_fn(this, fk, decl, body, sp, id, ());
            this.pop_repeating_id(body.id);
            this.gather_fn_arg_patterns(decl, body);
        }
    }
}

fn gather_loans_in_block(this: &mut GatherLoanCtxt,
                         blk: &ast::Block) {
    this.id_range.add(blk.id);
    visit::walk_block(this, blk, ());
}

fn gather_loans_in_local(this: &mut GatherLoanCtxt,
                         local: &ast::Local) {
    match local.init {
        None => {
            // Variable declarations without initializers are considered "moves":
            let tcx = this.bccx.tcx;
            pat_util::pat_bindings(tcx.def_map, local.pat, |_, id, span, _| {
                gather_moves::gather_decl(this.bccx,
                                          &this.move_data,
                                          id,
                                          span,
                                          id);
            })
        }
        Some(init) => {
            // Variable declarations with initializers are considered "assigns":
            let tcx = this.bccx.tcx;
            pat_util::pat_bindings(tcx.def_map, local.pat, |_, id, span, _| {
                gather_moves::gather_assignment(this.bccx,
                                                &this.move_data,
                                                id,
                                                span,
                                                @LpVar(id),
                                                id);
            });
            let init_cmt = this.bccx.cat_expr(init);
            this.gather_pat(init_cmt, local.pat, None);
        }
    }

    visit::walk_local(this, local, ());
}


fn gather_loans_in_expr(this: &mut GatherLoanCtxt,
                        ex: &ast::Expr) {
    let bccx = this.bccx;
    let tcx = bccx.tcx;

    debug!("gather_loans_in_expr(expr={:?}/{})",
           ex.id, pprust::expr_to_str(ex, tcx.sess.intr()));

    this.id_range.add(ex.id);

    {
        let r = ex.get_callee_id();
        for callee_id in r.iter() {
            this.id_range.add(*callee_id);
        }
    }

    // If this expression is borrowed, have to ensure it remains valid:
    {
        let adjustments = tcx.adjustments.borrow();
        let r = adjustments.get().find(&ex.id);
        for &adjustments in r.iter() {
            this.guarantee_adjustments(ex, *adjustments);
        }
    }

    // If this expression is a move, gather it:
    if this.bccx.is_move(ex.id) {
        let cmt = this.bccx.cat_expr(ex);
        gather_moves::gather_move_from_expr(
            this.bccx, &this.move_data, ex, cmt);
    }

    // Special checks for various kinds of expressions:
    let method_map = this.bccx.method_map.borrow();
    match ex.node {
      ast::ExprAddrOf(mutbl, base) => {
        let base_cmt = this.bccx.cat_expr(base);

        // make sure that the thing we are pointing out stays valid
        // for the lifetime `scope_r` of the resulting ptr:
        let expr_ty = ty::expr_ty(tcx, ex);
        if !ty::type_is_bot(expr_ty) {
            let scope_r = ty_region(tcx, ex.span, expr_ty);
            this.guarantee_valid(ex.id,
                                 ex.span,
                                 base_cmt,
                                 LoanMutability::from_ast_mutability(mutbl),
                                 scope_r);
        }
        visit::walk_expr(this, ex, ());
      }

      ast::ExprAssign(l, _) | ast::ExprAssignOp(_, _, l, _) => {
          let l_cmt = this.bccx.cat_expr(l);
          match opt_loan_path(l_cmt) {
              Some(l_lp) => {
                  gather_moves::gather_assignment(this.bccx, &this.move_data,
                                                  ex.id, ex.span,
                                                  l_lp, l.id);
              }
              None => {
                  // This can occur with e.g. `*foo() = 5`.  In such
                  // cases, there is no need to check for conflicts
                  // with moves etc, just ignore.
              }
          }
          visit::walk_expr(this, ex, ());
      }

      ast::ExprMatch(ex_v, ref arms) => {
        let cmt = this.bccx.cat_expr(ex_v);
        for arm in arms.iter() {
            for pat in arm.pats.iter() {
                this.gather_pat(cmt, *pat, Some((arm.body.id, ex.id)));
            }
        }
        visit::walk_expr(this, ex, ());
      }

      ast::ExprIndex(_, _, arg) |
      ast::ExprBinary(_, _, _, arg)
      if method_map.get().contains_key(&ex.id) => {
          // Arguments in method calls are always passed by ref.
          //
          // Currently these do not use adjustments, so we have to
          // hardcode this check here (note that the receiver DOES use
          // adjustments).
          let scope_r = ty::ReScope(ex.id);
          let arg_cmt = this.bccx.cat_expr(arg);
          this.guarantee_valid(arg.id,
                               arg.span,
                               arg_cmt,
                               ImmutableMutability,
                               scope_r);
          visit::walk_expr(this, ex, ());
      }

      // see explanation attached to the `root_ub` field:
      ast::ExprWhile(cond, body) => {
          // during the condition, can only root for the condition
          this.push_repeating_id(cond.id);
          this.visit_expr(cond, ());
          this.pop_repeating_id(cond.id);

          // during body, can only root for the body
          this.push_repeating_id(body.id);
          this.visit_block(body, ());
          this.pop_repeating_id(body.id);
      }

      // see explanation attached to the `root_ub` field:
      ast::ExprLoop(body, _) => {
          this.push_repeating_id(body.id);
          visit::walk_expr(this, ex, ());
          this.pop_repeating_id(body.id);
      }

      ast::ExprFnBlock(..) | ast::ExprProc(..) => {
          gather_moves::gather_captures(this.bccx, &this.move_data, ex);
          visit::walk_expr(this, ex, ());
      }

      ast::ExprInlineAsm(ref ia) => {
          for &(_, out) in ia.outputs.iter() {
              let out_cmt = this.bccx.cat_expr(out);
              match opt_loan_path(out_cmt) {
                  Some(out_lp) => {
                      gather_moves::gather_assignment(this.bccx, &this.move_data,
                                                      ex.id, ex.span,
                                                      out_lp, out.id);
                  }
                  None => {
                      // See the comment for ExprAssign.
                  }
              }
          }
          visit::walk_expr(this, ex, ());
      }

      _ => {
          visit::walk_expr(this, ex, ());
      }
    }
}

impl<'a> GatherLoanCtxt<'a> {
    pub fn tcx(&self) -> ty::ctxt { self.bccx.tcx }

    pub fn push_repeating_id(&mut self, id: ast::NodeId) {
        self.repeating_ids.push(id);
    }

    pub fn pop_repeating_id(&mut self, id: ast::NodeId) {
        let popped = self.repeating_ids.pop().unwrap();
        assert_eq!(id, popped);
    }

    pub fn guarantee_adjustments(&mut self,
                                 expr: &ast::Expr,
                                 adjustment: &ty::AutoAdjustment) {
        debug!("guarantee_adjustments(expr={}, adjustment={:?})",
               expr.repr(self.tcx()), adjustment);
        let _i = indenter();

        match *adjustment {
            ty::AutoAddEnv(..) => {
                debug!("autoaddenv -- no autoref");
                return;
            }

            ty::AutoDerefRef(
                ty::AutoDerefRef {
                    autoref: None, .. }) => {
                debug!("no autoref");
                return;
            }

            ty::AutoDerefRef(
                ty::AutoDerefRef {
                    autoref: Some(ref autoref),
                    autoderefs: autoderefs}) => {
                let mcx = &mc::mem_categorization_ctxt {
                    tcx: self.tcx(),
                    method_map: self.bccx.method_map};
                let cmt = mcx.cat_expr_autoderefd(expr, autoderefs);
                debug!("after autoderef, cmt={}", cmt.repr(self.tcx()));

                match *autoref {
                    ty::AutoPtr(r, m) => {
                        let loan_mutability =
                            LoanMutability::from_ast_mutability(m);
                        self.guarantee_valid(expr.id,
                                             expr.span,
                                             cmt,
                                             loan_mutability,
                                             r)
                    }
                    ty::AutoBorrowVec(r, m) | ty::AutoBorrowVecRef(r, m) => {
                        let cmt_index = mcx.cat_index(expr, cmt, autoderefs+1);
                        let loan_mutability =
                            LoanMutability::from_ast_mutability(m);
                        self.guarantee_valid(expr.id,
                                             expr.span,
                                             cmt_index,
                                             loan_mutability,
                                             r)
                    }
                    ty::AutoBorrowFn(r) => {
                        let cmt_deref = mcx.cat_deref_fn_or_obj(expr, cmt, 0);
                        self.guarantee_valid(expr.id,
                                             expr.span,
                                             cmt_deref,
                                             ImmutableMutability,
                                             r)
                    }
                    ty::AutoBorrowObj(r, m) => {
                        let cmt_deref = mcx.cat_deref_fn_or_obj(expr, cmt, 0);
                        let loan_mutability =
                            LoanMutability::from_ast_mutability(m);
                        self.guarantee_valid(expr.id,
                                             expr.span,
                                             cmt_deref,
                                             loan_mutability,
                                             r)
                    }
                    ty::AutoUnsafe(_) => {}
                }
            }

            ty::AutoObject(..) => {
                // FIXME: Handle ~Trait to &Trait casts here?
            }
        }
    }

    // Guarantees that addr_of(cmt) will be valid for the duration of
    // `static_scope_r`, or reports an error.  This may entail taking
    // out loans, which will be added to the `req_loan_map`.  This can
    // also entail "rooting" GC'd pointers, which means ensuring
    // dynamically that they are not freed.
    pub fn guarantee_valid(&mut self,
                           borrow_id: ast::NodeId,
                           borrow_span: Span,
                           cmt: mc::cmt,
                           req_mutbl: LoanMutability,
                           loan_region: ty::Region) {
        debug!("guarantee_valid(borrow_id={:?}, cmt={}, \
                req_mutbl={:?}, loan_region={:?})",
               borrow_id,
               cmt.repr(self.tcx()),
               req_mutbl,
               loan_region);

        // a loan for the empty region can never be dereferenced, so
        // it is always safe
        if loan_region == ty::ReEmpty {
            return;
        }

        let root_ub = { *self.repeating_ids.last().unwrap() }; // FIXME(#5074)

        // Check that the lifetime of the borrow does not exceed
        // the lifetime of the data being borrowed.
        if lifetime::guarantee_lifetime(self.bccx, self.item_ub, root_ub,
                                        borrow_span, cmt, loan_region,
                                        req_mutbl).is_err() {
            return; // reported an error, no sense in reporting more.
        }

        // Check that we don't allow mutable borrows of non-mutable data.
        if check_mutability(self.bccx, borrow_span, cmt, req_mutbl).is_err() {
            return; // reported an error, no sense in reporting more.
        }

        // Check that we don't allow mutable borrows of aliasable data.
        if check_aliasability(self.bccx, borrow_span, cmt, req_mutbl).is_err() {
            return; // reported an error, no sense in reporting more.
        }

        // Compute the restrictions that are required to enforce the
        // loan is safe.
        let restr = restrictions::compute_restrictions(
            self.bccx, borrow_span,
            cmt, loan_region, self.restriction_set(req_mutbl));

        // Create the loan record (if needed).
        let loan = match restr {
            restrictions::Safe => {
                // No restrictions---no loan record necessary
                return;
            }

            restrictions::SafeIf(loan_path, restrictions) => {
                let loan_scope = match loan_region {
                    ty::ReScope(id) => id,
                    ty::ReFree(ref fr) => fr.scope_id,

                    ty::ReStatic => {
                        // If we get here, an error must have been
                        // reported in
                        // `lifetime::guarantee_lifetime()`, because
                        // the only legal ways to have a borrow with a
                        // static lifetime should not require
                        // restrictions. To avoid reporting derived
                        // errors, we just return here without adding
                        // any loans.
                        return;
                    }

                    ty::ReEmpty |
                    ty::ReLateBound(..) |
                    ty::ReEarlyBound(..) |
                    ty::ReInfer(..) => {
                        self.tcx().sess.span_bug(
                            cmt.span,
                            format!("invalid borrow lifetime: {:?}", loan_region));
                    }
                };
                debug!("loan_scope = {:?}", loan_scope);

                let gen_scope = self.compute_gen_scope(borrow_id, loan_scope);
                debug!("gen_scope = {:?}", gen_scope);

                let kill_scope = self.compute_kill_scope(loan_scope, loan_path);
                debug!("kill_scope = {:?}", kill_scope);

                if req_mutbl == MutableMutability {
                    self.mark_loan_path_as_mutated(loan_path);
                }

                let all_loans = self.all_loans.borrow();
                Loan {
                    index: all_loans.get().len(),
                    loan_path: loan_path,
                    cmt: cmt,
                    mutbl: req_mutbl,
                    gen_scope: gen_scope,
                    kill_scope: kill_scope,
                    span: borrow_span,
                    restrictions: restrictions
                }
            }
        };

        debug!("guarantee_valid(borrow_id={:?}), loan={}",
               borrow_id, loan.repr(self.tcx()));

        // let loan_path = loan.loan_path;
        // let loan_gen_scope = loan.gen_scope;
        // let loan_kill_scope = loan.kill_scope;
        {
            let mut all_loans = self.all_loans.borrow_mut();
            all_loans.get().push(loan);
        }

        // if loan_gen_scope != borrow_id {
            // FIXME(#6268) Nested method calls
            //
            // Typically, the scope of the loan includes the point at
            // which the loan is originated. This
            // This is a subtle case. See the test case
            // <compile-fail/borrowck-bad-nested-calls-free.rs>
            // to see what we are guarding against.

            //let restr = restrictions::compute_restrictions(
            //    self.bccx, borrow_span, cmt, RESTR_EMPTY);
            //let loan = {
            //    let all_loans = &mut *self.all_loans; // FIXME(#5074)
            //    Loan {
            //        index: all_loans.len(),
            //        loan_path: loan_path,
            //        cmt: cmt,
            //        mutbl: ConstMutability,
            //        gen_scope: borrow_id,
            //        kill_scope: kill_scope,
            //        span: borrow_span,
            //        restrictions: restrictions
            //    }
        // }

        fn check_mutability(bccx: &BorrowckCtxt,
                            borrow_span: Span,
                            cmt: mc::cmt,
                            req_mutbl: LoanMutability) -> Result<(),()> {
            //! Implements the M-* rules in doc.rs.

            match req_mutbl {
                ImmutableMutability => {
                    // both imm and mut data can be lent as imm;
                    // for mutable data, this is a freeze
                    Ok(())
                }

                MutableMutability => {
                    // Only mutable data can be lent as mutable.
                    if !cmt.mutbl.is_mutable() {
                        Err(bccx.report(BckError {span: borrow_span,
                                                  cmt: cmt,
                                                  code: err_mutbl(req_mutbl)}))
                    } else {
                        Ok(())
                    }
                }
            }
        }

        fn check_aliasability(bccx: &BorrowckCtxt,
                              borrow_span: Span,
                              cmt: mc::cmt,
                              req_mutbl: LoanMutability) -> Result<(),()> {
            //! Implements the A-* rules in doc.rs.

            match req_mutbl {
                ImmutableMutability => {
                    // both imm and mut data can be lent as imm;
                    // for mutable data, this is a freeze
                    Ok(())
                }

                MutableMutability => {
                    // Check for those cases where we cannot control
                    // the aliasing and make sure that we are not
                    // being asked to.
                    match cmt.freely_aliasable() {
                        None => {
                            Ok(())
                        }
                        Some(mc::AliasableStaticMut) => {
                            // This is nasty, but we ignore the
                            // aliasing rules if the data is based in
                            // a `static mut`, since those are always
                            // unsafe. At your own peril and all that.
                            Ok(())
                        }
                        Some(cause) => {
                            bccx.report_aliasability_violation(
                                borrow_span,
                                BorrowViolation,
                                cause);
                            Err(())
                        }
                    }
                }
            }
        }
    }

    pub fn restriction_set(&self, req_mutbl: LoanMutability)
                           -> RestrictionSet {
        match req_mutbl {
            ImmutableMutability => RESTR_MUTATE | RESTR_CLAIM,
            MutableMutability => RESTR_MUTATE | RESTR_CLAIM | RESTR_FREEZE,
        }
    }

    pub fn mark_loan_path_as_mutated(&self, loan_path: @LoanPath) {
        //! For mutable loans of content whose mutability derives
        //! from a local variable, mark the mutability decl as necessary.

        match *loan_path {
            LpVar(local_id) => {
                let mut used_mut_nodes = self.tcx()
                                             .used_mut_nodes
                                             .borrow_mut();
                used_mut_nodes.get().insert(local_id);
            }
            LpExtend(base, mc::McInherited, _) => {
                self.mark_loan_path_as_mutated(base);
            }
            LpExtend(_, mc::McDeclared, _) |
            LpExtend(_, mc::McImmutable, _) => {
                // Nothing to do.
            }
        }
    }

    pub fn compute_gen_scope(&self,
                             borrow_id: ast::NodeId,
                             loan_scope: ast::NodeId)
                             -> ast::NodeId {
        //! Determine when to introduce the loan. Typically the loan
        //! is introduced at the point of the borrow, but in some cases,
        //! notably method arguments, the loan may be introduced only
        //! later, once it comes into scope.

        if self.bccx.tcx.region_maps.is_subscope_of(borrow_id, loan_scope) {
            borrow_id
        } else {
            loan_scope
        }
    }

    pub fn compute_kill_scope(&self, loan_scope: ast::NodeId, lp: @LoanPath)
                              -> ast::NodeId {
        //! Determine when the loan restrictions go out of scope.
        //! This is either when the lifetime expires or when the
        //! local variable which roots the loan-path goes out of scope,
        //! whichever happens faster.
        //!
        //! It may seem surprising that we might have a loan region
        //! larger than the variable which roots the loan-path; this can
        //! come about when variables of `&mut` type are re-borrowed,
        //! as in this example:
        //!
        //!     fn counter<'a>(v: &'a mut Foo) -> &'a mut uint {
        //!         &mut v.counter
        //!     }
        //!
        //! In this case, the reference (`'a`) outlives the
        //! variable `v` that hosts it. Note that this doesn't come up
        //! with immutable `&` pointers, because borrows of such pointers
        //! do not require restrictions and hence do not cause a loan.

        let rm = &self.bccx.tcx.region_maps;
        let lexical_scope = rm.var_scope(lp.node_id());
        if rm.is_subscope_of(lexical_scope, loan_scope) {
            lexical_scope
        } else {
            assert!(self.bccx.tcx.region_maps.is_subscope_of(loan_scope, lexical_scope));
            loan_scope
        }
    }

    fn gather_fn_arg_patterns(&mut self,
                              decl: &ast::FnDecl,
                              body: &ast::Block) {
        /*!
         * Walks the patterns for fn arguments, checking that they
         * do not attempt illegal moves or create refs that outlive
         * the arguments themselves. Just a shallow wrapper around
         * `gather_pat()`.
         */

        let mc_ctxt = self.bccx.mc_ctxt();
        for arg in decl.inputs.iter() {
            let arg_ty = ty::node_id_to_type(self.tcx(), arg.pat.id);

            let arg_cmt = mc_ctxt.cat_rvalue(
                arg.id,
                arg.pat.span,
                ty::ReScope(body.id), // Args live only as long as the fn body.
                arg_ty);

            self.gather_pat(arg_cmt, arg.pat, None);
        }
    }

    fn gather_pat(&mut self,
                  discr_cmt: mc::cmt,
                  root_pat: &ast::Pat,
                  arm_match_ids: Option<(ast::NodeId, ast::NodeId)>) {
        /*!
         * Walks patterns, examining the bindings to determine if they
         * cause borrows (`ref` bindings, vector patterns) or
         * moves (non-`ref` bindings with linear type).
         */

        self.bccx.cat_pattern(discr_cmt, root_pat, |cmt, pat| {
            match pat.node {
              ast::PatIdent(bm, _, _) if self.pat_is_binding(pat) => {
                match bm {
                  ast::BindByRef(mutbl) => {
                    // ref x or ref x @ p --- creates a ptr which must
                    // remain valid for the scope of the match

                    // find the region of the resulting pointer (note that
                    // the type of such a pattern will *always* be a
                    // region pointer)
                    let scope_r =
                        ty_region(self.tcx(), pat.span,
                                  ty::node_id_to_type(self.tcx(), pat.id));

                    // if the scope of the region ptr turns out to be
                    // specific to this arm, wrap the categorization
                    // with a cat_discr() node.  There is a detailed
                    // discussion of the function of this node in
                    // `lifetime.rs`:
                    let cmt_discr = match arm_match_ids {
                        None => cmt,
                        Some((arm_id, match_id)) => {
                            let arm_scope = ty::ReScope(arm_id);
                            if self.bccx.is_subregion_of(scope_r, arm_scope) {
                                self.bccx.cat_discr(cmt, match_id)
                            } else {
                                cmt
                            }
                        }
                    };
                    let loan_mutability =
                        LoanMutability::from_ast_mutability(mutbl);
                    self.guarantee_valid(pat.id,
                                         pat.span,
                                         cmt_discr,
                                         loan_mutability,
                                         scope_r);
                  }
                  ast::BindByValue(_) => {
                      // No borrows here, but there may be moves
                      if self.bccx.is_move(pat.id) {
                          gather_moves::gather_move_from_pat(
                              self.bccx, &self.move_data, pat, cmt);
                      }
                  }
                }
              }

              ast::PatVec(_, Some(slice_pat), _) => {
                  // The `slice_pat` here creates a slice into the
                  // original vector.  This is effectively a borrow of
                  // the elements of the vector being matched.

                  let slice_ty = ty::node_id_to_type(self.tcx(),
                                                     slice_pat.id);
                  let (slice_mutbl, slice_r) =
                      self.vec_slice_info(slice_pat, slice_ty);
                  let mcx = self.bccx.mc_ctxt();
                  let cmt_index = mcx.cat_index(slice_pat, cmt, 0);
                  let slice_loan_mutability =
                    LoanMutability::from_ast_mutability(slice_mutbl);

                  // Note: We declare here that the borrow occurs upon
                  // entering the `[...]` pattern. This implies that
                  // something like `[a, ..b]` where `a` is a move is
                  // illegal, because the borrow is already in effect.
                  // In fact such a move would be safe-ish, but it
                  // effectively *requires* that we use the nulling
                  // out semantics to indicate when a value has been
                  // moved, which we are trying to move away from.
                  // Otherwise, how can we indicate that the first
                  // element in the vector has been moved?
                  // Eventually, we could perhaps modify this rule to
                  // permit `[..a, b]` where `b` is a move, because in
                  // that case we can adjust the length of the
                  // original vec accordingly, but we'd have to make
                  // trans do the right thing, and it would only work
                  // for `~` vectors. It seems simpler to just require
                  // that people call `vec.pop()` or `vec.unshift()`.
                  self.guarantee_valid(pat.id,
                                       pat.span,
                                       cmt_index,
                                       slice_loan_mutability,
                                       slice_r);
              }

              _ => {}
            }
        })
    }

    pub fn vec_slice_info(&self, pat: &ast::Pat, slice_ty: ty::t)
                          -> (ast::Mutability, ty::Region) {
        /*!
         *
         * In a pattern like [a, b, ..c], normally `c` has slice type,
         * but if you have [a, b, ..ref c], then the type of `ref c`
         * will be `&&[]`, so to extract the slice details we have
         * to recurse through rptrs.
         */

        match ty::get(slice_ty).sty {
            ty::ty_vec(slice_mt, ty::vstore_slice(slice_r)) => {
                (slice_mt.mutbl, slice_r)
            }

            ty::ty_rptr(_, ref mt) => {
                self.vec_slice_info(pat, mt.ty)
            }

            _ => {
                self.tcx().sess.span_bug(
                    pat.span,
                    format!("type of slice pattern is not a slice"));
            }
        }
    }

    pub fn pat_is_binding(&self, pat: &ast::Pat) -> bool {
        pat_util::pat_is_binding(self.bccx.tcx.def_map, pat)
    }
}
