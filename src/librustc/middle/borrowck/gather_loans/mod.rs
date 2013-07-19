// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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

use syntax::ast::{m_const, m_imm, m_mutbl};
use syntax::ast;
use syntax::ast_util::id_range;
use syntax::codemap::span;
use syntax::print::pprust;
use syntax::visit;

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
struct GatherLoanCtxt {
    bccx: @BorrowckCtxt,
    id_range: id_range,
    move_data: @mut move_data::MoveData,
    all_loans: @mut ~[Loan],
    item_ub: ast::node_id,
    repeating_ids: ~[ast::node_id]
}

pub fn gather_loans(bccx: @BorrowckCtxt,
                    decl: &ast::fn_decl,
                    body: &ast::blk)
                    -> (id_range, @mut ~[Loan], @mut move_data::MoveData) {
    let glcx = @mut GatherLoanCtxt {
        bccx: bccx,
        id_range: id_range::max(),
        all_loans: @mut ~[],
        item_ub: body.id,
        repeating_ids: ~[body.id],
        move_data: @mut MoveData::new()
    };
    glcx.gather_fn_arg_patterns(decl, body);
    let v = visit::mk_vt(@visit::Visitor {visit_expr: gather_loans_in_expr,
                                          visit_block: gather_loans_in_block,
                                          visit_fn: gather_loans_in_fn,
                                          visit_stmt: add_stmt_to_map,
                                          visit_pat: add_pat_to_id_range,
                                          visit_local: gather_loans_in_local,
                                          .. *visit::default_visitor()});
    (v.visit_block)(body, (glcx, v));
    return (glcx.id_range, glcx.all_loans, glcx.move_data);
}

fn add_pat_to_id_range(p: @ast::pat,
                       (this, v): (@mut GatherLoanCtxt,
                                   visit::vt<@mut GatherLoanCtxt>)) {
    // NB: This visitor function just adds the pat ids into the id
    // range. We gather loans that occur in patterns using the
    // `gather_pat()` method below. Eventually these two should be
    // brought together.
    this.id_range.add(p.id);
    visit::visit_pat(p, (this, v));
}

fn gather_loans_in_fn(fk: &visit::fn_kind,
                      decl: &ast::fn_decl,
                      body: &ast::blk,
                      sp: span,
                      id: ast::node_id,
                      (this, v): (@mut GatherLoanCtxt,
                                  visit::vt<@mut GatherLoanCtxt>)) {
    match fk {
        // Do not visit items here, the outer loop in borrowck/mod
        // will visit them for us in turn.
        &visit::fk_item_fn(*) | &visit::fk_method(*) => {
            return;
        }

        // Visit closures as part of the containing item.
        &visit::fk_anon(*) | &visit::fk_fn_block(*) => {
            this.push_repeating_id(body.id);
            visit::visit_fn(fk, decl, body, sp, id, (this, v));
            this.pop_repeating_id(body.id);
            this.gather_fn_arg_patterns(decl, body);
        }
    }
}

fn gather_loans_in_block(blk: &ast::blk,
                         (this, vt): (@mut GatherLoanCtxt,
                                      visit::vt<@mut GatherLoanCtxt>)) {
    this.id_range.add(blk.id);
    visit::visit_block(blk, (this, vt));
}

fn gather_loans_in_local(local: @ast::local,
                         (this, vt): (@mut GatherLoanCtxt,
                                      visit::vt<@mut GatherLoanCtxt>)) {
    match local.node.init {
        None => {
            // Variable declarations without initializers are considered "moves":
            let tcx = this.bccx.tcx;
            do pat_util::pat_bindings(tcx.def_map, local.node.pat)
                |_, id, span, _| {
                gather_moves::gather_decl(this.bccx,
                                          this.move_data,
                                          id,
                                          span,
                                          id);
            }
        }
        Some(init) => {
            // Variable declarations with initializers are considered "assigns":
            let tcx = this.bccx.tcx;
            do pat_util::pat_bindings(tcx.def_map, local.node.pat)
                |_, id, span, _| {
                gather_moves::gather_assignment(this.bccx,
                                                this.move_data,
                                                id,
                                                span,
                                                @LpVar(id),
                                                id);
            }
            let init_cmt = this.bccx.cat_expr(init);
            this.gather_pat(init_cmt, local.node.pat, None);
        }
    }

    visit::visit_local(local, (this, vt));
}

fn gather_loans_in_expr(ex: @ast::expr,
                        (this, vt): (@mut GatherLoanCtxt,
                                     visit::vt<@mut GatherLoanCtxt>)) {
    let bccx = this.bccx;
    let tcx = bccx.tcx;

    debug!("gather_loans_in_expr(expr=%?/%s)",
           ex.id, pprust::expr_to_str(ex, tcx.sess.intr()));

    this.id_range.add(ex.id);

    {
        let r = ex.get_callee_id();
        for r.iter().advance |callee_id| {
            this.id_range.add(*callee_id);
        }
    }

    // If this expression is borrowed, have to ensure it remains valid:
    {
        let r = tcx.adjustments.find(&ex.id);
        for r.iter().advance |&adjustments| {
            this.guarantee_adjustments(ex, *adjustments);
        }
    }

    // If this expression is a move, gather it:
    if this.bccx.is_move(ex.id) {
        let cmt = this.bccx.cat_expr(ex);
        gather_moves::gather_move_from_expr(
            this.bccx, this.move_data, ex, cmt);
    }

    // Special checks for various kinds of expressions:
    match ex.node {
      ast::expr_addr_of(mutbl, base) => {
        let base_cmt = this.bccx.cat_expr(base);

        // make sure that the thing we are pointing out stays valid
        // for the lifetime `scope_r` of the resulting ptr:
        let scope_r = ty_region(tcx, ex.span, ty::expr_ty(tcx, ex));
        this.guarantee_valid(ex.id, ex.span, base_cmt, mutbl, scope_r);
        visit::visit_expr(ex, (this, vt));
      }

      ast::expr_assign(l, _) | ast::expr_assign_op(_, _, l, _) => {
          let l_cmt = this.bccx.cat_expr(l);
          match opt_loan_path(l_cmt) {
              Some(l_lp) => {
                  gather_moves::gather_assignment(this.bccx, this.move_data,
                                                  ex.id, ex.span,
                                                  l_lp, l.id);
              }
              None => {
                  // This can occur with e.g. `*foo() = 5`.  In such
                  // cases, there is no need to check for conflicts
                  // with moves etc, just ignore.
              }
          }
          visit::visit_expr(ex, (this, vt));
      }

      ast::expr_match(ex_v, ref arms) => {
        let cmt = this.bccx.cat_expr(ex_v);
        for arms.iter().advance |arm| {
            for arm.pats.iter().advance |pat| {
                this.gather_pat(cmt, *pat, Some((arm.body.id, ex.id)));
            }
        }
        visit::visit_expr(ex, (this, vt));
      }

      ast::expr_index(_, _, arg) |
      ast::expr_binary(_, _, _, arg)
      if this.bccx.method_map.contains_key(&ex.id) => {
          // Arguments in method calls are always passed by ref.
          //
          // Currently these do not use adjustments, so we have to
          // hardcode this check here (note that the receiver DOES use
          // adjustments).
          let scope_r = ty::re_scope(ex.id);
          let arg_cmt = this.bccx.cat_expr(arg);
          this.guarantee_valid(arg.id, arg.span, arg_cmt, m_imm, scope_r);
          visit::visit_expr(ex, (this, vt));
      }

      // see explanation attached to the `root_ub` field:
      ast::expr_while(cond, ref body) => {
          // during the condition, can only root for the condition
          this.push_repeating_id(cond.id);
          (vt.visit_expr)(cond, (this, vt));
          this.pop_repeating_id(cond.id);

          // during body, can only root for the body
          this.push_repeating_id(body.id);
          (vt.visit_block)(body, (this, vt));
          this.pop_repeating_id(body.id);
      }

      // see explanation attached to the `root_ub` field:
      ast::expr_loop(ref body, _) => {
          this.push_repeating_id(body.id);
          visit::visit_expr(ex, (this, vt));
          this.pop_repeating_id(body.id);
      }

      ast::expr_fn_block(*) => {
          gather_moves::gather_captures(this.bccx, this.move_data, ex);
          visit::visit_expr(ex, (this, vt));
      }

      _ => {
        visit::visit_expr(ex, (this, vt));
      }
    }
}

impl GatherLoanCtxt {
    pub fn tcx(&self) -> ty::ctxt { self.bccx.tcx }

    pub fn push_repeating_id(&mut self, id: ast::node_id) {
        self.repeating_ids.push(id);
    }

    pub fn pop_repeating_id(&mut self, id: ast::node_id) {
        let popped = self.repeating_ids.pop();
        assert_eq!(id, popped);
    }

    pub fn guarantee_adjustments(&mut self,
                                 expr: @ast::expr,
                                 adjustment: &ty::AutoAdjustment) {
        debug!("guarantee_adjustments(expr=%s, adjustment=%?)",
               expr.repr(self.tcx()), adjustment);
        let _i = indenter();

        match *adjustment {
            ty::AutoAddEnv(*) => {
                debug!("autoaddenv -- no autoref");
                return;
            }

            ty::AutoDerefRef(
                ty::AutoDerefRef {
                    autoref: None, _ }) => {
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
                debug!("after autoderef, cmt=%s", cmt.repr(self.tcx()));

                match *autoref {
                    ty::AutoPtr(r, m) => {
                        self.guarantee_valid(expr.id,
                                             expr.span,
                                             cmt,
                                             m,
                                             r)
                    }
                    ty::AutoBorrowVec(r, m) | ty::AutoBorrowVecRef(r, m) => {
                        let cmt_index = mcx.cat_index(expr, cmt, autoderefs+1);
                        self.guarantee_valid(expr.id,
                                             expr.span,
                                             cmt_index,
                                             m,
                                             r)
                    }
                    ty::AutoBorrowFn(r) => {
                        let cmt_deref = mcx.cat_deref_fn(expr, cmt, 0);
                        self.guarantee_valid(expr.id,
                                             expr.span,
                                             cmt_deref,
                                             m_imm,
                                             r)
                    }
                    ty::AutoUnsafe(_) => {}
                }
            }
        }
    }

    // Guarantees that addr_of(cmt) will be valid for the duration of
    // `static_scope_r`, or reports an error.  This may entail taking
    // out loans, which will be added to the `req_loan_map`.  This can
    // also entail "rooting" GC'd pointers, which means ensuring
    // dynamically that they are not freed.
    pub fn guarantee_valid(&mut self,
                           borrow_id: ast::node_id,
                           borrow_span: span,
                           cmt: mc::cmt,
                           req_mutbl: ast::mutability,
                           loan_region: ty::Region) {
        debug!("guarantee_valid(borrow_id=%?, cmt=%s, \
                req_mutbl=%?, loan_region=%?)",
               borrow_id,
               cmt.repr(self.tcx()),
               req_mutbl,
               loan_region);

        // a loan for the empty region can never be dereferenced, so
        // it is always safe
        if loan_region == ty::re_empty {
            return;
        }

        let root_ub = { *self.repeating_ids.last() }; // FIXME(#5074)

        // Check that the lifetime of the borrow does not exceed
        // the lifetime of the data being borrowed.
        lifetime::guarantee_lifetime(self.bccx, self.item_ub, root_ub,
                                     borrow_span, cmt, loan_region, req_mutbl);

        // Check that we don't allow mutable borrows of non-mutable data.
        check_mutability(self.bccx, borrow_span, cmt, req_mutbl);

        // Compute the restrictions that are required to enforce the
        // loan is safe.
        let restr = restrictions::compute_restrictions(
            self.bccx, borrow_span,
            cmt, self.restriction_set(req_mutbl));

        // Create the loan record (if needed).
        let loan = match restr {
            restrictions::Safe => {
                // No restrictions---no loan record necessary
                return;
            }

            restrictions::SafeIf(loan_path, restrictions) => {
                let loan_scope = match loan_region {
                    ty::re_scope(id) => id,
                    ty::re_free(ref fr) => fr.scope_id,

                    ty::re_static => {
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

                    ty::re_empty |
                    ty::re_bound(*) |
                    ty::re_infer(*) => {
                        self.tcx().sess.span_bug(
                            cmt.span,
                            fmt!("Invalid borrow lifetime: %?", loan_region));
                    }
                };
                debug!("loan_scope = %?", loan_scope);

                let gen_scope = self.compute_gen_scope(borrow_id, loan_scope);
                debug!("gen_scope = %?", gen_scope);

                let kill_scope = self.compute_kill_scope(loan_scope, loan_path);
                debug!("kill_scope = %?", kill_scope);

                if req_mutbl == m_mutbl {
                    self.mark_loan_path_as_mutated(loan_path);
                }

                let all_loans = &mut *self.all_loans; // FIXME(#5074)
                Loan {
                    index: all_loans.len(),
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

        debug!("guarantee_valid(borrow_id=%?), loan=%s",
               borrow_id, loan.repr(self.tcx()));

        // let loan_path = loan.loan_path;
        // let loan_gen_scope = loan.gen_scope;
        // let loan_kill_scope = loan.kill_scope;
        self.all_loans.push(loan);

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
            //        mutbl: m_const,
            //        gen_scope: borrow_id,
            //        kill_scope: kill_scope,
            //        span: borrow_span,
            //        restrictions: restrictions
            //    }
        // }

        fn check_mutability(bccx: @BorrowckCtxt,
                            borrow_span: span,
                            cmt: mc::cmt,
                            req_mutbl: ast::mutability) {
            //! Implements the M-* rules in doc.rs.

            match req_mutbl {
                m_const => {
                    // Data of any mutability can be lent as const.
                }

                m_imm => {
                    match cmt.mutbl {
                        mc::McImmutable | mc::McDeclared | mc::McInherited => {
                            // both imm and mut data can be lent as imm;
                            // for mutable data, this is a freeze
                        }
                        mc::McReadOnly => {
                            bccx.report(BckError {span: borrow_span,
                                                  cmt: cmt,
                                                  code: err_mutbl(req_mutbl)});
                        }
                    }
                }

                m_mutbl => {
                    // Only mutable data can be lent as mutable.
                    if !cmt.mutbl.is_mutable() {
                        bccx.report(BckError {span: borrow_span,
                                              cmt: cmt,
                                              code: err_mutbl(req_mutbl)});
                    }
                }
            }
        }
    }

    pub fn restriction_set(&self, req_mutbl: ast::mutability)
                           -> RestrictionSet {
        match req_mutbl {
            m_const => RESTR_EMPTY,
            m_imm   => RESTR_EMPTY | RESTR_MUTATE | RESTR_CLAIM,
            m_mutbl => RESTR_EMPTY | RESTR_MUTATE | RESTR_CLAIM | RESTR_FREEZE
        }
    }

    pub fn mark_loan_path_as_mutated(&self, loan_path: @LoanPath) {
        //! For mutable loans of content whose mutability derives
        //! from a local variable, mark the mutability decl as necessary.

        match *loan_path {
            LpVar(local_id) => {
                self.tcx().used_mut_nodes.insert(local_id);
            }
            LpExtend(base, mc::McInherited, _) => {
                self.mark_loan_path_as_mutated(base);
            }
            LpExtend(_, mc::McDeclared, _) |
            LpExtend(_, mc::McImmutable, _) |
            LpExtend(_, mc::McReadOnly, _) => {
            }
        }
    }

    pub fn compute_gen_scope(&self,
                             borrow_id: ast::node_id,
                             loan_scope: ast::node_id)
                             -> ast::node_id {
        //! Determine when to introduce the loan. Typically the loan
        //! is introduced at the point of the borrow, but in some cases,
        //! notably method arguments, the loan may be introduced only
        //! later, once it comes into scope.

        let rm = self.bccx.tcx.region_maps;
        if rm.is_subscope_of(borrow_id, loan_scope) {
            borrow_id
        } else {
            loan_scope
        }
    }

    pub fn compute_kill_scope(&self, loan_scope: ast::node_id, lp: @LoanPath)
                              -> ast::node_id {
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
        //! In this case, the borrowed pointer (`'a`) outlives the
        //! variable `v` that hosts it. Note that this doesn't come up
        //! with immutable `&` pointers, because borrows of such pointers
        //! do not require restrictions and hence do not cause a loan.

        let rm = self.bccx.tcx.region_maps;
        let lexical_scope = rm.encl_scope(lp.node_id());
        if rm.is_subscope_of(lexical_scope, loan_scope) {
            lexical_scope
        } else {
            assert!(rm.is_subscope_of(loan_scope, lexical_scope));
            loan_scope
        }
    }

    fn gather_fn_arg_patterns(&mut self,
                              decl: &ast::fn_decl,
                              body: &ast::blk) {
        /*!
         * Walks the patterns for fn arguments, checking that they
         * do not attempt illegal moves or create refs that outlive
         * the arguments themselves. Just a shallow wrapper around
         * `gather_pat()`.
         */

        let mc_ctxt = self.bccx.mc_ctxt();
        for decl.inputs.iter().advance |arg| {
            let arg_ty = ty::node_id_to_type(self.tcx(), arg.pat.id);

            let arg_cmt = mc_ctxt.cat_rvalue(
                arg.id,
                arg.pat.span,
                body.id, // Arguments live only as long as the fn body.
                arg_ty);

            self.gather_pat(arg_cmt, arg.pat, None);
        }
    }

    fn gather_pat(&mut self,
                  discr_cmt: mc::cmt,
                  root_pat: @ast::pat,
                  arm_match_ids: Option<(ast::node_id, ast::node_id)>) {
        /*!
         * Walks patterns, examining the bindings to determine if they
         * cause borrows (`ref` bindings, vector patterns) or
         * moves (non-`ref` bindings with linear type).
         */

        do self.bccx.cat_pattern(discr_cmt, root_pat) |cmt, pat| {
            match pat.node {
              ast::pat_ident(bm, _, _) if self.pat_is_binding(pat) => {
                match bm {
                  ast::bind_by_ref(mutbl) => {
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
                            let arm_scope = ty::re_scope(arm_id);
                            if self.bccx.is_subregion_of(scope_r, arm_scope) {
                                self.bccx.cat_discr(cmt, match_id)
                            } else {
                                cmt
                            }
                        }
                    };
                    self.guarantee_valid(pat.id, pat.span,
                                         cmt_discr, mutbl, scope_r);
                  }
                  ast::bind_infer => {
                      // No borrows here, but there may be moves
                      if self.bccx.is_move(pat.id) {
                          gather_moves::gather_move_from_pat(
                              self.bccx, self.move_data, pat, cmt);
                      }
                  }
                }
              }

              ast::pat_vec(_, Some(slice_pat), _) => {
                  // The `slice_pat` here creates a slice into the
                  // original vector.  This is effectively a borrow of
                  // the elements of the vector being matched.

                  let slice_ty = ty::node_id_to_type(self.tcx(),
                                                     slice_pat.id);
                  let (slice_mutbl, slice_r) =
                      self.vec_slice_info(slice_pat, slice_ty);
                  let mcx = self.bccx.mc_ctxt();
                  let cmt_index = mcx.cat_index(slice_pat, cmt, 0);

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
                  self.guarantee_valid(pat.id, pat.span,
                                       cmt_index, slice_mutbl, slice_r);
              }

              _ => {}
            }
        }
    }

    pub fn vec_slice_info(&self, pat: @ast::pat, slice_ty: ty::t)
                          -> (ast::mutability, ty::Region) {
        /*!
         *
         * In a pattern like [a, b, ..c], normally `c` has slice type,
         * but if you have [a, b, ..ref c], then the type of `ref c`
         * will be `&&[]`, so to extract the slice details we have
         * to recurse through rptrs.
         */

        match ty::get(slice_ty).sty {
            ty::ty_evec(slice_mt, ty::vstore_slice(slice_r)) => {
                (slice_mt.mutbl, slice_r)
            }

            ty::ty_rptr(_, ref mt) => {
                self.vec_slice_info(pat, mt.ty)
            }

            _ => {
                self.tcx().sess.span_bug(
                    pat.span,
                    fmt!("Type of slice pattern is not a slice"));
            }
        }
    }

    pub fn pat_is_variant_or_struct(&self, pat: @ast::pat) -> bool {
        pat_util::pat_is_variant_or_struct(self.bccx.tcx.def_map, pat)
    }

    pub fn pat_is_binding(&self, pat: @ast::pat) -> bool {
        pat_util::pat_is_binding(self.bccx.tcx.def_map, pat)
    }
}

// Setting up info that preserve needs.
// This is just the most convenient place to do it.
fn add_stmt_to_map(stmt: @ast::stmt,
                   (this, vt): (@mut GatherLoanCtxt,
                                visit::vt<@mut GatherLoanCtxt>)) {
    match stmt.node {
        ast::stmt_expr(_, id) | ast::stmt_semi(_, id) => {
            this.bccx.stmt_map.insert(id);
        }
        _ => ()
    }
    visit::visit_stmt(stmt, (this, vt));
}
