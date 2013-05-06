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

use middle::borrowck::preserve::{PreserveCondition, PcOk, PcIfPure};
use middle::borrowck::{Loan, bckerr, bckres, BorrowckCtxt, err_mutbl};
use middle::borrowck::{LoanKind, TotalFreeze, PartialFreeze,
                       TotalTake, PartialTake, Immobile};
use middle::borrowck::ReqMaps;
use middle::borrowck::loan;
use middle::mem_categorization::{cmt, mem_categorization_ctxt};
use middle::pat_util;
use middle::ty::{ty_region};
use middle::ty;
use util::common::indenter;
use util::ppaux::{Repr, region_to_str};

use core::hashmap::{HashSet, HashMap};
use syntax::ast::{m_const, m_imm, m_mutbl};
use syntax::ast;
use syntax::codemap::span;
use syntax::print::pprust;
use syntax::visit;

/// Context used while gathering loans:
///
/// - `bccx`: the the borrow check context
/// - `req_maps`: the maps computed by `gather_loans()`, see def'n of the
///   struct `ReqMaps` for more info
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
    req_maps: ReqMaps,
    item_ub: ast::node_id,
    root_ub: ast::node_id,
    ignore_adjustments: HashSet<ast::node_id>
}

pub fn gather_loans(bccx: @BorrowckCtxt, crate: @ast::crate) -> ReqMaps {
    let glcx = @mut GatherLoanCtxt {
        bccx: bccx,
        req_maps: ReqMaps { req_loan_map: HashMap::new(),
                            pure_map: HashMap::new() },
        item_ub: 0,
        root_ub: 0,
        ignore_adjustments: HashSet::new()
    };
    let v = visit::mk_vt(@visit::Visitor {visit_expr: req_loans_in_expr,
                                          visit_fn: req_loans_in_fn,
                                          visit_stmt: add_stmt_to_map,
                                          .. *visit::default_visitor()});
    visit::visit_crate(crate, glcx, v);
    let @GatherLoanCtxt{req_maps, _} = glcx;
    return req_maps;
}

fn req_loans_in_fn(fk: &visit::fn_kind,
                   decl: &ast::fn_decl,
                   body: &ast::blk,
                   sp: span,
                   id: ast::node_id,
                   self: @mut GatherLoanCtxt,
                   v: visit::vt<@mut GatherLoanCtxt>) {
    // see explanation attached to the `root_ub` field:
    let old_item_id = self.item_ub;
    let old_root_ub = self.root_ub;
    self.root_ub = body.node.id;

    match *fk {
        visit::fk_anon(*) | visit::fk_fn_block(*) => {}
        visit::fk_item_fn(*) | visit::fk_method(*) => {
            self.item_ub = body.node.id;
        }
    }

    visit::visit_fn(fk, decl, body, sp, id, self, v);
    self.root_ub = old_root_ub;
    self.item_ub = old_item_id;
}

fn req_loans_in_expr(ex: @ast::expr,
                     self: @mut GatherLoanCtxt,
                     vt: visit::vt<@mut GatherLoanCtxt>) {
    let bccx = self.bccx;
    let tcx = bccx.tcx;
    let old_root_ub = self.root_ub;

    debug!("req_loans_in_expr(expr=%?/%s)",
           ex.id, pprust::expr_to_str(ex, tcx.sess.intr()));

    // If this expression is borrowed, have to ensure it remains valid:
    {
        let mut this = &mut *self;
        if !this.ignore_adjustments.contains(&ex.id) {
            for tcx.adjustments.find(&ex.id).each |&adjustments| {
                this.guarantee_adjustments(ex, *adjustments);
            }
        }
    }

    // Special checks for various kinds of expressions:
    match ex.node {
      ast::expr_addr_of(mutbl, base) => {
        let base_cmt = self.bccx.cat_expr(base);

        // make sure that the thing we are pointing out stays valid
        // for the lifetime `scope_r` of the resulting ptr:
        let scope_r = ty_region(tcx, ex.span, tcx.ty(ex));
        self.guarantee_valid(base_cmt, mutbl, scope_r);
        visit::visit_expr(ex, self, vt);
      }

      ast::expr_match(ex_v, ref arms) => {
        let cmt = self.bccx.cat_expr(ex_v);
        for (*arms).each |arm| {
            for arm.pats.each |pat| {
                self.gather_pat(cmt, *pat, arm.body.node.id, ex.id);
            }
        }
        visit::visit_expr(ex, self, vt);
      }

      ast::expr_index(rcvr, _) |
      ast::expr_binary(_, rcvr, _) |
      ast::expr_unary(_, rcvr) |
      ast::expr_assign_op(_, rcvr, _)
      if self.bccx.method_map.contains_key(&ex.id) => {
        // Receivers in method calls are always passed by ref.
        //
        // Here, in an overloaded operator, the call is this expression,
        // and hence the scope of the borrow is this call.
        //
        // FIX? / NOT REALLY---technically we should check the other
        // argument and consider the argument mode.  But how annoying.
        // And this problem when goes away when argument modes are
        // phased out.  So I elect to leave this undone.
        let scope_r = ty::re_scope(ex.id);
        let rcvr_cmt = self.bccx.cat_expr(rcvr);
        self.guarantee_valid(rcvr_cmt, m_imm, scope_r);

        // FIXME (#3387): Total hack: Ignore adjustments for the left-hand
        // side. Their regions will be inferred to be too large.
        self.ignore_adjustments.insert(rcvr.id);

        visit::visit_expr(ex, self, vt);
      }

      // FIXME--#3387
      // ast::expr_binary(_, lhs, rhs) => {
      //     // Universal comparison operators like ==, >=, etc
      //     // take their arguments by reference.
      //     let lhs_ty = ty::expr_ty(self.tcx(), lhs);
      //     if !ty::type_is_scalar(lhs_ty) {
      //         let scope_r = ty::re_scope(ex.id);
      //         let lhs_cmt = self.bccx.cat_expr(lhs);
      //         self.guarantee_valid(lhs_cmt, m_imm, scope_r);
      //         let rhs_cmt = self.bccx.cat_expr(rhs);
      //         self.guarantee_valid(rhs_cmt, m_imm, scope_r);
      //     }
      //     visit::visit_expr(ex, self, vt);
      // }

      ast::expr_field(rcvr, _, _)
      if self.bccx.method_map.contains_key(&ex.id) => {
        // Receivers in method calls are always passed by ref.
        //
        // Here, the field a.b is in fact a closure.  Eventually, this
        // should be an &fn, but for now it's an @fn.  In any case,
        // the enclosing scope is either the call where it is a rcvr
        // (if used like `a.b(...)`), the call where it's an argument
        // (if used like `x(a.b)`), or the block (if used like `let x
        // = a.b`).
        let scope_r = self.tcx().region_maps.encl_region(ex.id);
        let rcvr_cmt = self.bccx.cat_expr(rcvr);
        self.guarantee_valid(rcvr_cmt, m_imm, scope_r);
        visit::visit_expr(ex, self, vt);
      }

      // see explanation attached to the `root_ub` field:
      ast::expr_while(cond, ref body) => {
        // during the condition, can only root for the condition
        self.root_ub = cond.id;
        (vt.visit_expr)(cond, self, vt);

        // during body, can only root for the body
        self.root_ub = body.node.id;
        (vt.visit_block)(body, self, vt);
      }

      // see explanation attached to the `root_ub` field:
      ast::expr_loop(ref body, _) => {
        self.root_ub = body.node.id;
        visit::visit_expr(ex, self, vt);
      }

      _ => {
        visit::visit_expr(ex, self, vt);
      }
    }

    // Check any contained expressions:

    self.root_ub = old_root_ub;
}

pub impl GatherLoanCtxt {
    fn tcx(&mut self) -> ty::ctxt { self.bccx.tcx }

    fn guarantee_adjustments(&mut self,
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
                let mcx = &mem_categorization_ctxt {
                    tcx: self.tcx(),
                    method_map: self.bccx.method_map};
                let cmt = mcx.cat_expr_autoderefd(expr, autoderefs);
                debug!("after autoderef, cmt=%s", self.bccx.cmt_to_repr(cmt));

                match autoref.kind {
                    ty::AutoPtr => {
                        self.guarantee_valid(cmt,
                                             autoref.mutbl,
                                             autoref.region)
                    }
                    ty::AutoBorrowVec | ty::AutoBorrowVecRef => {
                        let cmt_index = mcx.cat_index(expr, cmt);
                        self.guarantee_valid(cmt_index,
                                             autoref.mutbl,
                                             autoref.region)
                    }
                    ty::AutoBorrowFn => {
                        let cmt_deref = mcx.cat_deref_fn(expr, cmt, 0);
                        self.guarantee_valid(cmt_deref,
                                             autoref.mutbl,
                                             autoref.region)
                    }
                }
            }
        }
    }

    // guarantees that addr_of(cmt) will be valid for the duration of
    // `static_scope_r`, or reports an error.  This may entail taking
    // out loans, which will be added to the `req_loan_map`.  This can
    // also entail "rooting" GC'd pointers, which means ensuring
    // dynamically that they are not freed.
    fn guarantee_valid(&mut self,
                       cmt: cmt,
                       req_mutbl: ast::mutability,
                       scope_r: ty::Region)
    {

        let loan_kind = match req_mutbl {
            m_mutbl => TotalTake,
            m_imm => TotalFreeze,
            m_const => Immobile
        };

        self.bccx.stats.guaranteed_paths += 1;

        debug!("guarantee_valid(cmt=%s, req_mutbl=%?, \
                loan_kind=%?, scope_r=%s)",
               self.bccx.cmt_to_repr(cmt),
               req_mutbl,
               loan_kind,
               region_to_str(self.tcx(), scope_r));
        let _i = indenter();

        match cmt.lp {
          // If this expression is a loanable path, we MUST take out a
          // loan.  This is somewhat non-obvious.  You might think,
          // for example, that if we have an immutable local variable
          // `x` whose value is being borrowed, we could rely on `x`
          // not to change.  This is not so, however, because even
          // immutable locals can be moved.  So we take out a loan on
          // `x`, guaranteeing that it remains immutable for the
          // duration of the reference: if there is an attempt to move
          // it within that scope, the loan will be detected and an
          // error will be reported.
          Some(_) => {
              match loan::loan(self.bccx, cmt, scope_r, loan_kind) {
                  Err(ref e) => { self.bccx.report((*e)); }
                  Ok(loans) => {
                      self.add_loans(cmt, loan_kind, scope_r, loans);
                  }
              }
          }

          // The path is not loanable: in that case, we must try and
          // preserve it dynamically (or see that it is preserved by
          // virtue of being rooted in some immutable path).  We must
          // also check that the mutability of the desired pointer
          // matches with the actual mutability (but if an immutable
          // pointer is desired, that is ok as long as we are pure)
          None => {
            let result: bckres<PreserveCondition> = {
                do self.check_mutbl(loan_kind, cmt).chain |pc1| {
                    do self.bccx.preserve(cmt, scope_r,
                                          self.item_ub,
                                          self.root_ub).chain |pc2| {
                        Ok(pc1.combine(pc2))
                    }
                }
            };

            match result {
                Ok(PcOk) => {
                    debug!("result of preserve: PcOk");

                    // we were able guarantee the validity of the ptr,
                    // perhaps by rooting or because it is immutably
                    // rooted.  good.
                    self.bccx.stats.stable_paths += 1;
                }
                Ok(PcIfPure(ref e)) => {
                    debug!("result of preserve: %?", PcIfPure((*e)));

                    // we are only able to guarantee the validity if
                    // the scope is pure
                    match scope_r {
                        ty::re_scope(pure_id) => {
                            // if the scope is some block/expr in the
                            // fn, then just require that this scope
                            // be pure
                            self.req_maps.pure_map.insert(pure_id, *e);
                            self.bccx.stats.req_pure_paths += 1;

                            debug!("requiring purity for scope %?",
                                   scope_r);

                            if self.tcx().sess.borrowck_note_pure() {
                                self.bccx.span_note(
                                    cmt.span,
                                    fmt!("purity required"));
                            }
                        }
                        _ => {
                            // otherwise, we can't enforce purity for
                            // that scope, so give up and report an
                            // error
                            self.bccx.report((*e));
                        }
                    }
                }
                Err(ref e) => {
                    // we cannot guarantee the validity of this pointer
                    debug!("result of preserve: error");
                    self.bccx.report((*e));
                }
            }
          }
        }
    }

    // Check that the pat `cmt` is compatible with the required
    // mutability, presuming that it can be preserved to stay alive
    // long enough.
    //
    // For example, if you have an expression like `&x.f` where `x`
    // has type `@mut{f:int}`, this check might fail because `&x.f`
    // reqires an immutable pointer, but `f` lives in (aliased)
    // mutable memory.
    fn check_mutbl(&mut self,
                   loan_kind: LoanKind,
                   cmt: cmt)
                -> bckres<PreserveCondition> {
        debug!("check_mutbl(loan_kind=%?, cmt.mutbl=%?)",
               loan_kind, cmt.mutbl);

        match loan_kind {
            Immobile => Ok(PcOk),

            TotalTake | PartialTake => {
                if cmt.mutbl.is_mutable() {
                    Ok(PcOk)
                } else {
                    Err(bckerr { cmt: cmt, code: err_mutbl(loan_kind) })
                }
            }

            TotalFreeze | PartialFreeze => {
                if cmt.mutbl.is_immutable() {
                    Ok(PcOk)
                } else if cmt.cat.is_mutable_box() {
                    Ok(PcOk)
                } else {
                    // Eventually:
                    let e = bckerr {cmt: cmt,
                                    code: err_mutbl(loan_kind)};
                    Ok(PcIfPure(e))
                }
            }
        }
    }

    fn add_loans(&mut self,
                 cmt: cmt,
                 loan_kind: LoanKind,
                 scope_r: ty::Region,
                 loans: ~[Loan]) {
        if loans.len() == 0 {
            return;
        }

        // Normally we wouldn't allow `re_free` here. However, in this case
        // it should be sound. Below is nmatsakis' reasoning:
        //
        // Perhaps [this permits] a function kind of like this one here, which
        // consumes one mut pointer and returns a narrower one:
        //
        //     struct Foo { f: int }
        //     fn foo(p: &'v mut Foo) -> &'v mut int { &mut p.f }
        //
        // I think this should work fine but there is more subtlety to it than
        // I at first imagined. Unfortunately it's a very important use case,
        // I think, so it really ought to work. The changes you [pcwalton]
        // made to permit re_free() do permit this case, I think, but I'm not
        // sure what else they permit. I have to think that over a bit.
        //
        // Ordinarily, a loan with scope re_free wouldn't make sense, because
        // you couldn't enforce it. But in this case, your function signature
        // informs the caller that you demand exclusive access to p and its
        // contents for the lifetime v. Since borrowed pointers are
        // non-copyable, they must have (a) made a borrow which will enforce
        // those conditions and then (b) given you the resulting pointer.
        // Therefore, they should be respecting the loan. So it actually seems
        // that it's ok in this case to have a loan with re_free, so long as
        // the scope of the loan is no greater than the region pointer on
        // which it is based. Neat but not something I had previously
        // considered all the way through. (Note that we already rely on
        // similar reasoning to permit you to return borrowed pointers into
        // immutable structures, this is just the converse I suppose)

        let scope_id = match scope_r {
            ty::re_scope(scope_id) |
            ty::re_free(ty::FreeRegion {scope_id, _}) => {
                scope_id
            }
            _ => {
                self.bccx.tcx.sess.span_bug(
                    cmt.span,
                    fmt!("loans required but scope is scope_region is %s \
                          (%?)",
                         region_to_str(self.tcx(), scope_r),
                         scope_r));
            }
        };

        self.add_loans_to_scope_id(scope_id, loans);

        if loan_kind.is_freeze() && !cmt.mutbl.is_immutable() {
            self.bccx.stats.loaned_paths_imm += 1;

            if self.tcx().sess.borrowck_note_loan() {
                self.bccx.span_note(
                    cmt.span,
                    fmt!("immutable loan required"));
            }
        } else {
            self.bccx.stats.loaned_paths_same += 1;
        }
    }

    fn add_loans_to_scope_id(&mut self,
                             scope_id: ast::node_id,
                             loans: ~[Loan]) {
        debug!("adding %u loans to scope_id %?: %s",
               loans.len(), scope_id,
               str::connect(loans.map(|l| self.bccx.loan_to_repr(l)), ", "));
        match self.req_maps.req_loan_map.find(&scope_id) {
            Some(req_loans) => {
                req_loans.push_all(loans);
                return;
            }
            None => {}
        }
        self.req_maps.req_loan_map.insert(scope_id, @mut loans);
    }

    fn gather_pat(@mut self,
                  discr_cmt: cmt,
                  root_pat: @ast::pat,
                  arm_id: ast::node_id,
                  match_id: ast::node_id) {
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
                    let scope_r = ty_region(self.tcx(), pat.span,
                                            self.tcx().ty(pat));

                    // if the scope of the region ptr turns out to be
                    // specific to this arm, wrap the categorization with
                    // a cat_discr() node.  There is a detailed discussion
                    // of the function of this node in method preserve():
                    let arm_scope = ty::re_scope(arm_id);
                    if self.bccx.is_subregion_of(scope_r, arm_scope) {
                        let cmt_discr = self.bccx.cat_discr(cmt, match_id);
                        self.guarantee_valid(cmt_discr, mutbl, scope_r);
                    } else {
                        self.guarantee_valid(cmt, mutbl, scope_r);
                    }
                  }
                  ast::bind_by_copy | ast::bind_infer => {
                    // Nothing to do here; neither copies nor moves induce
                    // borrows.
                  }
                }
              }

              ast::pat_vec(_, Some(slice_pat), _) => {
                  // The `slice_pat` here creates a slice into the
                  // original vector.  This is effectively a borrow of
                  // the elements of the vector being matched.

                  let slice_ty = self.tcx().ty(slice_pat);
                  let (slice_mutbl, slice_r) =
                      self.vec_slice_info(slice_pat, slice_ty);
                  let mcx = self.bccx.mc_ctxt();
                  let cmt_index = mcx.cat_index(slice_pat, cmt);
                  self.guarantee_valid(cmt_index, slice_mutbl, slice_r);
              }

              _ => {}
            }
        }
    }

    fn vec_slice_info(@mut self,
                      pat: @ast::pat,
                      slice_ty: ty::t) -> (ast::mutability, ty::Region) {
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

    fn pat_is_variant_or_struct(@mut self, pat: @ast::pat) -> bool {
        pat_util::pat_is_variant_or_struct(self.bccx.tcx.def_map, pat)
    }

    fn pat_is_binding(@mut self, pat: @ast::pat) -> bool {
        pat_util::pat_is_binding(self.bccx.tcx.def_map, pat)
    }
}

// Setting up info that preserve needs.
// This is just the most convenient place to do it.
fn add_stmt_to_map(stmt: @ast::stmt,
                   self: @mut GatherLoanCtxt,
                   vt: visit::vt<@mut GatherLoanCtxt>) {
    match stmt.node {
        ast::stmt_expr(_, id) | ast::stmt_semi(_, id) => {
            self.bccx.stmt_map.insert(id);
        }
        _ => ()
    }
    visit::visit_stmt(stmt, self, vt);
}
