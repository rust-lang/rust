// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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
use euv = middle::expr_use_visitor;
use mc = middle::mem_categorization;
use middle::ty;
use util::ppaux::{Repr};

use syntax::ast;
use syntax::codemap::Span;
use syntax::visit;
use syntax::visit::{Visitor};
use syntax::ast::{Expr, FnDecl, Block, NodeId, Pat};

mod lifetime;
mod restrictions;
mod gather_moves;
mod move_error;

pub fn gather_loans_in_fn(bccx: &BorrowckCtxt,
                          decl: &ast::FnDecl,
                          body: &ast::Block)
                          -> (Vec<Loan>, move_data::MoveData)
{
    let mut glcx = GatherLoanCtxt {
        bccx: bccx,
        all_loans: Vec::new(),
        item_ub: body.id,
        move_data: MoveData::new(),
        move_error_collector: move_error::MoveErrorCollector::new(),
    };

    {
        let mut euv = euv::ExprUseVisitor::new(&mut glcx, bccx.tcx);
        euv.walk_fn(decl, body);
    }

    glcx.report_potential_errors();
    let GatherLoanCtxt { all_loans, move_data, .. } = glcx;
    (all_loans, move_data)
}

struct GatherLoanCtxt<'a> {
    bccx: &'a BorrowckCtxt<'a>,
    move_data: move_data::MoveData,
    move_error_collector: move_error::MoveErrorCollector,
    all_loans: Vec<Loan>,
    item_ub: ast::NodeId,
}

impl<'a> euv::Delegate for GatherLoanCtxt<'a> {
    fn consume(&mut self,
               consume_id: ast::NodeId,
               _consume_span: Span,
               cmt: mc::cmt,
               mode: euv::ConsumeMode) {
        debug!("consume(consume_id={}, cmt={}, mode={})",
               consume_id, cmt.repr(self.tcx()), mode);

        match mode {
            euv::Move(move_reason) => {
                gather_moves::gather_move_from_expr(
                    self.bccx, &self.move_data, &self.move_error_collector,
                    consume_id, cmt, move_reason);
            }
            euv::Copy => { }
        }
    }

    fn consume_pat(&mut self,
                   consume_pat: &ast::Pat,
                   cmt: mc::cmt,
                   mode: euv::ConsumeMode) {
        debug!("consume_pat(consume_pat={}, cmt={}, mode={})",
               consume_pat.repr(self.tcx()),
               cmt.repr(self.tcx()),
               mode);

        match mode {
            euv::Copy => { return; }
            euv::Move(_) => { }
        }

        gather_moves::gather_move_from_pat(
            self.bccx, &self.move_data, &self.move_error_collector,
            consume_pat, cmt);
    }

    fn borrow(&mut self,
              borrow_id: ast::NodeId,
              borrow_span: Span,
              cmt: mc::cmt,
              loan_region: ty::Region,
              bk: ty::BorrowKind,
              loan_cause: euv::LoanCause)
    {
        debug!("borrow(borrow_id={}, cmt={}, loan_region={}, \
               bk={}, loan_cause={:?})",
               borrow_id, cmt.repr(self.tcx()), loan_region,
               bk, loan_cause);

        self.guarantee_valid(borrow_id,
                             borrow_span,
                             cmt,
                             bk,
                             loan_region,
                             loan_cause);
    }

    fn mutate(&mut self,
              assignment_id: ast::NodeId,
              assignment_span: Span,
              assignee_cmt: mc::cmt,
              mode: euv::MutateMode)
    {
        debug!("mutate(assignment_id={}, assignee_cmt={})",
               assignment_id, assignee_cmt.repr(self.tcx()));

        match opt_loan_path(&assignee_cmt) {
            Some(lp) => {
                gather_moves::gather_assignment(self.bccx, &self.move_data,
                                                assignment_id, assignment_span,
                                                lp, assignee_cmt.id, mode);
            }
            None => {
                // This can occur with e.g. `*foo() = 5`.  In such
                // cases, there is no need to check for conflicts
                // with moves etc, just ignore.
            }
        }
    }

    fn decl_without_init(&mut self, id: ast::NodeId, span: Span) {
        gather_moves::gather_decl(self.bccx, &self.move_data, id, span, id);
    }
}

/// Implements the A-* rules in doc.rs.
fn check_aliasability(bccx: &BorrowckCtxt,
                      borrow_span: Span,
                      loan_cause: euv::LoanCause,
                      cmt: mc::cmt,
                      req_kind: ty::BorrowKind)
                      -> Result<(),()> {

    match (cmt.freely_aliasable(bccx.tcx), req_kind) {
        (None, _) => {
            /* Uniquely accessible path -- OK for `&` and `&mut` */
            Ok(())
        }
        (Some(mc::AliasableStatic(safety)), ty::ImmBorrow) => {
            // Borrow of an immutable static item:
            match safety {
                mc::InteriorUnsafe => {
                    // If the static item contains an Unsafe<T>, it has interior mutability.
                    // In such cases, we cannot permit it to be borrowed, because the
                    // static item resides in immutable memory and mutating it would
                    // cause segfaults.
                    bccx.tcx.sess.span_err(borrow_span,
                                           "borrow of immutable static items \
                                            with unsafe interior is not \
                                            allowed");
                    Err(())
                }
                mc::InteriorSafe => {
                    // Immutable static can be borrowed, no problem.
                    Ok(())
                }
            }
        }
        (Some(mc::AliasableStaticMut(..)), _) => {
            // Even touching a static mut is considered unsafe. We assume the
            // user knows what they're doing in these cases.
            Ok(())
        }
        (Some(alias_cause), ty::UniqueImmBorrow) |
        (Some(alias_cause), ty::MutBorrow) => {
            bccx.report_aliasability_violation(
                        borrow_span,
                        BorrowViolation(loan_cause),
                        alias_cause);
            Err(())
        }
        (_, _) => {
            Ok(())
        }
    }
}

impl<'a> GatherLoanCtxt<'a> {
    pub fn tcx(&self) -> &'a ty::ctxt { self.bccx.tcx }

    fn guarantee_valid(&mut self,
                       borrow_id: ast::NodeId,
                       borrow_span: Span,
                       cmt: mc::cmt,
                       req_kind: ty::BorrowKind,
                       loan_region: ty::Region,
                       cause: euv::LoanCause) {
        /*!
         * Guarantees that `addr_of(cmt)` will be valid for the duration of
         * `static_scope_r`, or reports an error.  This may entail taking
         * out loans, which will be added to the `req_loan_map`.  This can
         * also entail "rooting" GC'd pointers, which means ensuring
         * dynamically that they are not freed.
         */

        debug!("guarantee_valid(borrow_id={:?}, cmt={}, \
                req_mutbl={:?}, loan_region={:?})",
               borrow_id,
               cmt.repr(self.tcx()),
               req_kind,
               loan_region);

        // a loan for the empty region can never be dereferenced, so
        // it is always safe
        if loan_region == ty::ReEmpty {
            return;
        }

        // Check that the lifetime of the borrow does not exceed
        // the lifetime of the data being borrowed.
        if lifetime::guarantee_lifetime(self.bccx, self.item_ub,
                                        borrow_span, cause, cmt.clone(), loan_region,
                                        req_kind).is_err() {
            return; // reported an error, no sense in reporting more.
        }

        // Check that we don't allow mutable borrows of non-mutable data.
        if check_mutability(self.bccx, borrow_span, cause,
                            cmt.clone(), req_kind).is_err() {
            return; // reported an error, no sense in reporting more.
        }

        // Check that we don't allow mutable borrows of aliasable data.
        if check_aliasability(self.bccx, borrow_span, cause,
                              cmt.clone(), req_kind).is_err() {
            return; // reported an error, no sense in reporting more.
        }

        // Compute the restrictions that are required to enforce the
        // loan is safe.
        let restr = restrictions::compute_restrictions(
            self.bccx, borrow_span, cause,
            cmt.clone(), loan_region, self.restriction_set(req_kind));

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
                            format!("invalid borrow lifetime: {:?}",
                                    loan_region).as_slice());
                    }
                };
                debug!("loan_scope = {:?}", loan_scope);

                let gen_scope = self.compute_gen_scope(borrow_id, loan_scope);
                debug!("gen_scope = {:?}", gen_scope);

                let kill_scope = self.compute_kill_scope(loan_scope, &*loan_path);
                debug!("kill_scope = {:?}", kill_scope);

                if req_kind == ty::MutBorrow {
                    self.mark_loan_path_as_mutated(&*loan_path);
                }

                Loan {
                    index: self.all_loans.len(),
                    loan_path: loan_path,
                    kind: req_kind,
                    gen_scope: gen_scope,
                    kill_scope: kill_scope,
                    span: borrow_span,
                    restrictions: restrictions,
                    cause: cause,
                }
            }
        };

        debug!("guarantee_valid(borrow_id={:?}), loan={}",
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
            //        mutbl: ConstMutability,
            //        gen_scope: borrow_id,
            //        kill_scope: kill_scope,
            //        span: borrow_span,
            //        restrictions: restrictions
            //    }
        // }

        fn check_mutability(bccx: &BorrowckCtxt,
                            borrow_span: Span,
                            cause: euv::LoanCause,
                            cmt: mc::cmt,
                            req_kind: ty::BorrowKind)
                            -> Result<(),()> {
            //! Implements the M-* rules in doc.rs.

            match req_kind {
                ty::UniqueImmBorrow | ty::ImmBorrow => {
                    match cmt.mutbl {
                        // I am intentionally leaving this here to help
                        // refactoring if, in the future, we should add new
                        // kinds of mutability.
                        mc::McImmutable | mc::McDeclared | mc::McInherited => {
                            // both imm and mut data can be lent as imm;
                            // for mutable data, this is a freeze
                            Ok(())
                        }
                    }
                }

                ty::MutBorrow => {
                    // Only mutable data can be lent as mutable.
                    if !cmt.mutbl.is_mutable() {
                        Err(bccx.report(BckError { span: borrow_span,
                                                   cause: cause,
                                                   cmt: cmt,
                                                   code: err_mutbl }))
                    } else {
                        Ok(())
                    }
                }
            }
        }
    }

    fn restriction_set(&self, req_kind: ty::BorrowKind) -> RestrictionSet {
        match req_kind {
            // If borrowing data as immutable, no mutation allowed:
            ty::ImmBorrow => RESTR_MUTATE,

            // If borrowing data as mutable, no mutation nor other
            // borrows allowed:
            ty::MutBorrow => RESTR_MUTATE | RESTR_FREEZE,

            // If borrowing data as unique imm, no mutation nor other
            // borrows allowed:
            ty::UniqueImmBorrow => RESTR_MUTATE | RESTR_FREEZE,
        }
    }

    pub fn mark_loan_path_as_mutated(&self, loan_path: &LoanPath) {
        //! For mutable loans of content whose mutability derives
        //! from a local variable, mark the mutability decl as necessary.

        match *loan_path {
            LpVar(local_id) => {
                self.tcx().used_mut_nodes.borrow_mut().insert(local_id);
            }
            LpExtend(ref base, mc::McInherited, _) => {
                self.mark_loan_path_as_mutated(&**base);
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

    pub fn compute_kill_scope(&self, loan_scope: ast::NodeId, lp: &LoanPath)
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

    pub fn report_potential_errors(&self) {
        self.move_error_collector.report_potential_errors(self.bccx);
    }
}

/// Context used while gathering loans on static initializers
///
/// This visitor walks static initializer's expressions and makes
/// sure the loans being taken are sound.
struct StaticInitializerCtxt<'a> {
    bccx: &'a BorrowckCtxt<'a>
}

impl<'a> visit::Visitor<()> for StaticInitializerCtxt<'a> {
    fn visit_expr(&mut self, ex: &Expr, _: ()) {
        match ex.node {
            ast::ExprAddrOf(mutbl, ref base) => {
                let base_cmt = self.bccx.cat_expr(&**base);
                let borrow_kind = ty::BorrowKind::from_mutbl(mutbl);
                // Check that we don't allow borrows of unsafe static items.
                if check_aliasability(self.bccx, ex.span, euv::AddrOf,
                                      base_cmt, borrow_kind).is_err() {
                    return; // reported an error, no sense in reporting more.
                }
            }
            _ => {}
        }

        visit::walk_expr(self, ex, ());
    }
}

pub fn gather_loans_in_static_initializer(bccx: &mut BorrowckCtxt, expr: &ast::Expr) {

    debug!("gather_loans_in_static_initializer(expr={})", expr.repr(bccx.tcx));

    let mut sicx = StaticInitializerCtxt {
        bccx: bccx
    };

    sicx.visit_expr(expr, ());
}
