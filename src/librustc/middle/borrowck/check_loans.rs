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
// Checking loans
//
// Phase 2 of check: we walk down the tree and check that:
// 1. assignments are always made to mutable locations;
// 2. loans made in overlapping scopes do not conflict
// 3. assignments do not affect things loaned out as immutable
// 4. moves do not affect things loaned out in any way


use middle::borrowck::*;
use middle::expr_use_visitor as euv;
use middle::mem_categorization as mc;
use middle::ty;
use syntax::ast;
use syntax::codemap::Span;
use util::ppaux::Repr;

use std::rc::Rc;

// FIXME (#16118): These functions are intended to allow the borrow checker to
// be less precise in its handling of Box while still allowing moves out of a
// Box. They should be removed when OwnedPtr is removed from LoanPath.

fn owned_ptr_base_path<'a>(loan_path: &'a LoanPath) -> &'a LoanPath {
    //! Returns the base of the leftmost dereference of an OwnedPtr in
    //! `loan_path`. If there is no dereference of an OwnedPtr in `loan_path`,
    //! then it just returns `loan_path` itself.

    return match owned_ptr_base_path_helper(loan_path) {
        Some(new_loan_path) => new_loan_path,
        None => loan_path.clone()
    };

    fn owned_ptr_base_path_helper<'a>(loan_path: &'a LoanPath) -> Option<&'a LoanPath> {
        match *loan_path {
            LpVar(_) | LpUpvar(_) => None,
            LpExtend(ref lp_base, _, LpDeref(mc::OwnedPtr)) => {
                match owned_ptr_base_path_helper(&**lp_base) {
                    v @ Some(_) => v,
                    None => Some(&**lp_base)
                }
            }
            LpExtend(ref lp_base, _, _) => owned_ptr_base_path_helper(&**lp_base)
        }
    }
}

fn owned_ptr_base_path_rc(loan_path: &Rc<LoanPath>) -> Rc<LoanPath> {
    //! The equivalent of `owned_ptr_base_path` for an &Rc<LoanPath> rather than
    //! a &LoanPath.

    return match owned_ptr_base_path_helper(loan_path) {
        Some(new_loan_path) => new_loan_path,
        None => loan_path.clone()
    };

    fn owned_ptr_base_path_helper(loan_path: &Rc<LoanPath>) -> Option<Rc<LoanPath>> {
        match **loan_path {
            LpVar(_) | LpUpvar(_) => None,
            LpExtend(ref lp_base, _, LpDeref(mc::OwnedPtr)) => {
                match owned_ptr_base_path_helper(lp_base) {
                    v @ Some(_) => v,
                    None => Some(lp_base.clone())
                }
            }
            LpExtend(ref lp_base, _, _) => owned_ptr_base_path_helper(lp_base)
        }
    }
}

struct CheckLoanCtxt<'a, 'tcx: 'a> {
    bccx: &'a BorrowckCtxt<'a, 'tcx>,
    dfcx_loans: &'a LoanDataFlow<'a, 'tcx>,
    move_data: move_data::FlowedMoveData<'a, 'tcx>,
    all_loans: &'a [Loan],
}

impl<'a, 'tcx> euv::Delegate for CheckLoanCtxt<'a, 'tcx> {
    fn consume(&mut self,
               consume_id: ast::NodeId,
               consume_span: Span,
               cmt: mc::cmt,
               mode: euv::ConsumeMode) {
        debug!("consume(consume_id={}, cmt={}, mode={})",
               consume_id, cmt.repr(self.tcx()), mode);

        self.consume_common(consume_id, consume_span, cmt, mode);
    }

    fn consume_pat(&mut self,
                   consume_pat: &ast::Pat,
                   cmt: mc::cmt,
                   mode: euv::ConsumeMode) {
        debug!("consume_pat(consume_pat={}, cmt={}, mode={})",
               consume_pat.repr(self.tcx()),
               cmt.repr(self.tcx()),
               mode);

        self.consume_common(consume_pat.id, consume_pat.span, cmt, mode);
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

        match opt_loan_path(&cmt) {
            Some(lp) => {
                let moved_value_use_kind = match loan_cause {
                    euv::ClosureCapture(_) => MovedInCapture,
                    _ => MovedInUse,
                };
                self.check_if_path_is_moved(borrow_id, borrow_span, moved_value_use_kind, &lp);
            }
            None => { }
        }

        self.check_for_conflicting_loans(borrow_id);
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
                match mode {
                    euv::Init | euv::JustWrite => {
                        // In a case like `path = 1`, then path does not
                        // have to be *FULLY* initialized, but we still
                        // must be careful lest it contains derefs of
                        // pointers.
                        self.check_if_assigned_path_is_moved(assignee_cmt.id,
                                                             assignment_span,
                                                             MovedInUse,
                                                             &lp);
                    }
                    euv::WriteAndRead => {
                        // In a case like `path += 1`, then path must be
                        // fully initialized, since we will read it before
                        // we write it.
                        self.check_if_path_is_moved(assignee_cmt.id,
                                                    assignment_span,
                                                    MovedInUse,
                                                    &lp);
                    }
                }
            }
            None => { }
        }

        self.check_assignment(assignment_id, assignment_span, assignee_cmt, mode);
    }

    fn decl_without_init(&mut self, _id: ast::NodeId, _span: Span) { }
}

pub fn check_loans<'a, 'b, 'c, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                                     dfcx_loans: &LoanDataFlow<'b, 'tcx>,
                                     move_data: move_data::FlowedMoveData<'c, 'tcx>,
                                     all_loans: &[Loan],
                                     decl: &ast::FnDecl,
                                     body: &ast::Block) {
    debug!("check_loans(body id={:?})", body.id);

    let mut clcx = CheckLoanCtxt {
        bccx: bccx,
        dfcx_loans: dfcx_loans,
        move_data: move_data,
        all_loans: all_loans,
    };

    {
        let mut euv = euv::ExprUseVisitor::new(&mut clcx, bccx.tcx);
        euv.walk_fn(decl, body);
    }
}

#[deriving(PartialEq)]
enum UseError {
    UseOk,
    UseWhileBorrowed(/*loan*/Rc<LoanPath>, /*loan*/Span)
}

fn compatible_borrow_kinds(borrow_kind1: ty::BorrowKind,
                           borrow_kind2: ty::BorrowKind)
                           -> bool {
    borrow_kind1 == ty::ImmBorrow && borrow_kind2 == ty::ImmBorrow
}

impl<'a, 'tcx> CheckLoanCtxt<'a, 'tcx> {
    pub fn tcx(&self) -> &'a ty::ctxt<'tcx> { self.bccx.tcx }

    pub fn each_issued_loan(&self, scope_id: ast::NodeId, op: |&Loan| -> bool)
                            -> bool {
        //! Iterates over each loan that has been issued
        //! on entrance to `scope_id`, regardless of whether it is
        //! actually *in scope* at that point.  Sometimes loans
        //! are issued for future scopes and thus they may have been
        //! *issued* but not yet be in effect.

        self.dfcx_loans.each_bit_on_entry(scope_id, |loan_index| {
            let loan = &self.all_loans[loan_index];
            op(loan)
        })
    }

    pub fn each_in_scope_loan(&self,
                              scope_id: ast::NodeId,
                              op: |&Loan| -> bool)
                              -> bool {
        //! Like `each_issued_loan()`, but only considers loans that are
        //! currently in scope.

        let tcx = self.tcx();
        self.each_issued_loan(scope_id, |loan| {
            if tcx.region_maps.is_subscope_of(scope_id, loan.kill_scope) {
                op(loan)
            } else {
                true
            }
        })
    }

    fn each_in_scope_loan_affecting_path(&self,
                                         scope_id: ast::NodeId,
                                         loan_path: &LoanPath,
                                         op: |&Loan| -> bool)
                                         -> bool {
        //! Iterates through all of the in-scope loans affecting `loan_path`,
        //! calling `op`, and ceasing iteration if `false` is returned.

        // First, we check for a loan restricting the path P being used. This
        // accounts for borrows of P but also borrows of subpaths, like P.a.b.
        // Consider the following example:
        //
        //     let x = &mut a.b.c; // Restricts a, a.b, and a.b.c
        //     let y = a;          // Conflicts with restriction

        let loan_path = owned_ptr_base_path(loan_path);
        let cont = self.each_in_scope_loan(scope_id, |loan| {
            let mut ret = true;
            for restr_path in loan.restricted_paths.iter() {
                if **restr_path == *loan_path {
                    if !op(loan) {
                        ret = false;
                        break;
                    }
                }
            }
            ret
        });

        if !cont {
            return false;
        }

        // Next, we must check for *loans* (not restrictions) on the path P or
        // any base path. This rejects examples like the following:
        //
        //     let x = &mut a.b;
        //     let y = a.b.c;
        //
        // Limiting this search to *loans* and not *restrictions* means that
        // examples like the following continue to work:
        //
        //     let x = &mut a.b;
        //     let y = a.c;

        let mut loan_path = loan_path;
        loop {
            match *loan_path {
                LpVar(_) | LpUpvar(_) => {
                    break;
                }
                LpExtend(ref lp_base, _, _) => {
                    loan_path = &**lp_base;
                }
            }

            let cont = self.each_in_scope_loan(scope_id, |loan| {
                if *loan.loan_path == *loan_path {
                    op(loan)
                } else {
                    true
                }
            });

            if !cont {
                return false;
            }
        }

        return true;
    }

    pub fn loans_generated_by(&self, scope_id: ast::NodeId) -> Vec<uint> {
        //! Returns a vector of the loans that are generated as
        //! we encounter `scope_id`.

        let mut result = Vec::new();
        self.dfcx_loans.each_gen_bit(scope_id, |loan_index| {
            result.push(loan_index);
            true
        });
        return result;
    }

    pub fn check_for_conflicting_loans(&self, scope_id: ast::NodeId) {
        //! Checks to see whether any of the loans that are issued
        //! by `scope_id` conflict with loans that have already been
        //! issued when we enter `scope_id` (for example, we do not
        //! permit two `&mut` borrows of the same variable).

        debug!("check_for_conflicting_loans(scope_id={:?})", scope_id);

        let new_loan_indices = self.loans_generated_by(scope_id);
        debug!("new_loan_indices = {:?}", new_loan_indices);

        self.each_issued_loan(scope_id, |issued_loan| {
            for &new_loan_index in new_loan_indices.iter() {
                let new_loan = &self.all_loans[new_loan_index];
                self.report_error_if_loans_conflict(issued_loan, new_loan);
            }
            true
        });

        for (i, &x) in new_loan_indices.iter().enumerate() {
            let old_loan = &self.all_loans[x];
            for &y in new_loan_indices.slice_from(i+1).iter() {
                let new_loan = &self.all_loans[y];
                self.report_error_if_loans_conflict(old_loan, new_loan);
            }
        }
    }

    pub fn report_error_if_loans_conflict(&self,
                                          old_loan: &Loan,
                                          new_loan: &Loan) {
        //! Checks whether `old_loan` and `new_loan` can safely be issued
        //! simultaneously.

        debug!("report_error_if_loans_conflict(old_loan={}, new_loan={})",
               old_loan.repr(self.tcx()),
               new_loan.repr(self.tcx()));

        // Should only be called for loans that are in scope at the same time.
        assert!(self.tcx().region_maps.scopes_intersect(old_loan.kill_scope,
                                                        new_loan.kill_scope));

        self.report_error_if_loan_conflicts_with_restriction(
            old_loan, new_loan, old_loan, new_loan) &&
        self.report_error_if_loan_conflicts_with_restriction(
            new_loan, old_loan, old_loan, new_loan);
    }

    pub fn report_error_if_loan_conflicts_with_restriction(&self,
                                                           loan1: &Loan,
                                                           loan2: &Loan,
                                                           old_loan: &Loan,
                                                           new_loan: &Loan)
                                                           -> bool {
        //! Checks whether the restrictions introduced by `loan1` would
        //! prohibit `loan2`. Returns false if an error is reported.

        debug!("report_error_if_loan_conflicts_with_restriction(\
                loan1={}, loan2={})",
               loan1.repr(self.tcx()),
               loan2.repr(self.tcx()));

        if compatible_borrow_kinds(loan1.kind, loan2.kind) {
            return true;
        }

        let loan2_base_path = owned_ptr_base_path_rc(&loan2.loan_path);
        for restr_path in loan1.restricted_paths.iter() {
            if *restr_path != loan2_base_path { continue; }

            let old_pronoun = if new_loan.loan_path == old_loan.loan_path {
                "it".to_string()
            } else {
                format!("`{}`",
                        self.bccx.loan_path_to_string(&*old_loan.loan_path))
            };

            match (new_loan.kind, old_loan.kind) {
                (ty::MutBorrow, ty::MutBorrow) => {
                    self.bccx.span_err(
                        new_loan.span,
                        format!("cannot borrow `{}` as mutable \
                                more than once at a time",
                                self.bccx.loan_path_to_string(
                                    &*new_loan.loan_path)).as_slice());
                }

                (ty::UniqueImmBorrow, _) => {
                    self.bccx.span_err(
                        new_loan.span,
                        format!("closure requires unique access to `{}` \
                                but {} is already borrowed",
                                self.bccx.loan_path_to_string(&*new_loan.loan_path),
                                old_pronoun).as_slice());
                }

                (_, ty::UniqueImmBorrow) => {
                    self.bccx.span_err(
                        new_loan.span,
                        format!("cannot borrow `{}` as {} because \
                                previous closure requires unique access",
                                self.bccx.loan_path_to_string(&*new_loan.loan_path),
                                new_loan.kind.to_user_str()).as_slice());
                }

                (_, _) => {
                    self.bccx.span_err(
                        new_loan.span,
                        format!("cannot borrow `{}` as {} because \
                                {} is also borrowed as {}",
                                self.bccx.loan_path_to_string(&*new_loan.loan_path),
                                new_loan.kind.to_user_str(),
                                old_pronoun,
                                old_loan.kind.to_user_str()).as_slice());
                }
            }

            match new_loan.cause {
                euv::ClosureCapture(span) => {
                    self.bccx.span_note(
                        span,
                        format!("borrow occurs due to use of `{}` in closure",
                                self.bccx.loan_path_to_string(
                                    &*new_loan.loan_path)).as_slice());
                }
                _ => { }
            }

            let rule_summary = match old_loan.kind {
                ty::MutBorrow => {
                    format!("the mutable borrow prevents subsequent \
                            moves, borrows, or modification of `{0}` \
                            until the borrow ends",
                            self.bccx.loan_path_to_string(
                                &*old_loan.loan_path))
                }

                ty::ImmBorrow => {
                    format!("the immutable borrow prevents subsequent \
                            moves or mutable borrows of `{0}` \
                            until the borrow ends",
                            self.bccx.loan_path_to_string(&*old_loan.loan_path))
                }

                ty::UniqueImmBorrow => {
                    format!("the unique capture prevents subsequent \
                            moves or borrows of `{0}` \
                            until the borrow ends",
                            self.bccx.loan_path_to_string(&*old_loan.loan_path))
                }
            };

            let borrow_summary = match old_loan.cause {
                euv::ClosureCapture(_) => {
                    format!("previous borrow of `{}` occurs here due to \
                            use in closure",
                            self.bccx.loan_path_to_string(&*old_loan.loan_path))
                }

                euv::OverloadedOperator(..) |
                euv::AddrOf(..) |
                euv::AutoRef(..) |
                euv::ClosureInvocation(..) |
                euv::ForLoop(..) |
                euv::RefBinding(..) => {
                    format!("previous borrow of `{}` occurs here",
                            self.bccx.loan_path_to_string(&*old_loan.loan_path))
                }
            };

            self.bccx.span_note(
                old_loan.span,
                format!("{}; {}", borrow_summary, rule_summary).as_slice());

            let old_loan_span = self.tcx().map.span(old_loan.kill_scope);
            self.bccx.span_end_note(old_loan_span,
                                    "previous borrow ends here");

            return false;
        }

        true
    }

    fn is_local_variable_or_arg(&self, cmt: mc::cmt) -> bool {
        match cmt.cat {
          mc::cat_local(_) => true,
          _ => false
        }
    }

    fn consume_common(&self,
                      id: ast::NodeId,
                      span: Span,
                      cmt: mc::cmt,
                      mode: euv::ConsumeMode) {
        match opt_loan_path(&cmt) {
            Some(lp) => {
                let moved_value_use_kind = match mode {
                    euv::Copy => {
                        self.check_for_copy_of_frozen_path(id, span, &*lp);
                        MovedInUse
                    }
                    euv::Move(_) => {
                        match self.move_data.kind_of_move_of_path(id, &lp) {
                            None => {
                                // Sometimes moves don't have a move kind;
                                // this either means that the original move
                                // was from something illegal to move,
                                // or was moved from referent of an unsafe
                                // pointer or something like that.
                                MovedInUse
                            }
                            Some(move_kind) => {
                                self.check_for_move_of_borrowed_path(id, span,
                                                                     &*lp, move_kind);
                                if move_kind == move_data::Captured {
                                    MovedInCapture
                                } else {
                                    MovedInUse
                                }
                            }
                        }
                    }
                };

                self.check_if_path_is_moved(id, span, moved_value_use_kind, &lp);
            }
            None => { }
        }
    }

    fn check_for_copy_of_frozen_path(&self,
                                     id: ast::NodeId,
                                     span: Span,
                                     copy_path: &LoanPath) {
        match self.analyze_restrictions_on_use(id, copy_path, ty::ImmBorrow) {
            UseOk => { }
            UseWhileBorrowed(loan_path, loan_span) => {
                self.bccx.span_err(
                    span,
                    format!("cannot use `{}` because it was mutably borrowed",
                            self.bccx.loan_path_to_string(copy_path).as_slice())
                    .as_slice());
                self.bccx.span_note(
                    loan_span,
                    format!("borrow of `{}` occurs here",
                            self.bccx.loan_path_to_string(&*loan_path).as_slice())
                    .as_slice());
            }
        }
    }

    fn check_for_move_of_borrowed_path(&self,
                                       id: ast::NodeId,
                                       span: Span,
                                       move_path: &LoanPath,
                                       move_kind: move_data::MoveKind) {
        // We want to detect if there are any loans at all, so we search for
        // any loans incompatible with MutBorrrow, since all other kinds of
        // loans are incompatible with that.
        match self.analyze_restrictions_on_use(id, move_path, ty::MutBorrow) {
            UseOk => { }
            UseWhileBorrowed(loan_path, loan_span) => {
                let err_message = match move_kind {
                    move_data::Captured =>
                        format!("cannot move `{}` into closure because it is borrowed",
                                self.bccx.loan_path_to_string(move_path).as_slice()),
                    move_data::Declared |
                    move_data::MoveExpr |
                    move_data::MovePat =>
                        format!("cannot move out of `{}` because it is borrowed",
                                self.bccx.loan_path_to_string(move_path).as_slice())
                };

                self.bccx.span_err(span, err_message.as_slice());
                self.bccx.span_note(
                    loan_span,
                    format!("borrow of `{}` occurs here",
                            self.bccx.loan_path_to_string(&*loan_path).as_slice())
                    .as_slice());
            }
        }
    }

    pub fn analyze_restrictions_on_use(&self,
                                       expr_id: ast::NodeId,
                                       use_path: &LoanPath,
                                       borrow_kind: ty::BorrowKind)
                                       -> UseError {
        debug!("analyze_restrictions_on_use(expr_id={:?}, use_path={})",
               self.tcx().map.node_to_string(expr_id),
               use_path.repr(self.tcx()));

        let mut ret = UseOk;

        self.each_in_scope_loan_affecting_path(expr_id, use_path, |loan| {
            if !compatible_borrow_kinds(loan.kind, borrow_kind) {
                ret = UseWhileBorrowed(loan.loan_path.clone(), loan.span);
                false
            } else {
                true
            }
        });

        return ret;
    }

    fn check_if_path_is_moved(&self,
                              id: ast::NodeId,
                              span: Span,
                              use_kind: MovedValueUseKind,
                              lp: &Rc<LoanPath>) {
        /*!
         * Reports an error if `expr` (which should be a path)
         * is using a moved/uninitialized value
         */

        debug!("check_if_path_is_moved(id={:?}, use_kind={:?}, lp={})",
               id, use_kind, lp.repr(self.bccx.tcx));
        let base_lp = owned_ptr_base_path_rc(lp);
        self.move_data.each_move_of(id, &base_lp, |move, moved_lp| {
            self.bccx.report_use_of_moved_value(
                span,
                use_kind,
                &**lp,
                move,
                moved_lp);
            false
        });
    }

    fn check_if_assigned_path_is_moved(&self,
                                       id: ast::NodeId,
                                       span: Span,
                                       use_kind: MovedValueUseKind,
                                       lp: &Rc<LoanPath>)
    {
        /*!
         * Reports an error if assigning to `lp` will use a
         * moved/uninitialized value. Mainly this is concerned with
         * detecting derefs of uninitialized pointers.
         *
         * For example:
         *
         *     let a: int;
         *     a = 10; // ok, even though a is uninitialized
         *
         *     struct Point { x: uint, y: uint }
         *     let p: Point;
         *     p.x = 22; // ok, even though `p` is uninitialized
         *
         *     let p: ~Point;
         *     (*p).x = 22; // not ok, p is uninitialized, can't deref
         */

        match **lp {
            LpVar(_) | LpUpvar(_) => {
                // assigning to `x` does not require that `x` is initialized
            }
            LpExtend(ref lp_base, _, LpInterior(_)) => {
                // assigning to `P.f` is ok if assigning to `P` is ok
                self.check_if_assigned_path_is_moved(id, span,
                                                     use_kind, lp_base);
            }
            LpExtend(ref lp_base, _, LpDeref(_)) => {
                // assigning to `(*P)` requires that `P` be initialized
                self.check_if_path_is_moved(id, span,
                                            use_kind, lp_base);
            }
        }
    }

    fn check_assignment(&self,
                        assignment_id: ast::NodeId,
                        assignment_span: Span,
                        assignee_cmt: mc::cmt,
                        mode: euv::MutateMode) {
        debug!("check_assignment(assignee_cmt={})", assignee_cmt.repr(self.tcx()));

        // Mutable values can be assigned, as long as they obey loans
        // and aliasing restrictions:
        if assignee_cmt.mutbl.is_mutable() {
            if check_for_aliasable_mutable_writes(self, assignment_span, assignee_cmt.clone()) {
                if mode != euv::Init {
                    check_for_assignment_to_borrowed_path(
                        self, assignment_id, assignment_span, assignee_cmt.clone());
                    mark_variable_as_used_mut(self, assignee_cmt);
                }
            }
            return;
        }

        // Initializations are OK.
        if mode == euv::Init {
            return
        }

        // For immutable local variables, assignments are legal
        // if they cannot already have been assigned
        if self.is_local_variable_or_arg(assignee_cmt.clone()) {
            assert!(assignee_cmt.mutbl.is_immutable()); // no "const" locals
            let lp = opt_loan_path(&assignee_cmt).unwrap();
            self.move_data.each_assignment_of(assignment_id, &lp, |assign| {
                self.bccx.report_reassigned_immutable_variable(
                    assignment_span,
                    &*lp,
                    assign);
                false
            });
            return;
        }

        // Otherwise, just a plain error.
        match opt_loan_path(&assignee_cmt) {
            Some(lp) => {
                self.bccx.span_err(
                    assignment_span,
                    format!("cannot assign to {} {} `{}`",
                            assignee_cmt.mutbl.to_user_str(),
                            self.bccx.cmt_to_string(&*assignee_cmt),
                            self.bccx.loan_path_to_string(&*lp)).as_slice());
            }
            None => {
                self.bccx.span_err(
                    assignment_span,
                    format!("cannot assign to {} {}",
                            assignee_cmt.mutbl.to_user_str(),
                            self.bccx.cmt_to_string(&*assignee_cmt)).as_slice());
            }
        }
        return;

        fn mark_variable_as_used_mut(this: &CheckLoanCtxt,
                                     mut cmt: mc::cmt) {
            //! If the mutability of the `cmt` being written is inherited
            //! from a local variable, liveness will
            //! not have been able to detect that this variable's mutability
            //! is important, so we must add the variable to the
            //! `used_mut_nodes` table here.

            loop {
                debug!("mark_variable_as_used_mut(cmt={})", cmt.repr(this.tcx()));
                match cmt.cat.clone() {
                    mc::cat_copied_upvar(mc::CopiedUpvar { upvar_id: id, .. }) |
                    mc::cat_local(id) => {
                        this.tcx().used_mut_nodes.borrow_mut().insert(id);
                        return;
                    }

                    mc::cat_upvar(..) => {
                        return;
                    }

                    mc::cat_deref(_, _, mc::GcPtr) => {
                        assert_eq!(cmt.mutbl, mc::McImmutable);
                        return;
                    }

                    mc::cat_rvalue(..) |
                    mc::cat_static_item |
                    mc::cat_deref(_, _, mc::UnsafePtr(..)) |
                    mc::cat_deref(_, _, mc::BorrowedPtr(..)) |
                    mc::cat_deref(_, _, mc::Implicit(..)) => {
                        assert_eq!(cmt.mutbl, mc::McDeclared);
                        return;
                    }

                    mc::cat_discr(b, _) |
                    mc::cat_deref(b, _, mc::OwnedPtr) => {
                        assert_eq!(cmt.mutbl, mc::McInherited);
                        cmt = b;
                    }

                    mc::cat_downcast(b) |
                    mc::cat_interior(b, _) => {
                        assert_eq!(cmt.mutbl, mc::McInherited);
                        cmt = b;
                    }
                }
            }
        }

        fn check_for_aliasable_mutable_writes(this: &CheckLoanCtxt,
                                              span: Span,
                                              cmt: mc::cmt) -> bool {
            //! Safety checks related to writes to aliasable, mutable locations

            let guarantor = cmt.guarantor();
            debug!("check_for_aliasable_mutable_writes(cmt={}, guarantor={})",
                   cmt.repr(this.tcx()), guarantor.repr(this.tcx()));
            match guarantor.cat {
                mc::cat_deref(ref b, _, mc::BorrowedPtr(ty::MutBorrow, _)) => {
                    // Statically prohibit writes to `&mut` when aliasable

                    check_for_aliasability_violation(this, span, b.clone());
                }

                _ => {}
            }

            return true; // no errors reported
        }

        fn check_for_aliasability_violation(this: &CheckLoanCtxt,
                                            span: Span,
                                            cmt: mc::cmt)
                                            -> bool {
            match cmt.freely_aliasable(this.tcx()) {
                None => {
                    return true;
                }
                Some(mc::AliasableStaticMut(..)) => {
                    return true;
                }
                Some(cause) => {
                    this.bccx.report_aliasability_violation(
                        span,
                        MutabilityViolation,
                        cause);
                    return false;
                }
            }
        }

        fn check_for_assignment_to_borrowed_path(
            this: &CheckLoanCtxt,
            assignment_id: ast::NodeId,
            assignment_span: Span,
            assignee_cmt: mc::cmt)
        {
            //! Check for assignments that violate the terms of an
            //! outstanding loan.

            let loan_path = match opt_loan_path(&assignee_cmt) {
                Some(lp) => lp,
                None => { return; /* no loan path, can't be any loans */ }
            };

            this.each_in_scope_loan_affecting_path(assignment_id, &*loan_path, |loan| {
                this.report_illegal_mutation(assignment_span, &*loan_path, loan);
                false
            });
        }
    }

    pub fn report_illegal_mutation(&self,
                                   span: Span,
                                   loan_path: &LoanPath,
                                   loan: &Loan) {
        self.bccx.span_err(
            span,
            format!("cannot assign to `{}` because it is borrowed",
                    self.bccx.loan_path_to_string(loan_path)).as_slice());
        self.bccx.span_note(
            loan.span,
            format!("borrow of `{}` occurs here",
                    self.bccx.loan_path_to_string(loan_path)).as_slice());
    }
}

