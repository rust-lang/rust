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
use self::UseError::*;

use borrowck::*;
use borrowck::InteriorKind::{InteriorElement, InteriorField};
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::infer;
use rustc::middle::mem_categorization as mc;
use rustc::middle::mem_categorization::Categorization;
use rustc::middle::region;
use rustc::middle::ty;
use syntax::ast;
use syntax::codemap::Span;
use rustc_front::hir;

use std::rc::Rc;

// FIXME (#16118): These functions are intended to allow the borrow checker to
// be less precise in its handling of Box while still allowing moves out of a
// Box. They should be removed when Unique is removed from LoanPath.

fn owned_ptr_base_path<'a, 'tcx>(loan_path: &'a LoanPath<'tcx>) -> &'a LoanPath<'tcx> {
    //! Returns the base of the leftmost dereference of an Unique in
    //! `loan_path`. If there is no dereference of an Unique in `loan_path`,
    //! then it just returns `loan_path` itself.

    return match helper(loan_path) {
        Some(new_loan_path) => new_loan_path,
        None => loan_path.clone()
    };

    fn helper<'a, 'tcx>(loan_path: &'a LoanPath<'tcx>) -> Option<&'a LoanPath<'tcx>> {
        match loan_path.kind {
            LpVar(_) | LpUpvar(_) => None,
            LpExtend(ref lp_base, _, LpDeref(mc::Unique)) => {
                match helper(&**lp_base) {
                    v @ Some(_) => v,
                    None => Some(&**lp_base)
                }
            }
            LpDowncast(ref lp_base, _) |
            LpExtend(ref lp_base, _, _) => helper(&**lp_base)
        }
    }
}

fn owned_ptr_base_path_rc<'tcx>(loan_path: &Rc<LoanPath<'tcx>>) -> Rc<LoanPath<'tcx>> {
    //! The equivalent of `owned_ptr_base_path` for an &Rc<LoanPath> rather than
    //! a &LoanPath.

    return match helper(loan_path) {
        Some(new_loan_path) => new_loan_path,
        None => loan_path.clone()
    };

    fn helper<'tcx>(loan_path: &Rc<LoanPath<'tcx>>) -> Option<Rc<LoanPath<'tcx>>> {
        match loan_path.kind {
            LpVar(_) | LpUpvar(_) => None,
            LpExtend(ref lp_base, _, LpDeref(mc::Unique)) => {
                match helper(lp_base) {
                    v @ Some(_) => v,
                    None => Some(lp_base.clone())
                }
            }
            LpDowncast(ref lp_base, _) |
            LpExtend(ref lp_base, _, _) => helper(lp_base)
        }
    }
}

struct CheckLoanCtxt<'a, 'tcx: 'a> {
    bccx: &'a BorrowckCtxt<'a, 'tcx>,
    dfcx_loans: &'a LoanDataFlow<'a, 'tcx>,
    move_data: &'a move_data::FlowedMoveData<'a, 'tcx>,
    all_loans: &'a [Loan<'tcx>],
    param_env: &'a ty::ParameterEnvironment<'a, 'tcx>,
}

impl<'a, 'tcx> euv::Delegate<'tcx> for CheckLoanCtxt<'a, 'tcx> {
    fn consume(&mut self,
               consume_id: ast::NodeId,
               consume_span: Span,
               cmt: mc::cmt<'tcx>,
               mode: euv::ConsumeMode) {
        debug!("consume(consume_id={}, cmt={:?}, mode={:?})",
               consume_id, cmt, mode);

        self.consume_common(consume_id, consume_span, cmt, mode);
    }

    fn matched_pat(&mut self,
                   _matched_pat: &hir::Pat,
                   _cmt: mc::cmt,
                   _mode: euv::MatchMode) { }

    fn consume_pat(&mut self,
                   consume_pat: &hir::Pat,
                   cmt: mc::cmt<'tcx>,
                   mode: euv::ConsumeMode) {
        debug!("consume_pat(consume_pat={:?}, cmt={:?}, mode={:?})",
               consume_pat,
               cmt,
               mode);

        self.consume_common(consume_pat.id, consume_pat.span, cmt, mode);
    }

    fn borrow(&mut self,
              borrow_id: ast::NodeId,
              borrow_span: Span,
              cmt: mc::cmt<'tcx>,
              loan_region: ty::Region,
              bk: ty::BorrowKind,
              loan_cause: euv::LoanCause)
    {
        debug!("borrow(borrow_id={}, cmt={:?}, loan_region={:?}, \
               bk={:?}, loan_cause={:?})",
               borrow_id, cmt, loan_region,
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
              assignee_cmt: mc::cmt<'tcx>,
              mode: euv::MutateMode)
    {
        debug!("mutate(assignment_id={}, assignee_cmt={:?})",
               assignment_id, assignee_cmt);

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

        self.check_assignment(assignment_id, assignment_span, assignee_cmt);
    }

    fn decl_without_init(&mut self, _id: ast::NodeId, _span: Span) { }
}

pub fn check_loans<'a, 'b, 'c, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                                     dfcx_loans: &LoanDataFlow<'b, 'tcx>,
                                     move_data: &move_data::FlowedMoveData<'c, 'tcx>,
                                     all_loans: &[Loan<'tcx>],
                                     fn_id: ast::NodeId,
                                     decl: &hir::FnDecl,
                                     body: &hir::Block) {
    debug!("check_loans(body id={})", body.id);

    let param_env = ty::ParameterEnvironment::for_item(bccx.tcx, fn_id);
    let infcx = infer::new_infer_ctxt(bccx.tcx, &bccx.tcx.tables, Some(param_env), false);

    let mut clcx = CheckLoanCtxt {
        bccx: bccx,
        dfcx_loans: dfcx_loans,
        move_data: move_data,
        all_loans: all_loans,
        param_env: &infcx.parameter_environment
    };

    {
        let mut euv = euv::ExprUseVisitor::new(&mut clcx, &infcx);
        euv.walk_fn(decl, body);
    }
}

#[derive(PartialEq)]
enum UseError<'tcx> {
    UseOk,
    UseWhileBorrowed(/*loan*/Rc<LoanPath<'tcx>>, /*loan*/Span)
}

fn compatible_borrow_kinds(borrow_kind1: ty::BorrowKind,
                           borrow_kind2: ty::BorrowKind)
                           -> bool {
    borrow_kind1 == ty::ImmBorrow && borrow_kind2 == ty::ImmBorrow
}

impl<'a, 'tcx> CheckLoanCtxt<'a, 'tcx> {
    pub fn tcx(&self) -> &'a ty::ctxt<'tcx> { self.bccx.tcx }

    pub fn each_issued_loan<F>(&self, node: ast::NodeId, mut op: F) -> bool where
        F: FnMut(&Loan<'tcx>) -> bool,
    {
        //! Iterates over each loan that has been issued
        //! on entrance to `node`, regardless of whether it is
        //! actually *in scope* at that point.  Sometimes loans
        //! are issued for future scopes and thus they may have been
        //! *issued* but not yet be in effect.

        self.dfcx_loans.each_bit_on_entry(node, |loan_index| {
            let loan = &self.all_loans[loan_index];
            op(loan)
        })
    }

    pub fn each_in_scope_loan<F>(&self, scope: region::CodeExtent, mut op: F) -> bool where
        F: FnMut(&Loan<'tcx>) -> bool,
    {
        //! Like `each_issued_loan()`, but only considers loans that are
        //! currently in scope.

        let tcx = self.tcx();
        self.each_issued_loan(scope.node_id(&tcx.region_maps), |loan| {
            if tcx.region_maps.is_subscope_of(scope, loan.kill_scope) {
                op(loan)
            } else {
                true
            }
        })
    }

    fn each_in_scope_loan_affecting_path<F>(&self,
                                            scope: region::CodeExtent,
                                            loan_path: &LoanPath<'tcx>,
                                            mut op: F)
                                            -> bool where
        F: FnMut(&Loan<'tcx>) -> bool,
    {
        //! Iterates through all of the in-scope loans affecting `loan_path`,
        //! calling `op`, and ceasing iteration if `false` is returned.

        // First, we check for a loan restricting the path P being used. This
        // accounts for borrows of P but also borrows of subpaths, like P.a.b.
        // Consider the following example:
        //
        //     let x = &mut a.b.c; // Restricts a, a.b, and a.b.c
        //     let y = a;          // Conflicts with restriction

        let loan_path = owned_ptr_base_path(loan_path);
        let cont = self.each_in_scope_loan(scope, |loan| {
            let mut ret = true;
            for restr_path in &loan.restricted_paths {
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
            match loan_path.kind {
                LpVar(_) | LpUpvar(_) => {
                    break;
                }
                LpDowncast(ref lp_base, _) |
                LpExtend(ref lp_base, _, _) => {
                    loan_path = &**lp_base;
                }
            }

            let cont = self.each_in_scope_loan(scope, |loan| {
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

    pub fn loans_generated_by(&self, node: ast::NodeId) -> Vec<usize> {
        //! Returns a vector of the loans that are generated as
        //! we enter `node`.

        let mut result = Vec::new();
        self.dfcx_loans.each_gen_bit(node, |loan_index| {
            result.push(loan_index);
            true
        });
        return result;
    }

    pub fn check_for_conflicting_loans(&self, node: ast::NodeId) {
        //! Checks to see whether any of the loans that are issued
        //! on entrance to `node` conflict with loans that have already been
        //! issued when we enter `node` (for example, we do not
        //! permit two `&mut` borrows of the same variable).
        //!
        //! (Note that some loans can be *issued* without necessarily
        //! taking effect yet.)

        debug!("check_for_conflicting_loans(node={:?})", node);

        let new_loan_indices = self.loans_generated_by(node);
        debug!("new_loan_indices = {:?}", new_loan_indices);

        for &new_loan_index in &new_loan_indices {
            self.each_issued_loan(node, |issued_loan| {
                let new_loan = &self.all_loans[new_loan_index];
                // Only report an error for the first issued loan that conflicts
                // to avoid O(n^2) errors.
                self.report_error_if_loans_conflict(issued_loan, new_loan)
            });
        }

        for (i, &x) in new_loan_indices.iter().enumerate() {
            let old_loan = &self.all_loans[x];
            for &y in &new_loan_indices[(i+1) ..] {
                let new_loan = &self.all_loans[y];
                self.report_error_if_loans_conflict(old_loan, new_loan);
            }
        }
    }

    pub fn report_error_if_loans_conflict(&self,
                                          old_loan: &Loan<'tcx>,
                                          new_loan: &Loan<'tcx>)
                                          -> bool {
        //! Checks whether `old_loan` and `new_loan` can safely be issued
        //! simultaneously.

        debug!("report_error_if_loans_conflict(old_loan={:?}, new_loan={:?})",
               old_loan,
               new_loan);

        // Should only be called for loans that are in scope at the same time.
        assert!(self.tcx().region_maps.scopes_intersect(old_loan.kill_scope,
                                                        new_loan.kill_scope));

        self.report_error_if_loan_conflicts_with_restriction(
            old_loan, new_loan, old_loan, new_loan) &&
        self.report_error_if_loan_conflicts_with_restriction(
            new_loan, old_loan, old_loan, new_loan)
    }

    pub fn report_error_if_loan_conflicts_with_restriction(&self,
                                                           loan1: &Loan<'tcx>,
                                                           loan2: &Loan<'tcx>,
                                                           old_loan: &Loan<'tcx>,
                                                           new_loan: &Loan<'tcx>)
                                                           -> bool {
        //! Checks whether the restrictions introduced by `loan1` would
        //! prohibit `loan2`. Returns false if an error is reported.

        debug!("report_error_if_loan_conflicts_with_restriction(\
                loan1={:?}, loan2={:?})",
               loan1,
               loan2);

        if compatible_borrow_kinds(loan1.kind, loan2.kind) {
            return true;
        }

        let loan2_base_path = owned_ptr_base_path_rc(&loan2.loan_path);
        for restr_path in &loan1.restricted_paths {
            if *restr_path != loan2_base_path { continue; }

            // If new_loan is something like `x.a`, and old_loan is something like `x.b`, we would
            // normally generate a rather confusing message (in this case, for multiple mutable
            // borrows):
            //
            //     error: cannot borrow `x.b` as mutable more than once at a time
            //     note: previous borrow of `x.a` occurs here; the mutable borrow prevents
            //     subsequent moves, borrows, or modification of `x.a` until the borrow ends
            //
            // What we want to do instead is get the 'common ancestor' of the two borrow paths and
            // use that for most of the message instead, giving is something like this:
            //
            //     error: cannot borrow `x` as mutable more than once at a time
            //     note: previous borrow of `x` occurs here (through borrowing `x.a`); the mutable
            //     borrow prevents subsequent moves, borrows, or modification of `x` until the
            //     borrow ends

            let common = new_loan.loan_path.common(&*old_loan.loan_path);
            let (nl, ol, new_loan_msg, old_loan_msg) =
                if new_loan.loan_path.has_fork(&*old_loan.loan_path) && common.is_some() {
                    let nl = self.bccx.loan_path_to_string(&common.unwrap());
                    let ol = nl.clone();
                    let new_loan_msg = format!(" (here through borrowing `{}`)",
                                               self.bccx.loan_path_to_string(
                                                   &*new_loan.loan_path));
                    let old_loan_msg = format!(" (through borrowing `{}`)",
                                               self.bccx.loan_path_to_string(
                                                   &*old_loan.loan_path));
                    (nl, ol, new_loan_msg, old_loan_msg)
                } else {
                    (self.bccx.loan_path_to_string(&*new_loan.loan_path),
                     self.bccx.loan_path_to_string(&*old_loan.loan_path),
                     String::new(), String::new())
                };

            let ol_pronoun = if new_loan.loan_path == old_loan.loan_path {
                "it".to_string()
            } else {
                format!("`{}`", ol)
            };

            match (new_loan.kind, old_loan.kind) {
                (ty::MutBorrow, ty::MutBorrow) => {
                    span_err!(self.bccx, new_loan.span, E0499,
                              "cannot borrow `{}`{} as mutable \
                               more than once at a time",
                              nl, new_loan_msg);
                }

                (ty::UniqueImmBorrow, _) => {
                    span_err!(self.bccx, new_loan.span, E0500,
                              "closure requires unique access to `{}` \
                               but {} is already borrowed{}",
                              nl, ol_pronoun, old_loan_msg);
                }

                (_, ty::UniqueImmBorrow) => {
                    span_err!(self.bccx, new_loan.span, E0501,
                              "cannot borrow `{}`{} as {} because \
                               previous closure requires unique access",
                              nl, new_loan_msg, new_loan.kind.to_user_str());
                }

                (_, _) => {
                    span_err!(self.bccx, new_loan.span, E0502,
                              "cannot borrow `{}`{} as {} because \
                               {} is also borrowed as {}{}",
                              nl,
                              new_loan_msg,
                              new_loan.kind.to_user_str(),
                              ol_pronoun,
                              old_loan.kind.to_user_str(),
                              old_loan_msg);
                }
            }

            match new_loan.cause {
                euv::ClosureCapture(span) => {
                    self.bccx.span_note(
                        span,
                        &format!("borrow occurs due to use of `{}` in closure",
                                nl));
                }
                _ => { }
            }

            let rule_summary = match old_loan.kind {
                ty::MutBorrow => {
                    format!("the mutable borrow prevents subsequent \
                            moves, borrows, or modification of `{0}` \
                            until the borrow ends",
                            ol)
                }

                ty::ImmBorrow => {
                    format!("the immutable borrow prevents subsequent \
                            moves or mutable borrows of `{0}` \
                            until the borrow ends",
                            ol)
                }

                ty::UniqueImmBorrow => {
                    format!("the unique capture prevents subsequent \
                            moves or borrows of `{0}` \
                            until the borrow ends",
                            ol)
                }
            };

            let borrow_summary = match old_loan.cause {
                euv::ClosureCapture(_) => {
                    format!("previous borrow of `{}` occurs here{} due to \
                            use in closure",
                            ol, old_loan_msg)
                }

                euv::OverloadedOperator(..) |
                euv::AddrOf(..) |
                euv::AutoRef(..) |
                euv::AutoUnsafe(..) |
                euv::ClosureInvocation(..) |
                euv::ForLoop(..) |
                euv::RefBinding(..) |
                euv::MatchDiscriminant(..) => {
                    format!("previous borrow of `{}` occurs here{}",
                            ol, old_loan_msg)
                }
            };

            self.bccx.span_note(
                old_loan.span,
                &format!("{}; {}", borrow_summary, rule_summary));

            let old_loan_span = self.tcx().map.span(
                old_loan.kill_scope.node_id(&self.tcx().region_maps));
            self.bccx.span_end_note(old_loan_span,
                                    "previous borrow ends here");

            return false;
        }

        true
    }

    fn consume_common(&self,
                      id: ast::NodeId,
                      span: Span,
                      cmt: mc::cmt<'tcx>,
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
                                     copy_path: &LoanPath<'tcx>) {
        match self.analyze_restrictions_on_use(id, copy_path, ty::ImmBorrow) {
            UseOk => { }
            UseWhileBorrowed(loan_path, loan_span) => {
                span_err!(self.bccx, span, E0503,
                          "cannot use `{}` because it was mutably borrowed",
                          &self.bccx.loan_path_to_string(copy_path));
                self.bccx.span_note(
                    loan_span,
                    &format!("borrow of `{}` occurs here",
                            &self.bccx.loan_path_to_string(&*loan_path))
                    );
            }
        }
    }

    fn check_for_move_of_borrowed_path(&self,
                                       id: ast::NodeId,
                                       span: Span,
                                       move_path: &LoanPath<'tcx>,
                                       move_kind: move_data::MoveKind) {
        // We want to detect if there are any loans at all, so we search for
        // any loans incompatible with MutBorrrow, since all other kinds of
        // loans are incompatible with that.
        match self.analyze_restrictions_on_use(id, move_path, ty::MutBorrow) {
            UseOk => { }
            UseWhileBorrowed(loan_path, loan_span) => {
                match move_kind {
                    move_data::Captured =>
                        span_err!(self.bccx, span, E0504,
                                  "cannot move `{}` into closure because it is borrowed",
                                  &self.bccx.loan_path_to_string(move_path)),
                    move_data::Declared |
                    move_data::MoveExpr |
                    move_data::MovePat =>
                        span_err!(self.bccx, span, E0505,
                                  "cannot move out of `{}` because it is borrowed",
                                  &self.bccx.loan_path_to_string(move_path))
                };

                self.bccx.span_note(
                    loan_span,
                    &format!("borrow of `{}` occurs here",
                            &self.bccx.loan_path_to_string(&*loan_path))
                    );
            }
        }
    }

    pub fn analyze_restrictions_on_use(&self,
                                       expr_id: ast::NodeId,
                                       use_path: &LoanPath<'tcx>,
                                       borrow_kind: ty::BorrowKind)
                                       -> UseError<'tcx> {
        debug!("analyze_restrictions_on_use(expr_id={}, use_path={:?})",
               self.tcx().map.node_to_string(expr_id),
               use_path);

        let mut ret = UseOk;

        self.each_in_scope_loan_affecting_path(
            self.tcx().region_maps.node_extent(expr_id), use_path, |loan| {
            if !compatible_borrow_kinds(loan.kind, borrow_kind) {
                ret = UseWhileBorrowed(loan.loan_path.clone(), loan.span);
                false
            } else {
                true
            }
        });

        return ret;
    }

    /// Reports an error if `expr` (which should be a path)
    /// is using a moved/uninitialized value
    fn check_if_path_is_moved(&self,
                              id: ast::NodeId,
                              span: Span,
                              use_kind: MovedValueUseKind,
                              lp: &Rc<LoanPath<'tcx>>) {
        debug!("check_if_path_is_moved(id={}, use_kind={:?}, lp={:?})",
               id, use_kind, lp);

        // FIXME (22079): if you find yourself tempted to cut and paste
        // the body below and then specializing the error reporting,
        // consider refactoring this instead!

        let base_lp = owned_ptr_base_path_rc(lp);
        self.move_data.each_move_of(id, &base_lp, |the_move, moved_lp| {
            self.bccx.report_use_of_moved_value(
                span,
                use_kind,
                &**lp,
                the_move,
                moved_lp,
                self.param_env);
            false
        });
    }

    /// Reports an error if assigning to `lp` will use a
    /// moved/uninitialized value. Mainly this is concerned with
    /// detecting derefs of uninitialized pointers.
    ///
    /// For example:
    ///
    /// ```
    /// let a: int;
    /// a = 10; // ok, even though a is uninitialized
    ///
    /// struct Point { x: usize, y: usize }
    /// let p: Point;
    /// p.x = 22; // ok, even though `p` is uninitialized
    ///
    /// let p: Box<Point>;
    /// (*p).x = 22; // not ok, p is uninitialized, can't deref
    /// ```
    fn check_if_assigned_path_is_moved(&self,
                                       id: ast::NodeId,
                                       span: Span,
                                       use_kind: MovedValueUseKind,
                                       lp: &Rc<LoanPath<'tcx>>)
    {
        match lp.kind {
            LpVar(_) | LpUpvar(_) => {
                // assigning to `x` does not require that `x` is initialized
            }
            LpDowncast(ref lp_base, _) => {
                // assigning to `(P->Variant).f` is ok if assigning to `P` is ok
                self.check_if_assigned_path_is_moved(id, span,
                                                     use_kind, lp_base);
            }
            LpExtend(ref lp_base, _, LpInterior(InteriorField(_))) => {
                match lp_base.to_type().sty {
                    ty::TyStruct(def, _) | ty::TyEnum(def, _) if def.has_dtor() => {
                        // In the case where the owner implements drop, then
                        // the path must be initialized to prevent a case of
                        // partial reinitialization
                        //
                        // FIXME (22079): could refactor via hypothetical
                        // generalized check_if_path_is_moved
                        let loan_path = owned_ptr_base_path_rc(lp_base);
                        self.move_data.each_move_of(id, &loan_path, |_, _| {
                            self.bccx
                                .report_partial_reinitialization_of_uninitialized_structure(
                                    span,
                                    &*loan_path);
                            false
                        });
                        return;
                    },
                    _ => {},
                }

                // assigning to `P.f` is ok if assigning to `P` is ok
                self.check_if_assigned_path_is_moved(id, span,
                                                     use_kind, lp_base);
            }
            LpExtend(ref lp_base, _, LpInterior(InteriorElement(..))) |
            LpExtend(ref lp_base, _, LpDeref(_)) => {
                // assigning to `P[i]` requires `P` is initialized
                // assigning to `(*P)` requires `P` is initialized
                self.check_if_path_is_moved(id, span, use_kind, lp_base);
            }
        }
    }

    fn check_assignment(&self,
                        assignment_id: ast::NodeId,
                        assignment_span: Span,
                        assignee_cmt: mc::cmt<'tcx>) {
        debug!("check_assignment(assignee_cmt={:?})", assignee_cmt);

        // Check that we don't invalidate any outstanding loans
        if let Some(loan_path) = opt_loan_path(&assignee_cmt) {
            let scope = self.tcx().region_maps.node_extent(assignment_id);
            self.each_in_scope_loan_affecting_path(scope, &*loan_path, |loan| {
                self.report_illegal_mutation(assignment_span, &*loan_path, loan);
                false
            });
        }

        // Check for reassignments to (immutable) local variables. This
        // needs to be done here instead of in check_loans because we
        // depend on move data.
        if let Categorization::Local(local_id) = assignee_cmt.cat {
            let lp = opt_loan_path(&assignee_cmt).unwrap();
            self.move_data.each_assignment_of(assignment_id, &lp, |assign| {
                if assignee_cmt.mutbl.is_mutable() {
                    self.tcx().used_mut_nodes.borrow_mut().insert(local_id);
                } else {
                    self.bccx.report_reassigned_immutable_variable(
                        assignment_span,
                        &*lp,
                        assign);
                }
                false
            });
            return
        }
    }

    pub fn report_illegal_mutation(&self,
                                   span: Span,
                                   loan_path: &LoanPath<'tcx>,
                                   loan: &Loan) {
        span_err!(self.bccx, span, E0506,
                  "cannot assign to `{}` because it is borrowed",
                  self.bccx.loan_path_to_string(loan_path));
        self.bccx.span_note(
            loan.span,
            &format!("borrow of `{}` occurs here",
                    self.bccx.loan_path_to_string(loan_path)));
    }
}
