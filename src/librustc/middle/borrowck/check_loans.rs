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


use mc = middle::mem_categorization;
use middle::borrowck::*;
use middle::moves;
use middle::ty;
use middle::typeck::MethodCall;
use syntax::ast;
use syntax::ast_util;
use syntax::codemap::Span;
use syntax::visit::Visitor;
use syntax::visit;
use util::ppaux::Repr;

struct CheckLoanCtxt<'a> {
    bccx: &'a BorrowckCtxt<'a>,
    dfcx_loans: &'a LoanDataFlow<'a>,
    move_data: move_data::FlowedMoveData<'a>,
    all_loans: &'a [Loan],
}

impl<'a> Visitor<()> for CheckLoanCtxt<'a> {

    fn visit_expr(&mut self, ex: &ast::Expr, _: ()) {
        check_loans_in_expr(self, ex);
    }
    fn visit_local(&mut self, l: &ast::Local, _: ()) {
        check_loans_in_local(self, l);
    }
    fn visit_block(&mut self, b: &ast::Block, _: ()) {
        check_loans_in_block(self, b);
    }
    fn visit_pat(&mut self, p: &ast::Pat, _: ()) {
        check_loans_in_pat(self, p);
    }
    fn visit_fn(&mut self, _fk: &visit::FnKind, _fd: &ast::FnDecl,
                _b: &ast::Block, _s: Span, _n: ast::NodeId, _: ()) {
        // Don't process nested items or closures here,
        // the outer loop will take care of it.
        return;
    }

    // FIXME(#10894) should continue recursing
    fn visit_ty(&mut self, _t: &ast::Ty, _: ()) {}
}

pub fn check_loans(bccx: &BorrowckCtxt,
                   dfcx_loans: &LoanDataFlow,
                   move_data: move_data::FlowedMoveData,
                   all_loans: &[Loan],
                   body: &ast::Block) {
    debug!("check_loans(body id={:?})", body.id);

    let mut clcx = CheckLoanCtxt {
        bccx: bccx,
        dfcx_loans: dfcx_loans,
        move_data: move_data,
        all_loans: all_loans,
    };

    clcx.visit_block(body, ());
}

#[deriving(Eq)]
enum MoveError {
    MoveOk,
    MoveWhileBorrowed(/*loan*/@LoanPath, /*loan*/Span)
}

impl<'a> CheckLoanCtxt<'a> {
    pub fn tcx(&self) -> &'a ty::ctxt { self.bccx.tcx }

    pub fn each_issued_loan(&self, scope_id: ast::NodeId, op: |&Loan| -> bool)
                            -> bool {
        //! Iterates over each loan that has been issued
        //! on entrance to `scope_id`, regardless of whether it is
        //! actually *in scope* at that point.  Sometimes loans
        //! are issued for future scopes and thus they may have been
        //! *issued* but not yet be in effect.

        self.dfcx_loans.each_bit_on_entry_frozen(scope_id, |loan_index| {
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

    pub fn each_in_scope_restriction(&self,
                                     scope_id: ast::NodeId,
                                     loan_path: @LoanPath,
                                     op: |&Loan, &Restriction| -> bool)
                                     -> bool {
        //! Iterates through all the in-scope restrictions for the
        //! given `loan_path`

        self.each_in_scope_loan(scope_id, |loan| {
            debug!("each_in_scope_restriction found loan: {:?}",
                   loan.repr(self.tcx()));

            let mut ret = true;
            for restr in loan.restrictions.iter() {
                if restr.loan_path == loan_path {
                    if !op(loan, restr) {
                        ret = false;
                        break;
                    }
                }
            }
            ret
        })
    }

    pub fn loans_generated_by(&self, scope_id: ast::NodeId) -> Vec<uint> {
        //! Returns a vector of the loans that are generated as
        //! we encounter `scope_id`.

        let mut result = Vec::new();
        self.dfcx_loans.each_gen_bit_frozen(scope_id, |loan_index| {
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

        // Restrictions that would cause the new loan to be illegal:
        let illegal_if = match loan2.kind {
            // Look for restrictions against mutation. These are
            // generated by all other borrows.
            ty::MutBorrow => RESTR_MUTATE,

            // Look for restrictions against freezing (immutable borrows).
            // These are generated by `&mut` borrows.
            ty::ImmBorrow => RESTR_FREEZE,

            // No matter how the data is borrowed (as `&`, as `&mut`,
            // or as `&unique imm`) it will always generate a
            // restriction against mutating the data. So look for those.
            ty::UniqueImmBorrow => RESTR_MUTATE,
        };
        debug!("illegal_if={:?}", illegal_if);

        for restr in loan1.restrictions.iter() {
            if !restr.set.intersects(illegal_if) { continue; }
            if restr.loan_path != loan2.loan_path { continue; }

            let old_pronoun = if new_loan.loan_path == old_loan.loan_path {
                ~"it"
            } else {
                format!("`{}`",
                        self.bccx.loan_path_to_str(old_loan.loan_path))
            };

            match (new_loan.kind, old_loan.kind) {
                (ty::MutBorrow, ty::MutBorrow) => {
                    self.bccx.span_err(
                        new_loan.span,
                        format!("cannot borrow `{}` as mutable \
                                more than once at a time",
                                self.bccx.loan_path_to_str(new_loan.loan_path)));
                }

                (ty::UniqueImmBorrow, _) => {
                    self.bccx.span_err(
                        new_loan.span,
                        format!("closure requires unique access to `{}` \
                                but {} is already borrowed",
                                self.bccx.loan_path_to_str(new_loan.loan_path),
                                old_pronoun));
                }

                (_, ty::UniqueImmBorrow) => {
                    self.bccx.span_err(
                        new_loan.span,
                        format!("cannot borrow `{}` as {} because \
                                previous closure requires unique access",
                                self.bccx.loan_path_to_str(new_loan.loan_path),
                                new_loan.kind.to_user_str()));
                }

                (_, _) => {
                    self.bccx.span_err(
                        new_loan.span,
                        format!("cannot borrow `{}` as {} because \
                                {} is also borrowed as {}",
                                self.bccx.loan_path_to_str(new_loan.loan_path),
                                new_loan.kind.to_user_str(),
                                old_pronoun,
                                old_loan.kind.to_user_str()));
                }
            }

            match new_loan.cause {
                ClosureCapture(span) => {
                    self.bccx.span_note(
                        span,
                        format!("borrow occurs due to use of `{}` in closure",
                                self.bccx.loan_path_to_str(new_loan.loan_path)));
                }
                _ => { }
            }

            let rule_summary = match old_loan.kind {
                ty::MutBorrow => {
                    format!("the mutable borrow prevents subsequent \
                            moves, borrows, or modification of `{0}` \
                            until the borrow ends",
                            self.bccx.loan_path_to_str(old_loan.loan_path))
                }

                ty::ImmBorrow => {
                    format!("the immutable borrow prevents subsequent \
                            moves or mutable borrows of `{0}` \
                            until the borrow ends",
                            self.bccx.loan_path_to_str(old_loan.loan_path))
                }

                ty::UniqueImmBorrow => {
                    format!("the unique capture prevents subsequent \
                            moves or borrows of `{0}` \
                            until the borrow ends",
                            self.bccx.loan_path_to_str(old_loan.loan_path))
                }
            };

            let borrow_summary = match old_loan.cause {
                ClosureCapture(_) => {
                    format!("previous borrow of `{}` occurs here due to \
                            use in closure",
                            self.bccx.loan_path_to_str(old_loan.loan_path))
                }

                AddrOf | AutoRef | RefBinding => {
                    format!("previous borrow of `{}` occurs here",
                            self.bccx.loan_path_to_str(old_loan.loan_path))
                }
            };

            self.bccx.span_note(
                old_loan.span,
                format!("{}; {}", borrow_summary, rule_summary));

            let old_loan_span = self.tcx().map.span(old_loan.kill_scope);
            self.bccx.span_end_note(old_loan_span,
                                    "previous borrow ends here");

            return false;
        }

        true
    }

    pub fn is_local_variable(&self, cmt: mc::cmt) -> bool {
        match cmt.cat {
          mc::cat_local(_) => true,
          _ => false
        }
    }

    pub fn check_if_path_is_moved(&self,
                                  id: ast::NodeId,
                                  span: Span,
                                  use_kind: MovedValueUseKind,
                                  lp: @LoanPath) {
        /*!
         * Reports an error if `expr` (which should be a path)
         * is using a moved/uninitialized value
         */

        debug!("check_if_path_is_moved(id={:?}, use_kind={:?}, lp={})",
               id, use_kind, lp.repr(self.bccx.tcx));
        self.move_data.each_move_of(id, lp, |move, moved_lp| {
            self.bccx.report_use_of_moved_value(
                span,
                use_kind,
                lp,
                move,
                moved_lp);
            false
        });
    }

    pub fn check_assignment(&self, expr: &ast::Expr) {
        // We don't use cat_expr() here because we don't want to treat
        // auto-ref'd parameters in overloaded operators as rvalues.
        let adj = {
            let adjustments = self.bccx.tcx.adjustments.borrow();
            adjustments.get().find_copy(&expr.id)
        };
        let cmt = match adj {
            None => self.bccx.cat_expr_unadjusted(expr),
            Some(adj) => self.bccx.cat_expr_autoderefd(expr, adj)
        };

        debug!("check_assignment(cmt={})", cmt.repr(self.tcx()));

        // Mutable values can be assigned, as long as they obey loans
        // and aliasing restrictions:
        if cmt.mutbl.is_mutable() {
            if check_for_aliasable_mutable_writes(self, expr, cmt) {
                if check_for_assignment_to_restricted_or_frozen_location(
                    self, expr, cmt)
                {
                    // Safe, but record for lint pass later:
                    mark_variable_as_used_mut(self, cmt);
                }
            }
            return;
        }

        // For immutable local variables, assignments are legal
        // if they cannot already have been assigned
        if self.is_local_variable(cmt) {
            assert!(cmt.mutbl.is_immutable()); // no "const" locals
            let lp = opt_loan_path(cmt).unwrap();
            self.move_data.each_assignment_of(expr.id, lp, |assign| {
                self.bccx.report_reassigned_immutable_variable(
                    expr.span,
                    lp,
                    assign);
                false
            });
            return;
        }

        // Otherwise, just a plain error.
        match opt_loan_path(cmt) {
            Some(lp) => {
                self.bccx.span_err(
                    expr.span,
                    format!("cannot assign to {} {} `{}`",
                            cmt.mutbl.to_user_str(),
                            self.bccx.cmt_to_str(cmt),
                            self.bccx.loan_path_to_str(lp)));
            }
            None => {
                self.bccx.span_err(
                    expr.span,
                    format!("cannot assign to {} {}",
                            cmt.mutbl.to_user_str(),
                            self.bccx.cmt_to_str(cmt)));
            }
        }
        return;

        fn mark_variable_as_used_mut(this: &CheckLoanCtxt,
                                     cmt: mc::cmt) {
            //! If the mutability of the `cmt` being written is inherited
            //! from a local variable, liveness will
            //! not have been able to detect that this variable's mutability
            //! is important, so we must add the variable to the
            //! `used_mut_nodes` table here.

            let mut cmt = cmt;
            loop {
                debug!("mark_writes_through_upvars_as_used_mut(cmt={})",
                       cmt.repr(this.tcx()));
                match cmt.cat {
                    mc::cat_local(id) | mc::cat_arg(id) => {
                        let mut used_mut_nodes = this.tcx()
                                                     .used_mut_nodes
                                                     .borrow_mut();
                        used_mut_nodes.get().insert(id);
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
                    mc::cat_copied_upvar(..) |
                    mc::cat_deref(_, _, mc::UnsafePtr(..)) |
                    mc::cat_deref(_, _, mc::BorrowedPtr(..)) => {
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
                                              expr: &ast::Expr,
                                              cmt: mc::cmt) -> bool {
            //! Safety checks related to writes to aliasable, mutable locations

            let guarantor = cmt.guarantor();
            debug!("check_for_aliasable_mutable_writes(cmt={}, guarantor={})",
                   cmt.repr(this.tcx()), guarantor.repr(this.tcx()));
            match guarantor.cat {
                mc::cat_deref(b, _, mc::BorrowedPtr(ty::MutBorrow, _)) => {
                    // Statically prohibit writes to `&mut` when aliasable

                    check_for_aliasability_violation(this, expr, b);
                }

                _ => {}
            }

            return true; // no errors reported
        }

        fn check_for_aliasability_violation(this: &CheckLoanCtxt,
                                            expr: &ast::Expr,
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
                        expr.span,
                        MutabilityViolation,
                        cause);
                    return false;
                }
            }
        }

        fn check_for_assignment_to_restricted_or_frozen_location(
            this: &CheckLoanCtxt,
            expr: &ast::Expr,
            cmt: mc::cmt) -> bool
        {
            //! Check for assignments that violate the terms of an
            //! outstanding loan.

            let loan_path = match opt_loan_path(cmt) {
                Some(lp) => lp,
                None => { return true; /* no loan path, can't be any loans */ }
            };

            // Start by searching for an assignment to a *restricted*
            // location. Here is one example of the kind of error caught
            // by this check:
            //
            //    let mut v = ~[1, 2, 3];
            //    let p = &v;
            //    v = ~[4];
            //
            // In this case, creating `p` triggers a RESTR_MUTATE
            // restriction on the path `v`.
            //
            // Here is a second, more subtle example:
            //
            //    let mut v = ~[1, 2, 3];
            //    let p = &const v[0];
            //    v[0] = 4;                   // OK
            //    v[1] = 5;                   // OK
            //    v = ~[4, 5, 3];             // Error
            //
            // In this case, `p` is pointing to `v[0]`, and it is a
            // `const` pointer in any case. So the first two
            // assignments are legal (and would be permitted by this
            // check). However, the final assignment (which is
            // logically equivalent) is forbidden, because it would
            // cause the existing `v` array to be freed, thus
            // invalidating `p`. In the code, this error results
            // because `gather_loans::restrictions` adds a
            // `RESTR_MUTATE` restriction whenever the contents of an
            // owned pointer are borrowed, and hence while `v[*]` is not
            // restricted from being written, `v` is.
            let cont = this.each_in_scope_restriction(expr.id,
                                                      loan_path,
                                                      |loan, restr| {
                if restr.set.intersects(RESTR_MUTATE) {
                    this.report_illegal_mutation(expr, loan_path, loan);
                    false
                } else {
                    true
                }
            });

            if !cont { return false }

            // The previous code handled assignments to paths that
            // have been restricted. This covers paths that have been
            // directly lent out and their base paths, but does not
            // cover random extensions of those paths. For example,
            // the following program is not declared illegal by the
            // previous check:
            //
            //    let mut v = ~[1, 2, 3];
            //    let p = &v;
            //    v[0] = 4; // declared error by loop below, not code above
            //
            // The reason that this passes the previous check whereas
            // an assignment like `v = ~[4]` fails is because the assignment
            // here is to `v[*]`, and the existing restrictions were issued
            // for `v`, not `v[*]`.
            //
            // So in this loop, we walk back up the loan path so long
            // as the mutability of the path is dependent on a super
            // path, and check that the super path was not lent out as
            // mutable or immutable (a const loan is ok).
            //
            // Mutability of a path can be dependent on the super path
            // in two ways. First, it might be inherited mutability.
            // Second, the pointee of an `&mut` pointer can only be
            // mutated if it is found in an unaliased location, so we
            // have to check that the owner location is not borrowed.
            //
            // Note that we are *not* checking for any and all
            // restrictions.  We are only interested in the pointers
            // that the user created, whereas we add restrictions for
            // all kinds of paths that are not directly aliased. If we checked
            // for all restrictions, and not just loans, then the following
            // valid program would be considered illegal:
            //
            //    let mut v = ~[1, 2, 3];
            //    let p = &const v[0];
            //    v[1] = 5; // ok
            //
            // Here the restriction that `v` not be mutated would be misapplied
            // to block the subpath `v[1]`.
            let full_loan_path = loan_path;
            let mut loan_path = loan_path;
            loop {
                match *loan_path {
                    // Peel back one layer if, for `loan_path` to be
                    // mutable, `lp_base` must be mutable. This occurs
                    // with inherited mutability and with `&mut`
                    // pointers.
                    LpExtend(lp_base, mc::McInherited, _) |
                    LpExtend(lp_base, _, LpDeref(mc::BorrowedPtr(ty::MutBorrow, _))) => {
                        loan_path = lp_base;
                    }

                    // Otherwise stop iterating
                    LpExtend(_, mc::McDeclared, _) |
                    LpExtend(_, mc::McImmutable, _) |
                    LpVar(_) => {
                        return true;
                    }
                }

                // Check for a non-const loan of `loan_path`
                let cont = this.each_in_scope_loan(expr.id, |loan| {
                    if loan.loan_path == loan_path {
                        this.report_illegal_mutation(expr, full_loan_path, loan);
                        false
                    } else {
                        true
                    }
                });

                if !cont { return false }
            }
        }
    }

    pub fn report_illegal_mutation(&self,
                                   expr: &ast::Expr,
                                   loan_path: &LoanPath,
                                   loan: &Loan) {
        self.bccx.span_err(
            expr.span,
            format!("cannot assign to `{}` because it is borrowed",
                 self.bccx.loan_path_to_str(loan_path)));
        self.bccx.span_note(
            loan.span,
            format!("borrow of `{}` occurs here",
                 self.bccx.loan_path_to_str(loan_path)));
    }

    fn check_move_out_from_expr(&self, expr: &ast::Expr) {
        match expr.node {
            ast::ExprFnBlock(..) | ast::ExprProc(..) => {
                // Moves due to captures are checked in
                // check_captured_variables() because it allows
                // us to give a more precise error message with
                // a more precise span.
            }
            _ => {
                self.check_move_out_from_id(expr.id, expr.span)
            }
        }
    }

    fn check_move_out_from_id(&self, id: ast::NodeId, span: Span) {
        self.move_data.each_path_moved_by(id, |_, move_path| {
            match self.analyze_move_out_from(id, move_path) {
                MoveOk => {}
                MoveWhileBorrowed(loan_path, loan_span) => {
                    self.bccx.span_err(
                        span,
                        format!("cannot move out of `{}` \
                                because it is borrowed",
                             self.bccx.loan_path_to_str(move_path)));
                    self.bccx.span_note(
                        loan_span,
                        format!("borrow of `{}` occurs here",
                                self.bccx.loan_path_to_str(loan_path)));
                }
            }
            true
        });
    }

    fn check_captured_variables(&self,
                                closure_id: ast::NodeId,
                                span: Span) {
        for cap_var in self.bccx.capture_map.get(&closure_id).deref().iter() {
            let var_id = ast_util::def_id_of_def(cap_var.def).node;
            let var_path = @LpVar(var_id);
            self.check_if_path_is_moved(closure_id, span,
                                        MovedInCapture, var_path);
            match cap_var.mode {
                moves::CapRef | moves::CapCopy => {}
                moves::CapMove => {
                    check_by_move_capture(self, closure_id, cap_var, var_path);
                }
            }
        }
        return;

        fn check_by_move_capture(this: &CheckLoanCtxt,
                                 closure_id: ast::NodeId,
                                 cap_var: &moves::CaptureVar,
                                 move_path: @LoanPath) {
            let move_err = this.analyze_move_out_from(closure_id, move_path);
            match move_err {
                MoveOk => {}
                MoveWhileBorrowed(loan_path, loan_span) => {
                    this.bccx.span_err(
                        cap_var.span,
                        format!("cannot move `{}` into closure \
                                because it is borrowed",
                                this.bccx.loan_path_to_str(move_path)));
                    this.bccx.span_note(
                        loan_span,
                        format!("borrow of `{}` occurs here",
                                this.bccx.loan_path_to_str(loan_path)));
                }
            }
        }
    }

    pub fn analyze_move_out_from(&self,
                                 expr_id: ast::NodeId,
                                 mut move_path: @LoanPath)
                                 -> MoveError {
        debug!("analyze_move_out_from(expr_id={:?}, move_path={})",
               self.tcx().map.node_to_str(expr_id),
               move_path.repr(self.tcx()));

        // We must check every element of a move path. See
        // `borrowck-move-subcomponent.rs` for a test case.
        loop {
            // check for a conflicting loan:
            let mut ret = MoveOk;
            self.each_in_scope_restriction(expr_id, move_path, |loan, _| {
                // Any restriction prevents moves.
                ret = MoveWhileBorrowed(loan.loan_path, loan.span);
                false
            });

            if ret != MoveOk {
                return ret
            }

            match *move_path {
                LpVar(_) => return MoveOk,
                LpExtend(subpath, _, _) => move_path = subpath,
            }
        }
    }

    pub fn check_call(&self,
                      _expr: &ast::Expr,
                      _callee: Option<@ast::Expr>,
                      _callee_span: Span,
                      _args: &[@ast::Expr]) {
        // NB: This call to check for conflicting loans is not truly
        // necessary, because the callee_id never issues new loans.
        // However, I added it for consistency and lest the system
        // should change in the future.
        //
        // FIXME(#6268) nested method calls
        // self.check_for_conflicting_loans(callee_id);
    }
}

fn check_loans_in_local<'a>(this: &mut CheckLoanCtxt<'a>,
                            local: &ast::Local) {
    visit::walk_local(this, local, ());
}

fn check_loans_in_expr<'a>(this: &mut CheckLoanCtxt<'a>,
                           expr: &ast::Expr) {
    visit::walk_expr(this, expr, ());

    debug!("check_loans_in_expr(expr={})",
           expr.repr(this.tcx()));

    this.check_for_conflicting_loans(expr.id);
    this.check_move_out_from_expr(expr);

    let method_map = this.bccx.method_map.borrow();
    match expr.node {
      ast::ExprPath(..) => {
          if !this.move_data.is_assignee(expr.id) {
              let cmt = this.bccx.cat_expr_unadjusted(expr);
              debug!("path cmt={}", cmt.repr(this.tcx()));
              let r = opt_loan_path(cmt);
              for &lp in r.iter() {
                  this.check_if_path_is_moved(expr.id, expr.span, MovedInUse, lp);
              }
          }
      }
      ast::ExprFnBlock(..) | ast::ExprProc(..) => {
          this.check_captured_variables(expr.id, expr.span)
      }
      ast::ExprAssign(dest, _) |
      ast::ExprAssignOp(_, dest, _) => {
        this.check_assignment(dest);
      }
      ast::ExprCall(f, ref args) => {
        this.check_call(expr, Some(f), f.span, args.as_slice());
      }
      ast::ExprMethodCall(_, _, ref args) => {
        this.check_call(expr, None, expr.span, args.as_slice());
      }
      ast::ExprIndex(_, rval) | ast::ExprBinary(_, _, rval)
      if method_map.get().contains_key(&MethodCall::expr(expr.id)) => {
        this.check_call(expr, None, expr.span, [rval]);
      }
      ast::ExprUnary(_, _) | ast::ExprIndex(_, _)
      if method_map.get().contains_key(&MethodCall::expr(expr.id)) => {
        this.check_call(expr, None, expr.span, []);
      }
      ast::ExprInlineAsm(ref ia) => {
          for &(_, out) in ia.outputs.iter() {
              this.check_assignment(out);
          }
      }
      _ => {}
    }
}

fn check_loans_in_pat<'a>(this: &mut CheckLoanCtxt<'a>,
                          pat: &ast::Pat)
{
    this.check_for_conflicting_loans(pat.id);
    this.check_move_out_from_id(pat.id, pat.span);
    visit::walk_pat(this, pat, ());
}

fn check_loans_in_block<'a>(this: &mut CheckLoanCtxt<'a>,
                            blk: &ast::Block)
{
    visit::walk_block(this, blk, ());
    this.check_for_conflicting_loans(blk.id);
}
