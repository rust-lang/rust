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
// Checking loans
//
// Phase 2 of check: we walk down the tree and check that:
// 1. assignments are always made to mutable locations;
// 2. loans made in overlapping scopes do not conflict
// 3. assignments do not affect things loaned out as immutable
// 4. moves do not affect things loaned out in any way

use middle::moves;
use middle::borrowck::*;
use mc = middle::mem_categorization;
use middle::ty;
use util::ppaux::Repr;
use core::hashmap::HashSet;
use syntax::ast::{m_mutbl, m_imm, m_const};
use syntax::ast;
use syntax::ast_util;
use syntax::visit;
use syntax::codemap::span;

struct CheckLoanCtxt<'self> {
    bccx: @BorrowckCtxt,
    dfcx: &'self LoanDataFlow,
    all_loans: &'self [Loan],
    reported: @mut HashSet<ast::node_id>,
}

pub fn check_loans(bccx: @BorrowckCtxt,
                   dfcx: &LoanDataFlow,
                   all_loans: &[Loan],
                   body: &ast::blk) {
    debug!("check_loans(body id=%?)", body.node.id);

    let clcx = @mut CheckLoanCtxt {
        bccx: bccx,
        dfcx: dfcx,
        all_loans: all_loans,
        reported: @mut HashSet::new(),
    };

    let vt = visit::mk_vt(@visit::Visitor {visit_expr: check_loans_in_expr,
                                           visit_local: check_loans_in_local,
                                           visit_block: check_loans_in_block,
                                           visit_pat: check_loans_in_pat,
                                           visit_fn: check_loans_in_fn,
                                           .. *visit::default_visitor()});
    (vt.visit_block)(body, clcx, vt);
}

enum MoveError {
    MoveOk,
    MoveFromIllegalCmt(mc::cmt),
    MoveWhileBorrowed(/*loan*/@LoanPath, /*loan*/span)
}

pub impl<'self> CheckLoanCtxt<'self> {
    fn tcx(&self) -> ty::ctxt { self.bccx.tcx }

    #[cfg(stage0)]
    fn each_issued_loan(&self,
                        scope_id: ast::node_id,
                        op: &fn(&Loan) -> bool)
    {
        //! Iterates over each loan that that has been issued
        //! on entrance to `scope_id`, regardless of whether it is
        //! actually *in scope* at that point.  Sometimes loans
        //! are issued for future scopes and thus they may have been
        //! *issued* but not yet be in effect.

        for self.dfcx.each_bit_on_entry(scope_id) |loan_index| {
            let loan = &self.all_loans[loan_index];
            if !op(loan) {
                return;
            }
        }
    }
    #[cfg(not(stage0))]
    fn each_issued_loan(&self,
                        scope_id: ast::node_id,
                        op: &fn(&Loan) -> bool) -> bool
    {
        //! Iterates over each loan that that has been issued
        //! on entrance to `scope_id`, regardless of whether it is
        //! actually *in scope* at that point.  Sometimes loans
        //! are issued for future scopes and thus they may have been
        //! *issued* but not yet be in effect.

        for self.dfcx.each_bit_on_entry(scope_id) |loan_index| {
            let loan = &self.all_loans[loan_index];
            if !op(loan) {
                return false;
            }
        }
        return true;
    }

    #[cfg(stage0)]
    fn each_in_scope_loan(&self,
                          scope_id: ast::node_id,
                          op: &fn(&Loan) -> bool)
    {
        //! Like `each_issued_loan()`, but only considers loans that are
        //! currently in scope.

        let region_maps = self.tcx().region_maps;
        for self.each_issued_loan(scope_id) |loan| {
            if region_maps.is_subscope_of(scope_id, loan.kill_scope) {
                if !op(loan) {
                    return;
                }
            }
        }
    }
    #[cfg(not(stage0))]
    fn each_in_scope_loan(&self,
                          scope_id: ast::node_id,
                          op: &fn(&Loan) -> bool) -> bool
    {
        //! Like `each_issued_loan()`, but only considers loans that are
        //! currently in scope.

        let region_maps = self.tcx().region_maps;
        for self.each_issued_loan(scope_id) |loan| {
            if region_maps.is_subscope_of(scope_id, loan.kill_scope) {
                if !op(loan) {
                    return false;
                }
            }
        }
        return true;
    }

    #[cfg(stage0)]
    fn each_in_scope_restriction(&self,
                                 scope_id: ast::node_id,
                                 loan_path: @LoanPath,
                                 op: &fn(&Loan, &Restriction) -> bool)
    {
        //! Iterates through all the in-scope restrictions for the
        //! given `loan_path`

        for self.each_in_scope_loan(scope_id) |loan| {
            for loan.restrictions.each |restr| {
                if restr.loan_path == loan_path {
                    if !op(loan, restr) {
                        return;
                    }
                }
            }
        }
    }
    #[cfg(not(stage0))]
    fn each_in_scope_restriction(&self,
                                 scope_id: ast::node_id,
                                 loan_path: @LoanPath,
                                 op: &fn(&Loan, &Restriction) -> bool) -> bool
    {
        //! Iterates through all the in-scope restrictions for the
        //! given `loan_path`

        for self.each_in_scope_loan(scope_id) |loan| {
            for loan.restrictions.each |restr| {
                if restr.loan_path == loan_path {
                    if !op(loan, restr) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    fn loans_generated_by(&self, scope_id: ast::node_id) -> ~[uint] {
        //! Returns a vector of the loans that are generated as
        //! we encounter `scope_id`.

        let mut result = ~[];
        for self.dfcx.each_gen_bit(scope_id) |loan_index| {
            result.push(loan_index);
        }
        return result;
    }

    fn check_for_conflicting_loans(&mut self, scope_id: ast::node_id) {
        //! Checks to see whether any of the loans that are issued
        //! by `scope_id` conflict with loans that have already been
        //! issued when we enter `scope_id` (for example, we do not
        //! permit two `&mut` borrows of the same variable).

        debug!("check_for_conflicting_loans(scope_id=%?)", scope_id);

        let new_loan_indices = self.loans_generated_by(scope_id);
        debug!("new_loan_indices = %?", new_loan_indices);

        for self.each_issued_loan(scope_id) |issued_loan| {
            for new_loan_indices.each |&new_loan_index| {
                let new_loan = &self.all_loans[new_loan_index];
                self.report_error_if_loans_conflict(issued_loan, new_loan);
            }
        }

        for uint::range(0, new_loan_indices.len()) |i| {
            let old_loan = &self.all_loans[new_loan_indices[i]];
            for uint::range(i+1, new_loan_indices.len()) |j| {
                let new_loan = &self.all_loans[new_loan_indices[j]];
                self.report_error_if_loans_conflict(old_loan, new_loan);
            }
        }
    }

    fn report_error_if_loans_conflict(&self,
                                      old_loan: &Loan,
                                      new_loan: &Loan) {
        //! Checks whether `old_loan` and `new_loan` can safely be issued
        //! simultaneously.

        debug!("report_error_if_loans_conflict(old_loan=%s, new_loan=%s)",
               old_loan.repr(self.tcx()),
               new_loan.repr(self.tcx()));

        // Should only be called for loans that are in scope at the same time.
        let region_maps = self.tcx().region_maps;
        assert!(region_maps.scopes_intersect(old_loan.kill_scope,
                                             new_loan.kill_scope));

        self.report_error_if_loan_conflicts_with_restriction(
            old_loan, new_loan, old_loan, new_loan) &&
        self.report_error_if_loan_conflicts_with_restriction(
            new_loan, old_loan, old_loan, new_loan);
    }

    fn report_error_if_loan_conflicts_with_restriction(&self,
                                                       loan1: &Loan,
                                                       loan2: &Loan,
                                                       old_loan: &Loan,
                                                       new_loan: &Loan) -> bool {
        //! Checks whether the restrictions introduced by `loan1` would
        //! prohibit `loan2`. Returns false if an error is reported.

        debug!("report_error_if_loan_conflicts_with_restriction(\
                loan1=%s, loan2=%s)",
               loan1.repr(self.tcx()),
               loan2.repr(self.tcx()));

        // Restrictions that would cause the new loan to be immutable:
        let illegal_if = match loan2.mutbl {
            m_mutbl => RESTR_ALIAS | RESTR_FREEZE | RESTR_MUTATE,
            m_imm =>   RESTR_ALIAS | RESTR_FREEZE,
            m_const => RESTR_ALIAS,
        };
        debug!("illegal_if=%?", illegal_if);

        for loan1.restrictions.each |restr| {
            if !restr.set.intersects(illegal_if) { loop; }
            if restr.loan_path != loan2.loan_path { loop; }

            match (new_loan.mutbl, old_loan.mutbl) {
                (m_mutbl, m_mutbl) => {
                    self.bccx.span_err(
                        new_loan.span,
                        fmt!("cannot borrow `%s` as mutable \
                              more than once at at a time",
                             self.bccx.loan_path_to_str(new_loan.loan_path)));
                    self.bccx.span_note(
                        old_loan.span,
                        fmt!("second borrow of `%s` as mutable occurs here",
                             self.bccx.loan_path_to_str(new_loan.loan_path)));
                    return false;
                }

                _ => {
                    self.bccx.span_err(
                        new_loan.span,
                        fmt!("cannot borrow `%s` as %s because \
                              it is also borrowed as %s"
                             self.bccx.loan_path_to_str(new_loan.loan_path),
                             self.bccx.mut_to_str(new_loan.mutbl),
                             self.bccx.mut_to_str(old_loan.mutbl)));
                    self.bccx.span_note(
                        old_loan.span,
                        fmt!("second borrow of `%s` occurs here",
                             self.bccx.loan_path_to_str(new_loan.loan_path)));
                    return false;
                }
            }
        }

        true
    }

    fn is_local_variable(&self, cmt: mc::cmt) -> bool {
        match cmt.cat {
          mc::cat_local(_) => true,
          _ => false
        }
    }

    fn check_assignment(&self, expr: @ast::expr) {
        // We don't use cat_expr() here because we don't want to treat
        // auto-ref'd parameters in overloaded operators as rvalues.
        let cmt = match self.bccx.tcx.adjustments.find(&expr.id) {
            None => self.bccx.cat_expr_unadjusted(expr),
            Some(&adj) => self.bccx.cat_expr_autoderefd(expr, adj)
        };

        debug!("check_assignment(cmt=%s)", cmt.repr(self.tcx()));

        // check that the value being assigned is declared as mutable
        // and report an error otherwise.
        match cmt.mutbl {
            mc::McDeclared => {
                // OK, but we have to mark arguments as requiring mut
                // if they are assigned (other cases are handled by liveness,
                // since we need to distinguish local variables assigned
                // once vs those assigned multiple times)
                match cmt.cat {
                    mc::cat_self(*) |
                    mc::cat_arg(*) => {
                        mark_variable_as_used_mut(self, cmt);
                    }
                    _ => {}
                }
            }
            mc::McInherited => {
                // OK, but we may have to add an entry to `used_mut_nodes`
                mark_variable_as_used_mut(self, cmt);
            }
            mc::McReadOnly | mc::McImmutable => {
                // Subtle: liveness guarantees that immutable local
                // variables are only assigned once, so no need to
                // report an error for an assignment to a local
                // variable (note also that it is not legal to borrow
                // for a local variable before it has been assigned
                // for the first time).
                if !self.is_local_variable(cmt) {
                    self.bccx.span_err(
                        expr.span,
                        fmt!("cannot assign to %s %s"
                             cmt.mutbl.to_user_str(),
                             self.bccx.cmt_to_str(cmt)));
                }
                return;
            }
        }

        if check_for_aliasable_mutable_writes(self, expr, cmt) {
            check_for_assignment_to_restricted_or_frozen_location(
                self, expr, cmt);
        }

        fn mark_variable_as_used_mut(this: &CheckLoanCtxt,
                                     cmt: mc::cmt) {
            //! If the mutability of the `cmt` being written is inherited
            //! from a local variable, liveness will
            //! not have been able to detect that this variable's mutability
            //! is important, so we must add the variable to the
            //! `used_mut_nodes` table here.

            let mut cmt = cmt;
            loop {
                debug!("mark_writes_through_upvars_as_used_mut(cmt=%s)",
                       cmt.repr(this.tcx()));
                match cmt.cat {
                    mc::cat_local(id) |
                    mc::cat_arg(id) |
                    mc::cat_self(id) => {
                        this.tcx().used_mut_nodes.insert(id);
                        return;
                    }

                    mc::cat_stack_upvar(b) => {
                        cmt = b;
                    }

                    mc::cat_rvalue |
                    mc::cat_static_item |
                    mc::cat_implicit_self |
                    mc::cat_copied_upvar(*) |
                    mc::cat_deref(_, _, mc::unsafe_ptr(*)) |
                    mc::cat_deref(_, _, mc::gc_ptr(*)) |
                    mc::cat_deref(_, _, mc::region_ptr(*)) => {
                        assert_eq!(cmt.mutbl, mc::McDeclared);
                        return;
                    }

                    mc::cat_discr(b, _) |
                    mc::cat_deref(b, _, mc::uniq_ptr(*)) => {
                        assert_eq!(cmt.mutbl, mc::McInherited);
                        cmt = b;
                    }

                    mc::cat_interior(b, _) => {
                        if cmt.mutbl == mc::McInherited {
                            cmt = b;
                        } else {
                            return; // field declared as mutable or some such
                        }
                    }
                }
            }
        }

        fn check_for_aliasable_mutable_writes(this: &CheckLoanCtxt,
                                              expr: @ast::expr,
                                              cmt: mc::cmt) -> bool {
            //! Safety checks related to writes to aliasable, mutable locations

            let guarantor = cmt.guarantor();
            debug!("check_for_aliasable_mutable_writes(cmt=%s, guarantor=%s)",
                   cmt.repr(this.tcx()), guarantor.repr(this.tcx()));
            match guarantor.cat {
                mc::cat_deref(b, _, mc::region_ptr(m_mutbl, _)) => {
                    // Statically prohibit writes to `&mut` when aliasable

                    match b.freely_aliasable() {
                        None => {}
                        Some(cause) => {
                            this.bccx.report_aliasability_violation(
                                expr.span,
                                MutabilityViolation,
                                cause);
                        }
                    }
                }

                mc::cat_deref(_, deref_count, mc::gc_ptr(ast::m_mutbl)) => {
                    // Dynamically check writes to `@mut`

                    let key = root_map_key {
                        id: guarantor.id,
                        derefs: deref_count
                    };
                    debug!("Inserting write guard at %?", key);
                    this.bccx.write_guard_map.insert(key);
                }

                _ => {}
            }

            return true; // no errors reported
        }

        fn check_for_assignment_to_restricted_or_frozen_location(
            this: &CheckLoanCtxt,
            expr: @ast::expr,
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
            for this.each_in_scope_restriction(expr.id, loan_path)
                |loan, restr|
            {
                if restr.set.intersects(RESTR_MUTATE) {
                    this.report_illegal_mutation(expr, loan_path, loan);
                    return false;
                }
            }

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
                    // Peel back one layer if `loan_path` has
                    // inherited mutability
                    LpExtend(lp_base, mc::McInherited, _) => {
                        loan_path = lp_base;
                    }

                    // Otherwise stop iterating
                    LpExtend(_, mc::McDeclared, _) |
                    LpExtend(_, mc::McImmutable, _) |
                    LpExtend(_, mc::McReadOnly, _) |
                    LpVar(_) => {
                        return true;
                    }
                }

                // Check for a non-const loan of `loan_path`
                for this.each_in_scope_loan(expr.id) |loan| {
                    if loan.loan_path == loan_path && loan.mutbl != m_const {
                        this.report_illegal_mutation(expr, full_loan_path, loan);
                        return false;
                    }
                }
            }
        }
    }

    fn report_illegal_mutation(&self,
                               expr: @ast::expr,
                               loan_path: &LoanPath,
                               loan: &Loan) {
        self.bccx.span_err(
            expr.span,
            fmt!("cannot assign to `%s` because it is borrowed",
                 self.bccx.loan_path_to_str(loan_path)));
        self.bccx.span_note(
            loan.span,
            fmt!("borrow of `%s` occurs here",
                 self.bccx.loan_path_to_str(loan_path)));
    }

    fn check_move_out_from_expr(&self, ex: @ast::expr) {
        match ex.node {
            ast::expr_paren(*) => {
                /* In the case of an expr_paren(), the expression inside
                 * the parens will also be marked as being moved.  Ignore
                 * the parents then so as not to report duplicate errors. */
            }
            _ => {
                let cmt = self.bccx.cat_expr(ex);
                match self.analyze_move_out_from_cmt(cmt) {
                    MoveOk => {}
                    MoveFromIllegalCmt(_) => {
                        self.bccx.span_err(
                            cmt.span,
                            fmt!("cannot move out of %s",
                                 self.bccx.cmt_to_str(cmt)));
                    }
                    MoveWhileBorrowed(loan_path, loan_span) => {
                        self.bccx.span_err(
                            cmt.span,
                            fmt!("cannot move out of `%s` \
                                  because it is borrowed",
                                 self.bccx.loan_path_to_str(loan_path)));
                        self.bccx.span_note(
                            loan_span,
                            fmt!("borrow of `%s` occurs here",
                                 self.bccx.loan_path_to_str(loan_path)));
                    }
                }
            }
        }
    }

    fn analyze_move_out_from_cmt(&self, cmt: mc::cmt) -> MoveError {
        debug!("check_move_out_from_cmt(cmt=%s)", cmt.repr(self.tcx()));

        match cmt.cat {
            // Rvalues, locals, and arguments can be moved:
            mc::cat_rvalue | mc::cat_local(_) |
            mc::cat_arg(_) | mc::cat_self(_) => {}

            // It seems strange to allow a move out of a static item,
            // but what happens in practice is that you have a
            // reference to a constant with a type that should be
            // moved, like `None::<~int>`.  The type of this constant
            // is technically `Option<~int>`, which moves, but we know
            // that the content of static items will never actually
            // contain allocated pointers, so we can just memcpy it.
            mc::cat_static_item => {}

            mc::cat_deref(_, _, mc::unsafe_ptr(*)) => {}

            // Nothing else.
            _ => {
                return MoveFromIllegalCmt(cmt);
            }
        }

        // FIXME(#4384) inadequare if/when we permit `move a.b`

        // check for a conflicting loan:
        for opt_loan_path(cmt).each |&lp| {
            for self.each_in_scope_restriction(cmt.id, lp) |loan, _| {
                // Any restriction prevents moves.
                return MoveWhileBorrowed(loan.loan_path, loan.span);
            }
        }

        return MoveOk;
    }

    fn check_call(&mut self,
                  _expr: @ast::expr,
                  _callee: Option<@ast::expr>,
                  _callee_id: ast::node_id,
                  _callee_span: span,
                  _args: &[@ast::expr])
    {
        // NB: This call to check for conflicting loans is not truly
        // necessary, because the callee_id never issues new loans.
        // However, I added it for consistency and lest the system
        // should change in the future.
        //
        // FIXME(#6268) nested method calls
        // self.check_for_conflicting_loans(callee_id);
    }
}

fn check_loans_in_fn<'a>(fk: &visit::fn_kind,
                         decl: &ast::fn_decl,
                         body: &ast::blk,
                         sp: span,
                         id: ast::node_id,
                         this: @mut CheckLoanCtxt<'a>,
                         visitor: visit::vt<@mut CheckLoanCtxt<'a>>) {
    match *fk {
        visit::fk_item_fn(*) |
        visit::fk_method(*) => {
            // Don't process nested items.
            return;
        }

        visit::fk_anon(*) |
        visit::fk_fn_block(*) => {
            let fty = ty::node_id_to_type(this.tcx(), id);
            let fty_sigil = ty::ty_closure_sigil(fty);
            check_moves_from_captured_variables(this, id, fty_sigil);
        }
    }

    visit::visit_fn(fk, decl, body, sp, id, this, visitor);

    fn check_moves_from_captured_variables(this: @mut CheckLoanCtxt,
                                           id: ast::node_id,
                                           fty_sigil: ast::Sigil) {
        match fty_sigil {
            ast::ManagedSigil | ast::OwnedSigil => {
                let cap_vars = this.bccx.capture_map.get(&id);
                for cap_vars.each |cap_var| {
                    match cap_var.mode {
                        moves::CapRef | moves::CapCopy => { loop; }
                        moves::CapMove => { }
                    }
                    let def_id = ast_util::def_id_of_def(cap_var.def).node;
                    let ty = ty::node_id_to_type(this.tcx(), def_id);
                    let cmt = this.bccx.cat_def(id, cap_var.span,
                                                ty, cap_var.def);
                    let move_err = this.analyze_move_out_from_cmt(cmt);
                    match move_err {
                        MoveOk => {}
                        MoveFromIllegalCmt(move_cmt) => {
                            this.bccx.span_err(
                                cap_var.span,
                                fmt!("illegal by-move capture of %s",
                                     this.bccx.cmt_to_str(move_cmt)));
                        }
                        MoveWhileBorrowed(loan_path, loan_span) => {
                            this.bccx.span_err(
                                cap_var.span,
                                fmt!("cannot move `%s` into closure \
                                      because it is borrowed",
                                     this.bccx.loan_path_to_str(loan_path)));
                            this.bccx.span_note(
                                loan_span,
                                fmt!("borrow of `%s` occurs here",
                                     this.bccx.loan_path_to_str(loan_path)));
                        }
                    }
                }
            }

            ast::BorrowedSigil => {}
        }
    }
}

fn check_loans_in_local<'a>(local: @ast::local,
                            this: @mut CheckLoanCtxt<'a>,
                            vt: visit::vt<@mut CheckLoanCtxt<'a>>) {
    visit::visit_local(local, this, vt);
}

fn check_loans_in_expr<'a>(expr: @ast::expr,
                           this: @mut CheckLoanCtxt<'a>,
                           vt: visit::vt<@mut CheckLoanCtxt<'a>>) {
    debug!("check_loans_in_expr(expr=%s)",
           expr.repr(this.tcx()));

    visit::visit_expr(expr, this, vt);

    this.check_for_conflicting_loans(expr.id);

    if this.bccx.moves_map.contains(&expr.id) {
        this.check_move_out_from_expr(expr);
    }

    match expr.node {
      ast::expr_assign(dest, _) |
      ast::expr_assign_op(_, dest, _) => {
        this.check_assignment(dest);
      }
      ast::expr_call(f, ref args, _) => {
        this.check_call(expr, Some(f), f.id, f.span, *args);
      }
      ast::expr_method_call(_, _, _, ref args, _) => {
        this.check_call(expr, None, expr.callee_id, expr.span, *args);
      }
      ast::expr_index(_, rval) |
      ast::expr_binary(_, _, rval)
      if this.bccx.method_map.contains_key(&expr.id) => {
        this.check_call(expr,
                        None,
                        expr.callee_id,
                        expr.span,
                        ~[rval]);
      }
      ast::expr_unary(*) | ast::expr_index(*)
      if this.bccx.method_map.contains_key(&expr.id) => {
        this.check_call(expr,
                        None,
                        expr.callee_id,
                        expr.span,
                        ~[]);
      }
      _ => { }
    }
}

fn check_loans_in_pat<'a>(pat: @ast::pat,
                          this: @mut CheckLoanCtxt<'a>,
                          vt: visit::vt<@mut CheckLoanCtxt<'a>>)
{
    this.check_for_conflicting_loans(pat.id);

    // Note: moves out of pattern bindings are not checked by
    // the borrow checker, at least not directly.  What happens
    // is that if there are any moved bindings, the discriminant
    // will be considered a move, and this will be checked as
    // normal.  Then, in `middle::check_match`, we will check
    // that no move occurs in a binding that is underneath an
    // `@` or `&`.  Together these give the same guarantees as
    // `check_move_out_from_expr()` without requiring us to
    // rewalk the patterns and rebuild the pattern
    // categorizations.

    visit::visit_pat(pat, this, vt);
}

fn check_loans_in_block<'a>(blk: &ast::blk,
                            this: @mut CheckLoanCtxt<'a>,
                            vt: visit::vt<@mut CheckLoanCtxt<'a>>)
{
    visit::visit_block(blk, this, vt);
    this.check_for_conflicting_loans(blk.node.id);
}

