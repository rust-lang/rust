// ----------------------------------------------------------------------
// Checking loans
//
// Phase 2 of check: we walk down the tree and check that:
// 1. assignments are always made to mutable locations;
// 2. loans made in overlapping scopes do not conflict
// 3. assignments do not affect things loaned out as immutable
// 4. moves do not affect things loaned out in any way

use UseError::*;

use crate::borrowck::*;
use crate::borrowck::InteriorKind::{InteriorElement, InteriorField};
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::expr_use_visitor::MutateMode;
use rustc::middle::mem_categorization as mc;
use rustc::middle::mem_categorization::Categorization;
use rustc::middle::region;
use rustc::ty::{self, TyCtxt, RegionKind};
use syntax_pos::Span;
use rustc::hir;
use rustc::hir::Node;
use rustc_mir::util::borrowck_errors::{BorrowckErrors, Origin};
use log::debug;

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
        None => loan_path,
    };

    fn helper<'a, 'tcx>(loan_path: &'a LoanPath<'tcx>) -> Option<&'a LoanPath<'tcx>> {
        match loan_path.kind {
            LpVar(_) | LpUpvar(_) => None,
            LpExtend(ref lp_base, _, LpDeref(mc::Unique)) => {
                match helper(&lp_base) {
                    v @ Some(_) => v,
                    None => Some(&lp_base)
                }
            }
            LpDowncast(ref lp_base, _) |
            LpExtend(ref lp_base, ..) => helper(&lp_base)
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
            LpExtend(ref lp_base, ..) => helper(lp_base)
        }
    }
}

struct CheckLoanCtxt<'a, 'tcx> {
    bccx: &'a BorrowckCtxt<'a, 'tcx>,
    dfcx_loans: &'a LoanDataFlow<'tcx>,
    move_data: &'a move_data::FlowedMoveData<'tcx>,
    all_loans: &'a [Loan<'tcx>],
    movable_generator: bool,
}

impl<'a, 'tcx> euv::Delegate<'tcx> for CheckLoanCtxt<'a, 'tcx> {
    fn consume(&mut self,
               consume_id: hir::HirId,
               consume_span: Span,
               cmt: &mc::cmt_<'tcx>,
               mode: euv::ConsumeMode) {
        debug!("consume(consume_id={}, cmt={:?}, mode={:?})",
               consume_id, cmt, mode);

        self.consume_common(consume_id.local_id, consume_span, cmt, mode);
    }

    fn matched_pat(&mut self,
                   _matched_pat: &hir::Pat,
                   _cmt: &mc::cmt_<'_>,
                   _mode: euv::MatchMode) { }

    fn consume_pat(&mut self,
                   consume_pat: &hir::Pat,
                   cmt: &mc::cmt_<'tcx>,
                   mode: euv::ConsumeMode) {
        debug!("consume_pat(consume_pat={:?}, cmt={:?}, mode={:?})",
               consume_pat,
               cmt,
               mode);

        self.consume_common(consume_pat.hir_id.local_id, consume_pat.span, cmt, mode);
    }

    fn borrow(&mut self,
              borrow_id: hir::HirId,
              borrow_span: Span,
              cmt: &mc::cmt_<'tcx>,
              loan_region: ty::Region<'tcx>,
              bk: ty::BorrowKind,
              loan_cause: euv::LoanCause)
    {
        debug!("borrow(borrow_id={}, cmt={:?}, loan_region={:?}, \
               bk={:?}, loan_cause={:?})",
               borrow_id, cmt, loan_region,
               bk, loan_cause);

        if let Some(lp) = opt_loan_path(cmt) {
            let moved_value_use_kind = match loan_cause {
                euv::ClosureCapture(_) => MovedInCapture,
                _ => MovedInUse,
            };
            self.check_if_path_is_moved(borrow_id.local_id, borrow_span, moved_value_use_kind, &lp);
        }

        self.check_for_conflicting_loans(borrow_id.local_id);

        self.check_for_loans_across_yields(cmt, loan_region, borrow_span);
    }

    fn mutate(&mut self,
              assignment_id: hir::HirId,
              assignment_span: Span,
              assignee_cmt: &mc::cmt_<'tcx>,
              mode: euv::MutateMode)
    {
        debug!("mutate(assignment_id={}, assignee_cmt={:?})",
               assignment_id, assignee_cmt);

        if let Some(lp) = opt_loan_path(assignee_cmt) {
            match mode {
                MutateMode::Init | MutateMode::JustWrite => {
                    // In a case like `path = 1`, then path does not
                    // have to be *FULLY* initialized, but we still
                    // must be careful lest it contains derefs of
                    // pointers.
                    self.check_if_assigned_path_is_moved(assignee_cmt.hir_id.local_id,
                                                         assignment_span,
                                                         MovedInUse,
                                                         &lp);
                }
                MutateMode::WriteAndRead => {
                    // In a case like `path += 1`, then path must be
                    // fully initialized, since we will read it before
                    // we write it.
                    self.check_if_path_is_moved(assignee_cmt.hir_id.local_id,
                                                assignment_span,
                                                MovedInUse,
                                                &lp);
                }
            }
        }
        self.check_assignment(assignment_id.local_id, assignment_span, assignee_cmt);
    }

    fn decl_without_init(&mut self, _id: hir::HirId, _span: Span) { }
}

pub fn check_loans<'a, 'tcx>(
    bccx: &BorrowckCtxt<'a, 'tcx>,
    dfcx_loans: &LoanDataFlow<'tcx>,
    move_data: &move_data::FlowedMoveData<'tcx>,
    all_loans: &[Loan<'tcx>],
    body: &hir::Body,
) {
    debug!("check_loans(body id={})", body.value.hir_id);

    let def_id = bccx.tcx.hir().body_owner_def_id(body.id());

    let hir_id = bccx.tcx.hir().as_local_hir_id(def_id).unwrap();
    let movable_generator = !match bccx.tcx.hir().get(hir_id) {
        Node::Expr(&hir::Expr {
            node: hir::ExprKind::Closure(.., Some(hir::GeneratorMovability::Static)),
            ..
        }) => true,
        _ => false,
    };

    let param_env = bccx.tcx.param_env(def_id);
    let mut clcx = CheckLoanCtxt {
        bccx,
        dfcx_loans,
        move_data,
        all_loans,
        movable_generator,
    };
    let rvalue_promotable_map = bccx.tcx.rvalue_promotable_map(def_id);
    euv::ExprUseVisitor::new(&mut clcx,
                             bccx.tcx,
                             def_id,
                             param_env,
                             &bccx.region_scope_tree,
                             bccx.tables,
                             Some(rvalue_promotable_map))
        .consume_body(body);
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
    pub fn tcx(&self) -> TyCtxt<'tcx> { self.bccx.tcx }

    pub fn each_issued_loan<F>(&self, node: hir::ItemLocalId, mut op: F) -> bool where
        F: FnMut(&Loan<'tcx>) -> bool,
    {
        //! Iterates over each loan that has been issued
        //! on entrance to `node`, regardless of whether it is
        //! actually *in scope* at that point. Sometimes loans
        //! are issued for future scopes and thus they may have been
        //! *issued* but not yet be in effect.

        self.dfcx_loans.each_bit_on_entry(node, |loan_index| {
            let loan = &self.all_loans[loan_index];
            op(loan)
        })
    }

    pub fn each_in_scope_loan<F>(&self, scope: region::Scope, mut op: F) -> bool where
        F: FnMut(&Loan<'tcx>) -> bool,
    {
        //! Like `each_issued_loan()`, but only considers loans that are
        //! currently in scope.

        self.each_issued_loan(scope.item_local_id(), |loan| {
            if self.bccx.region_scope_tree.is_subscope_of(scope, loan.kill_scope) {
                op(loan)
            } else {
                true
            }
        })
    }

    fn each_in_scope_loan_affecting_path<F>(&self,
                                            scope: region::Scope,
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
                LpExtend(ref lp_base, ..) => {
                    loan_path = &lp_base;
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

    pub fn loans_generated_by(&self, node: hir::ItemLocalId) -> Vec<usize> {
        //! Returns a vector of the loans that are generated as
        //! we enter `node`.

        let mut result = Vec::new();
        self.dfcx_loans.each_gen_bit(node, |loan_index| {
            result.push(loan_index);
            true
        });
        return result;
    }

    pub fn check_for_loans_across_yields(&self,
                                         cmt: &mc::cmt_<'tcx>,
                                         loan_region: ty::Region<'tcx>,
                                         borrow_span: Span) {
        pub fn borrow_of_local_data(cmt: &mc::cmt_<'_>) -> bool {
            match cmt.cat {
                // Borrows of static items is allowed
                Categorization::StaticItem => false,
                // Reborrow of already borrowed data is ignored
                // Any errors will be caught on the initial borrow
                Categorization::Deref(..) => false,

                // By-ref upvars has Derefs so they will get ignored.
                // Generators counts as FnOnce so this leaves only
                // by-move upvars, which is local data for generators
                Categorization::Upvar(..) => true,

                Categorization::ThreadLocal(region) |
                Categorization::Rvalue(region) => {
                    // Rvalues promoted to 'static are no longer local
                    if let RegionKind::ReStatic = *region {
                        false
                    } else {
                        true
                    }
                }

                // Borrow of local data must be checked
                Categorization::Local(..) => true,

                // For interior references and downcasts, find out if the base is local
                Categorization::Downcast(ref cmt_base, _) |
                Categorization::Interior(ref cmt_base, _) => borrow_of_local_data(&cmt_base),
            }
        }

        if !self.movable_generator {
            return;
        }

        if !borrow_of_local_data(cmt) {
            return;
        }

        let scope = match *loan_region {
            // A concrete region in which we will look for a yield expression
            RegionKind::ReScope(scope) => scope,

            // There cannot be yields inside an empty region
            RegionKind::ReEmpty => return,

            // Local data cannot have these lifetimes
            RegionKind::ReEarlyBound(..) |
            RegionKind::ReLateBound(..) |
            RegionKind::ReFree(..) |
            RegionKind::ReStatic => {
                self.bccx
                    .tcx
                    .sess.delay_span_bug(borrow_span,
                                         &format!("unexpected region for local data {:?}",
                                                  loan_region));
                return
            }

            // These cannot exist in borrowck
            RegionKind::ReVar(..) |
            RegionKind::RePlaceholder(..) |
            RegionKind::ReClosureBound(..) |
            RegionKind::ReErased => span_bug!(borrow_span,
                                              "unexpected region in borrowck {:?}",
                                              loan_region),
        };

        let body_id = self.bccx.body.value.hir_id.local_id;

        if self.bccx.region_scope_tree.containing_body(scope) != Some(body_id) {
            // We are borrowing local data longer than its storage.
            // This should result in other borrowck errors.
            self.bccx.tcx.sess.delay_span_bug(borrow_span,
                                              "borrowing local data longer than its storage");
            return;
        }

        if let Some(yield_span) = self.bccx
                                      .region_scope_tree
                                      .yield_in_scope_for_expr(scope,
                                                               cmt.hir_id,
                                                               self.bccx.body)
        {
            self.bccx.cannot_borrow_across_generator_yield(borrow_span,
                                                           yield_span,
                                                           Origin::Ast).emit();
            self.bccx.signal_error();
        }
    }

    pub fn check_for_conflicting_loans(&self, node: hir::ItemLocalId) {
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
        assert!(self.bccx.region_scope_tree.scopes_intersect(old_loan.kill_scope,
                                                       new_loan.kill_scope));

        let err_old_new = self.report_error_if_loan_conflicts_with_restriction(
            old_loan, new_loan, old_loan, new_loan).err();
        let err_new_old = self.report_error_if_loan_conflicts_with_restriction(
            new_loan, old_loan, old_loan, new_loan).err();

        match (err_old_new, err_new_old) {
            (Some(mut err), None) | (None, Some(mut err)) => {
                err.emit();
                self.bccx.signal_error();
            }
            (Some(mut err_old), Some(mut err_new)) => {
                err_old.emit();
                self.bccx.signal_error();
                err_new.cancel();
            }
            (None, None) => return true,
        }

        false
    }

    pub fn report_error_if_loan_conflicts_with_restriction(&self,
                                                           loan1: &Loan<'tcx>,
                                                           loan2: &Loan<'tcx>,
                                                           old_loan: &Loan<'tcx>,
                                                           new_loan: &Loan<'tcx>)
                                                           -> Result<(), DiagnosticBuilder<'a>> {
        //! Checks whether the restrictions introduced by `loan1` would
        //! prohibit `loan2`. Returns false if an error is reported.

        debug!("report_error_if_loan_conflicts_with_restriction(\
                loan1={:?}, loan2={:?})",
               loan1,
               loan2);

        if compatible_borrow_kinds(loan1.kind, loan2.kind) {
            return Ok(());
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

            let common = new_loan.loan_path.common(&old_loan.loan_path);
            let (nl, ol, new_loan_msg, old_loan_msg) = {
                if new_loan.loan_path.has_fork(&old_loan.loan_path) && common.is_some() {
                    let nl = self.bccx.loan_path_to_string(&common.unwrap());
                    let ol = nl.clone();
                    let new_loan_msg = self.bccx.loan_path_to_string(&new_loan.loan_path);
                    let old_loan_msg = self.bccx.loan_path_to_string(&old_loan.loan_path);
                    (nl, ol, new_loan_msg, old_loan_msg)
                } else {
                    (self.bccx.loan_path_to_string(&new_loan.loan_path),
                     self.bccx.loan_path_to_string(&old_loan.loan_path),
                     String::new(),
                     String::new())
                }
            };

            let ol_pronoun = if new_loan.loan_path == old_loan.loan_path {
                "it".to_string()
            } else {
                format!("`{}`", ol)
            };

            // We want to assemble all the relevant locations for the error.
            //
            // 1. Where did the new loan occur.
            //    - if due to closure creation, where was the variable used in closure?
            // 2. Where did old loan occur.
            // 3. Where does old loan expire.

            let previous_end_span =
                Some(self.tcx().sess.source_map().end_point(
                        old_loan.kill_scope.span(self.tcx(), &self.bccx.region_scope_tree)));

            let mut err = match (new_loan.kind, old_loan.kind) {
                (ty::MutBorrow, ty::MutBorrow) =>
                    self.bccx.cannot_mutably_borrow_multiply(
                        new_loan.span, &nl, &new_loan_msg, old_loan.span, &old_loan_msg,
                        previous_end_span, Origin::Ast),
                (ty::UniqueImmBorrow, ty::UniqueImmBorrow) =>
                    self.bccx.cannot_uniquely_borrow_by_two_closures(
                        new_loan.span, &nl, old_loan.span, previous_end_span, Origin::Ast),
                (ty::UniqueImmBorrow, _) =>
                    self.bccx.cannot_uniquely_borrow_by_one_closure(
                        new_loan.span, "closure", &nl, &new_loan_msg,
                        old_loan.span, &ol_pronoun, &old_loan_msg, previous_end_span, Origin::Ast),
                (_, ty::UniqueImmBorrow) => {
                    let new_loan_str = &new_loan.kind.to_user_str();
                    self.bccx.cannot_reborrow_already_uniquely_borrowed(
                        new_loan.span, "closure", &nl, &new_loan_msg, new_loan_str,
                        old_loan.span, &old_loan_msg, previous_end_span, "", Origin::Ast)
                }
                (..) =>
                    self.bccx.cannot_reborrow_already_borrowed(
                        new_loan.span,
                        &nl, &new_loan_msg, &new_loan.kind.to_user_str(),
                        old_loan.span, &ol_pronoun, &old_loan.kind.to_user_str(), &old_loan_msg,
                        previous_end_span, Origin::Ast)
            };

            match new_loan.cause {
                euv::ClosureCapture(span) => {
                    err.span_label(
                        span,
                        format!("borrow occurs due to use of `{}` in closure", nl));
                }
                _ => { }
            }

            match old_loan.cause {
                euv::ClosureCapture(span) => {
                    err.span_label(
                        span,
                        format!("previous borrow occurs due to use of `{}` in closure",
                                 ol));
                }
                _ => { }
            }

            return Err(err);
        }

        Ok(())
    }

    fn consume_common(&self,
                      id: hir::ItemLocalId,
                      span: Span,
                      cmt: &mc::cmt_<'tcx>,
                      mode: euv::ConsumeMode) {
        if let Some(lp) = opt_loan_path(cmt) {
            let moved_value_use_kind = match mode {
                euv::Copy => {
                    self.check_for_copy_of_frozen_path(id, span, &lp);
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
                                                                 &lp, move_kind);
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
    }

    fn check_for_copy_of_frozen_path(&self,
                                     id: hir::ItemLocalId,
                                     span: Span,
                                     copy_path: &LoanPath<'tcx>) {
        match self.analyze_restrictions_on_use(id, copy_path, ty::ImmBorrow) {
            UseOk => { }
            UseWhileBorrowed(loan_path, loan_span) => {
                let desc = self.bccx.loan_path_to_string(copy_path);
                self.bccx.cannot_use_when_mutably_borrowed(
                        span, &desc,
                        loan_span, &self.bccx.loan_path_to_string(&loan_path),
                        Origin::Ast)
                    .emit();
                self.bccx.signal_error();
            }
        }
    }

    fn check_for_move_of_borrowed_path(&self,
                                       id: hir::ItemLocalId,
                                       span: Span,
                                       move_path: &LoanPath<'tcx>,
                                       move_kind: move_data::MoveKind) {
        // We want to detect if there are any loans at all, so we search for
        // any loans incompatible with MutBorrrow, since all other kinds of
        // loans are incompatible with that.
        match self.analyze_restrictions_on_use(id, move_path, ty::MutBorrow) {
            UseOk => { }
            UseWhileBorrowed(loan_path, loan_span) => {
                let mut err = match move_kind {
                    move_data::Captured => {
                        let mut err = self.bccx.cannot_move_into_closure(
                            span, &self.bccx.loan_path_to_string(move_path), Origin::Ast);
                        err.span_label(
                            loan_span,
                            format!("borrow of `{}` occurs here",
                                    &self.bccx.loan_path_to_string(&loan_path))
                            );
                        err.span_label(
                            span,
                            "move into closure occurs here"
                            );
                        err
                    }
                    move_data::Declared |
                    move_data::MoveExpr |
                    move_data::MovePat => {
                        let desc = self.bccx.loan_path_to_string(move_path);
                        let mut err = self.bccx.cannot_move_when_borrowed(span, &desc, Origin::Ast);
                        err.span_label(
                            loan_span,
                            format!("borrow of `{}` occurs here",
                                    &self.bccx.loan_path_to_string(&loan_path))
                            );
                        err.span_label(
                            span,
                            format!("move out of `{}` occurs here",
                                &self.bccx.loan_path_to_string(move_path))
                            );
                        err
                    }
                };

                err.emit();
                self.bccx.signal_error();
            }
        }
    }

    pub fn analyze_restrictions_on_use(&self,
                                       expr_id: hir::ItemLocalId,
                                       use_path: &LoanPath<'tcx>,
                                       borrow_kind: ty::BorrowKind)
                                       -> UseError<'tcx> {
        debug!("analyze_restrictions_on_use(expr_id={:?}, use_path={:?})",
               expr_id, use_path);

        let mut ret = UseOk;

        let scope = region::Scope {
            id: expr_id,
            data: region::ScopeData::Node
        };
        self.each_in_scope_loan_affecting_path(
            scope, use_path, |loan| {
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
                              id: hir::ItemLocalId,
                              span: Span,
                              use_kind: MovedValueUseKind,
                              lp: &Rc<LoanPath<'tcx>>) {
        debug!("check_if_path_is_moved(id={:?}, use_kind={:?}, lp={:?})",
               id, use_kind, lp);

        // FIXME: if you find yourself tempted to cut and paste
        // the body below and then specializing the error reporting,
        // consider refactoring this instead!

        let base_lp = owned_ptr_base_path_rc(lp);
        self.move_data.each_move_of(id, &base_lp, |the_move, moved_lp| {
            self.bccx.report_use_of_moved_value(
                span,
                use_kind,
                &lp,
                the_move,
                moved_lp);
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
    /// let a: i32;
    /// a = 10; // ok, even though a is uninitialized
    /// ```
    ///
    /// ```
    /// struct Point { x: u32, y: u32 }
    /// let mut p: Point;
    /// p.x = 22; // ok, even though `p` is uninitialized
    /// ```
    ///
    /// ```compile_fail,E0381
    /// # struct Point { x: u32, y: u32 }
    /// let mut p: Box<Point>;
    /// (*p).x = 22; // not ok, p is uninitialized, can't deref
    /// ```
    fn check_if_assigned_path_is_moved(&self,
                                       id: hir::ItemLocalId,
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
            LpExtend(ref lp_base, _, LpInterior(_, InteriorField(_))) => {
                match lp_base.to_type().sty {
                    ty::Adt(def, _) if def.has_dtor(self.tcx()) => {
                        // In the case where the owner implements drop, then
                        // the path must be initialized to prevent a case of
                        // partial reinitialization
                        //
                        // FIXME: could refactor via hypothetical
                        // generalized check_if_path_is_moved
                        let loan_path = owned_ptr_base_path_rc(lp_base);
                        self.move_data.each_move_of(id, &loan_path, |_, _| {
                            self.bccx
                                .report_partial_reinitialization_of_uninitialized_structure(
                                    span,
                                    &loan_path);
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
            LpExtend(ref lp_base, _, LpInterior(_, InteriorElement)) |
            LpExtend(ref lp_base, _, LpDeref(_)) => {
                // assigning to `P[i]` requires `P` is initialized
                // assigning to `(*P)` requires `P` is initialized
                self.check_if_path_is_moved(id, span, use_kind, lp_base);
            }
        }
    }

    fn check_assignment(&self,
                        assignment_id: hir::ItemLocalId,
                        assignment_span: Span,
                        assignee_cmt: &mc::cmt_<'tcx>) {
        debug!("check_assignment(assignee_cmt={:?})", assignee_cmt);

        // Check that we don't invalidate any outstanding loans
        if let Some(loan_path) = opt_loan_path(assignee_cmt) {
            let scope = region::Scope {
                id: assignment_id,
                data: region::ScopeData::Node
            };
            self.each_in_scope_loan_affecting_path(scope, &loan_path, |loan| {
                self.report_illegal_mutation(assignment_span, &loan_path, loan);
                false
            });
        }

        // Check for reassignments to (immutable) local variables. This
        // needs to be done here instead of in check_loans because we
        // depend on move data.
        if let Categorization::Local(hir_id) = assignee_cmt.cat {
            let lp = opt_loan_path(assignee_cmt).unwrap();
            self.move_data.each_assignment_of(assignment_id, &lp, |assign| {
                if assignee_cmt.mutbl.is_mutable() {
                    self.bccx.used_mut_nodes.borrow_mut().insert(hir_id);
                } else {
                    self.bccx.report_reassigned_immutable_variable(
                        assignment_span,
                        &lp,
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
                                   loan: &Loan<'_>) {
        self.bccx.cannot_assign_to_borrowed(
            span, loan.span, &self.bccx.loan_path_to_string(loan_path), Origin::Ast)
            .emit();
        self.bccx.signal_error();
    }
}
