// ----------------------------------------------------------------------
// Gathering loans
//
// The borrow check proceeds in two phases. In phase one, we gather the full
// set of loans that are required at any point.  These are sorted according to
// their associated scopes.  In phase two, checking loans, we will then make
// sure that all of these loans are honored.

use crate::borrowck::*;
use crate::borrowck::move_data::MoveData;
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::mem_categorization as mc;
use rustc::middle::mem_categorization::Categorization;
use rustc::middle::region;
use rustc::ty::{self, TyCtxt};

use syntax_pos::Span;
use rustc::hir;
use log::debug;

use restrictions::RestrictionResult;

mod lifetime;
mod restrictions;
mod gather_moves;
mod move_error;

pub fn gather_loans_in_fn<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                                    body: hir::BodyId)
                                    -> (Vec<Loan<'tcx>>, move_data::MoveData<'tcx>) {
    let def_id = bccx.tcx.hir().body_owner_def_id(body);
    let param_env = bccx.tcx.param_env(def_id);
    let mut glcx = GatherLoanCtxt {
        bccx,
        all_loans: Vec::new(),
        item_ub: region::Scope {
            id: bccx.tcx.hir().body(body).value.hir_id.local_id,
            data: region::ScopeData::Node
        },
        move_data: MoveData::default(),
        move_error_collector: move_error::MoveErrorCollector::new(),
    };

    let rvalue_promotable_map = bccx.tcx.rvalue_promotable_map(def_id);
    euv::ExprUseVisitor::new(&mut glcx,
                             bccx.tcx,
                             def_id,
                             param_env,
                             &bccx.region_scope_tree,
                             bccx.tables,
                             Some(rvalue_promotable_map))
        .consume_body(bccx.body);

    glcx.report_potential_errors();
    let GatherLoanCtxt { all_loans, move_data, .. } = glcx;
    (all_loans, move_data)
}

struct GatherLoanCtxt<'a, 'tcx> {
    bccx: &'a BorrowckCtxt<'a, 'tcx>,
    move_data: move_data::MoveData<'tcx>,
    move_error_collector: move_error::MoveErrorCollector<'tcx>,
    all_loans: Vec<Loan<'tcx>>,
    /// `item_ub` is used as an upper-bound on the lifetime whenever we
    /// ask for the scope of an expression categorized as an upvar.
    item_ub: region::Scope,
}

impl<'a, 'tcx> euv::Delegate<'tcx> for GatherLoanCtxt<'a, 'tcx> {
    fn consume(&mut self,
               consume_id: hir::HirId,
               _consume_span: Span,
               cmt: &mc::cmt_<'tcx>,
               mode: euv::ConsumeMode) {
        debug!("consume(consume_id={}, cmt={:?}, mode={:?})",
               consume_id, cmt, mode);

        match mode {
            euv::Move(move_reason) => {
                gather_moves::gather_move_from_expr(
                    self.bccx, &self.move_data, &mut self.move_error_collector,
                    consume_id.local_id, cmt, move_reason);
            }
            euv::Copy => { }
        }
    }

    fn matched_pat(&mut self,
                   matched_pat: &hir::Pat,
                   cmt: &mc::cmt_<'tcx>,
                   mode: euv::MatchMode) {
        debug!("matched_pat(matched_pat={:?}, cmt={:?}, mode={:?})",
               matched_pat,
               cmt,
               mode);
    }

    fn consume_pat(&mut self,
                   consume_pat: &hir::Pat,
                   cmt: &mc::cmt_<'tcx>,
                   mode: euv::ConsumeMode) {
        debug!("consume_pat(consume_pat={:?}, cmt={:?}, mode={:?})",
               consume_pat,
               cmt,
               mode);

        match mode {
            euv::Copy => { return; }
            euv::Move(_) => { }
        }

        gather_moves::gather_move_from_pat(
            self.bccx, &self.move_data, &mut self.move_error_collector,
            consume_pat, cmt);
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

        self.guarantee_valid(borrow_id.local_id,
                             borrow_span,
                             cmt,
                             bk,
                             loan_region,
                             loan_cause);
    }

    fn mutate(&mut self,
              assignment_id: hir::HirId,
              assignment_span: Span,
              assignee_cmt: &mc::cmt_<'tcx>,
              _: euv::MutateMode)
    {
        self.guarantee_assignment_valid(assignment_id,
                                        assignment_span,
                                        assignee_cmt);
    }

    fn decl_without_init(&mut self, id: hir::HirId, _span: Span) {
        let ty = self.bccx
                     .tables
                     .node_type(id);
        gather_moves::gather_decl(self.bccx, &self.move_data, id, ty);
    }

    fn nested_body(&mut self, body_id: hir::BodyId) {
        debug!("nested_body(body_id={:?})", body_id);
        // rust-lang/rust#58776: MIR and AST borrow check disagree on where
        // certain closure errors are reported. As such migrate borrowck has to
        // operate at the level of items, rather than bodies. Check if the
        // contained closure had any errors and set `signalled_any_error` if it
        // has.
        let bccx = self.bccx;
        if bccx.tcx.migrate_borrowck() {
            if let SignalledError::NoErrorsSeen = bccx.signalled_any_error.get() {
                let closure_def_id = bccx.tcx.hir().body_owner_def_id(body_id);
                debug!("checking closure: {:?}", closure_def_id);

                bccx.signalled_any_error.set(bccx.tcx.borrowck(closure_def_id).signalled_any_error);
            }
        }
    }
}

/// Implements the A-* rules in README.md.
fn check_aliasability<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                                borrow_span: Span,
                                loan_cause: AliasableViolationKind,
                                cmt: &mc::cmt_<'tcx>,
                                req_kind: ty::BorrowKind)
                                -> Result<(),()> {

    let aliasability = cmt.freely_aliasable();
    debug!("check_aliasability aliasability={:?} req_kind={:?}",
           aliasability, req_kind);

    match (aliasability, req_kind) {
        (mc::Aliasability::NonAliasable, _) => {
            /* Uniquely accessible path -- OK for `&` and `&mut` */
            Ok(())
        }
        (mc::Aliasability::FreelyAliasable(mc::AliasableStatic), ty::ImmBorrow) => {
            // Borrow of an immutable static item.
            Ok(())
        }
        (mc::Aliasability::FreelyAliasable(mc::AliasableStaticMut), _) => {
            // Even touching a static mut is considered unsafe. We assume the
            // user knows what they're doing in these cases.
            Ok(())
        }
        (mc::Aliasability::FreelyAliasable(alias_cause), ty::UniqueImmBorrow) |
        (mc::Aliasability::FreelyAliasable(alias_cause), ty::MutBorrow) => {
            bccx.report_aliasability_violation(
                        borrow_span,
                        loan_cause,
                        alias_cause,
                        cmt);
            Err(())
        }
        (..) => {
            Ok(())
        }
    }
}

/// Implements the M-* rules in README.md.
fn check_mutability<'a, 'tcx>(bccx: &BorrowckCtxt<'a, 'tcx>,
                              borrow_span: Span,
                              cause: AliasableViolationKind,
                              cmt: &mc::cmt_<'tcx>,
                              req_kind: ty::BorrowKind)
                              -> Result<(),()> {
    debug!("check_mutability(cause={:?} cmt={:?} req_kind={:?}",
           cause, cmt, req_kind);
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
                                           cause,
                                           cmt,
                                           code: err_mutbl }))
            } else {
                Ok(())
            }
        }
    }
}

impl<'a, 'tcx> GatherLoanCtxt<'a, 'tcx> {
    pub fn tcx(&self) -> TyCtxt<'tcx> { self.bccx.tcx }

    /// Guarantees that `cmt` is assignable, or reports an error.
    fn guarantee_assignment_valid(&mut self,
                                  assignment_id: hir::HirId,
                                  assignment_span: Span,
                                  cmt: &mc::cmt_<'tcx>) {

        let opt_lp = opt_loan_path(cmt);
        debug!("guarantee_assignment_valid(assignment_id={}, cmt={:?}) opt_lp={:?}",
               assignment_id, cmt, opt_lp);

        if let Categorization::Local(..) = cmt.cat {
            // Only re-assignments to locals require it to be
            // mutable - this is checked in check_loans.
        } else {
            // Check that we don't allow assignments to non-mutable data.
            if check_mutability(self.bccx, assignment_span, MutabilityViolation,
                                cmt, ty::MutBorrow).is_err() {
                return; // reported an error, no sense in reporting more.
            }
        }

        // Check that we don't allow assignments to aliasable data
        if check_aliasability(self.bccx, assignment_span, MutabilityViolation,
                              cmt, ty::MutBorrow).is_err() {
            return; // reported an error, no sense in reporting more.
        }

        match opt_lp {
            Some(lp) => {
                if let Categorization::Local(..) = cmt.cat {
                    // Only re-assignments to locals require it to be
                    // mutable - this is checked in check_loans.
                } else {
                    self.mark_loan_path_as_mutated(&lp);
                }
                gather_moves::gather_assignment(self.bccx, &self.move_data,
                                                assignment_id.local_id,
                                                assignment_span,
                                                lp);
            }
            None => {
                // This can occur with e.g., `*foo() = 5`.  In such
                // cases, there is no need to check for conflicts
                // with moves etc, just ignore.
            }
        }
    }

    /// Guarantees that `addr_of(cmt)` will be valid for the duration of `static_scope_r`, or
    /// reports an error. This may entail taking out loans, which will be added to the
    /// `req_loan_map`.
    fn guarantee_valid(&mut self,
                       borrow_id: hir::ItemLocalId,
                       borrow_span: Span,
                       cmt: &mc::cmt_<'tcx>,
                       req_kind: ty::BorrowKind,
                       loan_region: ty::Region<'tcx>,
                       cause: euv::LoanCause) {
        debug!("guarantee_valid(borrow_id={:?}, cmt={:?}, \
                req_mutbl={:?}, loan_region={:?})",
               borrow_id,
               cmt,
               req_kind,
               loan_region);

        // a loan for the empty region can never be dereferenced, so
        // it is always safe
        if *loan_region == ty::ReEmpty {
            return;
        }

        // Check that the lifetime of the borrow does not exceed
        // the lifetime of the data being borrowed.
        if lifetime::guarantee_lifetime(self.bccx, self.item_ub,
                                        borrow_span, cause, cmt, loan_region).is_err() {
            return; // reported an error, no sense in reporting more.
        }

        // Check that we don't allow mutable borrows of non-mutable data.
        if check_mutability(self.bccx, borrow_span, BorrowViolation(cause),
                            cmt, req_kind).is_err() {
            return; // reported an error, no sense in reporting more.
        }

        // Check that we don't allow mutable borrows of aliasable data.
        if check_aliasability(self.bccx, borrow_span, BorrowViolation(cause),
                              cmt, req_kind).is_err() {
            return; // reported an error, no sense in reporting more.
        }

        // Compute the restrictions that are required to enforce the
        // loan is safe.
        let restr = restrictions::compute_restrictions(
            self.bccx, borrow_span, cause, &cmt, loan_region);

        debug!("guarantee_valid(): restrictions={:?}", restr);

        // Create the loan record (if needed).
        let loan = match restr {
            RestrictionResult::Safe => {
                // No restrictions---no loan record necessary
                return;
            }

            RestrictionResult::SafeIf(loan_path, restricted_paths) => {
                let loan_scope = match *loan_region {
                    ty::ReScope(scope) => scope,

                    ty::ReEarlyBound(ref br) => {
                        self.bccx.region_scope_tree.early_free_scope(self.tcx(), br)
                    }

                    ty::ReFree(ref fr) => {
                        self.bccx.region_scope_tree.free_scope(self.tcx(), fr)
                    }

                    ty::ReStatic => self.item_ub,

                    ty::ReEmpty |
                    ty::ReClosureBound(..) |
                    ty::ReLateBound(..) |
                    ty::ReVar(..) |
                    ty::RePlaceholder(..) |
                    ty::ReErased => {
                        span_bug!(
                            cmt.span,
                            "invalid borrow lifetime: {:?}",
                            loan_region);
                    }
                };
                debug!("loan_scope = {:?}", loan_scope);

                let borrow_scope = region::Scope {
                    id: borrow_id,
                    data: region::ScopeData::Node
                };
                let gen_scope = self.compute_gen_scope(borrow_scope, loan_scope);
                debug!("gen_scope = {:?}", gen_scope);

                let kill_scope = self.compute_kill_scope(loan_scope, &loan_path);
                debug!("kill_scope = {:?}", kill_scope);

                if req_kind == ty::MutBorrow {
                    self.mark_loan_path_as_mutated(&loan_path);
                }

                Loan {
                    index: self.all_loans.len(),
                    loan_path,
                    kind: req_kind,
                    gen_scope,
                    kill_scope,
                    span: borrow_span,
                    restricted_paths,
                    cause,
                }
            }
        };

        debug!("guarantee_valid(borrow_id={:?}), loan={:?}",
               borrow_id, loan);

        // let loan_path = loan.loan_path;
        // let loan_gen_scope = loan.gen_scope;
        // let loan_kill_scope = loan.kill_scope;
        self.all_loans.push(loan);

        // if loan_gen_scope != borrow_id {
            // FIXME(https://github.com/rust-lang/rfcs/issues/811) Nested method calls
            //
            // Typically, the scope of the loan includes the point at
            // which the loan is originated. This
            // This is a subtle case. See the test case
            // <compile-fail/borrowck-bad-nested-calls-free.rs>
            // to see what we are guarding against.

            //let restr = restrictions::compute_restrictions(
            //    self.bccx, borrow_span, cmt, RESTR_EMPTY);
            //let loan = {
            //    Loan {
            //        index: self.all_loans.len(),
            //        loan_path,
            //        cmt,
            //        mutbl: ConstMutability,
            //        gen_scope: borrow_id,
            //        kill_scope,
            //        span: borrow_span,
            //        restrictions,
            //    }
        // }
    }

    pub fn mark_loan_path_as_mutated(&self, loan_path: &LoanPath<'_>) {
        //! For mutable loans of content whose mutability derives
        //! from a local variable, mark the mutability decl as necessary.

        let mut wrapped_path = Some(loan_path);
        let mut through_borrow = false;

        while let Some(current_path) = wrapped_path {
            wrapped_path = match current_path.kind {
                LpVar(hir_id) => {
                    if !through_borrow {
                        self.bccx.used_mut_nodes.borrow_mut().insert(hir_id);
                    }
                    None
                }
                LpUpvar(ty::UpvarId{ var_path: ty::UpvarPath { hir_id }, closure_expr_id: _ }) => {
                    self.bccx.used_mut_nodes.borrow_mut().insert(hir_id);
                    None
                }
                LpExtend(ref base, mc::McInherited, LpDeref(pointer_kind)) |
                LpExtend(ref base, mc::McDeclared, LpDeref(pointer_kind)) => {
                    if pointer_kind != mc::Unique {
                        through_borrow = true;
                    }
                    Some(base)
                }
                LpDowncast(ref base, _) |
                LpExtend(ref base, mc::McInherited, _) |
                LpExtend(ref base, mc::McDeclared, _) => {
                    Some(base)
                }
                LpExtend(_, mc::McImmutable, _) => {
                    // Nothing to do.
                    None
                }
            }
        }

    }

    pub fn compute_gen_scope(&self,
                             borrow_scope: region::Scope,
                             loan_scope: region::Scope)
                             -> region::Scope {
        //! Determine when to introduce the loan. Typically the loan
        //! is introduced at the point of the borrow, but in some cases,
        //! notably method arguments, the loan may be introduced only
        //! later, once it comes into scope.

        if self.bccx.region_scope_tree.is_subscope_of(borrow_scope, loan_scope) {
            borrow_scope
        } else {
            loan_scope
        }
    }

    pub fn compute_kill_scope(&self, loan_scope: region::Scope, lp: &LoanPath<'tcx>)
                              -> region::Scope {
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
        //!     struct Foo { counter: u32 }
        //!
        //!     fn counter<'a>(v: &'a mut Foo) -> &'a mut u32 {
        //!         &mut v.counter
        //!     }
        //!
        //! In this case, the reference (`'a`) outlives the
        //! variable `v` that hosts it. Note that this doesn't come up
        //! with immutable `&` pointers, because borrows of such pointers
        //! do not require restrictions and hence do not cause a loan.

        let lexical_scope = lp.kill_scope(self.bccx);
        if self.bccx.region_scope_tree.is_subscope_of(lexical_scope, loan_scope) {
            lexical_scope
        } else {
            assert!(self.bccx.region_scope_tree.is_subscope_of(loan_scope, lexical_scope));
            loan_scope
        }
    }

    pub fn report_potential_errors(&self) {
        self.move_error_collector.report_potential_errors(self.bccx);
    }
}
