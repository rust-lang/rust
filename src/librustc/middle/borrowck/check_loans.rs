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
use middle::typeck::check::PurityState;
use middle::borrowck::{Loan, bckerr, BorrowckCtxt, inherent_mutability};
use middle::borrowck::{ReqMaps, root_map_key, save_and_restore_managed};
use middle::borrowck::{MoveError, MoveOk, MoveFromIllegalCmt};
use middle::borrowck::{MoveWhileBorrowed};
use middle::mem_categorization::{cat_arg, cat_comp, cat_deref};
use middle::mem_categorization::{cat_local, cat_rvalue, cat_self};
use middle::mem_categorization::{cat_special, cmt, gc_ptr, loan_path, lp_arg};
use middle::mem_categorization::{lp_comp, lp_deref, lp_local};
use middle::ty;
use util::ppaux::ty_to_str;

use core::hashmap::HashSet;
use core::util::with;
use syntax::ast::m_mutbl;
use syntax::ast;
use syntax::ast_util;
use syntax::codemap::span;
use syntax::print::pprust;
use syntax::visit;

struct CheckLoanCtxt {
    bccx: @BorrowckCtxt,
    req_maps: ReqMaps,

    reported: HashSet<ast::node_id>,

    declared_purity: @mut PurityState,
    fn_args: @mut @~[ast::node_id]
}

// if we are enforcing purity, why are we doing so?
#[deriving(Eq)]
enum purity_cause {
    // enforcing purity because fn was declared pure:
    pc_pure_fn,

    // enforce purity because we need to guarantee the
    // validity of some alias; `bckerr` describes the
    // reason we needed to enforce purity.
    pc_cmt(bckerr)
}

// if we're not pure, why?
#[deriving(Eq)]
enum impurity_cause {
    // some surrounding block was marked as 'unsafe'
    pc_unsafe,

    // nothing was unsafe, and nothing was pure
    pc_default,
}

pub fn check_loans(bccx: @BorrowckCtxt,
                   req_maps: ReqMaps,
                   crate: @ast::crate) {
    let clcx = @mut CheckLoanCtxt {
        bccx: bccx,
        req_maps: req_maps,
        reported: HashSet::new(),
        declared_purity: @mut PurityState::function(ast::impure_fn, 0),
        fn_args: @mut @~[]
    };
    let vt = visit::mk_vt(@visit::Visitor {visit_expr: check_loans_in_expr,
                                           visit_local: check_loans_in_local,
                                           visit_block: check_loans_in_block,
                                           visit_fn: check_loans_in_fn,
                                           .. *visit::default_visitor()});
    visit::visit_crate(crate, clcx, vt);
}

#[deriving(Eq)]
enum assignment_type {
    at_straight_up,
    at_swap
}

pub impl assignment_type {
    fn checked_by_liveness(&self) -> bool {
        // the liveness pass guarantees that immutable local variables
        // are only assigned once; but it doesn't consider &mut
        match *self {
          at_straight_up => true,
          at_swap => true
        }
    }
    fn ing_form(&self, desc: ~str) -> ~str {
        match *self {
          at_straight_up => ~"assigning to " + desc,
          at_swap => ~"swapping to and from " + desc
        }
    }
}

pub impl CheckLoanCtxt {
    fn tcx(&self) -> ty::ctxt { self.bccx.tcx }

    fn purity(&mut self, scope_id: ast::node_id)
                -> Either<purity_cause, impurity_cause>
    {
        let default_purity = match self.declared_purity.purity {
          // an unsafe declaration overrides all
          ast::unsafe_fn => return Right(pc_unsafe),

          // otherwise, remember what was declared as the
          // default, but we must scan for requirements
          // imposed by the borrow check
          ast::pure_fn => Left(pc_pure_fn),
          ast::extern_fn | ast::impure_fn => Right(pc_default)
        };

        // scan to see if this scope or any enclosing scope requires
        // purity.  if so, that overrides the declaration.

        let mut scope_id = scope_id;
        loop {
            match self.req_maps.pure_map.find(&scope_id) {
              None => (),
              Some(e) => return Left(pc_cmt(*e))
            }

            match self.tcx().region_maps.opt_encl_scope(scope_id) {
              None => return default_purity,
              Some(next_scope_id) => scope_id = next_scope_id
            }
        }
    }

    fn walk_loans(&self,
                  mut scope_id: ast::node_id,
                  f: &fn(v: &Loan) -> bool) {

        loop {
            for self.req_maps.req_loan_map.find(&scope_id).each |loans| {
                for loans.each |loan| {
                    if !f(loan) { return; }
                }
            }

            match self.tcx().region_maps.opt_encl_scope(scope_id) {
              None => return,
              Some(next_scope_id) => scope_id = next_scope_id,
            }
        }
    }

    fn walk_loans_of(&mut self,
                     scope_id: ast::node_id,
                     lp: @loan_path,
                     f: &fn(v: &Loan) -> bool) {
        for self.walk_loans(scope_id) |loan| {
            if loan.lp == lp {
                if !f(loan) { return; }
            }
        }
    }

    // when we are in a pure context, we check each call to ensure
    // that the function which is invoked is itself pure.
    //
    // note: we take opt_expr and expr_id separately because for
    // overloaded operators the callee has an id but no expr.
    // annoying.
    fn check_pure_callee_or_arg(&mut self,
                                pc: Either<purity_cause, impurity_cause>,
                                opt_expr: Option<@ast::expr>,
                                callee_id: ast::node_id,
                                callee_span: span) {
        let tcx = self.tcx();

        debug!("check_pure_callee_or_arg(pc=%?, expr=%?, \
                callee_id=%d, ty=%s)",
               pc,
               opt_expr.map(|e| pprust::expr_to_str(*e, tcx.sess.intr()) ),
               callee_id,
               ty_to_str(self.tcx(), ty::node_id_to_type(tcx, callee_id)));

        // Purity rules: an expr B is a legal callee or argument to a
        // call within a pure function A if at least one of the
        // following holds:
        //
        // (a) A was declared pure and B is one of its arguments;
        // (b) B is a stack closure;
        // (c) B is a pure fn;
        // (d) B is not a fn.

        match opt_expr {
          Some(expr) => {
            match expr.node {
              ast::expr_path(_) if pc == Left(pc_pure_fn) => {
                let def = *self.tcx().def_map.get(&expr.id);
                let did = ast_util::def_id_of_def(def);
                let is_fn_arg =
                    did.crate == ast::local_crate &&
                    (*self.fn_args).contains(&(did.node));
                if is_fn_arg { return; } // case (a) above
              }
              ast::expr_fn_block(*) | ast::expr_loop_body(*) |
              ast::expr_do_body(*) => {
                if self.is_stack_closure(expr.id) {
                    // case (b) above
                    return;
                }
              }
              _ => ()
            }
          }
          None => ()
        }

        let callee_ty = ty::node_id_to_type(tcx, callee_id);
        match ty::get(callee_ty).sty {
            ty::ty_bare_fn(ty::BareFnTy {purity: purity, _}) |
            ty::ty_closure(ty::ClosureTy {purity: purity, _}) => {
                match purity {
                    ast::pure_fn => return, // case (c) above
                    ast::impure_fn | ast::unsafe_fn | ast::extern_fn => {
                        self.report_purity_error(
                            pc, callee_span,
                            fmt!("access to %s function",
                                 purity.to_str()));
                    }
                }
            }
            _ => return, // case (d) above
        }
    }

    // True if the expression with the given `id` is a stack closure.
    // The expression must be an expr_fn_block(*)
    fn is_stack_closure(&mut self, id: ast::node_id) -> bool {
        let fn_ty = ty::node_id_to_type(self.tcx(), id);
        match ty::get(fn_ty).sty {
            ty::ty_closure(ty::ClosureTy {sigil: ast::BorrowedSigil,
                                          _}) => true,
            _ => false
        }
    }

    fn is_allowed_pure_arg(&mut self, expr: @ast::expr) -> bool {
        return match expr.node {
          ast::expr_path(_) => {
            let def = *self.tcx().def_map.get(&expr.id);
            let did = ast_util::def_id_of_def(def);
            did.crate == ast::local_crate &&
                (*self.fn_args).contains(&(did.node))
          }
          ast::expr_fn_block(*) => self.is_stack_closure(expr.id),
          _ => false,
        };
    }

    fn check_for_conflicting_loans(&mut self, scope_id: ast::node_id) {
        debug!("check_for_conflicting_loans(scope_id=%?)", scope_id);

        let new_loans = match self.req_maps.req_loan_map.find(&scope_id) {
            None => return,
            Some(&loans) => loans
        };
        let new_loans: &mut ~[Loan] = new_loans;

        debug!("new_loans has length %?", new_loans.len());

        let par_scope_id = self.tcx().region_maps.encl_scope(scope_id);
        for self.walk_loans(par_scope_id) |old_loan| {
            debug!("old_loan=%?", self.bccx.loan_to_repr(old_loan));

            for new_loans.each |new_loan| {
                self.report_error_if_loans_conflict(old_loan, new_loan);
            }
        }

        let len = new_loans.len();
        for uint::range(0, len) |i| {
            let loan_i = new_loans[i];
            for uint::range(i+1, len) |j| {
                let loan_j = new_loans[j];
                self.report_error_if_loans_conflict(&loan_i, &loan_j);
            }
        }
    }

    fn report_error_if_loans_conflict(&self,
                                      old_loan: &Loan,
                                      new_loan: &Loan) {
        if old_loan.lp != new_loan.lp {
            return;
        }

        match (old_loan.kind, new_loan.kind) {
            (PartialFreeze, PartialTake) | (PartialTake, PartialFreeze) |
            (TotalFreeze, PartialFreeze) | (PartialFreeze, TotalFreeze) |
            (Immobile, _) | (_, Immobile) |
            (PartialFreeze, PartialFreeze) |
            (PartialTake, PartialTake) |
            (TotalFreeze, TotalFreeze) => {
                /* ok */
            }

            (PartialTake, TotalFreeze) | (TotalFreeze, PartialTake) |
            (TotalTake, TotalFreeze) | (TotalFreeze, TotalTake) |
            (TotalTake, PartialFreeze) | (PartialFreeze, TotalTake) |
            (TotalTake, PartialTake) | (PartialTake, TotalTake) |
            (TotalTake, TotalTake) => {
                self.bccx.span_err(
                    new_loan.cmt.span,
                    fmt!("loan of %s as %s \
                          conflicts with prior loan",
                         self.bccx.cmt_to_str(new_loan.cmt),
                         self.bccx.loan_kind_to_str(new_loan.kind)));
                self.bccx.span_note(
                    old_loan.cmt.span,
                    fmt!("prior loan as %s granted here",
                         self.bccx.loan_kind_to_str(old_loan.kind)));
            }
        }
    }

    fn is_local_variable(&self, cmt: cmt) -> bool {
        match cmt.cat {
          cat_local(_) => true,
          _ => false
        }
    }

    fn check_assignment(&mut self, at: assignment_type, ex: @ast::expr) {
        // We don't use cat_expr() here because we don't want to treat
        // auto-ref'd parameters in overloaded operators as rvalues.
        let cmt = match self.bccx.tcx.adjustments.find(&ex.id) {
            None => self.bccx.cat_expr_unadjusted(ex),
            Some(&adj) => self.bccx.cat_expr_autoderefd(ex, adj)
        };

        debug!("check_assignment(cmt=%s)",
               self.bccx.cmt_to_repr(cmt));

        if self.is_local_variable(cmt) && at.checked_by_liveness() {
            // liveness guarantees that immutable local variables
            // are only assigned once
        } else {
            match cmt.mutbl {
                McDeclared | McInherited => {
                    // Ok, but if this loan is a mutable loan, then mark the
                    // loan path (if it exists) as being used. This is similar
                    // to the check performed in loan.rs in issue_loan(). This
                    // type of use of mutable is different from issuing a loan,
                    // however.
                    for cmt.lp.each |lp| {
                        for lp.node_id().each |&id| {
                            self.tcx().used_mut_nodes.insert(id);
                        }
                    }
                }
                McReadOnly | McImmutable => {
                    self.bccx.span_err(
                        ex.span,
                        at.ing_form(self.bccx.cmt_to_str(cmt)));
                    return;
                }
            }
        }

        // if this is a pure function, only loan-able state can be
        // assigned, because it is uniquely tied to this function and
        // is not visible from the outside
        let purity = self.purity(ex.id);
        match purity {
          Right(_) => (),
          Left(pc_cmt(_)) => {
            // Subtle: Issue #3162.  If we are enforcing purity
            // because there is a reference to aliasable, mutable data
            // that we require to be immutable, we can't allow writes
            // even to data owned by the current stack frame.  This is
            // because that aliasable data might have been located on
            // the current stack frame, we don't know.
            self.report_purity_error(
                purity,
                ex.span,
                at.ing_form(self.bccx.cmt_to_str(cmt)));
          }
          Left(pc_pure_fn) => {
            if cmt.lp.is_none() {
                self.report_purity_error(
                    purity, ex.span,
                    at.ing_form(self.bccx.cmt_to_str(cmt)));
            }
          }
        }

        // check for a conflicting loan as well, except in the case of
        // taking a mutable ref.  that will create a loan of its own
        // which will be checked for compat separately in
        // check_for_conflicting_loans()
        for cmt.lp.each |lp| {
            self.check_for_loan_conflicting_with_assignment(
                at, ex, cmt, *lp);
        }

        self.bccx.add_to_mutbl_map(cmt);

        // Check for and insert write guards as necessary.
        self.add_write_guards_if_necessary(cmt);
    }

    fn add_write_guards_if_necessary(&mut self, cmt: cmt) {
        match cmt.cat {
            cat_deref(base, deref_count, ptr_kind) => {
                self.add_write_guards_if_necessary(base);

                match ptr_kind {
                    gc_ptr(ast::m_mutbl) => {
                        let key = root_map_key {
                            id: base.id,
                            derefs: deref_count
                        };
                        self.bccx.write_guard_map.insert(key);
                    }
                    _ => {}
                }
            }
            cat_comp(base, _) => {
                self.add_write_guards_if_necessary(base);
            }
            _ => {}
        }
    }

    fn check_for_loan_conflicting_with_assignment(&mut self,
                                                  at: assignment_type,
                                                  ex: @ast::expr,
                                                  cmt: cmt,
                                                  lp: @loan_path) {
        for self.walk_loans_of(ex.id, lp) |loan| {
            match loan.kind {
                Immobile => { /* ok */ }
                TotalFreeze | PartialFreeze |
                TotalTake | PartialTake => {
                    self.bccx.span_err(
                        ex.span,
                        fmt!("%s prohibited due to outstanding loan",
                             at.ing_form(self.bccx.cmt_to_str(cmt))));
                    self.bccx.span_note(
                        loan.cmt.span,
                        fmt!("loan of %s granted here",
                             self.bccx.cmt_to_str(loan.cmt)));
                    return;
                }
            }
        }

        // Subtle: if the mutability of the component being assigned
        // is inherited from the thing that the component is embedded
        // within, then we have to check whether that thing has been
        // loaned out as immutable!  An example:
        //    let mut x = {f: Some(3)};
        //    let y = &x; // x loaned out as immutable
        //    x.f = none; // changes type of y.f, which appears to be imm
        match *lp {
          lp_comp(lp_base, ck) if inherent_mutability(ck) != m_mutbl => {
            self.check_for_loan_conflicting_with_assignment(
                at, ex, cmt, lp_base);
          }
          lp_comp(*) | lp_self | lp_local(*) | lp_arg(*) | lp_deref(*) => ()
        }
    }

    fn report_purity_error(&mut self, pc: Either<purity_cause, impurity_cause>,
                           sp: span, msg: ~str) {
        match pc {
          Right(pc_default) => { fail!(~"pc_default should be filtered sooner") }
          Right(pc_unsafe) => {
            // this error was prevented by being marked as unsafe, so flag the
            // definition as having contributed to the validity of the program
            let def = self.declared_purity.def;
            debug!("flagging %? as a used unsafe source", def);
            self.tcx().used_unsafe.insert(def);
          }
          Left(pc_pure_fn) => {
            self.tcx().sess.span_err(
                sp,
                fmt!("%s prohibited in pure context", msg));
          }
          Left(pc_cmt(ref e)) => {
            if self.reported.insert((*e).cmt.id) {
                self.tcx().sess.span_err(
                    (*e).cmt.span,
                    fmt!("illegal borrow unless pure: %s",
                         self.bccx.bckerr_to_str((*e))));
                self.bccx.note_and_explain_bckerr((*e));
                self.tcx().sess.span_note(
                    sp,
                    fmt!("impure due to %s", msg));
            }
          }
        }
    }

    fn check_move_out_from_expr(@mut self, ex: @ast::expr) {
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
                            fmt!("moving out of %s",
                                 self.bccx.cmt_to_str(cmt)));
                    }
                    MoveWhileBorrowed(_, loan_cmt) => {
                        self.bccx.span_err(
                            cmt.span,
                            fmt!("moving out of %s prohibited \
                                  due to outstanding loan",
                                 self.bccx.cmt_to_str(cmt)));
                        self.bccx.span_note(
                            loan_cmt.span,
                            fmt!("loan of %s granted here",
                                 self.bccx.cmt_to_str(loan_cmt)));
                    }
                }
            }
        }
    }

    fn analyze_move_out_from_cmt(&mut self, cmt: cmt) -> MoveError {
        debug!("check_move_out_from_cmt(cmt=%s)",
               self.bccx.cmt_to_repr(cmt));

        match cmt.cat {
          // Rvalues, locals, and arguments can be moved:
          cat_rvalue | cat_local(_) | cat_arg(_) | cat_self(_) => {}

          // We allow moving out of static items because the old code
          // did.  This seems consistent with permitting moves out of
          // rvalues, I guess.
          cat_special(sk_static_item) => {}

          cat_deref(_, _, unsafe_ptr) => {}

          // Nothing else.
          _ => {
              return MoveFromIllegalCmt(cmt);
          }
        }

        self.bccx.add_to_mutbl_map(cmt);

        // check for a conflicting loan:
        for cmt.lp.each |lp| {
            for self.walk_loans_of(cmt.id, *lp) |loan| {
                return MoveWhileBorrowed(cmt, loan.cmt);
            }
        }

        return MoveOk;
    }

    fn check_call(&mut self,
                  expr: @ast::expr,
                  callee: Option<@ast::expr>,
                  callee_id: ast::node_id,
                  callee_span: span,
                  args: &[@ast::expr]) {
        let pc = self.purity(expr.id);
        match pc {
            // no purity, no need to check for anything
            Right(pc_default) => return,

            // some form of purity, definitely need to check
            Left(_) => (),

            // Unsafe trumped. To see if the unsafe is necessary, see what the
            // purity would have been without a trump, and if it's some form
            // of purity then we need to go ahead with the check
            Right(pc_unsafe) => {
                match do with(&mut self.declared_purity.purity,
                              ast::impure_fn) { self.purity(expr.id) } {
                    Right(pc_unsafe) => fail!(~"unsafe can't trump twice"),
                    Right(pc_default) => return,
                    Left(_) => ()
                }
            }

        }
        self.check_pure_callee_or_arg(
            pc, callee, callee_id, callee_span);
        for args.each |arg| {
            self.check_pure_callee_or_arg(
                pc, Some(*arg), arg.id, arg.span);
        }
    }
}

fn check_loans_in_fn(fk: &visit::fn_kind,
                     decl: &ast::fn_decl,
                     body: &ast::blk,
                     sp: span,
                     id: ast::node_id,
                     self: @mut CheckLoanCtxt,
                     visitor: visit::vt<@mut CheckLoanCtxt>) {
    let is_stack_closure = self.is_stack_closure(id);
    let fty = ty::node_id_to_type(self.tcx(), id);

    let declared_purity, src;
    match *fk {
        visit::fk_item_fn(*) | visit::fk_method(*) => {
            declared_purity = ty::ty_fn_purity(fty);
            src = id;
        }

        visit::fk_anon(*) | visit::fk_fn_block(*) => {
            let fty_sigil = ty::ty_closure_sigil(fty);
            check_moves_from_captured_variables(self, id, fty_sigil);
            let pair = ty::determine_inherited_purity(
                (self.declared_purity.purity, self.declared_purity.def),
                (ty::ty_fn_purity(fty), id),
                fty_sigil);
            declared_purity = pair.first();
            src = pair.second();
        }
    }

    debug!("purity on entry=%?", copy self.declared_purity);
    do save_and_restore_managed(self.declared_purity) {
        do save_and_restore_managed(self.fn_args) {
            self.declared_purity = @mut PurityState::function(declared_purity, src);

            match *fk {
                visit::fk_anon(*) |
                visit::fk_fn_block(*) if is_stack_closure => {
                    // inherits the fn_args from enclosing ctxt
                }
                visit::fk_anon(*) | visit::fk_fn_block(*) |
                visit::fk_method(*) | visit::fk_item_fn(*) => {
                    let mut fn_args = ~[];
                    for decl.inputs.each |input| {
                        // For the purposes of purity, only consider function-
                        // typed bindings in trivial patterns to be function
                        // arguments. For example, do not allow `f` and `g` in
                        // (f, g): (&fn(), &fn()) to be called.
                        match input.pat.node {
                            ast::pat_ident(_, _, None) => {
                                fn_args.push(input.pat.id);
                            }
                            _ => {} // Ignore this argument.
                        }
                    }
                    *self.fn_args = @fn_args;
                }
            }

            visit::visit_fn(fk, decl, body, sp, id, self, visitor);
        }
    }
    debug!("purity on exit=%?", copy self.declared_purity);

    fn check_moves_from_captured_variables(self: @mut CheckLoanCtxt,
                                           id: ast::node_id,
                                           fty_sigil: ast::Sigil) {
        match fty_sigil {
            ast::ManagedSigil | ast::OwnedSigil => {
                let cap_vars = self.bccx.capture_map.get(&id);
                for cap_vars.each |cap_var| {
                    match cap_var.mode {
                        moves::CapRef | moves::CapCopy => { loop; }
                        moves::CapMove => { }
                    }
                    let def_id = ast_util::def_id_of_def(cap_var.def).node;
                    let ty = ty::node_id_to_type(self.tcx(), def_id);
                    let cmt = self.bccx.cat_def(id, cap_var.span,
                                                ty, cap_var.def);
                    let move_err = self.analyze_move_out_from_cmt(cmt);
                    match move_err {
                        MoveOk => {}
                        MoveFromIllegalCmt(move_cmt) => {
                            self.bccx.span_err(
                                cap_var.span,
                                fmt!("illegal by-move capture of %s",
                                     self.bccx.cmt_to_str(move_cmt)));
                        }
                        MoveWhileBorrowed(move_cmt, loan_cmt) => {
                            self.bccx.span_err(
                                cap_var.span,
                                fmt!("by-move capture of %s prohibited \
                                      due to outstanding loan",
                                     self.bccx.cmt_to_str(move_cmt)));
                            self.bccx.span_note(
                                loan_cmt.span,
                                fmt!("loan of %s granted here",
                                     self.bccx.cmt_to_str(loan_cmt)));
                        }
                    }
                }
            }

            ast::BorrowedSigil => {}
        }
    }
}

fn check_loans_in_local(local: @ast::local,
                        self: @mut CheckLoanCtxt,
                        vt: visit::vt<@mut CheckLoanCtxt>) {
    visit::visit_local(local, self, vt);
}

fn check_loans_in_expr(expr: @ast::expr,
                       self: @mut CheckLoanCtxt,
                       vt: visit::vt<@mut CheckLoanCtxt>) {
    debug!("check_loans_in_expr(expr=%?/%s)",
           expr.id, pprust::expr_to_str(expr, self.tcx().sess.intr()));

    self.check_for_conflicting_loans(expr.id);

    if self.bccx.moves_map.contains(&expr.id) {
        self.check_move_out_from_expr(expr);
    }

    match expr.node {
      ast::expr_swap(l, r) => {
        self.check_assignment(at_swap, l);
        self.check_assignment(at_swap, r);
      }
      ast::expr_assign(dest, _) |
      ast::expr_assign_op(_, dest, _) => {
        self.check_assignment(at_straight_up, dest);
      }
      ast::expr_call(f, ref args, _) => {
        self.check_call(expr, Some(f), f.id, f.span, *args);
      }
      ast::expr_method_call(_, _, _, ref args, _) => {
        self.check_call(expr, None, expr.callee_id, expr.span, *args);
      }
      ast::expr_index(_, rval) |
      ast::expr_binary(_, _, rval)
      if self.bccx.method_map.contains_key(&expr.id) => {
        self.check_call(expr,
                        None,
                        expr.callee_id,
                        expr.span,
                        ~[rval]);
      }
      ast::expr_unary(*) | ast::expr_index(*)
      if self.bccx.method_map.contains_key(&expr.id) => {
        self.check_call(expr,
                        None,
                        expr.callee_id,
                        expr.span,
                        ~[]);
      }
      ast::expr_match(*) => {
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
      }
      _ => { }
    }

    visit::visit_expr(expr, self, vt);
}

fn check_loans_in_block(blk: &ast::blk,
                        self: @mut CheckLoanCtxt,
                        vt: visit::vt<@mut CheckLoanCtxt>) {
    do save_and_restore_managed(self.declared_purity) {
        self.check_for_conflicting_loans(blk.node.id);

        *self.declared_purity = self.declared_purity.recurse(blk);
        visit::visit_block(blk, self, vt);
    }
}
