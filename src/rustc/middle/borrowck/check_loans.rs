// ----------------------------------------------------------------------
// Checking loans
//
// Phase 2 of check: we walk down the tree and check that:
// 1. assignments are always made to mutable locations;
// 2. loans made in overlapping scopes do not conflict
// 3. assignments do not affect things loaned out as immutable
// 4. moves to dnot affect things loaned out in any way

use dvec::DVec;

export check_loans;

enum check_loan_ctxt = @{
    bccx: borrowck_ctxt,
    req_maps: req_maps,

    reported: HashMap<ast::node_id, ()>,

    // Keep track of whether we're inside a ctor, so as to
    // allow mutating immutable fields in the same class if
    // we are in a ctor, we track the self id
    mut in_ctor: bool,
    mut declared_purity: ast::purity,
    mut fn_args: @~[ast::node_id]
};

// if we are enforcing purity, why are we doing so?
enum purity_cause {
    // enforcing purity because fn was declared pure:
    pc_pure_fn,

    // enforce purity because we need to guarantee the
    // validity of some alias; `bckerr` describes the
    // reason we needed to enforce purity.
    pc_cmt(bckerr)
}

impl purity_cause : cmp::Eq {
    pure fn eq(other: &purity_cause) -> bool {
        match self {
            pc_pure_fn => {
                match (*other) {
                    pc_pure_fn => true,
                    _ => false
                }
            }
            pc_cmt(e0a) => {
                match (*other) {
                    pc_cmt(e0b) => e0a == e0b,
                    _ => false
                }
            }
        }
    }
    pure fn ne(other: &purity_cause) -> bool { !self.eq(other) }
}

fn check_loans(bccx: borrowck_ctxt,
               req_maps: req_maps,
               crate: @ast::crate) {
    let clcx = check_loan_ctxt(@{bccx: bccx,
                                 req_maps: req_maps,
                                 reported: HashMap(),
                                 mut in_ctor: false,
                                 mut declared_purity: ast::impure_fn,
                                 mut fn_args: @~[]});
    let vt = visit::mk_vt(@{visit_expr: check_loans_in_expr,
                            visit_local: check_loans_in_local,
                            visit_block: check_loans_in_block,
                            visit_fn: check_loans_in_fn,
                            .. *visit::default_visitor()});
    visit::visit_crate(*crate, clcx, vt);
}

enum assignment_type {
    at_straight_up,
    at_swap
}

impl assignment_type : cmp::Eq {
    pure fn eq(other: &assignment_type) -> bool {
        (self as uint) == ((*other) as uint)
    }
    pure fn ne(other: &assignment_type) -> bool { !self.eq(other) }
}

impl assignment_type {
    fn checked_by_liveness() -> bool {
        // the liveness pass guarantees that immutable local variables
        // are only assigned once; but it doesn't consider &mut
        match self {
          at_straight_up => true,
          at_swap => true
        }
    }
    fn ing_form(desc: ~str) -> ~str {
        match self {
          at_straight_up => ~"assigning to " + desc,
          at_swap => ~"swapping to and from " + desc
        }
    }
}

impl check_loan_ctxt {
    fn tcx() -> ty::ctxt { self.bccx.tcx }

    fn purity(scope_id: ast::node_id) -> Option<purity_cause> {
        let default_purity = match self.declared_purity {
          // an unsafe declaration overrides all
          ast::unsafe_fn => return None,

          // otherwise, remember what was declared as the
          // default, but we must scan for requirements
          // imposed by the borrow check
          ast::pure_fn => Some(pc_pure_fn),
          ast::extern_fn | ast::impure_fn => None
        };

        // scan to see if this scope or any enclosing scope requires
        // purity.  if so, that overrides the declaration.

        let mut scope_id = scope_id;
        let region_map = self.tcx().region_map;
        let pure_map = self.req_maps.pure_map;
        loop {
            match pure_map.find(scope_id) {
              None => (),
              Some(e) => return Some(pc_cmt(e))
            }

            match region_map.find(scope_id) {
              None => return default_purity,
              Some(next_scope_id) => scope_id = next_scope_id
            }
        }
    }

    fn walk_loans(scope_id: ast::node_id,
                  f: fn(v: &loan) -> bool) {
        let mut scope_id = scope_id;
        let region_map = self.tcx().region_map;
        let req_loan_map = self.req_maps.req_loan_map;

        loop {
            for req_loan_map.find(scope_id).each |loanss| {
                for loanss.each |loans| {
                    for loans.each |loan| {
                        if !f(loan) { return; }
                    }
                }
            }

            match region_map.find(scope_id) {
              None => return,
              Some(next_scope_id) => scope_id = next_scope_id,
            }
        }
    }

    fn walk_loans_of(scope_id: ast::node_id,
                     lp: @loan_path,
                     f: fn(v: &loan) -> bool) {
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
    fn check_pure_callee_or_arg(pc: purity_cause,
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
              ast::expr_path(_) if pc == pc_pure_fn => {
                let def = self.tcx().def_map.get(expr.id);
                let did = ast_util::def_id_of_def(def);
                let is_fn_arg =
                    did.crate == ast::local_crate &&
                    (*self.fn_args).contains(&(did.node));
                if is_fn_arg { return; } // case (a) above
              }
              ast::expr_fn_block(*) | ast::expr_fn(*) |
              ast::expr_loop_body(*) | ast::expr_do_body(*) => {
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
          ty::ty_fn(fn_ty) => {
            match fn_ty.meta.purity {
              ast::pure_fn => return, // case (c) above
              ast::impure_fn | ast::unsafe_fn | ast::extern_fn => {
                self.report_purity_error(
                    pc, callee_span,
                    fmt!("access to %s function",
                         pprust::purity_to_str(fn_ty.meta.purity)));
              }
            }
          }
          _ => return, // case (d) above
        }
    }

    // True if the expression with the given `id` is a stack closure.
    // The expression must be an expr_fn(*) or expr_fn_block(*)
    fn is_stack_closure(id: ast::node_id) -> bool {
        let fn_ty = ty::node_id_to_type(self.tcx(), id);
        let proto = ty::ty_fn_proto(fn_ty);
        return ty::is_blockish(proto);
    }

    fn is_allowed_pure_arg(expr: @ast::expr) -> bool {
        return match expr.node {
          ast::expr_path(_) => {
            let def = self.tcx().def_map.get(expr.id);
            let did = ast_util::def_id_of_def(def);
            did.crate == ast::local_crate &&
                (*self.fn_args).contains(&(did.node))
          }
          ast::expr_fn_block(*) | ast::expr_fn(*) => {
            self.is_stack_closure(expr.id)
          }
          _ => false
        };
    }

    fn check_for_conflicting_loans(scope_id: ast::node_id) {
        let new_loanss = match self.req_maps.req_loan_map.find(scope_id) {
            None => return,
            Some(loanss) => loanss
        };

        let par_scope_id = self.tcx().region_map.get(scope_id);
        for self.walk_loans(par_scope_id) |old_loan| {
            for new_loanss.each |new_loans| {
                for new_loans.each |new_loan| {
                    if old_loan.lp != new_loan.lp { loop; }
                    match (old_loan.mutbl, new_loan.mutbl) {
                      (m_const, _) | (_, m_const) |
                      (m_mutbl, m_mutbl) | (m_imm, m_imm) => {
                        /*ok*/
                      }

                      (m_mutbl, m_imm) | (m_imm, m_mutbl) => {
                        self.bccx.span_err(
                            new_loan.cmt.span,
                            fmt!("loan of %s as %s \
                                  conflicts with prior loan",
                                 self.bccx.cmt_to_str(new_loan.cmt),
                                 self.bccx.mut_to_str(new_loan.mutbl)));
                        self.bccx.span_note(
                            old_loan.cmt.span,
                            fmt!("prior loan as %s granted here",
                                 self.bccx.mut_to_str(old_loan.mutbl)));
                      }
                    }
                }
            }
        }
    }

    fn is_local_variable(cmt: cmt) -> bool {
        match cmt.cat {
          cat_local(_) => true,
          _ => false
        }
    }

    fn is_self_field(cmt: cmt) -> bool {
        match cmt.cat {
          cat_comp(cmt_base, comp_field(*)) => {
            match cmt_base.cat {
              cat_special(sk_self) => true,
              _ => false
            }
          }
          _ => false
        }
    }

    fn check_assignment(at: assignment_type, ex: @ast::expr) {
        let cmt = self.bccx.cat_expr(ex);

        debug!("check_assignment(cmt=%s)",
               self.bccx.cmt_to_repr(cmt));

        if self.in_ctor && self.is_self_field(cmt)
            && at.checked_by_liveness() {
            // assigning to self.foo in a ctor is always allowed.
        } else if self.is_local_variable(cmt) && at.checked_by_liveness() {
            // liveness guarantees that immutable local variables
            // are only assigned once
        } else {
            match cmt.mutbl {
              m_mutbl => { /*ok*/ }
              m_const | m_imm => {
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
        match self.purity(ex.id) {
          None => (),
          Some(pc @ pc_cmt(_)) => {
            // Subtle: Issue #3162.  If we are enforcing purity
            // because there is a reference to aliasable, mutable data
            // that we require to be immutable, we can't allow writes
            // even to data owned by the current stack frame.  This is
            // because that aliasable data might have been located on
            // the current stack frame, we don't know.
            self.report_purity_error(
                pc, ex.span, at.ing_form(self.bccx.cmt_to_str(cmt)));
          }
          Some(pc_pure_fn) => {
            if cmt.lp.is_none() {
                self.report_purity_error(
                    pc_pure_fn, ex.span,
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
    }

    fn check_for_loan_conflicting_with_assignment(
        at: assignment_type,
        ex: @ast::expr,
        cmt: cmt,
        lp: @loan_path) {

        for self.walk_loans_of(ex.id, lp) |loan| {
            match loan.mutbl {
              m_mutbl | m_const => { /*ok*/ }
              m_imm => {
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
          lp_comp(*) | lp_local(*) | lp_arg(*) | lp_deref(*) => ()
        }
    }

    fn report_purity_error(pc: purity_cause, sp: span, msg: ~str) {
        match pc {
          pc_pure_fn => {
            self.tcx().sess.span_err(
                sp,
                fmt!("%s prohibited in pure context", msg));
          }
          pc_cmt(e) => {
            if self.reported.insert(e.cmt.id, ()) {
                self.tcx().sess.span_err(
                    e.cmt.span,
                    fmt!("illegal borrow unless pure: %s",
                         self.bccx.bckerr_to_str(e)));
                self.bccx.note_and_explain_bckerr(e);
                self.tcx().sess.span_note(
                    sp,
                    fmt!("impure due to %s", msg));
            }
          }
        }
    }

    fn check_move_out(ex: @ast::expr) {
        let cmt = self.bccx.cat_expr(ex);
        self.check_move_out_from_cmt(cmt);
    }

    fn check_move_out_from_cmt(cmt: cmt) {
        debug!("check_move_out_from_cmt(cmt=%s)",
               self.bccx.cmt_to_repr(cmt));

        match cmt.cat {
          // Rvalues, locals, and arguments can be moved:
          cat_rvalue | cat_local(_) | cat_arg(_) => {}

          // We allow moving out of static items because the old code
          // did.  This seems consistent with permitting moves out of
          // rvalues, I guess.
          cat_special(sk_static_item) => {}

          cat_deref(_, _, unsafe_ptr) => {}

          // Nothing else.
          _ => {
            self.bccx.span_err(
                cmt.span,
                fmt!("moving out of %s", self.bccx.cmt_to_str(cmt)));
            return;
          }
        }

        self.bccx.add_to_mutbl_map(cmt);

        // check for a conflicting loan:
        let lp = match cmt.lp {
          None => return,
          Some(lp) => lp
        };
        for self.walk_loans_of(cmt.id, lp) |loan| {
            self.bccx.span_err(
                cmt.span,
                fmt!("moving out of %s prohibited due to outstanding loan",
                     self.bccx.cmt_to_str(cmt)));
            self.bccx.span_note(
                loan.cmt.span,
                fmt!("loan of %s granted here",
                     self.bccx.cmt_to_str(loan.cmt)));
            return;
        }
    }

    // Very subtle (#2633): liveness can mark options as last_use even
    // when there is an outstanding loan.  In that case, it is not
    // safe to consider the use a last_use.
    fn check_last_use(expr: @ast::expr) {
        debug!("Checking last use of expr %?", expr.id);
        let cmt = self.bccx.cat_expr(expr);
        let lp = match cmt.lp {
            None => {
                debug!("Not a loanable expression");
                return;
            }
            Some(lp) => lp
        };
        for self.walk_loans_of(cmt.id, lp) |_loan| {
            debug!("Removing last use entry %? due to outstanding loan",
                   expr.id);
            self.bccx.last_use_map.remove(expr.id);
            return;
        }
    }

    fn check_call(expr: @ast::expr,
                  callee: Option<@ast::expr>,
                  callee_id: ast::node_id,
                  callee_span: span,
                  args: ~[@ast::expr]) {
        match self.purity(expr.id) {
          None => {}
          Some(pc) => {
            self.check_pure_callee_or_arg(
                pc, callee, callee_id, callee_span);
            for args.each |arg| {
                self.check_pure_callee_or_arg(
                    pc, Some(*arg), arg.id, arg.span);
            }
          }
        }
        let arg_tys =
            ty::ty_fn_args(
                ty::node_id_to_type(self.tcx(), callee_id));
        do vec::iter2(args, arg_tys) |arg, arg_ty| {
            match ty::resolved_mode(self.tcx(), arg_ty.mode) {
                ast::by_move => {
                    self.check_move_out(*arg);
                }
                ast::by_mutbl_ref | ast::by_ref |
                ast::by_copy | ast::by_val => {
                }
            }
        }
    }
}

fn check_loans_in_fn(fk: visit::fn_kind, decl: ast::fn_decl, body: ast::blk,
                     sp: span, id: ast::node_id, &&self: check_loan_ctxt,
                     visitor: visit::vt<check_loan_ctxt>) {

    debug!("purity on entry=%?", copy self.declared_purity);
    do save_and_restore(self.in_ctor) {
        do save_and_restore(self.declared_purity) {
            do save_and_restore(self.fn_args) {
                let is_stack_closure = self.is_stack_closure(id);
                let fty = ty::node_id_to_type(self.tcx(), id);
                self.declared_purity = ty::determine_inherited_purity(
                    copy self.declared_purity,
                    ty::ty_fn_purity(fty),
                    ty::ty_fn_proto(fty));

                // In principle, we could consider fk_anon(*) or
                // fk_fn_block(*) to be in a ctor, I suppose, but the
                // purpose of the in_ctor flag is to allow modifications
                // of otherwise immutable fields and typestate wouldn't be
                // able to "see" into those functions anyway, so it
                // wouldn't be very helpful.
                match fk {
                  visit::fk_ctor(*) => {
                    self.in_ctor = true;
                    self.fn_args = @decl.inputs.map(|i| i.id );
                  }
                  visit::fk_anon(*) |
                  visit::fk_fn_block(*) if is_stack_closure => {
                    self.in_ctor = false;
                    // inherits the fn_args from enclosing ctxt
                  }
                  visit::fk_anon(*) | visit::fk_fn_block(*) |
                  visit::fk_method(*) | visit::fk_item_fn(*) |
                  visit::fk_dtor(*) => {
                    self.in_ctor = false;
                    self.fn_args = @decl.inputs.map(|i| i.id );
                  }
                }

                visit::visit_fn(fk, decl, body, sp, id, self, visitor);
            }
        }
    }
    debug!("purity on exit=%?", copy self.declared_purity);
}

fn check_loans_in_local(local: @ast::local,
                        &&self: check_loan_ctxt,
                        vt: visit::vt<check_loan_ctxt>) {
    match local.node.init {
      Some({op: ast::init_move, expr: expr}) => {
        self.check_move_out(expr);
      }
      Some({op: ast::init_assign, _}) | None => {}
    }
    visit::visit_local(local, self, vt);
}

fn check_loans_in_expr(expr: @ast::expr,
                       &&self: check_loan_ctxt,
                       vt: visit::vt<check_loan_ctxt>) {
    debug!("check_loans_in_expr(expr=%?/%s)",
           expr.id, pprust::expr_to_str(expr, self.tcx().sess.intr()));

    self.check_for_conflicting_loans(expr.id);

    match expr.node {
      ast::expr_path(*) if self.bccx.last_use_map.contains_key(expr.id) => {
        self.check_last_use(expr);
      }

      ast::expr_swap(l, r) => {
        self.check_assignment(at_swap, l);
        self.check_assignment(at_swap, r);
      }
      ast::expr_move(dest, src) => {
        self.check_assignment(at_straight_up, dest);
        self.check_move_out(src);
      }
      ast::expr_unary_move(src) => {
        self.check_move_out(src);
      }
      ast::expr_assign(dest, _) |
      ast::expr_assign_op(_, dest, _) => {
        self.check_assignment(at_straight_up, dest);
      }
      ast::expr_fn(_, _, _, cap_clause) |
      ast::expr_fn_block(_, _, cap_clause) => {
        for (*cap_clause).each |cap_item| {
            if cap_item.is_move {
                let def = self.tcx().def_map.get(cap_item.id);

                // Hack: the type that is used in the cmt doesn't actually
                // matter here, so just subst nil instead of looking up
                // the type of the def that is referred to
                let cmt = self.bccx.cat_def(cap_item.id, cap_item.span,
                                            ty::mk_nil(self.tcx()), def);
                self.check_move_out_from_cmt(cmt);
            }
        }
      }
      ast::expr_call(f, args, _) => {
        self.check_call(expr, Some(f), f.id, f.span, args);
      }
      ast::expr_index(_, rval) |
      ast::expr_binary(_, _, rval)
      if self.bccx.method_map.contains_key(expr.id) => {
        self.check_call(expr,
                        None,
                        expr.callee_id,
                        expr.span,
                        ~[rval]);
      }
      ast::expr_unary(*) | ast::expr_index(*)
      if self.bccx.method_map.contains_key(expr.id) => {
        self.check_call(expr,
                        None,
                        expr.callee_id,
                        expr.span,
                        ~[]);
      }
      _ => { }
    }

    visit::visit_expr(expr, self, vt);
}

fn check_loans_in_block(blk: ast::blk,
                        &&self: check_loan_ctxt,
                        vt: visit::vt<check_loan_ctxt>) {
    do save_and_restore(self.declared_purity) {
        self.check_for_conflicting_loans(blk.node.id);

        match blk.node.rules {
          ast::default_blk => {
          }
          ast::unsafe_blk => {
            self.declared_purity = ast::unsafe_fn;
          }
        }

        visit::visit_block(blk, self, vt);
    }
}

