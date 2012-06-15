// ----------------------------------------------------------------------
// Checking loans
//
// Phase 2 of check: we walk down the tree and check that:
// 1. assignments are always made to mutable locations;
// 2. loans made in overlapping scopes do not conflict
// 3. assignments do not affect things loaned out as immutable
// 4. moves to dnot affect things loaned out in any way

import dvec::{dvec, extensions};
import categorization::public_methods;

export check_loans;

enum check_loan_ctxt = @{
    bccx: borrowck_ctxt,
    req_maps: req_maps,

    reported: hashmap<ast::node_id, ()>,

    // Keep track of whether we're inside a ctor, so as to
    // allow mutating immutable fields in the same class if
    // we are in a ctor, we track the self id
    mut in_ctor: bool,
    mut declared_purity: ast::purity,
    mut fn_args: @[ast::node_id]
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

fn check_loans(bccx: borrowck_ctxt,
               req_maps: req_maps,
               crate: @ast::crate) {
    let clcx = check_loan_ctxt(@{bccx: bccx,
                                 req_maps: req_maps,
                                 reported: int_hash(),
                                 mut in_ctor: false,
                                 mut declared_purity: ast::impure_fn,
                                 mut fn_args: @[]});
    let vt = visit::mk_vt(@{visit_expr: check_loans_in_expr,
                            visit_block: check_loans_in_block,
                            visit_fn: check_loans_in_fn
                            with *visit::default_visitor()});
    visit::visit_crate(*crate, clcx, vt);
}

enum assignment_type {
    at_straight_up,
    at_swap,
    at_mutbl_ref,
}

impl methods for assignment_type {
    fn checked_by_liveness() -> bool {
        // the liveness pass guarantees that immutable local variables
        // are only assigned once; but it doesn't consider &mut
        alt self {
          at_straight_up {true}
          at_swap {true}
          at_mutbl_ref {false}
        }
    }
    fn ing_form(desc: str) -> str {
        alt self {
          at_straight_up { "assigning to " + desc }
          at_swap { "swapping to and from " + desc }
          at_mutbl_ref { "taking mut reference to " + desc }
        }
    }
}

impl methods for check_loan_ctxt {
    fn tcx() -> ty::ctxt { self.bccx.tcx }

    fn purity(scope_id: ast::node_id) -> option<purity_cause> {
        let default_purity = alt self.declared_purity {
          // an unsafe declaration overrides all
          ast::unsafe_fn { ret none; }

          // otherwise, remember what was declared as the
          // default, but we must scan for requirements
          // imposed by the borrow check
          ast::pure_fn { some(pc_pure_fn) }
          ast::crust_fn | ast::impure_fn { none }
        };

        // scan to see if this scope or any enclosing scope requires
        // purity.  if so, that overrides the declaration.

        let mut scope_id = scope_id;
        let region_map = self.tcx().region_map;
        let pure_map = self.req_maps.pure_map;
        loop {
            alt pure_map.find(scope_id) {
              none {}
              some(e) {ret some(pc_cmt(e));}
            }

            alt region_map.find(scope_id) {
              none { ret default_purity; }
              some(next_scope_id) { scope_id = next_scope_id; }
            }
        }
    }

    fn walk_loans(scope_id: ast::node_id,
                  f: fn(loan) -> bool) {
        let mut scope_id = scope_id;
        let region_map = self.tcx().region_map;
        let req_loan_map = self.req_maps.req_loan_map;

        loop {
            for req_loan_map.find(scope_id).each { |loanss|
                for (*loanss).each { |loans|
                    for (*loans).each { |loan|
                        if !f(loan) { ret; }
                    }
                }
            }

            alt region_map.find(scope_id) {
              none { ret; }
              some(next_scope_id) { scope_id = next_scope_id; }
            }
        }
    }

    fn walk_loans_of(scope_id: ast::node_id,
                     lp: @loan_path,
                     f: fn(loan) -> bool) {
        for self.walk_loans(scope_id) { |loan|
            if loan.lp == lp {
                if !f(loan) { ret; }
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
                                opt_expr: option<@ast::expr>,
                                callee_id: ast::node_id,
                                callee_span: span) {
        let tcx = self.tcx();

        #debug["check_pure_callee_or_arg(pc=%?, expr=%?, \
                callee_id=%d, ty=%s)",
               pc,
               opt_expr.map({|e| pprust::expr_to_str(e)}),
               callee_id,
               ty_to_str(self.tcx(), ty::node_id_to_type(tcx, callee_id))];

        // Purity rules: an expr B is a legal callee or argument to a
        // call within a pure function A if at least one of the
        // following holds:
        //
        // (a) A was declared pure and B is one of its arguments;
        // (b) B is a stack closure;
        // (c) B is a pure fn;
        // (d) B is not a fn.

        alt opt_expr {
          some(expr) {
            alt expr.node {
              ast::expr_path(_) if pc == pc_pure_fn {
                let def = self.tcx().def_map.get(expr.id);
                let did = ast_util::def_id_of_def(def);
                let is_fn_arg =
                    did.crate == ast::local_crate &&
                    (*self.fn_args).contains(did.node);
                if is_fn_arg { ret; } // case (a) above
              }
              ast::expr_fn_block(*) | ast::expr_fn(*) |
              ast::expr_loop_body(*) {
                if self.is_stack_closure(expr.id) { ret; } // case (b) above
              }
              _ {}
            }
          }
          none {}
        }

        let callee_ty = ty::node_id_to_type(tcx, callee_id);
        alt ty::get(callee_ty).struct {
          ty::ty_fn(fn_ty) {
            alt fn_ty.purity {
              ast::pure_fn { ret; } // case (c) above
              ast::impure_fn | ast::unsafe_fn | ast::crust_fn {
                self.report_purity_error(
                    pc, callee_span,
                    #fmt["access to %s function",
                         pprust::purity_to_str(fn_ty.purity)]);
              }
            }
          }
          _ { ret; } // case (d) above
        }
    }

    // True if the expression with the given `id` is a stack closure.
    // The expression must be an expr_fn(*) or expr_fn_block(*)
    fn is_stack_closure(id: ast::node_id) -> bool {
        let fn_ty = ty::node_id_to_type(self.tcx(), id);
        let proto = ty::ty_fn_proto(fn_ty);
        alt proto {
          ast::proto_block | ast::proto_any {true}
          ast::proto_bare | ast::proto_uniq | ast::proto_box {false}
        }
    }

    fn is_allowed_pure_arg(expr: @ast::expr) -> bool {
        ret alt expr.node {
          ast::expr_path(_) {
            let def = self.tcx().def_map.get(expr.id);
            let did = ast_util::def_id_of_def(def);
            did.crate == ast::local_crate &&
                (*self.fn_args).contains(did.node)
          }
          ast::expr_fn_block(*) | ast::expr_fn(*) {
            self.is_stack_closure(expr.id)
          }
          _ {false}
        };
    }

    fn check_for_conflicting_loans(scope_id: ast::node_id) {
        let new_loanss = alt self.req_maps.req_loan_map.find(scope_id) {
            none { ret; }
            some(loanss) { loanss }
        };

        let par_scope_id = self.tcx().region_map.get(scope_id);
        for self.walk_loans(par_scope_id) { |old_loan|
            for (*new_loanss).each { |new_loans|
                for (*new_loans).each { |new_loan|
                    if old_loan.lp != new_loan.lp { cont; }
                    alt (old_loan.mutbl, new_loan.mutbl) {
                      (m_const, _) | (_, m_const) |
                      (m_mutbl, m_mutbl) | (m_imm, m_imm) {
                        /*ok*/
                      }

                      (m_mutbl, m_imm) | (m_imm, m_mutbl) {
                        self.bccx.span_err(
                            new_loan.cmt.span,
                            #fmt["loan of %s as %s \
                                  conflicts with prior loan",
                                 self.bccx.cmt_to_str(new_loan.cmt),
                                 self.bccx.mut_to_str(new_loan.mutbl)]);
                        self.bccx.span_note(
                            old_loan.cmt.span,
                            #fmt["prior loan as %s granted here",
                                 self.bccx.mut_to_str(old_loan.mutbl)]);
                      }
                    }
                }
            }
        }
    }

    fn is_local_variable(cmt: cmt) -> bool {
        alt cmt.cat {
          cat_local(_) {true}
          _ {false}
        }
    }

    fn is_self_field(cmt: cmt) -> bool {
        alt cmt.cat {
          cat_comp(cmt_base, comp_field(*)) {
            alt cmt_base.cat {
              cat_special(sk_self) { true }
              _ { false }
            }
          }
          _ { false }
        }
    }

    fn check_assignment(at: assignment_type, ex: @ast::expr) {
        let cmt = self.bccx.cat_expr(ex);

        #debug["check_assignment(cmt=%s)",
               self.bccx.cmt_to_repr(cmt)];

        if self.in_ctor && self.is_self_field(cmt)
            && at.checked_by_liveness() {
            // assigning to self.foo in a ctor is always allowed.
        } else if self.is_local_variable(cmt) && at.checked_by_liveness() {
            // liveness guarantees that immutable local variables
            // are only assigned once
        } else {
            alt cmt.mutbl {
              m_mutbl { /*ok*/ }
              m_const | m_imm {
                self.bccx.span_err(
                    ex.span,
                    at.ing_form(self.bccx.cmt_to_str(cmt)));
                ret;
              }
            }
        }

        // if this is a pure function, only loan-able state can be
        // assigned, because it is uniquely tied to this function and
        // is not visible from the outside
        alt self.purity(ex.id) {
          none {}
          some(pc) {
            if cmt.lp.is_none() {
                self.report_purity_error(
                    pc, ex.span, at.ing_form(self.bccx.cmt_to_str(cmt)));
            }
          }
        }

        // check for a conflicting loan as well, except in the case of
        // taking a mutable ref.  that will create a loan of its own
        // which will be checked for compat separately in
        // check_for_conflicting_loans()
        if at != at_mutbl_ref {
            for cmt.lp.each { |lp|
                self.check_for_loan_conflicting_with_assignment(
                    at, ex, cmt, lp);
            }
        }

        self.bccx.add_to_mutbl_map(cmt);
    }

    fn check_for_loan_conflicting_with_assignment(
        at: assignment_type,
        ex: @ast::expr,
        cmt: cmt,
        lp: @loan_path) {

        for self.walk_loans_of(ex.id, lp) { |loan|
            alt loan.mutbl {
              m_mutbl | m_const { /*ok*/ }
              m_imm {
                self.bccx.span_err(
                    ex.span,
                    #fmt["%s prohibited due to outstanding loan",
                         at.ing_form(self.bccx.cmt_to_str(cmt))]);
                self.bccx.span_note(
                    loan.cmt.span,
                    #fmt["loan of %s granted here",
                         self.bccx.cmt_to_str(loan.cmt)]);
                ret;
              }
            }
        }

        // Subtle: if the mutability of the component being assigned
        // is inherited from the thing that the component is embedded
        // within, then we have to check whether that thing has been
        // loaned out as immutable!  An example:
        //    let mut x = {f: some(3)};
        //    let y = &x; // x loaned out as immutable
        //    x.f = none; // changes type of y.f, which appears to be imm
        alt *lp {
          lp_comp(lp_base, ck) if inherent_mutability(ck) != m_mutbl {
            self.check_for_loan_conflicting_with_assignment(
                at, ex, cmt, lp_base);
          }
          lp_comp(*) | lp_local(*) | lp_arg(*) | lp_deref(*) {}
        }
    }

    fn report_purity_error(pc: purity_cause, sp: span, msg: str) {
        alt pc {
          pc_pure_fn {
            self.tcx().sess.span_err(
                sp,
                #fmt["%s prohibited in pure context", msg]);
          }
          pc_cmt(e) {
            if self.reported.insert(e.cmt.id, ()) {
                self.tcx().sess.span_err(
                    e.cmt.span,
                    #fmt["illegal borrow unless pure: %s",
                         self.bccx.bckerr_code_to_str(e.code)]);
                self.tcx().sess.span_note(
                    sp,
                    #fmt["impure due to %s", msg]);
            }
          }
        }
    }

    fn check_move_out(ex: @ast::expr) {
        let cmt = self.bccx.cat_expr(ex);
        self.check_move_out_from_cmt(cmt);
    }

    fn check_move_out_from_cmt(cmt: cmt) {
        #debug["check_move_out_from_cmt(cmt=%s)",
               self.bccx.cmt_to_repr(cmt)];

        alt cmt.cat {
          // Rvalues, locals, and arguments can be moved:
          cat_rvalue | cat_local(_) | cat_arg(_) { }

          // We allow moving out of static items because the old code
          // did.  This seems consistent with permitting moves out of
          // rvalues, I guess.
          cat_special(sk_static_item) { }

          // Nothing else.
          _ {
            self.bccx.span_err(
                cmt.span,
                #fmt["moving out of %s", self.bccx.cmt_to_str(cmt)]);
            ret;
          }
        }

        self.bccx.add_to_mutbl_map(cmt);

        // check for a conflicting loan:
        let lp = alt cmt.lp {
          none { ret; }
          some(lp) { lp }
        };
        for self.walk_loans_of(cmt.id, lp) { |loan|
            self.bccx.span_err(
                cmt.span,
                #fmt["moving out of %s prohibited due to outstanding loan",
                     self.bccx.cmt_to_str(cmt)]);
            self.bccx.span_note(
                loan.cmt.span,
                #fmt["loan of %s granted here",
                     self.bccx.cmt_to_str(loan.cmt)]);
            ret;
        }
    }

    fn check_call(expr: @ast::expr,
                  callee: option<@ast::expr>,
                  callee_id: ast::node_id,
                  callee_span: span,
                  args: [@ast::expr]) {
        alt self.purity(expr.id) {
          none {}
          some(pc) {
            self.check_pure_callee_or_arg(
                pc, callee, callee_id, callee_span);
            for args.each { |arg|
                self.check_pure_callee_or_arg(
                    pc, some(arg), arg.id, arg.span);
            }
          }
        }
        let arg_tys =
            ty::ty_fn_args(
                ty::node_id_to_type(self.tcx(), callee_id));
        vec::iter2(args, arg_tys) { |arg, arg_ty|
            alt ty::resolved_mode(self.tcx(), arg_ty.mode) {
              ast::by_move {
                self.check_move_out(arg);
              }
              ast::by_mutbl_ref {
                self.check_assignment(at_mutbl_ref, arg);
              }
              ast::by_ref | ast::by_copy | ast::by_val {
              }
            }
        }
    }
}

fn check_loans_in_fn(fk: visit::fn_kind, decl: ast::fn_decl, body: ast::blk,
                     sp: span, id: ast::node_id, &&self: check_loan_ctxt,
                     visitor: visit::vt<check_loan_ctxt>) {

    #debug["purity on entry=%?", copy self.declared_purity];
    save_and_restore(self.in_ctor) {||
        save_and_restore(self.declared_purity) {||
            save_and_restore(self.fn_args) {||
                let is_stack_closure = self.is_stack_closure(id);

                // In principle, we could consider fk_anon(*) or
                // fk_fn_block(*) to be in a ctor, I suppose, but the
                // purpose of the in_ctor flag is to allow modifications
                // of otherwise immutable fields and typestate wouldn't be
                // able to "see" into those functions anyway, so it
                // wouldn't be very helpful.
                alt fk {
                  visit::fk_ctor(*) {
                    self.in_ctor = true;
                    self.declared_purity = decl.purity;
                    self.fn_args = @decl.inputs.map({|i| i.id});
                  }
                  visit::fk_anon(*) |
                  visit::fk_fn_block(*) if is_stack_closure {
                    self.in_ctor = false;
                    // inherits the purity/fn_args from enclosing ctxt
                  }
                  visit::fk_anon(*) | visit::fk_fn_block(*) |
                  visit::fk_method(*) | visit::fk_item_fn(*) |
                  visit::fk_res(*) | visit::fk_dtor(*) {
                    self.in_ctor = false;
                    self.declared_purity = decl.purity;
                    self.fn_args = @decl.inputs.map({|i| i.id});
                  }
                }

                visit::visit_fn(fk, decl, body, sp, id, self, visitor);
            }
        }
    }
    #debug["purity on exit=%?", copy self.declared_purity];
}

fn check_loans_in_expr(expr: @ast::expr,
                       &&self: check_loan_ctxt,
                       vt: visit::vt<check_loan_ctxt>) {
    self.check_for_conflicting_loans(expr.id);

    alt expr.node {
      ast::expr_swap(l, r) {
        self.check_assignment(at_swap, l);
        self.check_assignment(at_swap, r);
      }
      ast::expr_move(dest, src) {
        self.check_assignment(at_straight_up, dest);
        self.check_move_out(src);
      }
      ast::expr_assign(dest, _) |
      ast::expr_assign_op(_, dest, _) {
        self.check_assignment(at_straight_up, dest);
      }
      ast::expr_fn(_, _, _, cap_clause) |
      ast::expr_fn_block(_, _, cap_clause) {
        for (*cap_clause).each { |cap_item|
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
      ast::expr_addr_of(mutbl, base) {
        alt mutbl {
          m_const { /*all memory is const*/ }
          m_mutbl {
            // If we are taking an &mut ptr, make sure the memory
            // being pointed at is assignable in the first place:
            self.check_assignment(at_mutbl_ref, base);
          }
          m_imm {
            // XXX explain why no check is req'd here
          }
        }
      }
      ast::expr_call(f, args, _) {
        self.check_call(expr, some(f), f.id, f.span, args);
      }
      ast::expr_index(_, rval) |
      ast::expr_binary(_, _, rval)
      if self.bccx.method_map.contains_key(expr.id) {
        self.check_call(expr,
                        none,
                        ast_util::op_expr_callee_id(expr),
                        expr.span,
                        [rval]);
      }
      ast::expr_unary(*) | ast::expr_index(*)
      if self.bccx.method_map.contains_key(expr.id) {
        self.check_call(expr,
                        none,
                        ast_util::op_expr_callee_id(expr),
                        expr.span,
                        []);
      }
      _ { }
    }

    visit::visit_expr(expr, self, vt);
}

fn check_loans_in_block(blk: ast::blk,
                        &&self: check_loan_ctxt,
                        vt: visit::vt<check_loan_ctxt>) {
    save_and_restore(self.declared_purity) {||
        self.check_for_conflicting_loans(blk.node.id);

        alt blk.node.rules {
          ast::default_blk {
          }
          ast::unchecked_blk {
            self.declared_purity = ast::impure_fn;
          }
          ast::unsafe_blk {
            self.declared_purity = ast::unsafe_fn;
          }
        }

        visit::visit_block(blk, self, vt);
    }
}

