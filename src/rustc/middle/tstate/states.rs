use ann::*;
use aux::*;
use tritv::*;

use syntax::print::pprust::block_to_str;
use bitvectors::*;
use pat_util::*;
use syntax::ast::*;
use syntax::ast_util::*;
use syntax::print::pprust::{expr_to_str, stmt_to_str};
use syntax::codemap::span;
use middle::ty::{expr_ty, type_is_bot};
use util::common::{field_exprs, has_nonlocal_exits, may_break};
use driver::session::session;
use std::map::hashmap;

fn forbid_upvar(fcx: fn_ctxt, rhs_id: node_id, sp: span, t: oper_type) {
    match t {
      oper_move {
        match local_node_id_to_def(fcx, rhs_id) {
          Some(def_upvar(_, _, _)) {
            fcx.ccx.tcx.sess.span_err(sp,
                                      ~"tried to deinitialize a variable \
              declared in a different scope");
          }
          _ { }
        }
      }
      _ {/* do nothing */ }
    }
}

fn handle_move_or_copy(fcx: fn_ctxt, post: poststate, rhs_path: @path,
                       rhs_id: node_id, destlhs: dest, init_op: init_op) {
    forbid_upvar(fcx, rhs_id, rhs_path.span, op_to_oper_ty(init_op));

    let rhs_d_id = local_node_id_to_def_id(fcx, rhs_id);
    match rhs_d_id {
      Some(rhsid) {
        // RHS is a local var
        let instrhs =
            {ident: path_to_ident(rhs_path), node: rhsid.node};
        match destlhs {
          local_dest(instlhs) {
             copy_in_poststate(fcx, post, instlhs, instrhs,
                               op_to_oper_ty(init_op));
          }
          _ {}
        }
      }
      _ {
        // not a local -- do nothing
      }
    }
}

fn seq_states(fcx: fn_ctxt, pres: prestate, bindings: ~[binding]) ->
   {changed: bool, post: poststate} {
    let mut changed = false;
    let mut post = pres.clone();
    for bindings.each |b| {
        match b.rhs {
          Some(an_init) {
            // an expression, with or without a destination
            changed |=
                find_pre_post_state_expr(fcx, post, an_init.expr) || changed;
            post = expr_poststate(fcx.ccx, an_init.expr).clone();
            for b.lhs.each |d| {
                match an_init.expr.node {
                  expr_path(p) {
                    handle_move_or_copy(fcx, post, p, an_init.expr.id, d,
                                        an_init.op);
                  }
                  _ { }
                }
            }

            // Forget the RHS if we just moved it.
            if an_init.op == init_move {
                forget_in_poststate(fcx, post, an_init.expr.id);
            }
          }
          none {
          }
        }
    }
    return {changed: changed, post: post};
}

fn find_pre_post_state_sub(fcx: fn_ctxt, pres: prestate, e: @expr,
                           parent: node_id, c: Option<tsconstr>) -> bool {
    let mut changed = find_pre_post_state_expr(fcx, pres, e);

    changed = set_prestate_ann(fcx.ccx, parent, pres) || changed;

    let post = expr_poststate(fcx.ccx, e).clone();
    match c {
      none { }
      Some(c1) { set_in_poststate_(bit_num(fcx, c1), post); }
    }

    changed = set_poststate_ann(fcx.ccx, parent, post) || changed;
    return changed;
}

fn find_pre_post_state_two(fcx: fn_ctxt, pres: prestate, lhs: @expr,
                           rhs: @expr, parent: node_id, ty: oper_type) ->
   bool {
    let mut changed = set_prestate_ann(fcx.ccx, parent, pres);
    changed = find_pre_post_state_expr(fcx, pres, lhs) || changed;
    changed =
        find_pre_post_state_expr(fcx, expr_poststate(fcx.ccx, lhs), rhs) ||
            changed;
    forbid_upvar(fcx, rhs.id, rhs.span, ty);

    let post = expr_poststate(fcx.ccx, rhs).clone();

    match lhs.node {
      expr_path(p) {
        // for termination, need to make sure intermediate changes don't set
        // changed flag
        // tmp remembers "old" constraints we'd otherwise forget,
        // for substitution purposes
        let tmp = post.clone();

        match ty {
          oper_move {
            if is_path(rhs) { forget_in_poststate(fcx, post, rhs.id); }
            forget_in_poststate(fcx, post, lhs.id);
          }
          oper_swap {
            forget_in_poststate(fcx, post, lhs.id);
            forget_in_poststate(fcx, post, rhs.id);
          }
          _ { forget_in_poststate(fcx, post, lhs.id); }
        }

        match rhs.node {
          expr_path(p1) {
            let d = local_node_id_to_local_def_id(fcx, lhs.id);
            let d1 = local_node_id_to_local_def_id(fcx, rhs.id);
            match d {
              Some(id) {
                match d1 {
                  Some(id1) {
                    let instlhs =
                        {ident: path_to_ident(p), node: id};
                    let instrhs =
                        {ident: path_to_ident(p1), node: id1};
                    copy_in_poststate_two(fcx, tmp, post, instlhs, instrhs,
                                          ty);
                  }
                  _ { }
                }
              }
              _ { }
            }
          }
          _ {/* do nothing */ }
        }
      }
      _ { }
    }
    changed = set_poststate_ann(fcx.ccx, parent, post) || changed;
    return changed;
}

fn find_pre_post_state_call(fcx: fn_ctxt, pres: prestate, a: @expr,
                            id: node_id, ops: ~[init_op], bs: ~[@expr],
                            cf: ret_style) -> bool {
    let mut changed = find_pre_post_state_expr(fcx, pres, a);
    // FIXME (#2178): This could be a typestate constraint (except we're
    // not using them inside the compiler, I guess... see discussion in
    // bug)
    if vec::len(bs) != vec::len(ops) {
        fcx.ccx.tcx.sess.span_bug(a.span,
                                  fmt!("mismatched arg lengths: \
                                        %u exprs vs. %u ops",
                                       vec::len(bs), vec::len(ops)));
    }
    return find_pre_post_state_exprs(fcx, pres, id, ops,
                                   bs, cf) || changed;
}

fn find_pre_post_state_exprs(fcx: fn_ctxt, pres: prestate, id: node_id,
                             ops: ~[init_op], es: ~[@expr],
                             cf: ret_style) -> bool {
    let rs = seq_states(fcx, pres, arg_bindings(ops, es));
    let mut changed = rs.changed | set_prestate_ann(fcx.ccx, id, pres);
    /* if this is a failing call, it sets everything as initialized */
    match cf {
      noreturn {
        let post = false_postcond(num_constraints(fcx.enclosing));
        changed |= set_poststate_ann(fcx.ccx, id, post);
      }
      _ { changed |= set_poststate_ann(fcx.ccx, id, rs.post); }
    }
    return changed;
}

fn join_then_else(fcx: fn_ctxt, antec: @expr, conseq: blk,
                  maybe_alt: Option<@expr>, id: node_id, chk: if_ty,
                  pres: prestate) -> bool {
    let mut changed =
        set_prestate_ann(fcx.ccx, id, pres) |
            find_pre_post_state_expr(fcx, pres, antec);

    match maybe_alt {
      none {
        match chk {
          if_check {
            let c: sp_constr = expr_to_constr(fcx.ccx.tcx, antec);
            let conseq_prestate = expr_poststate(fcx.ccx, antec).clone();
            conseq_prestate.set(bit_num(fcx, c.node), ttrue);
            changed |=
                find_pre_post_state_block(fcx, conseq_prestate, conseq) |
                    set_poststate_ann(fcx.ccx, id,
                                      expr_poststate(fcx.ccx, antec));
          }
          _ {
            changed |=
                find_pre_post_state_block(fcx, expr_poststate(fcx.ccx, antec),
                                          conseq) |
                    set_poststate_ann(fcx.ccx, id,
                                      expr_poststate(fcx.ccx, antec));
          }
        }
      }
      Some(altern) {
        changed |=
            find_pre_post_state_expr(fcx, expr_poststate(fcx.ccx, antec),
                                     altern);

        let mut conseq_prestate = expr_poststate(fcx.ccx, antec);
        match chk {
          if_check {
            let c: sp_constr = expr_to_constr(fcx.ccx.tcx, antec);
            conseq_prestate = conseq_prestate.clone();
            conseq_prestate.set(bit_num(fcx, c.node),  ttrue);
          }
          _ { }
        }


        changed |= find_pre_post_state_block(fcx, conseq_prestate, conseq);

        let poststate_res =
            intersect_states(block_poststate(fcx.ccx, conseq),
                             expr_poststate(fcx.ccx, altern));
        /*
           fcx.ccx.tcx.sess.span_note(antec.span,
           "poststate_res = " + aux::tritv_to_str(fcx, poststate_res));
        fcx.ccx.tcx.sess.span_note(antec.span,
           "altern poststate = " +
            aux::tritv_to_str(fcx, expr_poststate(fcx.ccx, altern)));
        fcx.ccx.tcx.sess.span_note(antec.span,
        "conseq poststate = " + aux::tritv_to_str(fcx,
           block_poststate(fcx.ccx, conseq)));
        */

        changed |= set_poststate_ann(fcx.ccx, id, poststate_res);
      }
    }
    return changed;
}

fn find_pre_post_state_cap_clause(fcx: fn_ctxt, e_id: node_id,
                                  pres: prestate, cap_clause: capture_clause)
    -> bool
{
    let ccx = fcx.ccx;
    let pres_changed = set_prestate_ann(ccx, e_id, pres);
    let post = pres.clone();
    for (*cap_clause).each |cap_item| {
        if cap_item.is_move {
            forget_in_poststate(fcx, post, cap_item.id);
        }
    }
    return set_poststate_ann(ccx, e_id, post) || pres_changed;
}

fn find_pre_post_state_expr(fcx: fn_ctxt, pres: prestate, e: @expr) -> bool {
    let num_constrs = num_constraints(fcx.enclosing);

    match e.node {
      expr_new(p, _, v) {
        return find_pre_post_state_two(fcx, pres, p, v, e.id, oper_pure);
      }
      expr_vstore(ee, _) {
        let mut changed = find_pre_post_state_expr(fcx, pres, ee);
        set_prestate_ann(fcx.ccx, e.id, expr_prestate(fcx.ccx, ee));
        set_poststate_ann(fcx.ccx, e.id, expr_poststate(fcx.ccx, ee));
        return changed;
      }
      expr_vec(elts, _) {
        return find_pre_post_state_exprs(fcx, pres, e.id,
                                      vec::from_elem(vec::len(elts),
                                                    init_assign), elts,
                                      return_val);
      }
      expr_call(operator, operands, _) {
        debug!("hey it's a call: %s", expr_to_str(e));
        return find_pre_post_state_call(fcx, pres, operator, e.id,
                                     callee_arg_init_ops(fcx, operator.id),
                                     operands,
                                     controlflow_expr(fcx.ccx, operator));
      }
      expr_path(_) { return pure_exp(fcx.ccx, e.id, pres); }
      expr_log(_, lvl, ex) {
        return find_pre_post_state_two(fcx, pres, lvl, ex, e.id, oper_pure);
      }
      expr_mac(_) { fcx.ccx.tcx.sess.bug(~"unexpanded macro"); }
      expr_lit(l) { return pure_exp(fcx.ccx, e.id, pres); }
      expr_fn(_, _, _, cap_clause) {
        return find_pre_post_state_cap_clause(fcx, e.id, pres, cap_clause);
      }
      expr_fn_block(_, _, cap_clause) {
        return find_pre_post_state_cap_clause(fcx, e.id, pres, cap_clause);
      }
      expr_block(b) {
        return find_pre_post_state_block(fcx, pres, b) |
                set_prestate_ann(fcx.ccx, e.id, pres) |
                set_poststate_ann(fcx.ccx, e.id, block_poststate(fcx.ccx, b));
      }
      expr_rec(fields, maybe_base) {
        let exs = field_exprs(fields);
        let mut changed =
            find_pre_post_state_exprs(fcx, pres, e.id,
                                      vec::from_elem(vec::len(fields),
                                                    init_assign),
                                      exs, return_val);

        let base_pres = match vec::last_opt(exs) { none { pres }
                          Some(f) { expr_poststate(fcx.ccx, f) }};
        option::iter(maybe_base, |base| {
            changed |= find_pre_post_state_expr(fcx, base_pres, base) |
                set_poststate_ann(fcx.ccx, e.id,
                                  expr_poststate(fcx.ccx, base))
        });
        return changed;
      }
      expr_tup(elts) {
        return find_pre_post_state_exprs(fcx, pres, e.id,
                                      vec::from_elem(vec::len(elts),
                                                    init_assign), elts,
                                      return_val);
      }
      expr_move(lhs, rhs) {
        return find_pre_post_state_two(fcx, pres, lhs, rhs, e.id, oper_move);
      }
      expr_assign(lhs, rhs) {
        return find_pre_post_state_two(
            fcx, pres, lhs, rhs, e.id, oper_assign);
      }
      expr_swap(lhs, rhs) {
        return find_pre_post_state_two(fcx, pres, lhs, rhs, e.id, oper_swap);
        // Could be more precise and actually swap the role of
        // lhs and rhs in constraints
      }
      expr_ret(maybe_ret_val) {
        let mut changed = set_prestate_ann(fcx.ccx, e.id, pres);
        /* everything is true if execution continues after
           a return expression (since execution never continues locally
           after a return expression */
        let post = false_postcond(num_constrs);

        set_poststate_ann(fcx.ccx, e.id, post);

        match maybe_ret_val {
          none {/* do nothing */ }
          Some(ret_val) {
            changed |= find_pre_post_state_expr(fcx, pres, ret_val);
          }
        }
        return changed;
      }
      expr_if(antec, conseq, maybe_alt) {
        return join_then_else(fcx, antec, conseq, maybe_alt, e.id, plain_if,
                           pres);
      }
      expr_binary(bop, l, r) {
        if lazy_binop(bop) {
            let mut changed = find_pre_post_state_expr(fcx, pres, l);
            changed |=
                find_pre_post_state_expr(fcx, expr_poststate(fcx.ccx, l), r);
            return changed | set_prestate_ann(fcx.ccx, e.id, pres) |
                    set_poststate_ann(fcx.ccx, e.id,
                                      expr_poststate(fcx.ccx, l));
        } else {
            return find_pre_post_state_two(fcx, pres, l, r, e.id, oper_pure);
        }
      }
      expr_assign_op(op, lhs, rhs) {
        return find_pre_post_state_two(fcx, pres, lhs, rhs, e.id,
                                    oper_assign_op);
      }
      expr_while(test, body) {
        let loop_pres =
            intersect_states(block_poststate(fcx.ccx, body), pres);

        let mut changed =
            set_prestate_ann(fcx.ccx, e.id, loop_pres) |
                find_pre_post_state_expr(fcx, loop_pres, test) |
                find_pre_post_state_block(fcx, expr_poststate(fcx.ccx, test),
                                          body);

        /* conservative approximation: if a loop contains a break
           or cont, we assume nothing about the poststate */
        /* which is still unsound -- see ~[Break-unsound] */
        if has_nonlocal_exits(body) {
            return changed | set_poststate_ann(fcx.ccx, e.id, pres);
        } else {
            let e_post = expr_poststate(fcx.ccx, test);
            let b_post = block_poststate(fcx.ccx, body);
            return changed |
                    set_poststate_ann(fcx.ccx, e.id,
                                      intersect_states(e_post, b_post));
        }
      }
      expr_loop(body) {
        let loop_pres =
            intersect_states(block_poststate(fcx.ccx, body), pres);
        let mut changed = set_prestate_ann(fcx.ccx, e.id, loop_pres)
              | find_pre_post_state_block(fcx, loop_pres, body);
        /* conservative approximation: if a loop contains a break
           or cont, we assume nothing about the poststate (so, we
           set all predicates to "don't know" */
        /* which is still unsound -- see ~[Break-unsound] */
        if may_break(body) {
                /* Only do this if there are *breaks* not conts.
                 An infinite loop with conts is still an infinite loop.
                We assume all preds are FALSE, not '?' -- because in the
                worst case, the body could invalidate all preds and
                deinitialize everything before breaking */
            let post = empty_poststate(num_constrs);
            post.kill();
            return changed | set_poststate_ann(fcx.ccx, e.id, post);
        } else {
            return changed | set_poststate_ann(fcx.ccx, e.id,
                                            false_postcond(num_constrs));
        }
      }
      expr_index(val, sub) {
        return find_pre_post_state_two(fcx, pres, val, sub, e.id, oper_pure);
      }
      expr_match(val, alts, _) {
        let mut changed =
            set_prestate_ann(fcx.ccx, e.id, pres) |
                find_pre_post_state_expr(fcx, pres, val);
        let e_post = expr_poststate(fcx.ccx, val);
        let mut a_post;
        if vec::len(alts) > 0u {
            a_post = false_postcond(num_constrs);
            for alts.each |an_alt| {
                match an_alt.guard {
                  Some(e) {
                    changed |= find_pre_post_state_expr(fcx, e_post, e);
                  }
                  _ {}
                }
                changed |=
                    find_pre_post_state_block(fcx, e_post, an_alt.body);
                intersect(a_post, block_poststate(fcx.ccx, an_alt.body));
                // We deliberately do *not* update changed here, because
                // we'd go into an infinite loop that way, and the change
                // gets made after the if expression.

            }
        } else {
            // No alts; poststate is the poststate of the test

            a_post = e_post;
        }
        return changed | set_poststate_ann(fcx.ccx, e.id, a_post);
      }
      expr_field(x, _, _) | expr_loop_body(x) | expr_do_body(x) |
      expr_unary(_, x) |
      expr_addr_of(_, x) | expr_assert(x) | expr_cast(x, _) |
      expr_copy(x) {
        return find_pre_post_state_sub(fcx, pres, x, e.id, None);
      }
      expr_fail(maybe_fail_val) {
        /* if execution continues after fail, then everything is true!
        woo! */
        let post = false_postcond(num_constrs);
        return set_prestate_ann(fcx.ccx, e.id, pres) |
                set_poststate_ann(fcx.ccx, e.id, post) |
                option::map_default(
                    maybe_fail_val, false,
                    |fail_val|
                    find_pre_post_state_expr(fcx, pres, fail_val) );
      }
      expr_check(_, p) {
        /* predicate p holds after this expression executes */
        let c: sp_constr = expr_to_constr(fcx.ccx.tcx, p);
        return find_pre_post_state_sub(fcx, pres, p, e.id, Some(c.node));
      }
      expr_if_check(p, conseq, maybe_alt) {
        return join_then_else(
            fcx, p, conseq, maybe_alt, e.id, if_check, pres);
      }
      expr_break { return pure_exp(fcx.ccx, e.id, pres); }
      expr_again { return pure_exp(fcx.ccx, e.id, pres); }
    }
}

fn find_pre_post_state_stmt(fcx: fn_ctxt, pres: prestate, s: @stmt) -> bool {
    let stmt_ann = stmt_to_ann(fcx.ccx, *s);

    debug!("[ %s ]", *fcx.name);
    debug!("*At beginning: stmt = %s", stmt_to_str(*s));
    debug!("*prestate = %s", stmt_ann.states.prestate.to_str());
    debug!("*poststate = %s", stmt_ann.states.prestate.to_str());

    match s.node {
      stmt_decl(adecl, id) {
        match adecl.node {
          decl_local(alocals) {
            set_prestate(stmt_ann, pres);
            let c_and_p = seq_states(fcx, pres,
                  locals_to_bindings(fcx.ccx.tcx, alocals));
            /* important to do this in one step to ensure
            termination (don't want to set changed to true
            for intermediate changes) */

            let mut changed =
                set_poststate(stmt_ann, c_and_p.post) | c_and_p.changed;

            debug!("Summary: stmt = %s", stmt_to_str(*s));
            debug!("prestate = %s", stmt_ann.states.prestate.to_str());
            debug!("poststate = %s", stmt_ann.states.poststate.to_str());
            debug!("changed = %s", bool::to_str(changed));

            return changed;
          }
          decl_item(an_item) {
            return set_prestate(stmt_ann, pres)
                | set_poststate(stmt_ann, pres);
            /* the outer visitor will recurse into the item */
          }
        }
      }
      stmt_expr(ex, _) | stmt_semi(ex, _) {
        let mut changed =
            find_pre_post_state_expr(fcx, pres, ex) |
                set_prestate(stmt_ann, expr_prestate(fcx.ccx, ex)) |
                set_poststate(stmt_ann, expr_poststate(fcx.ccx, ex));


        debug!("Finally: %s", stmt_to_str(*s));
        debug!("prestate = %s", stmt_ann.states.prestate.to_str());
        debug!("poststate = %s", stmt_ann.states.poststate.to_str());
        debug!("changed = %s", bool::to_str(changed));

        return changed;
      }
    }
}


/* Updates the pre- and post-states of statements in the block,
   returns a boolean flag saying whether any pre- or poststates changed */
fn find_pre_post_state_block(fcx: fn_ctxt, pres0: prestate, b: blk) -> bool {
    /* First, set the pre-states and post-states for every expression */

    let mut pres = pres0;
    /* Iterate over each stmt. The new prestate is <pres>. The poststate
     consist of improving <pres> with whatever variables this stmt
     initializes.  Then <pres> becomes the new poststate. */

    let mut changed = false;
    for b.node.stmts.each |s| {
        changed |= find_pre_post_state_stmt(fcx, pres, s);
        pres = stmt_poststate(fcx.ccx, *s);
    }
    let mut post = pres;
    match b.node.expr {
      none { }
      Some(e) {
        changed |= find_pre_post_state_expr(fcx, pres, e);
        post = expr_poststate(fcx.ccx, e);
      }
    }

    set_prestate_ann(fcx.ccx, b.node.id, pres0);
    set_poststate_ann(fcx.ccx, b.node.id, post);

    return changed;
}

fn find_pre_post_state_fn(fcx: fn_ctxt,
                          f_decl: fn_decl,
                          f_body: blk) -> bool {
    // All constraints are considered false until proven otherwise.
    // This ensures that intersect works correctly.
    kill_all_prestate(fcx, f_body.node.id);

    // Instantiate any constraints on the arguments so we can use them
    let block_pre = block_prestate(fcx.ccx, f_body);
    for f_decl.constraints.each |c| {
        let tsc = ast_constr_to_ts_constr(fcx.ccx.tcx, f_decl.inputs, c);
        set_in_prestate_constr(fcx, tsc, block_pre);
    }

    let mut changed = find_pre_post_state_block(fcx, block_pre, f_body);

    /*
        error!("find_pre_post_state_fn");
        log(error, changed);
        fcx.ccx.tcx.sess.span_note(f_body.span, fcx.name);
    */

    return changed;
}
//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
