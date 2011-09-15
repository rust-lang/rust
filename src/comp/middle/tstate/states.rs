import syntax::print::pprust::path_to_str;
import util::ppaux::ty_to_str;
import std::{vec, str, option};
import std::option::{get, is_none, none, some, maybe};
import ann::*;
import aux::*;
import tritv::{tritv_clone, tritv_set, ttrue};

import bitvectors::*;
import syntax::ast::*;
import syntax::ast_util::*;
import syntax::codemap::span;
import middle::ty::{expr_ty, type_is_nil, type_is_bot};
import util::common::{new_def_hash, log_expr, log_block, log_block_err,
                      log_fn, field_exprs, has_nonlocal_exits, log_stmt,
                      log_stmt_err, log_expr_err};

fn forbid_upvar(fcx: fn_ctxt, rhs_id: node_id, sp: span, t: oper_type) {
    alt t {
      oper_move. {
        alt local_node_id_to_def(fcx, rhs_id) {
          some(def_upvar(_, _, _)) {
            fcx.ccx.tcx.sess.span_err(sp,
                                      "Tried to deinitialize a variable \
              declared in a different scope");
          }
          _ { }
        }
      }
      _ {/* do nothing */ }
    }
}

fn handle_move_or_copy(fcx: fn_ctxt, post: poststate, rhs_path: path,
                       rhs_id: node_id, instlhs: inst, init_op: init_op) {
    forbid_upvar(fcx, rhs_id, rhs_path.span, op_to_oper_ty(init_op));

    let rhs_d_id = local_node_id_to_def_id(fcx, rhs_id);
    alt rhs_d_id {
      some(rhsid) {
        // RHS is a local var
        let instrhs =
            {ident: path_to_ident(fcx.ccx.tcx, rhs_path), node: rhsid.node};
        copy_in_poststate(fcx, post, instlhs, instrhs,
                          op_to_oper_ty(init_op));
      }
      _ {
        // not a local -- do nothing
      }
    }
}

fn seq_states(fcx: fn_ctxt, pres: prestate, bindings: [binding]) ->
   {changed: bool, post: poststate} {
    let changed = false;
    let post = tritv_clone(pres);
    for b: binding in bindings {
        alt b.rhs {
          some(an_init) {
            // an expression, with or without a destination
            changed |=
                find_pre_post_state_expr(fcx, post, an_init.expr) || changed;
            post = tritv_clone(expr_poststate(fcx.ccx, an_init.expr));
            for i: inst in b.lhs {
                alt an_init.expr.node {
                  expr_path(p) {
                    handle_move_or_copy(fcx, post, p, an_init.expr.id, i,
                                        an_init.op);
                  }
                  _ { }
                }
                set_in_poststate_ident(fcx, i.node, i.ident, post);
            }

            // Forget the RHS if we just moved it.
            if an_init.op == init_move {
                forget_in_poststate(fcx, post, an_init.expr.id);
            }
          }
          none {
            for i: inst in b.lhs {
                // variables w/o an initializer
                clear_in_poststate_ident_(fcx, i.node, i.ident, post);
            }
          }
        }
    }
    ret {changed: changed, post: post};
}

fn find_pre_post_state_sub(fcx: fn_ctxt, pres: prestate, e: @expr,
                           parent: node_id, c: option::t<tsconstr>) -> bool {
    let changed = find_pre_post_state_expr(fcx, pres, e);

    changed = set_prestate_ann(fcx.ccx, parent, pres) || changed;

    let post = tritv_clone(expr_poststate(fcx.ccx, e));
    alt c {
      none. { }
      some(c1) { set_in_poststate_(bit_num(fcx, c1), post); }
    }

    changed = set_poststate_ann(fcx.ccx, parent, post) || changed;
    ret changed;
}

fn find_pre_post_state_two(fcx: fn_ctxt, pres: prestate, lhs: @expr,
                           rhs: @expr, parent: node_id, ty: oper_type) ->
   bool {
    let changed = set_prestate_ann(fcx.ccx, parent, pres);
    changed = find_pre_post_state_expr(fcx, pres, lhs) || changed;
    changed =
        find_pre_post_state_expr(fcx, expr_poststate(fcx.ccx, lhs), rhs) ||
            changed;
    forbid_upvar(fcx, rhs.id, rhs.span, ty);

    let post = tritv_clone(expr_poststate(fcx.ccx, rhs));

    alt lhs.node {
      expr_path(p) {
        // for termination, need to make sure intermediate changes don't set
        // changed flag
        // tmp remembers "old" constraints we'd otherwise forget,
        // for substitution purposes
        let tmp = tritv_clone(post);

        alt ty {
          oper_move. {
            if is_path(rhs) { forget_in_poststate(fcx, post, rhs.id); }
            forget_in_poststate_still_init(fcx, post, lhs.id);
          }
          oper_swap. {
            forget_in_poststate_still_init(fcx, post, lhs.id);
            forget_in_poststate_still_init(fcx, post, rhs.id);
          }
          _ { forget_in_poststate_still_init(fcx, post, lhs.id); }
        }

        gen_if_local(fcx, post, lhs);
        alt rhs.node {
          expr_path(p1) {
            let d = local_node_id_to_local_def_id(fcx, lhs.id);
            let d1 = local_node_id_to_local_def_id(fcx, rhs.id);
            alt d {
              some(id) {
                alt d1 {
                  some(id1) {
                    let instlhs =
                        {ident: path_to_ident(fcx.ccx.tcx, p), node: id};
                    let instrhs =
                        {ident: path_to_ident(fcx.ccx.tcx, p1), node: id1};
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
    ret changed;
}

fn find_pre_post_state_call(fcx: fn_ctxt, pres: prestate, a: @expr,
                            id: node_id, ops: [init_op], bs: [@expr],
                            cf: ret_style) -> bool {
    let changed = find_pre_post_state_expr(fcx, pres, a);
    // FIXME: This could be a typestate constraint
    if vec::len(bs) != vec::len(ops) {
        fcx.ccx.tcx.sess.span_bug(a.span,
                                  #fmt["mismatched arg lengths: \
                                        %u exprs vs. %u ops",
                                       vec::len(bs), vec::len(ops)]);
    }
    ret find_pre_post_state_exprs(fcx, expr_poststate(fcx.ccx, a), id, ops,
                                  bs, cf) || changed;
}

fn find_pre_post_state_exprs(fcx: fn_ctxt, pres: prestate, id: node_id,
                             ops: [init_op], es: [@expr], cf: ret_style) ->
   bool {
    let rs = seq_states(fcx, pres, anon_bindings(ops, es));
    let changed = rs.changed | set_prestate_ann(fcx.ccx, id, pres);
    /* if this is a failing call, it sets everything as initialized */
    alt cf {
      noreturn. {
        let post = false_postcond(num_constraints(fcx.enclosing));
        changed |= set_poststate_ann(fcx.ccx, id, post);
      }
      _ { changed |= set_poststate_ann(fcx.ccx, id, rs.post); }
    }
    ret changed;
}

fn find_pre_post_state_loop(fcx: fn_ctxt, pres: prestate, l: @local,
                            index: @expr, body: blk, id: node_id) -> bool {
    let loop_pres = intersect_states(pres, block_poststate(fcx.ccx, body));

    let changed =
        set_prestate_ann(fcx.ccx, id, loop_pres) |
            find_pre_post_state_expr(fcx, pres, index);

    // Make sure the index vars are considered initialized
    // in the body
    let index_post = tritv_clone(expr_poststate(fcx.ccx, index));
    for each p: @pat in pat_bindings(l.node.pat) {
        let ident = alt p.node { pat_bind(name) { name } };
        set_in_poststate_ident(fcx, p.id, ident, index_post);
    }

    changed |= find_pre_post_state_block(fcx, index_post, body);


    if has_nonlocal_exits(body) {
        // See [Break-unsound]
        ret changed | set_poststate_ann(fcx.ccx, id, pres);
    } else {
        let res_p =
            intersect_states(expr_poststate(fcx.ccx, index),
                             block_poststate(fcx.ccx, body));
        ret changed | set_poststate_ann(fcx.ccx, id, res_p);
    }
}

fn gen_if_local(fcx: fn_ctxt, p: poststate, e: @expr) -> bool {
    alt e.node {
      expr_path(pth) {
        alt fcx.ccx.tcx.def_map.find(e.id) {
          some(def_local(loc, _)) {
            ret set_in_poststate_ident(fcx, loc.node,
                                       path_to_ident(fcx.ccx.tcx, pth), p);
          }
          _ { ret false; }
        }
      }
      _ { ret false; }
    }
}

fn join_then_else(fcx: fn_ctxt, antec: @expr, conseq: blk,
                  maybe_alt: option::t<@expr>, id: node_id, chk: if_ty,
                  pres: prestate) -> bool {
    let changed =
        set_prestate_ann(fcx.ccx, id, pres) |
            find_pre_post_state_expr(fcx, pres, antec);

    /*
    log_err("join_then_else:");
    log_expr_err(*antec);
    log_bitv_err(fcx, expr_prestate(fcx.ccx, antec));
    log_bitv_err(fcx, expr_poststate(fcx.ccx, antec));
    log_block_err(conseq);
    log_bitv_err(fcx, block_prestate(fcx.ccx, conseq));
    log_bitv_err(fcx, block_poststate(fcx.ccx, conseq));
    log_err("****");
    log_bitv_err(fcx, expr_precond(fcx.ccx, antec));
    log_bitv_err(fcx, expr_postcond(fcx.ccx, antec));
    log_bitv_err(fcx, block_precond(fcx.ccx, conseq));
    log_bitv_err(fcx, block_postcond(fcx.ccx, conseq));
    */

    alt maybe_alt {
      none. {
        alt chk {
          if_check. {
            let c: sp_constr = expr_to_constr(fcx.ccx.tcx, antec);
            let conseq_prestate = tritv_clone(expr_poststate(fcx.ccx, antec));
            tritv_set(bit_num(fcx, c.node), conseq_prestate, ttrue);
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
      some(altern) {
        changed |=
            find_pre_post_state_expr(fcx, expr_poststate(fcx.ccx, antec),
                                     altern);

        let conseq_prestate = expr_poststate(fcx.ccx, antec);
        alt chk {
          if_check. {
            let c: sp_constr = expr_to_constr(fcx.ccx.tcx, antec);
            conseq_prestate = tritv_clone(conseq_prestate);
            tritv_set(bit_num(fcx, c.node), conseq_prestate, ttrue);
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
    ret changed;
}

fn find_pre_post_state_expr(fcx: fn_ctxt, pres: prestate, e: @expr) -> bool {
    let num_constrs = num_constraints(fcx.enclosing);


    alt e.node {
      expr_vec(elts, _) {
        ret find_pre_post_state_exprs(fcx, pres, e.id,
                                      vec::init_elt(init_assign,
                                                    vec::len(elts)), elts,
                                      return_val);
      }
      expr_call(operator, operands) {
        ret find_pre_post_state_call(fcx, pres, operator, e.id,
                                     callee_arg_init_ops(fcx, operator.id),
                                     operands,
                                     controlflow_expr(fcx.ccx, operator));
      }
      expr_bind(operator, maybe_args) {
        let args = [];
        let callee_ops = callee_arg_init_ops(fcx, operator.id);
        let ops = [];
        let i = 0;
        for a_opt: option::t<@expr> in maybe_args {
            alt a_opt {
              none. {/* no-op */ }
              some(a) { ops += [callee_ops[i]]; args += [a]; }
            }
            i += 1;
        }
        ret find_pre_post_state_call(fcx, pres, operator, e.id, ops, args,
                                     return_val);
      }
      expr_path(_) { ret pure_exp(fcx.ccx, e.id, pres); }
      expr_log(_, ex) {
        ret find_pre_post_state_sub(fcx, pres, ex, e.id, none);
      }
      expr_mac(_) { fcx.ccx.tcx.sess.bug("unexpanded macro"); }
      expr_put(maybe_e) {
        alt maybe_e {
          some(arg) {
            ret find_pre_post_state_sub(fcx, pres, arg, e.id, none);
          }
          none. { ret pure_exp(fcx.ccx, e.id, pres); }
        }
      }
      expr_lit(l) { ret pure_exp(fcx.ccx, e.id, pres); }
      expr_fn(f) { ret pure_exp(fcx.ccx, e.id, pres); }
      expr_block(b) {
        ret find_pre_post_state_block(fcx, pres, b) |
                set_prestate_ann(fcx.ccx, e.id, pres) |
                set_poststate_ann(fcx.ccx, e.id, block_poststate(fcx.ccx, b));
      }
      expr_rec(fields, maybe_base) {
        let changed =
            find_pre_post_state_exprs(fcx, pres, e.id,
                                      vec::init_elt(init_assign,
                                                    vec::len(fields)),
                                      field_exprs(fields), return_val);
        alt maybe_base {
          none. {/* do nothing */ }
          some(base) {
            changed |=
                find_pre_post_state_expr(fcx, pres, base) |
                    set_poststate_ann(fcx.ccx, e.id,
                                      expr_poststate(fcx.ccx, base));
          }
        }
        ret changed;
      }
      expr_tup(elts) {
        ret find_pre_post_state_exprs(fcx, pres, e.id,
                                      vec::init_elt(init_assign,
                                                    vec::len(elts)), elts,
                                      return_val);
      }
      expr_copy(a) { ret find_pre_post_state_sub(fcx, pres, a, e.id, none); }
      expr_move(lhs, rhs) {
        ret find_pre_post_state_two(fcx, pres, lhs, rhs, e.id, oper_move);
      }
      expr_assign(lhs, rhs) {
        ret find_pre_post_state_two(fcx, pres, lhs, rhs, e.id, oper_assign);
      }
      expr_swap(lhs, rhs) {
        ret find_pre_post_state_two(fcx, pres, lhs, rhs, e.id, oper_swap);
        // Could be more precise and actually swap the role of
        // lhs and rhs in constraints
      }
      expr_ret(maybe_ret_val) {
        let changed = set_prestate_ann(fcx.ccx, e.id, pres);
        /* normally, everything is true if execution continues after
           a ret expression (since execution never continues locally
           after a ret expression */
        // FIXME should factor this out
        let post = false_postcond(num_constrs);
        // except for the "diverges" bit...
        kill_poststate_(fcx, fcx.enclosing.i_diverge, post);

        set_poststate_ann(fcx.ccx, e.id, post);

        alt maybe_ret_val {
          none. {/* do nothing */ }
          some(ret_val) {
            changed |= find_pre_post_state_expr(fcx, pres, ret_val);
          }
        }
        ret changed;
      }
      expr_be(val) {
        let changed = set_prestate_ann(fcx.ccx, e.id, pres);
        let post = false_postcond(num_constrs);
        // except for the "diverges" bit...
        kill_poststate_(fcx, fcx.enclosing.i_diverge, post);
        set_poststate_ann(fcx.ccx, e.id, post);
        ret changed | find_pre_post_state_expr(fcx, pres, val);
      }
      expr_if(antec, conseq, maybe_alt) {
        ret join_then_else(fcx, antec, conseq, maybe_alt, e.id, plain_if,
                           pres);
      }
      expr_ternary(_, _, _) {
        ret find_pre_post_state_expr(fcx, pres, ternary_to_if(e));
      }
      expr_binary(bop, l, r) {
        if lazy_binop(bop) {
            let changed = find_pre_post_state_expr(fcx, pres, l);
            changed |=
                find_pre_post_state_expr(fcx, expr_poststate(fcx.ccx, l), r);
            ret changed | set_prestate_ann(fcx.ccx, e.id, pres) |
                    set_poststate_ann(fcx.ccx, e.id,
                                      expr_poststate(fcx.ccx, l));
        } else {
            ret find_pre_post_state_two(fcx, pres, l, r, e.id, oper_pure);
        }
      }
      expr_assign_op(op, lhs, rhs) {
        ret find_pre_post_state_two(fcx, pres, lhs, rhs, e.id,
                                    oper_assign_op);
      }
      expr_while(test, body) {
        /*
        log_err "in a while loop:";
        log_expr_err(*e);
        aux::log_tritv_err(fcx, block_poststate(fcx.ccx, body));
        aux::log_tritv_err(fcx, pres);
        */
        let loop_pres =
            intersect_states(block_poststate(fcx.ccx, body), pres);
        // aux::log_tritv_err(fcx, loop_pres);
        // log_err "---------------";

        let changed =
            set_prestate_ann(fcx.ccx, e.id, loop_pres) |
                find_pre_post_state_expr(fcx, loop_pres, test) |
                find_pre_post_state_block(fcx, expr_poststate(fcx.ccx, test),
                                          body);

        /* conservative approximation: if a loop contains a break
           or cont, we assume nothing about the poststate */
        /* which is still unsound -- see [Break-unsound] */
        if has_nonlocal_exits(body) {
            ret changed | set_poststate_ann(fcx.ccx, e.id, pres);
        } else {
            let e_post = expr_poststate(fcx.ccx, test);
            let b_post = block_poststate(fcx.ccx, body);
            ret changed |
                    set_poststate_ann(fcx.ccx, e.id,
                                      intersect_states(e_post, b_post));
        }
      }
      expr_do_while(body, test) {
        let loop_pres = intersect_states(expr_poststate(fcx.ccx, test), pres);

        let changed = set_prestate_ann(fcx.ccx, e.id, loop_pres);
        changed |= find_pre_post_state_block(fcx, loop_pres, body);
        /* conservative approximination: if the body of the loop
           could break or cont, we revert to the prestate
           (TODO: could treat cont differently from break, since
           if there's a cont, the test will execute) */

        changed |=
            find_pre_post_state_expr(fcx, block_poststate(fcx.ccx, body),
                                     test);

        let breaks = has_nonlocal_exits(body);
        if breaks {
            // this should probably be true_poststate and not pres,
            // b/c the body could invalidate stuff
            // FIXME [Break-unsound]
            // This is unsound as it is -- consider
            // while (true) {
            //    x <- y;
            //    break;
            // }
            // The poststate wouldn't take into account that
            // y gets deinitialized
            changed |= set_poststate_ann(fcx.ccx, e.id, pres);
        } else {
            changed |=
                set_poststate_ann(fcx.ccx, e.id,
                                  expr_poststate(fcx.ccx, test));
        }
        ret changed;
      }
      expr_for(d, index, body) {
        ret find_pre_post_state_loop(fcx, pres, d, index, body, e.id);
      }
      expr_for_each(d, index, body) {
        ret find_pre_post_state_loop(fcx, pres, d, index, body, e.id);
      }
      expr_index(val, sub) {
        ret find_pre_post_state_two(fcx, pres, val, sub, e.id, oper_pure);
      }
      expr_alt(val, alts) {
        let changed =
            set_prestate_ann(fcx.ccx, e.id, pres) |
                find_pre_post_state_expr(fcx, pres, val);
        let e_post = expr_poststate(fcx.ccx, val);
        let a_post;
        if vec::len(alts) > 0u {
            a_post = false_postcond(num_constrs);
            for an_alt: arm in alts {
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
        ret changed | set_poststate_ann(fcx.ccx, e.id, a_post);
      }
      expr_field(val, _) {
        ret find_pre_post_state_sub(fcx, pres, val, e.id, none);
      }
      expr_unary(_, operand) {
        ret find_pre_post_state_sub(fcx, pres, operand, e.id, none);
      }
      expr_cast(operand, _) {
        ret find_pre_post_state_sub(fcx, pres, operand, e.id, none);
      }
      expr_fail(maybe_fail_val) {
        // FIXME Should factor out this code,
        // which also appears in find_pre_post_state_exprs
        /* if execution continues after fail, then everything is true!
        woo! */
        let post = false_postcond(num_constrs);
        alt fcx.enclosing.cf {
          noreturn. { kill_poststate_(fcx, ninit(fcx.id, fcx.name), post); }
          _ { }
        }
        ret set_prestate_ann(fcx.ccx, e.id, pres) |
                set_poststate_ann(fcx.ccx, e.id, post) |
                alt maybe_fail_val {
                  none. { false }
                  some(fail_val) {
                    find_pre_post_state_expr(fcx, pres, fail_val)
                  }
                }
      }
      expr_assert(p) {
        ret find_pre_post_state_sub(fcx, pres, p, e.id, none);
      }
      expr_check(_, p) {
        /* predicate p holds after this expression executes */
        let c: sp_constr = expr_to_constr(fcx.ccx.tcx, p);
        ret find_pre_post_state_sub(fcx, pres, p, e.id, some(c.node));
      }
      expr_if_check(p, conseq, maybe_alt) {
        ret join_then_else(fcx, p, conseq, maybe_alt, e.id, if_check, pres);
      }
      expr_break. { ret pure_exp(fcx.ccx, e.id, pres); }
      expr_cont. { ret pure_exp(fcx.ccx, e.id, pres); }
      expr_self_method(_) { ret pure_exp(fcx.ccx, e.id, pres); }
      expr_anon_obj(anon_obj) {
        alt anon_obj.inner_obj {
          some(wt) { ret find_pre_post_state_sub(fcx, pres, wt, e.id, none); }
          none. { ret pure_exp(fcx.ccx, e.id, pres); }
        }
      }
      expr_uniq(_) { ret pure_exp(fcx.ccx, e.id, pres); }
    }
}

fn find_pre_post_state_stmt(fcx: fn_ctxt, pres: prestate, s: @stmt) -> bool {
    let stmt_ann = stmt_to_ann(fcx.ccx, *s);

    /*
        log_err ("[" + fcx.name + "]");
        log_err "*At beginning: stmt = ";
        log_stmt_err(*s);
        log_err "*prestate = ";
        log_tritv_err(fcx, stmt_ann.states.prestate);
        log_err "*poststate =";
        log_tritv_err(fcx, stmt_ann.states.poststate);
        log_err "pres = ";
        log_tritv_err(fcx, pres);
    */

    alt s.node {
      stmt_decl(adecl, id) {
        alt adecl.node {
          decl_local(alocals) {
            set_prestate(stmt_ann, pres);
            let c_and_p = seq_states(fcx, pres, locals_to_bindings(alocals));
            /* important to do this in one step to ensure
            termination (don't want to set changed to true
            for intermediate changes) */

            let changed =
                set_poststate(stmt_ann, c_and_p.post) | c_and_p.changed;

            /*
                            log_err "Summary: stmt = ";
                            log_stmt_err(*s);
                            log_err "prestate = ";
                            log_tritv_err(fcx, stmt_ann.states.prestate);
                            log_err "poststate =";
                            log_tritv_err(fcx, stmt_ann.states.poststate);
                            log_err "changed =";
                            log_err changed;
            */

            ret changed;
          }
          decl_item(an_item) {
            ret set_prestate(stmt_ann, pres) | set_poststate(stmt_ann, pres);
            /* the outer visitor will recurse into the item */
          }
        }
      }
      stmt_expr(ex, _) {
        let changed =
            find_pre_post_state_expr(fcx, pres, ex) |
                set_prestate(stmt_ann, expr_prestate(fcx.ccx, ex)) |
                set_poststate(stmt_ann, expr_poststate(fcx.ccx, ex));

        /*
        log_err "Finally:";
        log_stmt_err(*s);
        log_err("prestate = ");
        log_err(bitv::to_str(stmt_ann.states.prestate));
        log_tritv_err(fcx, stmt_ann.states.prestate);
        log_err("poststate =");
        log_err(bitv::to_str(stmt_ann.states.poststate));
        log_tritv_err(fcx, stmt_ann.states.poststate);
        log_err("changed =");
        */

        ret changed;
      }
      _ { ret false; }
    }
}


/* Updates the pre- and post-states of statements in the block,
   returns a boolean flag saying whether any pre- or poststates changed */
fn find_pre_post_state_block(fcx: fn_ctxt, pres0: prestate, b: blk) -> bool {
    /* First, set the pre-states and post-states for every expression */

    let pres = pres0;
    /* Iterate over each stmt. The new prestate is <pres>. The poststate
     consist of improving <pres> with whatever variables this stmt
     initializes.  Then <pres> becomes the new poststate. */

    let changed = false;
    for s: @stmt in b.node.stmts {
        changed |= find_pre_post_state_stmt(fcx, pres, s);
        pres = stmt_poststate(fcx.ccx, *s);
    }
    let post = pres;
    alt b.node.expr {
      none. { }
      some(e) {
        changed |= find_pre_post_state_expr(fcx, pres, e);
        post = expr_poststate(fcx.ccx, e);
      }
    }

    set_prestate_ann(fcx.ccx, b.node.id, pres0);
    set_poststate_ann(fcx.ccx, b.node.id, post);


    /*
        log_err "For block:";
        log_block_err(b);
        log_err "poststate = ";
        log_states_err(block_states(fcx.ccx, b));
        log_err "pres0:";
        log_tritv_err(fcx, pres0);
        log_err "post:";
        log_tritv_err(fcx, post);
        log_err "changed = ";
        log_err changed;
    */

    ret changed;
}

fn find_pre_post_state_fn(fcx: fn_ctxt, f: _fn) -> bool {
    let num_constrs = num_constraints(fcx.enclosing);
    // All constraints are considered false until proven otherwise.
    // This ensures that intersect works correctly.
    kill_all_prestate(fcx, f.body.node.id);

    // Arguments start out initialized
    let block_pre = block_prestate(fcx.ccx, f.body);
    for a: arg in f.decl.inputs {
        set_in_prestate_constr(fcx, ninit(a.id, a.ident), block_pre);
    }

    // Instantiate any constraints on the arguments so we can use them
    for c: @constr in f.decl.constraints {
        let tsc = ast_constr_to_ts_constr(fcx.ccx.tcx, f.decl.inputs, c);
        set_in_prestate_constr(fcx, tsc, block_pre);
    }

    let changed = find_pre_post_state_block(fcx, block_pre, f.body);

    // Treat the tail expression as a return statement
    alt f.body.node.expr {
      some(tailexpr) {

        // We don't want to clear the diverges bit for bottom typed things,
        // which really do diverge. I feel like there is a cleaner way
        // to do this than checking the type.
        if !type_is_bot(fcx.ccx.tcx, expr_ty(fcx.ccx.tcx, tailexpr)) {
            let post = false_postcond(num_constrs);
            // except for the "diverges" bit...
            kill_poststate_(fcx, fcx.enclosing.i_diverge, post);
            set_poststate_ann(fcx.ccx, f.body.node.id, post);
        }
      }
      none. {/* fallthrough */ }
    }

    /*
        log_err "find_pre_post_state_fn";
        log_err changed;
        fcx.ccx.tcx.sess.span_note(f.body.span, fcx.name);
    */

    ret changed;
}
//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
