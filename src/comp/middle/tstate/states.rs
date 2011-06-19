
import std::bitv;
import std::vec;
import std::vec::plus_option;
import std::vec::cat_options;
import std::option;
import std::option::get;
import std::option::is_none;
import std::option::none;
import std::option::some;
import std::option::maybe;
import tstate::ann::pre_and_post;
import tstate::ann::get_post;
import tstate::ann::postcond;
import tstate::ann::empty_pre_post;
import tstate::ann::empty_poststate;
import tstate::ann::require_and_preserve;
import tstate::ann::union;
import tstate::ann::intersect;
import tstate::ann::empty_prestate;
import tstate::ann::prestate;
import tstate::ann::poststate;
import tstate::ann::false_postcond;
import tstate::ann::ts_ann;
import tstate::ann::extend_prestate;
import tstate::ann::extend_poststate;
import aux::crate_ctxt;
import aux::fn_ctxt;
import aux::num_constraints;
import aux::expr_pp;
import aux::stmt_pp;
import aux::block_pp;
import aux::set_pre_and_post;
import aux::expr_prestate;
import aux::expr_precond;
import aux::expr_postcond;
import aux::stmt_poststate;
import aux::expr_poststate;
import aux::block_prestate;
import aux::block_poststate;
import aux::block_precond;
import aux::block_postcond;
import aux::fn_info;
import aux::log_pp;
import aux::log_pp_err;
import aux::extend_prestate_ann;
import aux::extend_poststate_ann;
import aux::set_prestate_ann;
import aux::set_poststate_ann;
import aux::pure_exp;
import aux::log_bitv;
import aux::log_bitv_err;
import aux::stmt_to_ann;
import aux::log_states;
import aux::log_states_err;
import aux::block_states;
import aux::controlflow_expr;
import aux::node_id_to_def;
import aux::expr_to_constr;
import aux::ninit;
import aux::npred;
import aux::path_to_ident;
import aux::if_ty;
import aux::if_check;
import aux::plain_if;

import bitvectors::seq_preconds;
import bitvectors::union_postconds;
import bitvectors::intersect_postconds;
import bitvectors::declare_var;
import bitvectors::bit_num;
import bitvectors::gen_poststate;
import bitvectors::kill_poststate;
import front::ast;
import front::ast::*;
import middle::ty::expr_node_id;
import middle::ty::expr_ty;
import middle::ty::type_is_nil;
import middle::ty::type_is_bot;
import util::common::new_def_hash;
import util::common::uistr;
import util::common::log_expr;
import util::common::log_block;
import util::common::log_block_err;
import util::common::log_fn;
import util::common::elt_exprs;
import util::common::field_exprs;
import util::common::has_nonlocal_exits;
import util::common::log_stmt;
import util::common::log_stmt_err;
import util::common::log_expr_err;

fn seq_states(&fn_ctxt fcx, prestate pres, vec[@expr] exprs) ->
   tup(bool, poststate) {
    auto changed = false;
    auto post = pres;
    for (@expr e in exprs) {
        changed = find_pre_post_state_expr(fcx, post, e) || changed;
        post = expr_poststate(fcx.ccx, e);
    }
    ret tup(changed, post);
}

fn find_pre_post_state_exprs(&fn_ctxt fcx, &prestate pres, ast::node_id id,
                             &vec[@expr] es) -> bool {
    auto res = seq_states(fcx, pres, es);
    auto changed = res._0;
    changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
    changed = extend_poststate_ann(fcx.ccx, id, res._1) || changed;
    ret changed;
}

fn find_pre_post_state_loop(&fn_ctxt fcx, prestate pres, &@local l,
                            &@expr index, &block body, node_id id) -> bool {
    auto changed = false;
    /* same issues as while */

    // FIXME: also want to set l as initialized, no?

    changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
    changed = find_pre_post_state_expr(fcx, pres, index) || changed;
    /* in general, would need the intersection of
       (poststate of index, poststate of body) */

    changed =
        find_pre_post_state_block(fcx, expr_poststate(fcx.ccx, index), body)
            || changed;

    if (has_nonlocal_exits(body)) { 
        changed = set_poststate_ann(fcx.ccx, id, pres) || changed;
    }

    auto res_p =
        intersect_postconds([expr_poststate(fcx.ccx, index),
                             block_poststate(fcx.ccx, body)]);
    changed = extend_poststate_ann(fcx.ccx, id, res_p) || changed;
    ret changed;
}

fn gen_if_local(&fn_ctxt fcx, node_id new_var, node_id id, &path p) -> bool {
    alt (node_id_to_def(fcx.ccx, new_var)) {
        case (some(def_local(?loc))) {
            ret gen_poststate(fcx, id,
                              rec(id=loc._1,
                                  c=ninit(path_to_ident(fcx.ccx.tcx, p))));
        }
        case (_) { ret false; }
    }
}

fn join_then_else(&fn_ctxt fcx, &@expr antec, &block conseq,
                  &option::t[@expr] maybe_alt, ast::node_id id, &if_ty chk,
                  &prestate pres) -> bool {
    auto changed = false;

    changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
    changed = find_pre_post_state_expr(fcx, pres, antec) || changed;
    
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

    alt (maybe_alt) {
        case (none) {

            changed =
                find_pre_post_state_block(fcx, expr_poststate(fcx.ccx, antec),
                                  conseq) || changed;
   
            changed =
                extend_poststate_ann(fcx.ccx, id,
                                     expr_poststate(fcx.ccx, antec))
                || changed;
        }
        case (some(?altern)) {
            changed =
                find_pre_post_state_expr(fcx,
                                         expr_poststate(fcx.ccx,
                                                        antec),
                                         altern) || changed;

            auto conseq_prestate = expr_poststate(fcx.ccx, antec);
            alt (chk) {
                case (if_check) {
                    let aux::constr c = expr_to_constr(fcx.ccx.tcx, antec);
                    conseq_prestate = bitv::clone(conseq_prestate);
                    bitv::set(conseq_prestate, bit_num(fcx, c.node), true);
                }
                case (_) {}
            }


            changed =
                find_pre_post_state_block(fcx, conseq_prestate, conseq)
                || changed;
   
            auto poststate_res =
                intersect_postconds([block_poststate(fcx.ccx, conseq),
                                     expr_poststate(fcx.ccx,
                                                    altern)]);
            /*   fcx.ccx.tcx.sess.span_note(antec.span,
               "poststate_res = " + aux::bitv_to_str(fcx, poststate_res));
            fcx.ccx.tcx.sess.span_note(antec.span,
               "altern poststate = " +
                aux::bitv_to_str(fcx, expr_poststate(fcx.ccx, altern)));
            fcx.ccx.tcx.sess.span_note(antec.span,
            "conseq poststate = " + aux::bitv_to_str(fcx,
               block_poststate(fcx.ccx, conseq))); */

            changed =
                extend_poststate_ann(fcx.ccx, id, poststate_res) ||
                changed;
        }
    }
    ret changed;
}

fn find_pre_post_state_expr(&fn_ctxt fcx, &prestate pres, @expr e) -> bool {
    auto changed = false;
    auto num_local_vars = num_constraints(fcx.enclosing);
    /*
    log_err("states:");
    log_expr_err(*e);
    aux::log_bitv_err(fcx, expr_prestate(fcx.ccx, e));
    aux::log_bitv_err(fcx, expr_poststate(fcx.ccx, e));
    */

    /* FIXME could get rid of some of the copy/paste */
    alt (e.node) {
        case (expr_vec(?elts, _, _, ?id)) {
            ret find_pre_post_state_exprs(fcx, pres, id, elts);
        }
        case (expr_tup(?elts, ?id)) {
            ret find_pre_post_state_exprs(fcx, pres, id, elt_exprs(elts));
        }
        case (expr_call(?operator, ?operands, ?id)) {
            /* do the prestate for the rator */

            /*            fcx.ccx.tcx.sess.span_note(operator.span, 
                         "pres = " + aux::bitv_to_str(fcx, pres));
            */

            changed =
                find_pre_post_state_expr(fcx, pres, operator) || changed;
            /* rands go left-to-right */

            changed =
                find_pre_post_state_exprs(fcx,
                                          expr_poststate(fcx.ccx, operator),
                                          id, operands) || changed;
            /* if this is a failing call, it sets everything as initialized */

            alt (controlflow_expr(fcx.ccx, operator)) {
                case (noreturn) {
                    changed =
                        set_poststate_ann(fcx.ccx, id,
                                          false_postcond(num_local_vars)) ||
                            changed;
                }
                case (_) { }
            }

            /*            fcx.ccx.tcx.sess.span_note(operator.span, 
               "pres = " + aux::bitv_to_str(fcx, expr_poststate(fcx.ccx, e)));
            */
            ret changed;
        }
        case (expr_spawn(_, _, ?operator, ?operands, ?id)) {
            changed = find_pre_post_state_expr(fcx, pres, operator);
            ret find_pre_post_state_exprs(fcx,
                                          expr_poststate(fcx.ccx, operator),
                                          id, operands) || changed;
        }
        case (expr_bind(?operator, ?maybe_args, ?id)) {
            changed =
                find_pre_post_state_expr(fcx, pres, operator) || changed;
            ret find_pre_post_state_exprs(fcx,
                                          expr_poststate(fcx.ccx, operator),
                                          id, cat_options[@expr](maybe_args))
                    || changed;
        }
        case (expr_path(_, ?id)) { ret pure_exp(fcx.ccx, id, pres); }
        case (expr_log(_, ?e, ?id)) {
            /* factor out the "one exp" pattern */

            changed = find_pre_post_state_expr(fcx, pres, e);
            changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
            changed =
                extend_poststate_ann(fcx.ccx, id, expr_poststate(fcx.ccx, e))
                    || changed;
            ret changed;
        }
        case (expr_chan(?e, ?id)) {
            changed = find_pre_post_state_expr(fcx, pres, e);
            changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
            changed =
                extend_poststate_ann(fcx.ccx, id, expr_poststate(fcx.ccx, e))
                    || changed;
            ret changed;
        }
        case (expr_ext(_, _, _, ?expanded, ?id)) {
            changed = find_pre_post_state_expr(fcx, pres, expanded);
            changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
            changed =
                extend_poststate_ann(fcx.ccx, id,
                                     expr_poststate(fcx.ccx, expanded)) ||
                    changed;
            ret changed;
        }
        case (expr_put(?maybe_e, ?id)) {
            alt (maybe_e) {
                case (some(?arg)) {
                    changed = find_pre_post_state_expr(fcx, pres, arg);
                    changed =
                        extend_prestate_ann(fcx.ccx, id, pres) || changed;
                    changed =
                        extend_poststate_ann(fcx.ccx, id,
                                             expr_poststate(fcx.ccx, arg)) ||
                            changed;
                    ret changed;
                }
                case (none) { ret pure_exp(fcx.ccx, id, pres); }
            }
        }
        case (expr_lit(?l, ?id)) { ret pure_exp(fcx.ccx, id, pres); }
        case (
             // FIXME This was just put in here as a placeholder
             expr_fn(?f, ?id)) {
            ret pure_exp(fcx.ccx, id, pres);
        }
        case (expr_block(?b, ?id)) {
            changed = find_pre_post_state_block(fcx, pres, b) || changed;
            changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
            changed =
                extend_poststate_ann(fcx.ccx, id, block_poststate(fcx.ccx, b))
                    || changed;
            ret changed;
        }
        case (expr_rec(?fields, ?maybe_base, ?id)) {
            changed =
                find_pre_post_state_exprs(fcx, pres, id, field_exprs(fields))
                    || changed;
            alt (maybe_base) {
                case (none) {/* do nothing */ }
                case (some(?base)) {
                    changed =
                        find_pre_post_state_expr(fcx, pres, base) || changed;
                    changed =
                        extend_poststate_ann(fcx.ccx, id,
                                             expr_poststate(fcx.ccx, base)) ||
                            changed;
                }
            }
            ret changed;
        }
        case (expr_move(?lhs, ?rhs, ?id)) {
            // FIXME: this needs to deinitialize the rhs

            extend_prestate_ann(fcx.ccx, id, pres);
            alt (lhs.node) {
                case (expr_path(?p, ?id_lhs)) {
                    // assignment to local var

                    changed = pure_exp(fcx.ccx, id_lhs, pres) || changed;
                    changed =
                        find_pre_post_state_expr(fcx, pres, rhs) || changed;
                    changed =
                        extend_poststate_ann(fcx.ccx, id,
                                             expr_poststate(fcx.ccx, rhs)) ||
                            changed;
                    changed = gen_if_local(fcx, id_lhs, id, p) || changed;
                }
                case (_) {
                    // assignment to something that must already have been
                    // init'd

                    changed =
                        find_pre_post_state_expr(fcx, pres, lhs) || changed;
                    changed =
                        find_pre_post_state_expr(fcx,
                                                 expr_poststate(fcx.ccx, lhs),
                                                 rhs) || changed;
                    changed =
                        extend_poststate_ann(fcx.ccx, id,
                                             expr_poststate(fcx.ccx, rhs)) ||
                            changed;
                }
            }
            ret changed;
        }
        case (expr_assign(?lhs, ?rhs, ?id)) {
            extend_prestate_ann(fcx.ccx, id, pres);
            alt (lhs.node) {
                case (expr_path(?p, ?a_lhs)) {
                    // assignment to local var

                    changed = pure_exp(fcx.ccx, a_lhs, pres) || changed;
                    changed =
                        find_pre_post_state_expr(fcx, pres, rhs) || changed;
                    changed =
                        extend_poststate_ann(fcx.ccx, id,
                                             expr_poststate(fcx.ccx, rhs)) ||
                            changed;
                    changed = gen_if_local(fcx, a_lhs, id, p) || changed;
                }
                case (_) {
                    // assignment to something that must already have been
                    // init'd

                    changed =
                        find_pre_post_state_expr(fcx, pres, lhs) || changed;
                    changed =
                        find_pre_post_state_expr(fcx,
                                                 expr_poststate(fcx.ccx, lhs),
                                                 rhs) || changed;
                    changed =
                        extend_poststate_ann(fcx.ccx, id,
                                             expr_poststate(fcx.ccx, rhs)) ||
                            changed;
                }
            }
            ret changed;
        }
        case (expr_swap(?lhs, ?rhs, ?id)) {
            /* quite similar to binary -- should abstract this */

            changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
            changed = find_pre_post_state_expr(fcx, pres, lhs) || changed;
            changed =
                find_pre_post_state_expr(fcx, expr_poststate(fcx.ccx, lhs),
                                         rhs) || changed;
            changed =
                extend_poststate_ann(fcx.ccx, id,
                                     expr_poststate(fcx.ccx, rhs)) || changed;
            ret changed;
        }
        case (expr_recv(?lhs, ?rhs, ?id)) {
            extend_prestate_ann(fcx.ccx, id, pres);
            alt (rhs.node) {
                case (expr_path(?p, ?id_rhs)) {
                    // receive to local var

                    changed = pure_exp(fcx.ccx, id_rhs, pres) || changed;
                    changed =
                        find_pre_post_state_expr(fcx, pres, lhs) || changed;
                    changed =
                        extend_poststate_ann(fcx.ccx, id,
                                             expr_poststate(fcx.ccx, lhs)) ||
                            changed;
                    changed = gen_if_local(fcx, id_rhs, id, p) || changed;
                }
                case (_) {
                    // receive to something that must already have been init'd

                    changed =
                        find_pre_post_state_expr(fcx, pres, rhs) || changed;
                    changed =
                        find_pre_post_state_expr(fcx,
                                                 expr_poststate(fcx.ccx, rhs),
                                                 lhs) || changed;
                    changed =
                        extend_poststate_ann(fcx.ccx, id,
                                             expr_poststate(fcx.ccx, lhs)) ||
                            changed;
                }
            }
            ret changed;
        }
        case (expr_ret(?maybe_ret_val, ?id)) {
            changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
            /* normally, everything is true if execution continues after
               a ret expression (since execution never continues locally
               after a ret expression */

            set_poststate_ann(fcx.ccx, id, false_postcond(num_local_vars));
            /* return from an always-failing function clears the return bit */

            alt (fcx.enclosing.cf) {
                case (noreturn) {
                    kill_poststate(fcx, id, rec(id=fcx.id,
                                                c=ninit(fcx.name)));
                }
                case (_) { }
            }
            alt (maybe_ret_val) {
                case (none) {/* do nothing */ }
                case (some(?ret_val)) {
                    changed =
                        find_pre_post_state_expr(fcx, pres, ret_val) ||
                            changed;
                }
            }
            ret changed;
        }
        case (expr_be(?e, ?id)) {
            changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
            set_poststate_ann(fcx.ccx, id, false_postcond(num_local_vars));
            changed = find_pre_post_state_expr(fcx, pres, e) || changed;
            ret changed;
        }
        case (expr_if(?antec, ?conseq, ?maybe_alt, ?id)) {
            changed = join_then_else(fcx, antec, conseq, maybe_alt, id,
                                     plain_if, pres)
                || changed;

            ret changed;
        }
        case (expr_binary(?bop, ?l, ?r, ?id)) {
            /* FIXME: what if bop is lazy? */

            changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
            changed = find_pre_post_state_expr(fcx, pres, l) || changed;
            changed =
                find_pre_post_state_expr(fcx, expr_poststate(fcx.ccx, l), r)
                    || changed;
            changed =
                extend_poststate_ann(fcx.ccx, id, expr_poststate(fcx.ccx, r))
                    || changed;
            ret changed;
        }
        case (expr_send(?l, ?r, ?id)) {
            changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
            changed = find_pre_post_state_expr(fcx, pres, l) || changed;
            changed =
                find_pre_post_state_expr(fcx, expr_poststate(fcx.ccx, l), r)
                    || changed;
            changed =
                extend_poststate_ann(fcx.ccx, id, expr_poststate(fcx.ccx, r))
                    || changed;
            ret changed;
        }
        case (expr_assign_op(?op, ?lhs, ?rhs, ?id)) {
            /* quite similar to binary -- should abstract this */

            changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
            changed = find_pre_post_state_expr(fcx, pres, lhs) || changed;
            changed =
                find_pre_post_state_expr(fcx, expr_poststate(fcx.ccx, lhs),
                                         rhs) || changed;
            changed =
                extend_poststate_ann(fcx.ccx, id,
                                     expr_poststate(fcx.ccx, rhs))
                    || changed;
            ret changed;
        }
        case (expr_while(?test, ?body, ?id)) {
            changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
            /* to handle general predicates, we need to pass in
                pres `intersect` (poststate(a)) 
             like: auto test_pres = intersect_postconds(pres,
             expr_postcond(a)); However, this doesn't work right now because
             we would be passing in an all-zero prestate initially
               FIXME
               maybe need a "don't know" state in addition to 0 or 1?
            */

            changed = find_pre_post_state_expr(fcx, pres, test) || changed;
            changed =
                find_pre_post_state_block(fcx, expr_poststate(fcx.ccx, test),
                                          body) || changed;
            /* conservative approximation: if a loop contains a break
               or cont, we assume nothing about the poststate */
            if (has_nonlocal_exits(body)) { 
                changed = set_poststate_ann(fcx.ccx, id, pres) || changed;
            }

            changed =
                {
                    auto e_post = expr_poststate(fcx.ccx, test);
                    auto b_post = block_poststate(fcx.ccx, body);
                    extend_poststate_ann(fcx.ccx, id,
                                         intersect_postconds([e_post,
                                                              b_post])) ||
                        changed
                };
            ret changed;
        }
        case (expr_do_while(?body, ?test, ?id)) {
            changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
            auto changed0 = changed;
            changed = find_pre_post_state_block(fcx, pres, body) || changed;
            /* conservative approximination: if the body of the loop
               could break or cont, we revert to the prestate
               (TODO: could treat cont differently from break, since
               if there's a cont, the test will execute) */

            auto breaks = has_nonlocal_exits(body);
            if (breaks) {
                // this should probably be true_poststate and not pres,
                // b/c the body could invalidate stuff
                // FIXME
                 // This doesn't set "changed", as if the previous state
                // was different, this might come back true every time
                set_poststate_ann(fcx.ccx, body.node.id, pres);
                changed = changed0;
            }

            changed =
                find_pre_post_state_expr(fcx, block_poststate(fcx.ccx, body),
                                         test) || changed;

            if (breaks) {
                set_poststate_ann(fcx.ccx, id, pres);
            }
            else {
                changed =  extend_poststate_ann(fcx.ccx, id,
                                            expr_poststate(fcx.ccx, test)) ||
                    changed;
            }
            ret changed;
        }
        case (expr_for(?d, ?index, ?body, ?id)) {
            ret find_pre_post_state_loop(fcx, pres, d, index, body, id);
        }
        case (expr_for_each(?d, ?index, ?body, ?id)) {
            ret find_pre_post_state_loop(fcx, pres, d, index, body, id);
        }
        case (expr_index(?e, ?sub, ?id)) {
            changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
            changed = find_pre_post_state_expr(fcx, pres, e) || changed;
            changed =
                find_pre_post_state_expr(fcx, expr_poststate(fcx.ccx, e), sub)
                    || changed;
            changed =
                extend_poststate_ann(fcx.ccx, id,
                                     expr_poststate(fcx.ccx, sub));
            ret changed;
        }
        case (expr_alt(?e, ?alts, ?id)) {
            changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
            changed = find_pre_post_state_expr(fcx, pres, e) || changed;
            auto e_post = expr_poststate(fcx.ccx, e);
            auto a_post;
            if (vec::len[arm](alts) > 0u) {
                a_post = false_postcond(num_local_vars);
                for (arm an_alt in alts) {
                    changed =
                        find_pre_post_state_block(fcx, e_post, an_alt.block)
                            || changed;
                    intersect(a_post, block_poststate(fcx.ccx, an_alt.block));
                    // We deliberately do *not* update changed here, because
                    // we'd go into an infinite loop that way, and the change
                    // gets made after the if expression.

                }
            } else {
                // No alts; poststate is the poststate of the test

                a_post = e_post;
            }
            changed = extend_poststate_ann(fcx.ccx, id, a_post) || changed;
            ret changed;
        }
        case (expr_field(?e, _, ?id)) {
            changed = find_pre_post_state_expr(fcx, pres, e);
            changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
            changed =
                extend_poststate_ann(fcx.ccx, id, expr_poststate(fcx.ccx, e))
                    || changed;
            ret changed;
        }
        case (expr_unary(_, ?operand, ?id)) {
            changed = find_pre_post_state_expr(fcx, pres, operand) || changed;
            changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
            changed =
                extend_poststate_ann(fcx.ccx, id,
                                     expr_poststate(fcx.ccx, operand)) ||
                    changed;
            ret changed;
        }
        case (expr_cast(?operand, _, ?id)) {
            changed = find_pre_post_state_expr(fcx, pres, operand) || changed;
            changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
            changed =
                extend_poststate_ann(fcx.ccx, id,
                                     expr_poststate(fcx.ccx, operand)) ||
                    changed;
            ret changed;
        }
        case (expr_fail(?id, _)) {
            changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
            /* if execution continues after fail, then everything is true!
               woo! */

            changed =
                set_poststate_ann(fcx.ccx, id, false_postcond(num_local_vars))
                    || changed;
            ret changed;
        }
        case (expr_assert(?p, ?id)) {
            changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
            changed = find_pre_post_state_expr(fcx, pres, p) || changed;
            changed = extend_poststate_ann(fcx.ccx, id, pres) || changed;
            ret changed;
        }
        case (expr_check(?p, ?id)) {
            changed = extend_prestate_ann(fcx.ccx, id, pres) || changed;
            changed = find_pre_post_state_expr(fcx, pres, p) || changed;
            changed = extend_poststate_ann(fcx.ccx, id, pres) || changed;
            /* predicate p holds after this expression executes */

            let aux::constr c = expr_to_constr(fcx.ccx.tcx, p);
            changed = gen_poststate(fcx, id, c.node) || changed;
            ret changed;
        }
        case (expr_if_check(?p, ?conseq, ?maybe_alt, ?id)) {
            changed = join_then_else(fcx, p, conseq, maybe_alt, id, if_check,
                                     pres)
                || changed;

            ret changed;
        }
        case (expr_break(?id)) { ret pure_exp(fcx.ccx, id, pres); }
        case (expr_cont(?id)) { ret pure_exp(fcx.ccx, id, pres); }
        case (expr_port(?id)) { ret pure_exp(fcx.ccx, id, pres); }
        case (expr_self_method(_, ?id)) { ret pure_exp(fcx.ccx, id, pres); }
        case (expr_anon_obj(?anon_obj, _, _, ?id)) {
            alt (anon_obj.with_obj) {
                case (some(?e)) {
                    changed = find_pre_post_state_expr(fcx, pres, e);
                    changed =
                        extend_prestate_ann(fcx.ccx, id, pres) || changed;
                    changed =
                        extend_poststate_ann(fcx.ccx, id,
                                             expr_poststate(fcx.ccx, e)) ||
                            changed;
                    ret changed;
                }
                case (none) { ret pure_exp(fcx.ccx, id, pres); }
            }
        }
    }
}

fn find_pre_post_state_stmt(&fn_ctxt fcx, &prestate pres, @stmt s) -> bool {
    auto changed = false;
    auto stmt_ann = stmt_to_ann(fcx.ccx, *s);

    /*
    log_err "*At beginning: stmt = ";
    log_stmt_err(*s);
    log_err "*prestate = ";
    log_err bitv::to_str(stmt_ann.states.prestate);
    log_err "*poststate =";
    log_err bitv::to_str(stmt_ann.states.poststate);
    log_err "*changed =";
    log_err changed;
    */

    alt (s.node) {
        case (stmt_decl(?adecl, ?id)) {
            alt (adecl.node) {
                case (decl_local(?alocal)) {
                    alt (alocal.node.init) {
                        case (some(?an_init)) {
                            changed =
                                extend_prestate(stmt_ann.states.prestate,
                                                pres) || changed;
                            changed =
                                find_pre_post_state_expr(fcx, pres,
                                                         an_init.expr)
                                || changed;
                            changed =
                                extend_poststate(stmt_ann.states.poststate,
                                                 expr_poststate(fcx.ccx,
                                                                an_init.expr))
                                    || changed;
                            changed =
                                gen_poststate(fcx, id,
                                              rec(id=alocal.node.id,
                                                  c=ninit(alocal.node.ident)))
                                || changed;
                            log "Summary: stmt = ";
                            log_stmt(*s);
                            log "prestate = ";
                            log bitv::to_str(stmt_ann.states.prestate);
                            log_bitv(fcx, stmt_ann.states.prestate);
                            log "poststate =";
                            log_bitv(fcx, stmt_ann.states.poststate);
                            log "changed =";
                            log changed;
                            ret changed;
                        }
                        case (none) {
                            changed =
                                extend_prestate(stmt_ann.states.prestate,
                                                pres) || changed;
                            changed =
                                extend_poststate(stmt_ann.states.poststate,
                                                 pres) || changed;
                            ret changed;
                        }
                    }
                }
                case (decl_item(?an_item)) {
                    changed =
                        extend_prestate(stmt_ann.states.prestate, pres) ||
                            changed;
                    changed =
                        extend_poststate(stmt_ann.states.poststate, pres) ||
                            changed;
                    /* the outer "walk" will recurse into the item */

                    ret changed;
                }
            }
        }
        case (stmt_expr(?e, _)) {
            changed = find_pre_post_state_expr(fcx, pres, e) || changed;
            changed =
                extend_prestate(stmt_ann.states.prestate,
                                expr_prestate(fcx.ccx, e)) || changed;
            changed =
                extend_poststate(stmt_ann.states.poststate,
                                 expr_poststate(fcx.ccx, e)) || changed;
            /*
              log("Summary: stmt = ");
              log_stmt(*s);
              log("prestate = ");
              log(bitv::to_str(stmt_ann.states.prestate));
              log_bitv(enclosing, stmt_ann.states.prestate);
              log("poststate =");
              log(bitv::to_str(stmt_ann.states.poststate));
              log_bitv(enclosing, stmt_ann.states.poststate);
              log("changed =");
              log(changed);
            */

            ret changed;
        }
        case (_) { ret false; }
    }
}


/* Updates the pre- and post-states of statements in the block,
   returns a boolean flag saying whether any pre- or poststates changed */
fn find_pre_post_state_block(&fn_ctxt fcx, &prestate pres0, &block b) ->
   bool {
    auto changed = false;
    auto num_local_vars = num_constraints(fcx.enclosing);
    /* First, set the pre-states and post-states for every expression */

    auto pres = pres0;
    /* Iterate over each stmt. The new prestate is <pres>. The poststate
     consist of improving <pres> with whatever variables this stmt
     initializes.  Then <pres> becomes the new poststate. */

    for (@stmt s in b.node.stmts) {
        changed = find_pre_post_state_stmt(fcx, pres, s) || changed;
        pres = stmt_poststate(fcx.ccx, *s);
    }
    auto post = pres;
    alt (b.node.expr) {
        case (none) { }
        case (some(?e)) {
            changed = find_pre_post_state_expr(fcx, pres, e) || changed;
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
    log_bitv_err(fcx, pres0);
    log_err "post:";
    log_bitv_err(fcx, post);
    */

    ret changed;
}

fn find_pre_post_state_fn(&fn_ctxt fcx, &_fn f) -> bool {
    auto num_local_vars = num_constraints(fcx.enclosing);
    auto changed =
        find_pre_post_state_block(fcx, empty_prestate(num_local_vars),
                                  f.body);
    // Treat the tail expression as a return statement

    alt (f.body.node.expr) {
        case (some(?tailexpr)) {
            auto tailann = expr_node_id(tailexpr);
            auto tailty = expr_ty(fcx.ccx.tcx, tailexpr);

            // Since blocks and alts and ifs that don't have results
            // implicitly result in nil, we have to be careful to not
            // interpret nil-typed block results as the result of a
            // function with some other return type
            if (!type_is_nil(fcx.ccx.tcx, tailty) &&
                    !type_is_bot(fcx.ccx.tcx, tailty)) {
                auto p = false_postcond(num_local_vars);
                set_poststate_ann(fcx.ccx, tailann, p);
                // be sure to set the block poststate to the same thing
                set_poststate_ann(fcx.ccx, f.body.node.id, p);
                alt (fcx.enclosing.cf) {
                    case (noreturn) {
                        kill_poststate(fcx, tailann,
                                       rec(id=fcx.id, c=ninit(fcx.name)));
                    }
                    case (_) { }
                }
            }
        }
        case (none) {/* fallthrough */ }
    }
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
