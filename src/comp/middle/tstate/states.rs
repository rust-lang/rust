import std::vec;
import std::vec::plus_option;
import std::vec::cat_options;
import std::option;
import std::option::get;
import std::option::is_none;
import std::option::none;
import std::option::some;
import std::option::maybe;
import tstate::ann::set_in_poststate_;
import tstate::ann::pre_and_post;
import tstate::ann::get_post;
import tstate::ann::postcond;
import tstate::ann::empty_pre_post;
import tstate::ann::empty_poststate;
import tstate::ann::clear_in_poststate;
import tstate::ann::intersect;
import tstate::ann::empty_prestate;
import tstate::ann::prestate;
import tstate::ann::poststate;
import tstate::ann::false_postcond;
import tstate::ann::ts_ann;
import tstate::ann::set_prestate;
import tstate::ann::set_poststate;
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
import aux::log_tritv;
import aux::log_tritv_err;
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
import aux::forget_in_poststate;
import aux::forget_in_poststate_still_init;
import tritv::tritv_clone;
import tritv::tritv_set;
import tritv::ttrue;

import bitvectors::set_in_poststate_ident;
import bitvectors::clear_in_poststate_expr;
import bitvectors::clear_in_prestate_ident;
import bitvectors::bit_num;
import bitvectors::gen_poststate;
import bitvectors::kill_poststate;
import bitvectors::clear_in_poststate_ident;
import bitvectors::intersect_states;
import front::ast;
import front::ast::*;
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

// Used to communicate which operands should be invalidated
// to helper functions
tag oper_type {
    oper_move;
    oper_swap;
    oper_assign;
    oper_pure;
}

fn seq_states(&fn_ctxt fcx, prestate pres, vec[@expr] exprs) ->
   tup(bool, poststate) {
    auto changed = false;
    auto post = pres;
    for (@expr e in exprs) {
        changed |= find_pre_post_state_expr(fcx, post, e) || changed;
        // log_err("Seq_states: changed =");
        // log_err changed;
        post = expr_poststate(fcx.ccx, e);
    }
    ret tup(changed, post);
}

fn find_pre_post_state_sub(&fn_ctxt fcx, &prestate pres, &@expr e,
                           node_id parent, option::t[aux::constr_] c)
    -> bool {
    auto changed = find_pre_post_state_expr(fcx, pres, e);

    changed = set_prestate_ann(fcx.ccx, parent, pres) || changed;

    auto post = tritv_clone(expr_poststate(fcx.ccx, e));
    alt (c) {
        case (none) {}
        case (some(?c1)) {
            set_in_poststate_(bit_num(fcx, c1), post);
        }
    }

    changed = set_poststate_ann(fcx.ccx, parent, post) || changed;
    ret changed;
}

fn find_pre_post_state_two(&fn_ctxt fcx, &prestate pres, &@expr a, &@expr b,
                           node_id parent, oper_type op) -> bool {
    auto changed = set_prestate_ann(fcx.ccx, parent, pres);
    changed = find_pre_post_state_expr(fcx, pres, a) || changed;
    changed = find_pre_post_state_expr(fcx, expr_poststate(fcx.ccx, a), b)
        || changed;

    // for termination, need to make sure intermediate changes don't set
    // changed flag
    auto post = tritv_clone(expr_poststate(fcx.ccx, b));
    alt (op) {
        case (oper_move) {
            forget_in_poststate(fcx, post, b.id);
            gen_if_local(fcx, post, a); 
        }
        case (oper_swap) {
            forget_in_poststate_still_init(fcx, post, a.id);
            forget_in_poststate_still_init(fcx, post, b.id);
        }
        case (oper_assign) {
            forget_in_poststate_still_init(fcx, post, a.id);
            gen_if_local(fcx, post, a); 
        }
        case (_) {}
    }
    changed = set_poststate_ann(fcx.ccx, parent, post) || changed;
    ret changed;
}

fn find_pre_post_state_call(&fn_ctxt fcx, &prestate pres, &@expr a,
                            node_id id, &vec[@expr] bs,
                            controlflow cf) -> bool {
    auto changed = find_pre_post_state_expr(fcx, pres, a);
    ret find_pre_post_state_exprs(fcx,
          expr_poststate(fcx.ccx, a), id, bs, cf) || changed;
}

fn find_pre_post_state_exprs(&fn_ctxt fcx, &prestate pres, ast::node_id id,
                             &vec[@expr] es, controlflow cf) -> bool {
    auto rs = seq_states(fcx, pres, es);
    auto changed = rs._0 | set_prestate_ann(fcx.ccx, id, pres);
    /* if this is a failing call, it sets everything as initialized */
    alt (cf) {
        case (noreturn) {
            changed |= set_poststate_ann
                (fcx.ccx, id, false_postcond(num_constraints(fcx.enclosing)));
        }
        case (_) { 
            changed |= set_poststate_ann(fcx.ccx, id, rs._1);
        }
    }
    ret changed;
}

fn find_pre_post_state_loop(&fn_ctxt fcx, prestate pres, &@local l,
                            &@expr index, &block body, node_id id) -> bool {
    auto loop_pres = intersect_states(pres,
                                      block_poststate(fcx.ccx, body));

    auto changed = set_prestate_ann(fcx.ccx, id, loop_pres) |
        find_pre_post_state_expr(fcx, pres, index);
        find_pre_post_state_block(fcx, expr_poststate(fcx.ccx, index), body);

    if (has_nonlocal_exits(body)) { 
        // See [Break-unsound]
        ret (changed | set_poststate_ann(fcx.ccx, id, pres));
    }
    else {
        auto res_p = intersect_states(expr_poststate(fcx.ccx, index),
                                      block_poststate(fcx.ccx, body));
    /*
    auto res_p =
        intersect_postconds([expr_poststate(fcx.ccx, index),
        block_poststate(fcx.ccx, body)]); */

        ret changed | set_poststate_ann(fcx.ccx, id, res_p);
    }
}

fn gen_if_local(&fn_ctxt fcx, &poststate p, &@expr e) -> bool {
    alt (e.node) {
        case (expr_path(?pth)) { 
            alt (node_id_to_def(fcx.ccx, e.id)) {
                case (some(def_local(?loc))) {
                    ret set_in_poststate_ident(fcx, loc._1,
                           path_to_ident(fcx.ccx.tcx, pth), p);
                }
            case (_) { ret false; }
            }
        }
    case (_) { ret false; }
    }
}

fn join_then_else(&fn_ctxt fcx, &@expr antec, &block conseq,
                  &option::t[@expr] maybe_alt, ast::node_id id, &if_ty chk,
                  &prestate pres) -> bool {
    auto changed = set_prestate_ann(fcx.ccx, id, pres) |
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

    alt (maybe_alt) {
        case (none) {
            changed |= find_pre_post_state_block
                (fcx, expr_poststate(fcx.ccx, antec), conseq) |
                set_poststate_ann(fcx.ccx, id,
                                  expr_poststate(fcx.ccx, antec));
        }
        case (some(?altern)) {
            changed |= find_pre_post_state_expr
                (fcx, expr_poststate(fcx.ccx, antec), altern);

            auto conseq_prestate = expr_poststate(fcx.ccx, antec);
            alt (chk) {
                case (if_check) {
                    let aux::constr c = expr_to_constr(fcx.ccx.tcx, antec);
                    conseq_prestate = tritv_clone(conseq_prestate);
                    tritv_set(bit_num(fcx, c.node), conseq_prestate, ttrue);
                }
                case (_) {}
            }


            changed |=
                find_pre_post_state_block(fcx, conseq_prestate, conseq);
   
            auto poststate_res =
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

fn find_pre_post_state_expr(&fn_ctxt fcx, &prestate pres, @expr e) -> bool {
    auto num_constrs = num_constraints(fcx.enclosing);

    alt (e.node) {
        case (expr_vec(?elts, _, _)) {
            ret find_pre_post_state_exprs(fcx, pres, e.id, elts, return);
        }
        case (expr_tup(?elts)) {
            ret find_pre_post_state_exprs(fcx, pres, e.id,
                                          elt_exprs(elts), return);
        }
        case (expr_call(?operator, ?operands)) {
            ret find_pre_post_state_call
                (fcx, pres, operator, e.id, operands,
                 controlflow_expr(fcx.ccx, operator));
        }
        case (expr_spawn(_, _, ?operator, ?operands)) {
            ret find_pre_post_state_call(fcx, pres, operator, e.id, operands,
                                         return);
        }
        case (expr_bind(?operator, ?maybe_args)) {
            ret find_pre_post_state_call(fcx, pres, operator, e.id,
                                         cat_options(maybe_args), return);
        }
        case (expr_path(_)) { ret pure_exp(fcx.ccx, e.id, pres); }
        case (expr_log(_, ?ex)) {
            ret find_pre_post_state_sub(fcx, pres, ex, e.id, none);
        }
        case (expr_chan(?ex)) {
            ret find_pre_post_state_sub(fcx, pres, ex, e.id, none);
        }
        case (expr_ext(_, _, _, ?expanded)) {
            ret find_pre_post_state_sub(fcx, pres, expanded, e.id, none);
        }
        case (expr_put(?maybe_e)) {
            alt (maybe_e) {
                case (some(?arg)) {
                    ret find_pre_post_state_sub(fcx, pres, arg, e.id, none);
                }
                case (none) { ret pure_exp(fcx.ccx, e.id, pres); }
            }
        }
        case (expr_lit(?l)) { ret pure_exp(fcx.ccx, e.id, pres); }
        case (expr_fn(?f)) { ret pure_exp(fcx.ccx, e.id, pres); }
        case (expr_block(?b)) {
            ret find_pre_post_state_block(fcx, pres, b) |
                set_prestate_ann(fcx.ccx, e.id, pres) |
                set_poststate_ann(fcx.ccx, e.id, block_poststate(fcx.ccx, b));
        }
        case (expr_rec(?fields, ?maybe_base)) {
            auto changed = find_pre_post_state_exprs
                (fcx, pres, e.id, field_exprs(fields), return);
            alt (maybe_base) {
                case (none) {/* do nothing */ }
                case (some(?base)) {
                    changed |= find_pre_post_state_expr(fcx, pres, base) |
                        set_poststate_ann(fcx.ccx, e.id,
                                          expr_poststate(fcx.ccx, base));
                }
            }
            ret changed;
        }
        case (expr_move(?lhs, ?rhs)) {
            ret find_pre_post_state_two(fcx, pres, lhs, rhs,
                                        e.id, oper_move);
        }
        case (expr_assign(?lhs, ?rhs)) {
            ret find_pre_post_state_two(fcx, pres, lhs, rhs,
                                        e.id, oper_assign);
        }
        case (expr_swap(?lhs, ?rhs)) {
           ret find_pre_post_state_two(fcx, pres, lhs, rhs, e.id,
                                       oper_swap);
             // Could be more precise and actually swap the role of
             // lhs and rhs in constraints
        }
        case (expr_recv(?lhs, ?rhs)) {
            // Opposite order as most other binary operations,
            // so not using find_pre_post_state_two
            auto changed = set_prestate_ann(fcx.ccx, e.id, pres) |
                find_pre_post_state_expr(fcx, pres, lhs) |
                find_pre_post_state_expr
                (fcx, expr_poststate(fcx.ccx, lhs), rhs);
            auto post = tritv_clone(expr_poststate(fcx.ccx, rhs));
            forget_in_poststate_still_init(fcx, post, rhs.id);
            gen_if_local(fcx, post, rhs); 
            ret changed | set_poststate_ann(fcx.ccx, e.id, post);
        }
        case (expr_ret(?maybe_ret_val)) {
            auto changed = set_prestate_ann(fcx.ccx, e.id, pres);
            /* normally, everything is true if execution continues after
               a ret expression (since execution never continues locally
               after a ret expression */

            set_poststate_ann(fcx.ccx, e.id, false_postcond(num_constrs));
            /* return from an always-failing function clears the return bit */

            alt (fcx.enclosing.cf) {
                case (noreturn) {
                    kill_poststate(fcx, e.id, rec(id=fcx.id,
                                                  c=ninit(fcx.name)));
                }
                case (_) { }
            }
            alt (maybe_ret_val) {
                case (none) {/* do nothing */ }
                case (some(?ret_val)) {
                    changed |= find_pre_post_state_expr(fcx, pres, ret_val);
                }
            }
            ret changed;
        }
        case (expr_be(?val)) {
            auto changed = set_prestate_ann(fcx.ccx, e.id, pres);
            set_poststate_ann(fcx.ccx, e.id, false_postcond(num_constrs));
            ret changed | find_pre_post_state_expr(fcx, pres, val);
        }
        case (expr_if(?antec, ?conseq, ?maybe_alt)) {
            ret join_then_else
                (fcx, antec, conseq, maybe_alt, e.id, plain_if, pres);
        }
        case (expr_ternary(_, _, _)) {
            ret find_pre_post_state_expr(fcx, pres, ternary_to_if(e));
        }
        case (expr_binary(?bop, ?l, ?r)) {
            if (lazy_binop(bop)) {
                auto changed = find_pre_post_state_expr(fcx, pres, l);
                changed |= find_pre_post_state_expr(fcx,
                              expr_poststate(fcx.ccx, l), r);
                ret changed
                    | set_prestate_ann(fcx.ccx, e.id, pres)
                    | set_poststate_ann(fcx.ccx, e.id,
                                        expr_poststate(fcx.ccx, l));
            }
            else {
                ret find_pre_post_state_two(fcx, pres, l, r, e.id, oper_pure);
            }
        }
        case (expr_send(?l, ?r)) {
            ret find_pre_post_state_two(fcx, pres, l, r, e.id, oper_pure);
        }
        case (expr_assign_op(?op, ?lhs, ?rhs)) {
            ret find_pre_post_state_two(fcx, pres, lhs, rhs, e.id,
                                        oper_assign);
        }
        case (expr_while(?test, ?body)) {
            /*
            log_err "in a while loop:";
            log_expr_err(*e);
            aux::log_tritv_err(fcx, block_poststate(fcx.ccx, body));
            aux::log_tritv_err(fcx, pres);
            */
            auto loop_pres = intersect_states
                (block_poststate(fcx.ccx, body), pres);
            // aux::log_tritv_err(fcx, loop_pres);
            // log_err "---------------";

            auto changed = set_prestate_ann(fcx.ccx, e.id, loop_pres) |
                find_pre_post_state_expr(fcx, loop_pres, test) |
                find_pre_post_state_block(fcx, expr_poststate(fcx.ccx, test),
                                          body);
            /* conservative approximation: if a loop contains a break
               or cont, we assume nothing about the poststate */
            /* which is still unsound -- see [Break-unsound] */
            if (has_nonlocal_exits(body)) { 
                ret changed | set_poststate_ann(fcx.ccx, e.id, pres);
            }
            else {
                auto e_post = expr_poststate(fcx.ccx, test);
                auto b_post = block_poststate(fcx.ccx, body);
                ret changed | set_poststate_ann
                    (fcx.ccx, e.id, intersect_states(e_post, b_post));
            }
        }
        case (expr_do_while(?body, ?test)) {
            auto loop_pres = intersect_states(expr_poststate(fcx.ccx, test),
                                              pres);

            auto changed = set_prestate_ann(fcx.ccx, e.id, loop_pres);
            changed |= find_pre_post_state_block(fcx, loop_pres, body);
            /* conservative approximination: if the body of the loop
               could break or cont, we revert to the prestate
               (TODO: could treat cont differently from break, since
               if there's a cont, the test will execute) */

            changed |= find_pre_post_state_expr
                (fcx, block_poststate(fcx.ccx, body), test);

            auto breaks = has_nonlocal_exits(body);
            if (breaks) {
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
             }
            else {
                changed |= set_poststate_ann
                    (fcx.ccx, e.id, expr_poststate(fcx.ccx, test));
            }
            ret changed;
        }
        case (expr_for(?d, ?index, ?body)) {
            ret find_pre_post_state_loop(fcx, pres, d, index, body, e.id);
        }
        case (expr_for_each(?d, ?index, ?body)) {
            ret find_pre_post_state_loop(fcx, pres, d, index, body, e.id);
        }
        case (expr_index(?val, ?sub)) {
            ret find_pre_post_state_two(fcx, pres, val, sub, e.id, oper_pure);
        }
        case (expr_alt(?val, ?alts)) {
            auto changed = set_prestate_ann(fcx.ccx, e.id, pres) |
                find_pre_post_state_expr(fcx, pres, val);
            auto e_post = expr_poststate(fcx.ccx, val);
            auto a_post;
            if (vec::len(alts) > 0u) {
                a_post = false_postcond(num_constrs);
                for (arm an_alt in alts) {
                    changed |= find_pre_post_state_block
                        (fcx, e_post, an_alt.block);
                    intersect(a_post, block_poststate(fcx.ccx, an_alt.block));
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
        case (expr_field(?val, _)) {
            ret find_pre_post_state_sub(fcx, pres, val, e.id, none);
        }
        case (expr_unary(_, ?operand)) {
            ret find_pre_post_state_sub(fcx, pres, operand, e.id, none);
        }
        case (expr_cast(?operand, _)) {
            ret find_pre_post_state_sub(fcx, pres, operand, e.id, none);
        }
        case (expr_fail(_)) {
            ret set_prestate_ann(fcx.ccx, e.id, pres) |
            /* if execution continues after fail, then everything is true!
               woo! */
                set_poststate_ann(fcx.ccx, e.id,
                                  false_postcond(num_constrs));
        }
        case (expr_assert(?p)) {
            ret find_pre_post_state_sub(fcx, pres, p, e.id, none);
        }
        case (expr_check(_, ?p)) {
            /* predicate p holds after this expression executes */
            let aux::constr c = expr_to_constr(fcx.ccx.tcx, p);
            ret find_pre_post_state_sub(fcx, pres, p, e.id, some(c.node));
        }
        case (expr_if_check(?p, ?conseq, ?maybe_alt)) {
            ret join_then_else
                (fcx, p, conseq, maybe_alt, e.id, if_check, pres);
        }
        case (expr_break) { ret pure_exp(fcx.ccx, e.id, pres); }
        case (expr_cont) { ret pure_exp(fcx.ccx, e.id, pres); }
        case (expr_port(_)) { ret pure_exp(fcx.ccx, e.id, pres); }
        case (expr_self_method(_)) { ret pure_exp(fcx.ccx, e.id, pres); }
        case (expr_anon_obj(?anon_obj, _, _)) {
            alt (anon_obj.with_obj) {
                case (some(?wt)) {
                    ret find_pre_post_state_sub(fcx, pres, wt, e.id, none);
                }
                case (none) { ret pure_exp(fcx.ccx, e.id, pres); }
            }
        }
    }
}

fn find_pre_post_state_stmt(&fn_ctxt fcx, &prestate pres, @stmt s) -> bool {
    auto stmt_ann = stmt_to_ann(fcx.ccx, *s);

    /*    
    log_err "*At beginning: stmt = ";
    log_stmt_err(*s);
    log_err "*prestate = ";
    log_err tritv::to_str(stmt_ann.states.prestate);
    log_err "*poststate =";
    log_err tritv::to_str(stmt_ann.states.poststate);
    log_err "pres = ";
    log_err tritv::to_str(pres);
    */

    alt (s.node) {
        case (stmt_decl(?adecl, ?id)) {
            alt (adecl.node) {
                case (decl_local(?alocal)) {
                    alt (alocal.node.init) {
                        case (some(?an_init)) {
                            auto changed = set_prestate(stmt_ann, pres) |
                                find_pre_post_state_expr(fcx, pres,
                                                         an_init.expr);

                            auto post = tritv_clone(expr_poststate(fcx.ccx,
                                                      an_init.expr));
                            if (an_init.op == init_move) {
                                clear_in_poststate_expr(fcx, an_init.expr,
                                                        post);
                            }

                            set_in_poststate_ident(fcx, alocal.node.id,
                                                   alocal.node.ident, post);

                            /* important to do this in one step to ensure
                               termination (don't want to set changed to true
                               for intermediate changes) */
                            ret changed | set_poststate(stmt_ann, post);

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
                        }
                        case (none) {
                            // let int = x; => x is uninit in poststate
                            set_poststate_ann(fcx.ccx, id, pres);
                            clear_in_poststate_ident(fcx, alocal.node.id,
                                                         alocal.node.ident, id);
                            set_prestate(stmt_ann, pres);
                            ret false;
                        }
                    }
                }
                case (decl_item(?an_item)) {
                    ret set_prestate(stmt_ann, pres) |
                        set_poststate(stmt_ann, pres);
                    /* the outer "walk" will recurse into the item */
                }
            }
        }
        case (stmt_expr(?ex, _)) {
            ret find_pre_post_state_expr(fcx, pres, ex) |
                set_prestate(stmt_ann, expr_prestate(fcx.ccx, ex)) |
                set_poststate(stmt_ann, expr_poststate(fcx.ccx, ex));
            /*
            log_err "Finally:";
              log_stmt_err(*s);
              log_err("prestate = ");
              //              log_err(bitv::to_str(stmt_ann.states.prestate));
              log_tritv_err(fcx, stmt_ann.states.prestate);
              log_err("poststate =");
              //   log_err(bitv::to_str(stmt_ann.states.poststate));
              log_tritv_err(fcx, stmt_ann.states.poststate);
              log_err("changed =");
              log_err(changed);
            */
        }
        case (_) { ret false; }
    }
}


/* Updates the pre- and post-states of statements in the block,
   returns a boolean flag saying whether any pre- or poststates changed */
fn find_pre_post_state_block(&fn_ctxt fcx, &prestate pres0, &block b)
    -> bool {
    auto num_local_vars = num_constraints(fcx.enclosing);
    /* First, set the pre-states and post-states for every expression */

    auto pres = pres0;
    /* Iterate over each stmt. The new prestate is <pres>. The poststate
     consist of improving <pres> with whatever variables this stmt
     initializes.  Then <pres> becomes the new poststate. */

    auto changed = false;
    for (@stmt s in b.node.stmts) {
        changed |= find_pre_post_state_stmt(fcx, pres, s);
        pres = stmt_poststate(fcx.ccx, *s);
    }
    auto post = pres;
    alt (b.node.expr) {
        case (none) { }
        case (some(?e)) {
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
    */

    ret changed;
}

fn find_pre_post_state_fn(&fn_ctxt fcx, &_fn f) -> bool {
    auto num_local_vars = num_constraints(fcx.enclosing);
    // make sure the return bit starts out False
    clear_in_prestate_ident(fcx, fcx.id, fcx.name, f.body.node.id);
    auto changed =
        find_pre_post_state_block(fcx, block_prestate(fcx.ccx, f.body),
                                  f.body);
    // Treat the tail expression as a return statement

    alt (f.body.node.expr) {
        case (some(?tailexpr)) {
            auto tailty = expr_ty(fcx.ccx.tcx, tailexpr);

            // Since blocks and alts and ifs that don't have results
            // implicitly result in nil, we have to be careful to not
            // interpret nil-typed block results as the result of a
            // function with some other return type
            if (!type_is_nil(fcx.ccx.tcx, tailty) &&
                    !type_is_bot(fcx.ccx.tcx, tailty)) {
                auto p = false_postcond(num_local_vars);
                set_poststate_ann(fcx.ccx, f.body.node.id, p);
            }
        }
        case (none) {/* fallthrough */ }
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
