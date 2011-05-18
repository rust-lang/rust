import std::vec;
import std::vec::plus_option;
import std::option;
import std::option::none;
import std::option::some;

import tstate::ann::pre_and_post;
import tstate::ann::get_post;
import tstate::ann::postcond;
import tstate::ann::true_precond;
import tstate::ann::false_postcond;
import tstate::ann::empty_pre_post;
import tstate::ann::empty_poststate;
import tstate::ann::require_and_preserve;
import tstate::ann::union;
import tstate::ann::intersect;
import tstate::ann::pp_clone;
import tstate::ann::empty_prestate;
import aux::var_info;
import aux::crate_ctxt;
import aux::fn_ctxt;
import aux::num_locals;
import aux::expr_pp;
import aux::stmt_pp;
import aux::block_pp;
import aux::set_pre_and_post;
import aux::expr_precond;
import aux::expr_postcond;
import aux::expr_prestate;
import aux::expr_poststate;
import aux::block_postcond;
import aux::fn_info;
import aux::log_pp;
import aux::ann_to_def;
import aux::ann_to_def_strict;

import bitvectors::seq_preconds;
import bitvectors::union_postconds;
import bitvectors::intersect_postconds;
import bitvectors::declare_var;
import bitvectors::bit_num;
import bitvectors::gen;

import front::ast::_mod;
import front::ast;
import front::ast::method;
import front::ast::ann;
import front::ast::ty;
import front::ast::mutability;
import front::ast::item_const;
import front::ast::item_mod;
import front::ast::item_ty;
import front::ast::item_tag;
import front::ast::item_native_mod;
import front::ast::obj_field;
import front::ast::stmt;
import front::ast::stmt_;
import front::ast::ident;
import front::ast::def_id;
import front::ast::expr;
import front::ast::path;
import front::ast::crate_directive;
import front::ast::fn_decl;
import front::ast::native_mod;
import front::ast::variant;
import front::ast::ty_param;
import front::ast::proto;
import front::ast::pat;
import front::ast::binop;
import front::ast::unop;
import front::ast::def;
import front::ast::lit;
import front::ast::init_op;
import front::ast::controlflow;
import front::ast::return;
import front::ast::noreturn;
import front::ast::_fn;
import front::ast::_obj;
import front::ast::crate;
import front::ast::item_fn;
import front::ast::item_obj;
import front::ast::def_local;
import front::ast::def_fn;
import front::ast::item;
import front::ast::elt;
import front::ast::field;
import front::ast::decl;
import front::ast::decl_local;
import front::ast::decl_item;
import front::ast::initializer;
import front::ast::local;
import front::ast::arm;
import front::ast::expr_call;
import front::ast::expr_vec;
import front::ast::expr_tup;
import front::ast::expr_path;
import front::ast::expr_field;
import front::ast::expr_index;
import front::ast::expr_log;
import front::ast::expr_block;
import front::ast::expr_rec;
import front::ast::expr_if;
import front::ast::expr_binary;
import front::ast::expr_unary;
import front::ast::expr_assign;
import front::ast::expr_assign_op;
import front::ast::expr_while;
import front::ast::expr_do_while;
import front::ast::expr_alt;
import front::ast::expr_lit;
import front::ast::expr_ret;
import front::ast::expr_self_method;
import front::ast::expr_bind;
import front::ast::expr_spawn;
import front::ast::expr_ext;
import front::ast::expr_fail;
import front::ast::expr_break;
import front::ast::expr_cont;
import front::ast::expr_send;
import front::ast::expr_recv;
import front::ast::expr_put;
import front::ast::expr_port;
import front::ast::expr_chan;
import front::ast::expr_be;
import front::ast::expr_check;
import front::ast::expr_assert;
import front::ast::expr_cast;
import front::ast::expr_for;
import front::ast::expr_for_each;
import front::ast::stmt_decl;
import front::ast::stmt_expr;
import front::ast::block;
import front::ast::block_;

import middle::fold::span;
import middle::fold::respan;

import util::common::new_def_hash;
import util::common::decl_lhs;
import util::common::uistr;
import util::common::log_expr;
import util::common::log_fn;
import util::common::elt_exprs;
import util::common::field_exprs;
import util::common::has_nonlocal_exits;
import util::common::log_stmt;
import util::common::log_expr_err;

fn find_pre_post_mod(&_mod m) -> _mod {
    log("implement find_pre_post_mod!");
    fail;
}

fn find_pre_post_native_mod(&native_mod m) -> native_mod {
    log("implement find_pre_post_native_mod");
    fail;
}


fn find_pre_post_obj(&crate_ctxt ccx, _obj o) -> () {
    fn do_a_method(crate_ctxt ccx, &@method m) -> () {
        assert (ccx.fm.contains_key(m.node.id));
        let fn_ctxt fcx = rec(enclosing=ccx.fm.get(m.node.id),
                       id=m.node.id, name=m.node.ident, ccx=ccx);
        find_pre_post_fn(fcx, m.node.meth);
    }
    auto f = bind do_a_method(ccx,_);
    vec::map[@method, ()](f, o.methods);
    option::map[@method, ()](f, o.dtor);
}

fn find_pre_post_item(&crate_ctxt ccx, &item i) -> () {
    alt (i.node) {
        case (item_const(?id, ?t, ?e, ?di, ?a)) {
            // make a fake fcx
            auto fake_fcx = rec(enclosing=rec(vars=@new_def_hash[var_info](),
                                              cf=return),
                                id=tup(0,0),
                                name="",
                                ccx=ccx);
            find_pre_post_expr(fake_fcx, e);
        }
        case (item_fn(?id, ?f, ?ps, ?di, ?a)) {
            assert (ccx.fm.contains_key(di));
            auto fcx = rec(enclosing=ccx.fm.get(di),
                           id=di, name=id, ccx=ccx);
            find_pre_post_fn(fcx, f);
        }
        case (item_mod(?id, ?m, ?di)) {
            find_pre_post_mod(m);
        }
        case (item_native_mod(?id, ?nm, ?di)) {
            find_pre_post_native_mod(nm);
        }
        case (item_ty(_,_,_,_,_)) {
            ret;
        }
        case (item_tag(_,_,_,_,_)) {
            ret;
        }
        case (item_obj(?id, ?o, ?ps, ?di, ?a)) {
            find_pre_post_obj(ccx, o);
        }
    }
}

/* Finds the pre and postcondition for each expr in <args>;
   sets the precondition in a to be the result of combining
   the preconditions for <args>, and the postcondition in a to 
   be the union of all postconditions for <args> */
fn find_pre_post_exprs(&fn_ctxt fcx, &vec[@expr] args, ann a) {
    auto enclosing = fcx.enclosing;
    auto fm        = fcx.ccx.fm;
    auto nv        = num_locals(enclosing);

    fn do_one(fn_ctxt fcx, &@expr e) -> () {
        find_pre_post_expr(fcx, e);
    }
    auto f = bind do_one(fcx, _);

    vec::map[@expr, ()](f, args);

    fn get_pp(&@expr e) -> pre_and_post {
        ret expr_pp(e);
    }
    auto g = get_pp;
    auto pps = vec::map[@expr, pre_and_post](g, args);
    auto h = get_post;

    set_pre_and_post(a,
       rec(precondition=seq_preconds(enclosing, pps),
           postcondition=union_postconds
           (nv, (vec::map[pre_and_post, postcond](h, pps)))));
}

fn find_pre_post_loop(&fn_ctxt fcx, &@decl d, &@expr index,
      &block body, &ann a) -> () {
    find_pre_post_expr(fcx, index);
    find_pre_post_block(fcx, body);
    auto loop_precond = declare_var(fcx.enclosing, decl_lhs(d),
           seq_preconds(fcx.enclosing, [expr_pp(index),
                                       block_pp(body)]));
    auto loop_postcond = intersect_postconds
        ([expr_postcond(index), block_postcond(body)]);
    set_pre_and_post(a, rec(precondition=loop_precond,
                            postcondition=loop_postcond));
}

fn gen_if_local(&fn_ctxt fcx, @expr lhs, @expr rhs, &ann larger_ann, &ann new_var) -> () {
  alt (ann_to_def(fcx.ccx, new_var)) {
    case (some[def](def_local(?d_id))) {
      find_pre_post_expr(fcx, rhs);
      set_pre_and_post(larger_ann, expr_pp(rhs));
      gen(fcx, larger_ann, d_id);
    }
    case (_) { find_pre_post_exprs(fcx, [lhs, rhs], larger_ann); }
  }
}

/* Fills in annotations as a side effect. Does not rebuild the expr */
fn find_pre_post_expr(&fn_ctxt fcx, @expr e) -> () {
    auto enclosing      = fcx.enclosing;
    auto num_local_vars = num_locals(enclosing);

    fn do_rand_(fn_ctxt fcx, &@expr e) -> () {
        find_pre_post_expr(fcx, e);
    }
    fn pp_one(&@expr e) -> pre_and_post {
        ret expr_pp(e);
    }
    
    /*
    log_err("find_pre_post_expr (num_locals =" +
        uistr(num_local_vars) + "):");
    log_expr_err(*e);
    */

    alt (e.node) {
        case (expr_call(?operator, ?operands, ?a)) {
            auto args = vec::clone[@expr](operands);
            vec::push[@expr](args, operator);
            find_pre_post_exprs(fcx, args, a);
        }
        case (expr_spawn(_, _, ?operator, ?operands, ?a)) {
            auto args = vec::clone[@expr](operands);
            vec::push[@expr](args, operator);
            find_pre_post_exprs(fcx, args, a);
        }
        case (expr_vec(?args, _, ?a)) {
            find_pre_post_exprs(fcx, args, a);
        }
        case (expr_tup(?elts, ?a)) {
            find_pre_post_exprs(fcx, elt_exprs(elts), a);
        }
        case (expr_path(?p, ?a)) {
            auto res = empty_pre_post(num_local_vars);
	    
	    auto df = ann_to_def_strict(fcx.ccx, a);
            alt (df) {
                case (def_local(?d_id)) {
                    auto i = bit_num(d_id, enclosing);
                    require_and_preserve(i, res);
                }
                case (_) { /* nothing to check */ }
            }

            // Otherwise, variable is global, so it must be initialized
            set_pre_and_post(a, res);
        }
        case (expr_self_method(?v, ?a)) {
            /* v is a method of the enclosing obj, so it must be
               initialized, right? */
            set_pre_and_post(a, empty_pre_post(num_local_vars));
        }
        case(expr_log(_, ?arg, ?a)) {
            find_pre_post_expr(fcx, arg);
            set_pre_and_post(a, expr_pp(arg));
        }
        case (expr_chan(?arg, ?a)) {
            find_pre_post_expr(fcx, arg);
            set_pre_and_post(a, expr_pp(arg));
        }
        case(expr_put(?opt, ?a)) {
            alt (opt) {
                case (some[@expr](?arg)) {
                    find_pre_post_expr(fcx, arg);
                    set_pre_and_post(a, expr_pp(arg));
                }
                case (none[@expr]) {
                    set_pre_and_post(a, empty_pre_post(num_local_vars));
                }
            }
        }
        case (expr_block(?b, ?a)) {
            find_pre_post_block(fcx, b);
            set_pre_and_post(a, block_pp(b));
        }
        case (expr_rec(?fields,?maybe_base,?a)) {
            auto es = field_exprs(fields);
            vec::plus_option[@expr](es, maybe_base);
            find_pre_post_exprs(fcx, es, a);
        }
        case (expr_assign(?lhs, ?rhs, ?a)) {
            alt (lhs.node) {
                case (expr_path(?p, ?a_lhs)) {
		  gen_if_local(fcx, lhs, rhs, a, a_lhs);
                }
                case (_) {
                    find_pre_post_exprs(fcx, [lhs, rhs], a);
                }
            }
        }
        case (expr_recv(?lhs, ?rhs, ?a)) {
            alt (lhs.node) {
                case (expr_path(?p, ?a_lhs)) {
		  gen_if_local(fcx, lhs, rhs, a, a_lhs);
                }
                case (_) {
                    // doesn't check that lhs is an lval, but
                    // that's probably ok
                    find_pre_post_exprs(fcx, [lhs, rhs], a);
                }
            }
        }
        case (expr_assign_op(_, ?lhs, ?rhs, ?a)) {
            /* Different from expr_assign in that the lhs *must*
               already be initialized */
            find_pre_post_exprs(fcx, [lhs, rhs], a);
        }
        case (expr_lit(_,?a)) {
            set_pre_and_post(a, empty_pre_post(num_local_vars));
        }
        case (expr_ret(?maybe_val, ?a)) {
            alt (maybe_val) {
                case (none[@expr]) {
                    set_pre_and_post(a,
                      rec(precondition=true_precond(num_local_vars),
                          postcondition=false_postcond(num_local_vars)));
                }
                case (some[@expr](?ret_val)) {
                    find_pre_post_expr(fcx, ret_val);
                    let pre_and_post pp =
                        rec(precondition=expr_precond(ret_val),
                            postcondition=false_postcond(num_local_vars));
                    set_pre_and_post(a, pp);
                }
            }
        }
        case (expr_be(?e, ?a)) {
            find_pre_post_expr(fcx, e);
            set_pre_and_post(a,
               rec(precondition=expr_prestate(e),
                   postcondition=false_postcond(num_local_vars)));
        }
        case (expr_if(?antec, ?conseq, ?maybe_alt, ?a)) {
            find_pre_post_expr(fcx, antec);
            find_pre_post_block(fcx, conseq);
            alt (maybe_alt) {
                case (none[@expr]) {
                    auto precond_res = seq_preconds(enclosing,
                                                    [expr_pp(antec),
                                                        block_pp(conseq)]);
                    set_pre_and_post(a, rec(precondition=precond_res,
                                            postcondition=
                                            expr_poststate(antec)));
                }
                case (some[@expr](?altern)) {
                    find_pre_post_expr(fcx, altern);
                    auto precond_true_case =
                        seq_preconds(enclosing,
                                     [expr_pp(antec), block_pp(conseq)]);
                    auto postcond_true_case = union_postconds
                        (num_local_vars,
                         [expr_postcond(antec), block_postcond(conseq)]);
                    auto precond_false_case = seq_preconds
                        (enclosing,
                         [expr_pp(antec), expr_pp(altern)]);
                    auto postcond_false_case = union_postconds
                        (num_local_vars,
                         [expr_postcond(antec), expr_postcond(altern)]);
                    auto precond_res = union_postconds
                        (num_local_vars,
                         [precond_true_case, precond_false_case]);
                    auto postcond_res = intersect_postconds
                        ([postcond_true_case, postcond_false_case]);
                    set_pre_and_post(a, rec(precondition=precond_res,
                                            postcondition=postcond_res));
                }
            }
        }
        case (expr_binary(?bop,?l,?r,?a)) {
            /* *unless* bop is lazy (e.g. and, or)? 
               FIXME */
            find_pre_post_exprs(fcx, [l, r], a);
        }
        case (expr_send(?l, ?r, ?a)) {
            find_pre_post_exprs(fcx, [l, r], a);
        }
        case (expr_unary(_,?operand,?a)) {
            find_pre_post_expr(fcx, operand);
            set_pre_and_post(a, expr_pp(operand));
        }
        case (expr_cast(?operand, _, ?a)) {
            find_pre_post_expr(fcx, operand);
            set_pre_and_post(a, expr_pp(operand));
        }
        case (expr_while(?test, ?body, ?a)) {
            find_pre_post_expr(fcx, test);
            find_pre_post_block(fcx, body);
            set_pre_and_post(a,
              rec(precondition=
                    seq_preconds(enclosing,
                               [expr_pp(test), 
                                   block_pp(body)]),
                  postcondition=
                    intersect_postconds([expr_postcond(test),
                                            block_postcond(body)])));
        }
        case (expr_do_while(?body, ?test, ?a)) {
            find_pre_post_block(fcx, body);
            find_pre_post_expr(fcx, test);
   
            auto loop_postcond = union_postconds(num_local_vars,
                   [block_postcond(body), expr_postcond(test)]);
            /* conservative approximination: if the body
               could break or cont, the test may never be executed */
            if (has_nonlocal_exits(body)) {
                loop_postcond = empty_poststate(num_local_vars);
            }

            set_pre_and_post(a, 
              rec(precondition=seq_preconds(enclosing,
                                            [block_pp(body),
                                                expr_pp(test)]),
                  postcondition=loop_postcond));
        }
        case (expr_for(?d, ?index, ?body, ?a)) {
            find_pre_post_loop(fcx, d, index, body, a);
        }
        case (expr_for_each(?d, ?index, ?body, ?a)) {
            find_pre_post_loop(fcx, d, index, body, a);
        }
        case (expr_index(?e, ?sub, ?a)) {
            find_pre_post_exprs(fcx, [e, sub], a);
        }
        case (expr_alt(?e, ?alts, ?a)) {
            find_pre_post_expr(fcx, e);
            fn do_an_alt(&fn_ctxt fcx, &arm an_alt) -> pre_and_post {
                find_pre_post_block(fcx, an_alt.block);
                ret block_pp(an_alt.block);
            }
            auto f = bind do_an_alt(fcx, _);
            auto alt_pps = vec::map[arm, pre_and_post](f, alts);
            fn combine_pp(pre_and_post antec, 
                          fn_info enclosing, &pre_and_post pp,
                          &pre_and_post next) -> pre_and_post {
                union(pp.precondition, seq_preconds(enclosing,
                                                    [antec, next]));
                intersect(pp.postcondition, next.postcondition);
                ret pp;
            }
            auto antec_pp = pp_clone(expr_pp(e)); 
            auto e_pp  = rec(precondition=empty_prestate(num_local_vars),
                             postcondition=false_postcond(num_local_vars));
            auto g = bind combine_pp(antec_pp, fcx.enclosing, _, _);

            auto alts_overall_pp = vec::foldl[pre_and_post, pre_and_post]
                                    (g, e_pp, alt_pps);

            set_pre_and_post(a, alts_overall_pp);
        }
        case (expr_field(?operator, _, ?a)) {
            find_pre_post_expr(fcx, operator);
            set_pre_and_post(a, expr_pp(operator));
        }
        case (expr_fail(?a)) {
            set_pre_and_post(a,
                             /* if execution continues after fail,
                                then everything is true! */
               rec(precondition=empty_prestate(num_local_vars),
                   postcondition=false_postcond(num_local_vars)));
        }
        case (expr_assert(?p, ?a)) {
            find_pre_post_expr(fcx, p);
            set_pre_and_post(a, expr_pp(p));
        }
        case (expr_check(?p, ?a)) {
            /* will need to change when we support arbitrary predicates... */
            find_pre_post_expr(fcx, p);
            set_pre_and_post(a, expr_pp(p));
        }
        case(expr_bind(?operator, ?maybe_args, ?a)) {
            auto args = vec::cat_options[@expr](maybe_args);
            vec::push[@expr](args, operator); /* ??? order of eval? */
            find_pre_post_exprs(fcx, args, a);
        }
        case (expr_break(?a)) {
            set_pre_and_post(a, empty_pre_post(num_local_vars));
        }
        case (expr_cont(?a)) {
            set_pre_and_post(a, empty_pre_post(num_local_vars));
        }
        case (expr_port(?a)) {
            set_pre_and_post(a, empty_pre_post(num_local_vars));
        }
        case (expr_ext(_, _, _, ?expanded, ?a)) {
            find_pre_post_expr(fcx, expanded);
            set_pre_and_post(a, expr_pp(expanded));
        }
    }
}


fn find_pre_post_stmt(&fn_ctxt fcx, &stmt s)
    -> () {
    log("stmt =");
    log_stmt(s);

    auto enclosing      = fcx.enclosing;
    auto num_local_vars = num_locals(enclosing);
    alt(s.node) {
        case(stmt_decl(?adecl, ?a)) {
            alt(adecl.node) {
                case(decl_local(?alocal)) {
                    alt(alocal.init) {
                        case(some[initializer](?an_init)) {
                            find_pre_post_expr(fcx, an_init.expr);
                            auto rhs_pp = expr_pp(an_init.expr);
                            set_pre_and_post(alocal.ann, rhs_pp);

                            /* Inherit ann from initializer, and add var being
                               initialized to the postcondition */
                            set_pre_and_post(a, rhs_pp);
                            /*  log("gen (decl):");
                                log_stmt(s); */
                            gen(fcx, a, alocal.id); 
                            /*                     log_err("for stmt");
                                                   log_stmt(s);
                                                   log_err("pp = ");
                                                   log_pp(stmt_pp(s)); */
                        }
                        case(none[initializer]) {
                            auto pp = empty_pre_post(num_local_vars);
                            set_pre_and_post(alocal.ann, pp);
                            set_pre_and_post(a, pp);
                        }
                    }
                }
                case(decl_item(?anitem)) {
                    auto pp = empty_pre_post(num_local_vars);
                    set_pre_and_post(a, pp);
                    find_pre_post_item(fcx.ccx, *anitem);
                }
            }
        }
        case(stmt_expr(?e,?a)) {
            find_pre_post_expr(fcx, e);
            set_pre_and_post(a, expr_pp(e));
        }    
    }
}

fn find_pre_post_block(&fn_ctxt fcx, block b) -> () {
    /* Want to say that if there is a break or cont in this
     block, then that invalidates the poststate upheld by
    any of the stmts after it. 
    Given that the typechecker has run, we know any break will be in
    a block that forms a loop body. So that's ok. There'll never be an
    expr_break outside a loop body, therefore, no expr_break outside a block.
    */

    /* Conservative approximation for now: This says that if a block contains
     *any* breaks or conts, then its postcondition doesn't promise anything.
     This will mean that:
     x = 0;
     break;

     won't have a postcondition that says x is initialized, but that's ok.
     */
    auto nv = num_locals(fcx.enclosing);

    fn do_one_(fn_ctxt fcx, &@stmt s) -> () {
        find_pre_post_stmt(fcx, *s);
        log("pre_post for stmt:");
        log_stmt(*s);
        log("is:");
        log_pp(stmt_pp(*s));
    }
    auto do_one = bind do_one_(fcx, _);
    
    vec::map[@stmt, ()](do_one, b.node.stmts);
    fn do_inner_(fn_ctxt fcx, &@expr e) -> () {
        find_pre_post_expr(fcx, e);
    }
    auto do_inner = bind do_inner_(fcx, _);
    option::map[@expr, ()](do_inner, b.node.expr);

    let vec[pre_and_post] pps = [];

    fn get_pp_stmt(&@stmt s) -> pre_and_post {
        ret stmt_pp(*s);
    }
    auto f = get_pp_stmt;
    pps += vec::map[@stmt, pre_and_post](f, b.node.stmts);
    fn get_pp_expr(&@expr e) -> pre_and_post {
        ret expr_pp(e);
    }
    auto g = get_pp_expr;
    plus_option[pre_and_post](pps,
       option::map[@expr, pre_and_post](g, b.node.expr));

    auto block_precond  = seq_preconds(fcx.enclosing, pps);
    auto h = get_post;
    auto postconds =  vec::map[pre_and_post, postcond](h, pps);
    /* A block may be empty, so this next line ensures that the postconds
       vector is non-empty. */
    vec::push[postcond](postconds, block_precond);
    auto block_postcond = empty_poststate(nv);
    /* conservative approximation */
    if (! has_nonlocal_exits(b)) {
        block_postcond = union_postconds(nv, postconds);
    }

    set_pre_and_post(b.node.a, rec(precondition=block_precond,
                                   postcondition=block_postcond));
}

fn find_pre_post_fn(&fn_ctxt fcx, &_fn f) -> () {
    find_pre_post_block(fcx, f.body);
}

fn check_item_fn(&crate_ctxt ccx, &span sp, &ident i, &_fn f,
                 &vec[ty_param] ty_params,
                 &def_id id, &ann a) -> @item {
    log("check_item_fn:");
    log_fn(f, i, ty_params);

    assert (ccx.fm.contains_key(id));
    auto fcx = rec(enclosing=ccx.fm.get(id),
                   id=id, name=i, ccx=ccx);
    find_pre_post_fn(fcx, f);  

    ret @respan(sp, item_fn(i, f, ty_params, id, a));
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

