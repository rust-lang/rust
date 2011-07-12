import std::vec;
import std::ivec;
import std::int::str;
import std::str;
import std::option;
import std::option::*;
import std::int;
import std::uint;
import syntax::ast::*;
import syntax::codemap::span;
import syntax::visit;
import util::common;
import util::common::log_block;
import std::map::new_int_hash;
import std::map::new_uint_hash;
import util::common::log_expr_err;
import util::common::lit_eq;
import syntax::print::pprust::path_to_str;
import tstate::ann::pre_and_post;
import tstate::ann::pre_and_post_state;
import tstate::ann::empty_ann;
import tstate::ann::prestate;
import tstate::ann::poststate;
import tstate::ann::precond;
import tstate::ann::postcond;
import tstate::ann::empty_states;
import tstate::ann::pps_len;
import tstate::ann::set_prestate;
import tstate::ann::set_poststate;
import tstate::ann::set_in_poststate_;
import tstate::ann::extend_prestate;
import tstate::ann::extend_poststate;
import tstate::ann::set_precondition;
import tstate::ann::set_postcondition;
import tstate::ann::set_in_postcond_;
import tstate::ann::ts_ann;
import tstate::ann::clear_in_postcond;
import tstate::ann::clear_in_poststate;
import tstate::ann::clear_in_poststate_;
import tritv::*;
import bitvectors::promises_;

import syntax::print::pprust::constr_args_to_str;
import syntax::print::pprust::constr_arg_to_str;
import syntax::print::pprust::lit_to_str;

// Used to communicate which operands should be invalidated
// to helper functions
tag oper_type {
    oper_move;
    oper_swap;
    oper_assign;
    oper_assign_op;
    oper_pure;
}

/* logging funs */
fn def_id_to_str(def_id d) -> str {
    ret int::str(d._0) + "," + int::str(d._1);
}

fn comma_str(&(@constr_arg_use)[] args) -> str {
    auto rslt = "";
    auto comma = false;
    for (@constr_arg_use a in args) {
        if (comma) { rslt += ", "; } else { comma = true; }
        alt (a.node) {
            case (carg_base) { rslt += "*"; }
            case (carg_ident(?i)) { rslt += i._0; }
            case (carg_lit(?l)) { rslt += lit_to_str(l); }
        }
    }
    ret rslt;
}

fn constraint_to_str(&ty::ctxt tcx, &constr c) -> str {
    alt (c.node.c) {
        case (ninit(?i)) {
            ret "init(" + i + " [" + tcx.sess.span_str(c.span) + "])";
        }
        case (npred(?p, ?args)) {
            ret path_to_str(p) + "(" + comma_str(args) + ")" + "[" +
                    tcx.sess.span_str(c.span) + "]";
        }
    }
}

fn tritv_to_str(fn_ctxt fcx, &tritv::t v) -> str {
    auto s = "";
    auto comma = false;
    for (norm_constraint p in constraints(fcx)) {
        alt (tritv_get(v, p.bit_num)) {
            case (dont_care) { }
            case (?t) {
                s +=
                    if (comma) { ", " } else { comma = true; "" } +
                    if (t == tfalse) { "!" } else { "" } +                  
                    constraint_to_str(fcx.ccx.tcx, p.c);
            }
        }
    }
    ret s;
}

fn log_tritv(&fn_ctxt fcx, &tritv::t v) { log tritv_to_str(fcx, v); }

fn first_difference_string(&fn_ctxt fcx, &tritv::t expected, &tritv::t actual)
   -> str {
    let str s = "";
    for (norm_constraint c in constraints(fcx)) {
        if (tritv_get(expected, c.bit_num) == ttrue &&
            tritv_get(actual, c.bit_num) != ttrue) {
            ret constraint_to_str(fcx.ccx.tcx, c.c);
        }
    }
    ret s;
}

fn log_tritv_err(fn_ctxt fcx, tritv::t v) { log_err tritv_to_str(fcx, v); }

fn tos(&uint[] v) -> str {
    auto rslt = "";
    for (uint i in v) { if (i == 0u) { rslt += "0"; } 
        else if (i == 1u) { rslt += "1"; }
        else { rslt += "?"; } }
    ret rslt;
}

fn log_cond(&uint[] v) { log tos(v); }

fn log_cond_err(&uint[] v) { log_err tos(v); }

fn log_pp(&pre_and_post pp) {
    auto p1 = tritv::to_vec(pp.precondition);
    auto p2 = tritv::to_vec(pp.postcondition);
    log "pre:";
    log_cond(p1);
    log "post:";
    log_cond(p2);
}

fn log_pp_err(&pre_and_post pp) {
    auto p1 = tritv::to_vec(pp.precondition);
    auto p2 = tritv::to_vec(pp.postcondition);
    log_err "pre:";
    log_cond_err(p1);
    log_err "post:";
    log_cond_err(p2);
}

fn log_states(&pre_and_post_state pp) {
    auto p1 = tritv::to_vec(pp.prestate);
    auto p2 = tritv::to_vec(pp.poststate);
    log "prestate:";
    log_cond(p1);
    log "poststate:";
    log_cond(p2);
}

fn log_states_err(&pre_and_post_state pp) {
    auto p1 = tritv::to_vec(pp.prestate);
    auto p2 = tritv::to_vec(pp.poststate);
    log_err "prestate:";
    log_cond_err(p1);
    log_err "poststate:";
    log_cond_err(p2);
}

fn print_ident(&ident i) { log " " + i + " "; }

fn print_idents(&mutable ident[] idents) {
    if (ivec::len[ident](idents) == 0u) { ret; }
    log "an ident: " + ivec::pop[ident](idents);
    print_idents(idents);
}


/* data structures */

/**********************************************************************/

/* Two different data structures represent constraints in different
 contexts: constraint and norm_constraint.

constraint gets used to record constraints in a table keyed by def_ids.
cinit constraints represent a single constraint, for the initialization
state of a variable; a cpred constraint, with a single operator and a
list of possible argument lists, could represent several constraints at
once.

norm_constraint, in contrast, gets used when handling an instance
of a constraint rather than a definition of a constraint. It can
also be init or pred (ninit or npred), but the npred case just has
a single argument list. 

The representation of constraints, where multiple instances of the
same predicate are collapsed into one entry in the table, makes it
easier to look up a specific instance.

Both types are in constrast with the constraint type defined in
syntax::ast, which is for predicate constraints only, and is what
gets generated by the parser. aux and ast share the same type
to represent predicate *arguments* however. This type
(constr_arg_general) is parameterized (see comments in syntax::ast).

Both types store an ident and span, for error-logging purposes.
*/
type pred_desc_ = rec((@constr_arg_use)[] args, uint bit_num);

type pred_desc = spanned[pred_desc_];

// FIXME: Should be node_id, since we can only talk
// about locals. 
type constr_arg_use = constr_arg_general[tup(ident, def_id)];

tag constraint {
    cinit(uint, span, ident);
    cpred(path, @mutable pred_desc[]);
}

tag constr__ {
    ninit(ident);
    npred(path, (@constr_arg_use)[]);
}

type constr_ = rec(node_id id, constr__ c);

type constr = spanned[constr_];

type norm_constraint = rec(uint bit_num, constr c);

type constr_map = @std::map::hashmap[node_id, constraint];

type fn_info = rec(constr_map constrs,
                   uint num_constraints,
                   controlflow cf,
                   /* list, accumulated during pre/postcondition
                    computation, of all local variables that may be
                    used*/
                   // Doesn't seem to work without the @ --
                   // bug?
                   @mutable node_id[] used_vars);


/* mapping from node ID to typestate annotation */
type node_ann_table = @mutable ts_ann[mutable];


/* mapping from function name to fn_info map */
type fn_info_map = @std::map::hashmap[node_id, fn_info];

type fn_ctxt = rec(fn_info enclosing, node_id id, ident name, crate_ctxt ccx);

type crate_ctxt = rec(ty::ctxt tcx, node_ann_table node_anns, fn_info_map fm);

fn get_fn_info(&crate_ctxt ccx, node_id id) -> fn_info {
    assert (ccx.fm.contains_key(id));
    ret ccx.fm.get(id);
}

fn add_node(&crate_ctxt ccx, node_id i, &ts_ann a) {
    auto sz = ivec::len(*ccx.node_anns);
    if (sz <= i as uint) {
        ivec::grow_mut(*ccx.node_anns, (i as uint) - sz + 1u, empty_ann(0u));
    }
    ccx.node_anns.(i) = a;
}

fn get_ts_ann(&crate_ctxt ccx, node_id i) -> option::t[ts_ann] {
    if (i as uint < ivec::len(*ccx.node_anns)) {
        ret some[ts_ann](ccx.node_anns.(i));
    } else { ret none[ts_ann]; }
}


/********* utils ********/
fn node_id_to_ts_ann(&crate_ctxt ccx, node_id id) -> ts_ann {
    alt (get_ts_ann(ccx, id)) {
        case (none) {
            log_err "node_id_to_ts_ann: no ts_ann for node_id " +
                int::str(id);
            fail;
        }
        case (some(?t)) { ret t; }
    }
}

fn node_id_to_poststate(&crate_ctxt ccx, node_id id) -> poststate {
    log "node_id_to_poststate";
    ret node_id_to_ts_ann(ccx, id).states.poststate;
}

fn stmt_to_ann(&crate_ctxt ccx, &stmt s) -> ts_ann {
    log "stmt_to_ann";
    alt (s.node) {
        case (stmt_decl(_, ?id)) { ret node_id_to_ts_ann(ccx, id); }
        case (stmt_expr(_, ?id)) { ret node_id_to_ts_ann(ccx, id); }
        case (stmt_crate_directive(_)) {
            log_err "expecting an annotated statement here";
            fail;
        }
    }
}


/* fails if e has no annotation */
fn expr_states(&crate_ctxt ccx, @expr e) -> pre_and_post_state {
    log "expr_states";
    ret node_id_to_ts_ann(ccx, e.id).states;
}


/* fails if e has no annotation */
fn expr_pp(&crate_ctxt ccx, @expr e) -> pre_and_post {
    log "expr_pp";
    ret node_id_to_ts_ann(ccx, e.id).conditions;
}

fn stmt_pp(&crate_ctxt ccx, &stmt s) -> pre_and_post {
    ret stmt_to_ann(ccx, s).conditions;
}


/* fails if b has no annotation */
fn block_pp(&crate_ctxt ccx, &block b) -> pre_and_post {
    log "block_pp";
    ret node_id_to_ts_ann(ccx, b.node.id).conditions;
}

fn clear_pp(pre_and_post pp) {
    ann::clear(pp.precondition);
    ann::clear(pp.postcondition);
}

fn clear_precond(&crate_ctxt ccx, node_id id) {
    auto pp = node_id_to_ts_ann(ccx, id);
    ann::clear(pp.conditions.precondition);
}

fn block_states(&crate_ctxt ccx, &block b) -> pre_and_post_state {
    log "block_states";
    ret node_id_to_ts_ann(ccx, b.node.id).states;
}

fn stmt_states(&crate_ctxt ccx, &stmt s) -> pre_and_post_state {
    ret stmt_to_ann(ccx, s).states;
}

fn expr_precond(&crate_ctxt ccx, @expr e) -> precond {
    ret expr_pp(ccx, e).precondition;
}

fn expr_postcond(&crate_ctxt ccx, @expr e) -> postcond {
    ret expr_pp(ccx, e).postcondition;
}

fn expr_prestate(&crate_ctxt ccx, @expr e) -> prestate {
    ret expr_states(ccx, e).prestate;
}

fn expr_poststate(&crate_ctxt ccx, @expr e) -> poststate {
    ret expr_states(ccx, e).poststate;
}

fn stmt_precond(&crate_ctxt ccx, &stmt s) -> precond {
    ret stmt_pp(ccx, s).precondition;
}

fn stmt_postcond(&crate_ctxt ccx, &stmt s) -> postcond {
    ret stmt_pp(ccx, s).postcondition;
}

fn states_to_poststate(&pre_and_post_state ss) -> poststate {
    ret ss.poststate;
}

fn stmt_prestate(&crate_ctxt ccx, &stmt s) -> prestate {
    ret stmt_states(ccx, s).prestate;
}

fn stmt_poststate(&crate_ctxt ccx, &stmt s) -> poststate {
    ret stmt_states(ccx, s).poststate;
}

fn block_precond(&crate_ctxt ccx, &block b) -> precond {
    ret block_pp(ccx, b).precondition;
}

fn block_postcond(&crate_ctxt ccx, &block b) -> postcond {
    ret block_pp(ccx, b).postcondition;
}

fn block_prestate(&crate_ctxt ccx, &block b) -> prestate {
    ret block_states(ccx, b).prestate;
}

fn block_poststate(&crate_ctxt ccx, &block b) -> poststate {
    ret block_states(ccx, b).poststate;
}

fn set_prestate_ann(&crate_ctxt ccx, node_id id, &prestate pre) -> bool {
    log "set_prestate_ann";
    ret set_prestate(node_id_to_ts_ann(ccx, id), pre);
}

fn extend_prestate_ann(&crate_ctxt ccx, node_id id, &prestate pre) -> bool {
    log "extend_prestate_ann";
    ret extend_prestate(node_id_to_ts_ann(ccx, id).states.prestate, pre);
}

fn set_poststate_ann(&crate_ctxt ccx, node_id id, &poststate post) -> bool {
    log "set_poststate_ann";
    ret set_poststate(node_id_to_ts_ann(ccx, id), post);
}

fn extend_poststate_ann(&crate_ctxt ccx, node_id id, &poststate post)
    -> bool {
    log "extend_poststate_ann";
    ret extend_poststate(node_id_to_ts_ann(ccx, id).states.poststate, post);
}

fn set_pre_and_post(&crate_ctxt ccx, node_id id, &precond pre,
                    &postcond post) {
    log "set_pre_and_post";
    auto t = node_id_to_ts_ann(ccx, id);
    set_precondition(t, pre);
    set_postcondition(t, post);
}

fn copy_pre_post(&crate_ctxt ccx, node_id id, &@expr sub) {
    log "set_pre_and_post";
    auto p = expr_pp(ccx, sub);
    copy_pre_post_(ccx, id, p.precondition, p.postcondition);
}

fn copy_pre_post_(&crate_ctxt ccx, node_id id, &prestate pre,
                  &poststate post) {
    log "set_pre_and_post";
    auto t = node_id_to_ts_ann(ccx, id);
    set_precondition(t, pre);
    set_postcondition(t, post);
}

/* sets all bits to *1* */
fn set_postcond_false(&crate_ctxt ccx, node_id id) {
    auto p = node_id_to_ts_ann(ccx, id);
    ann::set(p.conditions.postcondition);
}

fn pure_exp(&crate_ctxt ccx, node_id id, &prestate p) -> bool {
    ret set_prestate_ann(ccx, id, p) |
        set_poststate_ann(ccx, id, p);
}

fn num_constraints(fn_info m) -> uint { ret m.num_constraints; }

fn new_crate_ctxt(ty::ctxt cx) -> crate_ctxt {
    let ts_ann[mutable] na = ~[mutable];
    ret rec(tcx=cx, node_anns=@mutable na, fm=@new_int_hash[fn_info]());
}

/* Use e's type to determine whether it returns.
 If it has a function type with a ! annotation,
the answer is noreturn. */
fn controlflow_expr(&crate_ctxt ccx, @expr e) -> controlflow {
    alt (ty::struct(ccx.tcx, ty::node_id_to_type(ccx.tcx, e.id))) {
        case (ty::ty_fn(_, _, _, ?cf, _)) { ret cf; }
        case (_) { ret return; }
    }
}

fn constraints_expr(&ty::ctxt cx, @expr e) -> (@ty::constr_def)[] {
    alt (ty::struct(cx, ty::node_id_to_type(cx, e.id))) {
        case (ty::ty_fn(_, _, _, _, ?cs)) { ret cs; }
        case (_) { ret ~[]; }
    }
}

fn node_id_to_def_strict(&ty::ctxt cx, node_id id) -> def {
    alt (cx.def_map.find(id)) {
        case (none) {
            log_err "node_id_to_def: node_id " + int::str(id) + " has no def";
            fail;
        }
        case (some(?d)) { ret d; }
    }
}

fn node_id_to_def(&crate_ctxt ccx, node_id id) -> option::t[def] {
    ret ccx.tcx.def_map.find(id);
}

fn norm_a_constraint(node_id id, &constraint c) -> norm_constraint[] {
    alt (c) {
        case (cinit(?n, ?sp, ?i)) {
            ret ~[rec(bit_num=n, c=respan(sp, rec(id=id, c=ninit(i))))];
        }
        case (cpred(?p, ?descs)) {
            let norm_constraint[] rslt = ~[];
            for (pred_desc pd in *descs) {
                rslt += ~[rec(bit_num=pd.node.bit_num,
                              c=respan(pd.span,
                                       rec(id=id,
                                           c=npred(p, pd.node.args))))];
            }
            ret rslt;
        }
    }
}


// Tried to write this as an iterator, but I got a
// non-exhaustive match in trans.
fn constraints(&fn_ctxt fcx) -> norm_constraint[] {
    let norm_constraint[] rslt = ~[];
    for each (@tup(node_id, constraint) p in fcx.enclosing.constrs.items()) {
        rslt += norm_a_constraint(p._0, p._1);
    }
    ret rslt;
}

fn match_args(&fn_ctxt fcx, &pred_desc[] occs, &(@constr_arg_use)[] occ) ->
   uint {
    log "match_args: looking at " +
        constr_args_to_str(std::util::fst[ident, def_id], occ);
    for (pred_desc pd in occs) {
        log "match_args: candidate " + pred_desc_to_str(pd);
        fn eq(&tup(ident, def_id) p, &tup(ident, def_id) q) -> bool {
            ret p._1 == q._1;
        }
        if (ty::args_eq(eq, pd.node.args, occ)) { ret pd.node.bit_num; }
    }
    fcx.ccx.tcx.sess.bug("match_args: no match for occurring args");
}

fn node_id_for_constr(ty::ctxt tcx, node_id t) -> node_id {
    alt (tcx.def_map.find(t)) {
        case (none) {
            tcx.sess.bug("node_id_for_constr: bad node_id " + int::str(t));
        }
        case (some(def_fn(?i,_))) { ret i._1; }
        case (_) {
            tcx.sess.bug("node_id_for_constr: pred is not a function");
        }
    }
}

fn expr_to_constr_arg(ty::ctxt tcx, &@expr e) -> @constr_arg_use {
    alt (e.node) {
        case (expr_path(?p)) {
            alt (tcx.def_map.find(e.id)) {
                case (some(def_local(?l_id))) {
                    ret @respan(p.span, carg_ident(tup(p.node.idents.(0),
                                                       l_id)));
                }
                case (some(def_arg(?a_id))) {
                    ret @respan(p.span, carg_ident(tup(p.node.idents.(0),
                                                       a_id)));
                }
                case (_) {
                    tcx.sess.bug("exprs_to_constr_args: non-local variable " +
                                 "as pred arg");
                        
                }
            }
        }
        case (expr_lit(?l)) { ret @respan(e.span, carg_lit(l)); }
        case (_) {
            tcx.sess.span_fatal(e.span,
                              "Arguments to constrained functions must be "
                              + "literals or local variables");
        }
    }
}

fn exprs_to_constr_args(ty::ctxt tcx, &(@expr)[] args)
    -> (@constr_arg_use)[] {
    auto f = bind expr_to_constr_arg(tcx, _);
    let (@constr_arg_use)[] rslt = ~[];
    for (@expr e in args) {
        rslt += ~[f(e)];
    }
    rslt
}

fn expr_to_constr(ty::ctxt tcx, &@expr e) -> constr {
    alt (e.node) {
        case (
             // FIXME change the first pattern to expr_path to test a
             // typechecker bug
             expr_call(?operator, ?args)) {
            alt (operator.node) {
                case (expr_path(?p)) {
                    // FIXME: Remove this vec->ivec conversion.
                    auto args_ivec = ~[];
                    for (@expr e in args) { args_ivec += ~[e]; }

                    ret respan(e.span,
                               rec(id=node_id_for_constr(tcx, operator.id),
                                   c=npred(p, exprs_to_constr_args(tcx,
                                        args_ivec))));
                }
                case (_) {
                    tcx.sess.span_fatal(operator.span,
                                      "Internal error: " +
                                          " ill-formed operator \
                                            in predicate");
                }
            }
        }
        case (_) {
            tcx.sess.span_fatal(e.span,
                              "Internal error: " + " ill-formed predicate");
        }
    }
}

fn pred_desc_to_str(&pred_desc p) -> str {
    "<" + uint::str(p.node.bit_num) + ", " +
        constr_args_to_str(std::util::fst[ident, def_id],
                           p.node.args) + ">"
}

fn substitute_constr_args(&ty::ctxt cx, &(@expr)[] actuals,
                          &@ty::constr_def c) -> constr__ {
    let (@constr_arg_use)[] rslt = ~[];
    for (@constr_arg a in c.node.args) {
        rslt += ~[substitute_arg(cx, actuals, a)];
    }
    ret npred(c.node.path, rslt);
}

fn substitute_arg(&ty::ctxt cx, &(@expr)[] actuals, @constr_arg a) ->
   @constr_arg_use {
    auto num_actuals = ivec::len(actuals);
    alt (a.node) {
        case (carg_ident(?i)) {
            if (i < num_actuals) {
                ret expr_to_constr_arg(cx, actuals.(i));
            } else {
                cx.sess.span_fatal(a.span,
                                   "Constraint argument out of bounds");
            }
        }
        case (carg_base) { ret @respan(a.span, carg_base); }
        case (carg_lit(?l)) { ret @respan(a.span, carg_lit(l)); }
    }
}

fn pred_desc_matches(&(constr_arg_general_[tup(ident, def_id)])[] pattern,
                     &pred_desc desc) -> bool {
    auto i = 0u;
    for (@constr_arg_use c in desc.node.args) {
        auto n = pattern.(i);
        alt (c.node) {
            case (carg_ident(?p)) {
                alt (n) {
                    case (carg_ident(?q)) {
                        if (p._1 != q._1) {
                            ret false;
                        }
                    }
                    case (_) { ret false; }
                }
            }
            case (carg_base) {
                if (n != carg_base) {
                    ret false;
                }
            }
            case (carg_lit(?l)) {
                alt (n) {
                    case (carg_lit(?m)) {
                        if (!lit_eq(l, m)) {
                            ret false;
                        }
                    }
                    case (_) { ret false; }
                }
            }
        }
        i += 1u;
    }
    ret true;
}

fn find_instance_(&(constr_arg_general_[tup(ident, def_id)])[] pattern,
                  &pred_desc[] descs) -> option::t[uint] {
    for (pred_desc d in descs) {
        if (pred_desc_matches(pattern, d)) {
            ret some(d.node.bit_num);
        }
    }
    ret none;
}

type inst = tup(ident, def_id);
type subst = tup(inst, inst)[];

fn find_instances(&fn_ctxt fcx, &subst subst, &constraint c)
    -> vec[tup(uint, uint)] {
   
    let vec[tup(uint, uint)] rslt = [];
    if (ivec::len(subst) == 0u) {
        ret rslt;
    }

    alt (c) {
        case (cinit(_,_,_)) { /* this is dealt with separately */ }
        case (cpred(?p, ?descs)) {
            for (pred_desc d in *descs) {
                if (args_mention(d.node.args, find_in_subst_bool, subst)) {
                    auto old_bit_num = d.node.bit_num;
                    auto new = replace(subst, d);
                    alt (find_instance_(new, *descs)) {
                        case (some(?d1)) {
                            rslt += [tup(old_bit_num, d1)];
                        }
                        case (_) { }
                    }
                }
            }
        }
    }
    rslt
}

fn find_in_subst(def_id id, &subst s) -> option::t[inst] {
    for (tup(inst, inst) p in s) {
        if (id == p._0._1) {
            ret some(p._1);
        }
    }
    ret none;
}

fn find_in_subst_bool(&subst s, def_id id) -> bool {
    is_some(find_in_subst(id, s))
}

fn insts_to_str(&(constr_arg_general_[inst])[] stuff) -> str {
    auto rslt = "<";
    for (constr_arg_general_[inst] i in stuff) {
        rslt += " " + alt(i) {
            case (carg_ident(?p)) { p._0 }
            case (carg_base) { "*" }
            case (carg_lit(_)) { "[lit]" } } + " ";
    }
    rslt += ">";
    rslt
}

fn replace(subst subst, pred_desc d) -> (constr_arg_general_[inst])[] {
    let (constr_arg_general_[inst])[] rslt = ~[];
    for (@constr_arg_use c in d.node.args) {
        alt (c.node) {
            case (carg_ident(?p)) {
                alt (find_in_subst(p._1, subst)) {
                    case (some(?new)) {
                        rslt += ~[carg_ident(new)];
                    }
                    case (_) {
                        rslt += ~[c.node];
                    }
                }
            }
            case (_) {
                //  log_err "##";
                rslt += ~[c.node];
            }
         }
    }
    
    /*
    for (constr_arg_general_[tup(ident, def_id)] p in rslt) {
        alt (p) {
            case (carg_ident(?p)) {
                log_err p._0;
            }
            case (_) {}
        }
    }
    */

    ret rslt;
}

fn path_to_ident(&ty::ctxt cx, &path p) -> ident {
    alt (ivec::last(p.node.idents)) {
        case (none) { cx.sess.span_fatal(p.span, "Malformed path"); }
        case (some(?i)) { ret i; }
    }
}

tag if_ty { 
    if_check;
    plain_if;
}

fn local_node_id_to_def_id_strict(&fn_ctxt fcx, &span sp, &node_id i) 
    -> def_id {
    alt (local_node_id_to_def(fcx, i)) {
        case (some(def_local(?d_id))) {
            ret d_id;
        }
        case (some (def_arg(?a_id))) {
            ret a_id;
        }
        case (some(_)) {
            fcx.ccx.tcx.sess.span_fatal(sp, "local_node_id_to_def_id: id \
               isn't a local");
        }
        case (none) {
            // should really be bug. span_bug()?
            fcx.ccx.tcx.sess.span_fatal(sp, "local_node_id_to_def_id: id \
               is unbound");
        }
    }
}

fn local_node_id_to_def(&fn_ctxt fcx, &node_id i) -> option::t[def]
  { fcx.ccx.tcx.def_map.find(i) }

fn local_node_id_to_def_id(&fn_ctxt fcx, &node_id i) -> option::t[def_id] {
    alt (local_node_id_to_def(fcx, i)) {
        case (some(def_local(?d_id))) { some(d_id) }
        case (some (def_arg(?a_id)))  { some(a_id) } 
        case (_)                      { none }
    }
}

fn copy_in_postcond(&fn_ctxt fcx, node_id parent_exp, inst dest, inst src,
                    oper_type ty) {
    auto post = node_id_to_ts_ann(fcx.ccx, parent_exp).conditions.
        postcondition;
    copy_in_poststate_two(fcx, post, post, dest, src, ty);
}

// FIXME refactor
fn copy_in_poststate(&fn_ctxt fcx, &poststate post, inst dest, inst src,
                     oper_type ty) {
    copy_in_poststate_two(fcx, post, post, dest, src, ty);
}

// In target_post, set the bits corresponding to copies of any
// constraints mentioning src that are set in src_post, with
// dest substituted for src.
// (This doesn't create any new constraints. If a new, substituted
// constraint isn't already in the bit vector, it's ignored.)
fn copy_in_poststate_two(&fn_ctxt fcx, &poststate src_post,
                         &poststate target_post, inst dest, inst src,
                         oper_type ty) {
    auto subst;
    alt (ty) {
        case (oper_swap) {
            subst = ~[tup(dest, src),
                     tup(src, dest)];
        }
        case (oper_assign_op) {
            ret; // Don't do any propagation
        }
        case (_) {
            subst = ~[tup(src, dest)];
        }
    }

    for each (@tup(node_id, constraint) p in
              fcx.enclosing.constrs.items()) {
        // replace any occurrences of the src def_id with the
        // dest def_id
        auto instances = find_instances(fcx, subst, p._1);

        for (tup(uint,uint) p in instances) { 
            if (promises_(p._0, src_post)) {
                set_in_poststate_(p._1, target_post);
            }
        }
    }
}


/* FIXME should refactor this better */
fn forget_in_postcond(&fn_ctxt fcx, node_id parent_exp, node_id dead_v) {
    // In the postcondition given by parent_exp, clear the bits
    // for any constraints mentioning dead_v
    auto d = local_node_id_to_def_id(fcx, dead_v);
    alt (d) {
        case (some(?d_id)) {
            for (norm_constraint c in constraints(fcx)) {
                if (constraint_mentions(fcx, c, d_id)) {
                    clear_in_postcond(c.bit_num,
                      node_id_to_ts_ann(fcx.ccx, parent_exp).conditions);
                }
            }
        }
        case (_) {}
    }
}

fn forget_in_postcond_still_init(&fn_ctxt fcx, node_id parent_exp,
                                 node_id dead_v) {
    // In the postcondition given by parent_exp, clear the bits
    // for any constraints mentioning dead_v
    auto d = local_node_id_to_def_id(fcx, dead_v);
    alt (d) {
        case (some(?d_id)) {
            for (norm_constraint c in constraints(fcx)) {
                if (non_init_constraint_mentions(fcx, c, d_id)) {
                    clear_in_postcond(c.bit_num,
                      node_id_to_ts_ann(fcx.ccx, parent_exp).conditions);
                }
            }
        }
        case (_) { }
    }
}

fn forget_in_poststate(&fn_ctxt fcx, &poststate p, node_id dead_v) -> bool {
    // In the poststate given by parent_exp, clear the bits
    // for any constraints mentioning dead_v
    auto d = local_node_id_to_def_id(fcx, dead_v);
    auto changed = false;
    alt (d) {
        case (some(?d_id)) {
            for (norm_constraint c in constraints(fcx)) {
                if (constraint_mentions(fcx, c, d_id)) {
                    changed |= clear_in_poststate_(c.bit_num, p);
                }
            }
        }
        case (_) {}
    }
    ret changed;
}

fn forget_in_poststate_still_init(&fn_ctxt fcx, &poststate p, node_id dead_v)
    -> bool {
    // In the poststate given by parent_exp, clear the bits
    // for any constraints mentioning dead_v
    auto d = local_node_id_to_def_id(fcx, dead_v);
    auto changed = false;
    alt (d) {
        case (some(?d_id)) {
            for (norm_constraint c in constraints(fcx)) {
                if (non_init_constraint_mentions(fcx, c, d_id)) {
                    changed |= clear_in_poststate_(c.bit_num, p);
                }
            }
        }
        case (_) {}
    }
    ret changed;
}

fn any_eq(&(def_id)[] v, def_id d) -> bool {
    for (def_id i in v) {
        if (i == d) { ret true; }
    }
    false
}

fn constraint_mentions(&fn_ctxt fcx, &norm_constraint c, def_id v) -> bool {
    ret (alt (c.c.node.c) {
            case (ninit(_)) {
                v == local_def(c.c.node.id)
            }
            case (npred(_, ?args)) {
                args_mention(args, any_eq, ~[v])
            }
        });
}

fn non_init_constraint_mentions(&fn_ctxt fcx, &norm_constraint c,
                                &def_id v) -> bool {
    ret (alt (c.c.node.c) {
            case (ninit(_)) {
                false
            }
            case (npred(_, ?args)) {
                args_mention(args, any_eq, ~[v])
            }
        });
}

fn args_mention[T](&(@constr_arg_use)[] args, fn(&(T)[], def_id) -> bool q,
                   &(T)[] s) -> bool {
    /*
      FIXME
      The following version causes an assertion in trans to fail
      (something about type_is_tup_like)
    fn mentions[T](&(T)[] s, &fn(&(T)[], def_id) -> bool q,
                            &@constr_arg_use a) -> bool {
        alt (a.node) {
            case (carg_ident(?p1)) {
                auto res = q(s, p1._1);
                log_err (res);
                res
                    }
            case (_)               { false }
        }
    }
    ret ivec::any(bind mentions(s,q,_), args);
    */

    for (@constr_arg_use a in args) {
        alt (a.node) {
            case (carg_ident(?p1)) {
                if (q(s, p1._1)) {
                    ret true;
                }
            }
            case (_)  {}
        }
    }
    ret false;
}

fn use_var(&fn_ctxt fcx, &node_id v) {
    *fcx.enclosing.used_vars += ~[v];
}

// FIXME: This should be a function in std::ivec::.
fn vec_contains(&@mutable (node_id[]) v, &node_id i) -> bool {
    for (node_id d in *v) {
        if (d == i) { ret true; }
    }
    ret false;
}

fn op_to_oper_ty(init_op io) -> oper_type {
    alt (io) {
        case (init_move) { oper_move }
        case (_)         { oper_assign }
    }
}

// default function visitor
fn do_nothing[T](&_fn f, &ty_param[] tp, &span sp, &fn_ident i,
              node_id iid, &T cx, &visit::vt[T] v) {
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
