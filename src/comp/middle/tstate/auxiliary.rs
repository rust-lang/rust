import std::bitv;
import std::vec;
import std::vec::len;
import std::vec::grow;
import std::vec::pop;
import std::option;
import std::option::none;
import std::option::some;
import std::option::maybe;

import front::ast::*;

import middle::ty::expr_ann;

import util::common;
import util::common::span;
import util::common::respan;
import util::common::log_block;
import util::common::new_def_hash;
import util::common::new_uint_hash;
import util::common::log_expr_err;
import util::common::uistr;
import util::common::lit_eq;
import pretty::pprust::path_to_str;
import pretty::pprust::lit_to_str;

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
import tstate::ann::extend_prestate;
import tstate::ann::extend_poststate;
import tstate::ann::set_precondition;
import tstate::ann::set_postcondition;
import tstate::ann::ts_ann;

import util::common::istr;

/* logging funs */

fn def_id_to_str(def_id d) -> str {
   ret (istr(d._0) + "," + istr(d._1));
}

fn comma_str(vec[@constr_arg] args) -> str {
    auto res = "";
    auto comma = false;
    for (@constr_arg a in args) {
        if (comma) {
            res += ", ";
        }
        else {
            comma = true;
        }
        alt (a.node) {
            case (carg_base) {
                res += "*";
            }
            case (carg_ident(?i)) {
                res += i;
            }
            case (carg_lit(?l)) {
                res += lit_to_str(l);
            }
        }
    }
    ret res;
}

fn constraint_to_str(ty::ctxt tcx, constr c) -> str {
    alt (c.node) {
        case (ninit(?i)) {
            ret "init(" + i + " [" + tcx.sess.span_str(c.span) + "])";
        }
        case (npred(?p, ?args)) {
            ret path_to_str(p) + "(" + comma_str(args) + ")"
                + "[" + tcx.sess.span_str(c.span) + "]";
        }
    }
}

fn bitv_to_str(fn_ctxt fcx, bitv::t v) -> str {
  auto s = "";
  auto comma = false;

  for (norm_constraint p in constraints(fcx)) {
      if (bitv::get(v, p.bit_num)) {
          s += (if (comma) { ", " } else { comma = true; "" })
              + aux::constraint_to_str(fcx.ccx.tcx, p.c);
      }
  }
  ret s;
}

fn log_bitv(fn_ctxt fcx, bitv::t v) {
    log(bitv_to_str(fcx, v));
}

fn first_difference_string(&fn_ctxt fcx, &bitv::t expected,
                           &bitv::t actual) -> str {
    let str s = "";
    auto done = false;
    for (norm_constraint c in constraints(fcx)) {
        if (!done) {
            if (bitv::get(expected, c.bit_num) &&
                !bitv::get(actual, c.bit_num)) {
                
            /*
              FIXME
              for fun, try either:
              * "ret s" after the assignment to s
              or
              * using break here
              */
                s = constraint_to_str(fcx.ccx.tcx, c.c);
            
            done = true;
            }
        }
    }
    ret s;
}

fn log_bitv_err(fn_ctxt fcx, bitv::t v) {
    log_err(bitv_to_str(fcx, v));
}

fn tos (vec[uint] v) -> str {
  auto res = "";
  for (uint i in v) {
    if (i == 0u) {
      res += "0";
    }
    else {
      res += "1";
    }
  }
  ret res;
}

fn log_cond(vec[uint] v) -> () {
    log(tos(v));
}
fn log_cond_err(vec[uint] v) -> () {
    log_err(tos(v));
}

fn log_pp(&pre_and_post pp) -> () {
  auto p1 = bitv::to_vec(pp.precondition);
  auto p2 = bitv::to_vec(pp.postcondition);
  log("pre:");
  log_cond(p1);
  log("post:");
  log_cond(p2);
}

fn log_pp_err(&pre_and_post pp) -> () {
  auto p1 = bitv::to_vec(pp.precondition);
  auto p2 = bitv::to_vec(pp.postcondition);
  log_err("pre:");
  log_cond_err(p1);
  log_err("post:");
  log_cond_err(p2);
}

fn log_states(&pre_and_post_state pp) -> () {
  auto p1 = bitv::to_vec(pp.prestate);
  auto p2 = bitv::to_vec(pp.poststate);
  log("prestate:");
  log_cond(p1);
  log("poststate:");
  log_cond(p2);
}

fn log_states_err(&pre_and_post_state pp) -> () {
  auto p1 = bitv::to_vec(pp.prestate);
  auto p2 = bitv::to_vec(pp.poststate);
  log_err("prestate:");
  log_cond_err(p1);
  log_err("poststate:");
  log_cond_err(p2);
}

fn print_ident(&ident i) -> () {
  log(" " + i + " ");
}

fn print_idents(vec[ident] idents) -> () {
  if (len[ident](idents) == 0u) {
    ret;
  }
  else {
    log("an ident: " + pop[ident](idents));
    print_idents(idents);
  }
}


/* data structures */

/**********************************************************************/
/* mapping from def_id to bit number and other data
   (ident/path/span are there for error-logging purposes) */

type pred_desc_ = rec(vec[@constr_arg] args,
                      uint bit_num);
type pred_desc = spanned[pred_desc_];
tag constraint {
    cinit(uint, span, ident);
    cpred(path, vec[pred_desc]);
}
tag constr_ {
    ninit(ident);
    npred(path, vec[@constr_arg]);
}
type constr = spanned[constr_];
type norm_constraint = rec(uint bit_num,
                           constr c);
/* "constraint occurrence" to disambiguate
   between constraints. either "this is an
   init constraint", or the list of args for
   a pred. */
tag constr_occ {
    occ_init;
    occ_args(vec[@constr_arg]);
}
   
type constr_map = @std::map::hashmap[def_id, constraint];

type fn_info  = rec(constr_map constrs, uint num_constraints, controlflow cf);

/* mapping from node ID to typestate annotation */
type node_ann_table = @vec[ts_ann];

/* mapping from function name to fn_info map */
type fn_info_map = @std::map::hashmap[def_id, fn_info];

type fn_ctxt    = rec(fn_info enclosing,
                      def_id id,
                      ident name,
                      crate_ctxt ccx);

type crate_ctxt = rec(ty::ctxt tcx,
                      node_ann_table node_anns,
                      fn_info_map fm);

fn get_fn_info(&crate_ctxt ccx, def_id did) -> fn_info {
    assert (ccx.fm.contains_key(did));
    ret ccx.fm.get(did);
}

fn add_node(&crate_ctxt ccx, uint i, &ts_ann a) -> () {
    auto sz = len(*(ccx.node_anns)); 
    if (sz <= i) {
        grow(*(ccx.node_anns), (i - sz) + 1u, empty_ann(0u));
    }
    ccx.node_anns.(i) = a;
}

fn get_ts_ann(&crate_ctxt ccx, uint i) -> option::t[ts_ann] {
    if (i < len(*(ccx.node_anns))) {
        ret some[ts_ann](ccx.node_anns.(i));
    }
    else {
        ret none[ts_ann];
    }
}
/********* utils ********/

fn ann_to_ts_ann(&crate_ctxt ccx, &ann a) -> ts_ann {
    alt (get_ts_ann(ccx, a.id)) {
        case (none)         { 
            log_err ("ann_to_ts_ann: no ts_ann for node_id "
                     + uistr(a.id));
            fail;
        }
        case (some(?t))     { ret t; }
    }
}

fn ann_to_poststate(&crate_ctxt ccx, ann a) -> poststate {
    log "ann_to_poststate";
    ret (ann_to_ts_ann(ccx, a)).states.poststate;
}

fn stmt_to_ann(&crate_ctxt ccx, &stmt s) -> ts_ann {
    log "stmt_to_ann";
  alt (s.node) {
    case (stmt_decl(_,?a)) {
        ret ann_to_ts_ann(ccx, a);
    }
    case (stmt_expr(_,?a)) {
        ret ann_to_ts_ann(ccx, a);
    }
    case (stmt_crate_directive(_)) {
        log_err "expecting an annotated statement here";
        fail;
    }
  }
}

/* fails if e has no annotation */
fn expr_states(&crate_ctxt ccx, @expr e) -> pre_and_post_state {
    log "expr_states";
    ret (ann_to_ts_ann(ccx, expr_ann(e)).states);
}

/* fails if e has no annotation */
fn expr_pp(&crate_ctxt ccx, @expr e) -> pre_and_post {
    log "expr_pp";
    ret (ann_to_ts_ann(ccx, expr_ann(e)).conditions);
}

fn stmt_pp(&crate_ctxt ccx, &stmt s) -> pre_and_post {
    ret (stmt_to_ann(ccx, s).conditions);
}

/* fails if b has no annotation */
fn block_pp(&crate_ctxt ccx, &block b) -> pre_and_post {
    log "block_pp";
    ret (ann_to_ts_ann(ccx, b.node.a).conditions);
}

fn clear_pp(pre_and_post pp) {
    ann::clear(pp.precondition);
    ann::clear(pp.postcondition);
}

fn clear_precond(&crate_ctxt ccx, &ann a) {
    auto pp = ann_to_ts_ann(ccx, a);
    ann::clear(pp.conditions.precondition);
}

fn block_states(&crate_ctxt ccx, &block b) -> pre_and_post_state {
    log "block_states";
    ret (ann_to_ts_ann(ccx, b.node.a).states);
}

fn stmt_states(&crate_ctxt ccx, &stmt s) -> pre_and_post_state {
    ret (stmt_to_ann(ccx, s)).states;
}

fn expr_precond(&crate_ctxt ccx, @expr e) -> precond {
    ret (expr_pp(ccx, e)).precondition;
}

fn expr_postcond(&crate_ctxt ccx, @expr e) -> postcond {
    ret (expr_pp(ccx, e)).postcondition;
}

fn expr_prestate(&crate_ctxt ccx, @expr e) -> prestate {
    ret (expr_states(ccx, e)).prestate;
}

fn expr_poststate(&crate_ctxt ccx, @expr e) -> poststate {
    ret (expr_states(ccx, e)).poststate;
}

fn stmt_precond(&crate_ctxt ccx, &stmt s) -> precond {
    ret (stmt_pp(ccx, s)).precondition;
}

fn stmt_postcond(&crate_ctxt ccx, &stmt s) -> postcond {
    ret (stmt_pp(ccx, s)).postcondition;
}

fn states_to_poststate(&pre_and_post_state ss) -> poststate {
  ret ss.poststate;
}

fn stmt_prestate(&crate_ctxt ccx, &stmt s) -> prestate {
    ret (stmt_states(ccx, s)).prestate;
}

fn stmt_poststate(&crate_ctxt ccx, &stmt s) -> poststate {
    ret (stmt_states(ccx, s)).poststate;
}

fn block_postcond(&crate_ctxt ccx, &block b) -> postcond {
    ret (block_pp(ccx, b)).postcondition;
}

fn block_poststate(&crate_ctxt ccx, &block b) -> poststate {
    ret (block_states(ccx, b)).poststate;
}

/* sets the pre_and_post for an ann */
fn with_pp(&crate_ctxt ccx, &ann a, pre_and_post p) {
    add_node(ccx, a.id, @rec(conditions=p,
                             states=empty_states(pps_len(p))));
}

fn set_prestate_ann(&crate_ctxt ccx, &ann a, &prestate pre) -> bool {
    log "set_prestate_ann";
    ret set_prestate(ann_to_ts_ann(ccx, a), pre);
}


fn extend_prestate_ann(&crate_ctxt ccx, &ann a, &prestate pre) -> bool {
    log "extend_prestate_ann";
    ret extend_prestate(ann_to_ts_ann(ccx, a).states.prestate, pre);
}

fn set_poststate_ann(&crate_ctxt ccx, &ann a, &poststate post) -> bool {
    log "set_poststate_ann";
    ret set_poststate(ann_to_ts_ann(ccx, a), post);
}

fn extend_poststate_ann(&crate_ctxt ccx, &ann a, &poststate post) -> bool {
    log "extend_poststate_ann";
    ret extend_poststate(ann_to_ts_ann(ccx, a).states.poststate, post);
}

fn set_pre_and_post(&crate_ctxt ccx, &ann a,
                    &precond pre, &postcond post) -> () {
    log "set_pre_and_post";
    auto t = ann_to_ts_ann(ccx, a);
    set_precondition(t, pre);
    set_postcondition(t, post);
}

fn copy_pre_post(&crate_ctxt ccx, &ann a, &@expr sub) -> () {
    log "set_pre_and_post";
    auto p = expr_pp(ccx, sub);
    auto t = ann_to_ts_ann(ccx, a);
    set_precondition(t, p.precondition);
    set_postcondition(t, p.postcondition);
}


/* sets all bits to *1* */
fn set_postcond_false(&crate_ctxt ccx, &ann a) {
    auto p = ann_to_ts_ann(ccx, a);
    ann::set(p.conditions.postcondition);
}

fn pure_exp(&crate_ctxt ccx, &ann a, &prestate p) -> bool {
  auto changed = false;
  changed = extend_prestate_ann(ccx, a, p) || changed;
  changed = extend_poststate_ann(ccx, a, p) || changed;
  ret changed;
}

fn fixed_point_states(&fn_ctxt fcx,
    fn (&fn_ctxt, &_fn) -> bool f, &_fn start) -> () {

  auto changed = f(fcx, start);

  if (changed) {
    ret fixed_point_states(fcx, f, start);
  }
  else {
    // we're done!
    ret;
  }
}

fn num_constraints(fn_info m) -> uint {
    ret m.num_constraints;
}

fn new_crate_ctxt(ty::ctxt cx) -> crate_ctxt {
    let vec[ts_ann] na = [];
    ret rec(tcx=cx, node_anns=@na, fm=@new_def_hash[fn_info]());
}

fn controlflow_def_id(&crate_ctxt ccx, &def_id d) -> controlflow {
    alt (ccx.fm.find(d)) {
        case (some(?fi)) { ret fi.cf; }
        case (none)      { ret return; } 
    }
}

/* Use e's type to determine whether it returns.
 If it has a function type with a ! annotation,
the answer is noreturn. */
fn controlflow_expr(&crate_ctxt ccx, @expr e) -> controlflow {
    alt (ty::struct(ccx.tcx, ty::ann_to_type(ccx.tcx.node_types,
                                             expr_ann(e)))) {
        case (ty::ty_fn(_,_,_,?cf)) {
            ret cf;
        }
        case (_) {
            ret return;
        }
    }
}

fn ann_to_def_strict(&crate_ctxt ccx, &ann a) -> def {
    alt (ccx.tcx.def_map.find(a.id)) {
        case (none) { 
            log_err("ann_to_def: node_id " + uistr(a.id) + " has no def");
            fail;
        }
        case (some(?d)) { ret d; }
    }
}

fn ann_to_def(&crate_ctxt ccx, &ann a) -> option::t[def] {
    ret ccx.tcx.def_map.find(a.id);
}

fn norm_a_constraint(&constraint c) -> vec[norm_constraint] {
    alt (c) {
        case (cinit(?n, ?sp, ?i)) {
            ret [rec(bit_num=n, c=respan(sp, ninit(i)))];
        }
        case (cpred(?p, ?descs)) {
            let vec[norm_constraint] res = [];
            for (pred_desc pd in descs) {
                vec::push(res, rec(bit_num=pd.node.bit_num,
                  c=respan(pd.span, npred(p, pd.node.args))));
            }
            ret res;
        }
    }
}

// Tried to write this as an iterator, but I got a
// non-exhaustive match in trans.
fn constraints(&fn_ctxt fcx) -> vec[norm_constraint] {
    let vec[norm_constraint] res = [];
    for each (@tup(def_id, constraint) p in
              fcx.enclosing.constrs.items()) {
        res += norm_a_constraint(p._1);
    }
    ret res;
}

fn arg_eq(@constr_arg a, @constr_arg b) -> bool {
    alt (a.node) {
        case (carg_base) {
            alt (b.node) {
                case (carg_base) {
                    ret true;
                }
                case (_) {
                    ret false;
                }
            }
        }
        case (carg_ident(?s)) {
            alt (b.node) {
                case (carg_ident(?t)) {
                    ret (s == t);
                }
                case (_) {
                    ret false;
                }
            }
        }
        case (carg_lit(?l)) {
            alt (b.node) {
                case (carg_lit(?m)) {
                    ret lit_eq(l, m);
                }
                case (_) {
                    ret false;
                }
            }
        }
    }
}

fn args_eq(vec[@constr_arg] a, vec[@constr_arg] b) -> bool {
    let uint i = 0u;
    for (@constr_arg arg in a) {
        if (!arg_eq(arg, b.(i))) {
            ret false;
        }
        i += 1u;
    }
    ret true;
}

fn match_args(&fn_ctxt fcx, vec[pred_desc] occs,
              vec[@constr_arg] occ) -> uint {
    for (pred_desc pd in occs) {
        if (args_eq(pd.node.args, occ)) {
            ret pd.node.bit_num;
        }
    }
    fcx.ccx.tcx.sess.bug("match_args: no match for occurring args");  
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

