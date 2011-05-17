import std::bitv;
import std::vec::len;
import std::vec::pop;
import std::option;
import std::option::none;
import std::option::some;
import std::option::maybe;

import front::ast;
import front::ast::def;
import front::ast::def_fn;
import front::ast::_fn;
import front::ast::def_obj_field;
import front::ast::def_id;
import front::ast::expr_path;
import front::ast::ident;
import front::ast::controlflow;
import front::ast::ann;
import front::ast::ts_ann;
import front::ast::stmt;
import front::ast::expr;
import front::ast::block;
import front::ast::block_;
import front::ast::stmt_decl;
import front::ast::stmt_expr;
import front::ast::stmt_crate_directive;
import front::ast::return;
import front::ast::expr_field;

import middle::ty::expr_ann;
import middle::fold;
import middle::fold::respan;
import middle::fold::new_identity_fold;
import middle::fold::fold_block;

import util::common;
import util::common::span;
import util::common::log_block;
import util::common::new_def_hash;
import util::common::log_expr_err;
import util::common::uistr;

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

/* logging funs */

fn bitv_to_str(fn_info enclosing, bitv::t v) -> str {
  auto s = "";

  for each (@tup(def_id, tup(uint, ident)) p in enclosing.vars.items()) {
    if (bitv::get(v, p._1._0)) {
      s += " " + p._1._1 + " ";
    }
  }
  ret s;
}

fn log_bitv(fn_info enclosing, bitv::t v) {
    log(bitv_to_str(enclosing, v));
}

fn log_bitv_err(fn_info enclosing, bitv::t v) {
    log_err(bitv_to_str(enclosing, v));
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
/* mapping from variable name (def_id is assumed to be for a local
   variable in a given function) to bit number 
   (also remembers the ident for error-logging purposes) */
type var_info     = tup(uint, ident);
type fn_info      = rec(@std::map::hashmap[def_id, var_info] vars,
                        controlflow cf);
/* mapping from function name to fn_info map */
type fn_info_map = @std::map::hashmap[def_id, fn_info];

type fn_ctxt    = rec(fn_info enclosing,
                      def_id id,
                      ident name,
                      crate_ctxt ccx);

type crate_ctxt = rec(ty::ctxt tcx,
                      ty::node_type_table node_types,
                      fn_info_map fm);

fn get_fn_info(fn_info_map fm, def_id did) -> fn_info {
    assert (fm.contains_key(did));
    ret fm.get(did);
}

/********* utils ********/


fn ann_to_ts_ann(ann a, uint nv) -> @ts_ann {
    alt (ann_to_ts_ann_fail(a)) {
        case (none[@ts_ann])         { ret @empty_ann(nv); }
        case (some[@ts_ann](?t))     { ret t; }
    }
}


fn ann_to_ts_ann_fail(ann a) -> option::t[@ts_ann] { ret a.ts; }

fn ann_to_ts_ann_strict(ann a) -> @ts_ann {
    alt (ann_to_ts_ann_fail(a)) {
        case (none[@ts_ann]) {
            log("ann_to_ts_ann_strict: didn't expect none here");
            fail;
        }
        case (some[@ts_ann](?t)) { ret t; }
    }
}

fn ann_to_poststate(ann a) -> poststate {
    ret (ann_to_ts_ann_strict(a)).states.poststate;
}

fn stmt_to_ann(&stmt s) -> option::t[@ts_ann] {
  alt (s.node) {
    case (stmt_decl(_,?a)) {
        ret ann_to_ts_ann_fail(a);
    }
    case (stmt_expr(_,?a)) {
        ret ann_to_ts_ann_fail(a);
    }
    case (stmt_crate_directive(_)) {
      ret none[@ts_ann];
    }
  }
}

fn stmt_to_ann_strict(&stmt s) -> @ts_ann {
    alt (stmt_to_ann(s)) {
        case (none[@ts_ann]) {
            log_err("stmt_to_ann_strict: didn't expect none here");
            fail;
        }
        case (some[@ts_ann](?a)) { ret a; }
    }
}

/* fails if e has no annotation */
fn expr_states(@expr e) -> pre_and_post_state {
    ret (ann_to_ts_ann_strict(expr_ann(e)).states);
}

/* fails if e has no annotation */
fn expr_pp(@expr e) -> pre_and_post {
    ret (ann_to_ts_ann_strict(expr_ann(e)).conditions);
}

fn stmt_pp(&stmt s) -> pre_and_post {
    ret (stmt_to_ann_strict(s).conditions);
}

/* fails if b has no annotation */
fn block_pp(&block b) -> pre_and_post {
    ret (ann_to_ts_ann_strict(b.node.a).conditions);
}

fn block_states(&block b) -> pre_and_post_state {
    ret (ann_to_ts_ann_strict(b.node.a).states);
}

fn stmt_states(&stmt s, uint nv) -> pre_and_post_state {
  alt (stmt_to_ann(s)) {
    case (none[@ts_ann]) {
      ret empty_states(nv);
    }
    case (some[@ts_ann](?a)) {
      ret a.states;
    }
  }
}

fn expr_precond(@expr e) -> precond {
  ret (expr_pp(e)).precondition;
}

fn expr_postcond(@expr e) -> postcond {
  ret (expr_pp(e)).postcondition;
}

fn expr_prestate(@expr e) -> prestate {
  ret (expr_states(e)).prestate;
}

fn expr_poststate(@expr e) -> poststate {
  ret (expr_states(e)).poststate;
}

fn stmt_precond(&stmt s) -> precond {
  ret (stmt_pp(s)).precondition;
}

fn stmt_postcond(&stmt s) -> postcond {
  ret (stmt_pp(s)).postcondition;
}

fn states_to_poststate(&pre_and_post_state ss) -> poststate {
  ret ss.poststate;
}

fn stmt_prestate(&stmt s, uint nv) -> prestate {
  ret (stmt_states(s, nv)).prestate;
}

fn stmt_poststate(&stmt s, uint nv) -> poststate {
  ret (stmt_states(s, nv)).poststate;
}

fn block_postcond(&block b) -> postcond {
    ret (block_pp(b)).postcondition;
}

fn block_poststate(&block b) -> poststate {
    ret (block_states(b)).poststate;
}

/* returns a new annotation where the pre_and_post is p */
fn with_pp(ann a, pre_and_post p) -> ann {
    ret rec(id=a.id, ty=a.ty, tps=a.tps,
            ts=some[@ts_ann](@rec(conditions=p,
                                  states=empty_states(pps_len(p)))));
}

fn set_prestate_ann(&ann a, &prestate pre) -> bool {
    ret set_prestate(ann_to_ts_ann_strict(a), pre);
}


fn extend_prestate_ann(&ann a, &prestate pre) -> bool {
    ret extend_prestate(ann_to_ts_ann_strict(a).states.prestate, pre);
}

fn set_poststate_ann(&ann a, &poststate post) -> bool {
    ret set_poststate(ann_to_ts_ann_strict(a), post);
}

fn extend_poststate_ann(&ann a, &poststate post) -> bool {
    ret extend_poststate(ann_to_ts_ann_strict(a).states.poststate, post);
}

fn set_pre_and_post(&ann a, &pre_and_post pp) -> () {
    auto t = ann_to_ts_ann_strict(a);
    set_precondition(t, pp.precondition);
    set_postcondition(t, pp.postcondition);
}

fn pure_exp(&ann a, &prestate p) -> bool {
  auto changed = false;
  changed = extend_prestate_ann(a, p) || changed;
  changed = extend_poststate_ann(a, p) || changed;
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

fn init_ann(&fn_info fi, &ann a) -> ann {
    ret rec(id=a.id, ty=a.ty, tps=a.tps,
            ts=some[@ts_ann](@empty_ann(num_locals(fi))));
}

fn init_blank_ann(&() ignore, &ann a) -> ann {
    ret rec(id=a.id, ty=a.ty, tps=a.tps, ts=some[@ts_ann](@empty_ann(0u)));
}

fn init_block(&fn_info fi, &span sp, &block_ b) -> block {
    log("init_block:");
    log_block(respan(sp, b));

    auto fld0 = new_identity_fold[fn_info]();

    fld0 = @rec(fold_ann = bind init_ann(_,_) with *fld0);
    ret fold_block[fn_info](fi, fld0, respan(sp, b)); 
}

fn num_locals(fn_info m) -> uint {
  ret m.vars.size();
}

fn new_crate_ctxt(ty::node_type_table nt, ty::ctxt cx) -> crate_ctxt {
    ret rec(tcx=cx, node_types=nt, fm=@new_def_hash[fn_info]());
}

fn controlflow_def_id(&crate_ctxt ccx, &def_id d) -> controlflow {
    alt (ccx.fm.find(d)) {
        case (some[fn_info](?fi)) { ret fi.cf; }
        case (none[fn_info])      { ret return; } 
    }
}

/* conservative approximation: uses the mapping if e refers to a known
   function or method, assumes returning otherwise.
   There's no case for fail b/c we assume e is the callee and it
   seems unlikely that one would apply "fail" to arguments. */
fn controlflow_expr(&crate_ctxt ccx, @expr e) -> controlflow {
    auto f = expr_ann(e).id;
    alt (ccx.tcx.def_map.find(f)) {
        case (some[def](def_fn(?d)))        { ret controlflow_def_id(ccx, d); }
        case (some[def](def_obj_field(?d))) { ret controlflow_def_id(ccx, d); }
        case (_)                            { ret return; }
    }
}

fn ann_to_def_strict(&crate_ctxt ccx, &ann a) -> def {
    alt (ccx.tcx.def_map.find(a.id)) {
        case (none[def]) { 
            log_err("ann_to_def: node_id " + uistr(a.id) + " has no def");
            fail;
        }
        case (some[def](?d)) { ret d; }
    }
}

fn ann_to_def(&crate_ctxt ccx, &ann a) -> option::t[def] {
    ret ccx.tcx.def_map.find(a.id);
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

