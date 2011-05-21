import std::bitv;
import std::vec;
import std::vec::len;
import std::vec::slice;

import front::ast;
import front::ast::def_id;
import front::ast::expr;
import front::ast::ann;

import aux::fn_ctxt;
import aux::fn_info;
import aux::log_bitv;
import aux::num_locals;

import tstate::aux::ann_to_ts_ann;
import tstate::ann::pre_and_post;
import tstate::ann::precond;
import tstate::ann::postcond;
import tstate::ann::prestate;
import tstate::ann::poststate;
import tstate::ann::relax_prestate;
import tstate::ann::pps_len;
import tstate::ann::true_precond;
import tstate::ann::empty_prestate;
import tstate::ann::difference;
import tstate::ann::union;
import tstate::ann::intersect;
import tstate::ann::clone;
import tstate::ann::set_in_postcond;
import tstate::ann::set_in_poststate;
import tstate::ann::clear_in_poststate;
             
fn bit_num(def_id v, fn_info m) -> uint {
  assert (m.vars.contains_key(v));
  ret m.vars.get(v)._0;
}

fn promises(&poststate p, def_id v, fn_info m) -> bool {
    ret bitv::get(p, bit_num(v, m));
}

// Given a list of pres and posts for exprs e0 ... en,
// return the precondition for evaluating each expr in order.
// So, if e0's post is {x} and e1's pre is {x, y, z}, the entire
// precondition shouldn't include x.
fn seq_preconds(fn_info enclosing, vec[pre_and_post] pps) -> precond {
  let uint sz = len[pre_and_post](pps);
  let uint num_vars = num_locals(enclosing);

  if (sz >= 1u) {
    auto first = pps.(0);
    assert (pps_len(first) == num_vars);
    let precond rest = seq_preconds(enclosing,
                         slice[pre_and_post](pps, 1u, sz));
    difference(rest, first.postcondition);
    auto res = clone(first.precondition);
    union(res, rest);

    log("seq_preconds:");
    log("first.postcondition =");
    log_bitv(enclosing, first.postcondition);
    log("rest =");
    log_bitv(enclosing, rest);
    log("returning");
    log_bitv(enclosing, res);

    ret res;
  }
  else {
      ret true_precond(num_vars);
  }
}

/* works on either postconds or preconds
 should probably rethink the whole type synonym situation */
fn union_postconds_go(&postcond first, &vec[postcond] rest) -> postcond {
  auto sz = vec::len[postcond](rest);

  if (sz > 0u) {
    auto other = rest.(0);
    union(first, other);
    union_postconds_go(first, slice[postcond](rest, 1u, len[postcond](rest)));
  }

  ret first;
}

fn union_postconds(uint nv, &vec[postcond] pcs) -> postcond {
  if (len[postcond](pcs) > 0u) {
      ret union_postconds_go(bitv::clone(pcs.(0)), pcs);
  }
  else {
      ret empty_prestate(nv);
  }
}

/* Gee, maybe we could use foldl or something */
fn intersect_postconds_go(&postcond first, &vec[postcond] rest) -> postcond {
  auto sz = vec::len[postcond](rest);

  if (sz > 0u) {
    auto other = rest.(0);
    intersect(first, other);
    intersect_postconds_go(first, slice[postcond](rest, 1u,
                                                  len[postcond](rest)));
  }

  ret first;
}

fn intersect_postconds(&vec[postcond] pcs) -> postcond {
  assert (len[postcond](pcs) > 0u);

  ret intersect_postconds_go(bitv::clone(pcs.(0)), pcs);
}

fn gen(&fn_ctxt fcx, &ann a, def_id id) -> bool {
  log "gen";
  assert (fcx.enclosing.vars.contains_key(id));
  let uint i = (fcx.enclosing.vars.get(id))._0;
  ret set_in_postcond(i, (ann_to_ts_ann(fcx.ccx, a)).conditions);
}

fn declare_var(&fn_info enclosing, def_id id, prestate pre)
   -> prestate {
    assert (enclosing.vars.contains_key(id));
    let uint i = (enclosing.vars.get(id))._0;
    auto res = clone(pre);
    relax_prestate(i, res);
    ret res;
}

fn gen_poststate(&fn_ctxt fcx, &ann a, def_id id) -> bool {
  log "gen_poststate";
  assert (fcx.enclosing.vars.contains_key(id));
  let uint i = (fcx.enclosing.vars.get(id))._0;
  ret set_in_poststate(i, (ann_to_ts_ann(fcx.ccx, a)).states);
}

fn kill_poststate(&fn_ctxt fcx, &ann a, def_id id) -> bool {
  log "kill_poststate";
  assert (fcx.enclosing.vars.contains_key(id));
  let uint i = (fcx.enclosing.vars.get(id))._0;
  ret clear_in_poststate(i, (ann_to_ts_ann(fcx.ccx, a)).states);
}

