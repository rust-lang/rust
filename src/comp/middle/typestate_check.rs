import front.ast;
import front.ast.ann;
import front.ast.method;
import front.ast.ty;
import front.ast.mutability;
import front.ast.item;
import front.ast.block;
import front.ast.block_;
import front.ast.block_index_entry;
import front.ast.mod_index_entry;
import front.ast.obj_field;
import front.ast.decl;
import front.ast.arm;
import front.ast.stmt;
import front.ast.stmt_;
import front.ast.stmt_decl;
import front.ast.stmt_expr;
import front.ast.stmt_crate_directive;
import front.ast.decl_local;
import front.ast.decl_item;
import front.ast.ident;
import front.ast.def_id;
import front.ast.ann;
import front.ast.field;
import front.ast.expr;
import front.ast.expr_call;
import front.ast.expr_vec;
import front.ast.expr_tup;
import front.ast.expr_path;
import front.ast.expr_field;
import front.ast.expr_index;
import front.ast.expr_log;
import front.ast.expr_block;
import front.ast.expr_rec;
import front.ast.expr_if;
import front.ast.expr_binary;
import front.ast.expr_unary;
import front.ast.expr_assign;
import front.ast.expr_assign_op;
import front.ast.expr_while;
import front.ast.expr_do_while;
import front.ast.expr_alt;
import front.ast.expr_lit;
import front.ast.expr_ret;
import front.ast.expr_self_method;
import front.ast.expr_bind;
import front.ast.expr_spawn;
import front.ast.expr_ext;
import front.ast.expr_fail;
import front.ast.expr_break;
import front.ast.expr_cont;
import front.ast.expr_send;
import front.ast.expr_recv;
import front.ast.expr_put;
import front.ast.expr_port;
import front.ast.expr_chan;
import front.ast.expr_be;
import front.ast.expr_check_expr;
import front.ast.expr_cast;
import front.ast.expr_for;
import front.ast.expr_for_each;
import front.ast.path;
import front.ast.elt;
import front.ast.crate_directive;
import front.ast.fn_decl;
import front.ast._obj;
import front.ast.native_mod;
import front.ast.variant;
import front.ast.ty_param;
import front.ast.ty;
import front.ast.proto;
import front.ast.pat;
import front.ast.binop;
import front.ast.unop;
import front.ast.def;
import front.ast.lit;
import front.ast.init_op;
import front.ast.initializer;
import front.ast.local;
import front.ast._fn;
import front.ast.ann_none;
import front.ast.ann_type;
import front.ast._obj;
import front.ast._mod;
import front.ast.crate;
import front.ast.mod_index_entry;
import front.ast.mie_item;
import front.ast.item_fn;
import front.ast.item_obj;
import front.ast.def_local;

import middle.fold;
import middle.fold.respan;
import driver.session;
import util.common;
import util.common.span;
import util.common.spanned;
import util.common.new_str_hash;
import util.common.new_def_hash;
import util.common.uistr;
import util.common.elt_exprs;
import util.common.field_exprs;
import util.common.log_expr;
import util.common.log_stmt;
import util.common.log_block;
import util.typestate_ann;
import util.typestate_ann.ts_ann;
import util.typestate_ann.empty_pre_post;
import util.typestate_ann.empty_poststate;
import util.typestate_ann.true_precond;
import util.typestate_ann.true_postcond;
import util.typestate_ann.postcond;
import util.typestate_ann.precond;
import util.typestate_ann.poststate;
import util.typestate_ann.prestate;
import util.typestate_ann.pre_and_post;
import util.typestate_ann.get_pre;
import util.typestate_ann.get_post;
import util.typestate_ann.ann_precond;
import util.typestate_ann.ann_prestate;
import util.typestate_ann.set_precondition;
import util.typestate_ann.set_postcondition;
import util.typestate_ann.set_prestate;
import util.typestate_ann.set_poststate;
import util.typestate_ann.set_in_postcond;
import util.typestate_ann.set_in_poststate;
import util.typestate_ann.implies;
import util.typestate_ann.pre_and_post_state;
import util.typestate_ann.empty_states;
import util.typestate_ann.empty_prestate;
import util.typestate_ann.empty_ann;
import util.typestate_ann.extend_prestate;
import util.typestate_ann.extend_poststate;
import util.typestate_ann.intersect;
import util.typestate_ann.pp_clone;

import middle.ty;
import middle.ty.ann_to_type;
import middle.ty.arg;
import middle.ty.block_ty;
import middle.ty.expr_ty;
import middle.ty.ty_to_str;

import pretty.pprust.print_block;
import pretty.pprust.print_expr;
import pretty.pprust.print_decl;
import pretty.pp.mkstate;
import std.io.stdout;
import std.io.str_writer;
import std.io.string_writer;
import std._vec.map;
import std._vec;
import std._vec.len;
import std._vec.pop;
import std._vec.push;
import std._vec.slice;
import std._vec.unzip;
import std._vec.plus_option;
import std.option;
import std.option.t;
import std.option.some;
import std.option.none;
import std.option.from_maybe;
import std.option.maybe;
import std.option.is_none;
import std.option.get;
import std.map.hashmap;
import std.list;
import std.list.list;
import std.list.cons;
import std.list.nil;
import std.list.foldl;
import std.list.find;
import std._uint;
import std.bitv;
import std.util.fst;
import std.util.snd;

import util.typestate_ann;
import util.typestate_ann.difference;
import util.typestate_ann.union;
import util.typestate_ann.pps_len;
import util.typestate_ann.require_and_preserve;

/**** debugging junk  ****/

fn log_bitv(fn_info enclosing, bitv.t v) {
  auto s = "";

  for each (@tup(def_id, tup(uint, ident)) p in enclosing.items()) {
    if (bitv.get(v, p._1._0)) {
      s += " " + p._1._1 + " ";
    }
  }

  log(s);
}

fn log_cond(vec[uint] v) -> () {
  auto res = "";
  for (uint i in v) {
    if (i == 0u) {
      res += "0";
    }
    else {
      res += "1";
    }
  }
  log(res);
}
fn log_pp(&pre_and_post pp) -> () {
  auto p1 = bitv.to_vec(pp.precondition);
  auto p2 = bitv.to_vec(pp.postcondition);
  log("pre:");
  log_cond(p1);
  log("post:");
  log_cond(p2);
}

fn print_ident(&ident i) -> () {
  log(" " + i + " ");
}

fn print_idents(vec[ident] idents) -> () {
  if(len[ident](idents) == 0u) {
    ret;
  }
  else {
    log("an ident: " + pop[ident](idents));
    print_idents(idents);
  }
}
/**********************************************************************/
/* mapping from variable name (def_id is assumed to be for a local
   variable in a given function) to bit number 
   (also remembers the ident for error-logging purposes) */
type var_info     = tup(uint, ident);
type fn_info      = std.map.hashmap[def_id, var_info];
/* mapping from function name to fn_info map */
type fn_info_map = std.map.hashmap[def_id, fn_info];
 
fn bit_num(def_id v, fn_info m) -> uint {
  check (m.contains_key(v));
  ret m.get(v)._0;
}
fn get_fn_info(fn_info_map fm, def_id did) -> fn_info {
    check (fm.contains_key(did));
    ret fm.get(did);
}

fn var_is_local(def_id v, fn_info m) -> bool {
  ret (m.contains_key(v));
}

fn num_locals(fn_info m) -> uint {
  ret m.size();
}

fn find_locals(_fn f) -> vec[tup(ident,def_id)] {
  auto res = _vec.alloc[tup(ident,def_id)](0u);

  for each (@tup(ident, block_index_entry) p
          in f.body.node.index.items()) {
    alt (p._1) {
      case (ast.bie_local(?loc)) {
        res += vec(tup(loc.ident,loc.id));
      }
      case (_) { }
    }
  }

  ret res;
}

fn add_var(def_id v, ident nm, uint next, fn_info tbl) -> uint {
    log(nm + " |-> " + util.common.uistr(next));
  tbl.insert(v, tup(next,nm));
  ret (next + 1u);
}

/* builds a table mapping each local var defined in f
 to a bit number in the precondition/postcondition vectors */
fn mk_fn_info(_fn f) -> fn_info {
  auto res = new_def_hash[var_info]();
  let uint next = 0u;
  let vec[ast.arg] f_args = f.decl.inputs;

  for (ast.arg v in f_args) {
    next = add_var(v.id, v.ident, next, res);
  }

  let vec[tup(ident,def_id)] locals = find_locals(f);
  for (tup(ident,def_id) p in locals) {
    next = add_var(p._1, p._0, next, res);
  }

  ret res;
}

/* extends mk_fn_info to a function item, side-effecting the map fi from 
   function IDs to fn_info maps */
fn mk_fn_info_item_fn(&fn_info_map fi, &span sp, ident i, &ast._fn f,
                 vec[ast.ty_param] ty_params, def_id id, ann a) -> @item {
  fi.insert(id, mk_fn_info(f));
  log(i + " has " + uistr(num_locals(mk_fn_info(f))) + " local vars");
  ret @respan(sp, item_fn(i, f, ty_params, id, a));
}

/* extends mk_fn_info to an obj item, side-effecting the map fi from 
   function IDs to fn_info maps */
fn mk_fn_info_item_obj(&fn_info_map fi, &span sp, ident i, &ast._obj o,
     vec[ast.ty_param] ty_params, ast.obj_def_ids odid, ann a) -> @item {
    auto all_methods = _vec.clone[@method](o.methods);
    plus_option[@method](all_methods, o.dtor);
    for (@method m in all_methods) {
        /* FIXME: also need to pass in fields so we can say
           they're initialized? */
        fi.insert(m.node.id, mk_fn_info(m.node.meth));
        log(m.node.ident + " has " +
            uistr(num_locals(mk_fn_info(m.node.meth))) + " local vars");
    }
    ret @respan(sp, item_obj(i, o, ty_params, odid, a));
}

/* initializes the global fn_info_map (mapping each function ID, including
   nested locally defined functions, onto a mapping from local variable name
   to bit number) */
fn mk_f_to_fn_info(@ast.crate c) -> fn_info_map {
  auto res = new_def_hash[fn_info]();

  auto fld = fold.new_identity_fold[fn_info_map]();
  fld = @rec(fold_item_fn  = bind mk_fn_info_item_fn(_,_,_,_,_,_,_),
             fold_item_obj = bind mk_fn_info_item_obj(_,_,_,_,_,_,_)
               with *fld);
  fold.fold_crate[fn_info_map](res, fld, c);

  ret res;
}
/**** Helpers ****/
fn expr_ann(&expr e) -> ann {
  alt(e.node) {
    case (ast.expr_vec(_,_,?a)) {
      ret a;
    }
    case (ast.expr_tup(_,?a)) {
      ret a;
    }
    case (ast.expr_rec(_,_,?a)) {
      ret a;
    }
    case (ast.expr_call(_,_,?a)) {
      ret a;
    }
    case (ast.expr_bind(_,_,?a)) {
      ret a;
    }
    case (ast.expr_binary(_,_,_,?a)) {
      ret a;
    }
    case (ast.expr_unary(_,_,?a)) {
      ret a;
    }
    case (ast.expr_lit(_,?a)) {
      ret a;
    }
    case (ast.expr_cast(_,_,?a)) {
      ret a;
    }
    case (ast.expr_if(_,_,_,?a)) {
      ret a;
    }
    case (ast.expr_while(_,_,?a)) {
      ret a;
    }
    case (ast.expr_for(_,_,_,?a)) {
      ret a;
    }
    case (ast.expr_for_each(_,_,_,?a)) {
      ret a;
    }
    case (ast.expr_do_while(_,_,?a)) {
      ret a;
    }
    case (ast.expr_alt(_,_,?a)) {
      ret a;
    }
    case (ast.expr_block(_,?a)) {
      ret a;
    }
    case (ast.expr_assign(_,_,?a)) {
      ret a;
    }
    case (ast.expr_assign_op(_,_,_,?a)) {
      ret a;
    }
    case (ast.expr_send(_,_,?a)) {
      ret a;
    }
    case (ast.expr_recv(_,_,?a)) {
      ret a;
    }
    case (ast.expr_field(_,_,?a)) {
      ret a;
    }
    case (ast.expr_index(_,_,?a)) {
      ret a;
    }
    case (ast.expr_path(_,_,?a)) {
      ret a;
    }
    case (ast.expr_ext(_,_,_,_,?a)) {
      ret a;
    }
    case (ast.expr_fail(?a)) {
      ret a;
    }
    case (ast.expr_ret(_,?a)) {
      ret a; 
    }
    case (ast.expr_put(_,?a)) {
      ret a;
    }
    case (ast.expr_be(_,?a)) {
      ret a;
    }
    case (ast.expr_log(_,?a)) {
      ret a;
    }
    case (ast.expr_check_expr(_,?a)) {
      ret a;
    }
    case (ast.expr_port(?a)) {
      ret a;
    }
    case (ast.expr_chan(_,?a)) {
      ret a;
    }
  }
}

fn ann_to_ts_ann(ann a, uint nv) -> ts_ann {
  alt (a) {
    case (ann_none)         { ret empty_ann(nv); }
    case (ann_type(_,_,?t)) {
      alt (t) {
        /* Kind of inconsistent. empty_ann()s everywhere
         or an option of a ts_ann? */
        case (none[@ts_ann])     { ret empty_ann(nv); }
        case (some[@ts_ann](?t)) { ret *t; }
      }
    }
  }
}

fn ann_to_ts_ann_fail(ann a) -> option.t[@ts_ann] {
  alt (a) {
      case (ann_none) { 
          log("ann_to_ts_ann_fail: didn't expect ann_none here");
          fail;
      }
      case (ann_type(_,_,?t)) {
          ret t;
      }
  }
}

fn ann_to_ts_ann_fail_more(ann a) -> @ts_ann {
  alt (a) {
      case (ann_none) { 
          log("ann_to_ts_ann_fail: didn't expect ann_none here");
          fail;
      }
      case (ann_type(_,_,?t)) {
          check (! is_none[@ts_ann](t));
          ret get[@ts_ann](t);
      }
  }
}

fn ann_to_poststate(ann a) -> poststate {
    ret (ann_to_ts_ann_fail_more(a)).states.poststate;
}

fn stmt_to_ann(&stmt s) -> option.t[@ts_ann] {
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

/* fails if e has no annotation */
fn expr_states(&expr e) -> pre_and_post_state {
  alt (expr_ann(e)) {
    case (ann_none) {
      log "expr_pp: the impossible happened (no annotation)";
      fail;
    }
    case (ann_type(_, _, ?maybe_pp)) {
      alt (maybe_pp) {
        case (none[@ts_ann]) {
          log "expr_pp: the impossible happened (no pre/post)";
          fail;
        }
        case (some[@ts_ann](?p)) {
          ret p.states;
        }
      }
    }
  }
}

/* fails if e has no annotation */
fn expr_pp(&expr e) -> pre_and_post {
  alt (expr_ann(e)) {
    case (ann_none) {
      log "expr_pp: the impossible happened (no annotation)";
      fail;
    }
    case (ann_type(_, _, ?maybe_pp)) {
      alt (maybe_pp) {
        case (none[@ts_ann]) {
          log "expr_pp: the impossible happened (no pre/post)";
          fail;
        }
        case (some[@ts_ann](?p)) {
          ret p.conditions;
        }
      }
    }
  }
}

fn stmt_pp(&stmt s) -> pre_and_post {
    alt (stmt_to_ann(s)) {
        case (none[@ts_ann]) {
            log "stmt_pp: the impossible happened (no annotation)";
            fail;
        }
        case (some[@ts_ann](?p)) {
            ret p.conditions;
        }
    }
}

/* fails if b has no annotation */
/* FIXME: factor out code in the following two functions (block_ts_ann) */
fn block_pp(&block b) -> pre_and_post {
    alt (b.node.a) {
       case (ann_none) {
           log "block_pp: the impossible happened (no ann)";
           fail;
       }
       case (ann_type(_,_,?t)) {
           alt (t) {
               case (none[@ts_ann]) {
                   log "block_pp: the impossible happened (no ty)";
                   fail;
               }
               case (some[@ts_ann](?ts)) {
                   ret ts.conditions;
               }
           }
       }
    }
}

fn block_states(&block b) -> pre_and_post_state {
    alt (b.node.a) {
       case (ann_none) {
           log "block_pp: the impossible happened (no ann)";
           fail;
       }
       case (ann_type(_,_,?t)) {
           alt (t) {
               case (none[@ts_ann]) {
                   log "block_states: the impossible happened (no ty)";
                   fail;
               }
               case (some[@ts_ann](?ts)) {
                   ret ts.states;
               }
           }
       }
    }
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


fn expr_precond(&expr e) -> precond {
  ret (expr_pp(e)).precondition;
}

fn expr_postcond(&expr e) -> postcond {
  ret (expr_pp(e)).postcondition;
}

fn expr_prestate(&expr e) -> prestate {
  ret (expr_states(e)).prestate;
}

fn expr_poststate(&expr e) -> poststate {
  ret (expr_states(e)).poststate;
}

/*
fn stmt_precond(&stmt s) -> precond {
  ret (stmt_pp(s)).precondition;
}

fn stmt_postcond(&stmt s) -> postcond {
  ret (stmt_pp(s)).postcondition;
}
*/

fn states_to_poststate(&pre_and_post_state ss) -> poststate {
  ret ss.poststate;
}

/*
fn stmt_prestate(&stmt s) -> prestate {
  ret (stmt_states(s)).prestate;
}
*/
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
  alt (a) {
    case (ann_none) {
      log("with_pp: the impossible happened");
      fail; /* shouldn't happen b/c code is typechecked */
    }
    case (ann_type(?t, ?ps, _)) {
      ret (ann_type(t, ps,
                    some[@ts_ann]
                    (@rec(conditions=p,
                          states=empty_states(pps_len(p))))));
    }
  }
}

// Given a list of pres and posts for exprs e0 ... en,
// return the precondition for evaluating each expr in order.
// So, if e0's post is {x} and e1's pre is {x, y, z}, the entire
// precondition shouldn't include x.
fn seq_preconds(uint num_vars, vec[pre_and_post] pps) -> precond {
  let uint sz = len[pre_and_post](pps);
 
  if (sz >= 1u) {
    auto first   = pps.(0);
    check (pps_len(first) == num_vars);
    let precond rest = seq_preconds(num_vars,
                         slice[pre_and_post](pps, 1u, sz));
    difference(rest, first.postcondition);
    union(first.precondition, rest);
    ret first.precondition;
  }
  else {
      ret true_precond(num_vars);
  }
}

/* works on either postconds or preconds
 should probably rethink the whole type synonym situation */
fn union_postconds_go(&postcond first, &vec[postcond] rest) -> postcond {
  auto sz = _vec.len[postcond](rest);

  if (sz > 0u) {
    auto other = rest.(0);
    union(first, other);
    union_postconds_go(first, slice[postcond](rest, 1u, len[postcond](rest)));
  }

  ret first;
}

fn union_postconds(&vec[postcond] pcs) -> postcond {
  check (len[postcond](pcs) > 0u);

  ret union_postconds_go(bitv.clone(pcs.(0)), pcs);
}

/* Gee, maybe we could use foldl or something */
fn intersect_postconds_go(&postcond first, &vec[postcond] rest) -> postcond {
  auto sz = _vec.len[postcond](rest);

  if (sz > 0u) {
    auto other = rest.(0);
    intersect(first, other);
    intersect_postconds_go(first, slice[postcond](rest, 1u,
                                                  len[postcond](rest)));
  }

  ret first;
}

fn intersect_postconds(&vec[postcond] pcs) -> postcond {
  check (len[postcond](pcs) > 0u);

  ret intersect_postconds_go(bitv.clone(pcs.(0)), pcs);
}

/******* AST-traversing code ********/

fn find_pre_post_mod(&_mod m) -> _mod {
    log("implement find_pre_post_mod!");
    fail;
}

fn find_pre_post_state_mod(&_mod m) -> bool {
    log("implement find_pre_post_state_mod!");
    fail;
}

fn find_pre_post_native_mod(&native_mod m) -> native_mod {
    log("implement find_pre_post_native_mod");
    fail;
}

fn find_pre_post_state_native_mod(&native_mod m) -> bool {
    log("implement find_pre_post_state_native_mod!");
    fail;
}
 
fn find_pre_post_obj(&fn_info_map fm, _obj o) -> () {
    fn do_a_method(fn_info_map fm, &@method m) -> () {
        check(fm.contains_key(m.node.id));
        find_pre_post_fn(fm, fm.get(m.node.id), m.node.meth);
    }
    auto f = bind do_a_method(fm,_);
    _vec.map[@method, ()](f, o.methods);
    option.map[@method, ()](f, o.dtor);
}

fn find_pre_post_state_obj(fn_info_map fm, _obj o) -> bool {
    fn do_a_method(fn_info_map fm, &@method m) -> bool {
        check(fm.contains_key(m.node.id));
        ret find_pre_post_state_fn(fm, fm.get(m.node.id), m.node.meth);
    }
    auto f = bind do_a_method(fm,_);
    auto flags = _vec.map[@method, bool](f, o.methods);
    auto changed = _vec.or(flags);
    changed = changed || maybe[@method, bool](false, f, o.dtor);
    ret changed;
}

fn find_pre_post_item(fn_info_map fm, fn_info enclosing, &item i) -> () {
  alt (i.node) {
    case (ast.item_const(?id, ?t, ?e, ?di, ?a)) {
        find_pre_post_expr(fm, enclosing, *e);
    }
    case (ast.item_fn(?id, ?f, ?ps, ?di, ?a)) {
      check (fm.contains_key(di));
      find_pre_post_fn(fm, fm.get(di), f);
    }
    case (ast.item_mod(?id, ?m, ?di)) {
      find_pre_post_mod(m);
    }
    case (ast.item_native_mod(?id, ?nm, ?di)) {
      find_pre_post_native_mod(nm);
    }
    case (ast.item_ty(_,_,_,_,_)) {
      ret;
    }
    case (ast.item_tag(_,_,_,_,_)) {
      ret;
    }
    case (ast.item_obj(?id, ?o, ?ps, ?di, ?a)) {
        find_pre_post_obj(fm, o);
    }
  }
}

/* Finds the pre and postcondition for each expr in <args>;
   sets the precondition in a to be the result of combining
   the preconditions for <args>, and the postcondition in a to 
   be the union of all postconditions for <args> */
fn find_pre_post_exprs(&fn_info_map fm, &fn_info enclosing,
                       &vec[@expr] args, ann a) {
    fn do_one(fn_info_map fm, fn_info enclosing,
              &@expr e) -> () {
        find_pre_post_expr(fm, enclosing, *e);
    }
    auto f = bind do_one(fm, enclosing, _);

    _vec.map[@expr, ()](f, args);

    fn get_pp(&@expr e) -> pre_and_post {
        ret expr_pp(*e);
    }
    auto g = get_pp;
    auto pps = _vec.map[@expr, pre_and_post](g, args);
    auto h = get_post;

    set_pre_and_post(a,
       rec(precondition=seq_preconds(num_locals(enclosing), pps),
           postcondition=union_postconds
           (_vec.map[pre_and_post, postcond](h, pps))));
}

/* Fills in annotations as a side effect. Does not rebuild the expr */
fn find_pre_post_expr(&fn_info_map fm, &fn_info enclosing, &expr e) -> () {
    auto num_local_vars = num_locals(enclosing);

    fn do_rand_(fn_info_map fm, fn_info enclosing, &@expr e) -> () {
        find_pre_post_expr(fm, enclosing, *e);
    }
    fn pp_one(&@expr e) -> pre_and_post {
        be expr_pp(*e);
    }

    /*    log("find_pre_post_expr (num_locals =" +
             uistr(num_local_vars) + "):");
          log_expr(e);
    */

    alt(e.node) {
        case(expr_call(?operator, ?operands, ?a)) {
            auto args = _vec.clone[@expr](operands);
            _vec.push[@expr](args, operator);
            find_pre_post_exprs(fm, enclosing, args, a);
        }
        case(expr_path(?p, ?maybe_def, ?a)) {
            auto df;
            alt (maybe_def) {
                case (none[def])
                    { log("expr_path should have a def"); fail; }
                case (some[def](?d)) { df = d; }
            }

            auto res = empty_pre_post(num_local_vars);

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
        case(expr_log(?arg, ?a)) {
            find_pre_post_expr(fm, enclosing, *arg);
            set_pre_and_post(a, expr_pp(*arg));
        }
        case (expr_block(?b, ?a)) {
            find_pre_post_block(fm, enclosing, b);
            set_pre_and_post(a, block_pp(b));
        }
        case (expr_rec(?fields,?maybe_base,?a)) {
            auto es = field_exprs(fields);
            _vec.plus_option[@expr](es, maybe_base);
            find_pre_post_exprs(fm, enclosing, es, a);
        }
        case (expr_assign(?lhs, ?rhs, ?a)) {
            alt (lhs.node) {
                case (expr_path(?p, some[def](def_local(?d_id)), ?a_lhs)) {
                    find_pre_post_expr(fm, enclosing, *rhs);
                    set_pre_and_post(a, expr_pp(*rhs));
                    gen(enclosing, a, d_id);
                }
                case (_) {
                    // doesn't check that lhs is an lval, but
                    // that's probably ok
                    find_pre_post_exprs(fm, enclosing, vec(lhs, rhs), a);
                }
            }
        }
        case (expr_lit(_,?a)) {
            set_pre_and_post(a, empty_pre_post(num_local_vars));
        }
        case (expr_ret(?maybe_val, ?a)) {
            alt (maybe_val) {
                case (none[@expr]) {
                    set_pre_and_post(a, empty_pre_post(num_local_vars));
                }
                case (some[@expr](?ret_val)) {
                    find_pre_post_expr(fm, enclosing, *ret_val);
                    let pre_and_post pp =
                        rec(precondition=expr_precond(*ret_val),
                            postcondition=empty_poststate(num_local_vars));
                    set_pre_and_post(a, pp);
                }
            }
        }
        case (expr_if(?antec, ?conseq, ?maybe_alt, ?a)) {
            find_pre_post_expr(fm, enclosing, *antec);
            find_pre_post_block(fm, enclosing, conseq);
            alt (maybe_alt) {
                case (none[@expr]) {
                    auto precond_res = seq_preconds(num_local_vars,
                                                    vec(expr_pp(*antec),
                                                        block_pp(conseq)));
                    set_pre_and_post(a, rec(precondition=precond_res,
                                            postcondition=
                                            expr_poststate(*antec)));
                }
                case (some[@expr](?altern)) {
                    find_pre_post_expr(fm, enclosing, *altern);
                    auto precond_true_case =
                        seq_preconds(num_local_vars,
                                     vec(expr_pp(*antec), block_pp(conseq)));
                    auto postcond_true_case = union_postconds
                        (vec(expr_postcond(*antec), block_postcond(conseq)));
                    auto precond_false_case = seq_preconds
                        (num_local_vars,
                         vec(expr_pp(*antec), expr_pp(*altern)));
                    auto postcond_false_case = union_postconds
                        (vec(expr_postcond(*antec), expr_postcond(*altern)));
                    auto precond_res = union_postconds(vec(precond_true_case,
                                                       precond_false_case));
                    auto postcond_res = intersect_postconds
                        (vec(postcond_true_case, postcond_false_case));
                    set_pre_and_post(a, rec(precondition=precond_res,
                                            postcondition=postcond_res));
                }
            }
        }
        case (expr_binary(?bop,?l,?r,?a)) {
            /* *unless* bop is lazy (e.g. and, or)? 
             FIXME */
            find_pre_post_exprs(fm, enclosing, vec(l, r), a);
        }
        case (expr_unary(_,?operand,?a)) {
            find_pre_post_expr(fm, enclosing, *operand);
            set_pre_and_post(a, expr_pp(*operand));
        }
        case (expr_while(?test, ?body, ?a)) {
            find_pre_post_expr(fm, enclosing, *test);
            find_pre_post_block(fm, enclosing, body);
            set_pre_and_post(a,
              rec(precondition=
                    seq_preconds(num_local_vars,
                                 vec(expr_pp(*test), 
                                     block_pp(body))),
                  postcondition=
                  intersect_postconds(vec(expr_postcond(*test),
                                          block_postcond(body)))));
        }
        case (expr_index(?e, ?sub, ?a)) {
            find_pre_post_exprs(fm, enclosing, vec(e, sub), a);
        }
        case (expr_alt(?e, ?alts, ?a)) {
            find_pre_post_expr(fm, enclosing, *e);
            fn do_an_alt(fn_info_map fm, fn_info enc, &arm an_alt)
                -> pre_and_post {
                find_pre_post_block(fm, enc, an_alt.block);
                ret block_pp(an_alt.block);
            }
            auto f = bind do_an_alt(fm, enclosing, _);
            auto alt_pps = _vec.map[arm, pre_and_post](f, alts);
            fn combine_pp(pre_and_post antec, 
                          uint num_local_vars, &pre_and_post pp,
                          &pre_and_post next) -> pre_and_post {
                union(pp.precondition, seq_preconds(num_local_vars,
                                         vec(antec, next)));
                intersect(pp.postcondition, next.postcondition);
                ret pp;
            }
            auto e_pp1 = expr_pp(*e);
            auto e_pp  = pp_clone(e_pp1);
            auto g = bind combine_pp(e_pp, num_local_vars, _, _);
            set_pre_and_post(a, _vec.foldl[pre_and_post, pre_and_post]
                             (g, e_pp, alt_pps));
        }
        case (expr_field(?operator, _, ?a)) {
            find_pre_post_expr(fm, enclosing, *operator);
            set_pre_and_post(a, expr_pp(*operator));
        }
        case (expr_fail(?a)) {
            set_pre_and_post(a,
               rec(precondition=empty_prestate(num_local_vars),
                   postcondition=true_postcond(num_local_vars)));
        }
        case(_) {
            log("this sort of expr isn't implemented!");
            log_expr(e);
            fail;
        }
    }
}

impure fn gen(&fn_info enclosing, &ann a, def_id id) -> bool {
  check(enclosing.contains_key(id));
  let uint i = (enclosing.get(id))._0;

  ret set_in_postcond(i, (ann_to_ts_ann_fail_more(a)).conditions);
}

impure fn gen_poststate(&fn_info enclosing, &ann a, def_id id) -> bool {
  check(enclosing.contains_key(id));
  let uint i = (enclosing.get(id))._0;

  ret set_in_poststate(i, (ann_to_ts_ann_fail_more(a)).states);
}

fn find_pre_post_stmt(fn_info_map fm, &fn_info enclosing, &ast.stmt s)
    -> () {
  auto num_local_vars = num_locals(enclosing);
  alt(s.node) {
    case(ast.stmt_decl(?adecl, ?a)) {
        alt(adecl.node) {
            case(ast.decl_local(?alocal)) {
                alt(alocal.init) {
                    case(some[ast.initializer](?an_init)) {
                        find_pre_post_expr(fm, enclosing, *an_init.expr);
                        auto rhs_pp = expr_pp(*an_init.expr);
                        set_pre_and_post(alocal.ann, rhs_pp);

                        /* Inherit ann from initializer, and add var being
                           initialized to the postcondition */
                        set_pre_and_post(a, rhs_pp);
                        gen(enclosing, a, alocal.id); 
                    }
                    case(none[ast.initializer]) {
                        auto pp = empty_pre_post(num_local_vars);
                        set_pre_and_post(alocal.ann, pp);
                        set_pre_and_post(a, pp);
                    }
                }
            }
            case(decl_item(?anitem)) {
                auto pp = empty_pre_post(num_local_vars);
                set_pre_and_post(a, pp);
                find_pre_post_item(fm, enclosing, *anitem);
            }
        }
    }
    case(stmt_expr(?e,?a)) {
        find_pre_post_expr(fm, enclosing, *e);
        set_pre_and_post(a, expr_pp(*e));
    }    
  }
}

fn find_pre_post_block(&fn_info_map fm, &fn_info enclosing, block b)
    -> () {
    fn do_one_(fn_info_map fm, fn_info i, &@stmt s) -> () {
        find_pre_post_stmt(fm, i, *s);
    }
    auto do_one = bind do_one_(fm, enclosing, _);
    
    _vec.map[@stmt, ()](do_one, b.node.stmts);
    fn do_inner_(fn_info_map fm, fn_info i, &@expr e) -> () {
        find_pre_post_expr(fm, i, *e);
    }
    auto do_inner = bind do_inner_(fm, enclosing, _);
    option.map[@expr, ()](do_inner, b.node.expr);

    let vec[pre_and_post] pps = vec();

    fn get_pp_stmt(&@stmt s) -> pre_and_post {
        ret stmt_pp(*s);
    }
    auto f = get_pp_stmt;
    pps += _vec.map[@stmt, pre_and_post](f, b.node.stmts);
    fn get_pp_expr(&@expr e) -> pre_and_post {
        ret expr_pp(*e);
    }
    auto g = get_pp_expr;
    plus_option[pre_and_post](pps,
       option.map[@expr, pre_and_post](g, b.node.expr));
    auto block_precond  = seq_preconds(num_locals(enclosing), pps);
    auto h = get_post;
    auto postconds =  _vec.map[pre_and_post, postcond](h, pps);
    _vec.push[postcond](postconds, block_precond); // ensures that length(postconds) > 0 
    auto block_postcond = union_postconds(postconds);
    set_pre_and_post(b.node.a, rec(precondition=block_precond,
                                   postcondition=block_postcond));
}

fn find_pre_post_fn(&fn_info_map fm, &fn_info fi, &_fn f) -> () {
    find_pre_post_block(fm, fi, f.body);
}

fn check_item_fn(&fn_info_map fm, &span sp, ident i, &ast._fn f,
                 vec[ast.ty_param] ty_params, def_id id, ann a) -> @item {

  check (fm.contains_key(id));
  find_pre_post_fn(fm, fm.get(id), f);  

  ret @respan(sp, ast.item_fn(i, f, ty_params, id, a));
}

fn find_pre_post_state_item(fn_info_map fm, fn_info enclosing, @item i)
   -> bool {
 alt (i.node) {
    case (ast.item_const(?id, ?t, ?e, ?di, ?a)) {
        ret find_pre_post_state_expr(fm, enclosing,
              empty_prestate(num_locals(enclosing)), e);
    }
    case (ast.item_fn(?id, ?f, ?ps, ?di, ?a)) {
      check (fm.contains_key(di));
      ret find_pre_post_state_fn(fm, fm.get(di), f);
    }
    case (ast.item_mod(?id, ?m, ?di)) {
      ret find_pre_post_state_mod(m);
    }
    case (ast.item_native_mod(?id, ?nm, ?di)) {
      ret find_pre_post_state_native_mod(nm);
    }
    case (ast.item_ty(_,_,_,_,_)) {
      ret false;
    }
    case (ast.item_tag(_,_,_,_,_)) {
      ret false;
    }
    case (ast.item_obj(?id, ?o, ?ps, ?di, ?a)) {
        ret find_pre_post_state_obj(fm, o);
    }
  }
}

impure fn set_prestate_ann(ann a, prestate pre) -> bool {
  alt (a) {
    case (ann_type(_,_,?ts_a)) {
      check (! is_none[@ts_ann](ts_a));
      ret set_prestate(*get[@ts_ann](ts_a), pre);
    }
    case (ann_none) {
      log("set_prestate_ann: expected an ann_type here");
      fail;
    }
  }
}


impure fn extend_prestate_ann(ann a, prestate pre) -> bool {
  alt (a) {
    case (ann_type(_,_,?ts_a)) {
      check (! is_none[@ts_ann](ts_a));
      ret extend_prestate((*get[@ts_ann](ts_a)).states.prestate, pre);
    }
    case (ann_none) {
      log("set_prestate_ann: expected an ann_type here");
      fail;
    }
  }
}

impure fn set_poststate_ann(ann a, poststate post) -> bool {
  alt (a) {
    case (ann_type(_,_,?ts_a)) {
      check (! is_none[@ts_ann](ts_a));
      ret set_poststate(*get[@ts_ann](ts_a), post);
    }
    case (ann_none) {
      log("set_poststate_ann: expected an ann_type here");
      fail;
    }
  }
}

impure fn extend_poststate_ann(ann a, poststate post) -> bool {
  alt (a) {
    case (ann_type(_,_,?ts_a)) {
      check (! is_none[@ts_ann](ts_a));
      ret extend_poststate((*get[@ts_ann](ts_a)).states.poststate, post);
    }
    case (ann_none) {
      log("set_poststate_ann: expected an ann_type here");
      fail;
    }
  }
}

impure fn set_pre_and_post(&ann a, pre_and_post pp) -> () {
    alt (a) {
        case (ann_type(_,_,?ts_a)) {
            check (! is_none[@ts_ann](ts_a));
            auto t = *get[@ts_ann](ts_a);
            /*  log("set_pre_and_post, old =");
            log_pp(t.conditions);
            log("new =");
            log_pp(pp);
            */
            set_precondition(t, pp.precondition);
            set_postcondition(t, pp.postcondition);
        }
        case (ann_none) {
            log("set_pre_and_post: expected an ann_type here");
            fail;
        }
    }
}

fn seq_states(&fn_info_map fm, &fn_info enclosing,
    prestate pres, vec[@expr] exprs) -> tup(bool, poststate) {
  auto changed = false;
  auto post = pres;

  for (@expr e in exprs) {
    changed = find_pre_post_state_expr(fm, enclosing, post, e) || changed;
    post = expr_poststate(*e);
  }

  ret tup(changed, post);
}

fn find_pre_post_state_exprs(&fn_info_map fm,
                             &fn_info enclosing,
                             &prestate pres,
                             &ann a, &vec[@expr] es) -> bool {
  auto res = seq_states(fm, enclosing, pres, es);
  auto changed = res._0;
  changed = extend_prestate_ann(a, pres) || changed;
  changed = extend_poststate_ann(a, res._1) || changed;
  ret changed;
}

impure fn pure_exp(&ann a, &prestate p) -> bool {
  auto changed = false;
  changed = extend_prestate_ann(a, p) || changed;
  changed = extend_poststate_ann(a, p) || changed;
  ret changed;
}

fn find_pre_post_state_expr(&fn_info_map fm, &fn_info enclosing,
                            &prestate pres, &@expr e) -> bool {
  auto changed = false;
  auto num_local_vars = num_locals(enclosing);

  alt (e.node) {
    case (expr_vec(?elts, _, ?a)) {
      be find_pre_post_state_exprs(fm, enclosing, pres, a, elts); 
    }
    case (expr_tup(?elts, ?a)) {
      be find_pre_post_state_exprs(fm, enclosing, pres, a, elt_exprs(elts));
    }
    case (expr_call(?operator, ?operands, ?a)) {
      /* do the prestate for the rator */
      changed = find_pre_post_state_expr(fm, enclosing, pres, operator)
        || changed;
      /* rands go left-to-right */
      ret(find_pre_post_state_exprs(fm, enclosing,
                                    expr_poststate(*operator), a, operands)
          || changed);
    }
    case (expr_path(_,_,?a)) {
      ret pure_exp(a, pres);
    }
    case (expr_log(?e,?a)) {
        changed = find_pre_post_state_expr(fm, enclosing, pres, e);
        changed = extend_prestate_ann(a, pres) || changed;
        changed = extend_poststate_ann(a, expr_poststate(*e)) || changed;
        ret changed;
    }
    case (expr_lit(?l,?a)) {
        ret pure_exp(a, pres);
    }
    case (expr_block(?b,?a)) {
        changed = find_pre_post_state_block(fm, enclosing, pres, b)
           || changed;
        changed = extend_prestate_ann(a, pres) || changed;
        changed = extend_poststate_ann(a, block_poststate(b)) || changed;
        ret changed;
    }
    case (expr_rec(?fields,?maybe_base,?a)) {
        changed = find_pre_post_state_exprs(fm, enclosing, pres, a,
                                            field_exprs(fields)) || changed;
        alt (maybe_base) {
            case (none[@expr]) { /* do nothing */ }
            case (some[@expr](?base)) {
                changed = find_pre_post_state_expr(fm, enclosing, pres, base)
                    || changed;
                changed = extend_poststate_ann(a, expr_poststate(*base))
                    || changed;
            }
        }
        ret changed;
    }
    case (expr_assign(?lhs, ?rhs, ?a)) {
        extend_prestate_ann(a, pres);

        alt (lhs.node) {
            case (expr_path(?p, some[def](def_local(?d_id)), ?a_lhs)) {
                // assignment to local var
                changed = pure_exp(a_lhs, pres) || changed;
                changed = find_pre_post_state_expr(fm, enclosing, pres, rhs)
                    || changed;
                changed = extend_poststate_ann(a, expr_poststate(*rhs))
                    || changed;
                changed = gen_poststate(enclosing, a, d_id) || changed;
            }
            case (_) {
                // assignment to something that must already have been init'd
                changed = find_pre_post_state_expr(fm, enclosing, pres, lhs)
                    || changed;
                changed = find_pre_post_state_expr(fm, enclosing,
                     expr_poststate(*lhs), rhs) || changed;
                changed = extend_poststate_ann(a, expr_poststate(*rhs))
                    || changed;
            }
        }
        ret changed;
    }
    case (expr_ret(?maybe_ret_val, ?a)) {
        changed = extend_prestate_ann(a, pres) || changed;
        set_poststate_ann(a, empty_poststate(num_local_vars));
        alt(maybe_ret_val) {
            case (none[@expr]) { /* do nothing */ }
            case (some[@expr](?ret_val)) {
                changed = find_pre_post_state_expr(fm, enclosing,
                             pres, ret_val) || changed;
            }
        }
        ret changed;
    }
    case (expr_if(?antec, ?conseq, ?maybe_alt, ?a)) {
        changed = extend_prestate_ann(a, pres) || changed;
        changed = find_pre_post_state_expr(fm, enclosing, pres, antec)
            || changed;
        changed = find_pre_post_state_block(fm, enclosing,
          expr_poststate(*antec), conseq) || changed;
        alt (maybe_alt) {
            case (none[@expr]) {
                changed = extend_poststate_ann(a, expr_poststate(*antec))
                    || changed;
            }
            case (some[@expr](?altern)) {
                changed = find_pre_post_state_expr(fm, enclosing,
                   expr_poststate(*antec), altern) || changed;
                auto poststate_res = intersect_postconds
                    (vec(block_poststate(conseq), expr_poststate(*altern)));
                changed = extend_poststate_ann(a, poststate_res) || changed;
            }
        }
        ret changed;
    }
    case (expr_binary(?bop, ?l, ?r, ?a)) {
        /* FIXME: what if bop is lazy? */
        changed = extend_prestate_ann(a, pres) || changed;
        changed = find_pre_post_state_expr(fm, enclosing, pres, l)
                    || changed;
        changed = find_pre_post_state_expr(fm,
                    enclosing, expr_poststate(*l), r) || changed;
        changed = extend_poststate_ann(a, expr_poststate(*r)) || changed;
        ret changed;
    }
    case (expr_while(?test, ?body, ?a)) {
        changed = extend_prestate_ann(a, pres) || changed;
        /* to handle general predicates, we need to pass in
            pres `intersect` (poststate(a)) 
         like: auto test_pres = intersect_postconds(pres, expr_postcond(a));
         However, this doesn't work right now because we would be passing
         in an all-zero prestate initially
           FIXME
           maybe need a "don't know" state in addition to 0 or 1?
        */
        changed = find_pre_post_state_expr(fm, enclosing, pres, test)
            || changed;
        changed = find_pre_post_state_block(fm, 
                   enclosing, expr_poststate(*test), body) || changed; 
        changed = extend_poststate_ann(a,
                    intersect_postconds(vec(expr_poststate(*test),
                                        block_poststate(body)))) || changed;
        ret changed;
    }
    case (expr_index(?e, ?sub, ?a)) {
        changed = extend_prestate_ann(a, pres) || changed; 
        changed = find_pre_post_state_expr(fm, enclosing, pres, e) || changed;
        changed = find_pre_post_state_expr(fm, enclosing,
                     expr_poststate(*e), sub) || changed;
        changed = extend_poststate_ann(a, expr_poststate(*sub));
        ret changed;
    }
    case (expr_alt(?e, ?alts, ?a)) {
        changed = extend_prestate_ann(a, pres) || changed; 
        changed = find_pre_post_state_expr(fm, enclosing, pres, e) || changed;
        auto e_post = expr_poststate(*e);
        auto a_post = ann_to_poststate(a);
        for (arm an_alt in alts) {
            changed = find_pre_post_state_block(fm, enclosing, e_post,
                        an_alt.block) || changed;
            changed = intersect(a_post, block_poststate(an_alt.block))
               || changed;
        }
        ret changed;
    }
    case (expr_field(?e,_,?a)) {
        changed = find_pre_post_state_expr(fm, enclosing, pres, e);
        changed = extend_prestate_ann(a, pres) || changed;
        changed = extend_poststate_ann(a, expr_poststate(*e)) || changed;
        ret changed;
    }
    case (expr_unary(_,?operand,?a)) {
        changed = find_pre_post_state_expr(fm, enclosing, pres, operand)
          || changed;
        changed = extend_prestate_ann(a, pres) || changed;
        changed = extend_poststate_ann(a, expr_poststate(*operand))
          || changed;
        ret changed;
    }
    case (expr_fail(?a)) {
        changed = extend_prestate_ann(a, pres) || changed;
        changed = set_poststate_ann(a, true_postcond(num_local_vars))
          || changed;
        ret changed;
    }
    case (_) {
      log("find_pre_post_state_expr: implement this case!");
      fail;
    }
  }

}

fn find_pre_post_state_stmt(&fn_info_map fm, &fn_info enclosing,
                            &prestate pres, @stmt s) -> bool {
  auto changed = false;
  auto stmt_ann_ = stmt_to_ann(*s);
  check (!is_none[@ts_ann](stmt_ann_));
  auto stmt_ann = *(get[@ts_ann](stmt_ann_));
  /*                  
              log("*At beginning: stmt = ");
              log_stmt(*s);
              log("*prestate = ");
              log(bitv.to_str(stmt_ann.states.prestate));
              log("*poststate =");
              log(bitv.to_str(stmt_ann.states.poststate));
              log("*changed =");
              log(changed);
  */       
  alt (s.node) {
    case (stmt_decl(?adecl, ?a)) {
      alt (adecl.node) {
        case (ast.decl_local(?alocal)) {
          alt (alocal.init) {
            case (some[ast.initializer](?an_init)) {
              changed = find_pre_post_state_expr
                (fm, enclosing, pres, an_init.expr) || changed;
              changed = extend_poststate(stmt_ann.states.poststate,
                                         expr_poststate(*an_init.expr))
                  || changed;
              changed = gen_poststate(enclosing, a, alocal.id) || changed;
    
              /*
              log("Summary: stmt = ");
              log_stmt(*s);
              log("prestate = ");
              log(bitv.to_str(stmt_ann.states.prestate));
              log_bitv(enclosing, stmt_ann.states.prestate);
              log("poststate =");
              log_bitv(enclosing, stmt_ann.states.poststate);
              log("changed =");
              log(changed);
              */

              ret changed;
            }
            case (none[ast.initializer]) {
              changed = extend_prestate(stmt_ann.states.prestate, pres)
                  || changed;
              changed = extend_poststate(stmt_ann.states.poststate, pres)
                  || changed;
              ret changed;
            }
          }
        }
        case (ast.decl_item(?an_item)) {
            be find_pre_post_state_item(fm, enclosing, an_item);
        }
      }
    }
    case (stmt_expr(?e, _)) {
      changed = find_pre_post_state_expr(fm, enclosing, pres, e) || changed;
      changed = extend_prestate(stmt_ann.states.prestate, expr_prestate(*e))
          || changed;
      changed = extend_poststate(stmt_ann.states.poststate,
                                 expr_poststate(*e)) || changed;
      /*
              log("Summary: stmt = ");
              log_stmt(*s);
              log("prestate = ");
              log(bitv.to_str(stmt_ann.states.prestate));
              log_bitv(enclosing, stmt_ann.states.prestate);
              log("poststate =");
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
fn find_pre_post_state_block(&fn_info_map fm, &fn_info enclosing,
                             &prestate pres0, block b)
  -> bool {

  auto changed = false;
  auto num_local_vars = num_locals(enclosing);

  /* First, set the pre-states and post-states for every expression */
  auto pres = pres0;
  
  /* Iterate over each stmt. The new prestate is <pres>. The poststate
   consist of improving <pres> with whatever variables this stmt initializes.
  Then <pres> becomes the new poststate. */ 
  for (@stmt s in b.node.stmts) {
    changed = find_pre_post_state_stmt(fm, enclosing, pres, s) || changed;
    pres = stmt_poststate(*s, num_local_vars);
  }

  auto post = pres;

  alt (b.node.expr) {
    case (none[@expr]) {}
    case (some[@expr](?e)) {
      changed = find_pre_post_state_expr(fm, enclosing, pres, e) || changed;
      post = expr_poststate(*e);
    }
  }
  set_prestate_ann(b.node.a, pres0);
  set_poststate_ann(b.node.a, post);
  ret changed;
}

fn find_pre_post_state_fn(&fn_info_map f_info, &fn_info fi, &ast._fn f)
  -> bool {
    /* FIXME: where do we set args as being initialized?
       What about for methods? */
    auto num_local_vars = num_locals(fi);
    ret find_pre_post_state_block(f_info, fi,
                                  empty_prestate(num_local_vars), f.body);
}

fn fixed_point_states(fn_info_map fm, fn_info f_info,
                      fn (&fn_info_map, &fn_info, &ast._fn) -> bool f,
                      &ast._fn start) -> () {

  auto changed = f(fm, f_info, start);

  if (changed) {
    ret fixed_point_states(fm, f_info, f, start);
  }
  else {
    // we're done!
    ret;
  }
}

impure fn check_states_expr(fn_info enclosing, &expr e) -> () {
  let precond prec    = expr_precond(e);
  let prestate pres   = expr_prestate(e);

  if (!implies(pres, prec)) {
      log("check_states_expr: unsatisfied precondition for ");
      log_expr(e);
      log("Precondition: ");
      log_bitv(enclosing, prec);
      log("Prestate: ");
      log_bitv(enclosing, pres);
      fail;
  }
}

fn check_states_stmt(fn_info enclosing, &stmt s) -> () {
  alt (stmt_to_ann(s)) {
    case (none[@ts_ann]) {
      ret;
    }
    case (some[@ts_ann](?a)) {
      let precond prec    = ann_precond(*a);
      let prestate pres   = ann_prestate(*a);

      /*
      log("check_states_stmt:");
      log_stmt(s);
      log("prec = ");
      log_bitv(enclosing, prec);
      log("pres = ");
      log_bitv(enclosing, pres);
      */

      if (!implies(pres, prec)) {
        log("check_states_stmt: unsatisfied precondition for ");
        log_stmt(s);
        log("Precondition: ");
        log_bitv(enclosing, prec);
        log("Prestate: ");
        log_bitv(enclosing, pres);
        fail;
      }
    }
  }
}

fn check_states_against_conditions(fn_info enclosing, &ast._fn f) -> () {
  fn do_one_(fn_info i, &@stmt s) -> () {
    check_states_stmt(i, *s);
  }
  auto do_one = bind do_one_(enclosing, _);
 
  _vec.map[@stmt, ()](do_one, f.body.node.stmts);
  fn do_inner_(fn_info i, &@expr e) -> () {
    check_states_expr(i, *e);
  }
  auto do_inner = bind do_inner_(enclosing, _);
  option.map[@expr, ()](do_inner, f.body.node.expr);
  
}

fn check_fn_states(&fn_info_map f_info_map, &fn_info f_info, &ast._fn f)
    -> () {
  /* Compute the pre- and post-states for this function */
  auto g = find_pre_post_state_fn;
  fixed_point_states(f_info_map, f_info, g, f);

  /* Now compare each expr's pre-state to its precondition
     and post-state to its postcondition */
  check_states_against_conditions(f_info, f);
}

fn check_item_fn_state(&fn_info_map f_info_map, &span sp, ident i,
                       &ast._fn f, vec[ast.ty_param] ty_params, def_id id,
                       ann a) -> @item {

  /* Look up the var-to-bit-num map for this function */
  check(f_info_map.contains_key(id));
  auto f_info = f_info_map.get(id);

  check_fn_states(f_info_map, f_info, f);

  /* Rebuild the same function */
  ret @respan(sp, ast.item_fn(i, f, ty_params, id, a));
}

fn check_method_states(&fn_info_map f_info_map, @method m) -> () {
    check (f_info_map.contains_key(m.node.id));
    auto f_info = f_info_map.get(m.node.id);
    check_fn_states(f_info_map, f_info, m.node.meth);
}

fn check_obj_state(&fn_info_map f_info_map, vec[obj_field] fields,
                  vec[@method] methods, option.t[@method] dtor) -> ast._obj {
    fn one(fn_info_map fm, &@method m) -> () {
        ret check_method_states(fm, m);
    }
    auto f = bind one(f_info_map,_);
    _vec.map[@method, ()](f, methods);
    option.map[@method, ()](f, dtor);
    ret rec(fields=fields, methods=methods, dtor=dtor);
}

fn init_ann(&fn_info fi, ann a) -> ann {
    alt (a) {
        case (ann_none) {
            log("init_ann: shouldn't see ann_none");
            fail;
        }
        case (ann_type(?t,?ps,_)) {
            ret ann_type(t, ps, some[@ts_ann](@empty_ann(num_locals(fi))));
        }
    }
}

fn init_blank_ann(&() ignore, ann a) -> ann {
    alt (a) {
        case (ann_none) {
            log("init_ann: shouldn't see ann_none");
            fail;
        }
        case (ann_type(?t,?ps,_)) {
            ret ann_type(t, ps, some[@ts_ann](@empty_ann(0u)));
        }
    }
}

fn init_block(&fn_info fi, &span sp, &block_ b) -> block {
    log("init_block:");
    log_block(respan(sp, b));
    alt(b.a) {
        case (ann_none) {
            log("init_ann: shouldn't see ann_none");
            fail;
        }
        case (ann_type(?t,?ps,_)) {
            auto fld0 = fold.new_identity_fold[fn_info]();

            fld0 = @rec(fold_ann = bind init_ann(_,_) with *fld0);
            ret fold.fold_block[fn_info](fi, fld0, respan(sp, b)); 
        }
    }
    
}

fn item_fn_anns(&fn_info_map fm, &span sp, ident i, &ast._fn f,
                vec[ast.ty_param] ty_params, def_id id, ann a) -> @item {

    check(fm.contains_key(id));
    auto f_info = fm.get(id);

    log(i + " has " + uistr(num_locals(f_info)) + " local vars");

    auto fld0 = fold.new_identity_fold[fn_info]();

    fld0 = @rec(fold_ann = bind init_ann(_,_) 
                //   fold_block = bind init_block(_,_,_)
                    with *fld0);

    ret fold.fold_item[fn_info]
           (f_info, fld0, @respan(sp, item_fn(i, f, ty_params, id, a))); 
}

/* This is painstakingly written as an explicit recursion b/c the
   standard ast.fold doesn't traverse in the correct order:
   consider
   fn foo() {
      fn bar() {
        auto x = 5;
        log(x);
      }
   }
   With fold, first bar() would be processed and its subexps would
   correctly be annotated with length-1 bit vectors.
   But then, the process would be repeated with (fn bar()...) as
   a subexp of foo, which has 0 local variables -- so then
   the body of bar() would be incorrectly annotated with length-0 bit
   vectors. */
fn annotate_exprs(&fn_info_map fm, &vec[@expr] es) -> vec[@expr] {
    fn one(fn_info_map fm, &@expr e) -> @expr {
        ret annotate_expr(fm, e);
    }
    auto f = bind one(fm,_);
    ret _vec.map[@expr, @expr](f, es);
}
fn annotate_elts(&fn_info_map fm, &vec[elt] es) -> vec[elt] {
    fn one(fn_info_map fm, &elt e) -> elt {
        ret rec(mut=e.mut,
                expr=annotate_expr(fm, e.expr));
    }
    auto f = bind one(fm,_);
    ret _vec.map[elt, elt](f, es);
}
fn annotate_fields(&fn_info_map fm, &vec[field] fs) -> vec[field] {
    fn one(fn_info_map fm, &field f) -> field {
        ret rec(mut=f.mut,
                 ident=f.ident,
                 expr=annotate_expr(fm, f.expr));
    }
    auto f = bind one(fm,_);
    ret _vec.map[field, field](f, fs);
}
fn annotate_option_exp(&fn_info_map fm, &option.t[@expr] o)
  -> option.t[@expr] {
    fn one(fn_info_map fm, &@expr e) -> @expr {
        ret annotate_expr(fm, e);
    }
    auto f = bind one(fm,_);
    ret option.map[@expr, @expr](f, o);
}
fn annotate_option_exprs(&fn_info_map fm, &vec[option.t[@expr]] es)
  -> vec[option.t[@expr]] {
    fn one(fn_info_map fm, &option.t[@expr] o) -> option.t[@expr] {
        ret annotate_option_exp(fm, o);
    }
    auto f = bind one(fm,_);
    ret _vec.map[option.t[@expr], option.t[@expr]](f, es);
}
fn annotate_decl(&fn_info_map fm, &@decl d) -> @decl {
    auto d1 = d.node;
    alt (d.node) {
        case (decl_local(?l)) {
            alt(l.init) {
                case (some[initializer](?init)) {
                    let option.t[initializer] an_i =
                        some[initializer]
                          (rec(expr=annotate_expr(fm, init.expr)
                                 with init));
                    let @local new_l = @rec(init=an_i with *l);
                    d1 = decl_local(new_l);
                }
                case (_) { /* do nothing */ }
            }
        }
        case (decl_item(?item)) {
            d1 = decl_item(annotate_item(fm, item));
        }
    }
    ret @respan(d.span, d1);
}
fn annotate_alts(&fn_info_map fm, &vec[arm] alts) -> vec[arm] {
    fn one(fn_info_map fm, &arm a) -> arm {
        ret rec(pat=a.pat,
                 block=annotate_block(fm, a.block),
                 index=a.index);
    }
    auto f = bind one(fm,_);
    ret _vec.map[arm, arm](f, alts);

}
fn annotate_expr(&fn_info_map fm, &@expr e) -> @expr {
    auto e1 = e.node;
    alt (e.node) {
        case (expr_vec(?es, ?m, ?a)) {
            e1 = expr_vec(annotate_exprs(fm, es), m, a);
        }
        case (expr_tup(?es, ?a)) {
            e1 = expr_tup(annotate_elts(fm, es), a);
        }
        case (expr_rec(?fs, ?maybe_e, ?a)) {
            e1 = expr_rec(annotate_fields(fm, fs),
                          annotate_option_exp(fm, maybe_e), a);
        }
        case (expr_call(?e, ?es, ?a)) {
            e1 = expr_call(annotate_expr(fm, e),
                          annotate_exprs(fm, es), a);
        }
        case (expr_self_method(_,_)) {
            // no change
        }
        case (expr_bind(?e, ?maybe_es, ?a)) {
            e1 = expr_bind(annotate_expr(fm, e),
                           annotate_option_exprs(fm, maybe_es),
                           a);
        }
        case (expr_spawn(?s, ?maybe_s, ?e, ?es, ?a)) {
            e1 = expr_spawn(s, maybe_s, annotate_expr(fm, e),
                            annotate_exprs(fm, es), a);
        }
        case (expr_binary(?bop, ?w, ?x, ?a)) {
            e1 = expr_binary(bop, annotate_expr(fm, w),
                             annotate_expr(fm, x), a);
        }
        case (expr_unary(?uop, ?w, ?a)) {
            e1 = expr_unary(uop, annotate_expr(fm, w), a);
        }
        case (expr_lit(_,_)) {
            /* no change */
        }
        case (expr_cast(?e,?t,?a)) {
            e1 = expr_cast(annotate_expr(fm, e), t, a);
        }
        case (expr_if(?e, ?b, ?maybe_e, ?a)) {
            e1 = expr_if(annotate_expr(fm, e),
                         annotate_block(fm, b),
                         annotate_option_exp(fm, maybe_e), a);
        }
        case (expr_while(?e, ?b, ?a)) {
            e1 = expr_while(annotate_expr(fm, e),
                            annotate_block(fm, b), a);
        }
        case (expr_for(?d, ?e, ?b, ?a)) {
            e1 = expr_for(annotate_decl(fm, d),
                          annotate_expr(fm, e),
                          annotate_block(fm, b), a);
        }
        case (expr_for_each(?d, ?e, ?b, ?a)) {
            e1 = expr_for_each(annotate_decl(fm, d),
                          annotate_expr(fm, e),
                          annotate_block(fm, b), a);
        }
        case (expr_do_while(?b, ?e, ?a)) {
            e1 = expr_do_while(annotate_block(fm, b),
                               annotate_expr(fm, e), a);
        }
        case (expr_alt(?e, ?alts, ?a)) {
            e1 = expr_alt(annotate_expr(fm, e),
                          annotate_alts(fm, alts), a);
        }
        case (expr_block(?b, ?a)) {
            e1 = expr_block(annotate_block(fm, b), a);
        }
        case (expr_assign(?l, ?r, ?a)) {
            e1 = expr_assign(annotate_expr(fm, l), annotate_expr(fm, r), a);
        }
        case (expr_assign_op(?bop, ?l, ?r, ?a)) {
            e1 = expr_assign_op(bop,
               annotate_expr(fm, l), annotate_expr(fm, r), a);
        }
        case (expr_send(?l, ?r, ?a)) {
            e1 = expr_send(annotate_expr(fm, l),
                           annotate_expr(fm, r), a);
        }
        case (expr_recv(?l, ?r, ?a)) {
           e1 = expr_recv(annotate_expr(fm, l),
                           annotate_expr(fm, r), a);
        }
        case (expr_field(?e, ?i, ?a)) {
            e1 = expr_field(annotate_expr(fm, e),
                            i, a);
        }
        case (expr_index(?e, ?sub, ?a)) {
            e1 = expr_index(annotate_expr(fm, e),
                            annotate_expr(fm, sub), a);
        }
        case (expr_path(_,_,_)) {
            /* no change */
        }
        case (expr_ext(?p, ?es, ?e_opt, ?e, ?a)) {
            e1 = expr_ext(p, annotate_exprs(fm, es),
                          annotate_option_exp(fm, e_opt),
                          annotate_expr(fm, e), a);
        }
        /* no change, next 3 cases */
        case (expr_fail(_)) { }
        case (expr_break(_)) { }
        case (expr_cont(_)) { }
        case (expr_ret(?maybe_e, ?a)) {
            e1 = expr_ret(annotate_option_exp(fm, maybe_e), a);
        }
        case (expr_put(?maybe_e, ?a)) {
            e1 = expr_put(annotate_option_exp(fm, maybe_e), a);
        }
        case (expr_be(?e, ?a)) {
            e1 = expr_be(annotate_expr(fm, e), a);
        }
        case (expr_log(?e, ?a)) {
            e1 = expr_log(annotate_expr(fm, e), a);
        }
        case (expr_check_expr(?e, ?a)) {
            e1 = expr_check_expr(annotate_expr(fm, e), a);
        }
        case (expr_port(_)) { /* no change */ }
        case (expr_chan(?e, ?a)) {
            e1 = expr_chan(annotate_expr(fm, e), a);
        }
    }
    ret @respan(e.span, e1);
}

fn annotate_stmt(&fn_info_map fm, &@stmt s) -> @stmt {
    alt (s.node) {
        case (stmt_decl(?d, ?a)) {
            ret @respan(s.span, stmt_decl(annotate_decl(fm, d), a));
        }
        case (stmt_expr(?e, ?a)) {
            ret @respan(s.span, stmt_expr(annotate_expr(fm, e), a));
        }
    }
}
fn annotate_block(&fn_info_map fm, &block b) -> block {
    let vec[@stmt] new_stmts = vec();
    auto new_index = new_str_hash[block_index_entry]();

    for (@stmt s in b.node.stmts) {
        auto new_s = annotate_stmt(fm, s);
        _vec.push[@stmt](new_stmts, new_s);
        ast.index_stmt(new_index, new_s);
    }
    fn ann_e(fn_info_map fm, &@expr e) -> @expr {
        ret annotate_expr(fm, e);
    }
    auto f = bind ann_e(fm,_);

    auto new_e = option.map[@expr, @expr](f, b.node.expr);

    ret respan(b.span,
          rec(stmts=new_stmts, expr=new_e, index=new_index with b.node));
}
fn annotate_fn(&fn_info_map fm, &ast._fn f) -> ast._fn {
    // subexps have *already* been annotated based on
    // f's number-of-locals
    ret rec(body=annotate_block(fm, f.body) with f);
}
fn annotate_mod(&fn_info_map fm, &ast._mod m) -> ast._mod {
    let vec[@item] new_items = vec();
    auto new_index = new_str_hash[mod_index_entry]();

    for (@item i in m.items) {
        auto new_i = annotate_item(fm, i);
        _vec.push[@item](new_items, new_i);
        ast.index_item(new_index, new_i);
    }
    ret rec(items=new_items, index=new_index with m);
}
fn annotate_native_mod(&fn_info_map fm, &ast.native_mod m)
       -> ast.native_mod {
    log("implement annotate_native_mod!");
    fail;
}
fn annotate_method(&fn_info_map fm, &@method m) -> @method {
    auto f_info = get_fn_info(fm, m.node.id);
    auto fld0 = fold.new_identity_fold[fn_info]();
    fld0 = @rec(fold_ann = bind init_ann(_,_) 
                with *fld0);
    auto outer = fold.fold_method[fn_info](f_info, fld0, m);
    auto new_fn = annotate_fn(fm, outer.node.meth);
    ret @respan(m.span,
                rec(meth=new_fn with m.node));
}

fn annotate_obj(&fn_info_map fm, &ast._obj o) -> ast._obj {
    fn one(fn_info_map fm, &@method m) -> @method {
        ret annotate_method(fm, m);
    }
    auto f = bind one(fm,_);
    auto new_methods = _vec.map[@method, @method](f, o.methods);
    auto new_dtor    = option.map[@method, @method](f, o.dtor);
    ret rec(methods=new_methods, dtor=new_dtor with o);
}

 
// Only annotates the components of the item recursively.
fn annotate_item_inner(&fn_info_map fm, &@ast.item item) -> @ast.item {
    alt (item.node) {
        /* FIXME can't skip this case -- exprs contain blocks contain stmts,
         which contain decls */
        case (ast.item_const(_,_,_,_,_)) {
            // this has already been annotated by annotate_item
            ret item;
        }
        case (ast.item_fn(?ident, ?ff, ?tps, ?id, ?ann)) {
            ret @respan(item.span,
                       ast.item_fn(ident, annotate_fn(fm, ff), tps, id, ann));
        }
        case (ast.item_mod(?ident, ?mm, ?id)) {
            ret @respan(item.span,
                       ast.item_mod(ident, annotate_mod(fm, mm), id));
        }
        case (ast.item_native_mod(?ident, ?mm, ?id)) {
            ret @respan(item.span,
                  ast.item_native_mod(ident, 
                    annotate_native_mod(fm, mm), id));
        }
        case (ast.item_ty(_,_,_,_,_)) {
            ret item;
        }
        case (ast.item_tag(_,_,_,_,_)) {
            ret item;
        }
        case (ast.item_obj(?ident, ?ob, ?tps, ?odid, ?ann)) {
            ret @respan(item.span,
              ast.item_obj(ident, annotate_obj(fm, ob), tps, odid, ann));
        }
    } 
}

fn annotate_item(&fn_info_map fm, &@ast.item item) -> @ast.item {
    // Using a fold, recursively set all anns in this item
    // to be blank.
    // *Then*, call annotate_item recursively to do the right
    // thing for any nested items inside this one.
    
    alt (item.node) {
        case (ast.item_const(_,_,_,_,_)) {
            auto fld0 = fold.new_identity_fold[()]();
            fld0 = @rec(fold_ann = bind init_blank_ann(_,_) 
                        with *fld0);
            ret fold.fold_item[()]((), fld0, item);
        }
        case (ast.item_fn(?i,?ff,?tps,?id,?ann)) {
            auto f_info = get_fn_info(fm, id);
            auto fld0 = fold.new_identity_fold[fn_info]();
            fld0 = @rec(fold_ann = bind init_ann(_,_) 
                        with *fld0);
            auto outer = fold.fold_item[fn_info](f_info, fld0, item);
            // now recurse into any nested items
            ret annotate_item_inner(fm, outer);
         }
        case (ast.item_mod(?i, ?mm, ?id)) {
            auto fld0 = fold.new_identity_fold[()]();
            fld0 = @rec(fold_ann = bind init_blank_ann(_,_) 
                        with *fld0);
            auto outer = fold.fold_item[()]((), fld0, item);
            ret annotate_item_inner(fm, outer);
        }
        case (ast.item_ty(_,_,_,_,_)) {
            ret item;
        }
        case (ast.item_tag(_,_,_,_,_)) {
            ret item;
        }
        case (ast.item_obj(?i,?ob,?tps,?odid,?ann)) {
            auto fld0 = fold.new_identity_fold[()]();
            fld0 = @rec(fold_ann = bind init_blank_ann(_,_) 
                        with *fld0);
            auto outer = fold.fold_item[()]((), fld0, item);
            ret annotate_item_inner(fm, outer);
        }
    }
}

fn annotate_module(&fn_info_map fm, &ast._mod module) -> ast._mod {
    let vec[@item] new_items = vec();
    auto new_index = new_str_hash[ast.mod_index_entry]();

    for (@item i in module.items) {
        auto new_item = annotate_item(fm, i);
        _vec.push[@item](new_items, new_item);
        ast.index_item(new_index, new_item);
    }

    ret rec(items = new_items, index = new_index with module);
}

fn annotate_crate(&fn_info_map fm, &@ast.crate crate) -> @ast.crate {
    ret @respan(crate.span,
               rec(module = annotate_module(fm, crate.node.module)
                   with crate.node));
}

fn check_crate(@ast.crate crate) -> @ast.crate {
  /* Build the global map from function id to var-to-bit-num-map */
  auto fm = mk_f_to_fn_info(crate);
  
  /* Add a blank ts_ann to every statement (and expression) */
  /*
  auto fld0 = fold.new_identity_fold[fn_info_map]();
  fld0 = @rec(fold_item_fn = bind item_fn_anns(_,_,_,_,_,_,_)
              with *fld0);
  */
  auto with_anns = annotate_crate(fm, crate);
  
  /* Compute the pre and postcondition for every subexpression */
  auto fld = fold.new_identity_fold[fn_info_map]();
  fld = @rec(fold_item_fn = bind check_item_fn(_,_,_,_,_,_,_) with *fld);
  auto with_pre_postconditions = fold.fold_crate[fn_info_map]
    (fm, fld, with_anns);

  auto fld1 = fold.new_identity_fold[fn_info_map]();

  fld1 = @rec(fold_item_fn = bind check_item_fn_state(_,_,_,_,_,_,_),
              fold_obj     = bind check_obj_state(_,_,_,_)
              with *fld1);

  ret fold.fold_crate[fn_info_map](fm, fld1,
                                    with_pre_postconditions);
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
