import front.ast;
import front.ast.ann;
import front.ast.ty;
import front.ast.mutability;
import front.ast.item;
import front.ast.block;
import front.ast.block_;
import front.ast.block_index_entry;
import front.ast.decl;
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
import front.ast.expr;
import front.ast.expr_call;
import front.ast.expr_path;
import front.ast.expr_log;
import front.ast.expr_block;
import front.ast.expr_lit;
import front.ast.path;
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
import front.ast.def_local;

import middle.fold;
import middle.fold.respan;
import driver.session;
import util.common;
import util.common.span;
import util.common.spanned;
import util.common.new_str_hash;
import util.common.new_def_hash;
import util.typestate_ann;
import util.typestate_ann.ts_ann;
import util.typestate_ann.empty_pre_post;
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
import util.typestate_ann.set_in_postcond;
import util.typestate_ann.implies;
import util.typestate_ann.pre_and_post_state;
import util.typestate_ann.empty_states;
import util.typestate_ann.empty_prestate;
import util.typestate_ann.empty_ann;
import util.typestate_ann.extend_prestate;

import middle.ty;
import middle.ty.ann_to_type;
import middle.ty.arg;
import middle.ty.block_ty;
import middle.ty.expr_ty;
import middle.ty.ty_to_str;

import pretty.pprust.print_block;
import pretty.pprust.print_expr;
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
import std.option;
import std.option.t;
import std.option.some;
import std.option.none;
import std.option.from_maybe;
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
fn log_expr(@expr e) -> () {
  let str_writer s = string_writer();
  auto out_ = mkstate(s.get_writer(), 80u);
  auto out = @rec(s=out_,
                  comments=option.none[vec[front.lexer.cmnt]],
                  mutable cur_cmnt=0u);

  print_expr(out, e);
  log(s.get_str());
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
   variable in a given function) to bit number */
type fn_info      = std.map.hashmap[def_id, uint];
/* mapping from function name to fn_info map */
type _fn_info_map = std.map.hashmap[def_id, fn_info];
 
fn bit_num(def_id v, fn_info m) -> uint {
  check (m.contains_key(v));
  ret m.get(v);
}

fn var_is_local(def_id v, fn_info m) -> bool {
  ret (m.contains_key(v));
}

fn num_locals(fn_info m) -> uint {
  ret m.size();
}

fn find_locals(_fn f) -> vec[def_id] {
  auto res = _vec.alloc[def_id](0u);

  for each (@tup(ident, block_index_entry) p
          in f.body.node.index.items()) {
    alt (p._1) {
      case (ast.bie_local(?loc)) {
        res += vec(loc.id);
      }
      case (_) { }
    }
  }

  ret res;
}

fn add_var(def_id v, uint next, fn_info tbl) -> uint {
  tbl.insert(v, next);
  // log(v + " |-> " + _uint.to_str(next, 10u));
  ret (next + 1u);
}

/* builds a table mapping each local var defined in f
 to a bit number in the precondition/postcondition vectors */
fn mk_fn_info(_fn f) -> fn_info {
  auto res = new_def_hash[uint]();
  let uint next = 0u;
  let vec[ast.arg] f_args = f.decl.inputs;

  for (ast.arg v in f_args) {
    next = add_var(v.id, next, res);
  }

  let vec[def_id] locals = find_locals(f);
  for (def_id v in locals) {
    next = add_var(v, next, res);
  }

  ret res;
}

/* extends mk_fn_info to an item, side-effecting the map fi from 
   function IDs to fn_info maps */
fn mk_fn_info_item_fn(&_fn_info_map fi, &span sp, ident i, &ast._fn f,
                 vec[ast.ty_param] ty_params, def_id id, ann a) -> @item {
  fi.insert(id, mk_fn_info(f));
  ret @respan(sp, item_fn(i, f, ty_params, id, a));
}

/* initializes the global fn_info_map (mapping each function ID, including
   nested locally defined functions, onto a mapping from local variable name
   to bit number) */
fn mk_f_to_fn_info(@ast.crate c) -> _fn_info_map {
  auto res = new_def_hash[fn_info]();

  auto fld = fold.new_identity_fold[_fn_info_map]();
  fld = @rec(fold_item_fn = bind mk_fn_info_item_fn(_,_,_,_,_,_,_) with *fld);
  fold.fold_crate[_fn_info_map](res, fld, c);

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

fn stmt_ann(&stmt s) -> option.t[@ts_ann] {
  alt (s.node) {
    case (stmt_decl(_,?a)) {
      ret a;
    }
    case (stmt_expr(_,?a)) {
      ret a;
    }
    case (stmt_crate_directive(_)) {
      ret none[@ts_ann];
    }
  }
}

/*
/* fails if no annotation */
fn stmt_pp(&stmt s) -> pre_and_post {
  ret (stmt_ann(s)).conditions;
}
*/

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
          // ret p.states;
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

fn stmt_states(&stmt s, uint nv) -> pre_and_post_state {
  alt (stmt_ann(s)) {
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
  
  if (sz == 0u) {
    ret true_precond(num_vars);
  }
  else {
    auto first   = pps.(0);
    check (pps_len(first) == num_vars);
    let precond rest = seq_preconds(num_vars,
                         slice[pre_and_post](pps, 1u, sz));
    difference(rest, first.postcondition);
    union(first.precondition, rest);
    ret (first.precondition);
  }
}

fn union_postconds_go(postcond first, &vec[postcond] rest) -> postcond {
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

  be union_postconds_go(pcs.(0), pcs);
}

/******* AST-traversing code ********/

fn find_pre_post_mod(&_mod m) -> _mod {
  ret m; /* FIXME */
}

fn find_pre_post_native_mod(&native_mod m) -> native_mod {
  ret m; /* FIXME */
}
 
fn find_pre_post_obj(_obj o) -> _obj {
  ret o; /* FIXME */
}

fn find_pre_post_item(_fn_info_map fm, fn_info enclosing, &item i) -> item {
  alt (i.node) {
    case (ast.item_const(?id, ?t, ?e, ?di, ?a)) {
      auto e_pp = find_pre_post_expr(enclosing, *e);
      log("1");
      ret (respan(i.span,
                  ast.item_const(id, t, e_pp, di, a)));
    }
    case (ast.item_fn(?id, ?f, ?ps, ?di, ?a)) {
      check (fm.contains_key(di));
      auto f_pp = find_pre_post_fn(fm, fm.get(di), f);
      ret (respan(i.span, 
                   ast.item_fn(id, f_pp, ps, di, a)));
    }
    case (ast.item_mod(?id, ?m, ?di)) {
      auto m_pp = find_pre_post_mod(m);
      log("3");
      ret (respan(i.span,
                   ast.item_mod(id, m_pp, di)));
    }
    case (ast.item_native_mod(?id, ?nm, ?di)) {
      auto n_pp = find_pre_post_native_mod(nm);
      log("4");
      ret (respan(i.span,
                   ast.item_native_mod(id, n_pp, di)));
    }
    case (ast.item_ty(_,_,_,_,_)) {
      ret i;
    }
    case (ast.item_tag(_,_,_,_,_)) {
      ret i;
    }
    case (ast.item_obj(?id, ?o, ?ps, ?di, ?a)) {
      auto o_pp = find_pre_post_obj(o);
      log("5");
      ret (respan(i.span,
                   ast.item_obj(id, o_pp, ps, di, a)));
    }
  }
}

fn find_pre_post_expr(&fn_info enclosing, &expr e) -> @expr {
  auto num_local_vars = num_locals(enclosing);

  fn do_rand_(fn_info enclosing, &@expr e) -> @expr {
    log("for rand " );
    log_expr(e);
    log("pp = ");
    auto res = find_pre_post_expr(enclosing, *e);
    log_pp(expr_pp(*res));
    ret res;
  }

  auto do_rand = bind do_rand_(enclosing,_);

  alt(e.node) {
    case(expr_call(?oper, ?rands, ?a)) {
      auto pp_oper = find_pre_post_expr(enclosing, *oper);
      log("pp_oper =");
      log_pp(expr_pp(*pp_oper));
      
      auto f = do_rand;
      auto pp_rands = _vec.map[@expr, @expr](f, rands);
      
      fn pp_one(&@expr e) -> pre_and_post {
        be expr_pp(*e);
      }
      auto g = pp_one;
      auto pps = _vec.map[@expr, pre_and_post](g, pp_rands);
      _vec.push[pre_and_post](pps, expr_pp(*pp_oper));
      auto h = get_post;
      auto res_postconds = _vec.map[pre_and_post, postcond](h, pps);
      auto res_postcond = union_postconds(res_postconds);
      let pre_and_post pp =
        rec(precondition=seq_preconds(num_local_vars, pps),
             postcondition=res_postcond);
      let ann a_res = with_pp(a, pp);
      log("result for call");
      log_expr(@e);
      log("is:");
      log_pp(pp);
      ret (@respan(e.span,
                   expr_call(pp_oper, pp_rands, a_res)));
                        
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
      log("pre/post for:\n");
      log_expr(@e);
      log("is");
      log_pp(res);
      ret (@respan
           (e.span,
            expr_path(p, maybe_def,
                      with_pp(a, res))));
    }
    case(expr_log(?arg, ?a)) {
      log("log");
      auto e_pp = find_pre_post_expr(enclosing, *arg);
      log("pre/post for: ");
      log_expr(arg);
      log("is");
      log_pp(expr_pp(*e_pp));
      ret (@respan(e.span,
                   expr_log(e_pp, with_pp(a, expr_pp(*e_pp)))));
    }
    case (expr_block(?b, ?a)) {
      log("block!");
      fail;
    }
    case (expr_lit(?l, ?a)) {
      ret @respan(e.span,
                  expr_lit(l, with_pp(a, empty_pre_post(num_local_vars))));
    }
    case(_) {
      log("this sort of expr isn't implemented!");
      fail;
    }
  }
}

impure fn gen(&fn_info enclosing, ts_ann a, def_id id) {
  check(enclosing.contains_key(id));
  let uint i = enclosing.get(id);

  set_in_postcond(i, a.conditions);
}

fn find_pre_post_stmt(_fn_info_map fm, &fn_info enclosing, &ast.stmt s)
  -> ast.stmt {
  auto num_local_vars = num_locals(enclosing);

  alt(s.node) {
    case(ast.stmt_decl(?adecl, ?a)) {
      alt(adecl.node) {
        case(ast.decl_local(?alocal)) {
          alt(alocal.init) {
            case(some[ast.initializer](?an_init)) {
              let @expr r = find_pre_post_expr(enclosing, *an_init.expr);
              let init_op o = an_init.op;
              let initializer a_i = rec(op=o, expr=r);
              let ann res_ann = with_pp(alocal.ann, expr_pp(*r));
              let @local res_local =
                @rec(ty=alocal.ty, infer=alocal.infer,
                     ident=alocal.ident, init=some[initializer](a_i),
                     id=alocal.id, ann=res_ann);

              let ts_ann stmt_ann;
              alt (a) {
                case (none[@ts_ann]) {
                  stmt_ann = empty_ann(num_local_vars);
                }
                case (some[@ts_ann](?aa)) {
                  stmt_ann = *aa;
                }
              }
              /* Inherit ann from initializer, and add var being
                 initialized to the postcondition */
              set_precondition(stmt_ann, expr_precond(*r));
              set_postcondition(stmt_ann, expr_postcond(*r));
              gen(enclosing, stmt_ann, alocal.id); 
              let stmt_ res = stmt_decl(@respan(adecl.span,
                                                decl_local(res_local)),
                                        some[@ts_ann](@stmt_ann));
              ret (respan(s.span, res)); 
            }
            case(none[ast.initializer]) {
              // log("pre/post for init of " + alocal.ident + ": none");
              let ann res_ann = with_pp(alocal.ann,
                                        empty_pre_post(num_local_vars));
              let @local res_local =
                @rec(ty=alocal.ty, infer=alocal.infer,
                     ident=alocal.ident, init=none[initializer],
                     id=alocal.id, ann=res_ann);
              let stmt_ res =
                stmt_decl
                (@respan(adecl.span, decl_local(res_local)),
                 some[@ts_ann](@empty_ann(num_local_vars)));
              ret respan(s.span, res); /* inherit ann from initializer */
            }
          }
        }
        case(decl_item(?anitem)) {
          auto res_item = find_pre_post_item(fm, enclosing, *anitem);
          ret respan(s.span,
                     stmt_decl(@respan(adecl.span,
                                       decl_item(@res_item)),
                               some[@ts_ann](@empty_ann(num_local_vars))));
        }
      }
    }
    case(stmt_expr(?e,_)) {
      log_expr(e);
      let @expr e_pp = find_pre_post_expr(enclosing, *e);
      /* inherit ann from expr */
      ret respan(s.span,
                 stmt_expr(e_pp,
                           some[@ts_ann]
                           (@ann_to_ts_ann(expr_ann(*e_pp),
                                           num_local_vars)))); 
    }    
  }
}

fn find_pre_post_block(&_fn_info_map fm, &fn_info enclosing, block b)
  -> block {
  fn do_one_(_fn_info_map fm, fn_info i, &@stmt s) -> @stmt {
    ret (@find_pre_post_stmt(fm, i, *s));
  }
  auto do_one = bind do_one_(fm, enclosing, _);
 
  auto ss = _vec.map[@stmt, @stmt](do_one, b.node.stmts);
  fn do_inner_(fn_info i, &@expr e) -> @expr {
    ret find_pre_post_expr(i, *e);
  }
  auto do_inner = bind do_inner_(enclosing, _);
  let option.t[@expr] e_ = option.map[@expr, @expr](do_inner, b.node.expr);
  let block_ b_res = rec(stmts=ss, expr=e_, index=b.node.index);
  ret respan(b.span, b_res);
}

fn find_pre_post_fn(&_fn_info_map fm, &fn_info fi, &_fn f) -> _fn {
  ret rec(decl=f.decl, proto=f.proto,
          body=find_pre_post_block(fm, fi, f.body));
}

fn check_item_fn(&_fn_info_map fm, &span sp, ident i, &ast._fn f,
                 vec[ast.ty_param] ty_params, def_id id, ann a) -> @item {

  check (fm.contains_key(id));
  auto res_f = find_pre_post_fn(fm, fm.get(id), f);  

  ret @respan(sp, ast.item_fn(i, res_f, ty_params, id, a));
}

/* FIXME */
fn find_pre_post_state_expr(&_fn_info_map fm, &fn_info enclosing,
                            &prestate pres, expr e)
  -> tup(bool, @expr) {
  log("Implement find_pre_post_state_expr!");
  fail;
}

/* FIXME: This isn't done yet. */
fn find_pre_post_state_stmt(&_fn_info_map fm, &fn_info enclosing,
                            &prestate pres, @stmt s) -> bool {
  auto changed = false;
  alt (s.node) {
    case (stmt_decl(?adecl, ?a)) {
      alt (adecl.node) {
        case (ast.decl_local(?alocal)) {
          alt (alocal.init) {
            case (some[ast.initializer](?an_init)) {
              auto p = find_pre_post_state_expr(fm, enclosing,
                                                pres, *an_init.expr);
              fail; /* FIXME */
              /* Next: copy pres into a's prestate;
                 find the poststate by taking p's poststate
                 and setting the bit for alocal.id */
            }
          }
        }
      }
    }
  }
}

/* Returns a pair of a new block, with possibly a changed pre- or
   post-state, and a boolean flag saying whether the function's pre- or 
   poststate changed */
fn find_pre_post_state_block(&_fn_info_map fm, &fn_info enclosing, block b)
  -> tup(bool, block) {
  auto changed = false;
  auto num_local_vars = num_locals(enclosing);

  /* First, set the pre-states and post-states for every expression */
  auto pres = empty_prestate(num_local_vars);
  
  /* Iterate over each stmt. The new prestate is <pres>. The poststate
   consist of improving <pres> with whatever variables this stmt initializes.
  Then <pres> becomes the new poststate. */ 
  for (@stmt s in b.node.stmts) {
    changed = changed || find_pre_post_state_stmt(fm, enclosing, pres, s);
      /* Shouldn't need to rebuild the stmt.
         This just updates bit-vectors
         in a side-effecting way. */
    extend_prestate(pres, stmt_poststate(*s, num_local_vars));
  }

  fn do_inner_(_fn_info_map fm, fn_info i, prestate p, &@expr e)
    -> tup (bool, @expr) {
    ret find_pre_post_state_expr(fm, i, p, *e);
  }
  auto do_inner = bind do_inner_(fm, enclosing, pres, _);
  let option.t[tup(bool, @expr)] e_ =
    option.map[@expr, tup(bool, @expr)](do_inner, b.node.expr);
  auto s = snd[bool, @expr];
  auto f = fst[bool, @expr];
  changed = changed ||
    from_maybe[bool](false,
                     option.map[tup(bool, @expr), bool](f, e_));
  let block_ b_res = rec(stmts=b.node.stmts,
                         expr=option.map[tup(bool, @expr), @expr](s, e_),
                         index=b.node.index);
  ret tup(changed, respan(b.span, b_res));
}

fn find_pre_post_state_fn(_fn_info_map f_info, fn_info fi, &ast._fn f)
  -> tup(bool, ast._fn) {
  auto p = find_pre_post_state_block(f_info, fi, f.body);
  ret tup(p._0, rec(decl=f.decl, proto=f.proto, body=p._1));
}

fn fixed_point_states(_fn_info_map fm, fn_info f_info,
                      fn (_fn_info_map, fn_info, &ast._fn)
                           -> tup(bool, ast._fn) f,
                      &ast._fn start) -> ast._fn {
  auto next = f(fm, f_info, start);

  if (next._0) {
    // something changed
    be fixed_point_states(fm, f_info, f, next._1);
  }
  else {
    // we're done!
    ret next._1;
  }
}

impure fn check_states_expr(fn_info enclosing, &expr e) -> () {
  let precond prec    = expr_precond(e);
  let prestate pres   = expr_prestate(e);

  if (!implies(pres, prec)) {
    log("check_states_expr: unsatisfied precondition");
    fail;
  }
}

fn check_states_stmt(fn_info enclosing, &stmt s) -> () {
  alt (stmt_ann(s)) {
    case (none[@ts_ann]) {
      ret;
    }
    case (some[@ts_ann](?a)) {
      let precond prec    = ann_precond(*a);
      let prestate pres   = ann_prestate(*a);

      if (!implies(pres, prec)) {
        log("check_states_stmt: unsatisfied precondition");
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

fn check_item_fn_state(&_fn_info_map f_info_map, &span sp, ident i,
                       &ast._fn f, vec[ast.ty_param] ty_params, def_id id,
                       ann a) -> @item {

  /* Look up the var-to-bit-num map for this function */
  check(f_info_map.contains_key(id));
  auto f_info = f_info_map.get(id);

  /* Compute the pre- and post-states for this function */
  auto g = find_pre_post_state_fn;
  auto res_f = fixed_point_states(f_info_map, f_info, g, f);

  /* Now compare each expr's pre-state to its precondition
     and post-state to its postcondition */
  check_states_against_conditions(f_info, res_f);

  /* Rebuild the same function */
  ret @respan(sp, ast.item_fn(i, res_f, ty_params, id, a));
}

fn check_crate(@ast.crate crate) -> @ast.crate {

  /* Build the global map from function id to var-to-bit-num-map */
  auto fn_info_map = mk_f_to_fn_info(crate);

  /* Compute the pre and postcondition for every subexpression */
  auto fld = fold.new_identity_fold[_fn_info_map]();
  fld = @rec(fold_item_fn = bind check_item_fn(_,_,_,_,_,_,_) with *fld);
  auto with_pre_postconditions = fold.fold_crate[_fn_info_map]
    (fn_info_map, fld, crate);

  auto fld1 = fold.new_identity_fold[_fn_info_map]();

  fld1 = @rec(fold_item_fn = bind check_item_fn_state(_,_,_,_,_,_,_)
              with *fld1);

  ret fold.fold_crate[_fn_info_map](fn_info_map, fld1,
                                    with_pre_postconditions);
}
