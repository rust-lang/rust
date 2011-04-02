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
import front.ast.decl_local;
import front.ast.decl_item;
import front.ast.ident;
import front.ast.def_id;
import front.ast.ann;
import front.ast.expr;
import front.ast.expr_call;
import front.ast.expr_path;
import front.ast.expr_log;
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

import middle.fold;
import middle.fold.respan;
import driver.session;
import util.common;
import util.common.span;
import util.common.spanned;
import util.common.new_str_hash;
import util.typestate_ann;
import util.typestate_ann.ts_ann;
import util.typestate_ann.empty_pre_post;
import util.typestate_ann.true_precond;
import util.typestate_ann.true_postcond;
import util.typestate_ann.postcond;
import util.typestate_ann.precond;
import util.typestate_ann.pre_and_post;
import util.typestate_ann.get_pre;
import util.typestate_ann.get_post;

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
import std.map.hashmap;
import std.list;
import std.list.list;
import std.list.cons;
import std.list.nil;
import std.list.foldl;
import std.list.find;

import util.typestate_ann;
import util.typestate_ann.difference;
import util.typestate_ann.union;
import util.typestate_ann.pps_len;
import util.typestate_ann.require_and_preserve;

/**********************************************************************/
/* mapping from variable name to bit number */
type fn_info = std.map.hashmap[ident, uint];

fn bit_num(ident v, fn_info m) -> uint {
  ret m.get(v);
}
fn num_locals(fn_info m) -> uint {
  ret m.size();
}

fn mk_fn_info(_fn f) -> fn_info {
  ret new_str_hash[uint](); /* FIXME: actually implement this */
}
/**********************************************************************/
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

fn expr_pp(&@expr e) -> pre_and_post {
  alt (expr_ann(*e)) {
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
          ret *p;
        }
      }
    }
  }
}

fn expr_precond(&expr e) -> precond {
  ret (expr_pp(@e)).precondition;
}

fn expr_postcond(&@expr e) -> postcond {
  ret (expr_pp(e)).postcondition;
}

fn with_pp(ann a, @pre_and_post p) -> ann {
  alt (a) {
    case (ann_none) {
      log("with_pp: the impossible happened");
      fail; /* shouldn't happen b/c code is typechecked */
    }
    case (ann_type(?t, ?ps, _)) {
      ret (ann_type(t, ps, some[@ts_ann](p)));
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
  auto other = rest.(0);
  union(first, other);
  union_postconds_go(first, slice[postcond](rest, 1u, len[postcond](rest)));
  ret first;
}

fn union_postconds(&vec[postcond] pcs) -> postcond {
  check (len[postcond](pcs) > 0u);

  be union_postconds_go(pcs.(0), pcs);
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

fn find_pre_post_mod(&_mod m) -> _mod {
  ret m; /* FIXME */
}

fn find_pre_post_native_mod(&native_mod m) -> native_mod {
  ret m; /* FIXME */
}
 
fn find_pre_post_obj(_obj o) -> _obj {
  ret o; /* FIXME */
}

impure fn find_pre_post_item(fn_info enclosing, &item i) -> item {
  alt (i.node) {
    case (ast.item_const(?id, ?t, ?e, ?di, ?a)) {
      auto e_pp = find_pre_post_expr(enclosing, *e);
      ret (respan(i.span,
                   ast.item_const(id, t, e_pp, di,
                              with_pp(a, @expr_pp(e_pp)))));
    }
    case (ast.item_fn(?id, ?f, ?ps, ?di, ?a)) {
      auto f_pp = find_pre_post_fn(f);
      ret (respan(i.span, 
                   ast.item_fn(id, f_pp, ps, di, a)));
    }
    case (ast.item_mod(?id, ?m, ?di)) {
      auto m_pp = find_pre_post_mod(m);
      ret (respan(i.span,
                   ast.item_mod(id, m_pp, di)));
    }
    case (ast.item_native_mod(?id, ?nm, ?di)) {
      auto n_pp = find_pre_post_native_mod(nm);
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
      ret (respan(i.span,
                   ast.item_obj(id, o_pp, ps, di, a)));
    }
  }
}

impure fn find_pre_post_expr(&fn_info enclosing, &expr e) -> @expr {
  auto num_local_vars = num_locals(enclosing);

  fn do_rand_(fn_info enclosing, &@expr e) -> @expr {
    be find_pre_post_expr(enclosing, *e);
  }

  auto do_rand = bind do_rand_(enclosing,_);

  alt(e.node) {
    case(expr_call(?oper, ?rands, ?a)) {
      auto pp_oper = find_pre_post_expr(enclosing, *oper);
      
      auto f = do_rand;
      auto pp_rands = _vec.map[@expr, @expr](f, rands);
      
      auto g = expr_pp;
      auto pps = _vec.map[@expr, pre_and_post]      
        (g, pp_rands);
      _vec.push[pre_and_post](pps, expr_pp(pp_oper));
      auto h = get_post;
      auto res_postconds = _vec.map[pre_and_post, postcond](h, pps);
      auto res_postcond = union_postconds(res_postconds);
      let @pre_and_post pp =
        @rec(precondition=seq_preconds(num_local_vars, pps),
             postcondition=res_postcond);
      let ann a_res = with_pp(a, pp);
      ret (@respan(e.span,
                   expr_call(pp_oper, pp_rands, a_res)));
                        
    }
    case(expr_path(?p, ?maybe_def, ?a)) {
      check (len[ident](p.node.idents) > 0u);
      auto referent = p.node.idents.(0);
      auto i = bit_num(referent, enclosing);
      auto res = empty_pre_post(num_local_vars);
      require_and_preserve(i, *res);
      ret (@respan
           (e.span,
            expr_path(p, maybe_def,
                      with_pp(a, res))));
    }
    case(expr_log(?e, ?a)) {
      auto e_pp = find_pre_post_expr(enclosing, *e);
      ret (@respan(e.span,
                   expr_log(e_pp, with_pp(a, @expr_pp(e_pp)))));
    }
    case(_) {
      log("this sort of expr isn't implemented!");
      fail;
    }
  }
}

impure fn find_pre_post_for_each_stmt(&fn_info enclosing, &ast.stmt s)
  -> ast.stmt {
  auto num_local_vars = num_locals(enclosing);

  alt(s.node) {
    case(ast.stmt_decl(?adecl)) {
      alt(adecl.node) {
        case(ast.decl_local(?alocal)) {
          alt(alocal.init) {
            case(some[ast.initializer](?an_init)) {
              let @expr r = find_pre_post_expr(enclosing, *an_init.expr);
              let init_op o = an_init.op;
              let initializer a_i = rec(op=o, expr=r);
              let ann res_ann = with_pp(alocal.ann, @expr_pp(r));
              let @local res_local =
                @rec(ty=alocal.ty, infer=alocal.infer,
                     ident=alocal.ident, init=some[initializer](a_i),
                     id=alocal.id, ann=res_ann);
              let stmt_ res = stmt_decl(@respan(adecl.span,
                                                decl_local(res_local)));
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
                (@respan(adecl.span, decl_local(res_local)));
              ret (respan (s.span, res));
            }
          }
        }
        case(decl_item(?anitem)) {
          auto res_item = find_pre_post_item(enclosing, *anitem);
          ret (respan(s.span, stmt_decl(@respan(adecl.span,
                                               decl_item(@res_item))))); 
        }
      }
    }
    case(stmt_expr(?e)) {
      let @expr e_pp = find_pre_post_expr(enclosing, *e);
      ret (respan(s.span, stmt_expr(e_pp)));
    }    
  }
}

fn find_pre_post_block(fn_info enclosing, block b) -> block {
  fn do_one_(fn_info i, &@stmt s) -> @stmt {
    ret (@find_pre_post_for_each_stmt(i, *s));
  }
  auto do_one = bind do_one_(enclosing, _);
 
  auto ss = _vec.map[@stmt, @stmt](do_one, b.node.stmts);
  fn do_inner_(fn_info i, &@expr e) -> @expr {
    ret find_pre_post_expr(i, *e);
  }
  auto do_inner = bind do_inner_(enclosing, _);
  let option.t[@expr] e_ = option.map[@expr, @expr](do_inner, b.node.expr);
  let block_ b_res = rec(stmts=ss, expr=e_, index=b.node.index);
  ret respan(b.span, b_res);
}

fn find_pre_post_fn(&_fn f) -> _fn {
  let fn_info fi = mk_fn_info(f);
  ret rec(decl=f.decl, proto=f.proto,
          body=find_pre_post_block(fi, f.body));
}

fn check_item_fn(&@() ignore, &span sp, ident i, &ast._fn f,
                 vec[ast.ty_param] ty_params, def_id id, ann a) -> @item {

  auto res_f = find_pre_post_fn(f);  

  ret @respan(sp, ast.item_fn(i, res_f, ty_params, id, a));
}

fn check_crate(@ast.crate crate) -> @ast.crate {
  auto fld = fold.new_identity_fold[@()]();

  fld = @rec(fold_item_fn = bind check_item_fn(_,_,_,_,_,_,_) with *fld);

  ret fold.fold_crate[@()](@(), fld, crate);
}
