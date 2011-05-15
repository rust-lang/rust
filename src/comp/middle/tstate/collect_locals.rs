import std::_vec;
import std::_vec::plus_option;

import front::ast;
import front::ast::crate;
import front::ast::ann;
import front::ast::arg;
import front::ast::method;
import front::ast::local;
import front::ast::item;
import front::ast::item_fn;
import front::ast::item_obj;
import front::ast::_obj;
import front::ast::obj_def_ids;
import front::ast::_fn;
import front::ast::ty_param;
import front::ast::_mod;
import front::ast::decl;
import front::ast::decl_local;
import front::ast::def_id;
import front::ast::ident;
import middle::fold::span;
import middle::fold::respan;
import middle::fold::new_identity_fold;
import middle::fold::fold_block;
import middle::fold::fold_fn;
import middle::fold::fold_crate;

import aux::fn_info;
import aux::var_info;
import aux::crate_ctxt;

import util::common::new_def_hash;

fn var_is_local(def_id v, fn_info m) -> bool {
  ret (m.vars.contains_key(v));
}

fn collect_local(&@vec[tup(ident, def_id)] vars, &span sp, &@local loc)
    -> @decl {
    log("collect_local: pushing " + loc.ident);
    _vec::push[tup(ident, def_id)](*vars, tup(loc.ident, loc.id));
    ret @respan(sp, decl_local(loc));
}

fn find_locals(_fn f) -> @vec[tup(ident,def_id)] {
  auto res = @_vec::alloc[tup(ident,def_id)](0u);

  auto fld = new_identity_fold[@vec[tup(ident, def_id)]]();
  fld = @rec(fold_decl_local = bind collect_local(_,_,_) with *fld);
  auto ignore = fold_fn[@vec[tup(ident, def_id)]](res, fld, f);

  ret res;
}


fn add_var(def_id v, ident nm, uint next, fn_info tbl) -> uint {
  log(nm + " |-> " + util::common::uistr(next));
  tbl.vars.insert(v, tup(next,nm));
  ret (next + 1u);
}

/* builds a table mapping each local var defined in f
   to a bit number in the precondition/postcondition vectors */
fn mk_fn_info(_fn f, def_id f_id, ident f_name) -> fn_info {
    auto res = rec(vars=@new_def_hash[var_info](),
                   cf=f.decl.cf);
    let uint next = 0u;
    let vec[arg] f_args = f.decl.inputs;

    /* ignore args, which we know are initialized;
       just collect locally declared vars */

    let @vec[tup(ident,def_id)] locals = find_locals(f);
    // log (uistr(_vec::len[tup(ident, def_id)](locals)) + " locals");
    for (tup(ident,def_id) p in *locals) {
        next = add_var(p._1, p._0, next, res);
    }
    /* add a pseudo-entry for the function's return value
       we can safely use the function's name itself for this purpose */
    add_var(f_id, f_name, next, res);

    ret res;
}

/* extends mk_fn_info to a function item, side-effecting the map fi from
   function IDs to fn_info maps */
fn mk_fn_info_item_fn(&crate_ctxt ccx, &span sp, &ident i, &_fn f,
                 &vec[ty_param] ty_params, &def_id id, &ann a) -> @item {
    auto f_inf = mk_fn_info(f, id, i);
    ccx.fm.insert(id, f_inf);
    //  log_err("inserting: " + i);
    ret @respan(sp, item_fn(i, f, ty_params, id, a));
}

/* extends mk_fn_info to an obj item, side-effecting the map fi from
   function IDs to fn_info maps */
fn mk_fn_info_item_obj(&crate_ctxt ccx, &span sp, &ident i, &_obj o,
                       &vec[ty_param] ty_params,
                       &obj_def_ids odid, &ann a) -> @item {
    auto all_methods = _vec::clone[@method](o.methods);
    plus_option[@method](all_methods, o.dtor);
    auto f_inf;
    for (@method m in all_methods) {
        f_inf = mk_fn_info(m.node.meth, m.node.id, m.node.ident);
        ccx.fm.insert(m.node.id, f_inf);
    }
    ret @respan(sp, item_obj(i, o, ty_params, odid, a));
}


/* initializes the global fn_info_map (mapping each function ID, including
   nested locally defined functions, onto a mapping from local variable name
   to bit number) */
fn mk_f_to_fn_info(&crate_ctxt ccx, @crate c) -> () {

  auto fld = new_identity_fold[crate_ctxt]();
  fld = @rec(fold_item_fn  = bind mk_fn_info_item_fn(_,_,_,_,_,_,_),
             fold_item_obj = bind mk_fn_info_item_obj(_,_,_,_,_,_,_)
               with *fld);
  fold_crate[crate_ctxt](ccx, fld, c);
}
