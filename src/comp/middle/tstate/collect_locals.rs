import std::vec;
import std::vec::plus_option;

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
import middle::walk::walk_crate;
import middle::walk::walk_fn;
import middle::walk::ast_visitor;

import aux::fn_info;
import aux::var_info;
import aux::crate_ctxt;

import util::common::new_def_hash;
import util::common::uistr;

fn var_is_local(def_id v, fn_info m) -> bool {
  ret (m.vars.contains_key(v));
}

fn collect_local(&@vec[tup(ident, def_id)] vars, &@decl d) -> () {
    alt (d.node) {
      case (decl_local(?loc)) {
        log("collect_local: pushing " + loc.ident);
        vec::push[tup(ident, def_id)](*vars, tup(loc.ident, loc.id));
      }
      case (_) { ret; }
    }
}

fn find_locals(_fn f, def_id d) -> @vec[tup(ident,def_id)] {
  auto res = @vec::alloc[tup(ident,def_id)](0u);

  auto visitor = walk::default_visitor();
  visitor = rec(visit_decl_pre=bind collect_local(res,_) with visitor);
  walk_fn(visitor, f, d);

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

    let @vec[tup(ident,def_id)] locals = find_locals(f, f_id);
    // log (uistr(vec::len[tup(ident, def_id)](locals)) + " locals");
    for (tup(ident,def_id) p in *locals) {
        next = add_var(p._1, p._0, next, res);
    }
    /* add a pseudo-entry for the function's return value
       we can safely use the function's name itself for this purpose */
    add_var(f_id, f_name, next, res);

    log(f_name + " has " + uistr(vec::len[tup(ident, def_id)](*locals))
            + " locals");
   
    ret res;
}

/* extends mk_fn_info to an item, side-effecting the map fi from
   function IDs to fn_info maps
   only looks at function and object items. */
fn mk_fn_info_item (&crate_ctxt ccx, &@item i) -> () {
  alt (i.node) {
    case (item_fn(?i,?f,?ty_params,?id,?a)) {
      auto f_inf = mk_fn_info(f, id, i);
      ccx.fm.insert(id, f_inf);
    }
    case (item_obj(?i,?o,?ty_params,?odid,?a)) {
      auto all_methods = vec::clone[@method](o.methods);
      plus_option[@method](all_methods, o.dtor);
      auto f_inf;
      for (@method m in all_methods) {
        f_inf = mk_fn_info(m.node.meth, m.node.id, m.node.ident);
        ccx.fm.insert(m.node.id, f_inf);
      }
    }
    case (_) { ret; }
  }
}

/* initializes the global fn_info_map (mapping each function ID, including
   nested locally defined functions, onto a mapping from local variable name
   to bit number) */
fn mk_f_to_fn_info(&crate_ctxt ccx, @crate c) -> () {
  let ast_visitor vars_visitor = walk::default_visitor();
  vars_visitor = rec(visit_item_post=bind mk_fn_info_item(ccx,_)
                     with vars_visitor);

  walk_crate(vars_visitor, *c);
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

