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

import middle::walk::walk_crate;
import middle::walk::walk_fn;
import middle::walk::ast_visitor;

import aux::fn_info;
import aux::var_info;
import aux::crate_ctxt;

import util::common::new_def_hash;
import util::common::uistr;
import util::common::span;

type identifier = rec(ident name, def_id id, span sp);

fn var_is_local(def_id v, fn_info m) -> bool {
  ret (m.vars.contains_key(v));
}

fn collect_local(&@vec[identifier] vars, &@decl d) -> () {
    alt (d.node) {
      case (decl_local(?loc)) {
        log("collect_local: pushing " + loc.ident);
        vec::push[identifier](*vars, rec(name=loc.ident,
                                         id=loc.id,
                                         sp=d.span));
      }
      case (_) { ret; }
    }
}

fn find_locals(&_fn f, &span sp, &ident i, &def_id d, &ann a)
    -> @vec[identifier] {
  auto res = @vec::alloc[identifier](0u);
  auto visitor = walk::default_visitor();
  visitor = rec(visit_decl_pre=bind collect_local(res,_) with visitor);
  walk_fn(visitor, f, sp, i, d, a);
  ret res;
}


fn add_var(def_id v, span sp, ident nm, uint next, fn_info tbl) -> uint {
  log(nm + " |-> " + util::common::uistr(next));
  tbl.vars.insert(v, rec(bit_num=next, name=nm, sp=sp));
  ret (next + 1u);
}

/* builds a table mapping each local var defined in f
   to a bit number in the precondition/postcondition vectors */
fn mk_fn_info(&crate_ctxt ccx, &_fn f, &span f_sp,
              &ident f_name, &def_id f_id, &ann a)
    -> () {
    auto res = rec(vars=@new_def_hash[var_info](),
                   cf=f.decl.cf);
    let uint next = 0u;
    let vec[arg] f_args = f.decl.inputs;

    /* ignore args, which we know are initialized;
       just collect locally declared vars */

    let @vec[identifier] locals = find_locals(f, f_sp, f_name, f_id, a);
    for (identifier p in *locals) {
        next = add_var(p.id, p.sp, p.name, next, res);
    }
    /* add a pseudo-entry for the function's return value
       we can safely use the function's name itself for this purpose */
    add_var(f_id, f_sp, f_name, next, res);

    log(f_name + " has " + uistr(vec::len[identifier](*locals))
            + " locals");
   
    ccx.fm.insert(f_id, res);
}

/* initializes the global fn_info_map (mapping each function ID, including
   nested locally defined functions, onto a mapping from local variable name
   to bit number) */
fn mk_f_to_fn_info(&crate_ctxt ccx, @crate c) -> () {
  let ast_visitor vars_visitor = walk::default_visitor();
  vars_visitor = rec(visit_fn_pre=bind mk_fn_info(ccx,_,_,_,_,_)
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

