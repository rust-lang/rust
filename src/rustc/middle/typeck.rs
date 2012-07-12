/*

typeck.rs, an introduction

The type checker is responsible for:

1. Determining the type of each expression
2. Resolving methods and traits
3. Guaranteeing that most type rules are met ("most?", you say, "why most?"
   Well, dear reader, read on)

The main entry point is `check_crate()`.  Type checking operates in two major
phases: collect and check.  The collect phase passes over all items and
determines their type, without examining their "innards".  The check phase
then checks function bodies and so forth.

Within the check phase, we check each function body one at a time (bodies of
function expressions are checked as part of the containing function).
Inference is used to supply types wherever they are unknown. The actual
checking of a function itself has several phases (check, regionck, writeback),
as discussed in the documentation for the `check` module.

The type checker is defined into various submodules which are documented
independently:

- astconv: converts the AST representation of types
  into the `ty` representation

- collect: computes the types of each top-level item and enters them into
  the `cx.tcache` table for later use

- check: walks over function bodies and type checks them, inferring types for
  local variables, type parameters, etc as necessary.

- infer: finds the types to use for each type variable such that
  all subtyping and assignment constraints are met.  In essence, the check
  module specifies the constraints, and the infer module solves them.

*/

import result::{result, extensions};
import syntax::{ast, ast_util, ast_map};
import ast::spanned;
import syntax::ast_map::node_id_to_str;
import syntax::ast_util::{local_def, respan, split_class_items};
import syntax::visit;
import metadata::csearch;
import driver::session::session;
import util::common::may_break;
import syntax::codemap::span;
import pat_util::{pat_is_variant, pat_id_map};
import middle::ty;
import middle::ty::{arg, field, node_type_table, mk_nil,
                    ty_param_bounds_and_ty, lookup_public_fields};
import middle::typeck::infer::methods;
import std::smallintmap;
import std::smallintmap::map;
import std::map;
import std::map::{hashmap, int_hash};
import std::serialization::{serialize_uint, deserialize_uint};
import vec::each;
import syntax::print::pprust::*;
import util::ppaux::{ty_to_str, tys_to_str, region_to_str,
                     bound_region_to_str, vstore_to_str};
import util::common::{indent, indenter};
import std::list;
import list::{list, nil, cons};

export check_crate;
export infer;
export method_map;
export method_origin, serialize_method_origin, deserialize_method_origin;
export method_map_entry, serialize_method_map_entry;
export deserialize_method_map_entry;
export vtable_map;
export vtable_res;
export vtable_origin;
export method_static, method_param, method_trait;
export vtable_static, vtable_param, vtable_trait;

#[auto_serialize]
enum method_origin {
    // fully statically resolved method
    method_static(ast::def_id),

    // method invoked on a type parameter with a bounded trait
    method_param(method_param),

    // method invoked on a boxed trait
    method_trait(ast::def_id, uint),
}

// details for a method invoked with a receiver whose type is a type parameter
// with a bounded trait.
#[auto_serialize]
type method_param = {
    // the trait containing the method to be invoked
    trait_id: ast::def_id,

    // index of the method to be invoked amongst the trait's methods
    method_num: uint,

    // index of the type parameter (from those that are in scope) that is
    // the type of the receiver
    param_num: uint,

    // index of the bound for this type parameter which specifies the trait
    bound_num: uint
};

#[auto_serialize]
type method_map_entry = {
    // number of derefs that are required on the receiver
    derefs: uint,

    // method details being invoked
    origin: method_origin
};

// maps from an expression id that corresponds to a method call to the details
// of the method to be invoked
type method_map = hashmap<ast::node_id, method_map_entry>;

// Resolutions for bounds of all parameters, left to right, for a given path.
type vtable_res = @~[vtable_origin];

enum vtable_origin {
    /*
      Statically known vtable. def_id gives the class or impl item
      from whence comes the vtable, and tys are the type substs.
      vtable_res is the vtable itself
     */
    vtable_static(ast::def_id, ~[ty::t], vtable_res),
    /*
      Dynamic vtable, comes from a parameter that has a bound on it:
      fn foo<T: quux, baz, bar>(a: T) -- a's vtable would have a
      vtable_param origin

      The first uint is the param number (identifying T in the example),
      and the second is the bound number (identifying baz)
     */
    vtable_param(uint, uint),
    /*
      Dynamic vtable, comes from something known to have an interface
      type. def_id refers to the trait item, tys are the substs
     */
    vtable_trait(ast::def_id, ~[ty::t]),
}

type vtable_map = hashmap<ast::node_id, vtable_res>;

type ty_param_substs_and_ty = {substs: ty::substs, ty: ty::t};

type ty_table = hashmap<ast::def_id, ty::t>;

type crate_ctxt = {impl_map: resolve::impl_map,
                   method_map: method_map,
                   vtable_map: vtable_map,
                   tcx: ty::ctxt};

// Functions that write types into the node type table
fn write_ty_to_tcx(tcx: ty::ctxt, node_id: ast::node_id, ty: ty::t) {
    #debug["write_ty_to_tcx(%d, %s)", node_id, ty_to_str(tcx, ty)];
    smallintmap::insert(*tcx.node_types, node_id as uint, ty);
}
fn write_substs_to_tcx(tcx: ty::ctxt,
                       node_id: ast::node_id,
                       +substs: ~[ty::t]) {
    if substs.len() > 0u {
        tcx.node_type_substs.insert(node_id, substs);
    }
}

fn lookup_def_tcx(tcx: ty::ctxt, sp: span, id: ast::node_id) -> ast::def {
    alt tcx.def_map.find(id) {
      some(x) { x }
      _ {
        tcx.sess.span_fatal(sp, "internal error looking up a definition")
      }
    }
}

fn lookup_def_ccx(ccx: @crate_ctxt, sp: span, id: ast::node_id) -> ast::def {
    lookup_def_tcx(ccx.tcx, sp, id)
}

fn no_params(t: ty::t) -> ty::ty_param_bounds_and_ty {
    {bounds: @~[], rp: false, ty: t}
}

fn require_same_types(
    tcx: ty::ctxt,
    maybe_infcx: option<infer::infer_ctxt>,
    span: span,
    t1: ty::t,
    t2: ty::t,
    msg: fn() -> str) -> bool {

    let l_tcx, l_infcx;
    alt maybe_infcx {
      none {
        l_tcx = tcx;
        l_infcx = infer::new_infer_ctxt(tcx);
      }
      some(i) {
        l_tcx = i.tcx;
        l_infcx = i;
      }
    }

    alt infer::mk_eqty(l_infcx, t1, t2) {
      result::ok(()) { true }
      result::err(terr) {
        l_tcx.sess.span_err(span, msg() + ": " +
            ty::type_err_to_str(l_tcx, terr));
        false
      }
    }
}

fn arg_is_argv_ty(_tcx: ty::ctxt, a: ty::arg) -> bool {
    alt ty::get(a.ty).struct {
      ty::ty_evec(mt, vstore_uniq) {
        if mt.mutbl != ast::m_imm { ret false; }
        alt ty::get(mt.ty).struct {
          ty::ty_estr(vstore_uniq) { ret true; }
          _ { ret false; }
        }
      }
      _ { ret false; }
    }
}

fn check_main_fn_ty(ccx: @crate_ctxt,
                    main_id: ast::node_id,
                    main_span: span) {

    let tcx = ccx.tcx;
    let main_t = ty::node_id_to_type(tcx, main_id);
    alt ty::get(main_t).struct {
      ty::ty_fn({purity: ast::impure_fn, proto: ast::proto_bare,
                 inputs, output, ret_style: ast::return_val, constraints}) {
        alt tcx.items.find(main_id) {
         some(ast_map::node_item(it,_)) {
             alt it.node {
               ast::item_fn(_,ps,_) if vec::is_not_empty(ps) {
                  tcx.sess.span_err(main_span,
                    "main function is not allowed to have type parameters");
                  ret;
               }
               _ {}
             }
         }
         _ {}
        }
        let mut ok = vec::len(constraints) == 0u;
        ok &= ty::type_is_nil(output);
        let num_args = vec::len(inputs);
        ok &= num_args == 0u || num_args == 1u &&
              arg_is_argv_ty(tcx, inputs[0]);
        if !ok {
                tcx.sess.span_err(main_span,
                   #fmt("Wrong type in main function: found `%s`, \
                   expecting `extern fn(~[str]) -> ()` \
                   or `extern fn() -> ()`",
                         ty_to_str(tcx, main_t)));
         }
      }
      _ {
        tcx.sess.span_bug(main_span,
                          "main has a non-function type: found `" +
                              ty_to_str(tcx, main_t) + "`");
      }
    }
}

fn check_for_main_fn(ccx: @crate_ctxt) {
    let tcx = ccx.tcx;
    if !tcx.sess.building_library {
        alt copy tcx.sess.main_fn {
          some((id, sp)) { check_main_fn_ty(ccx, id, sp); }
          none { tcx.sess.err("main function not found"); }
        }
    }
}

fn check_crate(tcx: ty::ctxt, impl_map: resolve::impl_map,
               crate: @ast::crate) -> (method_map, vtable_map) {
    let ccx = @{impl_map: impl_map,
                method_map: std::map::int_hash(),
                vtable_map: std::map::int_hash(),
                tcx: tcx};
    collect::collect_item_types(ccx, crate);

    if tcx.sess.coherence() {
        coherence::check_coherence(ccx, crate);
    }

    check::check_item_types(ccx, crate);
    check_for_main_fn(ccx);
    tcx.sess.abort_if_errors();
    (ccx.method_map, ccx.vtable_map)
}
//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
