/*

typeck.rs, an introduction

The type checker is responsible for:

1. Determining the type of each expression
2. Resolving methods and ifaces
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
import util::common::*;
import syntax::codemap::span;
import pat_util::{pat_is_variant, pat_id_map};
import middle::ty;
import middle::ty::{arg, field, node_type_table, mk_nil,
                    ty_param_bounds_and_ty, lookup_public_fields};
import middle::ty::{ty_vid, region_vid, vid};
import middle::typeck::infer::methods;
import util::ppaux::{ty_to_str, tys_to_str, region_to_str,
                     bound_region_to_str, vstore_to_str};
import std::smallintmap;
import std::smallintmap::map;
import std::map;
import std::map::{hashmap, int_hash};
import std::serialization::{serialize_uint, deserialize_uint};
import vec::each;
import syntax::print::pprust::*;
import util::common::{indent, indenter};
import std::list;
import list::{list, nil, cons};

export check_crate;
export infer;
export method_map;
export method_origin, serialize_method_origin, deserialize_method_origin;
export vtable_map;
export vtable_res;
export vtable_origin;

#[auto_serialize]
enum method_origin {
    method_static(ast::def_id),
    // iface id, method num, param num, bound num
    method_param(ast::def_id, uint, uint, uint),
    method_iface(ast::def_id, uint),
}
type method_map = hashmap<ast::node_id, method_origin>;

// Resolutions for bounds of all parameters, left to right, for a given path.
type vtable_res = @[vtable_origin];

enum vtable_origin {
    /*
      Statically known vtable. def_id gives the class or impl item
      from whence comes the vtable, and tys are the type substs.
      vtable_res is the vtable itself
     */
    vtable_static(ast::def_id, [ty::t], vtable_res),
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
      type. def_id refers to the iface item, tys are the substs
     */
    vtable_iface(ast::def_id, [ty::t]),
}

type vtable_map = hashmap<ast::node_id, vtable_res>;

type ty_param_substs_and_ty = {substs: ty::substs, ty: ty::t};

type ty_table = hashmap<ast::def_id, ty::t>;

type crate_ctxt = {impl_map: resolve::impl_map,
                   method_map: method_map,
                   vtable_map: vtable_map,
                   // Not at all sure it's right to put these here
                   /* node_id for the class this fn is in --
                      none if it's not in a class */
                   enclosing_class_id: option<ast::node_id>,
                   /* map from node_ids for enclosing-class
                      vars and methods to types */
                   enclosing_class: class_map,
                   tcx: ty::ctxt};

type class_map = hashmap<ast::node_id, ty::t>;

// Functions that write types into the node type table
fn write_ty_to_tcx(tcx: ty::ctxt, node_id: ast::node_id, ty: ty::t) {
    #debug["write_ty_to_tcx(%d, %s)", node_id, ty_to_str(tcx, ty)];
    smallintmap::insert(*tcx.node_types, node_id as uint, ty);
}
fn write_substs_to_tcx(tcx: ty::ctxt,
                       node_id: ast::node_id,
                       +substs: [ty::t]) {
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
    {bounds: @[], rp: ast::rp_none, ty: t}
}

fn require_same_types(
    tcx: ty::ctxt,
    span: span,
    t1: ty::t,
    t2: ty::t,
    msg: fn() -> str) -> bool {

    alt infer::compare_tys(tcx, t1, t2) {
      result::ok(()) { true }
      result::err(terr) {
        tcx.sess.span_err(span, msg() + ": " +
            ty::type_err_to_str(tcx, terr));
        false
      }
    }
}

fn arg_is_argv_ty(_tcx: ty::ctxt, a: ty::arg) -> bool {
    alt ty::get(a.ty).struct {
      ty::ty_vec(mt) {
        if mt.mutbl != ast::m_imm { ret false; }
        alt ty::get(mt.ty).struct {
          ty::ty_str { ret true; }
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
                   expecting `native fn([str]) -> ()` or `native fn() -> ()`",
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

fn check_for_main_fn(ccx: @crate_ctxt, crate: @ast::crate) {
    let tcx = ccx.tcx;
    if !tcx.sess.building_library {
        alt copy tcx.sess.main_fn {
          some((id, sp)) { check_main_fn_ty(ccx, id, sp); }
          none { tcx.sess.span_err(crate.span, "main function not found"); }
        }
    }
}

fn check_crate(tcx: ty::ctxt, impl_map: resolve::impl_map,
               crate: @ast::crate) -> (method_map, vtable_map) {
    let ccx = @{impl_map: impl_map,
                method_map: std::map::int_hash(),
                vtable_map: std::map::int_hash(),
                enclosing_class_id: none,
                enclosing_class: std::map::int_hash(),
                tcx: tcx};
    collect::collect_item_types(ccx, crate);
    check::check_item_types(ccx, crate);
    check_for_main_fn(ccx, crate);
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
