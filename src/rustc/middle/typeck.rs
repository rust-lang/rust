import result::result;
import syntax::{ast, ast_util};
import ast::spanned;
import syntax::ast_util::{local_def, respan, split_class_items};
import syntax::visit;
import metadata::csearch;
import driver::session::session;
import util::common::*;
import syntax::codemap::span;
import pat_util::*;
import middle::ty;
import middle::ty::{arg, field, node_type_table, mk_nil,
                    ty_param_bounds_and_ty, lookup_public_fields};
import middle::ty::{ty_vid, region_vid, vid};
import util::ppaux::ty_to_str;
import std::smallintmap;
import std::smallintmap::map;
import std::map;
import std::map::{hashmap, int_hash};
import std::serialization::{serialize_uint, deserialize_uint};
import std::ufind;
import syntax::print::pprust::*;
import util::common::indent;

export check_crate;
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
    vtable_static(ast::def_id, [ty::t], vtable_res),
    // Param number, bound number
    vtable_param(uint, uint),
    vtable_iface(ast::def_id, [ty::t]),
}

type vtable_map = hashmap<ast::node_id, vtable_res>;

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

type fn_ctxt =
    // var_bindings, locals and next_var_id are shared
    // with any nested functions that capture the environment
    // (and with any functions whose environment is being captured).
    {self_ty: option<ty::t>,
     ret_ty: ty::t,
     // Used by loop bodies that return from the outer function
     indirect_ret_ty: option<ty::t>,
     purity: ast::purity,
     proto: ast::proto,
     infcx: infer::infer_ctxt,
     locals: hashmap<ast::node_id, ty_vid>,
     next_var_id: @mut uint,
     next_region_var_id: @mut uint,

     // While type checking a function, the intermediate types for the
     // expressions, blocks, and so forth contained within the function are
     // stored in these tables.  These types may contain unresolved type
     // variables.  After type checking is complete, the functions in the
     // writeback module are used to take the types from this table, resolve
     // them, and then write them into their permanent home in the type
     // context `ccx.tcx`.
     //
     // This means that during inferencing you should use `fcx.write_ty()`
     // and `fcx.expr_ty()` / `fcx.node_ty()` to write/obtain the types of
     // nodes within the function.
     //
     // The types of top-level items, which never contain unbound type
     // variables, are stored directly into the `tcx` tables.
     //
     // n.b.: A type variable is not the same thing as a type parameter.  A
     // type variable is rather an "instance" of a type parameter: that is,
     // given a generic function `fn foo<T>(t: T)`: while checking the
     // function `foo`, the type `ty_param(0)` refers to the type `T`, which
     // is treated in abstract.  When `foo()` is called, however, `T` will be
     // substituted for a fresh type variable `ty_var(N)`.  This variable will
     // eventually be resolved to some concrete type (which might itself be
     // type parameter).
     node_types: smallintmap::smallintmap<ty::t>,
     node_type_substs: hashmap<ast::node_id, [ty::t]>,

     ccx: @crate_ctxt};

// Determines whether the given node ID is a use of the def of
// the self ID for the current method, if there is one
fn self_ref(fcx: @fn_ctxt, id: ast::node_id) -> bool {
    // check what def `id` was resolved to (if anything)
    alt fcx.ccx.tcx.def_map.find(id) {
      some(ast::def_self(_)) { true }
      _ { false }
    }
}

fn lookup_local(fcx: @fn_ctxt, sp: span, id: ast::node_id) -> ty_vid {
    alt fcx.locals.find(id) {
      some(x) { x }
      _ {
        fcx.ccx.tcx.sess.span_fatal(sp,
                                    "internal error looking up a local var")
      }
    }
}

fn lookup_def(fcx: @fn_ctxt, sp: span, id: ast::node_id) -> ast::def {
    alt fcx.ccx.tcx.def_map.find(id) {
      some(x) { x }
      _ {
        fcx.ccx.tcx.sess.span_fatal(sp,
                                    "internal error looking up a definition")
      }
    }
}

// Returns the type parameter count and the type for the given definition.
fn ty_param_bounds_and_ty_for_def(fcx: @fn_ctxt, sp: span, defn: ast::def) ->
   ty_param_bounds_and_ty {
    alt defn {
      ast::def_arg(nid, _) {
        assert (fcx.locals.contains_key(nid));
        let typ = ty::mk_var(fcx.ccx.tcx, lookup_local(fcx, sp, nid));
        ret {bounds: @[], ty: typ};
      }
      ast::def_local(nid, _) {
        assert (fcx.locals.contains_key(nid));
        let typ = ty::mk_var(fcx.ccx.tcx, lookup_local(fcx, sp, nid));
        ret {bounds: @[], ty: typ};
      }
      ast::def_self(_) {
        alt fcx.self_ty {
          some(self_ty) {
            ret {bounds: @[], ty: self_ty};
          }
          none {
              fcx.ccx.tcx.sess.span_bug(sp, "def_self with no self_ty");
          }
        }
      }
      ast::def_fn(id, ast::crust_fn) {
        // Crust functions are just u8 pointers
        ret {
            bounds: @[],
            ty: ty::mk_ptr(
                fcx.ccx.tcx,
                {
                    ty: ty::mk_mach_uint(fcx.ccx.tcx, ast::ty_u8),
                    mutbl: ast::m_imm
                })
        };
      }
      ast::def_fn(id, _) | ast::def_const(id) |
      ast::def_variant(_, id) | ast::def_class(id)
         { ret ty::lookup_item_type(fcx.ccx.tcx, id); }
      ast::def_binding(nid) {
        assert (fcx.locals.contains_key(nid));
        let typ = ty::mk_var(fcx.ccx.tcx, lookup_local(fcx, sp, nid));
        ret {bounds: @[], ty: typ};
      }
      ast::def_ty(_) | ast::def_prim_ty(_) {
        fcx.ccx.tcx.sess.span_fatal(sp, "expected value but found type");
      }
      ast::def_upvar(_, inner, _) {
        ret ty_param_bounds_and_ty_for_def(fcx, sp, *inner);
      }
      _ {
        // FIXME: handle other names.
        fcx.ccx.tcx.sess.unimpl("definition variant");
      }
    }
}

// Instantiates the given path, which must refer to an item with the given
// number of type parameters and type.
fn instantiate_path(fcx: @fn_ctxt, pth: @ast::path,
                    tpt: ty_param_bounds_and_ty, sp: span,
                    id: ast::node_id) {
    let ty_param_count = vec::len(*tpt.bounds);
    let ty_substs_len = vec::len(pth.node.types);
    if ty_substs_len > 0u {
        if ty_param_count == 0u {
            fcx.ccx.tcx.sess.span_fatal
                (sp, "this item does not take type parameters");
        } else if ty_substs_len > ty_param_count {
            fcx.ccx.tcx.sess.span_fatal
                (sp, "too many type parameters provided for this item");
        } else if ty_substs_len < ty_param_count {
            fcx.ccx.tcx.sess.span_fatal
                (sp, "not enough type parameters provided for this item");
        }
        if ty_param_count == 0u {
            fcx.ccx.tcx.sess.span_fatal(
                sp, "this item does not take type parameters");
        }
        let substs = vec::map(pth.node.types, {|aty|
            ast_ty_to_ty_crate(fcx.ccx, aty)
        });
        fcx.write_ty_substs(id, tpt.ty, substs);
    } else if ty_param_count > 0u {
        let vars = vec::from_fn(ty_param_count, {|_i| next_ty_var(fcx)});
        fcx.write_ty_substs(id, tpt.ty, vars);
    } else {
        fcx.write_ty(id, tpt.ty);
    }
}

// Type tests
fn structurally_resolved_type(fcx: @fn_ctxt, sp: span, tp: ty::t) -> ty::t {
    alt infer::resolve_type_structure(fcx.infcx, tp) {
      // note: the bot type doesn't count as resolved; it's what we use when
      // there is no information about a variable.
      result::ok(t_s) if !ty::type_is_bot(t_s) { ret t_s; }
      _ {
        fcx.ccx.tcx.sess.span_fatal
            (sp, "the type of this value must be known in this context");
      }
    }
}


// Returns the one-level-deep structure of the given type.
fn structure_of(fcx: @fn_ctxt, sp: span, typ: ty::t) -> ty::sty {
    ty::get(structurally_resolved_type(fcx, sp, typ)).struct
}

// Returns the one-level-deep structure of the given type or none if it
// is not known yet.
fn structure_of_maybe(fcx: @fn_ctxt, _sp: span, typ: ty::t) ->
   option<ty::sty> {
    let r = infer::resolve_type_structure(fcx.infcx, typ);
    alt r {
      result::ok(typ_s) { some(ty::get(typ_s).struct) }
      result::err(_) { none }
    }
}

fn type_is_integral(fcx: @fn_ctxt, sp: span, typ: ty::t) -> bool {
    let typ_s = structurally_resolved_type(fcx, sp, typ);
    ret ty::type_is_integral(typ_s);
}

fn type_is_scalar(fcx: @fn_ctxt, sp: span, typ: ty::t) -> bool {
    let typ_s = structurally_resolved_type(fcx, sp, typ);
    ret ty::type_is_scalar(typ_s);
}

fn type_is_c_like_enum(fcx: @fn_ctxt, sp: span, typ: ty::t) -> bool {
    let typ_s = structurally_resolved_type(fcx, sp, typ);
    ret ty::type_is_c_like_enum(fcx.ccx.tcx, typ_s);
}

enum mode { m_collect, m_check, m_check_tyvar(@fn_ctxt), }

fn ast_ty_vstore_to_vstore(tcx: ty::ctxt, ty: @ast::ty,
                           v: ast::vstore) -> ty::vstore {
    alt v {
      ast::vstore_fixed(none) {
        tcx.sess.span_bug(ty.span,
                          "implied fixed length in ast_ty_vstore_to_vstore");
      }
      ast::vstore_fixed(some(u)) {
        ty::vstore_fixed(u)
      }
      ast::vstore_uniq { ty::vstore_uniq }
      ast::vstore_box { ty::vstore_box }
      ast::vstore_slice(r) {
        ty::vstore_slice(tcx.region_map.ast_type_to_region.get(ty.id))
      }
    }
}

fn ast_expr_vstore_to_vstore(fcx: @fn_ctxt, e: @ast::expr, n: uint,
                             v: ast::vstore) -> ty::vstore {
    alt v {
      ast::vstore_fixed(none) { ty::vstore_fixed(n) }
      ast::vstore_fixed(some(u)) {
        if n != u {
            let s = #fmt("fixed-size sequence mismatch: %u vs. %u",u, n);
            fcx.ccx.tcx.sess.span_err(e.span,s);
        }
        ty::vstore_fixed(u)
      }
      ast::vstore_uniq { ty::vstore_uniq }
      ast::vstore_box { ty::vstore_box }
      ast::vstore_slice(r) {
        ty::vstore_slice(region_of(fcx, e))
      }
    }
}

// Parses the programmer's textual representation of a type into our
// internal notion of a type. `getter` is a function that returns the type
// corresponding to a definition ID:
fn ast_ty_to_ty(tcx: ty::ctxt, mode: mode, &&ast_ty: @ast::ty) -> ty::t {
    fn getter(tcx: ty::ctxt, mode: mode, id: ast::def_id)
            -> ty::ty_param_bounds_and_ty {

        alt mode {
          m_check | m_check_tyvar(_) { ty::lookup_item_type(tcx, id) }
          m_collect {
            if id.crate != ast::local_crate { csearch::get_type(tcx, id) }
            else {
                alt tcx.items.find(id.node) {
                  some(ast_map::node_item(item, _)) {
                    ty_of_item(tcx, mode, item)
                  }
                  some(ast_map::node_native_item(native_item, _, _)) {
                    ty_of_native_item(tcx, mode, native_item)
                  }
                  _ {
                    tcx.sess.bug("unexpected sort of item in ast_ty_to_ty");
                  }
                }
            }
          }
        }
    }
    fn ast_mt_to_mt(tcx: ty::ctxt, mode: mode, mt: ast::mt) -> ty::mt {
        ret {ty: do_ast_ty_to_ty(tcx, mode, mt.ty), mutbl: mt.mutbl};
    }
    fn instantiate(tcx: ty::ctxt, sp: span, mode: mode, id: ast::def_id,
                   path_id: ast::node_id, args: [@ast::ty]) -> ty::t {
        let ty_param_bounds_and_ty = getter(tcx, mode, id);
        if vec::len(*ty_param_bounds_and_ty.bounds) == 0u {
            ret ty_param_bounds_and_ty.ty;
        }

        // The typedef is type-parametric. Do the type substitution.
        let mut param_bindings: [ty::t] = [];
        if vec::len(args) != vec::len(*ty_param_bounds_and_ty.bounds) {
            tcx.sess.span_fatal(sp, "wrong number of type arguments for a \
                                     polymorphic type");
        }
        for args.each {|ast_ty|
            param_bindings += [do_ast_ty_to_ty(tcx, mode, ast_ty)];
        }
        #debug("substituting(%s into %s)",
               str::concat(vec::map(param_bindings, {|t| ty_to_str(tcx, t)})),
               ty_to_str(tcx, ty_param_bounds_and_ty.ty));
        let typ =
            ty::substitute_type_params(tcx, param_bindings,
                                       ty_param_bounds_and_ty.ty);
        write_substs_to_tcx(tcx, path_id, param_bindings);
        ret typ;
    }
    fn do_ast_ty_to_ty(tcx: ty::ctxt, mode: mode, &&ast_ty: @ast::ty)
            -> ty::t {

        alt tcx.ast_ty_to_ty_cache.find(ast_ty) {
          some(ty::atttce_resolved(ty)) { ret ty; }
          some(ty::atttce_unresolved) {
            tcx.sess.span_fatal(ast_ty.span, "illegal recursive type. \
                                              insert a enum in the cycle, \
                                              if this is desired)");
          }
          none { /* go on */ }
        }

        tcx.ast_ty_to_ty_cache.insert(ast_ty, ty::atttce_unresolved);
        let typ = alt ast_ty.node {
          ast::ty_nil { ty::mk_nil(tcx) }
          ast::ty_bot { ty::mk_bot(tcx) }
          ast::ty_box(mt) {
            ty::mk_box(tcx, ast_mt_to_mt(tcx, mode, mt))
          }
          ast::ty_uniq(mt) {
            ty::mk_uniq(tcx, ast_mt_to_mt(tcx, mode, mt))
          }
          ast::ty_vec(mt) {
            ty::mk_vec(tcx, ast_mt_to_mt(tcx, mode, mt))
          }
          ast::ty_ptr(mt) {
            ty::mk_ptr(tcx, ast_mt_to_mt(tcx, mode, mt))
          }
          ast::ty_rptr(region, mt) {
            let region = tcx.region_map.ast_type_to_region.get(ast_ty.id);
            ty::mk_rptr(tcx, region, ast_mt_to_mt(tcx, mode, mt))
          }
          ast::ty_tup(fields) {
            let flds = vec::map(fields, bind do_ast_ty_to_ty(tcx, mode, _));
            ty::mk_tup(tcx, flds)
          }
          ast::ty_rec(fields) {
            let mut flds: [field] = [];
            for fields.each {|f|
                let tm = ast_mt_to_mt(tcx, mode, f.node.mt);
                flds += [{ident: f.node.ident, mt: tm}];
            }
            ty::mk_rec(tcx, flds)
          }
          ast::ty_fn(proto, decl) {
            ty::mk_fn(tcx, ty_of_fn_decl(tcx, mode, proto, decl))
          }
          ast::ty_path(path, id) {
            let a_def = alt tcx.def_map.find(id) {
              none { tcx.sess.span_fatal(ast_ty.span, #fmt("unbound path %s",
                                                       path_to_str(path))); }
              some(d) { d }};
            alt a_def {
              ast::def_ty(did) | ast::def_class(did) {
                instantiate(tcx, ast_ty.span, mode, did,
                            id, path.node.types)
              }
              ast::def_prim_ty(nty) {
                alt nty {
                  ast::ty_bool { ty::mk_bool(tcx) }
                  ast::ty_int(it) { ty::mk_mach_int(tcx, it) }
                  ast::ty_uint(uit) { ty::mk_mach_uint(tcx, uit) }
                  ast::ty_float(ft) { ty::mk_mach_float(tcx, ft) }
                  ast::ty_str { ty::mk_str(tcx) }
                }
              }
              ast::def_ty_param(id, n) {
                if vec::len(path.node.types) > 0u {
                    tcx.sess.span_err(ast_ty.span, "provided type parameters \
                                                    to a type parameter");
                }
                ty::mk_param(tcx, n, id)
              }
              ast::def_self(self_id) {
                alt check tcx.items.get(self_id) {
                  ast_map::node_item(@{node: ast::item_iface(tps, _), _}, _) {
                    if vec::len(tps) != vec::len(path.node.types) {
                        tcx.sess.span_err(ast_ty.span, "incorrect number of \
                                                        type parameters to \
                                                        self type");
                    }
                    ty::mk_self(tcx, vec::map(path.node.types, {|ast_ty|
                        do_ast_ty_to_ty(tcx, mode, ast_ty)
                    }))
                  }
                }
              }
             _ {
                tcx.sess.span_fatal(ast_ty.span,
                                    "found type name used as a variable");
              }
            }
          }
          ast::ty_vstore(t, vst) {
            let vst = ast_ty_vstore_to_vstore(tcx, ast_ty, vst);
            let ty = alt ty::get(do_ast_ty_to_ty(tcx, mode, t)).struct {
              ty::ty_vec(mt) { ty::mk_evec(tcx, mt, vst) }
              ty::ty_str { ty::mk_estr(tcx, vst) }
              _ {
                tcx.sess.span_fatal(ast_ty.span,
                                    "found sequence storage modifier \
                                     on non-sequence type");
              }
            };
            fixup_regions_to_block(tcx, ty, ast_ty)
          }
          ast::ty_constr(t, cs) {
            let mut out_cs = [];
            for cs.each {|constr|
                out_cs += [ty::ast_constr_to_constr(tcx, constr)];
            }
            ty::mk_constr(tcx, do_ast_ty_to_ty(tcx, mode, t), out_cs)
          }
          ast::ty_infer {
            alt mode {
              m_check_tyvar(fcx) { ret next_ty_var(fcx); }
              _ { tcx.sess.span_bug(ast_ty.span,
                                    "found `ty_infer` in unexpected place"); }
            }
          }
          ast::ty_mac(_) {
              tcx.sess.span_bug(ast_ty.span,
                                    "found `ty_mac` in unexpected place");
          }
        };

        tcx.ast_ty_to_ty_cache.insert(ast_ty, ty::atttce_resolved(typ));
        ret typ;
    }

    ret do_ast_ty_to_ty(tcx, mode, ast_ty);
}

fn ty_of_item(tcx: ty::ctxt, mode: mode, it: @ast::item)
    -> ty::ty_param_bounds_and_ty {
    let def_id = local_def(it.id);
    alt tcx.tcache.find(def_id) {
      some(tpt) { ret tpt; }
      _ {}
    }
    alt it.node {
      ast::item_const(t, _) {
        let typ = ast_ty_to_ty(tcx, mode, t);
        let tpt = {bounds: @[], ty: typ};
        tcx.tcache.insert(local_def(it.id), tpt);
        ret tpt;
      }
      ast::item_fn(decl, tps, _) {
        ret ty_of_fn(tcx, mode, decl, tps, local_def(it.id));
      }
      ast::item_ty(t, tps) {
        alt tcx.tcache.find(local_def(it.id)) {
          some(tpt) { ret tpt; }
          none { }
        }
        // Tell ast_ty_to_ty() that we want to perform a recursive
        // call to resolve any named types.
        let tpt = {
            let t0 = ast_ty_to_ty(tcx, mode, t);
            let t1 = {
                // Do not associate a def id with a named, parameterized type
                // like "foo<X>".  This is because otherwise ty_to_str will
                // print the name as merely "foo", as it has no way to
                // reconstruct the value of X.
                if vec::is_empty(tps) {
                    ty::mk_with_id(tcx, t0, def_id)
                } else {
                    t0
                }
            };
            {bounds: ty_param_bounds(tcx, mode, tps), ty: t1}
        };
        tcx.tcache.insert(local_def(it.id), tpt);
        ret tpt;
      }
      ast::item_res(decl, tps, _, _, _) {
        let {bounds, params} = mk_ty_params(tcx, tps);
        let t_arg = ty_of_arg(tcx, mode, decl.inputs[0]);
        let t = ty::mk_res(tcx, local_def(it.id), t_arg.ty, params);
        let t_res = {bounds: bounds, ty: t};
        tcx.tcache.insert(local_def(it.id), t_res);
        ret t_res;
      }
      ast::item_enum(_, tps) {
        // Create a new generic polytype.
        let {bounds, params} = mk_ty_params(tcx, tps);
        let t = ty::mk_enum(tcx, local_def(it.id), params);
        let tpt = {bounds: bounds, ty: t};
        tcx.tcache.insert(local_def(it.id), tpt);
        ret tpt;
      }
      ast::item_iface(tps, ms) {
        let {bounds, params} = mk_ty_params(tcx, tps);
        let t = ty::mk_iface(tcx, local_def(it.id), params);
        let tpt = {bounds: bounds, ty: t};
        tcx.tcache.insert(local_def(it.id), tpt);
        ret tpt;
      }
      ast::item_class(tps,_,_) {
          let {bounds,params} = mk_ty_params(tcx, tps);
          let t = ty::mk_class(tcx, local_def(it.id), params);
          let tpt = {bounds: bounds, ty: t};
          tcx.tcache.insert(local_def(it.id), tpt);
          ret tpt;
      }
      ast::item_impl(_, _, _, _) | ast::item_mod(_) |
      ast::item_native_mod(_) { fail; }
    }
}
fn ty_of_native_item(tcx: ty::ctxt, mode: mode, it: @ast::native_item)
    -> ty::ty_param_bounds_and_ty {
    alt it.node {
      ast::native_item_fn(fn_decl, params) {
        ret ty_of_native_fn_decl(tcx, mode, fn_decl, params,
                                 local_def(it.id));
      }
    }
}

type next_region_param_id = { mut id: uint };

// fn replace_inferred(tcx: ty::ctxt, ty: ty::t,
//                     rop: fn() -> ty::region) -> ty::t {
//     let tb = ty::get(ty);
//     if !tb.has_rptrs { ret ty; }
//     alt tb.struct {
//       // Replace re_inferred that are nested underneath
//       // a top-level region ptr with the region ptr itself.
//       ty::ty_rptr(ty::re_inferred, ty1) {
//         let r = rop();
//         ty::mk_rptr(r, replace_inferred(tcx, ty1) {|| r })
//       }
//       ty::ty_rptr(r, ty1) {
//         ty::mk_rptr(r, replace_inferred(tcx, ty1) {|| r })
//       }
//
//       // Do not replace inferred ptrs that appear within fn(...) types, as
//       // those are bound by the function.
//       ty::ty_fn(f) {
//         ty
//       }
//
//       // Otherwise just map.
//       sty {
//         ty::fold_sty_to_ty(tcx, sty) {|t|
//           replace_inferred(tcx, with_r, t) }
//       }
//     }
// }

fn replace_default_region(tcx: ty::ctxt,
                          with_region: ty::region,
                          ty: ty::t) -> ty::t {
    let mut last_region = with_region;
    ret ty::fold_region(tcx, ty) {|region, under_rptr|
        if !under_rptr {
            last_region = alt region {
              ty::re_default { with_region }
              _ { region }
            }
        }
        last_region
   };
}

fn default_region_to_bound_anon(tcx: ty::ctxt, ty: ty::t) -> ty::t {
    replace_default_region(tcx, ty::re_bound(ty::br_anon), ty)
}

fn default_region_to_bound_self(tcx: ty::ctxt, ty: ty::t) -> ty::t {
    replace_default_region(tcx, ty::re_bound(ty::br_self), ty)
}

fn fixup_regions_to_block(tcx: ty::ctxt, ty: ty::t, ast_ty: @ast::ty)
        -> ty::t {
    let region = tcx.region_map.ast_type_to_inferred_region.get(ast_ty.id);
    replace_default_region(tcx, region, ty)
}

fn replace_bound_regions_with_free_regions(
    tcx: ty::ctxt,
    id: ast::node_id,
    ty: ty::t) -> ty::t {

    ty::fold_region(tcx, ty) {|region, _under_rptr|
        alt region {
          ty::re_bound(br) { ty::re_free(id, br) }
          _ { region }
        }
    }
}

fn ty_of_arg(tcx: ty::ctxt, mode: mode, a: ast::arg) -> ty::arg {
    fn arg_mode(tcx: ty::ctxt, m: ast::mode, ty: ty::t) -> ast::mode {
        alt m {
          ast::infer(_) {
            alt ty::get(ty).struct {
              // If the type is not specified, then this must be a fn expr.
              // Leave the mode as infer(_), it will get inferred based
              // on constraints elsewhere.
              ty::ty_var(_) { m }

              // If the type is known, then use the default for that type.
              // Here we unify m and the default.  This should update the
              // tables in tcx but should never fail, because nothing else
              // will have been unified with m yet:
              _ {
                let m1 = ast::expl(ty::default_arg_mode_for_ty(ty));
                result::get(ty::unify_mode(tcx, m, m1))
              }
            }
          }
          ast::expl(_) { m }
        }
    }

    let ty = ast_ty_to_ty(tcx, mode, a.ty);
    let mode = arg_mode(tcx, a.mode, ty);
    {mode: mode, ty: ty}
}
fn ty_of_fn_decl(tcx: ty::ctxt,
                 mode: mode,
                 proto: ast::proto,
                 decl: ast::fn_decl) -> ty::fn_ty {
    let input_tys = vec::map(decl.inputs) {|a|
        let arg_ty = ty_of_arg(tcx, mode, a);
        {ty: default_region_to_bound_anon(tcx, arg_ty.ty)
         with arg_ty}
    };

    let output_ty = {
        let t = ast_ty_to_ty(tcx, mode, decl.output);
        default_region_to_bound_anon(tcx, t)
    };

    let out_constrs = vec::map(decl.constraints) {|constr|
        ty::ast_constr_to_constr(tcx, constr)
    };
    {proto: proto, inputs: input_tys,
     output: output_ty, ret_style: decl.cf, constraints: out_constrs}
}
fn ty_of_fn(tcx: ty::ctxt, mode: mode, decl: ast::fn_decl,
            ty_params: [ast::ty_param], def_id: ast::def_id)
    -> ty::ty_param_bounds_and_ty {
    let bounds = ty_param_bounds(tcx, mode, ty_params);
    let tofd = ty_of_fn_decl(tcx, mode, ast::proto_bare, decl);
    let tpt = {bounds: bounds, ty: ty::mk_fn(tcx, tofd)};
    tcx.tcache.insert(def_id, tpt);
    ret tpt;
}
fn ty_of_native_fn_decl(tcx: ty::ctxt, mode: mode, decl: ast::fn_decl,
                        ty_params: [ast::ty_param], def_id: ast::def_id)
    -> ty::ty_param_bounds_and_ty {
    let bounds = ty_param_bounds(tcx, mode, ty_params);
    let input_tys = vec::map(decl.inputs) {|a|
        ty_of_arg(tcx, mode, a)
    };
    let output_ty = ast_ty_to_ty(tcx, mode, decl.output);

    let t_fn = ty::mk_fn(tcx, {proto: ast::proto_bare,
                               inputs: input_tys,
                               output: output_ty,
                               ret_style: ast::return_val,
                               constraints: []});
    let tpt = {bounds: bounds, ty: t_fn};
    tcx.tcache.insert(def_id, tpt);
    ret tpt;
}
fn ty_param_bounds(tcx: ty::ctxt, mode: mode, params: [ast::ty_param])
    -> @[ty::param_bounds] {
    let mut result = [];
    for params.each {|param|
        result += [alt tcx.ty_param_bounds.find(param.id) {
          some(bs) { bs }
          none {
            let mut bounds = [];
            for vec::each(*param.bounds) {|b|
                bounds += [alt b {
                  ast::bound_send { ty::bound_send }
                  ast::bound_copy { ty::bound_copy }
                  ast::bound_iface(t) {
                    let ity = ast_ty_to_ty(tcx, mode, t);
                    alt ty::get(ity).struct {
                      ty::ty_iface(_, _) {}
                      _ {
                        tcx.sess.span_fatal(
                            t.span, "type parameter bounds must be \
                                     interface types");
                      }
                    }
                    ty::bound_iface(ity)
                  }
                }];
            }
            let boxed = @bounds;
            tcx.ty_param_bounds.insert(param.id, boxed);
            boxed
          }
        }];
    }
    @result
}
fn ty_of_method(tcx: ty::ctxt, mode: mode, m: @ast::method) -> ty::method {
    {ident: m.ident, tps: ty_param_bounds(tcx, mode, m.tps),
     fty: ty_of_fn_decl(tcx, mode, ast::proto_bare, m.decl),
     purity: m.decl.purity, privacy: m.privacy}
}
fn ty_of_ty_method(tcx: ty::ctxt, mode: mode, m: ast::ty_method)
    -> ty::method {
    {ident: m.ident, tps: ty_param_bounds(tcx, mode, m.tps),
     fty: ty_of_fn_decl(tcx, mode, ast::proto_bare, m.decl),
    // assume public, because this is only invoked on iface methods
     purity: m.decl.purity, privacy: ast::pub}
}

// A convenience function to use a crate_ctxt to resolve names for
// ast_ty_to_ty.
fn ast_ty_to_ty_crate(ccx: @crate_ctxt, &&ast_ty: @ast::ty) -> ty::t {
    ret ast_ty_to_ty(ccx.tcx, m_check, ast_ty);
}

// A wrapper around ast_ty_to_ty_crate that handles ty_infer.
fn ast_ty_to_ty_crate_infer(ccx: @crate_ctxt, &&ast_ty: @ast::ty) ->
   option<ty::t> {
    alt ast_ty.node {
      ast::ty_infer { none }
      _ { some(ast_ty_to_ty_crate(ccx, ast_ty)) }
    }
}


// Functions that write types into the node type table
fn write_ty_to_tcx(tcx: ty::ctxt, node_id: ast::node_id, ty: ty::t) {
    #debug["write_ty_to_tcx(%d, %s)", node_id, ty_to_str(tcx, ty)];
    smallintmap::insert(*tcx.node_types, node_id as uint, ty);
}
fn write_substs_to_tcx(tcx: ty::ctxt, node_id: ast::node_id,
                       +substs: [ty::t]) {
    tcx.node_type_substs.insert(node_id, substs);
}
fn write_ty_substs_to_tcx(tcx: ty::ctxt, node_id: ast::node_id, ty: ty::t,
                   +substs: [ty::t]) {
    let ty = if ty::type_has_params(ty) {
        ty::substitute_type_params(tcx, substs, ty)
    } else { ty };
    write_ty_to_tcx(tcx, node_id, ty);
    write_substs_to_tcx(tcx, node_id, substs);
}

impl methods for @fn_ctxt {
    fn tcx() -> ty::ctxt { self.ccx.tcx }
    fn tag() -> str { #fmt["%x", ptr::addr_of(*self) as uint] }
    fn ty_to_str(t: ty::t) -> str {
        ty_to_str(self.ccx.tcx, resolve_type_vars_if_possible(self, t))
    }
    fn write_ty(node_id: ast::node_id, ty: ty::t) {
        #debug["write_ty(%d, %s) in fcx %s",
               node_id, ty_to_str(self.tcx(), ty), self.tag()];
        self.node_types.insert(node_id as uint, ty);
    }
    fn write_substs(node_id: ast::node_id, +substs: [ty::t]) {
        self.node_type_substs.insert(node_id, substs);
    }
    fn write_ty_substs(node_id: ast::node_id, ty: ty::t, +substs: [ty::t]) {
        if substs.len() == 0u {
            self.write_ty(node_id, ty)
        } else {
            let ty = ty::substitute_type_params(self.tcx(), substs, ty);
            self.write_ty(node_id, ty);
            self.write_substs(node_id, substs);
        }
    }
    fn write_nil(node_id: ast::node_id) {
        self.write_ty(node_id, ty::mk_nil(self.tcx()));
    }
    fn write_bot(node_id: ast::node_id) {
        self.write_ty(node_id, ty::mk_bot(self.tcx()));
    }

    fn expr_ty(ex: @ast::expr) -> ty::t {
        alt self.node_types.find(ex.id as uint) {
          some(t) { t }
          none {
            self.tcx().sess.bug(#fmt["no type for expr %d (%s) in fcx %s",
                                     ex.id, expr_to_str(ex), self.tag()]);
          }
        }
    }
    fn node_ty(id: ast::node_id) -> ty::t {
        alt self.node_types.find(id as uint) {
          some(t) { t }
          none {
            self.tcx().sess.bug(
                #fmt["no type for node %d: %s in fcx %s",
                     id, ast_map::node_id_to_str(self.tcx().items, id),
                     self.tag()]);
          }
        }
    }
    fn node_ty_substs(id: ast::node_id) -> [ty::t] {
        alt self.node_type_substs.find(id) {
          some(ts) { ts }
          none {
            self.tcx().sess.bug(
                #fmt["no type substs for node %d: %s in fcx %s",
                     id, ast_map::node_id_to_str(self.tcx().items, id),
                     self.tag()]);
          }
        }
    }
    fn opt_node_ty_substs(id: ast::node_id) -> option<[ty::t]> {
        self.node_type_substs.find(id)
    }
}

fn mk_ty_params(tcx: ty::ctxt, atps: [ast::ty_param])
    -> {bounds: @[ty::param_bounds], params: [ty::t]} {
    let mut i = 0u;
    let bounds = ty_param_bounds(tcx, m_collect, atps);
    {bounds: bounds,
     params: vec::map(atps, {|atp|
         let t = ty::mk_param(tcx, i, local_def(atp.id));
         i += 1u;
         t
     })}
}

fn compare_impl_method(tcx: ty::ctxt, sp: span, impl_m: ty::method,
                       impl_tps: uint, if_m: ty::method, substs: [ty::t],
                       self_ty: ty::t) -> ty::t {
    if impl_m.tps != if_m.tps {
        tcx.sess.span_err(sp, "method `" + if_m.ident +
                          "` has an incompatible set of type parameters");
        ty::mk_fn(tcx, impl_m.fty)
    } else if vec::len(impl_m.fty.inputs) != vec::len(if_m.fty.inputs) {
        tcx.sess.span_err(sp,#fmt["method `%s` has %u parameters \
                                   but the iface has %u",
                                  if_m.ident,
                                  vec::len(impl_m.fty.inputs),
                                  vec::len(if_m.fty.inputs)]);
        ty::mk_fn(tcx, impl_m.fty)
    } else {
        let auto_modes = vec::map2(impl_m.fty.inputs, if_m.fty.inputs, {|i, f|
            alt ty::get(f.ty).struct {
              ty::ty_param(_, _) | ty::ty_self(_)
              if alt i.mode { ast::infer(_) { true } _ { false } } {
                {mode: ast::expl(ast::by_ref) with i}
              }
              _ { i }
            }
        });
        let impl_fty = ty::mk_fn(tcx, {inputs: auto_modes with impl_m.fty});
        // Add dummy substs for the parameters of the impl method
        let substs = substs + vec::from_fn(vec::len(*if_m.tps), {|i|
            ty::mk_param(tcx, i + impl_tps, {crate: 0, node: 0})
        });
        let mut if_fty = ty::mk_fn(tcx, if_m.fty);
        if_fty = ty::substitute_type_params(tcx, substs, if_fty);
        if ty::type_has_vars(if_fty) {
            if_fty = fixup_self_in_method_ty(tcx, if_fty, substs,
                                             self_full(self_ty, impl_tps));
        }
        require_same_types(
            tcx, sp, impl_fty, if_fty,
            {|| "method `" + if_m.ident +
                 "` has an incompatible type"});
        ret impl_fty;
    }
}

enum self_subst { self_param(ty::t, @fn_ctxt, span), self_full(ty::t, uint) }

// Mangles an iface method ty to make its self type conform to the self type
// of a specific impl or bounded type parameter. This is rather involved
// because the type parameters of ifaces and impls are not required to line up
// (an impl can have less or more parameters than the iface it implements), so
// some mangling of the substituted types is required.
fn fixup_self_in_method_ty(cx: ty::ctxt, mty: ty::t, m_substs: [ty::t],
                           self: self_subst) -> ty::t {
    if ty::type_has_vars(mty) {
        ty::fold_ty(cx, mty) {|t|
            alt ty::get(t).struct {
              ty::ty_self(tps) {
                if vec::len(tps) > 0u {
                    // Move the substs into the type param system of the
                    // context.
                    let mut substs = vec::map(tps, {|t|
                        let f = fixup_self_in_method_ty(cx, t, m_substs,
                                                        self);
                        ty::substitute_type_params(cx, m_substs, f)
                    });
                    alt self {
                      self_param(t, fcx, sp) {
                        // Simply ensure that the type parameters for the self
                        // type match the context.
                        vec::iter2(substs, m_substs) {|s, ms|
                            demand::simple(fcx, sp, s, ms);
                        }
                        t
                      }
                      self_full(selfty, impl_n_tps) {
                        // Add extra substs for impl type parameters.
                        while vec::len(substs) < impl_n_tps {
                            substs += [ty::mk_param(cx, vec::len(substs),
                                                    {crate: 0, node: 0})];
                        }
                        // And for method type parameters.
                        let method_n_tps =
                            (vec::len(m_substs) - vec::len(tps)) as int;
                        if method_n_tps > 0 {
                            substs += vec::tailn(m_substs, vec::len(m_substs)
                                                  - (method_n_tps as uint));
                        }
                        // And then instantiate the self type using all those.
                        ty::substitute_type_params(cx, substs, selfty)
                      }
                    }
                } else {
                    alt self { self_param(t, _, _) | self_full(t, _) { t } }
                }
              }
              _ { t }
            }
        }
    } else { mty }
}

// Replaces all occurrences of the `self` region with `with_region`.  Note
// that we descend into `fn()` types here, because `fn()` does not bind the
// `self` region.
fn replace_self_region(tcx: ty::ctxt, with_region: ty::region,
                       ty: ty::t) -> ty::t {

   ty::fold_region(tcx, ty) {|r, _under_rptr|
       alt r {
           ty::re_bound(re_self) { with_region }
           _ { r }
       }
   }
}

fn instantiate_bound_regions(tcx: ty::ctxt, region: ty::region, &&ty: ty::t)
        -> ty::t {
    ty::fold_region(tcx, ty) {|r, _under_rptr|
        alt r {
          ty::re_bound(_) { region }
          _ { r }
        }
    }
}

// Item collection - a pair of bootstrap passes:
//
// (1) Collect the IDs of all type items (typedefs) and store them in a table.
//
// (2) Translate the AST fragments that describe types to determine a type for
//     each item. When we encounter a named type, we consult the table built
//     in pass 1 to find its item, and recursively translate it.
//
// We then annotate the AST with the resulting types and return the annotated
// AST, along with a table mapping item IDs to their types.
mod collect {
    fn get_enum_variant_types(tcx: ty::ctxt, enum_ty: ty::t,
                              variants: [ast::variant],
                              ty_params: [ast::ty_param]) {
        // Create a set of parameter types shared among all the variants.
        for variants.each {|variant|
            // Nullary enum constructors get turned into constants; n-ary enum
            // constructors get turned into functions.
            let result_ty = if vec::len(variant.node.args) == 0u {
                enum_ty
            } else {
                // As above, tell ast_ty_to_ty() that trans_ty_item_to_ty()
                // should be called to resolve named types.
                let mut args: [arg] = [];
                for variant.node.args.each {|va|
                    let arg_ty = {
                        // NDM We need BOUNDS here.  It should be that this
                        // yields a type like "foo &anon".  Basically every
                        // nominal type is going to require a region bound.
                        let arg_ty = ast_ty_to_ty(tcx, m_collect, va.ty);
                        default_region_to_bound_anon(tcx, arg_ty)
                    };

                    args += [{mode: ast::expl(ast::by_copy), ty: arg_ty}];
                }
                // FIXME: this will be different for constrained types
                ty::mk_fn(tcx,
                          {proto: ast::proto_box,
                           inputs: args, output: enum_ty,
                           ret_style: ast::return_val, constraints: []})
            };
            let tpt = {bounds: ty_param_bounds(tcx, m_collect, ty_params),
                       ty: result_ty};
            tcx.tcache.insert(local_def(variant.node.id), tpt);
            write_ty_to_tcx(tcx, variant.node.id, result_ty);
        }
    }
    fn ensure_iface_methods(tcx: ty::ctxt, id: ast::node_id) {
        fn store_methods<T>(tcx: ty::ctxt, id: ast::node_id,
                            stuff: [T], f: fn@(T) -> ty::method) {
            ty::store_iface_methods(tcx, id, @vec::map(stuff, f));
        }

        alt check tcx.items.get(id) {
          ast_map::node_item(@{node: ast::item_iface(_, ms), _}, _) {
              store_methods::<ast::ty_method>(tcx, id, ms, {|m|
                          ty_of_ty_method(tcx, m_collect, m)});
          }
          ast_map::node_item(@{node: ast::item_class(_,its,_), _}, _) {
              let (_,ms) = split_class_items(its);
              // All methods need to be stored, since lookup_method
              // relies on the same method cache for self-calls
              store_methods::<@ast::method>(tcx, id, ms, {|m|
                          ty_of_method(tcx, m_collect, m)});
          }
        }
    }
    fn convert_class_item(tcx: ty::ctxt, v: ast_util::ivar) {
        /* we want to do something here, b/c within the
         scope of the class, it's ok to refer to fields &
        methods unqualified */

        /* they have these types *within the scope* of the
         class. outside the class, it's done with expr_field */
        let tt = ast_ty_to_ty(tcx, m_collect, v.ty);
        #debug("convert_class_item: %s %?", v.ident, v.id);
        write_ty_to_tcx(tcx, v.id, tt);
    }
    fn convert_methods(tcx: ty::ctxt, ms: [@ast::method],
        i_bounds: @[ty::param_bounds], maybe_self: option<ty::t>)
        -> [{mty: ty::method, id: ast::node_id, span: span}] {
        let mut my_methods = [];
        for ms.each {|m|
           alt maybe_self {
              some(selfty) {
                write_ty_to_tcx(tcx, m.self_id, selfty);
              }
              _ {}
           }
           let bounds = ty_param_bounds(tcx, m_collect, m.tps);
           let mty = ty_of_method(tcx, m_collect, m);
           my_methods += [{mty: mty, id: m.id, span: m.span}];
           let fty = ty::mk_fn(tcx, mty.fty);
           tcx.tcache.insert(local_def(m.id),
                             {bounds: @(*i_bounds + *bounds),
                                     ty: fty});
           write_ty_to_tcx(tcx, m.id, fty);
        }
        my_methods
    }
    fn convert(tcx: ty::ctxt, it: @ast::item) {
        alt it.node {
          // These don't define types.
          ast::item_mod(_) {}
          ast::item_native_mod(m) {
            if syntax::attr::native_abi(it.attrs) ==
               either::right(ast::native_abi_rust_intrinsic) {
                for m.items.each {|item| check_intrinsic_type(tcx, item); }
            }
          }
          ast::item_enum(variants, ty_params) {
            let tpt = ty_of_item(tcx, m_collect, it);
            write_ty_to_tcx(tcx, it.id, tpt.ty);
            get_enum_variant_types(tcx, tpt.ty, variants, ty_params);
          }
          ast::item_impl(tps, ifce, selfty, ms) {
            let i_bounds = ty_param_bounds(tcx, m_collect, tps);
            let selfty = ast_ty_to_ty(tcx, m_collect, selfty);
            write_ty_to_tcx(tcx, it.id, selfty);
            tcx.tcache.insert(local_def(it.id), {bounds: i_bounds,
                                                 ty: selfty});
            let my_methods = convert_methods(tcx, ms, i_bounds, some(selfty));
            alt ifce {
              some(t) {
                let iface_ty = ast_ty_to_ty(tcx, m_collect, t);
                alt ty::get(iface_ty).struct {
                  ty::ty_iface(did, tys) {
                    // Store the iface type in the type node
                    alt check t.node {
                      ast::ty_path(_, t_id) {
                        write_ty_to_tcx(tcx, t_id, iface_ty);
                      }
                    }
                    if did.crate == ast::local_crate {
                        ensure_iface_methods(tcx, did.node);
                    }
                    for vec::each(*ty::iface_methods(tcx, did)) {|if_m|
                        alt vec::find(my_methods,
                                      {|m| if_m.ident == m.mty.ident}) {
                          some({mty: m, id, span}) {
                            if m.purity != if_m.purity {
                                tcx.sess.span_err(
                                    span, "method `" + m.ident + "`'s purity \
                                           not match the iface method's \
                                           purity");
                            }
                            let mt = compare_impl_method(
                                tcx, span, m, vec::len(tps), if_m, tys,
                                selfty);
                            let old = tcx.tcache.get(local_def(id));
                            if old.ty != mt {
                                tcx.tcache.insert(local_def(id),
                                                     {bounds: old.bounds,
                                                     ty: mt});
                                write_ty_to_tcx(tcx, id, mt);
                            }
                          }
                          none {
                            tcx.sess.span_err(t.span, "missing method `" +
                                                 if_m.ident + "`");
                          }
                        }
                    }
                  }
                  _ {
                    tcx.sess.span_fatal(t.span, "can only implement \
                                                    interface types");
                  }
                }
              }
              _ {}
            }
          }
          ast::item_res(decl, tps, _, dtor_id, ctor_id) {
            let {bounds, params} = mk_ty_params(tcx, tps);
            let def_id = local_def(it.id);
            let t_arg = ty_of_arg(tcx, m_collect, decl.inputs[0]);
            let t_res = ty::mk_res(tcx, def_id, t_arg.ty, params);
            let t_ctor = ty::mk_fn(tcx, {
                proto: ast::proto_box,
                inputs: [{mode: ast::expl(ast::by_copy) with t_arg}],
                output: t_res,
                ret_style: ast::return_val, constraints: []
            });
            let t_dtor = ty::mk_fn(tcx, {
                proto: ast::proto_box,
                inputs: [t_arg], output: ty::mk_nil(tcx),
                ret_style: ast::return_val, constraints: []
            });
            write_ty_to_tcx(tcx, it.id, t_res);
            write_ty_to_tcx(tcx, ctor_id, t_ctor);
            tcx.tcache.insert(local_def(ctor_id),
                              {bounds: bounds, ty: t_ctor});
            tcx.tcache.insert(def_id, {bounds: bounds, ty: t_res});
            write_ty_to_tcx(tcx, dtor_id, t_dtor);
          }
          ast::item_iface(_, ms) {
            let tpt = ty_of_item(tcx, m_collect, it);
            write_ty_to_tcx(tcx, it.id, tpt.ty);
            ensure_iface_methods(tcx, it.id);
          }
          ast::item_class(tps, members, ctor) {
              // Write the class type
              let tpt = ty_of_item(tcx, m_collect, it);
              write_ty_to_tcx(tcx, it.id, tpt.ty);
              // Write the ctor type
              let t_ctor = ty::mk_fn(tcx,
                                     ty_of_fn_decl(tcx, m_collect,
                                             ast::proto_any, ctor.node.dec));
              write_ty_to_tcx(tcx, ctor.node.id, t_ctor);
              tcx.tcache.insert(local_def(ctor.node.id),
                                   {bounds: tpt.bounds, ty: t_ctor});
              ensure_iface_methods(tcx, it.id);
              /* FIXME: check for proper public/privateness */
              // Write the type of each of the members
              let (fields, methods) = split_class_items(members);
              for fields.each {|f|
                 convert_class_item(tcx, f);
              }
              // The selfty is just the class type
              let selfty = ty::mk_class(tcx, local_def(it.id),
                                        mk_ty_params(tcx, tps).params);
              // Need to convert all methods so we can check internal
              // references to private methods
              convert_methods(tcx, methods, @[], some(selfty));
          }
          _ {
            // This call populates the type cache with the converted type
            // of the item in passing. All we have to do here is to write
            // it into the node type table.
            let tpt = ty_of_item(tcx, m_collect, it);
            write_ty_to_tcx(tcx, it.id, tpt.ty);
          }
        }
    }
    fn convert_native(tcx: ty::ctxt, i: @ast::native_item) {
        // As above, this call populates the type table with the converted
        // type of the native item. We simply write it into the node type
        // table.
        let tpt = ty_of_native_item(tcx, m_collect, i);
        alt i.node {
          ast::native_item_fn(_, _) {
            write_ty_to_tcx(tcx, i.id, tpt.ty);
          }
        }
    }
    fn collect_item_types(tcx: ty::ctxt, crate: @ast::crate) {
        visit::visit_crate(*crate, (), visit::mk_simple_visitor(@{
            visit_item: bind convert(tcx, _),
            visit_native_item: bind convert_native(tcx, _)
            with *visit::default_simple_visitor()
        }));
    }
}


// Type unification
mod unify {
    fn unify(fcx: @fn_ctxt, expected: ty::t, actual: ty::t) ->
        result<(), ty::type_err> {
        ret infer::mk_subty(fcx.infcx, actual, expected);
    }
}


// FIXME This is almost a duplicate of ty::type_autoderef, with structure_of
// instead of ty::struct.
fn do_autoderef(fcx: @fn_ctxt, sp: span, t: ty::t) -> ty::t {
    let mut t1 = t;
    let mut enum_dids = [];
    loop {
        alt structure_of(fcx, sp, t1) {
          ty::ty_box(inner) | ty::ty_uniq(inner) | ty::ty_rptr(_, inner) {
            alt ty::get(t1).struct {
              ty::ty_var(v1) {
                ty::occurs_check(fcx.ccx.tcx, sp, v1,
                                 ty::mk_box(fcx.ccx.tcx, inner));
              }
              _ { }
            }
            t1 = inner.ty;
          }
          ty::ty_res(_, inner, tps) {
            t1 = ty::substitute_type_params(fcx.ccx.tcx, tps, inner);
          }
          ty::ty_enum(did, tps) {
            // Watch out for a type like `enum t = @t`.  Such a type would
            // otherwise infinitely auto-deref.  This is the only autoderef
            // loop that needs to be concerned with this, as an error will be
            // reported on the enum definition as well because the enum is not
            // instantiable.
            if vec::contains(enum_dids, did) {
                ret t1;
            }
            vec::push(enum_dids, did);

            let variants = ty::enum_variants(fcx.ccx.tcx, did);
            if vec::len(*variants) != 1u || vec::len(variants[0].args) != 1u {
                ret t1;
            }
            t1 =
                ty::substitute_type_params(fcx.ccx.tcx, tps,
                                           variants[0].args[0]);
          }
          _ { ret t1; }
        }
    };
}

fn resolve_type_vars_if_possible(fcx: @fn_ctxt, typ: ty::t) -> ty::t {
    alt infer::fixup_vars(fcx.infcx, typ) {
      result::ok(new_type) { ret new_type; }
      result::err(_) { ret typ; }
    }
}

// Demands - procedures that require that two types unify and emit an error
// message if they don't.
type ty_param_substs_and_ty = {substs: [ty::t], ty: ty::t};

fn require_same_types(
    tcx: ty::ctxt,
    span: span,
    t1: ty::t,
    t2: ty::t,
    msg: fn() -> str) -> bool {

    alt infer::compare_tys(tcx, t1, t2) {
      result::ok(()) { true }
      result::err(terr) {
        tcx.sess.span_err(
            span, msg() + ": " +
            ty::type_err_to_str(tcx, terr));
        false
      }
    }
}

mod demand {
    fn simple(fcx: @fn_ctxt, sp: span, expected: ty::t, actual: ty::t) ->
       ty::t {
        full(fcx, sp, unify::unify, expected, actual, []).ty
    }

    // n.b.: order of arguments is reversed.
    fn subty(fcx: @fn_ctxt, sp: span, actual: ty::t, expected: ty::t) {
        full(fcx, sp, unify::unify, expected, actual, []);
    }

    fn with_substs(fcx: @fn_ctxt, sp: span, expected: ty::t, actual: ty::t,
                   ty_param_substs_0: [ty::t]) -> ty_param_substs_and_ty {
        full(fcx, sp, unify::unify, expected, actual, ty_param_substs_0)
    }

    // Requires that the two types unify, and prints an error message if they
    // don't. Returns the unified type and the type parameter substitutions.
    fn full(fcx: @fn_ctxt,
            sp: span,
            unifier: fn@(@fn_ctxt, ty::t, ty::t)
                -> result<(), ty::type_err>,
            expected: ty::t,
            actual: ty::t,
            ty_param_substs_0: [ty::t]) ->
       ty_param_substs_and_ty {

        let mut ty_param_substs: [mut ty::t] = [mut];
        let mut ty_param_subst_var_ids: [ty_vid] = [];
        for ty_param_substs_0.each {|ty_param_subst|
            // Generate a type variable and unify it with the type parameter
            // substitution. We will then pull out these type variables.
            let t_0 = next_ty_var(fcx);
            ty_param_substs += [mut t_0];
            ty_param_subst_var_ids += [ty::ty_var_id(t_0)];
            simple(fcx, sp, ty_param_subst, t_0);
        }

        fn mk_result(fcx: @fn_ctxt, result_ty: ty::t,
                     ty_param_subst_var_ids: [ty_vid]) ->
           ty_param_substs_and_ty {
            let mut result_ty_param_substs: [ty::t] = [];
            for ty_param_subst_var_ids.each {|var_id|
                let tp_subst = ty::mk_var(fcx.ccx.tcx, var_id);
                result_ty_param_substs += [tp_subst];
            }
            ret {substs: result_ty_param_substs, ty: result_ty};
        }


        alt unifier(fcx, expected, actual) {
          result::ok(()) {
            ret mk_result(fcx, expected, ty_param_subst_var_ids);
          }
          result::err(err) {
            fcx.ccx.tcx.sess.span_err(sp,
                                      "mismatched types: expected `" +
                                      fcx.ty_to_str(expected) +
                                      "` but found `" +
                                      fcx.ty_to_str(actual) +
                                      "` (" +
                                      ty::type_err_to_str(
                                          fcx.ccx.tcx, err) +
                                      ")");
            ret mk_result(fcx, expected, ty_param_subst_var_ids);
          }
        }
    }
}


// Returns true if the two types unify and false if they don't.
fn are_compatible(fcx: @fn_ctxt, expected: ty::t, actual: ty::t) -> bool {
    alt unify::unify(fcx, expected, actual) {
      result::ok(_) { ret true; }
      result::err(_) { ret false; }
    }
}


// Returns the types of the arguments to a enum variant.
fn variant_arg_types(ccx: @crate_ctxt, _sp: span, vid: ast::def_id,
                     enum_ty_params: [ty::t]) -> [ty::t] {
    let mut result: [ty::t] = [];
    let tpt = ty::lookup_item_type(ccx.tcx, vid);
    alt ty::get(tpt.ty).struct {
      ty::ty_fn(f) {
        // N-ary variant.
        for f.inputs.each {|arg|
            let arg_ty =
                ty::substitute_type_params(ccx.tcx, enum_ty_params, arg.ty);
            result += [arg_ty];
        }
      }
      _ {
        // Nullary variant. Do nothing, as there are no arguments.
      }
    }
    /* result is a vector of the *expected* types of all the fields */

    ret result;
}


// Type resolution: the phase that finds all the types in the AST with
// unresolved type variables and replaces "ty_var" types with their
// substitutions.
mod writeback {

    export resolve_type_vars_in_fn;
    export resolve_type_vars_in_expr;

    fn resolve_type_vars_in_type(fcx: @fn_ctxt, sp: span, typ: ty::t) ->
       option<ty::t> {
        if !ty::type_has_vars(typ) { ret some(typ); }
        alt infer::fixup_vars(fcx.infcx, typ) {
          result::ok(new_type) { ret some(new_type); }
          result::err(e) {
            if !fcx.ccx.tcx.sess.has_errors() {
                fcx.ccx.tcx.sess.span_err(
                    sp,
                    #fmt["cannot determine a type \
                          for this expression: %s",
                         infer::fixup_err_to_str(e)])
            }
            ret none;
          }
        }
    }
    fn resolve_type_vars_for_node(wbcx: wb_ctxt, sp: span, id: ast::node_id)
        -> option<ty::t> {
        let fcx = wbcx.fcx, tcx = fcx.ccx.tcx;
        let n_ty = fcx.node_ty(id);
        alt resolve_type_vars_in_type(fcx, sp, n_ty) {
          none {
            wbcx.success = false;
            ret none;
          }

          some(t) {
            #debug["resolve_type_vars_for_node(id=%d, n_ty=%s, t=%s)",
                   id, ty_to_str(tcx, n_ty), ty_to_str(tcx, t)];
            write_ty_to_tcx(tcx, id, t);
            alt fcx.opt_node_ty_substs(id) {
              some(substs) {
                let mut new_substs = [];
                for substs.each {|subst|
                    alt resolve_type_vars_in_type(fcx, sp, subst) {
                      some(t) { new_substs += [t]; }
                      none { wbcx.success = false; ret none; }
                    }
                }
                write_substs_to_tcx(tcx, id, new_substs);
              }
              none {}
            }
            ret some(t);
          }
        }
    }

    fn maybe_resolve_type_vars_for_node(wbcx: wb_ctxt, sp: span,
                                        id: ast::node_id)
        -> option<ty::t> {
        if wbcx.fcx.node_types.contains_key(id as uint) {
            resolve_type_vars_for_node(wbcx, sp, id)
        } else {
            none
        }
    }

    type wb_ctxt =
        // As soon as we hit an error we have to stop resolving
        // the entire function
        {fcx: @fn_ctxt, mut success: bool};
    type wb_vt = visit::vt<wb_ctxt>;

    fn visit_stmt(s: @ast::stmt, wbcx: wb_ctxt, v: wb_vt) {
        if !wbcx.success { ret; }
        resolve_type_vars_for_node(wbcx, s.span, ty::stmt_node_id(s));
        visit::visit_stmt(s, wbcx, v);
    }
    fn visit_expr(e: @ast::expr, wbcx: wb_ctxt, v: wb_vt) {
        if !wbcx.success { ret; }
        resolve_type_vars_for_node(wbcx, e.span, e.id);
        alt e.node {
          ast::expr_fn(_, decl, _, _) |
          ast::expr_fn_block(decl, _) {
            vec::iter(decl.inputs) {|input|
                let r_ty = resolve_type_vars_for_node(wbcx, e.span, input.id);

                // Just in case we never constrained the mode to anything,
                // constrain it to the default for the type in question.
                alt (r_ty, input.mode) {
                  (some(t), ast::infer(_)) {
                    let tcx = wbcx.fcx.ccx.tcx;
                    let m_def = ty::default_arg_mode_for_ty(t);
                    ty::set_default_mode(tcx, input.mode, m_def);
                  }
                  _ {}
                }
            }
          }

          ast::expr_new(_, alloc_id, _) {
            resolve_type_vars_for_node(wbcx, e.span, alloc_id);
          }

          ast::expr_binary(_, _, _) | ast::expr_unary(_, _) |
          ast::expr_assign_op(_, _, _) | ast::expr_index(_, _) {
            maybe_resolve_type_vars_for_node(wbcx, e.span,
                                             ast_util::op_expr_callee_id(e));
          }

          _ { }
        }
        visit::visit_expr(e, wbcx, v);
    }
    fn visit_block(b: ast::blk, wbcx: wb_ctxt, v: wb_vt) {
        if !wbcx.success { ret; }
        resolve_type_vars_for_node(wbcx, b.span, b.node.id);
        visit::visit_block(b, wbcx, v);
    }
    fn visit_pat(p: @ast::pat, wbcx: wb_ctxt, v: wb_vt) {
        if !wbcx.success { ret; }
        resolve_type_vars_for_node(wbcx, p.span, p.id);
        #debug["Type for pattern binding %s (id %d) resolved to %s",
               pat_to_str(p), p.id,
               wbcx.fcx.ty_to_str(
                   ty::node_id_to_type(wbcx.fcx.ccx.tcx,
                                       p.id))];
        visit::visit_pat(p, wbcx, v);
    }
    fn visit_local(l: @ast::local, wbcx: wb_ctxt, v: wb_vt) {
        if !wbcx.success { ret; }
        let var_id = lookup_local(wbcx.fcx, l.span, l.node.id);
        alt infer::resolve_var(wbcx.fcx.infcx, var_id) {
          result::ok(lty) {
            #debug["Type for local %s (id %d) resolved to %s",
                   pat_to_str(l.node.pat), l.node.id,
                   wbcx.fcx.ty_to_str(lty)];
            write_ty_to_tcx(wbcx.fcx.ccx.tcx, l.node.id, lty);
          }
          result::err(e) {
            wbcx.fcx.ccx.tcx.sess.span_err(
                l.span,
                #fmt["cannot determine a type \
                      for this local variable: %s",
                     infer::fixup_err_to_str(e)]);
            wbcx.success = false;
          }
        }
        visit::visit_local(l, wbcx, v);
    }
    fn visit_item(_item: @ast::item, _wbcx: wb_ctxt, _v: wb_vt) {
        // Ignore items
    }

    fn resolve_type_vars_in_expr(fcx: @fn_ctxt, e: @ast::expr) -> bool {
        let wbcx = {fcx: fcx, mut success: true};
        let visit =
            visit::mk_vt(@{visit_item: visit_item,
                           visit_stmt: visit_stmt,
                           visit_expr: visit_expr,
                           visit_block: visit_block,
                           visit_pat: visit_pat,
                           visit_local: visit_local
                              with *visit::default_visitor()});
        visit.visit_expr(e, wbcx, visit);
        ret wbcx.success;
    }

    fn resolve_type_vars_in_fn(fcx: @fn_ctxt,
                               decl: ast::fn_decl,
                               blk: ast::blk) -> bool {
        let wbcx = {fcx: fcx, mut success: true};
        let visit =
            visit::mk_vt(@{visit_item: visit_item,
                           visit_stmt: visit_stmt,
                           visit_expr: visit_expr,
                           visit_block: visit_block,
                           visit_pat: visit_pat,
                           visit_local: visit_local
                              with *visit::default_visitor()});
        visit.visit_block(blk, wbcx, visit);
        for decl.inputs.each {|arg|
            resolve_type_vars_for_node(wbcx, arg.ty.span, arg.id);
        }
        ret wbcx.success;
    }

}

fn check_intrinsic_type(tcx: ty::ctxt, it: @ast::native_item) {
    fn param(tcx: ty::ctxt, n: uint) -> ty::t {
        ty::mk_param(tcx, n, local_def(0))
    }
    fn arg(m: ast::rmode, ty: ty::t) -> ty::arg {
        {mode: ast::expl(m), ty: ty}
    }
    let (n_tps, inputs, output) = alt it.ident {
      "size_of" | "align_of" { (1u, [], ty::mk_uint(tcx)) }
      "get_tydesc" { (1u, [], ty::mk_nil_ptr(tcx)) }
      "init" { (1u, [], param(tcx, 0u)) }
      "forget" { (1u, [arg(ast::by_move, param(tcx, 0u))],
                  ty::mk_nil(tcx)) }
      "reinterpret_cast" { (2u, [arg(ast::by_ref, param(tcx, 0u))],
                            param(tcx, 1u)) }
      "addr_of" { (1u, [arg(ast::by_ref, param(tcx, 0u))],
                   ty::mk_imm_ptr(tcx, param(tcx, 0u))) }
      other {
        tcx.sess.span_err(it.span, "unrecognized intrinsic function: `" +
                          other + "`");
        ret;
      }
    };
    let fty = ty::mk_fn(tcx, {proto: ast::proto_bare,
                              inputs: inputs, output: output,
                              ret_style: ast::return_val,
                              constraints: []});
    let i_ty = ty_of_native_item(tcx, m_collect, it);
    let i_n_tps = (*i_ty.bounds).len();
    if i_n_tps != n_tps {
        tcx.sess.span_err(it.span, #fmt("intrinsic has wrong number \
                                         of type parameters. found %u, \
                                         expected %u", i_n_tps, n_tps));
    } else {
        require_same_types(
            tcx, it.span, i_ty.ty, fty,
            {|| #fmt["intrinsic has wrong type. \
                      expected %s",
                     ty_to_str(tcx, fty)]});
    }
}

// Local variable gathering. We gather up all locals and create variable IDs
// for them before typechecking the function.
type gather_result =
    {infcx: infer::infer_ctxt,
     locals: hashmap<ast::node_id, ty_vid>,
     next_var_id: @mut uint};

// Used only as a helper for check_fn.
fn gather_locals(ccx: @crate_ctxt,
                 decl: ast::fn_decl,
                 body: ast::blk,
                 arg_tys: [ty::t],
                 old_fcx: option<@fn_ctxt>) -> gather_result {
    let {infcx, locals, nvi} = alt old_fcx {
      none {
        {infcx: infer::new_infer_ctxt(ccx.tcx),
         locals: int_hash(),
         nvi: @mut 0u}
      }
      some(fcx) {
        {infcx: fcx.infcx,
         locals: fcx.locals,
         nvi: fcx.next_var_id}
      }
    };
    let tcx = ccx.tcx;

    let next_var_id = fn@() -> uint {
        let rv = *nvi; *nvi += 1u; ret rv;
    };

    let assign = fn@(nid: ast::node_id, ty_opt: option<ty::t>) {
        let var_id = ty_vid(next_var_id());
        locals.insert(nid, var_id);
        alt ty_opt {
          none {/* nothing to do */ }
          some(typ) {
            infer::mk_eqty(infcx, ty::mk_var(tcx, var_id), typ);
          }
        }
    };

    // Add formal parameters.
    vec::iter2(arg_tys, decl.inputs) {|arg_ty, input|
        assign(input.id, some(arg_ty));
        #debug["Argument %s is assigned to %s",
               input.ident, locals.get(input.id).to_str()];
    }

    // Add explicitly-declared locals.
    let visit_local = fn@(local: @ast::local, &&e: (), v: visit::vt<()>) {
        let mut local_ty_opt = ast_ty_to_ty_crate_infer(ccx, local.node.ty);
        alt local_ty_opt {
            some(local_ty) if ty::type_has_rptrs(local_ty) {
                local_ty_opt = some(fixup_regions_to_block(ccx.tcx, local_ty,
                                                           local.node.ty));
            }
            _ { /* nothing to do */ }
        }

        assign(local.node.id, local_ty_opt);
        #debug["Local variable %s is assigned to %s",
               pat_to_str(local.node.pat),
               locals.get(local.node.id).to_str()];
        visit::visit_local(local, e, v);
    };

    // Add pattern bindings.
    let visit_pat = fn@(p: @ast::pat, &&e: (), v: visit::vt<()>) {
        alt p.node {
          ast::pat_ident(path, _)
          if !pat_util::pat_is_variant(ccx.tcx.def_map, p) {
            assign(p.id, none);
            #debug["Pattern binding %s is assigned to %s",
                   path.node.idents[0],
                   locals.get(p.id).to_str()];
          }
          _ {}
        }
        visit::visit_pat(p, e, v);
    };

    // Don't descend into fns and items
    fn visit_fn<T>(_fk: visit::fn_kind, _decl: ast::fn_decl, _body: ast::blk,
                   _sp: span, _id: ast::node_id, _t: T, _v: visit::vt<T>) {
    }
    fn visit_item<E>(_i: @ast::item, _e: E, _v: visit::vt<E>) { }

    let visit =
        @{visit_local: visit_local,
          visit_pat: visit_pat,
          visit_fn: bind visit_fn(_, _, _, _, _, _, _),
          visit_item: bind visit_item(_, _, _)
              with *visit::default_visitor()};

    visit::visit_block(body, (), visit::mk_vt(visit));
    ret {infcx: infcx,
         locals: locals,
         next_var_id: nvi};
}

// AST fragment checking
fn check_lit(ccx: @crate_ctxt, lit: @ast::lit) -> ty::t {
    alt lit.node {
      ast::lit_str(_) { ty::mk_str(ccx.tcx) }
      ast::lit_int(_, t) { ty::mk_mach_int(ccx.tcx, t) }
      ast::lit_uint(_, t) { ty::mk_mach_uint(ccx.tcx, t) }
      ast::lit_float(_, t) { ty::mk_mach_float(ccx.tcx, t) }
      ast::lit_nil { ty::mk_nil(ccx.tcx) }
      ast::lit_bool(_) { ty::mk_bool(ccx.tcx) }
    }
}

fn valid_range_bounds(tcx: ty::ctxt, from: @ast::expr, to: @ast::expr)
    -> bool {
    const_eval::compare_lit_exprs(tcx, from, to) <= 0
}

type pat_ctxt = {
    fcx: @fn_ctxt,
    map: pat_util::pat_id_map,
    alt_region: ty::region,
    block_region: ty::region,
    /* Equal to either alt_region or block_region. */
    pat_region: ty::region
};

fn count_region_params(ty: ty::t) -> uint {
    if (!ty::type_has_rptrs(ty)) { ret 0u; }

    let count = @mut 0u;
    ty::walk_ty(ty) {|ty|
        alt ty::get(ty).struct {
            ty::ty_rptr(ty::re_bound(ty::br_param(param_id, _)), _) {
                if param_id > *count {
                    *count = param_id;
                }
            }
            _ { /* no-op */ }
        }
    };
    ret *count;
}

type region_env = hashmap<ty::bound_region, region_vid>;

fn region_env() -> @region_env {
    ret @ty::br_hashmap();
}

// Replaces all region parameters in the given type with region variables.
// Does not descend into fn types.  This is used when deciding whether an impl
// applies at a given call site.  See also universally_quantify_before_call().
fn universally_quantify_regions(fcx: @fn_ctxt, renv: @region_env,
                                ty: ty::t) -> ty::t {
    ty::fold_region(fcx.ccx.tcx, ty) {|r, _under_rptr|
        alt r {
          ty::re_bound(br) {
            alt (*renv).find(br) {
              some(var_id) { ty::re_var(var_id) }
              none {
                let var_id = next_region_var_id(fcx);
                (*renv).insert(br, var_id);
                ty::re_var(var_id)
              }
            }
          }
          _ { r }
        }
    }
}

// Expects a function type.  Replaces all region parameters in the arguments
// and return type with fresh region variables. This is used when typechecking
// function calls, bind expressions, and method calls.
fn universally_quantify_before_call(
    fcx: @fn_ctxt, renv: @region_env, ty: ty::t) -> ty::t {
    if ty::type_has_rptrs(ty) {
        // This is subtle: we expect `ty` to be a function type, but
        // fold_region() will not descend into function types.  As it happens
        // we only want to descend 1 level, so we just bypass fold_region for
        // the outer type and apply it to all of the types contained with
        // `ty`.
        alt ty::get(ty).struct {
          sty @ ty::ty_fn(_) {
            ty::fold_sty_to_ty(fcx.ccx.tcx, sty) {|t|
                universally_quantify_regions(fcx, renv, t)
            }
          }
          _ {
            // if not a function type, we're gonna' report an error
            // at some point, since the user is trying to call this thing
            ty
          }
        }
    } else {
        ty
    }
}

fn check_pat_variant(pcx: pat_ctxt, pat: @ast::pat, path: @ast::path,
                     subpats: [@ast::pat], expected: ty::t) {
    // Typecheck the path.
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;
    let v_def = lookup_def(pcx.fcx, path.span, pat.id);
    let v_def_ids = ast_util::variant_def_ids(v_def);
    let ctor_tpt = ty::lookup_item_type(tcx, v_def_ids.enm);
    instantiate_path(pcx.fcx, path, ctor_tpt, pat.span, pat.id);

    // Take the enum type params out of `expected`.
    alt structure_of(pcx.fcx, pat.span, expected) {
      ty::ty_enum(_, expected_tps) {
        let ctor_ty = fcx.node_ty(pat.id);
        demand::with_substs(pcx.fcx, pat.span, expected, ctor_ty,
                            expected_tps);
        // Get the number of arguments in this enum variant.
        let arg_types = variant_arg_types(pcx.fcx.ccx, pat.span,
                                          v_def_ids.var, expected_tps);
        let arg_types = vec::map(arg_types) {|t|
            // NDM---is this reasonable?
            instantiate_bound_regions(pcx.fcx.ccx.tcx, pcx.pat_region, t)
        };
        let subpats_len = subpats.len(), arg_len = arg_types.len();
        if arg_len > 0u {
            // N-ary variant.
            if arg_len != subpats_len {
                let s = #fmt["this pattern has %u field%s, but the \
                              corresponding variant has %u field%s",
                             subpats_len,
                             if subpats_len == 1u { "" } else { "s" },
                             arg_len,
                             if arg_len == 1u { "" } else { "s" }];
                tcx.sess.span_err(pat.span, s);
            }

            vec::iter2(subpats, arg_types) {|subpat, arg_ty|
                check_pat(pcx, subpat, arg_ty);
            }
        } else if subpats_len > 0u {
            tcx.sess.span_err
                (pat.span, #fmt["this pattern has %u field%s, \
                                 but the corresponding variant has no fields",
                                subpats_len,
                                if subpats_len == 1u { "" }
                                else { "s" }]);
        }
      }
      _ {
        tcx.sess.span_err
            (pat.span,
             #fmt["mismatched types: expected enum but found `%s`",
                  ty_to_str(tcx, expected)]);
      }
    }
}

// Pattern checking is top-down rather than bottom-up so that bindings get
// their types immediately.
fn check_pat(pcx: pat_ctxt, pat: @ast::pat, expected: ty::t) {
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;
    alt pat.node {
      ast::pat_wild {
        fcx.write_ty(pat.id, expected);
      }
      ast::pat_lit(lt) {
        check_expr_with(pcx.fcx, lt, expected);
        fcx.write_ty(pat.id, fcx.expr_ty(lt));
      }
      ast::pat_range(begin, end) {
        check_expr_with(pcx.fcx, begin, expected);
        check_expr_with(pcx.fcx, end, expected);
        let b_ty = resolve_type_vars_if_possible(pcx.fcx,
                                                 fcx.expr_ty(begin));
        if !require_same_types(
            tcx, pat.span, b_ty,
            resolve_type_vars_if_possible(
                pcx.fcx, fcx.expr_ty(end)),
            {|| "mismatched types in range" }) {
            // no-op
        } else if !ty::type_is_numeric(b_ty) {
            tcx.sess.span_err(pat.span, "non-numeric type used in range");
        } else if !valid_range_bounds(tcx, begin, end) {
            tcx.sess.span_err(begin.span, "lower range bound must be less \
                                           than upper");
        }
        fcx.write_ty(pat.id, b_ty);
      }
      ast::pat_ident(name, sub)
      if !pat_util::pat_is_variant(tcx.def_map, pat) {
        let vid = lookup_local(pcx.fcx, pat.span, pat.id);
        let mut typ = ty::mk_var(tcx, vid);
        typ = demand::simple(pcx.fcx, pat.span, expected, typ);
        let canon_id = pcx.map.get(path_to_ident(name));
        if canon_id != pat.id {
            let tv_id = lookup_local(pcx.fcx, pat.span, canon_id);
            let ct = ty::mk_var(tcx, tv_id);
            typ = demand::simple(pcx.fcx, pat.span, ct, typ);
        }
        fcx.write_ty(pat.id, typ);
        alt sub {
          some(p) { check_pat(pcx, p, expected); }
          _ {}
        }
      }
      ast::pat_ident(path, _) {
        check_pat_variant(pcx, pat, path, [], expected);
      }
      ast::pat_enum(path, subpats) {
        check_pat_variant(pcx, pat, path, subpats, expected);
      }
      ast::pat_rec(fields, etc) {
        let ex_fields = alt structure_of(pcx.fcx, pat.span, expected) {
          ty::ty_rec(fields) { fields }
          _ {
            tcx.sess.span_fatal
                (pat.span,
                #fmt["mismatched types: expected `%s` but found record",
                     fcx.ty_to_str(expected)]);
          }
        };
        let f_count = vec::len(fields);
        let ex_f_count = vec::len(ex_fields);
        if ex_f_count < f_count || !etc && ex_f_count > f_count {
            tcx.sess.span_fatal
                (pat.span, #fmt["mismatched types: expected a record \
                      with %u fields, found one with %u \
                      fields",
                                ex_f_count, f_count]);
        }
        fn matches(name: str, f: ty::field) -> bool {
            ret str::eq(name, f.ident);
        }
        for fields.each {|f|
            alt vec::find(ex_fields, bind matches(f.ident, _)) {
              some(field) {
                check_pat(pcx, f.pat, field.mt.ty);
              }
              none {
                tcx.sess.span_fatal(pat.span,
                                    #fmt["mismatched types: did not \
                                          expect a record with a field `%s`",
                                         f.ident]);
              }
            }
        }
        fcx.write_ty(pat.id, expected);
      }
      ast::pat_tup(elts) {
        let ex_elts = alt structure_of(pcx.fcx, pat.span, expected) {
          ty::ty_tup(elts) { elts }
          _ {
            tcx.sess.span_fatal
                (pat.span,
                 #fmt["mismatched types: expected `%s`, found tuple",
                      fcx.ty_to_str(expected)]);
          }
        };
        let e_count = vec::len(elts);
        if e_count != vec::len(ex_elts) {
            tcx.sess.span_fatal
                (pat.span, #fmt["mismatched types: expected a tuple \
                      with %u fields, found one with %u \
                      fields", vec::len(ex_elts), e_count]);
        }
        let mut i = 0u;
        for elts.each {|elt|
            check_pat(pcx, elt, ex_elts[i]);
            i += 1u;
        }

        fcx.write_ty(pat.id, expected);
      }
      ast::pat_box(inner) {
        alt structure_of(pcx.fcx, pat.span, expected) {
          ty::ty_box(e_inner) {
            check_pat(pcx, inner, e_inner.ty);
            fcx.write_ty(pat.id, expected);
          }
          _ {
            tcx.sess.span_fatal(
                pat.span,
                "mismatched types: expected `" +
                pcx.fcx.ty_to_str(expected) +
                "` found box");
          }
        }
      }
      ast::pat_uniq(inner) {
        alt structure_of(pcx.fcx, pat.span, expected) {
          ty::ty_uniq(e_inner) {
            check_pat(pcx, inner, e_inner.ty);
            fcx.write_ty(pat.id, expected);
          }
          _ {
            tcx.sess.span_fatal(
                pat.span,
                "mismatched types: expected `" +
                pcx.fcx.ty_to_str(expected) +
                "` found uniq");
          }
        }
      }
    }
}

fn require_unsafe(sess: session, f_purity: ast::purity, sp: span) {
    alt f_purity {
      ast::unsafe_fn { ret; }
      _ {
        sess.span_err(
            sp,
            "unsafe operation requires unsafe function or block");
      }
    }
}

fn require_impure(sess: session, f_purity: ast::purity, sp: span) {
    alt f_purity {
      ast::unsafe_fn { ret; }
      ast::impure_fn | ast::crust_fn { ret; }
      ast::pure_fn {
        sess.span_err(sp, "found impure expression in pure function decl");
      }
    }
}

fn require_pure_call(ccx: @crate_ctxt, caller_purity: ast::purity,
                     callee: @ast::expr, sp: span) {
    if caller_purity == ast::unsafe_fn { ret; }
    let callee_purity = alt ccx.tcx.def_map.find(callee.id) {
      some(ast::def_fn(_, p)) { p }
      some(ast::def_variant(_, _)) { ast::pure_fn }
      _ {
        alt ccx.method_map.find(callee.id) {
          some(method_static(did)) {
            if did.crate == ast::local_crate {
                alt ccx.tcx.items.get(did.node) {
                  ast_map::node_method(m, _, _) { m.decl.purity }
                  _ { ccx.tcx.sess.span_bug(sp,
                             "Node not bound to a method") }
                }
            } else {
                csearch::lookup_method_purity(ccx.tcx.sess.cstore, did)
            }
          }
          some(method_param(iid, n_m, _, _)) | some(method_iface(iid, n_m)) {
            ty::iface_methods(ccx.tcx, iid)[n_m].purity
          }
          none { ast::impure_fn }
        }
      }
    };
    alt (caller_purity, callee_purity) {
      (ast::impure_fn, ast::unsafe_fn) | (ast::crust_fn, ast::unsafe_fn) {
        ccx.tcx.sess.span_err(sp, "safe function calls function marked \
                                   unsafe");
      }
      (ast::pure_fn, ast::unsafe_fn) | (ast::pure_fn, ast::impure_fn) {
        ccx.tcx.sess.span_err(sp, "pure function calls function not \
                                   known to be pure");
      }
      _ {}
    }
}

type unifier = fn@(@fn_ctxt, span, ty::t, ty::t) -> ty::t;

fn check_expr(fcx: @fn_ctxt, expr: @ast::expr) -> bool {
    fn dummy_unify(_fcx: @fn_ctxt, _sp: span, _expected: ty::t, actual: ty::t)
       -> ty::t {
        actual
    }
    ret check_expr_with_unifier(fcx, expr, dummy_unify,
                                ty::mk_nil(fcx.ccx.tcx));
}
fn check_expr_with(fcx: @fn_ctxt, expr: @ast::expr, expected: ty::t) -> bool {
    ret check_expr_with_unifier(fcx, expr, demand::simple, expected);
}

fn impl_self_ty(tcx: ty::ctxt, did: ast::def_id) -> {n_tps: uint, ty: ty::t} {
    if did.crate == ast::local_crate {
        alt check tcx.items.get(did.node) {
          ast_map::node_item(@{node: ast::item_impl(ts, _, st, _),
                               _}, _) {
            {n_tps: vec::len(ts), ty: ast_ty_to_ty(tcx, m_check, st)}
          }
        }
    } else {
        let ity = ty::lookup_item_type(tcx, did);
        {n_tps: vec::len(*ity.bounds), ty: ity.ty}
    }
}

/*
  Takes arguments describing a method, and returns either its origin,
  or <none> if it's unbound.

  expr: the entire method reference
  node_id: the method's ID
  name: the method's name
  ty: the type of the base expression
  tps: the ty substitutions that are part of the field expr
    (for example: in foo.bar<int,char>(), tps would be
     [int, char])
  include_private: true if we're inside the same class and should
     search private methods
 */
fn lookup_method(fcx: @fn_ctxt, expr: @ast::expr, node_id: ast::node_id,
                 name: ast::ident, ty: ty::t, tps: [ty::t],
                 include_private: bool)
    -> option<method_origin> {
    alt lookup_method_inner(fcx, expr, name, ty, include_private) {
      some({method_ty: fty, n_tps: method_n_tps, substs, origin, self_sub}) {
        let tcx = fcx.ccx.tcx;
        let mut substs = substs;
        let n_tps = vec::len(substs), n_tys = vec::len(tps);
        let has_self = ty::type_has_vars(fty);
        /* If either the method was declared to have ty params,
         or ty arguments were provided, or both... */
        if method_n_tps + n_tps > 0u {
          /* If no type arguments were given,
             or a different number of them were given than the
             method's declared types... */
            if n_tys == 0u || n_tys != method_n_tps {
                if n_tys != 0u {
                    tcx.sess.span_err
                        (expr.span, "incorrect number of type \
                                     parameters given for this method");

                }
                /* If not enough types were given, make some ty vars */
                substs += vec::from_fn(method_n_tps, {|_i|
                    ty::mk_var(tcx, next_ty_var_id(fcx))
                });
            } else {
             /* If the right number of types were given, just add them on */
                substs += tps;
            }
            /*
              For a class method, "substs" here begins with the class ty
              params
             */
            fcx.write_ty_substs(node_id, fty, substs);
        } else {
            if n_tys > 0u {
                tcx.sess.span_err(expr.span, "this method does not take type \
                                              parameters");
            }
            fcx.write_ty(node_id, fty);
        }
        if has_self && !option::is_none(self_sub) {
            let fty = fcx.node_ty(node_id);
            let fty = fixup_self_in_method_ty(
                tcx, fty, substs, option::get(self_sub));
            fcx.write_ty(node_id, fty);
        }
        if ty::type_has_rptrs(ty::ty_fn_ret(fty)) {
            let fty = fcx.node_ty(node_id);
            let self_region = region_of(fcx, expr);
            let fty = replace_self_region(fcx.ccx.tcx, self_region, fty);
            fcx.write_ty(node_id, fty);
        }
        some(origin)
      }
      none { none }
    }
}

enum method_kind {
    cls(ast::def_id), // *method* id (in both cases)
    an_iface(ast::def_id)
}

fn lookup_method_inner_(tcx: ty::ctxt, ms: [ty::method],
     tps: [ty::t], parent: method_kind, name: ast::ident, sp: span,
                        include_private: bool)
    -> option<{method_ty: ty::t, n_tps: uint, substs: [ty::t],
        origin: method_origin, self_sub: option<self_subst>}> {
    let mut i = 0u;
    for ms.each {|m|
       if m.ident == name {
          let fty = ty::mk_fn(tcx, {proto: ast::proto_box with m.fty});
          if ty::type_has_vars(fty) {
               tcx.sess.span_fatal(
                    sp, "can not call a method that contains a \
                                    self type through a boxed iface");
          } else if (*m.tps).len() > 0u  &&
               alt parent { an_iface(_) { true } cls(_) { false } } {
                   tcx.sess.span_fatal(
                        sp, "can not call a generic method through a \
                                    boxed iface");
          } else if m.privacy == ast::priv && !include_private {
                   tcx.sess.span_fatal(
                        sp, "Call to private method not allowed outside \
                          its defining class");
          }
          #debug("lookup_method_inner_: %s has %u ty params, by its \
            declaration", name, vec::len(*m.tps));
          ret some({method_ty: fty,
                    n_tps: vec::len(*m.tps),
                    substs: tps,
                    origin: alt parent {
                      cls(parent_id)          {
                        // look up method named <name>
                        // its id is did
                        let m_declared = ty::lookup_class_method_by_name(tcx,
                                      parent_id, name, sp);
                        method_static(m_declared)
                      }
                      an_iface(did) { method_iface(did, i) }
                      },
                    self_sub: none});
       }
       i += 1u;
    }
    ret none;
}

fn lookup_method_inner(fcx: @fn_ctxt, expr: @ast::expr,
                       name: ast::ident, ty: ty::t,
                       include_private: bool)
    -> option<{method_ty: ty::t, n_tps: uint, substs: [ty::t],
                  origin: method_origin,
                  self_sub: option<self_subst>}> {
    let tcx = fcx.ccx.tcx;

    #debug["lookup_method_inner: expr=%s name=%s ty=%s",
           expr_to_str(expr), name, fcx.ty_to_str(ty)];

    // First, see whether this is an interface-bounded parameter
    alt ty::get(ty).struct {
      ty::ty_param(n, did) {
        let mut bound_n = 0u;
        for vec::each(*tcx.ty_param_bounds.get(did.node)) {|bound|
            alt bound {
              ty::bound_iface(t) {
                let (iid, tps) = alt check ty::get(t).struct {
                  ty::ty_iface(i, tps) { (i, tps) }
                };
                let ifce_methods = ty::iface_methods(tcx, iid);
                alt vec::position(*ifce_methods, {|m| m.ident == name}) {
                  some(pos) {
                    let m = ifce_methods[pos];
                    ret some({method_ty: ty::mk_fn(tcx, {proto: ast::proto_box
                                                         with m.fty}),
                              n_tps: vec::len(*m.tps),
                              substs: tps,
                              origin: method_param(iid, pos, n, bound_n),
                              self_sub: some(self_param(ty, fcx, expr.span))
                             });
                  }
                  _ {}
                }
                bound_n += 1u;
              }
              _ {}
            }
        }
      }
      ty::ty_iface(did, tps) {
        alt lookup_method_inner_(tcx, *ty::iface_methods(tcx, did), tps,
              an_iface(did), name, expr.span, include_private) {
           some(r) { ret some(r); }
           none {  }
        }
      }
      ty::ty_class(did, tps) {
        alt lookup_method_inner_(tcx, *ty::iface_methods(tcx, did),
                                /* Need to include class tps so that the
                                    indices for ty params work out right */
                         tps, cls(did), name, expr.span, include_private) {
          some(r) { ret some(r); }
          none    { }
        }
      }
      _ {}
    }

    fn ty_from_did(tcx: ty::ctxt, did: ast::def_id) -> ty::t {
        if did.crate == ast::local_crate {
            alt check tcx.items.get(did.node) {
              ast_map::node_method(m, _, _) {
                let mt = ty_of_method(tcx, m_check, m);
                ty::mk_fn(tcx, {proto: ast::proto_box with mt.fty})
              }
            }
        } else {
            alt check ty::get(csearch::get_type(tcx, did).ty).struct {
              ty::ty_fn(fty) {
                ty::mk_fn(tcx, {proto: ast::proto_box with fty})
              }
            }
        }
    }

    let mut result = none, complained = false;
    std::list::iter(fcx.ccx.impl_map.get(expr.id)) {|impls|
        if option::is_none(result) {
            for vec::each(*impls) {|im|
                alt vec::find(im.methods, {|m| m.ident == name}) {
                  some(m) {
                    let mut {n_tps, ty: self_ty} = impl_self_ty(tcx, im.did);
                    let mut {vars, ty: self_ty} = if n_tps > 0u {
                        bind_params(fcx, self_ty, n_tps)
                    } else {
                        {vars: [], ty: self_ty}
                    };

                    // Here "self" refers to the callee side...
                    //let next_rid = count_region_params(self_ty);
                    //self_ty = instantiate_bound_regions(
                    //    fcx.ccx.tcx,
                    //    ty::re_bound(ty::br_param(next_rid)),
                    //    self_ty);
                    self_ty = universally_quantify_regions(
                        fcx, region_env(), self_ty);

                    // ... and "ty" refers to the caller side.
                    let ty = universally_quantify_regions(
                        fcx, region_env(), ty);

                    alt unify::unify(fcx, self_ty, ty) {
                      result::ok(_) {
                        if option::is_some(result) {
                            // FIXME[impl] score specificity?
                            if !complained {
                                tcx.sess.span_err(expr.span,
                                                  "multiple applicable \
                                                   methods in scope");
                                complained = true;
                            }
                        } else {
                            result = some({
                                method_ty: ty_from_did(tcx, m.did),
                                n_tps: m.n_tps,
                                substs: vars,
                                origin: method_static(m.did),
                                self_sub: none
                            });
                        }
                      }
                      result::err(_) {}
                    }
                  }
                  _ {}
                }
            }
        }
    }
    result
}

// Only for fields! Returns <none> for methods>
// FIXME: privacy flags
fn lookup_field_ty(tcx: ty::ctxt, class_id: ast::def_id,
   items:[ty::field_ty], fieldname: ast::ident, substs: [ty::t])
    -> option<ty::t> {
    option::map(vec::find(items, {|f| f.ident == fieldname}),
                {|f| ty::lookup_field_type(tcx, class_id, f.id, substs) })
}

/*
 * Returns the region that the value named by the given expression lives in.
 * The expression must have been typechecked. If the expression is not an
 * lvalue, returns the block region.
 *
 * Note that borrowing is not detected here, because we would have to
 * immediately structurally resolve too many types otherwise. Thus the
 * reference-counted heap and exchange heap regions will be reported as block
 * regions instead. This is cleaned up in the region checking pass.
 */
fn region_of(fcx: @fn_ctxt, expr: @ast::expr) -> ty::region {
    alt expr.node {
        ast::expr_path(path) {
            let defn = lookup_def(fcx, path.span, expr.id);
            alt defn {
                ast::def_local(local_id, _) |
                ast::def_upvar(local_id, _, _) {
                    let local_blocks = fcx.ccx.tcx.region_map.local_blocks;
                    let local_block_id = local_blocks.get(local_id);
                    ret ty::re_scope(local_block_id);
                }
                _ {
                    fcx.ccx.tcx.sess.span_unimpl(expr.span,
                                                 "immortal region");
                }
            }
        }
        ast::expr_field(base, _, _) {
            // FIXME: Insert borrowing!
            ret region_of(fcx, base);
        }
        ast::expr_index(base, _) {
            fcx.ccx.tcx.sess.span_unimpl(expr.span,
                                         "regions of index operations");
        }
        ast::expr_unary(ast::deref, base) {
            let expr_ty = fcx.expr_ty(base);
            let expr_ty = structurally_resolved_type(fcx, expr.span, expr_ty);
            alt ty::get(expr_ty).struct {
                ty::ty_rptr(region, _) { region }
                ty::ty_box(_) | ty::ty_uniq(_) {
                    fcx.ccx.tcx.sess.span_unimpl(expr.span, "borrowing");
                }
                _ { ret region_of(fcx, base); }
            }
        }
        _ {
            ret fcx.ccx.tcx.region_map.rvalue_to_region.get(expr.id);
        }
    }
}

fn check_expr_fn_with_unifier(fcx: @fn_ctxt,
                              expr: @ast::expr,
                              proto: ast::proto,
                              decl: ast::fn_decl,
                              body: ast::blk,
                              unify: unifier,
                              expected: ty::t,
                              is_loop_body: bool) {
    let tcx = fcx.ccx.tcx;
    let fty = ty::mk_fn(tcx,
                        ty_of_fn_decl(tcx, m_check_tyvar(fcx), proto, decl));

    #debug("check_expr_fn_with_unifier %s fty=%s",
           expr_to_str(expr), fcx.ty_to_str(fty));

    fcx.write_ty(expr.id, fty);

    // Unify the type of the function with the expected type before we
    // typecheck the body so that we have more information about the
    // argument types in the body. This is needed to make binops and
    // record projection work on type inferred arguments.
    unify(fcx, expr.span, expected, fty);

    let ret_ty = ty::ty_fn_ret(fty);
    let arg_tys = vec::map(ty::ty_fn_args(fty)) {|a| a.ty };

    check_fn(fcx.ccx, proto, decl, body, expr.id,
             ret_ty, arg_tys, is_loop_body, some(fcx),
             fcx.self_ty);
}

fn check_expr_with_unifier(fcx: @fn_ctxt, expr: @ast::expr, unify: unifier,
                           expected: ty::t) -> bool {

    #debug(">> typechecking expr %d (%s)",
           expr.id, syntax::print::pprust::expr_to_str(expr));

    // A generic function to factor out common logic from call and bind
    // expressions.
    fn check_call_or_bind(
        fcx: @fn_ctxt, sp: span, fty: ty::t,
        args: [option<@ast::expr>]) -> {fty: ty::t, bot: bool} {

        let fty = universally_quantify_before_call(fcx, region_env(), fty);
        #debug["check_call_or_bind: after universal quant., fty=%s",
               fcx.ty_to_str(fty)];
        let sty = structure_of(fcx, sp, fty);
        // Grab the argument types
        let mut arg_tys = alt sty {
          ty::ty_fn({inputs: arg_tys, _}) { arg_tys }
          _ {
            fcx.ccx.tcx.sess.span_fatal(sp, "mismatched types: \
                                             expected function or native \
                                             function but found "
                                        + fcx.ty_to_str(fty))
          }
        };

        // Check that the correct number of arguments were supplied.
        let expected_arg_count = vec::len(arg_tys);
        let supplied_arg_count = vec::len(args);
        if expected_arg_count != supplied_arg_count {
            fcx.ccx.tcx.sess.span_err(
                sp, #fmt["this function takes %u parameter%s but %u \
                          parameter%s supplied", expected_arg_count,
                         if expected_arg_count == 1u {
                             ""
                         } else {
                             "s"
                         },
                         supplied_arg_count,
                         if supplied_arg_count == 1u {
                             " was"
                         } else {
                             "s were"
                         }]);

            // Just use fresh type variables for the types,
            // since we don't know them.
            arg_tys = vec::from_fn(supplied_arg_count) {|_i|
                {mode: ast::expl(ast::by_ref),
                 ty: next_ty_var(fcx)}
            };
        }

        // Check the arguments.
        // We do this in a pretty awful way: first we typecheck any arguments
        // that are not anonymous functions, then we typecheck the anonymous
        // functions. This is so that we have more information about the types
        // of arguments when we typecheck the functions. This isn't really the
        // right way to do this.
        let check_args = fn@(check_blocks: bool) -> bool {
            let mut i = 0u;
            let mut bot = false;
            for args.each {|a_opt|
                alt a_opt {
                  some(a) {
                    let is_block = alt a.node {
                      ast::expr_fn_block(_, _) { true }
                      _ { false }
                    };
                    if is_block == check_blocks {
                        let t = arg_tys[i].ty;
                        bot |= check_expr_with_unifier(fcx, a, demand::simple,
                                                       t);
                    }
                  }
                  none { }
                }
                i += 1u;
            }
            ret bot;
        };

        let bot = check_args(false) | check_args(true);

        {fty: fty, bot: bot}
    }

    // A generic function for checking assignment expressions
    fn check_assignment(fcx: @fn_ctxt, _sp: span, lhs: @ast::expr,
                        rhs: @ast::expr, id: ast::node_id) -> bool {
        let mut bot = check_expr(fcx, lhs);
        bot |= check_expr_with(fcx, rhs, fcx.expr_ty(lhs));
        fcx.write_ty(id, ty::mk_nil(fcx.ccx.tcx));
        ret bot;
    }

    // A generic function for checking call expressions
    fn check_call(fcx: @fn_ctxt, sp: span, f: @ast::expr, args: [@ast::expr])
        -> {fty: ty::t, bot: bool} {
        let mut args_opt_0: [option<@ast::expr>] = [];
        for args.each {|arg|
            args_opt_0 += [some::<@ast::expr>(arg)];
        }

        let bot = check_expr(fcx, f);

        // Call the generic checker.
        let fn_ty = fcx.expr_ty(f);
        let {fty, bot: bot1} = check_call_or_bind(fcx, sp, fn_ty, args_opt_0);
        ret {fty: fty, bot: bot | bot1};
    }

    // A generic function for doing all of the checking for call expressions
    fn check_call_full(fcx: @fn_ctxt, sp: span, id: ast::node_id,
                       f: @ast::expr, args: [@ast::expr]) -> bool {
        let {fty, bot} = check_call(fcx, sp, f, args);

        /* need to restrict oper to being an explicit expr_path if we're
        inside a pure function */
        require_pure_call(fcx.ccx, fcx.purity, f, sp);

        // Pull the return type out of the type of the function.
        let {rt, bot} = alt structure_of(fcx, sp, fty) {
          ty::ty_fn(f) {
            {rt: f.output,
             bot: bot | (f.ret_style == ast::noreturn)}
          }
          _ { fcx.ccx.tcx.sess.span_fatal(sp, "calling non-function"); }
        };
        fcx.write_ty(id, rt);
        ret bot;
    }

    // A generic function for checking for or for-each loops
    fn check_for(fcx: @fn_ctxt, local: @ast::local,
                 element_ty: ty::t, body: ast::blk,
                 node_id: ast::node_id) -> bool {
        let locid = lookup_local(fcx, local.span, local.node.id);
        demand::simple(fcx, local.span,
                       ty::mk_var(fcx.ccx.tcx, locid),
                       element_ty);
        let bot = check_decl_local(fcx, local);
        check_block_no_value(fcx, body);
        fcx.write_nil(node_id);
        ret bot;
    }

    // A generic function for checking the then and else in an if
    // or if-check
    fn check_then_else(fcx: @fn_ctxt, thn: ast::blk,
                       elsopt: option<@ast::expr>, id: ast::node_id,
                       _sp: span) -> bool {
        let (if_t, if_bot) =
            alt elsopt {
              some(els) {
                let if_t = next_ty_var(fcx);
                let thn_bot = check_block(fcx, thn);
                let thn_t = fcx.node_ty(thn.node.id);
                demand::simple(fcx, thn.span, if_t, thn_t);
                let els_bot = check_expr_with(fcx, els, if_t);
                (if_t, thn_bot & els_bot)
              }
              none {
                check_block_no_value(fcx, thn);
                (ty::mk_nil(fcx.ccx.tcx), false)
              }
            };
        fcx.write_ty(id, if_t);
        ret if_bot;
    }

    fn binop_method(op: ast::binop) -> option<str> {
        alt op {
          ast::add | ast::subtract | ast::mul | ast::div | ast::rem |
          ast::bitxor | ast::bitand | ast::bitor | ast::lsl | ast::lsr |
          ast::asr { some(ast_util::binop_to_str(op)) }
          _ { none }
        }
    }
    fn lookup_op_method(fcx: @fn_ctxt, op_ex: @ast::expr, self_t: ty::t,
                        opname: str, args: [option<@ast::expr>])
        -> option<(ty::t, bool)> {
        let callee_id = ast_util::op_expr_callee_id(op_ex);
        alt lookup_method(fcx, op_ex, callee_id, opname, self_t, [], false) {
          some(origin) {
            let {fty: method_ty, bot: bot} = {
                let method_ty = fcx.node_ty(callee_id);
                check_call_or_bind(fcx, op_ex.span, method_ty, args)
            };
            fcx.ccx.method_map.insert(op_ex.id, origin);
            some((ty::ty_fn_ret(method_ty), bot))
          }
          _ { none }
        }
    }
    // could be either a expr_binop or an expr_assign_binop
    fn check_binop(fcx: @fn_ctxt, expr: @ast::expr,
                   op: ast::binop,
                   lhs: @ast::expr,
                   rhs: @ast::expr) -> bool {
        let tcx = fcx.ccx.tcx;
        let lhs_bot = check_expr(fcx, lhs);
        let lhs_t = fcx.expr_ty(lhs);
        let lhs_t = structurally_resolved_type(fcx, lhs.span, lhs_t);
        ret alt (op, ty::get(lhs_t).struct) {
          (ast::add, ty::ty_vec(lhs_mt)) {
            // For adding vectors with type L=[ML TL] and R=[MR TR], the the
            // result [ML T] where TL <: T and TR <: T.  In other words, the
            // result type is (generally) the LUB of (TL, TR) and takes the
            // mutability from the LHS.
            let t_var = next_ty_var(fcx);
            let const_vec_t = ty::mk_vec(tcx, {ty: t_var,
                                               mutbl: ast::m_const});
            demand::simple(fcx, lhs.span, const_vec_t, lhs_t);
            let rhs_bot = check_expr_with(fcx, rhs, const_vec_t);
            let result_vec_t = ty::mk_vec(tcx, {ty: t_var,
                                                mutbl: lhs_mt.mutbl});
            fcx.write_ty(expr.id, result_vec_t);
            lhs_bot | rhs_bot
          }

          (_, _) if ty::type_is_integral(lhs_t) &&
          ast_util::is_shift_binop(op) {
            // Shift is a special case: rhs can be any integral type
            let rhs_bot = check_expr(fcx, rhs);
            let rhs_t = fcx.expr_ty(rhs);
            require_integral(fcx, rhs.span, rhs_t);
            fcx.write_ty(expr.id, lhs_t);
            lhs_bot | rhs_bot
          }

          (_, _) if ty::is_binopable(tcx, lhs_t, op) {
            let tvar = next_ty_var(fcx);
            demand::simple(fcx, expr.span, tvar, lhs_t);
            let rhs_bot = check_expr_with(fcx, rhs, tvar);
            let rhs_t = alt op {
              ast::eq | ast::lt | ast::le | ast::ne | ast::ge |
              ast::gt {
                // these comparison operators are handled in a
                // separate case below.
                tcx.sess.span_bug(
                    expr.span,
                    #fmt["Comparison operator in expr_binop: %s",
                         ast_util::binop_to_str(op)]);
              }
              _ { lhs_t }
            };
            fcx.write_ty(expr.id, rhs_t);
            if !ast_util::lazy_binop(op) { lhs_bot | rhs_bot }
            else { lhs_bot }
          }

          (_, _) {
            let (result, rhs_bot) =
                check_user_binop(fcx, expr, lhs_t, op, rhs);
            fcx.write_ty(expr.id, result);
            lhs_bot | rhs_bot
          }
        };
    }
    fn check_user_binop(fcx: @fn_ctxt, ex: @ast::expr, lhs_resolved_t: ty::t,
                        op: ast::binop, rhs: @ast::expr) -> (ty::t, bool) {
        let tcx = fcx.ccx.tcx;
        alt binop_method(op) {
          some(name) {
            alt lookup_op_method(fcx, ex, lhs_resolved_t, name, [some(rhs)]) {
              some(pair) { ret pair; }
              _ {}
            }
          }
          _ {}
        }
        check_expr(fcx, rhs);
        tcx.sess.span_err(
            ex.span, "binary operation " + ast_util::binop_to_str(op) +
            " cannot be applied to type `" +
            fcx.ty_to_str(lhs_resolved_t) +
            "`");
        (lhs_resolved_t, false)
    }
    fn check_user_unop(fcx: @fn_ctxt, op_str: str, mname: str,
                       ex: @ast::expr, rhs_t: ty::t) -> ty::t {
        alt lookup_op_method(fcx, ex, rhs_t, mname, []) {
          some((ret_ty, _)) { ret_ty }
          _ {
            fcx.ccx.tcx.sess.span_err(
                ex.span, #fmt["cannot apply unary operator `%s` to type `%s`",
                              op_str, fcx.ty_to_str(rhs_t)]);
            rhs_t
          }
        }
    }

    let tcx = fcx.ccx.tcx;
    let id = expr.id;
    let mut bot = false;
    alt expr.node {

      ast::expr_vstore(ev, vst) {
        let mut typ = alt ev.node {
          ast::expr_lit(@{node: ast::lit_str(s), span:_}) {
            let tt = ast_expr_vstore_to_vstore(fcx, ev,
                                               str::len(s), vst);
            ty::mk_estr(tcx, tt)
          }
          ast::expr_vec(args, mutbl) {
            let tt = ast_expr_vstore_to_vstore(fcx, ev,
                                               vec::len(args), vst);
            let t: ty::t = next_ty_var(fcx);
            for args.each {|e| bot |= check_expr_with(fcx, e, t); }
            ty::mk_evec(tcx, {ty: t, mutbl: mutbl}, tt)
          }
          _ {
            tcx.sess.span_bug(expr.span, "vstore modifier on non-sequence")
          }
        };
        alt vst {
          ast::vstore_slice(_) {
            let region = fcx.ccx.tcx.region_map.rvalue_to_region.get(ev.id);
            typ = replace_default_region(tcx, region, typ);
          }
          _ { }
        }
        fcx.write_ty(ev.id, typ);
        fcx.write_ty(id, typ);
      }

      ast::expr_lit(lit) {
        let typ = check_lit(fcx.ccx, lit);
        fcx.write_ty(id, typ);
      }

      // Something of a hack: special rules for comparison operators that
      // simply unify LHS and RHS.  This helps with inference as LHS and RHS
      // do not need to be "resolvable".  Some tests, particularly those with
      // complicated iface requirements, fail without this---I think this code
      // can be removed if we improve iface resolution to be more eager when
      // possible.
      ast::expr_binary(ast::eq, lhs, rhs) |
      ast::expr_binary(ast::ne, lhs, rhs) |
      ast::expr_binary(ast::lt, lhs, rhs) |
      ast::expr_binary(ast::le, lhs, rhs) |
      ast::expr_binary(ast::gt, lhs, rhs) |
      ast::expr_binary(ast::ge, lhs, rhs) {
        let tcx = fcx.ccx.tcx;
        let tvar = next_ty_var(fcx);
        bot |= check_expr_with(fcx, lhs, tvar);
        bot |= check_expr_with(fcx, rhs, tvar);
        fcx.write_ty(id, ty::mk_bool(tcx));
      }
      ast::expr_binary(op, lhs, rhs) {
        bot |= check_binop(fcx, expr, op, lhs, rhs);
      }
      ast::expr_assign_op(op, lhs, rhs) {
        require_impure(tcx.sess, fcx.purity, expr.span);
        bot |= check_binop(fcx, expr, op, lhs, rhs);
        let lhs_t = fcx.expr_ty(lhs);
        let result_t = fcx.expr_ty(expr);
        demand::simple(fcx, expr.span, result_t, lhs_t);

        // Overwrite result of check_binop...this preserves existing behavior
        // but seems quite dubious with regard to user-defined methods
        // and so forth. - Niko
        fcx.write_nil(expr.id);
      }
      ast::expr_unary(unop, oper) {
        bot = check_expr(fcx, oper);
        let mut oper_t = fcx.expr_ty(oper);
        alt unop {
          ast::box(mutbl) {
            oper_t = ty::mk_box(tcx, {ty: oper_t, mutbl: mutbl});
          }
          ast::uniq(mutbl) {
            oper_t = ty::mk_uniq(tcx, {ty: oper_t, mutbl: mutbl});
          }
          ast::deref {
            alt structure_of(fcx, expr.span, oper_t) {
              ty::ty_box(inner) { oper_t = inner.ty; }
              ty::ty_uniq(inner) { oper_t = inner.ty; }
              ty::ty_res(_, inner, _) { oper_t = inner; }
              ty::ty_enum(id, tps) {
                let variants = ty::enum_variants(tcx, id);
                if vec::len(*variants) != 1u ||
                       vec::len(variants[0].args) != 1u {
                    tcx.sess.span_fatal(expr.span,
                                        "can only dereference enums " +
                                        "with a single variant which has a "
                                            + "single argument");
                }
                oper_t =
                    ty::substitute_type_params(tcx, tps, variants[0].args[0]);
              }
              ty::ty_ptr(inner) {
                oper_t = inner.ty;
                require_unsafe(tcx.sess, fcx.purity, expr.span);
              }
              ty::ty_rptr(_, inner) { oper_t = inner.ty; }
              _ {
                  tcx.sess.span_err(expr.span,
                      #fmt("Type %s cannot be dereferenced",
                           ty_to_str(tcx, oper_t)));
              }
            }
          }
          ast::not {
            oper_t = structurally_resolved_type(fcx, oper.span, oper_t);
            if !(ty::type_is_integral(oper_t) ||
                 ty::get(oper_t).struct == ty::ty_bool) {
                oper_t = check_user_unop(fcx, "!", "!", expr, oper_t);
            }
          }
          ast::neg {
            oper_t = structurally_resolved_type(fcx, oper.span, oper_t);
            if !(ty::type_is_integral(oper_t) ||
                 ty::type_is_fp(oper_t)) {
                oper_t = check_user_unop(fcx, "-", "unary-", expr, oper_t);
            }
          }
        }
        fcx.write_ty(id, oper_t);
      }
      ast::expr_addr_of(mutbl, oper) {
        bot = check_expr(fcx, oper);
        let mut oper_t = fcx.expr_ty(oper);

        let region = region_of(fcx, oper);
        let tm = { ty: oper_t, mutbl: mutbl };
        oper_t = ty::mk_rptr(tcx, region, tm);
        fcx.write_ty(id, oper_t);
      }
      ast::expr_path(pth) {
        let defn = lookup_def(fcx, pth.span, id);

        let tpt = ty_param_bounds_and_ty_for_def(fcx, expr.span, defn);
        instantiate_path(fcx, pth, tpt, expr.span, expr.id);
      }
      ast::expr_mac(_) { tcx.sess.bug("unexpanded macro"); }
      ast::expr_fail(expr_opt) {
        bot = true;
        alt expr_opt {
          none {/* do nothing */ }
          some(e) { check_expr_with(fcx, e, ty::mk_str(tcx)); }
        }
        fcx.write_bot(id);
      }
      ast::expr_break { fcx.write_bot(id); bot = true; }
      ast::expr_cont { fcx.write_bot(id); bot = true; }
      ast::expr_ret(expr_opt) {
        bot = true;
        let ret_ty = alt fcx.indirect_ret_ty {
          some(t) { t } none { fcx.ret_ty }
        };
        alt expr_opt {
          none {
            let nil = ty::mk_nil(tcx);
            if !are_compatible(fcx, ret_ty, nil) {
                tcx.sess.span_err(expr.span,
                                  "ret; in function returning non-nil");
            }
          }
          some(e) { check_expr_with(fcx, e, ret_ty); }
        }
        fcx.write_bot(id);
      }
      ast::expr_be(e) {
        // FIXME: prove instead of assert
        assert (ast_util::is_call_expr(e));
        check_expr_with(fcx, e, fcx.ret_ty);
        bot = true;
        fcx.write_nil(id);
      }
      ast::expr_log(_, lv, e) {
        bot = check_expr_with(fcx, lv, ty::mk_mach_uint(tcx, ast::ty_u32));
        // Note: this does not always execute, so do not propagate bot:
        check_expr(fcx, e);
        fcx.write_nil(id);
      }
      ast::expr_check(_, e) {
        bot = check_pred_expr(fcx, e);
        fcx.write_nil(id);
      }
      ast::expr_if_check(cond, thn, elsopt) {
        bot =
            check_pred_expr(fcx, cond) |
                check_then_else(fcx, thn, elsopt, id, expr.span);
      }
      ast::expr_assert(e) {
        bot = check_expr_with(fcx, e, ty::mk_bool(tcx));
        fcx.write_nil(id);
      }
      ast::expr_copy(a) {
        bot = check_expr_with_unifier(fcx, a, unify, expected);
        fcx.write_ty(id, fcx.expr_ty(a));
      }
      ast::expr_move(lhs, rhs) {
        require_impure(tcx.sess, fcx.purity, expr.span);
        bot = check_assignment(fcx, expr.span, lhs, rhs, id);
      }
      ast::expr_assign(lhs, rhs) {
        require_impure(tcx.sess, fcx.purity, expr.span);
        bot = check_assignment(fcx, expr.span, lhs, rhs, id);
      }
      ast::expr_swap(lhs, rhs) {
        require_impure(tcx.sess, fcx.purity, expr.span);
        bot = check_assignment(fcx, expr.span, lhs, rhs, id);
      }
      ast::expr_if(cond, thn, elsopt) {
        bot =
            check_expr_with(fcx, cond, ty::mk_bool(tcx)) |
                check_then_else(fcx, thn, elsopt, id, expr.span);
      }
      ast::expr_while(cond, body) {
        bot = check_expr_with(fcx, cond, ty::mk_bool(tcx));
        check_block_no_value(fcx, body);
        fcx.write_ty(id, ty::mk_nil(tcx));
      }
      ast::expr_do_while(body, cond) {
        bot = check_expr_with(fcx, cond, ty::mk_bool(tcx)) |
              check_block_no_value(fcx, body);
        fcx.write_ty(id, fcx.node_ty(body.node.id));
      }
      ast::expr_loop(body) {
          check_block_no_value(fcx, body);
          fcx.write_ty(id, ty::mk_nil(tcx));
          bot = !may_break(body);
      }
      ast::expr_alt(discrim, arms, _) {
        let pattern_ty = next_ty_var(fcx);
        bot = check_expr_with(fcx, discrim, pattern_ty);

        let parent_region = tcx.region_map.rvalue_to_region.get(discrim.id);

        // Typecheck the patterns first, so that we get types for all the
        // bindings.
        //let pattern_ty = fcx.expr_ty(discrim);
        for arms.each {|arm|
            let pcx = {
                fcx: fcx,
                map: pat_util::pat_id_map(tcx.def_map, arm.pats[0]),
                alt_region: parent_region,
                block_region: ty::re_scope(arm.body.node.id),
                pat_region: parent_region
            };

            for arm.pats.each {|p| check_pat(pcx, p, pattern_ty);}
        }
        // Now typecheck the blocks.
        let mut result_ty = next_ty_var(fcx);
        let mut arm_non_bot = false;
        for arms.each {|arm|
            alt arm.guard {
              some(e) { check_expr_with(fcx, e, ty::mk_bool(tcx)); }
              none { }
            }
            if !check_block(fcx, arm.body) { arm_non_bot = true; }
            let bty = fcx.node_ty(arm.body.node.id);
            result_ty = demand::simple(fcx, arm.body.span, result_ty, bty);
        }
        bot |= !arm_non_bot;
        if !arm_non_bot { result_ty = ty::mk_bot(tcx); }
        fcx.write_ty(id, result_ty);
      }
      ast::expr_fn(proto, decl, body, captures) {
        check_expr_fn_with_unifier(fcx, expr, proto, decl, body,
                                   unify, expected, false);
        capture::check_capture_clause(tcx, expr.id, proto, *captures);
      }
      ast::expr_fn_block(decl, body) {
        // Take the prototype from the expected type, but default to block:
        let proto = alt ty::get(expected).struct {
          ty::ty_fn({proto, _}) { proto }
          _ { ast::proto_box }
        };
        #debug("checking expr_fn_block %s expected=%s",
               expr_to_str(expr),
               ty_to_str(tcx, expected));
        check_expr_fn_with_unifier(fcx, expr, proto, decl, body,
                                   unify, expected, false);
      }
      ast::expr_loop_body(b) {
        let rty = structurally_resolved_type(fcx, expr.span, expected);
        let (inner_ty, proto) = alt check ty::get(rty).struct {
          ty::ty_fn(fty) {
            demand::simple(fcx, expr.span, fty.output, ty::mk_bool(tcx));
            (ty::mk_fn(tcx, {output: ty::mk_nil(tcx) with fty}),
             fty.proto)
          }
        };
        alt check b.node {
          ast::expr_fn_block(decl, body) {
            check_expr_fn_with_unifier(fcx, b, proto, decl, body,
                                       unify, inner_ty, true);
          }
        }
        let block_ty = structurally_resolved_type(
            fcx, expr.span, fcx.node_ty(b.id));
        alt check ty::get(block_ty).struct {
          ty::ty_fn(fty) {
            fcx.write_ty(expr.id, ty::mk_fn(tcx, {output: ty::mk_bool(tcx)
                                                  with fty}));
          }
        }
      }
      ast::expr_block(b) {
        // If this is an unchecked block, turn off purity-checking
        bot = check_block(fcx, b);
        let typ =
            alt b.node.expr {
              some(expr) { fcx.expr_ty(expr) }
              none { ty::mk_nil(tcx) }
            };
        fcx.write_ty(id, typ);
      }
      ast::expr_bind(f, args) {
        // Call the generic checker.
        bot = check_expr(fcx, f);

        let {fty, bot: ccob_bot} = {
            let fn_ty = fcx.expr_ty(f);
            check_call_or_bind(fcx, expr.span, fn_ty, args)
        };
        bot |= ccob_bot;

        // TODO: Perform substitutions on the return type.

        // Pull the argument and return types out.
        let mut proto, arg_tys, rt, cf, constrs;
        alt structure_of(fcx, expr.span, fty) {
          // FIXME:
          // probably need to munge the constrs to drop constraints
          // for any bound args
          ty::ty_fn(f) {
            proto = f.proto;
            arg_tys = f.inputs;
            rt = f.output;
            cf = f.ret_style;
            constrs = f.constraints;
          }
          _ { fail "LHS of bind expr didn't have a function type?!"; }
        }

        let proto = alt proto {
          ast::proto_bare | ast::proto_box | ast::proto_uniq {
            ast::proto_box
          }
          ast::proto_any | ast::proto_block {
            tcx.sess.span_err(expr.span,
                              #fmt["cannot bind %s closures",
                                   proto_to_str(proto)]);
            proto // dummy value so compilation can proceed
          }
        };

        // For each blank argument, add the type of that argument
        // to the resulting function type.
        let mut out_args = [];
        let mut i = 0u;
        while i < vec::len(args) {
            alt args[i] {
              some(_) {/* no-op */ }
              none { out_args += [arg_tys[i]]; }
            }
            i += 1u;
        }

        let ft = ty::mk_fn(tcx, {proto: proto,
                                 inputs: out_args, output: rt,
                                 ret_style: cf, constraints: constrs});
        fcx.write_ty(id, ft);
      }
      ast::expr_call(f, args, _) {
        bot = check_call_full(fcx, expr.span, expr.id, f, args);
      }
      ast::expr_cast(e, t) {
        bot = check_expr(fcx, e);
        let t_1 = ast_ty_to_ty_crate(fcx.ccx, t);
        let t_e = fcx.expr_ty(e);

        alt ty::get(t_1).struct {
          // This will be looked up later on
          ty::ty_iface(_, _) {}
          _ {
            if ty::type_is_nil(t_e) {
                tcx.sess.span_err(expr.span, "cast from nil: " +
                                  ty_to_str(tcx, t_e) + " as " +
                                  ty_to_str(tcx, t_1));
            } else if ty::type_is_nil(t_1) {
                tcx.sess.span_err(expr.span, "cast to nil: " +
                                  ty_to_str(tcx, t_e) + " as " +
                                  ty_to_str(tcx, t_1));
            }

            let t_1_is_scalar = type_is_scalar(fcx, expr.span, t_1);
            if type_is_c_like_enum(fcx,expr.span,t_e) && t_1_is_scalar {
                /* this case is allowed */
            } else if !(type_is_scalar(fcx,expr.span,t_e) && t_1_is_scalar) {
                // FIXME there are more forms of cast to support, eventually.
                tcx.sess.span_err(expr.span,
                                  "non-scalar cast: " +
                                  ty_to_str(tcx, t_e) + " as " +
                                  ty_to_str(tcx, t_1));
            }
          }
        }
        fcx.write_ty(id, t_1);
      }
      ast::expr_vec(args, mutbl) {
        let t: ty::t = next_ty_var(fcx);
        for args.each {|e| bot |= check_expr_with(fcx, e, t); }
        let typ = ty::mk_vec(tcx, {ty: t, mutbl: mutbl});
        fcx.write_ty(id, typ);
      }
      ast::expr_tup(elts) {
        let mut elt_ts = [];
        vec::reserve(elt_ts, vec::len(elts));
        for elts.each {|e|
            check_expr(fcx, e);
            let ety = fcx.expr_ty(e);
            elt_ts += [ety];
        }
        let typ = ty::mk_tup(tcx, elt_ts);
        fcx.write_ty(id, typ);
      }
      ast::expr_rec(fields, base) {
        option::iter(base) {|b| check_expr(fcx, b); }
        let fields_t = vec::map(fields, {|f|
            bot |= check_expr(fcx, f.node.expr);
            let expr_t = fcx.expr_ty(f.node.expr);
            let expr_mt = {ty: expr_t, mutbl: f.node.mutbl};
            // for the most precise error message,
            // should be f.node.expr.span, not f.span
            respan(f.node.expr.span, {ident: f.node.ident, mt: expr_mt})
        });
        alt base {
          none {
            fn get_node(f: spanned<field>) -> field { f.node }
            let typ = ty::mk_rec(tcx, vec::map(fields_t, get_node));
            fcx.write_ty(id, typ);
          }
          some(bexpr) {
            bot |= check_expr(fcx, bexpr);
            let bexpr_t = fcx.expr_ty(bexpr);
            let mut base_fields: [field] = [];
            alt structure_of(fcx, expr.span, bexpr_t) {
              ty::ty_rec(flds) { base_fields = flds; }
              _ {
                tcx.sess.span_fatal(expr.span,
                                    "record update has non-record base");
              }
            }
            fcx.write_ty(id, bexpr_t);
            for fields_t.each {|f|
                let mut found = false;
                for base_fields.each {|bf|
                    if str::eq(f.node.ident, bf.ident) {
                        demand::simple(fcx, f.span, bf.mt.ty, f.node.mt.ty);
                        found = true;
                    }
                }
                if !found {
                    tcx.sess.span_fatal(f.span,
                                        "unknown field in record update: " +
                                            f.node.ident);
                }
            }
          }
        }
      }
      ast::expr_field(base, field, tys) {
        bot |= check_expr(fcx, base);
        let expr_t = structurally_resolved_type(fcx, expr.span,
                                                fcx.expr_ty(base));
        let base_t = do_autoderef(fcx, expr.span, expr_t);
        let mut handled = false;
        let n_tys = vec::len(tys);
        alt structure_of(fcx, expr.span, base_t) {
          ty::ty_rec(fields) {
            alt ty::field_idx(field, fields) {
              some(ix) {
                if n_tys > 0u {
                    tcx.sess.span_err(expr.span,
                                      "can't provide type parameters \
                                       to a field access");
                }
                fcx.write_ty(id, fields[ix].mt.ty);
                handled = true;
              }
              _ {}
            }
          }
          ty::ty_class(base_id, params) {
              // This is just for fields -- the same code handles
              // methods in both classes and ifaces

              // (1) verify that the class id actually has a field called
              // field
              #debug("class named %s", ty_to_str(tcx, base_t));
              /*
                check whether this is a self-reference or not, which
                determines whether we look at all fields or only public
                ones
               */
              let cls_items = if self_ref(fcx, base.id) {
                  // base expr is "self" -- consider all fields
                  ty::lookup_class_fields(tcx, base_id)
              }
              else {
                  lookup_public_fields(tcx, base_id)
              };
              alt lookup_field_ty(tcx, base_id, cls_items, field, params) {
                 some(field_ty) {
                    // (2) look up what field's type is, and return it
                    // FIXME: actually instantiate any type params
                     fcx.write_ty(id, field_ty);
                     handled = true;
                 }
                 none {}
              }
          }
          _ {}
        }
        if !handled {
            let tps = vec::map(tys, {|ty| ast_ty_to_ty_crate(fcx.ccx, ty)});
            alt lookup_method(fcx, expr, expr.id, field, expr_t, tps,
                              self_ref(fcx, base.id)) {
              some(origin) {
                fcx.ccx.method_map.insert(id, origin);
              }
              none {
                let t_err = resolve_type_vars_if_possible(fcx, expr_t);
                let msg = #fmt["attempted access of field %s on type %s, but \
                          no public field or method with that name was found",
                               field, ty_to_str(tcx, t_err)];
                tcx.sess.span_err(expr.span, msg);
                // NB: Adding a bogus type to allow typechecking to continue
                fcx.write_ty(id, next_ty_var(fcx));
              }
            }
        }
      }
      ast::expr_index(base, idx) {
        bot |= check_expr(fcx, base);
        let raw_base_t = fcx.expr_ty(base);
        let base_t = do_autoderef(fcx, expr.span, raw_base_t);
        bot |= check_expr(fcx, idx);
        let idx_t = fcx.expr_ty(idx);
        alt structure_of(fcx, expr.span, base_t) {
          ty::ty_evec(mt, _) |
          ty::ty_vec(mt) {
            require_integral(fcx, idx.span, idx_t);
            fcx.write_ty(id, mt.ty);
          }
          ty::ty_estr(_) |
          ty::ty_str {
            require_integral(fcx, idx.span, idx_t);
            let typ = ty::mk_mach_uint(tcx, ast::ty_u8);
            fcx.write_ty(id, typ);
          }
          _ {
            let resolved = structurally_resolved_type(fcx, expr.span,
                                                      raw_base_t);
            alt lookup_op_method(fcx, expr, resolved, "[]",
                                 [some(idx)]) {
              some((ret_ty, _)) { fcx.write_ty(id, ret_ty); }
              _ {
                tcx.sess.span_fatal(
                    expr.span, "cannot index a value of type `" +
                    ty_to_str(tcx, base_t) + "`");
              }
            }
          }
        }
      }
      ast::expr_new(p, alloc_id, v) {
        bot |= check_expr(fcx, p);
        bot |= check_expr(fcx, v);

        let p_ty = fcx.expr_ty(p);

        alt lookup_method(fcx, p, alloc_id, "alloc", p_ty, [], false) {
          some(origin) {
            fcx.ccx.method_map.insert(alloc_id, origin);

            // Check that the alloc() method has the expected type, which
            // should be fn(sz: uint, align: uint) -> *().
            let expected_ty = {
                let ty_uint = ty::mk_uint(tcx);
                let ty_nilp = ty::mk_ptr(tcx, {ty: ty::mk_nil(tcx),
                                              mutbl: ast::m_imm});
                let m = ast::expl(ty::default_arg_mode_for_ty(ty_uint));
                ty::mk_fn(tcx, {proto: ast::proto_any,
                                inputs: [{mode: m, ty: ty_uint},
                                         {mode: m, ty: ty_uint}],
                                output: ty_nilp,
                                ret_style: ast::return_val,
                                constraints: []})
            };

            demand::simple(fcx, expr.span,
                           expected_ty, fcx.node_ty(alloc_id));
          }

          none {
            let t_err = resolve_type_vars_if_possible(fcx, p_ty);
            let msg = #fmt["no `alloc()` method found for type `%s`",
                           ty_to_str(tcx, t_err)];
            tcx.sess.span_err(expr.span, msg);
          }
        }

        // The region value must have a type like &r.T.  The resulting
        // memory will be allocated into the region `r`.
        let pool_region = region_of(fcx, p);
        let v_ty = fcx.expr_ty(v);
        let res_ty = ty::mk_rptr(tcx, pool_region, {ty: v_ty,
                                                    mutbl: ast::m_imm});
        fcx.write_ty(expr.id, res_ty);
      }
    }
    if bot { fcx.write_bot(expr.id); }

    #debug("type of expr %s is %s, expected is %s",
           syntax::print::pprust::expr_to_str(expr),
           ty_to_str(tcx, fcx.expr_ty(expr)),
           ty_to_str(tcx, expected));

    unify(fcx, expr.span, expected, fcx.expr_ty(expr));

    #debug("<< bot=%b", bot);
    ret bot;
}

fn require_integral(fcx: @fn_ctxt, sp: span, t: ty::t) {
    if !type_is_integral(fcx, sp, t) {
        fcx.ccx.tcx.sess.span_err(sp, "mismatched types: expected \
                                       `integer` but found `"
                                  + fcx.ty_to_str(t) + "`");
    }
}

fn next_ty_var_id(fcx: @fn_ctxt) -> ty_vid {
    let id = *fcx.next_var_id;
    *fcx.next_var_id += 1u;
    ret ty_vid(id);
}

fn next_ty_var(fcx: @fn_ctxt) -> ty::t {
    ret ty::mk_var(fcx.ccx.tcx, next_ty_var_id(fcx));
}

fn next_region_var_id(fcx: @fn_ctxt) -> region_vid {
    let id = *fcx.next_region_var_id;
    *fcx.next_region_var_id += 1u;
    ret region_vid(id);
}

fn next_region_var(fcx: @fn_ctxt) -> ty::region {
    ret ty::re_var(next_region_var_id(fcx));
}

fn bind_params(fcx: @fn_ctxt, tp: ty::t, count: uint)
    -> {vars: [ty::t], ty: ty::t} {
    let vars = vec::from_fn(count, {|_i| next_ty_var(fcx)});
    {vars: vars, ty: ty::substitute_type_params(fcx.ccx.tcx, vars, tp)}
}

fn check_decl_initializer(fcx: @fn_ctxt, nid: ast::node_id,
                          init: ast::initializer) -> bool {
    let lty = ty::mk_var(fcx.ccx.tcx, lookup_local(fcx, init.expr.span, nid));
    ret check_expr_with(fcx, init.expr, lty);
}

fn check_decl_local(fcx: @fn_ctxt, local: @ast::local) -> bool {
    let mut bot = false;

    let t = ty::mk_var(fcx.ccx.tcx, fcx.locals.get(local.node.id));
    fcx.write_ty(local.node.id, t);
    alt local.node.init {
      some(init) {
        bot = check_decl_initializer(fcx, local.node.id, init);
      }
      _ {/* fall through */ }
    }

    let region = fcx.ccx.tcx.region_map.rvalue_to_region.get(local.node.id);
    let pcx = {
        fcx: fcx,
        map: pat_util::pat_id_map(fcx.ccx.tcx.def_map, local.node.pat),
        alt_region: region,
        block_region: region,
        pat_region: region
    };

    check_pat(pcx, local.node.pat, t);
    ret bot;
}

fn check_stmt(fcx: @fn_ctxt, stmt: @ast::stmt) -> bool {
    let mut node_id;
    let mut bot = false;
    alt stmt.node {
      ast::stmt_decl(decl, id) {
        node_id = id;
        alt decl.node {
          ast::decl_local(ls) {
            for ls.each {|l| bot |= check_decl_local(fcx, l); }
          }
          ast::decl_item(_) {/* ignore for now */ }
        }
      }
      ast::stmt_expr(expr, id) {
        node_id = id;
        bot = check_expr_with(fcx, expr, ty::mk_nil(fcx.ccx.tcx));
      }
      ast::stmt_semi(expr, id) {
        node_id = id;
        bot = check_expr(fcx, expr);
      }
    }
    fcx.write_nil(node_id);
    ret bot;
}

fn check_block_no_value(fcx: @fn_ctxt, blk: ast::blk) -> bool {
    let bot = check_block(fcx, blk);
    if !bot {
        let blkty = fcx.node_ty(blk.node.id);
        let nilty = ty::mk_nil(fcx.ccx.tcx);
        demand::simple(fcx, blk.span, nilty, blkty);
    }
    ret bot;
}

fn check_block(fcx0: @fn_ctxt, blk: ast::blk) -> bool {
    let fcx = alt blk.node.rules {
      ast::unchecked_blk { @{purity: ast::impure_fn with *fcx0} }
      ast::unsafe_blk { @{purity: ast::unsafe_fn with *fcx0} }
      ast::default_blk { fcx0 }
    };
    let mut bot = false;
    let mut warned = false;
    for blk.node.stmts.each {|s|
        if bot && !warned &&
               alt s.node {
                 ast::stmt_decl(@{node: ast::decl_local(_), _}, _) |
                 ast::stmt_expr(_, _) | ast::stmt_semi(_, _) {
                   true
                 }
                 _ { false }
               } {
            fcx.ccx.tcx.sess.span_warn(s.span, "unreachable statement");
            warned = true;
        }
        bot |= check_stmt(fcx, s);
    }
    alt blk.node.expr {
      none { fcx.write_nil(blk.node.id); }
      some(e) {
        if bot && !warned {
            fcx.ccx.tcx.sess.span_warn(e.span, "unreachable expression");
        }
        bot |= check_expr(fcx, e);
        let ety = fcx.expr_ty(e);
        fcx.write_ty(blk.node.id, ety);
      }
    }
    if bot {
        fcx.write_bot(blk.node.id);
    }
    ret bot;
}

fn check_const(ccx: @crate_ctxt, _sp: span, e: @ast::expr, id: ast::node_id) {
    // FIXME: this is kinda a kludge; we manufacture a fake function context
    // and statement context for checking the initializer expression.
    let rty = ty::node_id_to_type(ccx.tcx, id);
    let fcx: @fn_ctxt =
        @{self_ty: none,
          ret_ty: rty,
          indirect_ret_ty: none,
          purity: ast::pure_fn,
          proto: ast::proto_box,
          infcx: infer::new_infer_ctxt(ccx.tcx),
          locals: int_hash(),
          next_var_id: @mut 0u,
          next_region_var_id: @mut 0u,
          node_types: smallintmap::mk(),
          node_type_substs: map::int_hash(),
          ccx: ccx};
    check_expr(fcx, e);
    let cty = fcx.expr_ty(e);
    let declty = fcx.ccx.tcx.tcache.get(local_def(id)).ty;
    demand::simple(fcx, e.span, declty, cty);
    writeback::resolve_type_vars_in_expr(fcx, e);
}

fn check_instantiable(tcx: ty::ctxt,
                      sp: span,
                      item_id: ast::node_id) {
    let rty = ty::node_id_to_type(tcx, item_id);
    if !ty::is_instantiable(tcx, rty) {
        tcx.sess.span_err(sp, #fmt["this type cannot be instantiated \
                                    without an instance of itself. \
                                    Consider using option<%s>.",
                                   ty_to_str(tcx, rty)]);
    }
}

fn check_enum_variants(ccx: @crate_ctxt, sp: span, vs: [ast::variant],
                      id: ast::node_id) {
    // FIXME: this is kinda a kludge; we manufacture a fake function context
    // and statement context for checking the initializer expression.
    let rty = ty::node_id_to_type(ccx.tcx, id);
    let fcx: @fn_ctxt =
        @{self_ty: none,
          ret_ty: rty,
          indirect_ret_ty: none,
          purity: ast::pure_fn,
          proto: ast::proto_box,
          infcx: infer::new_infer_ctxt(ccx.tcx),
          locals: int_hash(),
          next_var_id: @mut 0u,
          next_region_var_id: @mut 0u,
          node_types: smallintmap::mk(),
          node_type_substs: map::int_hash(),
          ccx: ccx};
    let mut disr_vals: [int] = [];
    let mut disr_val = 0;
    for vs.each {|v|
        alt v.node.disr_expr {
          some(e) {
            check_expr(fcx, e);
            let cty = fcx.expr_ty(e);
            let declty = ty::mk_int(ccx.tcx);
            demand::simple(fcx, e.span, declty, cty);
            // FIXME: issue #1417
            // Also, check_expr (from check_const pass) doesn't guarantee that
            // the expression in an form that eval_const_expr can handle, so
            // we may still get an internal compiler error
            alt const_eval::eval_const_expr(ccx.tcx, e) {
              const_eval::const_int(val) {
                disr_val = val as int;
              }
              _ {
                ccx.tcx.sess.span_err(e.span,
                                      "expected signed integer constant");
              }
            }
          }
          _ {}
        }
        if vec::contains(disr_vals, disr_val) {
            ccx.tcx.sess.span_err(v.span,
                                  "discriminator value already exists.");
        }
        disr_vals += [disr_val];
        disr_val += 1;
    }

    // Check that it is possible to represent this enum:
    let mut outer = true, did = local_def(id);
    if ty::type_structurally_contains(ccx.tcx, rty, {|sty|
        alt sty {
          ty::ty_enum(id, _) if id == did {
            if outer { outer = false; false }
            else { true }
          }
          _ { false }
        }
    }) {
        ccx.tcx.sess.span_err(sp, "illegal recursive enum type. \
                                   wrap the inner value in a box to \
                                   make it represenable");
    }

    // Check that it is possible to instantiate this enum:
    check_instantiable(ccx.tcx, sp, id);
}

// A generic function for checking the pred in a check
// or if-check
fn check_pred_expr(fcx: @fn_ctxt, e: @ast::expr) -> bool {
    let bot = check_expr_with(fcx, e, ty::mk_bool(fcx.ccx.tcx));

    /* e must be a call expr where all arguments are either
    literals or slots */
    alt e.node {
      ast::expr_call(operator, operands, _) {
        if !ty::is_pred_ty(fcx.expr_ty(operator)) {
            fcx.ccx.tcx.sess.span_err
                (operator.span,
                 "operator in constraint has non-boolean return type");
        }

        alt operator.node {
          ast::expr_path(oper_name) {
            alt fcx.ccx.tcx.def_map.find(operator.id) {
              some(ast::def_fn(_, ast::pure_fn)) {
                // do nothing
              }
              _ {
                fcx.ccx.tcx.sess.span_err(operator.span,
                                            "impure function as operator \
                                             in constraint");
              }
            }
            for operands.each {|operand|
                if !ast_util::is_constraint_arg(operand) {
                    let s =
                        "constraint args must be slot variables or literals";
                    fcx.ccx.tcx.sess.span_err(e.span, s);
                }
            }
          }
          _ {
            let s = "in a constraint, expected the \
                     constraint name to be an explicit name";
            fcx.ccx.tcx.sess.span_err(e.span, s);
          }
        }
      }
      _ { fcx.ccx.tcx.sess.span_err(e.span, "check on non-predicate"); }
    }
    ret bot;
}

fn check_constraints(fcx: @fn_ctxt, cs: [@ast::constr], args: [ast::arg]) {
    let num_args = vec::len(args);
    for cs.each {|c|
        let mut c_args = [];
        for c.node.args.each {|a|
            c_args += [
                 // "base" should not occur in a fn type thing, as of
                 // yet, b/c we don't allow constraints on the return type

                 // Works b/c no higher-order polymorphism
                 /*
                 This is kludgy, and we probably shouldn't be assigning
                 node IDs here, but we're creating exprs that are
                 ephemeral, just for the purposes of typechecking. So
                 that's my justification.
                 */
                 @alt a.node {
                    ast::carg_base {
                      fcx.ccx.tcx.sess.span_bug(a.span,
                                                "check_constraints:\
                    unexpected carg_base");
                    }
                    ast::carg_lit(l) {
                      let tmp_node_id = fcx.ccx.tcx.sess.next_node_id();
                      {id: tmp_node_id, node: ast::expr_lit(l), span: a.span}
                    }
                    ast::carg_ident(i) {
                      if i < num_args {
                          let p: ast::path_ =
                              {global: false,
                               idents: [args[i].ident],
                               types: []};
                          let arg_occ_node_id =
                              fcx.ccx.tcx.sess.next_node_id();
                          fcx.ccx.tcx.def_map.insert
                              (arg_occ_node_id,
                               ast::def_arg(args[i].id, args[i].mode));
                          {id: arg_occ_node_id,
                           node: ast::expr_path(@respan(a.span, p)),
                           span: a.span}
                      } else {
                          fcx.ccx.tcx.sess.span_bug(a.span,
                                                    "check_constraints:\
                     carg_ident index out of bounds");
                      }
                    }
                  }];
        }
        let p_op: ast::expr_ = ast::expr_path(c.node.path);
        let oper: @ast::expr = @{id: c.node.id, node: p_op, span: c.span};
        // Another ephemeral expr
        let call_expr_id = fcx.ccx.tcx.sess.next_node_id();
        let call_expr =
            @{id: call_expr_id,
              node: ast::expr_call(oper, c_args, false),
              span: c.span};
        check_pred_expr(fcx, call_expr);
    }
}

fn check_bare_fn(ccx: @crate_ctxt,
                 decl: ast::fn_decl,
                 body: ast::blk,
                 id: ast::node_id,
                 self_ty: option<ty::t>) {
    let fty = ty::node_id_to_type(ccx.tcx, id);
    let ret_ty = ty::ty_fn_ret(fty);
    let arg_tys = vec::map(ty::ty_fn_args(fty)) {|a| a.ty };
    check_fn(ccx, ast::proto_bare, decl, body, id,
             ret_ty, arg_tys, false, none, self_ty);
}

fn check_fn(ccx: @crate_ctxt,
            proto: ast::proto,
            decl: ast::fn_decl,
            body: ast::blk,
            fid: ast::node_id,
            ret_ty: ty::t,
            arg_tys: [ty::t],
            indirect_ret: bool,
            old_fcx: option<@fn_ctxt>,
            self_ty: option<ty::t>) {

    // See big comment in region.rs.
    let arg_tys = arg_tys.map {|arg_ty|
        replace_bound_regions_with_free_regions(ccx.tcx, fid, arg_ty)
    };
    let ret_ty =
        replace_bound_regions_with_free_regions(ccx.tcx, fid, ret_ty);
    let self_ty = option::map(self_ty) {|st|
        replace_bound_regions_with_free_regions(ccx.tcx, fid, st)
    };

    #debug["check_fn(arg_tys=%?, ret_ty=%?, self_ty=%?)",
           arg_tys.map {|a| ty_to_str(ccx.tcx, a) },
           ty_to_str(ccx.tcx, ret_ty),
           option::map(self_ty) {|st| ty_to_str(ccx.tcx, st) }];

    // If old_fcx is some(...), this is a block fn { |x| ... }.
    // In that case, the purity is inherited from the context.
    let {purity, node_types, node_type_substs} = alt old_fcx {
      none {
        {purity: decl.purity,
         node_types: smallintmap::mk(),
         node_type_substs: map::int_hash()}
      }
      some(f) {
        assert decl.purity == ast::impure_fn;
        {purity: f.purity,
         node_types: f.node_types,
         node_type_substs: f.node_type_substs}
      }
    };

    let gather_result = gather_locals(ccx, decl, body, arg_tys, old_fcx);
    let indirect_ret_ty = if indirect_ret {
        let ofcx = option::get(old_fcx);
        alt ofcx.indirect_ret_ty {
          some(t) { some(t) }
          none { some(ofcx.ret_ty) }
        }
    } else { none };
    let fcx: @fn_ctxt =
        @{self_ty: self_ty,
          ret_ty: ret_ty,
          indirect_ret_ty: indirect_ret_ty,
          purity: purity,
          proto: proto,
          infcx: gather_result.infcx,
          locals: gather_result.locals,
          next_var_id: gather_result.next_var_id,
          next_region_var_id: @mut 0u,
          node_types: node_types,
          node_type_substs: node_type_substs,
          ccx: ccx};

    check_constraints(fcx, decl.constraints, decl.inputs);
    check_block(fcx, body);

    // We unify the tail expr's type with the
    // function result type, if there is a tail expr.
    alt body.node.expr {
      some(tail_expr) {
        let tail_expr_ty = fcx.expr_ty(tail_expr);
        demand::simple(fcx, tail_expr.span, fcx.ret_ty, tail_expr_ty);
      }
      none { }
    }

    let mut i = 0u;
    vec::iter(arg_tys) {|arg|
        fcx.write_ty(decl.inputs[i].id, arg);
        i += 1u;
    }

    // If we don't have any enclosing function scope, it is time to
    // force any remaining type vars to be resolved.
    // If we have an enclosing function scope, our type variables will be
    // resolved when the enclosing scope finishes up.
    if option::is_none(old_fcx) {
        vtable::resolve_in_block(fcx, body);
        writeback::resolve_type_vars_in_fn(fcx, decl, body);
    }
}

fn check_method(ccx: @crate_ctxt, method: @ast::method, self_ty: ty::t) {
    check_bare_fn(ccx, method.decl, method.body, method.id, some(self_ty));
}

fn class_types(ccx: @crate_ctxt, members: [@ast::class_member]) -> class_map {
    let rslt = int_hash::<ty::t>();
    for members.each {|m|
      alt m.node {
         ast::instance_var(_,t,_,id,_) {
           rslt.insert(id, ast_ty_to_ty(ccx.tcx, m_collect, t));
         }
         ast::class_method(mth) {
             rslt.insert(mth.id, ty::mk_fn(ccx.tcx,
                ty_of_method(ccx.tcx, m_collect, mth).fty));
         }
      }
    }
    rslt
}

fn check_class_member(ccx: @crate_ctxt, class_t: ty::t,
                      cm: @ast::class_member) {
    alt cm.node {
      ast::instance_var(_,t,_,_,_) { }
      ast::class_method(m) {
          check_method(ccx, m, class_t);
      }
    }
}

fn check_item(ccx: @crate_ctxt, it: @ast::item) {
    alt it.node {
      ast::item_const(_, e) { check_const(ccx, it.span, e, it.id); }
      ast::item_enum(vs, _) { check_enum_variants(ccx, it.span, vs, it.id); }
      ast::item_fn(decl, tps, body) {
        check_bare_fn(ccx, decl, body, it.id, none);
      }
      ast::item_res(decl, tps, body, dtor_id, _) {
        check_instantiable(ccx.tcx, it.span, it.id);
        check_bare_fn(ccx, decl, body, dtor_id, none);
      }
      ast::item_impl(tps, _, ty, ms) {
        let self_ty = ast_ty_to_ty(ccx.tcx, m_check, ty);
        let self_region = ty::re_free(it.id, ty::br_self);
        let self_ty = replace_self_region(ccx.tcx, self_region, self_ty);
        for ms.each {|m| check_method(ccx, m, self_ty);}
      }
      ast::item_class(tps, members, ctor) {
          let cid = some(it.id);
          let class_t = ty::node_id_to_type(ccx.tcx, it.id);
          let members_info = class_types(ccx, members);
          // can also ditch the enclosing_class stuff once we move to self
          // FIXME
          let class_ccx = @{enclosing_class_id:cid,
                            enclosing_class:members_info with *ccx};
          // typecheck the ctor
          check_bare_fn(class_ccx, ctor.node.dec,
                        ctor.node.body, ctor.node.id,
                        some(class_t));

          // typecheck the members
          for members.each {|m| check_class_member(class_ccx, class_t, m); }
      }
      _ {/* nothing to do */ }
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

fn check_main_fn_ty(tcx: ty::ctxt, main_id: ast::node_id, main_span: span) {
    let main_t = ty::node_id_to_type(tcx, main_id);
    alt ty::get(main_t).struct {
      ty::ty_fn({proto: ast::proto_bare, inputs, output,
                 ret_style: ast::return_val, constraints}) {
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

fn check_for_main_fn(tcx: ty::ctxt, crate: @ast::crate) {
    if !tcx.sess.building_library {
        alt tcx.sess.main_fn {
          some((id, sp)) { check_main_fn_ty(tcx, id, sp); }
          none { tcx.sess.span_err(crate.span, "main function not found"); }
        }
    }
}

mod vtable {
    fn has_iface_bounds(tps: [ty::param_bounds]) -> bool {
        vec::any(tps, {|bs|
            vec::any(*bs, {|b|
                alt b { ty::bound_iface(_) { true } _ { false } }
            })
        })
    }

    fn lookup_vtables(fcx: @fn_ctxt, isc: resolve::iscopes, sp: span,
                      bounds: @[ty::param_bounds], tys: [ty::t],
                      allow_unsafe: bool) -> vtable_res {
        let tcx = fcx.ccx.tcx;
        let mut result = [], i = 0u;
        for tys.each {|ty|
            for vec::each(*bounds[i]) {|bound|
                alt bound {
                  ty::bound_iface(i_ty) {
                    let i_ty = ty::substitute_type_params(tcx, tys, i_ty);
                    result += [lookup_vtable(fcx, isc, sp, ty, i_ty,
                                             allow_unsafe)];
                  }
                  _ {}
                }
            }
            i += 1u;
        }
        @result
    }

    fn lookup_vtable(fcx: @fn_ctxt, isc: resolve::iscopes, sp: span,
                     ty: ty::t, iface_ty: ty::t, allow_unsafe: bool)
        -> vtable_origin {
        let tcx = fcx.ccx.tcx;
        let (iface_id, iface_tps) = alt check ty::get(iface_ty).struct {
            ty::ty_iface(did, tps) { (did, tps) }
        };
        let ty = fixup_ty(fcx, sp, ty);
        alt ty::get(ty).struct {
          ty::ty_param(n, did) {
            let mut n_bound = 0u;
            for vec::each(*tcx.ty_param_bounds.get(did.node)) {|bound|
                alt bound {
                  ty::bound_iface(ity) {
                    alt check ty::get(ity).struct {
                      ty::ty_iface(idid, _) {
                        if iface_id == idid { ret vtable_param(n, n_bound); }
                      }
                    }
                    n_bound += 1u;
                  }
                  _ {}
                }
            }
          }
          ty::ty_iface(did, tps) if iface_id == did {
            if !allow_unsafe {
                for vec::each(*ty::iface_methods(tcx, did)) {|m|
                    if ty::type_has_vars(ty::mk_fn(tcx, m.fty)) {
                        tcx.sess.span_err(
                            sp, "a boxed iface with self types may not be \
                                 passed as a bounded type");
                    } else if (*m.tps).len() > 0u {
                        tcx.sess.span_err(
                            sp, "a boxed iface with generic methods may not \
                                 be passed as a bounded type");

                    }
                }
            }
            ret vtable_iface(did, tps);
          }
          _ {
            let mut found = none;
            std::list::iter(isc) {|impls|
                if option::is_none(found) {
                    for vec::each(*impls) {|im|
                        let match = alt ty::impl_iface(tcx, im.did) {
                          some(ity) {
                            alt check ty::get(ity).struct {
                              ty::ty_iface(id, _) { id == iface_id }
                            }
                          }
                          _ { false }
                        };
                        if match {
                            let {n_tps, ty: self_ty} =
                                impl_self_ty(tcx, im.did);
                            let {vars, ty: self_ty} = if n_tps > 0u {
                                bind_params(fcx, self_ty, n_tps)
                            } else { {vars: [], ty: self_ty} };
                            let im_bs =
                                ty::lookup_item_type(tcx, im.did).bounds;
                            alt unify::unify(fcx, ty, self_ty) {
                              result::ok(_) {
                                if option::is_some(found) {
                                    tcx.sess.span_err(
                                        sp, "multiple applicable implemen\
                                             tations in scope");
                                } else {
                                    connect_iface_tps(fcx, sp, vars,
                                                      iface_tps, im.did);
                                    let params = vec::map(vars, {|t|
                                        fixup_ty(fcx, sp, t)});
                                    let subres = lookup_vtables(
                                        fcx, isc, sp, im_bs, params, false);
                                    found = some(vtable_static(im.did, params,
                                                               subres));
                                }
                              }
                              result::err(_) {}
                            }
                        }
                    }
                }
            }
            alt found {
              some(rslt) { ret rslt; }
              _ {}
            }
          }
        }

        tcx.sess.span_fatal(
            sp, "failed to find an implementation of interface " +
            ty_to_str(tcx, iface_ty) + " for " +
            ty_to_str(tcx, ty));
    }

    fn fixup_ty(fcx: @fn_ctxt, sp: span, ty: ty::t) -> ty::t {
        let tcx = fcx.ccx.tcx;
        alt infer::fixup_vars(fcx.infcx, ty) {
          result::ok(new_type) { new_type }
          result::err(e) {
            tcx.sess.span_fatal(
                sp,
                #fmt["cannot determine a type \
                      for this bounded type parameter: %s",
                     infer::fixup_err_to_str(e)])
          }
        }
    }

    fn connect_iface_tps(fcx: @fn_ctxt, sp: span, impl_tys: [ty::t],
                         iface_tys: [ty::t], impl_did: ast::def_id) {
        let tcx = fcx.ccx.tcx;
        let ity = option::get(ty::impl_iface(tcx, impl_did));
        let iface_ty = ty::substitute_type_params(tcx, impl_tys, ity);
        alt check ty::get(iface_ty).struct {
          ty::ty_iface(_, tps) {
            vec::iter2(tps, iface_tys,
                       {|a, b| demand::simple(fcx, sp, a, b);});
          }
        }
    }

    fn resolve_expr(ex: @ast::expr, &&fcx: @fn_ctxt, v: visit::vt<@fn_ctxt>) {
        let cx = fcx.ccx;
        alt ex.node {
          ast::expr_path(_) {
            alt fcx.opt_node_ty_substs(ex.id) {
              some(ts) {
                let did = ast_util::def_id_of_def(cx.tcx.def_map.get(ex.id));
                let item_ty = ty::lookup_item_type(cx.tcx, did);
                if has_iface_bounds(*item_ty.bounds) {
                    let impls = cx.impl_map.get(ex.id);
                    cx.vtable_map.insert(ex.id, lookup_vtables(
                        fcx, impls, ex.span, item_ty.bounds, ts, false));
                }
              }
              _ {}
            }
          }
          // Must resolve bounds on methods with bounded params
          ast::expr_field(_, _, _) | ast::expr_binary(_, _, _) |
          ast::expr_unary(_, _) | ast::expr_assign_op(_, _, _) |
          ast::expr_index(_, _) {
            alt cx.method_map.find(ex.id) {
              some(method_static(did)) {
                let bounds = ty::lookup_item_type(cx.tcx, did).bounds;
                if has_iface_bounds(*bounds) {
                    let callee_id = alt ex.node {
                      ast::expr_field(_, _, _) { ex.id }
                      _ { ast_util::op_expr_callee_id(ex) }
                    };
                    let ts = fcx.node_ty_substs(callee_id);
                    let iscs = cx.impl_map.get(ex.id);
                    cx.vtable_map.insert(callee_id, lookup_vtables(
                        fcx, iscs, ex.span, bounds, ts, false));
                }
              }
              _ {}
            }
          }
          ast::expr_cast(src, _) {
            let target_ty = fcx.expr_ty(ex);
            alt ty::get(target_ty).struct {
              ty::ty_iface(_, _) {
                let impls = cx.impl_map.get(ex.id);
                let vtable = lookup_vtable(fcx, impls, ex.span,
                                           fcx.expr_ty(src), target_ty,
                                           true);
                cx.vtable_map.insert(ex.id, @[vtable]);
              }
              _ {}
            }
          }
          _ {}
        }
        visit::visit_expr(ex, fcx, v);
    }

    // Detect points where an interface-bounded type parameter is
    // instantiated, resolve the impls for the parameters.
    fn resolve_in_block(fcx: @fn_ctxt, bl: ast::blk) {
        visit::visit_block(bl, fcx, visit::mk_vt(@{
            visit_expr: resolve_expr,
            visit_item: fn@(_i: @ast::item, &&_e: @fn_ctxt,
                            _v: visit::vt<@fn_ctxt>) {}
            with *visit::default_visitor()
        }));
    }
}

fn check_crate(tcx: ty::ctxt, impl_map: resolve::impl_map,
               crate: @ast::crate) -> (method_map, vtable_map) {
    collect::collect_item_types(tcx, crate);

    let ccx = @{impl_map: impl_map,
                method_map: std::map::int_hash(),
                vtable_map: std::map::int_hash(),
                enclosing_class_id: none,
                enclosing_class: std::map::int_hash(),
                tcx: tcx};
    let visit = visit::mk_simple_visitor(@{
        visit_item: bind check_item(ccx, _)
        with *visit::default_simple_visitor()
    });
    visit::visit_crate(*crate, (), visit);
    check_for_main_fn(tcx, crate);
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
