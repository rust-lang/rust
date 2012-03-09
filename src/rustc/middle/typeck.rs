import result::result;
import std::{map, list};
import syntax::{ast, ast_util};
import ast::spanned;
import syntax::ast_util::{local_def, respan};
import syntax::visit;
import metadata::csearch;
import driver::session::session;
import util::common::*;
import syntax::codemap::span;
import pat_util::*;
import middle::ty;
import middle::ty::{arg, field, node_type_table, mk_nil,
                    ty_param_bounds_and_ty, lookup_class_item_tys,
                    ty_ops};
import util::ppaux::{ty_to_str, ty_i_to_str};
import std::smallintmap;
import std::map::{hashmap, int_hash};
import std::serialization::{serialize_uint, deserialize_uint};
import syntax::print::pprust::*;
import ty::unify::uctxt;

export check_crate;
export method_map;
export method_origin, serialize_method_origin, deserialize_method_origin;
export vtable_map;
export vtable_res;
export vtable_origin;
export vtable_origin_base;

#[auto_serialize]
enum method_origin {
    method_static(ast::def_id),
    // iface id, method num, param num, bound num
    method_param(ast::def_id, uint, uint, uint),
    method_iface(ast::def_id, uint),
}
type method_map = hashmap<ast::node_id, method_origin>;

// Resolutions for bounds of all parameters, left to right, for a given path.
enum vtable_origin_base<T> {
    vtable_static(ast::def_id, [T], vtable_res),
    // Param number, bound number
    vtable_param(uint, uint),
    vtable_iface(ast::def_id, [ty::t]),
}

type vtable_origin_i = vtable_origin_base<ty::t_i>;
type vtable_res_i = @[vtable_origin_i];
type vtable_map_i = hashmap<ast::node_id, vtable_res_i>;

type vtable_origin = vtable_origin_base<ty::t>;
type vtable_res = @[vtable_origin];
type vtable_map = hashmap<ast::node_id, vtable_res>;

type ty_table = hashmap<ast::def_id, ty::t>;

// Used for typechecking the methods of an impl
enum self_info { self_impl(ty::t) }

enum ast_ty_to_ty_cache_entry {
    atttce_unresolved,        /* not resolved yet */
    atttce_resolved(ty::t),   /* resolved to a type, irrespective of region */
    atttce_has_regions        /* has regions; cannot be cached */
}

type crate_ctxt = {mutable self_infos: [self_info],
                   impl_map: resolve::impl_map,
                   method_map: method_map,
                   vtable_map: vtable_map,
                   // Not at all sure it's right to put these here
                   /* node_id for the class this fn is in --
                      none if it's not in a class */
                   enclosing_class_id: option<ast::node_id>,
                   /* map from node_ids for enclosing-class
                      vars and methods to types */
                   enclosing_class: class_map,
                   ast_ty_to_ty_cache:
                       hashmap<@ast::ty, ast_ty_to_ty_cache_entry>,
                   tcx: ty::ctxt};

type class_map = hashmap<ast::node_id, ty::t>;

type fn_ctxt =
    // var_bindings, locals and next_var_id are shared
    // with any nested functions that capture the environment
    // (and with any functions whose environment is being captured).
    {ret_ty: ty::t_i,
     purity: ast::purity,
     proto: ast::proto,
     vb: @ty::var_bindings,
     locals: hashmap<ast::node_id, int>,
     next_var_id: @mutable int,
     ccx: @crate_ctxt};

fn ty_str(fcx: @fn_ctxt, t: ty::t_i) -> str {
    ty_i_to_str(fcx.vb, t)
}

fn node_ty(fcx: @fn_ctxt, id: ast::node_id) -> ty::t_i {
    fcx.vb.node_types.get(id)
}

fn expr_ty(fcx: @fn_ctxt, expr: @ast::expr) -> ty::t_i {
    node_ty(fcx, expr.id)
}

fn node_ty_substs(fcx: @fn_ctxt, id: ast::node_id) -> [ty::t_i] {
    fcx.vb.node_type_substs.get(id)
}

fn node_ty_substs_find(fcx: @fn_ctxt, id: ast::node_id) -> option<[ty::t_i]> {
    fcx.vb.node_type_substs.find(id)
}

fn lookup_local(fcx: @fn_ctxt, sp: span, id: ast::node_id) -> int {
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

// Instantiates the given path, which must refer to an item with the given
// number of type parameters and type.
fn write_path_ty(fcx: @fn_ctxt, sp: span, pth: @ast::path, id: ast::node_id) {

    fn substd(fcx: @fn_ctxt, sp: span, pth: @ast::path, id: ast::node_id,
              tpt: ty_param_bounds_and_ty) {
        let ty_param_count = vec::len(*tpt.bounds);
        let ty_substs_len = vec::len(pth.node.types);
        let substs = if ty_substs_len > 0u {
            if ty_param_count == 0u {
                fcx.ccx.tcx.sess.span_err
                    (sp, "this item does not take type parameters");
                next_ty_vars(fcx, ty_param_count)
            } else if ty_substs_len > ty_param_count {
                fcx.ccx.tcx.sess.span_err
                    (sp, "too many type parameter provided for this item");
                next_ty_vars(fcx, ty_param_count)
            } else if ty_substs_len < ty_param_count {
                fcx.ccx.tcx.sess.span_err
                    (sp, "not enough type parameters provided for this item");
                next_ty_vars(fcx, ty_param_count)
            } else {
                vec::map(pth.node.types) {|aty| ast_ty_to_ty_i(fcx, aty) }
            }
        } else {
            next_ty_vars(fcx, ty_param_count)
        };
        write_ty_substs(fcx, id, tpt.ty, substs);
    }

    fn from_defn(fcx: @fn_ctxt, sp: span, pth: @ast::path, id: ast::node_id,
                 defn: ast::def) {

        alt defn {
          ast::def_arg(nid, _) {
            assert (fcx.locals.contains_key(nid));
            let typ = ty::mk_var(lookup_local(fcx, sp, nid));
            ret write_ty(fcx, id, typ);
          }
          ast::def_local(nid, _) {
            assert (fcx.locals.contains_key(nid));
            let typ = ty::mk_var(lookup_local(fcx, sp, nid));
            ret write_ty(fcx, id, typ);
          }
          ast::def_self(_) {
            alt get_self_info(fcx.ccx) {
              some(self_impl(impl_t)) {
                ret substd(fcx, sp, pth, id,
                           {bounds: @[], ty: impl_t});
              }
              none {
                fcx.ccx.tcx.sess.span_bug(sp, "def_self with no self_info");
              }
            }
          }
          ast::def_fn(_, ast::crust_fn) {
            // Crust functions are just u8 pointers
            let t = ty::mk_ptr(fcx.vb, {ty: ty::mk_mach_uint(fcx.vb, ast::ty_u8),
                                        mutbl: ast::m_imm});
            ret write_ty(fcx, id, t);
          }
          ast::def_fn(did, _) | ast::def_const(did) |
          ast::def_variant(_, did) | ast::def_class(did) {
            ret substd(fcx, sp, pth, id,
                       ty::lookup_item_type(fcx.ccx.tcx, did));
          }
          ast::def_binding(nid) {
            assert (fcx.locals.contains_key(nid));
            ret write_ty(fcx, id, ty::mk_var(lookup_local(fcx, sp, nid)));
          }
          ast::def_ty(_) | ast::def_prim_ty(_) {
            fcx.ccx.tcx.sess.span_fatal(sp, "expected value but found type");
          }
          ast::def_upvar(_, inner, _) {
            ret from_defn(fcx, sp, pth, id, *inner);
          }
          ast::def_class_method(_, did) | ast::def_class_field(_, did) {
            if did.crate != ast::local_crate {
                fcx.ccx.tcx.sess.span_fatal(sp,
                                            "class method or field referred to \
                                             out of scope");
            }
            alt fcx.ccx.enclosing_class.find(did.node) {
              some(a_ty) { ret substd(fcx, sp, pth, id,
                                      {bounds: @[], ty: a_ty}); }
              _ { fcx.ccx.tcx.sess.span_fatal(sp,
                                              "class method or field referred to \
                                               out of scope"); }
            }
          }

          _ {
            // FIXME: handle other names.
            fcx.ccx.tcx.sess.unimpl("definition variant");
          }
        }
    }

    let defn = lookup_def(fcx, pth.span, id);
    from_defn(fcx, sp, pth, id, defn);
}

// Returns the one-level-deep structure of the given type.
fn structure_of<R>(fcx: @fn_ctxt, sp: span, typ: ty::t_i,
                   f: fn(ty::sty_i) -> R) -> R {
  alt *typ {
    ty::ty_var_i(vid) {
        ty::unify::get_var_binding(
            fcx.vb, vid,
            {|_vid| /* ...if unbound */
                fcx.ccx.tcx.sess.span_fatal
                    (sp, "the type of this value must \
                          be known in this context");
            },
            f /* ...if bound */)
    }

    ty::sty_i(sty) {
        f(sty)
    }
  }
}

fn fn_args(fcx: @fn_ctxt, sp: span, fty: ty::t_i) -> [ty::arg_i] {
    structure_of(fcx, sp, fty) {|sty|
        alt sty {
          ty::ty_fn(f) { f.inputs }
          _ { fcx.ccx.tcx.sess.span_err(sp, "calling non-function"); [] }
        }
    }
}

fn fn_ret(fcx: @fn_ctxt, sp: span, fty: ty::t_i, &bot: bool) -> ty::t_i {
    structure_of(fcx, sp, fty) {|sty|
        alt sty {
          ty::ty_fn(f) {
            if f.ret_style == ast::noreturn { bot = true; }
            f.output
          }
          _ { fcx.ccx.tcx.sess.span_fatal(sp, "calling non-function"); }
        }
    }
}

// Just checks whether it's a fn that returns bool,
// not its purity.
fn type_is_pred_ty(fcx: @fn_ctxt, sp: span, typ: ty::t_i) -> bool {
    structure_of(fcx, sp, typ) {|sty|
        alt sty {
          ty::ty_fn(f) { type_is_bool(fcx, sp, f.output) }
          _ { false }
        }
    }
}

fn type_is_bool(fcx: @fn_ctxt, sp: span, typ: ty::t_i) -> bool {
    structure_of(fcx, sp, typ, ty::sty_is_bool(_))
}

fn type_is_integral(fcx: @fn_ctxt, sp: span, typ: ty::t_i) -> bool {
    structure_of(fcx, sp, typ, ty::sty_is_integral(_))
}

fn type_is_numeric(fcx: @fn_ctxt, sp: span, typ: ty::t_i) -> bool {
    structure_of(fcx, sp, typ, ty::sty_is_numeric(_))
}

fn type_is_nil(fcx: @fn_ctxt, sp: span, typ: ty::t_i) -> bool {
    structure_of(fcx, sp, typ) {|sty| sty == ty::ty_nil }
}

fn type_is_scalar(fcx: @fn_ctxt, sp: span, typ: ty::t_i) -> bool {
    structure_of(fcx, sp, typ, ty::sty_is_scalar(_))
}

fn type_is_c_like_enum(fcx: @fn_ctxt, sp: span, typ: ty::t_i) -> bool {
    structure_of(fcx, sp, typ, ty::sty_is_c_like_enum(fcx.ccx.tcx, _))
}

fn type_is_bot(fcx: @fn_ctxt, sp: span, typ: ty::t_i) -> bool {
    structure_of(fcx, sp, typ) {|sty|
        alt sty {
          ty::ty_bot { true }
          _ { false }
        }
    }
}

enum mode { m_collect, m_check }

// Parses the programmer's textual representation of a type into our
// internal notion of a type. `getter` is a function that returns the type
// corresponding to a definition ID:
fn ast_ty_to_ty(ccx: @crate_ctxt, mode: mode, &&ast_ty: @ast::ty) -> ty::t {

    fn subst_inferred_regions(ccx: @crate_ctxt, use_site: ast::node_id,
                              ty: ty::t) -> ty::t {
        let tcx = ccx.tcx;
        ret ty::fold_rptr(tcx, ty) {|r|
            alt r {
                ty::re_inferred | ty::re_self(_) {
                    tcx.region_map.ast_type_to_inferred_region.get(use_site)
                }
                _ { r }
            }
        };
    }

    fn getter(ccx: @crate_ctxt, use_site: ast::node_id, mode: mode,
              id: ast::def_id) -> ty::ty_param_bounds_and_ty {
        let tcx = ccx.tcx;
        let tpt = alt mode {
          m_check {
            ty::lookup_item_type(tcx, id)
          }
          m_collect {
            if id.crate != ast::local_crate {
                csearch::get_type(tcx, id)
            } else {
                alt tcx.items.find(id.node) {
                  some(ast_map::node_item(item, _)) {
                    ty_of_item(ccx, mode, item)
                  }
                  some(ast_map::node_native_item(native_item, _, _)) {
                    ty_of_native_item(ccx, mode, native_item)
                  }
                  _ {
                    tcx.sess.bug("unexpected sort of item in ast_ty_to_ty");
                  }
                }
            }
          }
        };

        if ty::type_has_rptrs(tpt.ty) {
            ret {bounds: tpt.bounds,
                 ty: subst_inferred_regions(ccx, use_site, tpt.ty)};
        }
        ret tpt;
    }

    fn ast_mt_to_mt(ccx: @crate_ctxt, use_site: ast::node_id, mode: mode,
                    mt: ast::mt) -> ty::mt {
        ret {ty: do_ast_ty_to_ty(ccx, use_site, mode, mt.ty),
             mutbl: mt.mutbl};
    }
    fn instantiate(ccx: @crate_ctxt, use_site: ast::node_id, sp: span,
                   mode: mode, id: ast::def_id, path_id: ast::node_id,
                   args: [@ast::ty]) -> ty::t {
        let tcx = ccx.tcx;
        let ty_param_bounds_and_ty = getter(ccx, use_site, mode, id);
        if vec::len(*ty_param_bounds_and_ty.bounds) == 0u {
            ret ty_param_bounds_and_ty.ty;
        }

        // The typedef is type-parametric. Do the type substitution.
        let param_bindings: [ty::t] = [];
        if vec::len(args) != vec::len(*ty_param_bounds_and_ty.bounds) {
            tcx.sess.span_fatal(sp, "wrong number of type arguments for a \
                                     polymorphic type");
        }
        for ast_ty: @ast::ty in args {
            param_bindings += [do_ast_ty_to_ty(ccx, use_site, mode, ast_ty)];
        }
        #debug("substituting(%s into %s)",
               str::concat(vec::map(param_bindings, {|t| ty_to_str(tcx, t)})),
               ty_to_str(tcx, ty_param_bounds_and_ty.ty));
        let typ =
            ty::substitute_type_params(ccx.tcx, param_bindings,
                                       ty_param_bounds_and_ty.ty);
        write_substs_to_tcx(tcx, path_id, param_bindings);
        ret typ;
    }

    fn do_ast_ty_to_ty(ccx: @crate_ctxt, use_site: ast::node_id,
                       mode: mode, &&ast_ty: @ast::ty) -> ty::t {
        let tcx = ccx.tcx;
        alt ccx.ast_ty_to_ty_cache.find(ast_ty) {
          some(atttce_resolved(ty)) { ret ty; }
          some(atttce_unresolved) {
            tcx.sess.span_fatal(ast_ty.span, "illegal recursive type. \
                                              insert a enum in the cycle, \
                                              if this is desired)");
          }
          some(atttce_has_regions) | none { /* go on */ }
        }

        ccx.ast_ty_to_ty_cache.insert(ast_ty, atttce_unresolved);
        let typ = alt ast_ty.node {
          ast::ty_nil { ty::mk_nil(tcx) }
          ast::ty_bot { ty::mk_bot(tcx) }
          ast::ty_box(mt) {
            ty::mk_box(tcx, ast_mt_to_mt(ccx, use_site, mode, mt))
          }
          ast::ty_uniq(mt) {
            ty::mk_uniq(tcx, ast_mt_to_mt(ccx, use_site, mode, mt))
          }
          ast::ty_vec(mt) {
            ty::mk_vec(tcx, ast_mt_to_mt(ccx, use_site, mode, mt))
          }
          ast::ty_ptr(mt) {
            ty::mk_ptr(tcx, ast_mt_to_mt(ccx, use_site, mode, mt))
          }
          ast::ty_rptr(region, mt) {
            let region = alt region.node {
                ast::re_inferred {
                    let attir = tcx.region_map.ast_type_to_inferred_region;
                    alt attir.find(ast_ty.id) {
                        some(resolved_region) { resolved_region }
                        none { ty::re_inferred }
                    }
                }
                ast::re_named(_) | ast::re_self {
                    tcx.region_map.ast_type_to_region.get(region.id)
                }
            };
            ty::mk_rptr(tcx, region, ast_mt_to_mt(ccx, use_site, mode, mt))
          }
          ast::ty_tup(fields) {
            let flds = vec::map(fields,
                                bind do_ast_ty_to_ty(ccx, use_site, mode, _));
            ty::mk_tup(tcx, flds)
          }
          ast::ty_rec(fields) {
            let flds: [field] = [];
            for f: ast::ty_field in fields {
                let tm = ast_mt_to_mt(ccx, use_site, mode, f.node.mt);
                flds += [{ident: f.node.ident, mt: tm}];
            }
            ty::mk_rec(tcx, flds)
          }
          ast::ty_fn(proto, decl) {
            ty::mk_fn(tcx, ty_of_fn_decl(ccx, mode, proto, decl))
          }
          ast::ty_path(path, id) {
            let a_def = alt tcx.def_map.find(id) {
              none { tcx.sess.span_fatal(ast_ty.span, #fmt("unbound path %s",
                                                       path_to_str(path))); }
              some(d) { d }};
            alt a_def {
              ast::def_ty(did) {
                instantiate(ccx, use_site, ast_ty.span, mode, did,
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
                        do_ast_ty_to_ty(ccx, use_site, mode, ast_ty)
                    }))
                  }
                }
              }
             ast::def_class(class_id) {
              if class_id.crate == ast::local_crate {
                 alt tcx.items.find(class_id.node) {
                   some(ast_map::node_item(
                     @{node: ast::item_class(tps, _, _), _}, _)) {
                        if vec::len(tps) != vec::len(path.node.types) {
                          tcx.sess.span_err(ast_ty.span, "incorrect number \
                            of type parameters to object type");
                        }
                        ty::mk_class(tcx, class_id, vec::map(path.node.types,
                          {|ast_ty| do_ast_ty_to_ty(ccx, use_site, mode, ast_ty)}))
                     }
                   _ {
                      tcx.sess.span_bug(ast_ty.span, #fmt("class id is \
                        unbound in items"));
                   }
                }
              }
              else {
                  getter(ccx, use_site, mode, class_id).ty
              }
             }
             _ {
                tcx.sess.span_fatal(ast_ty.span,
                                    "found type name used as a variable");
              }
            }
          }
          ast::ty_constr(t, cs) {
            let out_cs = [];
            for constr: @ast::ty_constr in cs {
                out_cs += [ty::ast_constr_to_constr(tcx, constr)];
            }
            ty::mk_constr(tcx, do_ast_ty_to_ty(ccx, use_site, mode, t),
                      out_cs)
          }
          ast::ty_infer {
            tcx.sess.span_bug(ast_ty.span,
                              "found `ty_infer` in unexpected place");
          }
          ast::ty_mac(_) {
              tcx.sess.span_bug(ast_ty.span,
                                    "found `ty_mac` in unexpected place");
          }
        };

        if ty::type_has_rptrs(typ) {
            ccx.ast_ty_to_ty_cache.insert(ast_ty, atttce_has_regions);
        } else {
            ccx.ast_ty_to_ty_cache.insert(ast_ty, atttce_resolved(typ));
        }

        ret typ;
    }

    ret do_ast_ty_to_ty(ccx, ast_ty.id, mode, ast_ty);
}

fn ast_ty_to_opt_ty_i(ccx: @crate_ctxt, t: @ast::ty) -> option<ty::t_i> {
    alt t.node {
      ast::ty_infer { none }
      _ { some(ty::ty_to_ty_i(ccx.tcx, ast_ty_to_ty(ccx, m_check, t))) }
    }
}

fn ast_ty_to_ty_i(fcx: @fn_ctxt, t: @ast::ty) -> ty::t_i {
    alt ast_ty_to_opt_ty_i(fcx.ccx, t) {
      none { next_ty_var(fcx) }
      some(r) { r }
    }
}

fn ty_of_item(ccx: @crate_ctxt, mode: mode, it: @ast::item)
    -> ty::ty_param_bounds_and_ty {
    let def_id = local_def(it.id);
    let tcx = ccx.tcx;
    alt tcx.tcache.find(def_id) {
      some(tpt) { ret tpt; }
      _ {}
    }
    alt it.node {
      ast::item_const(t, _) {
        let typ = ast_ty_to_ty(ccx, mode, t);
        let tpt = {bounds: @[], ty: typ};
        tcx.tcache.insert(local_def(it.id), tpt);
        ret tpt;
      }
      ast::item_fn(decl, tps, _) {
        ret ty_of_fn(ccx, mode, decl, tps, local_def(it.id));
      }
      ast::item_ty(t, tps) {
        alt tcx.tcache.find(local_def(it.id)) {
          some(tpt) { ret tpt; }
          none { }
        }
        // Tell ast_ty_to_ty() that we want to perform a recursive
        // call to resolve any named types.
        let tpt = {
            let t0 = ast_ty_to_ty(ccx, mode, t);
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
            {bounds: ty_param_bounds(ccx, mode, tps), ty: t1}
        };
        tcx.tcache.insert(local_def(it.id), tpt);
        ret tpt;
      }
      ast::item_res(decl, tps, _, _, _) {
        let {bounds, params} = mk_ty_params(ccx, tps);
        let t_arg = ty_of_arg(ccx, mode, decl.inputs[0]);
        let t = ty::mk_res(tcx, local_def(it.id), t_arg.ty, params);
        let t_res = {bounds: bounds, ty: t};
        tcx.tcache.insert(local_def(it.id), t_res);
        ret t_res;
      }
      ast::item_enum(_, tps) {
        // Create a new generic polytype.
        let {bounds, params} = mk_ty_params(ccx, tps);
        let t = ty::mk_enum(tcx, local_def(it.id), params);
        let tpt = {bounds: bounds, ty: t};
        tcx.tcache.insert(local_def(it.id), tpt);
        ret tpt;
      }
      ast::item_iface(tps, ms) {
        let {bounds, params} = mk_ty_params(ccx, tps);
        let t = ty::mk_iface(tcx, local_def(it.id), params);
        let tpt = {bounds: bounds, ty: t};
        tcx.tcache.insert(local_def(it.id), tpt);
        ret tpt;
      }
      ast::item_class(tps,_,_) {
          let {bounds,params} = mk_ty_params(ccx, tps);
          let t = ty::mk_class(tcx, local_def(it.id), params);
          let tpt = {bounds: bounds, ty: t};
          tcx.tcache.insert(local_def(it.id), tpt);
          ret tpt;
      }
      ast::item_impl(_, _, _, _) | ast::item_mod(_) |
      ast::item_native_mod(_) { fail; }
    }
}
fn ty_of_native_item(ccx: @crate_ctxt, mode: mode, it: @ast::native_item)
    -> ty::ty_param_bounds_and_ty {
    alt it.node {
      ast::native_item_fn(fn_decl, params) {
        ret ty_of_native_fn_decl(ccx, mode, fn_decl, params,
                                 local_def(it.id));
      }
    }
}

fn ty_of_arg(ccx: @crate_ctxt, mode: mode, a: ast::arg) -> ty::arg {
    fn arg_mode(ccx: @crate_ctxt, m: ast::mode, ty: ty::t) -> ast::mode {
        alt m {
          ast::infer(_) {
            // Unspecified modes in a non-inferred ctx (e.g., item decl)
            // use the default for the type:
            let m1 = ast::expl(ty::default_arg_mode_for_ty(ty));
            result::get(ty::unify_mode(ccx.tcx, m, m1))
          }
          ast::expl(_) { m }
        }
    }

    let ty = ast_ty_to_ty(ccx, mode, a.ty);
    let mode = arg_mode(ccx, a.mode, ty);
    {mode: mode, ty: ty}
}

fn ty_of_fn_decl_base<T>(tcx: ty::ctxt,
                         proto: ast::proto,
                         decl: ast::fn_decl,
                         arg_fn: fn(ast::arg) -> ty::arg_base<T>,
                         t_fn: fn(@ast::ty) -> T) -> ty::fn_ty_base<T> {
    let input_tys = vec::map(decl.inputs, arg_fn);
    let output_ty = t_fn(decl.output);
    let out_constrs = vec::map(decl.constraints) {|c|
        ty::ast_constr_to_constr(tcx, c)
    };
    {proto: proto,
     inputs: input_tys,
     output: output_ty,
     ret_style: decl.cf, constraints: out_constrs}
}

fn ty_of_fn_decl(ccx: @crate_ctxt,
                 mode: mode,
                 proto: ast::proto,
                 decl: ast::fn_decl) -> ty::fn_ty {
    ty_of_fn_decl_base(
        ccx.tcx, proto, decl,
        {|a| ty_of_arg(ccx, mode, a)},
        {|t| ast_ty_to_ty(ccx, mode, t)})
}

fn ty_of_fn(ccx: @crate_ctxt, mode: mode, decl: ast::fn_decl,
            ty_params: [ast::ty_param], def_id: ast::def_id)
    -> ty::ty_param_bounds_and_ty {
    let tcx = ccx.tcx;
    let bounds = ty_param_bounds(ccx, mode, ty_params);
    let tofd = ty_of_fn_decl(ccx, mode, ast::proto_bare, decl);
    let tpt = {bounds: bounds, ty: ty::mk_fn(tcx, tofd)};
    tcx.tcache.insert(def_id, tpt);
    ret tpt;
}
fn ty_of_native_fn_decl(ccx: @crate_ctxt, mode: mode, decl: ast::fn_decl,
                        ty_params: [ast::ty_param], def_id: ast::def_id)
    -> ty::ty_param_bounds_and_ty {
    let tcx = ccx.tcx;
    let input_tys = [], bounds = ty_param_bounds(ccx, mode, ty_params);
    for a: ast::arg in decl.inputs {
        input_tys += [ty_of_arg(ccx, mode, a)];
    }
    let output_ty = ast_ty_to_ty(ccx, mode, decl.output);

    let t_fn = ty::mk_fn(tcx, {proto: ast::proto_bare,
                               inputs: input_tys,
                               output: output_ty,
                               ret_style: ast::return_val,
                               constraints: []});
    let tpt = {bounds: bounds, ty: t_fn};
    tcx.tcache.insert(def_id, tpt);
    ret tpt;
}
fn ty_param_bounds(ccx: @crate_ctxt, mode: mode, params: [ast::ty_param])
    -> @[ty::param_bounds] {
    let tcx = ccx.tcx;
    let result = [];
    for param in params {
        result += [alt tcx.ty_param_bounds.find(param.id) {
          some(bs) { bs }
          none {
            let bounds = [];
            for b in *param.bounds {
                bounds += [alt b {
                  ast::bound_send { ty::bound_send }
                  ast::bound_copy { ty::bound_copy }
                  ast::bound_iface(t) {
                    let ity = ast_ty_to_ty(ccx, mode, t);
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

fn ty_of_method(ccx: @crate_ctxt, mode: mode, m: @ast::method) -> ty::method {
    {ident: m.ident, tps: ty_param_bounds(ccx, mode, m.tps),
     fty: ty_of_fn_decl(ccx, mode, ast::proto_bare, m.decl),
     purity: m.decl.purity}
}

fn ty_of_ty_method(ccx: @crate_ctxt, mode: mode, m: ast::ty_method)
    -> ty::method {
    {ident: m.ident, tps: ty_param_bounds(ccx, mode, m.tps),
     fty: ty_of_fn_decl(ccx, mode, ast::proto_bare, m.decl),
     purity: m.decl.purity}
}

// Functions that write types into the node type table
fn write_ty_to_tcx(tcx: ty::ctxt, node_id: ast::node_id, ty: ty::t) {
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

fn write_ty(fcx: @fn_ctxt, node_id: ast::node_id, ty: ty::t_i) {
    #debug["write_ty for %s = %s",
           ast_map::node_str(fcx.ccx.tcx.items, node_id),
           ty_str(fcx, ty)];
    fcx.vb.node_types.insert(node_id, ty);
}
fn write_ty_substs(fcx: @fn_ctxt,
                   node_id: ast::node_id,
                   t0: ty::t,
                   +substs: [ty::t_i]) {
    let t1 = ty::ty_to_ty_i_subst(fcx.ccx.tcx, t0, substs);
    write_ty(fcx, node_id, t1);
    fcx.vb.node_type_substs.insert(node_id, substs);
}
fn write_nil(fcx: @fn_ctxt, node_id: ast::node_id) {
    write_ty(fcx, node_id, ty::mk_nil(fcx.vb));
}
fn write_bot(fcx: @fn_ctxt, node_id: ast::node_id) {
    write_ty(fcx, node_id, ty::mk_bot(fcx.vb));
}

fn mk_ty_params(ccx: @crate_ctxt, atps: [ast::ty_param])
    -> {bounds: @[ty::param_bounds], params: [ty::t]} {
    let i = 0u, bounds = ty_param_bounds(ccx, m_collect, atps);
    {bounds: bounds,
     params: vec::map(atps, {|atp|
         let t = ty::mk_param(ccx.tcx, i, local_def(atp.id));
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
        let if_fty = ty::mk_fn(tcx, if_m.fty);
        let if_fty = ty::substitute_type_params(tcx, substs, if_fty);
        let if_fty = fixup_self_def(tcx, if_fty, substs, self_ty, impl_tps);

        alt ty::unify::unify(tcx, impl_fty, if_fty) {
          result::err(err) {
            tcx.sess.span_err(sp, "method `" + if_m.ident +
                              "` has an incompatible type: " +
                              ty::type_err_to_str(tcx, err));
            impl_fty
          }
          result::ok(tp) { tp }
        }
    }
}

// These two functions mangle an iface method ty to make its self type conform
// to the self type of a specific impl or bounded type parameter. This is
// rather involved because the type parameters of ifaces and impls are not
// required to line up (an impl can have less or more parameters than the
// iface it implements), so some mangling of the substituted types is
// required.
//
// There are two methods: one is used when performing a method call, and the
// other when checking implementations against the interfaces they implement.
// Both of these routines have similarities, and could potentially be
// collapsed, but they operate on different types and are also more different
// than similar, so I opted to divide them.

fn fixup_self_call(
    fcx: @fn_ctxt,
    span: span,
    mty: ty::t_i,
    m_substs: [ty::t_i],
    self_ty: ty::t_i) -> ty::t_i {

    if !ty::ty_i_has_self(mty) { ret mty; }

    ty::fold(fcx.vb, mty) {|foldt|
        structure_of(fcx, span, foldt) {|sty|
            alt sty {
              ty::ty_self(tps) {
                // Move the substs into the type param system of the
                // context.
                let tps1 = vec::map(tps) {|t|
                    let f = fixup_self_call(
                        fcx, span, t, m_substs, self_ty);
                    ty::substitute_type_params_i(
                        fcx.vb, m_substs, f)
                };

                // Simply ensure that the type parameters for the self
                // type match the context.
                demand::tys(fcx.vb, span, demand::ek_mismatched_types,
                            tps1, m_substs);
                self_ty
              }
              _ { foldt }
            }
        }
    }
}

fn fixup_self_def(
    tcx: ty::ctxt,
    mty: ty::t,
    m_substs: [ty::t],
    self_ty: ty::t,
    impl_n_tps: uint) -> ty::t {

    if !ty::type_has_self(mty) { ret mty; }

    ty::fold(tcx, mty) {|t|
        alt ty::get(t).struct {
          ty::ty_self(tps) {
            // Move the substs into the type param system of the
            // context.
            let substs = vec::map(tps) {|t|
                let f = fixup_self_def(tcx, t, m_substs, self_ty, impl_n_tps);
                ty::substitute_type_params(tcx, m_substs, f)
            };

            // Add extra substs for impl type parameters.
            while vec::len(substs) < impl_n_tps {
                substs += [ty::mk_param(tcx, vec::len(substs),
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
            ty::substitute_type_params(tcx, substs, self_ty)
          }
          _ { t }
        }
    }
}

// Mangles an iface method ty to instantiate its `self` region.
fn fixup_self_region_in_method_ty(fcx: @fn_ctxt, mty: ty::t_i,
                                  self_expr: @ast::expr) -> ty::t_i {
    let self_region = region_of(fcx, self_expr);
    ty::fold_rptr(fcx.vb, mty) {|r|
        alt r {
            ty::re_self(_) { self_region }
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
    fn get_enum_variant_types(ccx: @crate_ctxt, enum_ty: ty::t,
                              variants: [ast::variant],
                              ty_params: [ast::ty_param]) {
        let tcx = ccx.tcx;

        // Create a set of parameter types shared among all the variants.
        for variant in variants {
            // Nullary enum constructors get turned into constants; n-ary enum
            // constructors get turned into functions.
            let result_ty = if vec::len(variant.node.args) == 0u {
                enum_ty
            } else {
                // As above, tell ast_ty_to_ty() that trans_ty_item_to_ty()
                // should be called to resolve named types.
                let args: [arg] = [];
                for va: ast::variant_arg in variant.node.args {
                    let arg_ty = ast_ty_to_ty(ccx, m_collect, va.ty);
                    args += [{mode: ast::expl(ast::by_copy), ty: arg_ty}];
                }
                // FIXME: this will be different for constrained types
                ty::mk_fn(tcx,
                          {proto: ast::proto_box,
                           inputs: args, output: enum_ty,
                           ret_style: ast::return_val, constraints: []})
            };
            let tpt = {bounds: ty_param_bounds(ccx, m_collect, ty_params),
                       ty: result_ty};
            tcx.tcache.insert(local_def(variant.node.id), tpt);
            write_ty_to_tcx(tcx, variant.node.id, result_ty);
        }
    }
    fn ensure_iface_methods(ccx: @crate_ctxt, id: ast::node_id) {
        alt check ccx.tcx.items.get(id) {
          ast_map::node_item(@{node: ast::item_iface(_, ms), _}, _) {
            ty::store_iface_methods(ccx.tcx, id, @vec::map(ms, {|m|
                ty_of_ty_method(ccx, m_collect, m)
            }));
          }
        }
    }
    fn convert_class_item(ccx: @crate_ctxt, ci: ast::class_member) {
        /* we want to do something here, b/c within the
         scope of the class, it's ok to refer to fields &
        methods unqualified */

        /* they have these types *within the scope* of the
         class. outside the class, it's done with expr_field */
        alt ci {
         ast::instance_var(_,t,_,id) {
             let tt = ast_ty_to_ty(ccx, m_collect, t);
             write_ty_to_tcx(ccx.tcx, id, tt);
         }
         ast::class_method(it) { convert(ccx, it); }
        }
    }
    fn convert(ccx: @crate_ctxt, it: @ast::item) {
        let tcx = ccx.tcx;
        alt it.node {
          // These don't define types.
          ast::item_mod(_) | ast::item_native_mod(_) {}
          ast::item_enum(variants, ty_params) {
            let tpt = ty_of_item(ccx, m_collect, it);
            write_ty_to_tcx(tcx, it.id, tpt.ty);
            get_enum_variant_types(ccx, tpt.ty, variants, ty_params);
          }
          ast::item_impl(tps, ifce, selfty, ms) {
            let i_bounds = ty_param_bounds(ccx, m_collect, tps);
            let my_methods = [];
            let selfty = ast_ty_to_ty(ccx, m_collect, selfty);
            write_ty_to_tcx(tcx, it.id, selfty);
            tcx.tcache.insert(local_def(it.id), {bounds: i_bounds,
                                                 ty: selfty});
            for m in ms {
                write_ty_to_tcx(tcx, m.self_id, selfty);
                let bounds = ty_param_bounds(ccx, m_collect, m.tps);
                let mty = ty_of_method(ccx, m_collect, m);
                my_methods += [{mty: mty, id: m.id, span: m.span}];
                let fty = ty::mk_fn(tcx, mty.fty);
                tcx.tcache.insert(local_def(m.id),
                                     {bounds: @(*i_bounds + *bounds),
                                      ty: fty});
                write_ty_to_tcx(tcx, m.id, fty);
            }
            alt ifce {
              some(t) {
                let iface_ty = ast_ty_to_ty(ccx, m_collect, t);
                alt ty::get(iface_ty).struct {
                  ty::ty_iface(did, tys) {
                    // Store the iface type in the type node
                    alt check t.node {
                      ast::ty_path(_, t_id) {
                        write_ty_to_tcx(tcx, t_id, iface_ty);
                      }
                    }
                    if did.crate == ast::local_crate {
                        ensure_iface_methods(ccx, did.node);
                    }
                    for if_m in *ty::iface_methods(tcx, did) {
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
            let {bounds, params} = mk_ty_params(ccx, tps);
            let def_id = local_def(it.id);
            let t_arg = ty_of_arg(ccx, m_collect, decl.inputs[0]);
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
            let tpt = ty_of_item(ccx, m_collect, it);
            write_ty_to_tcx(tcx, it.id, tpt.ty);
            ensure_iface_methods(ccx, it.id);
          }
          ast::item_class(tps, members, ctor) {
              // Write the class type
              let tpt = ty_of_item(ccx, m_collect, it);
              write_ty_to_tcx(tcx, it.id, tpt.ty);
              // Write the ctor type
              let t_ctor = ty::mk_fn(tcx,
                                     ty_of_fn_decl(ccx,
                                                   m_collect,
                                                   ast::proto_any,
                                                   ctor.node.dec));
              write_ty_to_tcx(tcx, ctor.node.id, t_ctor);
              tcx.tcache.insert(local_def(ctor.node.id),
                                   {bounds: tpt.bounds, ty: t_ctor});
              /* FIXME: check for proper public/privateness */
              // Write the type of each of the members
              for m in members {
                 convert_class_item(ccx, m.node.decl);
              }
          }
          _ {
            // This call populates the type cache with the converted type
            // of the item in passing. All we have to do here is to write
            // it into the node type table.
            let tpt = ty_of_item(ccx, m_collect, it);
            write_ty_to_tcx(tcx, it.id, tpt.ty);
          }
        }
    }
    fn convert_native(ccx: @crate_ctxt, i: @ast::native_item) {
        // As above, this call populates the type table with the converted
        // type of the native item. We simply write it into the node type
        // table.
        let tpt = ty_of_native_item(ccx, m_collect, i);
        write_ty_to_tcx(ccx.tcx, i.id, tpt.ty);
    }
    fn collect_item_types(ccx: @crate_ctxt, crate: @ast::crate) {
        visit::visit_crate(*crate, (), visit::mk_simple_visitor(@{
            visit_item: bind convert(ccx, _),
            visit_native_item: bind convert_native(ccx, _)
            with *visit::default_simple_visitor()
        }));
    }
}

// FIXME Duplicate logic with ty::type_autoderef.  This could be refactored
// following the usual pattern (type_autoderef would take a helper for the
// actual recursion).
fn do_autoderef(fcx: @fn_ctxt, sp: span, t: ty::t_i) -> ty::t_i {
    let mut t = t;
    let mut visited = [];
    let mut done = false;
    while !done {
        alt *t {
          ty::ty_var_i(id) if vec::contains(visited, id) {
            fcx.ccx.tcx.sess.span_fatal(
                sp, #fmt["cyclic type `%s` of infinite size",
                         ty_str(fcx, t)]);
          }
          ty::ty_var_i(id) {
            visited += [id];
          }
          ty::sty_i(_) { }
        }

        structure_of(fcx, sp, t) {|sty|
            alt sty {
              ty::ty_box(inner) | ty::ty_uniq(inner) | ty::ty_rptr(_, inner) {
                t = inner.ty;
              }
              ty::ty_res(_, inner, tps) {
                t = ty::substitute_type_params_i(fcx.vb, tps, inner);
              }
              ty::ty_enum(did, tps) {
                let variants = ty::enum_variants(fcx.ccx.tcx, did);
                if vec::len(*variants) != 1u
                    || vec::len(variants[0].args) != 1u {
                    done = true;
                } else {
                    t = ty::ty_to_ty_i_subst(fcx.ccx.tcx, variants[0].args[0],
                                             tps);
                }
              }
                _ {
                done = true;
              }
            }
        }
    }
    ret t;
}

fn resolve_type_vars(fcx: @fn_ctxt, sp: span, typ: ty::t_i) -> ty::t {
    alt ty::unify::resolve_type(fcx.vb, typ) {
      result::ok(new_type) {
        ret new_type;
      }
      result::err(e) {
        fcx.ccx.tcx.sess.span_err(sp, ty::resolve_err_to_str(e));

        // better to return something than halt?
        ret ty::mk_bot(fcx.ccx.tcx);
      }
    }
}

// Demands - procedures that require that two types unify and emit an error
// message if they don't.
mod demand {
    enum err_kernel {
        ek_mismatched_types,
        ek_mismatched_types_in_range
    }

    fn ek_to_str(ek: err_kernel) -> str {
        alt ek {
          ek_mismatched_types { "mismatched types" }
          ek_mismatched_types_in_range { "mismatched types in range" }
        }
    }

    fn tys<T:copy,C:ty_ops<T> uctxt<T>>(
        cx: C, sp: span, ek: err_kernel,
        expecteds: [T], actuals: [T]) {

        if check vec::same_length(expecteds, actuals) {
            vec::iter2(expecteds, actuals) {|e, a|
                ty(cx, sp, ek, e, a);
            }
        } else {
            cx.sess().span_err(
                sp, #fmt["%s: expected %u types but found %u",
                         ek_to_str(ek),
                         expecteds.len(),
                         actuals.len()])
        }
    }

    // Requires that the two types unify, and prints an error message if they
    // don't. Returns the unified type and the type parameter substitutions.
    fn ty<T:copy,C:ty_ops<T> uctxt<T>>(cx: C, sp: span, ek: err_kernel,
                         expected: T, actual: T) -> T {
        alt ty::unify::unify(cx, expected, actual) {
          result::ok(t) {
            ret t;
          }
          result::err(err) {
            cx.sess().span_err(
                sp, #fmt["%s: expected `%s` but found `%s`",
                         ek_to_str(ek),
                         cx.to_str(expected),
                         cx.to_str(actual)]);
            ret expected;
          }
        }
    }
}


// Returns true if the two types unify and false if they don't.
fn are_compatible(fcx: @fn_ctxt, expected: ty::t_i, actual: ty::t_i) -> bool {
    alt ty::unify::unify(fcx.vb, expected, actual) {
      result::ok(_) { ret true; }
      result::err(_) { ret false; }
    }
}


// Returns the types of the arguments to a enum variant.
fn variant_arg_types(ccx: @crate_ctxt, _sp: span, vid: ast::def_id,
                     enum_ty_params: [ty::t_i]) -> [ty::t_i] {
    let tpt = ty::lookup_item_type(ccx.tcx, vid);
    alt ty::get(tpt.ty).struct {
      ty::ty_fn(f) {
        // N-ary variant.
        vec::map(f.inputs) {|a| ty::ty_to_ty_i_subst(ccx.tcx, a.ty, enum_ty_params) }
      }
      _ {
        // Nullary variant.
        []
      }
    }
}


// Type resolution: the phase that finds all the types in the AST with
// unresolved type variables and replaces "ty_var" types with their
// substitutions.
mod writeback {

    export resolve_type_vars_in_fn;
    export resolve_type_vars_in_expr;

    fn resolve_type_vars_in_type(fcx: @fn_ctxt, sp: span, typ: ty::t_i) ->
       option<ty::t> {
        alt ty::unify::resolve_type(fcx.vb, typ) {
          result::ok(new_type) { ret some(new_type); }
          result::err(e) {
            if !fcx.ccx.tcx.sess.has_errors() {
                fcx.ccx.tcx.sess.span_err(
                    sp,
                    #fmt["cannot determine a type for this expression: %s",
                         ty::resolve_err_to_str(e)]);
            }
            ret none;
          }
        }
    }
    fn resolve_type_vars_for_node(wbcx: wb_ctxt, sp: span, id: ast::node_id)
        -> option<ty::t> {
        let fcx = wbcx.fcx, tcx = fcx.ccx.tcx;
        alt resolve_type_vars_in_type(fcx, sp, node_ty(fcx, id)) {
          none {
            wbcx.success = false;
            ret none;
          }

          some(t) {
            #debug["resolve_type_vars_for_node(node=%s) t=%s",
                   ast_map::node_str(tcx.items, id),
                   ty_to_str(tcx, t)];

            write_ty_to_tcx(tcx, id, t);
            alt node_ty_substs_find(fcx, id) {
              some(substs) {
                let new_substs = [];
                for subst in substs {
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
        if wbcx.fcx.vb.node_types.contains_key(id) {
            resolve_type_vars_for_node(wbcx, sp, id)
        } else {
            none
        }
    }

    type wb_ctxt =
        // As soon as we hit an error we have to stop resolving
        // the entire function
        {fcx: @fn_ctxt, mutable success: bool};
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
        visit::visit_pat(p, wbcx, v);
    }
    fn visit_local(l: @ast::local, wbcx: wb_ctxt, v: wb_vt) {
        if !wbcx.success { ret; }
        let var_id = lookup_local(wbcx.fcx, l.span, l.node.id);
        #debug["Local %s corresponds to ty var %?",
               pat_to_str(l.node.pat), var_id];
        let fix_rslt =
            ty::unify::resolve_type_var(wbcx.fcx.vb, var_id);
        alt fix_rslt {
          result::ok(lty) {
            write_ty_to_tcx(wbcx.fcx.ccx.tcx, l.node.id, lty);
          }
          result::err(_) {
            wbcx.fcx.ccx.tcx.sess.span_err(l.span,
                                           "cannot determine a type \
                                                for this local variable");
            wbcx.success = false;
          }
        }
        visit::visit_local(l, wbcx, v);
    }
    fn visit_item(_item: @ast::item, _wbcx: wb_ctxt, _v: wb_vt) {
        // Ignore items
    }

    fn resolve_type_vars_in_expr(fcx: @fn_ctxt, e: @ast::expr) -> bool {
        let wbcx = {fcx: fcx, mutable success: true};
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
        let wbcx = {fcx: fcx, mutable success: true};
        let visit =
            visit::mk_vt(@{visit_item: visit_item,
                           visit_stmt: visit_stmt,
                           visit_expr: visit_expr,
                           visit_block: visit_block,
                           visit_pat: visit_pat,
                           visit_local: visit_local
                              with *visit::default_visitor()});
        visit.visit_block(blk, wbcx, visit);
        for arg in decl.inputs {
            resolve_type_vars_for_node(wbcx, arg.ty.span, arg.id);
        }
        ret wbcx.success;
    }
}

// Local variable gathering. We gather up all locals and create variable IDs
// for them before typechecking the function.
type gather_result =
    {var_bindings: @ty::var_bindings,
     locals: hashmap<ast::node_id, int>,
     next_var_id: @mutable int};

// Used only as a helper for check_fn.
fn gather_locals(ccx: @crate_ctxt,
                 decl: ast::fn_decl,
                 body: ast::blk,
                 _id: ast::node_id,
                 arg_tys: [ty::t_i],
                 old_fcx: option<@fn_ctxt>) -> gather_result {
    let {vb: vb, locals: locals, nvi: nvi} = alt old_fcx {
      none {
        {vb: ty::var_bindings(ccx.tcx),
         locals: int_hash(),
         nvi: @mutable 0}
      }
      some(fcx) {
        {vb: fcx.vb,
         locals: fcx.locals,
         nvi: fcx.next_var_id}
      }
    };
    let tcx = ccx.tcx;

    let next_var_id = fn@() -> int { let rv = *nvi; *nvi += 1; ret rv; };

    let assign = fn@(nid: ast::node_id, ty_opt: option<ty::t_i>) {
        let var_id = next_var_id();
        locals.insert(nid, var_id);
        alt ty_opt {
          none {/* nothing to do */ }
          some(typ) {
            ty::unify::unify(vb, ty_var(var_id), typ);
          }
        }
    };

    // Add for parameters.
    vec::iter2(decl.inputs, arg_tys) {|inp,aty|
        assign(inp.id, some(aty));
    }

    // Add explicitly-declared locals.
    let visit_local = fn@(local: @ast::local, &&e: (), v: visit::vt<()>) {
        let local_ty = ast_ty_to_opt_ty_i(ccx, local.node.ty);
        assign(local.node.id, local_ty);
        visit::visit_local(local, e, v);
    };

    // Add pattern bindings.
    let visit_pat = fn@(p: @ast::pat, &&e: (), v: visit::vt<()>) {
        alt p.node {
          ast::pat_ident(_, _)
          if !pat_util::pat_is_variant(tcx.def_map, p) {
            assign(p.id, none);
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
    ret {var_bindings: vb,
         locals: locals,
         next_var_id: nvi};
}

// AST fragment checking
fn check_lit(fcx: @fn_ctxt, lit: @ast::lit) -> ty::t_i {
    alt lit.node {
      ast::lit_str(_) { ty::mk_str(fcx.vb) }
      ast::lit_int(_, t) { ty::mk_mach_int(fcx.vb, t) }
      ast::lit_uint(_, t) { ty::mk_mach_uint(fcx.vb, t) }
      ast::lit_float(_, t) { ty::mk_mach_float(fcx.vb, t) }
      ast::lit_nil { ty::mk_nil(fcx.vb) }
      ast::lit_bool(_) { ty::mk_bool(fcx.vb) }
    }
}

fn valid_range_bounds(tcx: ty::ctxt, from: @ast::expr, to: @ast::expr)
    -> bool {
    ast_util::compare_lit_exprs(tcx, from, to) <= 0
}

type pat_ctxt = {
    fcx: @fn_ctxt,
    map: pat_util::pat_id_map,
    alt_region: ty::region,
    block_region: ty::region,
    /* Equal to either alt_region or block_region. */
    pat_region: ty::region
};

// Replaces self, caller, or inferred regions in the given type with the given
// region.
fn instantiate_self_regions<T:copy,C:ty_ops<T>>(
    cx: C, region: ty::region, &&ty: T) -> T {

    ty::fold_rptr(cx, ty) {|r|
        alt r {
          ty::re_inferred | ty::re_caller(_) | ty::re_self(_) { region }
          _ { r }
        }
    }
}

// Replaces all region variables in the given type with "inferred regions".
// This is used during method lookup to allow typeclass implementations to
// refer to inferred regions.
fn universally_quantify_regions(fcx: @fn_ctxt, ty: ty::t_i) -> ty::t_i {
    ty::fold_rptr(fcx.vb, ty) {|_r|
        ty::re_inferred
    }
}

fn check_pat_variant(pcx: pat_ctxt, pat: @ast::pat, path: @ast::path,
                     subpats: [@ast::pat], expected: ty::t_i) {
    // Typecheck the path.
    let tcx = pcx.fcx.ccx.tcx;
    let fcx = pcx.fcx;
    write_path_ty(fcx, pat.span, path, pat.id);

    let v_def = lookup_def(pcx.fcx, path.span, pat.id);
    let v_def_ids = ast_util::variant_def_ids(v_def);

    // Take the enum type params out of `expected`.
    structure_of(fcx, pat.span, expected) {|expected_sty|
        alt expected_sty {
          ty::ty_enum(_, expected_tps) {
            //let ctor_ty = node_ty(fcx, pat.id);

            // unify provided parameters (if any) with parameters present
            // on the expected type
            demand::tys(fcx.vb, pat.span, demand::ek_mismatched_types,
                        expected_tps, node_ty_substs(fcx, pat.id));

            // Get the number of arguments in this enum variant.
            let arg_types = variant_arg_types(
                fcx.ccx, pat.span, v_def_ids.var, expected_tps);
            let arg_types = vec::map(arg_types) {|t|
                instantiate_self_regions(fcx.vb, pcx.pat_region, t)
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
                tcx.sess.span_err(
                    pat.span,
                    #fmt["this pattern has %u field%s, \
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
                      ty_str(fcx, expected)]);
          }
        }
    }
}

fn ty_var(vid: int) -> ty::t_i {
    @ty::ty_var_i(vid)
}

// Pattern checking is top-down rather than bottom-up so that bindings get
// their types immediately.
fn check_pat(pcx: pat_ctxt, pat: @ast::pat, expected: ty::t_i) {
    let fcx = pcx.fcx;
    let tcx = pcx.fcx.ccx.tcx;
    alt pat.node {
      ast::pat_wild {
        write_ty(fcx, pat.id, expected);
      }
      ast::pat_lit(lt) {
        check_expr_with(fcx, lt, expected);
        write_ty(fcx, pat.id, expr_ty(fcx, lt));
      }
      ast::pat_range(begin, end) {
        check_expr_with(fcx, begin, expected);
        check_expr_with(fcx, end, expected);
        let b_ty = expr_ty(fcx, begin);
        let e_ty = expr_ty(fcx, end);

        // "mismatched types in range"
        demand::ty(fcx.vb, pat.span, demand::ek_mismatched_types_in_range,
                   b_ty, e_ty);

        if !type_is_numeric(fcx, pat.span, b_ty) {
            tcx.sess.span_err(pat.span, "non-numeric type used in range");
        } else if !valid_range_bounds(tcx, begin, end) {
            tcx.sess.span_err(begin.span, "lower range bound must be less \
                                           than upper");
        }

        write_ty(fcx, pat.id, b_ty);
      }
      ast::pat_ident(name, sub)
      if !pat_util::pat_is_variant(tcx.def_map, pat) {
        let vid = lookup_local(fcx, pat.span, pat.id);
        let typ = ty_var(vid);
        typ = demand::ty(fcx.vb, pat.span, demand::ek_mismatched_types,
                         expected, typ);
        let canon_id = pcx.map.get(path_to_ident(name));
        if canon_id != pat.id {
            let ct = ty_var(lookup_local(fcx, pat.span, canon_id));
            typ = demand::ty(fcx.vb, pat.span, demand::ek_mismatched_types,
                             ct, typ);
        }
        write_ty(fcx, pat.id, typ);
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
        let ex_fields = structure_of(fcx, pat.span, expected) {|sty|
            alt sty {
              ty::ty_rec(fields) { fields }
              _ {
                tcx.sess.span_fatal
                    (pat.span,
                     #fmt["mismatched types: expected `%s` but found record",
                          ty_str(fcx, expected)]);
              }
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
        fn matches(name: str, f: ty::field_i) -> bool {
            ret str::eq(name, f.ident);
        }
        for f: ast::field_pat in fields {
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
        write_ty(fcx, pat.id, expected);
      }
      ast::pat_tup(elts) {
        let ex_elts = structure_of(fcx, pat.span, expected) {|sty|
            alt sty {
              ty::ty_tup(elts) { elts }
              _ {
                tcx.sess.span_fatal
                    (pat.span,
                     #fmt["mismatched types: expected `%s`, found tuple",
                          ty_str(fcx, expected)]);
              }
            }
        };
        let e_count = vec::len(elts);
        if e_count != vec::len(ex_elts) {
            tcx.sess.span_fatal
                (pat.span, #fmt["mismatched types: expected a tuple \
                      with %u fields, found one with %u \
                      fields", vec::len(ex_elts), e_count]);
        }
        let i = 0u;
        for elt in elts {
            check_pat(pcx, elt, ex_elts[i]);
            i += 1u;
        }
        write_ty(fcx, pat.id, expected);
      }
      ast::pat_box(inner) {
        structure_of(pcx.fcx, pat.span, expected) {|sty|
            alt sty {
              ty::ty_box(e_inner) {
                check_pat(pcx, inner, e_inner.ty);
                write_ty(fcx, pat.id, expected);
              }
              _ {
                tcx.sess.span_fatal(
                    pat.span,
                    "mismatched types: expected `" +
                    ty_str(fcx, expected) +
                    "` found box");
              }
            }
        }
      }
      ast::pat_uniq(inner) {
        structure_of(pcx.fcx, pat.span, expected) {|sty|
            alt sty {
              ty::ty_uniq(e_inner) {
                check_pat(pcx, inner, e_inner.ty);
                write_ty(fcx, pat.id, expected);
              }
              _ {
                tcx.sess.span_fatal(pat.span,
                                    "mismatched types: expected `" +
                                    ty_str(fcx, expected) +
                                    "` found uniq");
              }
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
                alt check ccx.tcx.items.get(did.node) {
                  ast_map::node_method(m, _, _) { m.decl.purity }
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

type unifier = fn@(@fn_ctxt, span, ty::t_i, ty::t_i);

fn check_expr(fcx: @fn_ctxt, expr: @ast::expr) -> bool {
    fn dummy_unify(_fcx: @fn_ctxt, _sp: span, _e: ty::t_i, _a: ty::t_i) {
    }
    check_expr_with_unifier(fcx, expr, dummy_unify, @ty::sty_i(ty::ty_nil))
}

fn check_expr_with(fcx: @fn_ctxt, expr: @ast::expr, expected: ty::t_i) -> bool {
    fn demand_unify(fcx: @fn_ctxt, sp: span, e: ty::t_i, a: ty::t_i) {
        demand::ty(fcx.vb, sp, demand::ek_mismatched_types, e, a);
    }
    ret check_expr_with_unifier(fcx, expr, demand_unify, expected);
}

fn impl_self_ty(fcx: @fn_ctxt, did: ast::def_id)
    -> {vars: [ty::t_i], self_ty: ty::t_i} {

    let {n_tps, self_ty} = if did.crate == ast::local_crate {
        alt check fcx.ccx.tcx.items.get(did.node) {
          ast_map::node_item(@{node: ast::item_impl(ts, _, st, _),
                               _}, _) {
            {n_tps: ts.len(),
             self_ty: ast_ty_to_ty(fcx.ccx, m_check, st)}
          }
        }
    } else {
        let ity = ty::lookup_item_type(fcx.ccx.tcx, did);
        {n_tps: vec::len(*ity.bounds),
         self_ty: ity.ty}
    };

    let vars = next_ty_vars(fcx, n_tps);

    {vars: vars, self_ty: ty::ty_to_ty_i_subst(fcx.ccx.tcx, self_ty, vars)}
}

fn lookup_method(fcx: @fn_ctxt, expr: @ast::expr, node_id: ast::node_id,
                 name: ast::ident, ty: ty::t_i, tps: [ty::t_i])
    -> option<method_origin> {
    alt lookup_method_inner(fcx, expr, name, ty) {
      some({method_ty: fty, n_tps: method_n_tps, substs, origin, self_sub}) {
        let tcx = fcx.ccx.tcx;
        let substs = substs, n_tps = vec::len(substs), n_tys = vec::len(tps);
        let has_self = ty::type_has_self(fty);
        if method_n_tps + n_tps > 0u {
            if n_tys == 0u || n_tys != method_n_tps {
                if n_tys != 0u {
                    tcx.sess.span_err
                        (expr.span, "incorrect number of type \
                                     parameters given for this method");

                }
                substs += next_ty_vars(fcx, method_n_tps);
            } else {
                substs += tps;
            }
        } else {
            substs += tps;
        }
        write_ty_substs(fcx, node_id, fty, substs);

        if has_self && !option::is_none(self_sub) {
            let (self_ty, span) = option::get(self_sub);
            let fty = node_ty(fcx, node_id);
            let fty = fixup_self_call(
                fcx, span, fty, substs, self_ty);
            write_ty(fcx, node_id, fty);
        }
        if ty::type_has_rptrs(ty::ty_fn_ret(fty)) {
            let fty = node_ty(fcx, node_id);
            fty = fixup_self_region_in_method_ty(fcx, fty, expr);
            write_ty(fcx, node_id, fty);
        }
        some(origin)
      }
      none { none }
    }
}

fn lookup_method_inner(fcx: @fn_ctxt, expr: @ast::expr,
                       name: ast::ident, ty: ty::t_i)
    -> option<{method_ty: ty::t, n_tps: uint, substs: [ty::t_i],
                  origin: method_origin,
                  self_sub: option<(ty::t_i, span)>}> {
    let tcx = fcx.ccx.tcx;

    // First, see whether this is an interface-bounded parameter
    alt structure_of(fcx, expr.span, ty, {|sty| sty }) {
      ty::ty_param(n, did) {
        let bound_n = 0u;
        for bound in *tcx.ty_param_bounds.get(did.node) {
            alt bound {
              ty::bound_iface(t) {
                let (iid, tps) = alt check ty::get(t).struct {
                  ty::ty_iface(i, tps) { (i, tps) }
                };
                let ifce_methods = ty::iface_methods(tcx, iid);
                alt vec::position(*ifce_methods, {|m| m.ident == name}) {
                  some(pos) {
                    let m = ifce_methods[pos];
                    let tps_i = vec::map(tps) {|t| ty::ty_to_ty_i(tcx, t) };
                    ret some({method_ty: ty::mk_fn(tcx, {proto: ast::proto_box
                                                         with m.fty}),
                              n_tps: vec::len(*m.tps),
                              substs: tps_i,
                              origin: method_param(iid, pos, n, bound_n),
                              self_sub: some((ty, expr.span))
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
        let i = 0u;
        for m in *ty::iface_methods(tcx, did) {
            if m.ident == name {
                let fty = ty::mk_fn(tcx, {proto: ast::proto_box with m.fty});
                if ty::type_has_self(fty) {
                    tcx.sess.span_fatal(
                        expr.span, "can not call a method that contains a \
                                    self type through a boxed iface");
                } else if (*m.tps).len() > 0u {
                    tcx.sess.span_fatal(
                        expr.span, "can not call a generic method through a \
                                    boxed iface");
                }
                ret some({method_ty: fty,
                          n_tps: vec::len(*m.tps),
                          substs: tps,
                          origin: method_iface(did, i),
                          self_sub: none});
            }
            i += 1u;
        }
      }
      _ {}
    }

    fn ty_from_did(fcx: @fn_ctxt, did: ast::def_id) -> ty::t {
        let ccx = fcx.ccx;
        let tcx = ccx.tcx;
        /*if did.crate == ast::local_crate {
            alt check ccx.tcx.items.get(did.node) {
              ast_map::node_method(m, _, _) {
                let mt = ty_of_method(ccx, m_check, m);
                ty::mk_fn(tcx, {proto: ast::proto_box with mt.fty})
              }
            }
        } else {
            alt check ty::get(csearch::get_type(tcx, did).ty).struct {
              ty::ty_fn(fty) {
                ty::mk_fn(tcx, {proto: ast::proto_box with fty})
              }
            }
        }*/

        // FIXME/NDM--Why don't we just use proto_box for method types
        //            in the first place?
        let mty = ty::lookup_item_type(tcx, did);
        alt check ty::get(mty.ty).struct {
          ty::ty_fn(fty) {
            ty::mk_fn(tcx, {proto: ast::proto_box with fty})
          }
        }
    }

    let result = none, complained = false;
    std::list::iter(fcx.ccx.impl_map.get(expr.id)) {|impls|
        if option::is_some(result) { ret; }
        for @{did, methods, _} in *impls {
            alt vec::find(methods, {|m| m.ident == name}) {
              some(m) {
                let {vars, self_ty} = impl_self_ty(fcx, did);
                let ty = universally_quantify_regions(fcx, ty);
                alt ty::unify::unify(fcx.vb, self_ty, ty) {
                  result::ok(_) {
                    if option::is_some(result) {
                        // FIXME[impl] score specificity to resolve ambiguity?
                        if !complained {
                           tcx.sess.span_err(expr.span, "multiple applicable \
                                                         methods in scope");
                           complained = true;
                        }
                    } else {
                        result = some({
                            method_ty: ty_from_did(fcx, m.did),
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
    result
}

// problem -- class_item_ty should really be only used for internal stuff.
// or should have a privacy field.
fn lookup_field_ty(fcx: @fn_ctxt, items:[@ty::class_item_ty],
                   fieldname: ast::ident, sp: span, tps: [ty::t_i])
    -> ty::t_i {
    for item in items {
        #debug("%s $$$ %s", fieldname, item.ident);
        alt item.contents {
          ty::var_ty(t) if item.ident == fieldname {
            ret ty::ty_to_ty_i_subst(fcx.ccx.tcx, t, tps);
          }
          _ { }
        }
    }
    fcx.ccx.tcx.sess.span_fatal(sp, #fmt("unbound field %s", fieldname));
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
                    ret ty::re_block(local_block_id);
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
            let expr_ty = expr_ty(fcx, base);
            structure_of(fcx, expr.span, expr_ty) {|sty|
                alt sty {
                  ty::ty_rptr(region, _) { region }
                  ty::ty_box(_) | ty::ty_uniq(_) {
                    fcx.ccx.tcx.sess.span_unimpl(expr.span, "borrowing");
                  }
                  _ { ret region_of(fcx, base); }
                }
            }
        }
        _ {
            let blk_id = fcx.ccx.tcx.region_map.rvalue_to_block.get(expr.id);
            ret ty::re_block(blk_id);
        }
    }
}

fn check_expr_fn_with_unifier(fcx: @fn_ctxt,
                              expr: @ast::expr,
                              proto: ast::proto,
                              decl: ast::fn_decl,
                              body: ast::blk,
                              unifier: unifier,
                              expected: ty::t_i) {
    let tcx = fcx.ccx.tcx;
    let fty =
        @ty::sty_i(ty::ty_fn(
            ty_of_fn_decl_base(
                tcx, proto, decl,
                {|a| {mode: a.mode, ty: ast_ty_to_ty_i(fcx, a.ty)} },
                {|t| ast_ty_to_ty_i(fcx, t) })));

    #debug("check_expr_fn_with_unifier %s fty=%s",
           expr_to_str(expr), ty_str(fcx, fty));

    write_ty(fcx, expr.id, fty);

    // Unify the type of the function with the expected type before we
    // typecheck the body so that we have more information about the
    // argument types in the body. This is needed to make binops and
    // record projection work on type inferred arguments.
    unifier(fcx, expr.span, expected, fty);

    let bot = false;
    let ret_ty = fn_ret(fcx, body.span, fty, bot);
    let arg_tys = vec::map(fn_args(fcx, body.span, fty)) {|a| a.ty };

    check_fn(fcx.ccx, proto, decl, body, expr.id,
             ret_ty, arg_tys, some(fcx));
}

fn check_expr_with_unifier(fcx: @fn_ctxt, expr: @ast::expr, unifier: unifier,
                           expected: ty::t_i) -> bool {
    #debug("typechecking expr %s",
           syntax::print::pprust::expr_to_str(expr));

    // A generic function to factor out common logic from call and bind
    // expressions.
    fn check_call_or_bind(fcx: @fn_ctxt, sp: span, id: ast::node_id,
                          fty: ty::t_i, args: [option<@ast::expr>]) -> bool {

        // Replaces "caller" regions in the arguments with the local region.
        fn instantiate_caller_regions(fcx: @fn_ctxt, id: ast::node_id,
                                      args: [ty::arg_i]) -> [ty::arg_i] {
            let site_to_block = fcx.ccx.tcx.region_map.call_site_to_block;
            let block_id = alt site_to_block.find(id) {
                none {
                    // This can happen for those expressions that are
                    // synthesized during typechecking; e.g. during
                    // check_constraints().
                    ret args;
                }
                some(block_id) { block_id }
            };

            let region = ty::re_block(block_id);
            ret vec::map(args) {|arg|
                let ty = ty::fold_rptr(fcx.vb, arg.ty) {|r|
                    alt r {
                      ty::re_caller(_) {
                        // FIXME: We should not recurse into nested
                        // function types here.
                        region
                      }
                      _ { r }
                    }
                };
                {ty: ty with arg}
            };
        }

        let arg_tys = fn_args(fcx, sp, fty);

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
            // HACK: build an arguments list with dummy arguments to
            // check against
            let dummy = {mode: ast::expl(ast::by_ref),
                         ty: ty::mk_bot(fcx.vb)};
            arg_tys = vec::from_elem(supplied_arg_count, dummy);
        }

        arg_tys = instantiate_caller_regions(fcx, id, arg_tys);

        // Check the arguments.
        // We do this in a pretty awful way: first we typecheck any arguments
        // that are not anonymous functions, then we typecheck the anonymous
        // functions. This is so that we have more information about the types
        // of arguments when we typecheck the functions. This isn't really the
        // right way to do this.
        let check_args = fn@(check_blocks: bool) -> bool {
            let i = 0u;
            let bot = false;
            for a_opt in args {
                alt a_opt {
                  some(a) {
                    let is_block = alt a.node {
                      ast::expr_fn_block(_, _) { true }
                      _ { false }
                    };
                    if is_block == check_blocks {
                        bot |= check_expr_with(fcx, a, arg_tys[i].ty);
                    }
                  }
                  none { }
                }
                i += 1u;
            }
            ret bot;
        };
        check_args(false) | check_args(true)
    }

    // A generic function for checking assignment expressions
    fn check_assignment(fcx: @fn_ctxt, _sp: span, lhs: @ast::expr,
                        rhs: @ast::expr, id: ast::node_id) -> bool {
        let t = next_ty_var(fcx);
        let bot = check_expr_with(fcx, lhs, t) | check_expr_with(fcx, rhs, t);
        write_ty(fcx, id, ty::mk_nil(fcx.vb));
        ret bot;
    }

    // A generic function for checking call expressions
    fn check_call(fcx: @fn_ctxt, sp: span, id: ast::node_id, f: @ast::expr,
                  args: [@ast::expr])
        -> bool {
        let args_opt_0: [option<@ast::expr>] = [];
        for arg: @ast::expr in args {
            args_opt_0 += [some::<@ast::expr>(arg)];
        }

        let bot = check_expr(fcx, f);
        // Call the generic checker.
        bot | check_call_or_bind(fcx, sp, id, expr_ty(fcx, f),
                                 args_opt_0)
    }

    // A generic function for doing all of the checking for call expressions
    fn check_call_full(fcx: @fn_ctxt, sp: span, id: ast::node_id,
                       f: @ast::expr, args: [@ast::expr]) -> bool {
        let bot = check_call(fcx, sp, id, f, args);
        /* here we're kind of hosed, as f can be any expr
        need to restrict it to being an explicit expr_path if we're
        inside a pure function, and need an environment mapping from
        function name onto purity-designation */
        require_pure_call(fcx.ccx, fcx.purity, f, sp);

        // Pull the return type out of the type of the function.
        let fty = expr_ty(fcx, f);
        let rt_1 = fn_ret(fcx, sp, fty, bot);
        write_ty(fcx, id, rt_1);
        ret bot;
    }

    // A generic function for checking for or for-each loops
    fn check_for(fcx: @fn_ctxt, local: @ast::local,
                 element_ty: ty::t_i, body: ast::blk,
                 node_id: ast::node_id) -> bool {
        let locid = lookup_local(fcx, local.span, local.node.id);
        let element_ty = demand::ty(fcx.vb, local.span, demand::ek_mismatched_types,
                                    element_ty, ty::mk_var(locid));
        let bot = check_decl_local(fcx, local);
        check_block_no_value(fcx, body);
        // Unify type of decl with element type of the seq
        demand::ty(fcx.vb, local.span, demand::ek_mismatched_types,
                   node_ty(fcx, local.node.id), element_ty);
        write_nil(fcx, node_id);
        ret bot;
    }


    // A generic function for checking the then and else in an if
    // or if-check
    fn check_then_else(fcx: @fn_ctxt, thn: ast::blk,
                       elsopt: option<@ast::expr>, id: ast::node_id,
                       sp: span) -> bool {
        let (if_t, if_bot) =
            alt elsopt {
              some(els) {
                let thn_bot = check_block(fcx, thn);
                let thn_t = node_ty(fcx, thn.node.id);
                let els_bot = check_expr_with(fcx, els, thn_t);
                let els_t = expr_ty(fcx, els);
                let if_t = if !type_is_bot(fcx, sp, els_t) {
                    els_t
                } else {
                    thn_t
                };
                (if_t, thn_bot & els_bot)
              }
              none {
                check_block_no_value(fcx, thn);
                (ty::mk_nil(fcx.vb), false)
              }
            };
        write_ty(fcx, id, if_t);
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
    fn lookup_op_method(fcx: @fn_ctxt, op_ex: @ast::expr, self_t: ty::t_i,
                        opname: str, args: [option<@ast::expr>])
        -> option<ty::t_i> {
        let callee_id = ast_util::op_expr_callee_id(op_ex);
        alt lookup_method(fcx, op_ex, callee_id, opname, self_t, []) {
          some(origin) {
            let method_ty = node_ty(fcx, callee_id);
            check_call_or_bind(fcx, op_ex.span, op_ex.id, method_ty, args);
            fcx.ccx.method_map.insert(op_ex.id, origin);
            let bot = false;
            some(fn_ret(fcx, op_ex.span, method_ty, bot))
          }
          _ { none }
        }
    }
    fn check_binop(fcx: @fn_ctxt, ex: @ast::expr, ty: ty::t_i,
                   op: ast::binop, rhs: @ast::expr) -> ty::t_i {
        let tcx = fcx.ccx.tcx;
        if structure_of(fcx, ex.span, ty, ty::is_binopable(tcx, _, op)) {
            ret alt op {
              ast::eq | ast::lt | ast::le | ast::ne | ast::ge |
              ast::gt { ty::mk_bool(fcx.vb) }
              _ { ty }
            };
        }

        alt binop_method(op) {
          some(name) {
            alt lookup_op_method(fcx, ex, ty, name, [some(rhs)]) {
              some(ret_ty) { ret ret_ty; }
              _ {}
            }
          }
          _ {}
        }

        tcx.sess.span_err(
            ex.span, "binary operation " + ast_util::binop_to_str(op) +
            " cannot be applied to type `" + ty_str(fcx, ty) +
            "`");
        ty
    }
    fn check_user_unop(fcx: @fn_ctxt, op_str: str, mname: str,
                       ex: @ast::expr, rhs_t: ty::t_i) -> ty::t_i {
        alt lookup_op_method(fcx, ex, rhs_t, mname, []) {
          some(ret_ty) { ret_ty }
          _ {
            fcx.ccx.tcx.sess.span_err(
                ex.span, #fmt["cannot apply unary operator `%s` to type `%s`",
                              op_str, ty_str(fcx, rhs_t)]);
            rhs_t
          }
        }
    }

    let tcx = fcx.ccx.tcx;
    let id = expr.id;
    let bot = false;
    alt expr.node {
      ast::expr_lit(lit) {
        let typ = check_lit(fcx, lit);
        write_ty(fcx, id, typ);
      }
      ast::expr_binary(binop, lhs, rhs) {
        let lhs_t = next_ty_var(fcx);
        bot = check_expr_with(fcx, lhs, lhs_t);

        let rhs_bot = if !ast_util::is_shift_binop(binop) {
            check_expr_with(fcx, rhs, lhs_t)
        } else {
            let rhs_bot = check_expr(fcx, rhs);
            let rhs_t = expr_ty(fcx, rhs);
            require_integral(fcx, rhs.span, rhs_t);
            rhs_bot
        };

        if !ast_util::lazy_binop(binop) { bot |= rhs_bot; }

        let result = check_binop(fcx, expr, lhs_t, binop, rhs);
        write_ty(fcx, id, result);
      }
      ast::expr_assign_op(op, lhs, rhs) {
        require_impure(tcx.sess, fcx.purity, expr.span);
        bot = check_assignment(fcx, expr.span, lhs, rhs, id);
        let lhs_t = expr_ty(fcx, lhs);
        let result = check_binop(fcx, expr, lhs_t, op, rhs);
        demand::ty(fcx.vb, expr.span, demand::ek_mismatched_types,
                   result, lhs_t);
      }
      ast::expr_unary(unop, oper) {
        bot = check_expr(fcx, oper);
        let oper_t = expr_ty(fcx, oper);
        alt unop {
          ast::box(mutbl) {
            oper_t = @ty::sty_i(ty::ty_box({ty: oper_t, mutbl: mutbl}));
          }
          ast::uniq(mutbl) {
            oper_t = @ty::sty_i(ty::ty_uniq({ty: oper_t, mutbl: mutbl}));
          }
          ast::deref {
            structure_of(fcx, expr.span, oper_t) {|sty|
                alt sty {
                  ty::ty_box(inner) { oper_t = inner.ty; }
                  ty::ty_uniq(inner) { oper_t = inner.ty; }
                  ty::ty_res(_, inner, _) { oper_t = inner; }
                  ty::ty_enum(id, tps) {
                    let variants = ty::enum_variants(tcx, id);
                    if vec::len(*variants) != 1u ||
                        vec::len(variants[0].args) != 1u {
                        tcx.sess.span_err(expr.span,
                                          "can only dereference enums " +
                                          "with a single variant which has a "
                                          + "single argument");
                    }
                    oper_t = ty::ty_to_ty_i_subst(tcx, variants[0].args[0], tps);
                  }
                  ty::ty_ptr(inner) {
                    oper_t = inner.ty;
                    require_unsafe(tcx.sess, fcx.purity, expr.span);
                  }
                  ty::ty_rptr(_, inner) { oper_t = inner.ty; }
                  _ {
                    tcx.sess.span_err(expr.span,
                                      #fmt("Type %s cannot be dereferenced",
                                           ty_str(fcx, oper_t)));
                  }
                }
            }
          }
          ast::not {
            if !type_is_integral(fcx, expr.span, oper_t) {
                oper_t = check_user_unop(fcx, "!", "!", expr, oper_t);
            }
          }
          ast::neg {
            if !type_is_integral(fcx, expr.span, oper_t) {
                oper_t = check_user_unop(fcx, "-", "unary-", expr, oper_t);
            }
          }
        }
        write_ty(fcx, id, oper_t);
      }
      ast::expr_addr_of(mutbl, oper) {
        bot = check_expr(fcx, oper);
        let oper_t = expr_ty(fcx, oper);

        let region = region_of(fcx, oper);
        let tm = { ty: oper_t, mutbl: mutbl };
        oper_t = ty::mk_rptr(fcx.vb, region, tm);
        write_ty(fcx, id, oper_t);
      }
      ast::expr_path(pth) {
        write_path_ty(fcx, expr.span, pth, id);
      }
      ast::expr_mac(_) { tcx.sess.bug("unexpanded macro"); }
      ast::expr_fail(expr_opt) {
        bot = true;
        alt expr_opt {
          none {/* do nothing */ }
          some(e) { check_expr_with(fcx, e, ty::mk_str(fcx.vb)); }
        }
        write_bot(fcx, id);
      }
      ast::expr_break { write_bot(fcx, id); bot = true; }
      ast::expr_cont { write_bot(fcx, id); bot = true; }
      ast::expr_ret(expr_opt) {
        bot = true;
        alt expr_opt {
          none {
            let nil = ty::mk_nil(fcx.vb);
            if !are_compatible(fcx, fcx.ret_ty, nil) {
                tcx.sess.span_err(expr.span,
                                  "ret; in function returning non-nil");
            }
          }
          some(e) { check_expr_with(fcx, e, fcx.ret_ty); }
        }
        write_bot(fcx, id);
      }
      ast::expr_be(e) {
        // FIXME: prove instead of assert
        assert (ast_util::is_call_expr(e));
        check_expr_with(fcx, e, fcx.ret_ty);
        bot = true;
        write_nil(fcx, id);
      }
      ast::expr_log(_, lv, e) {
        bot = check_expr_with(fcx, lv, ty::mk_mach_uint(fcx.vb, ast::ty_u32));
        bot |= check_expr(fcx, e);
        write_nil(fcx, id);
      }
      ast::expr_check(_, e) {
        bot = check_pred_expr(fcx, e);
        write_nil(fcx, id);
      }
      ast::expr_if_check(cond, thn, elsopt) {
        bot =
            check_pred_expr(fcx, cond) |
                check_then_else(fcx, thn, elsopt, id, expr.span);
      }
      ast::expr_assert(e) {
        bot = check_expr_with(fcx, e, ty::mk_bool(fcx.vb));
        write_nil(fcx, id);
      }
      ast::expr_copy(a) {
        bot = check_expr_with_unifier(fcx, a, unifier, expected);
        write_ty(fcx, id, expr_ty(fcx, a));
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
            check_expr_with(fcx, cond, ty::mk_bool(fcx.vb)) |
                check_then_else(fcx, thn, elsopt, id, expr.span);
      }
      ast::expr_for(decl, seq, body) {
        bot = check_expr(fcx, seq);
        let ety = expr_ty(fcx, seq);
        let elt_ty = structure_of(fcx, expr.span, ety) {|sty|
            alt sty {
              ty::ty_vec(vec_elt_ty) { vec_elt_ty.ty }
              ty::ty_str { ty::mk_mach_uint(fcx.vb, ast::ty_u8) }
              _ {
                tcx.sess.span_fatal(expr.span,
                                    "mismatched types: expected vector or string "
                                    + "but found `" + ty_str(fcx, ety) + "`");
              }
            }
        };
        bot |= check_for(fcx, decl, elt_ty, body, id);
      }
      ast::expr_while(cond, body) {
        bot = check_expr_with(fcx, cond, ty::mk_bool(fcx.vb));
        check_block_no_value(fcx, body);
        write_ty(fcx, id, ty::mk_nil(fcx.vb));
      }
      ast::expr_do_while(body, cond) {
        bot = check_expr_with(fcx, cond, ty::mk_bool(fcx.vb)) |
              check_block_no_value(fcx, body);
        write_ty(fcx, id, node_ty(fcx, body.node.id));
      }
      ast::expr_loop(body) {
          check_block_no_value(fcx, body);
          write_ty(fcx, id, ty::mk_nil(fcx.vb));
          bot = !may_break(body);
      }
      ast::expr_alt(discrim, arms, _) {
        bot = check_expr(fcx, discrim);

        let parent_block = tcx.region_map.rvalue_to_block.get(discrim.id);

        // Typecheck the patterns first, so that we get types for all the
        // bindings.
        let pattern_ty = expr_ty(fcx, discrim);
        for arm: ast::arm in arms {
            let pcx = {
                fcx: fcx,
                map: pat_util::pat_id_map(tcx.def_map, arm.pats[0]),
                alt_region: ty::re_block(parent_block),
                block_region: ty::re_block(arm.body.node.id),
                pat_region: ty::re_block(parent_block)
            };

            for p: @ast::pat in arm.pats {
                check_pat(pcx, p, pattern_ty);
            }
        }
        // Now typecheck the blocks.
        let result_ty = next_ty_var(fcx);
        let arm_non_bot = false;
        for arm: ast::arm in arms {
            alt arm.guard {
              some(e) { check_expr_with(fcx, e, ty::mk_bool(fcx.vb)); }
              none { }
            }
            if !check_block(fcx, arm.body) { arm_non_bot = true; }
            let bty = node_ty(fcx, arm.body.node.id);
            result_ty = demand::ty(fcx.vb, arm.body.span, demand::ek_mismatched_types,
                                   result_ty, bty);
        }
        bot |= !arm_non_bot;
        if !arm_non_bot { result_ty = ty::mk_bot(fcx.vb); }
        write_ty(fcx, id, result_ty);
      }
      ast::expr_fn(proto, decl, body, captures) {
        check_expr_fn_with_unifier(fcx, expr, proto, decl, body,
                                   unifier, expected);
        capture::check_capture_clause(tcx, expr.id, proto, *captures);
      }
      ast::expr_fn_block(decl, body) {
        // Take the prototype from the expected type, but default to block:
        let proto = alt expected {
          @ty::sty_i(ty::ty_fn({proto, _})) { proto }
          _ { ast::proto_box }
        };
        #debug("checking expr_fn_block %s expected=%s",
               expr_to_str(expr),
               ty_str(fcx, expected));
        check_expr_fn_with_unifier(fcx, expr, proto, decl, body,
                                   unifier, expected);
      }
      ast::expr_block(b) {
        // If this is an unchecked block, turn off purity-checking
        bot = check_block(fcx, b);
        let typ =
            alt b.node.expr {
              some(expr) { expr_ty(fcx, expr) }
              none { ty::mk_nil(fcx.vb) }
            };
        write_ty(fcx, id, typ);
      }
      ast::expr_bind(f, args) {
        // Call the generic checker.
        bot = check_expr(fcx, f);
        bot |= check_call_or_bind(fcx, expr.span, expr.id, expr_ty(fcx, f),
                                  args);

        // Pull the argument and return types out.
        let {proto, arg_tys, rt, cf, constrs} =
            structure_of(fcx, expr.span, expr_ty(fcx, f)) {|sty|
                alt sty {
                  // FIXME:
                  // probably need to munge the constrs to drop constraints
                  // for any bound args
                  ty::ty_fn(f) {
                    {proto: f.proto,
                     arg_tys: f.inputs,
                     rt: f.output,
                     cf: f.ret_style,
                     constrs: f.constraints}
                  }
                  _ { fail "LHS of bind expr didn't have a function type?!"; }
                }
            };

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
        let out_args = [];
        let i = 0u;
        while i < vec::len(args) {
            alt args[i] {
              some(_) {/* no-op */ }
              none { out_args += [arg_tys[i]]; }
            }
            i += 1u;
        }

        let ft = ty::mk_fn(fcx.vb, {proto: proto,
                                inputs: out_args, output: rt,
                                ret_style: cf, constraints: constrs});
        write_ty(fcx, id, ft);
      }
      ast::expr_call(f, args, _) {
        bot = check_call_full(fcx, expr.span, expr.id, f, args);
      }
      ast::expr_cast(e, t) {
        bot = check_expr(fcx, e);
        let t_1 = ast_ty_to_ty_i(fcx, t);
        let t_e = expr_ty(fcx, e);

        alt t_1 {
          // This will be looked up later on
          @ty::sty_i(ty::ty_iface(_, _)) {}
          _ {
            if type_is_nil(fcx, expr.span, t_e) {
                tcx.sess.span_err(expr.span, "cast from nil: " +
                                  ty_str(fcx, t_e) + " as " +
                                  ty_str(fcx, t_1));
            } else if type_is_nil(fcx, expr.span, t_1) {
                tcx.sess.span_err(expr.span, "cast to nil: " +
                                  ty_str(fcx, t_e) + " as " +
                                  ty_str(fcx, t_1));
            }

            let t_1_is_scalar = type_is_scalar(fcx, expr.span, t_1);
            if type_is_c_like_enum(fcx,expr.span,t_e) && t_1_is_scalar {
                /* this case is allowed */
            } else if !(type_is_scalar(fcx,expr.span,t_e) && t_1_is_scalar) {
                // FIXME there are more forms of cast to support, eventually.
                tcx.sess.span_err(expr.span,
                                  "non-scalar cast: " +
                                  ty_str(fcx, t_e) + " as " +
                                  ty_str(fcx, t_1));
            }
          }
        }
        write_ty(fcx, id, t_1);
      }
      ast::expr_vec(args, mutbl) {
        let t = next_ty_var(fcx);
        for e: @ast::expr in args { bot |= check_expr_with(fcx, e, t); }
        let typ = ty::mk_vec(fcx.vb, {ty: t, mutbl: mutbl});
        write_ty(fcx, id, typ);
      }
      ast::expr_tup(elts) {
        let elt_ts = [];
        vec::reserve(elt_ts, vec::len(elts));
        for e in elts {
            check_expr(fcx, e);
            let ety = expr_ty(fcx, e);
            elt_ts += [ety];
        }
        let typ = ty::mk_tup(fcx.vb, elt_ts);
        write_ty(fcx, id, typ);
      }
      ast::expr_rec(fields, base) {
        alt base { none {/* no-op */ } some(b_0) { check_expr(fcx, b_0); } }
        let fields_t: [spanned<ty::field_i>] = [];
        for f: ast::field in fields {
            bot |= check_expr(fcx, f.node.expr);
            let expr_t = expr_ty(fcx, f.node.expr);
            let expr_mt = {ty: expr_t, mutbl: f.node.mutbl};
            // for the most precise error message,
            // should be f.node.expr.span, not f.span
            fields_t +=
                [respan(f.node.expr.span,
                        {ident: f.node.ident, mt: expr_mt})];
        }
        alt base {
          none {
            fn get_node(f: spanned<ty::field_i>) -> ty::field_i { f.node }
            let typ = ty::mk_rec(fcx.vb, vec::map(fields_t, get_node));
            write_ty(fcx, id, typ);
          }
          some(bexpr) {
            bot |= check_expr(fcx, bexpr);
            let bexpr_t = expr_ty(fcx, bexpr);
            let base_fields: [ty::field_i] = [];
            structure_of(fcx, expr.span, bexpr_t) {|sty|
                alt sty {
                  ty::ty_rec(flds) { base_fields = flds; }
                  _ {
                    tcx.sess.span_fatal(expr.span,
                                        "record update has non-record base");
                  }
                }
            }
            write_ty(fcx, id, bexpr_t);
            for f in fields_t {
                let found = false;
                for bf in base_fields {
                    if str::eq(f.node.ident, bf.ident) {
                        demand::ty(fcx.vb, f.span, demand::ek_mismatched_types,
                                   bf.mt.ty, f.node.mt.ty);
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
        let expr_t = expr_ty(fcx, base);
        let base_t = do_autoderef(fcx, expr.span, expr_t);
        let handled = false, n_tys = vec::len(tys);
        structure_of(fcx, expr.span, base_t) {|sty|
            alt sty {
              ty::ty_rec(fields) {
                alt ty::field_idx(field, fields) {
                  some(ix) {
                    if n_tys > 0u {
                        tcx.sess.span_err(expr.span,
                                          "can't provide type parameters \
                                           to a field access");
                    }
                    write_ty(fcx, id, fields[ix].mt.ty);
                    handled = true;
                  }
                  _ {}
                }
              }
              ty::ty_class(base_id, tps) {
                // (1) verify that the class id actually has a field called
                // field
                let cls_items = lookup_class_item_tys(tcx, base_id);
                let field_ty = lookup_field_ty(fcx, cls_items, field,
                                               expr.span, tps);
                // (2) look up what field's type is, and return it
                // FIXME: actually instantiate any type params
                write_ty(fcx, id, field_ty);
                handled = true;
              }
              _ {}
            }
        }
        if !handled {
            let tps = vec::map(tys, {|ty| ast_ty_to_ty_i(fcx, ty)});
            alt lookup_method(fcx, expr, expr.id, field, expr_t, tps) {
              some(origin) {
                fcx.ccx.method_map.insert(id, origin);
              }
              none {
                let msg = #fmt["attempted access of field %s on type %s, but \
                                no field or method implementation was found",
                               field, ty_str(fcx, expr_t)];
                tcx.sess.span_err(expr.span, msg);
                // NB: Adding a bogus type to allow typechecking to continue
                write_ty(fcx, id, next_ty_var(fcx));
              }
            }
        }
      }
      ast::expr_index(base, idx) {
        bot |= check_expr(fcx, base);
        let raw_base_t = expr_ty(fcx, base);
        let base_t = do_autoderef(fcx, expr.span, raw_base_t);
        bot |= check_expr(fcx, idx);
        let idx_t = expr_ty(fcx, idx);
        structure_of(fcx, expr.span, base_t) {|sty|
            alt sty {
              ty::ty_vec(mt) {
                require_integral(fcx, idx.span, idx_t);
                write_ty(fcx, id, mt.ty);
              }
              ty::ty_str {
                require_integral(fcx, idx.span, idx_t);
                let typ = ty::mk_mach_uint(fcx.vb, ast::ty_u8);
                write_ty(fcx, id, typ);
              }
              _ {
                alt lookup_op_method(fcx, expr, raw_base_t, "[]", [some(idx)]) {
                  some(ret_ty) { write_ty(fcx, id, ret_ty); }
                  _ {
                    tcx.sess.span_fatal(
                        expr.span, "cannot index a value of type `" +
                        ty_str(fcx, base_t) + "`");
                  }
                }
              }
            }
        }
      }
    }
    if bot { write_ty(fcx, expr.id, ty::mk_bot(fcx.vb)); }

    unifier(fcx, expr.span, expected, expr_ty(fcx, expr));
    ret bot;
}

fn require_integral(fcx: @fn_ctxt, sp: span, t: ty::t_i) {
    if !type_is_integral(fcx, sp, t) {
        fcx.ccx.tcx.sess.span_err(sp, "mismatched types: expected \
                                       `integer` but found `"
                                  + ty_str(fcx, t) + "`");
    }
}

fn next_ty_var_id(fcx: @fn_ctxt) -> int {
    let id = *fcx.next_var_id;
    *fcx.next_var_id += 1;
    ret id;
}

fn next_ty_var(fcx: @fn_ctxt) -> ty::t_i {
    ret ty::mk_var(next_ty_var_id(fcx));
}

fn next_ty_vars(fcx: @fn_ctxt, n: uint) -> [ty::t_i] {
    vec::from_fn(n) {|_i| next_ty_var(fcx)}
}

fn get_self_info(ccx: @crate_ctxt) -> option<self_info> {
    ret vec::last_opt(ccx.self_infos);
}

fn check_decl_initializer(fcx: @fn_ctxt, nid: ast::node_id,
                          init: ast::initializer) -> bool {
    let lty = ty::mk_var(lookup_local(fcx, init.expr.span, nid));
    ret check_expr_with(fcx, init.expr, lty);
}

fn check_decl_local(fcx: @fn_ctxt, local: @ast::local) -> bool {
    let bot = false;

    let t = ty::mk_var(fcx.locals.get(local.node.id));
    write_ty(fcx, local.node.id, t);
    alt local.node.init {
      some(init) {
        bot = check_decl_initializer(fcx, local.node.id, init);
      }
      _ {/* fall through */ }
    }

    let block_id = fcx.ccx.tcx.region_map.rvalue_to_block.get(local.node.id);
    let region = ty::re_block(block_id);
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
    let node_id;
    let bot = false;
    alt stmt.node {
      ast::stmt_decl(decl, id) {
        node_id = id;
        alt decl.node {
          ast::decl_local(ls) {
            for l in ls { bot |= check_decl_local(fcx, l); }
          }
          ast::decl_item(_) {/* ignore for now */ }
        }
      }
      ast::stmt_expr(expr, id) {
        node_id = id;
        bot = check_expr_with(fcx, expr, ty::mk_nil(fcx.vb));
      }
      ast::stmt_semi(expr, id) {
        node_id = id;
        bot = check_expr(fcx, expr);
      }
    }
    write_nil(fcx, node_id);
    ret bot;
}

fn check_block_no_value(fcx: @fn_ctxt, blk: ast::blk) -> bool {
    let bot = check_block(fcx, blk);
    if !bot {
        let blkty = node_ty(fcx, blk.node.id);
        let nilty = ty::mk_nil(fcx.vb);
        demand::ty(fcx.vb, blk.span, demand::ek_mismatched_types,
                   nilty, blkty);
    }
    ret bot;
}

fn check_block(fcx0: @fn_ctxt, blk: ast::blk) -> bool {
    let fcx = alt blk.node.rules {
      ast::unchecked_blk { @{purity: ast::impure_fn with *fcx0} }
      ast::unsafe_blk { @{purity: ast::unsafe_fn with *fcx0} }
      ast::default_blk { fcx0 }
    };
    let bot = false;
    let warned = false;
    for s: @ast::stmt in blk.node.stmts {
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
      none { write_nil(fcx, blk.node.id); }
      some(e) {
        if bot && !warned {
            fcx.ccx.tcx.sess.span_warn(e.span, "unreachable expression");
        }
        bot |= check_expr(fcx, e);
        let ety = expr_ty(fcx, e);
        write_ty(fcx, blk.node.id, ety);
      }
    }
    if bot {
        write_ty(fcx, blk.node.id, ty::mk_bot(fcx.vb));
    }
    ret bot;
}

fn check_const(ccx: @crate_ctxt, _sp: span, e: @ast::expr, id: ast::node_id) {
    // FIXME: this is kinda a kludge; we manufacture a fake function context
    // and statement context for checking the initializer expression.
    let rty = ty::node_id_to_type(ccx.tcx, id);
    let fcx: @fn_ctxt =
        @{ret_ty: ty::ty_to_ty_i(ccx.tcx, rty),
          purity: ast::pure_fn,
          proto: ast::proto_box,
          vb: ty::var_bindings(ccx.tcx),
          locals: int_hash(),
          next_var_id: @mutable 0,
          ccx: ccx};
    check_expr(fcx, e);
    writeback::resolve_type_vars_in_expr(fcx, e);

    let declty = fcx.ccx.tcx.tcache.get(local_def(id)).ty;
    demand::ty(fcx.ccx.tcx, e.span, demand::ek_mismatched_types,
               declty, ty::node_id_to_type(ccx.tcx, e.id));
}

fn check_enum_variants(ccx: @crate_ctxt, sp: span, vs: [ast::variant],
                      id: ast::node_id) {
    // FIXME: this is kinda a kludge; we manufacture a fake function context
    // and statement context for checking the initializer expression.
    let rty = ty::node_id_to_type(ccx.tcx, id);
    let fcx: @fn_ctxt =
        @{ret_ty: ty::ty_to_ty_i(ccx.tcx, rty),
          purity: ast::pure_fn,
          proto: ast::proto_box,
          vb: ty::var_bindings(ccx.tcx),
          locals: int_hash(),
          next_var_id: @mutable 0,
          ccx: ccx};
    let disr_vals: [int] = [];
    let disr_val = 0;
    for v in vs {
        alt v.node.disr_expr {
          some(e) {
            check_expr(fcx, e);
            let cty = expr_ty(fcx, e);
            let declty = ty::mk_int(fcx.vb);
            demand::ty(fcx.vb, e.span, demand::ek_mismatched_types,
                       declty, cty);
            writeback::resolve_type_vars_in_expr(fcx, e);

            // FIXME: issue #1417
            // Also, check_expr (from check_const pass) doesn't guarantee that
            // the expression in an form that eval_const_expr can handle, so
            // we may still get an internal compiler error
            alt syntax::ast_util::eval_const_expr(ccx.tcx, e) {
              syntax::ast_util::const_int(val) {
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
    let outer = true, did = local_def(id);
    if ty::type_structurally_contains(ccx.tcx, rty, {|sty|
        alt sty {
          ty::ty_enum(id, _) if id == did {
            if outer { outer = false; false }
            else { true }
          }
          _ { false }
        }
    }) {
        ccx.tcx.sess.span_fatal(sp, "illegal recursive enum type. \
                                     wrap the inner value in a box to \
                                     make it represenable");
    }
}

// A generic function for checking the pred in a check
// or if-check
fn check_pred_expr(fcx: @fn_ctxt, e: @ast::expr) -> bool {
    let bot = check_expr_with(fcx, e, ty::mk_bool(fcx.vb));

    /* e must be a call expr where all arguments are either
    literals or slots */
    alt e.node {
      ast::expr_call(operator, operands, _) {
        if !type_is_pred_ty(fcx, e.span, expr_ty(fcx, operator)) {
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
            for operand: @ast::expr in operands {
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
    let c_args;
    let num_args = vec::len(args);
    for c: @ast::constr in cs {
        c_args = [];
        for a: @spanned<ast::fn_constr_arg> in c.node.args {
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

// check_bare_fn: wrapper around check_fn for non-closure fns
fn check_bare_fn(ccx: @crate_ctxt,
                 decl: ast::fn_decl,
                 body: ast::blk,
                 id: ast::node_id) {
    let fty = ty::node_id_to_type(ccx.tcx, id);
    let ret_ty = ty::ty_to_ty_i(ccx.tcx, ty::ty_fn_ret(fty));
    let arg_tys = vec::map(ty::ty_fn_args(fty)) {|a| ty::ty_to_ty_i(ccx.tcx, a.ty) };
    check_fn(ccx, ast::proto_bare, decl, body, id, ret_ty, arg_tys, none);
}

fn check_fn(ccx: @crate_ctxt,
            proto: ast::proto,
            decl: ast::fn_decl,
            body: ast::blk,
            id: ast::node_id,
            ret_ty: ty::t_i,
            arg_tys: [ty::t_i],
            old_fcx: option<@fn_ctxt>) {
    // If old_fcx is some(...), this is a block fn { |x| ... }.
    // In that case, the purity is inherited from the context.
    let purity = alt old_fcx {
      none { decl.purity }
      some(f) { assert decl.purity == ast::impure_fn; f.purity }
    };

    let gather_result = gather_locals(ccx, decl, body, id, arg_tys, old_fcx);
    let fcx: @fn_ctxt =
        @{ret_ty: ret_ty,
          purity: purity,
          proto: proto,
          vb: gather_result.var_bindings,
          locals: gather_result.locals,
          next_var_id: gather_result.next_var_id,
          ccx: ccx};

    check_constraints(fcx, decl.constraints, decl.inputs);
    check_block(fcx, body);

    // We unify the tail expr's type with the
    // function result type, if there is a tail expr.
    alt body.node.expr {
      some(tail_expr) {
        let tail_expr_ty = expr_ty(fcx, tail_expr);
        demand::ty(fcx.vb, tail_expr.span, demand::ek_mismatched_types,
                   fcx.ret_ty, tail_expr_ty);
      }
      none { }
    }

    vec::iter2(decl.inputs, arg_tys) {|inp,aty| write_ty(fcx, inp.id, aty) };

    // If we don't have any enclosing function scope, it is time to
    // force any remaining type vars to be resolved.
    // If we have an enclosing function scope, our type variables will be
    // resolved when the enclosing scope finishes up.
    if option::is_none(old_fcx) {
        vtable::resolve_in_block(fcx, body);
        writeback::resolve_type_vars_in_fn(fcx, decl, body);
    }
}

fn check_method(ccx: @crate_ctxt, method: @ast::method) {
    check_bare_fn(ccx, method.decl, method.body, method.id);
}

fn class_types(ccx: @crate_ctxt, members: [@ast::class_item]) -> class_map {
    let rslt = int_hash::<ty::t>();
    for m in members {
      alt m.node.decl {
         ast::instance_var(_,t,_,id) {
           rslt.insert(id, ast_ty_to_ty(ccx, m_collect, t));
         }
         ast::class_method(it) {
             rslt.insert(it.id, ty_of_item(ccx, m_collect, it).ty);
         }
      }
    }
    rslt
}

fn check_class_member(ccx: @crate_ctxt, cm: ast::class_member) {
    alt cm {
      ast::instance_var(_,t,_,_) { // ??? Not sure
      }
      // not right yet -- need a scope
      ast::class_method(i) { check_item(ccx, i); }
    }
}

fn check_item(ccx: @crate_ctxt, it: @ast::item) {
    alt it.node {
      ast::item_const(_, e) { check_const(ccx, it.span, e, it.id); }
      ast::item_enum(vs, _) { check_enum_variants(ccx, it.span, vs, it.id); }
      ast::item_fn(decl, tps, body) {
        check_bare_fn(ccx, decl, body, it.id);
      }
      ast::item_res(decl, tps, body, dtor_id, _) {
        check_bare_fn(ccx, decl, body, dtor_id);
      }
      ast::item_impl(tps, _, ty, ms) {
        let self_ty = ast_ty_to_ty(ccx, m_check, ty);
        let self_region = ty::re_self({crate: ast::local_crate, node: it.id});
        self_ty = instantiate_self_regions(ccx.tcx, self_region, self_ty);
        ccx.self_infos += [self_impl(self_ty)];
        for m in ms { check_method(ccx, m); }
        vec::pop(ccx.self_infos);
      }
      ast::item_class(tps, members, ctor) {
          let cid = some(it.id);
          let members_info = class_types(ccx, members);
          let class_ccx = @{enclosing_class_id:cid,
                            enclosing_class:members_info with *ccx};

          // typecheck the ctor
          check_bare_fn(class_ccx, ctor.node.dec, ctor.node.body,
                        ctor.node.id);

          // typecheck the members
          for m in members { check_class_member(class_ccx, m.node.decl); }
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
        let ok = vec::len(constraints) == 0u;
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
                      bounds: @[ty::param_bounds], tys: [ty::t_i],
                      allow_unsafe: bool) -> vtable_res {
        let tcx = fcx.ccx.tcx, result = [], i = 0u;
        for ty in tys {
            for bound in *bounds[i] {
                alt bound {
                  ty::bound_iface(i_ty) {
                    let i_ty = ty::ty_to_ty_i_subst(tcx, i_ty, tys);
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
                     ty: ty::t_i, iface_ty: ty::t_i, allow_unsafe: bool)
        -> vtable_origin {
        let tcx = fcx.ccx.tcx;
        let (iface_id, iface_tps) = structure_of(fcx, sp, iface_ty) {|sty|
            alt check sty {
              ty::ty_iface(did, tps) { (did, tps) }
            }
        };

        let tyf = fixup_ty(fcx, sp, ty);
        alt ty::get(tyf).struct {
          ty::ty_param(n, did) {
            let n_bound = 0u;
            for bound in *tcx.ty_param_bounds.get(did.node) {
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
                for m in *ty::iface_methods(tcx, did) {
                    if ty::type_has_self(ty::mk_fn(tcx, m.fty)) {
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
          _ { /*fallthrough */ }
        }

        let found = none;
        std::list::iter(isc) {|impls|
            if option::is_some(found) { ret; }
            for im in *impls {
                let match = alt ty::impl_iface(tcx, im.did) {
                  some(ity) {
                    alt check ty::get(ity).struct {
                      ty::ty_iface(id, _) { id == iface_id }
                    }
                  }
                  _ { false }
                };
                if match {
                    let {vars, self_ty} = impl_self_ty(fcx, im.did);
                    let im_bs = ty::lookup_item_type(tcx, im.did).bounds;
                    alt ty::unify::unify(fcx.vb, ty, self_ty) {
                      result::ok(_) {
                        if option::is_some(found) {
                            tcx.sess.span_err(
                                sp, "multiple applicable implementations \
                                     in scope");
                        } else {
                            connect_iface_tps(fcx, sp, vars, iface_tps, im.did);
                            let subres = lookup_vtables(fcx, isc, sp, im_bs, vars,
                                                        allow_unsafe);
                            let params = vec::map(vars) {|v| fixup_ty(fcx, sp, v) };
                            found = some(vtable_static(im.did, params, subres));
                        }
                      }
                      result::err(_) {}
                    }
                }
            }
        }
        alt found {
          some(rslt) { ret rslt; }
          _ {}
        }

        tcx.sess.span_fatal(
            sp, "failed to find an implementation of interface " +
            ty_str(fcx, iface_ty) + " for " +
            ty_str(fcx, ty));
    }

    fn fixup_ty(fcx: @fn_ctxt, sp: span, ty: ty::t_i) -> ty::t {
        let tcx = fcx.ccx.tcx;
        alt ty::unify::resolve_type(fcx.vb, ty) {
          result::ok(new_type) { new_type }
          result::err(rerr) {
            tcx.sess.span_fatal(sp, ty::resolve_err_to_str(rerr));
          }
        }
    }

    fn connect_iface_tps(fcx: @fn_ctxt, sp: span, impl_tys: [ty::t_i],
                         iface_tys: [ty::t_i], impl_did: ast::def_id) {
        let tcx = fcx.ccx.tcx;
        let ity = option::get(ty::impl_iface(tcx, impl_did));
        let iface_ty = ty::ty_to_ty_i_subst(tcx, ity, impl_tys);
        structure_of(fcx, sp, iface_ty) {|sty|
            alt check sty {
              ty::ty_iface(_, tps) {
                demand::tys(fcx.vb, sp, demand::ek_mismatched_types,
                            tps, iface_tys);
              }
            }
        }
    }

    fn resolve_expr(ex: @ast::expr, &&fcx: @fn_ctxt, v: visit::vt<@fn_ctxt>) {
        let cx = fcx.ccx;
        alt ex.node {
          ast::expr_path(_) {
            alt node_ty_substs_find(fcx, ex.id) {
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
                    let ts = node_ty_substs(fcx, callee_id);
                    let iscs = cx.impl_map.get(ex.id);
                    cx.vtable_map.insert(callee_id, lookup_vtables(
                        fcx, iscs, ex.span, bounds, ts, false));
                }
              }
              _ {}
            }
          }
          ast::expr_cast(src, _) {
            let target_ty = expr_ty(fcx, ex);
            structure_of(fcx, ex.span, target_ty) {|sty|
                alt sty {
                  ty::ty_iface(_, _) {
                    let impls = cx.impl_map.get(ex.id);
                    let vtable = lookup_vtable(fcx, impls, ex.span,
                                               expr_ty(fcx, src), target_ty,
                                               true);
                    cx.vtable_map.insert(ex.id, @[vtable]);
                  }
                  _ {}
                }
            }
          }
          ast::expr_fn(p, _, _, _) if ast::is_blockish(p) {}
          ast::expr_fn(_, _, _, _) { ret; }
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
    let ccx = @{mutable self_infos: [],
                impl_map: impl_map,
                method_map: std::map::int_hash(),
                vtable_map: std::map::int_hash(),
                enclosing_class_id: none,
                enclosing_class: std::map::int_hash(),
                ast_ty_to_ty_cache: map::hashmap(
                    ast_util::hash_ty, ast_util::eq_ty),
                tcx: tcx};
    collect::collect_item_types(ccx, crate);
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
