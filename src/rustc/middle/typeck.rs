import result::result;
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
import middle::ty::{node_id_to_type, arg, block_ty,
                    expr_ty, field, node_type_table, mk_nil,
                    ty_param_bounds_and_ty, lookup_class_item_tys};
import util::ppaux::ty_to_str;
import std::smallintmap;
import std::map::{hashmap, int_hash};
import std::serialization::{serialize_uint, deserialize_uint};
import syntax::print::pprust::*;

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

// Used for typechecking the methods of an impl
enum self_info { self_impl(ty::t) }

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
                   tcx: ty::ctxt};

type class_map = hashmap<ast::node_id, ty::t>;

type fn_ctxt =
    // var_bindings, locals and next_var_id are shared
    // with any nested functions that capture the environment
    // (and with any functions whose environment is being captured).
    {ret_ty: ty::t,
     purity: ast::purity,
     proto: ast::proto,
     var_bindings: @ty::unify::var_bindings,
     locals: hashmap<ast::node_id, int>,
     next_var_id: @mutable int,
     ccx: @crate_ctxt};


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
        alt get_self_info(fcx.ccx) {
          some(self_impl(impl_t)) {
            ret {bounds: @[], ty: impl_t};
          }
          none {
              fcx.ccx.tcx.sess.span_bug(sp, "def_self with no self_info");
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
      ast::def_class_method(_, id) | ast::def_class_field(_, id) {
          if id.crate != ast::local_crate {
                  fcx.ccx.tcx.sess.span_fatal(sp,
                                 "class method or field referred to \
                                  out of scope");
          }
          alt fcx.ccx.enclosing_class.find(id.node) {
             some(a_ty) { ret {bounds: @[], ty: a_ty}; }
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
                (sp, "too many type parameter provided for this item");
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
        write_ty_substs(fcx.ccx.tcx, id, tpt.ty, substs);
    } else if ty_param_count > 0u {
        let vars = vec::from_fn(ty_param_count, {|_i| next_ty_var(fcx)});
        write_ty_substs(fcx.ccx.tcx, id, tpt.ty, vars);
    } else {
        write_ty(fcx.ccx.tcx, id, tpt.ty);
    }
}

// Type tests
fn structurally_resolved_type(fcx: @fn_ctxt, sp: span, tp: ty::t) -> ty::t {
    alt ty::unify::resolve_type_structure(fcx.var_bindings, tp) {
      result::ok(typ_s) { ret typ_s; }
      result::err(_) {
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
    let r = ty::unify::resolve_type_structure(fcx.var_bindings, typ);
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

// Parses the programmer's textual representation of a type into our
// internal notion of a type. `getter` is a function that returns the type
// corresponding to a definition ID:
fn ast_ty_to_ty(tcx: ty::ctxt, mode: mode, &&ast_ty: @ast::ty) -> ty::t {
    fn subst_inferred_regions(tcx: ty::ctxt, use_site: ast::node_id,
                              ty: ty::t) -> ty::t {
        ret ty::fold_ty(tcx, ty::fm_rptr({|r|
            alt r {
                ty::re_inferred | ty::re_self(_) {
                    tcx.region_map.ast_type_to_inferred_region.get(use_site)
                }
                _ { r }
            }
        }), ty);
    }
    fn getter(tcx: ty::ctxt, use_site: ast::node_id, mode: mode,
              id: ast::def_id) -> ty::ty_param_bounds_and_ty {
        let tpt = alt mode {
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
        };

        if ty::type_has_rptrs(tpt.ty) {
            ret {bounds: tpt.bounds,
                 ty: subst_inferred_regions(tcx, use_site, tpt.ty)};
        }
        ret tpt;
    }
    fn ast_mt_to_mt(tcx: ty::ctxt, use_site: ast::node_id, mode: mode,
                    mt: ast::mt) -> ty::mt {
        ret {ty: do_ast_ty_to_ty(tcx, use_site, mode, mt.ty),
             mutbl: mt.mutbl};
    }
    fn instantiate(tcx: ty::ctxt, use_site: ast::node_id, sp: span,
                   mode: mode, id: ast::def_id, path_id: ast::node_id,
                   args: [@ast::ty]) -> ty::t {
        let ty_param_bounds_and_ty = getter(tcx, use_site, mode, id);
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
            param_bindings += [do_ast_ty_to_ty(tcx, use_site, mode, ast_ty)];
        }
        #debug("substituting(%s into %s)",
               str::concat(vec::map(param_bindings, {|t| ty_to_str(tcx, t)})),
               ty_to_str(tcx, ty_param_bounds_and_ty.ty));
        let typ =
            ty::substitute_type_params(tcx, param_bindings,
                                       ty_param_bounds_and_ty.ty);
        write_substs(tcx, path_id, param_bindings);
        ret typ;
    }
    fn do_ast_ty_to_ty(tcx: ty::ctxt, use_site: ast::node_id, mode: mode,
                       &&ast_ty: @ast::ty) -> ty::t {
        alt tcx.ast_ty_to_ty_cache.find(ast_ty) {
          some(ty::atttce_resolved(ty)) { ret ty; }
          some(ty::atttce_unresolved) {
            tcx.sess.span_fatal(ast_ty.span, "illegal recursive type. \
                                              insert a enum in the cycle, \
                                              if this is desired)");
          }
          some(ty::atttce_has_regions) | none { /* go on */ }
        }

        tcx.ast_ty_to_ty_cache.insert(ast_ty, ty::atttce_unresolved);
        let typ = alt ast_ty.node {
          ast::ty_nil { ty::mk_nil(tcx) }
          ast::ty_bot { ty::mk_bot(tcx) }
          ast::ty_box(mt) {
            ty::mk_box(tcx, ast_mt_to_mt(tcx, use_site, mode, mt))
          }
          ast::ty_uniq(mt) {
            ty::mk_uniq(tcx, ast_mt_to_mt(tcx, use_site, mode, mt))
          }
          ast::ty_vec(mt) {
            ty::mk_vec(tcx, ast_mt_to_mt(tcx, use_site, mode, mt))
          }
          ast::ty_ptr(mt) {
            ty::mk_ptr(tcx, ast_mt_to_mt(tcx, use_site, mode, mt))
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
            ty::mk_rptr(tcx, region, ast_mt_to_mt(tcx, use_site, mode, mt))
          }
          ast::ty_tup(fields) {
            let flds = vec::map(fields,
                                bind do_ast_ty_to_ty(tcx, use_site, mode, _));
            ty::mk_tup(tcx, flds)
          }
          ast::ty_rec(fields) {
            let flds: [field] = [];
            for f: ast::ty_field in fields {
                let tm = ast_mt_to_mt(tcx, use_site, mode, f.node.mt);
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
              ast::def_ty(did) {
                instantiate(tcx, use_site, ast_ty.span, mode, did,
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
                        do_ast_ty_to_ty(tcx, use_site, mode, ast_ty)
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
                          {|ast_ty| ast_ty_to_ty(tcx, mode, ast_ty)}))
                     }
                   _ {
                      tcx.sess.span_bug(ast_ty.span, #fmt("class id is \
                        unbound in items"));
                   }
                }
              }
              else {
                  getter(tcx, use_site, mode, class_id).ty
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
            ty::mk_constr(tcx, do_ast_ty_to_ty(tcx, use_site, mode, t),
                          out_cs)
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

        if ty::type_has_rptrs(typ) {
            tcx.ast_ty_to_ty_cache.insert(ast_ty, ty::atttce_has_regions);
        } else {
            tcx.ast_ty_to_ty_cache.insert(ast_ty, ty::atttce_resolved(typ));
        }

        ret typ;
    }

    ret do_ast_ty_to_ty(tcx, ast_ty.id, mode, ast_ty);
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
    let input_tys = vec::map(decl.inputs) {|a| ty_of_arg(tcx, mode, a) };
    let output_ty = ast_ty_to_ty(tcx, mode, decl.output);

    let out_constrs = [];
    for constr: @ast::constr in decl.constraints {
        out_constrs += [ty::ast_constr_to_constr(tcx, constr)];
    }
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
    let input_tys = [], bounds = ty_param_bounds(tcx, mode, ty_params);
    for a: ast::arg in decl.inputs {
        input_tys += [ty_of_arg(tcx, mode, a)];
    }
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
     purity: m.decl.purity}
}
fn ty_of_ty_method(tcx: ty::ctxt, mode: mode, m: ast::ty_method)
    -> ty::method {
    {ident: m.ident, tps: ty_param_bounds(tcx, mode, m.tps),
     fty: ty_of_fn_decl(tcx, mode, ast::proto_bare, m.decl),
     purity: m.decl.purity}
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
fn write_ty(tcx: ty::ctxt, node_id: ast::node_id, ty: ty::t) {
    smallintmap::insert(*tcx.node_types, node_id as uint, ty);
}
fn write_substs(tcx: ty::ctxt, node_id: ast::node_id, +substs: [ty::t]) {
    tcx.node_type_substs.insert(node_id, substs);
}
fn write_ty_substs(tcx: ty::ctxt, node_id: ast::node_id, ty: ty::t,
                   +substs: [ty::t]) {
    let ty = if ty::type_has_params(ty) {
        ty::substitute_type_params(tcx, substs, ty)
    } else { ty };
    write_ty(tcx, node_id, ty);
    write_substs(tcx, node_id, substs);
}
fn write_nil(tcx: ty::ctxt, node_id: ast::node_id) {
    write_ty(tcx, node_id, ty::mk_nil(tcx));
}
fn write_bot(tcx: ty::ctxt, node_id: ast::node_id) {
    write_ty(tcx, node_id, ty::mk_bot(tcx));
}

fn mk_ty_params(tcx: ty::ctxt, atps: [ast::ty_param])
    -> {bounds: @[ty::param_bounds], params: [ty::t]} {
    let i = 0u, bounds = ty_param_bounds(tcx, m_collect, atps);
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
        let if_fty = ty::mk_fn(tcx, if_m.fty);
        if_fty = ty::substitute_type_params(tcx, substs, if_fty);
        if ty::type_has_vars(if_fty) {
            if_fty = fixup_self_in_method_ty(tcx, if_fty, substs,
                                             self_full(self_ty, impl_tps));
        }
        alt ty::unify::unify(impl_fty, if_fty, ty::unify::precise, tcx) {
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

enum self_subst { self_param(ty::t, @fn_ctxt, span), self_full(ty::t, uint) }

// Mangles an iface method ty to make its self type conform to the self type
// of a specific impl or bounded type parameter. This is rather involved
// because the type parameters of ifaces and impls are not required to line up
// (an impl can have less or more parameters than the iface it implements), so
// some mangling of the substituted types is required.
fn fixup_self_in_method_ty(cx: ty::ctxt, mty: ty::t, m_substs: [ty::t],
                           self: self_subst) -> ty::t {
    if ty::type_has_vars(mty) {
        ty::fold_ty(cx, ty::fm_general(fn@(t: ty::t) -> ty::t {
            alt ty::get(t).struct {
              ty::ty_self(tps) {
                if vec::len(tps) > 0u {
                    // Move the substs into the type param system of the
                    // context.
                    let substs = vec::map(tps, {|t|
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
        }), mty)
    } else { mty }
}

// Mangles an iface method ty to instantiate its `self` region.
fn fixup_self_region_in_method_ty(fcx: @fn_ctxt, mty: ty::t,
                                  self_expr: @ast::expr) -> ty::t {
    let self_region = region_of(fcx, self_expr);
    ty::fold_ty(fcx.ccx.tcx, ty::fm_rptr({|r|
        alt r {
            ty::re_self(_) { self_region }
            _ { r }
        }
    }), mty)
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
                    let arg_ty = ast_ty_to_ty(tcx, m_collect, va.ty);
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
            write_ty(tcx, variant.node.id, result_ty);
        }
    }
    fn ensure_iface_methods(tcx: ty::ctxt, id: ast::node_id) {
        alt check tcx.items.get(id) {
          ast_map::node_item(@{node: ast::item_iface(_, ms), _}, _) {
            ty::store_iface_methods(tcx, id, @vec::map(ms, {|m|
                ty_of_ty_method(tcx, m_collect, m)
            }));
          }
        }
    }
    fn convert_class_item(tcx: ty::ctxt, ci: ast::class_member) {
        /* we want to do something here, b/c within the
         scope of the class, it's ok to refer to fields &
        methods unqualified */

        /* they have these types *within the scope* of the
         class. outside the class, it's done with expr_field */
        alt ci {
         ast::instance_var(_,t,_,id) {
             let tt = ast_ty_to_ty(tcx, m_collect, t);
             write_ty(tcx, id, tt);
         }
         ast::class_method(it) { convert(tcx, it); }
        }
    }
    fn convert(tcx: ty::ctxt, it: @ast::item) {
        alt it.node {
          // These don't define types.
          ast::item_mod(_) | ast::item_native_mod(_) {}
          ast::item_enum(variants, ty_params) {
            let tpt = ty_of_item(tcx, m_collect, it);
            write_ty(tcx, it.id, tpt.ty);
            get_enum_variant_types(tcx, tpt.ty, variants, ty_params);
          }
          ast::item_impl(tps, ifce, selfty, ms) {
            let i_bounds = ty_param_bounds(tcx, m_collect, tps);
            let my_methods = [];
            let selfty = ast_ty_to_ty(tcx, m_collect, selfty);
            write_ty(tcx, it.id, selfty);
            tcx.tcache.insert(local_def(it.id), {bounds: i_bounds,
                                                 ty: selfty});
            for m in ms {
                write_ty(tcx, m.self_id, selfty);
                let bounds = ty_param_bounds(tcx, m_collect, m.tps);
                let mty = ty_of_method(tcx, m_collect, m);
                my_methods += [{mty: mty, id: m.id, span: m.span}];
                let fty = ty::mk_fn(tcx, mty.fty);
                tcx.tcache.insert(local_def(m.id),
                                     {bounds: @(*i_bounds + *bounds),
                                      ty: fty});
                write_ty(tcx, m.id, fty);
            }
            alt ifce {
              some(t) {
                let iface_ty = ast_ty_to_ty(tcx, m_collect, t);
                alt ty::get(iface_ty).struct {
                  ty::ty_iface(did, tys) {
                    // Store the iface type in the type node
                    alt check t.node {
                      ast::ty_path(_, t_id) { write_ty(tcx, t_id, iface_ty); }
                    }
                    if did.crate == ast::local_crate {
                        ensure_iface_methods(tcx, did.node);
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
                                write_ty(tcx, id, mt);
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
            write_ty(tcx, it.id, t_res);
            write_ty(tcx, ctor_id, t_ctor);
            tcx.tcache.insert(local_def(ctor_id),
                              {bounds: bounds, ty: t_ctor});
            tcx.tcache.insert(def_id, {bounds: bounds, ty: t_res});
            write_ty(tcx, dtor_id, t_dtor);
          }
          ast::item_iface(_, ms) {
            let tpt = ty_of_item(tcx, m_collect, it);
            write_ty(tcx, it.id, tpt.ty);
            ensure_iface_methods(tcx, it.id);
          }
          ast::item_class(tps, members, ctor) {
              // Write the class type
              let tpt = ty_of_item(tcx, m_collect, it);
              write_ty(tcx, it.id, tpt.ty);
              // Write the ctor type
              let t_ctor = ty::mk_fn(tcx,
                                     ty_of_fn_decl(tcx, m_collect,
                                             ast::proto_any, ctor.node.dec));
              write_ty(tcx, ctor.node.id, t_ctor);
              tcx.tcache.insert(local_def(ctor.node.id),
                                   {bounds: tpt.bounds, ty: t_ctor});
              /* FIXME: check for proper public/privateness */
              // Write the type of each of the members
              for m in members {
                 convert_class_item(tcx, m.node.decl);
              }
          }
          _ {
            // This call populates the type cache with the converted type
            // of the item in passing. All we have to do here is to write
            // it into the node type table.
            let tpt = ty_of_item(tcx, m_collect, it);
            write_ty(tcx, it.id, tpt.ty);
          }
        }
    }
    fn convert_native(tcx: ty::ctxt, i: @ast::native_item) {
        // As above, this call populates the type table with the converted
        // type of the native item. We simply write it into the node type
        // table.
        let tpt = ty_of_native_item(tcx, m_collect, i);
        alt i.node {
          ast::native_item_fn(_, _) { write_ty(tcx, i.id, tpt.ty); }
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
        result<ty::t, ty::type_err> {
        ret ty::unify::unify(expected, actual,
                             ty::unify::in_bindings(fcx.var_bindings),
                             fcx.ccx.tcx);
    }
}


// FIXME This is almost a duplicate of ty::type_autoderef, with structure_of
// instead of ty::struct.
fn do_autoderef(fcx: @fn_ctxt, sp: span, t: ty::t) -> ty::t {
    let t1 = t;
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
    alt ty::unify::fixup_vars(fcx.ccx.tcx, none, fcx.var_bindings, typ) {
      result::ok(new_type) { ret new_type; }
      result::err(_) { ret typ; }
    }
}


// Demands - procedures that require that two types unify and emit an error
// message if they don't.
type ty_param_substs_and_ty = {substs: [ty::t], ty: ty::t};

mod demand {
    fn simple(fcx: @fn_ctxt, sp: span, expected: ty::t, actual: ty::t) ->
       ty::t {
        full(fcx, sp, expected, actual, []).ty
    }

    fn with_substs(fcx: @fn_ctxt, sp: span, expected: ty::t, actual: ty::t,
                   ty_param_substs_0: [ty::t]) -> ty_param_substs_and_ty {
        full(fcx, sp, expected, actual, ty_param_substs_0)
    }

    // Requires that the two types unify, and prints an error message if they
    // don't. Returns the unified type and the type parameter substitutions.
    fn full(fcx: @fn_ctxt, sp: span, expected: ty::t, actual: ty::t,
            ty_param_substs_0: [ty::t]) ->
       ty_param_substs_and_ty {

        let ty_param_substs: [mutable ty::t] = [mutable];
        let ty_param_subst_var_ids: [int] = [];
        for ty_param_subst: ty::t in ty_param_substs_0 {
            // Generate a type variable and unify it with the type parameter
            // substitution. We will then pull out these type variables.
            let t_0 = next_ty_var(fcx);
            ty_param_substs += [mutable t_0];
            ty_param_subst_var_ids += [ty::ty_var_id(t_0)];
            simple(fcx, sp, ty_param_subst, t_0);
        }

        fn mk_result(fcx: @fn_ctxt, result_ty: ty::t,
                     ty_param_subst_var_ids: [int]) ->
           ty_param_substs_and_ty {
            let result_ty_param_substs: [ty::t] = [];
            for var_id: int in ty_param_subst_var_ids {
                let tp_subst = ty::mk_var(fcx.ccx.tcx, var_id);
                result_ty_param_substs += [tp_subst];
            }
            ret {substs: result_ty_param_substs, ty: result_ty};
        }


        alt unify::unify(fcx, expected, actual) {
          result::ok(t) { ret mk_result(fcx, t, ty_param_subst_var_ids); }
          result::err(err) {
            let e_err = resolve_type_vars_if_possible(fcx, expected);
            let a_err = resolve_type_vars_if_possible(fcx, actual);
            fcx.ccx.tcx.sess.span_err(sp,
                                      "mismatched types: expected `" +
                                          ty_to_str(fcx.ccx.tcx, e_err) +
                                          "` but found `" +
                                          ty_to_str(fcx.ccx.tcx, a_err) +
                                          "` (" +
                                          ty::type_err_to_str(fcx.ccx.tcx,
                                                              err) +
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
    let result: [ty::t] = [];
    let tpt = ty::lookup_item_type(ccx.tcx, vid);
    alt ty::get(tpt.ty).struct {
      ty::ty_fn(f) {
        // N-ary variant.
        for arg: ty::arg in f.inputs {
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

    export resolve_type_vars_in_block;
    export resolve_type_vars_in_expr;

    fn resolve_type_vars_in_type(fcx: @fn_ctxt, sp: span, typ: ty::t) ->
       option<ty::t> {
        if !ty::type_has_vars(typ) { ret some(typ); }
        alt ty::unify::fixup_vars(fcx.ccx.tcx, some(sp), fcx.var_bindings,
                                  typ) {
          result::ok(new_type) { ret some(new_type); }
          result::err(vid) {
            if !fcx.ccx.tcx.sess.has_errors() {
                fcx.ccx.tcx.sess.span_err(sp, "cannot determine a type \
                                               for this expression");
            }
            ret none;
          }
        }
    }
    fn resolve_type_vars_for_node(wbcx: wb_ctxt, sp: span, id: ast::node_id)
        -> option<ty::t> {
        let fcx = wbcx.fcx, tcx = fcx.ccx.tcx;
        alt resolve_type_vars_in_type(fcx, sp, ty::node_id_to_type(tcx, id)) {
          none {
            wbcx.success = false;
            ret none;
          }

          some(t) {
            write_ty(tcx, id, t);
            alt tcx.node_type_substs.find(id) {
              some(substs) {
                let new_substs = [];
                for subst: ty::t in substs {
                    alt resolve_type_vars_in_type(fcx, sp, subst) {
                      some(t) { new_substs += [t]; }
                      none { wbcx.success = false; ret none; }
                    }
                }
                write_substs(tcx, id, new_substs);
              }
              none {}
            }
            ret some(t);
          }
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
        let fix_rslt =
            ty::unify::resolve_type_var(wbcx.fcx.ccx.tcx, some(l.span),
                                        wbcx.fcx.var_bindings, var_id);
        alt fix_rslt {
          result::ok(lty) { write_ty(wbcx.fcx.ccx.tcx, l.node.id, lty); }
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
        visit::visit_expr(e, wbcx, visit);
        ret wbcx.success;
    }

    fn resolve_type_vars_in_block(fcx: @fn_ctxt, blk: ast::blk) -> bool {
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
        ret wbcx.success;
    }
}


// Local variable gathering. We gather up all locals and create variable IDs
// for them before typechecking the function.
type gather_result =
    {var_bindings: @ty::unify::var_bindings,
     locals: hashmap<ast::node_id, int>,
     next_var_id: @mutable int};

// Used only as a helper for check_fn.
fn gather_locals(ccx: @crate_ctxt,
                 decl: ast::fn_decl,
                 body: ast::blk,
                 id: ast::node_id,
                 old_fcx: option<@fn_ctxt>) -> gather_result {
    let {vb: vb, locals: locals, nvi: nvi} = alt old_fcx {
      none {
        {vb: ty::unify::mk_var_bindings(),
         locals: int_hash::<int>(),
         nvi: @mutable 0}
      }
      some(fcx) {
        {vb: fcx.var_bindings,
         locals: fcx.locals,
         nvi: fcx.next_var_id}
      }
    };
    let tcx = ccx.tcx;

    let next_var_id = fn@() -> int { let rv = *nvi; *nvi += 1; ret rv; };

    let assign = fn@(nid: ast::node_id, ty_opt: option<ty::t>) {
        let var_id = next_var_id();
        locals.insert(nid, var_id);
        alt ty_opt {
          none {/* nothing to do */ }
          some(typ) {
            ty::unify::unify(ty::mk_var(tcx, var_id), typ,
                             ty::unify::in_bindings(vb), tcx);
          }
        }
    };

    // Add formal parameters.
    let args = ty::ty_fn_args(ty::node_id_to_type(ccx.tcx, id));
    let i = 0u;
    for arg: ty::arg in args {
        assign(decl.inputs[i].id, some(arg.ty));
        i += 1u;
    }

    // Add explicitly-declared locals.
    let visit_local = fn@(local: @ast::local, &&e: (), v: visit::vt<()>) {
        let local_ty = ast_ty_to_ty_crate_infer(ccx, local.node.ty);
        assign(local.node.id, local_ty);
        visit::visit_local(local, e, v);
    };

    // Add pattern bindings.
    let visit_pat = fn@(p: @ast::pat, &&e: (), v: visit::vt<()>) {
        alt p.node {
          ast::pat_ident(_, _)
          if !pat_util::pat_is_variant(ccx.tcx.def_map, p) {
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
fn instantiate_self_regions(tcx: ty::ctxt, region: ty::region, &&ty: ty::t)
        -> ty::t {
    if ty::type_has_rptrs(ty) {
        ty::fold_ty(tcx, ty::fm_rptr({|r|
            alt r {
                ty::re_inferred | ty::re_caller(_) | ty::re_self(_) { region }
                _ { r }
            }
        }), ty)
    } else {
        ty
    }
}

// Replaces all region variables in the given type with "inferred regions".
// This is used during method lookup to allow typeclass implementations to
// refer to inferred regions.
fn universally_quantify_regions(tcx: ty::ctxt, ty: ty::t) -> ty::t {
    if ty::type_has_rptrs(ty) {
        ty::fold_ty(tcx, ty::fm_rptr({|_r| ty::re_inferred}), ty)
    } else {
        ty
    }
}

fn check_pat_variant(pcx: pat_ctxt, pat: @ast::pat, path: @ast::path,
                     subpats: [@ast::pat], expected: ty::t) {
    // Typecheck the path.
    let tcx = pcx.fcx.ccx.tcx;
    let v_def = lookup_def(pcx.fcx, path.span, pat.id);
    let v_def_ids = ast_util::variant_def_ids(v_def);
    let ctor_tpt = ty::lookup_item_type(tcx, v_def_ids.enm);
    instantiate_path(pcx.fcx, path, ctor_tpt, pat.span, pat.id);

    // Take the enum type params out of `expected`.
    alt structure_of(pcx.fcx, pat.span, expected) {
      ty::ty_enum(_, expected_tps) {
        let ctor_ty = ty::node_id_to_type(tcx, pat.id);
        demand::with_substs(pcx.fcx, pat.span, expected, ctor_ty,
                            expected_tps);
        // Get the number of arguments in this enum variant.
        let arg_types = variant_arg_types(pcx.fcx.ccx, pat.span,
                                          v_def_ids.var, expected_tps);
        arg_types = vec::map(arg_types,
                             bind instantiate_self_regions(pcx.fcx.ccx.tcx,
                                                           pcx.pat_region,
                                                           _));
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
    let tcx = pcx.fcx.ccx.tcx;
    alt pat.node {
      ast::pat_wild {
        write_ty(tcx, pat.id, expected);
      }
      ast::pat_lit(lt) {
        check_expr_with(pcx.fcx, lt, expected);
        write_ty(tcx, pat.id, expr_ty(tcx, lt));
      }
      ast::pat_range(begin, end) {
        check_expr_with(pcx.fcx, begin, expected);
        check_expr_with(pcx.fcx, end, expected);
        let b_ty = resolve_type_vars_if_possible(pcx.fcx,
                                                 expr_ty(tcx, begin));
        if !ty::same_type(tcx, b_ty, resolve_type_vars_if_possible(
            pcx.fcx, expr_ty(tcx, end))) {
            tcx.sess.span_err(pat.span, "mismatched types in range");
        } else if !ty::type_is_numeric(b_ty) {
            tcx.sess.span_err(pat.span, "non-numeric type used in range");
        } else if !valid_range_bounds(tcx, begin, end) {
            tcx.sess.span_err(begin.span, "lower range bound must be less \
                                           than upper");
        }
        write_ty(tcx, pat.id, b_ty);
      }
      ast::pat_ident(name, sub)
      if !pat_util::pat_is_variant(tcx.def_map, pat) {
        let vid = lookup_local(pcx.fcx, pat.span, pat.id);
        let typ = ty::mk_var(tcx, vid);
        typ = demand::simple(pcx.fcx, pat.span, expected, typ);
        let canon_id = pcx.map.get(path_to_ident(name));
        if canon_id != pat.id {
            let tv_id = lookup_local(pcx.fcx, pat.span, canon_id);
            let ct = ty::mk_var(tcx, tv_id);
            typ = demand::simple(pcx.fcx, pat.span, ct, typ);
        }
        write_ty(tcx, pat.id, typ);
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
        let ex_fields;
        alt structure_of(pcx.fcx, pat.span, expected) {
          ty::ty_rec(fields) { ex_fields = fields; }
          _ {
            tcx.sess.span_fatal
                (pat.span,
                #fmt["mismatched types: expected `%s` but found record",
                                ty_to_str(tcx, expected)]);
          }
        }
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
        write_ty(tcx, pat.id, expected);
      }
      ast::pat_tup(elts) {
        let ex_elts;
        alt structure_of(pcx.fcx, pat.span, expected) {
          ty::ty_tup(elts) { ex_elts = elts; }
          _ {
            tcx.sess.span_fatal
                (pat.span,
                 #fmt["mismatched types: expected `%s`, found tuple",
                        ty_to_str(tcx, expected)]);
          }
        }
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

        write_ty(tcx, pat.id, expected);
      }
      ast::pat_box(inner) {
        alt structure_of(pcx.fcx, pat.span, expected) {
          ty::ty_box(e_inner) {
            check_pat(pcx, inner, e_inner.ty);
            write_ty(tcx, pat.id, expected);
          }
          _ {
            tcx.sess.span_fatal(pat.span,
                                        "mismatched types: expected `" +
                                            ty_to_str(tcx, expected) +
                                            "` found box");
          }
        }
      }
      ast::pat_uniq(inner) {
        alt structure_of(pcx.fcx, pat.span, expected) {
          ty::ty_uniq(e_inner) {
            check_pat(pcx, inner, e_inner.ty);
            write_ty(tcx, pat.id, expected);
          }
          _ {
            tcx.sess.span_fatal(pat.span,
                                        "mismatched types: expected `" +
                                            ty_to_str(tcx, expected) +
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

fn lookup_method(fcx: @fn_ctxt, expr: @ast::expr, node_id: ast::node_id,
                 name: ast::ident, ty: ty::t, tps: [ty::t])
    -> option<method_origin> {
    alt lookup_method_inner(fcx, expr, name, ty) {
      some({method_ty: fty, n_tps: method_n_tps, substs, origin, self_sub}) {
        let tcx = fcx.ccx.tcx;
        let substs = substs, n_tps = vec::len(substs), n_tys = vec::len(tps);
        let has_self = ty::type_has_vars(fty);
        if method_n_tps + n_tps > 0u {
            if n_tys == 0u || n_tys != method_n_tps {
                if n_tys != 0u {
                    tcx.sess.span_err
                        (expr.span, "incorrect number of type \
                                     parameters given for this method");

                }
                substs += vec::from_fn(method_n_tps, {|_i|
                    ty::mk_var(tcx, next_ty_var_id(fcx))
                });
            } else {
                substs += tps;
            }
            write_ty_substs(tcx, node_id, fty, substs);
        } else {
            if n_tys > 0u {
                tcx.sess.span_err(expr.span, "this method does not take type \
                                              parameters");
            }
            write_ty(tcx, node_id, fty);
        }
        if has_self && !option::is_none(self_sub) {
            let fty = ty::node_id_to_type(tcx, node_id);
            fty = fixup_self_in_method_ty(
                tcx, fty, substs, option::get(self_sub));
            write_ty(tcx, node_id, fty);
        }
        if ty::type_has_rptrs(ty::ty_fn_ret(fty)) {
            let fty = ty::node_id_to_type(tcx, node_id);
            fty = fixup_self_region_in_method_ty(fcx, fty, expr);
            write_ty(tcx, node_id, fty);
        }
        some(origin)
      }
      none { none }
    }
}

fn lookup_method_inner(fcx: @fn_ctxt, expr: @ast::expr,
                       name: ast::ident, ty: ty::t)
    -> option<{method_ty: ty::t, n_tps: uint, substs: [ty::t],
                  origin: method_origin,
                  self_sub: option<self_subst>}> {
    let tcx = fcx.ccx.tcx;
    // First, see whether this is an interface-bounded parameter
    alt ty::get(ty).struct {
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
        let i = 0u;
        for m in *ty::iface_methods(tcx, did) {
            if m.ident == name {
                let fty = ty::mk_fn(tcx, {proto: ast::proto_box with m.fty});
                if ty::type_has_vars(fty) {
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

    let result = none, complained = false;
    std::list::iter(fcx.ccx.impl_map.get(expr.id)) {|impls|
        if option::is_some(result) { ret; }
        for @{did, methods, _} in *impls {
            alt vec::find(methods, {|m| m.ident == name}) {
              some(m) {
                let {n_tps, ty: self_ty} = impl_self_ty(tcx, did);
                let {vars, ty: self_ty} = if n_tps > 0u {
                    bind_params(fcx, self_ty, n_tps)
                } else {
                    {vars: [], ty: self_ty}
                };

                let ty = universally_quantify_regions(tcx, ty);

                alt unify::unify(fcx, self_ty, ty) {
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
    result
}

// problem -- class_item_ty should really be only used for internal stuff.
// or should have a privacy field.
fn lookup_field_ty(cx: ty::ctxt, items:[@ty::class_item_ty],
                   fieldname: ast::ident, sp: span)
    -> ty::t {
    for item in items {
            #debug("%s $$$ %s", fieldname, item.ident);
        alt item.contents {
          ty::var_ty(t) if item.ident == fieldname { ret t; }
          _ { }
        }
    }
    cx.sess.span_fatal(sp, #fmt("unbound field %s", fieldname));
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
            let expr_ty = ty::expr_ty(fcx.ccx.tcx, base);
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
                              unify: unifier,
                              expected: ty::t) {
    let tcx = fcx.ccx.tcx;
    let fty = ty::mk_fn(tcx,
                        ty_of_fn_decl(tcx, m_check_tyvar(fcx), proto, decl));

    #debug("check_expr_fn_with_unifier %s fty=%s",
           expr_to_str(expr),
           ty_to_str(tcx, fty));

    write_ty(tcx, expr.id, fty);

    // Unify the type of the function with the expected type before we
    // typecheck the body so that we have more information about the
    // argument types in the body. This is needed to make binops and
    // record projection work on type inferred arguments.
    unify(fcx, expr.span, expected, fty);

    check_fn(fcx.ccx, proto, decl, body, expr.id, some(fcx));
}

fn check_expr_with_unifier(fcx: @fn_ctxt, expr: @ast::expr, unify: unifier,
                           expected: ty::t) -> bool {
    #debug("typechecking expr %s",
           syntax::print::pprust::expr_to_str(expr));

    // A generic function to factor out common logic from call and bind
    // expressions.
    fn check_call_or_bind(fcx: @fn_ctxt, sp: span, id: ast::node_id,
                          fty: ty::t, args: [option<@ast::expr>]) -> bool {
        // Replaces "caller" regions in the arguments with the local region.
        fn instantiate_caller_regions(fcx: @fn_ctxt, id: ast::node_id,
                                      args: [ty::arg]) -> [ty::arg] {
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
                if ty::type_has_rptrs(arg.ty) {
                    let ty = ty::fold_ty(fcx.ccx.tcx, ty::fm_rptr({|r|
                        alt r {
                            ty::re_caller(_) {
                                // FIXME: We should not recurse into nested
                                // function types here.
                                region
                            }
                            _ { r }
                        }
                    }), arg.ty);
                    {ty: ty with arg}
                } else {
                    arg
                }
            };
        }

        let sty = structure_of(fcx, sp, fty);
        // Grab the argument types
        let arg_tys = alt sty {
          ty::ty_fn({inputs: arg_tys, _}) { arg_tys }
          _ {
            fcx.ccx.tcx.sess.span_fatal(sp, "mismatched types: \
                                             expected function or native \
                                             function but found "
                                        + ty_to_str(fcx.ccx.tcx, fty))
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
            // HACK: build an arguments list with dummy arguments to
            // check against
            let dummy = {mode: ast::expl(ast::by_ref),
                         ty: ty::mk_bot(fcx.ccx.tcx)};
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
                        bot |= check_expr_with_unifier(
                            fcx, a, demand::simple, arg_tys[i].ty);
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
        write_ty(fcx.ccx.tcx, id, ty::mk_nil(fcx.ccx.tcx));
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
        bot | check_call_or_bind(fcx, sp, id, expr_ty(fcx.ccx.tcx, f),
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
        let fty = ty::expr_ty(fcx.ccx.tcx, f);
        let rt_1 = alt structure_of(fcx, sp, fty) {
          ty::ty_fn(f) {
            bot |= f.ret_style == ast::noreturn;
            f.output
          }
          _ { fcx.ccx.tcx.sess.span_fatal(sp, "calling non-function"); }
        };
        write_ty(fcx.ccx.tcx, id, rt_1);
        ret bot;
    }

    // A generic function for checking for or for-each loops
    fn check_for(fcx: @fn_ctxt, local: @ast::local,
                 element_ty: ty::t, body: ast::blk,
                 node_id: ast::node_id) -> bool {
        let locid = lookup_local(fcx, local.span, local.node.id);
        let element_ty = demand::simple(fcx, local.span, element_ty,
                                        ty::mk_var(fcx.ccx.tcx, locid));
        let bot = check_decl_local(fcx, local);
        check_block_no_value(fcx, body);
        // Unify type of decl with element type of the seq
        demand::simple(fcx, local.span,
                       ty::node_id_to_type(fcx.ccx.tcx, local.node.id),
                       element_ty);
        write_nil(fcx.ccx.tcx, node_id);
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
                let thn_bot = check_block(fcx, thn);
                let thn_t = block_ty(fcx.ccx.tcx, thn);
                let els_bot = check_expr_with(fcx, els, thn_t);
                let els_t = expr_ty(fcx.ccx.tcx, els);
                let if_t = if !ty::type_is_bot(els_t) {
                    els_t
                } else {
                    thn_t
                };
                (if_t, thn_bot & els_bot)
              }
              none {
                check_block_no_value(fcx, thn);
                (ty::mk_nil(fcx.ccx.tcx), false)
              }
            };
        write_ty(fcx.ccx.tcx, id, if_t);
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
        -> option<ty::t> {
        let callee_id = ast_util::op_expr_callee_id(op_ex);
        alt lookup_method(fcx, op_ex, callee_id, opname, self_t, []) {
          some(origin) {
            let method_ty = ty::node_id_to_type(fcx.ccx.tcx, callee_id);
            check_call_or_bind(fcx, op_ex.span, op_ex.id, method_ty, args);
            fcx.ccx.method_map.insert(op_ex.id, origin);
            some(ty::ty_fn_ret(method_ty))
          }
          _ { none }
        }
    }
    fn check_binop(fcx: @fn_ctxt, ex: @ast::expr, ty: ty::t,
                   op: ast::binop, rhs: @ast::expr) -> ty::t {
        let resolved_t = structurally_resolved_type(fcx, ex.span, ty);
        let tcx = fcx.ccx.tcx;
        if ty::is_binopable(tcx, resolved_t, op) {
            ret alt op {
              ast::eq | ast::lt | ast::le | ast::ne | ast::ge |
              ast::gt { ty::mk_bool(tcx) }
              _ { resolved_t }
            };
        }

        alt binop_method(op) {
          some(name) {
            alt lookup_op_method(fcx, ex, resolved_t, name, [some(rhs)]) {
              some(ret_ty) { ret ret_ty; }
              _ {}
            }
          }
          _ {}
        }
        tcx.sess.span_err(
            ex.span, "binary operation " + ast_util::binop_to_str(op) +
            " cannot be applied to type `" + ty_to_str(tcx, resolved_t) +
            "`");
        resolved_t
    }
    fn check_user_unop(fcx: @fn_ctxt, op_str: str, mname: str,
                       ex: @ast::expr, rhs_t: ty::t) -> ty::t {
        alt lookup_op_method(fcx, ex, rhs_t, mname, []) {
          some(ret_ty) { ret_ty }
          _ {
            fcx.ccx.tcx.sess.span_err(
                ex.span, #fmt["cannot apply unary operator `%s` to type `%s`",
                              op_str, ty_to_str(fcx.ccx.tcx, rhs_t)]);
            rhs_t
          }
        }
    }

    let tcx = fcx.ccx.tcx;
    let id = expr.id;
    let bot = false;
    alt expr.node {
      ast::expr_lit(lit) {
        let typ = check_lit(fcx.ccx, lit);
        write_ty(tcx, id, typ);
      }
      ast::expr_binary(binop, lhs, rhs) {
        let lhs_t = next_ty_var(fcx);
        bot = check_expr_with(fcx, lhs, lhs_t);

        let rhs_bot = if !ast_util::is_shift_binop(binop) {
            check_expr_with(fcx, rhs, lhs_t)
        } else {
            let rhs_bot = check_expr(fcx, rhs);
            let rhs_t = expr_ty(tcx, rhs);
            require_integral(fcx, rhs.span, rhs_t);
            rhs_bot
        };

        if !ast_util::lazy_binop(binop) { bot |= rhs_bot; }

        let result = check_binop(fcx, expr, lhs_t, binop, rhs);
        write_ty(tcx, id, result);
      }
      ast::expr_assign_op(op, lhs, rhs) {
        require_impure(tcx.sess, fcx.purity, expr.span);
        bot = check_assignment(fcx, expr.span, lhs, rhs, id);
        let lhs_t = ty::expr_ty(tcx, lhs);
        let result = check_binop(fcx, expr, lhs_t, op, rhs);
        demand::simple(fcx, expr.span, result, lhs_t);
      }
      ast::expr_unary(unop, oper) {
        bot = check_expr(fcx, oper);
        let oper_t = expr_ty(tcx, oper);
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
                    tcx.sess.span_err(expr.span,
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
        write_ty(tcx, id, oper_t);
      }
      ast::expr_addr_of(mutbl, oper) {
        bot = check_expr(fcx, oper);
        let oper_t = expr_ty(tcx, oper);

        let region = region_of(fcx, oper);
        let tm = { ty: oper_t, mutbl: mutbl };
        oper_t = ty::mk_rptr(tcx, region, tm);
        write_ty(tcx, id, oper_t);
      }
      ast::expr_path(pth) {
        let defn = lookup_def(fcx, pth.span, id);

        let tpt = ty_param_bounds_and_ty_for_def(fcx, expr.span, defn);
        if ty::def_has_ty_params(defn) {
            instantiate_path(fcx, pth, tpt, expr.span, expr.id);
        } else {
            // The definition doesn't take type parameters. If the programmer
            // supplied some, that's an error
            if vec::len::<@ast::ty>(pth.node.types) > 0u {
                tcx.sess.span_fatal(expr.span,
                                    "this kind of value does not \
                                     take type parameters");
            }
            write_ty(tcx, id, tpt.ty);
        }
      }
      ast::expr_mac(_) { tcx.sess.bug("unexpanded macro"); }
      ast::expr_fail(expr_opt) {
        bot = true;
        alt expr_opt {
          none {/* do nothing */ }
          some(e) { check_expr_with(fcx, e, ty::mk_str(tcx)); }
        }
        write_bot(tcx, id);
      }
      ast::expr_break { write_bot(tcx, id); bot = true; }
      ast::expr_cont { write_bot(tcx, id); bot = true; }
      ast::expr_ret(expr_opt) {
        bot = true;
        alt expr_opt {
          none {
            let nil = ty::mk_nil(tcx);
            if !are_compatible(fcx, fcx.ret_ty, nil) {
                tcx.sess.span_err(expr.span,
                                  "ret; in function returning non-nil");
            }
          }
          some(e) { check_expr_with(fcx, e, fcx.ret_ty); }
        }
        write_bot(tcx, id);
      }
      ast::expr_be(e) {
        // FIXME: prove instead of assert
        assert (ast_util::is_call_expr(e));
        check_expr_with(fcx, e, fcx.ret_ty);
        bot = true;
        write_nil(tcx, id);
      }
      ast::expr_log(_, lv, e) {
        bot = check_expr_with(fcx, lv, ty::mk_mach_uint(tcx, ast::ty_u32));
        bot |= check_expr(fcx, e);
        write_nil(tcx, id);
      }
      ast::expr_check(_, e) {
        bot = check_pred_expr(fcx, e);
        write_nil(tcx, id);
      }
      ast::expr_if_check(cond, thn, elsopt) {
        bot =
            check_pred_expr(fcx, cond) |
                check_then_else(fcx, thn, elsopt, id, expr.span);
      }
      ast::expr_assert(e) {
        bot = check_expr_with(fcx, e, ty::mk_bool(tcx));
        write_nil(tcx, id);
      }
      ast::expr_copy(a) {
        bot = check_expr_with_unifier(fcx, a, unify, expected);
        write_ty(tcx, id, ty::node_id_to_type(tcx, a.id));
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
      ast::expr_for(decl, seq, body) {
        bot = check_expr(fcx, seq);
        let elt_ty;
        let ety = expr_ty(tcx, seq);
        alt structure_of(fcx, expr.span, ety) {
          ty::ty_vec(vec_elt_ty) { elt_ty = vec_elt_ty.ty; }
          ty::ty_str { elt_ty = ty::mk_mach_uint(tcx, ast::ty_u8); }
          _ {
            tcx.sess.span_fatal(expr.span,
                                "mismatched types: expected vector or string "
                                + "but found `" + ty_to_str(tcx, ety) + "`");
          }
        }
        bot |= check_for(fcx, decl, elt_ty, body, id);
      }
      ast::expr_while(cond, body) {
        bot = check_expr_with(fcx, cond, ty::mk_bool(tcx));
        check_block_no_value(fcx, body);
        write_ty(tcx, id, ty::mk_nil(tcx));
      }
      ast::expr_do_while(body, cond) {
        bot = check_expr_with(fcx, cond, ty::mk_bool(tcx)) |
              check_block_no_value(fcx, body);
        write_ty(tcx, id, block_ty(tcx, body));
      }
      ast::expr_loop(body) {
          check_block_no_value(fcx, body);
          write_ty(tcx, id, ty::mk_nil(tcx));
          bot = !may_break(body);
      }
      ast::expr_alt(discrim, arms, _) {
        bot = check_expr(fcx, discrim);

        let parent_block = tcx.region_map.rvalue_to_block.get(discrim.id);

        // Typecheck the patterns first, so that we get types for all the
        // bindings.
        let pattern_ty = ty::expr_ty(tcx, discrim);
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
              some(e) { check_expr_with(fcx, e, ty::mk_bool(tcx)); }
              none { }
            }
            if !check_block(fcx, arm.body) { arm_non_bot = true; }
            let bty = block_ty(tcx, arm.body);
            result_ty = demand::simple(fcx, arm.body.span, result_ty, bty);
        }
        bot |= !arm_non_bot;
        if !arm_non_bot { result_ty = ty::mk_bot(tcx); }
        write_ty(tcx, id, result_ty);
      }
      ast::expr_fn(proto, decl, body, captures) {
        check_expr_fn_with_unifier(fcx, expr, proto, decl, body,
                          unify, expected);
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
                                   unify, expected);
      }
      ast::expr_block(b) {
        // If this is an unchecked block, turn off purity-checking
        bot = check_block(fcx, b);
        let typ =
            alt b.node.expr {
              some(expr) { expr_ty(tcx, expr) }
              none { ty::mk_nil(tcx) }
            };
        write_ty(tcx, id, typ);
      }
      ast::expr_bind(f, args) {
        // Call the generic checker.
        bot = check_expr(fcx, f);
        bot |= check_call_or_bind(fcx, expr.span, expr.id, expr_ty(tcx, f),
                                  args);

        // Pull the argument and return types out.
        let proto, arg_tys, rt, cf, constrs;
        alt structure_of(fcx, expr.span, expr_ty(tcx, f)) {
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
        let out_args = [];
        let i = 0u;
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
        write_ty(tcx, id, ft);
      }
      ast::expr_call(f, args, _) {
        bot = check_call_full(fcx, expr.span, expr.id, f, args);
      }
      ast::expr_cast(e, t) {
        bot = check_expr(fcx, e);
        let t_1 = ast_ty_to_ty_crate(fcx.ccx, t);
        let t_e = ty::expr_ty(tcx, e);

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
        write_ty(tcx, id, t_1);
      }
      ast::expr_vec(args, mutbl) {
        let t: ty::t = next_ty_var(fcx);
        for e: @ast::expr in args { bot |= check_expr_with(fcx, e, t); }
        let typ = ty::mk_vec(tcx, {ty: t, mutbl: mutbl});
        write_ty(tcx, id, typ);
      }
      ast::expr_tup(elts) {
        let elt_ts = [];
        vec::reserve(elt_ts, vec::len(elts));
        for e in elts {
            check_expr(fcx, e);
            let ety = expr_ty(tcx, e);
            elt_ts += [ety];
        }
        let typ = ty::mk_tup(tcx, elt_ts);
        write_ty(tcx, id, typ);
      }
      ast::expr_rec(fields, base) {
        alt base { none {/* no-op */ } some(b_0) { check_expr(fcx, b_0); } }
        let fields_t: [spanned<field>] = [];
        for f: ast::field in fields {
            bot |= check_expr(fcx, f.node.expr);
            let expr_t = expr_ty(tcx, f.node.expr);
            let expr_mt = {ty: expr_t, mutbl: f.node.mutbl};
            // for the most precise error message,
            // should be f.node.expr.span, not f.span
            fields_t +=
                [respan(f.node.expr.span,
                        {ident: f.node.ident, mt: expr_mt})];
        }
        alt base {
          none {
            fn get_node(f: spanned<field>) -> field { f.node }
            let typ = ty::mk_rec(tcx, vec::map(fields_t, get_node));
            write_ty(tcx, id, typ);
          }
          some(bexpr) {
            bot |= check_expr(fcx, bexpr);
            let bexpr_t = expr_ty(tcx, bexpr);
            let base_fields: [field] = [];
            alt structure_of(fcx, expr.span, bexpr_t) {
              ty::ty_rec(flds) { base_fields = flds; }
              _ {
                tcx.sess.span_fatal(expr.span,
                                    "record update has non-record base");
              }
            }
            write_ty(tcx, id, bexpr_t);
            for f: spanned<ty::field> in fields_t {
                let found = false;
                for bf: ty::field in base_fields {
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
                                                expr_ty(tcx, base));
        let base_t = do_autoderef(fcx, expr.span, expr_t);
        let handled = false, n_tys = vec::len(tys);
        alt structure_of(fcx, expr.span, base_t) {
          ty::ty_rec(fields) {
            alt ty::field_idx(field, fields) {
              some(ix) {
                if n_tys > 0u {
                    tcx.sess.span_err(expr.span,
                                      "can't provide type parameters \
                                       to a field access");
                }
                write_ty(tcx, id, fields[ix].mt.ty);
                handled = true;
              }
              _ {}
            }
          }
          ty::ty_class(base_id, _params) {
              // (1) verify that the class id actually has a field called
              // field
              let cls_items = lookup_class_item_tys(tcx, base_id);
              let field_ty = lookup_field_ty(fcx.ccx.tcx, cls_items, field,
                                             expr.span);
              // (2) look up what field's type is, and return it
              // FIXME: actually instantiate any type params
              write_ty(tcx, id, field_ty);
              handled = true;
          }
          _ {}
        }
        if !handled {
            let tps = vec::map(tys, {|ty| ast_ty_to_ty_crate(fcx.ccx, ty)});
            alt lookup_method(fcx, expr, expr.id, field, expr_t, tps) {
              some(origin) {
                fcx.ccx.method_map.insert(id, origin);
              }
              none {
                let t_err = resolve_type_vars_if_possible(fcx, expr_t);
                let msg = #fmt["attempted access of field %s on type %s, but \
                                no field or method with that name was found",
                               field, ty_to_str(tcx, t_err)];
                tcx.sess.span_err(expr.span, msg);
                // NB: Adding a bogus type to allow typechecking to continue
                write_ty(tcx, id, next_ty_var(fcx));
              }
            }
        }
      }
      ast::expr_index(base, idx) {
        bot |= check_expr(fcx, base);
        let raw_base_t = expr_ty(tcx, base);
        let base_t = do_autoderef(fcx, expr.span, raw_base_t);
        bot |= check_expr(fcx, idx);
        let idx_t = expr_ty(tcx, idx);
        alt structure_of(fcx, expr.span, base_t) {
          ty::ty_vec(mt) {
            require_integral(fcx, idx.span, idx_t);
            write_ty(tcx, id, mt.ty);
          }
          ty::ty_str {
            require_integral(fcx, idx.span, idx_t);
            let typ = ty::mk_mach_uint(tcx, ast::ty_u8);
            write_ty(tcx, id, typ);
          }
          _ {
            let resolved = structurally_resolved_type(fcx, expr.span,
                                                      raw_base_t);
            alt lookup_op_method(fcx, expr, resolved, "[]",
                                 [some(idx)]) {
              some(ret_ty) { write_ty(tcx, id, ret_ty); }
              _ {
                tcx.sess.span_fatal(
                    expr.span, "cannot index a value of type `" +
                    ty_to_str(tcx, base_t) + "`");
              }
            }
          }
        }
      }
    }
    if bot { write_ty(tcx, expr.id, ty::mk_bot(tcx)); }

    unify(fcx, expr.span, expected, expr_ty(tcx, expr));
    ret bot;
}

fn require_integral(fcx: @fn_ctxt, sp: span, t: ty::t) {
    if !type_is_integral(fcx, sp, t) {
        fcx.ccx.tcx.sess.span_err(sp, "mismatched types: expected \
                                       `integer` but found `"
                                  + ty_to_str(fcx.ccx.tcx, t) + "`");
    }
}

fn next_ty_var_id(fcx: @fn_ctxt) -> int {
    let id = *fcx.next_var_id;
    *fcx.next_var_id += 1;
    ret id;
}

fn next_ty_var(fcx: @fn_ctxt) -> ty::t {
    ret ty::mk_var(fcx.ccx.tcx, next_ty_var_id(fcx));
}

fn bind_params(fcx: @fn_ctxt, tp: ty::t, count: uint)
    -> {vars: [ty::t], ty: ty::t} {
    let vars = vec::from_fn(count, {|_i| next_ty_var(fcx)});
    {vars: vars, ty: ty::substitute_type_params(fcx.ccx.tcx, vars, tp)}
}

fn get_self_info(ccx: @crate_ctxt) -> option<self_info> {
    ret vec::last_opt(ccx.self_infos);
}

fn check_decl_initializer(fcx: @fn_ctxt, nid: ast::node_id,
                          init: ast::initializer) -> bool {
    let lty = ty::mk_var(fcx.ccx.tcx, lookup_local(fcx, init.expr.span, nid));
    ret check_expr_with(fcx, init.expr, lty);
}

fn check_decl_local(fcx: @fn_ctxt, local: @ast::local) -> bool {
    let bot = false;

    let t = ty::mk_var(fcx.ccx.tcx, fcx.locals.get(local.node.id));
    write_ty(fcx.ccx.tcx, local.node.id, t);
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
        bot = check_expr_with(fcx, expr, ty::mk_nil(fcx.ccx.tcx));
      }
      ast::stmt_semi(expr, id) {
        node_id = id;
        bot = check_expr(fcx, expr);
      }
    }
    write_nil(fcx.ccx.tcx, node_id);
    ret bot;
}

fn check_block_no_value(fcx: @fn_ctxt, blk: ast::blk) -> bool {
    let bot = check_block(fcx, blk);
    if !bot {
        let blkty = ty::node_id_to_type(fcx.ccx.tcx, blk.node.id);
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
      none { write_nil(fcx.ccx.tcx, blk.node.id); }
      some(e) {
        if bot && !warned {
            fcx.ccx.tcx.sess.span_warn(e.span, "unreachable expression");
        }
        bot |= check_expr(fcx, e);
        let ety = expr_ty(fcx.ccx.tcx, e);
        write_ty(fcx.ccx.tcx, blk.node.id, ety);
      }
    }
    if bot {
        write_ty(fcx.ccx.tcx, blk.node.id, ty::mk_bot(fcx.ccx.tcx));
    }
    ret bot;
}

fn check_const(ccx: @crate_ctxt, _sp: span, e: @ast::expr, id: ast::node_id) {
    // FIXME: this is kinda a kludge; we manufacture a fake function context
    // and statement context for checking the initializer expression.
    let rty = node_id_to_type(ccx.tcx, id);
    let fcx: @fn_ctxt =
        @{ret_ty: rty,
          purity: ast::pure_fn,
          proto: ast::proto_box,
          var_bindings: ty::unify::mk_var_bindings(),
          locals: int_hash::<int>(),
          next_var_id: @mutable 0,
          ccx: ccx};
    check_expr(fcx, e);
    let cty = expr_ty(fcx.ccx.tcx, e);
    let declty = fcx.ccx.tcx.tcache.get(local_def(id)).ty;
    demand::simple(fcx, e.span, declty, cty);
}

fn check_enum_variants(ccx: @crate_ctxt, sp: span, vs: [ast::variant],
                      id: ast::node_id) {
    // FIXME: this is kinda a kludge; we manufacture a fake function context
    // and statement context for checking the initializer expression.
    let rty = node_id_to_type(ccx.tcx, id);
    let fcx: @fn_ctxt =
        @{ret_ty: rty,
          purity: ast::pure_fn,
          proto: ast::proto_box,
          var_bindings: ty::unify::mk_var_bindings(),
          locals: int_hash::<int>(),
          next_var_id: @mutable 0,
          ccx: ccx};
    let disr_vals: [int] = [];
    let disr_val = 0;
    for v in vs {
        alt v.node.disr_expr {
          some(e) {
            check_expr(fcx, e);
            let cty = expr_ty(ccx.tcx, e);
            let declty = ty::mk_int(ccx.tcx);
            demand::simple(fcx, e.span, declty, cty);
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
    let bot = check_expr_with(fcx, e, ty::mk_bool(fcx.ccx.tcx));

    /* e must be a call expr where all arguments are either
    literals or slots */
    alt e.node {
      ast::expr_call(operator, operands, _) {
        if !ty::is_pred_ty(expr_ty(fcx.ccx.tcx, operator)) {
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

fn check_fn(ccx: @crate_ctxt,
            proto: ast::proto,
            decl: ast::fn_decl,
            body: ast::blk,
            id: ast::node_id,
            old_fcx: option<@fn_ctxt>) {
    // If old_fcx is some(...), this is a block fn { |x| ... }.
    // In that case, the purity is inherited from the context.
    let purity = alt old_fcx {
      none { decl.purity }
      some(f) { assert decl.purity == ast::impure_fn; f.purity }
    };

    let gather_result = gather_locals(ccx, decl, body, id, old_fcx);
    let fcx: @fn_ctxt =
        @{ret_ty: ty::ty_fn_ret(ty::node_id_to_type(ccx.tcx, id)),
          purity: purity,
          proto: proto,
          var_bindings: gather_result.var_bindings,
          locals: gather_result.locals,
          next_var_id: gather_result.next_var_id,
          ccx: ccx};

    check_constraints(fcx, decl.constraints, decl.inputs);
    check_block(fcx, body);

    // We unify the tail expr's type with the
    // function result type, if there is a tail expr.
    alt body.node.expr {
      some(tail_expr) {
        let tail_expr_ty = expr_ty(ccx.tcx, tail_expr);
        demand::simple(fcx, tail_expr.span, fcx.ret_ty, tail_expr_ty);
      }
      none { }
    }

    let args = ty::ty_fn_args(ty::node_id_to_type(ccx.tcx, id));
    let i = 0u;
    for arg: ty::arg in args {
        write_ty(ccx.tcx, decl.inputs[i].id, arg.ty);
        i += 1u;
    }

    // If we don't have any enclosing function scope, it is time to
    // force any remaining type vars to be resolved.
    // If we have an enclosing function scope, our type variables will be
    // resolved when the enclosing scope finishes up.
    if option::is_none(old_fcx) {
        vtable::resolve_in_block(fcx, body);
        writeback::resolve_type_vars_in_block(fcx, body);
    }
}

fn check_method(ccx: @crate_ctxt, method: @ast::method) {
    check_fn(ccx, ast::proto_bare, method.decl, method.body, method.id, none);
}

fn class_types(ccx: @crate_ctxt, members: [@ast::class_item]) -> class_map {
    let rslt = int_hash::<ty::t>();
    for m in members {
      alt m.node.decl {
         ast::instance_var(_,t,_,id) {
           rslt.insert(id, ast_ty_to_ty(ccx.tcx, m_collect, t));
         }
         ast::class_method(it) {
             rslt.insert(it.id, ty_of_item(ccx.tcx, m_collect, it).ty);
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
        check_fn(ccx, ast::proto_bare, decl, body, it.id, none);
      }
      ast::item_res(decl, tps, body, dtor_id, _) {
        check_fn(ccx, ast::proto_bare, decl, body, dtor_id, none);
      }
      ast::item_impl(tps, _, ty, ms) {
        let self_ty = ast_ty_to_ty(ccx.tcx, m_check, ty);
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
          check_fn(class_ccx, ast::proto_bare, ctor.node.dec,
                   ctor.node.body, ctor.node.id, none);
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
                      bounds: @[ty::param_bounds], tys: [ty::t],
                      allow_unsafe: bool) -> vtable_res {
        let tcx = fcx.ccx.tcx, result = [], i = 0u;
        for ty in tys {
            for bound in *bounds[i] {
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
                        let {n_tps, ty: self_ty} = impl_self_ty(tcx, im.did);
                        let {vars, ty: self_ty} = if n_tps > 0u {
                            bind_params(fcx, self_ty, n_tps)
                        } else { {vars: [], ty: self_ty} };
                        let im_bs = ty::lookup_item_type(tcx, im.did).bounds;
                        alt unify::unify(fcx, ty, self_ty) {
                          result::ok(_) {
                            if option::is_some(found) {
                                tcx.sess.span_err(
                                    sp, "multiple applicable implementations \
                                         in scope");
                            } else {
                                connect_iface_tps(fcx, sp, vars, iface_tps,
                                                  im.did);
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
        alt ty::unify::fixup_vars(tcx, some(sp), fcx.var_bindings, ty) {
          result::ok(new_type) { new_type }
          result::err(vid) {
            tcx.sess.span_fatal(sp, "could not determine a type for a \
                                     bounded type parameter");
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
            alt cx.tcx.node_type_substs.find(ex.id) {
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
                    let ts = ty::node_id_to_type_params(cx.tcx, callee_id);
                    let iscs = cx.impl_map.get(ex.id);
                    cx.vtable_map.insert(callee_id, lookup_vtables(
                        fcx, iscs, ex.span, bounds, ts, false));
                }
              }
              _ {}
            }
          }
          ast::expr_cast(src, _) {
            let target_ty = expr_ty(cx.tcx, ex);
            alt ty::get(target_ty).struct {
              ty::ty_iface(_, _) {
                let impls = cx.impl_map.get(ex.id);
                let vtable = lookup_vtable(fcx, impls, ex.span,
                                           expr_ty(cx.tcx, src), target_ty,
                                           true);
                cx.vtable_map.insert(ex.id, @[vtable]);
              }
              _ {}
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
    collect::collect_item_types(tcx, crate);

    let ccx = @{mutable self_infos: [],
                impl_map: impl_map,
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
