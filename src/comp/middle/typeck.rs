import front.ast;
import front.ast.ann;
import front.ast.ann_none;
import front.ast.mutability;
import front.creader;
import middle.fold;
import driver.session;
import util.common;
import util.common.span;
import util.common.plain_ann;
import util.common.new_def_hash;
import util.common.log_expr_err;

import middle.ty;
import middle.ty.ann_to_type;
import middle.ty.arg;
import middle.ty.bind_params_in_type;
import middle.ty.block_ty;
import middle.ty.expr_ty;
import middle.ty.field;
import middle.ty.method;
import middle.ty.mo_val;
import middle.ty.mo_alias;
import middle.ty.mo_either;
import middle.ty.pat_ty;
import middle.ty.path_to_str;
import middle.ty.struct;
import middle.ty.triv_ann;
import middle.ty.ty_to_str;
import middle.ty.type_is_integral;
import middle.ty.type_is_scalar;
import middle.ty.ty_param_count_and_ty;
import middle.ty.ty_nil;
import middle.ty.Unify.ures_ok;
import middle.ty.Unify.ures_err;

import std.Str;
import std.UInt;
import std.Vec;
import std.Map;
import std.Map.hashmap;
import std.Option;
import std.Option.none;
import std.Option.some;
import std.Option.from_maybe;

import util.typestate_ann.ts_ann;

type ty_table = hashmap[ast.def_id, ty.t];

tag any_item {
    any_item_rust(@ast.item);
    any_item_native(@ast.native_item, ast.native_abi);
}

type ty_item_table = hashmap[ast.def_id,any_item];
type fn_purity_table = hashmap[ast.def_id, ast.purity];

type unify_cache_entry = tup(ty.t,ty.t,vec[mutable ty.t]);
type unify_cache = hashmap[unify_cache_entry,ty.Unify.result];

type crate_ctxt = rec(session.session sess,
                      ty.type_cache type_cache,
                      @ty_item_table item_items,
                      vec[ast.obj_field] obj_fields,
                      Option.t[ast.def_id] this_obj,
                      @fn_purity_table fn_purity_table,
                      mutable int next_var_id,
                      unify_cache unify_cache,
                      mutable uint cache_hits,
                      mutable uint cache_misses,
                      ty.ctxt tcx);

type fn_ctxt = rec(ty.t ret_ty,
                   ast.purity purity,
                   @ty_table locals,
                   @crate_ctxt ccx);

// Used for ast_ty_to_ty() below.
type ty_getter = fn(ast.def_id) -> ty.ty_param_count_and_ty;

// Substitutes the user's explicit types for the parameters in a path
// expression.
fn substitute_ty_params(&@crate_ctxt ccx,
                        ty.t typ,
                        uint ty_param_count,
                        vec[ty.t] supplied,
                        &span sp) -> ty.t {
    fn substituter(@crate_ctxt ccx, vec[ty.t] supplied, ty.t typ) -> ty.t {
        alt (struct(ccx.tcx, typ)) {
            case (ty.ty_bound_param(?pid)) { ret supplied.(pid); }
            case (_) { ret typ; }
        }
    }

    auto supplied_len = Vec.len[ty.t](supplied);
    if (ty_param_count != supplied_len) {
        ccx.sess.span_err(sp, "expected " +
                          UInt.to_str(ty_param_count, 10u) +
                          " type parameter(s) but found " +
                          UInt.to_str(supplied_len, 10u) + " parameter(s)");
        fail;
    }

    if (!ty.type_contains_bound_params(ccx.tcx, typ)) {
        ret typ;
    }

    auto f = bind substituter(ccx, supplied, _);
    ret ty.fold_ty(ccx.tcx, f, typ);
}


// Returns the type parameter count and the type for the given definition.
fn ty_param_count_and_ty_for_def(@fn_ctxt fcx, &ast.span sp, &ast.def defn)
        -> ty_param_count_and_ty {
    alt (defn) {
        case (ast.def_arg(?id)) {
            // assert (fcx.locals.contains_key(id));
            ret tup(0u, fcx.locals.get(id));
        }
        case (ast.def_local(?id)) {
            auto t;
            alt (fcx.locals.find(id)) {
                case (some[ty.t](?t1)) { t = t1; }
                case (none[ty.t]) { t = ty.mk_local(fcx.ccx.tcx, id); }
            }
            ret tup(0u, t);
        }
        case (ast.def_obj_field(?id)) {
            // assert (fcx.locals.contains_key(id));
            ret tup(0u, fcx.locals.get(id));
        }
        case (ast.def_fn(?id)) {
            ret ty.lookup_item_type(fcx.ccx.sess, fcx.ccx.tcx,
                                    fcx.ccx.type_cache, id);
        }
        case (ast.def_native_fn(?id)) {
            ret ty.lookup_item_type(fcx.ccx.sess, fcx.ccx.tcx,
                                    fcx.ccx.type_cache, id);
        }
        case (ast.def_const(?id)) {
            ret ty.lookup_item_type(fcx.ccx.sess, fcx.ccx.tcx,
                                    fcx.ccx.type_cache, id);
        }
        case (ast.def_variant(_, ?vid)) {
            ret ty.lookup_item_type(fcx.ccx.sess, fcx.ccx.tcx,
                                    fcx.ccx.type_cache, vid);
        }
        case (ast.def_binding(?id)) {
            // assert (fcx.locals.contains_key(id));
            ret tup(0u, fcx.locals.get(id));
        }
        case (ast.def_obj(?id)) {
            ret ty.lookup_item_type(fcx.ccx.sess, fcx.ccx.tcx,
                                    fcx.ccx.type_cache, id);
        }

        case (ast.def_mod(_)) {
            // Hopefully part of a path.
            // TODO: return a type that's more poisonous, perhaps?
            ret tup(0u, ty.mk_nil(fcx.ccx.tcx));
        }

        case (ast.def_ty(_)) {
            fcx.ccx.sess.span_err(sp, "expected value but found type");
            fail;
        }

        case (_) {
            // FIXME: handle other names.
            fcx.ccx.sess.unimpl("definition variant");
            fail;
        }
    }
}

// Instantiates the given path, which must refer to an item with the given
// number of type parameters and type.
fn instantiate_path(@fn_ctxt fcx, &ast.path pth, &ty_param_count_and_ty tpt,
        &span sp) -> ast.ann {
    auto ty_param_count = tpt._0;
    auto t = bind_params_in_type(fcx.ccx.tcx, tpt._1);

    auto ty_substs_opt;
    auto ty_substs_len = Vec.len[@ast.ty](pth.node.types);
    if (ty_substs_len > 0u) {
        let vec[ty.t] ty_substs = vec();
        auto i = 0u;
        while (i < ty_substs_len) {
            ty_substs += vec(ast_ty_to_ty_crate(fcx.ccx, pth.node.types.(i)));
            i += 1u;
        }
        ty_substs_opt = some[vec[ty.t]](ty_substs);

        if (ty_param_count == 0u) {
            fcx.ccx.sess.span_err(sp, "this item does not take type " +
                                  "parameters");
            fail;
        }
    } else {
        // We will acquire the type parameters through unification.
        let vec[ty.t] ty_substs = vec();
        auto i = 0u;
        while (i < ty_param_count) {
            ty_substs += vec(next_ty_var(fcx.ccx));
            i += 1u;
        }
        ty_substs_opt = some[vec[ty.t]](ty_substs);
    }

    ret ast.ann_type(t, ty_substs_opt, none[@ts_ann]);
}

fn ast_mode_to_mode(ast.mode mode) -> ty.mode {
    auto ty_mode;
    alt (mode) {
        case (ast.val) { ty_mode = mo_val; }
        case (ast.alias) { ty_mode = mo_alias; }
    }
    ret ty_mode;
}

// Parses the programmer's textual representation of a type into our internal
// notion of a type. `getter` is a function that returns the type
// corresponding to a definition ID.
fn ast_ty_to_ty(ty.ctxt tcx, ty_getter getter, &@ast.ty ast_ty) -> ty.t {
    fn ast_arg_to_arg(ty.ctxt tcx,
                      ty_getter getter,
                      &rec(ast.mode mode, @ast.ty ty) arg)
            -> rec(ty.mode mode, ty.t ty) {
        auto ty_mode = ast_mode_to_mode(arg.mode);
        ret rec(mode=ty_mode, ty=ast_ty_to_ty(tcx, getter, arg.ty));
    }

    fn ast_mt_to_mt(ty.ctxt tcx,
                    ty_getter getter,
                    &ast.mt mt) -> ty.mt {
        ret rec(ty=ast_ty_to_ty(tcx, getter, mt.ty), mut=mt.mut);
    }

    fn instantiate(ty.ctxt tcx,
                   ty_getter getter,
                   ast.def_id id,
                   vec[@ast.ty] args) -> ty.t {
        // TODO: maybe record cname chains so we can do
        // "foo = int" like OCaml?
        auto params_opt_and_ty = getter(id);

        if (params_opt_and_ty._0 == 0u) {
            ret params_opt_and_ty._1;
        }

        // The typedef is type-parametric. Do the type substitution.
        //
        // TODO: Make sure the number of supplied bindings matches the number
        // of type parameters in the typedef. Emit a friendly error otherwise.
        auto bound_ty = bind_params_in_type(tcx, params_opt_and_ty._1);
        let vec[ty.t] param_bindings = vec();
        for (@ast.ty ast_ty in args) {
            param_bindings += vec(ast_ty_to_ty(tcx, getter, ast_ty));
        }
        ret ty.substitute_type_params(tcx, param_bindings, bound_ty);
    }

    auto mut = ast.imm;
    auto typ;
    auto cname = none[str];
    alt (ast_ty.node) {
        case (ast.ty_nil)          { typ = ty.mk_nil(tcx); }
        case (ast.ty_bool)         { typ = ty.mk_bool(tcx); }
        case (ast.ty_int)          { typ = ty.mk_int(tcx); }
        case (ast.ty_uint)         { typ = ty.mk_uint(tcx); }
        case (ast.ty_float)        { typ = ty.mk_float(tcx); }
        case (ast.ty_machine(?tm)) { typ = ty.mk_mach(tcx, tm); }
        case (ast.ty_char)         { typ = ty.mk_char(tcx); }
        case (ast.ty_str)          { typ = ty.mk_str(tcx); }
        case (ast.ty_box(?mt)) {
            typ = ty.mk_box(tcx, ast_mt_to_mt(tcx, getter, mt));
        }
        case (ast.ty_vec(?mt)) {
            typ = ty.mk_vec(tcx, ast_mt_to_mt(tcx, getter, mt));
        }

        case (ast.ty_port(?t)) {
            typ = ty.mk_port(tcx, ast_ty_to_ty(tcx, getter, t));
        }

        case (ast.ty_chan(?t)) {
            typ = ty.mk_chan(tcx, ast_ty_to_ty(tcx, getter, t));
        }

        case (ast.ty_tup(?fields)) {
            let vec[ty.mt] flds = vec();
            for (ast.mt field in fields) {
                Vec.push[ty.mt](flds, ast_mt_to_mt(tcx, getter, field));
            }
            typ = ty.mk_tup(tcx, flds);
        }
        case (ast.ty_rec(?fields)) {
            let vec[field] flds = vec();
            for (ast.ty_field f in fields) {
                auto tm = ast_mt_to_mt(tcx, getter, f.mt);
                Vec.push[field](flds, rec(ident=f.ident, mt=tm));
            }
            typ = ty.mk_rec(tcx, flds);
        }

        case (ast.ty_fn(?proto, ?inputs, ?output)) {
            auto f = bind ast_arg_to_arg(tcx, getter, _);
            auto i = Vec.map[ast.ty_arg, arg](f, inputs);
            auto out_ty = ast_ty_to_ty(tcx, getter, output);
            typ = ty.mk_fn(tcx, proto, i, out_ty);
        }

        case (ast.ty_path(?path, ?def)) {
            assert (def != none[ast.def]);
            alt (Option.get[ast.def](def)) {
                case (ast.def_ty(?id)) {
                    typ = instantiate(tcx, getter, id, path.node.types);
                }
                case (ast.def_native_ty(?id)) { typ = getter(id)._1; }
                case (ast.def_obj(?id)) {
                    typ = instantiate(tcx, getter, id, path.node.types);
                }
                case (ast.def_ty_arg(?id)) { typ = ty.mk_param(tcx, id); }
                case (_)                   { 
                    tcx.sess.span_err(ast_ty.span,
                       "found type name used as a variable");
                    fail; }
            }

            cname = some(path_to_str(path));
        }

        case (ast.ty_obj(?meths)) {
            let vec[ty.method] tmeths = vec();
            auto f = bind ast_arg_to_arg(tcx, getter, _);
            for (ast.ty_method m in meths) {
                auto ins = Vec.map[ast.ty_arg, arg](f, m.inputs);
                auto out = ast_ty_to_ty(tcx, getter, m.output);
                Vec.push[ty.method](tmeths,
                                  rec(proto=m.proto,
                                      ident=m.ident,
                                      inputs=ins,
                                      output=out));
            }

            typ = ty.mk_obj(tcx, ty.sort_methods(tmeths));
        }
    }

    alt (cname) {
        case (none[str]) { /* no-op */ }
        case (some[str](?cname_str)) {
            typ = ty.rename(tcx, typ, cname_str);
        }
    }
    ret typ;
}

// A convenience function to use a crate_ctxt to resolve names for
// ast_ty_to_ty.
fn ast_ty_to_ty_crate(@crate_ctxt ccx, &@ast.ty ast_ty) -> ty.t {
    fn getter(@crate_ctxt ccx, ast.def_id id) -> ty.ty_param_count_and_ty {
        ret ty.lookup_item_type(ccx.sess, ccx.tcx, ccx.type_cache, id);
    }
    auto f = bind getter(ccx, _);
    ret ast_ty_to_ty(ccx.tcx, f, ast_ty);
}


// Item collection - a pair of bootstrap passes:
//
// 1. Collect the IDs of all type items (typedefs) and store them in a table.
//
// 2. Translate the AST fragments that describe types to determine a type for
//    each item. When we encounter a named type, we consult the table built in
//    pass 1 to find its item, and recursively translate it.
//
// We then annotate the AST with the resulting types and return the annotated
// AST, along with a table mapping item IDs to their types.

mod Collect {
    type ctxt = rec(session.session sess,
                    @ty_item_table id_to_ty_item,
                    ty.type_cache type_cache,
                    ty.ctxt tcx);
    type env = rec(@ctxt cx, ast.native_abi abi);

    fn ty_of_fn_decl(@ctxt cx,
                     fn(&@ast.ty ast_ty) -> ty.t convert,
                     fn(&ast.arg a) -> arg ty_of_arg,
                     &ast.fn_decl decl,
                     ast.proto proto,
                     vec[ast.ty_param] ty_params,
                     ast.def_id def_id) -> ty.ty_param_count_and_ty {
        auto input_tys = Vec.map[ast.arg,arg](ty_of_arg, decl.inputs);
        auto output_ty = convert(decl.output);
        auto t_fn = ty.mk_fn(cx.tcx, proto, input_tys, output_ty);
        auto ty_param_count = Vec.len[ast.ty_param](ty_params);
        auto tpt = tup(ty_param_count, t_fn);
        cx.type_cache.insert(def_id, tpt);
        ret tpt;
    }

    fn ty_of_native_fn_decl(@ctxt cx,
                            fn(&@ast.ty ast_ty) -> ty.t convert,
                            fn(&ast.arg a) -> arg ty_of_arg,
                            &ast.fn_decl decl,
                            ast.native_abi abi,
                            vec[ast.ty_param] ty_params,
                            ast.def_id def_id) -> ty.ty_param_count_and_ty {
        auto input_tys = Vec.map[ast.arg,arg](ty_of_arg, decl.inputs);
        auto output_ty = convert(decl.output);
        auto t_fn = ty.mk_native_fn(cx.tcx, abi, input_tys, output_ty);
        auto ty_param_count = Vec.len[ast.ty_param](ty_params);
        auto tpt = tup(ty_param_count, t_fn);
        cx.type_cache.insert(def_id, tpt);
        ret tpt;
    }

    fn getter(@ctxt cx, ast.def_id id) -> ty.ty_param_count_and_ty {

        if (id._0 != cx.sess.get_targ_crate_num()) {
            // This is a type we need to load in from the crate reader.
            ret creader.get_type(cx.sess, cx.tcx, id);
        }

        // assert (cx.id_to_ty_item.contains_key(id));

        auto it = cx.id_to_ty_item.get(id);
        auto tpt;
        alt (it) {
            case (any_item_rust(?item)) { tpt = ty_of_item(cx, item); }
            case (any_item_native(?native_item, ?abi)) {
                tpt = ty_of_native_item(cx, native_item, abi);
            }
        }

        ret tpt;
    }

    fn ty_of_arg(@ctxt cx, &ast.arg a) -> arg {
        auto ty_mode = ast_mode_to_mode(a.mode);
        auto f = bind getter(cx, _);
        ret rec(mode=ty_mode, ty=ast_ty_to_ty(cx.tcx, f, a.ty));
    }

    fn ty_of_method(@ctxt cx, &@ast.method m) -> method {
        auto get = bind getter(cx, _);
        auto convert = bind ast_ty_to_ty(cx.tcx, get, _);
        auto f = bind ty_of_arg(cx, _);
        auto inputs = Vec.map[ast.arg,arg](f, m.node.meth.decl.inputs);
        auto output = convert(m.node.meth.decl.output);
        ret rec(proto=m.node.meth.proto, ident=m.node.ident,
                inputs=inputs, output=output);
    }

    fn ty_of_obj(@ctxt cx,
                 ast.ident id,
                 &ast._obj obj_info,
                 vec[ast.ty_param] ty_params) -> ty.ty_param_count_and_ty {
        auto f = bind ty_of_method(cx, _);
        auto methods = Vec.map[@ast.method,method](f, obj_info.methods);

        auto t_obj = ty.mk_obj(cx.tcx, ty.sort_methods(methods));
        t_obj = ty.rename(cx.tcx, t_obj, id);
        auto ty_param_count = Vec.len[ast.ty_param](ty_params);
        ret tup(ty_param_count, t_obj);
    }

    fn ty_of_obj_ctor(@ctxt cx,
                      &ast.ident id,
                      &ast._obj obj_info,
                      ast.def_id obj_ty_id,
                      vec[ast.ty_param] ty_params)
            -> ty.ty_param_count_and_ty {
        auto t_obj = ty_of_obj(cx, id, obj_info, ty_params);
        let vec[arg] t_inputs = vec();
        for (ast.obj_field f in obj_info.fields) {
            auto g = bind getter(cx, _);
            auto t_field = ast_ty_to_ty(cx.tcx, g, f.ty);
            Vec.push[arg](t_inputs, rec(mode=ty.mo_alias, ty=t_field));
        }

        cx.type_cache.insert(obj_ty_id, t_obj);

        auto t_fn = ty.mk_fn(cx.tcx, ast.proto_fn, t_inputs, t_obj._1);
        ret tup(t_obj._0, t_fn);
    }

    fn ty_of_item(@ctxt cx, @ast.item it) -> ty.ty_param_count_and_ty {

        auto get = bind getter(cx, _);
        auto convert = bind ast_ty_to_ty(cx.tcx, get, _);

        alt (it.node) {

            case (ast.item_const(?ident, ?t, _, ?def_id, _)) {
                auto typ = convert(t);
                auto tpt = tup(0u, typ);
                cx.type_cache.insert(def_id, tpt);
                ret tpt;
            }

            case (ast.item_fn(?ident, ?fn_info, ?tps, ?def_id, _)) {
                auto f = bind ty_of_arg(cx, _);
                ret ty_of_fn_decl(cx, convert, f, fn_info.decl, fn_info.proto,
                                  tps, def_id);
            }

            case (ast.item_obj(?ident, ?obj_info, ?tps, ?odid, _)) {
                auto t_ctor = ty_of_obj_ctor(cx, ident, obj_info, odid.ty,
                                             tps);
                cx.type_cache.insert(odid.ctor, t_ctor);
                ret cx.type_cache.get(odid.ty);
            }

            case (ast.item_ty(?ident, ?t, ?tps, ?def_id, _)) {
                alt (cx.type_cache.find(def_id)) {
                    case (some[ty.ty_param_count_and_ty](?tpt)) {
                        ret tpt;
                    }
                    case (none[ty.ty_param_count_and_ty]) {}
                }

                // Tell ast_ty_to_ty() that we want to perform a recursive
                // call to resolve any named types.
                auto typ = convert(t);
                auto ty_param_count = Vec.len[ast.ty_param](tps);
                auto tpt = tup(ty_param_count, typ);
                cx.type_cache.insert(def_id, tpt);
                ret tpt;
            }

            case (ast.item_tag(_, _, ?tps, ?def_id, _)) {
                // Create a new generic polytype.
                let vec[ty.t] subtys = vec();

                auto i = 0u;
                for (ast.ty_param tp in tps) {
                    subtys += vec(ty.mk_param(cx.tcx, i));
                    i += 1u;
                }

                auto t = ty.mk_tag(cx.tcx, def_id, subtys);

                auto ty_param_count = Vec.len[ast.ty_param](tps);
                auto tpt = tup(ty_param_count, t);
                cx.type_cache.insert(def_id, tpt);
                ret tpt;
            }

            case (ast.item_mod(_, _, _)) { fail; }
            case (ast.item_native_mod(_, _, _)) { fail; }
        }
    }

    fn ty_of_native_item(@ctxt cx, @ast.native_item it, ast.native_abi abi)
            -> ty.ty_param_count_and_ty {
        alt (it.node) {
            case (ast.native_item_fn(?ident, ?lname, ?fn_decl,
                                     ?params, ?def_id, _)) {
                auto get = bind getter(cx, _);
                auto convert = bind ast_ty_to_ty(cx.tcx, get, _);
                auto f = bind ty_of_arg(cx, _);
                ret ty_of_native_fn_decl(cx, convert, f, fn_decl, abi, params,
                                         def_id);
            }
            case (ast.native_item_ty(_, ?def_id)) {
                alt (cx.type_cache.find(def_id)) {
                    case (some[ty.ty_param_count_and_ty](?tpt)) {
                        ret tpt;
                    }
                    case (none[ty.ty_param_count_and_ty]) {}
                }

                auto t = ty.mk_native(cx.tcx);
                auto tpt = tup(0u, t);
                cx.type_cache.insert(def_id, tpt);
                ret tpt;
            }
        }
    }

    fn get_tag_variant_types(@ctxt cx, &ast.def_id tag_id,
                             &vec[ast.variant] variants,
                             &vec[ast.ty_param] ty_params)
            -> vec[ast.variant] {
        let vec[ast.variant] result = vec();

        // Create a set of parameter types shared among all the variants.
        let vec[ty.t] ty_param_tys = vec();
        auto i = 0u;
        for (ast.ty_param tp in ty_params) {
            ty_param_tys += vec(ty.mk_param(cx.tcx, i));
            i += 1u;
        }

        auto ty_param_count = Vec.len[ast.ty_param](ty_params);

        for (ast.variant variant in variants) {
            // Nullary tag constructors get turned into constants; n-ary tag
            // constructors get turned into functions.
            auto result_ty;
            if (Vec.len[ast.variant_arg](variant.node.args) == 0u) {
                result_ty = ty.mk_tag(cx.tcx, tag_id, ty_param_tys);
            } else {
                // As above, tell ast_ty_to_ty() that trans_ty_item_to_ty()
                // should be called to resolve named types.
                auto f = bind getter(cx, _);

                let vec[arg] args = vec();
                for (ast.variant_arg va in variant.node.args) {
                    auto arg_ty = ast_ty_to_ty(cx.tcx, f, va.ty);
                    args += vec(rec(mode=ty.mo_alias, ty=arg_ty));
                }
                auto tag_t = ty.mk_tag(cx.tcx, tag_id, ty_param_tys);
                result_ty = ty.mk_fn(cx.tcx, ast.proto_fn, args, tag_t);
            }

            auto tpt = tup(ty_param_count, result_ty);
            cx.type_cache.insert(variant.node.id, tpt);
            auto variant_t = rec(ann=triv_ann(result_ty)
                with variant.node
            );
            result += vec(fold.respan[ast.variant_](variant.span, variant_t));
        }

        ret result;
    }

    fn collect(&@ty_item_table id_to_ty_item, @ast.item i) -> @ty_item_table {
        alt (i.node) {
            case (ast.item_ty(_, _, _, ?def_id, _)) {
                id_to_ty_item.insert(def_id, any_item_rust(i));
            }
            case (ast.item_tag(_, _, _, ?def_id, _)) {
                id_to_ty_item.insert(def_id, any_item_rust(i));
            }
            case (ast.item_obj(_, _, _, ?odid, _)) {
                id_to_ty_item.insert(odid.ty, any_item_rust(i));
            }
            case (_) { /* empty */ }
        }
        ret id_to_ty_item;
    }

    fn collect_native(&@ty_item_table id_to_ty_item, @ast.native_item i)
        -> @ty_item_table {
        alt (i.node) {
            case (ast.native_item_ty(_, ?def_id)) {
                // The abi of types is not used.
                id_to_ty_item.insert(def_id,
                                     any_item_native(i,
                                                     ast.native_abi_cdecl));
            }
            case (_) {
            }
        }
        ret id_to_ty_item;
    }

    fn convert(&@env e, @ast.item i) -> @env {
        auto abi = e.abi;
        alt (i.node) {
            case (ast.item_mod(_, _, _)) {
                // ignore item_mod, it has no type.
            }
            case (ast.item_native_mod(_, ?native_mod, _)) {
                // ignore item_native_mod, it has no type.
                abi = native_mod.abi;
            }
            case (_) {
                // This call populates the ty_table with the converted type of
                // the item in passing; we don't need to do anything else.
                ty_of_item(e.cx, i);
            }
        }
        ret @rec(abi=abi with *e);
    }

    fn convert_native(&@env e, @ast.native_item i) -> @env {
        ty_of_native_item(e.cx, i, e.abi);
        ret e;
    }

    fn fold_item_const(&@env e, &span sp, ast.ident i,
                       @ast.ty t, @ast.expr ex,
                       ast.def_id id, ast.ann a) -> @ast.item {
        // assert (e.cx.type_cache.contains_key(id));
        auto typ = e.cx.type_cache.get(id)._1;
        auto item = ast.item_const(i, t, ex, id, triv_ann(typ));
        ret @fold.respan[ast.item_](sp, item);
    }

    fn fold_item_fn(&@env e, &span sp, ast.ident i,
                    &ast._fn f, vec[ast.ty_param] ty_params,
                    ast.def_id id, ast.ann a) -> @ast.item {
        // assert (e.cx.type_cache.contains_key(id));
        auto typ = e.cx.type_cache.get(id)._1;
        auto item = ast.item_fn(i, f, ty_params, id, triv_ann(typ));
        ret @fold.respan[ast.item_](sp, item);
    }

    fn fold_native_item_fn(&@env e, &span sp, ast.ident i, Option.t[str] ln,
                           &ast.fn_decl d, vec[ast.ty_param] ty_params,
                           ast.def_id id, ast.ann a) -> @ast.native_item {
        // assert (e.cx.type_cache.contains_key(id));
        auto typ = e.cx.type_cache.get(id)._1;
        auto item = ast.native_item_fn(i, ln, d, ty_params, id,
                                       triv_ann(typ));
        ret @fold.respan[ast.native_item_](sp, item);
    }

    fn get_ctor_obj_methods(&@env e, ty.t t) -> vec[method] {
        alt (struct(e.cx.tcx, t)) {
            case (ty.ty_fn(_,_,?tobj)) {
                alt (struct(e.cx.tcx, tobj)) {
                    case (ty.ty_obj(?tm)) {
                        ret tm;
                    }
                    case (_) {
                        let vec[method] tm = vec();
                        ret tm;
                    }
                }
            }
            case (_) {
                let vec[method] tm = vec();
                ret tm;
            }
        }
    }


    fn fold_item_obj(&@env e, &span sp, ast.ident i,
                    &ast._obj ob, vec[ast.ty_param] ty_params,
                    ast.obj_def_ids odid, ast.ann a) -> @ast.item {
        // assert (e.cx.type_cache.contains_key(odid.ctor));
        auto t = e.cx.type_cache.get(odid.ctor)._1;
        let vec[method] meth_tys = get_ctor_obj_methods(e, t);
        let vec[@ast.method] methods = vec();
        let vec[ast.obj_field] fields = vec();

        for (@ast.method meth in ob.methods) {
            let uint ix = ty.method_idx(e.cx.sess,
                                        sp, meth.node.ident,
                                        meth_tys);
            let method meth_ty = meth_tys.(ix);
            let ast.method_ m_;
            let @ast.method m;
            auto meth_tfn = ty.mk_fn(e.cx.tcx,
                                     meth_ty.proto,
                                     meth_ty.inputs,
                                     meth_ty.output);
            m_ = rec(ann=triv_ann(meth_tfn)
                with meth.node
            );
            m = @rec(node=m_ with *meth);
            Vec.push[@ast.method](methods, m);
        }
        auto g = bind getter(e.cx, _);
        for (ast.obj_field fld in ob.fields) {
            let ty.t fty = ast_ty_to_ty(e.cx.tcx, g, fld.ty);
            let ast.obj_field f = rec(ann=triv_ann(fty)
                with fld
            );
            Vec.push[ast.obj_field](fields, f);
        }

        auto dtor = none[@ast.method];
        alt (ob.dtor) {
            case (some[@ast.method](?d)) {
                let vec[arg] inputs = vec();
                let ty.t output = ty.mk_nil(e.cx.tcx);
                auto dtor_tfn = ty.mk_fn(e.cx.tcx, ast.proto_fn, inputs,
                                         output);
                auto d_ = rec(ann=triv_ann(dtor_tfn) with d.node);
                dtor = some[@ast.method](@rec(node=d_ with *d));
            }
            case (none[@ast.method]) { }
        }

        auto ob_ = rec(methods = methods,
                       fields = fields,
                       dtor = dtor
                       with ob);
        auto item = ast.item_obj(i, ob_, ty_params, odid, triv_ann(t));
        ret @fold.respan[ast.item_](sp, item);
    }

    fn fold_item_ty(&@env e, &span sp, ast.ident i,
                    @ast.ty t, vec[ast.ty_param] ty_params,
                    ast.def_id id, ast.ann a) -> @ast.item {
        // assert (e.cx.type_cache.contains_key(id));
        auto typ = e.cx.type_cache.get(id)._1;
        auto item = ast.item_ty(i, t, ty_params, id, triv_ann(typ));
        ret @fold.respan[ast.item_](sp, item);
    }

    fn fold_item_tag(&@env e, &span sp, ast.ident i,
                     vec[ast.variant] variants,
                     vec[ast.ty_param] ty_params,
                     ast.def_id id, ast.ann a) -> @ast.item {
        auto variants_t = get_tag_variant_types(e.cx, id, variants,
                                                ty_params);
        auto typ = e.cx.type_cache.get(id)._1;
        auto item = ast.item_tag(i, variants_t, ty_params, id,
                                 ast.ann_type(typ, none[vec[ty.t]],
                                              none[@ts_ann]));
        ret @fold.respan[ast.item_](sp, item);
    }

    fn collect_item_types(session.session sess, ty.ctxt tcx, @ast.crate crate)
            -> tup(@ast.crate, ty.type_cache, @ty_item_table) {
        // First pass: collect all type item IDs.
        auto module = crate.node.module;
        auto id_to_ty_item = @common.new_def_hash[any_item]();

        auto fld_1 = fold.new_identity_fold[@ty_item_table]();
        fld_1 = @rec(update_env_for_item = bind collect(_, _),
                     update_env_for_native_item = bind collect_native(_, _)
                     with *fld_1);
        fold.fold_crate[@ty_item_table](id_to_ty_item, fld_1, crate);

        // Second pass: translate the types of all items.
        auto type_cache = common.new_def_hash[ty.ty_param_count_and_ty]();

        auto cx = @rec(sess=sess,
                       id_to_ty_item=id_to_ty_item,
                       type_cache=type_cache,
                       tcx=tcx);

        let @env e = @rec(cx=cx, abi=ast.native_abi_cdecl);

        auto fld_2 = fold.new_identity_fold[@env]();
        fld_2 =
            @rec(update_env_for_item = bind convert(_,_),
                 update_env_for_native_item = bind convert_native(_,_),
                 fold_item_const = bind fold_item_const(_,_,_,_,_,_,_),
                 fold_item_fn    = bind fold_item_fn(_,_,_,_,_,_,_),
                 fold_native_item_fn =
                    bind fold_native_item_fn(_,_,_,_,_,_,_,_),
                 fold_item_obj   = bind fold_item_obj(_,_,_,_,_,_,_),
                 fold_item_ty    = bind fold_item_ty(_,_,_,_,_,_,_),
                 fold_item_tag   = bind fold_item_tag(_,_,_,_,_,_,_)
                 with *fld_2);
        auto crate_ = fold.fold_crate[@env](e, fld_2, crate);
        ret tup(crate_, type_cache, id_to_ty_item);
    }
}


// Type unification

mod Unify {
    fn simple(@fn_ctxt fcx, ty.t expected, ty.t actual) -> ty.Unify.result {
        // FIXME: horrid botch
        let vec[mutable ty.t] param_substs =
            vec(mutable ty.mk_nil(fcx.ccx.tcx));
        Vec.pop(param_substs);
        ret with_params(fcx, expected, actual, param_substs);
    }

    fn with_params(@fn_ctxt fcx, ty.t expected, ty.t actual,
                   vec[mutable ty.t] param_substs) -> ty.Unify.result {
        auto cache_key = tup(expected, actual, param_substs);
        alt (fcx.ccx.unify_cache.find(cache_key)) {
            case (some[ty.Unify.result](?r)) {
                fcx.ccx.cache_hits += 1u;
                ret r;
            }
            case (none[ty.Unify.result]) {
                fcx.ccx.cache_misses += 1u;
            }
        }

        obj unify_handler(@fn_ctxt fcx, vec[mutable ty.t] param_substs) {
            fn resolve_local(ast.def_id id) -> Option.t[ty.t] {
                alt (fcx.locals.find(id)) {
                    case (none[ty.t]) { ret none[ty.t]; }
                    case (some[ty.t](?existing_type)) {
                        if (ty.type_contains_vars(fcx.ccx.tcx,
                                                  existing_type)) {
                            // Not fully resolved yet. The writeback phase
                            // will mop up.
                            ret none[ty.t];
                        }
                        ret some[ty.t](existing_type);
                    }
                }
            }
            fn record_local(ast.def_id id, ty.t new_type) {
                auto unified_type;
                alt (fcx.locals.find(id)) {
                    case (none[ty.t]) { unified_type = new_type; }
                    case (some[ty.t](?old_type)) {
                        alt (with_params(fcx, old_type, new_type,
                                         param_substs)) {
                            case (ures_ok(?ut)) { unified_type = ut; }
                            case (_) { fail; /* FIXME */ }
                        }
                    }
                }

                // TODO: "freeze"
                let vec[ty.t] param_substs_1 = vec();
                for (ty.t subst in param_substs) {
                    param_substs_1 += vec(subst);
                }

                unified_type =
                    ty.substitute_type_params(fcx.ccx.tcx, param_substs_1,
                                              unified_type);
                fcx.locals.insert(id, unified_type);
            }
            fn record_param(uint index, ty.t binding) -> ty.Unify.result {
                // Unify with the appropriate type in the parameter
                // substitution List.
                auto old_subst = param_substs.(index);

                auto result = with_params(fcx, old_subst, binding,
                                          param_substs);
                alt (result) {
                    case (ures_ok(?new_subst)) {
                        param_substs.(index) = new_subst;
                        ret ures_ok(ty.mk_bound_param(fcx.ccx.tcx, index));
                    }
                    case (_) { ret result; }
                }
            }
        }


        auto handler = unify_handler(fcx, param_substs);
        auto result = ty.Unify.unify(expected, actual, handler, fcx.ccx.tcx);
        fcx.ccx.unify_cache.insert(cache_key, result);
        ret result;
    }
}


tag autoderef_kind {
    AUTODEREF_OK;
    NO_AUTODEREF;
}

fn strip_boxes(ty.ctxt tcx, ty.t t) -> ty.t {
    auto t1 = t;
    while (true) {
        alt (struct(tcx, t1)) {
            case (ty.ty_box(?inner)) { t1 = inner.ty; }
            case (_) { ret t1; }
        }
    }
    fail;
}

fn add_boxes(@crate_ctxt ccx, uint n, ty.t t) -> ty.t {
    auto t1 = t;
    while (n != 0u) {
        t1 = ty.mk_imm_box(ccx.tcx, t1);
        n -= 1u;
    }
    ret t1;
}


fn count_boxes(ty.ctxt tcx, ty.t t) -> uint {
    auto n = 0u;
    auto t1 = t;
    while (true) {
        alt (struct(tcx, t1)) {
            case (ty.ty_box(?inner)) { n += 1u; t1 = inner.ty; }
            case (_) { ret n; }
        }
    }
    fail;
}


// Demands - procedures that require that two types unify and emit an error
// message if they don't.

type ty_param_substs_and_ty = tup(vec[ty.t], ty.t);

mod Demand {
    fn simple(@fn_ctxt fcx, &span sp, ty.t expected, ty.t actual) -> ty.t {
        let vec[ty.t] tps = vec();
        ret full(fcx, sp, expected, actual, tps, NO_AUTODEREF)._1;
    }

    fn autoderef(@fn_ctxt fcx, &span sp, ty.t expected, ty.t actual,
                 autoderef_kind adk) -> ty.t {
        let vec[ty.t] tps = vec();
        ret full(fcx, sp, expected, actual, tps, adk)._1;
    }

    // Requires that the two types unify, and prints an error message if they
    // don't. Returns the unified type and the type parameter substitutions.

    fn full(@fn_ctxt fcx, &span sp, ty.t expected, ty.t actual,
            vec[ty.t] ty_param_substs_0, autoderef_kind adk)
            -> ty_param_substs_and_ty {

        auto expected_1 = expected;
        auto actual_1 = actual;
        auto implicit_boxes = 0u;

        if (adk == AUTODEREF_OK) {
            expected_1 = strip_boxes(fcx.ccx.tcx, expected_1);
            actual_1 = strip_boxes(fcx.ccx.tcx, actual_1);
            implicit_boxes = count_boxes(fcx.ccx.tcx, actual);
        }

        let vec[mutable ty.t] ty_param_substs =
            vec(mutable ty.mk_nil(fcx.ccx.tcx));
        Vec.pop(ty_param_substs);   // FIXME: horrid botch
        for (ty.t ty_param_subst in ty_param_substs_0) {
            ty_param_substs += vec(mutable ty_param_subst);
        }

        alt (Unify.with_params(fcx, expected_1, actual_1, ty_param_substs)) {
            case (ures_ok(?t)) {
                // TODO: Use "freeze", when we have it.
                let vec[ty.t] result_ty_param_substs = vec();
                for (ty.t ty_param_subst in ty_param_substs) {
                    result_ty_param_substs += vec(ty_param_subst);
                }

                ret tup(result_ty_param_substs,
                        add_boxes(fcx.ccx, implicit_boxes, t));
            }

            case (ures_err(?err, ?expected, ?actual)) {
                fcx.ccx.sess.span_err(sp, "mismatched types: expected "
                    + ty_to_str(fcx.ccx.tcx, expected) + " but found "
                    + ty_to_str(fcx.ccx.tcx, actual) + " ("
                    + ty.type_err_to_str(err) + ")");

                // TODO: In the future, try returning "expected", reporting
                // the error, and continue.
                fail;
            }
        }
    }
}


// Returns true if the two types unify and false if they don't.
fn are_compatible(&@fn_ctxt fcx, ty.t expected, ty.t actual) -> bool {
    alt (Unify.simple(fcx, expected, actual)) {
        case (ures_ok(_))        { ret true;  }
        case (ures_err(_, _, _)) { ret false; }
    }
}

// Returns the types of the arguments to a tag variant.
fn variant_arg_types(@crate_ctxt ccx, &span sp, ast.def_id vid,
                     vec[ty.t] tag_ty_params) -> vec[ty.t] {
    auto ty_param_count = Vec.len[ty.t](tag_ty_params);

    let vec[ty.t] result = vec();

    auto tpt = ty.lookup_item_type(ccx.sess, ccx.tcx, ccx.type_cache, vid);
    alt (struct(ccx.tcx, tpt._1)) {
        case (ty.ty_fn(_, ?ins, _)) {
            // N-ary variant.
            for (ty.arg arg in ins) {
                auto arg_ty = bind_params_in_type(ccx.tcx, arg.ty);
                arg_ty = substitute_ty_params(ccx, arg_ty, ty_param_count,
                                              tag_ty_params, sp);
                result += vec(arg_ty);
            }
        }
        case (_) {
            // Nullary variant. Do nothing, as there are no arguments.
        }
    }

    ret result;
}


// The "push-down" phase, which takes a typed grammar production and pushes
// its type down into its constituent parts.
//
// For example, consider "auto x; x = 352;". check_expr() doesn't know the
// type of "x" at the time it sees it, so that function will simply store a
// type variable for the type of "x". However, after checking the entire
// assignment expression, check_expr() will assign the type of int to the
// expression "x = 352" as a whole. In this case, then, the job of these
// functions is to clean up by assigning the type of int to both sides of the
// assignment expression.
//
// TODO: We only need to do this once per statement: check_expr() bubbles the
// types up, and pushdown_expr() pushes the types down. However, in many cases
// we're more eager than we need to be, calling pushdown_expr() and friends
// directly inside check_expr(). This results in a quadratic algorithm.

mod Pushdown {
    // Push-down over typed patterns. Note that the pattern that you pass to
    // this function must have been passed to check_pat() first.
    //
    // TODO: enforce this via a predicate.

    fn pushdown_pat(&@fn_ctxt fcx, ty.t expected, @ast.pat pat) -> @ast.pat {
        auto p_1;

        alt (pat.node) {
            case (ast.pat_wild(?ann)) {
                auto t = Demand.simple(fcx, pat.span, expected,
                                       ann_to_type(ann));
                p_1 = ast.pat_wild(ast.ann_type(t, none[vec[ty.t]],
                                                none[@ts_ann]));
            }
            case (ast.pat_lit(?lit, ?ann)) {
                auto t = Demand.simple(fcx, pat.span, expected,
                                       ann_to_type(ann));
                p_1 = ast.pat_lit(lit, ast.ann_type(t, none[vec[ty.t]],
                                                    none[@ts_ann]));
            }
            case (ast.pat_bind(?id, ?did, ?ann)) {
                auto t = Demand.simple(fcx, pat.span, expected,
                                       ann_to_type(ann));
                fcx.locals.insert(did, t);
                p_1 = ast.pat_bind(id, did, ast.ann_type(t,
                                                         none[vec[ty.t]],
                                                         none[@ts_ann]));
            }
            case (ast.pat_tag(?id, ?subpats, ?vdef_opt, ?ann)) {
                // Take the variant's type parameters out of the expected
                // type.
                auto tag_tps;
                alt (struct(fcx.ccx.tcx, expected)) {
                    case (ty.ty_tag(_, ?tps)) { tag_tps = tps; }
                    case (_) {
                        log_err "tag pattern type not actually a tag?!";
                        fail;
                    }
                }

                // Get the types of the arguments of the variant.
                auto vdef = Option.get[ast.variant_def](vdef_opt);
                auto arg_tys = variant_arg_types(fcx.ccx, pat.span, vdef._1,
                                                 tag_tps);

                let vec[@ast.pat] subpats_1 = vec();
                auto i = 0u;
                for (@ast.pat subpat in subpats) {
                    subpats_1 += vec(pushdown_pat(fcx, arg_tys.(i), subpat));
                    i += 1u;
                }

                // TODO: push down type from "expected".
                p_1 = ast.pat_tag(id, subpats_1, vdef_opt, ann);
            }
        }

        ret @fold.respan[ast.pat_](pat.span, p_1);
    }

    // Push-down over typed expressions. Note that the expression that you
    // pass to this function must have been passed to check_expr() first.
    //
    // TODO: enforce this via a predicate.
    // TODO: This function is incomplete.

    fn pushdown_expr(&@fn_ctxt fcx, ty.t expected, @ast.expr e)
            -> @ast.expr {
        be pushdown_expr_full(fcx, expected, e, NO_AUTODEREF);
    }

    fn pushdown_expr_full(&@fn_ctxt fcx, ty.t expected, @ast.expr e,
                          autoderef_kind adk) -> @ast.expr {
        auto e_1;

        alt (e.node) {
            case (ast.expr_vec(?es_0, ?mut, ?ann)) {
                // TODO: enforce mutability

                auto t = Demand.simple(fcx, e.span, expected,
                                       ann_to_type(ann));
                let vec[@ast.expr] es_1 = vec();
                alt (struct(fcx.ccx.tcx, t)) {
                    case (ty.ty_vec(?mt)) {
                        for (@ast.expr e_0 in es_0) {
                            es_1 += vec(pushdown_expr(fcx, mt.ty, e_0));
                        }
                    }
                    case (_) {
                        log_err "vec expr doesn't have a vec type!";
                        fail;
                    }
                }
                e_1 = ast.expr_vec(es_1, mut, triv_ann(t));
            }
            case (ast.expr_tup(?es_0, ?ann)) {
                auto t = Demand.simple(fcx, e.span, expected,
                                       ann_to_type(ann));
                let vec[ast.elt] elts_1 = vec();
                alt (struct(fcx.ccx.tcx, t)) {
                    case (ty.ty_tup(?mts)) {
                        auto i = 0u;
                        for (ast.elt elt_0 in es_0) {
                            auto e_1 = pushdown_expr(fcx, mts.(i).ty,
                                                     elt_0.expr);
                            elts_1 += vec(rec(mut=elt_0.mut, expr=e_1));
                            i += 1u;
                        }
                    }
                    case (_) {
                        log_err "tup expr doesn't have a tup type!";
                        fail;
                    }
                }
                e_1 = ast.expr_tup(elts_1, triv_ann(t));
            }
            case (ast.expr_rec(?fields_0, ?base_0, ?ann)) {

                auto base_1 = base_0;

                auto t = Demand.simple(fcx, e.span, expected,
                                       ann_to_type(ann));
                let vec[ast.field] fields_1 = vec();
                alt (struct(fcx.ccx.tcx, t)) {
                    case (ty.ty_rec(?field_mts)) {
                        alt (base_0) {
                            case (none[@ast.expr]) {
                                auto i = 0u;
                                for (ast.field field_0 in fields_0) {
                                    assert (Str.eq(field_0.ident,
                                                   field_mts.(i).ident));
                                    auto e_1 =
                                        pushdown_expr(fcx,
                                                      field_mts.(i).mt.ty,
                                                      field_0.expr);
                                    fields_1 += vec(rec(mut=field_0.mut,
                                                        ident=field_0.ident,
                                                        expr=e_1));
                                    i += 1u;
                                }
                            }
                            case (some[@ast.expr](?bx)) {

                                base_1 = some[@ast.expr](pushdown_expr(fcx, t,
                                                                       bx));

                                let vec[field] base_fields = vec();

                                for (ast.field field_0 in fields_0) {

                                    for (ty.field ft in field_mts) {
                                        if (Str.eq(field_0.ident,
                                                    ft.ident)) {
                                            auto e_1 =
                                                pushdown_expr(fcx, ft.mt.ty,
                                                              field_0.expr);
                                            fields_1 +=
                                                vec(rec(mut=field_0.mut,
                                                        ident=field_0.ident,
                                                        expr=e_1));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    case (_) {
                        log_err "rec expr doesn't have a rec type!";
                        fail;
                    }
                }
                e_1 = ast.expr_rec(fields_1, base_1, triv_ann(t));
            }
            case (ast.expr_bind(?sube, ?es, ?ann)) {
                auto t = Demand.simple(fcx, e.span, expected,
                                       ann_to_type(ann));
                e_1 = ast.expr_bind(sube, es, triv_ann(t));
            }
            case (ast.expr_call(?sube, ?es, ?ann)) {
                // NB: we call 'Demand.autoderef' and pass in adk only in
                // cases where e is an expression that could *possibly*
                // produce a box; things like expr_binary or expr_bind can't,
                // so there's no need.
                auto t = Demand.autoderef(fcx, e.span, expected,
                                          ann_to_type(ann), adk);
                e_1 = ast.expr_call(sube, es, triv_ann(t));
            }
            case (ast.expr_self_method(?id, ?ann)) {
                auto t = Demand.simple(fcx, e.span, expected,
                                       ann_to_type(ann));
                e_1 = ast.expr_self_method(id, triv_ann(t));
            }
            case (ast.expr_binary(?bop, ?lhs, ?rhs, ?ann)) {
                auto t = Demand.simple(fcx, e.span, expected,
                                       ann_to_type(ann));
                e_1 = ast.expr_binary(bop, lhs, rhs, triv_ann(t));
            }
            case (ast.expr_unary(?uop, ?sube, ?ann)) {
                // See note in expr_unary for why we're calling
                // Demand.autoderef.
                auto t = Demand.autoderef(fcx, e.span, expected,
                                          ann_to_type(ann), adk);
                e_1 = ast.expr_unary(uop, sube, triv_ann(t));
            }
            case (ast.expr_lit(?lit, ?ann)) {
                auto t = Demand.simple(fcx, e.span, expected,
                                       ann_to_type(ann));
                e_1 = ast.expr_lit(lit, triv_ann(t));
            }
            case (ast.expr_cast(?sube, ?ast_ty, ?ann)) {
                auto t = Demand.simple(fcx, e.span, expected,
                                       ann_to_type(ann));
                e_1 = ast.expr_cast(sube, ast_ty, triv_ann(t));
            }
            case (ast.expr_if(?cond, ?then_0, ?else_0, ?ann)) {
                auto t = Demand.autoderef(fcx, e.span, expected,
                                          ann_to_type(ann), adk);
                auto then_1 = pushdown_block(fcx, expected, then_0);

                auto else_1;
                alt (else_0) {
                    case (none[@ast.expr]) { else_1 = none[@ast.expr]; }
                    case (some[@ast.expr](?e_0)) {
                        auto e_1 = pushdown_expr(fcx, expected, e_0);
                        else_1 = some[@ast.expr](e_1);
                    }
                }
                e_1 = ast.expr_if(cond, then_1, else_1, triv_ann(t));
            }
            case (ast.expr_for(?decl, ?seq, ?bloc, ?ann)) {
                auto t = Demand.simple(fcx, e.span, expected,
                                       ann_to_type(ann));
                e_1 = ast.expr_for(decl, seq, bloc, triv_ann(t));
            }
            case (ast.expr_for_each(?decl, ?seq, ?bloc, ?ann)) {
                auto t = Demand.simple(fcx, e.span, expected,
                                       ann_to_type(ann));
                e_1 = ast.expr_for_each(decl, seq, bloc, triv_ann(t));
            }
            case (ast.expr_while(?cond, ?bloc, ?ann)) {
                auto t = Demand.simple(fcx, e.span, expected,
                                       ann_to_type(ann));
                e_1 = ast.expr_while(cond, bloc, triv_ann(t));
            }
            case (ast.expr_do_while(?bloc, ?cond, ?ann)) {
                auto t = Demand.simple(fcx, e.span, expected,
                                       ann_to_type(ann));
                e_1 = ast.expr_do_while(bloc, cond, triv_ann(t));
            }
            case (ast.expr_block(?bloc, ?ann)) {
                auto t = Demand.autoderef(fcx, e.span, expected,
                                          ann_to_type(ann), adk);
                e_1 = ast.expr_block(bloc, triv_ann(t));
            }
            case (ast.expr_assign(?lhs_0, ?rhs_0, ?ann)) {
                auto t = Demand.autoderef(fcx, e.span, expected,
                                          ann_to_type(ann), adk);
                auto lhs_1 = pushdown_expr(fcx, expected, lhs_0);
                auto rhs_1 = pushdown_expr(fcx, expected, rhs_0);
                e_1 = ast.expr_assign(lhs_1, rhs_1, triv_ann(t));
            }
            case (ast.expr_assign_op(?op, ?lhs_0, ?rhs_0, ?ann)) {
                auto t = Demand.autoderef(fcx, e.span, expected,
                                          ann_to_type(ann), adk);
                auto lhs_1 = pushdown_expr(fcx, expected, lhs_0);
                auto rhs_1 = pushdown_expr(fcx, expected, rhs_0);
                e_1 = ast.expr_assign_op(op, lhs_1, rhs_1, triv_ann(t));
            }
            case (ast.expr_field(?lhs, ?rhs, ?ann)) {
                auto t = Demand.autoderef(fcx, e.span, expected,
                                          ann_to_type(ann), adk);
                e_1 = ast.expr_field(lhs, rhs, triv_ann(t));
            }
            case (ast.expr_index(?base, ?index, ?ann)) {
                auto t = Demand.autoderef(fcx, e.span, expected,
                                          ann_to_type(ann), adk);
                e_1 = ast.expr_index(base, index, triv_ann(t));
            }
            case (ast.expr_path(?pth, ?d, ?ann)) {
                auto tp_substs_0 = ty.ann_to_type_params(ann);
                auto t_0 = ann_to_type(ann);

                auto result_0 = Demand.full(fcx, e.span, expected, t_0,
                                            tp_substs_0, adk);
                auto t = result_0._1;

                // Fill in the type parameter substitutions if they weren't
                // provided by the programmer.
                auto ty_params_opt;
                alt (ann) {
                    case (ast.ann_none) {
                        log_err "pushdown_expr(): no type annotation for " +
                            "path expr; did you pass it to check_expr()?";
                        fail;
                    }
                    case (ast.ann_type(_, ?tps_opt, _)) {
                        alt (tps_opt) {
                            case (none[vec[ty.t]]) {
                                ty_params_opt = none[vec[ty.t]];
                            }
                            case (some[vec[ty.t]](?tps)) {
                                ty_params_opt = some[vec[ty.t]](tps);
                            }
                        }
                    }
                }

                e_1 = ast.expr_path(pth, d,
                                    ast.ann_type(t, ty_params_opt,
                                                 none[@ts_ann]));
            }
            case (ast.expr_ext(?p, ?args, ?body, ?expanded, ?ann)) {
                auto t = Demand.autoderef(fcx, e.span, expected,
                                          ann_to_type(ann), adk);
                e_1 = ast.expr_ext(p, args, body, expanded, triv_ann(t));
            }
            /* FIXME: should this check the type annotations? */
            case (ast.expr_fail(_))  { e_1 = e.node; } 
            case (ast.expr_log(_,_,_)) { e_1 = e.node; } 
            case (ast.expr_break(_)) { e_1 = e.node; }
            case (ast.expr_cont(_))  { e_1 = e.node; }
            case (ast.expr_ret(_,_)) { e_1 = e.node; }
            case (ast.expr_put(_,_)) { e_1 = e.node; }
            case (ast.expr_be(_,_))  { e_1 = e.node; }
            case (ast.expr_check(_,_)) { e_1 = e.node; }
            case (ast.expr_assert(_,_)) { e_1 = e.node; }

            case (ast.expr_port(?ann)) {
                auto t = Demand.simple(fcx, e.span, expected,
                                       ann_to_type(ann));
                e_1 = ast.expr_port(triv_ann(t));
            }

            case (ast.expr_chan(?es, ?ann)) {
                auto t = Demand.simple(fcx, e.span, expected,
                                       ann_to_type(ann));
                let @ast.expr es_1;
                alt (struct(fcx.ccx.tcx, t)) {
                    case (ty.ty_chan(?subty)) {
                        auto pt = ty.mk_port(fcx.ccx.tcx, subty);
                        es_1 = pushdown_expr(fcx, pt, es);
                    }
                    case (_) {
                        log "chan expr doesn't have a chan type!";
                        fail;
                    }
                }
                e_1 = ast.expr_chan(es_1, triv_ann(t));
            }

            case (ast.expr_alt(?discrim, ?arms_0, ?ann)) {
                auto t = expected;
                let vec[ast.arm] arms_1 = vec();
                for (ast.arm arm_0 in arms_0) {
                    auto block_1 = pushdown_block(fcx, expected, arm_0.block);
                    t = Demand.simple(fcx, e.span, t,
                                      block_ty(fcx.ccx.tcx, block_1));
                    auto arm_1 = rec(pat=arm_0.pat, block=block_1,
                                     index=arm_0.index);
                    arms_1 += vec(arm_1);
                }
                e_1 = ast.expr_alt(discrim, arms_1, triv_ann(t));
            }

            case (ast.expr_recv(?lval_0, ?expr_0, ?ann)) {
                auto lval_1 = pushdown_expr(fcx, next_ty_var(fcx.ccx),
                                            lval_0);
                auto t = expr_ty(fcx.ccx.tcx, lval_1);
                auto expr_1 = pushdown_expr(fcx, ty.mk_port(fcx.ccx.tcx, t),
                                            expr_0);
                e_1 = ast.expr_recv(lval_1, expr_1, ann);
            }

            case (ast.expr_send(?lval_0, ?expr_0, ?ann)) {
                auto expr_1 = pushdown_expr(fcx, next_ty_var(fcx.ccx),
                                            expr_0);
                auto t = expr_ty(fcx.ccx.tcx, expr_1);
                auto lval_1 = pushdown_expr(fcx, ty.mk_chan(fcx.ccx.tcx, t),
                                            lval_0);
                e_1 = ast.expr_send(lval_1, expr_1, ann);
            }

            case (_) {
                fcx.ccx.sess.span_unimpl(e.span,
                    #fmt("type unification for expression variant: %s",
                         util.common.expr_to_str(e)));
                fail;
            }
        }

        ret @fold.respan[ast.expr_](e.span, e_1);
    }

    // Push-down over typed blocks.
    fn pushdown_block(&@fn_ctxt fcx, ty.t expected, &ast.block bloc)
            -> ast.block {
        alt (bloc.node.expr) {
            case (some[@ast.expr](?e_0)) {
                auto e_1 = pushdown_expr(fcx, expected, e_0);
                auto block_ = rec(stmts=bloc.node.stmts,
                                  expr=some[@ast.expr](e_1),
                                  index=bloc.node.index,
                                  a=plain_ann(fcx.ccx.tcx));
                ret fold.respan[ast.block_](bloc.span, block_);
            }
            case (none[@ast.expr]) {
                Demand.simple(fcx, bloc.span, expected,
                              ty.mk_nil(fcx.ccx.tcx));
                ret fold.respan[ast.block_](bloc.span,
                      rec(a = plain_ann(fcx.ccx.tcx) with bloc.node));
            }
        }
    }
}


// Local variable resolution: the phase that finds all the types in the AST
// and replaces opaque "ty_local" types with the resolved local types.

fn writeback_local(&Option.t[@fn_ctxt] env, &span sp, @ast.local local)
        -> @ast.decl {
    auto fcx = Option.get[@fn_ctxt](env);

    auto local_ty;
    alt (fcx.locals.find(local.id)) {
        case (none[ty.t]) {
            fcx.ccx.sess.span_err(sp, "unable to determine type of local: "
                                  + local.ident);
            fail;
        }
        case (some[ty.t](?lt)) {
            local_ty = lt;
        }
    }

    auto local_wb = @rec(ann=triv_ann(local_ty)
        with *local
    );
    ret @fold.respan[ast.decl_](sp, ast.decl_local(local_wb));
}

fn resolve_local_types_in_annotation(&Option.t[@fn_ctxt] env, ast.ann ann)
        -> ast.ann {
    fn resolver(@fn_ctxt fcx, ty.t typ) -> ty.t {
        alt (struct(fcx.ccx.tcx, typ)) {
            case (ty.ty_local(?lid)) { ret fcx.locals.get(lid); }
            case (_)                 { ret typ; }
        }
    }

    auto fcx = Option.get[@fn_ctxt](env);
    alt (ann) {
        case (ast.ann_none) {
            log "warning: no type for expression";
            ret ann;
        }
        case (ast.ann_type(?typ, ?tps, ?ts_info)) {
            auto tt = ann_to_type(ann);
            if (!ty.type_contains_locals(fcx.ccx.tcx, tt)) {
                ret ann;
            }
            auto f = bind resolver(fcx, _);
            auto new_type = ty.fold_ty(fcx.ccx.tcx, f, ann_to_type(ann));
            ret ast.ann_type(new_type, tps, ts_info);
        }
    }
}

fn resolve_local_types_in_block(&@fn_ctxt fcx, &ast.block block)
        -> ast.block {
    fn update_env_for_item(&Option.t[@fn_ctxt] env, @ast.item i)
            -> Option.t[@fn_ctxt] {
        ret none[@fn_ctxt];
    }
    fn keep_going(&Option.t[@fn_ctxt] env) -> bool {
        ret !Option.is_none[@fn_ctxt](env);
    }

    // FIXME: rustboot bug prevents us from using these functions directly
    auto fld = fold.new_identity_fold[Option.t[@fn_ctxt]]();
    auto wbl = writeback_local;
    auto rltia = bind resolve_local_types_in_annotation(_,_);
    auto uefi = update_env_for_item;
    auto kg = keep_going;
    fld = @rec(
        fold_decl_local = wbl,
        fold_ann = rltia,
        update_env_for_item = uefi,
        keep_going = kg
        with *fld
    );
    ret fold.fold_block[Option.t[@fn_ctxt]](some[@fn_ctxt](fcx), fld, block);
}

// AST fragment checking

fn check_lit(@crate_ctxt ccx, @ast.lit lit) -> ty.t {
    alt (lit.node) {
        case (ast.lit_str(_))           { ret ty.mk_str(ccx.tcx); }
        case (ast.lit_char(_))          { ret ty.mk_char(ccx.tcx); }
        case (ast.lit_int(_))           { ret ty.mk_int(ccx.tcx);  }
        case (ast.lit_float(_))         { ret ty.mk_float(ccx.tcx);  }
        case (ast.lit_mach_float(?tm, _))
                                        { ret ty.mk_mach(ccx.tcx, tm); }
        case (ast.lit_uint(_))          { ret ty.mk_uint(ccx.tcx); }
        case (ast.lit_mach_int(?tm, _)) { ret ty.mk_mach(ccx.tcx, tm); }
        case (ast.lit_nil)              { ret ty.mk_nil(ccx.tcx);  }
        case (ast.lit_bool(_))          { ret ty.mk_bool(ccx.tcx); }
    }

    fail; // not reached
}

fn check_pat(&@fn_ctxt fcx, @ast.pat pat) -> @ast.pat {
    auto new_pat;
    alt (pat.node) {
        case (ast.pat_wild(_)) {
            new_pat = ast.pat_wild(triv_ann(next_ty_var(fcx.ccx)));
        }
        case (ast.pat_lit(?lt, _)) {
            new_pat = ast.pat_lit(lt, triv_ann(check_lit(fcx.ccx, lt)));
        }
        case (ast.pat_bind(?id, ?def_id, _)) {
            auto ann = triv_ann(next_ty_var(fcx.ccx));
            new_pat = ast.pat_bind(id, def_id, ann);
        }
        case (ast.pat_tag(?p, ?subpats, ?vdef_opt, _)) {
            auto vdef = Option.get[ast.variant_def](vdef_opt);
            auto t = ty.lookup_item_type(fcx.ccx.sess, fcx.ccx.tcx,
                                         fcx.ccx.type_cache, vdef._1)._1;
            auto len = Vec.len[ast.ident](p.node.idents);
            auto last_id = p.node.idents.(len - 1u);

            auto tpt = ty.lookup_item_type(fcx.ccx.sess, fcx.ccx.tcx,
                                           fcx.ccx.type_cache, vdef._0);
            auto ann = instantiate_path(fcx, p, tpt, pat.span);

            alt (struct(fcx.ccx.tcx, t)) {
                // N-ary variants have function types.
                case (ty.ty_fn(_, ?args, ?tag_ty)) {
                    auto arg_len = Vec.len[arg](args);
                    auto subpats_len = Vec.len[@ast.pat](subpats);
                    if (arg_len != subpats_len) {
                        // TODO: pluralize properly
                        auto err_msg = "tag type " + last_id + " has " +
                                       UInt.to_str(subpats_len, 10u) +
                                       " field(s), but this pattern has " +
                                       UInt.to_str(arg_len, 10u) +
                                       " field(s)";

                        fcx.ccx.sess.span_err(pat.span, err_msg);
                        fail;   // TODO: recover
                    }

                    let vec[@ast.pat] new_subpats = vec();
                    for (@ast.pat subpat in subpats) {
                        new_subpats += vec(check_pat(fcx, subpat));
                    }

                    new_pat = ast.pat_tag(p, new_subpats, vdef_opt, ann);
                }

                // Nullary variants have tag types.
                case (ty.ty_tag(?tid, _)) {
                    auto subpats_len = Vec.len[@ast.pat](subpats);
                    if (subpats_len > 0u) {
                        // TODO: pluralize properly
                        auto err_msg = "tag type " + last_id +
                                       " has no field(s)," +
                                       " but this pattern has " +
                                       UInt.to_str(subpats_len, 10u) +
                                       " field(s)";

                        fcx.ccx.sess.span_err(pat.span, err_msg);
                        fail;   // TODO: recover
                    }

                    new_pat = ast.pat_tag(p, subpats, vdef_opt, ann);
                }
            }
        }
    }

    ret @fold.respan[ast.pat_](pat.span, new_pat);
}

fn require_impure(&session.session sess,
      &ast.purity f_purity, &span sp) -> () {
    alt (f_purity) {
        case (ast.impure_fn) {
            ret;
        }
        case (ast.pure_fn) {
            sess.span_err(sp,
               "Found impure expression in pure function decl");
        }
    }
}

fn get_function_purity(@crate_ctxt ccx, &ast.def_id d_id) -> ast.purity {
    let Option.t[ast.purity] o = ccx.fn_purity_table.find(d_id);
    ret from_maybe[ast.purity](ast.impure_fn, o);
}

fn require_pure_call(@crate_ctxt ccx,
    &ast.purity caller_purity, @ast.expr callee, &span sp) -> () {
    alt (caller_purity) {
        case (ast.impure_fn) {
            ret;
        }
        case (ast.pure_fn) {
            alt (callee.node) {
                case (ast.expr_path(_, some[ast.def](ast.def_fn(?d_id)), _)) {
                    alt (get_function_purity(ccx, d_id)) {
                            case (ast.pure_fn) {
                                ret;
                            }
                            case (_) {
                                ccx.sess.span_err(sp,
                                  "Pure function calls impure function");

                            }
                        }
                }
                case (_) {
                    ccx.sess.span_err(sp,
                      "Pure function calls unknown function");
                }
            }
        }
    }
}

fn require_pure_function(@crate_ctxt ccx, &ast.def_id d_id, &span sp) -> () {
    alt (get_function_purity(ccx, d_id)) {
        case (ast.impure_fn) {
            ccx.sess.span_err(sp, "Found non-predicate in check expression");
        }
        case (_) { ret; }
    }
}

fn check_expr(&@fn_ctxt fcx, @ast.expr expr) -> @ast.expr {
    //fcx.ccx.sess.span_warn(expr.span, "typechecking expr " +
    //                       util.common.expr_to_str(expr));

    // A generic function to factor out common logic from call and bind
    // expressions.
    fn check_call_or_bind(&@fn_ctxt fcx, &@ast.expr f,
                          &vec[Option.t[@ast.expr]] args)
            -> tup(@ast.expr, vec[Option.t[@ast.expr]]) {

        // Check the function.
        auto f_0 = check_expr(fcx, f);

        // Check the arguments and generate the argument signature.
        let vec[Option.t[@ast.expr]] args_0 = vec();
        let vec[arg] arg_tys_0 = vec();
        for (Option.t[@ast.expr] a_opt in args) {
            alt (a_opt) {
                case (some[@ast.expr](?a)) {
                    auto a_0 = check_expr(fcx, a);
                    args_0 += vec(some[@ast.expr](a_0));

                    // FIXME: this breaks aliases. We need a ty_fn_arg.
                    auto arg_ty = rec(mode=mo_val,
                                      ty=expr_ty(fcx.ccx.tcx, a_0));
                    Vec.push[arg](arg_tys_0, arg_ty);
                }
                case (none[@ast.expr]) {
                    args_0 += vec(none[@ast.expr]);

                    // FIXME: breaks aliases too?
                    auto typ = next_ty_var(fcx.ccx);
                    Vec.push[arg](arg_tys_0, rec(mode=mo_val, ty=typ));
                }
            }
        }

        auto rt_0 = next_ty_var(fcx.ccx);
        auto t_0;
        alt (struct(fcx.ccx.tcx, expr_ty(fcx.ccx.tcx, f_0))) {
            case (ty.ty_fn(?proto, _, _))   {
                t_0 = ty.mk_fn(fcx.ccx.tcx, proto, arg_tys_0, rt_0);
            }
            case (ty.ty_native_fn(?abi, _, _))   {
                t_0 = ty.mk_native_fn(fcx.ccx.tcx, abi, arg_tys_0, rt_0);
            }
            case (_) {
                log_err "check_call_or_bind(): fn expr doesn't have fn type";
                fail;
            }
        }

        // Unify the callee and arguments.
        auto tpt_0 = ty.expr_ty_params_and_ty(fcx.ccx.tcx, f_0);
        auto tpt_1 = Demand.full(fcx, f.span, tpt_0._1, t_0, tpt_0._0,
                                 NO_AUTODEREF);
        auto f_1 = ty.replace_expr_type(f_0, tpt_1);

        ret tup(f_1, args_0);
    }

    // A generic function for checking assignment expressions
    fn check_assignment(&@fn_ctxt fcx, @ast.expr lhs, @ast.expr rhs)
        -> tup(@ast.expr, @ast.expr, ast.ann) {
        auto lhs_0 = check_expr(fcx, lhs);
        auto rhs_0 = check_expr(fcx, rhs);
        auto lhs_t0 = expr_ty(fcx.ccx.tcx, lhs_0);
        auto rhs_t0 = expr_ty(fcx.ccx.tcx, rhs_0);

        auto lhs_1 = Pushdown.pushdown_expr(fcx, rhs_t0, lhs_0);
        auto rhs_1 = Pushdown.pushdown_expr(fcx, expr_ty(fcx.ccx.tcx, lhs_1),
                                            rhs_0);

        auto ann = triv_ann(expr_ty(fcx.ccx.tcx, rhs_1));
        ret tup(lhs_1, rhs_1, ann);
    }

    // A generic function for checking call expressions
    fn check_call(&@fn_ctxt fcx, @ast.expr f, vec[@ast.expr] args)
        -> tup(@ast.expr, vec[@ast.expr]) {

        let vec[Option.t[@ast.expr]] args_opt_0 = vec();
        for (@ast.expr arg in args) {
            args_opt_0 += vec(some[@ast.expr](arg));
        }

        // Call the generic checker.
        auto result = check_call_or_bind(fcx, f, args_opt_0);

        // Pull out the arguments.
        let vec[@ast.expr] args_1 = vec();
        for (Option.t[@ast.expr] arg in result._1) {
            args_1 += vec(Option.get[@ast.expr](arg));
        }

        ret tup(result._0, args_1);
    }

    alt (expr.node) {
        case (ast.expr_lit(?lit, _)) {
            auto typ = check_lit(fcx.ccx, lit);
            auto ann = triv_ann(typ);
            ret @fold.respan[ast.expr_](expr.span, ast.expr_lit(lit, ann));
        }


        case (ast.expr_binary(?binop, ?lhs, ?rhs, _)) {
            auto lhs_0 = check_expr(fcx, lhs);
            auto rhs_0 = check_expr(fcx, rhs);
            auto lhs_t0 = expr_ty(fcx.ccx.tcx, lhs_0);
            auto rhs_t0 = expr_ty(fcx.ccx.tcx, rhs_0);

            // FIXME: Binops have a bit more subtlety than this.
            auto lhs_1 = Pushdown.pushdown_expr_full(fcx, rhs_t0, lhs_0,
                                                     AUTODEREF_OK);
            auto rhs_1 =
                Pushdown.pushdown_expr_full(fcx,
                                            expr_ty(fcx.ccx.tcx, lhs_1),
                                            rhs_0, AUTODEREF_OK);

            auto t = strip_boxes(fcx.ccx.tcx, lhs_t0);
            alt (binop) {
                case (ast.eq) { t = ty.mk_bool(fcx.ccx.tcx); }
                case (ast.lt) { t = ty.mk_bool(fcx.ccx.tcx); }
                case (ast.le) { t = ty.mk_bool(fcx.ccx.tcx); }
                case (ast.ne) { t = ty.mk_bool(fcx.ccx.tcx); }
                case (ast.ge) { t = ty.mk_bool(fcx.ccx.tcx); }
                case (ast.gt) { t = ty.mk_bool(fcx.ccx.tcx); }
                case (_) { /* fall through */ }
            }

            auto ann = triv_ann(t);
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_binary(binop, lhs_1, rhs_1,
                                                        ann));
        }


        case (ast.expr_unary(?unop, ?oper, _)) {
            auto oper_1 = check_expr(fcx, oper);
            auto oper_t = expr_ty(fcx.ccx.tcx, oper_1);
            alt (unop) {
                case (ast.box(?mut)) {
                    oper_t = ty.mk_box(fcx.ccx.tcx,
                                       rec(ty=oper_t, mut=mut));
                }
                case (ast.deref) {
                    alt (struct(fcx.ccx.tcx, oper_t)) {
                        case (ty.ty_box(?inner)) {
                            oper_t = inner.ty;
                        }
                        case (_) {
                            fcx.ccx.sess.span_err
                                (expr.span,
                                 "dereferencing non-box type: "
                                 + ty_to_str(fcx.ccx.tcx, oper_t));
                        }
                    }
                }
                case (_) { oper_t = strip_boxes(fcx.ccx.tcx, oper_t); }
            }

            auto ann = triv_ann(oper_t);
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_unary(unop, oper_1, ann));
        }

        case (ast.expr_path(?pth, ?defopt, _)) {
            auto t = ty.mk_nil(fcx.ccx.tcx);
            assert (defopt != none[ast.def]);
            auto defn = Option.get[ast.def](defopt);

            auto tpt = ty_param_count_and_ty_for_def(fcx, expr.span, defn);

            if (ty.def_has_ty_params(defn)) {
                auto ann = instantiate_path(fcx, pth, tpt, expr.span);
                ret @fold.respan[ast.expr_](expr.span,
                                            ast.expr_path(pth, defopt, ann));
            }

            // The definition doesn't take type parameters. If the programmer
            // supplied some, that's an error.
            if (Vec.len[@ast.ty](pth.node.types) > 0u) {
                fcx.ccx.sess.span_err(expr.span, "this kind of value does " +
                                      "not take type parameters");
                fail;
            }

            auto e = ast.expr_path(pth, defopt, triv_ann(tpt._1));
            ret @fold.respan[ast.expr_](expr.span, e);
        }

        case (ast.expr_ext(?p, ?args, ?body, ?expanded, _)) {
            auto exp_ = check_expr(fcx, expanded);
            auto t = expr_ty(fcx.ccx.tcx, exp_);
            auto ann = triv_ann(t);
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_ext(p, args, body, exp_,
                                                     ann));
        }

        case (ast.expr_fail(_)) {
            ret @fold.respan[ast.expr_](expr.span,
                ast.expr_fail(plain_ann(fcx.ccx.tcx)));
        }

        case (ast.expr_break(_)) {
            ret @fold.respan[ast.expr_](expr.span,
                ast.expr_break(plain_ann(fcx.ccx.tcx)));
        }

        case (ast.expr_cont(_)) {
            ret @fold.respan[ast.expr_](expr.span,
                ast.expr_cont(plain_ann(fcx.ccx.tcx)));
        }

        case (ast.expr_ret(?expr_opt, _)) {
            alt (expr_opt) {
                case (none[@ast.expr]) {
                    auto nil = ty.mk_nil(fcx.ccx.tcx);
                    if (!are_compatible(fcx, fcx.ret_ty, nil)) {
                        fcx.ccx.sess.err("ret; in function "
                                         + "returning non-nil");
                    }

                    ret @fold.respan[ast.expr_]
                        (expr.span,
                         ast.expr_ret(none[@ast.expr],
                                      plain_ann(fcx.ccx.tcx)));
                }

                case (some[@ast.expr](?e)) {
                    auto expr_0 = check_expr(fcx, e);
                    auto expr_1 = Pushdown.pushdown_expr(fcx, fcx.ret_ty,
                                                         expr_0);
                    ret @fold.respan[ast.expr_]
                        (expr.span, ast.expr_ret(some(expr_1),
                                                 plain_ann(fcx.ccx.tcx)));
                }
            }
        }

        case (ast.expr_put(?expr_opt, _)) {
            require_impure(fcx.ccx.sess, fcx.purity, expr.span);

            alt (expr_opt) {
                case (none[@ast.expr]) {
                    auto nil = ty.mk_nil(fcx.ccx.tcx);
                    if (!are_compatible(fcx, fcx.ret_ty, nil)) {
                        fcx.ccx.sess.err("put; in function "
                                         + "putting non-nil");
                    }

                    ret @fold.respan[ast.expr_]
                        (expr.span, ast.expr_put(none[@ast.expr],
                         plain_ann(fcx.ccx.tcx)));
                }

                case (some[@ast.expr](?e)) {
                    auto expr_0 = check_expr(fcx, e);
                    auto expr_1 = Pushdown.pushdown_expr(fcx, fcx.ret_ty,
                                                         expr_0);
                    ret @fold.respan[ast.expr_]
                        (expr.span, ast.expr_put(some(expr_1),
                                                 plain_ann(fcx.ccx.tcx)));
                }
            }
        }

        case (ast.expr_be(?e, _)) {
            /* FIXME: prove instead of check */
            assert (ast.is_call_expr(e));
            auto expr_0 = check_expr(fcx, e);
            auto expr_1 = Pushdown.pushdown_expr(fcx, fcx.ret_ty, expr_0);
            ret @fold.respan[ast.expr_](expr.span,
                ast.expr_be(expr_1, plain_ann(fcx.ccx.tcx)));
        }

        case (ast.expr_log(?l,?e,_)) {
            auto expr_t = check_expr(fcx, e);
            ret @fold.respan[ast.expr_]
                (expr.span, ast.expr_log(l, expr_t,
                                         plain_ann(fcx.ccx.tcx)));
        }

        case (ast.expr_check(?e, _)) {
            auto expr_t = check_expr(fcx, e);
            Demand.simple(fcx, expr.span, ty.mk_bool(fcx.ccx.tcx),
                          expr_ty(fcx.ccx.tcx, expr_t));
            /* e must be a call expr where all arguments are either
             literals or slots */
            alt (e.node) {
                case (ast.expr_call(?operator, ?operands, _)) {
                    alt (operator.node) {
                        case (ast.expr_path(?oper_name,
                                some[ast.def](ast.def_fn(?d_id)), _)) {

                            for (@ast.expr operand in operands) {
                                if (! ast.is_constraint_arg(operand)) {
                                    fcx.ccx.sess.span_err(expr.span,
                                       "Constraint args must be "
                                     + "slot variables or literals");
                                }
                            }
                            
                            require_pure_function(fcx.ccx, d_id, expr.span);

                            ret @fold.respan[ast.expr_]
                                (expr.span, ast.expr_check(expr_t,
                                   plain_ann(fcx.ccx.tcx)));
                        }
                        case (_) {
                           fcx.ccx.sess.span_err(expr.span,
                             "In a constraint, expected the constraint name "
                           + "to be an explicit name");
                        }
                    }
                }
                case (_) {
                    fcx.ccx.sess.span_err(expr.span,
                        "Check on non-predicate");
                }
            }
        }

        case (ast.expr_assert(?e, _)) {
            auto expr_t = check_expr(fcx, e);
            Demand.simple(fcx, expr.span, ty.mk_bool(fcx.ccx.tcx),
                          expr_ty(fcx.ccx.tcx, expr_t));
            ret @fold.respan[ast.expr_]
                (expr.span, ast.expr_assert(expr_t,
                                                plain_ann(fcx.ccx.tcx)));
        }

        case (ast.expr_assign(?lhs, ?rhs, _)) {
            require_impure(fcx.ccx.sess, fcx.purity, expr.span);

            auto checked = check_assignment(fcx, lhs, rhs);
            auto newexpr = ast.expr_assign(checked._0,
                                           checked._1,
                                           checked._2);
            ret @fold.respan[ast.expr_](expr.span, newexpr);
        }

        case (ast.expr_assign_op(?op, ?lhs, ?rhs, _)) {
            require_impure(fcx.ccx.sess, fcx.purity, expr.span);

            auto checked = check_assignment(fcx, lhs, rhs);
            auto newexpr = ast.expr_assign_op(op,
                                              checked._0,
                                              checked._1,
                                              checked._2);
            ret @fold.respan[ast.expr_](expr.span, newexpr);
        }

        case (ast.expr_send(?lhs, ?rhs, _)) {
            require_impure(fcx.ccx.sess, fcx.purity, expr.span);

            auto lhs_0 = check_expr(fcx, lhs);
            auto rhs_0 = check_expr(fcx, rhs);
            auto rhs_t = expr_ty(fcx.ccx.tcx, rhs_0);

            auto chan_t = ty.mk_chan(fcx.ccx.tcx, rhs_t);
            auto lhs_1 = Pushdown.pushdown_expr(fcx, chan_t, lhs_0);
            auto item_t;
            alt (struct(fcx.ccx.tcx, expr_ty(fcx.ccx.tcx, lhs_1))) {
                case (ty.ty_chan(?it)) {
                    item_t = it;
                }
                case (_) {
                    fail;
                }
            }
            auto rhs_1 = Pushdown.pushdown_expr(fcx, item_t, rhs_0);

            auto ann = triv_ann(chan_t);
            auto newexpr = ast.expr_send(lhs_1, rhs_1, ann);
            ret @fold.respan[ast.expr_](expr.span, newexpr);
        }

        case (ast.expr_recv(?lhs, ?rhs, _)) {
            require_impure(fcx.ccx.sess, fcx.purity, expr.span);

            auto lhs_0 = check_expr(fcx, lhs);
            auto rhs_0 = check_expr(fcx, rhs);
            auto lhs_t1 = expr_ty(fcx.ccx.tcx, lhs_0);

            auto port_t = ty.mk_port(fcx.ccx.tcx, lhs_t1);
            auto rhs_1 = Pushdown.pushdown_expr(fcx, port_t, rhs_0);
            auto item_t;
            alt (struct(fcx.ccx.tcx, expr_ty(fcx.ccx.tcx, rhs_0))) {
                case (ty.ty_port(?it)) {
                    item_t = it;
                }
                case (_) {
                    fail;
                }
            }
            auto lhs_1 = Pushdown.pushdown_expr(fcx, item_t, lhs_0);

            auto ann = triv_ann(item_t);
            auto newexpr = ast.expr_recv(lhs_1, rhs_1, ann);
            ret @fold.respan[ast.expr_](expr.span, newexpr);
        }

        case (ast.expr_if(?cond, ?thn, ?elsopt, _)) {
            auto cond_0 = check_expr(fcx, cond);
            auto cond_1 = Pushdown.pushdown_expr(fcx, ty.mk_bool(fcx.ccx.tcx),
                                                 cond_0);

            auto thn_0 = check_block(fcx, thn);
            auto thn_t = block_ty(fcx.ccx.tcx, thn_0);

            auto elsopt_1;
            auto elsopt_t;
            alt (elsopt) {
                case (some[@ast.expr](?els)) {
                    auto els_0 = check_expr(fcx, els);
                    auto els_1 = Pushdown.pushdown_expr(fcx, thn_t, els_0);
                    elsopt_1 = some[@ast.expr](els_1);
                    elsopt_t = expr_ty(fcx.ccx.tcx, els_1);
                }
                case (none[@ast.expr]) {
                    elsopt_1 = none[@ast.expr];
                    elsopt_t = ty.mk_nil(fcx.ccx.tcx);
                }
            }

            auto thn_1 = Pushdown.pushdown_block(fcx, elsopt_t, thn_0);

            auto ann = triv_ann(elsopt_t);
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_if(cond_1, thn_1,
                                                    elsopt_1, ann));
        }

        case (ast.expr_for(?decl, ?seq, ?body, _)) {
            auto decl_1 = check_decl_local(fcx, decl);
            auto seq_1 = check_expr(fcx, seq);
            auto body_1 = check_block(fcx, body);

            // FIXME: enforce that the type of the decl is the element type
            // of the seq.

            auto ann = triv_ann(ty.mk_nil(fcx.ccx.tcx));
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_for(decl_1, seq_1,
                                                     body_1, ann));
        }

        case (ast.expr_for_each(?decl, ?seq, ?body, _)) {
            auto decl_1 = check_decl_local(fcx, decl);
            auto seq_1 = check_expr(fcx, seq);
            auto body_1 = check_block(fcx, body);

            auto ann = triv_ann(ty.mk_nil(fcx.ccx.tcx));
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_for_each(decl_1, seq_1,
                                                          body_1, ann));
        }

        case (ast.expr_while(?cond, ?body, _)) {
            auto cond_0 = check_expr(fcx, cond);
            auto cond_1 = Pushdown.pushdown_expr(fcx, ty.mk_bool(fcx.ccx.tcx),
                                                 cond_0);
            auto body_1 = check_block(fcx, body);

            auto ann = triv_ann(ty.mk_nil(fcx.ccx.tcx));
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_while(cond_1, body_1, ann));
        }

        case (ast.expr_do_while(?body, ?cond, _)) {
            auto cond_0 = check_expr(fcx, cond);
            auto cond_1 = Pushdown.pushdown_expr(fcx, ty.mk_bool(fcx.ccx.tcx),
                                                 cond_0);
            auto body_1 = check_block(fcx, body);

            auto ann = triv_ann(block_ty(fcx.ccx.tcx, body_1));
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_do_while(body_1, cond_1,
                                                          ann));
        }

        case (ast.expr_alt(?expr, ?arms, _)) {
            auto expr_0 = check_expr(fcx, expr);

            // Typecheck the patterns first, so that we get types for all the
            // bindings.
            auto pattern_ty = expr_ty(fcx.ccx.tcx, expr_0);

            let vec[@ast.pat] pats_0 = vec();
            for (ast.arm arm in arms) {
                auto pat_0 = check_pat(fcx, arm.pat);
                pattern_ty = Demand.simple(fcx, pat_0.span, pattern_ty,
                                           pat_ty(fcx.ccx.tcx, pat_0));
                pats_0 += vec(pat_0);
            }

            let vec[@ast.pat] pats_1 = vec();
            for (@ast.pat pat_0 in pats_0) {
                pats_1 += vec(Pushdown.pushdown_pat(fcx, pattern_ty, pat_0));
            }

            // Now typecheck the blocks.
            auto result_ty = next_ty_var(fcx.ccx);

            let vec[ast.block] blocks_0 = vec();
            for (ast.arm arm in arms) {
                auto block_0 = check_block(fcx, arm.block);
                result_ty = Demand.simple(fcx, block_0.span, result_ty,
                                          block_ty(fcx.ccx.tcx, block_0));
                blocks_0 += vec(block_0);
            }

            let vec[ast.arm] arms_1 = vec();
            auto i = 0u;
            for (ast.block block_0 in blocks_0) {
                auto block_1 = Pushdown.pushdown_block(fcx, result_ty,
                                                       block_0);
                auto pat_1 = pats_1.(i);
                auto arm = arms.(i);
                auto arm_1 = rec(pat=pat_1, block=block_1, index=arm.index);
                arms_1 += vec(arm_1);
                i += 1u;
            }

            auto expr_1 = Pushdown.pushdown_expr(fcx, pattern_ty, expr_0);

            auto ann = triv_ann(result_ty);
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_alt(expr_1, arms_1, ann));
        }

        case (ast.expr_block(?b, _)) {
            auto b_0 = check_block(fcx, b);
            auto ann;
            alt (b_0.node.expr) {
                case (some[@ast.expr](?expr)) {
                    ann = triv_ann(expr_ty(fcx.ccx.tcx, expr));
                }
                case (none[@ast.expr]) {
                    ann = triv_ann(ty.mk_nil(fcx.ccx.tcx));
                }
            }
            ret @fold.respan[ast.expr_](expr.span, ast.expr_block(b_0, ann));
        }

        case (ast.expr_bind(?f, ?args, _)) {
            // Call the generic checker.
            auto result = check_call_or_bind(fcx, f, args);

            // Pull the argument and return types out.
            auto proto_1;
            let vec[ty.arg] arg_tys_1 = vec();
            auto rt_1;
            alt (struct(fcx.ccx.tcx, expr_ty(fcx.ccx.tcx, result._0))) {
                case (ty.ty_fn(?proto, ?arg_tys, ?rt)) {
                    proto_1 = proto;
                    rt_1 = rt;

                    // For each blank argument, add the type of that argument
                    // to the resulting function type.
                    auto i = 0u;
                    while (i < Vec.len[Option.t[@ast.expr]](args)) {
                        alt (args.(i)) {
                            case (some[@ast.expr](_)) { /* no-op */ }
                            case (none[@ast.expr]) {
                                arg_tys_1 += vec(arg_tys.(i));
                            }
                        }
                        i += 1u;
                    }
                }
                case (_) {
                    log_err "LHS of bind expr didn't have a function type?!";
                    fail;
                }
            }

            auto t_1 = ty.mk_fn(fcx.ccx.tcx, proto_1, arg_tys_1, rt_1);
            auto ann = triv_ann(t_1);
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_bind(result._0, result._1,
                                                      ann));
        }

        case (ast.expr_call(?f, ?args, _)) {
            /* here we're kind of hosed, as f can be any expr
             need to restrict it to being an explicit expr_path if we're
            inside a pure function, and need an environment mapping from 
            function name onto purity-designation */
            require_pure_call(fcx.ccx, fcx.purity, f, expr.span);

            auto result = check_call(fcx, f, args);
            auto f_1 = result._0;
            auto args_1 = result._1;

            // Pull the return type out of the type of the function.
            auto rt_1 = ty.mk_nil(fcx.ccx.tcx);  // FIXME: typestate botch
            alt (struct(fcx.ccx.tcx, expr_ty(fcx.ccx.tcx, f_1))) {
                case (ty.ty_fn(_,_,?rt))    { rt_1 = rt; }
                case (ty.ty_native_fn(_, _, ?rt))    { rt_1 = rt; }
                case (_) {
                    log_err "LHS of call expr didn't have a function type?!";
                    fail;
                }
            }

            auto ann = triv_ann(rt_1);
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_call(f_1, args_1, ann));
        }

        case (ast.expr_self_method(?id, _)) {
            auto t = ty.mk_nil(fcx.ccx.tcx);
            let ty.t this_obj_ty;

            // Grab the type of the current object
            auto this_obj_id = fcx.ccx.this_obj;
            alt (this_obj_id) {
                case (some[ast.def_id](?def_id)) {
                    this_obj_ty = ty.lookup_item_type(fcx.ccx.sess,
                        fcx.ccx.tcx, fcx.ccx.type_cache, def_id)._1;
                }
                case (_) { fail; }
            }


            // Grab this method's type out of the current object type

            // this_obj_ty is an ty.t
            alt (struct(fcx.ccx.tcx, this_obj_ty)) {
                case (ty.ty_obj(?methods)) {
                    for (ty.method method in methods) {
                        if (method.ident == id) {
                            t = ty.method_ty_to_fn_ty(fcx.ccx.tcx,
                                                      method);
                        }
                    }
                }
                case (_) { fail; }
            }

            auto ann = triv_ann(t);

            require_impure(fcx.ccx.sess, fcx.purity, expr.span);

            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_self_method(id, ann));
        }

        case (ast.expr_spawn(?dom, ?name, ?f, ?args, _)) {
            auto result = check_call(fcx, f, args);
            auto f_1 = result._0;
            auto args_1 = result._1;

            // Check the return type
            alt (struct(fcx.ccx.tcx, expr_ty(fcx.ccx.tcx, f_1))) {
                case (ty.ty_fn(_,_,?rt)) {
                    alt (struct(fcx.ccx.tcx, rt)) {
                        case (ty.ty_nil) {
                            // This is acceptable
                        }
                        case (_) {
                            auto err = "non-nil return type in "
                                + "spawned function";
                            fcx.ccx.sess.span_err(expr.span, err);
                            fail;
                        }
                    }
                }
            }

            // FIXME: Other typechecks needed

            auto ann = triv_ann(ty.mk_task(fcx.ccx.tcx));
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_spawn(dom, name,
                                                       f_1, args_1, ann));
        }

        case (ast.expr_cast(?e, ?t, _)) {
            auto e_1 = check_expr(fcx, e);
            auto t_1 = ast_ty_to_ty_crate(fcx.ccx, t);
            // FIXME: there are more forms of cast to support, eventually.
            if (! (type_is_scalar(fcx.ccx.tcx, expr_ty(fcx.ccx.tcx, e_1)) &&
                   type_is_scalar(fcx.ccx.tcx, t_1))) {
                fcx.ccx.sess.span_err(expr.span,
                    "non-scalar cast: " +
                    ty_to_str(fcx.ccx.tcx, expr_ty(fcx.ccx.tcx, e_1)) +
                    " as " + ty_to_str(fcx.ccx.tcx, t_1));
            }

            auto ann = triv_ann(t_1);
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_cast(e_1, t, ann));
        }

        case (ast.expr_vec(?args, ?mut, _)) {
            let vec[@ast.expr] args_1 = vec();

            let ty.t t;
            if (Vec.len[@ast.expr](args) == 0u) {
                t = next_ty_var(fcx.ccx);
            } else {
                auto expr_1 = check_expr(fcx, args.(0));
                t = expr_ty(fcx.ccx.tcx, expr_1);
            }

            for (@ast.expr e in args) {
                auto expr_1 = check_expr(fcx, e);
                auto expr_t = expr_ty(fcx.ccx.tcx, expr_1);
                Demand.simple(fcx, expr.span, t, expr_t);
                Vec.push[@ast.expr](args_1,expr_1);
            }

            auto ann = triv_ann(ty.mk_vec(fcx.ccx.tcx,
                                          rec(ty=t, mut=mut)));
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_vec(args_1, mut, ann));
        }

        case (ast.expr_tup(?elts, _)) {
            let vec[ast.elt] elts_1 = vec();
            let vec[ty.mt] elts_mt = vec();

            for (ast.elt e in elts) {
                auto expr_1 = check_expr(fcx, e.expr);
                auto expr_t = expr_ty(fcx.ccx.tcx, expr_1);
                Vec.push[ast.elt](elts_1, rec(expr=expr_1 with e));
                elts_mt += vec(rec(ty=expr_t, mut=e.mut));
            }

            auto ann = triv_ann(ty.mk_tup(fcx.ccx.tcx, elts_mt));
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_tup(elts_1, ann));
        }

        case (ast.expr_rec(?fields, ?base, _)) {

            auto base_1;
            alt (base) {
                case (none[@ast.expr]) { base_1 = none[@ast.expr]; }
                case (some[@ast.expr](?b_0)) {
                    base_1 = some[@ast.expr](check_expr(fcx, b_0));
                }
            }

            let vec[ast.field] fields_1 = vec();
            let vec[field] fields_t = vec();

            for (ast.field f in fields) {
                auto expr_1 = check_expr(fcx, f.expr);
                auto expr_t = expr_ty(fcx.ccx.tcx, expr_1);
                Vec.push[ast.field](fields_1, rec(expr=expr_1 with f));

                auto expr_mt = rec(ty=expr_t, mut=f.mut);
                Vec.push[field](fields_t, rec(ident=f.ident, mt=expr_mt));
            }

            auto ann = ast.ann_none;

            alt (base) {
                case (none[@ast.expr]) {
                    ann = triv_ann(ty.mk_rec(fcx.ccx.tcx, fields_t));
                }

                case (some[@ast.expr](?bexpr)) {
                    auto bexpr_1 = check_expr(fcx, bexpr);
                    auto bexpr_t = expr_ty(fcx.ccx.tcx, bexpr_1);

                    let vec[field] base_fields = vec();

                    alt (struct(fcx.ccx.tcx, bexpr_t)) {
                        case (ty.ty_rec(?flds)) {
                            base_fields = flds;
                        }
                        case (_) {
                            fcx.ccx.sess.span_err
                                (expr.span,
                                 "record update non-record base");
                        }
                    }

                    ann = triv_ann(bexpr_t);

                    for (ty.field f in fields_t) {
                        auto found = false;
                        for (ty.field bf in base_fields) {
                            if (Str.eq(f.ident, bf.ident)) {
                                Demand.simple(fcx, expr.span, f.mt.ty,
                                              bf.mt.ty);
                                found = true;
                            }
                        }
                        if (!found) {
                            fcx.ccx.sess.span_err
                                (expr.span,
                                 "unknown field in record update: "
                                 + f.ident);
                        }
                    }
                }
            }

            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_rec(fields_1, base_1, ann));
        }

        case (ast.expr_field(?base, ?field, _)) {
            auto base_1 = check_expr(fcx, base);
            auto base_t = strip_boxes(fcx.ccx.tcx,
                                      expr_ty(fcx.ccx.tcx, base_1));
            alt (struct(fcx.ccx.tcx, base_t)) {
                case (ty.ty_tup(?args)) {
                    let uint ix = ty.field_num(fcx.ccx.sess,
                                               expr.span, field);
                    if (ix >= Vec.len[ty.mt](args)) {
                        fcx.ccx.sess.span_err(expr.span,
                                              "bad index on tuple");
                    }
                    auto ann = triv_ann(args.(ix).ty);
                    ret @fold.respan[ast.expr_](expr.span,
                                                ast.expr_field(base_1,
                                                               field,
                                                               ann));
                }

                case (ty.ty_rec(?fields)) {
                    let uint ix = ty.field_idx(fcx.ccx.sess,
                                               expr.span, field, fields);
                    if (ix >= Vec.len[typeck.field](fields)) {
                        fcx.ccx.sess.span_err(expr.span,
                                              "bad index on record");
                    }
                    auto ann = triv_ann(fields.(ix).mt.ty);
                    ret @fold.respan[ast.expr_](expr.span,
                                                ast.expr_field(base_1,
                                                               field,
                                                               ann));
                }

                case (ty.ty_obj(?methods)) {
                    let uint ix = ty.method_idx(fcx.ccx.sess,
                                                expr.span, field, methods);
                    if (ix >= Vec.len[typeck.method](methods)) {
                        fcx.ccx.sess.span_err(expr.span,
                                              "bad index on obj");
                    }
                    auto meth = methods.(ix);
                    auto t = ty.mk_fn(fcx.ccx.tcx, meth.proto,
                                      meth.inputs, meth.output);
                    auto ann = triv_ann(t);
                    ret @fold.respan[ast.expr_](expr.span,
                                                ast.expr_field(base_1,
                                                               field,
                                                               ann));
                }

                case (_) {
                    fcx.ccx.sess.span_unimpl(expr.span,
                        "base type for expr_field in typeck.check_expr: " +
                        ty_to_str(fcx.ccx.tcx, base_t));
                }
            }
        }

        case (ast.expr_index(?base, ?idx, _)) {
            auto base_1 = check_expr(fcx, base);
            auto base_t = strip_boxes(fcx.ccx.tcx,
                                      expr_ty(fcx.ccx.tcx, base_1));

            auto idx_1 = check_expr(fcx, idx);
            auto idx_t = expr_ty(fcx.ccx.tcx, idx_1); 
            alt (struct(fcx.ccx.tcx, base_t)) {
                case (ty.ty_vec(?mt)) {
                    if (! type_is_integral(fcx.ccx.tcx, idx_t)) {
                        fcx.ccx.sess.span_err
                            (idx.span,
                             "non-integral type of vec index: "
                             + ty_to_str(fcx.ccx.tcx, idx_t));
                    }
                    auto ann = triv_ann(mt.ty);
                    ret @fold.respan[ast.expr_](expr.span,
                                                ast.expr_index(base_1,
                                                               idx_1,
                                                               ann));
                }
                case (ty.ty_str) {
                    if (! type_is_integral(fcx.ccx.tcx, idx_t)) {
                        fcx.ccx.sess.span_err
                            (idx.span,
                             "non-integral type of str index: "
                             + ty_to_str(fcx.ccx.tcx, idx_t));
                    }
                    auto ann = triv_ann(ty.mk_mach(fcx.ccx.tcx,
                                                   common.ty_u8));
                    ret @fold.respan[ast.expr_](expr.span,
                                                ast.expr_index(base_1,
                                                               idx_1,
                                                               ann));
                }
                case (_) {
                    fcx.ccx.sess.span_err
                        (expr.span,
                         "vector-indexing bad type: "
                         + ty_to_str(fcx.ccx.tcx, base_t));
                }
            }
        }

        case (ast.expr_port(_)) {
            auto t = next_ty_var(fcx.ccx);
            auto pt = ty.mk_port(fcx.ccx.tcx, t);
            auto ann = triv_ann(pt);
            ret @fold.respan[ast.expr_](expr.span, ast.expr_port(ann));
        }

        case (ast.expr_chan(?x, _)) {
            auto expr_1 = check_expr(fcx, x);
            auto port_t = expr_ty(fcx.ccx.tcx, expr_1);
            alt (struct(fcx.ccx.tcx, port_t)) {
                case (ty.ty_port(?subtype)) {
                    auto ct = ty.mk_chan(fcx.ccx.tcx, subtype);
                    auto ann = triv_ann(ct);
                    ret @fold.respan[ast.expr_](expr.span,
                                                ast.expr_chan(expr_1, ann));
                }
                case (_) {
                    fcx.ccx.sess.span_err(expr.span,
                        "bad port type: " + ty_to_str(fcx.ccx.tcx, port_t));
                }
            }
        }

        case (_) {
            fcx.ccx.sess.unimpl("expr type in typeck.check_expr");
            // TODO
            ret expr;
        }
    }
}

fn next_ty_var(@crate_ctxt ccx) -> ty.t {
    auto t = ty.mk_var(ccx.tcx, ccx.next_var_id);
    ccx.next_var_id += 1;
    ret t;
}

fn check_decl_local(&@fn_ctxt fcx, &@ast.decl decl) -> @ast.decl {
    alt (decl.node) {
        case (ast.decl_local(?local)) {

            auto t;

            t = ty.mk_nil(fcx.ccx.tcx);
            
            alt (local.ty) {
                case (none[@ast.ty]) {
                    // Auto slot. Do nothing for now.
                }

                case (some[@ast.ty](?ast_ty)) {
                    auto local_ty = ast_ty_to_ty_crate(fcx.ccx, ast_ty);
                    fcx.locals.insert(local.id, local_ty);
                    t = local_ty;
                }
            }

            auto a_res = local.ann; 
            alt (a_res) {
                case (ann_none) {
                    a_res = triv_ann(t);
                }
                case (_) {}
            }

            auto initopt = local.init;
            alt (local.init) {
                case (some[ast.initializer](?init)) {
                    auto expr_0 = check_expr(fcx, init.expr);
                    auto lty = ty.mk_local(fcx.ccx.tcx, local.id);
                    auto expr_1;
                    alt (init.op) {
                        case (ast.init_assign) {
                            expr_1 = Pushdown.pushdown_expr(fcx, lty, expr_0);
                        }
                        case (ast.init_recv) {
                            auto port_ty = ty.mk_port(fcx.ccx.tcx, lty);
                            expr_1 = Pushdown.pushdown_expr(fcx, port_ty,
                                                            expr_0);
                        }
                    }

                    auto init_0 = rec(expr = expr_1 with init);
                    initopt = some[ast.initializer](init_0);
                }
                case (_) { /* fall through */  }
            }
            auto local_1 = @rec(init = initopt, ann = a_res with *local);
            ret @rec(node=ast.decl_local(local_1)
                     with *decl);
        }
    }
}

fn check_stmt(&@fn_ctxt fcx, &@ast.stmt stmt) -> @ast.stmt {
    alt (stmt.node) {
        case (ast.stmt_decl(?decl,?a)) {
            alt (decl.node) {
                case (ast.decl_local(_)) {
                    auto decl_1 = check_decl_local(fcx, decl);
                    ret @fold.respan[ast.stmt_](stmt.span,
                           ast.stmt_decl(decl_1,
                             plain_ann(fcx.ccx.tcx)));
                }

                case (ast.decl_item(_)) {
                    // Ignore for now. We'll return later.
                    ret @fold.respan[ast.stmt_](stmt.span,
                           ast.stmt_decl(decl,
                             plain_ann(fcx.ccx.tcx)));
                }
            }

            //         ret stmt;
        }

        case (ast.stmt_expr(?expr,?a)) {
            auto expr_t = check_expr(fcx, expr);
            expr_t = Pushdown.pushdown_expr(fcx, expr_ty(fcx.ccx.tcx, expr_t),
                                            expr_t);
            ret @fold.respan[ast.stmt_](stmt.span,
                   ast.stmt_expr(expr_t, plain_ann(fcx.ccx.tcx)));
        }
    }

    fail;
}

fn check_block(&@fn_ctxt fcx, &ast.block block) -> ast.block {
    let vec[@ast.stmt] stmts = vec();
    for (@ast.stmt s in block.node.stmts) {
        Vec.push[@ast.stmt](stmts, check_stmt(fcx, s));
    }

    auto expr = none[@ast.expr];
    alt (block.node.expr) {
        case (none[@ast.expr]) { /* empty */ }
        case (some[@ast.expr](?e)) {
            auto expr_t = check_expr(fcx, e);
            expr_t = Pushdown.pushdown_expr(fcx,
                                            expr_ty(fcx.ccx.tcx, expr_t),
                                            expr_t);
            expr = some[@ast.expr](expr_t);
        }
    }

    ret fold.respan[ast.block_](block.span,
                                rec(stmts=stmts, expr=expr,
                                    index=block.node.index,
                                    a=plain_ann(fcx.ccx.tcx)));
}

fn check_const(&@crate_ctxt ccx, &span sp, ast.ident ident, @ast.ty t,
               @ast.expr e, ast.def_id id, ast.ann ann) -> @ast.item {
    // FIXME: this is kinda a kludge; we manufacture a fake "function context"
    // for checking the initializer expression.
    auto rty = ann_to_type(ann);
    let @fn_ctxt fcx = @rec(ret_ty = rty,
                            purity = ast.pure_fn,
                            locals = @common.new_def_hash[ty.t](),
                            ccx = ccx);
    auto e_ = check_expr(fcx, e);
    // FIXME: necessary? Correct sequence?
    Pushdown.pushdown_expr(fcx, rty, e_);
    auto item = ast.item_const(ident, t, e_, id, ann);
    ret @fold.respan[ast.item_](sp, item);
}

fn check_fn(&@crate_ctxt ccx, &ast.fn_decl decl, ast.proto proto,
            &ast.block body) -> ast._fn {
    auto local_ty_table = @common.new_def_hash[ty.t]();

    // FIXME: duplicate work: the item annotation already has the arg types
    // and return type translated to typeck.ty values. We don't need do to it
    // again here, we can extract them.


    for (ast.obj_field f in ccx.obj_fields) {
        auto field_ty = ty.ann_to_type(f.ann);
        local_ty_table.insert(f.id, field_ty);
    }

    // Store the type of each argument in the table.
    for (ast.arg arg in decl.inputs) {
        auto input_ty = ast_ty_to_ty_crate(ccx, arg.ty);
        local_ty_table.insert(arg.id, input_ty);
    }

    let @fn_ctxt fcx = @rec(ret_ty = ast_ty_to_ty_crate(ccx, decl.output),
                            purity = decl.purity,
                            locals = local_ty_table,
                            ccx = ccx);

    // TODO: Make sure the type of the block agrees with the function type.
    auto block_t = check_block(fcx, body);
    alt (decl.purity) {
        case (ast.pure_fn) {
            // per the previous comment, this just checks that the declared
            // type is bool, and trusts that that's the actual return type.
            if (!ty.type_is_bool(ccx.tcx, fcx.ret_ty)) {
              ccx.sess.span_err(body.span, "Non-boolean return type in pred");
            }
        }
        case (_) {} 
    }

    auto block_wb = resolve_local_types_in_block(fcx, block_t);

    auto fn_t = rec(decl=decl,
                    proto=proto,
                    body=block_wb);
    ret fn_t;
}

fn check_item_fn(&@crate_ctxt ccx, &span sp, ast.ident ident, &ast._fn f,
                 vec[ast.ty_param] ty_params, ast.def_id id,
                 ast.ann ann) -> @ast.item {

    // FIXME: duplicate work: the item annotation already has the arg types
    // and return type translated to typeck.ty values. We don't need do to it
    // again here, we can extract them.

    let vec[arg] inputs = vec();
    for (ast.arg arg in f.decl.inputs) {
        auto input_ty = ast_ty_to_ty_crate(ccx, arg.ty);
        inputs += vec(rec(mode=ast_mode_to_mode(arg.mode), ty=input_ty));
    }

    auto output_ty = ast_ty_to_ty_crate(ccx, f.decl.output);
    auto fn_ann = triv_ann(ty.mk_fn(ccx.tcx, f.proto, inputs, output_ty));

    auto item = ast.item_fn(ident, f, ty_params, id, fn_ann);
    ret @fold.respan[ast.item_](sp, item);
}

fn update_obj_fields(&@crate_ctxt ccx, @ast.item i) -> @crate_ctxt {
    alt (i.node) {
        case (ast.item_obj(_, ?ob, _, ?obj_def_ids, _)) {
            let ast.def_id di = obj_def_ids.ty;
            ret @rec(obj_fields = ob.fields, 
                     this_obj = some[ast.def_id](di) with *ccx);
        }
        case (_) {
        }
    }
    ret ccx;
}


// Utilities for the unification cache

fn hash_unify_cache_entry(&unify_cache_entry uce) -> uint {
    auto h = ty.hash_ty(uce._0);
    h += h << 5u + ty.hash_ty(uce._1);

    auto i = 0u;
    auto tys_len = Vec.len(uce._2);
    while (i < tys_len) {
        h += h << 5u + ty.hash_ty(uce._2.(i));
        i += 1u;
    }

    ret h;
}

fn eq_unify_cache_entry(&unify_cache_entry a, &unify_cache_entry b) -> bool {
    if (!ty.eq_ty(a._0, b._0) || !ty.eq_ty(a._1, b._1)) { ret false; }

    auto i = 0u;
    auto tys_len = Vec.len(a._2);
    if (Vec.len(b._2) != tys_len) { ret false; }

    while (i < tys_len) {
        if (!ty.eq_ty(a._2.(i), b._2.(i))) { ret false; }
        i += 1u;
    }

    ret true;
}

fn mk_fn_purity_table(@ast.crate crate) -> @fn_purity_table {
    auto res = @new_def_hash[ast.purity]();

    fn do_one(@fn_purity_table t, @ast.item i) -> () {
        alt (i.node) {
            case (ast.item_fn(_, ?f, _, ?d_id, _)) {
                t.insert(d_id, f.decl.purity);
            }
            case (_) {}
        }
    }

    auto do_one_fn = bind do_one(res,_);
    auto v = walk.default_visitor();

    auto add_fn_entry_visitor = rec(visit_item_post=do_one_fn with v);

    walk.walk_crate(add_fn_entry_visitor, *crate);
    ret res;
}

type typecheck_result = tup(@ast.crate, ty.type_cache);

fn check_crate(ty.ctxt tcx, @ast.crate crate)
        -> typecheck_result {
    auto sess = tcx.sess;
    auto result = Collect.collect_item_types(sess, tcx, crate);

    let vec[ast.obj_field] fields = vec();

    auto hasher = hash_unify_cache_entry;
    auto eqer = eq_unify_cache_entry;
    auto unify_cache =
        Map.mk_hashmap[unify_cache_entry,ty.Unify.result](hasher, eqer);
    auto fpt =
        mk_fn_purity_table(crate); // use a variation on Collect

    auto ccx = @rec(sess=sess,
                    type_cache=result._1,
                    item_items=result._2,
                    obj_fields=fields,
                    this_obj=none[ast.def_id],
                    fn_purity_table = fpt,
                    mutable next_var_id=0,
                    unify_cache=unify_cache,
                    mutable cache_hits=0u,
                    mutable cache_misses=0u,
                    tcx=tcx);

    auto fld = fold.new_identity_fold[@crate_ctxt]();

    fld = @rec(update_env_for_item = bind update_obj_fields(_, _),
               fold_fn      = bind check_fn(_,_,_,_),
               fold_item_fn = bind check_item_fn(_,_,_,_,_,_,_)
               with *fld);

    auto crate_1 = fold.fold_crate[@crate_ctxt](ccx, fld, result._0);

    log #fmt("cache hit rate: %u/%u", ccx.cache_hits,
             ccx.cache_hits + ccx.cache_misses);

    ret tup(crate_1, ccx.type_cache);
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
