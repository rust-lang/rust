
import syntax::ast;
import ast::mutability;
import ast::local_def;
import ast::path_to_str;
import ast::respan;
import syntax::walk;
import metadata::decoder;
import driver::session;
import util::common;
import syntax::codemap::span;
import std::map::new_int_hash;
import util::common::new_def_hash;
import util::common::log_expr_err;
import middle::ty;
import middle::ty::node_id_to_type;
import middle::ty::arg;
import middle::ty::bind_params_in_type;
import middle::ty::block_ty;
import middle::ty::expr_ty;
import middle::ty::field;
import middle::ty::method;
import middle::ty::mo_val;
import middle::ty::mo_alias;
import middle::ty::node_type_table;
import middle::ty::pat_ty;
import middle::ty::ty_param_substs_opt_and_ty;
import util::ppaux::ty_to_str;
import middle::ty::ty_param_count_and_ty;
import middle::ty::ty_nil;
import middle::ty::unify::ures_ok;
import middle::ty::unify::ures_err;
import middle::ty::unify::fixup_result;
import middle::ty::unify::fix_ok;
import middle::ty::unify::fix_err;
import std::int;
import std::ivec;
import std::str;
import std::ufind;
import std::uint;
import std::vec;
import std::map;
import std::map::hashmap;
import std::option;
import std::option::none;
import std::option::some;
import std::option::from_maybe;
import std::smallintmap;
import middle::tstate::ann::ts_ann;

export check_crate;

type ty_table = hashmap[ast::def_id, ty::t];

type obj_info = rec(vec[ast::obj_field] obj_fields, ast::node_id this_obj);

type crate_ctxt =
    rec(mutable vec[obj_info] obj_infos,
        ty::ctxt tcx);

type fn_ctxt =
    rec(ty::t ret_ty,
        ast::purity purity,
        @ty::unify::var_bindings var_bindings,
        hashmap[ast::node_id, int] locals,
        hashmap[ast::node_id, ast::ident] local_names,
        mutable int next_var_id,
        mutable vec[ast::node_id] fixups,
        @crate_ctxt ccx);


// Used for ast_ty_to_ty() below.
type ty_getter = fn(&ast::def_id) -> ty::ty_param_count_and_ty ;

fn lookup_local(&@fn_ctxt fcx, &span sp, ast::node_id id) -> int {
    alt (fcx.locals.find(id)) {
        case (some(?x)) { x }
        case (_) {
            fcx.ccx.tcx.sess.span_fatal(sp, "internal error looking up a \
              local var")
        }
    }
}

fn lookup_def(&@fn_ctxt fcx, &span sp, ast::node_id id) -> ast::def {
    alt (fcx.ccx.tcx.def_map.find(id)) {
        case (some(?x)) { x }
        case (_) {
            fcx.ccx.tcx.sess.span_fatal(sp, "internal error looking up \
              a definition")
        }
    }
}

// Returns the type parameter count and the type for the given definition.
fn ty_param_count_and_ty_for_def(&@fn_ctxt fcx, &span sp, &ast::def defn) ->
   ty_param_count_and_ty {
    alt (defn) {
        case (ast::def_arg(?id)) {
            assert (fcx.locals.contains_key(id._1));
            auto typ = ty::mk_var(fcx.ccx.tcx, 
                                  lookup_local(fcx, sp, id._1));
            ret tup(0u, typ);
        }
        case (ast::def_local(?id)) {
            assert (fcx.locals.contains_key(id._1));
            auto typ = ty::mk_var(fcx.ccx.tcx, lookup_local(fcx, sp, id._1));
            ret tup(0u, typ);
        }
        case (ast::def_obj_field(?id)) {
            assert (fcx.locals.contains_key(id._1));
            auto typ = ty::mk_var(fcx.ccx.tcx, lookup_local(fcx, sp, id._1));
            ret tup(0u, typ);
        }
        case (ast::def_fn(?id, _)) {
            ret ty::lookup_item_type(fcx.ccx.tcx, id);
        }
        case (ast::def_native_fn(?id)) {
            ret ty::lookup_item_type(fcx.ccx.tcx, id);
        }
        case (ast::def_const(?id)) {
            ret ty::lookup_item_type(fcx.ccx.tcx, id);
        }
        case (ast::def_variant(_, ?vid)) {
            ret ty::lookup_item_type(fcx.ccx.tcx, vid);
        }
        case (ast::def_binding(?id)) {
            assert (fcx.locals.contains_key(id._1));
            auto typ = ty::mk_var(fcx.ccx.tcx, lookup_local(fcx, sp, id._1));
            ret tup(0u, typ);
        }
        case (ast::def_mod(_)) {
            // Hopefully part of a path.
            // TODO: return a type that's more poisonous, perhaps?

            ret tup(0u, ty::mk_nil(fcx.ccx.tcx));
        }
        case (ast::def_ty(_)) {
            fcx.ccx.tcx.sess.span_fatal(sp, "expected value but found type");
        }
        case (_) {
            // FIXME: handle other names.

            fcx.ccx.tcx.sess.unimpl("definition variant");
        }
    }
}


// Instantiates the given path, which must refer to an item with the given
// number of type parameters and type.
fn instantiate_path(&@fn_ctxt fcx, &ast::path pth, &ty_param_count_and_ty tpt,
                    &span sp) -> ty_param_substs_opt_and_ty {
    auto ty_param_count = tpt._0;
    auto bind_result =
        bind_params_in_type(sp, fcx.ccx.tcx, bind next_ty_var_id(fcx), tpt._1,
                            ty_param_count);
    auto ty_param_vars = bind_result._0;
    auto ty_substs_opt;
    auto ty_substs_len = vec::len[@ast::ty](pth.node.types);
    if (ty_substs_len > 0u) {
        let ty::t[] ty_substs = ~[];
        auto i = 0u;
        while (i < ty_substs_len) {
            // TODO: Report an error if the number of type params in the item
            // and the supplied number of type params don't match.

            auto ty_var = ty::mk_var(fcx.ccx.tcx, ty_param_vars.(i));
            auto ty_subst = ast_ty_to_ty_crate(fcx.ccx, pth.node.types.(i));
            auto res_ty = demand::simple(fcx, pth.span, ty_var, ty_subst);
            ty_substs += ~[res_ty];
            i += 1u;
        }
        ty_substs_opt = some[ty::t[]](ty_substs);
        if (ty_param_count == 0u) {
            fcx.ccx.tcx.sess.span_fatal(sp,
                                      "this item does not take type " +
                                          "parameters");
            fail;
        }
    } else {
        // We will acquire the type parameters through unification.

        let ty::t[] ty_substs = ~[];
        auto i = 0u;
        while (i < ty_param_count) {
            ty_substs += ~[ty::mk_var(fcx.ccx.tcx, ty_param_vars.(i))];
            i += 1u;
        }
        ty_substs_opt = some[ty::t[]](ty_substs);
    }
    ret tup(ty_substs_opt, tpt._1);
}

fn ast_mode_to_mode(ast::mode mode) -> ty::mode {
    auto ty_mode;
    alt (mode) {
        case (ast::val) { ty_mode = mo_val; }
        case (ast::alias(?mut)) { ty_mode = mo_alias(mut); }
    }
    ret ty_mode;
}


// Type tests
fn structurally_resolved_type(&@fn_ctxt fcx, &span sp, ty::t typ) -> ty::t {
    auto r =
        ty::unify::resolve_type_structure(fcx.ccx.tcx, fcx.var_bindings, typ);
    alt (r) {
        case (fix_ok(?typ_s)) { ret typ_s; }
        case (fix_err(_)) {
            fcx.ccx.tcx.sess.span_fatal(sp,
                                      "the type of this value must be " +
                                          "known in this context");
        }
    }
}


// Returns the one-level-deep structure of the given type.
fn structure_of(&@fn_ctxt fcx, &span sp, ty::t typ) -> ty::sty {
    ret ty::struct(fcx.ccx.tcx, structurally_resolved_type(fcx, sp, typ));
}

fn type_is_integral(&@fn_ctxt fcx, &span sp, ty::t typ) -> bool {
    auto typ_s = structurally_resolved_type(fcx, sp, typ);
    ret ty::type_is_integral(fcx.ccx.tcx, typ_s);
}

fn type_is_scalar(&@fn_ctxt fcx, &span sp, ty::t typ) -> bool {
    auto typ_s = structurally_resolved_type(fcx, sp, typ);
    ret ty::type_is_scalar(fcx.ccx.tcx, typ_s);
}


// Parses the programmer's textual representation of a type into our internal
// notion of a type. `getter` is a function that returns the type
// corresponding to a definition ID:
fn ast_ty_to_ty(&ty::ctxt tcx, &ty_getter getter, &@ast::ty ast_ty) -> ty::t {
    alt (tcx.ast_ty_to_ty_cache.find(ast_ty)) {
        case (some[option::t[ty::t]](some[ty::t](?ty))) { ret ty; }
        case (some[option::t[ty::t]](none)) {
            tcx.sess.span_fatal(ast_ty.span,
                              "illegal recursive type " +
                              "(insert a tag in the cycle, " +
                              "if this is desired)");
        }
        case (none[option::t[ty::t]]) { }
    } /* go on */

    tcx.ast_ty_to_ty_cache.insert(ast_ty, none[ty::t]);
    fn ast_arg_to_arg(&ty::ctxt tcx, &ty_getter getter, &ast::ty_arg arg) ->
       rec(ty::mode mode, ty::t ty) {
        auto ty_mode = ast_mode_to_mode(arg.node.mode);
        ret rec(mode=ty_mode, ty=ast_ty_to_ty(tcx, getter, arg.node.ty));
    }
    fn ast_mt_to_mt(&ty::ctxt tcx, &ty_getter getter, &ast::mt mt) -> ty::mt {
        ret rec(ty=ast_ty_to_ty(tcx, getter, mt.ty), mut=mt.mut);
    }
    fn instantiate(&ty::ctxt tcx, &span sp, &ty_getter getter,
                   &ast::def_id id, &vec[@ast::ty] args) -> ty::t {
        // TODO: maybe record cname chains so we can do
        // "foo = int" like OCaml?

        auto params_opt_and_ty = getter(id);
        if (params_opt_and_ty._0 == 0u) { ret params_opt_and_ty._1; }
        // The typedef is type-parametric. Do the type substitution.
        //

        let ty::t[] param_bindings = ~[];
        for (@ast::ty ast_ty in args) {
            param_bindings += ~[ast_ty_to_ty(tcx, getter, ast_ty)];
        }
        if (ivec::len(param_bindings) !=
                ty::count_ty_params(tcx, params_opt_and_ty._1)) {
            tcx.sess.span_fatal(sp,
                              "Wrong number of type arguments for a" +
                                  " polymorphic tag");
        }
        auto typ =
            ty::substitute_type_params(tcx, param_bindings,
                                       params_opt_and_ty._1);
        ret typ;
    }
    auto typ;
    auto cname = none[str];
    alt (ast_ty.node) {
        case (ast::ty_nil) { typ = ty::mk_nil(tcx); }
        case (ast::ty_bot) { typ = ty::mk_bot(tcx); }
        case (ast::ty_bool) { typ = ty::mk_bool(tcx); }
        case (ast::ty_int) { typ = ty::mk_int(tcx); }
        case (ast::ty_uint) { typ = ty::mk_uint(tcx); }
        case (ast::ty_float) { typ = ty::mk_float(tcx); }
        case (ast::ty_machine(?tm)) { typ = ty::mk_mach(tcx, tm); }
        case (ast::ty_char) { typ = ty::mk_char(tcx); }
        case (ast::ty_str) { typ = ty::mk_str(tcx); }
        case (ast::ty_istr) { typ = ty::mk_istr(tcx); }
        case (ast::ty_box(?mt)) {
            typ = ty::mk_box(tcx, ast_mt_to_mt(tcx, getter, mt));
        }
        case (ast::ty_vec(?mt)) {
            typ = ty::mk_vec(tcx, ast_mt_to_mt(tcx, getter, mt));
        }
        case (ast::ty_ivec(?mt)) {
            typ = ty::mk_ivec(tcx, ast_mt_to_mt(tcx, getter, mt));
        }
        case (ast::ty_ptr(?mt)) {
            typ = ty::mk_ptr(tcx, ast_mt_to_mt(tcx, getter, mt));
        }
        case (ast::ty_task) { typ = ty::mk_task(tcx); }
        case (ast::ty_port(?t)) {
            typ = ty::mk_port(tcx, ast_ty_to_ty(tcx, getter, t));
        }
        case (ast::ty_chan(?t)) {
            typ = ty::mk_chan(tcx, ast_ty_to_ty(tcx, getter, t));
        }
        case (ast::ty_tup(?fields)) {
            let ty::mt[] flds = ~[];
            ivec::reserve(flds, vec::len(fields));
            for (ast::mt field in fields) {
                flds += ~[ast_mt_to_mt(tcx, getter, field)];
            }
            typ = ty::mk_tup(tcx, flds);
        }
        case (ast::ty_rec(?fields)) {
            let field[] flds = ~[];
            for (ast::ty_field f in fields) {
                auto tm = ast_mt_to_mt(tcx, getter, f.node.mt);
                flds += ~[rec(ident=f.node.ident, mt=tm)];
            }
            typ = ty::mk_rec(tcx, flds);
        }
        case (ast::ty_fn(?proto, ?inputs, ?output, ?cf, ?constrs)) {
            auto i = ~[];
            for (ast::ty_arg ta in inputs) {
                i += ~[ast_arg_to_arg(tcx, getter, ta)];
            }
            auto out_ty = ast_ty_to_ty(tcx, getter, output);

            auto out_constrs = ~[];
            for (@ast::constr constr in constrs) {
                out_constrs += ~[ast_constr_to_constr(tcx, constr)];
            }
            typ = ty::mk_fn(tcx, proto, i, out_ty, cf, out_constrs);
        }
        case (ast::ty_path(?path, ?id)) {
            alt (tcx.def_map.find(id)) {
                case (some(ast::def_ty(?id))) {
                    typ =
                        instantiate(tcx, ast_ty.span, getter, id,
                                    path.node.types);
                }
                case (some(ast::def_native_ty(?id))) { typ = getter(id)._1; }
                case (some(ast::def_ty_arg(?id))) {
                    typ = ty::mk_param(tcx, id);
                }
                case (some(_)) {
                    tcx.sess.span_fatal(ast_ty.span,
                                      "found type name used as a variable");
                }
                case (_) {
                    tcx.sess.span_fatal(ast_ty.span,
                                       "internal error in instantiate");
                }
            }
            cname = some(path_to_str(path));
        }
        case (ast::ty_obj(?meths)) {
            let vec[ty::method] tmeths = [];
            for (ast::ty_method m in meths) {
                auto ins = ~[];
                for (ast::ty_arg ta in m.node.inputs) {
                    ins += ~[ast_arg_to_arg(tcx, getter, ta)];
                }
                auto out = ast_ty_to_ty(tcx, getter, m.node.output);

                auto out_constrs = ~[];
                for (@ast::constr constr in m.node.constrs) {
                    out_constrs += ~[ast_constr_to_constr(tcx, constr)];
                }
                let ty::method new_m =
                    rec(proto=m.node.proto,
                        ident=m.node.ident,
                        inputs=ins,
                        output=out,
                        cf=m.node.cf,
                        constrs=out_constrs);
                vec::push[ty::method](tmeths, new_m);
            }
            typ = ty::mk_obj(tcx, ty::sort_methods(tmeths));
        }
    }
    alt (cname) {
        case (none) {/* no-op */ }
        case (some(?cname_str)) { typ = ty::rename(tcx, typ, cname_str); }
    }
    tcx.ast_ty_to_ty_cache.insert(ast_ty, some(typ));
    ret typ;
}


// A convenience function to use a crate_ctxt to resolve names for
// ast_ty_to_ty.
fn ast_ty_to_ty_crate(@crate_ctxt ccx, &@ast::ty ast_ty) -> ty::t {
    fn getter(@crate_ctxt ccx, &ast::def_id id) -> ty::ty_param_count_and_ty {
        ret ty::lookup_item_type(ccx.tcx, id);
    }
    auto f = bind getter(ccx, _);
    ret ast_ty_to_ty(ccx.tcx, f, ast_ty);
}


// Functions that write types into the node type table.
mod write {
    fn inner(&node_type_table ntt, ast::node_id node_id,
             &ty_param_substs_opt_and_ty tpot) {
        smallintmap::insert(*ntt, node_id as uint, tpot);
    }

    // Writes a type parameter count and type pair into the node type table.
    fn ty(&ty::ctxt tcx, ast::node_id node_id,
          &ty_param_substs_opt_and_ty tpot) {
        assert (!ty::type_contains_vars(tcx, tpot._1));
        ret inner(tcx.node_types, node_id, tpot);
    }

    // Writes a type parameter count and type pair into the node type table.
    // This function allows for the possibility of type variables, which will
    // be rewritten later during the fixup phase.
    fn ty_fixup(@fn_ctxt fcx, ast::node_id node_id,
                &ty_param_substs_opt_and_ty tpot) {
        inner(fcx.ccx.tcx.node_types, node_id, tpot);
        if (ty::type_contains_vars(fcx.ccx.tcx, tpot._1)) {
            fcx.fixups += [node_id];
        }
    }

    // Writes a type with no type parameters into the node type table.
    fn ty_only(&ty::ctxt tcx, ast::node_id node_id, ty::t typ) {
        ret ty(tcx, node_id, tup(none[ty::t[]], typ));
    }

    // Writes a type with no type parameters into the node type table. This
    // function allows for the possibility of type variables.
    fn ty_only_fixup(@fn_ctxt fcx, ast::node_id node_id, ty::t typ) {
        ret ty_fixup(fcx, node_id, tup(none[ty::t[]], typ));
    }

    // Writes a nil type into the node type table.
    fn nil_ty(&ty::ctxt tcx, ast::node_id node_id) {
        ret ty(tcx, node_id, tup(none[ty::t[]], ty::mk_nil(tcx)));
    }

    // Writes the bottom type into the node type table.
    fn bot_ty(&ty::ctxt tcx, ast::node_id node_id) {
        ret ty(tcx, node_id, tup(none[ty::t[]], ty::mk_bot(tcx)));
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
//
// TODO: This logic is quite convoluted; it's a relic of the time when we
// actually wrote types directly into the AST and didn't have a type cache.
// Could use some cleanup. Consider topologically sorting in phase (1) above.
mod collect {
    type ctxt = rec(ty::ctxt tcx);

    fn mk_ty_params(&@ctxt cx, uint n) -> ty::t[] {
        auto tps = ~[];
        auto i = 0u;
        while (i < n) {
            tps += ~[ty::mk_param(cx.tcx, i)];
            i += 1u;
        }
        ret tps;
    }
    fn ty_of_fn_decl(&@ctxt cx, &fn(&@ast::ty) -> ty::t  convert,
                     &fn(&ast::arg) -> arg  ty_of_arg, &ast::fn_decl decl,
                     ast::proto proto, &vec[ast::ty_param] ty_params,
                     &option::t[ast::def_id] def_id) ->
       ty::ty_param_count_and_ty {
        auto input_tys = ~[];
        for (ast::arg a in decl.inputs) { input_tys += ~[ty_of_arg(a)]; }
        auto output_ty = convert(decl.output);

        auto out_constrs = ~[];
        for (@ast::constr constr in decl.constraints) {
            out_constrs += ~[ast_constr_to_constr(cx.tcx, constr)];
        }
        auto t_fn =
            ty::mk_fn(cx.tcx, proto, input_tys, output_ty, decl.cf,
                      out_constrs);
        auto ty_param_count = vec::len[ast::ty_param](ty_params);
        auto tpt = tup(ty_param_count, t_fn);
        alt (def_id) {
            case (some(?did)) { cx.tcx.tcache.insert(did, tpt); }
            case (_) { }
        }
        ret tpt;
    }
    fn ty_of_native_fn_decl(&@ctxt cx, &fn(&@ast::ty) -> ty::t  convert,
                            &fn(&ast::arg) -> arg  ty_of_arg,
                            &ast::fn_decl decl, ast::native_abi abi,
                            &vec[ast::ty_param] ty_params,
                            &ast::def_id def_id) ->
       ty::ty_param_count_and_ty {
        auto input_tys = ~[];
        for (ast::arg a in decl.inputs) { input_tys += ~[ty_of_arg(a)]; }
        auto output_ty = convert(decl.output);

        auto t_fn = ty::mk_native_fn(cx.tcx, abi, input_tys, output_ty);
        auto ty_param_count = vec::len[ast::ty_param](ty_params);
        auto tpt = tup(ty_param_count, t_fn);
        cx.tcx.tcache.insert(def_id, tpt);
        ret tpt;
    }
    fn getter(@ctxt cx, &ast::def_id id) -> ty::ty_param_count_and_ty {
        if (id._0 != ast::local_crate) {
            // This is a type we need to load in from the crate reader.
            ret decoder::get_type(cx.tcx, id);
        }
        auto it = cx.tcx.items.find(id._1);
        auto tpt;
        alt (it) {
            case (some(ast_map::node_item(?item))) {
                tpt = ty_of_item(cx, item);
            }
            case (some(ast_map::node_native_item(?native_item))) {
                tpt = ty_of_native_item(cx, native_item,
                                        ast::native_abi_cdecl);
            }
            case (_) {
                cx.tcx.sess.fatal("internal error " +
                                  std::int::str(id._1));
            }
        }
        ret tpt;
    }
    fn ty_of_arg(@ctxt cx, &ast::arg a) -> ty::arg {
        auto ty_mode = ast_mode_to_mode(a.mode);
        auto f = bind getter(cx, _);
        auto tt = ast_ty_to_ty(cx.tcx, f, a.ty);
        if (ty::type_has_dynamic_size(cx.tcx, tt)) {
            alt (ty_mode) {
                case (mo_val) {
                    cx.tcx.sess.span_fatal(a.ty.span,
                                         "Dynamically sized arguments \
                                          must be passed by alias");
                }
                case (_) { }
            }
        }
        ret rec(mode=ty_mode, ty=tt);
    }
    fn ty_of_method(@ctxt cx, &@ast::method m) -> ty::method {
        auto get = bind getter(cx, _);
        auto convert = bind ast_ty_to_ty(cx.tcx, get, _);

        auto inputs = ~[];
        for (ast::arg a in m.node.meth.decl.inputs) {
            inputs += ~[ty_of_arg(cx, a)];
        }

        auto output = convert(m.node.meth.decl.output);

        auto out_constrs = ~[];
        for (@ast::constr constr in m.node.meth.decl.constraints) {
            out_constrs += ~[ast_constr_to_constr(cx.tcx, constr)];
        }
        ret rec(proto=m.node.meth.proto, ident=m.node.ident,
                inputs=inputs, output=output, cf=m.node.meth.decl.cf,
                constrs=out_constrs);
    }
    fn ty_of_obj(@ctxt cx, &ast::ident id, &ast::_obj obj_info,
                 &vec[ast::ty_param] ty_params) -> ty::ty_param_count_and_ty {
        auto methods = get_obj_method_types(cx, obj_info);
        auto t_obj = ty::mk_obj(cx.tcx, ty::sort_methods(methods));
        t_obj = ty::rename(cx.tcx, t_obj, id);
        ret tup(vec::len(ty_params), t_obj);
    }
    fn ty_of_obj_ctor(@ctxt cx, &ast::ident id, &ast::_obj obj_info,
                      ast::node_id ctor_id, &vec[ast::ty_param] ty_params) ->
       ty::ty_param_count_and_ty {
        auto t_obj = ty_of_obj(cx, id, obj_info, ty_params);

        let arg[] t_inputs = ~[];
        for (ast::obj_field f in obj_info.fields) {
            auto g = bind getter(cx, _);
            auto t_field = ast_ty_to_ty(cx.tcx, g, f.ty);
            t_inputs += ~[rec(mode=ty::mo_alias(false), ty=t_field)];
        }

        auto t_fn = ty::mk_fn(cx.tcx, ast::proto_fn, t_inputs, t_obj._1,
                              ast::return, ~[]);
        auto tpt = tup(t_obj._0, t_fn);
        cx.tcx.tcache.insert(local_def(ctor_id), tpt);
        ret tpt;
    }
    fn ty_of_item(&@ctxt cx, &@ast::item it) -> ty::ty_param_count_and_ty {
        auto get = bind getter(cx, _);
        auto convert = bind ast_ty_to_ty(cx.tcx, get, _);
        alt (it.node) {
            case (ast::item_const(?t, _)) {
                auto typ = convert(t);
                auto tpt = tup(0u, typ);
                cx.tcx.tcache.insert(local_def(it.id), tpt);
                ret tpt;
            }
            case (ast::item_fn(?fn_info, ?tps)) {
                auto f = bind ty_of_arg(cx, _);
                ret ty_of_fn_decl(cx, convert, f, fn_info.decl, fn_info.proto,
                                  tps, some(local_def(it.id)));
            }
            case (ast::item_obj(?obj_info, ?tps, _)) {
                auto t_obj = ty_of_obj(cx, it.ident, obj_info, tps);
                cx.tcx.tcache.insert(local_def(it.id), t_obj);
                ret t_obj;
            }
            case (ast::item_ty(?t, ?tps)) {
                alt (cx.tcx.tcache.find(local_def(it.id))) {
                    case (some(?tpt)) { ret tpt; }
                    case (none) { }
                }
                // Tell ast_ty_to_ty() that we want to perform a recursive
                // call to resolve any named types.

                auto typ = convert(t);
                auto ty_param_count = vec::len[ast::ty_param](tps);
                auto tpt = tup(ty_param_count, typ);
                cx.tcx.tcache.insert(local_def(it.id), tpt);
                ret tpt;
            }
            case (ast::item_res(?f, _, ?tps, _)) {
                auto t_arg = ty_of_arg(cx, f.decl.inputs.(0));
                auto t_res = tup(vec::len(tps), ty::mk_res
                                 (cx.tcx, local_def(it.id), t_arg.ty,
                                  mk_ty_params(cx, vec::len(tps))));
                cx.tcx.tcache.insert(local_def(it.id), t_res);
                ret t_res;
            }
            case (ast::item_tag(_, ?tps)) {
                // Create a new generic polytype.

                auto ty_param_count = vec::len[ast::ty_param](tps);

                let ty::t[] subtys = mk_ty_params(cx, ty_param_count);
                auto t = ty::mk_tag(cx.tcx, local_def(it.id), subtys);
                auto tpt = tup(ty_param_count, t);
                cx.tcx.tcache.insert(local_def(it.id), tpt);
                ret tpt;
            }
            case (ast::item_mod(_)) { fail; }
            case (ast::item_native_mod(_)) { fail; }
        }
    }
    fn ty_of_native_item(&@ctxt cx, &@ast::native_item it,
                         ast::native_abi abi) -> ty::ty_param_count_and_ty {
        alt (it.node) {
            case (ast::native_item_fn(_, ?fn_decl, ?params)) {
                auto get = bind getter(cx, _);
                auto convert = bind ast_ty_to_ty(cx.tcx, get, _);
                auto f = bind ty_of_arg(cx, _);
                ret ty_of_native_fn_decl(cx, convert, f, fn_decl, abi, params,
                                         ast::local_def(it.id));
            }
            case (ast::native_item_ty) {
                alt (cx.tcx.tcache.find(local_def(it.id))) {
                    case (some(?tpt)) { ret tpt; }
                    case (none) { }
                }
                auto t = ty::mk_native(cx.tcx, ast::local_def(it.id));
                auto tpt = tup(0u, t);
                cx.tcx.tcache.insert(local_def(it.id), tpt);
                ret tpt;
            }
        }
    }
    fn get_tag_variant_types(&@ctxt cx, &ast::def_id tag_id,
                             &vec[ast::variant] variants,
                             &vec[ast::ty_param] ty_params) {
        // Create a set of parameter types shared among all the variants.

        auto ty_param_count = vec::len[ast::ty_param](ty_params);
        let ty::t[] ty_param_tys = mk_ty_params(cx, ty_param_count);
        for (ast::variant variant in variants) {
            // Nullary tag constructors get turned into constants; n-ary tag
            // constructors get turned into functions.

            auto result_ty;
            if (vec::len[ast::variant_arg](variant.node.args) == 0u) {
                result_ty = ty::mk_tag(cx.tcx, tag_id, ty_param_tys);
            } else {
                // As above, tell ast_ty_to_ty() that trans_ty_item_to_ty()
                // should be called to resolve named types.

                auto f = bind getter(cx, _);
                let arg[] args = ~[];
                for (ast::variant_arg va in variant.node.args) {
                    auto arg_ty = ast_ty_to_ty(cx.tcx, f, va.ty);
                    args += ~[rec(mode=ty::mo_alias(false), ty=arg_ty)];
                }
                auto tag_t = ty::mk_tag(cx.tcx, tag_id, ty_param_tys);
                // FIXME: this will be different for constrained types
                result_ty = ty::mk_fn(cx.tcx, ast::proto_fn, args, tag_t,
                                      ast::return, ~[]);
            }
            auto tpt = tup(ty_param_count, result_ty);
            cx.tcx.tcache.insert(local_def(variant.node.id), tpt);
            write::ty_only(cx.tcx, variant.node.id, result_ty);
        }
    }
    fn get_obj_method_types(&@ctxt cx, &ast::_obj object) -> vec[ty::method] {
        ret vec::map[@ast::method,
                     method](bind ty_of_method(cx, _), object.methods);
    }
    fn convert(@ctxt cx, @mutable option::t[ast::native_abi] abi,
               &@ast::item it) {
        alt (it.node) {
            case (ast::item_mod(_)) {
                // ignore item_mod, it has no type.

            }
            case (ast::item_native_mod(?native_mod)) {
                // Propagate the native ABI down to convert_native() below,
                // but otherwise do nothing, as native modules have no types.

                *abi = some[ast::native_abi](native_mod.abi);
            }
            case (ast::item_tag(?variants, ?ty_params)) {
                auto tpt = ty_of_item(cx, it);
                write::ty_only(cx.tcx, it.id, tpt._1);
                get_tag_variant_types(cx, local_def(it.id), variants,
                                      ty_params);
            }
            case (ast::item_obj(?object, ?ty_params, ?ctor_id)) {
                // Now we need to call ty_of_obj_ctor(); this is the type that
                // we write into the table for this item.

                ty_of_item(cx, it);

                auto tpt =
                    ty_of_obj_ctor(cx, it.ident, object, ctor_id, ty_params);
                write::ty_only(cx.tcx, ctor_id, tpt._1);
                // Write the methods into the type table.
                //
                // FIXME: Inefficient; this ends up calling
                // get_obj_method_types() twice. (The first time was above in
                // ty_of_obj().)

                auto method_types = get_obj_method_types(cx, object);
                auto i = 0u;
                while (i < vec::len[@ast::method](object.methods)) {
                    write::ty_only(cx.tcx, object.methods.(i).node.id,
                                   ty::method_ty_to_fn_ty(cx.tcx,
                                                          method_types.(i)));
                    i += 1u;
                }
                // Write in the types of the object fields.
                //
                // FIXME: We want to use uint::range() here, but that causes
                // an assertion in trans.

                auto args = ty::ty_fn_args(cx.tcx, tpt._1);
                i = 0u;
                while (i < ivec::len[ty::arg](args)) {
                    auto fld = object.fields.(i);
                    write::ty_only(cx.tcx, fld.id, args.(i).ty);
                    i += 1u;
                }

                // Finally, write in the type of the destructor.
                alt (object.dtor) {
                    case (none) {/* nothing to do */ }
                    case (some(?m)) {
                        auto t = ty::mk_fn(cx.tcx, ast::proto_fn, ~[],
                                   ty::mk_nil(cx.tcx), ast::return, ~[]);
                        write::ty_only(cx.tcx, m.node.id, t);
                    }
                }
            }
            case (ast::item_res(?f, ?dtor_id, ?tps, ?ctor_id)) {
                auto t_arg = ty_of_arg(cx, f.decl.inputs.(0));
                auto t_res = ty::mk_res(cx.tcx, local_def(it.id), t_arg.ty,
                                        mk_ty_params(cx, vec::len(tps)));
                auto t_ctor = ty::mk_fn(cx.tcx, ast::proto_fn, ~[t_arg],
                                        t_res, ast::return, ~[]);
                auto t_dtor = ty::mk_fn(cx.tcx, ast::proto_fn, ~[t_arg],
                                        ty::mk_nil(cx.tcx), ast::return, ~[]);
                write::ty_only(cx.tcx, it.id, t_res);
                write::ty_only(cx.tcx, ctor_id, t_ctor);
                cx.tcx.tcache.insert(local_def(ctor_id),
                                     tup(vec::len(tps), t_ctor));
                write::ty_only(cx.tcx, dtor_id, t_dtor);
            }
            case (_) {
                // This call populates the type cache with the converted type
                // of the item in passing. All we have to do here is to write
                // it into the node type table.

                auto tpt = ty_of_item(cx, it);
                write::ty_only(cx.tcx, it.id, tpt._1);
            }
        }
    }
    fn convert_native(@ctxt cx, @mutable option::t[ast::native_abi] abi,
                      &@ast::native_item i) {
        // As above, this call populates the type table with the converted
        // type of the native item. We simply write it into the node type
        // table.

        auto tpt =
            ty_of_native_item(cx, i, option::get[ast::native_abi]({ *abi }));
        alt (i.node) {
            case (ast::native_item_ty) {
                // FIXME: Native types have no annotation. Should they? --pcw

            }
            case (ast::native_item_fn(_, _, _)) {
                write::ty_only(cx.tcx, i.id, tpt._1);
            }
        }
    }
    fn collect_item_types(&ty::ctxt tcx, &@ast::crate crate) {
        // We have to propagate the surrounding ABI to the native items
        // contained within the native module.

        auto abi = @mutable none[ast::native_abi];
        auto cx = @rec(tcx=tcx);
        auto visit =
            rec(visit_item_pre=bind convert(cx, abi, _),
                visit_native_item_pre=bind convert_native(cx, abi, _)
                with walk::default_visitor());
        walk::walk_crate(visit, *crate);
    }
}


// Type unification

// TODO: rename to just "unify"
mod unify {
    fn simple(&@fn_ctxt fcx, &ty::t expected, &ty::t actual) ->
       ty::unify::result {
        ret ty::unify::unify(expected, actual, fcx.var_bindings, fcx.ccx.tcx);
    }
}

tag autoderef_kind { AUTODEREF_OK; NO_AUTODEREF; }

// FIXME This is almost a duplicate of ty::type_autoderef, with structure_of
// instead of ty::struct.
fn do_autoderef(&@fn_ctxt fcx, &span sp, &ty::t t) -> ty::t {
    auto t1 = t;
    while (true) {
        alt (structure_of(fcx, sp, t1)) {
            case (ty::ty_box(?inner)) { t1 = inner.ty; }
            case (ty::ty_res(_, ?inner, ?tps)) {
                // FIXME: Remove this vec->ivec conversion.
                auto tps_ivec = ~[];
                for (ty::t tp in tps) { tps_ivec += ~[tp]; }

                t1 = ty::substitute_type_params(fcx.ccx.tcx, tps_ivec, inner);
            }
            case (ty::ty_tag(?did, ?tps)) {
                auto variants = ty::tag_variants(fcx.ccx.tcx, did);
                if (vec::len(variants) != 1u ||
                    vec::len(variants.(0).args) != 1u) {
                    ret t1;
                }
                t1 = ty::substitute_type_params(fcx.ccx.tcx, tps,
                                                variants.(0).args.(0));
            }
            case (_) { ret t1; }
        }
    }
    fail;
}

fn add_boxes(&@crate_ctxt ccx, uint n, &ty::t t) -> ty::t {
    auto t1 = t;
    while (n != 0u) { t1 = ty::mk_imm_box(ccx.tcx, t1); n -= 1u; }
    ret t1;
}

fn count_boxes(&@fn_ctxt fcx, &span sp, &ty::t t) -> uint {
    auto n = 0u;
    auto t1 = t;
    while (true) {
        alt (structure_of(fcx, sp, t1)) {
            case (ty::ty_box(?inner)) { n += 1u; t1 = inner.ty; }
            case (_) { ret n; }
        }
    }
    fail;
}

fn resolve_type_vars_if_possible(&@fn_ctxt fcx, ty::t typ) -> ty::t {
    alt (ty::unify::fixup_vars(fcx.ccx.tcx, fcx.var_bindings, typ)) {
        case (fix_ok(?new_type)) { ret new_type; }
        case (fix_err(_)) { ret typ; }
    }
}


// Demands - procedures that require that two types unify and emit an error
// message if they don't.
type ty_param_substs_and_ty = tup(ty::t[], ty::t);

mod demand {
    fn simple(&@fn_ctxt fcx, &span sp, &ty::t expected, &ty::t actual) ->
       ty::t {
        ret full(fcx, sp, expected, actual, ~[], NO_AUTODEREF)._1;
    }
    fn autoderef(&@fn_ctxt fcx, &span sp, &ty::t expected, &ty::t actual,
                 autoderef_kind adk) -> ty::t {
        ret full(fcx, sp, expected, actual, ~[], adk)._1;
    }

    // Requires that the two types unify, and prints an error message if they
    // don't. Returns the unified type and the type parameter substitutions.
    fn full(&@fn_ctxt fcx, &span sp, &ty::t expected, &ty::t actual,
            &ty::t[] ty_param_substs_0, autoderef_kind adk) ->
       ty_param_substs_and_ty {
        auto expected_1 = expected;
        auto actual_1 = actual;
        auto implicit_boxes = 0u;
        if (adk == AUTODEREF_OK) {
            expected_1 = do_autoderef(fcx, sp, expected_1);
            actual_1 = do_autoderef(fcx, sp, actual_1);
            implicit_boxes = count_boxes(fcx, sp, actual);
        }
        let vec[mutable ty::t] ty_param_substs = [mutable ];
        let vec[int] ty_param_subst_var_ids = [];
        for (ty::t ty_param_subst in ty_param_substs_0) {
            // Generate a type variable and unify it with the type parameter
            // substitution. We will then pull out these type variables.

            auto t_0 = next_ty_var(fcx);
            ty_param_substs += [mutable t_0];
            ty_param_subst_var_ids += [ty::ty_var_id(fcx.ccx.tcx, t_0)];
            simple(fcx, sp, ty_param_subst, t_0);
        }

        fn mk_result(&@fn_ctxt fcx, &ty::t result_ty,
                     &vec[int] ty_param_subst_var_ids,
                     uint implicit_boxes) -> ty_param_substs_and_ty {
            let ty::t[] result_ty_param_substs = ~[];
            for (int var_id in ty_param_subst_var_ids) {
                auto tp_subst = ty::mk_var(fcx.ccx.tcx, var_id);
                result_ty_param_substs += ~[tp_subst];
            }
            ret tup(result_ty_param_substs,
                    add_boxes(fcx.ccx, implicit_boxes, result_ty));
        }

        alt (unify::simple(fcx, expected_1, actual_1)) {
            case (ures_ok(?t)) {
                ret mk_result(fcx, t, ty_param_subst_var_ids,
                              implicit_boxes);
            }
            case (ures_err(?err)) {
                auto e_err = resolve_type_vars_if_possible(fcx, expected_1);
                auto a_err = resolve_type_vars_if_possible(fcx, actual_1);
                fcx.ccx.tcx.sess.span_err(sp,
                                          "mismatched types: expected " +
                                          ty_to_str(fcx.ccx.tcx, e_err) +
                                          " but found " +
                                          ty_to_str(fcx.ccx.tcx, a_err) +
                                          " (" + ty::type_err_to_str(err)
                                          + ")");
                ret mk_result(fcx, expected_1,
                              ty_param_subst_var_ids, implicit_boxes);
            }
        }
    }
}


// Returns true if the two types unify and false if they don't.
fn are_compatible(&@fn_ctxt fcx, &ty::t expected, &ty::t actual) -> bool {
    alt (unify::simple(fcx, expected, actual)) {
        case (ures_ok(_)) { ret true; }
        case (ures_err(_)) { ret false; }
    }
}


// Returns the types of the arguments to a tag variant.
fn variant_arg_types(&@crate_ctxt ccx, &span sp, &ast::def_id vid,
                     &ty::t[] tag_ty_params) -> vec[ty::t] {
    let vec[ty::t] result = [];
    auto tpt = ty::lookup_item_type(ccx.tcx, vid);
    alt (ty::struct(ccx.tcx, tpt._1)) {
        case (ty::ty_fn(_, ?ins, _, _, _)) {

            // N-ary variant.
            for (ty::arg arg in ins) {
                auto arg_ty =
                    ty::substitute_type_params(ccx.tcx, tag_ty_params,
                                               arg.ty);
                result += [arg_ty];
            }
        }
        case (_) {
            // Nullary variant. Do nothing, as there are no arguments.

        }
    }
    /* result is a vector of the *expected* types of all the fields */

    ret result;
}


// Type resolution: the phase that finds all the types in the AST with
// unresolved type variables and replaces "ty_var" types with their
// substitutions.
//
// TODO: inefficient since not all types have vars in them. It would be better
// to maintain a list of fixups.
mod writeback {

    export resolve_type_vars_in_block;

    fn resolve_type_vars_in_type(&@fn_ctxt fcx, &span sp, ty::t typ) ->
        option::t[ty::t] {
        if (!ty::type_contains_vars(fcx.ccx.tcx, typ)) { ret some(typ); }
        alt (ty::unify::fixup_vars(fcx.ccx.tcx, fcx.var_bindings, typ)) {
            case (fix_ok(?new_type)) { ret some(new_type); }
            case (fix_err(?vid)) {
                fcx.ccx.tcx.sess.span_err(sp,
                                          "cannot determine a type \
                                           for this expression");
                ret none;
            }
        }
    }
    fn resolve_type_vars_for_node(&@wb_ctxt wbcx,
                                  &span sp, ast::node_id id) {
        auto fcx = wbcx.fcx;
        auto tpot = ty::node_id_to_ty_param_substs_opt_and_ty
            (fcx.ccx.tcx, id);
        auto new_ty = alt (resolve_type_vars_in_type(fcx, sp, tpot._1)) {
            case (some(?t)) { t }
            case (none) {
                wbcx.success = false;
                ret
            }
        };
        auto new_substs_opt;
        alt (tpot._0) {
            case (none[ty::t[]]) { new_substs_opt = none[ty::t[]]; }
            case (some[ty::t[]](?substs)) {
                let ty::t[] new_substs = ~[];
                for (ty::t subst in substs) {
                    alt (resolve_type_vars_in_type(fcx, sp, subst)) {
                        case (some(?t)) {
                            new_substs += ~[t];
                        }
                        case (none) {
                            wbcx.success = false;
                            ret;
                        }
                    }
                }
                new_substs_opt = some[ty::t[]](new_substs);
            }
        }
        write::ty(fcx.ccx.tcx, id, tup(new_substs_opt, new_ty));
    }

    type wb_ctxt = rec(@fn_ctxt fcx,
                       // A flag to ignore contained items and lambdas
                       mutable bool ignore,
                       // As soon as we hit an error we have to stop resolving
                       // the entire function
                       mutable bool success);

    fn visit_stmt_pre(@wb_ctxt wbcx, &@ast::stmt s) {
        resolve_type_vars_for_node(wbcx, s.span, ty::stmt_node_id(s));
    }
    fn visit_expr_pre(@wb_ctxt wbcx, &@ast::expr e) {
        resolve_type_vars_for_node(wbcx, e.span, e.id);
    }
    fn visit_block_pre(@wb_ctxt wbcx, &ast::block b) {
        resolve_type_vars_for_node(wbcx, b.span, b.node.id);
    }
    fn visit_pat_pre(@wb_ctxt wbcx, &@ast::pat p) {
        resolve_type_vars_for_node(wbcx, p.span, p.id);
    }
    fn visit_local_pre(@wb_ctxt wbcx, &@ast::local l) {
        auto var_id = lookup_local(wbcx.fcx, l.span, l.node.id);
        auto fix_rslt =
            ty::unify::resolve_type_var(wbcx.fcx.ccx.tcx,
                                        wbcx.fcx.var_bindings,
                                        var_id);
        alt (fix_rslt) {
            case (fix_ok(?lty)) {
                write::ty_only(wbcx.fcx.ccx.tcx, l.node.id, lty);
            }
            case (fix_err(_)) {
                wbcx.fcx.ccx.tcx.sess.span_err(l.span,
                                               "cannot determine a type \
                                                for this local variable");
                wbcx.success = false;
            }
        }
    }
    fn visit_item_pre(@wb_ctxt wbcx, &@ast::item item) {
        wbcx.ignore = true;
    }
    fn visit_item_post(@wb_ctxt wbcx, &@ast::item item) {
        wbcx.ignore = false;
    }
    fn visit_fn_pre(@wb_ctxt wbcx, &ast::_fn f,
                    &vec[ast::ty_param] tps, &span sp,
                    &ast::fn_ident i, ast::node_id d) {
        wbcx.ignore = true;
    }
    fn visit_fn_post(@wb_ctxt wbcx, &ast::_fn f,
                     &vec[ast::ty_param] tps, &span sp,
                     &ast::fn_ident i, ast::node_id d) {
        wbcx.ignore = false;
    }
    fn keep_going(@wb_ctxt wbcx) -> bool { !wbcx.ignore && wbcx.success }

    fn resolve_type_vars_in_block(&@fn_ctxt fcx, &ast::block block) -> bool {
        auto wbcx = @rec(fcx = fcx,
                         mutable ignore = false,
                         mutable success = true);
        auto visit =
            rec(keep_going=bind keep_going(wbcx),
                visit_item_pre=bind visit_item_pre(wbcx, _),
                visit_item_post=bind visit_item_post(wbcx, _),
                visit_fn_pre=bind visit_fn_pre(wbcx, _, _, _, _, _),
                visit_fn_post=bind visit_fn_post(wbcx, _, _, _, _, _),
                visit_stmt_pre=bind visit_stmt_pre(wbcx, _),
                visit_expr_pre=bind visit_expr_pre(wbcx, _),
                visit_block_pre=bind visit_block_pre(wbcx, _),
                visit_pat_pre=bind visit_pat_pre(wbcx, _),
                visit_local_pre=bind visit_local_pre(wbcx, _)
                with walk::default_visitor());
        walk::walk_block(visit, block);
        ret wbcx.success;
    }
}


// Local variable gathering. We gather up all locals and create variable IDs
// for them before typechecking the function.
type gather_result =
    rec(@ty::unify::var_bindings var_bindings,
        hashmap[ast::node_id, int] locals,
        hashmap[ast::node_id, ast::ident] local_names,
        int next_var_id);

fn gather_locals(&@crate_ctxt ccx, &ast::fn_decl decl, &ast::block body,
                 &ast::node_id id) -> gather_result {
    fn next_var_id(@mutable int nvi) -> int {
        auto rv = *nvi;
        *nvi += 1;
        ret rv;
    }
    fn assign(&ty::ctxt tcx, &@ty::unify::var_bindings var_bindings,
              &hashmap[ast::node_id, int] locals,
              &hashmap[ast::node_id, ast::ident] local_names,
              @mutable int nvi,
              ast::node_id nid, &ast::ident ident, option::t[ty::t] ty_opt) {
        auto var_id = next_var_id(nvi);
        locals.insert(nid, var_id);
        local_names.insert(nid, ident);
        alt (ty_opt) {
            case (none[ty::t]) {/* nothing to do */ }
            case (some[ty::t](?typ)) {
                ty::unify::unify(ty::mk_var(tcx, var_id), typ, var_bindings,
                                 tcx);
            }
        }
    }
    auto vb = ty::unify::mk_var_bindings();
    auto locals = new_int_hash[int]();
    auto local_names = new_int_hash[ast::ident]();
    auto nvi = @mutable 0;
    // Add object fields, if any.

    alt (get_obj_info(ccx)) {
        case (option::some(?oinfo)) {
            for (ast::obj_field f in oinfo.obj_fields) {
                auto field_ty = ty::node_id_to_type(ccx.tcx, f.id);
                assign(ccx.tcx, vb, locals, local_names, nvi, f.id,
                       f.ident, some(field_ty));
            }
        }
        case (option::none) {/* no fields */ }
    }
    // Add formal parameters.

    auto args = ty::ty_fn_args(ccx.tcx, ty::node_id_to_type(ccx.tcx, id));
    auto i = 0u;
    for (ty::arg arg in args) {
        assign(ccx.tcx, vb, locals, local_names, nvi, decl.inputs.(i).id,
               decl.inputs.(i).ident, some[ty::t](arg.ty));
        i += 1u;
    }
    // Add explicitly-declared locals.

    fn visit_local_pre(@crate_ctxt ccx, @ty::unify::var_bindings vb,
                       hashmap[ast::node_id, int] locals,
                       hashmap[ast::node_id, ast::ident] local_names,
                       @mutable int nvi, &@ast::local local) {
        alt (local.node.ty) {
            case (none) {
                // Auto slot.

                assign(ccx.tcx, vb, locals, local_names, nvi, local.node.id,
                       local.node.ident, none[ty::t]);
            }
            case (some(?ast_ty)) {
                // Explicitly typed slot.

                auto local_ty = ast_ty_to_ty_crate(ccx, ast_ty);
                assign(ccx.tcx, vb, locals, local_names, nvi, local.node.id,
                       local.node.ident, some[ty::t](local_ty));
            }
        }
    }
    // Add pattern bindings.

    fn visit_pat_pre(@crate_ctxt ccx, @ty::unify::var_bindings vb,
                     hashmap[ast::node_id, int] locals,
                     hashmap[ast::node_id, ast::ident] local_names,
                     @mutable int nvi, &@ast::pat p) {
        alt (p.node) {
            case (ast::pat_bind(?ident)) {
                assign(ccx.tcx, vb, locals, local_names, nvi,
                       p.id, ident, none[ty::t]);
            }
            case (_) {/* no-op */ }
        }
    }
    auto visit =
        rec(visit_local_pre=bind visit_local_pre(ccx, vb, locals, local_names,
                                                 nvi, _),
            visit_pat_pre=bind visit_pat_pre(ccx, vb, locals, local_names,
                                             nvi, _)
            with walk::default_visitor());
    walk::walk_block(visit, body);
    ret rec(var_bindings=vb,
            locals=locals,
            local_names=local_names,
            next_var_id=*nvi);
}


// AST fragment utilities
fn replace_expr_type(&@fn_ctxt fcx, &@ast::expr expr,
                     &tup(ty::t[], ty::t) new_tyt) {
    auto new_tps;
    if (ty::expr_has_ty_params(fcx.ccx.tcx, expr)) {
        new_tps = some[ty::t[]](new_tyt._0);
    } else { new_tps = none[ty::t[]]; }
    write::ty_fixup(fcx, expr.id, tup(new_tps, new_tyt._1));
}


// AST fragment checking
fn check_lit(@crate_ctxt ccx, &@ast::lit lit) -> ty::t {
    alt (lit.node) {
        case (ast::lit_str(_, ast::sk_rc)) { ret ty::mk_str(ccx.tcx); }
        case (ast::lit_str(_, ast::sk_unique)) { ret ty::mk_istr(ccx.tcx); }
        case (ast::lit_char(_)) { ret ty::mk_char(ccx.tcx); }
        case (ast::lit_int(_)) { ret ty::mk_int(ccx.tcx); }
        case (ast::lit_float(_)) { ret ty::mk_float(ccx.tcx); }
        case (ast::lit_mach_float(?tm, _)) { ret ty::mk_mach(ccx.tcx, tm); }
        case (ast::lit_uint(_)) { ret ty::mk_uint(ccx.tcx); }
        case (ast::lit_mach_int(?tm, _)) { ret ty::mk_mach(ccx.tcx, tm); }
        case (ast::lit_nil) { ret ty::mk_nil(ccx.tcx); }
        case (ast::lit_bool(_)) { ret ty::mk_bool(ccx.tcx); }
    }
}


// Pattern checking is top-down rather than bottom-up so that bindings get
// their types immediately.
fn check_pat(&@fn_ctxt fcx, &@ast::pat pat, ty::t expected) {
    alt (pat.node) {
        case (ast::pat_wild) {
            write::ty_only_fixup(fcx, pat.id, expected);
        }
        case (ast::pat_lit(?lt)) {
            auto typ = check_lit(fcx.ccx, lt);
            typ = demand::simple(fcx, pat.span, expected, typ);
            write::ty_only_fixup(fcx, pat.id, typ);
        }
        case (ast::pat_bind(?name)) {
            auto vid = lookup_local(fcx, pat.span, pat.id);
            auto typ = ty::mk_var(fcx.ccx.tcx, vid);
            typ = demand::simple(fcx, pat.span, expected, typ);
            write::ty_only_fixup(fcx, pat.id, typ);
        }
        case (ast::pat_tag(?path, ?subpats)) {
            // Typecheck the path.
            auto v_def = lookup_def(fcx, path.span, pat.id);
            auto v_def_ids = ast::variant_def_ids(v_def);
            auto tag_tpt = ty::lookup_item_type(fcx.ccx.tcx, v_def_ids._0);
            auto path_tpot = instantiate_path(fcx, path, tag_tpt, pat.span);
            // Take the tag type params out of `expected`.

            alt (structure_of(fcx, pat.span, expected)) {
              case (ty::ty_tag(_, ?expected_tps)) {
                // Unify with the expected tag type.

                auto ctor_ty =
                    ty::ty_param_substs_opt_and_ty_to_monotype(fcx.ccx.tcx,
                                                               path_tpot);

                // FIXME: Remove this ivec->vec conversion.
                auto tps_vec = ~[];
                for (ty::t tp in expected_tps) { tps_vec += ~[tp]; }

                auto path_tpt =
                    demand::full(fcx, pat.span, expected, ctor_ty, tps_vec,
                                 NO_AUTODEREF);
                path_tpot = tup(some[ty::t[]](path_tpt._0), path_tpt._1);
                // Get the number of arguments in this tag variant.

                auto arg_types =
                    variant_arg_types(fcx.ccx, pat.span, v_def_ids._1,
                                      expected_tps);
                auto subpats_len = vec::len[@ast::pat](subpats);
                if (vec::len[ty::t](arg_types) > 0u) {
                    // N-ary variant.

                    auto arg_len = vec::len[ty::t](arg_types);
                    if (arg_len != subpats_len) {
                        // TODO: note definition of tag variant
                        // TODO (issue #448): Wrap a #fmt string over multiple
                        // lines...
                        auto s = #fmt("this pattern has %u field%s, but the \
                                       corresponding variant has %u field%s",
                                      subpats_len,
                                      if (subpats_len == 1u) {
                                          ""
                                      } else { "s" }, arg_len,
                                      if (arg_len == 1u) {
                                          ""
                                      } else { "s" });
                        fcx.ccx.tcx.sess.span_fatal(pat.span, s);
                    }
                    // TODO: vec::iter2

                    auto i = 0u;
                    for (@ast::pat subpat in subpats) {
                        check_pat(fcx, subpat, arg_types.(i));
                        i += 1u;
                    }
                } else if (subpats_len > 0u) {
                    // TODO: note definition of tag variant
                    // TODO (issue #448): Wrap a #fmt string over multiple
                    // lines...

                    fcx.ccx.tcx.sess.span_fatal(pat.span,
                                          #fmt("this pattern has %u field%s, \
                                                but the corresponding \
                                                variant has no fields",
                                               subpats_len,
                                               if (subpats_len == 1u) {
                                                   ""
                                               } else { "s" }));
                }
                write::ty_fixup(fcx, pat.id, path_tpot);
              }
              case (_) {
                // FIXME: Switch expected and actual in this message? I
                // can never tell.

                fcx.ccx.tcx.sess.span_fatal(pat.span,
                                            #fmt("mismatched types: \
                                                  expected tag, found %s",
                                                 ty_to_str(fcx.ccx.tcx,
                                                           expected)));
              }
            }
            write::ty_fixup(fcx, pat.id, path_tpot);
        }
    }
}

fn require_impure(&session::session sess, &ast::purity f_purity, &span sp) {
    alt (f_purity) {
        case (ast::impure_fn) { ret; }
        case (ast::pure_fn) {
            sess.span_fatal(sp,
                          "Found impure expression in pure function decl");
        }
    }
}

fn require_pure_call(@crate_ctxt ccx, &ast::purity caller_purity,
                     &@ast::expr callee, &span sp) {
    alt (caller_purity) {
        case (ast::impure_fn) { ret; }
        case (ast::pure_fn) {
            alt (ccx.tcx.def_map.find(callee.id)) {
                case (some(ast::def_fn(_, ast::pure_fn))) {
                    ret;
                }
                case (_) {
                    ccx.tcx.sess.span_fatal(sp,
                     "Pure function calls function not known to be pure");
                }
            }
        }
    }
}

fn check_expr(&@fn_ctxt fcx, &@ast::expr expr) {
    // fcx.ccx.tcx.sess.span_warn(expr.span, "typechecking expr " +
    //                            syntax::print::pprust::expr_to_str(expr));

    // A generic function to factor out common logic from call and bind
    // expressions.

    fn check_call_or_bind(&@fn_ctxt fcx, &span sp, &@ast::expr f,
                          &vec[option::t[@ast::expr]] args, bool is_call) {
        // Check the function.

        check_expr(fcx, f);
        // Get the function type.

        auto fty = expr_ty(fcx.ccx.tcx, f);

        // We want to autoderef calls but not binds
        auto fty_stripped =
            if (is_call) { do_autoderef(fcx, sp, fty) } else { fty };

        // Grab the argument types and the return type.
        auto arg_tys;
        alt (structure_of(fcx, sp, fty_stripped)) {
            case (ty::ty_fn(_, ?arg_tys_0, _, _, _)) { arg_tys = arg_tys_0; }
            case (ty::ty_native_fn(_, ?arg_tys_0, _)) { arg_tys = arg_tys_0; }
            case (_) {
                fcx.ccx.tcx.sess.span_fatal(f.span,
                                          "mismatched types: \
                                           expected function or native \
                                           function but found "
                                          + ty_to_str(fcx.ccx.tcx, fty));
            }
        }
        // Check that the correct number of arguments were supplied.

        auto expected_arg_count = ivec::len[ty::arg](arg_tys);
        auto supplied_arg_count = vec::len[option::t[@ast::expr]](args);
        if (expected_arg_count != supplied_arg_count) {
            fcx.ccx.tcx.sess.span_fatal(sp,
                                      #fmt("this function takes %u \
                                            parameter%s but %u parameter%s \
                                            supplied",
                                           expected_arg_count,
                                           if (expected_arg_count == 1u) {
                                               ""
                                           } else { "s" }, supplied_arg_count,
                                           if (supplied_arg_count == 1u) {
                                               " was"
                                           } else { "s were" }));
        }
        // Check the arguments.
        // TODO: iter2

        auto i = 0u;
        for (option::t[@ast::expr] a_opt in args) {
            alt (a_opt) {
                case (some(?a)) {
                    check_expr(fcx, a);
                    demand::simple(fcx, a.span, arg_tys.(i).ty,
                                   expr_ty(fcx.ccx.tcx, a));
                }
                case (none) {/* no-op */ }
            }
            i += 1u;
        }
    }
    // A generic function for checking assignment expressions

    fn check_assignment(&@fn_ctxt fcx, &span sp, &@ast::expr lhs,
                        &@ast::expr rhs, &ast::node_id id) {
        check_expr(fcx, lhs);
        check_expr(fcx, rhs);
        demand::simple(fcx, sp, expr_ty(fcx.ccx.tcx, lhs),
                       expr_ty(fcx.ccx.tcx, rhs));
        write::ty_only_fixup(fcx, id, ty::mk_nil(fcx.ccx.tcx));
    }
    // A generic function for checking call expressions

    fn check_call(&@fn_ctxt fcx, &span sp, &@ast::expr f,
                  &vec[@ast::expr] args) {
        let vec[option::t[@ast::expr]] args_opt_0 = [];
        for (@ast::expr arg in args) {
            args_opt_0 += [some[@ast::expr](arg)];
        }
        // Call the generic checker.

        check_call_or_bind(fcx, sp, f, args_opt_0, true);
    }
    // A generic function for checking for or for-each loops

    fn check_for_or_for_each(&@fn_ctxt fcx, &@ast::local local,
                             &ty::t element_ty, &ast::block body,
                             ast::node_id node_id) {
        check_decl_local(fcx, local);
        check_block(fcx, body);
        // Unify type of decl with element type of the seq
        demand::simple(fcx, local.span,
                       ty::decl_local_ty(fcx.ccx.tcx, local),
                       element_ty);
        auto typ = ty::mk_nil(fcx.ccx.tcx);
        write::ty_only_fixup(fcx, node_id, typ);
    }

    // A generic function for checking the pred in a check
    // or if-check
    fn check_pred_expr(&@fn_ctxt fcx, &@ast::expr e) {
        check_expr(fcx, e);
        demand::simple(fcx, e.span, ty::mk_bool(fcx.ccx.tcx),
                       expr_ty(fcx.ccx.tcx, e));
        
        /* e must be a call expr where all arguments are either
           literals or slots */
            alt (e.node) {
                case (ast::expr_call(?operator, ?operands)) {
                    alt (operator.node) {
                        case (ast::expr_path(?oper_name)) {
                            alt (fcx.ccx.tcx.def_map.find(operator.id)) {
                                case (some(ast::def_fn(?_d_id,
                                                       ast::pure_fn))) { 
                                    // do nothing
                                }
                                case (_) {
                                    fcx.ccx.tcx.sess.span_fatal(operator.span,
                                      "non-predicate as operator \
                                       in constraint");
                                }
                            }
                            for (@ast::expr operand in operands) {
                                if (!ast::is_constraint_arg(operand)) {
                                    auto s = "Constraint args must be \
                                              slot variables or literals";
                                    fcx.ccx.tcx.sess.span_fatal(e.span, s);
                                }
                            }
                        }
                        case (_) {
                            auto s = "In a constraint, expected the \
                                      constraint name to be an explicit name";
                            fcx.ccx.tcx.sess.span_fatal(e.span,s);
                        }
                    }
                }
                case (_) {
                    fcx.ccx.tcx.sess.span_fatal(e.span,
                                              "check on non-predicate");
                }
            }
    }

    // A generic function for checking the then and else in an if
    // or if-check
    fn check_then_else(&@fn_ctxt fcx, &ast::block thn,
                       &option::t[@ast::expr] elsopt,
                       ast::node_id id, &span sp) {
        check_block(fcx, thn);
        auto if_t =
            alt (elsopt) {
                    case (some(?els)) {
                        check_expr(fcx, els);
                        auto thn_t = block_ty(fcx.ccx.tcx, thn);
                        auto elsopt_t = expr_ty(fcx.ccx.tcx, els);
                        demand::simple(fcx, sp, thn_t, elsopt_t);
                        if (!ty::type_is_bot(fcx.ccx.tcx, elsopt_t)) {
                            elsopt_t
                                } else { thn_t }
                    }
                    case (none) { ty::mk_nil(fcx.ccx.tcx) }
            };
        write::ty_only_fixup(fcx, id, if_t);
    }

    // Checks the compatibility 
    fn check_binop_type_compat(&@fn_ctxt fcx, span span,
                               ty::t ty, ast::binop binop) {
        auto resolved_t = resolve_type_vars_if_possible(fcx, ty);
        if (!ty::is_binopable(fcx.ccx.tcx, resolved_t, binop)) {
            auto binopstr = ast::binop_to_str(binop);
            auto t_str = ty_to_str(fcx.ccx.tcx, resolved_t);
            auto errmsg = "binary operation " + binopstr
                + " cannot be applied to type `" + t_str + "`";
            fcx.ccx.tcx.sess.span_fatal(span, errmsg);
        }
    }

    auto id = expr.id;
    alt (expr.node) {
        case (ast::expr_lit(?lit)) {
            auto typ = check_lit(fcx.ccx, lit);
            write::ty_only_fixup(fcx, id, typ);
        }
        case (ast::expr_binary(?binop, ?lhs, ?rhs)) {
            check_expr(fcx, lhs);
            check_expr(fcx, rhs);

            auto lhs_t = expr_ty(fcx.ccx.tcx, lhs);
            auto rhs_t = expr_ty(fcx.ccx.tcx, rhs);

            demand::autoderef(fcx, rhs.span, lhs_t, rhs_t, AUTODEREF_OK);
            check_binop_type_compat(fcx, expr.span, lhs_t, binop);

            auto t = alt (binop) {
                case (ast::eq) { ty::mk_bool(fcx.ccx.tcx) }
                case (ast::lt) { ty::mk_bool(fcx.ccx.tcx) }
                case (ast::le) { ty::mk_bool(fcx.ccx.tcx) }
                case (ast::ne) { ty::mk_bool(fcx.ccx.tcx) }
                case (ast::ge) { ty::mk_bool(fcx.ccx.tcx) }
                case (ast::gt) { ty::mk_bool(fcx.ccx.tcx) }
                case (_) { do_autoderef(fcx, expr.span, lhs_t) }
            };
            write::ty_only_fixup(fcx, id, t);
        }
        case (ast::expr_unary(?unop, ?oper)) {
            check_expr(fcx, oper);
            auto oper_t = expr_ty(fcx.ccx.tcx, oper);
            alt (unop) {
                case (ast::box(?mut)) {
                    oper_t = ty::mk_box(fcx.ccx.tcx, rec(ty=oper_t, mut=mut));
                }
                case (ast::deref) {
                    alt (structure_of(fcx, expr.span, oper_t)) {
                        case (ty::ty_box(?inner)) { oper_t = inner.ty; }
                        case (ty::ty_res(_, ?inner, _)) { oper_t = inner; }
                        case (ty::ty_tag(?id, ?tps)) {
                            auto variants = ty::tag_variants(fcx.ccx.tcx, id);
                            if (vec::len(variants) != 1u ||
                                vec::len(variants.(0).args) != 1u) {
                                fcx.ccx.tcx.sess.span_fatal
                                    (expr.span, "can only dereference tags " +
                                     "with a single variant which has a " +
                                     "single argument");
                            }
                            oper_t = ty::substitute_type_params
                                (fcx.ccx.tcx, tps, variants.(0).args.(0));
                        }
                        case (_) {
                            fcx.ccx.tcx.sess.span_fatal
                                (expr.span, "dereferencing non-" + 
                                 "dereferenceable type: " +
                                 ty_to_str(fcx.ccx.tcx, oper_t));
                        }
                    }
                }
                case (ast::not) {
                    if (!type_is_integral(fcx, oper.span, oper_t) &&
                            structure_of(fcx, oper.span, oper_t) !=
                                ty::ty_bool) {
                        fcx.ccx.tcx.sess.span_fatal(expr.span,
                                                  #fmt("mismatched types: \
                                                        expected bool or \
                                                        integer but found %s",
                                                       ty_to_str(fcx.ccx.tcx,
                                                                 oper_t)));
                    }
                }
                case (_) { oper_t = do_autoderef(fcx, expr.span, oper_t); }
            }
            write::ty_only_fixup(fcx, id, oper_t);
        }
        case (ast::expr_path(?pth)) {
            auto defn = lookup_def(fcx, pth.span, id);
            auto tpt = ty_param_count_and_ty_for_def(fcx, expr.span, defn);
            if (ty::def_has_ty_params(defn)) {
                auto path_tpot = instantiate_path(fcx, pth, tpt, expr.span);
                write::ty_fixup(fcx, id, path_tpot);
                ret;
            }
            // The definition doesn't take type parameters. If the programmer
            // supplied some, that's an error.

            if (vec::len[@ast::ty](pth.node.types) > 0u) {
                fcx.ccx.tcx.sess.span_fatal(expr.span,
                                          "this kind of value does not \
                                           take type parameters");
            }
            write::ty_only_fixup(fcx, id, tpt._1);
        }
        case (ast::expr_ext(?p, ?args, ?body, ?expanded)) {
            check_expr(fcx, expanded);
            auto t = expr_ty(fcx.ccx.tcx, expanded);
            write::ty_only_fixup(fcx, id, t);
        }
        case (ast::expr_fail(?expr_opt)) {
            alt (expr_opt) {
                case (none) { /* do nothing */ }
                case (some(?e)) {
                    check_expr(fcx, e);
                    auto tcx = fcx.ccx.tcx;
                    auto ety = expr_ty(tcx, e);
                    demand::simple(fcx, e.span, ty::mk_str(tcx), ety);
                }
            }
            write::bot_ty(fcx.ccx.tcx, id);
        }
        case (ast::expr_break) { write::bot_ty(fcx.ccx.tcx, id); }
        case (ast::expr_cont) { write::bot_ty(fcx.ccx.tcx, id); }
        case (ast::expr_ret(?expr_opt)) {
            alt (expr_opt) {
                case (none) {
                    auto nil = ty::mk_nil(fcx.ccx.tcx);
                    if (!are_compatible(fcx, fcx.ret_ty, nil)) {
                        fcx.ccx.tcx.sess.span_fatal(expr.span,
                                                  "ret; in function \
                                                   returning non-nil");
                    }
                    write::bot_ty(fcx.ccx.tcx, id);
                }
                case (some(?e)) {
                    check_expr(fcx, e);
                    demand::simple(fcx, expr.span, fcx.ret_ty,
                                   expr_ty(fcx.ccx.tcx, e));
                    write::bot_ty(fcx.ccx.tcx, id);
                }
            }
        }
        case (ast::expr_put(?expr_opt)) {
            require_impure(fcx.ccx.tcx.sess, fcx.purity, expr.span);
            alt (expr_opt) {
                case (none) {
                    auto nil = ty::mk_nil(fcx.ccx.tcx);
                    if (!are_compatible(fcx, fcx.ret_ty, nil)) {
                        fcx.ccx.tcx.sess.span_fatal(expr.span,
                                                  "put; in iterator \
                                                   yielding non-nil");
                    }
                    write::nil_ty(fcx.ccx.tcx, id);
                }
                case (some(?e)) {
                    check_expr(fcx, e);
                    write::nil_ty(fcx.ccx.tcx, id);
                }
            }
        }
        case (ast::expr_be(?e)) {
            // FIXME: prove instead of assert

            assert (ast::is_call_expr(e));
            check_expr(fcx, e);
            demand::simple(fcx, e.span, fcx.ret_ty, expr_ty(fcx.ccx.tcx, e));
            write::nil_ty(fcx.ccx.tcx, id);
        }
        case (ast::expr_log(?l, ?e)) {
            check_expr(fcx, e);
            write::nil_ty(fcx.ccx.tcx, id);
        }
        case (ast::expr_check(_, ?e)) {
            check_pred_expr(fcx, e);
            write::nil_ty(fcx.ccx.tcx, id);
        }
        case (ast::expr_if_check(?cond, ?thn, ?elsopt)) {
            check_pred_expr(fcx, cond);
            check_then_else(fcx, thn, elsopt, id, expr.span);
        }
        case (ast::expr_ternary(_, _, _)) {
            check_expr(fcx, ast::ternary_to_if(expr));
        }
        case (ast::expr_assert(?e)) {
            check_expr(fcx, e);
            auto ety = expr_ty(fcx.ccx.tcx, e);
            demand::simple(fcx, expr.span, ty::mk_bool(fcx.ccx.tcx), ety);
            write::nil_ty(fcx.ccx.tcx, id);
        }
        case (ast::expr_move(?lhs, ?rhs)) {
            require_impure(fcx.ccx.tcx.sess, fcx.purity, expr.span);
            check_assignment(fcx, expr.span, lhs, rhs, id);
        }
        case (ast::expr_assign(?lhs, ?rhs)) {
            require_impure(fcx.ccx.tcx.sess, fcx.purity, expr.span);
            check_assignment(fcx, expr.span, lhs, rhs, id);
        }
        case (ast::expr_swap(?lhs, ?rhs)) {
            require_impure(fcx.ccx.tcx.sess, fcx.purity, expr.span);
            check_assignment(fcx, expr.span, lhs, rhs, id);
        }
        case (ast::expr_assign_op(?op, ?lhs, ?rhs)) {
            require_impure(fcx.ccx.tcx.sess, fcx.purity, expr.span);
            check_assignment(fcx, expr.span, lhs, rhs, id);
            check_binop_type_compat(fcx, expr.span,
                                    expr_ty(fcx.ccx.tcx, lhs), op);
        }
        case (ast::expr_send(?lhs, ?rhs)) {
            require_impure(fcx.ccx.tcx.sess, fcx.purity, expr.span);
            check_expr(fcx, lhs);
            check_expr(fcx, rhs);
            auto rhs_t = expr_ty(fcx.ccx.tcx, rhs);
            auto chan_t = ty::mk_chan(fcx.ccx.tcx, rhs_t);
            auto lhs_t = expr_ty(fcx.ccx.tcx, lhs);
            alt (structure_of(fcx, expr.span, lhs_t)) {
                case (ty::ty_chan(?it)) { }
                case (_) {
                    auto s = #fmt("mismatched types: expected chan \
                                   but found %s",
                                  ty_to_str(fcx.ccx.tcx,
                                            lhs_t));
                    fcx.ccx.tcx.sess.span_fatal(expr.span,s);
                }
            }
            write::ty_only_fixup(fcx, id, chan_t);
        }
        case (ast::expr_recv(?lhs, ?rhs)) {
            require_impure(fcx.ccx.tcx.sess, fcx.purity, expr.span);
            check_expr(fcx, lhs);
            check_expr(fcx, rhs);
            auto item_t = expr_ty(fcx.ccx.tcx, rhs);
            auto port_t = ty::mk_port(fcx.ccx.tcx, item_t);
            demand::simple(fcx, expr.span, port_t, expr_ty(fcx.ccx.tcx, lhs));
            write::ty_only_fixup(fcx, id, item_t);
        }
        case (ast::expr_if(?cond, ?thn, ?elsopt)) {
            check_expr(fcx, cond);
            demand::simple(fcx, cond.span,
                           ty::mk_bool(fcx.ccx.tcx),
                           expr_ty(fcx.ccx.tcx, cond));
            check_then_else(fcx, thn, elsopt, id, expr.span);
        }
        case (ast::expr_for(?decl, ?seq, ?body)) {
            check_expr(fcx, seq);
            auto elt_ty;
            auto ety = expr_ty(fcx.ccx.tcx, seq);
            alt (structure_of(fcx, expr.span, ety)) {
                case (ty::ty_vec(?vec_elt_ty)) { elt_ty = vec_elt_ty.ty; }
                case (ty::ty_str) {
                    elt_ty = ty::mk_mach(fcx.ccx.tcx, ast::ty_u8);
                }
                case (ty::ty_ivec(?vec_elt_ty)) { elt_ty = vec_elt_ty.ty; }
                case (ty::ty_istr) {
                    elt_ty = ty::mk_mach(fcx.ccx.tcx, ast::ty_u8);
                }
                case (_) {
                    fcx.ccx.tcx.sess.span_fatal(expr.span,
                        "mismatched types: expected vector or string but " +
                        "found " + ty_to_str(fcx.ccx.tcx, ety));
                }
            }
            check_for_or_for_each(fcx, decl, elt_ty, body, id);
        }
        case (ast::expr_for_each(?decl, ?seq, ?body)) {
            check_expr(fcx, seq);
            check_for_or_for_each(fcx, decl, expr_ty(fcx.ccx.tcx, seq), body,
                                  id);
        }
        case (ast::expr_while(?cond, ?body)) {
            check_expr(fcx, cond);
            check_block(fcx, body);
            demand::simple(fcx, cond.span, ty::mk_bool(fcx.ccx.tcx),
                           expr_ty(fcx.ccx.tcx, cond));
            auto typ = ty::mk_nil(fcx.ccx.tcx);
            write::ty_only_fixup(fcx, id, typ);
        }
        case (ast::expr_do_while(?body, ?cond)) {
            check_expr(fcx, cond);
            check_block(fcx, body);
            auto typ = block_ty(fcx.ccx.tcx, body);
            write::ty_only_fixup(fcx, id, typ);
        }
        case (ast::expr_alt(?expr, ?arms)) {
            check_expr(fcx, expr);
            // Typecheck the patterns first, so that we get types for all the
            // bindings.

            auto pattern_ty = ty::expr_ty(fcx.ccx.tcx, expr);
            let vec[@ast::pat] pats = [];
            for (ast::arm arm in arms) {
                check_pat(fcx, arm.pat, pattern_ty);
                pats += [arm.pat];
            }
            // Now typecheck the blocks.

            auto result_ty = next_ty_var(fcx);
            for (ast::arm arm in arms) {
                check_block(fcx, arm.block);
                auto bty = block_ty(fcx.ccx.tcx, arm.block);

                // Failing alt arms don't need to have a matching type
                if (!ty::type_is_bot(fcx.ccx.tcx, bty)) {
                    result_ty =
                        demand::simple(fcx, arm.block.span, result_ty, bty);
                }
            }
            write::ty_only_fixup(fcx, id, result_ty);
        }
        case (ast::expr_fn(?f)) {
            auto cx = @rec(tcx=fcx.ccx.tcx);
            auto convert =
                bind ast_ty_to_ty(cx.tcx, bind collect::getter(cx, _), _);
            auto ty_of_arg = bind collect::ty_of_arg(cx, _);
            auto fty =
                collect::ty_of_fn_decl(cx, convert, ty_of_arg, f.decl,
                                       f.proto, [], none)._1;
            write::ty_only_fixup(fcx, id, fty);
            check_fn(fcx.ccx, f.decl, f.proto, f.body, id);
        }
        case (ast::expr_block(?b)) {
            check_block(fcx, b);
            alt (b.node.expr) {
                case (some(?expr)) {
                    auto typ = expr_ty(fcx.ccx.tcx, expr);
                    write::ty_only_fixup(fcx, id, typ);
                }
                case (none) {
                    auto typ = ty::mk_nil(fcx.ccx.tcx);
                    write::ty_only_fixup(fcx, id, typ);
                }
            }
        }
        case (ast::expr_bind(?f, ?args)) {
            // Call the generic checker.

            check_call_or_bind(fcx, expr.span, f, args, false);
            // Pull the argument and return types out.

            auto proto_1;
            let ty::arg[] arg_tys_1 = ~[];
            auto rt_1;
            auto fty = expr_ty(fcx.ccx.tcx, f);
            auto t_1;
            alt (structure_of(fcx, expr.span, fty)) {
                case (ty::ty_fn(?proto, ?arg_tys, ?rt, ?cf, ?constrs)) {
                    proto_1 = proto;
                    rt_1 = rt;
                    // FIXME:
                    // probably need to munge the constrs to drop constraints
                    // for any bound args

                    // For each blank argument, add the type of that argument
                    // to the resulting function type.

                    auto i = 0u;
                    while (i < vec::len[option::t[@ast::expr]](args)) {
                        alt (args.(i)) {
                            case (some(_)) {/* no-op */ }
                            case (none) { arg_tys_1 += ~[arg_tys.(i)]; }
                        }
                        i += 1u;
                    }
                    t_1 =
                        ty::mk_fn(fcx.ccx.tcx, proto_1, arg_tys_1, rt_1, cf,
                                  constrs);
                }
                case (_) {
                    log_err "LHS of bind expr didn't have a function type?!";
                    fail;
                }
            }
            write::ty_only_fixup(fcx, id, t_1);
        }
        case (ast::expr_call(?f, ?args)) {
            /* here we're kind of hosed, as f can be any expr
             need to restrict it to being an explicit expr_path if we're
            inside a pure function, and need an environment mapping from
            function name onto purity-designation */

            require_pure_call(fcx.ccx, fcx.purity, f, expr.span);
            check_call(fcx, expr.span, f, args);
            // Pull the return type out of the type of the function.

            auto rt_1;
            auto fty = do_autoderef(fcx, expr.span,
                                   ty::expr_ty(fcx.ccx.tcx, f));
            alt (structure_of(fcx, expr.span, fty)) {
                case (ty::ty_fn(_, _, ?rt, _, _)) { rt_1 = rt; }
                case (ty::ty_native_fn(_, _, ?rt)) { rt_1 = rt; }
                case (_) {
                    log_err "LHS of call expr didn't have a function type?!";
                    fail;
                }
            }
            write::ty_only_fixup(fcx, id, rt_1);
        }
        case (ast::expr_self_method(?ident)) {
            auto t = ty::mk_nil(fcx.ccx.tcx);
            let ty::t this_obj_ty;
            let option::t[obj_info] this_obj_info = get_obj_info(fcx.ccx);
            alt (this_obj_info) {
                case (
                     // If we're inside a current object, grab its type.
                     some(?obj_info)) {
                    // FIXME: In the case of anonymous objects with methods
                    // containing self-calls, this lookup fails because
                    // obj_info.this_obj is not in the type cache

                    this_obj_ty =
                        ty::lookup_item_type(fcx.ccx.tcx,
                                             local_def(obj_info.this_obj))._1;
                }
                case (none) { fail; }
            }
            // Grab this method's type out of the current object type.

            alt (structure_of(fcx, expr.span, this_obj_ty)) {
                case (ty::ty_obj(?methods)) {
                    for (ty::method method in methods) {
                        if (method.ident == ident) {
                            t = ty::method_ty_to_fn_ty(fcx.ccx.tcx, method);
                        }
                    }
                }
                case (_) { fail; }
            }
            write::ty_only_fixup(fcx, id, t);
            require_impure(fcx.ccx.tcx.sess, fcx.purity, expr.span);
        }
        case (ast::expr_spawn(_, _, ?f, ?args)) {
            check_call(fcx, expr.span, f, args);
            auto fty = expr_ty(fcx.ccx.tcx, f);
            auto ret_ty = ty::ret_ty_of_fn_ty(fcx.ccx.tcx, fty);
            demand::simple(fcx, f.span, ty::mk_nil(fcx.ccx.tcx), ret_ty);
            // FIXME: Other typechecks needed

            auto typ = ty::mk_task(fcx.ccx.tcx);
            write::ty_only_fixup(fcx, id, typ);
        }
        case (ast::expr_cast(?e, ?t)) {
            check_expr(fcx, e);
            auto t_1 = ast_ty_to_ty_crate(fcx.ccx, t);
            // FIXME: there are more forms of cast to support, eventually.

            if (!(type_is_scalar(fcx, expr.span, expr_ty(fcx.ccx.tcx, e)) &&
                      type_is_scalar(fcx, expr.span, t_1))) {
                fcx.ccx.tcx.sess.span_fatal(expr.span,
                                          "non-scalar cast: " +
                                              ty_to_str(fcx.ccx.tcx,
                                                        expr_ty(fcx.ccx.tcx,
                                                                e)) + " as " +
                                              ty_to_str(fcx.ccx.tcx, t_1));
            }
            write::ty_only_fixup(fcx, id, t_1);
        }
        case (ast::expr_vec(?args, ?mut, ?kind)) {
            let ty::t t;
            if (vec::len[@ast::expr](args) == 0u) {
                t = next_ty_var(fcx);
            } else {
                check_expr(fcx, args.(0));
                t = expr_ty(fcx.ccx.tcx, args.(0));
            }
            for (@ast::expr e in args) {
                check_expr(fcx, e);
                auto expr_t = expr_ty(fcx.ccx.tcx, e);
                demand::simple(fcx, expr.span, t, expr_t);
            }
            auto typ;
            alt (kind) {
                case (ast::sk_rc) {
                    typ = ty::mk_vec(fcx.ccx.tcx, rec(ty=t, mut=mut));
                }
                case (ast::sk_unique) {
                    typ = ty::mk_ivec(fcx.ccx.tcx, rec(ty=t, mut=mut));
                }
            }
            write::ty_only_fixup(fcx, id, typ);
        }
        case (ast::expr_tup(?elts)) {
            let ty::mt[] elts_mt = ~[];
            ivec::reserve(elts_mt, vec::len(elts));
            for (ast::elt e in elts) {
                check_expr(fcx, e.expr);
                auto ety = expr_ty(fcx.ccx.tcx, e.expr);
                elts_mt += ~[rec(ty=ety, mut=e.mut)];
            }
            auto typ = ty::mk_tup(fcx.ccx.tcx, elts_mt);
            write::ty_only_fixup(fcx, id, typ);
        }
        case (ast::expr_rec(?fields, ?base)) {
            alt (base) {
                case (none) {/* no-op */ }
                case (some(?b_0)) { check_expr(fcx, b_0); }
            }
            let field[] fields_t = ~[];
            for (ast::field f in fields) {
                check_expr(fcx, f.node.expr);
                auto expr_t = expr_ty(fcx.ccx.tcx, f.node.expr);
                auto expr_mt = rec(ty=expr_t, mut=f.node.mut);
                fields_t += ~[rec(ident=f.node.ident, mt=expr_mt)];
            }
            alt (base) {
                case (none) {
                    auto typ = ty::mk_rec(fcx.ccx.tcx, fields_t);
                    write::ty_only_fixup(fcx, id, typ);
                }
                case (some(?bexpr)) {
                    check_expr(fcx, bexpr);
                    auto bexpr_t = expr_ty(fcx.ccx.tcx, bexpr);
                    let field[] base_fields = ~[];
                    alt (structure_of(fcx, expr.span, bexpr_t)) {
                        case (ty::ty_rec(?flds)) { base_fields = flds; }
                        case (_) {
                            fcx.ccx.tcx.sess.span_fatal(expr.span,
                                                      "record update \
                                                       non-record base");
                        }
                    }
                    write::ty_only_fixup(fcx, id, bexpr_t);
                    for (ty::field f in fields_t) {
                        auto found = false;
                        for (ty::field bf in base_fields) {
                            if (str::eq(f.ident, bf.ident)) {
                                demand::simple(fcx, expr.span, bf.mt.ty,
                                               f.mt.ty);
                                found = true;
                            }
                        }
                        if (!found) {
                            fcx.ccx.tcx.sess.span_fatal(expr.span,
                                                      "unknown field in \
                                                       record update: "
                                                      + f.ident);
                        }
                    }
                }
            }
        }
        case (ast::expr_field(?base, ?field)) {
            check_expr(fcx, base);
            auto base_t = expr_ty(fcx.ccx.tcx, base);
            base_t = do_autoderef(fcx, expr.span, base_t);
            alt (structure_of(fcx, expr.span, base_t)) {
                case (ty::ty_tup(?args)) {
                    let uint ix =
                        ty::field_num(fcx.ccx.tcx.sess, expr.span, field);
                    if (ix >= ivec::len[ty::mt](args)) {
                        fcx.ccx.tcx.sess.span_fatal(expr.span,
                                                  "bad index on tuple");
                    }
                    write::ty_only_fixup(fcx, id, args.(ix).ty);
                }
                case (ty::ty_rec(?fields)) {
                    let uint ix =
                        ty::field_idx(fcx.ccx.tcx.sess, expr.span, field,
                                      fields);
                    if (ix >= ivec::len[ty::field](fields)) {
                        fcx.ccx.tcx.sess.span_fatal(expr.span,
                                                  "bad index on record");
                    }
                    write::ty_only_fixup(fcx, id, fields.(ix).mt.ty);
                }
                case (ty::ty_obj(?methods)) {
                    // log_err "checking method_idx 1...";
                    let uint ix =
                        ty::method_idx(fcx.ccx.tcx.sess, expr.span, field,
                                       methods);
                    if (ix >= vec::len[ty::method](methods)) {
                        fcx.ccx.tcx.sess.span_fatal(expr.span,
                                                  "bad index on obj");
                    }
                    auto meth = methods.(ix);
                    auto t =
                        ty::mk_fn(fcx.ccx.tcx, meth.proto, meth.inputs,
                                  meth.output, meth.cf, meth.constrs);
                    write::ty_only_fixup(fcx, id, t);
                }
                case (_) {
                    auto t_err = resolve_type_vars_if_possible(fcx, base_t);
                    auto msg = #fmt("attempted field access on type %s",
                                    ty_to_str(fcx.ccx.tcx, t_err));
                    fcx.ccx.tcx.sess.span_fatal(expr.span, msg);
                }
            }
        }
        case (ast::expr_index(?base, ?idx)) {
            check_expr(fcx, base);
            auto base_t = expr_ty(fcx.ccx.tcx, base);
            base_t = do_autoderef(fcx, expr.span, base_t);
            check_expr(fcx, idx);
            auto idx_t = expr_ty(fcx.ccx.tcx, idx);
            if (!type_is_integral(fcx, idx.span, idx_t)) {
                fcx.ccx.tcx.sess.span_fatal(idx.span,
                                          "mismatched types: expected \
                                           integer but found "
                                          + ty_to_str(fcx.ccx.tcx, idx_t));
            }
            alt (structure_of(fcx, expr.span, base_t)) {
                case (ty::ty_vec(?mt)) {
                    write::ty_only_fixup(fcx, id, mt.ty);
                }
                case (ty::ty_ivec(?mt)) {
                    write::ty_only_fixup(fcx, id, mt.ty);
                }
                case (ty::ty_str) {
                    auto typ = ty::mk_mach(fcx.ccx.tcx, ast::ty_u8);
                    write::ty_only_fixup(fcx, id, typ);
                }
                case (ty::ty_istr) {
                    auto typ = ty::mk_mach(fcx.ccx.tcx, ast::ty_u8);
                    write::ty_only_fixup(fcx, id, typ);
                }
                case (_) {
                    fcx.ccx.tcx.sess.span_fatal(expr.span,
                                              "vector-indexing bad type: " +
                                                  ty_to_str(fcx.ccx.tcx,
                                                            base_t));
                }
            }
        }
        case (ast::expr_port(?typ)) {
            auto t = next_ty_var(fcx);
            alt(typ) {
                case (some(?_t)) {
                    demand::simple(fcx, expr.span, 
                                   ast_ty_to_ty_crate(fcx.ccx, _t), 
                                   t);
                }
                case (none) {}
            }
            auto pt = ty::mk_port(fcx.ccx.tcx, t);
            write::ty_only_fixup(fcx, id, pt);
        }
        case (ast::expr_chan(?x)) {
            check_expr(fcx, x);
            auto port_t = expr_ty(fcx.ccx.tcx, x);
            alt (structure_of(fcx, expr.span, port_t)) {
                case (ty::ty_port(?subtype)) {
                    auto ct = ty::mk_chan(fcx.ccx.tcx, subtype);
                    write::ty_only_fixup(fcx, id, ct);
                }
                case (_) {
                    fcx.ccx.tcx.sess.span_fatal(expr.span,
                                              "bad port type: " +
                                                  ty_to_str(fcx.ccx.tcx,
                                                            port_t));
                }
            }
        }
        case (ast::expr_anon_obj(?anon_obj, ?tps)) {
            // TODO: We probably need to do more work here to be able to
            // handle additional methods that use 'self'

            // We're entering an object, so gather up the info we need.

            let vec[ast::anon_obj_field] fields = [];
            alt (anon_obj.fields) {
                case (none) { }
                case (some(?v)) { fields = v; }
            }

            // FIXME: this is duplicated between here and trans -- it should
            // appear in one place
            fn anon_obj_field_to_obj_field(&ast::anon_obj_field f) 
                -> ast::obj_field {
                ret rec(mut=f.mut, ty=f.ty, ident=f.ident, id=f.id);
            }

            vec::push[obj_info](fcx.ccx.obj_infos,
                                rec(obj_fields=
                                    vec::map(anon_obj_field_to_obj_field, 
                                             fields),
                                    this_obj=id));

            // FIXME: These next three functions are largely ripped off from
            // similar ones in collect::.  Is there a better way to do this?
            fn ty_of_arg(@crate_ctxt ccx, &ast::arg a) -> ty::arg {
                auto ty_mode = ast_mode_to_mode(a.mode);
                ret rec(mode=ty_mode, ty=ast_ty_to_ty_crate(ccx, a.ty));
            }
            fn ty_of_method(@crate_ctxt ccx, &@ast::method m) -> ty::method {
                auto convert = bind ast_ty_to_ty_crate(ccx, _);

                auto inputs = ~[];
                for (ast::arg aa in m.node.meth.decl.inputs) {
                    inputs += ~[ty_of_arg(ccx, aa)];
                }

                auto output = convert(m.node.meth.decl.output);

                auto out_constrs = ~[];
                for (@ast::constr constr in m.node.meth.decl.constraints) {
                    out_constrs += ~[ast_constr_to_constr(ccx.tcx, constr)];
                }

                ret rec(proto=m.node.meth.proto, ident=m.node.ident,
                        inputs=inputs, output=output, cf=m.node.meth.decl.cf,
                        constrs=out_constrs);
            }
            fn get_anon_obj_method_types(@fn_ctxt fcx,
                                         &ast::anon_obj anon_obj) ->
               vec[ty::method] {

                let vec[ty::method] methods = [];

                // Outer methods.
                methods += vec::map[@ast::method,
                                    method](bind ty_of_method(fcx.ccx, _),
                                            anon_obj.methods);

                // Inner methods.

                // Typecheck 'with_obj'.  If it exists, it had better have
                // object type.
                let vec[ty::method] with_obj_methods = [];
                alt (anon_obj.with_obj) {
                    case (none) { }
                    case (some(?e)) {
                        check_expr(fcx, e);
                        auto with_obj_ty = expr_ty(fcx.ccx.tcx, e);

                        alt (structure_of(fcx, e.span, with_obj_ty)) {
                            case (ty::ty_obj(?ms)) {
                                with_obj_methods = ms;
                            }
                            case (_) {
                                // The user is trying to extend a non-object.
                                fcx.ccx.tcx.sess.span_fatal(
                                    e.span,
                                    syntax::print::pprust::expr_to_str(e) + 
                                    " does not have object type");
                            }
                        }
                    }
                }
                methods += with_obj_methods;

                ret methods;
            }

            auto method_types = get_anon_obj_method_types(fcx, anon_obj);
            auto ot = ty::mk_obj(fcx.ccx.tcx, ty::sort_methods(method_types));

            write::ty_only_fixup(fcx, id, ot);
            // Write the methods into the node type table.  (This happens in
            // collect::convert for regular objects.)

            auto i = 0u;
            while (i < vec::len[@ast::method](anon_obj.methods)) {
                write::ty_only(fcx.ccx.tcx, anon_obj.methods.(i).node.id,
                               ty::method_ty_to_fn_ty(fcx.ccx.tcx,
                                                      method_types.(i)));
                i += 1u;
            }
            // Typecheck the methods.

            for (@ast::method method in anon_obj.methods) {
                check_method(fcx.ccx, method);
            }
            next_ty_var(fcx);
            // Now remove the info from the stack.

            vec::pop[obj_info](fcx.ccx.obj_infos);
        }
        case (_) {
            fcx.ccx.tcx.sess.unimpl("expr type in typeck::check_expr");
        }
    }
}

fn next_ty_var_id(@fn_ctxt fcx) -> int {
    auto id = fcx.next_var_id;
    fcx.next_var_id += 1;
    ret id;
}

fn next_ty_var(&@fn_ctxt fcx) -> ty::t {
    ret ty::mk_var(fcx.ccx.tcx, next_ty_var_id(fcx));
}

fn get_obj_info(&@crate_ctxt ccx) -> option::t[obj_info] {
    ret vec::last[obj_info](ccx.obj_infos);
}

fn ast_constr_to_constr(ty::ctxt tcx, &@ast::constr c)
    -> @ty::constr_def {
    alt (tcx.def_map.find(c.node.id)) {
        case (some(ast::def_fn(?pred_id, ast::pure_fn))) {
            // FIXME: Remove this vec->ivec conversion.
            let (@ast::constr_arg_general[uint])[] cag_ivec = ~[];
            for (@ast::constr_arg_general[uint] cag in c.node.args) {
                cag_ivec += ~[cag];
            }

            ret @respan(c.span, rec(path=c.node.path, args=cag_ivec,
                                    id=pred_id));
        }
        case (_) {
            tcx.sess.span_fatal(c.span, "Predicate "
                              + path_to_str(c.node.path)
                              + " is unbound or bound to a non-function or an\
                                impure function");
        }
    }
}

fn check_decl_initializer(&@fn_ctxt fcx, ast::node_id nid,
                          &ast::initializer init) {
    check_expr(fcx, init.expr);
    auto lty = ty::mk_var(fcx.ccx.tcx,
                          lookup_local(fcx, init.expr.span, nid));
    alt (init.op) {
        case (ast::init_assign) {
            demand::simple(fcx, init.expr.span, lty,
                           expr_ty(fcx.ccx.tcx, init.expr));
        }
        case (ast::init_move) {
            demand::simple(fcx, init.expr.span, lty,
                           expr_ty(fcx.ccx.tcx, init.expr));
        }
        case (ast::init_recv) {
            auto port_ty = ty::mk_port(fcx.ccx.tcx, lty);
            demand::simple(fcx, init.expr.span, port_ty,
                           expr_ty(fcx.ccx.tcx, init.expr));
        }
    }
}

fn check_decl_local(&@fn_ctxt fcx, &@ast::local local) -> @ast::local {
    auto a_id = local.node.id;
    alt (fcx.locals.find(a_id)) {
        case (none) {

            fcx.ccx.tcx.sess.bug("check_decl_local: local id not found " +
                                     local.node.ident);
        }
        case (some(?i)) {
            auto t = ty::mk_var(fcx.ccx.tcx, i);
            write::ty_only_fixup(fcx, a_id, t);
            auto initopt = local.node.init;
            alt (initopt) {
                case (some(?init)) {
                    check_decl_initializer(fcx, local.node.id, init);
                }
                case (_) {/* fall through */ }
            }
            auto newlocal = rec(init=initopt with local.node);
            ret @rec(node=newlocal, span=local.span);
        }
    }
}

fn check_stmt(&@fn_ctxt fcx, &@ast::stmt stmt) {
    auto node_id;
    alt (stmt.node) {
        case (ast::stmt_decl(?decl, ?id)) {
            node_id = id;
            alt (decl.node) {
                case (ast::decl_local(?l)) { check_decl_local(fcx, l); }
                case (ast::decl_item(_)) {/* ignore for now */ }
            }
        }
        case (ast::stmt_expr(?expr, ?id)) {
            node_id = id;
            check_expr(fcx, expr);
        }
    }
    write::nil_ty(fcx.ccx.tcx, node_id);
}

fn check_block(&@fn_ctxt fcx, &ast::block block) {
    for (@ast::stmt s in block.node.stmts) { check_stmt(fcx, s); }
    alt (block.node.expr) {
        case (none) { write::nil_ty(fcx.ccx.tcx, block.node.id); }
        case (some(?e)) {
            check_expr(fcx, e);
            auto ety = expr_ty(fcx.ccx.tcx, e);
            write::ty_only_fixup(fcx, block.node.id, ety);
        }
    }
}

fn check_const(&@crate_ctxt ccx, &span sp, &@ast::expr e, &ast::node_id id) {
    // FIXME: this is kinda a kludge; we manufacture a fake function context
    // and statement context for checking the initializer expression.

    auto rty = node_id_to_type(ccx.tcx, id);
    let vec[ast::node_id] fixups = [];
    let @fn_ctxt fcx =
        @rec(ret_ty=rty,
             purity=ast::pure_fn,
             var_bindings=ty::unify::mk_var_bindings(),
             locals=new_int_hash[int](),
             local_names=new_int_hash[ast::ident](),
             mutable next_var_id=0,
             mutable fixups=fixups,
             ccx=ccx);
    check_expr(fcx, e);
}

fn check_fn(&@crate_ctxt ccx, &ast::fn_decl decl, ast::proto proto,
            &ast::block body, &ast::node_id id) {
    auto gather_result = gather_locals(ccx, decl, body, id);
    let vec[ast::node_id] fixups = [];
    let @fn_ctxt fcx =
        @rec(ret_ty=ast_ty_to_ty_crate(ccx, decl.output),
             purity=decl.purity,
             var_bindings=gather_result.var_bindings,
             locals=gather_result.locals,
             local_names=gather_result.local_names,
             mutable next_var_id=gather_result.next_var_id,
             mutable fixups=fixups,
             ccx=ccx);

    check_block(fcx, body);
    alt (decl.purity) {
        case (ast::pure_fn) {

            // per the previous comment, this just checks that the declared
            // type is bool, and trusts that that's the actual return type.
            if (!ty::type_is_bool(ccx.tcx, fcx.ret_ty)) {
                ccx.tcx.sess.span_fatal(body.span,
                                      "Non-boolean return type in pred");
            }
        }
        case (_) { }
    }

    auto success = writeback::resolve_type_vars_in_block(fcx, body);

    if (success && option::is_some(body.node.expr)) {
        auto tail_expr = option::get(body.node.expr);
        auto tail_expr_ty = expr_ty(ccx.tcx, tail_expr);
        // Have to exclude ty_nil to allow functions to end in
        // while expressions, etc.
        if (!ty::type_is_nil(ccx.tcx, tail_expr_ty)) {
            demand::simple(fcx, tail_expr.span,
                           fcx.ret_ty, tail_expr_ty);
        }
    }
}

fn check_method(&@crate_ctxt ccx, &@ast::method method) {
    auto m = method.node.meth;
    check_fn(ccx, m.decl, m.proto, m.body, method.node.id);
}

fn check_item(@crate_ctxt ccx, &@ast::item it) {
    alt (it.node) {
        case (ast::item_const(_, ?e)) {
            check_const(ccx, it.span, e, it.id);
        }
        case (ast::item_fn(?f, _)) {
            check_fn(ccx, f.decl, f.proto, f.body, it.id);
        }
        case (ast::item_res(?f, ?dtor_id, _, _)) {
            check_fn(ccx, f.decl, f.proto, f.body, dtor_id);
        }
        case (ast::item_obj(?ob, _, _)) {
            // We're entering an object, so gather up the info we need.

            vec::push[obj_info](ccx.obj_infos,
                                rec(obj_fields=ob.fields, this_obj=it.id));
            // Typecheck the methods.

            for (@ast::method method in ob.methods) {
                check_method(ccx, method);
            }
            option::may[@ast::method](bind check_method(ccx, _), ob.dtor);
            // Now remove the info from the stack.

            vec::pop[obj_info](ccx.obj_infos);
        }
        case (_) {/* nothing to do */ }
    }
}

fn check_crate(&ty::ctxt tcx, &@ast::crate crate) {
    collect::collect_item_types(tcx, crate);
    let vec[obj_info] obj_infos = [];

    auto ccx =
        @rec(mutable obj_infos=obj_infos, tcx=tcx);
    auto visit =
        rec(visit_item_pre=bind check_item(ccx, _)
            with walk::default_visitor());
    walk::walk_crate(visit, *crate);
    tcx.sess.abort_if_errors();
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
