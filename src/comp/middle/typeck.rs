import syntax::ast;
import ast::mutability;
import ast::local_def;
import ast::respan;
import ast::spanned;
import syntax::walk;
import syntax::visit;
import metadata::csearch;
import driver::session;
import util::common;
import util::common::*;
import syntax::codemap::span;
import std::map::new_int_hash;
import std::map::new_str_hash;
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
import std::uint;
import std::map;
import std::map::hashmap;
import std::option;
import std::option::none;
import std::option::some;
import std::option::from_maybe;
import std::smallintmap;
import middle::tstate::ann::ts_ann;
import syntax::print::pprust::*;

export check_crate;

type ty_table = hashmap[ast::def_id, ty::t];

// Used for typechecking the methods of an object.
tag obj_info {

    // Regular objects have a node_id at compile time.
    regular_obj(ast::obj_field[], ast::node_id);

    // Anonymous objects only have a type at compile time.  It's optional
    // because not all anonymous objects have a with_obj to attach to.
    anon_obj(ast::obj_field[], option::t[ty::sty]);
}

type crate_ctxt = {mutable obj_infos: obj_info[], tcx: ty::ctxt};

type fn_ctxt =
    // var_bindings, locals, local_names, and next_var_id are shared
    // with any nested functions that capture the environment
    // (and with any functions whose environment is being captured).
    {ret_ty: ty::t,
     purity: ast::purity,
     var_bindings: @ty::unify::var_bindings,
     locals: hashmap[ast::node_id, int],
     local_names: hashmap[ast::node_id, ast::ident],
     next_var_id: @mutable int,
     mutable fixups: ast::node_id[],
     ccx: @crate_ctxt};


// Used for ast_ty_to_ty() below.
type ty_getter = fn(&ast::def_id) -> ty::ty_param_count_and_ty ;

fn lookup_local(fcx: &@fn_ctxt, sp: &span, id: ast::node_id) -> int {
    alt fcx.locals.find(id) {
      some(x) { x }
      _ {
        fcx.ccx.tcx.sess.span_fatal(sp,
                                    "internal error looking up a \
              local var")
      }
    }
}

fn lookup_def(fcx: &@fn_ctxt, sp: &span, id: ast::node_id) -> ast::def {
    alt fcx.ccx.tcx.def_map.find(id) {
      some(x) { x }
      _ {
        fcx.ccx.tcx.sess.span_fatal(sp,
                                    "internal error looking up \
              a definition")
      }
    }
}

// Returns the type parameter count and the type for the given definition.
fn ty_param_count_and_ty_for_def(fcx: &@fn_ctxt, sp: &span, defn: &ast::def)
   -> ty_param_count_and_ty {
    alt defn {
      ast::def_arg(id) {
        assert (fcx.locals.contains_key(id.node));
        let typ = ty::mk_var(fcx.ccx.tcx, lookup_local(fcx, sp, id.node));
        ret {count: 0u, ty: typ};
      }
      ast::def_local(id) {
        assert (fcx.locals.contains_key(id.node));
        let typ = ty::mk_var(fcx.ccx.tcx, lookup_local(fcx, sp, id.node));
        ret {count: 0u, ty: typ};
      }
      ast::def_obj_field(id) {
        assert (fcx.locals.contains_key(id.node));
        let typ = ty::mk_var(fcx.ccx.tcx, lookup_local(fcx, sp, id.node));
        ret {count: 0u, ty: typ};
      }
      ast::def_fn(id, _) { ret ty::lookup_item_type(fcx.ccx.tcx, id); }
      ast::def_native_fn(id) { ret ty::lookup_item_type(fcx.ccx.tcx, id); }
      ast::def_const(id) { ret ty::lookup_item_type(fcx.ccx.tcx, id); }
      ast::def_variant(_, vid) { ret ty::lookup_item_type(fcx.ccx.tcx, vid); }
      ast::def_binding(id) {
        assert (fcx.locals.contains_key(id.node));
        let typ = ty::mk_var(fcx.ccx.tcx, lookup_local(fcx, sp, id.node));
        ret {count: 0u, ty: typ};
      }
      ast::def_mod(_) {
        // Hopefully part of a path.
        // TODO: return a type that's more poisonous, perhaps?

        ret {count: 0u, ty: ty::mk_nil(fcx.ccx.tcx)};
      }
      ast::def_ty(_) {
        fcx.ccx.tcx.sess.span_fatal(sp, "expected value but found type");
      }
      _ {
        // FIXME: handle other names.

        fcx.ccx.tcx.sess.unimpl("definition variant");
      }
    }
}


// Instantiates the given path, which must refer to an item with the given
// number of type parameters and type.
fn instantiate_path(fcx: &@fn_ctxt, pth: &ast::path,
                    tpt: &ty_param_count_and_ty, sp: &span) ->
   ty_param_substs_opt_and_ty {
    let ty_param_count = tpt.count;
    let bind_result =
        bind_params_in_type(sp, fcx.ccx.tcx, bind next_ty_var_id(fcx), tpt.ty,
                            ty_param_count);
    let ty_param_vars = bind_result.ids;
    let ty_substs_opt;
    let ty_substs_len = ivec::len[@ast::ty](pth.node.types);
    if ty_substs_len > 0u {
        let param_var_len = ivec::len(ty_param_vars);
        if param_var_len == 0u {
            fcx.ccx.tcx.sess.span_fatal
                (sp, "this item does not take type parameters");
        } else if (ty_substs_len > param_var_len) {
            fcx.ccx.tcx.sess.span_fatal
                (sp, "too many type parameter provided for this item");
        } else if (ty_substs_len < param_var_len) {
            fcx.ccx.tcx.sess.span_fatal
                (sp, "not enough type parameters provided for this item");
        }
        let ty_substs: ty::t[] = ~[];
        let i = 0u;
        while i < ty_substs_len {
            let ty_var = ty::mk_var(fcx.ccx.tcx, ty_param_vars.(i));
            let ty_subst = ast_ty_to_ty_crate(fcx.ccx, pth.node.types.(i));
            let res_ty = demand::simple(fcx, pth.span, ty_var, ty_subst);
            ty_substs += ~[res_ty];
            i += 1u;
        }
        ty_substs_opt = some[ty::t[]](ty_substs);
        if ty_param_count == 0u {
            fcx.ccx.tcx.sess.span_fatal(sp,
                                        "this item does not take type \
                                      parameters");
            fail;
        }
    } else {
        // We will acquire the type parameters through unification.

        let ty_substs: ty::t[] = ~[];
        let i = 0u;
        while i < ty_param_count {
            ty_substs += ~[ty::mk_var(fcx.ccx.tcx, ty_param_vars.(i))];
            i += 1u;
        }
        ty_substs_opt = some[ty::t[]](ty_substs);
    }
    ret {substs: ty_substs_opt, ty: tpt.ty};
}

fn ast_mode_to_mode(mode: ast::mode) -> ty::mode {
    let ty_mode;
    alt mode {
      ast::val. { ty_mode = mo_val; }
      ast::alias(mut) { ty_mode = mo_alias(mut); }
    }
    ret ty_mode;
}


// Type tests
fn structurally_resolved_type(fcx: &@fn_ctxt, sp: &span, typ: ty::t) ->
   ty::t {
    let r =
        ty::unify::resolve_type_structure(fcx.ccx.tcx, fcx.var_bindings, typ);
    alt r {
      fix_ok(typ_s) { ret typ_s; }
      fix_err(_) {
        fcx.ccx.tcx.sess.span_fatal(sp,
                                    "the type of this value must be \
                                        known in this context");
      }
    }
}


// Returns the one-level-deep structure of the given type.
fn structure_of(fcx: &@fn_ctxt, sp: &span, typ: ty::t) -> ty::sty {
    ret ty::struct(fcx.ccx.tcx, structurally_resolved_type(fcx, sp, typ));
}

// Returns the one-level-deep structure of the given type or none if it
// is not known yet.
fn structure_of_maybe(fcx: &@fn_ctxt, sp: &span, typ: ty::t) ->
   option::t[ty::sty] {
    let r =
        ty::unify::resolve_type_structure(fcx.ccx.tcx, fcx.var_bindings, typ);
    ret alt r {
          fix_ok(typ_s) { some(ty::struct(fcx.ccx.tcx, typ_s)) }
          fix_err(_) { none }
        }
}

fn type_is_integral(fcx: &@fn_ctxt, sp: &span, typ: ty::t) -> bool {
    let typ_s = structurally_resolved_type(fcx, sp, typ);
    ret ty::type_is_integral(fcx.ccx.tcx, typ_s);
}

fn type_is_scalar(fcx: &@fn_ctxt, sp: &span, typ: ty::t) -> bool {
    let typ_s = structurally_resolved_type(fcx, sp, typ);
    ret ty::type_is_scalar(fcx.ccx.tcx, typ_s);
}


// Parses the programmer's textual representation of a type into our internal
// notion of a type. `getter` is a function that returns the type
// corresponding to a definition ID:
fn ast_ty_to_ty(tcx: &ty::ctxt, getter: &ty_getter, ast_ty: &@ast::ty) ->
   ty::t {
    alt tcx.ast_ty_to_ty_cache.find(ast_ty) {
      some(some(ty)) { ret ty; }
      some(none.) {
        tcx.sess.span_fatal(ast_ty.span,
                            "illegal recursive type \
                              insert a tag in the cycle, \
                              if this is desired)");
      }
      none. { }
    } /* go on */

    tcx.ast_ty_to_ty_cache.insert(ast_ty, none[ty::t]);
    fn ast_arg_to_arg(tcx: &ty::ctxt, getter: &ty_getter, arg: &ast::ty_arg)
       -> {mode: ty::mode, ty: ty::t} {
        let ty_mode = ast_mode_to_mode(arg.node.mode);
        ret {mode: ty_mode, ty: ast_ty_to_ty(tcx, getter, arg.node.ty)};
    }
    fn ast_mt_to_mt(tcx: &ty::ctxt, getter: &ty_getter, mt: &ast::mt) ->
       ty::mt {
        ret {ty: ast_ty_to_ty(tcx, getter, mt.ty), mut: mt.mut};
    }
    fn instantiate(tcx: &ty::ctxt, sp: &span, getter: &ty_getter,
                   id: &ast::def_id, args: &(@ast::ty)[]) -> ty::t {
        // TODO: maybe record cname chains so we can do
        // "foo = int" like OCaml?

        let params_opt_and_ty = getter(id);
        if params_opt_and_ty.count == 0u { ret params_opt_and_ty.ty; }
        // The typedef is type-parametric. Do the type substitution.
        //

        let param_bindings: ty::t[] = ~[];
        for ast_ty: @ast::ty  in args {
            param_bindings += ~[ast_ty_to_ty(tcx, getter, ast_ty)];
        }
        if ivec::len(param_bindings) !=
               ty::count_ty_params(tcx, params_opt_and_ty.ty) {
            tcx.sess.span_fatal(sp,
                                "Wrong number of type arguments for a \
                                polymorphic tag");
        }
        let typ =
            ty::substitute_type_params(tcx, param_bindings,
                                       params_opt_and_ty.ty);
        ret typ;
    }
    let typ;
    let cname = none[str];
    alt ast_ty.node {
      ast::ty_nil. { typ = ty::mk_nil(tcx); }
      ast::ty_bot. { typ = ty::mk_bot(tcx); }
      ast::ty_bool. { typ = ty::mk_bool(tcx); }
      ast::ty_int. { typ = ty::mk_int(tcx); }
      ast::ty_uint. { typ = ty::mk_uint(tcx); }
      ast::ty_float. { typ = ty::mk_float(tcx); }
      ast::ty_machine(tm) { typ = ty::mk_mach(tcx, tm); }
      ast::ty_char. { typ = ty::mk_char(tcx); }
      ast::ty_str. { typ = ty::mk_str(tcx); }
      ast::ty_istr. { typ = ty::mk_istr(tcx); }
      ast::ty_box(mt) {
        typ = ty::mk_box(tcx, ast_mt_to_mt(tcx, getter, mt));
      }
      ast::ty_vec(mt) {
        typ = ty::mk_vec(tcx, ast_mt_to_mt(tcx, getter, mt));
      }
      ast::ty_ivec(mt) {
        typ = ty::mk_ivec(tcx, ast_mt_to_mt(tcx, getter, mt));
      }
      ast::ty_ptr(mt) {
        typ = ty::mk_ptr(tcx, ast_mt_to_mt(tcx, getter, mt));
      }
      ast::ty_task. { typ = ty::mk_task(tcx); }
      ast::ty_port(t) {
        typ = ty::mk_port(tcx, ast_ty_to_ty(tcx, getter, t));
      }
      ast::ty_chan(t) {
        typ = ty::mk_chan(tcx, ast_ty_to_ty(tcx, getter, t));
      }
      ast::ty_rec(fields) {
        let flds: field[] = ~[];
        for f: ast::ty_field  in fields {
            let tm = ast_mt_to_mt(tcx, getter, f.node.mt);
            flds += ~[{ident: f.node.ident, mt: tm}];
        }
        typ = ty::mk_rec(tcx, flds);
      }
      ast::ty_fn(proto, inputs, output, cf, constrs) {
        let i = ~[];
        for ta: ast::ty_arg  in inputs {
            i += ~[ast_arg_to_arg(tcx, getter, ta)];
        }
        let out_ty = ast_ty_to_ty(tcx, getter, output);

        let out_constrs = ~[];
        for constr: @ast::constr  in constrs {
            out_constrs += ~[ty::ast_constr_to_constr(tcx, constr)];
        }
        typ = ty::mk_fn(tcx, proto, i, out_ty, cf, out_constrs);
      }
      ast::ty_path(path, id) {
        alt tcx.def_map.find(id) {
          some(ast::def_ty(id)) {
            typ = instantiate(tcx, ast_ty.span, getter, id, path.node.types);
          }
          some(ast::def_native_ty(id)) { typ = getter(id).ty; }
          some(ast::def_ty_arg(id)) { typ = ty::mk_param(tcx, id); }
          some(_) {
            tcx.sess.span_fatal(ast_ty.span,
                                "found type name used as a variable");
          }
          _ {
            tcx.sess.span_fatal(ast_ty.span, "internal error in instantiate");
          }
        }
        cname = some(path_to_str(path));
      }
      ast::ty_obj(meths) {
        let tmeths: ty::method[] = ~[];
        for m: ast::ty_method  in meths {
            let ins = ~[];
            for ta: ast::ty_arg  in m.node.inputs {
                ins += ~[ast_arg_to_arg(tcx, getter, ta)];
            }
            let out = ast_ty_to_ty(tcx, getter, m.node.output);

            let out_constrs = ~[];
            for constr: @ast::constr  in m.node.constrs {
                out_constrs += ~[ty::ast_constr_to_constr(tcx, constr)];
            }
            let new_m: ty::method =
                {proto: m.node.proto,
                 ident: m.node.ident,
                 inputs: ins,
                 output: out,
                 cf: m.node.cf,
                 constrs: out_constrs};
            tmeths += ~[new_m];
        }
        typ = ty::mk_obj(tcx, ty::sort_methods(tmeths));
      }
      ast::ty_constr(t, cs) {
        let out_cs = ~[];
        for constr: @ast::ty_constr  in cs {
            out_cs += ~[ty::ast_constr_to_constr(tcx, constr)];
        }
        typ = ty::mk_constr(tcx, ast_ty_to_ty(tcx, getter, t), out_cs);
      }
    }
    alt cname {
      none. {/* no-op */ }
      some(cname_str) { typ = ty::rename(tcx, typ, cname_str); }
    }
    tcx.ast_ty_to_ty_cache.insert(ast_ty, some(typ));
    ret typ;
}


// A convenience function to use a crate_ctxt to resolve names for
// ast_ty_to_ty.
fn ast_ty_to_ty_crate(ccx: @crate_ctxt, ast_ty: &@ast::ty) -> ty::t {
    fn getter(ccx: @crate_ctxt, id: &ast::def_id) ->
       ty::ty_param_count_and_ty {
        ret ty::lookup_item_type(ccx.tcx, id);
    }
    let f = bind getter(ccx, _);
    ret ast_ty_to_ty(ccx.tcx, f, ast_ty);
}


// Functions that write types into the node type table.
mod write {
    fn inner(ntt: &node_type_table, node_id: ast::node_id,
             tpot: &ty_param_substs_opt_and_ty) {
        smallintmap::insert(*ntt, node_id as uint, tpot);
    }

    // Writes a type parameter count and type pair into the node type table.
    fn ty(tcx: &ty::ctxt, node_id: ast::node_id,
          tpot: &ty_param_substs_opt_and_ty) {
        assert (!ty::type_contains_vars(tcx, tpot.ty));
        inner(tcx.node_types, node_id, tpot);
    }

    // Writes a type parameter count and type pair into the node type table.
    // This function allows for the possibility of type variables, which will
    // be rewritten later during the fixup phase.
    fn ty_fixup(fcx: @fn_ctxt, node_id: ast::node_id,
                tpot: &ty_param_substs_opt_and_ty) {
        inner(fcx.ccx.tcx.node_types, node_id, tpot);
        if ty::type_contains_vars(fcx.ccx.tcx, tpot.ty) {
            fcx.fixups += ~[node_id];
        }
    }

    // Writes a type with no type parameters into the node type table.
    fn ty_only(tcx: &ty::ctxt, node_id: ast::node_id, typ: ty::t) {
        ty(tcx, node_id, {substs: none[ty::t[]], ty: typ});
    }

    // Writes a type with no type parameters into the node type table. This
    // function allows for the possibility of type variables.
    fn ty_only_fixup(fcx: @fn_ctxt, node_id: ast::node_id, typ: ty::t) {
        ret ty_fixup(fcx, node_id, {substs: none[ty::t[]], ty: typ});
    }

    // Writes a nil type into the node type table.
    fn nil_ty(tcx: &ty::ctxt, node_id: ast::node_id) {
        ret ty(tcx, node_id, {substs: none[ty::t[]], ty: ty::mk_nil(tcx)});
    }

    // Writes the bottom type into the node type table.
    fn bot_ty(tcx: &ty::ctxt, node_id: ast::node_id) {
        ret ty(tcx, node_id, {substs: none[ty::t[]], ty: ty::mk_bot(tcx)});
    }
}

// Determine the proto for a fn type given the proto for its associated
// code. This is needed because fn and lambda have fn type while iter
// has iter type and block has block type. This may end up changing.
fn proto_to_ty_proto(proto: &ast::proto) -> ast::proto {
    ret alt proto {
          ast::proto_iter. | ast::proto_block. { proto }
          _ { ast::proto_fn }
        };
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
    type ctxt = {tcx: ty::ctxt};

    fn mk_ty_params(cx: &@ctxt, n: uint) -> ty::t[] {
        let tps = ~[];
        let i = 0u;
        while i < n { tps += ~[ty::mk_param(cx.tcx, i)]; i += 1u; }
        ret tps;
    }
    fn ty_of_fn_decl(cx: &@ctxt, convert: &fn(&@ast::ty) -> ty::t ,
                     ty_of_arg: &fn(&ast::arg) -> arg , decl: &ast::fn_decl,
                     proto: ast::proto, ty_params: &ast::ty_param[],
                     def_id: &option::t[ast::def_id]) ->
       ty::ty_param_count_and_ty {
        let input_tys = ~[];
        for a: ast::arg  in decl.inputs { input_tys += ~[ty_of_arg(a)]; }
        let output_ty = convert(decl.output);

        let out_constrs = ~[];
        for constr: @ast::constr  in decl.constraints {
            out_constrs += ~[ty::ast_constr_to_constr(cx.tcx, constr)];
        }
        let t_fn =
            ty::mk_fn(cx.tcx, proto_to_ty_proto(proto), input_tys, output_ty,
                      decl.cf, out_constrs);
        let ty_param_count = ivec::len[ast::ty_param](ty_params);
        let tpt = {count: ty_param_count, ty: t_fn};
        alt def_id { some(did) { cx.tcx.tcache.insert(did, tpt); } _ { } }
        ret tpt;
    }
    fn ty_of_native_fn_decl(cx: &@ctxt, convert: &fn(&@ast::ty) -> ty::t ,
                            ty_of_arg: &fn(&ast::arg) -> arg ,
                            decl: &ast::fn_decl, abi: ast::native_abi,
                            ty_params: &ast::ty_param[], def_id: &ast::def_id)
       -> ty::ty_param_count_and_ty {
        let input_tys = ~[];
        for a: ast::arg  in decl.inputs { input_tys += ~[ty_of_arg(a)]; }
        let output_ty = convert(decl.output);

        let t_fn = ty::mk_native_fn(cx.tcx, abi, input_tys, output_ty);
        let ty_param_count = ivec::len[ast::ty_param](ty_params);
        let tpt = {count: ty_param_count, ty: t_fn};
        cx.tcx.tcache.insert(def_id, tpt);
        ret tpt;
    }
    fn getter(cx: @ctxt, id: &ast::def_id) -> ty::ty_param_count_and_ty {
        if id.crate != ast::local_crate {
            // This is a type we need to load in from the crate reader.
            ret csearch::get_type(cx.tcx, id);
        }
        let it = cx.tcx.items.find(id.node);
        let tpt;
        alt it {
          some(ast_map::node_item(item)) { tpt = ty_of_item(cx, item); }
          some(ast_map::node_native_item(native_item)) {
            tpt = ty_of_native_item(cx, native_item, ast::native_abi_cdecl);
          }
          _ { cx.tcx.sess.fatal("internal error " + std::int::str(id.node)); }
        }
        ret tpt;
    }
    fn ty_of_arg(cx: @ctxt, a: &ast::arg) -> ty::arg {
        let ty_mode = ast_mode_to_mode(a.mode);
        let f = bind getter(cx, _);
        let tt = ast_ty_to_ty(cx.tcx, f, a.ty);
        if ty::type_has_dynamic_size(cx.tcx, tt) {
            alt ty_mode {
              mo_val. {
                cx.tcx.sess.span_fatal(a.ty.span,
                                       "Dynamically sized arguments \
                                          must be passed by alias");
              }
              _ { }
            }
        }
        ret {mode: ty_mode, ty: tt};
    }
    fn ty_of_method(cx: @ctxt, m: &@ast::method) -> ty::method {
        let get = bind getter(cx, _);
        let convert = bind ast_ty_to_ty(cx.tcx, get, _);

        let inputs = ~[];
        for a: ast::arg  in m.node.meth.decl.inputs {
            inputs += ~[ty_of_arg(cx, a)];
        }

        let output = convert(m.node.meth.decl.output);

        let out_constrs = ~[];
        for constr: @ast::constr  in m.node.meth.decl.constraints {
            out_constrs += ~[ty::ast_constr_to_constr(cx.tcx, constr)];
        }
        ret {proto: proto_to_ty_proto(m.node.meth.proto),
             ident: m.node.ident,
             inputs: inputs,
             output: output,
             cf: m.node.meth.decl.cf,
             constrs: out_constrs};
    }
    fn ty_of_obj(cx: @ctxt, id: &ast::ident, ob: &ast::_obj,
                 ty_params: &ast::ty_param[]) -> ty::ty_param_count_and_ty {
        let methods = get_obj_method_types(cx, ob);
        let t_obj = ty::mk_obj(cx.tcx, ty::sort_methods(methods));
        t_obj = ty::rename(cx.tcx, t_obj, id);
        ret {count: ivec::len(ty_params), ty: t_obj};
    }
    fn ty_of_obj_ctor(cx: @ctxt, id: &ast::ident, ob: &ast::_obj,
                      ctor_id: ast::node_id, ty_params: &ast::ty_param[]) ->
       ty::ty_param_count_and_ty {
        let t_obj = ty_of_obj(cx, id, ob, ty_params);

        let t_inputs: arg[] = ~[];
        for f: ast::obj_field  in ob.fields {
            let g = bind getter(cx, _);
            let t_field = ast_ty_to_ty(cx.tcx, g, f.ty);
            t_inputs += ~[{mode: ty::mo_alias(false), ty: t_field}];
        }

        let t_fn =
            ty::mk_fn(cx.tcx, ast::proto_fn, t_inputs, t_obj.ty, ast::return,
                      ~[]);
        let tpt = {count: t_obj.count, ty: t_fn};
        cx.tcx.tcache.insert(local_def(ctor_id), tpt);
        ret tpt;
    }
    fn ty_of_item(cx: &@ctxt, it: &@ast::item) -> ty::ty_param_count_and_ty {
        let get = bind getter(cx, _);
        let convert = bind ast_ty_to_ty(cx.tcx, get, _);
        alt it.node {
          ast::item_const(t, _) {
            let typ = convert(t);
            let tpt = {count: 0u, ty: typ};
            cx.tcx.tcache.insert(local_def(it.id), tpt);
            ret tpt;
          }
          ast::item_fn(fn_info, tps) {
            let f = bind ty_of_arg(cx, _);
            ret ty_of_fn_decl(cx, convert, f, fn_info.decl, fn_info.proto,
                              tps, some(local_def(it.id)));
          }
          ast::item_obj(ob, tps, _) {
            let t_obj = ty_of_obj(cx, it.ident, ob, tps);
            cx.tcx.tcache.insert(local_def(it.id), t_obj);
            ret t_obj;
          }
          ast::item_ty(t, tps) {
            alt cx.tcx.tcache.find(local_def(it.id)) {
              some(tpt) { ret tpt; }
              none. { }
            }
            // Tell ast_ty_to_ty() that we want to perform a recursive
            // call to resolve any named types.

            let typ = convert(t);
            let ty_param_count = ivec::len[ast::ty_param](tps);
            let tpt = {count: ty_param_count, ty: typ};
            cx.tcx.tcache.insert(local_def(it.id), tpt);
            ret tpt;
          }
          ast::item_res(f, _, tps, _) {
            let t_arg = ty_of_arg(cx, f.decl.inputs.(0));
            let t_res =
                {count: ivec::len(tps),
                 ty:
                     ty::mk_res(cx.tcx, local_def(it.id), t_arg.ty,
                                mk_ty_params(cx, ivec::len(tps)))};
            cx.tcx.tcache.insert(local_def(it.id), t_res);
            ret t_res;
          }
          ast::item_tag(_, tps) {
            // Create a new generic polytype.

            let ty_param_count = ivec::len[ast::ty_param](tps);

            let subtys: ty::t[] = mk_ty_params(cx, ty_param_count);
            let t = ty::mk_tag(cx.tcx, local_def(it.id), subtys);
            let tpt = {count: ty_param_count, ty: t};
            cx.tcx.tcache.insert(local_def(it.id), tpt);
            ret tpt;
          }
          ast::item_mod(_) { fail; }
          ast::item_native_mod(_) { fail; }
        }
    }
    fn ty_of_native_item(cx: &@ctxt, it: &@ast::native_item,
                         abi: ast::native_abi) -> ty::ty_param_count_and_ty {
        alt it.node {
          ast::native_item_fn(_, fn_decl, params) {
            let get = bind getter(cx, _);
            let convert = bind ast_ty_to_ty(cx.tcx, get, _);
            let f = bind ty_of_arg(cx, _);
            ret ty_of_native_fn_decl(cx, convert, f, fn_decl, abi, params,
                                     ast::local_def(it.id));
          }
          ast::native_item_ty. {
            alt cx.tcx.tcache.find(local_def(it.id)) {
              some(tpt) { ret tpt; }
              none. { }
            }
            let t = ty::mk_native(cx.tcx, ast::local_def(it.id));
            let tpt = {count: 0u, ty: t};
            cx.tcx.tcache.insert(local_def(it.id), tpt);
            ret tpt;
          }
        }
    }
    fn get_tag_variant_types(cx: &@ctxt, tag_id: &ast::def_id,
                             variants: &ast::variant[],
                             ty_params: &ast::ty_param[]) {
        // Create a set of parameter types shared among all the variants.

        let ty_param_count = ivec::len[ast::ty_param](ty_params);
        let ty_param_tys: ty::t[] = mk_ty_params(cx, ty_param_count);
        for variant: ast::variant  in variants {
            // Nullary tag constructors get turned into constants; n-ary tag
            // constructors get turned into functions.

            let result_ty;
            if ivec::len[ast::variant_arg](variant.node.args) == 0u {
                result_ty = ty::mk_tag(cx.tcx, tag_id, ty_param_tys);
            } else {
                // As above, tell ast_ty_to_ty() that trans_ty_item_to_ty()
                // should be called to resolve named types.

                let f = bind getter(cx, _);
                let args: arg[] = ~[];
                for va: ast::variant_arg  in variant.node.args {
                    let arg_ty = ast_ty_to_ty(cx.tcx, f, va.ty);
                    args += ~[{mode: ty::mo_alias(false), ty: arg_ty}];
                }
                let tag_t = ty::mk_tag(cx.tcx, tag_id, ty_param_tys);
                // FIXME: this will be different for constrained types
                result_ty =
                    ty::mk_fn(cx.tcx, ast::proto_fn, args, tag_t, ast::return,
                              ~[]);
            }
            let tpt = {count: ty_param_count, ty: result_ty};
            cx.tcx.tcache.insert(local_def(variant.node.id), tpt);
            write::ty_only(cx.tcx, variant.node.id, result_ty);
        }
    }
    fn get_obj_method_types(cx: &@ctxt, object: &ast::_obj) -> ty::method[] {
        let meths = ~[];
        for m: @ast::method  in object.methods {
            meths += ~[ty_of_method(cx, m)];
        }
        ret meths;
    }
    fn convert(cx: @ctxt, abi: @mutable option::t[ast::native_abi],
               it: &@ast::item) {
        alt it.node {
          ast::item_mod(_) {
            // ignore item_mod, it has no type.

          }
          ast::item_native_mod(native_mod) {
            // Propagate the native ABI down to convert_native() below,
            // but otherwise do nothing, as native modules have no types.

            *abi = some[ast::native_abi](native_mod.abi);
          }
          ast::item_tag(variants, ty_params) {
            let tpt = ty_of_item(cx, it);
            write::ty_only(cx.tcx, it.id, tpt.ty);
            get_tag_variant_types(cx, local_def(it.id), variants, ty_params);
          }
          ast::item_obj(object, ty_params, ctor_id) {
            // Now we need to call ty_of_obj_ctor(); this is the type that
            // we write into the table for this item.

            ty_of_item(cx, it);

            let tpt =
                ty_of_obj_ctor(cx, it.ident, object, ctor_id, ty_params);
            write::ty_only(cx.tcx, ctor_id, tpt.ty);
            // Write the methods into the type table.
            //
            // FIXME: Inefficient; this ends up calling
            // get_obj_method_types() twice. (The first time was above in
            // ty_of_obj().)

            let method_types = get_obj_method_types(cx, object);
            let i = 0u;
            while i < ivec::len[@ast::method](object.methods) {
                write::ty_only(cx.tcx, object.methods.(i).node.id,
                               ty::method_ty_to_fn_ty(cx.tcx,
                                                      method_types.(i)));
                i += 1u;
            }
            // Write in the types of the object fields.
            //
            // FIXME: We want to use uint::range() here, but that causes
            // an assertion in trans.

            let args = ty::ty_fn_args(cx.tcx, tpt.ty);
            i = 0u;
            while i < ivec::len[ty::arg](args) {
                let fld = object.fields.(i);
                write::ty_only(cx.tcx, fld.id, args.(i).ty);
                i += 1u;
            }


            // Finally, write in the type of the destructor.
            alt object.dtor {
              none. {/* nothing to do */ }
              some(m) {
                let t =
                    ty::mk_fn(cx.tcx, ast::proto_fn, ~[], ty::mk_nil(cx.tcx),
                              ast::return, ~[]);
                write::ty_only(cx.tcx, m.node.id, t);
              }
            }
          }
          ast::item_res(f, dtor_id, tps, ctor_id) {
            let t_arg = ty_of_arg(cx, f.decl.inputs.(0));
            let t_res =
                ty::mk_res(cx.tcx, local_def(it.id), t_arg.ty,
                           mk_ty_params(cx, ivec::len(tps)));
            let t_ctor =
                ty::mk_fn(cx.tcx, ast::proto_fn, ~[t_arg], t_res, ast::return,
                          ~[]);
            let t_dtor =
                ty::mk_fn(cx.tcx, ast::proto_fn, ~[t_arg], ty::mk_nil(cx.tcx),
                          ast::return, ~[]);
            write::ty_only(cx.tcx, it.id, t_res);
            write::ty_only(cx.tcx, ctor_id, t_ctor);
            cx.tcx.tcache.insert(local_def(ctor_id),
                                 {count: ivec::len(tps), ty: t_ctor});
            write::ty_only(cx.tcx, dtor_id, t_dtor);
          }
          _ {
            // This call populates the type cache with the converted type
            // of the item in passing. All we have to do here is to write
            // it into the node type table.

            let tpt = ty_of_item(cx, it);
            write::ty_only(cx.tcx, it.id, tpt.ty);
          }
        }
    }
    fn convert_native(cx: @ctxt, abi: @mutable option::t[ast::native_abi],
                      i: &@ast::native_item) {
        // As above, this call populates the type table with the converted
        // type of the native item. We simply write it into the node type
        // table.

        let tpt =
            ty_of_native_item(cx, i, option::get[ast::native_abi]({ *abi }));
        alt i.node {
          ast::native_item_ty. {
            // FIXME: Native types have no annotation. Should they? --pcw

          }
          ast::native_item_fn(_, _, _) {
            write::ty_only(cx.tcx, i.id, tpt.ty);
          }
        }
    }
    fn collect_item_types(tcx: &ty::ctxt, crate: &@ast::crate) {
        // We have to propagate the surrounding ABI to the native items
        // contained within the native module.

        let abi = @mutable none[ast::native_abi];
        let cx = @{tcx: tcx};
        let visit =
            visit::mk_simple_visitor
            (@{visit_item: bind convert(cx, abi, _),
               visit_native_item: bind convert_native(cx, abi, _)
               with *visit::default_simple_visitor()});
        visit::visit_crate(*crate, (), visit);
    }
}


// Type unification

// TODO: rename to just "unify"
mod unify {
    fn simple(fcx: &@fn_ctxt, expected: &ty::t, actual: &ty::t) ->
       ty::unify::result {
        ret ty::unify::unify(expected, actual, fcx.var_bindings, fcx.ccx.tcx);
    }
}

tag autoderef_kind { AUTODEREF_OK; NO_AUTODEREF; AUTODEREF_BLOCK_COERCE; }

// FIXME This is almost a duplicate of ty::type_autoderef, with structure_of
// instead of ty::struct.
fn do_autoderef(fcx: &@fn_ctxt, sp: &span, t: &ty::t) -> ty::t {
    let t1 = t;
    while true {
        alt structure_of(fcx, sp, t1) {
          ty::ty_box(inner) { t1 = inner.ty; }
          ty::ty_res(_, inner, tps) {
            t1 = ty::substitute_type_params(fcx.ccx.tcx, tps, inner);
          }
          ty::ty_tag(did, tps) {
            let variants = ty::tag_variants(fcx.ccx.tcx, did);
            if ivec::len(variants) != 1u || ivec::len(variants.(0).args) != 1u
               {
                ret t1;
            }
            t1 =
                ty::substitute_type_params(fcx.ccx.tcx, tps,
                                           variants.(0).args.(0));
          }
          _ { ret t1; }
        }
    }
    fail;
}

fn add_boxes(ccx: &@crate_ctxt, n: uint, t: &ty::t) -> ty::t {
    let t1 = t;
    while n != 0u { t1 = ty::mk_imm_box(ccx.tcx, t1); n -= 1u; }
    ret t1;
}

fn count_boxes(fcx: &@fn_ctxt, sp: &span, t: &ty::t) -> uint {
    let n = 0u;
    let t1 = t;
    while true {
        alt structure_of(fcx, sp, t1) {
          ty::ty_box(inner) { n += 1u; t1 = inner.ty; }
          _ { ret n; }
        }
    }
    fail;
}

fn do_fn_block_coerce(fcx: &@fn_ctxt, sp: &span, actual: &ty::t,
                      expected: &ty::t) -> ty::t {

    // fns can be silently coerced to blocks when being used as
    // function call or bind arguments, but not the reverse.
    // If our actual type is a fn and our expected type is a block,
    // build up a new expected type that is identical to the old one
    // except for its proto. If we don't know the expected or actual
    // types, that's fine, but we can't do the coercion.
    ret alt structure_of_maybe(fcx, sp, actual) {
          some(ty::ty_fn(ast::proto_fn., args, ret_ty, cf, constrs)) {
            alt structure_of_maybe(fcx, sp, expected) {
              some(ty::ty_fn(ast::proto_block., _, _, _, _)) {
                ty::mk_fn(fcx.ccx.tcx, ast::proto_block, args, ret_ty, cf,
                          constrs)
              }
              _ { actual }
            }
          }
          _ { actual }
        }
}


fn resolve_type_vars_if_possible(fcx: &@fn_ctxt, typ: ty::t) -> ty::t {
    alt ty::unify::fixup_vars(fcx.ccx.tcx, fcx.var_bindings, typ) {
      fix_ok(new_type) { ret new_type; }
      fix_err(_) { ret typ; }
    }
}


// Demands - procedures that require that two types unify and emit an error
// message if they don't.
type ty_param_substs_and_ty = {substs: ty::t[], ty: ty::t};

mod demand {
    fn simple(fcx: &@fn_ctxt, sp: &span, expected: &ty::t, actual: &ty::t) ->
       ty::t {
        ret full(fcx, sp, expected, actual, ~[], NO_AUTODEREF).ty;
    }
    fn autoderef(fcx: &@fn_ctxt, sp: &span, expected: &ty::t, actual: &ty::t,
                 adk: autoderef_kind) -> ty::t {
        ret full(fcx, sp, expected, actual, ~[], adk).ty;
    }

    // Requires that the two types unify, and prints an error message if they
    // don't. Returns the unified type and the type parameter substitutions.
    fn full(fcx: &@fn_ctxt, sp: &span, expected: &ty::t, actual: &ty::t,
            ty_param_substs_0: &ty::t[], adk: autoderef_kind) ->
       ty_param_substs_and_ty {
        let expected_1 = expected;
        let actual_1 = actual;
        let implicit_boxes = 0u;
        if adk == AUTODEREF_OK {
            expected_1 = do_autoderef(fcx, sp, expected_1);
            actual_1 = do_autoderef(fcx, sp, actual_1);
            implicit_boxes = count_boxes(fcx, sp, actual);
        } else if (adk == AUTODEREF_BLOCK_COERCE) {
            actual_1 = do_fn_block_coerce(fcx, sp, actual, expected);
        }
        let ty_param_substs: ty::t[mutable ] = ~[mutable ];
        let ty_param_subst_var_ids: int[] = ~[];
        for ty_param_subst: ty::t  in ty_param_substs_0 {
            // Generate a type variable and unify it with the type parameter
            // substitution. We will then pull out these type variables.

            let t_0 = next_ty_var(fcx);
            ty_param_substs += ~[mutable t_0];
            ty_param_subst_var_ids += ~[ty::ty_var_id(fcx.ccx.tcx, t_0)];
            simple(fcx, sp, ty_param_subst, t_0);
        }

        fn mk_result(fcx: &@fn_ctxt, result_ty: &ty::t,
                     ty_param_subst_var_ids: &int[], implicit_boxes: uint) ->
           ty_param_substs_and_ty {
            let result_ty_param_substs: ty::t[] = ~[];
            for var_id: int  in ty_param_subst_var_ids {
                let tp_subst = ty::mk_var(fcx.ccx.tcx, var_id);
                result_ty_param_substs += ~[tp_subst];
            }
            ret {substs: result_ty_param_substs,
                 ty: add_boxes(fcx.ccx, implicit_boxes, result_ty)};
        }


        alt unify::simple(fcx, expected_1, actual_1) {
          ures_ok(t) {
            ret mk_result(fcx, t, ty_param_subst_var_ids, implicit_boxes);
          }
          ures_err(err) {
            let e_err = resolve_type_vars_if_possible(fcx, expected_1);
            let a_err = resolve_type_vars_if_possible(fcx, actual_1);
            fcx.ccx.tcx.sess.span_err(sp,
                                      "mismatched types: expected " +
                                          ty_to_str(fcx.ccx.tcx, e_err) +
                                          " but found " +
                                          ty_to_str(fcx.ccx.tcx, a_err) + " ("
                                          + ty::type_err_to_str(err) + ")");
            ret mk_result(fcx, expected_1, ty_param_subst_var_ids,
                          implicit_boxes);
          }
        }
    }
}


// Returns true if the two types unify and false if they don't.
fn are_compatible(fcx: &@fn_ctxt, expected: &ty::t, actual: &ty::t) -> bool {
    alt unify::simple(fcx, expected, actual) {
      ures_ok(_) { ret true; }
      ures_err(_) { ret false; }
    }
}


// Returns the types of the arguments to a tag variant.
fn variant_arg_types(ccx: &@crate_ctxt, sp: &span, vid: &ast::def_id,
                     tag_ty_params: &ty::t[]) -> ty::t[] {
    let result: ty::t[] = ~[];
    let tpt = ty::lookup_item_type(ccx.tcx, vid);
    alt ty::struct(ccx.tcx, tpt.ty) {
      ty::ty_fn(_, ins, _, _, _) {


        // N-ary variant.
        for arg: ty::arg  in ins {
            let arg_ty =
                ty::substitute_type_params(ccx.tcx, tag_ty_params, arg.ty);
            result += ~[arg_ty];
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
//
// TODO: inefficient since not all types have vars in them. It would be better
// to maintain a list of fixups.
mod writeback {

    export resolve_type_vars_in_block;

    fn resolve_type_vars_in_type(fcx: &@fn_ctxt, sp: &span, typ: ty::t) ->
       option::t[ty::t] {
        if !ty::type_contains_vars(fcx.ccx.tcx, typ) { ret some(typ); }
        alt ty::unify::fixup_vars(fcx.ccx.tcx, fcx.var_bindings, typ) {
          fix_ok(new_type) { ret some(new_type); }
          fix_err(vid) {
            fcx.ccx.tcx.sess.span_err(sp,
                                      "cannot determine a type \
                                           for this expression");
            ret none;
          }
        }
    }
    fn resolve_type_vars_for_node(wbcx: &@wb_ctxt, sp: &span,
                                  id: ast::node_id) {
        let fcx = wbcx.fcx;
        let tpot = ty::node_id_to_ty_param_substs_opt_and_ty(fcx.ccx.tcx, id);
        let new_ty =
            alt resolve_type_vars_in_type(fcx, sp, tpot.ty) {
              some(t) { t }
              none. { wbcx.success = false; ret }
            };
        let new_substs_opt;
        alt tpot.substs {
          none[ty::t[]]. { new_substs_opt = none[ty::t[]]; }
          some[ty::t[]](substs) {
            let new_substs: ty::t[] = ~[];
            for subst: ty::t  in substs {
                alt resolve_type_vars_in_type(fcx, sp, subst) {
                  some(t) { new_substs += ~[t]; }
                  none. { wbcx.success = false; ret; }
                }
            }
            new_substs_opt = some[ty::t[]](new_substs);
          }
        }
        write::ty(fcx.ccx.tcx, id, {substs: new_substs_opt, ty: new_ty});
    }

    type wb_ctxt =
        // A flag to ignore contained items and lambdas
        // As soon as we hit an error we have to stop resolving
        // the entire function
        {fcx: @fn_ctxt, mutable ignore: bool, mutable success: bool};

    fn visit_stmt_pre(wbcx: @wb_ctxt, s: &@ast::stmt) {
        resolve_type_vars_for_node(wbcx, s.span, ty::stmt_node_id(s));
    }
    fn visit_expr_pre(wbcx: @wb_ctxt, e: &@ast::expr) {
        resolve_type_vars_for_node(wbcx, e.span, e.id);
    }
    fn visit_block_pre(wbcx: @wb_ctxt, b: &ast::blk) {
        resolve_type_vars_for_node(wbcx, b.span, b.node.id);
    }
    fn visit_pat_pre(wbcx: @wb_ctxt, p: &@ast::pat) {
        resolve_type_vars_for_node(wbcx, p.span, p.id);
    }
    fn visit_local_pre(wbcx: @wb_ctxt, l: &@ast::local) {
        let var_id = lookup_local(wbcx.fcx, l.span, l.node.id);
        let fix_rslt =
            ty::unify::resolve_type_var(wbcx.fcx.ccx.tcx,
                                        wbcx.fcx.var_bindings, var_id);
        alt fix_rslt {
          fix_ok(lty) { write::ty_only(wbcx.fcx.ccx.tcx, l.node.id, lty); }
          fix_err(_) {
            wbcx.fcx.ccx.tcx.sess.span_err(l.span,
                                           "cannot determine a type \
                                                for this local variable");
            wbcx.success = false;
          }
        }
    }
    fn visit_item_pre(wbcx: @wb_ctxt, item: &@ast::item) {
        wbcx.ignore = true;
    }
    fn visit_item_post(wbcx: @wb_ctxt, item: &@ast::item) {
        wbcx.ignore = false;
    }
    fn visit_fn_pre(wbcx: @wb_ctxt, f: &ast::_fn, tps: &ast::ty_param[],
                    sp: &span, i: &ast::fn_ident, d: ast::node_id) {
        wbcx.ignore = true;
    }
    fn visit_fn_post(wbcx: @wb_ctxt, f: &ast::_fn, tps: &ast::ty_param[],
                     sp: &span, i: &ast::fn_ident, d: ast::node_id) {
        wbcx.ignore = false;
    }
    fn keep_going(wbcx: @wb_ctxt) -> bool { !wbcx.ignore && wbcx.success }

    fn resolve_type_vars_in_block(fcx: &@fn_ctxt, blk: &ast::blk) -> bool {
        let wbcx = @{fcx: fcx, mutable ignore: false, mutable success: true};
        let visit =
            {keep_going: bind keep_going(wbcx),
             visit_item_pre: bind visit_item_pre(wbcx, _),
             visit_item_post: bind visit_item_post(wbcx, _),
             visit_fn_pre: bind visit_fn_pre(wbcx, _, _, _, _, _),
             visit_fn_post: bind visit_fn_post(wbcx, _, _, _, _, _),
             visit_stmt_pre: bind visit_stmt_pre(wbcx, _),
             visit_expr_pre: bind visit_expr_pre(wbcx, _),
             visit_block_pre: bind visit_block_pre(wbcx, _),
             visit_pat_pre: bind visit_pat_pre(wbcx, _),
             visit_local_pre: bind visit_local_pre(wbcx, _)
                with walk::default_visitor()};
        walk::walk_block(visit, blk);
        ret wbcx.success;
    }
}


// Local variable gathering. We gather up all locals and create variable IDs
// for them before typechecking the function.
type gather_result =
    {var_bindings: @ty::unify::var_bindings,
     locals: hashmap[ast::node_id, int],
     local_names: hashmap[ast::node_id, ast::ident],
     next_var_id: @mutable int};

// Used only as a helper for check_fn.
fn gather_locals(ccx: &@crate_ctxt, f: &ast::_fn, id: &ast::node_id,
                 old_fcx: &option::t[@fn_ctxt]) -> gather_result {
    fn next_var_id(nvi: @mutable int) -> int {
        let rv = *nvi;
        *nvi += 1;
        ret rv;
    }
    fn assign(tcx: &ty::ctxt, var_bindings: &@ty::unify::var_bindings,
              locals: &hashmap[ast::node_id, int],
              local_names: &hashmap[ast::node_id, ast::ident],
              nvi: @mutable int, nid: ast::node_id, ident: &ast::ident,
              ty_opt: option::t[ty::t]) {
        let var_id = next_var_id(nvi);
        locals.insert(nid, var_id);
        local_names.insert(nid, ident);
        alt ty_opt {
          none[ty::t]. {/* nothing to do */ }
          some[ty::t](typ) {
            ty::unify::unify(ty::mk_var(tcx, var_id), typ, var_bindings, tcx);
          }
        }
    }

    let vb;
    let locals;
    let local_names;
    let nvi;
    alt old_fcx {
      none. {
        vb = ty::unify::mk_var_bindings();
        locals = new_int_hash[int]();
        local_names = new_int_hash[ast::ident]();
        nvi = @mutable 0;
      }
      some(fcx) {
        vb = fcx.var_bindings;
        locals = fcx.locals;
        local_names = fcx.local_names;
        nvi = fcx.next_var_id;
      }
    }

    // Add object fields, if any.
    let obj_fields = ~[];
    alt get_obj_info(ccx) {
      some(oinfo) {
        alt oinfo {
          regular_obj(ofs, _) { obj_fields = ofs; }
          anon_obj(ofs, _) { obj_fields = ofs; }
        }
      }
      none. {/* no fields */ }
    }
    for f: ast::obj_field  in obj_fields {
        let field_ty = ty::node_id_to_type(ccx.tcx, f.id);
        assign(ccx.tcx, vb, locals, local_names, nvi, f.id, f.ident,
               some(field_ty));
    }

    // Add formal parameters.
    let args = ty::ty_fn_args(ccx.tcx, ty::node_id_to_type(ccx.tcx, id));
    let i = 0u;
    for arg: ty::arg  in args {
        assign(ccx.tcx, vb, locals, local_names, nvi, f.decl.inputs.(i).id,
               f.decl.inputs.(i).ident, some[ty::t](arg.ty));
        i += 1u;
    }

    // Add explicitly-declared locals.
    fn visit_local(ccx: @crate_ctxt, vb: @ty::unify::var_bindings,
                   locals: hashmap[ast::node_id, int],
                   local_names: hashmap[ast::node_id, ast::ident],
                   nvi: @mutable int, local: &@ast::local, e: &(),
                   v: &visit::vt[()]) {
        alt local.node.ty {
          none. {
            // Auto slot.
            assign(ccx.tcx, vb, locals, local_names, nvi, local.node.id,
                   local.node.ident, none);
          }
          some(ast_ty) {
            // Explicitly typed slot.
            let local_ty = ast_ty_to_ty_crate(ccx, ast_ty);
            assign(ccx.tcx, vb, locals, local_names, nvi, local.node.id,
                   local.node.ident, some[ty::t](local_ty));
          }
        }
        visit::visit_local(local, e, v);
    }

    // Add pattern bindings.
    fn visit_pat(ccx: @crate_ctxt, vb: @ty::unify::var_bindings,
                 locals: hashmap[ast::node_id, int],
                 local_names: hashmap[ast::node_id, ast::ident],
                 nvi: @mutable int, p: &@ast::pat, e: &(),
                 v: &visit::vt[()]) {
        alt p.node {
          ast::pat_bind(ident) {
            assign(ccx.tcx, vb, locals, local_names, nvi, p.id, ident, none);
          }
          _ {/* no-op */ }
        }
        visit::visit_pat(p, e, v);
    }

    // Don't descend into fns and items
    fn visit_fn[E](f: &ast::_fn, tp: &ast::ty_param[], sp: &span,
                   i: &ast::fn_ident, id: ast::node_id, e: &E,
                   v: &visit::vt[E]) {
    }
    fn visit_item[E](i: &@ast::item, e: &E, v: &visit::vt[E]) { }

    let visit =
        @{visit_local:
              bind visit_local(ccx, vb, locals, local_names, nvi, _, _, _),
          visit_pat:
              bind visit_pat(ccx, vb, locals, local_names, nvi, _, _, _),
          visit_fn: visit_fn,
          visit_item: visit_item with *visit::default_visitor()};
    visit::visit_block(f.body, (), visit::mk_vt(visit));
    ret {var_bindings: vb,
         locals: locals,
         local_names: local_names,
         next_var_id: nvi};
}

// AST fragment checking
fn check_lit(ccx: @crate_ctxt, lit: &@ast::lit) -> ty::t {
    alt lit.node {
      ast::lit_str(_, ast::sk_rc.) { ret ty::mk_str(ccx.tcx); }
      ast::lit_str(_, ast::sk_unique.) { ret ty::mk_istr(ccx.tcx); }
      ast::lit_char(_) { ret ty::mk_char(ccx.tcx); }
      ast::lit_int(_) { ret ty::mk_int(ccx.tcx); }
      ast::lit_float(_) { ret ty::mk_float(ccx.tcx); }
      ast::lit_mach_float(tm, _) { ret ty::mk_mach(ccx.tcx, tm); }
      ast::lit_uint(_) { ret ty::mk_uint(ccx.tcx); }
      ast::lit_mach_int(tm, _) { ret ty::mk_mach(ccx.tcx, tm); }
      ast::lit_nil. { ret ty::mk_nil(ccx.tcx); }
      ast::lit_bool(_) { ret ty::mk_bool(ccx.tcx); }
    }
}

// Pattern checking is top-down rather than bottom-up so that bindings get
// their types immediately.
fn check_pat(fcx: &@fn_ctxt, map: &ast::pat_id_map, pat: &@ast::pat,
             expected: ty::t) {
    alt pat.node {
      ast::pat_wild. { write::ty_only_fixup(fcx, pat.id, expected); }
      ast::pat_lit(lt) {
        let typ = check_lit(fcx.ccx, lt);
        typ = demand::simple(fcx, pat.span, expected, typ);
        write::ty_only_fixup(fcx, pat.id, typ);
      }
      ast::pat_bind(name) {
        let vid = lookup_local(fcx, pat.span, pat.id);
        let typ = ty::mk_var(fcx.ccx.tcx, vid);
        typ = demand::simple(fcx, pat.span, expected, typ);
        let canon_id = map.get(name);
        if canon_id != pat.id {
            let ct =
                ty::mk_var(fcx.ccx.tcx,
                           lookup_local(fcx, pat.span, canon_id));
            typ = demand::simple(fcx, pat.span, ct, typ);
        }
        write::ty_only_fixup(fcx, pat.id, typ);
      }
      ast::pat_tag(path, subpats) {
        // Typecheck the path.
        let v_def = lookup_def(fcx, path.span, pat.id);
        let v_def_ids = ast::variant_def_ids(v_def);
        let tag_tpt = ty::lookup_item_type(fcx.ccx.tcx, v_def_ids.tg);
        let path_tpot = instantiate_path(fcx, path, tag_tpt, pat.span);
        // Take the tag type params out of `expected`.

        alt structure_of(fcx, pat.span, expected) {
          ty::ty_tag(_, expected_tps) {
            // Unify with the expected tag type.

            let ctor_ty =
                ty::ty_param_substs_opt_and_ty_to_monotype(fcx.ccx.tcx,
                                                           path_tpot);

            let path_tpt =
                demand::full(fcx, pat.span, expected, ctor_ty, expected_tps,
                             NO_AUTODEREF);
            path_tpot =
                {substs: some[ty::t[]](path_tpt.substs), ty: path_tpt.ty};
            // Get the number of arguments in this tag variant.

            let arg_types =
                variant_arg_types(fcx.ccx, pat.span, v_def_ids.var,
                                  expected_tps);
            let subpats_len = std::ivec::len[@ast::pat](subpats);
            if std::ivec::len[ty::t](arg_types) > 0u {
                // N-ary variant.

                let arg_len = ivec::len[ty::t](arg_types);
                if arg_len != subpats_len {
                    // TODO: note definition of tag variant
                    // TODO (issue #448): Wrap a #fmt string over multiple
                    // lines...
                    let s =
                        #fmt("this pattern has %u field%s, but the \
                                       corresponding variant has %u field%s",
                             subpats_len,
                             if subpats_len == 1u { "" } else { "s" },
                             arg_len, if arg_len == 1u { "" } else { "s" });
                    fcx.ccx.tcx.sess.span_fatal(pat.span, s);
                }

                // TODO: ivec::iter2

                let i = 0u;
                for subpat: @ast::pat  in subpats {
                    check_pat(fcx, map, subpat, arg_types.(i));
                    i += 1u;
                }
            } else if (subpats_len > 0u) {
                // TODO: note definition of tag variant
                // TODO (issue #448): Wrap a #fmt string over multiple
                // lines...

                fcx.ccx.tcx.sess.span_fatal
                    (pat.span, #fmt("this pattern has %u field%s, \
                                     but the corresponding \
                                     variant has no fields",
                                    subpats_len,
                                    if subpats_len == 1u { "" }
                                    else { "s" }));
            }
            write::ty_fixup(fcx, pat.id, path_tpot);
          }
          _ {
            // FIXME: Switch expected and actual in this message? I
            // can never tell.
            fcx.ccx.tcx.sess.span_fatal(pat.span,
                                        #fmt("mismatched types: \
                                                  expected %s, found tag",
                                             ty_to_str(fcx.ccx.tcx,
                                                       expected)));
          }
        }
        write::ty_fixup(fcx, pat.id, path_tpot);
      }
      ast::pat_rec(fields, etc) {
        let ex_fields;
        alt structure_of(fcx, pat.span, expected) {
          ty::ty_rec(fields) { ex_fields = fields; }
          _ {
            fcx.ccx.tcx.sess.span_fatal(pat.span,
                                        #fmt("mismatched types: expected %s, \
                                         found record",
                                             ty_to_str(fcx.ccx.tcx,
                                                       expected)));
          }
        }
        let f_count = ivec::len(fields);
        let ex_f_count = ivec::len(ex_fields);
        if ex_f_count < f_count || !etc && ex_f_count > f_count {
            fcx.ccx.tcx.sess.span_fatal
                (pat.span, #fmt("mismatched types: expected a record \
                                 with %u fields, found one with %u \
                                 fields", ex_f_count, f_count));
        }
        fn matches(name: &str, f: &ty::field) -> bool {
            ret str::eq(name, f.ident);
        }
        for f: ast::field_pat  in fields {
            alt ivec::find(bind matches(f.ident, _), ex_fields) {
              some(field) { check_pat(fcx, map, f.pat, field.mt.ty); }
              none. {
                fcx.ccx.tcx.sess.span_fatal(pat.span,
                                            #fmt("mismatched types: did not \
                                             expect a record with a field %s",
                                                 f.ident));
              }
            }
        }
        write::ty_only_fixup(fcx, pat.id, expected);
      }
      ast::pat_box(inner) {
        alt structure_of(fcx, pat.span, expected) {
          ty::ty_box(e_inner) {
            check_pat(fcx, map, inner, e_inner.ty);
            write::ty_only_fixup(fcx, pat.id, expected);
          }
          _ {
            fcx.ccx.tcx.sess.span_fatal(pat.span,
                                        "mismatched types: expected " +
                                            ty_to_str(fcx.ccx.tcx, expected) +
                                            " found box");
          }
        }
      }
    }
}

fn require_impure(sess: &session::session, f_purity: &ast::purity,
                  sp: &span) {
    alt f_purity {
      ast::impure_fn. { ret; }
      ast::pure_fn. {
        sess.span_fatal(sp, "Found impure expression in pure function decl");
      }
    }
}

fn require_pure_call(ccx: @crate_ctxt, caller_purity: &ast::purity,
                     callee: &@ast::expr, sp: &span) {
    alt caller_purity {
      ast::impure_fn. { ret; }
      ast::pure_fn. {
        alt ccx.tcx.def_map.find(callee.id) {
          some(ast::def_fn(_, ast::pure_fn.)) { ret; }
          _ {
            ccx.tcx.sess.span_fatal
                (sp, "Pure function calls function not known to be pure");
          }
        }
      }
    }
}

fn check_expr(fcx: &@fn_ctxt, expr: &@ast::expr) {
    // fcx.ccx.tcx.sess.span_warn(expr.span, "typechecking expr " +
    //                            syntax::print::pprust::expr_to_str(expr));

    // A generic function to factor out common logic from call and bind
    // expressions.
    fn check_call_or_bind(fcx: &@fn_ctxt, sp: &span, f: &@ast::expr,
                          args: &(option::t[@ast::expr])[],
                          call_kind: call_kind) {
        // Check the function.
        check_expr(fcx, f);

        // Get the function type.
        let fty = expr_ty(fcx.ccx.tcx, f);

        // We want to autoderef calls but not binds
        let fty_stripped =
            alt call_kind {
              kind_call. { do_autoderef(fcx, sp, fty) }
              _ { fty }
            };

        // Grab the argument types and the return type.
        let arg_tys;
        alt structure_of(fcx, sp, fty_stripped) {
          ty::ty_fn(_, arg_tys_0, _, _, _) { arg_tys = arg_tys_0; }
          ty::ty_native_fn(_, arg_tys_0, _) { arg_tys = arg_tys_0; }
          _ {
            fcx.ccx.tcx.sess.span_fatal(f.span,
                                        "mismatched types: \
                                           expected function or native \
                                           function but found "
                                            + ty_to_str(fcx.ccx.tcx, fty));
          }
        }

        // Check that the correct number of arguments were supplied.
        let expected_arg_count = ivec::len[ty::arg](arg_tys);
        let supplied_arg_count = ivec::len[option::t[@ast::expr]](args);
        if expected_arg_count != supplied_arg_count {
            fcx.ccx.tcx.sess.span_fatal(sp,
                                        #fmt("this function takes %u \
                                            parameter%s but %u parameter%s \
                                            supplied",
                                             expected_arg_count,
                                             if expected_arg_count == 1u {
                                                 ""
                                             } else { "s" },
                                             supplied_arg_count,
                                             if supplied_arg_count == 1u {
                                                 " was"
                                             } else { "s were" }));
        }

        // Check the arguments.
        // TODO: iter2
        let i = 0u;
        for a_opt: option::t[@ast::expr]  in args {
            alt a_opt {
              some(a) {
                check_expr(fcx, a);
                demand::full(fcx, a.span, arg_tys.(i).ty,
                             expr_ty(fcx.ccx.tcx, a), ~[],
                             AUTODEREF_BLOCK_COERCE);
              }
              none. { }
            }
            i += 1u;
        }
    }
    // A generic function for checking assignment expressions

    fn check_assignment(fcx: &@fn_ctxt, sp: &span, lhs: &@ast::expr,
                        rhs: &@ast::expr, id: &ast::node_id) {
        check_expr(fcx, lhs);
        check_expr(fcx, rhs);
        demand::simple(fcx, sp, expr_ty(fcx.ccx.tcx, lhs),
                       expr_ty(fcx.ccx.tcx, rhs));
        write::ty_only_fixup(fcx, id, ty::mk_nil(fcx.ccx.tcx));
    }
    // A generic function for checking call expressions

    fn check_call(fcx: &@fn_ctxt, sp: &span, f: &@ast::expr,
                  args: &(@ast::expr)[], call_kind: call_kind) {
        let args_opt_0: (option::t[@ast::expr])[] = ~[];
        for arg: @ast::expr  in args {
            args_opt_0 += ~[some[@ast::expr](arg)];
        }

        // Call the generic checker.
        check_call_or_bind(fcx, sp, f, args_opt_0, call_kind);
    }
    // A generic function for checking for or for-each loops

    fn check_for_or_for_each(fcx: &@fn_ctxt, local: &@ast::local,
                             element_ty: &ty::t, body: &ast::blk,
                             node_id: ast::node_id) {
        check_decl_local(fcx, local);
        check_block(fcx, body);
        // Unify type of decl with element type of the seq
        demand::simple(fcx, local.span, ty::decl_local_ty(fcx.ccx.tcx, local),
                       element_ty);
        let typ = ty::mk_nil(fcx.ccx.tcx);
        write::ty_only_fixup(fcx, node_id, typ);
    }

    // A generic function for checking the pred in a check
    // or if-check
    fn check_pred_expr(fcx: &@fn_ctxt, e: &@ast::expr) {
        check_expr(fcx, e);
        demand::simple(fcx, e.span, ty::mk_bool(fcx.ccx.tcx),
                       expr_ty(fcx.ccx.tcx, e));


        /* e must be a call expr where all arguments are either
           literals or slots */
        alt e.node {
          ast::expr_call(operator, operands) {
            alt operator.node {
              ast::expr_path(oper_name) {
                alt fcx.ccx.tcx.def_map.find(operator.id) {
                  some(ast::def_fn(_, ast::pure_fn.)) {
                    // do nothing
                  }
                  _ {
                    fcx.ccx.tcx.sess.span_fatal(operator.span,
                                                "non-predicate as operator \
                                       in constraint");
                  }
                }
                for operand: @ast::expr  in operands {
                    if !ast::is_constraint_arg(operand) {
                        let s =
                            "Constraint args must be \
                                              slot variables or literals";
                        fcx.ccx.tcx.sess.span_fatal(e.span, s);
                    }
                }
              }
              _ {
                let s =
                    "In a constraint, expected the \
                                      constraint name to be an explicit name";
                fcx.ccx.tcx.sess.span_fatal(e.span, s);
              }
            }
          }
          _ { fcx.ccx.tcx.sess.span_fatal(e.span, "check on non-predicate"); }
        }
    }

    // A generic function for checking the then and else in an if
    // or if-check
    fn check_then_else(fcx: &@fn_ctxt, thn: &ast::blk,
                       elsopt: &option::t[@ast::expr], id: ast::node_id,
                       sp: &span) {
        check_block(fcx, thn);
        let if_t =
            alt elsopt {
              some(els) {
                check_expr(fcx, els);
                let thn_t = block_ty(fcx.ccx.tcx, thn);
                let elsopt_t = expr_ty(fcx.ccx.tcx, els);
                demand::simple(fcx, sp, thn_t, elsopt_t);
                if !ty::type_is_bot(fcx.ccx.tcx, elsopt_t) {
                    elsopt_t
                } else { thn_t }
              }
              none. { ty::mk_nil(fcx.ccx.tcx) }
            };
        write::ty_only_fixup(fcx, id, if_t);
    }

    // Checks the compatibility
    fn check_binop_type_compat(fcx: &@fn_ctxt, span: span, ty: ty::t,
                               binop: ast::binop) {
        let resolved_t = resolve_type_vars_if_possible(fcx, ty);
        if !ty::is_binopable(fcx.ccx.tcx, resolved_t, binop) {
            let binopstr = ast::binop_to_str(binop);
            let t_str = ty_to_str(fcx.ccx.tcx, resolved_t);
            let errmsg =
                "binary operation " + binopstr +
                    " cannot be applied to type `" + t_str + "`";
            fcx.ccx.tcx.sess.span_fatal(span, errmsg);
        }
    }

    let id = expr.id;
    alt expr.node {
      ast::expr_lit(lit) {
        let typ = check_lit(fcx.ccx, lit);
        write::ty_only_fixup(fcx, id, typ);
      }
      ast::expr_binary(binop, lhs, rhs) {
        check_expr(fcx, lhs);
        check_expr(fcx, rhs);

        let lhs_t = expr_ty(fcx.ccx.tcx, lhs);
        let rhs_t = expr_ty(fcx.ccx.tcx, rhs);

        demand::autoderef(fcx, rhs.span, lhs_t, rhs_t, AUTODEREF_OK);
        let deref_t = do_autoderef(fcx, expr.span, lhs_t);
        check_binop_type_compat(fcx, expr.span, deref_t, binop);

        let t =
            alt binop {
              ast::eq. { ty::mk_bool(fcx.ccx.tcx) }
              ast::lt. { ty::mk_bool(fcx.ccx.tcx) }
              ast::le. { ty::mk_bool(fcx.ccx.tcx) }
              ast::ne. { ty::mk_bool(fcx.ccx.tcx) }
              ast::ge. { ty::mk_bool(fcx.ccx.tcx) }
              ast::gt. { ty::mk_bool(fcx.ccx.tcx) }
              _ { deref_t }
            };
        write::ty_only_fixup(fcx, id, t);
      }
      ast::expr_unary(unop, oper) {
        check_expr(fcx, oper);
        let oper_t = expr_ty(fcx.ccx.tcx, oper);
        alt unop {
          ast::box(mut) {
            oper_t = ty::mk_box(fcx.ccx.tcx, {ty: oper_t, mut: mut});
          }
          ast::deref. {
            alt structure_of(fcx, expr.span, oper_t) {
              ty::ty_box(inner) { oper_t = inner.ty; }
              ty::ty_res(_, inner, _) { oper_t = inner; }
              ty::ty_tag(id, tps) {
                let variants = ty::tag_variants(fcx.ccx.tcx, id);
                if ivec::len(variants) != 1u ||
                       ivec::len(variants.(0).args) != 1u {
                    fcx.ccx.tcx.sess.span_fatal
                        (expr.span, "can only dereference tags " +
                         "with a single variant which has a "
                         + "single argument");
                }
                oper_t =
                    ty::substitute_type_params(fcx.ccx.tcx, tps,
                                               variants.(0).args.(0));
              }
              ty::ty_ptr(inner) { oper_t = inner.ty; }
              _ {
                fcx.ccx.tcx.sess.span_fatal(expr.span,
                                            "dereferencing non-" +
                                                "dereferenceable type: " +
                                                ty_to_str(fcx.ccx.tcx,
                                                          oper_t));
              }
            }
          }
          ast::not. {
            if !type_is_integral(fcx, oper.span, oper_t) &&
                   structure_of(fcx, oper.span, oper_t) != ty::ty_bool {
                fcx.ccx.tcx.sess.span_fatal(expr.span,
                                            #fmt("mismatched types: \
                                                        expected bool or \
                                                        integer but found %s",
                                                 ty_to_str(fcx.ccx.tcx,
                                                           oper_t)));
            }
          }
          _ { oper_t = do_autoderef(fcx, expr.span, oper_t); }
        }
        write::ty_only_fixup(fcx, id, oper_t);
      }
      ast::expr_path(pth) {
        let defn = lookup_def(fcx, pth.span, id);
        let tpt = ty_param_count_and_ty_for_def(fcx, expr.span, defn);
        if ty::def_has_ty_params(defn) {
            let path_tpot = instantiate_path(fcx, pth, tpt, expr.span);
            write::ty_fixup(fcx, id, path_tpot);
            ret;
        }
        // The definition doesn't take type parameters. If the programmer
        // supplied some, that's an error.

        if ivec::len[@ast::ty](pth.node.types) > 0u {
            fcx.ccx.tcx.sess.span_fatal(expr.span,
                                        "this kind of value does not \
                                           take type parameters");
        }
        write::ty_only_fixup(fcx, id, tpt.ty);
      }
      ast::expr_mac(_) { fcx.ccx.tcx.sess.bug("unexpanded macro"); }
      ast::expr_fail(expr_opt) {
        alt expr_opt {
          none. {/* do nothing */ }
          some(e) {
            check_expr(fcx, e);
            let tcx = fcx.ccx.tcx;
            let ety = expr_ty(tcx, e);
            demand::simple(fcx, e.span, ty::mk_str(tcx), ety);
          }
        }
        write::bot_ty(fcx.ccx.tcx, id);
      }
      ast::expr_break. { write::bot_ty(fcx.ccx.tcx, id); }
      ast::expr_cont. { write::bot_ty(fcx.ccx.tcx, id); }
      ast::expr_ret(expr_opt) {
        alt expr_opt {
          none. {
            let nil = ty::mk_nil(fcx.ccx.tcx);
            if !are_compatible(fcx, fcx.ret_ty, nil) {
                fcx.ccx.tcx.sess.span_fatal(expr.span,
                                            "ret; in function \
                                                   returning non-nil");
            }
            write::bot_ty(fcx.ccx.tcx, id);
          }
          some(e) {
            check_expr(fcx, e);
            demand::simple(fcx, expr.span, fcx.ret_ty,
                           expr_ty(fcx.ccx.tcx, e));
            write::bot_ty(fcx.ccx.tcx, id);
          }
        }
      }
      ast::expr_put(expr_opt) {
        require_impure(fcx.ccx.tcx.sess, fcx.purity, expr.span);
        alt expr_opt {
          none. {
            let nil = ty::mk_nil(fcx.ccx.tcx);
            if !are_compatible(fcx, fcx.ret_ty, nil) {
                fcx.ccx.tcx.sess.span_fatal(expr.span,
                                            "put; in iterator \
                                                   yielding non-nil");
            }
            write::nil_ty(fcx.ccx.tcx, id);
          }
          some(e) {
            check_expr(fcx, e);
            demand::simple(fcx, expr.span, fcx.ret_ty,
                           expr_ty(fcx.ccx.tcx, e));
            write::nil_ty(fcx.ccx.tcx, id);
          }
        }
      }
      ast::expr_be(e) {
        // FIXME: prove instead of assert

        assert (ast::is_call_expr(e));
        check_expr(fcx, e);
        demand::simple(fcx, e.span, fcx.ret_ty, expr_ty(fcx.ccx.tcx, e));
        write::nil_ty(fcx.ccx.tcx, id);
      }
      ast::expr_log(l, e) {
        check_expr(fcx, e);
        write::nil_ty(fcx.ccx.tcx, id);
      }
      ast::expr_check(_, e) {
        check_pred_expr(fcx, e);
        write::nil_ty(fcx.ccx.tcx, id);
      }
      ast::expr_if_check(cond, thn, elsopt) {
        check_pred_expr(fcx, cond);
        check_then_else(fcx, thn, elsopt, id, expr.span);
      }
      ast::expr_ternary(_, _, _) {
        check_expr(fcx, ast::ternary_to_if(expr));
      }
      ast::expr_assert(e) {
        check_expr(fcx, e);
        let ety = expr_ty(fcx.ccx.tcx, e);
        demand::simple(fcx, expr.span, ty::mk_bool(fcx.ccx.tcx), ety);
        write::nil_ty(fcx.ccx.tcx, id);
      }
      ast::expr_move(lhs, rhs) {
        require_impure(fcx.ccx.tcx.sess, fcx.purity, expr.span);
        check_assignment(fcx, expr.span, lhs, rhs, id);
      }
      ast::expr_assign(lhs, rhs) {
        require_impure(fcx.ccx.tcx.sess, fcx.purity, expr.span);
        check_assignment(fcx, expr.span, lhs, rhs, id);
      }
      ast::expr_swap(lhs, rhs) {
        require_impure(fcx.ccx.tcx.sess, fcx.purity, expr.span);
        check_assignment(fcx, expr.span, lhs, rhs, id);
      }
      ast::expr_assign_op(op, lhs, rhs) {
        require_impure(fcx.ccx.tcx.sess, fcx.purity, expr.span);
        check_assignment(fcx, expr.span, lhs, rhs, id);
        check_binop_type_compat(fcx, expr.span, expr_ty(fcx.ccx.tcx, lhs),
                                op);
      }
      ast::expr_send(lhs, rhs) {
        require_impure(fcx.ccx.tcx.sess, fcx.purity, expr.span);
        check_expr(fcx, lhs);
        check_expr(fcx, rhs);
        let rhs_t = expr_ty(fcx.ccx.tcx, rhs);
        let chan_t = ty::mk_chan(fcx.ccx.tcx, rhs_t);
        let lhs_t = expr_ty(fcx.ccx.tcx, lhs);
        alt structure_of(fcx, expr.span, lhs_t) {
          ty::ty_chan(it) { }
          _ {
            let s =
                #fmt("mismatched types: expected chan \
                                   but found %s",
                     ty_to_str(fcx.ccx.tcx, lhs_t));
            fcx.ccx.tcx.sess.span_fatal(expr.span, s);
          }
        }
        demand::simple(fcx, expr.span, chan_t, lhs_t);
        write::ty_only_fixup(fcx, id, chan_t);
      }
      ast::expr_recv(lhs, rhs) {
        require_impure(fcx.ccx.tcx.sess, fcx.purity, expr.span);
        check_expr(fcx, lhs);
        check_expr(fcx, rhs);
        let item_t = expr_ty(fcx.ccx.tcx, rhs);
        let port_t = ty::mk_port(fcx.ccx.tcx, item_t);
        demand::simple(fcx, expr.span, port_t, expr_ty(fcx.ccx.tcx, lhs));
        write::ty_only_fixup(fcx, id, item_t);
      }
      ast::expr_if(cond, thn, elsopt) {
        check_expr(fcx, cond);
        demand::simple(fcx, cond.span, ty::mk_bool(fcx.ccx.tcx),
                       expr_ty(fcx.ccx.tcx, cond));
        check_then_else(fcx, thn, elsopt, id, expr.span);
      }
      ast::expr_for(decl, seq, body) {
        check_expr(fcx, seq);
        let elt_ty;
        let ety = expr_ty(fcx.ccx.tcx, seq);
        alt structure_of(fcx, expr.span, ety) {
          ty::ty_vec(vec_elt_ty) { elt_ty = vec_elt_ty.ty; }
          ty::ty_str. { elt_ty = ty::mk_mach(fcx.ccx.tcx, ast::ty_u8); }
          ty::ty_ivec(vec_elt_ty) { elt_ty = vec_elt_ty.ty; }
          ty::ty_istr. { elt_ty = ty::mk_mach(fcx.ccx.tcx, ast::ty_u8); }
          _ {
            fcx.ccx.tcx.sess.span_fatal
                (expr.span, "mismatched types: expected vector or string but "
                 + "found " + ty_to_str(fcx.ccx.tcx, ety));
          }
        }
        check_for_or_for_each(fcx, decl, elt_ty, body, id);
      }
      ast::expr_for_each(decl, seq, body) {
        check_expr(fcx, seq);
        check_for_or_for_each(fcx, decl, expr_ty(fcx.ccx.tcx, seq), body, id);
      }
      ast::expr_while(cond, body) {
        check_expr(fcx, cond);
        check_block(fcx, body);
        demand::simple(fcx, cond.span, ty::mk_bool(fcx.ccx.tcx),
                       expr_ty(fcx.ccx.tcx, cond));
        let typ = ty::mk_nil(fcx.ccx.tcx);
        write::ty_only_fixup(fcx, id, typ);
      }
      ast::expr_do_while(body, cond) {
        check_expr(fcx, cond);
        check_block(fcx, body);
        let typ = block_ty(fcx.ccx.tcx, body);
        write::ty_only_fixup(fcx, id, typ);
      }
      ast::expr_alt(expr, arms) {
        check_expr(fcx, expr);
        // Typecheck the patterns first, so that we get types for all the
        // bindings.

        let pattern_ty = ty::expr_ty(fcx.ccx.tcx, expr);
        for arm: ast::arm  in arms {
            let id_map = ast::pat_id_map(arm.pats.(0));
            for p: @ast::pat  in arm.pats {
                check_pat(fcx, id_map, p, pattern_ty);
            }
        }
        // Now typecheck the blocks.

        let result_ty = next_ty_var(fcx);
        for arm: ast::arm  in arms {
            check_block(fcx, arm.block);
            let bty = block_ty(fcx.ccx.tcx, arm.block);


            // Failing alt arms don't need to have a matching type
            if !ty::type_is_bot(fcx.ccx.tcx, bty) {
                result_ty =
                    demand::simple(fcx, arm.block.span, result_ty, bty);
            }
        }
        write::ty_only_fixup(fcx, id, result_ty);
      }
      ast::expr_fn(f) {
        let cx = @{tcx: fcx.ccx.tcx};
        let convert =
            bind ast_ty_to_ty(cx.tcx, bind collect::getter(cx, _), _);
        let ty_of_arg = bind collect::ty_of_arg(cx, _);
        let fty =
            collect::ty_of_fn_decl(cx, convert, ty_of_arg, f.decl, f.proto,
                                   ~[], none).ty;
        write::ty_only_fixup(fcx, id, fty);
        check_fn(fcx.ccx, f, id, some(fcx));
      }
      ast::expr_block(b) {
        check_block(fcx, b);
        alt b.node.expr {
          some(expr) {
            let typ = expr_ty(fcx.ccx.tcx, expr);
            write::ty_only_fixup(fcx, id, typ);
          }
          none. {
            let typ = ty::mk_nil(fcx.ccx.tcx);
            write::ty_only_fixup(fcx, id, typ);
          }
        }
      }
      ast::expr_bind(f, args) {
        // Call the generic checker.
        check_call_or_bind(fcx, expr.span, f, args, kind_bind);

        // Pull the argument and return types out.
        let proto_1;
        let arg_tys_1: ty::arg[] = ~[];
        let rt_1;
        let fty = expr_ty(fcx.ccx.tcx, f);
        let t_1;
        alt structure_of(fcx, expr.span, fty) {
          ty::ty_fn(proto, arg_tys, rt, cf, constrs) {
            proto_1 = proto;
            rt_1 = rt;
            // FIXME:
            // probably need to munge the constrs to drop constraints
            // for any bound args

            // For each blank argument, add the type of that argument
            // to the resulting function type.
            let i = 0u;
            while i < ivec::len[option::t[@ast::expr]](args) {
                alt args.(i) {
                  some(_) {/* no-op */ }
                  none. { arg_tys_1 += ~[arg_tys.(i)]; }
                }
                i += 1u;
            }
            t_1 =
                ty::mk_fn(fcx.ccx.tcx, proto_1, arg_tys_1, rt_1, cf, constrs);
          }
          _ {
            log_err "LHS of bind expr didn't have a function type?!";
            fail;
          }
        }
        write::ty_only_fixup(fcx, id, t_1);
      }
      ast::expr_call(f, args) {
        /* here we're kind of hosed, as f can be any expr
         need to restrict it to being an explicit expr_path if we're
        inside a pure function, and need an environment mapping from
        function name onto purity-designation */

        require_pure_call(fcx.ccx, fcx.purity, f, expr.span);
        check_call(fcx, expr.span, f, args, kind_call);
        // Pull the return type out of the type of the function.

        let rt_1;
        let fty = do_autoderef(fcx, expr.span, ty::expr_ty(fcx.ccx.tcx, f));
        alt structure_of(fcx, expr.span, fty) {
          ty::ty_fn(_, _, rt, _, _) { rt_1 = rt; }
          ty::ty_native_fn(_, _, rt) { rt_1 = rt; }
          _ {
            log_err "LHS of call expr didn't have a function type?!";
            fail;
          }
        }
        write::ty_only_fixup(fcx, id, rt_1);
      }
      ast::expr_self_method(ident) {
        let t = ty::mk_nil(fcx.ccx.tcx);
        let this_obj_sty: option::t[ty::sty] =
            some(structure_of(fcx, expr.span, ty::mk_nil(fcx.ccx.tcx)));
        let this_obj_info: option::t[obj_info] = get_obj_info(fcx.ccx);
        alt this_obj_info {
          some(oinfo) {
            alt oinfo {
              regular_obj(_, obj_id) {
                let did = local_def(obj_id);

                // Try looking up the current object in the type
                // cache.
                alt fcx.ccx.tcx.tcache.find(did) {
                  some(tpt) {
                    // If we're typechecking a self-method on
                    // a regular object, this lookup should
                    // succeed.
                    this_obj_sty = some(structure_of(fcx, expr.span, tpt.ty));
                  }
                  none. {
                    fcx.ccx.tcx.sess.bug("didn't find " + int::str(did.node) +
                                             " in type cache");
                  }
                }
              }
              anon_obj(_, obj_sty) { this_obj_sty = obj_sty; }
            }
          }
          none. {
            // Shouldn't happen.
            fcx.ccx.tcx.sess.span_err(expr.span,
                                      "self-call in non-object \
                                               context");
          }
        }

        // Grab this method's type out of the current object type.
        alt this_obj_sty {
          some(sty) {
            alt sty {
              ty::ty_obj(methods) {
                for method: ty::method  in methods {
                    if method.ident == ident {
                        t = ty::method_ty_to_fn_ty(fcx.ccx.tcx, method);
                    }
                }
              }
              _ { fail; }
            }
          }
          none. { }
        }
        write::ty_only_fixup(fcx, id, t);
        require_impure(fcx.ccx.tcx.sess, fcx.purity, expr.span);
      }
      ast::expr_spawn(_, _, f, args) {
        check_call(fcx, expr.span, f, args, kind_spawn);
        let fty = expr_ty(fcx.ccx.tcx, f);
        let ret_ty = ty::ret_ty_of_fn_ty(fcx.ccx.tcx, fty);
        demand::simple(fcx, f.span, ty::mk_nil(fcx.ccx.tcx), ret_ty);

        // make sure they aren't spawning a function with type params
        if ty::expr_has_ty_params(fcx.ccx.tcx, f) {
            fcx.ccx.tcx.sess.span_fatal(
                f.span,
                "spawning functions with type params not allowed (for now)");
        }

        // FIXME: Other typechecks needed
        let typ = ty::mk_task(fcx.ccx.tcx);
        write::ty_only_fixup(fcx, id, typ);
      }
      ast::expr_cast(e, t) {
        check_expr(fcx, e);
        let t_1 = ast_ty_to_ty_crate(fcx.ccx, t);
        // FIXME: there are more forms of cast to support, eventually.

        if !(type_is_scalar(fcx, expr.span, expr_ty(fcx.ccx.tcx, e)) &&
                 type_is_scalar(fcx, expr.span, t_1)) {
            fcx.ccx.tcx.sess.span_fatal(expr.span,
                                        "non-scalar cast: " +
                                            ty_to_str(fcx.ccx.tcx,
                                                      expr_ty(fcx.ccx.tcx, e))
                                            + " as " +
                                            ty_to_str(fcx.ccx.tcx, t_1));
        }
        write::ty_only_fixup(fcx, id, t_1);
      }
      ast::expr_vec(args, mut, kind) {
        let t: ty::t;
        if ivec::len[@ast::expr](args) == 0u {
            t = next_ty_var(fcx);
        } else {
            check_expr(fcx, args.(0));
            t = expr_ty(fcx.ccx.tcx, args.(0));
        }
        for e: @ast::expr  in args {
            check_expr(fcx, e);
            let expr_t = expr_ty(fcx.ccx.tcx, e);
            demand::simple(fcx, expr.span, t, expr_t);
        }
        let typ;
        alt kind {
          ast::sk_rc. { typ = ty::mk_vec(fcx.ccx.tcx, {ty: t, mut: mut}); }
          ast::sk_unique. {
            typ = ty::mk_ivec(fcx.ccx.tcx, {ty: t, mut: mut});
          }
        }
        write::ty_only_fixup(fcx, id, typ);
      }
      ast::expr_rec(fields, base) {
        alt base { none. {/* no-op */ } some(b_0) { check_expr(fcx, b_0); } }
        let fields_t: (spanned[field])[] = ~[];
        for f: ast::field  in fields {
            check_expr(fcx, f.node.expr);
            let expr_t = expr_ty(fcx.ccx.tcx, f.node.expr);
            let expr_mt = {ty: expr_t, mut: f.node.mut};
            // for the most precise error message,
            // should be f.node.expr.span, not f.span
            fields_t +=
                ~[respan(f.node.expr.span,
                         {ident: f.node.ident, mt: expr_mt})];
        }
        alt base {
          none. {
            fn get_node(f: &spanned[field]) -> field { f.node }
            let typ = ty::mk_rec(fcx.ccx.tcx, ivec::map(get_node, fields_t));
            write::ty_only_fixup(fcx, id, typ);
          }
          some(bexpr) {
            check_expr(fcx, bexpr);
            let bexpr_t = expr_ty(fcx.ccx.tcx, bexpr);
            let base_fields: field[] = ~[];
            alt structure_of(fcx, expr.span, bexpr_t) {
              ty::ty_rec(flds) { base_fields = flds; }
              _ {
                fcx.ccx.tcx.sess.span_fatal(expr.span,
                                            "record update \
                                                       non-record base");
              }
            }
            write::ty_only_fixup(fcx, id, bexpr_t);
            for f: spanned[ty::field]  in fields_t {
                let found = false;
                for bf: ty::field  in base_fields {
                    if str::eq(f.node.ident, bf.ident) {
                        demand::simple(fcx, f.span, bf.mt.ty, f.node.mt.ty);
                        found = true;
                    }
                }
                if !found {
                    fcx.ccx.tcx.sess.span_fatal(f.span,
                                                "unknown field in \
                                                       record update: "
                                                    + f.node.ident);
                }
            }
          }
        }
      }
      ast::expr_field(base, field) {
        check_expr(fcx, base);
        let base_t = expr_ty(fcx.ccx.tcx, base);
        base_t = do_autoderef(fcx, expr.span, base_t);
        alt structure_of(fcx, expr.span, base_t) {
          ty::ty_rec(fields) {
            let ix: uint =
                ty::field_idx(fcx.ccx.tcx.sess, expr.span, field, fields);
            if ix >= ivec::len[ty::field](fields) {
                fcx.ccx.tcx.sess.span_fatal(expr.span, "bad index on record");
            }
            write::ty_only_fixup(fcx, id, fields.(ix).mt.ty);
          }
          ty::ty_obj(methods) {
            let ix: uint =
                ty::method_idx(fcx.ccx.tcx.sess, expr.span, field, methods);
            if ix >= ivec::len[ty::method](methods) {
                fcx.ccx.tcx.sess.span_fatal(expr.span, "bad index on obj");
            }
            let meth = methods.(ix);
            let t =
                ty::mk_fn(fcx.ccx.tcx, meth.proto, meth.inputs, meth.output,
                          meth.cf, meth.constrs);
            write::ty_only_fixup(fcx, id, t);
          }
          _ {
            let t_err = resolve_type_vars_if_possible(fcx, base_t);
            let msg =
                #fmt("attempted field access on type %s",
                     ty_to_str(fcx.ccx.tcx, t_err));
            fcx.ccx.tcx.sess.span_fatal(expr.span, msg);
          }
        }
      }
      ast::expr_index(base, idx) {
        check_expr(fcx, base);
        let base_t = expr_ty(fcx.ccx.tcx, base);
        base_t = do_autoderef(fcx, expr.span, base_t);
        check_expr(fcx, idx);
        let idx_t = expr_ty(fcx.ccx.tcx, idx);
        if !type_is_integral(fcx, idx.span, idx_t) {
            fcx.ccx.tcx.sess.span_fatal(idx.span,
                                        "mismatched types: expected \
                                           integer but found "
                                            + ty_to_str(fcx.ccx.tcx, idx_t));
        }
        alt structure_of(fcx, expr.span, base_t) {
          ty::ty_vec(mt) { write::ty_only_fixup(fcx, id, mt.ty); }
          ty::ty_ivec(mt) { write::ty_only_fixup(fcx, id, mt.ty); }
          ty::ty_str. {
            let typ = ty::mk_mach(fcx.ccx.tcx, ast::ty_u8);
            write::ty_only_fixup(fcx, id, typ);
          }
          ty::ty_istr. {
            let typ = ty::mk_mach(fcx.ccx.tcx, ast::ty_u8);
            write::ty_only_fixup(fcx, id, typ);
          }
          _ {
            fcx.ccx.tcx.sess.span_fatal(expr.span,
                                        "vector-indexing bad type: " +
                                            ty_to_str(fcx.ccx.tcx, base_t));
          }
        }
      }
      ast::expr_port(typ) {
        let t = next_ty_var(fcx);
        alt typ {
          some(_t) {
            demand::simple(fcx, expr.span, ast_ty_to_ty_crate(fcx.ccx, _t),
                           t);
          }
          none. { }
        }
        let pt = ty::mk_port(fcx.ccx.tcx, t);
        write::ty_only_fixup(fcx, id, pt);
      }
      ast::expr_chan(x) {
        check_expr(fcx, x);
        let port_t = expr_ty(fcx.ccx.tcx, x);
        alt structure_of(fcx, expr.span, port_t) {
          ty::ty_port(subtype) {
            let ct = ty::mk_chan(fcx.ccx.tcx, subtype);
            write::ty_only_fixup(fcx, id, ct);
          }
          _ {
            fcx.ccx.tcx.sess.span_fatal(expr.span,
                                        "bad port type: " +
                                            ty_to_str(fcx.ccx.tcx, port_t));
          }
        }
      }
      ast::expr_anon_obj(ao) {

        let fields: ast::anon_obj_field[] = ~[];
        alt ao.fields { none. { } some(v) { fields = v; } }

        // FIXME: These next three functions are largely ripped off from
        // similar ones in collect::.  Is there a better way to do this?
        fn ty_of_arg(ccx: @crate_ctxt, a: &ast::arg) -> ty::arg {
            let ty_mode = ast_mode_to_mode(a.mode);
            ret {mode: ty_mode, ty: ast_ty_to_ty_crate(ccx, a.ty)};
        }

        fn ty_of_method(ccx: @crate_ctxt, m: &@ast::method) -> ty::method {
            let convert = bind ast_ty_to_ty_crate(ccx, _);

            let inputs = ~[];
            for aa: ast::arg  in m.node.meth.decl.inputs {
                inputs += ~[ty_of_arg(ccx, aa)];
            }

            let output = convert(m.node.meth.decl.output);

            let out_constrs = ~[];
            for constr: @ast::constr  in m.node.meth.decl.constraints {
                out_constrs += ~[ty::ast_constr_to_constr(ccx.tcx, constr)];
            }

            ret {proto: m.node.meth.proto,
                 ident: m.node.ident,
                 inputs: inputs,
                 output: output,
                 cf: m.node.meth.decl.cf,
                 constrs: out_constrs};
        }

        fn get_anon_obj_method_types(fcx: @fn_ctxt, ao: &ast::anon_obj,
                                     fields: &ast::anon_obj_field[]) ->
           ty::method[] {

            let methods: ty::method[] = ~[];

            // Outer methods.
            for m: @ast::method  in ao.methods {
                methods += ~[ty_of_method(fcx.ccx, m)];
            }

            // Inner methods.

            // Typecheck 'with_obj'.  If it exists, it had better have
            // object type.
            let with_obj_methods: ty::method[] = ~[];
            let with_obj_ty: ty::t = ty::mk_nil(fcx.ccx.tcx);
            let with_obj_sty: option::t[ty::sty] = none;
            alt ao.with_obj {
              none. { }
              some(e) {
                // If there's a with_obj, we push it onto the
                // obj_infos stack so that self-calls can be checked
                // within its context later.
                check_expr(fcx, e);
                with_obj_ty = expr_ty(fcx.ccx.tcx, e);
                with_obj_sty = some(structure_of(fcx, e.span, with_obj_ty));


                alt with_obj_sty {
                  none. { }
                  some(sty) {
                    alt sty {
                      ty::ty_obj(ms) { with_obj_methods = ms; }
                      _ {
                        // The user is trying to extend a
                        // non-object.
                        fcx.ccx.tcx.sess.span_fatal
                            (e.span, syntax::print::pprust::expr_to_str(e) +
                             " does not have object type");
                      }
                    }
                  }
                }
              }
            }

            fcx.ccx.obj_infos +=
                ~[anon_obj(ivec::map(ast::obj_field_from_anon_obj_field,
                                     fields), with_obj_sty)];

            methods += with_obj_methods;
            ret methods;
        }

        let method_types = get_anon_obj_method_types(fcx, ao, fields);
        let ot = ty::mk_obj(fcx.ccx.tcx, ty::sort_methods(method_types));

        write::ty_only_fixup(fcx, id, ot);

        // Write the methods into the node type table.  (This happens in
        // collect::convert for regular objects.)
        let i = 0u;
        while i < ivec::len[@ast::method](ao.methods) {
            write::ty_only(fcx.ccx.tcx, ao.methods.(i).node.id,
                           ty::method_ty_to_fn_ty(fcx.ccx.tcx,
                                                  method_types.(i)));
            i += 1u;
        }

        // Typecheck the methods.
        for method: @ast::method  in ao.methods {
            check_method(fcx.ccx, method);
        }
        next_ty_var(fcx);

        // Now remove the info from the stack.
        ivec::pop[obj_info](fcx.ccx.obj_infos);
      }
      _ { fcx.ccx.tcx.sess.unimpl("expr type in typeck::check_expr"); }
    }
}

fn next_ty_var_id(fcx: @fn_ctxt) -> int {
    let id = *fcx.next_var_id;
    *fcx.next_var_id = fcx.next_var_id + 1;
    ret id;
}

fn next_ty_var(fcx: &@fn_ctxt) -> ty::t {
    ret ty::mk_var(fcx.ccx.tcx, next_ty_var_id(fcx));
}

fn get_obj_info(ccx: &@crate_ctxt) -> option::t[obj_info] {
    ret ivec::last[obj_info](ccx.obj_infos);
}

fn check_decl_initializer(fcx: &@fn_ctxt, nid: ast::node_id,
                          init: &ast::initializer) {
    check_expr(fcx, init.expr);
    let lty = ty::mk_var(fcx.ccx.tcx, lookup_local(fcx, init.expr.span, nid));
    alt init.op {
      ast::init_assign. {
        demand::simple(fcx, init.expr.span, lty,
                       expr_ty(fcx.ccx.tcx, init.expr));
      }
      ast::init_move. {
        demand::simple(fcx, init.expr.span, lty,
                       expr_ty(fcx.ccx.tcx, init.expr));
      }
      ast::init_recv. {
        let port_ty = ty::mk_port(fcx.ccx.tcx, lty);
        demand::simple(fcx, init.expr.span, port_ty,
                       expr_ty(fcx.ccx.tcx, init.expr));
      }
    }
}

fn check_decl_local(fcx: &@fn_ctxt, local: &@ast::local) {
    let a_id = local.node.id;
    alt fcx.locals.find(a_id) {
      none. {
        fcx.ccx.tcx.sess.bug("check_decl_local: local id not found " +
                                 local.node.ident);
      }
      some(i) {
        let t = ty::mk_var(fcx.ccx.tcx, i);
        write::ty_only_fixup(fcx, a_id, t);
        alt local.node.init {
          some(init) { check_decl_initializer(fcx, local.node.id, init); }
          _ {/* fall through */ }
        }
      }
    }
}

fn check_stmt(fcx: &@fn_ctxt, stmt: &@ast::stmt) {
    let node_id;
    alt stmt.node {
      ast::stmt_decl(decl, id) {
        node_id = id;
        alt decl.node {
          ast::decl_local(ls) {
            for l: @ast::local  in ls { check_decl_local(fcx, l); }
          }
          ast::decl_item(_) {/* ignore for now */ }
        }
      }
      ast::stmt_expr(expr, id) { node_id = id; check_expr(fcx, expr); }
    }
    write::nil_ty(fcx.ccx.tcx, node_id);
}

fn check_block(fcx: &@fn_ctxt, blk: &ast::blk) {
    for s: @ast::stmt  in blk.node.stmts { check_stmt(fcx, s); }
    alt blk.node.expr {
      none. { write::nil_ty(fcx.ccx.tcx, blk.node.id); }
      some(e) {
        check_expr(fcx, e);
        let ety = expr_ty(fcx.ccx.tcx, e);
        write::ty_only_fixup(fcx, blk.node.id, ety);
      }
    }
}

fn check_const(ccx: &@crate_ctxt, sp: &span, e: &@ast::expr,
               id: &ast::node_id) {
    // FIXME: this is kinda a kludge; we manufacture a fake function context
    // and statement context for checking the initializer expression.

    let rty = node_id_to_type(ccx.tcx, id);
    let fixups: ast::node_id[] = ~[];
    let fcx: @fn_ctxt =
        @{ret_ty: rty,
          purity: ast::pure_fn,
          var_bindings: ty::unify::mk_var_bindings(),
          locals: new_int_hash[int](),
          local_names: new_int_hash[ast::ident](),
          next_var_id: @mutable 0,
          mutable fixups: fixups,
          ccx: ccx};
    check_expr(fcx, e);
}

fn check_fn(ccx: &@crate_ctxt, f: &ast::_fn, id: &ast::node_id,
            old_fcx: &option::t[@fn_ctxt]) {
    let decl = f.decl;
    let body = f.body;
    let gather_result = gather_locals(ccx, f, id, old_fcx);
    let fixups: ast::node_id[] = ~[];
    let fcx: @fn_ctxt =
        @{ret_ty: ast_ty_to_ty_crate(ccx, decl.output),
          purity: decl.purity,
          var_bindings: gather_result.var_bindings,
          locals: gather_result.locals,
          local_names: gather_result.local_names,
          next_var_id: gather_result.next_var_id,
          mutable fixups: fixups,
          ccx: ccx};

    check_block(fcx, body);
    alt decl.purity {
      ast::pure_fn. {

        // This just checks that the declared type is bool, and trusts
        // that that's the actual return type.
        if !ty::type_is_bool(ccx.tcx, fcx.ret_ty) {
            ccx.tcx.sess.span_fatal(body.span,
                                    "Non-boolean return type in pred");
        }
      }
      _ { }
    }

    if option::is_some(body.node.expr) {
        let tail_expr = option::get(body.node.expr);
        let tail_expr_ty = expr_ty(ccx.tcx, tail_expr);
        // Have to exclude ty_nil to allow functions to end in
        // while expressions, etc.
        let nil = ty::mk_nil(fcx.ccx.tcx);
        if !are_compatible(fcx, nil, tail_expr_ty) {
            demand::simple(fcx, tail_expr.span, fcx.ret_ty, tail_expr_ty);
        }
    }

    writeback::resolve_type_vars_in_block(fcx, body);
}

fn check_method(ccx: &@crate_ctxt, method: &@ast::method) {
    check_fn(ccx, method.node.meth, method.node.id, none);
}

fn check_item(ccx: @crate_ctxt, it: &@ast::item) {
    alt it.node {
      ast::item_const(_, e) { check_const(ccx, it.span, e, it.id); }
      ast::item_fn(f, _) { check_fn(ccx, f, it.id, none); }
      ast::item_res(f, dtor_id, _, _) { check_fn(ccx, f, dtor_id, none); }
      ast::item_obj(ob, _, _) {
        // We're entering an object, so gather up the info we need.
        ccx.obj_infos += ~[regular_obj(ob.fields, it.id)];

        // Typecheck the methods.
        for method: @ast::method  in ob.methods { check_method(ccx, method); }
        option::may[@ast::method](bind check_method(ccx, _), ob.dtor);

        // Now remove the info from the stack.
        ivec::pop[obj_info](ccx.obj_infos);
      }
      _ {/* nothing to do */ }
    }
}

fn arg_is_argv_ty(tcx: &ty::ctxt, a: &ty::arg) -> bool {
    alt ty::struct(tcx, a.ty) {
      ty::ty_vec(mt) {
        if mt.mut != ast::imm { ret false; }
        alt ty::struct(tcx, mt.ty) {
          ty::ty_str. { ret true; }
          _ { ret false; }
        }
      }
      _ { ret false; }
    }
}

fn check_main_fn_ty(tcx: &ty::ctxt, main_id: &ast::node_id) {
    let main_t = ty::node_id_to_monotype(tcx, main_id);
    alt ty::struct(tcx, main_t) {
      ty::ty_fn(ast::proto_fn., args, rs, ast::return., constrs) {
        let ok = ivec::len(constrs) == 0u;
        ok &= ty::type_is_nil(tcx, rs);
        let num_args = ivec::len(args);
        ok &=
            num_args == 0u || num_args == 1u && arg_is_argv_ty(tcx, args.(0));
        if !ok {
            tcx.sess.err("Wrong type in main function: found " +
                             ty_to_str(tcx, main_t));
        }
      }
      _ {
        tcx.sess.err("Main has a non-function type: found" +
                         ty_to_str(tcx, main_t));
      }
    }
}

fn check_for_main_fn(tcx: &ty::ctxt, crate: &@ast::crate) {
    if !tcx.sess.get_opts().library {
        alt tcx.sess.get_main_id() {
          some(id) { check_main_fn_ty(tcx, id); }
          none. { tcx.sess.span_err(crate.span, "Main function not found"); }
        }
    }
}

fn check_crate(tcx: &ty::ctxt, crate: &@ast::crate) {
    collect::collect_item_types(tcx, crate);

    let obj_infos: obj_info[] = ~[];

    let ccx = @{mutable obj_infos: obj_infos, tcx: tcx};
    let visit =
        {visit_item_pre: bind check_item(ccx, _)
            with walk::default_visitor()};
    walk::walk_crate(visit, *crate);
    check_for_main_fn(tcx, crate);
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
