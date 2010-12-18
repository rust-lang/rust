import front.ast;
import front.ast.ann;
import front.ast.mutability;
import middle.fold;
import driver.session;
import util.common;
import util.common.append;
import util.common.span;

import std._str;
import std._uint;
import std._vec;
import std.map;
import std.map.hashmap;
import std.option;
import std.option.none;
import std.option.some;

type ty_table = hashmap[ast.def_id, @ty];
type crate_ctxt = rec(session.session sess,
                      @ty_table item_types,
                      mutable int next_var_id);

type fn_ctxt = rec(@ty ret_ty,
                   @ty_table locals,
                   @crate_ctxt ccx);

type arg = rec(ast.mode mode, @ty ty);
type field = rec(ast.ident ident, @ty ty);
type method = rec(ast.ident ident, vec[arg] inputs, @ty output);

// NB: If you change this, you'll probably want to change the corresponding
// AST structure in front/ast.rs as well.
type ty = rec(sty struct, mutability mut, option.t[str] cname);
tag sty {
    ty_nil;
    ty_bool;
    ty_int;
    ty_uint;
    ty_machine(util.common.ty_mach);
    ty_char;
    ty_str;
    ty_tag(ast.def_id);
    ty_box(@ty);
    ty_vec(@ty);
    ty_tup(vec[@ty]);
    ty_rec(vec[field]);
    ty_fn(vec[arg], @ty);                           // TODO: effect
    ty_obj(vec[method]);
    ty_var(int);                                    // ephemeral type var
    ty_local(ast.def_id);                           // type of a local var
    ty_param(ast.def_id);                           // fn type param
    // TODO: ty_fn_arg(@ty), for a possibly-aliased function argument
}

tag type_err {
    terr_mismatch;
    terr_tuple_size(uint, uint);
    terr_tuple_mutability;
    terr_record_size(uint, uint);
    terr_record_mutability;
    terr_record_fields(ast.ident,ast.ident);
    terr_arg_count;
}

tag unify_result {
    ures_ok(@ty);
    ures_err(type_err, @ty, @ty);
}

// Used for ast_ty_to_ty() below.
type ty_getter = fn(ast.def_id) -> @ty;

// Error-reporting utility functions

fn ast_ty_to_str(&@ast.ty ty) -> str {

    fn ast_fn_input_to_str(&rec(ast.mode mode, @ast.ty ty) input) -> str {
        auto s;
        if (mode_is_alias(input.mode)) {
            s = "&";
        } else {
            s = "";
        }

        ret s + ast_ty_to_str(input.ty);
    }

    fn ast_ty_field_to_str(&ast.ty_field f) -> str {
        ret ast_ty_to_str(f.ty) + " " + f.ident;
    }

    auto s;
    alt (ty.node) {
        case (ast.ty_nil)          { s = "()";                            }
        case (ast.ty_bool)         { s = "bool";                          }
        case (ast.ty_int)          { s = "int";                           }
        case (ast.ty_uint)         { s = "uint";                          }
        case (ast.ty_machine(?tm)) { s = common.ty_mach_to_str(tm);       }
        case (ast.ty_char)         { s = "char";                          }
        case (ast.ty_str)          { s = "str";                           }
        case (ast.ty_box(?t))      { s = "@" + ast_ty_to_str(t);          }
        case (ast.ty_vec(?t))      { s = "vec[" + ast_ty_to_str(t) + "]"; }

        case (ast.ty_tup(?elts)) {
            auto f = ast_ty_to_str;
            s = "tup(";
            s += _str.connect(_vec.map[@ast.ty,str](f, elts), ",");
            s += ")";
        }

        case (ast.ty_rec(?fields)) {
            auto f = ast_ty_field_to_str;
            s = "rec(";
            s += _str.connect(_vec.map[ast.ty_field,str](f, fields), ",");
            s += ")";
        }

        case (ast.ty_fn(?inputs, ?output)) {
            auto f = ast_fn_input_to_str;
            s = "fn(";
            auto is = _vec.map[rec(ast.mode mode, @ast.ty ty),str](f, inputs);
            s += _str.connect(is, ", ");
            s += ")";

            if (output.node != ast.ty_nil) {
                s += " -> " + ast_ty_to_str(output);
            }
        }

        case (ast.ty_path(?path, _)) {
            s = path_to_str(path);
        }

        case (ast.ty_mutable(?t)) {
            s = "mutable " + ast_ty_to_str(t);
        }

        case (_) {
            fail;   // FIXME: typestate bug
        }
    }

    ret s;
}

fn name_to_str(&ast.name nm) -> str {
    auto result = nm.node.ident;
    if (_vec.len[@ast.ty](nm.node.types) > 0u) {
        auto f = ast_ty_to_str;
        result += "[";
        result += _str.connect(_vec.map[@ast.ty,str](f, nm.node.types), ",");
        result += "]";
    }
    ret result;
}

fn path_to_str(&ast.path path) -> str {
    auto f = name_to_str;
    ret _str.connect(_vec.map[ast.name,str](f, path), ".");
}

fn ty_to_str(&@ty typ) -> str {

    fn fn_input_to_str(&rec(ast.mode mode, @ty ty) input) -> str {
        auto s;
        if (mode_is_alias(input.mode)) {
            s = "&";
        } else {
            s = "";
        }

        ret s + ty_to_str(input.ty);
    }

    fn fn_to_str(option.t[ast.ident] ident,
                 vec[arg] inputs, @ty output) -> str {
            auto f = fn_input_to_str;
            auto s = "fn";
            alt (ident) {
                case (some[ast.ident](?i)) {
                    s += " ";
                    s += i;
                }
                case (_) { }
            }

            s += "(";
            s += _str.connect(_vec.map[arg,str](f, inputs), ", ");
            s += ")";

            if (output.struct != ty_nil) {
                s += " -> " + ty_to_str(output);
            }
            ret s;
    }

    fn method_to_str(&method m) -> str {
        ret fn_to_str(some[ast.ident](m.ident), m.inputs, m.output) + ";";
    }

    fn field_to_str(&field f) -> str {
        ret ty_to_str(f.ty) + " " + f.ident;
    }

    auto s = "";
    if (typ.mut == ast.mut) {
        s += "mutable ";
    }

    alt (typ.struct) {
        case (ty_nil)          { s = "()";                        }
        case (ty_bool)         { s = "bool";                      }
        case (ty_int)          { s = "int";                       }
        case (ty_uint)         { s = "uint";                      }
        case (ty_machine(?tm)) { s = common.ty_mach_to_str(tm);   }
        case (ty_char)         { s = "char";                      }
        case (ty_str)          { s = "str";                       }
        case (ty_box(?t))      { s = "@" + ty_to_str(t);          }
        case (ty_vec(?t))      { s = "vec[" + ty_to_str(t) + "]"; }

        case (ty_tup(?elems)) {
            auto f = ty_to_str;
            auto strs = _vec.map[@ty,str](f, elems);
            s = "tup(" + _str.connect(strs, ",") + ")";
        }

        case (ty_rec(?elems)) {
            auto f = field_to_str;
            auto strs = _vec.map[field,str](f, elems);
            s = "rec(" + _str.connect(strs, ",") + ")";
        }

        case (ty_tag(_)) {
            // The user should never see this if the cname is set properly!
            s = "<tag>";
        }

        case (ty_fn(?inputs, ?output)) {
            s = fn_to_str(none[ast.ident], inputs, output);
        }

        case (ty_obj(?meths)) {
            auto f = method_to_str;
            auto m = _vec.map[method,str](f, meths);
            s = "obj {\n\t" + _str.connect(m, "\n\t") + "\n}";
        }

        case (ty_var(?v)) {
            s = "<T" + util.common.istr(v) + ">";
        }

        case (ty_param(?id)) {
            s = "<P" + util.common.istr(id._0) + ":" + util.common.istr(id._1)
                + ">";
        }
    }

    ret s;
}

// Replaces parameter types inside a type with type variables.
fn generalize_ty(@crate_ctxt cx, @ty t) -> @ty {
    state obj ty_generalizer(@crate_ctxt cx,
                             @hashmap[ast.def_id,@ty] ty_params_to_ty_vars) {
        fn fold_simple_ty(@ty t) -> @ty {
            alt (t.struct) {
                case (ty_param(?pid)) {
                    if (ty_params_to_ty_vars.contains_key(pid)) {
                        ret ty_params_to_ty_vars.get(pid);
                    }
                    auto var_ty = next_ty_var(cx);
                    ty_params_to_ty_vars.insert(pid, var_ty);
                    ret var_ty;
                }
                case (_) { /* fall through */ }
            }
            ret t;
        }
    }

    auto generalizer = ty_generalizer(cx, @common.new_def_hash[@ty]());
    ret fold_ty(generalizer, t);
}

// Parses the programmer's textual representation of a type into our internal
// notion of a type. `getter` is a function that returns the type
// corresponding to a definition ID.
fn ast_ty_to_ty(ty_getter getter, &@ast.ty ast_ty) -> @ty {
    fn ast_arg_to_arg(ty_getter getter, &rec(ast.mode mode, @ast.ty ty) arg)
            -> rec(ast.mode mode, @ty ty) {
        ret rec(mode=arg.mode, ty=ast_ty_to_ty(getter, arg.ty));
    }

    auto mut = ast.imm;
    auto sty;
    auto cname = none[str];
    alt (ast_ty.node) {
        case (ast.ty_nil)          { sty = ty_nil; }
        case (ast.ty_bool)         { sty = ty_bool; }
        case (ast.ty_int)          { sty = ty_int; }
        case (ast.ty_uint)         { sty = ty_uint; }
        case (ast.ty_machine(?tm)) { sty = ty_machine(tm); }
        case (ast.ty_char)         { sty = ty_char; }
        case (ast.ty_str)          { sty = ty_str; }
        case (ast.ty_box(?t))      { sty = ty_box(ast_ty_to_ty(getter, t)); }
        case (ast.ty_vec(?t))      { sty = ty_vec(ast_ty_to_ty(getter, t)); }
        case (ast.ty_tup(?fields)) {
            let vec[@ty] flds = vec();
            for (@ast.ty field in fields) {
                append[@ty](flds, ast_ty_to_ty(getter, field));
            }
            sty = ty_tup(flds);
        }
        case (ast.ty_rec(?fields)) {
            let vec[field] flds = vec();
            for (ast.ty_field f in fields) {
                append[field](flds, rec(ident=f.ident,
                                        ty=ast_ty_to_ty(getter, f.ty)));
            }
            sty = ty_rec(flds);
        }

        case (ast.ty_fn(?inputs, ?output)) {
            auto f = bind ast_arg_to_arg(getter, _);
            auto i = _vec.map[rec(ast.mode mode, @ast.ty ty),arg](f, inputs);
            sty = ty_fn(i, ast_ty_to_ty(getter, output));
        }

        case (ast.ty_path(?path, ?def)) {
            check (def != none[ast.def]);
            alt (option.get[ast.def](def)) {
                case (ast.def_ty(?id)) {
                    // TODO: maybe record cname chains so we can do
                    // "foo = int" like OCaml?
                    sty = getter(id).struct;
                }
                case (ast.def_ty_arg(?id))  { sty = ty_param(id); }
                case (_)                    { fail; }
            }

            cname = some(path_to_str(path));
        }

        case (ast.ty_mutable(?t)) {
            mut = ast.mut;
            auto t0 = ast_ty_to_ty(getter, t);
            sty = t0.struct;
            cname = t0.cname;
        }

        case (_) {
            fail;
        }
    }

    ret @rec(struct=sty, mut=mut, cname=cname);
}

// A convenience function to use a crate_ctxt to resolve names for
// ast_ty_to_ty.
fn ast_ty_to_ty_crate(@crate_ctxt ccx, &@ast.ty ast_ty) -> @ty {
    fn getter(@crate_ctxt ccx, ast.def_id id) -> @ty {
        check (ccx.item_types.contains_key(id));
        ret ccx.item_types.get(id);
    }
    auto f = bind getter(ccx, _);
    ret ast_ty_to_ty(f, ast_ty);
}

fn type_err_to_str(&type_err err) -> str {
    alt (err) {
        case (terr_mismatch) {
            ret "types differ";
        }
        case (terr_tuple_size(?e_sz, ?a_sz)) {
            ret "expected a tuple with " + _uint.to_str(e_sz, 10u) +
                " elements but found one with " + _uint.to_str(a_sz, 10u) +
                " elements";
        }
        case (terr_tuple_mutability) {
            ret "tuple elements differ in mutability";
        }
        case (terr_record_size(?e_sz, ?a_sz)) {
            ret "expected a record with " + _uint.to_str(e_sz, 10u) +
                " fields but found one with " + _uint.to_str(a_sz, 10u) +
                " fields";
        }
        case (terr_record_mutability) {
            ret "record elements differ in mutability";
        }
        case (terr_record_fields(?e_fld, ?a_fld)) {
            ret "expected a record with field '" + e_fld +
                "' but found one with field '" + a_fld +
                "'";
        }
        case (terr_arg_count) {
            ret "incorrect number of function parameters";
        }
    }
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

fn collect_item_types(@ast.crate crate) -> tup(@ast.crate, @ty_table) {

    type ty_item_table = hashmap[ast.def_id,@ast.item];

    fn getter(@ty_item_table id_to_ty_item,
              @ty_table item_to_ty,
              ast.def_id id) -> @ty {
        check (id_to_ty_item.contains_key(id));
        auto item = id_to_ty_item.get(id);
        ret ty_of_item(id_to_ty_item, item_to_ty, item);
    }

    fn ty_of_arg(@ty_item_table id_to_ty_item,
                 @ty_table item_to_ty,
                 &ast.arg a) -> arg {
        auto f = bind getter(id_to_ty_item, item_to_ty, _);
        ret rec(mode=a.mode, ty=ast_ty_to_ty(f, a.ty));
    }

    fn ty_of_method(@ty_item_table id_to_ty_item,
                    @ty_table item_to_ty,
                    &@ast.method m) -> method {
        auto get = bind getter(id_to_ty_item, item_to_ty, _);
        auto convert = bind ast_ty_to_ty(get, _);
        auto f = bind ty_of_arg(id_to_ty_item, item_to_ty, _);
        auto inputs = _vec.map[ast.arg,arg](f, m.node.meth.inputs);
        auto output = convert(m.node.meth.output);
        ret rec(ident=m.node.ident, inputs=inputs, output=output);
    }

    fn ty_of_obj(@ty_item_table id_to_ty_item,
                 @ty_table item_to_ty,
                 &ast._obj obj_info) -> @ty {
        auto f = bind ty_of_method(id_to_ty_item, item_to_ty, _);
        auto methods =
            _vec.map[@ast.method,method](f, obj_info.methods);

        auto t_obj = plain_ty(ty_obj(methods));
        ret t_obj;
    }

    fn ty_of_obj_ctor(@ty_item_table id_to_ty_item,
                      @ty_table item_to_ty,
                      &ast._obj obj_info) -> @ty {
        auto t_obj = ty_of_obj(id_to_ty_item, item_to_ty, obj_info);
        let vec[arg] t_inputs = vec();
        for (ast.obj_field f in obj_info.fields) {
            auto t_field = getter(id_to_ty_item, item_to_ty, f.id);
            append[arg](t_inputs, rec(mode=ast.alias, ty=t_field));
        }
        auto t_fn = plain_ty(ty_fn(t_inputs, t_obj));
        ret t_fn;
    }

    fn ty_of_item(@ty_item_table id_to_ty_item,
                  @ty_table item_to_ty,
                  @ast.item it) -> @ty {

        auto get = bind getter(id_to_ty_item, item_to_ty, _);
        auto convert = bind ast_ty_to_ty(get, _);

        alt (it.node) {

            case (ast.item_const(?ident, ?t, _, ?def_id, _)) {
                item_to_ty.insert(def_id, convert(t));
            }

            case (ast.item_fn(?ident, ?fn_info, _, ?def_id, _)) {
                // TODO: handle ty-params

                auto f = bind ty_of_arg(id_to_ty_item, item_to_ty, _);
                auto input_tys = _vec.map[ast.arg,arg](f, fn_info.inputs);
                auto output_ty = convert(fn_info.output);

                auto t_fn = plain_ty(ty_fn(input_tys, output_ty));
                item_to_ty.insert(def_id, t_fn);
                ret t_fn;
            }

            case (ast.item_obj(?ident, ?obj_info, _, ?def_id, _)) {
                // TODO: handle ty-params
                auto t_ctor = ty_of_obj_ctor(id_to_ty_item,
                                             item_to_ty,
                                             obj_info);
                item_to_ty.insert(def_id, t_ctor);
                ret t_ctor;
            }

            case (ast.item_ty(?ident, ?ty, _, ?def_id, _)) {
                if (item_to_ty.contains_key(def_id)) {
                    // Avoid repeating work.
                    ret item_to_ty.get(def_id);
                }

                // Tell ast_ty_to_ty() that we want to perform a recursive
                // call to resolve any named types.
                auto ty_ = convert(ty);
                item_to_ty.insert(def_id, ty_);
                ret ty_;
            }

            case (ast.item_tag(_, _, _, ?def_id)) {
                auto ty = plain_ty(ty_tag(def_id));
                item_to_ty.insert(def_id, ty);
                ret ty;
            }

            case (ast.item_mod(_, _, _)) { fail; }
        }
    }

    fn get_tag_variant_types(@ty_item_table id_to_ty_item,
                             @ty_table item_to_ty,
                             &ast.def_id tag_id,
                             &vec[ast.variant] variants) -> vec[ast.variant] {
        let vec[ast.variant] result = vec();

        for (ast.variant variant in variants) {
            // Nullary tag constructors get truned into constants; n-ary tag
            // constructors get turned into functions.
            auto result_ty;
            if (_vec.len[ast.variant_arg](variant.args) == 0u) {
                result_ty = plain_ty(ty_tag(tag_id));
            } else {
                // As above, tell ast_ty_to_ty() that trans_ty_item_to_ty()
                // should be called to resolve named types.
                auto f = bind getter(id_to_ty_item, item_to_ty, _);

                let vec[arg] args = vec();
                for (ast.variant_arg va in variant.args) {
                    auto arg_ty = ast_ty_to_ty(f, va.ty);
                    args += vec(rec(mode=ast.alias, ty=arg_ty));
                }
                result_ty = plain_ty(ty_fn(args, plain_ty(ty_tag(tag_id))));
            }

            item_to_ty.insert(variant.id, result_ty);

            auto variant_t = rec(ann=ast.ann_type(result_ty) with variant);
            result += vec(variant_t);
        }

        ret result;
    }

    // First pass: collect all type item IDs.
    auto module = crate.node.module;
    auto id_to_ty_item = @common.new_def_hash[@ast.item]();
    fn collect(&@ty_item_table id_to_ty_item, @ast.item i)
        -> @ty_item_table {
        alt (i.node) {
            case (ast.item_ty(_, _, _, ?def_id, _)) {
                id_to_ty_item.insert(def_id, i);
            }
            case (ast.item_tag(_, _, _, ?def_id)) {
                id_to_ty_item.insert(def_id, i);
            }
            case (_) { /* empty */ }
        }
        ret id_to_ty_item;
    }
    auto fld_1 = fold.new_identity_fold[@ty_item_table]();
    fld_1 = @rec(update_env_for_item = bind collect(_, _)
                 with *fld_1);
    fold.fold_crate[@ty_item_table](id_to_ty_item, fld_1, crate);



    // Second pass: translate the types of all items.
    let @ty_table item_to_ty = @common.new_def_hash[@ty]();

    type env = rec(@ty_item_table id_to_ty_item,
                   @ty_table item_to_ty);
    let @env e = @rec(id_to_ty_item=id_to_ty_item,
                      item_to_ty=item_to_ty);

    fn convert(&@env e, @ast.item i) -> @env {
        alt (i.node) {
            case (ast.item_mod(_, _, _)) {
                // ignore item_mod, it has no type.
            }
            case (_) {
                // This call populates the ty_table with the converted type of
                // the item in passing; we don't need to do anything else.
                ty_of_item(e.id_to_ty_item, e.item_to_ty, i);
            }
        }
        ret e;
    }

    fn fold_item_const(&@env e, &span sp, ast.ident i,
                       @ast.ty t, @ast.expr ex,
                       ast.def_id id, ast.ann a) -> @ast.item {
        check (e.item_to_ty.contains_key(id));
        auto ty = e.item_to_ty.get(id);
        auto item = ast.item_const(i, t, ex, id,
                                   ast.ann_type(ty));
        ret @fold.respan[ast.item_](sp, item);
    }

    fn fold_item_fn(&@env e, &span sp, ast.ident i,
                    &ast._fn f, vec[ast.ty_param] ty_params,
                    ast.def_id id, ast.ann a) -> @ast.item {
        check (e.item_to_ty.contains_key(id));
        auto ty = e.item_to_ty.get(id);
        auto item = ast.item_fn(i, f, ty_params, id,
                                ast.ann_type(ty));
        ret @fold.respan[ast.item_](sp, item);
    }

    fn get_ctor_obj_methods(@ty t) -> vec[method] {
        alt (t.struct) {
            case (ty_fn(_,?tobj)) {
                alt (tobj.struct) {
                    case (ty_obj(?tm)) {
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
                    ast.def_id id, ast.ann a) -> @ast.item {
        check (e.item_to_ty.contains_key(id));
        auto ty = e.item_to_ty.get(id);
        let vec[method] meth_tys = get_ctor_obj_methods(ty);
        let vec[@ast.method] methods = vec();

        let uint n = 0u;
        for (method meth_ty in meth_tys) {
            let @ast.method meth = ob.methods.(n);
            let ast.method_ m_;
            let @ast.method m;
            auto meth_tfn = plain_ty(ty_fn(meth_ty.inputs,
                                           meth_ty.output));
            m_ = rec(ann=ast.ann_type(meth_tfn) with meth.node);
            m = @rec(node=m_ with *meth);
            append[@ast.method](methods, m);
            n += 1u;
        }

        auto ob_ = rec(methods = methods with ob);
        auto item = ast.item_obj(i, ob_, ty_params, id,
                                 ast.ann_type(ty));
        ret @fold.respan[ast.item_](sp, item);
    }

    fn fold_item_ty(&@env e, &span sp, ast.ident i,
                    @ast.ty t, vec[ast.ty_param] ty_params,
                    ast.def_id id, ast.ann a) -> @ast.item {
        check (e.item_to_ty.contains_key(id));
        auto ty = e.item_to_ty.get(id);
        auto item = ast.item_ty(i, t, ty_params, id,
                                ast.ann_type(ty));
        ret @fold.respan[ast.item_](sp, item);
    }

    fn fold_item_tag(&@env e, &span sp, ast.ident i,
                     vec[ast.variant] variants,
                     vec[ast.ty_param] ty_params,
                     ast.def_id id) -> @ast.item {
        auto variants_t = get_tag_variant_types(e.id_to_ty_item,
                                                e.item_to_ty,
                                                id, variants);
        auto item = ast.item_tag(i, variants_t, ty_params, id);
        ret @fold.respan[ast.item_](sp, item);
    }

    auto fld_2 = fold.new_identity_fold[@env]();
    fld_2 =
        @rec(update_env_for_item = bind convert(_,_),
             fold_item_const = bind fold_item_const(_,_,_,_,_,_,_),
             fold_item_fn    = bind fold_item_fn(_,_,_,_,_,_,_),
             fold_item_obj   = bind fold_item_obj(_,_,_,_,_,_,_),
             fold_item_ty    = bind fold_item_ty(_,_,_,_,_,_,_),
             fold_item_tag   = bind fold_item_tag(_,_,_,_,_,_)
             with *fld_2);
    auto crate_ = fold.fold_crate[@env](e, fld_2, crate);
    ret tup(crate_, item_to_ty);
}

// Expression utilities

fn field_num(session.session sess, &span sp, &ast.ident id) -> uint {
    let uint accum = 0u;
    let uint i = 0u;
    for (u8 c in id) {
        if (i == 0u) {
            if (c != ('_' as u8)) {
                sess.span_err(sp,
                              "bad numeric field on tuple: "
                              + "missing leading underscore");
            }
        } else {
            if (('0' as u8) <= c && c <= ('9' as u8)) {
                accum *= 10u;
                accum += (c as uint) - ('0' as uint);
            } else {
                auto s = "";
                s += c;
                sess.span_err(sp,
                              "bad numeric field on tuple: "
                              + " non-digit character: "
                              + s);
            }
        }
        i += 1u;
    }
    ret accum;
}

fn field_idx(session.session sess, &span sp,
             &ast.ident id, vec[field] fields) -> uint {
    let uint i = 0u;
    for (field f in fields) {
        if (_str.eq(f.ident, id)) {
            ret i;
        }
        i += 1u;
    }
    sess.span_err(sp, "unknown field '" + id + "' of record");
    fail;
}

fn method_idx(session.session sess, &span sp,
              &ast.ident id, vec[method] meths) -> uint {
    let uint i = 0u;
    for (method m in meths) {
        if (_str.eq(m.ident, id)) {
            ret i;
        }
        i += 1u;
    }
    sess.span_err(sp, "unknown method '" + id + "' of obj");
    fail;
}

fn is_lval(@ast.expr expr) -> bool {
    alt (expr.node) {
        case (ast.expr_field(_,_,_))    { ret true;  }
        case (ast.expr_index(_,_,_))    { ret true;  }
        case (ast.expr_name(_,_,_))     { ret true;  }
        case (_)                        { ret false; }
    }
}

// Type folds

type ty_fold = state obj {
    fn fold_simple_ty(@ty ty) -> @ty;
};

fn fold_ty(ty_fold fld, @ty t) -> @ty {
    fn rewrap(@ty orig, &sty new) -> @ty {
        ret @rec(struct=new, mut=orig.mut, cname=orig.cname);
    }

    alt (t.struct) {
        case (ty_nil)           { ret fld.fold_simple_ty(t); }
        case (ty_bool)          { ret fld.fold_simple_ty(t); }
        case (ty_int)           { ret fld.fold_simple_ty(t); }
        case (ty_uint)          { ret fld.fold_simple_ty(t); }
        case (ty_machine(_))    { ret fld.fold_simple_ty(t); }
        case (ty_char)          { ret fld.fold_simple_ty(t); }
        case (ty_str)           { ret fld.fold_simple_ty(t); }
        case (ty_tag(_))        { ret fld.fold_simple_ty(t); }
        case (ty_box(?subty)) {
            ret rewrap(t, ty_box(fold_ty(fld, subty)));
        }
        case (ty_vec(?subty)) {
            ret rewrap(t, ty_vec(fold_ty(fld, subty)));
        }
        case (ty_tup(?subtys)) {
            let vec[@ty] new_subtys = vec();
            for (@ty subty in subtys) {
                new_subtys += vec(fold_ty(fld, subty));
            }
            ret rewrap(t, ty_tup(new_subtys));
        }
        case (ty_rec(?fields)) {
            let vec[field] new_fields = vec();
            for (field fl in fields) {
                auto new_ty = fold_ty(fld, fl.ty);
                new_fields += vec(rec(ident=fl.ident, ty=new_ty));
            }
            ret rewrap(t, ty_rec(new_fields));
        }
        case (ty_fn(?args, ?ret_ty)) {
            let vec[arg] new_args = vec();
            for (arg a in args) {
                auto new_ty = fold_ty(fld, a.ty);
                new_args += vec(rec(mode=a.mode, ty=new_ty));
            }
            ret rewrap(t, ty_fn(new_args, fold_ty(fld, ret_ty)));
        }
        case (ty_obj(?methods)) {
            let vec[method] new_methods = vec();
            for (method m in methods) {
                let vec[arg] new_args = vec();
                for (arg a in m.inputs) {
                    new_args += vec(rec(mode=a.mode, ty=fold_ty(fld, a.ty)));
                }
                new_methods += vec(rec(ident=m.ident, inputs=new_args,
                                       output=fold_ty(fld, m.output)));
            }
            ret rewrap(t, ty_obj(new_methods));
        }
        case (ty_var(_))        { ret fld.fold_simple_ty(t); }
        case (ty_local(_))      { ret fld.fold_simple_ty(t); }
        case (ty_param(_))      { ret fld.fold_simple_ty(t); }
    }

    ret t;
}

// Type utilities

// FIXME: remove me when == works on these tags.
fn mode_is_alias(ast.mode m) -> bool {
    alt (m) {
        case (ast.val) { ret false; }
        case (ast.alias) { ret true; }
    }
    fail;
}

fn type_is_nil(@ty t) -> bool {
    alt (t.struct) {
        case (ty_nil) { ret true; }
        case (_) { ret false; }
    }
    fail;
}

fn type_is_structural(@ty t) -> bool {
    alt (t.struct) {
        case (ty_tup(_)) { ret true; }
        case (ty_rec(_)) { ret true; }
        case (ty_tag(_)) { ret true; }
        case (_) { ret false; }
    }
    fail;
}

fn type_is_binding(@ty t) -> bool {
    alt (t.struct) {
        // FIXME: cover obj when we support it.
        case (ty_fn(_,_)) { ret true; }
        case (_) { ret false; }
    }
    fail;
}

fn type_is_boxed(@ty t) -> bool {
    alt (t.struct) {
        case (ty_str) { ret true; }
        case (ty_vec(_)) { ret true; }
        case (ty_box(_)) { ret true; }
        case (_) { ret false; }
    }
    fail;
}

fn type_is_scalar(@ty t) -> bool {
    alt (t.struct) {
        case (ty_nil) { ret true; }
        case (ty_bool) { ret true; }
        case (ty_int) { ret true; }
        case (ty_uint) { ret true; }
        case (ty_machine(_)) { ret true; }
        case (ty_char) { ret true; }
        case (_) { ret false; }
    }
    fail;
}


fn type_is_integral(@ty t) -> bool {
    alt (t.struct) {
        case (ty_int) { ret true; }
        case (ty_uint) { ret true; }
        case (ty_machine(?m)) {
            alt (m) {
                case (common.ty_i8) { ret true; }
                case (common.ty_i16) { ret true; }
                case (common.ty_i32) { ret true; }
                case (common.ty_i64) { ret true; }

                case (common.ty_u8) { ret true; }
                case (common.ty_u16) { ret true; }
                case (common.ty_u32) { ret true; }
                case (common.ty_u64) { ret true; }
                case (_) { ret false; }
            }
        }
        case (ty_char) { ret true; }
        case (_) { ret false; }
    }
    fail;
}

fn type_is_fp(@ty t) -> bool {
    alt (t.struct) {
        case (ty_machine(?tm)) {
            alt (tm) {
                case (common.ty_f32) { ret true; }
                case (common.ty_f64) { ret true; }
                case (_) { ret false; }
            }
        }
        case (_) { ret false; }
    }
    fail;
}

fn type_is_signed(@ty t) -> bool {
    alt (t.struct) {
        case (ty_int) { ret true; }
        case (ty_machine(?tm)) {
            alt (tm) {
                case (common.ty_i8) { ret true; }
                case (common.ty_i16) { ret true; }
                case (common.ty_i32) { ret true; }
                case (common.ty_i64) { ret true; }
                case (_) { ret false; }
            }
        }
        case (_) { ret false; }
    }
    fail;
}

fn plain_ty(&sty st) -> @ty {
    ret @rec(struct=st, mut=ast.imm, cname=none[str]);
}

fn hash_ty(&@ty t) -> uint {
    ret _str.hash(ty_to_str(t));
}

fn eq_ty(&@ty a, &@ty b) -> bool {
    // FIXME: this is gross, but I think it's safe, and I don't think writing
    // a giant function to handle all the cases is necessary when structural
    // equality will someday save the day.
    ret _str.eq(ty_to_str(a), ty_to_str(b));
}

fn ann_to_type(&ast.ann ann) -> @ty {
    alt (ann) {
        case (ast.ann_none) {
            // shouldn't happen, but can until the typechecker is complete
            ret plain_ty(ty_var(-1));    // FIXME: broken, broken, broken
        }
        case (ast.ann_type(?ty)) {
            ret ty;
        }
    }
}

fn count_ty_params(@ty t) -> uint {
    state obj ty_param_counter(@mutable vec[ast.def_id] param_ids) {
        fn fold_simple_ty(@ty t) -> @ty {
            alt (t.struct) {
                case (ty_param(?param_id)) {
                    for (ast.def_id other_param_id in *param_ids) {
                        if (param_id._0 == other_param_id._0 &&
                                param_id._1 == other_param_id._1) {
                            ret t;
                        }
                    }
                    *param_ids += vec(param_id);
                }
                case (_) { /* fall through */ }
            }
            ret t;
        }
    }
  
    let vec[ast.def_id] param_ids_inner = vec();
    let @mutable vec[ast.def_id] param_ids = @mutable param_ids_inner;
    fold_ty(ty_param_counter(param_ids), t);
    ret _vec.len[ast.def_id](*param_ids);
}

// Type accessors for AST nodes

fn stmt_ty(@ast.stmt s) -> @ty {
    alt (s.node) {
        case (ast.stmt_expr(?e)) {
            ret expr_ty(e);
        }
        case (_) {
            ret plain_ty(ty_nil);
        }
    }
}

fn block_ty(&ast.block b) -> @ty {
    alt (b.node.expr) {
        case (some[@ast.expr](?e)) { ret expr_ty(e); }
        case (none[@ast.expr])     { ret plain_ty(ty_nil); }
    }
}

fn pat_ty(@ast.pat pat) -> @ty {
    alt (pat.node) {
        case (ast.pat_wild(?ann))           { ret ann_to_type(ann); }
        case (ast.pat_bind(_, _, ?ann))     { ret ann_to_type(ann); }
        case (ast.pat_tag(_, _, _, ?ann))   { ret ann_to_type(ann); }
    }
    fail;   // not reached
}

fn expr_ty(@ast.expr expr) -> @ty {
    alt (expr.node) {
        case (ast.expr_vec(_, ?ann))          { ret ann_to_type(ann); }
        case (ast.expr_tup(_, ?ann))          { ret ann_to_type(ann); }
        case (ast.expr_rec(_, ?ann))          { ret ann_to_type(ann); }
        case (ast.expr_call(_, _, ?ann))      { ret ann_to_type(ann); }
        case (ast.expr_binary(_, _, _, ?ann)) { ret ann_to_type(ann); }
        case (ast.expr_unary(_, _, ?ann))     { ret ann_to_type(ann); }
        case (ast.expr_lit(_, ?ann))          { ret ann_to_type(ann); }
        case (ast.expr_cast(_, _, ?ann))      { ret ann_to_type(ann); }
        case (ast.expr_if(_, _, _, ?ann))     { ret ann_to_type(ann); }
        case (ast.expr_while(_, _, ?ann))     { ret ann_to_type(ann); }
        case (ast.expr_do_while(_, _, ?ann))  { ret ann_to_type(ann); }
        case (ast.expr_alt(_, _, ?ann))       { ret ann_to_type(ann); }
        case (ast.expr_block(_, ?ann))        { ret ann_to_type(ann); }
        case (ast.expr_assign(_, _, ?ann))    { ret ann_to_type(ann); }
        case (ast.expr_assign_op(_, _, _, ?ann))
                                              { ret ann_to_type(ann); }
        case (ast.expr_field(_, _, ?ann))     { ret ann_to_type(ann); }
        case (ast.expr_index(_, _, ?ann))     { ret ann_to_type(ann); }
        case (ast.expr_name(_, _, ?ann))      { ret ann_to_type(ann); }
    }
    fail;
}

// Type unification

fn unify(&fn_ctxt fcx, @ty expected, @ty actual) -> unify_result {
    // Wraps the given type in an appropriate cname.
    //
    // TODO: This doesn't do anything yet. We should carry the cname up from
    // the expected and/or actual types when unification results in a type
    // identical to one or both of the two. The precise algorithm for this is
    // something we'll probably need to develop over time.

    // Simple structural type comparison.
    fn struct_cmp(@ty expected, @ty actual) -> unify_result {
        if (expected.struct == actual.struct) {
            ret ures_ok(expected);
        }

        ret ures_err(terr_mismatch, expected, actual);
    }

    fn unify_step(&fn_ctxt fcx, &hashmap[int,@ty] bindings, @ty expected,
                  @ty actual) -> unify_result {
        // TODO: rewrite this using tuple pattern matching when available, to
        // avoid all this rightward drift and spikiness.

        // If the RHS is a variable type, then just do the appropriate
        // binding.
        alt (actual.struct) {
            case (ty_var(?actual_id)) {
                alt (bindings.find(actual_id)) {
                    case (some[@ty](?actual_ty)) {
                        // FIXME: change the binding here?
                        // FIXME: "be"
                        ret unify_step(fcx, bindings, expected, actual_ty);
                    }
                    case (none[@ty]) {
                        bindings.insert(actual_id, expected);
                        ret ures_ok(expected);
                    }
                }
            }
            case (ty_local(?actual_id)) {
                check (fcx.locals.contains_key(actual_id));
                auto actual_ty = fcx.locals.get(actual_id);
                auto result = unify_step(fcx, bindings, expected, actual_ty);
                alt (result) {
                    case (ures_ok(?result_ty)) {
                        fcx.locals.insert(actual_id, result_ty);
                    }
                    case (_) { /* empty */ }
                }
                ret result;
            }
            case (_) { /* empty */ }
        }

        alt (expected.struct) {
            case (ty_nil)        { ret struct_cmp(expected, actual); }
            case (ty_bool)       { ret struct_cmp(expected, actual); }
            case (ty_int)        { ret struct_cmp(expected, actual); }
            case (ty_uint)       { ret struct_cmp(expected, actual); }
            case (ty_machine(_)) { ret struct_cmp(expected, actual); }
            case (ty_char)       { ret struct_cmp(expected, actual); }
            case (ty_str)        { ret struct_cmp(expected, actual); }

            case (ty_tag(?expected_id)) {
                alt (actual.struct) {
                    case (ty_tag(?actual_id)) {
                        if (expected_id._0 == actual_id._0 &&
                                expected_id._1 == actual_id._1) {
                            ret ures_ok(expected);
                        }
                    }
                    case (_) { /* fall through */ }
                }

                ret ures_err(terr_mismatch, expected, actual);
            }

            case (ty_box(?expected_sub)) {
                alt (actual.struct) {
                    case (ty_box(?actual_sub)) {
                        auto result = unify_step(fcx,
                                                 bindings,
                                                 expected_sub,
                                                 actual_sub);
                        alt (result) {
                            case (ures_ok(?result_sub)) {
                                ret ures_ok(plain_ty(ty_box(result_sub)));
                            }
                            case (_) {
                                ret result;
                            }
                        }
                    }

                    // TODO: ty_var

                    case (_) {
                        ret ures_err(terr_mismatch, expected, actual);
                    }
                }
            }

            case (ty_vec(?expected_sub)) {
                alt (actual.struct) {
                    case (ty_vec(?actual_sub)) {
                        auto result = unify_step(fcx,
                                                 bindings,
                                                 expected_sub,
                                                 actual_sub);
                        alt (result) {
                            case (ures_ok(?result_sub)) {
                                ret ures_ok(plain_ty(ty_vec(result_sub)));
                            }
                            case (_) {
                                ret result;
                            }
                        }
                    }

                    // TODO: ty_var

                    case (_) {
                        ret ures_err(terr_mismatch, expected, actual);
                   }
                }
            }

            case (ty_tup(?expected_elems)) {
                alt (actual.struct) {
                    case (ty_tup(?actual_elems)) {
                        auto expected_len = _vec.len[@ty](expected_elems);
                        auto actual_len = _vec.len[@ty](actual_elems);
                        if (expected_len != actual_len) {
                            auto err = terr_tuple_size(expected_len,
                                                       actual_len);
                            ret ures_err(err, expected, actual);
                        }

                        // TODO: implement an iterator that can iterate over
                        // two arrays simultaneously.
                        let vec[@ty] result_elems = vec();
                        auto i = 0u;
                        while (i < expected_len) {
                            auto expected_elem = expected_elems.(i);
                            auto actual_elem = actual_elems.(i);
                            if (expected_elem.mut != actual_elem.mut) {
                                auto err = terr_tuple_mutability;
                                ret ures_err(err, expected, actual);
                            }

                            auto result = unify_step(fcx,
                                                     bindings,
                                                     expected_elem,
                                                     actual_elem);
                            alt (result) {
                                case (ures_ok(?rty)) {
                                    append[@ty](result_elems,rty);
                                }
                                case (_) {
                                    ret result;
                                }
                            }

                            i += 1u;
                        }

                        ret ures_ok(plain_ty(ty_tup(result_elems)));
                    }

                    // TODO: ty_var

                    case (_) {
                        ret ures_err(terr_mismatch, expected, actual);
                    }
                }
            }

            case (ty_rec(?expected_fields)) {
                alt (actual.struct) {
                    case (ty_rec(?actual_fields)) {
                        auto expected_len = _vec.len[field](expected_fields);
                        auto actual_len = _vec.len[field](actual_fields);
                        if (expected_len != actual_len) {
                            auto err = terr_record_size(expected_len,
                                                        actual_len);
                            ret ures_err(err, expected, actual);
                        }

                        // TODO: implement an iterator that can iterate over
                        // two arrays simultaneously.
                        let vec[field] result_fields = vec();
                        auto i = 0u;
                        while (i < expected_len) {
                            auto expected_field = expected_fields.(i);
                            auto actual_field = actual_fields.(i);
                            if (expected_field.ty.mut
                                != actual_field.ty.mut) {
                                auto err = terr_record_mutability;
                                ret ures_err(err, expected, actual);
                            }

                            if (!_str.eq(expected_field.ident,
                                        actual_field.ident)) {
                                auto err =
                                    terr_record_fields(expected_field.ident,
                                                       actual_field.ident);
                                ret ures_err(err, expected, actual);
                            }

                            auto result = unify_step(fcx,
                                                     bindings,
                                                     expected_field.ty,
                                                     actual_field.ty);
                            alt (result) {
                                case (ures_ok(?rty)) {
                                    append[field]
                                        (result_fields,
                                         rec(ty=rty with expected_field));
                                }
                                case (_) {
                                    ret result;
                                }
                            }

                            i += 1u;
                        }

                        ret ures_ok(plain_ty(ty_rec(result_fields)));
                    }

                    // TODO: ty_var

                    case (_) {
                        ret ures_err(terr_mismatch, expected, actual);
                    }
                }
            }

            case (ty_fn(?expected_inputs, ?expected_output)) {
                alt (actual.struct) {
                    case (ty_fn(?actual_inputs, ?actual_output)) {
                        auto expected_len = _vec.len[arg](expected_inputs);
                        auto actual_len = _vec.len[arg](actual_inputs);
                        if (expected_len != actual_len) {
                            ret ures_err(terr_arg_count, expected, actual);
                        }

                        // TODO: as above, we should have an iter2 iterator.
                        let vec[arg] result_ins = vec();
                        auto i = 0u;
                        while (i < expected_len) {
                            auto expected_input = expected_inputs.(i);
                            auto actual_input = actual_inputs.(i);

                            // This should be safe, I think?
                            auto result_mode;
                            if (mode_is_alias(expected_input.mode) ||
                                    mode_is_alias(actual_input.mode)) {
                                result_mode = ast.alias;
                            } else {
                                result_mode = ast.val;
                            }

                            auto result = unify_step(fcx,
                                                     bindings,
                                                     actual_input.ty,
                                                     expected_input.ty);

                            alt (result) {
                                case (ures_ok(?rty)) {
                                    result_ins += vec(rec(mode=result_mode,
                                                          ty=rty));
                                }

                                case (_) {
                                    ret result;
                                }
                            }

                            i += 1u;
                        }

                        // Check the output.
                        auto result_out;
                        auto result = unify_step(fcx,
                                                 bindings,
                                                 expected_output,
                                                 actual_output);
                        alt (result) {
                            case (ures_ok(?rty)) {
                                result_out = rty;
                            }

                            case (_) {
                                ret result;
                            }
                        }

                        ret ures_ok(plain_ty(ty_fn(result_ins, result_out)));
                    }

                    case (_) {
                        ret ures_err(terr_mismatch, expected, actual);
                    }
                }
            }

            case (ty_var(?expected_id)) {
                alt (bindings.find(expected_id)) {
                    case (some[@ty](?expected_ty)) {
                        // FIXME: change the binding here?
                        // FIXME: "be"
                        ret unify_step(fcx, bindings, expected_ty, actual);
                    }
                    case (none[@ty]) {
                        bindings.insert(expected_id, actual);
                        ret ures_ok(actual);
                    }
                }
            }

            case (ty_local(?expected_id)) {
                check (fcx.locals.contains_key(expected_id));
                auto expected_ty = fcx.locals.get(expected_id);
                auto result = unify_step(fcx, bindings, expected_ty, actual);
                alt (result) {
                    case (ures_ok(?result_ty)) {
                        fcx.locals.insert(expected_id, result_ty);
                    }
                    case (_) { /* empty */ }
                }
                ret result;
            }

            case (ty_param(?expected_id)) {
                alt (actual.struct) {
                    case (ty_param(?actual_id)) {
                        if (expected_id._0 == actual_id._0 &&
                                expected_id._1 == actual_id._1) {
                            ret ures_ok(expected);
                        }
                    }
                    case (_) {
                        ret ures_err(terr_mismatch, expected, actual);
                    }
                }
            }
        }

        // TODO: remove me once match-exhaustiveness checking works
        fail;
    }

    fn hash_int(&int x) -> uint { ret x as uint; }
    fn eq_int(&int a, &int b) -> bool { ret a == b; }
    auto hasher = hash_int;
    auto eqer = eq_int;
    auto bindings = map.mk_hashmap[int,@ty](hasher, eqer);

    ret unify_step(fcx, bindings, expected, actual);
}

// Requires that the two types unify, and prints an error message if they
// don't. Returns the unified type.
fn demand(&fn_ctxt fcx, &span sp, @ty expected, @ty actual) -> @ty {
    alt (unify(fcx, expected, actual)) {
        case (ures_ok(?ty)) {
            ret ty;
        }

        case (ures_err(?err, ?expected, ?actual)) {
            fcx.ccx.sess.span_err(sp, "mismatched types: expected "
                                  + ty_to_str(expected) + " but found "
                                  + ty_to_str(actual) + " (" +
                                  type_err_to_str(err) + ")");

            // TODO: In the future, try returning "expected", reporting the
            // error, and continue.
            fail;
        }
    }
}

// Returns true if the two types unify and false if they don't.
fn are_compatible(&fn_ctxt fcx, @ty expected, @ty actual) -> bool {
    alt (unify(fcx, expected, actual)) {
        case (ures_ok(_))        { ret true;  }
        case (ures_err(_, _, _)) { ret false; }
    }
}

// Type unification over typed patterns. Note that the pattern that you pass
// to this function must have been passed to check_pat() first.
//
// TODO: enforce this via a predicate.

fn demand_pat(&fn_ctxt fcx, @ty expected, @ast.pat pat) -> @ast.pat {
    auto p_1 = ast.pat_wild(ast.ann_none);  // FIXME: typestate botch

    alt (pat.node) {
        case (ast.pat_wild(?ann)) {
            auto t = demand(fcx, pat.span, expected, ann_to_type(ann));
            p_1 = ast.pat_wild(ast.ann_type(t));
        }
        case (ast.pat_bind(?id, ?did, ?ann)) {
            auto t = demand(fcx, pat.span, expected, ann_to_type(ann));
            fcx.locals.insert(did, t);
            p_1 = ast.pat_bind(id, did, ast.ann_type(t));
        }
        case (ast.pat_tag(?id, ?subpats, ?vdef_opt, ?ann)) {
            auto t = demand(fcx, pat.span, expected, ann_to_type(ann));

            // The type of the tag isn't enough; we also have to get the type
            // of the variant, which is either a tag type in the case of
            // nullary variants or a function type in the case of n-ary
            // variants.
            //
            // TODO: When we have type-parametric tags, this will get a little
            // trickier. Basically, we have to instantiate the variant type we
            // acquire here with the type parameters provided to us by
            // "expected".

            auto vdef = option.get[ast.variant_def](vdef_opt);
            auto variant_ty = fcx.ccx.item_types.get(vdef._1);

            auto subpats_len = _vec.len[@ast.pat](subpats);
            alt (variant_ty.struct) {
                case (ty_tag(_)) {
                    // Nullary tag variant.
                    check (subpats_len == 0u);
                    p_1 = ast.pat_tag(id, subpats, vdef_opt, ast.ann_type(t));
                }
                case (ty_fn(?args, ?tag_ty)) {
                    let vec[@ast.pat] new_subpats = vec();
                    auto i = 0u;
                    for (arg a in args) {
                        auto new_subpat = demand_pat(fcx, a.ty, subpats.(i));
                        new_subpats += vec(new_subpat);
                        i += 1u;
                    }
                    p_1 = ast.pat_tag(id, new_subpats, vdef_opt,
                                      ast.ann_type(tag_ty));
                }
            }
        }
    }

    ret @fold.respan[ast.pat_](pat.span, p_1);
}

// Type unification over typed expressions. Note that the expression that you
// pass to this function must have been passed to check_expr() first.
//
// TODO: enforce this via a predicate.
// TODO: propagate the types downward. This makes the typechecker quadratic,
//       but we can mitigate that if expected == actual == unified.

fn demand_expr(&fn_ctxt fcx, @ty expected, @ast.expr e) -> @ast.expr {
    // FIXME: botch to work around typestate bug in rustboot
    let vec[@ast.expr] v = vec();
    auto e_1 = ast.expr_vec(v, ast.ann_none);

    alt (e.node) {
        case (ast.expr_vec(?es_0, ?ann)) {
            auto t = demand(fcx, e.span, expected, ann_to_type(ann));
            let vec[@ast.expr] es_1 = vec();
            alt (t.struct) {
                case (ty_vec(?subty)) {
                    for (@ast.expr e_0 in es_0) {
                        es_1 += vec(demand_expr(fcx, subty, e_0));
                    }
                }
                case (_) {
                    log "vec expr doesn't have a vec type!";
                    fail;
                }
            }
            e_1 = ast.expr_vec(es_1, ast.ann_type(t));
        }
        case (ast.expr_tup(?es_0, ?ann)) {
            auto t = demand(fcx, e.span, expected, ann_to_type(ann));
            let vec[ast.elt] elts_1 = vec();
            alt (t.struct) {
                case (ty_tup(?subtys)) {
                    auto i = 0u;
                    for (ast.elt elt_0 in es_0) {
                        auto e_1 = demand_expr(fcx, subtys.(i), elt_0.expr);
                        elts_1 += vec(rec(mut=elt_0.mut, expr=e_1));
                        i += 1u;
                    }
                }
                case (_) {
                    log "tup expr doesn't have a tup type!";
                    fail;
                }
            }
            e_1 = ast.expr_tup(elts_1, ast.ann_type(t));
        }
        case (ast.expr_rec(?fields_0, ?ann)) {
            auto t = demand(fcx, e.span, expected, ann_to_type(ann));
            let vec[ast.field] fields_1 = vec();
            alt (t.struct) {
                case (ty_rec(?field_tys)) {
                    auto i = 0u;
                    for (ast.field field_0 in fields_0) {
                        check (_str.eq(field_0.ident, field_tys.(i).ident));
                        auto e_1 = demand_expr(fcx, field_tys.(i).ty,
                                               field_0.expr);
                        fields_1 += vec(rec(mut=field_0.mut,
                                            ident=field_0.ident,
                                            expr=e_1));
                        i += 1u;
                    }
                }
                case (_) {
                    log "rec expr doesn't have a rec type!";
                    fail;
                }
            }
            e_1 = ast.expr_rec(fields_1, ast.ann_type(t));
        }
        case (ast.expr_call(?sube, ?es, ?ann)) {
            auto t = demand(fcx, e.span, expected, ann_to_type(ann));
            e_1 = ast.expr_call(sube, es, ast.ann_type(t));
        }
        case (ast.expr_binary(?bop, ?lhs, ?rhs, ?ann)) {
            auto t = demand(fcx, e.span, expected, ann_to_type(ann));
            e_1 = ast.expr_binary(bop, lhs, rhs, ast.ann_type(t));
        }
        case (ast.expr_unary(?uop, ?sube, ?ann)) {
            auto t = demand(fcx, e.span, expected, ann_to_type(ann));
            e_1 = ast.expr_unary(uop, sube, ast.ann_type(t));
        }
        case (ast.expr_lit(?lit, ?ann)) {
            auto t = demand(fcx, e.span, expected, ann_to_type(ann));
            e_1 = ast.expr_lit(lit, ast.ann_type(t));
        }
        case (ast.expr_cast(?sube, ?ast_ty, ?ann)) {
            auto t = demand(fcx, e.span, expected, ann_to_type(ann));
            e_1 = ast.expr_cast(sube, ast_ty, ast.ann_type(t));
        }
        case (ast.expr_if(?cond, ?then_0, ?else_0, ?ann)) {
            auto t = demand(fcx, e.span, expected, ann_to_type(ann));
            auto then_1 = demand_block(fcx, expected, then_0);
            auto else_1;
            alt (else_0) {
                case (none[ast.block]) { else_1 = none[ast.block]; }
                case (some[ast.block](?b_0)) {
                    auto b_1 = demand_block(fcx, expected, b_0);
                    else_1 = some[ast.block](b_1);
                }
            }
            e_1 = ast.expr_if(cond, then_1, else_1, ast.ann_type(t));
        }
        case (ast.expr_while(?cond, ?bloc, ?ann)) {
            auto t = demand(fcx, e.span, expected, ann_to_type(ann));
            e_1 = ast.expr_while(cond, bloc, ast.ann_type(t));
        }
        case (ast.expr_do_while(?bloc, ?cond, ?ann)) {
            auto t = demand(fcx, e.span, expected, ann_to_type(ann));
            e_1 = ast.expr_do_while(bloc, cond, ast.ann_type(t));
        }
        case (ast.expr_block(?bloc, ?ann)) {
            auto t = demand(fcx, e.span, expected, ann_to_type(ann));
            e_1 = ast.expr_block(bloc, ast.ann_type(t));
        }
        case (ast.expr_assign(?lhs_0, ?rhs_0, ?ann)) {
            auto t = demand(fcx, e.span, expected, ann_to_type(ann));
            auto lhs_1 = demand_expr(fcx, expected, lhs_0);
            auto rhs_1 = demand_expr(fcx, expected, rhs_0);
            e_1 = ast.expr_assign(lhs_1, rhs_1, ast.ann_type(t));
        }
        case (ast.expr_assign_op(?op, ?lhs_0, ?rhs_0, ?ann)) {
            auto t = demand(fcx, e.span, expected, ann_to_type(ann));
            auto lhs_1 = demand_expr(fcx, expected, lhs_0);
            auto rhs_1 = demand_expr(fcx, expected, rhs_0);
            e_1 = ast.expr_assign_op(op, lhs_1, rhs_1, ast.ann_type(t));
        }
        case (ast.expr_field(?lhs, ?rhs, ?ann)) {
            auto t = demand(fcx, e.span, expected, ann_to_type(ann));
            e_1 = ast.expr_field(lhs, rhs, ast.ann_type(t));
        }
        case (ast.expr_index(?base, ?index, ?ann)) {
            auto t = demand(fcx, e.span, expected, ann_to_type(ann));
            e_1 = ast.expr_index(base, index, ast.ann_type(t));
        }
        case (ast.expr_name(?name, ?d, ?ann)) {
            auto t = demand(fcx, e.span, expected, ann_to_type(ann));
            e_1 = ast.expr_name(name, d, ast.ann_type(t));
        }
        case (_) {
            fail;
        }
    }

    ret @fold.respan[ast.expr_](e.span, e_1);
}

// Type unification over typed blocks.
fn demand_block(&fn_ctxt fcx, @ty expected, &ast.block bloc) -> ast.block {
    alt (bloc.node.expr) {
        case (some[@ast.expr](?e_0)) {
            auto e_1 = demand_expr(fcx, expected, e_0);
            auto block_ = rec(stmts=bloc.node.stmts,
                              expr=some[@ast.expr](e_1),
                              index=bloc.node.index);
            ret fold.respan[ast.block_](bloc.span, block_);
        }
        case (none[@ast.expr]) {
            demand(fcx, bloc.span, expected, plain_ty(ty_nil));
            ret bloc;
        }
    }
}

// Writeback: the phase that writes inferred types back into the AST.

fn writeback_local(&fn_ctxt fcx, &span sp, @ast.local local)
        -> @ast.decl {
    if (!fcx.locals.contains_key(local.id)) {
        fcx.ccx.sess.span_err(sp, "unable to determine type of local: "
                              + local.ident);
    }
    auto local_ty = fcx.locals.get(local.id);
    auto local_wb = @rec(ann=ast.ann_type(local_ty) with *local);
    ret @fold.respan[ast.decl_](sp, ast.decl_local(local_wb));
}

fn writeback(&fn_ctxt fcx, &ast.block block) -> ast.block {
    auto fld = fold.new_identity_fold[fn_ctxt]();
    auto f = writeback_local;
    fld = @rec(fold_decl_local = f with *fld);
    ret fold.fold_block[fn_ctxt](fcx, fld, block);
}

// AST fragment checking

fn check_lit(@ast.lit lit) -> @ty {
    auto sty;
    alt (lit.node) {
        case (ast.lit_str(_))   { sty = ty_str;  }
        case (ast.lit_char(_))  { sty = ty_char; }
        case (ast.lit_int(_))   { sty = ty_int;  }
        case (ast.lit_uint(_))  { sty = ty_uint; }
        case (ast.lit_mach_int(?tm, _)) {
            sty = ty_machine(tm);
        }
        case (ast.lit_nil)      { sty = ty_nil;  }
        case (ast.lit_bool(_))  { sty = ty_bool; }
    }

    ret plain_ty(sty);
}

fn check_pat(&fn_ctxt fcx, @ast.pat pat) -> @ast.pat {
    auto new_pat;
    alt (pat.node) {
        case (ast.pat_wild(_)) {
            new_pat = ast.pat_wild(ast.ann_type(next_ty_var(fcx.ccx)));
        }
        case (ast.pat_bind(?id, ?def_id, _)) {
            auto ann = ast.ann_type(next_ty_var(fcx.ccx));
            new_pat = ast.pat_bind(id, def_id, ann);
        }
        case (ast.pat_tag(?id, ?subpats, ?vdef_opt, _)) {
            auto vdef = option.get[ast.variant_def](vdef_opt);
            auto t = fcx.ccx.item_types.get(vdef._1); 
            alt (t.struct) {
                // N-ary variants have function types.
                case (ty_fn(?args, ?tag_ty)) {
                    auto arg_len = _vec.len[arg](args);
                    auto subpats_len = _vec.len[@ast.pat](subpats);
                    if (arg_len != subpats_len) {
                        // TODO: pluralize properly
                        auto err_msg = "tag type " + id + " has " +
                                       _uint.to_str(subpats_len, 10u) +
                                       " fields, but this pattern has " +
                                       _uint.to_str(arg_len, 10u) + " fields";

                        fcx.ccx.sess.span_err(pat.span, err_msg);
                        fail;   // TODO: recover
                    }

                    let vec[@ast.pat] new_subpats = vec();
                    for (@ast.pat subpat in subpats) {
                        new_subpats += vec(check_pat(fcx, subpat));
                    }

                    auto ann = ast.ann_type(tag_ty);
                    new_pat = ast.pat_tag(id, new_subpats, vdef_opt, ann);
                }

                // Nullary variants have tag types.
                case (ty_tag(?tid)) {
                    auto subpats_len = _vec.len[@ast.pat](subpats);
                    if (subpats_len > 0u) {
                        // TODO: pluralize properly
                        auto err_msg = "tag type " + id + " has no fields," +
                                       " but this pattern has " +
                                       _uint.to_str(subpats_len, 10u) +
                                       " fields";

                        fcx.ccx.sess.span_err(pat.span, err_msg);
                        fail;   // TODO: recover
                    }

                    auto ann = ast.ann_type(plain_ty(ty_tag(tid)));
                    new_pat = ast.pat_tag(id, subpats, vdef_opt, ann);
                }
            }
        }
    }

    ret @fold.respan[ast.pat_](pat.span, new_pat);
}

fn check_expr(&fn_ctxt fcx, @ast.expr expr) -> @ast.expr {
    alt (expr.node) {
        case (ast.expr_lit(?lit, _)) {
            auto ty = check_lit(lit);
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_lit(lit, ast.ann_type(ty)));
        }


        case (ast.expr_binary(?binop, ?lhs, ?rhs, _)) {
            auto lhs_0 = check_expr(fcx, lhs);
            auto rhs_0 = check_expr(fcx, rhs);
            auto lhs_t0 = expr_ty(lhs_0);
            auto rhs_t0 = expr_ty(rhs_0);

            // FIXME: Binops have a bit more subtlety than this.
            auto lhs_1 = demand_expr(fcx, rhs_t0, lhs_0);
            auto rhs_1 = demand_expr(fcx, expr_ty(lhs_1), rhs_0);

            auto t = lhs_t0;
            alt (binop) {
                case (ast.eq) { t = plain_ty(ty_bool); }
                case (ast.lt) { t = plain_ty(ty_bool); }
                case (ast.le) { t = plain_ty(ty_bool); }
                case (ast.ne) { t = plain_ty(ty_bool); }
                case (ast.ge) { t = plain_ty(ty_bool); }
                case (ast.gt) { t = plain_ty(ty_bool); }
                case (_) { /* fall through */ }
            }
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_binary(binop, lhs_1, rhs_1,
                                                        ast.ann_type(t)));
        }


        case (ast.expr_unary(?unop, ?oper, _)) {
            auto oper_1 = check_expr(fcx, oper);
            auto oper_t = expr_ty(oper_1);
            alt (unop) {
                case (ast.box) { oper_t = plain_ty(ty_box(oper_t)); }
                case (ast.deref) {
                    alt (oper_t.struct) {
                        case (ty_box(?inner_t)) {
                            oper_t = inner_t;
                        }
                        case (_) {
                            fcx.ccx.sess.span_err
                                (expr.span,
                                 "dereferencing non-box type: "
                                 + ty_to_str(oper_t));
                        }
                    }
                }
                case (_) { /* fall through */ }
            }
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_unary(unop, oper_1,
                                                       ast.ann_type(oper_t)));
        }

        case (ast.expr_name(?name, ?defopt, _)) {
            auto t = plain_ty(ty_nil);
            check (defopt != none[ast.def]);
            alt (option.get[ast.def](defopt)) {
                case (ast.def_arg(?id)) {
                    check (fcx.locals.contains_key(id));
                    t = fcx.locals.get(id);
                }
                case (ast.def_local(?id)) {
                    alt (fcx.locals.find(id)) {
                        case (some[@ty](?t1)) { t = t1; }
                        case (none[@ty])      { t = plain_ty(ty_local(id)); }
                    }
                }
                case (ast.def_fn(?id)) {
                    check (fcx.ccx.item_types.contains_key(id));
                    t = generalize_ty(fcx.ccx, fcx.ccx.item_types.get(id));
                }
                case (ast.def_const(?id)) {
                    check (fcx.ccx.item_types.contains_key(id));
                    t = fcx.ccx.item_types.get(id);
                }
                case (ast.def_variant(_, ?variant_id)) {
                    check (fcx.ccx.item_types.contains_key(variant_id));
                    t = fcx.ccx.item_types.get(variant_id);
                }
                case (ast.def_binding(?id)) {
                    check (fcx.locals.contains_key(id));
                    t = fcx.locals.get(id);
                }
                case (ast.def_obj(?id)) {
                    check (fcx.ccx.item_types.contains_key(id));
                    t = generalize_ty(fcx.ccx, fcx.ccx.item_types.get(id));
                }

                case (_) {
                    // FIXME: handle other names.
                    fcx.ccx.sess.unimpl("definition variant for: "
                                        + name.node.ident);
                    fail;
                }
            }

            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_name(name, defopt,
                                                      ast.ann_type(t)));
        }

        case (ast.expr_assign(?lhs, ?rhs, _)) {
            auto lhs_0 = check_expr(fcx, lhs);
            auto rhs_0 = check_expr(fcx, rhs);
            auto lhs_t0 = expr_ty(lhs_0);
            auto rhs_t0 = expr_ty(rhs_0);

            auto lhs_1 = demand_expr(fcx, rhs_t0, lhs_0);
            auto rhs_1 = demand_expr(fcx, expr_ty(lhs_1), rhs_0);

            auto ann = ast.ann_type(rhs_t0);
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_assign(lhs_1, rhs_1, ann));
        }

        case (ast.expr_assign_op(?op, ?lhs, ?rhs, _)) {
            auto lhs_0 = check_expr(fcx, lhs);
            auto rhs_0 = check_expr(fcx, rhs);
            auto lhs_t0 = expr_ty(lhs_0);
            auto rhs_t0 = expr_ty(rhs_0);

            auto lhs_1 = demand_expr(fcx, rhs_t0, lhs_0);
            auto rhs_1 = demand_expr(fcx, expr_ty(lhs_1), rhs_0);

            auto ann = ast.ann_type(rhs_t0);
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_assign_op(op, lhs_1, rhs_1,
                                                           ann));
        }

        case (ast.expr_if(?cond, ?thn, ?elsopt, _)) {
            auto cond_0 = check_expr(fcx, cond);
            auto cond_1 = demand_expr(fcx, plain_ty(ty_bool), cond_0);

            auto thn_0 = check_block(fcx, thn);
            auto thn_t = block_ty(thn_0);

            auto elsopt_1;
            auto elsopt_t;
            alt (elsopt) {
                case (some[ast.block](?els)) {
                    auto els_0 = check_block(fcx, els);
                    auto els_1 = demand_block(fcx, thn_t, els_0);
                    elsopt_1 = some[ast.block](els_1);
                    elsopt_t = block_ty(els_1);
                }
                case (none[ast.block]) {
                    elsopt_1 = none[ast.block];
                    elsopt_t = plain_ty(ty_nil);
                }
            }

            auto thn_1 = demand_block(fcx, elsopt_t, thn_0);

            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_if(cond_1, thn_1, elsopt_1,
                                                    ast.ann_type(elsopt_t)));
        }

        case (ast.expr_while(?cond, ?body, _)) {
            auto cond_0 = check_expr(fcx, cond);
            auto cond_1 = demand_expr(fcx, plain_ty(ty_bool), cond_0);
            auto body_1 = check_block(fcx, body);

            auto ann = ast.ann_type(plain_ty(ty_nil));
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_while(cond_1, body_1, ann));
        }

        case (ast.expr_do_while(?body, ?cond, _)) {
            auto cond_0 = check_expr(fcx, cond);
            auto cond_1 = demand_expr(fcx, plain_ty(ty_bool), cond_0);
            auto body_1 = check_block(fcx, body);

            auto ann = ast.ann_type(block_ty(body_1));
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_do_while(body_1, cond_1,
                                                          ann));
        }

        case (ast.expr_alt(?expr, ?arms, _)) {
            auto expr_0 = check_expr(fcx, expr);

            // Typecheck the patterns first, so that we get types for all the
            // bindings.
            auto pattern_ty = expr_ty(expr_0);

            let vec[@ast.pat] pats_0 = vec();
            for (ast.arm arm in arms) {
                auto pat_0 = check_pat(fcx, arm.pat);
                pattern_ty = demand(fcx, pat_0.span, pattern_ty,
                                    pat_ty(pat_0));
                pats_0 += vec(pat_0);
            }

            let vec[@ast.pat] pats_1 = vec();
            for (@ast.pat pat_0 in pats_0) {
                pats_1 += vec(demand_pat(fcx, pattern_ty, pat_0));
            }

            // Now typecheck the blocks.
            auto result_ty = next_ty_var(fcx.ccx);

            let vec[ast.block] blocks_0 = vec();
            for (ast.arm arm in arms) {
                auto block_0 = check_block(fcx, arm.block);
                result_ty = demand(fcx, block_0.span, result_ty,
                                   block_ty(block_0));
                blocks_0 += vec(block_0);
            }

            let vec[ast.arm] arms_1 = vec();
            auto i = 0u;
            for (ast.block block_0 in blocks_0) {
                auto block_1 = demand_block(fcx, result_ty, block_0);
                auto pat_1 = pats_1.(i);
                auto arm = arms.(i);
                auto arm_1 = rec(pat=pat_1, block=block_1, index=arm.index);
                arms_1 += vec(arm_1);
                i += 1u;
            }

            auto expr_1 = demand_expr(fcx, pattern_ty, expr_0);

            auto ann = ast.ann_type(result_ty);
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_alt(expr_1, arms_1, ann));
        }

        case (ast.expr_call(?f, ?args, _)) {

            // Check the function.
            auto f_0 = check_expr(fcx, f);

            // Check the arguments and generate the argument signature.
            let vec[@ast.expr] args_0 = vec();
            let vec[arg] arg_tys_0 = vec();
            for (@ast.expr a in args) {
                auto a_0 = check_expr(fcx, a);
                append[@ast.expr](args_0, a_0);

                // FIXME: this breaks aliases. We need a ty_fn_arg.
                append[arg](arg_tys_0, rec(mode=ast.val, ty=expr_ty(a_0)));
            }
            auto rt_0 = next_ty_var(fcx.ccx);
            auto t_0 = plain_ty(ty_fn(arg_tys_0, rt_0));

            // Unify and write back to the function.
            auto f_1 = demand_expr(fcx, t_0, f_0);

            // Take the argument types out of the resulting function type.
            auto t_1 = expr_ty(f_1);
            let vec[arg] arg_tys_1 = vec();     // TODO: typestate botch
            let @ty rt_1 = plain_ty(ty_nil);    // TODO: typestate botch
            alt (t_1.struct) {
                case (ty_fn(?arg_tys, ?rt)) {
                    arg_tys_1 = arg_tys;
                    rt_1 = rt;
                }
                case (_) {
                    fcx.ccx.sess.span_err(f_1.span,
                                          "mismatched types: callee has " +
                                          "non-function type: " +
                                          ty_to_str(t_1));
                }
            }

            // Unify and write back to the arguments.
            auto i = 0u;
            let vec[@ast.expr] args_1 = vec();
            while (i < _vec.len[@ast.expr](args_0)) {
                auto arg_ty_1 = arg_tys_1.(i);
                auto e = demand_expr(fcx, arg_ty_1.ty, args_0.(i));
                append[@ast.expr](args_1, e);

                i += 1u;
            }

            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_call(f_1, args_1,
                                                      ast.ann_type(rt_1)));
        }

        case (ast.expr_cast(?e, ?t, _)) {
            auto e_1 = check_expr(fcx, e);
            auto t_1 = ast_ty_to_ty_crate(fcx.ccx, t);
            // FIXME: there are more forms of cast to support, eventually.
            if (! (type_is_scalar(expr_ty(e_1)) &&
                   type_is_scalar(t_1))) {
                fcx.ccx.sess.span_err(expr.span,
                                      "non-scalar cast: "
                                      + ty_to_str(expr_ty(e_1))
                                      + " as "
                                      +  ty_to_str(t_1));
            }
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_cast(e_1, t,
                                                      ast.ann_type(t_1)));
        }

        case (ast.expr_vec(?args, _)) {
            let vec[@ast.expr] args_1 = vec();

            // FIXME: implement mutable vectors with leading 'mutable' flag
            // marking the elements as mutable.

            let @ty t;
            if (_vec.len[@ast.expr](args) == 0u) {
                t = next_ty_var(fcx.ccx);
            } else {
                auto expr_1 = check_expr(fcx, args.(0));
                t = expr_ty(expr_1);
            }

            for (@ast.expr e in args) {
                auto expr_1 = check_expr(fcx, e);
                auto expr_t = expr_ty(expr_1);
                demand(fcx, expr.span, t, expr_t);
                append[@ast.expr](args_1,expr_1);
            }
            auto ann = ast.ann_type(plain_ty(ty_vec(t)));
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_vec(args_1, ann));
        }

        case (ast.expr_tup(?elts, _)) {
            let vec[ast.elt] elts_1 = vec();
            let vec[@ty] elts_t = vec();

            for (ast.elt e in elts) {
                auto expr_1 = check_expr(fcx, e.expr);
                auto expr_t = expr_ty(expr_1);
                if (e.mut == ast.mut) {
                    expr_t = @rec(mut=ast.mut with *expr_t);
                }
                append[ast.elt](elts_1, rec(expr=expr_1 with e));
                append[@ty](elts_t, expr_t);
            }

            auto ann = ast.ann_type(plain_ty(ty_tup(elts_t)));
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_tup(elts_1, ann));
        }

        case (ast.expr_rec(?fields, _)) {
            let vec[ast.field] fields_1 = vec();
            let vec[field] fields_t = vec();

            for (ast.field f in fields) {
                auto expr_1 = check_expr(fcx, f.expr);
                auto expr_t = expr_ty(expr_1);
                if (f.mut == ast.mut) {
                    expr_t = @rec(mut=ast.mut with *expr_t);
                }
                append[ast.field](fields_1, rec(expr=expr_1 with f));
                append[field](fields_t, rec(ident=f.ident, ty=expr_t));
            }

            auto ann = ast.ann_type(plain_ty(ty_rec(fields_t)));
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_rec(fields_1, ann));
        }

        case (ast.expr_field(?base, ?field, _)) {
            auto base_1 = check_expr(fcx, base);
            auto base_t = expr_ty(base_1);
            alt (base_t.struct) {
                case (ty_tup(?args)) {
                    let uint ix = field_num(fcx.ccx.sess,
                                            expr.span, field);
                    if (ix >= _vec.len[@ty](args)) {
                        fcx.ccx.sess.span_err(expr.span,
                                              "bad index on tuple");
                    }
                    auto ann = ast.ann_type(args.(ix));
                    ret @fold.respan[ast.expr_](expr.span,
                                                ast.expr_field(base_1,
                                                               field,
                                                               ann));
                }

                case (ty_rec(?fields)) {
                    let uint ix = field_idx(fcx.ccx.sess,
                                            expr.span, field, fields);
                    if (ix >= _vec.len[typeck.field](fields)) {
                        fcx.ccx.sess.span_err(expr.span,
                                              "bad index on record");
                    }
                    auto ann = ast.ann_type(fields.(ix).ty);
                    ret @fold.respan[ast.expr_](expr.span,
                                                ast.expr_field(base_1,
                                                               field,
                                                               ann));
                }

                case (ty_obj(?methods)) {
                    let uint ix = method_idx(fcx.ccx.sess,
                                             expr.span, field, methods);
                    if (ix >= _vec.len[typeck.method](methods)) {
                        fcx.ccx.sess.span_err(expr.span,
                                              "bad index on obj");
                    }
                    auto meth = methods.(ix);
                    auto ty = plain_ty(ty_fn(meth.inputs, meth.output));
                    auto ann = ast.ann_type(ty);
                    ret @fold.respan[ast.expr_](expr.span,
                                                ast.expr_field(base_1,
                                                               field,
                                                               ann));
                }

                case (_) {
                    fcx.ccx.sess.unimpl("base type for expr_field "
                                        + "in typeck.check_expr: "
                                        + ty_to_str(base_t));
                }
            }
        }

        case (ast.expr_index(?base, ?idx, _)) {
            auto base_1 = check_expr(fcx, base);
            auto base_t = expr_ty(base_1);

            auto idx_1 = check_expr(fcx, idx);
            auto idx_t = expr_ty(idx_1);

            alt (base_t.struct) {
                case (ty_vec(?t)) {
                    if (! type_is_integral(idx_t)) {
                        fcx.ccx.sess.span_err
                            (idx.span,
                             "non-integral type of vec index: "
                             + ty_to_str(idx_t));
                    }
                    auto ann = ast.ann_type(t);
                    ret @fold.respan[ast.expr_](expr.span,
                                                ast.expr_index(base_1,
                                                               idx_1,
                                                               ann));
                }
                case (ty_str) {
                    if (! type_is_integral(idx_t)) {
                        fcx.ccx.sess.span_err
                            (idx.span,
                             "non-integral type of str index: "
                             + ty_to_str(idx_t));
                    }
                    auto t = ty_machine(common.ty_u8);
                    auto ann = ast.ann_type(plain_ty(t));
                    ret @fold.respan[ast.expr_](expr.span,
                                                ast.expr_index(base_1,
                                                               idx_1,
                                                               ann));
                }
                case (_) {
                    fcx.ccx.sess.span_err
                        (expr.span,
                         "vector-indexing bad type: "
                         + ty_to_str(base_t));
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

fn next_ty_var(@crate_ctxt ccx) -> @ty {
    auto t = plain_ty(ty_var(ccx.next_var_id));
    ccx.next_var_id += 1;
    ret t;
}

fn check_stmt(&fn_ctxt fcx, &@ast.stmt stmt)
        -> @ast.stmt {
    alt (stmt.node) {
        case (ast.stmt_decl(?decl)) {
            alt (decl.node) {
                case (ast.decl_local(?local)) {

                    auto local_ty;
                    alt (local.ty) {
                        case (none[@ast.ty]) {
                            // Auto slot. Assign a ty_var.
                            local_ty = next_ty_var(fcx.ccx);
                        }

                        case (some[@ast.ty](?ast_ty)) {
                            local_ty = ast_ty_to_ty_crate(fcx.ccx, ast_ty);
                        }
                    }
                    fcx.locals.insert(local.id, local_ty);

                    auto rhs_ty = local_ty;
                    auto init = local.init;
                    alt (local.init) {
                        case (some[@ast.expr](?expr)) {
                            auto expr_0 = check_expr(fcx, expr);
                            auto lty = plain_ty(ty_local(local.id));
                            auto expr_1 = demand_expr(fcx, lty, expr_0);
                            init = some[@ast.expr](expr_1);
                        }
                        case (_) { /* fall through */  }
                    }

                    auto local_1 = @rec(init = init with *local);
                    auto decl_1 = @rec(node=ast.decl_local(local_1)
                                       with *decl);
                    ret @fold.respan[ast.stmt_](stmt.span,
                                                ast.stmt_decl(decl_1));
                }

                case (ast.decl_item(_)) {
                    // Ignore for now. We'll return later.
                }
            }

            ret stmt;
        }

        case (ast.stmt_ret(?expr_opt)) {
            alt (expr_opt) {
                case (none[@ast.expr]) {
                    if (!are_compatible(fcx, fcx.ret_ty, plain_ty(ty_nil))) {
                        fcx.ccx.sess.err("ret; in function "
                                         + "returning non-nil");
                    }

                    ret stmt;
                }

                case (some[@ast.expr](?expr)) {
                    auto expr_0 = check_expr(fcx, expr);
                    auto expr_1 = demand_expr(fcx, fcx.ret_ty, expr_0);
                    ret @fold.respan[ast.stmt_](stmt.span,
                                                ast.stmt_ret(some(expr_1)));
                }
            }
        }

        case (ast.stmt_log(?expr)) {
            auto expr_t = check_expr(fcx, expr);
            ret @fold.respan[ast.stmt_](stmt.span, ast.stmt_log(expr_t));
        }

        case (ast.stmt_check_expr(?expr)) {
            auto expr_t = check_expr(fcx, expr);
            demand(fcx, expr.span, plain_ty(ty_bool), expr_ty(expr_t));
            ret @fold.respan[ast.stmt_](stmt.span,
                                        ast.stmt_check_expr(expr_t));
        }

        case (ast.stmt_expr(?expr)) {
            auto expr_t = check_expr(fcx, expr);
            ret @fold.respan[ast.stmt_](stmt.span, ast.stmt_expr(expr_t));
        }
    }

    fail;
}

fn check_block(&fn_ctxt fcx, &ast.block block) -> ast.block {
    let vec[@ast.stmt] stmts = vec();
    for (@ast.stmt s in block.node.stmts) {
        append[@ast.stmt](stmts, check_stmt(fcx, s));
    }

    auto expr = none[@ast.expr];
    alt (block.node.expr) {
        case (none[@ast.expr]) { /* empty */ }
        case (some[@ast.expr](?e)) {
            expr = some[@ast.expr](check_expr(fcx, e));
        }
    }

    ret fold.respan[ast.block_](block.span,
                                rec(stmts=stmts, expr=expr,
                                    index=block.node.index));
}

fn check_const(&@crate_ctxt ccx, &span sp, ast.ident ident, @ast.ty t,
               @ast.expr e, ast.def_id id, ast.ann ann) -> @ast.item {
    // FIXME: this is kinda a kludge; we manufacture a fake "function context"
    // for checking the initializer expression.
    auto rty = ann_to_type(ann);
    let fn_ctxt fcx = rec(ret_ty = rty,
                          locals = @common.new_def_hash[@ty](),
                          ccx = ccx);
    auto e_ = check_expr(fcx, e);
    // FIXME: necessary? Correct sequence?
    demand_expr(fcx, rty, e_);
    auto item = ast.item_const(ident, t, e_, id, ann);
    ret @fold.respan[ast.item_](sp, item);
}

fn check_fn(&@crate_ctxt ccx, &span sp, ast.ident ident, &ast._fn f,
            vec[ast.ty_param] ty_params, ast.def_id id,
            ast.ann ann) -> @ast.item {
    auto local_ty_table = @common.new_def_hash[@ty]();

    // FIXME: duplicate work: the item annotation already has the arg types
    // and return type translated to typeck.ty values. We don't need do to it
    // again here, we can extract them.

    // Store the type of each argument in the table.
    let vec[arg] inputs = vec();
    for (ast.arg arg in f.inputs) {
        auto input_ty = ast_ty_to_ty_crate(ccx, arg.ty);
        inputs += vec(rec(mode=arg.mode, ty=input_ty));
        local_ty_table.insert(arg.id, input_ty);
    }

    auto output_ty = ast_ty_to_ty_crate(ccx, f.output);
    auto fn_sty = ty_fn(inputs, output_ty);
    auto fn_ann = ast.ann_type(plain_ty(fn_sty));

    let fn_ctxt fcx = rec(ret_ty = output_ty,
                          locals = local_ty_table,
                          ccx = ccx);

    // TODO: Make sure the type of the block agrees with the function type.
    auto block_t = check_block(fcx, f.body);
    auto block_wb = writeback(fcx, block_t);

    auto fn_t = rec(effect=f.effect, inputs=f.inputs, output=f.output,
                    body=block_wb);
    auto item = ast.item_fn(ident, fn_t, ty_params, id, fn_ann);
    ret @fold.respan[ast.item_](sp, item);
}

fn check_crate(session.session sess, @ast.crate crate) -> @ast.crate {
    auto result = collect_item_types(crate);

    auto ccx = @rec(sess=sess, item_types=result._1, mutable next_var_id=0);

    auto fld = fold.new_identity_fold[@crate_ctxt]();
    auto f = check_fn;  // FIXME: trans_const_lval bug
    fld = @rec(fold_item_fn = f with *fld);
    ret fold.fold_crate[@crate_ctxt](ccx, fld, result._0);
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
