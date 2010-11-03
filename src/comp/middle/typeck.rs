//
// comp/middle/typeck.rs
//

import front.ast;
import front.ast.ann;
import middle.fold;
import driver.session;
import util.common;
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
type env = rec(session.session sess,
               @ty_table item_types,
               mutable int next_var_id);

type arg = rec(ast.mode mode, @ty ty);

// NB: If you change this, you'll probably want to change the corresponding
// AST structure in front/ast.rs as well.
type ty = rec(sty struct, option.t[str] cname);
tag sty {
    ty_nil;
    ty_bool;
    ty_int;
    ty_uint;
    ty_machine(util.common.ty_mach);
    ty_char;
    ty_str;
    ty_box(@ty);
    ty_vec(@ty);
    ty_tup(vec[tup(bool /* mutability */, @ty)]);
    ty_fn(vec[arg], @ty);                           // TODO: effect
    ty_var(int);
}

tag type_err {
    terr_mismatch;
    terr_tuple_size(uint, uint);
    terr_tuple_mutability;
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
    fn ast_tup_elem_to_str(&tup(bool, @ast.ty) elem) -> str {
        auto s;
        if (elem._0) {
            s = "mutable ";
        } else {
            s = "";
        }

        ret s + ast_ty_to_str(elem._1);
    }

    fn ast_fn_input_to_str(&rec(ast.mode mode, @ast.ty ty) input) -> str {
        auto s;
        if (mode_is_alias(input.mode)) {
            s = "&";
        } else {
            s = "";
        }

        ret s + ast_ty_to_str(input.ty);
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

        case (ast.ty_tup(?elems)) {
            auto f = ast_tup_elem_to_str;
            s = "tup(";
            s += _str.connect(_vec.map[tup(bool,@ast.ty),str](f, elems), ",");
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

fn ty_to_str(@ty typ) -> str {
    fn tup_elem_to_str(&tup(bool, @ty) elem) -> str {
        auto s;
        if (elem._0) {
            s = "mutable ";
        } else {
            s = "";
        }

        ret s + ty_to_str(elem._1);
    }

    fn fn_input_to_str(&rec(ast.mode mode, @ty ty) input) -> str {
        auto s;
        if (mode_is_alias(input.mode)) {
            s = "&";
        } else {
            s = "";
        }

        ret s + ty_to_str(input.ty);
    }

    auto s;
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
            auto f = tup_elem_to_str;
            auto strs = _vec.map[tup(bool,@ty),str](f, elems);
            s = "tup(" + _str.connect(strs, ",") + ")";
        }

        case (ty_fn(?inputs, ?output)) {
            auto f = fn_input_to_str;
            s = "fn(" + _str.connect(_vec.map[arg,str](f, inputs), ", ") + ")";
            if (output.struct != ty_nil) {
                s += " -> " + ty_to_str(output);
            }
        }

        case (ty_var(?v)) {
            // FIXME: wrap around in the case of many variables
            auto ch = ('T' as u8) + (v as u8);
            s = _str.from_bytes(vec(ch));
        }
    }

    ret s;
}

// Parses the programmer's textual representation of a type into our internal
// notion of a type. `getter` is a function that returns the type
// corresponding to a definition ID.
fn ast_ty_to_ty(ty_getter getter, &@ast.ty ast_ty) -> @ty {
    fn ast_arg_to_arg(ty_getter getter, &rec(ast.mode mode, @ast.ty ty) arg)
            -> rec(ast.mode mode, @ty ty) {
        ret rec(mode=arg.mode, ty=ast_ty_to_ty(getter, arg.ty));
    }

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

        case (ast.ty_fn(?inputs, ?output)) {
            auto f = bind ast_arg_to_arg(getter, _);
            auto i = _vec.map[rec(ast.mode mode, @ast.ty ty),arg](f, inputs);
            sty = ty_fn(i, ast_ty_to_ty(getter, output));
        }

        case (ast.ty_path(?path, ?def)) {
            auto def_id;
            alt (option.get[ast.def](def)) {
                case (ast.def_ty(?id)) { def_id = id; }
                case (_) { fail; }
            }

            // TODO: maybe record cname chains so we can do "foo = int" like
            // OCaml?
            sty = getter(def_id).struct;
            cname = some(path_to_str(path));
        }
    }

    ret @rec(struct=sty, cname=cname);
}

// A convenience function to use an environment to resolve names for
// ast_ty_to_ty.
fn ast_ty_to_ty_env(@env e, &@ast.ty ast_ty) -> @ty {
    fn getter(@env e, ast.def_id id) -> @ty {
        ret e.item_types.get(id);
    }
    auto f = bind getter(e, _);
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
    fn trans_ty_item_id_to_ty(@hashmap[ast.def_id,@ast.item] id_to_ty_item,
                              @ty_table item_to_ty,
                              ast.def_id id) -> @ty {
        auto item = id_to_ty_item.get(id);
        ret trans_ty_item_to_ty(id_to_ty_item, item_to_ty, item);
    }

    fn trans_fn_arg_to_ty(@hashmap[ast.def_id,@ast.item] id_to_ty_item,
                          @ty_table item_to_ty,
                          &ast.arg a) -> arg {
        auto f = bind trans_ty_item_id_to_ty(id_to_ty_item, item_to_ty, _);
        ret rec(mode=a.mode, ty=ast_ty_to_ty(f, a.ty));
    }

    fn trans_ty_item_to_ty(@hashmap[ast.def_id,@ast.item] id_to_ty_item,
                           @ty_table item_to_ty,
                           @ast.item it) -> @ty {
        alt (it.node) {
            case (ast.item_fn(?ident, ?fn_info, ?def_id, _)) {
                auto f = bind trans_fn_arg_to_ty(id_to_ty_item, item_to_ty,
                                                 _);
                auto input_tys = _vec.map[ast.arg,arg](f, fn_info.inputs);

                auto g = bind trans_ty_item_id_to_ty(id_to_ty_item,
                                                     item_to_ty, _);
                auto output_ty = ast_ty_to_ty(g, fn_info.output);

                auto t_fn = plain_ty(ty_fn(input_tys, output_ty));
                item_to_ty.insert(def_id, t_fn);
                ret t_fn;
            }

            case (ast.item_ty(?ident, ?referent_ty, ?def_id, _)) {
                if (item_to_ty.contains_key(def_id)) {
                    // Avoid repeating work.
                    ret item_to_ty.get(def_id);
                }

                // Tell ast_ty_to_ty() that we want to perform a recursive
                // call to resolve any named types.
                auto f = bind trans_ty_item_id_to_ty(id_to_ty_item,
                                                     item_to_ty, _);
                auto ty = ast_ty_to_ty(f, referent_ty);
                item_to_ty.insert(def_id, ty);
                ret ty;
            }

            case (ast.item_mod(_, _, _)) { fail; }
        }
    }

    // First pass: collect all type item IDs.
    auto module = crate.node.module;
    auto id_to_ty_item = @common.new_def_hash[@ast.item]();
    for (@ast.item item in module.items) {
        alt (item.node) {
            case (ast.item_ty(_, _, ?def_id, _)) {
                id_to_ty_item.insert(def_id, item);
            }
            case (_) { /* empty */ }
        }
    }

    // Second pass: translate the types of all items.
    auto item_to_ty = @common.new_def_hash[@ty]();
    let vec[@ast.item] items_t = vec();
    for (@ast.item it in module.items) {
        let ast.item_ result;
        alt (it.node) {
            case (ast.item_fn(?ident, ?fn_info, ?def_id, _)) {
                auto t = trans_ty_item_to_ty(id_to_ty_item, item_to_ty, it);
                result = ast.item_fn(ident, fn_info, def_id, ast.ann_type(t));
            }
            case (ast.item_ty(?ident, ?referent_ty, ?def_id, _)) {
                auto t = trans_ty_item_to_ty(id_to_ty_item, item_to_ty, it);
                auto ann = ast.ann_type(t);
                result = ast.item_ty(ident, referent_ty, def_id, ann);
            }
            case (ast.item_mod(_, _, _)) {
                result = it.node;
            }
        }
        items_t += vec(@fold.respan[ast.item_](it.span, result));
    }

    auto module_t = rec(items=items_t, index=module.index);
    ret tup(@fold.respan[ast.crate_](crate.span, rec(module=module_t)),
            item_to_ty);
}

// Type utilities

// FIXME: remove me when == works on these tags.
fn mode_is_alias(ast.mode m) -> bool {
    alt (m) {
        case (ast.val) { ret false; }
        case (ast.alias) { ret true; }
    }
}

fn plain_ty(&sty st) -> @ty {
    ret @rec(struct=st, cname=none[str]);
}

fn ann_to_type(&ast.ann ann) -> @ty {
    alt (ann) {
        case (ast.ann_none) {
            // shouldn't happen, but can until the typechecker is complete
            ret plain_ty(ty_var(0));    // FIXME: broken, broken, broken
        }
        case (ast.ann_type(?ty)) {
            ret ty;
        }
    }
}

fn type_of(@ast.expr expr) -> @ty {
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
        case (ast.expr_block(_, ?ann))        { ret ann_to_type(ann); }
        case (ast.expr_assign(_, _, ?ann))    { ret ann_to_type(ann); }
        case (ast.expr_field(_, _, ?ann))     { ret ann_to_type(ann); }
        case (ast.expr_index(_, _, ?ann))     { ret ann_to_type(ann); }
        case (ast.expr_name(_, _, ?ann))      { ret ann_to_type(ann); }
    }

    fail;
}

// Type unification

fn unify(@ty expected, @ty actual) -> unify_result {
    // Wraps the given type in an appropriate cname.
    //
    // TODO: This doesn't do anything yet. We should carry the cname up from
    // the expected and/or actual types when unification results in a type
    // identical to one or both of the two. The precise algorithm for this is
    // something we'll probably need to develop over time.

    // Simple structural type comparison.
    fn struct_cmp(@ty expected, @ty actual) -> unify_result {
        if (expected == actual) {
            ret ures_ok(expected);
        }

        ret ures_err(terr_mismatch, expected, actual);
    }

    fn unify_step(@ty expected, @ty actual, &hashmap[int,@ty] bindings)
            -> unify_result {
        // TODO: rewrite this using tuple pattern matching when available, to
        // avoid all this rightward drift and spikiness.
        alt (expected.struct) {
            case (ty_nil)        { ret struct_cmp(expected, actual); }
            case (ty_bool)       { ret struct_cmp(expected, actual); }
            case (ty_int)        { ret struct_cmp(expected, actual); }
            case (ty_uint)       { ret struct_cmp(expected, actual); }
            case (ty_machine(_)) { ret struct_cmp(expected, actual); }
            case (ty_char)       { ret struct_cmp(expected, actual); }
            case (ty_str)        { ret struct_cmp(expected, actual); }

            case (ty_box(?expected_sub)) {
                alt (actual.struct) {
                    case (ty_box(?actual_sub)) {
                        auto result = unify_step(expected_sub, actual_sub,
                                                 bindings);
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
                        auto result = unify_step(expected_sub, actual_sub,
                                                 bindings);
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
                        auto expected_len =
                            _vec.len[tup(bool,@ty)](expected_elems);
                        auto actual_len =
                            _vec.len[tup(bool,@ty)](actual_elems);
                        if (expected_len != actual_len) {
                            auto err = terr_tuple_size(expected_len,
                                                       actual_len);
                            ret ures_err(err, expected, actual);
                        }

                        // TODO: implement an iterator that can iterate over
                        // two arrays simultaneously.
                        let vec[tup(bool, @ty)] result_elems = vec();
                        auto i = 0u;
                        while (i < expected_len) {
                            auto expected_elem = expected_elems.(i);
                            auto actual_elem = actual_elems.(i);
                            if (expected_elem._0 != actual_elem._0) {
                                auto err = terr_tuple_mutability;
                                ret ures_err(err, expected, actual);
                            }

                            auto result = unify_step(expected_elem._1,
                                                     actual_elem._1,
                                                     bindings);
                            alt (result) {
                                case (ures_ok(?rty)) {
                                    result_elems += vec(tup(expected_elem._0,
                                                            rty));
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

                            auto result = unify_step(expected_input.ty,
                                                     actual_input.ty,
                                                     bindings);

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
                        auto result = unify_step(expected_output,
                                                 actual_output, bindings);
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
                if (bindings.contains_key(expected_id)) {
                     auto binding = bindings.get(expected_id);
                     ret unify_step(binding, actual, bindings);
                }

                bindings.insert(expected_id, actual);
                ret ures_ok(actual);
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

    ret unify_step(expected, actual, bindings);
}

// Requires that the two types unify, and prints an error message if they
// don't. Returns the unified type.
fn demand(&@env e, &span sp, @ty expected, @ty actual) -> @ty {
    alt (unify(expected, actual)) {
        case (ures_ok(?ty)) {
            ret ty;
        }

        case (ures_err(?err, ?expected, ?actual)) {
            e.sess.err("mismatched types: expected " + ty_to_str(expected) +
                       " but found " + ty_to_str(actual) + " (" +
                       type_err_to_str(err) + ")");

            // TODO: In the future, try returning "expected", reporting the
            // error, and continue.
            fail;
        }
    }
}

// Unifies the supplied type with the type of the local `id`, and stores the
// unified type in the local table. Emits an error if the type is incompatible
// with the previously-stored type for this local.
fn demand_local(&@env e, &span sp, &@ty_table locals, ast.def_id local_id,
                @ty t) {
    auto prev_ty = locals.get(local_id);
    auto unified_ty = demand(e, sp, prev_ty, t);
    locals.insert(local_id, unified_ty);
}

// Returns true if the two types unify and false if they don't.
fn are_compatible(@ty expected, @ty actual) -> bool {
    alt (unify(expected, actual)) {
        case (ures_ok(_))        { ret true;  }
        case (ures_err(_, _, _)) { ret false; }
    }
}

// Writeback: the phase that writes inferred types back into the AST.

fn writeback_local(&@ty_table locals, &span sp, @ast.local local)
        -> @ast.decl {
    auto local_wb = @rec(ann=ast.ann_type(locals.get(local.id)) with *local);
    ret @fold.respan[ast.decl_](sp, ast.decl_local(local_wb));
}

fn writeback(&@env e, &@ty_table locals, &ast.block block) -> ast.block {
    auto fld = fold.new_identity_fold[@ty_table]();
    auto f = writeback_local;   // FIXME: trans_const_lval bug
    fld = @rec(fold_decl_local = f with *fld);
    ret fold.fold_block[@ty_table](locals, fld, block);
}

// AST fragment checking

fn check_lit(&@env e, @ast.lit lit) -> @ty {
    auto sty;
    alt (lit.node) {
        case (ast.lit_str(_))   { sty = ty_str;  }
        case (ast.lit_char(_))  { sty = ty_char; }
        case (ast.lit_int(_))   { sty = ty_int;  }
        case (ast.lit_uint(_))  { sty = ty_uint; }
        case (ast.lit_nil)      { sty = ty_nil;  }
        case (ast.lit_bool(_))  { sty = ty_bool; }
    }

    ret plain_ty(sty);
}

fn check_expr(&@env e, &@ty_table locals, @ast.expr expr) -> @ast.expr {
    alt (expr.node) {
        case (ast.expr_lit(?lit, _)) {
            auto ty = check_lit(e, lit);
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_lit(lit, ast.ann_type(ty)));
        }
        
        case (_) {
            // TODO
            ret expr;
        }
    }
}

fn check_stmt(@env e, @ty_table locals, @ty ret_ty, &@ast.stmt stmt)
        -> @ast.stmt {
    alt (stmt.node) {
        case (ast.stmt_decl(?decl)) {
            alt (decl.node) {
                case (ast.decl_local(?local)) {
                    alt (local.init) {
                        case (none[@ast.expr]) {
                            // empty
                        }

                        case (some[@ast.expr](?expr)) {
                            auto expr_t = check_expr(e, locals, expr);
                            locals.insert(local.id, type_of(expr_t));

                            alt (local.ty) {
                                case (none[@ast.ty]) {
                                    // Nothing to do. We'll figure out the
                                    // type later.
                                }

                                case (some[@ast.ty](?ast_ty)) {
                                    auto ty = ast_ty_to_ty_env(e, ast_ty);
                                    demand_local(e, decl.span, locals,
                                                 local.id, ty);
                                }
                            }
                        }
                    }
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
                    if (!are_compatible(ret_ty, plain_ty(ty_nil))) {
                        e.sess.err("ret; in function returning non-void");
                    }

                    ret stmt;
                }

                case (some[@ast.expr](?expr)) {
                    auto expr_t = check_expr(e, locals, expr);
                    demand(e, expr.span, ret_ty, type_of(expr_t));
                    ret @fold.respan[ast.stmt_](stmt.span,
                                                ast.stmt_ret(some(expr_t)));
                }
            }
        }

        case (ast.stmt_log(?expr)) {
            auto expr_t = check_expr(e, locals, expr);
            ret @fold.respan[ast.stmt_](stmt.span, ast.stmt_log(expr_t));
        }

        case (ast.stmt_check_expr(?expr)) {
            auto expr_t = check_expr(e, locals, expr);
            demand(e, expr.span, plain_ty(ty_bool), type_of(expr_t));
            ret @fold.respan[ast.stmt_](stmt.span,
                                        ast.stmt_check_expr(expr_t));
        }

        case (ast.stmt_expr(?expr)) {
            auto expr_t = check_expr(e, locals, expr);
            if (!are_compatible(type_of(expr_t), plain_ty(ty_nil))) {
                // TODO: real warning function
                log "warning: expression used as statement should have " +
                    "void type";
            }

            ret @fold.respan[ast.stmt_](stmt.span, ast.stmt_expr(expr_t));
        }
    }

    fail;
}

fn check_block(&@env e, &@ty_table locals, @ty ret_ty, &ast.block block)
        -> ast.block {
    auto f = bind check_stmt(e, locals, ret_ty, _);
    auto stmts_t = _vec.map[@ast.stmt,@ast.stmt](f, block.node.stmts);
    ret fold.respan[ast.block_](block.span,
                                rec(stmts=stmts_t, index=block.node.index));
}

fn check_fn(&@env e, &span sp, ast.ident ident, &ast._fn f, ast.def_id id,
            ast.ann ann) -> @ast.item {
    auto local_ty_table = @common.new_def_hash[@ty]();

    // Store the type of each argument in the table.
    let vec[arg] inputs = vec();
    for (ast.arg arg in f.inputs) {
        auto input_ty = ast_ty_to_ty_env(e, arg.ty);
        inputs += vec(rec(mode=arg.mode, ty=input_ty));
        local_ty_table.insert(arg.id, input_ty);
    }

    auto output_ty = ast_ty_to_ty_env(e, f.output);
    auto fn_sty = ty_fn(inputs, output_ty);
    auto fn_ann = ast.ann_type(plain_ty(fn_sty));

    auto block_t = check_block(e, local_ty_table, output_ty, f.body);
    auto block_wb = writeback(e, local_ty_table, block_t);
    auto fn_t = rec(inputs=f.inputs, output=f.output, body=block_wb);
    ret @fold.respan[ast.item_](sp, ast.item_fn(ident, fn_t, id, fn_ann));
}

fn check_crate(session.session sess, @ast.crate crate) -> @ast.crate {
    auto result = collect_item_types(crate);
    auto e = @rec(sess=sess, item_types=result._1, mutable next_var_id=0);

    auto fld = fold.new_identity_fold[@env]();
    auto f = check_fn;  // FIXME: trans_const_lval bug
    fld = @rec(fold_item_fn = f with *fld);
    ret fold.fold_crate[@env](e, fld, result._0);
}

