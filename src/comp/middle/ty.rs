import std._str;
import std._vec;
import std.option;
import std.option.none;
import std.option.some;

import driver.session;
import front.ast;
import front.ast.mutability;
import util.common;
import util.common.span;

// Data types

type arg = rec(ast.mode mode, @t ty);
type field = rec(ast.ident ident, @t ty);
type method = rec(ast.ident ident, vec[arg] inputs, @t output);

// NB: If you change this, you'll probably want to change the corresponding
// AST structure in front/ast.rs as well.
type t = rec(sty struct, mutability mut, option.t[str] cname);
tag sty {
    ty_nil;
    ty_bool;
    ty_int;
    ty_uint;
    ty_machine(util.common.ty_mach);
    ty_char;
    ty_str;
    ty_tag(ast.def_id);
    ty_box(@t);
    ty_vec(@t);
    ty_tup(vec[@t]);
    ty_rec(vec[field]);
    ty_fn(vec[arg], @t);                            // TODO: effect
    ty_obj(vec[method]);
    ty_var(int);                                    // ephemeral type var
    ty_local(ast.def_id);                           // type of a local var
    ty_param(ast.def_id);                           // fn type param
    // TODO: ty_fn_arg(@t), for a possibly-aliased function argument
}

// Stringification

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

fn ty_to_str(&@t typ) -> str {

    fn fn_input_to_str(&rec(ast.mode mode, @t ty) input) -> str {
        auto s;
        if (mode_is_alias(input.mode)) {
            s = "&";
        } else {
            s = "";
        }

        ret s + ty_to_str(input.ty);
    }

    fn fn_to_str(option.t[ast.ident] ident,
                 vec[arg] inputs, @t output) -> str {
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
            auto strs = _vec.map[@t,str](f, elems);
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

// Type folds

type ty_fold = state obj {
    fn fold_simple_ty(@t ty) -> @t;
};

fn fold_ty(ty_fold fld, @t ty) -> @t {
    fn rewrap(@t orig, &sty new) -> @t {
        ret @rec(struct=new, mut=orig.mut, cname=orig.cname);
    }

    alt (ty.struct) {
        case (ty_nil)           { ret fld.fold_simple_ty(ty); }
        case (ty_bool)          { ret fld.fold_simple_ty(ty); }
        case (ty_int)           { ret fld.fold_simple_ty(ty); }
        case (ty_uint)          { ret fld.fold_simple_ty(ty); }
        case (ty_machine(_))    { ret fld.fold_simple_ty(ty); }
        case (ty_char)          { ret fld.fold_simple_ty(ty); }
        case (ty_str)           { ret fld.fold_simple_ty(ty); }
        case (ty_tag(_))        { ret fld.fold_simple_ty(ty); }
        case (ty_box(?subty)) {
            ret rewrap(ty, ty_box(fold_ty(fld, subty)));
        }
        case (ty_vec(?subty)) {
            ret rewrap(ty, ty_vec(fold_ty(fld, subty)));
        }
        case (ty_tup(?subtys)) {
            let vec[@t] new_subtys = vec();
            for (@t subty in subtys) {
                new_subtys += vec(fold_ty(fld, subty));
            }
            ret rewrap(ty, ty_tup(new_subtys));
        }
        case (ty_rec(?fields)) {
            let vec[field] new_fields = vec();
            for (field fl in fields) {
                auto new_ty = fold_ty(fld, fl.ty);
                new_fields += vec(rec(ident=fl.ident, ty=new_ty));
            }
            ret rewrap(ty, ty_rec(new_fields));
        }
        case (ty_fn(?args, ?ret_ty)) {
            let vec[arg] new_args = vec();
            for (arg a in args) {
                auto new_ty = fold_ty(fld, a.ty);
                new_args += vec(rec(mode=a.mode, ty=new_ty));
            }
            ret rewrap(ty, ty_fn(new_args, fold_ty(fld, ret_ty)));
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
            ret rewrap(ty, ty_obj(new_methods));
        }
        case (ty_var(_))        { ret fld.fold_simple_ty(ty); }
        case (ty_local(_))      { ret fld.fold_simple_ty(ty); }
        case (ty_param(_))      { ret fld.fold_simple_ty(ty); }
    }

    ret ty;
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

fn type_is_nil(@t ty) -> bool {
    alt (ty.struct) {
        case (ty_nil) { ret true; }
        case (_) { ret false; }
    }
    fail;
}

fn type_is_structural(@t ty) -> bool {
    alt (ty.struct) {
        case (ty_tup(_)) { ret true; }
        case (ty_rec(_)) { ret true; }
        case (ty_tag(_)) { ret true; }
        case (ty_fn(_,_)) { ret true; }
        case (ty_obj(_)) { ret true; }
        case (_) { ret false; }
    }
    fail;
}

fn type_is_boxed(@t ty) -> bool {
    alt (ty.struct) {
        case (ty_str) { ret true; }
        case (ty_vec(_)) { ret true; }
        case (ty_box(_)) { ret true; }
        case (_) { ret false; }
    }
    fail;
}

fn type_is_scalar(@t ty) -> bool {
    alt (ty.struct) {
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


fn type_is_integral(@t ty) -> bool {
    alt (ty.struct) {
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

fn type_is_fp(@t ty) -> bool {
    alt (ty.struct) {
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

fn type_is_signed(@t ty) -> bool {
    alt (ty.struct) {
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

fn type_param(@t ty) -> option.t[ast.def_id] {
    alt (ty.struct) {
        case (ty_param(?id)) { ret some[ast.def_id](id); }
        case (_)             { /* fall through */        }
    }
    ret none[ast.def_id];
}

fn plain_ty(&sty st) -> @t {
    ret @rec(struct=st, mut=ast.imm, cname=none[str]);
}

fn hash_ty(&@t ty) -> uint {
    ret _str.hash(ty_to_str(ty));
}

fn eq_ty(&@t a, &@t b) -> bool {
    // FIXME: this is gross, but I think it's safe, and I don't think writing
    // a giant function to handle all the cases is necessary when structural
    // equality will someday save the day.
    ret _str.eq(ty_to_str(a), ty_to_str(b));
}

fn ann_to_type(&ast.ann ann) -> @t {
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

fn count_ty_params(@t ty) -> uint {
    state obj ty_param_counter(@mutable vec[ast.def_id] param_ids) {
        fn fold_simple_ty(@t ty) -> @t {
            alt (ty.struct) {
                case (ty_param(?param_id)) {
                    for (ast.def_id other_param_id in *param_ids) {
                        if (param_id._0 == other_param_id._0 &&
                                param_id._1 == other_param_id._1) {
                            ret ty;
                        }
                    }
                    *param_ids += vec(param_id);
                }
                case (_) { /* fall through */ }
            }
            ret ty;
        }
    }

    let vec[ast.def_id] param_ids_inner = vec();
    let @mutable vec[ast.def_id] param_ids = @mutable param_ids_inner;
    fold_ty(ty_param_counter(param_ids), ty);
    ret _vec.len[ast.def_id](*param_ids);
}

// Type accessors for AST nodes

fn stmt_ty(@ast.stmt s) -> @t {
    alt (s.node) {
        case (ast.stmt_expr(?e)) {
            ret expr_ty(e);
        }
        case (_) {
            ret plain_ty(ty_nil);
        }
    }
}

fn block_ty(&ast.block b) -> @t {
    alt (b.node.expr) {
        case (some[@ast.expr](?e)) { ret expr_ty(e); }
        case (none[@ast.expr])     { ret plain_ty(ty_nil); }
    }
}

fn pat_ty(@ast.pat pat) -> @t {
    alt (pat.node) {
        case (ast.pat_wild(?ann))           { ret ann_to_type(ann); }
        case (ast.pat_bind(_, _, ?ann))     { ret ann_to_type(ann); }
        case (ast.pat_tag(_, _, _, ?ann))   { ret ann_to_type(ann); }
    }
    fail;   // not reached
}

fn expr_ty(@ast.expr expr) -> @t {
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

