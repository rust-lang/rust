import std._str;
import std._uint;
import std._vec;
import std.UFind;
import std.map;
import std.map.hashmap;
import std.option;
import std.option.none;
import std.option.some;

import driver.session;
import front.ast;
import front.ast.mutability;
import front.creader;
import middle.metadata;
import util.common;
import util.common.new_def_hash;
import util.common.span;
import util.typestate_ann.ts_ann;

// Data types

type arg = rec(ast.mode mode, @t ty);
type field = rec(ast.ident ident, mt mt);
type method = rec(ast.proto proto,
                  ast.ident ident,
                  vec[arg] inputs,
                  @t output);

type mt = rec(@t ty, ast.mutability mut);

// Convert from method type to function type.  Pretty easy; we just drop
// 'ident'.
fn method_ty_to_fn_ty(method m) -> @ty.t {
    ret plain_ty(ty_fn(m.proto, m.inputs, m.output));
}

// NB: If you change this, you'll probably want to change the corresponding
// AST structure in front/ast.rs as well.
type t = rec(sty struct, option.t[str] cname);
tag sty {
    ty_nil;
    ty_bool;
    ty_int;
    ty_float;
    ty_uint;
    ty_machine(util.common.ty_mach);
    ty_char;
    ty_str;
    ty_tag(ast.def_id, vec[@t]);
    ty_box(mt);
    ty_vec(mt);
    ty_port(@t);
    ty_chan(@t);
    ty_task;
    ty_tup(vec[mt]);
    ty_rec(vec[field]);
    ty_fn(ast.proto, vec[arg], @t);
    ty_native_fn(ast.native_abi, vec[arg], @t);
    ty_obj(vec[method]);
    ty_var(int);                                    // ephemeral type var
    ty_local(ast.def_id);                           // type of a local var
    ty_param(uint);                                 // fn/tag type param
    ty_bound_param(uint);                           // bound param, only paths
    ty_type;
    ty_native;
    // TODO: ty_fn_arg(@t), for a possibly-aliased function argument
}

// Data structures used in type unification

type unify_handler = obj {
    fn resolve_local(ast.def_id id) -> option.t[@t];
    fn record_local(ast.def_id id, @t ty);  // TODO: -> unify_result
    fn record_param(uint index, @t binding) -> unify_result;
};

tag type_err {
    terr_mismatch;
    terr_box_mutability;
    terr_vec_mutability;
    terr_tuple_size(uint, uint);
    terr_tuple_mutability;
    terr_record_size(uint, uint);
    terr_record_mutability;
    terr_record_fields(ast.ident,ast.ident);
    terr_meth_count;
    terr_obj_meths(ast.ident,ast.ident);
    terr_arg_count;
}

tag unify_result {
    ures_ok(@ty.t);
    ures_err(type_err, @ty.t, @ty.t);
}


type ty_param_count_and_ty = tup(uint, @t);
type type_cache = hashmap[ast.def_id,ty_param_count_and_ty];


// Stringification

fn path_to_str(&ast.path pth) -> str {
    auto result = _str.connect(pth.node.idents,  ".");
    if (_vec.len[@ast.ty](pth.node.types) > 0u) {
        auto f = pretty.pprust.ty_to_str;
        result += "[";
        result += _str.connect(_vec.map[@ast.ty,str](f, pth.node.types), ",");
        result += "]";
    }
    ret result;
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

    fn fn_to_str(ast.proto proto,
                 option.t[ast.ident] ident,
                 vec[arg] inputs, @t output) -> str {
            auto f = fn_input_to_str;

            auto s;
            alt (proto) {
                case (ast.proto_iter) {
                    s = "iter";
                }
                case (ast.proto_fn) {
                    s = "fn";
                }
            }

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
        ret fn_to_str(m.proto, some[ast.ident](m.ident),
                      m.inputs, m.output) + ";";
    }

    fn field_to_str(&field f) -> str {
        ret mt_to_str(f.mt) + " " + f.ident;
    }

    fn mt_to_str(&mt m) -> str {
        auto mstr;
        alt (m.mut) {
            case (ast.mut)       { mstr = "mutable "; }
            case (ast.imm)       { mstr = "";         }
            case (ast.maybe_mut) { mstr = "mutable? "; }
        }

        ret mstr + ty_to_str(m.ty);
    }

    auto s = "";
    alt (typ.struct) {
        case (ty_native)       { s += "native";                     }
        case (ty_nil)          { s += "()";                         }
        case (ty_bool)         { s += "bool";                       }
        case (ty_int)          { s += "int";                        }
        case (ty_float)        { s += "float";                      }
        case (ty_uint)         { s += "uint";                       }
        case (ty_machine(?tm)) { s += common.ty_mach_to_str(tm);    }
        case (ty_char)         { s += "char";                       }
        case (ty_str)          { s += "str";                        }
        case (ty_box(?tm))     { s += "@" + mt_to_str(tm);          }
        case (ty_vec(?tm))     { s += "vec[" + mt_to_str(tm) + "]"; }
        case (ty_port(?t))     { s += "port[" + ty_to_str(t) + "]"; }
        case (ty_chan(?t))     { s += "chan[" + ty_to_str(t) + "]"; }
        case (ty_type)         { s += "type";                       }

        case (ty_tup(?elems)) {
            auto f = mt_to_str;
            auto strs = _vec.map[mt,str](f, elems);
            s += "tup(" + _str.connect(strs, ",") + ")";
        }

        case (ty_rec(?elems)) {
            auto f = field_to_str;
            auto strs = _vec.map[field,str](f, elems);
            s += "rec(" + _str.connect(strs, ",") + ")";
        }

        case (ty_tag(?id, ?tps)) {
            // The user should never see this if the cname is set properly!
            s += "<tag#" + util.common.istr(id._0) + ":" +
                util.common.istr(id._1) + ">";
            if (_vec.len[@t](tps) > 0u) {
                auto f = ty_to_str;
                auto strs = _vec.map[@t,str](f, tps);
                s += "[" + _str.connect(strs, ",") + "]";
            }
        }

        case (ty_fn(?proto, ?inputs, ?output)) {
            s += fn_to_str(proto, none[ast.ident], inputs, output);
        }

        case (ty_native_fn(_, ?inputs, ?output)) {
            s += fn_to_str(ast.proto_fn, none[ast.ident], inputs, output);
        }

        case (ty_obj(?meths)) {
            alt (typ.cname) {
                case (some[str](?cs)) {
                    s += cs;
                }
                case (_) {
                    auto f = method_to_str;
                    auto m = _vec.map[method,str](f, meths);
                    s += "obj {\n\t" + _str.connect(m, "\n\t") + "\n}";
                }
            }
        }

        case (ty_var(?v)) {
            s += "<T" + util.common.istr(v) + ">";
        }

        case (ty_local(?id)) {
            s += "<L" + util.common.istr(id._0) + ":" +
                util.common.istr(id._1) + ">";
        }

        case (ty_param(?id)) {
            s += "'" + _str.unsafe_from_bytes(vec(('a' as u8) + (id as u8)));
        }

        case (ty_bound_param(?id)) {
            s += "''" + _str.unsafe_from_bytes(vec(('a' as u8) + (id as u8)));
        }
    }

    ret s;
}

// Type folds

type ty_walk = fn(@t);

fn walk_ty(ty_walk walker, @t ty) {
    alt (ty.struct) {
        case (ty_nil)           { /* no-op */ }
        case (ty_bool)          { /* no-op */ }
        case (ty_int)           { /* no-op */ }
        case (ty_uint)          { /* no-op */ }
        case (ty_float)         { /* no-op */ }
        case (ty_machine(_))    { /* no-op */ }
        case (ty_char)          { /* no-op */ }
        case (ty_str)           { /* no-op */ }
        case (ty_type)          { /* no-op */ }
        case (ty_native)        { /* no-op */ }
        case (ty_box(?tm))      { walk_ty(walker, tm.ty); }
        case (ty_vec(?tm))      { walk_ty(walker, tm.ty); }
        case (ty_port(?subty))  { walk_ty(walker, subty); }
        case (ty_chan(?subty))  { walk_ty(walker, subty); }
        case (ty_tag(?tid, ?subtys)) {
            for (@t subty in subtys) {
                walk_ty(walker, subty);
            }
        }
        case (ty_tup(?mts)) {
            for (mt tm in mts) {
                walk_ty(walker, tm.ty);
            }
        }
        case (ty_rec(?fields)) {
            for (field fl in fields) {
                walk_ty(walker, fl.mt.ty);
            }
        }
        case (ty_fn(?proto, ?args, ?ret_ty)) {
            for (arg a in args) {
                walk_ty(walker, a.ty);
            }
            walk_ty(walker, ret_ty);
        }
        case (ty_native_fn(?abi, ?args, ?ret_ty)) {
            for (arg a in args) {
                walk_ty(walker, a.ty);
            }
            walk_ty(walker, ret_ty);
        }
        case (ty_obj(?methods)) {
            let vec[method] new_methods = vec();
            for (method m in methods) {
                for (arg a in m.inputs) {
                    walk_ty(walker, a.ty);
                }
                walk_ty(walker, m.output);
            }
        }
        case (ty_var(_))         { /* no-op */ }
        case (ty_local(_))       { /* no-op */ }
        case (ty_param(_))       { /* no-op */ }
        case (ty_bound_param(_)) { /* no-op */ }
    }

    walker(ty);
}

type ty_fold = fn(@t) -> @t;

fn fold_ty(ty_fold fld, @t ty_0) -> @t {
    fn rewrap(@t orig, &sty new) -> @t {
        ret @rec(struct=new, cname=orig.cname);
    }

    auto ty = ty_0;
    alt (ty.struct) {
        case (ty_nil)           { /* no-op */ }
        case (ty_bool)          { /* no-op */ }
        case (ty_int)           { /* no-op */ }
        case (ty_uint)          { /* no-op */ }
        case (ty_float)         { /* no-op */ }
        case (ty_machine(_))    { /* no-op */ }
        case (ty_char)          { /* no-op */ }
        case (ty_str)           { /* no-op */ }
        case (ty_type)          { /* no-op */ }
        case (ty_native)        { /* no-op */ }
        case (ty_box(?tm)) {
            ty = rewrap(ty, ty_box(rec(ty=fold_ty(fld, tm.ty), mut=tm.mut)));
        }
        case (ty_vec(?tm)) {
            ty = rewrap(ty, ty_vec(rec(ty=fold_ty(fld, tm.ty), mut=tm.mut)));
        }
        case (ty_port(?subty)) {
            ty = rewrap(ty, ty_port(fold_ty(fld, subty)));
        }
        case (ty_chan(?subty)) {
            ty = rewrap(ty, ty_chan(fold_ty(fld, subty)));
        }
        case (ty_tag(?tid, ?subtys)) {
            let vec[@t] new_subtys = vec();
            for (@t subty in subtys) {
                new_subtys += vec(fold_ty(fld, subty));
            }
            ty = rewrap(ty, ty_tag(tid, new_subtys));
        }
        case (ty_tup(?mts)) {
            let vec[mt] new_mts = vec();
            for (mt tm in mts) {
                auto new_subty = fold_ty(fld, tm.ty);
                new_mts += vec(rec(ty=new_subty, mut=tm.mut));
            }
            ty = rewrap(ty, ty_tup(new_mts));
        }
        case (ty_rec(?fields)) {
            let vec[field] new_fields = vec();
            for (field fl in fields) {
                auto new_ty = fold_ty(fld, fl.mt.ty);
                auto new_mt = rec(ty=new_ty, mut=fl.mt.mut);
                new_fields += vec(rec(ident=fl.ident, mt=new_mt));
            }
            ty = rewrap(ty, ty_rec(new_fields));
        }
        case (ty_fn(?proto, ?args, ?ret_ty)) {
            let vec[arg] new_args = vec();
            for (arg a in args) {
                auto new_ty = fold_ty(fld, a.ty);
                new_args += vec(rec(mode=a.mode, ty=new_ty));
            }
            ty = rewrap(ty, ty_fn(proto, new_args, fold_ty(fld, ret_ty)));
        }
        case (ty_native_fn(?abi, ?args, ?ret_ty)) {
            let vec[arg] new_args = vec();
            for (arg a in args) {
                auto new_ty = fold_ty(fld, a.ty);
                new_args += vec(rec(mode=a.mode, ty=new_ty));
            }
            ty = rewrap(ty, ty_native_fn(abi, new_args,
                                         fold_ty(fld, ret_ty)));
        }
        case (ty_obj(?methods)) {
            let vec[method] new_methods = vec();
            for (method m in methods) {
                let vec[arg] new_args = vec();
                for (arg a in m.inputs) {
                    new_args += vec(rec(mode=a.mode, ty=fold_ty(fld, a.ty)));
                }
                new_methods += vec(rec(proto=m.proto, ident=m.ident,
                                       inputs=new_args,
                                       output=fold_ty(fld, m.output)));
            }
            ty = rewrap(ty, ty_obj(new_methods));
        }
        case (ty_var(_))         { /* no-op */ }
        case (ty_local(_))       { /* no-op */ }
        case (ty_param(_))       { /* no-op */ }
        case (ty_bound_param(_)) { /* no-op */ }
    }

    ret fld(ty);
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

fn type_is_bool(@t ty) -> bool {
    alt (ty.struct) {
        case (ty_bool) { ret true; }
        case (_) { ret false; }
    }
}


fn type_is_structural(@t ty) -> bool {
    alt (ty.struct) {
        case (ty_tup(_))    { ret true; }
        case (ty_rec(_))    { ret true; }
        case (ty_tag(_,_))  { ret true; }
        case (ty_fn(_,_,_)) { ret true; }
        case (ty_obj(_))    { ret true; }
        case (_)            { ret false; }
    }
    fail;
}

fn type_is_sequence(@t ty) -> bool {
    alt (ty.struct) {
        case (ty_str)    { ret true; }
        case (ty_vec(_))    { ret true; }
        case (_)            { ret false; }
    }
    fail;
}

fn sequence_element_type(@t ty) -> @t {
    alt (ty.struct) {
        case (ty_str)      { ret plain_ty(ty_machine(common.ty_u8)); }
        case (ty_vec(?mt)) { ret mt.ty; }
    }
    fail;
}


fn type_is_tup_like(@t ty) -> bool {
    alt (ty.struct) {
        case (ty_box(_))    { ret true; }
        case (ty_tup(_))    { ret true; }
        case (ty_rec(_))    { ret true; }
        case (ty_tag(_,_))  { ret true; }
        case (_)            { ret false; }
    }
    fail;
}

fn get_element_type(@t ty, uint i) -> @t {
    check (type_is_tup_like(ty));
    alt (ty.struct) {
        case (ty_tup(?mts)) {
            ret mts.(i).ty;
        }
        case (ty_rec(?flds)) {
            ret flds.(i).mt.ty;
        }
    }
    fail;
}

fn type_is_box(@t ty) -> bool {
    alt (ty.struct) {
        case (ty_box(_)) { ret true; }
        case (_) { ret false; }
    }
    fail;
}

fn type_is_boxed(@t ty) -> bool {
    alt (ty.struct) {
        case (ty_str) { ret true; }
        case (ty_vec(_)) { ret true; }
        case (ty_box(_)) { ret true; }
        case (ty_port(_)) { ret true; }
        case (ty_chan(_)) { ret true; }
        case (_) { ret false; }
    }
    fail;
}

fn type_is_scalar(@t ty) -> bool {
    alt (ty.struct) {
        case (ty_nil) { ret true; }
        case (ty_bool) { ret true; }
        case (ty_int) { ret true; }
        case (ty_float) { ret true; }
        case (ty_uint) { ret true; }
        case (ty_machine(_)) { ret true; }
        case (ty_char) { ret true; }
        case (ty_type) { ret true; }
        case (ty_native) { ret true; }
        case (_) { ret false; }
    }
    fail;
}

// FIXME: should we just return true for native types in
// type_is_scalar?
fn type_is_native(@t ty) -> bool {
    alt (ty.struct) {
        case (ty_native) { ret true; }
        case (_) { ret false; }
    }
    fail;
}

fn type_has_dynamic_size(@t ty) -> bool {
    alt (ty.struct) {
        case (ty_tup(?mts)) {
            auto i = 0u;
            while (i < _vec.len[mt](mts)) {
                if (type_has_dynamic_size(mts.(i).ty)) { ret true; }
                i += 1u;
            }
        }
        case (ty_rec(?fields)) {
            auto i = 0u;
            while (i < _vec.len[field](fields)) {
                if (type_has_dynamic_size(fields.(i).mt.ty)) { ret true; }
                i += 1u;
            }
        }
        case (ty_tag(_, ?subtys)) {
            auto i = 0u;
            while (i < _vec.len[@t](subtys)) {
                if (type_has_dynamic_size(subtys.(i))) { ret true; }
                i += 1u;
            }
        }
        case (ty_param(_)) { ret true; }
        case (_) { /* fall through */ }
    }
    ret false;
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
        case (ty_float) {
            ret true;
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

fn type_param(@t ty) -> option.t[uint] {
    alt (ty.struct) {
        case (ty_param(?id)) { ret some[uint](id); }
        case (_)             { /* fall through */  }
    }
    ret none[uint];
}

fn plain_ty(&sty st) -> @t {
    ret @rec(struct=st, cname=none[str]);
}

fn plain_box_ty(@t subty, ast.mutability mut) -> @t {
    ret plain_ty(ty_box(rec(ty=subty, mut=mut)));
}

fn plain_tup_ty(vec[@t] elem_tys) -> @t {
    let vec[ty.mt] mts = vec();
    for (@ty.t typ in elem_tys) {
        mts += vec(rec(ty=typ, mut=ast.imm));
    }
    ret plain_ty(ty_tup(mts));
}

fn def_to_str(ast.def_id did) -> str {
    ret #fmt("%d:%d", did._0, did._1);
}

fn simple_ty_code(&@t ty) -> uint {
    alt (ty.struct) {
        case (ty_nil) { ret 0u; }
        case (ty_bool) { ret 1u; }
        case (ty_int) { ret 2u; }
        case (ty_float) { ret 3u; }
        case (ty_uint) { ret 4u; }
        case (ty_machine(?tm)) {
            alt (tm) {
                case (common.ty_i8) { ret 5u; }
                case (common.ty_i16) { ret 6u; }
                case (common.ty_i32) { ret 7u; }
                case (common.ty_i64) { ret 8u; }

                case (common.ty_u8) { ret 9u; }
                case (common.ty_u16) { ret 10u; }
                case (common.ty_u32) { ret 11u; }
                case (common.ty_u64) { ret 12u; }

                case (common.ty_f32) { ret 13u; }
                case (common.ty_f64) { ret 14u; }
            }
        }
        case (ty_char) { ret 15u; }
        case (ty_str) { ret 16u; }
        case (ty_task) { ret 17u; }
        case (ty_type) { ret 18u; }
        case (ty_native) { ret 19u; }
        case (_) {
        }
    }
    ret 0xffffu;
}

fn hash_ty(&@t ty) -> uint {
    auto s = simple_ty_code(ty);
    if (s != 0xffffu) {
        ret s;
    }
    auto f = def_to_str;
    ret _str.hash(metadata.ty_str(ty, f));
}

fn eq_ty(&@t a, &@t b) -> bool {

    auto sa = simple_ty_code(a);
    if (sa != 0xffffu) {
        auto sb = simple_ty_code(b);
        ret sa == sb;
    }

    // FIXME: this is gross, but I think it's safe, and I don't think writing
    // a giant function to handle all the cases is necessary when structural
    // equality will someday save the day.
    auto f = def_to_str;
    ret _str.eq(metadata.ty_str(a, f), metadata.ty_str(b, f));
}

fn ann_to_type(&ast.ann ann) -> @t {
    alt (ann) {
        case (ast.ann_none) {
            log_err "ann_to_type() called on node with no type";
            fail;
        }
        case (ast.ann_type(?ty, _, _)) {
            ret ty;
        }
    }
}

fn ann_to_type_params(&ast.ann ann) -> vec[@t] {
    alt (ann) {
        case (ast.ann_none) {
            log_err "ann_to_type_params() called on node with no type params";
            fail;
        }
        case (ast.ann_type(_, ?tps, _)) {
            alt (tps) {
                case (none[vec[@ty.t]]) {
                    let vec[@t] result = vec();
                    ret result;
                }
                case (some[vec[@ty.t]](?tps)) { ret tps; }
            }
        }
    }
}

// Returns the type of an annotation, with type parameter substitutions
// performed if applicable.
fn ann_to_monotype(ast.ann a) -> @ty.t {
    // TODO: Refactor to use recursive pattern matching when we're more
    // confident that it works.
    alt (a) {
        case (ast.ann_none) {
            log_err "ann_to_monotype() called on expression with no type!";
            fail;
        }
        case (ast.ann_type(?typ, ?tps_opt, _)) {
            alt (tps_opt) {
                case (none[vec[@ty.t]]) { ret typ; }
                case (some[vec[@ty.t]](?tps)) {
                    ret substitute_type_params(tps, typ);
                }
            }
        }
    }
}

// Turns a type into an ann_type, using defaults for other fields.
fn triv_ann(@ty.t typ) -> ast.ann {
    ret ast.ann_type(typ, none[vec[@ty.t]], none[@ts_ann]);
}

// Returns the number of distinct type parameters in the given type.
fn count_ty_params(@t ty) -> uint {
    fn counter(@mutable vec[uint] param_indices, @t ty) {
        alt (ty.struct) {
            case (ty_param(?param_idx)) {
                auto seen = false;
                for (uint other_param_idx in *param_indices) {
                    if (param_idx == other_param_idx) {
                        seen = true;
                    }
                }
                if (!seen) {
                    *param_indices += vec(param_idx);
                }
            }
            case (_) { /* fall through */ }
        }
    }

    let vec[uint] v = vec();    // FIXME: typechecker botch
    let @mutable vec[uint] param_indices = @mutable v;
    auto f = bind counter(param_indices, _);
    walk_ty(f, ty);
    ret _vec.len[uint](*param_indices);
}

fn type_contains_vars(@t typ) -> bool {
    fn checker(@mutable bool flag, @t typ) {
        alt (typ.struct) {
            case (ty_var(_)) { *flag = true; }
            case (_) { /* fall through */ }
        }
    }

    let @mutable bool flag = @mutable false;
    auto f = bind checker(flag, _);
    walk_ty(f, typ);
    ret *flag;
}

// Type accessors for substructures of types

fn ty_fn_args(@t fty) -> vec[arg] {
    alt (fty.struct) {
        case (ty.ty_fn(_, ?a, _)) { ret a; }
        case (ty.ty_native_fn(_, ?a, _)) { ret a; }
    }
    fail;
}

fn ty_fn_proto(@t fty) -> ast.proto {
    alt (fty.struct) {
        case (ty.ty_fn(?p, _, _)) { ret p; }
    }
    fail;
}

fn ty_fn_abi(@t fty) -> ast.native_abi {
    alt (fty.struct) {
        case (ty.ty_native_fn(?a, _, _)) { ret a; }
    }
    fail;
}

fn ty_fn_ret(@t fty) -> @t {
    alt (fty.struct) {
        case (ty.ty_fn(_, _, ?r)) { ret r; }
        case (ty.ty_native_fn(_, _, ?r)) { ret r; }
    }
    fail;
}

fn is_fn_ty(@t fty) -> bool {
    alt (fty.struct) {
        case (ty.ty_fn(_, _, _)) { ret true; }
        case (ty.ty_native_fn(_, _, _)) { ret true; }
        case (_) { ret false; }
    }
    ret false;
}


// Type accessors for AST nodes

// Given an item, returns the associated type as well as the number of type
// parameters it has.
fn native_item_ty(@ast.native_item it) -> ty_param_count_and_ty {
    auto ty_param_count;
    auto result_ty;
    alt (it.node) {
        case (ast.native_item_fn(_, _, _, ?tps, _, ?ann)) {
            ty_param_count = _vec.len[ast.ty_param](tps);
            result_ty = ann_to_type(ann);
        }
    }
    ret tup(ty_param_count, result_ty);
}

fn item_ty(@ast.item it) -> ty_param_count_and_ty {
    auto ty_param_count;
    auto result_ty;
    alt (it.node) {
        case (ast.item_const(_, _, _, _, ?ann)) {
            ty_param_count = 0u;
            result_ty = ann_to_type(ann);
        }
        case (ast.item_fn(_, _, ?tps, _, ?ann)) {
            ty_param_count = _vec.len[ast.ty_param](tps);
            result_ty = ann_to_type(ann);
        }
        case (ast.item_mod(_, _, _)) {
            fail;   // modules are typeless
        }
        case (ast.item_ty(_, _, ?tps, _, ?ann)) {
            ty_param_count = _vec.len[ast.ty_param](tps);
            result_ty = ann_to_type(ann);
        }
        case (ast.item_tag(_, _, ?tps, ?did, ?ann)) {
            ty_param_count = _vec.len[ast.ty_param](tps);
            result_ty = ann_to_type(ann);
        }
        case (ast.item_obj(_, _, ?tps, _, ?ann)) {
            ty_param_count = _vec.len[ast.ty_param](tps);
            result_ty = ann_to_type(ann);
        }
    }

    ret tup(ty_param_count, result_ty);
}

fn stmt_ty(@ast.stmt s) -> @t {
    alt (s.node) {
        case (ast.stmt_expr(?e,_)) {
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

// Returns the type of a pattern as a monotype. Like @expr_ty, this function
// doesn't provide type parameter substitutions.
fn pat_ty(@ast.pat pat) -> @t {
    alt (pat.node) {
        case (ast.pat_wild(?ann))           { ret ann_to_monotype(ann); }
        case (ast.pat_lit(_, ?ann))         { ret ann_to_monotype(ann); }
        case (ast.pat_bind(_, _, ?ann))     { ret ann_to_monotype(ann); }
        case (ast.pat_tag(_, _, _, ?ann))   { ret ann_to_monotype(ann); }
    }
    fail;   // not reached
}

fn expr_ann(@ast.expr expr) -> option.t[ast.ann] {
    alt (expr.node) {
        case (ast.expr_vec(_, _, ?ann))       { ret some[ast.ann](ann); }
        case (ast.expr_tup(_, ?ann))          { ret some[ast.ann](ann); }
        case (ast.expr_rec(_, _, ?ann))       { ret some[ast.ann](ann); }
        case (ast.expr_bind(_, _, ?ann))      { ret some[ast.ann](ann); }
        case (ast.expr_call(_, _, ?ann))      { ret some[ast.ann](ann); }
        case (ast.expr_self_method(_, ?ann))  { ret some[ast.ann](ann); }
        case (ast.expr_spawn(_, _, _, _, ?ann))
                                              { ret some[ast.ann](ann); }
        case (ast.expr_binary(_, _, _, ?ann)) { ret some[ast.ann](ann); }
        case (ast.expr_unary(_, _, ?ann))     { ret some[ast.ann](ann); }
        case (ast.expr_lit(_, ?ann))          { ret some[ast.ann](ann); }
        case (ast.expr_cast(_, _, ?ann))      { ret some[ast.ann](ann); }
        case (ast.expr_if(_, _, _, ?ann))     { ret some[ast.ann](ann); }
        case (ast.expr_for(_, _, _, ?ann))    { ret some[ast.ann](ann); }
        case (ast.expr_for_each(_, _, _, ?ann))
                                              { ret some[ast.ann](ann); }
        case (ast.expr_while(_, _, ?ann))     { ret some[ast.ann](ann); }
        case (ast.expr_do_while(_, _, ?ann))  { ret some[ast.ann](ann); }
        case (ast.expr_alt(_, _, ?ann))       { ret some[ast.ann](ann); }
        case (ast.expr_block(_, ?ann))        { ret some[ast.ann](ann); }
        case (ast.expr_assign(_, _, ?ann))    { ret some[ast.ann](ann); }
        case (ast.expr_assign_op(_, _, _, ?ann))
                                              { ret some[ast.ann](ann); }
        case (ast.expr_field(_, _, ?ann))     { ret some[ast.ann](ann); }
        case (ast.expr_index(_, _, ?ann))     { ret some[ast.ann](ann); }
        case (ast.expr_path(_, _, ?ann))      { ret some[ast.ann](ann); }
        case (ast.expr_ext(_, _, _, _, ?ann)) { ret some[ast.ann](ann); }
        case (ast.expr_port(?ann))            { ret some[ast.ann](ann); }
        case (ast.expr_chan(_, ?ann))         { ret some[ast.ann](ann); }
        case (ast.expr_send(_, _, ?ann))      { ret some[ast.ann](ann); }
        case (ast.expr_recv(_, _, ?ann))      { ret some[ast.ann](ann); }

        case (ast.expr_fail(_))               { ret none[ast.ann]; }
        case (ast.expr_break(_))              { ret none[ast.ann]; }
        case (ast.expr_cont(_))               { ret none[ast.ann]; }
        case (ast.expr_log(_,_,_))            { ret none[ast.ann]; }
        case (ast.expr_check_expr(_,_))       { ret none[ast.ann]; }
        case (ast.expr_ret(_,_))              { ret none[ast.ann]; }
        case (ast.expr_put(_,_))              { ret none[ast.ann]; }
        case (ast.expr_be(_,_))               { ret none[ast.ann]; }
    }
    fail;
}

// Returns the type of an expression as a monotype.
//
// NB: This type doesn't provide type parameter substitutions; e.g. if you
// ask for the type of "id" in "id(3)", it will return "fn(&int) -> int"
// instead of "fn(&T) -> T with T = int". If this isn't what you want, see
// expr_ty_params_and_ty() below.
fn expr_ty(@ast.expr expr) -> @t {
    alt (expr_ann(expr)) {
        case (none[ast.ann])     { ret plain_ty(ty_nil);   }
        case (some[ast.ann](?a)) { ret ann_to_monotype(a); }
    }
}

fn expr_ty_params_and_ty(@ast.expr expr) -> tup(vec[@t], @t) {
    alt (expr_ann(expr)) {
        case (none[ast.ann]) {
            let vec[@t] tps = vec();
            ret tup(tps, plain_ty(ty_nil));
        }
        case (some[ast.ann](?a)) {
            ret tup(ann_to_type_params(a), ann_to_type(a));
        }
    }
}

fn expr_has_ty_params(@ast.expr expr) -> bool {
    // FIXME: Rewrite using complex patterns when they're trustworthy.
    alt (expr_ann(expr)) {
        case (none[ast.ann]) { fail; }
        case (some[ast.ann](?a)) {
            alt (a) {
                case (ast.ann_none) { fail; }
                case (ast.ann_type(_, ?tps_opt, _)) {
                    ret !option.is_none[vec[@t]](tps_opt);
                }
            }
        }
    }
}

// FIXME: At the moment this works only for call, bind, and path expressions.
fn replace_expr_type(@ast.expr expr, tup(vec[@t], @t) new_tyt) -> @ast.expr {
    auto new_tps;
    if (expr_has_ty_params(expr)) {
        new_tps = some[vec[@t]](new_tyt._0);
    } else {
        new_tps = none[vec[@t]];
    }

    auto ann = ast.ann_type(new_tyt._1, new_tps, none[@ts_ann]);

    alt (expr.node) {
        case (ast.expr_call(?callee, ?args, _)) {
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_call(callee, args, ann));
        }
        case (ast.expr_self_method(?ident, _)) {
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_self_method(ident, ann));
        }
        case (ast.expr_bind(?callee, ?args, _)) {
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_bind(callee, args, ann));
        }
        case (ast.expr_field(?e, ?i, _)) {
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_field(e, i, ann));
        }
        case (ast.expr_path(?p, ?dopt, _)) {
            ret @fold.respan[ast.expr_](expr.span,
                                        ast.expr_path(p, dopt, ann));
        }
        case (_) {
            log_err "unhandled expr type in replace_expr_type(): " +
                pretty.pprust.expr_to_str(expr);
            fail;
        }
    }
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
                s += _str.unsafe_from_byte(c);
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

fn sort_methods(vec[method] meths) -> vec[method] {
    fn method_lteq(&method a, &method b) -> bool {
        ret _str.lteq(a.ident, b.ident);
    }

    ret std.sort.merge_sort[method](bind method_lteq(_,_), meths);
}

fn is_lval(@ast.expr expr) -> bool {
    alt (expr.node) {
        case (ast.expr_field(_,_,_))    { ret true;  }
        case (ast.expr_index(_,_,_))    { ret true;  }
        case (ast.expr_path(_,_,_))     { ret true;  }
        case (_)                        { ret false; }
    }
}

// Type unification via Robinson's algorithm (Robinson 1965). Implemented as
// described in Hoder and Voronkov:
//
//     http://www.cs.man.ac.uk/~hoderk/ubench/unification_full.pdf

type var_bindings = rec(UFind.ufind sets,
                        hashmap[int,uint] var_ids,
                        mutable vec[mutable vec[@t]] types);

fn unify(@ty.t expected, @ty.t actual, &unify_handler handler)
        -> unify_result {
    // Wraps the given type in an appropriate cname.
    //
    // TODO: This doesn't do anything yet. We should carry the cname up from
    // the expected and/or actual types when unification results in a type
    // identical to one or both of the two. The precise algorithm for this is
    // something we'll probably need to develop over time.

    // Simple structural type comparison.
    fn struct_cmp(@ty.t expected, @ty.t actual) -> unify_result {
        if (expected.struct == actual.struct) {
            ret ures_ok(expected);
        }

        ret ures_err(terr_mismatch, expected, actual);
    }

    // Unifies two mutability flags.
    fn unify_mut(ast.mutability expected, ast.mutability actual)
            -> option.t[ast.mutability] {
        if (expected == actual) {
            ret some[ast.mutability](expected);
        }
        if (expected == ast.maybe_mut) {
            ret some[ast.mutability](actual);
        }
        if (actual == ast.maybe_mut) {
            ret some[ast.mutability](expected);
        }
        ret none[ast.mutability];
    }

    tag fn_common_res {
        fn_common_res_err(unify_result);
        fn_common_res_ok(vec[arg], @t);
    }

    fn unify_fn_common(&var_bindings bindings,
                       @ty.t expected,
                       @ty.t actual,
                       &unify_handler handler,
                       vec[arg] expected_inputs, @t expected_output,
                       vec[arg] actual_inputs, @t actual_output)
        -> fn_common_res {
        auto expected_len = _vec.len[arg](expected_inputs);
        auto actual_len = _vec.len[arg](actual_inputs);
        if (expected_len != actual_len) {
            ret fn_common_res_err(ures_err(terr_arg_count,
                                           expected, actual));
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

            auto result = unify_step(bindings,
                                     actual_input.ty,
                                     expected_input.ty,
                                     handler);

            alt (result) {
                case (ures_ok(?rty)) {
                    result_ins += vec(rec(mode=result_mode,
                                          ty=rty));
                }

                case (_) {
                    ret fn_common_res_err(result);
                }
            }

            i += 1u;
        }

        // Check the output.
        auto result = unify_step(bindings,
                                 expected_output,
                                 actual_output,
                                 handler);
        alt (result) {
            case (ures_ok(?rty)) {
                ret fn_common_res_ok(result_ins, rty);
            }

            case (_) {
                ret fn_common_res_err(result);
            }
        }
    }

    fn unify_fn(&var_bindings bindings,
                ast.proto e_proto,
                ast.proto a_proto,
                @ty.t expected,
                @ty.t actual,
                &unify_handler handler,
                vec[arg] expected_inputs, @t expected_output,
                vec[arg] actual_inputs, @t actual_output)
        -> unify_result {

        if (e_proto != a_proto) {
            ret ures_err(terr_mismatch, expected, actual);
        }
        auto t = unify_fn_common(bindings, expected, actual,
                                 handler, expected_inputs, expected_output,
                                 actual_inputs, actual_output);
        alt (t) {
            case (fn_common_res_err(?r)) {
                ret r;
            }
            case (fn_common_res_ok(?result_ins, ?result_out)) {
                auto t2 = plain_ty(ty.ty_fn(e_proto, result_ins, result_out));
                ret ures_ok(t2);
            }
        }
    }

    fn unify_native_fn(&var_bindings bindings,
                       ast.native_abi e_abi,
                       ast.native_abi a_abi,
                       @ty.t expected,
                       @ty.t actual,
                       &unify_handler handler,
                       vec[arg] expected_inputs, @t expected_output,
                       vec[arg] actual_inputs, @t actual_output)
        -> unify_result {
        if (e_abi != a_abi) {
            ret ures_err(terr_mismatch, expected, actual);
        }

        auto t = unify_fn_common(bindings, expected, actual,
                                 handler, expected_inputs, expected_output,
                                 actual_inputs, actual_output);
        alt (t) {
            case (fn_common_res_err(?r)) {
                ret r;
            }
            case (fn_common_res_ok(?result_ins, ?result_out)) {
                auto t2 = plain_ty(ty.ty_native_fn(e_abi, result_ins,
                                                   result_out));
                ret ures_ok(t2);
            }
        }
    }

    fn unify_obj(&var_bindings bindings,
                 @ty.t expected,
                 @ty.t actual,
                 &unify_handler handler,
                 vec[method] expected_meths,
                 vec[method] actual_meths) -> unify_result {
      let vec[method] result_meths = vec();
      let uint i = 0u;
      let uint expected_len = _vec.len[method](expected_meths);
      let uint actual_len = _vec.len[method](actual_meths);

      if (expected_len != actual_len) {
        ret ures_err(terr_meth_count, expected, actual);
      }

      while (i < expected_len) {
        auto e_meth = expected_meths.(i);
        auto a_meth = actual_meths.(i);
        if (! _str.eq(e_meth.ident, a_meth.ident)) {
          ret ures_err(terr_obj_meths(e_meth.ident, a_meth.ident),
                       expected, actual);
        }
        auto r = unify_fn(bindings,
                          e_meth.proto, a_meth.proto,
                          expected, actual, handler,
                          e_meth.inputs, e_meth.output,
                          a_meth.inputs, a_meth.output);
        alt (r) {
            case (ures_ok(?tfn)) {
                alt (tfn.struct) {
                    case (ty_fn(?proto, ?ins, ?out)) {
                        result_meths += vec(rec(inputs = ins,
                                                output = out
                                                with e_meth));
                    }
                }
            }
            case (_) {
                ret r;
            }
        }
        i += 1u;
      }
      auto t = plain_ty(ty_obj(result_meths));
      ret ures_ok(t);
    }

    fn get_or_create_set(&var_bindings bindings, int id) -> uint {
        auto set_num;
        alt (bindings.var_ids.find(id)) {
        case (none[uint]) {
            set_num = UFind.make_set(bindings.sets);
            bindings.var_ids.insert(id, set_num);
        }
        case (some[uint](?n)) { set_num = n; }
        }
        ret set_num;
    }

    fn unify_step(&var_bindings bindings, @ty.t expected, @ty.t actual,
                  &unify_handler handler) -> unify_result {
        // TODO: rewrite this using tuple pattern matching when available, to
        // avoid all this rightward drift and spikiness.

        // TODO: occurs check, to make sure we don't loop forever when
        // unifying e.g. 'a and option['a]

        alt (actual.struct) {
            // If the RHS is a variable type, then just do the appropriate
            // binding.
            case (ty.ty_var(?actual_id)) {
                auto actual_n = get_or_create_set(bindings, actual_id);
                alt (expected.struct) {
                    case (ty.ty_var(?expected_id)) {
                        auto expected_n = get_or_create_set(bindings,
                                                            expected_id);
                        UFind.union(bindings.sets, expected_n, actual_n);
                    }

                    case (_) {
                        // Just bind the type variable to the expected type.
                        auto vlen = _vec.len[mutable vec[@t]](bindings.types);
                        if (actual_n < vlen) {
                            bindings.types.(actual_n) += vec(expected);
                        } else {
                            check (actual_n == vlen);
                            bindings.types += vec(mutable vec(expected));
                        }
                    }
                }
                ret ures_ok(actual);
            }
            case (ty.ty_local(?actual_id)) {
                auto result_ty;
                alt (handler.resolve_local(actual_id)) {
                    case (none[@ty.t]) { result_ty = expected; }
                    case (some[@ty.t](?actual_ty)) {
                        auto result = unify_step(bindings,
                                                 expected,
                                                 actual_ty,
                                                 handler);
                        alt (result) {
                            case (ures_ok(?rty)) { result_ty = rty; }
                            case (_) { ret result; }
                        }
                    }
                }

                handler.record_local(actual_id, result_ty);
                ret ures_ok(result_ty);
            }
            case (ty.ty_bound_param(?actual_id)) {
                alt (expected.struct) {
                    case (ty.ty_local(_)) {
                        log_err "TODO: bound param unifying with local";
                        fail;
                    }

                    case (_) {
                        ret handler.record_param(actual_id, expected);
                    }
                }
            }
            case (_) { /* empty */ }
        }

        alt (expected.struct) {
            case (ty.ty_nil)        { ret struct_cmp(expected, actual); }
            case (ty.ty_bool)       { ret struct_cmp(expected, actual); }
            case (ty.ty_int)        { ret struct_cmp(expected, actual); }
            case (ty.ty_uint)       { ret struct_cmp(expected, actual); }
            case (ty.ty_machine(_)) { ret struct_cmp(expected, actual); }
            case (ty.ty_float)      { ret struct_cmp(expected, actual); }
            case (ty.ty_char)       { ret struct_cmp(expected, actual); }
            case (ty.ty_str)        { ret struct_cmp(expected, actual); }
            case (ty.ty_type)       { ret struct_cmp(expected, actual); }
            case (ty.ty_native)     { ret struct_cmp(expected, actual); }
            case (ty.ty_param(_))   { ret struct_cmp(expected, actual); }

            case (ty.ty_tag(?expected_id, ?expected_tps)) {
                alt (actual.struct) {
                    case (ty.ty_tag(?actual_id, ?actual_tps)) {
                        if (expected_id._0 != actual_id._0 ||
                                expected_id._1 != actual_id._1) {
                            ret ures_err(terr_mismatch, expected, actual);
                        }

                        // TODO: factor this cruft out, see the TODO in the
                        // ty.ty_tup case
                        let vec[@ty.t] result_tps = vec();
                        auto i = 0u;
                        auto expected_len = _vec.len[@ty.t](expected_tps);
                        while (i < expected_len) {
                            auto expected_tp = expected_tps.(i);
                            auto actual_tp = actual_tps.(i);

                            auto result = unify_step(bindings,
                                                     expected_tp,
                                                     actual_tp,
                                                     handler);

                            alt (result) {
                                case (ures_ok(?rty)) {
                                    _vec.push[@ty.t](result_tps, rty);
                                }
                                case (_) {
                                    ret result;
                                }
                            }

                            i += 1u;
                        }

                        ret ures_ok(plain_ty(ty.ty_tag(expected_id,
                                                       result_tps)));
                    }
                    case (_) { /* fall through */ }
                }

                ret ures_err(terr_mismatch, expected, actual);
            }

            case (ty.ty_box(?expected_mt)) {
                alt (actual.struct) {
                    case (ty.ty_box(?actual_mt)) {
                        auto mut;
                        alt (unify_mut(expected_mt.mut, actual_mt.mut)) {
                            case (none[ast.mutability]) {
                                ret ures_err(terr_box_mutability, expected,
                                             actual);
                            }
                            case (some[ast.mutability](?m)) { mut = m; }
                        }

                        auto result = unify_step(bindings,
                                                 expected_mt.ty,
                                                 actual_mt.ty,
                                                 handler);
                        alt (result) {
                            case (ures_ok(?result_sub)) {
                                auto mt = rec(ty=result_sub, mut=mut);
                                ret ures_ok(plain_ty(ty.ty_box(mt)));
                            }
                            case (_) {
                                ret result;
                            }
                        }
                    }

                    case (_) {
                        ret ures_err(terr_mismatch, expected, actual);
                    }
                }
            }

            case (ty.ty_vec(?expected_mt)) {
                alt (actual.struct) {
                    case (ty.ty_vec(?actual_mt)) {
                        auto mut;
                        alt (unify_mut(expected_mt.mut, actual_mt.mut)) {
                            case (none[ast.mutability]) {
                                ret ures_err(terr_vec_mutability, expected,
                                             actual);
                            }
                            case (some[ast.mutability](?m)) { mut = m; }
                        }

                        auto result = unify_step(bindings,
                                                 expected_mt.ty,
                                                 actual_mt.ty,
                                                 handler);
                        alt (result) {
                            case (ures_ok(?result_sub)) {
                                auto mt = rec(ty=result_sub, mut=mut);
                                ret ures_ok(plain_ty(ty.ty_vec(mt)));
                            }
                            case (_) {
                                ret result;
                            }
                        }
                    }

                    case (_) {
                        ret ures_err(terr_mismatch, expected, actual);
                   }
                }
            }

            case (ty.ty_port(?expected_sub)) {
                alt (actual.struct) {
                    case (ty.ty_port(?actual_sub)) {
                        auto result = unify_step(bindings,
                                                 expected_sub,
                                                 actual_sub,
                                                 handler);
                        alt (result) {
                            case (ures_ok(?result_sub)) {
                                ret ures_ok(plain_ty(ty.ty_port(result_sub)));
                            }
                            case (_) {
                                ret result;
                            }
                        }
                    }

                    case (_) {
                        ret ures_err(terr_mismatch, expected, actual);
                    }
                }
            }

            case (ty.ty_chan(?expected_sub)) {
                alt (actual.struct) {
                    case (ty.ty_chan(?actual_sub)) {
                        auto result = unify_step(bindings,
                                                 expected_sub,
                                                 actual_sub,
                                                 handler);
                        alt (result) {
                            case (ures_ok(?result_sub)) {
                                ret ures_ok(plain_ty(ty.ty_chan(result_sub)));
                            }
                            case (_) {
                                ret result;
                            }
                        }
                    }

                    case (_) {
                        ret ures_err(terr_mismatch, expected, actual);
                    }
                }
            }

            case (ty.ty_tup(?expected_elems)) {
                alt (actual.struct) {
                    case (ty.ty_tup(?actual_elems)) {
                        auto expected_len = _vec.len[ty.mt](expected_elems);
                        auto actual_len = _vec.len[ty.mt](actual_elems);
                        if (expected_len != actual_len) {
                            auto err = terr_tuple_size(expected_len,
                                                       actual_len);
                            ret ures_err(err, expected, actual);
                        }

                        // TODO: implement an iterator that can iterate over
                        // two arrays simultaneously.
                        let vec[ty.mt] result_elems = vec();
                        auto i = 0u;
                        while (i < expected_len) {
                            auto expected_elem = expected_elems.(i);
                            auto actual_elem = actual_elems.(i);

                            auto mut;
                            alt (unify_mut(expected_elem.mut,
                                           actual_elem.mut)) {
                                case (none[ast.mutability]) {
                                    auto err = terr_tuple_mutability;
                                    ret ures_err(err, expected, actual);
                                }
                                case (some[ast.mutability](?m)) { mut = m; }
                            }

                            auto result = unify_step(bindings,
                                                     expected_elem.ty,
                                                     actual_elem.ty,
                                                     handler);
                            alt (result) {
                                case (ures_ok(?rty)) {
                                    auto mt = rec(ty=rty, mut=mut);
                                    result_elems += vec(mt);
                                }
                                case (_) {
                                    ret result;
                                }
                            }

                            i += 1u;
                        }

                        ret ures_ok(plain_ty(ty.ty_tup(result_elems)));
                    }

                    case (_) {
                        ret ures_err(terr_mismatch, expected, actual);
                    }
                }
            }

            case (ty.ty_rec(?expected_fields)) {
                alt (actual.struct) {
                    case (ty.ty_rec(?actual_fields)) {
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

                            auto mut;
                            alt (unify_mut(expected_field.mt.mut,
                                           actual_field.mt.mut)) {
                                case (none[ast.mutability]) {
                                    ret ures_err(terr_record_mutability,
                                                 expected, actual);
                                }
                                case (some[ast.mutability](?m)) { mut = m; }
                            }

                            if (!_str.eq(expected_field.ident,
                                         actual_field.ident)) {
                                auto err =
                                    terr_record_fields(expected_field.ident,
                                                       actual_field.ident);
                                ret ures_err(err, expected, actual);
                            }

                            auto result = unify_step(bindings,
                                                     expected_field.mt.ty,
                                                     actual_field.mt.ty,
                                                     handler);
                            alt (result) {
                                case (ures_ok(?rty)) {
                                    auto mt = rec(ty=rty, mut=mut);
                                    _vec.push[field]
                                        (result_fields,
                                         rec(mt=mt with expected_field));
                                }
                                case (_) {
                                    ret result;
                                }
                            }

                            i += 1u;
                        }

                        ret ures_ok(plain_ty(ty.ty_rec(result_fields)));
                    }

                    case (_) {
                        ret ures_err(terr_mismatch, expected, actual);
                    }
                }
            }

            case (ty.ty_fn(?ep, ?expected_inputs, ?expected_output)) {
                alt (actual.struct) {
                    case (ty.ty_fn(?ap, ?actual_inputs, ?actual_output)) {
                        ret unify_fn(bindings, ep, ap,
                                     expected, actual, handler,
                                     expected_inputs, expected_output,
                                     actual_inputs, actual_output);
                    }

                    case (_) {
                        ret ures_err(terr_mismatch, expected, actual);
                    }
                }
            }

            case (ty.ty_native_fn(?e_abi, ?expected_inputs,
                                  ?expected_output)) {
                alt (actual.struct) {
                    case (ty.ty_native_fn(?a_abi, ?actual_inputs,
                                          ?actual_output)) {
                        ret unify_native_fn(bindings, e_abi, a_abi,
                                            expected, actual, handler,
                                            expected_inputs, expected_output,
                                            actual_inputs, actual_output);
                    }
                    case (_) {
                        ret ures_err(terr_mismatch, expected, actual);
                    }
                }
            }

            case (ty.ty_obj(?expected_meths)) {
                alt (actual.struct) {
                    case (ty.ty_obj(?actual_meths)) {
                        ret unify_obj(bindings, expected, actual, handler,
                                      expected_meths, actual_meths);
                    }
                    case (_) {
                        ret ures_err(terr_mismatch, expected, actual);
                    }
                }
            }

            case (ty.ty_var(?expected_id)) {
                // Add a binding.
                auto expected_n = get_or_create_set(bindings, expected_id);
                auto vlen = _vec.len[mutable vec[@t]](bindings.types);
                if (expected_n < vlen) {
                    bindings.types.(expected_n) += vec(actual);
                } else {
                    check (expected_n == vlen);
                    bindings.types += vec(mutable vec(actual));
                }
                ret ures_ok(expected);
            }

            case (ty.ty_local(?expected_id)) {
                auto result_ty;
                alt (handler.resolve_local(expected_id)) {
                    case (none[@ty.t]) { result_ty = actual; }
                    case (some[@ty.t](?expected_ty)) {
                        auto result = unify_step(bindings,
                                                 expected_ty,
                                                 actual,
                                                 handler);
                        alt (result) {
                            case (ures_ok(?rty)) { result_ty = rty; }
                            case (_) { ret result; }
                        }
                    }
                }

                handler.record_local(expected_id, result_ty);
                ret ures_ok(result_ty);
            }

            case (ty.ty_bound_param(?expected_id)) {
                ret handler.record_param(expected_id, actual);
            }
        }

        // TODO: remove me once match-exhaustiveness checking works
        fail;
    }

    // Performs type binding substitution.
    fn substitute(var_bindings bindings, vec[@t] set_types, @t typ) -> @t {
        fn substituter(var_bindings bindings, vec[@t] types, @t typ) -> @t {
            alt (typ.struct) {
                case (ty_var(?id)) {
                    alt (bindings.var_ids.find(id)) {
                        case (some[uint](?n)) {
                            auto root = UFind.find(bindings.sets, n);
                            ret types.(root);
                        }
                        case (none[uint]) { ret typ; }
                    }
                }
                case (_) { ret typ; }
            }
        }

        auto f = bind substituter(bindings, set_types, _);
        ret fold_ty(f, typ);
    }

    fn unify_sets(&var_bindings bindings) -> vec[@t] {
        let vec[@t] throwaway = vec();
        let vec[mutable vec[@t]] set_types = vec(mutable throwaway);
        _vec.pop[mutable vec[@t]](set_types);   // FIXME: botch

        for (UFind.node node in bindings.sets.nodes) {
            let vec[@t] v = vec();
            set_types += vec(mutable v);
        }

        auto i = 0u;
        while (i < _vec.len[mutable vec[@t]](set_types)) {
            auto root = UFind.find(bindings.sets, i);
            set_types.(root) += bindings.types.(i);
            i += 1u;
        }

        let vec[@t] result = vec();
        for (vec[@t] types in set_types) {
            if (_vec.len[@t](types) > 1u) {
                log_err "unification of > 1 types in a type set is " +
                    "unimplemented";
                fail;
            }
            result += vec(types.(0));
        }

        ret result;
    }

    let vec[@t] throwaway = vec();
    let vec[mutable vec[@t]] types = vec(mutable throwaway);
    _vec.pop[mutable vec[@t]](types);   // FIXME: botch

    auto bindings = rec(sets=UFind.make(),
                        var_ids=common.new_int_hash[uint](),
                        mutable types=types);

    auto ures = unify_step(bindings, expected, actual, handler);
    alt (ures) {
    case (ures_ok(?t)) {
        auto set_types = unify_sets(bindings);
        auto t2 = substitute(bindings, set_types, t);
        ret ures_ok(t2);
    }
    case (_) { ret ures; }
    }
    fail;   // not reached
}

fn type_err_to_str(&ty.type_err err) -> str {
    alt (err) {
        case (terr_mismatch) {
            ret "types differ";
        }
        case (terr_box_mutability) {
            ret "boxed values differ in mutability";
        }
        case (terr_vec_mutability) {
            ret "vectors differ in mutability";
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
        case (terr_meth_count) {
            ret "incorrect number of object methods";
        }
        case (terr_obj_meths(?e_meth, ?a_meth)) {
            ret "expected an obj with method '" + e_meth +
                "' but found one with method '" + a_meth +
                "'";
        }
    }
}

// Performs bound type parameter replacement using the supplied mapping from
// parameter IDs to types.
fn substitute_type_params(vec[@t] bindings, @t typ) -> @t {
    fn replacer(vec[@t] bindings, @t typ) -> @t {
        alt (typ.struct) {
            case (ty_bound_param(?param_index)) {
                ret bindings.(param_index);
            }
            case (_) { ret typ; }
        }
    }

    auto f = bind replacer(bindings, _);
    ret fold_ty(f, typ);
}

// Converts type parameters in a type to bound type parameters.
fn bind_params_in_type(@t typ) -> @t {
    fn binder(@t typ) -> @t {
        alt (typ.struct) {
            case (ty_bound_param(?index)) {
                log_err "bind_params_in_type() called on type that already " +
                    "has bound params in it";
                fail;
            }
            case (ty_param(?index)) { ret plain_ty(ty_bound_param(index)); }
            case (_) { ret typ; }
        }
    }

    auto f = binder;
    ret fold_ty(f, typ);
}


fn def_has_ty_params(&ast.def def) -> bool {
    alt (def) {
        case (ast.def_fn(_))            { ret true;  }
        case (ast.def_obj(_))           { ret true;  }
        case (ast.def_obj_field(_))     { ret false; }
        case (ast.def_mod(_))           { ret false; }
        case (ast.def_const(_))         { ret false; }
        case (ast.def_arg(_))           { ret false; }
        case (ast.def_local(_))         { ret false; }
        case (ast.def_variant(_, _))    { ret true;  }
        case (ast.def_ty(_))            { ret false; }
        case (ast.def_ty_arg(_))        { ret false; }
        case (ast.def_binding(_))       { ret false; }
        case (ast.def_use(_))           { ret false; }
        case (ast.def_native_ty(_))     { ret false; }
        case (ast.def_native_fn(_))     { ret true;  }
    }
}

// If the given item is in an external crate, looks up its type and adds it to
// the type cache. Returns the type parameters and type.
fn lookup_item_type(session.session sess, &type_cache cache,
                    ast.def_id did) -> ty_param_count_and_ty {
    if (did._0 == sess.get_targ_crate_num()) {
        // The item is in this crate. The caller should have added it to the
        // type cache already; we simply return it.
        check (cache.contains_key(did));
        ret cache.get(did);
    }

    if (cache.contains_key(did)) {
        ret cache.get(did);
    }

    auto tyt = creader.get_type(sess, did);
    cache.insert(did, tyt);
    ret tyt;
}


// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
