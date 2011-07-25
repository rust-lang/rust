import std::int;
import std::ivec;
import std::str;
import std::uint;
import std::box;
import std::ufindivec;
import std::map;
import std::map::hashmap;
import std::option;
import std::option::none;
import std::option::some;
import std::smallintmap;
import driver::session;
import syntax::ast;
import syntax::ast::*;
import syntax::codemap::span;
import metadata::csearch;
import util::common::*;
import syntax::util::interner;
import util::ppaux::ty_to_str;
import util::ppaux::ty_constr_to_str;
import util::ppaux::mode_str_1;
import syntax::print::pprust::*;

export node_id_to_monotype;
export node_id_to_type;
export node_id_to_type_params;
export node_id_to_ty_param_substs_opt_and_ty;
export any_item_native;
export any_item_rust;
export arg;
export args_eq;
export ast_constr_to_constr;
export bind_params_in_type;
export block_ty;
export constr;
export constr_;
export constr_general;
export constr_table;
export count_ty_params;
export ctxt;
export decl_local_ty;
export def_has_ty_params;
export eq_ty;
export expr_has_ty_params;
export expr_ty;
export fold_ty;
export field;
export field_idx;
export field_num;
export fm_general;
export get_element_type;
export hash_ty;
export idx_nil;
export is_lval;
export is_binopable;
export item_table;
export lookup_item_type;
export method;
export method_idx;
export method_ty_to_fn_ty;
export mk_bool;
export mk_bot;
export mk_box;
export mk_chan;
export mk_char;
export mk_constr;
export mk_ctxt;
export mk_float;
export mk_fn;
export mk_imm_box;
export mk_imm_tup;
export mk_imm_vec;
export mk_int;
export mk_istr;
export mk_ivec;
export mk_mach;
export mk_native;
export mk_native_fn;
export mk_nil;
export mk_obj;
export mk_res;
export mk_param;
export mk_port;
export mk_ptr;
export mk_rec;
export mk_str;
export mk_tag;
export mk_task;
export mk_tup;
export mk_type;
export mk_uint;
export mk_var;
export mk_vec;
export mode;
export mo_val;
export mo_alias;
export mt;
export node_type_table;
export pat_ty;
export cname;
export rename;
export ret_ty_of_fn;
export ret_ty_of_fn_ty;
export ret_ty_to_fn_ty;
export sequence_element_type;
export sequence_is_interior;
export struct;
export sort_methods;
export stmt_node_id;
export strip_cname;
export sty;
export substitute_type_params;
export t;
export tag_variants;
export tag_variant_with_id;
export ty_param_substs_opt_and_ty;
export ty_param_count_and_ty;
export ty_native_fn;
export ty_bool;
export ty_bot;
export ty_box;
export ty_chan;
export ty_char;
export ty_constr;
export ty_constr_arg;
export ty_float;
export ty_fn;
export ty_fn_abi;
export ty_fn_proto;
export ty_fn_ret;
export ty_int;
export ty_istr;
export ty_ivec;
export ty_machine;
export ty_native;
export ty_nil;
export ty_obj;
export ty_res;
export ty_param;
export ty_port;
export ty_ptr;
export ty_rec;
export ty_str;
export ty_tag;
export ty_task;
export ty_tup;
export ty_type;
export ty_uint;
export ty_var;
export ty_var_id;
export ty_vec;
export ty_param_substs_opt_and_ty_to_monotype;
export ty_fn_args;
export type_constr;
export type_contains_params;
export type_contains_vars;
export type_err;
export type_err_to_str;
export type_has_dynamic_size;
export type_has_pointers;
export type_is_bool;
export type_is_bot;
export type_is_box;
export type_is_boxed;
export type_is_chan;
export type_is_fp;
export type_is_integral;
export type_is_native;
export type_is_nil;
export type_is_scalar;
export type_is_sequence;
export type_is_signed;
export type_is_structural;
export type_is_copyable;
export type_is_tup_like;
export type_is_str;
export type_owns_heap_mem;
export type_autoderef;
export type_param;
export unify;
export variant_info;
export walk_ty;

// Data types
tag mode { mo_val; mo_alias(bool); }

type arg = rec(mode mode, t ty);

type field = rec(ast::ident ident, mt mt);

type method =
    rec(ast::proto proto,
        ast::ident ident,
        arg[] inputs,
        t output,
        controlflow cf,
        (@constr)[] constrs);

type constr_table = hashmap[ast::node_id, constr[]];

type mt = rec(t ty, ast::mutability mut);


// Contains information needed to resolve types and (in the future) look up
// the types of AST nodes.
type creader_cache = hashmap[tup(int, uint, uint), ty::t];

type ctxt =
    @rec(@type_store ts,
         session::session sess,
         resolve::def_map def_map,
         node_type_table node_types,
         ast_map::map items,
         freevars::freevar_map freevars,

         //        constr_table fn_constrs,
         type_cache tcache,
         creader_cache rcache,
         hashmap[t, str] short_names_cache,
         hashmap[t, bool] has_pointer_cache,
         hashmap[t, bool] owns_heap_mem_cache,
         hashmap[@ast::ty, option::t[t]] ast_ty_to_ty_cache);

type ty_ctxt = ctxt;


// Needed for disambiguation from unify::ctxt.
// Convert from method type to function type.  Pretty easy; we just drop
// 'ident'.
fn method_ty_to_fn_ty(&ctxt cx, method m) -> t {
    ret mk_fn(cx, m.proto, m.inputs, m.output, m.cf, m.constrs);
}


// Never construct these manually. These are interned.
type raw_t =
    rec(sty struct,
        option::t[str] cname,
        uint hash,
        bool has_params,
        bool has_vars);

type t = uint;


// NB: If you change this, you'll probably want to change the corresponding
// AST structure in front/ast::rs as well.
tag sty {
    ty_nil;
    ty_bot;
    ty_bool;
    ty_int;
    ty_float;
    ty_uint;
    ty_machine(ast::ty_mach);
    ty_char;
    ty_str;
    ty_istr;
    ty_tag(def_id, t[]);
    ty_box(mt);
    ty_vec(mt);
    ty_ivec(mt);
    ty_ptr(mt);
    ty_port(t);
    ty_chan(t);
    ty_task;
    ty_tup(mt[]);
    ty_rec(field[]);
    ty_fn(ast::proto, arg[], t, controlflow, (@constr)[]);
    ty_native_fn(ast::native_abi, arg[], t);
    ty_obj(method[]);
    ty_res(def_id, t, t[]);
    ty_var(int); // type variable
    ty_param(uint); // fn/tag type param
    ty_type;
    ty_native(def_id);
    ty_constr(t, (@type_constr)[]);
    // TODO: ty_fn_arg(t), for a possibly-aliased function argument
}

// In the middle end, constraints have a def_id attached, referring
// to the definition of the operator in the constraint.
type constr_general[ARG] = spanned[constr_general_[ARG, def_id]];
type type_constr = constr_general[path];
type constr = constr_general[uint];

// Data structures used in type unification
tag type_err {
    terr_mismatch;
    terr_controlflow_mismatch;
    terr_box_mutability;
    terr_vec_mutability;
    terr_tuple_size(uint, uint);
    terr_tuple_mutability;
    terr_record_size(uint, uint);
    terr_record_mutability;
    terr_record_fields(ast::ident, ast::ident);
    terr_meth_count;
    terr_obj_meths(ast::ident, ast::ident);
    terr_arg_count;
    terr_mode_mismatch(mode, mode);
    terr_constr_len(uint, uint);
    terr_constr_mismatch(@type_constr, @type_constr);
}

type ty_param_count_and_ty = tup(uint, t);

type type_cache = hashmap[ast::def_id, ty_param_count_and_ty];

const uint idx_nil = 0u;

const uint idx_bool = 1u;

const uint idx_int = 2u;

const uint idx_float = 3u;

const uint idx_uint = 4u;

const uint idx_i8 = 5u;

const uint idx_i16 = 6u;

const uint idx_i32 = 7u;

const uint idx_i64 = 8u;

const uint idx_u8 = 9u;

const uint idx_u16 = 10u;

const uint idx_u32 = 11u;

const uint idx_u64 = 12u;

const uint idx_f32 = 13u;

const uint idx_f64 = 14u;

const uint idx_char = 15u;

const uint idx_str = 16u;

const uint idx_istr = 17u;

const uint idx_task = 18u;

const uint idx_type = 19u;

const uint idx_bot = 20u;

const uint idx_first_others = 21u;

type type_store = interner::interner[@raw_t];

type ty_param_substs_opt_and_ty = tup(option::t[ty::t[]], ty::t);

type node_type_table =
    @smallintmap::smallintmap[ty::ty_param_substs_opt_and_ty];

fn populate_type_store(&ctxt cx) {
    intern(cx, ty_nil, none);
    intern(cx, ty_bool, none);
    intern(cx, ty_int, none);
    intern(cx, ty_float, none);
    intern(cx, ty_uint, none);
    intern(cx, ty_machine(ast::ty_i8), none);
    intern(cx, ty_machine(ast::ty_i16), none);
    intern(cx, ty_machine(ast::ty_i32), none);
    intern(cx, ty_machine(ast::ty_i64), none);
    intern(cx, ty_machine(ast::ty_u8), none);
    intern(cx, ty_machine(ast::ty_u16), none);
    intern(cx, ty_machine(ast::ty_u32), none);
    intern(cx, ty_machine(ast::ty_u64), none);
    intern(cx, ty_machine(ast::ty_f32), none);
    intern(cx, ty_machine(ast::ty_f64), none);
    intern(cx, ty_char, none);
    intern(cx, ty_str, none);
    intern(cx, ty_istr, none);
    intern(cx, ty_task, none);
    intern(cx, ty_type, none);
    intern(cx, ty_bot, none);
    assert (ivec::len(cx.ts.vect) == idx_first_others);
}

fn mk_rcache() -> creader_cache {
    fn hash_cache_entry(&tup(int, uint, uint) k) -> uint {
        ret (k._0 as uint) + k._1 + k._2;
    }
    fn eq_cache_entries(&tup(int, uint, uint) a, &tup(int, uint, uint) b) ->
       bool {
        ret a._0 == b._0 && a._1 == b._1 && a._2 == b._2;
    }
    auto h = hash_cache_entry;
    auto e = eq_cache_entries;
    ret map::mk_hashmap[tup(int, uint, uint), t](h, e);
}


fn mk_ctxt(session::session s, resolve::def_map dm,
           ast_map::map amap, freevars::freevar_map freevars) -> ctxt {
    let node_type_table ntt =
        @smallintmap::mk[ty::ty_param_substs_opt_and_ty]();
    auto tcache = new_def_hash[ty::ty_param_count_and_ty]();
    auto ts = @interner::mk[@raw_t](hash_raw_ty, eq_raw_ty);
    auto cx =
        @rec(ts=ts,
             sess=s,
             def_map=dm,
             node_types=ntt,
             items=amap,
             freevars=freevars,
             tcache=tcache,
             rcache=mk_rcache(),
             short_names_cache=map::mk_hashmap(ty::hash_ty, ty::eq_ty),
             has_pointer_cache=map::mk_hashmap(ty::hash_ty, ty::eq_ty),
             owns_heap_mem_cache=map::mk_hashmap(ty::hash_ty, ty::eq_ty),
             ast_ty_to_ty_cache=map::mk_hashmap(ast::hash_ty, ast::eq_ty));
    populate_type_store(cx);
    ret cx;
}


// Type constructors
fn mk_raw_ty(&ctxt cx, &sty st, &option::t[str] in_cname) -> @raw_t {
    auto cname = none;
    auto h = hash_type_info(st, cname);
    let bool has_params = false;
    let bool has_vars = false;
    fn derive_flags_t(&ctxt cx, &mutable bool has_params,
                      &mutable bool has_vars, &t tt) {
        auto rt = interner::get[@raw_t](*cx.ts, tt);
        has_params = has_params || rt.has_params;
        has_vars = has_vars || rt.has_vars;
    }
    fn derive_flags_mt(&ctxt cx, &mutable bool has_params,
                       &mutable bool has_vars, &mt m) {
        derive_flags_t(cx, has_params, has_vars, m.ty);
    }
    fn derive_flags_arg(&ctxt cx, &mutable bool has_params,
                        &mutable bool has_vars, &arg a) {
        derive_flags_t(cx, has_params, has_vars, a.ty);
    }
    fn derive_flags_sig(&ctxt cx, &mutable bool has_params,
                        &mutable bool has_vars, &arg[] args, &t tt) {
        for (arg a in args) { derive_flags_arg(cx, has_params, has_vars, a); }
        derive_flags_t(cx, has_params, has_vars, tt);
    }
    alt (st) {
        case (ty_nil) {/* no-op */ }
        case (ty_bot) {/* no-op */ }
        case (ty_bool) {/* no-op */ }
        case (ty_int) {/* no-op */ }
        case (ty_float) {/* no-op */ }
        case (ty_uint) {/* no-op */ }
        case (ty_machine(_)) {/* no-op */ }
        case (ty_char) {/* no-op */ }
        case (ty_str) {/* no-op */ }
        case (ty_istr) {/* no-op */ }
        case (ty_task) {/* no-op */ }
        case (ty_type) {/* no-op */ }
        case (ty_native(_)) {/* no-op */ }
        case (ty_param(_)) { has_params = true; }
        case (ty_var(_)) { has_vars = true; }
        case (ty_tag(_, ?tys)) {
            for (t tt in tys) {
                derive_flags_t(cx, has_params, has_vars, tt);
            }
        }
        case (ty_box(?m)) { derive_flags_mt(cx, has_params, has_vars, m); }
        case (ty_vec(?m)) { derive_flags_mt(cx, has_params, has_vars, m); }
        case (ty_ivec(?m)) { derive_flags_mt(cx, has_params, has_vars, m); }
        case (ty_ptr(?m)) { derive_flags_mt(cx, has_params, has_vars, m); }
        case (ty_port(?tt)) { derive_flags_t(cx, has_params, has_vars, tt); }
        case (ty_chan(?tt)) { derive_flags_t(cx, has_params, has_vars, tt); }
        case (ty_tup(?mts)) {
            for (mt m in mts) {
                derive_flags_mt(cx, has_params, has_vars, m);
            }
        }
        case (ty_rec(?flds)) {
            for (field f in flds) {
                derive_flags_mt(cx, has_params, has_vars, f.mt);
            }
        }
        case (ty_fn(_, ?args, ?tt, _, _)) {
            derive_flags_sig(cx, has_params, has_vars, args, tt);
        }
        case (ty_native_fn(_, ?args, ?tt)) {
            derive_flags_sig(cx, has_params, has_vars, args, tt);
        }
        case (ty_obj(?meths)) {
            for (method m in meths) {
                derive_flags_sig(cx, has_params, has_vars, m.inputs,
                                 m.output);
            }
        }
        case (ty_res(_, ?tt, ?tps)) {
            derive_flags_t(cx, has_params, has_vars, tt);
            for (t tt in tps) {
                derive_flags_t(cx, has_params, has_vars, tt);
            }
        }
        case (ty_constr(?tt, _)) {
            derive_flags_t(cx, has_params, has_vars, tt);
        }
    }
    ret @rec(struct=st,
             cname=cname,
             hash=h,
             has_params=has_params,
             has_vars=has_vars);
}

fn intern(&ctxt cx, &sty st, &option::t[str] cname) {
    interner::intern(*cx.ts, mk_raw_ty(cx, st, cname));
}

fn gen_ty_full(&ctxt cx, &sty st, &option::t[str] cname) -> t {
    auto raw_type = mk_raw_ty(cx, st, cname);
    ret interner::intern(*cx.ts, raw_type);
}


// These are private constructors to this module. External users should always
// use the mk_foo() functions below.
fn gen_ty(&ctxt cx, &sty st) -> t { ret gen_ty_full(cx, st, none); }

fn mk_nil(&ctxt cx) -> t { ret idx_nil; }

fn mk_bot(&ctxt cx) -> t { ret idx_bot; }

fn mk_bool(&ctxt cx) -> t { ret idx_bool; }

fn mk_int(&ctxt cx) -> t { ret idx_int; }

fn mk_float(&ctxt cx) -> t { ret idx_float; }

fn mk_uint(&ctxt cx) -> t { ret idx_uint; }

fn mk_mach(&ctxt cx, &ast::ty_mach tm) -> t {
    alt (tm) {
        case (ast::ty_u8) { ret idx_u8; }
        case (ast::ty_u16) { ret idx_u16; }
        case (ast::ty_u32) { ret idx_u32; }
        case (ast::ty_u64) { ret idx_u64; }
        case (ast::ty_i8) { ret idx_i8; }
        case (ast::ty_i16) { ret idx_i16; }
        case (ast::ty_i32) { ret idx_i32; }
        case (ast::ty_i64) { ret idx_i64; }
        case (ast::ty_f32) { ret idx_f32; }
        case (ast::ty_f64) { ret idx_f64; }
    }
}

fn mk_char(&ctxt cx) -> t { ret idx_char; }

fn mk_str(&ctxt cx) -> t { ret idx_str; }

fn mk_istr(&ctxt cx) -> t { ret idx_istr; }

fn mk_tag(&ctxt cx, &ast::def_id did, &t[] tys) -> t {
    ret gen_ty(cx, ty_tag(did, tys));
}

fn mk_box(&ctxt cx, &mt tm) -> t { ret gen_ty(cx, ty_box(tm)); }

fn mk_ptr(&ctxt cx, &mt tm) -> t { ret gen_ty(cx, ty_ptr(tm)); }

fn mk_imm_box(&ctxt cx, &t ty) -> t {
    ret mk_box(cx, rec(ty=ty, mut=ast::imm));
}

fn mk_vec(&ctxt cx, &mt tm) -> t { ret gen_ty(cx, ty_vec(tm)); }

fn mk_ivec(&ctxt cx, &mt tm) -> t { ret gen_ty(cx, ty_ivec(tm)); }

fn mk_imm_vec(&ctxt cx, &t typ) -> t {
    ret gen_ty(cx, ty_vec(rec(ty=typ, mut=ast::imm)));
}

fn mk_port(&ctxt cx, &t ty) -> t { ret gen_ty(cx, ty_port(ty)); }

fn mk_chan(&ctxt cx, &t ty) -> t { ret gen_ty(cx, ty_chan(ty)); }

fn mk_task(&ctxt cx) -> t { ret gen_ty(cx, ty_task); }

fn mk_tup(&ctxt cx, &mt[] tms) -> t { ret gen_ty(cx, ty_tup(tms)); }

fn mk_imm_tup(&ctxt cx, &t[] tys) -> t {
    // TODO: map

    let ty::mt[] mts = ~[];
    for (t typ in tys) { mts += ~[rec(ty=typ, mut=ast::imm)]; }
    ret mk_tup(cx, mts);
}

fn mk_rec(&ctxt cx, &field[] fs) -> t { ret gen_ty(cx, ty_rec(fs)); }

fn mk_constr(&ctxt cx, &t t, &(@type_constr)[] cs) -> t {
    ret gen_ty(cx, ty_constr(t, cs));
}

fn mk_fn(&ctxt cx, &ast::proto proto, &arg[] args, &t ty, &controlflow cf,
         &(@constr)[] constrs) -> t {
    ret gen_ty(cx, ty_fn(proto, args, ty, cf, constrs));
}

fn mk_native_fn(&ctxt cx, &ast::native_abi abi, &arg[] args, &t ty) -> t {
    ret gen_ty(cx, ty_native_fn(abi, args, ty));
}

fn mk_obj(&ctxt cx, &method[] meths) -> t {
    ret gen_ty(cx, ty_obj(meths));
}

fn mk_res(&ctxt cx, &ast::def_id did, &t inner, &t[] tps) -> t {
    ret gen_ty(cx, ty_res(did, inner, tps));
}

fn mk_var(&ctxt cx, int v) -> t { ret gen_ty(cx, ty_var(v)); }

fn mk_param(&ctxt cx, uint n) -> t { ret gen_ty(cx, ty_param(n)); }

fn mk_type(&ctxt cx) -> t { ret idx_type; }

fn mk_native(&ctxt cx, &def_id did) -> t { ret gen_ty(cx, ty_native(did)); }


// Returns the one-level-deep type structure of the given type.
fn struct(&ctxt cx, &t typ) -> sty {
    ret interner::get(*cx.ts, typ).struct;
}


// Returns the canonical name of the given type.
fn cname(&ctxt cx, &t typ) -> option::t[str] {
    ret interner::get(*cx.ts, typ).cname;
}


// Type folds
type ty_walk = fn(t) ;

fn walk_ty(&ctxt cx, ty_walk walker, t ty) {
    alt (struct(cx, ty)) {
        case (ty_nil) {/* no-op */ }
        case (ty_bot) {/* no-op */ }
        case (ty_bool) {/* no-op */ }
        case (ty_int) {/* no-op */ }
        case (ty_uint) {/* no-op */ }
        case (ty_float) {/* no-op */ }
        case (ty_machine(_)) {/* no-op */ }
        case (ty_char) {/* no-op */ }
        case (ty_str) {/* no-op */ }
        case (ty_istr) {/* no-op */ }
        case (ty_type) {/* no-op */ }
        case (ty_native(_)) {/* no-op */ }
        case (ty_box(?tm)) { walk_ty(cx, walker, tm.ty); }
        case (ty_vec(?tm)) { walk_ty(cx, walker, tm.ty); }
        case (ty_ivec(?tm)) { walk_ty(cx, walker, tm.ty); }
        case (ty_ptr(?tm)) { walk_ty(cx, walker, tm.ty); }
        case (ty_port(?subty)) { walk_ty(cx, walker, subty); }
        case (ty_chan(?subty)) { walk_ty(cx, walker, subty); }
        case (ty_tag(?tid, ?subtys)) {
            for (t subty in subtys) { walk_ty(cx, walker, subty); }
        }
        case (ty_tup(?mts)) {
            for (mt tm in mts) { walk_ty(cx, walker, tm.ty); }
        }
        case (ty_rec(?fields)) {
            for (field fl in fields) { walk_ty(cx, walker, fl.mt.ty); }
        }
        case (ty_fn(?proto, ?args, ?ret_ty, _, _)) {
            for (arg a in args) { walk_ty(cx, walker, a.ty); }
            walk_ty(cx, walker, ret_ty);
        }
        case (ty_native_fn(?abi, ?args, ?ret_ty)) {
            for (arg a in args) { walk_ty(cx, walker, a.ty); }
            walk_ty(cx, walker, ret_ty);
        }
        case (ty_obj(?methods)) {
            for (method m in methods) {
                for (arg a in m.inputs) { walk_ty(cx, walker, a.ty); }
                walk_ty(cx, walker, m.output);
            }
        }
        case (ty_res(_, ?sub, ?tps)) {
            walk_ty(cx, walker, sub);
            for (t tp in tps) { walk_ty(cx, walker, tp); }
        }
        case (ty_var(_)) {/* no-op */ }
        case (ty_param(_)) {/* no-op */ }
    }
    walker(ty);
}

tag fold_mode {
    fm_var(fn(int) -> t );
    fm_param(fn(uint) -> t );
    fm_general(fn(t) -> t );
}

fn fold_ty(&ctxt cx, fold_mode fld, t ty_0) -> t {
    auto ty = ty_0;
    // Fast paths.

    alt (fld) {
        case (fm_var(_)) { if (!type_contains_vars(cx, ty)) { ret ty; } }
        case (fm_param(_)) { if (!type_contains_params(cx, ty)) { ret ty; } }
        case (fm_general(_)) {/* no fast path */ }
    }
    alt (struct(cx, ty)) {
        case (ty_nil) {/* no-op */ }
        case (ty_bot) {/* no-op */ }
        case (ty_bool) {/* no-op */ }
        case (ty_int) {/* no-op */ }
        case (ty_uint) {/* no-op */ }
        case (ty_float) {/* no-op */ }
        case (ty_machine(_)) {/* no-op */ }
        case (ty_char) {/* no-op */ }
        case (ty_str) {/* no-op */ }
        case (ty_istr) {/* no-op */ }
        case (ty_type) {/* no-op */ }
        case (ty_native(_)) {/* no-op */ }
        case (ty_task) {/* no-op */ }
        case (ty_box(?tm)) {
            ty =
                copy_cname(cx,
                           mk_box(cx,
                                  rec(ty=fold_ty(cx, fld, tm.ty),
                                      mut=tm.mut)), ty);
        }
        case (ty_ptr(?tm)) {
            ty =
                copy_cname(cx,
                           mk_ptr(cx,
                                  rec(ty=fold_ty(cx, fld, tm.ty),
                                      mut=tm.mut)), ty);
        }
        case (ty_vec(?tm)) {
            ty =
                copy_cname(cx,
                           mk_vec(cx,
                                  rec(ty=fold_ty(cx, fld, tm.ty),
                                      mut=tm.mut)), ty);
        }
        case (ty_ivec(?tm)) {
            ty =
                copy_cname(cx,
                           mk_ivec(cx,
                                   rec(ty=fold_ty(cx, fld, tm.ty),
                                       mut=tm.mut)), ty);
        }
        case (ty_port(?subty)) {
            ty = copy_cname(cx, mk_port(cx, fold_ty(cx, fld, subty)), ty);
        }
        case (ty_chan(?subty)) {
            ty = copy_cname(cx, mk_chan(cx, fold_ty(cx, fld, subty)), ty);
        }
        case (ty_tag(?tid, ?subtys)) {
            let t[] new_subtys = ~[];
            for (t subty in subtys) {
                new_subtys += ~[fold_ty(cx, fld, subty)];
            }
            ty = copy_cname(cx, mk_tag(cx, tid, new_subtys), ty);
        }
        case (ty_tup(?mts)) {
            let mt[] new_mts = ~[];
            for (mt tm in mts) {
                auto new_subty = fold_ty(cx, fld, tm.ty);
                new_mts += ~[rec(ty=new_subty, mut=tm.mut)];
            }
            ty = copy_cname(cx, mk_tup(cx, new_mts), ty);
        }
        case (ty_rec(?fields)) {
            let field[] new_fields = ~[];
            for (field fl in fields) {
                auto new_ty = fold_ty(cx, fld, fl.mt.ty);
                auto new_mt = rec(ty=new_ty, mut=fl.mt.mut);
                new_fields += ~[rec(ident=fl.ident, mt=new_mt)];
            }
            ty = copy_cname(cx, mk_rec(cx, new_fields), ty);
        }
        case (ty_fn(?proto, ?args, ?ret_ty, ?cf, ?constrs)) {
            let arg[] new_args = ~[];
            for (arg a in args) {
                auto new_ty = fold_ty(cx, fld, a.ty);
                new_args += ~[rec(mode=a.mode, ty=new_ty)];
            }
            ty =
                copy_cname(cx,
                           mk_fn(cx, proto, new_args,
                                 fold_ty(cx, fld, ret_ty), cf, constrs), ty);
        }
        case (ty_native_fn(?abi, ?args, ?ret_ty)) {
            let arg[] new_args = ~[];
            for (arg a in args) {
                auto new_ty = fold_ty(cx, fld, a.ty);
                new_args += ~[rec(mode=a.mode, ty=new_ty)];
            }
            ty =
                copy_cname(cx,
                           mk_native_fn(cx, abi, new_args,
                                        fold_ty(cx, fld, ret_ty)), ty);
        }
        case (ty_obj(?methods)) {
            let method[] new_methods = ~[];
            for (method m in methods) {
                let arg[] new_args = ~[];
                for (arg a in m.inputs) {
                    new_args += ~[rec(mode=a.mode,
                                      ty=fold_ty(cx, fld, a.ty))];
                }
                new_methods +=
                    ~[rec(proto=m.proto,
                          ident=m.ident,
                          inputs=new_args,
                          output=fold_ty(cx, fld, m.output),
                          cf=m.cf,
                          constrs=m.constrs)];
            }
            ty = copy_cname(cx, mk_obj(cx, new_methods), ty);
        }
        case (ty_res(?did, ?subty, ?tps)) {
            auto new_tps = ~[];
            for (t tp in tps) { new_tps += ~[fold_ty(cx, fld, tp)]; }
            ty = copy_cname(cx, mk_res(cx, did, fold_ty(cx, fld, subty),
                                       new_tps), ty);
        }
        case (ty_var(?id)) {
            alt (fld) {
                case (fm_var(?folder)) { ty = folder(id); }
                case (_) {/* no-op */ }
            }
        }
        case (ty_param(?id)) {
            alt (fld) {
                case (fm_param(?folder)) { ty = folder(id); }
                case (_) {/* no-op */ }
            }
        }
    }

    // If this is a general type fold, then we need to run it now.
    alt (fld) {
        case (fm_general(?folder)) { ret folder(ty); }
        case (_) { ret ty; }
    }
}


// Type utilities

fn rename(&ctxt cx, t typ, str new_cname) -> t {
    ret gen_ty_full(cx, struct(cx, typ), some(new_cname));
}

fn strip_cname(&ctxt cx, t typ) -> t {
    ret gen_ty_full(cx, struct(cx, typ), none);
}

// Returns a type with the structural part taken from `struct_ty` and the
// canonical name from `cname_ty`.
fn copy_cname(&ctxt cx, t struct_ty, t cname_ty) -> t {
    ret gen_ty_full(cx, struct(cx, struct_ty), cname(cx, cname_ty));
}

fn type_is_nil(&ctxt cx, &t ty) -> bool {
    alt (struct(cx, ty)) {
        case (ty_nil) { ret true; }
        case (_) { ret false; }
    }
}

fn type_is_bot(&ctxt cx, &t ty) -> bool {
    alt (struct(cx, ty)) {
        case (ty_bot) { ret true; }
        case (_) { ret false; }
    }
}

fn type_is_bool(&ctxt cx, &t ty) -> bool {
    alt (struct(cx, ty)) {
        case (ty_bool) { ret true; }
        case (_) { ret false; }
    }
}

fn type_is_chan(&ctxt cx, &t ty) -> bool {
    alt (struct(cx, ty)) {
        case (ty_chan(_)) { ret true; }
        case (_) { ret false; }
    }
}

fn type_is_structural(&ctxt cx, &t ty) -> bool {
    alt (struct(cx, ty)) {
        case (ty_tup(_)) { ret true; }
        case (ty_rec(_)) { ret true; }
        case (ty_tag(_, _)) { ret true; }
        case (ty_fn(_, _, _, _, _)) { ret true; }
        case (ty_obj(_)) { ret true; }
        case (ty_res(_, _, _)) { ret true; }
        case (ty_ivec(_)) { ret true; }
        case (ty_istr) { ret true; }
        case (_) { ret false; }
    }
}

fn type_is_copyable(&ctxt cx, &t ty) -> bool {
    ret alt (struct(cx, ty)) {
        case (ty_res(_, _, _)) { false }
        case (_) { true }
    };
}

fn type_is_sequence(&ctxt cx, &t ty) -> bool {
    alt (struct(cx, ty)) {
        case (ty_str) { ret true; }
        case (ty_istr) { ret true; }
        case (ty_vec(_)) { ret true; }
        case (ty_ivec(_)) { ret true; }
        case (_) { ret false; }
    }
}

fn type_is_str(&ctxt cx, &t ty) -> bool {
    alt (struct(cx, ty)) {
        case (ty_str) { ret true; }
        case (ty_istr) { ret true; }
        case (_) { ret false; }
    }
}

fn sequence_is_interior(&ctxt cx, &t ty) -> bool {
    alt (struct(cx, ty)) {
        case (
             // TODO: Or-patterns
             ty::ty_vec(_)) {
            ret false;
        }
        case (ty::ty_str) { ret false; }
        case (ty::ty_ivec(_)) { ret true; }
        case (ty::ty_istr) { ret true; }
        case (_) {
            cx.sess.bug("sequence_is_interior called on non-sequence type");
        }
    }
}

fn sequence_element_type(&ctxt cx, &t ty) -> t {
    alt (struct(cx, ty)) {
        case (ty_str) { ret mk_mach(cx, ast::ty_u8); }
        case (ty_istr) { ret mk_mach(cx, ast::ty_u8); }
        case (ty_vec(?mt)) { ret mt.ty; }
        case (ty_ivec(?mt)) { ret mt.ty; }
        case (_) {
            cx.sess.bug("sequence_element_type called on non-sequence value");
        }
    }
}

fn type_is_tup_like(&ctxt cx, &t ty) -> bool {
    alt (struct(cx, ty)) {
        case (ty_box(_)) { ret true; }
        case (ty_tup(_)) { ret true; }
        case (ty_rec(_)) { ret true; }
        case (ty_tag(_, _)) { ret true; }
        case (_) { ret false; }
    }
}

fn get_element_type(&ctxt cx, &t ty, uint i) -> t {
    alt (struct(cx, ty)) {
        case (ty_tup(?mts)) { ret mts.(i).ty; }
        case (ty_rec(?flds)) { ret flds.(i).mt.ty; }
        case (_)             {
            cx.sess.bug("get_element_type called on type "
                        + ty_to_str(cx, ty) + " - expected a \
            tuple or record");
        }
    }
    // NB: This is not exhaustive -- struct(cx, ty) could be a box or a
    // tag.
}

fn type_is_box(&ctxt cx, &t ty) -> bool {
    alt (struct(cx, ty)) {
        case (ty_box(_)) { ret true; }
        case (_) { ret false; }
    }
}

fn type_is_boxed(&ctxt cx, &t ty) -> bool {
    alt (struct(cx, ty)) {
        case (ty_str) { ret true; }
        case (ty_vec(_)) { ret true; }
        case (ty_box(_)) { ret true; }
        case (ty_port(_)) { ret true; }
        case (ty_chan(_)) { ret true; }
        case (ty_task) { ret true; }
        case (_) { ret false; }
    }
}

fn type_is_scalar(&ctxt cx, &t ty) -> bool {
    alt (struct(cx, ty)) {
        case (ty_nil) { ret true; }
        case (ty_bool) { ret true; }
        case (ty_int) { ret true; }
        case (ty_float) { ret true; }
        case (ty_uint) { ret true; }
        case (ty_machine(_)) { ret true; }
        case (ty_char) { ret true; }
        case (ty_type) { ret true; }
        case (ty_native(_)) { ret true; }
        case (ty_ptr(_)) { ret true; }
        case (_) { ret false; }
    }
}

fn type_has_pointers(&ctxt cx, &t ty) -> bool {
    alt (cx.has_pointer_cache.find(ty)) {
        case (some(?result)) { ret result; }
        case (none) { /* fall through */ }
    }

    auto result = false;
    alt (struct(cx, ty)) {
        // scalar types
        case (ty_nil) { /* no-op */ }
        case (ty_bot) { /* no-op */ }
        case (ty_bool) { /* no-op */ }
        case (ty_int) { /* no-op */ }
        case (ty_float) { /* no-op */ }
        case (ty_uint) { /* no-op */ }
        case (ty_machine(_)) { /* no-op */ }
        case (ty_char) { /* no-op */ }
        case (ty_type) { /* no-op */ }
        case (ty_native(_)) { /* no-op */ }
        case (ty_tup(?elts)) {
            for (mt m in elts) {
                if (type_has_pointers(cx, m.ty)) { result = true; }
            }
        }
        case (ty_rec(?flds)) {
            for (field f in flds) {
                if (type_has_pointers(cx, f.mt.ty)) { result = true; }
            }
        }
        case (ty_tag(?did, ?tps)) {
            auto variants = tag_variants(cx, did);
            for (variant_info variant in variants) {
                auto tup_ty = mk_imm_tup(cx, variant.args);

                // Perform any type parameter substitutions.
                tup_ty = substitute_type_params(cx, tps, tup_ty);
                if (type_has_pointers(cx, tup_ty)) { result = true; }
            }
        }
        case (ty_res(?did, ?inner, ?tps)) {
            result = type_has_pointers(cx,
                substitute_type_params(cx, tps, inner));
        }
        case (_) { result = true; }
    }

    cx.has_pointer_cache.insert(ty, result);
    ret result;
}


// FIXME: should we just return true for native types in
// type_is_scalar?
fn type_is_native(&ctxt cx, &t ty) -> bool {
    alt (struct(cx, ty)) {
        case (ty_native(_)) { ret true; }
        case (_) { ret false; }
    }
}

fn type_has_dynamic_size(&ctxt cx, &t ty) -> bool {
    alt (struct(cx, ty)) {
        case (ty_nil) { ret false; }
        case (ty_bot) { ret false; }
        case (ty_bool) { ret false; }
        case (ty_int) { ret false; }
        case (ty_float) { ret false; }
        case (ty_uint) { ret false; }
        case (ty_machine(_)) { ret false; }
        case (ty_char) { ret false; }
        case (ty_str) { ret false; }
        case (ty_istr) { ret false; }
        case (ty_tag(_, ?subtys)) {
            auto i = 0u;
            while (i < ivec::len[t](subtys)) {
                if (type_has_dynamic_size(cx, subtys.(i))) { ret true; }
                i += 1u;
            }
            ret false;
        }
        case (ty_box(_)) { ret false; }
        case (ty_vec(_)) { ret false; }
        case (ty_ivec(?mt)) { ret type_has_dynamic_size(cx, mt.ty); }
        case (ty_ptr(_)) { ret false; }
        case (ty_port(_)) { ret false; }
        case (ty_chan(_)) { ret false; }
        case (ty_task) { ret false; }
        case (ty_tup(?mts)) {
            auto i = 0u;
            while (i < ivec::len[mt](mts)) {
                if (type_has_dynamic_size(cx, mts.(i).ty)) { ret true; }
                i += 1u;
            }
            ret false;
        }
        case (ty_rec(?fields)) {
            auto i = 0u;
            while (i < ivec::len[field](fields)) {
                if (type_has_dynamic_size(cx, fields.(i).mt.ty)) { ret true; }
                i += 1u;
            }
            ret false;
        }
        case (ty_fn(_,_,_,_,_)) { ret false; }
        case (ty_native_fn(_,_,_)) { ret false; }
        case (ty_obj(_)) { ret false; }
        case (ty_res(_, ?sub, ?tps)) {
            for (t tp in tps) {
                if (type_has_dynamic_size(cx, tp)) { ret true; }
            }
            ret type_has_dynamic_size(cx, sub);
        }
        case (ty_var(_)) { fail "ty_var in type_has_dynamic_size()"; }
        case (ty_param(_)) { ret true; }
        case (ty_type) { ret false; }
        case (ty_native(_)) { ret false; }
    }
}

fn type_is_integral(&ctxt cx, &t ty) -> bool {
    alt (struct(cx, ty)) {
        case (ty_int) { ret true; }
        case (ty_uint) { ret true; }
        case (ty_machine(?m)) {
            alt (m) {
                case (ast::ty_i8) { ret true; }
                case (ast::ty_i16) { ret true; }
                case (ast::ty_i32) { ret true; }
                case (ast::ty_i64) { ret true; }
                case (ast::ty_u8) { ret true; }
                case (ast::ty_u16) { ret true; }
                case (ast::ty_u32) { ret true; }
                case (ast::ty_u64) { ret true; }
                case (_) { ret false; }
            }
        }
        case (ty_char) { ret true; }
        case (ty_bool) { ret true; }
        case (_) { ret false; }
    }
}

fn type_is_fp(&ctxt cx, &t ty) -> bool {
    alt (struct(cx, ty)) {
        case (ty_machine(?tm)) {
            alt (tm) {
                case (ast::ty_f32) { ret true; }
                case (ast::ty_f64) { ret true; }
                case (_) { ret false; }
            }
        }
        case (ty_float) { ret true; }
        case (_) { ret false; }
    }
}

fn type_is_signed(&ctxt cx, &t ty) -> bool {
    alt (struct(cx, ty)) {
        case (ty_int) { ret true; }
        case (ty_machine(?tm)) {
            alt (tm) {
                case (ast::ty_i8) { ret true; }
                case (ast::ty_i16) { ret true; }
                case (ast::ty_i32) { ret true; }
                case (ast::ty_i64) { ret true; }
                case (_) { ret false; }
            }
        }
        case (_) { ret false; }
    }
}

fn type_owns_heap_mem(&ctxt cx, &t ty) -> bool {
    alt (cx.owns_heap_mem_cache.find(ty)) {
        case (some(?result)) { ret result; }
        case (none) { /* fall through */ }
    }

    auto result = false;
    alt (struct(cx, ty)) {
        case (ty_ivec(_)) { result = true; }
        case (ty_istr) { result = true; }

        // scalar types
        case (ty_nil) { result = false; }
        case (ty_bot) { result = false; }
        case (ty_bool) { result = false; }
        case (ty_int) { result = false; }
        case (ty_float) { result = false; }
        case (ty_uint) { result = false; }
        case (ty_machine(_)) { result = false; }
        case (ty_char) { result = false; }
        case (ty_type) { result = false; }
        case (ty_native(_)) { result = false; }

        // boxed types
        case (ty_str) { result = false; }
        case (ty_box(_)) { result = false; }
        case (ty_vec(_)) { result = false; }
        case (ty_fn(_,_,_,_,_)) { result = false; }
        case (ty_native_fn(_,_,_)) { result = false; }
        case (ty_obj(_)) { result = false; }

        // structural types
        case (ty_tag(?did, ?tps)) {
            auto variants = tag_variants(cx, did);
            for (variant_info variant in variants) {
                auto tup_ty = mk_imm_tup(cx, variant.args);

                // Perform any type parameter substitutions.
                tup_ty = substitute_type_params(cx, tps, tup_ty);
                if (type_owns_heap_mem(cx, tup_ty)) { result = true; }
            }
        }
        case (ty_tup(?elts)) {
            for (mt m in elts) {
                if (type_owns_heap_mem(cx, m.ty)) { result = true; }
            }
        }
        case (ty_rec(?flds)) {
            for (field f in flds) {
                if (type_owns_heap_mem(cx, f.mt.ty)) { result = true; }
            }
        }
        case (ty_res(_, ?inner, ?tps)) {
            result = type_owns_heap_mem(cx,
                substitute_type_params(cx, tps, inner));
        }

        case (ty_ptr(_)) { result = false; }
        case (ty_port(_)) { result = false; }
        case (ty_chan(_)) { result = false; }
        case (ty_task) { result = false; }
        case (ty_var(_)) { fail "ty_var in type_owns_heap_mem"; }
        case (ty_param(_)) { result = false; }
    }

    cx.owns_heap_mem_cache.insert(ty, result);
    ret result;
}

fn type_param(&ctxt cx, &t ty) -> option::t[uint] {
    alt (struct(cx, ty)) {
        case (ty_param(?id)) { ret some(id); }
        case (_) {/* fall through */ }
    }
    ret none;
}

fn type_autoderef(&ctxt cx, &ty::t t) -> ty::t {
    let ty::t t1 = t;
    while (true) {
        alt (struct(cx, t1)) {
            case (ty::ty_box(?mt)) { t1 = mt.ty; }
            case (ty::ty_res(_, ?inner, ?tps)) {
                t1 = substitute_type_params(cx, tps, inner);
            }
            case (ty::ty_tag(?did, ?tps)) {
                auto variants = tag_variants(cx, did);
                if (ivec::len(variants) != 1u ||
                        ivec::len(variants.(0).args) != 1u) {
                    break;
                }
                t1 = substitute_type_params(cx, tps, variants.(0).args.(0));
            }
            case (_) { break; }
        }
    }
    ret t1;
}

// Type hashing. This function is private to this module (and slow); external
// users should use `hash_ty()` instead.
fn hash_type_structure(&sty st) -> uint {
    fn hash_uint(uint id, uint n) -> uint {
        auto h = id;
        h += h << 5u + n;
        ret h;
    }
    fn hash_def(uint id, ast::def_id did) -> uint {
        auto h = id;
        h += h << 5u + (did._0 as uint);
        h += h << 5u + (did._1 as uint);
        ret h;
    }
    fn hash_subty(uint id, &t subty) -> uint {
        auto h = id;
        h += h << 5u + hash_ty(subty);
        ret h;
    }
    fn hash_type_constr(uint id, &@type_constr c)
        -> uint {
        auto h = id;
        h += h << 5u + hash_def(h, c.node.id);
        ret hash_type_constr_args(h, c.node.args);
    }
    fn hash_type_constr_args(uint id, (@ty_constr_arg)[] args) -> uint {
        auto h = id;
        for (@ty_constr_arg a in args) {
            alt (a.node) {
                case (carg_base) {
                    h += h << 5u;
                }
                case (carg_lit(_)) {
                    // FIXME
                    fail "lit args not implemented yet";
                }
                case (carg_ident(?p)) {
                    // FIXME: Not sure what to do here.
                    h += h << 5u;
                }
            }
        }
        ret h;
    }


    fn hash_fn(uint id, &arg[] args, &t rty) -> uint {
        auto h = id;
        for (arg a in args) { h += h << 5u + hash_ty(a.ty); }
        h += h << 5u + hash_ty(rty);
        ret h;
    }
    alt (st) {
        case (ty_nil) { ret 0u; }
        case (ty_bool) { ret 1u; }
        case (ty_int) { ret 2u; }
        case (ty_float) { ret 3u; }
        case (ty_uint) { ret 4u; }
        case (ty_machine(?tm)) {
            alt (tm) {
                case (ast::ty_i8) { ret 5u; }
                case (ast::ty_i16) { ret 6u; }
                case (ast::ty_i32) { ret 7u; }
                case (ast::ty_i64) { ret 8u; }
                case (ast::ty_u8) { ret 9u; }
                case (ast::ty_u16) { ret 10u; }
                case (ast::ty_u32) { ret 11u; }
                case (ast::ty_u64) { ret 12u; }
                case (ast::ty_f32) { ret 13u; }
                case (ast::ty_f64) { ret 14u; }
            }
        }
        case (ty_char) { ret 15u; }
        case (ty_str) { ret 16u; }
        case (ty_istr) { ret 17u; }
        case (ty_tag(?did, ?tys)) {
            auto h = hash_def(18u, did);
            for (t typ in tys) { h += h << 5u + hash_ty(typ); }
            ret h;
        }
        case (ty_box(?mt)) { ret hash_subty(19u, mt.ty); }
        case (ty_vec(?mt)) { ret hash_subty(20u, mt.ty); }
        case (ty_ivec(?mt)) { ret hash_subty(21u, mt.ty); }
        case (ty_port(?typ)) { ret hash_subty(22u, typ); }
        case (ty_chan(?typ)) { ret hash_subty(23u, typ); }
        case (ty_task) { ret 24u; }
        case (ty_tup(?mts)) {
            auto h = 25u;
            for (mt tm in mts) { h += h << 5u + hash_ty(tm.ty); }
            ret h;
        }
        case (ty_rec(?fields)) {
            auto h = 26u;
            for (field f in fields) { h += h << 5u + hash_ty(f.mt.ty); }
            ret h;
        }
        // ???
        case (ty_fn(_, ?args, ?rty, _, _)) {
            ret hash_fn(27u, args, rty);
        }
        case (ty_native_fn(_, ?args, ?rty)) { ret hash_fn(28u, args, rty); }
        case (ty_obj(?methods)) {
            auto h = 29u;
            for (method m in methods) { h += h << 5u + str::hash(m.ident); }
            ret h;
        }
        case (ty_var(?v)) { ret hash_uint(30u, v as uint); }
        case (ty_param(?pid)) { ret hash_uint(31u, pid); }
        case (ty_type) { ret 32u; }
        case (ty_native(?did)) { ret hash_def(33u, did); }
        case (ty_bot) { ret 34u; }
        case (ty_ptr(?mt)) { ret hash_subty(35u, mt.ty); }
        case (ty_res(?did, ?sub, ?tps)) {
            auto h = hash_subty(hash_def(18u, did), sub);
            for (t tp in tps) { h += h << 5u + hash_ty(tp); }
            ret h;
        }
        case (ty_constr(?t, ?cs)) {
            auto h = 36u;
            for (@type_constr c in cs) {
                h += h << 5u + hash_type_constr(h, c);
            }
            ret h;
        }
    }
}

fn hash_type_info(&sty st, &option::t[str] cname_opt) -> uint {
    auto h = hash_type_structure(st);
    alt (cname_opt) {
        case (none) {/* no-op */ }
        case (some(?s)) { h += h << 5u + str::hash(s); }
    }
    ret h;
}

fn hash_raw_ty(&@raw_t rt) -> uint { ret rt.hash; }

fn hash_ty(&t typ) -> uint { ret typ; }


// Type equality. This function is private to this module (and slow); external
// users should use `eq_ty()` instead.
fn eq_int(&uint x, &uint y) -> bool { ret x == y; }

fn arg_eq[T](&fn(&T, &T) -> bool eq, @sp_constr_arg[T] a,
             @sp_constr_arg[T] b) -> bool {
    alt (a.node) {
        case (ast::carg_base) {
            alt (b.node) {
                case (ast::carg_base) { ret true; }
                case (_) { ret false; }
            }
        }
        case (ast::carg_ident(?s)) {
            alt (b.node) {
                case (ast::carg_ident(?t)) { ret eq(s, t); }
                case (_) { ret false; }
            }
        }
        case (ast::carg_lit(?l)) {
            alt (b.node) {
                case (ast::carg_lit(?m)) { ret lit_eq(l, m); }
                case (_) { ret false; }
            }
        }
    }
}

fn args_eq[T](fn(&T, &T) -> bool eq,
              &(@sp_constr_arg[T])[] a, &(@sp_constr_arg[T])[] b) -> bool {
    let uint i = 0u;
    for (@sp_constr_arg[T] arg in a) {
        if (!arg_eq(eq, arg, b.(i))) { ret false; }
        i += 1u;
    }
    ret true;
}

fn constr_eq(&@constr c, &@constr d) -> bool {
    ret path_to_str(c.node.path) == path_to_str(d.node.path) &&
            // FIXME: hack
            args_eq(eq_int, c.node.args, d.node.args);
}

fn constrs_eq(&(@constr)[] cs, &(@constr)[] ds) -> bool {
    if (ivec::len(cs) != ivec::len(ds)) { ret false; }
    auto i = 0u;
    for (@constr c in cs) {
        if (!constr_eq(c, ds.(i))) { ret false; }
        i += 1u;
    }
    ret true;
}

fn equal_type_structures(&sty a, &sty b) -> bool {
    fn equal_mt(&mt a, &mt b) -> bool {
        ret a.mut == b.mut && eq_ty(a.ty, b.ty);
    }
    fn equal_fn(&arg[] args_a, &t rty_a, &arg[] args_b, &t rty_b) ->
       bool {
        if (!eq_ty(rty_a, rty_b)) { ret false; }
        auto len = ivec::len[arg](args_a);
        if (len != ivec::len[arg](args_b)) { ret false; }
        auto i = 0u;
        while (i < len) {
            auto arg_a = args_a.(i);
            auto arg_b = args_b.(i);
            if (arg_a.mode != arg_b.mode) { ret false; }
            if (!eq_ty(arg_a.ty, arg_b.ty)) { ret false; }
            i += 1u;
        }
        ret true;
    }
    fn equal_def(&ast::def_id did_a, &ast::def_id did_b) -> bool {
        ret did_a._0 == did_b._0 && did_a._1 == did_b._1;
    }
    alt (a) {
        case (ty_nil) {
            alt (b) { case (ty_nil) { ret true; } case (_) { ret false; } }
        }
        case (ty_bot) {
            alt (b) { case (ty_bot) { ret true; } case (_) { ret false; } }
        }
        case (ty_bool) {
            alt (b) { case (ty_bool) { ret true; } case (_) { ret false; } }
        }
        case (ty_int) {
            alt (b) { case (ty_int) { ret true; } case (_) { ret false; } }
        }
        case (ty_float) {
            alt (b) { case (ty_float) { ret true; } case (_) { ret false; } }
        }
        case (ty_uint) {
            alt (b) { case (ty_uint) { ret true; } case (_) { ret false; } }
        }
        case (ty_machine(?tm_a)) {
            alt (b) {
                case (ty_machine(?tm_b)) {
                    ret hash_type_structure(a) == hash_type_structure(b);
                }
                case (_) { ret false; }
            }
        }
        case (ty_char) {
            alt (b) { case (ty_char) { ret true; } case (_) { ret false; } }
        }
        case (ty_str) {
            alt (b) { case (ty_str) { ret true; } case (_) { ret false; } }
        }
        case (ty_istr) {
            alt (b) { case (ty_istr) { ret true; } case (_) { ret false; } }
        }
        case (ty_tag(?id_a, ?tys_a)) {
            alt (b) {
                case (ty_tag(?id_b, ?tys_b)) {
                    if (!equal_def(id_a, id_b)) { ret false; }
                    auto len = ivec::len[t](tys_a);
                    if (len != ivec::len[t](tys_b)) { ret false; }
                    auto i = 0u;
                    while (i < len) {
                        if (!eq_ty(tys_a.(i), tys_b.(i))) { ret false; }
                        i += 1u;
                    }
                    ret true;
                }
                case (_) { ret false; }
            }
        }
        case (ty_box(?mt_a)) {
            alt (b) {
                case (ty_box(?mt_b)) { ret equal_mt(mt_a, mt_b); }
                case (_) { ret false; }
            }
        }
        case (ty_vec(?mt_a)) {
            alt (b) {
                case (ty_vec(?mt_b)) { ret equal_mt(mt_a, mt_b); }
                case (_) { ret false; }
            }
        }
        case (ty_ivec(?mt_a)) {
            alt (b) {
                case (ty_ivec(?mt_b)) { ret equal_mt(mt_a, mt_b); }
                case (_) { ret false; }
            }
        }
        case (ty_ptr(?mt_a)) {
            alt (b) {
                case (ty_ptr(?mt_b)) { ret equal_mt(mt_a, mt_b); }
                case (_) { ret false; }
            }
        }
        case (ty_port(?t_a)) {
            alt (b) {
                case (ty_port(?t_b)) { ret eq_ty(t_a, t_b); }
                case (_) { ret false; }
            }
        }
        case (ty_chan(?t_a)) {
            alt (b) {
                case (ty_chan(?t_b)) { ret eq_ty(t_a, t_b); }
                case (_) { ret false; }
            }
        }
        case (ty_task) {
            alt (b) { case (ty_task) { ret true; } case (_) { ret false; } }
        }
        case (ty_tup(?mts_a)) {
            alt (b) {
                case (ty_tup(?mts_b)) {
                    auto len = ivec::len[mt](mts_a);
                    if (len != ivec::len[mt](mts_b)) { ret false; }
                    auto i = 0u;
                    while (i < len) {
                        if (!equal_mt(mts_a.(i), mts_b.(i))) { ret false; }
                        i += 1u;
                    }
                    ret true;
                }
                case (_) { ret false; }
            }
        }
        case (ty_rec(?flds_a)) {
            alt (b) {
                case (ty_rec(?flds_b)) {
                    auto len = ivec::len[field](flds_a);
                    if (len != ivec::len[field](flds_b)) { ret false; }
                    auto i = 0u;
                    while (i < len) {
                        auto fld_a = flds_a.(i);
                        auto fld_b = flds_b.(i);
                        if (!str::eq(fld_a.ident, fld_b.ident) ||
                                !equal_mt(fld_a.mt, fld_b.mt)) {
                            ret false;
                        }
                        i += 1u;
                    }
                    ret true;
                }
                case (_) { ret false; }
            }
        }
        case (ty_fn(?p_a, ?args_a, ?rty_a, ?cf_a, ?constrs_a)) {
            alt (b) {
                case (ty_fn(?p_b, ?args_b, ?rty_b, ?cf_b, ?constrs_b)) {
                    ret p_a == p_b && cf_a == cf_b &&
                            constrs_eq(constrs_a, constrs_b) &&
                            equal_fn(args_a, rty_a, args_b, rty_b);
                }
                case (_) { ret false; }
            }
        }
        case (ty_native_fn(?abi_a, ?args_a, ?rty_a)) {
            alt (b) {
                case (ty_native_fn(?abi_b, ?args_b, ?rty_b)) {
                    ret abi_a == abi_b &&
                            equal_fn(args_a, rty_a, args_b, rty_b);
                }
                case (_) { ret false; }
            }
        }
        case (ty_obj(?methods_a)) {
            alt (b) {
                case (ty_obj(?methods_b)) {
                    auto len = ivec::len[method](methods_a);
                    if (len != ivec::len[method](methods_b)) { ret false; }
                    auto i = 0u;
                    while (i < len) {
                        auto m_a = methods_a.(i);
                        auto m_b = methods_b.(i);
                        if (m_a.proto != m_b.proto ||
                                !str::eq(m_a.ident, m_b.ident) ||
                                !equal_fn(m_a.inputs, m_a.output, m_b.inputs,
                                          m_b.output)) {
                            ret false;
                        }
                        i += 1u;
                    }
                    ret true;
                }
                case (_) { ret false; }
            }
        }
        case (ty_res(?id_a, ?inner_a, ?tps_a)) {
            alt (b) {
                case (ty_res(?id_b, ?inner_b, ?tps_b)) {
                    if (!equal_def(id_a, id_b) || !eq_ty(inner_a, inner_b)) {
                        ret false;
                    }
                    auto i = 0u;
                    for (t tp_a in tps_a) {
                        if (!eq_ty(tp_a, tps_b.(i))) { ret false; }
                        i += 1u;
                    }
                    ret true;
                }
                case (_) { ret false; }
            }
        }
        case (ty_var(?v_a)) {
            alt (b) {
                case (ty_var(?v_b)) { ret v_a == v_b; }
                case (_) { ret false; }
            }
        }
        case (ty_param(?pid_a)) {
            alt (b) {
                case (ty_param(?pid_b)) { ret pid_a == pid_b; }
                case (_) { ret false; }
            }
        }
        case (ty_type) {
            alt (b) { case (ty_type) { ret true; } case (_) { ret false; } }
        }
        case (ty_native(?a_id)) {
            alt (b) {
                case (ty_native(?b_id)) {
                    ret a_id._0 == b_id._0 && a_id._1 == b_id._1;
                }
                case (_) { ret false; } }
        }
    }
}


// An expensive type equality function. This function is private to this
// module.
//
// FIXME: Use structural comparison, but this loops forever and segfaults.
fn eq_raw_ty(&@raw_t a, &@raw_t b) -> bool {
    // Check hashes (fast path).

    if (a.hash != b.hash) { ret false; }
    // Check canonical names.

    alt (a.cname) {
        case (none) {
            alt (b.cname) {
                case (none) { /* ok */ }
                case (_) { ret false; }
            }
        }
        case (some(?s_a)) {
            alt (b.cname) {
                case (some(?s_b)) { if (!str::eq(s_a, s_b)) { ret false; } }
                case (_) { ret false; }
            }
        }
    }
    // Check structures.

    ret equal_type_structures(a.struct, b.struct);
}


// This is the equality function the public should use. It works as long as
// the types are interned.
fn eq_ty(&t a, &t b) -> bool { ret a == b; }


// Type lookups
fn node_id_to_ty_param_substs_opt_and_ty(&ctxt cx, &ast::node_id id) ->
   ty_param_substs_opt_and_ty {

    // Pull out the node type table.
    alt (smallintmap::find(*cx.node_types, id as uint)) {
        case (none) {
            cx.sess.bug("node_id_to_ty_param_substs_opt_and_ty() called on " +
                       "an untyped node (" + std::int::to_str(id, 10u) + ")");
        }
        case (some(?tpot)) { ret tpot; }
    }
}

fn node_id_to_type(&ctxt cx, &ast::node_id id) -> t {
    ret node_id_to_ty_param_substs_opt_and_ty(cx, id)._1;
}

fn node_id_to_type_params(&ctxt cx, &ast::node_id id) -> t[] {
    alt (node_id_to_ty_param_substs_opt_and_ty(cx, id)._0) {
        case (none)       { ret ~[]; }
        case (some(?tps)) { ret tps; }
    }
}

fn node_id_has_type_params(&ctxt cx, &ast::node_id id) -> bool {
    auto tpt = node_id_to_ty_param_substs_opt_and_ty(cx, id);
    ret !option::is_none[t[]](tpt._0);
}


// Returns a type with type parameter substitutions performed if applicable.
fn ty_param_substs_opt_and_ty_to_monotype(&ctxt cx,
                                          &ty_param_substs_opt_and_ty tpot) ->
   t {
    alt (tpot._0) {
        case (none) { ret tpot._1; }
        case (some(?tps)) { ret substitute_type_params(cx, tps, tpot._1); }
    }
}


// Returns the type of an annotation, with type parameter substitutions
// performed if applicable.
fn node_id_to_monotype(&ctxt cx, ast::node_id id) -> t {
    auto tpot = node_id_to_ty_param_substs_opt_and_ty(cx, id);
    ret ty_param_substs_opt_and_ty_to_monotype(cx, tpot);
}


// Returns the number of distinct type parameters in the given type.
fn count_ty_params(&ctxt cx, t ty) -> uint {
    fn counter(&ctxt cx, @mutable (uint[]) param_indices, t ty) {
        alt (struct(cx, ty)) {
            case (ty_param(?param_idx)) {
                auto seen = false;
                for (uint other_param_idx in *param_indices) {
                    if (param_idx == other_param_idx) { seen = true; }
                }
                if (!seen) { *param_indices += ~[param_idx]; }
            }
            case (_) {/* fall through */ }
        }
    }
    let @mutable (uint[]) param_indices = @mutable ~[];
    auto f = bind counter(cx, param_indices, _);
    walk_ty(cx, f, ty);
    ret ivec::len[uint](*param_indices);
}

fn type_contains_vars(&ctxt cx, &t typ) -> bool {
    ret interner::get(*cx.ts, typ).has_vars;
}

fn type_contains_params(&ctxt cx, &t typ) -> bool {
    ret interner::get(*cx.ts, typ).has_params;
}


// Type accessors for substructures of types
fn ty_fn_args(&ctxt cx, &t fty) -> arg[] {
    alt (struct(cx, fty)) {
        case (ty::ty_fn(_, ?a, _, _, _)) { ret a; }
        case (ty::ty_native_fn(_, ?a, _)) { ret a; }
    }
    cx.sess.bug("ty_fn_args() called on non-fn type");
}

fn ty_fn_proto(&ctxt cx, &t fty) -> ast::proto {
    alt (struct(cx, fty)) { case (ty::ty_fn(?p, _, _, _, _)) { ret p; } }
    cx.sess.bug("ty_fn_proto() called on non-fn type");
}

fn ty_fn_abi(&ctxt cx, &t fty) -> ast::native_abi {
    alt (struct(cx, fty)) { case (ty::ty_native_fn(?a, _, _)) { ret a; } }
    cx.sess.bug("ty_fn_abi() called on non-native-fn type");
}

fn ty_fn_ret(&ctxt cx, &t fty) -> t {
    alt (struct(cx, fty)) {
        case (ty::ty_fn(_, _, ?r, _, _)) { ret r; }
        case (ty::ty_native_fn(_, _, ?r)) { ret r; }
    }
    cx.sess.bug("ty_fn_ret() called on non-fn type");
}

fn is_fn_ty(&ctxt cx, &t fty) -> bool {
    alt (struct(cx, fty)) {
        case (ty::ty_fn(_, _, _, _, _)) { ret true; }
        case (ty::ty_native_fn(_, _, _)) { ret true; }
        case (_) { ret false; }
    }
}

fn ty_var_id(&ctxt cx, t typ) -> int {
    alt (struct(cx, typ)) {
        case (ty::ty_var(?vid)) { ret vid; }
        case (_) { log_err "ty_var_id called on non-var ty"; fail; }
    }
}


// Type accessors for AST nodes
fn block_ty(&ctxt cx, &ast::block b) -> t {
    ret node_id_to_type(cx, b.node.id);
}


// Returns the type of a pattern as a monotype. Like @expr_ty, this function
// doesn't provide type parameter substitutions.
fn pat_ty(&ctxt cx, &@ast::pat pat) -> t {
    ret node_id_to_monotype(cx, pat.id);
}


// Returns the type of an expression as a monotype.
//
// NB: This type doesn't provide type parameter substitutions; e.g. if you
// ask for the type of "id" in "id(3)", it will return "fn(&int) -> int"
// instead of "fn(&T) -> T with T = int". If this isn't what you want, see
// expr_ty_params_and_ty() below.
fn expr_ty(&ctxt cx, &@ast::expr expr) -> t {
    ret node_id_to_monotype(cx, expr.id);
}

fn expr_ty_params_and_ty(&ctxt cx, &@ast::expr expr) -> tup(t[], t) {
    ret tup(node_id_to_type_params(cx, expr.id),
            node_id_to_type(cx, expr.id));
}

fn expr_has_ty_params(&ctxt cx, &@ast::expr expr) -> bool {
    ret node_id_has_type_params(cx, expr.id);
}

fn decl_local_ty(&ctxt cx, &@ast::local l) -> t {
    ret node_id_to_type(cx, l.node.id);
}

fn stmt_node_id(&@ast::stmt s) -> ast::node_id {
    alt (s.node) {
        case (ast::stmt_decl(_, ?id)) { ret id; }
        case (ast::stmt_expr(_, ?id)) { ret id; }
        case (ast::stmt_crate_directive(_)) {
            log_err "ty::stmt_node_id(): crate directive found";
            fail;
        }
    }
}

// Expression utilities
fn field_num(&session::session sess, &span sp, &ast::ident id) -> uint {
    let uint accum = 0u;
    let uint i = 0u;
    for (u8 c in id) {
        if (i == 0u) {
            if (c != '_' as u8) {
                sess.span_fatal(sp,
                              "bad numeric field on tuple: " +
                                  "missing leading underscore");
            }
        } else {
            if ('0' as u8 <= c && c <= '9' as u8) {
                accum *= 10u;
                accum += (c as uint) - ('0' as uint);
            } else {
                auto s = "";
                s += str::unsafe_from_byte(c);
                sess.span_fatal(sp,
                              "bad numeric field on tuple: " +
                                  " non-digit character: " + s);
            }
        }
        i += 1u;
    }
    ret accum;
}

fn field_idx(&session::session sess, &span sp, &ast::ident id,
             &field[] fields) -> uint {
    let uint i = 0u;
    for (field f in fields) { if (str::eq(f.ident, id)) { ret i; } i += 1u; }
    sess.span_fatal(sp, "unknown field '" + id + "' of record");
}

fn method_idx(&session::session sess, &span sp, &ast::ident id,
              &method[] meths) -> uint {
    let uint i = 0u;
    for (method m in meths) { if (str::eq(m.ident, id)) { ret i; } i += 1u; }
    sess.span_fatal(sp, "unknown method '" + id + "' of obj");
}

fn sort_methods(&method[] meths) -> method[] {
    fn method_lteq(&method a, &method b) -> bool {
        ret str::lteq(a.ident, b.ident);
    }
    ret std::sort::ivector::merge_sort[method](bind method_lteq(_, _), meths);
}

fn is_lval(&@ast::expr expr) -> bool {
    alt (expr.node) {
        case (ast::expr_field(_, _)) { ret true; }
        case (ast::expr_index(_, _)) { ret true; }
        case (ast::expr_path(_)) { ret true; }
        case (ast::expr_unary(ast::deref, _)) { ret true; }
        case (_) { ret false; }
    }
}


// Type unification via Robinson's algorithm (Robinson 1965). Implemented as
// described in Hoder and Voronkov:
//
//     http://www.cs.man.ac.uk/~hoderk/ubench/unification_full.pdf
mod unify {

    export fixup_result;
    export fixup_vars;
    export fix_ok;
    export fix_err;
    export mk_var_bindings;
    export resolve_type_bindings;
    export resolve_type_structure;
    export resolve_type_var;
    export result;
    export unify;
    export ures_ok;
    export ures_err;
    export var_bindings;

    tag result { ures_ok(t); ures_err(type_err); }
    tag union_result { unres_ok; unres_err(type_err); }
    tag fixup_result {
        fix_ok(t); // fixup succeeded

        fix_err(int); // fixup failed because a type variable was unresolved

    }
    type var_bindings =
        rec(ufindivec::ufind sets, smallintmap::smallintmap[t] types);

    type ctxt = rec(@var_bindings vb, ty_ctxt tcx);

    fn mk_var_bindings() -> @var_bindings {
        ret @rec(sets=ufindivec::make(), types=smallintmap::mk[t]());
    }

    // Unifies two sets.
    fn union(&@ctxt cx, uint set_a, uint set_b) -> union_result {
        ufindivec::grow(cx.vb.sets, uint::max(set_a, set_b) + 1u);
        auto root_a = ufindivec::find(cx.vb.sets, set_a);
        auto root_b = ufindivec::find(cx.vb.sets, set_b);

        auto replace_type = bind fn (&@ctxt cx, t t, uint set_a, uint set_b) {
            ufindivec::union(cx.vb.sets, set_a, set_b);
            let uint root_c = ufindivec::find(cx.vb.sets, set_a);
            smallintmap::insert[t](cx.vb.types, root_c, t);
        } (_, _, set_a, set_b);

        alt (smallintmap::find(cx.vb.types, root_a)) {
            case (none) {
                alt (smallintmap::find(cx.vb.types, root_b)) {
                    case (none) {
                        ufindivec::union(cx.vb.sets, set_a, set_b);
                        ret unres_ok; }
                    case (some(?t_b)) {
                        replace_type(cx, t_b);
                        ret unres_ok;
                    }
                }
            }
            case (some(?t_a)) {
                alt (smallintmap::find(cx.vb.types, root_b)) {
                    case (none) {
                        replace_type(cx, t_a);
                        ret unres_ok;
                    }
                    case (some(?t_b)) {
                        alt (unify_step(cx, t_a, t_b)) {
                            case (ures_ok(?t_c)) {
                                replace_type(cx, t_c);
                                ret unres_ok;
                            }
                            case (ures_err(?terr)) {
                                ret unres_err(terr);
                            }
                        }
                    }
                }
            }
        }
    }
    fn record_var_binding(&@ctxt cx, int key, t typ) -> result {
        ufindivec::grow(cx.vb.sets, (key as uint) + 1u);
        auto root = ufindivec::find(cx.vb.sets, key as uint);
        auto result_type = typ;
        alt (smallintmap::find[t](cx.vb.types, root)) {
            case (some(?old_type)) {
                alt (unify_step(cx, old_type, typ)) {
                    case (ures_ok(?unified_type)) {
                        result_type = unified_type;
                    }
                    case (?rs) { ret rs; }
                }
            }
            case (none) {/* fall through */ }
        }
        smallintmap::insert[t](cx.vb.types, root, result_type);
        ret ures_ok(typ);
    }

    // Wraps the given type in an appropriate cname.
    //
    // TODO: This doesn't do anything yet. We should carry the cname up from
    // the expected and/or actual types when unification results in a type
    // identical to one or both of the two. The precise algorithm for this is
    // something we'll probably need to develop over time.

    // Simple structural type comparison.
    fn struct_cmp(@ctxt cx, t expected, t actual) -> result {
        if (struct(cx.tcx, expected) == struct(cx.tcx, actual)) {
            ret ures_ok(expected);
        }
        ret ures_err(terr_mismatch);
    }

    // Right now this just checks that the lists of constraints are
    // pairwise equal.
    fn unify_constrs(&t base_t, (@type_constr)[] expected,
                     &(@type_constr)[] actual) -> result {
        auto expected_len = ivec::len(expected);
        auto actual_len = ivec::len(actual);

        if (expected_len != actual_len) {
            ret ures_err(terr_constr_len(expected_len, actual_len));
        }
        auto i = 0u;
        auto rslt;
        for (@type_constr c in expected) {
            rslt = unify_constr(base_t, c, actual.(i));
            alt (rslt) {
                case (ures_ok(_)) { }
                case (ures_err(_)) { ret rslt; }
            }
            i += 1u;
        }
        ret ures_ok(base_t);
    }
    fn unify_constr(&t base_t, @type_constr expected,
                    &@type_constr actual_constr) -> result {
        auto ok_res = ures_ok(base_t);
        auto err_res = ures_err(terr_constr_mismatch(expected,
                                                     actual_constr));
        if (expected.node.id != actual_constr.node.id) {
            ret err_res;
        }
        auto expected_arg_len = ivec::len(expected.node.args);
        auto actual_arg_len = ivec::len(actual_constr.node.args);
        if (expected_arg_len != actual_arg_len) {
            ret err_res;
        }
        auto i = 0u;
        auto actual;
        for (@ty_constr_arg a in expected.node.args) {
            actual = actual_constr.node.args.(i);
            alt (a.node) {
                case (carg_base) {
                    alt (actual.node) {
                        case (carg_base) { }
                        case (_)         { ret err_res; }
                    }
                }
                case (carg_lit(?l)) {
                    alt (actual.node) {
                        case (carg_lit(?m)) {
                            if (l != m) { ret err_res; }
                        }
                        case (_) { ret err_res; }
                    }
                }
                case (carg_ident(?p)) {
                    alt (actual.node) {
                        case (carg_ident(?q)) {
                            if (p.node != q.node) {
                                ret err_res;
                            }
                        }
                        case (_) { ret err_res; }
                    }
                }
            }
            i += 1u;
        }
        ret ok_res;
    }

    // Unifies two mutability flags.
    fn unify_mut(ast::mutability expected, ast::mutability actual) ->
       option::t[ast::mutability] {
        if (expected == actual) { ret some(expected); }
        if (expected == ast::maybe_mut) { ret some(actual); }
        if (actual == ast::maybe_mut) { ret some(expected); }
        ret none;
    }
    tag fn_common_res {
        fn_common_res_err(result);
        fn_common_res_ok(arg[], t);
    }
    fn unify_fn_common(&@ctxt cx, &t expected, &t actual,
                       &arg[] expected_inputs, &t expected_output,
                       &arg[] actual_inputs, &t actual_output) ->
       fn_common_res {
        auto expected_len = ivec::len[arg](expected_inputs);
        auto actual_len = ivec::len[arg](actual_inputs);
        if (expected_len != actual_len) {
            ret fn_common_res_err(ures_err(terr_arg_count));
        }
        // TODO: as above, we should have an iter2 iterator.

        let arg[] result_ins = ~[];
        auto i = 0u;
        while (i < expected_len) {
            auto expected_input = expected_inputs.(i);
            auto actual_input = actual_inputs.(i);
            // Unify the result modes.

            auto result_mode;
            if (expected_input.mode != actual_input.mode) {
                ret fn_common_res_err(ures_err(terr_mode_mismatch(
                                    expected_input.mode, actual_input.mode)));
            } else { result_mode = expected_input.mode; }
            auto result = unify_step(cx, expected_input.ty, actual_input.ty);
            alt (result) {
                case (ures_ok(?rty)) {
                    result_ins += ~[rec(mode=result_mode, ty=rty)];
                }
                case (_) { ret fn_common_res_err(result); }
            }
            i += 1u;
        }
        // Check the output.

        auto result = unify_step(cx, expected_output, actual_output);
        alt (result) {
            case (ures_ok(?rty)) { ret fn_common_res_ok(result_ins, rty); }
            case (_) { ret fn_common_res_err(result); }
        }
    }
    fn unify_fn(&@ctxt cx, &ast::proto e_proto, &ast::proto a_proto,
                &t expected, &t actual, &arg[] expected_inputs,
                &t expected_output, &arg[] actual_inputs, &t actual_output,
                &controlflow expected_cf, &controlflow actual_cf,
                &(@constr)[] expected_constrs,
                &(@constr)[] actual_constrs) -> result {
        if (e_proto != a_proto) { ret ures_err(terr_mismatch); }
        alt (expected_cf) {
            case (ast::return) { }
            case ( // ok
                 ast::noreturn) {
                alt (actual_cf) {
                    case (ast::noreturn) {
                        // ok

                    }
                    case (_) {
                        /* even though typestate checking is mostly
                           responsible for checking control flow annotations,
                           this check is necessary to ensure that the
                           annotation in an object method matches the
                           declared object type */

                        ret ures_err(terr_controlflow_mismatch);
                    }
                }
            }
        }
        auto t =
            unify_fn_common(cx, expected, actual, expected_inputs,
                            expected_output, actual_inputs, actual_output);
        alt (t) {
            case (fn_common_res_err(?r)) { ret r; }
            case (fn_common_res_ok(?result_ins, ?result_out)) {
                auto t2 =
                    mk_fn(cx.tcx, e_proto, result_ins, result_out, actual_cf,
                          actual_constrs);
                ret ures_ok(t2);
            }
        }
    }
    fn unify_native_fn(&@ctxt cx, &ast::native_abi e_abi,
                       &ast::native_abi a_abi, &t expected, &t actual,
                       &arg[] expected_inputs, &t expected_output,
                       &arg[] actual_inputs, &t actual_output) -> result {
        if (e_abi != a_abi) { ret ures_err(terr_mismatch); }
        auto t =
            unify_fn_common(cx, expected, actual, expected_inputs,
                            expected_output, actual_inputs, actual_output);
        alt (t) {
            case (fn_common_res_err(?r)) { ret r; }
            case (fn_common_res_ok(?result_ins, ?result_out)) {
                auto t2 = mk_native_fn(cx.tcx, e_abi, result_ins, result_out);
                ret ures_ok(t2);
            }
        }
    }
    fn unify_obj(&@ctxt cx, &t expected, &t actual,
                 &method[] expected_meths, &method[] actual_meths) ->
       result {
        let method[] result_meths = ~[];
        let uint i = 0u;
        let uint expected_len = ivec::len[method](expected_meths);
        let uint actual_len = ivec::len[method](actual_meths);
        if (expected_len != actual_len) { ret ures_err(terr_meth_count); }
        while (i < expected_len) {
            auto e_meth = expected_meths.(i);
            auto a_meth = actual_meths.(i);
            if (!str::eq(e_meth.ident, a_meth.ident)) {
                ret ures_err(terr_obj_meths(e_meth.ident, a_meth.ident));
            }
            auto r =
                unify_fn(cx, e_meth.proto, a_meth.proto, expected, actual,
                         e_meth.inputs, e_meth.output, a_meth.inputs,
                         a_meth.output, e_meth.cf, a_meth.cf, e_meth.constrs,
                         a_meth.constrs);
            alt (r) {
                case (ures_ok(?tfn)) {
                    alt (struct(cx.tcx, tfn)) {
                        case (ty_fn(?proto, ?ins, ?out, ?cf, ?constrs)) {
                            result_meths +=
                                ~[rec(inputs=ins,
                                      output=out,
                                      cf=cf,
                                      constrs=constrs with e_meth)];
                        }
                    }
                }
                case (_) { ret r; }
            }
            i += 1u;
        }
        auto t = mk_obj(cx.tcx, result_meths);
        ret ures_ok(t);
    }

    // If the given type is a variable, returns the structure of that type.
    fn resolve_type_structure(&ty_ctxt tcx, &@var_bindings vb, t typ) ->
       fixup_result {
        alt (struct(tcx, typ)) {
            case (ty_var(?vid)) {
                if (vid as uint >= ufindivec::set_count(vb.sets)) {
                    ret fix_err(vid);
                }
                auto root_id = ufindivec::find(vb.sets, vid as uint);
                alt (smallintmap::find[t](vb.types, root_id)) {
                    case (none) { ret fix_err(vid); }
                    case (some(?rt)) { ret fix_ok(rt); }
                }
            }
            case (_) { ret fix_ok(typ); }
        }
    }
    fn unify_step(&@ctxt cx, &t expected, &t actual) -> result {
        // TODO: rewrite this using tuple pattern matching when available, to
        // avoid all this rightward drift and spikiness.

        // TODO: occurs check, to make sure we don't loop forever when
        // unifying e.g. 'a and option['a]

        // Fast path.

        if (eq_ty(expected, actual)) { ret ures_ok(expected); }
        // Stage 1: Handle the cases in which one side or another is a type
        // variable.

        alt (struct(cx.tcx, actual)) {
            case (
                 // If the RHS is a variable type, then just do the
                 // appropriate binding.
                 ty::ty_var(?actual_id)) {
                auto actual_n = actual_id as uint;
                alt (struct(cx.tcx, expected)) {
                    case (ty::ty_var(?expected_id)) {
                        auto expected_n = expected_id as uint;
                        alt(union(cx, expected_n, actual_n)) {
                            case (unres_ok) { /* fall through */ }
                            case (unres_err(?t_e)) {
                                ret ures_err(t_e);
                            }
                        }
                    }
                    case (_) {

                        // Just bind the type variable to the expected type.
                        alt (record_var_binding(cx, actual_id, expected)) {
                            case (ures_ok(_)) {/* fall through */ }
                            case (?rs) { ret rs; }
                        }
                    }
                }
                ret ures_ok(mk_var(cx.tcx, actual_id));
            }
            case (_) {/* empty */ }
        }
        alt (struct(cx.tcx, expected)) {
            case (ty::ty_var(?expected_id)) {
                // Add a binding. (`actual` can't actually be a var here.)

                alt (record_var_binding(cx, expected_id, actual)) {
                    case (ures_ok(_)) {/* fall through */ }
                    case (?rs) { ret rs; }
                }
                ret ures_ok(mk_var(cx.tcx, expected_id));
            }
            case (_) {/* fall through */ }
        }
        // Stage 2: Handle all other cases.

        alt (struct(cx.tcx, actual)) {
            case (ty::ty_bot) { ret ures_ok(expected); }
            case (_) {/* fall through */ }
        }
        alt (struct(cx.tcx, expected)) {
            case (ty::ty_nil) { ret struct_cmp(cx, expected, actual); }
            // _|_ unifies with anything
            case (ty::ty_bot) {
                ret ures_ok(actual);
            }
            case (ty::ty_bool) { ret struct_cmp(cx, expected, actual); }
            case (ty::ty_int) { ret struct_cmp(cx, expected, actual); }
            case (ty::ty_uint) { ret struct_cmp(cx, expected, actual); }
            case (ty::ty_machine(_)) { ret struct_cmp(cx, expected, actual); }
            case (ty::ty_float) { ret struct_cmp(cx, expected, actual); }
            case (ty::ty_char) { ret struct_cmp(cx, expected, actual); }
            case (ty::ty_str) { ret struct_cmp(cx, expected, actual); }
            case (ty::ty_istr) { ret struct_cmp(cx, expected, actual); }
            case (ty::ty_type) { ret struct_cmp(cx, expected, actual); }
            case (ty::ty_native(?ex_id)) {
                alt (struct(cx.tcx, actual)) {
                    case (ty_native(?act_id)) {
                        if (ex_id._0 == act_id._0 && ex_id._1 == act_id._1) {
                            ret ures_ok(actual);
                        } else {
                            ret ures_err(terr_mismatch);
                        }
                    }
                    case (_) { ret ures_err(terr_mismatch); }
                }
            }
            case (ty::ty_param(_)) { ret struct_cmp(cx, expected, actual); }
            case (ty::ty_tag(?expected_id, ?expected_tps)) {
                alt (struct(cx.tcx, actual)) {
                    case (ty::ty_tag(?actual_id, ?actual_tps)) {
                        if (expected_id._0 != actual_id._0 ||
                                expected_id._1 != actual_id._1) {
                            ret ures_err(terr_mismatch);
                        }
                        // TODO: factor this cruft out, see the TODO in the
                        // ty::ty_tup case

                        let t[] result_tps = ~[];
                        auto i = 0u;
                        auto expected_len = ivec::len[t](expected_tps);
                        while (i < expected_len) {
                            auto expected_tp = expected_tps.(i);
                            auto actual_tp = actual_tps.(i);
                            auto result =
                                unify_step(cx, expected_tp, actual_tp);
                            alt (result) {
                                case (ures_ok(?rty)) { result_tps += ~[rty]; }
                                case (_) { ret result; }
                            }
                            i += 1u;
                        }
                        ret ures_ok(mk_tag(cx.tcx, expected_id, result_tps));
                    }
                    case (_) {/* fall through */ }
                }
                ret ures_err(terr_mismatch);
            }
            case (ty::ty_box(?expected_mt)) {
                alt (struct(cx.tcx, actual)) {
                    case (ty::ty_box(?actual_mt)) {
                        auto mut;
                        alt (unify_mut(expected_mt.mut, actual_mt.mut)) {
                            case (none) { ret ures_err(terr_box_mutability); }
                            case (some(?m)) { mut = m; }
                        }
                        auto result =
                            unify_step(cx, expected_mt.ty, actual_mt.ty);
                        alt (result) {
                            case (ures_ok(?result_sub)) {
                                auto mt = rec(ty=result_sub, mut=mut);
                                ret ures_ok(mk_box(cx.tcx, mt));
                            }
                            case (_) { ret result; }
                        }
                    }
                    case (_) { ret ures_err(terr_mismatch); }
                }
            }
            case (ty::ty_vec(?expected_mt)) {
                alt (struct(cx.tcx, actual)) {
                    case (ty::ty_vec(?actual_mt)) {
                        auto mut;
                        alt (unify_mut(expected_mt.mut, actual_mt.mut)) {
                            case (none) { ret ures_err(terr_vec_mutability); }
                            case (some(?m)) { mut = m; }
                        }
                        auto result =
                            unify_step(cx, expected_mt.ty, actual_mt.ty);
                        alt (result) {
                            case (ures_ok(?result_sub)) {
                                auto mt = rec(ty=result_sub, mut=mut);
                                ret ures_ok(mk_vec(cx.tcx, mt));
                            }
                            case (_) { ret result; }
                        }
                    }
                    case (_) { ret ures_err(terr_mismatch); }
                }
            }
            case (ty::ty_ivec(?expected_mt)) {
                alt (struct(cx.tcx, actual)) {
                    case (ty::ty_ivec(?actual_mt)) {
                        auto mut;
                        alt (unify_mut(expected_mt.mut, actual_mt.mut)) {
                            case (none) { ret ures_err(terr_vec_mutability); }
                            case (some(?m)) { mut = m; }
                        }
                        auto result =
                            unify_step(cx, expected_mt.ty, actual_mt.ty);
                        alt (result) {
                            case (ures_ok(?result_sub)) {
                                auto mt = rec(ty=result_sub, mut=mut);
                                ret ures_ok(mk_ivec(cx.tcx, mt));
                            }
                            case (_) { ret result; }
                        }
                    }
                    case (_) { ret ures_err(terr_mismatch); }
                }
            }
            case (ty::ty_ptr(?expected_mt)) {
                alt (struct(cx.tcx, actual)) {
                    case (ty::ty_ptr(?actual_mt)) {
                        auto mut;
                        alt (unify_mut(expected_mt.mut, actual_mt.mut)) {
                            case (none) { ret ures_err(terr_vec_mutability); }
                            case (some(?m)) { mut = m; }
                        }
                        auto result =
                            unify_step(cx, expected_mt.ty, actual_mt.ty);
                        alt (result) {
                            case (ures_ok(?result_sub)) {
                                auto mt = rec(ty=result_sub, mut=mut);
                                ret ures_ok(mk_ptr(cx.tcx, mt));
                            }
                            case (_) { ret result; }
                        }
                    }
                    case (_) { ret ures_err(terr_mismatch); }
                }
            }
            case (ty::ty_port(?expected_sub)) {
                alt (struct(cx.tcx, actual)) {
                    case (ty::ty_port(?actual_sub)) {
                        auto result =
                            unify_step(cx, expected_sub, actual_sub);
                        alt (result) {
                            case (ures_ok(?result_sub)) {
                                ret ures_ok(mk_port(cx.tcx, result_sub));
                            }
                            case (_) { ret result; }
                        }
                    }
                    case (_) { ret ures_err(terr_mismatch); }
                }
            }
            case (ty::ty_res(?ex_id, ?ex_inner, ?ex_tps)) {
                alt (struct(cx.tcx, actual)) {
                    case (ty::ty_res(?act_id, ?act_inner, ?act_tps)) {
                        if (ex_id._0 != act_id._0 || ex_id._1 != act_id._1) {
                            ret ures_err(terr_mismatch);
                        }
                        auto result = unify_step(cx, ex_inner, act_inner);
                        alt (result) {
                            case (ures_ok(?res_inner)) {
                                auto i = 0u;
                                auto res_tps = ~[];
                                for (t ex_tp in ex_tps) {
                                    auto result =
                                        unify_step(cx, ex_tp, act_tps.(i));
                                    alt (result) {
                                        case (ures_ok(?rty)) {
                                            res_tps += ~[rty];
                                        }
                                        case (_) { ret result; }
                                    }
                                    i += 1u;
                                }
                                ret ures_ok(mk_res(cx.tcx, act_id, res_inner,
                                                   res_tps));
                            }
                            case (_) { ret result; }
                        }
                    }
                    case (_) { ret ures_err(terr_mismatch); }
                }
            }
            case (ty::ty_chan(?expected_sub)) {
                alt (struct(cx.tcx, actual)) {
                    case (ty::ty_chan(?actual_sub)) {
                        auto result =
                            unify_step(cx, expected_sub, actual_sub);
                        alt (result) {
                            case (ures_ok(?result_sub)) {
                                ret ures_ok(mk_chan(cx.tcx, result_sub));
                            }
                            case (_) { ret result; }
                        }
                    }
                    case (_) { ret ures_err(terr_mismatch); }
                }
            }
            case (ty::ty_tup(?expected_elems)) {
                alt (struct(cx.tcx, actual)) {
                    case (ty::ty_tup(?actual_elems)) {
                        auto expected_len = ivec::len[ty::mt](expected_elems);
                        auto actual_len = ivec::len[ty::mt](actual_elems);
                        if (expected_len != actual_len) {
                            auto err =
                                terr_tuple_size(expected_len, actual_len);
                            ret ures_err(err);
                        }
                        // TODO: implement an iterator that can iterate over
                        // two arrays simultaneously.

                        let ty::mt[] result_elems = ~[];
                        auto i = 0u;
                        while (i < expected_len) {
                            auto expected_elem = expected_elems.(i);
                            auto actual_elem = actual_elems.(i);
                            auto mut;
                            alt (unify_mut(expected_elem.mut,
                                           actual_elem.mut)) {
                                case (none) {
                                    auto err = terr_tuple_mutability;
                                    ret ures_err(err);
                                }
                                case (some(?m)) { mut = m; }
                            }
                            auto result =
                                unify_step(cx, expected_elem.ty,
                                           actual_elem.ty);
                            alt (result) {
                                case (ures_ok(?rty)) {
                                    auto mt = rec(ty=rty, mut=mut);
                                    result_elems += ~[mt];
                                }
                                case (_) { ret result; }
                            }
                            i += 1u;
                        }
                        ret ures_ok(mk_tup(cx.tcx, result_elems));
                    }
                    case (_) { ret ures_err(terr_mismatch); }
                }
            }
            case (ty::ty_rec(?expected_fields)) {
                alt (struct(cx.tcx, actual)) {
                    case (ty::ty_rec(?actual_fields)) {
                        auto expected_len = ivec::len[field](expected_fields);
                        auto actual_len = ivec::len[field](actual_fields);
                        if (expected_len != actual_len) {
                            auto err =
                                terr_record_size(expected_len, actual_len);
                            ret ures_err(err);
                        }
                        // TODO: implement an iterator that can iterate over
                        // two arrays simultaneously.

                        let field[] result_fields = ~[];
                        auto i = 0u;
                        while (i < expected_len) {
                            auto expected_field = expected_fields.(i);
                            auto actual_field = actual_fields.(i);
                            auto mut;
                            alt (unify_mut(expected_field.mt.mut,
                                           actual_field.mt.mut)) {
                                case (none) {
                                    ret ures_err(terr_record_mutability);
                                }
                                case (some(?m)) { mut = m; }
                            }
                            if (!str::eq(expected_field.ident,
                                         actual_field.ident)) {
                                auto err =
                                    terr_record_fields(expected_field.ident,
                                                       actual_field.ident);
                                ret ures_err(err);
                            }
                            auto result =
                                unify_step(cx, expected_field.mt.ty,
                                           actual_field.mt.ty);
                            alt (result) {
                                case (ures_ok(?rty)) {
                                    auto mt = rec(ty=rty, mut=mut);
                                    result_fields +=
                                        ~[rec(mt=mt with expected_field)];
                                }
                                case (_) { ret result; }
                            }
                            i += 1u;
                        }
                        ret ures_ok(mk_rec(cx.tcx, result_fields));
                    }
                    case (_) { ret ures_err(terr_mismatch); }
                }
            }
            case (ty::ty_fn(?ep, ?expected_inputs, ?expected_output,
                            ?expected_cf, ?expected_constrs)) {
                alt (struct(cx.tcx, actual)) {
                    case (ty::ty_fn(?ap, ?actual_inputs, ?actual_output,
                                    ?actual_cf, ?actual_constrs)) {
                        ret unify_fn(cx, ep, ap, expected, actual,
                                     expected_inputs, expected_output,
                                     actual_inputs, actual_output,
                                     expected_cf, actual_cf, expected_constrs,
                                     actual_constrs);
                    }
                    case (_) { ret ures_err(terr_mismatch); }
                }
            }
            case (ty::ty_native_fn(?e_abi, ?expected_inputs,
                                   ?expected_output)) {
                alt (struct(cx.tcx, actual)) {
                    case (ty::ty_native_fn(?a_abi, ?actual_inputs,
                                           ?actual_output)) {
                        ret unify_native_fn(cx, e_abi, a_abi, expected,
                                            actual, expected_inputs,
                                            expected_output, actual_inputs,
                                            actual_output);
                    }
                    case (_) { ret ures_err(terr_mismatch); }
                }
            }
            case (ty::ty_obj(?expected_meths)) {
                alt (struct(cx.tcx, actual)) {
                    case (ty::ty_obj(?actual_meths)) {
                        ret unify_obj(cx, expected, actual, expected_meths,
                                      actual_meths);
                    }
                    case (_) { ret ures_err(terr_mismatch); }
                }
            }
            case (ty::ty_constr(?expected_t, ?expected_constrs)) {
                // unify the base types...
                alt (struct(cx.tcx, actual)) {
                    case (ty::ty_constr(?actual_t, ?actual_constrs)) {
                        auto rslt = unify_step(cx, expected_t, actual_t);
                        alt (rslt) {
                            case (ures_ok(?rty)) {
                                // FIXME: probably too restrictive --
                                // requires the constraints to be
                                // syntactically equal
                                ret unify_constrs(expected,
                                                  expected_constrs,
                                                  actual_constrs);
                            }
                            case (_) { ret rslt; }
                        }
                    }
                    case (_) {
                        // If the actual type is *not* a constrained type,
                        // then we go ahead and just ignore the constraints on
                        // the expected type. typestate handles the rest.
                        ret unify_step(cx, expected_t, actual);
                    }
                }
            }
        }
    }
    fn unify(&t expected, &t actual, &@var_bindings vb, &ty_ctxt tcx) ->
       result {
        auto cx = @rec(vb=vb, tcx=tcx);
        ret unify_step(cx, expected, actual);
    }
    fn dump_var_bindings(ty_ctxt tcx, @var_bindings vb) {
        auto i = 0u;
        while (i < ivec::len[ufindivec::node](vb.sets.nodes)) {
            auto sets = "";
            auto j = 0u;
            while (j < ivec::len[option::t[uint]](vb.sets.nodes)) {
                if (ufindivec::find(vb.sets, j) == i) {
                    sets += #fmt(" %u", j);
                }
                j += 1u;
            }
            auto typespec;
            alt (smallintmap::find[t](vb.types, i)) {
                case (none) { typespec = ""; }
                case (some(?typ)) {
                    typespec = " =" + ty_to_str(tcx, typ);
                }
            }
            log_err #fmt("set %u:%s%s", i, typespec, sets);
            i += 1u;
        }
    }

    // Fixups and substitutions
    fn fixup_vars(ty_ctxt tcx, @var_bindings vb, t typ) -> fixup_result {
        fn subst_vars(ty_ctxt tcx, @var_bindings vb,
                      @mutable option::t[int] unresolved, int vid) -> t {
            if (vid as uint >= ufindivec::set_count(vb.sets)) {
                *unresolved = some(vid);
                ret ty::mk_var(tcx, vid);
            }
            auto root_id = ufindivec::find(vb.sets, vid as uint);
            alt (smallintmap::find[t](vb.types, root_id)) {
                case (none) {
                    *unresolved = some(vid);
                    ret ty::mk_var(tcx, vid);
                }
                case (some(?rt)) {
                    ret fold_ty(tcx,
                                fm_var(bind subst_vars(tcx, vb, unresolved,
                                                       _)), rt);
                }
            }
        }
        auto unresolved = @mutable none[int];
        auto rty =
            fold_ty(tcx, fm_var(bind subst_vars(tcx, vb, unresolved, _)),
                    typ);
        auto ur = *unresolved;
        alt (ur) {
            case (none) { ret fix_ok(rty); }
            case (some(?var_id)) { ret fix_err(var_id); }
        }
    }
    fn resolve_type_var(&ty_ctxt tcx, &@var_bindings vb, int vid) ->
       fixup_result {
        if (vid as uint >= ufindivec::set_count(vb.sets)) {
            ret fix_err(vid);
        }
        auto root_id = ufindivec::find(vb.sets, vid as uint);
        alt (smallintmap::find[t](vb.types, root_id)) {
            case (none) { ret fix_err(vid); }
            case (some(?rt)) { ret fixup_vars(tcx, vb, rt); }
        }
    }
}

fn type_err_to_str(&ty::type_err err) -> str {
    alt (err) {
        case (terr_mismatch) { ret "types differ"; }
        case (terr_controlflow_mismatch) {
            ret "returning function used where non-returning function" +
                    " was expected";
        }
        case (terr_box_mutability) {
            ret "boxed values differ in mutability";
        }
        case (terr_vec_mutability) { ret "vectors differ in mutability"; }
        case (terr_tuple_size(?e_sz, ?a_sz)) {
            ret "expected a tuple with " + uint::to_str(e_sz, 10u) +
                    " elements but found one with " + uint::to_str(a_sz, 10u)
                    + " elements";
        }
        case (terr_tuple_mutability) {
            ret "tuple elements differ in mutability";
        }
        case (terr_record_size(?e_sz, ?a_sz)) {
            ret "expected a record with " + uint::to_str(e_sz, 10u) +
                    " fields but found one with " + uint::to_str(a_sz, 10u) +
                    " fields";
        }
        case (terr_record_mutability) {
            ret "record elements differ in mutability";
        }
        case (terr_record_fields(?e_fld, ?a_fld)) {
            ret "expected a record with field '" + e_fld +
                    "' but found one with field '" + a_fld + "'";
        }
        case (terr_arg_count) {
            ret "incorrect number of function parameters";
        }
        case (terr_meth_count) { ret "incorrect number of object methods"; }
        case (terr_obj_meths(?e_meth, ?a_meth)) {
            ret "expected an obj with method '" + e_meth +
                    "' but found one with method '" + a_meth + "'";
        }
        case (terr_mode_mismatch(?e_mode, ?a_mode)) {
            ret "expected argument mode " + mode_str_1(e_mode) + " but found "
                + mode_str_1(a_mode);
            fail;
        }
        case (terr_constr_len(?e_len, ?a_len)) {
            ret "Expected a type with " + uint::str(e_len) + " constraints, \
              but found one with " + uint::str(a_len) + " constraints";
        }
        case (terr_constr_mismatch(?e_constr, ?a_constr)) {
            ret "Expected a type with constraint "
              + ty_constr_to_str(e_constr) + " but found one with constraint "
                + ty_constr_to_str(a_constr);
        }
    }
}


// Converts type parameters in a type to type variables and returns the
// resulting type along with a list of type variable IDs.
fn bind_params_in_type(&span sp, &ctxt cx, fn() -> int  next_ty_var, t typ,
                       uint ty_param_count) -> tup(int[], t) {
    let @mutable int[] param_var_ids = @mutable ~[];
    auto i = 0u;
    while (i < ty_param_count) {
        *param_var_ids += ~[next_ty_var()];
        i += 1u;
    }
    fn binder(span sp, ctxt cx, @mutable int[] param_var_ids,
              fn() -> int next_ty_var, uint index) -> t {
        if (index < ivec::len(*param_var_ids)) {
            ret mk_var(cx, param_var_ids.(index));
        }
        else {
            cx.sess.span_fatal(sp, "Unbound type parameter in callee's type");
        }
    }
    auto new_typ =
        fold_ty(cx, fm_param(bind binder(sp, cx, param_var_ids,
                                         next_ty_var, _)), typ);
    ret tup(*param_var_ids, new_typ);
}


// Replaces type parameters in the given type using the given list of
// substitions.
fn substitute_type_params(&ctxt cx, &ty::t[] substs, t typ) -> t {
    if (!type_contains_params(cx, typ)) { ret typ; }
    fn substituter(ctxt cx, @ty::t[] substs, uint idx) -> t {
        // FIXME: bounds check can fail
        ret substs.(idx);
    }
    ret fold_ty(cx, fm_param(bind substituter(cx, @substs, _)), typ);
}

fn def_has_ty_params(&ast::def def) -> bool {
    alt (def) {
        case (ast::def_fn(_,_)) { ret true; }
        case (ast::def_obj_field(_)) { ret false; }
        case (ast::def_mod(_)) { ret false; }
        case (ast::def_const(_)) { ret false; }
        case (ast::def_arg(_)) { ret false; }
        case (ast::def_local(_)) { ret false; }
        case (ast::def_variant(_, _)) { ret true; }
        case (ast::def_ty(_)) { ret false; }
        case (ast::def_ty_arg(_)) { ret false; }
        case (ast::def_binding(_)) { ret false; }
        case (ast::def_use(_)) { ret false; }
        case (ast::def_native_ty(_)) { ret false; }
        case (ast::def_native_fn(_)) { ret true; }
    }
}


// Tag information
type variant_info = rec(ty::t[] args, ty::t ctor_ty, ast::def_id id);

fn tag_variants(&ctxt cx, &ast::def_id id) -> variant_info[] {
    if (ast::local_crate != id._0) { ret csearch::get_tag_variants(cx, id); }
    auto item = alt (cx.items.find(id._1)) {
        case (some(?i)) { i }
        case (none) {
            cx.sess.bug("expected to find cached node_item")
        }
    };
    alt (item) {
        case (ast_map::node_item(?item)) {
            alt (item.node) {
                case (ast::item_tag(?variants, _)) {
                    let variant_info[] result = ~[];
                    for (ast::variant variant in variants) {
                        auto ctor_ty = node_id_to_monotype
                            (cx, variant.node.id);
                        let t[] arg_tys = ~[];
                        if (std::ivec::len(variant.node.args) > 0u) {
                            for (arg a in ty_fn_args(cx, ctor_ty)) {
                                arg_tys += ~[a.ty];
                            }
                        }
                        auto did = variant.node.id;
                        result +=
                            ~[rec(args=arg_tys,
                                  ctor_ty=ctor_ty,
                                  id=ast::local_def(did))];
                    }
                    ret result;
                }
            }
        }
    }
}


// Returns information about the tag variant with the given ID:
fn tag_variant_with_id(&ctxt cx, &ast::def_id tag_id, &ast::def_id variant_id)
   -> variant_info {
    auto variants = tag_variants(cx, tag_id);
    auto i = 0u;
    while (i < ivec::len[variant_info](variants)) {
        auto variant = variants.(i);
        if (def_eq(variant.id, variant_id)) { ret variant; }
        i += 1u;
    }
    cx.sess.bug("tag_variant_with_id(): no variant exists with that ID");
}


// If the given item is in an external crate, looks up its type and adds it to
// the type cache. Returns the type parameters and type.
fn lookup_item_type(ctxt cx, ast::def_id did) -> ty_param_count_and_ty {
    if (did._0 == ast::local_crate) {
        // The item is in this crate. The caller should have added it to the
        // type cache already; we simply return it.

        ret cx.tcache.get(did);
    }
    alt (cx.tcache.find(did)) {
        case (some(?tpt)) { ret tpt; }
        case (none) {
            auto tyt = csearch::get_type(cx, did);
            cx.tcache.insert(did, tyt);
            ret tyt;
        }
    }
}

fn ret_ty_of_fn_ty(ctxt cx, t a_ty) -> t {
    alt (ty::struct(cx, a_ty)) {
        case (ty::ty_fn(_, _, ?ret_ty, _, _)) { ret ret_ty; }
        case (ty::ty_native_fn(_, _, ?ret_ty)) { ret ret_ty; }
        case (_) {
            cx.sess.bug("ret_ty_of_fn_ty() called on non-function type: " +
                        ty_to_str(cx, a_ty));
        }
    }
}

fn ret_ty_of_fn(ctxt cx, ast::node_id id) -> t {
    ret ret_ty_of_fn_ty(cx, node_id_to_type(cx, id));
}

fn is_binopable(&ctxt cx, t ty, ast::binop op) -> bool {

    const int tycat_other = 0;
    const int tycat_bool = 1;
    const int tycat_int = 2;
    const int tycat_float = 3;
    const int tycat_str = 4;
    const int tycat_vec = 5;
    const int tycat_struct = 6;

    const int opcat_add = 0;
    const int opcat_sub = 1;
    const int opcat_mult = 2;
    const int opcat_shift = 3;
    const int opcat_rel = 4;
    const int opcat_eq = 5;
    const int opcat_bit = 6;
    const int opcat_logic = 7;

    fn opcat(ast::binop op) -> int {
        alt (op) {
            case (ast::add) { opcat_add }
            case (ast::sub) { opcat_sub }
            case (ast::mul) { opcat_mult }
            case (ast::div) { opcat_mult }
            case (ast::rem) { opcat_mult }
            case (ast::and) { opcat_logic }
            case (ast::or) { opcat_logic }
            case (ast::bitxor) { opcat_bit }
            case (ast::bitand) { opcat_bit }
            case (ast::bitor) { opcat_bit }
            case (ast::lsl) { opcat_shift }
            case (ast::lsr) { opcat_shift }
            case (ast::asr) { opcat_shift }
            case (ast::eq) { opcat_eq }
            case (ast::ne) { opcat_eq }
            case (ast::lt) { opcat_rel }
            case (ast::le) { opcat_rel }
            case (ast::ge) { opcat_rel }
            case (ast::gt) { opcat_rel }
        }
    }

    fn tycat(&ctxt cx, t ty) -> int {
        alt (struct(cx, ty)) {
            case (ty_bool) { tycat_bool }
            case (ty_int) { tycat_int }
            case (ty_uint) { tycat_int }
            case (ty_machine(ast::ty_i8)) { tycat_int }
            case (ty_machine(ast::ty_i16)) { tycat_int }
            case (ty_machine(ast::ty_i32)) { tycat_int }
            case (ty_machine(ast::ty_i64)) { tycat_int }
            case (ty_machine(ast::ty_u8)) { tycat_int }
            case (ty_machine(ast::ty_u16)) { tycat_int }
            case (ty_machine(ast::ty_u32)) { tycat_int }
            case (ty_machine(ast::ty_u64)) { tycat_int }
            case (ty_float) { tycat_float }
            case (ty_machine(ast::ty_f32)) { tycat_float }
            case (ty_machine(ast::ty_f64)) { tycat_float }
            case (ty_char) { tycat_int }
            case (ty_ptr(_)) { tycat_int }
            case (ty_str) { tycat_str }
            case (ty_istr) { tycat_str }
            case (ty_vec(_)) { tycat_vec }
            case (ty_ivec(_)) { tycat_vec }
            case (ty_tup(_)) { tycat_struct }
            case (ty_rec(_)) { tycat_struct }
            case (ty_tag(_, _)) { tycat_struct }
            case (_) { tycat_other }
        }
    }

    const bool t = true;
    const bool f = false;

    /*.          add,     shift,   bit
      .             sub,     rel,     logic
      .                mult,    eq,         */
    auto tbl = [[f, f, f, f, t, t, f, f], /*other*/
                [f, f, f, f, t, t, t, t], /*bool*/
                [t, t, t, t, t, t, t, f], /*int*/
                [t, t, t, f, t, t, f, f], /*float*/
                [t, f, f, f, t, t, f, f], /*str*/
                [t, f, f, f, t, t, f, f], /*vec*/
                [f, f, f, f, t, t, f, f]];/*struct*/

    ret tbl.(tycat(cx, ty)).(opcat(op));
}

fn ast_constr_to_constr[T](ty::ctxt tcx, &@ast::constr_general[T] c)
    -> @ty::constr_general[T] {
    alt (tcx.def_map.find(c.node.id)) {
        case (some(ast::def_fn(?pred_id, ast::pure_fn))) {
            ret @respan(c.span, rec(path=c.node.path, args=c.node.args,
                                    id=pred_id));
        }
        case (_) {
            tcx.sess.span_fatal(c.span, "Predicate "
                      + path_to_str(c.node.path)
                      + " is unbound or bound to a non-function or an \
                        impure function");
        }
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
