import vec;
import str;
import uint;
import std::ufind;
import std::map;
import std::map::hashmap;
import option;
import option::none;
import option::some;
import std::smallintmap;
import driver::session;
import syntax::ast;
import syntax::ast::*;
import syntax::ast_util;
import syntax::codemap::span;
import metadata::csearch;
import util::common::*;
import syntax::util::interner;
import util::ppaux::ty_to_str;
import util::ppaux::ty_constr_to_str;
import util::ppaux::mode_str;
import syntax::print::pprust::*;

export node_id_to_monotype;
export node_id_to_type;
export node_id_to_type_params;
export node_id_to_ty_param_substs_opt_and_ty;
export arg;
export args_eq;
export ast_constr_to_constr;
export block_ty;
export constr;
export constr_general;
export constr_table;
export count_ty_params;
export ctxt;
export def_has_ty_params;
export expr_has_ty_params;
export expr_ty;
export expr_ty_params_and_ty;
export expr_is_lval;
export fold_ty;
export field;
export field_idx;
export get_field;
export fm_general;
export get_element_type;
export idx_nil;
export is_binopable;
export is_pred_ty;
export lookup_item_type;
export method;
export method_idx;
export mk_bool;
export mk_bot;
export mk_box;
export mk_char;
export mk_constr;
export mk_ctxt;
export mk_float;
export mk_fn;
export mk_imm_box;
export mk_imm_uniq;
export mk_mut_ptr;
export mk_int;
export mk_str;
export mk_vec;
export mk_mach_int;
export mk_mach_uint;
export mk_mach_float;
export mk_native;
export mk_native_fn;
export mk_nil;
export mk_obj;
export mk_iface;
export mk_res;
export mk_param;
export mk_ptr;
export mk_rec;
export mk_tag;
export mk_tup;
export mk_type;
export mk_send_type;
export mk_uint;
export mk_uniq;
export mk_var;
export mk_opaque_closure_ptr;
export mk_named;
export gen_ty;
export mode;
export mt;
export node_type_table;
export pat_ty;
export ret_ty_of_fn;
export sequence_element_type;
export struct;
export ty_name;
export sort_methods;
export stmt_node_id;
export sty;
export substitute_type_params;
export t;
export new_ty_hash;
export tag_variants;
export iface_methods, store_iface_methods, impl_iface;
export tag_variant_with_id;
export ty_param_substs_opt_and_ty;
export ty_param_bounds_and_ty;
export ty_native_fn;
export ty_bool;
export ty_bot;
export ty_box;
export ty_constr;
export ty_opaque_closure_ptr;
export ty_constr_arg;
export ty_float;
export ty_fn, fn_ty;
export ty_fn_proto;
export ty_fn_ret;
export ty_fn_ret_style;
export ty_int;
export ty_str;
export ty_vec;
export ty_native;
export ty_nil;
export ty_obj;
export ty_iface;
export ty_res;
export ty_param;
export ty_ptr;
export ty_rec;
export ty_tag;
export ty_tup;
export ty_type;
export ty_send_type;
export ty_uint;
export ty_uniq;
export ty_var;
export ty_named;
export same_type, same_method;
export ty_var_id;
export ty_param_substs_opt_and_ty_to_monotype;
export ty_fn_args;
export type_constr;
export type_contains_params;
export type_contains_vars;
export kind, kind_sendable, kind_copyable, kind_noncopyable;
export kind_can_be_copied, kind_can_be_sent, proto_kind, kind_lteq, type_kind;
export type_err;
export type_err_to_str;
export type_has_dynamic_size;
export type_needs_drop;
export type_is_bool;
export type_is_bot;
export type_is_box;
export type_is_boxed;
export type_is_unique_box;
export type_is_unsafe_ptr;
export type_is_vec;
export type_is_fp;
export type_allows_implicit_copy;
export type_is_integral;
export type_is_numeric;
export type_is_native;
export type_is_nil;
export type_is_pod;
export type_is_scalar;
export type_is_immediate;
export type_is_sequence;
export type_is_signed;
export type_is_structural;
export type_is_copyable;
export type_is_tup_like;
export type_is_str;
export type_is_unique;
export type_structurally_contains_uniques;
export type_autoderef;
export type_param;
export unify;
export variant_info;
export walk_ty;
export occurs_check_fails;
export closure_kind;
export closure_block;
export closure_shared;
export closure_send;
export param_bound, param_bounds, bound_copy, bound_send, bound_iface;
export param_bounds_to_kind;

// Data types

type arg = {mode: mode, ty: t};

type field = {ident: ast::ident, mt: mt};

type param_bounds = @[param_bound];

type method = {ident: ast::ident, tps: @[param_bounds], fty: fn_ty};

type constr_table = hashmap<ast::node_id, [constr]>;

type mt = {ty: t, mut: ast::mutability};


// Contains information needed to resolve types and (in the future) look up
// the types of AST nodes.
type creader_cache = hashmap<{cnum: int, pos: uint, len: uint}, ty::t>;

type ctxt =
    @{ts: @type_store,
      sess: session::session,
      def_map: resolve::def_map,
      node_types: node_type_table,
      items: ast_map::map,
      freevars: freevars::freevar_map,
      tcache: type_cache,
      rcache: creader_cache,
      short_names_cache: hashmap<t, @str>,
      needs_drop_cache: hashmap<t, bool>,
      kind_cache: hashmap<t, kind>,
      ast_ty_to_ty_cache: hashmap<@ast::ty, option::t<t>>,
      tag_var_cache: hashmap<def_id, @[variant_info]>,
      iface_method_cache: hashmap<def_id, @[method]>,
      ty_param_bounds: hashmap<ast::node_id, param_bounds>};

type ty_ctxt = ctxt;

// Never construct these manually. These are interned.
type raw_t = {struct: sty,
              hash: uint,
              has_params: bool,
              has_vars: bool};

type t = uint;

tag closure_kind {
    closure_block;
    closure_shared;
    closure_send;
}

type fn_ty = {proto: ast::proto,
              inputs: [arg],
              output: t,
              ret_style: ret_style,
              constraints: [@constr]};

// NB: If you change this, you'll probably want to change the corresponding
// AST structure in front/ast::rs as well.
tag sty {
    ty_nil;
    ty_bot;
    ty_bool;
    ty_int(ast::int_ty);
    ty_uint(ast::uint_ty);
    ty_float(ast::float_ty);
    ty_str;
    ty_tag(def_id, [t]);
    ty_box(mt);
    ty_uniq(mt);
    ty_vec(mt);
    ty_ptr(mt);
    ty_rec([field]);
    ty_fn(fn_ty);
    ty_native_fn([arg], t);
    ty_obj([method]);
    ty_iface(def_id, [t]);
    ty_res(def_id, t, [t]);
    ty_tup([t]);
    ty_var(int); // type variable

    ty_param(uint, def_id); // fn/tag type param

    ty_type; // type_desc*
    ty_send_type; // type_desc* that has been cloned into exchange heap
    ty_native(def_id);
    ty_constr(t, [@type_constr]);
    ty_opaque_closure_ptr(closure_kind); // ptr to env for fn, fn@, fn~
    ty_named(t, @str);
}

// In the middle end, constraints have a def_id attached, referring
// to the definition of the operator in the constraint.
type constr_general<ARG> = spanned<constr_general_<ARG, def_id>>;
type type_constr = constr_general<@path>;
type constr = constr_general<uint>;

// Data structures used in type unification
tag type_err {
    terr_mismatch;
    terr_ret_style_mismatch(ast::ret_style, ast::ret_style);
    terr_box_mutability;
    terr_vec_mutability;
    terr_tuple_size(uint, uint);
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

tag param_bound {
    bound_copy;
    bound_send;
    bound_iface(t);
}

fn param_bounds_to_kind(bounds: param_bounds) -> kind {
    let kind = kind_noncopyable;
    for bound in *bounds {
        alt bound {
          bound_copy. {
            if kind != kind_sendable { kind = kind_copyable; }
          }
          bound_send. { kind = kind_sendable; }
          _ {}
        }
    }
    kind
}

type ty_param_bounds_and_ty = {bounds: @[param_bounds], ty: t};

type type_cache = hashmap<ast::def_id, ty_param_bounds_and_ty>;

const idx_nil: uint = 0u;

const idx_bool: uint = 1u;

const idx_int: uint = 2u;

const idx_float: uint = 3u;

const idx_uint: uint = 4u;

const idx_i8: uint = 5u;

const idx_i16: uint = 6u;

const idx_i32: uint = 7u;

const idx_i64: uint = 8u;

const idx_u8: uint = 9u;

const idx_u16: uint = 10u;

const idx_u32: uint = 11u;

const idx_u64: uint = 12u;

const idx_f32: uint = 13u;

const idx_f64: uint = 14u;

const idx_char: uint = 15u;

const idx_str: uint = 16u;

const idx_type: uint = 17u;

const idx_send_type: uint = 18u;

const idx_bot: uint = 19u;

const idx_first_others: uint = 20u;

type type_store = interner::interner<@raw_t>;

type ty_param_substs_opt_and_ty = {substs: option::t<[ty::t]>, ty: ty::t};

type node_type_table =
    @smallintmap::smallintmap<ty::ty_param_substs_opt_and_ty>;

fn populate_type_store(cx: ctxt) {
    intern(cx, ty_nil);
    intern(cx, ty_bool);
    intern(cx, ty_int(ast::ty_i));
    intern(cx, ty_float(ast::ty_f));
    intern(cx, ty_uint(ast::ty_u));
    intern(cx, ty_int(ast::ty_i8));
    intern(cx, ty_int(ast::ty_i16));
    intern(cx, ty_int(ast::ty_i32));
    intern(cx, ty_int(ast::ty_i64));
    intern(cx, ty_uint(ast::ty_u8));
    intern(cx, ty_uint(ast::ty_u16));
    intern(cx, ty_uint(ast::ty_u32));
    intern(cx, ty_uint(ast::ty_u64));
    intern(cx, ty_float(ast::ty_f32));
    intern(cx, ty_float(ast::ty_f64));
    intern(cx, ty_int(ast::ty_char));
    intern(cx, ty_str);
    intern(cx, ty_type);
    intern(cx, ty_send_type);
    intern(cx, ty_bot);
    assert (vec::len(cx.ts.vect) == idx_first_others);
}

fn mk_rcache() -> creader_cache {
    type val = {cnum: int, pos: uint, len: uint};
    fn hash_cache_entry(k: val) -> uint {
        ret (k.cnum as uint) + k.pos + k.len;
    }
    fn eq_cache_entries(a: val, b: val) -> bool {
        ret a.cnum == b.cnum && a.pos == b.pos && a.len == b.len;
    }
    ret map::mk_hashmap(hash_cache_entry, eq_cache_entries);
}

fn new_ty_hash<V: copy>() -> map::hashmap<t, V> { map::new_uint_hash() }

fn mk_ctxt(s: session::session, dm: resolve::def_map, amap: ast_map::map,
           freevars: freevars::freevar_map) -> ctxt {
    let ntt: node_type_table =
        @smallintmap::mk::<ty::ty_param_substs_opt_and_ty>();
    fn eq_raw_ty(&&a: @raw_t, &&b: @raw_t) -> bool {
        ret a.hash == b.hash && a.struct == b.struct;
    }
    let ts = @interner::mk::<@raw_t>(hash_raw_ty, eq_raw_ty);
    let cx =
        @{ts: ts,
          sess: s,
          def_map: dm,
          node_types: ntt,
          items: amap,
          freevars: freevars,
          tcache: new_def_hash(),
          rcache: mk_rcache(),
          short_names_cache: new_ty_hash(),
          needs_drop_cache: new_ty_hash(),
          kind_cache: new_ty_hash(),
          ast_ty_to_ty_cache:
              map::mk_hashmap(ast_util::hash_ty, ast_util::eq_ty),
          tag_var_cache: new_def_hash(),
          iface_method_cache: new_def_hash(),
          ty_param_bounds: map::new_int_hash()};
    populate_type_store(cx);
    ret cx;
}


// Type constructors
fn mk_raw_ty(cx: ctxt, st: sty) -> @raw_t {
    let h = hash_type_structure(st);
    let has_params: bool = false;
    let has_vars: bool = false;
    fn derive_flags_t(cx: ctxt, &has_params: bool, &has_vars: bool, tt: t) {
        let rt = interner::get::<@raw_t>(*cx.ts, tt);
        has_params = has_params || rt.has_params;
        has_vars = has_vars || rt.has_vars;
    }
    fn derive_flags_mt(cx: ctxt, &has_params: bool, &has_vars: bool, m: mt) {
        derive_flags_t(cx, has_params, has_vars, m.ty);
    }
    fn derive_flags_arg(cx: ctxt, &has_params: bool, &has_vars: bool,
                        a: arg) {
        derive_flags_t(cx, has_params, has_vars, a.ty);
    }
    fn derive_flags_sig(cx: ctxt, &has_params: bool, &has_vars: bool,
                        args: [arg], tt: t) {
        for a: arg in args { derive_flags_arg(cx, has_params, has_vars, a); }
        derive_flags_t(cx, has_params, has_vars, tt);
    }
    alt st {
      ty_nil. | ty_bot. | ty_bool. | ty_int(_) | ty_float(_) | ty_uint(_) |
      ty_str. | ty_send_type. | ty_type. | ty_native(_) |
      ty_opaque_closure_ptr(_) {
        /* no-op */
      }
      ty_param(_, _) { has_params = true; }
      ty_var(_) { has_vars = true; }
      ty_tag(_, tys) | ty_iface(_, tys) {
        for tt: t in tys { derive_flags_t(cx, has_params, has_vars, tt); }
      }
      ty_box(m) { derive_flags_mt(cx, has_params, has_vars, m); }
      ty_uniq(m) { derive_flags_mt(cx, has_params, has_vars, m); }
      ty_vec(m) { derive_flags_mt(cx, has_params, has_vars, m); }
      ty_ptr(m) { derive_flags_mt(cx, has_params, has_vars, m); }
      ty_rec(flds) {
        for f: field in flds {
            derive_flags_mt(cx, has_params, has_vars, f.mt);
        }
      }
      ty_tup(ts) {
        for tt in ts { derive_flags_t(cx, has_params, has_vars, tt); }
      }
      ty_fn(f) {
        derive_flags_sig(cx, has_params, has_vars, f.inputs, f.output);
      }
      ty_native_fn(args, tt) {
        derive_flags_sig(cx, has_params, has_vars, args, tt);
      }
      ty_obj(meths) {
        for m: method in meths {
            derive_flags_sig(cx, has_params, has_vars, m.fty.inputs,
                             m.fty.output);
        }
      }
      ty_res(_, tt, tps) {
        derive_flags_t(cx, has_params, has_vars, tt);
        for tt: t in tps { derive_flags_t(cx, has_params, has_vars, tt); }
      }
      ty_constr(tt, _) | ty_named(tt, _) {
        derive_flags_t(cx, has_params, has_vars, tt);
      }
    }
    ret @{struct: st,
          hash: h,
          has_params: has_params,
          has_vars: has_vars};
}

fn intern(cx: ctxt, st: sty) {
    interner::intern(*cx.ts, mk_raw_ty(cx, st));
}

// These are private constructors to this module. External users should always
// use the mk_foo() functions below.
fn gen_ty(cx: ctxt, st: sty) -> t {
    let raw_type = mk_raw_ty(cx, st);
    ret interner::intern(*cx.ts, raw_type);
}

fn mk_nil(_cx: ctxt) -> t { ret idx_nil; }

fn mk_bot(_cx: ctxt) -> t { ret idx_bot; }

fn mk_bool(_cx: ctxt) -> t { ret idx_bool; }

fn mk_int(_cx: ctxt) -> t { ret idx_int; }

fn mk_float(_cx: ctxt) -> t { ret idx_float; }

fn mk_uint(_cx: ctxt) -> t { ret idx_uint; }

fn mk_mach_int(_cx: ctxt, tm: ast::int_ty) -> t {
    alt tm {
      ast::ty_i. { ret idx_int; }
      ast::ty_char. { ret idx_char; }
      ast::ty_i8. { ret idx_i8; }
      ast::ty_i16. { ret idx_i16; }
      ast::ty_i32. { ret idx_i32; }
      ast::ty_i64. { ret idx_i64; }
    }
}

fn mk_mach_uint(_cx: ctxt, tm: ast::uint_ty) -> t {
    alt tm {
      ast::ty_u. { ret idx_uint; }
      ast::ty_u8. { ret idx_u8; }
      ast::ty_u16. { ret idx_u16; }
      ast::ty_u32. { ret idx_u32; }
      ast::ty_u64. { ret idx_u64; }
    }
}

fn mk_mach_float(_cx: ctxt, tm: ast::float_ty) -> t {
    alt tm {
      ast::ty_f. { ret idx_float; }
      ast::ty_f32. { ret idx_f32; }
      ast::ty_f64. { ret idx_f64; }
    }
}


fn mk_char(_cx: ctxt) -> t { ret idx_char; }

fn mk_str(_cx: ctxt) -> t { ret idx_str; }

fn mk_tag(cx: ctxt, did: ast::def_id, tys: [t]) -> t {
    ret gen_ty(cx, ty_tag(did, tys));
}

fn mk_box(cx: ctxt, tm: mt) -> t { ret gen_ty(cx, ty_box(tm)); }

fn mk_uniq(cx: ctxt, tm: mt) -> t { ret gen_ty(cx, ty_uniq(tm)); }

fn mk_imm_uniq(cx: ctxt, ty: t) -> t {
    ret mk_uniq(cx, {ty: ty, mut: ast::imm});
}

fn mk_ptr(cx: ctxt, tm: mt) -> t { ret gen_ty(cx, ty_ptr(tm)); }

fn mk_imm_box(cx: ctxt, ty: t) -> t {
    ret mk_box(cx, {ty: ty, mut: ast::imm});
}

fn mk_mut_ptr(cx: ctxt, ty: t) -> t {
    ret mk_ptr(cx, {ty: ty, mut: ast::mut});
}

fn mk_vec(cx: ctxt, tm: mt) -> t { ret gen_ty(cx, ty_vec(tm)); }

fn mk_rec(cx: ctxt, fs: [field]) -> t { ret gen_ty(cx, ty_rec(fs)); }

fn mk_constr(cx: ctxt, t: t, cs: [@type_constr]) -> t {
    ret gen_ty(cx, ty_constr(t, cs));
}

fn mk_tup(cx: ctxt, ts: [t]) -> t { ret gen_ty(cx, ty_tup(ts)); }

fn mk_fn(cx: ctxt, fty: fn_ty) -> t {
    ret gen_ty(cx, ty_fn(fty));
}

fn mk_native_fn(cx: ctxt, args: [arg], ty: t) -> t {
    ret gen_ty(cx, ty_native_fn(args, ty));
}

fn mk_obj(cx: ctxt, meths: [method]) -> t { ret gen_ty(cx, ty_obj(meths)); }

fn mk_iface(cx: ctxt, did: ast::def_id, tys: [t]) -> t {
    ret gen_ty(cx, ty_iface(did, tys));
}

fn mk_res(cx: ctxt, did: ast::def_id, inner: t, tps: [t]) -> t {
    ret gen_ty(cx, ty_res(did, inner, tps));
}

fn mk_var(cx: ctxt, v: int) -> t { ret gen_ty(cx, ty_var(v)); }

fn mk_param(cx: ctxt, n: uint, k: def_id) -> t {
    ret gen_ty(cx, ty_param(n, k));
}

fn mk_type(_cx: ctxt) -> t { ret idx_type; }

fn mk_send_type(_cx: ctxt) -> t { ret idx_send_type; }

fn mk_native(cx: ctxt, did: def_id) -> t { ret gen_ty(cx, ty_native(did)); }

fn mk_opaque_closure_ptr(cx: ctxt, ck: closure_kind) -> t {
    ret gen_ty(cx, ty_opaque_closure_ptr(ck));
}

fn mk_named(cx: ctxt, base: t, name: @str) -> t {
    gen_ty(cx, ty_named(base, name))
}

// Returns the one-level-deep type structure of the given type.
pure fn struct(cx: ctxt, typ: t) -> sty {
    alt interner::get(*cx.ts, typ).struct {
      ty_named(t, _) { struct(cx, t) }
      s { s }
    }
}

// Returns struact(cx, typ) but replaces all occurences of platform
// dependent primitive types with their machine type equivalent
pure fn mach_struct(cx: ctxt, cfg: @session::config, typ: t) -> sty {
    alt interner::get(*cx.ts, typ).struct {
      ty_named(t, _) { mach_struct(cx, cfg, t) }
      s { mach_sty(cfg, s) }
    }
}

// Converts s to its machine type equivalent
pure fn mach_sty(cfg: @session::config, s: sty) -> sty {
    alt s {
      ty_int(ast::ty_i.) { ty_int(cfg.int_type) }
      ty_uint(ast::ty_u.) { ty_uint(cfg.uint_type) }
      ty_float(ast::ty_f.) { ty_float(cfg.float_type) }
      s { s }
    }
}

pure fn ty_name(cx: ctxt, typ: t) -> option::t<@str> {
    alt interner::get(*cx.ts, typ).struct {
      ty_named(_, n) { some(n) }
      _ { none }
    }
}


// Type folds
type ty_walk = fn@(t);

fn walk_ty(cx: ctxt, walker: ty_walk, ty: t) {
    alt struct(cx, ty) {
      ty_nil. | ty_bot. | ty_bool. | ty_int(_) | ty_uint(_) | ty_float(_) |
      ty_str. | ty_send_type. | ty_type. | ty_native(_) |
      ty_opaque_closure_ptr(_) {
        /* no-op */
      }
      ty_box(tm) | ty_vec(tm) | ty_ptr(tm) { walk_ty(cx, walker, tm.ty); }
      ty_tag(_, subtys) | ty_iface(_, subtys) {
        for subty: t in subtys { walk_ty(cx, walker, subty); }
      }
      ty_rec(fields) {
        for fl: field in fields { walk_ty(cx, walker, fl.mt.ty); }
      }
      ty_tup(ts) { for tt in ts { walk_ty(cx, walker, tt); } }
      ty_fn(f) {
        for a: arg in f.inputs { walk_ty(cx, walker, a.ty); }
        walk_ty(cx, walker, f.output);
      }
      ty_native_fn(args, ret_ty) {
        for a: arg in args { walk_ty(cx, walker, a.ty); }
        walk_ty(cx, walker, ret_ty);
      }
      ty_obj(methods) {
        for m: method in methods {
            for a: arg in m.fty.inputs { walk_ty(cx, walker, a.ty); }
            walk_ty(cx, walker, m.fty.output);
        }
      }
      ty_res(_, sub, tps) {
        walk_ty(cx, walker, sub);
        for tp: t in tps { walk_ty(cx, walker, tp); }
      }
      ty_constr(sub, _) { walk_ty(cx, walker, sub); }
      ty_var(_) {/* no-op */ }
      ty_param(_, _) {/* no-op */ }
      ty_uniq(tm) { walk_ty(cx, walker, tm.ty); }
    }
    walker(ty);
}

tag fold_mode {
    fm_var(fn@(int) -> t);
    fm_param(fn@(uint, def_id) -> t);
    fm_general(fn@(t) -> t);
}

fn fold_ty(cx: ctxt, fld: fold_mode, ty_0: t) -> t {
    let ty = ty_0;
    // Fast paths.

    alt fld {
      fm_var(_) { if !type_contains_vars(cx, ty) { ret ty; } }
      fm_param(_) { if !type_contains_params(cx, ty) { ret ty; } }
      fm_general(_) {/* no fast path */ }
    }
    alt interner::get(*cx.ts, ty).struct {
      ty_nil. | ty_bot. | ty_bool. | ty_int(_) | ty_uint(_) | ty_float(_) |
      ty_str. | ty_send_type. | ty_type. | ty_native(_) |
      ty_opaque_closure_ptr(_) {
        /* no-op */
      }
      ty_box(tm) {
        ty = mk_box(cx, {ty: fold_ty(cx, fld, tm.ty), mut: tm.mut});
      }
      ty_uniq(tm) {
        ty = mk_uniq(cx, {ty: fold_ty(cx, fld, tm.ty), mut: tm.mut});
      }
      ty_named(t, nm) {
        ty = mk_named(cx, fold_ty(cx, fld, t), nm);
      }
      ty_ptr(tm) {
        ty = mk_ptr(cx, {ty: fold_ty(cx, fld, tm.ty), mut: tm.mut});
      }
      ty_vec(tm) {
        ty = mk_vec(cx, {ty: fold_ty(cx, fld, tm.ty), mut: tm.mut});
      }
      ty_tag(tid, subtys) {
        ty = mk_tag(cx, tid, vec::map(subtys, {|t| fold_ty(cx, fld, t) }));
      }
      ty_iface(did, subtys) {
        ty = mk_iface(cx, did, vec::map(subtys, {|t| fold_ty(cx, fld, t) }));
      }
      ty_rec(fields) {
        let new_fields: [field] = [];
        for fl: field in fields {
            let new_ty = fold_ty(cx, fld, fl.mt.ty);
            let new_mt = {ty: new_ty, mut: fl.mt.mut};
            new_fields += [{ident: fl.ident, mt: new_mt}];
        }
        ty = mk_rec(cx, new_fields);
      }
      ty_tup(ts) {
        let new_ts = [];
        for tt in ts { new_ts += [fold_ty(cx, fld, tt)]; }
        ty = mk_tup(cx, new_ts);
      }
      ty_fn(f) {
        let new_args: [arg] = [];
        for a: arg in f.inputs {
            let new_ty = fold_ty(cx, fld, a.ty);
            new_args += [{mode: a.mode, ty: new_ty}];
        }
        ty = mk_fn(cx, {inputs: new_args,
                        output: fold_ty(cx, fld, f.output)
                        with f});
      }
      ty_native_fn(args, ret_ty) {
        let new_args: [arg] = [];
        for a: arg in args {
            let new_ty = fold_ty(cx, fld, a.ty);
            new_args += [{mode: a.mode, ty: new_ty}];
        }
        ty = mk_native_fn(cx, new_args, fold_ty(cx, fld, ret_ty));
      }
      ty_obj(methods) {
        let new_methods = vec::map(methods, {|m|
            let new_args = vec::map(m.fty.inputs, {|a|
                {mode: a.mode, ty: fold_ty(cx, fld, a.ty)}
            });
            {ident: m.ident, tps: m.tps,
             fty: {inputs: new_args,
                   output: fold_ty(cx, fld, m.fty.output)
                   with m.fty}}
        });
        ty = mk_obj(cx, new_methods);
      }
      ty_res(did, subty, tps) {
        let new_tps = [];
        for tp: t in tps { new_tps += [fold_ty(cx, fld, tp)]; }
        ty = mk_res(cx, did, fold_ty(cx, fld, subty), new_tps);
      }
      ty_var(id) {
        alt fld { fm_var(folder) { ty = folder(id); } _ {/* no-op */ } }
      }
      ty_param(id, did) {
        alt fld { fm_param(folder) { ty = folder(id, did); } _ {} }
      }
      ty_constr(subty, cs) {
          ty = mk_constr(cx, fold_ty(cx, fld, subty), cs);
      }
      _ {
          cx.sess.fatal("Unsupported sort of type in fold_ty");
      }
    }

    // If this is a general type fold, then we need to run it now.
    alt fld { fm_general(folder) { ret folder(ty); } _ { ret ty; } }
}


// Type utilities

fn type_is_nil(cx: ctxt, ty: t) -> bool {
    alt struct(cx, ty) { ty_nil. { ret true; } _ { ret false; } }
}

fn type_is_bot(cx: ctxt, ty: t) -> bool {
    alt struct(cx, ty) { ty_bot. { ret true; } _ { ret false; } }
}

fn type_is_bool(cx: ctxt, ty: t) -> bool {
    alt struct(cx, ty) { ty_bool. { ret true; } _ { ret false; } }
}

fn type_is_structural(cx: ctxt, ty: t) -> bool {
    alt struct(cx, ty) {
      ty_rec(_) | ty_tup(_) | ty_tag(_, _) | ty_fn(_) |
      ty_native_fn(_, _) | ty_obj(_) | ty_res(_, _, _) |
      ty_iface(_, _) { true }
      _ { false }
    }
}

fn type_is_copyable(cx: ctxt, ty: t) -> bool {
    ret kind_can_be_copied(type_kind(cx, ty));
}

fn type_is_sequence(cx: ctxt, ty: t) -> bool {
    alt struct(cx, ty) {
      ty_str. { ret true; }
      ty_vec(_) { ret true; }
      _ { ret false; }
    }
}

fn type_is_str(cx: ctxt, ty: t) -> bool {
    alt struct(cx, ty) { ty_str. { ret true; } _ { ret false; } }
}

fn sequence_element_type(cx: ctxt, ty: t) -> t {
    alt struct(cx, ty) {
      ty_str. { ret mk_mach_uint(cx, ast::ty_u8); }
      ty_vec(mt) { ret mt.ty; }
      _ { cx.sess.bug("sequence_element_type called on non-sequence value"); }
    }
}

pure fn type_is_tup_like(cx: ctxt, ty: t) -> bool {
    let sty = struct(cx, ty);
    alt sty {
      ty_ptr(_) | ty_uniq(_) |
      ty_box(_) | ty_rec(_) | ty_tup(_) | ty_tag(_,_) { true }
      _ { false }
    }
}

fn get_element_type(cx: ctxt, ty: t, i: uint) -> t {
    alt struct(cx, ty) {
      ty_rec(flds) { ret flds[i].mt.ty; }
      ty_tup(ts) { ret ts[i]; }
      _ {
        cx.sess.bug("get_element_type called on type " + ty_to_str(cx, ty) +
                        " - expected a \
            tuple or record");
      }
    }
    // NB: This is not exhaustive -- struct(cx, ty) could be a box or a
    // tag.
}

pure fn type_is_box(cx: ctxt, ty: t) -> bool {
    alt struct(cx, ty) {
      ty_box(_) { ret true; }
      _ { ret false; }
    }
}

pure fn type_is_boxed(cx: ctxt, ty: t) -> bool {
    alt struct(cx, ty) {
      ty_box(_) | ty_iface(_, _) { ret true; }
      _ { ret false; }
    }
}

pure fn type_is_unique_box(cx: ctxt, ty: t) -> bool {
    alt struct(cx, ty) {
      ty_uniq(_) { ret true; }
      _ { ret false; }
    }
}

pure fn type_is_unsafe_ptr(cx: ctxt, ty: t) -> bool {
    alt struct(cx, ty) {
      ty_ptr(_) { ret true; }
      _ { ret false; }
    }
}

pure fn type_is_vec(cx: ctxt, ty: t) -> bool {
    ret alt struct(cx, ty) {
          ty_vec(_) { true }
          ty_str. { true }
          _ { false }
        };
}

pure fn type_is_unique(cx: ctxt, ty: t) -> bool {
    alt struct(cx, ty) {
      ty_uniq(_) { ret true; }
      ty_vec(_) { true }
      ty_str. { true }
      _ { ret false; }
    }
}

pure fn type_is_scalar(cx: ctxt, ty: t) -> bool {
    alt struct(cx, ty) {
      ty_nil. | ty_bool. | ty_int(_) | ty_float(_) | ty_uint(_) |
      ty_send_type. | ty_type. | ty_native(_) | ty_ptr(_) { true }
      _ { false }
    }
}

// FIXME maybe inline this for speed?
fn type_is_immediate(cx: ctxt, ty: t) -> bool {
    ret type_is_scalar(cx, ty) || type_is_boxed(cx, ty) ||
        type_is_unique(cx, ty) || type_is_native(cx, ty);
}

fn type_needs_drop(cx: ctxt, ty: t) -> bool {
    alt cx.needs_drop_cache.find(ty) {
      some(result) { ret result; }
      none. {/* fall through */ }
    }

    let accum = false;
    let result = alt struct(cx, ty) {
      // scalar types
      ty_nil. | ty_bot. | ty_bool. | ty_int(_) | ty_float(_) | ty_uint(_) |
      ty_type. | ty_native(_) | ty_ptr(_) { false }
      ty_rec(flds) {
        for f in flds { if type_needs_drop(cx, f.mt.ty) { accum = true; } }
        accum
      }
      ty_tup(elts) {
        for m in elts { if type_needs_drop(cx, m) { accum = true; } }
        accum
      }
      ty_tag(did, tps) {
        let variants = tag_variants(cx, did);
        for variant in *variants {
            for aty in variant.args {
                // Perform any type parameter substitutions.
                let arg_ty = substitute_type_params(cx, tps, aty);
                if type_needs_drop(cx, arg_ty) { accum = true; }
            }
            if accum { break; }
        }
        accum
      }
      _ { true }
    };

    cx.needs_drop_cache.insert(ty, result);
    ret result;
}

tag kind { kind_sendable; kind_copyable; kind_noncopyable; }

// Using these query functons is preferable to direct comparison or matching
// against the kind constants, as we may modify the kind hierarchy in the
// future.
pure fn kind_can_be_copied(k: kind) -> bool {
    ret alt k {
      kind_sendable. { true }
      kind_copyable. { true }
      kind_noncopyable. { false }
    };
}

pure fn kind_can_be_sent(k: kind) -> bool {
    ret alt k {
      kind_sendable. { true }
      kind_copyable. { false }
      kind_noncopyable. { false }
    };
}

fn proto_kind(p: proto) -> kind {
    alt p {
      ast::proto_block. { kind_noncopyable }
      ast::proto_shared(_) { kind_copyable }
      ast::proto_send. { kind_sendable }
      ast::proto_bare. { kind_sendable }
    }
}

fn kind_lteq(a: kind, b: kind) -> bool {
    alt a {
      kind_noncopyable. { true }
      kind_copyable. { b != kind_noncopyable }
      kind_sendable. { b == kind_sendable }
    }
}

fn lower_kind(a: kind, b: kind) -> kind {
    if ty::kind_lteq(a, b) { a } else { b }
}

fn type_kind(cx: ctxt, ty: t) -> kind {
    alt cx.kind_cache.find(ty) {
      some(result) { ret result; }
      none. {/* fall through */ }
    }

    // Insert a default in case we loop back on self recursively.
    cx.kind_cache.insert(ty, kind_sendable);

    let result = alt struct(cx, ty) {
      // Scalar and unique types are sendable
      ty_nil. | ty_bot. | ty_bool. | ty_int(_) | ty_uint(_) | ty_float(_) |
      ty_native(_) | ty_ptr(_) |
      ty_send_type. | ty_str. | ty_native_fn(_, _) { kind_sendable }
      ty_type. { kind_copyable }
      // FIXME: obj is broken for now, since we aren't asserting
      // anything about its fields.
      ty_obj(_) { kind_copyable }
      ty_fn(f) { proto_kind(f.proto) }
      ty_opaque_closure_ptr(closure_block.) { kind_noncopyable }
      ty_opaque_closure_ptr(closure_shared.) { kind_copyable }
      ty_opaque_closure_ptr(closure_send.) { kind_sendable }
      // Those with refcounts-to-inner raise pinned to shared,
      // lower unique to shared. Therefore just set result to shared.
      ty_box(_) | ty_iface(_, _) { kind_copyable }
      // Boxes and unique pointers raise pinned to shared.
      ty_vec(tm) | ty_uniq(tm) { type_kind(cx, tm.ty) }
      // Records lower to the lowest of their members.
      ty_rec(flds) {
        let lowest = kind_sendable;
        for f in flds { lowest = lower_kind(lowest, type_kind(cx, f.mt.ty)); }
        lowest
      }
      // Tuples lower to the lowest of their members.
      ty_tup(tys) {
        let lowest = kind_sendable;
        for ty in tys { lowest = lower_kind(lowest, type_kind(cx, ty)); }
        lowest
      }
      // Tags lower to the lowest of their variants.
      ty_tag(did, tps) {
        let lowest = kind_sendable;
        for variant in *tag_variants(cx, did) {
            for aty in variant.args {
                // Perform any type parameter substitutions.
                let arg_ty = substitute_type_params(cx, tps, aty);
                lowest = lower_kind(lowest, type_kind(cx, arg_ty));
                if lowest == kind_noncopyable { break; }
            }
        }
        lowest
      }
      // Resources are always noncopyable.
      ty_res(did, inner, tps) { kind_noncopyable }
      ty_param(_, did) {
          param_bounds_to_kind(cx.ty_param_bounds.get(did.node))
      }
      ty_constr(t, _) { type_kind(cx, t) }
    };

    cx.kind_cache.insert(ty, result);
    ret result;
}

// FIXME: should we just return true for native types in
// type_is_scalar?
fn type_is_native(cx: ctxt, ty: t) -> bool {
    alt struct(cx, ty) { ty_native(_) { ret true; } _ { ret false; } }
}

fn type_structurally_contains(cx: ctxt, ty: t, test: fn(sty) -> bool) ->
   bool {
    let sty = struct(cx, ty);
    if test(sty) { ret true; }
    alt sty {
      ty_tag(did, tps) {
        for variant in *tag_variants(cx, did) {
            for aty in variant.args {
                let sty = substitute_type_params(cx, tps, aty);
                if type_structurally_contains(cx, sty, test) { ret true; }
            }
        }
        ret false;
      }
      ty_rec(fields) {
        for field in fields {
            if type_structurally_contains(cx, field.mt.ty, test) { ret true; }
        }
        ret false;
      }
      ty_tup(ts) {
        for tt in ts {
            if type_structurally_contains(cx, tt, test) { ret true; }
        }
        ret false;
      }
      ty_res(_, sub, tps) {
        let sty = substitute_type_params(cx, tps, sub);
        ret type_structurally_contains(cx, sty, test);
      }
      _ { ret false; }
    }
}

pure fn type_has_dynamic_size(cx: ctxt, ty: t) -> bool unchecked {

    /* type_structurally_contains can't be declared pure
    because it takes a function argument. But it should be
    referentially transparent, since a given type's size should
    never change once it's created.
    (It would be interesting to think about how to make such properties
    actually checkable. It seems to me like a lot of properties
    that the type context tracks about types should be immutable.)
    */
    type_structurally_contains(cx, ty, fn (sty: sty) -> bool {
        alt sty {
          ty_param(_, _) { true }
          _ { false }
        }
    })
}

// Returns true for noncopyable types and types where a copy of a value can be
// distinguished from the value itself. I.e. types with mutable content that's
// not shared through a pointer.
fn type_allows_implicit_copy(cx: ctxt, ty: t) -> bool {
    ret !type_structurally_contains(cx, ty, fn (sty: sty) -> bool {
        ret alt sty {
          ty_param(_, _) { true }
          ty_vec(mt) {
            mt.mut != ast::imm
          }
          ty_rec(fields) {
            for field in fields {
                if field.mt.mut !=
                    ast::imm {
                    ret true;
                }
            }
            false
          }
          _ { false }
        };
    }) && type_kind(cx, ty) != kind_noncopyable;
}

fn type_structurally_contains_uniques(cx: ctxt, ty: t) -> bool {
    ret type_structurally_contains(cx, ty, fn (sty: sty) -> bool {
        ret alt sty {
          ty_uniq(_) { ret true; }
          ty_vec(_) { true }
          ty_str. { true }
          _ { ret false; }
        };
    });
}

fn type_is_integral(cx: ctxt, ty: t) -> bool {
    alt struct(cx, ty) {
      ty_int(_) | ty_uint(_) | ty_bool. { true }
      _ { false }
    }
}

fn type_is_fp(cx: ctxt, ty: t) -> bool {
    alt struct(cx, ty) {
      ty_float(_) { true }
      _ { false }
    }
}

fn type_is_numeric(cx: ctxt, ty: t) -> bool {
    ret type_is_integral(cx, ty) || type_is_fp(cx, ty);
}

fn type_is_signed(cx: ctxt, ty: t) -> bool {
    alt struct(cx, ty) {
      ty_int(_) { true }
      _ { false }
    }
}

// Whether a type is Plain Old Data -- meaning it does not contain pointers
// that the cycle collector might care about.
fn type_is_pod(cx: ctxt, ty: t) -> bool {
    let result = true;
    alt struct(cx, ty) {
      // Scalar types
      ty_nil. | ty_bot. | ty_bool. | ty_int(_) | ty_float(_) | ty_uint(_) |
      ty_send_type. | ty_type. | ty_native(_) | ty_ptr(_) { result = true; }
      // Boxed types
      ty_str. | ty_box(_) | ty_uniq(_) | ty_vec(_) | ty_fn(_) |
      ty_native_fn(_, _) | ty_obj(_) | ty_iface(_, _) { result = false; }
      // Structural types
      ty_tag(did, tps) {
        let variants = tag_variants(cx, did);
        for variant: variant_info in *variants {
            let tup_ty = mk_tup(cx, variant.args);

            // Perform any type parameter substitutions.
            tup_ty = substitute_type_params(cx, tps, tup_ty);
            if !type_is_pod(cx, tup_ty) { result = false; }
        }
      }
      ty_rec(flds) {
        for f: field in flds {
            if !type_is_pod(cx, f.mt.ty) { result = false; }
        }
      }
      ty_tup(elts) {
        for elt in elts { if !type_is_pod(cx, elt) { result = false; } }
      }
      ty_res(_, inner, tps) {
        result = type_is_pod(cx, substitute_type_params(cx, tps, inner));
      }
      ty_constr(subt, _) { result = type_is_pod(cx, subt); }
      ty_var(_) {
        fail "ty_var in type_is_pod";
      }
      ty_param(_, _) { result = false; }
    }

    ret result;
}

fn type_param(cx: ctxt, ty: t) -> option::t<uint> {
    alt struct(cx, ty) {
      ty_param(id, _) { ret some(id); }
      _ {/* fall through */ }
    }
    ret none;
}

// Returns a vec of all the type variables
// occurring in t. It may contain duplicates.
fn vars_in_type(cx: ctxt, ty: t) -> [int] {
    fn collect_var(cx: ctxt, vars: @mutable [int], ty: t) {
        alt struct(cx, ty) { ty_var(v) { *vars += [v]; } _ { } }
    }
    let rslt: @mutable [int] = @mutable [];
    walk_ty(cx, bind collect_var(cx, rslt, _), ty);
    // Works because of a "convenient" bug that lets us
    // return a mutable vec as if it's immutable
    ret *rslt;
}

fn type_autoderef(cx: ctxt, t: ty::t) -> ty::t {
    let t1 = t;
    while true {
        alt struct(cx, t1) {
          ty_box(mt) | ty_uniq(mt) { t1 = mt.ty; }
          ty_res(_, inner, tps) {
            t1 = substitute_type_params(cx, tps, inner);
          }
          ty_tag(did, tps) {
            let variants = tag_variants(cx, did);
            if vec::len(*variants) != 1u || vec::len(variants[0].args) != 1u {
                break;
            }
            t1 = substitute_type_params(cx, tps, variants[0].args[0]);
          }
          _ { break; }
        }
    }
    ret t1;
}

// Type hashing.
fn hash_type_structure(st: sty) -> uint {
    fn hash_uint(id: uint, n: uint) -> uint {
        let h = id;
        h += (h << 5u) + n;
        ret h;
    }
    fn hash_def(id: uint, did: ast::def_id) -> uint {
        let h = id;
        h += (h << 5u) + (did.crate as uint);
        h += (h << 5u) + (did.node as uint);
        ret h;
    }
    fn hash_subty(id: uint, subty: t) -> uint {
        let h = id;
        h += (h << 5u) + subty;
        ret h;
    }
    fn hash_subtys(id: uint, subtys: [t]) -> uint {
        let h = id;
        vec::iter(subtys) { |subty|
            h = hash_subty(h, subty);
        }
        ret h;
    }
    fn hash_type_constr(id: uint, c: @type_constr) -> uint {
        let h = id;
        h += (h << 5u) + hash_def(h, c.node.id);
        ret hash_type_constr_args(h, c.node.args);
    }
    fn hash_type_constr_args(id: uint, args: [@ty_constr_arg]) -> uint {
        let h = id;
        for a: @ty_constr_arg in args {
            alt a.node {
              carg_base. { h += h << 5u; }
              carg_lit(_) {
                // FIXME
                fail "lit args not implemented yet";
              }
              carg_ident(p) {
                // FIXME: Not sure what to do here.
                h += h << 5u;
              }
            }
        }
        ret h;
    }

    fn hash_fn(id: uint, args: [arg], rty: t) -> uint {
        let h = id;
        for a: arg in args { h += (h << 5u) + a.ty; }
        h += (h << 5u) + rty;
        ret h;
    }
    alt st {
      ty_nil. { 0u } ty_bool. { 1u }
      ty_int(t) {
        alt t {
          ast::ty_i. { 2u } ast::ty_char. { 3u } ast::ty_i8. { 4u }
          ast::ty_i16. { 5u } ast::ty_i32. { 6u } ast::ty_i64. { 7u }
        }
      }
      ty_uint(t) {
        alt t {
          ast::ty_u. { 8u } ast::ty_u8. { 9u } ast::ty_u16. { 10u }
          ast::ty_u32. { 11u } ast::ty_u64. { 12u }
        }
      }
      ty_float(t) {
        alt t { ast::ty_f. { 13u } ast::ty_f32. { 14u } ast::ty_f64. { 15u } }
      }
      ty_str. { ret 17u; }
      ty_tag(did, tys) {
        let h = hash_def(18u, did);
        for typ: t in tys { h += (h << 5u) + typ; }
        ret h;
      }
      ty_box(mt) { ret hash_subty(19u, mt.ty); }
      ty_vec(mt) { ret hash_subty(21u, mt.ty); }
      ty_rec(fields) {
        let h = 26u;
        for f: field in fields { h += (h << 5u) + f.mt.ty; }
        ret h;
      }
      ty_tup(ts) { ret hash_subtys(25u, ts); }

      // ???
      ty_fn(f) { ret hash_fn(27u, f.inputs, f.output); }
      ty_native_fn(args, rty) { ret hash_fn(28u, args, rty); }
      ty_obj(methods) {
        let h = 29u;
        for m: method in methods { h += (h << 5u) + str::hash(m.ident); }
        ret h;
      }
      ty_var(v) { ret hash_uint(30u, v as uint); }
      ty_param(pid, _) { ret hash_uint(31u, pid); }
      ty_type. { ret 32u; }
      ty_native(did) { ret hash_def(33u, did); }
      ty_bot. { ret 34u; }
      ty_ptr(mt) { ret hash_subty(35u, mt.ty); }
      ty_res(did, sub, tps) {
        let h = hash_subty(hash_def(18u, did), sub);
        ret hash_subtys(h, tps);
      }
      ty_constr(t, cs) {
        let h = hash_subty(36u, t);
        for c: @type_constr in cs { h += (h << 5u) + hash_type_constr(h, c); }
        ret h;
      }
      ty_uniq(mt) { ret hash_subty(37u, mt.ty); }
      ty_send_type. { ret 38u; }
      ty_named(t, name) { (str::hash(*name) << 5u) + hash_subty(39u, t) }
      ty_iface(did, tys) {
        let h = hash_def(40u, did);
        for typ: t in tys { h = hash_subty(h, typ); }
        ret h;
      }
      ty_opaque_closure_ptr(closure_block.) { ret 41u; }
      ty_opaque_closure_ptr(closure_shared.) { ret 42u; }
      ty_opaque_closure_ptr(closure_send.) { ret 43u; }
    }
}

fn hash_raw_ty(&&rt: @raw_t) -> uint { ret rt.hash; }

fn arg_eq<T>(eq: fn(T, T) -> bool, a: @sp_constr_arg<T>, b: @sp_constr_arg<T>)
   -> bool {
    alt a.node {
      ast::carg_base. {
        alt b.node { ast::carg_base. { ret true; } _ { ret false; } }
      }
      ast::carg_ident(s) {
        alt b.node { ast::carg_ident(t) { ret eq(s, t); } _ { ret false; } }
      }
      ast::carg_lit(l) {
        alt b.node {
          ast::carg_lit(m) { ret ast_util::lit_eq(l, m); } _ { ret false; }
        }
      }
    }
}

fn args_eq<T>(eq: fn(T, T) -> bool, a: [@sp_constr_arg<T>],
              b: [@sp_constr_arg<T>]) -> bool {
    let i: uint = 0u;
    for arg: @sp_constr_arg<T> in a {
        if !arg_eq(eq, arg, b[i]) { ret false; }
        i += 1u;
    }
    ret true;
}

fn constr_eq(c: @constr, d: @constr) -> bool {
    fn eq_int(&&x: uint, &&y: uint) -> bool { ret x == y; }
    ret path_to_str(c.node.path) == path_to_str(d.node.path) &&
            // FIXME: hack
            args_eq(eq_int, c.node.args, d.node.args);
}

fn constrs_eq(cs: [@constr], ds: [@constr]) -> bool {
    if vec::len(cs) != vec::len(ds) { ret false; }
    let i = 0u;
    for c: @constr in cs { if !constr_eq(c, ds[i]) { ret false; } i += 1u; }
    ret true;
}

// Type lookups
fn node_id_to_ty_param_substs_opt_and_ty(cx: ctxt, id: ast::node_id) ->
   ty_param_substs_opt_and_ty {
    // Pull out the node type table.
    alt smallintmap::find(*cx.node_types, id as uint) {
      none. {
        cx.sess.bug("node_id_to_ty_param_substs_opt_and_ty() called on " +
                        "an untyped node (" + int::to_str(id, 10u) +
                        ")");
      }
      some(tpot) { ret tpot; }
    }
}

fn node_id_to_type(cx: ctxt, id: ast::node_id) -> t {
    ret node_id_to_ty_param_substs_opt_and_ty(cx, id).ty;
}

fn node_id_to_type_params(cx: ctxt, id: ast::node_id) -> [t] {
    alt node_id_to_ty_param_substs_opt_and_ty(cx, id).substs {
      none. { ret []; }
      some(tps) { ret tps; }
    }
}

fn node_id_has_type_params(cx: ctxt, id: ast::node_id) -> bool {
    ret vec::len(node_id_to_type_params(cx, id)) > 0u;
}


// Returns a type with type parameter substitutions performed if applicable.
fn ty_param_substs_opt_and_ty_to_monotype(cx: ctxt,
                                          tpot: ty_param_substs_opt_and_ty) ->
   t {
    alt tpot.substs {
      none. { ret tpot.ty; }
      some(tps) { ret substitute_type_params(cx, tps, tpot.ty); }
    }
}


// Returns the type of an annotation, with type parameter substitutions
// performed if applicable.
fn node_id_to_monotype(cx: ctxt, id: ast::node_id) -> t {
    let tpot = node_id_to_ty_param_substs_opt_and_ty(cx, id);
    ret ty_param_substs_opt_and_ty_to_monotype(cx, tpot);
}


// Returns the number of distinct type parameters in the given type.
fn count_ty_params(cx: ctxt, ty: t) -> uint {
    fn counter(cx: ctxt, param_indices: @mutable [uint], ty: t) {
        alt struct(cx, ty) {
          ty_param(param_idx, _) {
            let seen = false;
            for other_param_idx: uint in *param_indices {
                if param_idx == other_param_idx { seen = true; }
            }
            if !seen { *param_indices += [param_idx]; }
          }
          _ {/* fall through */ }
        }
    }
    let param_indices: @mutable [uint] = @mutable [];
    let f = bind counter(cx, param_indices, _);
    walk_ty(cx, f, ty);
    ret vec::len::<uint>(*param_indices);
}

fn type_contains_vars(cx: ctxt, typ: t) -> bool {
    ret interner::get(*cx.ts, typ).has_vars;
}

fn type_contains_params(cx: ctxt, typ: t) -> bool {
    ret interner::get(*cx.ts, typ).has_params;
}


// Type accessors for substructures of types
fn ty_fn_args(cx: ctxt, fty: t) -> [arg] {
    alt struct(cx, fty) {
      ty::ty_fn(f) { ret f.inputs; }
      ty::ty_native_fn(a, _) { ret a; }
      _ { cx.sess.bug("ty_fn_args() called on non-fn type"); }
    }
}

fn ty_fn_proto(cx: ctxt, fty: t) -> ast::proto {
    alt struct(cx, fty) {
      ty::ty_fn(f) { ret f.proto; }
      ty::ty_native_fn(_, _) {
        // FIXME: This should probably be proto_bare
        ret ast::proto_shared(ast::sugar_normal);
      }
      _ { cx.sess.bug("ty_fn_proto() called on non-fn type"); }
    }
}

pure fn ty_fn_ret(cx: ctxt, fty: t) -> t {
    let sty = struct(cx, fty);
    alt sty {
      ty::ty_fn(f) { ret f.output; }
      ty::ty_native_fn(_, r) { ret r; }
      _ {
        // Unchecked is ok since we diverge here
        // (might want to change the typechecker to allow
        // it without an unchecked)
        // Or, it wouldn't be necessary if we had the right
        // typestate constraint on cx and t (then we could
        // call unreachable() instead)
        unchecked { cx.sess.bug("ty_fn_ret() called on non-fn type"); }}
    }
}

fn ty_fn_ret_style(cx: ctxt, fty: t) -> ast::ret_style {
    alt struct(cx, fty) {
      ty::ty_fn(f) { f.ret_style }
      ty::ty_native_fn(_, _) { ast::return_val }
      _ { cx.sess.bug("ty_fn_ret_style() called on non-fn type"); }
    }
}

fn is_fn_ty(cx: ctxt, fty: t) -> bool {
    alt struct(cx, fty) {
      ty::ty_fn(_) { ret true; }
      ty::ty_native_fn(_, _) { ret true; }
      _ { ret false; }
    }
}

// Just checks whether it's a fn that returns bool,
// not its purity.
fn is_pred_ty(cx: ctxt, fty: t) -> bool {
    is_fn_ty(cx, fty) && type_is_bool(cx, ty_fn_ret(cx, fty))
}

fn ty_var_id(cx: ctxt, typ: t) -> int {
    alt struct(cx, typ) {
      ty::ty_var(vid) { ret vid; }
      _ { #error("ty_var_id called on non-var ty"); fail; }
    }
}


// Type accessors for AST nodes
fn block_ty(cx: ctxt, b: ast::blk) -> t {
    ret node_id_to_type(cx, b.node.id);
}


// Returns the type of a pattern as a monotype. Like @expr_ty, this function
// doesn't provide type parameter substitutions.
fn pat_ty(cx: ctxt, pat: @ast::pat) -> t {
    ret node_id_to_monotype(cx, pat.id);
}


// Returns the type of an expression as a monotype.
//
// NB: This type doesn't provide type parameter substitutions; e.g. if you
// ask for the type of "id" in "id(3)", it will return "fn(&int) -> int"
// instead of "fn(t) -> T with T = int". If this isn't what you want, see
// expr_ty_params_and_ty() below.
fn expr_ty(cx: ctxt, expr: @ast::expr) -> t {
    ret node_id_to_monotype(cx, expr.id);
}

fn expr_ty_params_and_ty(cx: ctxt, expr: @ast::expr) -> {params: [t], ty: t} {
    ret {params: node_id_to_type_params(cx, expr.id),
         ty: node_id_to_type(cx, expr.id)};
}

fn expr_has_ty_params(cx: ctxt, expr: @ast::expr) -> bool {
    ret node_id_has_type_params(cx, expr.id);
}

fn expr_is_lval(method_map: typeck::method_map, tcx: ty::ctxt,
                e: @ast::expr) -> bool {
    alt e.node {
      ast::expr_path(_) | ast::expr_index(_, _) |
      ast::expr_unary(ast::deref., _) { true }
      ast::expr_field(base, ident, _) {
        method_map.contains_key(e.id) ? false : {
            let basety = type_autoderef(tcx, expr_ty(tcx, base));
            alt struct(tcx, basety) {
              ty_obj(_) { false }
              ty_rec(_) { true }
            }
        }
      }
      _ { false }
    }
}

fn stmt_node_id(s: @ast::stmt) -> ast::node_id {
    alt s.node {
      ast::stmt_decl(_, id) | stmt_expr(_, id) | stmt_semi(_, id) {
        ret id;
      }
    }
}

fn field_idx(id: ast::ident, fields: [field]) -> option::t<uint> {
    let i = 0u;
    for f in fields { if f.ident == id { ret some(i); } i += 1u; }
    ret none;
}

fn get_field(tcx: ctxt, rec_ty: t, id: ast::ident) -> field {
    alt struct(tcx, rec_ty) {
      ty_rec(fields) {
        alt vec::find(fields, {|f| str::eq(f.ident, id) }) {
            some(f) { ret f; }
        }
      }
    }
}

fn method_idx(id: ast::ident, meths: [method]) -> option::t<uint> {
    let i = 0u;
    for m in meths { if m.ident == id { ret some(i); } i += 1u; }
    ret none;
}

fn sort_methods(meths: [method]) -> [method] {
    fn method_lteq(a: method, b: method) -> bool {
        ret str::lteq(a.ident, b.ident);
    }
    ret std::sort::merge_sort(bind method_lteq(_, _), meths);
}

fn occurs_check_fails(tcx: ctxt, sp: option::t<span>, vid: int, rt: t) ->
   bool {
    if !type_contains_vars(tcx, rt) {
        // Fast path
        ret false;
    }

    // Occurs check!
    if vec::member(vid, vars_in_type(tcx, rt)) {
        alt sp {
          some(s) {
            // Maybe this should be span_err -- however, there's an
            // assertion later on that the type doesn't contain
            // variables, so in this case we have to be sure to die.
            tcx.sess.span_fatal
                (s, "Type inference failed because I \
                     could not find a type\n that's both of the form "
                 + ty_to_str(tcx, ty::mk_var(tcx, vid)) +
                 " and of the form " + ty_to_str(tcx, rt) +
                 ". Such a type would have to be infinitely large.");
          }
          _ { ret true; }
        }
    } else { ret false; }
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
    export resolve_type_structure;
    export resolve_type_var;
    export result;
    export unify;
    export ures_ok;
    export ures_err;
    export var_bindings;
    export precise, in_bindings;

    tag result { ures_ok(t); ures_err(type_err); }
    tag union_result { unres_ok; unres_err(type_err); }
    tag fixup_result {
        fix_ok(t); // fixup succeeded
        fix_err(int); // fixup failed because a type variable was unresolved
    }
    type var_bindings =
        {sets: ufind::ufind, types: smallintmap::smallintmap<t>};

    tag unify_style {
        precise;
        in_bindings(@var_bindings);
    }
    type ctxt = {st: unify_style, tcx: ty_ctxt};

    fn mk_var_bindings() -> @var_bindings {
        ret @{sets: ufind::make(), types: smallintmap::mk::<t>()};
    }

    // Unifies two sets.
    fn union(cx: @ctxt, set_a: uint, set_b: uint,
             variance: variance) -> union_result {
        let vb = alt cx.st {
            in_bindings(vb) { vb }
        };
        ufind::grow(vb.sets, math::max(set_a, set_b) + 1u);
        let root_a = ufind::find(vb.sets, set_a);
        let root_b = ufind::find(vb.sets, set_b);

        let replace_type =
            bind fn (vb: @var_bindings, t: t, set_a: uint, set_b: uint) {
                     ufind::union(vb.sets, set_a, set_b);
                     let root_c: uint = ufind::find(vb.sets, set_a);
                     smallintmap::insert::<t>(vb.types, root_c, t);
                 }(_, _, set_a, set_b);


        alt smallintmap::find(vb.types, root_a) {
          none. {
            alt smallintmap::find(vb.types, root_b) {
              none. { ufind::union(vb.sets, set_a, set_b); ret unres_ok; }
              some(t_b) { replace_type(vb, t_b); ret unres_ok; }
            }
          }
          some(t_a) {
            alt smallintmap::find(vb.types, root_b) {
              none. { replace_type(vb, t_a); ret unres_ok; }
              some(t_b) {
                alt unify_step(cx, t_a, t_b, variance) {
                  ures_ok(t_c) { replace_type(vb, t_c); ret unres_ok; }
                  ures_err(terr) { ret unres_err(terr); }
                }
              }
            }
          }
        }
    }

    fn record_var_binding_for_expected(
        cx: @ctxt, key: int, typ: t, variance: variance) -> result {
        record_var_binding(
            cx, key, typ, variance_transform(variance, covariant))
    }

    fn record_var_binding_for_actual(
        cx: @ctxt, key: int, typ: t, variance: variance) -> result {
        // Unifying in 'the other direction' so flip the variance
        record_var_binding(
            cx, key, typ, variance_transform(variance, contravariant))
    }

    fn record_var_binding(
        cx: @ctxt, key: int, typ: t, variance: variance) -> result {

        let vb = alt cx.st { in_bindings(vb) { vb } };
        ufind::grow(vb.sets, (key as uint) + 1u);
        let root = ufind::find(vb.sets, key as uint);
        let result_type = typ;
        alt smallintmap::find(vb.types, root) {
          some(old_type) {
            alt unify_step(cx, old_type, typ, variance) {
              ures_ok(unified_type) { result_type = unified_type; }
              rs { ret rs; }
            }
          }
          none. {/* fall through */ }
        }
        smallintmap::insert::<t>(vb.types, root, result_type);
        ret ures_ok(typ);
    }

    // Simple structural type comparison.
    fn struct_cmp(cx: @ctxt, expected: t, actual: t) -> result {
        let tcx = cx.tcx;
        let cfg = tcx.sess.get_targ_cfg();
        if mach_struct(tcx, cfg, expected) == mach_struct(tcx, cfg, actual) {
            ret ures_ok(expected);
        }
        ret ures_err(terr_mismatch);
    }

    // Right now this just checks that the lists of constraints are
    // pairwise equal.
    fn unify_constrs(base_t: t, expected: [@type_constr],
                     actual: [@type_constr]) -> result {
        let expected_len = vec::len(expected);
        let actual_len = vec::len(actual);

        if expected_len != actual_len {
            ret ures_err(terr_constr_len(expected_len, actual_len));
        }
        let i = 0u;
        let rslt;
        for c: @type_constr in expected {
            rslt = unify_constr(base_t, c, actual[i]);
            alt rslt { ures_ok(_) { } ures_err(_) { ret rslt; } }
            i += 1u;
        }
        ret ures_ok(base_t);
    }
    fn unify_constr(base_t: t, expected: @type_constr,
                    actual_constr: @type_constr) -> result {
        let ok_res = ures_ok(base_t);
        let err_res = ures_err(terr_constr_mismatch(expected, actual_constr));
        if expected.node.id != actual_constr.node.id { ret err_res; }
        let expected_arg_len = vec::len(expected.node.args);
        let actual_arg_len = vec::len(actual_constr.node.args);
        if expected_arg_len != actual_arg_len { ret err_res; }
        let i = 0u;
        let actual;
        for a: @ty_constr_arg in expected.node.args {
            actual = actual_constr.node.args[i];
            alt a.node {
              carg_base. {
                alt actual.node { carg_base. { } _ { ret err_res; } }
              }
              carg_lit(l) {
                alt actual.node {
                  carg_lit(m) { if l != m { ret err_res; } }
                  _ { ret err_res; }
                }
              }
              carg_ident(p) {
                alt actual.node {
                  carg_ident(q) { if p.node != q.node { ret err_res; } }
                  _ { ret err_res; }
                }
              }
            }
            i += 1u;
        }
        ret ok_res;
    }

    // Unifies two mutability flags.
    fn unify_mut(expected: ast::mutability, actual: ast::mutability,
                 variance: variance) ->
       option::t<(ast::mutability, variance)> {

        // If you're unifying on something mutable then we have to
        // be invariant on the inner type
        let newvariance = alt expected {
          ast::mut. {
            variance_transform(variance, invariant)
          }
          _ {
            variance_transform(variance, covariant)
          }
        };

        if expected == actual { ret some((expected, newvariance)); }
        if variance == covariant {
            if expected == ast::maybe_mut {
                ret some((actual, newvariance));
            }
        } else if variance == contravariant {
            if actual == ast::maybe_mut {
                ret some((expected, newvariance));
            }
        }
        ret none;
    }
    fn unify_fn_proto(e_proto: ast::proto, a_proto: ast::proto,
                      variance: variance) -> option::t<result> {
        // Prototypes form a diamond-shaped partial order:
        //
        //        block
        //        ^   ^
        //   shared   send
        //        ^   ^
        //        bare
        //
        // where "^" means "subtype of" (forgive the abuse of the term
        // subtype).
        fn sub_proto(p_sub: ast::proto, p_sup: ast::proto) -> bool {
            ret alt (p_sub, p_sup) {
              (_, ast::proto_block.) { true }
              (ast::proto_bare., _) { true }

              // Equal prototypes (modulo sugar) are always subprotos:
              (ast::proto_shared(_), ast::proto_shared(_)) { true }
              (_, _) { p_sub == p_sup }
            };
        }

        ret alt variance {
          invariant. when e_proto == a_proto { none }
          covariant. when sub_proto(a_proto, e_proto) { none }
          contravariant. when sub_proto(e_proto, a_proto) { none }
          _ { some(ures_err(terr_mismatch)) }
        };
    }
    fn unify_args(cx: @ctxt, e_args: [arg], a_args: [arg], variance: variance)
        -> either::t<result, [arg]> {
        if !vec::same_length(e_args, a_args) {
            ret either::left(ures_err(terr_arg_count));
        }
        // The variance changes (flips basically) when descending
        // into arguments of function types
        let variance = variance_transform(variance, contravariant);
        // Would use vec::map2(), but for the need to return in case of
        // error:
        let i = 0u, result = [];
        for expected_input in e_args {
            let actual_input = a_args[i];
            i += 1u;
            // Unify the result modes.
            let result_mode = if expected_input.mode == ast::mode_infer {
                actual_input.mode
            } else if actual_input.mode == ast::mode_infer {
                expected_input.mode
            } else if expected_input.mode != actual_input.mode {
                ret either::left(ures_err(terr_mode_mismatch(
                    expected_input.mode, actual_input.mode)));
            } else { expected_input.mode };

            alt unify_step(cx, expected_input.ty, actual_input.ty,
                           variance) {
              ures_ok(rty) { result += [{mode: result_mode, ty: rty}]; }
              err { ret either::left(err); }
            }
        }
        either::right(result)
    }
    fn unify_fn(cx: @ctxt, e_f: fn_ty, a_f: fn_ty, variance: variance)
        -> result {
        alt unify_fn_proto(e_f.proto, a_f.proto, variance) {
          some(err) { ret err; }
          none. { /* fall through */ }
        }

        if a_f.ret_style != ast::noreturn && a_f.ret_style != e_f.ret_style {
            /* even though typestate checking is mostly
               responsible for checking control flow annotations,
               this check is necessary to ensure that the
               annotation in an object method matches the
               declared object type */
            ret ures_err(terr_ret_style_mismatch(e_f.ret_style,
                                                 a_f.ret_style));
        }
        let result_ins = alt unify_args(cx, e_f.inputs, a_f.inputs,
                                        variance) {
            either::left(err) { ret err; }
            either::right(ts) { ts }
        };

        // Check the output.
        alt unify_step(cx, e_f.output, a_f.output, variance) {
          ures_ok(rty) {
            ures_ok(mk_fn(cx.tcx, {proto: e_f.proto,
                                   inputs: result_ins,
                                   output: rty
                                   with a_f}))
          }
          x { x }
        }
    }
    fn unify_native_fn(cx: @ctxt, expected_inputs: [arg], expected_output: t,
                       actual_inputs: [arg], actual_output: t,
                       variance: variance) -> result {
        let result_ins = alt unify_args(cx, expected_inputs,
                                        actual_inputs, variance) {
            either::left(err) { ret err; }
            either::right(ts) { ts }
        };
        alt unify_step(cx, expected_output, actual_output, variance) {
          ures_ok(out) { ures_ok(mk_native_fn(cx.tcx, result_ins, out)) }
          err { err }
        }
    }
    fn unify_obj(cx: @ctxt, expected_meths: [method],
                 actual_meths: [method], variance: variance) -> result {
        let result_meths: [method] = [];
        let i: uint = 0u;
        let expected_len: uint = vec::len(expected_meths);
        let actual_len: uint = vec::len(actual_meths);
        if expected_len != actual_len { ret ures_err(terr_meth_count); }
        while i < expected_len {
            let e_meth = expected_meths[i];
            let a_meth = actual_meths[i];
            if !str::eq(e_meth.ident, a_meth.ident) {
                ret ures_err(terr_obj_meths(e_meth.ident, a_meth.ident));
            }
            alt unify_fn(cx, e_meth.fty, a_meth.fty, variance) {
              ures_ok(tfn) {
                alt struct(cx.tcx, tfn) {
                  ty_fn(f) {
                    result_meths += [{ident: e_meth.ident,
                                      tps: a_meth.tps, fty: f}];
                  }
                }
              }
              err { ret err; }
            }
            i += 1u;
        }
        let t = mk_obj(cx.tcx, result_meths);
        ret ures_ok(t);
    }

    // If the given type is a variable, returns the structure of that type.
    fn resolve_type_structure(tcx: ty_ctxt, vb: @var_bindings, typ: t) ->
       fixup_result {
        alt struct(tcx, typ) {
          ty_var(vid) {
            if vid as uint >= ufind::set_count(vb.sets) { ret fix_err(vid); }
            let root_id = ufind::find(vb.sets, vid as uint);
            alt smallintmap::find::<t>(vb.types, root_id) {
              none. { ret fix_err(vid); }
              some(rt) { ret fix_ok(rt); }
            }
          }
          _ { ret fix_ok(typ); }
        }
    }

    // Specifies the allowable subtyping between expected and actual types
    tag variance {
        // Actual may be a subtype of expected
        covariant;
        // Actual may be a supertype of expected
        contravariant;
        // Actual must be the same type as expected
        invariant;
    }

    // The calculation for recursive variance
    // "Taming the Wildcards: Combining Definition- and Use-Site Variance"
    // by John Altidor, et. al.
    //
    // I'm just copying the table from figure 1 - haven't actually
    // read the paper (yet).
    fn variance_transform(a: variance, b: variance) -> variance {
        alt a {
          covariant. {
            alt b {
              covariant. { covariant }
              contravariant. { contravariant }
              invariant. { invariant }
            }
          }
          contravariant. {
            alt b {
              covariant. { contravariant }
              contravariant. { covariant }
              invariant. { invariant }
            }
          }
          invariant. {
            alt b {
              covariant. { invariant }
              contravariant. { invariant }
              invariant. { invariant }
            }
          }
        }
    }

    fn unify_tps(cx: @ctxt, expected_tps: [t], actual_tps: [t],
                 variance: variance, finish: block([t]) -> result) -> result {
        let result_tps = [], i = 0u;
        for exp in expected_tps {
            let act = actual_tps[i];
            i += 1u;
            let result = unify_step(cx, exp, act, variance);
            alt result {
              ures_ok(rty) { result_tps += [rty]; }
              _ { ret result; }
            }
        }
        finish(result_tps)
    }
    fn unify_step(cx: @ctxt, expected: t, actual: t,
                  variance: variance) -> result {
        // FIXME: rewrite this using tuple pattern matching when available, to
        // avoid all this rightward drift and spikiness.
        // NOTE: we have tuple matching now, but that involves copying the
        // matched elements into a tuple first, which is expensive, since sty
        // holds vectors, which are currently unique

        // Fast path.
        if expected == actual { ret ures_ok(expected); }

        // Stage 1: Handle the cases in which one side or another is a type
        // variable.

        alt struct(cx.tcx, actual) {
          // If the RHS is a variable type, then just do the
          // appropriate binding.
          ty::ty_var(actual_id) {
            let actual_n = actual_id as uint;
            alt struct(cx.tcx, expected) {
              ty::ty_var(expected_id) {
                let expected_n = expected_id as uint;
                alt union(cx, expected_n, actual_n, variance) {
                  unres_ok. {/* fall through */ }
                  unres_err(t_e) { ret ures_err(t_e); }
                }
              }
              _ {
                // Just bind the type variable to the expected type.
                alt record_var_binding_for_actual(
                    cx, actual_id, expected, variance) {
                  ures_ok(_) {/* fall through */ }
                  rs { ret rs; }
                }
              }
            }
            ret ures_ok(mk_var(cx.tcx, actual_id));
          }
          _ {/* empty */ }
        }
        alt struct(cx.tcx, expected) {
          ty::ty_var(expected_id) {
            // Add a binding. (`actual` can't actually be a var here.)
            alt record_var_binding_for_expected(
                cx, expected_id, actual,
                variance) {
              ures_ok(_) {/* fall through */ }
              rs { ret rs; }
            }
            ret ures_ok(mk_var(cx.tcx, expected_id));
          }
          _ {/* fall through */ }
        }
        // Stage 2: Handle all other cases.

        alt struct(cx.tcx, actual) {
          ty::ty_bot. { ret ures_ok(expected); }
          _ {/* fall through */ }
        }
        alt struct(cx.tcx, expected) {
          ty::ty_nil. { ret struct_cmp(cx, expected, actual); }
          // _|_ unifies with anything
          ty::ty_bot. {
            ret ures_ok(actual);
          }
          ty::ty_bool. | ty::ty_int(_) | ty_uint(_) | ty_float(_) |
          ty::ty_str. | ty::ty_type. | ty::ty_send_type. {
            ret struct_cmp(cx, expected, actual);
          }
          ty::ty_native(ex_id) {
            alt struct(cx.tcx, actual) {
              ty_native(act_id) {
                if ex_id.crate == act_id.crate && ex_id.node == act_id.node {
                    ret ures_ok(actual);
                } else { ret ures_err(terr_mismatch); }
              }
              _ { ret ures_err(terr_mismatch); }
            }
          }
          ty::ty_param(expected_n, _) {
            alt struct(cx.tcx, actual) {
              ty::ty_param(actual_n, _) when expected_n == actual_n {
                ret ures_ok(expected);
              }
              _ { ret ures_err(terr_mismatch); }
            }
          }
          ty::ty_tag(expected_id, expected_tps) {
            alt struct(cx.tcx, actual) {
              ty::ty_tag(actual_id, actual_tps) {
                if expected_id != actual_id {
                    ret ures_err(terr_mismatch);
                }
                ret unify_tps(cx, expected_tps, actual_tps, variance, {|tps|
                    ures_ok(mk_tag(cx.tcx, expected_id, tps))
                });
              }
              _ {/* fall through */ }
            }
            ret ures_err(terr_mismatch);
          }
          ty_iface(expected_id, expected_tps) {
            alt struct(cx.tcx, actual) {
              ty::ty_iface(actual_id, actual_tps) {
                if expected_id != actual_id {
                    ret ures_err(terr_mismatch);
                }
                ret unify_tps(cx, expected_tps, actual_tps, variance, {|tps|
                    ures_ok(mk_iface(cx.tcx, expected_id, tps))
                });
              }
              _ {}
            }
            ret ures_err(terr_mismatch);
          }
          ty::ty_box(expected_mt) {
            alt struct(cx.tcx, actual) {
              ty::ty_box(actual_mt) {
                let (mut, var) = alt unify_mut(
                    expected_mt.mut, actual_mt.mut, variance) {
                  none. { ret ures_err(terr_box_mutability); }
                  some(mv) { mv }
                };
                let result = unify_step(
                    cx, expected_mt.ty, actual_mt.ty, var);
                alt result {
                  ures_ok(result_sub) {
                    let mt = {ty: result_sub, mut: mut};
                    ret ures_ok(mk_box(cx.tcx, mt));
                  }
                  _ { ret result; }
                }
              }
              _ { ret ures_err(terr_mismatch); }
            }
          }
          ty::ty_uniq(expected_mt) {
            alt struct(cx.tcx, actual) {
              ty::ty_uniq(actual_mt) {
                let (mut, var) = alt unify_mut(
                    expected_mt.mut, actual_mt.mut, variance) {
                  none. { ret ures_err(terr_box_mutability); }
                  some(mv) { mv }
                };
                let result = unify_step(
                    cx, expected_mt.ty, actual_mt.ty, var);
                alt result {
                  ures_ok(result_mt) {
                    let mt = {ty: result_mt, mut: mut};
                    ret ures_ok(mk_uniq(cx.tcx, mt));
                  }
                  _ { ret result; }
                }
              }
              _ { ret ures_err(terr_mismatch); }
            }
          }
          ty::ty_vec(expected_mt) {
            alt struct(cx.tcx, actual) {
              ty::ty_vec(actual_mt) {
                let (mut, var) = alt unify_mut(
                    expected_mt.mut, actual_mt.mut, variance) {
                  none. { ret ures_err(terr_vec_mutability); }
                  some(mv) { mv }
                };
                let result = unify_step(
                    cx, expected_mt.ty, actual_mt.ty, var);
                alt result {
                  ures_ok(result_sub) {
                    let mt = {ty: result_sub, mut: mut};
                    ret ures_ok(mk_vec(cx.tcx, mt));
                  }
                  _ { ret result; }
                }
              }
              _ { ret ures_err(terr_mismatch); }
            }
          }
          ty::ty_ptr(expected_mt) {
            alt struct(cx.tcx, actual) {
              ty::ty_ptr(actual_mt) {
                let (mut, var) = alt unify_mut(
                    expected_mt.mut, actual_mt.mut, variance) {
                  none. { ret ures_err(terr_vec_mutability); }
                  some(mv) { mv }
                };
                let result = unify_step(
                    cx, expected_mt.ty, actual_mt.ty, var);
                alt result {
                  ures_ok(result_sub) {
                    let mt = {ty: result_sub, mut: mut};
                    ret ures_ok(mk_ptr(cx.tcx, mt));
                  }
                  _ { ret result; }
                }
              }
              _ { ret ures_err(terr_mismatch); }
            }
          }
          ty::ty_res(ex_id, ex_inner, ex_tps) {
            alt struct(cx.tcx, actual) {
              ty::ty_res(act_id, act_inner, act_tps) {
                if ex_id.crate != act_id.crate || ex_id.node != act_id.node {
                    ret ures_err(terr_mismatch);
                }
                let result = unify_step(
                    cx, ex_inner, act_inner, variance);
                alt result {
                  ures_ok(res_inner) {
                    let i = 0u;
                    let res_tps = [];
                    for ex_tp: t in ex_tps {
                        let result = unify_step(
                            cx, ex_tp, act_tps[i], variance);
                        alt result {
                          ures_ok(rty) { res_tps += [rty]; }
                          _ { ret result; }
                        }
                        i += 1u;
                    }
                    ret ures_ok(mk_res(cx.tcx, act_id, res_inner, res_tps));
                  }
                  _ { ret result; }
                }
              }
              _ { ret ures_err(terr_mismatch); }
            }
          }
          ty::ty_rec(expected_fields) {
            alt struct(cx.tcx, actual) {
              ty::ty_rec(actual_fields) {
                let expected_len = vec::len::<field>(expected_fields);
                let actual_len = vec::len::<field>(actual_fields);
                if expected_len != actual_len {
                    let err = terr_record_size(expected_len, actual_len);
                    ret ures_err(err);
                }
                // TODO: implement an iterator that can iterate over
                // two arrays simultaneously.

                let result_fields: [field] = [];
                let i = 0u;
                while i < expected_len {
                    let expected_field = expected_fields[i];
                    let actual_field = actual_fields[i];
                    let (mut, var) = alt unify_mut(
                        expected_field.mt.mut, actual_field.mt.mut, variance)
                        {
                      none. { ret ures_err(terr_record_mutability); }
                      some(mv) { mv }
                    };
                    if !str::eq(expected_field.ident, actual_field.ident) {
                        let err =
                            terr_record_fields(expected_field.ident,
                                               actual_field.ident);
                        ret ures_err(err);
                    }
                    let result =
                        unify_step(cx, expected_field.mt.ty,
                                   actual_field.mt.ty, var);
                    alt result {
                      ures_ok(rty) {
                        let mt = {ty: rty, mut: mut};
                        result_fields += [{mt: mt with expected_field}];
                      }
                      _ { ret result; }
                    }
                    i += 1u;
                }
                ret ures_ok(mk_rec(cx.tcx, result_fields));
              }
              _ { ret ures_err(terr_mismatch); }
            }
          }
          ty::ty_tup(expected_elems) {
            alt struct(cx.tcx, actual) {
              ty::ty_tup(actual_elems) {
                let expected_len = vec::len(expected_elems);
                let actual_len = vec::len(actual_elems);
                if expected_len != actual_len {
                    let err = terr_tuple_size(expected_len, actual_len);
                    ret ures_err(err);
                }
                // TODO: implement an iterator that can iterate over
                // two arrays simultaneously.

                let result_elems = [];
                let i = 0u;
                while i < expected_len {
                    let expected_elem = expected_elems[i];
                    let actual_elem = actual_elems[i];
                    let result = unify_step(
                        cx, expected_elem, actual_elem, variance);
                    alt result {
                      ures_ok(rty) { result_elems += [rty]; }
                      _ { ret result; }
                    }
                    i += 1u;
                }
                ret ures_ok(mk_tup(cx.tcx, result_elems));
              }
              _ { ret ures_err(terr_mismatch); }
            }
          }
          ty::ty_fn(expected_f) {
            alt struct(cx.tcx, actual) {
              ty::ty_fn(actual_f) {
                ret unify_fn(cx, expected_f, actual_f, variance);
              }
              _ { ret ures_err(terr_mismatch); }
            }
          }
          ty::ty_native_fn(expected_inputs, expected_output) {
            alt struct(cx.tcx, actual) {
              ty::ty_native_fn(actual_inputs, actual_output) {
                ret unify_native_fn(cx, expected_inputs, expected_output,
                                    actual_inputs, actual_output, variance);
              }
              _ { ret ures_err(terr_mismatch); }
            }
          }
          ty::ty_obj(expected_meths) {
            alt struct(cx.tcx, actual) {
              ty::ty_obj(actual_meths) {
                ret unify_obj(cx, expected_meths, actual_meths, variance);
              }
              _ { ret ures_err(terr_mismatch); }
            }
          }
          ty::ty_constr(expected_t, expected_constrs) {

            // unify the base types...
            alt struct(cx.tcx, actual) {
              ty::ty_constr(actual_t, actual_constrs) {
                let rslt = unify_step(
                    cx, expected_t, actual_t, variance);
                alt rslt {
                  ures_ok(rty) {
                    // FIXME: probably too restrictive --
                    // requires the constraints to be
                    // syntactically equal
                    ret unify_constrs(expected, expected_constrs,
                                      actual_constrs);
                  }
                  _ { ret rslt; }
                }
              }
              _ {
                // If the actual type is *not* a constrained type,
                // then we go ahead and just ignore the constraints on
                // the expected type. typestate handles the rest.
                ret unify_step(
                    cx, expected_t, actual, variance);
              }
            }
          }
        }
    }
    fn unify(expected: t, actual: t, st: unify_style,
             tcx: ty_ctxt) -> result {
        let cx = @{st: st, tcx: tcx};
        ret unify_step(cx, expected, actual, covariant);
    }
    fn dump_var_bindings(tcx: ty_ctxt, vb: @var_bindings) {
        let i = 0u;
        while i < vec::len::<ufind::node>(vb.sets.nodes) {
            let sets = "";
            let j = 0u;
            while j < vec::len::<option::t<uint>>(vb.sets.nodes) {
                if ufind::find(vb.sets, j) == i { sets += #fmt[" %u", j]; }
                j += 1u;
            }
            let typespec;
            alt smallintmap::find::<t>(vb.types, i) {
              none. { typespec = ""; }
              some(typ) { typespec = " =" + ty_to_str(tcx, typ); }
            }
            #error("set %u:%s%s", i, typespec, sets);
            i += 1u;
        }
    }

    // Fixups and substitutions
    //    Takes an optional span - complain about occurs check violations
    //    iff the span is present (so that if we already know we're going
    //    to error anyway, we don't complain)
    fn fixup_vars(tcx: ty_ctxt, sp: option::t<span>, vb: @var_bindings,
                  typ: t) -> fixup_result {
        fn subst_vars(tcx: ty_ctxt, sp: option::t<span>, vb: @var_bindings,
                      unresolved: @mutable option::t<int>, vid: int) -> t {
            // Should really return a fixup_result instead of a t, but fold_ty
            // doesn't allow returning anything but a t.
            if vid as uint >= ufind::set_count(vb.sets) {
                *unresolved = some(vid);
                ret ty::mk_var(tcx, vid);
            }
            let root_id = ufind::find(vb.sets, vid as uint);
            alt smallintmap::find::<t>(vb.types, root_id) {
              none. { *unresolved = some(vid); ret ty::mk_var(tcx, vid); }
              some(rt) {
                if occurs_check_fails(tcx, sp, vid, rt) {
                    // Return the type unchanged, so we can error out
                    // downstream
                    ret rt;
                }
                ret fold_ty(tcx,
                            fm_var(bind subst_vars(tcx, sp, vb, unresolved,
                                                   _)), rt);
              }
            }
        }
        let unresolved = @mutable none::<int>;
        let rty =
            fold_ty(tcx, fm_var(bind subst_vars(tcx, sp, vb, unresolved, _)),
                    typ);
        let ur = *unresolved;
        alt ur {
          none. { ret fix_ok(rty); }
          some(var_id) { ret fix_err(var_id); }
        }
    }
    fn resolve_type_var(tcx: ty_ctxt, sp: option::t<span>, vb: @var_bindings,
                        vid: int) -> fixup_result {
        if vid as uint >= ufind::set_count(vb.sets) { ret fix_err(vid); }
        let root_id = ufind::find(vb.sets, vid as uint);
        alt smallintmap::find::<t>(vb.types, root_id) {
          none. { ret fix_err(vid); }
          some(rt) { ret fixup_vars(tcx, sp, vb, rt); }
        }
    }
}

fn same_type(cx: ctxt, a: t, b: t) -> bool {
    alt unify::unify(a, b, unify::precise, cx) {
      unify::ures_ok(_) { true }
      _ { false }
    }
}
fn same_method(cx: ctxt, a: method, b: method) -> bool {
    a.tps == b.tps && a.fty.proto == b.fty.proto && a.ident == b.ident &&
    vec::all2(a.fty.inputs, b.fty.inputs,
              {|a, b| a.mode == b.mode && same_type(cx, a.ty, b.ty) }) &&
    same_type(cx, a.fty.output, b.fty.output) &&
    a.fty.ret_style == b.fty.ret_style
}

fn type_err_to_str(err: ty::type_err) -> str {
    alt err {
      terr_mismatch. { ret "types differ"; }
      terr_ret_style_mismatch(expect, actual) {
        fn to_str(s: ast::ret_style) -> str {
            alt s {
              ast::noreturn. { "non-returning" }
              ast::return_val. { "return-by-value" }
            }
        }
        ret to_str(actual) + " function found where " + to_str(expect) +
            " function was expected";
      }
      terr_box_mutability. { ret "boxed values differ in mutability"; }
      terr_vec_mutability. { ret "vectors differ in mutability"; }
      terr_tuple_size(e_sz, a_sz) {
        ret "expected a tuple with " + uint::to_str(e_sz, 10u) +
                " elements but found one with " + uint::to_str(a_sz, 10u) +
                " elements";
      }
      terr_record_size(e_sz, a_sz) {
        ret "expected a record with " + uint::to_str(e_sz, 10u) +
                " fields but found one with " + uint::to_str(a_sz, 10u) +
                " fields";
      }
      terr_record_mutability. { ret "record elements differ in mutability"; }
      terr_record_fields(e_fld, a_fld) {
        ret "expected a record with field '" + e_fld +
                "' but found one with field '" + a_fld + "'";
      }
      terr_arg_count. { ret "incorrect number of function parameters"; }
      terr_meth_count. { ret "incorrect number of object methods"; }
      terr_obj_meths(e_meth, a_meth) {
        ret "expected an obj with method '" + e_meth +
                "' but found one with method '" + a_meth + "'";
      }
      terr_mode_mismatch(e_mode, a_mode) {
        ret "expected argument mode " + mode_str(e_mode) + " but found " +
                mode_str(a_mode);
      }
      terr_constr_len(e_len, a_len) {
        ret "Expected a type with " + uint::str(e_len) +
                " constraints, but found one with " + uint::str(a_len) +
                " constraints";
      }
      terr_constr_mismatch(e_constr, a_constr) {
        ret "Expected a type with constraint " + ty_constr_to_str(e_constr) +
                " but found one with constraint " +
                ty_constr_to_str(a_constr);
      }
    }
}

// Replaces type parameters in the given type using the given list of
// substitions.
fn substitute_type_params(cx: ctxt, substs: [ty::t], typ: t) -> t {
    if !type_contains_params(cx, typ) { ret typ; }
    fn substituter(_cx: ctxt, substs: @[ty::t], idx: uint, _did: def_id)
        -> t {
        // FIXME: bounds check can fail
        ret substs[idx];
    }
    ret fold_ty(cx, fm_param(bind substituter(cx, @substs, _, _)), typ);
}

fn def_has_ty_params(def: ast::def) -> bool {
    alt def {
      ast::def_obj_field(_, _) | ast::def_mod(_) | ast::def_const(_) |
      ast::def_arg(_, _) | ast::def_local(_, _) | ast::def_upvar(_, _, _) |
      ast::def_ty_param(_, _) | ast::def_binding(_) | ast::def_use(_) |
      ast::def_native_ty(_) | ast::def_self(_) | ast::def_ty(_) { false }
      ast::def_fn(_, _) | ast::def_variant(_, _) |
      ast::def_native_fn(_, _) { true }
    }
}

fn store_iface_methods(cx: ctxt, id: ast::node_id, ms: @[method]) {
    cx.iface_method_cache.insert(ast_util::local_def(id), ms);
}

fn iface_methods(cx: ctxt, id: ast::def_id) -> @[method] {
    alt cx.iface_method_cache.find(id) {
      some(ms) { ret ms; }
      _ {}
    }
    // Local interfaces are supposed to have been added explicitly.
    assert id.crate != ast::local_crate;
    let result = csearch::get_iface_methods(cx, id);
    cx.iface_method_cache.insert(id, result);
    result
}

fn impl_iface(cx: ctxt, id: ast::def_id) -> option::t<t> {
    if id.crate == ast::local_crate {
        option::map(cx.tcache.find(id), {|it| it.ty})
    } else {
        csearch::get_impl_iface(cx, id)
    }
}

// Tag information
type variant_info = @{args: [ty::t], ctor_ty: ty::t, id: ast::def_id};

fn tag_variants(cx: ctxt, id: ast::def_id) -> @[variant_info] {
    alt cx.tag_var_cache.find(id) {
      some(variants) { ret variants; }
      _ { /* fallthrough */ }
    }
    let result = if ast::local_crate != id.crate {
        @csearch::get_tag_variants(cx, id)
    } else {
        alt cx.items.get(id.node) {
          ast_map::node_item(@{node: ast::item_tag(variants, _), _}) {
            @vec::map(variants, {|variant|
                let ctor_ty = node_id_to_monotype(cx, variant.node.id);
                let arg_tys = if vec::len(variant.node.args) > 0u {
                    vec::map(ty_fn_args(cx, ctor_ty), {|a| a.ty})
                } else { [] };
                @{args: arg_tys,
                  ctor_ty: ctor_ty,
                  id: ast_util::local_def(variant.node.id)}
            })
          }
        }
    };
    cx.tag_var_cache.insert(id, result);
    result
}


// Returns information about the tag variant with the given ID:
fn tag_variant_with_id(cx: ctxt, tag_id: ast::def_id, variant_id: ast::def_id)
   -> variant_info {
    let variants = tag_variants(cx, tag_id);
    let i = 0u;
    while i < vec::len::<variant_info>(*variants) {
        let variant = variants[i];
        if def_eq(variant.id, variant_id) { ret variant; }
        i += 1u;
    }
    cx.sess.bug("tag_variant_with_id(): no variant exists with that ID");
}


// If the given item is in an external crate, looks up its type and adds it to
// the type cache. Returns the type parameters and type.
fn lookup_item_type(cx: ctxt, did: ast::def_id) -> ty_param_bounds_and_ty {
    if did.crate == ast::local_crate {
        // The item is in this crate. The caller should have added it to the
        // type cache already; we simply return it.

        ret cx.tcache.get(did);
    }
    alt cx.tcache.find(did) {
      some(tpt) { ret tpt; }
      none. {
        let tyt = csearch::get_type(cx, did);
        cx.tcache.insert(did, tyt);
        ret tyt;
      }
    }
}

fn ret_ty_of_fn(cx: ctxt, id: ast::node_id) -> t {
    ty_fn_ret(cx, node_id_to_type(cx, id))
}

fn is_binopable(cx: ctxt, ty: t, op: ast::binop) -> bool {

    const tycat_other: int = 0;
    const tycat_bool: int = 1;
    const tycat_int: int = 2;
    const tycat_float: int = 3;
    const tycat_str: int = 4;
    const tycat_vec: int = 5;
    const tycat_struct: int = 6;
    const tycat_bot: int = 7;

    const opcat_add: int = 0;
    const opcat_sub: int = 1;
    const opcat_mult: int = 2;
    const opcat_shift: int = 3;
    const opcat_rel: int = 4;
    const opcat_eq: int = 5;
    const opcat_bit: int = 6;
    const opcat_logic: int = 7;

    fn opcat(op: ast::binop) -> int {
        alt op {
          ast::add. { opcat_add }
          ast::sub. { opcat_sub }
          ast::mul. { opcat_mult }
          ast::div. { opcat_mult }
          ast::rem. { opcat_mult }
          ast::and. { opcat_logic }
          ast::or. { opcat_logic }
          ast::bitxor. { opcat_bit }
          ast::bitand. { opcat_bit }
          ast::bitor. { opcat_bit }
          ast::lsl. { opcat_shift }
          ast::lsr. { opcat_shift }
          ast::asr. { opcat_shift }
          ast::eq. { opcat_eq }
          ast::ne. { opcat_eq }
          ast::lt. { opcat_rel }
          ast::le. { opcat_rel }
          ast::ge. { opcat_rel }
          ast::gt. { opcat_rel }
        }
    }

    fn tycat(cx: ctxt, ty: t) -> int {
        alt struct(cx, ty) {
          ty_bool. { tycat_bool }
          ty_int(_) { tycat_int }
          ty_uint(_) { tycat_int }
          ty_float(_) { tycat_float }
          ty_str. { tycat_str }
          ty_vec(_) { tycat_vec }
          ty_rec(_) { tycat_struct }
          ty_tup(_) { tycat_struct }
          ty_tag(_, _) { tycat_struct }
          ty_bot. { tycat_bot }
          _ { tycat_other }
        }
    }

    const t: bool = true;
    const f: bool = false;

    /*.          add,     shift,   bit
      .             sub,     rel,     logic
      .                mult,    eq,         */
    /*other*/
    /*bool*/
    /*int*/
    /*float*/
    /*str*/
    /*vec*/
    /*bot*/
    let tbl =
        [[f, f, f, f, t, t, f, f], [f, f, f, f, t, t, t, t],
         [t, t, t, t, t, t, t, f], [t, t, t, f, t, t, f, f],
         [t, f, f, f, t, t, f, f], [t, f, f, f, t, t, f, f],
         [f, f, f, f, t, t, f, f], [t, t, t, t, t, t, t, t]]; /*struct*/

    ret tbl[tycat(cx, ty)][opcat(op)];
}

fn ast_constr_to_constr<T>(tcx: ty::ctxt, c: @ast::constr_general<T>) ->
   @ty::constr_general<T> {
    alt tcx.def_map.find(c.node.id) {
      some(ast::def_fn(pred_id, ast::pure_fn.)) {
        ret @ast_util::respan(c.span,
                              {path: c.node.path,
                               args: c.node.args,
                               id: pred_id});
      }
      _ {
        tcx.sess.span_fatal(c.span,
                            "Predicate " + path_to_str(c.node.path) +
                            " is unbound or bound to a non-function or an \
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
// End:
