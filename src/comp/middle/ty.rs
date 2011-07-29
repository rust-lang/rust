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
export mk_imm_tup;
export mk_str;
export mk_tag;
export mk_task;
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
export type_kind;
export type_err;
export type_err_to_str;
export type_has_dynamic_size;
export type_has_pointers;
export type_needs_drop;
export type_is_bool;
export type_is_bot;
export type_is_box;
export type_is_boxed;
export type_is_chan;
export type_is_task;
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

type arg = {mode: mode, ty: t};

type field = {ident: ast::ident, mt: mt};

type method =
    {proto: ast::proto,
     ident: ast::ident,
     inputs: arg[],
     output: t,
     cf: controlflow,
     constrs: (@constr)[]};

type constr_table = hashmap[ast::node_id, constr[]];

type mt = {ty: t, mut: ast::mutability};


// Contains information needed to resolve types and (in the future) look up
// the types of AST nodes.
type creader_cache = hashmap[{cnum: int, pos: uint, len: uint}, ty::t];

type ctxt =

    //        constr_table fn_constrs,
    @{ts: @type_store,
      sess: session::session,
      def_map: resolve::def_map,
      node_types: node_type_table,
      items: ast_map::map,
      freevars: freevars::freevar_map,
      tcache: type_cache,
      rcache: creader_cache,
      short_names_cache: hashmap[t, str],
      has_pointer_cache: hashmap[t, bool],
      kind_cache: hashmap[t, ast::kind],
      owns_heap_mem_cache: hashmap[t, bool],
      ast_ty_to_ty_cache: hashmap[@ast::ty, option::t[t]]};

type ty_ctxt = ctxt;


// Needed for disambiguation from unify::ctxt.
// Convert from method type to function type.  Pretty easy; we just drop
// 'ident'.
fn method_ty_to_fn_ty(cx: &ctxt, m: method) -> t {
    ret mk_fn(cx, m.proto, m.inputs, m.output, m.cf, m.constrs);
}


// Never construct these manually. These are interned.
type raw_t =
    {struct: sty,
     cname: option::t[str],
     hash: uint,
     has_params: bool,
     has_vars: bool};

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
    ty_rec(field[]);
    ty_fn(ast::proto, arg[], t, controlflow, (@constr)[]);
    ty_native_fn(ast::native_abi, arg[], t);
    ty_obj(method[]);
    ty_res(def_id, t, t[]);
    ty_var(int); // type variable

    ty_param(uint, ast::kind); // fn/tag type param

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

type ty_param_count_and_ty = {count: uint, ty: t};

type type_cache = hashmap[ast::def_id, ty_param_count_and_ty];

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

const idx_istr: uint = 17u;

const idx_task: uint = 18u;

const idx_type: uint = 19u;

const idx_bot: uint = 20u;

const idx_first_others: uint = 21u;

type type_store = interner::interner[@raw_t];

type ty_param_substs_opt_and_ty = {substs: option::t[ty::t[]], ty: ty::t};

type node_type_table =
    @smallintmap::smallintmap[ty::ty_param_substs_opt_and_ty];

fn populate_type_store(cx: &ctxt) {
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
    type val = {cnum: int, pos: uint, len: uint};
    fn hash_cache_entry(k: &val) -> uint {
        ret (k.cnum as uint) + k.pos + k.len;
    }
    fn eq_cache_entries(a: &val, b: &val) -> bool {
        ret a.cnum == b.cnum && a.pos == b.pos && a.len == b.len;
    }
    ret map::mk_hashmap(hash_cache_entry, eq_cache_entries);
}


fn mk_ctxt(s: session::session, dm: resolve::def_map, amap: ast_map::map,
           freevars: freevars::freevar_map) -> ctxt {
    let ntt: node_type_table =
        @smallintmap::mk[ty::ty_param_substs_opt_and_ty]();
    let tcache = new_def_hash[ty::ty_param_count_and_ty]();
    let ts = @interner::mk[@raw_t](hash_raw_ty, eq_raw_ty);
    let cx =
        @{ts: ts,
          sess: s,
          def_map: dm,
          node_types: ntt,
          items: amap,
          freevars: freevars,
          tcache: tcache,
          rcache: mk_rcache(),
          short_names_cache: map::mk_hashmap(ty::hash_ty, ty::eq_ty),
          has_pointer_cache: map::mk_hashmap(ty::hash_ty, ty::eq_ty),
          kind_cache: map::mk_hashmap(ty::hash_ty, ty::eq_ty),
          owns_heap_mem_cache: map::mk_hashmap(ty::hash_ty, ty::eq_ty),
          ast_ty_to_ty_cache: map::mk_hashmap(ast::hash_ty, ast::eq_ty)};
    populate_type_store(cx);
    ret cx;
}


// Type constructors
fn mk_raw_ty(cx: &ctxt, st: &sty, in_cname: &option::t[str]) -> @raw_t {
    let cname = none;
    let h = hash_type_info(st, cname);
    let has_params: bool = false;
    let has_vars: bool = false;
    fn derive_flags_t(cx: &ctxt, has_params: &mutable bool,
                      has_vars: &mutable bool, tt: &t) {
        let rt = interner::get[@raw_t](*cx.ts, tt);
        has_params = has_params || rt.has_params;
        has_vars = has_vars || rt.has_vars;
    }
    fn derive_flags_mt(cx: &ctxt, has_params: &mutable bool,
                       has_vars: &mutable bool, m: &mt) {
        derive_flags_t(cx, has_params, has_vars, m.ty);
    }
    fn derive_flags_arg(cx: &ctxt, has_params: &mutable bool,
                        has_vars: &mutable bool, a: &arg) {
        derive_flags_t(cx, has_params, has_vars, a.ty);
    }
    fn derive_flags_sig(cx: &ctxt, has_params: &mutable bool,
                        has_vars: &mutable bool, args: &arg[], tt: &t) {
        for a: arg  in args { derive_flags_arg(cx, has_params, has_vars, a); }
        derive_flags_t(cx, has_params, has_vars, tt);
    }
    alt st {
      ty_nil. {/* no-op */ }
      ty_bot. {/* no-op */ }
      ty_bool. {/* no-op */ }
      ty_int. {/* no-op */ }
      ty_float. {/* no-op */ }
      ty_uint. {/* no-op */ }
      ty_machine(_) {/* no-op */ }
      ty_char. {/* no-op */ }
      ty_str. {/* no-op */ }
      ty_istr. {/* no-op */ }
      ty_task. {/* no-op */ }
      ty_type. {/* no-op */ }
      ty_native(_) {/* no-op */ }
      ty_param(_,_) { has_params = true; }
      ty_var(_) { has_vars = true; }
      ty_tag(_, tys) {
        for tt: t  in tys { derive_flags_t(cx, has_params, has_vars, tt); }
      }
      ty_box(m) { derive_flags_mt(cx, has_params, has_vars, m); }
      ty_vec(m) { derive_flags_mt(cx, has_params, has_vars, m); }
      ty_ivec(m) { derive_flags_mt(cx, has_params, has_vars, m); }
      ty_ptr(m) { derive_flags_mt(cx, has_params, has_vars, m); }
      ty_port(tt) { derive_flags_t(cx, has_params, has_vars, tt); }
      ty_chan(tt) { derive_flags_t(cx, has_params, has_vars, tt); }
      ty_rec(flds) {
        for f: field  in flds {
            derive_flags_mt(cx, has_params, has_vars, f.mt);
        }
      }
      ty_fn(_, args, tt, _, _) {
        derive_flags_sig(cx, has_params, has_vars, args, tt);
      }
      ty_native_fn(_, args, tt) {
        derive_flags_sig(cx, has_params, has_vars, args, tt);
      }
      ty_obj(meths) {
        for m: method  in meths {
            derive_flags_sig(cx, has_params, has_vars, m.inputs, m.output);
        }
      }
      ty_res(_, tt, tps) {
        derive_flags_t(cx, has_params, has_vars, tt);
        for tt: t  in tps { derive_flags_t(cx, has_params, has_vars, tt); }
      }
      ty_constr(tt, _) { derive_flags_t(cx, has_params, has_vars, tt); }
    }
    ret @{struct: st,
          cname: cname,
          hash: h,
          has_params: has_params,
          has_vars: has_vars};
}

fn intern(cx: &ctxt, st: &sty, cname: &option::t[str]) {
    interner::intern(*cx.ts, mk_raw_ty(cx, st, cname));
}

fn gen_ty_full(cx: &ctxt, st: &sty, cname: &option::t[str]) -> t {
    let raw_type = mk_raw_ty(cx, st, cname);
    ret interner::intern(*cx.ts, raw_type);
}


// These are private constructors to this module. External users should always
// use the mk_foo() functions below.
fn gen_ty(cx: &ctxt, st: &sty) -> t { ret gen_ty_full(cx, st, none); }

fn mk_nil(cx: &ctxt) -> t { ret idx_nil; }

fn mk_bot(cx: &ctxt) -> t { ret idx_bot; }

fn mk_bool(cx: &ctxt) -> t { ret idx_bool; }

fn mk_int(cx: &ctxt) -> t { ret idx_int; }

fn mk_float(cx: &ctxt) -> t { ret idx_float; }

fn mk_uint(cx: &ctxt) -> t { ret idx_uint; }

fn mk_mach(cx: &ctxt, tm: &ast::ty_mach) -> t {
    alt tm {
      ast::ty_u8. { ret idx_u8; }
      ast::ty_u16. { ret idx_u16; }
      ast::ty_u32. { ret idx_u32; }
      ast::ty_u64. { ret idx_u64; }
      ast::ty_i8. { ret idx_i8; }
      ast::ty_i16. { ret idx_i16; }
      ast::ty_i32. { ret idx_i32; }
      ast::ty_i64. { ret idx_i64; }
      ast::ty_f32. { ret idx_f32; }
      ast::ty_f64. { ret idx_f64; }
    }
}

fn mk_char(cx: &ctxt) -> t { ret idx_char; }

fn mk_str(cx: &ctxt) -> t { ret idx_str; }

fn mk_istr(cx: &ctxt) -> t { ret idx_istr; }

fn mk_tag(cx: &ctxt, did: &ast::def_id, tys: &t[]) -> t {
    ret gen_ty(cx, ty_tag(did, tys));
}

fn mk_box(cx: &ctxt, tm: &mt) -> t { ret gen_ty(cx, ty_box(tm)); }

fn mk_ptr(cx: &ctxt, tm: &mt) -> t { ret gen_ty(cx, ty_ptr(tm)); }

fn mk_imm_box(cx: &ctxt, ty: &t) -> t {
    ret mk_box(cx, {ty: ty, mut: ast::imm});
}

fn mk_vec(cx: &ctxt, tm: &mt) -> t { ret gen_ty(cx, ty_vec(tm)); }

fn mk_ivec(cx: &ctxt, tm: &mt) -> t { ret gen_ty(cx, ty_ivec(tm)); }

fn mk_imm_vec(cx: &ctxt, typ: &t) -> t {
    ret gen_ty(cx, ty_vec({ty: typ, mut: ast::imm}));
}

fn mk_port(cx: &ctxt, ty: &t) -> t { ret gen_ty(cx, ty_port(ty)); }

fn mk_chan(cx: &ctxt, ty: &t) -> t { ret gen_ty(cx, ty_chan(ty)); }

fn mk_task(cx: &ctxt) -> t { ret gen_ty(cx, ty_task); }

fn mk_rec(cx: &ctxt, fs: &field[]) -> t { ret gen_ty(cx, ty_rec(fs)); }

fn mk_imm_tup(cx: &ctxt, tys: &t[]) -> t {
    let fields = ~[];
    let i = 0u;
    for typ: t  in tys {
        fields += ~[{ident: #fmt("_%u", i), mt: {ty: typ, mut: ast::imm}}];
        i += 1u;
    }
    ret gen_ty(cx, ty_rec(fields));
}

fn mk_constr(cx: &ctxt, t: &t, cs: &(@type_constr)[]) -> t {
    ret gen_ty(cx, ty_constr(t, cs));
}

fn mk_fn(cx: &ctxt, proto: &ast::proto, args: &arg[], ty: &t,
         cf: &controlflow, constrs: &(@constr)[]) -> t {
    ret gen_ty(cx, ty_fn(proto, args, ty, cf, constrs));
}

fn mk_native_fn(cx: &ctxt, abi: &ast::native_abi, args: &arg[], ty: &t) -> t {
    ret gen_ty(cx, ty_native_fn(abi, args, ty));
}

fn mk_obj(cx: &ctxt, meths: &method[]) -> t { ret gen_ty(cx, ty_obj(meths)); }

fn mk_res(cx: &ctxt, did: &ast::def_id, inner: &t, tps: &t[]) -> t {
    ret gen_ty(cx, ty_res(did, inner, tps));
}

fn mk_var(cx: &ctxt, v: int) -> t { ret gen_ty(cx, ty_var(v)); }

fn mk_param(cx: &ctxt, n: uint, k: ast::kind) -> t {
    ret gen_ty(cx, ty_param(n, k));
}

fn mk_type(cx: &ctxt) -> t { ret idx_type; }

fn mk_native(cx: &ctxt, did: &def_id) -> t { ret gen_ty(cx, ty_native(did)); }


// Returns the one-level-deep type structure of the given type.
fn struct(cx: &ctxt, typ: &t) -> sty {
    ret interner::get(*cx.ts, typ).struct;
}


// Returns the canonical name of the given type.
fn cname(cx: &ctxt, typ: &t) -> option::t[str] {
    ret interner::get(*cx.ts, typ).cname;
}


// Type folds
type ty_walk = fn(t) ;

fn walk_ty(cx: &ctxt, walker: ty_walk, ty: t) {
    alt struct(cx, ty) {
      ty_nil. {/* no-op */ }
      ty_bot. {/* no-op */ }
      ty_bool. {/* no-op */ }
      ty_int. {/* no-op */ }
      ty_uint. {/* no-op */ }
      ty_float. {/* no-op */ }
      ty_machine(_) {/* no-op */ }
      ty_char. {/* no-op */ }
      ty_str. {/* no-op */ }
      ty_istr. {/* no-op */ }
      ty_type. {/* no-op */ }
      ty_native(_) {/* no-op */ }
      ty_box(tm) { walk_ty(cx, walker, tm.ty); }
      ty_vec(tm) { walk_ty(cx, walker, tm.ty); }
      ty_ivec(tm) { walk_ty(cx, walker, tm.ty); }
      ty_ptr(tm) { walk_ty(cx, walker, tm.ty); }
      ty_port(subty) { walk_ty(cx, walker, subty); }
      ty_chan(subty) { walk_ty(cx, walker, subty); }
      ty_tag(tid, subtys) {
        for subty: t  in subtys { walk_ty(cx, walker, subty); }
      }
      ty_rec(fields) {
        for fl: field  in fields { walk_ty(cx, walker, fl.mt.ty); }
      }
      ty_fn(proto, args, ret_ty, _, _) {
        for a: arg  in args { walk_ty(cx, walker, a.ty); }
        walk_ty(cx, walker, ret_ty);
      }
      ty_native_fn(abi, args, ret_ty) {
        for a: arg  in args { walk_ty(cx, walker, a.ty); }
        walk_ty(cx, walker, ret_ty);
      }
      ty_obj(methods) {
        for m: method  in methods {
            for a: arg  in m.inputs { walk_ty(cx, walker, a.ty); }
            walk_ty(cx, walker, m.output);
        }
      }
      ty_res(_, sub, tps) {
        walk_ty(cx, walker, sub);
        for tp: t  in tps { walk_ty(cx, walker, tp); }
      }
      ty_var(_) {/* no-op */ }
      ty_param(_,_) {/* no-op */ }
    }
    walker(ty);
}

tag fold_mode {
    fm_var(fn(int) -> t );
    fm_param(fn(uint,ast::kind) -> t );
    fm_general(fn(t) -> t );
}

fn fold_ty(cx: &ctxt, fld: fold_mode, ty_0: t) -> t {
    let ty = ty_0;
    // Fast paths.

    alt fld {
      fm_var(_) { if !type_contains_vars(cx, ty) { ret ty; } }
      fm_param(_) { if !type_contains_params(cx, ty) { ret ty; } }
      fm_general(_) {/* no fast path */ }
    }
    alt struct(cx, ty) {
      ty_nil. {/* no-op */ }
      ty_bot. {/* no-op */ }
      ty_bool. {/* no-op */ }
      ty_int. {/* no-op */ }
      ty_uint. {/* no-op */ }
      ty_float. {/* no-op */ }
      ty_machine(_) {/* no-op */ }
      ty_char. {/* no-op */ }
      ty_str. {/* no-op */ }
      ty_istr. {/* no-op */ }
      ty_type. {/* no-op */ }
      ty_native(_) {/* no-op */ }
      ty_task. {/* no-op */ }
      ty_box(tm) {
        ty =
            copy_cname(cx,
                       mk_box(cx, {ty: fold_ty(cx, fld, tm.ty), mut: tm.mut}),
                       ty);
      }
      ty_ptr(tm) {
        ty =
            copy_cname(cx,
                       mk_ptr(cx, {ty: fold_ty(cx, fld, tm.ty), mut: tm.mut}),
                       ty);
      }
      ty_vec(tm) {
        ty =
            copy_cname(cx,
                       mk_vec(cx, {ty: fold_ty(cx, fld, tm.ty), mut: tm.mut}),
                       ty);
      }
      ty_ivec(tm) {
        ty =
            copy_cname(cx,
                       mk_ivec(cx,
                               {ty: fold_ty(cx, fld, tm.ty), mut: tm.mut}),
                       ty);
      }
      ty_port(subty) {
        ty = copy_cname(cx, mk_port(cx, fold_ty(cx, fld, subty)), ty);
      }
      ty_chan(subty) {
        ty = copy_cname(cx, mk_chan(cx, fold_ty(cx, fld, subty)), ty);
      }
      ty_tag(tid, subtys) {
        let new_subtys: t[] = ~[];
        for subty: t  in subtys { new_subtys += ~[fold_ty(cx, fld, subty)]; }
        ty = copy_cname(cx, mk_tag(cx, tid, new_subtys), ty);
      }
      ty_rec(fields) {
        let new_fields: field[] = ~[];
        for fl: field  in fields {
            let new_ty = fold_ty(cx, fld, fl.mt.ty);
            let new_mt = {ty: new_ty, mut: fl.mt.mut};
            new_fields += ~[{ident: fl.ident, mt: new_mt}];
        }
        ty = copy_cname(cx, mk_rec(cx, new_fields), ty);
      }
      ty_fn(proto, args, ret_ty, cf, constrs) {
        let new_args: arg[] = ~[];
        for a: arg  in args {
            let new_ty = fold_ty(cx, fld, a.ty);
            new_args += ~[{mode: a.mode, ty: new_ty}];
        }
        ty =
            copy_cname(cx,
                       mk_fn(cx, proto, new_args, fold_ty(cx, fld, ret_ty),
                             cf, constrs), ty);
      }
      ty_native_fn(abi, args, ret_ty) {
        let new_args: arg[] = ~[];
        for a: arg  in args {
            let new_ty = fold_ty(cx, fld, a.ty);
            new_args += ~[{mode: a.mode, ty: new_ty}];
        }
        ty =
            copy_cname(cx,
                       mk_native_fn(cx, abi, new_args,
                                    fold_ty(cx, fld, ret_ty)), ty);
      }
      ty_obj(methods) {
        let new_methods: method[] = ~[];
        for m: method  in methods {
            let new_args: arg[] = ~[];
            for a: arg  in m.inputs {
                new_args += ~[{mode: a.mode, ty: fold_ty(cx, fld, a.ty)}];
            }
            new_methods +=
                ~[{proto: m.proto,
                   ident: m.ident,
                   inputs: new_args,
                   output: fold_ty(cx, fld, m.output),
                   cf: m.cf,
                   constrs: m.constrs}];
        }
        ty = copy_cname(cx, mk_obj(cx, new_methods), ty);
      }
      ty_res(did, subty, tps) {
        let new_tps = ~[];
        for tp: t  in tps { new_tps += ~[fold_ty(cx, fld, tp)]; }
        ty =
            copy_cname(cx, mk_res(cx, did, fold_ty(cx, fld, subty), new_tps),
                       ty);
      }
      ty_var(id) {
        alt fld { fm_var(folder) { ty = folder(id); } _ {/* no-op */ } }
      }
      ty_param(id,k) {
        alt fld { fm_param(folder) { ty = folder(id,k); } _ {/* no-op */ } }
      }
    }


    // If this is a general type fold, then we need to run it now.
    alt fld { fm_general(folder) { ret folder(ty); } _ { ret ty; } }
}


// Type utilities

fn rename(cx: &ctxt, typ: t, new_cname: str) -> t {
    ret gen_ty_full(cx, struct(cx, typ), some(new_cname));
}

fn strip_cname(cx: &ctxt, typ: t) -> t {
    ret gen_ty_full(cx, struct(cx, typ), none);
}

// Returns a type with the structural part taken from `struct_ty` and the
// canonical name from `cname_ty`.
fn copy_cname(cx: &ctxt, struct_ty: t, cname_ty: t) -> t {
    ret gen_ty_full(cx, struct(cx, struct_ty), cname(cx, cname_ty));
}

fn type_is_nil(cx: &ctxt, ty: &t) -> bool {
    alt struct(cx, ty) { ty_nil. { ret true; } _ { ret false; } }
}

fn type_is_bot(cx: &ctxt, ty: &t) -> bool {
    alt struct(cx, ty) { ty_bot. { ret true; } _ { ret false; } }
}

fn type_is_bool(cx: &ctxt, ty: &t) -> bool {
    alt struct(cx, ty) { ty_bool. { ret true; } _ { ret false; } }
}

fn type_is_chan(cx: &ctxt, ty: &t) -> bool {
    alt struct(cx, ty) { ty_chan(_) { ret true; } _ { ret false; } }
}

fn type_is_task(cx: &ctxt, ty: &t) -> bool {
    alt struct(cx, ty) { ty_task. { ret true; } _ { ret false; } }
}

fn type_is_structural(cx: &ctxt, ty: &t) -> bool {
    alt struct(cx, ty) {
      ty_rec(_) { ret true; }
      ty_tag(_, _) { ret true; }
      ty_fn(_, _, _, _, _) { ret true; }
      ty_obj(_) { ret true; }
      ty_res(_, _, _) { ret true; }
      ty_ivec(_) { ret true; }
      ty_istr. { ret true; }
      _ { ret false; }
    }
}

fn type_is_copyable(cx: &ctxt, ty: &t) -> bool {
    ret alt struct(cx, ty) {
          ty_res(_, _, _) { false }
          ty_fn(proto_block., _, _, _, _) { false }
          _ { true }
        };
}

fn type_is_sequence(cx: &ctxt, ty: &t) -> bool {
    alt struct(cx, ty) {
      ty_str. { ret true; }
      ty_istr. { ret true; }
      ty_vec(_) { ret true; }
      ty_ivec(_) { ret true; }
      _ { ret false; }
    }
}

fn type_is_str(cx: &ctxt, ty: &t) -> bool {
    alt struct(cx, ty) {
      ty_str. { ret true; }
      ty_istr. { ret true; }
      _ { ret false; }
    }
}

fn sequence_is_interior(cx: &ctxt, ty: &t) -> bool {
    alt struct(cx, ty) {

      // TODO: Or-patterns
      ty::ty_vec(_) {
        ret false;
      }
      ty::ty_str. { ret false; }
      ty::ty_ivec(_) { ret true; }
      ty::ty_istr. { ret true; }
      _ { cx.sess.bug("sequence_is_interior called on non-sequence type"); }
    }
}

fn sequence_element_type(cx: &ctxt, ty: &t) -> t {
    alt struct(cx, ty) {
      ty_str. { ret mk_mach(cx, ast::ty_u8); }
      ty_istr. { ret mk_mach(cx, ast::ty_u8); }
      ty_vec(mt) { ret mt.ty; }
      ty_ivec(mt) { ret mt.ty; }
      _ { cx.sess.bug("sequence_element_type called on non-sequence value"); }
    }
}

fn type_is_tup_like(cx: &ctxt, ty: &t) -> bool {
    alt struct(cx, ty) {
      ty_box(_) { ret true; }
      ty_rec(_) { ret true; }
      ty_tag(_, _) { ret true; }
      _ { ret false; }
    }
}

fn get_element_type(cx: &ctxt, ty: &t, i: uint) -> t {
    alt struct(cx, ty) {
      ty_rec(flds) { ret flds.(i).mt.ty; }
      _ {
        cx.sess.bug("get_element_type called on type " + ty_to_str(cx, ty) +
                        " - expected a \
            tuple or record");
      }
    }
    // NB: This is not exhaustive -- struct(cx, ty) could be a box or a
    // tag.
}

fn type_is_box(cx: &ctxt, ty: &t) -> bool {
    alt struct(cx, ty) { ty_box(_) { ret true; } _ { ret false; } }
}

fn type_is_boxed(cx: &ctxt, ty: &t) -> bool {
    alt struct(cx, ty) {
      ty_str. { ret true; }
      ty_vec(_) { ret true; }
      ty_box(_) { ret true; }
      ty_port(_) { ret true; }
      ty_chan(_) { ret true; }
      ty_task. { ret true; }
      _ { ret false; }
    }
}

fn type_is_scalar(cx: &ctxt, ty: &t) -> bool {
    alt struct(cx, ty) {
      ty_nil. { ret true; }
      ty_bool. { ret true; }
      ty_int. { ret true; }
      ty_float. { ret true; }
      ty_uint. { ret true; }
      ty_machine(_) { ret true; }
      ty_char. { ret true; }
      ty_type. { ret true; }
      ty_native(_) { ret true; }
      ty_ptr(_) { ret true; }
      _ { ret false; }
    }
}

fn type_has_pointers(cx: &ctxt, ty: &t) -> bool {
    alt cx.has_pointer_cache.find(ty) {
      some(result) { ret result; }
      none. {/* fall through */ }
    }

    let result = false;
    alt struct(cx, ty) {

      // scalar types
      ty_nil. {
        /* no-op */

      }
      ty_bot. {/* no-op */ }
      ty_bool. {/* no-op */ }
      ty_int. {/* no-op */ }
      ty_float. {/* no-op */ }
      ty_uint. {/* no-op */ }
      ty_machine(_) {/* no-op */ }
      ty_char. {/* no-op */ }
      ty_type. {/* no-op */ }
      ty_native(_) {/* no-op */ }
      ty_rec(flds) {
        for f: field  in flds {
            if type_has_pointers(cx, f.mt.ty) {
                result = true;
                break;
            }
        }
      }
      ty_tag(did, tps) {
        let variants = tag_variants(cx, did);
        for variant: variant_info  in variants {
            for aty: t  in variant.args {
                // Perform any type parameter substitutions.
                let arg_ty = substitute_type_params(cx, tps, aty);
                if type_has_pointers(cx, arg_ty) {
                    result = true;
                    break;
                }
            }
            if result { break; }
        }
      }
      ty_res(did, inner, tps) {
        result =
            type_has_pointers(cx, substitute_type_params(cx, tps, inner));
      }
      _ { result = true; }
    }

    cx.has_pointer_cache.insert(ty, result);
    ret result;
}

fn type_needs_drop(cx: &ctxt, ty: &t) -> bool {
    ret alt struct(cx, ty) {
      ty_res(_, _, _) { true }
      _ { type_has_pointers(cx, ty) }
    };
}

fn type_kind(cx: &ctxt, ty: &t) -> ast::kind {
    alt cx.kind_cache.find(ty) {
      some(result) { ret result; }
      none. {/* fall through */ }
    }

    let result = ast::kind_unique;

    // Insert a default in case we loop back on self recursively.
    cx.kind_cache.insert(ty, result);

    alt struct(cx, ty) {

      // Scalar types are unique-kind, no substructure.
      ty_nil. | ty_bot. | ty_bool. | ty_int. | ty_uint. | ty_float.
      | ty_machine(_) | ty_char. | ty_native(_) {
        // no-op
      }

      // A handful of other built-in are unique too.
      ty_type. | ty_istr. | ty_native_fn(_, _, _) {
        // no-op
      }

      // Those things with refcounts-to-interior are just shared.
      ty_str. | ty_task. {
        result = kind_shared;
      }

      // FIXME: obj is broken for now, since we aren't asserting
      // anything about its fields.
      ty_obj(_) { result = kind_shared; }

      // FIXME: the environment capture mode is not fully encoded
      // here yet, leading to weirdness around closure.
      ty_fn(proto, _, _, _, _) {
        result = alt proto {
          ast::proto_block. { ast::kind_pinned }
          ast::proto_closure. { ast::kind_shared }
          _ { ast::kind_unique }
        }
      }

      // Those with refcounts-to-inner raise pinned to shared,
      // lower unique to shared. Therefore just set result to shared.
      ty_box(mt) | ty_vec(mt) {
        result = ast::kind_shared;
      }

      // FIXME: remove ports. Ports currently contribute 'shared'
      ty_port(t) {
        result = kind::lower_kind(ast::kind_shared,
                                  type_kind(cx, t));
      }

      // FIXME: remove chans. Chans currently contribute only
      // their inner.
      ty_chan(t) {
        result = type_kind(cx, t);
      }

      // Pointers and unique boxes / vecs raise pinned to shared,
      // otherwise pass through their pointee kind.
      ty_ptr(tm) | ty_ivec(tm) {
        let k = type_kind(cx, tm.ty);
        if k == ast::kind_pinned { k = ast::kind_shared }
        result = kind::lower_kind(result, k);
      }

      // Records lower to the lowest of their members.
      ty_rec(flds) {
        for f: field  in flds {
            result = kind::lower_kind(result, type_kind(cx, f.mt.ty));
            if result == ast::kind_pinned { break; }
        }
      }

      // Tags lower to the lowest of their variants.
      ty_tag(did, tps) {
        let variants = tag_variants(cx, did);
        for variant: variant_info  in variants {
            for aty: t  in variant.args {
                // Perform any type parameter substitutions.
                let arg_ty = substitute_type_params(cx, tps, aty);
                result = kind::lower_kind(result, type_kind(cx, arg_ty));
                if result == ast::kind_pinned { break; }
            }
            if result == ast::kind_pinned { break; }
        }
      }

      // Resources are always pinned.
      ty_res(did, inner, tps) {
        result = ast::kind_pinned;
      }

      ty_var(_) { fail; }

      ty_param(_,k) {
        result = kind::lower_kind(result, k);
      }

      ty_constr(t, _) {
        result = type_kind(cx, t);
      }

        _ {
            cx.sess.bug("missed case: " + ty_to_str(cx, ty));
        }

    }

    cx.kind_cache.insert(ty, result);
    ret result;
}


// FIXME: should we just return true for native types in
// type_is_scalar?
fn type_is_native(cx: &ctxt, ty: &t) -> bool {
    alt struct(cx, ty) { ty_native(_) { ret true; } _ { ret false; } }
}

fn type_has_dynamic_size(cx: &ctxt, ty: &t) -> bool {
    alt struct(cx, ty) {
      ty_nil. { ret false; }
      ty_bot. { ret false; }
      ty_bool. { ret false; }
      ty_int. { ret false; }
      ty_float. { ret false; }
      ty_uint. { ret false; }
      ty_machine(_) { ret false; }
      ty_char. { ret false; }
      ty_str. { ret false; }
      ty_istr. { ret false; }
      ty_tag(_, subtys) {
        let i = 0u;
        while i < ivec::len[t](subtys) {
            if type_has_dynamic_size(cx, subtys.(i)) { ret true; }
            i += 1u;
        }
        ret false;
      }
      ty_box(_) { ret false; }
      ty_vec(_) { ret false; }
      ty_ivec(mt) { ret type_has_dynamic_size(cx, mt.ty); }
      ty_ptr(_) { ret false; }
      ty_port(_) { ret false; }
      ty_chan(_) { ret false; }
      ty_task. { ret false; }
      ty_rec(fields) {
        let i = 0u;
        while i < ivec::len[field](fields) {
            if type_has_dynamic_size(cx, fields.(i).mt.ty) { ret true; }
            i += 1u;
        }
        ret false;
      }
      ty_fn(_, _, _, _, _) { ret false; }
      ty_native_fn(_, _, _) { ret false; }
      ty_obj(_) { ret false; }
      ty_res(_, sub, tps) {
        for tp: t  in tps { if type_has_dynamic_size(cx, tp) { ret true; } }
        ret type_has_dynamic_size(cx, sub);
      }
      ty_var(_) { fail "ty_var in type_has_dynamic_size()"; }
      ty_param(_,_) { ret true; }
      ty_type. { ret false; }
      ty_native(_) { ret false; }
    }
}

fn type_is_integral(cx: &ctxt, ty: &t) -> bool {
    alt struct(cx, ty) {
      ty_int. { ret true; }
      ty_uint. { ret true; }
      ty_machine(m) {
        alt m {
          ast::ty_i8. { ret true; }
          ast::ty_i16. { ret true; }
          ast::ty_i32. { ret true; }
          ast::ty_i64. { ret true; }
          ast::ty_u8. { ret true; }
          ast::ty_u16. { ret true; }
          ast::ty_u32. { ret true; }
          ast::ty_u64. { ret true; }
          _ { ret false; }
        }
      }
      ty_char. { ret true; }
      ty_bool. { ret true; }
      _ { ret false; }
    }
}

fn type_is_fp(cx: &ctxt, ty: &t) -> bool {
    alt struct(cx, ty) {
      ty_machine(tm) {
        alt tm {
          ast::ty_f32. { ret true; }
          ast::ty_f64. { ret true; }
          _ { ret false; }
        }
      }
      ty_float. { ret true; }
      _ { ret false; }
    }
}

fn type_is_signed(cx: &ctxt, ty: &t) -> bool {
    alt struct(cx, ty) {
      ty_int. { ret true; }
      ty_machine(tm) {
        alt tm {
          ast::ty_i8. { ret true; }
          ast::ty_i16. { ret true; }
          ast::ty_i32. { ret true; }
          ast::ty_i64. { ret true; }
          _ { ret false; }
        }
      }
      _ { ret false; }
    }
}

fn type_owns_heap_mem(cx: &ctxt, ty: &t) -> bool {
    alt cx.owns_heap_mem_cache.find(ty) {
      some(result) { ret result; }
      none. {/* fall through */ }
    }

    let result = false;
    alt struct(cx, ty) {
      ty_ivec(_) { result = true; }
      ty_istr. { result = true; }


      // scalar types
      ty_nil. {
        result = false;
      }
      ty_bot. { result = false; }
      ty_bool. { result = false; }
      ty_int. { result = false; }
      ty_float. { result = false; }
      ty_uint. { result = false; }
      ty_machine(_) { result = false; }
      ty_char. { result = false; }
      ty_type. { result = false; }
      ty_native(_) { result = false; }


      // boxed types
      ty_str. {
        result = false;
      }
      ty_box(_) { result = false; }
      ty_vec(_) { result = false; }
      ty_fn(_, _, _, _, _) { result = false; }
      ty_native_fn(_, _, _) { result = false; }
      ty_obj(_) { result = false; }


      // structural types
      ty_tag(did, tps) {
        let variants = tag_variants(cx, did);
        for variant: variant_info  in variants {
            for aty: t  in variant.args {
                // Perform any type parameter substitutions.
                let arg_ty = substitute_type_params(cx, tps, aty);
                if type_owns_heap_mem(cx, arg_ty) { result = true; }
            }
        }
      }
      ty_rec(flds) {
        for f: field  in flds {
            if type_owns_heap_mem(cx, f.mt.ty) { result = true; }
        }
      }
      ty_res(_, inner, tps) {
        result =
            type_owns_heap_mem(cx, substitute_type_params(cx, tps, inner));
      }


      ty_ptr(_) {
        result = false;
      }
      ty_port(_) { result = false; }
      ty_chan(_) { result = false; }
      ty_task. { result = false; }
      ty_var(_) { fail "ty_var in type_owns_heap_mem"; }
      ty_param(_,_) { result = false; }
    }

    cx.owns_heap_mem_cache.insert(ty, result);
    ret result;
}

fn type_param(cx: &ctxt, ty: &t) -> option::t[uint] {
    alt struct(cx, ty) {
      ty_param(id,_) { ret some(id); }
      _ {/* fall through */ }
    }
    ret none;
}

fn type_autoderef(cx: &ctxt, t: &ty::t) -> ty::t {
    let t1: ty::t = t;
    while true {
        alt struct(cx, t1) {
          ty::ty_box(mt) { t1 = mt.ty; }
          ty::ty_res(_, inner, tps) {
            t1 = substitute_type_params(cx, tps, inner);
          }
          ty::ty_tag(did, tps) {
            let variants = tag_variants(cx, did);
            if ivec::len(variants) != 1u || ivec::len(variants.(0).args) != 1u
               {
                break;
            }
            t1 = substitute_type_params(cx, tps, variants.(0).args.(0));
          }
          _ { break; }
        }
    }
    ret t1;
}

// Type hashing. This function is private to this module (and slow); external
// users should use `hash_ty()` instead.
fn hash_type_structure(st: &sty) -> uint {
    fn hash_uint(id: uint, n: uint) -> uint {
        let h = id;
        h += h << 5u + n;
        ret h;
    }
    fn hash_def(id: uint, did: ast::def_id) -> uint {
        let h = id;
        h += h << 5u + (did.crate as uint);
        h += h << 5u + (did.node as uint);
        ret h;
    }
    fn hash_subty(id: uint, subty: &t) -> uint {
        let h = id;
        h += h << 5u + hash_ty(subty);
        ret h;
    }
    fn hash_type_constr(id: uint, c: &@type_constr) -> uint {
        let h = id;
        h += h << 5u + hash_def(h, c.node.id);
        ret hash_type_constr_args(h, c.node.args);
    }
    fn hash_type_constr_args(id: uint, args: (@ty_constr_arg)[]) -> uint {
        let h = id;
        for a: @ty_constr_arg  in args {
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


    fn hash_fn(id: uint, args: &arg[], rty: &t) -> uint {
        let h = id;
        for a: arg  in args { h += h << 5u + hash_ty(a.ty); }
        h += h << 5u + hash_ty(rty);
        ret h;
    }
    alt st {
      ty_nil. { ret 0u; }
      ty_bool. { ret 1u; }
      ty_int. { ret 2u; }
      ty_float. { ret 3u; }
      ty_uint. { ret 4u; }
      ty_machine(tm) {
        alt tm {
          ast::ty_i8. { ret 5u; }
          ast::ty_i16. { ret 6u; }
          ast::ty_i32. { ret 7u; }
          ast::ty_i64. { ret 8u; }
          ast::ty_u8. { ret 9u; }
          ast::ty_u16. { ret 10u; }
          ast::ty_u32. { ret 11u; }
          ast::ty_u64. { ret 12u; }
          ast::ty_f32. { ret 13u; }
          ast::ty_f64. { ret 14u; }
        }
      }
      ty_char. { ret 15u; }
      ty_str. { ret 16u; }
      ty_istr. { ret 17u; }
      ty_tag(did, tys) {
        let h = hash_def(18u, did);
        for typ: t  in tys { h += h << 5u + hash_ty(typ); }
        ret h;
      }
      ty_box(mt) { ret hash_subty(19u, mt.ty); }
      ty_vec(mt) { ret hash_subty(20u, mt.ty); }
      ty_ivec(mt) { ret hash_subty(21u, mt.ty); }
      ty_port(typ) { ret hash_subty(22u, typ); }
      ty_chan(typ) { ret hash_subty(23u, typ); }
      ty_task. { ret 24u; }
      ty_rec(fields) {
        let h = 26u;
        for f: field  in fields { h += h << 5u + hash_ty(f.mt.ty); }
        ret h;
      }

      // ???
      ty_fn(_, args, rty, _, _) {
        ret hash_fn(27u, args, rty);
      }
      ty_native_fn(_, args, rty) { ret hash_fn(28u, args, rty); }
      ty_obj(methods) {
        let h = 29u;
        for m: method  in methods { h += h << 5u + str::hash(m.ident); }
        ret h;
      }
      ty_var(v) { ret hash_uint(30u, v as uint); }
      ty_param(pid,_) { ret hash_uint(31u, pid); }
      ty_type. { ret 32u; }
      ty_native(did) { ret hash_def(33u, did); }
      ty_bot. { ret 34u; }
      ty_ptr(mt) { ret hash_subty(35u, mt.ty); }
      ty_res(did, sub, tps) {
        let h = hash_subty(hash_def(18u, did), sub);
        for tp: t  in tps { h += h << 5u + hash_ty(tp); }
        ret h;
      }
      ty_constr(t, cs) {
        let h = 36u;
        for c: @type_constr  in cs { h += h << 5u + hash_type_constr(h, c); }
        ret h;
      }
    }
}

fn hash_type_info(st: &sty, cname_opt: &option::t[str]) -> uint {
    let h = hash_type_structure(st);
    alt cname_opt {
      none. {/* no-op */ }
      some(s) { h += h << 5u + str::hash(s); }
    }
    ret h;
}

fn hash_raw_ty(rt: &@raw_t) -> uint { ret rt.hash; }

fn hash_ty(typ: &t) -> uint { ret typ; }


// Type equality. This function is private to this module (and slow); external
// users should use `eq_ty()` instead.
fn eq_int(x: &uint, y: &uint) -> bool { ret x == y; }

fn arg_eq[T](eq: &fn(&T, &T) -> bool , a: @sp_constr_arg[T],
             b: @sp_constr_arg[T]) -> bool {
    alt a.node {
      ast::carg_base. {
        alt b.node { ast::carg_base. { ret true; } _ { ret false; } }
      }
      ast::carg_ident(s) {
        alt b.node { ast::carg_ident(t) { ret eq(s, t); } _ { ret false; } }
      }
      ast::carg_lit(l) {
        alt b.node { ast::carg_lit(m) { ret lit_eq(l, m); } _ { ret false; } }
      }
    }
}

fn args_eq[T](eq: fn(&T, &T) -> bool , a: &(@sp_constr_arg[T])[],
              b: &(@sp_constr_arg[T])[]) -> bool {
    let i: uint = 0u;
    for arg: @sp_constr_arg[T]  in a {
        if !arg_eq(eq, arg, b.(i)) { ret false; }
        i += 1u;
    }
    ret true;
}

fn constr_eq(c: &@constr, d: &@constr) -> bool {
    ret path_to_str(c.node.path) == path_to_str(d.node.path) &&
            // FIXME: hack
            args_eq(eq_int, c.node.args, d.node.args);
}

fn constrs_eq(cs: &(@constr)[], ds: &(@constr)[]) -> bool {
    if ivec::len(cs) != ivec::len(ds) { ret false; }
    let i = 0u;
    for c: @constr  in cs { if !constr_eq(c, ds.(i)) { ret false; } i += 1u; }
    ret true;
}

fn equal_type_structures(a: &sty, b: &sty) -> bool {
    fn equal_mt(a: &mt, b: &mt) -> bool {
        ret a.mut == b.mut && eq_ty(a.ty, b.ty);
    }
    fn equal_fn(args_a: &arg[], rty_a: &t, args_b: &arg[], rty_b: &t) ->
       bool {
        if !eq_ty(rty_a, rty_b) { ret false; }
        let len = ivec::len[arg](args_a);
        if len != ivec::len[arg](args_b) { ret false; }
        let i = 0u;
        while i < len {
            let arg_a = args_a.(i);
            let arg_b = args_b.(i);
            if arg_a.mode != arg_b.mode { ret false; }
            if !eq_ty(arg_a.ty, arg_b.ty) { ret false; }
            i += 1u;
        }
        ret true;
    }
    fn equal_def(did_a: &ast::def_id, did_b: &ast::def_id) -> bool {
        ret did_a.crate == did_b.crate && did_a.node == did_b.node;
    }
    alt a {
      ty_nil. { alt b { ty_nil. { ret true; } _ { ret false; } } }
      ty_bot. { alt b { ty_bot. { ret true; } _ { ret false; } } }
      ty_bool. { alt b { ty_bool. { ret true; } _ { ret false; } } }
      ty_int. { alt b { ty_int. { ret true; } _ { ret false; } } }
      ty_float. { alt b { ty_float. { ret true; } _ { ret false; } } }
      ty_uint. { alt b { ty_uint. { ret true; } _ { ret false; } } }
      ty_machine(tm_a) {
        alt b {
          ty_machine(tm_b) {
            ret hash_type_structure(a) == hash_type_structure(b);
          }
          _ { ret false; }
        }
      }
      ty_char. { alt b { ty_char. { ret true; } _ { ret false; } } }
      ty_str. { alt b { ty_str. { ret true; } _ { ret false; } } }
      ty_istr. { alt b { ty_istr. { ret true; } _ { ret false; } } }
      ty_tag(id_a, tys_a) {
        alt b {
          ty_tag(id_b, tys_b) {
            if !equal_def(id_a, id_b) { ret false; }
            let len = ivec::len[t](tys_a);
            if len != ivec::len[t](tys_b) { ret false; }
            let i = 0u;
            while i < len {
                if !eq_ty(tys_a.(i), tys_b.(i)) { ret false; }
                i += 1u;
            }
            ret true;
          }
          _ { ret false; }
        }
      }
      ty_box(mt_a) {
        alt b { ty_box(mt_b) { ret equal_mt(mt_a, mt_b); } _ { ret false; } }
      }
      ty_vec(mt_a) {
        alt b { ty_vec(mt_b) { ret equal_mt(mt_a, mt_b); } _ { ret false; } }
      }
      ty_ivec(mt_a) {
        alt b { ty_ivec(mt_b) { ret equal_mt(mt_a, mt_b); } _ { ret false; } }
      }
      ty_ptr(mt_a) {
        alt b { ty_ptr(mt_b) { ret equal_mt(mt_a, mt_b); } _ { ret false; } }
      }
      ty_port(t_a) {
        alt b { ty_port(t_b) { ret eq_ty(t_a, t_b); } _ { ret false; } }
      }
      ty_chan(t_a) {
        alt b { ty_chan(t_b) { ret eq_ty(t_a, t_b); } _ { ret false; } }
      }
      ty_task. { alt b { ty_task. { ret true; } _ { ret false; } } }
      ty_rec(flds_a) {
        alt b {
          ty_rec(flds_b) {
            let len = ivec::len[field](flds_a);
            if len != ivec::len[field](flds_b) { ret false; }
            let i = 0u;
            while i < len {
                let fld_a = flds_a.(i);
                let fld_b = flds_b.(i);
                if !str::eq(fld_a.ident, fld_b.ident) ||
                       !equal_mt(fld_a.mt, fld_b.mt) {
                    ret false;
                }
                i += 1u;
            }
            ret true;
          }
          _ { ret false; }
        }
      }
      ty_fn(p_a, args_a, rty_a, cf_a, constrs_a) {
        alt b {
          ty_fn(p_b, args_b, rty_b, cf_b, constrs_b) {
            ret p_a == p_b && cf_a == cf_b && constrs_eq(constrs_a, constrs_b)
                    && equal_fn(args_a, rty_a, args_b, rty_b);
          }
          _ { ret false; }
        }
      }
      ty_native_fn(abi_a, args_a, rty_a) {
        alt b {
          ty_native_fn(abi_b, args_b, rty_b) {
            ret abi_a == abi_b && equal_fn(args_a, rty_a, args_b, rty_b);
          }
          _ { ret false; }
        }
      }
      ty_obj(methods_a) {
        alt b {
          ty_obj(methods_b) {
            let len = ivec::len[method](methods_a);
            if len != ivec::len[method](methods_b) { ret false; }
            let i = 0u;
            while i < len {
                let m_a = methods_a.(i);
                let m_b = methods_b.(i);
                if m_a.proto != m_b.proto || !str::eq(m_a.ident, m_b.ident) ||
                       !equal_fn(m_a.inputs, m_a.output, m_b.inputs,
                                 m_b.output) {
                    ret false;
                }
                i += 1u;
            }
            ret true;
          }
          _ { ret false; }
        }
      }
      ty_res(id_a, inner_a, tps_a) {
        alt b {
          ty_res(id_b, inner_b, tps_b) {
            if !equal_def(id_a, id_b) || !eq_ty(inner_a, inner_b) {
                ret false;
            }
            let i = 0u;
            for tp_a: t  in tps_a {
                if !eq_ty(tp_a, tps_b.(i)) { ret false; }
                i += 1u;
            }
            ret true;
          }
          _ { ret false; }
        }
      }
      ty_var(v_a) {
        alt b { ty_var(v_b) { ret v_a == v_b; } _ { ret false; } }
      }
      ty_param(pid_a,k_a) {
        alt b { ty_param(pid_b,k_b) { ret pid_a == pid_b && k_a == k_b; }
               _ { ret false; } }
      }
      ty_type. { alt b { ty_type. { ret true; } _ { ret false; } } }
      ty_native(a_id) {
        alt b {
          ty_native(b_id) {
            ret a_id.crate == b_id.crate && a_id.node == b_id.node;
          }
          _ { ret false; }
        }
      }
    }
}


// An expensive type equality function. This function is private to this
// module.
//
// FIXME: Use structural comparison, but this loops forever and segfaults.
fn eq_raw_ty(a: &@raw_t, b: &@raw_t) -> bool {
    // Check hashes (fast path).

    if a.hash != b.hash { ret false; }
    // Check canonical names.

    alt a.cname {
      none. { alt b.cname { none. {/* ok */ } _ { ret false; } } }
      some(s_a) {
        alt b.cname {
          some(s_b) { if !str::eq(s_a, s_b) { ret false; } }
          _ { ret false; }
        }
      }
    }
    // Check structures.

    ret equal_type_structures(a.struct, b.struct);
}


// This is the equality function the public should use. It works as long as
// the types are interned.
fn eq_ty(a: &t, b: &t) -> bool { ret a == b; }


// Type lookups
fn node_id_to_ty_param_substs_opt_and_ty(cx: &ctxt, id: &ast::node_id) ->
   ty_param_substs_opt_and_ty {


    // Pull out the node type table.
    alt smallintmap::find(*cx.node_types, id as uint) {
      none. {
        cx.sess.bug("node_id_to_ty_param_substs_opt_and_ty() called on " +
                        "an untyped node (" + std::int::to_str(id, 10u) +
                        ")");
      }
      some(tpot) { ret tpot; }
    }
}

fn node_id_to_type(cx: &ctxt, id: &ast::node_id) -> t {
    ret node_id_to_ty_param_substs_opt_and_ty(cx, id).ty;
}

fn node_id_to_type_params(cx: &ctxt, id: &ast::node_id) -> t[] {
    alt node_id_to_ty_param_substs_opt_and_ty(cx, id).substs {
      none. { ret ~[]; }
      some(tps) { ret tps; }
    }
}

fn node_id_has_type_params(cx: &ctxt, id: &ast::node_id) -> bool {
    ret ivec::len(node_id_to_type_params(cx, id)) > 0u;
}


// Returns a type with type parameter substitutions performed if applicable.
fn ty_param_substs_opt_and_ty_to_monotype(cx: &ctxt,
                                          tpot: &ty_param_substs_opt_and_ty)
   -> t {
    alt tpot.substs {
      none. { ret tpot.ty; }
      some(tps) { ret substitute_type_params(cx, tps, tpot.ty); }
    }
}


// Returns the type of an annotation, with type parameter substitutions
// performed if applicable.
fn node_id_to_monotype(cx: &ctxt, id: ast::node_id) -> t {
    let tpot = node_id_to_ty_param_substs_opt_and_ty(cx, id);
    ret ty_param_substs_opt_and_ty_to_monotype(cx, tpot);
}


// Returns the number of distinct type parameters in the given type.
fn count_ty_params(cx: &ctxt, ty: t) -> uint {
    fn counter(cx: &ctxt, param_indices: @mutable uint[], ty: t) {
        alt struct(cx, ty) {
          ty_param(param_idx,_) {
            let seen = false;
            for other_param_idx: uint  in *param_indices {
                if param_idx == other_param_idx { seen = true; }
            }
            if !seen { *param_indices += ~[param_idx]; }
          }
          _ {/* fall through */ }
        }
    }
    let param_indices: @mutable uint[] = @mutable ~[];
    let f = bind counter(cx, param_indices, _);
    walk_ty(cx, f, ty);
    ret ivec::len[uint](*param_indices);
}

fn type_contains_vars(cx: &ctxt, typ: &t) -> bool {
    ret interner::get(*cx.ts, typ).has_vars;
}

fn type_contains_params(cx: &ctxt, typ: &t) -> bool {
    ret interner::get(*cx.ts, typ).has_params;
}


// Type accessors for substructures of types
fn ty_fn_args(cx: &ctxt, fty: &t) -> arg[] {
    alt struct(cx, fty) {
      ty::ty_fn(_, a, _, _, _) { ret a; }
      ty::ty_native_fn(_, a, _) { ret a; }
      _ { cx.sess.bug("ty_fn_args() called on non-fn type"); }
    }
}

fn ty_fn_proto(cx: &ctxt, fty: &t) -> ast::proto {
    alt struct(cx, fty) {
      ty::ty_fn(p, _, _, _, _) { ret p; }
      _ { cx.sess.bug("ty_fn_proto() called on non-fn type"); }
    }
}

fn ty_fn_abi(cx: &ctxt, fty: &t) -> ast::native_abi {
    alt struct(cx, fty) {
      ty::ty_native_fn(a, _, _) { ret a; }
      _ { cx.sess.bug("ty_fn_abi() called on non-native-fn type"); }
    }
}

fn ty_fn_ret(cx: &ctxt, fty: &t) -> t {
    alt struct(cx, fty) {
      ty::ty_fn(_, _, r, _, _) { ret r; }
      ty::ty_native_fn(_, _, r) { ret r; }
      _ { cx.sess.bug("ty_fn_ret() called on non-fn type"); }
    }
}

fn is_fn_ty(cx: &ctxt, fty: &t) -> bool {
    alt struct(cx, fty) {
      ty::ty_fn(_, _, _, _, _) { ret true; }
      ty::ty_native_fn(_, _, _) { ret true; }
      _ { ret false; }
    }
}

fn ty_var_id(cx: &ctxt, typ: t) -> int {
    alt struct(cx, typ) {
      ty::ty_var(vid) { ret vid; }
      _ { log_err "ty_var_id called on non-var ty"; fail; }
    }
}


// Type accessors for AST nodes
fn block_ty(cx: &ctxt, b: &ast::blk) -> t {
    ret node_id_to_type(cx, b.node.id);
}


// Returns the type of a pattern as a monotype. Like @expr_ty, this function
// doesn't provide type parameter substitutions.
fn pat_ty(cx: &ctxt, pat: &@ast::pat) -> t {
    ret node_id_to_monotype(cx, pat.id);
}


// Returns the type of an expression as a monotype.
//
// NB: This type doesn't provide type parameter substitutions; e.g. if you
// ask for the type of "id" in "id(3)", it will return "fn(&int) -> int"
// instead of "fn(&T) -> T with T = int". If this isn't what you want, see
// expr_ty_params_and_ty() below.
fn expr_ty(cx: &ctxt, expr: &@ast::expr) -> t {
    ret node_id_to_monotype(cx, expr.id);
}

fn expr_ty_params_and_ty(cx: &ctxt, expr: &@ast::expr) ->
   {params: t[], ty: t} {
    ret {params: node_id_to_type_params(cx, expr.id),
         ty: node_id_to_type(cx, expr.id)};
}

fn expr_has_ty_params(cx: &ctxt, expr: &@ast::expr) -> bool {
    ret node_id_has_type_params(cx, expr.id);
}

fn decl_local_ty(cx: &ctxt, l: &@ast::local) -> t {
    ret node_id_to_type(cx, l.node.id);
}

fn stmt_node_id(s: &@ast::stmt) -> ast::node_id {
    alt s.node {
      ast::stmt_decl(_, id) { ret id; }
      ast::stmt_expr(_, id) { ret id; }
      ast::stmt_crate_directive(_) {
        log_err "ty::stmt_node_id(): crate directive found";
        fail;
      }
    }
}

fn field_idx(sess: &session::session, sp: &span, id: &ast::ident,
             fields: &field[]) -> uint {
    let i: uint = 0u;
    for f: field  in fields { if str::eq(f.ident, id) { ret i; } i += 1u; }
    sess.span_fatal(sp, "unknown field '" + id + "' of record");
}

fn method_idx(sess: &session::session, sp: &span, id: &ast::ident,
              meths: &method[]) -> uint {
    let i: uint = 0u;
    for m: method  in meths { if str::eq(m.ident, id) { ret i; } i += 1u; }
    sess.span_fatal(sp, "unknown method '" + id + "' of obj");
}

fn sort_methods(meths: &method[]) -> method[] {
    fn method_lteq(a: &method, b: &method) -> bool {
        ret str::lteq(a.ident, b.ident);
    }
    ret std::sort::ivector::merge_sort[method](bind method_lteq(_, _), meths);
}

fn is_lval(expr: &@ast::expr) -> bool {
    alt expr.node {
      ast::expr_field(_, _) { ret true; }
      ast::expr_index(_, _) { ret true; }
      ast::expr_path(_) { ret true; }
      ast::expr_unary(ast::deref., _) { ret true; }
      _ { ret false; }
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
        {sets: ufindivec::ufind, types: smallintmap::smallintmap[t]};

    type ctxt = {vb: @var_bindings, tcx: ty_ctxt};

    fn mk_var_bindings() -> @var_bindings {
        ret @{sets: ufindivec::make(), types: smallintmap::mk[t]()};
    }

    // Unifies two sets.
    fn union(cx: &@ctxt, set_a: uint, set_b: uint) -> union_result {
        ufindivec::grow(cx.vb.sets, uint::max(set_a, set_b) + 1u);
        let root_a = ufindivec::find(cx.vb.sets, set_a);
        let root_b = ufindivec::find(cx.vb.sets, set_b);

        let replace_type =
            bind fn (cx: &@ctxt, t: t, set_a: uint, set_b: uint) {
                     ufindivec::union(cx.vb.sets, set_a, set_b);
                     let root_c: uint = ufindivec::find(cx.vb.sets, set_a);
                     smallintmap::insert[t](cx.vb.types, root_c, t);
                 }(_, _, set_a, set_b);


        alt smallintmap::find(cx.vb.types, root_a) {
          none. {
            alt smallintmap::find(cx.vb.types, root_b) {
              none. {
                ufindivec::union(cx.vb.sets, set_a, set_b);
                ret unres_ok;
              }
              some(t_b) { replace_type(cx, t_b); ret unres_ok; }
            }
          }
          some(t_a) {
            alt smallintmap::find(cx.vb.types, root_b) {
              none. { replace_type(cx, t_a); ret unres_ok; }
              some(t_b) {
                alt unify_step(cx, t_a, t_b) {
                  ures_ok(t_c) { replace_type(cx, t_c); ret unres_ok; }
                  ures_err(terr) { ret unres_err(terr); }
                }
              }
            }
          }
        }
    }
    fn record_var_binding(cx: &@ctxt, key: int, typ: t) -> result {
        ufindivec::grow(cx.vb.sets, (key as uint) + 1u);
        let root = ufindivec::find(cx.vb.sets, key as uint);
        let result_type = typ;
        alt smallintmap::find[t](cx.vb.types, root) {
          some(old_type) {
            alt unify_step(cx, old_type, typ) {
              ures_ok(unified_type) { result_type = unified_type; }
              rs { ret rs; }
            }
          }
          none. {/* fall through */ }
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
    fn struct_cmp(cx: @ctxt, expected: t, actual: t) -> result {
        if struct(cx.tcx, expected) == struct(cx.tcx, actual) {
            ret ures_ok(expected);
        }
        ret ures_err(terr_mismatch);
    }

    // Right now this just checks that the lists of constraints are
    // pairwise equal.
    fn unify_constrs(base_t: &t, expected: (@type_constr)[],
                     actual: &(@type_constr)[]) -> result {
        let expected_len = ivec::len(expected);
        let actual_len = ivec::len(actual);

        if expected_len != actual_len {
            ret ures_err(terr_constr_len(expected_len, actual_len));
        }
        let i = 0u;
        let rslt;
        for c: @type_constr  in expected {
            rslt = unify_constr(base_t, c, actual.(i));
            alt rslt { ures_ok(_) { } ures_err(_) { ret rslt; } }
            i += 1u;
        }
        ret ures_ok(base_t);
    }
    fn unify_constr(base_t: &t, expected: @type_constr,
                    actual_constr: &@type_constr) -> result {
        let ok_res = ures_ok(base_t);
        let err_res = ures_err(terr_constr_mismatch(expected, actual_constr));
        if expected.node.id != actual_constr.node.id { ret err_res; }
        let expected_arg_len = ivec::len(expected.node.args);
        let actual_arg_len = ivec::len(actual_constr.node.args);
        if expected_arg_len != actual_arg_len { ret err_res; }
        let i = 0u;
        let actual;
        for a: @ty_constr_arg  in expected.node.args {
            actual = actual_constr.node.args.(i);
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
    fn unify_mut(expected: ast::mutability, actual: ast::mutability) ->
       option::t[ast::mutability] {
        if expected == actual { ret some(expected); }
        if expected == ast::maybe_mut { ret some(actual); }
        if actual == ast::maybe_mut { ret some(expected); }
        ret none;
    }
    tag fn_common_res {
        fn_common_res_err(result);
        fn_common_res_ok(arg[], t);
    }
    fn unify_fn_common(cx: &@ctxt, expected: &t, actual: &t,
                       expected_inputs: &arg[], expected_output: &t,
                       actual_inputs: &arg[], actual_output: &t) ->
       fn_common_res {
        let expected_len = ivec::len[arg](expected_inputs);
        let actual_len = ivec::len[arg](actual_inputs);
        if expected_len != actual_len {
            ret fn_common_res_err(ures_err(terr_arg_count));
        }
        // TODO: as above, we should have an iter2 iterator.

        let result_ins: arg[] = ~[];
        let i = 0u;
        while i < expected_len {
            let expected_input = expected_inputs.(i);
            let actual_input = actual_inputs.(i);
            // Unify the result modes.

            let result_mode;
            if expected_input.mode != actual_input.mode {
                ret fn_common_res_err
                    (ures_err(terr_mode_mismatch(expected_input.mode,
                                                 actual_input.mode)));
            } else { result_mode = expected_input.mode; }
            let result = unify_step(cx, expected_input.ty, actual_input.ty);
            alt result {
              ures_ok(rty) { result_ins += ~[{mode: result_mode, ty: rty}]; }
              _ { ret fn_common_res_err(result); }
            }
            i += 1u;
        }
        // Check the output.

        let result = unify_step(cx, expected_output, actual_output);
        alt result {
          ures_ok(rty) { ret fn_common_res_ok(result_ins, rty); }
          _ { ret fn_common_res_err(result); }
        }
    }
    fn unify_fn(cx: &@ctxt, e_proto: &ast::proto, a_proto: &ast::proto,
                expected: &t, actual: &t, expected_inputs: &arg[],
                expected_output: &t, actual_inputs: &arg[], actual_output: &t,
                expected_cf: &controlflow, actual_cf: &controlflow,
                expected_constrs: &(@constr)[], actual_constrs: &(@constr)[])
       -> result {
        if e_proto != a_proto { ret ures_err(terr_mismatch); }
        alt expected_cf {
          ast::return. { }
           // ok
          ast::noreturn. {
            alt actual_cf {
              ast::noreturn. {
                // ok

              }
              _ {
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
        let t =
            unify_fn_common(cx, expected, actual, expected_inputs,
                            expected_output, actual_inputs, actual_output);
        alt t {
          fn_common_res_err(r) { ret r; }
          fn_common_res_ok(result_ins, result_out) {
            let t2 =
                mk_fn(cx.tcx, e_proto, result_ins, result_out, actual_cf,
                      actual_constrs);
            ret ures_ok(t2);
          }
        }
    }
    fn unify_native_fn(cx: &@ctxt, e_abi: &ast::native_abi,
                       a_abi: &ast::native_abi, expected: &t, actual: &t,
                       expected_inputs: &arg[], expected_output: &t,
                       actual_inputs: &arg[], actual_output: &t) -> result {
        if e_abi != a_abi { ret ures_err(terr_mismatch); }
        let t =
            unify_fn_common(cx, expected, actual, expected_inputs,
                            expected_output, actual_inputs, actual_output);
        alt t {
          fn_common_res_err(r) { ret r; }
          fn_common_res_ok(result_ins, result_out) {
            let t2 = mk_native_fn(cx.tcx, e_abi, result_ins, result_out);
            ret ures_ok(t2);
          }
        }
    }
    fn unify_obj(cx: &@ctxt, expected: &t, actual: &t,
                 expected_meths: &method[], actual_meths: &method[]) ->
       result {
        let result_meths: method[] = ~[];
        let i: uint = 0u;
        let expected_len: uint = ivec::len[method](expected_meths);
        let actual_len: uint = ivec::len[method](actual_meths);
        if expected_len != actual_len { ret ures_err(terr_meth_count); }
        while i < expected_len {
            let e_meth = expected_meths.(i);
            let a_meth = actual_meths.(i);
            if !str::eq(e_meth.ident, a_meth.ident) {
                ret ures_err(terr_obj_meths(e_meth.ident, a_meth.ident));
            }
            let r =
                unify_fn(cx, e_meth.proto, a_meth.proto, expected, actual,
                         e_meth.inputs, e_meth.output, a_meth.inputs,
                         a_meth.output, e_meth.cf, a_meth.cf, e_meth.constrs,
                         a_meth.constrs);
            alt r {
              ures_ok(tfn) {
                alt struct(cx.tcx, tfn) {
                  ty_fn(proto, ins, out, cf, constrs) {
                    result_meths +=
                        ~[{inputs: ins, output: out, cf: cf, constrs: constrs
                              with e_meth}];
                  }
                }
              }
              _ { ret r; }
            }
            i += 1u;
        }
        let t = mk_obj(cx.tcx, result_meths);
        ret ures_ok(t);
    }

    // If the given type is a variable, returns the structure of that type.
    fn resolve_type_structure(tcx: &ty_ctxt, vb: &@var_bindings, typ: t) ->
       fixup_result {
        alt struct(tcx, typ) {
          ty_var(vid) {
            if vid as uint >= ufindivec::set_count(vb.sets) {
                ret fix_err(vid);
            }
            let root_id = ufindivec::find(vb.sets, vid as uint);
            alt smallintmap::find[t](vb.types, root_id) {
              none. { ret fix_err(vid); }
              some(rt) { ret fix_ok(rt); }
            }
          }
          _ { ret fix_ok(typ); }
        }
    }
    fn unify_step(cx: &@ctxt, expected: &t, actual: &t) -> result {
        // TODO: rewrite this using tuple pattern matching when available, to
        // avoid all this rightward drift and spikiness.

        // TODO: occurs check, to make sure we don't loop forever when
        // unifying e.g. 'a and option['a]

        // Fast path.

        if eq_ty(expected, actual) { ret ures_ok(expected); }
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
                alt union(cx, expected_n, actual_n) {
                  unres_ok. {/* fall through */ }
                  unres_err(t_e) { ret ures_err(t_e); }
                }
              }
              _ {


                // Just bind the type variable to the expected type.
                alt record_var_binding(cx, actual_id, expected) {
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

            alt record_var_binding(cx, expected_id, actual) {
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
          ty::ty_bool. { ret struct_cmp(cx, expected, actual); }
          ty::ty_int. { ret struct_cmp(cx, expected, actual); }
          ty::ty_uint. { ret struct_cmp(cx, expected, actual); }
          ty::ty_machine(_) { ret struct_cmp(cx, expected, actual); }
          ty::ty_float. { ret struct_cmp(cx, expected, actual); }
          ty::ty_char. { ret struct_cmp(cx, expected, actual); }
          ty::ty_str. { ret struct_cmp(cx, expected, actual); }
          ty::ty_istr. { ret struct_cmp(cx, expected, actual); }
          ty::ty_type. { ret struct_cmp(cx, expected, actual); }
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
          ty::ty_param(_,_) { ret struct_cmp(cx, expected, actual); }
          ty::ty_tag(expected_id, expected_tps) {
            alt struct(cx.tcx, actual) {
              ty::ty_tag(actual_id, actual_tps) {
                if expected_id.crate != actual_id.crate ||
                       expected_id.node != actual_id.node {
                    ret ures_err(terr_mismatch);
                }
                // TODO: factor this cruft out
                let result_tps: t[] = ~[];
                let i = 0u;
                let expected_len = ivec::len[t](expected_tps);
                while i < expected_len {
                    let expected_tp = expected_tps.(i);
                    let actual_tp = actual_tps.(i);
                    let result = unify_step(cx, expected_tp, actual_tp);
                    alt result {
                      ures_ok(rty) { result_tps += ~[rty]; }
                      _ { ret result; }
                    }
                    i += 1u;
                }
                ret ures_ok(mk_tag(cx.tcx, expected_id, result_tps));
              }
              _ {/* fall through */ }
            }
            ret ures_err(terr_mismatch);
          }
          ty::ty_box(expected_mt) {
            alt struct(cx.tcx, actual) {
              ty::ty_box(actual_mt) {
                let mut;
                alt unify_mut(expected_mt.mut, actual_mt.mut) {
                  none. { ret ures_err(terr_box_mutability); }
                  some(m) { mut = m; }
                }
                let result = unify_step(cx, expected_mt.ty, actual_mt.ty);
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
          ty::ty_vec(expected_mt) {
            alt struct(cx.tcx, actual) {
              ty::ty_vec(actual_mt) {
                let mut;
                alt unify_mut(expected_mt.mut, actual_mt.mut) {
                  none. { ret ures_err(terr_vec_mutability); }
                  some(m) { mut = m; }
                }
                let result = unify_step(cx, expected_mt.ty, actual_mt.ty);
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
          ty::ty_ivec(expected_mt) {
            alt struct(cx.tcx, actual) {
              ty::ty_ivec(actual_mt) {
                let mut;
                alt unify_mut(expected_mt.mut, actual_mt.mut) {
                  none. { ret ures_err(terr_vec_mutability); }
                  some(m) { mut = m; }
                }
                let result = unify_step(cx, expected_mt.ty, actual_mt.ty);
                alt result {
                  ures_ok(result_sub) {
                    let mt = {ty: result_sub, mut: mut};
                    ret ures_ok(mk_ivec(cx.tcx, mt));
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
                let mut;
                alt unify_mut(expected_mt.mut, actual_mt.mut) {
                  none. { ret ures_err(terr_vec_mutability); }
                  some(m) { mut = m; }
                }
                let result = unify_step(cx, expected_mt.ty, actual_mt.ty);
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
          ty::ty_port(expected_sub) {
            alt struct(cx.tcx, actual) {
              ty::ty_port(actual_sub) {
                let result = unify_step(cx, expected_sub, actual_sub);
                alt result {
                  ures_ok(result_sub) {
                    ret ures_ok(mk_port(cx.tcx, result_sub));
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
                let result = unify_step(cx, ex_inner, act_inner);
                alt result {
                  ures_ok(res_inner) {
                    let i = 0u;
                    let res_tps = ~[];
                    for ex_tp: t  in ex_tps {
                        let result = unify_step(cx, ex_tp, act_tps.(i));
                        alt result {
                          ures_ok(rty) { res_tps += ~[rty]; }
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
          ty::ty_chan(expected_sub) {
            alt struct(cx.tcx, actual) {
              ty::ty_chan(actual_sub) {
                let result = unify_step(cx, expected_sub, actual_sub);
                alt result {
                  ures_ok(result_sub) {
                    ret ures_ok(mk_chan(cx.tcx, result_sub));
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
                let expected_len = ivec::len[field](expected_fields);
                let actual_len = ivec::len[field](actual_fields);
                if expected_len != actual_len {
                    let err = terr_record_size(expected_len, actual_len);
                    ret ures_err(err);
                }
                // TODO: implement an iterator that can iterate over
                // two arrays simultaneously.

                let result_fields: field[] = ~[];
                let i = 0u;
                while i < expected_len {
                    let expected_field = expected_fields.(i);
                    let actual_field = actual_fields.(i);
                    let mut;
                    alt unify_mut(expected_field.mt.mut, actual_field.mt.mut)
                        {
                      none. { ret ures_err(terr_record_mutability); }
                      some(m) { mut = m; }
                    }
                    if !str::eq(expected_field.ident, actual_field.ident) {
                        let err =
                            terr_record_fields(expected_field.ident,
                                               actual_field.ident);
                        ret ures_err(err);
                    }
                    let result =
                        unify_step(cx, expected_field.mt.ty,
                                   actual_field.mt.ty);
                    alt result {
                      ures_ok(rty) {
                        let mt = {ty: rty, mut: mut};
                        result_fields += ~[{mt: mt with expected_field}];
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
          ty::ty_fn(ep, expected_inputs, expected_output, expected_cf,
                    expected_constrs) {
            alt struct(cx.tcx, actual) {
              ty::ty_fn(ap, actual_inputs, actual_output, actual_cf,
                        actual_constrs) {
                ret unify_fn(cx, ep, ap, expected, actual, expected_inputs,
                             expected_output, actual_inputs, actual_output,
                             expected_cf, actual_cf, expected_constrs,
                             actual_constrs);
              }
              _ { ret ures_err(terr_mismatch); }
            }
          }
          ty::ty_native_fn(e_abi, expected_inputs, expected_output) {
            alt struct(cx.tcx, actual) {
              ty::ty_native_fn(a_abi, actual_inputs, actual_output) {
                ret unify_native_fn(cx, e_abi, a_abi, expected, actual,
                                    expected_inputs, expected_output,
                                    actual_inputs, actual_output);
              }
              _ { ret ures_err(terr_mismatch); }
            }
          }
          ty::ty_obj(expected_meths) {
            alt struct(cx.tcx, actual) {
              ty::ty_obj(actual_meths) {
                ret unify_obj(cx, expected, actual, expected_meths,
                              actual_meths);
              }
              _ { ret ures_err(terr_mismatch); }
            }
          }
          ty::ty_constr(expected_t, expected_constrs) {

            // unify the base types...
            alt struct(cx.tcx, actual) {
              ty::ty_constr(actual_t, actual_constrs) {
                let rslt = unify_step(cx, expected_t, actual_t);
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
                ret unify_step(cx, expected_t, actual);
              }
            }
          }
        }
    }
    fn unify(expected: &t, actual: &t, vb: &@var_bindings, tcx: &ty_ctxt) ->
       result {
        let cx = @{vb: vb, tcx: tcx};
        ret unify_step(cx, expected, actual);
    }
    fn dump_var_bindings(tcx: ty_ctxt, vb: @var_bindings) {
        let i = 0u;
        while i < ivec::len[ufindivec::node](vb.sets.nodes) {
            let sets = "";
            let j = 0u;
            while j < ivec::len[option::t[uint]](vb.sets.nodes) {
                if ufindivec::find(vb.sets, j) == i {
                    sets += #fmt(" %u", j);
                }
                j += 1u;
            }
            let typespec;
            alt smallintmap::find[t](vb.types, i) {
              none. { typespec = ""; }
              some(typ) { typespec = " =" + ty_to_str(tcx, typ); }
            }
            log_err #fmt("set %u:%s%s", i, typespec, sets);
            i += 1u;
        }
    }

    // Fixups and substitutions
    fn fixup_vars(tcx: ty_ctxt, vb: @var_bindings, typ: t) -> fixup_result {
        fn subst_vars(tcx: ty_ctxt, vb: @var_bindings,
                      unresolved: @mutable option::t[int], vid: int) -> t {
            if vid as uint >= ufindivec::set_count(vb.sets) {
                *unresolved = some(vid);
                ret ty::mk_var(tcx, vid);
            }
            let root_id = ufindivec::find(vb.sets, vid as uint);
            alt smallintmap::find[t](vb.types, root_id) {
              none. { *unresolved = some(vid); ret ty::mk_var(tcx, vid); }
              some(rt) {
                ret fold_ty(tcx,
                            fm_var(bind subst_vars(tcx, vb, unresolved, _)),
                            rt);
              }
            }
        }
        let unresolved = @mutable none[int];
        let rty =
            fold_ty(tcx, fm_var(bind subst_vars(tcx, vb, unresolved, _)),
                    typ);
        let ur = *unresolved;
        alt ur {
          none. { ret fix_ok(rty); }
          some(var_id) { ret fix_err(var_id); }
        }
    }
    fn resolve_type_var(tcx: &ty_ctxt, vb: &@var_bindings, vid: int) ->
       fixup_result {
        if vid as uint >= ufindivec::set_count(vb.sets) { ret fix_err(vid); }
        let root_id = ufindivec::find(vb.sets, vid as uint);
        alt smallintmap::find[t](vb.types, root_id) {
          none. { ret fix_err(vid); }
          some(rt) { ret fixup_vars(tcx, vb, rt); }
        }
    }
}

fn type_err_to_str(err: &ty::type_err) -> str {
    alt err {
      terr_mismatch. { ret "types differ"; }
      terr_controlflow_mismatch. {
        ret "returning function used where non-returning function" +
                " was expected";
      }
      terr_box_mutability. { ret "boxed values differ in mutability"; }
      terr_vec_mutability. { ret "vectors differ in mutability"; }
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
        ret "expected argument mode " + mode_str_1(e_mode) + " but found " +
                mode_str_1(a_mode);
      }
      terr_constr_len(e_len, a_len) {
        ret "Expected a type with " + uint::str(e_len) +
                " constraints, \
              but found one with " +
                uint::str(a_len) + " constraints";
      }
      terr_constr_mismatch(e_constr, a_constr) {
        ret "Expected a type with constraint " + ty_constr_to_str(e_constr) +
                " but found one with constraint " +
                ty_constr_to_str(a_constr);
      }
    }
}


// Converts type parameters in a type to type variables and returns the
// resulting type along with a list of type variable IDs.
fn bind_params_in_type(sp: &span, cx: &ctxt, next_ty_var: fn() -> int ,
                       typ: t, ty_param_count: uint) -> {ids: int[], ty: t} {
    let param_var_ids: @mutable int[] = @mutable ~[];
    let i = 0u;
    while i < ty_param_count { *param_var_ids += ~[next_ty_var()]; i += 1u; }
    fn binder(sp: span, cx: ctxt, param_var_ids: @mutable int[],
              next_ty_var: fn() -> int , index: uint, kind: ast::kind) -> t {
        if index < ivec::len(*param_var_ids) {
            ret mk_var(cx, param_var_ids.(index));
        } else {
            cx.sess.span_fatal(sp, "Unbound type parameter in callee's type");
        }
    }
    let new_typ =
        fold_ty(cx,
                fm_param(bind binder(sp, cx, param_var_ids, next_ty_var,
                                     _, _)),
                typ);
    ret {ids: *param_var_ids, ty: new_typ};
}


// Replaces type parameters in the given type using the given list of
// substitions.
fn substitute_type_params(cx: &ctxt, substs: &ty::t[], typ: t) -> t {
    if !type_contains_params(cx, typ) { ret typ; }
    fn substituter(cx: ctxt, substs: @ty::t[], idx: uint,
                   kind: ast::kind) -> t {
        // FIXME: bounds check can fail
        ret substs.(idx);
    }
    ret fold_ty(cx, fm_param(bind substituter(cx, @substs, _, _)), typ);
}

fn def_has_ty_params(def: &ast::def) -> bool {
    alt def {
      ast::def_fn(_, _) { ret true; }
      ast::def_obj_field(_) { ret false; }
      ast::def_mod(_) { ret false; }
      ast::def_const(_) { ret false; }
      ast::def_arg(_) { ret false; }
      ast::def_local(_) { ret false; }
      ast::def_variant(_, _) { ret true; }
      ast::def_ty(_) { ret false; }
      ast::def_ty_arg(_,_) { ret false; }
      ast::def_binding(_) { ret false; }
      ast::def_use(_) { ret false; }
      ast::def_native_ty(_) { ret false; }
      ast::def_native_fn(_) { ret true; }
    }
}


// Tag information
type variant_info = {args: ty::t[], ctor_ty: ty::t, id: ast::def_id};

fn tag_variants(cx: &ctxt, id: &ast::def_id) -> variant_info[] {
    if ast::local_crate != id.crate { ret csearch::get_tag_variants(cx, id); }
    let item =
        alt cx.items.find(id.node) {
          some(i) { i }
          none. { cx.sess.bug("expected to find cached node_item") }
        };
    alt item {
      ast_map::node_item(item) {
        alt item.node {
          ast::item_tag(variants, _) {
            let result: variant_info[] = ~[];
            for variant: ast::variant  in variants {
                let ctor_ty = node_id_to_monotype(cx, variant.node.id);
                let arg_tys: t[] = ~[];
                if std::ivec::len(variant.node.args) > 0u {
                    for a: arg  in ty_fn_args(cx, ctor_ty) {
                        arg_tys += ~[a.ty];
                    }
                }
                let did = variant.node.id;
                result +=
                    ~[{args: arg_tys,
                       ctor_ty: ctor_ty,
                       id: ast::local_def(did)}];
            }
            ret result;
          }
        }
      }
    }
}


// Returns information about the tag variant with the given ID:
fn tag_variant_with_id(cx: &ctxt, tag_id: &ast::def_id,
                       variant_id: &ast::def_id) -> variant_info {
    let variants = tag_variants(cx, tag_id);
    let i = 0u;
    while i < ivec::len[variant_info](variants) {
        let variant = variants.(i);
        if def_eq(variant.id, variant_id) { ret variant; }
        i += 1u;
    }
    cx.sess.bug("tag_variant_with_id(): no variant exists with that ID");
}


// If the given item is in an external crate, looks up its type and adds it to
// the type cache. Returns the type parameters and type.
fn lookup_item_type(cx: ctxt, did: ast::def_id) -> ty_param_count_and_ty {
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

fn ret_ty_of_fn_ty(cx: ctxt, a_ty: t) -> t {
    alt ty::struct(cx, a_ty) {
      ty::ty_fn(_, _, ret_ty, _, _) { ret ret_ty; }
      ty::ty_native_fn(_, _, ret_ty) { ret ret_ty; }
      _ {
        cx.sess.bug("ret_ty_of_fn_ty() called on non-function type: " +
                        ty_to_str(cx, a_ty));
      }
    }
}

fn ret_ty_of_fn(cx: ctxt, id: ast::node_id) -> t {
    ret ret_ty_of_fn_ty(cx, node_id_to_type(cx, id));
}

fn is_binopable(cx: &ctxt, ty: t, op: ast::binop) -> bool {

    const tycat_other: int = 0;
    const tycat_bool: int = 1;
    const tycat_int: int = 2;
    const tycat_float: int = 3;
    const tycat_str: int = 4;
    const tycat_vec: int = 5;
    const tycat_struct: int = 6;

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

    fn tycat(cx: &ctxt, ty: t) -> int {
        alt struct(cx, ty) {
          ty_bool. { tycat_bool }
          ty_int. { tycat_int }
          ty_uint. { tycat_int }
          ty_machine(ast::ty_i8.) { tycat_int }
          ty_machine(ast::ty_i16.) { tycat_int }
          ty_machine(ast::ty_i32.) { tycat_int }
          ty_machine(ast::ty_i64.) { tycat_int }
          ty_machine(ast::ty_u8.) { tycat_int }
          ty_machine(ast::ty_u16.) { tycat_int }
          ty_machine(ast::ty_u32.) { tycat_int }
          ty_machine(ast::ty_u64.) { tycat_int }
          ty_float. { tycat_float }
          ty_machine(ast::ty_f32.) { tycat_float }
          ty_machine(ast::ty_f64.) { tycat_float }
          ty_char. { tycat_int }
          ty_ptr(_) { tycat_int }
          ty_str. { tycat_str }
          ty_istr. { tycat_str }
          ty_vec(_) { tycat_vec }
          ty_ivec(_) { tycat_vec }
          ty_rec(_) { tycat_struct }
          ty_tag(_, _) { tycat_struct }
          _ { tycat_other }
        }
    }

    const t: bool = true;
    const f: bool = false;

    /*.          add,     shift,   bit
      .             sub,     rel,     logic
      .                mult,    eq,         */
    let  /*other*/
         /*bool*/
         /*int*/
         /*float*/
         /*str*/
         /*vec*/
        tbl =
        [[f, f, f, f, t, t, f, f], [f, f, f, f, t, t, t, t],
         [t, t, t, t, t, t, t, f], [t, t, t, f, t, t, f, f],
         [t, f, f, f, t, t, f, f], [t, f, f, f, t, t, f, f],
         [f, f, f, f, t, t, f, f]]; /*struct*/

    ret tbl.(tycat(cx, ty)).(opcat(op));
}

fn ast_constr_to_constr[T](tcx: ty::ctxt, c: &@ast::constr_general[T]) ->
   @ty::constr_general[T] {
    alt tcx.def_map.find(c.node.id) {
      some(ast::def_fn(pred_id, ast::pure_fn.)) {
        ret @respan(c.span,
                    {path: c.node.path, args: c.node.args, id: pred_id});
      }
      _ {
        tcx.sess.span_fatal
            (c.span, "Predicate " + path_to_str(c.node.path) +
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
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
