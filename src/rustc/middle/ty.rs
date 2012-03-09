import std::{ufind, map, smallintmap};
import result::result;
import std::map::hashmap;
import driver::session;
import session::session;
import syntax::ast;
import syntax::ast::*;
import syntax::ast_util;
import syntax::codemap::span;
import metadata::csearch;
import util::common::*;
import util::ppaux::region_to_str;
import util::ppaux::ty_to_str;
import util::ppaux::ty_i_to_str;
import util::ppaux::ty_constr_to_str;
import syntax::print::pprust::*;

export node_id_to_type;
export node_id_to_type_params;
export arg;
export args_eq;
export ast_constr_to_constr;
export block_ty;
export class_contents_ty;
export class_item_ty;
export class_items_as_fields;
export constr;
export constr_general;
export constr_table;
export ctxt;
export def_has_ty_params;
export expr_has_ty_params;
export expr_ty;
export expr_ty_params_and_ty;
export expr_is_lval;
export fold, fold_rptr;
export field;
export field_idx;
export get_field;
export get_fields;
export get_element_type;
export is_binopable;
export lookup_class_item_tys;
export lookup_item_type;
export method;
export method_idx;
export mk_class;
export mk_ctxt;
export mk_with_id, type_def_id;
export mt;
export node_type_table;
export pat_ty;
export sequence_element_type;
export sort_methods;
export stmt_node_id;
export sty;
export substitute_type_params, substitute_type_params_i;
export t;
export new_ty_hash;
export enum_variants, substd_enum_variants;
export iface_methods, store_iface_methods, impl_iface;
export enum_variant_with_id;
export ty_ops;
export ty_param_bounds_and_ty, ty_param_bounds_and_ty_i;
export ty_bool, mk_bool, type_is_bool, sty_is_bool;
export ty_bot, mk_bot, type_is_bot;
export ty_box, mk_box, mk_imm_box;
export type_is_box, type_is_boxed;
export ty_constr, mk_constr;
export ty_opaque_closure_ptr, mk_opaque_closure_ptr;
export ty_opaque_box, mk_opaque_box;
export ty_constr_arg;
export ty_float, mk_float, mk_mach_float;
export type_is_fp;
export ty_fn, fn_ty, mk_fn;
export ty_fn_proto, ty_fn_ret, ty_fn_ret_style;
export ty_int, mk_int, mk_mach_int, mk_char;
export ty_str, mk_str, type_is_str;
export ty_vec, mk_vec, type_is_vec;
export ty_nil, mk_nil, type_is_nil, sty_is_nil;
export ty_iface, mk_iface;
export ty_res, mk_res;
export ty_param, mk_param;
export ty_ptr, mk_ptr, mk_mut_ptr, mk_nil_ptr, type_is_unsafe_ptr;
export ty_rptr, mk_rptr;
export ty_rec, mk_rec;
export ty_enum, mk_enum, type_is_enum;
export ty_tup, mk_tup;
export ty_type, mk_type;
export ty_uint, mk_uint, mk_mach_uint;
export ty_uniq, mk_uniq, mk_imm_uniq, type_is_unique_box;
export ty_var_i, mk_var;
export ty_self, mk_self;
export region, re_named, re_caller, re_block, re_inferred;
export get, type_has_rptrs, type_id;
export type_has_self, ty_i_has_self;
export type_has_params, ty_i_has_params;
export ty_to_def_id;
export ty_fn_args;
export type_constr;
export kind, kind_sendable, kind_copyable, kind_noncopyable;
export kind_can_be_copied, kind_can_be_sent, proto_kind, kind_lteq, type_kind;
export type_err, resolve_err;
export type_err_to_str, resolve_err_to_str;
export type_needs_drop;
export type_allows_implicit_copy;
export type_is_integral, sty_is_integral;
export type_is_numeric, sty_is_numeric;
export type_is_pod;
export type_is_scalar, sty_is_scalar;
export type_is_immediate;
export type_is_sequence;
export type_is_signed;
export type_is_structural;
export type_is_copyable;
export type_is_tup_like;
export type_is_unique;
export type_is_c_like_enum, sty_is_c_like_enum;
export type_structurally_contains;
export type_structurally_contains_uniques;
export type_autoderef;
export type_param;
export canon_mode;
export resolved_mode;
export arg_mode;
export unify_mode;
export set_default_mode;
export unify;
export variant_info;
export walk_ty, walk_ty_i, maybe_walk_ty;
export closure_kind;
export ck_block;
export ck_box;
export ck_uniq;
export param_bound, param_bounds, bound_copy, bound_send, bound_iface;
export param_bounds_to_kind;
export default_arg_mode_for_ty;
export item_path;
export item_path_str;

export var_bindings;
export t_i;
export sty_i, ty_var_i;
export arg_i;
export field_i;
export class_contents_ty_i;
export class_item_ty_i;
export param_bounds_i;
export param_bound_i;
export method_i;
export mt_i;
export sty_i;
export fn_ty_i;

export sty_base;
export arg_base;
export field_base;
export class_contents_ty_base;
export class_item_ty_base;
export param_bounds_base;
export param_bound_base;
export method_base;
export mt_base;
export fn_ty_base;

export ty_to_ty_i, ty_to_ty_i_subst;

// Data types
//
// To accommodate inference, the types are defined in two "layers".  The base
// layer (e.g., `sty_base`) is parameterized by a type variable T, which
// defines a reference to a type.  Typically, you will the type alias `sty`
// which is defined as `sty_base<t>`, where `t` is an entry in the type table.
// During inference, however, we use the infer family of types (e.g.,
// `sty_i`), defined with `T` equal to `t_i`.  `t_i` allows for
// both concrete types but also type variables.

// Note: after typeck, you should use resolved_mode() to convert this mode
// into an rmode, which will take into account the results of mode inference.
type arg_base<T> = {mode: ast::mode, ty: T};

type field_base<T> = {ident: ast::ident, mt: mt_base<T>};

type param_bounds_base<T> = @[param_bound_base<T>];

enum param_bound_base<T> {
    bound_copy,
    bound_send,
    bound_iface(T),
}

type method_base<T> = {ident: ast::ident,
                       tps: @[param_bounds_base<T>],
                       fty: fn_ty_base<T>,
                       purity: ast::purity};

type constr_table = hashmap<ast::node_id, [constr]>;

type mt_base<T> = {ty: T, mutbl: ast::mutability};

type class_item_ty_base<T> = {
  ident: ident,
  id: node_id,
  contents: class_contents_ty
};

enum class_contents_ty_base<T> {
  var_ty(T),   // FIXME: need mutability, too
  method_ty(fn_decl)
}

type fn_ty_base<T> = {proto: ast::proto,
                      inputs: [arg_base<T>],
                      output: T,
                      ret_style: ret_style,
                      constraints: [@constr]};

enum region {
    re_named(def_id),
    re_caller(def_id),
    re_self(def_id),
    re_block(node_id),
    re_inferred         /* currently unresolved (for typedefs) */
}

// NB: If you change this, you'll probably want to change the corresponding
// AST structure in front/ast::rs as well.
enum sty_base<T> {
    ty_nil,
    ty_bot,
    ty_bool,
    ty_int(ast::int_ty),
    ty_uint(ast::uint_ty),
    ty_float(ast::float_ty),
    ty_str,
    ty_enum(def_id, [T]),
    ty_box(mt_base<T>),
    ty_uniq(mt_base<T>),
    ty_vec(mt_base<T>),
    ty_ptr(mt_base<T>),
    ty_rptr(region, mt_base<T>),
    ty_rec([field_base<T>]),
    ty_fn(fn_ty_base<T>),
    ty_iface(def_id, [T]),
    ty_class(def_id, [T]),
    ty_res(def_id, T, [T]),
    ty_tup([T]),

    ty_param(uint, def_id), // type parameter
    ty_self([T]), // interface method self type

    ty_type, // type_desc*
    ty_opaque_box, // used by monomorphizer to represend any @ box
    ty_constr(T, [@type_constr]),
    ty_opaque_closure_ptr(closure_kind), // ptr to env for fn, fn@, fn~
}

enum t_i_box {
    ty_var_i(int), // var index relative to in-scope set of var_bindings
    sty_i(sty_i)
}
type t_i = @t_i_box;
type arg_i = arg_base<t_i>;
type field_i = field_base<t_i>;
type param_bounds_i = param_bounds_base<t_i>;
type param_bound_i = param_bound_base<t_i>;
type method_i = method_base<t_i>;
type mt_i = mt_base<t_i>;
type fn_ty_i = fn_ty_base<t_i>;
type sty_i = sty_base<t_i>;
type class_item_ty_i = class_item_ty_base<t_i>;
type class_contents_ty_i = class_contents_ty_base<t_i>;

type arg = arg_base<t>;
type field = field_base<t>;
type param_bounds = param_bounds_base<t>;
type param_bound = param_bound_base<t>;
type method = method_base<t>;
type mt = mt_base<t>;
type fn_ty = fn_ty_base<t>;
type sty = sty_base<t>;
type class_item_ty = class_item_ty_base<t>;
type class_contents_ty = class_contents_ty_base<t>;

// In the middle end, constraints have a def_id attached, referring
// to the definition of the operator in the constraint.
type constr_general<ARG> = spanned<constr_general_<ARG, def_id>>;
type type_constr = constr_general<@path>;
type constr = constr_general<uint>;

// Contains information needed to resolve types and (in the future) look up
// the types of AST nodes.
type creader_cache = hashmap<{cnum: int, pos: uint, len: uint}, t>;

type intern_key = {struct: sty, o_def_id: option<ast::def_id>};

type ctxt =
    @{interner: hashmap<intern_key, t_box>,
      mutable next_id: uint,
      sess: session::session,
      def_map: resolve::def_map,
      region_map: @middle::region::region_map,
      node_types: node_type_table,
      node_type_substs: hashmap<node_id, [t]>,
      items: ast_map::map,
      freevars: freevars::freevar_map,
      tcache: type_cache,
      rcache: creader_cache,
      short_names_cache: hashmap<t, @str>,
      needs_drop_cache: hashmap<t, bool>,
      kind_cache: hashmap<t, kind>,
      enum_var_cache: hashmap<def_id, @[variant_info]>,
      iface_method_cache: hashmap<def_id, @[method]>,
      ty_param_bounds: hashmap<ast::node_id, param_bounds>,
      inferred_modes: hashmap<ast::node_id, ast::mode>};

type var_bindings =
    {tcx: ctxt,
     sets: ufind::ufind,
     node_types: hashmap<node_id, t_i>,
     node_type_substs: hashmap<node_id, [t_i]>,
     var_types: smallintmap::smallintmap<sty_i>};

fn var_bindings(tcx: ctxt) -> @var_bindings {
    ret @{tcx: tcx,
          sets: ufind::make(),
          node_types: map::int_hash(),
          node_type_substs: map::int_hash(),
          var_types: smallintmap::mk()};
}

type t_flags = {has_params: bool,
                has_self: bool,
                has_rptrs: bool};

type t_box = @{struct: sty,
               id: uint,
               t_flags: t_flags,
               o_def_id: option<ast::def_id>};

// To reduce refcounting cost, we're representing types as unsafe pointers
// throughout the compiler. These are simply casted t_box values. Use ty::get
// to cast them back to a box. (Without the cast, compiler performance suffers
// ~15%.) This does mean that a t value relies on the ctxt to keep its box
// alive, and using ty::get is unsafe when the ctxt is no longer alive.
enum t_opaque {}
type t = *t_opaque;

pure fn get(t: t) -> t_box unsafe {
    let t2 = unsafe::reinterpret_cast::<t, t_box>(t);
    let t3 = t2;
    unsafe::forget(t2);
    t3
}

fn ty_i_has_params(t: t_i) -> bool { flags_for_t_i(t).has_params }
fn type_has_params(t: t) -> bool { get(t).t_flags.has_params }
fn ty_i_has_self(t: t_i) -> bool { flags_for_t_i(t).has_self }
fn type_has_self(t: t) -> bool { get(t).t_flags.has_self }
fn type_has_rptrs(t: t) -> bool { get(t).t_flags.has_rptrs }
fn type_def_id(t: t) -> option<ast::def_id> { get(t).o_def_id }
fn type_id(t: t) -> uint { get(t).id }

enum closure_kind {
    ck_block,
    ck_box,
    ck_uniq,
}

enum resolve_err {
    rerr_unresolved_var(int),
    rerr_cyclic_var(int)
}

// Data structures used in type unification
enum type_err {
    terr_mismatch,
    terr_ret_style_mismatch(ast::ret_style, ast::ret_style),
    terr_box_mutability,
    terr_ptr_mutability,
    terr_ref_mutability,
    terr_vec_mutability,
    terr_tuple_size(uint, uint),
    terr_ty_param_size(uint, uint),
    terr_record_size(uint, uint),
    terr_record_mutability,
    terr_record_fields(ast::ident, ast::ident),
    terr_arg_count,
    terr_mode_mismatch(mode, mode),
    terr_constr_len(uint, uint),
    terr_constr_mismatch(@type_constr, @type_constr),
    terr_regions_differ(bool /* variance */, region, region),
    terr_cyclic_type
}

fn param_bounds_to_kind(bounds: param_bounds) -> kind {
    let kind = kind_noncopyable;
    for bound in *bounds {
        alt bound {
          bound_copy {
            if kind != kind_sendable { kind = kind_copyable; }
          }
          bound_send { kind = kind_sendable; }
          _ {}
        }
    }
    kind
}

type ty_param_bounds_and_ty = {bounds: @[param_bounds], ty: t};

type ty_param_bounds_and_ty_i = {bounds: @[param_bounds_i], ty: t_i};

type type_cache = hashmap<ast::def_id, ty_param_bounds_and_ty>;

type node_type_table = @smallintmap::smallintmap<t>;

fn mk_rcache() -> creader_cache {
    type val = {cnum: int, pos: uint, len: uint};
    fn hash_cache_entry(k: val) -> uint {
        ret (k.cnum as uint) + k.pos + k.len;
    }
    fn eq_cache_entries(a: val, b: val) -> bool {
        ret a.cnum == b.cnum && a.pos == b.pos && a.len == b.len;
    }
    ret map::hashmap(hash_cache_entry, eq_cache_entries);
}

fn new_ty_hash<V: copy>() -> map::hashmap<t, V> {
    map::hashmap({|&&t: t| type_id(t)},
                    {|&&a: t, &&b: t| type_id(a) == type_id(b)})
}

fn mk_ctxt(s: session::session, dm: resolve::def_map, amap: ast_map::map,
           freevars: freevars::freevar_map,
           region_map: @middle::region::region_map) -> ctxt {
    let interner = map::hashmap({|&&k: intern_key|
        hash_type_structure(k.struct) +
            option::maybe(k.o_def_id, 0u, ast_util::hash_def_id)
    }, {|&&a, &&b| a == b});
    @{interner: interner,
      mutable next_id: 0u,
      sess: s,
      def_map: dm,
      region_map: region_map,
      node_types: @smallintmap::mk(),
      node_type_substs: map::int_hash(),
      items: amap,
      freevars: freevars,
      tcache: new_def_hash(),
      rcache: mk_rcache(),
      short_names_cache: new_ty_hash(),
      needs_drop_cache: new_ty_hash(),
      kind_cache: new_ty_hash(),
      enum_var_cache: new_def_hash(),
      iface_method_cache: new_def_hash(),
      ty_param_bounds: map::int_hash(),
      inferred_modes: map::int_hash()}
}


// Type constructors
fn mk_t(cx: ctxt, st: sty) -> t { mk_t_with_id(cx, st, none) }

fn flags_for_sty_base<T>(sty: sty_base<T>,
                         t_flags: fn(T) -> t_flags) -> t_flags {
    let has_params = false;
    let has_self = false;
    let has_rptrs = false;

    fn derive_flags<T>(&has_params: bool,
                       &has_self: bool,
                       &has_rptrs: bool,
                       t: T,
                       t_flags: fn(T) -> t_flags) {
        let f = t_flags(t);
        has_params = has_params || f.has_params;
        has_rptrs = has_rptrs || f.has_rptrs;
        has_self = has_self || f.has_self;
    }

    alt sty {
      ty_nil | ty_bot | ty_bool | ty_int(_) | ty_float(_) | ty_uint(_) |
      ty_str | ty_type | ty_opaque_closure_ptr(_) | ty_opaque_box {
      }
      ty_self(_) {
        has_self = true;
      }
      ty_param(_, _) {
        has_params = true;
      }
      ty_enum(_, tys) | ty_iface(_, tys) | ty_class(_, tys) {
        for tt in tys {
            derive_flags(has_params, has_self, has_rptrs, tt, t_flags);
        }
      }
      ty_box(m) | ty_uniq(m) | ty_vec(m) | ty_ptr(m) {
        derive_flags(has_params, has_self, has_rptrs, m.ty, t_flags);
      }
      ty_rptr(_, m) {
        has_rptrs = true;
        derive_flags(has_params, has_self, has_rptrs, m.ty, t_flags);
      }
      ty_rec(flds) {
        for f in flds {
          derive_flags(has_params, has_self, has_rptrs, f.mt.ty, t_flags);
        }
      }
      ty_tup(ts) {
        for tt in ts {
            derive_flags(has_params, has_self, has_rptrs, tt, t_flags);
        }
      }
      ty_fn(f) {
        for a in f.inputs {
            derive_flags(has_params, has_self, has_rptrs, a.ty, t_flags);
        }
        derive_flags(has_params, has_self, has_rptrs, f.output, t_flags);
      }
      ty_res(_, tt, tps) {
        derive_flags(has_params, has_self, has_rptrs, tt, t_flags);
        for tt in tps {
            derive_flags(has_params, has_self, has_rptrs, tt, t_flags);
        }
      }
      ty_constr(tt, _) {
        derive_flags(has_params, has_self, has_rptrs, tt, t_flags);
      }
    }

    ret {has_params: has_params, has_self: has_self, has_rptrs: has_rptrs};
}

fn flags_for_sty(st: sty) -> t_flags {
    flags_for_sty_base(st) {|t| get(t).t_flags }
}

fn flags_for_t_i(&&t: t_i) -> t_flags {
    alt *t {
      ty_var_i(_) { {has_params: false, has_self: false, has_rptrs: false} }
      sty_i(st) { flags_for_sty_base(st, flags_for_t_i(_)) }
    }
}

// Interns a type/name combination, stores the resulting box in cx.interner,
// and returns the box as cast to an unsafe ptr (see comments for t above).
fn mk_t_with_id(cx: ctxt, st: sty, o_def_id: option<ast::def_id>) -> t {
    let key = {struct: st, o_def_id: o_def_id};
    alt cx.interner.find(key) {
      some(t) { unsafe { ret unsafe::reinterpret_cast(t); } }
      _ {}
    }
    let t_flags = flags_for_sty(st);
    let t = @{struct: st,
              id: cx.next_id,
              t_flags: t_flags,
              o_def_id: o_def_id};
    cx.interner.insert(key, t);
    cx.next_id += 1u;
    unsafe { unsafe::reinterpret_cast(t) }
}

fn mk_nil<T:copy,C:ty_ops<T>>(cx: C) -> T { cx.mk(ty_nil) }

fn mk_bot<T:copy,C:ty_ops<T>>(cx: C) -> T { cx.mk(ty_bot) }

fn mk_bool<T:copy,C:ty_ops<T>>(cx: C) -> T { cx.mk(ty_bool) }

fn mk_int<T:copy,C:ty_ops<T>>(cx: C) -> T { cx.mk(ty_int(ast::ty_i)) }

fn mk_float<T:copy,C:ty_ops<T>>(cx: C) -> T { cx.mk(ty_float(ast::ty_f)) }

fn mk_uint<T:copy,C:ty_ops<T>>(cx: C) -> T { cx.mk(ty_uint(ast::ty_u)) }

fn mk_mach_int<T:copy,C:ty_ops<T>>(cx: C, tm: ast::int_ty) -> T { cx.mk(ty_int(tm)) }

fn mk_mach_uint<T:copy,C:ty_ops<T>>(cx: C, tm: ast::uint_ty) -> T { cx.mk(ty_uint(tm)) }

fn mk_mach_float<T:copy,C:ty_ops<T>>(cx: C, tm: ast::float_ty) -> T { cx.mk(ty_float(tm)) }

fn mk_char<T:copy,C:ty_ops<T>>(cx: C) -> T { cx.mk(ty_int(ast::ty_char)) }

fn mk_str<T:copy,C:ty_ops<T>>(cx: C) -> T { cx.mk(ty_str) }

fn mk_enum<T:copy,C:ty_ops<T>>(cx: C, did: ast::def_id, tys: [T]) -> T {
    cx.mk(ty_enum(did, tys))
}

fn mk_box<T:copy,C:ty_ops<T>>(cx: C, tm: mt_base<T>) -> T { cx.mk(ty_box(tm)) }

fn mk_imm_box<T:copy,C:ty_ops<T>>(cx: C, ty: T) -> T { mk_box(cx, {ty: ty,
                                                              mutbl: ast::m_imm}) }

fn mk_uniq<T:copy,C:ty_ops<T>>(cx: C, tm: mt_base<T>) -> T { cx.mk(ty_uniq(tm)) }

fn mk_imm_uniq<T:copy,C:ty_ops<T>>(cx: C, ty: T) -> T { mk_uniq(cx, {ty: ty,
                                                                mutbl: ast::m_imm}) }

fn mk_ptr<T:copy,C:ty_ops<T>>(cx: C, tm: mt_base<T>) -> T {
    cx.mk(ty_ptr(tm))
}

fn mk_rptr<T:copy,C:ty_ops<T>>(cx: C, r: region, tm: mt_base<T>) -> T {
    cx.mk(ty_rptr(r, tm))
}

fn mk_mut_ptr<T:copy,C:ty_ops<T>>(cx: C, ty: T) -> T { mk_ptr(cx, {ty: ty,
                                                              mutbl: ast::m_mutbl}) }

fn mk_nil_ptr<T:copy,C:ty_ops<T>>(cx: C) -> T {
    mk_ptr(cx, {ty: mk_nil(cx), mutbl: ast::m_imm})
}

fn mk_vec<T:copy,C:ty_ops<T>>(cx: C, tm: mt_base<T>) -> T { cx.mk(ty_vec(tm)) }

fn mk_rec<T:copy,C:ty_ops<T>>(cx: C, fs: [field_base<T>]) -> T { cx.mk(ty_rec(fs)) }

fn mk_constr<T:copy,C:ty_ops<T>>(cx: C, t: T, cs: [@type_constr]) -> T {
    cx.mk(ty_constr(t, cs))
}

fn mk_tup<T:copy,C:ty_ops<T>>(cx: C, ts: [T]) -> T { cx.mk(ty_tup(ts)) }

fn mk_fn<T:copy,C:ty_ops<T>>(cx: C, fty: fn_ty_base<T>) -> T { cx.mk(ty_fn(fty)) }

fn mk_iface<T:copy,C:ty_ops<T>>(cx: C, did: ast::def_id, tys: [T]) -> T {
    cx.mk(ty_iface(did, tys))
}

fn mk_class<T:copy,C:ty_ops<T>>(cx: C, class_id: ast::def_id, tys: [T]) -> T {
    cx.mk(ty_class(class_id, tys))
}

fn mk_res<T:copy,C:ty_ops<T>>(cx: C, did: ast::def_id, inner: T, tps: [T]) -> T {
    cx.mk(ty_res(did, inner, tps))
}

fn mk_var(v: int) -> t_i { @ty_var_i(v) }

fn mk_self<T:copy,C:ty_ops<T>>(cx: C, tps: [T]) -> T { cx.mk(ty_self(tps)) }

fn mk_param<T:copy,C:ty_ops<T>>(cx: C, n: uint, k: def_id) -> T { cx.mk(ty_param(n, k)) }

fn mk_type<T:copy,C:ty_ops<T>>(cx: C) -> T { cx.mk(ty_type) }

fn mk_opaque_closure_ptr<T:copy,C:ty_ops<T>>(cx: C, ck: closure_kind) -> T {
    cx.mk(ty_opaque_closure_ptr(ck))
}

fn mk_opaque_box<T:copy,C:ty_ops<T>>(cx: C) -> T { cx.mk(ty_opaque_box) }

fn mk_with_id(cx: ctxt, base: t, def_id: ast::def_id) -> t {
    mk_t_with_id(cx, get(base).struct, some(def_id))
}

// Converts s to its machine type equivalent
pure fn mach_sty<T:copy>(
    cfg: @session::config, t: sty_base<T>) -> sty_base<T> {

    alt t {
      ty_int(ast::ty_i) { ty_int(cfg.int_type) }
      ty_uint(ast::ty_u) { ty_uint(cfg.uint_type) }
      ty_float(ast::ty_f) { ty_float(cfg.float_type) }
      _ { t }
    }
}

fn default_arg_mode_for_ty(ty: ty::t) -> ast::rmode {
    if ty::type_is_immediate(ty) { ast::by_val }
    else { ast::by_ref }
}

fn walk_ty(ty: t, f: fn(t)) {
    f(ty);
    walk_sty_base(get(ty).struct) {|t|
        walk_ty(t, f)
    }
}

fn maybe_walk_ty(ty: t, f: fn(t) -> bool) {
    if !f(ty) { ret; }
    walk_sty_base(get(ty).struct) {|t|
        maybe_walk_ty(t, f)
    }
}

fn walk_sty_base<T>(sty: sty_base<T>, walkf: fn(T)) {
    alt sty {
      ty_nil | ty_bot | ty_bool | ty_int(_) | ty_uint(_) | ty_float(_) |
      ty_str | ty_type | ty_opaque_box |
      ty_opaque_closure_ptr(_) | ty_param(_, _) {}
      ty_box(tm) | ty_vec(tm) | ty_ptr(tm) | ty_rptr(_, tm) {
        walkf(tm.ty);
      }
      ty_enum(_, subtys) | ty_iface(_, subtys) | ty_class(_, subtys)
      | ty_self(subtys) {
        for subty in subtys { walkf(subty); }
      }
      ty_rec(fields) {
        for fl in fields { walkf(fl.mt.ty); }
      }
      ty_tup(ts) { for tt in ts { walkf(tt); } }
      ty_fn(ft) {
        for a in ft.inputs { walkf(a.ty); }
        walkf(ft.output);
      }
      ty_res(_, sub, tps) {
        walkf(sub);
        for tp in tps { walkf(tp); }
      }
      ty_constr(sub, _) { walkf(sub); }
      ty_uniq(tm) { walkf(tm.ty); }
    }
}

fn walk_ty_i(cx: ctxt, &&ty: t_i, f: fn(t_i)) {
    alt *ty {
      ty_var_i(_) { }
      sty_i(sty) { walk_sty_base(sty) {|t| walk_ty_i(cx, t, f) } }
    }
}

// The base for all folding operations.  This is written in a "result monad"
// style so that errors can be propagated if necessary.  I chose not to
// "deforest" this version though.  If you wish to write a folding routine
// which cannot fail, then use `fold_sty_base()` below, which presents
// a friendlier interface.
fn fold_sty_base_err<T:copy,U:copy,E:copy>(
    sty: sty_base<T>,
    fold_t: fn(T) -> result<U,E>) -> result<sty_base<U>,E> {
    import result::{chain, ok, map};

    fn fold_mt<T:copy,U:copy,E:copy>(
        mt: mt_base<T>,
        fold_t: fn(T) -> result<U,E>) -> result<mt_base<U>,E> {

        chain(fold_t(mt.ty)) {|t|
            ok({ty: t, mutbl: mt.mutbl})
        }
    }

    fn fold_f<T:copy,U:copy,E:copy>(
        fld: field_base<T>,
        fold_t: fn(T) -> result<U,E>) -> result<field_base<U>,E> {

        chain(fold_mt(fld.mt, fold_t)) {|fld_mt|
            ok({ident: fld.ident, mt: fld_mt})
        }
    }

    fn fold_arg<T:copy,U:copy,E:copy>(
        arg: arg_base<T>,
        fold_t: fn(T) -> result<U,E>) -> result<arg_base<U>,E> {

        chain(fold_t(arg.ty)) {|arg_ty|
            ok({mode: arg.mode, ty: arg_ty})
        }
    }

    alt sty {
      ty_nil { ok(ty_nil) }
      ty_bot { ok(ty_bot) }
      ty_bool { ok(ty_bool) }
      ty_int(i) { ok(ty_int(i)) }
      ty_uint(i) { ok(ty_uint(i)) }
      ty_float(i) { ok(ty_float(i)) }
      ty_str { ok(ty_str) }
      ty_type { ok(ty_type) }
      ty_opaque_closure_ptr(c) { ok(ty_opaque_closure_ptr(c)) }
      ty_opaque_box { ok(ty_opaque_box) }
      ty_box(mt) {
        chain(fold_mt(mt, fold_t)) {|mt|
            ok(ty_box(mt))
        }
      }
      ty_uniq(mt) {
        chain(fold_mt(mt, fold_t)) {|mt|
            ok(ty_uniq(mt))
        }
      }
      ty_vec(mt) {
        chain(fold_mt(mt, fold_t)) {|mt|
            ok(ty_vec(mt))
        }
      }
      ty_ptr(mt) {
        chain(fold_mt(mt, fold_t)) {|mt|
            ok(ty_ptr(mt))
        }
      }
      ty_enum(tid, subtys) {
        map(subtys, fold_t) {|subtys|
            ok(ty_enum(tid, subtys))
        }
      }
      ty_iface(tid, subtys) {
        map(subtys, fold_t) {|subtys|
            ok(ty_iface(tid, subtys))
        }
      }
      ty_class(did, subtys) {
        map(subtys, fold_t) {|subtys|
            ok(ty_class(did, subtys))
        }
      }
      ty_self(subtys) {
        map(subtys, fold_t) {|subtys|
            ok(ty_self(subtys))
        }
      }
      ty_rec(fields) {
        map(fields, {|f| fold_f(f, fold_t)}) {|fields|
            ok(ty_rec(fields))
        }
      }
      ty_tup(ts) {
        map(ts, fold_t) {|ts|
            ok(ty_tup(ts))
        }
      }
      ty_fn(f) {
        map(f.inputs, {|a| fold_arg(a, fold_t)}) {|inputs|
            chain(fold_t(f.output)) {|output|
                ok(ty_fn({proto: f.proto,
                          inputs: inputs,
                          output: output,
                          ret_style: f.ret_style,
                          constraints: f.constraints}))
            }
        }
      }
      ty_res(did, subty, tps) {
        chain(fold_t(subty)) {|subty|
            map(tps, fold_t) {|tps|
                ok(ty_res(did, subty, tps))
            }
        }
      }
      ty_param(id, did) {
        ok(ty_param(id, did))
      }
      ty_rptr(r, mt) {
        chain(fold_mt(mt, fold_t)) {|mt|
            ok(ty_rptr(r, mt))
        }
      }
      ty_constr(subty, cs) {
        chain(fold_t(subty)) {|subty|
            ok(ty_constr(subty, cs))
        }
      }
    }
}

iface ty_ops<T> {
    fn tcx() -> ctxt;
    fn no_rptrs(t: T) -> bool;
    fn if_struct<R:copy>(t: T, r0: R, op: fn(sty_base<T>) -> R) -> R;
    fn mk(s: sty_base<T>) -> T;
    fn swap_T(t: T, s: sty_base<T>) -> T;
    fn to_str(t: T) -> str;
    fn sess() -> session::session;
}

impl of ty_ops<t> for ctxt {
    fn tcx() -> ctxt { self }

    fn no_rptrs(&&t: t) -> bool { !get(t).t_flags.has_rptrs }

    fn if_struct<R:copy>(&&t: t, _r0: R, op: fn(sty_base<t>) -> R) -> R {
        op(get(t).struct)
    }

    fn mk(s: sty_base<t>) -> t {
        mk_t(self, s)
    }

    fn swap_T(&&t: t, s: sty_base<t>) -> t {
        mk_t_with_id(self, s, get(t).o_def_id)
    }

    fn to_str(&&t: t) -> str {
        ty_to_str(self, t)
    }

    fn sess() -> session::session {
        self.sess
    }
}

impl of ty_ops<t_i> for @var_bindings {
    fn tcx() -> ctxt { self.tcx }

    fn no_rptrs(&&_t: t_i) -> bool { false } // no cheap way to check

    fn if_struct<R:copy>(&&t: t_i, r0: R, op: fn(sty_base<t_i>) -> R) -> R {
        alt *t {
          ty_var_i(_) { r0 }
          sty_i(s) { op(s) }
        }
    }

    fn mk(s: sty_base<t_i>) -> t_i {
        @sty_i(s)
    }

    fn swap_T(&&_t: t_i, s: sty_base<t_i>) -> t_i {
        self.mk(s)
    }

    fn to_str(&&t: t_i) -> str {
        ty_i_to_str(self, t)
    }

    fn sess() -> session::session {
        self.tcx.sess
    }
}

// Friendlier version of `fold_sty_base_err` for the case where the fold
// cannot fail.
fn fold_sty_base<T:copy,U:copy>(
    sty: sty_base<T>,
    fold_t: fn(T) -> U) -> sty_base<U> {

    result::get(fold_sty_base_err(sty) {|t|
        result::ok::<U,type_err>(
            fold_t(t))
    })
}

fn fold<T:copy,C:ty_ops<T>>(
    cx: C, t0: T, f: fn(T) -> T) -> T {

    let t1 = cx.if_struct(t0, t0) {|sty|
        cx.mk(
            fold_sty_base(sty, {|t| fold(cx, t, f) }))
    };
    f(t1)
}

fn fold_rptr<T:copy,C:ty_ops<T>>(
    cx: C, t0: T, f: fn(region) -> region) -> T {

    if cx.no_rptrs(t0) { ret t0; }

    fold(cx, t0) {|t|
        cx.if_struct(t, t) {|sty|
            let sty1 = alt sty {
              ty_rptr(r, rt) { ty_rptr(f(r), rt) }
              _ { sty }
            };
            cx.mk(sty1)
        }
    }
}

// Type utilities

fn type_is_nil(ty: t) -> bool { get(ty).struct == ty_nil }

fn sty_is_nil<T>(sty: sty_base<T>) -> bool { sty == ty_nil }

fn type_is_bot(ty: t) -> bool { get(ty).struct == ty_bot }

fn sty_is_bool<T>(sty: sty_base<T>) -> bool { sty == ty_bool }

fn type_is_bool(ty: t) -> bool { get(ty).struct == ty_bool }

fn type_is_structural(ty: t) -> bool {
    alt get(ty).struct {
      ty_rec(_) | ty_class(_,_) | ty_tup(_) | ty_enum(_, _) | ty_fn(_) |
      ty_iface(_, _) | ty_res(_, _, _) { true }
      _ { false }
    }
}

fn type_is_copyable(cx: ctxt, ty: t) -> bool {
    ret kind_can_be_copied(type_kind(cx, ty));
}

fn type_is_sequence(ty: t) -> bool {
    alt get(ty).struct {
      ty_str { ret true; }
      ty_vec(_) { ret true; }
      _ { ret false; }
    }
}

fn type_is_str(ty: t) -> bool { get(ty).struct == ty_str }

fn sequence_element_type(cx: ctxt, ty: t) -> t {
    alt get(ty).struct {
      ty_str { ret mk_mach_uint(cx, ast::ty_u8); }
      ty_vec(mt) { ret mt.ty; }
      _ { cx.sess.bug("sequence_element_type called on non-sequence value"); }
    }
}

pure fn type_is_tup_like(ty: t) -> bool {
    alt get(ty).struct {
      ty_rec(_) | ty_tup(_) { true }
      _ { false }
    }
}

fn get_element_type(ty: t, i: uint) -> t {
    alt get(ty).struct {
      ty_rec(flds) { ret flds[i].mt.ty; }
      ty_tup(ts) { ret ts[i]; }
      _ { fail "get_element_type called on invalid type"; }
    }
}

pure fn sty_is_box<T>(sty: sty_base<T>) -> bool {
    alt sty {
      ty_box(_) { ret true; }
      _ { ret false; }
    }
}

pure fn type_is_box(ty: t) -> bool {
    sty_is_box(get(ty).struct)
}

pure fn type_is_boxed(ty: t) -> bool {
    alt get(ty).struct {
      ty_box(_) | ty_opaque_box { true }
      _ { false }
    }
}

pure fn type_is_unique_box(ty: t) -> bool {
    alt get(ty).struct {
      ty_uniq(_) { ret true; }
      _ { ret false; }
    }
}

pure fn type_is_unsafe_ptr(ty: t) -> bool {
    alt get(ty).struct {
      ty_ptr(_) { ret true; }
      _ { ret false; }
    }
}

pure fn type_is_vec(ty: t) -> bool {
    ret alt get(ty).struct {
          ty_vec(_) { true }
          ty_str { true }
          _ { false }
        };
}

pure fn type_is_unique(ty: t) -> bool {
    alt get(ty).struct {
      ty_uniq(_) { ret true; }
      ty_vec(_) { true }
      ty_str { true }
      _ { ret false; }
    }
}

pure fn sty_is_scalar<T>(ty: sty_base<T>) -> bool {
    alt ty {
      ty_nil | ty_bool | ty_int(_) | ty_float(_) | ty_uint(_) |
      ty_type | ty_ptr(_) | ty_rptr(_, _) { true }
      _ { false }
    }
}

pure fn type_is_scalar(ty: t) -> bool {
    sty_is_scalar(get(ty).struct)
}

// FIXME maybe inline this for speed?
fn type_is_immediate(ty: t) -> bool {
    ret type_is_scalar(ty) || type_is_boxed(ty) ||
        type_is_unique(ty);
}

fn type_needs_drop(cx: ctxt, ty: t) -> bool {
    alt cx.needs_drop_cache.find(ty) {
      some(result) { ret result; }
      none {/* fall through */ }
    }

    let accum = false;
    let result = alt get(ty).struct {
      // scalar types
      ty_nil | ty_bot | ty_bool | ty_int(_) | ty_float(_) | ty_uint(_) |
      ty_type | ty_ptr(_) | ty_rptr(_, _) { false }
      ty_rec(flds) {
        for f in flds { if type_needs_drop(cx, f.mt.ty) { accum = true; } }
        accum
      }
      ty_tup(elts) {
        for m in elts { if type_needs_drop(cx, m) { accum = true; } }
        accum
      }
      ty_enum(did, tps) {
        let variants = enum_variants(cx, did);
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

enum kind { kind_sendable, kind_copyable, kind_noncopyable, }

// Using these query functons is preferable to direct comparison or matching
// against the kind constants, as we may modify the kind hierarchy in the
// future.
pure fn kind_can_be_copied(k: kind) -> bool {
    ret alt k {
      kind_sendable { true }
      kind_copyable { true }
      kind_noncopyable { false }
    };
}

pure fn kind_can_be_sent(k: kind) -> bool {
    ret alt k {
      kind_sendable { true }
      kind_copyable { false }
      kind_noncopyable { false }
    };
}

fn proto_kind(p: proto) -> kind {
    alt p {
      ast::proto_any { kind_noncopyable }
      ast::proto_block { kind_noncopyable }
      ast::proto_box { kind_copyable }
      ast::proto_uniq { kind_sendable }
      ast::proto_bare { kind_sendable }
    }
}

fn kind_lteq(a: kind, b: kind) -> bool {
    alt a {
      kind_noncopyable { true }
      kind_copyable { b != kind_noncopyable }
      kind_sendable { b == kind_sendable }
    }
}

fn lower_kind(a: kind, b: kind) -> kind {
    if kind_lteq(a, b) { a } else { b }
}

fn type_kind(cx: ctxt, ty: t) -> kind {
    alt cx.kind_cache.find(ty) {
      some(result) { ret result; }
      none {/* fall through */ }
    }

    // Insert a default in case we loop back on self recursively.
    cx.kind_cache.insert(ty, kind_sendable);

    let result = alt get(ty).struct {
      // Scalar and unique types are sendable
      ty_nil | ty_bot | ty_bool | ty_int(_) | ty_uint(_) | ty_float(_) |
      ty_ptr(_) | ty_str { kind_sendable }
      ty_type { kind_copyable }
      ty_fn(f) { proto_kind(f.proto) }
      ty_opaque_closure_ptr(ck_block) { kind_noncopyable }
      ty_opaque_closure_ptr(ck_box) { kind_copyable }
      ty_opaque_closure_ptr(ck_uniq) { kind_sendable }
      // Those with refcounts-to-inner raise pinned to shared,
      // lower unique to shared. Therefore just set result to shared.
      ty_box(_) | ty_iface(_, _) | ty_opaque_box { kind_copyable }
      ty_rptr(_, _) { kind_copyable }
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
      // Enums lower to the lowest of their variants.
      ty_enum(did, tps) {
        let lowest = kind_sendable;
        for variant in *enum_variants(cx, did) {
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
      _ { cx.sess.bug("bad type in type_kind"); }
    };

    cx.kind_cache.insert(ty, result);
    ret result;
}

fn type_structurally_contains(cx: ctxt, ty: t, test: fn(sty) -> bool) ->
   bool {
    let sty = get(ty).struct;
    if test(sty) { ret true; }
    alt sty {
      ty_enum(did, tps) {
        for variant in *enum_variants(cx, did) {
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

// Returns true for noncopyable types and types where a copy of a value can be
// distinguished from the value itself. I.e. types with mutable content that's
// not shared through a pointer.
fn type_allows_implicit_copy(cx: ctxt, ty: t) -> bool {
    ret !type_structurally_contains(cx, ty, {|sty|
        alt sty {
          ty_param(_, _) { true }
          ty_vec(mt) {
            mt.mutbl != ast::m_imm
          }
          ty_rec(fields) {
            for field in fields {
                if field.mt.mutbl != ast::m_imm {
                    ret true;
                }
            }
            false
          }
          _ { false }
        }
    }) && type_kind(cx, ty) != kind_noncopyable;
}

fn type_structurally_contains_uniques(cx: ctxt, ty: t) -> bool {
    ret type_structurally_contains(cx, ty, {|sty|
        ret alt sty {
          ty_uniq(_) { ret true; }
          ty_vec(_) { true }
          ty_str { true }
          _ { ret false; }
        };
    });
}

fn sty_is_integral<T>(ty: sty_base<T>) -> bool {
    alt ty {
      ty_int(_) | ty_uint(_) | ty_bool { true }
      _ { false }
    }
}

fn struct_test(ty: t_i, pfn: fn(sty_base<t_i>) -> bool) -> bool {
    alt *ty {
      ty_var_i(_) { false }
      sty_i(sty) { pfn(sty) }
    }
}

fn type_is_integral(ty: t) -> bool {
    sty_is_integral(get(ty).struct)
}

fn sty_is_fp<T>(ty: sty_base<T>) -> bool {
    alt ty {
      ty_float(_) { true }
      _ { false }
    }
}

fn type_is_fp(ty: t) -> bool {
    sty_is_fp(get(ty).struct)
}

fn sty_is_numeric<T>(sty: sty_base<T>) -> bool {
    ret sty_is_integral(sty) || sty_is_fp(sty);
}

fn type_is_numeric(ty: t) -> bool {
    ret type_is_integral(ty) || type_is_fp(ty);
}

fn type_is_signed(ty: t) -> bool {
    alt get(ty).struct {
      ty_int(_) { true }
      _ { false }
    }
}

// Whether a type is Plain Old Data -- meaning it does not contain pointers
// that the cycle collector might care about.
fn type_is_pod(cx: ctxt, ty: t) -> bool {
    let result = true;
    alt get(ty).struct {
      // Scalar types
      ty_nil | ty_bot | ty_bool | ty_int(_) | ty_float(_) | ty_uint(_) |
      ty_type | ty_ptr(_) { result = true; }
      // Boxed types
      ty_str | ty_box(_) | ty_uniq(_) | ty_vec(_) | ty_fn(_) |
      ty_iface(_, _) | ty_rptr(_,_) | ty_opaque_box { result = false; }
      // Structural types
      ty_enum(did, tps) {
        let variants = enum_variants(cx, did);
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
      ty_param(_, _) { result = false; }
      ty_opaque_closure_ptr(_) { result = true; }
      _ { cx.sess.bug("unexpected type in type_is_pod"); }
    }

    ret result;
}

fn type_is_enum(ty: t) -> bool {
    alt get(ty).struct {
      ty_enum(_, _) { ret true; }
      _ { ret false;}
    }
}

fn sty_is_c_like_enum<T>(cx: ctxt, ty: sty_base<T>) -> bool {
    alt ty {
      ty_enum(did, tps) {
        let variants = enum_variants(cx, did);
        let some_n_ary = vec::any(*variants, {|v| vec::len(v.args) > 0u});
        ret !some_n_ary;
      }
      _ { ret false;}
    }
}

// Whether a type is enum like, that is a enum type with only nullary
// constructors
fn type_is_c_like_enum(cx: ctxt, ty: t) -> bool {
    sty_is_c_like_enum(cx, get(ty).struct)
}

fn type_param(ty: t) -> option<uint> {
    alt get(ty).struct {
      ty_param(id, _) { ret some(id); }
      _ {/* fall through */ }
    }
    ret none;
}

fn type_autoderef(cx: ctxt, t: t) -> t {
    let t1 = t;
    loop {
        alt get(t1).struct {
          ty_box(mt) | ty_uniq(mt) | ty::ty_rptr(_, mt) { t1 = mt.ty; }
          ty_res(_, inner, tps) {
            t1 = substitute_type_params(cx, tps, inner);
          }
          ty_enum(did, tps) {
            let variants = enum_variants(cx, did);
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
    fn hash_uint(id: uint, n: uint) -> uint { (id << 2u) + n }
    fn hash_def(id: uint, did: ast::def_id) -> uint {
        let h = (id << 2u) + (did.crate as uint);
        (h << 2u) + (did.node as uint)
    }
    fn hash_subty(id: uint, subty: t) -> uint { (id << 2u) + type_id(subty) }
    fn hash_subtys(id: uint, subtys: [t]) -> uint {
        let h = id;
        for s in subtys { h = (h << 2u) + type_id(s) }
        h
    }
    fn hash_type_constr(id: uint, c: @type_constr) -> uint {
        let h = id;
        h = (h << 2u) + hash_def(h, c.node.id);
        // FIXME this makes little sense
        for a in c.node.args {
            alt a.node {
              carg_base { h += h << 2u; }
              carg_lit(_) { fail "lit args not implemented yet"; }
              carg_ident(p) { h += h << 2u; }
            }
        }
        h
    }
    fn hash_region(r: region) -> uint {
        alt r {
          re_named(_)   { 1u }
          re_caller(_)  { 2u }
          re_self(_)    { 3u }
          re_block(_)   { 4u }
          re_inferred   { 5u }
        }
    }
    alt st {
      ty_nil { 0u } ty_bool { 1u }
      ty_int(t) {
        alt t {
          ast::ty_i { 2u } ast::ty_char { 3u } ast::ty_i8 { 4u }
          ast::ty_i16 { 5u } ast::ty_i32 { 6u } ast::ty_i64 { 7u }
        }
      }
      ty_uint(t) {
        alt t {
          ast::ty_u { 8u } ast::ty_u8 { 9u } ast::ty_u16 { 10u }
          ast::ty_u32 { 11u } ast::ty_u64 { 12u }
        }
      }
      ty_float(t) {
        alt t { ast::ty_f { 13u } ast::ty_f32 { 14u } ast::ty_f64 { 15u } }
      }
      ty_str { 17u }
      ty_enum(did, tys) {
        let h = hash_def(18u, did);
        for typ: t in tys { h = hash_subty(h, typ); }
        h
      }
      ty_box(mt) { hash_subty(19u, mt.ty) }
      ty_vec(mt) { hash_subty(21u, mt.ty) }
      ty_rec(fields) {
        let h = 26u;
        for f in fields { h = hash_subty(h, f.mt.ty); }
        h
      }
      ty_tup(ts) { hash_subtys(25u, ts) }
      ty_fn(f) {
        let h = 27u;
        for a in f.inputs { h = hash_subty(h, a.ty); }
        hash_subty(h, f.output)
      }
      ty_param(pid, did) { hash_def(hash_uint(31u, pid), did) }
      ty_self(ts) {
        let h = 28u;
        for t in ts { h = hash_subty(h, t); }
        h
      }
      ty_type { 32u }
      ty_bot { 34u }
      ty_ptr(mt) { hash_subty(35u, mt.ty) }
      ty_rptr(region, mt) {
        let h = (46u << 2u) + hash_region(region);
        hash_subty(h, mt.ty)
      }
      ty_res(did, sub, tps) {
        let h = hash_subty(hash_def(18u, did), sub);
        hash_subtys(h, tps)
      }
      ty_constr(t, cs) {
        let h = hash_subty(36u, t);
        for c in cs { h = (h << 2u) + hash_type_constr(h, c); }
        h
      }
      ty_uniq(mt) { hash_subty(37u, mt.ty) }
      ty_iface(did, tys) {
        let h = hash_def(40u, did);
        for typ: t in tys { h = hash_subty(h, typ); }
        h
      }
      ty_opaque_closure_ptr(ck_block) { 41u }
      ty_opaque_closure_ptr(ck_box) { 42u }
      ty_opaque_closure_ptr(ck_uniq) { 43u }
      ty_opaque_box { 44u }
      ty_class(did, tys) {
          let h = hash_def(45u, did);
          for typ: t in tys { h = hash_subty(h, typ); }
          h
      }
    }
}

fn arg_eq<T>(eq: fn(T, T) -> bool,
             a: @sp_constr_arg<T>,
             b: @sp_constr_arg<T>)
   -> bool {
    alt a.node {
      ast::carg_base {
        alt b.node { ast::carg_base { ret true; } _ { ret false; } }
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

fn args_eq<T>(eq: fn(T, T) -> bool,
              a: [@sp_constr_arg<T>],
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

fn node_id_to_type(cx: ctxt, id: ast::node_id) -> t {
    alt smallintmap::find(*cx.node_types, id as uint) {
       some(t) { t }
       none { cx.sess.bug(#fmt("node_id_to_type: no type for node %s",
                               ast_map::node_str(cx.items, id))); }
    }
}

fn node_id_to_type_params(cx: ctxt, id: ast::node_id) -> [t] {
    alt cx.node_type_substs.find(id) {
      none { ret []; }
      some(ts) { ret ts; }
    }
}

fn node_id_has_type_params(cx: ctxt, id: ast::node_id) -> bool {
    ret cx.node_type_substs.contains_key(id);
}

// Type accessors for substructures of types
fn ty_fn_args(fty: t) -> [arg] {
    alt get(fty).struct {
      ty_fn(f) { f.inputs }
      _ { fail "ty_fn_args() called on non-fn type"; }
    }
}

fn ty_fn_proto(fty: t) -> ast::proto {
    alt get(fty).struct {
      ty_fn(f) { f.proto }
      _ { fail "ty_fn_proto() called on non-fn type"; }
    }
}

pure fn ty_fn_ret(fty: t) -> t {
    alt get(fty).struct {
      ty_fn(f) { f.output }
      _ { fail "ty_fn_ret() called on non-fn type"; }
    }
}

fn ty_fn_ret_style(fty: t) -> ast::ret_style {
    alt get(fty).struct {
      ty_fn(f) { f.ret_style }
      _ { fail "ty_fn_ret_style() called on non-fn type"; }
    }
}

fn is_fn_ty(fty: t) -> bool {
    alt get(fty).struct {
      ty_fn(_) { ret true; }
      _ { ret false; }
    }
}

// Type accessors for AST nodes
fn block_ty(cx: ctxt, b: ast::blk) -> t {
    ret node_id_to_type(cx, b.node.id);
}


// Returns the type of a pattern as a monotype. Like @expr_ty, this function
// doesn't provide type parameter substitutions.
fn pat_ty(cx: ctxt, pat: @ast::pat) -> t {
    ret node_id_to_type(cx, pat.id);
}

// Returns the type of an expression as a monotype.
//
// NB: This type doesn't provide type parameter substitutions; e.g. if you
// ask for the type of "id" in "id(3)", it will return "fn(&int) -> int"
// instead of "fn(t) -> T with T = int". If this isn't what you want, see
// expr_ty_params_and_ty() below.
fn expr_ty(cx: ctxt, expr: @ast::expr) -> t {
    ret node_id_to_type(cx, expr.id);
}

fn expr_ty_params_and_ty(cx: ctxt, expr: @ast::expr) -> {params: [t], ty: t} {
    ret {params: node_id_to_type_params(cx, expr.id),
         ty: node_id_to_type(cx, expr.id)};
}

fn expr_has_ty_params(cx: ctxt, expr: @ast::expr) -> bool {
    ret node_id_has_type_params(cx, expr.id);
}

fn expr_is_lval(method_map: typeck::method_map, e: @ast::expr) -> bool {
    alt e.node {
      ast::expr_path(_) | ast::expr_unary(ast::deref, _) { true }
      ast::expr_field(_, _, _) | ast::expr_index(_, _) {
        !method_map.contains_key(e.id)
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

fn field_idx<T>(id: ast::ident, fields: [field_base<T>]) -> option<uint> {
    let i = 0u;
    for f in fields { if f.ident == id { ret some(i); } i += 1u; }
    ret none;
}

fn get_field(rec_ty: t, id: ast::ident) -> field {
    alt check vec::find(get_fields(rec_ty), {|f| str::eq(f.ident, id) }) {
      some(f) { f }
    }
}

fn get_fields(rec_ty:t) -> [field] {
    alt check get(rec_ty).struct {
      ty_rec(fields) { fields }
    }
}

fn method_idx(id: ast::ident, meths: [method]) -> option<uint> {
    let i = 0u;
    for m in meths { if m.ident == id { ret some(i); } i += 1u; }
    ret none;
}

fn sort_methods(meths: [method]) -> [method] {
    fn method_lteq(a: method, b: method) -> bool {
        ret str::le(a.ident, b.ident);
    }
    ret std::sort::merge_sort(bind method_lteq(_, _), meths);
}

// Maintains a little union-set tree for inferred modes.  `canon()` returns
// the current head value for `m0`.
fn canon<T:copy>(tbl: hashmap<ast::node_id, ast::inferable<T>>,
                 m0: ast::inferable<T>) -> ast::inferable<T> {
    alt m0 {
      ast::infer(id) {
        alt tbl.find(id) {
          none { m0 }
          some(m1) {
            let cm1 = canon(tbl, m1);
            // path compression:
            if cm1 != m1 { tbl.insert(id, cm1); }
            cm1
          }
        }
      }
      _ { m0 }
    }
}

// Maintains a little union-set tree for inferred modes.  `resolve_mode()`
// returns the current head value for `m0`.
fn canon_mode(cx: ctxt, m0: ast::mode) -> ast::mode {
    canon(cx.inferred_modes, m0)
}

// Returns the head value for mode, failing if `m` was a infer(_) that
// was never inferred.  This should be safe for use after typeck.
fn resolved_mode(cx: ctxt, m: ast::mode) -> ast::rmode {
    alt canon_mode(cx, m) {
      ast::infer(_) {
        cx.sess.bug(#fmt["mode %? was never resolved", m]);
      }
      ast::expl(m0) { m0 }
    }
}

fn arg_mode(cx: ctxt, a: arg) -> ast::rmode { resolved_mode(cx, a.mode) }

// Unifies `m1` and `m2`.  Returns unified value or failure code.
fn unify_mode(cx: ctxt, m1: ast::mode, m2: ast::mode)
    -> result<ast::mode, type_err> {
    alt (canon_mode(cx, m1), canon_mode(cx, m2)) {
      (m1, m2) if (m1 == m2) {
        result::ok(m1)
      }
      (ast::infer(id1), ast::infer(id2)) {
        cx.inferred_modes.insert(id2, m1);
        result::ok(m1)
      }
      (ast::infer(id), m) | (m, ast::infer(id)) {
        cx.inferred_modes.insert(id, m);
        result::ok(m1)
      }
      (m1, m2) {
        result::err(terr_mode_mismatch(m1, m2))
      }
    }
}

// If `m` was never unified, unifies it with `m_def`.  Returns the final value
// for `m`.
fn set_default_mode(cx: ctxt, m: ast::mode, m_def: ast::rmode) {
    alt canon_mode(cx, m) {
      ast::infer(id) {
        cx.inferred_modes.insert(id, ast::expl(m_def));
      }
      ast::expl(_) { }
    }
}

fn ty_to_ty_i(tcx: ctxt, t: t) -> t_i {
    fn ty_to_sty_i(tcx: ctxt, t: t) -> sty_i {
        fold_sty_base(get(t).struct, {|t| ty_to_ty_i(tcx, t)})
    }

    @sty_i(ty_to_sty_i(tcx, t))
}

fn ty_to_ty_i_subst(tcx: ctxt, t: t, substs: [t_i]) -> t_i {
    alt get(t).struct {
      ty_param(idx, _) {
        substs[idx]
      }
      sty {
        @sty_i(fold_sty_base(sty) {|t1|
            ty_to_ty_i_subst(tcx, t1, substs)
        })
      }
    }
}

// Type unification via Robinson's algorithm (Robinson 1965). Implemented as
// described in Hoder and Voronkov:
//
//     http://www.cs.man.ac.uk/~hoderk/ubench/unification_full.pdf
mod unify {
    import result::{result, ok, ok1, err, chain, map, map2};
    import std::list;

    export resolve_type;
    export resolve_type_var;
    export unify;
    export get_var_binding;
    export uctxt;

    type ures<T> = result<T,type_err>;

    // in case of failure, value is the idx of an unresolved type var
    type fres<T> = result<T,resolve_err>;

    // Unifies two sets.
    fn union<T:copy>(
        vb: @var_bindings, set_a: uint, set_b: uint,
        variance: variance, nxt: fn() -> ures<T>) -> ures<T> {

        ufind::grow(vb.sets, uint::max(set_a, set_b) + 1u);
        let root_a = ufind::find(vb.sets, set_a);
        let root_b = ufind::find(vb.sets, set_b);

        let replace_type = fn@(vb: @var_bindings, t: sty_i) {
            ufind::union(vb.sets, set_a, set_b);
            let root_c: uint = ufind::find(vb.sets, set_a);
            smallintmap::insert(vb.var_types, root_c, t);
        };

        alt smallintmap::find(vb.var_types, root_a) {
          none {
            alt smallintmap::find(vb.var_types, root_b) {
              none { ufind::union(vb.sets, set_a, set_b); ret nxt(); }
              some(t_b) { replace_type(vb, t_b); ret nxt(); }
            }
          }
          some(t_a) {
            alt smallintmap::find(vb.var_types, root_b) {
              none { replace_type(vb, t_a); ret nxt(); }
              some(t_b) {
                ret unify_sty(vb, t_a, t_b, variance) {|t_c|
                    replace_type(vb, t_c);
                    nxt()
                };
              }
            }
          }
        }
    }

    fn record_var_binding<T:copy>(
        vb: @var_bindings, key: int,
        typ: sty_i, variance: variance,
        nxt: fn(&&t_i) -> ures<T>) -> ures<T> {

        ufind::grow(vb.sets, (key as uint) + 1u);
        let root = ufind::find(vb.sets, key as uint);
        alt smallintmap::find(vb.var_types, root) {
          some(old_type) {
            alt unify_sty(vb, old_type, typ, variance, ok1(_)) {
              err(e) {
                ret err(e);
              }
              ok(unified_type) {
                smallintmap::insert(vb.var_types, root, unified_type);
                ret nxt(@ty_var_i(key));
              }
            }
          }
          none {
            smallintmap::insert(vb.var_types, root, typ);
            ret nxt(@ty_var_i(key));
          }
        }
    }

    fn get_var_binding<T>(vb: @var_bindings, vid: int,
                          unbound: fn(int) -> T,
                          bound: fn(sty_i) -> T) -> T {
        if vid as uint >= ufind::set_count(vb.sets) {
            ret unbound(vid);
        }
        let root_id = ufind::find(vb.sets, vid as uint);
        #debug["vid=%d root_id=%?", vid, root_id];
        alt smallintmap::find(vb.var_types, root_id) {
          none { ret unbound(vid); }
          some(rt) { ret bound(rt); }
        }
    }

    iface uctxt<T> {
        fn tcx() -> ctxt;
        fn unify_step<R:copy>(
            expected: T, actual: T, variance: variance,
            nxt: fn(T) -> ures<R>) -> ures<R>;
    }

    impl of uctxt<t_i> for @var_bindings {
        fn tcx() -> ctxt { self.tcx }

        fn unify_step<R:copy>(
            expected: t_i, actual: t_i, variance: variance,
            nxt: fn(&&t_i) -> ures<R>) -> ures<R> {

            alt (*expected, *actual) {
              (ty_var_i(e_id), ty_var_i(a_id)) {
                union(self, e_id as uint, a_id as uint, variance) {||
                    nxt(actual)
                }
              }
              (sty_i(e), ty_var_i(a_id)) {
                let v = variance_transform(variance, contravariant);
                record_var_binding(self, a_id, e, v, nxt)
              }
              (ty_var_i(e_id), sty_i(a)) {
                let v = variance_transform(variance, covariant);
                record_var_binding(self, e_id, a, v, nxt)
              }
              (sty_i(e), sty_i(a)) {
                unify_sty(self, e, a, variance) {|r|
                    nxt(@sty_i(r))
                }
              }
            }
        }
    }

    impl of uctxt<t> for ctxt {
        fn tcx() -> ctxt { self }

        fn unify_step<R:copy>(
            expected: t, actual: t, variance: variance,
            nxt: fn(&&t) -> ures<R>) -> ures<R> {

            if expected == actual { ret nxt(expected); } // fast path

            unify_sty(self, get(expected).struct,
                      get(actual).struct, variance) {|sty|
                nxt(mk_t(self, sty))
            }
        }
    }

    fn unify<T:copy,U:uctxt<T>>(cx: U, expected: T, actual: T) -> ures<T> {
        ret cx.unify_step(expected, actual, covariant, ok1(_));
    }

    // Simple structural type comparison.
    fn struct_cmp<T:copy,U:uctxt<T>,R:copy>(
        cx: U, expected: sty_base<T>, actual: sty_base<T>,
        nxt: fn(&&sty_base<T>) -> ures<R>) -> ures<R> {

        let tcx = cx.tcx();
        let cfg = tcx.sess.targ_cfg;
        if mach_sty(cfg, expected) == mach_sty(cfg, actual) {
            ret nxt(expected);
        }
        ret err(terr_mismatch);
    }

    // Right now this just checks that the lists of constraints are
    // pairwise equal.
    fn unify_constrs<R:copy>(
        expected: [@type_constr],
        actual: [@type_constr],
        nxt: fn([@type_constr]) -> ures<R>) -> ures<R> {

        if check vec::same_length(expected, actual) {
            map2(expected, actual,
                 {|e,a| unify_constr(e, a, ok1(_))}, nxt)
        } else {
            ret err(terr_constr_len(expected.len(), actual.len()));
        }
    }

    fn unify_constr<R:copy>(
        e_constr: @type_constr,
        a_constr: @type_constr,
        nxt: fn(&&@type_constr) -> ures<R>) -> ures<R> {

        if e_constr.node.id != a_constr.node.id {
            ret err(terr_constr_mismatch(e_constr, a_constr));
        }
        let e_args = e_constr.node.args;
        let a_args = a_constr.node.args;
        if check vec::same_length(e_args, a_args) {
            let check_arg = fn@(&&e_arg: @ty_constr_arg,
                                &&a_arg: @ty_constr_arg) -> ures<()> {
                if e_arg.node != a_arg.node {
                    ret err(terr_constr_mismatch(e_constr, a_constr));
                } else {
                    ret ok(());
                }
            };
            map2(e_args, a_args, check_arg) {|_i|
                ret nxt(e_constr);
            }
        } else {
            ret err(terr_constr_mismatch(e_constr, a_constr));
        }
    }

    // Unifies two mutability flags.
    fn unify_mut<R:copy>(
        expected: ast::mutability, actual: ast::mutability,
        variance: variance, mut_err: type_err,
        nxt: fn(ast::mutability, variance) -> ures<R>) -> ures<R> {

        // If you're unifying on something mutable then we have to
        // be invariant on the inner type
        let newvariance = alt expected {
          ast::m_mutbl {
            variance_transform(variance, invariant)
          }
          _ {
            variance_transform(variance, covariant)
          }
        };

        if expected == actual {
            ret nxt(expected, newvariance);
        }
        if variance == covariant {
            if expected == ast::m_const {
                ret nxt(actual, newvariance);
            }
        } else if variance == contravariant {
            if actual == ast::m_const {
                ret nxt(expected, newvariance);
            }
        }
        ret err(mut_err);
    }

    fn unify_fn_proto<R:copy>(
        e_proto: ast::proto, a_proto: ast::proto, variance: variance,
        nxt: fn(ast::proto) -> ures<R>) -> ures<R> {

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
              (_, ast::proto_any) { true }
              (ast::proto_bare, _) { true }

              // Equal prototypes are always subprotos:
              (_, _) { p_sub == p_sup }
            };
        }

        ret alt variance {
          invariant if e_proto == a_proto { nxt(e_proto) }
          covariant if sub_proto(a_proto, e_proto) { nxt(e_proto) }
          contravariant if sub_proto(e_proto, a_proto) { nxt(e_proto) }
          _ { ret err(terr_mismatch) }
        };
    }

    fn unify_arg<T:copy,U:uctxt<T>,R:copy>(
        cx: U, e_arg: arg_base<T>, a_arg: arg_base<T>,
        variance: variance, nxt: fn(arg_base<T>) -> ures<R>) -> ures<R> {

        // Unify the result modes.
        chain(unify_mode(cx.tcx(), e_arg.mode, a_arg.mode)) {|mode|
            cx.unify_step(e_arg.ty, a_arg.ty, variance) {|ty|
                nxt({mode: mode, ty: ty})
            }
        }
    }

    fn unify_args<T:copy,U:uctxt<T>,R:copy>(
        cx: U, e_args: [arg_base<T>], a_args: [arg_base<T>],
        variance: variance, nxt: fn([arg_base<T>]) -> ures<R>) -> ures<R> {

        if check vec::same_length(e_args, a_args) {
            // The variance changes (flips basically) when descending
            // into arguments of function types
            let variance = variance_transform(variance, contravariant);
            map2(e_args, a_args,
                 {|e,a| unify_arg(cx, e, a, variance, ok1(_))},
                 nxt)
        } else {
            ret err(terr_arg_count);
        }
    }

    fn unify_ret_style<R:copy>(
        e_ret_style: ret_style,
        a_ret_style: ret_style,
        nxt: fn(ret_style) -> ures<R>) -> ures<R> {

        if a_ret_style != ast::noreturn && a_ret_style != e_ret_style {
            /* even though typestate checking is mostly
               responsible for checking control flow annotations,
               this check is necessary to ensure that the
               annotation in an object method matches the
               declared object type */
            ret err(terr_ret_style_mismatch(e_ret_style, a_ret_style));
        } else {
            nxt(a_ret_style)
        }
    }

    fn unify_fn<T:copy,U:uctxt<T>,R:copy>(
        cx: U, e_f: fn_ty_base<T>, a_f: fn_ty_base<T>, variance: variance,
        nxt: fn(&&fn_ty_base<T>) -> ures<R>) -> ures<R> {

        unify_fn_proto(e_f.proto, a_f.proto, variance) {|proto|
            unify_ret_style(e_f.ret_style, a_f.ret_style) {|rs|
                unify_args(cx, e_f.inputs, a_f.inputs, variance) {|args|
                    cx.unify_step(e_f.output, a_f.output, variance) {|rty|
                        let cs = e_f.constraints; // FIXME: Unify?
                        nxt({proto: proto,
                             inputs: args,
                             output: rty,
                             ret_style: rs,
                             constraints: cs})
                    }
                }
            }
        }
    }

    fn unify_tys<T:copy,U:uctxt<T>,R:copy>(
        cx: U, expected_tps: [T], actual_tps: [T],
        variance: variance, nxt: fn([T]) -> ures<R>)
        : vec::same_length(expected_tps, actual_tps)
        -> ures<R> {

        map2(expected_tps, actual_tps,
             {|e,a| cx.unify_step(e, a, variance, ok1(_))},
             nxt)
    }

    fn unify_tps<T:copy,U:uctxt<T>,R:copy>(
        cx: U, expected_tps: [T], actual_tps: [T],
        variance: variance, nxt: fn([T]) -> ures<R>)
        -> ures<R> {

        if check vec::same_length(expected_tps, actual_tps) {
            unify_tys(cx, expected_tps, actual_tps, variance, nxt)
        } else {
            err(terr_ty_param_size(expected_tps.len(),
                                   actual_tps.len()))
        }
    }

    fn unify_mt<T:copy,U:uctxt<T>,R:copy>(
        cx: U, e_mt: mt_base<T>, a_mt: mt_base<T>, variance: variance,
        mut_err: type_err, nxt: fn(mt_base<T>) -> ures<R>) -> ures<R> {

        unify_mut(e_mt.mutbl, a_mt.mutbl, variance, mut_err) {|mutbl,var|
            cx.unify_step(e_mt.ty, a_mt.ty, var) {|ty|
                nxt({ty: ty, mutbl: mutbl})
            }
        }
    }

    fn unify_regions<T:copy,U:uctxt<T>,R:copy>(
        cx: U, e_region: region, a_region: region,
        variance: variance, nxt: fn(region) -> ures<R>) -> ures<R> {

        let sub, super;
        alt variance {
            covariant { super = e_region; sub = a_region; }
            contravariant { super = a_region; sub = e_region; }
            invariant {
              ret if e_region == a_region {
                  nxt(e_region)
              } else {
                  err(terr_regions_differ(true, e_region, a_region))
              };
            }
        }

        if sub == ty::re_inferred || super == ty::re_inferred {
            ret if sub == super {
                nxt(super)
            } else {
                err(terr_regions_differ(true, super, sub))
            };
        }

        // Outer regions are subtypes of inner regions. (This is somewhat
        // surprising!)
        let superscope = region::region_to_scope(cx.tcx().region_map, super);
        let subscope = region::region_to_scope(cx.tcx().region_map, sub);
        if region::scope_contains(cx.tcx().region_map, subscope, superscope) {
            ret nxt(super);
        }
        ret err(terr_regions_differ(false, sub, super));
    }

    fn unify_field<T:copy,U:uctxt<T>,R:copy>(
        cx: U, e_field: field_base<T>, a_field: field_base<T>,
        variance: variance, nxt: fn(field_base<T>) -> ures<R>) -> ures<R> {

        if e_field.ident != a_field.ident {
            ret err(terr_record_fields(e_field.ident,
                                       a_field.ident));
        }

        unify_mt(cx, e_field.mt, a_field.mt, variance,
                 terr_record_mutability) {|mt|
            nxt({ident: e_field.ident, mt: mt})
        }
    }

    fn unify_sty<T:copy,U:uctxt<T>,R:copy>(
        cx: U, expected: sty_base<T>, actual: sty_base<T>,
        variance: variance, nxt: fn(&&sty_base<T>) -> ures<R>) -> ures<R> {

        alt (expected, actual) { // induces copies, uncool.
          (_, ty_bot) { nxt(expected) }
          (ty_bot, _) { nxt(actual) }
          (ty_nil, _) | (ty_bool, _) | (ty_int(_), _) | (ty_uint(_), _) |
          (ty_float(_), _) | (ty_str, _) {
            struct_cmp(cx, expected, actual, nxt)
          }
          (ty_param(e_n, _), ty_param(a_n, _)) if e_n == a_n {
            nxt(expected)
          }
          (ty_enum(e_id, e_tps), ty_enum(a_id, a_tps)) if e_id == a_id {
            unify_tps(cx, e_tps, a_tps, variance) {|tps|
                nxt(ty_enum(e_id, tps))
            }
          }
          (ty_iface(e_id, e_tps), ty_iface(a_id, a_tps)) if e_id == a_id {
            unify_tps(cx, e_tps, a_tps, variance) {|tps|
                nxt(ty_iface(e_id, tps))
            }
          }
          (ty_class(e_id, e_tps), ty_class(a_id, a_tps)) if e_id == a_id {
            unify_tps(cx, e_tps, a_tps, variance) {|tps|
                nxt(ty_class(e_id, tps))
            }
          }
          (ty_box(e_mt), ty_box(a_mt)) {
            unify_mt(cx, e_mt, a_mt, variance, terr_box_mutability,
                     {|mt| nxt(ty_box(mt))})
          }
          (ty_uniq(e_mt), ty_uniq(a_mt)) {
            unify_mt(cx, e_mt, a_mt, variance, terr_box_mutability,
                     {|mt| nxt(ty_uniq(mt))})
          }
          (ty_vec(e_mt), ty_vec(a_mt)) {
            unify_mt(cx, e_mt, a_mt, variance, terr_vec_mutability,
                     {|mt| nxt(ty_vec(mt))})
          }
          (ty_ptr(e_mt), ty_ptr(a_mt)) {
            unify_mt(cx, e_mt, a_mt, variance, terr_ptr_mutability,
                     {|mt| nxt(ty_ptr(mt))})
          }
          (ty_rptr(e_region, e_mt), ty_rptr(a_region, a_mt)) {
            unify_regions::<T,U,R>(cx, e_region, a_region, variance) {|r|
                unify_mt(cx, e_mt, a_mt, variance, terr_ref_mutability,
                         {|mt| nxt(ty_rptr(r, mt))})
            }
          }
          (ty_res(e_id, e_inner, e_tps), ty_res(a_id, a_inner, a_tps))
          if e_id == a_id {
            cx.unify_step(e_inner, a_inner, variance) {|t|
                unify_tps(cx, e_tps, a_tps, variance) {|tps|
                    nxt(ty_res(a_id, t, tps))
                }
            }
          }
          (ty_rec(e_fields), ty_rec(a_fields)) {
              if check vec::same_length(e_fields, a_fields) {
                  map2(e_fields, a_fields,
                       {|e,a| unify_field(cx, e, a, variance, ok1(_))},
                       {|fields| nxt(ty_rec(fields))})
              } else {
                  ret err(terr_record_size(e_fields.len(),
                                           a_fields.len()));
              }
          }
          (ty_tup(e_elems), ty_tup(a_elems)) {
            if check vec::same_length(e_elems, a_elems) {
                unify_tys(cx, e_elems, a_elems, variance) {|elems|
                    nxt(ty_tup(elems))
                }
            } else {
                err(terr_tuple_size(e_elems.len(), a_elems.len()))
            }
          }
          (ty_fn(e_fty), ty_fn(a_fty)) {
            unify_fn(cx, e_fty, a_fty, variance) {|fty|
                nxt(ty_fn(fty))
            }
          }
          (ty_constr(e_t, e_constrs), ty_constr(a_t, a_constrs)) {
            // unify the base types...
            cx.unify_step(e_t, a_t, variance) {|rty|
                // FIXME: probably too restrictive --
                // requires the constraints to be syntactically equal
                unify_constrs(e_constrs, a_constrs) {|constrs|
                    nxt(ty_constr(rty, constrs))
                }
            }
          }
          /*NDM
          (ty_constr(e_t, e_constrs), _) {
            // If the actual type is *not* a constrained type,
            // then we go ahead and just ignore the constraints on
            // the expected type. typestate handles the rest.
            cx.unify_step(e_t, sty_i(actual), variance) {|e_t|
                // FIXME--is it ok to assign this type? seems a bit risky.
                nxt(ty_constr(e_t, e_constrs))
            }
          }
          */
          _ { err(terr_mismatch) }
        }
    }

    fn refresh_var<T>(vb: @var_bindings, vid: int) -> t_i {
        get_var_binding(
            vb, vid,
            /* if unbound: */ {|vid| @ty_var_i(vid)},
            /* if bound:   */ {|s| @sty_i(s) })
    }

    type resolve_ctxt = {
        vb: @var_bindings,
        vars_seen: @list::list<int>
    };

    fn resolve_sty<T:copy>(
        rc: resolve_ctxt, t: sty_i,
        nxt: fn(&&sty_base<t>) -> fres<T>) -> fres<T> {

        chain(fold_sty_base_err(t, resolve_t(rc, _, ok1(_))), nxt)
    }

    fn resolve_t<T:copy>(
        rc: resolve_ctxt, &&t: t_i, nxt: fn(&&t) -> fres<T>) -> fres<T> {

        alt *t {
          ty_var_i(vid) {
            if list::has(*rc.vars_seen, vid) {
                err(rerr_cyclic_var(vid))
            } else {
                get_var_binding(
                    rc.vb, vid,

                    // if unbound:
                    {|vid| err(rerr_unresolved_var(vid))},

                    // if bound:
                    {|sty|
                        let rc1 = {vars_seen: @list::cons(vid, rc.vars_seen)
                                   with rc};
                        resolve_sty(rc1, sty) {|sty|
                            nxt(mk_t(rc.vb.tcx, sty))
                        }
                    })
            }
          }

          sty_i(sty) {
            resolve_sty(rc, sty) {|sty|
                nxt(mk_t(rc.vb.tcx, sty))
            }
          }
        }
    }

    // If the given type is a variable, returns the structure of that type.
    fn resolve_type(vb: @var_bindings, t: t_i) -> fres<t> {
        let rc = {vb: vb, vars_seen: @list::nil};
        resolve_t(rc, t, ok1(_))
    }

    // Specifies the allowable subtyping between expected and actual types
    enum variance {
        // Actual may be a subtype of expected
        covariant,
        // Actual may be a supertype of expected
        contravariant,
        // Actual must be the same type as expected
        invariant,
    }

    // The calculation for recursive variance
    // "Taming the Wildcards: Combining Definition- and Use-Site Variance"
    // by John Altidor, et. al.
    //
    // I'm just copying the table from figure 1 - haven't actually
    // read the paper (yet).
    fn variance_transform(a: variance, b: variance) -> variance {
        alt a {
          covariant {
            alt b {
              covariant { covariant }
              contravariant { contravariant }
              invariant { invariant }
            }
          }
          contravariant {
            alt b {
              covariant { contravariant }
              contravariant { covariant }
              invariant { invariant }
            }
          }
          invariant {
            alt b {
              covariant { invariant }
              contravariant { invariant }
              invariant { invariant }
            }
          }
        }
    }

    fn dump_var_bindings(vb: @var_bindings) {
        let i = 0u;
        while i < vec::len::<ufind::node>(vb.sets.nodes) {
            let sets = "";
            let j = 0u;
            while j < vec::len::<option<uint>>(vb.sets.nodes) {
                if ufind::find(vb.sets, j) == i { sets += #fmt[" %u", j]; }
                j += 1u;
            }
            let typespec;
            alt smallintmap::find(vb.var_types, i) {
              none { typespec = ""; }
              some(typ) { typespec = " =" + ty_i_to_str(vb, @sty_i(typ)); }
            }
            #error("set %u:%s%s", i, typespec, sets);
            i += 1u;
        }
    }

    fn resolve_type_var(vb: @var_bindings, vid: int) -> fres<t> {
        resolve_type(vb, @ty_var_i(vid))
    }
}

fn resolve_err_to_str(err: resolve_err) -> str {
    alt err {
      rerr_unresolved_var(_) { ret "type variable unresolved"; }
      rerr_cyclic_var(_) { ret "type of infinite size"; }
    }
}

fn type_err_to_str(cx: ctxt, err: type_err) -> str {
    alt err {
      terr_mismatch { ret "types differ"; }
      terr_ret_style_mismatch(expect, actual) {
        fn to_str(s: ast::ret_style) -> str {
            alt s {
              ast::noreturn { "non-returning" }
              ast::return_val { "return-by-value" }
            }
        }
        ret to_str(actual) + " function found where " + to_str(expect) +
            " function was expected";
      }
      terr_box_mutability { ret "boxed values differ in mutability"; }
      terr_vec_mutability { ret "vectors differ in mutability"; }
      terr_ptr_mutability { ret "pointers differ in mutability"; }
      terr_ref_mutability { ret "references differ in mutability"; }
      terr_ty_param_size(e_sz, a_sz) {
        ret "expected a type with " + uint::to_str(e_sz, 10u) +
                " type params but found one with " + uint::to_str(a_sz, 10u) +
                " type params";
      }
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
      terr_record_mutability { ret "record elements differ in mutability"; }
      terr_record_fields(e_fld, a_fld) {
        ret "expected a record with field '" + e_fld +
                "' but found one with field '" + a_fld + "'";
      }
      terr_arg_count { ret "incorrect number of function parameters"; }
      terr_mode_mismatch(e_mode, a_mode) {
        ret "expected argument mode " + mode_to_str(e_mode) + " but found " +
                mode_to_str(a_mode);
      }
      terr_constr_len(e_len, a_len) {
        ret "expected a type with " + uint::str(e_len) +
                " constraints, but found one with " + uint::str(a_len) +
                " constraints";
      }
      terr_constr_mismatch(e_constr, a_constr) {
        ret "expected a type with constraint " + ty_constr_to_str(e_constr) +
                " but found one with constraint " +
                ty_constr_to_str(a_constr);
      }
      terr_regions_differ(true, region_a, region_b) {
        ret #fmt("reference lifetime %s does not match reference lifetime %s",
                 region_to_str(cx, region_a), region_to_str(cx, region_b));
      }
      terr_regions_differ(false, subregion, superregion) {
        ret #fmt("references with lifetime %s do not outlive references with \
                  lifetime %s",
                 region_to_str(cx, subregion),
                 region_to_str(cx, superregion));
      }
      terr_cyclic_type {
        ret "cyclic type of infinite size";
      }
    }
}

// Replaces type parameters in the given type using the given list of
// substitions.
fn substitute_type_params(cx: ctxt, substs: [ty::t], typ: t) -> t {
    if !type_has_params(typ) { ret typ; }

    // Precondition? idx < vec::len(substs)
    if !get(typ).t_flags.has_params { ret typ; }
    fold(cx, typ) {|t|
        alt get(t).struct {
          ty_param(i, d) { substs[i] }
          _ { t }
        }
    }
}

fn substitute_type_params_i(vb: @var_bindings, substs: [ty::t_i], &&typ: t_i) -> t_i {
    fold(vb, typ) {|t|
        alt *t {
          sty_i(ty_param(idx, _id)) { substs[idx] }
          _ { t }
        }
    }
}

fn def_has_ty_params(def: ast::def) -> bool {
    alt def {
      ast::def_fn(_, _) | ast::def_variant(_, _) { true }
      _ { false }
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

fn impl_iface(cx: ctxt, id: ast::def_id) -> option<t> {
    if id.crate == ast::local_crate {
        alt cx.items.get(id.node) {
          ast_map::node_item(@{node: ast::item_impl(
              _, some(@{node: ast::ty_path(_, id), _}), _, _), _}, _) {
            some(node_id_to_type(cx, id))
          }
          _ { none }
        }
    } else {
        csearch::get_impl_iface(cx, id)
    }
}

fn ty_to_def_id(ty: t) -> ast::def_id {
    alt check get(ty).struct {
      ty_iface(id, _) | ty_class(id, _) | ty_res(id, _, _) | ty_enum(id, _) {
        id
      }
    }
}

// Enum information
type variant_info = @{args: [t], ctor_ty: t, name: str,
                      id: ast::def_id, disr_val: int};

fn substd_enum_variants(cx: ctxt, id: ast::def_id, tps: [ty::t])
    -> [variant_info] {
    vec::map(*enum_variants(cx, id)) { |variant_info|
        let substd_args = vec::map(variant_info.args) {|aty|
            substitute_type_params(cx, tps, aty)
        };

        let substd_ctor_ty =
            substitute_type_params(cx, tps, variant_info.ctor_ty);

        @{args: substd_args, ctor_ty: substd_ctor_ty with *variant_info}
    }
}

fn item_path_str(cx: ctxt, id: ast::def_id) -> str {
    ast_map::path_to_str(item_path(cx, id))
}

fn item_path(cx: ctxt, id: ast::def_id) -> ast_map::path {
    if id.crate != ast::local_crate {
        csearch::get_item_path(cx, id)
    } else {
        let node = cx.items.get(id.node);
        alt node {
          ast_map::node_item(item, path) {
            let item_elt = alt item.node {
              item_mod(_) | item_native_mod(_) {
                ast_map::path_mod(item.ident)
              }
              _ {
                ast_map::path_name(item.ident)
              }
            };
            *path + [item_elt]
          }

          ast_map::node_native_item(nitem, _, path) {
            *path + [ast_map::path_name(nitem.ident)]
          }

          ast_map::node_method(method, _, path) {
            *path + [ast_map::path_name(method.ident)]
          }

          ast_map::node_variant(variant, _, path) {
            vec::init(*path) + [ast_map::path_name(variant.node.name)]
          }

          ast_map::node_ctor(i, path) {
              *path + [ast_map::path_name(i.ident)]
          }

          ast_map::node_expr(_) | ast_map::node_arg(_, _) |
          ast_map::node_local(_) | ast_map::node_export(_, _) |
          ast_map::node_block(_) {
            cx.sess.bug(#fmt["cannot find item_path for node %?", node]);
          }
        }
    }
}

fn enum_variants(cx: ctxt, id: ast::def_id) -> @[variant_info] {
    alt cx.enum_var_cache.find(id) {
      some(variants) { ret variants; }
      _ { /* fallthrough */ }
    }
    let result = if ast::local_crate != id.crate {
        @csearch::get_enum_variants(cx, id)
    } else {
        // FIXME: Now that the variants are run through the type checker (to
        // check the disr_expr if it exists), this code should likely be
        // moved there to avoid having to call eval_const_expr twice.
        alt cx.items.get(id.node) {
          ast_map::node_item(@{node: ast::item_enum(variants, _), _}, _) {
            let disr_val = -1;
            @vec::map(variants, {|variant|
                let ctor_ty = node_id_to_type(cx, variant.node.id);
                let arg_tys = if vec::len(variant.node.args) > 0u {
                    vec::map(ty_fn_args(ctor_ty), {|a| a.ty})
                } else { [] };
                alt variant.node.disr_expr {
                  some (ex) {
                    // FIXME: issue #1417
                    disr_val = alt syntax::ast_util::eval_const_expr(cx, ex) {
                      ast_util::const_int(val) {val as int}
                      _ { cx.sess.bug("tag_variants: bad disr expr"); }
                    }
                  }
                  _ {disr_val += 1;}
                }
                @{args: arg_tys,
                  ctor_ty: ctor_ty,
                  name: variant.node.name,
                  id: ast_util::local_def(variant.node.id),
                  disr_val: disr_val
                 }
            })
          }
          _ { cx.sess.bug("tag_variants: id not bound to an enum"); }
        }
    };
    cx.enum_var_cache.insert(id, result);
    result
}


// Returns information about the enum variant with the given ID:
fn enum_variant_with_id(cx: ctxt, enum_id: ast::def_id,
                        variant_id: ast::def_id) -> variant_info {
    let variants = enum_variants(cx, enum_id);
    let i = 0u;
    while i < vec::len::<variant_info>(*variants) {
        let variant = variants[i];
        if def_eq(variant.id, variant_id) { ret variant; }
        i += 1u;
    }
    cx.sess.bug("enum_variant_with_id(): no variant exists with that ID");
}


// If the given item is in an external crate, looks up its type and adds it to
// the type cache. Returns the type parameters and type.
fn lookup_item_type(cx: ctxt, did: ast::def_id) -> ty_param_bounds_and_ty {
    /*
      Are we putting class ids in the tcache (where does that happen?)
     */
    alt cx.tcache.find(did) {
      some(tpt) { ret tpt; }
      none {
          #debug("lookup_item_type: looking up %?", did);
        // The item is in this crate. The caller should have added it to the
        // type cache already
        assert did.crate != ast::local_crate;
        let tyt = csearch::get_type(cx, did);
        cx.tcache.insert(did, tyt);
        ret tyt;
      }
    }
}

// Look up the list of item types for a given class
// Fails if the id is not bound to a class.
fn lookup_class_item_tys(cx: ctxt, did: ast::def_id) -> [@class_item_ty] {
  if did.crate == ast::local_crate {
    alt cx.items.find(did.node) {
       some(ast_map::node_item(i,_)) {
         alt i.node {
           ast::item_class(_, items, _) {
               class_item_tys(cx, items)
           }
           _ { cx.sess.bug("class ID bound to non-class"); }
         }
       }
       _ { cx.sess.bug("class ID not bound to an item"); }
    }
        }
  else {
        ret csearch::get_class_items(cx, did);
    }
}

// must be called after typechecking?
fn class_item_tys(cx: ctxt, items: [@class_item]) -> [@class_item_ty] {
    let rslt = [];
    for it in items {
       alt it.node.decl {
          instance_var(nm, _, _, id) {
              rslt += [@{ident: nm, id: id,
                        contents: var_ty(node_id_to_type(cx, id)) }];
          }
          class_method(it) {
              alt it.node {
                 item_fn(dec, _, _) {
                     rslt += [@{ident: it.ident, id: it.id,
                                 contents: method_ty(dec)}];
                 }
                 _ { fail; /* TODO */ }
               }
          }
       }
    }
    rslt
}

// Return a list of fields corresponding to the class's items
// (as if the class was a record). trans uses this
fn class_items_as_fields(cx:ctxt, did: ast::def_id) -> [field] {
    let rslt = [];
    for ci in lookup_class_item_tys(cx, did) {
       alt ci.contents {
          var_ty(t) {
             // consider all instance vars mutable, because the
             // constructor may mutate all vars
             rslt += [{ident: ci.ident, mt: {ty: t,
                             mutbl: m_mutbl}}];
          }
         /* do nothing, since methods don't have a runtime
          representation? */
          method_ty(_) {
          }
       }
    }
    rslt
}

fn is_binopable<T>(_cx: ctxt, ty: sty_base<T>, op: ast::binop) -> bool {
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
          ast::add { opcat_add }
          ast::subtract { opcat_sub }
          ast::mul { opcat_mult }
          ast::div { opcat_mult }
          ast::rem { opcat_mult }
          ast::and { opcat_logic }
          ast::or { opcat_logic }
          ast::bitxor { opcat_bit }
          ast::bitand { opcat_bit }
          ast::bitor { opcat_bit }
          ast::lsl { opcat_shift }
          ast::lsr { opcat_shift }
          ast::asr { opcat_shift }
          ast::eq { opcat_eq }
          ast::ne { opcat_eq }
          ast::lt { opcat_rel }
          ast::le { opcat_rel }
          ast::ge { opcat_rel }
          ast::gt { opcat_rel }
        }
    }

    fn tycat<T>(ty: sty_base<T>) -> int {
        alt ty {
          ty_bool { tycat_bool }
          ty_int(_) { tycat_int }
          ty_uint(_) { tycat_int }
          ty_float(_) { tycat_float }
          ty_str { tycat_str }
          ty_vec(_) { tycat_vec }
          ty_rec(_) { tycat_struct }
          ty_tup(_) { tycat_struct }
          ty_enum(_, _) { tycat_struct }
          ty_bot { tycat_bot }
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

    ret tbl[tycat(ty)][opcat(op)];
}

fn ast_constr_to_constr<T>(tcx: ctxt, c: @ast::constr_general<T>) ->
   @constr_general<T> {
    alt tcx.def_map.find(c.node.id) {
      some(ast::def_fn(pred_id, ast::pure_fn)) {
        ret @ast_util::respan(c.span,
                              {path: c.node.path,
                               args: c.node.args,
                               id: pred_id});
      }
      _ {
        tcx.sess.span_fatal(c.span,
                            "predicate " + path_to_str(c.node.path) +
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
