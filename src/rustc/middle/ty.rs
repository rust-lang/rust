import std::{ufind, map, smallintmap};
import result::result;
import std::map::hashmap;
import driver::session;
import session::session;
import syntax::ast;
import syntax::ast::*;
import syntax::ast_util;
import syntax::ast_util::{is_local, split_class_items};
import syntax::codemap::span;
import metadata::csearch;
import util::common::*;
import util::ppaux::region_to_str;
import util::ppaux::vstore_to_str;
import util::ppaux::{ty_to_str, tys_to_str, ty_constr_to_str};
import syntax::print::pprust::*;

export ty_vid, region_vid, vid;
export br_hashmap;
export is_instantiable;
export node_id_to_type;
export node_id_to_type_params;
export arg;
export args_eq;
export ast_constr_to_constr;
export block_ty;
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
export field_ty;
export fold_ty, fold_sty_to_ty, fold_region, fold_regions;
export fold_regions_and_ty, walk_regions_and_ty;
export field;
export field_idx;
export get_field;
export get_fields;
export get_element_type;
export is_binopable;
export is_pred_ty;
export lookup_class_field, lookup_class_fields;
export lookup_class_method_by_name;
export lookup_field_type;
export lookup_item_type;
export lookup_public_fields;
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
export subst, subst_tps, substs_is_noop, substs_to_str, substs;
export t;
export new_ty_hash;
export enum_variants, substd_enum_variants;
export iface_methods, store_iface_methods, impl_iface;
export enum_variant_with_id;
export ty_param_bounds_and_ty;
export ty_bool, mk_bool, type_is_bool;
export ty_bot, mk_bot, type_is_bot;
export ty_box, mk_box, mk_imm_box, type_is_box, type_is_boxed;
export ty_constr, mk_constr;
export ty_opaque_closure_ptr, mk_opaque_closure_ptr;
export ty_opaque_box, mk_opaque_box;
export ty_constr_arg;
export ty_float, mk_float, mk_mach_float, type_is_fp;
export ty_fn, fn_ty, mk_fn;
export ty_fn_proto, ty_fn_ret, ty_fn_ret_style;
export ty_int, mk_int, mk_mach_int, mk_char;
export ty_str, mk_str, type_is_str;
export ty_vec, mk_vec, type_is_vec;
export ty_estr, mk_estr;
export ty_evec, mk_evec;
export vstore, vstore_fixed, vstore_uniq, vstore_box, vstore_slice;
export ty_nil, mk_nil, type_is_nil;
export ty_iface, mk_iface;
export ty_res, mk_res;
export ty_param, mk_param, ty_params_to_tys;
export ty_ptr, mk_ptr, mk_mut_ptr, mk_imm_ptr, mk_nil_ptr, type_is_unsafe_ptr;
export ty_rptr, mk_rptr;
export ty_rec, mk_rec;
export ty_enum, mk_enum, type_is_enum;
export ty_tup, mk_tup;
export ty_type, mk_type;
export ty_uint, mk_uint, mk_mach_uint;
export ty_uniq, mk_uniq, mk_imm_uniq, type_is_unique_box;
export ty_var, mk_var, type_is_var;
export ty_self, mk_self, type_has_self;
export region, bound_region;
export get, type_has_params, type_needs_infer, type_has_regions;
export type_has_resources, type_id;
export tbox_has_flag;
export ty_var_id;
export ty_to_def_id;
export ty_fn_args;
export type_constr;
export kind, kind_sendable, kind_copyable, kind_noncopyable;
export kind_can_be_copied, kind_can_be_sent, proto_kind, kind_lteq, type_kind;
export type_err, terr_vstore_kind;
export type_err_to_str;
export type_needs_drop;
export type_allows_implicit_copy;
export type_is_integral;
export type_is_numeric;
export type_is_pod;
export type_is_scalar;
export type_is_immediate;
export type_is_sequence;
export type_is_signed;
export type_is_structural;
export type_is_copyable;
export type_is_slice;
export type_is_unique;
export type_is_c_like_enum;
export type_structurally_contains;
export type_structurally_contains_uniques;
export type_autoderef;
export type_param;
export type_needs_unwind_cleanup;
export canon_mode;
export resolved_mode;
export arg_mode;
export unify_mode;
export set_default_mode;
export variant_info;
export walk_ty, maybe_walk_ty;
export occurs_check;
export closure_kind;
export ck_block;
export ck_box;
export ck_uniq;
export param_bound, param_bounds, bound_copy, bound_send, bound_iface;
export param_bounds_to_kind;
export default_arg_mode_for_ty;
export item_path;
export item_path_str;
export ast_ty_to_ty_cache_entry;
export atttce_unresolved, atttce_resolved;
export mach_sty;
export ty_sort_str;
export normalize_ty;

// Data types

// Note: after typeck, you should use resolved_mode() to convert this mode
// into an rmode, which will take into account the results of mode inference.
type arg = {mode: ast::mode, ty: t};

type field = {ident: ast::ident, mt: mt};

type param_bounds = @[param_bound];

type method = {ident: ast::ident,
               tps: @[param_bounds],
               fty: fn_ty,
               purity: ast::purity,
               vis: ast::visibility};

type constr_table = hashmap<ast::node_id, [constr]>;

type mt = {ty: t, mutbl: ast::mutability};

enum vstore {
    vstore_fixed(uint),
    vstore_uniq,
    vstore_box,
    vstore_slice(region)
}

type field_ty = {
  ident: ident,
  id: def_id,
  vis: ast::visibility,
  mutability: ast::class_mutability
};

// Contains information needed to resolve types and (in the future) look up
// the types of AST nodes.
type creader_cache = hashmap<{cnum: int, pos: uint, len: uint}, t>;

type intern_key = {struct: sty, o_def_id: option<ast::def_id>};

enum ast_ty_to_ty_cache_entry {
    atttce_unresolved,  /* not resolved yet */
    atttce_resolved(t)  /* resolved to a type, irrespective of region */
}

type ctxt =
    @{interner: hashmap<intern_key, t_box>,
      mut next_id: uint,
      sess: session::session,
      def_map: resolve::def_map,
      region_map: @middle::region::region_map,

      // Stores the types for various nodes in the AST.  Note that this table
      // is not guaranteed to be populated until after typeck.  See
      // typeck::fn_ctxt for details.
      node_types: node_type_table,

      // Stores the type parameters which were substituted to obtain the type
      // of this node.  This only applies to nodes that refer to entities
      // parameterized by type parameters, such as generic fns, types, or
      // other items.
      node_type_substs: hashmap<node_id, [t]>,

      items: ast_map::map,
      freevars: freevars::freevar_map,
      tcache: type_cache,
      rcache: creader_cache,
      short_names_cache: hashmap<t, @str>,
      needs_drop_cache: hashmap<t, bool>,
      needs_unwind_cleanup_cache: hashmap<t, bool>,
      kind_cache: hashmap<t, kind>,
      ast_ty_to_ty_cache: hashmap<@ast::ty, ast_ty_to_ty_cache_entry>,
      enum_var_cache: hashmap<def_id, @[variant_info]>,
      iface_method_cache: hashmap<def_id, @[method]>,
      ty_param_bounds: hashmap<ast::node_id, param_bounds>,
      inferred_modes: hashmap<ast::node_id, ast::mode>,
      borrowings: hashmap<ast::node_id, ()>,
      normalized_cache: hashmap<t, t>};

enum tbox_flag {
    has_params = 1,
    has_self = 2,
    needs_infer = 4,
    has_regions = 8,
    has_resources = 16,

    // a meta-flag: subst may be required if the type has parameters, a self
    // type, or references bound regions
    needs_subst = 1 | 2 | 8
}

type t_box = @{struct: sty,
               id: uint,
               flags: uint,
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

fn tbox_has_flag(tb: t_box, flag: tbox_flag) -> bool {
    (tb.flags & (flag as uint)) != 0u
}
fn type_has_params(t: t) -> bool { tbox_has_flag(get(t), has_params) }
fn type_has_self(t: t) -> bool { tbox_has_flag(get(t), has_self) }
fn type_needs_infer(t: t) -> bool { tbox_has_flag(get(t), needs_infer) }
fn type_has_regions(t: t) -> bool { tbox_has_flag(get(t), has_regions) }
fn type_has_resources(t: t) -> bool { tbox_has_flag(get(t), has_resources) }
fn type_def_id(t: t) -> option<ast::def_id> { get(t).o_def_id }
fn type_id(t: t) -> uint { get(t).id }

enum closure_kind {
    ck_block,
    ck_box,
    ck_uniq,
}

type fn_ty = {proto: ast::proto,
              inputs: [arg],
              output: t,
              ret_style: ret_style,
              constraints: [@constr]};

// See discussion at head of region.rs
enum region {
    re_bound(bound_region),
    re_free(node_id, bound_region),
    re_scope(node_id),
    re_var(region_vid),
    re_static // effectively `top` in the region lattice
}

enum bound_region {
    br_self,      // The self region for classes, impls
    br_anon,      // The anonymous region parameter for a given function.
    br_named(str) // A named region parameter.
}

type opt_region = option<region>;

// The type substs represents the kinds of things that can be substituted into
// a type.  There may be at most one region parameter (self_r), along with
// some number of type parameters (tps).
//
// The region parameter is present on nominative types (enums, resources,
// classes) that are declared as having a region parameter.  If the type is
// declared as `enum foo&`, then self_r should always be non-none.  If the
// type is declared as `enum foo`, then self_r will always be none.  In the
// latter case, typeck::ast_ty_to_ty() will reject any references to `&T` or
// `&self.T` within the type and report an error.
type substs = {
    self_r: opt_region,
    self_ty: option<ty::t>,
    tps: [t]
};

// NB: If you change this, you'll probably want to change the corresponding
// AST structure in front/ast::rs as well.
enum sty {
    ty_nil,
    ty_bot,
    ty_bool,
    ty_int(ast::int_ty),
    ty_uint(ast::uint_ty),
    ty_float(ast::float_ty),
    ty_str,
    ty_estr(vstore),
    ty_enum(def_id, substs),
    ty_box(mt),
    ty_uniq(mt),
    ty_vec(mt),
    ty_evec(mt, vstore),
    ty_ptr(mt),
    ty_rptr(region, mt),
    ty_rec([field]),
    ty_fn(fn_ty),
    ty_iface(def_id, substs),
    ty_class(def_id, substs),
    ty_res(def_id, t, substs),
    ty_tup([t]),

    ty_var(ty_vid), // type variable during typechecking
    ty_param(uint, def_id), // type parameter
    ty_self, // special, implicit `self` type parameter

    ty_type, // type_desc*
    ty_opaque_box, // used by monomorphizer to represent any @ box
    ty_constr(t, [@type_constr]),
    ty_opaque_closure_ptr(closure_kind), // ptr to env for fn, fn@, fn~
}

// In the middle end, constraints have a def_id attached, referring
// to the definition of the operator in the constraint.
type constr_general<ARG> = spanned<constr_general_<ARG, def_id>>;
type type_constr = constr_general<@path>;
type constr = constr_general<uint>;

enum terr_vstore_kind {
    terr_vec, terr_str
}

// Data structures used in type unification
enum type_err {
    terr_mismatch,
    terr_ret_style_mismatch(ast::ret_style, ast::ret_style),
    terr_mutability,
    terr_proto_mismatch(ast::proto, ast::proto),
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
    terr_regions_differ(region, region),
    terr_vstores_differ(terr_vstore_kind, vstore, vstore),
    terr_in_field(@type_err, str),
    terr_sorts(t, t),
    terr_self_substs
}

enum param_bound {
    bound_copy,
    bound_send,
    bound_iface(t),
}

enum ty_vid = uint;
enum region_vid = uint;

iface vid {
    fn to_uint() -> uint;
    fn to_str() -> str;
}

impl of vid for ty_vid {
    fn to_uint() -> uint { *self }
    fn to_str() -> str { #fmt["<V%u>", self.to_uint()] }
}

impl of vid for region_vid {
    fn to_uint() -> uint { *self }
    fn to_str() -> str { #fmt["<R%u>", self.to_uint()] }
}

fn param_bounds_to_kind(bounds: param_bounds) -> kind {
    let mut kind = kind_noncopyable;
    for vec::each(*bounds) {|bound|
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

type ty_param_bounds_and_ty = {bounds: @[param_bounds],
                               rp: ast::region_param,
                               ty: t};

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
            option::map_default(k.o_def_id, 0u, ast_util::hash_def_id)
    }, {|&&a, &&b| a == b});
    @{interner: interner,
      mut next_id: 0u,
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
      needs_unwind_cleanup_cache: new_ty_hash(),
      kind_cache: new_ty_hash(),
      ast_ty_to_ty_cache: map::hashmap(
          ast_util::hash_ty, ast_util::eq_ty),
      enum_var_cache: new_def_hash(),
      iface_method_cache: new_def_hash(),
      ty_param_bounds: map::int_hash(),
      inferred_modes: map::int_hash(),
      borrowings: map::int_hash(),
      normalized_cache: new_ty_hash()}
}


// Type constructors
fn mk_t(cx: ctxt, st: sty) -> t { mk_t_with_id(cx, st, none) }

// Interns a type/name combination, stores the resulting box in cx.interner,
// and returns the box as cast to an unsafe ptr (see comments for t above).
fn mk_t_with_id(cx: ctxt, st: sty, o_def_id: option<ast::def_id>) -> t {
    let key = {struct: st, o_def_id: o_def_id};
    alt cx.interner.find(key) {
      some(t) { unsafe { ret unsafe::reinterpret_cast(t); } }
      _ {}
    }
    let mut flags = 0u;
    fn rflags(r: region) -> uint {
        (has_regions as uint) | {
            alt r {
              ty::re_var(_) {needs_infer as uint}
              _ {0u}
            }
        }
    }
    fn sflags(substs: substs) -> uint {
        let mut f = 0u;
        for substs.tps.each {|tt| f |= get(tt).flags; }
        substs.self_r.iter { |r| f |= rflags(r) }
        ret f;
    }
    alt st {
      ty_estr(vstore_slice(r)) {
        flags |= rflags(r);
      }
      ty_evec(mt, vstore_slice(r)) {
        flags |= rflags(r);
        flags |= get(mt.ty).flags;
      }
      ty_nil | ty_bot | ty_bool | ty_int(_) | ty_float(_) | ty_uint(_) |
      ty_str | ty_estr(_) | ty_type | ty_opaque_closure_ptr(_) |
      ty_opaque_box {}
      ty_param(_, _) { flags |= has_params as uint; }
      ty_var(_) { flags |= needs_infer as uint; }
      ty_self { flags |= has_self as uint; }
      ty_enum(_, substs) | ty_class(_, substs) | ty_iface(_, substs) {
        flags |= sflags(substs);
      }
      ty_box(m) | ty_uniq(m) | ty_vec(m) | ty_evec(m, _) | ty_ptr(m) {
        flags |= get(m.ty).flags;
      }
      ty_rptr(r, m) {
        flags |= rflags(r);
        flags |= get(m.ty).flags;
      }
      ty_rec(flds) {
        for flds.each {|f| flags |= get(f.mt.ty).flags; }
      }
      ty_tup(ts) {
        for ts.each {|tt| flags |= get(tt).flags; }
      }
      ty_fn(f) {
        for f.inputs.each {|a| flags |= get(a.ty).flags; }
        flags |= get(f.output).flags;
      }
      ty_res(_, tt, substs) {
        flags |= (has_resources as uint);
        flags |= get(tt).flags;
        flags |= sflags(substs);
      }
      ty_constr(tt, _) {
        flags |= get(tt).flags;
      }
    }
    let t = @{struct: st, id: cx.next_id, flags: flags, o_def_id: o_def_id};
    cx.interner.insert(key, t);
    cx.next_id += 1u;
    unsafe { unsafe::reinterpret_cast(t) }
}

fn mk_nil(cx: ctxt) -> t { mk_t(cx, ty_nil) }

fn mk_bot(cx: ctxt) -> t { mk_t(cx, ty_bot) }

fn mk_bool(cx: ctxt) -> t { mk_t(cx, ty_bool) }

fn mk_int(cx: ctxt) -> t { mk_t(cx, ty_int(ast::ty_i)) }

fn mk_float(cx: ctxt) -> t { mk_t(cx, ty_float(ast::ty_f)) }

fn mk_uint(cx: ctxt) -> t { mk_t(cx, ty_uint(ast::ty_u)) }

fn mk_mach_int(cx: ctxt, tm: ast::int_ty) -> t { mk_t(cx, ty_int(tm)) }

fn mk_mach_uint(cx: ctxt, tm: ast::uint_ty) -> t { mk_t(cx, ty_uint(tm)) }

fn mk_mach_float(cx: ctxt, tm: ast::float_ty) -> t { mk_t(cx, ty_float(tm)) }

fn mk_char(cx: ctxt) -> t { mk_t(cx, ty_int(ast::ty_char)) }

fn mk_str(cx: ctxt) -> t { mk_t(cx, ty_str) }

fn mk_estr(cx: ctxt, t: vstore) -> t {
    mk_t(cx, ty_estr(t))
}

fn mk_enum(cx: ctxt, did: ast::def_id, substs: substs) -> t {
    mk_t(cx, ty_enum(did, substs))
}

fn mk_box(cx: ctxt, tm: mt) -> t { mk_t(cx, ty_box(tm)) }

fn mk_imm_box(cx: ctxt, ty: t) -> t { mk_box(cx, {ty: ty,
                                                  mutbl: ast::m_imm}) }

fn mk_uniq(cx: ctxt, tm: mt) -> t { mk_t(cx, ty_uniq(tm)) }

fn mk_imm_uniq(cx: ctxt, ty: t) -> t { mk_uniq(cx, {ty: ty,
                                                    mutbl: ast::m_imm}) }

fn mk_ptr(cx: ctxt, tm: mt) -> t { mk_t(cx, ty_ptr(tm)) }

fn mk_rptr(cx: ctxt, r: region, tm: mt) -> t { mk_t(cx, ty_rptr(r, tm)) }

fn mk_mut_ptr(cx: ctxt, ty: t) -> t { mk_ptr(cx, {ty: ty,
                                                  mutbl: ast::m_mutbl}) }

fn mk_imm_ptr(cx: ctxt, ty: t) -> t {
    mk_ptr(cx, {ty: ty, mutbl: ast::m_imm})
}

fn mk_nil_ptr(cx: ctxt) -> t {
    mk_ptr(cx, {ty: mk_nil(cx), mutbl: ast::m_imm})
}

fn mk_vec(cx: ctxt, tm: mt) -> t { mk_t(cx, ty_vec(tm)) }

fn mk_evec(cx: ctxt, tm: mt, t: vstore) -> t {
    mk_t(cx, ty_evec(tm, t))
}

fn mk_rec(cx: ctxt, fs: [field]) -> t { mk_t(cx, ty_rec(fs)) }

fn mk_constr(cx: ctxt, t: t, cs: [@type_constr]) -> t {
    mk_t(cx, ty_constr(t, cs))
}

fn mk_tup(cx: ctxt, ts: [t]) -> t { mk_t(cx, ty_tup(ts)) }

fn mk_fn(cx: ctxt, fty: fn_ty) -> t { mk_t(cx, ty_fn(fty)) }

fn mk_iface(cx: ctxt, did: ast::def_id, substs: substs) -> t {
    mk_t(cx, ty_iface(did, substs))
}

fn mk_class(cx: ctxt, class_id: ast::def_id, substs: substs) -> t {
    mk_t(cx, ty_class(class_id, substs))
}

fn mk_res(cx: ctxt, did: ast::def_id,
          inner: t, substs: substs) -> t {
    mk_t(cx, ty_res(did, inner, substs))
}

fn mk_var(cx: ctxt, v: ty_vid) -> t { mk_t(cx, ty_var(v)) }

fn mk_self(cx: ctxt) -> t { mk_t(cx, ty_self) }

fn mk_param(cx: ctxt, n: uint, k: def_id) -> t { mk_t(cx, ty_param(n, k)) }

fn mk_type(cx: ctxt) -> t { mk_t(cx, ty_type) }

fn mk_opaque_closure_ptr(cx: ctxt, ck: closure_kind) -> t {
    mk_t(cx, ty_opaque_closure_ptr(ck))
}

fn mk_opaque_box(cx: ctxt) -> t { mk_t(cx, ty_opaque_box) }

fn mk_with_id(cx: ctxt, base: t, def_id: ast::def_id) -> t {
    mk_t_with_id(cx, get(base).struct, some(def_id))
}

// Converts s to its machine type equivalent
pure fn mach_sty(cfg: @session::config, t: t) -> sty {
    alt get(t).struct {
      ty_int(ast::ty_i) { ty_int(cfg.int_type) }
      ty_uint(ast::ty_u) { ty_uint(cfg.uint_type) }
      ty_float(ast::ty_f) { ty_float(cfg.float_type) }
      s { s }
    }
}

fn default_arg_mode_for_ty(ty: ty::t) -> ast::rmode {
    if ty::type_is_immediate(ty) { ast::by_val }
    else { ast::by_ref }
}

fn walk_ty(ty: t, f: fn(t)) {
    maybe_walk_ty(ty, {|t| f(t); true});
}

fn maybe_walk_ty(ty: t, f: fn(t) -> bool) {
    if !f(ty) { ret; }
    alt get(ty).struct {
      ty_nil | ty_bot | ty_bool | ty_int(_) | ty_uint(_) | ty_float(_) |
      ty_str | ty_estr(_) | ty_type | ty_opaque_box | ty_self |
      ty_opaque_closure_ptr(_) | ty_var(_) | ty_param(_, _) {
      }
      ty_box(tm) | ty_vec(tm) | ty_evec(tm, _) |
      ty_ptr(tm) | ty_rptr(_, tm) {
        maybe_walk_ty(tm.ty, f);
      }
      ty_enum(_, substs) | ty_class(_, substs) |
      ty_iface(_, substs) {
        for substs.tps.each {|subty| maybe_walk_ty(subty, f); }
      }
      ty_rec(fields) {
        for fields.each {|fl| maybe_walk_ty(fl.mt.ty, f); }
      }
      ty_tup(ts) { for ts.each {|tt| maybe_walk_ty(tt, f); } }
      ty_fn(ft) {
        for ft.inputs.each {|a| maybe_walk_ty(a.ty, f); }
        maybe_walk_ty(ft.output, f);
      }
      ty_res(_, sub, substs) {
        maybe_walk_ty(sub, f);
        for substs.tps.each {|tp| maybe_walk_ty(tp, f); }
      }
      ty_constr(sub, _) { maybe_walk_ty(sub, f); }
      ty_uniq(tm) { maybe_walk_ty(tm.ty, f); }
    }
}

fn fold_sty_to_ty(tcx: ty::ctxt, sty: sty, foldop: fn(t) -> t) -> t {
    mk_t(tcx, fold_sty(sty, foldop))
}

fn fold_sty(sty: sty, fldop: fn(t) -> t) -> sty {
    fn fold_substs(substs: substs, fldop: fn(t) -> t) -> substs {
        {self_r: substs.self_r,
         self_ty: substs.self_ty.map { |t| fldop(t) },
         tps: substs.tps.map { |t| fldop(t) }}
    }

    alt sty {
      ty_box(tm) {
        ty_box({ty: fldop(tm.ty), mutbl: tm.mutbl})
      }
      ty_uniq(tm) {
        ty_uniq({ty: fldop(tm.ty), mutbl: tm.mutbl})
      }
      ty_ptr(tm) {
        ty_ptr({ty: fldop(tm.ty), mutbl: tm.mutbl})
      }
      ty_vec(tm) {
        ty_vec({ty: fldop(tm.ty), mutbl: tm.mutbl})
      }
      ty_evec(tm, vst) {
        ty_evec({ty: fldop(tm.ty), mutbl: tm.mutbl}, vst)
      }
      ty_enum(tid, substs) {
        ty_enum(tid, fold_substs(substs, fldop))
      }
      ty_iface(did, substs) {
        ty_iface(did, fold_substs(substs, fldop))
      }
      ty_rec(fields) {
        let new_fields = vec::map(fields) {|fl|
            let new_ty = fldop(fl.mt.ty);
            let new_mt = {ty: new_ty, mutbl: fl.mt.mutbl};
            {ident: fl.ident, mt: new_mt}
        };
        ty_rec(new_fields)
      }
      ty_tup(ts) {
        let new_ts = vec::map(ts) {|tt| fldop(tt) };
        ty_tup(new_ts)
      }
      ty_fn(f) {
        let new_args = vec::map(f.inputs) {|a|
            let new_ty = fldop(a.ty);
            {mode: a.mode, ty: new_ty}
        };
        let new_output = fldop(f.output);
        ty_fn({inputs: new_args, output: new_output with f})
      }
      ty_res(did, subty, substs) {
        ty_res(did, fldop(subty),
               fold_substs(substs, fldop))
      }
      ty_rptr(r, tm) {
        ty_rptr(r, {ty: fldop(tm.ty), mutbl: tm.mutbl})
      }
      ty_constr(subty, cs) {
        ty_constr(fldop(subty), cs)
      }
      ty_class(did, substs) {
        ty_class(did, fold_substs(substs, fldop))
      }
      ty_nil | ty_bot | ty_bool | ty_int(_) | ty_uint(_) | ty_float(_) |
      ty_str | ty_estr(_) | ty_type | ty_opaque_closure_ptr(_) |
      ty_opaque_box | ty_var(_) | ty_param(*) | ty_self {
        sty
      }
    }
}

// Folds types from the bottom up.
fn fold_ty(cx: ctxt, t0: t, fldop: fn(t) -> t) -> t {
    let sty = fold_sty(get(t0).struct) {|t| fold_ty(cx, t, fldop) };
    fldop(mk_t(cx, sty))
}

fn walk_regions_and_ty(
    cx: ctxt,
    ty: t,
    walkr: fn(r: region),
    walkt: fn(t: t) -> bool) {

    if (walkt(ty)) {
        fold_regions_and_ty(
            cx, ty,
            { |r| walkr(r); r },
            { |t| walkt(t); walk_regions_and_ty(cx, t, walkr, walkt); t },
            { |t| walkt(t); walk_regions_and_ty(cx, t, walkr, walkt); t });
    }
}

fn fold_regions_and_ty(
    cx: ctxt,
    ty: t,
    fldr: fn(r: region) -> region,
    fldfnt: fn(t: t) -> t,
    fldt: fn(t: t) -> t) -> t {

    fn fold_substs(
        substs: substs,
        fldr: fn(r: region) -> region,
        fldt: fn(t: t) -> t) -> substs {

        {self_r: substs.self_r.map { |r| fldr(r) },
         self_ty: substs.self_ty.map { |t| fldt(t) },
         tps: substs.tps.map { |t| fldt(t) }}
    }

    let tb = ty::get(ty);
    alt tb.struct {
      ty::ty_rptr(r, mt) {
        let m_r = fldr(r);
        let m_t = fldt(mt.ty);
        ty::mk_rptr(cx, m_r, {ty: m_t, mutbl: mt.mutbl})
      }
      ty_estr(vstore_slice(r)) {
        let m_r = fldr(r);
        ty::mk_estr(cx, vstore_slice(m_r))
      }
      ty_evec(mt, vstore_slice(r)) {
        let m_r = fldr(r);
        let m_t = fldt(mt.ty);
        ty::mk_evec(cx, {ty: m_t, mutbl: mt.mutbl}, vstore_slice(m_r))
      }
      ty_enum(def_id, substs) {
        ty::mk_enum(cx, def_id, fold_substs(substs, fldr, fldt))
      }
      ty_class(def_id, substs) {
        ty::mk_class(cx, def_id, fold_substs(substs, fldr, fldt))
      }
      ty_iface(def_id, substs) {
        ty::mk_iface(cx, def_id, fold_substs(substs, fldr, fldt))
      }
      ty_res(def_id, t, substs) {
        ty::mk_res(cx, def_id, fldt(t),
                   fold_substs(substs, fldr, fldt))
      }
      sty @ ty_fn(_) {
        fold_sty_to_ty(cx, sty) {|t|
            fldfnt(t)
        }
      }
      sty {
        fold_sty_to_ty(cx, sty) {|t|
            fldt(t)
        }
      }
    }
}

// n.b. this function is intended to eventually replace fold_region() below,
// that is why its name is so similar.
fn fold_regions(
    cx: ctxt,
    ty: t,
    fldr: fn(r: region, in_fn: bool) -> region) -> t {

    fn do_fold(cx: ctxt, ty: t, in_fn: bool,
               fldr: fn(region, bool) -> region) -> t {
        if !type_has_regions(ty) { ret ty; }
        fold_regions_and_ty(
            cx, ty,
            { |r| fldr(r, in_fn) },
            { |t| do_fold(cx, t, true, fldr) },
            { |t| do_fold(cx, t, in_fn, fldr) })
    }
    do_fold(cx, ty, false, fldr)
}

fn fold_region(cx: ctxt, t0: t, fldop: fn(region, bool) -> region) -> t {
    fn do_fold(cx: ctxt, t0: t, under_r: bool,
               fldop: fn(region, bool) -> region) -> t {
        let tb = get(t0);
        if !tbox_has_flag(tb, has_regions) { ret t0; }
        alt tb.struct {
          ty_rptr(r, {ty: t1, mutbl: m}) {
            let m_r = fldop(r, under_r);
            let m_t1 = do_fold(cx, t1, true, fldop);
            ty::mk_rptr(cx, m_r, {ty: m_t1, mutbl: m})
          }
          ty_estr(vstore_slice(r)) {
            let m_r = fldop(r, under_r);
            ty::mk_estr(cx, vstore_slice(m_r))
          }
          ty_evec({ty: t1, mutbl: m}, vstore_slice(r)) {
            let m_r = fldop(r, under_r);
            let m_t1 = do_fold(cx, t1, true, fldop);
            ty::mk_evec(cx, {ty: m_t1, mutbl: m}, vstore_slice(m_r))
          }
          ty_fn(_) {
            // do not recurse into functions, which introduce fresh bindings
            t0
          }
          sty {
            fold_sty_to_ty(cx, sty) {|t|
                do_fold(cx, t, under_r, fldop)
            }
          }
      }
    }

    do_fold(cx, t0, false, fldop)
}

// Substitute *only* type parameters.  Used in trans where regions are erased.
fn subst_tps(cx: ctxt, tps: [t], typ: t) -> t {
    if tps.len() == 0u { ret typ; }
    let tb = ty::get(typ);
    if !tbox_has_flag(tb, has_params) { ret typ; }
    alt tb.struct {
      ty_param(idx, _) { tps[idx] }
      sty { fold_sty_to_ty(cx, sty) {|t| subst_tps(cx, tps, t) } }
    }
}

fn substs_is_noop(substs: substs) -> bool {
    substs.tps.len() == 0u &&
        substs.self_r.is_none() &&
        substs.self_ty.is_none()
}

fn substs_to_str(cx: ctxt, substs: substs) -> str {
    #fmt["substs(self_r=%s, self_ty=%s, tps=%?)",
         substs.self_r.map_default("none", { |r| region_to_str(cx, r) }),
         substs.self_ty.map_default("none", { |t| ty_to_str(cx, t) }),
         substs.tps.map { |t| ty_to_str(cx, t) }]
}

fn subst(cx: ctxt,
         substs: substs,
         typ: t) -> t {

    #debug["subst(substs=%s, typ=%s)",
           substs_to_str(cx, substs),
           ty_to_str(cx, typ)];

    if substs_is_noop(substs) { ret typ; }
    let r = do_subst(cx, substs, typ);
    #debug["  r = %s", ty_to_str(cx, r)];
    ret r;

    fn do_subst(cx: ctxt,
                substs: substs,
                typ: t) -> t {
        let tb = get(typ);
        if !tbox_has_flag(tb, needs_subst) { ret typ; }
        alt tb.struct {
          ty_param(idx, _) {substs.tps[idx]}
          ty_self {substs.self_ty.get()}
          _ {
            fold_regions_and_ty(
                cx, typ,
                { |r|
                    alt r {
                      re_bound(br_self) {substs.self_r.get()}
                      _ {r}
                    }
                },
                { |t| do_subst(cx, substs, t) },
                { |t| do_subst(cx, substs, t) })
          }
        }
    }
}

// Type utilities

fn type_is_nil(ty: t) -> bool { get(ty).struct == ty_nil }

fn type_is_bot(ty: t) -> bool { get(ty).struct == ty_bot }

fn type_is_var(ty: t) -> bool {
    alt get(ty).struct {
      ty_var(_) { true }
      _ { false }
    }
}

fn type_is_bool(ty: t) -> bool { get(ty).struct == ty_bool }

fn type_is_structural(ty: t) -> bool {
    alt get(ty).struct {
      ty_rec(_) | ty_class(_, _) | ty_tup(_) | ty_enum(_, _) | ty_fn(_) |
      ty_iface(_, _) | ty_res(_, _, _) | ty_evec(_, vstore_fixed(_))
      | ty_estr(vstore_fixed(_)) { true }
      _ { false }
    }
}

fn type_is_copyable(cx: ctxt, ty: t) -> bool {
    ret kind_can_be_copied(type_kind(cx, ty));
}

fn type_is_sequence(ty: t) -> bool {
    alt get(ty).struct {
      ty_str | ty_estr(_) | ty_vec(_) | ty_evec(_, _) { true }
      _ { false }
    }
}

fn type_is_str(ty: t) -> bool {
    alt get(ty).struct {
      ty_str | ty_estr(_) { true }
      _ { false }
    }
}

fn sequence_element_type(cx: ctxt, ty: t) -> t {
    alt get(ty).struct {
      ty_str | ty_estr(_) { ret mk_mach_uint(cx, ast::ty_u8); }
      ty_vec(mt) | ty_evec(mt, _) { ret mt.ty; }
      _ { cx.sess.bug("sequence_element_type called on non-sequence value"); }
    }
}

fn get_element_type(ty: t, i: uint) -> t {
    alt get(ty).struct {
      ty_rec(flds) { ret flds[i].mt.ty; }
      ty_tup(ts) { ret ts[i]; }
      _ { fail "get_element_type called on invalid type"; }
    }
}

pure fn type_is_box(ty: t) -> bool {
    alt get(ty).struct {
      ty_box(_) { ret true; }
      _ { ret false; }
    }
}

pure fn type_is_boxed(ty: t) -> bool {
    alt get(ty).struct {
      ty_box(_) | ty_opaque_box { true }
      _ { false }
    }
}

pure fn type_is_region_ptr(ty: t) -> bool {
    alt get(ty).struct {
      ty_rptr(_, _) { true }
      _ { false }
    }
}

pure fn type_is_slice(ty: t) -> bool {
    alt get(ty).struct {
      ty_evec(_, vstore_slice(_)) | ty_estr(vstore_slice(_)) { true }
      _ { ret false; }
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
          ty_vec(_) | ty_evec(_, _) { true }
          ty_str | ty_estr(_) { true }
          _ { false }
        };
}

pure fn type_is_unique(ty: t) -> bool {
    alt get(ty).struct {
      ty_uniq(_) { ret true; }
      ty_vec(_) | ty_evec(_, vstore_uniq) { true }
      ty_str | ty_estr(vstore_uniq) { true }
      _ { ret false; }
    }
}

pure fn type_is_scalar(ty: t) -> bool {
    alt get(ty).struct {
      ty_nil | ty_bool | ty_int(_) | ty_float(_) | ty_uint(_) |
      ty_type | ty_ptr(_) | ty_rptr(_, _) { true }
      _ { false }
    }
}

// FIXME maybe inline this for speed?
fn type_is_immediate(ty: t) -> bool {
    ret type_is_scalar(ty) || type_is_boxed(ty) ||
        type_is_unique(ty) || type_is_region_ptr(ty);
}

fn type_needs_drop(cx: ctxt, ty: t) -> bool {
    alt cx.needs_drop_cache.find(ty) {
      some(result) { ret result; }
      none {/* fall through */ }
    }

    let mut accum = false;
    let result = alt get(ty).struct {
      // scalar types
      ty_nil | ty_bot | ty_bool | ty_int(_) | ty_float(_) | ty_uint(_) |
      ty_type | ty_ptr(_) | ty_rptr(_, _) |
      ty_estr(vstore_fixed(_)) | ty_estr(vstore_slice(_)) |
      ty_evec(_, vstore_slice(_)) { false }
      ty_evec(mt, vstore_fixed(_)) { type_needs_drop(cx, mt.ty) }
      ty_rec(flds) {
        for flds.each {|f| if type_needs_drop(cx, f.mt.ty) { accum = true; } }
        accum
      }
      ty_class(did, substs) {
        for vec::each(ty::class_items_as_fields(cx, did, substs)) {|f|
            if type_needs_drop(cx, f.mt.ty) { accum = true; }
        }
        accum
      }

      ty_tup(elts) {
        for elts.each {|m| if type_needs_drop(cx, m) { accum = true; } }
        accum
      }
      ty_enum(did, substs) {
        let variants = enum_variants(cx, did);
        for vec::each(*variants) {|variant|
            for variant.args.each {|aty|
                // Perform any type parameter substitutions.
                let arg_ty = subst(cx, substs, aty);
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

// Some things don't need cleanups during unwinding because the
// task can free them all at once later. Currently only things
// that only contain scalars and shared boxes can avoid unwind
// cleanups.
fn type_needs_unwind_cleanup(cx: ctxt, ty: t) -> bool {
    alt cx.needs_unwind_cleanup_cache.find(ty) {
      some(result) { ret result; }
      none { }
    }

    let tycache = new_ty_hash();
    let needs_unwind_cleanup =
        type_needs_unwind_cleanup_(cx, ty, tycache, false);
    cx.needs_unwind_cleanup_cache.insert(ty, needs_unwind_cleanup);
    ret needs_unwind_cleanup;
}

fn type_needs_unwind_cleanup_(cx: ctxt, ty: t,
                              tycache: map::hashmap<t, ()>,
                              encountered_box: bool) -> bool {

    // Prevent infinite recursion
    alt tycache.find(ty) {
      some(_) { ret false; }
      none { tycache.insert(ty, ()); }
    }

    let mut encountered_box = encountered_box;
    let mut needs_unwind_cleanup = false;
    maybe_walk_ty(ty) {|ty|
        let old_encountered_box = encountered_box;
        let result = alt get(ty).struct {
          ty_box(_) | ty_opaque_box {
            encountered_box = true;
            true
          }
          ty_nil | ty_bot | ty_bool |
          ty_int(_) | ty_uint(_) | ty_float(_) |
          ty_rec(_) | ty_tup(_) | ty_ptr(_) {
            true
          }
          ty_enum(did, substs) {
            for vec::each(*enum_variants(cx, did)) {|v|
                for v.args.each {|aty|
                    let t = subst(cx, substs, aty);
                    needs_unwind_cleanup |=
                        type_needs_unwind_cleanup_(cx, t, tycache,
                                                   encountered_box);
                }
            }
            !needs_unwind_cleanup
          }
          ty_uniq(_) | ty_str | ty_vec(_) | ty_res(_, _, _) |
          ty_estr(vstore_uniq) |
          ty_estr(vstore_box) |
          ty_evec(_, vstore_uniq) |
          ty_evec(_, vstore_box)
          {
            // Once we're inside a box, the annihilator will find
            // it and destroy it.
            if !encountered_box {
                needs_unwind_cleanup = true;
                false
            } else {
                true
            }
          }
          _ {
            needs_unwind_cleanup = true;
            false
          }
        };

        encountered_box = old_encountered_box;
        result
    }

    ret needs_unwind_cleanup;
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

#[test]
fn test_kinds() {
    // The kind "lattice" is nocopy <= copy <= send
    assert kind_lteq(kind_sendable, kind_sendable);
    assert kind_lteq(kind_copyable, kind_sendable);
    assert kind_lteq(kind_copyable, kind_copyable);
    assert kind_lteq(kind_noncopyable, kind_sendable);
    assert kind_lteq(kind_noncopyable, kind_copyable);
    assert kind_lteq(kind_noncopyable, kind_noncopyable);
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

      // Closures have kind determined by capture mode
      ty_opaque_closure_ptr(ck_block) { kind_noncopyable }
      ty_opaque_closure_ptr(ck_box) { kind_copyable }
      ty_opaque_closure_ptr(ck_uniq) { kind_sendable }

      // Those with refcounts raise noncopyable to copyable,
      // lower sendable to copyable. Therefore just set result to copyable.
      ty_box(_) | ty_iface(_, _) | ty_opaque_box { kind_copyable }
      ty_rptr(_, _) { kind_copyable }

      // Unique boxes and vecs have the kind of their contained type.
      ty_vec(tm) | ty_uniq(tm) { type_kind(cx, tm.ty) }

      // Slice and refcounted evecs are copyable; uniques and interiors
      // depend on the their contained type.
      ty_evec(_, vstore_box) |
      ty_evec(_, vstore_slice(_)) { kind_copyable }
      ty_evec(tm, vstore_uniq) |
      ty_evec(tm, vstore_fixed(_)) { type_kind(cx, tm.ty)  }

      // All estrs are copyable; uniques and interiors are sendable.
      ty_estr(vstore_box) |
      ty_estr(vstore_slice(_)) { kind_copyable }
      ty_estr(vstore_uniq) |
      ty_estr(vstore_fixed(_)) { kind_sendable  }

      // Records lower to the lowest of their members.
      ty_rec(flds) {
        let mut lowest = kind_sendable;
        for flds.each {|f|
            lowest = lower_kind(lowest, type_kind(cx, f.mt.ty));
        }
        lowest
      }
      // FIXME: (tjc) there are rules about when classes are copyable/
      // sendable, but I'm just treating them like records (#1726)
      ty_class(did, substs) {
          // also factor out this code, copied from the records case
          let mut lowest = kind_sendable;
          let flds = class_items_as_fields(cx, did, substs);
          for flds.each {|f|
            lowest = lower_kind(lowest, type_kind(cx, f.mt.ty));
          }
          lowest
      }
      // Tuples lower to the lowest of their members.
      ty_tup(tys) {
        let mut lowest = kind_sendable;
        for tys.each {|ty| lowest = lower_kind(lowest, type_kind(cx, ty)); }
        lowest
      }
      // Enums lower to the lowest of their variants.
      ty_enum(did, substs) {
        let mut lowest = kind_sendable;
        let variants = enum_variants(cx, did);
        if vec::len(*variants) == 0u {
            lowest = kind_noncopyable;
        } else {
            for vec::each(*variants) {|variant|
                for variant.args.each {|aty|
                    // Perform any type parameter substitutions.
                    let arg_ty = subst(cx, substs, aty);
                    lowest = lower_kind(lowest, type_kind(cx, arg_ty));
                    if lowest == kind_noncopyable { break; }
                }
            }
        }
        lowest
      }
      ty_res(did, inner, tps) { kind_noncopyable }
      ty_param(_, did) {
          param_bounds_to_kind(cx.ty_param_bounds.get(did.node))
      }
      ty_constr(t, _) { type_kind(cx, t) }
      ty_self { kind_noncopyable }

      ty_var(_) { cx.sess.bug("Asked to compute kind of a type variable"); }
    };

    cx.kind_cache.insert(ty, result);
    ret result;
}

// True if instantiating an instance of `ty` requires an instead of `r_ty`.
fn is_instantiable(cx: ctxt, r_ty: t) -> bool {

    fn type_requires(cx: ctxt, seen: @mut [def_id],
                     r_ty: t, ty: t) -> bool {
        #debug["type_requires(%s, %s)?",
               ty_to_str(cx, r_ty),
               ty_to_str(cx, ty)];

        let r = {
            get(r_ty).struct == get(ty).struct ||
                subtypes_require(cx, seen, r_ty, ty)
        };

        #debug["type_requires(%s, %s)? %b",
               ty_to_str(cx, r_ty),
               ty_to_str(cx, ty),
               r];
        ret r;
    }

    fn subtypes_require(cx: ctxt, seen: @mut [def_id],
                        r_ty: t, ty: t) -> bool {
        #debug["subtypes_require(%s, %s)?",
               ty_to_str(cx, r_ty),
               ty_to_str(cx, ty)];

        let r = alt get(ty).struct {
          ty_nil |
          ty_bot |
          ty_bool |
          ty_int(_) |
          ty_uint(_) |
          ty_float(_) |
          ty_str |
          ty_estr(_) |
          ty_fn(_) |
          ty_var(_) |
          ty_param(_, _) |
          ty_self |
          ty_type |
          ty_opaque_box |
          ty_opaque_closure_ptr(_) |
          ty_evec(_, _) |
          ty_vec(_) {
            false
          }

          ty_constr(t, _) {
            type_requires(cx, seen, r_ty, t)
          }

          ty_box(mt) |
          ty_uniq(mt) |
          ty_rptr(_, mt) {
            be type_requires(cx, seen, r_ty, mt.ty);
          }

          ty_ptr(mt) {
            false           // unsafe ptrs can always be NULL
          }

          ty_rec(fields) {
            vec::any(fields) {|field|
                type_requires(cx, seen, r_ty, field.mt.ty)
            }
          }

          ty_iface(_, _) {
            false
          }

          ty_class(did, _) if vec::contains(*seen, did) {
            false
          }

          ty_class(did, substs) {
            vec::push(*seen, did);
            let r = vec::any(lookup_class_fields(cx, did)) {|f|
                let fty = ty::lookup_item_type(cx, f.id);
                let sty = subst(cx, substs, fty.ty);
                type_requires(cx, seen, r_ty, sty)
            };
            vec::pop(*seen);
            r
          }

          ty_res(did, _, _) if vec::contains(*seen, did) {
            false
          }

          ty_res(did, sub, substs) {
            vec::push(*seen, did);
            let sty = subst(cx, substs, sub);
            let r = type_requires(cx, seen, r_ty, sty);
            vec::pop(*seen);
            r
          }

          ty_tup(ts) {
            vec::any(ts) {|t|
                type_requires(cx, seen, r_ty, t)
            }
          }

          ty_enum(did, _) if vec::contains(*seen, did) {
            false
          }

          ty_enum(did, substs) {
            vec::push(*seen, did);
            let vs = enum_variants(cx, did);
            let r = vec::len(*vs) > 0u && vec::all(*vs) {|variant|
                vec::any(variant.args) {|aty|
                    let sty = subst(cx, substs, aty);
                    type_requires(cx, seen, r_ty, sty)
                }
            };
            vec::pop(*seen);
            r
          }
        };

        #debug["subtypes_require(%s, %s)? %b",
               ty_to_str(cx, r_ty),
               ty_to_str(cx, ty),
               r];

        ret r;
    }

    let seen = @mut [];
    !subtypes_require(cx, seen, r_ty, r_ty)
}

fn type_structurally_contains(cx: ctxt, ty: t, test: fn(sty) -> bool) ->
   bool {
    let sty = get(ty).struct;
    if test(sty) { ret true; }
    alt sty {
      ty_enum(did, substs) {
        for vec::each(*enum_variants(cx, did)) {|variant|
            for variant.args.each {|aty|
                let sty = subst(cx, substs, aty);
                if type_structurally_contains(cx, sty, test) { ret true; }
            }
        }
        ret false;
      }
      ty_rec(fields) {
        for fields.each {|field|
            if type_structurally_contains(cx, field.mt.ty, test) { ret true; }
        }
        ret false;
      }
      ty_tup(ts) {
        for ts.each {|tt|
            if type_structurally_contains(cx, tt, test) { ret true; }
        }
        ret false;
      }
      ty_res(_, sub, substs) {
        let sty = subst(cx, substs, sub);
        ret type_structurally_contains(cx, sty, test);
      }
      ty_evec(mt, vstore_fixed(_)) {
        ret type_structurally_contains(cx, mt.ty, test);
      }
      _ { ret false; }
    }
}

// Returns true for noncopyable types and types where a copy of a value can be
// distinguished from the value itself. I.e. types with mut content that's
// not shared through a pointer.
fn type_allows_implicit_copy(cx: ctxt, ty: t) -> bool {
    ret !type_structurally_contains(cx, ty, {|sty|
        alt sty {
          ty_param(_, _) { true }

          ty_evec(_, _) | ty_estr(_) {
            cx.sess.unimpl("estr/evec in type_allows_implicit_copy");
          }

          ty_vec(mt) {
            mt.mutbl != ast::m_imm
          }
          ty_rec(fields) {
            vec::any(fields, {|f| f.mt.mutbl != ast::m_imm})
          }
          _ { false }
        }
    }) && type_kind(cx, ty) != kind_noncopyable;
}

fn type_structurally_contains_uniques(cx: ctxt, ty: t) -> bool {
    ret type_structurally_contains(cx, ty, {|sty|
        alt sty {
          ty_uniq(_) |
          ty_vec(_) |
          ty_evec(_, vstore_uniq) |
          ty_str |
          ty_estr(vstore_uniq) { true }
          _ { false }
        }
    });
}

fn type_is_integral(ty: t) -> bool {
    alt get(ty).struct {
      ty_int(_) | ty_uint(_) | ty_bool { true }
      _ { false }
    }
}

fn type_is_fp(ty: t) -> bool {
    alt get(ty).struct {
      ty_float(_) { true }
      _ { false }
    }
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
    let mut result = true;
    alt get(ty).struct {
      // Scalar types
      ty_nil | ty_bot | ty_bool | ty_int(_) | ty_float(_) | ty_uint(_) |
      ty_type | ty_ptr(_) { result = true; }
      // Boxed types
      ty_str | ty_box(_) | ty_uniq(_) | ty_vec(_) | ty_fn(_) |
      ty_iface(_, _) | ty_rptr(_,_) | ty_opaque_box { result = false; }
      // Structural types
      ty_enum(did, substs) {
        let variants = enum_variants(cx, did);
        for vec::each(*variants) {|variant|
            let tup_ty = mk_tup(cx, variant.args);

            // Perform any type parameter substitutions.
            let tup_ty = subst(cx, substs, tup_ty);
            if !type_is_pod(cx, tup_ty) { result = false; }
        }
      }
      ty_rec(flds) {
        for flds.each {|f|
            if !type_is_pod(cx, f.mt.ty) { result = false; }
        }
      }
      ty_tup(elts) {
        for elts.each {|elt| if !type_is_pod(cx, elt) { result = false; } }
      }
      ty_estr(vstore_fixed(_)) { result = true; }
      ty_evec(mt, vstore_fixed(_)) {
        result = type_is_pod(cx, mt.ty);
      }
      ty_res(_, inner, substs) {
        result = type_is_pod(cx, subst(cx, substs, inner));
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

// Whether a type is enum like, that is a enum type with only nullary
// constructors
fn type_is_c_like_enum(cx: ctxt, ty: t) -> bool {
    alt get(ty).struct {
      ty_enum(did, substs) {
        let variants = enum_variants(cx, did);
        let some_n_ary = vec::any(*variants, {|v| vec::len(v.args) > 0u});
        ret !some_n_ary;
      }
      _ { ret false;}
    }
}

fn type_param(ty: t) -> option<uint> {
    alt get(ty).struct {
      ty_param(id, _) { ret some(id); }
      _ {/* fall through */ }
    }
    ret none;
}

// Returns a vec of all the type variables
// occurring in t. It may contain duplicates.
fn vars_in_type(ty: t) -> [ty_vid] {
    let mut rslt = [];
    walk_ty(ty) {|ty|
        alt get(ty).struct { ty_var(v) { rslt += [v]; } _ { } }
    }
    rslt
}

fn type_autoderef(cx: ctxt, t: t) -> t {
    let mut t1 = t;
    loop {
        alt get(t1).struct {
          ty_box(mt) | ty_uniq(mt) | ty::ty_rptr(_, mt) { t1 = mt.ty; }
          ty_res(_, inner, substs) {
            t1 = subst(cx, substs, inner);
          }
          ty_enum(did, substs) {
            let variants = enum_variants(cx, did);
            if vec::len(*variants) != 1u || vec::len(variants[0].args) != 1u {
                break;
            }
            t1 = subst(cx, substs, variants[0].args[0]);
          }
          _ { break; }
        }
    }
    ret t1;
}

fn hash_bound_region(br: bound_region) -> uint {
    alt br { // no idea if this is any good
      ty::br_self { 0u }
      ty::br_anon { 1u }
      ty::br_named(str) { str::hash(str) }
    }
}

fn br_hashmap<V:copy>() -> hashmap<bound_region, V> {
    map::hashmap(hash_bound_region,
                 {|&&a: bound_region, &&b: bound_region| a == b })
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
        let mut h = id;
        for subtys.each {|s| h = (h << 2u) + type_id(s) }
        h
    }
    fn hash_type_constr(id: uint, c: @type_constr) -> uint {
        let mut h = id;
        h = (h << 2u) + hash_def(h, c.node.id);
        // FIXME this makes little sense
        for c.node.args.each {|a|
            alt a.node {
              carg_base { h += h << 2u; }
              carg_lit(_) { fail "lit args not implemented yet"; }
              carg_ident(p) { h += h << 2u; }
            }
        }
        h
    }
    fn hash_region(r: region) -> uint {
        alt r { // no idea if this is any good
          re_bound(br) { (hash_bound_region(br)) << 2u | 0u }
          re_free(id, br) { ((id as uint) << 4u) |
                               (hash_bound_region(br)) << 2u | 1u }
          re_scope(id)  { ((id as uint) << 2u) | 2u }
          re_var(id)    { (id.to_uint() << 2u) | 3u }
          re_bot        { 4u }
        }
    }
    fn hash_substs(h: uint, substs: substs) -> uint {
        let h = hash_subtys(h, substs.tps);
        h + substs.self_r.map_default(0u, hash_region)
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
      ty_estr(_) { 16u }
      ty_str { 17u }
      ty_enum(did, substs) {
        let mut h = hash_def(18u, did);
        hash_substs(h, substs)
      }
      ty_box(mt) { hash_subty(19u, mt.ty) }
      ty_evec(mt, _) { hash_subty(20u, mt.ty) }
      ty_vec(mt) { hash_subty(21u, mt.ty) }
      ty_rec(fields) {
        let mut h = 26u;
        for fields.each {|f| h = hash_subty(h, f.mt.ty); }
        h
      }
      ty_tup(ts) { hash_subtys(25u, ts) }
      ty_fn(f) {
        let mut h = 27u;
        for f.inputs.each {|a| h = hash_subty(h, a.ty); }
        hash_subty(h, f.output)
      }
      ty_var(v) { hash_uint(30u, v.to_uint()) }
      ty_param(pid, did) { hash_def(hash_uint(31u, pid), did) }
      ty_self { 28u }
      ty_type { 32u }
      ty_bot { 34u }
      ty_ptr(mt) { hash_subty(35u, mt.ty) }
      ty_rptr(region, mt) {
        let mut h = (46u << 2u) + hash_region(region);
        hash_subty(h, mt.ty)
      }
      ty_res(did, sub, substs) {
        let mut h = hash_subty(hash_def(18u, did), sub);
        hash_substs(h, substs)
      }
      ty_constr(t, cs) {
        let mut h = hash_subty(36u, t);
        for cs.each {|c| h = (h << 2u) + hash_type_constr(h, c); }
        h
      }
      ty_uniq(mt) { hash_subty(37u, mt.ty) }
      ty_iface(did, substs) {
        let mut h = hash_def(40u, did);
        hash_substs(h, substs)
      }
      ty_opaque_closure_ptr(ck_block) { 41u }
      ty_opaque_closure_ptr(ck_box) { 42u }
      ty_opaque_closure_ptr(ck_uniq) { 43u }
      ty_opaque_box { 44u }
      ty_class(did, substs) {
        let mut h = hash_def(45u, did);
        hash_substs(h, substs)
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
          ast::carg_lit(m) { ret const_eval::lit_eq(l, m); } _ { ret false; }
        }
      }
    }
}

fn args_eq<T>(eq: fn(T, T) -> bool,
              a: [@sp_constr_arg<T>],
              b: [@sp_constr_arg<T>]) -> bool {
    let mut i: uint = 0u;
    for a.each {|arg|
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
    let mut i = 0u;
    for cs.each {|c| if !constr_eq(c, ds[i]) { ret false; } i += 1u; }
    ret true;
}

fn node_id_to_type(cx: ctxt, id: ast::node_id) -> t {
    alt smallintmap::find(*cx.node_types, id as uint) {
       some(t) { t }
       none { cx.sess.bug(#fmt("node_id_to_type: unbound node ID %s",
                               ast_map::node_id_to_str(cx.items, id))); }
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

// Just checks whether it's a fn that returns bool,
// not its purity.
fn is_pred_ty(fty: t) -> bool {
    is_fn_ty(fty) && type_is_bool(ty_fn_ret(fty))
}

fn ty_var_id(typ: t) -> ty_vid {
    alt get(typ).struct {
      ty_var(vid) { ret vid; }
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

fn field_idx(id: ast::ident, fields: [field]) -> option<uint> {
    let mut i = 0u;
    for fields.each {|f| if f.ident == id { ret some(i); } i += 1u; }
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
    let mut i = 0u;
    for meths.each {|m| if m.ident == id { ret some(i); } i += 1u; }
    ret none;
}

fn sort_methods(meths: [method]) -> [method] {
    fn method_lteq(a: method, b: method) -> bool {
        ret str::le(a.ident, b.ident);
    }
    ret std::sort::merge_sort(bind method_lteq(_, _), meths);
}

fn occurs_check(tcx: ctxt, sp: span, vid: ty_vid, rt: t) {
    // Fast path
    if !type_needs_infer(rt) { ret; }

    // Occurs check!
    if vec::contains(vars_in_type(rt), vid) {
            // Maybe this should be span_err -- however, there's an
            // assertion later on that the type doesn't contain
            // variables, so in this case we have to be sure to die.
            tcx.sess.span_fatal
                (sp, "type inference failed because I \
                     could not find a type\n that's both of the form "
                 + ty_to_str(tcx, mk_var(tcx, vid)) +
                 " and of the form " + ty_to_str(tcx, rt) +
                 " - such a type would have to be infinitely large.");
    }
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

fn ty_sort_str(cx: ctxt, t: t) -> str {
    alt get(t).struct {
      ty_nil | ty_bot | ty_bool | ty_int(_) |
      ty_uint(_) | ty_float(_) | ty_estr(_) | ty_str |
      ty_type | ty_opaque_box | ty_opaque_closure_ptr(_) {
        ty_to_str(cx, t)
      }

      ty_enum(id, _) { #fmt["enum %s", item_path_str(cx, id)] }
      ty_box(_) { "@-ptr" }
      ty_uniq(_) { "~-ptr" }
      ty_evec(_, _) | ty_vec(_) { "vector" }
      ty_ptr(_) { "*-ptr" }
      ty_rptr(_, _) { "&-ptr" }
      ty_rec(_) { "record" }
      ty_fn(_) { "fn" }
      ty_iface(id, _) { #fmt["iface %s", item_path_str(cx, id)] }
      ty_class(id, _) { #fmt["class %s", item_path_str(cx, id)] }
      ty_res(id, _, _) { #fmt["resource %s", item_path_str(cx, id)] }
      ty_tup(_) { "tuple" }
      ty_var(_) { "variable" }
      ty_param(_, _) { "type parameter" }
      ty_self { "self" }
      ty_constr(t, _) { ty_sort_str(cx, t) }
    }
}

fn type_err_to_str(cx: ctxt, err: type_err) -> str {
    fn terr_vstore_kind_to_str(k: terr_vstore_kind) -> str {
        alt k { terr_vec { "[]" } terr_str { "str" } }
    }

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
      terr_proto_mismatch(e, a) {
        ret #fmt["closure protocol mismatch (%s vs %s)",
                 proto_to_str(e), proto_to_str(a)];
      }
      terr_mutability { ret "values differ in mutability"; }
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
        ret "expected a record with field `" + e_fld +
                "` but found one with field `" + a_fld + "`";
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
      terr_regions_differ(subregion, superregion) {
        ret #fmt("references with lifetime %s do not necessarily \
                  outlive references with lifetime %s",
                 region_to_str(cx, subregion),
                 region_to_str(cx, superregion));
      }
      terr_vstores_differ(k, e_vs, a_vs) {
        ret #fmt("%s storage differs: expected %s but found %s",
                 terr_vstore_kind_to_str(k),
                 vstore_to_str(cx, e_vs),
                 vstore_to_str(cx, a_vs));
      }
      terr_in_field(err, fname) {
        ret #fmt("in field `%s`, %s", fname, type_err_to_str(cx, *err));
      }
      terr_sorts(exp, act) {
        ret #fmt("%s vs %s", ty_sort_str(cx, exp), ty_sort_str(cx, act));
      }
      terr_self_substs {
        ret "inconsistent self substitution"; // XXX this is more of a bug
      }
    }
}

fn def_has_ty_params(def: ast::def) -> bool {
    alt def {
      ast::def_fn(_, _) | ast::def_variant(_, _) | ast::def_class(_)
        { true }
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
        alt cx.items.find(id.node) {
           some(ast_map::node_item(@{node: ast::item_impl(
              _, _, some(@{id: id, _}), _, _), _}, _)) {
              some(node_id_to_type(cx, id))
           }
           some(ast_map::node_item(@{node: ast::item_class(_, _, _, _, _),
                           _},_)) {
             alt cx.def_map.find(id.node) {
               some(def_ty(iface_id)) {
                   some(node_id_to_type(cx, id.node))
               }
               _ {
                 cx.sess.bug("impl_iface: iface ref isn't in iface map \
                         and isn't bound to a def_ty");
               }
             }
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

fn substd_enum_variants(cx: ctxt,
                        id: ast::def_id,
                        substs: substs) -> [variant_info] {
    vec::map(*enum_variants(cx, id)) { |variant_info|
        let substd_args = vec::map(variant_info.args) {|aty|
            subst(cx, substs, aty)
        };

        let substd_ctor_ty = subst(cx, substs, variant_info.ctor_ty);

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

          ast_map::node_ctor(nm, _, _, path) {
              *path + [ast_map::path_name(nm)]
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
          ast_map::node_item(@{node: ast::item_enum(variants, _, _), _}, _) {
            let mut disr_val = -1;
            @vec::map(variants, {|variant|
                let ctor_ty = node_id_to_type(cx, variant.node.id);
                let arg_tys = {
                    if vec::len(variant.node.args) > 0u {
                        ty_fn_args(ctor_ty).map { |a| a.ty }
                    } else { [] }
                };
                alt variant.node.disr_expr {
                  some (ex) {
                    // FIXME: issue #1417
                    disr_val = alt const_eval::eval_const_expr(cx, ex) {
                      const_eval::const_int(val) {val as int}
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
    let mut i = 0u;
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
    alt cx.tcache.find(did) {
      some(tpt) { ret tpt; }
      none {
        // The item is in this crate. The caller should have added it to the
        // type cache already
        assert did.crate != ast::local_crate;
        let tyt = csearch::get_type(cx, did);
        cx.tcache.insert(did, tyt);
        ret tyt;
      }
    }
}

// Look up a field ID, whether or not it's local
// Takes a list of type substs in case the class is generic
fn lookup_field_type(tcx: ctxt, class_id: def_id, id: def_id,
                     substs: substs) -> ty::t {

    let t = if id.crate == ast::local_crate {
        node_id_to_type(tcx, id.node)
    }
    else {
        alt tcx.tcache.find(id) {
           some(tpt) { tpt.ty }
           none {
               let tpt = csearch::get_field_type(tcx, class_id, id);
               // ok b/c fields are monomorphic
               // TODO: Comment might be a lie, what if it mentions
               // class-bound ty params?
               tcx.tcache.insert(id, tpt);
               tpt.ty
           }
        }
    };
    subst(tcx, substs, t)
}

// Look up the list of field names and IDs for a given class
// Fails if the id is not bound to a class.
fn lookup_class_fields(cx: ctxt, did: ast::def_id) -> [field_ty] {
  if did.crate == ast::local_crate {
    alt cx.items.find(did.node) {
       some(ast_map::node_item(i,_)) {
         alt i.node {
           ast::item_class(_, _, items, _, _) {
               class_field_tys(items)
           }
           _ { cx.sess.bug("class ID bound to non-class"); }
         }
       }
       _ {
           cx.sess.bug(#fmt("class ID not bound to an item: %s",
                            ast_map::node_id_to_str(cx.items, did.node)));
       }
    }
        }
  else {
        ret csearch::get_class_fields(cx, did);
    }
}

fn lookup_class_field(cx: ctxt, parent: ast::def_id, field_id: ast::def_id)
    -> field_ty {
    alt vec::find(lookup_class_fields(cx, parent))
                 {|f| f.id.node == field_id.node} {
        some(t) { t }
        none { cx.sess.bug("class ID not found in parent's fields"); }
    }
}

fn lookup_public_fields(cx: ctxt, did: ast::def_id) -> [field_ty] {
    vec::filter(lookup_class_fields(cx, did), is_public)
}

pure fn is_public(f: field_ty) -> bool {
  alt f.vis {
    public { true }
    private { false }
  }
}

// Look up the list of method names and IDs for a given class
// Fails if the id is not bound to a class.
fn lookup_class_method_ids(cx: ctxt, did: ast::def_id)
    : is_local(did) -> [{name: ident, id: node_id, vis: visibility}] {
    alt cx.items.find(did.node) {
       some(ast_map::node_item(@{node: item_class(_,_,items,_,_), _}, _)) {
         let (_,ms) = split_class_items(items);
         vec::map(ms, {|m| {name: m.ident, id: m.id,
                            vis: m.vis}})
       }
       _ {
           cx.sess.bug("lookup_class_method_ids: id not bound to a class");
       }
    }
}

/* Given a class def_id and a method name, return the method's
 def_id. Needed so we can do static dispatch for methods
 Doesn't care about the method's privacy. (It's assumed that
 the caller already checked that.)
*/
fn lookup_class_method_by_name(cx:ctxt, did: ast::def_id, name: ident,
                               sp: span) -> def_id {
    if check is_local(did) {
       let ms = lookup_class_method_ids(cx, did);
       for ms.each {|m|
         if m.name == name {
             ret ast_util::local_def(m.id);
         }
       }
       cx.sess.span_fatal(sp, #fmt("Class doesn't have a method \
           named %s", name));
    }
    else {
      csearch::get_class_method(cx.sess.cstore, did, name)
    }
}

fn class_field_tys(items: [@class_member]) -> [field_ty] {
    let mut rslt = [];
    for items.each {|it|
       alt it.node {
          instance_var(nm, _, cm, id, vis) {
              rslt += [{ident: nm, id: ast_util::local_def(id),
                        vis: vis, mutability: cm}];
          }
          class_method(_) { }
       }
    }
    rslt
}

// Return a list of fields corresponding to the class's items
// (as if the class was a record). trans uses this
// Takes a list of substs with which to instantiate field types
fn class_items_as_fields(cx:ctxt, did: ast::def_id,
                         substs: substs) -> [field] {
    let mut rslt = [];
    for lookup_class_fields(cx, did).each {|f|
       // consider all instance vars mut, because the
       // constructor may mutate all vars
       rslt += [{ident: f.ident, mt:
               {ty: lookup_field_type(cx, did, f.id, substs),
                    mutbl: m_mutbl}}];
    }
    rslt
}

fn is_binopable(_cx: ctxt, ty: t, op: ast::binop) -> bool {
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

    fn tycat(ty: t) -> int {
        alt get(ty).struct {
          ty_bool { tycat_bool }
          ty_int(_) | ty_uint(_) { tycat_int }
          ty_float(_) { tycat_float }
          ty_estr(_) | ty_str { tycat_str }
          ty_evec(_, _) | ty_vec(_) { tycat_vec }
          ty_rec(_) | ty_tup(_) | ty_enum(_, _) { tycat_struct }
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

fn ty_params_to_tys(tcx: ty::ctxt, tps: [ast::ty_param]) -> [t] {
    vec::from_fn(tps.len(), {|i|
                ty::mk_param(tcx, i, ast_util::local_def(tps[i].id))
        })
}

#[doc = "
Returns an equivalent type with all the typedefs and self regions removed.
"]
fn normalize_ty(cx: ctxt, t: t) -> t {
    alt cx.normalized_cache.find(t) {
      some(t) { ret t; }
      none { }
    }

    let t = alt get(t).struct {
        ty_enum(did, r) {
            alt r.self_r {
              some(_) {
                // This enum has a self region. Get rid of it
                mk_enum(cx, did, {self_r: none,
                                  self_ty: none,
                                  tps: r.tps})
              }
              none { t }
            }
        }
        _ { t }
    };

    // FIXME #2187: This also reduced int types to their compatible machine
    // types, which isn't necessary after #2187
    let t = mk_t(cx, mach_sty(cx.sess.targ_cfg, t));

    let sty = fold_sty(get(t).struct) {|t| normalize_ty(cx, t) };
    let t_norm = mk_t(cx, sty);
    cx.normalized_cache.insert(t, t_norm);
    ret t_norm;
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
