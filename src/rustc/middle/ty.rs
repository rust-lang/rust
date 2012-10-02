// #[warn(deprecated_mode)];
#[warn(deprecated_pattern)];

use std::{map, smallintmap};
use result::Result;
use std::map::HashMap;
use driver::session;
use session::session;
use syntax::{ast, ast_map};
use syntax::ast_util;
use syntax::ast_util::{is_local, local_def};
use syntax::codemap::span;
use metadata::csearch;
use util::ppaux::{region_to_str, explain_region, vstore_to_str,
                  note_and_explain_region};
use middle::lint;
use middle::lint::{get_lint_level, allow};
use syntax::ast::*;
use syntax::print::pprust::*;
use util::ppaux::{ty_to_str, proto_ty_to_str, tys_to_str};
use std::serialization::{serialize_Option,
                         deserialize_Option,
                         serialize_uint,
                         deserialize_uint};

export TyVid, IntVid, FnVid, RegionVid, vid;
export br_hashmap;
export is_instantiable;
export node_id_to_type;
export node_id_to_type_params;
export arg;
export args_eq;
export block_ty;
export class_items_as_fields, class_items_as_mutable_fields;
export ctxt;
export deref, deref_sty;
export index, index_sty;
export def_has_ty_params;
export expr_has_ty_params;
export expr_ty;
export expr_ty_params_and_ty;
export expr_is_lval, expr_kind;
export ExprKind, LvalueExpr, RvalueDatumExpr, RvalueDpsExpr, RvalueStmtExpr;
export field_ty;
export fold_ty, fold_sty_to_ty, fold_region, fold_regions;
export fold_regions_and_ty, walk_regions_and_ty;
export field;
export field_idx, field_idx_strict;
export get_field;
export get_fields;
export get_element_type;
export has_dtor;
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
export stmt_node_id;
export sty;
export subst, subst_tps, substs_is_noop, substs_to_str, substs;
export t;
export new_ty_hash;
export enum_variants, substd_enum_variants, enum_is_univariant;
export trait_methods, store_trait_methods, impl_traits;
export enum_variant_with_id;
export ty_dtor;
export ty_param_bounds_and_ty;
export ty_bool, mk_bool, type_is_bool;
export ty_bot, mk_bot, type_is_bot;
export ty_box, mk_box, mk_imm_box, type_is_box, type_is_boxed;
export ty_opaque_closure_ptr, mk_opaque_closure_ptr;
export ty_opaque_box, mk_opaque_box;
export ty_float, mk_float, mk_mach_float, type_is_fp;
export ty_fn, FnTy, FnTyBase, FnMeta, FnSig, mk_fn;
export ty_fn_proto, ty_fn_purity, ty_fn_ret, ty_fn_ret_style, tys_in_fn_ty;
export ty_int, mk_int, mk_mach_int, mk_char;
export mk_i8, mk_u8, mk_i16, mk_u16, mk_i32, mk_u32, mk_i64, mk_u64;
export ty_estr, mk_estr, type_is_str;
export ty_evec, mk_evec, type_is_vec;
export ty_unboxed_vec, mk_unboxed_vec, mk_mut_unboxed_vec;
export vstore, vstore_fixed, vstore_uniq, vstore_box, vstore_slice;
export ty_nil, mk_nil, type_is_nil;
export ty_trait, mk_trait;
export ty_param, mk_param, ty_params_to_tys;
export ty_ptr, mk_ptr, mk_mut_ptr, mk_imm_ptr, mk_nil_ptr, type_is_unsafe_ptr;
export ty_rptr, mk_rptr, mk_mut_rptr, mk_imm_rptr;
export ty_rec, mk_rec;
export ty_enum, mk_enum, type_is_enum;
export ty_tup, mk_tup;
export ty_type, mk_type;
export ty_uint, mk_uint, mk_mach_uint;
export ty_uniq, mk_uniq, mk_imm_uniq, type_is_unique_box;
export ty_infer, mk_infer, type_is_ty_var, mk_var, mk_int_var;
export InferTy, TyVar, IntVar;
export ty_self, mk_self, type_has_self;
export ty_class;
export region, bound_region, encl_region;
export re_bound, re_free, re_scope, re_static, re_var;
export br_self, br_anon, br_named, br_cap_avoid;
export get, type_has_params, type_needs_infer, type_has_regions;
export type_is_region_ptr;
export type_id;
export tbox_has_flag;
export ty_var_id;
export ty_to_def_id;
export ty_fn_args;
export ty_region;
export kind, kind_implicitly_copyable, kind_send_copy, kind_copyable;
export kind_noncopyable, kind_const;
export kind_can_be_copied, kind_can_be_sent, kind_can_be_implicitly_copied;
export kind_is_safe_for_default_mode;
export kind_is_owned;
export meta_kind, kind_lteq, type_kind;
export operators;
export type_err, terr_vstore_kind;
export type_err_to_str, note_and_explain_type_err;
export expected_found;
export type_needs_drop;
export type_is_empty;
export type_is_integral;
export type_is_numeric;
export type_is_pod;
export type_is_scalar;
export type_is_immediate;
export type_is_borrowed;
export type_is_sequence;
export type_is_signed;
export type_is_structural;
export type_is_copyable;
export type_is_slice;
export type_is_unique;
export type_is_c_like_enum;
export type_structurally_contains;
export type_structurally_contains_uniques;
export type_autoderef, deref, deref_sty;
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
export param_ty;
export param_bound, param_bounds, bound_copy, bound_owned;
export param_bounds_to_str, param_bound_to_str;
export bound_send, bound_trait;
export param_bounds_to_kind;
export default_arg_mode_for_ty;
export item_path;
export item_path_str;
export ast_ty_to_ty_cache_entry;
export atttce_unresolved, atttce_resolved;
export mach_sty;
export ty_sort_str;
export normalize_ty;
export to_str;
export borrow, serialize_borrow, deserialize_borrow;
export bound_const;
export terr_no_integral_type, terr_ty_param_size, terr_self_substs;
export terr_in_field, terr_record_fields, terr_vstores_differ, terr_arg_count;
export terr_sorts, terr_vec, terr_str, terr_record_size, terr_tuple_size;
export terr_regions_does_not_outlive, terr_mutability, terr_purity_mismatch;
export terr_regions_not_same, terr_regions_no_overlap;
export terr_proto_mismatch;
export terr_ret_style_mismatch;
export terr_fn, terr_trait;
export purity_to_str;
export param_tys_in_type;
export eval_repeat_count;
export fn_proto, proto_bare, proto_vstore;
export ast_proto_to_proto;
export is_blockish;
export method_call_bounds;
export hash_region;
export region_variance, rv_covariant, rv_invariant, rv_contravariant;
export serialize_region_variance, deserialize_region_variance;
export opt_region_variance;
export serialize_opt_region_variance, deserialize_opt_region_variance;
export determine_inherited_purity;
export provided_trait_methods;
export AutoAdjustment, serialize_AutoAdjustment, deserialize_AutoAdjustment;
export AutoRef, AutoRefKind, AutoSlice, AutoPtr;

// Data types

// Note: after typeck, you should use resolved_mode() to convert this mode
// into an rmode, which will take into account the results of mode inference.
type arg = {mode: ast::mode, ty: t};

type field = {ident: ast::ident, mt: mt};

type param_bounds = @~[param_bound];

type method = {ident: ast::ident,
               tps: @~[param_bounds],
               fty: FnTy,
               self_ty: ast::self_ty_,
               vis: ast::visibility};

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
type creader_cache_key = {cnum: int, pos: uint, len: uint};
type creader_cache = HashMap<creader_cache_key, t>;

impl creader_cache_key : cmp::Eq {
    pure fn eq(other: &creader_cache_key) -> bool {
        self.cnum == (*other).cnum &&
            self.pos == (*other).pos &&
            self.len == (*other).len
    }
    pure fn ne(other: &creader_cache_key) -> bool {
        !(self == (*other))
    }
}

impl creader_cache_key : to_bytes::IterBytes {
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
        to_bytes::iter_bytes_3(&self.cnum, &self.pos, &self.len, lsb0, f);
    }
}

type intern_key = {sty: sty, o_def_id: Option<ast::def_id>};

impl intern_key : cmp::Eq {
    pure fn eq(other: &intern_key) -> bool {
        self.sty == (*other).sty && self.o_def_id == (*other).o_def_id
    }
    pure fn ne(other: &intern_key) -> bool { !self.eq(other) }
}

impl intern_key : to_bytes::IterBytes {
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
        to_bytes::iter_bytes_2(&self.sty, &self.o_def_id, lsb0, f);
    }
}

enum ast_ty_to_ty_cache_entry {
    atttce_unresolved,  /* not resolved yet */
    atttce_resolved(t)  /* resolved to a type, irrespective of region */
}

#[auto_serialize]
type opt_region_variance = Option<region_variance>;

#[auto_serialize]
enum region_variance { rv_covariant, rv_invariant, rv_contravariant }

impl region_variance : cmp::Eq {
    pure fn eq(other: &region_variance) -> bool {
        match (self, (*other)) {
            (rv_covariant, rv_covariant) => true,
            (rv_invariant, rv_invariant) => true,
            (rv_contravariant, rv_contravariant) => true,
            (rv_covariant, _) => false,
            (rv_invariant, _) => false,
            (rv_contravariant, _) => false
        }
    }
    pure fn ne(other: &region_variance) -> bool { !self.eq(other) }
}

#[auto_serialize]
type AutoAdjustment = {
    autoderefs: uint,
    autoref: Option<AutoRef>
};

#[auto_serialize]
type AutoRef = {
    kind: AutoRefKind,
    region: region,
    mutbl: ast::mutability
};

#[auto_serialize]
enum AutoRefKind {
    /// Convert from @[]/~[] to &[] (or str)
    AutoSlice,

    /// Convert from T to &T
    AutoPtr
}

type ctxt =
    @{diag: syntax::diagnostic::span_handler,
      interner: HashMap<intern_key, t_box>,
      mut next_id: uint,
      vecs_implicitly_copyable: bool,
      legacy_modes: bool,
      cstore: metadata::cstore::cstore,
      sess: session::session,
      def_map: resolve::DefMap,

      region_map: middle::region::region_map,
      region_paramd_items: middle::region::region_paramd_items,

      // Stores the types for various nodes in the AST.  Note that this table
      // is not guaranteed to be populated until after typeck.  See
      // typeck::check::fn_ctxt for details.
      node_types: node_type_table,

      // Stores the type parameters which were substituted to obtain the type
      // of this node.  This only applies to nodes that refer to entities
      // parameterized by type parameters, such as generic fns, types, or
      // other items.
      node_type_substs: HashMap<node_id, ~[t]>,

      items: ast_map::map,
      intrinsic_defs: HashMap<ast::ident, (ast::def_id, t)>,
      freevars: freevars::freevar_map,
      tcache: type_cache,
      rcache: creader_cache,
      ccache: constness_cache,
      short_names_cache: HashMap<t, @~str>,
      needs_drop_cache: HashMap<t, bool>,
      needs_unwind_cleanup_cache: HashMap<t, bool>,
      kind_cache: HashMap<t, kind>,
      ast_ty_to_ty_cache: HashMap<@ast::ty, ast_ty_to_ty_cache_entry>,
      enum_var_cache: HashMap<def_id, @~[variant_info]>,
      trait_method_cache: HashMap<def_id, @~[method]>,
      ty_param_bounds: HashMap<ast::node_id, param_bounds>,
      inferred_modes: HashMap<ast::node_id, ast::mode>,
      adjustments: HashMap<ast::node_id, @AutoAdjustment>,
      normalized_cache: HashMap<t, t>,
      lang_items: middle::lang_items::LanguageItems,
      legacy_boxed_traits: HashMap<node_id, ()>};

enum tbox_flag {
    has_params = 1,
    has_self = 2,
    needs_infer = 4,
    has_regions = 8,

    // a meta-flag: subst may be required if the type has parameters, a self
    // type, or references bound regions
    needs_subst = 1 | 2 | 8
}

type t_box = @{sty: sty,
               id: uint,
               flags: uint,
               o_def_id: Option<ast::def_id>};

// To reduce refcounting cost, we're representing types as unsafe pointers
// throughout the compiler. These are simply casted t_box values. Use ty::get
// to cast them back to a box. (Without the cast, compiler performance suffers
// ~15%.) This does mean that a t value relies on the ctxt to keep its box
// alive, and using ty::get is unsafe when the ctxt is no longer alive.
enum t_opaque {}
type t = *t_opaque;

pure fn get(t: t) -> t_box unsafe {
    let t2 = cast::reinterpret_cast::<t, t_box>(&t);
    let t3 = t2;
    cast::forget(move t2);
    t3
}

pure fn tbox_has_flag(tb: t_box, flag: tbox_flag) -> bool {
    (tb.flags & (flag as uint)) != 0u
}
pure fn type_has_params(t: t) -> bool { tbox_has_flag(get(t), has_params) }
pure fn type_has_self(t: t) -> bool { tbox_has_flag(get(t), has_self) }
pure fn type_needs_infer(t: t) -> bool { tbox_has_flag(get(t), needs_infer) }
pure fn type_has_regions(t: t) -> bool { tbox_has_flag(get(t), has_regions) }
pure fn type_def_id(t: t) -> Option<ast::def_id> { get(t).o_def_id }
pure fn type_id(t: t) -> uint { get(t).id }

enum closure_kind {
    ck_block,
    ck_box,
    ck_uniq,
}

impl closure_kind : to_bytes::IterBytes {
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
        (self as u8).iter_bytes(lsb0, f)
    }
}

impl closure_kind : cmp::Eq {
    pure fn eq(other: &closure_kind) -> bool {
        (self as uint) == ((*other) as uint)
    }
    pure fn ne(other: &closure_kind) -> bool { !self.eq(other) }
}

enum fn_proto {
    proto_bare,             // supertype of all other protocols
    proto_vstore(vstore)
}

impl fn_proto : to_bytes::IterBytes {
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
        match self {
          proto_bare =>
          0u8.iter_bytes(lsb0, f),

          proto_vstore(ref v) =>
          to_bytes::iter_bytes_2(&1u8, v, lsb0, f)
        }
    }
}

impl fn_proto : cmp::Eq {
    pure fn eq(other: &fn_proto) -> bool {
        match self {
            proto_bare => {
                match (*other) {
                    proto_bare => true,
                    _ => false
                }
            }
            proto_vstore(e0a) => {
                match (*other) {
                    proto_vstore(e0b) => e0a == e0b,
                    _ => false
                }
            }
        }
    }
    pure fn ne(other: &fn_proto) -> bool { !self.eq(other) }
}

/**
 * Meta information about a closure.
 *
 * - `purity` is the function's effect (pure, impure, unsafe).
 * - `proto` is the protocol (fn@, fn~, etc).
 * - `bounds` is the parameter bounds on the function's upvars.
 * - `ret_style` indicates whether the function returns a value or fails. */
struct FnMeta {
    purity: ast::purity,
    proto: fn_proto,
    bounds: @~[param_bound],
    ret_style: ret_style
}

/**
 * Signature of a function type, which I have arbitrarily
 * decided to use to refer to the input/output types.
 *
 * - `inputs` is the list of arguments and their modes.
 * - `output` is the return type. */
struct FnSig {
    inputs: ~[arg],
    output: t
}

/**
 * Function type: combines the meta information and the
 * type signature.  This particular type is parameterized
 * by the meta information because, in some cases, the
 * meta information is inferred. */
struct FnTyBase<M: cmp::Eq> {
    meta: M,
    sig: FnSig
}

type FnTy = FnTyBase<FnMeta>;

type param_ty = {idx: uint, def_id: def_id};

impl param_ty : cmp::Eq {
    pure fn eq(other: &param_ty) -> bool {
        self.idx == (*other).idx && self.def_id == (*other).def_id
    }
    pure fn ne(other: &param_ty) -> bool { !self.eq(other) }
}

impl param_ty : to_bytes::IterBytes {
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
        to_bytes::iter_bytes_2(&self.idx, &self.def_id, lsb0, f)
    }
}


/// Representation of regions:
#[auto_serialize]
enum region {
    /// Bound regions are found (primarily) in function types.  They indicate
    /// region parameters that have yet to be replaced with actual regions
    /// (analogous to type parameters, except that due to the monomorphic
    /// nature of our type system, bound type parameters are always replaced
    /// with fresh type variables whenever an item is referenced, so type
    /// parameters only appear "free" in types.  Regions in contrast can
    /// appear free or bound.).  When a function is called, all bound regions
    /// tied to that function's node-id are replaced with fresh region
    /// variables whose value is then inferred.
    re_bound(bound_region),

    /// When checking a function body, the types of all arguments and so forth
    /// that refer to bound region parameters are modified to refer to free
    /// region parameters.
    re_free(node_id, bound_region),

    /// A concrete region naming some expression within the current function.
    re_scope(node_id),

    /// Static data that has an "infinite" lifetime.
    re_static,

    /// A region variable.  Should not exist after typeck.
    re_var(RegionVid)
}

#[auto_serialize]
enum bound_region {
    /// The self region for classes, impls (&T in a type defn or &self/T)
    br_self,

    /// An anonymous region parameter for a given fn (&T)
    br_anon(uint),

    /// Named region parameters for functions (a in &a/T)
    br_named(ast::ident),

    /**
     * Handles capture-avoiding substitution in a rather subtle case.  If you
     * have a closure whose argument types are being inferred based on the
     * expected type, and the expected type includes bound regions, then we
     * will wrap those bound regions in a br_cap_avoid() with the id of the
     * fn expression.  This ensures that the names are not "captured" by the
     * enclosing scope, which may define the same names.  For an example of
     * where this comes up, see src/test/compile-fail/regions-ret-borrowed.rs
     * and regions-ret-borrowed-1.rs. */
    br_cap_avoid(ast::node_id, @bound_region),
}

type opt_region = Option<region>;

/**
 * The type substs represents the kinds of things that can be substituted to
 * convert a polytype into a monotype.  Note however that substituting bound
 * regions other than `self` is done through a different mechanism:
 *
 * - `tps` represents the type parameters in scope.  They are indexed
 *   according to the order in which they were declared.
 *
 * - `self_r` indicates the region parameter `self` that is present on nominal
 *   types (enums, classes) declared as having a region parameter.  `self_r`
 *   should always be none for types that are not region-parameterized and
 *   Some(_) for types that are.  The only bound region parameter that should
 *   appear within a region-parameterized type is `self`.
 *
 * - `self_ty` is the type to which `self` should be remapped, if any.  The
 *   `self` type is rather funny in that it can only appear on traits and is
 *   always substituted away to the implementing type for a trait. */
type substs = {
    self_r: opt_region,
    self_ty: Option<ty::t>,
    tps: ~[t]
};

// NB: If you change this, you'll probably want to change the corresponding
// AST structure in libsyntax/ast.rs as well.
enum sty {
    ty_nil,
    ty_bot,
    ty_bool,
    ty_int(ast::int_ty),
    ty_uint(ast::uint_ty),
    ty_float(ast::float_ty),
    ty_estr(vstore),
    ty_enum(def_id, substs),
    ty_box(mt),
    ty_uniq(mt),
    ty_evec(mt, vstore),
    ty_ptr(mt),
    ty_rptr(region, mt),
    ty_rec(~[field]),
    ty_fn(FnTy),
    ty_trait(def_id, substs, vstore),
    ty_class(def_id, substs),
    ty_tup(~[t]),

    ty_param(param_ty), // type parameter
    ty_self, // special, implicit `self` type parameter

    ty_infer(InferTy), // soething used only during inference/typeck

    // "Fake" types, used for trans purposes
    ty_type, // type_desc*
    ty_opaque_box, // used by monomorphizer to represent any @ box
    ty_opaque_closure_ptr(closure_kind), // ptr to env for fn, fn@, fn~
    ty_unboxed_vec(mt),
}

enum terr_vstore_kind {
    terr_vec, terr_str, terr_fn, terr_trait
}

struct expected_found<T> {
    expected: T,
    found: T
}

// Data structures used in type unification
enum type_err {
    terr_mismatch,
    terr_ret_style_mismatch(expected_found<ast::ret_style>),
    terr_purity_mismatch(expected_found<purity>),
    terr_mutability,
    terr_proto_mismatch(expected_found<ty::fn_proto>),
    terr_box_mutability,
    terr_ptr_mutability,
    terr_ref_mutability,
    terr_vec_mutability,
    terr_tuple_size(expected_found<uint>),
    terr_ty_param_size(expected_found<uint>),
    terr_record_size(expected_found<uint>),
    terr_record_mutability,
    terr_record_fields(expected_found<ident>),
    terr_arg_count,
    terr_mode_mismatch(expected_found<mode>),
    terr_regions_does_not_outlive(region, region),
    terr_regions_not_same(region, region),
    terr_regions_no_overlap(region, region),
    terr_vstores_differ(terr_vstore_kind, expected_found<vstore>),
    terr_in_field(@type_err, ast::ident),
    terr_sorts(expected_found<t>),
    terr_self_substs,
    terr_no_integral_type,
}

enum param_bound {
    bound_copy,
    bound_owned,
    bound_send,
    bound_const,
    bound_trait(t),
}

enum TyVid = uint;
enum IntVid = uint;
enum FnVid = uint;
#[auto_serialize]
enum RegionVid = uint;

enum InferTy {
    TyVar(TyVid),
    IntVar(IntVid)
}

impl InferTy : to_bytes::IterBytes {
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
        match self {
          TyVar(ref tv) => to_bytes::iter_bytes_2(&0u8, tv, lsb0, f),
          IntVar(ref iv) => to_bytes::iter_bytes_2(&1u8, iv, lsb0, f)
        }
    }
}

impl param_bound : to_bytes::IterBytes {
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
        match self {
          bound_copy => 0u8.iter_bytes(lsb0, f),
          bound_owned => 1u8.iter_bytes(lsb0, f),
          bound_send => 2u8.iter_bytes(lsb0, f),
          bound_const => 3u8.iter_bytes(lsb0, f),
          bound_trait(ref t) =>
          to_bytes::iter_bytes_2(&4u8, t, lsb0, f)
        }
    }
}

trait vid {
    pure fn to_uint() -> uint;
    pure fn to_str() -> ~str;
}

impl TyVid: vid {
    pure fn to_uint() -> uint { *self }
    pure fn to_str() -> ~str { fmt!("<V%u>", self.to_uint()) }
}

impl IntVid: vid {
    pure fn to_uint() -> uint { *self }
    pure fn to_str() -> ~str { fmt!("<VI%u>", self.to_uint()) }
}

impl FnVid: vid {
    pure fn to_uint() -> uint { *self }
    pure fn to_str() -> ~str { fmt!("<F%u>", self.to_uint()) }
}

impl RegionVid: vid {
    pure fn to_uint() -> uint { *self }
    pure fn to_str() -> ~str { fmt!("%?", self) }
}

impl InferTy {
    pure fn to_hash() -> uint {
        match self {
            TyVar(v) => v.to_uint() << 1,
            IntVar(v) => (v.to_uint() << 1) + 1,
        }
    }

    pure fn to_str() -> ~str {
        match self {
            TyVar(v) => v.to_str(),
            IntVar(v) => v.to_str(),
        }
    }
}

trait purity_to_str {
    pure fn to_str() -> ~str;
}

impl purity: purity_to_str {
    pure fn to_str() -> ~str {
        purity_to_str(self)
    }
}

impl RegionVid : to_bytes::IterBytes {
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
        (*self).iter_bytes(lsb0, f)
    }
}

impl TyVid : to_bytes::IterBytes {
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
        (*self).iter_bytes(lsb0, f)
    }
}

impl IntVid : to_bytes::IterBytes {
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
        (*self).iter_bytes(lsb0, f)
    }
}

impl FnVid : to_bytes::IterBytes {
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
        (*self).iter_bytes(lsb0, f)
    }
}

fn param_bounds_to_kind(bounds: param_bounds) -> kind {
    let mut kind = kind_noncopyable();
    for vec::each(*bounds) |bound| {
        match *bound {
          bound_copy => {
            kind = raise_kind(kind, kind_implicitly_copyable());
          }
          bound_owned => {
            kind = raise_kind(kind, kind_owned());
          }
          bound_send => {
            kind = raise_kind(kind, kind_send_only() | kind_owned());
          }
          bound_const => {
            kind = raise_kind(kind, kind_const());
          }
          bound_trait(_) => ()
        }
    }
    kind
}

/// A polytype.
///
/// - `bounds`: The list of bounds for each type parameter.  The length of the
///   list also tells you how many type parameters there are.
///
/// - `rp`: true if the type is region-parameterized.  Types can have at
///   most one region parameter, always called `&self`.
///
/// - `ty`: the base type.  May have reference to the (unsubstituted) bound
///   region `&self` or to (unsubstituted) ty_param types
type ty_param_bounds_and_ty = {bounds: @~[param_bounds],
                               region_param: Option<region_variance>,
                               ty: t};

type type_cache = HashMap<ast::def_id, ty_param_bounds_and_ty>;

type constness_cache = HashMap<ast::def_id, const_eval::constness>;

type node_type_table = @smallintmap::SmallIntMap<t>;

fn mk_rcache() -> creader_cache {
    type val = {cnum: int, pos: uint, len: uint};
    return map::HashMap();
}

fn new_ty_hash<V: Copy>() -> map::HashMap<t, V> {
    map::HashMap()
}

fn mk_ctxt(s: session::session,
           dm: resolve::DefMap,
           amap: ast_map::map,
           freevars: freevars::freevar_map,
           region_map: middle::region::region_map,
           region_paramd_items: middle::region::region_paramd_items,
           +lang_items: middle::lang_items::LanguageItems,
           crate: @ast::crate) -> ctxt {
    let mut legacy_modes = false;
    for crate.node.attrs.each |attribute| {
        match attribute.node.value.node {
            ast::meta_word(w) if w == ~"legacy_modes" => {
                legacy_modes = true;
                break;
            }
            _ => {}
        }
    }

    let interner = map::HashMap();
    let vecs_implicitly_copyable =
        get_lint_level(s.lint_settings.default_settings,
                       lint::vecs_implicitly_copyable) == allow;
    @{diag: s.diagnostic(),
      interner: interner,
      mut next_id: 0u,
      vecs_implicitly_copyable: vecs_implicitly_copyable,
      legacy_modes: legacy_modes,
      cstore: s.cstore,
      sess: s,
      def_map: dm,
      region_map: region_map,
      region_paramd_items: region_paramd_items,
      node_types: @smallintmap::mk(),
      node_type_substs: map::HashMap(),
      items: amap,
      intrinsic_defs: map::HashMap(),
      freevars: freevars,
      tcache: HashMap(),
      rcache: mk_rcache(),
      ccache: HashMap(),
      short_names_cache: new_ty_hash(),
      needs_drop_cache: new_ty_hash(),
      needs_unwind_cleanup_cache: new_ty_hash(),
      kind_cache: new_ty_hash(),
      ast_ty_to_ty_cache: HashMap(),
      enum_var_cache: HashMap(),
      trait_method_cache: HashMap(),
      ty_param_bounds: HashMap(),
      inferred_modes: HashMap(),
      adjustments: HashMap(),
      normalized_cache: new_ty_hash(),
      lang_items: move lang_items,
      legacy_boxed_traits: HashMap()}
}


// Type constructors
fn mk_t(cx: ctxt, +st: sty) -> t { mk_t_with_id(cx, st, None) }

// Interns a type/name combination, stores the resulting box in cx.interner,
// and returns the box as cast to an unsafe ptr (see comments for t above).
fn mk_t_with_id(cx: ctxt, +st: sty, o_def_id: Option<ast::def_id>) -> t {
    let key = {sty: st, o_def_id: o_def_id};
    match cx.interner.find(key) {
      Some(t) => unsafe { return cast::reinterpret_cast(&t); },
      _ => ()
    }
    let mut flags = 0u;
    fn rflags(r: region) -> uint {
        (has_regions as uint) | {
            match r {
              ty::re_var(_) => needs_infer as uint,
              _ => 0u
            }
        }
    }
    fn sflags(substs: &substs) -> uint {
        let mut f = 0u;
        for substs.tps.each |tt| { f |= get(*tt).flags; }
        substs.self_r.iter(|r| f |= rflags(*r));
        return f;
    }
    match st {
      ty_estr(vstore_slice(r)) => {
        flags |= rflags(r);
      }
      ty_evec(mt, vstore_slice(r)) => {
        flags |= rflags(r);
        flags |= get(mt.ty).flags;
      }
      ty_nil | ty_bot | ty_bool | ty_int(_) | ty_float(_) | ty_uint(_) |
      ty_estr(_) | ty_type | ty_opaque_closure_ptr(_) |
      ty_opaque_box => (),
      ty_param(_) => flags |= has_params as uint,
      ty_infer(_) => flags |= needs_infer as uint,
      ty_self => flags |= has_self as uint,
      ty_enum(_, ref substs) | ty_class(_, ref substs)
      | ty_trait(_, ref substs, _) => {
        flags |= sflags(substs);
      }
      ty_box(m) | ty_uniq(m) | ty_evec(m, _) |
      ty_ptr(m) | ty_unboxed_vec(m) => {
        flags |= get(m.ty).flags;
      }
      ty_rptr(r, m) => {
        flags |= rflags(r);
        flags |= get(m.ty).flags;
      }
      ty_rec(flds) => for flds.each |f| { flags |= get(f.mt.ty).flags; },
      ty_tup(ts) => for ts.each |tt| { flags |= get(*tt).flags; },
      ty_fn(ref f) => {
        match f.meta.proto {
            ty::proto_vstore(vstore_slice(r)) => flags |= rflags(r),
            ty::proto_bare | ty::proto_vstore(_) => {}
        }
        for f.sig.inputs.each |a| { flags |= get(a.ty).flags; }
        flags |= get(f.sig.output).flags;
      }
    }
    let t = @{sty: st, id: cx.next_id, flags: flags, o_def_id: o_def_id};
    cx.interner.insert(key, t);
    cx.next_id += 1u;
    unsafe { cast::reinterpret_cast(&t) }
}

fn mk_nil(cx: ctxt) -> t { mk_t(cx, ty_nil) }

fn mk_bot(cx: ctxt) -> t { mk_t(cx, ty_bot) }

fn mk_bool(cx: ctxt) -> t { mk_t(cx, ty_bool) }

fn mk_int(cx: ctxt) -> t { mk_t(cx, ty_int(ast::ty_i)) }

fn mk_i8(cx: ctxt) -> t { mk_t(cx, ty_int(ast::ty_i8)) }

fn mk_i16(cx: ctxt) -> t { mk_t(cx, ty_int(ast::ty_i16)) }

fn mk_i32(cx: ctxt) -> t { mk_t(cx, ty_int(ast::ty_i32)) }

fn mk_i64(cx: ctxt) -> t { mk_t(cx, ty_int(ast::ty_i64)) }

fn mk_float(cx: ctxt) -> t { mk_t(cx, ty_float(ast::ty_f)) }

fn mk_uint(cx: ctxt) -> t { mk_t(cx, ty_uint(ast::ty_u)) }

fn mk_u8(cx: ctxt) -> t { mk_t(cx, ty_uint(ast::ty_u8)) }

fn mk_u16(cx: ctxt) -> t { mk_t(cx, ty_uint(ast::ty_u16)) }

fn mk_u32(cx: ctxt) -> t { mk_t(cx, ty_uint(ast::ty_u32)) }

fn mk_u64(cx: ctxt) -> t { mk_t(cx, ty_uint(ast::ty_u64)) }

fn mk_mach_int(cx: ctxt, tm: ast::int_ty) -> t { mk_t(cx, ty_int(tm)) }

fn mk_mach_uint(cx: ctxt, tm: ast::uint_ty) -> t { mk_t(cx, ty_uint(tm)) }

fn mk_mach_float(cx: ctxt, tm: ast::float_ty) -> t { mk_t(cx, ty_float(tm)) }

fn mk_char(cx: ctxt) -> t { mk_t(cx, ty_int(ast::ty_char)) }

fn mk_estr(cx: ctxt, t: vstore) -> t {
    mk_t(cx, ty_estr(t))
}

fn mk_enum(cx: ctxt, did: ast::def_id, +substs: substs) -> t {
    // take a copy of substs so that we own the vectors inside
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

fn mk_mut_rptr(cx: ctxt, r: region, ty: t) -> t {
    mk_rptr(cx, r, {ty: ty, mutbl: ast::m_mutbl})
}
fn mk_imm_rptr(cx: ctxt, r: region, ty: t) -> t {
    mk_rptr(cx, r, {ty: ty, mutbl: ast::m_imm})
}

fn mk_mut_ptr(cx: ctxt, ty: t) -> t { mk_ptr(cx, {ty: ty,
                                                  mutbl: ast::m_mutbl}) }

fn mk_imm_ptr(cx: ctxt, ty: t) -> t {
    mk_ptr(cx, {ty: ty, mutbl: ast::m_imm})
}

fn mk_nil_ptr(cx: ctxt) -> t {
    mk_ptr(cx, {ty: mk_nil(cx), mutbl: ast::m_imm})
}

fn mk_evec(cx: ctxt, tm: mt, t: vstore) -> t {
    mk_t(cx, ty_evec(tm, t))
}

fn mk_unboxed_vec(cx: ctxt, tm: mt) -> t {
    mk_t(cx, ty_unboxed_vec(tm))
}
fn mk_mut_unboxed_vec(cx: ctxt, ty: t) -> t {
    mk_t(cx, ty_unboxed_vec({ty: ty, mutbl: ast::m_imm}))
}

fn mk_rec(cx: ctxt, fs: ~[field]) -> t { mk_t(cx, ty_rec(fs)) }

fn mk_tup(cx: ctxt, ts: ~[t]) -> t { mk_t(cx, ty_tup(ts)) }

// take a copy because we want to own the various vectors inside
fn mk_fn(cx: ctxt, +fty: FnTy) -> t { mk_t(cx, ty_fn(fty)) }

fn mk_trait(cx: ctxt, did: ast::def_id, +substs: substs, vstore: vstore)
         -> t {
    // take a copy of substs so that we own the vectors inside
    mk_t(cx, ty_trait(did, substs, vstore))
}

fn mk_class(cx: ctxt, class_id: ast::def_id, +substs: substs) -> t {
    // take a copy of substs so that we own the vectors inside
    mk_t(cx, ty_class(class_id, substs))
}

fn mk_var(cx: ctxt, v: TyVid) -> t { mk_infer(cx, TyVar(v)) }

fn mk_int_var(cx: ctxt, v: IntVid) -> t {
    mk_infer(cx, IntVar(v))
}

fn mk_infer(cx: ctxt, it: InferTy) -> t { mk_t(cx, ty_infer(it)) }

fn mk_self(cx: ctxt) -> t { mk_t(cx, ty_self) }

fn mk_param(cx: ctxt, n: uint, k: def_id) -> t {
    mk_t(cx, ty_param({idx: n, def_id: k}))
}

fn mk_type(cx: ctxt) -> t { mk_t(cx, ty_type) }

fn mk_opaque_closure_ptr(cx: ctxt, ck: closure_kind) -> t {
    mk_t(cx, ty_opaque_closure_ptr(ck))
}

fn mk_opaque_box(cx: ctxt) -> t { mk_t(cx, ty_opaque_box) }

fn mk_with_id(cx: ctxt, base: t, def_id: ast::def_id) -> t {
    mk_t_with_id(cx, get(base).sty, Some(def_id))
}

// Converts s to its machine type equivalent
pure fn mach_sty(cfg: @session::config, t: t) -> sty {
    match get(t).sty {
      ty_int(ast::ty_i) => ty_int(cfg.int_type),
      ty_uint(ast::ty_u) => ty_uint(cfg.uint_type),
      ty_float(ast::ty_f) => ty_float(cfg.float_type),
      s => s
    }
}

fn default_arg_mode_for_ty(tcx: ctxt, ty: ty::t) -> ast::rmode {
    return if type_is_fn(ty) {
        //    ^^^^^^^^^^^^^^
        // FIXME(#2202) --- We retain by-ref by default to workaround a memory
        // leak that otherwise results when @fn is upcast to &fn.
        ast::by_ref
    } else if tcx.legacy_modes {
        if type_is_borrowed(ty) {
            // the old mode default was ++ for things like &ptr, but to be
            // forward-compatible with non-legacy, we should use +
            ast::by_copy
        } else if ty::type_is_immediate(ty) {
            ast::by_val
        } else {
            ast::by_ref
        }
    } else {
        ast::by_copy
    };

    fn type_is_fn(ty: t) -> bool {
        match get(ty).sty {
            ty_fn(*) => true,
            _ => false
        }
    }

    fn type_is_borrowed(ty: t) -> bool {
        match ty::get(ty).sty {
            ty::ty_rptr(*) => true,
            ty_evec(_, vstore_slice(_)) => true,
            ty_estr(vstore_slice(_)) => true,

            // technically, we prob ought to include
            // &fn(), but that is treated specially due to #2202
            _ => false
        }
    }
}

// Returns the narrowest lifetime enclosing the evaluation of the expression
// with id `id`.
fn encl_region(cx: ctxt, id: ast::node_id) -> ty::region {
    match cx.region_map.find(id) {
      Some(encl_scope) => ty::re_scope(encl_scope),
      None => ty::re_static
    }
}

fn walk_ty(ty: t, f: fn(t)) {
    maybe_walk_ty(ty, |t| { f(t); true });
}

fn maybe_walk_ty(ty: t, f: fn(t) -> bool) {
    if !f(ty) { return; }
    match get(ty).sty {
      ty_nil | ty_bot | ty_bool | ty_int(_) | ty_uint(_) | ty_float(_) |
      ty_estr(_) | ty_type | ty_opaque_box | ty_self |
      ty_opaque_closure_ptr(_) | ty_infer(_) | ty_param(_) => {
      }
      ty_box(tm) | ty_evec(tm, _) | ty_unboxed_vec(tm) |
      ty_ptr(tm) | ty_rptr(_, tm) => {
        maybe_walk_ty(tm.ty, f);
      }
      ty_enum(_, substs) | ty_class(_, substs) |
      ty_trait(_, substs, _) => {
        for substs.tps.each |subty| { maybe_walk_ty(*subty, f); }
      }
      ty_rec(fields) => {
        for fields.each |fl| { maybe_walk_ty(fl.mt.ty, f); }
      }
      ty_tup(ts) => { for ts.each |tt| { maybe_walk_ty(*tt, f); } }
      ty_fn(ref ft) => {
        for ft.sig.inputs.each |a| { maybe_walk_ty(a.ty, f); }
        maybe_walk_ty(ft.sig.output, f);
      }
      ty_uniq(tm) => { maybe_walk_ty(tm.ty, f); }
    }
}

fn fold_sty_to_ty(tcx: ty::ctxt, sty: &sty, foldop: fn(t) -> t) -> t {
    mk_t(tcx, fold_sty(sty, foldop))
}

fn fold_sty(sty: &sty, fldop: fn(t) -> t) -> sty {
    fn fold_substs(substs: &substs, fldop: fn(t) -> t) -> substs {
        {self_r: substs.self_r,
         self_ty: substs.self_ty.map(|t| fldop(*t)),
         tps: substs.tps.map(|t| fldop(*t))}
    }

    match *sty {
        ty_box(tm) => {
            ty_box({ty: fldop(tm.ty), mutbl: tm.mutbl})
        }
        ty_uniq(tm) => {
            ty_uniq({ty: fldop(tm.ty), mutbl: tm.mutbl})
        }
        ty_ptr(tm) => {
            ty_ptr({ty: fldop(tm.ty), mutbl: tm.mutbl})
        }
        ty_unboxed_vec(tm) => {
            ty_unboxed_vec({ty: fldop(tm.ty), mutbl: tm.mutbl})
        }
        ty_evec(tm, vst) => {
            ty_evec({ty: fldop(tm.ty), mutbl: tm.mutbl}, vst)
        }
        ty_enum(tid, ref substs) => {
            ty_enum(tid, fold_substs(substs, fldop))
        }
        ty_trait(did, ref substs, vst) => {
            ty_trait(did, fold_substs(substs, fldop), vst)
        }
        ty_rec(fields) => {
            let new_fields = do vec::map(fields) |fl| {
                let new_ty = fldop(fl.mt.ty);
                let new_mt = {ty: new_ty, mutbl: fl.mt.mutbl};
                {ident: fl.ident, mt: new_mt}
            };
            ty_rec(new_fields)
        }
        ty_tup(ts) => {
            let new_ts = vec::map(ts, |tt| fldop(*tt));
            ty_tup(new_ts)
        }
        ty_fn(ref f) => {
            let new_args = f.sig.inputs.map(|a| {
                let new_ty = fldop(a.ty);
                {mode: a.mode, ty: new_ty}
            });
            let new_output = fldop(f.sig.output);
            ty_fn(FnTyBase {
                meta: f.meta,
                sig: FnSig {inputs: new_args, output: new_output}
            })
        }
        ty_rptr(r, tm) => {
            ty_rptr(r, {ty: fldop(tm.ty), mutbl: tm.mutbl})
        }
        ty_class(did, ref substs) => {
            ty_class(did, fold_substs(substs, fldop))
        }
        ty_nil | ty_bot | ty_bool | ty_int(_) | ty_uint(_) | ty_float(_) |
        ty_estr(_) | ty_type | ty_opaque_closure_ptr(_) |
        ty_opaque_box | ty_infer(_) | ty_param(*) | ty_self => {
            *sty
        }
    }
}

// Folds types from the bottom up.
fn fold_ty(cx: ctxt, t0: t, fldop: fn(t) -> t) -> t {
    let sty = fold_sty(&get(t0).sty, |t| fold_ty(cx, fldop(t), fldop));
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
            |r| { walkr(r); r },
            |t| { walkt(t); walk_regions_and_ty(cx, t, walkr, walkt); t },
            |t| { walkt(t); walk_regions_and_ty(cx, t, walkr, walkt); t });
    }
}

fn fold_regions_and_ty(
    cx: ctxt,
    ty: t,
    fldr: fn(r: region) -> region,
    fldfnt: fn(t: t) -> t,
    fldt: fn(t: t) -> t) -> t {

    fn fold_substs(
        substs: &substs,
        fldr: fn(r: region) -> region,
        fldt: fn(t: t) -> t) -> substs {

        {self_r: substs.self_r.map(|r| fldr(*r)),
         self_ty: substs.self_ty.map(|t| fldt(*t)),
         tps: substs.tps.map(|t| fldt(*t))}
    }

    let tb = ty::get(ty);
    match tb.sty {
      ty::ty_rptr(r, mt) => {
        let m_r = fldr(r);
        let m_t = fldt(mt.ty);
        ty::mk_rptr(cx, m_r, {ty: m_t, mutbl: mt.mutbl})
      }
      ty_estr(vstore_slice(r)) => {
        let m_r = fldr(r);
        ty::mk_estr(cx, vstore_slice(m_r))
      }
      ty_evec(mt, vstore_slice(r)) => {
        let m_r = fldr(r);
        let m_t = fldt(mt.ty);
        ty::mk_evec(cx, {ty: m_t, mutbl: mt.mutbl}, vstore_slice(m_r))
      }
      ty_enum(def_id, ref substs) => {
        ty::mk_enum(cx, def_id, fold_substs(substs, fldr, fldt))
      }
      ty_class(def_id, ref substs) => {
        ty::mk_class(cx, def_id, fold_substs(substs, fldr, fldt))
      }
      ty_trait(def_id, ref substs, vst) => {
        ty::mk_trait(cx, def_id, fold_substs(substs, fldr, fldt), vst)
      }
      ty_fn(ref f) => {
        let new_proto;
        match f.meta.proto {
            proto_bare =>
                new_proto = proto_bare,
            proto_vstore(vstore_slice(r)) =>
                new_proto = proto_vstore(vstore_slice(fldr(r))),
            proto_vstore(old_vstore) =>
                new_proto = proto_vstore(old_vstore)
        }
        let new_args = vec::map(f.sig.inputs, |a| {
            let new_ty = fldfnt(a.ty);
            {mode: a.mode, ty: new_ty}
        });
        let new_output = fldfnt(f.sig.output);
        ty::mk_fn(cx, FnTyBase {
            meta: FnMeta {proto: new_proto, ..f.meta},
            sig: FnSig {inputs: new_args,
                        output: new_output}
        })
      }
      ref sty => {
        fold_sty_to_ty(cx, sty, |t| fldt(t))
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
        if !type_has_regions(ty) { return ty; }
        fold_regions_and_ty(
            cx, ty,
            |r| fldr(r, in_fn),
            |t| do_fold(cx, t, true, fldr),
            |t| do_fold(cx, t, in_fn, fldr))
    }
    do_fold(cx, ty, false, fldr)
}

fn fold_region(cx: ctxt, t0: t, fldop: fn(region, bool) -> region) -> t {
    fn do_fold(cx: ctxt, t0: t, under_r: bool,
               fldop: fn(region, bool) -> region) -> t {
        let tb = get(t0);
        if !tbox_has_flag(tb, has_regions) { return t0; }
        match tb.sty {
          ty_rptr(r, {ty: t1, mutbl: m}) => {
            let m_r = fldop(r, under_r);
            let m_t1 = do_fold(cx, t1, true, fldop);
            ty::mk_rptr(cx, m_r, {ty: m_t1, mutbl: m})
          }
          ty_estr(vstore_slice(r)) => {
            let m_r = fldop(r, under_r);
            ty::mk_estr(cx, vstore_slice(m_r))
          }
          ty_evec({ty: t1, mutbl: m}, vstore_slice(r)) => {
            let m_r = fldop(r, under_r);
            let m_t1 = do_fold(cx, t1, true, fldop);
            ty::mk_evec(cx, {ty: m_t1, mutbl: m}, vstore_slice(m_r))
          }
          ty_fn(_) => {
            // do not recurse into functions, which introduce fresh bindings
            t0
          }
          ref sty => {
            do fold_sty_to_ty(cx, sty) |t| {
                do_fold(cx, t, under_r, fldop)
            }
          }
      }
    }

    do_fold(cx, t0, false, fldop)
}

// Substitute *only* type parameters.  Used in trans where regions are erased.
fn subst_tps(cx: ctxt, tps: &[t], typ: t) -> t {
    if tps.len() == 0u { return typ; }
    let tb = ty::get(typ);
    if !tbox_has_flag(tb, has_params) { return typ; }
    match tb.sty {
      ty_param(p) => tps[p.idx],
      ref sty => fold_sty_to_ty(cx, sty, |t| subst_tps(cx, tps, t))
    }
}

fn substs_is_noop(substs: &substs) -> bool {
    substs.tps.len() == 0u &&
        substs.self_r.is_none() &&
        substs.self_ty.is_none()
}

fn substs_to_str(cx: ctxt, substs: &substs) -> ~str {
    fmt!("substs(self_r=%s, self_ty=%s, tps=%?)",
         substs.self_r.map_default(~"none", |r| region_to_str(cx, *r)),
         substs.self_ty.map_default(~"none", |t| ty_to_str(cx, *t)),
         tys_to_str(cx, substs.tps))
}

fn param_bound_to_str(cx: ctxt, pb: &param_bound) -> ~str {
    match *pb {
        bound_copy => ~"copy",
        bound_owned => ~"owned",
        bound_send => ~"send",
        bound_const => ~"const",
        bound_trait(t) => ty_to_str(cx, t)
    }
}

fn param_bounds_to_str(cx: ctxt, pbs: param_bounds) -> ~str {
    fmt!("%?", pbs.map(|pb| param_bound_to_str(cx, pb)))
}

fn subst(cx: ctxt,
         substs: &substs,
         typ: t) -> t {

    debug!("subst(substs=%s, typ=%s)",
           substs_to_str(cx, substs),
           ty_to_str(cx, typ));

    if substs_is_noop(substs) { return typ; }
    let r = do_subst(cx, substs, typ);
    debug!("  r = %s", ty_to_str(cx, r));
    return r;

    fn do_subst(cx: ctxt,
                substs: &substs,
                typ: t) -> t {
        let tb = get(typ);
        if !tbox_has_flag(tb, needs_subst) { return typ; }
        match tb.sty {
          ty_param(p) => substs.tps[p.idx],
          ty_self => substs.self_ty.get(),
          _ => {
            fold_regions_and_ty(
                cx, typ,
                |r| match r {
                    re_bound(br_self) => substs.self_r.get(),
                    _ => r
                },
                |t| do_subst(cx, substs, t),
                |t| do_subst(cx, substs, t))
          }
        }
    }
}

// Type utilities

fn type_is_nil(ty: t) -> bool { get(ty).sty == ty_nil }

fn type_is_bot(ty: t) -> bool { get(ty).sty == ty_bot }

fn type_is_ty_var(ty: t) -> bool {
    match get(ty).sty {
      ty_infer(TyVar(_)) => true,
      _ => false
    }
}

fn type_is_bool(ty: t) -> bool { get(ty).sty == ty_bool }

fn type_is_structural(ty: t) -> bool {
    match get(ty).sty {
      ty_rec(_) | ty_class(*) | ty_tup(_) | ty_enum(*) | ty_fn(_) |
      ty_trait(*) |
      ty_evec(_, vstore_fixed(_)) | ty_estr(vstore_fixed(_)) |
      ty_evec(_, vstore_slice(_)) | ty_estr(vstore_slice(_))
      => true,
      _ => false
    }
}

fn type_is_copyable(cx: ctxt, ty: t) -> bool {
    return kind_can_be_copied(type_kind(cx, ty));
}

fn type_is_sequence(ty: t) -> bool {
    match get(ty).sty {
      ty_estr(_) | ty_evec(_, _) => true,
      _ => false
    }
}

fn type_is_str(ty: t) -> bool {
    match get(ty).sty {
      ty_estr(_) => true,
      _ => false
    }
}

fn sequence_element_type(cx: ctxt, ty: t) -> t {
    match get(ty).sty {
      ty_estr(_) => return mk_mach_uint(cx, ast::ty_u8),
      ty_evec(mt, _) | ty_unboxed_vec(mt) => return mt.ty,
      _ => cx.sess.bug(
          ~"sequence_element_type called on non-sequence value"),
    }
}

fn get_element_type(ty: t, i: uint) -> t {
    match get(ty).sty {
      ty_rec(flds) => return flds[i].mt.ty,
      ty_tup(ts) => return ts[i],
      _ => fail ~"get_element_type called on invalid type"
    }
}

pure fn type_is_box(ty: t) -> bool {
    match get(ty).sty {
      ty_box(_) => return true,
      _ => return false
    }
}

pure fn type_is_boxed(ty: t) -> bool {
    match get(ty).sty {
      ty_box(_) | ty_opaque_box |
      ty_evec(_, vstore_box) | ty_estr(vstore_box) => true,
      _ => false
    }
}

pure fn type_is_region_ptr(ty: t) -> bool {
    match get(ty).sty {
      ty_rptr(_, _) => true,
      _ => false
    }
}

pure fn type_is_slice(ty: t) -> bool {
    match get(ty).sty {
      ty_evec(_, vstore_slice(_)) | ty_estr(vstore_slice(_)) => true,
      _ => return false
    }
}

pure fn type_is_unique_box(ty: t) -> bool {
    match get(ty).sty {
      ty_uniq(_) => return true,
      _ => return false
    }
}

pure fn type_is_unsafe_ptr(ty: t) -> bool {
    match get(ty).sty {
      ty_ptr(_) => return true,
      _ => return false
    }
}

pure fn type_is_vec(ty: t) -> bool {
    return match get(ty).sty {
          ty_evec(_, _) | ty_unboxed_vec(_) => true,
          ty_estr(_) => true,
          _ => false
        };
}

pure fn type_is_unique(ty: t) -> bool {
    match get(ty).sty {
      ty_uniq(_) => return true,
      ty_evec(_, vstore_uniq) => true,
      ty_estr(vstore_uniq) => true,
      _ => return false
    }
}

/*
 A scalar type is one that denotes an atomic datum, with no sub-components.
 (A ty_ptr is scalar because it represents a non-managed pointer, so its
 contents are abstract to rustc.)
*/
pure fn type_is_scalar(ty: t) -> bool {
    match get(ty).sty {
      ty_nil | ty_bool | ty_int(_) | ty_float(_) | ty_uint(_) |
      ty_infer(IntVar(_)) | ty_type | ty_ptr(_) => true,
      _ => false
    }
}

fn type_is_immediate(ty: t) -> bool {
    return type_is_scalar(ty) || type_is_boxed(ty) ||
        type_is_unique(ty) || type_is_region_ptr(ty);
}

fn type_needs_drop(cx: ctxt, ty: t) -> bool {
    match cx.needs_drop_cache.find(ty) {
      Some(result) => return result,
      None => {/* fall through */ }
    }

    let mut accum = false;
    let result = match get(ty).sty {
      // scalar types
      ty_nil | ty_bot | ty_bool | ty_int(_) | ty_float(_) | ty_uint(_) |
      ty_type | ty_ptr(_) | ty_rptr(_, _) |
      ty_estr(vstore_fixed(_)) |
      ty_estr(vstore_slice(_)) |
      ty_evec(_, vstore_slice(_)) |
      ty_self => false,

      ty_box(_) | ty_uniq(_) |
      ty_opaque_box | ty_opaque_closure_ptr(*) |
      ty_estr(vstore_uniq) |
      ty_estr(vstore_box) |
      ty_evec(_, vstore_uniq) |
      ty_evec(_, vstore_box) => true,

      ty_trait(*) => true,

      ty_param(*) | ty_infer(*) => true,

      ty_evec(mt, vstore_fixed(_)) => type_needs_drop(cx, mt.ty),
      ty_unboxed_vec(mt) => type_needs_drop(cx, mt.ty),
      ty_rec(flds) => {
        for flds.each |f| {
            if type_needs_drop(cx, f.mt.ty) { accum = true; }
        }
        accum
      }
      ty_class(did, ref substs) => {
         // Any class with a dtor needs a drop
         ty_dtor(cx, did).is_some() || {
             for vec::each(ty::class_items_as_fields(cx, did, substs)) |f| {
                 if type_needs_drop(cx, f.mt.ty) { accum = true; }
             }
             accum
         }
      }
      ty_tup(elts) => {
          for elts.each |m| { if type_needs_drop(cx, *m) { accum = true; } }
        accum
      }
      ty_enum(did, ref substs) => {
        let variants = enum_variants(cx, did);
          for vec::each(*variants) |variant| {
              for variant.args.each |aty| {
                // Perform any type parameter substitutions.
                let arg_ty = subst(cx, substs, *aty);
                if type_needs_drop(cx, arg_ty) { accum = true; }
            }
            if accum { break; }
        }
        accum
      }
      ty_fn(ref fty) => {
        match fty.meta.proto {
          proto_bare | proto_vstore(vstore_slice(_)) => false,
          _ => true
        }
      }
    };

    cx.needs_drop_cache.insert(ty, result);
    return result;
}

// Some things don't need cleanups during unwinding because the
// task can free them all at once later. Currently only things
// that only contain scalars and shared boxes can avoid unwind
// cleanups.
fn type_needs_unwind_cleanup(cx: ctxt, ty: t) -> bool {
    match cx.needs_unwind_cleanup_cache.find(ty) {
      Some(result) => return result,
      None => ()
    }

    let tycache = new_ty_hash();
    let needs_unwind_cleanup =
        type_needs_unwind_cleanup_(cx, ty, tycache, false);
    cx.needs_unwind_cleanup_cache.insert(ty, needs_unwind_cleanup);
    return needs_unwind_cleanup;
}

fn type_needs_unwind_cleanup_(cx: ctxt, ty: t,
                              tycache: map::HashMap<t, ()>,
                              encountered_box: bool) -> bool {

    // Prevent infinite recursion
    match tycache.find(ty) {
      Some(_) => return false,
      None => { tycache.insert(ty, ()); }
    }

    let mut encountered_box = encountered_box;
    let mut needs_unwind_cleanup = false;
    do maybe_walk_ty(ty) |ty| {
        let old_encountered_box = encountered_box;
        let result = match get(ty).sty {
          ty_box(_) | ty_opaque_box => {
            encountered_box = true;
            true
          }
          ty_nil | ty_bot | ty_bool |
          ty_int(_) | ty_uint(_) | ty_float(_) |
          ty_rec(_) | ty_tup(_) | ty_ptr(_) => {
            true
          }
          ty_enum(did, ref substs) => {
            for vec::each(*enum_variants(cx, did)) |v| {
                for v.args.each |aty| {
                    let t = subst(cx, substs, *aty);
                    needs_unwind_cleanup |=
                        type_needs_unwind_cleanup_(cx, t, tycache,
                                                   encountered_box);
                }
            }
            !needs_unwind_cleanup
          }
          ty_uniq(_) |
          ty_estr(vstore_uniq) |
          ty_estr(vstore_box) |
          ty_evec(_, vstore_uniq) |
          ty_evec(_, vstore_box)
          => {
            // Once we're inside a box, the annihilator will find
            // it and destroy it.
            if !encountered_box {
                needs_unwind_cleanup = true;
                false
            } else {
                true
            }
          }
          _ => {
            needs_unwind_cleanup = true;
            false
          }
        };

        encountered_box = old_encountered_box;
        result
    }

    return needs_unwind_cleanup;
}

enum kind { kind_(u32) }

/// can be copied (implicitly or explicitly)
const KIND_MASK_COPY         : u32 = 0b000000000000000000000000001_u32;

/// can be sent: no shared box, borrowed ptr (must imply OWNED)
const KIND_MASK_SEND         : u32 = 0b000000000000000000000000010_u32;

/// is owned (no borrowed ptrs)
const KIND_MASK_OWNED        : u32 = 0b000000000000000000000000100_u32;

/// is deeply immutable
const KIND_MASK_CONST        : u32 = 0b000000000000000000000001000_u32;

/// can be implicitly copied (must imply COPY)
const KIND_MASK_IMPLICIT     : u32 = 0b000000000000000000000010000_u32;

/// safe for default mode (subset of KIND_MASK_IMPLICIT)
const KIND_MASK_DEFAULT_MODE : u32 = 0b000000000000000000000100000_u32;

fn kind_noncopyable() -> kind {
    kind_(0u32)
}

fn kind_copyable() -> kind {
    kind_(KIND_MASK_COPY)
}

fn kind_implicitly_copyable() -> kind {
    kind_(KIND_MASK_IMPLICIT | KIND_MASK_COPY)
}

fn kind_safe_for_default_mode() -> kind {
    // similar to implicit copy, but always includes vectors and strings
    kind_(KIND_MASK_DEFAULT_MODE | KIND_MASK_IMPLICIT | KIND_MASK_COPY)
}

fn kind_implicitly_sendable() -> kind {
    kind_(KIND_MASK_IMPLICIT | KIND_MASK_COPY | KIND_MASK_SEND)
}

fn kind_safe_for_default_mode_send() -> kind {
    // similar to implicit copy, but always includes vectors and strings
    kind_(KIND_MASK_DEFAULT_MODE | KIND_MASK_IMPLICIT |
          KIND_MASK_COPY | KIND_MASK_SEND)
}


fn kind_send_copy() -> kind {
    kind_(KIND_MASK_COPY | KIND_MASK_SEND)
}

fn kind_send_only() -> kind {
    kind_(KIND_MASK_SEND)
}

fn kind_const() -> kind {
    kind_(KIND_MASK_CONST)
}

fn kind_owned() -> kind {
    kind_(KIND_MASK_OWNED)
}

fn kind_top() -> kind {
    kind_(0xffffffffu32)
}

fn remove_const(k: kind) -> kind {
    k - kind_const()
}

fn remove_implicit(k: kind) -> kind {
    k - kind_(KIND_MASK_IMPLICIT | KIND_MASK_DEFAULT_MODE)
}

fn remove_send(k: kind) -> kind {
    k - kind_(KIND_MASK_SEND)
}

fn remove_owned_send(k: kind) -> kind {
    k - kind_(KIND_MASK_OWNED) - kind_(KIND_MASK_SEND)
}

fn remove_copyable(k: kind) -> kind {
    k - kind_(KIND_MASK_COPY | KIND_MASK_DEFAULT_MODE)
}

impl kind : ops::BitAnd<kind,kind> {
    pure fn bitand(other: &kind) -> kind {
        unsafe {
            lower_kind(self, (*other))
        }
    }
}

impl kind : ops::BitOr<kind,kind> {
    pure fn bitor(other: &kind) -> kind {
        unsafe {
            raise_kind(self, (*other))
        }
    }
}

impl kind : ops::Sub<kind,kind> {
    pure fn sub(other: &kind) -> kind {
        unsafe {
            kind_(*self & !*(*other))
        }
    }
}

// Using these query functions is preferable to direct comparison or matching
// against the kind constants, as we may modify the kind hierarchy in the
// future.
pure fn kind_can_be_implicitly_copied(k: kind) -> bool {
    *k & KIND_MASK_IMPLICIT == KIND_MASK_IMPLICIT
}

pure fn kind_is_safe_for_default_mode(k: kind) -> bool {
    *k & KIND_MASK_DEFAULT_MODE == KIND_MASK_DEFAULT_MODE
}

pure fn kind_can_be_copied(k: kind) -> bool {
    *k & KIND_MASK_COPY == KIND_MASK_COPY
}

pure fn kind_can_be_sent(k: kind) -> bool {
    *k & KIND_MASK_SEND == KIND_MASK_SEND
}

pure fn kind_is_owned(k: kind) -> bool {
    *k & KIND_MASK_OWNED == KIND_MASK_OWNED
}

fn meta_kind(p: FnMeta) -> kind {
    match p.proto { // XXX consider the kind bounds!
      proto_vstore(vstore_slice(_)) =>
        kind_noncopyable() | kind_(KIND_MASK_DEFAULT_MODE),
      proto_vstore(vstore_box) =>
        kind_safe_for_default_mode() | kind_owned(),
      proto_vstore(vstore_uniq) =>
        kind_send_copy() | kind_owned(),
      proto_vstore(vstore_fixed(_)) =>
        fail ~"fixed vstore protos are not allowed",
      proto_bare =>
        kind_safe_for_default_mode_send() | kind_const() | kind_owned()
    }
}

fn kind_lteq(a: kind, b: kind) -> bool {
    *a & *b == *a
}

fn lower_kind(a: kind, b: kind) -> kind {
    kind_(*a & *b)
}

fn raise_kind(a: kind, b: kind) -> kind {
    kind_(*a | *b)
}

#[test]
fn test_kinds() {
    // The kind "lattice" is defined by the subset operation on the
    // set of permitted operations.
    assert kind_lteq(kind_send_copy(), kind_send_copy());
    assert kind_lteq(kind_copyable(), kind_send_copy());
    assert kind_lteq(kind_copyable(), kind_copyable());
    assert kind_lteq(kind_noncopyable(), kind_send_copy());
    assert kind_lteq(kind_noncopyable(), kind_copyable());
    assert kind_lteq(kind_noncopyable(), kind_noncopyable());
    assert kind_lteq(kind_copyable(), kind_implicitly_copyable());
    assert kind_lteq(kind_copyable(), kind_implicitly_sendable());
    assert kind_lteq(kind_send_copy(), kind_implicitly_sendable());
    assert !kind_lteq(kind_send_copy(), kind_implicitly_copyable());
    assert !kind_lteq(kind_copyable(), kind_send_only());
}

// Return the most permissive kind that a composite object containing a field
// with the given mutability can have.
// This is used to prevent objects containing mutable state from being
// implicitly copied and to compute whether things have const kind.
fn mutability_kind(m: mutability) -> kind {
    match (m) {
      m_mutbl => remove_const(remove_implicit(kind_top())),
      m_const => remove_implicit(kind_top()),
      m_imm => kind_top()
    }
}

fn mutable_type_kind(cx: ctxt, ty: mt) -> kind {
    lower_kind(mutability_kind(ty.mutbl), type_kind(cx, ty.ty))
}

fn type_kind(cx: ctxt, ty: t) -> kind {
    match cx.kind_cache.find(ty) {
      Some(result) => return result,
      None => {/* fall through */ }
    }

    // Insert a default in case we loop back on self recursively.
    cx.kind_cache.insert(ty, kind_top());

    let mut result = match get(ty).sty {
      // Scalar and unique types are sendable, constant, and owned
      ty_nil | ty_bot | ty_bool | ty_int(_) | ty_uint(_) | ty_float(_) |
      ty_ptr(_) => {
        kind_safe_for_default_mode_send() | kind_const() | kind_owned()
      }

      // Implicit copyability of strs is configurable
      ty_estr(vstore_uniq) => {
        if cx.vecs_implicitly_copyable {
            kind_implicitly_sendable() | kind_const() | kind_owned()
        } else {
            kind_send_copy() | kind_const() | kind_owned()
        }
      }

      // functions depend on the protocol
      ty_fn(ref f) => meta_kind(f.meta),

      // Those with refcounts raise noncopyable to copyable,
      // lower sendable to copyable. Therefore just set result to copyable.
      ty_box(tm) => {
        remove_send(mutable_type_kind(cx, tm) | kind_safe_for_default_mode())
      }

      // Trait instances are (for now) like shared boxes, basically
      ty_trait(_, _, _) => kind_safe_for_default_mode() | kind_owned(),

      // Static region pointers are copyable and sendable, but not owned
      ty_rptr(re_static, mt) =>
      kind_safe_for_default_mode() | mutable_type_kind(cx, mt),

      // General region pointers are copyable but NOT owned nor sendable
      ty_rptr(_, _) => kind_safe_for_default_mode(),

      // Unique boxes and vecs have the kind of their contained type,
      // but unique boxes can't be implicitly copyable.
      ty_uniq(tm) => remove_implicit(mutable_type_kind(cx, tm)),

      // Implicit copyability of vecs is configurable
      ty_evec(tm, vstore_uniq) => {
          if cx.vecs_implicitly_copyable {
              mutable_type_kind(cx, tm)
          } else {
              remove_implicit(mutable_type_kind(cx, tm))
          }
      }

      // Slices, refcounted evecs are copyable; uniques depend on the their
      // contained type, but aren't implicitly copyable.  Fixed vectors have
      // the kind of the element they contain, taking mutability into account.
      ty_evec(tm, vstore_box) => {
        remove_send(kind_safe_for_default_mode() | mutable_type_kind(cx, tm))
      }
      ty_evec(tm, vstore_slice(re_static)) => {
        kind_safe_for_default_mode() | mutable_type_kind(cx, tm)
      }
      ty_evec(tm, vstore_slice(_)) => {
        remove_owned_send(kind_safe_for_default_mode() |
                          mutable_type_kind(cx, tm))
      }
      ty_evec(tm, vstore_fixed(_)) => {
        mutable_type_kind(cx, tm)
      }

      // All estrs are copyable; uniques and interiors are sendable.
      ty_estr(vstore_box) => {
        kind_safe_for_default_mode() | kind_const() | kind_owned()
      }
      ty_estr(vstore_slice(re_static)) => {
        kind_safe_for_default_mode() | kind_send_copy() | kind_const()
      }
      ty_estr(vstore_slice(_)) => {
        kind_safe_for_default_mode() | kind_const()
      }
      ty_estr(vstore_fixed(_)) => {
        kind_safe_for_default_mode_send() | kind_const() | kind_owned()
      }

      // Records lower to the lowest of their members.
      ty_rec(flds) => {
        let mut lowest = kind_top();
        for flds.each |f| {
            lowest = lower_kind(lowest, mutable_type_kind(cx, f.mt));
        }
        lowest
      }

      ty_class(did, ref substs) => {
        // Classes are sendable if all their fields are sendable,
        // likewise for copyable...
        // also factor out this code, copied from the records case
        let mut lowest = kind_top();
        let flds = class_items_as_fields(cx, did, substs);
        for flds.each |f| {
            lowest = lower_kind(lowest, mutable_type_kind(cx, f.mt));
        }
        // ...but classes with dtors are never copyable (they can be
        // sendable)
        if ty::has_dtor(cx, did) {
           lowest = remove_copyable(lowest);
        }
        lowest
      }

      // Tuples lower to the lowest of their members.
      ty_tup(tys) => {
        let mut lowest = kind_top();
        for tys.each |ty| { lowest = lower_kind(lowest, type_kind(cx, *ty)); }
        lowest
      }

      // Enums lower to the lowest of their variants.
      ty_enum(did, ref substs) => {
        let mut lowest = kind_top();
        let variants = enum_variants(cx, did);
        if vec::len(*variants) == 0u {
            lowest = kind_send_only() | kind_owned();
        } else {
            for vec::each(*variants) |variant| {
                for variant.args.each |aty| {
                    // Perform any type parameter substitutions.
                    let arg_ty = subst(cx, substs, *aty);
                    lowest = lower_kind(lowest, type_kind(cx, arg_ty));
                    if lowest == kind_noncopyable() { break; }
                }
            }
        }
        lowest
      }

      ty_param(p) => {
        param_bounds_to_kind(cx.ty_param_bounds.get(p.def_id.node))
      }

      // self is a special type parameter that can only appear in traits; it
      // is never bounded in any way, hence it has the bottom kind.
      ty_self => kind_noncopyable(),

      ty_infer(_) => {
        cx.sess.bug(~"Asked to compute kind of a type variable");
      }
      ty_type | ty_opaque_closure_ptr(_)
      | ty_opaque_box | ty_unboxed_vec(_) => {
        cx.sess.bug(~"Asked to compute kind of fictitious type");
      }
    };

    // arbitrary threshold to prevent by-value copying of big records
    if kind_is_safe_for_default_mode(result) {
        if type_size(cx, ty) > 4 {
            result -= kind_(KIND_MASK_DEFAULT_MODE);
        }
    }

    cx.kind_cache.insert(ty, result);
    return result;
}

/// gives a rough estimate of how much space it takes to represent
/// an instance of `ty`.  Used for the mode transition.
fn type_size(cx: ctxt, ty: t) -> uint {
    match get(ty).sty {
      ty_nil | ty_bot | ty_bool | ty_int(_) | ty_uint(_) | ty_float(_) |
      ty_ptr(_) | ty_box(_) | ty_uniq(_) | ty_estr(vstore_uniq) |
      ty_trait(*) | ty_rptr(*) | ty_evec(_, vstore_uniq) |
      ty_evec(_, vstore_box) | ty_estr(vstore_box) => {
        1
      }

      ty_evec(_, vstore_slice(_)) |
      ty_estr(vstore_slice(_)) |
      ty_fn(_) => {
        2
      }

      ty_evec(t, vstore_fixed(n)) => {
        type_size(cx, t.ty) * n
      }

      ty_estr(vstore_fixed(n)) => {
        n
      }

      ty_rec(flds) => {
        flds.foldl(0, |s, f| *s + type_size(cx, f.mt.ty))
      }

      ty_class(did, ref substs) => {
        let flds = class_items_as_fields(cx, did, substs);
        flds.foldl(0, |s, f| *s + type_size(cx, f.mt.ty))
      }

      ty_tup(tys) => {
        tys.foldl(0, |s, t| *s + type_size(cx, *t))
      }

      ty_enum(did, ref substs) => {
        let variants = substd_enum_variants(cx, did, substs);
        variants.foldl( // find max size of any variant
            0,
            |m, v| uint::max(*m,
                             // find size of this variant:
                             v.args.foldl(0, |s, a| *s + type_size(cx, *a))))
      }

      ty_param(_) | ty_self => {
        1
      }

      ty_infer(_) => {
        cx.sess.bug(~"Asked to compute kind of a type variable");
      }
      ty_type | ty_opaque_closure_ptr(_)
      | ty_opaque_box | ty_unboxed_vec(_) => {
        cx.sess.bug(~"Asked to compute kind of fictitious type");
      }
    }
}

// True if instantiating an instance of `r_ty` requires an instance of `r_ty`.
fn is_instantiable(cx: ctxt, r_ty: t) -> bool {

    fn type_requires(cx: ctxt, seen: @mut ~[def_id],
                     r_ty: t, ty: t) -> bool {
        debug!("type_requires(%s, %s)?",
               ty_to_str(cx, r_ty),
               ty_to_str(cx, ty));

        let r = {
            get(r_ty).sty == get(ty).sty ||
                subtypes_require(cx, seen, r_ty, ty)
        };

        debug!("type_requires(%s, %s)? %b",
               ty_to_str(cx, r_ty),
               ty_to_str(cx, ty),
               r);
        return r;
    }

    fn subtypes_require(cx: ctxt, seen: @mut ~[def_id],
                        r_ty: t, ty: t) -> bool {
        debug!("subtypes_require(%s, %s)?",
               ty_to_str(cx, r_ty),
               ty_to_str(cx, ty));

        let r = match get(ty).sty {
          ty_nil |
          ty_bot |
          ty_bool |
          ty_int(_) |
          ty_uint(_) |
          ty_float(_) |
          ty_estr(_) |
          ty_fn(_) |
          ty_infer(_) |
          ty_param(_) |
          ty_self |
          ty_type |
          ty_opaque_box |
          ty_opaque_closure_ptr(_) |
          ty_evec(_, _) |
          ty_unboxed_vec(_) => {
            false
          }
          ty_box(mt) |
          ty_uniq(mt) |
          ty_rptr(_, mt) => {
            return type_requires(cx, seen, r_ty, mt.ty);
          }

          ty_ptr(*) => {
            false           // unsafe ptrs can always be NULL
          }

          ty_rec(fields) => {
            do vec::any(fields) |field| {
                type_requires(cx, seen, r_ty, field.mt.ty)
            }
          }

          ty_trait(_, _, _) => {
            false
          }

          ty_class(ref did, _) if vec::contains(*seen, did) => {
            false
          }

          ty_class(did, ref substs) => {
              seen.push(did);
              let r = vec::any(class_items_as_fields(cx, did, substs),
                               |f| type_requires(cx, seen, r_ty, f.mt.ty));
              seen.pop();
            r
          }

          ty_tup(ts) => {
            vec::any(ts, |t| type_requires(cx, seen, r_ty, *t))
          }

          ty_enum(ref did, _) if vec::contains(*seen, did) => {
            false
          }

            ty_enum(did, ref substs) => {
                seen.push(did);
                let vs = enum_variants(cx, did);
                let r = vec::len(*vs) > 0u && vec::all(*vs, |variant| {
                    vec::any(variant.args, |aty| {
                        let sty = subst(cx, substs, *aty);
                        type_requires(cx, seen, r_ty, sty)
                    })
                });
                seen.pop();
                r
            }
        };

        debug!("subtypes_require(%s, %s)? %b",
               ty_to_str(cx, r_ty),
               ty_to_str(cx, ty),
               r);

        return r;
    }

    let seen = @mut ~[];
    !subtypes_require(cx, seen, r_ty, r_ty)
}

fn type_structurally_contains(cx: ctxt, ty: t, test: fn(x: &sty) -> bool) ->
   bool {
    let sty = &get(ty).sty;
    debug!("type_structurally_contains: %s", ty_to_str(cx, ty));
    if test(sty) { return true; }
    match *sty {
      ty_enum(did, ref substs) => {
        for vec::each(*enum_variants(cx, did)) |variant| {
            for variant.args.each |aty| {
                let sty = subst(cx, substs, *aty);
                if type_structurally_contains(cx, sty, test) { return true; }
            }
        }
        return false;
      }
      ty_rec(fields) => {
        for fields.each |field| {
            if type_structurally_contains(cx, field.mt.ty, test) {
                return true;
            }
        }
        return false;
      }
      ty_class(did, ref substs) => {
        for lookup_class_fields(cx, did).each |field| {
            let ft = lookup_field_type(cx, did, field.id, substs);
            if type_structurally_contains(cx, ft, test) { return true; }
        }
        return false;
      }

      ty_tup(ts) => {
        for ts.each |tt| {
            if type_structurally_contains(cx, *tt, test) { return true; }
        }
        return false;
      }
      ty_evec(mt, vstore_fixed(_)) => {
        return type_structurally_contains(cx, mt.ty, test);
      }
      _ => return false
    }
}

fn type_structurally_contains_uniques(cx: ctxt, ty: t) -> bool {
    return type_structurally_contains(cx, ty, |sty| {
        match *sty {
          ty_uniq(_) |
          ty_evec(_, vstore_uniq) |
          ty_estr(vstore_uniq) => true,
          _ => false,
        }
    });
}

fn type_is_integral(ty: t) -> bool {
    match get(ty).sty {
      ty_infer(IntVar(_)) | ty_int(_) | ty_uint(_) | ty_bool => true,
      _ => false
    }
}

fn type_is_fp(ty: t) -> bool {
    match get(ty).sty {
      ty_float(_) => true,
      _ => false
    }
}

fn type_is_numeric(ty: t) -> bool {
    return type_is_integral(ty) || type_is_fp(ty);
}

fn type_is_signed(ty: t) -> bool {
    match get(ty).sty {
      ty_int(_) => true,
      _ => false
    }
}

// Whether a type is Plain Old Data -- meaning it does not contain pointers
// that the cycle collector might care about.
fn type_is_pod(cx: ctxt, ty: t) -> bool {
    let mut result = true;
    match get(ty).sty {
      // Scalar types
      ty_nil | ty_bot | ty_bool | ty_int(_) | ty_float(_) | ty_uint(_) |
      ty_type | ty_ptr(_) => result = true,
      // Boxed types
      ty_box(_) | ty_uniq(_) | ty_fn(_) |
      ty_estr(vstore_uniq) | ty_estr(vstore_box) |
      ty_evec(_, vstore_uniq) | ty_evec(_, vstore_box) |
      ty_trait(_, _, _) | ty_rptr(_,_) | ty_opaque_box => result = false,
      // Structural types
      ty_enum(did, ref substs) => {
        let variants = enum_variants(cx, did);
        for vec::each(*variants) |variant| {
            let tup_ty = mk_tup(cx, variant.args);

            // Perform any type parameter substitutions.
            let tup_ty = subst(cx, substs, tup_ty);
            if !type_is_pod(cx, tup_ty) { result = false; }
        }
      }
      ty_rec(flds) => {
        for flds.each |f| {
            if !type_is_pod(cx, f.mt.ty) { result = false; }
        }
      }
      ty_tup(elts) => {
        for elts.each |elt| { if !type_is_pod(cx, *elt) { result = false; } }
      }
      ty_estr(vstore_fixed(_)) => result = true,
      ty_evec(mt, vstore_fixed(_)) | ty_unboxed_vec(mt) => {
        result = type_is_pod(cx, mt.ty);
      }
      ty_param(_) => result = false,
      ty_opaque_closure_ptr(_) => result = true,
      ty_class(did, ref substs) => {
        result = vec::any(lookup_class_fields(cx, did), |f| {
            let fty = ty::lookup_item_type(cx, f.id);
            let sty = subst(cx, substs, fty.ty);
            type_is_pod(cx, sty)
        });
      }

      ty_estr(vstore_slice(*)) | ty_evec(_, vstore_slice(*)) => {
        result = false;
      }

      ty_infer(*) | ty_self(*) => {
        cx.sess.bug(~"non concrete type in type_is_pod");
      }
    }

    return result;
}

fn type_is_enum(ty: t) -> bool {
    match get(ty).sty {
      ty_enum(_, _) => return true,
      _ => return false
    }
}

// Whether a type is enum like, that is a enum type with only nullary
// constructors
fn type_is_c_like_enum(cx: ctxt, ty: t) -> bool {
    match get(ty).sty {
      ty_enum(did, _) => {
        let variants = enum_variants(cx, did);
        let some_n_ary = vec::any(*variants, |v| vec::len(v.args) > 0u);
        return !some_n_ary;
      }
      _ => return false
    }
}

fn type_param(ty: t) -> Option<uint> {
    match get(ty).sty {
      ty_param(p) => return Some(p.idx),
      _ => {/* fall through */ }
    }
    return None;
}

// Returns the type and mutability of *t.
//
// The parameter `expl` indicates if this is an *explicit* dereference.  Some
// types---notably unsafe ptrs---can only be dereferenced explicitly.
fn deref(cx: ctxt, t: t, expl: bool) -> Option<mt> {
    deref_sty(cx, &get(t).sty, expl)
}
fn deref_sty(cx: ctxt, sty: &sty, expl: bool) -> Option<mt> {
    match *sty {
      ty_rptr(_, mt) | ty_box(mt) | ty_uniq(mt) => {
        Some(mt)
      }

      ty_ptr(mt) if expl => {
        Some(mt)
      }

      ty_enum(did, ref substs) => {
        let variants = enum_variants(cx, did);
        if vec::len(*variants) == 1u && vec::len(variants[0].args) == 1u {
            let v_t = subst(cx, substs, variants[0].args[0]);
            Some({ty: v_t, mutbl: ast::m_imm})
        } else {
            None
        }
      }

      _ => None
    }
}

fn type_autoderef(cx: ctxt, t: t) -> t {
    let mut t = t;
    loop {
        match deref(cx, t, false) {
          None => return t,
          Some(mt) => t = mt.ty
        }
    }
}

// Returns the type and mutability of t[i]
fn index(cx: ctxt, t: t) -> Option<mt> {
    index_sty(cx, &get(t).sty)
}

fn index_sty(cx: ctxt, sty: &sty) -> Option<mt> {
    match *sty {
      ty_evec(mt, _) => Some(mt),
      ty_estr(_) => Some({ty: mk_u8(cx), mutbl: ast::m_imm}),
      _ => None
    }
}

impl bound_region : to_bytes::IterBytes {
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
        match self {
          ty::br_self => 0u8.iter_bytes(lsb0, f),

          ty::br_anon(ref idx) =>
          to_bytes::iter_bytes_2(&1u8, idx, lsb0, f),

          ty::br_named(ref ident) =>
          to_bytes::iter_bytes_2(&2u8, ident, lsb0, f),

          ty::br_cap_avoid(ref id, ref br) =>
          to_bytes::iter_bytes_3(&3u8, id, br, lsb0, f)
        }
    }
}

impl region : to_bytes::IterBytes {
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
        match self {
          re_bound(ref br) =>
          to_bytes::iter_bytes_2(&0u8, br, lsb0, f),

          re_free(ref id, ref br) =>
          to_bytes::iter_bytes_3(&1u8, id, br, lsb0, f),

          re_scope(ref id) =>
          to_bytes::iter_bytes_2(&2u8, id, lsb0, f),

          re_var(ref id) =>
          to_bytes::iter_bytes_2(&3u8, id, lsb0, f),

          re_static => 4u8.iter_bytes(lsb0, f)
        }
    }
}

impl vstore : to_bytes::IterBytes {
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
        match self {
          vstore_fixed(ref u) =>
          to_bytes::iter_bytes_2(&0u8, u, lsb0, f),

          vstore_uniq => 1u8.iter_bytes(lsb0, f),
          vstore_box => 2u8.iter_bytes(lsb0, f),

          vstore_slice(ref r) =>
          to_bytes::iter_bytes_2(&3u8, r, lsb0, f),
        }
    }
}

impl substs : to_bytes::IterBytes {
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
          to_bytes::iter_bytes_3(&self.self_r,
                                 &self.self_ty,
                                 &self.tps, lsb0, f)
    }
}

impl mt : to_bytes::IterBytes {
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
          to_bytes::iter_bytes_2(&self.ty,
                                 &self.mutbl, lsb0, f)
    }
}

impl field : to_bytes::IterBytes {
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
          to_bytes::iter_bytes_2(&self.ident,
                                 &self.mt, lsb0, f)
    }
}

impl arg : to_bytes::IterBytes {
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
          to_bytes::iter_bytes_2(&self.mode,
                                 &self.ty, lsb0, f)
    }
}

impl sty : to_bytes::IterBytes {
    pure fn iter_bytes(+lsb0: bool, f: to_bytes::Cb) {
        match self {
          ty_nil => 0u8.iter_bytes(lsb0, f),
          ty_bool => 1u8.iter_bytes(lsb0, f),

          ty_int(ref t) =>
          to_bytes::iter_bytes_2(&2u8, t, lsb0, f),

          ty_uint(ref t) =>
          to_bytes::iter_bytes_2(&3u8, t, lsb0, f),

          ty_float(ref t) =>
          to_bytes::iter_bytes_2(&4u8, t, lsb0, f),

          ty_estr(ref v) =>
          to_bytes::iter_bytes_2(&5u8, v, lsb0, f),

          ty_enum(ref did, ref substs) =>
          to_bytes::iter_bytes_3(&6u8, did, substs, lsb0, f),

          ty_box(ref mt) =>
          to_bytes::iter_bytes_2(&7u8, mt, lsb0, f),

          ty_evec(ref mt, ref v) =>
          to_bytes::iter_bytes_3(&8u8, mt, v, lsb0, f),

          ty_unboxed_vec(ref mt) =>
          to_bytes::iter_bytes_2(&9u8, mt, lsb0, f),

          ty_tup(ref ts) =>
          to_bytes::iter_bytes_2(&10u8, ts, lsb0, f),

          ty_rec(ref fs) =>
          to_bytes::iter_bytes_2(&11u8, fs, lsb0, f),

          ty_fn(ref ft) =>
          to_bytes::iter_bytes_7(&12u8,
                                 &ft.meta.purity,
                                 &ft.meta.proto,
                                 &ft.meta.bounds,
                                 &ft.sig.inputs,
                                 &ft.sig.output,
                                 &ft.meta.ret_style,
                                 lsb0, f),

          ty_self => 13u8.iter_bytes(lsb0, f),

          ty_infer(ref v) =>
          to_bytes::iter_bytes_2(&14u8, v, lsb0, f),

          ty_param(ref p) =>
          to_bytes::iter_bytes_2(&15u8, p, lsb0, f),

          ty_type => 16u8.iter_bytes(lsb0, f),
          ty_bot => 17u8.iter_bytes(lsb0, f),

          ty_ptr(ref mt) =>
          to_bytes::iter_bytes_2(&18u8, mt, lsb0, f),

          ty_uniq(ref mt) =>
          to_bytes::iter_bytes_2(&19u8, mt, lsb0, f),

          ty_trait(ref did, ref substs, ref v) =>
          to_bytes::iter_bytes_4(&20u8, did, substs, v, lsb0, f),

          ty_opaque_closure_ptr(ref ck) =>
          to_bytes::iter_bytes_2(&21u8, ck, lsb0, f),

          ty_opaque_box => 22u8.iter_bytes(lsb0, f),

          ty_class(ref did, ref substs) =>
          to_bytes::iter_bytes_3(&23u8, did, substs, lsb0, f),

          ty_rptr(ref r, ref mt) =>
          to_bytes::iter_bytes_3(&24u8, r, mt, lsb0, f),
        }
    }
}

fn br_hashmap<V:Copy>() -> HashMap<bound_region, V> {
    map::HashMap()
}

fn node_id_to_type(cx: ctxt, id: ast::node_id) -> t {
    //io::println(fmt!("%?/%?", id, cx.node_types.size()));
    match smallintmap::find(*cx.node_types, id as uint) {
       Some(t) => t,
       None => cx.sess.bug(
           fmt!("node_id_to_type: no type for node `%s`",
                ast_map::node_id_to_str(cx.items, id,
                                        cx.sess.parse_sess.interner)))
    }
}

fn node_id_to_type_params(cx: ctxt, id: ast::node_id) -> ~[t] {
    match cx.node_type_substs.find(id) {
      None => return ~[],
      Some(ts) => return ts
    }
}

fn node_id_has_type_params(cx: ctxt, id: ast::node_id) -> bool {
    return cx.node_type_substs.contains_key(id);
}

// Type accessors for substructures of types
fn ty_fn_args(fty: t) -> ~[arg] {
    match get(fty).sty {
      ty_fn(ref f) => f.sig.inputs,
      _ => fail ~"ty_fn_args() called on non-fn type"
    }
}

fn ty_fn_proto(fty: t) -> fn_proto {
    match get(fty).sty {
      ty_fn(ref f) => f.meta.proto,
      _ => fail ~"ty_fn_proto() called on non-fn type"
    }
}

fn ty_fn_purity(fty: t) -> ast::purity {
    match get(fty).sty {
      ty_fn(ref f) => f.meta.purity,
      _ => fail ~"ty_fn_purity() called on non-fn type"
    }
}

pure fn ty_fn_ret(fty: t) -> t {
    match get(fty).sty {
      ty_fn(ref f) => f.sig.output,
      _ => fail ~"ty_fn_ret() called on non-fn type"
    }
}

fn ty_fn_ret_style(fty: t) -> ast::ret_style {
    match get(fty).sty {
      ty_fn(ref f) => f.meta.ret_style,
      _ => fail ~"ty_fn_ret_style() called on non-fn type"
    }
}

fn is_fn_ty(fty: t) -> bool {
    match get(fty).sty {
      ty_fn(_) => true,
      _ => false
    }
}

fn ty_region(ty: t) -> region {
    match get(ty).sty {
      ty_rptr(r, _) => r,
      s => fail fmt!("ty_region() invoked on non-rptr: %?", s)
    }
}

// Returns a vec of all the input and output types of fty.
fn tys_in_fn_ty(fty: &FnTy) -> ~[t] {
    vec::append_one(fty.sig.inputs.map(|a| a.ty), fty.sig.output)
}

// Just checks whether it's a fn that returns bool,
// not its purity.
fn is_pred_ty(fty: t) -> bool {
    is_fn_ty(fty) && type_is_bool(ty_fn_ret(fty))
}

/*
fn ty_var_id(typ: t) -> TyVid {
    match get(typ).sty {
      ty_infer(TyVar(vid)) => return vid,
      _ => { error!("ty_var_id called on non-var ty"); fail; }
    }
}

fn int_var_id(typ: t) -> IntVid {
    match get(typ).sty {
      ty_infer(IntVar(vid)) => return vid,
      _ => { error!("ty_var_integral_id called on ty other than \
                  ty_var_integral");
         fail; }
    }
}
*/

// Type accessors for AST nodes
fn block_ty(cx: ctxt, b: &ast::blk) -> t {
    return node_id_to_type(cx, b.node.id);
}


// Returns the type of a pattern as a monotype. Like @expr_ty, this function
// doesn't provide type parameter substitutions.
fn pat_ty(cx: ctxt, pat: @ast::pat) -> t {
    return node_id_to_type(cx, pat.id);
}


// Returns the type of an expression as a monotype.
//
// NB: This type doesn't provide type parameter substitutions; e.g. if you
// ask for the type of "id" in "id(3)", it will return "fn(&int) -> int"
// instead of "fn(t) -> T with T = int". If this isn't what you want, see
// expr_ty_params_and_ty() below.
fn expr_ty(cx: ctxt, expr: @ast::expr) -> t {
    return node_id_to_type(cx, expr.id);
}

fn expr_ty_params_and_ty(cx: ctxt,
                         expr: @ast::expr) -> {params: ~[t], ty: t} {
    return {params: node_id_to_type_params(cx, expr.id),
         ty: node_id_to_type(cx, expr.id)};
}

fn expr_has_ty_params(cx: ctxt, expr: @ast::expr) -> bool {
    return node_id_has_type_params(cx, expr.id);
}

fn method_call_bounds(tcx: ctxt, method_map: typeck::method_map,
                      id: ast::node_id)
    -> Option<@~[param_bounds]> {
    do method_map.find(id).map |method| {
        match method.origin {
          typeck::method_static(did) => {
            // n.b.: When we encode class/impl methods, the bounds
            // that we encode include both the class/impl bounds
            // and then the method bounds themselves...
            ty::lookup_item_type(tcx, did).bounds
          }
          typeck::method_param({trait_id:trt_id,
                                method_num:n_mth, _}) |
          typeck::method_trait(trt_id, n_mth) => {
            // ...trait methods bounds, in contrast, include only the
            // method bounds, so we must preprend the tps from the
            // trait itself.  This ought to be harmonized.
            let trt_bounds =
                ty::lookup_item_type(tcx, trt_id).bounds;
            let mth = ty::trait_methods(tcx, trt_id)[n_mth];
            @(vec::append(*trt_bounds, *mth.tps))
          }
        }
    }
}

fn resolve_expr(tcx: ctxt, expr: @ast::expr) -> ast::def {
    match tcx.def_map.find(expr.id) {
        Some(def) => def,
        None => {
            tcx.sess.span_bug(expr.span, fmt!(
                "No def-map entry for expr %?", expr.id));
        }
    }
}

fn expr_is_lval(tcx: ctxt,
                method_map: typeck::method_map,
                e: @ast::expr) -> bool {
    match expr_kind(tcx, method_map, e) {
        LvalueExpr => true,
        RvalueDpsExpr | RvalueDatumExpr | RvalueStmtExpr => false
    }
}

/// We categorize expressions into three kinds.  The distinction between
/// lvalue/rvalue is fundamental to the language.  The distinction between the
/// two kinds of rvalues is an artifact of trans which reflects how we will
/// generate code for that kind of expression.  See trans/expr.rs for more
/// information.
enum ExprKind {
    LvalueExpr,
    RvalueDpsExpr,
    RvalueDatumExpr,
    RvalueStmtExpr
}

fn expr_kind(tcx: ctxt,
             method_map: typeck::method_map,
             expr: @ast::expr) -> ExprKind {
    if method_map.contains_key(expr.id) {
        // Overloaded operations are generally calls, and hence they are
        // generated via DPS.  However, assign_op (e.g., `x += y`) is an
        // exception, as its result is always unit.
        return match expr.node {
            ast::expr_assign_op(*) => RvalueStmtExpr,
            _ => RvalueDpsExpr
        };
    }

    match expr.node {
        ast::expr_path(*) => {
            match resolve_expr(tcx, expr) {
                ast::def_fn(*) | ast::def_static_method(*) |
                ast::def_variant(*) => RvalueDpsExpr,

                // Note: there is actually a good case to be made that
                // def_args, particularly those of immediate type, ought to
                // considered rvalues.
                ast::def_const(*) |
                ast::def_binding(*) |
                ast::def_upvar(*) |
                ast::def_arg(*) |
                ast::def_local(*) |
                ast::def_self(*) => LvalueExpr,

                move def => {
                    tcx.sess.span_bug(expr.span, fmt!(
                        "Uncategorized def for expr %?: %?",
                        expr.id, def));
                }
            }
        }

        ast::expr_unary(ast::deref, _) |
        ast::expr_field(*) |
        ast::expr_index(*) => {
            LvalueExpr
        }

        ast::expr_call(*) |
        ast::expr_rec(*) |
        ast::expr_struct(*) |
        ast::expr_tup(*) |
        ast::expr_if(*) |
        ast::expr_match(*) |
        ast::expr_fn(*) |
        ast::expr_fn_block(*) |
        ast::expr_loop_body(*) |
        ast::expr_do_body(*) |
        ast::expr_block(*) |
        ast::expr_copy(*) |
        ast::expr_unary_move(*) |
        ast::expr_repeat(*) |
        ast::expr_lit(@{node: lit_str(_), _}) |
        ast::expr_vstore(_, ast::expr_vstore_slice) |
        ast::expr_vstore(_, ast::expr_vstore_fixed(_)) |
        ast::expr_vec(*) => {
            RvalueDpsExpr
        }

        ast::expr_cast(*) => {
            match smallintmap::find(*tcx.node_types, expr.id as uint) {
                Some(t) => {
                    if ty::type_is_immediate(t) {
                        RvalueDatumExpr
                    } else {
                        RvalueDpsExpr
                    }
                }
                None => {
                    // Technically, it should not happen that the expr is not
                    // present within the table.  However, it DOES happen
                    // during type check, because the final types from the
                    // expressions are not yet recorded in the tcx.  At that
                    // time, though, we are only interested in knowing lvalue
                    // vs rvalue.  It would be better to base this decision on
                    // the AST type in cast node---but (at the time of this
                    // writing) it's not easy to distinguish casts to traits
                    // from other casts based on the AST.  This should be
                    // easier in the future, when casts to traits would like
                    // like @Foo, ~Foo, or &Foo.
                    RvalueDatumExpr
                }
            }
        }

        ast::expr_break(*) |
        ast::expr_again(*) |
        ast::expr_ret(*) |
        ast::expr_log(*) |
        ast::expr_fail(*) |
        ast::expr_assert(*) |
        ast::expr_while(*) |
        ast::expr_loop(*) |
        ast::expr_assign(*) |
        ast::expr_move(*) |
        ast::expr_swap(*) |
        ast::expr_assign_op(*) => {
            RvalueStmtExpr
        }

        ast::expr_lit(_) | // Note: lit_str is carved out above
        ast::expr_unary(*) |
        ast::expr_addr_of(*) |
        ast::expr_binary(*) |
        ast::expr_vstore(_, ast::expr_vstore_box) |
        ast::expr_vstore(_, ast::expr_vstore_uniq) => {
            RvalueDatumExpr
        }

        ast::expr_mac(*) => {
            tcx.sess.span_bug(
                expr.span,
                ~"macro expression remains after expansion");
        }
    }
}

fn stmt_node_id(s: @ast::stmt) -> ast::node_id {
    match s.node {
      ast::stmt_decl(_, id) | stmt_expr(_, id) | stmt_semi(_, id) => {
        return id;
      }
    }
}

fn field_idx(id: ast::ident, fields: &[field]) -> Option<uint> {
    let mut i = 0u;
    for fields.each |f| { if f.ident == id { return Some(i); } i += 1u; }
    return None;
}

fn field_idx_strict(tcx: ty::ctxt, id: ast::ident, fields: &[field]) -> uint {
    let mut i = 0u;
    for fields.each |f| { if f.ident == id { return i; } i += 1u; }
    tcx.sess.bug(fmt!(
        "No field named `%s` found in the list of fields `%?`",
        tcx.sess.str_of(id),
        fields.map(|f| tcx.sess.str_of(f.ident))));
}

fn get_field(tcx: ctxt, rec_ty: t, id: ast::ident) -> field {
    match vec::find(get_fields(rec_ty), |f| f.ident == id) {
      Some(f) => f,
      // Do we only call this when we know the field is legit?
      None => fail (#fmt("get_field: ty doesn't have a field %s",
                         tcx.sess.str_of(id)))
    }
}

fn get_fields(rec_ty:t) -> ~[field] {
    match get(rec_ty).sty {
      ty_rec(fields) => fields,
      // Can we check at the caller?
      _ => fail ~"get_fields: not a record type"
    }
}

fn method_idx(id: ast::ident, meths: &[method]) -> Option<uint> {
    let mut i = 0u;
    for meths.each |m| { if m.ident == id { return Some(i); } i += 1u; }
    return None;
}

/// Returns a vector containing the indices of all type parameters that appear
/// in `ty`.  The vector may contain duplicates.  Probably should be converted
/// to a bitset or some other representation.
fn param_tys_in_type(ty: t) -> ~[param_ty] {
    let mut rslt = ~[];
    do walk_ty(ty) |ty| {
        match get(ty).sty {
          ty_param(p) => {
            rslt.push(p);
          }
          _ => ()
        }
    }
    rslt
}

fn occurs_check(tcx: ctxt, sp: span, vid: TyVid, rt: t) {

    // Returns a vec of all the type variables occurring in `ty`. It may
    // contain duplicates.  (Integral type vars aren't counted.)
    fn vars_in_type(ty: t) -> ~[TyVid] {
        let mut rslt = ~[];
        do walk_ty(ty) |ty| {
            match get(ty).sty {
              ty_infer(TyVar(v)) => rslt.push(v),
              _ => ()
            }
        }
        rslt
    }

    // Fast path
    if !type_needs_infer(rt) { return; }

    // Occurs check!
    if vec::contains(vars_in_type(rt), &vid) {
            // Maybe this should be span_err -- however, there's an
            // assertion later on that the type doesn't contain
            // variables, so in this case we have to be sure to die.
            tcx.sess.span_fatal
                (sp, ~"type inference failed because I \
                     could not find a type\n that's both of the form "
                 + ty_to_str(tcx, mk_var(tcx, vid)) +
                 ~" and of the form " + ty_to_str(tcx, rt) +
                 ~" - such a type would have to be infinitely large.");
    }
}

// Maintains a little union-set tree for inferred modes.  `canon()` returns
// the current head value for `m0`.
fn canon<T:Copy cmp::Eq>(tbl: HashMap<ast::node_id, ast::inferable<T>>,
                         +m0: ast::inferable<T>) -> ast::inferable<T> {
    match m0 {
      ast::infer(id) => match tbl.find(id) {
        None => m0,
        Some(m1) => {
            let cm1 = canon(tbl, m1);
            // path compression:
            if cm1 != m1 { tbl.insert(id, cm1); }
            cm1
        }
      },
      _ => m0
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
    match canon_mode(cx, m) {
      ast::infer(_) => {
        cx.sess.bug(fmt!("mode %? was never resolved", m));
      }
      ast::expl(m0) => m0
    }
}

fn arg_mode(cx: ctxt, a: arg) -> ast::rmode { resolved_mode(cx, a.mode) }

// Unifies `m1` and `m2`.  Returns unified value or failure code.
fn unify_mode(cx: ctxt, modes: expected_found<ast::mode>)
    -> Result<ast::mode, type_err> {

    let m1 = modes.expected;
    let m2 = modes.found;
    match (canon_mode(cx, m1), canon_mode(cx, m2)) {
      (m1, m2) if (m1 == m2) => {
        result::Ok(m1)
      }
      (ast::infer(_), ast::infer(id2)) => {
        cx.inferred_modes.insert(id2, m1);
        result::Ok(m1)
      }
      (ast::infer(id), m) | (m, ast::infer(id)) => {
        cx.inferred_modes.insert(id, m);
        result::Ok(m1)
      }
      (_, _) => {
        result::Err(terr_mode_mismatch(modes))
      }
    }
}

// If `m` was never unified, unifies it with `m_def`.  Returns the final value
// for `m`.
fn set_default_mode(cx: ctxt, m: ast::mode, m_def: ast::rmode) {
    match canon_mode(cx, m) {
      ast::infer(id) => {
        cx.inferred_modes.insert(id, ast::expl(m_def));
      }
      ast::expl(_) => ()
    }
}

fn ty_sort_str(cx: ctxt, t: t) -> ~str {
    match get(t).sty {
      ty_nil | ty_bot | ty_bool | ty_int(_) |
      ty_uint(_) | ty_float(_) | ty_estr(_) |
      ty_type | ty_opaque_box | ty_opaque_closure_ptr(_) => {
        ty_to_str(cx, t)
      }

      ty_enum(id, _) => fmt!("enum %s", item_path_str(cx, id)),
      ty_box(_) => ~"@-ptr",
      ty_uniq(_) => ~"~-ptr",
      ty_evec(_, _) => ~"vector",
      ty_unboxed_vec(_) => ~"unboxed vector",
      ty_ptr(_) => ~"*-ptr",
      ty_rptr(_, _) => ~"&-ptr",
      ty_rec(_) => ~"record",
      ty_fn(_) => ~"fn",
      ty_trait(id, _, _) => fmt!("trait %s", item_path_str(cx, id)),
      ty_class(id, _) => fmt!("class %s", item_path_str(cx, id)),
      ty_tup(_) => ~"tuple",
      ty_infer(TyVar(_)) => ~"inferred type",
      ty_infer(IntVar(_)) => ~"integral variable",
      ty_param(_) => ~"type parameter",
      ty_self => ~"self"
    }
}

fn type_err_to_str(cx: ctxt, err: &type_err) -> ~str {
    /*!
     *
     * Explains the source of a type err in a short,
     * human readable way.  This is meant to be placed in
     * parentheses after some larger message.  You should
     * also invoke `note_and_explain_type_err()` afterwards
     * to present additional details, particularly when
     * it comes to lifetime-related errors. */

    fn terr_vstore_kind_to_str(k: terr_vstore_kind) -> ~str {
        match k {
            terr_vec => ~"[]",
            terr_str => ~"str",
            terr_fn => ~"fn",
            terr_trait => ~"trait"
        }
    }

    match *err {
      terr_mismatch => ~"types differ",
      terr_ret_style_mismatch(values) => {
        fn to_str(s: ast::ret_style) -> ~str {
            match s {
              ast::noreturn => ~"non-returning",
              ast::return_val => ~"return-by-value"
            }
        }
        fmt!("expected %s function, found %s function",
                    to_str(values.expected),
                    to_str(values.expected))
      }
      terr_purity_mismatch(values) => {
        fmt!("expected %s fn but found %s fn",
                    purity_to_str(values.expected),
                    purity_to_str(values.found))
      }
      terr_proto_mismatch(values) => {
        fmt!("expected %s closure, found %s closure",
             proto_ty_to_str(cx, values.expected),
             proto_ty_to_str(cx, values.found))
      }
      terr_mutability => ~"values differ in mutability",
      terr_box_mutability => ~"boxed values differ in mutability",
      terr_vec_mutability => ~"vectors differ in mutability",
      terr_ptr_mutability => ~"pointers differ in mutability",
      terr_ref_mutability => ~"references differ in mutability",
      terr_ty_param_size(values) => {
        fmt!("expected a type with %? type params \
              but found one with %? type params",
             values.expected, values.found)
      }
      terr_tuple_size(values) => {
        fmt!("expected a tuple with %? elements \
              but found one with %? elements",
             values.expected, values.found)
      }
      terr_record_size(values) => {
        fmt!("expected a record with %? fields \
              but found one with %? fields",
             values.expected, values.found)
      }
      terr_record_mutability => {
        ~"record elements differ in mutability"
      }
      terr_record_fields(values) => {
        fmt!("expected a record with field `%s` but found one with field \
              `%s`",
             cx.sess.str_of(values.expected), cx.sess.str_of(values.found))
      }
      terr_arg_count => ~"incorrect number of function parameters",
      terr_mode_mismatch(values) => {
        fmt!("expected argument mode %s, but found %s",
             mode_to_str(values.expected), mode_to_str(values.found))
      }
      terr_regions_does_not_outlive(*) => {
        fmt!("lifetime mismatch")
      }
      terr_regions_not_same(*) => {
        fmt!("lifetimes are not the same")
      }
      terr_regions_no_overlap(*) => {
        fmt!("lifetimes do not intersect")
      }
      terr_vstores_differ(k, values) => {
        fmt!("%s storage differs: expected %s but found %s",
                    terr_vstore_kind_to_str(k),
                    vstore_to_str(cx, values.expected),
                    vstore_to_str(cx, values.found))
      }
      terr_in_field(err, fname) => {
        fmt!("in field `%s`, %s", cx.sess.str_of(fname),
             type_err_to_str(cx, err))
      }
      terr_sorts(values) => {
        fmt!("expected %s but found %s",
                    ty_sort_str(cx, values.expected),
                    ty_sort_str(cx, values.found))
      }
      terr_self_substs => {
        ~"inconsistent self substitution" // XXX this is more of a bug
      }
      terr_no_integral_type => {
        ~"couldn't determine an appropriate integral type for integer \
          literal"
      }
    }
}

fn note_and_explain_type_err(cx: ctxt, err: &type_err) {
    match *err {
        terr_regions_does_not_outlive(subregion, superregion) => {
            note_and_explain_region(cx, ~"", subregion, ~"...");
            note_and_explain_region(cx, ~"...does not necessarily outlive ",
                                    superregion, ~"");
        }
        terr_regions_not_same(region1, region2) => {
            note_and_explain_region(cx, ~"", region1, ~"...");
            note_and_explain_region(cx, ~"...is not the same lifetime as ",
                                    region2, ~"");
        }
        terr_regions_no_overlap(region1, region2) => {
            note_and_explain_region(cx, ~"", region1, ~"...");
            note_and_explain_region(cx, ~"...does not overlap ",
                                    region2, ~"");
        }
        _ => {}
    }
}

fn def_has_ty_params(def: ast::def) -> bool {
    match def {
      ast::def_fn(_, _) | ast::def_variant(_, _) | ast::def_class(_, _)
        => true,
      _ => false
    }
}

fn store_trait_methods(cx: ctxt, id: ast::node_id, ms: @~[method]) {
    cx.trait_method_cache.insert(ast_util::local_def(id), ms);
}

fn provided_trait_methods(cx: ctxt, id: ast::def_id) -> ~[@ast::method] {
    if is_local(id) {
        match cx.items.find(id.node) {
            Some(ast_map::node_item(@{node: item_trait(_, _, ms),_}, _)) =>
                match ast_util::split_trait_methods(ms) {
                   (_, p) => p
                },
            _ => cx.sess.bug(#fmt("provided_trait_methods: %? is not a trait",
                                  id))
        }
    }
    else {
        // FIXME #2794: default methods for traits don't work cross-crate
        ~[]
    }
}


fn trait_methods(cx: ctxt, id: ast::def_id) -> @~[method] {
    match cx.trait_method_cache.find(id) {
      // Local traits are supposed to have been added explicitly.
      Some(ms) => ms,
      _ => {
        // If the lookup in trait_method_cache fails, assume that the trait
        // method we're trying to look up is in a different crate, and look
        // for it there.
        assert id.crate != ast::local_crate;
        let result = csearch::get_trait_methods(cx, id);

        // Store the trait method in the local trait_method_cache so that
        // future lookups succeed.
        cx.trait_method_cache.insert(id, result);
        result
      }
    }
}

/*
  Could this return a list of (def_id, substs) pairs?
 */
fn impl_traits(cx: ctxt, id: ast::def_id) -> ~[t] {
    if id.crate == ast::local_crate {
        debug!("(impl_traits) searching for trait impl %?", id);
        match cx.items.find(id.node) {
           Some(ast_map::node_item(@{
                        node: ast::item_impl(_, opt_trait, _, _),
                        _},
                    _)) => {

               do option::map_default(&opt_trait, ~[]) |trait_ref| {
                       ~[node_id_to_type(cx, trait_ref.ref_id)]
                   }
           }
           Some(ast_map::node_item(@{node: ast::item_class(sd,_),
                           _},_)) => {
               do vec::map(sd.traits) |trait_ref| {
                    node_id_to_type(cx, trait_ref.ref_id)
                }
           }
           _ => ~[]
        }
    } else {
        csearch::get_impl_traits(cx, id)
    }
}

fn ty_to_def_id(ty: t) -> Option<ast::def_id> {
    match get(ty).sty {
      ty_trait(id, _, _) | ty_class(id, _) | ty_enum(id, _) => Some(id),
      _ => None
    }
}

// Enum information
type variant_info = @{args: ~[t], ctor_ty: t, name: ast::ident,
                      id: ast::def_id, disr_val: int};

fn substd_enum_variants(cx: ctxt,
                        id: ast::def_id,
                        substs: &substs) -> ~[variant_info] {
    do vec::map(*enum_variants(cx, id)) |variant_info| {
        let substd_args = vec::map(variant_info.args,
                                   |aty| subst(cx, substs, *aty));

        let substd_ctor_ty = subst(cx, substs, variant_info.ctor_ty);

        @{args: substd_args, ctor_ty: substd_ctor_ty, ..**variant_info}
    }
}

fn item_path_str(cx: ctxt, id: ast::def_id) -> ~str {
    ast_map::path_to_str(item_path(cx, id), cx.sess.parse_sess.interner)
}

/* If class_id names a class with a dtor, return Some(the dtor's id).
   Otherwise return none. */
fn ty_dtor(cx: ctxt, class_id: def_id) -> Option<def_id> {
    if is_local(class_id) {
       match cx.items.find(class_id.node) {
           Some(ast_map::node_item(@{
               node: ast::item_class(@{ dtor: Some(dtor), _ }, _),
               _
           }, _)) =>
               Some(local_def(dtor.node.id)),
           _ =>
               None
       }
    }
    else {
      csearch::class_dtor(cx.sess.cstore, class_id)
    }
}

fn has_dtor(cx: ctxt, class_id: def_id) -> bool {
    ty_dtor(cx, class_id).is_some()
}

fn item_path(cx: ctxt, id: ast::def_id) -> ast_map::path {
    if id.crate != ast::local_crate {
        csearch::get_item_path(cx, id)
    } else {
        let node = cx.items.get(id.node);
        match node {
          ast_map::node_item(item, path) => {
            let item_elt = match item.node {
              item_mod(_) | item_foreign_mod(_) => {
                ast_map::path_mod(item.ident)
              }
              _ => {
                ast_map::path_name(item.ident)
              }
            };
            vec::append_one(*path, item_elt)
          }

          ast_map::node_foreign_item(nitem, _, path) => {
            vec::append_one(*path, ast_map::path_name(nitem.ident))
          }

          ast_map::node_method(method, _, path) => {
            vec::append_one(*path, ast_map::path_name(method.ident))
          }
          ast_map::node_trait_method(trait_method, _, path) => {
            let method = ast_util::trait_method_to_ty_method(*trait_method);
            vec::append_one(*path, ast_map::path_name(method.ident))
          }

          ast_map::node_variant(variant, _, path) => {
            vec::append_one(vec::init(*path),
                            ast_map::path_name(variant.node.name))
          }

          ast_map::node_ctor(nm, _, _, _, path) => {
            vec::append_one(*path, ast_map::path_name(nm))
          }
          ast_map::node_dtor(_, _, _, path) => {
            vec::append_one(*path, ast_map::path_name(
                syntax::parse::token::special_idents::literally_dtor))
          }

          ast_map::node_stmt(*) | ast_map::node_expr(*) |
          ast_map::node_arg(*) | ast_map::node_local(*) |
          ast_map::node_export(*) | ast_map::node_block(*) => {
            cx.sess.bug(fmt!("cannot find item_path for node %?", node));
          }
        }
    }
}

fn enum_is_univariant(cx: ctxt, id: ast::def_id) -> bool {
    vec::len(*enum_variants(cx, id)) == 1u
}

fn type_is_empty(cx: ctxt, t: t) -> bool {
    match ty::get(t).sty {
       ty_enum(did, _) => (*enum_variants(cx, did)).is_empty(),
       _ => false
     }
}

fn enum_variants(cx: ctxt, id: ast::def_id) -> @~[variant_info] {
    match cx.enum_var_cache.find(id) {
      Some(variants) => return variants,
      _ => { /* fallthrough */ }
    }

    let result = if ast::local_crate != id.crate {
        @csearch::get_enum_variants(cx, id)
    } else {
        /*
          Although both this code and check_enum_variants in typeck/check
          call eval_const_expr, it should never get called twice for the same
          expr, since check_enum_variants also updates the enum_var_cache
         */
        match cx.items.get(id.node) {
          ast_map::node_item(@{node: ast::item_enum(enum_definition, _), _},
                             _) => {
            let variants = enum_definition.variants;
            let mut disr_val = -1;
            @vec::map(variants, |variant| {
                match variant.node.kind {
                    ast::tuple_variant_kind(args) => {
                        let ctor_ty = node_id_to_type(cx, variant.node.id);
                        let arg_tys = {
                            if vec::len(args) > 0u {
                                ty_fn_args(ctor_ty).map(|a| a.ty)
                            } else {
                                ~[]
                            }
                        };
                        match variant.node.disr_expr {
                          Some (ex) => {
                            // FIXME: issue #1417
                            disr_val = match const_eval::eval_const_expr(cx,
                                                                         ex) {
                              const_eval::const_int(val) => val as int,
                              _ => cx.sess.bug(~"tag_variants: bad disr expr")
                            }
                          }
                          _ => disr_val += 1
                        }
                        @{args: arg_tys,
                          ctor_ty: ctor_ty,
                          name: variant.node.name,
                          id: ast_util::local_def(variant.node.id),
                          disr_val: disr_val
                         }
                    }
                    ast::struct_variant_kind(_) => {
                        fail ~"struct variant kinds unimpl in enum_variants"
                    }
                    ast::enum_variant_kind(_) => {
                        fail ~"enum variant kinds unimpl in enum_variants"
                    }
                }
            })
          }
          _ => cx.sess.bug(~"tag_variants: id not bound to an enum")
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
        if variant.id == variant_id { return variant; }
        i += 1u;
    }
    cx.sess.bug(~"enum_variant_with_id(): no variant exists with that ID");
}


// If the given item is in an external crate, looks up its type and adds it to
// the type cache. Returns the type parameters and type.
fn lookup_item_type(cx: ctxt, did: ast::def_id) -> ty_param_bounds_and_ty {
    match cx.tcache.find(did) {
      Some(tpt) => return tpt,
      None => {
        // The item is in this crate. The caller should have added it to the
        // type cache already
        assert did.crate != ast::local_crate;
        let tyt = csearch::get_type(cx, did);
        cx.tcache.insert(did, tyt);
        return tyt;
      }
    }
}

// Look up a field ID, whether or not it's local
// Takes a list of type substs in case the class is generic
fn lookup_field_type(tcx: ctxt, class_id: def_id, id: def_id,
                     substs: &substs) -> ty::t {

    let t = if id.crate == ast::local_crate {
        node_id_to_type(tcx, id.node)
    }
    else {
        match tcx.tcache.find(id) {
           Some(tpt) => tpt.ty,
           None => {
               let tpt = csearch::get_field_type(tcx, class_id, id);
               tcx.tcache.insert(id, tpt);
               tpt.ty
           }
        }
    };
    subst(tcx, substs, t)
}

// Look up the list of field names and IDs for a given class
// Fails if the id is not bound to a class.
fn lookup_class_fields(cx: ctxt, did: ast::def_id) -> ~[field_ty] {
  if did.crate == ast::local_crate {
    match cx.items.find(did.node) {
       Some(ast_map::node_item(i,_)) => {
         match i.node {
            ast::item_class(struct_def, _) => {
               class_field_tys(struct_def.fields)
            }
            _ => cx.sess.bug(~"class ID bound to non-class")
         }
       }
       _ => {
           cx.sess.bug(
               fmt!("class ID not bound to an item: %s",
                    ast_map::node_id_to_str(cx.items, did.node,
                                            cx.sess.parse_sess.interner)));
       }
    }
        }
  else {
        return csearch::get_class_fields(cx, did);
    }
}

fn lookup_class_field(cx: ctxt, parent: ast::def_id, field_id: ast::def_id)
    -> field_ty {
    match vec::find(lookup_class_fields(cx, parent),
                 |f| f.id.node == field_id.node) {
        Some(t) => t,
        None => cx.sess.bug(~"class ID not found in parent's fields")
    }
}

pure fn is_public(f: field_ty) -> bool {
    // XXX: This is wrong.
    match f.vis {
        public | inherited => true,
        private => false
    }
}

/* Given a class def_id and a method name, return the method's
 def_id. Needed so we can do static dispatch for methods
 Doesn't care about the method's privacy. (It's assumed that
 the caller already checked that.)
*/
fn lookup_class_method_by_name(cx:ctxt, did: ast::def_id, name: ident,
                               sp: span) -> def_id {

    // Look up the list of method names and IDs for a given class
    // Fails if the id is not bound to a class.
    fn lookup_class_method_ids(cx: ctxt, did: ast::def_id)
        -> ~[{name: ident, id: node_id, vis: visibility}] {

        assert is_local(did);
        match cx.items.find(did.node) {
          Some(ast_map::node_item(@{
            node: item_class(struct_def, _), _
          }, _)) => {
            vec::map(struct_def.methods, |m| {name: m.ident,
                                              id: m.id,
                                              vis: m.vis})
          }
          _ => {
            cx.sess.bug(~"lookup_class_method_ids: id not bound to a class");
          }
        }
    }

    if is_local(did) {
       let ms = lookup_class_method_ids(cx, did);
        for ms.each |m| {
            if m.name == name {
                return ast_util::local_def(m.id);
            }
       }
       cx.sess.span_fatal(sp, fmt!("Class doesn't have a method \
           named %s", cx.sess.str_of(name)));
    }
    else {
      csearch::get_class_method(cx.sess.cstore, did, name)
    }
}

fn class_field_tys(fields: ~[@struct_field]) -> ~[field_ty] {
    let mut rslt = ~[];
    for fields.each |field| {
        match field.node.kind {
            named_field(ident, mutability, visibility) => {
                rslt.push({ident: ident,
                           id: ast_util::local_def(field.node.id),
                           vis: visibility,
                           mutability: mutability});
            }
            unnamed_field => {}
       }
    }
    rslt
}

// Return a list of fields corresponding to the class's items
// (as if the class was a record). trans uses this
// Takes a list of substs with which to instantiate field types
// Keep in mind that this function reports that all fields are
// mutable, regardless of how they were declared. It's meant to
// be used in trans.
fn class_items_as_mutable_fields(cx:ctxt,
                                 did: ast::def_id,
                                 substs: &substs) -> ~[field] {
    class_item_fields(cx, did, substs, |_mt| m_mutbl)
}

// Same as class_items_as_mutable_fields, but doesn't change
// mutability.
fn class_items_as_fields(cx:ctxt,
                         did: ast::def_id,
                         substs: &substs) -> ~[field] {
    class_item_fields(cx, did, substs, |mt| match mt {
      class_mutable => m_mutbl,
        class_immutable => m_imm })
}


fn class_item_fields(cx:ctxt,
                     did: ast::def_id,
                     substs: &substs,
                     frob_mutability: fn(class_mutability) -> mutability)
    -> ~[field] {
    let mut rslt = ~[];
    for lookup_class_fields(cx, did).each |f| {
       // consider all instance vars mut, because the
       // constructor may mutate all vars
       rslt.push({ident: f.ident, mt:
               {ty: lookup_field_type(cx, did, f.id, substs),
                    mutbl: frob_mutability(f.mutability)}});
    }
    rslt
}

fn is_binopable(_cx: ctxt, ty: t, op: ast::binop) -> bool {
    const tycat_other: int = 0;
    const tycat_bool: int = 1;
    const tycat_int: int = 2;
    const tycat_float: int = 3;
    const tycat_struct: int = 4;
    const tycat_bot: int = 5;

    const opcat_add: int = 0;
    const opcat_sub: int = 1;
    const opcat_mult: int = 2;
    const opcat_shift: int = 3;
    const opcat_rel: int = 4;
    const opcat_eq: int = 5;
    const opcat_bit: int = 6;
    const opcat_logic: int = 7;

    fn opcat(op: ast::binop) -> int {
        match op {
          ast::add => opcat_add,
          ast::subtract => opcat_sub,
          ast::mul => opcat_mult,
          ast::div => opcat_mult,
          ast::rem => opcat_mult,
          ast::and => opcat_logic,
          ast::or => opcat_logic,
          ast::bitxor => opcat_bit,
          ast::bitand => opcat_bit,
          ast::bitor => opcat_bit,
          ast::shl => opcat_shift,
          ast::shr => opcat_shift,
          ast::eq => opcat_eq,
          ast::ne => opcat_eq,
          ast::lt => opcat_rel,
          ast::le => opcat_rel,
          ast::ge => opcat_rel,
          ast::gt => opcat_rel
        }
    }

    fn tycat(ty: t) -> int {
        match get(ty).sty {
          ty_bool => tycat_bool,
          ty_int(_) | ty_uint(_) | ty_infer(IntVar(_)) => tycat_int,
          ty_float(_) => tycat_float,
          ty_rec(_) | ty_tup(_) | ty_enum(_, _) => tycat_struct,
          ty_bot => tycat_bot,
          _ => tycat_other
        }
    }

    const t: bool = true;
    const f: bool = false;

    let tbl = ~[
    /*.          add,     shift,   bit
      .             sub,     rel,     logic
      .                mult,    eq,         */
    /*other*/   ~[f, f, f, f, f, f, f, f],
    /*bool*/    ~[f, f, f, f, t, t, t, t],
    /*int*/     ~[t, t, t, t, t, t, t, f],
    /*float*/   ~[t, t, t, f, t, t, f, f],
    /*bot*/     ~[f, f, f, f, f, f, f, f],
    /*struct*/  ~[t, t, t, t, f, f, t, t]];

    return tbl[tycat(ty)][opcat(op)];
}

fn ty_params_to_tys(tcx: ty::ctxt, tps: ~[ast::ty_param]) -> ~[t] {
    vec::from_fn(tps.len(), |i| {
                ty::mk_param(tcx, i, ast_util::local_def(tps[i].id))
        })
}

/// Returns an equivalent type with all the typedefs and self regions removed.
fn normalize_ty(cx: ctxt, t: t) -> t {
    fn normalize_mt(cx: ctxt, mt: mt) -> mt {
        { ty: normalize_ty(cx, mt.ty), mutbl: mt.mutbl }
    }
    fn normalize_vstore(vstore: vstore) -> vstore {
        match vstore {
            vstore_fixed(*) | vstore_uniq | vstore_box => vstore,
            vstore_slice(_) => vstore_slice(re_static)
        }
    }

    match cx.normalized_cache.find(t) {
      Some(t) => return t,
      None => ()
    }

    let t = match get(t).sty {
        ty_evec(mt, vstore) =>
            // This type has a vstore. Get rid of it
            mk_evec(cx, normalize_mt(cx, mt), normalize_vstore(vstore)),

        ty_estr(vstore) =>
            // This type has a vstore. Get rid of it
            mk_estr(cx, normalize_vstore(vstore)),

        ty_rptr(_, mt) =>
            // This type has a region. Get rid of it
            mk_rptr(cx, re_static, normalize_mt(cx, mt)),

        ty_fn(ref fn_ty) => {
            let proto = match fn_ty.meta.proto {
                proto_bare => proto_bare,
                proto_vstore(vstore) => proto_vstore(normalize_vstore(vstore))
            };
            mk_fn(cx, FnTyBase {
                meta: FnMeta {proto: proto, ..fn_ty.meta},
                sig: fn_ty.sig
            })
        }

        ty_enum(did, r) =>
            match r.self_r {
                Some(_) =>
                    // This enum has a self region. Get rid of it
                    mk_enum(cx, did,
                            {self_r: None, self_ty: None, tps: r.tps}),
                None =>
                    t
            },

        ty_class(did, r) =>
            match r.self_r {
              Some(_) =>
                // Ditto.
                mk_class(cx, did, {self_r: None, self_ty: None, tps: r.tps}),
              None =>
                t
            },

        _ =>
            t
    };

    // FIXME #2187: This also reduced int types to their compatible machine
    // types, which isn't necessary after #2187
    let t = mk_t(cx, mach_sty(cx.sess.targ_cfg, t));

    let sty = fold_sty(&get(t).sty, |t| { normalize_ty(cx, t) });
    let t_norm = mk_t(cx, sty);
    cx.normalized_cache.insert(t, t_norm);
    return t_norm;
}

// Returns the repeat count for a repeating vector expression.
fn eval_repeat_count(tcx: ctxt, count_expr: @ast::expr, span: span) -> uint {
    match const_eval::eval_const_expr(tcx, count_expr) {
        const_eval::const_int(count) => return count as uint,
        const_eval::const_uint(count) => return count as uint,
        const_eval::const_float(count) => {
            tcx.sess.span_err(span,
                              ~"expected signed or unsigned integer for \
                                repeat count but found float");
            return count as uint;
        }
        const_eval::const_str(_) => {
            tcx.sess.span_err(span,
                              ~"expected signed or unsigned integer for \
                                repeat count but found string");
            return 0;
        }
        const_eval::const_bool(_) => {
            tcx.sess.span_err(span,
                              ~"expected signed or unsigned integer for \
                                repeat count but found boolean");
            return 0;
        }

    }
}

pure fn is_blockish(proto: fn_proto) -> bool {
    match proto {
        proto_vstore(vstore_slice(_)) =>
            true,
        proto_vstore(vstore_box) | proto_vstore(vstore_uniq) | proto_bare =>
            false,
        proto_vstore(vstore_fixed(_)) =>
            fail ~"fixed vstore not allowed here"
    }
}

// Determine what purity to check a nested function under
pure fn determine_inherited_purity(parent_purity: ast::purity,
                                   child_purity: ast::purity,
                                   child_proto: ty::fn_proto) -> ast::purity {
    // If the closure is a stack closure and hasn't had some non-standard
    // purity inferred for it, then check it under its parent's purity.
    // Otherwise, use its own
    if ty::is_blockish(child_proto) && child_purity == ast::impure_fn {
        parent_purity
    } else { child_purity }
}

impl mt : cmp::Eq {
    pure fn eq(other: &mt) -> bool {
        self.ty == (*other).ty && self.mutbl == (*other).mutbl
    }
    pure fn ne(other: &mt) -> bool { !self.eq(other) }
}

impl arg : cmp::Eq {
    pure fn eq(other: &arg) -> bool {
        self.mode == (*other).mode && self.ty == (*other).ty
    }
    pure fn ne(other: &arg) -> bool { !self.eq(other) }
}

impl field : cmp::Eq {
    pure fn eq(other: &field) -> bool {
        self.ident == (*other).ident && self.mt == (*other).mt
    }
    pure fn ne(other: &field) -> bool { !self.eq(other) }
}

impl vstore : cmp::Eq {
    pure fn eq(other: &vstore) -> bool {
        match self {
            vstore_fixed(e0a) => {
                match (*other) {
                    vstore_fixed(e0b) => e0a == e0b,
                    _ => false
                }
            }
            vstore_uniq => {
                match (*other) {
                    vstore_uniq => true,
                    _ => false
                }
            }
            vstore_box => {
                match (*other) {
                    vstore_box => true,
                    _ => false
                }
            }
            vstore_slice(e0a) => {
                match (*other) {
                    vstore_slice(e0b) => e0a == e0b,
                    _ => false
                }
            }
        }
    }
    pure fn ne(other: &vstore) -> bool { !self.eq(other) }
}

impl FnMeta : cmp::Eq {
    pure fn eq(other: &FnMeta) -> bool {
        self.purity == (*other).purity &&
        self.proto == (*other).proto &&
        self.bounds == (*other).bounds &&
        self.ret_style == (*other).ret_style
    }
    pure fn ne(other: &FnMeta) -> bool { !self.eq(other) }
}

impl FnSig : cmp::Eq {
    pure fn eq(other: &FnSig) -> bool {
        self.inputs == (*other).inputs &&
        self.output == (*other).output
    }
    pure fn ne(other: &FnSig) -> bool { !self.eq(other) }
}

impl<M: cmp::Eq> FnTyBase<M> : cmp::Eq {
    pure fn eq(other: &FnTyBase<M>) -> bool {
        self.meta == (*other).meta && self.sig == (*other).sig
    }
    pure fn ne(other: &FnTyBase<M>) -> bool { !self.eq(other) }
}

impl TyVid : cmp::Eq {
    pure fn eq(other: &TyVid) -> bool { *self == *(*other) }
    pure fn ne(other: &TyVid) -> bool { *self != *(*other) }
}

impl IntVid : cmp::Eq {
    pure fn eq(other: &IntVid) -> bool { *self == *(*other) }
    pure fn ne(other: &IntVid) -> bool { *self != *(*other) }
}

impl FnVid : cmp::Eq {
    pure fn eq(other: &FnVid) -> bool { *self == *(*other) }
    pure fn ne(other: &FnVid) -> bool { *self != *(*other) }
}

impl RegionVid : cmp::Eq {
    pure fn eq(other: &RegionVid) -> bool { *self == *(*other) }
    pure fn ne(other: &RegionVid) -> bool { *self != *(*other) }
}

impl region : cmp::Eq {
    pure fn eq(other: &region) -> bool {
        match self {
            re_bound(e0a) => {
                match (*other) {
                    re_bound(e0b) => e0a == e0b,
                    _ => false
                }
            }
            re_free(e0a, e1a) => {
                match (*other) {
                    re_free(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            re_scope(e0a) => {
                match (*other) {
                    re_scope(e0b) => e0a == e0b,
                    _ => false
                }
            }
            re_static => {
                match (*other) {
                    re_static => true,
                    _ => false
                }
            }
            re_var(e0a) => {
                match (*other) {
                    re_var(e0b) => e0a == e0b,
                    _ => false
                }
            }
        }
    }
    pure fn ne(other: &region) -> bool { !self.eq(other) }
}

impl bound_region : cmp::Eq {
    pure fn eq(other: &bound_region) -> bool {
        match self {
            br_self => {
                match (*other) {
                    br_self => true,
                    _ => false
                }
            }
            br_anon(e0a) => {
                match (*other) {
                    br_anon(e0b) => e0a == e0b,
                    _ => false
                }
            }
            br_named(e0a) => {
                match (*other) {
                    br_named(e0b) => e0a == e0b,
                    _ => false
                }
            }
            br_cap_avoid(e0a, e1a) => {
                match (*other) {
                    br_cap_avoid(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
        }
    }
    pure fn ne(other: &bound_region) -> bool { !self.eq(other) }
}

impl substs : cmp::Eq {
    pure fn eq(other: &substs) -> bool {
        self.self_r == (*other).self_r &&
        self.self_ty == (*other).self_ty &&
        self.tps == (*other).tps
    }
    pure fn ne(other: &substs) -> bool { !self.eq(other) }
}

impl InferTy : cmp::Eq {
    pure fn eq(other: &InferTy) -> bool {
        self.to_hash() == (*other).to_hash()
    }
    pure fn ne(other: &InferTy) -> bool { !self.eq(other) }
}

impl sty : cmp::Eq {
    pure fn eq(other: &sty) -> bool {
        match self {
            ty_nil => {
                match (*other) {
                    ty_nil => true,
                    _ => false
                }
            }
            ty_bot => {
                match (*other) {
                    ty_bot => true,
                    _ => false
                }
            }
            ty_bool => {
                match (*other) {
                    ty_bool => true,
                    _ => false
                }
            }
            ty_int(e0a) => {
                match (*other) {
                    ty_int(e0b) => e0a == e0b,
                    _ => false
                }
            }
            ty_uint(e0a) => {
                match (*other) {
                    ty_uint(e0b) => e0a == e0b,
                    _ => false
                }
            }
            ty_float(e0a) => {
                match (*other) {
                    ty_float(e0b) => e0a == e0b,
                    _ => false
                }
            }
            ty_estr(e0a) => {
                match (*other) {
                    ty_estr(e0b) => e0a == e0b,
                    _ => false
                }
            }
            ty_enum(e0a, e1a) => {
                match (*other) {
                    ty_enum(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            ty_box(e0a) => {
                match (*other) {
                    ty_box(e0b) => e0a == e0b,
                    _ => false
                }
            }
            ty_uniq(e0a) => {
                match (*other) {
                    ty_uniq(e0b) => e0a == e0b,
                    _ => false
                }
            }
            ty_evec(e0a, e1a) => {
                match (*other) {
                    ty_evec(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            ty_ptr(e0a) => {
                match (*other) {
                    ty_ptr(e0b) => e0a == e0b,
                    _ => false
                }
            }
            ty_rptr(e0a, e1a) => {
                match (*other) {
                    ty_rptr(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            ty_rec(e0a) => {
                match (*other) {
                    ty_rec(e0b) => e0a == e0b,
                    _ => false
                }
            }
            ty_fn(e0a) => {
                match (*other) {
                    ty_fn(e0b) => e0a == e0b,
                    _ => false
                }
            }
            ty_trait(e0a, e1a, e2a) => {
                match (*other) {
                    ty_trait(e0b, e1b, e2b) =>
                        e0a == e0b && e1a == e1b && e2a == e2b,
                    _ => false
                }
            }
            ty_class(e0a, e1a) => {
                match (*other) {
                    ty_class(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            ty_tup(e0a) => {
                match (*other) {
                    ty_tup(e0b) => e0a == e0b,
                    _ => false
                }
            }
            ty_infer(e0a) => {
                match (*other) {
                    ty_infer(e0b) => e0a == e0b,
                    _ => false
                }
            }
            ty_param(e0a) => {
                match (*other) {
                    ty_param(e0b) => e0a == e0b,
                    _ => false
                }
            }
            ty_self => {
                match (*other) {
                    ty_self => true,
                    _ => false
                }
            }
            ty_type => {
                match (*other) {
                    ty_type => true,
                    _ => false
                }
            }
            ty_opaque_box => {
                match (*other) {
                    ty_opaque_box => true,
                    _ => false
                }
            }
            ty_opaque_closure_ptr(e0a) => {
                match (*other) {
                    ty_opaque_closure_ptr(e0b) => e0a == e0b,
                    _ => false
                }
            }
            ty_unboxed_vec(e0a) => {
                match (*other) {
                    ty_unboxed_vec(e0b) => e0a == e0b,
                    _ => false
                }
            }
        }
    }
    pure fn ne(other: &sty) -> bool { !self.eq(other) }
}

impl param_bound : cmp::Eq {
    pure fn eq(other: &param_bound) -> bool {
        match self {
            bound_copy => {
                match (*other) {
                    bound_copy => true,
                    _ => false
                }
            }
            bound_owned => {
                match (*other) {
                    bound_owned => true,
                    _ => false
                }
            }
            bound_send => {
                match (*other) {
                    bound_send => true,
                    _ => false
                }
            }
            bound_const => {
                match (*other) {
                    bound_const => true,
                    _ => false
                }
            }
            bound_trait(e0a) => {
                match (*other) {
                    bound_trait(e0b) => e0a == e0b,
                    _ => false
                }
            }
        }
    }
    pure fn ne(other: &param_bound) -> bool { !self.eq(other) }
}

impl kind : cmp::Eq {
    pure fn eq(other: &kind) -> bool { *self == *(*other) }
    pure fn ne(other: &kind) -> bool { *self != *(*other) }
}


// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
