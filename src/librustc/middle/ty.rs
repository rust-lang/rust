// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use driver::session;
use metadata::csearch;
use metadata;
use middle::const_eval;
use middle::freevars;
use middle::lint::{get_lint_level, allow};
use middle::lint;
use middle::resolve::{Impl, MethodInfo};
use middle::resolve;
use middle::ty;
use middle::typeck;
use middle;
use util::ppaux::{note_and_explain_region, bound_region_to_str};
use util::ppaux::{region_to_str, explain_region, vstore_to_str};
use util::ppaux::{trait_store_to_str, ty_to_str, tys_to_str};
use util::common::{indenter};

use core::cast;
use core::cmp;
use core::ops;
use core::ptr::to_unsafe_ptr;
use core::result::Result;
use core::result;
use core::to_bytes;
use core::uint;
use core::vec;
use core::hashmap::linear::{LinearMap, LinearSet};
use std::smallintmap::SmallIntMap;
use syntax::ast::*;
use syntax::ast_util::{is_local, local_def};
use syntax::ast_util;
use syntax::codemap::span;
use syntax::codemap;
use syntax::print::pprust;
use syntax::{ast, ast_map};
use syntax::opt_vec::OptVec;
use syntax::opt_vec;
use syntax;

// Data types

// Note: after typeck, you should use resolved_mode() to convert this mode
// into an rmode, which will take into account the results of mode inference.
#[deriving(Eq)]
pub struct arg {
    mode: ast::mode,
    ty: t
}

#[deriving(Eq)]
pub struct field {
    ident: ast::ident,
    mt: mt
}

pub type param_bounds = @~[param_bound];

pub struct method {
    ident: ast::ident,
    tps: @~[param_bounds],
    fty: BareFnTy,
    self_ty: ast::self_ty_,
    vis: ast::visibility,
    def_id: ast::def_id
}

#[deriving(Eq)]
pub struct mt {
    ty: t,
    mutbl: ast::mutability,
}

#[auto_encode]
#[auto_decode]
#[deriving(Eq)]
pub enum vstore {
    vstore_fixed(uint),
    vstore_uniq,
    vstore_box,
    vstore_slice(Region)
}

#[auto_encode]
#[auto_decode]
#[deriving(Eq)]
pub enum TraitStore {
    BareTraitStore,             // a plain trait without a sigil
    BoxTraitStore,              // @Trait
    UniqTraitStore,             // ~Trait
    RegionTraitStore(Region),   // &Trait
}

pub struct field_ty {
  ident: ident,
  id: def_id,
  vis: ast::visibility,
  mutability: ast::struct_mutability,
}

// Contains information needed to resolve types and (in the future) look up
// the types of AST nodes.
#[deriving(Eq)]
pub struct creader_cache_key {
    cnum: int,
    pos: uint,
    len: uint
}

type creader_cache = @mut LinearMap<creader_cache_key, t>;

impl to_bytes::IterBytes for creader_cache_key {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        to_bytes::iter_bytes_3(&self.cnum, &self.pos, &self.len, lsb0, f);
    }
}

struct intern_key {
    sty: *sty,
    o_def_id: Option<ast::def_id>
}

// NB: Do not replace this with #[deriving(Eq)]. The automatically-derived
// implementation will not recurse through sty and you will get stack
// exhaustion.
impl cmp::Eq for intern_key {
    fn eq(&self, other: &intern_key) -> bool {
        unsafe {
            *self.sty == *other.sty && self.o_def_id == other.o_def_id
        }
    }
    fn ne(&self, other: &intern_key) -> bool {
        !self.eq(other)
    }
}

impl to_bytes::IterBytes for intern_key {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        unsafe {
            to_bytes::iter_bytes_2(&*self.sty, &self.o_def_id, lsb0, f);
        }
    }
}

pub enum ast_ty_to_ty_cache_entry {
    atttce_unresolved,  /* not resolved yet */
    atttce_resolved(t)  /* resolved to a type, irrespective of region */
}

pub type opt_region_variance = Option<region_variance>;

#[auto_encode]
#[auto_decode]
#[deriving(Eq)]
pub enum region_variance { rv_covariant, rv_invariant, rv_contravariant }

#[auto_encode]
#[auto_decode]
pub enum AutoAdjustment {
    AutoAddEnv(ty::Region, ast::Sigil),
    AutoDerefRef(AutoDerefRef)
}

#[auto_encode]
#[auto_decode]
pub struct AutoDerefRef {
    autoderefs: uint,
    autoref: Option<AutoRef>
}

#[auto_encode]
#[auto_decode]
pub struct AutoRef {
    kind: AutoRefKind,
    region: Region,
    mutbl: ast::mutability
}

#[auto_encode]
#[auto_decode]
pub enum AutoRefKind {
    /// Convert from T to &T
    AutoPtr,

    /// Convert from @[]/~[] to &[] (or str)
    AutoBorrowVec,

    /// Convert from @[]/~[] to &&[] (or str)
    AutoBorrowVecRef,

    /// Convert from @fn()/~fn() to &fn()
    AutoBorrowFn
}

// Stores information about provided methods (a.k.a. default methods) in
// implementations.
//
// This is a map from ID of each implementation to the method info and trait
// method ID of each of the default methods belonging to the trait that that
// implementation implements.
pub type ProvidedMethodsMap = @mut LinearMap<def_id,@mut ~[@ProvidedMethodInfo]>;

// Stores the method info and definition ID of the associated trait method for
// each instantiation of each provided method.
pub struct ProvidedMethodInfo {
    method_info: @MethodInfo,
    trait_method_def_id: def_id
}

pub struct ProvidedMethodSource {
    method_id: ast::def_id,
    impl_id: ast::def_id
}

pub struct InstantiatedTraitRef {
    def_id: ast::def_id,
    tpt: ty_param_substs_and_ty
}

pub type ctxt = @ctxt_;

struct ctxt_ {
    diag: @syntax::diagnostic::span_handler,
    interner: @mut LinearMap<intern_key, t_box>,
    next_id: @mut uint,
    vecs_implicitly_copyable: bool,
    legacy_modes: bool,
    cstore: @mut metadata::cstore::CStore,
    sess: session::Session,
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
    node_type_substs: @mut LinearMap<node_id, ~[t]>,

    items: ast_map::map,
    intrinsic_defs: @mut LinearMap<ast::ident, (ast::def_id, t)>,
    freevars: freevars::freevar_map,
    tcache: type_cache,
    rcache: creader_cache,
    ccache: constness_cache,
    short_names_cache: @mut LinearMap<t, @~str>,
    needs_unwind_cleanup_cache: @mut LinearMap<t, bool>,
    tc_cache: @mut LinearMap<uint, TypeContents>,
    ast_ty_to_ty_cache: @mut LinearMap<node_id, ast_ty_to_ty_cache_entry>,
    enum_var_cache: @mut LinearMap<def_id, @~[VariantInfo]>,
    trait_method_cache: @mut LinearMap<def_id, @~[method]>,
    ty_param_bounds: @mut LinearMap<ast::node_id, param_bounds>,
    inferred_modes: @mut LinearMap<ast::node_id, ast::mode>,
    adjustments: @mut LinearMap<ast::node_id, @AutoAdjustment>,
    normalized_cache: @mut LinearMap<t, t>,
    lang_items: middle::lang_items::LanguageItems,
    // A mapping from an implementation ID to the method info and trait
    // method ID of the provided (a.k.a. default) methods in the traits that
    // that implementation implements.
    provided_methods: ProvidedMethodsMap,
    provided_method_sources: @mut LinearMap<ast::def_id, ProvidedMethodSource>,
    supertraits: @mut LinearMap<ast::def_id, @~[InstantiatedTraitRef]>,

    // A mapping from the def ID of an enum or struct type to the def ID
    // of the method that implements its destructor. If the type is not
    // present in this map, it does not have a destructor. This map is
    // populated during the coherence phase of typechecking.
    destructor_for_type: @mut LinearMap<ast::def_id, ast::def_id>,

    // A method will be in this list if and only if it is a destructor.
    destructors: @mut LinearSet<ast::def_id>,

    // Maps a trait onto a mapping from self-ty to impl
    trait_impls: @mut LinearMap<ast::def_id, @mut LinearMap<t, @Impl>>
}

enum tbox_flag {
    has_params = 1,
    has_self = 2,
    needs_infer = 4,
    has_regions = 8,
    has_ty_err = 16,
    has_ty_bot = 32,

    // a meta-flag: subst may be required if the type has parameters, a self
    // type, or references bound regions
    needs_subst = 1 | 2 | 8
}

type t_box = @t_box_;

struct t_box_ {
    sty: sty,
    id: uint,
    flags: uint,
    o_def_id: Option<ast::def_id>
}

// To reduce refcounting cost, we're representing types as unsafe pointers
// throughout the compiler. These are simply casted t_box values. Use ty::get
// to cast them back to a box. (Without the cast, compiler performance suffers
// ~15%.) This does mean that a t value relies on the ctxt to keep its box
// alive, and using ty::get is unsafe when the ctxt is no longer alive.
enum t_opaque {}
pub type t = *t_opaque;

pub fn get(t: t) -> t_box {
    unsafe {
        let t2 = cast::reinterpret_cast::<t, t_box>(&t);
        let t3 = t2;
        cast::forget(t2);
        t3
    }
}

pub fn tbox_has_flag(tb: t_box, flag: tbox_flag) -> bool {
    (tb.flags & (flag as uint)) != 0u
}
pub fn type_has_params(t: t) -> bool {
    tbox_has_flag(get(t), has_params)
}
pub fn type_has_self(t: t) -> bool { tbox_has_flag(get(t), has_self) }
pub fn type_needs_infer(t: t) -> bool {
    tbox_has_flag(get(t), needs_infer)
}
pub fn type_has_regions(t: t) -> bool {
    tbox_has_flag(get(t), has_regions)
}
pub fn type_def_id(t: t) -> Option<ast::def_id> { get(t).o_def_id }
pub fn type_id(t: t) -> uint { get(t).id }

#[deriving(Eq)]
pub struct BareFnTy {
    purity: ast::purity,
    abi: Abi,
    sig: FnSig
}

#[deriving(Eq)]
pub struct ClosureTy {
    purity: ast::purity,
    sigil: ast::Sigil,
    onceness: ast::Onceness,
    region: Region,
    sig: FnSig
}

/**
 * Signature of a function type, which I have arbitrarily
 * decided to use to refer to the input/output types.
 *
 * - `lifetimes` is the list of region names bound in this fn.
 * - `inputs` is the list of arguments and their modes.
 * - `output` is the return type. */
#[deriving(Eq)]
pub struct FnSig {
    bound_lifetime_names: OptVec<ast::ident>,
    inputs: ~[arg],
    output: t
}

impl to_bytes::IterBytes for BareFnTy {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        to_bytes::iter_bytes_3(&self.purity, &self.abi, &self.sig, lsb0, f)
    }
}

impl to_bytes::IterBytes for ClosureTy {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        to_bytes::iter_bytes_5(&self.purity, &self.sigil, &self.onceness,
                               &self.region, &self.sig, lsb0, f)
    }
}

#[deriving(Eq)]
pub struct param_ty {
    idx: uint,
    def_id: def_id
}

impl to_bytes::IterBytes for param_ty {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        to_bytes::iter_bytes_2(&self.idx, &self.def_id, lsb0, f)
    }
}


/// Representation of regions:
#[auto_encode]
#[auto_decode]
#[deriving(Eq)]
pub enum Region {
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
    re_infer(InferRegion)
}

#[auto_encode]
#[auto_decode]
#[deriving(Eq)]
pub enum bound_region {
    /// The self region for structs, impls (&T in a type defn or &'self T)
    br_self,

    /// An anonymous region parameter for a given fn (&T)
    br_anon(uint),

    /// Named region parameters for functions (a in &'a T)
    br_named(ast::ident),

    /// Fresh bound identifiers created during GLB computations.
    br_fresh(uint),

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

type opt_region = Option<Region>;

/**
 * The type substs represents the kinds of things that can be substituted to
 * convert a polytype into a monotype.  Note however that substituting bound
 * regions other than `self` is done through a different mechanism:
 *
 * - `tps` represents the type parameters in scope.  They are indexed
 *   according to the order in which they were declared.
 *
 * - `self_r` indicates the region parameter `self` that is present on nominal
 *   types (enums, structs) declared as having a region parameter.  `self_r`
 *   should always be none for types that are not region-parameterized and
 *   Some(_) for types that are.  The only bound region parameter that should
 *   appear within a region-parameterized type is `self`.
 *
 * - `self_ty` is the type to which `self` should be remapped, if any.  The
 *   `self` type is rather funny in that it can only appear on traits and is
 *   always substituted away to the implementing type for a trait. */
#[deriving(Eq)]
pub struct substs {
    self_r: opt_region,
    self_ty: Option<ty::t>,
    tps: ~[t]
}

// NB: If you change this, you'll probably want to change the corresponding
// AST structure in libsyntax/ast.rs as well.
#[deriving(Eq)]
pub enum sty {
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
    ty_rptr(Region, mt),
    ty_bare_fn(BareFnTy),
    ty_closure(ClosureTy),
    ty_trait(def_id, substs, TraitStore),
    ty_struct(def_id, substs),
    ty_tup(~[t]),

    ty_param(param_ty), // type parameter
    ty_self(def_id), /* special, implicit `self` type parameter;
                      * def_id is the id of the trait */

    ty_infer(InferTy), // something used only during inference/typeck
    ty_err, // Also only used during inference/typeck, to represent
            // the type of an erroneous expression (helps cut down
            // on non-useful type error messages)

    // "Fake" types, used for trans purposes
    ty_type, // type_desc*
    ty_opaque_box, // used by monomorphizer to represent any @ box
    ty_opaque_closure_ptr(Sigil), // ptr to env for &fn, @fn, ~fn
    ty_unboxed_vec(mt),
}

#[deriving(Eq)]
pub enum IntVarValue {
    IntType(ast::int_ty),
    UintType(ast::uint_ty),
}

pub enum terr_vstore_kind {
    terr_vec, terr_str, terr_fn, terr_trait
}

pub struct expected_found<T> {
    expected: T,
    found: T
}

// Data structures used in type unification
pub enum type_err {
    terr_mismatch,
    terr_purity_mismatch(expected_found<purity>),
    terr_onceness_mismatch(expected_found<Onceness>),
    terr_abi_mismatch(expected_found<ast::Abi>),
    terr_mutability,
    terr_sigil_mismatch(expected_found<ast::Sigil>),
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
    terr_regions_does_not_outlive(Region, Region),
    terr_regions_not_same(Region, Region),
    terr_regions_no_overlap(Region, Region),
    terr_regions_insufficiently_polymorphic(bound_region, Region),
    terr_regions_overly_polymorphic(bound_region, Region),
    terr_vstores_differ(terr_vstore_kind, expected_found<vstore>),
    terr_trait_stores_differ(terr_vstore_kind, expected_found<TraitStore>),
    terr_in_field(@type_err, ast::ident),
    terr_sorts(expected_found<t>),
    terr_self_substs,
    terr_integer_as_char,
    terr_int_mismatch(expected_found<IntVarValue>),
    terr_float_mismatch(expected_found<ast::float_ty>)
}

#[deriving(Eq)]
pub enum param_bound {
    bound_copy,
    bound_durable,
    bound_owned,
    bound_const,
    bound_trait(t),
}

#[deriving(Eq)]
pub struct TyVid(uint);

#[deriving(Eq)]
pub struct IntVid(uint);

#[deriving(Eq)]
pub struct FloatVid(uint);

#[deriving(Eq)]
#[auto_encode]
#[auto_decode]
pub struct RegionVid {
    id: uint
}

#[deriving(Eq)]
pub enum InferTy {
    TyVar(TyVid),
    IntVar(IntVid),
    FloatVar(FloatVid)
}

impl to_bytes::IterBytes for InferTy {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        match *self {
          TyVar(ref tv) => to_bytes::iter_bytes_2(&0u8, tv, lsb0, f),
          IntVar(ref iv) => to_bytes::iter_bytes_2(&1u8, iv, lsb0, f),
          FloatVar(ref fv) => to_bytes::iter_bytes_2(&2u8, fv, lsb0, f),
        }
    }
}

#[auto_encode]
#[auto_decode]
pub enum InferRegion {
    ReVar(RegionVid),
    ReSkolemized(uint, bound_region)
}

impl to_bytes::IterBytes for InferRegion {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        match *self {
            ReVar(ref rv) => to_bytes::iter_bytes_2(&0u8, rv, lsb0, f),
            ReSkolemized(ref v, _) => to_bytes::iter_bytes_2(&1u8, v, lsb0, f)
        }
    }
}

impl cmp::Eq for InferRegion {
    fn eq(&self, other: &InferRegion) -> bool {
        match ((*self), *other) {
            (ReVar(rva), ReVar(rvb)) => {
                rva == rvb
            }
            (ReSkolemized(rva, _), ReSkolemized(rvb, _)) => {
                rva == rvb
            }
            _ => false
        }
    }
    fn ne(&self, other: &InferRegion) -> bool {
        !((*self) == (*other))
    }
}

impl to_bytes::IterBytes for param_bound {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        match *self {
          bound_copy => 0u8.iter_bytes(lsb0, f),
          bound_durable => 1u8.iter_bytes(lsb0, f),
          bound_owned => 2u8.iter_bytes(lsb0, f),
          bound_const => 3u8.iter_bytes(lsb0, f),
          bound_trait(ref t) =>
          to_bytes::iter_bytes_2(&4u8, t, lsb0, f)
        }
    }
}

pub trait Vid {
    fn to_uint(&self) -> uint;
}

impl Vid for TyVid {
    fn to_uint(&self) -> uint { **self }
}

impl ToStr for TyVid {
    fn to_str(&self) -> ~str { fmt!("<V%u>", self.to_uint()) }
}

impl Vid for IntVid {
    fn to_uint(&self) -> uint { **self }
}

impl ToStr for IntVid {
    fn to_str(&self) -> ~str { fmt!("<VI%u>", self.to_uint()) }
}

impl Vid for FloatVid {
    fn to_uint(&self) -> uint { **self }
}

impl ToStr for FloatVid {
    fn to_str(&self) -> ~str { fmt!("<VF%u>", self.to_uint()) }
}

impl Vid for RegionVid {
    fn to_uint(&self) -> uint { self.id }
}

impl ToStr for RegionVid {
    fn to_str(&self) -> ~str { fmt!("%?", self.id) }
}

impl ToStr for FnSig {
    fn to_str(&self) -> ~str {
        // grr, without tcx not much we can do.
        return ~"(...)";
    }
}

impl ToStr for InferTy {
    fn to_str(&self) -> ~str {
        match *self {
            TyVar(ref v) => v.to_str(),
            IntVar(ref v) => v.to_str(),
            FloatVar(ref v) => v.to_str()
        }
    }
}

impl ToStr for IntVarValue {
    fn to_str(&self) -> ~str {
        match *self {
            IntType(ref v) => v.to_str(),
            UintType(ref v) => v.to_str(),
        }
    }
}

impl to_bytes::IterBytes for TyVid {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        self.to_uint().iter_bytes(lsb0, f)
    }
}

impl to_bytes::IterBytes for IntVid {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        self.to_uint().iter_bytes(lsb0, f)
    }
}

impl to_bytes::IterBytes for FloatVid {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        self.to_uint().iter_bytes(lsb0, f)
    }
}

impl to_bytes::IterBytes for RegionVid {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        self.to_uint().iter_bytes(lsb0, f)
    }
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
pub struct ty_param_bounds_and_ty {
    bounds: @~[param_bounds],
    region_param: Option<region_variance>,
    ty: t
}

pub struct ty_param_substs_and_ty {
    substs: ty::substs,
    ty: ty::t
}

type type_cache = @mut LinearMap<ast::def_id, ty_param_bounds_and_ty>;

type constness_cache = @mut LinearMap<ast::def_id, const_eval::constness>;

pub type node_type_table = @mut SmallIntMap<t>;

fn mk_rcache() -> creader_cache {
    return @mut LinearMap::new();
}

pub fn new_ty_hash<V:Copy>() -> @mut LinearMap<t, V> {
    @mut LinearMap::new()
}

pub fn mk_ctxt(s: session::Session,
               dm: resolve::DefMap,
               amap: ast_map::map,
               freevars: freevars::freevar_map,
               region_map: middle::region::region_map,
               region_paramd_items: middle::region::region_paramd_items,
               +lang_items: middle::lang_items::LanguageItems,
               crate: @ast::crate)
            -> ctxt {
    let mut legacy_modes = false;
    for crate.node.attrs.each |attribute| {
        match attribute.node.value.node {
            ast::meta_word(w) if *w == ~"legacy_modes" => {
                legacy_modes = true;
            }
            _ => {}
        }
    }

    let vecs_implicitly_copyable =
        get_lint_level(s.lint_settings.default_settings,
                       lint::vecs_implicitly_copyable) == allow;
    @ctxt_ {
        diag: s.diagnostic(),
        interner: @mut LinearMap::new(),
        next_id: @mut 0,
        vecs_implicitly_copyable: vecs_implicitly_copyable,
        legacy_modes: legacy_modes,
        cstore: s.cstore,
        sess: s,
        def_map: dm,
        region_map: region_map,
        region_paramd_items: region_paramd_items,
        node_types: @mut SmallIntMap::new(),
        node_type_substs: @mut LinearMap::new(),
        items: amap,
        intrinsic_defs: @mut LinearMap::new(),
        freevars: freevars,
        tcache: @mut LinearMap::new(),
        rcache: mk_rcache(),
        ccache: @mut LinearMap::new(),
        short_names_cache: new_ty_hash(),
        needs_unwind_cleanup_cache: new_ty_hash(),
        tc_cache: @mut LinearMap::new(),
        ast_ty_to_ty_cache: @mut LinearMap::new(),
        enum_var_cache: @mut LinearMap::new(),
        trait_method_cache: @mut LinearMap::new(),
        ty_param_bounds: @mut LinearMap::new(),
        inferred_modes: @mut LinearMap::new(),
        adjustments: @mut LinearMap::new(),
        normalized_cache: new_ty_hash(),
        lang_items: lang_items,
        provided_methods: @mut LinearMap::new(),
        provided_method_sources: @mut LinearMap::new(),
        supertraits: @mut LinearMap::new(),
        destructor_for_type: @mut LinearMap::new(),
        destructors: @mut LinearSet::new(),
        trait_impls: @mut LinearMap::new()
     }
}


// Type constructors
fn mk_t(cx: ctxt, +st: sty) -> t { mk_t_with_id(cx, st, None) }

// Interns a type/name combination, stores the resulting box in cx.interner,
// and returns the box as cast to an unsafe ptr (see comments for t above).
fn mk_t_with_id(cx: ctxt, +st: sty, o_def_id: Option<ast::def_id>) -> t {
    let key = intern_key { sty: to_unsafe_ptr(&st), o_def_id: o_def_id };
    match cx.interner.find(&key) {
      Some(&t) => unsafe { return cast::reinterpret_cast(&t); },
      _ => ()
    }

    let mut flags = 0u;
    fn rflags(r: Region) -> uint {
        (has_regions as uint) | {
            match r {
              ty::re_infer(_) => needs_infer as uint,
              _ => 0u
            }
        }
    }
    fn sflags(substs: &substs) -> uint {
        let mut f = 0u;
        for substs.tps.each |tt| { f |= get(*tt).flags; }
        for substs.self_r.each |r| { f |= rflags(*r) }
        return f;
    }
    match &st {
      &ty_estr(vstore_slice(r)) => {
        flags |= rflags(r);
      }
      &ty_evec(ref mt, vstore_slice(r)) => {
        flags |= rflags(r);
        flags |= get(mt.ty).flags;
      }
      &ty_nil | &ty_bool | &ty_int(_) | &ty_float(_) | &ty_uint(_) |
      &ty_estr(_) | &ty_type | &ty_opaque_closure_ptr(_) |
      &ty_opaque_box => (),
      // You might think that we could just return ty_err for
      // any type containing ty_err as a component, and get
      // rid of the has_ty_err flag -- likewise for ty_bot (with
      // the exception of function types that return bot).
      // But doing so caused sporadic memory corruption, and
      // neither I (tjc) nor nmatsakis could figure out why,
      // so we're doing it this way.
      &ty_bot => flags |= has_ty_bot as uint,
      &ty_err => flags |= has_ty_err as uint,
      &ty_param(_) => flags |= has_params as uint,
      &ty_infer(_) => flags |= needs_infer as uint,
      &ty_self(_) => flags |= has_self as uint,
      &ty_enum(_, ref substs) | &ty_struct(_, ref substs) |
      &ty_trait(_, ref substs, _) => {
        flags |= sflags(substs);
      }
      &ty_box(ref m) | &ty_uniq(ref m) | &ty_evec(ref m, _) |
      &ty_ptr(ref m) | &ty_unboxed_vec(ref m) => {
        flags |= get(m.ty).flags;
      }
      &ty_rptr(r, ref m) => {
        flags |= rflags(r);
        flags |= get(m.ty).flags;
      }
      &ty_tup(ref ts) => for ts.each |tt| { flags |= get(*tt).flags; },
      &ty_bare_fn(ref f) => {
        for f.sig.inputs.each |a| { flags |= get(a.ty).flags; }
         flags |= get(f.sig.output).flags;
         // T -> _|_ is *not* _|_ !
         flags &= !(has_ty_bot as uint);
      }
      &ty_closure(ref f) => {
        flags |= rflags(f.region);
        for f.sig.inputs.each |a| { flags |= get(a.ty).flags; }
        flags |= get(f.sig.output).flags;
        // T -> _|_ is *not* _|_ !
        flags &= !(has_ty_bot as uint);
      }
    }

    let t = @t_box_ {
        sty: st,
        id: *cx.next_id,
        flags: flags,
        o_def_id: o_def_id
    };
    let key = intern_key {
        sty: to_unsafe_ptr(&t.sty),
        o_def_id: o_def_id
    };

    cx.interner.insert(key, t);

    *cx.next_id += 1;
    unsafe { cast::reinterpret_cast(&t) }
}

pub fn mk_nil(cx: ctxt) -> t { mk_t(cx, ty_nil) }

pub fn mk_err(cx: ctxt) -> t { mk_t(cx, ty_err) }

pub fn mk_bot(cx: ctxt) -> t { mk_t(cx, ty_bot) }

pub fn mk_bool(cx: ctxt) -> t { mk_t(cx, ty_bool) }

pub fn mk_int(cx: ctxt) -> t { mk_t(cx, ty_int(ast::ty_i)) }

pub fn mk_i8(cx: ctxt) -> t { mk_t(cx, ty_int(ast::ty_i8)) }

pub fn mk_i16(cx: ctxt) -> t { mk_t(cx, ty_int(ast::ty_i16)) }

pub fn mk_i32(cx: ctxt) -> t { mk_t(cx, ty_int(ast::ty_i32)) }

pub fn mk_i64(cx: ctxt) -> t { mk_t(cx, ty_int(ast::ty_i64)) }

pub fn mk_float(cx: ctxt) -> t { mk_t(cx, ty_float(ast::ty_f)) }

pub fn mk_uint(cx: ctxt) -> t { mk_t(cx, ty_uint(ast::ty_u)) }

pub fn mk_u8(cx: ctxt) -> t { mk_t(cx, ty_uint(ast::ty_u8)) }

pub fn mk_u16(cx: ctxt) -> t { mk_t(cx, ty_uint(ast::ty_u16)) }

pub fn mk_u32(cx: ctxt) -> t { mk_t(cx, ty_uint(ast::ty_u32)) }

pub fn mk_u64(cx: ctxt) -> t { mk_t(cx, ty_uint(ast::ty_u64)) }

pub fn mk_f32(cx: ctxt) -> t { mk_t(cx, ty_float(ast::ty_f32)) }

pub fn mk_f64(cx: ctxt) -> t { mk_t(cx, ty_float(ast::ty_f64)) }

pub fn mk_mach_int(cx: ctxt, tm: ast::int_ty) -> t { mk_t(cx, ty_int(tm)) }

pub fn mk_mach_uint(cx: ctxt, tm: ast::uint_ty) -> t { mk_t(cx, ty_uint(tm)) }

pub fn mk_mach_float(cx: ctxt, tm: ast::float_ty) -> t {
    mk_t(cx, ty_float(tm))
}

pub fn mk_char(cx: ctxt) -> t { mk_t(cx, ty_int(ast::ty_char)) }

pub fn mk_estr(cx: ctxt, t: vstore) -> t {
    mk_t(cx, ty_estr(t))
}

pub fn mk_enum(cx: ctxt, did: ast::def_id, +substs: substs) -> t {
    // take a copy of substs so that we own the vectors inside
    mk_t(cx, ty_enum(did, substs))
}

pub fn mk_box(cx: ctxt, tm: mt) -> t { mk_t(cx, ty_box(tm)) }

pub fn mk_imm_box(cx: ctxt, ty: t) -> t {
    mk_box(cx, mt {ty: ty, mutbl: ast::m_imm})
}

pub fn mk_uniq(cx: ctxt, tm: mt) -> t { mk_t(cx, ty_uniq(tm)) }

pub fn mk_imm_uniq(cx: ctxt, ty: t) -> t {
    mk_uniq(cx, mt {ty: ty, mutbl: ast::m_imm})
}

pub fn mk_ptr(cx: ctxt, tm: mt) -> t { mk_t(cx, ty_ptr(tm)) }

pub fn mk_rptr(cx: ctxt, r: Region, tm: mt) -> t { mk_t(cx, ty_rptr(r, tm)) }

pub fn mk_mut_rptr(cx: ctxt, r: Region, ty: t) -> t {
    mk_rptr(cx, r, mt {ty: ty, mutbl: ast::m_mutbl})
}
pub fn mk_imm_rptr(cx: ctxt, r: Region, ty: t) -> t {
    mk_rptr(cx, r, mt {ty: ty, mutbl: ast::m_imm})
}

pub fn mk_mut_ptr(cx: ctxt, ty: t) -> t {
    mk_ptr(cx, mt {ty: ty, mutbl: ast::m_mutbl})
}

pub fn mk_imm_ptr(cx: ctxt, ty: t) -> t {
    mk_ptr(cx, mt {ty: ty, mutbl: ast::m_imm})
}

pub fn mk_nil_ptr(cx: ctxt) -> t {
    mk_ptr(cx, mt {ty: mk_nil(cx), mutbl: ast::m_imm})
}

pub fn mk_evec(cx: ctxt, tm: mt, t: vstore) -> t {
    mk_t(cx, ty_evec(tm, t))
}

pub fn mk_unboxed_vec(cx: ctxt, tm: mt) -> t {
    mk_t(cx, ty_unboxed_vec(tm))
}
pub fn mk_mut_unboxed_vec(cx: ctxt, ty: t) -> t {
    mk_t(cx, ty_unboxed_vec(mt {ty: ty, mutbl: ast::m_imm}))
}

pub fn mk_tup(cx: ctxt, +ts: ~[t]) -> t { mk_t(cx, ty_tup(ts)) }

pub fn mk_closure(cx: ctxt, +fty: ClosureTy) -> t {
    mk_t(cx, ty_closure(fty))
}

pub fn mk_bare_fn(cx: ctxt, +fty: BareFnTy) -> t {
    mk_t(cx, ty_bare_fn(fty))
}

pub fn mk_ctor_fn(cx: ctxt, input_tys: &[ty::t], output: ty::t) -> t {
    let input_args = input_tys.map(|t| arg {mode: ast::expl(ast::by_copy),
                                            ty: *t});
    mk_bare_fn(cx,
               BareFnTy {
                   purity: ast::pure_fn,
                   abi: ast::RustAbi,
                   sig: FnSig {bound_lifetime_names: opt_vec::Empty,
                               inputs: input_args,
                               output: output}})
}


pub fn mk_trait(cx: ctxt,
                did: ast::def_id,
                +substs: substs,
                store: TraitStore)
             -> t {
    // take a copy of substs so that we own the vectors inside
    mk_t(cx, ty_trait(did, substs, store))
}

pub fn mk_struct(cx: ctxt, struct_id: ast::def_id, +substs: substs) -> t {
    // take a copy of substs so that we own the vectors inside
    mk_t(cx, ty_struct(struct_id, substs))
}

pub fn mk_var(cx: ctxt, v: TyVid) -> t { mk_infer(cx, TyVar(v)) }

pub fn mk_int_var(cx: ctxt, v: IntVid) -> t { mk_infer(cx, IntVar(v)) }

pub fn mk_float_var(cx: ctxt, v: FloatVid) -> t { mk_infer(cx, FloatVar(v)) }

pub fn mk_infer(cx: ctxt, +it: InferTy) -> t { mk_t(cx, ty_infer(it)) }

pub fn mk_self(cx: ctxt, did: ast::def_id) -> t { mk_t(cx, ty_self(did)) }

pub fn mk_param(cx: ctxt, n: uint, k: def_id) -> t {
    mk_t(cx, ty_param(param_ty { idx: n, def_id: k }))
}

pub fn mk_type(cx: ctxt) -> t { mk_t(cx, ty_type) }

pub fn mk_opaque_closure_ptr(cx: ctxt, sigil: ast::Sigil) -> t {
    mk_t(cx, ty_opaque_closure_ptr(sigil))
}

pub fn mk_opaque_box(cx: ctxt) -> t { mk_t(cx, ty_opaque_box) }

pub fn mk_with_id(cx: ctxt, base: t, def_id: ast::def_id) -> t {
    mk_t_with_id(cx, /*bad*/copy get(base).sty, Some(def_id))
}

// Converts s to its machine type equivalent
pub fn mach_sty(cfg: @session::config, t: t) -> sty {
    match get(t).sty {
      ty_int(ast::ty_i) => ty_int(cfg.int_type),
      ty_uint(ast::ty_u) => ty_uint(cfg.uint_type),
      ty_float(ast::ty_f) => ty_float(cfg.float_type),
      ref s => (/*bad*/copy *s)
    }
}

pub fn default_arg_mode_for_ty(tcx: ctxt, ty: ty::t) -> ast::rmode {
    // FIXME(#2202) --- We retain by-ref for &fn things to workaround a
    // memory leak that otherwise results when @fn is upcast to &fn.
    match ty::get(ty).sty {
        ty::ty_closure(ClosureTy {sigil: ast::BorrowedSigil, _}) => {
            return ast::by_ref;
        }
        _ => {}
    }
    return if tcx.legacy_modes {
        if type_is_borrowed(ty) {
            // the old mode default was ++ for things like &ptr, but to be
            // forward-compatible with non-legacy, we should use +
            ast::by_copy
        } else if ty::type_is_immediate(ty) {
            ast::by_copy
        } else {
            ast::by_ref
        }
    } else {
        ast::by_copy
    };

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
pub fn encl_region(cx: ctxt, id: ast::node_id) -> ty::Region {
    match cx.region_map.find(&id) {
      Some(&encl_scope) => ty::re_scope(encl_scope),
      None => ty::re_static
    }
}

pub fn walk_ty(ty: t, f: &fn(t)) {
    maybe_walk_ty(ty, |t| { f(t); true });
}

pub fn maybe_walk_ty(ty: t, f: &fn(t) -> bool) {
    if !f(ty) { return; }
    match get(ty).sty {
      ty_nil | ty_bot | ty_bool | ty_int(_) | ty_uint(_) | ty_float(_) |
      ty_estr(_) | ty_type | ty_opaque_box | ty_self(_) |
      ty_opaque_closure_ptr(_) | ty_infer(_) | ty_param(_) | ty_err => {
      }
      ty_box(ref tm) | ty_evec(ref tm, _) | ty_unboxed_vec(ref tm) |
      ty_ptr(ref tm) | ty_rptr(_, ref tm) | ty_uniq(ref tm) => {
        maybe_walk_ty(tm.ty, f);
      }
      ty_enum(_, ref substs) | ty_struct(_, ref substs) |
      ty_trait(_, ref substs, _) => {
        for (*substs).tps.each |subty| { maybe_walk_ty(*subty, f); }
      }
      ty_tup(ref ts) => { for ts.each |tt| { maybe_walk_ty(*tt, f); } }
      ty_bare_fn(ref ft) => {
        for ft.sig.inputs.each |a| { maybe_walk_ty(a.ty, f); }
        maybe_walk_ty(ft.sig.output, f);
      }
      ty_closure(ref ft) => {
        for ft.sig.inputs.each |a| { maybe_walk_ty(a.ty, f); }
        maybe_walk_ty(ft.sig.output, f);
      }
    }
}

pub fn fold_sty_to_ty(tcx: ty::ctxt, sty: &sty, foldop: &fn(t) -> t) -> t {
    mk_t(tcx, fold_sty(sty, foldop))
}

pub fn fold_sig(sig: &FnSig, fldop: &fn(t) -> t) -> FnSig {
    let args = do sig.inputs.map |arg| {
        arg { mode: arg.mode, ty: fldop(arg.ty) }
    };

    FnSig {
        bound_lifetime_names: copy sig.bound_lifetime_names,
        inputs: args,
        output: fldop(sig.output)
    }
}

fn fold_sty(sty: &sty, fldop: &fn(t) -> t) -> sty {
    fn fold_substs(substs: &substs, fldop: &fn(t) -> t) -> substs {
        substs {self_r: substs.self_r,
                self_ty: substs.self_ty.map(|t| fldop(*t)),
                tps: substs.tps.map(|t| fldop(*t))}
    }

    match *sty {
        ty_box(ref tm) => {
            ty_box(mt {ty: fldop(tm.ty), mutbl: tm.mutbl})
        }
        ty_uniq(ref tm) => {
            ty_uniq(mt {ty: fldop(tm.ty), mutbl: tm.mutbl})
        }
        ty_ptr(ref tm) => {
            ty_ptr(mt {ty: fldop(tm.ty), mutbl: tm.mutbl})
        }
        ty_unboxed_vec(ref tm) => {
            ty_unboxed_vec(mt {ty: fldop(tm.ty), mutbl: tm.mutbl})
        }
        ty_evec(ref tm, vst) => {
            ty_evec(mt {ty: fldop(tm.ty), mutbl: tm.mutbl}, vst)
        }
        ty_enum(tid, ref substs) => {
            ty_enum(tid, fold_substs(substs, fldop))
        }
        ty_trait(did, ref substs, st) => {
            ty_trait(did, fold_substs(substs, fldop), st)
        }
        ty_tup(ref ts) => {
            let new_ts = ts.map(|tt| fldop(*tt));
            ty_tup(new_ts)
        }
        ty_bare_fn(ref f) => {
            let sig = fold_sig(&f.sig, fldop);
            ty_bare_fn(BareFnTy {sig: sig, abi: f.abi, purity: f.purity})
        }
        ty_closure(ref f) => {
            let sig = fold_sig(&f.sig, fldop);
            ty_closure(ClosureTy {sig: sig, ..copy *f})
        }
        ty_rptr(r, ref tm) => {
            ty_rptr(r, mt {ty: fldop(tm.ty), mutbl: tm.mutbl})
        }
        ty_struct(did, ref substs) => {
            ty_struct(did, fold_substs(substs, fldop))
        }
        ty_nil | ty_bot | ty_bool | ty_int(_) | ty_uint(_) | ty_float(_) |
        ty_estr(_) | ty_type | ty_opaque_closure_ptr(_) | ty_err |
        ty_opaque_box | ty_infer(_) | ty_param(*) | ty_self(_) => {
            /*bad*/copy *sty
        }
    }
}

// Folds types from the bottom up.
pub fn fold_ty(cx: ctxt, t0: t, fldop: &fn(t) -> t) -> t {
    let sty = fold_sty(&get(t0).sty, |t| fold_ty(cx, fldop(t), fldop));
    fldop(mk_t(cx, sty))
}

pub fn walk_regions_and_ty(
    cx: ctxt,
    ty: t,
    walkr: &fn(r: Region),
    walkt: &fn(t: t) -> bool) {

    if (walkt(ty)) {
        fold_regions_and_ty(
            cx, ty,
            |r| { walkr(r); r },
            |t| { walkt(t); walk_regions_and_ty(cx, t, walkr, walkt); t },
            |t| { walkt(t); walk_regions_and_ty(cx, t, walkr, walkt); t });
    }
}

pub fn fold_regions_and_ty(
    cx: ctxt,
    ty: t,
    fldr: &fn(r: Region) -> Region,
    fldfnt: &fn(t: t) -> t,
    fldt: &fn(t: t) -> t) -> t {

    fn fold_substs(
        substs: &substs,
        fldr: &fn(r: Region) -> Region,
        fldt: &fn(t: t) -> t)
     -> substs {
        substs {
            self_r: substs.self_r.map(|r| fldr(*r)),
            self_ty: substs.self_ty.map(|t| fldt(*t)),
            tps: substs.tps.map(|t| fldt(*t))
        }
    }

    let tb = ty::get(ty);
    match tb.sty {
      ty::ty_rptr(r, mt) => {
        let m_r = fldr(r);
        let m_t = fldt(mt.ty);
        ty::mk_rptr(cx, m_r, mt {ty: m_t, mutbl: mt.mutbl})
      }
      ty_estr(vstore_slice(r)) => {
        let m_r = fldr(r);
        ty::mk_estr(cx, vstore_slice(m_r))
      }
      ty_evec(mt, vstore_slice(r)) => {
        let m_r = fldr(r);
        let m_t = fldt(mt.ty);
        ty::mk_evec(cx, mt {ty: m_t, mutbl: mt.mutbl}, vstore_slice(m_r))
      }
      ty_enum(def_id, ref substs) => {
        ty::mk_enum(cx, def_id, fold_substs(substs, fldr, fldt))
      }
      ty_struct(def_id, ref substs) => {
        ty::mk_struct(cx, def_id, fold_substs(substs, fldr, fldt))
      }
      ty_trait(def_id, ref substs, st) => {
        ty::mk_trait(cx, def_id, fold_substs(substs, fldr, fldt), st)
      }
      ty_bare_fn(ref f) => {
          ty::mk_bare_fn(cx, BareFnTy {sig: fold_sig(&f.sig, fldfnt),
                                       ..copy *f})
      }
      ty_closure(ref f) => {
          ty::mk_closure(cx, ClosureTy {region: fldr(f.region),
                                        sig: fold_sig(&f.sig, fldfnt),
                                        ..copy *f})
      }
      ref sty => {
        fold_sty_to_ty(cx, sty, |t| fldt(t))
      }
    }
}

// n.b. this function is intended to eventually replace fold_region() below,
// that is why its name is so similar.
pub fn fold_regions(
    cx: ctxt,
    ty: t,
    fldr: &fn(r: Region, in_fn: bool) -> Region) -> t {
    fn do_fold(cx: ctxt, ty: t, in_fn: bool,
               fldr: &fn(Region, bool) -> Region) -> t {
        debug!("do_fold(ty=%s, in_fn=%b)", ty_to_str(cx, ty), in_fn);
        if !type_has_regions(ty) { return ty; }
        fold_regions_and_ty(
            cx, ty,
            |r| fldr(r, in_fn),
            |t| do_fold(cx, t, true, fldr),
            |t| do_fold(cx, t, in_fn, fldr))
    }
    do_fold(cx, ty, false, fldr)
}

// Substitute *only* type parameters.  Used in trans where regions are erased.
pub fn subst_tps(cx: ctxt, tps: &[t], self_ty_opt: Option<t>, typ: t) -> t {
    if tps.len() == 0u && self_ty_opt.is_none() { return typ; }
    let tb = ty::get(typ);
    if self_ty_opt.is_none() && !tbox_has_flag(tb, has_params) { return typ; }
    match tb.sty {
        ty_param(p) => tps[p.idx],
        ty_self(_) => {
            match self_ty_opt {
                None => cx.sess.bug(~"ty_self unexpected here"),
                Some(self_ty) => {
                    subst_tps(cx, tps, self_ty_opt, self_ty)
                }
            }
        }
        ref sty => {
            fold_sty_to_ty(cx, sty, |t| subst_tps(cx, tps, self_ty_opt, t))
        }
    }
}

pub fn substs_is_noop(substs: &substs) -> bool {
    substs.tps.len() == 0u &&
        substs.self_r.is_none() &&
        substs.self_ty.is_none()
}

pub fn substs_to_str(cx: ctxt, substs: &substs) -> ~str {
    fmt!("substs(self_r=%s, self_ty=%s, tps=%?)",
         substs.self_r.map_default(~"none", |r| region_to_str(cx, *r)),
         substs.self_ty.map_default(~"none",
                                    |t| ::util::ppaux::ty_to_str(cx, *t)),
         tys_to_str(cx, substs.tps))
}

pub fn param_bound_to_str(cx: ctxt, pb: &param_bound) -> ~str {
    match *pb {
        bound_copy => ~"copy",
        bound_durable => ~"'static",
        bound_owned => ~"owned",
        bound_const => ~"const",
        bound_trait(t) => ::util::ppaux::ty_to_str(cx, t)
    }
}

pub fn param_bounds_to_str(cx: ctxt, pbs: param_bounds) -> ~str {
    fmt!("%?", pbs.map(|pb| param_bound_to_str(cx, pb)))
}

pub fn subst(cx: ctxt,
             substs: &substs,
             typ: t)
          -> t {
    debug!("subst(substs=%s, typ=%s)",
           substs_to_str(cx, substs),
           ::util::ppaux::ty_to_str(cx, typ));

    if substs_is_noop(substs) { return typ; }
    let r = do_subst(cx, substs, typ);
    debug!("  r = %s", ::util::ppaux::ty_to_str(cx, r));
    return r;

    fn do_subst(cx: ctxt,
                substs: &substs,
                typ: t) -> t {
        let tb = get(typ);
        if !tbox_has_flag(tb, needs_subst) { return typ; }
        match tb.sty {
          ty_param(p) => substs.tps[p.idx],
          ty_self(_) => substs.self_ty.get(),
          _ => {
            fold_regions_and_ty(
                cx, typ,
                |r| match r {
                    re_bound(br_self) => {
                        match substs.self_r {
                            None => {
                                cx.sess.bug(
                                    fmt!("ty::subst: \
                                  Reference to self region when given substs \
                                  with no self region, ty = %s",
                                  ::util::ppaux::ty_to_str(cx, typ)))
                            }
                            Some(self_r) => self_r
                        }
                    }
                    _ => r
                },
                |t| do_subst(cx, substs, t),
                |t| do_subst(cx, substs, t))
          }
        }
    }
}

// Performs substitutions on a set of substitutions (result = sup(sub)) to
// yield a new set of substitutions. This is used in trait inheritance.
pub fn subst_substs(cx: ctxt, sup: &substs, sub: &substs) -> substs {
    substs {
        self_r: sup.self_r,
        self_ty: sup.self_ty.map(|typ| subst(cx, sub, *typ)),
        tps: sup.tps.map(|typ| subst(cx, sub, *typ))
    }
}

// Type utilities

pub fn type_is_nil(ty: t) -> bool { get(ty).sty == ty_nil }

pub fn type_is_bot(ty: t) -> bool {
    (get(ty).flags & (has_ty_bot as uint)) != 0
}

pub fn type_is_error(ty: t) -> bool {
    (get(ty).flags & (has_ty_err as uint)) != 0
}

pub fn type_is_ty_var(ty: t) -> bool {
    match get(ty).sty {
      ty_infer(TyVar(_)) => true,
      _ => false
    }
}

pub fn type_is_bool(ty: t) -> bool { get(ty).sty == ty_bool }

pub fn type_is_structural(ty: t) -> bool {
    match get(ty).sty {
      ty_struct(*) | ty_tup(_) | ty_enum(*) | ty_closure(_) | ty_trait(*) |
      ty_evec(_, vstore_fixed(_)) | ty_estr(vstore_fixed(_)) |
      ty_evec(_, vstore_slice(_)) | ty_estr(vstore_slice(_))
      => true,
      _ => false
    }
}

pub fn type_is_sequence(ty: t) -> bool {
    match get(ty).sty {
      ty_estr(_) | ty_evec(_, _) => true,
      _ => false
    }
}

pub fn type_is_str(ty: t) -> bool {
    match get(ty).sty {
      ty_estr(_) => true,
      _ => false
    }
}

pub fn sequence_element_type(cx: ctxt, ty: t) -> t {
    match get(ty).sty {
      ty_estr(_) => return mk_mach_uint(cx, ast::ty_u8),
      ty_evec(mt, _) | ty_unboxed_vec(mt) => return mt.ty,
      _ => cx.sess.bug(
          ~"sequence_element_type called on non-sequence value"),
    }
}

pub fn get_element_type(ty: t, i: uint) -> t {
    match get(ty).sty {
      ty_tup(ref ts) => return ts[i],
      _ => fail!(~"get_element_type called on invalid type")
    }
}

pub fn type_is_box(ty: t) -> bool {
    match get(ty).sty {
      ty_box(_) => return true,
      _ => return false
    }
}

pub fn type_is_boxed(ty: t) -> bool {
    match get(ty).sty {
      ty_box(_) | ty_opaque_box |
      ty_evec(_, vstore_box) | ty_estr(vstore_box) => true,
      _ => false
    }
}

pub fn type_is_region_ptr(ty: t) -> bool {
    match get(ty).sty {
      ty_rptr(_, _) => true,
      _ => false
    }
}

pub fn type_is_slice(ty: t) -> bool {
    match get(ty).sty {
      ty_evec(_, vstore_slice(_)) | ty_estr(vstore_slice(_)) => true,
      _ => return false
    }
}

pub fn type_is_unique_box(ty: t) -> bool {
    match get(ty).sty {
      ty_uniq(_) => return true,
      _ => return false
    }
}

pub fn type_is_unsafe_ptr(ty: t) -> bool {
    match get(ty).sty {
      ty_ptr(_) => return true,
      _ => return false
    }
}

pub fn type_is_vec(ty: t) -> bool {
    return match get(ty).sty {
          ty_evec(_, _) | ty_unboxed_vec(_) => true,
          ty_estr(_) => true,
          _ => false
        };
}

pub fn type_is_unique(ty: t) -> bool {
    match get(ty).sty {
        ty_uniq(_) |
        ty_evec(_, vstore_uniq) |
        ty_estr(vstore_uniq) |
        ty_opaque_closure_ptr(ast::OwnedSigil) => true,
        _ => return false
    }
}

/*
 A scalar type is one that denotes an atomic datum, with no sub-components.
 (A ty_ptr is scalar because it represents a non-managed pointer, so its
 contents are abstract to rustc.)
*/
pub fn type_is_scalar(ty: t) -> bool {
    match get(ty).sty {
      ty_nil | ty_bool | ty_int(_) | ty_float(_) | ty_uint(_) |
      ty_infer(IntVar(_)) | ty_infer(FloatVar(_)) | ty_type |
      ty_bare_fn(*) | ty_ptr(_) => true,
      _ => false
    }
}

pub fn type_is_immediate(ty: t) -> bool {
    return type_is_scalar(ty) || type_is_boxed(ty) ||
        type_is_unique(ty) || type_is_region_ptr(ty);
}

pub fn type_needs_drop(cx: ctxt, ty: t) -> bool {
    type_contents(cx, ty).needs_drop(cx)
}

// Some things don't need cleanups during unwinding because the
// task can free them all at once later. Currently only things
// that only contain scalars and shared boxes can avoid unwind
// cleanups.
pub fn type_needs_unwind_cleanup(cx: ctxt, ty: t) -> bool {
    match cx.needs_unwind_cleanup_cache.find(&ty) {
      Some(&result) => return result,
      None => ()
    }

    let mut tycache = LinearSet::new();
    let needs_unwind_cleanup =
        type_needs_unwind_cleanup_(cx, ty, &mut tycache, false);
    cx.needs_unwind_cleanup_cache.insert(ty, needs_unwind_cleanup);
    return needs_unwind_cleanup;
}

fn type_needs_unwind_cleanup_(cx: ctxt, ty: t,
                              tycache: &mut LinearSet<t>,
                              encountered_box: bool) -> bool {

    // Prevent infinite recursion
    if !tycache.insert(ty) {
        return false;
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
          ty_nil | ty_bot | ty_bool | ty_int(_) | ty_uint(_) | ty_float(_) |
          ty_tup(_) | ty_ptr(_) => {
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

/**
 * Type contents is how the type checker reasons about kinds.
 * They track what kinds of things are found within a type.  You can
 * think of them as kind of an "anti-kind".  They track the kinds of values
 * and thinks that are contained in types.  Having a larger contents for
 * a type tends to rule that type *out* from various kinds.  For example,
 * a type that contains a borrowed pointer is not sendable.
 *
 * The reason we compute type contents and not kinds is that it is
 * easier for me (nmatsakis) to think about what is contained within
 * a type than to think about what is *not* contained within a type.
 */
pub struct TypeContents {
    bits: u32
}

pub impl TypeContents {
    fn intersects(&self, tc: TypeContents) -> bool {
        (self.bits & tc.bits) != 0
    }

    fn is_copy(&self, cx: ctxt) -> bool {
        !self.intersects(TypeContents::noncopyable(cx))
    }

    fn noncopyable(_cx: ctxt) -> TypeContents {
        TC_DTOR + TC_BORROWED_MUT + TC_ONCE_CLOSURE + TC_OWNED_CLOSURE +
            TC_EMPTY_ENUM
    }

    fn is_durable(&self, cx: ctxt) -> bool {
        !self.intersects(TypeContents::nondurable(cx))
    }

    fn nondurable(_cx: ctxt) -> TypeContents {
        TC_BORROWED_POINTER
    }

    fn is_owned(&self, cx: ctxt) -> bool {
        !self.intersects(TypeContents::nonowned(cx))
    }

    fn nonowned(_cx: ctxt) -> TypeContents {
        TC_MANAGED + TC_BORROWED_POINTER
    }

    fn contains_managed(&self) -> bool {
        self.intersects(TC_MANAGED)
    }

    fn is_const(&self, cx: ctxt) -> bool {
        !self.intersects(TypeContents::nonconst(cx))
    }

    fn nonconst(_cx: ctxt) -> TypeContents {
        TC_MUTABLE
    }

    fn moves_by_default(&self, cx: ctxt) -> bool {
        self.intersects(TypeContents::nonimplicitly_copyable(cx))
    }

    fn nonimplicitly_copyable(cx: ctxt) -> TypeContents {
        let base = TypeContents::noncopyable(cx) + TC_OWNED_POINTER;
        if cx.vecs_implicitly_copyable {base} else {base + TC_OWNED_VEC}
    }

    fn is_safe_for_default_mode(&self, cx: ctxt) -> bool {
        !self.intersects(TypeContents::nondefault_mode(cx))
    }

    fn nondefault_mode(cx: ctxt) -> TypeContents {
        let tc = TypeContents::nonimplicitly_copyable(cx);
        tc + TC_BIG + TC_OWNED_VEC // disregard cx.vecs_implicitly_copyable
    }

    fn needs_drop(&self, cx: ctxt) -> bool {
        let tc = TC_MANAGED + TC_DTOR + TypeContents::owned(cx);
        self.intersects(tc)
    }

    fn owned(_cx: ctxt) -> TypeContents {
        //! Any kind of owned contents.
        TC_OWNED_CLOSURE + TC_OWNED_POINTER + TC_OWNED_VEC
    }
}

impl ops::Add<TypeContents,TypeContents> for TypeContents {
    fn add(&self, other: &TypeContents) -> TypeContents {
        TypeContents {bits: self.bits | other.bits}
    }
}

impl ops::Sub<TypeContents,TypeContents> for TypeContents {
    fn sub(&self, other: &TypeContents) -> TypeContents {
        TypeContents {bits: self.bits & !other.bits}
    }
}

impl ToStr for TypeContents {
    fn to_str(&self) -> ~str {
        fmt!("TypeContents(%s)", u32::to_str_radix(self.bits, 2))
    }
}

/// Constant for a type containing nothing of interest.
static TC_NONE: TypeContents =             TypeContents{bits:0b0000_00000000};

/// Contains a borrowed value with a lifetime other than static
static TC_BORROWED_POINTER: TypeContents = TypeContents{bits:0b0000_00000001};

/// Contains an owned pointer (~T) but not slice of some kind
static TC_OWNED_POINTER: TypeContents =    TypeContents{bits:0b000000000010};

/// Contains an owned vector ~[] or owned string ~str
static TC_OWNED_VEC: TypeContents =        TypeContents{bits:0b000000000100};

/// Contains a ~fn() or a ~Trait, which is non-copyable.
static TC_OWNED_CLOSURE: TypeContents =    TypeContents{bits:0b000000001000};

/// Type with a destructor
static TC_DTOR: TypeContents =             TypeContents{bits:0b000000010000};

/// Contains a managed value
static TC_MANAGED: TypeContents =          TypeContents{bits:0b000000100000};

/// &mut with any region
static TC_BORROWED_MUT: TypeContents =     TypeContents{bits:0b000001000000};

/// Mutable content, whether owned or by ref
static TC_MUTABLE: TypeContents =          TypeContents{bits:0b000010000000};

/// Mutable content, whether owned or by ref
static TC_ONCE_CLOSURE: TypeContents =     TypeContents{bits:0b000100000000};

/// Something we estimate to be "big"
static TC_BIG: TypeContents =              TypeContents{bits:0b001000000000};

/// An enum with no variants.
static TC_EMPTY_ENUM: TypeContents =       TypeContents{bits:0b010000000000};

/// All possible contents.
static TC_ALL: TypeContents =              TypeContents{bits:0b011111111111};

pub fn type_is_copyable(cx: ctxt, t: ty::t) -> bool {
    type_contents(cx, t).is_copy(cx)
}

pub fn type_is_durable(cx: ctxt, t: ty::t) -> bool {
    type_contents(cx, t).is_durable(cx)
}

pub fn type_is_owned(cx: ctxt, t: ty::t) -> bool {
    type_contents(cx, t).is_owned(cx)
}

pub fn type_is_const(cx: ctxt, t: ty::t) -> bool {
    type_contents(cx, t).is_const(cx)
}

pub fn type_contents(cx: ctxt, ty: t) -> TypeContents {
    let ty_id = type_id(ty);
    match cx.tc_cache.find(&ty_id) {
        Some(tc) => { return *tc; }
        None => {}
    }

    let mut cache = LinearMap::new();
    let result = tc_ty(cx, ty, &mut cache);
    cx.tc_cache.insert(ty_id, result);
    return result;

    fn tc_ty(cx: ctxt,
             ty: t,
             cache: &mut LinearMap<uint, TypeContents>) -> TypeContents
    {
        // Subtle: Note that we are *not* using cx.tc_cache here but rather a
        // private cache for this walk.  This is needed in the case of cyclic
        // types like:
        //
        //     struct List { next: ~Option<List>, ... }
        //
        // When computing the type contents of such a type, we wind up deeply
        // recursing as we go.  So when we encounter the recursive reference
        // to List, we temporarily use TC_NONE as its contents.  Later we'll
        // patch up the cache with the correct value, once we've computed it
        // (this is basically a co-inductive process, if that helps).  So in
        // the end we'll compute TC_OWNED_POINTER, in this case.
        //
        // The problem is, as we are doing the computation, we will also
        // compute an *intermediate* contents for, e.g., Option<List> of
        // TC_NONE.  This is ok during the computation of List itself, but if
        // we stored this intermediate value into cx.tc_cache, then later
        // requests for the contents of Option<List> would also yield TC_NONE
        // which is incorrect.  This value was computed based on the crutch
        // value for the type contents of list.  The correct value is
        // TC_OWNED_POINTER.  This manifested as issue #4821.
        let ty_id = type_id(ty);
        match cache.find(&ty_id) {
            Some(tc) => { return *tc; }
            None => {}
        }
        match cx.tc_cache.find(&ty_id) {    // Must check both caches!
            Some(tc) => { return *tc; }
            None => {}
        }
        cache.insert(ty_id, TC_NONE);

        let _i = indenter();

        let mut result = match get(ty).sty {
            // Scalar and unique types are sendable, constant, and owned
            ty_nil | ty_bot | ty_bool | ty_int(_) | ty_uint(_) | ty_float(_) |
            ty_bare_fn(_) | ty_ptr(_) => {
                TC_NONE
            }

            ty_estr(vstore_uniq) => {
                TC_OWNED_VEC
            }

            ty_closure(ref c) => {
                closure_contents(c)
            }

            ty_box(mt) => {
                TC_MANAGED + nonowned(tc_mt(cx, mt, cache))
            }

            ty_trait(_, _, UniqTraitStore) => {
                TC_OWNED_CLOSURE
            }

            ty_trait(_, _, BoxTraitStore) |
            ty_trait(_, _, BareTraitStore) => {
                TC_MANAGED
            }

            ty_trait(_, _, RegionTraitStore(r)) => {
                borrowed_contents(r, m_imm)
            }

            ty_rptr(r, mt) => {
                borrowed_contents(r, mt.mutbl) +
                    nonowned(tc_mt(cx, mt, cache))
            }

            ty_uniq(mt) => {
                TC_OWNED_POINTER + tc_mt(cx, mt, cache)
            }

            ty_evec(mt, vstore_uniq) => {
                TC_OWNED_VEC + tc_mt(cx, mt, cache)
            }

            ty_evec(mt, vstore_box) => {
                TC_MANAGED + nonowned(tc_mt(cx, mt, cache))
            }

            ty_evec(mt, vstore_slice(r)) => {
                borrowed_contents(r, mt.mutbl) +
                    nonowned(tc_mt(cx, mt, cache))
            }

            ty_evec(mt, vstore_fixed(_)) => {
                tc_mt(cx, mt, cache)
            }

            ty_estr(vstore_box) => {
                TC_MANAGED
            }

            ty_estr(vstore_slice(r)) => {
                borrowed_contents(r, m_imm)
            }

            ty_estr(vstore_fixed(_)) => {
                TC_NONE
            }

            ty_struct(did, ref substs) => {
                let flds = struct_fields(cx, did, substs);
                let flds_tc = flds.foldl(
                    TC_NONE,
                    |tc, f| tc + tc_mt(cx, f.mt, cache));
                if ty::has_dtor(cx, did) {
                    flds_tc + TC_DTOR
                } else {
                    flds_tc
                }
            }

            ty_tup(ref tys) => {
                tys.foldl(TC_NONE, |tc, ty| *tc + tc_ty(cx, *ty, cache))
            }

            ty_enum(did, ref substs) => {
                let variants = substd_enum_variants(cx, did, substs);
                if variants.is_empty() {
                    // we somewhat arbitrary declare that empty enums
                    // are non-copyable
                    TC_EMPTY_ENUM
                } else {
                    variants.foldl(TC_NONE, |tc, variant| {
                        variant.args.foldl(
                            *tc,
                            |tc, arg_ty| *tc + tc_ty(cx, *arg_ty, cache))
                    })
                }
            }

            ty_param(p) => {
                // We only ever ask for the kind of types that are defined in
                // the current crate; therefore, the only type parameters that
                // could be in scope are those defined in the current crate.
                // If this assertion failures, it is likely because of a
                // failure in the cross-crate inlining code to translate a
                // def-id.
                fail_unless!(p.def_id.crate == ast::local_crate);

                param_bounds_to_contents(
                    cx, *cx.ty_param_bounds.get(&p.def_id.node))
            }

            ty_self(_) => {
                // Currently, self is not bounded, so we must assume the
                // worst.  But in the future we should examine the super
                // traits.
                //
                // FIXME(#4678)---self should just be a ty param
                TC_ALL
            }

            ty_infer(_) => {
                // This occurs during coherence, but shouldn't occur at other
                // times.
                TC_ALL
            }

            ty_opaque_box => TC_MANAGED,
            ty_unboxed_vec(mt) => tc_mt(cx, mt, cache),
            ty_opaque_closure_ptr(sigil) => {
                match sigil {
                    ast::BorrowedSigil => TC_BORROWED_POINTER,
                    ast::ManagedSigil => TC_MANAGED,
                    ast::OwnedSigil => TC_OWNED_CLOSURE
                }
            }

            ty_type => TC_NONE,

            ty_err => {
                cx.sess.bug(~"Asked to compute contents of fictitious type");
            }
        };

        if type_size(cx, ty) > 4 {
            result = result + TC_BIG;
        }

        cache.insert(ty_id, result);
        return result;
    }

    fn tc_mt(cx: ctxt,
             mt: mt,
             cache: &mut LinearMap<uint, TypeContents>) -> TypeContents
    {
        let mc = if mt.mutbl == m_mutbl {TC_MUTABLE} else {TC_NONE};
        mc + tc_ty(cx, mt.ty, cache)
    }

    fn borrowed_contents(region: ty::Region,
                         mutbl: ast::mutability) -> TypeContents
    {
        let mc = if mutbl == m_mutbl {
            TC_MUTABLE + TC_BORROWED_MUT
        } else {
            TC_NONE
        };
        let rc = if region != ty::re_static {
            TC_BORROWED_POINTER
        } else {
            TC_NONE
        };
        mc + rc
    }

    fn nonowned(pointee: TypeContents) -> TypeContents {
        /*!
         *
         * Given a non-owning pointer to some type `T` with
         * contents `pointee` (like `@T` or
         * `&T`), returns the relevant bits that
         * apply to the owner of the pointer.
         */

        let mask = TC_MUTABLE.bits | TC_BORROWED_POINTER.bits;
        TypeContents {bits: pointee.bits & mask}
    }

    fn closure_contents(cty: &ClosureTy) -> TypeContents {
        let st = match cty.sigil {
            ast::BorrowedSigil => TC_BORROWED_POINTER,
            ast::ManagedSigil => TC_MANAGED,
            ast::OwnedSigil => TC_OWNED_CLOSURE
        };
        let rt = borrowed_contents(cty.region, m_imm);
        let ot = match cty.onceness {
            ast::Once => TC_ONCE_CLOSURE,
            ast::Many => TC_NONE
        };
        st + rt + ot
    }

    fn param_bounds_to_contents(cx: ctxt,
                                bounds: param_bounds) -> TypeContents
    {
        debug!("param_bounds_to_contents()");
        let _i = indenter();

        let r = bounds.foldl(TC_ALL, |tc, bound| {
            debug!("tc = %s, bound = %?", tc.to_str(), bound);
            match *bound {
                bound_copy => tc - TypeContents::nonimplicitly_copyable(cx),
                bound_durable => tc - TypeContents::nondurable(cx),
                bound_owned => tc - TypeContents::nonowned(cx),
                bound_const => tc - TypeContents::nonconst(cx),
                bound_trait(_) => *tc
            }
        });

        debug!("result = %s", r.to_str());
        return r;
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
          ty_bare_fn(*) |
          ty_closure(*) => {
            2
          }

          ty_evec(t, vstore_fixed(n)) => {
            type_size(cx, t.ty) * n
          }

          ty_estr(vstore_fixed(n)) => {
            n
          }

          ty_struct(did, ref substs) => {
            let flds = struct_fields(cx, did, substs);
            flds.foldl(0, |s, f| *s + type_size(cx, f.mt.ty))
          }

          ty_tup(ref tys) => {
            tys.foldl(0, |s, t| *s + type_size(cx, *t))
          }

          ty_enum(did, ref substs) => {
            let variants = substd_enum_variants(cx, did, substs);
            variants.foldl( // find max size of any variant
                0,
                |m, v| uint::max(
                    *m,
                    // find size of this variant:
                    v.args.foldl(0, |s, a| *s + type_size(cx, *a))))
          }

          ty_param(_) | ty_self(_) => {
            1
          }

          ty_infer(_) => {
            cx.sess.bug(~"Asked to compute kind of a type variable");
          }
          ty_type => 1,
          ty_opaque_closure_ptr(_) => 1,
          ty_opaque_box => 1,
          ty_unboxed_vec(_) => 10,
          ty_err => {
            cx.sess.bug(~"Asked to compute kind of fictitious type");
          }
        }
    }
}

pub fn type_moves_by_default(cx: ctxt, ty: t) -> bool {
    type_contents(cx, ty).moves_by_default(cx)
}

// True if instantiating an instance of `r_ty` requires an instance of `r_ty`.
pub fn is_instantiable(cx: ctxt, r_ty: t) -> bool {
    fn type_requires(cx: ctxt, seen: &mut ~[def_id],
                     r_ty: t, ty: t) -> bool {
        debug!("type_requires(%s, %s)?",
               ::util::ppaux::ty_to_str(cx, r_ty),
               ::util::ppaux::ty_to_str(cx, ty));

        let r = {
            get(r_ty).sty == get(ty).sty ||
                subtypes_require(cx, seen, r_ty, ty)
        };

        debug!("type_requires(%s, %s)? %b",
               ::util::ppaux::ty_to_str(cx, r_ty),
               ::util::ppaux::ty_to_str(cx, ty),
               r);
        return r;
    }

    fn subtypes_require(cx: ctxt, seen: &mut ~[def_id],
                        r_ty: t, ty: t) -> bool {
        debug!("subtypes_require(%s, %s)?",
               ::util::ppaux::ty_to_str(cx, r_ty),
               ::util::ppaux::ty_to_str(cx, ty));

        let r = match get(ty).sty {
          ty_nil |
          ty_bot |
          ty_bool |
          ty_int(_) |
          ty_uint(_) |
          ty_float(_) |
          ty_estr(_) |
          ty_bare_fn(_) |
          ty_closure(_) |
          ty_infer(_) |
          ty_err |
          ty_param(_) |
          ty_self(_) |
          ty_type |
          ty_opaque_box |
          ty_opaque_closure_ptr(_) |
          ty_evec(_, _) |
          ty_unboxed_vec(_) => {
            false
          }
          ty_box(ref mt) |
          ty_uniq(ref mt) |
          ty_rptr(_, ref mt) => {
            return type_requires(cx, seen, r_ty, mt.ty);
          }

          ty_ptr(*) => {
            false           // unsafe ptrs can always be NULL
          }

          ty_trait(_, _, _) => {
            false
          }

          ty_struct(ref did, _) if vec::contains(*seen, did) => {
            false
          }

          ty_struct(did, ref substs) => {
              seen.push(did);
              let r = vec::any(struct_fields(cx, did, substs),
                               |f| type_requires(cx, seen, r_ty, f.mt.ty));
              seen.pop();
            r
          }

          ty_tup(ref ts) => {
            ts.any(|t| type_requires(cx, seen, r_ty, *t))
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
               ::util::ppaux::ty_to_str(cx, r_ty),
               ::util::ppaux::ty_to_str(cx, ty),
               r);

        return r;
    }

    let seen = @mut ~[];
    !subtypes_require(cx, seen, r_ty, r_ty)
}

pub fn type_structurally_contains(cx: ctxt,
                                  ty: t,
                                  test: &fn(x: &sty) -> bool)
                               -> bool {
    let sty = &get(ty).sty;
    debug!("type_structurally_contains: %s",
           ::util::ppaux::ty_to_str(cx, ty));
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
      ty_struct(did, ref substs) => {
        for lookup_struct_fields(cx, did).each |field| {
            let ft = lookup_field_type(cx, did, field.id, substs);
            if type_structurally_contains(cx, ft, test) { return true; }
        }
        return false;
      }

      ty_tup(ref ts) => {
        for ts.each |tt| {
            if type_structurally_contains(cx, *tt, test) { return true; }
        }
        return false;
      }
      ty_evec(ref mt, vstore_fixed(_)) => {
        return type_structurally_contains(cx, mt.ty, test);
      }
      _ => return false
    }
}

pub fn type_structurally_contains_uniques(cx: ctxt, ty: t) -> bool {
    return type_structurally_contains(cx, ty, |sty| {
        match *sty {
          ty_uniq(_) |
          ty_evec(_, vstore_uniq) |
          ty_estr(vstore_uniq) => true,
          _ => false,
        }
    });
}

pub fn type_is_integral(ty: t) -> bool {
    match get(ty).sty {
      ty_infer(IntVar(_)) | ty_int(_) | ty_uint(_) => true,
      _ => false
    }
}

pub fn type_is_char(ty: t) -> bool {
    match get(ty).sty {
        ty_int(ty_char) => true,
        _ => false
    }
}

pub fn type_is_fp(ty: t) -> bool {
    match get(ty).sty {
      ty_infer(FloatVar(_)) | ty_float(_) => true,
      _ => false
    }
}

pub fn type_is_numeric(ty: t) -> bool {
    return type_is_integral(ty) || type_is_fp(ty);
}

pub fn type_is_signed(ty: t) -> bool {
    match get(ty).sty {
      ty_int(_) => true,
      _ => false
    }
}

// Whether a type is Plain Old Data -- meaning it does not contain pointers
// that the cycle collector might care about.
pub fn type_is_pod(cx: ctxt, ty: t) -> bool {
    let mut result = true;
    match get(ty).sty {
      // Scalar types
      ty_nil | ty_bot | ty_bool | ty_int(_) | ty_float(_) | ty_uint(_) |
      ty_type | ty_ptr(_) | ty_bare_fn(_) => result = true,
      // Boxed types
      ty_box(_) | ty_uniq(_) | ty_closure(_) |
      ty_estr(vstore_uniq) | ty_estr(vstore_box) |
      ty_evec(_, vstore_uniq) | ty_evec(_, vstore_box) |
      ty_trait(_, _, _) | ty_rptr(_,_) | ty_opaque_box => result = false,
      // Structural types
      ty_enum(did, ref substs) => {
        let variants = enum_variants(cx, did);
        for vec::each(*variants) |variant| {
            let tup_ty = mk_tup(cx, /*bad*/copy variant.args);

            // Perform any type parameter substitutions.
            let tup_ty = subst(cx, substs, tup_ty);
            if !type_is_pod(cx, tup_ty) { result = false; }
        }
      }
      ty_tup(ref elts) => {
        for elts.each |elt| { if !type_is_pod(cx, *elt) { result = false; } }
      }
      ty_estr(vstore_fixed(_)) => result = true,
      ty_evec(ref mt, vstore_fixed(_)) | ty_unboxed_vec(ref mt) => {
        result = type_is_pod(cx, mt.ty);
      }
      ty_param(_) => result = false,
      ty_opaque_closure_ptr(_) => result = true,
      ty_struct(did, ref substs) => {
        result = vec::any(lookup_struct_fields(cx, did), |f| {
            let fty = ty::lookup_item_type(cx, f.id);
            let sty = subst(cx, substs, fty.ty);
            type_is_pod(cx, sty)
        });
      }

      ty_estr(vstore_slice(*)) | ty_evec(_, vstore_slice(*)) => {
        result = false;
      }

      ty_infer(*) | ty_self(*) | ty_err => {
        cx.sess.bug(~"non concrete type in type_is_pod");
      }
    }

    return result;
}

pub fn type_is_enum(ty: t) -> bool {
    match get(ty).sty {
      ty_enum(_, _) => return true,
      _ => return false
    }
}

// Whether a type is enum like, that is a enum type with only nullary
// constructors
pub fn type_is_c_like_enum(cx: ctxt, ty: t) -> bool {
    match get(ty).sty {
      ty_enum(did, _) => {
        let variants = enum_variants(cx, did);
        let some_n_ary = vec::any(*variants, |v| vec::len(v.args) > 0u);
        return !some_n_ary;
      }
      _ => return false
    }
}

pub fn type_param(ty: t) -> Option<uint> {
    match get(ty).sty {
      ty_param(p) => return Some(p.idx),
      _ => {/* fall through */ }
    }
    return None;
}

// Returns the type and mutability of *t.
//
// The parameter `explicit` indicates if this is an *explicit* dereference.
// Some types---notably unsafe ptrs---can only be dereferenced explicitly.
pub fn deref(cx: ctxt, t: t, explicit: bool) -> Option<mt> {
    deref_sty(cx, &get(t).sty, explicit)
}

pub fn deref_sty(cx: ctxt, sty: &sty, explicit: bool) -> Option<mt> {
    match *sty {
      ty_rptr(_, mt) | ty_box(mt) | ty_uniq(mt) => {
        Some(mt)
      }

      ty_ptr(mt) if explicit => {
        Some(mt)
      }

      ty_enum(did, ref substs) => {
        let variants = enum_variants(cx, did);
        if vec::len(*variants) == 1u && vec::len(variants[0].args) == 1u {
            let v_t = subst(cx, substs, variants[0].args[0]);
            Some(mt {ty: v_t, mutbl: ast::m_imm})
        } else {
            None
        }
      }

      ty_struct(did, ref substs) => {
        let fields = struct_fields(cx, did, substs);
        if fields.len() == 1 && fields[0].ident ==
                syntax::parse::token::special_idents::unnamed_field {
            Some(mt {ty: fields[0].mt.ty, mutbl: ast::m_imm})
        } else {
            None
        }
      }

      _ => None
    }
}

pub fn type_autoderef(cx: ctxt, t: t) -> t {
    let mut t = t;
    loop {
        match deref(cx, t, false) {
          None => return t,
          Some(mt) => t = mt.ty
        }
    }
}

// Returns the type and mutability of t[i]
pub fn index(cx: ctxt, t: t) -> Option<mt> {
    index_sty(cx, &get(t).sty)
}

pub fn index_sty(cx: ctxt, sty: &sty) -> Option<mt> {
    match *sty {
      ty_evec(mt, _) => Some(mt),
      ty_estr(_) => Some(mt {ty: mk_u8(cx), mutbl: ast::m_imm}),
      _ => None
    }
}

impl to_bytes::IterBytes for bound_region {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        match *self {
          ty::br_self => 0u8.iter_bytes(lsb0, f),

          ty::br_anon(ref idx) =>
          to_bytes::iter_bytes_2(&1u8, idx, lsb0, f),

          ty::br_named(ref ident) =>
          to_bytes::iter_bytes_2(&2u8, ident, lsb0, f),

          ty::br_cap_avoid(ref id, ref br) =>
          to_bytes::iter_bytes_3(&3u8, id, br, lsb0, f),

          ty::br_fresh(ref x) =>
          to_bytes::iter_bytes_2(&4u8, x, lsb0, f)
        }
    }
}

impl to_bytes::IterBytes for Region {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        match *self {
          re_bound(ref br) =>
          to_bytes::iter_bytes_2(&0u8, br, lsb0, f),

          re_free(ref id, ref br) =>
          to_bytes::iter_bytes_3(&1u8, id, br, lsb0, f),

          re_scope(ref id) =>
          to_bytes::iter_bytes_2(&2u8, id, lsb0, f),

          re_infer(ref id) =>
          to_bytes::iter_bytes_2(&3u8, id, lsb0, f),

          re_static => 4u8.iter_bytes(lsb0, f)
        }
    }
}

impl to_bytes::IterBytes for vstore {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        match *self {
          vstore_fixed(ref u) =>
          to_bytes::iter_bytes_2(&0u8, u, lsb0, f),

          vstore_uniq => 1u8.iter_bytes(lsb0, f),
          vstore_box => 2u8.iter_bytes(lsb0, f),

          vstore_slice(ref r) =>
          to_bytes::iter_bytes_2(&3u8, r, lsb0, f),
        }
    }
}

impl to_bytes::IterBytes for TraitStore {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        match *self {
          BareTraitStore => 0u8.iter_bytes(lsb0, f),
          UniqTraitStore => 1u8.iter_bytes(lsb0, f),
          BoxTraitStore => 2u8.iter_bytes(lsb0, f),
          RegionTraitStore(ref r) => to_bytes::iter_bytes_2(&3u8, r, lsb0, f),
        }
    }
}

impl to_bytes::IterBytes for substs {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
          to_bytes::iter_bytes_3(&self.self_r,
                                 &self.self_ty,
                                 &self.tps, lsb0, f)
    }
}

impl to_bytes::IterBytes for mt {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
          to_bytes::iter_bytes_2(&self.ty,
                                 &self.mutbl, lsb0, f)
    }
}

impl to_bytes::IterBytes for field {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
          to_bytes::iter_bytes_2(&self.ident,
                                 &self.mt, lsb0, f)
    }
}

impl to_bytes::IterBytes for arg {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        to_bytes::iter_bytes_2(&self.mode,
                               &self.ty, lsb0, f)
    }
}

impl to_bytes::IterBytes for FnSig {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        to_bytes::iter_bytes_2(&self.inputs,
                               &self.output,
                               lsb0, f);
    }
}

impl to_bytes::IterBytes for sty {
    fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        match *self {
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

          ty_bare_fn(ref ft) =>
          to_bytes::iter_bytes_2(&12u8, ft, lsb0, f),

          ty_self(ref did) => to_bytes::iter_bytes_2(&13u8, did, lsb0, f),

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

          ty_struct(ref did, ref substs) =>
          to_bytes::iter_bytes_3(&23u8, did, substs, lsb0, f),

          ty_rptr(ref r, ref mt) =>
          to_bytes::iter_bytes_3(&24u8, r, mt, lsb0, f),

          ty_err => 25u8.iter_bytes(lsb0, f),

          ty_closure(ref ct) =>
          to_bytes::iter_bytes_2(&26u8, ct, lsb0, f),
        }
    }
}

pub fn node_id_to_type(cx: ctxt, id: ast::node_id) -> t {
    //io::println(fmt!("%?/%?", id, cx.node_types.len()));
    match cx.node_types.find(&(id as uint)) {
       Some(&t) => t,
       None => cx.sess.bug(
           fmt!("node_id_to_type: no type for node `%s`",
                ast_map::node_id_to_str(cx.items, id,
                                        cx.sess.parse_sess.interner)))
    }
}

pub fn node_id_to_type_params(cx: ctxt, id: ast::node_id) -> ~[t] {
    match cx.node_type_substs.find(&id) {
      None => return ~[],
      Some(ts) => return /*bad*/ copy *ts
    }
}

fn node_id_has_type_params(cx: ctxt, id: ast::node_id) -> bool {
    cx.node_type_substs.contains_key(&id)
}

// Type accessors for substructures of types
pub fn ty_fn_args(fty: t) -> ~[arg] {
    match get(fty).sty {
        ty_bare_fn(ref f) => copy f.sig.inputs,
        ty_closure(ref f) => copy f.sig.inputs,
        ref s => {
            fail!(fmt!("ty_fn_args() called on non-fn type: %?", s))
        }
    }
}

pub fn ty_closure_sigil(fty: t) -> Sigil {
    match get(fty).sty {
        ty_closure(ref f) => f.sigil,
        ref s => {
            fail!(fmt!("ty_closure_sigil() called on non-closure type: %?",
                       s))
        }
    }
}

pub fn ty_fn_purity(fty: t) -> ast::purity {
    match get(fty).sty {
        ty_bare_fn(ref f) => f.purity,
        ty_closure(ref f) => f.purity,
        ref s => {
            fail!(fmt!("ty_fn_purity() called on non-fn type: %?", s))
        }
    }
}

pub fn ty_fn_ret(fty: t) -> t {
    match get(fty).sty {
        ty_bare_fn(ref f) => f.sig.output,
        ty_closure(ref f) => f.sig.output,
        ref s => {
            fail!(fmt!("ty_fn_ret() called on non-fn type: %?", s))
        }
    }
}

pub fn is_fn_ty(fty: t) -> bool {
    match get(fty).sty {
        ty_bare_fn(_) => true,
        ty_closure(_) => true,
        _ => false
    }
}

pub fn ty_vstore(ty: t) -> vstore {
    match get(ty).sty {
        ty_evec(_, vstore) => vstore,
        ty_estr(vstore) => vstore,
        ref s => fail!(fmt!("ty_vstore() called on invalid sty: %?", s))
    }
}

pub fn ty_region(tcx: ctxt,
                 span: span,
                 ty: t) -> Region {
    match get(ty).sty {
        ty_rptr(r, _) => r,
        ty_evec(_, vstore_slice(r)) => r,
        ty_estr(vstore_slice(r)) => r,
        ref s => {
            tcx.sess.span_bug(
                span,
                fmt!("ty_region() invoked on in appropriate ty: %?", s));
        }
    }
}

pub fn replace_closure_return_type(tcx: ctxt, fn_type: t, ret_type: t) -> t {
    /*!
     *
     * Returns a new function type based on `fn_type` but returning a value of
     * type `ret_type` instead. */

    match ty::get(fn_type).sty {
        ty::ty_closure(ref fty) => {
            ty::mk_closure(tcx, ClosureTy {
                sig: FnSig {output: ret_type, ..copy fty.sig},
                ..copy *fty
            })
        }
        _ => {
            tcx.sess.bug(fmt!(
                "replace_fn_ret() invoked with non-fn-type: %s",
                ty_to_str(tcx, fn_type)));
        }
    }
}

// Returns a vec of all the input and output types of fty.
pub fn tys_in_fn_sig(sig: &FnSig) -> ~[t] {
    vec::append_one(sig.inputs.map(|a| a.ty), sig.output)
}

// Type accessors for AST nodes
pub fn block_ty(cx: ctxt, b: &ast::blk) -> t {
    return node_id_to_type(cx, b.node.id);
}


// Returns the type of a pattern as a monotype. Like @expr_ty, this function
// doesn't provide type parameter substitutions.
pub fn pat_ty(cx: ctxt, pat: @ast::pat) -> t {
    return node_id_to_type(cx, pat.id);
}


// Returns the type of an expression as a monotype.
//
// NB (1): This is the PRE-ADJUSTMENT TYPE for the expression.  That is, in
// some cases, we insert `AutoAdjustment` annotations such as auto-deref or
// auto-ref.  The type returned by this function does not consider such
// adjustments.  See `expr_ty_adjusted()` instead.
//
// NB (2): This type doesn't provide type parameter substitutions; e.g. if you
// ask for the type of "id" in "id(3)", it will return "fn(&int) -> int"
// instead of "fn(t) -> T with T = int". If this isn't what you want, see
// expr_ty_params_and_ty() below.
pub fn expr_ty(cx: ctxt, expr: @ast::expr) -> t {
    return node_id_to_type(cx, expr.id);
}

pub fn expr_ty_adjusted(cx: ctxt, expr: @ast::expr) -> t {
    /*!
     *
     * Returns the type of `expr`, considering any `AutoAdjustment`
     * entry recorded for that expression.
     *
     * It would almost certainly be better to store the adjusted ty in with
     * the `AutoAdjustment`, but I opted not to do this because it would
     * require serializing and deserializing the type and, although that's not
     * hard to do, I just hate that code so much I didn't want to touch it
     * unless it was to fix it properly, which seemed a distraction from the
     * task at hand! -nmatsakis
     */

    let unadjusted_ty = expr_ty(cx, expr);

    return match cx.adjustments.find(&expr.id) {
        None => unadjusted_ty,

        Some(&@AutoAddEnv(r, s)) => {
            match ty::get(unadjusted_ty).sty {
                ty::ty_bare_fn(ref b) => {
                    ty::mk_closure(
                        cx,
                        ty::ClosureTy {purity: b.purity,
                                       sigil: s,
                                       onceness: ast::Many,
                                       region: r,
                                       sig: copy b.sig})
                }
                ref b => {
                    cx.sess.bug(
                        fmt!("add_env adjustment on non-bare-fn: %?", b));
                }
            }
        }

        Some(&@AutoDerefRef(ref adj)) => {
            let mut adjusted_ty = unadjusted_ty;

            for uint::range(0, adj.autoderefs) |i| {
                match ty::deref(cx, adjusted_ty, true) {
                    Some(mt) => { adjusted_ty = mt.ty; }
                    None => {
                        cx.sess.span_bug(
                            expr.span,
                            fmt!("The %uth autoderef failed: %s",
                                 i, ty_to_str(cx,
                                              adjusted_ty)));
                    }
                }
            }

            match adj.autoref {
                None => adjusted_ty,
                Some(ref autoref) => {
                    match autoref.kind {
                        AutoPtr => {
                            mk_rptr(cx, autoref.region,
                                    mt {ty: adjusted_ty,
                                        mutbl: autoref.mutbl})
                        }

                        AutoBorrowVec => {
                            borrow_vec(cx, expr, autoref, adjusted_ty)
                        }

                        AutoBorrowVecRef => {
                            adjusted_ty = borrow_vec(cx, expr, autoref,
                                                     adjusted_ty);
                            mk_rptr(cx, autoref.region,
                                    mt {ty: adjusted_ty, mutbl: ast::m_imm})
                        }

                        AutoBorrowFn => {
                            borrow_fn(cx, expr, autoref, adjusted_ty)
                        }
                    }
                }
            }
        }
    };

    fn borrow_vec(cx: ctxt, expr: @ast::expr,
                  autoref: &AutoRef, ty: ty::t) -> ty::t {
        match get(ty).sty {
            ty_evec(mt, _) => {
                ty::mk_evec(cx, mt {ty: mt.ty, mutbl: autoref.mutbl},
                            vstore_slice(autoref.region))
            }

            ty_estr(_) => {
                ty::mk_estr(cx, vstore_slice(autoref.region))
            }

            ref s => {
                cx.sess.span_bug(
                    expr.span,
                    fmt!("borrow-vec associated with bad sty: %?",
                         s));
            }
        }
    }

    fn borrow_fn(cx: ctxt, expr: @ast::expr,
                 autoref: &AutoRef, ty: ty::t) -> ty::t {
        match get(ty).sty {
            ty_closure(ref fty) => {
                ty::mk_closure(cx, ClosureTy {
                    sigil: BorrowedSigil,
                    region: autoref.region,
                    ..copy *fty
                })
            }

            ref s => {
                cx.sess.span_bug(
                    expr.span,
                    fmt!("borrow-fn associated with bad sty: %?",
                         s));
            }
        }
    }
}

pub struct ParamsTy {
    params: ~[t],
    ty: t
}

pub fn expr_ty_params_and_ty(cx: ctxt,
                             expr: @ast::expr)
                          -> ParamsTy {
    ParamsTy {
        params: node_id_to_type_params(cx, expr.id),
        ty: node_id_to_type(cx, expr.id)
    }
}

pub fn expr_has_ty_params(cx: ctxt, expr: @ast::expr) -> bool {
    return node_id_has_type_params(cx, expr.id);
}

pub fn method_call_bounds(tcx: ctxt, method_map: typeck::method_map,
                          id: ast::node_id)
    -> Option<@~[param_bounds]> {
    do method_map.find(&id).map |method| {
        match method.origin {
          typeck::method_static(did) => {
            // n.b.: When we encode impl methods, the bounds
            // that we encode include both the impl bounds
            // and then the method bounds themselves...
            ty::lookup_item_type(tcx, did).bounds
          }
          typeck::method_param(typeck::method_param {
              trait_id: trt_id,
              method_num: n_mth, _}) |
          typeck::method_trait(trt_id, n_mth, _) |
          typeck::method_self(trt_id, n_mth) |
          typeck::method_super(trt_id, n_mth) => {
            // ...trait methods bounds, in contrast, include only the
            // method bounds, so we must preprend the tps from the
            // trait itself.  This ought to be harmonized.
            let trt_bounds =
                ty::lookup_item_type(tcx, trt_id).bounds;
            @(vec::append(/*bad*/copy *trt_bounds,
                          *ty::trait_methods(tcx, trt_id)[n_mth].tps))
          }
        }
    }
}

pub fn resolve_expr(tcx: ctxt, expr: @ast::expr) -> ast::def {
    match tcx.def_map.find(&expr.id) {
        Some(&def) => def,
        None => {
            tcx.sess.span_bug(expr.span, fmt!(
                "No def-map entry for expr %?", expr.id));
        }
    }
}

pub fn expr_is_lval(tcx: ctxt,
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
pub enum ExprKind {
    LvalueExpr,
    RvalueDpsExpr,
    RvalueDatumExpr,
    RvalueStmtExpr
}

pub fn expr_kind(tcx: ctxt,
                 method_map: typeck::method_map,
                 expr: @ast::expr) -> ExprKind {
    if method_map.contains_key(&expr.id) {
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
                ast::def_variant(*) | ast::def_struct(*) => RvalueDpsExpr,

                // Fn pointers are just scalar values.
                ast::def_fn(*) | ast::def_static_method(*) => RvalueDatumExpr,

                // Note: there is actually a good case to be made that
                // def_args, particularly those of immediate type, ought to
                // considered rvalues.
                ast::def_const(*) |
                ast::def_binding(*) |
                ast::def_upvar(*) |
                ast::def_arg(*) |
                ast::def_local(*) |
                ast::def_self(*) => LvalueExpr,

                def => {
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
        ast::expr_method_call(*) |
        ast::expr_struct(*) |
        ast::expr_tup(*) |
        ast::expr_if(*) |
        ast::expr_match(*) |
        ast::expr_fn_block(*) |
        ast::expr_loop_body(*) |
        ast::expr_do_body(*) |
        ast::expr_block(*) |
        ast::expr_copy(*) |
        ast::expr_repeat(*) |
        ast::expr_lit(@codemap::spanned {node: lit_str(_), _}) |
        ast::expr_vstore(_, ast::expr_vstore_slice) |
        ast::expr_vstore(_, ast::expr_vstore_mut_slice) |
        ast::expr_vstore(_, ast::expr_vstore_fixed(_)) |
        ast::expr_vec(*) => {
            RvalueDpsExpr
        }

        ast::expr_cast(*) => {
            match tcx.node_types.find(&(expr.id as uint)) {
                Some(&t) => {
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
        ast::expr_while(*) |
        ast::expr_loop(*) |
        ast::expr_assign(*) |
        ast::expr_swap(*) |
        ast::expr_inline_asm(*) |
        ast::expr_assign_op(*) => {
            RvalueStmtExpr
        }

        ast::expr_lit(_) | // Note: lit_str is carved out above
        ast::expr_unary(*) |
        ast::expr_addr_of(*) |
        ast::expr_binary(*) |
        ast::expr_vstore(_, ast::expr_vstore_box) |
        ast::expr_vstore(_, ast::expr_vstore_mut_box) |
        ast::expr_vstore(_, ast::expr_vstore_uniq) => {
            RvalueDatumExpr
        }

        ast::expr_paren(e) => expr_kind(tcx, method_map, e),

        ast::expr_mac(*) => {
            tcx.sess.span_bug(
                expr.span,
                ~"macro expression remains after expansion");
        }
    }
}

pub fn stmt_node_id(s: @ast::stmt) -> ast::node_id {
    match s.node {
      ast::stmt_decl(_, id) | stmt_expr(_, id) | stmt_semi(_, id) => {
        return id;
      }
      ast::stmt_mac(*) => fail!(~"unexpanded macro in trans")
    }
}

pub fn field_idx(id: ast::ident, fields: &[field]) -> Option<uint> {
    let mut i = 0u;
    for fields.each |f| { if f.ident == id { return Some(i); } i += 1u; }
    return None;
}

pub fn field_idx_strict(tcx: ty::ctxt, id: ast::ident, fields: &[field])
                     -> uint {
    let mut i = 0u;
    for fields.each |f| { if f.ident == id { return i; } i += 1u; }
    tcx.sess.bug(fmt!(
        "No field named `%s` found in the list of fields `%?`",
        *tcx.sess.str_of(id),
        fields.map(|f| tcx.sess.str_of(f.ident))));
}

pub fn method_idx(id: ast::ident, meths: &[method]) -> Option<uint> {
    let mut i = 0u;
    for meths.each |m| { if m.ident == id { return Some(i); } i += 1u; }
    return None;
}

/// Returns a vector containing the indices of all type parameters that appear
/// in `ty`.  The vector may contain duplicates.  Probably should be converted
/// to a bitset or some other representation.
pub fn param_tys_in_type(ty: t) -> ~[param_ty] {
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

pub fn occurs_check(tcx: ctxt, sp: span, vid: TyVid, rt: t) {
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
                 + ::util::ppaux::ty_to_str(tcx, mk_var(tcx, vid)) +
                 ~" and of the form " + ::util::ppaux::ty_to_str(tcx, rt) +
                 ~" - such a type would have to be infinitely large.");
    }
}

// Maintains a little union-set tree for inferred modes.  `canon()` returns
// the current head value for `m0`.
fn canon<T:Copy + cmp::Eq>(tbl: &mut LinearMap<ast::node_id, ast::inferable<T>>,
                         +m0: ast::inferable<T>) -> ast::inferable<T> {
    match m0 {
        ast::infer(id) => {
            let m1 = match tbl.find(&id) {
                None => return m0,
                Some(&m1) => m1
            };
            let cm1 = canon(tbl, m1);
            // path compression:
            if cm1 != m1 { tbl.insert(id, cm1); }
            cm1
        },
        _ => m0
    }
}

// Maintains a little union-set tree for inferred modes.  `resolve_mode()`
// returns the current head value for `m0`.
pub fn canon_mode(cx: ctxt, m0: ast::mode) -> ast::mode {
    canon(cx.inferred_modes, m0)
}

// Returns the head value for mode, failing if `m` was a infer(_) that
// was never inferred.  This should be safe for use after typeck.
pub fn resolved_mode(cx: ctxt, m: ast::mode) -> ast::rmode {
    match canon_mode(cx, m) {
      ast::infer(_) => {
        cx.sess.bug(fmt!("mode %? was never resolved", m));
      }
      ast::expl(m0) => m0
    }
}

pub fn arg_mode(cx: ctxt, a: arg) -> ast::rmode { resolved_mode(cx, a.mode) }

// Unifies `m1` and `m2`.  Returns unified value or failure code.
pub fn unify_mode(cx: ctxt, modes: expected_found<ast::mode>)
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
pub fn set_default_mode(cx: ctxt, m: ast::mode, m_def: ast::rmode) {
    match canon_mode(cx, m) {
      ast::infer(id) => {
        cx.inferred_modes.insert(id, ast::expl(m_def));
      }
      ast::expl(_) => ()
    }
}

pub fn ty_sort_str(cx: ctxt, t: t) -> ~str {
    match get(t).sty {
      ty_nil | ty_bot | ty_bool | ty_int(_) |
      ty_uint(_) | ty_float(_) | ty_estr(_) |
      ty_type | ty_opaque_box | ty_opaque_closure_ptr(_) => {
        ::util::ppaux::ty_to_str(cx, t)
      }

      ty_enum(id, _) => fmt!("enum %s", item_path_str(cx, id)),
      ty_box(_) => ~"@-ptr",
      ty_uniq(_) => ~"~-ptr",
      ty_evec(_, _) => ~"vector",
      ty_unboxed_vec(_) => ~"unboxed vector",
      ty_ptr(_) => ~"*-ptr",
      ty_rptr(_, _) => ~"&-ptr",
      ty_bare_fn(_) => ~"extern fn",
      ty_closure(_) => ~"fn",
      ty_trait(id, _, _) => fmt!("trait %s", item_path_str(cx, id)),
      ty_struct(id, _) => fmt!("struct %s", item_path_str(cx, id)),
      ty_tup(_) => ~"tuple",
      ty_infer(TyVar(_)) => ~"inferred type",
      ty_infer(IntVar(_)) => ~"integral variable",
      ty_infer(FloatVar(_)) => ~"floating-point variable",
      ty_param(_) => ~"type parameter",
      ty_self(_) => ~"self",
      ty_err => ~"type error"
    }
}

pub fn type_err_to_str(cx: ctxt, err: &type_err) -> ~str {
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
        terr_purity_mismatch(values) => {
            fmt!("expected %s fn but found %s fn",
                 values.expected.to_str(), values.found.to_str())
        }
        terr_abi_mismatch(values) => {
            fmt!("expected %s fn but found %s fn",
                 values.expected.to_str(), values.found.to_str())
        }
        terr_onceness_mismatch(values) => {
            fmt!("expected %s fn but found %s fn",
                 values.expected.to_str(), values.found.to_str())
        }
        terr_sigil_mismatch(values) => {
            fmt!("expected %s closure, found %s closure",
                 values.expected.to_str(),
                 values.found.to_str())
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
                 *cx.sess.str_of(values.expected),
                 *cx.sess.str_of(values.found))
        }
        terr_arg_count => ~"incorrect number of function parameters",
        terr_mode_mismatch(values) => {
            fmt!("expected argument mode %s, but found %s",
                 pprust::mode_to_str(values.expected),
                 pprust::mode_to_str(values.found))
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
        terr_regions_insufficiently_polymorphic(br, _) => {
            fmt!("expected bound lifetime parameter %s, \
                  but found concrete lifetime",
                 bound_region_to_str(cx, br))
        }
        terr_regions_overly_polymorphic(br, _) => {
            fmt!("expected concrete lifetime, \
                  but found bound lifetime parameter %s",
                 bound_region_to_str(cx, br))
        }
        terr_vstores_differ(k, ref values) => {
            fmt!("%s storage differs: expected %s but found %s",
                 terr_vstore_kind_to_str(k),
                 vstore_to_str(cx, (*values).expected),
                 vstore_to_str(cx, (*values).found))
        }
        terr_trait_stores_differ(_, ref values) => {
            fmt!("trait storage differs: expected %s but found %s",
                 trait_store_to_str(cx, (*values).expected),
                 trait_store_to_str(cx, (*values).found))
        }
        terr_in_field(err, fname) => {
            fmt!("in field `%s`, %s", *cx.sess.str_of(fname),
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
        terr_integer_as_char => {
            fmt!("expected an integral type but found char")
        }
        terr_int_mismatch(ref values) => {
            fmt!("expected %s but found %s",
                 values.expected.to_str(),
                 values.found.to_str())
        }
        terr_float_mismatch(ref values) => {
            fmt!("expected %s but found %s",
                 values.expected.to_str(),
                 values.found.to_str())
        }
    }
}

pub fn note_and_explain_type_err(cx: ctxt, err: &type_err) {
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
        terr_regions_insufficiently_polymorphic(_, conc_region) => {
            note_and_explain_region(cx,
                                    ~"concrete lifetime that was found is ",
                                    conc_region, ~"");
        }
        terr_regions_overly_polymorphic(_, conc_region) => {
            note_and_explain_region(cx,
                                    ~"expected concrete lifetime is ",
                                    conc_region, ~"");
        }
        _ => {}
    }
}

pub fn def_has_ty_params(def: ast::def) -> bool {
    match def {
      ast::def_fn(_, _) | ast::def_variant(_, _) | ast::def_struct(_)
        => true,
      _ => false
    }
}

pub fn store_trait_methods(cx: ctxt, id: ast::node_id, ms: @~[method]) {
    cx.trait_method_cache.insert(ast_util::local_def(id), ms);
}

pub fn provided_trait_methods(cx: ctxt, id: ast::def_id) -> ~[ast::ident] {
    if is_local(id) {
        match cx.items.find(&id.node) {
            Some(&ast_map::node_item(@ast::item {
                        node: item_trait(_, _, ref ms),
                        _
                    }, _)) =>
                match ast_util::split_trait_methods(*ms) {
                   (_, p) => p.map(|method| method.ident)
                },
            _ => cx.sess.bug(fmt!("provided_trait_methods: %? is not a trait",
                                  id))
        }
    } else {
        csearch::get_provided_trait_methods(cx, id).map(|ifo| ifo.ty.ident)
    }
}

pub fn trait_supertraits(cx: ctxt,
                         id: ast::def_id)
                      -> @~[InstantiatedTraitRef] {
    // Check the cache.
    match cx.supertraits.find(&id) {
        Some(&instantiated_trait_info) => { return instantiated_trait_info; }
        None => {}  // Continue.
    }

    // Not in the cache. It had better be in the metadata, which means it
    // shouldn't be local.
    fail_unless!(!is_local(id));

    // Get the supertraits out of the metadata and create the
    // InstantiatedTraitRef for each.
    let mut result = ~[];
    for csearch::get_supertraits(cx, id).each |trait_type| {
        match get(*trait_type).sty {
            ty_trait(def_id, ref substs, _) => {
                result.push(InstantiatedTraitRef {
                    def_id: def_id,
                    tpt: ty_param_substs_and_ty {
                        substs: (/*bad*/copy *substs),
                        ty: *trait_type
                    }
                });
            }
            _ => cx.sess.bug(~"trait_supertraits: trait ref wasn't a trait")
        }
    }

    // Unwrap and return the result.
    return @result;
}

pub fn trait_methods(cx: ctxt, id: ast::def_id) -> @~[method] {
    match cx.trait_method_cache.find(&id) {
      // Local traits are supposed to have been added explicitly.
      Some(&ms) => ms,
      _ => {
        // If the lookup in trait_method_cache fails, assume that the trait
        // method we're trying to look up is in a different crate, and look
        // for it there.
        fail_unless!(id.crate != ast::local_crate);
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
pub fn impl_traits(cx: ctxt, id: ast::def_id, store: TraitStore) -> ~[t] {
    fn storeify(cx: ctxt, ty: t, store: TraitStore) -> t {
        match ty::get(ty).sty {
            ty::ty_trait(did, ref substs, trait_store) => {
                if store == trait_store {
                    ty
                } else {
                    mk_trait(cx, did, (/*bad*/copy *substs), store)
                }
            }
            _ => cx.sess.bug(~"impl_traits: not a trait")
        }
    }

    if id.crate == ast::local_crate {
        debug!("(impl_traits) searching for trait impl %?", id);
        match cx.items.find(&id.node) {
           Some(&ast_map::node_item(@ast::item {
                        node: ast::item_impl(_, opt_trait, _, _),
                        _},
                    _)) => {

               do opt_trait.map_default(~[]) |trait_ref| {
                   ~[storeify(cx, node_id_to_type(cx, trait_ref.ref_id),
                              store)]
               }
           }
           _ => ~[]
        }
    } else {
        vec::map(csearch::get_impl_traits(cx, id),
                 |x| storeify(cx, *x, store))
    }
}

pub fn ty_to_def_id(ty: t) -> Option<ast::def_id> {
    match get(ty).sty {
      ty_trait(id, _, _) | ty_struct(id, _) | ty_enum(id, _) => Some(id),
      _ => None
    }
}

/// Returns the def ID of the constructor for the given tuple-like struct, or
/// None if the struct is not tuple-like. Fails if the given def ID does not
/// refer to a struct at all.
fn struct_ctor_id(cx: ctxt, struct_did: ast::def_id) -> Option<ast::def_id> {
    if struct_did.crate != ast::local_crate {
        // XXX: Cross-crate functionality.
        cx.sess.unimpl(~"constructor ID of cross-crate tuple structs");
    }

    match cx.items.find(&struct_did.node) {
        Some(&ast_map::node_item(item, _)) => {
            match item.node {
                ast::item_struct(struct_def, _) => {
                    struct_def.ctor_id.map(|ctor_id|
                        ast_util::local_def(*ctor_id))
                }
                _ => cx.sess.bug(~"called struct_ctor_id on non-struct")
            }
        }
        _ => cx.sess.bug(~"called struct_ctor_id on non-struct")
    }
}

// Enum information
pub struct VariantInfo_ {
    args: ~[t],
    ctor_ty: t,
    name: ast::ident,
    id: ast::def_id,
    disr_val: int,
    vis: visibility
}

pub type VariantInfo = @VariantInfo_;

pub fn substd_enum_variants(cx: ctxt,
                            id: ast::def_id,
                            substs: &substs)
                         -> ~[VariantInfo] {
    do vec::map(*enum_variants(cx, id)) |variant_info| {
        let substd_args = vec::map(variant_info.args,
                                   |aty| subst(cx, substs, *aty));

        let substd_ctor_ty = subst(cx, substs, variant_info.ctor_ty);

        @VariantInfo_{args: substd_args, ctor_ty: substd_ctor_ty,
                      ../*bad*/copy **variant_info}
    }
}

pub fn item_path_str(cx: ctxt, id: ast::def_id) -> ~str {
    ast_map::path_to_str(item_path(cx, id), cx.sess.parse_sess.interner)
}

pub enum DtorKind {
    NoDtor,
    LegacyDtor(def_id),
    TraitDtor(def_id)
}

pub impl DtorKind {
    fn is_not_present(&const self) -> bool {
        match *self {
            NoDtor => true,
            _ => false
        }
    }
    fn is_present(&const self) -> bool {
        !self.is_not_present()
    }
}

/* If struct_id names a struct with a dtor, return Some(the dtor's id).
   Otherwise return none. */
pub fn ty_dtor(cx: ctxt, struct_id: def_id) -> DtorKind {
    match cx.destructor_for_type.find(&struct_id) {
        Some(&method_def_id) => return TraitDtor(method_def_id),
        None => {}  // Continue.
    }

    if is_local(struct_id) {
       match cx.items.find(&struct_id.node) {
           Some(&ast_map::node_item(@ast::item {
               node: ast::item_struct(@ast::struct_def { dtor: Some(ref dtor),
                                                         _ },
                                      _),
               _
           }, _)) =>
               LegacyDtor(local_def((*dtor).node.id)),
           _ =>
               NoDtor
       }
    }
    else {
      match csearch::struct_dtor(cx.sess.cstore, struct_id) {
        None => NoDtor,
        Some(did) => LegacyDtor(did),
      }
    }
}

pub fn has_dtor(cx: ctxt, struct_id: def_id) -> bool {
    ty_dtor(cx, struct_id).is_present()
}

pub fn item_path(cx: ctxt, id: ast::def_id) -> ast_map::path {
    if id.crate != ast::local_crate {
        csearch::get_item_path(cx, id)
    } else {
        // FIXME (#5521): uncomment this code and don't have a catch-all at the
        //                end of the match statement. Favor explicitly listing
        //                each variant.
        // let node = cx.items.get(&id.node);
        // match *node {
        match *cx.items.get(&id.node) {
          ast_map::node_item(item, path) => {
            let item_elt = match item.node {
              item_mod(_) | item_foreign_mod(_) => {
                ast_map::path_mod(item.ident)
              }
              _ => {
                ast_map::path_name(item.ident)
              }
            };
            vec::append_one(/*bad*/copy *path, item_elt)
          }

          ast_map::node_foreign_item(nitem, _, _, path) => {
            vec::append_one(/*bad*/copy *path,
                            ast_map::path_name(nitem.ident))
          }

          ast_map::node_method(method, _, path) => {
            vec::append_one(/*bad*/copy *path,
                            ast_map::path_name(method.ident))
          }
          ast_map::node_trait_method(trait_method, _, path) => {
            let method = ast_util::trait_method_to_ty_method(&*trait_method);
            vec::append_one(/*bad*/copy *path,
                            ast_map::path_name(method.ident))
          }

          ast_map::node_variant(ref variant, _, path) => {
            vec::append_one(vec::from_slice(vec::init(*path)),
                            ast_map::path_name((*variant).node.name))
          }

          ast_map::node_dtor(_, _, _, path) => {
            vec::append_one(/*bad*/copy *path, ast_map::path_name(
                syntax::parse::token::special_idents::literally_dtor))
          }

          ast_map::node_struct_ctor(_, item, path) => {
            vec::append_one(/*bad*/copy *path, ast_map::path_name(item.ident))
          }

          ref node => {
            cx.sess.bug(fmt!("cannot find item_path for node %?", node));
          }
        }
    }
}

pub fn enum_is_univariant(cx: ctxt, id: ast::def_id) -> bool {
    enum_variants(cx, id).len() == 1
}

pub fn type_is_empty(cx: ctxt, t: t) -> bool {
    match ty::get(t).sty {
       ty_enum(did, _) => (*enum_variants(cx, did)).is_empty(),
       _ => false
     }
}

pub fn enum_variants(cx: ctxt, id: ast::def_id) -> @~[VariantInfo] {
    match cx.enum_var_cache.find(&id) {
      Some(&variants) => return variants,
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
        match *cx.items.get(&id.node) {
          ast_map::node_item(@ast::item {
                    node: ast::item_enum(ref enum_definition, _),
                    _
                }, _) => {
            let mut disr_val = -1;
            @vec::map(enum_definition.variants, |variant| {
                match variant.node.kind {
                    ast::tuple_variant_kind(ref args) => {
                        let ctor_ty = node_id_to_type(cx, variant.node.id);
                        let arg_tys = {
                            if args.len() > 0u {
                                ty_fn_args(ctor_ty).map(|a| a.ty)
                            } else {
                                ~[]
                            }
                        };
                        match variant.node.disr_expr {
                          Some (ex) => {
                            disr_val = match const_eval::eval_const_expr(cx,
                                                                         ex) {
                              const_eval::const_int(val) => val as int,
                              _ => cx.sess.bug(~"tag_variants: bad disr expr")
                            }
                          }
                          _ => disr_val += 1
                        }
                        @VariantInfo_{args: arg_tys,
                          ctor_ty: ctor_ty,
                          name: variant.node.name,
                          id: ast_util::local_def(variant.node.id),
                          disr_val: disr_val,
                          vis: variant.node.vis
                         }
                    }
                    ast::struct_variant_kind(_) => {
                        fail!(~"struct variant kinds unimpl in enum_variants")
                    }
                    ast::enum_variant_kind(_) => {
                        fail!(~"enum variant kinds unimpl in enum_variants")
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
pub fn enum_variant_with_id(cx: ctxt,
                            enum_id: ast::def_id,
                            variant_id: ast::def_id)
                         -> VariantInfo {
    let variants = enum_variants(cx, enum_id);
    let mut i = 0;
    while i < variants.len() {
        let variant = variants[i];
        if variant.id == variant_id { return variant; }
        i += 1;
    }
    cx.sess.bug(~"enum_variant_with_id(): no variant exists with that ID");
}


// If the given item is in an external crate, looks up its type and adds it to
// the type cache. Returns the type parameters and type.
pub fn lookup_item_type(cx: ctxt,
                        did: ast::def_id)
                     -> ty_param_bounds_and_ty {
    match cx.tcache.find(&did) {
      Some(&tpt) => {
        // The item is in this crate. The caller should have added it to the
        // type cache already
        return tpt;
      }
      None => {
        fail_unless!(did.crate != ast::local_crate);
        let tyt = csearch::get_type(cx, did);
        cx.tcache.insert(did, tyt);
        return tyt;
      }
    }
}

// Look up a field ID, whether or not it's local
// Takes a list of type substs in case the struct is generic
pub fn lookup_field_type(tcx: ctxt,
                         struct_id: def_id,
                         id: def_id,
                         substs: &substs)
                      -> ty::t {
    let t = if id.crate == ast::local_crate {
        node_id_to_type(tcx, id.node)
    }
    else {
        match tcx.tcache.find(&id) {
           Some(tpt) => tpt.ty,
           None => {
               let tpt = csearch::get_field_type(tcx, struct_id, id);
               tcx.tcache.insert(id, tpt);
               tpt.ty
           }
        }
    };
    subst(tcx, substs, t)
}

// Look up the list of field names and IDs for a given struct
// Fails if the id is not bound to a struct.
pub fn lookup_struct_fields(cx: ctxt, did: ast::def_id) -> ~[field_ty] {
  if did.crate == ast::local_crate {
    match cx.items.find(&did.node) {
       Some(&ast_map::node_item(i,_)) => {
         match i.node {
            ast::item_struct(struct_def, _) => {
               struct_field_tys(struct_def.fields)
            }
            _ => cx.sess.bug(~"struct ID bound to non-struct")
         }
       }
       Some(&ast_map::node_variant(ref variant, _, _)) => {
          match (*variant).node.kind {
            ast::struct_variant_kind(struct_def) => {
              struct_field_tys(struct_def.fields)
            }
            _ => {
              cx.sess.bug(~"struct ID bound to enum variant that isn't \
                            struct-like")
            }
          }
       }
       _ => {
           cx.sess.bug(
               fmt!("struct ID not bound to an item: %s",
                    ast_map::node_id_to_str(cx.items, did.node,
                                            cx.sess.parse_sess.interner)));
       }
    }
        }
  else {
        return csearch::get_struct_fields(cx.sess.cstore, did);
    }
}

pub fn lookup_struct_field(cx: ctxt,
                           parent: ast::def_id,
                           field_id: ast::def_id)
                        -> field_ty {
    match vec::find(lookup_struct_fields(cx, parent),
                 |f| f.id.node == field_id.node) {
        Some(t) => t,
        None => cx.sess.bug(~"struct ID not found in parent's fields")
    }
}

fn is_public(f: field_ty) -> bool {
    // XXX: This is wrong.
    match f.vis {
        public | inherited => true,
        private => false
    }
}

fn struct_field_tys(fields: &[@struct_field]) -> ~[field_ty] {
    do fields.map |field| {
        match field.node.kind {
            named_field(ident, mutability, visibility) => {
                field_ty {
                    ident: ident,
                    id: ast_util::local_def(field.node.id),
                    vis: visibility,
                    mutability: mutability,
                }
            }
            unnamed_field => {
                field_ty {
                    ident:
                        syntax::parse::token::special_idents::unnamed_field,
                    id: ast_util::local_def(field.node.id),
                    vis: ast::public,
                    mutability: ast::struct_immutable,
                }
            }
        }
    }
}

// Return a list of fields corresponding to the struct's items
// (as if the struct was a record). trans uses this
// Takes a list of substs with which to instantiate field types
// Keep in mind that this function reports that all fields are
// mutable, regardless of how they were declared. It's meant to
// be used in trans.
pub fn struct_mutable_fields(cx: ctxt,
                             did: ast::def_id,
                             substs: &substs)
                          -> ~[field] {
    struct_item_fields(cx, did, substs, |_mt| m_mutbl)
}

// Same as struct_mutable_fields, but doesn't change
// mutability.
pub fn struct_fields(cx: ctxt,
                     did: ast::def_id,
                     substs: &substs)
                  -> ~[field] {
    struct_item_fields(cx, did, substs, |mt| match mt {
      struct_mutable => m_mutbl,
        struct_immutable => m_imm })
}


fn struct_item_fields(cx:ctxt,
                     did: ast::def_id,
                     substs: &substs,
                     frob_mutability: &fn(struct_mutability) -> mutability)
    -> ~[field] {
    do lookup_struct_fields(cx, did).map |f| {
       // consider all instance vars mut, because the
       // constructor may mutate all vars
       field {
           ident: f.ident,
            mt: mt {
                ty: lookup_field_type(cx, did, f.id, substs),
                mutbl: frob_mutability(f.mutability)
            }
        }
    }
}

pub fn is_binopable(_cx: ctxt, ty: t, op: ast::binop) -> bool {
    static tycat_other: int = 0;
    static tycat_bool: int = 1;
    static tycat_int: int = 2;
    static tycat_float: int = 3;
    static tycat_struct: int = 4;
    static tycat_bot: int = 5;

    static opcat_add: int = 0;
    static opcat_sub: int = 1;
    static opcat_mult: int = 2;
    static opcat_shift: int = 3;
    static opcat_rel: int = 4;
    static opcat_eq: int = 5;
    static opcat_bit: int = 6;
    static opcat_logic: int = 7;

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
          ty_float(_) | ty_infer(FloatVar(_)) => tycat_float,
          ty_tup(_) | ty_enum(_, _) => tycat_struct,
          ty_bot => tycat_bot,
          _ => tycat_other
        }
    }

    static t: bool = true;
    static f: bool = false;

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

pub fn ty_params_to_tys(tcx: ty::ctxt, generics: &ast::Generics) -> ~[t] {
    vec::from_fn(generics.ty_params.len(), |i| {
        let id = generics.ty_params.get(i).id;
        ty::mk_param(tcx, i, ast_util::local_def(id))
    })
}

/// Returns an equivalent type with all the typedefs and self regions removed.
pub fn normalize_ty(cx: ctxt, t: t) -> t {
    fn normalize_mt(cx: ctxt, mt: mt) -> mt {
        mt { ty: normalize_ty(cx, mt.ty), mutbl: mt.mutbl }
    }
    fn normalize_vstore(vstore: vstore) -> vstore {
        match vstore {
            vstore_fixed(*) | vstore_uniq | vstore_box => vstore,
            vstore_slice(_) => vstore_slice(re_static)
        }
    }

    match cx.normalized_cache.find(&t) {
      Some(&t) => return t,
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

        ty_closure(ref closure_ty) => {
            mk_closure(cx, ClosureTy {
                region: ty::re_static,
                ..copy *closure_ty
            })
        }

        ty_enum(did, ref r) =>
            match (*r).self_r {
                Some(_) =>
                    // Use re_static since trans doesn't care about regions
                    mk_enum(cx, did,
                     substs {
                        self_r: Some(ty::re_static),
                        self_ty: None,
                        tps: /*bad*/copy (*r).tps
                     }),
                None =>
                    t
            },

        ty_struct(did, ref r) =>
            match (*r).self_r {
              Some(_) =>
                // Ditto.
                mk_struct(cx, did, substs {self_r: Some(ty::re_static),
                                           self_ty: None,
                                           tps: /*bad*/copy (*r).tps}),
              None =>
                t
            },

        ty_trait(did, ref substs, BareTraitStore) =>
            mk_trait(cx, did, copy *substs, BoxTraitStore),

        _ =>
            t
    };

    let sty = fold_sty(&get(t).sty, |t| { normalize_ty(cx, t) });
    let t_norm = mk_t(cx, sty);
    cx.normalized_cache.insert(t, t_norm);
    return t_norm;
}

// Returns the repeat count for a repeating vector expression.
pub fn eval_repeat_count(tcx: ctxt, count_expr: @ast::expr) -> uint {
    match const_eval::eval_const_expr_partial(tcx, count_expr) {
      Ok(ref const_val) => match *const_val {
        const_eval::const_int(count) => return count as uint,
        const_eval::const_uint(count) => return count as uint,
        const_eval::const_float(count) => {
            tcx.sess.span_err(count_expr.span,
                              ~"expected signed or unsigned integer for \
                                repeat count but found float");
            return count as uint;
        }
        const_eval::const_str(_) => {
            tcx.sess.span_err(count_expr.span,
                              ~"expected signed or unsigned integer for \
                                repeat count but found string");
            return 0;
        }
        const_eval::const_bool(_) => {
            tcx.sess.span_err(count_expr.span,
                              ~"expected signed or unsigned integer for \
                                repeat count but found boolean");
            return 0;
        }
      },
      Err(*) => {
        tcx.sess.span_err(count_expr.span,
                          ~"expected constant integer for repeat count \
                            but found variable");
        return 0;
      }
    }
}

// Determine what purity to check a nested function under
pub fn determine_inherited_purity(parent_purity: ast::purity,
                                       child_purity: ast::purity,
                                       child_sigil: ast::Sigil)
                                    -> ast::purity {
    // If the closure is a stack closure and hasn't had some non-standard
    // purity inferred for it, then check it under its parent's purity.
    // Otherwise, use its own
    match child_sigil {
        ast::BorrowedSigil if child_purity == ast::impure_fn => parent_purity,
        _ => child_purity
    }
}

// Iterate over a type parameter's bounded traits and any supertraits
// of those traits, ignoring kinds.
// Here, the supertraits are the transitive closure of the supertrait
// relation on the supertraits from each bounded trait's constraint
// list.
pub fn iter_bound_traits_and_supertraits(tcx: ctxt,
                                         bounds: param_bounds,
                                         f: &fn(t) -> bool) {
    let mut fin = false;

    for bounds.each |bound| {

        let bound_trait_ty = match *bound {
            ty::bound_trait(bound_t) => bound_t,

            ty::bound_copy | ty::bound_owned |
            ty::bound_const | ty::bound_durable => {
                loop; // skip non-trait bounds
            }
        };

        let mut supertrait_map = LinearMap::new();
        let mut seen_def_ids = ~[];
        let mut i = 0;
        let trait_ty_id = ty_to_def_id(bound_trait_ty).expect(
            ~"iter_trait_ty_supertraits got a non-trait type");
        let mut trait_ty = bound_trait_ty;

        debug!("iter_bound_traits_and_supertraits: trait_ty = %s",
               ty_to_str(tcx, trait_ty));

        // Add the given trait ty to the hash map
        supertrait_map.insert(trait_ty_id, trait_ty);
        seen_def_ids.push(trait_ty_id);

        if f(trait_ty) {
            // Add all the supertraits to the hash map,
            // executing <f> on each of them
            while i < supertrait_map.len() && !fin {
                let init_trait_id = seen_def_ids[i];
                i += 1;
                 // Add supertraits to supertrait_map
                let supertraits = trait_supertraits(tcx, init_trait_id);
                for supertraits.each |supertrait| {
                    let super_t = supertrait.tpt.ty;
                    let d_id = ty_to_def_id(super_t).expect("supertrait \
                        should be a trait ty");
                    if !supertrait_map.contains_key(&d_id) {
                        supertrait_map.insert(d_id, super_t);
                        trait_ty = super_t;
                        seen_def_ids.push(d_id);
                    }
                    debug!("A super_t = %s", ty_to_str(tcx, trait_ty));
                    if !f(trait_ty) {
                        fin = true;
                    }
                }
            }
        };
        fin = false;
    }
}

pub fn count_traits_and_supertraits(tcx: ctxt,
                                    boundses: &[param_bounds]) -> uint {
    let mut total = 0;
    for boundses.each |bounds| {
        for iter_bound_traits_and_supertraits(tcx, *bounds) |_trait_ty| {
            total += 1;
        }
    }
    return total;
}

// Given a trait and a type, returns the impl of that type
pub fn get_impl_id(tcx: ctxt, trait_id: def_id, self_ty: t) -> def_id {
    match tcx.trait_impls.find(&trait_id) {
        Some(ty_to_impl) => match ty_to_impl.find(&self_ty) {
            Some(the_impl) => the_impl.did,
            None => // try autoderef!
                match deref(tcx, self_ty, false) {
                    Some(some_ty) => get_impl_id(tcx, trait_id, some_ty.ty),
                    None => tcx.sess.bug(~"get_impl_id: no impl of trait for \
                                           this type")
            }
        },
        None => tcx.sess.bug(~"get_impl_id: trait isn't in trait_impls")
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
