// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
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
use middle::resolve::{Impl, MethodInfo};
use middle::resolve;
use middle::ty;
use middle::subst::Subst;
use middle::typeck;
use middle;
use util::ppaux::{note_and_explain_region, bound_region_to_str};
use util::ppaux::{trait_store_to_str, ty_to_str, vstore_to_str};
use util::ppaux::{Repr, UserString};
use util::common::{indenter};
use util::enum_set::{EnumSet, CLike};

use core::ptr::to_unsafe_ptr;
use core::to_bytes;
use core::hashmap::{HashMap, HashSet};
use extra::smallintmap::SmallIntMap;
use syntax::ast::*;
use syntax::ast_util::is_local;
use syntax::ast_util;
use syntax::attr;
use syntax::codemap::span;
use syntax::codemap;
use syntax::parse::token::special_idents;
use syntax::{ast, ast_map};
use syntax::opt_vec::OptVec;
use syntax::opt_vec;
use syntax::abi::AbiSet;
use syntax;

// Data types

#[deriving(Eq)]
pub struct field {
    ident: ast::ident,
    mt: mt
}

pub struct Method {
    ident: ast::ident,
    generics: ty::Generics,
    transformed_self_ty: Option<ty::t>,
    fty: BareFnTy,
    explicit_self: ast::explicit_self_,
    vis: ast::visibility,
    def_id: ast::def_id
}

pub impl Method {
    fn new(ident: ast::ident,
           generics: ty::Generics,
           transformed_self_ty: Option<ty::t>,
           fty: BareFnTy,
           explicit_self: ast::explicit_self_,
           vis: ast::visibility,
           def_id: ast::def_id) -> Method {
        // Check the invariants.
        if explicit_self == ast::sty_static {
            assert!(transformed_self_ty.is_none());
        } else {
            assert!(transformed_self_ty.is_some());
        }

       Method {
            ident: ident,
            generics: generics,
            transformed_self_ty: transformed_self_ty,
            fty: fty,
            explicit_self: explicit_self,
            vis: vis,
            def_id: def_id
        }
    }
}

#[deriving(Eq)]
pub struct mt {
    ty: t,
    mutbl: ast::mutability,
}

#[deriving(Eq, Encodable, Decodable)]
pub enum vstore {
    vstore_fixed(uint),
    vstore_uniq,
    vstore_box,
    vstore_slice(Region)
}

#[deriving(Eq, IterBytes, Encodable, Decodable)]
pub enum TraitStore {
    BoxTraitStore,              // @Trait
    UniqTraitStore,             // ~Trait
    RegionTraitStore(Region),   // &Trait
}

// XXX: This should probably go away at some point. Maybe after destructors
// do?
#[deriving(Eq, Encodable, Decodable)]
pub enum SelfMode {
    ByCopy,
    ByRef,
}

pub struct field_ty {
    ident: ident,
    id: def_id,
    vis: ast::visibility,
}

// Contains information needed to resolve types and (in the future) look up
// the types of AST nodes.
#[deriving(Eq)]
pub struct creader_cache_key {
    cnum: int,
    pos: uint,
    len: uint
}

type creader_cache = @mut HashMap<creader_cache_key, t>;

impl to_bytes::IterBytes for creader_cache_key {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        self.cnum.iter_bytes(lsb0, f) &&
        self.pos.iter_bytes(lsb0, f) &&
        self.len.iter_bytes(lsb0, f)
    }
}

struct intern_key {
    sty: *sty,
}

// NB: Do not replace this with #[deriving(Eq)]. The automatically-derived
// implementation will not recurse through sty and you will get stack
// exhaustion.
impl cmp::Eq for intern_key {
    fn eq(&self, other: &intern_key) -> bool {
        unsafe {
            *self.sty == *other.sty
        }
    }
    fn ne(&self, other: &intern_key) -> bool {
        !self.eq(other)
    }
}

impl to_bytes::IterBytes for intern_key {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        unsafe {
            (*self.sty).iter_bytes(lsb0, f)
        }
    }
}

pub enum ast_ty_to_ty_cache_entry {
    atttce_unresolved,  /* not resolved yet */
    atttce_resolved(t)  /* resolved to a type, irrespective of region */
}

pub type opt_region_variance = Option<region_variance>;

#[deriving(Eq, Decodable, Encodable)]
pub enum region_variance { rv_covariant, rv_invariant, rv_contravariant }

#[deriving(Decodable, Encodable)]
pub enum AutoAdjustment {
    AutoAddEnv(ty::Region, ast::Sigil),
    AutoDerefRef(AutoDerefRef)
}

#[deriving(Decodable, Encodable)]
pub struct AutoDerefRef {
    autoderefs: uint,
    autoref: Option<AutoRef>
}

#[deriving(Decodable, Encodable)]
pub enum AutoRef {
    /// Convert from T to &T
    AutoPtr(Region, ast::mutability),

    /// Convert from @[]/~[]/&[] to &[] (or str)
    AutoBorrowVec(Region, ast::mutability),

    /// Convert from @[]/~[]/&[] to &&[] (or str)
    AutoBorrowVecRef(Region, ast::mutability),

    /// Convert from @fn()/~fn()/&fn() to &fn()
    AutoBorrowFn(Region),

    /// Convert from T to *T
    AutoUnsafe(ast::mutability)
}

// Stores information about provided methods (a.k.a. default methods) in
// implementations.
//
// This is a map from ID of each implementation to the method info and trait
// method ID of each of the default methods belonging to the trait that that
// implementation implements.
pub type ProvidedMethodsMap = @mut HashMap<def_id,@mut ~[@ProvidedMethodInfo]>;

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

pub type ctxt = @ctxt_;

struct ctxt_ {
    diag: @syntax::diagnostic::span_handler,
    interner: @mut HashMap<intern_key, ~t_box_>,
    next_id: @mut uint,
    cstore: @mut metadata::cstore::CStore,
    sess: session::Session,
    def_map: resolve::DefMap,

    region_maps: @mut middle::region::RegionMaps,
    region_paramd_items: middle::region::region_paramd_items,

    // Stores the types for various nodes in the AST.  Note that this table
    // is not guaranteed to be populated until after typeck.  See
    // typeck::check::fn_ctxt for details.
    node_types: node_type_table,

    // Stores the type parameters which were substituted to obtain the type
    // of this node.  This only applies to nodes that refer to entities
    // parameterized by type parameters, such as generic fns, types, or
    // other items.
    node_type_substs: @mut HashMap<node_id, ~[t]>,

    // Maps from a method to the method "descriptor"
    methods: @mut HashMap<def_id, @Method>,

    // Maps from a trait def-id to a list of the def-ids of its methods
    trait_method_def_ids: @mut HashMap<def_id, @~[def_id]>,

    // A cache for the trait_methods() routine
    trait_methods_cache: @mut HashMap<def_id, @~[@Method]>,

    trait_refs: @mut HashMap<node_id, @TraitRef>,
    trait_defs: @mut HashMap<def_id, @TraitDef>,

    items: ast_map::map,
    intrinsic_defs: @mut HashMap<ast::ident, (ast::def_id, t)>,
    intrinsic_traits: @mut HashMap<ast::ident, @TraitRef>,
    freevars: freevars::freevar_map,
    tcache: type_cache,
    rcache: creader_cache,
    ccache: constness_cache,
    short_names_cache: @mut HashMap<t, @~str>,
    needs_unwind_cleanup_cache: @mut HashMap<t, bool>,
    tc_cache: @mut HashMap<uint, TypeContents>,
    ast_ty_to_ty_cache: @mut HashMap<node_id, ast_ty_to_ty_cache_entry>,
    enum_var_cache: @mut HashMap<def_id, @~[VariantInfo]>,
    ty_param_defs: @mut HashMap<ast::node_id, TypeParameterDef>,
    adjustments: @mut HashMap<ast::node_id, @AutoAdjustment>,
    normalized_cache: @mut HashMap<t, t>,
    lang_items: middle::lang_items::LanguageItems,
    // A mapping from an implementation ID to the method info and trait
    // method ID of the provided (a.k.a. default) methods in the traits that
    // that implementation implements.
    provided_methods: ProvidedMethodsMap,
    provided_method_sources: @mut HashMap<ast::def_id, ProvidedMethodSource>,
    supertraits: @mut HashMap<ast::def_id, @~[@TraitRef]>,

    // A mapping from the def ID of an enum or struct type to the def ID
    // of the method that implements its destructor. If the type is not
    // present in this map, it does not have a destructor. This map is
    // populated during the coherence phase of typechecking.
    destructor_for_type: @mut HashMap<ast::def_id, ast::def_id>,

    // A method will be in this list if and only if it is a destructor.
    destructors: @mut HashSet<ast::def_id>,

    // Maps a trait onto a mapping from self-ty to impl
    trait_impls: @mut HashMap<ast::def_id, @mut HashMap<t, @Impl>>,

    // Set of used unsafe nodes (functions or blocks). Unsafe nodes not
    // present in this set can be warned about.
    used_unsafe: @mut HashSet<ast::node_id>,

    // Set of nodes which mark locals as mutable which end up getting used at
    // some point. Local variable definitions not in this set can be warned
    // about.
    used_mut_nodes: @mut HashSet<ast::node_id>,
}

pub enum tbox_flag {
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

pub type t_box = &'static t_box_;

pub struct t_box_ {
    sty: sty,
    id: uint,
    flags: uint,
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
        let t2: t_box = cast::transmute(t);
        t2
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
pub fn type_id(t: t) -> uint { get(t).id }

#[deriving(Eq)]
pub struct BareFnTy {
    purity: ast::purity,
    abis: AbiSet,
    sig: FnSig
}

#[deriving(Eq)]
pub struct ClosureTy {
    purity: ast::purity,
    sigil: ast::Sigil,
    onceness: ast::Onceness,
    region: Region,
    bounds: BuiltinBounds,
    sig: FnSig,
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
    inputs: ~[t],
    output: t
}

impl to_bytes::IterBytes for BareFnTy {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        self.purity.iter_bytes(lsb0, f) &&
        self.abis.iter_bytes(lsb0, f) &&
        self.sig.iter_bytes(lsb0, f)
    }
}

impl to_bytes::IterBytes for ClosureTy {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        self.purity.iter_bytes(lsb0, f) &&
        self.sigil.iter_bytes(lsb0, f) &&
        self.onceness.iter_bytes(lsb0, f) &&
        self.region.iter_bytes(lsb0, f) &&
        self.sig.iter_bytes(lsb0, f)
    }
}

#[deriving(Eq, IterBytes)]
pub struct param_ty {
    idx: uint,
    def_id: def_id
}

/// Representation of regions:
#[deriving(Eq, IterBytes, Encodable, Decodable)]
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
    re_free(FreeRegion),

    /// A concrete region naming some expression within the current function.
    re_scope(node_id),

    /// Static data that has an "infinite" lifetime. Top in the region lattice.
    re_static,

    /// A region variable.  Should not exist after typeck.
    re_infer(InferRegion),

    /// Empty lifetime is for data that is never accessed.
    /// Bottom in the region lattice. We treat re_empty somewhat
    /// specially; at least right now, we do not generate instances of
    /// it during the GLB computations, but rather
    /// generate an error instead. This is to improve error messages.
    /// The only way to get an instance of re_empty is to have a region
    /// variable with no constraints.
    re_empty,
}

pub impl Region {
    fn is_bound(&self) -> bool {
        match self {
            &re_bound(*) => true,
            _ => false
        }
    }
}

#[deriving(Eq, IterBytes, Encodable, Decodable)]
pub struct FreeRegion {
    scope_id: node_id,
    bound_region: bound_region
}

#[deriving(Eq, IterBytes, Encodable, Decodable)]
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

mod primitives {
    use super::t_box_;

    use syntax::ast;

    macro_rules! def_prim_ty(
        ($name:ident, $sty:expr, $id:expr) => (
            pub static $name: t_box_ = t_box_ {
                sty: $sty,
                id: $id,
                flags: 0,
            };
        )
    )

    def_prim_ty!(TY_NIL,    super::ty_nil,                  0)
    def_prim_ty!(TY_BOOL,   super::ty_bool,                 1)
    def_prim_ty!(TY_INT,    super::ty_int(ast::ty_i),       2)
    def_prim_ty!(TY_CHAR,   super::ty_int(ast::ty_char),    3)
    def_prim_ty!(TY_I8,     super::ty_int(ast::ty_i8),      4)
    def_prim_ty!(TY_I16,    super::ty_int(ast::ty_i16),     5)
    def_prim_ty!(TY_I32,    super::ty_int(ast::ty_i32),     6)
    def_prim_ty!(TY_I64,    super::ty_int(ast::ty_i64),     7)
    def_prim_ty!(TY_UINT,   super::ty_uint(ast::ty_u),      8)
    def_prim_ty!(TY_U8,     super::ty_uint(ast::ty_u8),     9)
    def_prim_ty!(TY_U16,    super::ty_uint(ast::ty_u16),    10)
    def_prim_ty!(TY_U32,    super::ty_uint(ast::ty_u32),    11)
    def_prim_ty!(TY_U64,    super::ty_uint(ast::ty_u64),    12)
    def_prim_ty!(TY_FLOAT,  super::ty_float(ast::ty_f),     13)
    def_prim_ty!(TY_F32,    super::ty_float(ast::ty_f32),   14)
    def_prim_ty!(TY_F64,    super::ty_float(ast::ty_f64),   15)

    pub static TY_BOT: t_box_ = t_box_ {
        sty: super::ty_bot,
        id: 16,
        flags: super::has_ty_bot as uint,
    };

    pub static TY_ERR: t_box_ = t_box_ {
        sty: super::ty_err,
        id: 17,
        flags: super::has_ty_err as uint,
    };

    pub static LAST_PRIMITIVE_ID: uint = 18;
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
    ty_trait(def_id, substs, TraitStore, ast::mutability),
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

#[deriving(Eq, IterBytes)]
pub struct TraitRef {
    def_id: def_id,
    substs: substs
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
    terr_abi_mismatch(expected_found<AbiSet>),
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
    terr_regions_does_not_outlive(Region, Region),
    terr_regions_not_same(Region, Region),
    terr_regions_no_overlap(Region, Region),
    terr_regions_insufficiently_polymorphic(bound_region, Region),
    terr_regions_overly_polymorphic(bound_region, Region),
    terr_vstores_differ(terr_vstore_kind, expected_found<vstore>),
    terr_trait_stores_differ(terr_vstore_kind, expected_found<TraitStore>),
    terr_in_field(@type_err, ast::ident),
    terr_sorts(expected_found<t>),
    terr_integer_as_char,
    terr_int_mismatch(expected_found<IntVarValue>),
    terr_float_mismatch(expected_found<ast::float_ty>),
    terr_traits(expected_found<ast::def_id>),
    terr_builtin_bounds(expected_found<BuiltinBounds>),
}

#[deriving(Eq, IterBytes)]
pub struct ParamBounds {
    builtin_bounds: BuiltinBounds,
    trait_bounds: ~[@TraitRef]
}

pub type BuiltinBounds = EnumSet<BuiltinBound>;

#[deriving(Eq, IterBytes)]
pub enum BuiltinBound {
    BoundCopy,
    BoundStatic,
    BoundOwned,
    BoundConst,
}

pub fn EmptyBuiltinBounds() -> BuiltinBounds {
    EnumSet::empty()
}

pub fn AllBuiltinBounds() -> BuiltinBounds {
    let mut set = EnumSet::empty();
    set.add(BoundCopy);
    set.add(BoundStatic);
    set.add(BoundOwned);
    set.add(BoundConst);
    set
}

impl CLike for BuiltinBound {
    pub fn to_uint(&self) -> uint {
        *self as uint
    }
    pub fn from_uint(v: uint) -> BuiltinBound {
        unsafe { cast::transmute(v) }
    }
}

#[deriving(Eq)]
pub struct TyVid(uint);

#[deriving(Eq)]
pub struct IntVid(uint);

#[deriving(Eq)]
pub struct FloatVid(uint);

#[deriving(Eq, Encodable, Decodable)]
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
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        match *self {
            TyVar(ref tv) => {
                0u8.iter_bytes(lsb0, f) && tv.iter_bytes(lsb0, f)
            }
            IntVar(ref iv) => {
                1u8.iter_bytes(lsb0, f) && iv.iter_bytes(lsb0, f)
            }
            FloatVar(ref fv) => {
                2u8.iter_bytes(lsb0, f) && fv.iter_bytes(lsb0, f)
            }
        }
    }
}

#[deriving(Encodable, Decodable)]
pub enum InferRegion {
    ReVar(RegionVid),
    ReSkolemized(uint, bound_region)
}

impl to_bytes::IterBytes for InferRegion {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        match *self {
            ReVar(ref rv) => {
                0u8.iter_bytes(lsb0, f) && rv.iter_bytes(lsb0, f)
            }
            ReSkolemized(ref v, _) => {
                1u8.iter_bytes(lsb0, f) && v.iter_bytes(lsb0, f)
            }
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
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        self.to_uint().iter_bytes(lsb0, f)
    }
}

impl to_bytes::IterBytes for IntVid {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        self.to_uint().iter_bytes(lsb0, f)
    }
}

impl to_bytes::IterBytes for FloatVid {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        self.to_uint().iter_bytes(lsb0, f)
    }
}

impl to_bytes::IterBytes for RegionVid {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        self.to_uint().iter_bytes(lsb0, f)
    }
}

pub struct TypeParameterDef {
    def_id: ast::def_id,
    bounds: @ParamBounds
}

/// Information about the type/lifetime parametesr associated with an item.
/// Analogous to ast::Generics.
pub struct Generics {
    type_param_defs: @~[TypeParameterDef],
    region_param: Option<region_variance>,
}

pub impl Generics {
    fn has_type_params(&self) -> bool {
        !self.type_param_defs.is_empty()
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
    generics: Generics,
    ty: t
}

/// As `ty_param_bounds_and_ty` but for a trait ref.
pub struct TraitDef {
    generics: Generics,
    trait_ref: @ty::TraitRef,
}

pub struct ty_param_substs_and_ty {
    substs: ty::substs,
    ty: ty::t
}

type type_cache = @mut HashMap<ast::def_id, ty_param_bounds_and_ty>;

type constness_cache = @mut HashMap<ast::def_id, const_eval::constness>;

pub type node_type_table = @mut SmallIntMap<t>;

fn mk_rcache() -> creader_cache {
    return @mut HashMap::new();
}

pub fn new_ty_hash<V:Copy>() -> @mut HashMap<t, V> {
    @mut HashMap::new()
}

pub fn mk_ctxt(s: session::Session,
               dm: resolve::DefMap,
               amap: ast_map::map,
               freevars: freevars::freevar_map,
               region_maps: @mut middle::region::RegionMaps,
               region_paramd_items: middle::region::region_paramd_items,
               lang_items: middle::lang_items::LanguageItems)
            -> ctxt {
    @ctxt_ {
        diag: s.diagnostic(),
        interner: @mut HashMap::new(),
        next_id: @mut primitives::LAST_PRIMITIVE_ID,
        cstore: s.cstore,
        sess: s,
        def_map: dm,
        region_maps: region_maps,
        region_paramd_items: region_paramd_items,
        node_types: @mut SmallIntMap::new(),
        node_type_substs: @mut HashMap::new(),
        trait_refs: @mut HashMap::new(),
        trait_defs: @mut HashMap::new(),
        intrinsic_traits: @mut HashMap::new(),
        items: amap,
        intrinsic_defs: @mut HashMap::new(),
        freevars: freevars,
        tcache: @mut HashMap::new(),
        rcache: mk_rcache(),
        ccache: @mut HashMap::new(),
        short_names_cache: new_ty_hash(),
        needs_unwind_cleanup_cache: new_ty_hash(),
        tc_cache: @mut HashMap::new(),
        ast_ty_to_ty_cache: @mut HashMap::new(),
        enum_var_cache: @mut HashMap::new(),
        methods: @mut HashMap::new(),
        trait_method_def_ids: @mut HashMap::new(),
        trait_methods_cache: @mut HashMap::new(),
        ty_param_defs: @mut HashMap::new(),
        adjustments: @mut HashMap::new(),
        normalized_cache: new_ty_hash(),
        lang_items: lang_items,
        provided_methods: @mut HashMap::new(),
        provided_method_sources: @mut HashMap::new(),
        supertraits: @mut HashMap::new(),
        destructor_for_type: @mut HashMap::new(),
        destructors: @mut HashSet::new(),
        trait_impls: @mut HashMap::new(),
        used_unsafe: @mut HashSet::new(),
        used_mut_nodes: @mut HashSet::new(),
     }
}

// Type constructors

// Interns a type/name combination, stores the resulting box in cx.interner,
// and returns the box as cast to an unsafe ptr (see comments for t above).
fn mk_t(cx: ctxt, st: sty) -> t {
    // Check for primitive types.
    match st {
        ty_nil => return mk_nil(),
        ty_err => return mk_err(),
        ty_bool => return mk_bool(),
        ty_int(i) => return mk_mach_int(i),
        ty_uint(u) => return mk_mach_uint(u),
        ty_float(f) => return mk_mach_float(f),
        _ => {}
    };

    let key = intern_key { sty: to_unsafe_ptr(&st) };
    match cx.interner.find(&key) {
      Some(t) => unsafe { return cast::transmute(&t.sty); },
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
      &ty_trait(_, ref substs, _, _) => {
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
        for f.sig.inputs.each |a| { flags |= get(*a).flags; }
         flags |= get(f.sig.output).flags;
         // T -> _|_ is *not* _|_ !
         flags &= !(has_ty_bot as uint);
      }
      &ty_closure(ref f) => {
        flags |= rflags(f.region);
        for f.sig.inputs.each |a| { flags |= get(*a).flags; }
        flags |= get(f.sig.output).flags;
        // T -> _|_ is *not* _|_ !
        flags &= !(has_ty_bot as uint);
      }
    }

    let t = ~t_box_ {
        sty: st,
        id: *cx.next_id,
        flags: flags,
    };

    let sty_ptr = to_unsafe_ptr(&t.sty);

    let key = intern_key {
        sty: sty_ptr,
    };

    cx.interner.insert(key, t);

    *cx.next_id += 1;

    unsafe {
        cast::transmute::<*sty, t>(sty_ptr)
    }
}

#[inline(always)]
pub fn mk_prim_t(primitive: &'static t_box_) -> t {
    unsafe {
        cast::transmute::<&'static t_box_, t>(primitive)
    }
}

#[inline(always)]
pub fn mk_nil() -> t { mk_prim_t(&primitives::TY_NIL) }

#[inline(always)]
pub fn mk_err() -> t { mk_prim_t(&primitives::TY_ERR) }

#[inline(always)]
pub fn mk_bot() -> t { mk_prim_t(&primitives::TY_BOT) }

#[inline(always)]
pub fn mk_bool() -> t { mk_prim_t(&primitives::TY_BOOL) }

#[inline(always)]
pub fn mk_int() -> t { mk_prim_t(&primitives::TY_INT) }

#[inline(always)]
pub fn mk_i8() -> t { mk_prim_t(&primitives::TY_I8) }

#[inline(always)]
pub fn mk_i16() -> t { mk_prim_t(&primitives::TY_I16) }

#[inline(always)]
pub fn mk_i32() -> t { mk_prim_t(&primitives::TY_I32) }

#[inline(always)]
pub fn mk_i64() -> t { mk_prim_t(&primitives::TY_I64) }

#[inline(always)]
pub fn mk_float() -> t { mk_prim_t(&primitives::TY_FLOAT) }

#[inline(always)]
pub fn mk_f32() -> t { mk_prim_t(&primitives::TY_F32) }

#[inline(always)]
pub fn mk_f64() -> t { mk_prim_t(&primitives::TY_F64) }

#[inline(always)]
pub fn mk_uint() -> t { mk_prim_t(&primitives::TY_UINT) }

#[inline(always)]
pub fn mk_u8() -> t { mk_prim_t(&primitives::TY_U8) }

#[inline(always)]
pub fn mk_u16() -> t { mk_prim_t(&primitives::TY_U16) }

#[inline(always)]
pub fn mk_u32() -> t { mk_prim_t(&primitives::TY_U32) }

#[inline(always)]
pub fn mk_u64() -> t { mk_prim_t(&primitives::TY_U64) }

pub fn mk_mach_int(tm: ast::int_ty) -> t {
    match tm {
        ast::ty_i    => mk_int(),
        ast::ty_char => mk_char(),
        ast::ty_i8   => mk_i8(),
        ast::ty_i16  => mk_i16(),
        ast::ty_i32  => mk_i32(),
        ast::ty_i64  => mk_i64(),
    }
}

pub fn mk_mach_uint(tm: ast::uint_ty) -> t {
    match tm {
        ast::ty_u    => mk_uint(),
        ast::ty_u8   => mk_u8(),
        ast::ty_u16  => mk_u16(),
        ast::ty_u32  => mk_u32(),
        ast::ty_u64  => mk_u64(),
    }
}

pub fn mk_mach_float(tm: ast::float_ty) -> t {
    match tm {
        ast::ty_f    => mk_float(),
        ast::ty_f32  => mk_f32(),
        ast::ty_f64  => mk_f64(),
    }
}

#[inline(always)]
pub fn mk_char() -> t { mk_prim_t(&primitives::TY_CHAR) }

pub fn mk_estr(cx: ctxt, t: vstore) -> t {
    mk_t(cx, ty_estr(t))
}

pub fn mk_enum(cx: ctxt, did: ast::def_id, substs: substs) -> t {
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
    mk_ptr(cx, mt {ty: mk_nil(), mutbl: ast::m_imm})
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

pub fn mk_tup(cx: ctxt, ts: ~[t]) -> t { mk_t(cx, ty_tup(ts)) }

pub fn mk_closure(cx: ctxt, fty: ClosureTy) -> t {
    mk_t(cx, ty_closure(fty))
}

pub fn mk_bare_fn(cx: ctxt, fty: BareFnTy) -> t {
    mk_t(cx, ty_bare_fn(fty))
}

pub fn mk_ctor_fn(cx: ctxt, input_tys: &[ty::t], output: ty::t) -> t {
    let input_args = input_tys.map(|t| *t);
    mk_bare_fn(cx,
               BareFnTy {
                   purity: ast::pure_fn,
                   abis: AbiSet::Rust(),
                   sig: FnSig {
                    bound_lifetime_names: opt_vec::Empty,
                    inputs: input_args,
                    output: output
                   }
                })
}


pub fn mk_trait(cx: ctxt,
                did: ast::def_id,
                substs: substs,
                store: TraitStore,
                mutability: ast::mutability)
             -> t {
    // take a copy of substs so that we own the vectors inside
    mk_t(cx, ty_trait(did, substs, store, mutability))
}

pub fn mk_struct(cx: ctxt, struct_id: ast::def_id, substs: substs) -> t {
    // take a copy of substs so that we own the vectors inside
    mk_t(cx, ty_struct(struct_id, substs))
}

pub fn mk_var(cx: ctxt, v: TyVid) -> t { mk_infer(cx, TyVar(v)) }

pub fn mk_int_var(cx: ctxt, v: IntVid) -> t { mk_infer(cx, IntVar(v)) }

pub fn mk_float_var(cx: ctxt, v: FloatVid) -> t { mk_infer(cx, FloatVar(v)) }

pub fn mk_infer(cx: ctxt, it: InferTy) -> t { mk_t(cx, ty_infer(it)) }

pub fn mk_self(cx: ctxt, did: ast::def_id) -> t { mk_t(cx, ty_self(did)) }

pub fn mk_param(cx: ctxt, n: uint, k: def_id) -> t {
    mk_t(cx, ty_param(param_ty { idx: n, def_id: k }))
}

pub fn mk_type(cx: ctxt) -> t { mk_t(cx, ty_type) }

pub fn mk_opaque_closure_ptr(cx: ctxt, sigil: ast::Sigil) -> t {
    mk_t(cx, ty_opaque_closure_ptr(sigil))
}

pub fn mk_opaque_box(cx: ctxt) -> t { mk_t(cx, ty_opaque_box) }

pub fn walk_ty(ty: t, f: &fn(t)) {
    maybe_walk_ty(ty, |t| { f(t); true });
}

pub fn maybe_walk_ty(ty: t, f: &fn(t) -> bool) {
    if !f(ty) {
        return;
    }
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
      ty_trait(_, ref substs, _, _) => {
        for (*substs).tps.each |subty| { maybe_walk_ty(*subty, f); }
      }
      ty_tup(ref ts) => { for ts.each |tt| { maybe_walk_ty(*tt, f); } }
      ty_bare_fn(ref ft) => {
        for ft.sig.inputs.each |a| { maybe_walk_ty(*a, f); }
        maybe_walk_ty(ft.sig.output, f);
      }
      ty_closure(ref ft) => {
        for ft.sig.inputs.each |a| { maybe_walk_ty(*a, f); }
        maybe_walk_ty(ft.sig.output, f);
      }
    }
}

pub fn fold_sty_to_ty(tcx: ty::ctxt, sty: &sty, foldop: &fn(t) -> t) -> t {
    mk_t(tcx, fold_sty(sty, foldop))
}

pub fn fold_sig(sig: &FnSig, fldop: &fn(t) -> t) -> FnSig {
    let args = sig.inputs.map(|arg| fldop(*arg));

    FnSig {
        bound_lifetime_names: copy sig.bound_lifetime_names,
        inputs: args,
        output: fldop(sig.output)
    }
}

pub fn fold_bare_fn_ty(fty: &BareFnTy, fldop: &fn(t) -> t) -> BareFnTy {
    BareFnTy {sig: fold_sig(&fty.sig, fldop),
              abis: fty.abis,
              purity: fty.purity}
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
        ty_trait(did, ref substs, st, mutbl) => {
            ty_trait(did, fold_substs(substs, fldop), st, mutbl)
        }
        ty_tup(ref ts) => {
            let new_ts = ts.map(|tt| fldop(*tt));
            ty_tup(new_ts)
        }
        ty_bare_fn(ref f) => {
            ty_bare_fn(fold_bare_fn_ty(f, fldop))
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
            |t| { walk_regions_and_ty(cx, t, walkr, walkt); t },
            |t| { walk_regions_and_ty(cx, t, walkr, walkt); t });
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
      ty_trait(def_id, ref substs, st, mutbl) => {
        ty::mk_trait(cx, def_id, fold_substs(substs, fldr, fldt), st, mutbl)
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
                None => cx.sess.bug("ty_self unexpected here"),
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
    substs.repr(cx)
}

pub fn subst(cx: ctxt,
             substs: &substs,
             typ: t)
          -> t {
    typ.subst(cx, substs)
}

// Type utilities

pub fn type_is_nil(ty: t) -> bool { get(ty).sty == ty_nil }

pub fn type_is_bot(ty: t) -> bool {
    (get(ty).flags & (has_ty_bot as uint)) != 0
}

pub fn type_is_error(ty: t) -> bool {
    (get(ty).flags & (has_ty_err as uint)) != 0
}

pub fn type_needs_subst(ty: t) -> bool {
    tbox_has_flag(get(ty), needs_subst)
}

pub fn trait_ref_contains_error(tref: &ty::TraitRef) -> bool {
    tref.substs.self_ty.any(|&t| type_is_error(t)) ||
        tref.substs.tps.any(|&t| type_is_error(t))
}

pub fn type_is_ty_var(ty: t) -> bool {
    match get(ty).sty {
      ty_infer(TyVar(_)) => true,
      _ => false
    }
}

pub fn type_is_bool(ty: t) -> bool { get(ty).sty == ty_bool }

pub fn type_is_self(ty: t) -> bool {
    match get(ty).sty {
        ty_self(*) => true,
        _ => false
    }
}

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

pub fn type_is_simd(cx: ctxt, ty: t) -> bool {
    match get(ty).sty {
        ty_struct(did, _) => lookup_simd(cx, did),
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
      ty_estr(_) => return mk_mach_uint(ast::ty_u8),
      ty_evec(mt, _) | ty_unboxed_vec(mt) => return mt.ty,
      _ => cx.sess.bug("sequence_element_type called on non-sequence value"),
    }
}

pub fn simd_type(cx: ctxt, ty: t) -> t {
    match get(ty).sty {
        ty_struct(did, ref substs) => {
            let fields = lookup_struct_fields(cx, did);
            lookup_field_type(cx, did, fields[0].id, substs)
        }
        _ => fail!("simd_type called on invalid type")
    }
}

pub fn simd_size(cx: ctxt, ty: t) -> uint {
    match get(ty).sty {
        ty_struct(did, _) => {
            let fields = lookup_struct_fields(cx, did);
            fields.len()
        }
        _ => fail!("simd_size called on invalid type")
    }
}

pub fn get_element_type(ty: t, i: uint) -> t {
    match get(ty).sty {
      ty_tup(ref ts) => return ts[i],
      _ => fail!("get_element_type called on invalid type")
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

    let mut tycache = HashSet::new();
    let needs_unwind_cleanup =
        type_needs_unwind_cleanup_(cx, ty, &mut tycache, false);
    cx.needs_unwind_cleanup_cache.insert(ty, needs_unwind_cleanup);
    return needs_unwind_cleanup;
}

fn type_needs_unwind_cleanup_(cx: ctxt, ty: t,
                              tycache: &mut HashSet<t>,
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
            for (*enum_variants(cx, did)).each |v| {
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
    fn meets_bounds(&self, cx: ctxt, bbs: BuiltinBounds) -> bool {
        iter::all(|bb| self.meets_bound(cx, bb), |f| bbs.each(f))
    }

    fn meets_bound(&self, cx: ctxt, bb: BuiltinBound) -> bool {
        match bb {
            BoundCopy => self.is_copy(cx),
            BoundStatic => self.is_static(cx),
            BoundConst => self.is_const(cx),
            BoundOwned => self.is_owned(cx)
        }
    }

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

    fn is_static(&self, cx: ctxt) -> bool {
        !self.intersects(TypeContents::nonstatic(cx))
    }

    fn nonstatic(_cx: ctxt) -> TypeContents {
        TC_BORROWED_POINTER
    }

    fn is_owned(&self, cx: ctxt) -> bool {
        !self.intersects(TypeContents::nonowned(cx))
    }

    fn nonowned(_cx: ctxt) -> TypeContents {
        TC_MANAGED + TC_BORROWED_POINTER + TC_NON_OWNED
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
        TypeContents::noncopyable(cx) + TC_OWNED_POINTER + TC_OWNED_VEC
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
static TC_NONE: TypeContents =             TypeContents{bits: 0b0000_0000_0000};

/// Contains a borrowed value with a lifetime other than static
static TC_BORROWED_POINTER: TypeContents = TypeContents{bits: 0b0000_0000_0001};

/// Contains an owned pointer (~T) but not slice of some kind
static TC_OWNED_POINTER: TypeContents =    TypeContents{bits: 0b0000_0000_0010};

/// Contains an owned vector ~[] or owned string ~str
static TC_OWNED_VEC: TypeContents =        TypeContents{bits: 0b0000_0000_0100};

/// Contains a ~fn() or a ~Trait, which is non-copyable.
static TC_OWNED_CLOSURE: TypeContents =    TypeContents{bits: 0b0000_0000_1000};

/// Type with a destructor
static TC_DTOR: TypeContents =             TypeContents{bits: 0b0000_0001_0000};

/// Contains a managed value
static TC_MANAGED: TypeContents =          TypeContents{bits: 0b0000_0010_0000};

/// &mut with any region
static TC_BORROWED_MUT: TypeContents =     TypeContents{bits: 0b0000_0100_0000};

/// Mutable content, whether owned or by ref
static TC_MUTABLE: TypeContents =          TypeContents{bits: 0b0000_1000_0000};

/// One-shot closure
static TC_ONCE_CLOSURE: TypeContents =     TypeContents{bits: 0b0001_0000_0000};

/// An enum with no variants.
static TC_EMPTY_ENUM: TypeContents =       TypeContents{bits: 0b0010_0000_0000};

/// Contains a type marked with `#[non_owned]`
static TC_NON_OWNED: TypeContents =        TypeContents{bits: 0b0100_0000_0000};

/// All possible contents.
static TC_ALL: TypeContents =              TypeContents{bits: 0b0111_1111_1111};

pub fn type_is_copyable(cx: ctxt, t: ty::t) -> bool {
    type_contents(cx, t).is_copy(cx)
}

pub fn type_is_static(cx: ctxt, t: ty::t) -> bool {
    type_contents(cx, t).is_static(cx)
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

    let mut cache = HashMap::new();
    let result = tc_ty(cx, ty, &mut cache);
    cx.tc_cache.insert(ty_id, result);
    return result;

    fn tc_ty(cx: ctxt,
             ty: t,
             cache: &mut HashMap<uint, TypeContents>) -> TypeContents
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

        let result = match get(ty).sty {
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

            ty_trait(_, _, UniqTraitStore, _) => {
                TC_OWNED_CLOSURE
            }

            ty_trait(_, _, BoxTraitStore, mutbl) => {
                match mutbl {
                    ast::m_mutbl => TC_MANAGED + TC_MUTABLE,
                    _ => TC_MANAGED
                }
            }

            ty_trait(_, _, RegionTraitStore(r), mutbl) => {
                borrowed_contents(r, mutbl)
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
                let mut res = flds.foldl(
                    TC_NONE,
                    |tc, f| tc + tc_mt(cx, f.mt, cache));
                if ty::has_dtor(cx, did) {
                    res += TC_DTOR;
                }
                apply_tc_attr(cx, did, res)
            }

            ty_tup(ref tys) => {
                tys.foldl(TC_NONE, |tc, ty| *tc + tc_ty(cx, *ty, cache))
            }

            ty_enum(did, ref substs) => {
                let variants = substd_enum_variants(cx, did, substs);
                let res = if variants.is_empty() {
                    // we somewhat arbitrary declare that empty enums
                    // are non-copyable
                    TC_EMPTY_ENUM
                } else {
                    variants.foldl(TC_NONE, |tc, variant| {
                        variant.args.foldl(
                            *tc,
                            |tc, arg_ty| *tc + tc_ty(cx, *arg_ty, cache))
                    })
                };
                apply_tc_attr(cx, did, res)
            }

            ty_param(p) => {
                // We only ever ask for the kind of types that are defined in
                // the current crate; therefore, the only type parameters that
                // could be in scope are those defined in the current crate.
                // If this assertion failures, it is likely because of a
                // failure in the cross-crate inlining code to translate a
                // def-id.
                assert_eq!(p.def_id.crate, ast::local_crate);

                type_param_def_to_contents(
                    cx, cx.ty_param_defs.get(&p.def_id.node))
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
                cx.sess.bug("Asked to compute contents of fictitious type");
            }
        };

        cache.insert(ty_id, result);
        return result;
    }

    fn tc_mt(cx: ctxt,
             mt: mt,
             cache: &mut HashMap<uint, TypeContents>) -> TypeContents
    {
        let mc = if mt.mutbl == m_mutbl {TC_MUTABLE} else {TC_NONE};
        mc + tc_ty(cx, mt.ty, cache)
    }

    fn apply_tc_attr(cx: ctxt, did: def_id, mut tc: TypeContents) -> TypeContents {
        if has_attr(cx, did, "mutable") {
            tc += TC_MUTABLE;
        }
        if has_attr(cx, did, "non_owned") {
            tc += TC_NON_OWNED;
        }
        tc
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

    fn type_param_def_to_contents(cx: ctxt,
                                  type_param_def: &TypeParameterDef) -> TypeContents
    {
        debug!("type_param_def_to_contents(%s)", type_param_def.repr(cx));
        let _i = indenter();

        let mut tc = TC_ALL;
        for type_param_def.bounds.builtin_bounds.each |bound| {
            debug!("tc = %s, bound = %?", tc.to_str(), bound);
            tc = tc - match bound {
                BoundCopy => TypeContents::nonimplicitly_copyable(cx),
                BoundStatic => TypeContents::nonstatic(cx),
                BoundOwned => TypeContents::nonowned(cx),
                BoundConst => TypeContents::nonconst(cx),
            };
        }

        debug!("result = %s", tc.to_str());
        return tc;
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

          ty_trait(_, _, _, _) => {
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
        for (*enum_variants(cx, did)).each |variant| {
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

pub fn type_is_machine(ty: t) -> bool {
    match get(ty).sty {
        ty_int(ast::ty_i) | ty_uint(ast::ty_u) | ty_float(ast::ty_f) => false,
        ty_int(*) | ty_uint(*) | ty_float(*) => true,
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
      ty_trait(_, _, _, _) | ty_rptr(_,_) | ty_opaque_box => result = false,
      // Structural types
      ty_enum(did, ref substs) => {
        let variants = enum_variants(cx, did);
        for (*variants).each |variant| {
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
        cx.sess.bug("non concrete type in type_is_pod");
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
            if variants.len() == 0 {
                false
            } else {
                variants.all(|v| v.args.len() == 0)
            }
        }
        _ => false
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
        if (*variants).len() == 1u && variants[0].args.len() == 1u {
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
pub fn index(t: t) -> Option<mt> {
    index_sty(&get(t).sty)
}

pub fn index_sty(sty: &sty) -> Option<mt> {
    match *sty {
      ty_evec(mt, _) => Some(mt),
      ty_estr(_) => Some(mt {ty: mk_u8(), mutbl: ast::m_imm}),
      _ => None
    }
}

/**
 * Enforces an arbitrary but consistent total ordering over
 * free regions.  This is needed for establishing a consistent
 * LUB in region_inference. */
impl cmp::TotalOrd for FreeRegion {
    fn cmp(&self, other: &FreeRegion) -> Ordering {
        cmp::cmp2(&self.scope_id, &self.bound_region,
                  &other.scope_id, &other.bound_region)
    }
}

impl cmp::TotalEq for FreeRegion {
    fn equals(&self, other: &FreeRegion) -> bool {
        *self == *other
    }
}

/**
 * Enforces an arbitrary but consistent total ordering over
 * bound regions.  This is needed for establishing a consistent
 * LUB in region_inference. */
impl cmp::TotalOrd for bound_region {
    fn cmp(&self, other: &bound_region) -> Ordering {
        match (self, other) {
            (&ty::br_self, &ty::br_self) => cmp::Equal,
            (&ty::br_self, _) => cmp::Less,

            (&ty::br_anon(ref a1), &ty::br_anon(ref a2)) => a1.cmp(a2),
            (&ty::br_anon(*), _) => cmp::Less,

            (&ty::br_named(ref a1), &ty::br_named(ref a2)) => a1.repr.cmp(&a2.repr),
            (&ty::br_named(*), _) => cmp::Less,

            (&ty::br_cap_avoid(ref a1, @ref b1),
             &ty::br_cap_avoid(ref a2, @ref b2)) => cmp::cmp2(a1, b1, a2, b2),
            (&ty::br_cap_avoid(*), _) => cmp::Less,

            (&ty::br_fresh(ref a1), &ty::br_fresh(ref a2)) => a1.cmp(a2),
            (&ty::br_fresh(*), _) => cmp::Less,
        }
    }
}

impl cmp::TotalEq for bound_region {
    fn equals(&self, other: &bound_region) -> bool {
        *self == *other
    }
}

impl to_bytes::IterBytes for vstore {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        match *self {
            vstore_fixed(ref u) => {
                0u8.iter_bytes(lsb0, f) && u.iter_bytes(lsb0, f)
            }
            vstore_uniq => 1u8.iter_bytes(lsb0, f),
            vstore_box => 2u8.iter_bytes(lsb0, f),

            vstore_slice(ref r) => {
                3u8.iter_bytes(lsb0, f) && r.iter_bytes(lsb0, f)
            }
        }
    }
}

impl to_bytes::IterBytes for substs {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        self.self_r.iter_bytes(lsb0, f) &&
        self.self_ty.iter_bytes(lsb0, f) &&
        self.tps.iter_bytes(lsb0, f)
    }
}

impl to_bytes::IterBytes for mt {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        self.ty.iter_bytes(lsb0, f) && self.mutbl.iter_bytes(lsb0, f)
    }
}

impl to_bytes::IterBytes for field {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        self.ident.iter_bytes(lsb0, f) && self.mt.iter_bytes(lsb0, f)
    }
}

impl to_bytes::IterBytes for FnSig {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        self.inputs.iter_bytes(lsb0, f) && self.output.iter_bytes(lsb0, f)
    }
}

impl to_bytes::IterBytes for sty {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        match *self {
            ty_nil => 0u8.iter_bytes(lsb0, f),
            ty_bool => 1u8.iter_bytes(lsb0, f),

            ty_int(ref t) => 2u8.iter_bytes(lsb0, f) && t.iter_bytes(lsb0, f),

            ty_uint(ref t) => 3u8.iter_bytes(lsb0, f) && t.iter_bytes(lsb0, f),

            ty_float(ref t) => 4u8.iter_bytes(lsb0, f) && t.iter_bytes(lsb0, f),

            ty_estr(ref v) => 5u8.iter_bytes(lsb0, f) && v.iter_bytes(lsb0, f),

            ty_enum(ref did, ref substs) => {
                6u8.iter_bytes(lsb0, f) &&
                did.iter_bytes(lsb0, f) &&
                substs.iter_bytes(lsb0, f)
            }

            ty_box(ref mt) => 7u8.iter_bytes(lsb0, f) && mt.iter_bytes(lsb0, f),

            ty_evec(ref mt, ref v) => {
                8u8.iter_bytes(lsb0, f) &&
                mt.iter_bytes(lsb0, f) &&
                v.iter_bytes(lsb0, f)
            }

            ty_unboxed_vec(ref mt) => 9u8.iter_bytes(lsb0, f) && mt.iter_bytes(lsb0, f),

            ty_tup(ref ts) => 10u8.iter_bytes(lsb0, f) && ts.iter_bytes(lsb0, f),

            ty_bare_fn(ref ft) => 12u8.iter_bytes(lsb0, f) && ft.iter_bytes(lsb0, f),

            ty_self(ref did) => 13u8.iter_bytes(lsb0, f) && did.iter_bytes(lsb0, f),

            ty_infer(ref v) => 14u8.iter_bytes(lsb0, f) && v.iter_bytes(lsb0, f),

            ty_param(ref p) => 15u8.iter_bytes(lsb0, f) && p.iter_bytes(lsb0, f),

            ty_type => 16u8.iter_bytes(lsb0, f),
            ty_bot => 17u8.iter_bytes(lsb0, f),

            ty_ptr(ref mt) => 18u8.iter_bytes(lsb0, f) && mt.iter_bytes(lsb0, f),

            ty_uniq(ref mt) => 19u8.iter_bytes(lsb0, f) && mt.iter_bytes(lsb0, f),

            ty_trait(ref did, ref substs, ref v, ref mutbl) => {
                20u8.iter_bytes(lsb0, f) &&
                did.iter_bytes(lsb0, f) &&
                substs.iter_bytes(lsb0, f) &&
                v.iter_bytes(lsb0, f) &&
                mutbl.iter_bytes(lsb0, f)
            }

            ty_opaque_closure_ptr(ref ck) => 21u8.iter_bytes(lsb0, f) && ck.iter_bytes(lsb0, f),

            ty_opaque_box => 22u8.iter_bytes(lsb0, f),

            ty_struct(ref did, ref substs) => {
                23u8.iter_bytes(lsb0, f) && did.iter_bytes(lsb0, f) && substs.iter_bytes(lsb0, f)
            }

            ty_rptr(ref r, ref mt) => {
                24u8.iter_bytes(lsb0, f) && r.iter_bytes(lsb0, f) && mt.iter_bytes(lsb0, f)
            }

            ty_err => 25u8.iter_bytes(lsb0, f),

            ty_closure(ref ct) => 26u8.iter_bytes(lsb0, f) && ct.iter_bytes(lsb0, f),
        }
    }
}

pub fn node_id_to_trait_ref(cx: ctxt, id: ast::node_id) -> @ty::TraitRef {
    match cx.trait_refs.find(&id) {
       Some(&t) => t,
       None => cx.sess.bug(
           fmt!("node_id_to_trait_ref: no trait ref for node `%s`",
                ast_map::node_id_to_str(cx.items, id,
                                        cx.sess.parse_sess.interner)))
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

pub fn ty_fn_sig(fty: t) -> FnSig {
    match get(fty).sty {
        ty_bare_fn(ref f) => copy f.sig,
        ty_closure(ref f) => copy f.sig,
        ref s => {
            fail!("ty_fn_sig() called on non-fn type: %?", s)
        }
    }
}

// Type accessors for substructures of types
pub fn ty_fn_args(fty: t) -> ~[t] {
    match get(fty).sty {
        ty_bare_fn(ref f) => copy f.sig.inputs,
        ty_closure(ref f) => copy f.sig.inputs,
        ref s => {
            fail!("ty_fn_args() called on non-fn type: %?", s)
        }
    }
}

pub fn ty_closure_sigil(fty: t) -> Sigil {
    match get(fty).sty {
        ty_closure(ref f) => f.sigil,
        ref s => {
            fail!("ty_closure_sigil() called on non-closure type: %?", s)
        }
    }
}

pub fn ty_fn_purity(fty: t) -> ast::purity {
    match get(fty).sty {
        ty_bare_fn(ref f) => f.purity,
        ty_closure(ref f) => f.purity,
        ref s => {
            fail!("ty_fn_purity() called on non-fn type: %?", s)
        }
    }
}

pub fn ty_fn_ret(fty: t) -> t {
    match get(fty).sty {
        ty_bare_fn(ref f) => f.sig.output,
        ty_closure(ref f) => f.sig.output,
        ref s => {
            fail!("ty_fn_ret() called on non-fn type: %?", s)
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
        ref s => fail!("ty_vstore() called on invalid sty: %?", s)
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

pub fn replace_fn_sig(cx: ctxt, fsty: &sty, new_sig: FnSig) -> t {
    match *fsty {
        ty_bare_fn(ref f) => mk_bare_fn(cx, BareFnTy {sig: new_sig, ..*f}),
        ty_closure(ref f) => mk_closure(cx, ClosureTy {sig: new_sig, ..*f}),
        ref s => {
            cx.sess.bug(
                fmt!("ty_fn_sig() called on non-fn type: %?", s));
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
    vec::append_one(sig.inputs.map(|a| *a), sig.output)
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
    adjust_ty(cx, expr.span, unadjusted_ty, cx.adjustments.find_copy(&expr.id))
}

pub fn adjust_ty(cx: ctxt,
                 span: span,
                 unadjusted_ty: ty::t,
                 adjustment: Option<@AutoAdjustment>) -> ty::t
{
    /*! See `expr_ty_adjusted` */

    return match adjustment {
        None => unadjusted_ty,

        Some(@AutoAddEnv(r, s)) => {
            match ty::get(unadjusted_ty).sty {
                ty::ty_bare_fn(ref b) => {
                    ty::mk_closure(
                        cx,
                        ty::ClosureTy {purity: b.purity,
                                       sigil: s,
                                       onceness: ast::Many,
                                       region: r,
                                       bounds: ty::AllBuiltinBounds(),
                                       sig: copy b.sig})
                }
                ref b => {
                    cx.sess.bug(
                        fmt!("add_env adjustment on non-bare-fn: %?", b));
                }
            }
        }

        Some(@AutoDerefRef(ref adj)) => {
            let mut adjusted_ty = unadjusted_ty;

            for uint::range(0, adj.autoderefs) |i| {
                match ty::deref(cx, adjusted_ty, true) {
                    Some(mt) => { adjusted_ty = mt.ty; }
                    None => {
                        cx.sess.span_bug(
                            span,
                            fmt!("The %uth autoderef failed: %s",
                                 i, ty_to_str(cx,
                                              adjusted_ty)));
                    }
                }
            }

            match adj.autoref {
                None => adjusted_ty,
                Some(ref autoref) => {
                    match *autoref {
                        AutoPtr(r, m) => {
                            mk_rptr(cx, r, mt {ty: adjusted_ty, mutbl: m})
                        }

                        AutoBorrowVec(r, m) => {
                            borrow_vec(cx, span, r, m, adjusted_ty)
                        }

                        AutoBorrowVecRef(r, m) => {
                            adjusted_ty = borrow_vec(cx, span, r, m, adjusted_ty);
                            mk_rptr(cx, r, mt {ty: adjusted_ty, mutbl: ast::m_imm})
                        }

                        AutoBorrowFn(r) => {
                            borrow_fn(cx, span, r, adjusted_ty)
                        }

                        AutoUnsafe(m) => {
                            mk_ptr(cx, mt {ty: adjusted_ty, mutbl: m})
                        }
                    }
                }
            }
        }
    };

    fn borrow_vec(cx: ctxt, span: span,
                  r: Region, m: ast::mutability,
                  ty: ty::t) -> ty::t {
        match get(ty).sty {
            ty_evec(mt, _) => {
                ty::mk_evec(cx, mt {ty: mt.ty, mutbl: m}, vstore_slice(r))
            }

            ty_estr(_) => {
                ty::mk_estr(cx, vstore_slice(r))
            }

            ref s => {
                cx.sess.span_bug(
                    span,
                    fmt!("borrow-vec associated with bad sty: %?",
                         s));
            }
        }
    }

    fn borrow_fn(cx: ctxt, span: span, r: Region, ty: ty::t) -> ty::t {
        match get(ty).sty {
            ty_closure(ref fty) => {
                ty::mk_closure(cx, ClosureTy {
                    sigil: BorrowedSigil,
                    region: r,
                    ..copy *fty
                })
            }

            ref s => {
                cx.sess.span_bug(
                    span,
                    fmt!("borrow-fn associated with bad sty: %?",
                         s));
            }
        }
    }
}

pub impl AutoRef {
    fn map_region(&self, f: &fn(Region) -> Region) -> AutoRef {
        match *self {
            ty::AutoPtr(r, m) => ty::AutoPtr(f(r), m),
            ty::AutoBorrowVec(r, m) => ty::AutoBorrowVec(f(r), m),
            ty::AutoBorrowVecRef(r, m) => ty::AutoBorrowVecRef(f(r), m),
            ty::AutoBorrowFn(r) => ty::AutoBorrowFn(f(r)),
            ty::AutoUnsafe(m) => ty::AutoUnsafe(m),
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

pub fn method_call_type_param_defs(
    tcx: ctxt,
    method_map: typeck::method_map,
    id: ast::node_id) -> Option<@~[TypeParameterDef]>
{
    do method_map.find(&id).map |method| {
        match method.origin {
          typeck::method_static(did) => {
            // n.b.: When we encode impl methods, the bounds
            // that we encode include both the impl bounds
            // and then the method bounds themselves...
            ty::lookup_item_type(tcx, did).generics.type_param_defs
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
            let trait_type_param_defs =
                ty::lookup_trait_def(tcx, trt_id).generics.type_param_defs;
            @vec::append(
                copy *trait_type_param_defs,
                *ty::trait_method(tcx, trt_id, n_mth).generics.type_param_defs)
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
        ast::expr_path(*) | ast::expr_self => {
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
                "macro expression remains after expansion");
        }
    }
}

pub fn stmt_node_id(s: @ast::stmt) -> ast::node_id {
    match s.node {
      ast::stmt_decl(_, id) | stmt_expr(_, id) | stmt_semi(_, id) => {
        return id;
      }
      ast::stmt_mac(*) => fail!("unexpanded macro in trans")
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

pub fn method_idx(id: ast::ident, meths: &[@Method]) -> Option<uint> {
    vec::position(meths, |m| m.ident == id)
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
      ty_trait(id, _, _, _) => fmt!("trait %s", item_path_str(cx, id)),
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
        terr_traits(values) => {
            fmt!("expected trait %s but found trait %s",
                 item_path_str(cx, values.expected),
                 item_path_str(cx, values.found))
        }
        terr_builtin_bounds(values) => {
            if values.expected.is_empty() {
                fmt!("expected no bounds but found `%s`",
                     values.found.user_string(cx))
            } else if values.found.is_empty() {
                fmt!("expected bounds `%s` but found no bounds",
                     values.expected.user_string(cx))
            } else {
                fmt!("expected bounds `%s` but found bounds `%s`",
                     values.expected.user_string(cx),
                     values.found.user_string(cx))
            }
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
            note_and_explain_region(cx, "", subregion, "...");
            note_and_explain_region(cx, "...does not necessarily outlive ",
                                    superregion, "");
        }
        terr_regions_not_same(region1, region2) => {
            note_and_explain_region(cx, "", region1, "...");
            note_and_explain_region(cx, "...is not the same lifetime as ",
                                    region2, "");
        }
        terr_regions_no_overlap(region1, region2) => {
            note_and_explain_region(cx, "", region1, "...");
            note_and_explain_region(cx, "...does not overlap ",
                                    region2, "");
        }
        terr_regions_insufficiently_polymorphic(_, conc_region) => {
            note_and_explain_region(cx,
                                    "concrete lifetime that was found is ",
                                    conc_region, "");
        }
        terr_regions_overly_polymorphic(_, conc_region) => {
            note_and_explain_region(cx,
                                    "expected concrete lifetime is ",
                                    conc_region, "");
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
                         id: ast::def_id) -> @~[@TraitRef]
{
    // Check the cache.
    match cx.supertraits.find(&id) {
        Some(&trait_refs) => { return trait_refs; }
        None => {}  // Continue.
    }

    // Not in the cache. It had better be in the metadata, which means it
    // shouldn't be local.
    assert!(!is_local(id));

    // Get the supertraits out of the metadata and create the
    // TraitRef for each.
    let result = @csearch::get_supertraits(cx, id);
    cx.supertraits.insert(id, result);
    return result;
}

pub fn trait_ref_supertraits(cx: ctxt, trait_ref: &ty::TraitRef) -> ~[@TraitRef] {
    let supertrait_refs = trait_supertraits(cx, trait_ref.def_id);
    supertrait_refs.map(
        |supertrait_ref| supertrait_ref.subst(cx, &trait_ref.substs))
}

fn lookup_locally_or_in_crate_store<V:Copy>(
    descr: &str,
    def_id: ast::def_id,
    map: &mut HashMap<ast::def_id, V>,
    load_external: &fn() -> V) -> V
{
    /*!
     *
     * Helper for looking things up in the various maps
     * that are populated during typeck::collect (e.g.,
     * `cx.methods`, `cx.tcache`, etc).  All of these share
     * the pattern that if the id is local, it should have
     * been loaded into the map by the `typeck::collect` phase.
     * If the def-id is external, then we have to go consult
     * the crate loading code (and cache the result for the future).
     */

    match map.find(&def_id) {
        Some(&v) => { return v; }
        None => { }
    }

    if def_id.crate == ast::local_crate {
        fail!("No def'n found for %? in tcx.%s", def_id, descr);
    }
    let v = load_external();
    map.insert(def_id, v);
    return v;
}

pub fn trait_method(cx: ctxt, trait_did: ast::def_id, idx: uint) -> @Method {
    let method_def_id = ty::trait_method_def_ids(cx, trait_did)[idx];
    ty::method(cx, method_def_id)
}

pub fn trait_methods(cx: ctxt, trait_did: ast::def_id) -> @~[@Method] {
    match cx.trait_methods_cache.find(&trait_did) {
        Some(&methods) => methods,
        None => {
            let def_ids = ty::trait_method_def_ids(cx, trait_did);
            let methods = @def_ids.map(|d| ty::method(cx, *d));
            cx.trait_methods_cache.insert(trait_did, methods);
            methods
        }
    }
}

pub fn method(cx: ctxt, id: ast::def_id) -> @Method {
    lookup_locally_or_in_crate_store(
        "methods", id, cx.methods,
        || @csearch::get_method(cx, id))
}

pub fn trait_method_def_ids(cx: ctxt, id: ast::def_id) -> @~[def_id] {
    lookup_locally_or_in_crate_store(
        "methods", id, cx.trait_method_def_ids,
        || @csearch::get_trait_method_def_ids(cx.cstore, id))
}

pub fn impl_trait_ref(cx: ctxt, id: ast::def_id) -> Option<@TraitRef> {
    if id.crate == ast::local_crate {
        debug!("(impl_trait_ref) searching for trait impl %?", id);
        match cx.items.find(&id.node) {
           Some(&ast_map::node_item(@ast::item {
                        node: ast::item_impl(_, opt_trait, _, _),
                        _},
                    _)) => {
               match opt_trait {
                   Some(t) => Some(ty::node_id_to_trait_ref(cx, t.ref_id)),
                   None => None
               }
           }
           _ => None
        }
    } else {
        csearch::get_impl_trait(cx, id)
    }
}

pub fn ty_to_def_id(ty: t) -> Option<ast::def_id> {
    match get(ty).sty {
      ty_trait(id, _, _, _) | ty_struct(id, _) | ty_enum(id, _) => Some(id),
      _ => None
    }
}

/// Returns the def ID of the constructor for the given tuple-like struct, or
/// None if the struct is not tuple-like. Fails if the given def ID does not
/// refer to a struct at all.
fn struct_ctor_id(cx: ctxt, struct_did: ast::def_id) -> Option<ast::def_id> {
    if struct_did.crate != ast::local_crate {
        // XXX: Cross-crate functionality.
        cx.sess.unimpl("constructor ID of cross-crate tuple structs");
    }

    match cx.items.find(&struct_did.node) {
        Some(&ast_map::node_item(item, _)) => {
            match item.node {
                ast::item_struct(struct_def, _) => {
                    struct_def.ctor_id.map(|ctor_id|
                        ast_util::local_def(*ctor_id))
                }
                _ => cx.sess.bug("called struct_ctor_id on non-struct")
            }
        }
        _ => cx.sess.bug("called struct_ctor_id on non-struct")
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
        Some(&method_def_id) => TraitDtor(method_def_id),
        None => NoDtor,
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
            vec::append_one(vec::to_owned(vec::init(*path)),
                            ast_map::path_name((*variant).node.name))
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
        match cx.items.get_copy(&id.node) {
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
                                ty_fn_args(ctor_ty).map(|a| *a)
                            } else {
                                ~[]
                            }
                        };
                        match variant.node.disr_expr {
                          Some (ex) => {
                            disr_val = match const_eval::eval_const_expr(cx,
                                                                         ex) {
                              const_eval::const_int(val) => val as int,
                              _ => cx.sess.bug("tag_variants: bad disr expr")
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
                        fail!("struct variant kinds unimpl in enum_variants")
                    }
                }
            })
          }
          _ => cx.sess.bug("tag_variants: id not bound to an enum")
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
    cx.sess.bug("enum_variant_with_id(): no variant exists with that ID");
}


// If the given item is in an external crate, looks up its type and adds it to
// the type cache. Returns the type parameters and type.
pub fn lookup_item_type(cx: ctxt,
                        did: ast::def_id)
                     -> ty_param_bounds_and_ty {
    lookup_locally_or_in_crate_store(
        "tcache", did, cx.tcache,
        || csearch::get_type(cx, did))
}

/// Given the did of a trait, returns its canonical trait ref.
pub fn lookup_trait_def(cx: ctxt, did: ast::def_id) -> @ty::TraitDef {
    match cx.trait_defs.find(&did) {
        Some(&trait_def) => {
            // The item is in this crate. The caller should have added it to the
            // type cache already
            return trait_def;
        }
        None => {
            assert!(did.crate != ast::local_crate);
            let trait_def = @csearch::get_trait_def(cx, did);
            cx.trait_defs.insert(did, trait_def);
            return trait_def;
        }
    }
}

/// Determine whether an item is annotated with an attribute
pub fn has_attr(tcx: ctxt, did: def_id, attr: &str) -> bool {
    if is_local(did) {
        match tcx.items.find(&did.node) {
            Some(
                &ast_map::node_item(@ast::item {
                    attrs: ref attrs,
                    _
                }, _)) => attr::attrs_contains_name(*attrs, attr),
            _ => tcx.sess.bug(fmt!("has_attr: %? is not an item",
                                   did))
        }
    } else {
        let mut ret = false;
        do csearch::get_item_attrs(tcx.cstore, did) |meta_items| {
            ret = attr::contains_name(meta_items, attr);
        }
        ret
    }
}

/// Determine whether an item is annotated with `#[packed]`
pub fn lookup_packed(tcx: ctxt, did: def_id) -> bool {
    has_attr(tcx, did, "packed")
}

/// Determine whether an item is annotated with `#[simd]`
pub fn lookup_simd(tcx: ctxt, did: def_id) -> bool {
    has_attr(tcx, did, "simd")
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
           Some(&ty_param_bounds_and_ty {ty, _}) => ty,
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
            _ => cx.sess.bug("struct ID bound to non-struct")
         }
       }
       Some(&ast_map::node_variant(ref variant, _, _)) => {
          match (*variant).node.kind {
            ast::struct_variant_kind(struct_def) => {
              struct_field_tys(struct_def.fields)
            }
            _ => {
              cx.sess.bug("struct ID bound to enum variant that isn't \
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
        None => cx.sess.bug("struct ID not found in parent's fields")
    }
}

fn struct_field_tys(fields: &[@struct_field]) -> ~[field_ty] {
    do fields.map |field| {
        match field.node.kind {
            named_field(ident, visibility) => {
                field_ty {
                    ident: ident,
                    id: ast_util::local_def(field.node.id),
                    vis: visibility,
                }
            }
            unnamed_field => {
                field_ty {
                    ident:
                        syntax::parse::token::special_idents::unnamed_field,
                    id: ast_util::local_def(field.node.id),
                    vis: ast::public,
                }
            }
        }
    }
}

// Returns a list of fields corresponding to the struct's items. trans uses
// this. Takes a list of substs with which to instantiate field types.
pub fn struct_fields(cx: ctxt, did: ast::def_id, substs: &substs)
                     -> ~[field] {
    do lookup_struct_fields(cx, did).map |f| {
       field {
            ident: f.ident,
            mt: mt {
                ty: lookup_field_type(cx, did, f.id, substs),
                mutbl: m_imm
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
                              "expected signed or unsigned integer for \
                               repeat count but found float");
            return count as uint;
        }
        const_eval::const_str(_) => {
            tcx.sess.span_err(count_expr.span,
                              "expected signed or unsigned integer for \
                               repeat count but found string");
            return 0;
        }
        const_eval::const_bool(_) => {
            tcx.sess.span_err(count_expr.span,
                              "expected signed or unsigned integer for \
                               repeat count but found boolean");
            return 0;
        }
      },
      Err(*) => {
        tcx.sess.span_err(count_expr.span,
                          "expected constant integer for repeat count \
                           but found variable");
        return 0;
      }
    }
}

// Determine what purity to check a nested function under
pub fn determine_inherited_purity(parent: (ast::purity, ast::node_id),
                                  child: (ast::purity, ast::node_id),
                                  child_sigil: ast::Sigil)
                                    -> (ast::purity, ast::node_id) {
    // If the closure is a stack closure and hasn't had some non-standard
    // purity inferred for it, then check it under its parent's purity.
    // Otherwise, use its own
    match child_sigil {
        ast::BorrowedSigil if child.first() == ast::impure_fn => parent,
        _ => child
    }
}

// Iterate over a type parameter's bounded traits and any supertraits
// of those traits, ignoring kinds.
// Here, the supertraits are the transitive closure of the supertrait
// relation on the supertraits from each bounded trait's constraint
// list.
pub fn each_bound_trait_and_supertraits(tcx: ctxt,
                                        bounds: &ParamBounds,
                                        f: &fn(@TraitRef) -> bool) -> bool {
    for bounds.trait_bounds.each |&bound_trait_ref| {
        let mut supertrait_set = HashMap::new();
        let mut trait_refs = ~[];
        let mut i = 0;

        // Seed the worklist with the trait from the bound
        supertrait_set.insert(bound_trait_ref.def_id, ());
        trait_refs.push(bound_trait_ref);

        // Add the given trait ty to the hash map
        while i < trait_refs.len() {
            debug!("each_bound_trait_and_supertraits(i=%?, trait_ref=%s)",
                   i, trait_refs[i].repr(tcx));

            if !f(trait_refs[i]) {
                return false;
            }

            // Add supertraits to supertrait_set
            let supertrait_refs = trait_ref_supertraits(tcx, trait_refs[i]);
            for supertrait_refs.each |&supertrait_ref| {
                debug!("each_bound_trait_and_supertraits(supertrait_ref=%s)",
                       supertrait_ref.repr(tcx));

                let d_id = supertrait_ref.def_id;
                if !supertrait_set.contains_key(&d_id) {
                    // FIXME(#5527) Could have same trait multiple times
                    supertrait_set.insert(d_id, ());
                    trait_refs.push(supertrait_ref);
                }
            }

            i += 1;
        }
    }
    return true;
}

pub fn count_traits_and_supertraits(tcx: ctxt,
                                    type_param_defs: &[TypeParameterDef]) -> uint {
    let mut total = 0;
    for type_param_defs.each |type_param_def| {
        for each_bound_trait_and_supertraits(tcx, type_param_def.bounds) |_| {
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
                    None => tcx.sess.bug("get_impl_id: no impl of trait for \
                                          this type")
            }
        },
        None => tcx.sess.bug("get_impl_id: trait isn't in trait_impls")
    }
}

pub fn visitor_object_ty(tcx: ctxt) -> (@TraitRef, t) {
    let ty_visitor_name = special_idents::ty_visitor;
    assert!(tcx.intrinsic_traits.contains_key(&ty_visitor_name));
    let trait_ref = tcx.intrinsic_traits.get_copy(&ty_visitor_name);
    (trait_ref,
     mk_trait(tcx, trait_ref.def_id, copy trait_ref.substs, BoxTraitStore, ast::m_imm))
}
