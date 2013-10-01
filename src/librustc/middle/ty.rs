// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use driver::session;
use metadata::csearch;
use metadata;
use middle::const_eval;
use middle::lang_items::{TyDescStructLangItem, TyVisitorTraitLangItem};
use middle::lang_items::OpaqueStructLangItem;
use middle::freevars;
use middle::resolve;
use middle::ty;
use middle::subst::Subst;
use middle::typeck;
use middle;
use util::ppaux::{note_and_explain_region, bound_region_ptr_to_str};
use util::ppaux::{trait_store_to_str, ty_to_str, vstore_to_str};
use util::ppaux::{Repr, UserString};
use util::common::{indenter};

use std::cast;
use std::cmp;
use std::hashmap::{HashMap, HashSet};
use std::ops;
use std::ptr::to_unsafe_ptr;
use std::to_bytes;
use std::to_str::ToStr;
use std::vec;
use syntax::ast::*;
use syntax::ast_util::is_local;
use syntax::ast_util;
use syntax::attr;
use syntax::codemap::Span;
use syntax::codemap;
use syntax::parse::token;
use syntax::{ast, ast_map};
use syntax::opt_vec::OptVec;
use syntax::opt_vec;
use syntax::abi::AbiSet;
use syntax;
use extra::enum_set::{EnumSet, CLike};

pub type Disr = u64;

pub static INITIAL_DISCRIMINANT_VALUE: Disr = 0;

// Data types

#[deriving(Eq, IterBytes)]
pub struct field {
    ident: ast::Ident,
    mt: mt
}

#[deriving(Clone)]
pub enum MethodContainer {
    TraitContainer(ast::DefId),
    ImplContainer(ast::DefId),
}

#[deriving(Clone)]
pub struct Method {
    ident: ast::Ident,
    generics: ty::Generics,
    transformed_self_ty: Option<ty::t>,
    fty: BareFnTy,
    explicit_self: ast::explicit_self_,
    vis: ast::visibility,
    def_id: ast::DefId,
    container: MethodContainer,

    // If this method is provided, we need to know where it came from
    provided_source: Option<ast::DefId>
}

impl Method {
    pub fn new(ident: ast::Ident,
               generics: ty::Generics,
               transformed_self_ty: Option<ty::t>,
               fty: BareFnTy,
               explicit_self: ast::explicit_self_,
               vis: ast::visibility,
               def_id: ast::DefId,
               container: MethodContainer,
               provided_source: Option<ast::DefId>)
               -> Method {
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
            def_id: def_id,
            container: container,
            provided_source: provided_source
        }
    }

    pub fn container_id(&self) -> ast::DefId {
        match self.container {
            TraitContainer(id) => id,
            ImplContainer(id) => id,
        }
    }
}

pub struct Impl {
    did: DefId,
    ident: Ident,
    methods: ~[@Method]
}

#[deriving(Clone, Eq, IterBytes)]
pub struct mt {
    ty: t,
    mutbl: ast::Mutability,
}

#[deriving(Clone, Eq, Encodable, Decodable, IterBytes, ToStr)]
pub enum vstore {
    vstore_fixed(uint),
    vstore_uniq,
    vstore_box,
    vstore_slice(Region)
}

#[deriving(Clone, Eq, IterBytes, Encodable, Decodable, ToStr)]
pub enum TraitStore {
    BoxTraitStore,              // @Trait
    UniqTraitStore,             // ~Trait
    RegionTraitStore(Region),   // &Trait
}

// XXX: This should probably go away at some point. Maybe after destructors
// do?
#[deriving(Clone, Eq, Encodable, Decodable)]
pub enum SelfMode {
    ByCopy,
    ByRef,
}

pub struct field_ty {
    name: Name,
    id: DefId,
    vis: ast::visibility,
}

// Contains information needed to resolve types and (in the future) look up
// the types of AST nodes.
#[deriving(Eq,IterBytes)]
pub struct creader_cache_key {
    cnum: int,
    pos: uint,
    len: uint
}

type creader_cache = @mut HashMap<creader_cache_key, t>;

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

// NB: Do not replace this with #[deriving(IterBytes)], as above. (Figured
// this out the hard way.)
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

#[deriving(Clone, Eq, Decodable, Encodable)]
pub enum region_variance {
    rv_covariant,
    rv_invariant,
    rv_contravariant,
}

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
    AutoPtr(Region, ast::Mutability),

    /// Convert from @[]/~[]/&[] to &[] (or str)
    AutoBorrowVec(Region, ast::Mutability),

    /// Convert from @[]/~[]/&[] to &&[] (or str)
    AutoBorrowVecRef(Region, ast::Mutability),

    /// Convert from @fn()/~fn()/&fn() to &fn()
    AutoBorrowFn(Region),

    /// Convert from T to *T
    AutoUnsafe(ast::Mutability),

    /// Convert from @Trait/~Trait/&Trait to &Trait
    AutoBorrowObj(Region, ast::Mutability),
}

pub type ctxt = @ctxt_;

struct ctxt_ {
    diag: @mut syntax::diagnostic::span_handler,
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
    node_type_substs: @mut HashMap<NodeId, ~[t]>,

    // Maps from a method to the method "descriptor"
    methods: @mut HashMap<DefId, @Method>,

    // Maps from a trait def-id to a list of the def-ids of its methods
    trait_method_def_ids: @mut HashMap<DefId, @~[DefId]>,

    // A cache for the trait_methods() routine
    trait_methods_cache: @mut HashMap<DefId, @~[@Method]>,

    impl_trait_cache: @mut HashMap<ast::DefId, Option<@ty::TraitRef>>,

    trait_refs: @mut HashMap<NodeId, @TraitRef>,
    trait_defs: @mut HashMap<DefId, @TraitDef>,

    items: ast_map::map,
    intrinsic_defs: @mut HashMap<ast::DefId, t>,
    freevars: freevars::freevar_map,
    tcache: type_cache,
    rcache: creader_cache,
    ccache: constness_cache,
    short_names_cache: @mut HashMap<t, @str>,
    needs_unwind_cleanup_cache: @mut HashMap<t, bool>,
    tc_cache: @mut HashMap<uint, TypeContents>,
    ast_ty_to_ty_cache: @mut HashMap<NodeId, ast_ty_to_ty_cache_entry>,
    enum_var_cache: @mut HashMap<DefId, @~[@VariantInfo]>,
    ty_param_defs: @mut HashMap<ast::NodeId, TypeParameterDef>,
    adjustments: @mut HashMap<ast::NodeId, @AutoAdjustment>,
    normalized_cache: @mut HashMap<t, t>,
    lang_items: middle::lang_items::LanguageItems,
    // A mapping of fake provided method def_ids to the default implementation
    provided_method_sources: @mut HashMap<ast::DefId, ast::DefId>,
    supertraits: @mut HashMap<ast::DefId, @~[@TraitRef]>,

    // A mapping from the def ID of an enum or struct type to the def ID
    // of the method that implements its destructor. If the type is not
    // present in this map, it does not have a destructor. This map is
    // populated during the coherence phase of typechecking.
    destructor_for_type: @mut HashMap<ast::DefId, ast::DefId>,

    // A method will be in this list if and only if it is a destructor.
    destructors: @mut HashSet<ast::DefId>,

    // Maps a trait onto a list of impls of that trait.
    trait_impls: @mut HashMap<ast::DefId, @mut ~[@Impl]>,

    // Maps a def_id of a type to a list of its inherent impls.
    // Contains implementations of methods that are inherent to a type.
    // Methods in these implementations don't need to be exported.
    inherent_impls: @mut HashMap<ast::DefId, @mut ~[@Impl]>,

    // Maps a def_id of an impl to an Impl structure.
    // Note that this contains all of the impls that we know about,
    // including ones in other crates. It's not clear that this is the best
    // way to do it.
    impls: @mut HashMap<ast::DefId, @Impl>,

    // Set of used unsafe nodes (functions or blocks). Unsafe nodes not
    // present in this set can be warned about.
    used_unsafe: @mut HashSet<ast::NodeId>,

    // Set of nodes which mark locals as mutable which end up getting used at
    // some point. Local variable definitions not in this set can be warned
    // about.
    used_mut_nodes: @mut HashSet<ast::NodeId>,

    // vtable resolution information for impl declarations
    impl_vtables: typeck::impl_vtable_map,

    // The set of external nominal types whose implementations have been read.
    // This is used for lazy resolution of methods.
    populated_external_types: @mut HashSet<ast::DefId>,

    // The set of external traits whose implementations have been read. This
    // is used for lazy resolution of traits.
    populated_external_traits: @mut HashSet<ast::DefId>,
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

impl ToStr for t {
    fn to_str(&self) -> ~str {
        ~"*t_opaque"
    }
}

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

#[deriving(Clone, Eq, IterBytes)]
pub struct BareFnTy {
    purity: ast::purity,
    abis: AbiSet,
    sig: FnSig
}

#[deriving(Clone, Eq, IterBytes)]
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
#[deriving(Clone, Eq, IterBytes)]
pub struct FnSig {
    bound_lifetime_names: OptVec<ast::Ident>,
    inputs: ~[t],
    output: t
}

#[deriving(Clone, Eq, IterBytes)]
pub struct param_ty {
    idx: uint,
    def_id: DefId
}

/// Representation of regions:
#[deriving(Clone, Eq, IterBytes, Encodable, Decodable, ToStr)]
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
    re_scope(NodeId),

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

impl Region {
    pub fn is_bound(&self) -> bool {
        match self {
            &re_bound(*) => true,
            _ => false
        }
    }
}

#[deriving(Clone, Eq, IterBytes, Encodable, Decodable, ToStr)]
pub struct FreeRegion {
    scope_id: NodeId,
    bound_region: bound_region
}

#[deriving(Clone, Eq, IterBytes, Encodable, Decodable, ToStr)]
pub enum bound_region {
    /// The self region for structs, impls (&T in a type defn or &'self T)
    br_self,

    /// An anonymous region parameter for a given fn (&T)
    br_anon(uint),

    /// Named region parameters for functions (a in &'a T)
    br_named(ast::Ident),

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
    br_cap_avoid(ast::NodeId, @bound_region),
}

/**
 * Represents the values to use when substituting lifetime parameters.
 * If the value is `ErasedRegions`, then this subst is occurring during
 * trans, and all region parameters will be replaced with `ty::re_static`. */
#[deriving(Clone, Eq, IterBytes)]
pub enum RegionSubsts {
    ErasedRegions,
    NonerasedRegions(OptVec<ty::Region>)
}

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
#[deriving(Clone, Eq, IterBytes)]
pub struct substs {
    self_ty: Option<ty::t>,
    tps: ~[t],
    regions: RegionSubsts,
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
    def_prim_ty!(TY_CHAR,   super::ty_char,                 2)
    def_prim_ty!(TY_INT,    super::ty_int(ast::ty_i),       3)
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
#[deriving(Clone, Eq, IterBytes)]
pub enum sty {
    ty_nil,
    ty_bot,
    ty_bool,
    ty_char,
    ty_int(ast::int_ty),
    ty_uint(ast::uint_ty),
    ty_float(ast::float_ty),
    ty_estr(vstore),
    ty_enum(DefId, substs),
    ty_box(mt),
    ty_uniq(mt),
    ty_evec(mt, vstore),
    ty_ptr(mt),
    ty_rptr(Region, mt),
    ty_bare_fn(BareFnTy),
    ty_closure(ClosureTy),
    ty_trait(DefId, substs, TraitStore, ast::Mutability, BuiltinBounds),
    ty_struct(DefId, substs),
    ty_tup(~[t]),

    ty_param(param_ty), // type parameter
    ty_self(DefId), /* special, implicit `self` type parameter;
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
    def_id: DefId,
    substs: substs
}

#[deriving(Clone, Eq)]
pub enum IntVarValue {
    IntType(ast::int_ty),
    UintType(ast::uint_ty),
}

#[deriving(Clone, ToStr)]
pub enum terr_vstore_kind {
    terr_vec,
    terr_str,
    terr_fn,
    terr_trait
}

#[deriving(Clone, ToStr)]
pub struct expected_found<T> {
    expected: T,
    found: T
}

// Data structures used in type unification
#[deriving(Clone, ToStr)]
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
    terr_record_fields(expected_found<Ident>),
    terr_arg_count,
    terr_regions_does_not_outlive(Region, Region),
    terr_regions_not_same(Region, Region),
    terr_regions_no_overlap(Region, Region),
    terr_regions_insufficiently_polymorphic(bound_region, Region),
    terr_regions_overly_polymorphic(bound_region, Region),
    terr_vstores_differ(terr_vstore_kind, expected_found<vstore>),
    terr_trait_stores_differ(terr_vstore_kind, expected_found<TraitStore>),
    terr_in_field(@type_err, ast::Ident),
    terr_sorts(expected_found<t>),
    terr_integer_as_char,
    terr_int_mismatch(expected_found<IntVarValue>),
    terr_float_mismatch(expected_found<ast::float_ty>),
    terr_traits(expected_found<ast::DefId>),
    terr_builtin_bounds(expected_found<BuiltinBounds>),
}

#[deriving(Eq, IterBytes)]
pub struct ParamBounds {
    builtin_bounds: BuiltinBounds,
    trait_bounds: ~[@TraitRef]
}

pub type BuiltinBounds = EnumSet<BuiltinBound>;

#[deriving(Clone, Eq, IterBytes, ToStr)]
pub enum BuiltinBound {
    BoundStatic,
    BoundSend,
    BoundFreeze,
    BoundSized,
}

pub fn EmptyBuiltinBounds() -> BuiltinBounds {
    EnumSet::empty()
}

pub fn AllBuiltinBounds() -> BuiltinBounds {
    let mut set = EnumSet::empty();
    set.add(BoundStatic);
    set.add(BoundSend);
    set.add(BoundFreeze);
    set.add(BoundSized);
    set
}

impl CLike for BuiltinBound {
    fn to_uint(&self) -> uint {
        *self as uint
    }
    fn from_uint(v: uint) -> BuiltinBound {
        unsafe { cast::transmute(v) }
    }
}

#[deriving(Clone, Eq, IterBytes)]
pub struct TyVid(uint);

#[deriving(Clone, Eq, IterBytes)]
pub struct IntVid(uint);

#[deriving(Clone, Eq, IterBytes)]
pub struct FloatVid(uint);

#[deriving(Clone, Eq, Encodable, Decodable, IterBytes)]
pub struct RegionVid {
    id: uint
}

#[deriving(Clone, Eq, IterBytes)]
pub enum InferTy {
    TyVar(TyVid),
    IntVar(IntVid),
    FloatVar(FloatVid)
}

#[deriving(Clone, Encodable, Decodable, IterBytes, ToStr)]
pub enum InferRegion {
    ReVar(RegionVid),
    ReSkolemized(uint, bound_region)
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
    fn to_str(&self) -> ~str { format!("<V{}>", self.to_uint()) }
}

impl Vid for IntVid {
    fn to_uint(&self) -> uint { **self }
}

impl ToStr for IntVid {
    fn to_str(&self) -> ~str { format!("<VI{}>", self.to_uint()) }
}

impl Vid for FloatVid {
    fn to_uint(&self) -> uint { **self }
}

impl ToStr for FloatVid {
    fn to_str(&self) -> ~str { format!("<VF{}>", self.to_uint()) }
}

impl Vid for RegionVid {
    fn to_uint(&self) -> uint { self.id }
}

impl ToStr for RegionVid {
    fn to_str(&self) -> ~str { format!("{:?}", self.id) }
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

#[deriving(Clone)]
pub struct TypeParameterDef {
    ident: ast::Ident,
    def_id: ast::DefId,
    bounds: @ParamBounds
}

/// Information about the type/lifetime parametesr associated with an item.
/// Analogous to ast::Generics.
#[deriving(Clone)]
pub struct Generics {
    type_param_defs: @~[TypeParameterDef],
    region_param: Option<region_variance>,
}

impl Generics {
    pub fn has_type_params(&self) -> bool {
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
#[deriving(Clone)]
pub struct ty_param_bounds_and_ty {
    generics: Generics,
    ty: t
}

/// As `ty_param_bounds_and_ty` but for a trait ref.
pub struct TraitDef {
    generics: Generics,
    bounds: BuiltinBounds,
    trait_ref: @ty::TraitRef,
}

pub struct ty_param_substs_and_ty {
    substs: ty::substs,
    ty: ty::t
}

type type_cache = @mut HashMap<ast::DefId, ty_param_bounds_and_ty>;

type constness_cache = @mut HashMap<ast::DefId, const_eval::constness>;

pub type node_type_table = @mut HashMap<uint,t>;

fn mk_rcache() -> creader_cache {
    return @mut HashMap::new();
}

pub fn new_ty_hash<V:'static>() -> @mut HashMap<t, V> {
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
        node_types: @mut HashMap::new(),
        node_type_substs: @mut HashMap::new(),
        trait_refs: @mut HashMap::new(),
        trait_defs: @mut HashMap::new(),
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
        impl_trait_cache: @mut HashMap::new(),
        ty_param_defs: @mut HashMap::new(),
        adjustments: @mut HashMap::new(),
        normalized_cache: new_ty_hash(),
        lang_items: lang_items,
        provided_method_sources: @mut HashMap::new(),
        supertraits: @mut HashMap::new(),
        destructor_for_type: @mut HashMap::new(),
        destructors: @mut HashSet::new(),
        trait_impls: @mut HashMap::new(),
        inherent_impls:  @mut HashMap::new(),
        impls:  @mut HashMap::new(),
        used_unsafe: @mut HashSet::new(),
        used_mut_nodes: @mut HashSet::new(),
        impl_vtables: @mut HashMap::new(),
        populated_external_types: @mut HashSet::new(),
        populated_external_traits: @mut HashSet::new(),
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
        for tt in substs.tps.iter() { f |= get(*tt).flags; }
        match substs.regions {
            ErasedRegions => {}
            NonerasedRegions(ref regions) => {
                for r in regions.iter() {
                    f |= rflags(*r)
                }
            }
        }
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
      &ty_nil | &ty_bool | &ty_char | &ty_int(_) | &ty_float(_) | &ty_uint(_) |
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
      &ty_trait(_, ref substs, _, _, _) => {
          flags |= sflags(substs);
          match st {
              ty_trait(_, _, RegionTraitStore(r), _, _) => {
                    flags |= rflags(r);
                }
              _ => {}
          }
      }
      &ty_box(ref m) | &ty_uniq(ref m) | &ty_evec(ref m, _) |
      &ty_ptr(ref m) | &ty_unboxed_vec(ref m) => {
        flags |= get(m.ty).flags;
      }
      &ty_rptr(r, ref m) => {
        flags |= rflags(r);
        flags |= get(m.ty).flags;
      }
      &ty_tup(ref ts) => for tt in ts.iter() { flags |= get(*tt).flags; },
      &ty_bare_fn(ref f) => {
        for a in f.sig.inputs.iter() { flags |= get(*a).flags; }
        flags |= get(f.sig.output).flags;
        // T -> _|_ is *not* _|_ !
        flags &= !(has_ty_bot as uint);
      }
      &ty_closure(ref f) => {
        flags |= rflags(f.region);
        for a in f.sig.inputs.iter() { flags |= get(*a).flags; }
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

#[inline]
pub fn mk_prim_t(primitive: &'static t_box_) -> t {
    unsafe {
        cast::transmute::<&'static t_box_, t>(primitive)
    }
}

#[inline]
pub fn mk_nil() -> t { mk_prim_t(&primitives::TY_NIL) }

#[inline]
pub fn mk_err() -> t { mk_prim_t(&primitives::TY_ERR) }

#[inline]
pub fn mk_bot() -> t { mk_prim_t(&primitives::TY_BOT) }

#[inline]
pub fn mk_bool() -> t { mk_prim_t(&primitives::TY_BOOL) }

#[inline]
pub fn mk_int() -> t { mk_prim_t(&primitives::TY_INT) }

#[inline]
pub fn mk_i8() -> t { mk_prim_t(&primitives::TY_I8) }

#[inline]
pub fn mk_i16() -> t { mk_prim_t(&primitives::TY_I16) }

#[inline]
pub fn mk_i32() -> t { mk_prim_t(&primitives::TY_I32) }

#[inline]
pub fn mk_i64() -> t { mk_prim_t(&primitives::TY_I64) }

#[inline]
pub fn mk_float() -> t { mk_prim_t(&primitives::TY_FLOAT) }

#[inline]
pub fn mk_f32() -> t { mk_prim_t(&primitives::TY_F32) }

#[inline]
pub fn mk_f64() -> t { mk_prim_t(&primitives::TY_F64) }

#[inline]
pub fn mk_uint() -> t { mk_prim_t(&primitives::TY_UINT) }

#[inline]
pub fn mk_u8() -> t { mk_prim_t(&primitives::TY_U8) }

#[inline]
pub fn mk_u16() -> t { mk_prim_t(&primitives::TY_U16) }

#[inline]
pub fn mk_u32() -> t { mk_prim_t(&primitives::TY_U32) }

#[inline]
pub fn mk_u64() -> t { mk_prim_t(&primitives::TY_U64) }

pub fn mk_mach_int(tm: ast::int_ty) -> t {
    match tm {
        ast::ty_i    => mk_int(),
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

#[inline]
pub fn mk_char() -> t { mk_prim_t(&primitives::TY_CHAR) }

pub fn mk_estr(cx: ctxt, t: vstore) -> t {
    mk_t(cx, ty_estr(t))
}

pub fn mk_enum(cx: ctxt, did: ast::DefId, substs: substs) -> t {
    // take a copy of substs so that we own the vectors inside
    mk_t(cx, ty_enum(did, substs))
}

pub fn mk_box(cx: ctxt, tm: mt) -> t { mk_t(cx, ty_box(tm)) }

pub fn mk_imm_box(cx: ctxt, ty: t) -> t {
    mk_box(cx, mt {ty: ty, mutbl: ast::MutImmutable})
}

pub fn mk_uniq(cx: ctxt, tm: mt) -> t { mk_t(cx, ty_uniq(tm)) }

pub fn mk_imm_uniq(cx: ctxt, ty: t) -> t {
    mk_uniq(cx, mt {ty: ty, mutbl: ast::MutImmutable})
}

pub fn mk_ptr(cx: ctxt, tm: mt) -> t { mk_t(cx, ty_ptr(tm)) }

pub fn mk_rptr(cx: ctxt, r: Region, tm: mt) -> t { mk_t(cx, ty_rptr(r, tm)) }

pub fn mk_mut_rptr(cx: ctxt, r: Region, ty: t) -> t {
    mk_rptr(cx, r, mt {ty: ty, mutbl: ast::MutMutable})
}
pub fn mk_imm_rptr(cx: ctxt, r: Region, ty: t) -> t {
    mk_rptr(cx, r, mt {ty: ty, mutbl: ast::MutImmutable})
}

pub fn mk_mut_ptr(cx: ctxt, ty: t) -> t {
    mk_ptr(cx, mt {ty: ty, mutbl: ast::MutMutable})
}

pub fn mk_imm_ptr(cx: ctxt, ty: t) -> t {
    mk_ptr(cx, mt {ty: ty, mutbl: ast::MutImmutable})
}

pub fn mk_nil_ptr(cx: ctxt) -> t {
    mk_ptr(cx, mt {ty: mk_nil(), mutbl: ast::MutImmutable})
}

pub fn mk_evec(cx: ctxt, tm: mt, t: vstore) -> t {
    mk_t(cx, ty_evec(tm, t))
}

pub fn mk_unboxed_vec(cx: ctxt, tm: mt) -> t {
    mk_t(cx, ty_unboxed_vec(tm))
}
pub fn mk_mut_unboxed_vec(cx: ctxt, ty: t) -> t {
    mk_t(cx, ty_unboxed_vec(mt {ty: ty, mutbl: ast::MutImmutable}))
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
                   purity: ast::impure_fn,
                   abis: AbiSet::Rust(),
                   sig: FnSig {
                    bound_lifetime_names: opt_vec::Empty,
                    inputs: input_args,
                    output: output
                   }
                })
}


pub fn mk_trait(cx: ctxt,
                did: ast::DefId,
                substs: substs,
                store: TraitStore,
                mutability: ast::Mutability,
                bounds: BuiltinBounds)
             -> t {
    // take a copy of substs so that we own the vectors inside
    mk_t(cx, ty_trait(did, substs, store, mutability, bounds))
}

pub fn mk_struct(cx: ctxt, struct_id: ast::DefId, substs: substs) -> t {
    // take a copy of substs so that we own the vectors inside
    mk_t(cx, ty_struct(struct_id, substs))
}

pub fn mk_var(cx: ctxt, v: TyVid) -> t { mk_infer(cx, TyVar(v)) }

pub fn mk_int_var(cx: ctxt, v: IntVid) -> t { mk_infer(cx, IntVar(v)) }

pub fn mk_float_var(cx: ctxt, v: FloatVid) -> t { mk_infer(cx, FloatVar(v)) }

pub fn mk_infer(cx: ctxt, it: InferTy) -> t { mk_t(cx, ty_infer(it)) }

pub fn mk_self(cx: ctxt, did: ast::DefId) -> t { mk_t(cx, ty_self(did)) }

pub fn mk_param(cx: ctxt, n: uint, k: DefId) -> t {
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
      ty_nil | ty_bot | ty_bool | ty_char | ty_int(_) | ty_uint(_) | ty_float(_) |
      ty_estr(_) | ty_type | ty_opaque_box | ty_self(_) |
      ty_opaque_closure_ptr(_) | ty_infer(_) | ty_param(_) | ty_err => {
      }
      ty_box(ref tm) | ty_evec(ref tm, _) | ty_unboxed_vec(ref tm) |
      ty_ptr(ref tm) | ty_rptr(_, ref tm) | ty_uniq(ref tm) => {
        maybe_walk_ty(tm.ty, f);
      }
      ty_enum(_, ref substs) | ty_struct(_, ref substs) |
      ty_trait(_, ref substs, _, _, _) => {
        for subty in (*substs).tps.iter() { maybe_walk_ty(*subty, |x| f(x)); }
      }
      ty_tup(ref ts) => { for tt in ts.iter() { maybe_walk_ty(*tt, |x| f(x)); } }
      ty_bare_fn(ref ft) => {
        for a in ft.sig.inputs.iter() { maybe_walk_ty(*a, |x| f(x)); }
        maybe_walk_ty(ft.sig.output, f);
      }
      ty_closure(ref ft) => {
        for a in ft.sig.inputs.iter() { maybe_walk_ty(*a, |x| f(x)); }
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
        bound_lifetime_names: sig.bound_lifetime_names.clone(),
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
        substs {regions: substs.regions.clone(),
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
        ty_trait(did, ref substs, st, mutbl, bounds) => {
            ty_trait(did, fold_substs(substs, fldop), st, mutbl, bounds)
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
            ty_closure(ClosureTy {
                sig: sig,
                purity: f.purity,
                sigil: f.sigil,
                onceness: f.onceness,
                region: f.region,
                bounds: f.bounds,
            })
        }
        ty_rptr(r, ref tm) => {
            ty_rptr(r, mt {ty: fldop(tm.ty), mutbl: tm.mutbl})
        }
        ty_struct(did, ref substs) => {
            ty_struct(did, fold_substs(substs, fldop))
        }
        ty_nil | ty_bot | ty_bool | ty_char | ty_int(_) | ty_uint(_) | ty_float(_) |
        ty_estr(_) | ty_type | ty_opaque_closure_ptr(_) | ty_err |
        ty_opaque_box | ty_infer(_) | ty_param(*) | ty_self(_) => {
            (*sty).clone()
        }
    }
}

// Folds types from the bottom up.
pub fn fold_ty(cx: ctxt, t0: t, fldop: &fn(t) -> t) -> t {
    let sty = fold_sty(&get(t0).sty, |t| fold_ty(cx, fldop(t), |t| fldop(t)));
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
            |t| { walk_regions_and_ty(cx, t, |r| walkr(r), |t| walkt(t)); t },
            |t| { walk_regions_and_ty(cx, t, |r| walkr(r), |t| walkt(t)); t });
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
        let regions = match substs.regions {
            ErasedRegions => ErasedRegions,
            NonerasedRegions(ref regions) => {
                NonerasedRegions(regions.map(|r| fldr(*r)))
            }
        };

        substs {
            regions: regions,
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
      ty_trait(def_id, ref substs, st, mutbl, bounds) => {
        let st = match st {
            RegionTraitStore(region) => RegionTraitStore(fldr(region)),
            st => st,
        };
        ty::mk_trait(cx, def_id, fold_substs(substs, fldr, fldt), st, mutbl, bounds)
      }
      ty_bare_fn(ref f) => {
          ty::mk_bare_fn(cx, BareFnTy {
            sig: fold_sig(&f.sig, fldfnt),
            purity: f.purity,
            abis: f.abis.clone(),
          })
      }
      ty_closure(ref f) => {
          ty::mk_closure(cx, ClosureTy {
            region: fldr(f.region),
            sig: fold_sig(&f.sig, fldfnt),
            purity: f.purity,
            sigil: f.sigil,
            onceness: f.onceness,
            bounds: f.bounds,
          })
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
        debug2!("do_fold(ty={}, in_fn={})", ty_to_str(cx, ty), in_fn);
        if !type_has_regions(ty) { return ty; }
        fold_regions_and_ty(
            cx, ty,
            |r| fldr(r, in_fn),
            |t| do_fold(cx, t, true,  |r,b| fldr(r,b)),
            |t| do_fold(cx, t, in_fn, |r,b| fldr(r,b)))
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
    let regions_is_noop = match substs.regions {
        ErasedRegions => false, // may be used to canonicalize
        NonerasedRegions(ref regions) => regions.is_empty()
    };

    substs.tps.len() == 0u &&
        regions_is_noop &&
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

pub fn type_is_voidish(ty: t) -> bool {
    //! "nil" and "bot" are void types in that they represent 0 bits of information
    type_is_nil(ty) || type_is_bot(ty)
}

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
    tref.substs.self_ty.iter().any(|&t| type_is_error(t)) ||
        tref.substs.tps.iter().any(|&t| type_is_error(t))
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
        _ => fail2!("simd_type called on invalid type")
    }
}

pub fn simd_size(cx: ctxt, ty: t) -> uint {
    match get(ty).sty {
        ty_struct(did, _) => {
            let fields = lookup_struct_fields(cx, did);
            fields.len()
        }
        _ => fail2!("simd_size called on invalid type")
    }
}

pub fn get_element_type(ty: t, i: uint) -> t {
    match get(ty).sty {
      ty_tup(ref ts) => return ts[i],
      _ => fail2!("get_element_type called on invalid type")
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
      ty_nil | ty_bool | ty_char | ty_int(_) | ty_float(_) | ty_uint(_) |
      ty_infer(IntVar(_)) | ty_infer(FloatVar(_)) | ty_type |
      ty_bare_fn(*) | ty_ptr(_) => true,
      _ => false
    }
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
            for v in (*enum_variants(cx, did)).iter() {
                for aty in v.args.iter() {
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

impl TypeContents {
    pub fn meets_bounds(&self, cx: ctxt, bbs: BuiltinBounds) -> bool {
        bbs.iter().all(|bb| self.meets_bound(cx, bb))
    }

    pub fn meets_bound(&self, cx: ctxt, bb: BuiltinBound) -> bool {
        match bb {
            BoundStatic => self.is_static(cx),
            BoundFreeze => self.is_freezable(cx),
            BoundSend => self.is_sendable(cx),
            BoundSized => self.is_sized(cx),
        }
    }

    pub fn intersects(&self, tc: TypeContents) -> bool {
        (self.bits & tc.bits) != 0
    }

    pub fn noncopyable(_cx: ctxt) -> TypeContents {
        TC_DTOR + TC_BORROWED_MUT + TC_ONCE_CLOSURE + TC_NONCOPY_TRAIT +
            TC_EMPTY_ENUM
    }

    pub fn is_static(&self, cx: ctxt) -> bool {
        !self.intersects(TypeContents::nonstatic(cx))
    }

    pub fn nonstatic(_cx: ctxt) -> TypeContents {
        TC_BORROWED_POINTER
    }

    pub fn is_sendable(&self, cx: ctxt) -> bool {
        !self.intersects(TypeContents::nonsendable(cx))
    }

    pub fn nonsendable(_cx: ctxt) -> TypeContents {
        TC_MANAGED + TC_BORROWED_POINTER + TC_NON_SENDABLE
    }

    pub fn contains_managed(&self) -> bool {
        self.intersects(TC_MANAGED)
    }

    pub fn is_freezable(&self, cx: ctxt) -> bool {
        !self.intersects(TypeContents::nonfreezable(cx))
    }

    pub fn nonfreezable(_cx: ctxt) -> TypeContents {
        TC_MUTABLE
    }

    pub fn is_sized(&self, cx: ctxt) -> bool {
        !self.intersects(TypeContents::dynamically_sized(cx))
    }

    pub fn dynamically_sized(_cx: ctxt) -> TypeContents {
        TC_DYNAMIC_SIZE
    }

    pub fn moves_by_default(&self, cx: ctxt) -> bool {
        self.intersects(TypeContents::nonimplicitly_copyable(cx))
    }

    pub fn nonimplicitly_copyable(cx: ctxt) -> TypeContents {
        TypeContents::noncopyable(cx) + TC_OWNED_POINTER + TC_OWNED_VEC
    }

    pub fn needs_drop(&self, cx: ctxt) -> bool {
        if self.intersects(TC_NONCOPY_TRAIT) {
            // Currently all noncopyable existentials are 2nd-class types
            // behind owned pointers. With dynamically-sized types, remove
            // this assertion.
            assert!(self.intersects(TC_OWNED_POINTER) ||
                    // (...or stack closures without a copy bound.)
                    self.intersects(TC_BORROWED_POINTER));
        }
        let tc = TC_MANAGED + TC_DTOR + TypeContents::sendable(cx);
        self.intersects(tc)
    }

    pub fn sendable(_cx: ctxt) -> TypeContents {
        //! Any kind of sendable contents.
        TC_OWNED_POINTER + TC_OWNED_VEC
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
        format!("TypeContents({})", self.bits.to_str_radix(2))
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

/// Contains a non-copyable ~fn() or a ~Trait (NOT a ~fn:Copy() or ~Trait:Copy).
static TC_NONCOPY_TRAIT: TypeContents =    TypeContents{bits: 0b0000_0000_1000};

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

/// Contains a type marked with `#[no_send]`
static TC_NON_SENDABLE: TypeContents =     TypeContents{bits: 0b0100_0000_0000};

/// Is a bare vector, str, function, trait, etc (only relevant at top level).
static TC_DYNAMIC_SIZE: TypeContents =     TypeContents{bits: 0b1000_0000_0000};

/// All possible contents.
static TC_ALL: TypeContents =              TypeContents{bits: 0b1111_1111_1111};

pub fn type_is_static(cx: ctxt, t: ty::t) -> bool {
    type_contents(cx, t).is_static(cx)
}

pub fn type_is_sendable(cx: ctxt, t: ty::t) -> bool {
    type_contents(cx, t).is_sendable(cx)
}

pub fn type_is_freezable(cx: ctxt, t: ty::t) -> bool {
    type_contents(cx, t).is_freezable(cx)
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
            // Scalar and unique types are sendable, freezable, and durable
            ty_nil | ty_bot | ty_bool | ty_char | ty_int(_) | ty_uint(_) | ty_float(_) |
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
                TC_MANAGED +
                    statically_sized(nonsendable(tc_mt(cx, mt, cache)))
            }

            ty_trait(_, _, store, mutbl, bounds) => {
                trait_contents(store, mutbl, bounds)
            }

            ty_rptr(r, mt) => {
                borrowed_contents(r, mt.mutbl) +
                    statically_sized(nonsendable(tc_mt(cx, mt, cache)))
            }

            ty_uniq(mt) => {
                TC_OWNED_POINTER + statically_sized(tc_mt(cx, mt, cache))
            }

            ty_evec(mt, vstore_uniq) => {
                TC_OWNED_VEC + statically_sized(tc_mt(cx, mt, cache))
            }

            ty_evec(mt, vstore_box) => {
                TC_MANAGED +
                    statically_sized(nonsendable(tc_mt(cx, mt, cache)))
            }

            ty_evec(mt, vstore_slice(r)) => {
                borrowed_contents(r, mt.mutbl) +
                    statically_sized(nonsendable(tc_mt(cx, mt, cache)))
            }

            ty_evec(mt, vstore_fixed(_)) => {
                let contents = tc_mt(cx, mt, cache);
                // FIXME(#6308) Uncomment this when construction of such
                // vectors is prevented earlier in compilation.
                // if !contents.is_sized(cx) {
                //     cx.sess.bug("Fixed-length vector of unsized type \
                //                  should be impossible");
                // }
                contents
            }

            ty_estr(vstore_box) => {
                TC_MANAGED
            }

            ty_estr(vstore_slice(r)) => {
                borrowed_contents(r, MutImmutable)
            }

            ty_estr(vstore_fixed(_)) => {
                TC_NONE
            }

            ty_struct(did, ref substs) => {
                let flds = struct_fields(cx, did, substs);
                let mut res = flds.iter().fold(
                    TC_NONE,
                    |tc, f| tc + tc_mt(cx, f.mt, cache));
                if ty::has_dtor(cx, did) {
                    res = res + TC_DTOR;
                }
                apply_tc_attr(cx, did, res)
            }

            ty_tup(ref tys) => {
                tys.iter().fold(TC_NONE, |tc, ty| tc + tc_ty(cx, *ty, cache))
            }

            ty_enum(did, ref substs) => {
                let variants = substd_enum_variants(cx, did, substs);
                let res = if variants.is_empty() {
                    // we somewhat arbitrary declare that empty enums
                    // are non-copyable
                    TC_EMPTY_ENUM
                } else {
                    variants.iter().fold(TC_NONE, |tc, variant| {
                        variant.args.iter().fold(tc,
                            |tc, arg_ty| tc + tc_ty(cx, *arg_ty, cache))
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
                assert_eq!(p.def_id.crate, ast::LOCAL_CRATE);

                let tp_def = cx.ty_param_defs.get(&p.def_id.node);
                kind_bounds_to_contents(cx, &tp_def.bounds.builtin_bounds,
                                        tp_def.bounds.trait_bounds)
            }

            ty_self(def_id) => {
                // FIXME(#4678)---self should just be a ty param

                // Self may be bounded if the associated trait has builtin kinds
                // for supertraits. If so we can use those bounds.
                let trait_def = lookup_trait_def(cx, def_id);
                let traits = [trait_def.trait_ref];
                kind_bounds_to_contents(cx, &trait_def.bounds, traits)
            }

            ty_infer(_) => {
                // This occurs during coherence, but shouldn't occur at other
                // times.
                TC_ALL
            }

            ty_opaque_box => TC_MANAGED,
            ty_unboxed_vec(mt) => TC_DYNAMIC_SIZE + tc_mt(cx, mt, cache),
            ty_opaque_closure_ptr(sigil) => {
                match sigil {
                    ast::BorrowedSigil => TC_BORROWED_POINTER,
                    ast::ManagedSigil => TC_MANAGED,
                    // FIXME(#3569): Looks like noncopyability should depend
                    // on the bounds, but I don't think this case ever comes up.
                    ast::OwnedSigil => TC_NONCOPY_TRAIT + TC_OWNED_POINTER,
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
        let mc = if mt.mutbl == MutMutable {TC_MUTABLE} else {TC_NONE};
        mc + tc_ty(cx, mt.ty, cache)
    }

    fn apply_tc_attr(cx: ctxt, did: DefId, mut tc: TypeContents) -> TypeContents {
        if has_attr(cx, did, "no_freeze") {
            tc = tc + TC_MUTABLE;
        }
        if has_attr(cx, did, "no_send") {
            tc = tc + TC_NON_SENDABLE;
        }
        tc
    }

    fn borrowed_contents(region: ty::Region,
                         mutbl: ast::Mutability) -> TypeContents
    {
        let mc = if mutbl == MutMutable {
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

    fn nonsendable(pointee: TypeContents) -> TypeContents {
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

    fn statically_sized(pointee: TypeContents) -> TypeContents {
        /*!
         * If a dynamically-sized type is found behind a pointer, we should
         * restore the 'Sized' kind to the pointer and things that contain it.
         */
        TypeContents {bits: pointee.bits & !TC_DYNAMIC_SIZE.bits}
    }

    fn closure_contents(cty: &ClosureTy) -> TypeContents {
        // Closure contents are just like trait contents, but with potentially
        // even more stuff.
        let st = match cty.sigil {
            ast::BorrowedSigil =>
                trait_contents(RegionTraitStore(cty.region), MutImmutable, cty.bounds)
                    + TC_BORROWED_POINTER, // might be an env packet even if static
            ast::ManagedSigil =>
                trait_contents(BoxTraitStore, MutImmutable, cty.bounds),
            ast::OwnedSigil =>
                trait_contents(UniqTraitStore, MutImmutable, cty.bounds),
        };
        // FIXME(#3569): This borrowed_contents call should be taken care of in
        // trait_contents, after ~Traits and @Traits can have region bounds too.
        // This one here is redundant for &fns but important for ~fns and @fns.
        let rt = borrowed_contents(cty.region, MutImmutable);
        // This also prohibits "@once fn" from being copied, which allows it to
        // be called. Neither way really makes much sense.
        let ot = match cty.onceness {
            ast::Once => TC_ONCE_CLOSURE,
            ast::Many => TC_NONE
        };
        // Prevent noncopyable types captured in the environment from being copied.
        st + rt + ot + TC_NONCOPY_TRAIT
    }

    fn trait_contents(store: TraitStore, mutbl: ast::Mutability,
                      bounds: BuiltinBounds) -> TypeContents {
        let st = match store {
            UniqTraitStore      => TC_OWNED_POINTER,
            BoxTraitStore       => TC_MANAGED,
            RegionTraitStore(r) => borrowed_contents(r, mutbl),
        };
        let mt = match mutbl { ast::MutMutable => TC_MUTABLE, _ => TC_NONE };
        // We get additional "special type contents" for each bound that *isn't*
        // on the trait. So iterate over the inverse of the bounds that are set.
        // This is like with typarams below, but less "pessimistic" and also
        // dependent on the trait store.
        let mut bt = TC_NONE;
        for bound in (AllBuiltinBounds() - bounds).iter() {
            bt = bt + match bound {
                BoundStatic if bounds.contains_elem(BoundSend)
                            => TC_NONE, // Send bound implies static bound.
                BoundStatic => TC_BORROWED_POINTER, // Useful for "@Trait:'static"
                BoundSend   => TC_NON_SENDABLE,
                BoundFreeze => TC_MUTABLE,
                BoundSized  => TC_NONE, // don't care if interior is sized
            };
        }
        st + mt + bt
    }

    fn kind_bounds_to_contents(cx: ctxt, bounds: &BuiltinBounds, traits: &[@TraitRef])
            -> TypeContents {
        let _i = indenter();

        let mut tc = TC_ALL;
        do each_inherited_builtin_bound(cx, bounds, traits) |bound| {
            debug2!("tc = {}, bound = {:?}", tc.to_str(), bound);
            tc = tc - match bound {
                BoundStatic => TypeContents::nonstatic(cx),
                BoundSend => TypeContents::nonsendable(cx),
                BoundFreeze => TypeContents::nonfreezable(cx),
                // The dynamic-size bit can be removed at pointer-level, etc.
                BoundSized => TypeContents::dynamically_sized(cx),
            };
        }

        debug2!("result = {}", tc.to_str());
        return tc;

        // Iterates over all builtin bounds on the type parameter def, including
        // those inherited from traits with builtin-kind-supertraits.
        fn each_inherited_builtin_bound(cx: ctxt, bounds: &BuiltinBounds,
                                        traits: &[@TraitRef], f: &fn(BuiltinBound)) {
            for bound in bounds.iter() {
                f(bound);
            }

            do each_bound_trait_and_supertraits(cx, traits) |trait_ref| {
                let trait_def = lookup_trait_def(cx, trait_ref.def_id);
                for bound in trait_def.bounds.iter() {
                    f(bound);
                }
                true
            };
        }
    }
}

pub fn type_moves_by_default(cx: ctxt, ty: t) -> bool {
    type_contents(cx, ty).moves_by_default(cx)
}

// True if instantiating an instance of `r_ty` requires an instance of `r_ty`.
pub fn is_instantiable(cx: ctxt, r_ty: t) -> bool {
    fn type_requires(cx: ctxt, seen: &mut ~[DefId],
                     r_ty: t, ty: t) -> bool {
        debug2!("type_requires({}, {})?",
               ::util::ppaux::ty_to_str(cx, r_ty),
               ::util::ppaux::ty_to_str(cx, ty));

        let r = {
            get(r_ty).sty == get(ty).sty ||
                subtypes_require(cx, seen, r_ty, ty)
        };

        debug2!("type_requires({}, {})? {}",
               ::util::ppaux::ty_to_str(cx, r_ty),
               ::util::ppaux::ty_to_str(cx, ty),
               r);
        return r;
    }

    fn subtypes_require(cx: ctxt, seen: &mut ~[DefId],
                        r_ty: t, ty: t) -> bool {
        debug2!("subtypes_require({}, {})?",
               ::util::ppaux::ty_to_str(cx, r_ty),
               ::util::ppaux::ty_to_str(cx, ty));

        let r = match get(ty).sty {
            ty_nil |
            ty_bot |
            ty_bool |
            ty_char |
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
                type_requires(cx, seen, r_ty, mt.ty)
            }

            ty_ptr(*) => {
                false           // unsafe ptrs can always be NULL
            }

            ty_trait(_, _, _, _, _) => {
                false
            }

            ty_struct(ref did, _) if seen.contains(did) => {
                false
            }

            ty_struct(did, ref substs) => {
                seen.push(did);
                let fields = struct_fields(cx, did, substs);
                let r = fields.iter().any(|f| type_requires(cx, seen, r_ty, f.mt.ty));
                seen.pop();
                r
            }

            ty_tup(ref ts) => {
                ts.iter().any(|t| type_requires(cx, seen, r_ty, *t))
            }

            ty_enum(ref did, _) if seen.contains(did) => {
                false
            }

            ty_enum(did, ref substs) => {
                seen.push(did);
                let vs = enum_variants(cx, did);
                let r = !vs.is_empty() && do vs.iter().all |variant| {
                    do variant.args.iter().any |aty| {
                        let sty = subst(cx, substs, *aty);
                        type_requires(cx, seen, r_ty, sty)
                    }
                };
                seen.pop();
                r
            }
        };

        debug2!("subtypes_require({}, {})? {}",
               ::util::ppaux::ty_to_str(cx, r_ty),
               ::util::ppaux::ty_to_str(cx, ty),
               r);

        return r;
    }

    let mut seen = ~[];
    !subtypes_require(cx, &mut seen, r_ty, r_ty)
}

pub fn type_structurally_contains(cx: ctxt,
                                  ty: t,
                                  test: &fn(x: &sty) -> bool)
                               -> bool {
    let sty = &get(ty).sty;
    debug2!("type_structurally_contains: {}",
           ::util::ppaux::ty_to_str(cx, ty));
    if test(sty) { return true; }
    match *sty {
      ty_enum(did, ref substs) => {
        for variant in (*enum_variants(cx, did)).iter() {
            for aty in variant.args.iter() {
                let sty = subst(cx, substs, *aty);
                if type_structurally_contains(cx, sty, |x| test(x)) { return true; }
            }
        }
        return false;
      }
      ty_struct(did, ref substs) => {
        let r = lookup_struct_fields(cx, did);
        for field in r.iter() {
            let ft = lookup_field_type(cx, did, field.id, substs);
            if type_structurally_contains(cx, ft, |x| test(x)) { return true; }
        }
        return false;
      }

      ty_tup(ref ts) => {
        for tt in ts.iter() {
            if type_structurally_contains(cx, *tt, |x| test(x)) { return true; }
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

pub fn type_is_trait(ty: t) -> bool {
    match get(ty).sty {
        ty_trait(*) => true,
        _ => false
    }
}

pub fn type_is_integral(ty: t) -> bool {
    match get(ty).sty {
      ty_infer(IntVar(_)) | ty_int(_) | ty_uint(_) => true,
      _ => false
    }
}

pub fn type_is_char(ty: t) -> bool {
    match get(ty).sty {
        ty_char => true,
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
      ty_nil | ty_bot | ty_bool | ty_char | ty_int(_) | ty_float(_) | ty_uint(_) |
      ty_type | ty_ptr(_) | ty_bare_fn(_) => result = true,
      // Boxed types
      ty_box(_) | ty_uniq(_) | ty_closure(_) |
      ty_estr(vstore_uniq) | ty_estr(vstore_box) |
      ty_evec(_, vstore_uniq) | ty_evec(_, vstore_box) |
      ty_trait(_, _, _, _, _) | ty_rptr(_,_) | ty_opaque_box => result = false,
      // Structural types
      ty_enum(did, ref substs) => {
        let variants = enum_variants(cx, did);
        for variant in (*variants).iter() {
            // XXX(pcwalton): This is an inefficient way to do this. Don't
            // synthesize a tuple!
            //
            // Perform any type parameter substitutions.
            let tup_ty = mk_tup(cx, variant.args.clone());
            let tup_ty = subst(cx, substs, tup_ty);
            if !type_is_pod(cx, tup_ty) { result = false; }
        }
      }
      ty_tup(ref elts) => {
        for elt in elts.iter() { if !type_is_pod(cx, *elt) { result = false; } }
      }
      ty_estr(vstore_fixed(_)) => result = true,
      ty_evec(ref mt, vstore_fixed(_)) | ty_unboxed_vec(ref mt) => {
        result = type_is_pod(cx, mt.ty);
      }
      ty_param(_) => result = false,
      ty_opaque_closure_ptr(_) => result = true,
      ty_struct(did, ref substs) => {
        let fields = lookup_struct_fields(cx, did);
        result = do fields.iter().all |f| {
            let fty = ty::lookup_item_type(cx, f.id);
            let sty = subst(cx, substs, fty.ty);
            type_is_pod(cx, sty)
        };
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

// Is the type's representation size known at compile time?
pub fn type_is_sized(cx: ctxt, ty: ty::t) -> bool {
    match get(ty).sty {
        // FIXME(#6308) add trait, vec, str, etc here.
        ty_param(p) => {
            let param_def = cx.ty_param_defs.get(&p.def_id.node);
            if param_def.bounds.builtin_bounds.contains_elem(BoundSized) {
                return true;
            }
            return false;
        },
        _ => return true,
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
                variants.iter().all(|v| v.args.len() == 0)
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
            Some(mt {ty: v_t, mutbl: ast::MutImmutable})
        } else {
            None
        }
      }

      ty_struct(did, ref substs) => {
        let fields = struct_fields(cx, did, substs);
        if fields.len() == 1 && fields[0].ident ==
                syntax::parse::token::special_idents::unnamed_field {
            Some(mt {ty: fields[0].mt.ty, mutbl: ast::MutImmutable})
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
      ty_estr(_) => Some(mt {ty: mk_u8(), mutbl: ast::MutImmutable}),
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

            (&ty::br_named(ref a1), &ty::br_named(ref a2)) => a1.name.cmp(&a2.name),
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

pub fn node_id_to_trait_ref(cx: ctxt, id: ast::NodeId) -> @ty::TraitRef {
    match cx.trait_refs.find(&id) {
       Some(&t) => t,
       None => cx.sess.bug(
           format!("node_id_to_trait_ref: no trait ref for node `{}`",
                ast_map::node_id_to_str(cx.items, id,
                                        token::get_ident_interner())))
    }
}

pub fn node_id_to_type(cx: ctxt, id: ast::NodeId) -> t {
    //printfln!("{:?}/{:?}", id, cx.node_types.len());
    match cx.node_types.find(&(id as uint)) {
       Some(&t) => t,
       None => cx.sess.bug(
           format!("node_id_to_type: no type for node `{}`",
                ast_map::node_id_to_str(cx.items, id,
                                        token::get_ident_interner())))
    }
}

// XXX(pcwalton): Makes a copy, bleh. Probably better to not do that.
pub fn node_id_to_type_params(cx: ctxt, id: ast::NodeId) -> ~[t] {
    match cx.node_type_substs.find(&id) {
      None => return ~[],
      Some(ts) => return (*ts).clone(),
    }
}

fn node_id_has_type_params(cx: ctxt, id: ast::NodeId) -> bool {
    cx.node_type_substs.contains_key(&id)
}

pub fn ty_fn_sig(fty: t) -> FnSig {
    match get(fty).sty {
        ty_bare_fn(ref f) => f.sig.clone(),
        ty_closure(ref f) => f.sig.clone(),
        ref s => {
            fail2!("ty_fn_sig() called on non-fn type: {:?}", s)
        }
    }
}

// Type accessors for substructures of types
pub fn ty_fn_args(fty: t) -> ~[t] {
    match get(fty).sty {
        ty_bare_fn(ref f) => f.sig.inputs.clone(),
        ty_closure(ref f) => f.sig.inputs.clone(),
        ref s => {
            fail2!("ty_fn_args() called on non-fn type: {:?}", s)
        }
    }
}

pub fn ty_closure_sigil(fty: t) -> Sigil {
    match get(fty).sty {
        ty_closure(ref f) => f.sigil,
        ref s => {
            fail2!("ty_closure_sigil() called on non-closure type: {:?}", s)
        }
    }
}

pub fn ty_fn_purity(fty: t) -> ast::purity {
    match get(fty).sty {
        ty_bare_fn(ref f) => f.purity,
        ty_closure(ref f) => f.purity,
        ref s => {
            fail2!("ty_fn_purity() called on non-fn type: {:?}", s)
        }
    }
}

pub fn ty_fn_ret(fty: t) -> t {
    match get(fty).sty {
        ty_bare_fn(ref f) => f.sig.output,
        ty_closure(ref f) => f.sig.output,
        ref s => {
            fail2!("ty_fn_ret() called on non-fn type: {:?}", s)
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
        ref s => fail2!("ty_vstore() called on invalid sty: {:?}", s)
    }
}

pub fn ty_region(tcx: ctxt,
                 span: Span,
                 ty: t) -> Region {
    match get(ty).sty {
        ty_rptr(r, _) => r,
        ty_evec(_, vstore_slice(r)) => r,
        ty_estr(vstore_slice(r)) => r,
        ref s => {
            tcx.sess.span_bug(
                span,
                format!("ty_region() invoked on in appropriate ty: {:?}", s));
        }
    }
}

pub fn replace_fn_sig(cx: ctxt, fsty: &sty, new_sig: FnSig) -> t {
    match *fsty {
        ty_bare_fn(ref f) => mk_bare_fn(cx, BareFnTy {sig: new_sig, ..*f}),
        ty_closure(ref f) => mk_closure(cx, ClosureTy {sig: new_sig, ..*f}),
        ref s => {
            cx.sess.bug(
                format!("ty_fn_sig() called on non-fn type: {:?}", s));
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
                sig: FnSig {output: ret_type, ..fty.sig.clone()},
                ..(*fty).clone()
            })
        }
        _ => {
            tcx.sess.bug(format!(
                "replace_fn_ret() invoked with non-fn-type: {}",
                ty_to_str(tcx, fn_type)));
        }
    }
}

// Returns a vec of all the input and output types of fty.
pub fn tys_in_fn_sig(sig: &FnSig) -> ~[t] {
    vec::append_one(sig.inputs.map(|a| *a), sig.output)
}

// Type accessors for AST nodes
pub fn block_ty(cx: ctxt, b: &ast::Block) -> t {
    return node_id_to_type(cx, b.id);
}


// Returns the type of a pattern as a monotype. Like @expr_ty, this function
// doesn't provide type parameter substitutions.
pub fn pat_ty(cx: ctxt, pat: &ast::Pat) -> t {
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
pub fn expr_ty(cx: ctxt, expr: &ast::Expr) -> t {
    return node_id_to_type(cx, expr.id);
}

pub fn expr_ty_adjusted(cx: ctxt, expr: &ast::Expr) -> t {
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
                 span: Span,
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
                                       sig: b.sig.clone()})
                }
                ref b => {
                    cx.sess.bug(
                        format!("add_env adjustment on non-bare-fn: {:?}", b));
                }
            }
        }

        Some(@AutoDerefRef(ref adj)) => {
            let mut adjusted_ty = unadjusted_ty;

            if (!ty::type_is_error(adjusted_ty)) {
                for i in range(0, adj.autoderefs) {
                    match ty::deref(cx, adjusted_ty, true) {
                        Some(mt) => { adjusted_ty = mt.ty; }
                        None => {
                            cx.sess.span_bug(
                                span,
                                format!("The {}th autoderef failed: {}",
                                     i, ty_to_str(cx,
                                                  adjusted_ty)));
                        }
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
                            mk_rptr(cx, r, mt {ty: adjusted_ty, mutbl: ast::MutImmutable})
                        }

                        AutoBorrowFn(r) => {
                            borrow_fn(cx, span, r, adjusted_ty)
                        }

                        AutoUnsafe(m) => {
                            mk_ptr(cx, mt {ty: adjusted_ty, mutbl: m})
                        }

                        AutoBorrowObj(r, m) => {
                            borrow_obj(cx, span, r, m, adjusted_ty)
                        }
                    }
                }
            }
        }
    };

    fn borrow_vec(cx: ctxt, span: Span,
                  r: Region, m: ast::Mutability,
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
                    format!("borrow-vec associated with bad sty: {:?}",
                         s));
            }
        }
    }

    fn borrow_fn(cx: ctxt, span: Span, r: Region, ty: ty::t) -> ty::t {
        match get(ty).sty {
            ty_closure(ref fty) => {
                ty::mk_closure(cx, ClosureTy {
                    sigil: BorrowedSigil,
                    region: r,
                    ..(*fty).clone()
                })
            }

            ref s => {
                cx.sess.span_bug(
                    span,
                    format!("borrow-fn associated with bad sty: {:?}",
                         s));
            }
        }
    }

    fn borrow_obj(cx: ctxt, span: Span, r: Region,
                  m: ast::Mutability, ty: ty::t) -> ty::t {
        match get(ty).sty {
            ty_trait(trt_did, ref trt_substs, _, _, b) => {
                ty::mk_trait(cx, trt_did, trt_substs.clone(),
                             RegionTraitStore(r), m, b)
            }
            ref s => {
                cx.sess.span_bug(
                    span,
                    format!("borrow-trait-obj associated with bad sty: {:?}",
                         s));
            }
        }
    }
}

impl AutoRef {
    pub fn map_region(&self, f: &fn(Region) -> Region) -> AutoRef {
        match *self {
            ty::AutoPtr(r, m) => ty::AutoPtr(f(r), m),
            ty::AutoBorrowVec(r, m) => ty::AutoBorrowVec(f(r), m),
            ty::AutoBorrowVecRef(r, m) => ty::AutoBorrowVecRef(f(r), m),
            ty::AutoBorrowFn(r) => ty::AutoBorrowFn(f(r)),
            ty::AutoUnsafe(m) => ty::AutoUnsafe(m),
            ty::AutoBorrowObj(r, m) => ty::AutoBorrowObj(f(r), m),
        }
    }
}

pub struct ParamsTy {
    params: ~[t],
    ty: t
}

pub fn expr_ty_params_and_ty(cx: ctxt,
                             expr: &ast::Expr)
                          -> ParamsTy {
    ParamsTy {
        params: node_id_to_type_params(cx, expr.id),
        ty: node_id_to_type(cx, expr.id)
    }
}

pub fn expr_has_ty_params(cx: ctxt, expr: &ast::Expr) -> bool {
    return node_id_has_type_params(cx, expr.id);
}

pub fn method_call_type_param_defs(tcx: ctxt,
                                   method_map: typeck::method_map,
                                   id: ast::NodeId)
                                   -> Option<@~[TypeParameterDef]> {
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
          typeck::method_object(typeck::method_object {
              trait_id: trt_id,
              method_num: n_mth, _}) => {
            // ...trait methods bounds, in contrast, include only the
            // method bounds, so we must preprend the tps from the
            // trait itself.  This ought to be harmonized.
            let trait_type_param_defs =
                lookup_trait_def(tcx, trt_id).generics.type_param_defs;
            @vec::append(
                (*trait_type_param_defs).clone(),
                *ty::trait_method(tcx,
                                  trt_id,
                                  n_mth).generics.type_param_defs)
          }
        }
    }
}

pub fn resolve_expr(tcx: ctxt, expr: &ast::Expr) -> ast::Def {
    match tcx.def_map.find(&expr.id) {
        Some(&def) => def,
        None => {
            tcx.sess.span_bug(expr.span, format!(
                "No def-map entry for expr {:?}", expr.id));
        }
    }
}

pub fn expr_is_lval(tcx: ctxt,
                    method_map: typeck::method_map,
                    e: &ast::Expr) -> bool {
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
                 expr: &ast::Expr) -> ExprKind {
    if method_map.contains_key(&expr.id) {
        // Overloaded operations are generally calls, and hence they are
        // generated via DPS.  However, assign_op (e.g., `x += y`) is an
        // exception, as its result is always unit.
        return match expr.node {
            ast::ExprAssignOp(*) => RvalueStmtExpr,
            _ => RvalueDpsExpr
        };
    }

    match expr.node {
        ast::ExprPath(*) | ast::ExprSelf => {
            match resolve_expr(tcx, expr) {
                ast::DefVariant(*) | ast::DefStruct(*) => RvalueDpsExpr,

                // Fn pointers are just scalar values.
                ast::DefFn(*) | ast::DefStaticMethod(*) => RvalueDatumExpr,

                // Note: there is actually a good case to be made that
                // def_args, particularly those of immediate type, ought to
                // considered rvalues.
                ast::DefStatic(*) |
                ast::DefBinding(*) |
                ast::DefUpvar(*) |
                ast::DefArg(*) |
                ast::DefLocal(*) |
                ast::DefSelf(*) => LvalueExpr,

                def => {
                    tcx.sess.span_bug(expr.span, format!(
                        "Uncategorized def for expr {:?}: {:?}",
                        expr.id, def));
                }
            }
        }

        ast::ExprUnary(_, ast::UnDeref, _) |
        ast::ExprField(*) |
        ast::ExprIndex(*) => {
            LvalueExpr
        }

        ast::ExprCall(*) |
        ast::ExprMethodCall(*) |
        ast::ExprStruct(*) |
        ast::ExprTup(*) |
        ast::ExprIf(*) |
        ast::ExprMatch(*) |
        ast::ExprFnBlock(*) |
        ast::ExprDoBody(*) |
        ast::ExprBlock(*) |
        ast::ExprRepeat(*) |
        ast::ExprLit(@codemap::Spanned {node: lit_str(_), _}) |
        ast::ExprVstore(_, ast::ExprVstoreSlice) |
        ast::ExprVstore(_, ast::ExprVstoreMutSlice) |
        ast::ExprVec(*) => {
            RvalueDpsExpr
        }

        ast::ExprCast(*) => {
            match tcx.node_types.find(&(expr.id as uint)) {
                Some(&t) => {
                    if type_is_trait(t) {
                        RvalueDpsExpr
                    } else {
                        RvalueDatumExpr
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

        ast::ExprBreak(*) |
        ast::ExprAgain(*) |
        ast::ExprRet(*) |
        ast::ExprWhile(*) |
        ast::ExprLoop(*) |
        ast::ExprAssign(*) |
        ast::ExprInlineAsm(*) |
        ast::ExprAssignOp(*) => {
            RvalueStmtExpr
        }

        ast::ExprForLoop(*) => fail2!("non-desugared expr_for_loop"),

        ast::ExprLogLevel |
        ast::ExprLit(_) | // Note: lit_str is carved out above
        ast::ExprUnary(*) |
        ast::ExprAddrOf(*) |
        ast::ExprBinary(*) |
        ast::ExprVstore(_, ast::ExprVstoreBox) |
        ast::ExprVstore(_, ast::ExprVstoreMutBox) |
        ast::ExprVstore(_, ast::ExprVstoreUniq) => {
            RvalueDatumExpr
        }

        ast::ExprParen(e) => expr_kind(tcx, method_map, e),

        ast::ExprMac(*) => {
            tcx.sess.span_bug(
                expr.span,
                "macro expression remains after expansion");
        }
    }
}

pub fn stmt_node_id(s: &ast::Stmt) -> ast::NodeId {
    match s.node {
      ast::StmtDecl(_, id) | StmtExpr(_, id) | StmtSemi(_, id) => {
        return id;
      }
      ast::StmtMac(*) => fail2!("unexpanded macro in trans")
    }
}

pub fn field_idx(name: ast::Name, fields: &[field]) -> Option<uint> {
    let mut i = 0u;
    for f in fields.iter() { if f.ident.name == name { return Some(i); } i += 1u; }
    return None;
}

pub fn field_idx_strict(tcx: ty::ctxt, name: ast::Name, fields: &[field])
                     -> uint {
    let mut i = 0u;
    for f in fields.iter() { if f.ident.name == name { return i; } i += 1u; }
    tcx.sess.bug(format!(
        "No field named `{}` found in the list of fields `{:?}`",
        token::interner_get(name),
        fields.map(|f| tcx.sess.str_of(f.ident))));
}

pub fn method_idx(id: ast::Ident, meths: &[@Method]) -> Option<uint> {
    meths.iter().position(|m| m.ident == id)
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

pub fn occurs_check(tcx: ctxt, sp: Span, vid: TyVid, rt: t) {
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
    if vars_in_type(rt).contains(&vid) {
            // Maybe this should be span_err -- however, there's an
            // assertion later on that the type doesn't contain
            // variables, so in this case we have to be sure to die.
            tcx.sess.span_fatal
                (sp, ~"type inference failed because I \
                     could not find a type\n that's both of the form "
                 + ::util::ppaux::ty_to_str(tcx, mk_var(tcx, vid)) +
                 " and of the form " + ::util::ppaux::ty_to_str(tcx, rt) +
                 " - such a type would have to be infinitely large.");
    }
}

pub fn ty_sort_str(cx: ctxt, t: t) -> ~str {
    match get(t).sty {
      ty_nil | ty_bot | ty_bool | ty_char | ty_int(_) |
      ty_uint(_) | ty_float(_) | ty_estr(_) |
      ty_type | ty_opaque_box | ty_opaque_closure_ptr(_) => {
        ::util::ppaux::ty_to_str(cx, t)
      }

      ty_enum(id, _) => format!("enum {}", item_path_str(cx, id)),
      ty_box(_) => ~"@-ptr",
      ty_uniq(_) => ~"~-ptr",
      ty_evec(_, _) => ~"vector",
      ty_unboxed_vec(_) => ~"unboxed vector",
      ty_ptr(_) => ~"*-ptr",
      ty_rptr(_, _) => ~"&-ptr",
      ty_bare_fn(_) => ~"extern fn",
      ty_closure(_) => ~"fn",
      ty_trait(id, _, _, _, _) => format!("trait {}", item_path_str(cx, id)),
      ty_struct(id, _) => format!("struct {}", item_path_str(cx, id)),
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
            format!("expected {} fn but found {} fn",
                 values.expected.to_str(), values.found.to_str())
        }
        terr_abi_mismatch(values) => {
            format!("expected {} fn but found {} fn",
                 values.expected.to_str(), values.found.to_str())
        }
        terr_onceness_mismatch(values) => {
            format!("expected {} fn but found {} fn",
                 values.expected.to_str(), values.found.to_str())
        }
        terr_sigil_mismatch(values) => {
            format!("expected {} closure, found {} closure",
                 values.expected.to_str(),
                 values.found.to_str())
        }
        terr_mutability => ~"values differ in mutability",
        terr_box_mutability => ~"boxed values differ in mutability",
        terr_vec_mutability => ~"vectors differ in mutability",
        terr_ptr_mutability => ~"pointers differ in mutability",
        terr_ref_mutability => ~"references differ in mutability",
        terr_ty_param_size(values) => {
            format!("expected a type with {} type params \
                  but found one with {} type params",
                 values.expected, values.found)
        }
        terr_tuple_size(values) => {
            format!("expected a tuple with {} elements \
                  but found one with {} elements",
                 values.expected, values.found)
        }
        terr_record_size(values) => {
            format!("expected a record with {} fields \
                  but found one with {} fields",
                 values.expected, values.found)
        }
        terr_record_mutability => {
            ~"record elements differ in mutability"
        }
        terr_record_fields(values) => {
            format!("expected a record with field `{}` but found one with field \
                  `{}`",
                 cx.sess.str_of(values.expected),
                 cx.sess.str_of(values.found))
        }
        terr_arg_count => ~"incorrect number of function parameters",
        terr_regions_does_not_outlive(*) => {
            format!("lifetime mismatch")
        }
        terr_regions_not_same(*) => {
            format!("lifetimes are not the same")
        }
        terr_regions_no_overlap(*) => {
            format!("lifetimes do not intersect")
        }
        terr_regions_insufficiently_polymorphic(br, _) => {
            format!("expected bound lifetime parameter {}, \
                  but found concrete lifetime",
                 bound_region_ptr_to_str(cx, br))
        }
        terr_regions_overly_polymorphic(br, _) => {
            format!("expected concrete lifetime, \
                  but found bound lifetime parameter {}",
                 bound_region_ptr_to_str(cx, br))
        }
        terr_vstores_differ(k, ref values) => {
            format!("{} storage differs: expected {} but found {}",
                 terr_vstore_kind_to_str(k),
                 vstore_to_str(cx, (*values).expected),
                 vstore_to_str(cx, (*values).found))
        }
        terr_trait_stores_differ(_, ref values) => {
            format!("trait storage differs: expected {} but found {}",
                 trait_store_to_str(cx, (*values).expected),
                 trait_store_to_str(cx, (*values).found))
        }
        terr_in_field(err, fname) => {
            format!("in field `{}`, {}", cx.sess.str_of(fname),
                 type_err_to_str(cx, err))
        }
        terr_sorts(values) => {
            format!("expected {} but found {}",
                 ty_sort_str(cx, values.expected),
                 ty_sort_str(cx, values.found))
        }
        terr_traits(values) => {
            format!("expected trait {} but found trait {}",
                 item_path_str(cx, values.expected),
                 item_path_str(cx, values.found))
        }
        terr_builtin_bounds(values) => {
            if values.expected.is_empty() {
                format!("expected no bounds but found `{}`",
                     values.found.user_string(cx))
            } else if values.found.is_empty() {
                format!("expected bounds `{}` but found no bounds",
                     values.expected.user_string(cx))
            } else {
                format!("expected bounds `{}` but found bounds `{}`",
                     values.expected.user_string(cx),
                     values.found.user_string(cx))
            }
        }
        terr_integer_as_char => {
            format!("expected an integral type but found char")
        }
        terr_int_mismatch(ref values) => {
            format!("expected {} but found {}",
                 values.expected.to_str(),
                 values.found.to_str())
        }
        terr_float_mismatch(ref values) => {
            format!("expected {} but found {}",
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

pub fn def_has_ty_params(def: ast::Def) -> bool {
    match def {
      ast::DefFn(_, _) | ast::DefVariant(_, _, _) | ast::DefStruct(_)
        => true,
      _ => false
    }
}

pub fn provided_source(cx: ctxt, id: ast::DefId) -> Option<ast::DefId> {
    cx.provided_method_sources.find(&id).map_move(|x| *x)
}

pub fn provided_trait_methods(cx: ctxt, id: ast::DefId) -> ~[@Method] {
    if is_local(id) {
        match cx.items.find(&id.node) {
            Some(&ast_map::node_item(@ast::item {
                        node: item_trait(_, _, ref ms),
                        _
                    }, _)) =>
                match ast_util::split_trait_methods(*ms) {
                   (_, p) => p.map(|m| method(cx, ast_util::local_def(m.id)))
                },
            _ => cx.sess.bug(format!("provided_trait_methods: {:?} is not a trait",
                                  id))
        }
    } else {
        csearch::get_provided_trait_methods(cx, id)
    }
}

pub fn trait_supertraits(cx: ctxt,
                         id: ast::DefId) -> @~[@TraitRef]
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

fn lookup_locally_or_in_crate_store<V:Clone>(
    descr: &str,
    def_id: ast::DefId,
    map: &mut HashMap<ast::DefId, V>,
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
        Some(&ref v) => { return (*v).clone(); }
        None => { }
    }

    if def_id.crate == ast::LOCAL_CRATE {
        fail2!("No def'n found for {:?} in tcx.{}", def_id, descr);
    }
    let v = load_external();
    map.insert(def_id, v.clone());
    v
}

pub fn trait_method(cx: ctxt, trait_did: ast::DefId, idx: uint) -> @Method {
    let method_def_id = ty::trait_method_def_ids(cx, trait_did)[idx];
    ty::method(cx, method_def_id)
}


pub fn trait_methods(cx: ctxt, trait_did: ast::DefId) -> @~[@Method] {
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

pub fn method(cx: ctxt, id: ast::DefId) -> @Method {
    lookup_locally_or_in_crate_store(
        "methods", id, cx.methods,
        || @csearch::get_method(cx, id))
}

pub fn trait_method_def_ids(cx: ctxt, id: ast::DefId) -> @~[DefId] {
    lookup_locally_or_in_crate_store(
        "methods", id, cx.trait_method_def_ids,
        || @csearch::get_trait_method_def_ids(cx.cstore, id))
}

pub fn impl_trait_ref(cx: ctxt, id: ast::DefId) -> Option<@TraitRef> {
    match cx.impl_trait_cache.find(&id) {
        Some(&ret) => { return ret; }
        None => {}
    }
    let ret = if id.crate == ast::LOCAL_CRATE {
        debug2!("(impl_trait_ref) searching for trait impl {:?}", id);
        match cx.items.find(&id.node) {
            Some(&ast_map::node_item(@ast::item {
                                     node: ast::item_impl(_, ref opt_trait, _, _),
                                     _},
                                     _)) => {
                match opt_trait {
                    &Some(ref t) => Some(ty::node_id_to_trait_ref(cx, t.ref_id)),
                    &None => None
                }
            }
            _ => None
        }
    } else {
        csearch::get_impl_trait(cx, id)
    };
    cx.impl_trait_cache.insert(id, ret);
    return ret;
}

pub fn trait_ref_to_def_id(tcx: ctxt, tr: &ast::trait_ref) -> ast::DefId {
    let def = tcx.def_map.find(&tr.ref_id).expect("no def-map entry for trait");
    ast_util::def_id_of_def(*def)
}

pub fn try_add_builtin_trait(tcx: ctxt,
                             trait_def_id: ast::DefId,
                             builtin_bounds: &mut BuiltinBounds) -> bool {
    //! Checks whether `trait_ref` refers to one of the builtin
    //! traits, like `Send`, and adds the corresponding
    //! bound to the set `builtin_bounds` if so. Returns true if `trait_ref`
    //! is a builtin trait.

    match tcx.lang_items.to_builtin_kind(trait_def_id) {
        Some(bound) => { builtin_bounds.add(bound); true }
        None => false
    }
}

pub fn ty_to_def_id(ty: t) -> Option<ast::DefId> {
    match get(ty).sty {
      ty_trait(id, _, _, _, _) | ty_struct(id, _) | ty_enum(id, _) => Some(id),
      _ => None
    }
}

/// Returns the def ID of the constructor for the given tuple-like struct, or
/// None if the struct is not tuple-like. Fails if the given def ID does not
/// refer to a struct at all.
fn struct_ctor_id(cx: ctxt, struct_did: ast::DefId) -> Option<ast::DefId> {
    if struct_did.crate != ast::LOCAL_CRATE {
        // XXX: Cross-crate functionality.
        cx.sess.unimpl("constructor ID of cross-crate tuple structs");
    }

    match cx.items.find(&struct_did.node) {
        Some(&ast_map::node_item(item, _)) => {
            match item.node {
                ast::item_struct(struct_def, _) => {
                    do struct_def.ctor_id.map_move |ctor_id| {
                        ast_util::local_def(ctor_id)
                    }
                }
                _ => cx.sess.bug("called struct_ctor_id on non-struct")
            }
        }
        _ => cx.sess.bug("called struct_ctor_id on non-struct")
    }
}

// Enum information
#[deriving(Clone)]
pub struct VariantInfo {
    args: ~[t],
    arg_names: Option<~[ast::Ident]>,
    ctor_ty: t,
    name: ast::Ident,
    id: ast::DefId,
    disr_val: Disr,
    vis: visibility
}

impl VariantInfo {

    /// Creates a new VariantInfo from the corresponding ast representation.
    ///
    /// Does not do any caching of the value in the type context.
    pub fn from_ast_variant(cx: ctxt,
                            ast_variant: &ast::variant,
                            discriminant: Disr) -> VariantInfo {
        let ctor_ty = node_id_to_type(cx, ast_variant.node.id);

        match ast_variant.node.kind {
            ast::tuple_variant_kind(ref args) => {
                let arg_tys = if args.len() > 0 { ty_fn_args(ctor_ty).map(|a| *a) } else { ~[] };

                return VariantInfo {
                    args: arg_tys,
                    arg_names: None,
                    ctor_ty: ctor_ty,
                    name: ast_variant.node.name,
                    id: ast_util::local_def(ast_variant.node.id),
                    disr_val: discriminant,
                    vis: ast_variant.node.vis
                };
            },
            ast::struct_variant_kind(ref struct_def) => {

                let fields: &[@struct_field] = struct_def.fields;

                assert!(fields.len() > 0);

                let arg_tys = ty_fn_args(ctor_ty).map(|a| *a);
                let arg_names = do fields.map |field| {
                    match field.node.kind {
                        named_field(ident, _) => ident,
                        unnamed_field => cx.sess.bug(
                            "enum_variants: all fields in struct must have a name")
                    }
                };

                return VariantInfo {
                    args: arg_tys,
                    arg_names: Some(arg_names),
                    ctor_ty: ctor_ty,
                    name: ast_variant.node.name,
                    id: ast_util::local_def(ast_variant.node.id),
                    disr_val: discriminant,
                    vis: ast_variant.node.vis
                };
            }
        }
    }
}

pub fn substd_enum_variants(cx: ctxt,
                            id: ast::DefId,
                            substs: &substs)
                         -> ~[@VariantInfo] {
    do enum_variants(cx, id).iter().map |variant_info| {
        let substd_args = variant_info.args.iter()
            .map(|aty| subst(cx, substs, *aty)).collect();

        let substd_ctor_ty = subst(cx, substs, variant_info.ctor_ty);

        @VariantInfo {
            args: substd_args,
            ctor_ty: substd_ctor_ty,
            ..(**variant_info).clone()
        }
    }.collect()
}

pub fn item_path_str(cx: ctxt, id: ast::DefId) -> ~str {
    ast_map::path_to_str(item_path(cx, id), token::get_ident_interner())
}

pub enum DtorKind {
    NoDtor,
    TraitDtor(DefId, bool)
}

impl DtorKind {
    pub fn is_not_present(&self) -> bool {
        match *self {
            NoDtor => true,
            _ => false
        }
    }

    pub fn is_present(&self) -> bool {
        !self.is_not_present()
    }

    pub fn has_drop_flag(&self) -> bool {
        match self {
            &NoDtor => false,
            &TraitDtor(_, flag) => flag
        }
    }
}

/* If struct_id names a struct with a dtor, return Some(the dtor's id).
   Otherwise return none. */
pub fn ty_dtor(cx: ctxt, struct_id: DefId) -> DtorKind {
    match cx.destructor_for_type.find(&struct_id) {
        Some(&method_def_id) => {
            let flag = !has_attr(cx, struct_id, "unsafe_no_drop_flag");

            TraitDtor(method_def_id, flag)
        }
        None => NoDtor,
    }
}

pub fn has_dtor(cx: ctxt, struct_id: DefId) -> bool {
    ty_dtor(cx, struct_id).is_present()
}

pub fn item_path(cx: ctxt, id: ast::DefId) -> ast_map::path {
    if id.crate != ast::LOCAL_CRATE {
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
            vec::append_one((*path).clone(), item_elt)
          }

          ast_map::node_foreign_item(nitem, _, _, path) => {
            vec::append_one((*path).clone(),
                            ast_map::path_name(nitem.ident))
          }

          ast_map::node_method(method, _, path) => {
            vec::append_one((*path).clone(),
                            ast_map::path_name(method.ident))
          }
          ast_map::node_trait_method(trait_method, _, path) => {
            let method = ast_util::trait_method_to_ty_method(&*trait_method);
            vec::append_one((*path).clone(),
                            ast_map::path_name(method.ident))
          }

          ast_map::node_variant(ref variant, _, path) => {
            vec::append_one(path.init().to_owned(),
                            ast_map::path_name((*variant).node.name))
          }

          ast_map::node_struct_ctor(_, item, path) => {
            vec::append_one((*path).clone(), ast_map::path_name(item.ident))
          }

          ref node => {
            cx.sess.bug(format!("cannot find item_path for node {:?}", node));
          }
        }
    }
}

pub fn enum_is_univariant(cx: ctxt, id: ast::DefId) -> bool {
    enum_variants(cx, id).len() == 1
}

pub fn type_is_empty(cx: ctxt, t: t) -> bool {
    match ty::get(t).sty {
       ty_enum(did, _) => (*enum_variants(cx, did)).is_empty(),
       _ => false
     }
}

pub fn enum_variants(cx: ctxt, id: ast::DefId) -> @~[@VariantInfo] {
    match cx.enum_var_cache.find(&id) {
      Some(&variants) => return variants,
      _ => { /* fallthrough */ }
    }

    let result = if ast::LOCAL_CRATE != id.crate {
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
            let mut last_discriminant: Option<Disr> = None;
            @enum_definition.variants.iter().map(|variant| {

                let mut discriminant = match last_discriminant {
                    Some(val) => val + 1,
                    None => INITIAL_DISCRIMINANT_VALUE
                };

                match variant.node.disr_expr {
                    Some(e) => match const_eval::eval_const_expr_partial(&cx, e) {
                        Ok(const_eval::const_int(val)) => discriminant = val as Disr,
                        Ok(const_eval::const_uint(val)) => discriminant = val as Disr,
                        Ok(_) => {
                            cx.sess.span_err(e.span, "expected signed integer constant");
                        }
                        Err(ref err) => {
                            cx.sess.span_err(e.span, format!("expected constant: {}", (*err)));
                        }
                    },
                    None => {}
                };

                let variant_info = @VariantInfo::from_ast_variant(cx, variant, discriminant);
                last_discriminant = Some(discriminant);
                variant_info

            }).collect()
          }
          _ => cx.sess.bug("enum_variants: id not bound to an enum")
        }
    };
    cx.enum_var_cache.insert(id, result);
    result
}


// Returns information about the enum variant with the given ID:
pub fn enum_variant_with_id(cx: ctxt,
                            enum_id: ast::DefId,
                            variant_id: ast::DefId)
                         -> @VariantInfo {
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
                        did: ast::DefId)
                     -> ty_param_bounds_and_ty {
    lookup_locally_or_in_crate_store(
        "tcache", did, cx.tcache,
        || csearch::get_type(cx, did))
}

pub fn lookup_impl_vtables(cx: ctxt,
                           did: ast::DefId)
                     -> typeck::impl_res {
    lookup_locally_or_in_crate_store(
        "impl_vtables", did, cx.impl_vtables,
        || csearch::get_impl_vtables(cx, did) )
}

/// Given the did of a trait, returns its canonical trait ref.
pub fn lookup_trait_def(cx: ctxt, did: ast::DefId) -> @ty::TraitDef {
    match cx.trait_defs.find(&did) {
        Some(&trait_def) => {
            // The item is in this crate. The caller should have added it to the
            // type cache already
            return trait_def;
        }
        None => {
            assert!(did.crate != ast::LOCAL_CRATE);
            let trait_def = @csearch::get_trait_def(cx, did);
            cx.trait_defs.insert(did, trait_def);
            return trait_def;
        }
    }
}

/// Determine whether an item is annotated with an attribute
pub fn has_attr(tcx: ctxt, did: DefId, attr: &str) -> bool {
    if is_local(did) {
        match tcx.items.find(&did.node) {
            Some(
                &ast_map::node_item(@ast::item {
                    attrs: ref attrs,
                    _
                }, _)) => attr::contains_name(*attrs, attr),
            _ => tcx.sess.bug(format!("has_attr: {:?} is not an item",
                                   did))
        }
    } else {
        let mut ret = false;
        do csearch::get_item_attrs(tcx.cstore, did) |meta_items| {
            ret = ret || attr::contains_name(meta_items, attr);
        }
        ret
    }
}

/// Determine whether an item is annotated with `#[packed]`
pub fn lookup_packed(tcx: ctxt, did: DefId) -> bool {
    has_attr(tcx, did, "packed")
}

/// Determine whether an item is annotated with `#[simd]`
pub fn lookup_simd(tcx: ctxt, did: DefId) -> bool {
    has_attr(tcx, did, "simd")
}

// Look up a field ID, whether or not it's local
// Takes a list of type substs in case the struct is generic
pub fn lookup_field_type(tcx: ctxt,
                         struct_id: DefId,
                         id: DefId,
                         substs: &substs)
                      -> ty::t {
    let t = if id.crate == ast::LOCAL_CRATE {
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
pub fn lookup_struct_fields(cx: ctxt, did: ast::DefId) -> ~[field_ty] {
  if did.crate == ast::LOCAL_CRATE {
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
               format!("struct ID not bound to an item: {}",
                    ast_map::node_id_to_str(cx.items, did.node,
                                            token::get_ident_interner())));
       }
    }
        }
  else {
        return csearch::get_struct_fields(cx.sess.cstore, did);
    }
}

pub fn lookup_struct_field(cx: ctxt,
                           parent: ast::DefId,
                           field_id: ast::DefId)
                        -> field_ty {
    let r = lookup_struct_fields(cx, parent);
    match r.iter().find(
                 |f| f.id.node == field_id.node) {
        Some(t) => *t,
        None => cx.sess.bug("struct ID not found in parent's fields")
    }
}

fn struct_field_tys(fields: &[@struct_field]) -> ~[field_ty] {
    do fields.map |field| {
        match field.node.kind {
            named_field(ident, visibility) => {
                field_ty {
                    name: ident.name,
                    id: ast_util::local_def(field.node.id),
                    vis: visibility,
                }
            }
            unnamed_field => {
                field_ty {
                    name:
                        syntax::parse::token::special_idents::unnamed_field.name,
                    id: ast_util::local_def(field.node.id),
                    vis: ast::public,
                }
            }
        }
    }
}

// Returns a list of fields corresponding to the struct's items. trans uses
// this. Takes a list of substs with which to instantiate field types.
pub fn struct_fields(cx: ctxt, did: ast::DefId, substs: &substs)
                     -> ~[field] {
    do lookup_struct_fields(cx, did).map |f| {
       field {
            // FIXME #6993: change type of field to Name and get rid of new()
            ident: ast::Ident::new(f.name),
            mt: mt {
                ty: lookup_field_type(cx, did, f.id, substs),
                mutbl: MutImmutable
            }
        }
    }
}

pub fn is_binopable(cx: ctxt, ty: t, op: ast::BinOp) -> bool {
    static tycat_other: int = 0;
    static tycat_bool: int = 1;
    static tycat_char: int = 2;
    static tycat_int: int = 3;
    static tycat_float: int = 4;
    static tycat_bot: int = 5;
    static tycat_raw_ptr: int = 6;

    static opcat_add: int = 0;
    static opcat_sub: int = 1;
    static opcat_mult: int = 2;
    static opcat_shift: int = 3;
    static opcat_rel: int = 4;
    static opcat_eq: int = 5;
    static opcat_bit: int = 6;
    static opcat_logic: int = 7;

    fn opcat(op: ast::BinOp) -> int {
        match op {
          ast::BiAdd => opcat_add,
          ast::BiSub => opcat_sub,
          ast::BiMul => opcat_mult,
          ast::BiDiv => opcat_mult,
          ast::BiRem => opcat_mult,
          ast::BiAnd => opcat_logic,
          ast::BiOr => opcat_logic,
          ast::BiBitXor => opcat_bit,
          ast::BiBitAnd => opcat_bit,
          ast::BiBitOr => opcat_bit,
          ast::BiShl => opcat_shift,
          ast::BiShr => opcat_shift,
          ast::BiEq => opcat_eq,
          ast::BiNe => opcat_eq,
          ast::BiLt => opcat_rel,
          ast::BiLe => opcat_rel,
          ast::BiGe => opcat_rel,
          ast::BiGt => opcat_rel
        }
    }

    fn tycat(cx: ctxt, ty: t) -> int {
        if type_is_simd(cx, ty) {
            return tycat(cx, simd_type(cx, ty))
        }
        match get(ty).sty {
          ty_char => tycat_char,
          ty_bool => tycat_bool,
          ty_int(_) | ty_uint(_) | ty_infer(IntVar(_)) => tycat_int,
          ty_float(_) | ty_infer(FloatVar(_)) => tycat_float,
          ty_bot => tycat_bot,
          ty_ptr(_) => tycat_raw_ptr,
          _ => tycat_other
        }
    }

    static t: bool = true;
    static f: bool = false;

    let tbl = [
    //           +, -, *, shift, rel, ==, bit, logic
    /*other*/   [f, f, f, f,     f,   f,  f,   f],
    /*bool*/    [f, f, f, f,     t,   t,  t,   t],
    /*char*/    [f, f, f, f,     t,   t,  f,   f],
    /*int*/     [t, t, t, t,     t,   t,  t,   f],
    /*float*/   [t, t, t, f,     t,   t,  f,   f],
    /*bot*/     [t, t, t, t,     t,   t,  t,   t],
    /*raw ptr*/ [f, f, f, f,     t,   t,  f,   f]];

    return tbl[tycat(cx, ty)][opcat(op)];
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
                ..(*closure_ty).clone()
            })
        }

        ty_enum(did, ref r) => {
            match (*r).regions {
                NonerasedRegions(_) => {
                    // trans doesn't care about regions
                    mk_enum(cx, did, substs {regions: ty::ErasedRegions,
                                             self_ty: None,
                                             tps: (*r).tps.clone()})
                }
                ErasedRegions => {
                    t
                }
            }
        }

        ty_struct(did, ref r) => {
            match (*r).regions {
                NonerasedRegions(_) => {
                    // Ditto.
                    mk_struct(cx, did, substs {regions: ty::ErasedRegions,
                                               self_ty: None,
                                               tps: (*r).tps.clone()})
                }
                ErasedRegions => {
                    t
                }
            }
        }

        _ =>
            t
    };

    let sty = fold_sty(&get(t).sty, |t| { normalize_ty(cx, t) });
    let t_norm = mk_t(cx, sty);
    cx.normalized_cache.insert(t, t_norm);
    return t_norm;
}

pub trait ExprTyProvider {
    fn expr_ty(&self, ex: &ast::Expr) -> t;
    fn ty_ctxt(&self) -> ctxt;
}

impl ExprTyProvider for ctxt {
    fn expr_ty(&self, ex: &ast::Expr) -> t {
        expr_ty(*self, ex)
    }

    fn ty_ctxt(&self) -> ctxt {
        *self
    }
}

// Returns the repeat count for a repeating vector expression.
pub fn eval_repeat_count<T: ExprTyProvider>(tcx: &T, count_expr: &ast::Expr) -> uint {
    match const_eval::eval_const_expr_partial(tcx, count_expr) {
      Ok(ref const_val) => match *const_val {
        const_eval::const_int(count) => if count < 0 {
            tcx.ty_ctxt().sess.span_err(count_expr.span,
                                        "expected positive integer for \
                                         repeat count but found negative integer");
            return 0;
        } else {
            return count as uint
        },
        const_eval::const_uint(count) => return count as uint,
        const_eval::const_float(count) => {
            tcx.ty_ctxt().sess.span_err(count_expr.span,
                                        "expected positive integer for \
                                         repeat count but found float");
            return count as uint;
        }
        const_eval::const_str(_) => {
            tcx.ty_ctxt().sess.span_err(count_expr.span,
                                        "expected positive integer for \
                                         repeat count but found string");
            return 0;
        }
        const_eval::const_bool(_) => {
            tcx.ty_ctxt().sess.span_err(count_expr.span,
                                        "expected positive integer for \
                                         repeat count but found boolean");
            return 0;
        }
      },
      Err(*) => {
        tcx.ty_ctxt().sess.span_err(count_expr.span,
                                    "expected constant integer for repeat count \
                                     but found variable");
        return 0;
      }
    }
}

// Determine what purity to check a nested function under
pub fn determine_inherited_purity(parent: (ast::purity, ast::NodeId),
                                  child: (ast::purity, ast::NodeId),
                                  child_sigil: ast::Sigil)
                                    -> (ast::purity, ast::NodeId) {
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
                                        bounds: &[@TraitRef],
                                        f: &fn(@TraitRef) -> bool) -> bool {
    for &bound_trait_ref in bounds.iter() {
        let mut supertrait_set = HashMap::new();
        let mut trait_refs = ~[];
        let mut i = 0;

        // Seed the worklist with the trait from the bound
        supertrait_set.insert(bound_trait_ref.def_id, ());
        trait_refs.push(bound_trait_ref);

        // Add the given trait ty to the hash map
        while i < trait_refs.len() {
            debug2!("each_bound_trait_and_supertraits(i={:?}, trait_ref={})",
                   i, trait_refs[i].repr(tcx));

            if !f(trait_refs[i]) {
                return false;
            }

            // Add supertraits to supertrait_set
            let supertrait_refs = trait_ref_supertraits(tcx, trait_refs[i]);
            for &supertrait_ref in supertrait_refs.iter() {
                debug2!("each_bound_trait_and_supertraits(supertrait_ref={})",
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
    for type_param_def in type_param_defs.iter() {
        do each_bound_trait_and_supertraits(
            tcx, type_param_def.bounds.trait_bounds) |_| {
            total += 1;
            true
        };
    }
    return total;
}

pub fn get_tydesc_ty(tcx: ctxt) -> Result<t, ~str> {
    do tcx.lang_items.require(TyDescStructLangItem).map_move |tydesc_lang_item| {
        tcx.intrinsic_defs.find_copy(&tydesc_lang_item)
            .expect("Failed to resolve TyDesc")
    }
}

pub fn get_opaque_ty(tcx: ctxt) -> Result<t, ~str> {
    do tcx.lang_items.require(OpaqueStructLangItem).map_move |opaque_lang_item| {
        tcx.intrinsic_defs.find_copy(&opaque_lang_item)
            .expect("Failed to resolve Opaque")
    }
}

pub fn visitor_object_ty(tcx: ctxt,
                         region: ty::Region) -> Result<(@TraitRef, t), ~str> {
    let trait_lang_item = match tcx.lang_items.require(TyVisitorTraitLangItem) {
        Ok(id) => id,
        Err(s) => { return Err(s); }
    };
    let substs = substs {
        regions: ty::NonerasedRegions(opt_vec::Empty),
        self_ty: None,
        tps: ~[]
    };
    let trait_ref = @TraitRef { def_id: trait_lang_item, substs: substs };
    Ok((trait_ref,
        mk_trait(tcx,
                 trait_ref.def_id,
                 trait_ref.substs.clone(),
                 RegionTraitStore(region),
                 ast::MutMutable,
                 EmptyBuiltinBounds())))
}

/// Records a trait-to-implementation mapping.
fn record_trait_implementation(tcx: ctxt,
                               trait_def_id: DefId,
                               implementation: @Impl) {
    let implementation_list;
    match tcx.trait_impls.find(&trait_def_id) {
        None => {
            implementation_list = @mut ~[];
            tcx.trait_impls.insert(trait_def_id, implementation_list);
        }
        Some(&existing_implementation_list) => {
            implementation_list = existing_implementation_list
        }
    }

    implementation_list.push(implementation);
}

/// Populates the type context with all the implementations for the given type
/// if necessary.
pub fn populate_implementations_for_type_if_necessary(tcx: ctxt,
                                                      type_id: ast::DefId) {
    if type_id.crate == LOCAL_CRATE {
        return
    }
    if tcx.populated_external_types.contains(&type_id) {
        return
    }

    do csearch::each_implementation_for_type(tcx.sess.cstore, type_id)
            |implementation_def_id| {
        let implementation = @csearch::get_impl(tcx, implementation_def_id);

        // Record the trait->implementation mappings, if applicable.
        let associated_traits = csearch::get_impl_trait(tcx,
                                                        implementation.did);
        for trait_ref in associated_traits.iter() {
            record_trait_implementation(tcx,
                                        trait_ref.def_id,
                                        implementation);
        }

        // For any methods that use a default implementation, add them to
        // the map. This is a bit unfortunate.
        for method in implementation.methods.iter() {
            for source in method.provided_source.iter() {
                tcx.provided_method_sources.insert(method.def_id, *source);
            }
        }

        // If this is an inherent implementation, record it.
        if associated_traits.is_none() {
            let implementation_list;
            match tcx.inherent_impls.find(&type_id) {
                None => {
                    implementation_list = @mut ~[];
                    tcx.inherent_impls.insert(type_id, implementation_list);
                }
                Some(&existing_implementation_list) => {
                    implementation_list = existing_implementation_list;
                }
            }
            implementation_list.push(implementation);
        }

        // Store the implementation info.
        tcx.impls.insert(implementation_def_id, implementation);
    }

    tcx.populated_external_types.insert(type_id);
}

/// Populates the type context with all the implementations for the given
/// trait if necessary.
pub fn populate_implementations_for_trait_if_necessary(
        tcx: ctxt,
        trait_id: ast::DefId) {
    if trait_id.crate == LOCAL_CRATE {
        return
    }
    if tcx.populated_external_traits.contains(&trait_id) {
        return
    }

    do csearch::each_implementation_for_trait(tcx.sess.cstore, trait_id)
            |implementation_def_id| {
        let implementation = @csearch::get_impl(tcx, implementation_def_id);

        // Record the trait->implementation mapping.
        record_trait_implementation(tcx, trait_id, implementation);

        // For any methods that use a default implementation, add them to
        // the map. This is a bit unfortunate.
        for method in implementation.methods.iter() {
            for source in method.provided_source.iter() {
                tcx.provided_method_sources.insert(method.def_id, *source);
            }
        }

        // Store the implementation info.
        tcx.impls.insert(implementation_def_id, implementation);
    }

    tcx.populated_external_traits.insert(trait_id);
}

/// If the given def ID describes a trait method, returns the ID of the trait
/// that the method belongs to. Otherwise, returns `None`.
pub fn trait_of_method(tcx: ctxt, def_id: ast::DefId)
                       -> Option<ast::DefId> {
    match tcx.methods.find(&def_id) {
        Some(method_descriptor) => {
            match method_descriptor.container {
                TraitContainer(id) => return Some(id),
                _ => {}
            }
        }
        None => {}
    }

    // If the method was in the local crate, then if we got here we know the
    // answer is negative.
    if def_id.crate == LOCAL_CRATE {
        return None
    }

    let result = csearch::get_trait_of_method(tcx.cstore, def_id, tcx);

    result
}
