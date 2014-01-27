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
use middle::lang_items::{ExchangeHeapLangItem, OpaqueStructLangItem};
use middle::lang_items::{TyDescStructLangItem, TyVisitorTraitLangItem};
use middle::freevars;
use middle::resolve;
use middle::resolve_lifetime;
use middle::ty;
use middle::subst::Subst;
use middle::typeck;
use middle::ty_fold;
use middle::ty_fold::TypeFolder;
use middle;
use util::ppaux::{note_and_explain_region, bound_region_ptr_to_str};
use util::ppaux::{trait_store_to_str, ty_to_str, vstore_to_str};
use util::ppaux::{Repr, UserString};
use util::common::{indenter};

use std::cast;
use std::cell::{Cell, RefCell};
use std::cmp;
use std::hashmap::{HashMap, HashSet};
use std::ops;
use std::ptr::to_unsafe_ptr;
use std::to_bytes;
use std::to_str::ToStr;
use std::vec;
use syntax::ast::*;
use syntax::ast_util::{is_local, lit_is_str};
use syntax::ast_util;
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::codemap::Span;
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
    fty: BareFnTy,
    explicit_self: ast::ExplicitSelf_,
    vis: ast::Visibility,
    def_id: ast::DefId,
    container: MethodContainer,

    // If this method is provided, we need to know where it came from
    provided_source: Option<ast::DefId>
}

impl Method {
    pub fn new(ident: ast::Ident,
               generics: ty::Generics,
               fty: BareFnTy,
               explicit_self: ast::ExplicitSelf_,
               vis: ast::Visibility,
               def_id: ast::DefId,
               container: MethodContainer,
               provided_source: Option<ast::DefId>)
               -> Method {
       Method {
            ident: ident,
            generics: generics,
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

pub struct field_ty {
    name: Name,
    id: DefId,
    vis: ast::Visibility,
}

// Contains information needed to resolve types and (in the future) look up
// the types of AST nodes.
#[deriving(Eq,IterBytes)]
pub struct creader_cache_key {
    cnum: CrateNum,
    pos: uint,
    len: uint
}

type creader_cache = RefCell<HashMap<creader_cache_key, t>>;

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

#[deriving(Clone, Eq, Decodable, Encodable)]
pub struct ItemVariances {
    self_param: Option<Variance>,
    type_params: OptVec<Variance>,
    region_params: OptVec<Variance>
}

#[deriving(Clone, Eq, Decodable, Encodable)]
pub enum Variance {
    Covariant,      // T<A> <: T<B> iff A <: B -- e.g., function return type
    Invariant,      // T<A> <: T<B> iff B == A -- e.g., type of mutable cell
    Contravariant,  // T<A> <: T<B> iff B <: A -- e.g., function param type
    Bivariant,      // T<A> <: T<B>            -- e.g., unused type parameter
}

pub enum AutoAdjustment {
    AutoAddEnv(ty::Region, ast::Sigil),
    AutoDerefRef(AutoDerefRef),
    AutoObject(ast::Sigil, Option<ty::Region>,
               ast::Mutability,
               ty::BuiltinBounds,
               ast::DefId, /* Trait ID */
               ty::substs /* Trait substitutions */)
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

    /// Convert from @fn()/~fn()/|| to ||
    AutoBorrowFn(Region),

    /// Convert from T to *T
    AutoUnsafe(ast::Mutability),

    /// Convert from @Trait/~Trait/&Trait to &Trait
    AutoBorrowObj(Region, ast::Mutability),
}

pub type ctxt = @ctxt_;

/// The data structure to keep track of all the information that typechecker
/// generates so that so that it can be reused and doesn't have to be redone
/// later on.
pub struct ctxt_ {
    diag: @syntax::diagnostic::SpanHandler,
    interner: RefCell<HashMap<intern_key, ~t_box_>>,
    next_id: Cell<uint>,
    cstore: @metadata::cstore::CStore,
    sess: session::Session,
    def_map: resolve::DefMap,

    named_region_map: @RefCell<resolve_lifetime::NamedRegionMap>,

    region_maps: middle::region::RegionMaps,

    // Stores the types for various nodes in the AST.  Note that this table
    // is not guaranteed to be populated until after typeck.  See
    // typeck::check::fn_ctxt for details.
    node_types: node_type_table,

    // Stores the type parameters which were substituted to obtain the type
    // of this node.  This only applies to nodes that refer to entities
    // parameterized by type parameters, such as generic fns, types, or
    // other items.
    node_type_substs: RefCell<HashMap<NodeId, ~[t]>>,

    // Maps from a method to the method "descriptor"
    methods: RefCell<HashMap<DefId, @Method>>,

    // Maps from a trait def-id to a list of the def-ids of its methods
    trait_method_def_ids: RefCell<HashMap<DefId, @~[DefId]>>,

    // A cache for the trait_methods() routine
    trait_methods_cache: RefCell<HashMap<DefId, @~[@Method]>>,

    impl_trait_cache: RefCell<HashMap<ast::DefId, Option<@ty::TraitRef>>>,

    trait_refs: RefCell<HashMap<NodeId, @TraitRef>>,
    trait_defs: RefCell<HashMap<DefId, @TraitDef>>,

    /// Despite its name, `items` does not only map NodeId to an item but
    /// also to expr/stmt/local/arg/etc
    items: ast_map::Map,
    intrinsic_defs: RefCell<HashMap<ast::DefId, t>>,
    freevars: RefCell<freevars::freevar_map>,
    tcache: type_cache,
    rcache: creader_cache,
    short_names_cache: RefCell<HashMap<t, @str>>,
    needs_unwind_cleanup_cache: RefCell<HashMap<t, bool>>,
    tc_cache: RefCell<HashMap<uint, TypeContents>>,
    ast_ty_to_ty_cache: RefCell<HashMap<NodeId, ast_ty_to_ty_cache_entry>>,
    enum_var_cache: RefCell<HashMap<DefId, @~[@VariantInfo]>>,
    ty_param_defs: RefCell<HashMap<ast::NodeId, TypeParameterDef>>,
    adjustments: RefCell<HashMap<ast::NodeId, @AutoAdjustment>>,
    normalized_cache: RefCell<HashMap<t, t>>,
    lang_items: middle::lang_items::LanguageItems,
    // A mapping of fake provided method def_ids to the default implementation
    provided_method_sources: RefCell<HashMap<ast::DefId, ast::DefId>>,
    supertraits: RefCell<HashMap<ast::DefId, @~[@TraitRef]>>,

    // Maps from def-id of a type or region parameter to its
    // (inferred) variance.
    item_variance_map: RefCell<HashMap<ast::DefId, @ItemVariances>>,

    // A mapping from the def ID of an enum or struct type to the def ID
    // of the method that implements its destructor. If the type is not
    // present in this map, it does not have a destructor. This map is
    // populated during the coherence phase of typechecking.
    destructor_for_type: RefCell<HashMap<ast::DefId, ast::DefId>>,

    // A method will be in this list if and only if it is a destructor.
    destructors: RefCell<HashSet<ast::DefId>>,

    // Maps a trait onto a list of impls of that trait.
    trait_impls: RefCell<HashMap<ast::DefId, @RefCell<~[@Impl]>>>,

    // Maps a def_id of a type to a list of its inherent impls.
    // Contains implementations of methods that are inherent to a type.
    // Methods in these implementations don't need to be exported.
    inherent_impls: RefCell<HashMap<ast::DefId, @RefCell<~[@Impl]>>>,

    // Maps a def_id of an impl to an Impl structure.
    // Note that this contains all of the impls that we know about,
    // including ones in other crates. It's not clear that this is the best
    // way to do it.
    impls: RefCell<HashMap<ast::DefId, @Impl>>,

    // Set of used unsafe nodes (functions or blocks). Unsafe nodes not
    // present in this set can be warned about.
    used_unsafe: RefCell<HashSet<ast::NodeId>>,

    // Set of nodes which mark locals as mutable which end up getting used at
    // some point. Local variable definitions not in this set can be warned
    // about.
    used_mut_nodes: RefCell<HashSet<ast::NodeId>>,

    // vtable resolution information for impl declarations
    impl_vtables: typeck::impl_vtable_map,

    // The set of external nominal types whose implementations have been read.
    // This is used for lazy resolution of methods.
    populated_external_types: RefCell<HashSet<ast::DefId>>,

    // The set of external traits whose implementations have been read. This
    // is used for lazy resolution of traits.
    populated_external_traits: RefCell<HashSet<ast::DefId>>,

    // These two caches are used by const_eval when decoding external statics
    // and variants that are found.
    extern_const_statics: RefCell<HashMap<ast::DefId, Option<@ast::Expr>>>,
    extern_const_variants: RefCell<HashMap<ast::DefId, Option<@ast::Expr>>>,
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
    purity: ast::Purity,
    abis: AbiSet,
    sig: FnSig
}

#[deriving(Clone, Eq, IterBytes)]
pub struct ClosureTy {
    purity: ast::Purity,
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
 * - `binder_id` is the node id where this fn type appeared;
 *   it is used to identify all the bound regions appearing
 *   in the input/output types that are bound by this fn type
 *   (vs some enclosing or enclosed fn type)
 * - `inputs` is the list of arguments and their modes.
 * - `output` is the return type.
 * - `variadic` indicates whether this is a varidic function. (only true for foreign fns)
 */
#[deriving(Clone, Eq, IterBytes)]
pub struct FnSig {
    binder_id: ast::NodeId,
    inputs: ~[t],
    output: t,
    variadic: bool
}

#[deriving(Clone, Eq, IterBytes)]
pub struct param_ty {
    idx: uint,
    def_id: DefId
}

/// Representation of regions:
#[deriving(Clone, Eq, IterBytes, Encodable, Decodable, ToStr)]
pub enum Region {
    // Region bound in a type or fn declaration which will be
    // substituted 'early' -- that is, at the same time when type
    // parameters are substituted.
    ReEarlyBound(/* param id */ ast::NodeId, /*index*/ uint, ast::Ident),

    // Region bound in a function scope, which will be substituted when the
    // function is called. The first argument must be the `binder_id` of
    // some enclosing function signature.
    ReLateBound(/* binder_id */ ast::NodeId, BoundRegion),

    /// When checking a function body, the types of all arguments and so forth
    /// that refer to bound region parameters are modified to refer to free
    /// region parameters.
    ReFree(FreeRegion),

    /// A concrete region naming some expression within the current function.
    ReScope(NodeId),

    /// Static data that has an "infinite" lifetime. Top in the region lattice.
    ReStatic,

    /// A region variable.  Should not exist after typeck.
    ReInfer(InferRegion),

    /// Empty lifetime is for data that is never accessed.
    /// Bottom in the region lattice. We treat ReEmpty somewhat
    /// specially; at least right now, we do not generate instances of
    /// it during the GLB computations, but rather
    /// generate an error instead. This is to improve error messages.
    /// The only way to get an instance of ReEmpty is to have a region
    /// variable with no constraints.
    ReEmpty,
}

impl Region {
    pub fn is_bound(&self) -> bool {
        match self {
            &ty::ReEarlyBound(..) => true,
            &ty::ReLateBound(..) => true,
            _ => false
        }
    }
}

#[deriving(Clone, Eq, TotalOrd, TotalEq, IterBytes, Encodable, Decodable, ToStr)]
pub struct FreeRegion {
    scope_id: NodeId,
    bound_region: BoundRegion
}

#[deriving(Clone, Eq, TotalEq, TotalOrd, IterBytes, Encodable, Decodable, ToStr)]
pub enum BoundRegion {
    /// An anonymous region parameter for a given fn (&T)
    BrAnon(uint),

    /// Named region parameters for functions (a in &'a T)
    ///
    /// The def-id is needed to distinguish free regions in
    /// the event of shadowing.
    BrNamed(ast::DefId, ast::Ident),

    /// Fresh bound identifiers created during GLB computations.
    BrFresh(uint),
}

/**
 * Represents the values to use when substituting lifetime parameters.
 * If the value is `ErasedRegions`, then this subst is occurring during
 * trans, and all region parameters will be replaced with `ty::ReStatic`. */
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
    def_prim_ty!(TY_INT,    super::ty_int(ast::TyI),        3)
    def_prim_ty!(TY_I8,     super::ty_int(ast::TyI8),       4)
    def_prim_ty!(TY_I16,    super::ty_int(ast::TyI16),      5)
    def_prim_ty!(TY_I32,    super::ty_int(ast::TyI32),      6)
    def_prim_ty!(TY_I64,    super::ty_int(ast::TyI64),      7)
    def_prim_ty!(TY_UINT,   super::ty_uint(ast::TyU),       8)
    def_prim_ty!(TY_U8,     super::ty_uint(ast::TyU8),      9)
    def_prim_ty!(TY_U16,    super::ty_uint(ast::TyU16),     10)
    def_prim_ty!(TY_U32,    super::ty_uint(ast::TyU32),     11)
    def_prim_ty!(TY_U64,    super::ty_uint(ast::TyU64),     12)
    def_prim_ty!(TY_F32,    super::ty_float(ast::TyF32),    14)
    def_prim_ty!(TY_F64,    super::ty_float(ast::TyF64),    15)

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
    ty_int(ast::IntTy),
    ty_uint(ast::UintTy),
    ty_float(ast::FloatTy),
    ty_str(vstore),
    ty_enum(DefId, substs),
    ty_box(t),
    ty_uniq(t),
    ty_vec(mt, vstore),
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
    ty_unboxed_vec(mt),
}

#[deriving(Eq, IterBytes)]
pub struct TraitRef {
    def_id: DefId,
    substs: substs
}

#[deriving(Clone, Eq)]
pub enum IntVarValue {
    IntType(ast::IntTy),
    UintType(ast::UintTy),
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
    terr_purity_mismatch(expected_found<Purity>),
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
    terr_regions_insufficiently_polymorphic(BoundRegion, Region),
    terr_regions_overly_polymorphic(BoundRegion, Region),
    terr_vstores_differ(terr_vstore_kind, expected_found<vstore>),
    terr_trait_stores_differ(terr_vstore_kind, expected_found<TraitStore>),
    terr_in_field(@type_err, ast::Ident),
    terr_sorts(expected_found<t>),
    terr_integer_as_char,
    terr_int_mismatch(expected_found<IntVarValue>),
    terr_float_mismatch(expected_found<ast::FloatTy>),
    terr_traits(expected_found<ast::DefId>),
    terr_builtin_bounds(expected_found<BuiltinBounds>),
    terr_variadic_mismatch(expected_found<bool>)
}

#[deriving(Eq, IterBytes)]
pub struct ParamBounds {
    builtin_bounds: BuiltinBounds,
    trait_bounds: ~[@TraitRef]
}

pub type BuiltinBounds = EnumSet<BuiltinBound>;

#[deriving(Clone, Encodable, Eq, Decodable, IterBytes, ToStr)]
#[repr(uint)]
pub enum BuiltinBound {
    BoundStatic,
    BoundSend,
    BoundFreeze,
    BoundSized,
    BoundPod,
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
    ReSkolemized(uint, BoundRegion)
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
    fn to_uint(&self) -> uint { let TyVid(v) = *self; v }
}

impl ToStr for TyVid {
    fn to_str(&self) -> ~str { format!("<generic \\#{}>", self.to_uint()) }
}

impl Vid for IntVid {
    fn to_uint(&self) -> uint { let IntVid(v) = *self; v }
}

impl ToStr for IntVid {
    fn to_str(&self) -> ~str { format!("<generic integer \\#{}>", self.to_uint()) }
}

impl Vid for FloatVid {
    fn to_uint(&self) -> uint { let FloatVid(v) = *self; v }
}

impl ToStr for FloatVid {
    fn to_str(&self) -> ~str { format!("<generic float \\#{}>", self.to_uint()) }
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

#[deriving(Encodable, Decodable, Clone)]
pub struct RegionParameterDef {
    ident: ast::Ident,
    def_id: ast::DefId,
}

/// Information about the type/lifetime parameters associated with an item.
/// Analogous to ast::Generics.
#[deriving(Clone)]
pub struct Generics {
    /// List of type parameters declared on the item.
    type_param_defs: @~[TypeParameterDef],

    /// List of region parameters declared on the item.
    region_param_defs: @[RegionParameterDef],
}

impl Generics {
    pub fn has_type_params(&self) -> bool {
        !self.type_param_defs.is_empty()
    }
}

/// When type checking, we use the `ParameterEnvironment` to track
/// details about the type/lifetime parameters that are in scope.
/// It primarily stores the bounds information.
///
/// Note: This information might seem to be redundant with the data in
/// `tcx.ty_param_defs`, but it is not. That table contains the
/// parameter definitions from an "outside" perspective, but this
/// struct will contain the bounds for a parameter as seen from inside
/// the function body. Currently the only real distinction is that
/// bound lifetime parameters are replaced with free ones, but in the
/// future I hope to refine the representation of types so as to make
/// more distinctions clearer.
pub struct ParameterEnvironment {
    /// A substitution that can be applied to move from
    /// the "outer" view of a type or method to the "inner" view.
    /// In general, this means converting from bound parameters to
    /// free parameters. Since we currently represent bound/free type
    /// parameters in the same way, this only has an affect on regions.
    free_substs: ty::substs,

    /// Bound on the Self parameter
    self_param_bound: Option<@TraitRef>,

    /// Bounds on each numbered type parameter
    type_param_bounds: ~[ParamBounds],
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

type type_cache = RefCell<HashMap<ast::DefId, ty_param_bounds_and_ty>>;

pub type node_type_table = RefCell<HashMap<uint,t>>;

pub fn mk_ctxt(s: session::Session,
               dm: resolve::DefMap,
               named_region_map: @RefCell<resolve_lifetime::NamedRegionMap>,
               amap: ast_map::Map,
               freevars: freevars::freevar_map,
               region_maps: middle::region::RegionMaps,
               lang_items: middle::lang_items::LanguageItems)
            -> ctxt {
    @ctxt_ {
        named_region_map: named_region_map,
        item_variance_map: RefCell::new(HashMap::new()),
        diag: s.diagnostic(),
        interner: RefCell::new(HashMap::new()),
        next_id: Cell::new(primitives::LAST_PRIMITIVE_ID),
        cstore: s.cstore,
        sess: s,
        def_map: dm,
        region_maps: region_maps,
        node_types: RefCell::new(HashMap::new()),
        node_type_substs: RefCell::new(HashMap::new()),
        trait_refs: RefCell::new(HashMap::new()),
        trait_defs: RefCell::new(HashMap::new()),
        items: amap,
        intrinsic_defs: RefCell::new(HashMap::new()),
        freevars: RefCell::new(freevars),
        tcache: RefCell::new(HashMap::new()),
        rcache: RefCell::new(HashMap::new()),
        short_names_cache: RefCell::new(HashMap::new()),
        needs_unwind_cleanup_cache: RefCell::new(HashMap::new()),
        tc_cache: RefCell::new(HashMap::new()),
        ast_ty_to_ty_cache: RefCell::new(HashMap::new()),
        enum_var_cache: RefCell::new(HashMap::new()),
        methods: RefCell::new(HashMap::new()),
        trait_method_def_ids: RefCell::new(HashMap::new()),
        trait_methods_cache: RefCell::new(HashMap::new()),
        impl_trait_cache: RefCell::new(HashMap::new()),
        ty_param_defs: RefCell::new(HashMap::new()),
        adjustments: RefCell::new(HashMap::new()),
        normalized_cache: RefCell::new(HashMap::new()),
        lang_items: lang_items,
        provided_method_sources: RefCell::new(HashMap::new()),
        supertraits: RefCell::new(HashMap::new()),
        destructor_for_type: RefCell::new(HashMap::new()),
        destructors: RefCell::new(HashSet::new()),
        trait_impls: RefCell::new(HashMap::new()),
        inherent_impls: RefCell::new(HashMap::new()),
        impls: RefCell::new(HashMap::new()),
        used_unsafe: RefCell::new(HashSet::new()),
        used_mut_nodes: RefCell::new(HashSet::new()),
        impl_vtables: RefCell::new(HashMap::new()),
        populated_external_types: RefCell::new(HashSet::new()),
        populated_external_traits: RefCell::new(HashSet::new()),

        extern_const_statics: RefCell::new(HashMap::new()),
        extern_const_variants: RefCell::new(HashMap::new()),
     }
}

// Type constructors

// Interns a type/name combination, stores the resulting box in cx.interner,
// and returns the box as cast to an unsafe ptr (see comments for t above).
pub fn mk_t(cx: ctxt, st: sty) -> t {
    // Check for primitive types.
    match st {
        ty_nil => return mk_nil(),
        ty_err => return mk_err(),
        ty_bool => return mk_bool(),
        ty_int(i) => return mk_mach_int(i),
        ty_uint(u) => return mk_mach_uint(u),
        ty_float(f) => return mk_mach_float(f),
        ty_char => return mk_char(),
        ty_bot => return mk_bot(),
        _ => {}
    };

    let key = intern_key { sty: to_unsafe_ptr(&st) };

    {
        let mut interner = cx.interner.borrow_mut();
        match interner.get().find(&key) {
          Some(t) => unsafe { return cast::transmute(&t.sty); },
          _ => ()
        }
    }

    let mut flags = 0u;
    fn rflags(r: Region) -> uint {
        (has_regions as uint) | {
            match r {
              ty::ReInfer(_) => needs_infer as uint,
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
      &ty_str(vstore_slice(r)) => {
        flags |= rflags(r);
      }
      &ty_vec(ref mt, vstore_slice(r)) => {
        flags |= rflags(r);
        flags |= get(mt.ty).flags;
      }
      &ty_nil | &ty_bool | &ty_char | &ty_int(_) | &ty_float(_) | &ty_uint(_) |
      &ty_str(_) | &ty_type => {}
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
      &ty_box(tt) | &ty_uniq(tt) => {
        flags |= get(tt).flags
      }
      &ty_vec(ref m, _) | &ty_ptr(ref m) |
      &ty_unboxed_vec(ref m) => {
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
        id: cx.next_id.get(),
        flags: flags,
    };

    let sty_ptr = to_unsafe_ptr(&t.sty);

    let key = intern_key {
        sty: sty_ptr,
    };

    let mut interner = cx.interner.borrow_mut();
    interner.get().insert(key, t);

    cx.next_id.set(cx.next_id.get() + 1);

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

pub fn mk_mach_int(tm: ast::IntTy) -> t {
    match tm {
        ast::TyI    => mk_int(),
        ast::TyI8   => mk_i8(),
        ast::TyI16  => mk_i16(),
        ast::TyI32  => mk_i32(),
        ast::TyI64  => mk_i64(),
    }
}

pub fn mk_mach_uint(tm: ast::UintTy) -> t {
    match tm {
        ast::TyU    => mk_uint(),
        ast::TyU8   => mk_u8(),
        ast::TyU16  => mk_u16(),
        ast::TyU32  => mk_u32(),
        ast::TyU64  => mk_u64(),
    }
}

pub fn mk_mach_float(tm: ast::FloatTy) -> t {
    match tm {
        ast::TyF32  => mk_f32(),
        ast::TyF64  => mk_f64(),
    }
}

#[inline]
pub fn mk_char() -> t { mk_prim_t(&primitives::TY_CHAR) }

pub fn mk_str(cx: ctxt, t: vstore) -> t {
    mk_t(cx, ty_str(t))
}

pub fn mk_enum(cx: ctxt, did: ast::DefId, substs: substs) -> t {
    // take a copy of substs so that we own the vectors inside
    mk_t(cx, ty_enum(did, substs))
}

pub fn mk_box(cx: ctxt, ty: t) -> t { mk_t(cx, ty_box(ty)) }

pub fn mk_uniq(cx: ctxt, ty: t) -> t { mk_t(cx, ty_uniq(ty)) }

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

pub fn mk_vec(cx: ctxt, tm: mt, t: vstore) -> t {
    mk_t(cx, ty_vec(tm, t))
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

pub fn mk_ctor_fn(cx: ctxt,
                  binder_id: ast::NodeId,
                  input_tys: &[ty::t],
                  output: ty::t) -> t {
    let input_args = input_tys.map(|t| *t);
    mk_bare_fn(cx,
               BareFnTy {
                   purity: ast::ImpureFn,
                   abis: AbiSet::Rust(),
                   sig: FnSig {
                    binder_id: binder_id,
                    inputs: input_args,
                    output: output,
                    variadic: false
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

pub fn walk_ty(ty: t, f: |t|) {
    maybe_walk_ty(ty, |t| { f(t); true });
}

pub fn maybe_walk_ty(ty: t, f: |t| -> bool) {
    if !f(ty) {
        return;
    }
    match get(ty).sty {
        ty_nil | ty_bot | ty_bool | ty_char | ty_int(_) | ty_uint(_) | ty_float(_) |
        ty_str(_) | ty_type | ty_self(_) |
        ty_infer(_) | ty_param(_) | ty_err => {}
        ty_box(ty) | ty_uniq(ty) => maybe_walk_ty(ty, f),
        ty_vec(ref tm, _) | ty_unboxed_vec(ref tm) | ty_ptr(ref tm) |
        ty_rptr(_, ref tm) => {
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

// Folds types from the bottom up.
pub fn fold_ty(cx: ctxt, t0: t, fldop: |t| -> t) -> t {
    let mut f = ty_fold::BottomUpFolder {tcx: cx, fldop: fldop};
    f.fold_ty(t0)
}

pub fn walk_regions_and_ty(cx: ctxt, ty: t, fldr: |r: Region|, fldt: |t: t|)
                           -> t {
    ty_fold::RegionFolder::general(cx,
                                   |r| { fldr(r); r },
                                   |t| { fldt(t); t }).fold_ty(ty)
}

pub fn fold_regions(cx: ctxt, ty: t, fldr: |r: Region| -> Region) -> t {
    ty_fold::RegionFolder::regions(cx, fldr).fold_ty(ty)
}

// Substitute *only* type parameters.  Used in trans where regions are erased.
pub fn subst_tps(tcx: ctxt, tps: &[t], self_ty_opt: Option<t>, typ: t) -> t {
    let mut subst = TpsSubst { tcx: tcx, self_ty_opt: self_ty_opt, tps: tps };
    return subst.fold_ty(typ);

    struct TpsSubst<'a> {
        tcx: ctxt,
        self_ty_opt: Option<t>,
        tps: &'a [t],
    }

    impl<'a> TypeFolder for TpsSubst<'a> {
        fn tcx(&self) -> ty::ctxt { self.tcx }

        fn fold_ty(&mut self, t: ty::t) -> ty::t {
            if self.tps.len() == 0u && self.self_ty_opt.is_none() {
                return t;
            }

            let tb = ty::get(t);
            if self.self_ty_opt.is_none() && !tbox_has_flag(tb, has_params) {
                return t;
            }

            match ty::get(t).sty {
                ty_param(p) => {
                    self.tps[p.idx]
                }

                ty_self(_) => {
                    match self.self_ty_opt {
                        None => self.tcx.sess.bug("ty_self unexpected here"),
                        Some(self_ty) => self_ty
                    }
                }

                _ => {
                    ty_fold::super_fold_ty(self, t)
                }
            }
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
        ty_self(..) => true,
        _ => false
    }
}

pub fn type_is_structural(ty: t) -> bool {
    match get(ty).sty {
      ty_struct(..) | ty_tup(_) | ty_enum(..) | ty_closure(_) | ty_trait(..) |
      ty_vec(_, vstore_fixed(_)) | ty_str(vstore_fixed(_)) |
      ty_vec(_, vstore_slice(_)) | ty_str(vstore_slice(_))
      => true,
      _ => false
    }
}

pub fn type_is_sequence(ty: t) -> bool {
    match get(ty).sty {
      ty_str(_) | ty_vec(_, _) => true,
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
      ty_str(_) => true,
      _ => false
    }
}

pub fn sequence_element_type(cx: ctxt, ty: t) -> t {
    match get(ty).sty {
      ty_str(_) => return mk_mach_uint(ast::TyU8),
      ty_vec(mt, _) | ty_unboxed_vec(mt) => return mt.ty,
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
      ty_box(_) | ty_vec(_, vstore_box) | ty_str(vstore_box) => true,
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
      ty_vec(_, vstore_slice(_)) | ty_str(vstore_slice(_)) => true,
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
          ty_vec(_, _) | ty_unboxed_vec(_) => true,
          ty_str(_) => true,
          _ => false
        };
}

pub fn type_is_unique(ty: t) -> bool {
    match get(ty).sty {
        ty_uniq(_) | ty_vec(_, vstore_uniq) | ty_str(vstore_uniq) => true,
        _ => false
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
      ty_bare_fn(..) | ty_ptr(_) => true,
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
    {
        let needs_unwind_cleanup_cache = cx.needs_unwind_cleanup_cache
                                           .borrow();
        match needs_unwind_cleanup_cache.get().find(&ty) {
            Some(&result) => return result,
            None => ()
        }
    }

    let mut tycache = HashSet::new();
    let needs_unwind_cleanup =
        type_needs_unwind_cleanup_(cx, ty, &mut tycache, false);
    let mut needs_unwind_cleanup_cache = cx.needs_unwind_cleanup_cache
                                           .borrow_mut();
    needs_unwind_cleanup_cache.get().insert(ty, needs_unwind_cleanup);
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
    maybe_walk_ty(ty, |ty| {
        let old_encountered_box = encountered_box;
        let result = match get(ty).sty {
          ty_box(_) => {
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
          ty_str(vstore_uniq) |
          ty_str(vstore_box) |
          ty_vec(_, vstore_uniq) |
          ty_vec(_, vstore_box)
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
    });

    return needs_unwind_cleanup;
}

/**
 * Type contents is how the type checker reasons about kinds.
 * They track what kinds of things are found within a type.  You can
 * think of them as kind of an "anti-kind".  They track the kinds of values
 * and thinks that are contained in types.  Having a larger contents for
 * a type tends to rule that type *out* from various kinds.  For example,
 * a type that contains a reference is not sendable.
 *
 * The reason we compute type contents and not kinds is that it is
 * easier for me (nmatsakis) to think about what is contained within
 * a type than to think about what is *not* contained within a type.
 */
pub struct TypeContents {
    bits: u64
}

macro_rules! def_type_content_sets(
    (mod $mname:ident { $($name:ident = $bits:expr),+ }) => {
        mod $mname {
            use middle::ty::TypeContents;
            $(pub static $name: TypeContents = TypeContents { bits: $bits };)+
        }
    }
)

def_type_content_sets!(
    mod TC {
        None                                = 0b0000__00000000__0000,

        // Things that are interior to the value (first nibble):
        InteriorUnsized                     = 0b0000__00000000__0001,
        // InteriorAll                         = 0b0000__00000000__1111,

        // Things that are owned by the value (second and third nibbles):
        OwnsOwned                           = 0b0000__00000001__0000,
        OwnsDtor                            = 0b0000__00000010__0000,
        OwnsManaged /* see [1] below */     = 0b0000__00000100__0000,
        OwnsAffine                          = 0b0000__00001000__0000,
        OwnsAll                             = 0b0000__11111111__0000,

        // Things that are reachable by the value in any way (fourth nibble):
        ReachesNonsendAnnot                 = 0b0001__00000000__0000,
        ReachesBorrowed                     = 0b0010__00000000__0000,
        // ReachesManaged /* see [1] below */  = 0b0100__00000000__0000,
        ReachesMutable                      = 0b1000__00000000__0000,
        ReachesAll                          = 0b1111__00000000__0000,

        // Things that cause values to *move* rather than *copy*
        Moves                               = 0b0000__00001011__0000,

        // Things that mean drop glue is necessary
        NeedsDrop                           = 0b0000__00000111__0000,

        // Things that prevent values from being sent
        //
        // Note: For checking whether something is sendable, it'd
        //       be sufficient to have ReachesManaged. However, we include
        //       both ReachesManaged and OwnsManaged so that when
        //       a parameter has a bound T:Send, we are able to deduce
        //       that it neither reaches nor owns a managed pointer.
        Nonsendable                         = 0b0111__00000100__0000,

        // Things that prevent values from being considered freezable
        Nonfreezable                        = 0b1000__00000000__0000,

        // Things that prevent values from being considered 'static
        Nonstatic                           = 0b0010__00000000__0000,

        // Things that prevent values from being considered sized
        Nonsized                            = 0b0000__00000000__0001,

        // Things that make values considered not POD (same as `Moves`)
        Nonpod                              = 0b0000__00001111__0000,

        // Bits to set when a managed value is encountered
        //
        // [1] Do not set the bits TC::OwnsManaged or
        //     TC::ReachesManaged directly, instead reference
        //     TC::Managed to set them both at once.
        Managed                             = 0b0100__00000100__0000,

        // All bits
        All                                 = 0b1111__11111111__1111
    }
)

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
            BoundPod => self.is_pod(cx),
        }
    }

    pub fn when(&self, cond: bool) -> TypeContents {
        if cond {*self} else {TC::None}
    }

    pub fn intersects(&self, tc: TypeContents) -> bool {
        (self.bits & tc.bits) != 0
    }

    pub fn is_static(&self, _: ctxt) -> bool {
        !self.intersects(TC::Nonstatic)
    }

    pub fn is_sendable(&self, _: ctxt) -> bool {
        !self.intersects(TC::Nonsendable)
    }

    pub fn owns_managed(&self) -> bool {
        self.intersects(TC::OwnsManaged)
    }

    pub fn is_freezable(&self, _: ctxt) -> bool {
        !self.intersects(TC::Nonfreezable)
    }

    pub fn is_sized(&self, _: ctxt) -> bool {
        !self.intersects(TC::Nonsized)
    }

    pub fn is_pod(&self, _: ctxt) -> bool {
        !self.intersects(TC::Nonpod)
    }

    pub fn moves_by_default(&self, _: ctxt) -> bool {
        self.intersects(TC::Moves)
    }

    pub fn needs_drop(&self, _: ctxt) -> bool {
        self.intersects(TC::NeedsDrop)
    }

    pub fn owned_pointer(&self) -> TypeContents {
        /*!
         * Includes only those bits that still apply
         * when indirected through a `~` pointer
         */
        TC::OwnsOwned | (
            *self & (TC::OwnsAll | TC::ReachesAll))
    }

    pub fn reference(&self, bits: TypeContents) -> TypeContents {
        /*!
         * Includes only those bits that still apply
         * when indirected through a reference (`&`)
         */
        bits | (
            *self & TC::ReachesAll)
    }

    pub fn managed_pointer(&self) -> TypeContents {
        /*!
         * Includes only those bits that still apply
         * when indirected through a managed pointer (`@`)
         */
        TC::Managed | (
            *self & TC::ReachesAll)
    }

    pub fn unsafe_pointer(&self) -> TypeContents {
        /*!
         * Includes only those bits that still apply
         * when indirected through an unsafe pointer (`*`)
         */
        *self & TC::ReachesAll
    }

    pub fn union<T>(v: &[T], f: |&T| -> TypeContents) -> TypeContents {
        v.iter().fold(TC::None, |tc, t| tc | f(t))
    }

    pub fn inverse(&self) -> TypeContents {
        TypeContents { bits: !self.bits }
    }
}

impl ops::BitOr<TypeContents,TypeContents> for TypeContents {
    fn bitor(&self, other: &TypeContents) -> TypeContents {
        TypeContents {bits: self.bits | other.bits}
    }
}

impl ops::BitAnd<TypeContents,TypeContents> for TypeContents {
    fn bitand(&self, other: &TypeContents) -> TypeContents {
        TypeContents {bits: self.bits & other.bits}
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

    {
        let tc_cache = cx.tc_cache.borrow();
        match tc_cache.get().find(&ty_id) {
            Some(tc) => { return *tc; }
            None => {}
        }
    }

    let mut cache = HashMap::new();
    let result = tc_ty(cx, ty, &mut cache);

    let mut tc_cache = cx.tc_cache.borrow_mut();
    tc_cache.get().insert(ty_id, result);
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
        // to List, we temporarily use TC::None as its contents.  Later we'll
        // patch up the cache with the correct value, once we've computed it
        // (this is basically a co-inductive process, if that helps).  So in
        // the end we'll compute TC::OwnsOwned, in this case.
        //
        // The problem is, as we are doing the computation, we will also
        // compute an *intermediate* contents for, e.g., Option<List> of
        // TC::None.  This is ok during the computation of List itself, but if
        // we stored this intermediate value into cx.tc_cache, then later
        // requests for the contents of Option<List> would also yield TC::None
        // which is incorrect.  This value was computed based on the crutch
        // value for the type contents of list.  The correct value is
        // TC::OwnsOwned.  This manifested as issue #4821.
        let ty_id = type_id(ty);
        match cache.find(&ty_id) {
            Some(tc) => { return *tc; }
            None => {}
        }
        {
            let tc_cache = cx.tc_cache.borrow();
            match tc_cache.get().find(&ty_id) {    // Must check both caches!
                Some(tc) => { return *tc; }
                None => {}
            }
        }
        cache.insert(ty_id, TC::None);

        let result = match get(ty).sty {
            // Scalar and unique types are sendable, freezable, and durable
            ty_nil | ty_bot | ty_bool | ty_int(_) | ty_uint(_) | ty_float(_) |
            ty_bare_fn(_) | ty::ty_char => {
                TC::None
            }

            ty_str(vstore_uniq) => {
                TC::OwnsOwned
            }

            ty_closure(ref c) => {
                closure_contents(cx, c)
            }

            ty_box(typ) => {
                tc_ty(cx, typ, cache).managed_pointer()
            }

            ty_uniq(typ) => {
                tc_ty(cx, typ, cache).owned_pointer()
            }

            ty_trait(_, _, store, mutbl, bounds) => {
                object_contents(cx, store, mutbl, bounds)
            }

            ty_ptr(ref mt) => {
                tc_ty(cx, mt.ty, cache).unsafe_pointer()
            }

            ty_rptr(r, ref mt) => {
                tc_ty(cx, mt.ty, cache).reference(
                    borrowed_contents(r, mt.mutbl))
            }

            ty_vec(mt, vstore_uniq) => {
                tc_mt(cx, mt, cache).owned_pointer()
            }

            ty_vec(mt, vstore_box) => {
                tc_mt(cx, mt, cache).managed_pointer()
            }

            ty_vec(ref mt, vstore_slice(r)) => {
                tc_ty(cx, mt.ty, cache).reference(
                    borrowed_contents(r, mt.mutbl))
            }

            ty_vec(mt, vstore_fixed(_)) => {
                tc_mt(cx, mt, cache)
            }

            ty_str(vstore_box) => {
                TC::Managed
            }

            ty_str(vstore_slice(r)) => {
                borrowed_contents(r, ast::MutImmutable)
            }

            ty_str(vstore_fixed(_)) => {
                TC::None
            }

            ty_struct(did, ref substs) => {
                let flds = struct_fields(cx, did, substs);
                let mut res =
                    TypeContents::union(flds, |f| tc_mt(cx, f.mt, cache));
                if ty::has_dtor(cx, did) {
                    res = res | TC::OwnsDtor;
                }
                apply_attributes(cx, did, res)
            }

            ty_tup(ref tys) => {
                TypeContents::union(*tys, |ty| tc_ty(cx, *ty, cache))
            }

            ty_enum(did, ref substs) => {
                let variants = substd_enum_variants(cx, did, substs);
                let res =
                    TypeContents::union(variants, |variant| {
                        TypeContents::union(variant.args, |arg_ty| {
                            tc_ty(cx, *arg_ty, cache)
                        })
                    });
                apply_attributes(cx, did, res)
            }

            ty_param(p) => {
                // We only ever ask for the kind of types that are defined in
                // the current crate; therefore, the only type parameters that
                // could be in scope are those defined in the current crate.
                // If this assertion failures, it is likely because of a
                // failure in the cross-crate inlining code to translate a
                // def-id.
                assert_eq!(p.def_id.crate, ast::LOCAL_CRATE);

                let ty_param_defs = cx.ty_param_defs.borrow();
                let tp_def = ty_param_defs.get().get(&p.def_id.node);
                kind_bounds_to_contents(cx,
                                        tp_def.bounds.builtin_bounds,
                                        tp_def.bounds.trait_bounds)
            }

            ty_self(def_id) => {
                // FIXME(#4678)---self should just be a ty param

                // Self may be bounded if the associated trait has builtin kinds
                // for supertraits. If so we can use those bounds.
                let trait_def = lookup_trait_def(cx, def_id);
                let traits = [trait_def.trait_ref];
                kind_bounds_to_contents(cx, trait_def.bounds, traits)
            }

            ty_infer(_) => {
                // This occurs during coherence, but shouldn't occur at other
                // times.
                TC::All
            }
            ty_unboxed_vec(mt) => TC::InteriorUnsized | tc_mt(cx, mt, cache),

            ty_type => TC::None,

            ty_err => {
                cx.sess.bug("Asked to compute contents of error type");
            }
        };

        cache.insert(ty_id, result);
        return result;
    }

    fn tc_mt(cx: ctxt,
             mt: mt,
             cache: &mut HashMap<uint, TypeContents>) -> TypeContents
    {
        let mc = TC::ReachesMutable.when(mt.mutbl == MutMutable);
        mc | tc_ty(cx, mt.ty, cache)
    }

    fn apply_attributes(cx: ctxt,
                        did: ast::DefId,
                        tc: TypeContents)
                        -> TypeContents {
        tc |
            TC::ReachesMutable.when(has_attr(cx, did, "no_freeze")) |
            TC::ReachesNonsendAnnot.when(has_attr(cx, did, "no_send"))
    }

    fn borrowed_contents(region: ty::Region,
                         mutbl: ast::Mutability)
                         -> TypeContents {
        /*!
         * Type contents due to containing a reference
         * with the region `region` and borrow kind `bk`
         */

        let b = match mutbl {
            ast::MutMutable => TC::ReachesMutable | TC::OwnsAffine,
            ast::MutImmutable => TC::None,
        };
        b | (TC::ReachesBorrowed).when(region != ty::ReStatic)
    }

    fn closure_contents(cx: ctxt, cty: &ClosureTy) -> TypeContents {
        // Closure contents are just like trait contents, but with potentially
        // even more stuff.
        let st = match cty.sigil {
            ast::BorrowedSigil =>
                object_contents(cx, RegionTraitStore(cty.region), MutMutable, cty.bounds),
            ast::ManagedSigil =>
                object_contents(cx, BoxTraitStore, MutImmutable, cty.bounds),
            ast::OwnedSigil =>
                object_contents(cx, UniqTraitStore, MutImmutable, cty.bounds),
        };

        // FIXME(#3569): This borrowed_contents call should be taken care of in
        // object_contents, after ~Traits and @Traits can have region bounds too.
        // This one here is redundant for &fns but important for ~fns and @fns.
        let rt = borrowed_contents(cty.region, ast::MutImmutable);

        // This also prohibits "@once fn" from being copied, which allows it to
        // be called. Neither way really makes much sense.
        let ot = match cty.onceness {
            ast::Once => TC::OwnsAffine,
            ast::Many => TC::None,
        };

        st | rt | ot
    }

    fn object_contents(cx: ctxt,
                       store: TraitStore,
                       mutbl: ast::Mutability,
                       bounds: BuiltinBounds)
                       -> TypeContents {
        // These are the type contents of the (opaque) interior
        let contents = (TC::ReachesMutable.when(mutbl == ast::MutMutable) |
                        kind_bounds_to_contents(cx, bounds, []));

        match store {
            UniqTraitStore => {
                contents.owned_pointer()
            }
            BoxTraitStore => {
                contents.managed_pointer()
            }
            RegionTraitStore(r) => {
                contents.reference(borrowed_contents(r, mutbl))
            }
        }
    }

    fn kind_bounds_to_contents(cx: ctxt,
                               bounds: BuiltinBounds,
                               traits: &[@TraitRef])
                               -> TypeContents {
        let _i = indenter();
        let mut tc = TC::All;
        each_inherited_builtin_bound(cx, bounds, traits, |bound| {
            tc = tc - match bound {
                BoundStatic => TC::Nonstatic,
                BoundSend => TC::Nonsendable,
                BoundFreeze => TC::Nonfreezable,
                BoundSized => TC::Nonsized,
                BoundPod => TC::Nonpod,
            };
        });
        return tc;

        // Iterates over all builtin bounds on the type parameter def, including
        // those inherited from traits with builtin-kind-supertraits.
        fn each_inherited_builtin_bound(cx: ctxt,
                                        bounds: BuiltinBounds,
                                        traits: &[@TraitRef],
                                        f: |BuiltinBound|) {
            for bound in bounds.iter() {
                f(bound);
            }

            each_bound_trait_and_supertraits(cx, traits, |trait_ref| {
                let trait_def = lookup_trait_def(cx, trait_ref.def_id);
                for bound in trait_def.bounds.iter() {
                    f(bound);
                }
                true
            });
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
        debug!("type_requires({}, {})?",
               ::util::ppaux::ty_to_str(cx, r_ty),
               ::util::ppaux::ty_to_str(cx, ty));

        let r = {
            get(r_ty).sty == get(ty).sty ||
                subtypes_require(cx, seen, r_ty, ty)
        };

        debug!("type_requires({}, {})? {}",
               ::util::ppaux::ty_to_str(cx, r_ty),
               ::util::ppaux::ty_to_str(cx, ty),
               r);
        return r;
    }

    fn subtypes_require(cx: ctxt, seen: &mut ~[DefId],
                        r_ty: t, ty: t) -> bool {
        debug!("subtypes_require({}, {})?",
               ::util::ppaux::ty_to_str(cx, r_ty),
               ::util::ppaux::ty_to_str(cx, ty));

        let r = match get(ty).sty {
            // fixed length vectors need special treatment compared to
            // normal vectors, since they don't necessarily have the
            // possibilty to have length zero.
            ty_vec(_, vstore_fixed(0)) => false, // don't need no contents
            ty_vec(mt, vstore_fixed(_)) => type_requires(cx, seen, r_ty, mt.ty),

            ty_nil |
            ty_bot |
            ty_bool |
            ty_char |
            ty_int(_) |
            ty_uint(_) |
            ty_float(_) |
            ty_str(_) |
            ty_bare_fn(_) |
            ty_closure(_) |
            ty_infer(_) |
            ty_err |
            ty_param(_) |
            ty_self(_) |
            ty_type |
            ty_vec(_, _) |
            ty_unboxed_vec(_) => {
                false
            }
            ty_box(typ) | ty_uniq(typ) => {
                type_requires(cx, seen, r_ty, typ)
            }
            ty_rptr(_, ref mt) => {
                type_requires(cx, seen, r_ty, mt.ty)
            }

            ty_ptr(..) => {
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
                seen.pop().unwrap();
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
                let r = !vs.is_empty() && vs.iter().all(|variant| {
                    variant.args.iter().any(|aty| {
                        let sty = subst(cx, substs, *aty);
                        type_requires(cx, seen, r_ty, sty)
                    })
                });
                seen.pop().unwrap();
                r
            }
        };

        debug!("subtypes_require({}, {})? {}",
               ::util::ppaux::ty_to_str(cx, r_ty),
               ::util::ppaux::ty_to_str(cx, ty),
               r);

        return r;
    }

    let mut seen = ~[];
    !subtypes_require(cx, &mut seen, r_ty, r_ty)
}

pub fn type_structurally_contains(cx: ctxt, ty: t, test: |x: &sty| -> bool)
                                  -> bool {
    let sty = &get(ty).sty;
    debug!("type_structurally_contains: {}",
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
      ty_vec(ref mt, vstore_fixed(_)) => {
        return type_structurally_contains(cx, mt.ty, test);
      }
      _ => return false
    }
}

pub fn type_structurally_contains_uniques(cx: ctxt, ty: t) -> bool {
    return type_structurally_contains(cx, ty, |sty| {
        match *sty {
          ty_uniq(_) |
          ty_vec(_, vstore_uniq) |
          ty_str(vstore_uniq) => true,
          _ => false,
        }
    });
}

pub fn type_is_trait(ty: t) -> bool {
    match get(ty).sty {
        ty_trait(..) => true,
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

pub fn type_is_bare_fn(ty: t) -> bool {
    match get(ty).sty {
        ty_bare_fn(..) => true,
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
        ty_int(ast::TyI) | ty_uint(ast::TyU) => false,
        ty_int(..) | ty_uint(..) | ty_float(..) => true,
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
      ty_str(vstore_uniq) | ty_str(vstore_box) |
      ty_vec(_, vstore_uniq) | ty_vec(_, vstore_box) |
      ty_trait(_, _, _, _, _) | ty_rptr(_,_) => result = false,
      // Structural types
      ty_enum(did, ref substs) => {
        let variants = enum_variants(cx, did);
        for variant in (*variants).iter() {
            // FIXME(pcwalton): This is an inefficient way to do this. Don't
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
      ty_str(vstore_fixed(_)) => result = true,
      ty_vec(ref mt, vstore_fixed(_)) | ty_unboxed_vec(ref mt) => {
        result = type_is_pod(cx, mt.ty);
      }
      ty_param(_) => result = false,
      ty_struct(did, ref substs) => {
        let fields = lookup_struct_fields(cx, did);
        result = fields.iter().all(|f| {
            let fty = ty::lookup_item_type(cx, f.id);
            let sty = subst(cx, substs, fty.ty);
            type_is_pod(cx, sty)
        });
      }

      ty_str(vstore_slice(..)) | ty_vec(_, vstore_slice(..)) => {
        result = false;
      }

      ty_infer(..) | ty_self(..) | ty_err => {
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
            let ty_param_defs = cx.ty_param_defs.borrow();
            let param_def = ty_param_defs.get().get(&p.def_id.node);
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
pub fn deref(t: t, explicit: bool) -> Option<mt> {
    deref_sty(&get(t).sty, explicit)
}

pub fn deref_sty(sty: &sty, explicit: bool) -> Option<mt> {
    match *sty {
        ty_box(typ) | ty_uniq(typ) => {
            Some(mt {
                ty: typ,
                mutbl: ast::MutImmutable,
            })
        }

        ty_rptr(_, mt) => {
            Some(mt)
        }

        ty_ptr(mt) if explicit => {
            Some(mt)
        }

        _ => None
    }
}

pub fn type_autoderef(t: t) -> t {
    let mut t = t;
    loop {
        match deref(t, false) {
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
      ty_vec(mt, _) => Some(mt),
      ty_str(_) => Some(mt {ty: mk_u8(), mutbl: ast::MutImmutable}),
      _ => None
    }
}

pub fn node_id_to_trait_ref(cx: ctxt, id: ast::NodeId) -> @ty::TraitRef {
    let trait_refs = cx.trait_refs.borrow();
    match trait_refs.get().find(&id) {
       Some(&t) => t,
       None => cx.sess.bug(
           format!("node_id_to_trait_ref: no trait ref for node `{}`",
                ast_map::node_id_to_str(cx.items, id,
                                        token::get_ident_interner())))
    }
}

pub fn node_id_to_type(cx: ctxt, id: ast::NodeId) -> t {
    //printfln!("{:?}/{:?}", id, cx.node_types.len());
    let node_types = cx.node_types.borrow();
    match node_types.get().find(&(id as uint)) {
       Some(&t) => t,
       None => cx.sess.bug(
           format!("node_id_to_type: no type for node `{}`",
                ast_map::node_id_to_str(cx.items, id,
                                        token::get_ident_interner())))
    }
}

// FIXME(pcwalton): Makes a copy, bleh. Probably better to not do that.
pub fn node_id_to_type_params(cx: ctxt, id: ast::NodeId) -> ~[t] {
    let node_type_substs = cx.node_type_substs.borrow();
    match node_type_substs.get().find(&id) {
      None => return ~[],
      Some(ts) => return (*ts).clone(),
    }
}

fn node_id_has_type_params(cx: ctxt, id: ast::NodeId) -> bool {
    let node_type_substs = cx.node_type_substs.borrow();
    node_type_substs.get().contains_key(&id)
}

pub fn fn_is_variadic(fty: t) -> bool {
    match get(fty).sty {
        ty_bare_fn(ref f) => f.sig.variadic,
        ty_closure(ref f) => f.sig.variadic,
        ref s => {
            fail!("fn_is_variadic() called on non-fn type: {:?}", s)
        }
    }
}

pub fn ty_fn_sig(fty: t) -> FnSig {
    match get(fty).sty {
        ty_bare_fn(ref f) => f.sig.clone(),
        ty_closure(ref f) => f.sig.clone(),
        ref s => {
            fail!("ty_fn_sig() called on non-fn type: {:?}", s)
        }
    }
}

// Type accessors for substructures of types
pub fn ty_fn_args(fty: t) -> ~[t] {
    match get(fty).sty {
        ty_bare_fn(ref f) => f.sig.inputs.clone(),
        ty_closure(ref f) => f.sig.inputs.clone(),
        ref s => {
            fail!("ty_fn_args() called on non-fn type: {:?}", s)
        }
    }
}

pub fn ty_closure_sigil(fty: t) -> Sigil {
    match get(fty).sty {
        ty_closure(ref f) => f.sigil,
        ref s => {
            fail!("ty_closure_sigil() called on non-closure type: {:?}", s)
        }
    }
}

pub fn ty_fn_purity(fty: t) -> ast::Purity {
    match get(fty).sty {
        ty_bare_fn(ref f) => f.purity,
        ty_closure(ref f) => f.purity,
        ref s => {
            fail!("ty_fn_purity() called on non-fn type: {:?}", s)
        }
    }
}

pub fn ty_fn_ret(fty: t) -> t {
    match get(fty).sty {
        ty_bare_fn(ref f) => f.sig.output,
        ty_closure(ref f) => f.sig.output,
        ref s => {
            fail!("ty_fn_ret() called on non-fn type: {:?}", s)
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
        ty_vec(_, vstore) => vstore,
        ty_str(vstore) => vstore,
        ref s => fail!("ty_vstore() called on invalid sty: {:?}", s)
    }
}

pub fn ty_region(tcx: ctxt,
                 span: Span,
                 ty: t) -> Region {
    match get(ty).sty {
        ty_rptr(r, _) => r,
        ty_vec(_, vstore_slice(r)) => r,
        ty_str(vstore_slice(r)) => r,
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
    let adjustment = {
        let adjustments = cx.adjustments.borrow();
        adjustments.get().find_copy(&expr.id)
    };
    adjust_ty(cx, expr.span, unadjusted_ty, adjustment)
}

pub fn adjust_ty(cx: ctxt,
                 span: Span,
                 unadjusted_ty: ty::t,
                 adjustment: Option<@AutoAdjustment>)
                 -> ty::t {
    /*! See `expr_ty_adjusted` */

    return match adjustment {
        None => unadjusted_ty,

        Some(adjustment) => {
            match *adjustment {
                AutoAddEnv(r, s) => {
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
                                format!("add_env adjustment on non-bare-fn: \
                                         {:?}",
                                        b));
                        }
                    }
                }

                AutoDerefRef(ref adj) => {
                    let mut adjusted_ty = unadjusted_ty;

                    if !ty::type_is_error(adjusted_ty) {
                        for i in range(0, adj.autoderefs) {
                            match ty::deref(adjusted_ty, true) {
                                Some(mt) => { adjusted_ty = mt.ty; }
                                None => {
                                    cx.sess.span_bug(
                                        span,
                                        format!("The {}th autoderef failed: \
                                                {}",
                                                i,
                                                ty_to_str(cx, adjusted_ty)));
                                }
                            }
                        }
                    }

                    match adj.autoref {
                        None => adjusted_ty,
                        Some(ref autoref) => {
                            match *autoref {
                                AutoPtr(r, m) => {
                                    mk_rptr(cx, r, mt {
                                        ty: adjusted_ty,
                                        mutbl: m
                                    })
                                }

                                AutoBorrowVec(r, m) => {
                                    borrow_vec(cx, span, r, m, adjusted_ty)
                                }

                                AutoBorrowVecRef(r, m) => {
                                    adjusted_ty = borrow_vec(cx,
                                                             span,
                                                             r,
                                                             m,
                                                             adjusted_ty);
                                    mk_rptr(cx, r, mt {
                                        ty: adjusted_ty,
                                        mutbl: ast::MutImmutable
                                    })
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

                AutoObject(ref sigil, ref region, m, b, def_id, ref substs) => {
                    trait_adjustment_to_ty(cx,
                                           sigil,
                                           region,
                                           def_id,
                                           substs,
                                           m,
                                           b)
                }
            }
        }
    };

    fn borrow_vec(cx: ctxt, span: Span,
                  r: Region, m: ast::Mutability,
                  ty: ty::t) -> ty::t {
        match get(ty).sty {
            ty_vec(mt, _) => {
                ty::mk_vec(cx, mt {ty: mt.ty, mutbl: m}, vstore_slice(r))
            }

            ty_str(_) => {
                ty::mk_str(cx, vstore_slice(r))
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

pub fn trait_adjustment_to_ty(cx: ctxt, sigil: &ast::Sigil, region: &Option<Region>,
                              def_id: ast::DefId, substs: &substs, m: ast::Mutability,
                              bounds: BuiltinBounds) -> t {

    let trait_store = match *sigil {
        BorrowedSigil => RegionTraitStore(region.expect("expected valid region")),
        OwnedSigil => UniqTraitStore,
        ManagedSigil => BoxTraitStore
    };

    mk_trait(cx, def_id, substs.clone(), trait_store, m, bounds)
}

impl AutoRef {
    pub fn map_region(&self, f: |Region| -> Region) -> AutoRef {
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
    let method_map = method_map.borrow();
    method_map.get().find(&id).map(|method| {
        match method.origin {
          typeck::method_static(did) => {
            // n.b.: When we encode impl methods, the bounds
            // that we encode include both the impl bounds
            // and then the method bounds themselves...
            ty::lookup_item_type(tcx, did).generics.type_param_defs
          }
          typeck::method_param(typeck::method_param {
              trait_id: trt_id,
              method_num: n_mth, ..}) |
          typeck::method_object(typeck::method_object {
              trait_id: trt_id,
              method_num: n_mth, ..}) => {
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
    })
}

pub fn resolve_expr(tcx: ctxt, expr: &ast::Expr) -> ast::Def {
    let def_map = tcx.def_map.borrow();
    match def_map.get().find(&expr.id) {
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
    {
        let method_map = method_map.borrow();
        if method_map.get().contains_key(&expr.id) {
            // Overloaded operations are generally calls, and hence they are
            // generated via DPS.  However, assign_op (e.g., `x += y`) is an
            // exception, as its result is always unit.
            return match expr.node {
                ast::ExprAssignOp(..) => RvalueStmtExpr,
                _ => RvalueDpsExpr
            };
        }
    }

    match expr.node {
        ast::ExprPath(..) => {
            match resolve_expr(tcx, expr) {
                ast::DefVariant(tid, vid, _) => {
                    let variant_info = enum_variant_with_id(tcx, tid, vid);
                    if variant_info.args.len() > 0u {
                        // N-ary variant.
                        RvalueDatumExpr
                    } else {
                        // Nullary variant.
                        RvalueDpsExpr
                    }
                }

                ast::DefStruct(_) => {
                    match get(expr_ty(tcx, expr)).sty {
                        ty_bare_fn(..) => RvalueDatumExpr,
                        _ => RvalueDpsExpr
                    }
                }

                // Fn pointers are just scalar values.
                ast::DefFn(..) | ast::DefStaticMethod(..) => RvalueDatumExpr,

                // Note: there is actually a good case to be made that
                // DefArg's, particularly those of immediate type, ought to
                // considered rvalues.
                ast::DefStatic(..) |
                ast::DefBinding(..) |
                ast::DefUpvar(..) |
                ast::DefArg(..) |
                ast::DefLocal(..) => LvalueExpr,

                def => {
                    tcx.sess.span_bug(expr.span, format!(
                        "Uncategorized def for expr {:?}: {:?}",
                        expr.id, def));
                }
            }
        }

        ast::ExprUnary(_, ast::UnDeref, _) |
        ast::ExprField(..) |
        ast::ExprIndex(..) => {
            LvalueExpr
        }

        ast::ExprCall(..) |
        ast::ExprMethodCall(..) |
        ast::ExprStruct(..) |
        ast::ExprTup(..) |
        ast::ExprIf(..) |
        ast::ExprMatch(..) |
        ast::ExprFnBlock(..) |
        ast::ExprProc(..) |
        ast::ExprDoBody(..) |
        ast::ExprBlock(..) |
        ast::ExprRepeat(..) |
        ast::ExprVstore(_, ast::ExprVstoreSlice) |
        ast::ExprVstore(_, ast::ExprVstoreMutSlice) |
        ast::ExprVec(..) => {
            RvalueDpsExpr
        }

        ast::ExprLit(lit) if lit_is_str(lit) => {
            RvalueDpsExpr
        }

        ast::ExprCast(..) => {
            let node_types = tcx.node_types.borrow();
            match node_types.get().find(&(expr.id as uint)) {
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

        ast::ExprBreak(..) |
        ast::ExprAgain(..) |
        ast::ExprRet(..) |
        ast::ExprWhile(..) |
        ast::ExprLoop(..) |
        ast::ExprAssign(..) |
        ast::ExprInlineAsm(..) |
        ast::ExprAssignOp(..) => {
            RvalueStmtExpr
        }

        ast::ExprForLoop(..) => fail!("non-desugared expr_for_loop"),

        ast::ExprLogLevel |
        ast::ExprLit(_) | // Note: LitStr is carved out above
        ast::ExprUnary(..) |
        ast::ExprAddrOf(..) |
        ast::ExprBinary(..) |
        ast::ExprVstore(_, ast::ExprVstoreBox) |
        ast::ExprVstore(_, ast::ExprVstoreUniq) => {
            RvalueDatumExpr
        }

        ast::ExprBox(place, _) => {
            // Special case `~T` for now:
            let def_map = tcx.def_map.borrow();
            let definition = match def_map.get().find(&place.id) {
                Some(&def) => def,
                None => fail!("no def for place"),
            };
            let def_id = ast_util::def_id_of_def(definition);
            match tcx.lang_items.items[ExchangeHeapLangItem as uint] {
                Some(item_def_id) if def_id == item_def_id => RvalueDatumExpr,
                Some(_) | None => RvalueDpsExpr,
            }
        }

        ast::ExprParen(e) => expr_kind(tcx, method_map, e),

        ast::ExprMac(..) => {
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
      ast::StmtMac(..) => fail!("unexpanded macro in trans")
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
    walk_ty(ty, |ty| {
        match get(ty).sty {
          ty_param(p) => {
            rslt.push(p);
          }
          _ => ()
        }
    });
    rslt
}

pub fn occurs_check(tcx: ctxt, sp: Span, vid: TyVid, rt: t) {
    // Returns a vec of all the type variables occurring in `ty`. It may
    // contain duplicates.  (Integral type vars aren't counted.)
    fn vars_in_type(ty: t) -> ~[TyVid] {
        let mut rslt = ~[];
        walk_ty(ty, |ty| {
            match get(ty).sty {
              ty_infer(TyVar(v)) => rslt.push(v),
              _ => ()
            }
        });
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
        ty_uint(_) | ty_float(_) | ty_str(_) | ty_type => {
            ::util::ppaux::ty_to_str(cx, t)
        }

        ty_enum(id, _) => format!("enum {}", item_path_str(cx, id)),
        ty_box(_) => ~"@-ptr",
        ty_uniq(_) => ~"~-ptr",
        ty_vec(_, _) => ~"vector",
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
        terr_regions_does_not_outlive(..) => {
            format!("lifetime mismatch")
        }
        terr_regions_not_same(..) => {
            format!("lifetimes are not the same")
        }
        terr_regions_no_overlap(..) => {
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
            format!("{} storage differs: expected `{}` but found `{}`",
                 terr_vstore_kind_to_str(k),
                 vstore_to_str(cx, (*values).expected),
                 vstore_to_str(cx, (*values).found))
        }
        terr_trait_stores_differ(_, ref values) => {
            format!("trait storage differs: expected `{}` but found `{}`",
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
            format!("expected trait `{}` but found trait `{}`",
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
        terr_variadic_mismatch(ref values) => {
            format!("expected {} fn but found {} function",
                    if values.expected { "variadic" } else { "non-variadic" },
                    if values.found { "variadic" } else { "non-variadic" })
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
    let provided_method_sources = cx.provided_method_sources.borrow();
    provided_method_sources.get().find(&id).map(|x| *x)
}

pub fn provided_trait_methods(cx: ctxt, id: ast::DefId) -> ~[@Method] {
    if is_local(id) {
        {
            match cx.items.find(id.node) {
                Some(ast_map::NodeItem(item, _)) => {
                    match item.node {
                        ItemTrait(_, _, ref ms) => {
                            let (_, p) = ast_util::split_trait_methods(*ms);
                            p.map(|m| method(cx, ast_util::local_def(m.id)))
                        }
                        _ => {
                            cx.sess.bug(format!("provided_trait_methods: \
                                                 {:?} is not a trait",
                                                id))
                        }
                    }
                }
                _ => {
                    cx.sess.bug(format!("provided_trait_methods: {:?} is not \
                                         a trait",
                                        id))
                }
            }
        }
    } else {
        csearch::get_provided_trait_methods(cx, id)
    }
}

pub fn trait_supertraits(cx: ctxt, id: ast::DefId) -> @~[@TraitRef] {
    // Check the cache.
    {
        let supertraits = cx.supertraits.borrow();
        match supertraits.get().find(&id) {
            Some(&trait_refs) => { return trait_refs; }
            None => {}  // Continue.
        }
    }

    // Not in the cache. It had better be in the metadata, which means it
    // shouldn't be local.
    assert!(!is_local(id));

    // Get the supertraits out of the metadata and create the
    // TraitRef for each.
    let result = @csearch::get_supertraits(cx, id);
    let mut supertraits = cx.supertraits.borrow_mut();
    supertraits.get().insert(id, result);
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
                                    load_external: || -> V) -> V {
    /*!
     * Helper for looking things up in the various maps
     * that are populated during typeck::collect (e.g.,
     * `cx.methods`, `cx.tcache`, etc).  All of these share
     * the pattern that if the id is local, it should have
     * been loaded into the map by the `typeck::collect` phase.
     * If the def-id is external, then we have to go consult
     * the crate loading code (and cache the result for the future).
     */

    match map.find_copy(&def_id) {
        Some(v) => { return v; }
        None => { }
    }

    if def_id.crate == ast::LOCAL_CRATE {
        fail!("No def'n found for {:?} in tcx.{}", def_id, descr);
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
    let mut trait_methods_cache = cx.trait_methods_cache.borrow_mut();
    match trait_methods_cache.get().find(&trait_did) {
        Some(&methods) => methods,
        None => {
            let def_ids = ty::trait_method_def_ids(cx, trait_did);
            let methods = @def_ids.map(|d| ty::method(cx, *d));
            trait_methods_cache.get().insert(trait_did, methods);
            methods
        }
    }
}

pub fn method(cx: ctxt, id: ast::DefId) -> @Method {
    let mut methods = cx.methods.borrow_mut();
    lookup_locally_or_in_crate_store("methods", id, methods.get(), || {
        @csearch::get_method(cx, id)
    })
}

pub fn trait_method_def_ids(cx: ctxt, id: ast::DefId) -> @~[DefId] {
    let mut trait_method_def_ids = cx.trait_method_def_ids.borrow_mut();
    lookup_locally_or_in_crate_store("trait_method_def_ids",
                                     id,
                                     trait_method_def_ids.get(),
                                     || {
        @csearch::get_trait_method_def_ids(cx.cstore, id)
    })
}

pub fn impl_trait_ref(cx: ctxt, id: ast::DefId) -> Option<@TraitRef> {
    {
        let mut impl_trait_cache = cx.impl_trait_cache.borrow_mut();
        match impl_trait_cache.get().find(&id) {
            Some(&ret) => { return ret; }
            None => {}
        }
    }

    let ret = if id.crate == ast::LOCAL_CRATE {
        debug!("(impl_trait_ref) searching for trait impl {:?}", id);
        {
            match cx.items.find(id.node) {
                Some(ast_map::NodeItem(item, _)) => {
                    match item.node {
                        ast::ItemImpl(_, ref opt_trait, _, _) => {
                            match opt_trait {
                                &Some(ref t) => {
                                    Some(ty::node_id_to_trait_ref(cx,
                                                                  t.ref_id))
                                }
                                &None => None
                            }
                        }
                        _ => None
                    }
                }
                _ => None
            }
        }
    } else {
        csearch::get_impl_trait(cx, id)
    };

    let mut impl_trait_cache = cx.impl_trait_cache.borrow_mut();
    impl_trait_cache.get().insert(id, ret);
    return ret;
}

pub fn trait_ref_to_def_id(tcx: ctxt, tr: &ast::TraitRef) -> ast::DefId {
    let def_map = tcx.def_map.borrow();
    let def = def_map.get()
                     .find(&tr.ref_id)
                     .expect("no def-map entry for trait");
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

// Enum information
#[deriving(Clone)]
pub struct VariantInfo {
    args: ~[t],
    arg_names: Option<~[ast::Ident]>,
    ctor_ty: t,
    name: ast::Ident,
    id: ast::DefId,
    disr_val: Disr,
    vis: Visibility
}

impl VariantInfo {

    /// Creates a new VariantInfo from the corresponding ast representation.
    ///
    /// Does not do any caching of the value in the type context.
    pub fn from_ast_variant(cx: ctxt,
                            ast_variant: &ast::Variant,
                            discriminant: Disr) -> VariantInfo {
        let ctor_ty = node_id_to_type(cx, ast_variant.node.id);

        match ast_variant.node.kind {
            ast::TupleVariantKind(ref args) => {
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
            ast::StructVariantKind(ref struct_def) => {

                let fields: &[StructField] = struct_def.fields;

                assert!(fields.len() > 0);

                let arg_tys = ty_fn_args(ctor_ty).map(|a| *a);
                let arg_names = fields.map(|field| {
                    match field.node.kind {
                        NamedField(ident, _) => ident,
                        UnnamedField => cx.sess.bug(
                            "enum_variants: all fields in struct must have a name")
                    }
                });

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
    enum_variants(cx, id).iter().map(|variant_info| {
        let substd_args = variant_info.args.iter()
            .map(|aty| subst(cx, substs, *aty)).collect();

        let substd_ctor_ty = subst(cx, substs, variant_info.ctor_ty);

        @VariantInfo {
            args: substd_args,
            ctor_ty: substd_ctor_ty,
            ..(**variant_info).clone()
        }
    }).collect()
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
    let destructor_for_type = cx.destructor_for_type.borrow();
    match destructor_for_type.get().find(&struct_id) {
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

pub fn item_path(cx: ctxt, id: ast::DefId) -> ast_map::Path {
    if id.crate != ast::LOCAL_CRATE {
        return csearch::get_item_path(cx, id)
    }

    // FIXME (#5521): uncomment this code and don't have a catch-all at the
    //                end of the match statement. Favor explicitly listing
    //                each variant.
    // let node = cx.items.get(&id.node);
    // match *node {
    match cx.items.get(id.node) {
        ast_map::NodeItem(item, path) => {
            let item_elt = match item.node {
                ItemMod(_) | ItemForeignMod(_) => {
                    ast_map::PathMod(item.ident)
                }
                _ => ast_map::PathName(item.ident)
            };
            vec::append_one((*path).clone(), item_elt)
        }

        ast_map::NodeForeignItem(nitem, _, _, path) => {
            vec::append_one((*path).clone(),
                            ast_map::PathName(nitem.ident))
        }

        ast_map::NodeMethod(method, _, path) => {
            vec::append_one((*path).clone(),
                            ast_map::PathName(method.ident))
        }
        ast_map::NodeTraitMethod(trait_method, _, path) => {
            let method = ast_util::trait_method_to_ty_method(&*trait_method);
            vec::append_one((*path).clone(),
                            ast_map::PathName(method.ident))
        }

        ast_map::NodeVariant(ref variant, _, path) => {
            vec::append_one(path.init().to_owned(),
                            ast_map::PathName((*variant).node.name))
        }

        ast_map::NodeStructCtor(_, item, path) => {
            vec::append_one((*path).clone(), ast_map::PathName(item.ident))
        }

        ref node => {
            cx.sess.bug(format!("cannot find item_path for node {:?}", node));
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
    {
        let enum_var_cache = cx.enum_var_cache.borrow();
        match enum_var_cache.get().find(&id) {
            Some(&variants) => return variants,
            _ => { /* fallthrough */ }
        }
    }

    let result = if ast::LOCAL_CRATE != id.crate {
        @csearch::get_enum_variants(cx, id)
    } else {
        /*
          Although both this code and check_enum_variants in typeck/check
          call eval_const_expr, it should never get called twice for the same
          expr, since check_enum_variants also updates the enum_var_cache
         */
        {
            match cx.items.get(id.node) {
              ast_map::NodeItem(item, _) => {
                  match item.node {
                    ast::ItemEnum(ref enum_definition, _) => {
                        let mut last_discriminant: Option<Disr> = None;
                        @enum_definition.variants.iter().map(|&variant| {

                            let mut discriminant = match last_discriminant {
                                Some(val) => val + 1,
                                None => INITIAL_DISCRIMINANT_VALUE
                            };

                            match variant.node.disr_expr {
                                Some(e) => match const_eval::eval_const_expr_partial(&cx, e) {
                                    Ok(const_eval::const_int(val)) => {
                                        discriminant = val as Disr
                                    }
                                    Ok(const_eval::const_uint(val)) => {
                                        discriminant = val as Disr
                                    }
                                    Ok(_) => {
                                        cx.sess
                                          .span_err(e.span,
                                                    "expected signed integer \
                                                     constant");
                                    }
                                    Err(ref err) => {
                                        cx.sess
                                          .span_err(e.span,
                                                    format!("expected \
                                                             constant: {}",
                                                            (*err)));
                                    }
                                },
                                None => {}
                            };

                            let variant_info =
                                @VariantInfo::from_ast_variant(cx,
                                                               variant,
                                                               discriminant);
                            last_discriminant = Some(discriminant);
                            variant_info

                        }).collect()
                    }
                    _ => {
                        cx.sess.bug("enum_variants: id not bound to an enum")
                    }
                  }
              }
              _ => cx.sess.bug("enum_variants: id not bound to an enum")
            }
        }
    };

    {
        let mut enum_var_cache = cx.enum_var_cache.borrow_mut();
        enum_var_cache.get().insert(id, result);
        result
    }
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
    let mut tcache = cx.tcache.borrow_mut();
    lookup_locally_or_in_crate_store(
        "tcache", did, tcache.get(),
        || csearch::get_type(cx, did))
}

pub fn lookup_impl_vtables(cx: ctxt,
                           did: ast::DefId)
                     -> typeck::impl_res {
    let mut impl_vtables = cx.impl_vtables.borrow_mut();
    lookup_locally_or_in_crate_store(
        "impl_vtables", did, impl_vtables.get(),
        || csearch::get_impl_vtables(cx, did) )
}

/// Given the did of a trait, returns its canonical trait ref.
pub fn lookup_trait_def(cx: ctxt, did: ast::DefId) -> @ty::TraitDef {
    let mut trait_defs = cx.trait_defs.borrow_mut();
    match trait_defs.get().find(&did) {
        Some(&trait_def) => {
            // The item is in this crate. The caller should have added it to the
            // type cache already
            return trait_def;
        }
        None => {
            assert!(did.crate != ast::LOCAL_CRATE);
            let trait_def = @csearch::get_trait_def(cx, did);
            trait_defs.get().insert(did, trait_def);
            return trait_def;
        }
    }
}

/// Iterate over meta_items of a definition.
// (This should really be an iterator, but that would require csearch and
// decoder to use iterators instead of higher-order functions.)
pub fn each_attr(tcx: ctxt, did: DefId, f: |@MetaItem| -> bool) -> bool {
    if is_local(did) {
        {
            match tcx.items.find(did.node) {
                Some(ast_map::NodeItem(item, _)) => {
                    item.attrs.iter().advance(|attr| f(attr.node.value))
                }
                _ => tcx.sess.bug(format!("has_attr: {:?} is not an item",
                                          did))
            }
        }
    } else {
        let mut cont = true;
        csearch::get_item_attrs(tcx.cstore, did, |meta_items| {
            if cont {
                cont = meta_items.iter().advance(|ptrptr| f(*ptrptr));
            }
        });
        return cont;
    }
}

/// Determine whether an item is annotated with an attribute
pub fn has_attr(tcx: ctxt, did: DefId, attr: &str) -> bool {
    let mut found = false;
    each_attr(tcx, did, |item| {
        if attr == item.name() {
            found = true;
            false
        } else {
            true
        }
    });
    return found;
}

/// Determine whether an item is annotated with `#[packed]`
pub fn lookup_packed(tcx: ctxt, did: DefId) -> bool {
    has_attr(tcx, did, "packed")
}

/// Determine whether an item is annotated with `#[simd]`
pub fn lookup_simd(tcx: ctxt, did: DefId) -> bool {
    has_attr(tcx, did, "simd")
}

// Obtain the the representation annotation for a definition.
pub fn lookup_repr_hint(tcx: ctxt, did: DefId) -> attr::ReprAttr {
    let mut acc = attr::ReprAny;
    ty::each_attr(tcx, did, |meta| {
        acc = attr::find_repr_attr(tcx.sess.diagnostic(), meta, acc);
        true
    });
    return acc;
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
    } else {
        {
            let mut tcache = tcx.tcache.borrow_mut();
            match tcache.get().find(&id) {
               Some(&ty_param_bounds_and_ty {ty, ..}) => ty,
               None => {
                   let tpt = csearch::get_field_type(tcx, struct_id, id);
                   tcache.get().insert(id, tpt);
                   tpt.ty
               }
            }
        }
    };
    subst(tcx, substs, t)
}

// Look up the list of field names and IDs for a given struct
// Fails if the id is not bound to a struct.
pub fn lookup_struct_fields(cx: ctxt, did: ast::DefId) -> ~[field_ty] {
  if did.crate == ast::LOCAL_CRATE {
      {
          match cx.items.find(did.node) {
           Some(ast_map::NodeItem(i,_)) => {
             match i.node {
                ast::ItemStruct(struct_def, _) => {
                   struct_field_tys(struct_def.fields)
                }
                _ => cx.sess.bug("struct ID bound to non-struct")
             }
           }
           Some(ast_map::NodeVariant(ref variant, _, _)) => {
              match (*variant).node.kind {
                ast::StructVariantKind(struct_def) => {
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
  } else {
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

fn struct_field_tys(fields: &[StructField]) -> ~[field_ty] {
    fields.map(|field| {
        match field.node.kind {
            NamedField(ident, visibility) => {
                field_ty {
                    name: ident.name,
                    id: ast_util::local_def(field.node.id),
                    vis: visibility,
                }
            }
            UnnamedField => {
                field_ty {
                    name:
                        syntax::parse::token::special_idents::unnamed_field.name,
                    id: ast_util::local_def(field.node.id),
                    vis: ast::Public,
                }
            }
        }
    })
}

// Returns a list of fields corresponding to the struct's items. trans uses
// this. Takes a list of substs with which to instantiate field types.
pub fn struct_fields(cx: ctxt, did: ast::DefId, substs: &substs)
                     -> ~[field] {
    lookup_struct_fields(cx, did).map(|f| {
       field {
            // FIXME #6993: change type of field to Name and get rid of new()
            ident: ast::Ident::new(f.name),
            mt: mt {
                ty: lookup_field_type(cx, did, f.id, substs),
                mutbl: MutImmutable
            }
        }
    })
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
    let u = TypeNormalizer(cx).fold_ty(t);
    return u;

    struct TypeNormalizer(ctxt);

    impl TypeFolder for TypeNormalizer {
        fn tcx(&self) -> ty::ctxt { let TypeNormalizer(c) = *self; c }

        fn fold_ty(&mut self, t: ty::t) -> ty::t {
            let normalized_opt = {
                let normalized_cache = self.tcx().normalized_cache.borrow();
                normalized_cache.get().find_copy(&t)
            };
            match normalized_opt {
                Some(u) => {
                    return u;
                }
                None => {
                    let t_norm = ty_fold::super_fold_ty(self, t);
                    let mut normalized_cache = self.tcx()
                                                   .normalized_cache
                                                   .borrow_mut();
                    normalized_cache.get().insert(t, t_norm);
                    return t_norm;
                }
            }
        }

        fn fold_vstore(&mut self, vstore: vstore) -> vstore {
            match vstore {
                vstore_fixed(..) | vstore_uniq | vstore_box => vstore,
                vstore_slice(_) => vstore_slice(ReStatic)
            }
        }

        fn fold_region(&mut self, _: ty::Region) -> ty::Region {
            ty::ReStatic
        }

        fn fold_substs(&mut self,
                       substs: &substs)
                       -> substs {
            substs { regions: ErasedRegions,
                     self_ty: ty_fold::fold_opt_ty(self, substs.self_ty),
                     tps: ty_fold::fold_ty_vec(self, substs.tps) }
        }

        fn fold_sig(&mut self,
                    sig: &ty::FnSig)
                    -> ty::FnSig {
            // The binder-id is only relevant to bound regions, which
            // are erased at trans time.
            ty::FnSig { binder_id: ast::DUMMY_NODE_ID,
                        inputs: ty_fold::fold_ty_vec(self, sig.inputs),
                        output: self.fold_ty(sig.output),
                        variadic: sig.variadic }
        }
    }
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
        const_eval::const_binary(_) => {
            tcx.ty_ctxt().sess.span_err(count_expr.span,
                                        "expected positive integer for \
                                         repeat count but found binary array");
            return 0;
        }
      },
      Err(..) => {
        tcx.ty_ctxt().sess.span_err(count_expr.span,
                                    "expected constant integer for repeat count \
                                     but found variable");
        return 0;
      }
    }
}

// Determine what purity to check a nested function under
pub fn determine_inherited_purity(parent: (ast::Purity, ast::NodeId),
                                  child: (ast::Purity, ast::NodeId),
                                  child_sigil: ast::Sigil)
                                    -> (ast::Purity, ast::NodeId) {
    // If the closure is a stack closure and hasn't had some non-standard
    // purity inferred for it, then check it under its parent's purity.
    // Otherwise, use its own
    match child_sigil {
        ast::BorrowedSigil if child.first() == ast::ImpureFn => parent,
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
                                        f: |@TraitRef| -> bool)
                                        -> bool {
    for &bound_trait_ref in bounds.iter() {
        let mut supertrait_set = HashMap::new();
        let mut trait_refs = ~[];
        let mut i = 0;

        // Seed the worklist with the trait from the bound
        supertrait_set.insert(bound_trait_ref.def_id, ());
        trait_refs.push(bound_trait_ref);

        // Add the given trait ty to the hash map
        while i < trait_refs.len() {
            debug!("each_bound_trait_and_supertraits(i={:?}, trait_ref={})",
                   i, trait_refs[i].repr(tcx));

            if !f(trait_refs[i]) {
                return false;
            }

            // Add supertraits to supertrait_set
            let supertrait_refs = trait_ref_supertraits(tcx, trait_refs[i]);
            for &supertrait_ref in supertrait_refs.iter() {
                debug!("each_bound_trait_and_supertraits(supertrait_ref={})",
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
        each_bound_trait_and_supertraits(
            tcx, type_param_def.bounds.trait_bounds, |_| {
            total += 1;
            true
        });
    }
    return total;
}

pub fn get_tydesc_ty(tcx: ctxt) -> Result<t, ~str> {
    tcx.lang_items.require(TyDescStructLangItem).map(|tydesc_lang_item| {
        let intrinsic_defs = tcx.intrinsic_defs.borrow();
        intrinsic_defs.get().find_copy(&tydesc_lang_item)
            .expect("Failed to resolve TyDesc")
    })
}

pub fn get_opaque_ty(tcx: ctxt) -> Result<t, ~str> {
    tcx.lang_items.require(OpaqueStructLangItem).map(|opaque_lang_item| {
        let intrinsic_defs = tcx.intrinsic_defs.borrow();
        intrinsic_defs.get().find_copy(&opaque_lang_item)
            .expect("Failed to resolve Opaque")
    })
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

pub fn item_variances(tcx: ctxt, item_id: ast::DefId) -> @ItemVariances {
    let mut item_variance_map = tcx.item_variance_map.borrow_mut();
    lookup_locally_or_in_crate_store(
        "item_variance_map", item_id, item_variance_map.get(),
        || @csearch::get_item_variances(tcx.cstore, item_id))
}

/// Records a trait-to-implementation mapping.
fn record_trait_implementation(tcx: ctxt,
                               trait_def_id: DefId,
                               implementation: @Impl) {
    let implementation_list;
    let mut trait_impls = tcx.trait_impls.borrow_mut();
    match trait_impls.get().find(&trait_def_id) {
        None => {
            implementation_list = @RefCell::new(~[]);
            trait_impls.get().insert(trait_def_id, implementation_list);
        }
        Some(&existing_implementation_list) => {
            implementation_list = existing_implementation_list
        }
    }

    let mut implementation_list = implementation_list.borrow_mut();
    implementation_list.get().push(implementation);
}

/// Populates the type context with all the implementations for the given type
/// if necessary.
pub fn populate_implementations_for_type_if_necessary(tcx: ctxt,
                                                      type_id: ast::DefId) {
    if type_id.crate == LOCAL_CRATE {
        return
    }
    {
        let populated_external_types = tcx.populated_external_types.borrow();
        if populated_external_types.get().contains(&type_id) {
            return
        }
    }

    csearch::each_implementation_for_type(tcx.sess.cstore, type_id,
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
                let mut provided_method_sources =
                    tcx.provided_method_sources.borrow_mut();
                provided_method_sources.get().insert(method.def_id, *source);
            }
        }

        // If this is an inherent implementation, record it.
        if associated_traits.is_none() {
            let implementation_list;
            let mut inherent_impls = tcx.inherent_impls.borrow_mut();
            match inherent_impls.get().find(&type_id) {
                None => {
                    implementation_list = @RefCell::new(~[]);
                    inherent_impls.get().insert(type_id, implementation_list);
                }
                Some(&existing_implementation_list) => {
                    implementation_list = existing_implementation_list;
                }
            }
            {
                let mut implementation_list =
                    implementation_list.borrow_mut();
                implementation_list.get().push(implementation);
            }
        }

        // Store the implementation info.
        let mut impls = tcx.impls.borrow_mut();
        impls.get().insert(implementation_def_id, implementation);
    });

    let mut populated_external_types = tcx.populated_external_types
                                          .borrow_mut();
    populated_external_types.get().insert(type_id);
}

/// Populates the type context with all the implementations for the given
/// trait if necessary.
pub fn populate_implementations_for_trait_if_necessary(
        tcx: ctxt,
        trait_id: ast::DefId) {
    if trait_id.crate == LOCAL_CRATE {
        return
    }
    {
        let populated_external_traits = tcx.populated_external_traits
                                           .borrow();
        if populated_external_traits.get().contains(&trait_id) {
            return
        }
    }

    csearch::each_implementation_for_trait(tcx.sess.cstore, trait_id,
            |implementation_def_id| {
        let implementation = @csearch::get_impl(tcx, implementation_def_id);

        // Record the trait->implementation mapping.
        record_trait_implementation(tcx, trait_id, implementation);

        // For any methods that use a default implementation, add them to
        // the map. This is a bit unfortunate.
        for method in implementation.methods.iter() {
            for source in method.provided_source.iter() {
                let mut provided_method_sources =
                    tcx.provided_method_sources.borrow_mut();
                provided_method_sources.get().insert(method.def_id, *source);
            }
        }

        // Store the implementation info.
        let mut impls = tcx.impls.borrow_mut();
        impls.get().insert(implementation_def_id, implementation);
    });

    let mut populated_external_traits = tcx.populated_external_traits
                                           .borrow_mut();
    populated_external_traits.get().insert(trait_id);
}

/// Given the def_id of an impl, return the def_id of the trait it implements.
/// If it implements no trait, return `None`.
pub fn trait_id_of_impl(tcx: ctxt,
                        def_id: ast::DefId) -> Option<ast::DefId> {
    let node = match tcx.items.find(def_id.node) {
        Some(node) => node,
        None => return None
    };
    match node {
        ast_map::NodeItem(item, _) => {
            match item.node {
                ast::ItemImpl(_, Some(ref trait_ref), _, _) => {
                    Some(node_id_to_trait_ref(tcx, trait_ref.ref_id).def_id)
                }
                _ => None
            }
        }
        _ => None
    }
}

/// If the given def ID describes a method belonging to a trait (either a
/// default method or an implementation of a trait method), return the ID of
/// the trait that the method belongs to. Otherwise, return `None`.
pub fn trait_of_method(tcx: ctxt, def_id: ast::DefId)
                       -> Option<ast::DefId> {
    if def_id.crate != LOCAL_CRATE {
        return csearch::get_trait_of_method(tcx.cstore, def_id, tcx);
    }
    let method;
    {
        let methods = tcx.methods.borrow();
        method = methods.get().find(&def_id).map(|method| *method);
    }
    match method {
        Some(method) => {
            match method.container {
                TraitContainer(def_id) => Some(def_id),
                ImplContainer(def_id) => trait_id_of_impl(tcx, def_id),
            }
        }
        None => None
    }
}

/// If the given def ID describes a method belonging to a trait, (either a
/// default method or an implementation of a trait method), return the ID of
/// the method inside trait definition (this means that if the given def ID
/// is already that of the original trait method, then the return value is
/// the same).
/// Otherwise, return `None`.
pub fn trait_method_of_method(tcx: ctxt,
                              def_id: ast::DefId) -> Option<ast::DefId> {
    let method;
    {
        let methods = tcx.methods.borrow();
        match methods.get().find(&def_id) {
            Some(m) => method = *m,
            None => return None,
        }
    }
    let name = method.ident.name;
    match trait_of_method(tcx, def_id) {
        Some(trait_did) => {
            let trait_methods = ty::trait_methods(tcx, trait_did);
            trait_methods.iter()
                .position(|m| m.ident.name == name)
                .map(|idx| ty::trait_method(tcx, trait_did, idx).def_id)
        }
        None => None
    }
}

/// Creates a hash of the type `t` which will be the same no matter what crate
/// context it's calculated within. This is used by the `type_id` intrinsic.
pub fn hash_crate_independent(tcx: ctxt, t: t, local_hash: @str) -> u64 {
    use std::hash::{SipState, Streaming};

    let mut hash = SipState::new(0, 0);
    let region = |_hash: &mut SipState, r: Region| {
        match r {
            ReStatic => {}

            ReEmpty |
            ReEarlyBound(..) |
            ReLateBound(..) |
            ReFree(..) |
            ReScope(..) |
            ReInfer(..) => {
                tcx.sess.bug("non-static region found when hashing a type")
            }
        }
    };
    let vstore = |hash: &mut SipState, v: vstore| {
        match v {
            vstore_fixed(_) => hash.input([0]),
            vstore_uniq => hash.input([1]),
            vstore_box => hash.input([2]),
            vstore_slice(r) => {
                hash.input([3]);
                region(hash, r);
            }
        }
    };
    let did = |hash: &mut SipState, did: DefId| {
        let h = if ast_util::is_local(did) {
            local_hash
        } else {
            tcx.sess.cstore.get_crate_hash(did.crate)
        };
        hash.input(h.as_bytes());
        iter(hash, &did.node);
    };
    let mt = |hash: &mut SipState, mt: mt| {
        iter(hash, &mt.mutbl);
    };
    fn iter<T: IterBytes>(hash: &mut SipState, t: &T) {
        t.iter_bytes(true, |bytes| { hash.input(bytes); true });
    }
    ty::walk_ty(t, |t| {
        match ty::get(t).sty {
            ty_nil => hash.input([0]),
            ty_bot => hash.input([1]),
            ty_bool => hash.input([2]),
            ty_char => hash.input([3]),
            ty_int(i) => {
                hash.input([4]);
                iter(&mut hash, &i);
            }
            ty_uint(u) => {
                hash.input([5]);
                iter(&mut hash, &u);
            }
            ty_float(f) => {
                hash.input([6]);
                iter(&mut hash, &f);
            }
            ty_str(v) => {
                hash.input([7]);
                vstore(&mut hash, v);
            }
            ty_enum(d, _) => {
                hash.input([8]);
                did(&mut hash, d);
            }
            ty_box(_) => {
                hash.input([9]);
            }
            ty_uniq(_) => {
                hash.input([10]);
            }
            ty_vec(m, v) => {
                hash.input([11]);
                mt(&mut hash, m);
                vstore(&mut hash, v);
            }
            ty_ptr(m) => {
                hash.input([12]);
                mt(&mut hash, m);
            }
            ty_rptr(r, m) => {
                hash.input([13]);
                region(&mut hash, r);
                mt(&mut hash, m);
            }
            ty_bare_fn(ref b) => {
                hash.input([14]);
                iter(&mut hash, &b.purity);
                iter(&mut hash, &b.abis);
            }
            ty_closure(ref c) => {
                hash.input([15]);
                iter(&mut hash, &c.purity);
                iter(&mut hash, &c.sigil);
                iter(&mut hash, &c.onceness);
                iter(&mut hash, &c.bounds);
                region(&mut hash, c.region);
            }
            ty_trait(d, _, store, m, bounds) => {
                hash.input([17]);
                did(&mut hash, d);
                match store {
                    BoxTraitStore => hash.input([0]),
                    UniqTraitStore => hash.input([1]),
                    RegionTraitStore(r) => {
                        hash.input([2]);
                        region(&mut hash, r);
                    }
                }
                iter(&mut hash, &m);
                iter(&mut hash, &bounds);
            }
            ty_struct(d, _) => {
                hash.input([18]);
                did(&mut hash, d);
            }
            ty_tup(ref inner) => {
                hash.input([19]);
                iter(&mut hash, &inner.len());
            }
            ty_param(p) => {
                hash.input([20]);
                iter(&mut hash, &p.idx);
                did(&mut hash, p.def_id);
            }
            ty_self(d) => {
                hash.input([21]);
                did(&mut hash, d);
            }
            ty_infer(_) => unreachable!(),
            ty_err => hash.input([23]),
            ty_type => hash.input([24]),
            ty_unboxed_vec(m) => {
                hash.input([25]);
                mt(&mut hash, m);
            }
        }
    });

    hash.result_u64()
}

impl Variance {
    pub fn to_str(self) -> &'static str {
        match self {
            Covariant => "+",
            Contravariant => "-",
            Invariant => "o",
            Bivariant => "*",
        }
    }
}

pub fn construct_parameter_environment(
    tcx: ctxt,
    self_bound: Option<@TraitRef>,
    item_type_params: &[TypeParameterDef],
    method_type_params: &[TypeParameterDef],
    item_region_params: &[RegionParameterDef],
    free_id: ast::NodeId)
    -> ParameterEnvironment
{
    /*! See `ParameterEnvironment` struct def'n for details */

    //
    // Construct the free substs.
    //

    // map Self => Self
    let self_ty = self_bound.map(|t| ty::mk_self(tcx, t.def_id));

    // map A => A
    let num_item_type_params = item_type_params.len();
    let num_method_type_params = method_type_params.len();
    let num_type_params = num_item_type_params + num_method_type_params;
    let type_params = vec::from_fn(num_type_params, |i| {
            let def_id = if i < num_item_type_params {
                item_type_params[i].def_id
            } else {
                method_type_params[i - num_item_type_params].def_id
            };

            ty::mk_param(tcx, i, def_id)
        });

    // map bound 'a => free 'a
    let region_params = item_region_params.iter().
        map(|r| ty::ReFree(ty::FreeRegion {
                scope_id: free_id,
                bound_region: ty::BrNamed(r.def_id, r.ident)})).
        collect();

    let free_substs = substs {
        self_ty: self_ty,
        tps: type_params,
        regions: ty::NonerasedRegions(region_params)
    };

    //
    // Compute the bounds on Self and the type parameters.
    //

    let self_bound_substd = self_bound.map(|b| b.subst(tcx, &free_substs));
    let type_param_bounds_substd = vec::from_fn(num_type_params, |i| {
        if i < num_item_type_params {
            (*item_type_params[i].bounds).subst(tcx, &free_substs)
        } else {
            let j = i - num_item_type_params;
            (*method_type_params[j].bounds).subst(tcx, &free_substs)
        }
    });

    ty::ParameterEnvironment {
        free_substs: free_substs,
        self_param_bound: self_bound_substd,
        type_param_bounds: type_param_bounds_substd,
    }
}

impl substs {
    pub fn empty() -> substs {
        substs {
            self_ty: None,
            tps: ~[],
            regions: NonerasedRegions(opt_vec::Empty)
        }
    }
}
