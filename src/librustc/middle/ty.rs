// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types)]

use back::svh::Svh;
use driver::session::Session;
use lint;
use metadata::csearch;
use middle::const_eval;
use middle::def;
use middle::dependency_format;
use middle::lang_items::{FnTraitLangItem, FnMutTraitLangItem};
use middle::lang_items::{FnOnceTraitLangItem, OpaqueStructLangItem};
use middle::lang_items::{TyDescStructLangItem, TyVisitorTraitLangItem};
use middle::mem_categorization as mc;
use middle::resolve;
use middle::resolve_lifetime;
use middle::stability;
use middle::subst::{Subst, Substs, VecPerParamSpace};
use middle::subst;
use middle::traits;
use middle::ty;
use middle::typeck;
use middle::ty_fold;
use middle::ty_fold::{TypeFoldable,TypeFolder};
use middle;
use util::ppaux::{note_and_explain_region, bound_region_ptr_to_string};
use util::ppaux::{trait_store_to_string, ty_to_string};
use util::ppaux::{Repr, UserString};
use util::common::{indenter};
use util::nodemap::{NodeMap, NodeSet, DefIdMap, DefIdSet, FnvHashMap};

use std::cell::{Cell, RefCell};
use std::cmp;
use std::fmt::Show;
use std::fmt;
use std::hash::{Hash, sip, Writer};
use std::iter::AdditiveIterator;
use std::mem;
use std::ops;
use std::rc::Rc;
use std::collections::{HashMap, HashSet};
use std::collections::hashmap::{Occupied, Vacant};
use arena::TypedArena;
use syntax::abi;
use syntax::ast::{CrateNum, DefId, FnStyle, Ident, ItemTrait, LOCAL_CRATE};
use syntax::ast::{MutImmutable, MutMutable, Name, NamedField, NodeId};
use syntax::ast::{Onceness, StmtExpr, StmtSemi, StructField, UnnamedField};
use syntax::ast::{Visibility};
use syntax::ast_util::{PostExpansionMethod, is_local, lit_is_str};
use syntax::ast_util;
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::parse::token::InternedString;
use syntax::{ast, ast_map};
use syntax::util::small_vector::SmallVector;
use std::collections::enum_set::{EnumSet, CLike};

pub type Disr = u64;

pub static INITIAL_DISCRIMINANT_VALUE: Disr = 0;

// Data types

#[deriving(PartialEq, Eq, Hash)]
pub struct field {
    pub ident: ast::Ident,
    pub mt: mt
}

#[deriving(Clone)]
pub enum ImplOrTraitItemContainer {
    TraitContainer(ast::DefId),
    ImplContainer(ast::DefId),
}

impl ImplOrTraitItemContainer {
    pub fn id(&self) -> ast::DefId {
        match *self {
            TraitContainer(id) => id,
            ImplContainer(id) => id,
        }
    }
}

#[deriving(Clone)]
pub enum ImplOrTraitItem {
    MethodTraitItem(Rc<Method>),
    TypeTraitItem(Rc<AssociatedType>),
}

impl ImplOrTraitItem {
    fn id(&self) -> ImplOrTraitItemId {
        match *self {
            MethodTraitItem(ref method) => MethodTraitItemId(method.def_id),
            TypeTraitItem(ref associated_type) => {
                TypeTraitItemId(associated_type.def_id)
            }
        }
    }

    pub fn def_id(&self) -> ast::DefId {
        match *self {
            MethodTraitItem(ref method) => method.def_id,
            TypeTraitItem(ref associated_type) => associated_type.def_id,
        }
    }

    pub fn ident(&self) -> ast::Ident {
        match *self {
            MethodTraitItem(ref method) => method.ident,
            TypeTraitItem(ref associated_type) => associated_type.ident,
        }
    }

    pub fn container(&self) -> ImplOrTraitItemContainer {
        match *self {
            MethodTraitItem(ref method) => method.container,
            TypeTraitItem(ref associated_type) => associated_type.container,
        }
    }
}

#[deriving(Clone)]
pub enum ImplOrTraitItemId {
    MethodTraitItemId(ast::DefId),
    TypeTraitItemId(ast::DefId),
}

impl ImplOrTraitItemId {
    pub fn def_id(&self) -> ast::DefId {
        match *self {
            MethodTraitItemId(def_id) => def_id,
            TypeTraitItemId(def_id) => def_id,
        }
    }
}

#[deriving(Clone)]
pub struct Method {
    pub ident: ast::Ident,
    pub generics: ty::Generics,
    pub fty: BareFnTy,
    pub explicit_self: ExplicitSelfCategory,
    pub vis: ast::Visibility,
    pub def_id: ast::DefId,
    pub container: ImplOrTraitItemContainer,

    // If this method is provided, we need to know where it came from
    pub provided_source: Option<ast::DefId>
}

impl Method {
    pub fn new(ident: ast::Ident,
               generics: ty::Generics,
               fty: BareFnTy,
               explicit_self: ExplicitSelfCategory,
               vis: ast::Visibility,
               def_id: ast::DefId,
               container: ImplOrTraitItemContainer,
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

#[deriving(Clone)]
pub struct AssociatedType {
    pub ident: ast::Ident,
    pub vis: ast::Visibility,
    pub def_id: ast::DefId,
    pub container: ImplOrTraitItemContainer,
}

#[deriving(Clone, PartialEq, Eq, Hash, Show)]
pub struct mt {
    pub ty: t,
    pub mutbl: ast::Mutability,
}

#[deriving(Clone, PartialEq, Eq, Hash, Encodable, Decodable, Show)]
pub enum TraitStore {
    /// Box<Trait>
    UniqTraitStore,
    /// &Trait and &mut Trait
    RegionTraitStore(Region, ast::Mutability),
}

#[deriving(Clone, Show)]
pub struct field_ty {
    pub name: Name,
    pub id: DefId,
    pub vis: ast::Visibility,
    pub origin: ast::DefId,  // The DefId of the struct in which the field is declared.
}

// Contains information needed to resolve types and (in the future) look up
// the types of AST nodes.
#[deriving(PartialEq, Eq, Hash)]
pub struct creader_cache_key {
    pub cnum: CrateNum,
    pub pos: uint,
    pub len: uint
}

pub type creader_cache = RefCell<HashMap<creader_cache_key, t>>;

pub struct intern_key {
    sty: *const sty,
}

// NB: Do not replace this with #[deriving(PartialEq)]. The automatically-derived
// implementation will not recurse through sty and you will get stack
// exhaustion.
impl cmp::PartialEq for intern_key {
    fn eq(&self, other: &intern_key) -> bool {
        unsafe {
            *self.sty == *other.sty
        }
    }
    fn ne(&self, other: &intern_key) -> bool {
        !self.eq(other)
    }
}

impl Eq for intern_key {}

impl<W:Writer> Hash<W> for intern_key {
    fn hash(&self, s: &mut W) {
        unsafe { (*self.sty).hash(s) }
    }
}

pub enum ast_ty_to_ty_cache_entry {
    atttce_unresolved,  /* not resolved yet */
    atttce_resolved(t)  /* resolved to a type, irrespective of region */
}

#[deriving(Clone, PartialEq, Decodable, Encodable)]
pub struct ItemVariances {
    pub types: VecPerParamSpace<Variance>,
    pub regions: VecPerParamSpace<Variance>,
}

#[deriving(Clone, PartialEq, Decodable, Encodable, Show)]
pub enum Variance {
    Covariant,      // T<A> <: T<B> iff A <: B -- e.g., function return type
    Invariant,      // T<A> <: T<B> iff B == A -- e.g., type of mutable cell
    Contravariant,  // T<A> <: T<B> iff B <: A -- e.g., function param type
    Bivariant,      // T<A> <: T<B>            -- e.g., unused type parameter
}

#[deriving(Clone)]
pub enum AutoAdjustment {
    AdjustAddEnv(ty::TraitStore),
    AdjustDerefRef(AutoDerefRef)
}

#[deriving(Clone, PartialEq)]
pub enum UnsizeKind {
    // [T, ..n] -> [T], the uint field is n.
    UnsizeLength(uint),
    // An unsize coercion applied to the tail field of a struct.
    // The uint is the index of the type parameter which is unsized.
    UnsizeStruct(Box<UnsizeKind>, uint),
    UnsizeVtable(TyTrait, /* the self type of the trait */ ty::t)
}

#[deriving(Clone)]
pub struct AutoDerefRef {
    pub autoderefs: uint,
    pub autoref: Option<AutoRef>
}

#[deriving(Clone, PartialEq)]
pub enum AutoRef {
    /// Convert from T to &T
    /// The third field allows us to wrap other AutoRef adjustments.
    AutoPtr(Region, ast::Mutability, Option<Box<AutoRef>>),

    /// Convert [T, ..n] to [T] (or similar, depending on the kind)
    AutoUnsize(UnsizeKind),

    /// Convert Box<[T, ..n]> to Box<[T]> or something similar in a Box.
    /// With DST and Box a library type, this should be replaced by UnsizeStruct.
    AutoUnsizeUniq(UnsizeKind),

    /// Convert from T to *T
    /// Value to thin pointer
    /// The second field allows us to wrap other AutoRef adjustments.
    AutoUnsafe(ast::Mutability, Option<Box<AutoRef>>),
}

// Ugly little helper function. The first bool in the returned tuple is true if
// there is an 'unsize to trait object' adjustment at the bottom of the
// adjustment. If that is surrounded by an AutoPtr, then we also return the
// region of the AutoPtr (in the third argument). The second bool is true if the
// adjustment is unique.
fn autoref_object_region(autoref: &AutoRef) -> (bool, bool, Option<Region>) {
    fn unsize_kind_is_object(k: &UnsizeKind) -> bool {
        match k {
            &UnsizeVtable(..) => true,
            &UnsizeStruct(box ref k, _) => unsize_kind_is_object(k),
            _ => false
        }
    }

    match autoref {
        &AutoUnsize(ref k) => (unsize_kind_is_object(k), false, None),
        &AutoUnsizeUniq(ref k) => (unsize_kind_is_object(k), true, None),
        &AutoPtr(adj_r, _, Some(box ref autoref)) => {
            let (b, u, r) = autoref_object_region(autoref);
            if r.is_some() || u {
                (b, u, r)
            } else {
                (b, u, Some(adj_r))
            }
        }
        &AutoUnsafe(_, Some(box ref autoref)) => autoref_object_region(autoref),
        _ => (false, false, None)
    }
}

// If the adjustment introduces a borrowed reference to a trait object, then
// returns the region of the borrowed reference.
pub fn adjusted_object_region(adj: &AutoAdjustment) -> Option<Region> {
    match adj {
        &AdjustDerefRef(AutoDerefRef{autoref: Some(ref autoref), ..}) => {
            let (b, _, r) = autoref_object_region(autoref);
            if b {
                r
            } else {
                None
            }
        }
        _ => None
    }
}

// Returns true if there is a trait cast at the bottom of the adjustment.
pub fn adjust_is_object(adj: &AutoAdjustment) -> bool {
    match adj {
        &AdjustDerefRef(AutoDerefRef{autoref: Some(ref autoref), ..}) => {
            let (b, _, _) = autoref_object_region(autoref);
            b
        }
        _ => false
    }
}

// If possible, returns the type expected from the given adjustment. This is not
// possible if the adjustment depends on the type of the adjusted expression.
pub fn type_of_adjust(cx: &ctxt, adj: &AutoAdjustment) -> Option<t> {
    fn type_of_autoref(cx: &ctxt, autoref: &AutoRef) -> Option<t> {
        match autoref {
            &AutoUnsize(ref k) => match k {
                &UnsizeVtable(TyTrait { def_id, substs: ref substs, bounds }, _) => {
                    Some(mk_trait(cx, def_id, substs.clone(), bounds))
                }
                _ => None
            },
            &AutoUnsizeUniq(ref k) => match k {
                &UnsizeVtable(TyTrait { def_id, substs: ref substs, bounds }, _) => {
                    Some(mk_uniq(cx, mk_trait(cx, def_id, substs.clone(), bounds)))
                }
                _ => None
            },
            &AutoPtr(r, m, Some(box ref autoref)) => {
                match type_of_autoref(cx, autoref) {
                    Some(t) => Some(mk_rptr(cx, r, mt {mutbl: m, ty: t})),
                    None => None
                }
            }
            &AutoUnsafe(m, Some(box ref autoref)) => {
                match type_of_autoref(cx, autoref) {
                    Some(t) => Some(mk_ptr(cx, mt {mutbl: m, ty: t})),
                    None => None
                }
            }
            _ => None
        }
    }

    match adj {
        &AdjustDerefRef(AutoDerefRef{autoref: Some(ref autoref), ..}) => {
            type_of_autoref(cx, autoref)
        }
        _ => None
    }
}



/// A restriction that certain types must be the same size. The use of
/// `transmute` gives rise to these restrictions.
pub struct TransmuteRestriction {
    /// The span from whence the restriction comes.
    pub span: Span,
    /// The type being transmuted from.
    pub from: t,
    /// The type being transmuted to.
    pub to: t,
    /// NodeIf of the transmute intrinsic.
    pub id: ast::NodeId,
}

/// The data structure to keep track of all the information that typechecker
/// generates so that so that it can be reused and doesn't have to be redone
/// later on.
pub struct ctxt<'tcx> {
    /// The arena that types are allocated from.
    type_arena: &'tcx TypedArena<t_box_>,

    /// Specifically use a speedy hash algorithm for this hash map, it's used
    /// quite often.
    interner: RefCell<FnvHashMap<intern_key, &'tcx t_box_>>,
    pub next_id: Cell<uint>,
    pub sess: Session,
    pub def_map: resolve::DefMap,

    pub named_region_map: resolve_lifetime::NamedRegionMap,

    pub region_maps: middle::region::RegionMaps,

    /// Stores the types for various nodes in the AST.  Note that this table
    /// is not guaranteed to be populated until after typeck.  See
    /// typeck::check::fn_ctxt for details.
    pub node_types: node_type_table,

    /// Stores the type parameters which were substituted to obtain the type
    /// of this node.  This only applies to nodes that refer to entities
    /// parameterized by type parameters, such as generic fns, types, or
    /// other items.
    pub item_substs: RefCell<NodeMap<ItemSubsts>>,

    /// Maps from a trait item to the trait item "descriptor"
    pub impl_or_trait_items: RefCell<DefIdMap<ImplOrTraitItem>>,

    /// Maps from a trait def-id to a list of the def-ids of its trait items
    pub trait_item_def_ids: RefCell<DefIdMap<Rc<Vec<ImplOrTraitItemId>>>>,

    /// A cache for the trait_items() routine
    pub trait_items_cache: RefCell<DefIdMap<Rc<Vec<ImplOrTraitItem>>>>,

    pub impl_trait_cache: RefCell<DefIdMap<Option<Rc<ty::TraitRef>>>>,

    pub trait_refs: RefCell<NodeMap<Rc<TraitRef>>>,
    pub trait_defs: RefCell<DefIdMap<Rc<TraitDef>>>,

    /// Maps from node-id of a trait object cast (like `foo as
    /// Box<Trait>`) to the trait reference.
    pub object_cast_map: typeck::ObjectCastMap,

    pub map: ast_map::Map<'tcx>,
    pub intrinsic_defs: RefCell<DefIdMap<t>>,
    pub freevars: RefCell<FreevarMap>,
    pub tcache: type_cache,
    pub rcache: creader_cache,
    pub short_names_cache: RefCell<HashMap<t, String>>,
    pub needs_unwind_cleanup_cache: RefCell<HashMap<t, bool>>,
    pub tc_cache: RefCell<HashMap<uint, TypeContents>>,
    pub ast_ty_to_ty_cache: RefCell<NodeMap<ast_ty_to_ty_cache_entry>>,
    pub enum_var_cache: RefCell<DefIdMap<Rc<Vec<Rc<VariantInfo>>>>>,
    pub ty_param_defs: RefCell<NodeMap<TypeParameterDef>>,
    pub adjustments: RefCell<NodeMap<AutoAdjustment>>,
    pub normalized_cache: RefCell<HashMap<t, t>>,
    pub lang_items: middle::lang_items::LanguageItems,
    /// A mapping of fake provided method def_ids to the default implementation
    pub provided_method_sources: RefCell<DefIdMap<ast::DefId>>,
    pub superstructs: RefCell<DefIdMap<Option<ast::DefId>>>,
    pub struct_fields: RefCell<DefIdMap<Rc<Vec<field_ty>>>>,

    /// Maps from def-id of a type or region parameter to its
    /// (inferred) variance.
    pub item_variance_map: RefCell<DefIdMap<Rc<ItemVariances>>>,

    /// True if the variance has been computed yet; false otherwise.
    pub variance_computed: Cell<bool>,

    /// A mapping from the def ID of an enum or struct type to the def ID
    /// of the method that implements its destructor. If the type is not
    /// present in this map, it does not have a destructor. This map is
    /// populated during the coherence phase of typechecking.
    pub destructor_for_type: RefCell<DefIdMap<ast::DefId>>,

    /// A method will be in this list if and only if it is a destructor.
    pub destructors: RefCell<DefIdSet>,

    /// Maps a trait onto a list of impls of that trait.
    pub trait_impls: RefCell<DefIdMap<Rc<RefCell<Vec<ast::DefId>>>>>,

    /// Maps a DefId of a type to a list of its inherent impls.
    /// Contains implementations of methods that are inherent to a type.
    /// Methods in these implementations don't need to be exported.
    pub inherent_impls: RefCell<DefIdMap<Rc<Vec<ast::DefId>>>>,

    /// Maps a DefId of an impl to a list of its items.
    /// Note that this contains all of the impls that we know about,
    /// including ones in other crates. It's not clear that this is the best
    /// way to do it.
    pub impl_items: RefCell<DefIdMap<Vec<ImplOrTraitItemId>>>,

    /// Set of used unsafe nodes (functions or blocks). Unsafe nodes not
    /// present in this set can be warned about.
    pub used_unsafe: RefCell<NodeSet>,

    /// Set of nodes which mark locals as mutable which end up getting used at
    /// some point. Local variable definitions not in this set can be warned
    /// about.
    pub used_mut_nodes: RefCell<NodeSet>,

    /// The set of external nominal types whose implementations have been read.
    /// This is used for lazy resolution of methods.
    pub populated_external_types: RefCell<DefIdSet>,

    /// The set of external traits whose implementations have been read. This
    /// is used for lazy resolution of traits.
    pub populated_external_traits: RefCell<DefIdSet>,

    /// Borrows
    pub upvar_borrow_map: RefCell<UpvarBorrowMap>,

    /// These two caches are used by const_eval when decoding external statics
    /// and variants that are found.
    pub extern_const_statics: RefCell<DefIdMap<ast::NodeId>>,
    pub extern_const_variants: RefCell<DefIdMap<ast::NodeId>>,

    pub method_map: typeck::MethodMap,

    pub dependency_formats: RefCell<dependency_format::Dependencies>,

    /// Records the type of each unboxed closure. The def ID is the ID of the
    /// expression defining the unboxed closure.
    pub unboxed_closures: RefCell<DefIdMap<UnboxedClosure>>,

    pub node_lint_levels: RefCell<HashMap<(ast::NodeId, lint::LintId),
                                          lint::LevelSource>>,

    /// The types that must be asserted to be the same size for `transmute`
    /// to be valid. We gather up these restrictions in the intrinsicck pass
    /// and check them in trans.
    pub transmute_restrictions: RefCell<Vec<TransmuteRestriction>>,

    /// Maps any item's def-id to its stability index.
    pub stability: RefCell<stability::Index>,

    /// Maps closures to their capture clauses.
    pub capture_modes: RefCell<CaptureModeMap>,

    /// Maps def IDs to true if and only if they're associated types.
    pub associated_types: RefCell<DefIdMap<bool>>,

    /// Maps def IDs of traits to information about their associated types.
    pub trait_associated_types:
        RefCell<DefIdMap<Rc<Vec<AssociatedTypeInfo>>>>,

    /// Caches the results of trait selection. This cache is used
    /// for things that do not have to do with the parameters in scope.
    pub selection_cache: traits::SelectionCache,

    /// Caches the representation hints for struct definitions.
    pub repr_hint_cache: RefCell<DefIdMap<Rc<Vec<attr::ReprAttr>>>>,
}

pub enum tbox_flag {
    has_params = 1,
    has_self = 2,
    needs_infer = 4,
    has_regions = 8,
    has_ty_err = 16,
    has_ty_bot = 32,

    // a meta-pub flag: subst may be required if the type has parameters, a self
    // type, or references bound regions
    needs_subst = 1 | 2 | 8
}

pub type t_box = &'static t_box_;

#[deriving(Show)]
pub struct t_box_ {
    pub sty: sty,
    pub id: uint,
    pub flags: uint,
}

// To reduce refcounting cost, we're representing types as unsafe pointers
// throughout the compiler. These are simply casted t_box values. Use ty::get
// to cast them back to a box. (Without the cast, compiler performance suffers
// ~15%.) This does mean that a t value relies on the ctxt to keep its box
// alive, and using ty::get is unsafe when the ctxt is no longer alive.
enum t_opaque {}

#[allow(raw_pointer_deriving)]
#[deriving(Clone, PartialEq, Eq, Hash)]
pub struct t { inner: *const t_opaque }

impl fmt::Show for t {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", get(*self))
    }
}

pub fn get(t: t) -> t_box {
    unsafe {
        let t2: t_box = mem::transmute(t);
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
pub fn type_id(t: t) -> uint { get(t).id }

#[deriving(Clone, PartialEq, Eq, Hash, Show)]
pub struct BareFnTy {
    pub fn_style: ast::FnStyle,
    pub abi: abi::Abi,
    pub sig: FnSig,
}

#[deriving(Clone, PartialEq, Eq, Hash, Show)]
pub struct ClosureTy {
    pub fn_style: ast::FnStyle,
    pub onceness: ast::Onceness,
    pub store: TraitStore,
    pub bounds: ExistentialBounds,
    pub sig: FnSig,
    pub abi: abi::Abi,
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
#[deriving(Clone, PartialEq, Eq, Hash)]
pub struct FnSig {
    pub binder_id: ast::NodeId,
    pub inputs: Vec<t>,
    pub output: t,
    pub variadic: bool
}

#[deriving(Clone, PartialEq, Eq, Hash, Show)]
pub struct ParamTy {
    pub space: subst::ParamSpace,
    pub idx: uint,
    pub def_id: DefId
}

/// Representation of regions:
#[deriving(Clone, PartialEq, Eq, Hash, Encodable, Decodable, Show)]
pub enum Region {
    // Region bound in a type or fn declaration which will be
    // substituted 'early' -- that is, at the same time when type
    // parameters are substituted.
    ReEarlyBound(/* param id */ ast::NodeId,
                 subst::ParamSpace,
                 /*index*/ uint,
                 ast::Name),

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

/**
 * Upvars do not get their own node-id. Instead, we use the pair of
 * the original var id (that is, the root variable that is referenced
 * by the upvar) and the id of the closure expression.
 */
#[deriving(Clone, PartialEq, Eq, Hash)]
pub struct UpvarId {
    pub var_id: ast::NodeId,
    pub closure_expr_id: ast::NodeId,
}

#[deriving(Clone, PartialEq, Eq, Hash, Show, Encodable, Decodable)]
pub enum BorrowKind {
    /// Data must be immutable and is aliasable.
    ImmBorrow,

    /// Data must be immutable but not aliasable.  This kind of borrow
    /// cannot currently be expressed by the user and is used only in
    /// implicit closure bindings. It is needed when you the closure
    /// is borrowing or mutating a mutable referent, e.g.:
    ///
    ///    let x: &mut int = ...;
    ///    let y = || *x += 5;
    ///
    /// If we were to try to translate this closure into a more explicit
    /// form, we'd encounter an error with the code as written:
    ///
    ///    struct Env { x: & &mut int }
    ///    let x: &mut int = ...;
    ///    let y = (&mut Env { &x }, fn_ptr);  // Closure is pair of env and fn
    ///    fn fn_ptr(env: &mut Env) { **env.x += 5; }
    ///
    /// This is then illegal because you cannot mutate a `&mut` found
    /// in an aliasable location. To solve, you'd have to translate with
    /// an `&mut` borrow:
    ///
    ///    struct Env { x: & &mut int }
    ///    let x: &mut int = ...;
    ///    let y = (&mut Env { &mut x }, fn_ptr); // changed from &x to &mut x
    ///    fn fn_ptr(env: &mut Env) { **env.x += 5; }
    ///
    /// Now the assignment to `**env.x` is legal, but creating a
    /// mutable pointer to `x` is not because `x` is not mutable. We
    /// could fix this by declaring `x` as `let mut x`. This is ok in
    /// user code, if awkward, but extra weird for closures, since the
    /// borrow is hidden.
    ///
    /// So we introduce a "unique imm" borrow -- the referent is
    /// immutable, but not aliasable. This solves the problem. For
    /// simplicity, we don't give users the way to express this
    /// borrow, it's just used when translating closures.
    UniqueImmBorrow,

    /// Data is mutable and not aliasable.
    MutBorrow
}

/**
 * Information describing the borrowing of an upvar. This is computed
 * during `typeck`, specifically by `regionck`. The general idea is
 * that the compiler analyses treat closures like:
 *
 *     let closure: &'e fn() = || {
 *        x = 1;   // upvar x is assigned to
 *        use(y);  // upvar y is read
 *        foo(&z); // upvar z is borrowed immutably
 *     };
 *
 * as if they were "desugared" to something loosely like:
 *
 *     struct Vars<'x,'y,'z> { x: &'x mut int,
 *                             y: &'y const int,
 *                             z: &'z int }
 *     let closure: &'e fn() = {
 *         fn f(env: &Vars) {
 *             *env.x = 1;
 *             use(*env.y);
 *             foo(env.z);
 *         }
 *         let env: &'e mut Vars<'x,'y,'z> = &mut Vars { x: &'x mut x,
 *                                                       y: &'y const y,
 *                                                       z: &'z z };
 *         (env, f)
 *     };
 *
 * This is basically what happens at runtime. The closure is basically
 * an existentially quantified version of the `(env, f)` pair.
 *
 * This data structure indicates the region and mutability of a single
 * one of the `x...z` borrows.
 *
 * It may not be obvious why each borrowed variable gets its own
 * lifetime (in the desugared version of the example, these are indicated
 * by the lifetime parameters `'x`, `'y`, and `'z` in the `Vars` definition).
 * Each such lifetime must encompass the lifetime `'e` of the closure itself,
 * but need not be identical to it. The reason that this makes sense:
 *
 * - Callers are only permitted to invoke the closure, and hence to
 *   use the pointers, within the lifetime `'e`, so clearly `'e` must
 *   be a sublifetime of `'x...'z`.
 * - The closure creator knows which upvars were borrowed by the closure
 *   and thus `x...z` will be reserved for `'x...'z` respectively.
 * - Through mutation, the borrowed upvars can actually escape
 *   the closure, so sometimes it is necessary for them to be larger
 *   than the closure lifetime itself.
 */
#[deriving(PartialEq, Clone, Encodable, Decodable)]
pub struct UpvarBorrow {
    pub kind: BorrowKind,
    pub region: ty::Region,
}

pub type UpvarBorrowMap = HashMap<UpvarId, UpvarBorrow>;

impl Region {
    pub fn is_bound(&self) -> bool {
        match self {
            &ty::ReEarlyBound(..) => true,
            &ty::ReLateBound(..) => true,
            _ => false
        }
    }
}

#[deriving(Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Encodable, Decodable, Show)]
pub struct FreeRegion {
    pub scope_id: NodeId,
    pub bound_region: BoundRegion
}

#[deriving(Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Encodable, Decodable, Show)]
pub enum BoundRegion {
    /// An anonymous region parameter for a given fn (&T)
    BrAnon(uint),

    /// Named region parameters for functions (a in &'a T)
    ///
    /// The def-id is needed to distinguish free regions in
    /// the event of shadowing.
    BrNamed(ast::DefId, ast::Name),

    /// Fresh bound identifiers created during GLB computations.
    BrFresh(uint),
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
#[deriving(Clone, PartialEq, Eq, Hash, Show)]
pub enum sty {
    ty_nil,
    ty_bot,
    ty_bool,
    ty_char,
    ty_int(ast::IntTy),
    ty_uint(ast::UintTy),
    ty_float(ast::FloatTy),
    /// Substs here, possibly against intuition, *may* contain `ty_param`s.
    /// That is, even after substitution it is possible that there are type
    /// variables. This happens when the `ty_enum` corresponds to an enum
    /// definition and not a concrete use of it. To get the correct `ty_enum`
    /// from the tcx, use the `NodeId` from the `ast::Ty` and look it up in
    /// the `ast_ty_to_ty_cache`. This is probably true for `ty_struct` as
    /// well.`
    ty_enum(DefId, Substs),
    ty_uniq(t),
    ty_str,
    ty_vec(t, Option<uint>), // Second field is length.
    ty_ptr(mt),
    ty_rptr(Region, mt),
    ty_bare_fn(BareFnTy),
    ty_closure(Box<ClosureTy>),
    ty_trait(Box<TyTrait>),
    ty_struct(DefId, Substs),
    ty_unboxed_closure(DefId, Region),
    ty_tup(Vec<t>),

    ty_param(ParamTy), // type parameter
    ty_open(t),  // A deref'ed fat pointer, i.e., a dynamically sized value
                 // and its size. Only ever used in trans. It is not necessary
                 // earlier since we don't need to distinguish a DST with its
                 // size (e.g., in a deref) vs a DST with the size elsewhere (
                 // e.g., in a field).

    ty_infer(InferTy), // something used only during inference/typeck
    ty_err, // Also only used during inference/typeck, to represent
            // the type of an erroneous expression (helps cut down
            // on non-useful type error messages)
}

#[deriving(Clone, PartialEq, Eq, Hash, Show)]
pub struct TyTrait {
    pub def_id: DefId,
    pub substs: Substs,
    pub bounds: ExistentialBounds
}

#[deriving(PartialEq, Eq, Hash, Show)]
pub struct TraitRef {
    pub def_id: DefId,
    pub substs: Substs,
}

#[deriving(Clone, PartialEq)]
pub enum IntVarValue {
    IntType(ast::IntTy),
    UintType(ast::UintTy),
}

#[deriving(Clone, Show)]
pub enum terr_vstore_kind {
    terr_vec,
    terr_str,
    terr_fn,
    terr_trait
}

#[deriving(Clone, Show)]
pub struct expected_found<T> {
    pub expected: T,
    pub found: T
}

// Data structures used in type unification
#[deriving(Clone, Show)]
pub enum type_err {
    terr_mismatch,
    terr_fn_style_mismatch(expected_found<FnStyle>),
    terr_onceness_mismatch(expected_found<Onceness>),
    terr_abi_mismatch(expected_found<abi::Abi>),
    terr_mutability,
    terr_sigil_mismatch(expected_found<TraitStore>),
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
    terr_trait_stores_differ(terr_vstore_kind, expected_found<TraitStore>),
    terr_sorts(expected_found<t>),
    terr_integer_as_char,
    terr_int_mismatch(expected_found<IntVarValue>),
    terr_float_mismatch(expected_found<ast::FloatTy>),
    terr_traits(expected_found<ast::DefId>),
    terr_builtin_bounds(expected_found<BuiltinBounds>),
    terr_variadic_mismatch(expected_found<bool>),
    terr_cyclic_ty,
}

/// Bounds suitable for a named type parameter like `A` in `fn foo<A>`
/// as well as the existential type parameter in an object type.
#[deriving(PartialEq, Eq, Hash, Clone, Show)]
pub struct ParamBounds {
    pub region_bounds: Vec<ty::Region>,
    pub builtin_bounds: BuiltinBounds,
    pub trait_bounds: Vec<Rc<TraitRef>>
}

/// Bounds suitable for an existentially quantified type parameter
/// such as those that appear in object types or closure types. The
/// major difference between this case and `ParamBounds` is that
/// general purpose trait bounds are omitted and there must be
/// *exactly one* region.
#[deriving(PartialEq, Eq, Hash, Clone, Show)]
pub struct ExistentialBounds {
    pub region_bound: ty::Region,
    pub builtin_bounds: BuiltinBounds
}

pub type BuiltinBounds = EnumSet<BuiltinBound>;

#[deriving(Clone, Encodable, PartialEq, Eq, Decodable, Hash, Show)]
#[repr(uint)]
pub enum BuiltinBound {
    BoundSend,
    BoundSized,
    BoundCopy,
    BoundSync,
}

pub fn empty_builtin_bounds() -> BuiltinBounds {
    EnumSet::empty()
}

pub fn all_builtin_bounds() -> BuiltinBounds {
    let mut set = EnumSet::empty();
    set.add(BoundSend);
    set.add(BoundSized);
    set.add(BoundSync);
    set
}

pub fn region_existential_bound(r: ty::Region) -> ExistentialBounds {
    /*!
     * An existential bound that does not implement any traits.
     */

    ty::ExistentialBounds { region_bound: r,
                            builtin_bounds: empty_builtin_bounds() }
}

impl CLike for BuiltinBound {
    fn to_uint(&self) -> uint {
        *self as uint
    }
    fn from_uint(v: uint) -> BuiltinBound {
        unsafe { mem::transmute(v) }
    }
}

#[deriving(Clone, PartialEq, Eq, Hash)]
pub struct TyVid {
    pub index: uint
}

#[deriving(Clone, PartialEq, Eq, Hash)]
pub struct IntVid {
    pub index: uint
}

#[deriving(Clone, PartialEq, Eq, Hash)]
pub struct FloatVid {
    pub index: uint
}

#[deriving(Clone, PartialEq, Eq, Encodable, Decodable, Hash)]
pub struct RegionVid {
    pub index: uint
}

#[deriving(Clone, PartialEq, Eq, Hash)]
pub enum InferTy {
    TyVar(TyVid),
    IntVar(IntVid),
    FloatVar(FloatVid),
    SkolemizedTy(uint),

    // FIXME -- once integral fallback is impl'd, we should remove
    // this type. It's only needed to prevent spurious errors for
    // integers whose type winds up never being constrained.
    SkolemizedIntTy(uint),
}

#[deriving(Clone, Encodable, Decodable, Eq, Hash, Show)]
pub enum InferRegion {
    ReVar(RegionVid),
    ReSkolemized(uint, BoundRegion)
}

impl cmp::PartialEq for InferRegion {
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

impl fmt::Show for TyVid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result{
        write!(f, "<generic #{}>", self.index)
    }
}

impl fmt::Show for IntVid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<generic integer #{}>", self.index)
    }
}

impl fmt::Show for FloatVid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<generic float #{}>", self.index)
    }
}

impl fmt::Show for RegionVid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "'<generic lifetime #{}>", self.index)
    }
}

impl fmt::Show for FnSig {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // grr, without tcx not much we can do.
        write!(f, "(...)")
    }
}

impl fmt::Show for InferTy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            TyVar(ref v) => v.fmt(f),
            IntVar(ref v) => v.fmt(f),
            FloatVar(ref v) => v.fmt(f),
            SkolemizedTy(v) => write!(f, "SkolemizedTy({})", v),
            SkolemizedIntTy(v) => write!(f, "SkolemizedIntTy({})", v),
        }
    }
}

impl fmt::Show for IntVarValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            IntType(ref v) => v.fmt(f),
            UintType(ref v) => v.fmt(f),
        }
    }
}

#[deriving(Clone, Show)]
pub struct TypeParameterDef {
    pub ident: ast::Ident,
    pub def_id: ast::DefId,
    pub space: subst::ParamSpace,
    pub index: uint,
    pub associated_with: Option<ast::DefId>,
    pub bounds: ParamBounds,
    pub default: Option<ty::t>,
}

#[deriving(Encodable, Decodable, Clone, Show)]
pub struct RegionParameterDef {
    pub name: ast::Name,
    pub def_id: ast::DefId,
    pub space: subst::ParamSpace,
    pub index: uint,
    pub bounds: Vec<ty::Region>,
}

/// Information about the type/lifetime parameters associated with an
/// item or method. Analogous to ast::Generics.
#[deriving(Clone, Show)]
pub struct Generics {
    pub types: VecPerParamSpace<TypeParameterDef>,
    pub regions: VecPerParamSpace<RegionParameterDef>,
}

impl Generics {
    pub fn empty() -> Generics {
        Generics { types: VecPerParamSpace::empty(),
                   regions: VecPerParamSpace::empty() }
    }

    pub fn has_type_params(&self, space: subst::ParamSpace) -> bool {
        !self.types.is_empty_in(space)
    }

    pub fn has_region_params(&self, space: subst::ParamSpace) -> bool {
        !self.regions.is_empty_in(space)
    }
}

impl TraitRef {
    pub fn self_ty(&self) -> ty::t {
        self.substs.self_ty().unwrap()
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
    /// parameters in the same way, this only has an effect on regions.
    pub free_substs: Substs,

    /// Bounds on the various type parameters
    pub bounds: VecPerParamSpace<ParamBounds>,

    /// Each type parameter has an implicit region bound that
    /// indicates it must outlive at least the function body (the user
    /// may specify stronger requirements). This field indicates the
    /// region of the callee.
    pub implicit_region_bound: ty::Region,

    /// Obligations that the caller must satisfy. This is basically
    /// the set of bounds on the in-scope type parameters, translated
    /// into Obligations.
    ///
    /// Note: This effectively *duplicates* the `bounds` array for
    /// now.
    pub caller_obligations: VecPerParamSpace<traits::Obligation>,

    /// Caches the results of trait selection. This cache is used
    /// for things that have to do with the parameters in scope.
    pub selection_cache: traits::SelectionCache,
}

impl ParameterEnvironment {
    pub fn for_item(cx: &ctxt, id: NodeId) -> ParameterEnvironment {
        match cx.map.find(id) {
            Some(ast_map::NodeImplItem(ref impl_item)) => {
                match **impl_item {
                    ast::MethodImplItem(ref method) => {
                        let method_def_id = ast_util::local_def(id);
                        match ty::impl_or_trait_item(cx, method_def_id) {
                            MethodTraitItem(ref method_ty) => {
                                let method_generics = &method_ty.generics;
                                construct_parameter_environment(
                                    cx,
                                    method.span,
                                    method_generics,
                                    method.pe_body().id)
                            }
                            TypeTraitItem(_) => {
                                cx.sess
                                  .bug("ParameterEnvironment::from_item(): \
                                        can't create a parameter environment \
                                        for type trait items")
                            }
                        }
                    }
                    ast::TypeImplItem(_) => {
                        cx.sess.bug("ParameterEnvironment::from_item(): \
                                     can't create a parameter environment \
                                     for type impl items")
                    }
                }
            }
            Some(ast_map::NodeTraitItem(trait_method)) => {
                match *trait_method {
                    ast::RequiredMethod(ref required) => {
                        cx.sess.span_bug(required.span,
                                         "ParameterEnvironment::from_item():
                                          can't create a parameter \
                                          environment for required trait \
                                          methods")
                    }
                    ast::ProvidedMethod(ref method) => {
                        let method_def_id = ast_util::local_def(id);
                        match ty::impl_or_trait_item(cx, method_def_id) {
                            MethodTraitItem(ref method_ty) => {
                                let method_generics = &method_ty.generics;
                                construct_parameter_environment(
                                    cx,
                                    method.span,
                                    method_generics,
                                    method.pe_body().id)
                            }
                            TypeTraitItem(_) => {
                                cx.sess
                                  .bug("ParameterEnvironment::from_item(): \
                                        can't create a parameter environment \
                                        for type trait items")
                            }
                        }
                    }
                    ast::TypeTraitItem(_) => {
                        cx.sess.bug("ParameterEnvironment::from_item(): \
                                     can't create a parameter environment \
                                     for type trait items")
                    }
                }
            }
            Some(ast_map::NodeItem(item)) => {
                match item.node {
                    ast::ItemFn(_, _, _, _, ref body) => {
                        // We assume this is a function.
                        let fn_def_id = ast_util::local_def(id);
                        let fn_pty = ty::lookup_item_type(cx, fn_def_id);

                        construct_parameter_environment(cx,
                                                        item.span,
                                                        &fn_pty.generics,
                                                        body.id)
                    }
                    ast::ItemEnum(..) |
                    ast::ItemStruct(..) |
                    ast::ItemImpl(..) |
                    ast::ItemStatic(..) => {
                        let def_id = ast_util::local_def(id);
                        let pty = ty::lookup_item_type(cx, def_id);
                        construct_parameter_environment(cx, item.span,
                                                        &pty.generics, id)
                    }
                    _ => {
                        cx.sess.span_bug(item.span,
                                         "ParameterEnvironment::from_item():
                                          can't create a parameter \
                                          environment for this kind of item")
                    }
                }
            }
            _ => {
                cx.sess.bug(format!("ParameterEnvironment::from_item(): \
                                     `{}` is not an item",
                                    cx.map.node_to_string(id)).as_slice())
            }
        }
    }
}

/// A polytype.
///
/// - `generics`: the set of type parameters and their bounds
/// - `ty`: the base types, which may reference the parameters defined
///   in `generics`
#[deriving(Clone, Show)]
pub struct Polytype {
    pub generics: Generics,
    pub ty: t
}

/// As `Polytype` but for a trait ref.
pub struct TraitDef {
    /// Generic type definitions. Note that `Self` is listed in here
    /// as having a single bound, the trait itself (e.g., in the trait
    /// `Eq`, there is a single bound `Self : Eq`). This is so that
    /// default methods get to assume that the `Self` parameters
    /// implements the trait.
    pub generics: Generics,

    /// The "supertrait" bounds.
    pub bounds: ParamBounds,
    pub trait_ref: Rc<ty::TraitRef>,
}

/// Records the substitutions used to translate the polytype for an
/// item into the monotype of an item reference.
#[deriving(Clone)]
pub struct ItemSubsts {
    pub substs: Substs,
}

pub type type_cache = RefCell<DefIdMap<Polytype>>;

pub type node_type_table = RefCell<HashMap<uint,t>>;

/// Records information about each unboxed closure.
#[deriving(Clone)]
pub struct UnboxedClosure {
    /// The type of the unboxed closure.
    pub closure_type: ClosureTy,
    /// The kind of unboxed closure this is.
    pub kind: UnboxedClosureKind,
}

#[deriving(Clone, PartialEq, Eq)]
pub enum UnboxedClosureKind {
    FnUnboxedClosureKind,
    FnMutUnboxedClosureKind,
    FnOnceUnboxedClosureKind,
}

impl UnboxedClosureKind {
    pub fn trait_did(&self, cx: &ctxt) -> ast::DefId {
        let result = match *self {
            FnUnboxedClosureKind => cx.lang_items.require(FnTraitLangItem),
            FnMutUnboxedClosureKind => {
                cx.lang_items.require(FnMutTraitLangItem)
            }
            FnOnceUnboxedClosureKind => {
                cx.lang_items.require(FnOnceTraitLangItem)
            }
        };
        match result {
            Ok(trait_did) => trait_did,
            Err(err) => cx.sess.fatal(err.as_slice()),
        }
    }
}

pub fn mk_ctxt<'tcx>(s: Session,
                     type_arena: &'tcx TypedArena<t_box_>,
                     dm: resolve::DefMap,
                     named_region_map: resolve_lifetime::NamedRegionMap,
                     map: ast_map::Map<'tcx>,
                     freevars: RefCell<FreevarMap>,
                     capture_modes: RefCell<CaptureModeMap>,
                     region_maps: middle::region::RegionMaps,
                     lang_items: middle::lang_items::LanguageItems,
                     stability: stability::Index) -> ctxt<'tcx> {
    ctxt {
        type_arena: type_arena,
        interner: RefCell::new(FnvHashMap::new()),
        named_region_map: named_region_map,
        item_variance_map: RefCell::new(DefIdMap::new()),
        variance_computed: Cell::new(false),
        next_id: Cell::new(primitives::LAST_PRIMITIVE_ID),
        sess: s,
        def_map: dm,
        region_maps: region_maps,
        node_types: RefCell::new(HashMap::new()),
        item_substs: RefCell::new(NodeMap::new()),
        trait_refs: RefCell::new(NodeMap::new()),
        trait_defs: RefCell::new(DefIdMap::new()),
        object_cast_map: RefCell::new(NodeMap::new()),
        map: map,
        intrinsic_defs: RefCell::new(DefIdMap::new()),
        freevars: freevars,
        tcache: RefCell::new(DefIdMap::new()),
        rcache: RefCell::new(HashMap::new()),
        short_names_cache: RefCell::new(HashMap::new()),
        needs_unwind_cleanup_cache: RefCell::new(HashMap::new()),
        tc_cache: RefCell::new(HashMap::new()),
        ast_ty_to_ty_cache: RefCell::new(NodeMap::new()),
        enum_var_cache: RefCell::new(DefIdMap::new()),
        impl_or_trait_items: RefCell::new(DefIdMap::new()),
        trait_item_def_ids: RefCell::new(DefIdMap::new()),
        trait_items_cache: RefCell::new(DefIdMap::new()),
        impl_trait_cache: RefCell::new(DefIdMap::new()),
        ty_param_defs: RefCell::new(NodeMap::new()),
        adjustments: RefCell::new(NodeMap::new()),
        normalized_cache: RefCell::new(HashMap::new()),
        lang_items: lang_items,
        provided_method_sources: RefCell::new(DefIdMap::new()),
        superstructs: RefCell::new(DefIdMap::new()),
        struct_fields: RefCell::new(DefIdMap::new()),
        destructor_for_type: RefCell::new(DefIdMap::new()),
        destructors: RefCell::new(DefIdSet::new()),
        trait_impls: RefCell::new(DefIdMap::new()),
        inherent_impls: RefCell::new(DefIdMap::new()),
        impl_items: RefCell::new(DefIdMap::new()),
        used_unsafe: RefCell::new(NodeSet::new()),
        used_mut_nodes: RefCell::new(NodeSet::new()),
        populated_external_types: RefCell::new(DefIdSet::new()),
        populated_external_traits: RefCell::new(DefIdSet::new()),
        upvar_borrow_map: RefCell::new(HashMap::new()),
        extern_const_statics: RefCell::new(DefIdMap::new()),
        extern_const_variants: RefCell::new(DefIdMap::new()),
        method_map: RefCell::new(FnvHashMap::new()),
        dependency_formats: RefCell::new(HashMap::new()),
        unboxed_closures: RefCell::new(DefIdMap::new()),
        node_lint_levels: RefCell::new(HashMap::new()),
        transmute_restrictions: RefCell::new(Vec::new()),
        stability: RefCell::new(stability),
        capture_modes: capture_modes,
        associated_types: RefCell::new(DefIdMap::new()),
        trait_associated_types: RefCell::new(DefIdMap::new()),
        selection_cache: traits::SelectionCache::new(),
        repr_hint_cache: RefCell::new(DefIdMap::new()),
   }
}

// Type constructors

// Interns a type/name combination, stores the resulting box in cx.interner,
// and returns the box as cast to an unsafe ptr (see comments for t above).
pub fn mk_t(cx: &ctxt, st: sty) -> t {
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

    let key = intern_key { sty: &st };

    match cx.interner.borrow().find(&key) {
        Some(t) => unsafe { return mem::transmute(&t.sty); },
        _ => ()
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
    fn sflags(substs: &Substs) -> uint {
        let mut f = 0u;
        let mut i = substs.types.iter();
        for tt in i {
            f |= get(*tt).flags;
        }
        match substs.regions {
            subst::ErasedRegions => {}
            subst::NonerasedRegions(ref regions) => {
                for r in regions.iter() {
                    f |= rflags(*r)
                }
            }
        }
        return f;
    }
    fn flags_for_bounds(bounds: &ExistentialBounds) -> uint {
        rflags(bounds.region_bound)
    }
    match &st {
      &ty_nil | &ty_bool | &ty_char | &ty_int(_) | &ty_float(_) | &ty_uint(_) |
      &ty_str => {}
      // You might think that we could just return ty_err for
      // any type containing ty_err as a component, and get
      // rid of the has_ty_err flag -- likewise for ty_bot (with
      // the exception of function types that return bot).
      // But doing so caused sporadic memory corruption, and
      // neither I (tjc) nor nmatsakis could figure out why,
      // so we're doing it this way.
      &ty_bot => flags |= has_ty_bot as uint,
      &ty_err => flags |= has_ty_err as uint,
      &ty_param(ref p) => {
          if p.space == subst::SelfSpace {
              flags |= has_self as uint;
          } else {
              flags |= has_params as uint;
          }
      }
      &ty_unboxed_closure(_, ref region) => flags |= rflags(*region),
      &ty_infer(_) => flags |= needs_infer as uint,
      &ty_enum(_, ref substs) | &ty_struct(_, ref substs) => {
          flags |= sflags(substs);
      }
      &ty_trait(box TyTrait { ref substs, ref bounds, .. }) => {
          flags |= sflags(substs);
          flags |= flags_for_bounds(bounds);
      }
      &ty_uniq(tt) | &ty_vec(tt, _) | &ty_open(tt) => {
        flags |= get(tt).flags
      }
      &ty_ptr(ref m) => {
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
        match f.store {
            RegionTraitStore(r, _) => {
                flags |= rflags(r);
            }
            _ => {}
        }
        for a in f.sig.inputs.iter() { flags |= get(*a).flags; }
        flags |= get(f.sig.output).flags;
        // T -> _|_ is *not* _|_ !
        flags &= !(has_ty_bot as uint);
        flags |= flags_for_bounds(&f.bounds);
      }
    }

    let t = cx.type_arena.alloc(t_box_ {
        sty: st,
        id: cx.next_id.get(),
        flags: flags,
    });

    let sty_ptr = &t.sty as *const sty;

    let key = intern_key {
        sty: sty_ptr,
    };

    cx.interner.borrow_mut().insert(key, t);

    cx.next_id.set(cx.next_id.get() + 1);

    unsafe {
        mem::transmute::<*const sty, t>(sty_ptr)
    }
}

#[inline]
pub fn mk_prim_t(primitive: &'static t_box_) -> t {
    unsafe {
        mem::transmute::<&'static t_box_, t>(primitive)
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

pub fn mk_str(cx: &ctxt) -> t {
    mk_t(cx, ty_str)
}

pub fn mk_str_slice(cx: &ctxt, r: Region, m: ast::Mutability) -> t {
    mk_rptr(cx, r,
            mt {
                ty: mk_t(cx, ty_str),
                mutbl: m
            })
}

pub fn mk_enum(cx: &ctxt, did: ast::DefId, substs: Substs) -> t {
    // take a copy of substs so that we own the vectors inside
    mk_t(cx, ty_enum(did, substs))
}

pub fn mk_uniq(cx: &ctxt, ty: t) -> t { mk_t(cx, ty_uniq(ty)) }

pub fn mk_ptr(cx: &ctxt, tm: mt) -> t { mk_t(cx, ty_ptr(tm)) }

pub fn mk_rptr(cx: &ctxt, r: Region, tm: mt) -> t { mk_t(cx, ty_rptr(r, tm)) }

pub fn mk_mut_rptr(cx: &ctxt, r: Region, ty: t) -> t {
    mk_rptr(cx, r, mt {ty: ty, mutbl: ast::MutMutable})
}
pub fn mk_imm_rptr(cx: &ctxt, r: Region, ty: t) -> t {
    mk_rptr(cx, r, mt {ty: ty, mutbl: ast::MutImmutable})
}

pub fn mk_mut_ptr(cx: &ctxt, ty: t) -> t {
    mk_ptr(cx, mt {ty: ty, mutbl: ast::MutMutable})
}

pub fn mk_imm_ptr(cx: &ctxt, ty: t) -> t {
    mk_ptr(cx, mt {ty: ty, mutbl: ast::MutImmutable})
}

pub fn mk_nil_ptr(cx: &ctxt) -> t {
    mk_ptr(cx, mt {ty: mk_nil(), mutbl: ast::MutImmutable})
}

pub fn mk_vec(cx: &ctxt, t: t, sz: Option<uint>) -> t {
    mk_t(cx, ty_vec(t, sz))
}

pub fn mk_slice(cx: &ctxt, r: Region, tm: mt) -> t {
    mk_rptr(cx, r,
            mt {
                ty: mk_vec(cx, tm.ty, None),
                mutbl: tm.mutbl
            })
}

pub fn mk_tup(cx: &ctxt, ts: Vec<t>) -> t { mk_t(cx, ty_tup(ts)) }

pub fn mk_closure(cx: &ctxt, fty: ClosureTy) -> t {
    mk_t(cx, ty_closure(box fty))
}

pub fn mk_bare_fn(cx: &ctxt, fty: BareFnTy) -> t {
    mk_t(cx, ty_bare_fn(fty))
}

pub fn mk_ctor_fn(cx: &ctxt,
                  binder_id: ast::NodeId,
                  input_tys: &[ty::t],
                  output: ty::t) -> t {
    let input_args = input_tys.iter().map(|t| *t).collect();
    mk_bare_fn(cx,
               BareFnTy {
                   fn_style: ast::NormalFn,
                   abi: abi::Rust,
                   sig: FnSig {
                    binder_id: binder_id,
                    inputs: input_args,
                    output: output,
                    variadic: false
                   }
                })
}


pub fn mk_trait(cx: &ctxt,
                did: ast::DefId,
                substs: Substs,
                bounds: ExistentialBounds)
                -> t {
    // take a copy of substs so that we own the vectors inside
    let inner = box TyTrait {
        def_id: did,
        substs: substs,
        bounds: bounds
    };
    mk_t(cx, ty_trait(inner))
}

pub fn mk_struct(cx: &ctxt, struct_id: ast::DefId, substs: Substs) -> t {
    // take a copy of substs so that we own the vectors inside
    mk_t(cx, ty_struct(struct_id, substs))
}

pub fn mk_unboxed_closure(cx: &ctxt, closure_id: ast::DefId, region: Region)
                          -> t {
    mk_t(cx, ty_unboxed_closure(closure_id, region))
}

pub fn mk_var(cx: &ctxt, v: TyVid) -> t { mk_infer(cx, TyVar(v)) }

pub fn mk_int_var(cx: &ctxt, v: IntVid) -> t { mk_infer(cx, IntVar(v)) }

pub fn mk_float_var(cx: &ctxt, v: FloatVid) -> t { mk_infer(cx, FloatVar(v)) }

pub fn mk_infer(cx: &ctxt, it: InferTy) -> t { mk_t(cx, ty_infer(it)) }

pub fn mk_param(cx: &ctxt, space: subst::ParamSpace, n: uint, k: DefId) -> t {
    mk_t(cx, ty_param(ParamTy { space: space, idx: n, def_id: k }))
}

pub fn mk_self_type(cx: &ctxt, did: ast::DefId) -> t {
    mk_param(cx, subst::SelfSpace, 0, did)
}

pub fn mk_param_from_def(cx: &ctxt, def: &TypeParameterDef) -> t {
    mk_param(cx, def.space, def.index, def.def_id)
}

pub fn mk_open(cx: &ctxt, t: t) -> t { mk_t(cx, ty_open(t)) }

pub fn walk_ty(ty: t, f: |t|) {
    maybe_walk_ty(ty, |t| { f(t); true });
}

pub fn maybe_walk_ty(ty: t, f: |t| -> bool) {
    if !f(ty) {
        return;
    }
    match get(ty).sty {
        ty_nil | ty_bot | ty_bool | ty_char | ty_int(_) | ty_uint(_) | ty_float(_) |
        ty_str | ty_infer(_) | ty_param(_) | ty_unboxed_closure(_, _) | ty_err => {}
        ty_uniq(ty) | ty_vec(ty, _) | ty_open(ty) => maybe_walk_ty(ty, f),
        ty_ptr(ref tm) | ty_rptr(_, ref tm) => {
            maybe_walk_ty(tm.ty, f);
        }
        ty_enum(_, ref substs) | ty_struct(_, ref substs) |
        ty_trait(box TyTrait { ref substs, .. }) => {
            for subty in (*substs).types.iter() {
                maybe_walk_ty(*subty, |x| f(x));
            }
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
pub fn fold_ty(cx: &ctxt, t0: t, fldop: |t| -> t) -> t {
    let mut f = ty_fold::BottomUpFolder {tcx: cx, fldop: fldop};
    f.fold_ty(t0)
}

impl ParamTy {
    pub fn new(space: subst::ParamSpace,
               index: uint,
               def_id: ast::DefId)
               -> ParamTy {
        ParamTy { space: space, idx: index, def_id: def_id }
    }

    pub fn for_self(trait_def_id: ast::DefId) -> ParamTy {
        ParamTy::new(subst::SelfSpace, 0, trait_def_id)
    }

    pub fn for_def(def: &TypeParameterDef) -> ParamTy {
        ParamTy::new(def.space, def.index, def.def_id)
    }

    pub fn to_ty(self, tcx: &ty::ctxt) -> ty::t {
        ty::mk_param(tcx, self.space, self.idx, self.def_id)
    }

    pub fn is_self(&self) -> bool {
        self.space == subst::SelfSpace && self.idx == 0
    }
}

impl ItemSubsts {
    pub fn empty() -> ItemSubsts {
        ItemSubsts { substs: Substs::empty() }
    }

    pub fn is_noop(&self) -> bool {
        self.substs.is_noop()
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

pub fn type_needs_subst(ty: t) -> bool {
    tbox_has_flag(get(ty), needs_subst)
}

pub fn trait_ref_contains_error(tref: &ty::TraitRef) -> bool {
    tref.substs.types.any(|&t| type_is_error(t))
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
        ty_param(ref p) => p.space == subst::SelfSpace,
        _ => false
    }
}

fn type_is_slice(ty: t) -> bool {
    match get(ty).sty {
        ty_ptr(mt) | ty_rptr(_, mt) => match get(mt.ty).sty {
            ty_vec(_, None) | ty_str => true,
            _ => false,
        },
        _ => false
    }
}

pub fn type_is_vec(ty: t) -> bool {
    match get(ty).sty {
        ty_vec(..) => true,
        ty_ptr(mt{ty: t, ..}) | ty_rptr(_, mt{ty: t, ..}) |
        ty_uniq(t) => match get(t).sty {
            ty_vec(_, None) => true,
            _ => false
        },
        _ => false
    }
}

pub fn type_is_structural(ty: t) -> bool {
    match get(ty).sty {
      ty_struct(..) | ty_tup(_) | ty_enum(..) | ty_closure(_) |
      ty_vec(_, Some(_)) | ty_unboxed_closure(..) => true,
      _ => type_is_slice(ty) | type_is_trait(ty)
    }
}

pub fn type_is_simd(cx: &ctxt, ty: t) -> bool {
    match get(ty).sty {
        ty_struct(did, _) => lookup_simd(cx, did),
        _ => false
    }
}

pub fn sequence_element_type(cx: &ctxt, ty: t) -> t {
    match get(ty).sty {
        ty_vec(ty, _) => ty,
        ty_str => mk_mach_uint(ast::TyU8),
        ty_open(ty) => sequence_element_type(cx, ty),
        _ => cx.sess.bug(format!("sequence_element_type called on non-sequence value: {}",
                                 ty_to_string(cx, ty)).as_slice()),
    }
}

pub fn simd_type(cx: &ctxt, ty: t) -> t {
    match get(ty).sty {
        ty_struct(did, ref substs) => {
            let fields = lookup_struct_fields(cx, did);
            lookup_field_type(cx, did, fields.get(0).id, substs)
        }
        _ => fail!("simd_type called on invalid type")
    }
}

pub fn simd_size(cx: &ctxt, ty: t) -> uint {
    match get(ty).sty {
        ty_struct(did, _) => {
            let fields = lookup_struct_fields(cx, did);
            fields.len()
        }
        _ => fail!("simd_size called on invalid type")
    }
}

pub fn type_is_region_ptr(ty: t) -> bool {
    match get(ty).sty {
        ty_rptr(..) => true,
        _ => false
    }
}

pub fn type_is_unsafe_ptr(ty: t) -> bool {
    match get(ty).sty {
      ty_ptr(_) => return true,
      _ => return false
    }
}

pub fn type_is_unique(ty: t) -> bool {
    match get(ty).sty {
        ty_uniq(_) => match get(ty).sty {
            ty_trait(..) => false,
            _ => true
        },
        _ => false
    }
}

pub fn type_is_fat_ptr(cx: &ctxt, ty: t) -> bool {
    match get(ty).sty {
        ty_ptr(mt{ty, ..}) | ty_rptr(_, mt{ty, ..})
        | ty_uniq(ty) if !type_is_sized(cx, ty) => true,
        _ => false,
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
      ty_infer(IntVar(_)) | ty_infer(FloatVar(_)) |
      ty_bare_fn(..) | ty_ptr(_) => true,
      _ => false
    }
}

/// Returns true if this type is a floating point type and false otherwise.
pub fn type_is_floating_point(ty: t) -> bool {
    match get(ty).sty {
        ty_float(_) => true,
        _ => false,
    }
}

pub fn type_needs_drop(cx: &ctxt, ty: t) -> bool {
    type_contents(cx, ty).needs_drop(cx)
}

// Some things don't need cleanups during unwinding because the
// task can free them all at once later. Currently only things
// that only contain scalars and shared boxes can avoid unwind
// cleanups.
pub fn type_needs_unwind_cleanup(cx: &ctxt, ty: t) -> bool {
    match cx.needs_unwind_cleanup_cache.borrow().find(&ty) {
        Some(&result) => return result,
        None => ()
    }

    let mut tycache = HashSet::new();
    let needs_unwind_cleanup =
        type_needs_unwind_cleanup_(cx, ty, &mut tycache);
    cx.needs_unwind_cleanup_cache.borrow_mut().insert(ty, needs_unwind_cleanup);
    needs_unwind_cleanup
}

fn type_needs_unwind_cleanup_(cx: &ctxt, ty: t,
                              tycache: &mut HashSet<t>) -> bool {

    // Prevent infinite recursion
    if !tycache.insert(ty) {
        return false;
    }

    let mut needs_unwind_cleanup = false;
    maybe_walk_ty(ty, |ty| {
        let result = match get(ty).sty {
          ty_nil | ty_bot | ty_bool | ty_int(_) | ty_uint(_) | ty_float(_) |
          ty_tup(_) | ty_ptr(_) => {
            true
          }
          ty_enum(did, ref substs) => {
            for v in (*enum_variants(cx, did)).iter() {
                for aty in v.args.iter() {
                    let t = aty.subst(cx, substs);
                    needs_unwind_cleanup |=
                        type_needs_unwind_cleanup_(cx, t, tycache);
                }
            }
            !needs_unwind_cleanup
          }
          _ => {
            needs_unwind_cleanup = true;
            false
          }
        };

        result
    });

    needs_unwind_cleanup
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
    pub bits: u64
}

macro_rules! def_type_content_sets(
    (mod $mname:ident { $($name:ident = $bits:expr),+ }) => {
        #[allow(non_snake_case)]
        mod $mname {
            use middle::ty::TypeContents;
            $(pub static $name: TypeContents = TypeContents { bits: $bits };)+
        }
    }
)

def_type_content_sets!(
    mod TC {
        None                                = 0b0000_0000__0000_0000__0000,

        // Things that are interior to the value (first nibble):
        InteriorUnsized                     = 0b0000_0000__0000_0000__0001,
        InteriorUnsafe                      = 0b0000_0000__0000_0000__0010,
        // InteriorAll                         = 0b00000000__00000000__1111,

        // Things that are owned by the value (second and third nibbles):
        OwnsOwned                           = 0b0000_0000__0000_0001__0000,
        OwnsDtor                            = 0b0000_0000__0000_0010__0000,
        OwnsManaged /* see [1] below */     = 0b0000_0000__0000_0100__0000,
        OwnsAffine                          = 0b0000_0000__0000_1000__0000,
        OwnsAll                             = 0b0000_0000__1111_1111__0000,

        // Things that are reachable by the value in any way (fourth nibble):
        ReachesBorrowed                     = 0b0000_0010__0000_0000__0000,
        // ReachesManaged /* see [1] below */  = 0b0000_0100__0000_0000__0000,
        ReachesMutable                      = 0b0000_1000__0000_0000__0000,
        ReachesFfiUnsafe                    = 0b0010_0000__0000_0000__0000,
        ReachesAll                          = 0b0011_1111__0000_0000__0000,

        // Things that cause values to *move* rather than *copy*. This
        // is almost the same as the `Copy` trait, but for managed
        // data -- atm, we consider managed data to copy, not move,
        // but it does not impl Copy as a pure memcpy is not good
        // enough. Yuck.
        Moves                               = 0b0000_0000__0000_1011__0000,

        // Things that mean drop glue is necessary
        NeedsDrop                           = 0b0000_0000__0000_0111__0000,

        // Things that prevent values from being considered sized
        Nonsized                            = 0b0000_0000__0000_0000__0001,

        // Things that make values considered not POD (would be same
        // as `Moves`, but for the fact that managed data `@` is
        // not considered POD)
        Noncopy                              = 0b0000_0000__0000_1111__0000,

        // Bits to set when a managed value is encountered
        //
        // [1] Do not set the bits TC::OwnsManaged or
        //     TC::ReachesManaged directly, instead reference
        //     TC::Managed to set them both at once.
        Managed                             = 0b0000_0100__0000_0100__0000,

        // All bits
        All                                 = 0b1111_1111__1111_1111__1111
    }
)

impl TypeContents {
    pub fn when(&self, cond: bool) -> TypeContents {
        if cond {*self} else {TC::None}
    }

    pub fn intersects(&self, tc: TypeContents) -> bool {
        (self.bits & tc.bits) != 0
    }

    pub fn owns_managed(&self) -> bool {
        self.intersects(TC::OwnsManaged)
    }

    pub fn owns_owned(&self) -> bool {
        self.intersects(TC::OwnsOwned)
    }

    pub fn is_sized(&self, _: &ctxt) -> bool {
        !self.intersects(TC::Nonsized)
    }

    pub fn interior_unsafe(&self) -> bool {
        self.intersects(TC::InteriorUnsafe)
    }

    pub fn interior_unsized(&self) -> bool {
        self.intersects(TC::InteriorUnsized)
    }

    pub fn moves_by_default(&self, _: &ctxt) -> bool {
        self.intersects(TC::Moves)
    }

    pub fn needs_drop(&self, _: &ctxt) -> bool {
        self.intersects(TC::NeedsDrop)
    }

    pub fn owned_pointer(&self) -> TypeContents {
        /*!
         * Includes only those bits that still apply
         * when indirected through a `Box` pointer
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

    pub fn has_dtor(&self) -> bool {
        self.intersects(TC::OwnsDtor)
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

impl fmt::Show for TypeContents {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TypeContents({:t})", self.bits)
    }
}

pub fn type_interior_is_unsafe(cx: &ctxt, t: ty::t) -> bool {
    type_contents(cx, t).interior_unsafe()
}

pub fn type_contents(cx: &ctxt, ty: t) -> TypeContents {
    let ty_id = type_id(ty);

    match cx.tc_cache.borrow().find(&ty_id) {
        Some(tc) => { return *tc; }
        None => {}
    }

    let mut cache = HashMap::new();
    let result = tc_ty(cx, ty, &mut cache);

    cx.tc_cache.borrow_mut().insert(ty_id, result);
    return result;

    fn tc_ty(cx: &ctxt,
             ty: t,
             cache: &mut HashMap<uint, TypeContents>) -> TypeContents
    {
        // Subtle: Note that we are *not* using cx.tc_cache here but rather a
        // private cache for this walk.  This is needed in the case of cyclic
        // types like:
        //
        //     struct List { next: Box<Option<List>>, ... }
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
        match cx.tc_cache.borrow().find(&ty_id) {    // Must check both caches!
            Some(tc) => { return *tc; }
            None => {}
        }
        cache.insert(ty_id, TC::None);

        let result = match get(ty).sty {
            // uint and int are ffi-unsafe
            ty_uint(ast::TyU) | ty_int(ast::TyI) => {
                TC::ReachesFfiUnsafe
            }

            // Scalar and unique types are sendable, and durable
            ty_infer(ty::SkolemizedIntTy(_)) |
            ty_nil | ty_bot | ty_bool | ty_int(_) | ty_uint(_) | ty_float(_) |
            ty_bare_fn(_) | ty::ty_char => {
                TC::None
            }

            ty_closure(ref c) => {
                closure_contents(cx, &**c) | TC::ReachesFfiUnsafe
            }

            ty_uniq(typ) => {
                TC::ReachesFfiUnsafe | match get(typ).sty {
                    ty_str => TC::OwnsOwned,
                    _ => tc_ty(cx, typ, cache).owned_pointer(),
                }
            }

            ty_trait(box TyTrait { bounds, .. }) => {
                object_contents(cx, bounds) | TC::ReachesFfiUnsafe | TC::Nonsized
            }

            ty_ptr(ref mt) => {
                tc_ty(cx, mt.ty, cache).unsafe_pointer()
            }

            ty_rptr(r, ref mt) => {
                TC::ReachesFfiUnsafe | match get(mt.ty).sty {
                    ty_str => borrowed_contents(r, ast::MutImmutable),
                    ty_vec(..) => tc_ty(cx, mt.ty, cache).reference(borrowed_contents(r, mt.mutbl)),
                    _ => tc_ty(cx, mt.ty, cache).reference(borrowed_contents(r, mt.mutbl)),
                }
            }

            ty_vec(t, Some(_)) => {
                tc_ty(cx, t, cache)
            }

            ty_vec(t, None) => {
                tc_ty(cx, t, cache) | TC::Nonsized
            }
            ty_str => TC::Nonsized,

            ty_struct(did, ref substs) => {
                let flds = struct_fields(cx, did, substs);
                let mut res =
                    TypeContents::union(flds.as_slice(),
                                        |f| tc_mt(cx, f.mt, cache));

                if !lookup_repr_hints(cx, did).contains(&attr::ReprExtern) {
                    res = res | TC::ReachesFfiUnsafe;
                }

                if ty::has_dtor(cx, did) {
                    res = res | TC::OwnsDtor;
                }
                apply_lang_items(cx, did, res)
            }

            ty_unboxed_closure(did, r) => {
                // FIXME(#14449): `borrowed_contents` below assumes `&mut`
                // unboxed closure.
                let upvars = unboxed_closure_upvars(cx, did);
                TypeContents::union(upvars.as_slice(),
                                    |f| tc_ty(cx, f.ty, cache)) |
                    borrowed_contents(r, MutMutable)
            }

            ty_tup(ref tys) => {
                TypeContents::union(tys.as_slice(),
                                    |ty| tc_ty(cx, *ty, cache))
            }

            ty_enum(did, ref substs) => {
                let variants = substd_enum_variants(cx, did, substs);
                let mut res =
                    TypeContents::union(variants.as_slice(), |variant| {
                        TypeContents::union(variant.args.as_slice(),
                                            |arg_ty| {
                            tc_ty(cx, *arg_ty, cache)
                        })
                    });

                if ty::has_dtor(cx, did) {
                    res = res | TC::OwnsDtor;
                }

                if variants.len() != 0 {
                    let repr_hints = lookup_repr_hints(cx, did);
                    if repr_hints.len() > 1 {
                        // this is an error later on, but this type isn't safe
                        res = res | TC::ReachesFfiUnsafe;
                    }

                    match repr_hints.as_slice().get(0) {
                        Some(h) => if !h.is_ffi_safe() {
                            res = res | TC::ReachesFfiUnsafe;
                        },
                        // ReprAny
                        None => {
                            res = res | TC::ReachesFfiUnsafe;

                            // We allow ReprAny enums if they are eligible for
                            // the nullable pointer optimization and the
                            // contained type is an `extern fn`

                            if variants.len() == 2 {
                                let mut data_idx = 0;

                                if variants.get(0).args.len() == 0 {
                                    data_idx = 1;
                                }

                                if variants.get(data_idx).args.len() == 1 {
                                    match get(*variants.get(data_idx).args.get(0)).sty {
                                        ty_bare_fn(..) => { res = res - TC::ReachesFfiUnsafe; }
                                        _ => { }
                                    }
                                }
                            }
                        }
                    }
                }


                apply_lang_items(cx, did, res)
            }

            ty_param(p) => {
                // We only ever ask for the kind of types that are defined in
                // the current crate; therefore, the only type parameters that
                // could be in scope are those defined in the current crate.
                // If this assertion failures, it is likely because of a
                // failure in the cross-crate inlining code to translate a
                // def-id.
                assert_eq!(p.def_id.krate, ast::LOCAL_CRATE);

                let ty_param_defs = cx.ty_param_defs.borrow();
                let tp_def = ty_param_defs.get(&p.def_id.node);
                kind_bounds_to_contents(
                    cx,
                    tp_def.bounds.builtin_bounds,
                    tp_def.bounds.trait_bounds.as_slice())
            }

            ty_infer(_) => {
                // This occurs during coherence, but shouldn't occur at other
                // times.
                TC::All
            }

            ty_open(t) => {
                let result = tc_ty(cx, t, cache);
                assert!(!result.is_sized(cx))
                result.unsafe_pointer() | TC::Nonsized
            }

            ty_err => {
                cx.sess.bug("asked to compute contents of error type");
            }
        };

        cache.insert(ty_id, result);
        return result;
    }

    fn tc_mt(cx: &ctxt,
             mt: mt,
             cache: &mut HashMap<uint, TypeContents>) -> TypeContents
    {
        let mc = TC::ReachesMutable.when(mt.mutbl == MutMutable);
        mc | tc_ty(cx, mt.ty, cache)
    }

    fn apply_lang_items(cx: &ctxt,
                        did: ast::DefId,
                        tc: TypeContents)
                        -> TypeContents
    {
        if Some(did) == cx.lang_items.managed_bound() {
            tc | TC::Managed
        } else if Some(did) == cx.lang_items.no_copy_bound() {
            tc | TC::OwnsAffine
        } else if Some(did) == cx.lang_items.unsafe_type() {
            tc | TC::InteriorUnsafe
        } else {
            tc
        }
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

    fn closure_contents(cx: &ctxt, cty: &ClosureTy) -> TypeContents {
        // Closure contents are just like trait contents, but with potentially
        // even more stuff.
        let st = object_contents(cx, cty.bounds);

        let st = match cty.store {
            UniqTraitStore => {
                st.owned_pointer()
            }
            RegionTraitStore(r, mutbl) => {
                st.reference(borrowed_contents(r, mutbl))
            }
        };

        // This also prohibits "@once fn" from being copied, which allows it to
        // be called. Neither way really makes much sense.
        let ot = match cty.onceness {
            ast::Once => TC::OwnsAffine,
            ast::Many => TC::None,
        };

        st | ot
    }

    fn object_contents(cx: &ctxt,
                       bounds: ExistentialBounds)
                       -> TypeContents {
        // These are the type contents of the (opaque) interior
        kind_bounds_to_contents(cx, bounds.builtin_bounds, [])
    }

    fn kind_bounds_to_contents(cx: &ctxt,
                               bounds: BuiltinBounds,
                               traits: &[Rc<TraitRef>])
                               -> TypeContents {
        let _i = indenter();
        let mut tc = TC::All;
        each_inherited_builtin_bound(cx, bounds, traits, |bound| {
            tc = tc - match bound {
                BoundSync | BoundSend => TC::None,
                BoundSized => TC::Nonsized,
                BoundCopy => TC::Noncopy,
            };
        });
        return tc;

        // Iterates over all builtin bounds on the type parameter def, including
        // those inherited from traits with builtin-kind-supertraits.
        fn each_inherited_builtin_bound(cx: &ctxt,
                                        bounds: BuiltinBounds,
                                        traits: &[Rc<TraitRef>],
                                        f: |BuiltinBound|) {
            for bound in bounds.iter() {
                f(bound);
            }

            each_bound_trait_and_supertraits(cx, traits, |trait_ref| {
                let trait_def = lookup_trait_def(cx, trait_ref.def_id);
                for bound in trait_def.bounds.builtin_bounds.iter() {
                    f(bound);
                }
                true
            });
        }
    }
}

pub fn type_moves_by_default(cx: &ctxt, ty: t) -> bool {
    type_contents(cx, ty).moves_by_default(cx)
}

pub fn is_ffi_safe(cx: &ctxt, ty: t) -> bool {
    !type_contents(cx, ty).intersects(TC::ReachesFfiUnsafe)
}

// True if instantiating an instance of `r_ty` requires an instance of `r_ty`.
pub fn is_instantiable(cx: &ctxt, r_ty: t) -> bool {
    fn type_requires(cx: &ctxt, seen: &mut Vec<DefId>,
                     r_ty: t, ty: t) -> bool {
        debug!("type_requires({}, {})?",
               ::util::ppaux::ty_to_string(cx, r_ty),
               ::util::ppaux::ty_to_string(cx, ty));

        let r = {
            get(r_ty).sty == get(ty).sty ||
                subtypes_require(cx, seen, r_ty, ty)
        };

        debug!("type_requires({}, {})? {}",
               ::util::ppaux::ty_to_string(cx, r_ty),
               ::util::ppaux::ty_to_string(cx, ty),
               r);
        return r;
    }

    fn subtypes_require(cx: &ctxt, seen: &mut Vec<DefId>,
                        r_ty: t, ty: t) -> bool {
        debug!("subtypes_require({}, {})?",
               ::util::ppaux::ty_to_string(cx, r_ty),
               ::util::ppaux::ty_to_string(cx, ty));

        let r = match get(ty).sty {
            // fixed length vectors need special treatment compared to
            // normal vectors, since they don't necessarily have the
            // possibility to have length zero.
            ty_vec(_, Some(0)) => false, // don't need no contents
            ty_vec(ty, Some(_)) => type_requires(cx, seen, r_ty, ty),

            ty_nil |
            ty_bot |
            ty_bool |
            ty_char |
            ty_int(_) |
            ty_uint(_) |
            ty_float(_) |
            ty_str |
            ty_bare_fn(_) |
            ty_closure(_) |
            ty_infer(_) |
            ty_err |
            ty_param(_) |
            ty_vec(_, None) => {
                false
            }
            ty_uniq(typ) | ty_open(typ) => {
                type_requires(cx, seen, r_ty, typ)
            }
            ty_rptr(_, ref mt) => {
                type_requires(cx, seen, r_ty, mt.ty)
            }

            ty_ptr(..) => {
                false           // unsafe ptrs can always be NULL
            }

            ty_trait(..) => {
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

            ty_unboxed_closure(did, _) => {
                let upvars = unboxed_closure_upvars(cx, did);
                upvars.iter().any(|f| type_requires(cx, seen, r_ty, f.ty))
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
                        let sty = aty.subst(cx, substs);
                        type_requires(cx, seen, r_ty, sty)
                    })
                });
                seen.pop().unwrap();
                r
            }
        };

        debug!("subtypes_require({}, {})? {}",
               ::util::ppaux::ty_to_string(cx, r_ty),
               ::util::ppaux::ty_to_string(cx, ty),
               r);

        return r;
    }

    let mut seen = Vec::new();
    !subtypes_require(cx, &mut seen, r_ty, r_ty)
}

/// Describes whether a type is representable. For types that are not
/// representable, 'SelfRecursive' and 'ContainsRecursive' are used to
/// distinguish between types that are recursive with themselves and types that
/// contain a different recursive type. These cases can therefore be treated
/// differently when reporting errors.
#[deriving(PartialEq)]
pub enum Representability {
    Representable,
    SelfRecursive,
    ContainsRecursive,
}

/// Check whether a type is representable. This means it cannot contain unboxed
/// structural recursion. This check is needed for structs and enums.
pub fn is_type_representable(cx: &ctxt, sp: Span, ty: t) -> Representability {

    // Iterate until something non-representable is found
    fn find_nonrepresentable<It: Iterator<t>>(cx: &ctxt, sp: Span, seen: &mut Vec<DefId>,
                                              mut iter: It) -> Representability {
        for ty in iter {
            let r = type_structurally_recursive(cx, sp, seen, ty);
            if r != Representable {
                 return r
            }
        }
        Representable
    }

    // Does the type `ty` directly (without indirection through a pointer)
    // contain any types on stack `seen`?
    fn type_structurally_recursive(cx: &ctxt, sp: Span, seen: &mut Vec<DefId>,
                                   ty: t) -> Representability {
        debug!("type_structurally_recursive: {}",
               ::util::ppaux::ty_to_string(cx, ty));

        // Compare current type to previously seen types
        match get(ty).sty {
            ty_struct(did, _) |
            ty_enum(did, _) => {
                for (i, &seen_did) in seen.iter().enumerate() {
                    if did == seen_did {
                        return if i == 0 { SelfRecursive }
                               else { ContainsRecursive }
                    }
                }
            }
            _ => (),
        }

        // Check inner types
        match get(ty).sty {
            // Tuples
            ty_tup(ref ts) => {
                find_nonrepresentable(cx, sp, seen, ts.iter().map(|t| *t))
            }
            // Fixed-length vectors.
            // FIXME(#11924) Behavior undecided for zero-length vectors.
            ty_vec(ty, Some(_)) => {
                type_structurally_recursive(cx, sp, seen, ty)
            }

            // Push struct and enum def-ids onto `seen` before recursing.
            ty_struct(did, ref substs) => {
                seen.push(did);
                let fields = struct_fields(cx, did, substs);
                let r = find_nonrepresentable(cx, sp, seen,
                                              fields.iter().map(|f| f.mt.ty));
                seen.pop();
                r
            }

            ty_enum(did, ref substs) => {
                seen.push(did);
                let vs = enum_variants(cx, did);

                let mut r = Representable;
                for variant in vs.iter() {
                    let iter = variant.args.iter().map(|aty| {
                        aty.subst_spanned(cx, substs, Some(sp))
                    });
                    r = find_nonrepresentable(cx, sp, seen, iter);

                    if r != Representable { break }
                }

                seen.pop();
                r
            }

            ty_unboxed_closure(did, _) => {
                let upvars = unboxed_closure_upvars(cx, did);
                find_nonrepresentable(cx,
                                      sp,
                                      seen,
                                      upvars.iter().map(|f| f.ty))
            }

            _ => Representable,
        }
    }

    debug!("is_type_representable: {}",
           ::util::ppaux::ty_to_string(cx, ty));

    // To avoid a stack overflow when checking an enum variant or struct that
    // contains a different, structurally recursive type, maintain a stack
    // of seen types and check recursion for each of them (issues #3008, #3779).
    let mut seen: Vec<DefId> = Vec::new();
    type_structurally_recursive(cx, sp, &mut seen, ty)
}

pub fn type_is_trait(ty: t) -> bool {
    type_trait_info(ty).is_some()
}

pub fn type_trait_info(ty: t) -> Option<&'static TyTrait> {
    match get(ty).sty {
        ty_uniq(ty) | ty_rptr(_, mt { ty, ..}) | ty_ptr(mt { ty, ..}) => match get(ty).sty {
            ty_trait(ref t) => Some(&**t),
            _ => None
        },
        ty_trait(ref t) => Some(&**t),
        _ => None
    }
}

pub fn type_is_integral(ty: t) -> bool {
    match get(ty).sty {
      ty_infer(IntVar(_)) | ty_int(_) | ty_uint(_) => true,
      _ => false
    }
}

pub fn type_is_skolemized(ty: t) -> bool {
    match get(ty).sty {
      ty_infer(SkolemizedTy(_)) => true,
      ty_infer(SkolemizedIntTy(_)) => true,
      _ => false
    }
}

pub fn type_is_uint(ty: t) -> bool {
    match get(ty).sty {
      ty_infer(IntVar(_)) | ty_uint(ast::TyU) => true,
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

// Is the type's representation size known at compile time?
pub fn type_is_sized(cx: &ctxt, ty: t) -> bool {
    type_contents(cx, ty).is_sized(cx)
}

pub fn lltype_is_sized(cx: &ctxt, ty: t) -> bool {
    match get(ty).sty {
        ty_open(_) => true,
        _ => type_contents(cx, ty).is_sized(cx)
    }
}

// Return the smallest part of t which is unsized. Fails if t is sized.
// 'Smallest' here means component of the static representation of the type; not
// the size of an object at runtime.
pub fn unsized_part_of_type(cx: &ctxt, ty: t) -> t {
    match get(ty).sty {
        ty_str | ty_trait(..) | ty_vec(..) => ty,
        ty_struct(def_id, ref substs) => {
            let unsized_fields: Vec<_> = struct_fields(cx, def_id, substs).iter()
                .map(|f| f.mt.ty).filter(|ty| !type_is_sized(cx, *ty)).collect();
            // Exactly one of the fields must be unsized.
            assert!(unsized_fields.len() == 1)

            unsized_part_of_type(cx, unsized_fields[0])
        }
        _ => {
            assert!(type_is_sized(cx, ty),
                    "unsized_part_of_type failed even though ty is unsized");
            fail!("called unsized_part_of_type with sized ty");
        }
    }
}

// Whether a type is enum like, that is an enum type with only nullary
// constructors
pub fn type_is_c_like_enum(cx: &ctxt, ty: t) -> bool {
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

// Returns the type and mutability of *t.
//
// The parameter `explicit` indicates if this is an *explicit* dereference.
// Some types---notably unsafe ptrs---can only be dereferenced explicitly.
pub fn deref(t: t, explicit: bool) -> Option<mt> {
    match get(t).sty {
        ty_uniq(ty) => {
            Some(mt {
                ty: ty,
                mutbl: ast::MutImmutable,
            })
        },
        ty_rptr(_, mt) => Some(mt),
        ty_ptr(mt) if explicit => Some(mt),
        _ => None
    }
}

pub fn deref_or_dont(t: t) -> t {
    match get(t).sty {
        ty_uniq(ty) => ty,
        ty_rptr(_, mt) | ty_ptr(mt) => mt.ty,
        _ => t
    }
}

pub fn close_type(cx: &ctxt, t: t) -> t {
    match get(t).sty {
        ty_open(t) => mk_rptr(cx, ReStatic, mt {ty: t, mutbl:ast::MutImmutable}),
        _ => cx.sess.bug(format!("Trying to close a non-open type {}",
                                 ty_to_string(cx, t)).as_slice())
    }
}

pub fn type_content(t: t) -> t {
    match get(t).sty {
        ty_uniq(ty) => ty,
        ty_rptr(_, mt) |ty_ptr(mt) => mt.ty,
        _ => t
    }

}

// Extract the unsized type in an open type (or just return t if it is not open).
pub fn unopen_type(t: t) -> t {
    match get(t).sty {
        ty_open(t) => t,
        _ => t
    }
}

// Returns the type of t[i]
pub fn index(ty: t) -> Option<t> {
    match get(ty).sty {
        ty_vec(t, _) => Some(t),
        _ => None
    }
}

// Returns the type of elements contained within an 'array-like' type.
// This is exactly the same as the above, except it supports strings,
// which can't actually be indexed.
pub fn array_element_ty(t: t) -> Option<t> {
    match get(t).sty {
        ty_vec(t, _) => Some(t),
        ty_str => Some(mk_u8()),
        _ => None
    }
}

pub fn node_id_to_trait_ref(cx: &ctxt, id: ast::NodeId) -> Rc<ty::TraitRef> {
    match cx.trait_refs.borrow().find(&id) {
        Some(t) => t.clone(),
        None => cx.sess.bug(
            format!("node_id_to_trait_ref: no trait ref for node `{}`",
                    cx.map.node_to_string(id)).as_slice())
    }
}

pub fn try_node_id_to_type(cx: &ctxt, id: ast::NodeId) -> Option<t> {
    cx.node_types.borrow().find_copy(&(id as uint))
}

pub fn node_id_to_type(cx: &ctxt, id: ast::NodeId) -> t {
    match try_node_id_to_type(cx, id) {
       Some(t) => t,
       None => cx.sess.bug(
           format!("node_id_to_type: no type for node `{}`",
                   cx.map.node_to_string(id)).as_slice())
    }
}

pub fn node_id_to_type_opt(cx: &ctxt, id: ast::NodeId) -> Option<t> {
    match cx.node_types.borrow().find(&(id as uint)) {
       Some(&t) => Some(t),
       None => None
    }
}

pub fn node_id_item_substs(cx: &ctxt, id: ast::NodeId) -> ItemSubsts {
    match cx.item_substs.borrow().find(&id) {
      None => ItemSubsts::empty(),
      Some(ts) => ts.clone(),
    }
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

/// Returns the ABI of the given function.
pub fn ty_fn_abi(fty: t) -> abi::Abi {
    match get(fty).sty {
        ty_bare_fn(ref f) => f.abi,
        ty_closure(ref f) => f.abi,
        _ => fail!("ty_fn_abi() called on non-fn type"),
    }
}

// Type accessors for substructures of types
pub fn ty_fn_args(fty: t) -> Vec<t> {
    match get(fty).sty {
        ty_bare_fn(ref f) => f.sig.inputs.clone(),
        ty_closure(ref f) => f.sig.inputs.clone(),
        ref s => {
            fail!("ty_fn_args() called on non-fn type: {:?}", s)
        }
    }
}

pub fn ty_closure_store(fty: t) -> TraitStore {
    match get(fty).sty {
        ty_closure(ref f) => f.store,
        ty_unboxed_closure(..) => {
            // Close enough for the purposes of all the callers of this
            // function (which is soon to be deprecated anyhow).
            UniqTraitStore
        }
        ref s => {
            fail!("ty_closure_store() called on non-closure type: {:?}", s)
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

pub fn ty_region(tcx: &ctxt,
                 span: Span,
                 ty: t) -> Region {
    match get(ty).sty {
        ty_rptr(r, _) => r,
        ref s => {
            tcx.sess.span_bug(
                span,
                format!("ty_region() invoked on an inappropriate ty: {:?}",
                        s).as_slice());
        }
    }
}

pub fn free_region_from_def(free_id: ast::NodeId, def: &RegionParameterDef)
    -> ty::Region
{
    ty::ReFree(ty::FreeRegion { scope_id: free_id,
                                bound_region: ty::BrNamed(def.def_id,
                                                          def.name) })
}

// Returns the type of a pattern as a monotype. Like @expr_ty, this function
// doesn't provide type parameter substitutions.
pub fn pat_ty(cx: &ctxt, pat: &ast::Pat) -> t {
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
// instead of "fn(t) -> T with T = int".
pub fn expr_ty(cx: &ctxt, expr: &ast::Expr) -> t {
    return node_id_to_type(cx, expr.id);
}

pub fn expr_ty_opt(cx: &ctxt, expr: &ast::Expr) -> Option<t> {
    return node_id_to_type_opt(cx, expr.id);
}

pub fn expr_ty_adjusted(cx: &ctxt, expr: &ast::Expr) -> t {
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

    adjust_ty(cx, expr.span, expr.id, expr_ty(cx, expr),
              cx.adjustments.borrow().find(&expr.id),
              |method_call| cx.method_map.borrow().find(&method_call).map(|method| method.ty))
}

pub fn expr_span(cx: &ctxt, id: NodeId) -> Span {
    match cx.map.find(id) {
        Some(ast_map::NodeExpr(e)) => {
            e.span
        }
        Some(f) => {
            cx.sess.bug(format!("Node id {} is not an expr: {:?}",
                                id,
                                f).as_slice());
        }
        None => {
            cx.sess.bug(format!("Node id {} is not present \
                                in the node map", id).as_slice());
        }
    }
}

pub fn local_var_name_str(cx: &ctxt, id: NodeId) -> InternedString {
    match cx.map.find(id) {
        Some(ast_map::NodeLocal(pat)) => {
            match pat.node {
                ast::PatIdent(_, ref path1, _) => {
                    token::get_ident(path1.node)
                }
                _ => {
                    cx.sess.bug(
                        format!("Variable id {} maps to {:?}, not local",
                                id,
                                pat).as_slice());
                }
            }
        }
        r => {
            cx.sess.bug(format!("Variable id {} maps to {:?}, not local",
                                id,
                                r).as_slice());
        }
    }
}

pub fn adjust_ty(cx: &ctxt,
                 span: Span,
                 expr_id: ast::NodeId,
                 unadjusted_ty: ty::t,
                 adjustment: Option<&AutoAdjustment>,
                 method_type: |typeck::MethodCall| -> Option<ty::t>)
                 -> ty::t {
    /*! See `expr_ty_adjusted` */

    match get(unadjusted_ty).sty {
        ty_err => return unadjusted_ty,
        _ => {}
    }

    return match adjustment {
        Some(adjustment) => {
            match *adjustment {
                AdjustAddEnv(store) => {
                    match ty::get(unadjusted_ty).sty {
                        ty::ty_bare_fn(ref b) => {
                            let bounds = ty::ExistentialBounds {
                                region_bound: ReStatic,
                                builtin_bounds: all_builtin_bounds(),
                            };

                            ty::mk_closure(
                                cx,
                                ty::ClosureTy {fn_style: b.fn_style,
                                               onceness: ast::Many,
                                               store: store,
                                               bounds: bounds,
                                               sig: b.sig.clone(),
                                               abi: b.abi})
                        }
                        ref b => {
                            cx.sess.bug(
                                format!("add_env adjustment on non-bare-fn: \
                                         {:?}",
                                        b).as_slice());
                        }
                    }
                }

                AdjustDerefRef(ref adj) => {
                    let mut adjusted_ty = unadjusted_ty;

                    if !ty::type_is_error(adjusted_ty) {
                        for i in range(0, adj.autoderefs) {
                            let method_call = typeck::MethodCall::autoderef(expr_id, i);
                            match method_type(method_call) {
                                Some(method_ty) => {
                                    adjusted_ty = ty_fn_ret(method_ty);
                                }
                                None => {}
                            }
                            match deref(adjusted_ty, true) {
                                Some(mt) => { adjusted_ty = mt.ty; }
                                None => {
                                    cx.sess.span_bug(
                                        span,
                                        format!("the {}th autoderef failed: \
                                                {}",
                                                i,
                                                ty_to_string(cx, adjusted_ty))
                                                          .as_slice());
                                }
                            }
                        }
                    }

                    match adj.autoref {
                        None => adjusted_ty,
                        Some(ref autoref) => adjust_for_autoref(cx, span, adjusted_ty, autoref)
                    }
                }
            }
        }
        None => unadjusted_ty
    };

    fn adjust_for_autoref(cx: &ctxt,
                          span: Span,
                          ty: ty::t,
                          autoref: &AutoRef) -> ty::t{
        match *autoref {
            AutoPtr(r, m, ref a) => {
                let adjusted_ty = match a {
                    &Some(box ref a) => adjust_for_autoref(cx, span, ty, a),
                    &None => ty
                };
                mk_rptr(cx, r, mt {
                    ty: adjusted_ty,
                    mutbl: m
                })
            }

            AutoUnsafe(m, ref a) => {
                let adjusted_ty = match a {
                    &Some(box ref a) => adjust_for_autoref(cx, span, ty, a),
                    &None => ty
                };
                mk_ptr(cx, mt {ty: adjusted_ty, mutbl: m})
            }

            AutoUnsize(ref k) => unsize_ty(cx, ty, k, span),
            AutoUnsizeUniq(ref k) => ty::mk_uniq(cx, unsize_ty(cx, ty, k, span)),
        }
    }
}

// Take a sized type and a sizing adjustment and produce an unsized version of
// the type.
pub fn unsize_ty(cx: &ctxt,
                 ty: ty::t,
                 kind: &UnsizeKind,
                 span: Span)
                 -> ty::t {
    match kind {
        &UnsizeLength(len) => match get(ty).sty {
            ty_vec(t, Some(n)) => {
                assert!(len == n);
                mk_vec(cx, t, None)
            }
            _ => cx.sess.span_bug(span,
                                  format!("UnsizeLength with bad sty: {}",
                                          ty_to_string(cx, ty)).as_slice())
        },
        &UnsizeStruct(box ref k, tp_index) => match get(ty).sty {
            ty_struct(did, ref substs) => {
                let ty_substs = substs.types.get_slice(subst::TypeSpace);
                let new_ty = unsize_ty(cx, ty_substs[tp_index], k, span);
                let mut unsized_substs = substs.clone();
                unsized_substs.types.get_mut_slice(subst::TypeSpace)[tp_index] = new_ty;
                mk_struct(cx, did, unsized_substs)
            }
            _ => cx.sess.span_bug(span,
                                  format!("UnsizeStruct with bad sty: {}",
                                          ty_to_string(cx, ty)).as_slice())
        },
        &UnsizeVtable(TyTrait { def_id, substs: ref substs, bounds }, _) => {
            mk_trait(cx, def_id, substs.clone(), bounds)
        }
    }
}

pub fn resolve_expr(tcx: &ctxt, expr: &ast::Expr) -> def::Def {
    match tcx.def_map.borrow().find(&expr.id) {
        Some(&def) => def,
        None => {
            tcx.sess.span_bug(expr.span, format!(
                "no def-map entry for expr {:?}", expr.id).as_slice());
        }
    }
}

pub fn expr_is_lval(tcx: &ctxt, e: &ast::Expr) -> bool {
    match expr_kind(tcx, e) {
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

pub fn expr_kind(tcx: &ctxt, expr: &ast::Expr) -> ExprKind {
    if tcx.method_map.borrow().contains_key(&typeck::MethodCall::expr(expr.id)) {
        // Overloaded operations are generally calls, and hence they are
        // generated via DPS, but there are a few exceptions:
        return match expr.node {
            // `a += b` has a unit result.
            ast::ExprAssignOp(..) => RvalueStmtExpr,

            // the deref method invoked for `*a` always yields an `&T`
            ast::ExprUnary(ast::UnDeref, _) => LvalueExpr,

            // the index method invoked for `a[i]` always yields an `&T`
            ast::ExprIndex(..) => LvalueExpr,

            // the slice method invoked for `a[..]` always yields an `&T`
            ast::ExprSlice(..) => LvalueExpr,

            // `for` loops are statements
            ast::ExprForLoop(..) => RvalueStmtExpr,

            // in the general case, result could be any type, use DPS
            _ => RvalueDpsExpr
        };
    }

    match expr.node {
        ast::ExprPath(..) => {
            match resolve_expr(tcx, expr) {
                def::DefVariant(tid, vid, _) => {
                    let variant_info = enum_variant_with_id(tcx, tid, vid);
                    if variant_info.args.len() > 0u {
                        // N-ary variant.
                        RvalueDatumExpr
                    } else {
                        // Nullary variant.
                        RvalueDpsExpr
                    }
                }

                def::DefStruct(_) => {
                    match get(expr_ty(tcx, expr)).sty {
                        ty_bare_fn(..) => RvalueDatumExpr,
                        _ => RvalueDpsExpr
                    }
                }

                // Special case: A unit like struct's constructor must be called without () at the
                // end (like `UnitStruct`) which means this is an ExprPath to a DefFn. But in case
                // of unit structs this is should not be interpretet as function pointer but as
                // call to the constructor.
                def::DefFn(_, _, true) => RvalueDpsExpr,

                // Fn pointers are just scalar values.
                def::DefFn(..) | def::DefStaticMethod(..) => RvalueDatumExpr,

                // Note: there is actually a good case to be made that
                // DefArg's, particularly those of immediate type, ought to
                // considered rvalues.
                def::DefStatic(..) |
                def::DefUpvar(..) |
                def::DefLocal(..) => LvalueExpr,

                def => {
                    tcx.sess.span_bug(
                        expr.span,
                        format!("uncategorized def for expr {:?}: {:?}",
                                expr.id,
                                def).as_slice());
                }
            }
        }

        ast::ExprUnary(ast::UnDeref, _) |
        ast::ExprField(..) |
        ast::ExprTupField(..) |
        ast::ExprIndex(..) |
        ast::ExprSlice(..) => {
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
        ast::ExprUnboxedFn(..) |
        ast::ExprBlock(..) |
        ast::ExprRepeat(..) |
        ast::ExprVec(..) => {
            RvalueDpsExpr
        }

        ast::ExprIfLet(..) => {
            tcx.sess.span_bug(expr.span, "non-desugared ExprIfLet");
        }

        ast::ExprLit(ref lit) if lit_is_str(&**lit) => {
            RvalueDpsExpr
        }

        ast::ExprCast(..) => {
            match tcx.node_types.borrow().find(&(expr.id as uint)) {
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
                    // easier in the future, when casts to traits
                    // would like @Foo, Box<Foo>, or &Foo.
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
        ast::ExprAssignOp(..) |
        ast::ExprForLoop(..) => {
            RvalueStmtExpr
        }

        ast::ExprLit(_) | // Note: LitStr is carved out above
        ast::ExprUnary(..) |
        ast::ExprAddrOf(..) |
        ast::ExprBinary(..) => {
            RvalueDatumExpr
        }

        ast::ExprBox(ref place, _) => {
            // Special case `Box<T>` for now:
            let definition = match tcx.def_map.borrow().find(&place.id) {
                Some(&def) => def,
                None => fail!("no def for place"),
            };
            let def_id = definition.def_id();
            if tcx.lang_items.exchange_heap() == Some(def_id) {
                RvalueDatumExpr
            } else {
                RvalueDpsExpr
            }
        }

        ast::ExprParen(ref e) => expr_kind(tcx, &**e),

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

pub fn field_idx_strict(tcx: &ctxt, name: ast::Name, fields: &[field])
                     -> uint {
    let mut i = 0u;
    for f in fields.iter() { if f.ident.name == name { return i; } i += 1u; }
    tcx.sess.bug(format!(
        "no field named `{}` found in the list of fields `{:?}`",
        token::get_name(name),
        fields.iter()
              .map(|f| token::get_ident(f.ident).get().to_string())
              .collect::<Vec<String>>()).as_slice());
}

pub fn impl_or_trait_item_idx(id: ast::Ident, trait_items: &[ImplOrTraitItem])
                              -> Option<uint> {
    trait_items.iter().position(|m| m.ident() == id)
}

pub fn ty_sort_string(cx: &ctxt, t: t) -> String {
    match get(t).sty {
        ty_nil | ty_bot | ty_bool | ty_char | ty_int(_) |
        ty_uint(_) | ty_float(_) | ty_str => {
            ::util::ppaux::ty_to_string(cx, t)
        }

        ty_enum(id, _) => format!("enum {}", item_path_str(cx, id)),
        ty_uniq(_) => "box".to_string(),
        ty_vec(_, Some(_)) => "array".to_string(),
        ty_vec(_, None) => "unsized array".to_string(),
        ty_ptr(_) => "*-ptr".to_string(),
        ty_rptr(_, _) => "&-ptr".to_string(),
        ty_bare_fn(_) => "extern fn".to_string(),
        ty_closure(_) => "fn".to_string(),
        ty_trait(ref inner) => {
            format!("trait {}", item_path_str(cx, inner.def_id))
        }
        ty_struct(id, _) => {
            format!("struct {}", item_path_str(cx, id))
        }
        ty_unboxed_closure(..) => "closure".to_string(),
        ty_tup(_) => "tuple".to_string(),
        ty_infer(TyVar(_)) => "inferred type".to_string(),
        ty_infer(IntVar(_)) => "integral variable".to_string(),
        ty_infer(FloatVar(_)) => "floating-point variable".to_string(),
        ty_infer(SkolemizedTy(_)) => "skolemized type".to_string(),
        ty_infer(SkolemizedIntTy(_)) => "skolemized integral type".to_string(),
        ty_param(ref p) => {
            if p.space == subst::SelfSpace {
                "Self".to_string()
            } else {
                "type parameter".to_string()
            }
        }
        ty_err => "type error".to_string(),
        ty_open(_) => "opened DST".to_string(),
    }
}

pub fn type_err_to_str(cx: &ctxt, err: &type_err) -> String {
    /*!
     *
     * Explains the source of a type err in a short,
     * human readable way.  This is meant to be placed in
     * parentheses after some larger message.  You should
     * also invoke `note_and_explain_type_err()` afterwards
     * to present additional details, particularly when
     * it comes to lifetime-related errors. */

    fn tstore_to_closure(s: &TraitStore) -> String {
        match s {
            &UniqTraitStore => "proc".to_string(),
            &RegionTraitStore(..) => "closure".to_string()
        }
    }

    match *err {
        terr_cyclic_ty => "cyclic type of infinite size".to_string(),
        terr_mismatch => "types differ".to_string(),
        terr_fn_style_mismatch(values) => {
            format!("expected {} fn, found {} fn",
                    values.expected.to_string(),
                    values.found.to_string())
        }
        terr_abi_mismatch(values) => {
            format!("expected {} fn, found {} fn",
                    values.expected.to_string(),
                    values.found.to_string())
        }
        terr_onceness_mismatch(values) => {
            format!("expected {} fn, found {} fn",
                    values.expected.to_string(),
                    values.found.to_string())
        }
        terr_sigil_mismatch(values) => {
            format!("expected {}, found {}",
                    tstore_to_closure(&values.expected),
                    tstore_to_closure(&values.found))
        }
        terr_mutability => "values differ in mutability".to_string(),
        terr_box_mutability => {
            "boxed values differ in mutability".to_string()
        }
        terr_vec_mutability => "vectors differ in mutability".to_string(),
        terr_ptr_mutability => "pointers differ in mutability".to_string(),
        terr_ref_mutability => "references differ in mutability".to_string(),
        terr_ty_param_size(values) => {
            format!("expected a type with {} type params, \
                     found one with {} type params",
                    values.expected,
                    values.found)
        }
        terr_tuple_size(values) => {
            format!("expected a tuple with {} elements, \
                     found one with {} elements",
                    values.expected,
                    values.found)
        }
        terr_record_size(values) => {
            format!("expected a record with {} fields, \
                     found one with {} fields",
                    values.expected,
                    values.found)
        }
        terr_record_mutability => {
            "record elements differ in mutability".to_string()
        }
        terr_record_fields(values) => {
            format!("expected a record with field `{}`, found one \
                     with field `{}`",
                    token::get_ident(values.expected),
                    token::get_ident(values.found))
        }
        terr_arg_count => {
            "incorrect number of function parameters".to_string()
        }
        terr_regions_does_not_outlive(..) => {
            "lifetime mismatch".to_string()
        }
        terr_regions_not_same(..) => {
            "lifetimes are not the same".to_string()
        }
        terr_regions_no_overlap(..) => {
            "lifetimes do not intersect".to_string()
        }
        terr_regions_insufficiently_polymorphic(br, _) => {
            format!("expected bound lifetime parameter {}, \
                     found concrete lifetime",
                    bound_region_ptr_to_string(cx, br))
        }
        terr_regions_overly_polymorphic(br, _) => {
            format!("expected concrete lifetime, \
                     found bound lifetime parameter {}",
                    bound_region_ptr_to_string(cx, br))
        }
        terr_trait_stores_differ(_, ref values) => {
            format!("trait storage differs: expected `{}`, found `{}`",
                    trait_store_to_string(cx, (*values).expected),
                    trait_store_to_string(cx, (*values).found))
        }
        terr_sorts(values) => {
            format!("expected {}, found {}",
                    ty_sort_string(cx, values.expected),
                    ty_sort_string(cx, values.found))
        }
        terr_traits(values) => {
            format!("expected trait `{}`, found trait `{}`",
                    item_path_str(cx, values.expected),
                    item_path_str(cx, values.found))
        }
        terr_builtin_bounds(values) => {
            if values.expected.is_empty() {
                format!("expected no bounds, found `{}`",
                        values.found.user_string(cx))
            } else if values.found.is_empty() {
                format!("expected bounds `{}`, found no bounds",
                        values.expected.user_string(cx))
            } else {
                format!("expected bounds `{}`, found bounds `{}`",
                        values.expected.user_string(cx),
                        values.found.user_string(cx))
            }
        }
        terr_integer_as_char => {
            "expected an integral type, found `char`".to_string()
        }
        terr_int_mismatch(ref values) => {
            format!("expected `{}`, found `{}`",
                    values.expected.to_string(),
                    values.found.to_string())
        }
        terr_float_mismatch(ref values) => {
            format!("expected `{}`, found `{}`",
                    values.expected.to_string(),
                    values.found.to_string())
        }
        terr_variadic_mismatch(ref values) => {
            format!("expected {} fn, found {} function",
                    if values.expected { "variadic" } else { "non-variadic" },
                    if values.found { "variadic" } else { "non-variadic" })
        }
    }
}

pub fn note_and_explain_type_err(cx: &ctxt, err: &type_err) {
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

pub fn provided_source(cx: &ctxt, id: ast::DefId) -> Option<ast::DefId> {
    cx.provided_method_sources.borrow().find(&id).map(|x| *x)
}

pub fn provided_trait_methods(cx: &ctxt, id: ast::DefId) -> Vec<Rc<Method>> {
    if is_local(id) {
        match cx.map.find(id.node) {
            Some(ast_map::NodeItem(item)) => {
                match item.node {
                    ItemTrait(_, _, _, ref ms) => {
                        let (_, p) =
                            ast_util::split_trait_methods(ms.as_slice());
                        p.iter()
                         .map(|m| {
                            match impl_or_trait_item(
                                    cx,
                                    ast_util::local_def(m.id)) {
                                MethodTraitItem(m) => m,
                                TypeTraitItem(_) => {
                                    cx.sess.bug("provided_trait_methods(): \
                                                 split_trait_methods() put \
                                                 associated types in the \
                                                 provided method bucket?!")
                                }
                            }
                         }).collect()
                    }
                    _ => {
                        cx.sess.bug(format!("provided_trait_methods: `{}` is \
                                             not a trait",
                                            id).as_slice())
                    }
                }
            }
            _ => {
                cx.sess.bug(format!("provided_trait_methods: `{}` is not a \
                                     trait",
                                    id).as_slice())
            }
        }
    } else {
        csearch::get_provided_trait_methods(cx, id)
    }
}

fn lookup_locally_or_in_crate_store<V:Clone>(
                                    descr: &str,
                                    def_id: ast::DefId,
                                    map: &mut DefIdMap<V>,
                                    load_external: || -> V) -> V {
    /*!
     * Helper for looking things up in the various maps
     * that are populated during typeck::collect (e.g.,
     * `cx.impl_or_trait_items`, `cx.tcache`, etc).  All of these share
     * the pattern that if the id is local, it should have
     * been loaded into the map by the `typeck::collect` phase.
     * If the def-id is external, then we have to go consult
     * the crate loading code (and cache the result for the future).
     */

    match map.find_copy(&def_id) {
        Some(v) => { return v; }
        None => { }
    }

    if def_id.krate == ast::LOCAL_CRATE {
        fail!("No def'n found for {:?} in tcx.{}", def_id, descr);
    }
    let v = load_external();
    map.insert(def_id, v.clone());
    v
}

pub fn trait_item(cx: &ctxt, trait_did: ast::DefId, idx: uint)
                  -> ImplOrTraitItem {
    let method_def_id = ty::trait_item_def_ids(cx, trait_did).get(idx)
                                                             .def_id();
    impl_or_trait_item(cx, method_def_id)
}

pub fn trait_items(cx: &ctxt, trait_did: ast::DefId)
                   -> Rc<Vec<ImplOrTraitItem>> {
    let mut trait_items = cx.trait_items_cache.borrow_mut();
    match trait_items.find_copy(&trait_did) {
        Some(trait_items) => trait_items,
        None => {
            let def_ids = ty::trait_item_def_ids(cx, trait_did);
            let items: Rc<Vec<ImplOrTraitItem>> =
                Rc::new(def_ids.iter()
                               .map(|d| impl_or_trait_item(cx, d.def_id()))
                               .collect());
            trait_items.insert(trait_did, items.clone());
            items
        }
    }
}

pub fn impl_or_trait_item(cx: &ctxt, id: ast::DefId) -> ImplOrTraitItem {
    lookup_locally_or_in_crate_store("impl_or_trait_items",
                                     id,
                                     &mut *cx.impl_or_trait_items
                                             .borrow_mut(),
                                     || {
        csearch::get_impl_or_trait_item(cx, id)
    })
}

/// Returns true if the given ID refers to an associated type and false if it
/// refers to anything else.
pub fn is_associated_type(cx: &ctxt, id: ast::DefId) -> bool {
    let result = match cx.associated_types.borrow_mut().find(&id) {
        Some(result) => return *result,
        None if id.krate == ast::LOCAL_CRATE => {
            match cx.impl_or_trait_items.borrow().find(&id) {
                Some(ref item) => {
                    match **item {
                        TypeTraitItem(_) => true,
                        MethodTraitItem(_) => false,
                    }
                }
                None => false,
            }
        }
        None => {
            csearch::is_associated_type(&cx.sess.cstore, id)
        }
    };

    cx.associated_types.borrow_mut().insert(id, result);
    result
}

/// Returns the parameter index that the given associated type corresponds to.
pub fn associated_type_parameter_index(cx: &ctxt,
                                       trait_def: &TraitDef,
                                       associated_type_id: ast::DefId)
                                       -> uint {
    for type_parameter_def in trait_def.generics.types.iter() {
        if type_parameter_def.def_id == associated_type_id {
            return type_parameter_def.index
        }
    }
    cx.sess.bug("couldn't find associated type parameter index")
}

#[deriving(PartialEq, Eq)]
pub struct AssociatedTypeInfo {
    pub def_id: ast::DefId,
    pub index: uint,
    pub ident: ast::Ident,
}

impl PartialOrd for AssociatedTypeInfo {
    fn partial_cmp(&self, other: &AssociatedTypeInfo) -> Option<Ordering> {
        Some(self.index.cmp(&other.index))
    }
}

impl Ord for AssociatedTypeInfo {
    fn cmp(&self, other: &AssociatedTypeInfo) -> Ordering {
        self.index.cmp(&other.index)
    }
}

/// Returns the associated types belonging to the given trait, in parameter
/// order.
pub fn associated_types_for_trait(cx: &ctxt, trait_id: ast::DefId)
                                  -> Rc<Vec<AssociatedTypeInfo>> {
    cx.trait_associated_types
      .borrow()
      .find(&trait_id)
      .expect("associated_types_for_trait(): trait not found, try calling \
               ensure_associated_types()")
      .clone()
}

pub fn trait_item_def_ids(cx: &ctxt, id: ast::DefId)
                          -> Rc<Vec<ImplOrTraitItemId>> {
    lookup_locally_or_in_crate_store("trait_item_def_ids",
                                     id,
                                     &mut *cx.trait_item_def_ids.borrow_mut(),
                                     || {
        Rc::new(csearch::get_trait_item_def_ids(&cx.sess.cstore, id))
    })
}

pub fn impl_trait_ref(cx: &ctxt, id: ast::DefId) -> Option<Rc<TraitRef>> {
    match cx.impl_trait_cache.borrow().find(&id) {
        Some(ret) => { return ret.clone(); }
        None => {}
    }

    let ret = if id.krate == ast::LOCAL_CRATE {
        debug!("(impl_trait_ref) searching for trait impl {:?}", id);
        match cx.map.find(id.node) {
            Some(ast_map::NodeItem(item)) => {
                match item.node {
                    ast::ItemImpl(_, ref opt_trait, _, _) => {
                        match opt_trait {
                            &Some(ref t) => {
                                Some(ty::node_id_to_trait_ref(cx, t.ref_id))
                            }
                            &None => None
                        }
                    }
                    _ => None
                }
            }
            _ => None
        }
    } else {
        csearch::get_impl_trait(cx, id)
    };

    cx.impl_trait_cache.borrow_mut().insert(id, ret.clone());
    ret
}

pub fn trait_ref_to_def_id(tcx: &ctxt, tr: &ast::TraitRef) -> ast::DefId {
    let def = *tcx.def_map.borrow()
                     .find(&tr.ref_id)
                     .expect("no def-map entry for trait");
    def.def_id()
}

pub fn try_add_builtin_trait(
    tcx: &ctxt,
    trait_def_id: ast::DefId,
    builtin_bounds: &mut EnumSet<BuiltinBound>)
    -> bool
{
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
        ty_trait(box TyTrait { def_id: id, .. }) |
        ty_struct(id, _) |
        ty_enum(id, _) |
        ty_unboxed_closure(id, _) => Some(id),
        _ => None
    }
}

// Enum information
#[deriving(Clone)]
pub struct VariantInfo {
    pub args: Vec<t>,
    pub arg_names: Option<Vec<ast::Ident> >,
    pub ctor_ty: t,
    pub name: ast::Ident,
    pub id: ast::DefId,
    pub disr_val: Disr,
    pub vis: Visibility
}

impl VariantInfo {

    /// Creates a new VariantInfo from the corresponding ast representation.
    ///
    /// Does not do any caching of the value in the type context.
    pub fn from_ast_variant(cx: &ctxt,
                            ast_variant: &ast::Variant,
                            discriminant: Disr) -> VariantInfo {
        let ctor_ty = node_id_to_type(cx, ast_variant.node.id);

        match ast_variant.node.kind {
            ast::TupleVariantKind(ref args) => {
                let arg_tys = if args.len() > 0 {
                    ty_fn_args(ctor_ty).iter().map(|a| *a).collect()
                } else {
                    Vec::new()
                };

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

                let fields: &[StructField] = struct_def.fields.as_slice();

                assert!(fields.len() > 0);

                let arg_tys = ty_fn_args(ctor_ty).iter().map(|a| *a).collect();
                let arg_names = fields.iter().map(|field| {
                    match field.node.kind {
                        NamedField(ident, _) => ident,
                        UnnamedField(..) => cx.sess.bug(
                            "enum_variants: all fields in struct must have a name")
                    }
                }).collect();

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

pub fn substd_enum_variants(cx: &ctxt,
                            id: ast::DefId,
                            substs: &Substs)
                         -> Vec<Rc<VariantInfo>> {
    enum_variants(cx, id).iter().map(|variant_info| {
        let substd_args = variant_info.args.iter()
            .map(|aty| aty.subst(cx, substs)).collect::<Vec<_>>();

        let substd_ctor_ty = variant_info.ctor_ty.subst(cx, substs);

        Rc::new(VariantInfo {
            args: substd_args,
            ctor_ty: substd_ctor_ty,
            ..(**variant_info).clone()
        })
    }).collect()
}

pub fn item_path_str(cx: &ctxt, id: ast::DefId) -> String {
    with_path(cx, id, |path| ast_map::path_to_string(path)).to_string()
}

pub enum DtorKind {
    NoDtor,
    TraitDtor(DefId, bool)
}

impl DtorKind {
    pub fn is_present(&self) -> bool {
        match *self {
            TraitDtor(..) => true,
            _ => false
        }
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
pub fn ty_dtor(cx: &ctxt, struct_id: DefId) -> DtorKind {
    match cx.destructor_for_type.borrow().find(&struct_id) {
        Some(&method_def_id) => {
            let flag = !has_attr(cx, struct_id, "unsafe_no_drop_flag");

            TraitDtor(method_def_id, flag)
        }
        None => NoDtor,
    }
}

pub fn has_dtor(cx: &ctxt, struct_id: DefId) -> bool {
    cx.destructor_for_type.borrow().contains_key(&struct_id)
}

pub fn with_path<T>(cx: &ctxt, id: ast::DefId, f: |ast_map::PathElems| -> T) -> T {
    if id.krate == ast::LOCAL_CRATE {
        cx.map.with_path(id.node, f)
    } else {
        f(ast_map::Values(csearch::get_item_path(cx, id).iter()).chain(None))
    }
}

pub fn enum_is_univariant(cx: &ctxt, id: ast::DefId) -> bool {
    enum_variants(cx, id).len() == 1
}

pub fn type_is_empty(cx: &ctxt, t: t) -> bool {
    match ty::get(t).sty {
       ty_enum(did, _) => (*enum_variants(cx, did)).is_empty(),
       _ => false
     }
}

pub fn enum_variants(cx: &ctxt, id: ast::DefId) -> Rc<Vec<Rc<VariantInfo>>> {
    match cx.enum_var_cache.borrow().find(&id) {
        Some(variants) => return variants.clone(),
        _ => { /* fallthrough */ }
    }

    let result = if ast::LOCAL_CRATE != id.krate {
        Rc::new(csearch::get_enum_variants(cx, id))
    } else {
        /*
          Although both this code and check_enum_variants in typeck/check
          call eval_const_expr, it should never get called twice for the same
          expr, since check_enum_variants also updates the enum_var_cache
         */
        match cx.map.get(id.node) {
            ast_map::NodeItem(ref item) => {
                match item.node {
                    ast::ItemEnum(ref enum_definition, _) => {
                        let mut last_discriminant: Option<Disr> = None;
                        Rc::new(enum_definition.variants.iter().map(|variant| {

                            let mut discriminant = match last_discriminant {
                                Some(val) => val + 1,
                                None => INITIAL_DISCRIMINANT_VALUE
                            };

                            match variant.node.disr_expr {
                                Some(ref e) => match const_eval::eval_const_expr_partial(cx, &**e) {
                                    Ok(const_eval::const_int(val)) => {
                                        discriminant = val as Disr
                                    }
                                    Ok(const_eval::const_uint(val)) => {
                                        discriminant = val as Disr
                                    }
                                    Ok(_) => {
                                        cx.sess
                                          .span_err(e.span,
                                                    "expected signed integer constant");
                                    }
                                    Err(ref err) => {
                                        cx.sess
                                          .span_err(e.span,
                                                    format!("expected constant: {}",
                                                            *err).as_slice());
                                    }
                                },
                                None => {}
                            };

                            last_discriminant = Some(discriminant);
                            Rc::new(VariantInfo::from_ast_variant(cx, &**variant,
                                                                  discriminant))
                        }).collect())
                    }
                    _ => {
                        cx.sess.bug("enum_variants: id not bound to an enum")
                    }
                }
            }
            _ => cx.sess.bug("enum_variants: id not bound to an enum")
        }
    };

    cx.enum_var_cache.borrow_mut().insert(id, result.clone());
    result
}


// Returns information about the enum variant with the given ID:
pub fn enum_variant_with_id(cx: &ctxt,
                            enum_id: ast::DefId,
                            variant_id: ast::DefId)
                         -> Rc<VariantInfo> {
    enum_variants(cx, enum_id).iter()
                              .find(|variant| variant.id == variant_id)
                              .expect("enum_variant_with_id(): no variant exists with that ID")
                              .clone()
}


// If the given item is in an external crate, looks up its type and adds it to
// the type cache. Returns the type parameters and type.
pub fn lookup_item_type(cx: &ctxt,
                        did: ast::DefId)
                     -> Polytype {
    lookup_locally_or_in_crate_store(
        "tcache", did, &mut *cx.tcache.borrow_mut(),
        || csearch::get_type(cx, did))
}

/// Given the did of a trait, returns its canonical trait ref.
pub fn lookup_trait_def(cx: &ctxt, did: ast::DefId) -> Rc<ty::TraitDef> {
    let mut trait_defs = cx.trait_defs.borrow_mut();
    match trait_defs.find_copy(&did) {
        Some(trait_def) => {
            // The item is in this crate. The caller should have added it to the
            // type cache already
            trait_def
        }
        None => {
            assert!(did.krate != ast::LOCAL_CRATE);
            let trait_def = Rc::new(csearch::get_trait_def(cx, did));
            trait_defs.insert(did, trait_def.clone());
            trait_def
        }
    }
}

/// Given a reference to a trait, returns the bounds declared on the
/// trait, with appropriate substitutions applied.
pub fn bounds_for_trait_ref(tcx: &ctxt,
                            trait_ref: &TraitRef)
                            -> ty::ParamBounds
{
    let trait_def = lookup_trait_def(tcx, trait_ref.def_id);
    debug!("bounds_for_trait_ref(trait_def={}, trait_ref={})",
           trait_def.repr(tcx), trait_ref.repr(tcx));
    trait_def.bounds.subst(tcx, &trait_ref.substs)
}

/// Iterate over attributes of a definition.
// (This should really be an iterator, but that would require csearch and
// decoder to use iterators instead of higher-order functions.)
pub fn each_attr(tcx: &ctxt, did: DefId, f: |&ast::Attribute| -> bool) -> bool {
    if is_local(did) {
        let item = tcx.map.expect_item(did.node);
        item.attrs.iter().all(|attr| f(attr))
    } else {
        info!("getting foreign attrs");
        let mut cont = true;
        csearch::get_item_attrs(&tcx.sess.cstore, did, |attrs| {
            if cont {
                cont = attrs.iter().all(|attr| f(attr));
            }
        });
        info!("done");
        cont
    }
}

/// Determine whether an item is annotated with an attribute
pub fn has_attr(tcx: &ctxt, did: DefId, attr: &str) -> bool {
    let mut found = false;
    each_attr(tcx, did, |item| {
        if item.check_name(attr) {
            found = true;
            false
        } else {
            true
        }
    });
    found
}

/// Determine whether an item is annotated with `#[repr(packed)]`
pub fn lookup_packed(tcx: &ctxt, did: DefId) -> bool {
    lookup_repr_hints(tcx, did).contains(&attr::ReprPacked)
}

/// Determine whether an item is annotated with `#[simd]`
pub fn lookup_simd(tcx: &ctxt, did: DefId) -> bool {
    has_attr(tcx, did, "simd")
}

/// Obtain the representation annotation for a struct definition.
pub fn lookup_repr_hints(tcx: &ctxt, did: DefId) -> Rc<Vec<attr::ReprAttr>> {
    match tcx.repr_hint_cache.borrow().find(&did) {
        None => {}
        Some(ref hints) => return (*hints).clone(),
    }

    let acc = if did.krate == LOCAL_CRATE {
        let mut acc = Vec::new();
        ty::each_attr(tcx, did, |meta| {
            acc.extend(attr::find_repr_attrs(tcx.sess.diagnostic(),
                                             meta).into_iter());
            true
        });
        acc
    } else {
        csearch::get_repr_attrs(&tcx.sess.cstore, did)
    };

    let acc = Rc::new(acc);
    tcx.repr_hint_cache.borrow_mut().insert(did, acc.clone());
    acc
}

// Look up a field ID, whether or not it's local
// Takes a list of type substs in case the struct is generic
pub fn lookup_field_type(tcx: &ctxt,
                         struct_id: DefId,
                         id: DefId,
                         substs: &Substs)
                      -> ty::t {
    let t = if id.krate == ast::LOCAL_CRATE {
        node_id_to_type(tcx, id.node)
    } else {
        let mut tcache = tcx.tcache.borrow_mut();
        let pty = match tcache.entry(id) {
            Occupied(entry) => entry.into_mut(),
            Vacant(entry) => entry.set(csearch::get_field_type(tcx, struct_id, id)),
        };
        pty.ty
    };
    t.subst(tcx, substs)
}

// Lookup all ancestor structs of a struct indicated by did. That is the reflexive,
// transitive closure of doing a single lookup in cx.superstructs.
fn each_super_struct(cx: &ctxt, mut did: ast::DefId, f: |ast::DefId|) {
    let superstructs = cx.superstructs.borrow();

    loop {
        f(did);
        match superstructs.find(&did) {
            Some(&Some(def_id)) => {
                did = def_id;
            },
            Some(&None) => break,
            None => {
                cx.sess.bug(
                    format!("ID not mapped to super-struct: {}",
                            cx.map.node_to_string(did.node)).as_slice());
            }
        }
    }
}

// Look up the list of field names and IDs for a given struct.
// Fails if the id is not bound to a struct.
pub fn lookup_struct_fields(cx: &ctxt, did: ast::DefId) -> Vec<field_ty> {
    if did.krate == ast::LOCAL_CRATE {
        // We store the fields which are syntactically in each struct in cx. So
        // we have to walk the inheritance chain of the struct to get all the
        // fields (explicit and inherited) for a struct. If this is expensive
        // we could cache the whole list of fields here.
        let struct_fields = cx.struct_fields.borrow();
        let mut results: SmallVector<&[field_ty]> = SmallVector::zero();
        each_super_struct(cx, did, |s| {
            match struct_fields.find(&s) {
                Some(fields) => results.push(fields.as_slice()),
                _ => {
                    cx.sess.bug(
                        format!("ID not mapped to struct fields: {}",
                                cx.map.node_to_string(did.node)).as_slice());
                }
            }
        });

        let len = results.as_slice().iter().map(|x| x.len()).sum();
        let mut result: Vec<field_ty> = Vec::with_capacity(len);
        result.extend(results.as_slice().iter().flat_map(|rs| rs.iter().map(|f| f.clone())));
        assert!(result.len() == len);
        result
    } else {
        csearch::get_struct_fields(&cx.sess.cstore, did)
    }
}

pub fn is_tuple_struct(cx: &ctxt, did: ast::DefId) -> bool {
    let fields = lookup_struct_fields(cx, did);
    !fields.is_empty() && fields.iter().all(|f| f.name == token::special_names::unnamed_field)
}

// Returns a list of fields corresponding to the struct's items. trans uses
// this. Takes a list of substs with which to instantiate field types.
pub fn struct_fields(cx: &ctxt, did: ast::DefId, substs: &Substs)
                     -> Vec<field> {
    lookup_struct_fields(cx, did).iter().map(|f| {
       field {
            // FIXME #6993: change type of field to Name and get rid of new()
            ident: ast::Ident::new(f.name),
            mt: mt {
                ty: lookup_field_type(cx, did, f.id, substs),
                mutbl: MutImmutable
            }
        }
    }).collect()
}

// Returns a list of fields corresponding to the tuple's items. trans uses
// this.
pub fn tup_fields(v: &[t]) -> Vec<field> {
    v.iter().enumerate().map(|(i, &f)| {
       field {
            // FIXME #6993: change type of field to Name and get rid of new()
            ident: ast::Ident::new(token::intern(i.to_string().as_slice())),
            mt: mt {
                ty: f,
                mutbl: MutImmutable
            }
        }
    }).collect()
}

pub struct UnboxedClosureUpvar {
    pub def: def::Def,
    pub span: Span,
    pub ty: t,
}

// Returns a list of `UnboxedClosureUpvar`s for each upvar.
pub fn unboxed_closure_upvars(tcx: &ctxt, closure_id: ast::DefId)
                              -> Vec<UnboxedClosureUpvar> {
    if closure_id.krate == ast::LOCAL_CRATE {
        match tcx.freevars.borrow().find(&closure_id.node) {
            None => vec![],
            Some(ref freevars) => {
                freevars.iter().map(|freevar| {
                    let freevar_def_id = freevar.def.def_id();
                    UnboxedClosureUpvar {
                        def: freevar.def,
                        span: freevar.span,
                        ty: node_id_to_type(tcx, freevar_def_id.node),
                    }
                }).collect()
            }
        }
    } else {
        tcx.sess.bug("unimplemented cross-crate closure upvars")
    }
}

pub fn is_binopable(cx: &ctxt, ty: t, op: ast::BinOp) -> bool {
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
    static opcat_mod: int = 8;

    fn opcat(op: ast::BinOp) -> int {
        match op {
          ast::BiAdd => opcat_add,
          ast::BiSub => opcat_sub,
          ast::BiMul => opcat_mult,
          ast::BiDiv => opcat_mult,
          ast::BiRem => opcat_mod,
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

    fn tycat(cx: &ctxt, ty: t) -> int {
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
    //           +, -, *, shift, rel, ==, bit, logic, mod
    /*other*/   [f, f, f, f,     f,   f,  f,   f,     f],
    /*bool*/    [f, f, f, f,     t,   t,  t,   t,     f],
    /*char*/    [f, f, f, f,     t,   t,  f,   f,     f],
    /*int*/     [t, t, t, t,     t,   t,  t,   f,     t],
    /*float*/   [t, t, t, f,     t,   t,  f,   f,     f],
    /*bot*/     [t, t, t, t,     t,   t,  t,   t,     t],
    /*raw ptr*/ [f, f, f, f,     t,   t,  f,   f,     f]];

    return tbl[tycat(cx, ty) as uint ][opcat(op) as uint];
}

/// Returns an equivalent type with all the typedefs and self regions removed.
pub fn normalize_ty(cx: &ctxt, t: t) -> t {
    let u = TypeNormalizer(cx).fold_ty(t);
    return u;

    struct TypeNormalizer<'a, 'tcx: 'a>(&'a ctxt<'tcx>);

    impl<'a, 'tcx> TypeFolder<'tcx> for TypeNormalizer<'a, 'tcx> {
        fn tcx(&self) -> &ctxt<'tcx> { let TypeNormalizer(c) = *self; c }

        fn fold_ty(&mut self, t: ty::t) -> ty::t {
            match self.tcx().normalized_cache.borrow().find_copy(&t) {
                None => {}
                Some(u) => return u
            }

            let t_norm = ty_fold::super_fold_ty(self, t);
            self.tcx().normalized_cache.borrow_mut().insert(t, t_norm);
            return t_norm;
        }

        fn fold_region(&mut self, _: ty::Region) -> ty::Region {
            ty::ReStatic
        }

        fn fold_substs(&mut self,
                       substs: &subst::Substs)
                       -> subst::Substs {
            subst::Substs { regions: subst::ErasedRegions,
                            types: substs.types.fold_with(self) }
        }

        fn fold_sig(&mut self,
                    sig: &ty::FnSig)
                    -> ty::FnSig {
            // The binder-id is only relevant to bound regions, which
            // are erased at trans time.
            ty::FnSig {
                binder_id: ast::DUMMY_NODE_ID,
                inputs: sig.inputs.fold_with(self),
                output: sig.output.fold_with(self),
                variadic: sig.variadic,
            }
        }
    }
}

// Returns the repeat count for a repeating vector expression.
pub fn eval_repeat_count(tcx: &ctxt, count_expr: &ast::Expr) -> uint {
    match const_eval::eval_const_expr_partial(tcx, count_expr) {
      Ok(ref const_val) => match *const_val {
        const_eval::const_int(count) => if count < 0 {
            tcx.sess.span_err(count_expr.span,
                              "expected positive integer for \
                               repeat count, found negative integer");
            0
        } else {
            count as uint
        },
        const_eval::const_uint(count) => count as uint,
        const_eval::const_float(count) => {
            tcx.sess.span_err(count_expr.span,
                              "expected positive integer for \
                               repeat count, found float");
            count as uint
        }
        const_eval::const_str(_) => {
            tcx.sess.span_err(count_expr.span,
                              "expected positive integer for \
                               repeat count, found string");
            0
        }
        const_eval::const_bool(_) => {
            tcx.sess.span_err(count_expr.span,
                              "expected positive integer for \
                               repeat count, found boolean");
            0
        }
        const_eval::const_binary(_) => {
            tcx.sess.span_err(count_expr.span,
                              "expected positive integer for \
                               repeat count, found binary array");
            0
        }
        const_eval::const_nil => {
            tcx.sess.span_err(count_expr.span,
                              "expected positive integer for \
                               repeat count, found ()");
            0
        }
      },
      Err(..) => {
        tcx.sess.span_err(count_expr.span,
                          "expected constant integer for repeat count, \
                           found variable");
        0
      }
    }
}

// Iterate over a type parameter's bounded traits and any supertraits
// of those traits, ignoring kinds.
// Here, the supertraits are the transitive closure of the supertrait
// relation on the supertraits from each bounded trait's constraint
// list.
pub fn each_bound_trait_and_supertraits(tcx: &ctxt,
                                        bounds: &[Rc<TraitRef>],
                                        f: |Rc<TraitRef>| -> bool)
                                        -> bool
{
    for bound_trait_ref in traits::transitive_bounds(tcx, bounds) {
        if !f(bound_trait_ref) {
            return false;
        }
    }
    return true;
}

pub fn required_region_bounds(tcx: &ctxt,
                              region_bounds: &[ty::Region],
                              builtin_bounds: BuiltinBounds,
                              trait_bounds: &[Rc<TraitRef>])
                              -> Vec<ty::Region>
{
    /*!
     * Given a type which must meet the builtin bounds and trait
     * bounds, returns a set of lifetimes which the type must outlive.
     *
     * Requires that trait definitions have been processed.
     */

    let mut all_bounds = Vec::new();

    debug!("required_region_bounds(builtin_bounds={}, trait_bounds={})",
           builtin_bounds.repr(tcx),
           trait_bounds.repr(tcx));

    all_bounds.push_all(region_bounds);

    push_region_bounds([],
                       builtin_bounds,
                       &mut all_bounds);

    debug!("from builtin bounds: all_bounds={}", all_bounds.repr(tcx));

    each_bound_trait_and_supertraits(
        tcx,
        trait_bounds,
        |trait_ref| {
            let bounds = ty::bounds_for_trait_ref(tcx, &*trait_ref);
            push_region_bounds(bounds.region_bounds.as_slice(),
                               bounds.builtin_bounds,
                               &mut all_bounds);
            debug!("from {}: bounds={} all_bounds={}",
                   trait_ref.repr(tcx),
                   bounds.repr(tcx),
                   all_bounds.repr(tcx));
            true
        });

    return all_bounds;

    fn push_region_bounds(region_bounds: &[ty::Region],
                          builtin_bounds: ty::BuiltinBounds,
                          all_bounds: &mut Vec<ty::Region>) {
        all_bounds.push_all(region_bounds.as_slice());

        if builtin_bounds.contains_elem(ty::BoundSend) {
            all_bounds.push(ty::ReStatic);
        }
    }
}

pub fn get_tydesc_ty(tcx: &ctxt) -> Result<t, String> {
    tcx.lang_items.require(TyDescStructLangItem).map(|tydesc_lang_item| {
        tcx.intrinsic_defs.borrow().find_copy(&tydesc_lang_item)
            .expect("Failed to resolve TyDesc")
    })
}

pub fn get_opaque_ty(tcx: &ctxt) -> Result<t, String> {
    tcx.lang_items.require(OpaqueStructLangItem).map(|opaque_lang_item| {
        tcx.intrinsic_defs.borrow().find_copy(&opaque_lang_item)
            .expect("Failed to resolve Opaque")
    })
}

pub fn visitor_object_ty(tcx: &ctxt,
                         ptr_region: ty::Region,
                         trait_region: ty::Region)
                         -> Result<(Rc<TraitRef>, t), String>
{
    let trait_lang_item = match tcx.lang_items.require(TyVisitorTraitLangItem) {
        Ok(id) => id,
        Err(s) => { return Err(s); }
    };
    let substs = Substs::empty();
    let trait_ref = Rc::new(TraitRef { def_id: trait_lang_item, substs: substs });
    Ok((trait_ref.clone(),
        mk_rptr(tcx, ptr_region,
                mt {mutbl: ast::MutMutable,
                    ty: mk_trait(tcx,
                                 trait_ref.def_id,
                                 trait_ref.substs.clone(),
                                 ty::region_existential_bound(trait_region))})))
}

pub fn item_variances(tcx: &ctxt, item_id: ast::DefId) -> Rc<ItemVariances> {
    lookup_locally_or_in_crate_store(
        "item_variance_map", item_id, &mut *tcx.item_variance_map.borrow_mut(),
        || Rc::new(csearch::get_item_variances(&tcx.sess.cstore, item_id)))
}

/// Records a trait-to-implementation mapping.
pub fn record_trait_implementation(tcx: &ctxt,
                                   trait_def_id: DefId,
                                   impl_def_id: DefId) {
    match tcx.trait_impls.borrow().find(&trait_def_id) {
        Some(impls_for_trait) => {
            impls_for_trait.borrow_mut().push(impl_def_id);
            return;
        }
        None => {}
    }
    tcx.trait_impls.borrow_mut().insert(trait_def_id, Rc::new(RefCell::new(vec!(impl_def_id))));
}

/// Populates the type context with all the implementations for the given type
/// if necessary.
pub fn populate_implementations_for_type_if_necessary(tcx: &ctxt,
                                                      type_id: ast::DefId) {
    if type_id.krate == LOCAL_CRATE {
        return
    }
    if tcx.populated_external_types.borrow().contains(&type_id) {
        return
    }

    let mut inherent_impls = Vec::new();
    csearch::each_implementation_for_type(&tcx.sess.cstore, type_id,
            |impl_def_id| {
        let impl_items = csearch::get_impl_items(&tcx.sess.cstore,
                                                 impl_def_id);

        // Record the trait->implementation mappings, if applicable.
        let associated_traits = csearch::get_impl_trait(tcx, impl_def_id);
        for trait_ref in associated_traits.iter() {
            record_trait_implementation(tcx, trait_ref.def_id, impl_def_id);
        }

        // For any methods that use a default implementation, add them to
        // the map. This is a bit unfortunate.
        for impl_item_def_id in impl_items.iter() {
            let method_def_id = impl_item_def_id.def_id();
            match impl_or_trait_item(tcx, method_def_id) {
                MethodTraitItem(method) => {
                    for &source in method.provided_source.iter() {
                        tcx.provided_method_sources
                           .borrow_mut()
                           .insert(method_def_id, source);
                    }
                }
                TypeTraitItem(_) => {}
            }
        }

        // Store the implementation info.
        tcx.impl_items.borrow_mut().insert(impl_def_id, impl_items);

        // If this is an inherent implementation, record it.
        if associated_traits.is_none() {
            inherent_impls.push(impl_def_id);
        }
    });

    tcx.inherent_impls.borrow_mut().insert(type_id, Rc::new(inherent_impls));
    tcx.populated_external_types.borrow_mut().insert(type_id);
}

/// Populates the type context with all the implementations for the given
/// trait if necessary.
pub fn populate_implementations_for_trait_if_necessary(
        tcx: &ctxt,
        trait_id: ast::DefId) {
    if trait_id.krate == LOCAL_CRATE {
        return
    }
    if tcx.populated_external_traits.borrow().contains(&trait_id) {
        return
    }

    csearch::each_implementation_for_trait(&tcx.sess.cstore, trait_id,
            |implementation_def_id| {
        let impl_items = csearch::get_impl_items(&tcx.sess.cstore, implementation_def_id);

        // Record the trait->implementation mapping.
        record_trait_implementation(tcx, trait_id, implementation_def_id);

        // For any methods that use a default implementation, add them to
        // the map. This is a bit unfortunate.
        for impl_item_def_id in impl_items.iter() {
            let method_def_id = impl_item_def_id.def_id();
            match impl_or_trait_item(tcx, method_def_id) {
                MethodTraitItem(method) => {
                    for &source in method.provided_source.iter() {
                        tcx.provided_method_sources
                           .borrow_mut()
                           .insert(method_def_id, source);
                    }
                }
                TypeTraitItem(_) => {}
            }
        }

        // Store the implementation info.
        tcx.impl_items.borrow_mut().insert(implementation_def_id, impl_items);
    });

    tcx.populated_external_traits.borrow_mut().insert(trait_id);
}

/// Given the def_id of an impl, return the def_id of the trait it implements.
/// If it implements no trait, return `None`.
pub fn trait_id_of_impl(tcx: &ctxt,
                        def_id: ast::DefId) -> Option<ast::DefId> {
    let node = match tcx.map.find(def_id.node) {
        Some(node) => node,
        None => return None
    };
    match node {
        ast_map::NodeItem(item) => {
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

/// If the given def ID describes a method belonging to an impl, return the
/// ID of the impl that the method belongs to. Otherwise, return `None`.
pub fn impl_of_method(tcx: &ctxt, def_id: ast::DefId)
                       -> Option<ast::DefId> {
    if def_id.krate != LOCAL_CRATE {
        return match csearch::get_impl_or_trait_item(tcx,
                                                     def_id).container() {
            TraitContainer(_) => None,
            ImplContainer(def_id) => Some(def_id),
        };
    }
    match tcx.impl_or_trait_items.borrow().find_copy(&def_id) {
        Some(trait_item) => {
            match trait_item.container() {
                TraitContainer(_) => None,
                ImplContainer(def_id) => Some(def_id),
            }
        }
        None => None
    }
}

/// If the given def ID describes an item belonging to a trait (either a
/// default method or an implementation of a trait method), return the ID of
/// the trait that the method belongs to. Otherwise, return `None`.
pub fn trait_of_item(tcx: &ctxt, def_id: ast::DefId) -> Option<ast::DefId> {
    if def_id.krate != LOCAL_CRATE {
        return csearch::get_trait_of_item(&tcx.sess.cstore, def_id, tcx);
    }
    match tcx.impl_or_trait_items.borrow().find_copy(&def_id) {
        Some(impl_or_trait_item) => {
            match impl_or_trait_item.container() {
                TraitContainer(def_id) => Some(def_id),
                ImplContainer(def_id) => trait_id_of_impl(tcx, def_id),
            }
        }
        None => None
    }
}

/// If the given def ID describes an item belonging to a trait, (either a
/// default method or an implementation of a trait method), return the ID of
/// the method inside trait definition (this means that if the given def ID
/// is already that of the original trait method, then the return value is
/// the same).
/// Otherwise, return `None`.
pub fn trait_item_of_item(tcx: &ctxt, def_id: ast::DefId)
                          -> Option<ImplOrTraitItemId> {
    let impl_item = match tcx.impl_or_trait_items.borrow().find(&def_id) {
        Some(m) => m.clone(),
        None => return None,
    };
    let name = impl_item.ident().name;
    match trait_of_item(tcx, def_id) {
        Some(trait_did) => {
            let trait_items = ty::trait_items(tcx, trait_did);
            trait_items.iter()
                .position(|m| m.ident().name == name)
                .map(|idx| ty::trait_item(tcx, trait_did, idx).id())
        }
        None => None
    }
}

/// Creates a hash of the type `t` which will be the same no matter what crate
/// context it's calculated within. This is used by the `type_id` intrinsic.
pub fn hash_crate_independent(tcx: &ctxt, t: t, svh: &Svh) -> u64 {
    let mut state = sip::SipState::new();
    macro_rules! byte( ($b:expr) => { ($b as u8).hash(&mut state) } );
    macro_rules! hash( ($e:expr) => { $e.hash(&mut state) } );

    let region = |_state: &mut sip::SipState, r: Region| {
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
    let did = |state: &mut sip::SipState, did: DefId| {
        let h = if ast_util::is_local(did) {
            svh.clone()
        } else {
            tcx.sess.cstore.get_crate_hash(did.krate)
        };
        h.as_str().hash(state);
        did.node.hash(state);
    };
    let mt = |state: &mut sip::SipState, mt: mt| {
        mt.mutbl.hash(state);
    };
    ty::walk_ty(t, |t| {
        match ty::get(t).sty {
            ty_nil => byte!(0),
            ty_bot => byte!(1),
            ty_bool => byte!(2),
            ty_char => byte!(3),
            ty_int(i) => {
                byte!(4);
                hash!(i);
            }
            ty_uint(u) => {
                byte!(5);
                hash!(u);
            }
            ty_float(f) => {
                byte!(6);
                hash!(f);
            }
            ty_str => {
                byte!(7);
            }
            ty_enum(d, _) => {
                byte!(8);
                did(&mut state, d);
            }
            ty_uniq(_) => {
                byte!(9);
            }
            ty_vec(_, Some(n)) => {
                byte!(10);
                n.hash(&mut state);
            }
            ty_vec(_, None) => {
                byte!(11);
            }
            ty_ptr(m) => {
                byte!(12);
                mt(&mut state, m);
            }
            ty_rptr(r, m) => {
                byte!(13);
                region(&mut state, r);
                mt(&mut state, m);
            }
            ty_bare_fn(ref b) => {
                byte!(14);
                hash!(b.fn_style);
                hash!(b.abi);
            }
            ty_closure(ref c) => {
                byte!(15);
                hash!(c.fn_style);
                hash!(c.onceness);
                hash!(c.bounds);
                match c.store {
                    UniqTraitStore => byte!(0),
                    RegionTraitStore(r, m) => {
                        byte!(1)
                        region(&mut state, r);
                        assert_eq!(m, ast::MutMutable);
                    }
                }
            }
            ty_trait(box TyTrait { def_id: d, bounds, .. }) => {
                byte!(17);
                did(&mut state, d);
                hash!(bounds);
            }
            ty_struct(d, _) => {
                byte!(18);
                did(&mut state, d);
            }
            ty_tup(ref inner) => {
                byte!(19);
                hash!(inner.len());
            }
            ty_param(p) => {
                byte!(20);
                hash!(p.idx);
                did(&mut state, p.def_id);
            }
            ty_open(_) => byte!(22),
            ty_infer(_) => unreachable!(),
            ty_err => byte!(23),
            ty_unboxed_closure(d, r) => {
                byte!(24);
                did(&mut state, d);
                region(&mut state, r);
            }
        }
    });

    state.result()
}

impl Variance {
    pub fn to_string(self) -> &'static str {
        match self {
            Covariant => "+",
            Contravariant => "-",
            Invariant => "o",
            Bivariant => "*",
        }
    }
}

pub fn empty_parameter_environment() -> ParameterEnvironment {
    /*!
     * Construct a parameter environment suitable for static contexts
     * or other contexts where there are no free type/lifetime
     * parameters in scope.
     */

    ty::ParameterEnvironment { free_substs: Substs::empty(),
                               bounds: VecPerParamSpace::empty(),
                               caller_obligations: VecPerParamSpace::empty(),
                               implicit_region_bound: ty::ReEmpty,
                               selection_cache: traits::SelectionCache::new(), }
}

pub fn construct_parameter_environment(
    tcx: &ctxt,
    span: Span,
    generics: &ty::Generics,
    free_id: ast::NodeId)
    -> ParameterEnvironment
{
    /*! See `ParameterEnvironment` struct def'n for details */

    //
    // Construct the free substs.
    //

    // map T => T
    let mut types = VecPerParamSpace::empty();
    for &space in subst::ParamSpace::all().iter() {
        push_types_from_defs(tcx, &mut types, space,
                             generics.types.get_slice(space));
    }

    // map bound 'a => free 'a
    let mut regions = VecPerParamSpace::empty();
    for &space in subst::ParamSpace::all().iter() {
        push_region_params(&mut regions, space, free_id,
                           generics.regions.get_slice(space));
    }

    let free_substs = Substs {
        types: types,
        regions: subst::NonerasedRegions(regions)
    };

    //
    // Compute the bounds on Self and the type parameters.
    //

    let mut bounds = VecPerParamSpace::empty();
    for &space in subst::ParamSpace::all().iter() {
        push_bounds_from_defs(tcx, &mut bounds, space, &free_substs,
                              generics.types.get_slice(space));
    }

    //
    // Compute region bounds. For now, these relations are stored in a
    // global table on the tcx, so just enter them there. I'm not
    // crazy about this scheme, but it's convenient, at least.
    //

    for &space in subst::ParamSpace::all().iter() {
        record_region_bounds_from_defs(tcx, space, &free_substs,
                                       generics.regions.get_slice(space));
    }


    debug!("construct_parameter_environment: free_id={} \
           free_subst={} \
           bounds={}",
           free_id,
           free_substs.repr(tcx),
           bounds.repr(tcx));

    let obligations = traits::obligations_for_generics(tcx, traits::ObligationCause::misc(span),
                                                       generics, &free_substs);

    return ty::ParameterEnvironment {
        free_substs: free_substs,
        bounds: bounds,
        implicit_region_bound: ty::ReScope(free_id),
        caller_obligations: obligations,
        selection_cache: traits::SelectionCache::new(),
    };

    fn push_region_params(regions: &mut VecPerParamSpace<ty::Region>,
                          space: subst::ParamSpace,
                          free_id: ast::NodeId,
                          region_params: &[RegionParameterDef])
    {
        for r in region_params.iter() {
            regions.push(space, ty::free_region_from_def(free_id, r));
        }
    }

    fn push_types_from_defs(tcx: &ty::ctxt,
                            types: &mut subst::VecPerParamSpace<ty::t>,
                            space: subst::ParamSpace,
                            defs: &[TypeParameterDef]) {
        for (i, def) in defs.iter().enumerate() {
            debug!("construct_parameter_environment(): push_types_from_defs: \
                    space={} def={} index={}",
                   space,
                   def.repr(tcx),
                   i);
            let ty = ty::mk_param(tcx, space, i, def.def_id);
            types.push(space, ty);
        }
    }

    fn push_bounds_from_defs(tcx: &ty::ctxt,
                             bounds: &mut subst::VecPerParamSpace<ParamBounds>,
                             space: subst::ParamSpace,
                             free_substs: &subst::Substs,
                             defs: &[TypeParameterDef]) {
        for def in defs.iter() {
            let b = def.bounds.subst(tcx, free_substs);
            bounds.push(space, b);
        }
    }

    fn record_region_bounds_from_defs(tcx: &ty::ctxt,
                                      space: subst::ParamSpace,
                                      free_substs: &subst::Substs,
                                      defs: &[RegionParameterDef]) {
        for (subst_region, def) in
            free_substs.regions().get_slice(space).iter().zip(
                defs.iter())
        {
            // For each region parameter 'subst...
            let bounds = def.bounds.subst(tcx, free_substs);
            for bound_region in bounds.iter() {
                // Which is declared with a bound like 'subst:'bound...
                match (subst_region, bound_region) {
                    (&ty::ReFree(subst_fr), &ty::ReFree(bound_fr)) => {
                        // Record that 'subst outlives 'bound. Or, put
                        // another way, 'bound <= 'subst.
                        tcx.region_maps.relate_free_regions(bound_fr, subst_fr);
                    },
                    _ => {
                        // All named regions are instantiated with free regions.
                        tcx.sess.bug(
                            format!("push_region_bounds_from_defs: \
                                     non free region: {} / {}",
                                    subst_region.repr(tcx),
                                    bound_region.repr(tcx)).as_slice());
                    }
                }
            }
        }
    }
}

impl BorrowKind {
    pub fn from_mutbl(m: ast::Mutability) -> BorrowKind {
        match m {
            ast::MutMutable => MutBorrow,
            ast::MutImmutable => ImmBorrow,
        }
    }

    pub fn to_mutbl_lossy(self) -> ast::Mutability {
        /*!
         * Returns a mutability `m` such that an `&m T` pointer could
         * be used to obtain this borrow kind. Because borrow kinds
         * are richer than mutabilities, we sometimes have to pick a
         * mutability that is stronger than necessary so that it at
         * least *would permit* the borrow in question.
         */

        match self {
            MutBorrow => ast::MutMutable,
            ImmBorrow => ast::MutImmutable,

            // We have no type correponding to a unique imm borrow, so
            // use `&mut`. It gives all the capabilities of an `&uniq`
            // and hence is a safe "over approximation".
            UniqueImmBorrow => ast::MutMutable,
        }
    }

    pub fn to_user_str(&self) -> &'static str {
        match *self {
            MutBorrow => "mutable",
            ImmBorrow => "immutable",
            UniqueImmBorrow => "uniquely immutable",
        }
    }
}

impl<'tcx> mc::Typer<'tcx> for ty::ctxt<'tcx> {
    fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> {
        self
    }

    fn node_ty(&self, id: ast::NodeId) -> mc::McResult<ty::t> {
        Ok(ty::node_id_to_type(self, id))
    }

    fn node_method_ty(&self, method_call: typeck::MethodCall) -> Option<ty::t> {
        self.method_map.borrow().find(&method_call).map(|method| method.ty)
    }

    fn adjustments<'a>(&'a self) -> &'a RefCell<NodeMap<ty::AutoAdjustment>> {
        &self.adjustments
    }

    fn is_method_call(&self, id: ast::NodeId) -> bool {
        self.method_map.borrow().contains_key(&typeck::MethodCall::expr(id))
    }

    fn temporary_scope(&self, rvalue_id: ast::NodeId) -> Option<ast::NodeId> {
        self.region_maps.temporary_scope(rvalue_id)
    }

    fn upvar_borrow(&self, upvar_id: ty::UpvarId) -> ty::UpvarBorrow {
        self.upvar_borrow_map.borrow().get_copy(&upvar_id)
    }

    fn capture_mode(&self, closure_expr_id: ast::NodeId)
                    -> ast::CaptureClause {
        self.capture_modes.borrow().get_copy(&closure_expr_id)
    }

    fn unboxed_closures<'a>(&'a self)
                        -> &'a RefCell<DefIdMap<UnboxedClosure>> {
        &self.unboxed_closures
    }
}

/// The category of explicit self.
#[deriving(Clone, Eq, PartialEq)]
pub enum ExplicitSelfCategory {
    StaticExplicitSelfCategory,
    ByValueExplicitSelfCategory,
    ByReferenceExplicitSelfCategory(Region, ast::Mutability),
    ByBoxExplicitSelfCategory,
}

/// Pushes all the lifetimes in the given type onto the given list. A
/// "lifetime in a type" is a lifetime specified by a reference or a lifetime
/// in a list of type substitutions. This does *not* traverse into nominal
/// types, nor does it resolve fictitious types.
pub fn accumulate_lifetimes_in_type(accumulator: &mut Vec<ty::Region>,
                                    typ: t) {
    walk_ty(typ, |typ| {
        match get(typ).sty {
            ty_rptr(region, _) => accumulator.push(region),
            ty_enum(_, ref substs) |
            ty_trait(box TyTrait {
                substs: ref substs,
                ..
            }) |
            ty_struct(_, ref substs) => {
                match substs.regions {
                    subst::ErasedRegions => {}
                    subst::NonerasedRegions(ref regions) => {
                        for region in regions.iter() {
                            accumulator.push(*region)
                        }
                    }
                }
            }
            ty_closure(ref closure_ty) => {
                match closure_ty.store {
                    RegionTraitStore(region, _) => accumulator.push(region),
                    UniqTraitStore => {}
                }
            }
            ty_unboxed_closure(_, ref region) => accumulator.push(*region),
            ty_nil |
            ty_bot |
            ty_bool |
            ty_char |
            ty_int(_) |
            ty_uint(_) |
            ty_float(_) |
            ty_uniq(_) |
            ty_str |
            ty_vec(_, _) |
            ty_ptr(_) |
            ty_bare_fn(_) |
            ty_tup(_) |
            ty_param(_) |
            ty_infer(_) |
            ty_open(_) |
            ty_err => {}
        }
    })
}

/// A free variable referred to in a function.
#[deriving(Encodable, Decodable)]
pub struct Freevar {
    /// The variable being accessed free.
    pub def: def::Def,

    // First span where it is accessed (there can be multiple).
    pub span: Span
}

pub type FreevarMap = NodeMap<Vec<Freevar>>;

pub type CaptureModeMap = NodeMap<ast::CaptureClause>;

pub fn with_freevars<T>(tcx: &ty::ctxt, fid: ast::NodeId, f: |&[Freevar]| -> T) -> T {
    match tcx.freevars.borrow().find(&fid) {
        None => f(&[]),
        Some(d) => f(d.as_slice())
    }
}
