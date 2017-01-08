// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::Variance::*;
pub use self::DtorKind::*;
pub use self::AssociatedItemContainer::*;
pub use self::BorrowKind::*;
pub use self::IntVarValue::*;
pub use self::LvaluePreference::*;
pub use self::fold::TypeFoldable;

use dep_graph::{self, DepNode};
use hir::{map as ast_map, FreevarMap, TraitMap};
use middle;
use hir::def::{Def, CtorKind, ExportMap};
use hir::def_id::{CrateNum, DefId, CRATE_DEF_INDEX, LOCAL_CRATE};
use middle::lang_items::{FnTraitLangItem, FnMutTraitLangItem, FnOnceTraitLangItem};
use middle::region::{CodeExtent, ROOT_CODE_EXTENT};
use mir::Mir;
use traits;
use ty;
use ty::subst::{Subst, Substs};
use ty::walk::TypeWalker;
use util::common::MemoizationMap;
use util::nodemap::{NodeSet, NodeMap, FxHashMap};

use serialize::{self, Encodable, Encoder};
use std::borrow::Cow;
use std::cell::{Cell, RefCell, Ref};
use std::hash::{Hash, Hasher};
use std::iter;
use std::ops::Deref;
use std::rc::Rc;
use std::slice;
use std::vec::IntoIter;
use std::mem;
use syntax::ast::{self, Name, NodeId};
use syntax::attr;
use syntax::symbol::{Symbol, InternedString};
use syntax_pos::{DUMMY_SP, Span};

use rustc_const_math::ConstInt;
use rustc_data_structures::accumulate_vec::IntoIter as AccIntoIter;

use hir;
use hir::itemlikevisit::ItemLikeVisitor;

pub use self::sty::{Binder, DebruijnIndex};
pub use self::sty::{BareFnTy, FnSig, PolyFnSig};
pub use self::sty::{ClosureTy, InferTy, ParamTy, ProjectionTy, ExistentialPredicate};
pub use self::sty::{ClosureSubsts, TypeAndMut};
pub use self::sty::{TraitRef, TypeVariants, PolyTraitRef};
pub use self::sty::{ExistentialTraitRef, PolyExistentialTraitRef};
pub use self::sty::{ExistentialProjection, PolyExistentialProjection};
pub use self::sty::{BoundRegion, EarlyBoundRegion, FreeRegion, Region};
pub use self::sty::Issue32330;
pub use self::sty::{TyVid, IntVid, FloatVid, RegionVid, SkolemizedRegionVid};
pub use self::sty::BoundRegion::*;
pub use self::sty::InferTy::*;
pub use self::sty::Region::*;
pub use self::sty::TypeVariants::*;

pub use self::contents::TypeContents;
pub use self::context::{TyCtxt, GlobalArenas, tls};
pub use self::context::{Lift, Tables};

pub use self::trait_def::{TraitDef, TraitFlags};

pub mod adjustment;
pub mod cast;
pub mod error;
pub mod fast_reject;
pub mod fold;
pub mod inhabitedness;
pub mod item_path;
pub mod layout;
pub mod _match;
pub mod maps;
pub mod outlives;
pub mod relate;
pub mod subst;
pub mod trait_def;
pub mod walk;
pub mod wf;
pub mod util;

mod contents;
mod context;
mod flags;
mod structural_impls;
mod sty;

pub type Disr = ConstInt;

// Data types

/// The complete set of all analyses described in this module. This is
/// produced by the driver and fed to trans and later passes.
#[derive(Clone)]
pub struct CrateAnalysis<'tcx> {
    pub export_map: ExportMap,
    pub access_levels: middle::privacy::AccessLevels,
    pub reachable: NodeSet,
    pub name: String,
    pub glob_map: Option<hir::GlobMap>,
    pub hir_ty_to_ty: NodeMap<Ty<'tcx>>,
}

#[derive(Clone)]
pub struct Resolutions {
    pub freevars: FreevarMap,
    pub trait_map: TraitMap,
    pub maybe_unused_trait_imports: NodeSet,
}

#[derive(Copy, Clone)]
pub enum DtorKind {
    NoDtor,
    TraitDtor
}

impl DtorKind {
    pub fn is_present(&self) -> bool {
        match *self {
            TraitDtor => true,
            _ => false
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AssociatedItemContainer {
    TraitContainer(DefId),
    ImplContainer(DefId),
}

impl AssociatedItemContainer {
    pub fn id(&self) -> DefId {
        match *self {
            TraitContainer(id) => id,
            ImplContainer(id) => id,
        }
    }
}

/// The "header" of an impl is everything outside the body: a Self type, a trait
/// ref (in the case of a trait impl), and a set of predicates (from the
/// bounds/where clauses).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ImplHeader<'tcx> {
    pub impl_def_id: DefId,
    pub self_ty: Ty<'tcx>,
    pub trait_ref: Option<TraitRef<'tcx>>,
    pub predicates: Vec<Predicate<'tcx>>,
}

impl<'a, 'gcx, 'tcx> ImplHeader<'tcx> {
    pub fn with_fresh_ty_vars(selcx: &mut traits::SelectionContext<'a, 'gcx, 'tcx>,
                              impl_def_id: DefId)
                              -> ImplHeader<'tcx>
    {
        let tcx = selcx.tcx();
        let impl_substs = selcx.infcx().fresh_substs_for_item(DUMMY_SP, impl_def_id);

        let header = ImplHeader {
            impl_def_id: impl_def_id,
            self_ty: tcx.item_type(impl_def_id),
            trait_ref: tcx.impl_trait_ref(impl_def_id),
            predicates: tcx.item_predicates(impl_def_id).predicates
        }.subst(tcx, impl_substs);

        let traits::Normalized { value: mut header, obligations } =
            traits::normalize(selcx, traits::ObligationCause::dummy(), &header);

        header.predicates.extend(obligations.into_iter().map(|o| o.predicate));
        header
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AssociatedItem {
    pub def_id: DefId,
    pub name: Name,
    pub kind: AssociatedKind,
    pub vis: Visibility,
    pub defaultness: hir::Defaultness,
    pub container: AssociatedItemContainer,

    /// Whether this is a method with an explicit self
    /// as its first argument, allowing method calls.
    pub method_has_self_argument: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, RustcEncodable, RustcDecodable)]
pub enum AssociatedKind {
    Const,
    Method,
    Type
}

impl AssociatedItem {
    pub fn def(&self) -> Def {
        match self.kind {
            AssociatedKind::Const => Def::AssociatedConst(self.def_id),
            AssociatedKind::Method => Def::Method(self.def_id),
            AssociatedKind::Type => Def::AssociatedTy(self.def_id),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Copy, RustcEncodable, RustcDecodable)]
pub enum Visibility {
    /// Visible everywhere (including in other crates).
    Public,
    /// Visible only in the given crate-local module.
    Restricted(DefId),
    /// Not visible anywhere in the local crate. This is the visibility of private external items.
    Invisible,
}

pub trait DefIdTree: Copy {
    fn parent(self, id: DefId) -> Option<DefId>;

    fn is_descendant_of(self, mut descendant: DefId, ancestor: DefId) -> bool {
        if descendant.krate != ancestor.krate {
            return false;
        }

        while descendant != ancestor {
            match self.parent(descendant) {
                Some(parent) => descendant = parent,
                None => return false,
            }
        }
        true
    }
}

impl<'a, 'gcx, 'tcx> DefIdTree for TyCtxt<'a, 'gcx, 'tcx> {
    fn parent(self, id: DefId) -> Option<DefId> {
        self.def_key(id).parent.map(|index| DefId { index: index, ..id })
    }
}

impl Visibility {
    pub fn from_hir(visibility: &hir::Visibility, id: NodeId, tcx: TyCtxt) -> Self {
        match *visibility {
            hir::Public => Visibility::Public,
            hir::Visibility::Crate => Visibility::Restricted(DefId::local(CRATE_DEF_INDEX)),
            hir::Visibility::Restricted { ref path, .. } => match path.def {
                // If there is no resolution, `resolve` will have already reported an error, so
                // assume that the visibility is public to avoid reporting more privacy errors.
                Def::Err => Visibility::Public,
                def => Visibility::Restricted(def.def_id()),
            },
            hir::Inherited => {
                Visibility::Restricted(tcx.map.local_def_id(tcx.map.get_module_parent(id)))
            }
        }
    }

    /// Returns true if an item with this visibility is accessible from the given block.
    pub fn is_accessible_from<T: DefIdTree>(self, module: DefId, tree: T) -> bool {
        let restriction = match self {
            // Public items are visible everywhere.
            Visibility::Public => return true,
            // Private items from other crates are visible nowhere.
            Visibility::Invisible => return false,
            // Restricted items are visible in an arbitrary local module.
            Visibility::Restricted(other) if other.krate != module.krate => return false,
            Visibility::Restricted(module) => module,
        };

        tree.is_descendant_of(module, restriction)
    }

    /// Returns true if this visibility is at least as accessible as the given visibility
    pub fn is_at_least<T: DefIdTree>(self, vis: Visibility, tree: T) -> bool {
        let vis_restriction = match vis {
            Visibility::Public => return self == Visibility::Public,
            Visibility::Invisible => return true,
            Visibility::Restricted(module) => module,
        };

        self.is_accessible_from(vis_restriction, tree)
    }
}

#[derive(Clone, PartialEq, RustcDecodable, RustcEncodable, Copy)]
pub enum Variance {
    Covariant,      // T<A> <: T<B> iff A <: B -- e.g., function return type
    Invariant,      // T<A> <: T<B> iff B == A -- e.g., type of mutable cell
    Contravariant,  // T<A> <: T<B> iff B <: A -- e.g., function param type
    Bivariant,      // T<A> <: T<B>            -- e.g., unused type parameter
}

#[derive(Clone, Copy, Debug, RustcDecodable, RustcEncodable)]
pub struct MethodCallee<'tcx> {
    /// Impl method ID, for inherent methods, or trait method ID, otherwise.
    pub def_id: DefId,
    pub ty: Ty<'tcx>,
    pub substs: &'tcx Substs<'tcx>
}

/// With method calls, we store some extra information in
/// side tables (i.e method_map). We use
/// MethodCall as a key to index into these tables instead of
/// just directly using the expression's NodeId. The reason
/// for this being that we may apply adjustments (coercions)
/// with the resulting expression also needing to use the
/// side tables. The problem with this is that we don't
/// assign a separate NodeId to this new expression
/// and so it would clash with the base expression if both
/// needed to add to the side tables. Thus to disambiguate
/// we also keep track of whether there's an adjustment in
/// our key.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct MethodCall {
    pub expr_id: NodeId,
    pub autoderef: u32
}

impl MethodCall {
    pub fn expr(id: NodeId) -> MethodCall {
        MethodCall {
            expr_id: id,
            autoderef: 0
        }
    }

    pub fn autoderef(expr_id: NodeId, autoderef: u32) -> MethodCall {
        MethodCall {
            expr_id: expr_id,
            autoderef: 1 + autoderef
        }
    }
}

// maps from an expression id that corresponds to a method call to the details
// of the method to be invoked
pub type MethodMap<'tcx> = FxHashMap<MethodCall, MethodCallee<'tcx>>;

// Contains information needed to resolve types and (in the future) look up
// the types of AST nodes.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct CReaderCacheKey {
    pub cnum: CrateNum,
    pub pos: usize,
}

/// Describes the fragment-state associated with a NodeId.
///
/// Currently only unfragmented paths have entries in the table,
/// but longer-term this enum is expected to expand to also
/// include data for fragmented paths.
#[derive(Copy, Clone, Debug)]
pub enum FragmentInfo {
    Moved { var: NodeId, move_expr: NodeId },
    Assigned { var: NodeId, assign_expr: NodeId, assignee_id: NodeId },
}

// Flags that we track on types. These flags are propagated upwards
// through the type during type construction, so that we can quickly
// check whether the type has various kinds of types in it without
// recursing over the type itself.
bitflags! {
    flags TypeFlags: u32 {
        const HAS_PARAMS         = 1 << 0,
        const HAS_SELF           = 1 << 1,
        const HAS_TY_INFER       = 1 << 2,
        const HAS_RE_INFER       = 1 << 3,
        const HAS_RE_SKOL        = 1 << 4,
        const HAS_RE_EARLY_BOUND = 1 << 5,
        const HAS_FREE_REGIONS   = 1 << 6,
        const HAS_TY_ERR         = 1 << 7,
        const HAS_PROJECTION     = 1 << 8,
        const HAS_TY_CLOSURE     = 1 << 9,

        // true if there are "names" of types and regions and so forth
        // that are local to a particular fn
        const HAS_LOCAL_NAMES    = 1 << 10,

        // Present if the type belongs in a local type context.
        // Only set for TyInfer other than Fresh.
        const KEEP_IN_LOCAL_TCX  = 1 << 11,

        // Is there a projection that does not involve a bound region?
        // Currently we can't normalize projections w/ bound regions.
        const HAS_NORMALIZABLE_PROJECTION = 1 << 12,

        const NEEDS_SUBST        = TypeFlags::HAS_PARAMS.bits |
                                   TypeFlags::HAS_SELF.bits |
                                   TypeFlags::HAS_RE_EARLY_BOUND.bits,

        // Flags representing the nominal content of a type,
        // computed by FlagsComputation. If you add a new nominal
        // flag, it should be added here too.
        const NOMINAL_FLAGS     = TypeFlags::HAS_PARAMS.bits |
                                  TypeFlags::HAS_SELF.bits |
                                  TypeFlags::HAS_TY_INFER.bits |
                                  TypeFlags::HAS_RE_INFER.bits |
                                  TypeFlags::HAS_RE_SKOL.bits |
                                  TypeFlags::HAS_RE_EARLY_BOUND.bits |
                                  TypeFlags::HAS_FREE_REGIONS.bits |
                                  TypeFlags::HAS_TY_ERR.bits |
                                  TypeFlags::HAS_PROJECTION.bits |
                                  TypeFlags::HAS_TY_CLOSURE.bits |
                                  TypeFlags::HAS_LOCAL_NAMES.bits |
                                  TypeFlags::KEEP_IN_LOCAL_TCX.bits,

        // Caches for type_is_sized, type_moves_by_default
        const SIZEDNESS_CACHED  = 1 << 16,
        const IS_SIZED          = 1 << 17,
        const MOVENESS_CACHED   = 1 << 18,
        const MOVES_BY_DEFAULT  = 1 << 19,
    }
}

pub struct TyS<'tcx> {
    pub sty: TypeVariants<'tcx>,
    pub flags: Cell<TypeFlags>,

    // the maximal depth of any bound regions appearing in this type.
    region_depth: u32,
}

impl<'tcx> PartialEq for TyS<'tcx> {
    #[inline]
    fn eq(&self, other: &TyS<'tcx>) -> bool {
        // (self as *const _) == (other as *const _)
        (self as *const TyS<'tcx>) == (other as *const TyS<'tcx>)
    }
}
impl<'tcx> Eq for TyS<'tcx> {}

impl<'tcx> Hash for TyS<'tcx> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        (self as *const TyS).hash(s)
    }
}

pub type Ty<'tcx> = &'tcx TyS<'tcx>;

impl<'tcx> serialize::UseSpecializedEncodable for Ty<'tcx> {}
impl<'tcx> serialize::UseSpecializedDecodable for Ty<'tcx> {}

/// A wrapper for slices with the additional invariant
/// that the slice is interned and no other slice with
/// the same contents can exist in the same context.
/// This means we can use pointer + length for both
/// equality comparisons and hashing.
#[derive(Debug, RustcEncodable)]
pub struct Slice<T>([T]);

impl<T> PartialEq for Slice<T> {
    #[inline]
    fn eq(&self, other: &Slice<T>) -> bool {
        (&self.0 as *const [T]) == (&other.0 as *const [T])
    }
}
impl<T> Eq for Slice<T> {}

impl<T> Hash for Slice<T> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        (self.as_ptr(), self.len()).hash(s)
    }
}

impl<T> Deref for Slice<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.0
    }
}

impl<'a, T> IntoIterator for &'a Slice<T> {
    type Item = &'a T;
    type IntoIter = <&'a [T] as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self[..].iter()
    }
}

impl<'tcx> serialize::UseSpecializedDecodable for &'tcx Slice<Ty<'tcx>> {}

impl<T> Slice<T> {
    pub fn empty<'a>() -> &'a Slice<T> {
        unsafe {
            mem::transmute(slice::from_raw_parts(0x1 as *const T, 0))
        }
    }
}

/// Upvars do not get their own node-id. Instead, we use the pair of
/// the original var id (that is, the root variable that is referenced
/// by the upvar) and the id of the closure expression.
#[derive(Clone, Copy, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct UpvarId {
    pub var_id: NodeId,
    pub closure_expr_id: NodeId,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable, Copy)]
pub enum BorrowKind {
    /// Data must be immutable and is aliasable.
    ImmBorrow,

    /// Data must be immutable but not aliasable.  This kind of borrow
    /// cannot currently be expressed by the user and is used only in
    /// implicit closure bindings. It is needed when you the closure
    /// is borrowing or mutating a mutable referent, e.g.:
    ///
    ///    let x: &mut isize = ...;
    ///    let y = || *x += 5;
    ///
    /// If we were to try to translate this closure into a more explicit
    /// form, we'd encounter an error with the code as written:
    ///
    ///    struct Env { x: & &mut isize }
    ///    let x: &mut isize = ...;
    ///    let y = (&mut Env { &x }, fn_ptr);  // Closure is pair of env and fn
    ///    fn fn_ptr(env: &mut Env) { **env.x += 5; }
    ///
    /// This is then illegal because you cannot mutate a `&mut` found
    /// in an aliasable location. To solve, you'd have to translate with
    /// an `&mut` borrow:
    ///
    ///    struct Env { x: & &mut isize }
    ///    let x: &mut isize = ...;
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

/// Information describing the capture of an upvar. This is computed
/// during `typeck`, specifically by `regionck`.
#[derive(PartialEq, Clone, Debug, Copy, RustcEncodable, RustcDecodable)]
pub enum UpvarCapture<'tcx> {
    /// Upvar is captured by value. This is always true when the
    /// closure is labeled `move`, but can also be true in other cases
    /// depending on inference.
    ByValue,

    /// Upvar is captured by reference.
    ByRef(UpvarBorrow<'tcx>),
}

#[derive(PartialEq, Clone, Copy, RustcEncodable, RustcDecodable)]
pub struct UpvarBorrow<'tcx> {
    /// The kind of borrow: by-ref upvars have access to shared
    /// immutable borrows, which are not part of the normal language
    /// syntax.
    pub kind: BorrowKind,

    /// Region of the resulting reference.
    pub region: &'tcx ty::Region,
}

pub type UpvarCaptureMap<'tcx> = FxHashMap<UpvarId, UpvarCapture<'tcx>>;

#[derive(Copy, Clone)]
pub struct ClosureUpvar<'tcx> {
    pub def: Def,
    pub span: Span,
    pub ty: Ty<'tcx>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum IntVarValue {
    IntType(ast::IntTy),
    UintType(ast::UintTy),
}

/// Default region to use for the bound of objects that are
/// supplied as the value for this type parameter. This is derived
/// from `T:'a` annotations appearing in the type definition.  If
/// this is `None`, then the default is inherited from the
/// surrounding context. See RFC #599 for details.
#[derive(Copy, Clone, RustcEncodable, RustcDecodable)]
pub enum ObjectLifetimeDefault<'tcx> {
    /// Require an explicit annotation. Occurs when multiple
    /// `T:'a` constraints are found.
    Ambiguous,

    /// Use the base default, typically 'static, but in a fn body it is a fresh variable
    BaseDefault,

    /// Use the given region as the default.
    Specific(&'tcx Region),
}

#[derive(Clone, RustcEncodable, RustcDecodable)]
pub struct TypeParameterDef<'tcx> {
    pub name: Name,
    pub def_id: DefId,
    pub index: u32,
    pub default_def_id: DefId, // for use in error reporing about defaults
    pub default: Option<Ty<'tcx>>,
    pub object_lifetime_default: ObjectLifetimeDefault<'tcx>,

    /// `pure_wrt_drop`, set by the (unsafe) `#[may_dangle]` attribute
    /// on generic parameter `T`, asserts data behind the parameter
    /// `T` won't be accessed during the parent type's `Drop` impl.
    pub pure_wrt_drop: bool,
}

#[derive(Clone, RustcEncodable, RustcDecodable)]
pub struct RegionParameterDef<'tcx> {
    pub name: Name,
    pub def_id: DefId,
    pub index: u32,
    pub bounds: Vec<&'tcx ty::Region>,

    /// `pure_wrt_drop`, set by the (unsafe) `#[may_dangle]` attribute
    /// on generic parameter `'a`, asserts data of lifetime `'a`
    /// won't be accessed during the parent type's `Drop` impl.
    pub pure_wrt_drop: bool,
}

impl<'tcx> RegionParameterDef<'tcx> {
    pub fn to_early_bound_region_data(&self) -> ty::EarlyBoundRegion {
        ty::EarlyBoundRegion {
            index: self.index,
            name: self.name,
        }
    }

    pub fn to_bound_region(&self) -> ty::BoundRegion {
        // this is an early bound region, so unaffected by #32330
        ty::BoundRegion::BrNamed(self.def_id, self.name, Issue32330::WontChange)
    }
}

/// Information about the formal type/lifetime parameters associated
/// with an item or method. Analogous to hir::Generics.
#[derive(Clone, Debug, RustcEncodable, RustcDecodable)]
pub struct Generics<'tcx> {
    pub parent: Option<DefId>,
    pub parent_regions: u32,
    pub parent_types: u32,
    pub regions: Vec<RegionParameterDef<'tcx>>,
    pub types: Vec<TypeParameterDef<'tcx>>,
    pub has_self: bool,
}

impl<'tcx> Generics<'tcx> {
    pub fn parent_count(&self) -> usize {
        self.parent_regions as usize + self.parent_types as usize
    }

    pub fn own_count(&self) -> usize {
        self.regions.len() + self.types.len()
    }

    pub fn count(&self) -> usize {
        self.parent_count() + self.own_count()
    }

    pub fn region_param(&self, param: &EarlyBoundRegion) -> &RegionParameterDef<'tcx> {
        &self.regions[param.index as usize - self.has_self as usize]
    }

    pub fn type_param(&self, param: &ParamTy) -> &TypeParameterDef<'tcx> {
        &self.types[param.idx as usize - self.has_self as usize - self.regions.len()]
    }
}

/// Bounds on generics.
#[derive(Clone)]
pub struct GenericPredicates<'tcx> {
    pub parent: Option<DefId>,
    pub predicates: Vec<Predicate<'tcx>>,
}

impl<'tcx> serialize::UseSpecializedEncodable for GenericPredicates<'tcx> {}
impl<'tcx> serialize::UseSpecializedDecodable for GenericPredicates<'tcx> {}

impl<'a, 'gcx, 'tcx> GenericPredicates<'tcx> {
    pub fn instantiate(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>, substs: &Substs<'tcx>)
                       -> InstantiatedPredicates<'tcx> {
        let mut instantiated = InstantiatedPredicates::empty();
        self.instantiate_into(tcx, &mut instantiated, substs);
        instantiated
    }
    pub fn instantiate_own(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>, substs: &Substs<'tcx>)
                           -> InstantiatedPredicates<'tcx> {
        InstantiatedPredicates {
            predicates: self.predicates.subst(tcx, substs)
        }
    }

    fn instantiate_into(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                        instantiated: &mut InstantiatedPredicates<'tcx>,
                        substs: &Substs<'tcx>) {
        if let Some(def_id) = self.parent {
            tcx.item_predicates(def_id).instantiate_into(tcx, instantiated, substs);
        }
        instantiated.predicates.extend(self.predicates.iter().map(|p| p.subst(tcx, substs)))
    }

    pub fn instantiate_supertrait(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                  poly_trait_ref: &ty::PolyTraitRef<'tcx>)
                                  -> InstantiatedPredicates<'tcx>
    {
        assert_eq!(self.parent, None);
        InstantiatedPredicates {
            predicates: self.predicates.iter().map(|pred| {
                pred.subst_supertrait(tcx, poly_trait_ref)
            }).collect()
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub enum Predicate<'tcx> {
    /// Corresponds to `where Foo : Bar<A,B,C>`. `Foo` here would be
    /// the `Self` type of the trait reference and `A`, `B`, and `C`
    /// would be the type parameters.
    Trait(PolyTraitPredicate<'tcx>),

    /// where `T1 == T2`.
    Equate(PolyEquatePredicate<'tcx>),

    /// where 'a : 'b
    RegionOutlives(PolyRegionOutlivesPredicate<'tcx>),

    /// where T : 'a
    TypeOutlives(PolyTypeOutlivesPredicate<'tcx>),

    /// where <T as TraitRef>::Name == X, approximately.
    /// See `ProjectionPredicate` struct for details.
    Projection(PolyProjectionPredicate<'tcx>),

    /// no syntax: T WF
    WellFormed(Ty<'tcx>),

    /// trait must be object-safe
    ObjectSafe(DefId),

    /// No direct syntax. May be thought of as `where T : FnFoo<...>`
    /// for some substitutions `...` and T being a closure type.
    /// Satisfied (or refuted) once we know the closure's kind.
    ClosureKind(DefId, ClosureKind),
}

impl<'a, 'gcx, 'tcx> Predicate<'tcx> {
    /// Performs a substitution suitable for going from a
    /// poly-trait-ref to supertraits that must hold if that
    /// poly-trait-ref holds. This is slightly different from a normal
    /// substitution in terms of what happens with bound regions.  See
    /// lengthy comment below for details.
    pub fn subst_supertrait(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                            trait_ref: &ty::PolyTraitRef<'tcx>)
                            -> ty::Predicate<'tcx>
    {
        // The interaction between HRTB and supertraits is not entirely
        // obvious. Let me walk you (and myself) through an example.
        //
        // Let's start with an easy case. Consider two traits:
        //
        //     trait Foo<'a> : Bar<'a,'a> { }
        //     trait Bar<'b,'c> { }
        //
        // Now, if we have a trait reference `for<'x> T : Foo<'x>`, then
        // we can deduce that `for<'x> T : Bar<'x,'x>`. Basically, if we
        // knew that `Foo<'x>` (for any 'x) then we also know that
        // `Bar<'x,'x>` (for any 'x). This more-or-less falls out from
        // normal substitution.
        //
        // In terms of why this is sound, the idea is that whenever there
        // is an impl of `T:Foo<'a>`, it must show that `T:Bar<'a,'a>`
        // holds.  So if there is an impl of `T:Foo<'a>` that applies to
        // all `'a`, then we must know that `T:Bar<'a,'a>` holds for all
        // `'a`.
        //
        // Another example to be careful of is this:
        //
        //     trait Foo1<'a> : for<'b> Bar1<'a,'b> { }
        //     trait Bar1<'b,'c> { }
        //
        // Here, if we have `for<'x> T : Foo1<'x>`, then what do we know?
        // The answer is that we know `for<'x,'b> T : Bar1<'x,'b>`. The
        // reason is similar to the previous example: any impl of
        // `T:Foo1<'x>` must show that `for<'b> T : Bar1<'x, 'b>`.  So
        // basically we would want to collapse the bound lifetimes from
        // the input (`trait_ref`) and the supertraits.
        //
        // To achieve this in practice is fairly straightforward. Let's
        // consider the more complicated scenario:
        //
        // - We start out with `for<'x> T : Foo1<'x>`. In this case, `'x`
        //   has a De Bruijn index of 1. We want to produce `for<'x,'b> T : Bar1<'x,'b>`,
        //   where both `'x` and `'b` would have a DB index of 1.
        //   The substitution from the input trait-ref is therefore going to be
        //   `'a => 'x` (where `'x` has a DB index of 1).
        // - The super-trait-ref is `for<'b> Bar1<'a,'b>`, where `'a` is an
        //   early-bound parameter and `'b' is a late-bound parameter with a
        //   DB index of 1.
        // - If we replace `'a` with `'x` from the input, it too will have
        //   a DB index of 1, and thus we'll have `for<'x,'b> Bar1<'x,'b>`
        //   just as we wanted.
        //
        // There is only one catch. If we just apply the substitution `'a
        // => 'x` to `for<'b> Bar1<'a,'b>`, the substitution code will
        // adjust the DB index because we substituting into a binder (it
        // tries to be so smart...) resulting in `for<'x> for<'b>
        // Bar1<'x,'b>` (we have no syntax for this, so use your
        // imagination). Basically the 'x will have DB index of 2 and 'b
        // will have DB index of 1. Not quite what we want. So we apply
        // the substitution to the *contents* of the trait reference,
        // rather than the trait reference itself (put another way, the
        // substitution code expects equal binding levels in the values
        // from the substitution and the value being substituted into, and
        // this trick achieves that).

        let substs = &trait_ref.0.substs;
        match *self {
            Predicate::Trait(ty::Binder(ref data)) =>
                Predicate::Trait(ty::Binder(data.subst(tcx, substs))),
            Predicate::Equate(ty::Binder(ref data)) =>
                Predicate::Equate(ty::Binder(data.subst(tcx, substs))),
            Predicate::RegionOutlives(ty::Binder(ref data)) =>
                Predicate::RegionOutlives(ty::Binder(data.subst(tcx, substs))),
            Predicate::TypeOutlives(ty::Binder(ref data)) =>
                Predicate::TypeOutlives(ty::Binder(data.subst(tcx, substs))),
            Predicate::Projection(ty::Binder(ref data)) =>
                Predicate::Projection(ty::Binder(data.subst(tcx, substs))),
            Predicate::WellFormed(data) =>
                Predicate::WellFormed(data.subst(tcx, substs)),
            Predicate::ObjectSafe(trait_def_id) =>
                Predicate::ObjectSafe(trait_def_id),
            Predicate::ClosureKind(closure_def_id, kind) =>
                Predicate::ClosureKind(closure_def_id, kind),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct TraitPredicate<'tcx> {
    pub trait_ref: TraitRef<'tcx>
}
pub type PolyTraitPredicate<'tcx> = ty::Binder<TraitPredicate<'tcx>>;

impl<'tcx> TraitPredicate<'tcx> {
    pub fn def_id(&self) -> DefId {
        self.trait_ref.def_id
    }

    /// Creates the dep-node for selecting/evaluating this trait reference.
    fn dep_node(&self) -> DepNode<DefId> {
        // Ideally, the dep-node would just have all the input types
        // in it.  But they are limited to including def-ids. So as an
        // approximation we include the def-ids for all nominal types
        // found somewhere. This means that we will e.g. conflate the
        // dep-nodes for `u32: SomeTrait` and `u64: SomeTrait`, but we
        // would have distinct dep-nodes for `Vec<u32>: SomeTrait`,
        // `Rc<u32>: SomeTrait`, and `(Vec<u32>, Rc<u32>): SomeTrait`.
        // Note that it's always sound to conflate dep-nodes, it just
        // leads to more recompilation.
        let def_ids: Vec<_> =
            self.input_types()
                .flat_map(|t| t.walk())
                .filter_map(|t| match t.sty {
                    ty::TyAdt(adt_def, _) =>
                        Some(adt_def.did),
                    _ =>
                        None
                })
                .chain(iter::once(self.def_id()))
                .collect();
        DepNode::TraitSelect(def_ids)
    }

    pub fn input_types<'a>(&'a self) -> impl DoubleEndedIterator<Item=Ty<'tcx>> + 'a {
        self.trait_ref.input_types()
    }

    pub fn self_ty(&self) -> Ty<'tcx> {
        self.trait_ref.self_ty()
    }
}

impl<'tcx> PolyTraitPredicate<'tcx> {
    pub fn def_id(&self) -> DefId {
        // ok to skip binder since trait def-id does not care about regions
        self.0.def_id()
    }

    pub fn dep_node(&self) -> DepNode<DefId> {
        // ok to skip binder since depnode does not care about regions
        self.0.dep_node()
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct EquatePredicate<'tcx>(pub Ty<'tcx>, pub Ty<'tcx>); // `0 == 1`
pub type PolyEquatePredicate<'tcx> = ty::Binder<EquatePredicate<'tcx>>;

#[derive(Clone, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct OutlivesPredicate<A,B>(pub A, pub B); // `A : B`
pub type PolyOutlivesPredicate<A,B> = ty::Binder<OutlivesPredicate<A,B>>;
pub type PolyRegionOutlivesPredicate<'tcx> = PolyOutlivesPredicate<&'tcx ty::Region,
                                                                   &'tcx ty::Region>;
pub type PolyTypeOutlivesPredicate<'tcx> = PolyOutlivesPredicate<Ty<'tcx>, &'tcx ty::Region>;

/// This kind of predicate has no *direct* correspondent in the
/// syntax, but it roughly corresponds to the syntactic forms:
///
/// 1. `T : TraitRef<..., Item=Type>`
/// 2. `<T as TraitRef<...>>::Item == Type` (NYI)
///
/// In particular, form #1 is "desugared" to the combination of a
/// normal trait predicate (`T : TraitRef<...>`) and one of these
/// predicates. Form #2 is a broader form in that it also permits
/// equality between arbitrary types. Processing an instance of Form
/// #2 eventually yields one of these `ProjectionPredicate`
/// instances to normalize the LHS.
#[derive(Copy, Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct ProjectionPredicate<'tcx> {
    pub projection_ty: ProjectionTy<'tcx>,
    pub ty: Ty<'tcx>,
}

pub type PolyProjectionPredicate<'tcx> = Binder<ProjectionPredicate<'tcx>>;

impl<'tcx> PolyProjectionPredicate<'tcx> {
    pub fn item_name(&self) -> Name {
        self.0.projection_ty.item_name // safe to skip the binder to access a name
    }
}

pub trait ToPolyTraitRef<'tcx> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx>;
}

impl<'tcx> ToPolyTraitRef<'tcx> for TraitRef<'tcx> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx> {
        assert!(!self.has_escaping_regions());
        ty::Binder(self.clone())
    }
}

impl<'tcx> ToPolyTraitRef<'tcx> for PolyTraitPredicate<'tcx> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx> {
        self.map_bound_ref(|trait_pred| trait_pred.trait_ref)
    }
}

impl<'tcx> ToPolyTraitRef<'tcx> for PolyProjectionPredicate<'tcx> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx> {
        // Note: unlike with TraitRef::to_poly_trait_ref(),
        // self.0.trait_ref is permitted to have escaping regions.
        // This is because here `self` has a `Binder` and so does our
        // return value, so we are preserving the number of binding
        // levels.
        ty::Binder(self.0.projection_ty.trait_ref)
    }
}

pub trait ToPredicate<'tcx> {
    fn to_predicate(&self) -> Predicate<'tcx>;
}

impl<'tcx> ToPredicate<'tcx> for TraitRef<'tcx> {
    fn to_predicate(&self) -> Predicate<'tcx> {
        // we're about to add a binder, so let's check that we don't
        // accidentally capture anything, or else that might be some
        // weird debruijn accounting.
        assert!(!self.has_escaping_regions());

        ty::Predicate::Trait(ty::Binder(ty::TraitPredicate {
            trait_ref: self.clone()
        }))
    }
}

impl<'tcx> ToPredicate<'tcx> for PolyTraitRef<'tcx> {
    fn to_predicate(&self) -> Predicate<'tcx> {
        ty::Predicate::Trait(self.to_poly_trait_predicate())
    }
}

impl<'tcx> ToPredicate<'tcx> for PolyEquatePredicate<'tcx> {
    fn to_predicate(&self) -> Predicate<'tcx> {
        Predicate::Equate(self.clone())
    }
}

impl<'tcx> ToPredicate<'tcx> for PolyRegionOutlivesPredicate<'tcx> {
    fn to_predicate(&self) -> Predicate<'tcx> {
        Predicate::RegionOutlives(self.clone())
    }
}

impl<'tcx> ToPredicate<'tcx> for PolyTypeOutlivesPredicate<'tcx> {
    fn to_predicate(&self) -> Predicate<'tcx> {
        Predicate::TypeOutlives(self.clone())
    }
}

impl<'tcx> ToPredicate<'tcx> for PolyProjectionPredicate<'tcx> {
    fn to_predicate(&self) -> Predicate<'tcx> {
        Predicate::Projection(self.clone())
    }
}

impl<'tcx> Predicate<'tcx> {
    /// Iterates over the types in this predicate. Note that in all
    /// cases this is skipping over a binder, so late-bound regions
    /// with depth 0 are bound by the predicate.
    pub fn walk_tys(&self) -> IntoIter<Ty<'tcx>> {
        let vec: Vec<_> = match *self {
            ty::Predicate::Trait(ref data) => {
                data.skip_binder().input_types().collect()
            }
            ty::Predicate::Equate(ty::Binder(ref data)) => {
                vec![data.0, data.1]
            }
            ty::Predicate::TypeOutlives(ty::Binder(ref data)) => {
                vec![data.0]
            }
            ty::Predicate::RegionOutlives(..) => {
                vec![]
            }
            ty::Predicate::Projection(ref data) => {
                let trait_inputs = data.0.projection_ty.trait_ref.input_types();
                trait_inputs.chain(Some(data.0.ty)).collect()
            }
            ty::Predicate::WellFormed(data) => {
                vec![data]
            }
            ty::Predicate::ObjectSafe(_trait_def_id) => {
                vec![]
            }
            ty::Predicate::ClosureKind(_closure_def_id, _kind) => {
                vec![]
            }
        };

        // The only reason to collect into a vector here is that I was
        // too lazy to make the full (somewhat complicated) iterator
        // type that would be needed here. But I wanted this fn to
        // return an iterator conceptually, rather than a `Vec`, so as
        // to be closer to `Ty::walk`.
        vec.into_iter()
    }

    pub fn to_opt_poly_trait_ref(&self) -> Option<PolyTraitRef<'tcx>> {
        match *self {
            Predicate::Trait(ref t) => {
                Some(t.to_poly_trait_ref())
            }
            Predicate::Projection(..) |
            Predicate::Equate(..) |
            Predicate::RegionOutlives(..) |
            Predicate::WellFormed(..) |
            Predicate::ObjectSafe(..) |
            Predicate::ClosureKind(..) |
            Predicate::TypeOutlives(..) => {
                None
            }
        }
    }
}

/// Represents the bounds declared on a particular set of type
/// parameters.  Should eventually be generalized into a flag list of
/// where clauses.  You can obtain a `InstantiatedPredicates` list from a
/// `GenericPredicates` by using the `instantiate` method. Note that this method
/// reflects an important semantic invariant of `InstantiatedPredicates`: while
/// the `GenericPredicates` are expressed in terms of the bound type
/// parameters of the impl/trait/whatever, an `InstantiatedPredicates` instance
/// represented a set of bounds for some particular instantiation,
/// meaning that the generic parameters have been substituted with
/// their values.
///
/// Example:
///
///     struct Foo<T,U:Bar<T>> { ... }
///
/// Here, the `GenericPredicates` for `Foo` would contain a list of bounds like
/// `[[], [U:Bar<T>]]`.  Now if there were some particular reference
/// like `Foo<isize,usize>`, then the `InstantiatedPredicates` would be `[[],
/// [usize:Bar<isize>]]`.
#[derive(Clone)]
pub struct InstantiatedPredicates<'tcx> {
    pub predicates: Vec<Predicate<'tcx>>,
}

impl<'tcx> InstantiatedPredicates<'tcx> {
    pub fn empty() -> InstantiatedPredicates<'tcx> {
        InstantiatedPredicates { predicates: vec![] }
    }

    pub fn is_empty(&self) -> bool {
        self.predicates.is_empty()
    }
}

impl<'tcx> TraitRef<'tcx> {
    pub fn new(def_id: DefId, substs: &'tcx Substs<'tcx>) -> TraitRef<'tcx> {
        TraitRef { def_id: def_id, substs: substs }
    }

    pub fn self_ty(&self) -> Ty<'tcx> {
        self.substs.type_at(0)
    }

    pub fn input_types<'a>(&'a self) -> impl DoubleEndedIterator<Item=Ty<'tcx>> + 'a {
        // Select only the "input types" from a trait-reference. For
        // now this is all the types that appear in the
        // trait-reference, but it should eventually exclude
        // associated types.
        self.substs.types()
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
#[derive(Clone)]
pub struct ParameterEnvironment<'tcx> {
    /// See `construct_free_substs` for details.
    pub free_substs: &'tcx Substs<'tcx>,

    /// Each type parameter has an implicit region bound that
    /// indicates it must outlive at least the function body (the user
    /// may specify stronger requirements). This field indicates the
    /// region of the callee.
    pub implicit_region_bound: &'tcx ty::Region,

    /// Obligations that the caller must satisfy. This is basically
    /// the set of bounds on the in-scope type parameters, translated
    /// into Obligations, and elaborated and normalized.
    pub caller_bounds: Vec<ty::Predicate<'tcx>>,

    /// Scope that is attached to free regions for this scope. This
    /// is usually the id of the fn body, but for more abstract scopes
    /// like structs we often use the node-id of the struct.
    ///
    /// FIXME(#3696). It would be nice to refactor so that free
    /// regions don't have this implicit scope and instead introduce
    /// relationships in the environment.
    pub free_id_outlive: CodeExtent,

    /// A cache for `moves_by_default`.
    pub is_copy_cache: RefCell<FxHashMap<Ty<'tcx>, bool>>,

    /// A cache for `type_is_sized`
    pub is_sized_cache: RefCell<FxHashMap<Ty<'tcx>, bool>>,
}

impl<'a, 'tcx> ParameterEnvironment<'tcx> {
    pub fn with_caller_bounds(&self,
                              caller_bounds: Vec<ty::Predicate<'tcx>>)
                              -> ParameterEnvironment<'tcx>
    {
        ParameterEnvironment {
            free_substs: self.free_substs,
            implicit_region_bound: self.implicit_region_bound,
            caller_bounds: caller_bounds,
            free_id_outlive: self.free_id_outlive,
            is_copy_cache: RefCell::new(FxHashMap()),
            is_sized_cache: RefCell::new(FxHashMap()),
        }
    }

    /// Construct a parameter environment given an item, impl item, or trait item
    pub fn for_item(tcx: TyCtxt<'a, 'tcx, 'tcx>, id: NodeId)
                    -> ParameterEnvironment<'tcx> {
        match tcx.map.find(id) {
            Some(ast_map::NodeImplItem(ref impl_item)) => {
                match impl_item.node {
                    hir::ImplItemKind::Type(_) | hir::ImplItemKind::Const(..) => {
                        // associated types don't have their own entry (for some reason),
                        // so for now just grab environment for the impl
                        let impl_id = tcx.map.get_parent(id);
                        let impl_def_id = tcx.map.local_def_id(impl_id);
                        tcx.construct_parameter_environment(impl_item.span,
                                                            impl_def_id,
                                                            tcx.region_maps.item_extent(id))
                    }
                    hir::ImplItemKind::Method(_, ref body) => {
                        tcx.construct_parameter_environment(
                            impl_item.span,
                            tcx.map.local_def_id(id),
                            tcx.region_maps.call_site_extent(id, body.node_id))
                    }
                }
            }
            Some(ast_map::NodeTraitItem(trait_item)) => {
                match trait_item.node {
                    hir::TraitItemKind::Type(..) | hir::TraitItemKind::Const(..) => {
                        // associated types don't have their own entry (for some reason),
                        // so for now just grab environment for the trait
                        let trait_id = tcx.map.get_parent(id);
                        let trait_def_id = tcx.map.local_def_id(trait_id);
                        tcx.construct_parameter_environment(trait_item.span,
                                                            trait_def_id,
                                                            tcx.region_maps.item_extent(id))
                    }
                    hir::TraitItemKind::Method(_, ref body) => {
                        // Use call-site for extent (unless this is a
                        // trait method with no default; then fallback
                        // to the method id).
                        let extent = if let hir::TraitMethod::Provided(body_id) = *body {
                            // default impl: use call_site extent as free_id_outlive bound.
                            tcx.region_maps.call_site_extent(id, body_id.node_id)
                        } else {
                            // no default impl: use item extent as free_id_outlive bound.
                            tcx.region_maps.item_extent(id)
                        };
                        tcx.construct_parameter_environment(
                            trait_item.span,
                            tcx.map.local_def_id(id),
                            extent)
                    }
                }
            }
            Some(ast_map::NodeItem(item)) => {
                match item.node {
                    hir::ItemFn(.., body_id) => {
                        // We assume this is a function.
                        let fn_def_id = tcx.map.local_def_id(id);

                        tcx.construct_parameter_environment(
                            item.span,
                            fn_def_id,
                            tcx.region_maps.call_site_extent(id, body_id.node_id))
                    }
                    hir::ItemEnum(..) |
                    hir::ItemStruct(..) |
                    hir::ItemUnion(..) |
                    hir::ItemTy(..) |
                    hir::ItemImpl(..) |
                    hir::ItemConst(..) |
                    hir::ItemStatic(..) => {
                        let def_id = tcx.map.local_def_id(id);
                        tcx.construct_parameter_environment(item.span,
                                                            def_id,
                                                            tcx.region_maps.item_extent(id))
                    }
                    hir::ItemTrait(..) => {
                        let def_id = tcx.map.local_def_id(id);
                        tcx.construct_parameter_environment(item.span,
                                                            def_id,
                                                            tcx.region_maps.item_extent(id))
                    }
                    _ => {
                        span_bug!(item.span,
                                  "ParameterEnvironment::for_item():
                                   can't create a parameter \
                                   environment for this kind of item")
                    }
                }
            }
            Some(ast_map::NodeExpr(expr)) => {
                // This is a convenience to allow closures to work.
                if let hir::ExprClosure(.., body, _) = expr.node {
                    let def_id = tcx.map.local_def_id(id);
                    let base_def_id = tcx.closure_base_def_id(def_id);
                    tcx.construct_parameter_environment(
                        expr.span,
                        base_def_id,
                        tcx.region_maps.call_site_extent(id, body.node_id))
                } else {
                    tcx.empty_parameter_environment()
                }
            }
            Some(ast_map::NodeForeignItem(item)) => {
                let def_id = tcx.map.local_def_id(id);
                tcx.construct_parameter_environment(item.span,
                                                    def_id,
                                                    ROOT_CODE_EXTENT)
            }
            _ => {
                bug!("ParameterEnvironment::from_item(): \
                      `{}` is not an item",
                     tcx.map.node_to_string(id))
            }
        }
    }
}

bitflags! {
    flags AdtFlags: u32 {
        const NO_ADT_FLAGS        = 0,
        const IS_ENUM             = 1 << 0,
        const IS_DTORCK           = 1 << 1, // is this a dtorck type?
        const IS_DTORCK_VALID     = 1 << 2,
        const IS_PHANTOM_DATA     = 1 << 3,
        const IS_SIMD             = 1 << 4,
        const IS_FUNDAMENTAL      = 1 << 5,
        const IS_UNION            = 1 << 6,
    }
}

pub struct VariantDef {
    /// The variant's DefId. If this is a tuple-like struct,
    /// this is the DefId of the struct's ctor.
    pub did: DefId,
    pub name: Name, // struct's name if this is a struct
    pub disr_val: Disr,
    pub fields: Vec<FieldDef>,
    pub ctor_kind: CtorKind,
}

pub struct FieldDef {
    pub did: DefId,
    pub name: Name,
    pub vis: Visibility,
}

/// The definition of an abstract data type - a struct or enum.
///
/// These are all interned (by intern_adt_def) into the adt_defs
/// table.
pub struct AdtDef {
    pub did: DefId,
    pub variants: Vec<VariantDef>,
    destructor: Cell<Option<DefId>>,
    flags: Cell<AdtFlags>
}

impl PartialEq for AdtDef {
    // AdtDef are always interned and this is part of TyS equality
    #[inline]
    fn eq(&self, other: &Self) -> bool { self as *const _ == other as *const _ }
}

impl Eq for AdtDef {}

impl Hash for AdtDef {
    #[inline]
    fn hash<H: Hasher>(&self, s: &mut H) {
        (self as *const AdtDef).hash(s)
    }
}

impl<'tcx> serialize::UseSpecializedEncodable for &'tcx AdtDef {
    fn default_encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        self.did.encode(s)
    }
}

impl<'tcx> serialize::UseSpecializedDecodable for &'tcx AdtDef {}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AdtKind { Struct, Union, Enum }

impl<'a, 'gcx, 'tcx> AdtDef {
    fn new(tcx: TyCtxt<'a, 'gcx, 'tcx>,
           did: DefId,
           kind: AdtKind,
           variants: Vec<VariantDef>) -> Self {
        let mut flags = AdtFlags::NO_ADT_FLAGS;
        let attrs = tcx.get_attrs(did);
        if attr::contains_name(&attrs, "fundamental") {
            flags = flags | AdtFlags::IS_FUNDAMENTAL;
        }
        if tcx.lookup_simd(did) {
            flags = flags | AdtFlags::IS_SIMD;
        }
        if Some(did) == tcx.lang_items.phantom_data() {
            flags = flags | AdtFlags::IS_PHANTOM_DATA;
        }
        match kind {
            AdtKind::Enum => flags = flags | AdtFlags::IS_ENUM,
            AdtKind::Union => flags = flags | AdtFlags::IS_UNION,
            AdtKind::Struct => {}
        }
        AdtDef {
            did: did,
            variants: variants,
            flags: Cell::new(flags),
            destructor: Cell::new(None),
        }
    }

    fn calculate_dtorck(&'gcx self, tcx: TyCtxt) {
        if tcx.is_adt_dtorck(self) {
            self.flags.set(self.flags.get() | AdtFlags::IS_DTORCK);
        }
        self.flags.set(self.flags.get() | AdtFlags::IS_DTORCK_VALID)
    }

    #[inline]
    pub fn is_struct(&self) -> bool {
        !self.is_union() && !self.is_enum()
    }

    #[inline]
    pub fn is_union(&self) -> bool {
        self.flags.get().intersects(AdtFlags::IS_UNION)
    }

    #[inline]
    pub fn is_enum(&self) -> bool {
        self.flags.get().intersects(AdtFlags::IS_ENUM)
    }

    /// Returns the kind of the ADT - Struct or Enum.
    #[inline]
    pub fn adt_kind(&self) -> AdtKind {
        if self.is_enum() {
            AdtKind::Enum
        } else if self.is_union() {
            AdtKind::Union
        } else {
            AdtKind::Struct
        }
    }

    pub fn descr(&self) -> &'static str {
        match self.adt_kind() {
            AdtKind::Struct => "struct",
            AdtKind::Union => "union",
            AdtKind::Enum => "enum",
        }
    }

    pub fn variant_descr(&self) -> &'static str {
        match self.adt_kind() {
            AdtKind::Struct => "struct",
            AdtKind::Union => "union",
            AdtKind::Enum => "variant",
        }
    }

    /// Returns whether this is a dtorck type. If this returns
    /// true, this type being safe for destruction requires it to be
    /// alive; Otherwise, only the contents are required to be.
    #[inline]
    pub fn is_dtorck(&'gcx self, tcx: TyCtxt) -> bool {
        if !self.flags.get().intersects(AdtFlags::IS_DTORCK_VALID) {
            self.calculate_dtorck(tcx)
        }
        self.flags.get().intersects(AdtFlags::IS_DTORCK)
    }

    /// Returns whether this type is #[fundamental] for the purposes
    /// of coherence checking.
    #[inline]
    pub fn is_fundamental(&self) -> bool {
        self.flags.get().intersects(AdtFlags::IS_FUNDAMENTAL)
    }

    #[inline]
    pub fn is_simd(&self) -> bool {
        self.flags.get().intersects(AdtFlags::IS_SIMD)
    }

    /// Returns true if this is PhantomData<T>.
    #[inline]
    pub fn is_phantom_data(&self) -> bool {
        self.flags.get().intersects(AdtFlags::IS_PHANTOM_DATA)
    }

    /// Returns whether this type has a destructor.
    pub fn has_dtor(&self) -> bool {
        self.dtor_kind().is_present()
    }

    /// Asserts this is a struct and returns the struct's unique
    /// variant.
    pub fn struct_variant(&self) -> &VariantDef {
        assert!(!self.is_enum());
        &self.variants[0]
    }

    #[inline]
    pub fn predicates(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> GenericPredicates<'gcx> {
        tcx.item_predicates(self.did)
    }

    /// Returns an iterator over all fields contained
    /// by this ADT.
    #[inline]
    pub fn all_fields<'s>(&'s self) -> impl Iterator<Item = &'s FieldDef> {
        self.variants.iter().flat_map(|v| v.fields.iter())
    }

    #[inline]
    pub fn is_univariant(&self) -> bool {
        self.variants.len() == 1
    }

    pub fn is_payloadfree(&self) -> bool {
        !self.variants.is_empty() &&
            self.variants.iter().all(|v| v.fields.is_empty())
    }

    pub fn variant_with_id(&self, vid: DefId) -> &VariantDef {
        self.variants
            .iter()
            .find(|v| v.did == vid)
            .expect("variant_with_id: unknown variant")
    }

    pub fn variant_index_with_id(&self, vid: DefId) -> usize {
        self.variants
            .iter()
            .position(|v| v.did == vid)
            .expect("variant_index_with_id: unknown variant")
    }

    pub fn variant_of_def(&self, def: Def) -> &VariantDef {
        match def {
            Def::Variant(vid) | Def::VariantCtor(vid, ..) => self.variant_with_id(vid),
            Def::Struct(..) | Def::StructCtor(..) | Def::Union(..) |
            Def::TyAlias(..) | Def::AssociatedTy(..) | Def::SelfTy(..) => self.struct_variant(),
            _ => bug!("unexpected def {:?} in variant_of_def", def)
        }
    }

    pub fn destructor(&self) -> Option<DefId> {
        self.destructor.get()
    }

    pub fn set_destructor(&self, dtor: DefId) {
        self.destructor.set(Some(dtor));
    }

    pub fn dtor_kind(&self) -> DtorKind {
        match self.destructor.get() {
            Some(_) => TraitDtor,
            None => NoDtor,
        }
    }

    /// Returns a simpler type such that `Self: Sized` if and only
    /// if that type is Sized, or `TyErr` if this type is recursive.
    ///
    /// HACK: instead of returning a list of types, this function can
    /// return a tuple. In that case, the result is Sized only if
    /// all elements of the tuple are Sized.
    ///
    /// This is generally the `struct_tail` if this is a struct, or a
    /// tuple of them if this is an enum.
    ///
    /// Oddly enough, checking that the sized-constraint is Sized is
    /// actually more expressive than checking all members:
    /// the Sized trait is inductive, so an associated type that references
    /// Self would prevent its containing ADT from being Sized.
    ///
    /// Due to normalization being eager, this applies even if
    /// the associated type is behind a pointer, e.g. issue #31299.
    pub fn sized_constraint(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx> {
        self.calculate_sized_constraint_inner(tcx.global_tcx(), &mut Vec::new())
    }

    /// Calculates the Sized-constraint.
    ///
    /// As the Sized-constraint of enums can be a *set* of types,
    /// the Sized-constraint may need to be a set also. Because introducing
    /// a new type of IVar is currently a complex affair, the Sized-constraint
    /// may be a tuple.
    ///
    /// In fact, there are only a few options for the constraint:
    ///     - `bool`, if the type is always Sized
    ///     - an obviously-unsized type
    ///     - a type parameter or projection whose Sizedness can't be known
    ///     - a tuple of type parameters or projections, if there are multiple
    ///       such.
    ///     - a TyError, if a type contained itself. The representability
    ///       check should catch this case.
    fn calculate_sized_constraint_inner(&self,
                                        tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                        stack: &mut Vec<DefId>)
                                        -> Ty<'tcx>
    {
        if let Some(ty) = tcx.adt_sized_constraint.borrow().get(&self.did) {
            return ty;
        }

        // Follow the memoization pattern: push the computation of
        // DepNode::SizedConstraint as our current task.
        let _task = tcx.dep_graph.in_task(DepNode::SizedConstraint(self.did));

        if stack.contains(&self.did) {
            debug!("calculate_sized_constraint: {:?} is recursive", self);
            // This should be reported as an error by `check_representable`.
            //
            // Consider the type as Sized in the meanwhile to avoid
            // further errors.
            tcx.adt_sized_constraint.borrow_mut().insert(self.did, tcx.types.err);
            return tcx.types.err;
        }

        stack.push(self.did);

        let tys : Vec<_> =
            self.variants.iter().flat_map(|v| {
                v.fields.last()
            }).flat_map(|f| {
                let ty = tcx.item_type(f.did);
                self.sized_constraint_for_ty(tcx, stack, ty)
            }).collect();

        let self_ = stack.pop().unwrap();
        assert_eq!(self_, self.did);

        let ty = match tys.len() {
            _ if tys.references_error() => tcx.types.err,
            0 => tcx.types.bool,
            1 => tys[0],
            _ => tcx.intern_tup(&tys[..])
        };

        let old = tcx.adt_sized_constraint.borrow().get(&self.did).cloned();
        match old {
            Some(old_ty) => {
                debug!("calculate_sized_constraint: {:?} recurred", self);
                assert_eq!(old_ty, tcx.types.err);
                old_ty
            }
            None => {
                debug!("calculate_sized_constraint: {:?} => {:?}", self, ty);
                tcx.adt_sized_constraint.borrow_mut().insert(self.did, ty);
                ty
            }
        }
    }

    fn sized_constraint_for_ty(&self,
                               tcx: TyCtxt<'a, 'tcx, 'tcx>,
                               stack: &mut Vec<DefId>,
                               ty: Ty<'tcx>)
                               -> Vec<Ty<'tcx>> {
        let result = match ty.sty {
            TyBool | TyChar | TyInt(..) | TyUint(..) | TyFloat(..) |
            TyBox(..) | TyRawPtr(..) | TyRef(..) | TyFnDef(..) | TyFnPtr(_) |
            TyArray(..) | TyClosure(..) | TyNever => {
                vec![]
            }

            TyStr | TyDynamic(..) | TySlice(_) | TyError => {
                // these are never sized - return the target type
                vec![ty]
            }

            TyTuple(ref tys) => {
                match tys.last() {
                    None => vec![],
                    Some(ty) => self.sized_constraint_for_ty(tcx, stack, ty)
                }
            }

            TyAdt(adt, substs) => {
                // recursive case
                let adt_ty =
                    adt.calculate_sized_constraint_inner(tcx, stack)
                       .subst(tcx, substs);
                debug!("sized_constraint_for_ty({:?}) intermediate = {:?}",
                       ty, adt_ty);
                if let ty::TyTuple(ref tys) = adt_ty.sty {
                    tys.iter().flat_map(|ty| {
                        self.sized_constraint_for_ty(tcx, stack, ty)
                    }).collect()
                } else {
                    self.sized_constraint_for_ty(tcx, stack, adt_ty)
                }
            }

            TyProjection(..) | TyAnon(..) => {
                // must calculate explicitly.
                // FIXME: consider special-casing always-Sized projections
                vec![ty]
            }

            TyParam(..) => {
                // perf hack: if there is a `T: Sized` bound, then
                // we know that `T` is Sized and do not need to check
                // it on the impl.

                let sized_trait = match tcx.lang_items.sized_trait() {
                    Some(x) => x,
                    _ => return vec![ty]
                };
                let sized_predicate = Binder(TraitRef {
                    def_id: sized_trait,
                    substs: tcx.mk_substs_trait(ty, &[])
                }).to_predicate();
                let predicates = tcx.item_predicates(self.did).predicates;
                if predicates.into_iter().any(|p| p == sized_predicate) {
                    vec![]
                } else {
                    vec![ty]
                }
            }

            TyInfer(..) => {
                bug!("unexpected type `{:?}` in sized_constraint_for_ty",
                     ty)
            }
        };
        debug!("sized_constraint_for_ty({:?}) = {:?}", ty, result);
        result
    }
}

impl<'a, 'gcx, 'tcx> VariantDef {
    #[inline]
    pub fn find_field_named(&self,
                            name: ast::Name)
                            -> Option<&FieldDef> {
        self.fields.iter().find(|f| f.name == name)
    }

    #[inline]
    pub fn index_of_field_named(&self,
                                name: ast::Name)
                                -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }

    #[inline]
    pub fn field_named(&self, name: ast::Name) -> &FieldDef {
        self.find_field_named(name).unwrap()
    }
}

impl<'a, 'gcx, 'tcx> FieldDef {
    pub fn ty(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>, subst: &Substs<'tcx>) -> Ty<'tcx> {
        tcx.item_type(self.did).subst(tcx, subst)
    }
}

/// Records the substitutions used to translate the polytype for an
/// item into the monotype of an item reference.
#[derive(Clone, RustcEncodable, RustcDecodable)]
pub struct ItemSubsts<'tcx> {
    pub substs: &'tcx Substs<'tcx>,
}

#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
pub enum ClosureKind {
    // Warning: Ordering is significant here! The ordering is chosen
    // because the trait Fn is a subtrait of FnMut and so in turn, and
    // hence we order it so that Fn < FnMut < FnOnce.
    Fn,
    FnMut,
    FnOnce,
}

impl<'a, 'tcx> ClosureKind {
    pub fn trait_did(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> DefId {
        match *self {
            ClosureKind::Fn => tcx.require_lang_item(FnTraitLangItem),
            ClosureKind::FnMut => {
                tcx.require_lang_item(FnMutTraitLangItem)
            }
            ClosureKind::FnOnce => {
                tcx.require_lang_item(FnOnceTraitLangItem)
            }
        }
    }

    /// True if this a type that impls this closure kind
    /// must also implement `other`.
    pub fn extends(self, other: ty::ClosureKind) -> bool {
        match (self, other) {
            (ClosureKind::Fn, ClosureKind::Fn) => true,
            (ClosureKind::Fn, ClosureKind::FnMut) => true,
            (ClosureKind::Fn, ClosureKind::FnOnce) => true,
            (ClosureKind::FnMut, ClosureKind::FnMut) => true,
            (ClosureKind::FnMut, ClosureKind::FnOnce) => true,
            (ClosureKind::FnOnce, ClosureKind::FnOnce) => true,
            _ => false,
        }
    }
}

impl<'tcx> TyS<'tcx> {
    /// Iterator that walks `self` and any types reachable from
    /// `self`, in depth-first order. Note that just walks the types
    /// that appear in `self`, it does not descend into the fields of
    /// structs or variants. For example:
    ///
    /// ```notrust
    /// isize => { isize }
    /// Foo<Bar<isize>> => { Foo<Bar<isize>>, Bar<isize>, isize }
    /// [isize] => { [isize], isize }
    /// ```
    pub fn walk(&'tcx self) -> TypeWalker<'tcx> {
        TypeWalker::new(self)
    }

    /// Iterator that walks the immediate children of `self`.  Hence
    /// `Foo<Bar<i32>, u32>` yields the sequence `[Bar<i32>, u32]`
    /// (but not `i32`, like `walk`).
    pub fn walk_shallow(&'tcx self) -> AccIntoIter<walk::TypeWalkerArray<'tcx>> {
        walk::walk_shallow(self)
    }

    /// Walks `ty` and any types appearing within `ty`, invoking the
    /// callback `f` on each type. If the callback returns false, then the
    /// children of the current type are ignored.
    ///
    /// Note: prefer `ty.walk()` where possible.
    pub fn maybe_walk<F>(&'tcx self, mut f: F)
        where F : FnMut(Ty<'tcx>) -> bool
    {
        let mut walker = self.walk();
        while let Some(ty) = walker.next() {
            if !f(ty) {
                walker.skip_current_subtree();
            }
        }
    }
}

impl<'tcx> ItemSubsts<'tcx> {
    pub fn is_noop(&self) -> bool {
        self.substs.is_noop()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LvaluePreference {
    PreferMutLvalue,
    NoPreference
}

impl LvaluePreference {
    pub fn from_mutbl(m: hir::Mutability) -> Self {
        match m {
            hir::MutMutable => PreferMutLvalue,
            hir::MutImmutable => NoPreference,
        }
    }
}

/// Helper for looking things up in the various maps that are populated during
/// typeck::collect (e.g., `tcx.associated_items`, `tcx.types`, etc).  All of
/// these share the pattern that if the id is local, it should have been loaded
/// into the map by the `typeck::collect` phase.  If the def-id is external,
/// then we have to go consult the crate loading code (and cache the result for
/// the future).
fn lookup_locally_or_in_crate_store<M, F>(descr: &str,
                                          def_id: DefId,
                                          map: &M,
                                          load_external: F)
                                          -> M::Value where
    M: MemoizationMap<Key=DefId>,
    F: FnOnce() -> M::Value,
{
    map.memoize(def_id, || {
        if def_id.is_local() {
            bug!("No def'n found for {:?} in tcx.{}", def_id, descr);
        }
        load_external()
    })
}

impl BorrowKind {
    pub fn from_mutbl(m: hir::Mutability) -> BorrowKind {
        match m {
            hir::MutMutable => MutBorrow,
            hir::MutImmutable => ImmBorrow,
        }
    }

    /// Returns a mutability `m` such that an `&m T` pointer could be used to obtain this borrow
    /// kind. Because borrow kinds are richer than mutabilities, we sometimes have to pick a
    /// mutability that is stronger than necessary so that it at least *would permit* the borrow in
    /// question.
    pub fn to_mutbl_lossy(self) -> hir::Mutability {
        match self {
            MutBorrow => hir::MutMutable,
            ImmBorrow => hir::MutImmutable,

            // We have no type corresponding to a unique imm borrow, so
            // use `&mut`. It gives all the capabilities of an `&uniq`
            // and hence is a safe "over approximation".
            UniqueImmBorrow => hir::MutMutable,
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

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    pub fn body_tables(self, body: hir::BodyId) -> &'gcx Tables<'gcx> {
        self.item_tables(self.map.body_owner_def_id(body))
    }

    pub fn item_tables(self, def_id: DefId) -> &'gcx Tables<'gcx> {
        self.tables.memoize(def_id, || {
            if def_id.is_local() {
                // Closures' tables come from their outermost function,
                // as they are part of the same "inference environment".
                let outer_def_id = self.closure_base_def_id(def_id);
                if outer_def_id != def_id {
                    return self.item_tables(outer_def_id);
                }

                bug!("No def'n found for {:?} in tcx.tables", def_id);
            }

            // Cross-crate side-tables only exist alongside serialized HIR.
            self.sess.cstore.maybe_get_item_body(self.global_tcx(), def_id).map(|_| {
                self.tables.borrow()[&def_id]
            }).unwrap_or_else(|| {
                bug!("tcx.item_tables({:?}): missing from metadata", def_id)
            })
        })
    }

    pub fn expr_span(self, id: NodeId) -> Span {
        match self.map.find(id) {
            Some(ast_map::NodeExpr(e)) => {
                e.span
            }
            Some(f) => {
                bug!("Node id {} is not an expr: {:?}", id, f);
            }
            None => {
                bug!("Node id {} is not present in the node map", id);
            }
        }
    }

    pub fn local_var_name_str(self, id: NodeId) -> InternedString {
        match self.map.find(id) {
            Some(ast_map::NodeLocal(pat)) => {
                match pat.node {
                    hir::PatKind::Binding(_, _, ref path1, _) => path1.node.as_str(),
                    _ => {
                        bug!("Variable id {} maps to {:?}, not local", id, pat);
                    },
                }
            },
            r => bug!("Variable id {} maps to {:?}, not local", id, r),
        }
    }

    pub fn expr_is_lval(self, expr: &hir::Expr) -> bool {
         match expr.node {
            hir::ExprPath(hir::QPath::Resolved(_, ref path)) => {
                match path.def {
                    Def::Local(..) | Def::Upvar(..) | Def::Static(..) | Def::Err => true,
                    _ => false,
                }
            }

            hir::ExprType(ref e, _) => {
                self.expr_is_lval(e)
            }

            hir::ExprUnary(hir::UnDeref, _) |
            hir::ExprField(..) |
            hir::ExprTupField(..) |
            hir::ExprIndex(..) => {
                true
            }

            // Partially qualified paths in expressions can only legally
            // refer to associated items which are always rvalues.
            hir::ExprPath(hir::QPath::TypeRelative(..)) |

            hir::ExprCall(..) |
            hir::ExprMethodCall(..) |
            hir::ExprStruct(..) |
            hir::ExprTup(..) |
            hir::ExprIf(..) |
            hir::ExprMatch(..) |
            hir::ExprClosure(..) |
            hir::ExprBlock(..) |
            hir::ExprRepeat(..) |
            hir::ExprArray(..) |
            hir::ExprBreak(..) |
            hir::ExprAgain(..) |
            hir::ExprRet(..) |
            hir::ExprWhile(..) |
            hir::ExprLoop(..) |
            hir::ExprAssign(..) |
            hir::ExprInlineAsm(..) |
            hir::ExprAssignOp(..) |
            hir::ExprLit(_) |
            hir::ExprUnary(..) |
            hir::ExprBox(..) |
            hir::ExprAddrOf(..) |
            hir::ExprBinary(..) |
            hir::ExprCast(..) => {
                false
            }
        }
    }

    pub fn provided_trait_methods(self, id: DefId) -> Vec<AssociatedItem> {
        self.associated_items(id)
            .filter(|item| item.kind == AssociatedKind::Method && item.defaultness.has_value())
            .collect()
    }

    pub fn trait_impl_polarity(self, id: DefId) -> hir::ImplPolarity {
        if let Some(id) = self.map.as_local_node_id(id) {
            match self.map.expect_item(id).node {
                hir::ItemImpl(_, polarity, ..) => polarity,
                ref item => bug!("trait_impl_polarity: {:?} not an impl", item)
            }
        } else {
            self.sess.cstore.impl_polarity(id)
        }
    }

    pub fn custom_coerce_unsized_kind(self, did: DefId) -> adjustment::CustomCoerceUnsized {
        self.custom_coerce_unsized_kinds.memoize(did, || {
            let (kind, src) = if did.krate != LOCAL_CRATE {
                (self.sess.cstore.custom_coerce_unsized_kind(did), "external")
            } else {
                (None, "local")
            };

            match kind {
                Some(kind) => kind,
                None => {
                    bug!("custom_coerce_unsized_kind: \
                          {} impl `{}` is missing its kind",
                          src, self.item_path_str(did));
                }
            }
        })
    }

    pub fn associated_item(self, def_id: DefId) -> AssociatedItem {
        self.associated_items.memoize(def_id, || {
            if !def_id.is_local() {
                return self.sess.cstore.associated_item(def_id)
                           .expect("missing AssociatedItem in metadata");
            }

            // When the user asks for a given associated item, we
            // always go ahead and convert all the associated items in
            // the container. Note that we are also careful only to
            // ever register a read on the *container* of the assoc
            // item, not the assoc item itself. This prevents changes
            // in the details of an item (for example, the type to
            // which an associated type is bound) from contaminating
            // those tasks that just need to scan the names of items
            // and so forth.

            let id = self.map.as_local_node_id(def_id).unwrap();
            let parent_id = self.map.get_parent(id);
            let parent_def_id = self.map.local_def_id(parent_id);
            let parent_item = self.map.expect_item(parent_id);
            match parent_item.node {
                hir::ItemImpl(.., ref impl_trait_ref, _, ref impl_item_refs) => {
                    for impl_item_ref in impl_item_refs {
                        let assoc_item =
                            self.associated_item_from_impl_item_ref(parent_def_id,
                                                                    impl_trait_ref.is_some(),
                                                                    impl_item_ref);
                        self.associated_items.borrow_mut().insert(assoc_item.def_id, assoc_item);
                    }
                }

                hir::ItemTrait(.., ref trait_item_refs) => {
                    for trait_item_ref in trait_item_refs {
                        let assoc_item =
                            self.associated_item_from_trait_item_ref(parent_def_id, trait_item_ref);
                        self.associated_items.borrow_mut().insert(assoc_item.def_id, assoc_item);
                    }
                }

                ref r => {
                    panic!("unexpected container of associated items: {:?}", r)
                }
            }

            // memoize wants us to return something, so return
            // the one we generated for this def-id
            *self.associated_items.borrow().get(&def_id).unwrap()
        })
    }

    fn associated_item_from_trait_item_ref(self,
                                           parent_def_id: DefId,
                                           trait_item_ref: &hir::TraitItemRef)
                                           -> AssociatedItem {
        let def_id = self.map.local_def_id(trait_item_ref.id.node_id);
        let (kind, has_self) = match trait_item_ref.kind {
            hir::AssociatedItemKind::Const => (ty::AssociatedKind::Const, false),
            hir::AssociatedItemKind::Method { has_self } => {
                (ty::AssociatedKind::Method, has_self)
            }
            hir::AssociatedItemKind::Type => (ty::AssociatedKind::Type, false),
        };

        AssociatedItem {
            name: trait_item_ref.name,
            kind: kind,
            vis: Visibility::from_hir(&hir::Inherited, trait_item_ref.id.node_id, self),
            defaultness: trait_item_ref.defaultness,
            def_id: def_id,
            container: TraitContainer(parent_def_id),
            method_has_self_argument: has_self
        }
    }

    fn associated_item_from_impl_item_ref(self,
                                          parent_def_id: DefId,
                                          from_trait_impl: bool,
                                          impl_item_ref: &hir::ImplItemRef)
                                          -> AssociatedItem {
        let def_id = self.map.local_def_id(impl_item_ref.id.node_id);
        let (kind, has_self) = match impl_item_ref.kind {
            hir::AssociatedItemKind::Const => (ty::AssociatedKind::Const, false),
            hir::AssociatedItemKind::Method { has_self } => {
                (ty::AssociatedKind::Method, has_self)
            }
            hir::AssociatedItemKind::Type => (ty::AssociatedKind::Type, false),
        };

        // Trait impl items are always public.
        let public = hir::Public;
        let vis = if from_trait_impl { &public } else { &impl_item_ref.vis };

        ty::AssociatedItem {
            name: impl_item_ref.name,
            kind: kind,
            vis: ty::Visibility::from_hir(vis, impl_item_ref.id.node_id, self),
            defaultness: impl_item_ref.defaultness,
            def_id: def_id,
            container: ImplContainer(parent_def_id),
            method_has_self_argument: has_self
        }
    }

    pub fn associated_item_def_ids(self, def_id: DefId) -> Rc<Vec<DefId>> {
        self.associated_item_def_ids.memoize(def_id, || {
            if !def_id.is_local() {
                return Rc::new(self.sess.cstore.associated_item_def_ids(def_id));
            }

            let id = self.map.as_local_node_id(def_id).unwrap();
            let item = self.map.expect_item(id);
            let vec: Vec<_> = match item.node {
                hir::ItemTrait(.., ref trait_item_refs) => {
                    trait_item_refs.iter()
                                   .map(|trait_item_ref| trait_item_ref.id)
                                   .map(|id| self.map.local_def_id(id.node_id))
                                   .collect()
                }
                hir::ItemImpl(.., ref impl_item_refs) => {
                    impl_item_refs.iter()
                                  .map(|impl_item_ref| impl_item_ref.id)
                                  .map(|id| self.map.local_def_id(id.node_id))
                                  .collect()
                }
                _ => span_bug!(item.span, "associated_item_def_ids: not impl or trait")
            };
            Rc::new(vec)
        })
    }

    #[inline] // FIXME(#35870) Avoid closures being unexported due to impl Trait.
    pub fn associated_items(self, def_id: DefId)
                            -> impl Iterator<Item = ty::AssociatedItem> + 'a {
        let def_ids = self.associated_item_def_ids(def_id);
        (0..def_ids.len()).map(move |i| self.associated_item(def_ids[i]))
    }

    /// Returns the trait-ref corresponding to a given impl, or None if it is
    /// an inherent impl.
    pub fn impl_trait_ref(self, id: DefId) -> Option<TraitRef<'gcx>> {
        lookup_locally_or_in_crate_store(
            "impl_trait_refs", id, &self.impl_trait_refs,
            || self.sess.cstore.impl_trait_ref(self.global_tcx(), id))
    }

    // Returns `ty::VariantDef` if `def` refers to a struct,
    // or variant or their constructors, panics otherwise.
    pub fn expect_variant_def(self, def: Def) -> &'tcx VariantDef {
        match def {
            Def::Variant(did) | Def::VariantCtor(did, ..) => {
                let enum_did = self.parent_def_id(did).unwrap();
                self.lookup_adt_def(enum_did).variant_with_id(did)
            }
            Def::Struct(did) | Def::Union(did) => {
                self.lookup_adt_def(did).struct_variant()
            }
            Def::StructCtor(ctor_did, ..) => {
                let did = self.parent_def_id(ctor_did).expect("struct ctor has no parent");
                self.lookup_adt_def(did).struct_variant()
            }
            _ => bug!("expect_variant_def used with unexpected def {:?}", def)
        }
    }

    pub fn def_key(self, id: DefId) -> ast_map::DefKey {
        if id.is_local() {
            self.map.def_key(id)
        } else {
            self.sess.cstore.def_key(id)
        }
    }

    /// Convert a `DefId` into its fully expanded `DefPath` (every
    /// `DefId` is really just an interned def-path).
    ///
    /// Note that if `id` is not local to this crate, the result will
    //  be a non-local `DefPath`.
    pub fn def_path(self, id: DefId) -> ast_map::DefPath {
        if id.is_local() {
            self.map.def_path(id)
        } else {
            self.sess.cstore.def_path(id)
        }
    }

    pub fn def_span(self, def_id: DefId) -> Span {
        if let Some(id) = self.map.as_local_node_id(def_id) {
            self.map.span(id)
        } else {
            self.sess.cstore.def_span(&self.sess, def_id)
        }
    }

    pub fn vis_is_accessible_from(self, vis: Visibility, block: NodeId) -> bool {
        vis.is_accessible_from(self.map.local_def_id(self.map.get_module_parent(block)), self)
    }

    pub fn item_name(self, id: DefId) -> ast::Name {
        if let Some(id) = self.map.as_local_node_id(id) {
            self.map.name(id)
        } else if id.index == CRATE_DEF_INDEX {
            self.sess.cstore.original_crate_name(id.krate)
        } else {
            let def_key = self.sess.cstore.def_key(id);
            // The name of a StructCtor is that of its struct parent.
            if let ast_map::DefPathData::StructCtor = def_key.disambiguated_data.data {
                self.item_name(DefId {
                    krate: id.krate,
                    index: def_key.parent.unwrap()
                })
            } else {
                def_key.disambiguated_data.data.get_opt_name().unwrap_or_else(|| {
                    bug!("item_name: no name for {:?}", self.def_path(id));
                })
            }
        }
    }

    // If the given item is in an external crate, looks up its type and adds it to
    // the type cache. Returns the type parameters and type.
    pub fn item_type(self, did: DefId) -> Ty<'gcx> {
        lookup_locally_or_in_crate_store(
            "item_types", did, &self.item_types,
            || self.sess.cstore.item_type(self.global_tcx(), did))
    }

    /// Given the did of a trait, returns its canonical trait ref.
    pub fn lookup_trait_def(self, did: DefId) -> &'gcx TraitDef {
        lookup_locally_or_in_crate_store(
            "trait_defs", did, &self.trait_defs,
            || self.alloc_trait_def(self.sess.cstore.trait_def(self.global_tcx(), did))
        )
    }

    /// Given the did of an ADT, return a reference to its definition.
    pub fn lookup_adt_def(self, did: DefId) -> &'gcx AdtDef {
        lookup_locally_or_in_crate_store(
            "adt_defs", did, &self.adt_defs,
            || self.sess.cstore.adt_def(self.global_tcx(), did))
    }

    /// Given the did of an item, returns its generics.
    pub fn item_generics(self, did: DefId) -> &'gcx Generics<'gcx> {
        lookup_locally_or_in_crate_store(
            "generics", did, &self.generics,
            || self.alloc_generics(self.sess.cstore.item_generics(self.global_tcx(), did)))
    }

    /// Given the did of an item, returns its full set of predicates.
    pub fn item_predicates(self, did: DefId) -> GenericPredicates<'gcx> {
        lookup_locally_or_in_crate_store(
            "predicates", did, &self.predicates,
            || self.sess.cstore.item_predicates(self.global_tcx(), did))
    }

    /// Given the did of a trait, returns its superpredicates.
    pub fn item_super_predicates(self, did: DefId) -> GenericPredicates<'gcx> {
        lookup_locally_or_in_crate_store(
            "super_predicates", did, &self.super_predicates,
            || self.sess.cstore.item_super_predicates(self.global_tcx(), did))
    }

    /// Given the did of an item, returns its MIR, borrowed immutably.
    pub fn item_mir(self, did: DefId) -> Ref<'gcx, Mir<'gcx>> {
        lookup_locally_or_in_crate_store("mir_map", did, &self.mir_map, || {
            let mir = self.sess.cstore.get_item_mir(self.global_tcx(), did);
            let mir = self.alloc_mir(mir);

            // Perma-borrow MIR from extern crates to prevent mutation.
            mem::forget(mir.borrow());

            mir
        }).borrow()
    }

    /// If `type_needs_drop` returns true, then `ty` is definitely
    /// non-copy and *might* have a destructor attached; if it returns
    /// false, then `ty` definitely has no destructor (i.e. no drop glue).
    ///
    /// (Note that this implies that if `ty` has a destructor attached,
    /// then `type_needs_drop` will definitely return `true` for `ty`.)
    pub fn type_needs_drop_given_env(self,
                                     ty: Ty<'gcx>,
                                     param_env: &ty::ParameterEnvironment<'gcx>) -> bool {
        // Issue #22536: We first query type_moves_by_default.  It sees a
        // normalized version of the type, and therefore will definitely
        // know whether the type implements Copy (and thus needs no
        // cleanup/drop/zeroing) ...
        let tcx = self.global_tcx();
        let implements_copy = !ty.moves_by_default(tcx, param_env, DUMMY_SP);

        if implements_copy { return false; }

        // ... (issue #22536 continued) but as an optimization, still use
        // prior logic of asking if the `needs_drop` bit is set; we need
        // not zero non-Copy types if they have no destructor.

        // FIXME(#22815): Note that calling `ty::type_contents` is a
        // conservative heuristic; it may report that `needs_drop` is set
        // when actual type does not actually have a destructor associated
        // with it. But since `ty` absolutely did not have the `Copy`
        // bound attached (see above), it is sound to treat it as having a
        // destructor (e.g. zero its memory on move).

        let contents = ty.type_contents(tcx);
        debug!("type_needs_drop ty={:?} contents={:?}", ty, contents);
        contents.needs_drop(tcx)
    }

    /// Get the attributes of a definition.
    pub fn get_attrs(self, did: DefId) -> Cow<'gcx, [ast::Attribute]> {
        if let Some(id) = self.map.as_local_node_id(did) {
            Cow::Borrowed(self.map.attrs(id))
        } else {
            Cow::Owned(self.sess.cstore.item_attrs(did))
        }
    }

    /// Determine whether an item is annotated with an attribute
    pub fn has_attr(self, did: DefId, attr: &str) -> bool {
        self.get_attrs(did).iter().any(|item| item.check_name(attr))
    }

    /// Determine whether an item is annotated with `#[repr(packed)]`
    pub fn lookup_packed(self, did: DefId) -> bool {
        self.lookup_repr_hints(did).contains(&attr::ReprPacked)
    }

    /// Determine whether an item is annotated with `#[simd]`
    pub fn lookup_simd(self, did: DefId) -> bool {
        self.has_attr(did, "simd")
            || self.lookup_repr_hints(did).contains(&attr::ReprSimd)
    }

    pub fn item_variances(self, item_id: DefId) -> Rc<Vec<ty::Variance>> {
        lookup_locally_or_in_crate_store(
            "item_variance_map", item_id, &self.item_variance_map,
            || Rc::new(self.sess.cstore.item_variances(item_id)))
    }

    pub fn trait_has_default_impl(self, trait_def_id: DefId) -> bool {
        self.populate_implementations_for_trait_if_necessary(trait_def_id);

        let def = self.lookup_trait_def(trait_def_id);
        def.flags.get().intersects(TraitFlags::HAS_DEFAULT_IMPL)
    }

    /// Records a trait-to-implementation mapping.
    pub fn record_trait_has_default_impl(self, trait_def_id: DefId) {
        let def = self.lookup_trait_def(trait_def_id);
        def.flags.set(def.flags.get() | TraitFlags::HAS_DEFAULT_IMPL)
    }

    /// Populates the type context with all the inherent implementations for
    /// the given type if necessary.
    pub fn populate_inherent_implementations_for_type_if_necessary(self,
                                                                   type_id: DefId) {
        if type_id.is_local() {
            return
        }

        // The type is not local, hence we are reading this out of
        // metadata and don't need to track edges.
        let _ignore = self.dep_graph.in_ignore();

        if self.populated_external_types.borrow().contains(&type_id) {
            return
        }

        debug!("populate_inherent_implementations_for_type_if_necessary: searching for {:?}",
               type_id);

        let inherent_impls = self.sess.cstore.inherent_implementations_for_type(type_id);

        self.inherent_impls.borrow_mut().insert(type_id, inherent_impls);
        self.populated_external_types.borrow_mut().insert(type_id);
    }

    /// Populates the type context with all the implementations for the given
    /// trait if necessary.
    pub fn populate_implementations_for_trait_if_necessary(self, trait_id: DefId) {
        if trait_id.is_local() {
            return
        }

        // The type is not local, hence we are reading this out of
        // metadata and don't need to track edges.
        let _ignore = self.dep_graph.in_ignore();

        let def = self.lookup_trait_def(trait_id);
        if def.flags.get().intersects(TraitFlags::IMPLS_VALID) {
            return;
        }

        debug!("populate_implementations_for_trait_if_necessary: searching for {:?}", def);

        if self.sess.cstore.is_defaulted_trait(trait_id) {
            self.record_trait_has_default_impl(trait_id);
        }

        for impl_def_id in self.sess.cstore.implementations_of_trait(Some(trait_id)) {
            let trait_ref = self.impl_trait_ref(impl_def_id).unwrap();

            // Record the trait->implementation mapping.
            let parent = self.sess.cstore.impl_parent(impl_def_id).unwrap_or(trait_id);
            def.record_remote_impl(self, impl_def_id, trait_ref, parent);
        }

        def.flags.set(def.flags.get() | TraitFlags::IMPLS_VALID);
    }

    pub fn closure_kind(self, def_id: DefId) -> ty::ClosureKind {
        // If this is a local def-id, it should be inserted into the
        // tables by typeck; else, it will be retreived from
        // the external crate metadata.
        if let Some(&kind) = self.closure_kinds.borrow().get(&def_id) {
            return kind;
        }

        let kind = self.sess.cstore.closure_kind(def_id);
        self.closure_kinds.borrow_mut().insert(def_id, kind);
        kind
    }

    pub fn closure_type(self,
                        def_id: DefId,
                        substs: ClosureSubsts<'tcx>)
                        -> ty::ClosureTy<'tcx>
    {
        // If this is a local def-id, it should be inserted into the
        // tables by typeck; else, it will be retreived from
        // the external crate metadata.
        if let Some(ty) = self.closure_tys.borrow().get(&def_id) {
            return ty.subst(self, substs.substs);
        }

        let ty = self.sess.cstore.closure_ty(self.global_tcx(), def_id);
        self.closure_tys.borrow_mut().insert(def_id, ty.clone());
        ty.subst(self, substs.substs)
    }

    /// Given the def_id of an impl, return the def_id of the trait it implements.
    /// If it implements no trait, return `None`.
    pub fn trait_id_of_impl(self, def_id: DefId) -> Option<DefId> {
        self.impl_trait_ref(def_id).map(|tr| tr.def_id)
    }

    /// If the given def ID describes a method belonging to an impl, return the
    /// ID of the impl that the method belongs to. Otherwise, return `None`.
    pub fn impl_of_method(self, def_id: DefId) -> Option<DefId> {
        if def_id.krate != LOCAL_CRATE {
            return self.sess.cstore.associated_item(def_id).and_then(|item| {
                match item.container {
                    TraitContainer(_) => None,
                    ImplContainer(def_id) => Some(def_id),
                }
            });
        }
        match self.associated_items.borrow().get(&def_id).cloned() {
            Some(trait_item) => {
                match trait_item.container {
                    TraitContainer(_) => None,
                    ImplContainer(def_id) => Some(def_id),
                }
            }
            None => None
        }
    }

    /// If the given def ID describes an item belonging to a trait,
    /// return the ID of the trait that the trait item belongs to.
    /// Otherwise, return `None`.
    pub fn trait_of_item(self, def_id: DefId) -> Option<DefId> {
        if def_id.krate != LOCAL_CRATE {
            return self.sess.cstore.trait_of_item(def_id);
        }
        match self.associated_items.borrow().get(&def_id) {
            Some(associated_item) => {
                match associated_item.container {
                    TraitContainer(def_id) => Some(def_id),
                    ImplContainer(_) => None
                }
            }
            None => None
        }
    }

    /// Construct a parameter environment suitable for static contexts or other contexts where there
    /// are no free type/lifetime parameters in scope.
    pub fn empty_parameter_environment(self) -> ParameterEnvironment<'tcx> {

        // for an empty parameter environment, there ARE no free
        // regions, so it shouldn't matter what we use for the free id
        let free_id_outlive = self.region_maps.node_extent(ast::DUMMY_NODE_ID);
        ty::ParameterEnvironment {
            free_substs: self.intern_substs(&[]),
            caller_bounds: Vec::new(),
            implicit_region_bound: self.mk_region(ty::ReEmpty),
            free_id_outlive: free_id_outlive,
            is_copy_cache: RefCell::new(FxHashMap()),
            is_sized_cache: RefCell::new(FxHashMap()),
        }
    }

    /// Constructs and returns a substitution that can be applied to move from
    /// the "outer" view of a type or method to the "inner" view.
    /// In general, this means converting from bound parameters to
    /// free parameters. Since we currently represent bound/free type
    /// parameters in the same way, this only has an effect on regions.
    pub fn construct_free_substs(self, def_id: DefId,
                                 free_id_outlive: CodeExtent)
                                 -> &'gcx Substs<'gcx> {

        let substs = Substs::for_item(self.global_tcx(), def_id, |def, _| {
            // map bound 'a => free 'a
            self.global_tcx().mk_region(ReFree(FreeRegion {
                scope: free_id_outlive,
                bound_region: def.to_bound_region()
            }))
        }, |def, _| {
            // map T => T
            self.global_tcx().mk_param_from_def(def)
        });

        debug!("construct_parameter_environment: {:?}", substs);
        substs
    }

    /// See `ParameterEnvironment` struct def'n for details.
    /// If you were using `free_id: NodeId`, you might try `self.region_maps.item_extent(free_id)`
    /// for the `free_id_outlive` parameter. (But note that this is not always quite right.)
    pub fn construct_parameter_environment(self,
                                           span: Span,
                                           def_id: DefId,
                                           free_id_outlive: CodeExtent)
                                           -> ParameterEnvironment<'gcx>
    {
        //
        // Construct the free substs.
        //

        let free_substs = self.construct_free_substs(def_id, free_id_outlive);

        //
        // Compute the bounds on Self and the type parameters.
        //

        let tcx = self.global_tcx();
        let generic_predicates = tcx.item_predicates(def_id);
        let bounds = generic_predicates.instantiate(tcx, free_substs);
        let bounds = tcx.liberate_late_bound_regions(free_id_outlive, &ty::Binder(bounds));
        let predicates = bounds.predicates;

        // Finally, we have to normalize the bounds in the environment, in
        // case they contain any associated type projections. This process
        // can yield errors if the put in illegal associated types, like
        // `<i32 as Foo>::Bar` where `i32` does not implement `Foo`. We
        // report these errors right here; this doesn't actually feel
        // right to me, because constructing the environment feels like a
        // kind of a "idempotent" action, but I'm not sure where would be
        // a better place. In practice, we construct environments for
        // every fn once during type checking, and we'll abort if there
        // are any errors at that point, so after type checking you can be
        // sure that this will succeed without errors anyway.
        //

        let unnormalized_env = ty::ParameterEnvironment {
            free_substs: free_substs,
            implicit_region_bound: tcx.mk_region(ty::ReScope(free_id_outlive)),
            caller_bounds: predicates,
            free_id_outlive: free_id_outlive,
            is_copy_cache: RefCell::new(FxHashMap()),
            is_sized_cache: RefCell::new(FxHashMap()),
        };

        let cause = traits::ObligationCause::misc(span, free_id_outlive.node_id(&self.region_maps));
        traits::normalize_param_env_or_error(tcx, unnormalized_env, cause)
    }

    pub fn node_scope_region(self, id: NodeId) -> &'tcx Region {
        self.mk_region(ty::ReScope(self.region_maps.node_extent(id)))
    }

    pub fn visit_all_item_likes_in_krate<V,F>(self,
                                              dep_node_fn: F,
                                              visitor: &mut V)
        where F: FnMut(DefId) -> DepNode<DefId>, V: ItemLikeVisitor<'gcx>
    {
        dep_graph::visit_all_item_likes_in_krate(self.global_tcx(), dep_node_fn, visitor);
    }

    /// Looks up the span of `impl_did` if the impl is local; otherwise returns `Err`
    /// with the name of the crate containing the impl.
    pub fn span_of_impl(self, impl_did: DefId) -> Result<Span, Symbol> {
        if impl_did.is_local() {
            let node_id = self.map.as_local_node_id(impl_did).unwrap();
            Ok(self.map.span(node_id))
        } else {
            Err(self.sess.cstore.crate_name(impl_did.krate))
        }
    }
}

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    pub fn with_freevars<T, F>(self, fid: NodeId, f: F) -> T where
        F: FnOnce(&[hir::Freevar]) -> T,
    {
        match self.freevars.borrow().get(&fid) {
            None => f(&[]),
            Some(d) => f(&d[..])
        }
    }
}
