pub use self::fold::{TypeFoldable, TypeVisitor};
pub use self::AssocItemContainer::*;
pub use self::BorrowKind::*;
pub use self::IntVarValue::*;
pub use self::Variance::*;

use crate::arena::Arena;
use crate::hir::exports::ExportMap;
use crate::ich::StableHashingContext;
use crate::infer::canonical::Canonical;
use crate::middle::cstore::CrateStoreDyn;
use crate::middle::resolve_lifetime::ObjectLifetimeDefault;
use crate::mir::interpret::ErrorHandled;
use crate::mir::Body;
use crate::mir::GeneratorLayout;
use crate::traits::{self, Reveal};
use crate::ty;
use crate::ty::subst::{InternalSubsts, Subst, SubstsRef};
use crate::ty::util::{Discr, IntTypeExt};
use rustc_ast::ast::{self, Ident, Name};
use rustc_ast::node_id::{NodeId, NodeMap, NodeSet};
use rustc_attr as attr;
use rustc_data_structures::captures::Captures;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::sorted_map::SortedIndexMultiMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::{self, par_iter, ParallelIterator};
use rustc_errors::ErrorReported;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, CtorOf, DefKind, Namespace, Res};
use rustc_hir::def_id::{CrateNum, DefId, DefIdMap, LocalDefId, CRATE_DEF_INDEX};
use rustc_hir::lang_items::{FnMutTraitLangItem, FnOnceTraitLangItem, FnTraitLangItem};
use rustc_hir::{Constness, GlobMap, Node, TraitMap};
use rustc_index::vec::{Idx, IndexVec};
use rustc_macros::HashStable;
use rustc_serialize::{self, Encodable, Encoder};
use rustc_session::DataTypeKind;
use rustc_span::hygiene::ExpnId;
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_span::Span;
use rustc_target::abi::{Align, VariantIdx};

use std::cell::RefCell;
use std::cmp::{self, Ordering};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::ops::Range;
use std::slice;
use std::{mem, ptr};

pub use self::sty::BoundRegion::*;
pub use self::sty::InferTy::*;
pub use self::sty::RegionKind;
pub use self::sty::RegionKind::*;
pub use self::sty::TyKind::*;
pub use self::sty::{Binder, BoundTy, BoundTyKind, BoundVar, DebruijnIndex, INNERMOST};
pub use self::sty::{BoundRegion, EarlyBoundRegion, FreeRegion, Region};
pub use self::sty::{CanonicalPolyFnSig, FnSig, GenSig, PolyFnSig, PolyGenSig};
pub use self::sty::{ClosureSubsts, GeneratorSubsts, TypeAndMut, UpvarSubsts};
pub use self::sty::{Const, ConstKind, ExistentialProjection, PolyExistentialProjection};
pub use self::sty::{ConstVid, FloatVid, IntVid, RegionVid, TyVid};
pub use self::sty::{ExistentialPredicate, InferConst, InferTy, ParamConst, ParamTy, ProjectionTy};
pub use self::sty::{ExistentialTraitRef, PolyExistentialTraitRef};
pub use self::sty::{PolyTraitRef, TraitRef, TyKind};
pub use crate::ty::diagnostics::*;

pub use self::binding::BindingMode;
pub use self::binding::BindingMode::*;

pub use self::context::{tls, FreeRegionInfo, TyCtxt};
pub use self::context::{
    CanonicalUserType, CanonicalUserTypeAnnotation, CanonicalUserTypeAnnotations, ResolvedOpaqueTy,
    UserType, UserTypeAnnotationIndex,
};
pub use self::context::{
    CtxtInterners, GeneratorInteriorTypeCause, GlobalCtxt, Lift, TypeckTables,
};

pub use self::instance::{Instance, InstanceDef};

pub use self::trait_def::TraitDef;

pub use self::query::queries;

pub mod adjustment;
pub mod binding;
pub mod cast;
#[macro_use]
pub mod codec;
pub mod _match;
mod erase_regions;
pub mod error;
pub mod fast_reject;
pub mod flags;
pub mod fold;
pub mod inhabitedness;
pub mod layout;
pub mod normalize_erasing_regions;
pub mod outlives;
pub mod print;
pub mod query;
pub mod relate;
pub mod steal;
pub mod subst;
pub mod trait_def;
pub mod util;
pub mod walk;

mod context;
mod diagnostics;
mod instance;
mod structural_impls;
mod sty;

// Data types

pub struct ResolverOutputs {
    pub definitions: rustc_hir::definitions::Definitions,
    pub cstore: Box<CrateStoreDyn>,
    pub extern_crate_map: NodeMap<CrateNum>,
    pub trait_map: TraitMap<NodeId>,
    pub maybe_unused_trait_imports: NodeSet,
    pub maybe_unused_extern_crates: Vec<(NodeId, Span)>,
    pub export_map: ExportMap<NodeId>,
    pub glob_map: GlobMap,
    /// Extern prelude entries. The value is `true` if the entry was introduced
    /// via `extern crate` item and not `--extern` option or compiler built-in.
    pub extern_prelude: FxHashMap<Name, bool>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, HashStable)]
pub enum AssocItemContainer {
    TraitContainer(DefId),
    ImplContainer(DefId),
}

impl AssocItemContainer {
    /// Asserts that this is the `DefId` of an associated item declared
    /// in a trait, and returns the trait `DefId`.
    pub fn assert_trait(&self) -> DefId {
        match *self {
            TraitContainer(id) => id,
            _ => bug!("associated item has wrong container type: {:?}", self),
        }
    }

    pub fn id(&self) -> DefId {
        match *self {
            TraitContainer(id) => id,
            ImplContainer(id) => id,
        }
    }
}

/// The "header" of an impl is everything outside the body: a Self type, a trait
/// ref (in the case of a trait impl), and a set of predicates (from the
/// bounds / where-clauses).
#[derive(Clone, Debug, TypeFoldable)]
pub struct ImplHeader<'tcx> {
    pub impl_def_id: DefId,
    pub self_ty: Ty<'tcx>,
    pub trait_ref: Option<TraitRef<'tcx>>,
    pub predicates: Vec<Predicate<'tcx>>,
}

#[derive(Copy, Clone, PartialEq, RustcEncodable, RustcDecodable, HashStable)]
pub enum ImplPolarity {
    /// `impl Trait for Type`
    Positive,
    /// `impl !Trait for Type`
    Negative,
    /// `#[rustc_reservation_impl] impl Trait for Type`
    ///
    /// This is a "stability hack", not a real Rust feature.
    /// See #64631 for details.
    Reservation,
}

#[derive(Copy, Clone, Debug, PartialEq, HashStable)]
pub struct AssocItem {
    pub def_id: DefId,
    #[stable_hasher(project(name))]
    pub ident: Ident,
    pub kind: AssocKind,
    pub vis: Visibility,
    pub defaultness: hir::Defaultness,
    pub container: AssocItemContainer,

    /// Whether this is a method with an explicit self
    /// as its first parameter, allowing method calls.
    pub fn_has_self_parameter: bool,
}

#[derive(Copy, Clone, PartialEq, Debug, HashStable)]
pub enum AssocKind {
    Const,
    Fn,
    OpaqueTy,
    Type,
}

impl AssocKind {
    pub fn namespace(&self) -> Namespace {
        match *self {
            ty::AssocKind::OpaqueTy | ty::AssocKind::Type => Namespace::TypeNS,
            ty::AssocKind::Const | ty::AssocKind::Fn => Namespace::ValueNS,
        }
    }

    pub fn as_def_kind(&self) -> DefKind {
        match self {
            AssocKind::Const => DefKind::AssocConst,
            AssocKind::Fn => DefKind::AssocFn,
            AssocKind::Type => DefKind::AssocTy,
            AssocKind::OpaqueTy => DefKind::AssocOpaqueTy,
        }
    }
}

impl AssocItem {
    /// Tests whether the associated item admits a non-trivial implementation
    /// for !
    pub fn relevant_for_never(&self) -> bool {
        match self.kind {
            AssocKind::OpaqueTy | AssocKind::Const | AssocKind::Type => true,
            // FIXME(canndrew): Be more thorough here, check if any argument is uninhabited.
            AssocKind::Fn => !self.fn_has_self_parameter,
        }
    }

    pub fn signature(&self, tcx: TyCtxt<'_>) -> String {
        match self.kind {
            ty::AssocKind::Fn => {
                // We skip the binder here because the binder would deanonymize all
                // late-bound regions, and we don't want method signatures to show up
                // `as for<'r> fn(&'r MyType)`.  Pretty-printing handles late-bound
                // regions just fine, showing `fn(&MyType)`.
                tcx.fn_sig(self.def_id).skip_binder().to_string()
            }
            ty::AssocKind::Type => format!("type {};", self.ident),
            // FIXME(type_alias_impl_trait): we should print bounds here too.
            ty::AssocKind::OpaqueTy => format!("type {};", self.ident),
            ty::AssocKind::Const => {
                format!("const {}: {:?};", self.ident, tcx.type_of(self.def_id))
            }
        }
    }
}

/// A list of `ty::AssocItem`s in definition order that allows for efficient lookup by name.
///
/// When doing lookup by name, we try to postpone hygienic comparison for as long as possible since
/// it is relatively expensive. Instead, items are indexed by `Symbol` and hygienic comparison is
/// done only on items with the same name.
#[derive(Debug, Clone, PartialEq, HashStable)]
pub struct AssociatedItems {
    items: SortedIndexMultiMap<u32, Symbol, ty::AssocItem>,
}

impl AssociatedItems {
    /// Constructs an `AssociatedItems` map from a series of `ty::AssocItem`s in definition order.
    pub fn new(items_in_def_order: impl IntoIterator<Item = ty::AssocItem>) -> Self {
        let items = items_in_def_order.into_iter().map(|item| (item.ident.name, item)).collect();
        AssociatedItems { items }
    }

    /// Returns a slice of associated items in the order they were defined.
    ///
    /// New code should avoid relying on definition order. If you need a particular associated item
    /// for a known trait, make that trait a lang item instead of indexing this array.
    pub fn in_definition_order(&self) -> impl '_ + Iterator<Item = &ty::AssocItem> {
        self.items.iter().map(|(_, v)| v)
    }

    /// Returns an iterator over all associated items with the given name, ignoring hygiene.
    pub fn filter_by_name_unhygienic(
        &self,
        name: Symbol,
    ) -> impl '_ + Iterator<Item = &ty::AssocItem> {
        self.items.get_by_key(&name)
    }

    /// Returns an iterator over all associated items with the given name.
    ///
    /// Multiple items may have the same name if they are in different `Namespace`s. For example,
    /// an associated type can have the same name as a method. Use one of the `find_by_name_and_*`
    /// methods below if you know which item you are looking for.
    pub fn filter_by_name(
        &'a self,
        tcx: TyCtxt<'a>,
        ident: Ident,
        parent_def_id: DefId,
    ) -> impl 'a + Iterator<Item = &'a ty::AssocItem> {
        self.filter_by_name_unhygienic(ident.name)
            .filter(move |item| tcx.hygienic_eq(ident, item.ident, parent_def_id))
    }

    /// Returns the associated item with the given name and `AssocKind`, if one exists.
    pub fn find_by_name_and_kind(
        &self,
        tcx: TyCtxt<'_>,
        ident: Ident,
        kind: AssocKind,
        parent_def_id: DefId,
    ) -> Option<&ty::AssocItem> {
        self.filter_by_name_unhygienic(ident.name)
            .filter(|item| item.kind == kind)
            .find(|item| tcx.hygienic_eq(ident, item.ident, parent_def_id))
    }

    /// Returns the associated item with the given name in the given `Namespace`, if one exists.
    pub fn find_by_name_and_namespace(
        &self,
        tcx: TyCtxt<'_>,
        ident: Ident,
        ns: Namespace,
        parent_def_id: DefId,
    ) -> Option<&ty::AssocItem> {
        self.filter_by_name_unhygienic(ident.name)
            .filter(|item| item.kind.namespace() == ns)
            .find(|item| tcx.hygienic_eq(ident, item.ident, parent_def_id))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Copy, RustcEncodable, RustcDecodable, HashStable)]
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

impl<'tcx> DefIdTree for TyCtxt<'tcx> {
    fn parent(self, id: DefId) -> Option<DefId> {
        self.def_key(id).parent.map(|index| DefId { index, ..id })
    }
}

impl Visibility {
    pub fn from_hir(visibility: &hir::Visibility<'_>, id: hir::HirId, tcx: TyCtxt<'_>) -> Self {
        match visibility.node {
            hir::VisibilityKind::Public => Visibility::Public,
            hir::VisibilityKind::Crate(_) => Visibility::Restricted(DefId::local(CRATE_DEF_INDEX)),
            hir::VisibilityKind::Restricted { ref path, .. } => match path.res {
                // If there is no resolution, `resolve` will have already reported an error, so
                // assume that the visibility is public to avoid reporting more privacy errors.
                Res::Err => Visibility::Public,
                def => Visibility::Restricted(def.def_id()),
            },
            hir::VisibilityKind::Inherited => {
                Visibility::Restricted(tcx.parent_module(id).to_def_id())
            }
        }
    }

    /// Returns `true` if an item with this visibility is accessible from the given block.
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

    /// Returns `true` if this visibility is at least as accessible as the given visibility
    pub fn is_at_least<T: DefIdTree>(self, vis: Visibility, tree: T) -> bool {
        let vis_restriction = match vis {
            Visibility::Public => return self == Visibility::Public,
            Visibility::Invisible => return true,
            Visibility::Restricted(module) => module,
        };

        self.is_accessible_from(vis_restriction, tree)
    }

    // Returns `true` if this item is visible anywhere in the local crate.
    pub fn is_visible_locally(self) -> bool {
        match self {
            Visibility::Public => true,
            Visibility::Restricted(def_id) => def_id.is_local(),
            Visibility::Invisible => false,
        }
    }
}

#[derive(Copy, Clone, PartialEq, RustcDecodable, RustcEncodable, HashStable)]
pub enum Variance {
    Covariant,     // T<A> <: T<B> iff A <: B -- e.g., function return type
    Invariant,     // T<A> <: T<B> iff B == A -- e.g., type of mutable cell
    Contravariant, // T<A> <: T<B> iff B <: A -- e.g., function param type
    Bivariant,     // T<A> <: T<B>            -- e.g., unused type parameter
}

/// The crate variances map is computed during typeck and contains the
/// variance of every item in the local crate. You should not use it
/// directly, because to do so will make your pass dependent on the
/// HIR of every item in the local crate. Instead, use
/// `tcx.variances_of()` to get the variance for a *particular*
/// item.
#[derive(HashStable)]
pub struct CrateVariancesMap<'tcx> {
    /// For each item with generics, maps to a vector of the variance
    /// of its generics. If an item has no generics, it will have no
    /// entry.
    pub variances: FxHashMap<DefId, &'tcx [ty::Variance]>,
}

impl Variance {
    /// `a.xform(b)` combines the variance of a context with the
    /// variance of a type with the following meaning. If we are in a
    /// context with variance `a`, and we encounter a type argument in
    /// a position with variance `b`, then `a.xform(b)` is the new
    /// variance with which the argument appears.
    ///
    /// Example 1:
    ///
    ///     *mut Vec<i32>
    ///
    /// Here, the "ambient" variance starts as covariant. `*mut T` is
    /// invariant with respect to `T`, so the variance in which the
    /// `Vec<i32>` appears is `Covariant.xform(Invariant)`, which
    /// yields `Invariant`. Now, the type `Vec<T>` is covariant with
    /// respect to its type argument `T`, and hence the variance of
    /// the `i32` here is `Invariant.xform(Covariant)`, which results
    /// (again) in `Invariant`.
    ///
    /// Example 2:
    ///
    ///     fn(*const Vec<i32>, *mut Vec<i32)
    ///
    /// The ambient variance is covariant. A `fn` type is
    /// contravariant with respect to its parameters, so the variance
    /// within which both pointer types appear is
    /// `Covariant.xform(Contravariant)`, or `Contravariant`. `*const
    /// T` is covariant with respect to `T`, so the variance within
    /// which the first `Vec<i32>` appears is
    /// `Contravariant.xform(Covariant)` or `Contravariant`. The same
    /// is true for its `i32` argument. In the `*mut T` case, the
    /// variance of `Vec<i32>` is `Contravariant.xform(Invariant)`,
    /// and hence the outermost type is `Invariant` with respect to
    /// `Vec<i32>` (and its `i32` argument).
    ///
    /// Source: Figure 1 of "Taming the Wildcards:
    /// Combining Definition- and Use-Site Variance" published in PLDI'11.
    pub fn xform(self, v: ty::Variance) -> ty::Variance {
        match (self, v) {
            // Figure 1, column 1.
            (ty::Covariant, ty::Covariant) => ty::Covariant,
            (ty::Covariant, ty::Contravariant) => ty::Contravariant,
            (ty::Covariant, ty::Invariant) => ty::Invariant,
            (ty::Covariant, ty::Bivariant) => ty::Bivariant,

            // Figure 1, column 2.
            (ty::Contravariant, ty::Covariant) => ty::Contravariant,
            (ty::Contravariant, ty::Contravariant) => ty::Covariant,
            (ty::Contravariant, ty::Invariant) => ty::Invariant,
            (ty::Contravariant, ty::Bivariant) => ty::Bivariant,

            // Figure 1, column 3.
            (ty::Invariant, _) => ty::Invariant,

            // Figure 1, column 4.
            (ty::Bivariant, _) => ty::Bivariant,
        }
    }
}

// Contains information needed to resolve types and (in the future) look up
// the types of AST nodes.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct CReaderCacheKey {
    pub cnum: CrateNum,
    pub pos: usize,
}

bitflags! {
    /// Flags that we track on types. These flags are propagated upwards
    /// through the type during type construction, so that we can quickly check
    /// whether the type has various kinds of types in it without recursing
    /// over the type itself.
    pub struct TypeFlags: u32 {
        // Does this have parameters? Used to determine whether substitution is
        // required.
        /// Does this have [Param]?
        const HAS_TY_PARAM                = 1 << 0;
        /// Does this have [ReEarlyBound]?
        const HAS_RE_PARAM                = 1 << 1;
        /// Does this have [ConstKind::Param]?
        const HAS_CT_PARAM                = 1 << 2;

        const NEEDS_SUBST                 = TypeFlags::HAS_TY_PARAM.bits
                                          | TypeFlags::HAS_RE_PARAM.bits
                                          | TypeFlags::HAS_CT_PARAM.bits;

        /// Does this have [Infer]?
        const HAS_TY_INFER                = 1 << 3;
        /// Does this have [ReVar]?
        const HAS_RE_INFER                = 1 << 4;
        /// Does this have [ConstKind::Infer]?
        const HAS_CT_INFER                = 1 << 5;

        /// Does this have inference variables? Used to determine whether
        /// inference is required.
        const NEEDS_INFER                 = TypeFlags::HAS_TY_INFER.bits
                                          | TypeFlags::HAS_RE_INFER.bits
                                          | TypeFlags::HAS_CT_INFER.bits;

        /// Does this have [Placeholder]?
        const HAS_TY_PLACEHOLDER          = 1 << 6;
        /// Does this have [RePlaceholder]?
        const HAS_RE_PLACEHOLDER          = 1 << 7;
        /// Does this have [ConstKind::Placeholder]?
        const HAS_CT_PLACEHOLDER          = 1 << 8;

        /// `true` if there are "names" of regions and so forth
        /// that are local to a particular fn/inferctxt
        const HAS_FREE_LOCAL_REGIONS      = 1 << 9;

        /// `true` if there are "names" of types and regions and so forth
        /// that are local to a particular fn
        const HAS_FREE_LOCAL_NAMES        = TypeFlags::HAS_TY_PARAM.bits
                                          | TypeFlags::HAS_CT_PARAM.bits
                                          | TypeFlags::HAS_TY_INFER.bits
                                          | TypeFlags::HAS_CT_INFER.bits
                                          | TypeFlags::HAS_TY_PLACEHOLDER.bits
                                          | TypeFlags::HAS_CT_PLACEHOLDER.bits
                                          | TypeFlags::HAS_FREE_LOCAL_REGIONS.bits;

        /// Does this have [Projection] or [UnnormalizedProjection]?
        const HAS_TY_PROJECTION           = 1 << 10;
        /// Does this have [Opaque]?
        const HAS_TY_OPAQUE               = 1 << 11;
        /// Does this have [ConstKind::Unevaluated]?
        const HAS_CT_PROJECTION           = 1 << 12;

        /// Could this type be normalized further?
        const HAS_PROJECTION              = TypeFlags::HAS_TY_PROJECTION.bits
                                          | TypeFlags::HAS_TY_OPAQUE.bits
                                          | TypeFlags::HAS_CT_PROJECTION.bits;

        /// Is an error type/const reachable?
        const HAS_ERROR                   = 1 << 13;

        /// Does this have any region that "appears free" in the type?
        /// Basically anything but [ReLateBound] and [ReErased].
        const HAS_FREE_REGIONS            = 1 << 14;

        /// Does this have any [ReLateBound] regions? Used to check
        /// if a global bound is safe to evaluate.
        const HAS_RE_LATE_BOUND           = 1 << 15;

        /// Does this have any [ReErased] regions?
        const HAS_RE_ERASED               = 1 << 16;

        /// Does this value have parameters/placeholders/inference variables which could be
        /// replaced later, in a way that would change the results of `impl` specialization?
        const STILL_FURTHER_SPECIALIZABLE = 1 << 17;
    }
}

#[allow(rustc::usage_of_ty_tykind)]
pub struct TyS<'tcx> {
    pub kind: TyKind<'tcx>,
    pub flags: TypeFlags,

    /// This is a kind of confusing thing: it stores the smallest
    /// binder such that
    ///
    /// (a) the binder itself captures nothing but
    /// (b) all the late-bound things within the type are captured
    ///     by some sub-binder.
    ///
    /// So, for a type without any late-bound things, like `u32`, this
    /// will be *innermost*, because that is the innermost binder that
    /// captures nothing. But for a type `&'D u32`, where `'D` is a
    /// late-bound region with De Bruijn index `D`, this would be `D + 1`
    /// -- the binder itself does not capture `D`, but `D` is captured
    /// by an inner binder.
    ///
    /// We call this concept an "exclusive" binder `D` because all
    /// De Bruijn indices within the type are contained within `0..D`
    /// (exclusive).
    outer_exclusive_binder: ty::DebruijnIndex,
}

// `TyS` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(target_arch = "x86_64")]
static_assert_size!(TyS<'_>, 32);

impl<'tcx> Ord for TyS<'tcx> {
    fn cmp(&self, other: &TyS<'tcx>) -> Ordering {
        self.kind.cmp(&other.kind)
    }
}

impl<'tcx> PartialOrd for TyS<'tcx> {
    fn partial_cmp(&self, other: &TyS<'tcx>) -> Option<Ordering> {
        Some(self.kind.cmp(&other.kind))
    }
}

impl<'tcx> PartialEq for TyS<'tcx> {
    #[inline]
    fn eq(&self, other: &TyS<'tcx>) -> bool {
        ptr::eq(self, other)
    }
}
impl<'tcx> Eq for TyS<'tcx> {}

impl<'tcx> Hash for TyS<'tcx> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        (self as *const TyS<'_>).hash(s)
    }
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for ty::TyS<'tcx> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        let ty::TyS {
            ref kind,

            // The other fields just provide fast access to information that is
            // also contained in `kind`, so no need to hash them.
            flags: _,

            outer_exclusive_binder: _,
        } = *self;

        kind.hash_stable(hcx, hasher);
    }
}

#[rustc_diagnostic_item = "Ty"]
pub type Ty<'tcx> = &'tcx TyS<'tcx>;

impl<'tcx> rustc_serialize::UseSpecializedEncodable for Ty<'tcx> {}
impl<'tcx> rustc_serialize::UseSpecializedDecodable for Ty<'tcx> {}

pub type CanonicalTy<'tcx> = Canonical<'tcx, Ty<'tcx>>;

extern "C" {
    /// A dummy type used to force `List` to be unsized while not requiring references to it be wide
    /// pointers.
    type OpaqueListContents;
}

/// A wrapper for slices with the additional invariant
/// that the slice is interned and no other slice with
/// the same contents can exist in the same context.
/// This means we can use pointer for both
/// equality comparisons and hashing.
/// Note: `Slice` was already taken by the `Ty`.
#[repr(C)]
pub struct List<T> {
    len: usize,
    data: [T; 0],
    opaque: OpaqueListContents,
}

unsafe impl<T: Sync> Sync for List<T> {}

impl<T: Copy> List<T> {
    #[inline]
    fn from_arena<'tcx>(arena: &'tcx Arena<'tcx>, slice: &[T]) -> &'tcx List<T> {
        assert!(!mem::needs_drop::<T>());
        assert!(mem::size_of::<T>() != 0);
        assert!(!slice.is_empty());

        // Align up the size of the len (usize) field
        let align = mem::align_of::<T>();
        let align_mask = align - 1;
        let offset = mem::size_of::<usize>();
        let offset = (offset + align_mask) & !align_mask;

        let size = offset + slice.len() * mem::size_of::<T>();

        let mem = arena
            .dropless
            .alloc_raw(size, cmp::max(mem::align_of::<T>(), mem::align_of::<usize>()));
        unsafe {
            let result = &mut *(mem.as_mut_ptr() as *mut List<T>);
            // Write the length
            result.len = slice.len();

            // Write the elements
            let arena_slice = slice::from_raw_parts_mut(result.data.as_mut_ptr(), result.len);
            arena_slice.copy_from_slice(slice);

            result
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for List<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl<T: Encodable> Encodable for List<T> {
    #[inline]
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        (**self).encode(s)
    }
}

impl<T> Ord for List<T>
where
    T: Ord,
{
    fn cmp(&self, other: &List<T>) -> Ordering {
        if self == other { Ordering::Equal } else { <[T] as Ord>::cmp(&**self, &**other) }
    }
}

impl<T> PartialOrd for List<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &List<T>) -> Option<Ordering> {
        if self == other {
            Some(Ordering::Equal)
        } else {
            <[T] as PartialOrd>::partial_cmp(&**self, &**other)
        }
    }
}

impl<T: PartialEq> PartialEq for List<T> {
    #[inline]
    fn eq(&self, other: &List<T>) -> bool {
        ptr::eq(self, other)
    }
}
impl<T: Eq> Eq for List<T> {}

impl<T> Hash for List<T> {
    #[inline]
    fn hash<H: Hasher>(&self, s: &mut H) {
        (self as *const List<T>).hash(s)
    }
}

impl<T> Deref for List<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        self.as_ref()
    }
}

impl<T> AsRef<[T]> for List<T> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }
}

impl<'a, T> IntoIterator for &'a List<T> {
    type Item = &'a T;
    type IntoIter = <&'a [T] as IntoIterator>::IntoIter;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self[..].iter()
    }
}

impl<'tcx> rustc_serialize::UseSpecializedDecodable for &'tcx List<Ty<'tcx>> {}

impl<T> List<T> {
    #[inline(always)]
    pub fn empty<'a>() -> &'a List<T> {
        #[repr(align(64), C)]
        struct EmptySlice([u8; 64]);
        static EMPTY_SLICE: EmptySlice = EmptySlice([0; 64]);
        assert!(mem::align_of::<T>() <= 64);
        unsafe { &*(&EMPTY_SLICE as *const _ as *const List<T>) }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, HashStable)]
pub struct UpvarPath {
    pub hir_id: hir::HirId,
}

/// Upvars do not get their own `NodeId`. Instead, we use the pair of
/// the original var ID (that is, the root variable that is referenced
/// by the upvar) and the ID of the closure expression.
#[derive(Clone, Copy, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, HashStable)]
pub struct UpvarId {
    pub var_path: UpvarPath,
    pub closure_expr_id: LocalDefId,
}

#[derive(Clone, PartialEq, Debug, RustcEncodable, RustcDecodable, Copy, HashStable)]
pub enum BorrowKind {
    /// Data must be immutable and is aliasable.
    ImmBorrow,

    /// Data must be immutable but not aliasable. This kind of borrow
    /// cannot currently be expressed by the user and is used only in
    /// implicit closure bindings. It is needed when the closure
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
    MutBorrow,
}

/// Information describing the capture of an upvar. This is computed
/// during `typeck`, specifically by `regionck`.
#[derive(PartialEq, Clone, Debug, Copy, RustcEncodable, RustcDecodable, HashStable)]
pub enum UpvarCapture<'tcx> {
    /// Upvar is captured by value. This is always true when the
    /// closure is labeled `move`, but can also be true in other cases
    /// depending on inference.
    ByValue,

    /// Upvar is captured by reference.
    ByRef(UpvarBorrow<'tcx>),
}

#[derive(PartialEq, Clone, Copy, RustcEncodable, RustcDecodable, HashStable)]
pub struct UpvarBorrow<'tcx> {
    /// The kind of borrow: by-ref upvars have access to shared
    /// immutable borrows, which are not part of the normal language
    /// syntax.
    pub kind: BorrowKind,

    /// Region of the resulting reference.
    pub region: ty::Region<'tcx>,
}

pub type UpvarListMap = FxHashMap<DefId, FxIndexMap<hir::HirId, UpvarId>>;
pub type UpvarCaptureMap<'tcx> = FxHashMap<UpvarId, UpvarCapture<'tcx>>;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum IntVarValue {
    IntType(ast::IntTy),
    UintType(ast::UintTy),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct FloatVarValue(pub ast::FloatTy);

impl ty::EarlyBoundRegion {
    pub fn to_bound_region(&self) -> ty::BoundRegion {
        ty::BoundRegion::BrNamed(self.def_id, self.name)
    }

    /// Does this early bound region have a name? Early bound regions normally
    /// always have names except when using anonymous lifetimes (`'_`).
    pub fn has_name(&self) -> bool {
        self.name != kw::UnderscoreLifetime
    }
}

#[derive(Clone, Debug, RustcEncodable, RustcDecodable, HashStable)]
pub enum GenericParamDefKind {
    Lifetime,
    Type {
        has_default: bool,
        object_lifetime_default: ObjectLifetimeDefault,
        synthetic: Option<hir::SyntheticTyParamKind>,
    },
    Const,
}

impl GenericParamDefKind {
    pub fn descr(&self) -> &'static str {
        match self {
            GenericParamDefKind::Lifetime => "lifetime",
            GenericParamDefKind::Type { .. } => "type",
            GenericParamDefKind::Const => "constant",
        }
    }
}

#[derive(Clone, Debug, RustcEncodable, RustcDecodable, HashStable)]
pub struct GenericParamDef {
    pub name: Symbol,
    pub def_id: DefId,
    pub index: u32,

    /// `pure_wrt_drop`, set by the (unsafe) `#[may_dangle]` attribute
    /// on generic parameter `'a`/`T`, asserts data behind the parameter
    /// `'a`/`T` won't be accessed during the parent type's `Drop` impl.
    pub pure_wrt_drop: bool,

    pub kind: GenericParamDefKind,
}

impl GenericParamDef {
    pub fn to_early_bound_region_data(&self) -> ty::EarlyBoundRegion {
        if let GenericParamDefKind::Lifetime = self.kind {
            ty::EarlyBoundRegion { def_id: self.def_id, index: self.index, name: self.name }
        } else {
            bug!("cannot convert a non-lifetime parameter def to an early bound region")
        }
    }

    pub fn to_bound_region(&self) -> ty::BoundRegion {
        if let GenericParamDefKind::Lifetime = self.kind {
            self.to_early_bound_region_data().to_bound_region()
        } else {
            bug!("cannot convert a non-lifetime parameter def to an early bound region")
        }
    }
}

#[derive(Default)]
pub struct GenericParamCount {
    pub lifetimes: usize,
    pub types: usize,
    pub consts: usize,
}

/// Information about the formal type/lifetime parameters associated
/// with an item or method. Analogous to `hir::Generics`.
///
/// The ordering of parameters is the same as in `Subst` (excluding child generics):
/// `Self` (optionally), `Lifetime` params..., `Type` params...
#[derive(Clone, Debug, RustcEncodable, RustcDecodable, HashStable)]
pub struct Generics {
    pub parent: Option<DefId>,
    pub parent_count: usize,
    pub params: Vec<GenericParamDef>,

    /// Reverse map to the `index` field of each `GenericParamDef`.
    #[stable_hasher(ignore)]
    pub param_def_id_to_index: FxHashMap<DefId, u32>,

    pub has_self: bool,
    pub has_late_bound_regions: Option<Span>,
}

impl<'tcx> Generics {
    pub fn count(&self) -> usize {
        self.parent_count + self.params.len()
    }

    pub fn own_counts(&self) -> GenericParamCount {
        // We could cache this as a property of `GenericParamCount`, but
        // the aim is to refactor this away entirely eventually and the
        // presence of this method will be a constant reminder.
        let mut own_counts: GenericParamCount = Default::default();

        for param in &self.params {
            match param.kind {
                GenericParamDefKind::Lifetime => own_counts.lifetimes += 1,
                GenericParamDefKind::Type { .. } => own_counts.types += 1,
                GenericParamDefKind::Const => own_counts.consts += 1,
            };
        }

        own_counts
    }

    pub fn requires_monomorphization(&self, tcx: TyCtxt<'tcx>) -> bool {
        if self.own_requires_monomorphization() {
            return true;
        }

        if let Some(parent_def_id) = self.parent {
            let parent = tcx.generics_of(parent_def_id);
            parent.requires_monomorphization(tcx)
        } else {
            false
        }
    }

    pub fn own_requires_monomorphization(&self) -> bool {
        for param in &self.params {
            match param.kind {
                GenericParamDefKind::Type { .. } | GenericParamDefKind::Const => return true,
                GenericParamDefKind::Lifetime => {}
            }
        }
        false
    }

    pub fn param_at(&'tcx self, param_index: usize, tcx: TyCtxt<'tcx>) -> &'tcx GenericParamDef {
        if let Some(index) = param_index.checked_sub(self.parent_count) {
            &self.params[index]
        } else {
            tcx.generics_of(self.parent.expect("parent_count > 0 but no parent?"))
                .param_at(param_index, tcx)
        }
    }

    pub fn region_param(
        &'tcx self,
        param: &EarlyBoundRegion,
        tcx: TyCtxt<'tcx>,
    ) -> &'tcx GenericParamDef {
        let param = self.param_at(param.index as usize, tcx);
        match param.kind {
            GenericParamDefKind::Lifetime => param,
            _ => bug!("expected lifetime parameter, but found another generic parameter"),
        }
    }

    /// Returns the `GenericParamDef` associated with this `ParamTy`.
    pub fn type_param(&'tcx self, param: &ParamTy, tcx: TyCtxt<'tcx>) -> &'tcx GenericParamDef {
        let param = self.param_at(param.index as usize, tcx);
        match param.kind {
            GenericParamDefKind::Type { .. } => param,
            _ => bug!("expected type parameter, but found another generic parameter"),
        }
    }

    /// Returns the `ConstParameterDef` associated with this `ParamConst`.
    pub fn const_param(&'tcx self, param: &ParamConst, tcx: TyCtxt<'tcx>) -> &GenericParamDef {
        let param = self.param_at(param.index as usize, tcx);
        match param.kind {
            GenericParamDefKind::Const => param,
            _ => bug!("expected const parameter, but found another generic parameter"),
        }
    }
}

/// Bounds on generics.
#[derive(Copy, Clone, Default, Debug, RustcEncodable, RustcDecodable, HashStable)]
pub struct GenericPredicates<'tcx> {
    pub parent: Option<DefId>,
    pub predicates: &'tcx [(Predicate<'tcx>, Span)],
}

impl<'tcx> GenericPredicates<'tcx> {
    pub fn instantiate(
        &self,
        tcx: TyCtxt<'tcx>,
        substs: SubstsRef<'tcx>,
    ) -> InstantiatedPredicates<'tcx> {
        let mut instantiated = InstantiatedPredicates::empty();
        self.instantiate_into(tcx, &mut instantiated, substs);
        instantiated
    }

    pub fn instantiate_own(
        &self,
        tcx: TyCtxt<'tcx>,
        substs: SubstsRef<'tcx>,
    ) -> InstantiatedPredicates<'tcx> {
        InstantiatedPredicates {
            predicates: self.predicates.iter().map(|(p, _)| p.subst(tcx, substs)).collect(),
            spans: self.predicates.iter().map(|(_, sp)| *sp).collect(),
        }
    }

    fn instantiate_into(
        &self,
        tcx: TyCtxt<'tcx>,
        instantiated: &mut InstantiatedPredicates<'tcx>,
        substs: SubstsRef<'tcx>,
    ) {
        if let Some(def_id) = self.parent {
            tcx.predicates_of(def_id).instantiate_into(tcx, instantiated, substs);
        }
        instantiated.predicates.extend(self.predicates.iter().map(|(p, _)| p.subst(tcx, substs)));
        instantiated.spans.extend(self.predicates.iter().map(|(_, sp)| *sp));
    }

    pub fn instantiate_identity(&self, tcx: TyCtxt<'tcx>) -> InstantiatedPredicates<'tcx> {
        let mut instantiated = InstantiatedPredicates::empty();
        self.instantiate_identity_into(tcx, &mut instantiated);
        instantiated
    }

    fn instantiate_identity_into(
        &self,
        tcx: TyCtxt<'tcx>,
        instantiated: &mut InstantiatedPredicates<'tcx>,
    ) {
        if let Some(def_id) = self.parent {
            tcx.predicates_of(def_id).instantiate_identity_into(tcx, instantiated);
        }
        instantiated.predicates.extend(self.predicates.iter().map(|(p, _)| p));
        instantiated.spans.extend(self.predicates.iter().map(|(_, s)| s));
    }

    pub fn instantiate_supertrait(
        &self,
        tcx: TyCtxt<'tcx>,
        poly_trait_ref: &ty::PolyTraitRef<'tcx>,
    ) -> InstantiatedPredicates<'tcx> {
        assert_eq!(self.parent, None);
        InstantiatedPredicates {
            predicates: self
                .predicates
                .iter()
                .map(|(pred, _)| pred.subst_supertrait(tcx, poly_trait_ref))
                .collect(),
            spans: self.predicates.iter().map(|(_, sp)| *sp).collect(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
#[derive(HashStable, TypeFoldable)]
pub enum Predicate<'tcx> {
    /// Corresponds to `where Foo: Bar<A, B, C>`. `Foo` here would be
    /// the `Self` type of the trait reference and `A`, `B`, and `C`
    /// would be the type parameters.
    ///
    /// A trait predicate will have `Constness::Const` if it originates
    /// from a bound on a `const fn` without the `?const` opt-out (e.g.,
    /// `const fn foobar<Foo: Bar>() {}`).
    Trait(PolyTraitPredicate<'tcx>, Constness),

    /// `where 'a: 'b`
    RegionOutlives(PolyRegionOutlivesPredicate<'tcx>),

    /// `where T: 'a`
    TypeOutlives(PolyTypeOutlivesPredicate<'tcx>),

    /// `where <T as TraitRef>::Name == X`, approximately.
    /// See the `ProjectionPredicate` struct for details.
    Projection(PolyProjectionPredicate<'tcx>),

    /// No syntax: `T` well-formed.
    WellFormed(Ty<'tcx>),

    /// Trait must be object-safe.
    ObjectSafe(DefId),

    /// No direct syntax. May be thought of as `where T: FnFoo<...>`
    /// for some substitutions `...` and `T` being a closure type.
    /// Satisfied (or refuted) once we know the closure's kind.
    ClosureKind(DefId, SubstsRef<'tcx>, ClosureKind),

    /// `T1 <: T2`
    Subtype(PolySubtypePredicate<'tcx>),

    /// Constant initializer must evaluate successfully.
    ConstEvaluatable(DefId, SubstsRef<'tcx>),
}

/// The crate outlives map is computed during typeck and contains the
/// outlives of every item in the local crate. You should not use it
/// directly, because to do so will make your pass dependent on the
/// HIR of every item in the local crate. Instead, use
/// `tcx.inferred_outlives_of()` to get the outlives for a *particular*
/// item.
#[derive(HashStable)]
pub struct CratePredicatesMap<'tcx> {
    /// For each struct with outlive bounds, maps to a vector of the
    /// predicate of its outlive bounds. If an item has no outlives
    /// bounds, it will have no entry.
    pub predicates: FxHashMap<DefId, &'tcx [(ty::Predicate<'tcx>, Span)]>,
}

impl<'tcx> AsRef<Predicate<'tcx>> for Predicate<'tcx> {
    fn as_ref(&self) -> &Predicate<'tcx> {
        self
    }
}

impl<'tcx> Predicate<'tcx> {
    /// Performs a substitution suitable for going from a
    /// poly-trait-ref to supertraits that must hold if that
    /// poly-trait-ref holds. This is slightly different from a normal
    /// substitution in terms of what happens with bound regions. See
    /// lengthy comment below for details.
    pub fn subst_supertrait(
        &self,
        tcx: TyCtxt<'tcx>,
        trait_ref: &ty::PolyTraitRef<'tcx>,
    ) -> ty::Predicate<'tcx> {
        // The interaction between HRTB and supertraits is not entirely
        // obvious. Let me walk you (and myself) through an example.
        //
        // Let's start with an easy case. Consider two traits:
        //
        //     trait Foo<'a>: Bar<'a,'a> { }
        //     trait Bar<'b,'c> { }
        //
        // Now, if we have a trait reference `for<'x> T: Foo<'x>`, then
        // we can deduce that `for<'x> T: Bar<'x,'x>`. Basically, if we
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
        //     trait Foo1<'a>: for<'b> Bar1<'a,'b> { }
        //     trait Bar1<'b,'c> { }
        //
        // Here, if we have `for<'x> T: Foo1<'x>`, then what do we know?
        // The answer is that we know `for<'x,'b> T: Bar1<'x,'b>`. The
        // reason is similar to the previous example: any impl of
        // `T:Foo1<'x>` must show that `for<'b> T: Bar1<'x, 'b>`.  So
        // basically we would want to collapse the bound lifetimes from
        // the input (`trait_ref`) and the supertraits.
        //
        // To achieve this in practice is fairly straightforward. Let's
        // consider the more complicated scenario:
        //
        // - We start out with `for<'x> T: Foo1<'x>`. In this case, `'x`
        //   has a De Bruijn index of 1. We want to produce `for<'x,'b> T: Bar1<'x,'b>`,
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

        let substs = &trait_ref.skip_binder().substs;
        match *self {
            Predicate::Trait(ref binder, constness) => {
                Predicate::Trait(binder.map_bound(|data| data.subst(tcx, substs)), constness)
            }
            Predicate::Subtype(ref binder) => {
                Predicate::Subtype(binder.map_bound(|data| data.subst(tcx, substs)))
            }
            Predicate::RegionOutlives(ref binder) => {
                Predicate::RegionOutlives(binder.map_bound(|data| data.subst(tcx, substs)))
            }
            Predicate::TypeOutlives(ref binder) => {
                Predicate::TypeOutlives(binder.map_bound(|data| data.subst(tcx, substs)))
            }
            Predicate::Projection(ref binder) => {
                Predicate::Projection(binder.map_bound(|data| data.subst(tcx, substs)))
            }
            Predicate::WellFormed(data) => Predicate::WellFormed(data.subst(tcx, substs)),
            Predicate::ObjectSafe(trait_def_id) => Predicate::ObjectSafe(trait_def_id),
            Predicate::ClosureKind(closure_def_id, closure_substs, kind) => {
                Predicate::ClosureKind(closure_def_id, closure_substs.subst(tcx, substs), kind)
            }
            Predicate::ConstEvaluatable(def_id, const_substs) => {
                Predicate::ConstEvaluatable(def_id, const_substs.subst(tcx, substs))
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
#[derive(HashStable, TypeFoldable)]
pub struct TraitPredicate<'tcx> {
    pub trait_ref: TraitRef<'tcx>,
}

pub type PolyTraitPredicate<'tcx> = ty::Binder<TraitPredicate<'tcx>>;

impl<'tcx> TraitPredicate<'tcx> {
    pub fn def_id(&self) -> DefId {
        self.trait_ref.def_id
    }

    pub fn self_ty(&self) -> Ty<'tcx> {
        self.trait_ref.self_ty()
    }
}

impl<'tcx> PolyTraitPredicate<'tcx> {
    pub fn def_id(&self) -> DefId {
        // Ok to skip binder since trait `DefId` does not care about regions.
        self.skip_binder().def_id()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, RustcEncodable, RustcDecodable)]
#[derive(HashStable, TypeFoldable)]
pub struct OutlivesPredicate<A, B>(pub A, pub B); // `A: B`
pub type PolyOutlivesPredicate<A, B> = ty::Binder<OutlivesPredicate<A, B>>;
pub type RegionOutlivesPredicate<'tcx> = OutlivesPredicate<ty::Region<'tcx>, ty::Region<'tcx>>;
pub type TypeOutlivesPredicate<'tcx> = OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>>;
pub type PolyRegionOutlivesPredicate<'tcx> = ty::Binder<RegionOutlivesPredicate<'tcx>>;
pub type PolyTypeOutlivesPredicate<'tcx> = ty::Binder<TypeOutlivesPredicate<'tcx>>;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
#[derive(HashStable, TypeFoldable)]
pub struct SubtypePredicate<'tcx> {
    pub a_is_expected: bool,
    pub a: Ty<'tcx>,
    pub b: Ty<'tcx>,
}
pub type PolySubtypePredicate<'tcx> = ty::Binder<SubtypePredicate<'tcx>>;

/// This kind of predicate has no *direct* correspondent in the
/// syntax, but it roughly corresponds to the syntactic forms:
///
/// 1. `T: TraitRef<..., Item = Type>`
/// 2. `<T as TraitRef<...>>::Item == Type` (NYI)
///
/// In particular, form #1 is "desugared" to the combination of a
/// normal trait predicate (`T: TraitRef<...>`) and one of these
/// predicates. Form #2 is a broader form in that it also permits
/// equality between arbitrary types. Processing an instance of
/// Form #2 eventually yields one of these `ProjectionPredicate`
/// instances to normalize the LHS.
#[derive(Copy, Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
#[derive(HashStable, TypeFoldable)]
pub struct ProjectionPredicate<'tcx> {
    pub projection_ty: ProjectionTy<'tcx>,
    pub ty: Ty<'tcx>,
}

pub type PolyProjectionPredicate<'tcx> = Binder<ProjectionPredicate<'tcx>>;

impl<'tcx> PolyProjectionPredicate<'tcx> {
    /// Returns the `DefId` of the associated item being projected.
    pub fn item_def_id(&self) -> DefId {
        self.skip_binder().projection_ty.item_def_id
    }

    #[inline]
    pub fn to_poly_trait_ref(&self, tcx: TyCtxt<'tcx>) -> PolyTraitRef<'tcx> {
        // Note: unlike with `TraitRef::to_poly_trait_ref()`,
        // `self.0.trait_ref` is permitted to have escaping regions.
        // This is because here `self` has a `Binder` and so does our
        // return value, so we are preserving the number of binding
        // levels.
        self.map_bound(|predicate| predicate.projection_ty.trait_ref(tcx))
    }

    pub fn ty(&self) -> Binder<Ty<'tcx>> {
        self.map_bound(|predicate| predicate.ty)
    }

    /// The `DefId` of the `TraitItem` for the associated type.
    ///
    /// Note that this is not the `DefId` of the `TraitRef` containing this
    /// associated type, which is in `tcx.associated_item(projection_def_id()).container`.
    pub fn projection_def_id(&self) -> DefId {
        // Ok to skip binder since trait `DefId` does not care about regions.
        self.skip_binder().projection_ty.item_def_id
    }
}

pub trait ToPolyTraitRef<'tcx> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx>;
}

impl<'tcx> ToPolyTraitRef<'tcx> for TraitRef<'tcx> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx> {
        ty::Binder::dummy(*self)
    }
}

impl<'tcx> ToPolyTraitRef<'tcx> for PolyTraitPredicate<'tcx> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx> {
        self.map_bound_ref(|trait_pred| trait_pred.trait_ref)
    }
}

pub trait ToPredicate<'tcx> {
    fn to_predicate(&self) -> Predicate<'tcx>;
}

impl<'tcx> ToPredicate<'tcx> for ConstnessAnd<TraitRef<'tcx>> {
    fn to_predicate(&self) -> Predicate<'tcx> {
        ty::Predicate::Trait(
            ty::Binder::dummy(ty::TraitPredicate { trait_ref: self.value }),
            self.constness,
        )
    }
}

impl<'tcx> ToPredicate<'tcx> for ConstnessAnd<&TraitRef<'tcx>> {
    fn to_predicate(&self) -> Predicate<'tcx> {
        ty::Predicate::Trait(
            ty::Binder::dummy(ty::TraitPredicate { trait_ref: *self.value }),
            self.constness,
        )
    }
}

impl<'tcx> ToPredicate<'tcx> for ConstnessAnd<PolyTraitRef<'tcx>> {
    fn to_predicate(&self) -> Predicate<'tcx> {
        ty::Predicate::Trait(self.value.to_poly_trait_predicate(), self.constness)
    }
}

impl<'tcx> ToPredicate<'tcx> for ConstnessAnd<&PolyTraitRef<'tcx>> {
    fn to_predicate(&self) -> Predicate<'tcx> {
        ty::Predicate::Trait(self.value.to_poly_trait_predicate(), self.constness)
    }
}

impl<'tcx> ToPredicate<'tcx> for PolyRegionOutlivesPredicate<'tcx> {
    fn to_predicate(&self) -> Predicate<'tcx> {
        Predicate::RegionOutlives(*self)
    }
}

impl<'tcx> ToPredicate<'tcx> for PolyTypeOutlivesPredicate<'tcx> {
    fn to_predicate(&self) -> Predicate<'tcx> {
        Predicate::TypeOutlives(*self)
    }
}

impl<'tcx> ToPredicate<'tcx> for PolyProjectionPredicate<'tcx> {
    fn to_predicate(&self) -> Predicate<'tcx> {
        Predicate::Projection(*self)
    }
}

impl<'tcx> Predicate<'tcx> {
    pub fn to_opt_poly_trait_ref(&self) -> Option<PolyTraitRef<'tcx>> {
        match *self {
            Predicate::Trait(ref t, _) => Some(t.to_poly_trait_ref()),
            Predicate::Projection(..)
            | Predicate::Subtype(..)
            | Predicate::RegionOutlives(..)
            | Predicate::WellFormed(..)
            | Predicate::ObjectSafe(..)
            | Predicate::ClosureKind(..)
            | Predicate::TypeOutlives(..)
            | Predicate::ConstEvaluatable(..) => None,
        }
    }

    pub fn to_opt_type_outlives(&self) -> Option<PolyTypeOutlivesPredicate<'tcx>> {
        match *self {
            Predicate::TypeOutlives(data) => Some(data),
            Predicate::Trait(..)
            | Predicate::Projection(..)
            | Predicate::Subtype(..)
            | Predicate::RegionOutlives(..)
            | Predicate::WellFormed(..)
            | Predicate::ObjectSafe(..)
            | Predicate::ClosureKind(..)
            | Predicate::ConstEvaluatable(..) => None,
        }
    }
}

/// Represents the bounds declared on a particular set of type
/// parameters. Should eventually be generalized into a flag list of
/// where-clauses. You can obtain a `InstantiatedPredicates` list from a
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
///     struct Foo<T, U: Bar<T>> { ... }
///
/// Here, the `GenericPredicates` for `Foo` would contain a list of bounds like
/// `[[], [U:Bar<T>]]`. Now if there were some particular reference
/// like `Foo<isize,usize>`, then the `InstantiatedPredicates` would be `[[],
/// [usize:Bar<isize>]]`.
#[derive(Clone, Debug, TypeFoldable)]
pub struct InstantiatedPredicates<'tcx> {
    pub predicates: Vec<Predicate<'tcx>>,
    pub spans: Vec<Span>,
}

impl<'tcx> InstantiatedPredicates<'tcx> {
    pub fn empty() -> InstantiatedPredicates<'tcx> {
        InstantiatedPredicates { predicates: vec![], spans: vec![] }
    }

    pub fn is_empty(&self) -> bool {
        self.predicates.is_empty()
    }
}

rustc_index::newtype_index! {
    /// "Universes" are used during type- and trait-checking in the
    /// presence of `for<..>` binders to control what sets of names are
    /// visible. Universes are arranged into a tree: the root universe
    /// contains names that are always visible. Each child then adds a new
    /// set of names that are visible, in addition to those of its parent.
    /// We say that the child universe "extends" the parent universe with
    /// new names.
    ///
    /// To make this more concrete, consider this program:
    ///
    /// ```
    /// struct Foo { }
    /// fn bar<T>(x: T) {
    ///   let y: for<'a> fn(&'a u8, Foo) = ...;
    /// }
    /// ```
    ///
    /// The struct name `Foo` is in the root universe U0. But the type
    /// parameter `T`, introduced on `bar`, is in an extended universe U1
    /// -- i.e., within `bar`, we can name both `T` and `Foo`, but outside
    /// of `bar`, we cannot name `T`. Then, within the type of `y`, the
    /// region `'a` is in a universe U2 that extends U1, because we can
    /// name it inside the fn type but not outside.
    ///
    /// Universes are used to do type- and trait-checking around these
    /// "forall" binders (also called **universal quantification**). The
    /// idea is that when, in the body of `bar`, we refer to `T` as a
    /// type, we aren't referring to any type in particular, but rather a
    /// kind of "fresh" type that is distinct from all other types we have
    /// actually declared. This is called a **placeholder** type, and we
    /// use universes to talk about this. In other words, a type name in
    /// universe 0 always corresponds to some "ground" type that the user
    /// declared, but a type name in a non-zero universe is a placeholder
    /// type -- an idealized representative of "types in general" that we
    /// use for checking generic functions.
    pub struct UniverseIndex {
        derive [HashStable]
        DEBUG_FORMAT = "U{}",
    }
}

impl UniverseIndex {
    pub const ROOT: UniverseIndex = UniverseIndex::from_u32(0);

    /// Returns the "next" universe index in order -- this new index
    /// is considered to extend all previous universes. This
    /// corresponds to entering a `forall` quantifier. So, for
    /// example, suppose we have this type in universe `U`:
    ///
    /// ```
    /// for<'a> fn(&'a u32)
    /// ```
    ///
    /// Once we "enter" into this `for<'a>` quantifier, we are in a
    /// new universe that extends `U` -- in this new universe, we can
    /// name the region `'a`, but that region was not nameable from
    /// `U` because it was not in scope there.
    pub fn next_universe(self) -> UniverseIndex {
        UniverseIndex::from_u32(self.private.checked_add(1).unwrap())
    }

    /// Returns `true` if `self` can name a name from `other` -- in other words,
    /// if the set of names in `self` is a superset of those in
    /// `other` (`self >= other`).
    pub fn can_name(self, other: UniverseIndex) -> bool {
        self.private >= other.private
    }

    /// Returns `true` if `self` cannot name some names from `other` -- in other
    /// words, if the set of names in `self` is a strict subset of
    /// those in `other` (`self < other`).
    pub fn cannot_name(self, other: UniverseIndex) -> bool {
        self.private < other.private
    }
}

/// The "placeholder index" fully defines a placeholder region.
/// Placeholder regions are identified by both a **universe** as well
/// as a "bound-region" within that universe. The `bound_region` is
/// basically a name -- distinct bound regions within the same
/// universe are just two regions with an unknown relationship to one
/// another.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, PartialOrd, Ord)]
pub struct Placeholder<T> {
    pub universe: UniverseIndex,
    pub name: T,
}

impl<'a, T> HashStable<StableHashingContext<'a>> for Placeholder<T>
where
    T: HashStable<StableHashingContext<'a>>,
{
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        self.universe.hash_stable(hcx, hasher);
        self.name.hash_stable(hcx, hasher);
    }
}

pub type PlaceholderRegion = Placeholder<BoundRegion>;

pub type PlaceholderType = Placeholder<BoundVar>;

pub type PlaceholderConst = Placeholder<BoundVar>;

/// When type checking, we use the `ParamEnv` to track
/// details about the set of where-clauses that are in scope at this
/// particular point.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, HashStable, TypeFoldable)]
pub struct ParamEnv<'tcx> {
    /// `Obligation`s that the caller must satisfy. This is basically
    /// the set of bounds on the in-scope type parameters, translated
    /// into `Obligation`s, and elaborated and normalized.
    pub caller_bounds: &'tcx List<ty::Predicate<'tcx>>,

    /// Typically, this is `Reveal::UserFacing`, but during codegen we
    /// want `Reveal::All` -- note that this is always paired with an
    /// empty environment. To get that, use `ParamEnv::reveal()`.
    pub reveal: traits::Reveal,

    /// If this `ParamEnv` comes from a call to `tcx.param_env(def_id)`,
    /// register that `def_id` (useful for transitioning to the chalk trait
    /// solver).
    pub def_id: Option<DefId>,
}

impl<'tcx> ParamEnv<'tcx> {
    /// Construct a trait environment suitable for contexts where
    /// there are no where-clauses in scope. Hidden types (like `impl
    /// Trait`) are left hidden, so this is suitable for ordinary
    /// type-checking.
    #[inline]
    pub fn empty() -> Self {
        Self::new(List::empty(), Reveal::UserFacing, None)
    }

    /// Construct a trait environment with no where-clauses in scope
    /// where the values of all `impl Trait` and other hidden types
    /// are revealed. This is suitable for monomorphized, post-typeck
    /// environments like codegen or doing optimizations.
    ///
    /// N.B., if you want to have predicates in scope, use `ParamEnv::new`,
    /// or invoke `param_env.with_reveal_all()`.
    #[inline]
    pub fn reveal_all() -> Self {
        Self::new(List::empty(), Reveal::All, None)
    }

    /// Construct a trait environment with the given set of predicates.
    #[inline]
    pub fn new(
        caller_bounds: &'tcx List<ty::Predicate<'tcx>>,
        reveal: Reveal,
        def_id: Option<DefId>,
    ) -> Self {
        ty::ParamEnv { caller_bounds, reveal, def_id }
    }

    /// Returns a new parameter environment with the same clauses, but
    /// which "reveals" the true results of projections in all cases
    /// (even for associated types that are specializable). This is
    /// the desired behavior during codegen and certain other special
    /// contexts; normally though we want to use `Reveal::UserFacing`,
    /// which is the default.
    pub fn with_reveal_all(self) -> Self {
        ty::ParamEnv { reveal: Reveal::All, ..self }
    }

    /// Returns this same environment but with no caller bounds.
    pub fn without_caller_bounds(self) -> Self {
        ty::ParamEnv { caller_bounds: List::empty(), ..self }
    }

    /// Creates a suitable environment in which to perform trait
    /// queries on the given value. When type-checking, this is simply
    /// the pair of the environment plus value. But when reveal is set to
    /// All, then if `value` does not reference any type parameters, we will
    /// pair it with the empty environment. This improves caching and is generally
    /// invisible.
    ///
    /// N.B., we preserve the environment when type-checking because it
    /// is possible for the user to have wacky where-clauses like
    /// `where Box<u32>: Copy`, which are clearly never
    /// satisfiable. We generally want to behave as if they were true,
    /// although the surrounding function is never reachable.
    pub fn and<T: TypeFoldable<'tcx>>(self, value: T) -> ParamEnvAnd<'tcx, T> {
        match self.reveal {
            Reveal::UserFacing => ParamEnvAnd { param_env: self, value },

            Reveal::All => {
                if value.is_global() {
                    ParamEnvAnd { param_env: self.without_caller_bounds(), value }
                } else {
                    ParamEnvAnd { param_env: self, value }
                }
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConstnessAnd<T> {
    pub constness: Constness,
    pub value: T,
}

// FIXME(ecstaticmorse): Audit all occurrences of `without_const().to_predicate()` to ensure that
// the constness of trait bounds is being propagated correctly.
pub trait WithConstness: Sized {
    #[inline]
    fn with_constness(self, constness: Constness) -> ConstnessAnd<Self> {
        ConstnessAnd { constness, value: self }
    }

    #[inline]
    fn with_const(self) -> ConstnessAnd<Self> {
        self.with_constness(Constness::Const)
    }

    #[inline]
    fn without_const(self) -> ConstnessAnd<Self> {
        self.with_constness(Constness::NotConst)
    }
}

impl<T> WithConstness for T {}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, TypeFoldable)]
pub struct ParamEnvAnd<'tcx, T> {
    pub param_env: ParamEnv<'tcx>,
    pub value: T,
}

impl<'tcx, T> ParamEnvAnd<'tcx, T> {
    pub fn into_parts(self) -> (ParamEnv<'tcx>, T) {
        (self.param_env, self.value)
    }
}

impl<'a, 'tcx, T> HashStable<StableHashingContext<'a>> for ParamEnvAnd<'tcx, T>
where
    T: HashStable<StableHashingContext<'a>>,
{
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        let ParamEnvAnd { ref param_env, ref value } = *self;

        param_env.hash_stable(hcx, hasher);
        value.hash_stable(hcx, hasher);
    }
}

#[derive(Copy, Clone, Debug, HashStable)]
pub struct Destructor {
    /// The `DefId` of the destructor method
    pub did: DefId,
}

bitflags! {
    #[derive(HashStable)]
    pub struct AdtFlags: u32 {
        const NO_ADT_FLAGS        = 0;
        /// Indicates whether the ADT is an enum.
        const IS_ENUM             = 1 << 0;
        /// Indicates whether the ADT is a union.
        const IS_UNION            = 1 << 1;
        /// Indicates whether the ADT is a struct.
        const IS_STRUCT           = 1 << 2;
        /// Indicates whether the ADT is a struct and has a constructor.
        const HAS_CTOR            = 1 << 3;
        /// Indicates whether the type is `PhantomData`.
        const IS_PHANTOM_DATA     = 1 << 4;
        /// Indicates whether the type has a `#[fundamental]` attribute.
        const IS_FUNDAMENTAL      = 1 << 5;
        /// Indicates whether the type is `Box`.
        const IS_BOX              = 1 << 6;
        /// Indicates whether the type is `ManuallyDrop`.
        const IS_MANUALLY_DROP    = 1 << 7;
        /// Indicates whether the variant list of this ADT is `#[non_exhaustive]`.
        /// (i.e., this flag is never set unless this ADT is an enum).
        const IS_VARIANT_LIST_NON_EXHAUSTIVE = 1 << 8;
    }
}

bitflags! {
    #[derive(HashStable)]
    pub struct VariantFlags: u32 {
        const NO_VARIANT_FLAGS        = 0;
        /// Indicates whether the field list of this variant is `#[non_exhaustive]`.
        const IS_FIELD_LIST_NON_EXHAUSTIVE = 1 << 0;
    }
}

/// Definition of a variant -- a struct's fields or a enum variant.
#[derive(Debug, HashStable)]
pub struct VariantDef {
    /// `DefId` that identifies the variant itself.
    /// If this variant belongs to a struct or union, then this is a copy of its `DefId`.
    pub def_id: DefId,
    /// `DefId` that identifies the variant's constructor.
    /// If this variant is a struct variant, then this is `None`.
    pub ctor_def_id: Option<DefId>,
    /// Variant or struct name.
    #[stable_hasher(project(name))]
    pub ident: Ident,
    /// Discriminant of this variant.
    pub discr: VariantDiscr,
    /// Fields of this variant.
    pub fields: Vec<FieldDef>,
    /// Type of constructor of variant.
    pub ctor_kind: CtorKind,
    /// Flags of the variant (e.g. is field list non-exhaustive)?
    flags: VariantFlags,
    /// Variant is obtained as part of recovering from a syntactic error.
    /// May be incomplete or bogus.
    pub recovered: bool,
}

impl<'tcx> VariantDef {
    /// Creates a new `VariantDef`.
    ///
    /// `variant_did` is the `DefId` that identifies the enum variant (if this `VariantDef`
    /// represents an enum variant).
    ///
    /// `ctor_did` is the `DefId` that identifies the constructor of unit or
    /// tuple-variants/structs. If this is a `struct`-variant then this should be `None`.
    ///
    /// `parent_did` is the `DefId` of the `AdtDef` representing the enum or struct that
    /// owns this variant. It is used for checking if a struct has `#[non_exhaustive]` w/out having
    /// to go through the redirect of checking the ctor's attributes - but compiling a small crate
    /// requires loading the `AdtDef`s for all the structs in the universe (e.g., coherence for any
    /// built-in trait), and we do not want to load attributes twice.
    ///
    /// If someone speeds up attribute loading to not be a performance concern, they can
    /// remove this hack and use the constructor `DefId` everywhere.
    pub fn new(
        tcx: TyCtxt<'tcx>,
        ident: Ident,
        variant_did: Option<DefId>,
        ctor_def_id: Option<DefId>,
        discr: VariantDiscr,
        fields: Vec<FieldDef>,
        ctor_kind: CtorKind,
        adt_kind: AdtKind,
        parent_did: DefId,
        recovered: bool,
    ) -> Self {
        debug!(
            "VariantDef::new(ident = {:?}, variant_did = {:?}, ctor_def_id = {:?}, discr = {:?},
             fields = {:?}, ctor_kind = {:?}, adt_kind = {:?}, parent_did = {:?})",
            ident, variant_did, ctor_def_id, discr, fields, ctor_kind, adt_kind, parent_did,
        );

        let mut flags = VariantFlags::NO_VARIANT_FLAGS;
        if adt_kind == AdtKind::Struct && tcx.has_attr(parent_did, sym::non_exhaustive) {
            debug!("found non-exhaustive field list for {:?}", parent_did);
            flags = flags | VariantFlags::IS_FIELD_LIST_NON_EXHAUSTIVE;
        } else if let Some(variant_did) = variant_did {
            if tcx.has_attr(variant_did, sym::non_exhaustive) {
                debug!("found non-exhaustive field list for {:?}", variant_did);
                flags = flags | VariantFlags::IS_FIELD_LIST_NON_EXHAUSTIVE;
            }
        }

        VariantDef {
            def_id: variant_did.unwrap_or(parent_did),
            ctor_def_id,
            ident,
            discr,
            fields,
            ctor_kind,
            flags,
            recovered,
        }
    }

    /// Is this field list non-exhaustive?
    #[inline]
    pub fn is_field_list_non_exhaustive(&self) -> bool {
        self.flags.intersects(VariantFlags::IS_FIELD_LIST_NON_EXHAUSTIVE)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, RustcEncodable, RustcDecodable, HashStable)]
pub enum VariantDiscr {
    /// Explicit value for this variant, i.e., `X = 123`.
    /// The `DefId` corresponds to the embedded constant.
    Explicit(DefId),

    /// The previous variant's discriminant plus one.
    /// For efficiency reasons, the distance from the
    /// last `Explicit` discriminant is being stored,
    /// or `0` for the first variant, if it has none.
    Relative(u32),
}

#[derive(Debug, HashStable)]
pub struct FieldDef {
    pub did: DefId,
    #[stable_hasher(project(name))]
    pub ident: Ident,
    pub vis: Visibility,
}

/// The definition of a user-defined type, e.g., a `struct`, `enum`, or `union`.
///
/// These are all interned (by `intern_adt_def`) into the `adt_defs` table.
///
/// The initialism *ADT* stands for an [*algebraic data type (ADT)*][adt].
/// This is slightly wrong because `union`s are not ADTs.
/// Moreover, Rust only allows recursive data types through indirection.
///
/// [adt]: https://en.wikipedia.org/wiki/Algebraic_data_type
pub struct AdtDef {
    /// The `DefId` of the struct, enum or union item.
    pub did: DefId,
    /// Variants of the ADT. If this is a struct or union, then there will be a single variant.
    pub variants: IndexVec<VariantIdx, VariantDef>,
    /// Flags of the ADT (e.g., is this a struct? is this non-exhaustive?).
    flags: AdtFlags,
    /// Repr options provided by the user.
    pub repr: ReprOptions,
}

impl PartialOrd for AdtDef {
    fn partial_cmp(&self, other: &AdtDef) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

/// There should be only one AdtDef for each `did`, therefore
/// it is fine to implement `Ord` only based on `did`.
impl Ord for AdtDef {
    fn cmp(&self, other: &AdtDef) -> Ordering {
        self.did.cmp(&other.did)
    }
}

impl PartialEq for AdtDef {
    // `AdtDef`s are always interned, and this is part of `TyS` equality.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}

impl Eq for AdtDef {}

impl Hash for AdtDef {
    #[inline]
    fn hash<H: Hasher>(&self, s: &mut H) {
        (self as *const AdtDef).hash(s)
    }
}

impl<'tcx> rustc_serialize::UseSpecializedEncodable for &'tcx AdtDef {
    fn default_encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        self.did.encode(s)
    }
}

impl<'tcx> rustc_serialize::UseSpecializedDecodable for &'tcx AdtDef {}

impl<'a> HashStable<StableHashingContext<'a>> for AdtDef {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        thread_local! {
            static CACHE: RefCell<FxHashMap<usize, Fingerprint>> = Default::default();
        }

        let hash: Fingerprint = CACHE.with(|cache| {
            let addr = self as *const AdtDef as usize;
            *cache.borrow_mut().entry(addr).or_insert_with(|| {
                let ty::AdtDef { did, ref variants, ref flags, ref repr } = *self;

                let mut hasher = StableHasher::new();
                did.hash_stable(hcx, &mut hasher);
                variants.hash_stable(hcx, &mut hasher);
                flags.hash_stable(hcx, &mut hasher);
                repr.hash_stable(hcx, &mut hasher);

                hasher.finish()
            })
        });

        hash.hash_stable(hcx, hasher);
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum AdtKind {
    Struct,
    Union,
    Enum,
}

impl Into<DataTypeKind> for AdtKind {
    fn into(self) -> DataTypeKind {
        match self {
            AdtKind::Struct => DataTypeKind::Struct,
            AdtKind::Union => DataTypeKind::Union,
            AdtKind::Enum => DataTypeKind::Enum,
        }
    }
}

bitflags! {
    #[derive(RustcEncodable, RustcDecodable, Default, HashStable)]
    pub struct ReprFlags: u8 {
        const IS_C               = 1 << 0;
        const IS_SIMD            = 1 << 1;
        const IS_TRANSPARENT     = 1 << 2;
        // Internal only for now. If true, don't reorder fields.
        const IS_LINEAR          = 1 << 3;
        // If true, don't expose any niche to type's context.
        const HIDE_NICHE         = 1 << 4;
        // Any of these flags being set prevent field reordering optimisation.
        const IS_UNOPTIMISABLE   = ReprFlags::IS_C.bits |
                                   ReprFlags::IS_SIMD.bits |
                                   ReprFlags::IS_LINEAR.bits;
    }
}

/// Represents the repr options provided by the user,
#[derive(Copy, Clone, Debug, Eq, PartialEq, RustcEncodable, RustcDecodable, Default, HashStable)]
pub struct ReprOptions {
    pub int: Option<attr::IntType>,
    pub align: Option<Align>,
    pub pack: Option<Align>,
    pub flags: ReprFlags,
}

impl ReprOptions {
    pub fn new(tcx: TyCtxt<'_>, did: DefId) -> ReprOptions {
        let mut flags = ReprFlags::empty();
        let mut size = None;
        let mut max_align: Option<Align> = None;
        let mut min_pack: Option<Align> = None;
        for attr in tcx.get_attrs(did).iter() {
            for r in attr::find_repr_attrs(&tcx.sess.parse_sess, attr) {
                flags.insert(match r {
                    attr::ReprC => ReprFlags::IS_C,
                    attr::ReprPacked(pack) => {
                        let pack = Align::from_bytes(pack as u64).unwrap();
                        min_pack = Some(if let Some(min_pack) = min_pack {
                            min_pack.min(pack)
                        } else {
                            pack
                        });
                        ReprFlags::empty()
                    }
                    attr::ReprTransparent => ReprFlags::IS_TRANSPARENT,
                    attr::ReprNoNiche => ReprFlags::HIDE_NICHE,
                    attr::ReprSimd => ReprFlags::IS_SIMD,
                    attr::ReprInt(i) => {
                        size = Some(i);
                        ReprFlags::empty()
                    }
                    attr::ReprAlign(align) => {
                        max_align = max_align.max(Some(Align::from_bytes(align as u64).unwrap()));
                        ReprFlags::empty()
                    }
                });
            }
        }

        // This is here instead of layout because the choice must make it into metadata.
        if !tcx.consider_optimizing(|| format!("Reorder fields of {:?}", tcx.def_path_str(did))) {
            flags.insert(ReprFlags::IS_LINEAR);
        }
        ReprOptions { int: size, align: max_align, pack: min_pack, flags }
    }

    #[inline]
    pub fn simd(&self) -> bool {
        self.flags.contains(ReprFlags::IS_SIMD)
    }
    #[inline]
    pub fn c(&self) -> bool {
        self.flags.contains(ReprFlags::IS_C)
    }
    #[inline]
    pub fn packed(&self) -> bool {
        self.pack.is_some()
    }
    #[inline]
    pub fn transparent(&self) -> bool {
        self.flags.contains(ReprFlags::IS_TRANSPARENT)
    }
    #[inline]
    pub fn linear(&self) -> bool {
        self.flags.contains(ReprFlags::IS_LINEAR)
    }
    #[inline]
    pub fn hide_niche(&self) -> bool {
        self.flags.contains(ReprFlags::HIDE_NICHE)
    }

    pub fn discr_type(&self) -> attr::IntType {
        self.int.unwrap_or(attr::SignedInt(ast::IntTy::Isize))
    }

    /// Returns `true` if this `#[repr()]` should inhabit "smart enum
    /// layout" optimizations, such as representing `Foo<&T>` as a
    /// single pointer.
    pub fn inhibit_enum_layout_opt(&self) -> bool {
        self.c() || self.int.is_some()
    }

    /// Returns `true` if this `#[repr()]` should inhibit struct field reordering
    /// optimizations, such as with `repr(C)`, `repr(packed(1))`, or `repr(<int>)`.
    pub fn inhibit_struct_field_reordering_opt(&self) -> bool {
        if let Some(pack) = self.pack {
            if pack.bytes() == 1 {
                return true;
            }
        }
        self.flags.intersects(ReprFlags::IS_UNOPTIMISABLE) || self.int.is_some()
    }

    /// Returns `true` if this `#[repr()]` should inhibit union ABI optimisations.
    pub fn inhibit_union_abi_opt(&self) -> bool {
        self.c()
    }
}

impl<'tcx> AdtDef {
    /// Creates a new `AdtDef`.
    fn new(
        tcx: TyCtxt<'_>,
        did: DefId,
        kind: AdtKind,
        variants: IndexVec<VariantIdx, VariantDef>,
        repr: ReprOptions,
    ) -> Self {
        debug!("AdtDef::new({:?}, {:?}, {:?}, {:?})", did, kind, variants, repr);
        let mut flags = AdtFlags::NO_ADT_FLAGS;

        if kind == AdtKind::Enum && tcx.has_attr(did, sym::non_exhaustive) {
            debug!("found non-exhaustive variant list for {:?}", did);
            flags = flags | AdtFlags::IS_VARIANT_LIST_NON_EXHAUSTIVE;
        }

        flags |= match kind {
            AdtKind::Enum => AdtFlags::IS_ENUM,
            AdtKind::Union => AdtFlags::IS_UNION,
            AdtKind::Struct => AdtFlags::IS_STRUCT,
        };

        if kind == AdtKind::Struct && variants[VariantIdx::new(0)].ctor_def_id.is_some() {
            flags |= AdtFlags::HAS_CTOR;
        }

        let attrs = tcx.get_attrs(did);
        if attr::contains_name(&attrs, sym::fundamental) {
            flags |= AdtFlags::IS_FUNDAMENTAL;
        }
        if Some(did) == tcx.lang_items().phantom_data() {
            flags |= AdtFlags::IS_PHANTOM_DATA;
        }
        if Some(did) == tcx.lang_items().owned_box() {
            flags |= AdtFlags::IS_BOX;
        }
        if Some(did) == tcx.lang_items().manually_drop() {
            flags |= AdtFlags::IS_MANUALLY_DROP;
        }

        AdtDef { did, variants, flags, repr }
    }

    /// Returns `true` if this is a struct.
    #[inline]
    pub fn is_struct(&self) -> bool {
        self.flags.contains(AdtFlags::IS_STRUCT)
    }

    /// Returns `true` if this is a union.
    #[inline]
    pub fn is_union(&self) -> bool {
        self.flags.contains(AdtFlags::IS_UNION)
    }

    /// Returns `true` if this is a enum.
    #[inline]
    pub fn is_enum(&self) -> bool {
        self.flags.contains(AdtFlags::IS_ENUM)
    }

    /// Returns `true` if the variant list of this ADT is `#[non_exhaustive]`.
    #[inline]
    pub fn is_variant_list_non_exhaustive(&self) -> bool {
        self.flags.contains(AdtFlags::IS_VARIANT_LIST_NON_EXHAUSTIVE)
    }

    /// Returns the kind of the ADT.
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

    /// Returns a description of this abstract data type.
    pub fn descr(&self) -> &'static str {
        match self.adt_kind() {
            AdtKind::Struct => "struct",
            AdtKind::Union => "union",
            AdtKind::Enum => "enum",
        }
    }

    /// Returns a description of a variant of this abstract data type.
    #[inline]
    pub fn variant_descr(&self) -> &'static str {
        match self.adt_kind() {
            AdtKind::Struct => "struct",
            AdtKind::Union => "union",
            AdtKind::Enum => "variant",
        }
    }

    /// If this function returns `true`, it implies that `is_struct` must return `true`.
    #[inline]
    pub fn has_ctor(&self) -> bool {
        self.flags.contains(AdtFlags::HAS_CTOR)
    }

    /// Returns `true` if this type is `#[fundamental]` for the purposes
    /// of coherence checking.
    #[inline]
    pub fn is_fundamental(&self) -> bool {
        self.flags.contains(AdtFlags::IS_FUNDAMENTAL)
    }

    /// Returns `true` if this is `PhantomData<T>`.
    #[inline]
    pub fn is_phantom_data(&self) -> bool {
        self.flags.contains(AdtFlags::IS_PHANTOM_DATA)
    }

    /// Returns `true` if this is Box<T>.
    #[inline]
    pub fn is_box(&self) -> bool {
        self.flags.contains(AdtFlags::IS_BOX)
    }

    /// Returns `true` if this is `ManuallyDrop<T>`.
    #[inline]
    pub fn is_manually_drop(&self) -> bool {
        self.flags.contains(AdtFlags::IS_MANUALLY_DROP)
    }

    /// Returns `true` if this type has a destructor.
    pub fn has_dtor(&self, tcx: TyCtxt<'tcx>) -> bool {
        self.destructor(tcx).is_some()
    }

    /// Asserts this is a struct or union and returns its unique variant.
    pub fn non_enum_variant(&self) -> &VariantDef {
        assert!(self.is_struct() || self.is_union());
        &self.variants[VariantIdx::new(0)]
    }

    #[inline]
    pub fn predicates(&self, tcx: TyCtxt<'tcx>) -> GenericPredicates<'tcx> {
        tcx.predicates_of(self.did)
    }

    /// Returns an iterator over all fields contained
    /// by this ADT.
    #[inline]
    pub fn all_fields(&self) -> impl Iterator<Item = &FieldDef> + Clone {
        self.variants.iter().flat_map(|v| v.fields.iter())
    }

    pub fn is_payloadfree(&self) -> bool {
        !self.variants.is_empty() && self.variants.iter().all(|v| v.fields.is_empty())
    }

    /// Return a `VariantDef` given a variant id.
    pub fn variant_with_id(&self, vid: DefId) -> &VariantDef {
        self.variants.iter().find(|v| v.def_id == vid).expect("variant_with_id: unknown variant")
    }

    /// Return a `VariantDef` given a constructor id.
    pub fn variant_with_ctor_id(&self, cid: DefId) -> &VariantDef {
        self.variants
            .iter()
            .find(|v| v.ctor_def_id == Some(cid))
            .expect("variant_with_ctor_id: unknown variant")
    }

    /// Return the index of `VariantDef` given a variant id.
    pub fn variant_index_with_id(&self, vid: DefId) -> VariantIdx {
        self.variants
            .iter_enumerated()
            .find(|(_, v)| v.def_id == vid)
            .expect("variant_index_with_id: unknown variant")
            .0
    }

    /// Return the index of `VariantDef` given a constructor id.
    pub fn variant_index_with_ctor_id(&self, cid: DefId) -> VariantIdx {
        self.variants
            .iter_enumerated()
            .find(|(_, v)| v.ctor_def_id == Some(cid))
            .expect("variant_index_with_ctor_id: unknown variant")
            .0
    }

    pub fn variant_of_res(&self, res: Res) -> &VariantDef {
        match res {
            Res::Def(DefKind::Variant, vid) => self.variant_with_id(vid),
            Res::Def(DefKind::Ctor(..), cid) => self.variant_with_ctor_id(cid),
            Res::Def(DefKind::Struct, _)
            | Res::Def(DefKind::Union, _)
            | Res::Def(DefKind::TyAlias, _)
            | Res::Def(DefKind::AssocTy, _)
            | Res::SelfTy(..)
            | Res::SelfCtor(..) => self.non_enum_variant(),
            _ => bug!("unexpected res {:?} in variant_of_res", res),
        }
    }

    #[inline]
    pub fn eval_explicit_discr(&self, tcx: TyCtxt<'tcx>, expr_did: DefId) -> Option<Discr<'tcx>> {
        let param_env = tcx.param_env(expr_did);
        let repr_type = self.repr.discr_type();
        match tcx.const_eval_poly(expr_did) {
            Ok(val) => {
                let ty = repr_type.to_ty(tcx);
                if let Some(b) = val.try_to_bits_for_ty(tcx, param_env, ty) {
                    trace!("discriminants: {} ({:?})", b, repr_type);
                    Some(Discr { val: b, ty })
                } else {
                    info!("invalid enum discriminant: {:#?}", val);
                    crate::mir::interpret::struct_error(
                        tcx.at(tcx.def_span(expr_did)),
                        "constant evaluation of enum discriminant resulted in non-integer",
                    )
                    .emit();
                    None
                }
            }
            Err(ErrorHandled::Reported(ErrorReported) | ErrorHandled::Linted) => {
                if !expr_did.is_local() {
                    span_bug!(
                        tcx.def_span(expr_did),
                        "variant discriminant evaluation succeeded \
                         in its crate but failed locally"
                    );
                }
                None
            }
            Err(ErrorHandled::TooGeneric) => {
                tcx.sess.delay_span_bug(
                    tcx.def_span(expr_did),
                    "enum discriminant depends on generic arguments",
                );
                None
            }
        }
    }

    #[inline]
    pub fn discriminants(
        &'tcx self,
        tcx: TyCtxt<'tcx>,
    ) -> impl Iterator<Item = (VariantIdx, Discr<'tcx>)> + Captures<'tcx> {
        let repr_type = self.repr.discr_type();
        let initial = repr_type.initial_discriminant(tcx);
        let mut prev_discr = None::<Discr<'tcx>>;
        self.variants.iter_enumerated().map(move |(i, v)| {
            let mut discr = prev_discr.map_or(initial, |d| d.wrap_incr(tcx));
            if let VariantDiscr::Explicit(expr_did) = v.discr {
                if let Some(new_discr) = self.eval_explicit_discr(tcx, expr_did) {
                    discr = new_discr;
                }
            }
            prev_discr = Some(discr);

            (i, discr)
        })
    }

    #[inline]
    pub fn variant_range(&self) -> Range<VariantIdx> {
        VariantIdx::new(0)..VariantIdx::new(self.variants.len())
    }

    /// Computes the discriminant value used by a specific variant.
    /// Unlike `discriminants`, this is (amortized) constant-time,
    /// only doing at most one query for evaluating an explicit
    /// discriminant (the last one before the requested variant),
    /// assuming there are no constant-evaluation errors there.
    #[inline]
    pub fn discriminant_for_variant(
        &self,
        tcx: TyCtxt<'tcx>,
        variant_index: VariantIdx,
    ) -> Discr<'tcx> {
        let (val, offset) = self.discriminant_def_for_variant(variant_index);
        let explicit_value = val
            .and_then(|expr_did| self.eval_explicit_discr(tcx, expr_did))
            .unwrap_or_else(|| self.repr.discr_type().initial_discriminant(tcx));
        explicit_value.checked_add(tcx, offset as u128).0
    }

    /// Yields a `DefId` for the discriminant and an offset to add to it
    /// Alternatively, if there is no explicit discriminant, returns the
    /// inferred discriminant directly.
    pub fn discriminant_def_for_variant(&self, variant_index: VariantIdx) -> (Option<DefId>, u32) {
        let mut explicit_index = variant_index.as_u32();
        let expr_did;
        loop {
            match self.variants[VariantIdx::from_u32(explicit_index)].discr {
                ty::VariantDiscr::Relative(0) => {
                    expr_did = None;
                    break;
                }
                ty::VariantDiscr::Relative(distance) => {
                    explicit_index -= distance;
                }
                ty::VariantDiscr::Explicit(did) => {
                    expr_did = Some(did);
                    break;
                }
            }
        }
        (expr_did, variant_index.as_u32() - explicit_index)
    }

    pub fn destructor(&self, tcx: TyCtxt<'tcx>) -> Option<Destructor> {
        tcx.adt_destructor(self.did)
    }

    /// Returns a list of types such that `Self: Sized` if and only
    /// if that type is `Sized`, or `TyErr` if this type is recursive.
    ///
    /// Oddly enough, checking that the sized-constraint is `Sized` is
    /// actually more expressive than checking all members:
    /// the `Sized` trait is inductive, so an associated type that references
    /// `Self` would prevent its containing ADT from being `Sized`.
    ///
    /// Due to normalization being eager, this applies even if
    /// the associated type is behind a pointer (e.g., issue #31299).
    pub fn sized_constraint(&self, tcx: TyCtxt<'tcx>) -> &'tcx [Ty<'tcx>] {
        tcx.adt_sized_constraint(self.did).0
    }
}

impl<'tcx> FieldDef {
    /// Returns the type of this field. The `subst` is typically obtained
    /// via the second field of `TyKind::AdtDef`.
    pub fn ty(&self, tcx: TyCtxt<'tcx>, subst: SubstsRef<'tcx>) -> Ty<'tcx> {
        tcx.type_of(self.did).subst(tcx, subst)
    }
}

/// Represents the various closure traits in the language. This
/// will determine the type of the environment (`self`, in the
/// desugaring) argument that the closure expects.
///
/// You can get the environment type of a closure using
/// `tcx.closure_env_ty()`.
#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
#[derive(HashStable)]
pub enum ClosureKind {
    // Warning: Ordering is significant here! The ordering is chosen
    // because the trait Fn is a subtrait of FnMut and so in turn, and
    // hence we order it so that Fn < FnMut < FnOnce.
    Fn,
    FnMut,
    FnOnce,
}

impl<'tcx> ClosureKind {
    // This is the initial value used when doing upvar inference.
    pub const LATTICE_BOTTOM: ClosureKind = ClosureKind::Fn;

    pub fn trait_did(&self, tcx: TyCtxt<'tcx>) -> DefId {
        match *self {
            ClosureKind::Fn => tcx.require_lang_item(FnTraitLangItem, None),
            ClosureKind::FnMut => tcx.require_lang_item(FnMutTraitLangItem, None),
            ClosureKind::FnOnce => tcx.require_lang_item(FnOnceTraitLangItem, None),
        }
    }

    /// Returns `true` if this a type that impls this closure kind
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

    /// Returns the representative scalar type for this closure kind.
    /// See `TyS::to_opt_closure_kind` for more details.
    pub fn to_ty(self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        match self {
            ty::ClosureKind::Fn => tcx.types.i8,
            ty::ClosureKind::FnMut => tcx.types.i16,
            ty::ClosureKind::FnOnce => tcx.types.i32,
        }
    }
}

impl BorrowKind {
    pub fn from_mutbl(m: hir::Mutability) -> BorrowKind {
        match m {
            hir::Mutability::Mut => MutBorrow,
            hir::Mutability::Not => ImmBorrow,
        }
    }

    /// Returns a mutability `m` such that an `&m T` pointer could be used to obtain this borrow
    /// kind. Because borrow kinds are richer than mutabilities, we sometimes have to pick a
    /// mutability that is stronger than necessary so that it at least *would permit* the borrow in
    /// question.
    pub fn to_mutbl_lossy(self) -> hir::Mutability {
        match self {
            MutBorrow => hir::Mutability::Mut,
            ImmBorrow => hir::Mutability::Not,

            // We have no type corresponding to a unique imm borrow, so
            // use `&mut`. It gives all the capabilities of an `&uniq`
            // and hence is a safe "over approximation".
            UniqueImmBorrow => hir::Mutability::Mut,
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

pub type Attributes<'tcx> = &'tcx [ast::Attribute];

#[derive(Debug, PartialEq, Eq)]
pub enum ImplOverlapKind {
    /// These impls are always allowed to overlap.
    Permitted {
        /// Whether or not the impl is permitted due to the trait being a `#[marker]` trait
        marker: bool,
    },
    /// These impls are allowed to overlap, but that raises
    /// an issue #33140 future-compatibility warning.
    ///
    /// Some background: in Rust 1.0, the trait-object types `Send + Sync` (today's
    /// `dyn Send + Sync`) and `Sync + Send` (now `dyn Sync + Send`) were different.
    ///
    /// The widely-used version 0.1.0 of the crate `traitobject` had accidentally relied
    /// that difference, making what reduces to the following set of impls:
    ///
    /// ```
    /// trait Trait {}
    /// impl Trait for dyn Send + Sync {}
    /// impl Trait for dyn Sync + Send {}
    /// ```
    ///
    /// Obviously, once we made these types be identical, that code causes a coherence
    /// error and a fairly big headache for us. However, luckily for us, the trait
    /// `Trait` used in this case is basically a marker trait, and therefore having
    /// overlapping impls for it is sound.
    ///
    /// To handle this, we basically regard the trait as a marker trait, with an additional
    /// future-compatibility warning. To avoid accidentally "stabilizing" this feature,
    /// it has the following restrictions:
    ///
    /// 1. The trait must indeed be a marker-like trait (i.e., no items), and must be
    /// positive impls.
    /// 2. The trait-ref of both impls must be equal.
    /// 3. The trait-ref of both impls must be a trait object type consisting only of
    /// marker traits.
    /// 4. Neither of the impls can have any where-clauses.
    ///
    /// Once `traitobject` 0.1.0 is no longer an active concern, this hack can be removed.
    Issue33140,
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn body_tables(self, body: hir::BodyId) -> &'tcx TypeckTables<'tcx> {
        self.typeck_tables_of(self.hir().body_owner_def_id(body))
    }

    /// Returns an iterator of the `DefId`s for all body-owners in this
    /// crate. If you would prefer to iterate over the bodies
    /// themselves, you can do `self.hir().krate().body_ids.iter()`.
    pub fn body_owners(self) -> impl Iterator<Item = LocalDefId> + Captures<'tcx> + 'tcx {
        self.hir()
            .krate()
            .body_ids
            .iter()
            .map(move |&body_id| self.hir().body_owner_def_id(body_id))
    }

    pub fn par_body_owners<F: Fn(LocalDefId) + sync::Sync + sync::Send>(self, f: F) {
        par_iter(&self.hir().krate().body_ids)
            .for_each(|&body_id| f(self.hir().body_owner_def_id(body_id)));
    }

    pub fn provided_trait_methods(self, id: DefId) -> impl 'tcx + Iterator<Item = &'tcx AssocItem> {
        self.associated_items(id)
            .in_definition_order()
            .filter(|item| item.kind == AssocKind::Fn && item.defaultness.has_value())
    }

    pub fn trait_relevant_for_never(self, did: DefId) -> bool {
        self.associated_items(did).in_definition_order().any(|item| item.relevant_for_never())
    }

    pub fn opt_item_name(self, def_id: DefId) -> Option<Ident> {
        def_id
            .as_local()
            .and_then(|def_id| self.hir().get(self.hir().as_local_hir_id(def_id)).ident())
    }

    pub fn opt_associated_item(self, def_id: DefId) -> Option<AssocItem> {
        let is_associated_item = if let Some(def_id) = def_id.as_local() {
            match self.hir().get(self.hir().as_local_hir_id(def_id)) {
                Node::TraitItem(_) | Node::ImplItem(_) => true,
                _ => false,
            }
        } else {
            match self.def_kind(def_id) {
                DefKind::AssocConst | DefKind::AssocFn | DefKind::AssocTy => true,
                _ => false,
            }
        };

        is_associated_item.then(|| self.associated_item(def_id))
    }

    pub fn field_index(self, hir_id: hir::HirId, tables: &TypeckTables<'_>) -> usize {
        tables.field_indices().get(hir_id).cloned().expect("no index for a field")
    }

    pub fn find_field_index(self, ident: Ident, variant: &VariantDef) -> Option<usize> {
        variant.fields.iter().position(|field| self.hygienic_eq(ident, field.ident, variant.def_id))
    }

    /// Returns `true` if the impls are the same polarity and the trait either
    /// has no items or is annotated #[marker] and prevents item overrides.
    pub fn impls_are_allowed_to_overlap(
        self,
        def_id1: DefId,
        def_id2: DefId,
    ) -> Option<ImplOverlapKind> {
        // If either trait impl references an error, they're allowed to overlap,
        // as one of them essentially doesn't exist.
        if self.impl_trait_ref(def_id1).map_or(false, |tr| tr.references_error())
            || self.impl_trait_ref(def_id2).map_or(false, |tr| tr.references_error())
        {
            return Some(ImplOverlapKind::Permitted { marker: false });
        }

        match (self.impl_polarity(def_id1), self.impl_polarity(def_id2)) {
            (ImplPolarity::Reservation, _) | (_, ImplPolarity::Reservation) => {
                // `#[rustc_reservation_impl]` impls don't overlap with anything
                debug!(
                    "impls_are_allowed_to_overlap({:?}, {:?}) = Some(Permitted) (reservations)",
                    def_id1, def_id2
                );
                return Some(ImplOverlapKind::Permitted { marker: false });
            }
            (ImplPolarity::Positive, ImplPolarity::Negative)
            | (ImplPolarity::Negative, ImplPolarity::Positive) => {
                // `impl AutoTrait for Type` + `impl !AutoTrait for Type`
                debug!(
                    "impls_are_allowed_to_overlap({:?}, {:?}) - None (differing polarities)",
                    def_id1, def_id2
                );
                return None;
            }
            (ImplPolarity::Positive, ImplPolarity::Positive)
            | (ImplPolarity::Negative, ImplPolarity::Negative) => {}
        };

        let is_marker_overlap = {
            let is_marker_impl = |def_id: DefId| -> bool {
                let trait_ref = self.impl_trait_ref(def_id);
                trait_ref.map_or(false, |tr| self.trait_def(tr.def_id).is_marker)
            };
            is_marker_impl(def_id1) && is_marker_impl(def_id2)
        };

        if is_marker_overlap {
            debug!(
                "impls_are_allowed_to_overlap({:?}, {:?}) = Some(Permitted) (marker overlap)",
                def_id1, def_id2
            );
            Some(ImplOverlapKind::Permitted { marker: true })
        } else {
            if let Some(self_ty1) = self.issue33140_self_ty(def_id1) {
                if let Some(self_ty2) = self.issue33140_self_ty(def_id2) {
                    if self_ty1 == self_ty2 {
                        debug!(
                            "impls_are_allowed_to_overlap({:?}, {:?}) - issue #33140 HACK",
                            def_id1, def_id2
                        );
                        return Some(ImplOverlapKind::Issue33140);
                    } else {
                        debug!(
                            "impls_are_allowed_to_overlap({:?}, {:?}) - found {:?} != {:?}",
                            def_id1, def_id2, self_ty1, self_ty2
                        );
                    }
                }
            }

            debug!("impls_are_allowed_to_overlap({:?}, {:?}) = None", def_id1, def_id2);
            None
        }
    }

    /// Returns `ty::VariantDef` if `res` refers to a struct,
    /// or variant or their constructors, panics otherwise.
    pub fn expect_variant_res(self, res: Res) -> &'tcx VariantDef {
        match res {
            Res::Def(DefKind::Variant, did) => {
                let enum_did = self.parent(did).unwrap();
                self.adt_def(enum_did).variant_with_id(did)
            }
            Res::Def(DefKind::Struct | DefKind::Union, did) => self.adt_def(did).non_enum_variant(),
            Res::Def(DefKind::Ctor(CtorOf::Variant, ..), variant_ctor_did) => {
                let variant_did = self.parent(variant_ctor_did).unwrap();
                let enum_did = self.parent(variant_did).unwrap();
                self.adt_def(enum_did).variant_with_ctor_id(variant_ctor_did)
            }
            Res::Def(DefKind::Ctor(CtorOf::Struct, ..), ctor_did) => {
                let struct_did = self.parent(ctor_did).expect("struct ctor has no parent");
                self.adt_def(struct_did).non_enum_variant()
            }
            _ => bug!("expect_variant_res used with unexpected res {:?}", res),
        }
    }

    pub fn item_name(self, id: DefId) -> Symbol {
        if id.index == CRATE_DEF_INDEX {
            self.original_crate_name(id.krate)
        } else {
            let def_key = self.def_key(id);
            match def_key.disambiguated_data.data {
                // The name of a constructor is that of its parent.
                rustc_hir::definitions::DefPathData::Ctor => {
                    self.item_name(DefId { krate: id.krate, index: def_key.parent.unwrap() })
                }
                _ => def_key.disambiguated_data.data.get_opt_name().unwrap_or_else(|| {
                    bug!("item_name: no name for {:?}", self.def_path(id));
                }),
            }
        }
    }

    /// Returns the possibly-auto-generated MIR of a `(DefId, Subst)` pair.
    pub fn instance_mir(self, instance: ty::InstanceDef<'tcx>) -> &'tcx Body<'tcx> {
        match instance {
            ty::InstanceDef::Item(did) => self.optimized_mir(did),
            ty::InstanceDef::VtableShim(..)
            | ty::InstanceDef::ReifyShim(..)
            | ty::InstanceDef::Intrinsic(..)
            | ty::InstanceDef::FnPtrShim(..)
            | ty::InstanceDef::Virtual(..)
            | ty::InstanceDef::ClosureOnceShim { .. }
            | ty::InstanceDef::DropGlue(..)
            | ty::InstanceDef::CloneShim(..) => self.mir_shims(instance),
        }
    }

    /// Gets the attributes of a definition.
    pub fn get_attrs(self, did: DefId) -> Attributes<'tcx> {
        if let Some(did) = did.as_local() {
            self.hir().attrs(self.hir().as_local_hir_id(did))
        } else {
            self.item_attrs(did)
        }
    }

    /// Determines whether an item is annotated with an attribute.
    pub fn has_attr(self, did: DefId, attr: Symbol) -> bool {
        attr::contains_name(&self.get_attrs(did), attr)
    }

    /// Returns `true` if this is an `auto trait`.
    pub fn trait_is_auto(self, trait_def_id: DefId) -> bool {
        self.trait_def(trait_def_id).has_auto_impl
    }

    pub fn generator_layout(self, def_id: DefId) -> &'tcx GeneratorLayout<'tcx> {
        self.optimized_mir(def_id).generator_layout.as_ref().unwrap()
    }

    /// Given the `DefId` of an impl, returns the `DefId` of the trait it implements.
    /// If it implements no trait, returns `None`.
    pub fn trait_id_of_impl(self, def_id: DefId) -> Option<DefId> {
        self.impl_trait_ref(def_id).map(|tr| tr.def_id)
    }

    /// If the given defid describes a method belonging to an impl, returns the
    /// `DefId` of the impl that the method belongs to; otherwise, returns `None`.
    pub fn impl_of_method(self, def_id: DefId) -> Option<DefId> {
        self.opt_associated_item(def_id).and_then(|trait_item| match trait_item.container {
            TraitContainer(_) => None,
            ImplContainer(def_id) => Some(def_id),
        })
    }

    /// Looks up the span of `impl_did` if the impl is local; otherwise returns `Err`
    /// with the name of the crate containing the impl.
    pub fn span_of_impl(self, impl_did: DefId) -> Result<Span, Symbol> {
        if let Some(impl_did) = impl_did.as_local() {
            let hir_id = self.hir().as_local_hir_id(impl_did);
            Ok(self.hir().span(hir_id))
        } else {
            Err(self.crate_name(impl_did.krate))
        }
    }

    /// Hygienically compares a use-site name (`use_name`) for a field or an associated item with
    /// its supposed definition name (`def_name`). The method also needs `DefId` of the supposed
    /// definition's parent/scope to perform comparison.
    pub fn hygienic_eq(self, use_name: Ident, def_name: Ident, def_parent_def_id: DefId) -> bool {
        // We could use `Ident::eq` here, but we deliberately don't. The name
        // comparison fails frequently, and we want to avoid the expensive
        // `normalize_to_macros_2_0()` calls required for the span comparison whenever possible.
        use_name.name == def_name.name
            && use_name
                .span
                .ctxt()
                .hygienic_eq(def_name.span.ctxt(), self.expansion_that_defined(def_parent_def_id))
    }

    fn expansion_that_defined(self, scope: DefId) -> ExpnId {
        match scope.as_local() {
            Some(scope) => self.hir().definitions().expansion_that_defined(scope),
            None => ExpnId::root(),
        }
    }

    pub fn adjust_ident(self, mut ident: Ident, scope: DefId) -> Ident {
        ident.span.normalize_to_macros_2_0_and_adjust(self.expansion_that_defined(scope));
        ident
    }

    pub fn adjust_ident_and_get_scope(
        self,
        mut ident: Ident,
        scope: DefId,
        block: hir::HirId,
    ) -> (Ident, DefId) {
        let scope =
            match ident.span.normalize_to_macros_2_0_and_adjust(self.expansion_that_defined(scope))
            {
                Some(actual_expansion) => {
                    self.hir().definitions().parent_module_of_macro_def(actual_expansion)
                }
                None => self.parent_module(block).to_def_id(),
            };
        (ident, scope)
    }

    pub fn is_object_safe(self, key: DefId) -> bool {
        self.object_safety_violations(key).is_empty()
    }
}

#[derive(Clone, HashStable)]
pub struct AdtSizedConstraint<'tcx>(pub &'tcx [Ty<'tcx>]);

/// Yields the parent function's `DefId` if `def_id` is an `impl Trait` definition.
pub fn is_impl_trait_defn(tcx: TyCtxt<'_>, def_id: DefId) -> Option<DefId> {
    if let Some(def_id) = def_id.as_local() {
        if let Node::Item(item) = tcx.hir().get(tcx.hir().as_local_hir_id(def_id)) {
            if let hir::ItemKind::OpaqueTy(ref opaque_ty) = item.kind {
                return opaque_ty.impl_trait_fn;
            }
        }
    }
    None
}

pub fn provide(providers: &mut ty::query::Providers<'_>) {
    context::provide(providers);
    erase_regions::provide(providers);
    layout::provide(providers);
    super::util::bug::provide(providers);
    *providers = ty::query::Providers {
        trait_impls_of: trait_def::trait_impls_of_provider,
        all_local_trait_impls: trait_def::all_local_trait_impls,
        ..*providers
    };
}

/// A map for the local crate mapping each type to a vector of its
/// inherent impls. This is not meant to be used outside of coherence;
/// rather, you should request the vector for a specific type via
/// `tcx.inherent_impls(def_id)` so as to minimize your dependencies
/// (constructing this map requires touching the entire crate).
#[derive(Clone, Debug, Default, HashStable)]
pub struct CrateInherentImpls {
    pub inherent_impls: DefIdMap<Vec<DefId>>,
}

#[derive(Clone, Copy, PartialEq, Eq, RustcEncodable, RustcDecodable, HashStable)]
pub struct SymbolName {
    // FIXME: we don't rely on interning or equality here - better have
    // this be a `&'tcx str`.
    pub name: Symbol,
}

impl SymbolName {
    pub fn new(name: &str) -> SymbolName {
        SymbolName { name: Symbol::intern(name) }
    }
}

impl PartialOrd for SymbolName {
    fn partial_cmp(&self, other: &SymbolName) -> Option<Ordering> {
        self.name.as_str().partial_cmp(&other.name.as_str())
    }
}

/// Ordering must use the chars to ensure reproducible builds.
impl Ord for SymbolName {
    fn cmp(&self, other: &SymbolName) -> Ordering {
        self.name.as_str().cmp(&other.name.as_str())
    }
}

impl fmt::Display for SymbolName {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.name, fmt)
    }
}

impl fmt::Debug for SymbolName {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.name, fmt)
    }
}
