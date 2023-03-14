//! Defines how the compiler represents types internally.
//!
//! Two important entities in this module are:
//!
//! - [`rustc_middle::ty::Ty`], used to represent the semantics of a type.
//! - [`rustc_middle::ty::TyCtxt`], the central data structure in the compiler.
//!
//! For more information, see ["The `ty` module: representing types"] in the rustc-dev-guide.
//!
//! ["The `ty` module: representing types"]: https://rustc-dev-guide.rust-lang.org/ty.html

#![allow(rustc::usage_of_ty_tykind)]

pub use self::fold::{FallibleTypeFolder, TypeFoldable, TypeFolder, TypeSuperFoldable};
pub use self::visit::{TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor};
pub use self::AssocItemContainer::*;
pub use self::BorrowKind::*;
pub use self::IntVarValue::*;
pub use self::Variance::*;
use crate::error::{OpaqueHiddenTypeMismatch, TypeMismatchReason};
use crate::metadata::ModChild;
use crate::middle::privacy::EffectiveVisibilities;
use crate::mir::{Body, GeneratorLayout};
use crate::traits::{self, Reveal};
use crate::ty;
use crate::ty::fast_reject::SimplifiedType;
use crate::ty::util::Discr;
pub use adt::*;
pub use assoc::*;
pub use generics::*;
use rustc_ast as ast;
use rustc_ast::node_id::NodeMap;
use rustc_attr as attr;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap, FxIndexSet};
use rustc_data_structures::intern::Interned;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::steal::Steal;
use rustc_data_structures::tagged_ptr::CopyTaggedPtr;
use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, CtorOf, DefKind, DocLinkResMap, LifetimeRes, Res};
use rustc_hir::def_id::{CrateNum, DefId, DefIdMap, LocalDefId, LocalDefIdMap};
use rustc_hir::Node;
use rustc_index::vec::IndexVec;
use rustc_macros::HashStable;
use rustc_query_system::ich::StableHashingContext;
use rustc_serialize::{Decodable, Encodable};
use rustc_session::lint::LintBuffer;
pub use rustc_session::lint::RegisteredTools;
use rustc_span::hygiene::MacroKind;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{ExpnId, ExpnKind, Span};
use rustc_target::abi::{Align, Integer, IntegerType, VariantIdx};
pub use rustc_target::abi::{ReprFlags, ReprOptions};
use rustc_type_ir::WithCachedTypeInfo;
pub use subst::*;
pub use vtable::*;

use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::mem;
use std::num::NonZeroUsize;
use std::ops::ControlFlow;
use std::{fmt, str};

pub use crate::ty::diagnostics::*;
pub use rustc_type_ir::AliasKind::*;
pub use rustc_type_ir::DynKind::*;
pub use rustc_type_ir::InferTy::*;
pub use rustc_type_ir::RegionKind::*;
pub use rustc_type_ir::TyKind::*;
pub use rustc_type_ir::*;

pub use self::binding::BindingMode;
pub use self::binding::BindingMode::*;
pub use self::closure::{
    is_ancestor_or_same_capture, place_to_string_for_capture, BorrowKind, CaptureInfo,
    CapturedPlace, ClosureKind, ClosureTypeInfo, MinCaptureInformationMap, MinCaptureList,
    RootVariableMinCaptureList, UpvarCapture, UpvarCaptureMap, UpvarId, UpvarListMap, UpvarPath,
    CAPTURE_STRUCT_LOCAL,
};
pub use self::consts::{
    Const, ConstData, ConstInt, ConstKind, Expr, InferConst, ScalarInt, UnevaluatedConst, ValTree,
};
pub use self::context::{
    tls, CtxtInterners, DeducedParamAttrs, FreeRegionInfo, GlobalCtxt, Lift, OnDiskCache, TyCtxt,
    TyCtxtFeed,
};
pub use self::instance::{Instance, InstanceDef, ShortInstance, UnusedGenericParams};
pub use self::list::List;
pub use self::parameterized::ParameterizedOverTcx;
pub use self::rvalue_scopes::RvalueScopes;
pub use self::sty::BoundRegionKind::*;
pub use self::sty::{
    AliasTy, Article, Binder, BoundRegion, BoundRegionKind, BoundTy, BoundTyKind, BoundVar,
    BoundVariableKind, CanonicalPolyFnSig, ClosureSubsts, ClosureSubstsParts, ConstVid,
    EarlyBoundRegion, ExistentialPredicate, ExistentialProjection, ExistentialTraitRef, FnSig,
    FreeRegion, GenSig, GeneratorSubsts, GeneratorSubstsParts, InlineConstSubsts,
    InlineConstSubstsParts, ParamConst, ParamTy, PolyExistentialPredicate,
    PolyExistentialProjection, PolyExistentialTraitRef, PolyFnSig, PolyGenSig, PolyTraitRef,
    Region, RegionKind, RegionVid, TraitRef, TyKind, TypeAndMut, UpvarSubsts, VarianceDiagInfo,
};
pub use self::trait_def::TraitDef;
pub use self::typeck_results::{
    CanonicalUserType, CanonicalUserTypeAnnotation, CanonicalUserTypeAnnotations,
    GeneratorDiagnosticData, GeneratorInteriorTypeCause, TypeckResults, UserType,
    UserTypeAnnotationIndex,
};

pub mod _match;
pub mod abstract_const;
pub mod adjustment;
pub mod binding;
pub mod cast;
pub mod codec;
pub mod error;
pub mod fast_reject;
pub mod flags;
pub mod fold;
pub mod inhabitedness;
pub mod layout;
pub mod normalize_erasing_regions;
pub mod print;
pub mod query;
pub mod relate;
pub mod subst;
pub mod trait_def;
pub mod util;
pub mod visit;
pub mod vtable;
pub mod walk;

mod adt;
mod assoc;
mod closure;
mod consts;
mod context;
mod diagnostics;
mod erase_regions;
mod generics;
mod impls_ty;
mod instance;
mod list;
mod opaque_types;
mod parameterized;
mod rvalue_scopes;
mod structural_impls;
mod sty;
mod typeck_results;

// Data types

pub struct ResolverOutputs {
    pub global_ctxt: ResolverGlobalCtxt,
    pub ast_lowering: ResolverAstLowering,
}

#[derive(Debug)]
pub struct ResolverGlobalCtxt {
    pub visibilities: FxHashMap<LocalDefId, Visibility>,
    /// This field is used to decide whether we should make `PRIVATE_IN_PUBLIC` a hard error.
    pub has_pub_restricted: bool,
    /// Item with a given `LocalDefId` was defined during macro expansion with ID `ExpnId`.
    pub expn_that_defined: FxHashMap<LocalDefId, ExpnId>,
    pub effective_visibilities: EffectiveVisibilities,
    pub extern_crate_map: FxHashMap<LocalDefId, CrateNum>,
    pub maybe_unused_trait_imports: FxIndexSet<LocalDefId>,
    pub reexport_map: FxHashMap<LocalDefId, Vec<ModChild>>,
    pub glob_map: FxHashMap<LocalDefId, FxHashSet<Symbol>>,
    pub main_def: Option<MainDefinition>,
    pub trait_impls: FxIndexMap<DefId, Vec<LocalDefId>>,
    /// A list of proc macro LocalDefIds, written out in the order in which
    /// they are declared in the static array generated by proc_macro_harness.
    pub proc_macros: Vec<LocalDefId>,
    /// Mapping from ident span to path span for paths that don't exist as written, but that
    /// exist under `std`. For example, wrote `str::from_utf8` instead of `std::str::from_utf8`.
    pub confused_type_with_std_module: FxHashMap<Span, Span>,
    pub doc_link_resolutions: FxHashMap<LocalDefId, DocLinkResMap>,
    pub doc_link_traits_in_scope: FxHashMap<LocalDefId, Vec<DefId>>,
    pub all_macro_rules: FxHashMap<Symbol, Res<ast::NodeId>>,
}

/// Resolutions that should only be used for lowering.
/// This struct is meant to be consumed by lowering.
#[derive(Debug)]
pub struct ResolverAstLowering {
    pub legacy_const_generic_args: FxHashMap<DefId, Option<Vec<usize>>>,

    /// Resolutions for nodes that have a single resolution.
    pub partial_res_map: NodeMap<hir::def::PartialRes>,
    /// Resolutions for import nodes, which have multiple resolutions in different namespaces.
    pub import_res_map: NodeMap<hir::def::PerNS<Option<Res<ast::NodeId>>>>,
    /// Resolutions for labels (node IDs of their corresponding blocks or loops).
    pub label_res_map: NodeMap<ast::NodeId>,
    /// Resolutions for lifetimes.
    pub lifetimes_res_map: NodeMap<LifetimeRes>,
    /// Lifetime parameters that lowering will have to introduce.
    pub extra_lifetime_params_map: NodeMap<Vec<(Ident, ast::NodeId, LifetimeRes)>>,

    pub next_node_id: ast::NodeId,

    pub node_id_to_def_id: FxHashMap<ast::NodeId, LocalDefId>,
    pub def_id_to_node_id: IndexVec<LocalDefId, ast::NodeId>,

    pub trait_map: NodeMap<Vec<hir::TraitCandidate>>,
    /// A small map keeping true kinds of built-in macros that appear to be fn-like on
    /// the surface (`macro` items in libcore), but are actually attributes or derives.
    pub builtin_macro_kinds: FxHashMap<LocalDefId, MacroKind>,
    /// List functions and methods for which lifetime elision was successful.
    pub lifetime_elision_allowed: FxHashSet<ast::NodeId>,

    /// Lints that were emitted by the resolver and early lints.
    pub lint_buffer: Steal<LintBuffer>,
}

#[derive(Clone, Copy, Debug)]
pub struct MainDefinition {
    pub res: Res<ast::NodeId>,
    pub is_import: bool,
    pub span: Span,
}

impl MainDefinition {
    pub fn opt_fn_def_id(self) -> Option<DefId> {
        if let Res::Def(DefKind::Fn, def_id) = self.res { Some(def_id) } else { None }
    }
}

/// The "header" of an impl is everything outside the body: a Self type, a trait
/// ref (in the case of a trait impl), and a set of predicates (from the
/// bounds / where-clauses).
#[derive(Clone, Debug, TypeFoldable, TypeVisitable)]
pub struct ImplHeader<'tcx> {
    pub impl_def_id: DefId,
    pub self_ty: Ty<'tcx>,
    pub trait_ref: Option<TraitRef<'tcx>>,
    pub predicates: Vec<Predicate<'tcx>>,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, TypeFoldable, TypeVisitable)]
pub enum ImplSubject<'tcx> {
    Trait(TraitRef<'tcx>),
    Inherent(Ty<'tcx>),
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, TyEncodable, TyDecodable, HashStable, Debug)]
#[derive(TypeFoldable, TypeVisitable)]
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

impl ImplPolarity {
    /// Flips polarity by turning `Positive` into `Negative` and `Negative` into `Positive`.
    pub fn flip(&self) -> Option<ImplPolarity> {
        match self {
            ImplPolarity::Positive => Some(ImplPolarity::Negative),
            ImplPolarity::Negative => Some(ImplPolarity::Positive),
            ImplPolarity::Reservation => None,
        }
    }
}

impl fmt::Display for ImplPolarity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Positive => f.write_str("positive"),
            Self::Negative => f.write_str("negative"),
            Self::Reservation => f.write_str("reservation"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Copy, Hash, Encodable, Decodable, HashStable)]
pub enum Visibility<Id = LocalDefId> {
    /// Visible everywhere (including in other crates).
    Public,
    /// Visible only in the given crate-local module.
    Restricted(Id),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, HashStable, TyEncodable, TyDecodable)]
pub enum BoundConstness {
    /// `T: Trait`
    NotConst,
    /// `T: ~const Trait`
    ///
    /// Requires resolving to const only when we are in a const context.
    ConstIfConst,
}

impl BoundConstness {
    /// Reduce `self` and `constness` to two possible combined states instead of four.
    pub fn and(&mut self, constness: hir::Constness) -> hir::Constness {
        match (constness, self) {
            (hir::Constness::Const, BoundConstness::ConstIfConst) => hir::Constness::Const,
            (_, this) => {
                *this = BoundConstness::NotConst;
                hir::Constness::NotConst
            }
        }
    }
}

impl fmt::Display for BoundConstness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotConst => f.write_str("normal"),
            Self::ConstIfConst => f.write_str("`~const`"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Copy, Hash, TyEncodable, TyDecodable, HashStable)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct ClosureSizeProfileData<'tcx> {
    /// Tuple containing the types of closure captures before the feature `capture_disjoint_fields`
    pub before_feature_tys: Ty<'tcx>,
    /// Tuple containing the types of closure captures after the feature `capture_disjoint_fields`
    pub after_feature_tys: Ty<'tcx>,
}

impl TyCtxt<'_> {
    #[inline]
    pub fn opt_parent(self, id: DefId) -> Option<DefId> {
        self.def_key(id).parent.map(|index| DefId { index, ..id })
    }

    #[inline]
    #[track_caller]
    pub fn parent(self, id: DefId) -> DefId {
        match self.opt_parent(id) {
            Some(id) => id,
            // not `unwrap_or_else` to avoid breaking caller tracking
            None => bug!("{id:?} doesn't have a parent"),
        }
    }

    #[inline]
    #[track_caller]
    pub fn opt_local_parent(self, id: LocalDefId) -> Option<LocalDefId> {
        self.opt_parent(id.to_def_id()).map(DefId::expect_local)
    }

    #[inline]
    #[track_caller]
    pub fn local_parent(self, id: LocalDefId) -> LocalDefId {
        self.parent(id.to_def_id()).expect_local()
    }

    pub fn is_descendant_of(self, mut descendant: DefId, ancestor: DefId) -> bool {
        if descendant.krate != ancestor.krate {
            return false;
        }

        while descendant != ancestor {
            match self.opt_parent(descendant) {
                Some(parent) => descendant = parent,
                None => return false,
            }
        }
        true
    }
}

impl<Id> Visibility<Id> {
    pub fn is_public(self) -> bool {
        matches!(self, Visibility::Public)
    }

    pub fn map_id<OutId>(self, f: impl FnOnce(Id) -> OutId) -> Visibility<OutId> {
        match self {
            Visibility::Public => Visibility::Public,
            Visibility::Restricted(id) => Visibility::Restricted(f(id)),
        }
    }
}

impl<Id: Into<DefId>> Visibility<Id> {
    pub fn to_def_id(self) -> Visibility<DefId> {
        self.map_id(Into::into)
    }

    /// Returns `true` if an item with this visibility is accessible from the given module.
    pub fn is_accessible_from(self, module: impl Into<DefId>, tcx: TyCtxt<'_>) -> bool {
        match self {
            // Public items are visible everywhere.
            Visibility::Public => true,
            Visibility::Restricted(id) => tcx.is_descendant_of(module.into(), id.into()),
        }
    }

    /// Returns `true` if this visibility is at least as accessible as the given visibility
    pub fn is_at_least(self, vis: Visibility<impl Into<DefId>>, tcx: TyCtxt<'_>) -> bool {
        match vis {
            Visibility::Public => self.is_public(),
            Visibility::Restricted(id) => self.is_accessible_from(id, tcx),
        }
    }
}

impl Visibility<DefId> {
    pub fn expect_local(self) -> Visibility {
        self.map_id(|id| id.expect_local())
    }

    /// Returns `true` if this item is visible anywhere in the local crate.
    pub fn is_visible_locally(self) -> bool {
        match self {
            Visibility::Public => true,
            Visibility::Restricted(def_id) => def_id.is_local(),
        }
    }
}

/// The crate variances map is computed during typeck and contains the
/// variance of every item in the local crate. You should not use it
/// directly, because to do so will make your pass dependent on the
/// HIR of every item in the local crate. Instead, use
/// `tcx.variances_of()` to get the variance for a *particular*
/// item.
#[derive(HashStable, Debug)]
pub struct CrateVariancesMap<'tcx> {
    /// For each item with generics, maps to a vector of the variance
    /// of its generics. If an item has no generics, it will have no
    /// entry.
    pub variances: DefIdMap<&'tcx [ty::Variance]>,
}

// Contains information needed to resolve types and (in the future) look up
// the types of AST nodes.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct CReaderCacheKey {
    pub cnum: Option<CrateNum>,
    pub pos: usize,
}

/// Use this rather than `TyKind`, whenever possible.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, HashStable)]
#[rustc_diagnostic_item = "Ty"]
#[rustc_pass_by_value]
pub struct Ty<'tcx>(Interned<'tcx, WithCachedTypeInfo<TyKind<'tcx>>>);

impl ty::EarlyBoundRegion {
    /// Does this early bound region have a name? Early bound regions normally
    /// always have names except when using anonymous lifetimes (`'_`).
    pub fn has_name(&self) -> bool {
        self.name != kw::UnderscoreLifetime && self.name != kw::Empty
    }
}

/// Use this rather than `PredicateKind`, whenever possible.
#[derive(Clone, Copy, PartialEq, Eq, Hash, HashStable)]
#[rustc_pass_by_value]
pub struct Predicate<'tcx>(
    Interned<'tcx, WithCachedTypeInfo<ty::Binder<'tcx, PredicateKind<'tcx>>>>,
);

impl<'tcx> Predicate<'tcx> {
    /// Gets the inner `Binder<'tcx, PredicateKind<'tcx>>`.
    #[inline]
    pub fn kind(self) -> Binder<'tcx, PredicateKind<'tcx>> {
        self.0.internee
    }

    #[inline(always)]
    pub fn flags(self) -> TypeFlags {
        self.0.flags
    }

    #[inline(always)]
    pub fn outer_exclusive_binder(self) -> DebruijnIndex {
        self.0.outer_exclusive_binder
    }

    /// Flips the polarity of a Predicate.
    ///
    /// Given `T: Trait` predicate it returns `T: !Trait` and given `T: !Trait` returns `T: Trait`.
    pub fn flip_polarity(self, tcx: TyCtxt<'tcx>) -> Option<Predicate<'tcx>> {
        let kind = self
            .kind()
            .map_bound(|kind| match kind {
                PredicateKind::Clause(Clause::Trait(TraitPredicate {
                    trait_ref,
                    constness,
                    polarity,
                })) => Some(PredicateKind::Clause(Clause::Trait(TraitPredicate {
                    trait_ref,
                    constness,
                    polarity: polarity.flip()?,
                }))),

                _ => None,
            })
            .transpose()?;

        Some(tcx.mk_predicate(kind))
    }

    pub fn without_const(mut self, tcx: TyCtxt<'tcx>) -> Self {
        if let PredicateKind::Clause(Clause::Trait(TraitPredicate { trait_ref, constness, polarity })) = self.kind().skip_binder()
            && constness != BoundConstness::NotConst
        {
            self = tcx.mk_predicate(self.kind().rebind(PredicateKind::Clause(Clause::Trait(TraitPredicate {
                trait_ref,
                constness: BoundConstness::NotConst,
                polarity,
            }))));
        }
        self
    }

    #[instrument(level = "debug", skip(tcx), ret)]
    pub fn is_coinductive(self, tcx: TyCtxt<'tcx>) -> bool {
        match self.kind().skip_binder() {
            ty::PredicateKind::Clause(ty::Clause::Trait(data)) => {
                tcx.trait_is_coinductive(data.def_id())
            }
            ty::PredicateKind::WellFormed(_) => true,
            _ => false,
        }
    }

    /// Whether this projection can be soundly normalized.
    ///
    /// Wf predicates must not be normalized, as normalization
    /// can remove required bounds which would cause us to
    /// unsoundly accept some programs. See #91068.
    #[inline]
    pub fn allow_normalization(self) -> bool {
        match self.kind().skip_binder() {
            PredicateKind::WellFormed(_) => false,
            PredicateKind::Clause(Clause::Trait(_))
            | PredicateKind::Clause(Clause::RegionOutlives(_))
            | PredicateKind::Clause(Clause::TypeOutlives(_))
            | PredicateKind::Clause(Clause::Projection(_))
            | PredicateKind::Clause(Clause::ConstArgHasType(..))
            | PredicateKind::AliasEq(..)
            | PredicateKind::ObjectSafe(_)
            | PredicateKind::ClosureKind(_, _, _)
            | PredicateKind::Subtype(_)
            | PredicateKind::Coerce(_)
            | PredicateKind::ConstEvaluatable(_)
            | PredicateKind::ConstEquate(_, _)
            | PredicateKind::Ambiguous
            | PredicateKind::TypeWellFormedFromEnv(_) => true,
        }
    }
}

impl rustc_errors::IntoDiagnosticArg for Predicate<'_> {
    fn into_diagnostic_arg(self) -> rustc_errors::DiagnosticArgValue<'static> {
        rustc_errors::DiagnosticArgValue::Str(std::borrow::Cow::Owned(self.to_string()))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
/// A clause is something that can appear in where bounds or be inferred
/// by implied bounds.
pub enum Clause<'tcx> {
    /// Corresponds to `where Foo: Bar<A, B, C>`. `Foo` here would be
    /// the `Self` type of the trait reference and `A`, `B`, and `C`
    /// would be the type parameters.
    Trait(TraitPredicate<'tcx>),

    /// `where 'a: 'b`
    RegionOutlives(RegionOutlivesPredicate<'tcx>),

    /// `where T: 'a`
    TypeOutlives(TypeOutlivesPredicate<'tcx>),

    /// `where <T as TraitRef>::Name == X`, approximately.
    /// See the `ProjectionPredicate` struct for details.
    Projection(ProjectionPredicate<'tcx>),

    /// Ensures that a const generic argument to a parameter `const N: u8`
    /// is of type `u8`.
    ConstArgHasType(Const<'tcx>, Ty<'tcx>),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub enum PredicateKind<'tcx> {
    /// Prove a clause
    Clause(Clause<'tcx>),

    /// No syntax: `T` well-formed.
    WellFormed(GenericArg<'tcx>),

    /// Trait must be object-safe.
    ObjectSafe(DefId),

    /// No direct syntax. May be thought of as `where T: FnFoo<...>`
    /// for some substitutions `...` and `T` being a closure type.
    /// Satisfied (or refuted) once we know the closure's kind.
    ClosureKind(DefId, SubstsRef<'tcx>, ClosureKind),

    /// `T1 <: T2`
    ///
    /// This obligation is created most often when we have two
    /// unresolved type variables and hence don't have enough
    /// information to process the subtyping obligation yet.
    Subtype(SubtypePredicate<'tcx>),

    /// `T1` coerced to `T2`
    ///
    /// Like a subtyping obligation, this is created most often
    /// when we have two unresolved type variables and hence
    /// don't have enough information to process the coercion
    /// obligation yet. At the moment, we actually process coercions
    /// very much like subtyping and don't handle the full coercion
    /// logic.
    Coerce(CoercePredicate<'tcx>),

    /// Constant initializer must evaluate successfully.
    ConstEvaluatable(ty::Const<'tcx>),

    /// Constants must be equal. The first component is the const that is expected.
    ConstEquate(Const<'tcx>, Const<'tcx>),

    /// Represents a type found in the environment that we can use for implied bounds.
    ///
    /// Only used for Chalk.
    TypeWellFormedFromEnv(Ty<'tcx>),

    /// A marker predicate that is always ambiguous.
    /// Used for coherence to mark opaque types as possibly equal to each other but ambiguous.
    Ambiguous,

    /// Separate from `Clause::Projection` which is used for normalization in new solver.
    /// This predicate requires two terms to be equal to eachother.
    ///
    /// Only used for new solver
    AliasEq(Term<'tcx>, Term<'tcx>),
}

/// The crate outlives map is computed during typeck and contains the
/// outlives of every item in the local crate. You should not use it
/// directly, because to do so will make your pass dependent on the
/// HIR of every item in the local crate. Instead, use
/// `tcx.inferred_outlives_of()` to get the outlives for a *particular*
/// item.
#[derive(HashStable, Debug)]
pub struct CratePredicatesMap<'tcx> {
    /// For each struct with outlive bounds, maps to a vector of the
    /// predicate of its outlive bounds. If an item has no outlives
    /// bounds, it will have no entry.
    pub predicates: FxHashMap<DefId, &'tcx [(Clause<'tcx>, Span)]>,
}

impl<'tcx> Predicate<'tcx> {
    /// Performs a substitution suitable for going from a
    /// poly-trait-ref to supertraits that must hold if that
    /// poly-trait-ref holds. This is slightly different from a normal
    /// substitution in terms of what happens with bound regions. See
    /// lengthy comment below for details.
    pub fn subst_supertrait(
        self,
        tcx: TyCtxt<'tcx>,
        trait_ref: &ty::PolyTraitRef<'tcx>,
    ) -> Predicate<'tcx> {
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
        // holds. So if there is an impl of `T:Foo<'a>` that applies to
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
        // `T:Foo1<'x>` must show that `for<'b> T: Bar1<'x, 'b>`. So
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
        // - The supertrait-ref is `for<'b> Bar1<'a,'b>`, where `'a` is an
        //   early-bound parameter and `'b` is a late-bound parameter with a
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

        // Working through the second example:
        // trait_ref: for<'x> T: Foo1<'^0.0>; substs: [T, '^0.0]
        // predicate: for<'b> Self: Bar1<'a, '^0.0>; substs: [Self, 'a, '^0.0]
        // We want to end up with:
        //     for<'x, 'b> T: Bar1<'^0.0, '^0.1>
        // To do this:
        // 1) We must shift all bound vars in predicate by the length
        //    of trait ref's bound vars. So, we would end up with predicate like
        //    Self: Bar1<'a, '^0.1>
        // 2) We can then apply the trait substs to this, ending up with
        //    T: Bar1<'^0.0, '^0.1>
        // 3) Finally, to create the final bound vars, we concatenate the bound
        //    vars of the trait ref with those of the predicate:
        //    ['x, 'b]
        let bound_pred = self.kind();
        let pred_bound_vars = bound_pred.bound_vars();
        let trait_bound_vars = trait_ref.bound_vars();
        // 1) Self: Bar1<'a, '^0.0> -> Self: Bar1<'a, '^0.1>
        let shifted_pred =
            tcx.shift_bound_var_indices(trait_bound_vars.len(), bound_pred.skip_binder());
        // 2) Self: Bar1<'a, '^0.1> -> T: Bar1<'^0.0, '^0.1>
        let new = EarlyBinder(shifted_pred).subst(tcx, trait_ref.skip_binder().substs);
        // 3) ['x] + ['b] -> ['x, 'b]
        let bound_vars =
            tcx.mk_bound_variable_kinds_from_iter(trait_bound_vars.iter().chain(pred_bound_vars));
        tcx.reuse_or_mk_predicate(self, ty::Binder::bind_with_vars(new, bound_vars))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct TraitPredicate<'tcx> {
    pub trait_ref: TraitRef<'tcx>,

    pub constness: BoundConstness,

    /// If polarity is Positive: we are proving that the trait is implemented.
    ///
    /// If polarity is Negative: we are proving that a negative impl of this trait
    /// exists. (Note that coherence also checks whether negative impls of supertraits
    /// exist via a series of predicates.)
    ///
    /// If polarity is Reserved: that's a bug.
    pub polarity: ImplPolarity,
}

pub type PolyTraitPredicate<'tcx> = ty::Binder<'tcx, TraitPredicate<'tcx>>;

impl<'tcx> TraitPredicate<'tcx> {
    pub fn remap_constness(&mut self, param_env: &mut ParamEnv<'tcx>) {
        *param_env = param_env.with_constness(self.constness.and(param_env.constness()))
    }

    /// Remap the constness of this predicate before emitting it for diagnostics.
    pub fn remap_constness_diag(&mut self, param_env: ParamEnv<'tcx>) {
        // this is different to `remap_constness` that callees want to print this predicate
        // in case of selection errors. `T: ~const Drop` bounds cannot end up here when the
        // param_env is not const because it is always satisfied in non-const contexts.
        if let hir::Constness::NotConst = param_env.constness() {
            self.constness = ty::BoundConstness::NotConst;
        }
    }

    pub fn with_self_ty(self, tcx: TyCtxt<'tcx>, self_ty: Ty<'tcx>) -> Self {
        Self { trait_ref: self.trait_ref.with_self_ty(tcx, self_ty), ..self }
    }

    pub fn def_id(self) -> DefId {
        self.trait_ref.def_id
    }

    pub fn self_ty(self) -> Ty<'tcx> {
        self.trait_ref.self_ty()
    }

    #[inline]
    pub fn is_const_if_const(self) -> bool {
        self.constness == BoundConstness::ConstIfConst
    }

    pub fn is_constness_satisfied_by(self, constness: hir::Constness) -> bool {
        match (self.constness, constness) {
            (BoundConstness::NotConst, _)
            | (BoundConstness::ConstIfConst, hir::Constness::Const) => true,
            (BoundConstness::ConstIfConst, hir::Constness::NotConst) => false,
        }
    }

    pub fn without_const(mut self) -> Self {
        self.constness = BoundConstness::NotConst;
        self
    }
}

impl<'tcx> PolyTraitPredicate<'tcx> {
    pub fn def_id(self) -> DefId {
        // Ok to skip binder since trait `DefId` does not care about regions.
        self.skip_binder().def_id()
    }

    pub fn self_ty(self) -> ty::Binder<'tcx, Ty<'tcx>> {
        self.map_bound(|trait_ref| trait_ref.self_ty())
    }

    /// Remap the constness of this predicate before emitting it for diagnostics.
    pub fn remap_constness_diag(&mut self, param_env: ParamEnv<'tcx>) {
        *self = self.map_bound(|mut p| {
            p.remap_constness_diag(param_env);
            p
        });
    }

    #[inline]
    pub fn is_const_if_const(self) -> bool {
        self.skip_binder().is_const_if_const()
    }
}

/// `A: B`
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct OutlivesPredicate<A, B>(pub A, pub B);
pub type RegionOutlivesPredicate<'tcx> = OutlivesPredicate<ty::Region<'tcx>, ty::Region<'tcx>>;
pub type TypeOutlivesPredicate<'tcx> = OutlivesPredicate<Ty<'tcx>, ty::Region<'tcx>>;
pub type PolyRegionOutlivesPredicate<'tcx> = ty::Binder<'tcx, RegionOutlivesPredicate<'tcx>>;
pub type PolyTypeOutlivesPredicate<'tcx> = ty::Binder<'tcx, TypeOutlivesPredicate<'tcx>>;

/// Encodes that `a` must be a subtype of `b`. The `a_is_expected` flag indicates
/// whether the `a` type is the type that we should label as "expected" when
/// presenting user diagnostics.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct SubtypePredicate<'tcx> {
    pub a_is_expected: bool,
    pub a: Ty<'tcx>,
    pub b: Ty<'tcx>,
}
pub type PolySubtypePredicate<'tcx> = ty::Binder<'tcx, SubtypePredicate<'tcx>>;

/// Encodes that we have to coerce *from* the `a` type to the `b` type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct CoercePredicate<'tcx> {
    pub a: Ty<'tcx>,
    pub b: Ty<'tcx>,
}
pub type PolyCoercePredicate<'tcx> = ty::Binder<'tcx, CoercePredicate<'tcx>>;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Term<'tcx> {
    ptr: NonZeroUsize,
    marker: PhantomData<(Ty<'tcx>, Const<'tcx>)>,
}

impl Debug for Term<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = if let Some(ty) = self.ty() {
            format!("Term::Ty({:?})", ty)
        } else if let Some(ct) = self.ct() {
            format!("Term::Ct({:?})", ct)
        } else {
            unreachable!()
        };
        f.write_str(&data)
    }
}

impl<'tcx> From<Ty<'tcx>> for Term<'tcx> {
    fn from(ty: Ty<'tcx>) -> Self {
        TermKind::Ty(ty).pack()
    }
}

impl<'tcx> From<Const<'tcx>> for Term<'tcx> {
    fn from(c: Const<'tcx>) -> Self {
        TermKind::Const(c).pack()
    }
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for Term<'tcx> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        self.unpack().hash_stable(hcx, hasher);
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for Term<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(self.unpack().try_fold_with(folder)?.pack())
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for Term<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.unpack().visit_with(visitor)
    }
}

impl<'tcx, E: TyEncoder<I = TyCtxt<'tcx>>> Encodable<E> for Term<'tcx> {
    fn encode(&self, e: &mut E) {
        self.unpack().encode(e)
    }
}

impl<'tcx, D: TyDecoder<I = TyCtxt<'tcx>>> Decodable<D> for Term<'tcx> {
    fn decode(d: &mut D) -> Self {
        let res: TermKind<'tcx> = Decodable::decode(d);
        res.pack()
    }
}

impl<'tcx> Term<'tcx> {
    #[inline]
    pub fn unpack(self) -> TermKind<'tcx> {
        let ptr = self.ptr.get();
        // SAFETY: use of `Interned::new_unchecked` here is ok because these
        // pointers were originally created from `Interned` types in `pack()`,
        // and this is just going in the other direction.
        unsafe {
            match ptr & TAG_MASK {
                TYPE_TAG => TermKind::Ty(Ty(Interned::new_unchecked(
                    &*((ptr & !TAG_MASK) as *const WithCachedTypeInfo<ty::TyKind<'tcx>>),
                ))),
                CONST_TAG => TermKind::Const(ty::Const(Interned::new_unchecked(
                    &*((ptr & !TAG_MASK) as *const ty::ConstData<'tcx>),
                ))),
                _ => core::intrinsics::unreachable(),
            }
        }
    }

    pub fn ty(&self) -> Option<Ty<'tcx>> {
        if let TermKind::Ty(ty) = self.unpack() { Some(ty) } else { None }
    }

    pub fn ct(&self) -> Option<Const<'tcx>> {
        if let TermKind::Const(c) = self.unpack() { Some(c) } else { None }
    }

    pub fn into_arg(self) -> GenericArg<'tcx> {
        match self.unpack() {
            TermKind::Ty(ty) => ty.into(),
            TermKind::Const(c) => c.into(),
        }
    }

    /// This function returns `None` for `AliasKind::Opaque`.
    ///
    /// FIXME: rename `AliasTy` to `AliasTerm` and make sure we correctly
    /// deal with constants.
    pub fn to_alias_term_no_opaque(&self, tcx: TyCtxt<'tcx>) -> Option<AliasTy<'tcx>> {
        match self.unpack() {
            TermKind::Ty(ty) => match ty.kind() {
                ty::Alias(kind, alias_ty) => match kind {
                    AliasKind::Projection => Some(*alias_ty),
                    AliasKind::Opaque => None,
                },
                _ => None,
            },
            TermKind::Const(ct) => match ct.kind() {
                ConstKind::Unevaluated(uv) => Some(tcx.mk_alias_ty(uv.def.did, uv.substs)),
                _ => None,
            },
        }
    }

    pub fn is_infer(&self) -> bool {
        match self.unpack() {
            TermKind::Ty(ty) => ty.is_ty_var(),
            TermKind::Const(ct) => ct.is_ct_infer(),
        }
    }
}

const TAG_MASK: usize = 0b11;
const TYPE_TAG: usize = 0b00;
const CONST_TAG: usize = 0b01;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable)]
pub enum TermKind<'tcx> {
    Ty(Ty<'tcx>),
    Const(Const<'tcx>),
}

impl<'tcx> TermKind<'tcx> {
    #[inline]
    fn pack(self) -> Term<'tcx> {
        let (tag, ptr) = match self {
            TermKind::Ty(ty) => {
                // Ensure we can use the tag bits.
                assert_eq!(mem::align_of_val(&*ty.0.0) & TAG_MASK, 0);
                (TYPE_TAG, ty.0.0 as *const WithCachedTypeInfo<ty::TyKind<'tcx>> as usize)
            }
            TermKind::Const(ct) => {
                // Ensure we can use the tag bits.
                assert_eq!(mem::align_of_val(&*ct.0.0) & TAG_MASK, 0);
                (CONST_TAG, ct.0.0 as *const ty::ConstData<'tcx> as usize)
            }
        };

        Term { ptr: unsafe { NonZeroUsize::new_unchecked(ptr | tag) }, marker: PhantomData }
    }
}

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
#[derive(Copy, Clone, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct ProjectionPredicate<'tcx> {
    pub projection_ty: AliasTy<'tcx>,
    pub term: Term<'tcx>,
}

impl<'tcx> ProjectionPredicate<'tcx> {
    pub fn self_ty(self) -> Ty<'tcx> {
        self.projection_ty.self_ty()
    }

    pub fn with_self_ty(self, tcx: TyCtxt<'tcx>, self_ty: Ty<'tcx>) -> ProjectionPredicate<'tcx> {
        Self { projection_ty: self.projection_ty.with_self_ty(tcx, self_ty), ..self }
    }

    pub fn trait_def_id(self, tcx: TyCtxt<'tcx>) -> DefId {
        self.projection_ty.trait_def_id(tcx)
    }

    pub fn def_id(self) -> DefId {
        self.projection_ty.def_id
    }
}

pub type PolyProjectionPredicate<'tcx> = Binder<'tcx, ProjectionPredicate<'tcx>>;

impl<'tcx> PolyProjectionPredicate<'tcx> {
    /// Returns the `DefId` of the trait of the associated item being projected.
    #[inline]
    pub fn trait_def_id(&self, tcx: TyCtxt<'tcx>) -> DefId {
        self.skip_binder().projection_ty.trait_def_id(tcx)
    }

    /// Get the [PolyTraitRef] required for this projection to be well formed.
    /// Note that for generic associated types the predicates of the associated
    /// type also need to be checked.
    #[inline]
    pub fn required_poly_trait_ref(&self, tcx: TyCtxt<'tcx>) -> PolyTraitRef<'tcx> {
        // Note: unlike with `TraitRef::to_poly_trait_ref()`,
        // `self.0.trait_ref` is permitted to have escaping regions.
        // This is because here `self` has a `Binder` and so does our
        // return value, so we are preserving the number of binding
        // levels.
        self.map_bound(|predicate| predicate.projection_ty.trait_ref(tcx))
    }

    pub fn term(&self) -> Binder<'tcx, Term<'tcx>> {
        self.map_bound(|predicate| predicate.term)
    }

    /// The `DefId` of the `TraitItem` for the associated type.
    ///
    /// Note that this is not the `DefId` of the `TraitRef` containing this
    /// associated type, which is in `tcx.associated_item(projection_def_id()).container`.
    pub fn projection_def_id(&self) -> DefId {
        // Ok to skip binder since trait `DefId` does not care about regions.
        self.skip_binder().projection_ty.def_id
    }
}

pub trait ToPolyTraitRef<'tcx> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx>;
}

impl<'tcx> ToPolyTraitRef<'tcx> for PolyTraitPredicate<'tcx> {
    fn to_poly_trait_ref(&self) -> PolyTraitRef<'tcx> {
        self.map_bound_ref(|trait_pred| trait_pred.trait_ref)
    }
}

pub trait ToPredicate<'tcx, P = Predicate<'tcx>> {
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> P;
}

impl<'tcx, T> ToPredicate<'tcx, T> for T {
    fn to_predicate(self, _tcx: TyCtxt<'tcx>) -> T {
        self
    }
}

impl<'tcx> ToPredicate<'tcx> for Binder<'tcx, PredicateKind<'tcx>> {
    #[inline(always)]
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        tcx.mk_predicate(self)
    }
}

impl<'tcx> ToPredicate<'tcx> for Clause<'tcx> {
    #[inline(always)]
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        tcx.mk_predicate(ty::Binder::dummy(ty::PredicateKind::Clause(self)))
    }
}

impl<'tcx> ToPredicate<'tcx> for Binder<'tcx, TraitRef<'tcx>> {
    #[inline(always)]
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        let pred: PolyTraitPredicate<'tcx> = self.to_predicate(tcx);
        pred.to_predicate(tcx)
    }
}

impl<'tcx> ToPredicate<'tcx, PolyTraitPredicate<'tcx>> for Binder<'tcx, TraitRef<'tcx>> {
    #[inline(always)]
    fn to_predicate(self, _: TyCtxt<'tcx>) -> PolyTraitPredicate<'tcx> {
        self.map_bound(|trait_ref| TraitPredicate {
            trait_ref,
            constness: ty::BoundConstness::NotConst,
            polarity: ty::ImplPolarity::Positive,
        })
    }
}

impl<'tcx> ToPredicate<'tcx> for PolyTraitPredicate<'tcx> {
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        self.map_bound(|p| PredicateKind::Clause(Clause::Trait(p))).to_predicate(tcx)
    }
}

impl<'tcx> ToPredicate<'tcx> for PolyRegionOutlivesPredicate<'tcx> {
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        self.map_bound(|p| PredicateKind::Clause(Clause::RegionOutlives(p))).to_predicate(tcx)
    }
}

impl<'tcx> ToPredicate<'tcx> for PolyTypeOutlivesPredicate<'tcx> {
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        self.map_bound(|p| PredicateKind::Clause(Clause::TypeOutlives(p))).to_predicate(tcx)
    }
}

impl<'tcx> ToPredicate<'tcx> for PolyProjectionPredicate<'tcx> {
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        self.map_bound(|p| PredicateKind::Clause(Clause::Projection(p))).to_predicate(tcx)
    }
}

impl<'tcx> Predicate<'tcx> {
    pub fn to_opt_poly_trait_pred(self) -> Option<PolyTraitPredicate<'tcx>> {
        let predicate = self.kind();
        match predicate.skip_binder() {
            PredicateKind::Clause(Clause::Trait(t)) => Some(predicate.rebind(t)),
            PredicateKind::Clause(Clause::Projection(..))
            | PredicateKind::Clause(Clause::ConstArgHasType(..))
            | PredicateKind::AliasEq(..)
            | PredicateKind::Subtype(..)
            | PredicateKind::Coerce(..)
            | PredicateKind::Clause(Clause::RegionOutlives(..))
            | PredicateKind::WellFormed(..)
            | PredicateKind::ObjectSafe(..)
            | PredicateKind::ClosureKind(..)
            | PredicateKind::Clause(Clause::TypeOutlives(..))
            | PredicateKind::ConstEvaluatable(..)
            | PredicateKind::ConstEquate(..)
            | PredicateKind::Ambiguous
            | PredicateKind::TypeWellFormedFromEnv(..) => None,
        }
    }

    pub fn to_opt_poly_projection_pred(self) -> Option<PolyProjectionPredicate<'tcx>> {
        let predicate = self.kind();
        match predicate.skip_binder() {
            PredicateKind::Clause(Clause::Projection(t)) => Some(predicate.rebind(t)),
            PredicateKind::Clause(Clause::Trait(..))
            | PredicateKind::Clause(Clause::ConstArgHasType(..))
            | PredicateKind::AliasEq(..)
            | PredicateKind::Subtype(..)
            | PredicateKind::Coerce(..)
            | PredicateKind::Clause(Clause::RegionOutlives(..))
            | PredicateKind::WellFormed(..)
            | PredicateKind::ObjectSafe(..)
            | PredicateKind::ClosureKind(..)
            | PredicateKind::Clause(Clause::TypeOutlives(..))
            | PredicateKind::ConstEvaluatable(..)
            | PredicateKind::ConstEquate(..)
            | PredicateKind::Ambiguous
            | PredicateKind::TypeWellFormedFromEnv(..) => None,
        }
    }

    pub fn to_opt_type_outlives(self) -> Option<PolyTypeOutlivesPredicate<'tcx>> {
        let predicate = self.kind();
        match predicate.skip_binder() {
            PredicateKind::Clause(Clause::TypeOutlives(data)) => Some(predicate.rebind(data)),
            PredicateKind::Clause(Clause::Trait(..))
            | PredicateKind::Clause(Clause::ConstArgHasType(..))
            | PredicateKind::Clause(Clause::Projection(..))
            | PredicateKind::AliasEq(..)
            | PredicateKind::Subtype(..)
            | PredicateKind::Coerce(..)
            | PredicateKind::Clause(Clause::RegionOutlives(..))
            | PredicateKind::WellFormed(..)
            | PredicateKind::ObjectSafe(..)
            | PredicateKind::ClosureKind(..)
            | PredicateKind::ConstEvaluatable(..)
            | PredicateKind::ConstEquate(..)
            | PredicateKind::Ambiguous
            | PredicateKind::TypeWellFormedFromEnv(..) => None,
        }
    }
}

/// Represents the bounds declared on a particular set of type
/// parameters. Should eventually be generalized into a flag list of
/// where-clauses. You can obtain an `InstantiatedPredicates` list from a
/// `GenericPredicates` by using the `instantiate` method. Note that this method
/// reflects an important semantic invariant of `InstantiatedPredicates`: while
/// the `GenericPredicates` are expressed in terms of the bound type
/// parameters of the impl/trait/whatever, an `InstantiatedPredicates` instance
/// represented a set of bounds for some particular instantiation,
/// meaning that the generic parameters have been substituted with
/// their values.
///
/// Example:
/// ```ignore (illustrative)
/// struct Foo<T, U: Bar<T>> { ... }
/// ```
/// Here, the `GenericPredicates` for `Foo` would contain a list of bounds like
/// `[[], [U:Bar<T>]]`. Now if there were some particular reference
/// like `Foo<isize,usize>`, then the `InstantiatedPredicates` would be `[[],
/// [usize:Bar<isize>]]`.
#[derive(Clone, Debug, TypeFoldable, TypeVisitable)]
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

    pub fn iter(&self) -> <&Self as IntoIterator>::IntoIter {
        (&self).into_iter()
    }
}

impl<'tcx> IntoIterator for InstantiatedPredicates<'tcx> {
    type Item = (Predicate<'tcx>, Span);

    type IntoIter = std::iter::Zip<std::vec::IntoIter<Predicate<'tcx>>, std::vec::IntoIter<Span>>;

    fn into_iter(self) -> Self::IntoIter {
        debug_assert_eq!(self.predicates.len(), self.spans.len());
        std::iter::zip(self.predicates, self.spans)
    }
}

impl<'a, 'tcx> IntoIterator for &'a InstantiatedPredicates<'tcx> {
    type Item = (Predicate<'tcx>, Span);

    type IntoIter = std::iter::Zip<
        std::iter::Copied<std::slice::Iter<'a, Predicate<'tcx>>>,
        std::iter::Copied<std::slice::Iter<'a, Span>>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        debug_assert_eq!(self.predicates.len(), self.spans.len());
        std::iter::zip(self.predicates.iter().copied(), self.spans.iter().copied())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, HashStable, TyEncodable, TyDecodable, Lift)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct OpaqueTypeKey<'tcx> {
    pub def_id: LocalDefId,
    pub substs: SubstsRef<'tcx>,
}

#[derive(Copy, Clone, Debug, TypeFoldable, TypeVisitable, HashStable, TyEncodable, TyDecodable)]
pub struct OpaqueHiddenType<'tcx> {
    /// The span of this particular definition of the opaque type. So
    /// for example:
    ///
    /// ```ignore (incomplete snippet)
    /// type Foo = impl Baz;
    /// fn bar() -> Foo {
    /// //          ^^^ This is the span we are looking for!
    /// }
    /// ```
    ///
    /// In cases where the fn returns `(impl Trait, impl Trait)` or
    /// other such combinations, the result is currently
    /// over-approximated, but better than nothing.
    pub span: Span,

    /// The type variable that represents the value of the opaque type
    /// that we require. In other words, after we compile this function,
    /// we will be created a constraint like:
    /// ```ignore (pseudo-rust)
    /// Foo<'a, T> = ?C
    /// ```
    /// where `?C` is the value of this type variable. =) It may
    /// naturally refer to the type and lifetime parameters in scope
    /// in this function, though ultimately it should only reference
    /// those that are arguments to `Foo` in the constraint above. (In
    /// other words, `?C` should not include `'b`, even though it's a
    /// lifetime parameter on `foo`.)
    pub ty: Ty<'tcx>,
}

impl<'tcx> OpaqueHiddenType<'tcx> {
    pub fn report_mismatch(&self, other: &Self, tcx: TyCtxt<'tcx>) -> ErrorGuaranteed {
        // Found different concrete types for the opaque type.
        let sub_diag = if self.span == other.span {
            TypeMismatchReason::ConflictType { span: self.span }
        } else {
            TypeMismatchReason::PreviousUse { span: self.span }
        };
        tcx.sess.emit_err(OpaqueHiddenTypeMismatch {
            self_ty: self.ty,
            other_ty: other.ty,
            other_span: other.span,
            sub: sub_diag,
        })
    }

    #[instrument(level = "debug", skip(tcx), ret)]
    pub fn remap_generic_params_to_declaration_params(
        self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        tcx: TyCtxt<'tcx>,
        // typeck errors have subpar spans for opaque types, so delay error reporting until borrowck.
        ignore_errors: bool,
    ) -> Self {
        let OpaqueTypeKey { def_id, substs } = opaque_type_key;

        // Use substs to build up a reverse map from regions to their
        // identity mappings. This is necessary because of `impl
        // Trait` lifetimes are computed by replacing existing
        // lifetimes with 'static and remapping only those used in the
        // `impl Trait` return type, resulting in the parameters
        // shifting.
        let id_substs = InternalSubsts::identity_for_item(tcx, def_id.to_def_id());
        debug!(?id_substs);

        // This zip may have several times the same lifetime in `substs` paired with a different
        // lifetime from `id_substs`. Simply `collect`ing the iterator is the correct behaviour:
        // it will pick the last one, which is the one we introduced in the impl-trait desugaring.
        let map = substs.iter().zip(id_substs).collect();
        debug!("map = {:#?}", map);

        // Convert the type from the function into a type valid outside
        // the function, by replacing invalid regions with 'static,
        // after producing an error for each of them.
        self.fold_with(&mut opaque_types::ReverseMapper::new(tcx, map, self.span, ignore_errors))
    }
}

/// The "placeholder index" fully defines a placeholder region, type, or const. Placeholders are
/// identified by both a universe, as well as a name residing within that universe. Distinct bound
/// regions/types/consts within the same universe simply have an unknown relationship to one
/// another.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[derive(HashStable, TyEncodable, TyDecodable)]
pub struct Placeholder<T> {
    pub universe: UniverseIndex,
    pub name: T,
}

pub type PlaceholderRegion = Placeholder<BoundRegionKind>;

pub type PlaceholderType = Placeholder<BoundTyKind>;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, HashStable)]
#[derive(TyEncodable, TyDecodable, PartialOrd, Ord)]
pub struct BoundConst<'tcx> {
    pub var: BoundVar,
    pub ty: Ty<'tcx>,
}

pub type PlaceholderConst<'tcx> = Placeholder<BoundVar>;

/// A `DefId` which, in case it is a const argument, is potentially bundled with
/// the `DefId` of the generic parameter it instantiates.
///
/// This is used to avoid calls to `type_of` for const arguments during typeck
/// which cause cycle errors.
///
/// ```rust
/// struct A;
/// impl A {
///     fn foo<const N: usize>(&self) -> [u8; N] { [0; N] }
///     //           ^ const parameter
/// }
/// struct B;
/// impl B {
///     fn foo<const M: u8>(&self) -> usize { 42 }
///     //           ^ const parameter
/// }
///
/// fn main() {
///     let a = A;
///     let _b = a.foo::<{ 3 + 7 }>();
///     //               ^^^^^^^^^ const argument
/// }
/// ```
///
/// Let's look at the call `a.foo::<{ 3 + 7 }>()` here. We do not know
/// which `foo` is used until we know the type of `a`.
///
/// We only know the type of `a` once we are inside of `typeck(main)`.
/// We also end up normalizing the type of `_b` during `typeck(main)` which
/// requires us to evaluate the const argument.
///
/// To evaluate that const argument we need to know its type,
/// which we would get using `type_of(const_arg)`. This requires us to
/// resolve `foo` as it can be either `usize` or `u8` in this example.
/// However, resolving `foo` once again requires `typeck(main)` to get the type of `a`,
/// which results in a cycle.
///
/// In short we must not call `type_of(const_arg)` during `typeck(main)`.
///
/// When first creating the `ty::Const` of the const argument inside of `typeck` we have
/// already resolved `foo` so we know which const parameter this argument instantiates.
/// This means that we also know the expected result of `type_of(const_arg)` even if we
/// aren't allowed to call that query: it is equal to `type_of(const_param)` which is
/// trivial to compute.
///
/// If we now want to use that constant in a place which potentially needs its type
/// we also pass the type of its `const_param`. This is the point of `WithOptConstParam`,
/// except that instead of a `Ty` we bundle the `DefId` of the const parameter.
/// Meaning that we need to use `type_of(const_param_did)` if `const_param_did` is `Some`
/// to get the type of `did`.
#[derive(Copy, Clone, Debug, TypeFoldable, TypeVisitable, Lift, TyEncodable, TyDecodable)]
#[derive(PartialEq, Eq, PartialOrd, Ord)]
#[derive(Hash, HashStable)]
pub struct WithOptConstParam<T> {
    pub did: T,
    /// The `DefId` of the corresponding generic parameter in case `did` is
    /// a const argument.
    ///
    /// Note that even if `did` is a const argument, this may still be `None`.
    /// All queries taking `WithOptConstParam` start by calling `tcx.opt_const_param_of(def.did)`
    /// to potentially update `param_did` in the case it is `None`.
    pub const_param_did: Option<DefId>,
}

impl<T> WithOptConstParam<T> {
    /// Creates a new `WithOptConstParam` setting `const_param_did` to `None`.
    #[inline(always)]
    pub fn unknown(did: T) -> WithOptConstParam<T> {
        WithOptConstParam { did, const_param_did: None }
    }
}

impl WithOptConstParam<LocalDefId> {
    /// Returns `Some((did, param_did))` if `def_id` is a const argument,
    /// `None` otherwise.
    #[inline(always)]
    pub fn try_lookup(did: LocalDefId, tcx: TyCtxt<'_>) -> Option<(LocalDefId, DefId)> {
        tcx.opt_const_param_of(did).map(|param_did| (did, param_did))
    }

    /// In case `self` is unknown but `self.did` is a const argument, this returns
    /// a `WithOptConstParam` with the correct `const_param_did`.
    #[inline(always)]
    pub fn try_upgrade(self, tcx: TyCtxt<'_>) -> Option<WithOptConstParam<LocalDefId>> {
        if self.const_param_did.is_none() {
            if let const_param_did @ Some(_) = tcx.opt_const_param_of(self.did) {
                return Some(WithOptConstParam { did: self.did, const_param_did });
            }
        }

        None
    }

    pub fn to_global(self) -> WithOptConstParam<DefId> {
        WithOptConstParam { did: self.did.to_def_id(), const_param_did: self.const_param_did }
    }

    pub fn def_id_for_type_of(self) -> DefId {
        if let Some(did) = self.const_param_did { did } else { self.did.to_def_id() }
    }
}

impl WithOptConstParam<DefId> {
    pub fn as_local(self) -> Option<WithOptConstParam<LocalDefId>> {
        self.did
            .as_local()
            .map(|did| WithOptConstParam { did, const_param_did: self.const_param_did })
    }

    pub fn as_const_arg(self) -> Option<(LocalDefId, DefId)> {
        if let Some(param_did) = self.const_param_did {
            if let Some(did) = self.did.as_local() {
                return Some((did, param_did));
            }
        }

        None
    }

    pub fn is_local(self) -> bool {
        self.did.is_local()
    }

    pub fn def_id_for_type_of(self) -> DefId {
        self.const_param_did.unwrap_or(self.did)
    }
}

/// When type checking, we use the `ParamEnv` to track
/// details about the set of where-clauses that are in scope at this
/// particular point.
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct ParamEnv<'tcx> {
    /// This packs both caller bounds and the reveal enum into one pointer.
    ///
    /// Caller bounds are `Obligation`s that the caller must satisfy. This is
    /// basically the set of bounds on the in-scope type parameters, translated
    /// into `Obligation`s, and elaborated and normalized.
    ///
    /// Use the `caller_bounds()` method to access.
    ///
    /// Typically, this is `Reveal::UserFacing`, but during codegen we
    /// want `Reveal::All`.
    ///
    /// Note: This is packed, use the reveal() method to access it.
    packed: CopyTaggedPtr<&'tcx List<Predicate<'tcx>>, ParamTag, true>,
}

#[derive(Copy, Clone)]
struct ParamTag {
    reveal: traits::Reveal,
    constness: hir::Constness,
}

unsafe impl rustc_data_structures::tagged_ptr::Tag for ParamTag {
    const BITS: usize = 2;
    #[inline]
    fn into_usize(self) -> usize {
        match self {
            Self { reveal: traits::Reveal::UserFacing, constness: hir::Constness::NotConst } => 0,
            Self { reveal: traits::Reveal::All, constness: hir::Constness::NotConst } => 1,
            Self { reveal: traits::Reveal::UserFacing, constness: hir::Constness::Const } => 2,
            Self { reveal: traits::Reveal::All, constness: hir::Constness::Const } => 3,
        }
    }
    #[inline]
    unsafe fn from_usize(ptr: usize) -> Self {
        match ptr {
            0 => Self { reveal: traits::Reveal::UserFacing, constness: hir::Constness::NotConst },
            1 => Self { reveal: traits::Reveal::All, constness: hir::Constness::NotConst },
            2 => Self { reveal: traits::Reveal::UserFacing, constness: hir::Constness::Const },
            3 => Self { reveal: traits::Reveal::All, constness: hir::Constness::Const },
            _ => std::hint::unreachable_unchecked(),
        }
    }
}

impl<'tcx> fmt::Debug for ParamEnv<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParamEnv")
            .field("caller_bounds", &self.caller_bounds())
            .field("reveal", &self.reveal())
            .field("constness", &self.constness())
            .finish()
    }
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for ParamEnv<'tcx> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        self.caller_bounds().hash_stable(hcx, hasher);
        self.reveal().hash_stable(hcx, hasher);
        self.constness().hash_stable(hcx, hasher);
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for ParamEnv<'tcx> {
    fn try_fold_with<F: ty::fold::FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(ParamEnv::new(
            self.caller_bounds().try_fold_with(folder)?,
            self.reveal().try_fold_with(folder)?,
            self.constness(),
        ))
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for ParamEnv<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.caller_bounds().visit_with(visitor)?;
        self.reveal().visit_with(visitor)
    }
}

impl<'tcx> ParamEnv<'tcx> {
    /// Construct a trait environment suitable for contexts where
    /// there are no where-clauses in scope. Hidden types (like `impl
    /// Trait`) are left hidden, so this is suitable for ordinary
    /// type-checking.
    #[inline]
    pub fn empty() -> Self {
        Self::new(List::empty(), Reveal::UserFacing, hir::Constness::NotConst)
    }

    #[inline]
    pub fn caller_bounds(self) -> &'tcx List<Predicate<'tcx>> {
        self.packed.pointer()
    }

    #[inline]
    pub fn reveal(self) -> traits::Reveal {
        self.packed.tag().reveal
    }

    #[inline]
    pub fn constness(self) -> hir::Constness {
        self.packed.tag().constness
    }

    #[inline]
    pub fn is_const(self) -> bool {
        self.packed.tag().constness == hir::Constness::Const
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
        Self::new(List::empty(), Reveal::All, hir::Constness::NotConst)
    }

    /// Construct a trait environment with the given set of predicates.
    #[inline]
    pub fn new(
        caller_bounds: &'tcx List<Predicate<'tcx>>,
        reveal: Reveal,
        constness: hir::Constness,
    ) -> Self {
        ty::ParamEnv { packed: CopyTaggedPtr::new(caller_bounds, ParamTag { reveal, constness }) }
    }

    pub fn with_user_facing(mut self) -> Self {
        self.packed.set_tag(ParamTag { reveal: Reveal::UserFacing, ..self.packed.tag() });
        self
    }

    #[inline]
    pub fn with_constness(mut self, constness: hir::Constness) -> Self {
        self.packed.set_tag(ParamTag { constness, ..self.packed.tag() });
        self
    }

    #[inline]
    pub fn with_const(mut self) -> Self {
        self.packed.set_tag(ParamTag { constness: hir::Constness::Const, ..self.packed.tag() });
        self
    }

    #[inline]
    pub fn without_const(mut self) -> Self {
        self.packed.set_tag(ParamTag { constness: hir::Constness::NotConst, ..self.packed.tag() });
        self
    }

    #[inline]
    pub fn remap_constness_with(&mut self, mut constness: ty::BoundConstness) {
        *self = self.with_constness(constness.and(self.constness()))
    }

    /// Returns a new parameter environment with the same clauses, but
    /// which "reveals" the true results of projections in all cases
    /// (even for associated types that are specializable). This is
    /// the desired behavior during codegen and certain other special
    /// contexts; normally though we want to use `Reveal::UserFacing`,
    /// which is the default.
    /// All opaque types in the caller_bounds of the `ParamEnv`
    /// will be normalized to their underlying types.
    /// See PR #65989 and issue #65918 for more details
    pub fn with_reveal_all_normalized(self, tcx: TyCtxt<'tcx>) -> Self {
        if self.packed.tag().reveal == traits::Reveal::All {
            return self;
        }

        ParamEnv::new(
            tcx.reveal_opaque_types_in_bounds(self.caller_bounds()),
            Reveal::All,
            self.constness(),
        )
    }

    /// Returns this same environment but with no caller bounds.
    #[inline]
    pub fn without_caller_bounds(self) -> Self {
        Self::new(List::empty(), self.reveal(), self.constness())
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
    pub fn and<T: TypeVisitable<TyCtxt<'tcx>>>(self, value: T) -> ParamEnvAnd<'tcx, T> {
        match self.reveal() {
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

// FIXME(ecstaticmorse): Audit all occurrences of `without_const().to_predicate(tcx)` to ensure that
// the constness of trait bounds is being propagated correctly.
impl<'tcx> PolyTraitRef<'tcx> {
    #[inline]
    pub fn with_constness(self, constness: BoundConstness) -> PolyTraitPredicate<'tcx> {
        self.map_bound(|trait_ref| ty::TraitPredicate {
            trait_ref,
            constness,
            polarity: ty::ImplPolarity::Positive,
        })
    }

    #[inline]
    pub fn without_const(self) -> PolyTraitPredicate<'tcx> {
        self.with_constness(BoundConstness::NotConst)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, TypeFoldable, TypeVisitable)]
#[derive(HashStable, Lift)]
pub struct ParamEnvAnd<'tcx, T> {
    pub param_env: ParamEnv<'tcx>,
    pub value: T,
}

impl<'tcx, T> ParamEnvAnd<'tcx, T> {
    pub fn into_parts(self) -> (ParamEnv<'tcx>, T) {
        (self.param_env, self.value)
    }

    #[inline]
    pub fn without_const(mut self) -> Self {
        self.param_env = self.param_env.without_const();
        self
    }
}

#[derive(Copy, Clone, Debug, HashStable, Encodable, Decodable)]
pub struct Destructor {
    /// The `DefId` of the destructor method
    pub did: DefId,
    /// The constness of the destructor method
    pub constness: hir::Constness,
}

bitflags! {
    #[derive(HashStable, TyEncodable, TyDecodable)]
    pub struct VariantFlags: u32 {
        const NO_VARIANT_FLAGS        = 0;
        /// Indicates whether the field list of this variant is `#[non_exhaustive]`.
        const IS_FIELD_LIST_NON_EXHAUSTIVE = 1 << 0;
        /// Indicates whether this variant was obtained as part of recovering from
        /// a syntactic error. May be incomplete or bogus.
        const IS_RECOVERED = 1 << 1;
    }
}

/// Definition of a variant -- a struct's fields or an enum variant.
#[derive(Debug, HashStable, TyEncodable, TyDecodable)]
pub struct VariantDef {
    /// `DefId` that identifies the variant itself.
    /// If this variant belongs to a struct or union, then this is a copy of its `DefId`.
    pub def_id: DefId,
    /// `DefId` that identifies the variant's constructor.
    /// If this variant is a struct variant, then this is `None`.
    pub ctor: Option<(CtorKind, DefId)>,
    /// Variant or struct name.
    pub name: Symbol,
    /// Discriminant of this variant.
    pub discr: VariantDiscr,
    /// Fields of this variant.
    pub fields: Vec<FieldDef>,
    /// Flags of the variant (e.g. is field list non-exhaustive)?
    flags: VariantFlags,
}

impl VariantDef {
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
        name: Symbol,
        variant_did: Option<DefId>,
        ctor: Option<(CtorKind, DefId)>,
        discr: VariantDiscr,
        fields: Vec<FieldDef>,
        adt_kind: AdtKind,
        parent_did: DefId,
        recovered: bool,
        is_field_list_non_exhaustive: bool,
    ) -> Self {
        debug!(
            "VariantDef::new(name = {:?}, variant_did = {:?}, ctor = {:?}, discr = {:?},
             fields = {:?}, adt_kind = {:?}, parent_did = {:?})",
            name, variant_did, ctor, discr, fields, adt_kind, parent_did,
        );

        let mut flags = VariantFlags::NO_VARIANT_FLAGS;
        if is_field_list_non_exhaustive {
            flags |= VariantFlags::IS_FIELD_LIST_NON_EXHAUSTIVE;
        }

        if recovered {
            flags |= VariantFlags::IS_RECOVERED;
        }

        VariantDef { def_id: variant_did.unwrap_or(parent_did), ctor, name, discr, fields, flags }
    }

    /// Is this field list non-exhaustive?
    #[inline]
    pub fn is_field_list_non_exhaustive(&self) -> bool {
        self.flags.intersects(VariantFlags::IS_FIELD_LIST_NON_EXHAUSTIVE)
    }

    /// Was this variant obtained as part of recovering from a syntactic error?
    #[inline]
    pub fn is_recovered(&self) -> bool {
        self.flags.intersects(VariantFlags::IS_RECOVERED)
    }

    /// Computes the `Ident` of this variant by looking up the `Span`
    pub fn ident(&self, tcx: TyCtxt<'_>) -> Ident {
        Ident::new(self.name, tcx.def_ident_span(self.def_id).unwrap())
    }

    #[inline]
    pub fn ctor_kind(&self) -> Option<CtorKind> {
        self.ctor.map(|(kind, _)| kind)
    }

    #[inline]
    pub fn ctor_def_id(&self) -> Option<DefId> {
        self.ctor.map(|(_, def_id)| def_id)
    }
}

impl PartialEq for VariantDef {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // There should be only one `VariantDef` for each `def_id`, therefore
        // it is fine to implement `PartialEq` only based on `def_id`.
        //
        // Below, we exhaustively destructure `self` and `other` so that if the
        // definition of `VariantDef` changes, a compile-error will be produced,
        // reminding us to revisit this assumption.

        let Self { def_id: lhs_def_id, ctor: _, name: _, discr: _, fields: _, flags: _ } = &self;
        let Self { def_id: rhs_def_id, ctor: _, name: _, discr: _, fields: _, flags: _ } = other;
        lhs_def_id == rhs_def_id
    }
}

impl Eq for VariantDef {}

impl Hash for VariantDef {
    #[inline]
    fn hash<H: Hasher>(&self, s: &mut H) {
        // There should be only one `VariantDef` for each `def_id`, therefore
        // it is fine to implement `Hash` only based on `def_id`.
        //
        // Below, we exhaustively destructure `self` so that if the definition
        // of `VariantDef` changes, a compile-error will be produced, reminding
        // us to revisit this assumption.

        let Self { def_id, ctor: _, name: _, discr: _, fields: _, flags: _ } = &self;
        def_id.hash(s)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, TyEncodable, TyDecodable, HashStable)]
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

#[derive(Debug, HashStable, TyEncodable, TyDecodable)]
pub struct FieldDef {
    pub did: DefId,
    pub name: Symbol,
    pub vis: Visibility<DefId>,
}

impl PartialEq for FieldDef {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // There should be only one `FieldDef` for each `did`, therefore it is
        // fine to implement `PartialEq` only based on `did`.
        //
        // Below, we exhaustively destructure `self` so that if the definition
        // of `FieldDef` changes, a compile-error will be produced, reminding
        // us to revisit this assumption.

        let Self { did: lhs_did, name: _, vis: _ } = &self;

        let Self { did: rhs_did, name: _, vis: _ } = other;

        lhs_did == rhs_did
    }
}

impl Eq for FieldDef {}

impl Hash for FieldDef {
    #[inline]
    fn hash<H: Hasher>(&self, s: &mut H) {
        // There should be only one `FieldDef` for each `did`, therefore it is
        // fine to implement `Hash` only based on `did`.
        //
        // Below, we exhaustively destructure `self` so that if the definition
        // of `FieldDef` changes, a compile-error will be produced, reminding
        // us to revisit this assumption.

        let Self { did, name: _, vis: _ } = &self;

        did.hash(s)
    }
}

impl<'tcx> FieldDef {
    /// Returns the type of this field. The resulting type is not normalized. The `subst` is
    /// typically obtained via the second field of [`TyKind::Adt`].
    pub fn ty(&self, tcx: TyCtxt<'tcx>, subst: SubstsRef<'tcx>) -> Ty<'tcx> {
        tcx.type_of(self.did).subst(tcx, subst)
    }

    /// Computes the `Ident` of this variant by looking up the `Span`
    pub fn ident(&self, tcx: TyCtxt<'_>) -> Ident {
        Ident::new(self.name, tcx.def_ident_span(self.did).unwrap())
    }
}

pub type Attributes<'tcx> = impl Iterator<Item = &'tcx ast::Attribute>;
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
    /// ```compile_fail,(E0119)
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

/// Useful source information about where a desugared associated type for an
/// RPITIT originated from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encodable, Decodable, HashStable)]
pub enum ImplTraitInTraitData {
    Trait { fn_def_id: DefId, opaque_def_id: DefId },
    Impl { fn_def_id: DefId },
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn typeck_body(self, body: hir::BodyId) -> &'tcx TypeckResults<'tcx> {
        self.typeck(self.hir().body_owner_def_id(body))
    }

    pub fn provided_trait_methods(self, id: DefId) -> impl 'tcx + Iterator<Item = &'tcx AssocItem> {
        self.associated_items(id)
            .in_definition_order()
            .filter(move |item| item.kind == AssocKind::Fn && item.defaultness(self).has_value())
    }

    pub fn repr_options_of_def(self, did: DefId) -> ReprOptions {
        let mut flags = ReprFlags::empty();
        let mut size = None;
        let mut max_align: Option<Align> = None;
        let mut min_pack: Option<Align> = None;

        // Generate a deterministically-derived seed from the item's path hash
        // to allow for cross-crate compilation to actually work
        let mut field_shuffle_seed = self.def_path_hash(did).0.to_smaller_hash();

        // If the user defined a custom seed for layout randomization, xor the item's
        // path hash with the user defined seed, this will allowing determinism while
        // still allowing users to further randomize layout generation for e.g. fuzzing
        if let Some(user_seed) = self.sess.opts.unstable_opts.layout_seed {
            field_shuffle_seed ^= user_seed;
        }

        for attr in self.get_attrs(did, sym::repr) {
            for r in attr::parse_repr_attr(&self.sess, attr) {
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
                    attr::ReprSimd => ReprFlags::IS_SIMD,
                    attr::ReprInt(i) => {
                        size = Some(match i {
                            attr::IntType::SignedInt(x) => match x {
                                ast::IntTy::Isize => IntegerType::Pointer(true),
                                ast::IntTy::I8 => IntegerType::Fixed(Integer::I8, true),
                                ast::IntTy::I16 => IntegerType::Fixed(Integer::I16, true),
                                ast::IntTy::I32 => IntegerType::Fixed(Integer::I32, true),
                                ast::IntTy::I64 => IntegerType::Fixed(Integer::I64, true),
                                ast::IntTy::I128 => IntegerType::Fixed(Integer::I128, true),
                            },
                            attr::IntType::UnsignedInt(x) => match x {
                                ast::UintTy::Usize => IntegerType::Pointer(false),
                                ast::UintTy::U8 => IntegerType::Fixed(Integer::I8, false),
                                ast::UintTy::U16 => IntegerType::Fixed(Integer::I16, false),
                                ast::UintTy::U32 => IntegerType::Fixed(Integer::I32, false),
                                ast::UintTy::U64 => IntegerType::Fixed(Integer::I64, false),
                                ast::UintTy::U128 => IntegerType::Fixed(Integer::I128, false),
                            },
                        });
                        ReprFlags::empty()
                    }
                    attr::ReprAlign(align) => {
                        max_align = max_align.max(Some(Align::from_bytes(align as u64).unwrap()));
                        ReprFlags::empty()
                    }
                });
            }
        }

        // If `-Z randomize-layout` was enabled for the type definition then we can
        // consider performing layout randomization
        if self.sess.opts.unstable_opts.randomize_layout {
            flags.insert(ReprFlags::RANDOMIZE_LAYOUT);
        }

        // This is here instead of layout because the choice must make it into metadata.
        if !self.consider_optimizing(|| format!("Reorder fields of {:?}", self.def_path_str(did))) {
            flags.insert(ReprFlags::IS_LINEAR);
        }

        ReprOptions { int: size, align: max_align, pack: min_pack, flags, field_shuffle_seed }
    }

    /// Look up the name of a definition across crates. This does not look at HIR.
    pub fn opt_item_name(self, def_id: DefId) -> Option<Symbol> {
        if let Some(cnum) = def_id.as_crate_root() {
            Some(self.crate_name(cnum))
        } else {
            let def_key = self.def_key(def_id);
            match def_key.disambiguated_data.data {
                // The name of a constructor is that of its parent.
                rustc_hir::definitions::DefPathData::Ctor => self
                    .opt_item_name(DefId { krate: def_id.krate, index: def_key.parent.unwrap() }),
                // The name of opaque types only exists in HIR.
                rustc_hir::definitions::DefPathData::ImplTrait
                    if let Some(def_id) = def_id.as_local() =>
                    self.hir().opt_name(self.hir().local_def_id_to_hir_id(def_id)),
                _ => def_key.get_opt_name(),
            }
        }
    }

    /// Look up the name of a definition across crates. This does not look at HIR.
    ///
    /// This method will ICE if the corresponding item does not have a name. In these cases, use
    /// [`opt_item_name`] instead.
    ///
    /// [`opt_item_name`]: Self::opt_item_name
    pub fn item_name(self, id: DefId) -> Symbol {
        self.opt_item_name(id).unwrap_or_else(|| {
            bug!("item_name: no name for {:?}", self.def_path(id));
        })
    }

    /// Look up the name and span of a definition.
    ///
    /// See [`item_name`][Self::item_name] for more information.
    pub fn opt_item_ident(self, def_id: DefId) -> Option<Ident> {
        let def = self.opt_item_name(def_id)?;
        let span = def_id
            .as_local()
            .and_then(|id| self.def_ident_span(id))
            .unwrap_or(rustc_span::DUMMY_SP);
        Some(Ident::new(def, span))
    }

    pub fn opt_associated_item(self, def_id: DefId) -> Option<AssocItem> {
        if let DefKind::AssocConst | DefKind::AssocFn | DefKind::AssocTy = self.def_kind(def_id) {
            Some(self.associated_item(def_id))
        } else {
            None
        }
    }

    /// If the def-id is an associated type that was desugared from a
    /// return-position `impl Trait` from a trait, then provide the source info
    /// about where that RPITIT came from.
    pub fn opt_rpitit_info(self, def_id: DefId) -> Option<ImplTraitInTraitData> {
        if let DefKind::AssocTy = self.def_kind(def_id) {
            self.associated_item(def_id).opt_rpitit_info
        } else {
            None
        }
    }

    pub fn find_field_index(self, ident: Ident, variant: &VariantDef) -> Option<usize> {
        variant
            .fields
            .iter()
            .position(|field| self.hygienic_eq(ident, field.ident(self), variant.def_id))
    }

    /// Returns `true` if the impls are the same polarity and the trait either
    /// has no items or is annotated `#[marker]` and prevents item overrides.
    pub fn impls_are_allowed_to_overlap(
        self,
        def_id1: DefId,
        def_id2: DefId,
    ) -> Option<ImplOverlapKind> {
        // If either trait impl references an error, they're allowed to overlap,
        // as one of them essentially doesn't exist.
        if self.impl_trait_ref(def_id1).map_or(false, |tr| tr.subst_identity().references_error())
            || self
                .impl_trait_ref(def_id2)
                .map_or(false, |tr| tr.subst_identity().references_error())
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
                trait_ref.map_or(false, |tr| self.trait_def(tr.skip_binder().def_id).is_marker)
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
                let enum_did = self.parent(did);
                self.adt_def(enum_did).variant_with_id(did)
            }
            Res::Def(DefKind::Struct | DefKind::Union, did) => self.adt_def(did).non_enum_variant(),
            Res::Def(DefKind::Ctor(CtorOf::Variant, ..), variant_ctor_did) => {
                let variant_did = self.parent(variant_ctor_did);
                let enum_did = self.parent(variant_did);
                self.adt_def(enum_did).variant_with_ctor_id(variant_ctor_did)
            }
            Res::Def(DefKind::Ctor(CtorOf::Struct, ..), ctor_did) => {
                let struct_did = self.parent(ctor_did);
                self.adt_def(struct_did).non_enum_variant()
            }
            _ => bug!("expect_variant_res used with unexpected res {:?}", res),
        }
    }

    /// Returns the possibly-auto-generated MIR of a `(DefId, Subst)` pair.
    #[instrument(skip(self), level = "debug")]
    pub fn instance_mir(self, instance: ty::InstanceDef<'tcx>) -> &'tcx Body<'tcx> {
        match instance {
            ty::InstanceDef::Item(def) => {
                debug!("calling def_kind on def: {:?}", def);
                let def_kind = self.def_kind(def.did);
                debug!("returned from def_kind: {:?}", def_kind);
                match def_kind {
                    DefKind::Const
                    | DefKind::Static(..)
                    | DefKind::AssocConst
                    | DefKind::Ctor(..)
                    | DefKind::AnonConst
                    | DefKind::InlineConst => self.mir_for_ctfe_opt_const_arg(def),
                    // If the caller wants `mir_for_ctfe` of a function they should not be using
                    // `instance_mir`, so we'll assume const fn also wants the optimized version.
                    _ => {
                        assert_eq!(def.const_param_did, None);
                        self.optimized_mir(def.did)
                    }
                }
            }
            ty::InstanceDef::VTableShim(..)
            | ty::InstanceDef::ReifyShim(..)
            | ty::InstanceDef::Intrinsic(..)
            | ty::InstanceDef::FnPtrShim(..)
            | ty::InstanceDef::Virtual(..)
            | ty::InstanceDef::ClosureOnceShim { .. }
            | ty::InstanceDef::DropGlue(..)
            | ty::InstanceDef::CloneShim(..) => self.mir_shims(instance),
        }
    }

    // FIXME(@lcnr): Remove this function.
    pub fn get_attrs_unchecked(self, did: DefId) -> &'tcx [ast::Attribute] {
        if let Some(did) = did.as_local() {
            self.hir().attrs(self.hir().local_def_id_to_hir_id(did))
        } else {
            self.item_attrs(did)
        }
    }

    /// Gets all attributes with the given name.
    pub fn get_attrs(self, did: DefId, attr: Symbol) -> ty::Attributes<'tcx> {
        let filter_fn = move |a: &&ast::Attribute| a.has_name(attr);
        if let Some(did) = did.as_local() {
            self.hir().attrs(self.hir().local_def_id_to_hir_id(did)).iter().filter(filter_fn)
        } else if cfg!(debug_assertions) && rustc_feature::is_builtin_only_local(attr) {
            bug!("tried to access the `only_local` attribute `{}` from an extern crate", attr);
        } else {
            self.item_attrs(did).iter().filter(filter_fn)
        }
    }

    pub fn get_attr(self, did: DefId, attr: Symbol) -> Option<&'tcx ast::Attribute> {
        if cfg!(debug_assertions) && !rustc_feature::is_valid_for_get_attr(attr) {
            bug!("get_attr: unexpected called with DefId `{:?}`, attr `{:?}`", did, attr);
        } else {
            self.get_attrs(did, attr).next()
        }
    }

    /// Determines whether an item is annotated with an attribute.
    pub fn has_attr(self, did: DefId, attr: Symbol) -> bool {
        if cfg!(debug_assertions) && !did.is_local() && rustc_feature::is_builtin_only_local(attr) {
            bug!("tried to access the `only_local` attribute `{}` from an extern crate", attr);
        } else {
            self.get_attrs(did, attr).next().is_some()
        }
    }

    /// Returns `true` if this is an `auto trait`.
    pub fn trait_is_auto(self, trait_def_id: DefId) -> bool {
        self.trait_def(trait_def_id).has_auto_impl
    }

    /// Returns `true` if this is coinductive, either because it is
    /// an auto trait or because it has the `#[rustc_coinductive]` attribute.
    pub fn trait_is_coinductive(self, trait_def_id: DefId) -> bool {
        self.trait_def(trait_def_id).is_coinductive
    }

    /// Returns `true` if this is a trait alias.
    pub fn trait_is_alias(self, trait_def_id: DefId) -> bool {
        self.def_kind(trait_def_id) == DefKind::TraitAlias
    }

    /// Returns layout of a generator. Layout might be unavailable if the
    /// generator is tainted by errors.
    pub fn generator_layout(self, def_id: DefId) -> Option<&'tcx GeneratorLayout<'tcx>> {
        self.optimized_mir(def_id).generator_layout()
    }

    /// Given the `DefId` of an impl, returns the `DefId` of the trait it implements.
    /// If it implements no trait, returns `None`.
    pub fn trait_id_of_impl(self, def_id: DefId) -> Option<DefId> {
        self.impl_trait_ref(def_id).map(|tr| tr.skip_binder().def_id)
    }

    /// If the given `DefId` describes an item belonging to a trait,
    /// returns the `DefId` of the trait that the trait item belongs to;
    /// otherwise, returns `None`.
    pub fn trait_of_item(self, def_id: DefId) -> Option<DefId> {
        if let DefKind::AssocConst | DefKind::AssocFn | DefKind::AssocTy = self.def_kind(def_id) {
            let parent = self.parent(def_id);
            if let DefKind::Trait | DefKind::TraitAlias = self.def_kind(parent) {
                return Some(parent);
            }
        }
        None
    }

    /// If the given `DefId` describes a method belonging to an impl, returns the
    /// `DefId` of the impl that the method belongs to; otherwise, returns `None`.
    pub fn impl_of_method(self, def_id: DefId) -> Option<DefId> {
        if let DefKind::AssocConst | DefKind::AssocFn | DefKind::AssocTy = self.def_kind(def_id) {
            let parent = self.parent(def_id);
            if let DefKind::Impl { .. } = self.def_kind(parent) {
                return Some(parent);
            }
        }
        None
    }

    /// Check if the given `DefId` is `#\[automatically_derived\]`, *and*
    /// whether it was produced by expanding a builtin derive macro.
    pub fn is_builtin_derived(self, def_id: DefId) -> bool {
        if self.is_automatically_derived(def_id)
            && let Some(def_id) = def_id.as_local()
            && let outer = self.def_span(def_id).ctxt().outer_expn_data()
            && matches!(outer.kind, ExpnKind::Macro(MacroKind::Derive, _))
            && self.has_attr(outer.macro_def_id.unwrap(), sym::rustc_builtin_macro)
        {
            true
        } else {
            false
        }
    }

    /// Check if the given `DefId` is `#\[automatically_derived\]`.
    pub fn is_automatically_derived(self, def_id: DefId) -> bool {
        self.has_attr(def_id, sym::automatically_derived)
    }

    /// Looks up the span of `impl_did` if the impl is local; otherwise returns `Err`
    /// with the name of the crate containing the impl.
    pub fn span_of_impl(self, impl_def_id: DefId) -> Result<Span, Symbol> {
        if let Some(impl_def_id) = impl_def_id.as_local() {
            Ok(self.def_span(impl_def_id))
        } else {
            Err(self.crate_name(impl_def_id.krate))
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
                .hygienic_eq(def_name.span.ctxt(), self.expn_that_defined(def_parent_def_id))
    }

    pub fn adjust_ident(self, mut ident: Ident, scope: DefId) -> Ident {
        ident.span.normalize_to_macros_2_0_and_adjust(self.expn_that_defined(scope));
        ident
    }

    // FIXME(vincenzoapalzzo): move the HirId to a LocalDefId
    pub fn adjust_ident_and_get_scope(
        self,
        mut ident: Ident,
        scope: DefId,
        block: hir::HirId,
    ) -> (Ident, DefId) {
        let scope = ident
            .span
            .normalize_to_macros_2_0_and_adjust(self.expn_that_defined(scope))
            .and_then(|actual_expansion| actual_expansion.expn_data().parent_module)
            .unwrap_or_else(|| self.parent_module(block).to_def_id());
        (ident, scope)
    }

    /// Returns `true` if the debuginfo for `span` should be collapsed to the outermost expansion
    /// site. Only applies when `Span` is the result of macro expansion.
    ///
    /// - If the `collapse_debuginfo` feature is enabled then debuginfo is not collapsed by default
    ///   and only when a macro definition is annotated with `#[collapse_debuginfo]`.
    /// - If `collapse_debuginfo` is not enabled, then debuginfo is collapsed by default.
    ///
    /// When `-Zdebug-macros` is provided then debuginfo will never be collapsed.
    pub fn should_collapse_debuginfo(self, span: Span) -> bool {
        !self.sess.opts.unstable_opts.debug_macros
            && if self.features().collapse_debuginfo {
                span.in_macro_expansion_with_collapse_debuginfo()
            } else {
                // Inlined spans should not be collapsed as that leads to all of the
                // inlined code being attributed to the inline callsite.
                span.from_expansion() && !span.is_inlined()
            }
    }

    #[inline]
    pub fn is_const_fn_raw(self, def_id: DefId) -> bool {
        matches!(
            self.def_kind(def_id),
            DefKind::Fn | DefKind::AssocFn | DefKind::Ctor(..) | DefKind::Closure
        ) && self.constness(def_id) == hir::Constness::Const
    }

    #[inline]
    pub fn is_const_default_method(self, def_id: DefId) -> bool {
        matches!(self.trait_of_item(def_id), Some(trait_id) if self.has_attr(trait_id, sym::const_trait))
    }

    pub fn impl_trait_in_trait_parent_fn(self, mut def_id: DefId) -> DefId {
        match self.opt_rpitit_info(def_id) {
            Some(ImplTraitInTraitData::Trait { fn_def_id, .. })
            | Some(ImplTraitInTraitData::Impl { fn_def_id, .. }) => fn_def_id,
            None => {
                while let def_kind = self.def_kind(def_id) && def_kind != DefKind::AssocFn {
                    debug_assert_eq!(def_kind, DefKind::ImplTraitPlaceholder);
                    def_id = self.parent(def_id);
                }
                def_id
            }
        }
    }

    pub fn impl_method_has_trait_impl_trait_tys(self, def_id: DefId) -> bool {
        if self.def_kind(def_id) != DefKind::AssocFn {
            return false;
        }

        let Some(item) = self.opt_associated_item(def_id) else { return false; };
        if item.container != ty::AssocItemContainer::ImplContainer {
            return false;
        }

        let Some(trait_item_def_id) = item.trait_item_def_id else { return false; };

        if self.lower_impl_trait_in_trait_to_assoc_ty() {
            return !self.associated_items_for_impl_trait_in_trait(trait_item_def_id).is_empty();
        }

        // FIXME(RPITIT): This does a somewhat manual walk through the signature
        // of the trait fn to look for any RPITITs, but that's kinda doing a lot
        // of work. We can probably remove this when we refactor RPITITs to be
        // associated types.
        self.fn_sig(trait_item_def_id).subst_identity().skip_binder().output().walk().any(|arg| {
            if let ty::GenericArgKind::Type(ty) = arg.unpack()
                && let ty::Alias(ty::Projection, data) = ty.kind()
                && self.def_kind(data.def_id) == DefKind::ImplTraitPlaceholder
            {
                true
            } else {
                false
            }
        })
    }
}

/// Yields the parent function's `LocalDefId` if `def_id` is an `impl Trait` definition.
pub fn is_impl_trait_defn(tcx: TyCtxt<'_>, def_id: DefId) -> Option<LocalDefId> {
    let def_id = def_id.as_local()?;
    if let Node::Item(item) = tcx.hir().get_by_def_id(def_id) {
        if let hir::ItemKind::OpaqueTy(ref opaque_ty) = item.kind {
            return match opaque_ty.origin {
                hir::OpaqueTyOrigin::FnReturn(parent) | hir::OpaqueTyOrigin::AsyncFn(parent) => {
                    Some(parent)
                }
                hir::OpaqueTyOrigin::TyAlias => None,
            };
        }
    }
    None
}

pub fn int_ty(ity: ast::IntTy) -> IntTy {
    match ity {
        ast::IntTy::Isize => IntTy::Isize,
        ast::IntTy::I8 => IntTy::I8,
        ast::IntTy::I16 => IntTy::I16,
        ast::IntTy::I32 => IntTy::I32,
        ast::IntTy::I64 => IntTy::I64,
        ast::IntTy::I128 => IntTy::I128,
    }
}

pub fn uint_ty(uty: ast::UintTy) -> UintTy {
    match uty {
        ast::UintTy::Usize => UintTy::Usize,
        ast::UintTy::U8 => UintTy::U8,
        ast::UintTy::U16 => UintTy::U16,
        ast::UintTy::U32 => UintTy::U32,
        ast::UintTy::U64 => UintTy::U64,
        ast::UintTy::U128 => UintTy::U128,
    }
}

pub fn float_ty(fty: ast::FloatTy) -> FloatTy {
    match fty {
        ast::FloatTy::F32 => FloatTy::F32,
        ast::FloatTy::F64 => FloatTy::F64,
    }
}

pub fn ast_int_ty(ity: IntTy) -> ast::IntTy {
    match ity {
        IntTy::Isize => ast::IntTy::Isize,
        IntTy::I8 => ast::IntTy::I8,
        IntTy::I16 => ast::IntTy::I16,
        IntTy::I32 => ast::IntTy::I32,
        IntTy::I64 => ast::IntTy::I64,
        IntTy::I128 => ast::IntTy::I128,
    }
}

pub fn ast_uint_ty(uty: UintTy) -> ast::UintTy {
    match uty {
        UintTy::Usize => ast::UintTy::Usize,
        UintTy::U8 => ast::UintTy::U8,
        UintTy::U16 => ast::UintTy::U16,
        UintTy::U32 => ast::UintTy::U32,
        UintTy::U64 => ast::UintTy::U64,
        UintTy::U128 => ast::UintTy::U128,
    }
}

pub fn provide(providers: &mut ty::query::Providers) {
    closure::provide(providers);
    context::provide(providers);
    erase_regions::provide(providers);
    inhabitedness::provide(providers);
    util::provide(providers);
    print::provide(providers);
    super::util::bug::provide(providers);
    super::middle::provide(providers);
    *providers = ty::query::Providers {
        trait_impls_of: trait_def::trait_impls_of_provider,
        incoherent_impls: trait_def::incoherent_impls_provider,
        const_param_default: consts::const_param_default,
        vtable_allocation: vtable::vtable_allocation_provider,
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
    pub inherent_impls: LocalDefIdMap<Vec<DefId>>,
    pub incoherent_impls: FxHashMap<SimplifiedType, Vec<LocalDefId>>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, TyEncodable, HashStable)]
pub struct SymbolName<'tcx> {
    /// `&str` gives a consistent ordering, which ensures reproducible builds.
    pub name: &'tcx str,
}

impl<'tcx> SymbolName<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, name: &str) -> SymbolName<'tcx> {
        SymbolName {
            name: unsafe { str::from_utf8_unchecked(tcx.arena.alloc_slice(name.as_bytes())) },
        }
    }
}

impl<'tcx> fmt::Display for SymbolName<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.name, fmt)
    }
}

impl<'tcx> fmt::Debug for SymbolName<'tcx> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.name, fmt)
    }
}

#[derive(Debug, Default, Copy, Clone)]
pub struct InferVarInfo {
    /// This is true if we identified that this Ty (`?T`) is found in a `?T: Foo`
    /// obligation, where:
    ///
    ///  * `Foo` is not `Sized`
    ///  * `(): Foo` may be satisfied
    pub self_in_trait: bool,
    /// This is true if we identified that this Ty (`?T`) is found in a `<_ as
    /// _>::AssocType = ?T`
    pub output: bool,
}

/// The constituent parts of a type level constant of kind ADT or array.
#[derive(Copy, Clone, Debug, HashStable)]
pub struct DestructuredConst<'tcx> {
    pub variant: Option<VariantIdx>,
    pub fields: &'tcx [ty::Const<'tcx>],
}

// Some types are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod size_asserts {
    use super::*;
    use rustc_data_structures::static_assert_size;
    // tidy-alphabetical-start
    static_assert_size!(PredicateKind<'_>, 32);
    static_assert_size!(WithCachedTypeInfo<TyKind<'_>>, 56);
    // tidy-alphabetical-end
}
