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

use std::assert_matches::assert_matches;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::NonZero;
use std::ptr::NonNull;
use std::{fmt, iter, str};

pub use adt::*;
pub use assoc::*;
pub use generic_args::{GenericArgKind, TermKind, *};
pub use generics::*;
pub use intrinsic::IntrinsicDef;
use rustc_abi::{Align, FieldIdx, Integer, IntegerType, ReprFlags, ReprOptions, VariantIdx};
use rustc_ast::expand::StrippedCfgItem;
use rustc_ast::node_id::NodeMap;
pub use rustc_ast_ir::{Movability, Mutability, try_visit};
use rustc_attr_data_structures::AttributeKind;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap, FxIndexSet};
use rustc_data_structures::intern::Interned;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::steal::Steal;
use rustc_data_structures::unord::UnordMap;
use rustc_errors::{Diag, ErrorGuaranteed};
use rustc_hir::LangItem;
use rustc_hir::def::{CtorKind, CtorOf, DefKind, DocLinkResMap, LifetimeRes, Res};
use rustc_hir::def_id::{CrateNum, DefId, DefIdMap, LocalDefId, LocalDefIdMap};
use rustc_hir::definitions::DisambiguatorState;
use rustc_index::IndexVec;
use rustc_index::bit_set::BitMatrix;
use rustc_macros::{
    Decodable, Encodable, HashStable, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable,
    extension,
};
use rustc_query_system::ich::StableHashingContext;
use rustc_serialize::{Decodable, Encodable};
use rustc_session::lint::LintBuffer;
pub use rustc_session::lint::RegisteredTools;
use rustc_span::hygiene::MacroKind;
use rustc_span::{DUMMY_SP, ExpnId, ExpnKind, Ident, Span, Symbol, kw, sym};
pub use rustc_type_ir::data_structures::{DelayedMap, DelayedSet};
pub use rustc_type_ir::fast_reject::DeepRejectCtxt;
#[allow(
    hidden_glob_reexports,
    rustc::usage_of_type_ir_inherent,
    rustc::non_glob_import_of_type_ir_inherent
)]
use rustc_type_ir::inherent;
pub use rustc_type_ir::relate::VarianceDiagInfo;
pub use rustc_type_ir::*;
#[allow(hidden_glob_reexports, unused_imports)]
use rustc_type_ir::{InferCtxtLike, Interner};
use tracing::{debug, instrument};
pub use vtable::*;
use {rustc_ast as ast, rustc_attr_data_structures as attr, rustc_hir as hir};

pub use self::closure::{
    BorrowKind, CAPTURE_STRUCT_LOCAL, CaptureInfo, CapturedPlace, ClosureTypeInfo,
    MinCaptureInformationMap, MinCaptureList, RootVariableMinCaptureList, UpvarCapture, UpvarId,
    UpvarPath, analyze_coroutine_closure_captures, is_ancestor_or_same_capture,
    place_to_string_for_capture,
};
pub use self::consts::{
    Const, ConstInt, ConstKind, Expr, ExprKind, ScalarInt, UnevaluatedConst, ValTree, ValTreeKind,
    Value,
};
pub use self::context::{
    CtxtInterners, CurrentGcx, DeducedParamAttrs, Feed, FreeRegionInfo, GlobalCtxt, Lift, TyCtxt,
    TyCtxtFeed, tls,
};
pub use self::fold::*;
pub use self::instance::{Instance, InstanceKind, ReifyReason, ShortInstance, UnusedGenericParams};
pub use self::list::{List, ListWithCachedTypeInfo};
pub use self::opaque_types::OpaqueTypeKey;
pub use self::parameterized::ParameterizedOverTcx;
pub use self::pattern::{Pattern, PatternKind};
pub use self::predicate::{
    AliasTerm, Clause, ClauseKind, CoercePredicate, ExistentialPredicate,
    ExistentialPredicateStableCmpExt, ExistentialProjection, ExistentialTraitRef,
    HostEffectPredicate, NormalizesTo, OutlivesPredicate, PolyCoercePredicate,
    PolyExistentialPredicate, PolyExistentialProjection, PolyExistentialTraitRef,
    PolyProjectionPredicate, PolyRegionOutlivesPredicate, PolySubtypePredicate, PolyTraitPredicate,
    PolyTraitRef, PolyTypeOutlivesPredicate, Predicate, PredicateKind, ProjectionPredicate,
    RegionOutlivesPredicate, SubtypePredicate, TraitPredicate, TraitRef, TypeOutlivesPredicate,
};
pub use self::region::{
    BoundRegion, BoundRegionKind, EarlyParamRegion, LateParamRegion, LateParamRegionKind, Region,
    RegionKind, RegionVid,
};
pub use self::rvalue_scopes::RvalueScopes;
pub use self::sty::{
    AliasTy, Article, Binder, BoundTy, BoundTyKind, BoundVariableKind, CanonicalPolyFnSig,
    CoroutineArgsExt, EarlyBinder, FnSig, InlineConstArgs, InlineConstArgsParts, ParamConst,
    ParamTy, PolyFnSig, TyKind, TypeAndMut, TypingMode, UpvarArgs,
};
pub use self::trait_def::TraitDef;
pub use self::typeck_results::{
    CanonicalUserType, CanonicalUserTypeAnnotation, CanonicalUserTypeAnnotations, IsIdentity,
    Rust2024IncompatiblePatInfo, TypeckResults, UserType, UserTypeAnnotationIndex, UserTypeKind,
};
pub use self::visit::*;
use crate::error::{OpaqueHiddenTypeMismatch, TypeMismatchReason};
use crate::metadata::ModChild;
use crate::middle::privacy::EffectiveVisibilities;
use crate::mir::{Body, CoroutineLayout, CoroutineSavedLocal, SourceInfo};
use crate::query::{IntoQueryParam, Providers};
use crate::ty;
use crate::ty::codec::{TyDecoder, TyEncoder};
pub use crate::ty::diagnostics::*;
use crate::ty::fast_reject::SimplifiedType;
use crate::ty::util::Discr;
use crate::ty::walk::TypeWalker;

pub mod abstract_const;
pub mod adjustment;
pub mod cast;
pub mod codec;
pub mod error;
pub mod fast_reject;
pub mod inhabitedness;
pub mod layout;
pub mod normalize_erasing_regions;
pub mod pattern;
pub mod print;
pub mod relate;
pub mod significant_drop_order;
pub mod trait_def;
pub mod util;
pub mod vtable;

mod adt;
mod assoc;
mod closure;
mod consts;
mod context;
mod diagnostics;
mod elaborate_impl;
mod erase_regions;
mod fold;
mod generic_args;
mod generics;
mod impls_ty;
mod instance;
mod intrinsic;
mod list;
mod opaque_types;
mod parameterized;
mod predicate;
mod region;
mod rvalue_scopes;
mod structural_impls;
#[allow(hidden_glob_reexports)]
mod sty;
mod typeck_results;
mod visit;

// Data types

pub struct ResolverOutputs {
    pub global_ctxt: ResolverGlobalCtxt,
    pub ast_lowering: ResolverAstLowering,
}

#[derive(Debug)]
pub struct ResolverGlobalCtxt {
    pub visibilities_for_hashing: Vec<(LocalDefId, Visibility)>,
    /// Item with a given `LocalDefId` was defined during macro expansion with ID `ExpnId`.
    pub expn_that_defined: FxHashMap<LocalDefId, ExpnId>,
    pub effective_visibilities: EffectiveVisibilities,
    pub extern_crate_map: UnordMap<LocalDefId, CrateNum>,
    pub maybe_unused_trait_imports: FxIndexSet<LocalDefId>,
    pub module_children: LocalDefIdMap<Vec<ModChild>>,
    pub glob_map: FxIndexMap<LocalDefId, FxIndexSet<Symbol>>,
    pub main_def: Option<MainDefinition>,
    pub trait_impls: FxIndexMap<DefId, Vec<LocalDefId>>,
    /// A list of proc macro LocalDefIds, written out in the order in which
    /// they are declared in the static array generated by proc_macro_harness.
    pub proc_macros: Vec<LocalDefId>,
    /// Mapping from ident span to path span for paths that don't exist as written, but that
    /// exist under `std`. For example, wrote `str::from_utf8` instead of `std::str::from_utf8`.
    pub confused_type_with_std_module: FxIndexMap<Span, Span>,
    pub doc_link_resolutions: FxIndexMap<LocalDefId, DocLinkResMap>,
    pub doc_link_traits_in_scope: FxIndexMap<LocalDefId, Vec<DefId>>,
    pub all_macro_rules: FxHashSet<Symbol>,
    pub stripped_cfg_items: Steal<Vec<StrippedCfgItem>>,
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

    pub node_id_to_def_id: NodeMap<LocalDefId>,

    pub disambiguator: DisambiguatorState,

    pub trait_map: NodeMap<Vec<hir::TraitCandidate>>,
    /// List functions and methods for which lifetime elision was successful.
    pub lifetime_elision_allowed: FxHashSet<ast::NodeId>,

    /// Lints that were emitted by the resolver and early lints.
    pub lint_buffer: Steal<LintBuffer>,

    /// Information about functions signatures for delegation items expansion
    pub delegation_fn_sigs: LocalDefIdMap<DelegationFnSig>,
}

#[derive(Debug)]
pub struct DelegationFnSig {
    pub header: ast::FnHeader,
    pub param_count: usize,
    pub has_self: bool,
    pub c_variadic: bool,
    pub target_feature: bool,
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
    pub impl_args: ty::GenericArgsRef<'tcx>,
    pub self_ty: Ty<'tcx>,
    pub trait_ref: Option<TraitRef<'tcx>>,
    pub predicates: Vec<Predicate<'tcx>>,
}

#[derive(Copy, Clone, Debug, TyEncodable, TyDecodable, HashStable)]
pub struct ImplTraitHeader<'tcx> {
    pub trait_ref: ty::EarlyBinder<'tcx, ty::TraitRef<'tcx>>,
    pub polarity: ImplPolarity,
    pub safety: hir::Safety,
    pub constness: hir::Constness,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, TypeFoldable, TypeVisitable)]
pub enum ImplSubject<'tcx> {
    Trait(TraitRef<'tcx>),
    Inherent(Ty<'tcx>),
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, TyEncodable, TyDecodable, HashStable, Debug)]
#[derive(TypeFoldable, TypeVisitable)]
pub enum Asyncness {
    Yes,
    No,
}

impl Asyncness {
    pub fn is_async(self) -> bool {
        matches!(self, Asyncness::Yes)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Copy, Hash, Encodable, Decodable, HashStable)]
pub enum Visibility<Id = LocalDefId> {
    /// Visible everywhere (including in other crates).
    Public,
    /// Visible only in the given crate-local module.
    Restricted(Id),
}

impl Visibility {
    pub fn to_string(self, def_id: LocalDefId, tcx: TyCtxt<'_>) -> String {
        match self {
            ty::Visibility::Restricted(restricted_id) => {
                if restricted_id.is_top_level_module() {
                    "pub(crate)".to_string()
                } else if restricted_id == tcx.parent_module_from_def_id(def_id).to_local_def_id() {
                    "pub(self)".to_string()
                } else {
                    format!(
                        "pub(in crate{})",
                        tcx.def_path(restricted_id.to_def_id()).to_string_no_crate_verbose()
                    )
                }
            }
            ty::Visibility::Public => "pub".to_string(),
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
    pub fn local_parent(self, id: impl Into<LocalDefId>) -> LocalDefId {
        self.parent(id.into().to_def_id()).expect_local()
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
#[derive(Copy, Clone, PartialEq, Eq, Hash, HashStable)]
#[rustc_diagnostic_item = "Ty"]
#[rustc_pass_by_value]
pub struct Ty<'tcx>(Interned<'tcx, WithCachedTypeInfo<TyKind<'tcx>>>);

impl<'tcx> rustc_type_ir::inherent::IntoKind for Ty<'tcx> {
    type Kind = TyKind<'tcx>;

    fn kind(self) -> TyKind<'tcx> {
        *self.kind()
    }
}

impl<'tcx> rustc_type_ir::Flags for Ty<'tcx> {
    fn flags(&self) -> TypeFlags {
        self.0.flags
    }

    fn outer_exclusive_binder(&self) -> DebruijnIndex {
        self.0.outer_exclusive_binder
    }
}

impl EarlyParamRegion {
    /// Does this early bound region have a name? Early bound regions normally
    /// always have names except when using anonymous lifetimes (`'_`).
    pub fn has_name(&self) -> bool {
        self.name != kw::UnderscoreLifetime
    }
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
    pub predicates: DefIdMap<&'tcx [(Clause<'tcx>, Span)]>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Term<'tcx> {
    ptr: NonNull<()>,
    marker: PhantomData<(Ty<'tcx>, Const<'tcx>)>,
}

impl<'tcx> rustc_type_ir::inherent::Term<TyCtxt<'tcx>> for Term<'tcx> {}

impl<'tcx> rustc_type_ir::inherent::IntoKind for Term<'tcx> {
    type Kind = TermKind<'tcx>;

    fn kind(self) -> Self::Kind {
        self.unpack()
    }
}

unsafe impl<'tcx> rustc_data_structures::sync::DynSend for Term<'tcx> where
    &'tcx (Ty<'tcx>, Const<'tcx>): rustc_data_structures::sync::DynSend
{
}
unsafe impl<'tcx> rustc_data_structures::sync::DynSync for Term<'tcx> where
    &'tcx (Ty<'tcx>, Const<'tcx>): rustc_data_structures::sync::DynSync
{
}
unsafe impl<'tcx> Send for Term<'tcx> where &'tcx (Ty<'tcx>, Const<'tcx>): Send {}
unsafe impl<'tcx> Sync for Term<'tcx> where &'tcx (Ty<'tcx>, Const<'tcx>): Sync {}

impl Debug for Term<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.unpack() {
            TermKind::Ty(ty) => write!(f, "Term::Ty({ty:?})"),
            TermKind::Const(ct) => write!(f, "Term::Const({ct:?})"),
        }
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
        match self.unpack() {
            ty::TermKind::Ty(ty) => ty.try_fold_with(folder).map(Into::into),
            ty::TermKind::Const(ct) => ct.try_fold_with(folder).map(Into::into),
        }
    }

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        match self.unpack() {
            ty::TermKind::Ty(ty) => ty.fold_with(folder).into(),
            ty::TermKind::Const(ct) => ct.fold_with(folder).into(),
        }
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for Term<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        match self.unpack() {
            ty::TermKind::Ty(ty) => ty.visit_with(visitor),
            ty::TermKind::Const(ct) => ct.visit_with(visitor),
        }
    }
}

impl<'tcx, E: TyEncoder<'tcx>> Encodable<E> for Term<'tcx> {
    fn encode(&self, e: &mut E) {
        self.unpack().encode(e)
    }
}

impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for Term<'tcx> {
    fn decode(d: &mut D) -> Self {
        let res: TermKind<'tcx> = Decodable::decode(d);
        res.pack()
    }
}

impl<'tcx> Term<'tcx> {
    #[inline]
    pub fn unpack(self) -> TermKind<'tcx> {
        let ptr =
            unsafe { self.ptr.map_addr(|addr| NonZero::new_unchecked(addr.get() & !TAG_MASK)) };
        // SAFETY: use of `Interned::new_unchecked` here is ok because these
        // pointers were originally created from `Interned` types in `pack()`,
        // and this is just going in the other direction.
        unsafe {
            match self.ptr.addr().get() & TAG_MASK {
                TYPE_TAG => TermKind::Ty(Ty(Interned::new_unchecked(
                    ptr.cast::<WithCachedTypeInfo<ty::TyKind<'tcx>>>().as_ref(),
                ))),
                CONST_TAG => TermKind::Const(ty::Const(Interned::new_unchecked(
                    ptr.cast::<WithCachedTypeInfo<ty::ConstKind<'tcx>>>().as_ref(),
                ))),
                _ => core::intrinsics::unreachable(),
            }
        }
    }

    pub fn as_type(&self) -> Option<Ty<'tcx>> {
        if let TermKind::Ty(ty) = self.unpack() { Some(ty) } else { None }
    }

    pub fn expect_type(&self) -> Ty<'tcx> {
        self.as_type().expect("expected a type, but found a const")
    }

    pub fn as_const(&self) -> Option<Const<'tcx>> {
        if let TermKind::Const(c) = self.unpack() { Some(c) } else { None }
    }

    pub fn expect_const(&self) -> Const<'tcx> {
        self.as_const().expect("expected a const, but found a type")
    }

    pub fn into_arg(self) -> GenericArg<'tcx> {
        match self.unpack() {
            TermKind::Ty(ty) => ty.into(),
            TermKind::Const(c) => c.into(),
        }
    }

    pub fn to_alias_term(self) -> Option<AliasTerm<'tcx>> {
        match self.unpack() {
            TermKind::Ty(ty) => match *ty.kind() {
                ty::Alias(_kind, alias_ty) => Some(alias_ty.into()),
                _ => None,
            },
            TermKind::Const(ct) => match ct.kind() {
                ConstKind::Unevaluated(uv) => Some(uv.into()),
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

    /// Iterator that walks `self` and any types reachable from
    /// `self`, in depth-first order. Note that just walks the types
    /// that appear in `self`, it does not descend into the fields of
    /// structs or variants. For example:
    ///
    /// ```text
    /// isize => { isize }
    /// Foo<Bar<isize>> => { Foo<Bar<isize>>, Bar<isize>, isize }
    /// [isize] => { [isize], isize }
    /// ```
    pub fn walk(self) -> TypeWalker<TyCtxt<'tcx>> {
        TypeWalker::new(self.into())
    }
}

const TAG_MASK: usize = 0b11;
const TYPE_TAG: usize = 0b00;
const CONST_TAG: usize = 0b01;

#[extension(pub trait TermKindPackExt<'tcx>)]
impl<'tcx> TermKind<'tcx> {
    #[inline]
    fn pack(self) -> Term<'tcx> {
        let (tag, ptr) = match self {
            TermKind::Ty(ty) => {
                // Ensure we can use the tag bits.
                assert_eq!(align_of_val(&*ty.0.0) & TAG_MASK, 0);
                (TYPE_TAG, NonNull::from(ty.0.0).cast())
            }
            TermKind::Const(ct) => {
                // Ensure we can use the tag bits.
                assert_eq!(align_of_val(&*ct.0.0) & TAG_MASK, 0);
                (CONST_TAG, NonNull::from(ct.0.0).cast())
            }
        };

        Term { ptr: ptr.map_addr(|addr| addr | tag), marker: PhantomData }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ParamTerm {
    Ty(ParamTy),
    Const(ParamConst),
}

impl ParamTerm {
    pub fn index(self) -> usize {
        match self {
            ParamTerm::Ty(ty) => ty.index as usize,
            ParamTerm::Const(ct) => ct.index as usize,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum TermVid {
    Ty(ty::TyVid),
    Const(ty::ConstVid),
}

impl From<ty::TyVid> for TermVid {
    fn from(value: ty::TyVid) -> Self {
        TermVid::Ty(value)
    }
}

impl From<ty::ConstVid> for TermVid {
    fn from(value: ty::ConstVid) -> Self {
        TermVid::Const(value)
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
/// meaning that the generic parameters have been instantiated with
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
    pub predicates: Vec<Clause<'tcx>>,
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
        self.into_iter()
    }
}

impl<'tcx> IntoIterator for InstantiatedPredicates<'tcx> {
    type Item = (Clause<'tcx>, Span);

    type IntoIter = std::iter::Zip<std::vec::IntoIter<Clause<'tcx>>, std::vec::IntoIter<Span>>;

    fn into_iter(self) -> Self::IntoIter {
        debug_assert_eq!(self.predicates.len(), self.spans.len());
        std::iter::zip(self.predicates, self.spans)
    }
}

impl<'a, 'tcx> IntoIterator for &'a InstantiatedPredicates<'tcx> {
    type Item = (Clause<'tcx>, Span);

    type IntoIter = std::iter::Zip<
        std::iter::Copied<std::slice::Iter<'a, Clause<'tcx>>>,
        std::iter::Copied<std::slice::Iter<'a, Span>>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        debug_assert_eq!(self.predicates.len(), self.spans.len());
        std::iter::zip(self.predicates.iter().copied(), self.spans.iter().copied())
    }
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

/// Whether we're currently in HIR typeck or MIR borrowck.
#[derive(Debug, Clone, Copy)]
pub enum DefiningScopeKind {
    /// During writeback in typeck, we don't care about regions and simply
    /// erase them. This means we also don't check whether regions are
    /// universal in the opaque type key. This will only be checked in
    /// MIR borrowck.
    HirTypeck,
    MirBorrowck,
}

impl<'tcx> OpaqueHiddenType<'tcx> {
    pub fn new_error(tcx: TyCtxt<'tcx>, guar: ErrorGuaranteed) -> OpaqueHiddenType<'tcx> {
        OpaqueHiddenType { span: DUMMY_SP, ty: Ty::new_error(tcx, guar) }
    }

    pub fn build_mismatch_error(
        &self,
        other: &Self,
        tcx: TyCtxt<'tcx>,
    ) -> Result<Diag<'tcx>, ErrorGuaranteed> {
        (self.ty, other.ty).error_reported()?;
        // Found different concrete types for the opaque type.
        let sub_diag = if self.span == other.span {
            TypeMismatchReason::ConflictType { span: self.span }
        } else {
            TypeMismatchReason::PreviousUse { span: self.span }
        };
        Ok(tcx.dcx().create_err(OpaqueHiddenTypeMismatch {
            self_ty: self.ty,
            other_ty: other.ty,
            other_span: other.span,
            sub: sub_diag,
        }))
    }

    #[instrument(level = "debug", skip(tcx), ret)]
    pub fn remap_generic_params_to_declaration_params(
        self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        tcx: TyCtxt<'tcx>,
        defining_scope_kind: DefiningScopeKind,
    ) -> Self {
        let OpaqueTypeKey { def_id, args } = opaque_type_key;

        // Use args to build up a reverse map from regions to their
        // identity mappings. This is necessary because of `impl
        // Trait` lifetimes are computed by replacing existing
        // lifetimes with 'static and remapping only those used in the
        // `impl Trait` return type, resulting in the parameters
        // shifting.
        let id_args = GenericArgs::identity_for_item(tcx, def_id);
        debug!(?id_args);

        // This zip may have several times the same lifetime in `args` paired with a different
        // lifetime from `id_args`. Simply `collect`ing the iterator is the correct behaviour:
        // it will pick the last one, which is the one we introduced in the impl-trait desugaring.
        let map = args.iter().zip(id_args).collect();
        debug!("map = {:#?}", map);

        // Convert the type from the function into a type valid outside by mapping generic
        // parameters to into the context of the opaque.
        //
        // We erase regions when doing this during HIR typeck.
        let this = match defining_scope_kind {
            DefiningScopeKind::HirTypeck => tcx.erase_regions(self),
            DefiningScopeKind::MirBorrowck => self,
        };
        let result = this.fold_with(&mut opaque_types::ReverseMapper::new(tcx, map, self.span));
        if cfg!(debug_assertions) && matches!(defining_scope_kind, DefiningScopeKind::HirTypeck) {
            assert_eq!(result.ty, tcx.erase_regions(result.ty));
        }
        result
    }
}

/// The "placeholder index" fully defines a placeholder region, type, or const. Placeholders are
/// identified by both a universe, as well as a name residing within that universe. Distinct bound
/// regions/types/consts within the same universe simply have an unknown relationship to one
/// another.
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[derive(HashStable, TyEncodable, TyDecodable)]
pub struct Placeholder<T> {
    pub universe: UniverseIndex,
    pub bound: T,
}
impl Placeholder<BoundVar> {
    pub fn find_const_ty_from_env<'tcx>(self, env: ParamEnv<'tcx>) -> Ty<'tcx> {
        let mut candidates = env.caller_bounds().iter().filter_map(|clause| {
            // `ConstArgHasType` are never desugared to be higher ranked.
            match clause.kind().skip_binder() {
                ty::ClauseKind::ConstArgHasType(placeholder_ct, ty) => {
                    assert!(!(placeholder_ct, ty).has_escaping_bound_vars());

                    match placeholder_ct.kind() {
                        ty::ConstKind::Placeholder(placeholder_ct) if placeholder_ct == self => {
                            Some(ty)
                        }
                        _ => None,
                    }
                }
                _ => None,
            }
        });

        let ty = candidates.next().unwrap();
        assert!(candidates.next().is_none());
        ty
    }
}

pub type PlaceholderRegion = Placeholder<BoundRegion>;

impl rustc_type_ir::inherent::PlaceholderLike for PlaceholderRegion {
    fn universe(self) -> UniverseIndex {
        self.universe
    }

    fn var(self) -> BoundVar {
        self.bound.var
    }

    fn with_updated_universe(self, ui: UniverseIndex) -> Self {
        Placeholder { universe: ui, ..self }
    }

    fn new(ui: UniverseIndex, var: BoundVar) -> Self {
        Placeholder { universe: ui, bound: BoundRegion { var, kind: BoundRegionKind::Anon } }
    }
}

pub type PlaceholderType = Placeholder<BoundTy>;

impl rustc_type_ir::inherent::PlaceholderLike for PlaceholderType {
    fn universe(self) -> UniverseIndex {
        self.universe
    }

    fn var(self) -> BoundVar {
        self.bound.var
    }

    fn with_updated_universe(self, ui: UniverseIndex) -> Self {
        Placeholder { universe: ui, ..self }
    }

    fn new(ui: UniverseIndex, var: BoundVar) -> Self {
        Placeholder { universe: ui, bound: BoundTy { var, kind: BoundTyKind::Anon } }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, HashStable)]
#[derive(TyEncodable, TyDecodable)]
pub struct BoundConst<'tcx> {
    pub var: BoundVar,
    pub ty: Ty<'tcx>,
}

pub type PlaceholderConst = Placeholder<BoundVar>;

impl rustc_type_ir::inherent::PlaceholderLike for PlaceholderConst {
    fn universe(self) -> UniverseIndex {
        self.universe
    }

    fn var(self) -> BoundVar {
        self.bound
    }

    fn with_updated_universe(self, ui: UniverseIndex) -> Self {
        Placeholder { universe: ui, ..self }
    }

    fn new(ui: UniverseIndex, var: BoundVar) -> Self {
        Placeholder { universe: ui, bound: var }
    }
}

pub type Clauses<'tcx> = &'tcx ListWithCachedTypeInfo<Clause<'tcx>>;

impl<'tcx> rustc_type_ir::Flags for Clauses<'tcx> {
    fn flags(&self) -> TypeFlags {
        (**self).flags()
    }

    fn outer_exclusive_binder(&self) -> DebruijnIndex {
        (**self).outer_exclusive_binder()
    }
}

/// When interacting with the type system we must provide information about the
/// environment. `ParamEnv` is the type that represents this information. See the
/// [dev guide chapter][param_env_guide] for more information.
///
/// [param_env_guide]: https://rustc-dev-guide.rust-lang.org/typing_parameter_envs.html
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[derive(HashStable, TypeVisitable, TypeFoldable)]
pub struct ParamEnv<'tcx> {
    /// Caller bounds are `Obligation`s that the caller must satisfy. This is
    /// basically the set of bounds on the in-scope type parameters, translated
    /// into `Obligation`s, and elaborated and normalized.
    ///
    /// Use the `caller_bounds()` method to access.
    caller_bounds: Clauses<'tcx>,
}

impl<'tcx> rustc_type_ir::inherent::ParamEnv<TyCtxt<'tcx>> for ParamEnv<'tcx> {
    fn caller_bounds(self) -> impl inherent::SliceLike<Item = ty::Clause<'tcx>> {
        self.caller_bounds()
    }
}

impl<'tcx> ParamEnv<'tcx> {
    /// Construct a trait environment suitable for contexts where there are
    /// no where-clauses in scope. In the majority of cases it is incorrect
    /// to use an empty environment. See the [dev guide section][param_env_guide]
    /// for information on what a `ParamEnv` is and how to acquire one.
    ///
    /// [param_env_guide]: https://rustc-dev-guide.rust-lang.org/typing_parameter_envs.html
    #[inline]
    pub fn empty() -> Self {
        Self::new(ListWithCachedTypeInfo::empty())
    }

    #[inline]
    pub fn caller_bounds(self) -> Clauses<'tcx> {
        self.caller_bounds
    }

    /// Construct a trait environment with the given set of predicates.
    #[inline]
    pub fn new(caller_bounds: Clauses<'tcx>) -> Self {
        ParamEnv { caller_bounds }
    }

    /// Creates a pair of param-env and value for use in queries.
    pub fn and<T: TypeVisitable<TyCtxt<'tcx>>>(self, value: T) -> ParamEnvAnd<'tcx, T> {
        ParamEnvAnd { param_env: self, value }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, TypeFoldable, TypeVisitable)]
#[derive(HashStable)]
pub struct ParamEnvAnd<'tcx, T> {
    pub param_env: ParamEnv<'tcx>,
    pub value: T,
}

impl<'tcx, T> ParamEnvAnd<'tcx, T> {
    pub fn into_parts(self) -> (ParamEnv<'tcx>, T) {
        (self.param_env, self.value)
    }
}

/// The environment in which to do trait solving.
///
/// Most of the time you only need to care about the `ParamEnv`
/// as the `TypingMode` is simply stored in the `InferCtxt`.
///
/// However, there are some places which rely on trait solving
/// without using an `InferCtxt` themselves. For these to be
/// able to use the trait system they have to be able to initialize
/// such an `InferCtxt` with the right `typing_mode`, so they need
/// to track both.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, HashStable)]
#[derive(TypeVisitable, TypeFoldable)]
pub struct TypingEnv<'tcx> {
    pub typing_mode: TypingMode<'tcx>,
    pub param_env: ParamEnv<'tcx>,
}

impl<'tcx> TypingEnv<'tcx> {
    /// Create a typing environment with no where-clauses in scope
    /// where all opaque types and default associated items are revealed.
    ///
    /// This is only suitable for monomorphized, post-typeck environments.
    /// Do not use this for MIR optimizations, as even though they also
    /// use `TypingMode::PostAnalysis`, they may still have where-clauses
    /// in scope.
    pub fn fully_monomorphized() -> TypingEnv<'tcx> {
        TypingEnv { typing_mode: TypingMode::PostAnalysis, param_env: ParamEnv::empty() }
    }

    /// Create a typing environment for use during analysis outside of a body.
    ///
    /// Using a typing environment inside of bodies is not supported as the body
    /// may define opaque types. In this case the used functions have to be
    /// converted to use proper canonical inputs instead.
    pub fn non_body_analysis(
        tcx: TyCtxt<'tcx>,
        def_id: impl IntoQueryParam<DefId>,
    ) -> TypingEnv<'tcx> {
        TypingEnv { typing_mode: TypingMode::non_body_analysis(), param_env: tcx.param_env(def_id) }
    }

    pub fn post_analysis(tcx: TyCtxt<'tcx>, def_id: impl IntoQueryParam<DefId>) -> TypingEnv<'tcx> {
        TypingEnv {
            typing_mode: TypingMode::PostAnalysis,
            param_env: tcx.param_env_normalized_for_post_analysis(def_id),
        }
    }

    /// Modify the `typing_mode` to `PostAnalysis` and eagerly reveal all
    /// opaque types in the `param_env`.
    pub fn with_post_analysis_normalized(self, tcx: TyCtxt<'tcx>) -> TypingEnv<'tcx> {
        let TypingEnv { typing_mode, param_env } = self;
        if let TypingMode::PostAnalysis = typing_mode {
            return self;
        }

        // No need to reveal opaques with the new solver enabled,
        // since we have lazy norm.
        let param_env = if tcx.next_trait_solver_globally() {
            ParamEnv::new(param_env.caller_bounds())
        } else {
            ParamEnv::new(tcx.reveal_opaque_types_in_bounds(param_env.caller_bounds()))
        };
        TypingEnv { typing_mode: TypingMode::PostAnalysis, param_env }
    }

    /// Combine this typing environment with the given `value` to be used by
    /// not (yet) canonicalized queries. This only works if the value does not
    /// contain anything local to some `InferCtxt`, i.e. inference variables or
    /// placeholders.
    pub fn as_query_input<T>(self, value: T) -> PseudoCanonicalInput<'tcx, T>
    where
        T: TypeVisitable<TyCtxt<'tcx>>,
    {
        // FIXME(#132279): We should assert that the value does not contain any placeholders
        // as these placeholders are also local to the current inference context. However, we
        // currently use pseudo-canonical queries in the trait solver which replaces params with
        // placeholders. We should also simply not use pseudo-canonical queries in the trait
        // solver, at which point we can readd this assert. As of writing this comment, this is
        // only used by `fn layout_is_pointer_like` when calling `layout_of`.
        //
        // debug_assert!(!value.has_placeholders());
        PseudoCanonicalInput { typing_env: self, value }
    }
}

/// Similar to `CanonicalInput`, this carries the `typing_mode` and the environment
/// necessary to do any kind of trait solving inside of nested queries.
///
/// Unlike proper canonicalization, this requires the `param_env` and the `value` to not
/// contain anything local to the `infcx` of the caller, so we don't actually canonicalize
/// anything.
///
/// This should be created by using `infcx.pseudo_canonicalize_query(param_env, value)`
/// or by using `typing_env.as_query_input(value)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[derive(HashStable, TypeVisitable, TypeFoldable)]
pub struct PseudoCanonicalInput<'tcx, T> {
    pub typing_env: TypingEnv<'tcx>,
    pub value: T,
}

#[derive(Copy, Clone, Debug, HashStable, Encodable, Decodable)]
pub struct Destructor {
    /// The `DefId` of the destructor method
    pub did: DefId,
}

// FIXME: consider combining this definition with regular `Destructor`
#[derive(Copy, Clone, Debug, HashStable, Encodable, Decodable)]
pub struct AsyncDestructor {
    /// The `DefId` of the `impl AsyncDrop`
    pub impl_did: LocalDefId,
}

#[derive(Clone, Copy, PartialEq, Eq, HashStable, TyEncodable, TyDecodable)]
pub struct VariantFlags(u8);
bitflags::bitflags! {
    impl VariantFlags: u8 {
        const NO_VARIANT_FLAGS        = 0;
        /// Indicates whether the field list of this variant is `#[non_exhaustive]`.
        const IS_FIELD_LIST_NON_EXHAUSTIVE = 1 << 0;
    }
}
rustc_data_structures::external_bitflags_debug! { VariantFlags }

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
    pub fields: IndexVec<FieldIdx, FieldDef>,
    /// The error guarantees from parser, if any.
    tainted: Option<ErrorGuaranteed>,
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
    #[instrument(level = "debug")]
    pub fn new(
        name: Symbol,
        variant_did: Option<DefId>,
        ctor: Option<(CtorKind, DefId)>,
        discr: VariantDiscr,
        fields: IndexVec<FieldIdx, FieldDef>,
        parent_did: DefId,
        recover_tainted: Option<ErrorGuaranteed>,
        is_field_list_non_exhaustive: bool,
    ) -> Self {
        let mut flags = VariantFlags::NO_VARIANT_FLAGS;
        if is_field_list_non_exhaustive {
            flags |= VariantFlags::IS_FIELD_LIST_NON_EXHAUSTIVE;
        }

        VariantDef {
            def_id: variant_did.unwrap_or(parent_did),
            ctor,
            name,
            discr,
            fields,
            flags,
            tainted: recover_tainted,
        }
    }

    /// Returns `true` if the field list of this variant is `#[non_exhaustive]`.
    ///
    /// Note that this function will return `true` even if the type has been
    /// defined in the crate currently being compiled. If that's not what you
    /// want, see [`Self::field_list_has_applicable_non_exhaustive`].
    #[inline]
    pub fn is_field_list_non_exhaustive(&self) -> bool {
        self.flags.intersects(VariantFlags::IS_FIELD_LIST_NON_EXHAUSTIVE)
    }

    /// Returns `true` if the field list of this variant is `#[non_exhaustive]`
    /// and the type has been defined in another crate.
    #[inline]
    pub fn field_list_has_applicable_non_exhaustive(&self) -> bool {
        self.is_field_list_non_exhaustive() && !self.def_id.is_local()
    }

    /// Computes the `Ident` of this variant by looking up the `Span`
    pub fn ident(&self, tcx: TyCtxt<'_>) -> Ident {
        Ident::new(self.name, tcx.def_ident_span(self.def_id).unwrap())
    }

    /// Was this variant obtained as part of recovering from a syntactic error?
    #[inline]
    pub fn has_errors(&self) -> Result<(), ErrorGuaranteed> {
        self.tainted.map_or(Ok(()), Err)
    }

    #[inline]
    pub fn ctor_kind(&self) -> Option<CtorKind> {
        self.ctor.map(|(kind, _)| kind)
    }

    #[inline]
    pub fn ctor_def_id(&self) -> Option<DefId> {
        self.ctor.map(|(_, def_id)| def_id)
    }

    /// Returns the one field in this variant.
    ///
    /// `panic!`s if there are no fields or multiple fields.
    #[inline]
    pub fn single_field(&self) -> &FieldDef {
        assert!(self.fields.len() == 1);

        &self.fields[FieldIdx::ZERO]
    }

    /// Returns the last field in this variant, if present.
    #[inline]
    pub fn tail_opt(&self) -> Option<&FieldDef> {
        self.fields.raw.last()
    }

    /// Returns the last field in this variant.
    ///
    /// # Panics
    ///
    /// Panics, if the variant has no fields.
    #[inline]
    pub fn tail(&self) -> &FieldDef {
        self.tail_opt().expect("expected unsized ADT to have a tail field")
    }

    /// Returns whether this variant has unsafe fields.
    pub fn has_unsafe_fields(&self) -> bool {
        self.fields.iter().any(|x| x.safety.is_unsafe())
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

        let Self {
            def_id: lhs_def_id,
            ctor: _,
            name: _,
            discr: _,
            fields: _,
            flags: _,
            tainted: _,
        } = &self;
        let Self {
            def_id: rhs_def_id,
            ctor: _,
            name: _,
            discr: _,
            fields: _,
            flags: _,
            tainted: _,
        } = other;

        let res = lhs_def_id == rhs_def_id;

        // Double check that implicit assumption detailed above.
        if cfg!(debug_assertions) && res {
            let deep = self.ctor == other.ctor
                && self.name == other.name
                && self.discr == other.discr
                && self.fields == other.fields
                && self.flags == other.flags;
            assert!(deep, "VariantDef for the same def-id has differing data");
        }

        res
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

        let Self { def_id, ctor: _, name: _, discr: _, fields: _, flags: _, tainted: _ } = &self;
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
    pub safety: hir::Safety,
    pub value: Option<DefId>,
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

        let Self { did: lhs_did, name: _, vis: _, safety: _, value: _ } = &self;

        let Self { did: rhs_did, name: _, vis: _, safety: _, value: _ } = other;

        let res = lhs_did == rhs_did;

        // Double check that implicit assumption detailed above.
        if cfg!(debug_assertions) && res {
            let deep =
                self.name == other.name && self.vis == other.vis && self.safety == other.safety;
            assert!(deep, "FieldDef for the same def-id has differing data");
        }

        res
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

        let Self { did, name: _, vis: _, safety: _, value: _ } = &self;

        did.hash(s)
    }
}

impl<'tcx> FieldDef {
    /// Returns the type of this field. The resulting type is not normalized. The `arg` is
    /// typically obtained via the second field of [`TyKind::Adt`].
    pub fn ty(&self, tcx: TyCtxt<'tcx>, args: GenericArgsRef<'tcx>) -> Ty<'tcx> {
        tcx.type_of(self.did).instantiate(tcx, args)
    }

    /// Computes the `Ident` of this variant by looking up the `Span`
    pub fn ident(&self, tcx: TyCtxt<'_>) -> Ident {
        Ident::new(self.name, tcx.def_ident_span(self.did).unwrap())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ImplOverlapKind {
    /// These impls are always allowed to overlap.
    Permitted {
        /// Whether or not the impl is permitted due to the trait being a `#[marker]` trait
        marker: bool,
    },
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
        self.typeck(self.hir_body_owner_def_id(body))
    }

    pub fn provided_trait_methods(self, id: DefId) -> impl 'tcx + Iterator<Item = &'tcx AssocItem> {
        self.associated_items(id)
            .in_definition_order()
            .filter(move |item| item.is_fn() && item.defaultness(self).has_value())
    }

    pub fn repr_options_of_def(self, did: LocalDefId) -> ReprOptions {
        let mut flags = ReprFlags::empty();
        let mut size = None;
        let mut max_align: Option<Align> = None;
        let mut min_pack: Option<Align> = None;

        // Generate a deterministically-derived seed from the item's path hash
        // to allow for cross-crate compilation to actually work
        let mut field_shuffle_seed = self.def_path_hash(did.to_def_id()).0.to_smaller_hash();

        // If the user defined a custom seed for layout randomization, xor the item's
        // path hash with the user defined seed, this will allowing determinism while
        // still allowing users to further randomize layout generation for e.g. fuzzing
        if let Some(user_seed) = self.sess.opts.unstable_opts.layout_seed {
            field_shuffle_seed ^= user_seed;
        }

        if let Some(reprs) = attr::find_attr!(self.get_all_attrs(did), AttributeKind::Repr(r) => r)
        {
            for (r, _) in reprs {
                flags.insert(match *r {
                    attr::ReprRust => ReprFlags::empty(),
                    attr::ReprC => ReprFlags::IS_C,
                    attr::ReprPacked(pack) => {
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
                        max_align = max_align.max(Some(align));
                        ReprFlags::empty()
                    }
                    attr::ReprEmpty => {
                        /* skip these, they're just for diagnostics */
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

        // box is special, on the one hand the compiler assumes an ordered layout, with the pointer
        // always at offset zero. On the other hand we want scalar abi optimizations.
        let is_box = self.is_lang_item(did.to_def_id(), LangItem::OwnedBox);

        // This is here instead of layout because the choice must make it into metadata.
        if is_box {
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
        let span = self
            .def_ident_span(def_id)
            .unwrap_or_else(|| bug!("missing ident span for {def_id:?}"));
        Some(Ident::new(def, span))
    }

    /// Look up the name and span of a definition.
    ///
    /// See [`item_name`][Self::item_name] for more information.
    pub fn item_ident(self, def_id: DefId) -> Ident {
        self.opt_item_ident(def_id).unwrap_or_else(|| {
            bug!("item_ident: no name for {:?}", self.def_path(def_id));
        })
    }

    pub fn opt_associated_item(self, def_id: DefId) -> Option<AssocItem> {
        if let DefKind::AssocConst | DefKind::AssocFn | DefKind::AssocTy = self.def_kind(def_id) {
            Some(self.associated_item(def_id))
        } else {
            None
        }
    }

    /// If the `def_id` is an associated type that was desugared from a
    /// return-position `impl Trait` from a trait, then provide the source info
    /// about where that RPITIT came from.
    pub fn opt_rpitit_info(self, def_id: DefId) -> Option<ImplTraitInTraitData> {
        if let DefKind::AssocTy = self.def_kind(def_id)
            && let AssocKind::Type { data: AssocTypeData::Rpitit(rpitit_info) } =
                self.associated_item(def_id).kind
        {
            Some(rpitit_info)
        } else {
            None
        }
    }

    pub fn find_field_index(self, ident: Ident, variant: &VariantDef) -> Option<FieldIdx> {
        variant.fields.iter_enumerated().find_map(|(i, field)| {
            self.hygienic_eq(ident, field.ident(self), variant.def_id).then_some(i)
        })
    }

    /// Returns `Some` if the impls are the same polarity and the trait either
    /// has no items or is annotated `#[marker]` and prevents item overrides.
    #[instrument(level = "debug", skip(self), ret)]
    pub fn impls_are_allowed_to_overlap(
        self,
        def_id1: DefId,
        def_id2: DefId,
    ) -> Option<ImplOverlapKind> {
        let impl1 = self.impl_trait_header(def_id1).unwrap();
        let impl2 = self.impl_trait_header(def_id2).unwrap();

        let trait_ref1 = impl1.trait_ref.skip_binder();
        let trait_ref2 = impl2.trait_ref.skip_binder();

        // If either trait impl references an error, they're allowed to overlap,
        // as one of them essentially doesn't exist.
        if trait_ref1.references_error() || trait_ref2.references_error() {
            return Some(ImplOverlapKind::Permitted { marker: false });
        }

        match (impl1.polarity, impl2.polarity) {
            (ImplPolarity::Reservation, _) | (_, ImplPolarity::Reservation) => {
                // `#[rustc_reservation_impl]` impls don't overlap with anything
                return Some(ImplOverlapKind::Permitted { marker: false });
            }
            (ImplPolarity::Positive, ImplPolarity::Negative)
            | (ImplPolarity::Negative, ImplPolarity::Positive) => {
                // `impl AutoTrait for Type` + `impl !AutoTrait for Type`
                return None;
            }
            (ImplPolarity::Positive, ImplPolarity::Positive)
            | (ImplPolarity::Negative, ImplPolarity::Negative) => {}
        };

        let is_marker_impl = |trait_ref: TraitRef<'_>| self.trait_def(trait_ref.def_id).is_marker;
        let is_marker_overlap = is_marker_impl(trait_ref1) && is_marker_impl(trait_ref2);

        if is_marker_overlap {
            return Some(ImplOverlapKind::Permitted { marker: true });
        }

        None
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

    /// Returns the possibly-auto-generated MIR of a [`ty::InstanceKind`].
    #[instrument(skip(self), level = "debug")]
    pub fn instance_mir(self, instance: ty::InstanceKind<'tcx>) -> &'tcx Body<'tcx> {
        match instance {
            ty::InstanceKind::Item(def) => {
                debug!("calling def_kind on def: {:?}", def);
                let def_kind = self.def_kind(def);
                debug!("returned from def_kind: {:?}", def_kind);
                match def_kind {
                    DefKind::Const
                    | DefKind::Static { .. }
                    | DefKind::AssocConst
                    | DefKind::Ctor(..)
                    | DefKind::AnonConst
                    | DefKind::InlineConst => self.mir_for_ctfe(def),
                    // If the caller wants `mir_for_ctfe` of a function they should not be using
                    // `instance_mir`, so we'll assume const fn also wants the optimized version.
                    _ => self.optimized_mir(def),
                }
            }
            ty::InstanceKind::VTableShim(..)
            | ty::InstanceKind::ReifyShim(..)
            | ty::InstanceKind::Intrinsic(..)
            | ty::InstanceKind::FnPtrShim(..)
            | ty::InstanceKind::Virtual(..)
            | ty::InstanceKind::ClosureOnceShim { .. }
            | ty::InstanceKind::ConstructCoroutineInClosureShim { .. }
            | ty::InstanceKind::FutureDropPollShim(..)
            | ty::InstanceKind::DropGlue(..)
            | ty::InstanceKind::CloneShim(..)
            | ty::InstanceKind::ThreadLocalShim(..)
            | ty::InstanceKind::FnPtrAddrShim(..)
            | ty::InstanceKind::AsyncDropGlueCtorShim(..)
            | ty::InstanceKind::AsyncDropGlue(..) => self.mir_shims(instance),
        }
    }

    // FIXME(@lcnr): Remove this function.
    pub fn get_attrs_unchecked(self, did: DefId) -> &'tcx [hir::Attribute] {
        if let Some(did) = did.as_local() {
            self.hir_attrs(self.local_def_id_to_hir_id(did))
        } else {
            self.attrs_for_def(did)
        }
    }

    /// Gets all attributes with the given name.
    pub fn get_attrs(
        self,
        did: impl Into<DefId>,
        attr: Symbol,
    ) -> impl Iterator<Item = &'tcx hir::Attribute> {
        self.get_all_attrs(did).filter(move |a: &&hir::Attribute| a.has_name(attr))
    }

    /// Gets all attributes.
    ///
    /// To see if an item has a specific attribute, you should use [`rustc_attr_data_structures::find_attr!`] so you can use matching.
    pub fn get_all_attrs(
        self,
        did: impl Into<DefId>,
    ) -> impl Iterator<Item = &'tcx hir::Attribute> {
        let did: DefId = did.into();
        if let Some(did) = did.as_local() {
            self.hir_attrs(self.local_def_id_to_hir_id(did)).iter()
        } else {
            self.attrs_for_def(did).iter()
        }
    }

    /// Get an attribute from the diagnostic attribute namespace
    ///
    /// This function requests an attribute with the following structure:
    ///
    /// `#[diagnostic::$attr]`
    ///
    /// This function performs feature checking, so if an attribute is returned
    /// it can be used by the consumer
    pub fn get_diagnostic_attr(
        self,
        did: impl Into<DefId>,
        attr: Symbol,
    ) -> Option<&'tcx hir::Attribute> {
        let did: DefId = did.into();
        if did.as_local().is_some() {
            // it's a crate local item, we need to check feature flags
            if rustc_feature::is_stable_diagnostic_attribute(attr, self.features()) {
                self.get_attrs_by_path(did, &[sym::diagnostic, sym::do_not_recommend]).next()
            } else {
                None
            }
        } else {
            // we filter out unstable diagnostic attributes before
            // encoding attributes
            debug_assert!(rustc_feature::encode_cross_crate(attr));
            self.attrs_for_def(did)
                .iter()
                .find(|a| matches!(a.path().as_ref(), [sym::diagnostic, a] if *a == attr))
        }
    }

    pub fn get_attrs_by_path(
        self,
        did: DefId,
        attr: &[Symbol],
    ) -> impl Iterator<Item = &'tcx hir::Attribute> {
        let filter_fn = move |a: &&hir::Attribute| a.path_matches(attr);
        if let Some(did) = did.as_local() {
            self.hir_attrs(self.local_def_id_to_hir_id(did)).iter().filter(filter_fn)
        } else {
            self.attrs_for_def(did).iter().filter(filter_fn)
        }
    }

    pub fn get_attr(self, did: impl Into<DefId>, attr: Symbol) -> Option<&'tcx hir::Attribute> {
        if cfg!(debug_assertions) && !rustc_feature::is_valid_for_get_attr(attr) {
            let did: DefId = did.into();
            bug!("get_attr: unexpected called with DefId `{:?}`, attr `{:?}`", did, attr);
        } else {
            self.get_attrs(did, attr).next()
        }
    }

    /// Determines whether an item is annotated with an attribute.
    pub fn has_attr(self, did: impl Into<DefId>, attr: Symbol) -> bool {
        self.get_attrs(did, attr).next().is_some()
    }

    /// Determines whether an item is annotated with a multi-segment attribute
    pub fn has_attrs_with_path(self, did: impl Into<DefId>, attrs: &[Symbol]) -> bool {
        self.get_attrs_by_path(did.into(), attrs).next().is_some()
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

    /// Returns layout of a non-async-drop coroutine. Layout might be unavailable if the
    /// coroutine is tainted by errors.
    ///
    /// Takes `coroutine_kind` which can be acquired from the `CoroutineArgs::kind_ty`,
    /// e.g. `args.as_coroutine().kind_ty()`.
    fn ordinary_coroutine_layout(
        self,
        def_id: DefId,
        coroutine_kind_ty: Ty<'tcx>,
    ) -> Option<&'tcx CoroutineLayout<'tcx>> {
        let mir = self.optimized_mir(def_id);
        // Regular coroutine
        if coroutine_kind_ty.is_unit() {
            mir.coroutine_layout_raw()
        } else {
            // If we have a `Coroutine` that comes from an coroutine-closure,
            // then it may be a by-move or by-ref body.
            let ty::Coroutine(_, identity_args) =
                *self.type_of(def_id).instantiate_identity().kind()
            else {
                unreachable!();
            };
            let identity_kind_ty = identity_args.as_coroutine().kind_ty();
            // If the types differ, then we must be getting the by-move body of
            // a by-ref coroutine.
            if identity_kind_ty == coroutine_kind_ty {
                mir.coroutine_layout_raw()
            } else {
                assert_matches!(coroutine_kind_ty.to_opt_closure_kind(), Some(ClosureKind::FnOnce));
                assert_matches!(
                    identity_kind_ty.to_opt_closure_kind(),
                    Some(ClosureKind::Fn | ClosureKind::FnMut)
                );
                self.optimized_mir(self.coroutine_by_move_body_def_id(def_id))
                    .coroutine_layout_raw()
            }
        }
    }

    /// Returns layout of a `async_drop_in_place::{closure}` coroutine
    ///   (returned from `async fn async_drop_in_place<T>(..)`).
    /// Layout might be unavailable if the coroutine is tainted by errors.
    fn async_drop_coroutine_layout(
        self,
        def_id: DefId,
        args: GenericArgsRef<'tcx>,
    ) -> Option<&'tcx CoroutineLayout<'tcx>> {
        if args[0].has_placeholders() || args[0].has_non_region_param() {
            return None;
        }
        let instance = InstanceKind::AsyncDropGlue(def_id, Ty::new_coroutine(self, def_id, args));
        self.mir_shims(instance).coroutine_layout_raw()
    }

    /// Returns layout of a coroutine. Layout might be unavailable if the
    /// coroutine is tainted by errors.
    pub fn coroutine_layout(
        self,
        def_id: DefId,
        args: GenericArgsRef<'tcx>,
    ) -> Option<&'tcx CoroutineLayout<'tcx>> {
        if self.is_async_drop_in_place_coroutine(def_id) {
            // layout of `async_drop_in_place<T>::{closure}` in case,
            // when T is a coroutine, contains this internal coroutine's ptr in upvars
            // and doesn't require any locals. Here is an `empty coroutine's layout`
            let arg_cor_ty = args.first().unwrap().expect_ty();
            if arg_cor_ty.is_coroutine() {
                let span = self.def_span(def_id);
                let source_info = SourceInfo::outermost(span);
                // Even minimal, empty coroutine has 3 states (RESERVED_VARIANTS),
                // so variant_fields and variant_source_info should have 3 elements.
                let variant_fields: IndexVec<VariantIdx, IndexVec<FieldIdx, CoroutineSavedLocal>> =
                    iter::repeat(IndexVec::new()).take(CoroutineArgs::RESERVED_VARIANTS).collect();
                let variant_source_info: IndexVec<VariantIdx, SourceInfo> =
                    iter::repeat(source_info).take(CoroutineArgs::RESERVED_VARIANTS).collect();
                let proxy_layout = CoroutineLayout {
                    field_tys: [].into(),
                    field_names: [].into(),
                    variant_fields,
                    variant_source_info,
                    storage_conflicts: BitMatrix::new(0, 0),
                };
                return Some(self.arena.alloc(proxy_layout));
            } else {
                self.async_drop_coroutine_layout(def_id, args)
            }
        } else {
            self.ordinary_coroutine_layout(def_id, args.as_coroutine().kind_ty())
        }
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

    pub fn is_exportable(self, def_id: DefId) -> bool {
        self.exportable_items(def_id.krate).contains(&def_id)
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
    pub fn hygienic_eq(self, use_ident: Ident, def_ident: Ident, def_parent_def_id: DefId) -> bool {
        // We could use `Ident::eq` here, but we deliberately don't. The identifier
        // comparison fails frequently, and we want to avoid the expensive
        // `normalize_to_macros_2_0()` calls required for the span comparison whenever possible.
        use_ident.name == def_ident.name
            && use_ident
                .span
                .ctxt()
                .hygienic_eq(def_ident.span.ctxt(), self.expn_that_defined(def_parent_def_id))
    }

    pub fn adjust_ident(self, mut ident: Ident, scope: DefId) -> Ident {
        ident.span.normalize_to_macros_2_0_and_adjust(self.expn_that_defined(scope));
        ident
    }

    // FIXME(vincenzopalazzo): move the HirId to a LocalDefId
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

    /// Checks whether this is a `const fn`. Returns `false` for non-functions.
    ///
    /// Even if this returns `true`, constness may still be unstable!
    #[inline]
    pub fn is_const_fn(self, def_id: DefId) -> bool {
        matches!(
            self.def_kind(def_id),
            DefKind::Fn | DefKind::AssocFn | DefKind::Ctor(_, CtorKind::Fn) | DefKind::Closure
        ) && self.constness(def_id) == hir::Constness::Const
    }

    /// Whether this item is conditionally constant for the purposes of the
    /// effects implementation.
    ///
    /// This roughly corresponds to all const functions and other callable
    /// items, along with const impls and traits, and associated types within
    /// those impls and traits.
    pub fn is_conditionally_const(self, def_id: impl Into<DefId>) -> bool {
        let def_id: DefId = def_id.into();
        match self.def_kind(def_id) {
            DefKind::Impl { of_trait: true } => {
                let header = self.impl_trait_header(def_id).unwrap();
                header.constness == hir::Constness::Const
                    && self.is_const_trait(header.trait_ref.skip_binder().def_id)
            }
            DefKind::Fn | DefKind::Ctor(_, CtorKind::Fn) => {
                self.constness(def_id) == hir::Constness::Const
            }
            DefKind::Trait => self.is_const_trait(def_id),
            DefKind::AssocTy => {
                let parent_def_id = self.parent(def_id);
                match self.def_kind(parent_def_id) {
                    DefKind::Impl { of_trait: false } => false,
                    DefKind::Impl { of_trait: true } | DefKind::Trait => {
                        self.is_conditionally_const(parent_def_id)
                    }
                    _ => bug!("unexpected parent item of associated type: {parent_def_id:?}"),
                }
            }
            DefKind::AssocFn => {
                let parent_def_id = self.parent(def_id);
                match self.def_kind(parent_def_id) {
                    DefKind::Impl { of_trait: false } => {
                        self.constness(def_id) == hir::Constness::Const
                    }
                    DefKind::Impl { of_trait: true } | DefKind::Trait => {
                        self.is_conditionally_const(parent_def_id)
                    }
                    _ => bug!("unexpected parent item of associated fn: {parent_def_id:?}"),
                }
            }
            DefKind::OpaqueTy => match self.opaque_ty_origin(def_id) {
                hir::OpaqueTyOrigin::FnReturn { parent, .. } => self.is_conditionally_const(parent),
                hir::OpaqueTyOrigin::AsyncFn { .. } => false,
                // FIXME(const_trait_impl): ATPITs could be conditionally const?
                hir::OpaqueTyOrigin::TyAlias { .. } => false,
            },
            DefKind::Closure => {
                // Closures and RPITs will eventually have const conditions
                // for `~const` bounds.
                false
            }
            DefKind::Ctor(_, CtorKind::Const)
            | DefKind::Impl { of_trait: false }
            | DefKind::Mod
            | DefKind::Struct
            | DefKind::Union
            | DefKind::Enum
            | DefKind::Variant
            | DefKind::TyAlias
            | DefKind::ForeignTy
            | DefKind::TraitAlias
            | DefKind::TyParam
            | DefKind::Const
            | DefKind::ConstParam
            | DefKind::Static { .. }
            | DefKind::AssocConst
            | DefKind::Macro(_)
            | DefKind::ExternCrate
            | DefKind::Use
            | DefKind::ForeignMod
            | DefKind::AnonConst
            | DefKind::InlineConst
            | DefKind::Field
            | DefKind::LifetimeParam
            | DefKind::GlobalAsm
            | DefKind::SyntheticCoroutineBody => false,
        }
    }

    #[inline]
    pub fn is_const_trait(self, def_id: DefId) -> bool {
        self.trait_def(def_id).constness == hir::Constness::Const
    }

    #[inline]
    pub fn is_const_default_method(self, def_id: DefId) -> bool {
        matches!(self.trait_of_item(def_id), Some(trait_id) if self.is_const_trait(trait_id))
    }

    pub fn impl_method_has_trait_impl_trait_tys(self, def_id: DefId) -> bool {
        if self.def_kind(def_id) != DefKind::AssocFn {
            return false;
        }

        let Some(item) = self.opt_associated_item(def_id) else {
            return false;
        };
        if item.container != ty::AssocItemContainer::Impl {
            return false;
        }

        let Some(trait_item_def_id) = item.trait_item_def_id else {
            return false;
        };

        return !self
            .associated_types_for_impl_traits_in_associated_fn(trait_item_def_id)
            .is_empty();
    }
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
        ast::FloatTy::F16 => FloatTy::F16,
        ast::FloatTy::F32 => FloatTy::F32,
        ast::FloatTy::F64 => FloatTy::F64,
        ast::FloatTy::F128 => FloatTy::F128,
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

pub fn provide(providers: &mut Providers) {
    closure::provide(providers);
    context::provide(providers);
    erase_regions::provide(providers);
    inhabitedness::provide(providers);
    util::provide(providers);
    print::provide(providers);
    super::util::bug::provide(providers);
    *providers = Providers {
        trait_impls_of: trait_def::trait_impls_of_provider,
        incoherent_impls: trait_def::incoherent_impls_provider,
        trait_impls_in_crate: trait_def::trait_impls_in_crate_provider,
        traits: trait_def::traits_provider,
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
    pub inherent_impls: FxIndexMap<LocalDefId, Vec<DefId>>,
    pub incoherent_impls: FxIndexMap<SimplifiedType, Vec<LocalDefId>>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, TyEncodable, HashStable)]
pub struct SymbolName<'tcx> {
    /// `&str` gives a consistent ordering, which ensures reproducible builds.
    pub name: &'tcx str,
}

impl<'tcx> SymbolName<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, name: &str) -> SymbolName<'tcx> {
        SymbolName { name: tcx.arena.alloc_str(name) }
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
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use rustc_data_structures::static_assert_size;

    use super::*;
    // tidy-alphabetical-start
    static_assert_size!(PredicateKind<'_>, 32);
    static_assert_size!(WithCachedTypeInfo<TyKind<'_>>, 48);
    // tidy-alphabetical-end
}
