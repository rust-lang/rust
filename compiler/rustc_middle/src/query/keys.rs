//! Defines the set of legal keys that can be used in queries.

use std::ffi::OsStr;
use std::fmt::Debug;
use std::hash::Hash;

use rustc_ast::tokenstream::TokenStream;
use rustc_data_structures::stable_hasher::HashStable;
use rustc_hir::def_id::{CrateNum, DefId, LOCAL_CRATE, LocalDefId, LocalModDefId};
use rustc_hir::hir_id::OwnerId;
use rustc_span::{DUMMY_SP, Ident, LocalExpnId, Span, Symbol};

use crate::dep_graph::DepNodeIndex;
use crate::ich::StableHashingContext;
use crate::infer::canonical::CanonicalQueryInput;
use crate::mir::mono::CollectionMode;
use crate::query::{DefIdCache, DefaultCache, SingleCache, VecCache};
use crate::ty::fast_reject::SimplifiedType;
use crate::ty::layout::ValidityRequirement;
use crate::ty::{self, GenericArg, GenericArgsRef, Ty, TyCtxt};
use crate::{mir, traits};

/// Placeholder for `CrateNum`'s "local" counterpart
#[derive(Copy, Clone, Debug)]
pub struct LocalCrate;

pub trait QueryKeyBounds = Copy + Debug + Eq + Hash + for<'a> HashStable<StableHashingContext<'a>>;

/// Controls what types can legally be used as the key for a query.
pub trait QueryKey: Sized + QueryKeyBounds {
    /// The type of in-memory cache to use for queries with this key type.
    ///
    /// In practice the cache type must implement [`QueryCache`], though that
    /// constraint is not enforced here.
    ///
    /// [`QueryCache`]: rustc_middle::query::QueryCache
    type Cache<V> = DefaultCache<Self, V>;

    /// In the event that a cycle occurs, if no explicit span has been
    /// given for a query with key `self`, what span should we use?
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span;

    /// If the key is a [`DefId`] or `DefId`--equivalent, return that `DefId`.
    /// Otherwise, return `None`.
    fn key_as_def_id(&self) -> Option<DefId> {
        None
    }

    /// Used to detect when ADT def ids are used as keys in a cycle for better error reporting.
    fn def_id_for_ty_in_cycle(&self) -> Option<DefId> {
        None
    }
}

pub trait AsLocalQueryKey: QueryKey {
    type LocalQueryKey;

    /// Given an instance of this key, what crate is it referring to?
    /// This is used to find the provider.
    fn as_local_key(&self) -> Option<Self::LocalQueryKey>;
}

impl QueryKey for () {
    type Cache<V> = SingleCache<V>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> QueryKey for ty::InstanceKind<'tcx> {
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> QueryKey for ty::Instance<'tcx> {
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> QueryKey for mir::interpret::GlobalId<'tcx> {
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.instance.default_span(tcx)
    }
}

impl<'tcx> QueryKey for (Ty<'tcx>, Option<ty::ExistentialTraitRef<'tcx>>) {
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> QueryKey for ty::LitToConstInput<'tcx> {
    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl QueryKey for CrateNum {
    type Cache<V> = VecCache<Self, V, DepNodeIndex>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl AsLocalQueryKey for CrateNum {
    type LocalQueryKey = LocalCrate;

    #[inline(always)]
    fn as_local_key(&self) -> Option<Self::LocalQueryKey> {
        (*self == LOCAL_CRATE).then_some(LocalCrate)
    }
}

impl QueryKey for OwnerId {
    type Cache<V> = VecCache<Self, V, DepNodeIndex>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.to_def_id().default_span(tcx)
    }

    fn key_as_def_id(&self) -> Option<DefId> {
        Some(self.to_def_id())
    }
}

impl QueryKey for LocalDefId {
    type Cache<V> = VecCache<Self, V, DepNodeIndex>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.to_def_id().default_span(tcx)
    }

    fn key_as_def_id(&self) -> Option<DefId> {
        Some(self.to_def_id())
    }
}

impl QueryKey for DefId {
    type Cache<V> = DefIdCache<V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(*self)
    }

    #[inline(always)]
    fn key_as_def_id(&self) -> Option<DefId> {
        Some(*self)
    }
}

impl AsLocalQueryKey for DefId {
    type LocalQueryKey = LocalDefId;

    #[inline(always)]
    fn as_local_key(&self) -> Option<Self::LocalQueryKey> {
        self.as_local()
    }
}

impl QueryKey for LocalModDefId {
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(*self)
    }

    #[inline(always)]
    fn key_as_def_id(&self) -> Option<DefId> {
        Some(self.to_def_id())
    }
}

impl QueryKey for SimplifiedType {
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl QueryKey for (DefId, DefId) {
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.1.default_span(tcx)
    }
}

impl QueryKey for (DefId, Ident) {
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.0)
    }

    #[inline(always)]
    fn key_as_def_id(&self) -> Option<DefId> {
        Some(self.0)
    }
}

impl QueryKey for (LocalDefId, LocalDefId, Ident) {
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.1.default_span(tcx)
    }
}

impl QueryKey for (CrateNum, DefId) {
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.1.default_span(tcx)
    }
}

impl AsLocalQueryKey for (CrateNum, DefId) {
    type LocalQueryKey = DefId;

    #[inline(always)]
    fn as_local_key(&self) -> Option<Self::LocalQueryKey> {
        (self.0 == LOCAL_CRATE).then(|| self.1)
    }
}

impl QueryKey for (CrateNum, SimplifiedType) {
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl AsLocalQueryKey for (CrateNum, SimplifiedType) {
    type LocalQueryKey = SimplifiedType;

    #[inline(always)]
    fn as_local_key(&self) -> Option<Self::LocalQueryKey> {
        (self.0 == LOCAL_CRATE).then(|| self.1)
    }
}

impl QueryKey for (DefId, ty::SizedTraitKind) {
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> QueryKey for GenericArgsRef<'tcx> {
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> QueryKey for (DefId, GenericArgsRef<'tcx>) {
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> QueryKey for ty::TraitRef<'tcx> {
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.def_id)
    }
}

impl<'tcx> QueryKey for GenericArg<'tcx> {
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> QueryKey for Ty<'tcx> {
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }

    fn def_id_for_ty_in_cycle(&self) -> Option<DefId> {
        match *self.kind() {
            ty::Adt(adt, _) => Some(adt.did()),
            ty::Coroutine(def_id, ..) => Some(def_id),
            _ => None,
        }
    }
}

impl<'tcx> QueryKey for (Ty<'tcx>, Ty<'tcx>) {
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> QueryKey for ty::Clauses<'tcx> {
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx, T: QueryKey> QueryKey for ty::PseudoCanonicalInput<'tcx, T> {
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.value.default_span(tcx)
    }

    fn def_id_for_ty_in_cycle(&self) -> Option<DefId> {
        self.value.def_id_for_ty_in_cycle()
    }
}

impl QueryKey for Symbol {
    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl QueryKey for Option<Symbol> {
    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> QueryKey for &'tcx OsStr {
    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

/// Canonical query goals correspond to abstract trait operations that
/// are not tied to any crate in particular.
impl<'tcx, T: QueryKeyBounds> QueryKey for CanonicalQueryInput<'tcx, T> {
    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx, T: QueryKeyBounds> QueryKey for (CanonicalQueryInput<'tcx, T>, bool) {
    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> QueryKey for (Ty<'tcx>, rustc_abi::VariantIdx) {
    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> QueryKey for (ty::Predicate<'tcx>, traits::WellFormedLoc) {
    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> QueryKey for (ty::PolyFnSig<'tcx>, &'tcx ty::List<Ty<'tcx>>) {
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> QueryKey for (ty::Instance<'tcx>, &'tcx ty::List<Ty<'tcx>>) {
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> QueryKey for ty::Value<'tcx> {
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> QueryKey for (LocalExpnId, &'tcx TokenStream) {
    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        self.0.expn_data().call_site
    }
}

impl<'tcx> QueryKey for (ValidityRequirement, ty::PseudoCanonicalInput<'tcx, Ty<'tcx>>) {
    // Just forward to `Ty<'tcx>`

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }

    fn def_id_for_ty_in_cycle(&self) -> Option<DefId> {
        match self.1.value.kind() {
            ty::Adt(adt, _) => Some(adt.did()),
            _ => None,
        }
    }
}

impl<'tcx> QueryKey for (ty::Instance<'tcx>, CollectionMode) {
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}
