//! Defines the set of legal keys that can be used in queries.

use std::ffi::OsStr;

use rustc_hir::def_id::{CrateNum, DefId, LOCAL_CRATE, LocalDefId, LocalModDefId, ModDefId};
use rustc_hir::hir_id::{HirId, OwnerId};
use rustc_query_system::dep_graph::DepNodeIndex;
use rustc_query_system::query::{DefIdCache, DefaultCache, SingleCache, VecCache};
use rustc_span::{DUMMY_SP, Ident, Span, Symbol};

use crate::infer::canonical::CanonicalQueryInput;
use crate::mir::mono::CollectionMode;
use crate::ty::fast_reject::SimplifiedType;
use crate::ty::layout::{TyAndLayout, ValidityRequirement};
use crate::ty::{self, GenericArg, GenericArgsRef, Ty, TyCtxt};
use crate::{mir, traits};

/// Placeholder for `CrateNum`'s "local" counterpart
#[derive(Copy, Clone, Debug)]
pub struct LocalCrate;

/// The `Key` trait controls what types can legally be used as the key
/// for a query.
pub trait Key: Sized {
    /// The type of in-memory cache to use for queries with this key type.
    ///
    /// In practice the cache type must implement [`QueryCache`], though that
    /// constraint is not enforced here.
    ///
    /// [`QueryCache`]: rustc_query_system::query::QueryCache
    // N.B. Most of the keys down below have `type Cache<V> = DefaultCache<Self, V>;`,
    //      it would be reasonable to use associated type defaults, to remove the duplication...
    //
    //      ...But r-a doesn't support them yet and using a default here causes r-a to not infer
    //      return types of queries which is very annoying. Thus, until r-a support associated
    //      type defaults, please restrain from using them here <3
    //
    //      r-a issue: <https://github.com/rust-lang/rust-analyzer/issues/13693>
    type Cache<V>;

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

pub trait AsLocalKey: Key {
    type LocalKey;

    /// Given an instance of this key, what crate is it referring to?
    /// This is used to find the provider.
    fn as_local_key(&self) -> Option<Self::LocalKey>;
}

impl Key for () {
    type Cache<V> = SingleCache<V>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for ty::InstanceKind<'tcx> {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> AsLocalKey for ty::InstanceKind<'tcx> {
    type LocalKey = Self;

    #[inline(always)]
    fn as_local_key(&self) -> Option<Self::LocalKey> {
        self.def_id().is_local().then(|| *self)
    }
}

impl<'tcx> Key for ty::Instance<'tcx> {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> Key for mir::interpret::GlobalId<'tcx> {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.instance.default_span(tcx)
    }
}

impl<'tcx> Key for (Ty<'tcx>, Option<ty::ExistentialTraitRef<'tcx>>) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for mir::interpret::LitToConstInput<'tcx> {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl Key for CrateNum {
    type Cache<V> = VecCache<Self, V, DepNodeIndex>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl AsLocalKey for CrateNum {
    type LocalKey = LocalCrate;

    #[inline(always)]
    fn as_local_key(&self) -> Option<Self::LocalKey> {
        (*self == LOCAL_CRATE).then_some(LocalCrate)
    }
}

impl Key for OwnerId {
    type Cache<V> = VecCache<Self, V, DepNodeIndex>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.to_def_id().default_span(tcx)
    }

    fn key_as_def_id(&self) -> Option<DefId> {
        Some(self.to_def_id())
    }
}

impl Key for LocalDefId {
    type Cache<V> = VecCache<Self, V, DepNodeIndex>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.to_def_id().default_span(tcx)
    }

    fn key_as_def_id(&self) -> Option<DefId> {
        Some(self.to_def_id())
    }
}

impl Key for DefId {
    type Cache<V> = DefIdCache<V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(*self)
    }

    #[inline(always)]
    fn key_as_def_id(&self) -> Option<DefId> {
        Some(*self)
    }
}

impl AsLocalKey for DefId {
    type LocalKey = LocalDefId;

    #[inline(always)]
    fn as_local_key(&self) -> Option<Self::LocalKey> {
        self.as_local()
    }
}

impl Key for LocalModDefId {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(*self)
    }

    #[inline(always)]
    fn key_as_def_id(&self) -> Option<DefId> {
        Some(self.to_def_id())
    }
}

impl Key for ModDefId {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(*self)
    }

    #[inline(always)]
    fn key_as_def_id(&self) -> Option<DefId> {
        Some(self.to_def_id())
    }
}

impl AsLocalKey for ModDefId {
    type LocalKey = LocalModDefId;

    #[inline(always)]
    fn as_local_key(&self) -> Option<Self::LocalKey> {
        self.as_local()
    }
}

impl Key for SimplifiedType {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl Key for (DefId, DefId) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.1.default_span(tcx)
    }
}

impl<'tcx> Key for (ty::Instance<'tcx>, LocalDefId) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl Key for (DefId, LocalDefId) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.1.default_span(tcx)
    }
}

impl Key for (LocalDefId, DefId) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl Key for (LocalDefId, LocalDefId) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl Key for (DefId, Ident) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.0)
    }

    #[inline(always)]
    fn key_as_def_id(&self) -> Option<DefId> {
        Some(self.0)
    }
}

impl Key for (LocalDefId, LocalDefId, Ident) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.1.default_span(tcx)
    }
}

impl Key for (CrateNum, DefId) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.1.default_span(tcx)
    }
}

impl AsLocalKey for (CrateNum, DefId) {
    type LocalKey = DefId;

    #[inline(always)]
    fn as_local_key(&self) -> Option<Self::LocalKey> {
        (self.0 == LOCAL_CRATE).then(|| self.1)
    }
}

impl Key for (CrateNum, SimplifiedType) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl AsLocalKey for (CrateNum, SimplifiedType) {
    type LocalKey = SimplifiedType;

    #[inline(always)]
    fn as_local_key(&self) -> Option<Self::LocalKey> {
        (self.0 == LOCAL_CRATE).then(|| self.1)
    }
}

impl Key for (DefId, SimplifiedType) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl Key for (DefId, ty::SizedTraitKind) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> Key for GenericArgsRef<'tcx> {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (DefId, GenericArgsRef<'tcx>) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> Key for (ty::UnevaluatedConst<'tcx>, ty::UnevaluatedConst<'tcx>) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        (self.0).def.default_span(tcx)
    }
}

impl<'tcx> Key for (LocalDefId, DefId, GenericArgsRef<'tcx>) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> Key for (ty::ParamEnv<'tcx>, ty::TraitRef<'tcx>) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.1.def_id)
    }
}

impl<'tcx> Key for ty::ParamEnvAnd<'tcx, Ty<'tcx>> {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for ty::TraitRef<'tcx> {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.def_id)
    }
}

impl<'tcx> Key for ty::PolyTraitRef<'tcx> {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> Key for ty::PolyExistentialTraitRef<'tcx> {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> Key for (ty::PolyTraitRef<'tcx>, ty::PolyTraitRef<'tcx>) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.0.def_id())
    }
}

impl<'tcx> Key for GenericArg<'tcx> {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for ty::Const<'tcx> {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for Ty<'tcx> {
    type Cache<V> = DefaultCache<Self, V>;

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

impl<'tcx> Key for TyAndLayout<'tcx> {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (Ty<'tcx>, Ty<'tcx>) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for ty::Clauses<'tcx> {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for ty::ParamEnv<'tcx> {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx, T: Key> Key for ty::PseudoCanonicalInput<'tcx, T> {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.value.default_span(tcx)
    }

    fn def_id_for_ty_in_cycle(&self) -> Option<DefId> {
        self.value.def_id_for_ty_in_cycle()
    }
}

impl Key for Symbol {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl Key for Option<Symbol> {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for &'tcx OsStr {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

/// Canonical query goals correspond to abstract trait operations that
/// are not tied to any crate in particular.
impl<'tcx, T: Clone> Key for CanonicalQueryInput<'tcx, T> {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx, T: Clone> Key for (CanonicalQueryInput<'tcx, T>, bool) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl Key for (Symbol, u32, u32) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (DefId, Ty<'tcx>, GenericArgsRef<'tcx>, ty::ParamEnv<'tcx>) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (Ty<'tcx>, rustc_abi::VariantIdx) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (ty::Predicate<'tcx>, traits::WellFormedLoc) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (ty::PolyFnSig<'tcx>, &'tcx ty::List<Ty<'tcx>>) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (ty::Instance<'tcx>, &'tcx ty::List<Ty<'tcx>>) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> Key for ty::Value<'tcx> {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl Key for HirId {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.hir_span(*self)
    }

    #[inline(always)]
    fn key_as_def_id(&self) -> Option<DefId> {
        None
    }
}

impl Key for (LocalDefId, HirId) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.hir_span(self.1)
    }

    #[inline(always)]
    fn key_as_def_id(&self) -> Option<DefId> {
        Some(self.0.into())
    }
}

impl<'tcx> Key for (ValidityRequirement, ty::PseudoCanonicalInput<'tcx, Ty<'tcx>>) {
    type Cache<V> = DefaultCache<Self, V>;

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

impl<'tcx> Key for (ty::Instance<'tcx>, CollectionMode) {
    type Cache<V> = DefaultCache<Self, V>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}
