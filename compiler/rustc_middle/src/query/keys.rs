//! Defines the set of legal keys that can be used in queries.

use crate::infer::canonical::Canonical;
use crate::mir;
use crate::traits;
use crate::ty::fast_reject::SimplifiedType;
use crate::ty::layout::{TyAndLayout, ValidityRequirement};
use crate::ty::subst::{GenericArg, SubstsRef};
use crate::ty::{self, Ty, TyCtxt};
use rustc_hir::def_id::{CrateNum, DefId, LocalDefId, LOCAL_CRATE};
use rustc_hir::hir_id::{HirId, OwnerId};
use rustc_query_system::query::{DefaultCacheSelector, SingleCacheSelector, VecCacheSelector};
use rustc_span::symbol::{Ident, Symbol};
use rustc_span::{Span, DUMMY_SP};
use rustc_target::abi::FieldIdx;

/// Placeholder for `CrateNum`'s "local" counterpart
#[derive(Copy, Clone, Debug)]
pub struct LocalCrate;

/// The `Key` trait controls what types can legally be used as the key
/// for a query.
pub trait Key: Sized {
    // N.B. Most of the keys down below have `type CacheSelector = DefaultCacheSelector<Self>;`,
    //      it would be reasonable to use associated type defaults, to remove the duplication...
    //
    //      ...But r-a doesn't support them yet and using a default here causes r-a to not infer
    //      return types of queries which is very annoying. Thus, until r-a support associated
    //      type defaults, please restrain from using them here <3
    //
    //      r-a issue: <https://github.com/rust-lang/rust-analyzer/issues/13693>
    type CacheSelector;

    /// In the event that a cycle occurs, if no explicit span has been
    /// given for a query with key `self`, what span should we use?
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span;

    /// If the key is a [`DefId`] or `DefId`--equivalent, return that `DefId`.
    /// Otherwise, return `None`.
    fn key_as_def_id(&self) -> Option<DefId> {
        None
    }

    fn ty_adt_id(&self) -> Option<DefId> {
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
    type CacheSelector = SingleCacheSelector;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for ty::InstanceDef<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> AsLocalKey for ty::InstanceDef<'tcx> {
    type LocalKey = Self;

    #[inline(always)]
    fn as_local_key(&self) -> Option<Self::LocalKey> {
        self.def_id().is_local().then(|| *self)
    }
}

impl<'tcx> Key for ty::Instance<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> Key for mir::interpret::GlobalId<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.instance.default_span(tcx)
    }
}

impl<'tcx> Key for (Ty<'tcx>, Option<ty::PolyExistentialTraitRef<'tcx>>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for mir::interpret::LitToConstInput<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl Key for CrateNum {
    type CacheSelector = VecCacheSelector<Self>;

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
    type CacheSelector = VecCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.to_def_id().default_span(tcx)
    }

    fn key_as_def_id(&self) -> Option<DefId> {
        Some(self.to_def_id())
    }
}

impl Key for LocalDefId {
    type CacheSelector = VecCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.to_def_id().default_span(tcx)
    }

    fn key_as_def_id(&self) -> Option<DefId> {
        Some(self.to_def_id())
    }
}

impl Key for DefId {
    type CacheSelector = DefaultCacheSelector<Self>;

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

impl Key for SimplifiedType {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl Key for (DefId, DefId) {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.1.default_span(tcx)
    }
}

impl<'tcx> Key for (ty::Instance<'tcx>, LocalDefId) {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl Key for (DefId, LocalDefId) {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.1.default_span(tcx)
    }
}

impl Key for (LocalDefId, DefId) {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl Key for (LocalDefId, LocalDefId) {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl Key for (DefId, Ident) {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.0)
    }

    #[inline(always)]
    fn key_as_def_id(&self) -> Option<DefId> {
        Some(self.0)
    }
}

impl Key for (LocalDefId, LocalDefId, Ident) {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.1.default_span(tcx)
    }
}

impl Key for (CrateNum, DefId) {
    type CacheSelector = DefaultCacheSelector<Self>;

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
    type CacheSelector = DefaultCacheSelector<Self>;

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
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> Key for SubstsRef<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (DefId, SubstsRef<'tcx>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> Key for (ty::UnevaluatedConst<'tcx>, ty::UnevaluatedConst<'tcx>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        (self.0).def.default_span(tcx)
    }
}

impl<'tcx> Key for (LocalDefId, DefId, SubstsRef<'tcx>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> Key for (ty::ParamEnv<'tcx>, ty::PolyTraitRef<'tcx>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.1.def_id())
    }
}

impl<'tcx> Key for (ty::Const<'tcx>, FieldIdx) {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for mir::interpret::ConstAlloc<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for ty::PolyTraitRef<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> Key for ty::PolyExistentialTraitRef<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> Key for (ty::PolyTraitRef<'tcx>, ty::PolyTraitRef<'tcx>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.0.def_id())
    }
}

impl<'tcx> Key for GenericArg<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for mir::ConstantKind<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for ty::Const<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for Ty<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }

    fn ty_adt_id(&self) -> Option<DefId> {
        match self.kind() {
            ty::Adt(adt, _) => Some(adt.did()),
            _ => None,
        }
    }
}

impl<'tcx> Key for TyAndLayout<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (Ty<'tcx>, Ty<'tcx>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for &'tcx ty::List<ty::Clause<'tcx>> {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for ty::ParamEnv<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx, T: Key> Key for ty::ParamEnvAnd<'tcx, T> {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.value.default_span(tcx)
    }
}

impl Key for Symbol {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl Key for Option<Symbol> {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

/// Canonical query goals correspond to abstract trait operations that
/// are not tied to any crate in particular.
impl<'tcx, T: Clone> Key for Canonical<'tcx, T> {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl Key for (Symbol, u32, u32) {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (DefId, Ty<'tcx>, SubstsRef<'tcx>, ty::ParamEnv<'tcx>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (ty::Predicate<'tcx>, traits::WellFormedLoc) {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (ty::PolyFnSig<'tcx>, &'tcx ty::List<Ty<'tcx>>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (ty::Instance<'tcx>, &'tcx ty::List<Ty<'tcx>>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> Key for (Ty<'tcx>, ty::ValTree<'tcx>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl Key for HirId {
    type CacheSelector = DefaultCacheSelector<Self>;

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.hir().span(*self)
    }

    #[inline(always)]
    fn key_as_def_id(&self) -> Option<DefId> {
        None
    }
}

impl<'tcx> Key for (ValidityRequirement, ty::ParamEnvAnd<'tcx, Ty<'tcx>>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    // Just forward to `Ty<'tcx>`

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }

    fn ty_adt_id(&self) -> Option<DefId> {
        match self.1.value.kind() {
            ty::Adt(adt, _) => Some(adt.did()),
            _ => None,
        }
    }
}
