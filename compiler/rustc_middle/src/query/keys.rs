//! Defines the set of legal keys that can be used in queries.

use crate::infer::canonical::Canonical;
use crate::mir;
use crate::traits;
use crate::traits::ChalkEnvironmentAndGoal;
use crate::ty::fast_reject::SimplifiedType;
use crate::ty::subst::{GenericArg, SubstsRef};
use crate::ty::{self, layout::TyAndLayout, Ty, TyCtxt};
use rustc_data_structures::remap::Remap;
use rustc_hir::def_id::{CrateNum, DefId, LocalDefId, LOCAL_CRATE};
use rustc_hir::hir_id::{HirId, OwnerId};
pub use rustc_middle::traits::query::type_op;
use rustc_query_system::query::{DefaultCacheSelector, VecCacheSelector};
use rustc_span::symbol::{Ident, Symbol};
use rustc_span::{Span, DUMMY_SP};

/// The `Key` trait controls what types can legally be used as the key
/// for a query.
pub trait Key: Sized + Remap {
    // N.B. Most of the keys down below have `type CacheSelector = DefaultCacheSelector<Self>;`,
    //      it would be reasonable to use associated type defaults, to remove the duplication...
    //
    //      ...But r-a doesn't support them yet and using a default here causes r-a to not infer
    //      return types of queries which is very annoying. Thus, until r-a support associated
    //      type defaults, plese restrain from using them here <3
    //
    //      r-a issue: <https://github.com/rust-lang/rust-analyzer/issues/13693>
    type CacheSelector;

    /// Given an instance of this key, what crate is it referring to?
    /// This is used to find the provider.
    fn query_crate_is_local(&self) -> bool;

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

impl Key for () {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for ty::InstanceDef<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> Key for ty::Instance<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> Key for mir::interpret::GlobalId<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.instance.default_span(tcx)
    }
}

impl<'tcx> Key for (Ty<'tcx>, Option<ty::PolyExistentialTraitRef<'tcx>>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for mir::interpret::LitToConstInput<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl Key for CrateNum {
    type CacheSelector = VecCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        *self == LOCAL_CRATE
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl Key for OwnerId {
    type CacheSelector = VecCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.to_def_id().default_span(tcx)
    }
    fn key_as_def_id(&self) -> Option<DefId> {
        Some(self.to_def_id())
    }
}

impl Key for LocalDefId {
    type CacheSelector = VecCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.to_def_id().default_span(tcx)
    }
    fn key_as_def_id(&self) -> Option<DefId> {
        Some(self.to_def_id())
    }
}

impl Key for DefId {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(*self)
    }
    #[inline(always)]
    fn key_as_def_id(&self) -> Option<DefId> {
        Some(*self)
    }
}

impl Key for ty::WithOptConstParam<LocalDefId> {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.did.default_span(tcx)
    }
}

impl Key for SimplifiedType {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl Key for (DefId, DefId) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.0.krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.1.default_span(tcx)
    }
}

impl<'tcx> Key for (ty::Instance<'tcx>, LocalDefId) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl Key for (DefId, LocalDefId) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.0.krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.1.default_span(tcx)
    }
}

impl Key for (LocalDefId, DefId) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl Key for (LocalDefId, LocalDefId) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl Key for (DefId, Option<Ident>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.0.krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.0)
    }
    #[inline(always)]
    fn key_as_def_id(&self) -> Option<DefId> {
        Some(self.0)
    }
}

impl Key for (DefId, LocalDefId, Ident) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.0.krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.1.default_span(tcx)
    }
}

impl Key for (CrateNum, DefId) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.0 == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.1.default_span(tcx)
    }
}

impl Key for (CrateNum, SimplifiedType) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.0 == LOCAL_CRATE
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl Key for (DefId, SimplifiedType) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.0.krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> Key for SubstsRef<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (DefId, SubstsRef<'tcx>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.0.krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> Key for (ty::UnevaluatedConst<'tcx>, ty::UnevaluatedConst<'tcx>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        (self.0).def.did.krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        (self.0).def.did.default_span(tcx)
    }
}

impl<'tcx> Key for (LocalDefId, DefId, SubstsRef<'tcx>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> Key for (ty::ParamEnv<'tcx>, ty::PolyTraitRef<'tcx>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.1.def_id().krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.1.def_id())
    }
}

impl<'tcx> Key for (ty::Const<'tcx>, mir::Field) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for mir::interpret::ConstAlloc<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for ty::PolyTraitRef<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.def_id().krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> Key for ty::PolyExistentialTraitRef<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.def_id().krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> Key for (ty::PolyTraitRef<'tcx>, ty::PolyTraitRef<'tcx>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.0.def_id().krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.0.def_id())
    }
}

impl<'tcx> Key for GenericArg<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for mir::ConstantKind<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for ty::Const<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for Ty<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
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

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (Ty<'tcx>, Ty<'tcx>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for &'tcx ty::List<ty::Predicate<'tcx>> {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for ty::ParamEnv<'tcx> {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx, T: Key> Key for ty::ParamEnvAnd<'tcx, T> {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.value.query_crate_is_local()
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.value.default_span(tcx)
    }
}

impl Key for Symbol {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl Key for Option<Symbol> {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

/// Canonical query goals correspond to abstract trait operations that
/// are not tied to any crate in particular.
impl<'tcx, T: Remap> Key for Canonical<'tcx, T> {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl Key for (Symbol, u32, u32) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (DefId, Ty<'tcx>, SubstsRef<'tcx>, ty::ParamEnv<'tcx>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (ty::Predicate<'tcx>, traits::WellFormedLoc) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (ty::PolyFnSig<'tcx>, &'tcx ty::List<Ty<'tcx>>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (ty::Instance<'tcx>, &'tcx ty::List<Ty<'tcx>>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> Key for (Ty<'tcx>, ty::ValTree<'tcx>) {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl Key for HirId {
    type CacheSelector = DefaultCacheSelector<Self>;

    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.hir().span(*self)
    }

    #[inline(always)]
    fn key_as_def_id(&self) -> Option<DefId> {
        None
    }
}

// Remap implementations

impl<'tcx, T: Remap> Remap for ty::ParamEnvAnd<'tcx, T> {
    type Remap<'a> = ty::ParamEnvAnd<'a, T::Remap<'a>>;
}

impl<'tcx, T: Remap> Remap for ty::Binder<'tcx, T> {
    type Remap<'a> = ty::Binder<'a, T::Remap<'a>>;
}

impl<'tcx, T: Remap> Remap for Canonical<'tcx, T> {
    type Remap<'a> = Canonical<'a, T::Remap<'a>>;
}

impl<T: Remap> Remap for type_op::Normalize<T> {
    type Remap<'a> = type_op::Normalize<T::Remap<'a>>;
}

impl<'tcx> Remap for type_op::AscribeUserType<'tcx> {
    type Remap<'a> = type_op::AscribeUserType<'a>;
}

impl<'tcx> Remap for type_op::Subtype<'tcx> {
    type Remap<'a> = type_op::Subtype<'a>;
}

impl<'tcx> Remap for type_op::Eq<'tcx> {
    type Remap<'a> = type_op::Eq<'a>;
}

impl<'tcx> Remap for type_op::ProvePredicate<'tcx> {
    type Remap<'a> = type_op::ProvePredicate<'a>;
}

impl<'tcx> Remap for ty::FnSig<'tcx> {
    type Remap<'a> = ty::FnSig<'a>;
}

impl<'tcx> Remap for ty::AliasTy<'tcx> {
    type Remap<'a> = ty::AliasTy<'a>;
}

impl<'tcx> Remap for Ty<'tcx> {
    type Remap<'a> = Ty<'a>;
}

impl<'tcx> Remap for ty::Predicate<'tcx> {
    type Remap<'a> = ty::Predicate<'a>;
}

impl<'tcx> Remap for ChalkEnvironmentAndGoal<'tcx> {
    type Remap<'a> = ChalkEnvironmentAndGoal<'a>;
}

impl<'tcx> Remap for ty::Instance<'tcx> {
    type Remap<'a> = ty::Instance<'a>;
}

impl<'tcx> Remap for ty::InstanceDef<'tcx> {
    type Remap<'a> = ty::InstanceDef<'a>;
}

impl<T: Remap> Remap for ty::WithOptConstParam<T> {
    type Remap<'a> = ty::WithOptConstParam<T::Remap<'a>>;
}

impl Remap for SimplifiedType {
    type Remap<'a> = SimplifiedType;
}

impl<'tcx> Remap for mir::interpret::GlobalId<'tcx> {
    type Remap<'a> = mir::interpret::GlobalId<'a>;
}

impl<'tcx> Remap for mir::interpret::LitToConstInput<'tcx> {
    type Remap<'a> = mir::interpret::LitToConstInput<'a>;
}

impl<'tcx> Remap for mir::interpret::ConstAlloc<'tcx> {
    type Remap<'a> = mir::interpret::ConstAlloc<'a>;
}

impl<'tcx> Remap for mir::ConstantKind<'tcx> {
    type Remap<'a> = mir::ConstantKind<'a>;
}

impl Remap for mir::Field {
    type Remap<'a> = mir::Field;
}

impl<'tcx> Remap for ty::ValTree<'tcx> {
    type Remap<'a> = ty::ValTree<'a>;
}

impl<'tcx> Remap for ty::ParamEnv<'tcx> {
    type Remap<'a> = ty::ParamEnv<'a>;
}

impl<'tcx> Remap for ty::GenericArg<'tcx> {
    type Remap<'a> = ty::GenericArg<'a>;
}

impl<'tcx, T: Remap> Remap for &'tcx ty::List<T>
where
    for<'a> <T as Remap>::Remap<'a>: 'a,
{
    type Remap<'a> = &'a ty::List<T::Remap<'a>>;
}

impl<'tcx> Remap for ty::ExistentialTraitRef<'tcx> {
    type Remap<'a> = ty::ExistentialTraitRef<'a>;
}

impl<'tcx> Remap for ty::Const<'tcx> {
    type Remap<'a> = ty::Const<'a>;
}

impl<'tcx> Remap for ty::TraitRef<'tcx> {
    type Remap<'a> = ty::TraitRef<'a>;
}

impl<'tcx> Remap for ty::UnevaluatedConst<'tcx> {
    type Remap<'a> = ty::UnevaluatedConst<'a>;
}

impl Remap for traits::WellFormedLoc {
    type Remap<'a> = traits::WellFormedLoc;
}
