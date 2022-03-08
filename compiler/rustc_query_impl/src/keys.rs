//! Defines the set of legal keys that can be used in queries.

use rustc_hir::def_id::{CrateNum, DefId, LocalDefId, LOCAL_CRATE};
use rustc_middle::infer::canonical::Canonical;
use rustc_middle::mir;
use rustc_middle::traits;
use rustc_middle::ty::fast_reject::SimplifiedType;
use rustc_middle::ty::subst::{GenericArg, SubstsRef};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::symbol::{Ident, Symbol};
use rustc_span::{Span, DUMMY_SP};

/// The `Key` trait controls what types can legally be used as the key
/// for a query.
pub trait Key {
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
}

impl Key for () {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for ty::InstanceDef<'tcx> {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> Key for ty::Instance<'tcx> {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> Key for mir::interpret::GlobalId<'tcx> {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.instance.default_span(tcx)
    }
}

impl<'tcx> Key for (Ty<'tcx>, Option<ty::PolyExistentialTraitRef<'tcx>>) {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for mir::interpret::LitToConstInput<'tcx> {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl Key for CrateNum {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        *self == LOCAL_CRATE
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl Key for LocalDefId {
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
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.did.default_span(tcx)
    }
}

impl Key for (DefId, DefId) {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.0.krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.1.default_span(tcx)
    }
}

impl<'tcx> Key for (ty::Instance<'tcx>, LocalDefId) {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl Key for (DefId, LocalDefId) {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.0.krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.1.default_span(tcx)
    }
}

impl Key for (LocalDefId, DefId) {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl Key for (DefId, Option<Ident>) {
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
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.0.krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.1.default_span(tcx)
    }
}

impl Key for (CrateNum, DefId) {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.0 == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.1.default_span(tcx)
    }
}

impl Key for (DefId, SimplifiedType) {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.0.krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> Key for SubstsRef<'tcx> {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (DefId, SubstsRef<'tcx>) {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.0.krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> Key for (ty::Unevaluated<'tcx, ()>, ty::Unevaluated<'tcx, ()>) {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        (self.0).def.did.krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        (self.0).def.did.default_span(tcx)
    }
}

impl<'tcx> Key for (LocalDefId, DefId, SubstsRef<'tcx>) {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> Key for (ty::ParamEnv<'tcx>, ty::PolyTraitRef<'tcx>) {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.1.def_id().krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.1.def_id())
    }
}

impl<'tcx> Key for (ty::Const<'tcx>, mir::Field) {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for mir::interpret::ConstAlloc<'tcx> {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for ty::PolyTraitRef<'tcx> {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.def_id().krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> Key for ty::PolyExistentialTraitRef<'tcx> {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.def_id().krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> Key for (ty::PolyTraitRef<'tcx>, ty::PolyTraitRef<'tcx>) {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.0.def_id().krate == LOCAL_CRATE
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.def_span(self.0.def_id())
    }
}

impl<'tcx> Key for GenericArg<'tcx> {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for mir::ConstantKind<'tcx> {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for ty::Const<'tcx> {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for Ty<'tcx> {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (Ty<'tcx>, Ty<'tcx>) {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for &'tcx ty::List<ty::Predicate<'tcx>> {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for ty::ParamEnv<'tcx> {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }
    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx, T: Key> Key for ty::ParamEnvAnd<'tcx, T> {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        self.value.query_crate_is_local()
    }
    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.value.default_span(tcx)
    }
}

impl Key for Symbol {
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
impl<'tcx, T> Key for Canonical<'tcx, T> {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl Key for (Symbol, u32, u32) {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (DefId, Ty<'tcx>, SubstsRef<'tcx>, ty::ParamEnv<'tcx>) {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (ty::Predicate<'tcx>, traits::WellFormedLoc) {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, _tcx: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (ty::PolyFnSig<'tcx>, &'tcx ty::List<Ty<'tcx>>) {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, _: TyCtxt<'_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for (ty::Instance<'tcx>, &'tcx ty::List<Ty<'tcx>>) {
    #[inline(always)]
    fn query_crate_is_local(&self) -> bool {
        true
    }

    fn default_span(&self, tcx: TyCtxt<'_>) -> Span {
        self.0.default_span(tcx)
    }
}
