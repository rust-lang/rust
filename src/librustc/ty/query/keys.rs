//! Defines the set of legal keys that can be used in queries.

use infer::canonical::Canonical;
use hir::def_id::{CrateNum, DefId, LOCAL_CRATE, DefIndex};
use traits;
use ty::{self, Ty, TyCtxt};
use ty::subst::Substs;
use ty::fast_reject::SimplifiedType;
use mir;

use std::fmt::Debug;
use std::hash::Hash;
use syntax_pos::{Span, DUMMY_SP};
use syntax_pos::symbol::InternedString;

/// The `Key` trait controls what types can legally be used as the key
/// for a query.
pub(super) trait Key: Clone + Hash + Eq + Debug {
    /// Given an instance of this key, what crate is it referring to?
    /// This is used to find the provider.
    fn query_crate(&self) -> CrateNum;

    /// In the event that a cycle occurs, if no explicit span has been
    /// given for a query with key `self`, what span should we use?
    fn default_span(&self, tcx: TyCtxt<'_, '_, '_>) -> Span;
}

impl<'tcx> Key for ty::InstanceDef<'tcx> {
    fn query_crate(&self) -> CrateNum {
        LOCAL_CRATE
    }

    fn default_span(&self, tcx: TyCtxt<'_, '_, '_>) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> Key for ty::Instance<'tcx> {
    fn query_crate(&self) -> CrateNum {
        LOCAL_CRATE
    }

    fn default_span(&self, tcx: TyCtxt<'_, '_, '_>) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> Key for mir::interpret::GlobalId<'tcx> {
    fn query_crate(&self) -> CrateNum {
        self.instance.query_crate()
    }

    fn default_span(&self, tcx: TyCtxt<'_, '_, '_>) -> Span {
        self.instance.default_span(tcx)
    }
}

impl Key for CrateNum {
    fn query_crate(&self) -> CrateNum {
        *self
    }
    fn default_span(&self, _: TyCtxt<'_, '_, '_>) -> Span {
        DUMMY_SP
    }
}

impl Key for DefIndex {
    fn query_crate(&self) -> CrateNum {
        LOCAL_CRATE
    }
    fn default_span(&self, _tcx: TyCtxt<'_, '_, '_>) -> Span {
        DUMMY_SP
    }
}

impl Key for DefId {
    fn query_crate(&self) -> CrateNum {
        self.krate
    }
    fn default_span(&self, tcx: TyCtxt<'_, '_, '_>) -> Span {
        tcx.def_span(*self)
    }
}

impl Key for (DefId, DefId) {
    fn query_crate(&self) -> CrateNum {
        self.0.krate
    }
    fn default_span(&self, tcx: TyCtxt<'_, '_, '_>) -> Span {
        self.1.default_span(tcx)
    }
}

impl Key for (CrateNum, DefId) {
    fn query_crate(&self) -> CrateNum {
        self.0
    }
    fn default_span(&self, tcx: TyCtxt<'_, '_, '_>) -> Span {
        self.1.default_span(tcx)
    }
}

impl Key for (DefId, SimplifiedType) {
    fn query_crate(&self) -> CrateNum {
        self.0.krate
    }
    fn default_span(&self, tcx: TyCtxt<'_, '_, '_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> Key for (DefId, &'tcx Substs<'tcx>) {
    fn query_crate(&self) -> CrateNum {
        self.0.krate
    }
    fn default_span(&self, tcx: TyCtxt<'_, '_, '_>) -> Span {
        self.0.default_span(tcx)
    }
}

impl<'tcx> Key for (ty::ParamEnv<'tcx>, ty::PolyTraitRef<'tcx>) {
    fn query_crate(&self) -> CrateNum {
        self.1.def_id().krate
    }
    fn default_span(&self, tcx: TyCtxt<'_, '_, '_>) -> Span {
        tcx.def_span(self.1.def_id())
    }
}

impl<'tcx> Key for ty::PolyTraitRef<'tcx>{
    fn query_crate(&self) -> CrateNum {
        self.def_id().krate
    }
    fn default_span(&self, tcx: TyCtxt<'_, '_, '_>) -> Span {
        tcx.def_span(self.def_id())
    }
}

impl<'tcx> Key for &'tcx ty::Const<'tcx> {
    fn query_crate(&self) -> CrateNum {
        LOCAL_CRATE
    }
    fn default_span(&self, _: TyCtxt<'_, '_, '_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for Ty<'tcx> {
    fn query_crate(&self) -> CrateNum {
        LOCAL_CRATE
    }
    fn default_span(&self, _: TyCtxt<'_, '_, '_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx> Key for ty::ParamEnv<'tcx> {
    fn query_crate(&self) -> CrateNum {
        LOCAL_CRATE
    }
    fn default_span(&self, _: TyCtxt<'_, '_, '_>) -> Span {
        DUMMY_SP
    }
}

impl<'tcx, T: Key> Key for ty::ParamEnvAnd<'tcx, T> {
    fn query_crate(&self) -> CrateNum {
        self.value.query_crate()
    }
    fn default_span(&self, tcx: TyCtxt<'_, '_, '_>) -> Span {
        self.value.default_span(tcx)
    }
}

impl<'tcx> Key for traits::Environment<'tcx> {
    fn query_crate(&self) -> CrateNum {
        LOCAL_CRATE
    }
    fn default_span(&self, _: TyCtxt<'_, '_, '_>) -> Span {
        DUMMY_SP
    }
}

impl Key for InternedString {
    fn query_crate(&self) -> CrateNum {
        LOCAL_CRATE
    }
    fn default_span(&self, _tcx: TyCtxt<'_, '_, '_>) -> Span {
        DUMMY_SP
    }
}

/// Canonical query goals correspond to abstract trait operations that
/// are not tied to any crate in particular.
impl<'tcx, T> Key for Canonical<'tcx, T>
where
    T: Debug + Hash + Clone + Eq,
{
    fn query_crate(&self) -> CrateNum {
        LOCAL_CRATE
    }

    fn default_span(&self, _tcx: TyCtxt<'_, '_, '_>) -> Span {
        DUMMY_SP
    }
}
