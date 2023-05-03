use rustc_hir as hir;
use rustc_hir::def_id::DefId;

use crate::ty::{self, BoundConstness, ImplPolarity, ParamEnv, TraitRef, Ty, TyCtxt};

pub type PolyTraitPredicate<'tcx> = ty::Binder<'tcx, TraitPredicate<'tcx>>;

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

    #[inline]
    pub fn polarity(self) -> ImplPolarity {
        self.skip_binder().polarity
    }
}
