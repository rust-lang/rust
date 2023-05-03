use crate::ty::{
    self, Clause, PolyProjectionPredicate, PolyRegionOutlivesPredicate, PolyTraitPredicate,
    PolyTypeOutlivesPredicate, Predicate, PredicateKind, TraitPredicate, TraitRef, TyCtxt,
};

pub trait ToPredicate<'tcx, P = Predicate<'tcx>> {
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> P;
}

impl<'tcx, T> ToPredicate<'tcx, T> for T {
    fn to_predicate(self, _tcx: TyCtxt<'tcx>) -> T {
        self
    }
}

impl<'tcx> ToPredicate<'tcx> for PredicateKind<'tcx> {
    #[inline(always)]
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        ty::Binder::dummy(self).to_predicate(tcx)
    }
}

impl<'tcx> ToPredicate<'tcx> for ty::Binder<'tcx, PredicateKind<'tcx>> {
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

impl<'tcx> ToPredicate<'tcx> for TraitRef<'tcx> {
    #[inline(always)]
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        ty::Binder::dummy(self).to_predicate(tcx)
    }
}

impl<'tcx> ToPredicate<'tcx> for ty::Binder<'tcx, TraitRef<'tcx>> {
    #[inline(always)]
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        let pred: PolyTraitPredicate<'tcx> = self.to_predicate(tcx);
        pred.to_predicate(tcx)
    }
}

impl<'tcx> ToPredicate<'tcx, PolyTraitPredicate<'tcx>> for ty::Binder<'tcx, TraitRef<'tcx>> {
    #[inline(always)]
    fn to_predicate(self, _: TyCtxt<'tcx>) -> PolyTraitPredicate<'tcx> {
        self.map_bound(|trait_ref| TraitPredicate {
            trait_ref,
            constness: ty::BoundConstness::NotConst,
            polarity: ty::ImplPolarity::Positive,
        })
    }
}

impl<'tcx> ToPredicate<'tcx, PolyTraitPredicate<'tcx>> for TraitRef<'tcx> {
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> PolyTraitPredicate<'tcx> {
        ty::Binder::dummy(self).to_predicate(tcx)
    }
}

impl<'tcx> ToPredicate<'tcx, PolyTraitPredicate<'tcx>> for TraitPredicate<'tcx> {
    fn to_predicate(self, _tcx: TyCtxt<'tcx>) -> PolyTraitPredicate<'tcx> {
        ty::Binder::dummy(self)
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

impl<'tcx> ToPredicate<'tcx> for TraitPredicate<'tcx> {
    fn to_predicate(self, tcx: TyCtxt<'tcx>) -> Predicate<'tcx> {
        PredicateKind::Clause(Clause::Trait(self)).to_predicate(tcx)
    }
}
