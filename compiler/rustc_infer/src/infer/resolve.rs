use rustc_middle::bug;
use rustc_middle::ty::{
    self, Const, DelayedMap, FallibleTypeFolder, InferConst, Ty, TyCtxt, TypeFoldable, TypeFolder,
    TypeSuperFoldable, TypeVisitableExt,
};

use super::{FixupError, FixupResult, InferCtxt};

///////////////////////////////////////////////////////////////////////////
// OPPORTUNISTIC VAR RESOLVER

/// The opportunistic resolver can be used at any time. It simply replaces
/// type/const variables that have been unified with the things they have
/// been unified with (similar to `shallow_resolve`, but deep). This is
/// useful for printing messages etc but also required at various
/// points for correctness.
pub struct OpportunisticVarResolver<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
    /// We're able to use a cache here as the folder does
    /// not have any mutable state.
    cache: DelayedMap<Ty<'tcx>, Ty<'tcx>>,
}

impl<'a, 'tcx> OpportunisticVarResolver<'a, 'tcx> {
    #[inline]
    pub fn new(infcx: &'a InferCtxt<'tcx>) -> Self {
        OpportunisticVarResolver { infcx, cache: Default::default() }
    }
}

impl<'a, 'tcx> TypeFolder<TyCtxt<'tcx>> for OpportunisticVarResolver<'a, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    #[inline]
    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !t.has_non_region_infer() {
            t // micro-optimize -- if there is nothing in this type that this fold affects...
        } else if let Some(&ty) = self.cache.get(&t) {
            ty
        } else {
            let shallow = self.infcx.shallow_resolve(t);
            let res = shallow.super_fold_with(self);
            assert!(self.cache.insert(t, res));
            res
        }
    }

    fn fold_const(&mut self, ct: Const<'tcx>) -> Const<'tcx> {
        if !ct.has_non_region_infer() {
            ct // micro-optimize -- if there is nothing in this const that this fold affects...
        } else {
            let ct = self.infcx.shallow_resolve_const(ct);
            ct.super_fold_with(self)
        }
    }

    fn fold_predicate(&mut self, p: ty::Predicate<'tcx>) -> ty::Predicate<'tcx> {
        if !p.has_non_region_infer() { p } else { p.super_fold_with(self) }
    }

    fn fold_clauses(&mut self, c: ty::Clauses<'tcx>) -> ty::Clauses<'tcx> {
        if !c.has_non_region_infer() { c } else { c.super_fold_with(self) }
    }
}

/// The opportunistic region resolver opportunistically resolves regions
/// variables to the variable with the least variable id. It is used when
/// normalizing projections to avoid hitting the recursion limit by creating
/// many versions of a predicate for types that in the end have to unify.
///
/// If you want to resolve type and const variables as well, call
/// [InferCtxt::resolve_vars_if_possible] first.
pub struct OpportunisticRegionResolver<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
}

impl<'a, 'tcx> OpportunisticRegionResolver<'a, 'tcx> {
    pub fn new(infcx: &'a InferCtxt<'tcx>) -> Self {
        OpportunisticRegionResolver { infcx }
    }
}

impl<'a, 'tcx> TypeFolder<TyCtxt<'tcx>> for OpportunisticRegionResolver<'a, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !t.has_infer_regions() {
            t // micro-optimize -- if there is nothing in this type that this fold affects...
        } else {
            t.super_fold_with(self)
        }
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match r.kind() {
            ty::ReVar(vid) => self
                .infcx
                .inner
                .borrow_mut()
                .unwrap_region_constraints()
                .opportunistic_resolve_var(TypeFolder::cx(self), vid),
            _ => r,
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        if !ct.has_infer_regions() {
            ct // micro-optimize -- if there is nothing in this const that this fold affects...
        } else {
            ct.super_fold_with(self)
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// FULL TYPE RESOLUTION

/// Full type resolution replaces all type and region variables with
/// their concrete results. If any variable cannot be replaced (never unified, etc)
/// then an `Err` result is returned.
pub fn fully_resolve<'tcx, T>(infcx: &InferCtxt<'tcx>, value: T) -> FixupResult<T>
where
    T: TypeFoldable<TyCtxt<'tcx>>,
{
    value.try_fold_with(&mut FullTypeResolver { infcx })
}

struct FullTypeResolver<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
}

impl<'a, 'tcx> FallibleTypeFolder<TyCtxt<'tcx>> for FullTypeResolver<'a, 'tcx> {
    type Error = FixupError;

    fn cx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn try_fold_ty(&mut self, t: Ty<'tcx>) -> Result<Ty<'tcx>, Self::Error> {
        if !t.has_infer() {
            Ok(t) // micro-optimize -- if there is nothing in this type that this fold affects...
        } else {
            use super::TyOrConstInferVar::*;

            let t = self.infcx.shallow_resolve(t);
            match *t.kind() {
                ty::Infer(ty::TyVar(vid)) => Err(FixupError { unresolved: Ty(vid) }),
                ty::Infer(ty::IntVar(vid)) => Err(FixupError { unresolved: TyInt(vid) }),
                ty::Infer(ty::FloatVar(vid)) => Err(FixupError { unresolved: TyFloat(vid) }),
                ty::Infer(_) => {
                    bug!("Unexpected type in full type resolver: {:?}", t);
                }
                _ => t.try_super_fold_with(self),
            }
        }
    }

    fn try_fold_region(&mut self, r: ty::Region<'tcx>) -> Result<ty::Region<'tcx>, Self::Error> {
        match r.kind() {
            ty::ReVar(_) => Ok(self
                .infcx
                .lexical_region_resolutions
                .borrow()
                .as_ref()
                .expect("region resolution not performed")
                .resolve_region(self.infcx.tcx, r)),
            _ => Ok(r),
        }
    }

    fn try_fold_const(&mut self, c: ty::Const<'tcx>) -> Result<ty::Const<'tcx>, Self::Error> {
        if !c.has_infer() {
            Ok(c) // micro-optimize -- if there is nothing in this const that this fold affects...
        } else {
            let c = self.infcx.shallow_resolve_const(c);
            match c.kind() {
                ty::ConstKind::Infer(InferConst::Var(vid)) => {
                    return Err(FixupError { unresolved: super::TyOrConstInferVar::Const(vid) });
                }
                ty::ConstKind::Infer(InferConst::Fresh(_)) => {
                    bug!("Unexpected const in full const resolver: {:?}", c);
                }
                _ => {}
            }
            c.try_super_fold_with(self)
        }
    }
}
