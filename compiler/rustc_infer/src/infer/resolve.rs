use super::{FixupError, FixupResult, InferCtxt};
use rustc_middle::bug;
use rustc_middle::ty::fold::{FallibleTypeFolder, TypeFolder, TypeSuperFoldable};
use rustc_middle::ty::visit::TypeVisitableExt;
use rustc_middle::ty::{self, Const, InferConst, Ty, TyCtxt, TypeFoldable};

///////////////////////////////////////////////////////////////////////////
// OPPORTUNISTIC VAR RESOLVER

/// The opportunistic resolver can be used at any time. It simply replaces
/// type/const variables that have been unified with the things they have
/// been unified with (similar to `shallow_resolve`, but deep). This is
/// useful for printing messages etc but also required at various
/// points for correctness.
pub struct OpportunisticVarResolver<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
}

impl<'a, 'tcx> OpportunisticVarResolver<'a, 'tcx> {
    #[inline]
    pub fn new(infcx: &'a InferCtxt<'tcx>) -> Self {
        OpportunisticVarResolver { infcx }
    }
}

impl<'a, 'tcx> TypeFolder<TyCtxt<'tcx>> for OpportunisticVarResolver<'a, 'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    #[inline]
    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !t.has_non_region_infer() {
            t // micro-optimize -- if there is nothing in this type that this fold affects...
        } else {
            let t = self.infcx.shallow_resolve(t);
            t.super_fold_with(self)
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
    fn interner(&self) -> TyCtxt<'tcx> {
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
        match *r {
            ty::ReVar(vid) => self
                .infcx
                .inner
                .borrow_mut()
                .unwrap_region_constraints()
                .opportunistic_resolve_var(TypeFolder::interner(self), vid),
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

// N.B. This type is not public because the protocol around checking the
// `err` field is not enforceable otherwise.
struct FullTypeResolver<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
}

impl<'a, 'tcx> FallibleTypeFolder<TyCtxt<'tcx>> for FullTypeResolver<'a, 'tcx> {
    type Error = FixupError;

    fn interner(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn try_fold_ty(&mut self, t: Ty<'tcx>) -> Result<Ty<'tcx>, Self::Error> {
        if !t.has_infer() {
            Ok(t) // micro-optimize -- if there is nothing in this type that this fold affects...
        } else {
            let t = self.infcx.shallow_resolve(t);
            match *t.kind() {
                ty::Infer(ty::TyVar(vid)) => Err(FixupError::UnresolvedTy(vid)),
                ty::Infer(ty::IntVar(vid)) => Err(FixupError::UnresolvedIntTy(vid)),
                ty::Infer(ty::FloatVar(vid)) => Err(FixupError::UnresolvedFloatTy(vid)),
                ty::Infer(_) => {
                    bug!("Unexpected type in full type resolver: {:?}", t);
                }
                _ => t.try_super_fold_with(self),
            }
        }
    }

    fn try_fold_region(&mut self, r: ty::Region<'tcx>) -> Result<ty::Region<'tcx>, Self::Error> {
        match *r {
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
                    return Err(FixupError::UnresolvedConst(vid));
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
