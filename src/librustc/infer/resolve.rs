use super::{InferCtxt, FixupError, FixupResult};
use crate::ty::{self, Ty, TyCtxt, TypeFoldable};
use crate::ty::fold::{TypeFolder, TypeVisitor};

///////////////////////////////////////////////////////////////////////////
// OPPORTUNISTIC TYPE RESOLVER

/// The opportunistic type resolver can be used at any time. It simply replaces
/// type variables that have been unified with the things they have
/// been unified with (similar to `shallow_resolve`, but deep). This is
/// useful for printing messages etc but also required at various
/// points for correctness.
pub struct OpportunisticTypeResolver<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
}

impl<'a, 'gcx, 'tcx> OpportunisticTypeResolver<'a, 'gcx, 'tcx> {
    #[inline]
    pub fn new(infcx: &'a InferCtxt<'a, 'gcx, 'tcx>) -> Self {
        OpportunisticTypeResolver { infcx }
    }
}

impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for OpportunisticTypeResolver<'a, 'gcx, 'tcx> {
    type Error = !;

    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Result<Ty<'tcx>, !> {
        if !t.has_infer_types() {
            Ok(t) // micro-optimize -- if there is nothing in this type that this fold affects...
        } else {
            let t0 = self.infcx.shallow_resolve(t);
            t0.super_fold_with(self)
        }
    }
}

/// The opportunistic type and region resolver is similar to the
/// opportunistic type resolver, but also opportunistically resolves
/// regions. It is useful for canonicalization.
pub struct OpportunisticTypeAndRegionResolver<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
}

impl<'a, 'gcx, 'tcx> OpportunisticTypeAndRegionResolver<'a, 'gcx, 'tcx> {
    pub fn new(infcx: &'a InferCtxt<'a, 'gcx, 'tcx>) -> Self {
        OpportunisticTypeAndRegionResolver { infcx }
    }
}

impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for OpportunisticTypeAndRegionResolver<'a, 'gcx, 'tcx> {
    type Error = !;

    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Result<Ty<'tcx>, !> {
        if !t.needs_infer() {
            Ok(t) // micro-optimize -- if there is nothing in this type that this fold affects...
        } else {
            let t0 = self.infcx.shallow_resolve(t);
            t0.super_fold_with(self)
        }
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> Result<ty::Region<'tcx>, !> {
        Ok(match *r {
            ty::ReVar(rid) =>
                self.infcx.borrow_region_constraints()
                          .opportunistic_resolve_var(self.tcx(), rid),
            _ =>
                r,
        })
    }
}

///////////////////////////////////////////////////////////////////////////
// UNRESOLVED TYPE FINDER

/// The unresolved type **finder** walks your type and searches for
/// type variables that don't yet have a value. They get pushed into a
/// vector. It does not construct the fully resolved type (which might
/// involve some hashing and so forth).
pub struct UnresolvedTypeFinder<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
}

impl<'a, 'gcx, 'tcx> UnresolvedTypeFinder<'a, 'gcx, 'tcx> {
    pub fn new(infcx: &'a InferCtxt<'a, 'gcx, 'tcx>) -> Self {
        UnresolvedTypeFinder { infcx }
    }
}

impl<'a, 'gcx, 'tcx> TypeVisitor<'tcx> for UnresolvedTypeFinder<'a, 'gcx, 'tcx> {
    type Error = ();

    fn visit_ty(&mut self, t: Ty<'tcx>) -> Result<(), ()> {
        let t = self.infcx.shallow_resolve(t);
        if t.has_infer_types() {
            if let ty::Infer(_) = t.sty {
                // Since we called `shallow_resolve` above, this must
                // be an (as yet...) unresolved inference variable.
                Err(())
            } else {
                // Otherwise, visit its contents.
                t.super_visit_with(self)
            }
        } else {
            // Micro-optimize: no inference types at all Can't have unresolved type
            // variables, no need to visit the contents.
            Ok(())
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// FULL TYPE RESOLUTION

/// Full type resolution replaces all type and region variables with
/// their concrete results. If any variable cannot be replaced (never unified, etc)
/// then an `Err` result is returned.
pub fn fully_resolve<'a, 'gcx, 'tcx, T>(infcx: &InferCtxt<'a, 'gcx, 'tcx>,
                                        value: &T) -> FixupResult<T>
    where T : TypeFoldable<'tcx>
{
    let mut full_resolver = FullTypeResolver { infcx: infcx, err: None };
    let Ok(result) = value.fold_with(&mut full_resolver);
    match full_resolver.err {
        None => Ok(result),
        Some(e) => Err(e),
    }
}

// N.B. This type is not public because the protocol around checking the
// `err` field is not enforcable otherwise.
struct FullTypeResolver<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    err: Option<FixupError>,
}

impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for FullTypeResolver<'a, 'gcx, 'tcx> {
    type Error = !;

    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Result<Ty<'tcx>, !> {
        if !t.needs_infer() && !ty::keep_local(&t) {
            Ok(t) // micro-optimize -- if there is nothing in this type that this fold affects...
                  // ^ we need to have the `keep_local` check to un-default
                  // defaulted tuples.
        } else {
            let t = self.infcx.shallow_resolve(t);
            // FIXME: use the error propagation support.
            match t.sty {
                ty::Infer(ty::TyVar(vid)) => {
                    self.err = Some(FixupError::UnresolvedTy(vid));
                    Ok(self.tcx().types.err)
                }
                ty::Infer(ty::IntVar(vid)) => {
                    self.err = Some(FixupError::UnresolvedIntTy(vid));
                    Ok(self.tcx().types.err)
                }
                ty::Infer(ty::FloatVar(vid)) => {
                    self.err = Some(FixupError::UnresolvedFloatTy(vid));
                    Ok(self.tcx().types.err)
                }
                ty::Infer(_) => {
                    bug!("Unexpected type in full type resolver: {:?}", t);
                }
                _ => {
                    t.super_fold_with(self)
                }
            }
        }
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> Result<ty::Region<'tcx>, !> {
        Ok(match *r {
            ty::ReVar(rid) => self.infcx.lexical_region_resolutions
                                        .borrow()
                                        .as_ref()
                                        .expect("region resolution not performed")
                                        .resolve_var(rid),
            _ => r,
        })
    }
}
