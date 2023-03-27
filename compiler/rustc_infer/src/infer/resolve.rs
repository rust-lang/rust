use super::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use super::{FixupError, FixupResult, InferCtxt, Span};
use rustc_middle::infer::unify_key::{ConstVariableOrigin, ConstVariableOriginKind};
use rustc_middle::ty::fold::{FallibleTypeFolder, TypeFolder, TypeSuperFoldable};
use rustc_middle::ty::visit::{TypeSuperVisitable, TypeVisitableExt, TypeVisitor};
use rustc_middle::ty::{self, Const, InferConst, Ty, TyCtxt, TypeFoldable};

use std::ops::ControlFlow;

///////////////////////////////////////////////////////////////////////////
// OPPORTUNISTIC VAR RESOLVER

/// The opportunistic resolver can be used at any time. It simply replaces
/// type/const variables that have been unified with the things they have
/// been unified with (similar to `shallow_resolve`, but deep). This is
/// useful for printing messages etc but also required at various
/// points for correctness.
pub struct OpportunisticVarResolver<'a, 'tcx> {
    // The shallow resolver is used to resolve inference variables at every
    // level of the type.
    shallow_resolver: crate::infer::ShallowResolver<'a, 'tcx>,
}

impl<'a, 'tcx> OpportunisticVarResolver<'a, 'tcx> {
    #[inline]
    pub fn new(infcx: &'a InferCtxt<'tcx>) -> Self {
        OpportunisticVarResolver { shallow_resolver: crate::infer::ShallowResolver { infcx } }
    }
}

impl<'a, 'tcx> TypeFolder<TyCtxt<'tcx>> for OpportunisticVarResolver<'a, 'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        TypeFolder::interner(&self.shallow_resolver)
    }

    #[inline]
    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !t.has_non_region_infer() {
            t // micro-optimize -- if there is nothing in this type that this fold affects...
        } else {
            let t = self.shallow_resolver.fold_ty(t);
            t.super_fold_with(self)
        }
    }

    fn fold_const(&mut self, ct: Const<'tcx>) -> Const<'tcx> {
        if !ct.has_non_region_infer() {
            ct // micro-optimize -- if there is nothing in this const that this fold affects...
        } else {
            let ct = self.shallow_resolver.fold_const(ct);
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
// UNRESOLVED TYPE FINDER

/// The unresolved type **finder** walks a type searching for
/// type variables that don't yet have a value. The first unresolved type is stored.
/// It does not construct the fully resolved type (which might
/// involve some hashing and so forth).
pub struct UnresolvedTypeOrConstFinder<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
}

impl<'a, 'tcx> UnresolvedTypeOrConstFinder<'a, 'tcx> {
    pub fn new(infcx: &'a InferCtxt<'tcx>) -> Self {
        UnresolvedTypeOrConstFinder { infcx }
    }
}

impl<'a, 'tcx> TypeVisitor<TyCtxt<'tcx>> for UnresolvedTypeOrConstFinder<'a, 'tcx> {
    type BreakTy = (ty::Term<'tcx>, Option<Span>);
    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        let t = self.infcx.shallow_resolve(t);
        if let ty::Infer(infer_ty) = *t.kind() {
            // Since we called `shallow_resolve` above, this must
            // be an (as yet...) unresolved inference variable.
            let ty_var_span = if let ty::TyVar(ty_vid) = infer_ty {
                let mut inner = self.infcx.inner.borrow_mut();
                let ty_vars = &inner.type_variables();
                if let TypeVariableOrigin {
                    kind: TypeVariableOriginKind::TypeParameterDefinition(_, _),
                    span,
                } = *ty_vars.var_origin(ty_vid)
                {
                    Some(span)
                } else {
                    None
                }
            } else {
                None
            };
            ControlFlow::Break((t.into(), ty_var_span))
        } else if !t.has_non_region_infer() {
            // All const/type variables in inference types must already be resolved,
            // no need to visit the contents.
            ControlFlow::Continue(())
        } else {
            // Otherwise, keep visiting.
            t.super_visit_with(self)
        }
    }

    fn visit_const(&mut self, ct: ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
        let ct = self.infcx.shallow_resolve(ct);
        if let ty::ConstKind::Infer(i) = ct.kind() {
            // Since we called `shallow_resolve` above, this must
            // be an (as yet...) unresolved inference variable.
            let ct_var_span = if let ty::InferConst::Var(vid) = i {
                let mut inner = self.infcx.inner.borrow_mut();
                let ct_vars = &mut inner.const_unification_table();
                if let ConstVariableOrigin {
                    span,
                    kind: ConstVariableOriginKind::ConstParameterDefinition(_, _),
                } = ct_vars.probe_value(vid).origin
                {
                    Some(span)
                } else {
                    None
                }
            } else {
                None
            };
            ControlFlow::Break((ct.into(), ct_var_span))
        } else if !ct.has_non_region_infer() {
            // All const/type variables in inference types must already be resolved,
            // no need to visit the contents.
            ControlFlow::Continue(())
        } else {
            // Otherwise, keep visiting.
            ct.super_visit_with(self)
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// FULL TYPE RESOLUTION

/// Full type resolution replaces all type and region variables with
/// their concrete results. If any variable cannot be replaced (never unified, etc)
/// then an `Err` result is returned.
pub fn fully_resolve<'tcx, T>(infcx: &InferCtxt<'tcx>, value: T) -> FixupResult<'tcx, T>
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
    type Error = FixupError<'tcx>;

    fn interner(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn try_fold_ty(&mut self, t: Ty<'tcx>) -> Result<Ty<'tcx>, Self::Error> {
        if !t.needs_infer() {
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
        if !c.needs_infer() {
            Ok(c) // micro-optimize -- if there is nothing in this const that this fold affects...
        } else {
            let c = self.infcx.shallow_resolve(c);
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
