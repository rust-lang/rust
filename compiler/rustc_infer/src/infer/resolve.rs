use super::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use super::{FixupError, FixupResult, InferCtxt, Span};
use rustc_middle::ty::fold::{TypeFolder, TypeVisitor};
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
    infcx: &'a InferCtxt<'a, 'tcx>,
}

impl<'a, 'tcx> OpportunisticVarResolver<'a, 'tcx> {
    #[inline]
    pub fn new(infcx: &'a InferCtxt<'a, 'tcx>) -> Self {
        OpportunisticVarResolver { infcx }
    }
}

impl<'a, 'tcx> TypeFolder<'tcx> for OpportunisticVarResolver<'a, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !t.has_infer_types_or_consts() {
            t // micro-optimize -- if there is nothing in this type that this fold affects...
        } else {
            let t = self.infcx.shallow_resolve(t);
            t.super_fold_with(self)
        }
    }

    fn fold_const(&mut self, ct: &'tcx Const<'tcx>) -> &'tcx Const<'tcx> {
        if !ct.has_infer_types_or_consts() {
            ct // micro-optimize -- if there is nothing in this const that this fold affects...
        } else {
            let ct = self.infcx.shallow_resolve(ct);
            ct.super_fold_with(self)
        }
    }
}

/// The opportunistic region resolver opportunistically resolves regions
/// variables to the variable with the least variable id. It is used when
/// normlizing projections to avoid hitting the recursion limit by creating
/// many versions of a predicate for types that in the end have to unify.
///
/// If you want to resolve type and const variables as well, call
/// [InferCtxt::resolve_vars_if_possible] first.
pub struct OpportunisticRegionResolver<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
}

impl<'a, 'tcx> OpportunisticRegionResolver<'a, 'tcx> {
    pub fn new(infcx: &'a InferCtxt<'a, 'tcx>) -> Self {
        OpportunisticRegionResolver { infcx }
    }
}

impl<'a, 'tcx> TypeFolder<'tcx> for OpportunisticRegionResolver<'a, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
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
            ty::ReVar(rid) => {
                let resolved = self
                    .infcx
                    .inner
                    .borrow_mut()
                    .unwrap_region_constraints()
                    .opportunistic_resolve_var(rid);
                self.tcx().reuse_or_mk_region(r, ty::ReVar(resolved))
            }
            _ => r,
        }
    }

    fn fold_const(&mut self, ct: &'tcx ty::Const<'tcx>) -> &'tcx ty::Const<'tcx> {
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
pub struct UnresolvedTypeFinder<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,

    /// Used to find the type parameter name and location for error reporting.
    pub first_unresolved: Option<(Ty<'tcx>, Option<Span>)>,
}

impl<'a, 'tcx> UnresolvedTypeFinder<'a, 'tcx> {
    pub fn new(infcx: &'a InferCtxt<'a, 'tcx>) -> Self {
        UnresolvedTypeFinder { infcx, first_unresolved: None }
    }
}

impl<'a, 'tcx> TypeVisitor<'tcx> for UnresolvedTypeFinder<'a, 'tcx> {
    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<()> {
        let t = self.infcx.shallow_resolve(t);
        if t.has_infer_types() {
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
                self.first_unresolved = Some((t, ty_var_span));
                ControlFlow::BREAK
            } else {
                // Otherwise, visit its contents.
                t.super_visit_with(self)
            }
        } else {
            // All type variables in inference types must already be resolved,
            // - no need to visit the contents, continue visiting.
            ControlFlow::CONTINUE
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// FULL TYPE RESOLUTION

/// Full type resolution replaces all type and region variables with
/// their concrete results. If any variable cannot be replaced (never unified, etc)
/// then an `Err` result is returned.
pub fn fully_resolve<'a, 'tcx, T>(infcx: &InferCtxt<'a, 'tcx>, value: &T) -> FixupResult<'tcx, T>
where
    T: TypeFoldable<'tcx>,
{
    let mut full_resolver = FullTypeResolver { infcx, err: None };
    let result = value.fold_with(&mut full_resolver);
    match full_resolver.err {
        None => Ok(result),
        Some(e) => Err(e),
    }
}

// N.B. This type is not public because the protocol around checking the
// `err` field is not enforceable otherwise.
struct FullTypeResolver<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    err: Option<FixupError<'tcx>>,
}

impl<'a, 'tcx> TypeFolder<'tcx> for FullTypeResolver<'a, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !t.needs_infer() {
            t // micro-optimize -- if there is nothing in this type that this fold affects...
        } else {
            let t = self.infcx.shallow_resolve(t);
            match *t.kind() {
                ty::Infer(ty::TyVar(vid)) => {
                    self.err = Some(FixupError::UnresolvedTy(vid));
                    self.tcx().ty_error()
                }
                ty::Infer(ty::IntVar(vid)) => {
                    self.err = Some(FixupError::UnresolvedIntTy(vid));
                    self.tcx().ty_error()
                }
                ty::Infer(ty::FloatVar(vid)) => {
                    self.err = Some(FixupError::UnresolvedFloatTy(vid));
                    self.tcx().ty_error()
                }
                ty::Infer(_) => {
                    bug!("Unexpected type in full type resolver: {:?}", t);
                }
                _ => t.super_fold_with(self),
            }
        }
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            ty::ReVar(rid) => self
                .infcx
                .lexical_region_resolutions
                .borrow()
                .as_ref()
                .expect("region resolution not performed")
                .resolve_var(rid),
            _ => r,
        }
    }

    fn fold_const(&mut self, c: &'tcx ty::Const<'tcx>) -> &'tcx ty::Const<'tcx> {
        if !c.needs_infer() {
            c // micro-optimize -- if there is nothing in this const that this fold affects...
        } else {
            let c = self.infcx.shallow_resolve(c);
            match c.val {
                ty::ConstKind::Infer(InferConst::Var(vid)) => {
                    self.err = Some(FixupError::UnresolvedConst(vid));
                    return self.tcx().const_error(c.ty);
                }
                ty::ConstKind::Infer(InferConst::Fresh(_)) => {
                    bug!("Unexpected const in full const resolver: {:?}", c);
                }
                _ => {}
            }
            c.super_fold_with(self)
        }
    }
}
