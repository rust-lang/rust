//! # Lattice variables
//!
//! Generic code for operating on [lattices] of inference variables
//! that are characterized by an upper- and lower-bound.
//!
//! The code is defined quite generically so that it can be
//! applied both to type variables, which represent types being inferred,
//! and fn variables, which represent function types being inferred.
//! (It may eventually be applied to their types as well.)
//! In some cases, the functions are also generic with respect to the
//! operation on the lattice (GLB vs LUB).
//!
//! ## Note
//!
//! Although all the functions are generic, for simplicity, comments in the source code
//! generally refer to type variables and the LUB operation.
//!
//! [lattices]: https://en.wikipedia.org/wiki/Lattice_(order)

use rustc_middle::traits::solve::Goal;
use rustc_middle::ty::relate::{Relate, RelateResult, TypeRelation};
use rustc_middle::ty::{self, Ty, TyCtxt, TyVar, TypeVisitableExt};
use rustc_span::Span;
use tracing::{debug, instrument};

use super::StructurallyRelateAliases;
use super::combine::{CombineFields, PredicateEmittingRelation};
use crate::infer::{DefineOpaqueTypes, InferCtxt, SubregionOrigin};

#[derive(Clone, Copy)]
pub(crate) enum LatticeOpKind {
    Glb,
    Lub,
}

impl LatticeOpKind {
    fn invert(self) -> Self {
        match self {
            LatticeOpKind::Glb => LatticeOpKind::Lub,
            LatticeOpKind::Lub => LatticeOpKind::Glb,
        }
    }
}

/// A greatest lower bound" (common subtype) or least upper bound (common supertype).
pub(crate) struct LatticeOp<'combine, 'infcx, 'tcx> {
    fields: &'combine mut CombineFields<'infcx, 'tcx>,
    kind: LatticeOpKind,
}

impl<'combine, 'infcx, 'tcx> LatticeOp<'combine, 'infcx, 'tcx> {
    pub(crate) fn new(
        fields: &'combine mut CombineFields<'infcx, 'tcx>,
        kind: LatticeOpKind,
    ) -> LatticeOp<'combine, 'infcx, 'tcx> {
        LatticeOp { fields, kind }
    }
}

impl<'tcx> TypeRelation<TyCtxt<'tcx>> for LatticeOp<'_, '_, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.fields.tcx()
    }

    fn relate_with_variance<T: Relate<TyCtxt<'tcx>>>(
        &mut self,
        variance: ty::Variance,
        _info: ty::VarianceDiagInfo<TyCtxt<'tcx>>,
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        match variance {
            ty::Invariant => self.fields.equate(StructurallyRelateAliases::No).relate(a, b),
            ty::Covariant => self.relate(a, b),
            // FIXME(#41044) -- not correct, need test
            ty::Bivariant => Ok(a),
            ty::Contravariant => {
                self.kind = self.kind.invert();
                let res = self.relate(a, b);
                self.kind = self.kind.invert();
                res
            }
        }
    }

    /// Relates two types using a given lattice.
    #[instrument(skip(self), level = "trace")]
    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        if a == b {
            return Ok(a);
        }

        let infcx = self.fields.infcx;

        let a = infcx.shallow_resolve(a);
        let b = infcx.shallow_resolve(b);

        match (a.kind(), b.kind()) {
            // If one side is known to be a variable and one is not,
            // create a variable (`v`) to represent the LUB. Make sure to
            // relate `v` to the non-type-variable first (by passing it
            // first to `relate_bound`). Otherwise, we would produce a
            // subtype obligation that must then be processed.
            //
            // Example: if the LHS is a type variable, and RHS is
            // `Box<i32>`, then we current compare `v` to the RHS first,
            // which will instantiate `v` with `Box<i32>`. Then when `v`
            // is compared to the LHS, we instantiate LHS with `Box<i32>`.
            // But if we did in reverse order, we would create a `v <:
            // LHS` (or vice versa) constraint and then instantiate
            // `v`. This would require further processing to achieve same
            // end-result; in particular, this screws up some of the logic
            // in coercion, which expects LUB to figure out that the LHS
            // is (e.g.) `Box<i32>`. A more obvious solution might be to
            // iterate on the subtype obligations that are returned, but I
            // think this suffices. -nmatsakis
            (&ty::Infer(TyVar(..)), _) => {
                let v = infcx.next_ty_var(self.fields.trace.cause.span);
                self.relate_bound(v, b, a)?;
                Ok(v)
            }
            (_, &ty::Infer(TyVar(..))) => {
                let v = infcx.next_ty_var(self.fields.trace.cause.span);
                self.relate_bound(v, a, b)?;
                Ok(v)
            }

            (
                &ty::Alias(ty::Opaque, ty::AliasTy { def_id: a_def_id, .. }),
                &ty::Alias(ty::Opaque, ty::AliasTy { def_id: b_def_id, .. }),
            ) if a_def_id == b_def_id => infcx.super_combine_tys(self, a, b),

            (&ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }), _)
            | (_, &ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }))
                if self.fields.define_opaque_types == DefineOpaqueTypes::Yes
                    && def_id.is_local()
                    && !infcx.next_trait_solver() =>
            {
                self.register_goals(infcx.handle_opaque_type(
                    a,
                    b,
                    self.span(),
                    self.param_env(),
                )?);
                Ok(a)
            }

            _ => infcx.super_combine_tys(self, a, b),
        }
    }

    #[instrument(skip(self), level = "trace")]
    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        let origin = SubregionOrigin::Subtype(Box::new(self.fields.trace.clone()));
        let mut inner = self.fields.infcx.inner.borrow_mut();
        let mut constraints = inner.unwrap_region_constraints();
        Ok(match self.kind {
            // GLB(&'static u8, &'a u8) == &RegionLUB('static, 'a) u8 == &'static u8
            LatticeOpKind::Glb => constraints.lub_regions(self.cx(), origin, a, b),

            // LUB(&'static u8, &'a u8) == &RegionGLB('static, 'a) u8 == &'a u8
            LatticeOpKind::Lub => constraints.glb_regions(self.cx(), origin, a, b),
        })
    }

    #[instrument(skip(self), level = "trace")]
    fn consts(
        &mut self,
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        self.fields.infcx.super_combine_consts(self, a, b)
    }

    fn binders<T>(
        &mut self,
        a: ty::Binder<'tcx, T>,
        b: ty::Binder<'tcx, T>,
    ) -> RelateResult<'tcx, ty::Binder<'tcx, T>>
    where
        T: Relate<TyCtxt<'tcx>>,
    {
        // GLB/LUB of a binder and itself is just itself
        if a == b {
            return Ok(a);
        }

        debug!("binders(a={:?}, b={:?})", a, b);
        if a.skip_binder().has_escaping_bound_vars() || b.skip_binder().has_escaping_bound_vars() {
            // When higher-ranked types are involved, computing the GLB/LUB is
            // very challenging, switch to invariance. This is obviously
            // overly conservative but works ok in practice.
            self.relate_with_variance(ty::Invariant, ty::VarianceDiagInfo::default(), a, b)?;
            Ok(a)
        } else {
            Ok(ty::Binder::dummy(self.relate(a.skip_binder(), b.skip_binder())?))
        }
    }
}

impl<'combine, 'infcx, 'tcx> LatticeOp<'combine, 'infcx, 'tcx> {
    // Relates the type `v` to `a` and `b` such that `v` represents
    // the LUB/GLB of `a` and `b` as appropriate.
    //
    // Subtle hack: ordering *may* be significant here. This method
    // relates `v` to `a` first, which may help us to avoid unnecessary
    // type variable obligations. See caller for details.
    fn relate_bound(&mut self, v: Ty<'tcx>, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, ()> {
        let mut sub = self.fields.sub();
        match self.kind {
            LatticeOpKind::Glb => {
                sub.relate(v, a)?;
                sub.relate(v, b)?;
            }
            LatticeOpKind::Lub => {
                sub.relate(a, v)?;
                sub.relate(b, v)?;
            }
        }
        Ok(())
    }
}

impl<'tcx> PredicateEmittingRelation<InferCtxt<'tcx>> for LatticeOp<'_, '_, 'tcx> {
    fn span(&self) -> Span {
        self.fields.trace.span()
    }

    fn structurally_relate_aliases(&self) -> StructurallyRelateAliases {
        StructurallyRelateAliases::No
    }

    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.fields.param_env
    }

    fn register_predicates(
        &mut self,
        obligations: impl IntoIterator<Item: ty::Upcast<TyCtxt<'tcx>, ty::Predicate<'tcx>>>,
    ) {
        self.fields.register_predicates(obligations);
    }

    fn register_goals(
        &mut self,
        obligations: impl IntoIterator<Item = Goal<'tcx, ty::Predicate<'tcx>>>,
    ) {
        self.fields.register_obligations(obligations);
    }

    fn register_alias_relate_predicate(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) {
        self.register_predicates([ty::Binder::dummy(ty::PredicateKind::AliasRelate(
            a.into(),
            b.into(),
            // FIXME(deferred_projection_equality): This isn't right, I think?
            ty::AliasRelationDirection::Equate,
        ))]);
    }
}
