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

use rustc_type_ir::{
    AliasRelationDirection, TypeVisitableExt, Upcast, Variance,
    inherent::{IntoKind, Span as _},
    relate::{
        Relate, StructurallyRelateAliases, TypeRelation, VarianceDiagInfo,
        combine::{PredicateEmittingRelation, super_combine_consts, super_combine_tys},
    },
};

use crate::next_solver::{
    AliasTy, Binder, Const, DbInterner, Goal, ParamEnv, Predicate, PredicateKind, Region, Span, Ty,
    TyKind,
    infer::{
        InferCtxt, TypeTrace,
        relate::RelateResult,
        traits::{Obligation, PredicateObligations},
    },
};

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
pub(crate) struct LatticeOp<'infcx, 'db> {
    infcx: &'infcx InferCtxt<'db>,
    // Immutable fields
    trace: TypeTrace<'db>,
    param_env: ParamEnv<'db>,
    // Mutable fields
    kind: LatticeOpKind,
    obligations: PredicateObligations<'db>,
}

impl<'infcx, 'db> LatticeOp<'infcx, 'db> {
    pub(crate) fn new(
        infcx: &'infcx InferCtxt<'db>,
        trace: TypeTrace<'db>,
        param_env: ParamEnv<'db>,
        kind: LatticeOpKind,
    ) -> LatticeOp<'infcx, 'db> {
        LatticeOp { infcx, trace, param_env, kind, obligations: PredicateObligations::new() }
    }

    pub(crate) fn into_obligations(self) -> PredicateObligations<'db> {
        self.obligations
    }
}

impl<'db> TypeRelation<DbInterner<'db>> for LatticeOp<'_, 'db> {
    fn cx(&self) -> DbInterner<'db> {
        self.infcx.interner
    }

    fn relate_with_variance<T: Relate<DbInterner<'db>>>(
        &mut self,
        variance: Variance,
        _info: VarianceDiagInfo<DbInterner<'db>>,
        a: T,
        b: T,
    ) -> RelateResult<'db, T> {
        match variance {
            Variance::Invariant => {
                self.obligations.extend(
                    self.infcx.at(&self.trace.cause, self.param_env).eq(a, b)?.into_obligations(),
                );
                Ok(a)
            }
            Variance::Covariant => self.relate(a, b),
            // FIXME(#41044) -- not correct, need test
            Variance::Bivariant => Ok(a),
            Variance::Contravariant => {
                self.kind = self.kind.invert();
                let res = self.relate(a, b);
                self.kind = self.kind.invert();
                res
            }
        }
    }

    /// Relates two types using a given lattice.
    fn tys(&mut self, a: Ty<'db>, b: Ty<'db>) -> RelateResult<'db, Ty<'db>> {
        if a == b {
            return Ok(a);
        }

        let infcx = self.infcx;

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
            (TyKind::Infer(rustc_type_ir::TyVar(..)), _) => {
                let v = infcx.next_ty_var();
                self.relate_bound(v, b, a)?;
                Ok(v)
            }
            (_, TyKind::Infer(rustc_type_ir::TyVar(..))) => {
                let v = infcx.next_ty_var();
                self.relate_bound(v, a, b)?;
                Ok(v)
            }

            (
                TyKind::Alias(rustc_type_ir::Opaque, AliasTy { def_id: a_def_id, .. }),
                TyKind::Alias(rustc_type_ir::Opaque, AliasTy { def_id: b_def_id, .. }),
            ) if a_def_id == b_def_id => super_combine_tys(infcx, self, a, b),

            _ => super_combine_tys(infcx, self, a, b),
        }
    }

    fn regions(&mut self, a: Region<'db>, b: Region<'db>) -> RelateResult<'db, Region<'db>> {
        let mut inner = self.infcx.inner.borrow_mut();
        let mut constraints = inner.unwrap_region_constraints();
        Ok(match self.kind {
            // GLB(&'static u8, &'a u8) == &RegionLUB('static, 'a) u8 == &'static u8
            LatticeOpKind::Glb => constraints.lub_regions(self.cx(), a, b),

            // LUB(&'static u8, &'a u8) == &RegionGLB('static, 'a) u8 == &'a u8
            LatticeOpKind::Lub => constraints.glb_regions(self.cx(), a, b),
        })
    }

    fn consts(&mut self, a: Const<'db>, b: Const<'db>) -> RelateResult<'db, Const<'db>> {
        super_combine_consts(self.infcx, self, a, b)
    }

    fn binders<T>(
        &mut self,
        a: Binder<'db, T>,
        b: Binder<'db, T>,
    ) -> RelateResult<'db, Binder<'db, T>>
    where
        T: Relate<DbInterner<'db>>,
    {
        // GLB/LUB of a binder and itself is just itself
        if a == b {
            return Ok(a);
        }

        if a.skip_binder().has_escaping_bound_vars() || b.skip_binder().has_escaping_bound_vars() {
            // When higher-ranked types are involved, computing the GLB/LUB is
            // very challenging, switch to invariance. This is obviously
            // overly conservative but works ok in practice.
            self.relate_with_variance(Variance::Invariant, VarianceDiagInfo::default(), a, b)?;
            Ok(a)
        } else {
            Ok(Binder::dummy(self.relate(a.skip_binder(), b.skip_binder())?))
        }
    }
}

impl<'infcx, 'db> LatticeOp<'infcx, 'db> {
    // Relates the type `v` to `a` and `b` such that `v` represents
    // the LUB/GLB of `a` and `b` as appropriate.
    //
    // Subtle hack: ordering *may* be significant here. This method
    // relates `v` to `a` first, which may help us to avoid unnecessary
    // type variable obligations. See caller for details.
    fn relate_bound(&mut self, v: Ty<'db>, a: Ty<'db>, b: Ty<'db>) -> RelateResult<'db, ()> {
        let at = self.infcx.at(&self.trace.cause, self.param_env);
        match self.kind {
            LatticeOpKind::Glb => {
                self.obligations.extend(at.sub(v, a)?.into_obligations());
                self.obligations.extend(at.sub(v, b)?.into_obligations());
            }
            LatticeOpKind::Lub => {
                self.obligations.extend(at.sub(a, v)?.into_obligations());
                self.obligations.extend(at.sub(b, v)?.into_obligations());
            }
        }
        Ok(())
    }
}

impl<'db> PredicateEmittingRelation<InferCtxt<'db>> for LatticeOp<'_, 'db> {
    fn span(&self) -> Span {
        Span::dummy()
    }

    fn structurally_relate_aliases(&self) -> StructurallyRelateAliases {
        StructurallyRelateAliases::No
    }

    fn param_env(&self) -> ParamEnv<'db> {
        self.param_env
    }

    fn register_predicates(
        &mut self,
        preds: impl IntoIterator<Item: Upcast<DbInterner<'db>, Predicate<'db>>>,
    ) {
        self.obligations.extend(preds.into_iter().map(|pred| {
            Obligation::new(self.infcx.interner, self.trace.cause.clone(), self.param_env, pred)
        }))
    }

    fn register_goals(&mut self, goals: impl IntoIterator<Item = Goal<'db, Predicate<'db>>>) {
        self.obligations.extend(goals.into_iter().map(|goal| {
            Obligation::new(
                self.infcx.interner,
                self.trace.cause.clone(),
                goal.param_env,
                goal.predicate,
            )
        }))
    }

    fn register_alias_relate_predicate(&mut self, a: Ty<'db>, b: Ty<'db>) {
        self.register_predicates([Binder::dummy(PredicateKind::AliasRelate(
            a.into(),
            b.into(),
            // FIXME(deferred_projection_equality): This isn't right, I think?
            AliasRelationDirection::Equate,
        ))]);
    }
}
