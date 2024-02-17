//! This code is kind of an alternate way of doing subtyping,
//! supertyping, and type equating, distinct from the `combine.rs`
//! code but very similar in its effect and design. Eventually the two
//! ought to be merged. This code is intended for use in NLL and chalk.
//!
//! Here are the key differences:
//!
//! - This code may choose to bypass some checks (e.g., the occurs check)
//!   in the case where we know that there are no unbound type inference
//!   variables. This is the case for NLL, because at NLL time types are fully
//!   inferred up-to regions.
//! - This code uses "universes" to handle higher-ranked regions and
//!   not the leak-check. This is "more correct" than what rustc does
//!   and we are generally migrating in this direction, but NLL had to
//!   get there first.
//!
//! Also, this code assumes that there are no bound types at all, not even
//! free ones. This is ok because:
//! - we are not relating anything quantified over some type variable
//! - we will have instantiated all the bound type vars already (the one
//!   thing we relate in chalk are basically domain goals and their
//!   constituents)

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::fold::FnMutDelegate;
use rustc_middle::ty::relate::{Relate, RelateResult, TypeRelation};
use rustc_middle::ty::TypeVisitableExt;
use rustc_middle::ty::{self, InferConst, Ty, TyCtxt};
use rustc_span::{Span, Symbol};

use super::combine::ObligationEmittingRelation;
use crate::infer::InferCtxt;
use crate::infer::{TypeVariableOrigin, TypeVariableOriginKind};
use crate::traits::{Obligation, PredicateObligations};

pub struct TypeRelating<'me, 'tcx, D>
where
    D: TypeRelatingDelegate<'tcx>,
{
    infcx: &'me InferCtxt<'tcx>,

    /// Callback to use when we deduce an outlives relationship.
    delegate: D,

    /// How are we relating `a` and `b`?
    ///
    /// - Covariant means `a <: b`.
    /// - Contravariant means `b <: a`.
    /// - Invariant means `a == b`.
    /// - Bivariant means that it doesn't matter.
    ambient_variance: ty::Variance,

    ambient_variance_info: ty::VarianceDiagInfo<'tcx>,
}

pub trait TypeRelatingDelegate<'tcx> {
    fn param_env(&self) -> ty::ParamEnv<'tcx>;
    fn span(&self) -> Span;

    /// Push a constraint `sup: sub` -- this constraint must be
    /// satisfied for the two types to be related. `sub` and `sup` may
    /// be regions from the type or new variables created through the
    /// delegate.
    fn push_outlives(
        &mut self,
        sup: ty::Region<'tcx>,
        sub: ty::Region<'tcx>,
        info: ty::VarianceDiagInfo<'tcx>,
    );

    fn register_obligations(&mut self, obligations: PredicateObligations<'tcx>);

    /// Creates a new universe index. Used when instantiating placeholders.
    fn create_next_universe(&mut self) -> ty::UniverseIndex;

    /// Creates a new region variable representing a higher-ranked
    /// region that is instantiated existentially. This creates an
    /// inference variable, typically.
    ///
    /// So e.g., if you have `for<'a> fn(..) <: for<'b> fn(..)`, then
    /// we will invoke this method to instantiate `'a` with an
    /// inference variable (though `'b` would be instantiated first,
    /// as a placeholder).
    fn next_existential_region_var(
        &mut self,
        was_placeholder: bool,
        name: Option<Symbol>,
    ) -> ty::Region<'tcx>;

    /// Creates a new region variable representing a
    /// higher-ranked region that is instantiated universally.
    /// This creates a new region placeholder, typically.
    ///
    /// So e.g., if you have `for<'a> fn(..) <: for<'b> fn(..)`, then
    /// we will invoke this method to instantiate `'b` with a
    /// placeholder region.
    fn next_placeholder_region(&mut self, placeholder: ty::PlaceholderRegion) -> ty::Region<'tcx>;

    /// Enables some optimizations if we do not expect inference variables
    /// in the RHS of the relation.
    fn forbid_inference_vars() -> bool;
}

impl<'me, 'tcx, D> TypeRelating<'me, 'tcx, D>
where
    D: TypeRelatingDelegate<'tcx>,
{
    pub fn new(infcx: &'me InferCtxt<'tcx>, delegate: D, ambient_variance: ty::Variance) -> Self {
        Self {
            infcx,
            delegate,
            ambient_variance,
            ambient_variance_info: ty::VarianceDiagInfo::default(),
        }
    }

    fn ambient_covariance(&self) -> bool {
        match self.ambient_variance {
            ty::Variance::Covariant | ty::Variance::Invariant => true,
            ty::Variance::Contravariant | ty::Variance::Bivariant => false,
        }
    }

    fn ambient_contravariance(&self) -> bool {
        match self.ambient_variance {
            ty::Variance::Contravariant | ty::Variance::Invariant => true,
            ty::Variance::Covariant | ty::Variance::Bivariant => false,
        }
    }

    /// Push a new outlives requirement into our output set of
    /// constraints.
    fn push_outlives(
        &mut self,
        sup: ty::Region<'tcx>,
        sub: ty::Region<'tcx>,
        info: ty::VarianceDiagInfo<'tcx>,
    ) {
        debug!("push_outlives({:?}: {:?})", sup, sub);

        self.delegate.push_outlives(sup, sub, info);
    }

    fn relate_opaques(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, ()> {
        let infcx = self.infcx;
        debug_assert!(!infcx.next_trait_solver());
        let (a, b) = if self.a_is_expected() { (a, b) } else { (b, a) };
        // `handle_opaque_type` cannot handle subtyping, so to support subtyping
        // we instead eagerly generalize here. This is a bit of a mess but will go
        // away once we're using the new solver.
        let mut enable_subtyping = |ty, ty_is_expected| {
            let ty_vid = infcx.next_ty_var_id_in_universe(
                TypeVariableOrigin {
                    kind: TypeVariableOriginKind::MiscVariable,
                    span: self.delegate.span(),
                },
                ty::UniverseIndex::ROOT,
            );

            let variance = if ty_is_expected {
                self.ambient_variance
            } else {
                self.ambient_variance.xform(ty::Contravariant)
            };

            self.infcx.instantiate_ty_var(self, ty_is_expected, ty_vid, variance, ty)?;
            Ok(infcx.resolve_vars_if_possible(Ty::new_infer(infcx.tcx, ty::TyVar(ty_vid))))
        };

        let (a, b) = match (a.kind(), b.kind()) {
            (&ty::Alias(ty::Opaque, ..), _) => (a, enable_subtyping(b, false)?),
            (_, &ty::Alias(ty::Opaque, ..)) => (enable_subtyping(a, true)?, b),
            _ => unreachable!(
                "expected at least one opaque type in `relate_opaques`, got {a} and {b}."
            ),
        };
        let cause = ObligationCause::dummy_with_span(self.delegate.span());
        let obligations =
            infcx.handle_opaque_type(a, b, true, &cause, self.delegate.param_env())?.obligations;
        self.delegate.register_obligations(obligations);
        Ok(())
    }

    fn enter_forall<T, U>(
        &mut self,
        binder: ty::Binder<'tcx, T>,
        f: impl FnOnce(&mut Self, T) -> U,
    ) -> U
    where
        T: ty::TypeFoldable<TyCtxt<'tcx>> + Copy,
    {
        let value = if let Some(inner) = binder.no_bound_vars() {
            inner
        } else {
            let mut next_region = {
                let nll_delegate = &mut self.delegate;
                let mut lazy_universe = None;

                move |br: ty::BoundRegion| {
                    // The first time this closure is called, create a
                    // new universe for the placeholders we will make
                    // from here out.
                    let universe = lazy_universe.unwrap_or_else(|| {
                        let universe = nll_delegate.create_next_universe();
                        lazy_universe = Some(universe);
                        universe
                    });

                    let placeholder = ty::PlaceholderRegion { universe, bound: br };
                    debug!(?placeholder);
                    let placeholder_reg = nll_delegate.next_placeholder_region(placeholder);
                    debug!(?placeholder_reg);

                    placeholder_reg
                }
            };

            let delegate = FnMutDelegate {
                regions: &mut next_region,
                types: &mut |_bound_ty: ty::BoundTy| {
                    unreachable!("we only replace regions in nll_relate, not types")
                },
                consts: &mut |_bound_var: ty::BoundVar, _ty| {
                    unreachable!("we only replace regions in nll_relate, not consts")
                },
            };

            self.infcx.tcx.replace_bound_vars_uncached(binder, delegate)
        };

        debug!(?value);
        f(self, value)
    }

    #[instrument(skip(self), level = "debug")]
    fn instantiate_binder_with_existentials<T>(&mut self, binder: ty::Binder<'tcx, T>) -> T
    where
        T: ty::TypeFoldable<TyCtxt<'tcx>> + Copy,
    {
        if let Some(inner) = binder.no_bound_vars() {
            return inner;
        }

        let mut next_region = {
            let nll_delegate = &mut self.delegate;
            let mut reg_map = FxHashMap::default();

            move |br: ty::BoundRegion| {
                if let Some(ex_reg_var) = reg_map.get(&br) {
                    return *ex_reg_var;
                } else {
                    let ex_reg_var =
                        nll_delegate.next_existential_region_var(true, br.kind.get_name());
                    debug!(?ex_reg_var);
                    reg_map.insert(br, ex_reg_var);

                    ex_reg_var
                }
            }
        };

        let delegate = FnMutDelegate {
            regions: &mut next_region,
            types: &mut |_bound_ty: ty::BoundTy| {
                unreachable!("we only replace regions in nll_relate, not types")
            },
            consts: &mut |_bound_var: ty::BoundVar, _ty| {
                unreachable!("we only replace regions in nll_relate, not consts")
            },
        };

        let replaced = self.infcx.tcx.replace_bound_vars_uncached(binder, delegate);
        debug!(?replaced);

        replaced
    }
}

impl<'tcx, D> TypeRelation<'tcx> for TypeRelating<'_, 'tcx, D>
where
    D: TypeRelatingDelegate<'tcx>,
{
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn tag(&self) -> &'static str {
        "nll::subtype"
    }

    fn a_is_expected(&self) -> bool {
        true
    }

    #[instrument(skip(self, info), level = "trace", ret)]
    fn relate_with_variance<T: Relate<'tcx>>(
        &mut self,
        variance: ty::Variance,
        info: ty::VarianceDiagInfo<'tcx>,
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        let old_ambient_variance = self.ambient_variance;
        self.ambient_variance = self.ambient_variance.xform(variance);
        self.ambient_variance_info = self.ambient_variance_info.xform(info);

        debug!(?self.ambient_variance);
        // In a bivariant context this always succeeds.
        let r =
            if self.ambient_variance == ty::Variance::Bivariant { a } else { self.relate(a, b)? };

        self.ambient_variance = old_ambient_variance;

        Ok(r)
    }

    #[instrument(skip(self), level = "debug")]
    fn tys(&mut self, a: Ty<'tcx>, mut b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        let infcx = self.infcx;

        let a = self.infcx.shallow_resolve(a);

        if !D::forbid_inference_vars() {
            b = self.infcx.shallow_resolve(b);
        } else {
            assert!(!b.has_non_region_infer(), "unexpected inference var {:?}", b);
        }

        if a == b {
            return Ok(a);
        }

        match (a.kind(), b.kind()) {
            (&ty::Infer(ty::TyVar(a_vid)), &ty::Infer(ty::TyVar(b_vid))) => {
                match self.ambient_variance {
                    ty::Invariant => infcx.inner.borrow_mut().type_variables().equate(a_vid, b_vid),
                    _ => unimplemented!(),
                }
            }

            (&ty::Infer(ty::TyVar(a_vid)), _) => {
                infcx.instantiate_ty_var(self, true, a_vid, self.ambient_variance, b)?
            }

            (_, &ty::Infer(ty::TyVar(b_vid))) => infcx.instantiate_ty_var(
                self,
                false,
                b_vid,
                self.ambient_variance.xform(ty::Contravariant),
                a,
            )?,

            (
                &ty::Alias(ty::Opaque, ty::AliasTy { def_id: a_def_id, .. }),
                &ty::Alias(ty::Opaque, ty::AliasTy { def_id: b_def_id, .. }),
            ) if a_def_id == b_def_id || infcx.next_trait_solver() => {
                infcx.super_combine_tys(self, a, b).map(|_| ()).or_else(|err| {
                    // This behavior is only there for the old solver, the new solver
                    // shouldn't ever fail. Instead, it unconditionally emits an
                    // alias-relate goal.
                    assert!(!self.infcx.next_trait_solver());
                    self.tcx().dcx().span_delayed_bug(
                        self.delegate.span(),
                        "failure to relate an opaque to itself should result in an error later on",
                    );
                    if a_def_id.is_local() { self.relate_opaques(a, b) } else { Err(err) }
                })?;
            }
            (&ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }), _)
            | (_, &ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }))
                if def_id.is_local() && !self.infcx.next_trait_solver() =>
            {
                self.relate_opaques(a, b)?;
            }

            _ => {
                debug!(?a, ?b, ?self.ambient_variance);

                // Will also handle unification of `IntVar` and `FloatVar`.
                self.infcx.super_combine_tys(self, a, b)?;
            }
        }

        Ok(a)
    }

    #[instrument(skip(self), level = "trace")]
    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        debug!(?self.ambient_variance);

        if self.ambient_covariance() {
            // Covariant: &'a u8 <: &'b u8. Hence, `'a: 'b`.
            self.push_outlives(a, b, self.ambient_variance_info);
        }

        if self.ambient_contravariance() {
            // Contravariant: &'b u8 <: &'a u8. Hence, `'b: 'a`.
            self.push_outlives(b, a, self.ambient_variance_info);
        }

        Ok(a)
    }

    fn consts(
        &mut self,
        a: ty::Const<'tcx>,
        mut b: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        let a = self.infcx.shallow_resolve(a);

        if !D::forbid_inference_vars() {
            b = self.infcx.shallow_resolve(b);
        }

        match b.kind() {
            ty::ConstKind::Infer(InferConst::Var(_)) if D::forbid_inference_vars() => {
                // Forbid inference variables in the RHS.
                self.infcx.dcx().span_delayed_bug(
                    self.delegate.span(),
                    format!("unexpected inference var {b:?}",),
                );
                Ok(a)
            }
            // FIXME(invariance): see the related FIXME above.
            _ => self.infcx.super_combine_consts(self, a, b),
        }
    }

    #[instrument(skip(self), level = "trace")]
    fn binders<T>(
        &mut self,
        a: ty::Binder<'tcx, T>,
        b: ty::Binder<'tcx, T>,
    ) -> RelateResult<'tcx, ty::Binder<'tcx, T>>
    where
        T: Relate<'tcx>,
    {
        // We want that
        //
        // ```
        // for<'a> fn(&'a u32) -> &'a u32 <:
        //   fn(&'b u32) -> &'b u32
        // ```
        //
        // but not
        //
        // ```
        // fn(&'a u32) -> &'a u32 <:
        //   for<'b> fn(&'b u32) -> &'b u32
        // ```
        //
        // We therefore proceed as follows:
        //
        // - Instantiate binders on `b` universally, yielding a universe U1.
        // - Instantiate binders on `a` existentially in U1.

        debug!(?self.ambient_variance);

        if let (Some(a), Some(b)) = (a.no_bound_vars(), b.no_bound_vars()) {
            // Fast path for the common case.
            self.relate(a, b)?;
            return Ok(ty::Binder::dummy(a));
        }

        if self.ambient_covariance() {
            // Covariance, so we want `for<..> A <: for<..> B` --
            // therefore we compare any instantiation of A (i.e., A
            // instantiated with existentials) against every
            // instantiation of B (i.e., B instantiated with
            // universals).

            // Reset the ambient variance to covariant. This is needed
            // to correctly handle cases like
            //
            //     for<'a> fn(&'a u32, &'a u32) == for<'b, 'c> fn(&'b u32, &'c u32)
            //
            // Somewhat surprisingly, these two types are actually
            // **equal**, even though the one on the right looks more
            // polymorphic. The reason is due to subtyping. To see it,
            // consider that each function can call the other:
            //
            // - The left function can call the right with `'b` and
            //   `'c` both equal to `'a`
            //
            // - The right function can call the left with `'a` set to
            //   `{P}`, where P is the point in the CFG where the call
            //   itself occurs. Note that `'b` and `'c` must both
            //   include P. At the point, the call works because of
            //   subtyping (i.e., `&'b u32 <: &{P} u32`).
            let variance = std::mem::replace(&mut self.ambient_variance, ty::Variance::Covariant);

            // Note: the order here is important. Create the placeholders first, otherwise
            // we assign the wrong universe to the existential!
            self.enter_forall(b, |this, b| {
                let a = this.instantiate_binder_with_existentials(a);
                this.relate(a, b)
            })?;

            self.ambient_variance = variance;
        }

        if self.ambient_contravariance() {
            // Contravariance, so we want `for<..> A :> for<..> B`
            // -- therefore we compare every instantiation of A (i.e.,
            // A instantiated with universals) against any
            // instantiation of B (i.e., B instantiated with
            // existentials). Opposite of above.

            // Reset ambient variance to contravariance. See the
            // covariant case above for an explanation.
            let variance =
                std::mem::replace(&mut self.ambient_variance, ty::Variance::Contravariant);

            self.enter_forall(a, |this, a| {
                let b = this.instantiate_binder_with_existentials(b);
                this.relate(a, b)
            })?;

            self.ambient_variance = variance;
        }

        Ok(a)
    }
}

impl<'tcx, D> ObligationEmittingRelation<'tcx> for TypeRelating<'_, 'tcx, D>
where
    D: TypeRelatingDelegate<'tcx>,
{
    fn span(&self) -> Span {
        self.delegate.span()
    }

    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.delegate.param_env()
    }

    fn register_predicates(&mut self, obligations: impl IntoIterator<Item: ty::ToPredicate<'tcx>>) {
        self.delegate.register_obligations(
            obligations
                .into_iter()
                .map(|to_pred| {
                    Obligation::new(self.tcx(), ObligationCause::dummy(), self.param_env(), to_pred)
                })
                .collect(),
        );
    }

    fn register_obligations(&mut self, obligations: PredicateObligations<'tcx>) {
        self.delegate.register_obligations(obligations);
    }

    fn alias_relate_direction(&self) -> ty::AliasRelationDirection {
        unreachable!("manually overridden to handle ty::Variance::Contravariant ambient variance")
    }

    fn register_type_relate_obligation(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) {
        self.register_predicates([ty::Binder::dummy(match self.ambient_variance {
            ty::Variance::Covariant => ty::PredicateKind::AliasRelate(
                a.into(),
                b.into(),
                ty::AliasRelationDirection::Subtype,
            ),
            // a :> b is b <: a
            ty::Variance::Contravariant => ty::PredicateKind::AliasRelate(
                b.into(),
                a.into(),
                ty::AliasRelationDirection::Subtype,
            ),
            ty::Variance::Invariant => ty::PredicateKind::AliasRelate(
                a.into(),
                b.into(),
                ty::AliasRelationDirection::Equate,
            ),
            // FIXME(deferred_projection_equality): Implement this when we trigger it.
            // Probably just need to do nothing here.
            ty::Variance::Bivariant => {
                unreachable!("cannot defer an alias-relate goal with Bivariant variance (yet?)")
            }
        })]);
    }
}
