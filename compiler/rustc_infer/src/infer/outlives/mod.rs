//! Various code related to computing outlives relations.

use rustc_data_structures::undo_log::UndoLogs;
use rustc_middle::traits::query::{NoSolution, OutlivesBound};
use rustc_middle::ty::{self, Region, Ty, TypeVisitableExt};
use tracing::{debug, instrument};

use self::env::OutlivesEnvironment;
use super::region_constraints::{RegionConstraintData, UndoLog};
use super::{InferCtxt, RegionResolutionError, SubregionOrigin};
use crate::infer::free_regions::RegionRelations;
use crate::infer::region_constraints::ConstraintKind;
use crate::infer::{TypeOutlivesConstraint, lexical_region_resolve};
use crate::traits::{ObligationCause, ObligationCauseCode};

pub mod env;
pub mod for_liveness;
pub mod obligations;
pub mod test_type_match;
pub(crate) mod verify;

#[instrument(level = "debug", skip(param_env), ret)]
pub fn explicit_outlives_bounds<'tcx>(
    param_env: ty::ParamEnv<'tcx>,
) -> impl Iterator<Item = OutlivesBound<'tcx>> {
    param_env
        .caller_bounds()
        .into_iter()
        .filter_map(ty::Clause::as_region_outlives_clause)
        .filter_map(ty::Binder::no_bound_vars)
        .map(|ty::OutlivesPredicate(r_a, r_b)| OutlivesBound::RegionSubRegion(r_b, r_a))
}

impl<'tcx> InferCtxt<'tcx> {
    pub fn register_outlives_constraint(
        &self,
        ty::OutlivesPredicate(arg, r2): ty::ArgOutlivesPredicate<'tcx>,
        cause: &ObligationCause<'tcx>,
    ) {
        match arg.kind() {
            ty::GenericArgKind::Lifetime(r1) => {
                self.register_region_outlives_constraint(ty::OutlivesPredicate(r1, r2), cause);
            }
            ty::GenericArgKind::Type(ty1) => {
                self.register_type_outlives_constraint(ty1, r2, cause);
            }
            ty::GenericArgKind::Const(_) => unreachable!(),
        }
    }

    pub fn register_region_outlives_constraint(
        &self,
        ty::OutlivesPredicate(r_a, r_b): ty::RegionOutlivesPredicate<'tcx>,
        cause: &ObligationCause<'tcx>,
    ) {
        let origin = SubregionOrigin::from_obligation_cause(cause, || {
            SubregionOrigin::RelateRegionParamBound(cause.span, None)
        });
        // `'a: 'b` ==> `'b <= 'a`
        self.sub_regions(origin, r_b, r_a);
    }

    /// Registers that the given region obligation must be resolved
    /// from within the scope of `body_id`. These regions are enqueued
    /// and later processed by regionck, when full type information is
    /// available (see `region_obligations` field for more
    /// information).
    #[instrument(level = "debug", skip(self))]
    pub fn register_type_outlives_constraint_inner(
        &self,
        obligation: TypeOutlivesConstraint<'tcx>,
    ) {
        let mut inner = self.inner.borrow_mut();
        inner.undo_log.push(crate::infer::UndoLog::PushTypeOutlivesConstraint);
        inner.region_obligations.push(obligation);
    }

    pub fn register_type_outlives_constraint(
        &self,
        sup_type: Ty<'tcx>,
        sub_region: Region<'tcx>,
        cause: &ObligationCause<'tcx>,
    ) {
        // `is_global` means the type has no params, infer, placeholder, or non-`'static`
        // free regions. If the type has none of these things, then we can skip registering
        // this outlives obligation since it has no components which affect lifetime
        // checking in an interesting way.
        if sup_type.is_global() {
            return;
        }

        debug!(?sup_type, ?sub_region, ?cause);
        let origin = SubregionOrigin::from_obligation_cause(cause, || {
            SubregionOrigin::RelateParamBound(
                cause.span,
                sup_type,
                match cause.code().peel_derives() {
                    ObligationCauseCode::WhereClause(_, span)
                    | ObligationCauseCode::WhereClauseInExpr(_, span, ..)
                    | ObligationCauseCode::OpaqueTypeBound(span, _)
                        if !span.is_dummy() =>
                    {
                        Some(*span)
                    }
                    _ => None,
                },
            )
        });

        self.register_type_outlives_constraint_inner(TypeOutlivesConstraint {
            sup_type,
            sub_region,
            origin,
        });
    }

    /// Process the region constraints and return any errors that
    /// result. After this, no more unification operations should be
    /// done -- or the compiler will panic -- but it is legal to use
    /// `resolve_vars_if_possible` as well as `fully_resolve`.
    ///
    /// If you are in a crate that has access to `rustc_trait_selection`,
    /// then it's probably better to use `resolve_regions`,
    /// which knows how to normalize registered region obligations.
    #[must_use]
    pub fn resolve_regions_with_normalize(
        &self,
        outlives_env: &OutlivesEnvironment<'tcx>,
        deeply_normalize_ty: impl Fn(
            ty::PolyTypeOutlivesPredicate<'tcx>,
            SubregionOrigin<'tcx>,
        ) -> Result<ty::PolyTypeOutlivesPredicate<'tcx>, NoSolution>,
    ) -> Vec<RegionResolutionError<'tcx>> {
        match self.process_registered_region_obligations(outlives_env, deeply_normalize_ty) {
            Ok(()) => {}
            Err((clause, origin)) => {
                return vec![RegionResolutionError::CannotNormalize(clause, origin)];
            }
        };

        let mut storage = {
            let mut inner = self.inner.borrow_mut();
            let inner = &mut *inner;
            assert!(
                self.tainted_by_errors().is_some() || inner.region_obligations.is_empty(),
                "region_obligations not empty: {:#?}",
                inner.region_obligations,
            );
            assert!(!UndoLogs::<UndoLog<'_>>::in_snapshot(&inner.undo_log));
            inner.region_constraint_storage.take().expect("regions already resolved")
        };

        // Filter out any region-region outlives assumptions that are implied by
        // coroutine well-formedness.
        if self.tcx.sess.opts.unstable_opts.higher_ranked_assumptions {
            storage.data.constraints.retain(|(c, _)| match c.kind {
                ConstraintKind::RegSubReg => !outlives_env
                    .higher_ranked_assumptions()
                    .contains(&ty::OutlivesPredicate(c.sup.into(), c.sub)),
                _ => true,
            });
        }

        let region_rels = &RegionRelations::new(self.tcx, outlives_env.free_region_map());

        let (lexical_region_resolutions, errors) =
            lexical_region_resolve::resolve(region_rels, storage.var_infos, storage.data);

        let old_value = self.lexical_region_resolutions.replace(Some(lexical_region_resolutions));
        assert!(old_value.is_none());

        errors
    }

    /// Trait queries just want to pass back type obligations "as is"
    pub fn take_registered_region_obligations(&self) -> Vec<TypeOutlivesConstraint<'tcx>> {
        assert!(!self.in_snapshot(), "cannot take registered region obligations in a snapshot");
        std::mem::take(&mut self.inner.borrow_mut().region_obligations)
    }

    pub fn clone_registered_region_obligations(&self) -> Vec<TypeOutlivesConstraint<'tcx>> {
        self.inner.borrow().region_obligations.clone()
    }

    pub fn register_region_assumption(&self, assumption: ty::ArgOutlivesPredicate<'tcx>) {
        let mut inner = self.inner.borrow_mut();
        inner.undo_log.push(crate::infer::UndoLog::PushRegionAssumption);
        inner.region_assumptions.push(assumption);
    }

    pub fn take_registered_region_assumptions(&self) -> Vec<ty::ArgOutlivesPredicate<'tcx>> {
        assert!(!self.in_snapshot(), "cannot take registered region assumptions in a snapshot");
        std::mem::take(&mut self.inner.borrow_mut().region_assumptions)
    }

    /// Obtains (and clears) the current set of region
    /// constraints. The inference context is still usable: further
    /// unifications will simply add new constraints.
    ///
    /// This method is not meant to be used with normal lexical region
    /// resolution. Rather, it is used in the NLL mode as a kind of
    /// interim hack: basically we run normal type-check and generate
    /// region constraints as normal, but then we take them and
    /// translate them into the form that the NLL solver
    /// understands. See the NLL module for mode details.
    pub fn take_and_reset_region_constraints(&self) -> RegionConstraintData<'tcx> {
        assert!(
            self.inner.borrow().region_obligations.is_empty(),
            "region_obligations not empty: {:#?}",
            self.inner.borrow().region_obligations
        );
        assert!(
            self.inner.borrow().region_assumptions.is_empty(),
            "region_assumptions not empty: {:#?}",
            self.inner.borrow().region_assumptions
        );

        self.inner.borrow_mut().unwrap_region_constraints().take_and_reset_data()
    }

    /// Gives temporary access to the region constraint data.
    pub fn with_region_constraints<R>(
        &self,
        op: impl FnOnce(&RegionConstraintData<'tcx>) -> R,
    ) -> R {
        let mut inner = self.inner.borrow_mut();
        op(inner.unwrap_region_constraints().data())
    }
}
