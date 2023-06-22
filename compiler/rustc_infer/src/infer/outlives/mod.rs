//! Various code related to computing outlives relations.
use self::env::OutlivesEnvironment;
use super::region_constraints::RegionConstraintData;
use super::{InferCtxt, RegionResolutionError};
use crate::infer::free_regions::RegionRelations;
use crate::infer::lexical_region_resolve::{self, LexicalRegionResolutions};
use rustc_middle::traits::query::OutlivesBound;
use rustc_middle::ty;

pub mod components;
pub mod env;
pub mod obligations;
pub mod test_type_match;
pub mod verify;

#[instrument(level = "debug", skip(param_env), ret)]
pub fn explicit_outlives_bounds<'tcx>(
    param_env: ty::ParamEnv<'tcx>,
) -> impl Iterator<Item = OutlivesBound<'tcx>> + 'tcx {
    param_env
        .caller_bounds()
        .into_iter()
        .map(ty::Clause::kind)
        .filter_map(ty::Binder::no_bound_vars)
        .filter_map(move |kind| match kind {
            ty::ClauseKind::RegionOutlives(ty::OutlivesPredicate(r_a, r_b)) => {
                Some(OutlivesBound::RegionSubRegion(r_b, r_a))
            }
            ty::ClauseKind::Trait(_)
            | ty::ClauseKind::TypeOutlives(_)
            | ty::ClauseKind::Projection(_)
            | ty::ClauseKind::ConstArgHasType(_, _)
            | ty::ClauseKind::WellFormed(_)
            | ty::ClauseKind::ConstEvaluatable(_) => None,
        })
}

impl<'tcx> InferCtxt<'tcx> {
    pub fn skip_region_resolution(&self) {
        let (var_infos, _) = {
            let mut inner = self.inner.borrow_mut();
            let inner = &mut *inner;
            // Note: `inner.region_obligations` may not be empty, because we
            // didn't necessarily call `process_registered_region_obligations`.
            // This is okay, because that doesn't introduce new vars.
            inner
                .region_constraint_storage
                .take()
                .expect("regions already resolved")
                .with_log(&mut inner.undo_log)
                .into_infos_and_data()
        };

        let lexical_region_resolutions = LexicalRegionResolutions {
            values: rustc_index::IndexVec::from_elem_n(
                crate::infer::lexical_region_resolve::VarValue::Value(self.tcx.lifetimes.re_erased),
                var_infos.len(),
            ),
        };

        let old_value = self.lexical_region_resolutions.replace(Some(lexical_region_resolutions));
        assert!(old_value.is_none());
    }

    /// Process the region constraints and return any errors that
    /// result. After this, no more unification operations should be
    /// done -- or the compiler will panic -- but it is legal to use
    /// `resolve_vars_if_possible` as well as `fully_resolve`.
    #[must_use]
    pub fn resolve_regions(
        &self,
        outlives_env: &OutlivesEnvironment<'tcx>,
    ) -> Vec<RegionResolutionError<'tcx>> {
        self.process_registered_region_obligations(outlives_env);

        let (var_infos, data) = {
            let mut inner = self.inner.borrow_mut();
            let inner = &mut *inner;
            assert!(
                self.tainted_by_errors().is_some() || inner.region_obligations.is_empty(),
                "region_obligations not empty: {:#?}",
                inner.region_obligations
            );
            inner
                .region_constraint_storage
                .take()
                .expect("regions already resolved")
                .with_log(&mut inner.undo_log)
                .into_infos_and_data()
        };

        let region_rels = &RegionRelations::new(self.tcx, outlives_env.free_region_map());

        let (lexical_region_resolutions, errors) =
            lexical_region_resolve::resolve(outlives_env.param_env, region_rels, var_infos, data);

        let old_value = self.lexical_region_resolutions.replace(Some(lexical_region_resolutions));
        assert!(old_value.is_none());

        errors
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
