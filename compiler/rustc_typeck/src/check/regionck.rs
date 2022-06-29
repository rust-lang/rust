use crate::outlives::outlives_bounds::InferCtxtExt as _;
use rustc_data_structures::stable_set::FxHashSet;
use rustc_hir as hir;
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::InferCtxt;
use rustc_middle::ty::Ty;

pub(crate) trait OutlivesEnvironmentExt<'tcx> {
    fn add_implied_bounds(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
        fn_sig_tys: FxHashSet<Ty<'tcx>>,
        body_id: hir::HirId,
    );
}

impl<'tcx> OutlivesEnvironmentExt<'tcx> for OutlivesEnvironment<'tcx> {
    /// This method adds "implied bounds" into the outlives environment.
    /// Implied bounds are outlives relationships that we can deduce
    /// on the basis that certain types must be well-formed -- these are
    /// either the types that appear in the function signature or else
    /// the input types to an impl. For example, if you have a function
    /// like
    ///
    /// ```
    /// fn foo<'a, 'b, T>(x: &'a &'b [T]) { }
    /// ```
    ///
    /// we can assume in the caller's body that `'b: 'a` and that `T:
    /// 'b` (and hence, transitively, that `T: 'a`). This method would
    /// add those assumptions into the outlives-environment.
    ///
    /// Tests: `src/test/ui/regions/regions-free-region-ordering-*.rs`
    #[instrument(level = "debug", skip(self, infcx))]
    fn add_implied_bounds<'a>(
        &mut self,
        infcx: &InferCtxt<'a, 'tcx>,
        fn_sig_tys: FxHashSet<Ty<'tcx>>,
        body_id: hir::HirId,
    ) {
        for ty in fn_sig_tys {
            let ty = infcx.resolve_vars_if_possible(ty);
            let implied_bounds = infcx.implied_outlives_bounds(self.param_env, body_id, ty);
            self.add_outlives_bounds(Some(infcx), implied_bounds)
        }
    }
}
