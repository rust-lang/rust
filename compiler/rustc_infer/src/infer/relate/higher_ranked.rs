//! Helper routines for higher-ranked things. See the `doc` module at
//! the end of the file for details.

use rustc_middle::ty::{self, FnMutDelegate, Ty, TyCtxt, TypeFoldable, TypeVisitableExt};
use tracing::{debug, instrument};

use super::RelateResult;
use crate::infer::InferCtxt;
use crate::infer::snapshot::CombinedSnapshot;

impl<'tcx> InferCtxt<'tcx> {
    /// Replaces all bound variables (lifetimes, types, and constants) bound by
    /// `binder` with placeholder variables in a new universe. This means that the
    /// new placeholders can only be named by inference variables created after
    /// this method has been called.
    ///
    /// This is the first step of checking subtyping when higher-ranked things are involved.
    /// For more details visit the relevant sections of the [rustc dev guide].
    ///
    /// `fn enter_forall` should be preferred over this method.
    ///
    /// [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/traits/hrtb.html
    #[instrument(level = "debug", skip(self), ret)]
    pub fn enter_forall_and_leak_universe<T>(&self, binder: ty::Binder<'tcx, T>) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        // Inlined `no_bound_vars`.
        if !binder.as_ref().skip_binder().has_escaping_bound_vars() {
            return binder.skip_binder();
        }

        let next_universe = self.create_next_universe();

        let delegate = FnMutDelegate {
            regions: &mut |br: ty::BoundRegion| {
                ty::Region::new_placeholder(
                    self.tcx,
                    ty::PlaceholderRegion { universe: next_universe, bound: br },
                )
            },
            types: &mut |bound_ty: ty::BoundTy| {
                Ty::new_placeholder(
                    self.tcx,
                    ty::PlaceholderType { universe: next_universe, bound: bound_ty },
                )
            },
            consts: &mut |bound_const: ty::BoundConst| {
                ty::Const::new_placeholder(
                    self.tcx,
                    ty::PlaceholderConst { universe: next_universe, bound: bound_const },
                )
            },
        };

        debug!(?next_universe);
        self.tcx.replace_bound_vars_uncached(binder, delegate)
    }

    /// Replaces all bound variables (lifetimes, types, and constants) bound by
    /// `binder` with placeholder variables in a new universe and then calls the
    /// closure `f` with the instantiated value. The new placeholders can only be
    /// named by inference variables created inside of the closure `f` or afterwards.
    ///
    /// This is the first step of checking subtyping when higher-ranked things are involved.
    /// For more details visit the relevant sections of the [rustc dev guide].
    ///
    /// This method should be preferred over `fn enter_forall_and_leak_universe`.
    ///
    /// [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/traits/hrtb.html
    #[instrument(level = "debug", skip(self, f))]
    pub fn enter_forall<T, U>(&self, forall: ty::Binder<'tcx, T>, f: impl FnOnce(T) -> U) -> U
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        // FIXME: currently we do nothing to prevent placeholders with the new universe being
        // used after exiting `f`. For example region subtyping can result in outlives constraints
        // that name placeholders created in this function. Nested goals from type relations can
        // also contain placeholders created by this function.
        let value = self.enter_forall_and_leak_universe(forall);
        debug!(?value);
        f(value)
    }

    /// See [RegionConstraintCollector::leak_check][1]. We only check placeholder
    /// leaking into `outer_universe`, i.e. placeholders which cannot be named by that
    /// universe.
    ///
    /// [1]: crate::infer::region_constraints::RegionConstraintCollector::leak_check
    pub fn leak_check(
        &self,
        outer_universe: ty::UniverseIndex,
        only_consider_snapshot: Option<&CombinedSnapshot<'tcx>>,
    ) -> RelateResult<'tcx, ()> {
        // If the user gave `-Zno-leak-check`, or we have been
        // configured to skip the leak check, then skip the leak check
        // completely. The leak check is deprecated. Any legitimate
        // subtyping errors that it would have caught will now be
        // caught later on, during region checking. However, we
        // continue to use it for a transition period.
        if self.tcx.sess.opts.unstable_opts.no_leak_check || self.skip_leak_check {
            return Ok(());
        }

        self.inner.borrow_mut().unwrap_region_constraints().leak_check(
            self.tcx,
            outer_universe,
            self.universe(),
            only_consider_snapshot,
        )
    }
}
