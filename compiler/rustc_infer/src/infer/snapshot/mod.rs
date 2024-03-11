use super::region_constraints::RegionSnapshot;
use super::InferCtxt;
use rustc_data_structures::undo_log::UndoLogs;
use rustc_middle::ty;

pub mod check_leaks;
mod fudge;
pub(crate) mod undo_log;

use undo_log::{Snapshot, UndoLog};

#[must_use = "once you start a snapshot, you should always consume it"]
pub struct CombinedSnapshot<'tcx> {
    pub(super) undo_snapshot: Snapshot<'tcx>,
    region_constraints_snapshot: RegionSnapshot,
    universe: ty::UniverseIndex,
}

impl<'tcx> InferCtxt<'tcx> {
    pub fn in_snapshot(&self) -> bool {
        UndoLogs::<UndoLog<'tcx>>::in_snapshot(&self.inner.borrow_mut().undo_log)
    }

    pub fn num_open_snapshots(&self) -> usize {
        UndoLogs::<UndoLog<'tcx>>::num_open_snapshots(&self.inner.borrow_mut().undo_log)
    }

    fn start_snapshot(&self) -> CombinedSnapshot<'tcx> {
        debug!("start_snapshot()");

        let mut inner = self.inner.borrow_mut();

        CombinedSnapshot {
            undo_snapshot: inner.undo_log.start_snapshot(),
            region_constraints_snapshot: inner.unwrap_region_constraints().start_snapshot(),
            universe: self.universe(),
        }
    }

    #[instrument(skip(self, snapshot), level = "debug")]
    fn rollback_to(&self, snapshot: CombinedSnapshot<'tcx>) {
        let CombinedSnapshot { undo_snapshot, region_constraints_snapshot, universe } = snapshot;

        self.universe.set(universe);

        let mut inner = self.inner.borrow_mut();
        inner.rollback_to(undo_snapshot);
        inner.unwrap_region_constraints().rollback_to(region_constraints_snapshot);
    }

    #[instrument(skip(self, snapshot), level = "debug")]
    fn commit_from(&self, snapshot: CombinedSnapshot<'tcx>) {
        let CombinedSnapshot { undo_snapshot, region_constraints_snapshot: _, universe: _ } =
            snapshot;

        self.inner.borrow_mut().commit(undo_snapshot);
    }

    /// Execute `f` and commit the bindings if closure `f` returns `Ok(_)`.
    #[instrument(skip(self, f), level = "debug")]
    pub fn commit_if_ok<T, E, F>(&self, f: F) -> Result<T, E>
    where
        F: FnOnce(&CombinedSnapshot<'tcx>) -> Result<T, E>,
        E: NoSnapshotLeaks<'tcx>,
    {
        let no_leaks_data = E::snapshot_start_data(self);
        let snapshot = self.start_snapshot();
        let r = f(&snapshot);
        debug!("commit_if_ok() -- r.is_ok() = {}", r.is_ok());
        match r {
            Ok(value) => {
                self.commit_from(snapshot);
                Ok(value)
            }
            Err(err) => {
                let no_leaks_data = E::end_of_snapshot(self, err, no_leaks_data);
                self.rollback_to(snapshot);
                Err(E::avoid_leaks(self, no_leaks_data))
            }
        }
    }

    /// Execute `f` then unroll any bindings it creates.
    #[instrument(skip(self, f), level = "debug")]
    pub fn probe<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&CombinedSnapshot<'tcx>) -> R,
        R: NoSnapshotLeaks<'tcx>,
    {
        let no_leaks_data = R::snapshot_start_data(self);
        let snapshot = self.start_snapshot();
        let r = f(&snapshot);
        let no_leaks_data = R::end_of_snapshot(self, r, no_leaks_data);
        self.rollback_to(snapshot);
        R::avoid_leaks(self, no_leaks_data)
    }

    pub fn probe_unchecked<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&CombinedSnapshot<'tcx>) -> R,
    {
        let snapshot = self.start_snapshot();
        let r = f(&snapshot);
        self.rollback_to(snapshot);
        r
    }

    /// Scan the constraints produced since `snapshot` and check whether
    /// we added any region constraints.
    pub fn region_constraints_added_in_snapshot(&self, snapshot: &CombinedSnapshot<'tcx>) -> bool {
        self.inner
            .borrow_mut()
            .unwrap_region_constraints()
            .region_constraints_added_in_snapshot(&snapshot.undo_snapshot)
    }

    pub fn opaque_types_added_in_snapshot(&self, snapshot: &CombinedSnapshot<'tcx>) -> bool {
        self.inner.borrow().undo_log.opaque_types_in_snapshot(&snapshot.undo_snapshot)
    }

    pub fn variable_lengths(&self) -> VariableLengths {
        let mut inner = self.inner.borrow_mut();
        VariableLengths {
            type_vars: inner.type_variables().num_vars(),
            const_vars: inner.const_unification_table().len(),
            int_vars: inner.int_unification_table().len(),
            float_vars: inner.float_unification_table().len(),
            region_vars: inner.unwrap_region_constraints().num_region_vars(),
        }
    }
}

pub struct VariableLengths {
    region_vars: usize,
    type_vars: usize,
    int_vars: usize,
    float_vars: usize,
    const_vars: usize,
}

/// When rolling back a snapshot, we discard all inference constraints
/// added during that snapshot. We also completely remove any inference
/// variables created during the snapshot. Leaking these inference
/// variables from the snapshot and later using them can then result
/// either in an ICE or even accidentally reuse a newly created, totally
/// separate, inference variable.
///
/// To avoid this we make sure that when rolling back snapshots in
///  `fn probe` and `fn commit_if_ok` we do not return any inference
/// variables created during this snapshot.
///
/// This has a fairly involved setup as we previously did not check this
/// and now rely on leaking inference variables, e.g. via `TypeError`.
/// To still avoid ICE we now "fudge inference" in these cases, replacing
/// any newly created inference variables from inside the snapshot with
/// new inference variables created outside of it.
pub trait NoSnapshotLeaks<'tcx> {
    type StartData;
    type EndData;
    fn snapshot_start_data(infcx: &InferCtxt<'tcx>) -> Self::StartData;
    fn end_of_snapshot(
        infcx: &InferCtxt<'tcx>,
        this: Self,
        start: Self::StartData,
    ) -> Self::EndData;
    fn avoid_leaks(infcx: &InferCtxt<'tcx>, data: Self::EndData) -> Self;
}

/// A trait implemented by types which cannot contain any inference variables
/// which could be leaked. The [`NoSnapshotLeaks`] impl for these types is
/// trivial.
///
/// You can mostly think of this as if it is an auto-trait with negative
/// impls for `Region`, `Ty` and `Const` and a positive impl for `Canonical`.
/// Actually using an auto-trait instead of manually implementing it for
/// all types of interest results in overlaps during coherence. Not using
/// auto-traits will also make it easier to share this code with Rust Analyzer
/// in the future, as they want to avoid any unstable features.
pub trait TrivialNoSnapshotLeaks<'tcx> {}
impl<'tcx, T: TrivialNoSnapshotLeaks<'tcx>> NoSnapshotLeaks<'tcx> for T {
    type StartData = ();
    type EndData = T;
    #[inline]
    fn snapshot_start_data(_: &InferCtxt<'tcx>) {}
    #[inline]
    fn end_of_snapshot(_: &InferCtxt<'tcx>, this: Self, _: ()) -> T {
        this
    }
    #[inline]
    fn avoid_leaks(_: &InferCtxt<'tcx>, this: T) -> Self {
        this
    }
}

#[macro_export]
macro_rules! trivial_no_snapshot_leaks {
    ($tcx:lifetime, $t:ty) => {
        impl<$tcx> $crate::infer::snapshot::TrivialNoSnapshotLeaks<$tcx> for $t {}
    };
}

mod impls {
    use super::{NoSnapshotLeaks, TrivialNoSnapshotLeaks};
    use crate::fudge_vars_no_snapshot_leaks;
    use crate::infer::InferCtxt;
    use crate::traits::solve::{CanonicalResponse, Certainty};
    use crate::traits::MismatchedProjectionTypes;
    use crate::type_foldable_verify_no_snapshot_leaks;
    use rustc_hir::def_id::DefId;
    use rustc_middle::infer::canonical::Canonical;
    use rustc_middle::traits::query::{MethodAutoderefStepsResult, NoSolution};
    use rustc_middle::traits::{BuiltinImplSource, EvaluationResult, OverflowError};
    use rustc_middle::ty;
    use rustc_middle::ty::error::TypeError;
    use rustc_span::symbol::Ident;
    use rustc_span::ErrorGuaranteed;

    trivial_no_snapshot_leaks!('tcx, ());
    trivial_no_snapshot_leaks!('tcx, bool);
    trivial_no_snapshot_leaks!('tcx, usize);
    trivial_no_snapshot_leaks!('tcx, ty::AssocItem);
    trivial_no_snapshot_leaks!('tcx, BuiltinImplSource);
    trivial_no_snapshot_leaks!('tcx, DefId);
    trivial_no_snapshot_leaks!('tcx, ErrorGuaranteed);
    trivial_no_snapshot_leaks!('tcx, EvaluationResult);
    trivial_no_snapshot_leaks!('tcx, Ident);
    trivial_no_snapshot_leaks!('tcx, OverflowError);
    trivial_no_snapshot_leaks!('tcx, NoSolution);
    trivial_no_snapshot_leaks!('tcx, Vec<(CanonicalResponse<'tcx>, BuiltinImplSource)>);
    trivial_no_snapshot_leaks!('tcx, (bool, Certainty));
    // FIXME(#122188): This is wrong, this can leak inference vars in `opt_bad_ty` and `steps`.
    trivial_no_snapshot_leaks!('tcx, MethodAutoderefStepsResult<'tcx>);
    type_foldable_verify_no_snapshot_leaks!('tcx, ty::PolyFnSig<'tcx>);
    fudge_vars_no_snapshot_leaks!('tcx, TypeError<'tcx>);
    fudge_vars_no_snapshot_leaks!('tcx, MismatchedProjectionTypes<'tcx>);

    impl<'tcx, T: NoSnapshotLeaks<'tcx>> NoSnapshotLeaks<'tcx> for Option<T> {
        type StartData = T::StartData;
        type EndData = Option<T::EndData>;
        #[inline]
        fn snapshot_start_data(infcx: &InferCtxt<'tcx>) -> T::StartData {
            T::snapshot_start_data(infcx)
        }
        #[inline]
        fn end_of_snapshot(
            infcx: &InferCtxt<'tcx>,
            this: Option<T>,
            start_data: T::StartData,
        ) -> Option<T::EndData> {
            this.map(|this| T::end_of_snapshot(infcx, this, start_data))
        }
        #[inline]
        fn avoid_leaks(infcx: &InferCtxt<'tcx>, data: Self::EndData) -> Self {
            data.map(|data| T::avoid_leaks(infcx, data))
        }
    }

    impl<'tcx, T, E> NoSnapshotLeaks<'tcx> for Result<T, E>
    where
        T: NoSnapshotLeaks<'tcx>,
        E: NoSnapshotLeaks<'tcx>,
    {
        type StartData = (T::StartData, E::StartData);
        type EndData = Result<T::EndData, E::EndData>;
        #[inline]
        fn snapshot_start_data(infcx: &InferCtxt<'tcx>) -> Self::StartData {
            (T::snapshot_start_data(infcx), E::snapshot_start_data(infcx))
        }
        #[inline]
        fn end_of_snapshot(
            infcx: &InferCtxt<'tcx>,
            this: Self,
            (t, e): Self::StartData,
        ) -> Self::EndData {
            match this {
                Ok(value) => Ok(T::end_of_snapshot(infcx, value, t)),
                Err(err) => Err(E::end_of_snapshot(infcx, err, e)),
            }
        }

        #[inline]
        fn avoid_leaks(infcx: &InferCtxt<'tcx>, data: Self::EndData) -> Self {
            match data {
                Ok(value) => Ok(T::avoid_leaks(infcx, value)),
                Err(err) => Err(E::avoid_leaks(infcx, err)),
            }
        }
    }

    impl<'tcx, T: TrivialNoSnapshotLeaks<'tcx>> TrivialNoSnapshotLeaks<'tcx> for Vec<T> {}
    impl<'tcx, V> TrivialNoSnapshotLeaks<'tcx> for Canonical<'tcx, V> {}
}
