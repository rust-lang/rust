use rustc_data_structures::undo_log::UndoLogs;
use rustc_middle::ty;
use tracing::{debug, instrument};

use super::InferCtxt;
use super::region_constraints::RegionSnapshot;

pub(crate) mod fudge;
pub(crate) mod undo_log;

use undo_log::{Snapshot, UndoLog};

#[must_use = "once you start a snapshot, you should always consume it"]
pub struct CombinedSnapshot<'tcx> {
    pub(super) undo_snapshot: Snapshot<'tcx>,
    region_constraints_snapshot: RegionSnapshot,
    universe: ty::UniverseIndex,
}

struct VariableLengths {
    region_constraints_len: usize,
    type_var_len: usize,
    int_var_len: usize,
    float_var_len: usize,
    const_var_len: usize,
}

impl<'tcx> InferCtxt<'tcx> {
    fn variable_lengths(&self) -> VariableLengths {
        let mut inner = self.inner.borrow_mut();
        VariableLengths {
            region_constraints_len: inner.unwrap_region_constraints().num_region_vars(),
            type_var_len: inner.type_variables().num_vars(),
            int_var_len: inner.int_unification_table().len(),
            float_var_len: inner.float_unification_table().len(),
            const_var_len: inner.const_unification_table().len(),
        }
    }

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
    {
        let snapshot = self.start_snapshot();
        let r = f(&snapshot);
        debug!("commit_if_ok() -- r.is_ok() = {}", r.is_ok());
        match r {
            Ok(_) => {
                self.commit_from(snapshot);
            }
            Err(_) => {
                self.rollback_to(snapshot);
            }
        }
        r
    }

    /// Execute `f` then unroll any bindings it creates.
    #[instrument(skip(self, f), level = "debug")]
    pub fn probe<R, F>(&self, f: F) -> R
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
}
