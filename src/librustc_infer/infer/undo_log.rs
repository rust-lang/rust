use std::marker::PhantomData;

use rustc::ty;
use rustc_data_structures::snapshot_vec as sv;
use rustc_data_structures::undo_log::{Rollback, Snapshots, UndoLogs};
use rustc_data_structures::unify as ut;
use rustc_hir as hir;

use crate::{
    infer::{
        region_constraints::{self, RegionConstraintStorage},
        type_variable, RegionObligation,
    },
    traits,
};

pub struct Snapshot<'tcx> {
    pub(crate) undo_len: usize,
    _marker: PhantomData<&'tcx ()>,
}

pub(crate) enum UndoLog<'tcx> {
    TypeVariables(type_variable::UndoLog<'tcx>),
    ConstUnificationTable(sv::UndoLog<ut::Delegate<ty::ConstVid<'tcx>>>),
    IntUnificationTable(sv::UndoLog<ut::Delegate<ty::IntVid>>),
    FloatUnificationTable(sv::UndoLog<ut::Delegate<ty::FloatVid>>),
    RegionConstraintCollector(region_constraints::UndoLog<'tcx>),
    RegionUnificationTable(sv::UndoLog<ut::Delegate<ty::RegionVid>>),
    ProjectionCache(traits::UndoLog<'tcx>),
    PushRegionObligation,
}

impl<'tcx> From<region_constraints::UndoLog<'tcx>> for UndoLog<'tcx> {
    fn from(l: region_constraints::UndoLog<'tcx>) -> Self {
        UndoLog::RegionConstraintCollector(l)
    }
}

impl<'tcx> From<sv::UndoLog<ut::Delegate<type_variable::TyVidEqKey<'tcx>>>> for UndoLog<'tcx> {
    fn from(l: sv::UndoLog<ut::Delegate<type_variable::TyVidEqKey<'tcx>>>) -> Self {
        UndoLog::TypeVariables(type_variable::UndoLog::EqRelation(l))
    }
}

impl<'tcx> From<sv::UndoLog<ut::Delegate<ty::TyVid>>> for UndoLog<'tcx> {
    fn from(l: sv::UndoLog<ut::Delegate<ty::TyVid>>) -> Self {
        UndoLog::TypeVariables(type_variable::UndoLog::SubRelation(l))
    }
}

impl<'tcx> From<sv::UndoLog<type_variable::Delegate>> for UndoLog<'tcx> {
    fn from(l: sv::UndoLog<type_variable::Delegate>) -> Self {
        UndoLog::TypeVariables(type_variable::UndoLog::Values(l))
    }
}

impl<'tcx> From<type_variable::Instantiate> for UndoLog<'tcx> {
    fn from(l: type_variable::Instantiate) -> Self {
        UndoLog::TypeVariables(type_variable::UndoLog::from(l))
    }
}

impl From<type_variable::UndoLog<'tcx>> for UndoLog<'tcx> {
    fn from(t: type_variable::UndoLog<'tcx>) -> Self {
        Self::TypeVariables(t)
    }
}

impl<'tcx> From<sv::UndoLog<ut::Delegate<ty::ConstVid<'tcx>>>> for UndoLog<'tcx> {
    fn from(l: sv::UndoLog<ut::Delegate<ty::ConstVid<'tcx>>>) -> Self {
        Self::ConstUnificationTable(l)
    }
}

impl<'tcx> From<sv::UndoLog<ut::Delegate<ty::IntVid>>> for UndoLog<'tcx> {
    fn from(l: sv::UndoLog<ut::Delegate<ty::IntVid>>) -> Self {
        Self::IntUnificationTable(l)
    }
}

impl<'tcx> From<sv::UndoLog<ut::Delegate<ty::FloatVid>>> for UndoLog<'tcx> {
    fn from(l: sv::UndoLog<ut::Delegate<ty::FloatVid>>) -> Self {
        Self::FloatUnificationTable(l)
    }
}

impl<'tcx> From<sv::UndoLog<ut::Delegate<ty::RegionVid>>> for UndoLog<'tcx> {
    fn from(l: sv::UndoLog<ut::Delegate<ty::RegionVid>>) -> Self {
        Self::RegionUnificationTable(l)
    }
}

impl<'tcx> From<traits::UndoLog<'tcx>> for UndoLog<'tcx> {
    fn from(l: traits::UndoLog<'tcx>) -> Self {
        Self::ProjectionCache(l)
    }
}

pub(super) struct RollbackView<'tcx, 'a> {
    pub(super) type_variables: &'a mut type_variable::TypeVariableStorage<'tcx>,
    pub(super) const_unification_table: &'a mut ut::UnificationStorage<ty::ConstVid<'tcx>>,
    pub(super) int_unification_table: &'a mut ut::UnificationStorage<ty::IntVid>,
    pub(super) float_unification_table: &'a mut ut::UnificationStorage<ty::FloatVid>,
    pub(super) region_constraints: &'a mut RegionConstraintStorage<'tcx>,
    pub(super) projection_cache: &'a mut traits::ProjectionCacheStorage<'tcx>,
    pub(super) region_obligations: &'a mut Vec<(hir::HirId, RegionObligation<'tcx>)>,
}

impl<'tcx> Rollback<UndoLog<'tcx>> for RollbackView<'tcx, '_> {
    fn reverse(&mut self, undo: UndoLog<'tcx>) {
        match undo {
            UndoLog::TypeVariables(undo) => self.type_variables.reverse(undo),
            UndoLog::ConstUnificationTable(undo) => self.const_unification_table.reverse(undo),
            UndoLog::IntUnificationTable(undo) => self.int_unification_table.reverse(undo),
            UndoLog::FloatUnificationTable(undo) => self.float_unification_table.reverse(undo),
            UndoLog::RegionConstraintCollector(undo) => self.region_constraints.reverse(undo),
            UndoLog::RegionUnificationTable(undo) => {
                self.region_constraints.unification_table.reverse(undo)
            }
            UndoLog::ProjectionCache(undo) => self.projection_cache.reverse(undo),
            UndoLog::PushRegionObligation => {
                self.region_obligations.pop();
            }
        }
    }
}

pub(crate) struct InferCtxtUndoLogs<'tcx> {
    logs: Vec<UndoLog<'tcx>>,
    num_open_snapshots: usize,
}

impl Default for InferCtxtUndoLogs<'_> {
    fn default() -> Self {
        Self { logs: Default::default(), num_open_snapshots: Default::default() }
    }
}

impl<'tcx, T> UndoLogs<T> for InferCtxtUndoLogs<'tcx>
where
    UndoLog<'tcx>: From<T>,
{
    fn num_open_snapshots(&self) -> usize {
        self.num_open_snapshots
    }
    fn push(&mut self, undo: T) {
        if self.in_snapshot() {
            self.logs.push(undo.into())
        }
    }
    fn clear(&mut self) {
        self.logs.clear();
        self.num_open_snapshots = 0;
    }
    fn extend<J>(&mut self, undos: J)
    where
        Self: Sized,
        J: IntoIterator<Item = T>,
    {
        if self.in_snapshot() {
            self.logs.extend(undos.into_iter().map(UndoLog::from))
        }
    }
}

impl<'tcx> Snapshots<UndoLog<'tcx>> for InferCtxtUndoLogs<'tcx> {
    type Snapshot = Snapshot<'tcx>;
    fn actions_since_snapshot(&self, snapshot: &Self::Snapshot) -> &[UndoLog<'tcx>] {
        &self.logs[snapshot.undo_len..]
    }

    fn start_snapshot(&mut self) -> Self::Snapshot {
        self.num_open_snapshots += 1;
        Snapshot { undo_len: self.logs.len(), _marker: PhantomData }
    }

    fn rollback_to<R>(&mut self, values: impl FnOnce() -> R, snapshot: Self::Snapshot)
    where
        R: Rollback<UndoLog<'tcx>>,
    {
        debug!("rollback_to({})", snapshot.undo_len);
        self.assert_open_snapshot(&snapshot);

        if self.logs.len() > snapshot.undo_len {
            let mut values = values();
            while self.logs.len() > snapshot.undo_len {
                values.reverse(self.logs.pop().unwrap());
            }
        }

        if self.num_open_snapshots == 1 {
            // The root snapshot. It's safe to clear the undo log because
            // there's no snapshot further out that we might need to roll back
            // to.
            assert!(snapshot.undo_len == 0);
            self.logs.clear();
        }

        self.num_open_snapshots -= 1;
    }

    fn commit(&mut self, snapshot: Self::Snapshot) {
        debug!("commit({})", snapshot.undo_len);

        if self.num_open_snapshots == 1 {
            // The root snapshot. It's safe to clear the undo log because
            // there's no snapshot further out that we might need to roll back
            // to.
            assert!(snapshot.undo_len == 0);
            self.logs.clear();
        }

        self.num_open_snapshots -= 1;
    }
}

impl<'tcx> InferCtxtUndoLogs<'tcx> {
    pub(crate) fn region_constraints_in_snapshot(
        &self,
        s: &Snapshot<'tcx>,
    ) -> impl Iterator<Item = &'_ region_constraints::UndoLog<'tcx>> + Clone {
        self.logs[s.undo_len..].iter().filter_map(|log| match log {
            UndoLog::RegionConstraintCollector(log) => Some(log),
            _ => None,
        })
    }

    pub(crate) fn region_constraints(
        &self,
    ) -> impl Iterator<Item = &'_ region_constraints::UndoLog<'tcx>> + Clone {
        self.logs.iter().filter_map(|log| match log {
            UndoLog::RegionConstraintCollector(log) => Some(log),
            _ => None,
        })
    }

    fn assert_open_snapshot(&self, snapshot: &Snapshot<'tcx>) {
        // Failures here may indicate a failure to follow a stack discipline.
        assert!(self.logs.len() >= snapshot.undo_len);
        assert!(self.num_open_snapshots > 0);
    }

    pub(crate) fn iter(&self) -> std::slice::Iter<'_, UndoLog<'tcx>> {
        self.logs.iter()
    }
}

impl<'tcx> std::ops::Index<usize> for InferCtxtUndoLogs<'tcx> {
    type Output = UndoLog<'tcx>;
    fn index(&self, key: usize) -> &Self::Output {
        &self.logs[key]
    }
}

impl<'tcx> std::ops::IndexMut<usize> for InferCtxtUndoLogs<'tcx> {
    fn index_mut(&mut self, key: usize) -> &mut Self::Output {
        &mut self.logs[key]
    }
}
