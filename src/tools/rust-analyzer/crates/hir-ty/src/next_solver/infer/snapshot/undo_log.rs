//! Snapshotting in the infer ctxt of the next-trait-solver.

use ena::snapshot_vec as sv;
use ena::undo_log::{Rollback, UndoLogs};
use ena::unify as ut;
use rustc_type_ir::FloatVid;
use rustc_type_ir::IntVid;
use tracing::debug;

use crate::next_solver::OpaqueTypeKey;
use crate::next_solver::infer::opaque_types::OpaqueHiddenType;
use crate::next_solver::infer::unify_key::ConstVidKey;
use crate::next_solver::infer::unify_key::RegionVidKey;
use crate::next_solver::infer::{InferCtxtInner, region_constraints, type_variable};

pub struct Snapshot {
    pub(crate) undo_len: usize,
}

/// Records the "undo" data for a single operation that affects some form of inference variable.
#[derive(Clone)]
pub(crate) enum UndoLog<'db> {
    DuplicateOpaqueType,
    OpaqueTypes(OpaqueTypeKey<'db>, Option<OpaqueHiddenType<'db>>),
    TypeVariables(type_variable::UndoLog<'db>),
    ConstUnificationTable(sv::UndoLog<ut::Delegate<ConstVidKey<'db>>>),
    IntUnificationTable(sv::UndoLog<ut::Delegate<IntVid>>),
    FloatUnificationTable(sv::UndoLog<ut::Delegate<FloatVid>>),
    RegionConstraintCollector(region_constraints::UndoLog<'db>),
    RegionUnificationTable(sv::UndoLog<ut::Delegate<RegionVidKey<'db>>>),
    #[expect(dead_code, reason = "this is used in rustc")]
    PushRegionObligation,
}

macro_rules! impl_from {
    ($($ctor:ident ($ty:ty),)*) => {
        $(
        impl<'db> From<$ty> for UndoLog<'db> {
            fn from(x: $ty) -> Self {
                UndoLog::$ctor(x.into())
            }
        }
        )*
    }
}

// Upcast from a single kind of "undoable action" to the general enum
impl_from! {
    RegionConstraintCollector(region_constraints::UndoLog<'db>),

    TypeVariables(sv::UndoLog<ut::Delegate<type_variable::TyVidEqKey<'db>>>),
    TypeVariables(sv::UndoLog<ut::Delegate<type_variable::TyVidSubKey>>),
    TypeVariables(type_variable::UndoLog<'db>),
    IntUnificationTable(sv::UndoLog<ut::Delegate<IntVid>>),
    FloatUnificationTable(sv::UndoLog<ut::Delegate<FloatVid>>),

    ConstUnificationTable(sv::UndoLog<ut::Delegate<ConstVidKey<'db>>>),

    RegionUnificationTable(sv::UndoLog<ut::Delegate<RegionVidKey<'db>>>),
}

/// The Rollback trait defines how to rollback a particular action.
impl<'db> Rollback<UndoLog<'db>> for InferCtxtInner<'db> {
    fn reverse(&mut self, undo: UndoLog<'db>) {
        match undo {
            UndoLog::DuplicateOpaqueType => self.opaque_type_storage.pop_duplicate_entry(),
            UndoLog::OpaqueTypes(key, idx) => self.opaque_type_storage.remove(key, idx),
            UndoLog::TypeVariables(undo) => self.type_variable_storage.reverse(undo),
            UndoLog::ConstUnificationTable(undo) => self.const_unification_storage.reverse(undo),
            UndoLog::IntUnificationTable(undo) => self.int_unification_storage.reverse(undo),
            UndoLog::FloatUnificationTable(undo) => self.float_unification_storage.reverse(undo),
            UndoLog::RegionConstraintCollector(undo) => {
                self.region_constraint_storage.as_mut().unwrap().reverse(undo)
            }
            UndoLog::RegionUnificationTable(undo) => {
                self.region_constraint_storage.as_mut().unwrap().unification_table.reverse(undo)
            }
            UndoLog::PushRegionObligation => {
                self.region_obligations.pop();
            }
        }
    }
}

/// The combined undo log for all the various unification tables. For each change to the storage
/// for any kind of inference variable, we record an UndoLog entry in the vector here.
#[derive(Clone, Default)]
pub(crate) struct InferCtxtUndoLogs<'db> {
    logs: Vec<UndoLog<'db>>,
    num_open_snapshots: usize,
}

/// The UndoLogs trait defines how we undo a particular kind of action (of type T). We can undo any
/// action that is convertible into an UndoLog (per the From impls above).
impl<'db, T> UndoLogs<T> for InferCtxtUndoLogs<'db>
where
    UndoLog<'db>: From<T>,
{
    #[inline]
    fn num_open_snapshots(&self) -> usize {
        self.num_open_snapshots
    }

    #[inline]
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

impl<'db> InferCtxtInner<'db> {
    pub fn rollback_to(&mut self, snapshot: Snapshot) {
        debug!("rollback_to({})", snapshot.undo_len);
        self.undo_log.assert_open_snapshot(&snapshot);

        while self.undo_log.logs.len() > snapshot.undo_len {
            let undo = self.undo_log.logs.pop().unwrap();
            self.reverse(undo);
        }

        self.type_variable_storage.finalize_rollback();

        if self.undo_log.num_open_snapshots == 1 {
            // After the root snapshot the undo log should be empty.
            assert!(snapshot.undo_len == 0);
            assert!(self.undo_log.logs.is_empty());
        }

        self.undo_log.num_open_snapshots -= 1;
    }

    pub fn commit(&mut self, snapshot: Snapshot) {
        debug!("commit({})", snapshot.undo_len);

        if self.undo_log.num_open_snapshots == 1 {
            // The root snapshot. It's safe to clear the undo log because
            // there's no snapshot further out that we might need to roll back
            // to.
            assert!(snapshot.undo_len == 0);
            self.undo_log.logs.clear();
        }

        self.undo_log.num_open_snapshots -= 1;
    }
}

impl<'db> InferCtxtUndoLogs<'db> {
    pub(crate) fn start_snapshot(&mut self) -> Snapshot {
        self.num_open_snapshots += 1;
        Snapshot { undo_len: self.logs.len() }
    }

    pub(crate) fn region_constraints_in_snapshot(
        &self,
        s: &Snapshot,
    ) -> impl Iterator<Item = &'_ region_constraints::UndoLog<'db>> + Clone {
        self.logs[s.undo_len..].iter().filter_map(|log| match log {
            UndoLog::RegionConstraintCollector(log) => Some(log),
            _ => None,
        })
    }

    pub(crate) fn opaque_types_in_snapshot(&self, s: &Snapshot) -> bool {
        self.logs[s.undo_len..].iter().any(|log| matches!(log, UndoLog::OpaqueTypes(..)))
    }

    fn assert_open_snapshot(&self, snapshot: &Snapshot) {
        // Failures here may indicate a failure to follow a stack discipline.
        assert!(self.logs.len() >= snapshot.undo_len);
        assert!(self.num_open_snapshots > 0);
    }
}

impl<'db> std::ops::Index<usize> for InferCtxtUndoLogs<'db> {
    type Output = UndoLog<'db>;

    fn index(&self, key: usize) -> &Self::Output {
        &self.logs[key]
    }
}

impl<'db> std::ops::IndexMut<usize> for InferCtxtUndoLogs<'db> {
    fn index_mut(&mut self, key: usize) -> &mut Self::Output {
        &mut self.logs[key]
    }
}
