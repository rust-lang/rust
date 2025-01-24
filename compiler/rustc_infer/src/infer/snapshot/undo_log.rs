use std::marker::PhantomData;

use rustc_data_structures::undo_log::{Rollback, UndoLogs};
use rustc_data_structures::{snapshot_vec as sv, unify as ut};
use rustc_middle::infer::unify_key::{ConstVidKey, RegionVidKey};
use rustc_middle::ty::{self, OpaqueHiddenType, OpaqueTypeKey};
use tracing::debug;

use crate::infer::{InferCtxtInner, region_constraints, type_variable};
use crate::traits;

pub struct Snapshot<'tcx> {
    pub(crate) undo_len: usize,
    _marker: PhantomData<&'tcx ()>,
}

/// Records the "undo" data for a single operation that affects some form of inference variable.
#[derive(Clone)]
pub(crate) enum UndoLog<'tcx> {
    OpaqueTypes(OpaqueTypeKey<'tcx>, Option<OpaqueHiddenType<'tcx>>),
    TypeVariables(sv::UndoLog<ut::Delegate<type_variable::TyVidEqKey<'tcx>>>),
    ConstUnificationTable(sv::UndoLog<ut::Delegate<ConstVidKey<'tcx>>>),
    IntUnificationTable(sv::UndoLog<ut::Delegate<ty::IntVid>>),
    FloatUnificationTable(sv::UndoLog<ut::Delegate<ty::FloatVid>>),
    RegionConstraintCollector(region_constraints::UndoLog<'tcx>),
    RegionUnificationTable(sv::UndoLog<ut::Delegate<RegionVidKey<'tcx>>>),
    ProjectionCache(traits::UndoLog<'tcx>),
    PushRegionObligation,
}

macro_rules! impl_from {
    ($($ctor:ident ($ty:ty),)*) => {
        $(
        impl<'tcx> From<$ty> for UndoLog<'tcx> {
            fn from(x: $ty) -> Self {
                UndoLog::$ctor(x.into())
            }
        }
        )*
    }
}

// Upcast from a single kind of "undoable action" to the general enum
impl_from! {
    RegionConstraintCollector(region_constraints::UndoLog<'tcx>),

    TypeVariables(sv::UndoLog<ut::Delegate<type_variable::TyVidEqKey<'tcx>>>),
    IntUnificationTable(sv::UndoLog<ut::Delegate<ty::IntVid>>),
    FloatUnificationTable(sv::UndoLog<ut::Delegate<ty::FloatVid>>),

    ConstUnificationTable(sv::UndoLog<ut::Delegate<ConstVidKey<'tcx>>>),

    RegionUnificationTable(sv::UndoLog<ut::Delegate<RegionVidKey<'tcx>>>),
    ProjectionCache(traits::UndoLog<'tcx>),
}

/// The Rollback trait defines how to rollback a particular action.
impl<'tcx> Rollback<UndoLog<'tcx>> for InferCtxtInner<'tcx> {
    fn reverse(&mut self, undo: UndoLog<'tcx>) {
        match undo {
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
            UndoLog::ProjectionCache(undo) => self.projection_cache.reverse(undo),
            UndoLog::PushRegionObligation => {
                self.region_obligations.pop();
            }
        }
    }
}

/// The combined undo log for all the various unification tables. For each change to the storage
/// for any kind of inference variable, we record an UndoLog entry in the vector here.
#[derive(Clone, Default)]
pub(crate) struct InferCtxtUndoLogs<'tcx> {
    logs: Vec<UndoLog<'tcx>>,
    num_open_snapshots: usize,
}

/// The UndoLogs trait defines how we undo a particular kind of action (of type T). We can undo any
/// action that is convertible into an UndoLog (per the From impls above).
impl<'tcx, T> UndoLogs<T> for InferCtxtUndoLogs<'tcx>
where
    UndoLog<'tcx>: From<T>,
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

impl<'tcx> InferCtxtInner<'tcx> {
    pub fn rollback_to(&mut self, snapshot: Snapshot<'tcx>) {
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

    pub fn commit(&mut self, snapshot: Snapshot<'tcx>) {
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

impl<'tcx> InferCtxtUndoLogs<'tcx> {
    pub(crate) fn start_snapshot(&mut self) -> Snapshot<'tcx> {
        self.num_open_snapshots += 1;
        Snapshot { undo_len: self.logs.len(), _marker: PhantomData }
    }

    pub(crate) fn region_constraints_in_snapshot(
        &self,
        s: &Snapshot<'tcx>,
    ) -> impl Iterator<Item = &'_ region_constraints::UndoLog<'tcx>> + Clone {
        self.logs[s.undo_len..].iter().filter_map(|log| match log {
            UndoLog::RegionConstraintCollector(log) => Some(log),
            _ => None,
        })
    }

    pub(crate) fn opaque_types_in_snapshot(&self, s: &Snapshot<'tcx>) -> bool {
        self.logs[s.undo_len..].iter().any(|log| matches!(log, UndoLog::OpaqueTypes(..)))
    }

    fn assert_open_snapshot(&self, snapshot: &Snapshot<'tcx>) {
        // Failures here may indicate a failure to follow a stack discipline.
        assert!(self.logs.len() >= snapshot.undo_len);
        assert!(self.num_open_snapshots > 0);
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
