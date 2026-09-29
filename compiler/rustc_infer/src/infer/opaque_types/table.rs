use std::iter;
use std::ops::Deref;

use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::indexmap::map::Entry;
use rustc_data_structures::undo_log::UndoLogs;
use rustc_middle::ty::{self as ty, OpaqueTypeKey, ProvisionalHiddenType, Ty};
use rustc_span::bug;
use tracing::instrument;

use crate::infer::snapshot::undo_log::{InferCtxtUndoLogs, UndoLog};

#[derive(Default, Debug, Clone)]
pub struct OpaqueTypeStorage<'tcx> {
    opaque_types: FxIndexMap<OpaqueTypeKey<'tcx>, ProvisionalHiddenType<'tcx>>,
    duplicate_entries: Vec<(OpaqueTypeKey<'tcx>, ProvisionalHiddenType<'tcx>)>,
    /// We define and register unresolved infer vars as *pseudo-rigid* types and there bounds
    /// in the following, recursive manner:
    ///
    /// - Opaque types are pseudo-rigids in their defining scopes. We register their item-self
    ///   bounds along with them, e.g., if we have `impl Iterator<Item = i32>`, the bounds are
    ///   `?pseudo-rigid: Iterator` and `<?pseudo-rigid as Iterator>::Item = i32`.
    /// - When we assemble candidates for a goal whose self-ty is pseudo-rigid, we match those
    ///   bounds for that pseudo-rigid with the goal.
    /// - From the above case, when we normalize an associated type whose self-ty is pseudo-rigid,
    ///   and the relevant candidate is one of those pseudo-rigid bounds,  we register that
    ///   projection term as a new pseudo-rigid, along with the self-bounds for that associated
    ///   type.
    ///
    /// We consider those registered unresolved infer vars as pseudo-rigid and allow them to be
    /// used in some of the non-defining usages such as being a self-type in a method call.
    pseudo_rigids_due_to_opaques:
        FxIndexMap<Ty<'tcx>, FxIndexSet<ty::PseudoRigidDueToOpaquesBound<'tcx>>>,
    /// The flattened version of the above `pseudo_rigids_due_to_opaques`. This is a pure duplication
    /// but we need this to track things linearly, so that we can track the number of those bounds
    /// in [`OpaqueTypeStorageEntries`] without a map and can lookup `pseudo_rigid_due_to_opaques_bounds`
    /// in O(1).
    pseudo_rigid_due_to_opaques_bounds: Vec<(Ty<'tcx>, ty::PseudoRigidDueToOpaquesBound<'tcx>)>,
}

/// The number of entries in the opaque type storage at a given point.
///
/// Used to check that we haven't added any new opaque types after checking
/// the opaque types currently in the storage.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpaqueTypeStorageEntries {
    opaque_types: usize,
    duplicate_entries: usize,
    pseudo_rigid_due_to_opaques_bounds: usize,
}

impl rustc_type_ir::inherent::OpaqueTypeStorageEntries for OpaqueTypeStorageEntries {
    fn needs_reevaluation(self, opaques: usize, pseudo_rigid_bounds: usize) -> bool {
        let OpaqueTypeStorageEntries {
            opaque_types,
            duplicate_entries: _,
            pseudo_rigid_due_to_opaques_bounds,
        } = self;
        opaques != opaque_types || pseudo_rigid_bounds != pseudo_rigid_due_to_opaques_bounds
    }
}

impl<'tcx> OpaqueTypeStorage<'tcx> {
    #[instrument(level = "debug")]
    pub(crate) fn remove(
        &mut self,
        key: OpaqueTypeKey<'tcx>,
        prev: Option<ProvisionalHiddenType<'tcx>>,
    ) {
        if let Some(prev) = prev {
            *self.opaque_types.get_mut(&key).unwrap() = prev;
        } else {
            match self.opaque_types.swap_remove(&key) {
                None => bug!("reverted opaque type inference that was never registered: {:?}", key),
                Some(_) => {}
            }
        }
    }

    pub(crate) fn pop_duplicate_entry(&mut self) {
        let entry = self.duplicate_entries.pop();
        assert!(entry.is_some());
    }

    pub(crate) fn undo_pseudo_rigid_due_to_opaques(
        &mut self,
        pseudo_rigid: Ty<'tcx>,
        len: Option<usize>,
    ) {
        let removed = if let Some(len) = len {
            let bounds = self.pseudo_rigids_due_to_opaques.get_mut(&pseudo_rigid).unwrap();
            let removed = bounds.len() - len;
            bounds.truncate(len);
            removed
        } else {
            match self.pseudo_rigids_due_to_opaques.swap_remove(&pseudo_rigid) {
                None => bug!(
                    "reverted pseudo-rigid type inference that was never registered: {:?}",
                    pseudo_rigid
                ),
                Some(bounds) => bounds.len(),
            }
        };

        let truncate_to = self.pseudo_rigid_due_to_opaques_bounds.len() - removed;
        debug_assert!(
            (&self.pseudo_rigid_due_to_opaques_bounds[truncate_to..])
                .iter()
                .all(|(pr, _)| *pr == pseudo_rigid)
        );
        self.pseudo_rigid_due_to_opaques_bounds.truncate(truncate_to);
    }

    pub fn is_empty(&self) -> bool {
        let OpaqueTypeStorage {
            opaque_types,
            duplicate_entries,
            pseudo_rigids_due_to_opaques,
            pseudo_rigid_due_to_opaques_bounds,
        } = self;
        opaque_types.is_empty()
            && duplicate_entries.is_empty()
            && pseudo_rigids_due_to_opaques.is_empty()
            && pseudo_rigid_due_to_opaques_bounds.is_empty()
    }

    pub(crate) fn take_opaque_types(
        &mut self,
    ) -> (
        impl Iterator<Item = (OpaqueTypeKey<'tcx>, ProvisionalHiddenType<'tcx>)>,
        impl Iterator<Item = (Ty<'tcx>, FxIndexSet<ty::PseudoRigidDueToOpaquesBound<'tcx>>)>,
    ) {
        let OpaqueTypeStorage {
            opaque_types,
            duplicate_entries,
            pseudo_rigids_due_to_opaques,
            pseudo_rigid_due_to_opaques_bounds,
        } = self;
        let _ = std::mem::take(pseudo_rigid_due_to_opaques_bounds);
        (
            std::mem::take(opaque_types).into_iter().chain(std::mem::take(duplicate_entries)),
            std::mem::take(pseudo_rigids_due_to_opaques).into_iter(),
        )
    }

    pub fn num_entries(&self) -> OpaqueTypeStorageEntries {
        OpaqueTypeStorageEntries {
            opaque_types: self.opaque_types.len(),
            duplicate_entries: self.duplicate_entries.len(),
            pseudo_rigid_due_to_opaques_bounds: self.pseudo_rigid_due_to_opaques_bounds.len(),
        }
    }

    pub fn num_pseudo_rigid_due_to_opaques_bounds(&self) -> usize {
        self.pseudo_rigid_due_to_opaques_bounds.len()
    }

    pub fn opaque_types_added_since(
        &self,
        prev_entries: OpaqueTypeStorageEntries,
    ) -> impl Iterator<Item = (OpaqueTypeKey<'tcx>, ProvisionalHiddenType<'tcx>)> {
        self.opaque_types
            .iter()
            .skip(prev_entries.opaque_types)
            .map(|(k, v)| (*k, *v))
            .chain(self.duplicate_entries.iter().skip(prev_entries.duplicate_entries).copied())
    }

    pub fn pseudo_rigid_due_to_opaques_bounds_added_since(
        &self,
        prev_entries: OpaqueTypeStorageEntries,
    ) -> impl Iterator<Item = (Ty<'tcx>, ty::PseudoRigidDueToOpaquesBound<'tcx>)> {
        self.pseudo_rigid_due_to_opaques_bounds
            .iter()
            .skip(prev_entries.pseudo_rigid_due_to_opaques_bounds)
            .copied()
    }

    /// Only returns the opaque types from the lookup table. These are used
    /// when normalizing opaque types and have a unique key.
    ///
    /// Outside of canonicalization one should generally use `iter_opaque_types`
    /// to also consider duplicate entries.
    pub fn iter_lookup_table(
        &self,
    ) -> impl Iterator<Item = (OpaqueTypeKey<'tcx>, ProvisionalHiddenType<'tcx>)> {
        self.opaque_types.iter().map(|(k, v)| (*k, *v))
    }

    /// Only returns the opaque types which are stored in `duplicate_entries`.
    ///
    /// These have to considered when checking all opaque type uses but are e.g.
    /// irrelevant for canonical inputs as nested queries never meaningfully
    /// accesses them.
    pub fn iter_duplicate_entries(
        &self,
    ) -> impl Iterator<Item = (OpaqueTypeKey<'tcx>, ProvisionalHiddenType<'tcx>)> {
        self.duplicate_entries.iter().copied()
    }

    pub fn iter_opaque_types(
        &self,
    ) -> impl Iterator<Item = (OpaqueTypeKey<'tcx>, ProvisionalHiddenType<'tcx>)> {
        let OpaqueTypeStorage {
            opaque_types,
            duplicate_entries,
            pseudo_rigids_due_to_opaques: _,
            pseudo_rigid_due_to_opaques_bounds: _,
        } = self;
        opaque_types.iter().map(|(k, v)| (*k, *v)).chain(duplicate_entries.iter().copied())
    }

    pub fn iter_pseudo_rigids_due_to_opaques(
        &self,
    ) -> impl Iterator<Item = (Ty<'tcx>, &FxIndexSet<ty::PseudoRigidDueToOpaquesBound<'tcx>>)> {
        let OpaqueTypeStorage {
            opaque_types: _,
            duplicate_entries: _,
            pseudo_rigids_due_to_opaques,
            pseudo_rigid_due_to_opaques_bounds: _,
        } = self;
        pseudo_rigids_due_to_opaques.iter().map(|(pr, bounds)| (*pr, bounds))
    }

    pub fn iter_pseudo_rigid_due_to_opaques_bounds(
        &self,
    ) -> impl Iterator<Item = (Ty<'tcx>, ty::PseudoRigidDueToOpaquesBound<'tcx>)> {
        let OpaqueTypeStorage {
            opaque_types: _,
            duplicate_entries: _,
            pseudo_rigids_due_to_opaques: _,
            pseudo_rigid_due_to_opaques_bounds,
        } = self;
        pseudo_rigid_due_to_opaques_bounds.iter().copied()
    }

    #[inline]
    pub(crate) fn with_log<'a>(
        &'a mut self,
        undo_log: &'a mut InferCtxtUndoLogs<'tcx>,
    ) -> OpaqueTypeTable<'a, 'tcx> {
        OpaqueTypeTable { storage: self, undo_log }
    }
}

pub struct OpaqueTypeTable<'a, 'tcx> {
    storage: &'a mut OpaqueTypeStorage<'tcx>,

    undo_log: &'a mut InferCtxtUndoLogs<'tcx>,
}
impl<'tcx> Deref for OpaqueTypeTable<'_, 'tcx> {
    type Target = OpaqueTypeStorage<'tcx>;
    fn deref(&self) -> &Self::Target {
        self.storage
    }
}

impl<'a, 'tcx> OpaqueTypeTable<'a, 'tcx> {
    #[instrument(skip(self), level = "debug")]
    pub fn register(
        &mut self,
        key: OpaqueTypeKey<'tcx>,
        hidden_type: ProvisionalHiddenType<'tcx>,
    ) -> Option<Ty<'tcx>> {
        if let Some(entry) = self.storage.opaque_types.get_mut(&key) {
            let prev = std::mem::replace(entry, hidden_type);
            self.undo_log.push(UndoLog::OpaqueTypes(key, Some(prev)));
            return Some(prev.ty);
        }
        self.storage.opaque_types.insert(key, hidden_type);
        self.undo_log.push(UndoLog::OpaqueTypes(key, None));
        None
    }

    pub fn add_duplicate(
        &mut self,
        key: OpaqueTypeKey<'tcx>,
        hidden_type: ProvisionalHiddenType<'tcx>,
    ) {
        self.storage.duplicate_entries.push((key, hidden_type));
        self.undo_log.push(UndoLog::DuplicateOpaqueType);
    }

    pub fn add_pseudo_rigid_due_to_opaques(
        &mut self,
        pseudo_rigid: Ty<'tcx>,
        bounds: impl IntoIterator<Item = ty::PseudoRigidDueToOpaquesBound<'tcx>>,
    ) {
        let OpaqueTypeStorage {
            opaque_types: _,
            duplicate_entries: _,
            pseudo_rigids_due_to_opaques,
            pseudo_rigid_due_to_opaques_bounds,
        } = self.storage;
        let prev_len = match pseudo_rigids_due_to_opaques.entry(pseudo_rigid) {
            Entry::Occupied(mut entry) => {
                let entry = entry.get_mut();
                let len = entry.len();
                entry.extend(bounds);
                if entry.len() == len {
                    return;
                }
                pseudo_rigid_due_to_opaques_bounds
                    .extend(iter::repeat(pseudo_rigid).zip(entry.iter().skip(len).copied()));
                Some(len)
            }
            Entry::Vacant(vacant) => {
                let bounds: FxIndexSet<_> = bounds.into_iter().collect();
                if bounds.is_empty() {
                    return;
                }
                let entry = vacant.insert(bounds);
                pseudo_rigid_due_to_opaques_bounds
                    .extend(iter::repeat(pseudo_rigid).zip(entry.iter().copied()));
                None
            }
        };
        self.undo_log.push(UndoLog::PseudoRigidDueToOpaques(pseudo_rigid, prev_len));
    }
}
