use std::iter;
use std::ops::Deref;

use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::indexmap::map::Entry;
use rustc_data_structures::undo_log::UndoLogs;
use rustc_middle::bug;
use rustc_middle::ty::{self as ty, OpaqueTypeKey, ProvisionalHiddenType, Ty};
use tracing::instrument;

use crate::infer::snapshot::undo_log::{InferCtxtUndoLogs, UndoLog};

#[derive(Default, Debug, Clone)]
pub struct OpaqueTypeStorage<'tcx> {
    opaque_types: FxIndexMap<OpaqueTypeKey<'tcx>, ProvisionalHiddenType<'tcx>>,
    duplicate_entries: Vec<(OpaqueTypeKey<'tcx>, ProvisionalHiddenType<'tcx>)>,
    // FIXME: document those two fields
    hidden_types_of_opaques: FxIndexMap<Ty<'tcx>, FxIndexSet<ty::OpaqueHiddenTyBound<'tcx>>>,
    opaque_hidden_type_bounds: Vec<(Ty<'tcx>, ty::OpaqueHiddenTyBound<'tcx>)>,
}

/// The number of entries in the opaque type storage at a given point.
///
/// Used to check that we haven't added any new opaque types after checking
/// the opaque types currently in the storage.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpaqueTypeStorageEntries {
    opaque_types: usize,
    duplicate_entries: usize,
    opaque_hidden_type_bounds: usize,
}

impl rustc_type_ir::inherent::OpaqueTypeStorageEntries for OpaqueTypeStorageEntries {
    fn needs_reevaluation(self, opaques: usize, hidden_ty_bounds: usize) -> bool {
        let OpaqueTypeStorageEntries {
            opaque_types,
            duplicate_entries: _,
            opaque_hidden_type_bounds,
        } = self;
        opaques != opaque_types || hidden_ty_bounds != opaque_hidden_type_bounds
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

    pub(crate) fn truncate_hidden_types_of_opaques(
        &mut self,
        hidden_ty: Ty<'tcx>,
        len: Option<usize>,
    ) {
        let removed = if let Some(len) = len {
            let bounds = self.hidden_types_of_opaques.get_mut(&hidden_ty).unwrap();
            let removed = bounds.len() - len;
            bounds.truncate(len);
            removed
        } else {
            match self.hidden_types_of_opaques.swap_remove(&hidden_ty) {
                None => bug!(
                    "reverted opaque hidden type inference that was never registered: {:?}",
                    hidden_ty
                ),
                Some(bounds) => bounds.len(),
            }
        };

        let truncate_to = self.opaque_hidden_type_bounds.len() - removed;
        debug_assert!(
            (&self.opaque_hidden_type_bounds[truncate_to..]).iter().all(|(h, _)| *h == hidden_ty)
        );
        self.opaque_hidden_type_bounds.truncate(truncate_to);
    }

    pub fn is_empty(&self) -> bool {
        let OpaqueTypeStorage {
            opaque_types,
            duplicate_entries,
            hidden_types_of_opaques,
            opaque_hidden_type_bounds,
        } = self;
        opaque_types.is_empty()
            && duplicate_entries.is_empty()
            && hidden_types_of_opaques.is_empty()
            && opaque_hidden_type_bounds.is_empty()
    }

    pub(crate) fn take_opaque_types(
        &mut self,
    ) -> (
        impl Iterator<Item = (OpaqueTypeKey<'tcx>, ProvisionalHiddenType<'tcx>)>,
        impl Iterator<Item = (Ty<'tcx>, FxIndexSet<ty::OpaqueHiddenTyBound<'tcx>>)>,
    ) {
        let OpaqueTypeStorage {
            opaque_types,
            duplicate_entries,
            hidden_types_of_opaques,
            opaque_hidden_type_bounds,
        } = self;
        let _ = std::mem::take(opaque_hidden_type_bounds);
        (
            std::mem::take(opaque_types).into_iter().chain(std::mem::take(duplicate_entries)),
            std::mem::take(hidden_types_of_opaques).into_iter(),
        )
    }

    pub fn num_entries(&self) -> OpaqueTypeStorageEntries {
        OpaqueTypeStorageEntries {
            opaque_types: self.opaque_types.len(),
            duplicate_entries: self.duplicate_entries.len(),
            opaque_hidden_type_bounds: self.opaque_hidden_type_bounds.len(),
        }
    }

    pub fn num_opaque_hidden_type_bounds(&self) -> usize {
        self.hidden_types_of_opaques.iter().map(|(_, bounds)| bounds.len()).sum()
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

    pub fn opaque_hidden_ty_bounds_added_since(
        &self,
        prev_entries: OpaqueTypeStorageEntries,
    ) -> impl Iterator<Item = (Ty<'tcx>, ty::OpaqueHiddenTyBound<'tcx>)> {
        self.opaque_hidden_type_bounds.iter().skip(prev_entries.opaque_hidden_type_bounds).copied()
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
            hidden_types_of_opaques: _,
            opaque_hidden_type_bounds: _,
        } = self;
        opaque_types.iter().map(|(k, v)| (*k, *v)).chain(duplicate_entries.iter().copied())
    }

    pub fn iter_hidden_types_of_opaques(
        &self,
    ) -> impl Iterator<Item = (Ty<'tcx>, &FxIndexSet<ty::OpaqueHiddenTyBound<'tcx>>)> {
        let OpaqueTypeStorage {
            opaque_types: _,
            duplicate_entries: _,
            hidden_types_of_opaques,
            opaque_hidden_type_bounds: _,
        } = self;
        hidden_types_of_opaques.iter().map(|(hidden, bounds)| (*hidden, bounds))
    }

    pub fn iter_opaque_hidden_type_bounds(
        &self,
    ) -> impl Iterator<Item = (Ty<'tcx>, ty::OpaqueHiddenTyBound<'tcx>)> {
        let OpaqueTypeStorage {
            opaque_types: _,
            duplicate_entries: _,
            hidden_types_of_opaques: _,
            opaque_hidden_type_bounds,
        } = self;
        opaque_hidden_type_bounds.iter().copied()
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

    pub fn add_hidden_type_of_opaque(
        &mut self,
        hidden_ty: Ty<'tcx>,
        bounds: impl IntoIterator<Item = ty::OpaqueHiddenTyBound<'tcx>>,
    ) {
        let OpaqueTypeStorage {
            opaque_types: _,
            duplicate_entries: _,
            hidden_types_of_opaques,
            opaque_hidden_type_bounds,
        } = self.storage;
        let prev_len = match hidden_types_of_opaques.entry(hidden_ty) {
            Entry::Occupied(mut entry) => {
                let entry = entry.get_mut();
                let len = entry.len();
                entry.extend(bounds);
                if entry.len() == len {
                    return;
                }
                opaque_hidden_type_bounds
                    .extend(iter::repeat(hidden_ty).zip(entry.iter().skip(len).copied()));
                Some(len)
            }
            Entry::Vacant(vacant) => {
                let entry = vacant.insert(bounds.into_iter().collect());
                opaque_hidden_type_bounds
                    .extend(iter::repeat(hidden_ty).zip(entry.iter().copied()));
                None
            }
        };
        self.undo_log.push(UndoLog::HiddenTypesOfOpaques(hidden_ty, prev_len));
    }
}
