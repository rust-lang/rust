//! Things related to storage opaques in the infer context of the next-trait-solver.

use std::ops::Deref;

use ena::undo_log::UndoLogs;
use tracing::instrument;

use super::OpaqueHiddenType;
use crate::next_solver::{
    FxIndexMap, OpaqueTypeKey, Ty,
    infer::snapshot::undo_log::{InferCtxtUndoLogs, UndoLog},
};

#[derive(Default, Debug, Clone)]
pub(crate) struct OpaqueTypeStorage<'db> {
    opaque_types: FxIndexMap<OpaqueTypeKey<'db>, OpaqueHiddenType<'db>>,
    duplicate_entries: Vec<(OpaqueTypeKey<'db>, OpaqueHiddenType<'db>)>,
}

/// The number of entries in the opaque type storage at a given point.
///
/// Used to check that we haven't added any new opaque types after checking
/// the opaque types currently in the storage.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpaqueTypeStorageEntries {
    opaque_types: usize,
    duplicate_entries: usize,
}

impl rustc_type_ir::inherent::OpaqueTypeStorageEntries for OpaqueTypeStorageEntries {
    fn needs_reevaluation(self, canonicalized: usize) -> bool {
        self.opaque_types != canonicalized
    }
}

impl<'db> OpaqueTypeStorage<'db> {
    #[instrument(level = "debug")]
    pub(crate) fn remove(&mut self, key: OpaqueTypeKey<'db>, prev: Option<OpaqueHiddenType<'db>>) {
        if let Some(prev) = prev {
            *self.opaque_types.get_mut(&key).unwrap() = prev;
        } else {
            // FIXME(#120456) - is `swap_remove` correct?
            match self.opaque_types.swap_remove(&key) {
                None => {
                    panic!("reverted opaque type inference that was never registered: {key:?}")
                }
                Some(_) => {}
            }
        }
    }

    pub(crate) fn pop_duplicate_entry(&mut self) {
        let entry = self.duplicate_entries.pop();
        assert!(entry.is_some());
    }

    pub(crate) fn is_empty(&self) -> bool {
        let OpaqueTypeStorage { opaque_types, duplicate_entries } = self;
        opaque_types.is_empty() && duplicate_entries.is_empty()
    }

    pub(crate) fn take_opaque_types(
        &mut self,
    ) -> impl Iterator<Item = (OpaqueTypeKey<'db>, OpaqueHiddenType<'db>)> {
        let OpaqueTypeStorage { opaque_types, duplicate_entries } = self;
        std::mem::take(opaque_types).into_iter().chain(std::mem::take(duplicate_entries))
    }

    pub(crate) fn num_entries(&self) -> OpaqueTypeStorageEntries {
        OpaqueTypeStorageEntries {
            opaque_types: self.opaque_types.len(),
            duplicate_entries: self.duplicate_entries.len(),
        }
    }

    pub(crate) fn opaque_types_added_since(
        &self,
        prev_entries: OpaqueTypeStorageEntries,
    ) -> impl Iterator<Item = (OpaqueTypeKey<'db>, OpaqueHiddenType<'db>)> {
        self.opaque_types
            .iter()
            .skip(prev_entries.opaque_types)
            .map(|(k, v)| (*k, *v))
            .chain(self.duplicate_entries.iter().skip(prev_entries.duplicate_entries).copied())
    }

    /// Only returns the opaque types from the lookup table. These are used
    /// when normalizing opaque types and have a unique key.
    ///
    /// Outside of canonicalization one should generally use `iter_opaque_types`
    /// to also consider duplicate entries.
    pub(crate) fn iter_lookup_table(
        &self,
    ) -> impl Iterator<Item = (OpaqueTypeKey<'db>, OpaqueHiddenType<'db>)> {
        self.opaque_types.iter().map(|(k, v)| (*k, *v))
    }

    /// Only returns the opaque types which are stored in `duplicate_entries`.
    ///
    /// These have to considered when checking all opaque type uses but are e.g.
    /// irrelevant for canonical inputs as nested queries never meaningfully
    /// accesses them.
    pub(crate) fn iter_duplicate_entries(
        &self,
    ) -> impl Iterator<Item = (OpaqueTypeKey<'db>, OpaqueHiddenType<'db>)> {
        self.duplicate_entries.iter().copied()
    }

    pub(crate) fn iter_opaque_types(
        &self,
    ) -> impl Iterator<Item = (OpaqueTypeKey<'db>, OpaqueHiddenType<'db>)> {
        let OpaqueTypeStorage { opaque_types, duplicate_entries } = self;
        opaque_types.iter().map(|(k, v)| (*k, *v)).chain(duplicate_entries.iter().copied())
    }

    #[inline]
    pub(crate) fn with_log<'a>(
        &'a mut self,
        undo_log: &'a mut InferCtxtUndoLogs<'db>,
    ) -> OpaqueTypeTable<'a, 'db> {
        OpaqueTypeTable { storage: self, undo_log }
    }
}

pub(crate) struct OpaqueTypeTable<'a, 'db> {
    storage: &'a mut OpaqueTypeStorage<'db>,

    undo_log: &'a mut InferCtxtUndoLogs<'db>,
}
impl<'db> Deref for OpaqueTypeTable<'_, 'db> {
    type Target = OpaqueTypeStorage<'db>;
    fn deref(&self) -> &Self::Target {
        self.storage
    }
}

impl<'a, 'db> OpaqueTypeTable<'a, 'db> {
    #[instrument(skip(self), level = "debug")]
    pub(crate) fn register(
        &mut self,
        key: OpaqueTypeKey<'db>,
        hidden_type: OpaqueHiddenType<'db>,
    ) -> Option<Ty<'db>> {
        if let Some(entry) = self.storage.opaque_types.get_mut(&key) {
            let prev = std::mem::replace(entry, hidden_type);
            self.undo_log.push(UndoLog::OpaqueTypes(key, Some(prev)));
            return Some(prev.ty);
        }
        self.storage.opaque_types.insert(key, hidden_type);
        self.undo_log.push(UndoLog::OpaqueTypes(key, None));
        None
    }

    pub(crate) fn add_duplicate(
        &mut self,
        key: OpaqueTypeKey<'db>,
        hidden_type: OpaqueHiddenType<'db>,
    ) {
        self.storage.duplicate_entries.push((key, hidden_type));
        self.undo_log.push(UndoLog::DuplicateOpaqueType);
    }
}
