use std::ops::Deref;

use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::undo_log::UndoLogs;
use rustc_middle::ty::{self as ty, OpaqueTypeKey, ProvisionalHiddenType, Ty};
use rustc_span::bug;
use tracing::instrument;

use crate::infer::snapshot::undo_log::{InferCtxtUndoLogs, UndoLog};

#[derive(Default, Debug, Clone)]
pub struct OpaqueTypeStorage<'tcx> {
    opaque_types: FxIndexMap<OpaqueTypeKey<'tcx>, ProvisionalHiddenType<'tcx>>,
    duplicate_entries: Vec<(OpaqueTypeKey<'tcx>, ProvisionalHiddenType<'tcx>)>,
    /// We consider inference variables which are the hidden type of an opaque type or
    /// an unconstrained associated type of an opaque as pseudo-rigid. A pseudo-rigid
    /// inference variable is allowed as the self-type for method calls and we use the
    /// item bounds of the opaque to incompletely guide inference. We define and register
    /// unresolved infer vars as *pseudo-rigid* types and there bounds in the following,
    /// recursive manner:
    ///
    /// - The hidden types of opaques are pseudo-rigid. We register their item-self
    ///   bounds along with them, e.g., if we have `impl Iterator<Item = i32>`, the bounds are
    ///   `?pseudo-rigid: Iterator` and `<?pseudo-rigid as Iterator>::Item = i32`.
    /// - When we normalize an associated type whose self-ty is pseudo-rigid, and there does
    ///   not exist a `Projection` clause for that associated type, we register the normalized-to
    ///   term as a new pseudo-rigid. This fixes trait-system-refactor-initiative#248.
    pseudo_rigid_due_to_opaques: Vec<(Ty<'tcx>, ty::PseudoRigidDueToOpaquesBound<'tcx>)>,
}

/// The number of entries in the opaque type storage at a given point.
///
/// Used to check that we haven't added any new opaque types after checking
/// the opaque types currently in the storage.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpaqueTypeStorageEntries {
    opaque_types: usize,
    duplicate_entries: usize,
    pseudo_rigid_due_to_opaques: usize,
}

impl rustc_type_ir::inherent::OpaqueTypeStorageEntries for OpaqueTypeStorageEntries {
    fn needs_reevaluation(self, opaques: usize, pseudo_rigid: usize) -> bool {
        let OpaqueTypeStorageEntries {
            opaque_types,
            duplicate_entries: _,
            pseudo_rigid_due_to_opaques,
        } = self;
        opaques != opaque_types || pseudo_rigid != pseudo_rigid_due_to_opaques
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

    pub(crate) fn undo_pseudo_rigid_due_to_opaques(&mut self, len: usize) {
        debug_assert!(self.pseudo_rigid_due_to_opaques.len() > len);
        self.pseudo_rigid_due_to_opaques.truncate(len);
    }

    pub fn is_empty(&self) -> bool {
        let OpaqueTypeStorage { opaque_types, duplicate_entries, pseudo_rigid_due_to_opaques } =
            self;
        if opaque_types.is_empty() {
            debug_assert!(duplicate_entries.is_empty());
            debug_assert!(pseudo_rigid_due_to_opaques.is_empty());
            true
        } else {
            false
        }
    }

    pub(crate) fn take_opaque_types(
        &mut self,
    ) -> (
        impl Iterator<Item = (OpaqueTypeKey<'tcx>, ProvisionalHiddenType<'tcx>)>,
        Vec<(Ty<'tcx>, ty::PseudoRigidDueToOpaquesBound<'tcx>)>,
    ) {
        let OpaqueTypeStorage { opaque_types, duplicate_entries, pseudo_rigid_due_to_opaques } =
            self;
        (
            std::mem::take(opaque_types).into_iter().chain(std::mem::take(duplicate_entries)),
            std::mem::take(pseudo_rigid_due_to_opaques),
        )
    }

    pub fn num_entries(&self) -> OpaqueTypeStorageEntries {
        OpaqueTypeStorageEntries {
            opaque_types: self.opaque_types.len(),
            duplicate_entries: self.duplicate_entries.len(),
            pseudo_rigid_due_to_opaques: self.pseudo_rigid_due_to_opaques.len(),
        }
    }

    pub fn num_pseudo_rigid_due_to_opaques(&self) -> usize {
        self.pseudo_rigid_due_to_opaques.len()
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

    pub fn pseudo_rigid_due_to_opaques_added_since(
        &self,
        prev_entries: OpaqueTypeStorageEntries,
    ) -> impl Iterator<Item = (Ty<'tcx>, ty::PseudoRigidDueToOpaquesBound<'tcx>)> {
        self.pseudo_rigid_due_to_opaques
            .iter()
            .skip(prev_entries.pseudo_rigid_due_to_opaques)
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
        let OpaqueTypeStorage { opaque_types, duplicate_entries, pseudo_rigid_due_to_opaques: _ } =
            self;
        opaque_types.iter().map(|(k, v)| (*k, *v)).chain(duplicate_entries.iter().copied())
    }

    pub fn iter_pseudo_rigid_due_to_opaques(
        &self,
    ) -> impl Iterator<Item = (Ty<'tcx>, ty::PseudoRigidDueToOpaquesBound<'tcx>)> {
        let OpaqueTypeStorage {
            opaque_types: _,
            duplicate_entries: _,
            pseudo_rigid_due_to_opaques,
        } = self;
        pseudo_rigid_due_to_opaques.iter().copied()
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
            pseudo_rigid_due_to_opaques,
        } = self.storage;
        let prev_len = pseudo_rigid_due_to_opaques.len();
        for bound in bounds {
            if !pseudo_rigid_due_to_opaques.contains(&(pseudo_rigid, bound)) {
                pseudo_rigid_due_to_opaques.push((pseudo_rigid, bound));
            }
        }

        if prev_len != pseudo_rigid_due_to_opaques.len() {
            self.undo_log.push(UndoLog::PseudoRigidDueToOpaques(prev_len));
        }
    }
}
