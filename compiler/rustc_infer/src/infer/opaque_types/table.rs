use std::ops::Deref;

use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::indexmap::map::Entry;
use rustc_data_structures::undo_log::UndoLogs;
use rustc_middle::bug;
use rustc_middle::ty::{self as ty, OpaqueTypeKey, ProvisionalHiddenType, Ty, TyVid};
use tracing::instrument;

use crate::infer::snapshot::undo_log::{InferCtxtUndoLogs, UndoLog};

#[derive(Default, Debug, Clone)]
pub struct OpaqueTypeStorage<'tcx> {
    opaque_types: FxIndexMap<OpaqueTypeKey<'tcx>, ProvisionalHiddenType<'tcx>>,
    duplicate_entries: Vec<(OpaqueTypeKey<'tcx>, ProvisionalHiddenType<'tcx>)>,
    hidden_types_of_opaques: FxIndexMap<Ty<'tcx>, FxIndexSet<ty::OpaqueHiddenTyBound<'tcx>>>,
}

/// The number of entries in the opaque type storage at a given point.
///
/// Used to check that we haven't added any new opaque types after checking
/// the opaque types currently in the storage.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct OpaqueTypeStorageEntries<'tcx> {
    opaque_types: usize,
    duplicate_entries: usize,
    hidden_types_of_opaques: FxIndexMap<Ty<'tcx>, usize>,
}

impl rustc_type_ir::inherent::OpaqueTypeStorageEntries for OpaqueTypeStorageEntries<'_> {
    fn needs_reevaluation(self, opaques: usize, hidden_bounds: usize) -> bool {
        let OpaqueTypeStorageEntries {
            opaque_types,
            duplicate_entries: _,
            ref hidden_types_of_opaques,
        } = self;

        opaque_types != opaques || hidden_types_of_opaques.values().sum::<usize>() != hidden_bounds
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
        if let Some(len) = len {
            let bounds = self.hidden_types_of_opaques.get_mut(&hidden_ty).unwrap();
            assert!(bounds.len() > len);
            bounds.truncate(len);
        } else {
            match self.hidden_types_of_opaques.swap_remove(&hidden_ty) {
                None => bug!(
                    "reverted opaque hidden type inference that was never registered: {:?}",
                    hidden_ty
                ),
                Some(_) => {}
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        let OpaqueTypeStorage { opaque_types, duplicate_entries, hidden_types_of_opaques } = self;
        opaque_types.is_empty()
            && duplicate_entries.is_empty()
            && hidden_types_of_opaques.is_empty()
    }

    pub(crate) fn take_opaque_types(
        &mut self,
    ) -> (
        impl Iterator<Item = (OpaqueTypeKey<'tcx>, ProvisionalHiddenType<'tcx>)>,
        impl Iterator<Item = (Ty<'tcx>, FxIndexSet<ty::OpaqueHiddenTyBound<'tcx>>)>,
    ) {
        let OpaqueTypeStorage { opaque_types, duplicate_entries, hidden_types_of_opaques } = self;
        (
            std::mem::take(opaque_types).into_iter().chain(std::mem::take(duplicate_entries)),
            std::mem::take(hidden_types_of_opaques).into_iter(),
        )
    }

    pub fn num_entries(&self) -> OpaqueTypeStorageEntries<'tcx> {
        OpaqueTypeStorageEntries {
            opaque_types: self.opaque_types.len(),
            duplicate_entries: self.duplicate_entries.len(),
            hidden_types_of_opaques: self
                .hidden_types_of_opaques
                .iter()
                .map(|(hidden_ty, bounds)| (*hidden_ty, bounds.len()))
                .collect(),
        }
    }

    pub fn opaque_types_added_since(
        &self,
        prev_entries: &OpaqueTypeStorageEntries<'tcx>,
    ) -> impl Iterator<Item = (OpaqueTypeKey<'tcx>, ProvisionalHiddenType<'tcx>)> {
        self.opaque_types
            .iter()
            .skip(prev_entries.opaque_types)
            .map(|(k, v)| (*k, *v))
            .chain(self.duplicate_entries.iter().skip(prev_entries.duplicate_entries).copied())
    }

    pub fn opaque_hidden_ty_bounds_added_since(
        &self,
        prev_entries: &OpaqueTypeStorageEntries<'tcx>,
    ) -> impl Iterator<Item = (Ty<'tcx>, Vec<ty::OpaqueHiddenTyBound<'tcx>>)> {
        self.hidden_types_of_opaques.iter().filter_map(|(hidden, bounds)| {
            if let Some(&len) = prev_entries.hidden_types_of_opaques.get(hidden) {
                assert!(bounds.len() >= len);
                if bounds.len() == len {
                    None
                } else {
                    Some((*hidden, bounds.iter().skip(len).copied().collect()))
                }
            } else {
                Some((*hidden, bounds.iter().copied().collect()))
            }
        })
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
        let OpaqueTypeStorage { opaque_types, duplicate_entries, hidden_types_of_opaques: _ } =
            self;
        opaque_types.iter().map(|(k, v)| (*k, *v)).chain(duplicate_entries.iter().copied())
    }

    pub fn iter_opaque_hidden_ty_bounds(
        &self,
    ) -> impl Iterator<Item = (Ty<'tcx>, &FxIndexSet<ty::OpaqueHiddenTyBound<'tcx>>)> {
        let OpaqueTypeStorage { opaque_types: _, duplicate_entries: _, hidden_types_of_opaques } =
            self;
        hidden_types_of_opaques.iter().map(|(hidden, bounds)| (*hidden, bounds))
    }

    pub(super) fn has_hidden_type_of_opaque_for_exact_vid(&self, vid: TyVid) -> bool {
        self.hidden_types_of_opaques
            .keys()
            .any(|ty| matches!(*ty.kind(), ty::Infer(ty::TyVar(ty_vid)) if ty_vid == vid))
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
        let prev_len = match self.storage.hidden_types_of_opaques.entry(hidden_ty) {
            Entry::Occupied(mut occupied) => {
                let occupied = occupied.get_mut();
                let len = occupied.len();
                occupied.extend(bounds);
                if occupied.len() == len {
                    return;
                }
                Some(len)
            }
            Entry::Vacant(vacant) => {
                vacant.insert(bounds.into_iter().collect());
                None
            }
        };
        self.undo_log.push(UndoLog::HiddenTypesOfOpaques(hidden_ty, prev_len));
    }
}
