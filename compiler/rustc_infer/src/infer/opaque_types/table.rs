use rustc_data_structures::undo_log::UndoLogs;
use rustc_middle::bug;
use rustc_middle::ty::{self, OpaqueHiddenType, OpaqueTypeKey, Ty};
use tracing::instrument;

use super::{OpaqueTypeDecl, OpaqueTypeMap};
use crate::infer::snapshot::undo_log::{InferCtxtUndoLogs, UndoLog};

#[derive(Default, Debug, Clone)]
pub(crate) struct OpaqueTypeStorage<'tcx> {
    /// Opaque types found in explicit return types and their
    /// associated fresh inference variable. Writeback resolves these
    /// variables to get the concrete type, which can be used to
    /// 'de-opaque' OpaqueTypeDecl, after typeck is done with all functions.
    pub opaque_types: OpaqueTypeMap<'tcx>,
}

impl<'tcx> OpaqueTypeStorage<'tcx> {
    #[instrument(level = "debug")]
    pub(crate) fn remove(&mut self, key: OpaqueTypeKey<'tcx>, idx: Option<OpaqueHiddenType<'tcx>>) {
        if let Some(idx) = idx {
            self.opaque_types.get_mut(&key).unwrap().hidden_type = idx;
        } else {
            // FIXME(#120456) - is `swap_remove` correct?
            match self.opaque_types.swap_remove(&key) {
                None => bug!("reverted opaque type inference that was never registered: {:?}", key),
                Some(_) => {}
            }
        }
    }

    #[inline]
    pub(crate) fn with_log<'a>(
        &'a mut self,
        undo_log: &'a mut InferCtxtUndoLogs<'tcx>,
    ) -> OpaqueTypeTable<'a, 'tcx> {
        OpaqueTypeTable { storage: self, undo_log }
    }
}

impl<'tcx> Drop for OpaqueTypeStorage<'tcx> {
    fn drop(&mut self) {
        if !self.opaque_types.is_empty() {
            ty::tls::with(|tcx| tcx.dcx().delayed_bug(format!("{:?}", self.opaque_types)));
        }
    }
}

pub(crate) struct OpaqueTypeTable<'a, 'tcx> {
    storage: &'a mut OpaqueTypeStorage<'tcx>,

    undo_log: &'a mut InferCtxtUndoLogs<'tcx>,
}

impl<'a, 'tcx> OpaqueTypeTable<'a, 'tcx> {
    #[instrument(skip(self), level = "debug")]
    pub(crate) fn register(
        &mut self,
        key: OpaqueTypeKey<'tcx>,
        hidden_type: OpaqueHiddenType<'tcx>,
    ) -> Option<Ty<'tcx>> {
        if let Some(decl) = self.storage.opaque_types.get_mut(&key) {
            let prev = std::mem::replace(&mut decl.hidden_type, hidden_type);
            self.undo_log.push(UndoLog::OpaqueTypes(key, Some(prev)));
            return Some(prev.ty);
        }
        let decl = OpaqueTypeDecl { hidden_type };
        self.storage.opaque_types.insert(key, decl);
        self.undo_log.push(UndoLog::OpaqueTypes(key, None));
        None
    }
}
