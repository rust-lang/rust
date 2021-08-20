use rustc_data_structures::undo_log::UndoLogs;
use rustc_hir::OpaqueTyOrigin;
use rustc_middle::ty::{self, OpaqueTypeKey, Ty};
use rustc_span::DUMMY_SP;

use crate::infer::{InferCtxtUndoLogs, UndoLog};

use super::{OpaqueHiddenType, OpaqueTypeDecl, OpaqueTypeMap};

#[derive(Default, Debug)]
pub struct OpaqueTypeStorage<'tcx> {
    // Opaque types found in explicit return types and their
    // associated fresh inference variable. Writeback resolves these
    // variables to get the concrete type, which can be used to
    // 'de-opaque' OpaqueTypeDecl, after typeck is done with all functions.
    pub opaque_types: OpaqueTypeMap<'tcx>,
}

impl<'tcx> OpaqueTypeStorage<'tcx> {
    #[instrument(level = "debug")]
    pub(crate) fn remove(&mut self, key: OpaqueTypeKey<'tcx>, idx: usize) {
        if idx == 0 {
            match self.opaque_types.remove(&key) {
                None => bug!("reverted opaque type inference that was never registered: {:?}", key),
                Some(_) => {}
            }
        } else {
            self.opaque_types.get_mut(&key).unwrap().hidden_types.drain(idx..);
        }
    }

    pub fn get_decl(&self, key: &OpaqueTypeKey<'tcx>) -> Option<&OpaqueTypeDecl<'tcx>> {
        self.opaque_types.get(key)
    }

    pub fn opaque_types(&self) -> OpaqueTypeMap<'tcx> {
        self.opaque_types.clone()
    }

    #[instrument(level = "debug")]
    pub fn take_opaque_types(&mut self) -> OpaqueTypeMap<'tcx> {
        std::mem::take(&mut self.opaque_types)
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
            ty::tls::with(|tcx| {
                tcx.sess.delay_span_bug(DUMMY_SP, &format!("{:?}", self.opaque_types))
            });
        }
    }
}

pub struct OpaqueTypeTable<'a, 'tcx> {
    storage: &'a mut OpaqueTypeStorage<'tcx>,

    undo_log: &'a mut InferCtxtUndoLogs<'tcx>,
}

impl<'a, 'tcx> OpaqueTypeTable<'a, 'tcx> {
    #[instrument(skip(self), level = "debug")]
    pub fn register(
        &mut self,
        key: OpaqueTypeKey<'tcx>,
        opaque_type: Ty<'tcx>,
        ty: OpaqueHiddenType<'tcx>,
        origin: OpaqueTyOrigin,
    ) {
        if let Some(decl) = self.storage.opaque_types.get_mut(&key) {
            decl.hidden_types.push(ty);
            self.undo_log.push(UndoLog::OpaqueTypes(key, decl.hidden_types.len()));
            return;
        }
        let decl = OpaqueTypeDecl { opaque_type, hidden_types: vec![ty], origin };
        self.storage.opaque_types.insert(key, decl);
        self.undo_log.push(UndoLog::OpaqueTypes(key, 0));
    }
}
