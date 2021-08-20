use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::undo_log::UndoLogs;
use rustc_middle::ty::{self, OpaqueTypeKey, Ty};
use rustc_span::DUMMY_SP;

use crate::infer::InferCtxtUndoLogs;

use super::{OpaqueTypeDecl, OpaqueTypeMap};

#[derive(Default, Debug)]
pub struct OpaqueTypeStorage<'tcx> {
    // Opaque types found in explicit return types and their
    // associated fresh inference variable. Writeback resolves these
    // variables to get the concrete type, which can be used to
    // 'de-opaque' OpaqueTypeDecl, after typeck is done with all functions.
    pub opaque_types: OpaqueTypeMap<'tcx>,

    /// A map from inference variables created from opaque
    /// type instantiations (`ty::Infer`) to the actual opaque
    /// type (`ty::Opaque`). Used during fallback to map unconstrained
    /// opaque type inference variables to their corresponding
    /// opaque type.
    pub opaque_types_vars: FxHashMap<Ty<'tcx>, Ty<'tcx>>,
}

impl<'tcx> OpaqueTypeStorage<'tcx> {
    #[instrument(level = "debug")]
    pub(crate) fn remove(&mut self, key: OpaqueTypeKey<'tcx>) {
        match self.opaque_types.remove(&key) {
            None => bug!("reverted opaque type inference that was never registered"),
            Some(decl) => assert_ne!(self.opaque_types_vars.remove(decl.concrete_ty), None),
        }
    }

    pub fn get_decl(&self, key: &OpaqueTypeKey<'tcx>) -> Option<&OpaqueTypeDecl<'tcx>> {
        self.opaque_types.get(key)
    }

    pub fn get_opaque_type_for_infer_var(&self, key: Ty<'tcx>) -> Option<Ty<'tcx>> {
        self.opaque_types_vars.get(key).copied()
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
    pub fn register(&mut self, key: OpaqueTypeKey<'tcx>, decl: OpaqueTypeDecl<'tcx>) {
        self.undo_log.push(key);
        self.storage.opaque_types.insert(key, decl);
        self.storage.opaque_types_vars.insert(decl.concrete_ty, decl.opaque_type);
    }
}
