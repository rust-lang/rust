use crate::ty::TyCtxt;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;

pub fn impl_is_default(tcx: TyCtxt<'_>, node_item_def_id: DefId) -> bool {
    match tcx.hir().as_local_hir_id(node_item_def_id) {
        Some(hir_id) => {
            let item = tcx.hir().expect_item(hir_id);
            if let hir::ItemKind::Impl { defaultness, .. } = item.kind {
                defaultness.is_default()
            } else {
                false
            }
        }
        None => tcx.impl_defaultness(node_item_def_id).is_default(),
    }
}
