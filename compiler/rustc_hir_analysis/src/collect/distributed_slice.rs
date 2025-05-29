use rustc_hir::def_id::{DefIdMap, LocalDefId};
use rustc_hir::{DistributedSlice, ItemKind};
use rustc_middle::ty::TyCtxt;

pub(super) fn distributed_slice_elements<'tcx>(
    tcx: TyCtxt<'tcx>,
    _: (),
) -> DefIdMap<Vec<LocalDefId>> {
    let mut res = DefIdMap::<Vec<LocalDefId>>::default();

    for i in tcx.hir_free_items() {
        let addition_def_id = i.owner_id.def_id;
        if let ItemKind::Const(.., DistributedSlice::Addition(declaration_def_id)) =
            tcx.hir_expect_item(addition_def_id).kind
        {
            res.entry(declaration_def_id.to_def_id()).or_default().push(addition_def_id);
        }
    }

    res
}
