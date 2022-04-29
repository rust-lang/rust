use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_middle::ty::TyCtxt;
use rustc_session::cstore::ForeignModule;

crate fn collect(tcx: TyCtxt<'_>) -> Vec<ForeignModule> {
    let mut modules = Vec::new();
    for id in tcx.hir().items() {
        if !matches!(tcx.def_kind(id.def_id), DefKind::ForeignMod) {
            continue;
        }
        let item = tcx.hir().item(id);
        if let hir::ItemKind::ForeignMod { items, .. } = item.kind {
            let foreign_items = items.iter().map(|it| it.id.def_id.to_def_id()).collect();
            modules.push(ForeignModule { foreign_items, def_id: id.def_id.to_def_id() });
        }
    }
    modules
}
