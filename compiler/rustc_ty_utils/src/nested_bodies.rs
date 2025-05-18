use rustc_hir::def_id::LocalDefId;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, TyCtxt};

fn nested_bodies_within<'tcx>(tcx: TyCtxt<'tcx>, item: LocalDefId) -> &'tcx ty::List<LocalDefId> {
    let owner = tcx.local_def_id_to_hir_id(item).owner;
    let children = tcx.mk_local_def_ids_from_iter(
        tcx.hir_owner_nodes(owner)
            .bodies
            .iter()
            .map(|&(_, body)| tcx.hir_body_owner_def_id(body.id()))
            .filter(|&child_item| {
                // Anon consts are not owner IDs, but they may have (e.g.) closures in them.
                // Filter this just down to bodies that share the typeck root.
                child_item != item
                    && tcx.typeck_root_def_id(child_item.to_def_id()).expect_local() == item
            }),
    );
    children
}

pub(super) fn provide(providers: &mut Providers) {
    *providers = Providers { nested_bodies_within, ..*providers };
}
