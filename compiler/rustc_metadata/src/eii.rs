use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::attrs::{AttributeKind, EiiDecl, EiiImpl};
use rustc_hir::def_id::DefId;
use rustc_hir::find_attr;
use rustc_middle::query::LocalCrate;
use rustc_middle::ty::TyCtxt;

// basically the map below but flattened out
pub(crate) type EiiMapEncodedKeyValue = (DefId, (EiiDecl, Vec<(DefId, EiiImpl)>));

pub(crate) type EiiMap = FxIndexMap<
    DefId, // the defid of the macro that declared the eii
    (
        // the corresponding declaration
        EiiDecl,
        // all the given implementations, indexed by defid.
        // We expect there to be only one, but collect them all to give errors if there are more
        // (or if there are none) in the final crate we build.
        FxIndexMap<DefId, EiiImpl>,
    ),
>;

pub(crate) fn collect<'tcx>(tcx: TyCtxt<'tcx>, LocalCrate: LocalCrate) -> EiiMap {
    let mut eiis = EiiMap::default();

    // iterate over all items in the current crate
    for id in tcx.hir_crate_items(()).eiis() {
        for i in
            find_attr!(tcx.get_all_attrs(id), AttributeKind::EiiImpls(e) => e).into_iter().flatten()
        {
            // Find the decl for this one if it wasn't in yet (maybe it's from the local crate? not very useful but not illegal).
            // The eii_macro DefId might not have an EiiExternTarget attribute if a macro shadows the eii function,
            // so we need to handle the None case gracefully instead of panicking.
            let Some(decl) =
                find_attr!(tcx.get_all_attrs(i.eii_macro), AttributeKind::EiiExternTarget(d) => *d)
            else {
                continue;
            };
            eiis.entry(i.eii_macro)
                .or_insert_with(|| (decl, Default::default()))
                .1
                .insert(id.into(), *i);
        }

        // if we find a new declaration, add it to the list without a known implementation
        if let Some(decl) =
            find_attr!(tcx.get_all_attrs(id), AttributeKind::EiiExternTarget(d) => *d)
        {
            eiis.entry(id.into()).or_insert((decl, Default::default()));
        }
    }

    eiis
}
