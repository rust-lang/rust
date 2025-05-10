use rustc_attr_parsing::{AttributeKind, EIIDecl, EIIImpl, find_attr};
use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::def_id::DefId;
use rustc_middle::query::LocalCrate;
use rustc_middle::ty::TyCtxt;

type EIIMap = FxIndexMap<
    DefId, // the defid of the macro that declared the eii
    (
        EIIDecl, // the corresponding declaration
        FxIndexMap<DefId, EIIImpl>, // all the given implementations, indexed by defid.
                 // We expect there to be only one, but collect them all to give errors if there are more
                 // (or if there are none) in the final crate we build.
    ),
>;

pub(crate) fn collect<'tcx>(tcx: TyCtxt<'tcx>, LocalCrate: LocalCrate) -> EIIMap {
    let mut eiis = EIIMap::default();

    // now we've seen all EIIs declared and maybe even implemented in dependencies. Let's look at
    // the current crate!
    for id in tcx.hir_crate_items(()).definitions() {
        for i in
            find_attr!(tcx.get_all_attrs(id), AttributeKind::EiiImpl(e) => e).into_iter().flatten()
        {
            eiis.entry(i.eii_macro)
                .or_insert_with(|| {
                    // find the decl for this one if it wasn't in yet (maybe it's from the local crate? not very useful but not illegal)
                    (find_attr!(tcx.get_all_attrs(i.eii_macro), AttributeKind::EiiMacroFor(d) => *d).unwrap(), Default::default())
                }).1.insert(id.into(), *i);
        }

        // if we find a new declaration, add it to the list without a known implementation
        if let Some(decl) = find_attr!(tcx.get_all_attrs(id), AttributeKind::EiiMacroFor(d) => *d) {
            eiis.entry(id.into()).or_insert((decl, Default::default()));
        }
    }

    eiis
}
