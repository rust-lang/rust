use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::attrs::{EiiDecl, EiiImpl, EiiImplResolution};
use rustc_hir::def_id::DefId;
use rustc_hir::find_attr;
use rustc_middle::bug;
use rustc_middle::query::LocalCrate;
use rustc_middle::ty::TyCtxt;

// basically the map below but flattened out
pub(crate) type EiiMapEncodedKeyValue = (DefId, (EiiDecl, Vec<(DefId, EiiImpl)>));

pub(crate) type EiiMap = FxIndexMap<
    DefId, // the defid of the foreign item associated with the eii
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

    // Needed because Known only stores a DefId (not the full EiiDecl),
    // and we can't call the externally_implementable_items query (cycle).
    let decls_by_foreign_item: FxIndexMap<DefId, EiiDecl> = tcx
        .hir_crate_items(())
        .eiis()
        .filter_map(|id| find_attr!(tcx, id, EiiDeclaration(d) => *d))
        .map(|decl| (decl.foreign_item, decl))
        .collect();

    // iterate over all items in the current crate
    for id in tcx.hir_crate_items(()).eiis() {
        // if we find a new declaration, add it to the list without a known implementation
        if let Some(decl) = find_attr!(tcx, id, EiiDeclaration(d) => *d) {
            eiis.entry(decl.foreign_item).or_insert((decl, Default::default()));
        }

        if let Some(i) = find_attr!(tcx, id, EiiImpl(i) => i) {
            let (foreign_item, decl) = match i.resolution {
                EiiImplResolution::Macro(macro_defid) => {
                    // find the decl for this one if it wasn't in yet (maybe it's from the local crate? not very useful but not illegal)
                    let Some(decl) = find_attr!(tcx, macro_defid, EiiDeclaration(d) => *d) else {
                        // skip if it doesn't have eii_declaration (if we resolved to another macro that's not an EII)
                        tcx.dcx()
                            .span_delayed_bug(i.span, "resolved to something that's not an EII");
                        continue;
                    };
                    (decl.foreign_item, decl)
                }
                // Recover the EiiDecl from the local lookup map.
                EiiImplResolution::Known(foreign_item_did) => {
                    let decl = decls_by_foreign_item.get(&foreign_item_did).unwrap_or_else(|| {
                        bug!(
                            "EII impl has Known resolution but can't find EiiDeclaration for {:?}",
                            foreign_item_did
                        )
                    });
                    (foreign_item_did, *decl)
                }
                EiiImplResolution::Error(_eg) => continue,
            };

            eiis.entry(foreign_item)
                .or_insert_with(|| (decl, Default::default()))
                .1
                .insert(id.into(), *i);
        }
    }

    eiis
}
