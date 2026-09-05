//! Collecting fake doc items.
//!
use rustc_attr_ir::{DocAttribute, find_attr};
use rustc_middle::query::{LocalCrate, Providers};
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::{DefId, LOCAL_CRATE};
use rustc_span::sym;

/// Traverse and collect the fake doc items in the current crate
fn fake_doc_items(tcx: TyCtxt<'_>, _: LocalCrate) -> Vec<DefId> {
    let mut fake_doc_items = Vec::new();

    // Optimization: can this crate even define fake doc items?
    let features = tcx.features().enabled_features();
    if features.contains(&sym::rustc_attrs) || features.contains(&sym::rustdoc_internals) {
        // Collect fake doc items in this crate.
        for id in tcx.hir_root_module().item_ids {
            let id = id.hir_id();
            if find_attr!(
                tcx,
                id,
                RustcDocPrimitive(..)
                    | Doc(DocAttribute { keyword: Some(..), .. })
                    | Doc(DocAttribute { attribute: Some(..), .. })
            ) {
                fake_doc_items.push(id.expect_owner().to_def_id());
            }
        }
    }

    fake_doc_items
}

/// Traverse and collect all the fake doc items in all crates.
fn all_fake_doc_items(tcx: TyCtxt<'_>, (): ()) -> Vec<DefId> {
    let mut fake_doc_items = Vec::new();

    // Collect fake doc items in visible crates.
    for cnum in tcx
        .crates(())
        .iter()
        .copied()
        .filter(|cnum| tcx.is_user_visible_dep(*cnum))
        .chain(std::iter::once(LOCAL_CRATE))
    {
        fake_doc_items.extend_from_slice(tcx.fake_doc_items(cnum))
    }

    fake_doc_items
}

pub(crate) fn provide(providers: &mut Providers) {
    providers.fake_doc_items = fake_doc_items;
    providers.all_fake_doc_items = all_fake_doc_items;
}
