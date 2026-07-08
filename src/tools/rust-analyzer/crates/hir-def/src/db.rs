//! Defines database & queries for name resolution.
use base_db::{Crate, SourceDatabase};
use hir_expand::{EditionedFileId, HirFileId, MacroCallId};
use salsa::{Durability, Setter};
use triomphe::Arc;

use crate::{
    AssocItemId, AttrDefId, TraitId,
    attrs::AttrFlags,
    item_tree::{ItemTree, file_item_tree},
    nameres::crate_def_map,
    visibility::{self, Visibility},
};

#[query_group::query_group]
pub trait DefDatabase: SourceDatabase {
    /// Computes an [`ItemTree`] for the given file or macro expansion.
    #[salsa::invoke(file_item_tree)]
    #[salsa::transparent]
    fn file_item_tree(&self, file_id: HirFileId, krate: Crate) -> &ItemTree;

    // region:visibilities

    #[salsa::invoke(visibility::assoc_visibility_query)]
    fn assoc_visibility(&self, def: AssocItemId) -> Visibility;

    // endregion:visibilities

    #[salsa::invoke(crate::lang_item::crate_notable_traits)]
    #[salsa::transparent]
    fn crate_notable_traits(&self, krate: Crate) -> Option<&[TraitId]>;

    #[salsa::invoke(crate_supports_no_std)]
    fn crate_supports_no_std(&self, crate_id: Crate) -> bool;

    #[salsa::invoke(include_macro_invoc)]
    fn include_macro_invoc(&self, crate_id: Crate) -> Arc<[(MacroCallId, EditionedFileId)]>;
}

/// Whether to expand procedural macros during name resolution.
///
/// Note: this struct shouldn't be exposed to ide crates -- consider using
/// [`set_expand_proc_attr_macros`] instead, if possible.
#[salsa::input(singleton, debug)]
pub(crate) struct ExpandProcAttrMacros {
    #[returns(copy)]
    pub(crate) enabled: bool,
}

pub fn set_expand_proc_attr_macros(db: &mut dyn DefDatabase, enabled: bool) {
    if let Some(expand_proc_attr_macros) = ExpandProcAttrMacros::try_get(db) {
        if expand_proc_attr_macros.enabled(db) != enabled {
            expand_proc_attr_macros.set_enabled(db).with_durability(Durability::HIGH).to(enabled);
        }
    } else {
        _ = ExpandProcAttrMacros::builder(enabled).durability(Durability::HIGH).new(db);
    }
}

// return: macro call id and include file id
fn include_macro_invoc(
    db: &dyn DefDatabase,
    krate: Crate,
) -> Arc<[(MacroCallId, EditionedFileId)]> {
    crate_def_map(db, krate)
        .modules
        .values()
        .flat_map(|m| m.scope.iter_macro_invoc())
        .filter_map(|(_, &invoc)| invoc.loc(db).include_file_id(db, invoc).map(|x| (invoc, x)))
        .collect()
}

fn crate_supports_no_std(db: &dyn DefDatabase, crate_id: Crate) -> bool {
    let root_module = crate_def_map(db, crate_id).root_module_id();
    let attrs = AttrFlags::query(db, AttrDefId::ModuleId(root_module));
    attrs.contains(AttrFlags::IS_NO_STD)
}
