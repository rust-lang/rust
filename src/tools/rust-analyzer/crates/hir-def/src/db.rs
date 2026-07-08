//! Defines database & queries for name resolution.
use base_db::{Crate, SourceDatabase};
use hir_expand::{EditionedFileId, HirFileId, MacroCallId};
use salsa::{Durability, Setter};
use triomphe::Arc;

use crate::{
    TraitId,
    item_tree::{ItemTree, file_item_tree},
    nameres::crate_def_map,
};

#[query_group::query_group]
pub trait DefDatabase: SourceDatabase {
    /// Computes an [`ItemTree`] for the given file or macro expansion.
    #[salsa::invoke(file_item_tree)]
    #[salsa::transparent]
    fn file_item_tree(&self, file_id: HirFileId, krate: Crate) -> &ItemTree;

    #[salsa::invoke(crate::lang_item::crate_notable_traits)]
    #[salsa::transparent]
    fn crate_notable_traits(&self, krate: Crate) -> Option<&[TraitId]>;

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
