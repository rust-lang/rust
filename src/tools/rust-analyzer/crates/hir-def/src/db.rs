//! Defines database & queries for name resolution.
use base_db::{Crate, SourceDatabase};
use hir_expand::{EditionedFileId, MacroCallId};
use salsa::{Durability, Setter};
use triomphe::Arc;

use crate::nameres::crate_def_map;

#[query_group::query_group]
pub trait DefDatabase: SourceDatabase {
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
