//! Defines database & queries for name resolution.
use base_db::SourceDatabase;
use salsa::{Durability, Setter};

#[query_group::query_group]
pub trait DefDatabase: SourceDatabase {}

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
