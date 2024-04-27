//! Definitions shared by macros / syntax extensions and e.g. `rustc_middle`.

use rustc_span::{def_id::DefId, symbol::Ident};

use crate::MetaItem;

pub mod allocator;

#[derive(Debug, Clone, Encodable, Decodable, HashStable_Generic)]
pub struct StrippedCfgItem<ModId = DefId> {
    pub parent_module: ModId,
    pub name: Ident,
    pub cfg: MetaItem,
}

impl<ModId> StrippedCfgItem<ModId> {
    pub fn map_mod_id<New>(self, f: impl FnOnce(ModId) -> New) -> StrippedCfgItem<New> {
        StrippedCfgItem { parent_module: f(self.parent_module), name: self.name, cfg: self.cfg }
    }
}
