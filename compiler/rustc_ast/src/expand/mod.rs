//! Definitions shared by macros / syntax extensions and e.g. `rustc_middle`.

use rustc_macros::{Decodable, Encodable, HashStable_Generic};
use rustc_span::Ident;
use rustc_span::def_id::DefId;

use crate::MetaItem;

pub mod allocator;
pub mod autodiff_attrs;
pub mod typetree;

#[derive(Debug, Clone, Encodable, Decodable, HashStable_Generic)]
pub struct StrippedCfgItem<ModId = DefId> {
    pub parent_module: ModId,
    pub ident: Ident,
    pub cfg: MetaItem,
}

impl<ModId> StrippedCfgItem<ModId> {
    pub fn map_mod_id<New>(self, f: impl FnOnce(ModId) -> New) -> StrippedCfgItem<New> {
        StrippedCfgItem { parent_module: f(self.parent_module), ident: self.ident, cfg: self.cfg }
    }
}
