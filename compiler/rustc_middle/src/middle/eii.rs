use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_macros::{HashStable, TyDecodable, TyEncodable};

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
#[derive(TyEncodable, TyDecodable, HashStable)]
pub struct EiiMapping {
    pub extern_item: DefId,
    pub chosen_impl: DefId,
    pub weak_linkage: bool,
}

pub type EiiMap = FxIndexMap<LocalDefId, EiiMapping>;
