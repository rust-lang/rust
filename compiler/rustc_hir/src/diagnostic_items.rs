use rustc_data_structures::fx::FxIndexMap;
use rustc_macros::StableHash;
use rustc_span::Symbol;
use rustc_span::def_id::DefIdMap;

use crate::def_id::DefId;

#[derive(Debug, Default, StableHash)]
pub struct DiagnosticItems {
    #[stable_hash(ignore)]
    pub id_to_name: DefIdMap<Symbol>,
    pub name_to_id: FxIndexMap<Symbol, DefId>,
}
