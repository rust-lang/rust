use rustc_data_structures::fx::FxIndexMap;
use rustc_macros::HashStable_Generic;
use rustc_span::Symbol;
use rustc_span::def_id::DefIdMap;

use crate::def_id::DefId;

#[derive(Debug, Default, HashStable_Generic)]
pub struct DiagnosticItems {
    #[stable_hasher(ignore)]
    pub id_to_name: DefIdMap<Symbol>,
    pub name_to_id: FxIndexMap<Symbol, DefId>,
}
