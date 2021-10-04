use crate::def_id::DefId;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_span::Symbol;

#[derive(Debug, Default)]
pub struct DiagnosticItems {
    pub id_to_name: FxHashMap<DefId, Symbol>,
    pub name_to_id: FxHashMap<Symbol, DefId>,
}

impl<CTX: crate::HashStableContext> HashStable<CTX> for DiagnosticItems {
    #[inline]
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        self.name_to_id.hash_stable(ctx, hasher);
    }
}
