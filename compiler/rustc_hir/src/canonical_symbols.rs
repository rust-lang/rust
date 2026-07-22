use rustc_data_structures::fx::FxIndexMap;
use rustc_macros::{Encodable, StableHash};
use rustc_span::Symbol;
use rustc_span::def_id::DefId;

/// A representation of a canonical symbol
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Encodable, StableHash)]
pub struct CanonicalSymbol {
    pub def_id: DefId,
    pub symbol: Symbol,
}

#[derive(StableHash, Debug)]
pub struct CanonicalSymbols {
    symbols: FxIndexMap<Symbol, CanonicalSymbol>,
}

impl CanonicalSymbols {
    /// Construct an empty collection of canonical symbols
    pub fn new() -> Self {
        Self { symbols: FxIndexMap::default() }
    }

    pub fn get(&self, symbol: Symbol) -> Option<CanonicalSymbol> {
        self.symbols.get(&symbol).copied()
    }

    pub fn set(&mut self, symbol: Symbol, def_id: DefId) -> Option<DefId> {
        let preexisting = self.symbols.insert(symbol, CanonicalSymbol { def_id, symbol });

        if let Some(preexisting) = preexisting {
            (preexisting.def_id != def_id).then_some(preexisting.def_id)
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = CanonicalSymbol> {
        self.symbols.values().copied()
    }
}
