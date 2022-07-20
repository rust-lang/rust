use once_cell::sync::Lazy;
use std::{collections::HashMap, sync::Mutex};
use tt::SmolStr;

pub(super) static SYMBOL_INTERNER: Lazy<Mutex<SymbolInterner>> = Lazy::new(|| Default::default());

// ID for an interned symbol.
#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub struct Symbol(u32);

#[derive(Default)]
pub(super) struct SymbolInterner {
    idents: HashMap<SmolStr, u32>,
    ident_data: Vec<SmolStr>,
}

impl SymbolInterner {
    pub(super) fn intern(&mut self, data: &str) -> Symbol {
        if let Some(index) = self.idents.get(data) {
            return Symbol(*index);
        }

        let index = self.idents.len() as u32;
        let data = SmolStr::from(data);
        self.ident_data.push(data.clone());
        self.idents.insert(data, index);
        Symbol(index)
    }

    pub(super) fn get(&self, index: &Symbol) -> &SmolStr {
        &self.ident_data[index.0 as usize]
    }
}
