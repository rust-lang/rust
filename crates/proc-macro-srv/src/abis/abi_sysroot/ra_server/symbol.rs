//! Symbol interner for proc-macro-srv

use std::{cell::RefCell, collections::HashMap};
use tt::SmolStr;

thread_local! {
    static SYMBOL_INTERNER: RefCell<SymbolInterner> = Default::default();
}

// ID for an interned symbol.
#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub struct Symbol(u32);

#[derive(Default)]
struct SymbolInterner {
    idents: HashMap<SmolStr, u32>,
    ident_data: Vec<SmolStr>,
}

impl SymbolInterner {
    fn intern(&mut self, data: &str) -> Symbol {
        if let Some(index) = self.idents.get(data) {
            return Symbol(*index);
        }

        let index = self.idents.len() as u32;
        let data = SmolStr::from(data);
        self.ident_data.push(data.clone());
        self.idents.insert(data, index);
        Symbol(index)
    }

    fn get(&self, sym: &Symbol) -> &SmolStr {
        &self.ident_data[sym.0 as usize]
    }
}

pub(super) struct ThreadLocalSymbolInterner;

impl ThreadLocalSymbolInterner {
    pub(super) fn intern(data: &str) -> Symbol {
        SYMBOL_INTERNER.with(|i| i.borrow_mut().intern(data))
    }

    pub(super) fn with<T>(sym: &Symbol, f: impl FnOnce(&SmolStr) -> T) -> T {
        SYMBOL_INTERNER.with(|i| f(i.borrow().get(sym)))
    }

    pub(super) fn get_cloned(sym: &Symbol) -> SmolStr {
        Self::with(sym, |s| s.clone())
    }
}
