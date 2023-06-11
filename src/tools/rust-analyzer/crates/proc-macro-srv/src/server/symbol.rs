//! Symbol interner for proc-macro-srv

use std::{cell::RefCell, collections::HashMap, thread::LocalKey};
use tt::SmolStr;

thread_local! {
    pub(crate) static SYMBOL_INTERNER: RefCell<SymbolInterner> = Default::default();
}

// ID for an interned symbol.
#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub struct Symbol(u32);

pub(crate) type SymbolInternerRef = &'static LocalKey<RefCell<SymbolInterner>>;

impl Symbol {
    pub(super) fn intern(interner: SymbolInternerRef, data: &str) -> Symbol {
        interner.with(|i| i.borrow_mut().intern(data))
    }

    pub(super) fn text(&self, interner: SymbolInternerRef) -> SmolStr {
        interner.with(|i| i.borrow().get(self).clone())
    }
}

#[derive(Default)]
pub(crate) struct SymbolInterner {
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
