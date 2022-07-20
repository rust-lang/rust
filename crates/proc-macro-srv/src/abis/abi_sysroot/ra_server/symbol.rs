use std::collections::HashMap;
use tt::SmolStr;

// Identifier for an interned symbol.
#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub struct Symbol(u32);

#[derive(Default)]
struct IdentInterner {
    idents: HashMap<SmolStr, u32>,
    ident_data: Vec<SmolStr>,
}

impl IdentInterner {
    fn intern(&mut self, data: &str) -> Symbol {
        if let Some(index) = self.idents.get(data) {
            return *index;
        }

        let index = self.idents.len() as u32;
        self.ident_data.push(data.clone());
        self.idents.insert(data.clone(), index);
        index
    }

    fn get(&self, index: u32) -> &SmolStr {
        &self.ident_data[index as usize]
    }

    #[allow(unused)]
    fn get_mut(&mut self, index: u32) -> &mut SmolStr {
        self.ident_data.get_mut(index as usize).expect("Should be consistent")
    }
}
