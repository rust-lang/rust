use crate::queries;
use std::cell::Cell;

#[salsa::database(queries::GroupStruct)]
#[derive(Default)]
pub(crate) struct DatabaseImpl {
    storage: salsa::Storage<Self>,
    counter: Cell<usize>,
}

impl queries::Counter for DatabaseImpl {
    fn increment(&self) -> usize {
        let v = self.counter.get();
        self.counter.set(v + 1);
        v
    }
}

impl salsa::Database for DatabaseImpl {}
