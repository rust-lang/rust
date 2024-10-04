use crate::queries;
use std::cell::Cell;

#[ra_salsa::database(queries::GroupStruct)]
#[derive(Default)]
pub(crate) struct DatabaseImpl {
    storage: ra_salsa::Storage<Self>,
    counter: Cell<usize>,
}

impl queries::Counter for DatabaseImpl {
    fn increment(&self) -> usize {
        let v = self.counter.get();
        self.counter.set(v + 1);
        v
    }
}

impl ra_salsa::Database for DatabaseImpl {}
