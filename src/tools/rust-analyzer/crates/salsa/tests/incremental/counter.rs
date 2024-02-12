use std::cell::Cell;

#[derive(Default)]
pub(crate) struct Counter {
    value: Cell<usize>,
}

impl Counter {
    pub(crate) fn increment(&self) -> usize {
        let v = self.value.get();
        self.value.set(v + 1);
        v
    }
}
