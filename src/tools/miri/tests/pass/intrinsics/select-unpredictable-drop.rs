//! Check that `select_unpredictable` properly forgets the value it does not select.
#![feature(core_intrinsics)]
use std::cell::Cell;
use std::intrinsics::select_unpredictable;

fn main() {
    let (true_val, false_val) = (Cell::new(false), Cell::new(false));
    _ = select_unpredictable(true, TraceDrop(&true_val), TraceDrop(&false_val));
    assert!(true_val.get());
    assert!(!false_val.get());
}

struct TraceDrop<'a>(&'a Cell<bool>);

impl<'a> Drop for TraceDrop<'a> {
    fn drop(&mut self) {
        self.0.set(true);
    }
}
