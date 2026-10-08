//@ edition:2015
//@ run-pass

use std::cell::Cell;

struct NoisyDrop<'a>(&'a Cell<bool>);

impl Drop for NoisyDrop<'_> {
    fn drop(&mut self) {
        self.0.set(true);
    }
}

fn main() {
    let dropped = Cell::new(false);
    let t = (NoisyDrop(&dropped), 0i32);

    let c = || {
        let _t = t.1;
    };

    c();

    assert!(!dropped.get());
}
