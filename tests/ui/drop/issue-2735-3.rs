//@ run-pass

use std::cell::Cell;

// This test should behave exactly like issue-2735-2
struct Defer<'a> {
    b: &'a Cell<bool>,
}

impl<'a> Drop for Defer<'a> {
    fn drop(&mut self) {
        self.b.set(true);
    }
}

fn defer(b: &Cell<bool>) -> Defer<'_> {
    Defer { b }
}

pub fn main() {
    let dtor_ran = &Cell::new(false);
    defer(dtor_ran);
    assert!(dtor_ran.get());
}
