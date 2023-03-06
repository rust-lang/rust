// run-pass
#![allow(non_camel_case_types)]

use std::cell::Cell;

// This test should behave exactly like issue-2735-2
struct defer<'a> {
    b: &'a Cell<bool>,
}

impl<'a> Drop for defer<'a> {
    fn drop(&mut self) {
        self.b.set(true);
    }
}

fn defer(b: &Cell<bool>) -> defer {
    defer {
        b: b
    }
}

pub fn main() {
    let dtor_ran = &Cell::new(false);
    defer(dtor_ran);
    assert!(dtor_ran.get());
}
