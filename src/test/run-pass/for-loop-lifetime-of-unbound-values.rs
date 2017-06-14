// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::Cell;

struct Flag<'a>(&'a Cell<bool>);

impl<'a> Drop for Flag<'a> {
    fn drop(&mut self) {
        self.0.set(false)
    }
}

fn main() {
    let alive2 = Cell::new(true);
    for _i in std::iter::once(Flag(&alive2)) {
        // The Flag value should be alive in the for loop body
        assert_eq!(alive2.get(), true);
    }
    // The Flag value should be dead outside of the loop
    assert_eq!(alive2.get(), false);

    let alive = Cell::new(true);
    for _ in std::iter::once(Flag(&alive)) {
        // The Flag value should be alive in the for loop body even if it wasn't
        // bound by the for loop
        assert_eq!(alive.get(), true);
    }
    // The Flag value should be dead outside of the loop
    assert_eq!(alive.get(), false);
}