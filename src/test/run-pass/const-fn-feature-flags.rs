// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test use of stabilized const fns in std formerly using individual feature gates.

use std::cell::Cell;

const CELL: Cell<i32> = Cell::new(42);

fn main() {
    let v = CELL.get();
    CELL.set(v+1);

    assert_eq!(CELL.get(), v);
}

