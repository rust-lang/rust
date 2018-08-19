// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z borrowck=mir

use std::cell::Cell;

#[inline(never)]
fn tuple_field() -> &'static u32 {
    // This test is MIR-borrowck-only because the old borrowck
    // doesn't agree that borrows of "frozen" (i.e. without any
    // interior mutability) fields of non-frozen temporaries,
    // should be promoted, while MIR promotion does promote them.
    &(Cell::new(5), 42).1
}

fn main() {
    assert_eq!(tuple_field().to_string(), "42");
}
