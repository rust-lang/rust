// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the crate hash is not affected by reordering items.

// revisions:rpass1 rpass2 rpass3
// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]

// Check that reordering otherwise identical items is not considered a
// change at all.
#[rustc_clean(label="Krate", cfg="rpass2")]

// But removing an item, naturally, is.
#[rustc_dirty(label="Krate", cfg="rpass3")]

#[cfg(rpass1)]
pub struct X {
    pub x: u32,
}

pub struct Y {
    pub x: u32,
}

#[cfg(rpass2)]
pub struct X {
    pub x: u32,
}

pub fn main() { }
