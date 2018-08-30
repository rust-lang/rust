// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that using the `vec!` macro nested within itself works
// when the contents implement Drop

struct D(u32);

impl Drop for D {
    fn drop(&mut self) { println!("Dropping {}", self.0); }
}

fn main() {
    let nested = vec![vec![D(1u32), D(2u32), D(3u32)]];
    assert_eq!(nested[0][1].0, 2);
}
