// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensure that we can do a destructuring bind of a fixed-size array,
// even when the element type has a destructor.

// pretty-expanded FIXME #23616

struct D { x: u8 }

impl Drop for D { fn drop(&mut self) { } }

fn main() {
    fn d(x: u8) -> D { D { x: x } }

    let d1 = foo([d(1), d(2), d(3), d(4)], 1);
    let d3 = foo([d(5), d(6), d(7), d(8)], 3);
    assert_eq!(d1.x, 2);
    assert_eq!(d3.x, 8);
}

fn foo([a, b, c, d]: [D; 4], i: usize) -> D {
    match i {
        0 => a,
        1 => b,
        2 => c,
        3 => d,
        _ => panic!("unmatched"),
    }
}
