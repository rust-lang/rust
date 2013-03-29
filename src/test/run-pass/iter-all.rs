// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn is_even(x: &uint) -> bool { (*x % 2) == 0 }

pub fn main() {
    assert!(![1u, 2u].all(is_even));
    assert!([2u, 4u].all(is_even));
    assert!([].all(is_even));

    assert!(!iter::all(&Some(1u), is_even));
    assert!(iter::all(&Some(2u), is_even));
    assert!(iter::all(&None::<uint>, is_even));
}
