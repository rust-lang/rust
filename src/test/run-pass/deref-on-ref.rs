// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that `&T` and `&mut T` implement `Deref<T>`

use std::ops::Deref;

fn deref<U:Copy,T:Deref<Target=U>>(t: T) -> U {
    *t
}

fn main() {
    let x: int = 3;
    let y = deref(&x);
    assert_eq!(y, 3);

    let mut x: int = 4;
    let y = deref(&mut x);
    assert_eq!(y, 4);
}
