// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem::size_of;

enum E {
    A = 1,
    B = 2,
    C = 3,
}

struct S {
    a: u16,
    b: u8,
    e: E,
}

fn main() {
    assert_eq!(size_of::<E>(), 1);
    assert_eq!(size_of::<Option<E>>(), 1);
    assert_eq!(size_of::<Result<E, ()>>(), 1);
    assert_eq!(size_of::<Option<S>>(), size_of::<S>());
    let enone = None::<E>;
    let esome = Some(E::A);
    if let Some(..) = enone {
        panic!();
    }
    if let None = esome {
        panic!();
    }
}
