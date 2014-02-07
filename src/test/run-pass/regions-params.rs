// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast

fn region_identity<'r>(x: &'r uint) -> &'r uint { x }

fn apply<T>(t: T, f: |T| -> T) -> T { f(t) }

fn parameterized(x: &uint) -> uint {
    let z = apply(x, ({|y|
        region_identity(y)
    }));
    *z
}

pub fn main() {
    let x = 3u;
    assert_eq!(parameterized(&x), 3u);
}
