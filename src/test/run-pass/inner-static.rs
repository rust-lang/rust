// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:inner_static.rs
// xfail-fast

extern mod inner_static;

pub fn main() {
    let a = inner_static::A::<()>;
    let b = inner_static::B::<()>;
    let c = inner_static::test::A::<()>;
    assert_eq!(a.bar(), 2);
    assert_eq!(b.bar(), 4);
    assert_eq!(c.bar(), 6);
}
