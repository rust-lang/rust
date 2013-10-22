// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct A { a: int, b: @int }
struct B { a: int, b: C }
struct D { a: int, d: C }
struct C { c: int }

pub fn main() {
    match A {a: 10, b: @20} {
        x@A {a, b: @20} => { assert!(x.a == 10); assert!(a == 10); }
        A {b: _b, _} => { fail!(); }
    }
    let mut x@B {b, _} = B {a: 10, b: C {c: 20}};
    x.b.c = 30;
    assert_eq!(b.c, 20);
    let mut y@D {d, _} = D {a: 10, d: C {c: 20}};
    y.d.c = 30;
    assert_eq!(d.c, 20);
}
