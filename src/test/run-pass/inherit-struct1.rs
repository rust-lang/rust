// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test struct inheritance.
#![feature(struct_inherit)]

virtual struct S1 {
    f1: int,
}

virtual struct S2 : S1 {
    f2: int,
}

struct S3 : S2 {
    f3: int,
}

// With lifetime parameters.
struct S5<'a> : S4<'a> {
    f4: int,
}

virtual struct S4<'a> {
    f3: &'a int,
}

// With type parameters.
struct S7<T> : S6<T> {
    f4: int,
}

virtual struct S6<T> {
    f3: T,
}

pub fn main() {
    let s = S2{f1: 115, f2: 113};
    assert!(s.f1 == 115);
    assert!(s.f2 == 113);

    let s = S3{f1: 15, f2: 13, f3: 17};
    assert!(s.f1 == 15);
    assert!(s.f2 == 13);
    assert!(s.f3 == 17);

    let s = S5{f3: &5, f4: 3};
    assert!(*s.f3 == 5);
    assert!(s.f4 == 3);

    let s = S7{f3: 5u, f4: 3};
    assert!(s.f3 == 5u);
    assert!(s.f4 == 3);
}
