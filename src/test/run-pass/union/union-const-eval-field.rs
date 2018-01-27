// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(const_fn)]

union DummyUnion {
    field1: i32,
    field2: i32,
}

const fn read_field() -> i32 {
    const UNION: DummyUnion = DummyUnion { field1: 5 };
    const FIELD: i32 = unsafe { UNION.field2 };
    FIELD
}

fn main() {
    assert_eq!(read_field(), 5);
}
