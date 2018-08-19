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

type Field1 = i32;
type Field2 = f32;
type Field3 = i64;

union DummyUnion {
    field1: Field1,
    field2: Field2,
    field3: Field3,
}

const FLOAT1_AS_I32: i32 = 1065353216;
const UNION: DummyUnion = DummyUnion { field1: FLOAT1_AS_I32 };

const fn read_field1() -> Field1 {
    const FIELD1: Field1 = unsafe { UNION.field1 };
    FIELD1
}

const fn read_field2() -> Field2 {
    const FIELD2: Field2 = unsafe { UNION.field2 };
    FIELD2
}

const fn read_field3() -> Field3 {
    const FIELD3: Field3 = unsafe { UNION.field3 }; //~ ERROR cannot be used
    FIELD3
}

fn main() {
    assert_eq!(read_field1(), FLOAT1_AS_I32);
    assert_eq!(read_field2(), 1.0);
    assert_eq!(read_field3(), unsafe { UNION.field3 });
}
