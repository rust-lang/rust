// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

union DummyUnion {
    u8: u8,
    bool: bool,
}

#[repr(C)]
#[derive(Copy, Clone)]
enum Enum {
    A,
    B,
    C,
}

#[derive(Copy, Clone)]
union Foo {
    a: bool,
    b: Enum,
}

union Bar {
    foo: Foo,
    u8: u8,
}

// the value is not valid for bools
const BAD_BOOL: bool = unsafe { DummyUnion { u8: 42 }.bool};
//~^ ERROR this constant likely exhibits undefined behavior

// The value is not valid for any union variant, but that's fine
// unions are just a convenient way to transmute bits around
const BAD_UNION: Foo = unsafe { Bar { u8: 42 }.foo };


fn main() {
}
