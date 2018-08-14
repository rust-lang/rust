// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// See also repr-transparent.rs

#[repr(transparent)] //~ ERROR unsupported representation for zero-variant enum
enum Void {}         //~| ERROR should be applied to struct

#[repr(transparent)] //~ ERROR should be applied to struct
enum FieldlessEnum {
    Foo,
    Bar,
}

#[repr(transparent)] //~ ERROR should be applied to struct
enum Enum {
    Foo(String),
    Bar(u32),
}

#[repr(transparent)] //~ ERROR should be applied to struct
union Foo {
    u: u32,
    s: i32
}

#[repr(transparent)] //~ ERROR should be applied to struct
fn cant_repr_this() {}

#[repr(transparent)] //~ ERROR should be applied to struct
static CANT_REPR_THIS: u32 = 0;
