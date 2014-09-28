// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(struct_variant)]
#![allow(unused_variable)]
#![allow(non_camel_case_types)]
#![deny(dead_code)]

extern crate libc;

use std::num;

struct Foo {
    x: uint,
    b: bool, //~ ERROR: struct field is never used
    marker: std::kinds::marker::NoCopy
}

fn field_read(f: Foo) -> uint {
    num::pow(f.x, 2)
}

enum XYZ {
    X, //~ ERROR variant is never used
    Y { //~ ERROR variant is never used
        a: String,
        b: int //~ ERROR: struct field is never used
    },
    Z
}

fn field_match_in_patterns(b: XYZ) -> String {
    match b {
        Y { a: a, .. } => a,
        _ => "".to_string()
    }
}

struct Bar {
    x: uint, //~ ERROR: struct field is never used
    b: bool,
    _guard: ()
}

#[repr(C)]
struct Baz {
    x: libc::c_uint
}

fn field_match_in_let(f: Bar) -> bool {
    let Bar { b, .. } = f;
    b
}

fn main() {
    field_read(Foo { x: 1, b: false, marker: std::kinds::marker::NoCopy });
    field_match_in_patterns(Z);
    field_match_in_let(Bar { x: 42u, b: true, _guard: () });
    let _ = Baz { x: 0 };
}
