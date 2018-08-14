// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This should never be allowed -- `foo.a` and `foo.b` are
// overlapping, so since `x` is not `mut` we should not permit
// reassignment.

union Foo {
    a: u32,
    b: u32,
}

unsafe fn overlapping_fields() {
    let x: Foo;
    x.a = 1;  //~ ERROR
    x.b = 22; //~ ERROR
}

fn main() { }
