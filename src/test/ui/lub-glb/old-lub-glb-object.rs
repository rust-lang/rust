// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we give a note when the old LUB/GLB algorithm would have
// succeeded but the new code (which is stricter) gives an error.

trait Foo<T, U> { }

fn foo(
    x: &for<'a, 'b> Foo<&'a u8, &'b u8>,
    y: &for<'a> Foo<&'a u8, &'a u8>,
) {
    let z = match 22 { //~ ERROR incompatible types
        0 => x,
        _ => y,
    };
}

fn bar(
    x: &for<'a, 'b> Foo<&'a u8, &'b u8>,
    y: &for<'a> Foo<&'a u8, &'a u8>,
) {
    // Accepted with explicit case:
    let z = match 22 {
        0 => x as &for<'a> Foo<&'a u8, &'a u8>,
        _ => y,
    };
}

fn main() {
}
