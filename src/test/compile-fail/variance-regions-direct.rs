// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we correctly infer variance for region parameters in
// various self-contained types.

#![feature(rustc_attrs)]

// Regions that just appear in normal spots are contravariant:

#[rustc_variance]
struct Test2<'a, 'b, 'c> { //~ ERROR [-, -, -]
    x: &'a isize,
    y: &'b [isize],
    c: &'c str
}

// Those same annotations in function arguments become covariant:

#[rustc_variance]
struct Test3<'a, 'b, 'c> { //~ ERROR [+, +, +]
    x: extern "Rust" fn(&'a isize),
    y: extern "Rust" fn(&'b [isize]),
    c: extern "Rust" fn(&'c str),
}

// Mutability induces invariance:

#[rustc_variance]
struct Test4<'a, 'b:'a> { //~ ERROR [-, o]
    x: &'a mut &'b isize,
}

// Mutability induces invariance, even when in a
// contravariant context:

#[rustc_variance]
struct Test5<'a, 'b:'a> { //~ ERROR [+, o]
    x: extern "Rust" fn(&'a mut &'b isize),
}

// Invariance is a trap from which NO ONE CAN ESCAPE.
// In other words, even though the `&'b isize` occurs in
// an argument list (which is contravariant), that
// argument list occurs in an invariant context.

#[rustc_variance]
struct Test6<'a, 'b:'a> { //~ ERROR [-, o]
    x: &'a mut extern "Rust" fn(&'b isize),
}

// No uses at all is bivariant:

#[rustc_variance]
struct Test7<'a> { //~ ERROR [*]
    //~^ ERROR parameter `'a` is never used
    x: isize
}

// Try enums too.

#[rustc_variance]
enum Test8<'a, 'b, 'c:'b> { //~ ERROR [+, -, o]
    Test8A(extern "Rust" fn(&'a isize)),
    Test8B(&'b [isize]),
    Test8C(&'b mut &'c str),
}

fn main() {}
