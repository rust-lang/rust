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
// case that involve multiple intricate types.
// Try enums too.

#![feature(rustc_attrs)]

#[rustc_variance]
enum Base<'a, 'b, 'c:'b, 'd> { //~ ERROR [+, -, o, *]
    //~^ ERROR parameter `'d` is never used
    Test8A(extern "Rust" fn(&'a isize)),
    Test8B(&'b [isize]),
    Test8C(&'b mut &'c str),
}

#[rustc_variance]
struct Derived1<'w, 'x:'y, 'y, 'z> { //~ ERROR [*, o, -, +]
    //~^ ERROR parameter `'w` is never used
    f: Base<'z, 'y, 'x, 'w>
}

#[rustc_variance] // Combine - and + to yield o
struct Derived2<'a, 'b:'a, 'c> { //~ ERROR [o, o, *]
    //~^ ERROR parameter `'c` is never used
    f: Base<'a, 'a, 'b, 'c>
}

#[rustc_variance] // Combine + and o to yield o (just pay attention to 'a here)
struct Derived3<'a:'b, 'b, 'c> { //~ ERROR [o, -, *]
    //~^ ERROR parameter `'c` is never used
    f: Base<'a, 'b, 'a, 'c>
}

#[rustc_variance] // Combine + and * to yield + (just pay attention to 'a here)
struct Derived4<'a, 'b, 'c:'b> { //~ ERROR [+, -, o]
    f: Base<'a, 'b, 'c, 'a>
}

fn main() {}
