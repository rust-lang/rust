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
// case that involve multiple intracrate types.
// Try enums too.

#[rustc_variance]
enum Base<'a, 'b, 'c, 'd> { //~ ERROR regions=[[+, -, o, *];[];[]]
    Test8A(extern "Rust" fn(&'a int)),
    Test8B(&'b [int]),
    Test8C(&'b mut &'c str),
}

#[rustc_variance]
struct Derived1<'w, 'x, 'y, 'z> { //~ ERROR regions=[[*, o, -, +];[];[]]
    f: Base<'z, 'y, 'x, 'w>
}

#[rustc_variance] // Combine - and + to yield o
struct Derived2<'a, 'b, 'c> { //~ ERROR regions=[[o, o, *];[];[]]
    f: Base<'a, 'a, 'b, 'c>
}

#[rustc_variance] // Combine + and o to yield o (just pay attention to 'a here)
struct Derived3<'a, 'b, 'c> { //~ ERROR regions=[[o, -, *];[];[]]
    f: Base<'a, 'b, 'a, 'c>
}

#[rustc_variance] // Combine + and * to yield + (just pay attention to 'a here)
struct Derived4<'a, 'b, 'c> { //~ ERROR regions=[[+, -, o];[];[]]
    f: Base<'a, 'b, 'c, 'a>
}

fn main() {}
