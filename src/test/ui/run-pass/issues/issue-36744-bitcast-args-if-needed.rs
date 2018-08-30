// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This tests for an ICE (and, if ignored, subsequent LLVM abort) when
// a lifetime-parametric fn is passed into a context whose expected
// type has a differing lifetime parameterization.

struct A<'a> {
    _a: &'a i32,
}

fn call<T>(s: T, functions: &Vec<for <'n> fn(&'n T)>) {
    for function in functions {
        function(&s);
    }
}

fn f(a: &A) { println!("a holds {}", a._a); }

fn main() {
    let a = A { _a: &10 };

    let vec: Vec<for <'u,'v> fn(&'u A<'v>)> = vec![f];
    call(a, &vec);
}
