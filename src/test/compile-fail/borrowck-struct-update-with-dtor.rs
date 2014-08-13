// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue 4691: Ensure that functional-struct-update can only copy, not
// move, when the struct implements Drop.

// NoCopy
use std::kinds::marker::NoCopy as NP;


struct S { a: int, np: NP }
impl Drop for S { fn drop(&mut self) { } }

struct T { a: int, mv: Box<int> }
impl Drop for T { fn drop(&mut self) { } }

fn f(s0:S) {
    let _s2 = S{a: 2, ..s0}; //~error: cannot move out of type `S`, which defines the `Drop` trait
}

fn g(s0:T) {
    let _s2 = T{a: 2, ..s0}; //~error: cannot move out of type `T`, which defines the `Drop` trait
}

fn main() { }
