// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue 46036: [NLL] false edges on infinite loops
// Infinite loops should create false edges to the cleanup block.
#![feature(nll)]

struct Foo { x: &'static u32 }

fn foo() {
    let a = 3;
    let foo = Foo { x: &a }; //~ ERROR E0597
    loop { }
}

fn main() { }
