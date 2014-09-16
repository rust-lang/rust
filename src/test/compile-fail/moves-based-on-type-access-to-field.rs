// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that if you move from `x.f` or `x[0]`, `x` is inaccessible.
// Also tests that we give a more specific error message.

extern crate debug;

struct Foo { f: String, y: int }
fn consume(_s: String) {}
fn touch<A>(_a: &A) {}

fn f20() {
    let x = vec!("hi".to_string());
    consume(x.into_iter().next().unwrap());
    touch(x.get(0)); //~ ERROR use of moved value: `x`
}

fn main() {}
