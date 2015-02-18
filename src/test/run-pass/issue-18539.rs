// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that coercing bare fn's that return a zero sized type to
// a closure doesn't cause an LLVM ERROR

struct Foo;

fn uint_to_foo(_: uint) -> Foo {
    Foo
}

#[allow(unused_must_use)]
fn main() {
    (0_usize..10).map(uint_to_foo);
}
