// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that when we pick a trait based on coercion, versus subtyping,
// we consider all possible coercions equivalent and don't try to pick
// a best one.

trait Object { }

trait foo {
    fn foo(self) -> int;
}

impl foo for Box<Object+'static> {
    fn foo(self) -> int {1}
}

impl foo for Box<Object+Send> {
    fn foo(self) -> int {2}
}

fn test1(x: Box<Object+Send+Sync>) {
    // Ambiguous because we could coerce to either impl:
    x.foo(); //~ ERROR E0034
}

fn test2(x: Box<Object+Send>) {
    // Not ambiguous because it is a precise match:
    x.foo();
}

fn test3(x: Box<Object+'static>) {
    // Not ambiguous because it is a precise match:
    x.foo();
}

fn main() { }
