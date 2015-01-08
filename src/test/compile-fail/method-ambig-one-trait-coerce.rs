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
    fn foo(self) -> isize;
}

impl foo for Box<Object+'static> {
    fn foo(self) -> isize {1}
}

impl foo for Box<Object+Send> {
    fn foo(self) -> isize {2}
}

fn test1(x: Box<Object+Send+Sync>) {
    // FIXME(#18737) -- we ought to consider this to be ambiguous,
    // since we could coerce to either impl. However, what actually
    // happens is that we consider both impls applicable because of
    // incorrect subtyping relation. We then decide to call this a
    // call to the `foo` trait, leading to the following error
    // message.

    x.foo(); //~ ERROR `foo` is not implemented
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
