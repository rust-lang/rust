// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we pick which version of `Foo` to run based on whether
// the type we (ultimately) inferred for `x` is copyable or not.
//
// In this case, the two versions are both impls of same trait, and
// hence we we can resolve method even without knowing yet which
// version will run (note that the `push` occurs after the call to
// `foo()`).

// pretty-expanded FIXME #23616

#![allow(unknown_features)]
#![feature(box_syntax)]

trait Foo {
    fn foo(&self) -> isize;
}

impl<T:Copy> Foo for Vec<T> {
    fn foo(&self) -> isize {1}
}

impl<T> Foo for Vec<Box<T>> {
    fn foo(&self) -> isize {2}
}

fn call_foo_copy() -> isize {
    let mut x = Vec::new();
    let y = x.foo();
    x.push(0_usize);
    y
}

fn call_foo_other() -> isize {
    let mut x: Vec<Box<_>> = Vec::new();
    let y = x.foo();
    x.push(box 0);
    y
}

fn main() {
    assert_eq!(call_foo_copy(), 1);
    assert_eq!(call_foo_other(), 2);
}
