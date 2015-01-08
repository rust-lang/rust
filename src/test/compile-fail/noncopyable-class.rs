// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that a class with a non-copyable field can't be
// copied

#[derive(Show)]
struct bar {
  x: isize,
}

impl Drop for bar {
    fn drop(&mut self) {}
}

fn bar(x:isize) -> bar {
    bar {
        x: x
    }
}

#[derive(Show)]
struct foo {
  i: isize,
  j: bar,
}

fn foo(i:isize) -> foo {
    foo {
        i: i,
        j: bar(5)
    }
}

fn main() {
    let x = foo(10);
    let _y = x.clone(); //~ ERROR does not implement any method in scope
    println!("{:?}", x);
}
