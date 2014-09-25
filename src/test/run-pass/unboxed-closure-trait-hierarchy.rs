// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(overloaded_calls, unboxed_closures)]

fn foo<F>(f: F) -> int where F: FnOnce(int) -> int {
    f(1i)
}

fn bar<F>(mut f: F) -> int where F: FnMut(int) -> int {
    f(1i)
}

fn main() {
    assert_eq!(foo(|&mut: x: int| x + 3), 4i);
    assert_eq!(bar(|&: x: int| x + 7), 8i);
}

