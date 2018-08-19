// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test is a simple example of code that violates the dropck
// rules: it pushes `&x` and `&y` into a bag (with dtor), but the
// referenced data will be dropped before the bag is.







fn main() {
    let mut v = Bag::new();

    let x: i8 = 3;
    let y: i8 = 4;

    v.push(&x);
    //~^ ERROR `x` does not live long enough
    v.push(&y);
    //~^ ERROR `y` does not live long enough

    assert_eq!(v.0, [&3, &4]);
}

//`Vec<T>` is #[may_dangle] w.r.t. `T`; putting a bag over its head
// forces borrowck to treat dropping the bag as a potential use.
struct Bag<T>(Vec<T>);
impl<T> Drop for Bag<T> { fn drop(&mut self) { } }

impl<T> Bag<T> {
    fn new() -> Self { Bag(Vec::new()) }
    fn push(&mut self, t: T) { self.0.push(t); }
}
