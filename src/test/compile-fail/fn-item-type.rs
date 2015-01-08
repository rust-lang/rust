// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the types of distinct fn items are not compatible by
// default. See also `run-pass/fn-item-type-*.rs`.

fn foo(x: isize) -> isize { x * 2 }
fn bar(x: isize) -> isize { x * 4 }

fn eq<T>(x: T, y: T) { }

fn main() {
    let f = if true { foo } else { bar };
    //~^ ERROR expected fn item, found a different fn item

    eq(foo, bar);
    //~^ ERROR expected fn item, found a different fn item
}
