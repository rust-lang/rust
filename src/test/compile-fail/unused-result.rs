// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(unused_result, unused_must_use)]
#![allow(dead_code)]

#[must_use]
enum MustUse { Test }

fn foo<T>() -> T { fail!() }

fn bar() -> int { return foo::<int>(); }
fn baz() -> MustUse { return foo::<MustUse>(); }

#[allow(unused_result)]
fn test() {
    foo::<int>();
    foo::<MustUse>(); //~ ERROR: unused result which must be used
}

#[allow(unused_result, unused_must_use)]
fn test2() {
    foo::<int>();
    foo::<MustUse>();
}

fn main() {
    foo::<int>(); //~ ERROR: unused result
    foo::<MustUse>(); //~ ERROR: unused result which must be used

    let _ = foo::<int>();
    let _ = foo::<MustUse>();
}
