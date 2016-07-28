// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Verify that UnsafeCell is *always* !Sync regardless if `T` is sync.

#![feature(optin_builtin_traits)]

use std::cell::UnsafeCell;
use std::marker::Sync;

struct MySync<T> {
    u: UnsafeCell<T>
}

struct NoSync;
impl !Sync for NoSync {}

fn test<T: Sync>(s: T) {}

fn main() {
    let us = UnsafeCell::new(MySync{u: UnsafeCell::new(0)});
    test(us);
    //~^ ERROR `std::cell::UnsafeCell<MySync<{integer}>>: std::marker::Sync` is not satisfied

    let uns = UnsafeCell::new(NoSync);
    test(uns);
    //~^ ERROR `std::cell::UnsafeCell<NoSync>: std::marker::Sync` is not satisfied

    let ms = MySync{u: uns};
    test(ms);
    //~^ ERROR `std::cell::UnsafeCell<NoSync>: std::marker::Sync` is not satisfied

    test(NoSync);
    //~^ ERROR `NoSync: std::marker::Sync` is not satisfied
}
