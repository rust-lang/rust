// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn let_in<T, F>(x: T, f: F) where F: FnOnce(T) {}

fn main() {
    let_in(3u32, |i| { assert!(i == 3i32); });
    //~^ ERROR mismatched types
    //~| expected u32, found i32

    let_in(3i32, |i| { assert!(i == 3u32); });
    //~^ ERROR mismatched types
    //~| expected i32, found u32
}
