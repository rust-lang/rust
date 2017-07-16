// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern fn f() {
}

fn is_fn<F>(_: F) where F: Fn() {}

fn main() {
    // extern functions are extern "C" fn
    let _x: extern "C" fn() = f; // OK
    is_fn(f);
    //~^ ERROR `extern "C" fn() {f}: std::ops::Fn<()>` is not satisfied
}
