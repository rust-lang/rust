// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]
#![deny(const_err)]

fn black_box<T>(_: T) {
    unimplemented!()
}

fn main() {
    let b = 200u8 + 200u8 + 200u8;
    //~^ ERROR const_err
    let c = 200u8 * 4;
    //~^ ERROR const_err
    let d = 42u8 - (42u8 + 1);
    //~^ ERROR const_err
    let _e = [5u8][1];
    //~^ ERROR const_err
    black_box(b);
    black_box(c);
    black_box(d);
}
