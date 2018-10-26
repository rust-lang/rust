// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Zforce-overflow-checks=on

#![allow(exceeding_bitshifts)]
#![warn(const_err)]

fn black_box<T>(_: T) {
    unimplemented!()
}

const FOO: u8 = [5u8][1];
//~^ WARN any use of this value will cause an error

fn main() {
    black_box((FOO, FOO));
    //~^ ERROR erroneous constant used
}
