// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that you can't dereference raw pointers in constants.

fn main() {
    static C: u64 = unsafe {*(0xdeadbeef as *const u64)}; //~ ERROR E0396
                    //~| NOTE dereference of raw pointer in constant
    println!("{}", C);
}
