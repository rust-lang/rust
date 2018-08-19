// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn xyz() -> u8 { 42 }

const NUM: u8 = xyz();
//~^ ERROR calls in constants are limited to constant functions, tuple structs and tuple variants

fn main() {
    match 1 {
        NUM => unimplemented!(),
        //~^ ERROR could not evaluate constant pattern
        _ => unimplemented!(),
    }
}
