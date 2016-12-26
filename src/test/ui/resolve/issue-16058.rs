// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


pub struct GslResult {
    pub val: f64,
    pub err: f64
}

impl GslResult {
    pub fn new() -> GslResult {
        Result {
//~^ ERROR expected struct, variant or union type, found enum `Result`
//~| HELP possible better candidates are found in other modules, you can import them into scope
//~| HELP std::fmt::Result
//~| HELP std::io::Result
//~| HELP std::thread::Result
            val: 0f64,
            err: 0f64
        }
    }
}

fn main() {}
