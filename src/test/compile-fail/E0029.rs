// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let s = "hoho";

    match s {
        "hello" ... "world" => {}
        //~^ ERROR only char and numeric types are allowed in range patterns
        //~| NOTE ranges require char or numeric types
        //~| NOTE start type: &'static str
        //~| NOTE end type: &'static str
        _ => {}
    }
}
