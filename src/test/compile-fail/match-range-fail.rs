// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    match "wow" {
        "bar" ... "foo" => { }
    };
    //~^^ ERROR only char and numeric types are allowed in range
    //~| start type: &'static str
    //~| end type: &'static str

    match "wow" {
        10 ... "what" => ()
    };
    //~^^ ERROR only char and numeric types are allowed in range
    //~| start type: {integer}
    //~| end type: &'static str

    match 5 {
        'c' ... 100 => { }
        _ => { }
    };
    //~^^^ ERROR mismatched types
    //~| expected type `{integer}`
    //~| found type `char`
}
