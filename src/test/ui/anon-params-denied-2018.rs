// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that anonymous parameters are a hard error in edition 2018.

// edition:2018

trait T {
    fn foo(i32); //~ expected one of `:` or `@`, found `)`

    fn bar_with_default_impl(String, String) {}
    //~^ ERROR expected one of `:` or `@`, found `,`
}

fn main() {}
