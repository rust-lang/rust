// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]
#![allow(unused)]

pub struct Foo;

mod bar {
    struct Foo;

    mod baz {
        use *; //~ NOTE `Foo` could refer to the name imported here
        use bar::*; //~ NOTE `Foo` could also refer to the name imported here
        fn f(_: Foo) {}
        //~^ WARN `Foo` is ambiguous
        //~| WARN hard error in a future release
        //~| NOTE see issue #38260
    }
}

#[rustc_error]
fn main() {} //~ ERROR compilation successful
