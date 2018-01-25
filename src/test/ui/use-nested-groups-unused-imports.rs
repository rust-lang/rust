// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(use_nested_groups)]
#![allow(dead_code)]
#![deny(unused_imports)]

mod foo {
    pub mod bar {
        pub mod baz {
            pub struct Bar();
        }
        pub mod foobar {}
    }

    pub struct Foo();
}

use foo::{Foo, bar::{baz::{}, foobar::*}, *};
    //~^ ERROR unused imports: `*`, `Foo`, `baz::{}`, `foobar::*`
use foo::bar::baz::{*, *};
    //~^ ERROR unused import: `*`
use foo::{};
    //~^ ERROR unused import: `use foo::{};`

fn main() {
    let _: Bar;
}
