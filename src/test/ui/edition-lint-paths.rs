// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(crate_in_paths)]
#![deny(absolute_paths_not_starting_with_crate)]
#![allow(unused)]

pub mod foo {
    use ::bar::Bar;
    //~^ ERROR Absolute
    //~| WARN this was previously accepted
    use super::bar::Bar2;
    use crate::bar::Bar3;
}


use bar::Bar;
//~^ ERROR Absolute
//~| WARN this was previously accepted

pub mod bar {
    pub struct Bar;
    pub type Bar2 = Bar;
    pub type Bar3 = Bar;
}

fn main() {
    let x = ::bar::Bar;
    //~^ ERROR Absolute
    //~| WARN this was previously accepted
    let x = bar::Bar;
    let x = ::crate::bar::Bar;
    let x = self::bar::Bar;
}
