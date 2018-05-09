// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(decl_macro)]

macro m($mod_name:ident) {
    pub mod $#mod_name {
    //~^ ERROR expected identifier, found `#`
    //~| ERROR unknown macro variable ``
    //~| ERROR expected identifier, found reserved identifier ``
    //~| ERROR expected one of `;` or `{`, found `mod_name`
    }
}

fn main() {
    m!(foo);
}
