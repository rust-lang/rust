// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![feature(existential_type)]

fn main() {}

mod boo {
    // declared in module but not defined inside of it
    pub existential type Boo: ::std::fmt::Debug; //~ ERROR could not find defining uses
}

fn bomp() -> boo::Boo {
    ""
}
