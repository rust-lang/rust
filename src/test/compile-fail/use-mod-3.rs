// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use foo::bar::{
    mod //~ ERROR module `bar` is private
};
use foo::bar::{
    Bar, //~ ERROR type `Bar` is inaccessible
    //~^ NOTE module `bar` is private
    mod //~ ERROR module `bar` is private
};

mod foo {
    mod bar { pub type Bar = int; }
}

fn main() {}
