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
    self,
//~^ ERROR `self` import can only appear once in the list
//~^^ NOTE previous import of the module `bar` here
    Bar,
    self
//~^ NOTE another `self` import appears here
//~| ERROR the name `bar` is defined multiple times
//~| NOTE `bar` reimported here
//~| NOTE `bar` must be defined only once in the type namespace of this module
};

use {self};
//~^ ERROR `self` import can only appear in an import list with a non-empty prefix

mod foo {
    pub mod bar {
        pub struct Bar;
        pub struct Baz;
    }
}

fn main() {}
