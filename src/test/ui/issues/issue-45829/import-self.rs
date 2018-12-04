// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod foo {
    pub struct A;
    pub struct B;
}

use foo::{self};
//~^ ERROR is defined multiple times

use foo as self;
//~^ ERROR expected identifier

use foo::self;
//~^ ERROR `self` imports are only allowed within a { } list

use foo::A;
use foo::{self as A};
//~^ ERROR is defined multiple times

fn main() {}
