// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused_imports)]

mod foo {}

use foo::{
    ::bar,       //~ ERROR crate root in paths can only be used in start position
    super::bar,  //~ ERROR `super` in paths can only be used in start position
    self::bar,   //~ ERROR `self` in paths can only be used in start position
};

fn main() {}
