// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;

// @has issue_29503/trait.MyTrait.html
pub trait MyTrait {
    fn my_string(&self) -> String;
}

// @has - "//ul[@id='implementors-list']/li" "impl<T> MyTrait for T where T: Debug"
impl<T> MyTrait for T where T: fmt::Debug {
    fn my_string(&self) -> String {
        format!("{:?}", self)
    }
}

pub fn main() {
}
