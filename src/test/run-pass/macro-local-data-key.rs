// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::local_data;

local_data_key!(foo: int)

mod bar {
    local_data_key!(pub baz: float)
}

pub fn main() {
    local_data::get(foo, |x| assert!(x.is_none()));
    local_data::get(bar::baz, |y| assert!(y.is_none()));

    local_data::set(foo, 3);
    local_data::set(bar::baz, -10.0);

    local_data::get(foo, |x| assert_eq!(*x.unwrap(), 3));
    local_data::get(bar::baz, |y| assert_eq!(*y.unwrap(), -10.0));
}
