// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::any::Any;

fn foo<T: Any>(value: &T) -> Box<Any> {
    Box::new(value) as Box<Any>
    //~^ ERROR explicit lifetime required in the type of `value` [E0621]
}

fn main() {
    let _ = foo(&5);
}
