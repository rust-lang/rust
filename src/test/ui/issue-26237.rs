// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! macro_panic {
    ($not_a_function:expr, $some_argument:ident) => {
        $not_a_function($some_argument)
        //~^ ERROR expected function, found `{integer}`
    }
}

fn main() {
    let mut value_a = 0;
    let mut value_b = 0;
    macro_panic!(value_a, value_b);
    //~^ in this expansion of macro_panic!
}
