// Copyright 2014–2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(nonstandard_style)]
#![allow(dead_code)]

fn CamelCase() {} //~ ERROR should have a snake

#[allow(nonstandard_style)]
mod test {
    fn CamelCase() {}

    #[forbid(nonstandard_style)]
    mod bad {
        fn CamelCase() {} //~ ERROR should have a snake

        static bad: isize = 1; //~ ERROR should have an upper
    }

    mod warn {
        #![warn(nonstandard_style)]

        fn CamelCase() {} //~ WARN should have a snake

        struct snake_case; //~ WARN should have a camel
    }
}

fn main() {}
