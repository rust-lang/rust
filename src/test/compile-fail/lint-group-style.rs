// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(bad_style)]
//~^ NOTE lint level defined here
#![allow(dead_code)]

fn CamelCase() {}
//~^ ERROR function `CamelCase` should have a snake case name
//~| NOTE #[deny(bad_style)] implies #[deny(non_snake_case)]

#[allow(bad_style)]
mod test {
    fn CamelCase() {}

    #[forbid(bad_style)]
    //~^ NOTE lint level defined here
    //~^^ NOTE lint level defined here
    mod bad {
        fn CamelCase() {}
        //~^ ERROR function `CamelCase` should have a snake case name
        //~| NOTE #[forbid(bad_style)] implies #[forbid(non_snake_case)]

        static bad: isize = 1;
        //~^ ERROR static variable `bad` should have an upper case name
        //~| NOTE #[forbid(bad_style)] implies #[forbid(non_upper_case_globals)]
    }

    mod warn {
        #![warn(bad_style)]
        //~^ NOTE lint level defined here
        //~| NOTE lint level defined here

        fn CamelCase() {}
        //~^ WARN function `CamelCase` should have a snake case name
        //~| NOTE #[warn(bad_style)] implies #[warn(non_snake_case)]

        struct snake_case;
        //~^ WARN type `snake_case` should have a camel case name
        //~| NOTE #[warn(bad_style)] implies #[warn(non_camel_case_types)]
    }
}

fn main() {}
