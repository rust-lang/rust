// Copyright 2014–2017 The Rust Project Developers. See the COPYRIGHT
// http://rust-lang.org/COPYRIGHT.
//

#![deny(bad_style)]
#![allow(dead_code)]

fn CamelCase() {} //~ ERROR should have a snake

#[allow(bad_style)]
mod test {
    fn CamelCase() {}

    #[forbid(bad_style)]
    mod bad {
        fn CamelCase() {} //~ ERROR should have a snake

        static bad: isize = 1; //~ ERROR should have an upper
    }

    mod warn {
        #![warn(bad_style)]

        fn CamelCase() {} //~ WARN should have a snake

        struct snake_case; //~ WARN should have a camel
    }
}

fn main() {}
