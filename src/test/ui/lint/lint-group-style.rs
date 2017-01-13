// Copyright 2014–2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(bad_style)]
#![allow(dead_code)]

fn CamelCase() {}

#[allow(bad_style)]
mod test {
    fn CamelCase() {}

    #[forbid(bad_style)]
    mod bad {
        fn CamelCase() {}

        static bad: isize = 1;
    }

    mod warn {
        #![warn(bad_style)]

        fn CamelCase() {}

        struct snake_case;
    }
}

fn main() {}
