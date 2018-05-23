// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// this is a rust 2015 crate

#[macro_export]
macro_rules! inject_me_at_the_root {
    ($name1:ident, $name2:ident) => {
        mod $name1 {
            pub(crate) const THE_CONSTANT: u32 = 22;
        }

        fn $name2() -> u32 {
            // Key point: this `use` statement -- in Rust 2018 --
            // would be an error. But because this crate is in Rust
            // 2015, it works, even when executed from a Rust 2018
            // environment.
            use $name1::THE_CONSTANT;
            THE_CONSTANT
        }
    }
}

#[macro_export]
macro_rules! print_me {
    ($p:path) => {
        {
            use $p as V;
            println!("{}", V);
        }
    }
}
