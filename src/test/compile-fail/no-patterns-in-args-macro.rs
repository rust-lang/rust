// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! m {
    ($pat: pat) => {
        trait Tr {
            fn trait_method($pat: u8);
        }

        type A = fn($pat: u8);

        extern {
            fn foreign_fn($pat: u8);
        }
    }
}

mod good_pat {
    m!(good_pat); // OK
}

mod bad_pat {
    m!((bad, pat));
    //~^ ERROR patterns aren't allowed in function pointer types
    //~| ERROR patterns aren't allowed in foreign function declarations
    //~| ERROR patterns aren't allowed in methods without bodies
}

fn main() {}
