// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we properly check for private types in public signatures, even
// inside a private module (#22261).

mod a {
    struct Priv;

    pub fn expose_a() -> Priv { //~Error: private type in public interface
        panic!();
    }

    mod b {
        pub fn expose_b() -> super::Priv { //~Error: private type in public interface
            panic!();
        }
    }
}

pub fn main() {}
