// ignore-fast

// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod a {
    pub mod b {
        pub mod a {
            pub fn foo() -> int { return 1; }
        }
    }
}

mod c {
    use a::b::a;
    pub fn bar() { assert!((a::foo() == 1)); }
}

pub fn main() { c::bar(); }
