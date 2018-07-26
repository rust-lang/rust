// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Confirm that we don't accidently divide or mod by zero in llvm_type

// compile-pass

mod a {
    pub trait A {}
}

mod b {
    pub struct Builder {}

    pub fn new() -> Builder {
        Builder {}
    }

    impl Builder {
        pub fn with_a(&mut self, _a: fn() -> ::a::A) {}
    }
}

pub use self::b::new;

fn main() {}
