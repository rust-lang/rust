// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub mod test2 {
    // This used to generate an ICE (make sure that default functions are
    // parented to their trait to find the first private thing as the trait).

    struct B;
    trait A { fn foo(&self) {} }
    impl A for B {}

    mod tests {
        use super::A;
        fn foo() {
            let a = super::B;
            a.foo();
        }
    }
}


pub fn main() {}
