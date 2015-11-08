// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test to ensure private traits are inaccessible with UFCS angle-bracket syntax.

mod foo {
    trait Bar {
        fn baz() {}
    }

    impl Bar for i32 {}
}

fn main() {
    <i32 as ::foo::Bar>::baz(); //~ERROR method `baz` is inaccessible
                                //~^NOTE: trait `Bar` is private
}
