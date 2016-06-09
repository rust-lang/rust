// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(associated_consts)]

mod bar1 {
    pub use self::bar2::Foo;
    mod bar2 {
        pub struct Foo;

        impl Foo {
            const ID: i32 = 1;
        }
    }
}

fn main() {
    assert_eq!(1, bar1::Foo::ID);
    //~^ERROR associated constant `ID` is private
}
