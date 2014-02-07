// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast windows doesn't like extern mod
// aux-build:issue-9906.rs

pub use other::FooBar;
pub use other::foo;

mod other {
    pub struct FooBar{value: int}
    impl FooBar{
        pub fn new(val: int) -> FooBar {
            FooBar{value: val}
        }
    }

    pub fn foo(){
        1+1;
    }
}
