// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(start, no_core)]
#![no_core] // makes debugging this test *a lot* easier (during resolve)

// Test to make sure that globs don't leak in regular `use` statements.

mod bar {
    pub use self::glob::*;

    pub mod glob {
        use foo;
    }
}

pub fn foo() {}

fn test1() {
    use bar::foo;
    //~^ ERROR unresolved import `bar::foo` [E0432]
    //~| no `foo` in `bar`
}

fn test2() {
    use bar::glob::foo;
    //~^ ERROR `foo` is private
}

#[start] fn main(_: isize, _: *const *const u8) -> isize { 3 }
