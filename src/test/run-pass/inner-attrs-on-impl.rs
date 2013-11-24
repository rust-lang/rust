// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


struct Foo;

impl Foo {
    #[cfg(cfg_that_surely_doesnt_exist)];

    fn method(&self) -> bool { false }
}

impl Foo {
    #[cfg(not(cfg_that_surely_doesnt_exist))];

    // check that we don't eat attributes too eagerly.
    #[cfg(cfg_that_surely_doesnt_exist)]
    fn method(&self) -> bool { false }

    fn method(&self) -> bool { true }
}


pub fn main() {
    assert!(Foo.method());
}
