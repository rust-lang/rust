// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:associated-const-cc-lib.rs

extern crate associated_const_cc_lib as foolib;

pub struct LocalFooUseDefault;

impl foolib::FooDefault for LocalFooUseDefault {}

pub struct LocalFooOverwriteDefault;

impl foolib::FooDefault for LocalFooOverwriteDefault {
    const BAR: usize = 4;
}

fn main() {
    assert_eq!(1, <foolib::FooUseDefault as foolib::FooDefault>::BAR);
    assert_eq!(2, <foolib::FooOverwriteDefault as foolib::FooDefault>::BAR);
    assert_eq!(1, <LocalFooUseDefault as foolib::FooDefault>::BAR);
    assert_eq!(4, <LocalFooOverwriteDefault as foolib::FooDefault>::BAR);
}
