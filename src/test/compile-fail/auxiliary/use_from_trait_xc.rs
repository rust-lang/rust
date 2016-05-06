// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(associated_consts)]

pub use self::sub::{Bar, Baz};

pub trait Trait {
    fn foo(&self);
    type Assoc;
    const CONST: u32;
}

struct Foo;

impl Foo {
    pub fn new() {}

    pub const C: u32 = 0;
}

mod sub {
    pub struct Bar;

    impl Bar {
        pub fn new() {}
    }

    pub enum Baz {}

    impl Baz {
        pub fn new() {}
    }
}
