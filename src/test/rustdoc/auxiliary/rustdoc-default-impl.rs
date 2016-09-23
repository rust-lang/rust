// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(optin_builtin_traits)]
#![feature(core)]

pub mod bar {
    use std::marker;

    pub trait Bar {}

    impl Bar for .. {}

    pub trait Foo {
        fn foo(&self) {}
    }

    impl Foo {
        pub fn test<T: Bar>(&self) {}
    }

    pub struct TypeId;

    impl TypeId {
        pub fn of<T: Bar + ?Sized>() -> TypeId {
            panic!()
        }
    }
}
