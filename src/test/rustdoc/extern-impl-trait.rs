// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:extern-impl-trait.rs

#![crate_name = "foo"]

extern crate extern_impl_trait;

// @has 'foo/struct.X.html' '//code' "impl Foo<Associated = ()> + 'a"
pub use extern_impl_trait::X;

// @has 'foo/struct.Y.html' '//code' "impl ?Sized + Foo<Associated = ()> + 'a"
pub use extern_impl_trait::Y;
