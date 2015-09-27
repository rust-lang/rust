// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:rustdoc-const-fn.rs

#![feature(const_fn)]

extern crate rustdoc_const_fn as ext;

// @has reexport_const_fn/fn.foo.html //pre 'pub const fn foo'
pub use ext::foo;

// @has reexport_const_fn/struct.Bar.html //code 'const fn baz'
pub use ext::Bar;
