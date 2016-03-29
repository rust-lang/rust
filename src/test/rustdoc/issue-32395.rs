// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:variant-struct.rs
// build-aux-docs
// ignore-cross-compile

// @has variant_struct/enum.Foo.html
// @!has - 'pub qux'
// @!has - 'pub Bar'
extern crate variant_struct;

// @has issue_32395/enum.Foo.html
// @!has - 'pub qux'
// @!has - 'pub Bar'
pub use variant_struct::Foo;
