// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:rustdoc-nonreachable-impls.rs
// build-aux-docs
// ignore-cross-compile

extern crate rustdoc_nonreachable_impls;

// @has issue_31948/Foo.t.html
// @has - '//*[@class="impl"]//code' 'Bark for'
// @has - '//*[@class="impl"]//code' 'Woof for'
// @!has - '//*[@class="impl"]//code' 'Bar for'
// @!has - '//*[@class="impl"]//code' 'Qux for'
pub use rustdoc_nonreachable_impls::Foo;

// @has issue_31948/Bark.t.html
// @has - '//code' 'for Foo'
// @!has - '//code' 'for Wibble'
// @!has - '//code' 'for Wobble'
pub use rustdoc_nonreachable_impls::Bark;

// @has issue_31948/Woof.t.html
// @has - '//code' 'for Foo'
// @!has - '//code' 'for Wibble'
// @!has - '//code' 'for Wobble'
pub use rustdoc_nonreachable_impls::Woof;

// @!has issue_31948/Bar.t.html
// @!has issue_31948/Qux.t.html
// @!has issue_31948/Wibble.t.html
// @!has issue_31948/Wobble.t.html
