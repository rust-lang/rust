// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:rustdoc-hidden-sig.rs
// build-aux-docs
// ignore-cross-compile

// @has rustdoc_hidden_sig/struct.Bar.html
// @!has -  '//a/@title' 'Hidden'
// @has -  '//a' 'u8'
extern crate rustdoc_hidden_sig;

// @has issue_28480/struct.Bar.html
// @!has -  '//a/@title' 'Hidden'
// @has -  '//a' 'u8'
pub use rustdoc_hidden_sig::Bar;
