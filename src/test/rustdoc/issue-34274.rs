// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-34274.rs
// build-aux-docs
// ignore-cross-compile

#![crate_name = "foo"]

extern crate issue_34274;

// @has foo/fn.extern_c_fn.html '//a/@href' '../issue_34274/fn.extern_c_fn.html?gotosrc='
pub use issue_34274::extern_c_fn;
