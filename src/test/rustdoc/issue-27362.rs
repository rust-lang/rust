// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-27362.rs
// ignore-cross-compile

extern crate issue_27362;
pub use issue_27362 as quux;

// @matches issue_27362/quux/fn.foo.html '//pre' "pub const fn foo()"
// @matches issue_27362/quux/fn.bar.html '//pre' "pub const unsafe fn bar()"
// @matches issue_27362/quux/struct.Foo.html '//code' "const unsafe fn baz()"
