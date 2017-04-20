// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-41211.rs

// FIXME: https://github.com/rust-lang/rust/issues/41430
// This is a temporary regression test for the ICE reported in #41211

#![feature(proc_macro)]
#![emit_unchanged]
//~^ ERROR: cannot find attribute macro `emit_unchanged` in this scope
extern crate issue_41211;
use issue_41211::emit_unchanged;

fn main() {}
