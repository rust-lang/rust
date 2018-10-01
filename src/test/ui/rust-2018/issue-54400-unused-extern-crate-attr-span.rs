// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:edition-lint-paths.rs
// run-rustfix
// compile-flags:--extern edition_lint_paths --cfg blandiloquence
// edition:2018

#![deny(rust_2018_idioms)]
#![allow(dead_code)]

// The suggestion span should include the attribute.

#[cfg(blandiloquence)] //~ HELP remove it
extern crate edition_lint_paths;
//~^ ERROR unused extern crate

fn main() {}
