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
// compile-flags:--edition 2018

// The "normal case". Ideally we would remove the `extern crate` here,
// but we don't.

#![feature(rust_2018_preview)]
#![deny(rust_2018_idioms)]
#![allow(dead_code)]

extern crate edition_lint_paths;
//~^ ERROR unused extern crate

extern crate edition_lint_paths as bar;
//~^ ERROR `extern crate` is not idiomatic in the new edition

fn main() {
    // This is not considered to *use* the `extern crate` in Rust 2018:
    use edition_lint_paths::foo;
    foo();

    // But this should be a use of the (renamed) crate:
    crate::bar::foo();
}

