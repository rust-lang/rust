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

// Oddball: extern crate appears in a submodule, making it harder for
// us to rewrite paths. We don't (and we leave the `extern crate` in
// place).

#![feature(rust_2018_preview)]
#![deny(absolute_paths_not_starting_with_crate)]

mod m {
    // Because this extern crate does not appear at the root, we
    // ignore it altogether.
    pub extern crate edition_lint_paths;
}

// And we don't being smart about paths like this, even though you
// *could* rewrite it to `use edition_lint_paths::foo`
use m::edition_lint_paths::foo;
//~^ ERROR absolute paths must start
//~| WARNING this was previously accepted

fn main() {
    foo();
}

