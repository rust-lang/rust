// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

// error-pattern: cannot declare a new module at this location
// error-pattern: maybe move this module `src/test/compile-fail/auxiliary/foo/bar.rs` to its own directory via `src/test/compile-fail/auxiliary/foo/bar/mod.rs`

mod auxiliary {
    mod foo;
}

fn main() {}
