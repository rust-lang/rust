// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*
The test runner should check that, after `rustpkg build hello-world`:
  * testsuite/hello-world/build/ exists
  * testsuite/hello-world/build/ contains an executable named hello-world
  * testsuite/hello-world/build/ does not contain a library
*/

use core::io;

fn main() {
    io::println(~"Hello world!");
}
