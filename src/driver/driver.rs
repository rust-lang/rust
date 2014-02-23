// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[no_uv];

// These tools have no real use for libgreen/libuv (not much IO/task
// spawning).
extern crate native;

#[cfg(rustdoc)]
extern crate this = "rustdoc";

#[cfg(rustc)]
extern crate this = "rustc";

#[start]
fn start(argc: int, argv: **u8) -> int {
    native::start(argc, argv, this::main)
}
