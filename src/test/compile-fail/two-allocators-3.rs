// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:allocator1.rs
// error-pattern: cannot link together two allocators
// ignore-musl no dylibs on musl yet
// ignore-emscripten

// We're linking std dynamically (via -C prefer-dynamic for this test) which
// has an allocator and then we're also linking in a new allocator (allocator1)
// and this should be an error

extern crate allocator1;

fn main() {
}
