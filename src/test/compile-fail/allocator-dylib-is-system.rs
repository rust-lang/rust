// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-msvc everything is the system allocator on msvc
// aux-build:allocator-dylib.rs
// no-prefer-dynamic
// error-pattern: cannot link together two allocators

// Verify that the allocator for statically linked dynamic libraries is the
// system allocator. Do this by linking in jemalloc and making sure that we get
// an error.

#![feature(alloc_jemalloc)]

extern crate allocator_dylib;
extern crate alloc_jemalloc;

fn main() {
    allocator_dylib::foo();
}
