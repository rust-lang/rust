// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-windows everything is the system allocator on windows
// ignore-musl no dylibs on musl right now
// ignore-bitrig no jemalloc on bitrig
// ignore-openbsd no jemalloc on openbsd
// aux-build:allocator-dylib2.rs
// error-pattern: cannot link together two allocators

// Ensure that rust dynamic libraries use jemalloc as their allocator, verifying
// by linking in the system allocator here and ensuring that we get a complaint.

#![feature(alloc_system)]

extern crate allocator_dylib2;
extern crate alloc_system;

fn main() {
    allocator_dylib2::foo();
}

