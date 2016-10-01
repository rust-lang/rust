// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-musl no dylibs
// aux-build:allocator-dylib.rs
// aux-build:allocator1.rs
// no-prefer-dynamic
// error-pattern: cannot link together two allocators

// Verify that the allocator for statically linked dynamic libraries is the
// system allocator. Do this by linking in jemalloc and making sure that we get
// an error.

// ignore-emscripten FIXME: What "other allocator" should we use for emcc?

#![feature(alloc_jemalloc)]

extern crate allocator_dylib;

// The main purpose of this test is to ensure that `alloc_jemalloc` **fails**
// here (specifically the jemalloc allocator), but currently jemalloc is
// disabled on quite a few platforms (bsds, emscripten, msvc, etc). To ensure
// that this just passes on those platforms we link in some other allocator to
// ensure we get the same error.
//
// So long as we CI linux/OSX we should be good.
#[cfg(any(target_os = "linux", target_os = "macos"))]
extern crate alloc_jemalloc;
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
extern crate allocator1;

fn main() {
    allocator_dylib::foo();
}
