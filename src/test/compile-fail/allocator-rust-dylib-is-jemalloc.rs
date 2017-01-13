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
// aux-build:allocator-dylib2.rs
// aux-build:allocator1.rs
// error-pattern: cannot link together two allocators

// Ensure that rust dynamic libraries use jemalloc as their allocator, verifying
// by linking in the system allocator here and ensuring that we get a complaint.

// ignore-emscripten FIXME: What "other allocator" is correct for emscripten?

#![feature(alloc_system)]

extern crate allocator_dylib2;

// The main purpose of this test is to ensure that `alloc_system` **fails**
// here (specifically the system allocator), but currently system is
// disabled on quite a few platforms (bsds, emscripten, msvc, etc). To ensure
// that this just passes on those platforms we link in some other allocator to
// ensure we get the same error.
//
// So long as we CI linux/OSX we should be good.
#[cfg(any(all(target_os = "linux", any(target_arch = "x86", target_arch = "x86_64")),
          target_os = "macos"))]
extern crate alloc_system;
#[cfg(not(any(all(target_os = "linux", any(target_arch = "x86", target_arch = "x86_64")),
              target_os = "macos")))]
extern crate allocator1;

fn main() {
    allocator_dylib2::foo();
}
