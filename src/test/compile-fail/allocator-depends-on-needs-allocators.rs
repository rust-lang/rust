// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern: `allocator3` cannot depend on a crate that needs an allocator
// aux-build:needs_allocator.rs
// aux-build:allocator3.rs

// The needs_allocator crate is a dependency of the allocator crate allocator3,
// which is not allowed

extern crate allocator3;

fn main() {
}
