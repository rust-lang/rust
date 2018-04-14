// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that it is possible to create a global allocator in a submodule, rather than in the crate
// root.

#![feature(global_allocator, alloc, allocator_api)]

extern crate alloc;

mod mymod {
    use alloc::heap::Global;

    #[global_allocator]
    static MY_HEAP: Global = Global::default(); //~ ERROR
}

fn main() {}
