// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// gate-test-thread_local

// Test that `#[thread_local]` attribute is gated by `thread_local`
// feature gate.
//
// (Note that the `thread_local!` macro is explicitly *not* gated; it
// is given permission to expand into this unstable attribute even
// when the surrounding context does not have permission to use it.)

#[thread_local] //~ ERROR `#[thread_local]` is an experimental feature
static FOO: i32 = 3;

pub fn main() {
    FOO.with(|x| {
        println!("x: {}", x);
    });
}
