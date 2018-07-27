// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

union Foo {
    a: &'static u8,
    b: usize,
}

// This might point to an invalid address, but that's the user's problem
const USIZE_AS_STATIC_REF: &'static u8 = unsafe { Foo { b: 1337 }.a};

fn main() {
}
