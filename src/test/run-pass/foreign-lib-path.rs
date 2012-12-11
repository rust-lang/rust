// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test FIXME I don't know how to test this (#2604)
// compile-flags:-L.
// The -L flag is also used for linking foreign libraries

// FIXME: I want to name a mod that would not link successfully
// wouthout providing a -L argument to the compiler, and that
// will also be found successfully at runtime.
extern mod WHATGOESHERE {
    #[legacy_exports];
    fn IDONTKNOW() -> u32;
}

fn main() {
    assert IDONTKNOW() == 0x_BAD_DOOD_u32;
}