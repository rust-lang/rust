// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure nested functions are separate, even if they have
// equal name.
//
// Issue #8587

pub struct X;

impl X {
    fn f(&self) -> int {
        #[inline(never)]
        fn inner() -> int {
            0
        }
        inner()
    }

    fn g(&self) -> int {
        #[inline(never)]
        fn inner_2() -> int {
            1
        }
        inner_2()
    }

    fn h(&self) -> int {
        #[inline(never)]
        fn inner() -> int {
            2
        }
        inner()
    }
}

fn main() {
    let n = X;
    assert_eq!(n.f(), 0);
    assert_eq!(n.g(), 1);
    // This test `h` used to fail.
    assert_eq!(n.h(), 2);
}
