// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the recursion limit can be changed and that the compiler
// suggests a fix. In this case, we have a long chain of Deref impls
// which will cause an overflow during the autoderef loop.

#![allow(dead_code)]
#![recursion_limit="10"]

macro_rules! link {
    ($outer:ident, $inner:ident) => {
        struct $outer($inner);

        impl $outer {
            fn new() -> $outer {
                $outer($inner::new())
            }
        }

        impl std::ops::Deref for $outer {
            type Target = $inner;

            fn deref(&self) -> &$inner {
                &self.0
            }
        }
    }
}

struct Bottom;
impl Bottom {
    fn new() -> Bottom {
        Bottom
    }
}

link!(Top, A);
link!(A, B);
link!(B, C);
link!(C, D);
link!(D, E);
link!(E, F);
link!(F, G);
link!(G, H);
link!(H, I);
link!(I, J);
link!(J, K);
link!(K, Bottom);

fn main() {
    let t = Top::new();
    let x: &Bottom = &t; //~ ERROR mismatched types
    //~^ error recursion limit
}

