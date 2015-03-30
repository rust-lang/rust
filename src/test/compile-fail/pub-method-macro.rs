// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #18317

mod bleh {
    macro_rules! defn {
        ($n:ident) => (
            fn $n (&self) -> i32 {
                println!("{}", stringify!($n));
                1
            }
        )
    }

    #[derive(Copy, Clone)]
    pub struct S;

    impl S {
        pub defn!(f); //~ ERROR can't qualify macro invocation with `pub`
        //~^ HELP try adjusting the macro to put `pub` inside the invocation
    }
}

fn main() {
    bleh::S.f();
}
