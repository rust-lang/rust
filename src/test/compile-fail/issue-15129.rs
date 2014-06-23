// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub enum T {
    T1(()),
    T2(())
}

pub enum V {
    V1(int),
    V2(bool)
}

fn main() {
    match (T1(()), V2(true)) {
    //~^ ERROR non-exhaustive patterns: `(T1(()), V2(_))` not covered
        (T1(()), V1(i)) => (),
        (T2(()), V2(b)) => ()
    }
}
