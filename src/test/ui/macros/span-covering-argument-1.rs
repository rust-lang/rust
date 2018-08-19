// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! bad {
    ($s:ident whatever) => {
        {
            let $s = 0;
            *&mut $s = 0;
            //~^ ERROR cannot borrow immutable local variable `foo` as mutable [E0596]
        }
    }
}

fn main() {
    bad!(foo whatever);
}
