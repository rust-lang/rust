// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


fn func(
    (
        1, //~ ERROR refutable pattern in function argument
        (
            Some( //~ ERROR refutable pattern in function argument
                1), // nested, so no warning.
            2..3 //~ ERROR refutable pattern in function argument
            )
        ): (int, (Option<int>, int))
        ) {}

fn main() {
    let (
        1, //~ ERROR refutable pattern in local binding
        (
            Some( //~ ERROR refutable pattern in local binding
                1), // nested, so no warning.
            2..3 //~ ERROR refutable pattern in local binding
            )
        ) = (1, (None, 2));
}
