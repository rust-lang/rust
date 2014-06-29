// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


fn func((1, (Some(1), 2..3)): (int, (Option<int>, int))) { }
//~^ ERROR refutable pattern in function argument: `(_, _)` not covered

fn main() {
    let (1i, (Some(1i), 2i..3i)) = (1i, (None, 2i));
    //~^ ERROR refutable pattern in local binding: `(_, _)` not covered
}
