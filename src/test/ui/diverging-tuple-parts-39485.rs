// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// After #39485, this test used to pass, but that change was reverted
// due to numerous inference failures like #39808, so it now fails
// again. #39485 made it so that diverging types never propagate
// upward; but we now do propagate such types upward in many more
// cases.

fn g() {
    &panic!() //~ ERROR mismatched types
}

fn f() -> isize {
    (return 1, return 2) //~ ERROR mismatched types
}

fn main() {}
