// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern: mismatched types

fn main() {
    let v = ~[~[0]];

    // This is ok because the outer vec is covariant with respect
    // to the inner vec. If the outer vec was mut then we
    // couldn't do this.
    fn f(&&v: ~[~[const int]]) {
    }

    f(v);
}
