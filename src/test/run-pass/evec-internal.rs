// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test
// xfail-fast

// Doesn't work; needs a design decision.

pub fn main() {
    let x : [int * 5] = [1,2,3,4,5];
    let _y : [int * 5] = [1,2,3,4,5];
    let mut z = [1,2,3,4,5];
    z = x;
    fail_unless!(z[0] == 1);
    fail_unless!(z[4] == 5);

    let a : [int * 5] = [1,1,1,1,1];
    let b : [int * 5] = [2,2,2,2,2];
    let c : [int * 5] = [2,2,2,2,3];

    log(debug, a);

    fail_unless!(a < b);
    fail_unless!(a <= b);
    fail_unless!(a != b);
    fail_unless!(b >= a);
    fail_unless!(b > a);

    log(debug, b);

    fail_unless!(b < c);
    fail_unless!(b <= c);
    fail_unless!(b != c);
    fail_unless!(c >= b);
    fail_unless!(c > b);

    fail_unless!(a < c);
    fail_unless!(a <= c);
    fail_unless!(a != c);
    fail_unless!(c >= a);
    fail_unless!(c > a);

    log(debug, c);


}
