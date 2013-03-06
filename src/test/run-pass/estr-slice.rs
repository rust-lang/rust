// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


pub fn main() {
    let x = &"hello";
    let v = &"hello";
    let mut y : &str = &"there";

    log(debug, x);
    log(debug, y);

    fail_unless!(x[0] == 'h' as u8);
    fail_unless!(x[4] == 'o' as u8);

    let z : &str = &"thing";
    fail_unless!(v == x);
    fail_unless!(x != z);

    let a = &"aaaa";
    let b = &"bbbb";

    let c = &"cccc";
    let cc = &"ccccc";

    log(debug, a);

    fail_unless!(a < b);
    fail_unless!(a <= b);
    fail_unless!(a != b);
    fail_unless!(b >= a);
    fail_unless!(b > a);

    log(debug, b);

    fail_unless!(a < c);
    fail_unless!(a <= c);
    fail_unless!(a != c);
    fail_unless!(c >= a);
    fail_unless!(c > a);

    log(debug, c);

    fail_unless!(c < cc);
    fail_unless!(c <= cc);
    fail_unless!(c != cc);
    fail_unless!(cc >= c);
    fail_unless!(cc > c);

    log(debug, cc);
}
