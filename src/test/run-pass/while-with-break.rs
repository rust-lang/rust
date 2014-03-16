// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


pub fn main() {
    let mut i: int = 90;
    while i < 100 {
        println!("{}", i);
        i = i + 1;
        if i == 95 {
            let _v: ~[int] =
                ~[1, 2, 3, 4, 5]; // we check that it is freed by break

            println!("breaking");
            break;
        }
    }
    assert_eq!(i, 95);
}
