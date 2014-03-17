// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.




fn f(x: int) -> int {
    // println!("in f:");

    println!("{}", x);
    if x == 1 {
        // println!("bottoming out");

        return 1;
    } else {
        // println!("recurring");

        let y: int = x * f(x - 1);
        // println!("returned");

        println!("{}", y);
        return y;
    }
}

pub fn main() {
    assert_eq!(f(5), 120);
    // println!("all done");

}
