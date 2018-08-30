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
    let pi = 3.1415927f64;
    println!("{}", -pi * (pi + 2.0 / pi) - pi * 5.0);
    if pi == 5.0 || pi < 10.0 || pi <= 2.0 || pi != 22.0 / 7.0 || pi >= 10.0
           || pi > 1.0 {
        println!("yes");
    }
}
